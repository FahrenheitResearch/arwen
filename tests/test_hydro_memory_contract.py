"""CPU metadata/admission contracts; no hydrologic numerical stand-ins."""
from dataclasses import replace
from contextlib import nullcontext
from datetime import datetime
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from gpuwm.hydro.memory import (
    ArrayMemory, HydroMemoryPlan, MemoryContext, StageMemory,
    admit_plan, forecast_plan, plan, verify_driver,
)
from gpuwm.hydro.runtime import settings_scope


@pytest.fixture
def factories(monkeypatch):
    from gpuwm import numerics
    from gpuwm.hydro import memory
    # These CPU metadata fixtures describe a fast-mode forecast. Preserve the
    # enclosing battery's process mode; no kernels are compiled in this scope.
    monkeypatch.setattr(numerics, "_mode", None)
    monkeypatch.setenv(numerics.ENV, "fast")
    monkeypatch.setattr(memory, "_device_scope", lambda device_id: nullcontext())
    module = ModuleType("gpuwm.hydro._memory_contract_test")
    module.prepare = lambda **kw: pytest.fail("model factory ran during metadata planning")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    settings = {"enabled": True, "numerics": "fast", "interval_seconds": 300,
                "stages": [{"name": "runoff", "factory": module.__name__ + ":prepare"}]}
    monkeypatch.delenv("GPUWM_NEXTGEN_ENABLED", raising=False)
    monkeypatch.delenv("GPUWM_NEXTGEN_SETTINGS", raising=False)
    return module, settings


def context():
    return MemoryContext("standalone", "fast", 300)


def test_missing_owner_inventory_is_named_and_never_admitted(factories, monkeypatch):
    from gpuwm.core import resident_admission
    monkeypatch.setattr(resident_admission, "admit", lambda *a, **kw: pytest.fail("incomplete gate"))
    _, settings = factories
    row = admit_plan(plan(settings, context=context()))
    assert not row["complete"] and not row["pre_factory_admission"]
    assert row["admission"]["fits"] is None
    assert "runoff:" in row["gaps"][0] and "prepare_memory" in row["gaps"][0]
    assert json.loads(json.dumps(row))["inventory_sha256"] == row["inventory_sha256"]


def test_hooks_are_ordered_metadata_only_and_cannot_mutate_dependencies(factories):
    module, settings = factories
    seen = []
    def inventory(**kw):
        seen.append(kw["context"])
        with pytest.raises(TypeError):
            kw["components"]["bad"] = None
        kw["options"]["changed"] = True
        return StageMemory((ArrayMemory("work", (129,), "float32"),), "prepared SHA", 1024)
    module.prepare_memory = inventory
    row = plan(settings, context=context()).inventory(members=3)
    assert row["complete"] and len(seen) == 1
    assert "options" not in settings["stages"][0]
    assert row["resident_bytes"] == 1024 * 3  # round each member's 516-byte backing
    assert row["non_pool_bytes"] == 1024 * 3
    assert row["required_bytes"] == 2048 * 3


def test_alias_backings_are_deduplicated_but_transient_lifetimes_are_summed():
    stages = (("runoff", StageMemory((
        ArrayMemory("slab", (256,), "float32"),
        ArrayMemory("field", (128,), "float32", alias_of="runoff/slab", offset_bytes=512),
        ArrayMemory("load", (1,), "uint8", lifetime="prepare"),
        ArrayMemory("step", (1,), "uint8", lifetime="step"),
        ArrayMemory("write", (1,), "uint8", lifetime="output"),
    ), "roster SHA; separate lifetimes", 0)),)
    row = HydroMemoryPlan(context(), stages, (), "settings SHA").inventory(members=2)
    assert len(row["backings"]) == 4 and len(row["arrays"]) == 5
    assert row["resident_bytes"] == 2048 and row["transient_bytes"] == 3072


@pytest.mark.parametrize('target, offset, lifetime', [
    pytest.param('s/slab', 4, 'resident', id='s/slab-4-resident'),
    pytest.param('s/slab', 0, 'output', id='s/slab-0-output'),
])
def test_alias_cannot_escape_backing_or_lifetime(target, offset, lifetime):
    stage = StageMemory((ArrayMemory("slab", (2,), "float32", lifetime="step"),
        ArrayMemory("view", (2,), "float32", lifetime=lifetime, alias_of=target,
                    offset_bytes=offset)), "prepared", 0)
    with pytest.raises(ValueError, match="alias"):
        HydroMemoryPlan(context(), (("s", stage),), (), "sha").inventory()


def test_checkpoint_packing_counts_alias_views_and_resource_gap(factories):
    module, settings = factories
    module.prepare_memory = lambda **kw: StageMemory((
        ArrayMemory("slab", (256,), "float32", checkpointed=True),
        ArrayMemory("view", (128,), "float32", alias_of="runoff/slab", checkpointed=True),
        ArrayMemory("status", (1,), "int32", checkpointed=True, status=True),
    ), "prepared SHA", 0)
    row = plan(settings, context=context()).inventory()
    assert row["resident_bytes"] == 1536
    # Pack + restore: 2*(1024+512+512); status scratch + scalar + stack: 3*512.
    assert row["transient_bytes"] == 5632
    assert row["gaps"] == ["_assembly: generated status/copy kernel device resource bound is unpriced"]


def test_unknown_kernel_resources_and_custom_paths_cannot_claim_complete(factories):
    module, settings = factories
    module.prepare_memory = lambda **kw: StageMemory((), "prepared", gaps=("missing roster",))
    settings.update(driver_factory="gpuwm.hydro.custom:prepare", output_factory="gpuwm.hydro.custom:write")
    gaps = plan(settings, context=context()).inventory()["gaps"]
    assert any("missing roster" in gap for gap in gaps)
    assert any("kernel/local-memory" in gap for gap in gaps)
    assert any("custom driver" in gap for gap in gaps)
    assert any("custom output" in gap for gap in gaps)


def test_assembly_refuses_complete_overbudget_contract_before_prebuild_or_factory(factories, monkeypatch):
    from gpuwm.hydro import assembly
    from gpuwm.core import resident_admission
    from gpuwm.ingest.memory_refusal import ResidentMemoryRefused
    module, settings = factories
    module.prepare_memory = lambda **kw: StageMemory((ArrayMemory("work", (1024,), "float64"),), "roster", 0)
    monkeypatch.setattr(resident_admission, "device_free_bytes", lambda: 8191)
    monkeypatch.delenv("GPUWM_NO_MEMORY_GATE", raising=False)
    monkeypatch.setattr(assembly, "prebuild_kernels", lambda *a, **kw: pytest.fail("compiled before admission"))
    with pytest.raises(ResidentMemoryRefused, match="hydro resident arrays"):
        assembly.prepare(settings=settings, state=None, physics=None, device_id=0, mode="standalone")


def test_incomplete_contract_keeps_existing_assembly_running_with_named_receipt(factories, monkeypatch):
    from gpuwm.hydro import assembly
    module, settings = factories
    calls = []
    settings["stages"][0]["phase"] = "prepare"
    module.prepare = lambda **kw: calls.append("factory") or SimpleNamespace()
    monkeypatch.setattr(assembly, "prebuild_kernels", lambda *a, **kw: calls.append("prebuild"))
    monkeypatch.setattr(assembly, "ResidentHydroDriver", lambda **kw: SimpleNamespace(**kw))
    driver = assembly.prepare(settings=settings, state=None, physics=None, device_id=0, mode="standalone")
    assert calls == ["prebuild", "factory"]
    assert not driver.memory_admission["complete"]
    assert not driver.memory_admission["pre_factory_admission"]


@pytest.mark.parametrize('free, override, expected', [
    pytest.param(8192, False, True, id='8192-False-True'),
    pytest.param(1, True, False, id='1-True-False'),
])
def test_admission_distinguishes_fit_unread_and_existing_override(factories, monkeypatch, free, override, expected):
    from gpuwm.core import resident_admission
    module, settings = factories
    module.prepare_memory = lambda **kw: StageMemory((ArrayMemory("work", (1024,), "float64"),), "roster", 0)
    monkeypatch.setattr(resident_admission, "device_free_bytes", lambda: free)
    monkeypatch.setenv("GPUWM_NO_MEMORY_GATE", "1" if override else "0")
    row = admit_plan(plan(settings, context=context()))
    assert row["pre_factory_admission"] is expected
    assert row["admission"]["need_bytes"] == 8192


def test_forecast_uses_parent_rounded_lsm_cadence_and_opt_out(factories):
    module, settings = factories
    module.prepare_memory = lambda **kw: StageMemory((), "prepared", 0)
    exp = SimpleNamespace(root=SimpleNamespace(run=SimpleNamespace(
        bldt=1.1, dt=40, nz=44, ny=70, nx=120)), run_seconds=86400)
    with settings_scope(settings):
        value = forecast_plan(exp)
    assert value.context.interval_seconds == 80  # not the requested 66 seconds
    assert value.context.parent_shape == (44, 70, 120)
    with settings_scope({"enabled": False}):
        assert forecast_plan(exp) is None


def test_actual_checkpoint_roster_is_cross_checked_without_reading_values():
    stage = StageMemory((ArrayMemory("state", (5,), "float64", checkpointed=True),), "prepared", 0)
    row = HydroMemoryPlan(context(), (("s", stage),), (), "sha").inventory()
    driver = SimpleNamespace(arrays={"s/state": SimpleNamespace(shape=(5,), dtype="float64")}, status_arrays={})
    verify_driver(row, driver)
    driver.arrays["s/state"].shape = (6,)
    with pytest.raises(ValueError, match="shape/dtype"):
        verify_driver(row, driver)
    driver.arrays.clear()
    with pytest.raises(ValueError, match="roster"):
        verify_driver(row, driver)


def test_forecast_envelope_and_named_terms_include_hydro(factories):
    from gpuwm.config import RunConfig
    from gpuwm.experiment import experiment_from_run_config
    from gpuwm.core import preflight, resident_admission
    module, settings = factories
    module.prepare_memory = lambda **kw: StageMemory((
        ArrayMemory("state", (10000,), "float64"),
        ArrayMemory("step", (20000,), "float32", lifetime="step"),
    ), "prepared roster SHA", 4096)
    exp = experiment_from_run_config(RunConfig(nx=8, ny=6, nz=4, dx=1000., dy=1000.,
        ztop=10000., dt=1., run_seconds=10.), datetime(2026, 1, 1))
    with settings_scope({"enabled": False}):
        before = preflight.estimate_experiment(exp)
    with settings_scope(settings):
        after = preflight.estimate_experiment(exp)
    row = after.hydro_memory
    assert not row["complete"]
    assert any("product collector/scheduler" in gap for gap in row["gaps"])
    assert after.resident_bytes - before.resident_bytes == row["resident_bytes"]
    assert after.transient_peak_bytes - before.transient_peak_bytes == row["transient_bytes"]
    assert after.envelope_intercept_bytes - before.envelope_intercept_bytes == 4096
    assert after.peak_envelope_bytes > before.peak_envelope_bytes + row["required_bytes"]
    assert sum(resident_admission.resident_forecast_terms(after).values()) == after.peak_envelope_bytes
    partial = replace(after, hydro_memory=dict(row, complete=False, gaps=["routing: no roster"]))
    assert "PARTIAL: routing: no roster" in partial.envelope_basis


@pytest.mark.parametrize('carrier, label', [
    pytest.param('status', 'runoff', id='status-runoff'),
])
def test_diagnostic_label_is_bound_to_unchanged_checkpoint_carrier(carrier, label):
    stage = StageMemory((ArrayMemory(carrier, (5,), "int32", checkpointed=True,
                                   status=True, status_name=label),), "prepared", 0)
    value = HydroMemoryPlan(context(), (("stage", stage),), (), "sha")
    row = json.loads(json.dumps(value.inventory(members=3)))
    assert list(row["backings"]) == ["stage/" + carrier]
    assert row["resident_bytes"] == 3 * 512
    assert row["arrays"][0]["status_name"] == "stage/" + label
    array = SimpleNamespace(shape=(5,), dtype="int32")
    driver = SimpleNamespace(arrays={"stage/" + carrier: array},
                             status_arrays={"stage/" + label: array})
    verify_driver(row, driver)
    driver.status_arrays = {"stage/wrong": array}
    with pytest.raises(ValueError, match="status carrier roster"):
        verify_driver(row, driver)


def test_status_contract_rejects_another_checkpoint_vector_with_same_metadata():
    stage = StageMemory((ArrayMemory("status", (5,), "int32", checkpointed=True, status=True),
                         ArrayMemory("other", (5,), "int32", checkpointed=True)), "prepared", 0)
    row = HydroMemoryPlan(context(), (("s", stage),), (), "sha").inventory()
    first = SimpleNamespace(shape=(5,), dtype="int32", nbytes=20, data=SimpleNamespace(ptr=100))
    second = SimpleNamespace(shape=(5,), dtype="int32", nbytes=20, data=SimpleNamespace(ptr=200))
    driver = SimpleNamespace(arrays={"s/status": first, "s/other": second},
                             status_arrays={"s/status": second})
    with pytest.raises(ValueError, match="status carrier binding"):
        verify_driver(row, driver)
    # A separate array view of exactly the declared carrier is valid.
    driver.status_arrays = {"s/status": SimpleNamespace(shape=(5,), dtype="int32", nbytes=20,
                                                       data=SimpleNamespace(ptr=100))}
    verify_driver(row, driver)
    driver.status_arrays["s/status"].shape = (1, 5)
    with pytest.raises(ValueError, match="status carrier binding"):
        verify_driver(row, driver)


@pytest.mark.parametrize('label, status', [
    pytest.param('', True, id='-True'),
])
def test_diagnostic_labels_require_a_flat_status_binding(label, status):
    with pytest.raises(ValueError, match="status name"):
        ArrayMemory("sticky_status", (5,), "int32", checkpointed=True,
                    status=status, status_name=label)


def test_duplicate_diagnostic_labels_are_rejected_before_construction():
    with pytest.raises(ValueError, match="duplicate hydro status names"):
        StageMemory(tuple(ArrayMemory(name, (5,), "int32", checkpointed=True,
                        status=True, status_name="lsm") for name in ("one", "two")), "prepared", 0)
