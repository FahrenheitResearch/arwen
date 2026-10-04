"""Shared bootstrap records and integer-only legacy geometry adapters."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import ast
import inspect
import sys
from unittest.mock import Mock

import numpy as np
import pytest

from gpuwm.ensemble.batch_physics_init import (
    InitializedMemberPhysics, _clone_tree, legacy_member_source,
    prepare_legacy_member_radiation,
)


def test_frozen_diagnostics_rebind_words_without_losing_aliases():
    @dataclass(frozen=True)
    class Record:
        rain: object
        snow: object
    source = np.array([0x80000000, 0x7fc00001], np.uint32)
    destination = source.copy()
    record = Record(source, source)
    result = _clone_tree({"record": record, "aliases": [source, (source,)]}, np.ndarray,
                         lambda value: destination)
    assert result["record"].rain is destination
    assert result["record"].snow is destination
    assert result["aliases"][0] is destination
    assert result["aliases"][1][0] is destination
    assert source.tobytes() == destination.tobytes()


def test_bootstrap_carrier_controls_are_private():
    source = SimpleNamespace(records={"swdown": {"time": 0}})
    cloned = _clone_tree(source, np.ndarray, lambda value: value)
    cloned.records["swdown"]["time"] = 12
    assert source.records["swdown"]["time"] == 0


def test_legacy_source_changes_only_declared_integer_addresses():
    path = Path(__file__).parents[1] / "gpuwm/core/kernels/rrtmg_legacy_adapter.cu"
    source = path.read_text(encoding="utf-8")
    result = legacy_member_source(source, members=10, member_columns=22500)
    assert "ozmixt + ((long long)col % __ensemble_member_columns) * levsiz" in result
    assert "long long dest = (long long)k * ncol + actual;" in result
    assert "glw[actual] =" in result and "olr[actual] =" in result
    for spelling in ("RLA_DV(", "RLA_ML(", "RLA_AD(", "__fdiv_rn(", "__fmul_rn("):
        assert source.count(spelling) == result.count(spelling)
    with pytest.raises(ValueError, match="new audit"):
        legacy_member_source(source.replace("glw[c0 + col]", "glw[c0+col]"), members=10, member_columns=22500)


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_legacy_member_extent_is_not_coerced(invalid):
    with pytest.raises((TypeError, ValueError)):
        legacy_member_source("", members=invalid, member_columns=1)


def test_single_member_keeps_original_radiation_callable_without_importing_cuda():
    original = object()
    assert prepare_legacy_member_radiation(original, members=1, available_bytes=0) is original


def test_microphysics_facade_keeps_step_and_updates_only_declared_fields(monkeypatch):
    calls = []
    batch = SimpleNamespace(cfg=SimpleNamespace(dt=12), elapsed_seconds=12,
                            storage=SimpleNamespace(arrays={"thp": object(), "p": object()}))
    bank = SimpleNamespace(pack=lambda name, array: calls.append(("pack", name)),
                           unpack=lambda name, array: calls.append(("unpack", name)))
    driver = SimpleNamespace(accept_microphysics=lambda result, **kw: calls.append(("accept", kw)))
    result = object()
    adapter = InitializedMemberPhysics(batch, driver, SimpleNamespace(), bank, None,
                                       lambda **kw: result, {"model_member_fields": ["thp", "p"]})
    assert adapter.apply_microphysics() is result
    assert ("accept", {"dt": 12}) in calls
    assert ("unpack", "thp") in calls
    assert ("unpack", "p") not in calls


def test_wrf_wrapper_preserves_ordinary_initializer_inputs_and_delegates_once(monkeypatch):
    import gpuwm.ensemble.batch_physics_init as module
    calls = []
    pool = SimpleNamespace(used_bytes=Mock(side_effect=[100, 356, 900]))
    xp = SimpleNamespace(get_default_memory_pool=lambda: pool,
                         cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)))
    batch, cfg, restored = object(), object(), SimpleNamespace(raw={"XLAT": object(), "XLONG": object()})
    state = SimpleNamespace(p_top=5000.0)
    driver, radiation, landuse, ozone, trace = object(), SimpleNamespace(), object(), object(), object()
    def restore(actual, actual_cfg):
        assert actual is restored and actual_cfg is cfg
        calls.append("restore")
        return state
    def make_radiation(actual_cfg, start, latitude, longitude, **kwargs):
        assert actual_cfg is cfg and start == "clock-authority"
        assert latitude is restored.raw["XLAT"] and longitude is restored.raw["XLONG"]
        assert kwargs == {"p_top": 5000.0, "column_chunk": 16, "trace_gas_overrides": trace}
        calls.append("radiation")
        return radiation
    def initialize(actual_state, actual_restored, actual_cfg, **kwargs):
        assert actual_state is state and actual_restored is restored and actual_cfg is cfg
        assert kwargs == {"radiation": radiation, "radiation_start_time": "clock-authority",
                          "radiation_latitude": restored.raw["XLAT"], "radiation_longitude": restored.raw["XLONG"],
                          "landuse": landuse, "constant_glw_wm2": 310.0, "fractional_seaice": True, "cam_ozone": ozone}
        calls.append("initialize")
        return driver
    def packed(actual_batch, actual_state, actual_driver, **kwargs):
        assert (actual_batch, actual_state, actual_driver) == (batch, state, driver)
        assert kwargs["available_bytes"] == 123456
        assert kwargs["bootstrap_pool_live_increment_bytes"] == 256
        assert kwargs["array_module"] is xp
        calls.append("pack")
        return SimpleNamespace(receipt={"initialization_source": kwargs["initialization_source"]})
    monkeypatch.setattr(module, "_native_bootstrap_requirements", lambda value: (cfg, 4, ()))
    monkeypatch.setattr(module, "initialize_member_physics_from_bootstrap", packed)
    monkeypatch.setitem(sys.modules, "gpuwm.ingest.wrfinput",
                        SimpleNamespace(restore_domain_state=restore, initialize_wrfinput_physics=initialize))
    monkeypatch.setitem(sys.modules, "gpuwm.core.radiation_composition", SimpleNamespace(make_radiation=make_radiation))
    result = module.initialize_wrfinput_member_physics(batch, restored, start_time="clock-authority",
        landuse=landuse, available_bytes=123456, column_chunk=16, constant_glw_wm2=310.0,
        fractional_seaice=True, cam_ozone=ozone, trace_gas_overrides=trace, array_module=xp)
    assert calls == ["restore", "radiation", "initialize", "pack"]
    assert radiation.column_chunk == 16
    assert result.receipt["pool_live_after_initialization_bytes"] == 900


def test_native_bootstrap_binding_is_source_neutral_and_uses_existing_radiation_owner():
    from gpuwm.ensemble.batch_physics_init import initialize_member_physics_from_bootstrap
    tree = ast.parse(inspect.getsource(initialize_member_physics_from_bootstrap))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name and (name.startswith("gpuwm.ingest") or "wrfinput" in name) for name in imports)
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "make_radiation" not in calls
    assert "legacy_radiation_adapter" in calls
    assert not any(isinstance(node, ast.Attribute) and node.attr == "raw" for node in ast.walk(tree))


def test_bootstrap_plan_uses_one_alias_census_without_retaining_source_arrays():
    import gc
    import weakref
    from gpuwm.config import RunConfig
    from gpuwm.ensemble.batch_state import state_array_specs, SHARED_STATE_CANDIDATES
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.ensemble.batch_physics_init import native_bootstrap_allocation_plans
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=3000.0, dy=3000.0, ztop=12000.0,
                    dt=12.0, run_seconds=120.0, moist=True, mp_physics=8)
    private, shared = np.zeros((8, 8), np.float32), np.ones((8, 8), np.float32)
    private_ref, shared_ref = weakref.ref(private), weakref.ref(shared)
    driver = SimpleNamespace(state=SimpleNamespace(p=np.zeros((8, 8, 8), np.float32)),
                             fields={"xland": shared, "t2": private}, diagnostic_alias=private)
    specs = state_array_specs(cfg, shared_fields=tuple(SHARED_STATE_CANDIDATES & state_array_shapes(cfg).keys()))
    plans = native_bootstrap_allocation_plans(cfg, driver, state_specs=specs)
    assert len(plans.bootstrap_array_paths) == 2
    assert len(plans.shared_plan.arrays) == 1
    assert sum(field.name.startswith("bootstrap_") for field in plans.column_fields) == 1
    assert plans.bank_plan.required_bytes(10) > plans.bank_plan.required_bytes(4)
    assert plans.shared_plan.required_bytes(10) == plans.shared_plan.required_bytes(4)
    del driver, private, shared
    gc.collect()
    assert private_ref() is None and shared_ref() is None
