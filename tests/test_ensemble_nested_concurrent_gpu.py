"""Original nested production schemes match standalone members on one card.

The optional retained fixture must be a sealed prepared tree with Thompson,
MYNN, six-layer RUC, RTE-RRTMGP, feedback and an adaptive clock. Only its run
duration is shortened; grids, selectors, forcing, placement and clocks keep
their ordinary authority. This qualifies ordinary concurrency, not packing.
"""
from collections.abc import Mapping
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import struct
from threading import RLock

import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def _typed_words(value):
    """Preserve scalar types and exact floating words in carried metadata."""
    if value is None:
        return ("none",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", str(value))
    if type(value) is float:
        return ("float64", struct.pack("!d", value).hex())
    if type(value) is str:
        return ("str", value)
    if isinstance(value, Mapping):
        return ("mapping", tuple((str(key), _typed_words(item)) for key, item in sorted(value.items())))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_typed_words(item) for item in value))
    if hasattr(value, "dtype") and hasattr(value, "tobytes") and getattr(value, "shape", ()) == ():
        return ("numpy_scalar", value.dtype.str, value.tobytes().hex())
    raise TypeError(f"identity metadata has an unclassified scalar type: {type(value).__name__}")


def _driver_header_words(driver):
    return _typed_words({
        "call_counts": {key: int(value) for key, value in driver.call_counts.items()},
        "ysu_nan_guard_fires": int(driver.ysu_nan_guard_fires),
        "microphysics_updates": int(driver.microphysics_updates),
        "carriers": driver.carriers.state(),
        "surface_radiation_policy": driver.carriers.policy})


def _carried_words(model):
    from gpuwm.io.restart import (
        state_manifest, _scratch_manifest, _driver_manifest,
        root_external_lbc_clock_identity)
    result = {}
    for node in model.walk_parent_first():
        inventory = state_manifest(node.state)
        inventory.update(_scratch_manifest(node.state))
        inventory.update(_driver_manifest(node.state.physics))
        arrays = {}
        for name, value in sorted(inventory.items()):
            host = value.get() if hasattr(value, "get") else value
            arrays[name] = (tuple(host.shape), str(host.dtype),
                            hashlib.sha256(host.tobytes(order="C")).hexdigest())
        result[node.cfg.grid_id] = {
            "arrays": arrays,
            "clock": _typed_words({name: getattr(node.clock, name) for name in (
                "ticks", "step_ticks", "tick_den", "run_ticks", "step_count", "dt_fp32", "dtbc_fp32")}),
            "state_elapsed_seconds": _typed_words(node.state.elapsed_seconds),
            "driver_header": _driver_header_words(node.state.physics),
            "root_external_lbc_clock_identity": root_external_lbc_clock_identity(node.state, node.cfg.run),
            "physics_calls": dict(node.state.physics.call_counts),
            "microphysics_updates": node.state.physics.microphysics_updates,
        }
    return result


class _HashCollector:
    """Capture deadlines with the ordinary collector's allocation envelope."""
    def __init__(self, members):
        from gpuwm.ensemble.batch_products import default_product_requests, DEFAULT_THRESHOLDS
        self.members, self.tile_rows = members, 32
        self.member_order = tuple(range(members))
        self.requests = default_product_requests(DEFAULT_THRESHOLDS)[0]
        self.member_archive = None
        self.frames = []
        self.lock = RLock()

    def memory_plan(self, shape, *, refl_levels=0, host_inputs=False):
        from gpuwm.ensemble.batch_product_output import headline_diagnostic_memory_plan
        return headline_diagnostic_memory_plan(shape, refl_levels=refl_levels, host_inputs=host_inputs)

    def submit(self, *, valid_time, grid_id, member_id, episode, **unused):
        with self.lock:
            self.frames.append((member_id, grid_id, episode, str(valid_time)))

    def finish_run(self):
        return {"frames": sorted(self.frames), "simulation_data_written": False}

    def require_complete(self):
        return None


def test_distinct_nested_adaptive_members_equal_the_same_members_run_alone(tmp_path):
    import cupy as cp
    from gpuwm import stage_cli
    from gpuwm.ensemble.production import PreparedEnsembleSession, finished_member_release
    from gpuwm.ensemble.request import EnsembleRequest
    from gpuwm.ensemble.runtime_context import MemberOutputCapture, member_output_scope
    from gpuwm.prepared_domain_tree_forecast import preflight_prepared_tree, run_prepared_tree
    prepared = os.environ.get("GPUWM_TEST_NESTED_PREPARED_ROOT")
    config = os.environ.get("GPUWM_TEST_NESTED_EXPERIMENT_CONFIG")
    if not prepared or not config:
        pytest.skip("set GPUWM_TEST_NESTED_PREPARED_ROOT and GPUWM_TEST_NESTED_EXPERIMENT_CONFIG to the retained sealed tree")
    bundle = stage_cli.resolve_bundle(Path(prepared))
    assert bundle["layout"] == "tree"
    authority = stage_cli.tree_digests(bundle, Path(config))
    inputs = preflight_prepared_tree(prepared_root=Path(prepared), experiment_config=Path(config),
        preparation_receipt_sha256=authority["preparation_receipt"],
        experiment_config_sha256=authority["experiment_config"])
    exp = inputs.experiment
    assert len(exp.domains) >= 2 and exp.feedback == 1
    for domain in exp.domains:
        cfg = domain.run
        assert (cfg.mp_physics, cfg.bl_pbl_physics, cfg.sf_sfclay_physics,
                cfg.sf_surface_physics, cfg.num_soil_layers) == (8, 5, 5, 3, 6)
        assert cfg.ra_rrtmg_variant == "rte-rrtmgp" and cfg.use_adaptive_time_step
    prefix = replace(inputs, experiment=replace(exp, run_seconds=min(60., exp.run_seconds)))
    request = EnsembleRequest(2, base_seed=77321, recipe="surface-state", perturbation={
        "kind": "surface-state", "soil_moisture_scale": [.8, 1.2], "sst_offset_k": [-1., 1.]})
    def owner(collector, path):
        return PreparedEnsembleSession(request, output_directory=path, collector=collector,
            input_provider=lambda **unused: prefix, array_module=cp)
    def tracked(member_inputs, *, output_directory, observer=None, **options):
        holder = {}
        previous = options.pop("ensemble_bootstrap", None)
        def capture(**kwargs):
            holder["model"] = kwargs["model"]
            holder["initial_words"] = _carried_words(kwargs["model"])
            return None if previous is None else previous(**kwargs)
        result = run_prepared_tree(member_inputs, output_directory=output_directory,
            ensemble_bootstrap=capture, observer=observer, **options)
        cp.cuda.get_current_stream().synchronize()
        result["initial_identity_words"] = holder.pop("initial_words")
        result["identity_words"] = _carried_words(holder.pop("model"))
        return result
    ordinary_collector = _HashCollector(2)
    ordinary = owner(ordinary_collector, tmp_path / "ordinary")
    references = {}
    for member in range(2):
        output = tmp_path / "ordinary" / f"member-{member}"
        output.mkdir(parents=True)
        capture = MemberOutputCapture(ordinary_collector.submit, member,
            initialize_callback=ordinary._initialization_callback(member))
        with finished_member_release(cp), member_output_scope(capture):
            report = tracked(prefix, output_directory=output, io_mode="none")
            references[member] = {key: report[key] for key in ("initial_identity_words", "identity_words")}
    assert references[0] != references[1]
    concurrent_collector = _HashCollector(2)
    concurrent = owner(concurrent_collector, tmp_path / "concurrent")
    result = concurrent.run_prepared(tracked, prefix, output_directory=tmp_path / "concurrent", io_mode="none")
    actual = {}
    concurrent_rows = 0
    for row in result["member_results"]:
        value = row["result"]
        if value.get("backend") == "ordinary_concurrent_members":
            concurrent_rows += 1
            for member in value["members"]:
                actual[member["member_id"]] = {key: member["result"][key]
                    for key in ("initial_identity_words", "identity_words")}
                assert member["cuda_scope"]["adaptive_cfl_ownership"] == "independent_member_context"
        else:
            actual[row["packing"]["member_indices"][0]] = {key: value[key]
                for key in ("initial_identity_words", "identity_words")}
    assert concurrent_rows > 0, "the retained fixture needs room for two concurrent original models"
    assert actual == references
    assert sorted(concurrent_collector.frames) == sorted(ordinary_collector.frames)
    assert concurrent._surface_receipts == ordinary._surface_receipts
    assert stage_cli.tree_digests(bundle, Path(config)) == authority
