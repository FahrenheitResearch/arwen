"""Explicit layout trials must preserve the complete RK driver and history."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu


def test_layout_trial_rejects_an_unknown_selection_before_binding():
    from gpuwm.ensemble.batch_dycore import prepare_dry_step
    with pytest.raises(ValueError, match="layout_trial"):
        prepare_dry_step(None, layout_trial="unknown")


def test_layout_trial_estimate_keeps_mixing_inventory_and_counts_exact_buffers():
    from tools.ensemble_batch_step_probe import run
    from gpuwm.ensemble.batch_layout_trials import layout_trial_workspace_bytes
    args = SimpleNamespace(nx=11, ny=9, nz=5, dx=1000, dt=3, steps=2, warmup=0,
                           members=[1, 4, 10, 20, 40], reserve_mib=0, km_opt=4, diff6=2,
                           estimate_only=True, price_usd_per_hour=None,
                           layout_trial="none", receipt=Path("unused-estimate.json"))
    baseline = run(args)
    args.layout_trial = "innermost"
    inner = run(args)
    for ordinary, trial in zip(baseline["rows"], inner["rows"]):
        members = ordinary["members"]
        extra = layout_trial_workspace_bytes(args, members, "innermost")
        assert trial["required_bytes"] == ordinary["required_bytes"] + extra
        assert trial["additional_trial_workspace_bytes"] == extra
    assert inner["rows"][0]["additional_trial_workspace_bytes"] == 0


def test_inner_admission_rounds_each_odd_buffer_separately():
    from gpuwm.ensemble.batch_layout_trials import (
        layout_trial_workspace_inventory, layout_trial_workspace_bytes,
        layout_trial_workspace_payload_bytes)
    cfg = SimpleNamespace(nx=11, ny=9, nz=5)
    rows = layout_trial_workspace_inventory(cfg, 4, "innermost")
    assert len(rows) == 11
    assert all(row["allocated_bytes"] % 512 == 0 for row in rows)
    assert layout_trial_workspace_bytes(cfg, 4, "innermost") == sum(
        row["allocated_bytes"] for row in rows)
    assert layout_trial_workspace_payload_bytes(cfg, 4, "innermost") == sum(
        row["payload_bytes"] for row in rows)
    assert layout_trial_workspace_bytes(cfg, 4, "innermost") > layout_trial_workspace_payload_bytes(cfg, 4, "innermost")


def _state(count, *, km_opt, diff6, terrain=True, mapped=True):
    import cupy as cp
    from test_ensemble_batch_dycore_gpu import _prepared
    from gpuwm.ensemble.batch_state import BatchedDomainState, SHARED_STATE_CANDIDATES
    inputs, specs, slots = _prepared(count, terrain=terrain, mapped=mapped, order=5,
                                    emdiv=0.01, km_opt=km_opt, diff6=diff6)
    shared = tuple(sorted(SHARED_STATE_CANDIDATES & inputs[0].arrays.keys()))
    available, _ = cp.cuda.runtime.memGetInfo()
    state = BatchedDomainState.from_prepared(
        inputs, array_module=cp, available_bytes=available, shared_fields=shared,
        scratch_slots=slots, extra_specs=specs)
    return state, inputs


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("count", (1, 4, 10, 20, 40))
@pytest.mark.parametrize("selection", ("outermost", "innermost"))
@pytest.mark.parametrize("km_opt,diff6", ((1, 0), (1, 2), (4, 2)))
@pytest.mark.parametrize("family", (False, True))
def test_complete_trial_rk_words_match_independent_scalar(count, selection, km_opt, diff6, family, monkeypatch):
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.ensemble.batch_dycore import prepare_dry_step
    from gpuwm.ensemble import batch_bookkeeping, batch_layout_trials
    from gpuwm.wrf_exact import ENABLED
    from test_ensemble_batch_bigstep_gpu import _scalar_state
    if ENABLED and count > 1:
        pytest.skip("strict full graph needs its existing helper/face qualification")
    state, inputs = _state(count, km_opt=km_opt, diff6=diff6)
    for name in dycore._TENDENCIES:
        getattr(state, name).fill(-19.5)
    zero_rows = []
    original_bookkeeping = batch_bookkeeping.prepare_bookkeeping
    def observe(pairs, **options):
        pairs = tuple(pairs)
        if options.get("zero"):
            zero_rows.append(tuple(value[1].data.ptr for value in pairs))
        return original_bookkeeping(pairs, **options)
    monkeypatch.setattr(batch_bookkeeping, "prepare_bookkeeping", observe)
    unpack_calls = []
    original_unpack = batch_layout_trials.unpack_member_innermost
    def observed_unpack(value, **options):
        unpack_calls.append(value.shape)
        return original_unpack(value, **options)
    monkeypatch.setattr(batch_layout_trials, "unpack_member_innermost", observed_unpack)
    step = prepare_dry_step(state, layout_trial=selection, advection_family=family)
    if count == 1:
        assert step.trial_receipt["effective"] == "original_n1"
        assert not zero_rows
    else:
        excluded = {"rth_t", "ru_t", "rv_t", "rw_t"}
        expected_rows = tuple(getattr(state, name).data.ptr for name in dycore._TENDENCIES if name not in excluded)
        assert zero_rows == [expected_rows]
        assert set(step.trial_receipt["fused_zero_fields"]) == excluded
        assert step.trial_receipt["additional_workspace_bytes"] == batch_layout_trials.layout_trial_workspace_bytes(
            state.cfg, count, selection)
    step()
    cp.cuda.get_current_stream().synchronize()
    warmed_live_bytes = cp.get_default_memory_pool().used_bytes()
    allocated = []
    original_allocator = cp.cuda.get_allocator()
    def observed_allocator(nbytes):
        allocated.append(int(nbytes))
        return original_allocator(nbytes)
    with cp.cuda.using_allocator(observed_allocator):
        step()
    cp.cuda.get_current_stream().synchronize()
    if count > 1:
        assert not allocated, (selection, "warmed trial step allocated CUDA temporaries", allocated)
        assert cp.get_default_memory_pool().used_bytes() == warmed_live_bytes
    if count > 1 and selection == "innermost":
        # Five unpack calls in each of three stages must run inside each step.
        assert len(unpack_calls) == 30
        assert step.trial_receipt["transitions_in_step"]
    for member_index, member in enumerate(inputs):
        ordinary = replace(member, arrays={name: member.arrays[name]
                                           for name in state_array_shapes(member.cfg)})
        scalar = _scalar_state(ordinary)
        dycore.step(scalar, member.cfg, acoustic=True)
        dycore.step(scalar, member.cfg, acoustic=True)
        for name in state_array_shapes(member.cfg):
            actual = cp.asnumpy(state.member_view(name, member_index)).view(np.uint32)
            expected = cp.asnumpy(getattr(scalar, name)).view(np.uint32)
            assert actual.tobytes() == expected.tobytes(), (
                selection, km_opt, diff6, member_index, name, int(np.count_nonzero(actual != expected)))
        assert state.elapsed_seconds == scalar.elapsed_seconds
    assert state.clock["step_count"] == 2
    assert state.clock["ticks"] == 2 * state.clock["step_ticks"]


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("count", (1, 4, 10, 20, 40))
@pytest.mark.parametrize("selection", ("outermost", "innermost"))
@pytest.mark.parametrize("family", (False, True))
def test_complete_trial_history_bytes_match_scalar(count, selection, family, tmp_path):
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.ensemble.batch_dycore import member_domain_view, prepare_dry_step
    from gpuwm.io import nc_writer_bridge
    from gpuwm.io.wrfout import WrfoutWriter, state_frame
    from gpuwm.wrf_exact import ENABLED
    from test_ensemble_batch_bigstep_gpu import _scalar_state
    if ENABLED and count > 1:
        pytest.skip("strict full graph needs its existing helper/face qualification")
    reason = nc_writer_bridge.unavailable_reason()
    if reason is not None:
        pytest.fail("Rust history writer is required for this output proof: " + reason)
    state, inputs = _state(count, km_opt=1, diff6=0)
    cfg = state.cfg
    advance = prepare_dry_step(state, layout_trial=selection, advection_family=family)
    advance()
    cp.cuda.get_current_stream().synchronize()
    valid_time = "2026-10-02_00:00:03"
    attrs = {"START_DATE": "2026-10-02_00:00:00", "DT": np.float32(cfg.dt)}
    def write(path, view):
        fields = state_frame(view, include_diagnostic_pressure=True)
        with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
                          global_attrs=attrs, field_schema=fields, engine="rust") as writer:
            writer.write_frame(valid_time, fields)
    for member_index, member in enumerate(inputs):
        ordinary = replace(member, arrays={name: member.arrays[name]
                                           for name in state_array_shapes(cfg)})
        scalar = _scalar_state(ordinary)
        dycore.step(scalar, cfg, acoustic=True)
        batch_file = tmp_path / f"trial-{member_index}.nc"
        scalar_file = tmp_path / f"scalar-{member_index}.nc"
        write(batch_file, member_domain_view(state, member_index))
        write(scalar_file, scalar)
        assert batch_file.read_bytes() == scalar_file.read_bytes(), (selection, member_index)
