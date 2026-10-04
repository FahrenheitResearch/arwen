"""Full dry acoustic RK trajectories against independent scalar drivers."""
from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu
from test_ensemble_batch_bigstep_gpu import _inputs, _scalar_state
from gpuwm.core.device_inventory import state_array_shapes
from gpuwm.ensemble.batch_dycore import member_domain_view, prepare_dry_step
from gpuwm.ensemble.batch_glue import workspace_specs
from gpuwm.ensemble.batch_state import BatchedDomainState, SHARED_STATE_CANDIDATES
from gpuwm.wrf_exact import ENABLED

pytestmark = [pytest.mark.gpu, requires_gpu]


def _prepared(count, *, terrain, mapped, order, emdiv, km_opt=1, diff6=0):
    inputs = _inputs(count, terrain=terrain, mapped=mapped, order=order)
    cfg = replace(inputs[0].cfg, km_opt=km_opt, diff_opt=2, emdiv=emdiv, diff_6th_opt=diff6)
    specs = workspace_specs(cfg)
    mixing_slots = {}
    if km_opt == 4 or diff6:
        from gpuwm.ensemble import batch_mixing
        specs += batch_mixing.workspace_specs(cfg, has_msf=inputs[0].scalars["has_msf"])
        mixing_slots = batch_mixing.required_scratch_slots(cfg)
    prepared = []
    for member in inputs:
        arrays = dict(member.arrays)
        arrays.update({spec.name: np.zeros(spec.shape, spec.dtype) for spec in specs})
        scratch = {name: value for name, value in member.scratch.items() if name != "smag_mut"}
        prepared.append(replace(member, cfg=cfg, arrays=arrays, scratch=scratch))
    slots = {name: np.float32 for name in (
        "acoustic_mu_pp_old", "acoustic_th_pp_old", "acoustic_c2a",
        "acoustic_a", "acoustic_alpha", "acoustic_gamma")}
    if emdiv:
        slots["acoustic_mudf"] = np.float32
    slots.update(mixing_slots)
    return tuple(prepared), specs, slots


@pytest.mark.parametrize("count", [1, 4, 10, 20, 40])
@pytest.mark.parametrize("terrain,mapped,order,emdiv", [
    (False, False, 2, 0.0), (False, True, 5, 0.0),
    (True, False, 2, 0.01), (True, True, 5, 0.01),
])
def test_two_complete_acoustic_rk_steps_match_every_scalar_state_word(count, terrain, mapped, order, emdiv):
    import cupy as cp
    from gpuwm.core import dycore
    if ENABLED and count > 1:
        pytest.skip("strict complete graph awaits separate helper/face qualification")
    inputs, specs, slots = _prepared(count, terrain=terrain, mapped=mapped, order=order, emdiv=emdiv)
    shared = tuple(sorted(SHARED_STATE_CANDIDATES & inputs[0].arrays.keys()))
    batch = BatchedDomainState.from_prepared(
        inputs, array_module=cp, available_bytes=2**30,
        shared_fields=shared, scratch_slots=slots, extra_specs=specs)
    step = prepare_dry_step(batch)
    for step_index in range(2):
        step()
        cp.cuda.get_current_stream().synchronize()
        # New independent scalar inputs and original driver, outside batch code.
        for member_index, member in enumerate(inputs):
            ordinary = replace(member, arrays={name: member.arrays[name]
                                               for name in state_array_shapes(member.cfg)})
            scalar = _scalar_state(ordinary)
            for _ in range(step_index + 1):
                dycore.step(scalar, member.cfg, acoustic=True)
            cp.cuda.get_current_stream().synchronize()
            for name in state_array_shapes(member.cfg):
                got = cp.asnumpy(batch.member_view(name, member_index)).view(np.uint32)
                expected = cp.asnumpy(getattr(scalar, name)).view(np.uint32)
                assert got.tobytes() == expected.tobytes(), (
                    step_index, member_index, name,
                    int(np.count_nonzero(got != expected)))
            assert batch.elapsed_seconds == scalar.elapsed_seconds
        assert batch.clock["step_count"] == step_index + 1
        assert batch.clock["ticks"] == (step_index + 1) * batch.clock["step_ticks"]


@pytest.mark.parametrize("count", [1, 4, 10, 20])
@pytest.mark.parametrize("km_opt,diff6", [(4, 0), (4, 2), (1, 2)])
@pytest.mark.parametrize("terrain,mapped", [(False, False), (True, True)])
def test_complete_mixed_rk_steps_preserve_original_state_words(count, km_opt, diff6, terrain, mapped):
    import cupy as cp
    from gpuwm.core import dycore
    inputs, specs, slots = _prepared(count, terrain=terrain, mapped=mapped, order=5,
                                    emdiv=0.01, km_opt=km_opt, diff6=diff6)
    shared = tuple(sorted(SHARED_STATE_CANDIDATES & inputs[0].arrays.keys()))
    batch = BatchedDomainState.from_prepared(
        inputs, array_module=cp, available_bytes=2**30, shared_fields=shared,
        scratch_slots=slots, extra_specs=specs)
    step = prepare_dry_step(batch)
    step()
    step()
    cp.cuda.get_current_stream().synchronize()
    for member_index, member in enumerate(inputs):
        ordinary = replace(member, arrays={name: member.arrays[name]
                                           for name in state_array_shapes(member.cfg)})
        scalar = _scalar_state(ordinary)
        dycore.step(scalar, member.cfg, acoustic=True)
        dycore.step(scalar, member.cfg, acoustic=True)
        for name in state_array_shapes(member.cfg):
            got = cp.asnumpy(batch.member_view(name, member_index)).view(np.uint32)
            expected = cp.asnumpy(getattr(scalar, name)).view(np.uint32)
            different = got != expected
            assert not np.any(different), (member_index, name, int(np.count_nonzero(different)),
                                          np.argwhere(different)[:8].tolist())


@pytest.mark.parametrize("count", [1, 4, 10, 20, 40])
def test_member_history_files_are_byte_identical_to_independent_single_runs(count, tmp_path):
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.io import nc_writer_bridge
    from gpuwm.io.wrfout import WrfoutWriter, state_frame
    if ENABLED and count > 1:
        pytest.skip("strict complete graph awaits separate helper/face qualification")
    reason = nc_writer_bridge.unavailable_reason()
    if reason is not None:
        pytest.fail("Rust history writer must be supplied for the trajectory proof: " + reason)
    inputs, specs, slots = _prepared(count, terrain=True, mapped=True, order=5, emdiv=0.01)
    cfg = inputs[0].cfg
    shared = tuple(sorted(SHARED_STATE_CANDIDATES & inputs[0].arrays.keys()))
    batch = BatchedDomainState.from_prepared(
        inputs, array_module=cp, available_bytes=2**30,
        shared_fields=shared, scratch_slots=slots, extra_specs=specs)
    step = prepare_dry_step(batch)
    times = ("2026-10-02_00:00:03", "2026-10-02_00:00:06")
    attrs = {"START_DATE": "2026-10-02_00:00:00", "DT": np.float32(cfg.dt)}

    def write(path, view, valid_time):
        fields = state_frame(view, include_diagnostic_pressure=True)
        with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                          dx=cfg.dx, dy=cfg.dy, global_attrs=attrs,
                          field_schema=fields, engine="rust") as writer:
            writer.write_frame(valid_time, fields)

    for step_index, valid_time in enumerate(times):
        step()
        cp.cuda.get_current_stream().synchronize()
        for member_index, member in enumerate(inputs):
            ordinary = replace(member, arrays={name: member.arrays[name]
                                               for name in state_array_shapes(cfg)})
            scalar = _scalar_state(ordinary)
            for _ in range(step_index + 1):
                dycore.step(scalar, cfg, acoustic=True)
            batch_path = tmp_path / f"batch-{step_index}-{member_index}.nc"
            scalar_path = tmp_path / f"single-{step_index}-{member_index}.nc"
            write(batch_path, member_domain_view(batch, member_index), valid_time)
            write(scalar_path, scalar, valid_time)
            assert batch_path.read_bytes() == scalar_path.read_bytes(), (step_index, member_index)
