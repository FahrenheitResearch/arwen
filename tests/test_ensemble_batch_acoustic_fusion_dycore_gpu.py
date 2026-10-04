"""Joined acoustic phases preserve the complete independent RK trajectory."""
from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def _batch(count, terrain, top_lid):
    import cupy as cp
    from test_ensemble_batch_dycore_gpu import _prepared
    from gpuwm.ensemble.batch_state import BatchedDomainState, SHARED_STATE_CANDIDATES
    inputs, specs, slots = _prepared(count, terrain=terrain, mapped=terrain,
                                    order=5, emdiv=0.0, km_opt=4, diff6=2)
    inputs = tuple(replace(member, cfg=replace(member.cfg, top_lid=top_lid)) for member in inputs)
    shared = tuple(sorted(SHARED_STATE_CANDIDATES & inputs[0].arrays.keys()))
    available, _ = cp.cuda.runtime.memGetInfo()
    state = BatchedDomainState.from_prepared(inputs, array_module=cp,
        available_bytes=available, shared_fields=shared, scratch_slots=slots, extra_specs=specs)
    return state, inputs


@pytest.mark.parametrize("count", (1, 4, 10, 20))
@pytest.mark.parametrize("terrain,top_lid", ((False, False), (True, True)))
@pytest.mark.parametrize("layout,family", (("none", False), ("outermost", False),
                                         ("innermost", False), ("outermost", True)))
@pytest.mark.parametrize("shared", (False, True))
def test_fused_acoustic_complete_words_and_history(count, terrain, top_lid, layout, family, shared, tmp_path, monkeypatch):
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.ensemble.batch_dycore import prepare_dry_step, member_domain_view
    from gpuwm.io import nc_writer_bridge
    from gpuwm.io.wrfout import WrfoutWriter, state_frame
    from test_ensemble_batch_bigstep_gpu import _scalar_state
    assert nc_writer_bridge.unavailable_reason() is None, "Rust writer is required for exact history proof"
    state, inputs = _batch(count, terrain, top_lid)
    advance = prepare_dry_step(state, layout_trial=layout, advection_family=family,
                               acoustic_fusion=True, acoustic_shared=shared)
    if count > 1:
        from gpuwm.ensemble import batch_acoustic_fusion
        def unexpected_factory(*args, **kwargs):
            pytest.fail("a prepared step rebuilt acoustic source or launch handles")
        monkeypatch.setattr(batch_acoustic_fusion, "fusion_source", unexpected_factory)
        monkeypatch.setattr(batch_acoustic_fusion, "prepare_acoustic_substep_launch", unexpected_factory)
    cfg = state.cfg
    attrs = {"START_DATE": "2026-10-02_00:00:00", "DT": np.float32(cfg.dt)}

    def write(path, view, valid):
        fields = state_frame(view, include_diagnostic_pressure=True)
        with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx, dy=cfg.dy,
                         global_attrs=attrs, field_schema=fields, engine="rust") as writer:
            writer.write_frame(valid, fields)

    for step_index in range(2):
        if step_index == 1 and count > 1:
            # A prepared second step may not allocate a new state/intermediate.
            warmed = cp.get_default_memory_pool().used_bytes()
            allocated = []
            original = cp.cuda.get_allocator()
            def observe(nbytes):
                allocated.append(int(nbytes))
                return original(nbytes)
            with cp.cuda.using_allocator(observe):
                advance()
            cp.cuda.get_current_stream().synchronize()
            assert not allocated, ("warmed fused step allocated CUDA memory", allocated)
            assert cp.get_default_memory_pool().used_bytes() == warmed
        else:
            advance()
            cp.cuda.get_current_stream().synchronize()
        for member_index, member in enumerate(inputs):
            ordinary = replace(member, arrays={name: member.arrays[name] for name in state_array_shapes(cfg)})
            scalar = _scalar_state(ordinary)
            for _ in range(step_index + 1):
                dycore.step(scalar, cfg, acoustic=True)
            for name in state_array_shapes(cfg):
                actual = cp.asnumpy(state.member_view(name, member_index)).view(np.uint32)
                expected = cp.asnumpy(getattr(scalar, name)).view(np.uint32)
                assert actual.tobytes() == expected.tobytes(), (step_index, member_index, name)
            valid = ("2026-10-02_00:00:03", "2026-10-02_00:00:06")[step_index]
            batch_file = tmp_path / f"batch-{step_index}-{member_index}.nc"
            scalar_file = tmp_path / f"scalar-{step_index}-{member_index}.nc"
            write(batch_file, member_domain_view(state, member_index), valid)
            write(scalar_file, scalar, valid)
            assert batch_file.read_bytes() == scalar_file.read_bytes(), (step_index, member_index, "history")
            assert state.elapsed_seconds == scalar.elapsed_seconds
        assert state.clock["step_count"] == step_index + 1
        assert state.clock["ticks"] == (step_index + 1) * state.clock["step_ticks"]
