"""Actual-state dry EOS binding against independent original scalar helpers.

Run both ordinary and strict process modes to exercise their distinct pressure
ABIs. These tests grade diagnostics, not complete forecast/history execution.
"""
from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.core.device_inventory import state_array_shapes
from gpuwm.core.grid import make_base_state, make_vertical_coord
from gpuwm.core.state import DomainState
from gpuwm.ensemble.batch_diagnostics import (
    diagnostic_pointer_fields, diagnostics_kernel_spec, diagnostics_specs,
    prepare_update_diagnostics,
)
from gpuwm.ensemble.batch_state import (
    BatchStateUnsupported, BatchedDomainState, PreparedHostMember, SHARED_STATE_CANDIDATES,
)
from gpuwm.wrf_exact import DIAGNOSTICS_ENABLED

pytestmark = [pytest.mark.gpu, requires_gpu]


def _inputs(count, *, hypso=1, hybrid=0, terrain=False, mapped=False, p_top_none=False):
    cfg = RunConfig(nx=9, ny=7, nz=8, dx=1000.0, dy=1000.0, ztop=6400.0,
                    dt=0.5, run_seconds=1.0, terrain_opt=int(terrain),
                    hybrid_opt=hybrid, hypsometric_opt=hypso)
    coord = make_vertical_coord(cfg.nz, hybrid_opt=hybrid, etac=cfg.etac)
    rows, cols = np.indices((cfg.ny, cfg.nx))
    topo = (50.0 + 20.0 * np.sin(rows) + 4.0 * np.cos(cols)) if terrain else None
    # Strict WRF diagnostics consume canonical theta-minus-300. Its existing
    # source deliberately reads 300+thp, rather than the ordinary thb+thp.
    sounding = (lambda z: np.full_like(z, 300.0)) if DIAGNOSTICS_ENABLED else (
        lambda z: 300.0 + 0.003 * np.asarray(z, float))
    base = make_base_state(coord, sounding, cfg.p_surf, cfg.ztop, terrain_z=topo)
    rng = np.random.default_rng(417)
    extras = diagnostics_specs(cfg)
    shapes = state_array_shapes(cfg)
    inputs = []
    for member in range(count):
        state = DomainState(cfg, array_module=np)
        state.load_base(coord, base)
        state.thp[...] = rng.normal(0, 0.3, shapes["thp"]).astype(np.float32)
        state.php[...] = rng.normal(0, 0.1, shapes["php"]).astype(np.float32)
        state.mup[...] = rng.normal(0, 20, shapes["mup"]).astype(np.float32)
        if mapped:
            state.set_map_coriolis(msft=1.02 + 0.0001 * rows + 0.0002 * cols,
                                   msfu=np.full((cfg.ny, cfg.nx + 1), 1.03),
                                   msfv=np.full((cfg.ny + 1, cfg.nx), 1.04))
        if p_top_none:
            state.p_top = None
        arrays = {name: getattr(state, name).copy() for name in shapes}
        for index, name in enumerate(("p", "al", "alt")):
            arrays[name].fill(np.float32(-123.75 - index - 0.01 * member))
        for spec in extras:
            arrays[spec.name] = np.full(spec.shape, np.float32(-130.5 - member), dtype=spec.dtype)
        scalars = {name: value for name, value in vars(state).items()
                   if name not in shapes and name not in {
                       "physics", "lateral_boundaries", "_scratch", "_scratch_arena",
                       "_host_setup_state", "_phb_host"}}
        clock = {"ticks": 0, "step_ticks": 1, "tick_den": 2, "run_ticks": 2,
                 "step_count": 0, "dt_fp32": np.float32(0.5), "dtbc_fp32": np.float32(0)}
        inputs.append(PreparedHostMember(cfg, arrays, scalars, clock,
                                        phb_host=state._phb_host.copy()))
    return tuple(inputs)


def _batch(inputs, *, share):
    import cupy as cp
    shared = tuple(sorted(SHARED_STATE_CANDIDATES & inputs[0].arrays.keys())) if share else ()
    return BatchedDomainState.from_prepared(
        inputs, array_module=cp, available_bytes=2**30, shared_fields=shared,
        extra_specs=diagnostics_specs(inputs[0].cfg))


def _scalar(member):
    import cupy as cp
    state = DomainState(member.cfg)
    for name in state_array_shapes(member.cfg):
        getattr(state, name).set(member.arrays[name])
    # Explicitly declared optional scalar output. The reference helper cannot
    # allocate it lazily, and no batch implementation supplies reference words.
    for spec in diagnostics_specs(member.cfg):
        value = cp.zeros(spec.shape, dtype=spec.dtype)
        value.set(member.arrays[spec.name])
        setattr(state, spec.name, value)
    for name, value in member.scalars.items():
        setattr(state, name, value)
    state._phb_host = member.phb_host.copy()
    cp.cuda.get_current_stream().synchronize()
    return state


def _outputs():
    return ("p", "al", "alt", "p_perturbation") if DIAGNOSTICS_ENABLED else ("p", "al", "alt")


def _prove(inputs, *, share, window=None):
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    batch = _batch(inputs, share=share)
    before = {name: cp.asnumpy(array).view(np.uint32).tobytes()
              for name, array in batch.storage.arrays.items()}
    pool = cp.get_default_memory_pool()
    live_before = pool.used_bytes()
    launch = prepare_update_diagnostics(batch, inputs[0].cfg.hypsometric_opt, window)
    launch()
    cp.cuda.get_current_stream().synchronize()
    assert pool.used_bytes() == live_before, "EOS binding/launch allocated unplanned member fields"
    observed = {}
    # Scalar iteration is an independent test oracle, not an executor path.
    for index, member in enumerate(inputs):
        reference = _scalar(member)
        update_diagnostics(reference, member.cfg.hypsometric_opt, window)
        cp.cuda.get_current_stream().synchronize()
        for name in _outputs():
            got = cp.asnumpy(batch.member_view(name, index))
            expected = cp.asnumpy(getattr(reference, name))
            assert got.view(np.uint32).tobytes() == expected.view(np.uint32).tobytes(), (name, index)
            assert np.isfinite(got).all(), (name, index)
            observed.setdefault(name, []).append(got)
    if len(inputs) > 1:
        assert observed["p"][0].tobytes() != observed["p"][1].tobytes(), "distinct members were silently broadcast"
    for name, original in before.items():
        if name not in _outputs():
            assert cp.asnumpy(batch.storage.arrays[name]).view(np.uint32).tobytes() == original, name
    assert launch.numerical_entries == ("calc_p_alpha",)
    if len(inputs) > 1:
        assert launch.binding_receipt["members"] == len(inputs)
        assert launch.binding_receipt["block"] == (256,)


@pytest.mark.parametrize("members", [1, 4, 10, 20, 40])
@pytest.mark.parametrize("hypso", [1, 2])
@pytest.mark.parametrize("hybrid", [0, 2])
@pytest.mark.parametrize("terrain,mapped", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("share", [False, True])
@pytest.mark.parametrize("window", [None, (1, 2, 4, 5)])
def test_initialized_dry_eos_words_equal_original_helper(members, hypso, hybrid, terrain, mapped, share, window):
    _prove(_inputs(members, hypso=hypso, hybrid=hybrid, terrain=terrain, mapped=mapped),
           share=share, window=window)


@pytest.mark.parametrize("members", [1, 4, 10, 20, 40])
def test_hypsometric_one_keeps_original_absent_top_pressure_behavior(members):
    _prove(_inputs(members, hypso=1, terrain=True, p_top_none=True), share=True)


def test_bound_eos_observes_updated_member_inputs_without_reallocation():
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    inputs = _inputs(4, hypso=2, hybrid=2, terrain=True, mapped=True)
    batch = _batch(inputs, share=True)
    launch = prepare_update_diagnostics(batch, 2)
    launch()
    batch.thp[...] += np.float32(0.02)
    launch()
    for index, member in enumerate(inputs):
        reference = _scalar(member)
        reference.thp[...] += np.float32(0.02)
        update_diagnostics(reference, 2)
        for name in _outputs():
            assert cp.asnumpy(batch.member_view(name, index)).view(np.uint32).tobytes() == (
                cp.asnumpy(getattr(reference, name)).view(np.uint32).tobytes()), (name, index)


def test_invalid_option_window_and_top_pressure_refuse_before_launch():
    inputs = _inputs(4)
    batch = _batch(inputs, share=True)
    with pytest.raises(ValueError, match="hypsometric_opt must"):
        prepare_update_diagnostics(batch, 3)
    with pytest.raises(ValueError, match="does not fit"):
        prepare_update_diagnostics(batch, 1, (6, 8, 2, 2))
    no_top = _batch(_inputs(4, p_top_none=True), share=True)
    with pytest.raises(RuntimeError, match="needs state.p_top"):
        prepare_update_diagnostics(no_top, 2)
    batch.physics = object()
    physics = batch.physics
    prepare_update_diagnostics(batch, 1)()
    assert batch.physics is physics


def test_diagnostic_pressure_inventory_is_explicit_and_member_owned():
    cfg = _inputs(1)[0].cfg
    assert diagnostics_specs(cfg, strict=False) == ()
    (spec,) = diagnostics_specs(cfg, strict=True)
    assert spec.name == "p_perturbation"
    assert spec.shape == state_array_shapes(cfg)["p"]
    assert spec.allocation_shape(40) == (40,) + state_array_shapes(cfg)["p"]
    assert spec.ownership == "member"
    with pytest.raises(TypeError, match="must be boolean"):
        diagnostics_specs(cfg, strict=1)
    # Explicitly omit the optional strict carrier to grade the named refusal.
    inputs = _inputs(4)
    inputs = tuple(replace(member, arrays={name: array for name, array in member.arrays.items()
                                          if name != "p_perturbation"}) for member in inputs)
    import cupy as cp
    missing = BatchedDomainState.from_prepared(inputs, array_module=cp, available_bytes=2**30)
    with pytest.raises(BatchStateUnsupported, match="planned member allocation.*p_perturbation"):
        diagnostics_kernel_spec(missing, strict=True)


def test_actual_scalar_n1_delegates_without_replacing_state_fields():
    import cupy as cp
    member = _inputs(1, hypso=2, terrain=True)[0]
    scalar = _scalar(member)
    expected = _scalar(member)
    pointers = {name: getattr(scalar, name).data.ptr for name in _outputs()}
    prepare_update_diagnostics(scalar, 2)()
    from gpuwm.core.diagnostics import update_diagnostics
    update_diagnostics(expected, 2)
    for name in _outputs():
        assert getattr(scalar, name).data.ptr == pointers[name]
        assert cp.asnumpy(getattr(scalar, name)).view(np.uint32).tobytes() == (
            cp.asnumpy(getattr(expected, name)).view(np.uint32).tobytes())
