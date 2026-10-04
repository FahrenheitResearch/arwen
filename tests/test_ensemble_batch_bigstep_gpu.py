"""Dry raw big-step primitives against unchanged scalar helpers, every word.

This covers initialized inputs and individual numerical entries. It does not
advance a complete RK step or certify trajectories, output history or physics.
"""
from dataclasses import replace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.core.device_inventory import state_array_shapes
from gpuwm.core.grid import make_base_state, make_vertical_coord
from gpuwm.core.state import DomainState, mu_at_u_faces, mu_at_v_faces
from gpuwm.ensemble.batch_bigstep import (
    POINTER_FIELDS, kernel_spec_for, prepare_coriolis_curvature,
    prepare_slow_buoyancy, prepare_slow_geopotential, prepare_slow_pgf,
    prepare_small_step_finish, prepare_small_step_init, prepare_w_surface,
)
from gpuwm.ensemble.batch_state import (
    BatchStateUnsupported, BatchedDomainState, PreparedHostMember, SHARED_STATE_CANDIDATES,
)
from gpuwm.ensemble.batch_diagnostics import diagnostics_specs
from gpuwm.wrf_exact import DIAGNOSTICS_ENABLED

pytestmark = [pytest.mark.gpu, requires_gpu]

_OUTPUTS = {
    "pgf": ("ru_t", "rv_t"), "buoyancy": ("rw_t",), "geopotential": ("rph_t",),
    "init": ("u_pp", "v_pp", "w_pp", "th_pp", "ph_pp", "mu_pp", "al_pp", "p_pp", "p_pp_old"),
    "finish": ("u", "v", "w", "thp", "php", "mup"), "surface": ("w",),
    "coriolis": ("ru_t", "rv_t", "rw_t"),
}


def _inputs(count, *, terrain=False, mapped=False, order=2, specified=False, open_x=False,
            strict_device_init=True):
    """Use existing base/EOS setup helpers, with distinct moderate member winds."""
    from gpuwm.core.diagnostics import update_diagnostics
    cfg = RunConfig(nx=9, ny=8, nz=5, dx=3000.0, dy=3000.0, ztop=10000.0,
                    dt=3.0, run_seconds=30.0, terrain_opt=int(terrain),
                    km_opt=4, diff_opt=1, h_sca_adv_order=order,
                    specified=specified, open_x=open_x)
    coord = make_vertical_coord(cfg.nz)
    row, col = np.indices((cfg.ny, cfg.nx))
    topo = (100.0 + 30.0 * np.sin(col) * np.cos(row)) if terrain else None
    base = make_base_state(coord, lambda z: np.full_like(z, 300.0),
                           cfg.p_surf, cfg.ztop, terrain_z=topo)
    inputs = []
    shapes = state_array_shapes(cfg)
    extras = diagnostics_specs(cfg)
    for member in range(count):
        state = DomainState(cfg, array_module=np)
        state.load_base(coord, base)
        for name, origin in (("u", 3.0), ("v", -2.0), ("w", 0.03),
                             ("thp", 0.2), ("php", 0.5), ("mup", 2.0)):
            target = getattr(state, name)
            sample = np.arange(target.size).reshape(target.shape)
            target[...] = origin + 0.02 * member + 0.005 * np.sin(sample + member)
            getattr(state, name + "0")[...] = target + np.float32(0.004)
        if mapped:
            state.set_map_coriolis(
                msft=1.01 + 0.0005 * col + 0.0002 * row,
                msfu=1.01 + 0.0005 * np.indices((cfg.ny, cfg.nx + 1))[1],
                msfv=1.01 + 0.0002 * np.indices((cfg.ny + 1, cfg.nx))[0],
                f=np.full((cfg.ny, cfg.nx), 8e-5), e=np.full((cfg.ny, cfg.nx), 1e-4),
                sina=np.full((cfg.ny, cfg.nx), 0.15), cosa=np.full((cfg.ny, cfg.nx), 0.988686))
        update_diagnostics(state, cfg.hypsometric_opt)
        mass = state.mub2d + state.mup
        ux, vy = mu_at_u_faces(mass), mu_at_v_faces(mass)
        ru = (state.c1h[:, None, None] * ux + state.c2h[:, None, None]) * state.u / state.msfu
        rv = (state.c1h[:, None, None] * vy + state.c2h[:, None, None]) * state.v / state.msfv
        ww = np.full(shapes["w"], np.float32(0.02 + 0.001 * member))
        ww[0] = ww[-1] = 0
        scratch = {"rk_ru": ru.astype(np.float32), "rk_rv": rv.astype(np.float32),
                   "rk_ww": ww, "smag_mut": mass.astype(np.float32)}
        for name in ("u_pp", "v_pp", "w_pp", "th_pp", "ph_pp", "mu_pp"):
            getattr(state, name)[...] = np.float32(0.01 + 0.0003 * member)
        scalars = {name: value for name, value in vars(state).items()
                   if name not in shapes and name not in {
                       "physics", "lateral_boundaries", "_scratch", "_scratch_arena",
                       "_host_setup_state", "_phb_host", "p_perturbation"}}
        clock = {"ticks": 0, "step_ticks": 3, "tick_den": 1, "run_ticks": 30,
                 "step_count": 0, "dt_fp32": np.float32(3), "dtbc_fp32": np.float32(0)}
        arrays = {name: getattr(state, name).copy() for name in shapes}
        for spec in extras:
            arrays[spec.name] = np.zeros(spec.shape, dtype=spec.dtype)
        if extras and strict_device_init:
            # Strict pressure is an independently produced carrier, not a
            # subtraction from the already rounded full-pressure output.
            import cupy as cp
            original = DomainState(cfg)
            for name in shapes:
                getattr(original, name).set(arrays[name])
            for name, value in scalars.items():
                setattr(original, name, value)
            for spec in extras:
                setattr(original, spec.name, cp.zeros(spec.shape, dtype=spec.dtype))
            update_diagnostics(original, cfg.hypsometric_opt)
            for name in ("p", "al", "alt", "p_perturbation"):
                arrays[name] = cp.asnumpy(getattr(original, name))
        inputs.append(PreparedHostMember(cfg, arrays,
                                        scalars, clock, scratch=scratch,
                                        phb_host=state._phb_host.copy()))
    return tuple(inputs)


def _scalar_state(member):
    """Independent original-shaped reference, constructed outside batch code."""
    import cupy as cp
    state = DomainState(member.cfg)
    for name in state_array_shapes(member.cfg):
        getattr(state, name).set(member.arrays[name])
    for spec in diagnostics_specs(member.cfg):
        value = cp.zeros(spec.shape, dtype=spec.dtype)
        value.set(member.arrays[spec.name])
        setattr(state, spec.name, value)
    for name, value in member.scalars.items():
        setattr(state, name, value)
    for slot, value in member.scratch.items():
        state.scratch(value.shape, slot, dtype=value.dtype).set(value)
    state._phb_host = member.phb_host.copy()
    cp.cuda.get_current_stream().synchronize()
    return state


def _batch_launch(primitive, batch, **kwargs):
    return {
        "pgf": prepare_slow_pgf, "buoyancy": prepare_slow_buoyancy,
        "geopotential": prepare_slow_geopotential, "init": prepare_small_step_init,
        "finish": prepare_small_step_finish, "surface": prepare_w_surface,
        "coriolis": prepare_coriolis_curvature,
    }[primitive](batch, **kwargs)


def _scalar_launch(primitive, state, cfg, **kwargs):
    from gpuwm.core import dycore
    if primitive == "pgf":
        dycore._launch_slow_pgf(state, cfg)
    elif primitive == "buoyancy":
        dycore._launch_slow_buoyancy(state, cfg)
    elif primitive == "geopotential":
        dycore._launch_slow_geopotential(state, cfg, state.existing_scratch("rk_ww"),
                                       add_vertical=kwargs.get("add_vertical", True))
    elif primitive == "init":
        dycore._prepare_small_step_init_launch(state, cfg, kwargs.get("rk_step", 1))()
    elif primitive == "finish":
        dycore._prepare_small_step_finish_launch(state, cfg)()
    elif primitive == "surface":
        dycore.set_w_surface(state, cfg)
    else:
        dycore.launch_coriolis_curvature(
            state.existing_scratch("rk_ru"), state.existing_scratch("rk_rv"),
            state.u, state.v, state.w, state.existing_scratch("smag_mut"),
            state.msft, state.msfu, state.msfv, state.f, state.e,
            state.c1f, state.c2f, state.fnm, state.fnp, cfg.dx, cfg.dy,
            state.ru_t, state.rv_t, state.rw_t, sina=state.sina, cosa=state.cosa,
            boundary_x=dycore._boundary_x(cfg), boundary_y=dycore._boundary_y(cfg))


def _prove(primitive, inputs, *, share, **kwargs):
    import cupy as cp
    shared = tuple(sorted(SHARED_STATE_CANDIDATES & inputs[0].arrays.keys())) if share else ()
    batch = BatchedDomainState.from_prepared(inputs, array_module=cp,
                                             available_bytes=2**30, shared_fields=shared,
                                             extra_specs=diagnostics_specs(inputs[0].cfg))
    before = {name: cp.asnumpy(array).view(np.uint32).tobytes()
              for name, array in batch.storage.arrays.items()}
    launch = _batch_launch(primitive, batch, **kwargs)
    launch()
    cp.cuda.get_current_stream().synchronize()
    # Scalar reference iteration is only a test oracle, never a batch executor.
    for index, member in enumerate(inputs):
        reference = _scalar_state(member)
        _scalar_launch(primitive, reference, member.cfg, **kwargs)
        cp.cuda.get_current_stream().synchronize()
        for name in _OUTPUTS[primitive]:
            got = cp.asnumpy(batch.member_view(name, index)).view(np.uint32)
            expected = cp.asnumpy(getattr(reference, name)).view(np.uint32)
            assert got.tobytes() == expected.tobytes(), (primitive, name, index)
            assert np.isfinite(got.view(np.float32)).all(), (primitive, name, index)
    for name, original in before.items():
        if name not in _OUTPUTS[primitive]:
            assert cp.asnumpy(batch.storage.arrays[name]).view(np.uint32).tobytes() == original, (primitive, name)
    assert len(launch.numerical_entries) == (2 if primitive in ("init", "finish") else 1)


@pytest.mark.parametrize("members", [1, 4, 10, 20, 40])
@pytest.mark.parametrize("terrain,mapped", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("share", [False, True])
@pytest.mark.parametrize("primitive", list(_OUTPUTS))
def test_dry_bigstep_member_words_equal_original_helpers(members, terrain, mapped, share, primitive):
    inputs = _inputs(members, terrain=terrain, mapped=mapped)
    kwargs = {"rk_step": 2} if primitive == "init" else {}
    _prove(primitive, inputs, share=share, **kwargs)


@pytest.mark.parametrize("members", [1, 4])
@pytest.mark.parametrize("rk_step", [1, 2, 3])
def test_all_small_step_init_stage_arguments_preserve_original_words(members, rk_step):
    _prove("init", _inputs(members, terrain=True, mapped=True), share=True, rk_step=rk_step)


@pytest.mark.parametrize("members", [1, 4])
@pytest.mark.parametrize("order,specified,open_x", [(2, False, True), (2, True, False), (5, True, False), (5, False, False)])
@pytest.mark.parametrize("add_vertical", [False, True])
def test_geopotential_order_boundary_and_vertical_switches(members, order, specified, open_x, add_vertical):
    _prove("geopotential", _inputs(members, terrain=True, mapped=True, order=order,
                                  specified=specified, open_x=open_x),
           share=True, add_vertical=add_vertical)


def test_pointer_roles_and_missing_owners_are_explicit():
    import cupy as cp
    inputs = _inputs(4)
    batch = BatchedDomainState.from_prepared(inputs, array_module=cp, available_bytes=2**30,
                                             shared_fields=("pb", "phb", "c1h", "msft"),
                                             extra_specs=diagnostics_specs(inputs[0].cfg))
    for entry in POINTER_FIELDS:
        spec = kernel_spec_for(batch, entry)
        for pointer in spec.pointers:
            name = dict(POINTER_FIELDS[entry])[pointer.name]
            assert pointer.role == batch.storage.specs[name].ownership
    with pytest.raises(BatchStateUnsupported, match="planned allocation"):
        kernel_spec_for(batch, "coriolis_curvature", bindings={"mut": "missing"})
    with pytest.raises(BatchStateUnsupported, match="admitted member h_diabatic"):
        prepare_small_step_finish(batch, hdiab_dt=3.0)
    with pytest.raises(BatchStateUnsupported, match="configuration differs"):
        prepare_small_step_init(batch, replace(batch.cfg, dt=4.0))
    batch.physics = object()
    physics = batch.physics
    prepare_small_step_init(batch)()
    assert batch.physics is physics


def test_actual_scalar_state_n1_uses_original_factory():
    member = _inputs(1, terrain=True, mapped=True)[0]
    state = _scalar_state(member)
    before = state.u.data.ptr
    prepare_small_step_init(state, member.cfg, rk_step=2)()
    assert state.u.data.ptr == before
    assert state.u.shape == member.arrays["u"].shape


def test_pressure_binding_uses_the_original_produced_carrier():
    import cupy as cp
    inputs = _inputs(4, terrain=True, mapped=True)
    batch = BatchedDomainState.from_prepared(inputs, array_module=cp, available_bytes=2**30,
                                             extra_specs=diagnostics_specs(inputs[0].cfg))
    pressure = batch.p_perturbation if DIAGNOSTICS_ENABLED else batch.p
    pgf = prepare_slow_pgf(batch)
    buoyancy = prepare_slow_buoyancy(batch)
    assert pgf.binding_receipt["arrays"][2]["pointer"] == pressure.data.ptr
    assert buoyancy.binding_receipt["arrays"][1]["pointer"] == pressure.data.ptr
    if DIAGNOSTICS_ENABLED:
        assert pressure.data.ptr != batch.p.data.ptr
        missing = tuple(replace(member, arrays={name: value for name, value in member.arrays.items()
                                                if name != "p_perturbation"}) for member in inputs)
        missing = BatchedDomainState.from_prepared(missing, array_module=cp, available_bytes=2**30)
        with pytest.raises(BatchStateUnsupported, match="planned p_perturbation"):
            prepare_slow_pgf(missing)
        with pytest.raises(BatchStateUnsupported, match="planned p_perturbation"):
            prepare_slow_buoyancy(missing)
