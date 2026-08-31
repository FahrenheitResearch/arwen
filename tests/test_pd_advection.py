# tests/test_pd_advection.py
"""Positive-definite scalar transport (Phase 2 Task 5).

Kernel-vs-mirror tests for the two pd_advection.cu kernels (flux
decomposition F = F_upwind1 + F_corr, then per-cell renormalization of
outgoing correction fluxes -- WRF advect_scalar_pd / Skamarock 2006), plus
the plan's acceptance gates on a strongly deformational 2-D flow:
min(q) >= 0 exactly, total coupled scalar mass conserved to <= 1e-6
relative, and PD-vs-unlimited < 5% RMS for a smooth blob.
"""
import numpy as np
import pytest
from conftest import requires_gpu


def _pd_case(seed=7, nz=12, ny=4, nx=24):
    """Random PD-advection inputs on a stretched hybrid column, float32.

    ``q`` is the (any-sign) RK stage estimate feeding the high-order
    fluxes; ``q0 >= 0`` is the time-t field feeding the upwind fluxes,
    with a sprinkling of exact zeros so the renormalization engages.
    Flux/velocity amplitudes put the face Courant numbers at O(0.2-0.7),
    exercising the WRF flux_upwind CFL clamp.
    """
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    vc = make_vertical_coord(nz, stretch=1.5, hybrid_opt=2, etac=0.2)
    b = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                        p_surf=1.0e5, ztop=12000.0)
    rng = np.random.default_rng(seed)
    q = rng.normal(0.01, 0.004, (nz, ny, nx)).astype(np.float32)
    q0 = np.clip(rng.normal(0.008, 0.008, (nz, ny, nx)), 0.0, None)
    q0[rng.random(q0.shape) < 0.25] = 0.0
    q0 = q0.astype(np.float32)
    mub = float(b.mub)
    mut = (mub * (1.0 + 0.02 * rng.standard_normal((ny, nx)))
           ).astype(np.float32)
    mu_old = (mub * (1.0 + 0.02 * rng.standard_normal((ny, nx)))
              ).astype(np.float32)
    u = rng.normal(0, 6.0, (nz, ny, nx + 1))
    u[..., -1] = u[..., 0]
    ru = (mub * u).astype(np.float32)
    v = rng.normal(0, 6.0, (nz, ny + 1, nx))
    v[:, -1] = v[:, 0]
    rv = (mub * v).astype(np.float32)
    rw = (mub * rng.normal(0, 3e-3, (nz + 1, ny, nx))).astype(np.float32)
    rw[0] = rw[-1] = 0.0
    return vc, q, q0, ru, rv, rw, mut, mu_old, 500.0, 500.0, 20.0


def _assert_close(got, ref, scale, rtol=5e-4):
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=rtol * scale)


def test_pd_mirror_reduces_to_unlimited_when_inactive():
    # Float64 mirrors only: with q0 == q well away from zero and small
    # Courant numbers the limiter never engages and the recombined
    # tendency -div(F_upwind1 + F_corr) equals the unlimited 5th/3rd-order
    # flux divergence to float64 round-off.
    from gpuwm.core.grid import make_vertical_coord
    from gpuwm.verify.npref import (np_flux_div_scalar, np_pd_fluxes,
                                    np_pd_renorm_apply)
    nz, ny, nx = 10, 3, 16
    vc = make_vertical_coord(nz)
    rng = np.random.default_rng(3)
    q = 1.5 + 0.3 * rng.standard_normal((nz, ny, nx))
    ru = rng.normal(0, 1.0, (nz, ny, nx + 1))
    ru[..., -1] = ru[..., 0]
    rv = rng.normal(0, 1.0, (nz, ny + 1, nx))
    rv[:, -1] = rv[:, 0]
    rw = rng.normal(0, 0.02, (nz + 1, ny, nx))
    rw[0] = rw[-1] = 0.0
    dx = dy = 1.0
    dt = 0.01
    fl = np_pd_fluxes(q, q, ru, rv, rw, 1.0, vc, dx, dy, dt)
    tend_pd = np_pd_renorm_apply(q, 1.0, *fl, coord=vc, dx=dx, dy=dy, dt=dt)
    tend_ref = np_flux_div_scalar(q, ru, rv, rw, vc, dx, dy)
    np.testing.assert_allclose(tend_pd, tend_ref, rtol=1e-12,
                               atol=1e-12 * np.abs(tend_ref).max())


def test_nested_pd_lbc_is_folded_once_before_advection(monkeypatch):
    """WRF folds ``sc_tend`` into q0, then zeroes it before advection.

    A nested final-stage boundary tendency must therefore enter once even
    with a non-unit map factor.  Retaining it in the advection accumulator
    produces ``B + msft*B`` in addition to the time-t scalar.
    """
    from types import SimpleNamespace

    import gpuwm.core.moist as moist
    import gpuwm.ingest.lateral_bc as lateral_bc

    shape = (1, 3, 4)

    class FakeState:
        def __init__(self):
            self.p = np.zeros(shape, dtype=np.float32)
            self.c1h = np.ones(1, dtype=np.float32)
            self.c2h = np.zeros(1, dtype=np.float32)
            self.mub2d = np.ones(shape[1:], dtype=np.float32)
            self.mup0 = np.zeros(shape[1:], dtype=np.float32)
            self.mup = np.zeros(shape[1:], dtype=np.float32)
            self.msft = np.full(shape[1:], 2.0, dtype=np.float32)
            self.has_msf = True
            self.lateral_boundaries = object()
            self.qi = None
            self._scratch = {}
            for name in moist.SPECIES:
                setattr(self, name, np.zeros(shape, dtype=np.float32))
                setattr(self, name + "0", np.zeros(shape, dtype=np.float32))

        def scratch(self, requested_shape, slot, dtype=None):
            requested_shape = tuple(requested_shape)
            value = self._scratch.get(slot)
            if value is None:
                value = np.zeros(requested_shape, dtype=np.float32)
                self._scratch[slot] = value
            assert value.shape == requested_shape
            return value

    boundary_rate = np.float32(0.125)

    def install_boundary(_state, _cfg, name, tendency, **_kwargs):
        tendency[...] = boundary_rate if name == "qv" else np.float32(0.0)

    monkeypatch.setattr(moist, "cp", np)
    monkeypatch.setattr(lateral_bc, "apply_state_scalar_lateral_boundary",
                        install_boundary)
    monkeypatch.setattr(moist, "launch_pd_fluxes",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(moist, "launch_pd_renorm_apply",
                        lambda *_args, **_kwargs: None)

    # The fused update kernel is lru_cached, so with cp mocked to numpy
    # this test was green only when an earlier same-process test had
    # already built the exact (msft, physics, fixed, clamp) key on the
    # real device -- a pass that depended on shard ORDER, and the 2.6.0
    # shard ordering went red on the cold key.  The mirror below is the
    # kernel's own documented ufunc chain (moist.py:_update_scalar_kernel:
    # tend*msft, +physics, +fixed, then chm0*q0 + dt*t over chm, clamped),
    # so the fold-count claim this test makes stays measured against real
    # arithmetic rather than against a warm cache.
    def numpy_update_kernel(has_msf, has_physics, has_fixed, clamp):
        def kern(*args):
            q0, tend, c1h, c2h, mu0, mu = args[:6]
            rest = list(args[6:])
            msft = rest.pop(0) if has_msf else None
            physics = rest.pop(0) if has_physics else None
            fixed = rest.pop(0) if has_fixed else None
            dt_eff, _ncol, q = rest
            # The kernel takes the column fields FLAT (ny*nx,) and derives
            # the column ordinal from its linear index; the mirror restores
            # the (ny, nx) plane from the field shape instead.
            ny, nx = q0.shape[1], q0.shape[2]
            plane = lambda a: np.asarray(a, dtype=np.float32).reshape(ny, nx)
            t = np.array(tend, dtype=np.float32, copy=True)
            if msft is not None:
                t *= plane(msft)[None]
            if physics is not None:
                t += physics
            if fixed is not None:
                t += fixed
            chm0 = (c1h[:, None, None] * plane(mu0)[None]
                    + c2h[:, None, None]).astype(np.float32)
            chm = (c1h[:, None, None] * plane(mu)[None]
                   + c2h[:, None, None]).astype(np.float32)
            v = (chm0 * q0 + np.float32(dt_eff) * t) / chm
            if clamp:
                v = np.where(np.isnan(v), np.float32("nan"),
                             np.maximum(v, np.float32(0.0)))
            q[...] = v
        return kern

    monkeypatch.setattr(moist, "_update_scalar_kernel", numpy_update_kernel)

    state = FakeState()
    cfg = SimpleNamespace(
        nested=True, specified=False, open_x=False, open_y=False,
        moist_adv_opt=1, dx=1.0, dy=1.0, spec_zone=1)
    dt = 2.0
    moist.advance_scalars_stage(
        state, cfg, np.zeros((1, 3, 5), dtype=np.float32),
        np.zeros((1, 4, 4), dtype=np.float32),
        np.zeros((2, 3, 4), dtype=np.float32),
        dt_eff=dt, final=True)

    np.testing.assert_array_equal(
        state.qv, np.full(shape, dt * boundary_rate, dtype=np.float32))


@pytest.mark.gpu
@requires_gpu
def test_pd_fluxes_match_reference():
    import cupy as cp
    from gpuwm.core.moist import launch_pd_fluxes
    from gpuwm.verify.npref import np_pd_fluxes
    vc, q, q0, ru, rv, rw, mut, _, dx, dy, dt = _pd_case()
    nz, ny, nx = q.shape
    f64 = lambda a: a.astype(np.float64)
    ref = np_pd_fluxes(f64(q), f64(q0), f64(ru), f64(rv), f64(rw),
                       f64(mut), vc, dx, dy, dt)
    bufs = (cp.zeros((nz, ny, nx + 1), cp.float32),
            cp.zeros((nz, ny, nx + 1), cp.float32),
            cp.zeros((nz, ny + 1, nx), cp.float32),
            cp.zeros((nz, ny + 1, nx), cp.float32),
            cp.zeros((nz + 1, ny, nx), cp.float32),
            cp.zeros((nz + 1, ny, nx), cp.float32))
    launch_pd_fluxes(cp.asarray(q), cp.asarray(q0), cp.asarray(ru),
                     cp.asarray(rv), cp.asarray(rw), cp.asarray(mut),
                     vc, dx, dy, dt, *bufs)
    for got, want, name in zip(bufs, ref,
                               ("fxl", "fxc", "fyl", "fyc", "fzl", "fzc")):
        pair = ref[0] if name.startswith("fx") else (
            ref[2] if name.startswith("fy") else ref[4])
        scale = max(np.abs(pair).max(), np.abs(want).max())
        _assert_close(cp.asnumpy(got), want, scale)
    # boundary eta faces carry exactly zero flux
    assert not cp.asnumpy(bufs[4])[[0, -1]].any()
    assert not cp.asnumpy(bufs[5])[[0, -1]].any()


@pytest.mark.gpu
@requires_gpu
def test_pd_renorm_apply_matches_reference():
    import cupy as cp
    from gpuwm.core.moist import launch_pd_renorm_apply
    from gpuwm.verify.npref import np_pd_fluxes, np_pd_renorm_apply
    vc, q, q0, ru, rv, rw, mut, mu_old, dx, dy, dt = _pd_case(seed=11)
    nz, ny, nx = q.shape
    f64 = lambda a: a.astype(np.float64)
    # fluxes from the float64 mirror, cast to fp32 so kernel and mirror
    # consume bitwise-identical inputs (isolates the renormalize+apply).
    fl32 = [a.astype(np.float32)
            for a in np_pd_fluxes(f64(q), f64(q0), f64(ru), f64(rv),
                                  f64(rw), f64(mut), vc, dx, dy, dt)]
    ref = np_pd_renorm_apply(f64(q0), f64(mu_old), *[f64(a) for a in fl32],
                             coord=vc, dx=dx, dy=dy, dt=dt)
    tend = cp.zeros((nz, ny, nx), cp.float32)
    launch_pd_renorm_apply(cp.asarray(q0), cp.asarray(mu_old),
                           *[cp.asarray(a) for a in fl32],
                           tend=tend, coord=vc, dx=dx, dy=dy, dt=dt)
    _assert_close(cp.asnumpy(tend), ref, np.abs(ref).max(), rtol=1e-3)


def _swirl_fluxes(nx, ny):
    """Discretely divergence-free single-cell swirl (LeVeque) on [0,1]^2.

    Coupled fluxes from corner-point streamfunction differences:
    psi = sin^2(pi x) sin^2(pi y) / pi, ru = d(psi)/dy, rv = -d(psi)/dx.
    max |u| ~ 1; psi = 0 on the domain edge so nothing crosses it.
    """
    dx, dy = 1.0 / nx, 1.0 / ny
    xc = np.arange(nx + 1) * dx
    yc = np.arange(ny + 1) * dy
    psi = (np.sin(np.pi * yc)[:, None] ** 2
           * np.sin(np.pi * xc)[None, :] ** 2) / np.pi   # corners (ny+1,nx+1)
    ru = ((psi[1:, :] - psi[:-1, :]) / dy)[None].astype(np.float32)
    rv = (-(psi[:, 1:] - psi[:, :-1]) / dx)[None].astype(np.float32)
    return ru, rv, dx, dy


@pytest.mark.gpu
@requires_gpu
def test_deformational_flow_pd_gates():
    """Plan acceptance: compact blob through strong deformation, 1000 steps.

    PD: min(q) >= 0 exactly and coupled mass conserved to <= 1e-6 relative
    (FP32 fields, FP64 sums); the unlimited control DOES go negative (the
    gate has teeth); for a smooth blob PD differs from unlimited by < 5%
    RMS (limiter inactive where unneeded).
    """
    import cupy as cp
    from gpuwm.core.advection import launch_flux_div_scalar
    from gpuwm.core.grid import make_vertical_coord
    from gpuwm.core.moist import launch_pd_fluxes, launch_pd_renorm_apply
    nx = ny = 96
    nz = 1
    vc = make_vertical_coord(nz)
    ru_h, rv_h, dx, dy = _swirl_fluxes(nx, ny)
    ru = cp.asarray(ru_h)
    rv = cp.asarray(rv_h)
    rw = cp.zeros((nz + 1, ny, nx), cp.float32)
    mut = cp.ones((ny, nx), cp.float32)
    dt = 0.35 * dx                               # CFL ~ 0.35 at |u| ~ 1
    tend = cp.zeros((nz, ny, nx), cp.float32)
    bufs = (cp.zeros((nz, ny, nx + 1), cp.float32),
            cp.zeros((nz, ny, nx + 1), cp.float32),
            cp.zeros((nz, ny + 1, nx), cp.float32),
            cp.zeros((nz, ny + 1, nx), cp.float32),
            cp.zeros((nz + 1, ny, nx), cp.float32),
            cp.zeros((nz + 1, ny, nx), cp.float32))

    def rhs(qs):
        tend[...] = 0
        launch_flux_div_scalar(qs, ru, rv, rw, tend, vc, dx, dy)
        return tend

    def run(q0_host, pd, nsteps=1000):
        q = cp.asarray(q0_host, cp.float32).reshape(nz, ny, nx)
        for _ in range(nsteps):
            q1 = q + (dt / 3.0) * rhs(q)
            q2 = q + (dt / 2.0) * rhs(q1)
            if pd:
                tend[...] = 0
                launch_pd_fluxes(q2, q, ru, rv, rw, mut, vc, dx, dy, dt,
                                 *bufs)
                launch_pd_renorm_apply(q, mut, *bufs, tend=tend, coord=vc,
                                       dx=dx, dy=dy, dt=dt)
                updated = q + dt * tend
                # FP32-FLOOR: absolute tolerance covers float32 cancellation
                # at a zero/near-zero reference; the limiter must keep its
                # raw update within rounding distance of zero before the
                # production final-stage clamp is applied.
                floor = 8.0 * np.finfo(np.float32).eps * max(
                    float(cp.abs(q).max()), 1.0e-20)
                assert float(updated.min()) >= -floor
                q = cp.maximum(updated, 0.0)
            else:
                q = q + dt * rhs(q2)
        return q

    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    r = np.sqrt((x[None, :] - 0.5) ** 2 + (y[:, None] - 0.75) ** 2)

    # --- compact (sharp-edged) blob: positivity + mass gates
    compact = np.where(r < 0.15, np.cos(np.pi * r / 0.30) ** 2, 0.0)
    mass = lambda q: float(cp.asnumpy(q).astype(np.float64).sum())
    m0 = float(compact.astype(np.float64).sum())
    q_un = run(compact, pd=False)
    assert float(q_un.min()) < -1e-4            # unlimited undershoots
    q_pd = run(compact, pd=True)
    assert float(q_pd.min()) >= 0.0             # PD: exactly nonnegative
    assert abs(mass(q_pd) - m0) / m0 <= 1e-6    # mass to 1e-6 relative

    # --- smooth blob: limiter inactive where unneeded (< 5% RMS) through
    # the same plan-mandated 1000-step deformational experiment.
    smooth = np.exp(-((x[None, :] - 0.5) ** 2
                      + (y[:, None] - 0.75) ** 2) / 0.08)
    s_un = cp.asnumpy(run(smooth, pd=False)).astype(np.float64)
    s_pd = cp.asnumpy(run(smooth, pd=True)).astype(np.float64)
    rms = np.sqrt(np.mean((s_pd - s_un) ** 2))
    assert rms < 0.05 * np.sqrt(np.mean(s_un ** 2))
    assert abs(mass(cp.asarray(s_pd)) - float(smooth.astype(np.float64).sum())
               ) / float(smooth.sum()) <= 1e-6


@pytest.mark.gpu
@requires_gpu
def test_pd_limiter_sees_physics_source_before_limiting():
    """WRF folds accumulated physics tendencies into the scalar BEFORE the
    PD limiter: rk_update_scalar_pd (module_em.F:1803-1916) applies
    dt*sc_tend to moist_old with the time-t mass and ZEROES sc_tend at the
    start of the final RK step (solve_em.F:1839-1867, 'add in physics
    tendency first if positive definite advection is used'), so
    advect_scalar_pd's ph_low budget (module_advect_em.F:7733-7737) sees
    the sources.

    Construction: a qc blob is 90% depleted by a physics sink while a
    strong uniform flow (face Courant 0.5) drains its edge cells.  With
    the source folded first, the limiter renormalizes the outflow against
    the depleted budget and the coupled water mass closes exactly (the
    periodic fluxes telescope).  Pre-fix (physics added AFTER pd_renorm)
    the limiter budgeted against the undepleted q0, the update went
    negative, and the clamp manufactured mass -- this test fails there.
    """
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.core.moist import advance_scalars_stage, init_moist_balanced
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord

    nx, ny, nz = 16, 8, 12
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=500.0, dy=500.0, ztop=6000.0,
                    dt=10.0, run_seconds=0.0, moist=True)
    vc = make_vertical_coord(nz)
    b = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                        p_surf=cfg.p_surf, ztop=cfg.ztop)
    s = init_moist_balanced(cfg, vc, b, lambda z: np.full(nz, 1.0e-3))
    dt_eff = cfg.dt

    qc = np.zeros((nz, ny, nx), dtype=np.float32)
    qc[4:8, 3:6, 6:10] = 1.0e-3
    s.qc[...] = cp.asarray(qc)
    s.qc0[...] = s.qc                      # final stage: q == q0

    chm = (s.c1h[:, None, None] * s.total_mu()[None]
           + s.c2h[:, None, None])         # mup == mup0 == 0
    # uniform x flow at face Courant 0.5
    ru = cp.zeros((nz, ny, nx + 1), cp.float32)
    ru[...] = 0.5 * (cfg.dx / dt_eff) * chm[:, :, :1]
    rv = cp.zeros((nz, ny + 1, nx), cp.float32)
    ww = cp.zeros((nz + 1, ny, nx), cp.float32)

    # physics sink: remove 90% of the blob over dt (coupled tendency units)
    sink = cp.zeros((nz, ny, nx), cp.float32)
    sink[...] = -0.9 * chm * s.qc0 / cp.float32(dt_eff)
    physics = SimpleNamespace(
        scalar_for=lambda name: sink if name == "qc" else None)

    dnw_abs = -s.dnw[:, None, None]
    mass = lambda: float(cp.sum((chm * s.qc * dnw_abs).astype(cp.float64)))
    m0 = mass()
    sink_mass = float(cp.sum((sink * dnw_abs).astype(cp.float64)))
    expected = m0 + dt_eff * sink_mass

    advance_scalars_stage(s, cfg, ru, rv, ww, dt_eff, final=True,
                          physics_tendencies=physics)
    m1 = mass()
    assert float(s.qc.min()) >= 0.0
    residual = abs(m1 - expected) / m0
    assert residual < 1e-5, (
        f"limiter did not budget against the physics-depleted scalar: "
        f"mass residual {residual} relative (m0 {m0}, expected {expected}, "
        f"got {m1})")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("open_x,open_y",
                         [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("with_msf", [False, True])
def test_pd_specified_bounds_match_mirror(open_x, open_y, with_msf):
    """WRF advect_scalar_pd specified/open bounds (module_advect_em.F:
    7697-7715 limiter bounds, 7817-7856 applied-tendency bounds, plus the
    degrade bands in the flux blocks): device kernels vs the float64 npref
    mirrors with the same flags, with and without map factors (the real74
    production combination is specified + Lambert msf)."""
    import cupy as cp
    from gpuwm.core.moist import launch_pd_fluxes, launch_pd_renorm_apply
    from gpuwm.verify.npref import np_pd_fluxes, np_pd_renorm_apply
    vc, q, q0, ru, rv, rw, mut, mu_old, dx, dy, dt = _pd_case(seed=23,
                                                              ny=8, nx=16)
    nz, ny, nx = q.shape
    rng = np.random.default_rng(29)
    msft = ((0.96 + 0.05 * rng.random((ny, nx))).astype(np.float32)
            if with_msf else None)
    f64 = lambda a: a.astype(np.float64)
    m64 = None if msft is None else msft.astype(np.float64)
    fl = np_pd_fluxes(f64(q), f64(q0), f64(ru), f64(rv), f64(rw), f64(mut),
                      vc, dx, dy, dt, msft=m64, open_x=open_x, open_y=open_y)
    ref = np_pd_renorm_apply(f64(q0), f64(mu_old), *fl, coord=vc, dx=dx,
                             dy=dy, dt=dt, msft=m64,
                             open_x=open_x, open_y=open_y)
    bufs = (cp.zeros((nz, ny, nx + 1), cp.float32),
            cp.zeros((nz, ny, nx + 1), cp.float32),
            cp.zeros((nz, ny + 1, nx), cp.float32),
            cp.zeros((nz, ny + 1, nx), cp.float32),
            cp.zeros((nz + 1, ny, nx), cp.float32),
            cp.zeros((nz + 1, ny, nx), cp.float32))
    dmsft = None if msft is None else cp.asarray(msft)
    launch_pd_fluxes(cp.asarray(q), cp.asarray(q0), cp.asarray(ru),
                     cp.asarray(rv), cp.asarray(rw), cp.asarray(mut),
                     vc, dx, dy, dt, *bufs, msft=dmsft,
                     open_x=open_x, open_y=open_y)
    for got, want, name in zip(bufs, fl,
                               ("fxl", "fxc", "fyl", "fyc", "fzl", "fzc")):
        pair = fl[0] if name.startswith("fx") else (
            fl[2] if name.startswith("fy") else fl[4])
        scale = max(np.abs(pair).max(), np.abs(want).max())
        _assert_close(cp.asnumpy(got), want, scale, rtol=1e-3)
    if open_x:      # boundary-normal faces carry exactly zero flux
        assert not cp.asnumpy(bufs[0])[:, :, [0, -1]].any()
        assert not cp.asnumpy(bufs[1])[:, :, [0, -1]].any()
    if open_y:
        assert not cp.asnumpy(bufs[2])[:, [0, -1], :].any()
        assert not cp.asnumpy(bufs[3])[:, [0, -1], :].any()
    tend = cp.zeros((nz, ny, nx), cp.float32)
    launch_pd_renorm_apply(cp.asarray(q0), cp.asarray(mu_old), *bufs,
                           tend=tend, coord=vc, dx=dx, dy=dy, dt=dt,
                           msft=dmsft, open_x=open_x, open_y=open_y)
    np.testing.assert_allclose(cp.asnumpy(tend), ref, rtol=1e-3,
                               atol=1e-3 * np.abs(ref).max())


@pytest.mark.gpu
@requires_gpu
def test_pd_fold_covers_specified_ring():
    """WRF's rk_update_scalar_pd folds the physics tendency over the FULL
    tile INCLUDING the specified ring: the _spc loop bounds are captured
    at module_em.F:1863-1868 BEFORE the specified narrowing at
    F:1870-1878 (dead code for the fold -- the narrowed bounds are never
    used afterwards), and the fold loop at F:1889-1893 runs over the _spc
    bounds.  The ring cell's folded value is LIVE interior forcing: it is
    the upwind donor for the first interior x face and enters the first
    interior cell's ph_low, so where the limiter bites at a
    boundary-adjacent cell fed by ring inflow the interior tendency
    depends on the ring fold (~10% here).

    Construction: ring qc = 1e-3 with a ring SINK folding it to 2e-4,
    empty first interior cell fed by cr=0.3 ring inflow, a 1e-3 spike two
    cells in draining at cr=0.95 -- the limiter renormalizes the
    boundary-adjacent cells in both variants.  The device update must
    match the float64 mirror with the fold applied EVERYWHERE; the
    ring-excluded fold (the pre-fix behavior) differs at interior cells
    by ~50x the comparison tolerance (asserted, so the pin discriminates).
    """
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import advance_scalars_stage, init_moist_balanced
    from gpuwm.verify.npref import np_pd_fluxes, np_pd_renorm_apply

    nx, ny, nz = 12, 8, 8
    dx = dy = 500.0
    dt_eff = 10.0
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, ztop=6000.0,
                    dt=dt_eff, run_seconds=0.0, moist=True, specified=True)
    vc = make_vertical_coord(nz)
    b = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                        p_surf=cfg.p_surf, ztop=cfg.ztop)
    s = init_moist_balanced(cfg, vc, b, lambda z: np.full(nz, 1.0e-3))
    mub = float(b.mub)
    c1h = vc.c1h[:, None, None]
    c2h = vc.c2h[:, None, None]
    chm = c1h * mub + c2h                       # mup == mup0 == 0

    prof = np.zeros(nx)
    prof[0], prof[2] = 1.0e-3, 1.0e-3           # ring source cell + spike
    q64 = np.broadcast_to(prof[None, None, :],
                          (nz, ny, nx)).astype(np.float64).copy()
    crf = np.full(nx + 1, 0.95)
    crf[1] = 0.3                                # ring-inflow face
    ru64 = crf[None, None, :] * (dx / dt_eff) * np.broadcast_to(
        chm, (nz, ny, nx + 1))
    rv64 = np.zeros((nz, ny + 1, nx))
    rw64 = np.zeros((nz + 1, ny, nx))
    rate = np.zeros((nz, ny, nx))
    rate[:, :, 0] = -8.0e-4                     # ring sink, dt*phys/chm0

    def mirror(fold_ring):
        q0e = q64.copy()
        if fold_ring:                           # WRF: full-tile fold
            q0e += rate
        else:                                   # pre-fix: ring excluded
            q0e[:, 1:-1, 1:-1] += rate[:, 1:-1, 1:-1]
        fl = np_pd_fluxes(q64, q0e, ru64, rv64, rw64, mub, vc, dx, dy,
                          dt_eff, open_x=True, open_y=True)
        tend = np_pd_renorm_apply(q0e, mub, *fl, coord=vc, dx=dx, dy=dy,
                                  dt=dt_eff, open_x=True, open_y=True)
        return np.maximum((chm * q0e + dt_eff * tend) / chm, 0.0)

    wrf = mirror(True)
    ring_excluded = mirror(False)
    inner = np.s_[:, 1:-1, 1:-1]
    atol = 3.0e-6
    gap = np.abs(wrf[inner] - ring_excluded[inner]).max()
    assert gap > 20 * atol, f"construction lost its discrimination ({gap})"

    s.qc[...] = cp.asarray(q64, dtype=cp.float32)
    s.qc0[...] = s.qc
    physics_qc = cp.asarray(chm * rate / dt_eff, dtype=cp.float32)
    physics = SimpleNamespace(
        scalar_for=lambda name: physics_qc if name == "qc" else None)
    advance_scalars_stage(s, cfg,
                          cp.asarray(ru64, dtype=cp.float32),
                          cp.asarray(rv64, dtype=cp.float32),
                          cp.asarray(rw64, dtype=cp.float32),
                          dt_eff, final=True, physics_tendencies=physics)
    got = cp.asnumpy(s.qc).astype(np.float64)
    np.testing.assert_allclose(got[inner], wrf[inner], rtol=0.0, atol=atol)
