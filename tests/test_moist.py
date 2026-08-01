# tests/test_moist.py
"""Moisture arrays, shared transport surface, and moist thermodynamics
(Phase 2 Task 5): state allocation, the theta_m EOS kernel vs mirror, the
public dycore.stage_fluxes interface (with the advection.py rw-placeholder
retirement), and the plan's integration gates -- balanced moist atmosphere
at rest stays at rest, and a moist warm bubble rises faster than the
identical dry one.

Phase 2 Task 10 extends this file with the WK82 analytic sounding
(gpuwm.verify.cases.wk82.wk82_sounding -- CPU float64, checked against the
formulas and the WRF-shipped em_quarter_ss input_sounding tabulation) and
the moist Kessler bubble integration case (gpuwm.verify.cases.moist_bubble).
"""
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.core.grid import make_base_state, make_vertical_coord

# Task 5's tests are all GPU tests and carried a module-level
# ``pytestmark = pytest.mark.gpu``; Task 10 adds CPU-only float64 sounding
# tests, so the module mark became per-test ``@pytest.mark.gpu`` decorators
# (marker-equivalent for every pre-existing test).

_WRF_BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
_INPUT_SOUNDING = (_WRF_BUNDLE / "WRF_source_v4.6.1_group" / "test"
                   / "em_quarter_ss" / "input_sounding")


def _setup(nx=32, nz=16, moist=True):
    cfg = RunConfig(nx=nx, ny=1, nz=nz, dx=100.0, dy=100.0, ztop=6400.0,
                    dt=0.5, run_seconds=1.0, moist=moist)
    vc = make_vertical_coord(nz)
    b = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                        p_surf=cfg.p_surf, ztop=cfg.ztop)
    return cfg, vc, b


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("clamp", [False, True])
def test_scalar_update_in_place_matches_expression_bits(clamp):
    """The scratch-reuse form preserves every eager CuPy FP32 boundary."""
    import cupy as cp

    from gpuwm.core.moist import _update_scalar_in_place

    rng = cp.random.RandomState(20260731)
    shape = (5, 3, 4)
    q0 = rng.standard_normal(shape, dtype=cp.float32)
    chm0 = rng.uniform(0.75, 1.25, shape).astype(cp.float32)
    chm = rng.uniform(0.75, 1.25, shape).astype(cp.float32)
    tend = rng.standard_normal(shape, dtype=cp.float32)
    dt_eff = 0.375
    expected = (chm0 * q0 + dt_eff * tend) / chm
    if clamp:
        expected = cp.maximum(expected, 0.0)

    got = cp.empty_like(q0)
    tend_scratch = tend.copy()
    _update_scalar_in_place(
        got, q0, chm0, chm, tend_scratch, dt_eff, clamp=clamp)

    cp.testing.assert_array_equal(got, expected)


@pytest.mark.gpu
@requires_gpu
def test_moist_state_allocation():
    import cupy as cp
    from gpuwm.core.state import DomainState
    cfg, _, _ = _setup(nx=8, nz=8)
    s = DomainState(cfg)
    for name in ("qv", "qc", "qr", "qv0", "qc0", "qr0"):
        arr = getattr(s, name)
        assert isinstance(arr, cp.ndarray)
        assert arr.shape == (cfg.nz, cfg.ny, cfg.nx)
        assert arr.dtype == cp.float32
        assert float(cp.abs(arr).max()) == 0.0
    dry_cfg, _, _ = _setup(nx=8, nz=8, moist=False)
    s_dry = DomainState(dry_cfg)
    for name in ("qv", "qc", "qr", "qv0", "qc0", "qr0"):
        assert getattr(s_dry, name) is None


@pytest.mark.gpu
@requires_gpu
def test_moist_eos_matches_reference():
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.state import init_theta_perturbation
    from gpuwm.verify.npref import np_calc_p_alpha
    cfg, vc, b = _setup()
    rng = np.random.default_rng(0)
    thp = rng.normal(0.0, 1.0, (cfg.nz, cfg.ny, cfg.nx))
    s = init_theta_perturbation(cfg, vc, b, lambda x, z: thp)
    qv = rng.uniform(0.0, 0.02, (cfg.nz, cfg.ny, cfg.nx)).astype(np.float32)
    s.qv[...] = cp.asarray(qv)
    update_diagnostics(s)
    p_ref, al_ref, alt_ref = np_calc_p_alpha(
        cp.asnumpy(s.thp).astype(np.float64),
        cp.asnumpy(s.php).astype(np.float64),
        cp.asnumpy(s.mup).astype(np.float64), b, vc,
        qv=qv.astype(np.float64))
    np.testing.assert_allclose(cp.asnumpy(s.p), p_ref, rtol=2e-5)
    np.testing.assert_allclose(cp.asnumpy(s.alt), alt_ref, rtol=2e-5)
    np.testing.assert_allclose(cp.asnumpy(s.al), al_ref, rtol=2e-5,
                               atol=2e-5)
    # theta_m raises p where qv > 0 relative to the dry diagnosis
    p_dry, _, _ = np_calc_p_alpha(
        cp.asnumpy(s.thp).astype(np.float64),
        cp.asnumpy(s.php).astype(np.float64),
        cp.asnumpy(s.mup).astype(np.float64), b, vc)
    assert (p_ref > p_dry).all()

    # qv identically zero reduces bitwise to the dry (moist=False) kernel
    dry_cfg, _, _ = _setup(moist=False)
    s_dry = init_theta_perturbation(dry_cfg, vc, b, lambda x, z: thp)
    update_diagnostics(s_dry)
    s.qv[...] = 0.0
    update_diagnostics(s)
    np.testing.assert_array_equal(cp.asnumpy(s.p), cp.asnumpy(s_dry.p))


@pytest.mark.gpu
@requires_gpu
def test_stage_fluxes_public_surface():
    """dycore.stage_fluxes is the shared transport surface: coupled ru/rv
    plus the diagnosed Omega (WRF calc_ww_cp), checked against a float64
    mirror; Omega vanishes exactly at the surface and the rigid lid."""
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.core.state import mu_at_u_faces, mu_at_v_faces
    from gpuwm.verify.npref import random_acoustic_state, s_meta
    s, cfg = random_acoustic_state(seed=3, stretch=1.4)
    ru, rv, ww = dycore.stage_fluxes(s, cfg)
    nz, ny, nx = s.p.shape
    assert ru.shape == (nz, ny, nx + 1)
    assert rv.shape == (nz, ny + 1, nx)
    assert ww.shape == (nz + 1, ny, nx)
    mu = s.total_mu()
    c1h = s.c1h[:, None, None]
    c2h = s.c2h[:, None, None]
    np.testing.assert_array_equal(
        cp.asnumpy(ru), cp.asnumpy((c1h * mu_at_u_faces(mu)[None] + c2h)
                                   * s.u))
    np.testing.assert_array_equal(
        cp.asnumpy(rv), cp.asnumpy((c1h * mu_at_v_faces(mu)[None] + c2h)
                                   * s.v))
    # float64 Omega mirror (WRF calc_ww_cp)
    m = s_meta(s)
    ru64 = cp.asnumpy(ru).astype(np.float64)
    rv64 = cp.asnumpy(rv).astype(np.float64)
    dnw = m["dnw"][:, None, None]
    c1h64 = m["c1h"][:, None, None]
    divv = dnw * ((ru64[:, :, 1:] - ru64[:, :, :-1]) / cfg.dx
                  + (rv64[:, 1:, :] - rv64[:, :-1, :]) / cfg.dy)
    dmdt = divv.sum(axis=0)
    ww_ref = np.zeros((nz + 1, ny, nx))
    ww_ref[1:nz] = -np.cumsum((c1h64 * dnw)[:nz - 1] * dmdt[None]
                              + divv[:nz - 1], axis=0)
    got = cp.asnumpy(ww).astype(np.float64)
    assert not got[0].any() and not got[nz].any()
    assert np.abs(got[1:nz]).max() > 0.0
    np.testing.assert_allclose(got[1:nz], ww_ref[1:nz], rtol=2e-3,
                               atol=2e-3 * np.abs(ww_ref).max())


@pytest.mark.gpu
@requires_gpu
def test_stage_fluxes_public_surface_with_map_factors():
    """Exercise the production stage_fluxes has_msf branch, including the
    face divisions and the msft-weighted Omega recurrence."""
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.core.state import mu_at_u_faces, mu_at_v_faces
    from gpuwm.verify.npref import random_acoustic_state, s_meta

    s, cfg = random_acoustic_state(seed=31, stretch=1.4)
    nz, ny, nx = s.p.shape
    x = np.linspace(0.97, 1.03, nx)
    y = np.linspace(0.98, 1.02, ny)
    msft = y[:, None] * x[None, :]
    msfu = np.concatenate((msft, msft[:, :1]), axis=1)
    msfv = np.concatenate((msft, msft[:1, :]), axis=0)
    s.set_map_coriolis(msft, msfu, msfv)
    assert s.has_msf

    ru, rv, ww = dycore.stage_fluxes(s, cfg)
    mu = s.total_mu()
    c1h = s.c1h[:, None, None]
    c2h = s.c2h[:, None, None]
    ru_expected = ((c1h * mu_at_u_faces(mu)[None] + c2h) * s.u
                   / s.msfu[None])
    rv_expected = ((c1h * mu_at_v_faces(mu)[None] + c2h) * s.v
                   / s.msfv[None])
    np.testing.assert_array_equal(cp.asnumpy(ru), cp.asnumpy(ru_expected))
    np.testing.assert_array_equal(cp.asnumpy(rv), cp.asnumpy(rv_expected))

    m = s_meta(s)
    ru64 = cp.asnumpy(ru).astype(np.float64)
    rv64 = cp.asnumpy(rv).astype(np.float64)
    dnw = m["dnw"][:, None, None]
    divv = dnw * ((ru64[:, :, 1:] - ru64[:, :, :-1]) / cfg.dx
                  + (rv64[:, 1:, :] - rv64[:, :-1, :]) / cfg.dy)
    divv *= msft[None]
    dmdt = divv.sum(axis=0)
    ww_ref = np.zeros((nz + 1, ny, nx))
    ww_ref[1:nz] = -np.cumsum(
        (m["c1h"][:, None, None] * dnw)[:nz - 1] * dmdt[None]
        + divv[:nz - 1], axis=0)
    np.testing.assert_allclose(cp.asnumpy(ww)[1:nz], ww_ref[1:nz],
                               rtol=2e-3,
                               atol=2e-3 * np.abs(ww_ref).max())


@pytest.mark.gpu
@requires_gpu
def test_advection_only_path_requires_zero_w():
    """The dimensionally wrong rw = -(mu*w) placeholder is retired: the
    advection-only path now asserts w == 0 (its only valid regime) and
    the full path advects with stage_fluxes' Omega."""
    from gpuwm.core.advection import add_advection_tendencies
    from gpuwm.verify.npref import random_acoustic_state
    s, cfg = random_acoustic_state(seed=4)
    with pytest.raises(ValueError, match="w == 0"):
        add_advection_tendencies(s, cfg)          # random state has w != 0
    s.w[...] = 0.0
    add_advection_tendencies(s, cfg)              # valid regime unchanged


def _n2_sounding(N=0.01):
    return lambda z: 300.0 * np.exp(N * N * np.asarray(z, float) / 9.81)


@pytest.mark.gpu
@requires_gpu
def test_moist_at_rest_stays_at_rest():
    """Plan gate: a hydrostatically balanced moist atmosphere (qv up to
    12 g/kg) stays at rest through the full moist dycore, < 1e-3 m/s."""
    import cupy as cp
    from gpuwm.core.dycore import run_steps, stability_report
    from gpuwm.core.moist import init_moist_balanced
    cfg = RunConfig(nx=32, ny=1, nz=40, dx=1000.0, dy=1000.0, ztop=10000.0,
                    dt=6.0, run_seconds=0.0, moist=True)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, _n2_sounding(), p_surf=cfg.p_surf, ztop=cfg.ztop)
    qv_prof = lambda z: 0.012 * np.exp(-np.asarray(z, float) / 3000.0)
    s = init_moist_balanced(cfg, vc, b, qv_prof)
    run_steps(s, cfg, n=500)                      # 3000 s
    r = stability_report(s, cfg)
    assert not r["nan"]
    assert r["w_max"] < 1e-3
    assert r["u_max"] < 1e-3
    # vapor is passively conserved at rest
    assert float(cp.abs(s.qv).max()) < 0.0121


@pytest.mark.gpu
@requires_gpu
def test_specified_domain_water_mass_closure():
    """Water-mass budget closes on a specified-BC domain (audit FIX-C).

    WRF's advect_scalar_pd fully supports specified domains
    (module_advect_em.F:7697-7702 sets the limiter's specified bounds) and
    the ratified real74 namelist runs moist_adv_opt=1 -- so production
    scalar transport is flux-renormalized and conservative.  gpuwm's
    pre-fix routing force-disabled the PD limiter under specified BCs
    (moist.py pd gate) and ran unlimited 5th-order advection plus a
    clamp-at-zero, which MANUFACTURES water mass wherever the unlimited
    fluxes overdraw a cell.

    Budget: a sharp interior qc blob rides a nondivergent interior vortex
    on a specified domain whose boundary zone stays dry and calm, so the
    domain-integrated coupled water mass sum((c1h*mu+c2h)*qc*(-dnw)) must
    be conserved (boundary fluxes = 0 by construction -- asserted; fallout
    = 0 with mp_physics=0).  Pre-fix the clamp path fails the closure
    gate; the enabled PD path conserves to FP tolerance.
    """
    import cupy as cp
    from gpuwm.core.dycore import run_steps
    from gpuwm.core.state import init_at_rest
    from gpuwm.ingest.lateral_bc import (attach_lateral_boundaries,
                                         build_state_lateral_boundaries)

    nx, ny, nz = 30, 26, 16
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=500.0, dy=500.0, ztop=8000.0,
                    dt=2.0, run_seconds=0.0, moist=True, specified=True)
    vc = make_vertical_coord(nz)
    b = make_base_state(vc, lambda z: 300.0 * np.exp(
        1e-4 * np.asarray(z, float) / 9.81), p_surf=cfg.p_surf,
        ztop=cfg.ztop)
    s = init_at_rest(cfg, vc, b)
    boundaries = build_state_lateral_boundaries([s, s], [0.0, 3600.0])
    attach_lateral_boundaries(s, boundaries)

    # nondivergent interior vortex from a Gaussian streamfunction (zero to
    # machine precision at the boundary frame)
    xg = (np.arange(nx + 1)) * cfg.dx - 0.5 * nx * cfg.dx      # corners
    yg = (np.arange(ny + 1)) * cfg.dy - 0.5 * ny * cfg.dy
    R = 3.0 * cfg.dx
    psi = 5.0 * R * np.exp(-(xg[None, :] ** 2 + yg[:, None] ** 2)
                           / (2.0 * R * R))                    # (ny+1, nx+1)
    u2d = -(psi[1:, :] - psi[:-1, :]) / cfg.dy                 # (ny, nx+1)
    v2d = (psi[:, 1:] - psi[:, :-1]) / cfg.dx                  # (ny+1, nx)
    s.u[...] = cp.asarray(np.broadcast_to(u2d, (nz, ny, nx + 1)),
                          dtype=cp.float32)
    s.v[...] = cp.asarray(np.broadcast_to(v2d, (nz, ny + 1, nx)),
                          dtype=cp.float32)

    # sharp box blob of qc in the vortex core, far from the boundary zone
    qc = np.zeros((nz, ny, nx), dtype=np.float32)
    qc[5:11, ny // 2 - 2:ny // 2 + 2, nx // 2 - 2:nx // 2 + 2] = 1.0e-3
    s.qc[...] = cp.asarray(qc)
    s.qc0[...] = s.qc

    def water_mass():
        chm = (s.c1h[:, None, None] * s.total_mu()[None]
               + s.c2h[:, None, None])
        dnw_abs = -s.dnw[:, None, None]
        return float(cp.sum((chm * s.qc * dnw_abs).astype(cp.float64)))

    m0 = water_mass()
    run_steps(s, cfg, 25)
    m1 = water_mass()

    frame = np.zeros((ny, nx), bool)
    frame[:6, :] = frame[-6:, :] = True
    frame[:, :6] = frame[:, -6:] = True
    leak = float(cp.abs(s.qc[:, cp.asarray(frame)]).max())
    assert leak < 1e-8, f"blob reached the boundary zone (qc {leak})"
    assert float(s.qc.min()) >= 0.0
    residual = abs(m1 - m0) / m0
    assert residual < 1e-4, (
        f"specified-domain water mass not conserved: {residual} relative "
        f"(m0 {m0}, m1 {m1})")


@pytest.mark.gpu
@requires_gpu
def test_uniform_scalar_stays_uniform_sk2008_d11():
    """SK2008 design constraint D11: an initially uniform scalar stays
    uniform under the full split-explicit step, by construction.

    WRF closes this identity by advecting scalars with the acoustic-substep
    TIME-AVERAGED mass fluxes ru_m/rv_m/ww_m accumulated by sumflux
    (module_small_step_em.F:1473 'needed for consistent mass-conserving
    scalar advection'; solve_em.F:2210-2212 passes them to rk_scalar_tend)
    -- the identical discrete operator that advances mu -- so the coupled
    q==const update telescopes exactly against the mu update.

    Setup: periodic moist domain, non-uniform (periodic-consistent) map
    factors, a 5 K warm bubble driving a strongly divergent acoustic/
    convective flow, and uniform qv = 5 g/kg.  Ten full steps must leave
    qv uniform to FP32 accumulation tolerance.

    Pre-fix this FAILS two ways (audit D11 verdict): (a) scalars were
    advected with the pre-acoustic stage fluxes, missing the acoustic
    perturbation momenta that advance mu (residual ~ c1h*sum(dtau*
    div(u_pp))/C(mu_new)); (b) with msf != 1 the pre-FIX-A kernels dropped
    msft from the scalar operator while the Omega diagnosis kept it.
    """
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.dycore import run_steps
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced

    nx, ny, nz = 16, 12, 16
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=500.0, dy=500.0, ztop=8000.0,
                    dt=2.0, run_seconds=0.0, moist=True)
    vc = make_vertical_coord(nz)
    th = lambda z: 300.0 * np.exp(1e-4 * np.asarray(z, float) / 9.81)
    b = make_base_state(vc, th, p_surf=cfg.p_surf, ztop=cfg.ztop)
    qv0 = 0.005

    def bubble(x, z):
        zz = z[:, None, None] if np.ndim(z) == 1 else z
        L = np.sqrt((x[None, None, :] / 2000.0) ** 2
                    + ((zz - 2000.0) / 1500.0) ** 2)
        return (np.where(L < 1.0, 5.0 * np.cos(np.pi * L / 2) ** 2, 0.0)
                * np.ones((nz, ny, nx)))

    s = init_moist_balanced(cfg, vc, b, lambda z: np.full(nz, qv0),
                            thp_func=bubble)
    # smooth periodic map factors (duplicate staggered column/row wraps)
    ic = (np.arange(nx) + 0.5) / nx
    jc = (np.arange(ny) + 0.5) / ny
    iu = np.arange(nx + 1) / nx
    jv = np.arange(ny + 1) / ny
    fxy = lambda xi, yj: (1.0 + 0.04 * np.sin(2 * np.pi * xi)[None, :]
                          * np.cos(2 * np.pi * yj)[:, None])
    s.set_map_coriolis(msft=fxy(ic, jc), msfu=fxy(iu, jc), msfv=fxy(ic, jv))

    run_steps(s, cfg, 10)
    w_max = float(cp.abs(s.w).max())
    assert w_max > 0.5, f"bubble failed to drive divergent flow (w {w_max})"
    dev = float(cp.abs(s.qv - np.float32(qv0)).max()) / qv0
    assert dev < 2e-5, f"uniform qv drifted by {dev} (relative)"


@pytest.mark.gpu
@requires_gpu
def test_moist_bubble_rises_faster_than_dry():
    """Plan gate: identical 3 K bubbles, one in a 12 g/kg boundary-layer
    moist atmosphere and one dry -- theta_m buoyancy makes the moist one
    rise faster by construction of the coupling."""
    import cupy as cp
    from gpuwm.core.dycore import run_steps, stability_report
    from gpuwm.core.moist import init_moist_balanced
    cfg = RunConfig(nx=100, ny=1, nz=50, dx=200.0, dy=200.0, ztop=10000.0,
                    dt=1.0, run_seconds=0.0, moist=True)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, _n2_sounding(), p_surf=cfg.p_surf, ztop=cfg.ztop)

    def bubble(x, z):
        L = np.sqrt((x[None, None, :] / 2000.0) ** 2
                    + ((z[:, None, None] - 2000.0) / 2000.0) ** 2)
        return np.where(L < 1.0, 3.0 * np.cos(np.pi * L / 2) ** 2, 0.0) \
            * np.ones((cfg.nz, cfg.ny, cfg.nx))

    def qv_bl(z):                                 # 12 g/kg boundary layer
        z = np.asarray(z, float)
        return 0.012 * np.clip((6000.0 - z) / 3000.0, 0.0, 1.0)

    w_max = {}
    for tag, qv_prof in (("moist", qv_bl), ("dry", lambda z: 0.0 * z)):
        s = init_moist_balanced(cfg, vc, b, qv_prof, thp_func=bubble)
        run_steps(s, cfg, n=300)                  # 300 s: growth phase
        r = stability_report(s, cfg)
        assert not r["nan"]
        w_max[tag] = r["w_max"]
    assert w_max["dry"] > 1.0                     # bubble clearly rising
    assert w_max["moist"] > 1.005 * w_max["dry"]


# ---------------------------------------------------------------------------
# Task 10: WK82 analytic sounding (CPU float64, no GPU required)
# ---------------------------------------------------------------------------

def test_wk82_sounding_analytic_form():
    """wk82_sounding transcribes Weisman & Klemp (1982) eqs. (1)-(2):
    theta = 300 + 43*(z/12000)^1.25 below the 12 km tropopause and the
    isothermal (T = 213 K) exponential above; qv = RH*qvs capped at
    14 g/kg, with RH = 1 - 0.75*(z/ztr)^1.25 below ztr and 0.25 above."""
    from gpuwm.core import constants as c
    from gpuwm.verify.cases.wk82 import (QV0, T_TR, THETA_0, THETA_TR, Z_TR,
                                         saturation_mixing_ratio,
                                         wk82_sounding)
    z = np.array([0.0, 10.0, 1000.0, 3000.0, 6000.0, 12000.0, 15000.0,
                  19000.0])
    theta, qv = wk82_sounding(z)
    assert theta.shape == z.shape and qv.shape == z.shape

    # theta: exact analytic values (float64)
    below = z <= Z_TR
    th_ref = np.where(
        below, THETA_0 + (THETA_TR - THETA_0) * (z / Z_TR) ** 1.25,
        THETA_TR * np.exp(c.G * (z - Z_TR) / (c.CP * T_TR)))
    np.testing.assert_allclose(theta, th_ref, rtol=1e-12)
    assert THETA_0 == 300.0 and THETA_TR == 343.0 and Z_TR == 12000.0

    # qv: capped at exactly 14 g/kg in the boundary layer, positive,
    # decreasing with height through the troposphere above the cap
    assert QV0 == 0.014
    np.testing.assert_array_equal(qv[z <= 1000.0], QV0)
    assert (qv > 0.0).all()
    mid = (z > 2000.0) & (z <= 12000.0)
    assert (np.diff(qv[mid]) < 0.0).all()
    assert qv[z == 12000.0] < 1e-4                # dry upper troposphere

    # RH self-consistency: recomputing RH = qv/qvs from the sounding's own
    # hydrostatic (p, T) recovers eq. (2) exactly where the cap is inactive
    p, T = wk82_sounding(z, full=True)[2:]
    rh = qv / saturation_mixing_ratio(p, T)
    rh_ref = np.where(below, 1.0 - 0.75 * (z / Z_TR) ** 1.25, 0.25)
    uncapped = qv < QV0
    np.testing.assert_allclose(rh[uncapped], rh_ref[uncapped], rtol=1e-8)
    assert (rh[~uncapped] <= rh_ref[~uncapped] + 1e-12).all()


@pytest.mark.skipif(not _INPUT_SOUNDING.exists(),
                    reason="WRF bundle input_sounding not present")
def test_wk82_sounding_matches_wrf_tabulation():
    """The shipped em_quarter_ss input_sounding IS the WK82 sounding the
    local get_sounding reads (it has no analytic branch): theta below the
    tropopause must match to the file's precision, qv to a few percent
    (the file generator's saturation formula is not in the tree), and the
    14 g/kg cap must end in the same layer."""
    rows = np.array([[float(v) for v in line.split()[:3]]
                     for line in _INPUT_SOUNDING.read_text().splitlines()[1:]
                     if line.split()])
    z_f, th_f, qv_f = rows[:, 0], rows[:, 1], rows[:, 2] * 1e-3  # kg/kg

    from gpuwm.verify.cases.wk82 import QV0, wk82_sounding
    theta, qv = wk82_sounding(z_f)

    trop = z_f <= 12000.0
    np.testing.assert_allclose(theta[trop], th_f[trop], atol=1e-3)
    np.testing.assert_allclose(qv[trop], qv_f[trop], rtol=0.05)
    # the cap releases in the same layer as the file (1200-1300 m)
    capped_f, capped = qv_f >= 0.0139999, qv >= QV0 - 1e-12
    assert z_f[capped_f].max() == z_f[capped].max() == 1200.0


# ---------------------------------------------------------------------------
# Task 10: moist Kessler bubble integration case (pre-supercell shakeout)
# ---------------------------------------------------------------------------

def test_moist_bubble_case_admits_complete_no_pbl_operator():
    from gpuwm.config import validate_run_config
    from gpuwm.verify.cases.moist_bubble import default_config

    cfg = validate_run_config(default_config())
    assert cfg.km_opt == 4 and cfg.bl_pbl_physics == 0
