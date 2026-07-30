# tests/test_terrain_dynamics.py  (Phase 2 Task 4: damp_opt=3 rework +
# terrain metric/PGF wiring)
import dataclasses

import numpy as np
import pytest
from conftest import requires_gpu

from gpuwm.config import RunConfig
from gpuwm.core.grid import make_base_state, make_vertical_coord

pytestmark = pytest.mark.gpu


def theta_constant_N(z, N=0.01):
    """Constant Brunt-Vaisala sounding theta(z) = 300*exp(N^2 z / g)."""
    return 300.0 * np.exp(N * N * np.asarray(z, dtype=np.float64) / 9.81)


# ---- damp_opt=3 rework (task prerequisite) ---------------------------------
#
# WRF damp_opt=3 (Klemp-Dudhia-Hassiotis 2008) is a w-ONLY implicit Rayleigh
# damper applied inside the acoustic w solve (module_small_step_em.F
# advance_w):
#   w'' <- (w'' - dampwt*(c1f*mut+c2f)*w_ref) / (1 + dampwt),
#   dampwt = dtau*dampcoef*sin^2(pi/2 * (z - (ztop_col - zdamp))/zdamp)
# with per-column heights from the t* geopotential.  The Phase 1 relaxational
# damper decelerated the mean flow (it relaxed u/v toward zero), which would
# corrupt the hill2d momentum-flux gate.


@requires_gpu
def test_implicit_w_damper_formula():
    # From an at-rest isothermal state (w_ref = 0, phi' = 0) one acoustic
    # substep with damp_opt=3 must yield exactly the undamped w'' divided by
    # (1 + dampwt(z)) - the KDH implicit filter - and leave the levels below
    # the damping layer bitwise untouched.
    import cupy as cp
    from gpuwm.core.acoustic import acoustic_substep
    from gpuwm.verify.npref import build_isothermal_rest_state

    zdamp, dampcoef, dtau = 3000.0, 0.4, 0.5
    s0, cfg0, _ = build_isothermal_rest_state(nx=16, nz=16, dx=100.0, T=300.0,
                                              pulse_amp=1.0, pulse_x0=800.0,
                                              pulse_halfwidth=300.0)
    s1, _, _ = build_isothermal_rest_state(nx=16, nz=16, dx=100.0, T=300.0,
                                           pulse_amp=1.0, pulse_x0=800.0,
                                           pulse_halfwidth=300.0)
    cfg1 = dataclasses.replace(cfg0, damp_opt=3, zdamp=zdamp,
                               dampcoef=dampcoef)
    acoustic_substep(s0, cfg0, dtau=dtau, first=True)
    acoustic_substep(s1, cfg1, dtau=dtau, first=True)

    w0 = cp.asnumpy(s0.w_pp).astype(np.float64)
    w1 = cp.asnumpy(s1.w_pp).astype(np.float64)
    z = cp.asnumpy(s0.phb).astype(np.float64) / 9.81      # (nz+1,) at rest
    hbot = z[-1] - zdamp
    arg = np.clip((z - hbot) / zdamp, 0.0, None) * (0.5 * np.pi)
    dampwt = np.where(z >= hbot, dtau * dampcoef * np.sin(arg) ** 2, 0.0)

    in_layer = (z >= hbot + 1.0) & (z < z[-1] - 1.0)      # interior w levels
    below = z < hbot - 1.0
    assert dampwt[in_layer].max() > 0.05                  # damper engaged
    assert np.array_equal(w1[below], w0[below])           # untouched below
    np.testing.assert_allclose(
        w1[in_layer], w0[in_layer] / (1.0 + dampwt[in_layer, None, None]),
        rtol=1e-5, atol=1e-9)
    assert np.abs(w0[in_layer]).max() > 0.0               # non-vacuous


@requires_gpu
def test_implicit_w_damper_matches_mirror():
    # Full-substep kernel-vs-float64-mirror match with the damper active
    # (stretched grid so fnm/fnp/rdn/rdnw orientations are exercised).
    import cupy as cp
    from gpuwm.core.acoustic import acoustic_substep
    from gpuwm.verify.npref import (np_acoustic_substep, random_acoustic_state,
                                    snapshot)
    s, cfg = random_acoustic_state(seed=21, stretch=1.2)
    cfg = dataclasses.replace(cfg, damp_opt=3, zdamp=3000.0, dampcoef=0.4)
    before = snapshot(s)
    acoustic_substep(s, cfg, dtau=0.5, first=False)
    ref = np_acoustic_substep(before, cfg, dtau=0.5)
    for name in ("u_pp", "v_pp", "w_pp", "ph_pp", "mu_pp", "th_pp", "p_pp"):
        np.testing.assert_allclose(cp.asnumpy(getattr(s, name)), ref[name],
                                   rtol=2e-4, atol=1e-6, err_msg=name)


@requires_gpu
def test_sponge_does_not_decelerate_mean_flow():
    # Plan gate: uniform flow U through the damp_opt=3 sponge layer must not
    # be decelerated - max|u - U| < 0.01 m/s after 1000 steps.  The Phase 1
    # relaxational damper relaxed u toward zero and fails this by ~U.
    import cupy as cp
    from gpuwm.core.dycore import run_steps, stability_report
    from gpuwm.core.state import init_at_rest

    U = 10.0
    cfg = RunConfig(nx=32, ny=1, nz=32, dx=1000.0, dy=1000.0, ztop=10000.0,
                    dt=6.0, run_seconds=0.0, damp_opt=3, zdamp=4000.0,
                    dampcoef=0.2)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, theta_constant_N, p_surf=cfg.p_surf,
                        ztop=cfg.ztop)
    s = init_at_rest(cfg, vc, b)
    s.u[...] = U
    run_steps(s, cfg, n=1000)
    assert not stability_report(s)["nan"]
    du = float(cp.abs(s.u - U).max())
    assert du < 0.01, f"sponge decelerated the mean flow: max|u-U| = {du}"


# ---- general hybrid/terrain acoustic kernels (kernel vs float64 mirror) ----

_SUBSTEP_FIELDS = ("u_pp", "v_pp", "w_pp", "ph_pp", "mu_pp", "th_pp",
                   "ww_pp", "p_pp", "al_pp")


def _assert_substep_matches(s, ref):
    import cupy as cp
    tol = {"al_pp": dict(rtol=3e-4, atol=1e-9)}
    for name in _SUBSTEP_FIELDS:
        np.testing.assert_allclose(
            cp.asnumpy(getattr(s, name)), ref[name], err_msg=name,
            **tol.get(name, dict(rtol=2e-4, atol=1e-6)))


@requires_gpu
def test_acoustic_substep_matches_mirror_terrain_hybrid():
    # Bell-ridge terrain + WRF cubic-B hybrid + stretched eta: the general
    # c1/c2-weighted couplings, the terrain base-pressure-gradient (al'')
    # term, and the kinematic w'' surface BC are all exercised.
    from gpuwm.core.acoustic import acoustic_substep
    from gpuwm.verify.npref import (np_acoustic_substep, random_acoustic_state,
                                    snapshot)
    s, cfg = random_acoustic_state(seed=13, nz=10, ny=2, nx=12, stretch=1.2,
                                   hybrid_opt=2, hill_height=800.0)
    before = snapshot(s)
    acoustic_substep(s, cfg, dtau=0.5, first=False)
    _assert_substep_matches(s, np_acoustic_substep(before, cfg, dtau=0.5))


@requires_gpu
def test_acoustic_substep_matches_mirror_terrain_damped():
    # Terrain + hybrid + the KDH damper together (per-column layer heights).
    from gpuwm.core.acoustic import acoustic_substep
    from gpuwm.verify.npref import (np_acoustic_substep, random_acoustic_state,
                                    snapshot)
    s, cfg = random_acoustic_state(seed=17, nz=12, ny=1, nx=10,
                                   hybrid_opt=2, hill_height=600.0)
    cfg = dataclasses.replace(cfg, damp_opt=3, zdamp=2500.0, dampcoef=0.3)
    before = snapshot(s)
    acoustic_substep(s, cfg, dtau=0.4, first=False)
    _assert_substep_matches(s, np_acoustic_substep(before, cfg, dtau=0.4))


# ---- horizontal PGF term 4: FULL (phb-inclusive) half-level geopotential ----
#
# WRF's term-4 coefficient is the FULL t* half-level geopotential php =
# 0.5*(phb(k)+phb(k+1)+ph(k)+ph(k+1)) (calc_php,
# module_big_step_utilities_em.F:1261, refreshed every RK stage at
# module_em.F:181), consumed by the large-step PGF
# (module_big_step_utilities_em.F:2315-2316, 2390-2391) and by every
# acoustic substep (advance_uv, module_small_step_em.F:861-862, 935-936;
# grid%php passed at solve_em.F:1276).  Over a flat base state phb is
# horizontally constant and the phb part differences to exactly zero, so
# the flat-terrain full-step bitwise pins
# (tests/test_coriolis_map.py::test_msf_one_f_zero_bitwise_phase2
# [dry_flat|moist|open], goldens captured from the PRE-fix tree) are the
# bit-identity assertion for flat cases; over terrain the phb term is the
# leading term-4 contribution and these hand pins fail against the
# perturbation-only coefficient.


def _hand_dpxy_wrf(ph1, pe, al_pert, alt, pb, ph_full, mu4, mut, coord,
                   axis, rd, *, top_rule):
    """Float64 hand evaluation of the WRF 4-term horizontal PGF ``dpxy``.

    Transcribed directly from WRF v4.6.1 module_small_step_em.F advance_uv
    (:820-945) == module_big_step_utilities_em.F
    horizontal_pressure_gradient (:2281-2412), periodic faces along
    ``axis`` (2 = u faces, 1 = v faces), independent of gpuwm.verify.npref:

      dpxy = 0.5*rd*(c1h*<mut>_f + c2h) * ( d(ph1)[k+1] + d(ph1)[k]
                 + (alt_A + alt_B)*d(pe) + (al_A + al_B)*d(pb) )
           + rd*d(php_half_FULL) * (rdnw*(dpn[k+1] - dpn[k]) - c1h*<mu4>_f)

    ``ph1`` is the term-1 geopotential (phi'' in the acoustic step, the t*
    perturbation phi in the large step), ``pe`` the (damped) perturbation
    pressure, ``al_pert`` the perturbation specific volume riding d(pb),
    ``ph_full`` the FULL t* geopotential phb+php at full levels (term 4's
    coefficient via its half-level average), ``mu4`` the term-4 column
    mass (mu'' / mu'), ``mut`` the t* total dry mass.
    """
    nz = alt.shape[0]
    ax2 = axis - 1
    d3 = lambda f: f - np.roll(f, 1, axis=axis)
    s3 = lambda f: f + np.roll(f, 1, axis=axis)
    r2 = lambda f: np.roll(f, 1, axis=ax2)
    c1h = coord["c1h"][:, None, None]
    c2h = coord["c2h"][:, None, None]
    muf = 0.5 * (mut + r2(mut))
    dpxy = 0.5 * rd * (c1h * muf[None] + c2h) * (
        d3(ph1[1:]) + d3(ph1[:-1])
        + s3(alt) * d3(pe) + s3(al_pert) * d3(pb))
    ps = s3(pe)
    dpn = np.zeros((nz + 1,) + ps.shape[1:])
    dpn[0] = 0.5 * (coord["cf1"] * ps[0] + coord["cf2"] * ps[1]
                    + coord["cf3"] * ps[2])
    fnm = coord["fnm"][1:nz, None, None]
    fnp = coord["fnp"][1:nz, None, None]
    dpn[1:nz] = 0.5 * (fnm * ps[1:nz] + fnp * ps[:nz - 1])
    if top_rule == "acoustic":
        dpn[nz] = 0.5 * (
            coord["cf1"] * ps[nz - 1]
            + coord["cf2"] * ps[nz - 2]
            + coord["cf3"] * ps[nz - 3])
    elif top_rule == "slow":
        dpn[nz] = 0.5 * (
            coord["cfn"] * ps[nz - 1]
            + coord["cfn1"] * ps[nz - 2])
    elif top_rule != "open":
        raise ValueError(f"unknown top pressure extrapolation {top_rule!r}")
    php_h = 0.5 * (ph_full[:-1] + ph_full[1:])         # WRF calc_php
    dmu = 0.5 * (mu4 + r2(mu4))
    rdnw = coord["rdnw"][:, None, None]
    return dpxy + rd * d3(php_h) * (rdnw * (dpn[1:] - dpn[:-1])
                                    - c1h * dmu[None])


def _coord64(s):
    import cupy as cp
    coord = {n: cp.asnumpy(getattr(s, n)).astype(np.float64)
             for n in ("c1h", "c2h", "fnm", "fnp", "rdnw")}
    for n in ("cf1", "cf2", "cf3", "cfn", "cfn1"):
        coord[n] = float(getattr(s, n))
    return coord


@requires_gpu
def test_acoustic_pgf_term4_full_geopotential_over_terrain():
    # Every acoustic substep's term-4 coefficient must be the FULL t*
    # half-level geopotential (phb half-level + php), per WRF advance_uv
    # (module_small_step_em.F:861-862, 935-936 with grid%php from
    # calc_php).  Hand-computed WRF-style over a bell-ridge terrain base:
    # both the kernel and the float64 mirror must match; the pre-fix
    # perturbation-only coefficient misses rd*d(phb_half)*(dp''/dnu -
    # c1h*mu'') and fails by O(1) in u''.
    import cupy as cp
    from gpuwm.core.acoustic import acoustic_substep_explicit
    from gpuwm.verify.npref import (np_advance_uv, random_acoustic_state,
                                    s_meta)

    s, cfg = random_acoustic_state(seed=41, nz=10, ny=6, nx=12, stretch=1.2,
                                   hybrid_opt=2, hill_height=800.0)
    before = {n: cp.asnumpy(getattr(s, n)).astype(np.float64)
              for n in ("u_pp", "v_pp", "mu_pp", "th_pp", "p_pp",
                        "ph_pp", "al_pp", "p_pp_old")}
    meta = s_meta(s)
    dtau = 0.5
    acoustic_substep_explicit(s, cfg, dtau=dtau, first=False)

    coord = _coord64(s)
    mut = meta["mub2d"] + meta["mup"]
    ph_full = meta["phb"] + meta["php"]                # 3-D over terrain
    assert meta["phb"].ndim == 3                       # non-flat base state
    pe = before["p_pp"] + cfg.smdiv * (before["p_pp"] - before["p_pp_old"])
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    for name, arr, axis, n, rd in (
            ("u_pp", s.u_pp, 2, nx, 1.0 / cfg.dx),
            ("v_pp", s.v_pp, 1, ny, 1.0 / cfg.dy)):
        dpxy = _hand_dpxy_wrf(before["ph_pp"], pe, before["al_pp"],
                              meta["alt"], meta["pb"], ph_full,
                              before["mu_pp"], mut, coord, axis, rd,
                              top_rule="acoustic")
        tend = meta["ru_t"] if name == "u_pp" else meta["rv_t"]
        if axis == 2:
            core = before[name][:, :, :nx] + dtau * (tend[:, :, :nx] - dpxy)
            expected = np.concatenate([core, core[:, :, :1]], axis=2)
        else:
            core = before[name][:, :ny, :] + dtau * (tend[:, :ny, :] - dpxy)
            expected = np.concatenate([core, core[:, :1, :]], axis=1)
        got = cp.asnumpy(arr).astype(np.float64)
        scale = np.abs(expected).max()
        np.testing.assert_allclose(got, expected, rtol=2e-4,
                                   atol=2e-4 * scale, err_msg=name)

    # the float64 mirror must carry the same full-geopotential term 4
    u_ref, v_ref = np_advance_uv(before, meta, cfg, dtau=dtau)
    np.testing.assert_allclose(cp.asnumpy(s.u_pp), u_ref,
                               rtol=2e-4, atol=1e-5, err_msg="u_pp mirror")
    np.testing.assert_allclose(cp.asnumpy(s.v_pp), v_ref,
                               rtol=2e-4, atol=1e-5, err_msg="v_pp mirror")


@requires_gpu
def test_large_step_pgf_term4_full_geopotential_over_terrain():
    # The large-step horizontal PGF (dycore._add_slow_tendencies) must use
    # the same FULL half-level geopotential in term 4 (WRF
    # horizontal_pressure_gradient, module_big_step_utilities_em.F:
    # 2390-2391 x / 2315-2316 y, php from calc_php :1261).  With zero
    # winds the stage's flux-advection terms vanish and ru_t/rv_t are
    # exactly -dpxy, hand-computed here WRF-style in float64.
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.verify.npref import random_acoustic_state, s_meta

    s, cfg = random_acoustic_state(seed=43, nz=10, ny=6, nx=12, stretch=1.2,
                                   hybrid_opt=2, hill_height=800.0)
    s.u[...] = 0.0
    s.v[...] = 0.0
    s.w[...] = 0.0
    update_diagnostics(s)
    for name in ("ru_t", "rv_t", "rw_t", "rth_t", "rph_t", "rmu_t"):
        getattr(s, name)[...] = 0
    ru, rv, ww = dycore.stage_fluxes(s, cfg)
    dycore._add_slow_tendencies(s, cfg, ru, rv, ww)

    meta = s_meta(s)
    coord = _coord64(s)
    mut = meta["mub2d"] + meta["mup"]
    assert meta["phb"].ndim == 3                       # non-flat base state
    ph_full = meta["phb"] + meta["php"]
    pp = meta["p"] - meta["pb"]                        # perturbation pressure
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    for name, arr, axis, n, rd in (
            ("ru_t", s.ru_t, 2, nx, 1.0 / cfg.dx),
            ("rv_t", s.rv_t, 1, ny, 1.0 / cfg.dy)):
        dpxy = _hand_dpxy_wrf(meta["php"], pp, meta["al"], meta["alt"],
                              meta["pb"], ph_full, meta["mup"], mut,
                              coord, axis, rd, top_rule="slow")
        got = cp.asnumpy(arr).astype(np.float64)
        scale = np.abs(dpxy).max()
        if axis == 2:
            np.testing.assert_allclose(got[:, :, :nx], -dpxy, rtol=2e-4,
                                       atol=2e-4 * scale, err_msg=name)
            np.testing.assert_array_equal(got[:, :, nx], got[:, :, 0])
        else:
            np.testing.assert_allclose(got[:, :ny, :], -dpxy, rtol=2e-4,
                                       atol=2e-4 * scale, err_msg=name)
            np.testing.assert_array_equal(got[:, ny, :], got[:, 0, :])


@requires_gpu
def test_large_step_horizontal_pgf_uses_stage_moist_cq():
    """WRF scales the slow nonhydrostatic dpx/dpy by cqu/cqv too.

    ``horizontal_pressure_gradient`` applies the stage-fixed ``calc_cq``
    factors at module_big_step_utilities_em.F:2317/2392.  The acoustic PGF
    already consumes the same factors; this pins the large-step half of that
    shared contract independently.
    """
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.verify.npref import (np_calc_cq, random_acoustic_state,
                                    s_meta)

    s, cfg = random_acoustic_state(
        seed=144, nz=8, ny=5, nx=10, stretch=1.15, hybrid_opt=2,
        hill_height=650.0, moist=True, mp_physics=10)
    cfg = dataclasses.replace(cfg, top_lid=False, moist_cq=True)
    s.u[...] = 0.0
    s.v[...] = 0.0
    s.w[...] = 0.0
    qv = np.linspace(0.001, 0.018, cfg.nz * cfg.ny * cfg.nx).reshape(
        cfg.nz, cfg.ny, cfg.nx)
    for scale, name in zip(
            (1.0, 0.20, 0.10, 0.05, 0.03, 0.02),
            ("qv", "qc", "qr", "qi", "qs", "qg"), strict=True):
        getattr(s, name)[...] = cp.asarray(scale * qv, dtype=cp.float32)
    update_diagnostics(s)
    for name in ("ru_t", "rv_t", "rw_t", "rth_t", "rph_t", "rmu_t"):
        getattr(s, name)[...] = 0

    ru, rv, ww = dycore.stage_fluxes(s, cfg)
    dycore._add_slow_tendencies(s, cfg, ru, rv, ww)

    meta = s_meta(s)
    coord = _coord64(s)
    mut = meta["mub2d"] + meta["mup"]
    ph_full = meta["phb"] + meta["php"]
    pp = meta["p"] - meta["pb"]
    cqu, cqv, _ = np_calc_cq(meta, mp_physics=cfg.mp_physics)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    for name, arr, cq, axis, n, rd in (
            ("ru_t", s.ru_t, cqu, 2, nx, 1.0 / cfg.dx),
            ("rv_t", s.rv_t, cqv, 1, ny, 1.0 / cfg.dy)):
        dpxy = _hand_dpxy_wrf(meta["php"], pp, meta["al"], meta["alt"],
                              meta["pb"], ph_full, meta["mup"], mut,
                              coord, axis, rd, top_rule="open")
        got = cp.asnumpy(arr).astype(np.float64)
        expect = -cq.take(indices=range(n), axis=axis) * dpxy
        scale = np.abs(expect).max()
        if axis == 2:
            np.testing.assert_allclose(got[:, :, :nx], expect, rtol=2e-4,
                                       atol=2e-4 * scale, err_msg=name)
        else:
            np.testing.assert_allclose(got[:, :ny, :], expect, rtol=2e-4,
                                       atol=2e-4 * scale, err_msg=name)


@requires_gpu
@pytest.mark.parametrize("moist_mode", [0, 1, 2],
                         ids=["dry", "warm-rain", "morrison"])
def test_slow_buoyancy_includes_wrf_one_sided_top_face(moist_mode):
    """WRF ``pg_buoy_w`` evaluates the top full face with one-sided p'."""
    import cupy as cp
    from gpuwm.core import constants as c
    from gpuwm.core import dycore
    from gpuwm.verify.npref import random_acoustic_state, s_meta

    s, cfg = random_acoustic_state(
        seed=45 + moist_mode, nz=8, ny=4, nx=10, stretch=1.2,
        hybrid_opt=2, msf_amp=0.08, moist=moist_mode > 0,
        mp_physics=(10 if moist_mode == 2 else 0))
    cfg = dataclasses.replace(cfg, top_lid=False)
    if moist_mode:
        fields = [s.qv, s.qc, s.qr]
        if moist_mode == 2:
            fields += [s.qi, s.qs, s.qg]
        for n, field in enumerate(fields, start=1):
            field[...] = cp.float32(n * 2.0e-4)
    s.rw_t[...] = 0.0
    meta = s_meta(s)

    dycore._launch_slow_buoyancy(s, cfg)
    got = cp.asnumpy(s.rw_t[-1]).astype(np.float64)

    pb = np.broadcast_to(
        meta["pb"][:, None, None] if meta["pb"].ndim == 1 else meta["pb"],
        meta["p"].shape)
    ppert_top = meta["p"][-1] - pb[-1]
    qhat = np.zeros_like(ppert_top)
    if moist_mode:
        names = ["qv", "qc", "qr"] + (
            ["qi", "qs", "qg"] if moist_mode == 2 else [])
        qtot = sum(meta[name] for name in names)
        qhat = 0.5 * (qtot[-1] + qtot[-2])
    cq1 = 1.0 / (1.0 + qhat)
    cq2 = qhat * cq1
    expected = c.G * (
        cq1 * 2.0 * meta["rdnw"][-1] * (-ppert_top)
        - meta["c1f"][-1] * meta["mup"]
        - cq2 * (meta["c1f"][-1] * meta["mub2d"]
                 + meta["c2f"][-1])) / meta["msft"]
    assert np.abs(expected).max() > 0.0
    np.testing.assert_allclose(got, expected, rtol=3e-4,
                               atol=3e-4 * np.abs(expected).max())


@requires_gpu
@pytest.mark.parametrize("order", [2, 5])
def test_rhs_ph_hadv_includes_wrf_extrapolated_top_face(order):
    """WRF's top rhs_ph row extrapolates u/v with cfn/cfn1."""
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.core.state import mu_at_u_faces, mu_at_v_faces
    from gpuwm.verify.npref import random_acoustic_state, s_meta

    s, cfg = random_acoustic_state(
        seed=54 + order, nz=8, ny=8, nx=14, stretch=1.2,
        hybrid_opt=2, hill_height=500.0, msf_amp=0.08)
    cfg = dataclasses.replace(cfg, top_lid=False, h_sca_adv_order=order)
    s.rph_t[...] = 0.0
    mu = s.total_mu()
    dycore.add_rhs_ph_hadv(s, cfg, mu_at_u_faces(mu), mu_at_v_faces(mu))
    got = cp.asnumpy(s.rph_t[-1]).astype(np.float64)

    meta = s_meta(s)
    ph = np.broadcast_to(
        (meta["phb"][:, None, None] if meta["phb"].ndim == 1
         else meta["phb"]), meta["php"].shape) + meta["php"]
    ph = ph[-1]
    mut = meta["mub2d"] + meta["mup"]
    mux = 0.5 * (mut + np.roll(mut, 1, axis=1))
    muy = 0.5 * (mut + np.roll(mut, 1, axis=0))
    u_top = meta["cfn"] * meta["u"][-1, :, :cfg.nx] \
        + meta["cfn1"] * meta["u"][-2, :, :cfg.nx]
    v_top = meta["cfn"] * meta["v"][-1, :cfg.ny, :] \
        + meta["cfn1"] * meta["v"][-2, :cfg.ny, :]
    fcx = ((meta["c1f"][-1] * mux + meta["c2f"][-1]) * u_top
           * meta["msfu"][:, :cfg.nx])
    fcy = ((meta["c1f"][-1] * muy + meta["c2f"][-1]) * v_top
           * meta["msfv"][:cfg.ny, :])
    if order == 2:
        fx = fcx * (ph - np.roll(ph, 1, axis=1))
        fy = fcy * (ph - np.roll(ph, 1, axis=0))
        hx = 0.5 / cfg.dx * (fx + np.roll(fx, -1, axis=1))
        hy = 0.5 / cfg.dy * (fy + np.roll(fy, -1, axis=0))
    else:
        def cdiff(axis):
            return (45.0 * (np.roll(ph, -1, axis=axis)
                            - np.roll(ph, 1, axis=axis))
                    - 9.0 * (np.roll(ph, -2, axis=axis)
                             - np.roll(ph, 2, axis=axis))
                    + (np.roll(ph, -3, axis=axis)
                       - np.roll(ph, 3, axis=axis))) / 60.0
        hx = 0.5 / cfg.dx * (fcx + np.roll(fcx, -1, axis=1)) * cdiff(1)
        hy = 0.5 / cfg.dy * (fcy + np.roll(fcy, -1, axis=0)) * cdiff(0)
    expected = -(hx + hy) / meta["msft"]
    assert np.abs(expected).max() > 0.0
    np.testing.assert_allclose(got, expected, rtol=3e-4,
                               atol=3e-4 * np.abs(expected).max())


@requires_gpu
def test_slow_geopotential_fused_kernel_forces_the_open_top_row():
    """WRF resets top rph_t, then adds g*w and horizontal advection."""
    import cupy as cp
    from gpuwm.core import constants as c
    from gpuwm.core import dycore
    from gpuwm.verify.npref import (np_rhs_ph_hadv, random_acoustic_state,
                                    s_meta)

    s, cfg = random_acoustic_state(
        seed=61, nz=8, ny=8, nx=14, stretch=1.2, hybrid_opt=2,
        hill_height=500.0, msf_amp=0.08)
    cfg = dataclasses.replace(cfg, top_lid=False, h_sca_adv_order=5)
    s.rph_t[...] = 0.0
    s.rph_t[-1] = cp.float32(1234.5)
    meta = s_meta(s)

    dycore._launch_slow_geopotential(
        s, cfg, cp.zeros_like(s.w), add_vertical=True)
    got = cp.asnumpy(s.rph_t[-1]).astype(np.float64)

    mass = (meta["c1f"][-1] * (meta["mub2d"] + meta["mup"])
            + meta["c2f"][-1])
    expected = (np_rhs_ph_hadv(meta, cfg)[-1]
                + mass * c.G * meta["w"][-1] / meta["msft"])
    assert np.abs(expected).max() > 0.0
    assert not np.allclose(got, 1234.5)
    np.testing.assert_allclose(got, expected, rtol=3e-4,
                               atol=3e-4 * np.abs(expected).max())


# ---- rhs_ph horizontal advection: h_sca_adv_order = 5 ------------------------
#
# WRF's geopotential equation advects phi with the h_sca_adv_order stencil
# (rhs_ph, module_big_step_utilities_em.F:1435 advective_order =
# config_flags%h_sca_adv_order; Registry.EM_COMMON:2872 default 5, unset in
# the reference namelist).  The reference therefore ran the <=6 branch's
# 1/60-weighted 7-point centered stencil on ph+phb
# (module_big_step_utilities_em.F:1786-1795 y / 1949-1959 x) with the
# spec-zone narrowing: interior [jds+3, jde-4], 4th-order 1/12 rows at
# jds+2/jde-3 (y gated on open OR specified :1819/1850; x gated on open
# ONLY :1973/:1997 -- under specified the x rows ids+2/ide-3 get NO x
# advection), 2nd-order rows at jds+1/jde-2 and ids+1/ide-2
# (:1885/1911/2024/2051), nothing on the spec rows themselves.  gpuwm's
# default stays the frozen order-2 path (bitwise pins); the real74
# production configs select order 5.


def _hand_hadv_coef(meta):
    """Float64 face advective coefficients (c1f*<mu>_f+c2f)*(u_k+u_{k-1})
    *msf_face for interior full levels, per WRF rhs_ph."""
    nz = meta["alt"].shape[0]
    c1f = meta["c1f"][1:nz, None, None]
    c2f = meta["c2f"][1:nz, None, None]
    mut = meta["mub2d"] + meta["mup"]
    mux = 0.5 * (mut + np.roll(mut, 1, axis=1))
    mux = np.concatenate([mux, mux[:, :1]], axis=1)
    muy = 0.5 * (mut + np.roll(mut, 1, axis=0))
    muy = np.concatenate([muy, muy[:1, :]], axis=0)
    fcx = ((c1f * mux[None] + c2f)
           * (meta["u"][1:nz] + meta["u"][:nz - 1]) * meta["msfu"][None])
    fcy = ((c1f * muy[None] + c2f)
           * (meta["v"][1:nz] + meta["v"][:nz - 1]) * meta["msfv"][None])
    return fcx, fcy


@requires_gpu
def test_rhs_ph_hadv_order5_interior_matches_hand_coefficients():
    # Periodic interior pin: the order-5 phi advection must equal the
    # hand-evaluated WRF <=6-branch stencil -(0.25*rd/msft)*(coef_f+1 +
    # coef_f) * sum_o w_o*phF(i+o) with w = [1,-9,45,0,-45,9,-1]/60
    # (module_big_step_utilities_em.F:1786-1795, 1949-1959), on both the
    # kernel-side CuPy path and the float64 mirror.
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.core.state import mu_at_u_faces, mu_at_v_faces
    from gpuwm.verify.npref import (np_rhs_ph_hadv, random_acoustic_state,
                                    s_meta)

    s, cfg = random_acoustic_state(seed=47, nz=8, ny=8, nx=14, stretch=1.2,
                                   hybrid_opt=2, hill_height=600.0,
                                   msf_amp=0.1)
    cfg5 = dataclasses.replace(cfg, h_sca_adv_order=5)
    s.rph_t[...] = 0
    mu = s.total_mu()
    dycore.add_rhs_ph_hadv(s, cfg5, mu_at_u_faces(mu), mu_at_v_faces(mu))
    got = cp.asnumpy(s.rph_t).astype(np.float64)

    meta = s_meta(s)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    ph_f = meta["phb"] + meta["php"]
    ph_i = ph_f[1:nz]
    fcx, fcy = _hand_hadv_coef(meta)
    w = np.array([1.0, -9.0, 45.0, 0.0, -45.0, 9.0, -1.0]) / 60.0
    d5 = lambda f, ax: sum(w[o + 3] * np.roll(f, o, axis=ax)
                           for o in range(-3, 4))
    hadv = (0.25 / cfg.dx) * (fcx[:, :, :nx]
                              + np.roll(fcx[:, :, :nx], -1, axis=2)) \
        * d5(ph_i, 2) / meta["msft"][None]
    hadv += (0.25 / cfg.dy) * (fcy[:, :ny, :]
                               + np.roll(fcy[:, :ny, :], -1, axis=1)) \
        * d5(ph_i, 1) / meta["msft"][None]
    scale = np.abs(hadv).max()
    np.testing.assert_allclose(got[1:nz], -hadv, rtol=2e-4,
                               atol=2e-4 * scale)
    assert np.all(got[0] == 0.0)
    assert np.abs(got[nz]).max() > 0.0

    ref = np_rhs_ph_hadv(meta, cfg5)
    np.testing.assert_allclose(got, ref, rtol=2e-4, atol=2e-4 * scale,
                               err_msg="float64 mirror")


@requires_gpu
def test_rhs_ph_hadv_order5_specified_narrowing():
    # Specified-BC narrowing per the WRF <=6 branch, including the binding
    # quirk that the 4th-order x rows are gated on open_xs/open_xe ONLY
    # (module_big_step_utilities_em.F:1973, :1997): under specified BCs
    # columns ids+2/ide-3 receive NO x advection while rows jds+2/jde-3 DO
    # receive the 4th-order y stencil (:1819, 1850).
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.core.state import mu_at_u_faces, mu_at_v_faces
    from gpuwm.verify.npref import (np_rhs_ph_hadv, random_acoustic_state,
                                    s_meta)

    def hadv_of(seed, kill):
        s, cfg = random_acoustic_state(seed=seed, nz=8, ny=14, nx=16,
                                       hybrid_opt=2, hill_height=600.0)
        getattr(s, kill)[...] = 0.0            # isolate one direction
        cfg5 = dataclasses.replace(cfg, h_sca_adv_order=5, specified=True,
                                   spec_zone=1, relax_zone=4,
                                   spec_bdy_width=5)
        s.rph_t[...] = 0
        mu = s.total_mu()
        dycore.add_rhs_ph_hadv(s, cfg5, mu_at_u_faces(mu),
                               mu_at_v_faces(mu))
        got = cp.asnumpy(s.rph_t).astype(np.float64)
        meta = s_meta(s)
        ref = np_rhs_ph_hadv(meta, cfg5)
        scale = max(np.abs(got).max(), 1e-30)
        np.testing.assert_allclose(got, ref, rtol=2e-4, atol=2e-4 * scale,
                                   err_msg="float64 mirror")
        return got, meta, cfg

    # ---- x advection only (v = 0) ----
    got, meta, cfg = hadv_of(53, "v")
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    hx = got[1:nz]
    for i in (0, 2, nx - 3, nx - 1):           # spec rows + the x quirk
        assert np.all(hx[:, :, i] == 0.0), f"column {i} must get no x adv"
    assert np.abs(hx[:, :, 1]).max() > 0.0     # 2nd-order columns live
    assert np.abs(hx[:, :, nx - 2]).max() > 0.0
    assert np.abs(hx[:, :, 3:nx - 3]).max() > 0.0

    ph_f = meta["phb"] + meta["php"]
    ph_i = ph_f[1:nz]
    fcx, _ = _hand_hadv_coef(meta)
    # 2nd-order column i=1 (WRF :2024-2036): two-face form
    dph = ph_i - np.roll(ph_i, 1, axis=2)      # face i: ph(i)-ph(i-1)
    fx = fcx[:, :, :nx] * dph
    hadv2 = (0.25 / cfg.dx) * (np.roll(fx, -1, axis=2) + fx) \
        / meta["msft"][None]
    scale = np.abs(hadv2[:, :, 1]).max()
    np.testing.assert_allclose(hx[:, :, 1], -hadv2[:, :, 1], rtol=2e-4,
                               atol=2e-4 * scale, err_msg="2nd-order col 1")
    np.testing.assert_allclose(hx[:, :, nx - 2], -hadv2[:, :, nx - 2],
                               rtol=2e-4, atol=2e-4 * scale,
                               err_msg=f"2nd-order col {nx - 2}")

    # ---- y advection only (u = 0) ----
    got, meta, cfg = hadv_of(59, "u")
    hy = got[1:nz]
    for j in (0, ny - 1):                      # spec rows only
        assert np.all(hy[:, j, :] == 0.0), f"row {j} must get no y adv"
    ph_f = meta["phb"] + meta["php"]
    ph_i = ph_f[1:nz]
    _, fcy = _hand_hadv_coef(meta)
    csum = fcy[:, :ny, :] + np.roll(fcy[:, :ny, :], -1, axis=1)
    w4 = np.array([-1.0, 8.0, 0.0, -8.0, 1.0]) / 12.0
    d4 = sum(w4[o + 2] * np.roll(ph_i, o, axis=1) for o in range(-2, 3))
    hadv4 = (0.25 / cfg.dy) * csum * d4 / meta["msft"][None]
    for j in (2, ny - 3):                      # 4th-order rows (y keeps
        scale = np.abs(hadv4[:, j, :]).max()   # them under specified)
        np.testing.assert_allclose(hy[:, j, :], -hadv4[:, j, :], rtol=2e-4,
                                   atol=2e-4 * scale,
                                   err_msg=f"4th-order row {j}")
    dph = ph_i - np.roll(ph_i, 1, axis=1)
    fy = fcy[:, :ny, :] * dph
    hadv2 = (0.25 / cfg.dy) * (np.roll(fy, -1, axis=1) + fy) \
        / meta["msft"][None]
    for j in (1, ny - 2):                      # 2nd-order rows
        scale = np.abs(hadv2[:, j, :]).max()
        np.testing.assert_allclose(hy[:, j, :], -hadv2[:, j, :], rtol=2e-4,
                                   atol=2e-4 * scale,
                                   err_msg=f"2nd-order row {j}")


def test_rhs_ph_order_config_surface():
    # Default stays the frozen order-2 path; the real74 production configs
    # select the reference's h_sca_adv_order=5 (Registry.EM_COMMON:2872
    # default, unset in the reference namelist); order 5 with radiative
    # open boundaries is not wired.
    import dataclasses as dc

    from gpuwm.config import RunConfig
    from gpuwm.verify.cases.real74_d01 import config, phase3_config

    assert RunConfig(nx=8, ny=8, nz=8, dx=1.0, dy=1.0, ztop=1.0, dt=1.0,
                     run_seconds=0.0).h_sca_adv_order == 2
    assert phase3_config().h_sca_adv_order == 5
    assert config().h_sca_adv_order == 5


# ---- at rest over terrain (the money test) ----------------------------------

@pytest.mark.parametrize("hybrid_opt,nsteps", [(0, 1000), (2, 300)])
@requires_gpu
def test_at_rest_over_hill_stays_at_rest(hybrid_opt, nsteps):
    # Plan gate: 1000 m bell hill, stratified base, no wind - after 1000
    # steps max|w| < 0.02 m/s and max|u| < 0.02 m/s (spurious-flow bound).
    # The hybrid_opt=2 variant (shorter, same bound) guards the c1/c2
    # wiring of the slow tendencies, which the mirrors do not cover.
    from gpuwm.core.dycore import run_steps, stability_report
    from gpuwm.core.state import init_at_rest
    from gpuwm.core.terrain import bell_hill

    cfg = RunConfig(nx=64, ny=1, nz=48, dx=2000.0, dy=2000.0, ztop=16000.0,
                    dt=6.0, run_seconds=0.0, terrain_opt=1,
                    hill_height=1000.0, hill_halfwidth=10000.0,
                    hybrid_opt=hybrid_opt)
    terrain = bell_hill(cfg)
    vc = make_vertical_coord(cfg.nz, hybrid_opt=hybrid_opt, etac=cfg.etac)
    b = make_base_state(vc, theta_constant_N, p_surf=cfg.p_surf,
                        ztop=cfg.ztop, terrain_z=terrain)
    s = init_at_rest(cfg, vc, b, terrain_z=terrain)
    run_steps(s, cfg, n=nsteps)
    r = stability_report(s, cfg)
    assert not r["nan"]
    assert r["w_max"] < 0.02, r
    assert r["u_max"] < 0.02, r
