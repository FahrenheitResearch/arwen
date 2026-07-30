# tests/test_acoustic.py  (part 1 of this file; Task 11 appends)
import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = pytest.mark.gpu


@requires_gpu
def test_full_theta_frame_tendency_includes_wrf_300k_mass_offset():
    """Full-theta th_pp is algebraically equivalent to WRF's theta-300 form."""
    import dataclasses

    import cupy as cp

    from gpuwm.core.acoustic import acoustic_substep_explicit
    from gpuwm.verify.npref import random_acoustic_state

    state, cfg = random_acoustic_state(seed=83, ny=6, nx=12)
    cfg = dataclasses.replace(cfg, specified=True, spec_zone=1,
                              relax_zone=4, spec_bdy_width=5)
    state.th_pp[...] = 0.0
    state.rth_t[...] = 0.0
    state.rmu_t[...] = cp.float32(20.0)
    dtau = 0.5

    acoustic_substep_explicit(state, cfg, dtau=dtau, first=True)

    frame = np.zeros((cfg.ny, cfg.nx), dtype=bool)
    frame[[0, -1], :] = True
    frame[:, [0, -1]] = True
    expected = (dtau * 300.0 * cp.asnumpy(state.c1h)[:, None]
                * cp.asnumpy(state.rmu_t)[frame][None, :])
    np.testing.assert_allclose(cp.asnumpy(state.th_pp)[:, frame], expected,
                               rtol=2.0e-6, atol=2.0e-3)


@requires_gpu
def test_advance_uv_matches_reference():
    import cupy as cp
    from gpuwm.core.acoustic import acoustic_substep_explicit
    from gpuwm.verify.npref import (np_advance_uv, np_advance_mu_th,
                                    random_acoustic_state, s_meta)
    # random_acoustic_state builds a small consistent DomainState (nz=8, ny=2, nx=12)
    s, cfg = random_acoustic_state(seed=3)
    before = {n: cp.asnumpy(getattr(s, n)).astype(np.float64)
              for n in ("u_pp", "v_pp", "mu_pp", "th_pp", "p_pp",
                        "ph_pp", "p_pp_old")}
    acoustic_substep_explicit(s, cfg, dtau=1.0, first=False)
    u_ref, v_ref = np_advance_uv(before, s_meta(s), cfg, dtau=1.0)
    np.testing.assert_allclose(cp.asnumpy(s.u_pp), u_ref, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(s.v_pp), v_ref, rtol=1e-4, atol=1e-6)


@requires_gpu
def test_advance_mu_th_matches_reference():
    import cupy as cp
    from gpuwm.core.acoustic import acoustic_substep_explicit
    from gpuwm.verify.npref import (np_advance_uv, np_advance_mu_th,
                                    random_acoustic_state, s_meta)
    s, cfg = random_acoustic_state(seed=5)
    before = {n: cp.asnumpy(getattr(s, n)).astype(np.float64)
              for n in ("u_pp", "v_pp", "mu_pp", "th_pp", "p_pp",
                        "ph_pp", "p_pp_old")}
    acoustic_substep_explicit(s, cfg, dtau=0.7, first=False)
    meta = s_meta(s)
    # the mu/theta step consumes the momenta already updated by advance_uv
    u_new, v_new = np_advance_uv(before, meta, cfg, dtau=0.7)
    mu_ref, ww_ref, th_ref = np_advance_mu_th(
        {**before, "u_pp": u_new, "v_pp": v_new}, meta, cfg, dtau=0.7)
    np.testing.assert_allclose(cp.asnumpy(s.mu_pp), mu_ref,
                               rtol=2e-4, atol=1e-5)
    np.testing.assert_allclose(cp.asnumpy(s.ww_pp), ww_ref,
                               rtol=2e-4, atol=1e-5)
    np.testing.assert_allclose(cp.asnumpy(s.th_pp), th_ref,
                               rtol=2e-4, atol=1e-5)
    # the substep saves p_pp for the next substep's divergence damping
    np.testing.assert_array_equal(cp.asnumpy(s.p_pp_old),
                                  before["p_pp"].astype(np.float32))


@requires_gpu
def test_first_substep_skips_divergence_damping():
    # On the first acoustic substep there is no pressure history: two states
    # differing only in p_pp_old must produce identical updates.
    import cupy as cp
    from gpuwm.core.acoustic import acoustic_substep_explicit
    from gpuwm.verify.npref import random_acoustic_state
    s1, cfg = random_acoustic_state(seed=11)
    s2, _ = random_acoustic_state(seed=11)
    s2.p_pp_old[...] = 777.0                      # garbage damping history
    acoustic_substep_explicit(s1, cfg, dtau=1.0, first=True)
    acoustic_substep_explicit(s2, cfg, dtau=1.0, first=True)
    assert bool((s1.u_pp == s2.u_pp).all())
    assert bool((s1.v_pp == s2.v_pp).all())
    # and the history is (re)initialized for the following substep
    np.testing.assert_array_equal(cp.asnumpy(s2.p_pp_old),
                                  cp.asnumpy(s2.p_pp))


# ---- Task 11: vertically implicit w-phi solve -------------------------------

@requires_gpu
def test_w_phi_solve_matches_reference():
    import cupy as cp
    from gpuwm.core.acoustic import acoustic_substep
    from gpuwm.verify.npref import np_acoustic_substep, random_acoustic_state, snapshot
    s, cfg = random_acoustic_state(seed=7)
    before = snapshot(s)                       # all _pp + reference fields, float64
    acoustic_substep(s, cfg, dtau=0.5, first=False)
    ref = np_acoustic_substep(before, cfg, dtau=0.5)
    for name in ("u_pp", "v_pp", "w_pp", "ph_pp", "mu_pp", "th_pp", "p_pp"):
        np.testing.assert_allclose(cp.asnumpy(getattr(s, name)), ref[name],
                                   rtol=2e-4, atol=1e-6, err_msg=name)


@requires_gpu
def test_open_top_moist_cq_matches_reference():
    """Exercise the independent open-top + moist-cq device cross-product."""
    import dataclasses

    import cupy as cp

    from gpuwm.core.acoustic import acoustic_substep
    from gpuwm.verify.npref import (np_acoustic_substep, np_calc_cq,
                                    random_acoustic_state, snapshot)

    state, cfg = random_acoustic_state(
        seed=107, moist=True, mp_physics=10)
    cfg = dataclasses.replace(cfg, top_lid=False, moist_cq=True)
    assert cfg.top_lid is False
    assert cfg.moist_cq is True

    qv = np.linspace(
        0.001, 0.018, cfg.nz * cfg.ny * cfg.nx).reshape(
            cfg.nz, cfg.ny, cfg.nx)
    for scale, name in zip(
            (1.0, 0.20, 0.10, 0.05, 0.03, 0.02),
            ("qv", "qc", "qr", "qi", "qs", "qg"), strict=True):
        getattr(state, name)[...] = cp.asarray(scale * qv, dtype=cp.float32)

    before = snapshot(state)
    cqu, _cqv, _cqw = np_calc_cq(before, mp_physics=cfg.mp_physics)
    assert np.max(np.abs(cqu - 1.0)) > 1.0e-3

    acoustic_substep(state, cfg, dtau=0.5, first=False)
    ref = np_acoustic_substep(before, cfg, dtau=0.5)

    for name in ("u_pp", "v_pp", "w_pp", "ph_pp", "mu_pp", "th_pp",
                 "ww_pp", "p_pp", "al_pp"):
        np.testing.assert_allclose(
            cp.asnumpy(getattr(state, name)), ref[name],
            rtol=3.0e-4, atol=2.0e-5, err_msg=name)
    assert np.max(np.abs(cp.asnumpy(state.w_pp[-1]))) > 1.0e-6


@requires_gpu
def test_nssl_moist_cq_device_path_matches_reference():
    """Exercise option-18 QV..QH argument marshaling and acoustic use."""
    import dataclasses

    import cupy as cp

    from gpuwm.core.acoustic import (prepare_acoustic_coefficients,
                                     prepare_acoustic_substep_launch,
                                     prepare_moist_cq)
    from gpuwm.verify.npref import (np_acoustic_substep, np_calc_cq,
                                    random_acoustic_state, snapshot)

    state, cfg = random_acoustic_state(
        seed=109, moist=True, mp_physics=18)
    cfg = dataclasses.replace(cfg, top_lid=False, moist_cq=True)
    qv = np.linspace(
        0.001, 0.018, cfg.nz * cfg.ny * cfg.nx).reshape(
            cfg.nz, cfg.ny, cfg.nx)
    for scale, name in zip(
            (1.0, 0.20, 0.10, 0.05, 0.03, 0.02, 0.01),
            ("qv", "qc", "qr", "qi", "qs", "qg", "qh"), strict=True):
        getattr(state, name)[...] = cp.asarray(scale * qv, dtype=cp.float32)

    before = snapshot(state)
    expected_cq = np_calc_cq(before, mp_physics=18)
    device_cq = prepare_moist_cq(state, cfg)
    assert device_cq[3] is True
    for got, expected in zip(device_cq[:3], expected_cq, strict=True):
        np.testing.assert_allclose(
            cp.asnumpy(got), expected, rtol=3.0e-6, atol=2.0e-7)

    coefficients = prepare_acoustic_coefficients(
        state, cfg, dtau=0.5, cq=device_cq)
    launch = prepare_acoustic_substep_launch(
        state, cfg, dtau=0.5, coefficients=coefficients)
    launch(first=False)
    cp.cuda.runtime.deviceSynchronize()
    ref = np_acoustic_substep(before, cfg, dtau=0.5)
    for name in ("u_pp", "v_pp", "w_pp", "ph_pp", "mu_pp", "th_pp",
                 "ww_pp", "p_pp", "al_pp"):
        np.testing.assert_allclose(
            cp.asnumpy(getattr(state, name)), ref[name],
            rtol=3.0e-4, atol=2.0e-5, err_msg=name)


@requires_gpu
def test_specified_frame_uses_only_boundary_tendencies_each_substep():
    """WRF solve_em.F specified-zone ordering inside the acoustic loop."""
    import dataclasses

    import cupy as cp

    from gpuwm.core.acoustic import acoustic_substep
    from gpuwm.verify.npref import (np_acoustic_substep,
                                    random_acoustic_state, snapshot)

    s, cfg = random_acoustic_state(seed=73, ny=6, nx=12)
    cfg = dataclasses.replace(cfg, specified=True, spec_zone=1,
                              relax_zone=4, spec_bdy_width=5)
    before = snapshot(s)
    dtau = 0.5
    acoustic_substep(s, cfg, dtau=dtau, first=False)
    ref = np_acoustic_substep(before, cfg, dtau=dtau)

    for name in ("u_pp", "v_pp", "w_pp", "ph_pp", "mu_pp", "th_pp",
                 "ww_pp", "p_pp", "al_pp"):
        np.testing.assert_allclose(cp.asnumpy(getattr(s, name)), ref[name],
                                   rtol=3e-4, atol=2e-5, err_msg=name)

    def frame(shape):
        ny, nx = shape[-2:]
        j, i = np.ogrid[:ny, :nx]
        return ((i < cfg.spec_zone) | (i >= nx - cfg.spec_zone)
                | (j < cfg.spec_zone) | (j >= ny - cfg.spec_zone))

    # advance_uv + spec_bdyupdate(u/v), then advance_mu_t +
    # spec_bdyupdate(mu/theta): acoustic terms are absent from the frame.
    for name, tendency in (("u_pp", "ru_t"), ("v_pp", "rv_t"),
                           ("mu_pp", "rmu_t"), ("th_pp", "rth_t")):
        got = cp.asnumpy(getattr(s, name))
        mask = frame(got.shape)
        want = before[name] + dtau * before[tendency]
        if name == "th_pp":
            want += (dtau * 300.0 * before["c1h"][:, None, None]
                     * before["rmu_t"][None])
        np.testing.assert_allclose(got[..., mask], want[..., mask],
                                   rtol=2e-6, atol=2e-5, err_msg=name)

    # The test is non-vacuous: an interior mass cell did receive acoustic
    # divergence in addition to the large-step tendency.
    pure_mu = before["mu_pp"] + dtau * before["rmu_t"]
    got_mu = cp.asnumpy(s.mu_pp)
    assert np.max(np.abs(got_mu[1:-1, 1:-1] - pure_mu[1:-1, 1:-1])) > 1.0e-4

    # WRF applies zero_grad_bdy to the coupled w perturbation before the
    # diagnostic pressure update on every substep.
    got_w = cp.asnumpy(s.w_pp)
    np.testing.assert_allclose(got_w[:, 0, 1:-1], got_w[:, 1, 1:-1],
                               rtol=0.0, atol=0.0)
    np.testing.assert_allclose(got_w[:, -1, 1:-1], got_w[:, -2, 1:-1],
                               rtol=0.0, atol=0.0)
    np.testing.assert_allclose(got_w[:, 1:-1, 0], got_w[:, 1:-1, 1],
                               rtol=0.0, atol=0.0)
    np.testing.assert_allclose(got_w[:, 1:-1, -1], got_w[:, 1:-1, -2],
                               rtol=0.0, atol=0.0)


@requires_gpu
def test_wide_spec_zone_still_forms_mudf_on_second_row():
    """WRF advance_mu_t excludes one physical row, not all spec_zone rows."""
    import dataclasses

    import cupy as cp

    from gpuwm.core.acoustic import acoustic_substep_explicit
    from gpuwm.verify.npref import random_acoustic_state

    state, cfg = random_acoustic_state(seed=107, ny=8, nx=12)
    cfg = dataclasses.replace(
        cfg, specified=True, spec_zone=2, relax_zone=4, spec_bdy_width=6)
    state.u[...] = 0.0
    state.v[...] = 0.0
    state.u_pp[...] = 0.0
    state.v_pp[...] = 0.0
    state.rmu_t[...] = 0.0
    j, i = 3, 1                 # second x row: specified, but dynamically live
    state.u_pp[:, j, i + 1] = 1.0
    mudf = cp.full(state.mup.shape, cp.nan, dtype=cp.float32)

    acoustic_substep_explicit(
        state, cfg, dtau=0.5, first=False, mudf=mudf)

    # advance_uv precedes advance_mu_t, so compute WRF's DMDT from the
    # post-pressure-gradient face momenta actually consumed by advance_mu_t.
    u_pp = cp.asnumpy(state.u_pp)
    v_pp = cp.asnumpy(state.v_pp)
    dnw = cp.asnumpy(state.dnw)
    msft = float(cp.asnumpy(state.msft[j, i]))
    div = msft * msft * (
        (u_pp[:, j, i + 1] - u_pp[:, j, i]) / cfg.dx
        + (v_pp[:, j + 1, i] - v_pp[:, j, i]) / cfg.dy)
    expected = float(np.sum(dnw * div, dtype=np.float64))
    assert abs(expected) > 1.0e-6
    np.testing.assert_allclose(
        float(mudf[j, i].get()), expected,
        rtol=16.0 * np.finfo(np.float32).eps, atol=1.0e-10)
    # Only the true physical outer row takes WRF's boundary-only branch.
    assert float(mudf[j, 0].get()) == 0.0


@requires_gpu
def test_nested_frame_advances_w_from_rw_tend_each_substep():
    """solve_em.F:1602-1611 nested spec_bdyupdate(w_2,rw_tend,dts_rk)."""
    import dataclasses

    import cupy as cp

    from gpuwm.core.acoustic import acoustic_substep
    from gpuwm.verify.npref import (np_acoustic_substep,
                                    random_acoustic_state, snapshot)

    state, cfg = random_acoustic_state(seed=79, ny=8, nx=12)
    cfg = dataclasses.replace(
        cfg, specified=False, nested=True, spec_zone=1,
        relax_zone=4, spec_bdy_width=5)
    before = snapshot(state)
    dtau = 0.5
    acoustic_substep(state, cfg, dtau=dtau, first=False)
    ref = np_acoustic_substep(before, cfg, dtau=dtau)
    for name in ("u_pp", "v_pp", "w_pp", "ph_pp", "mu_pp", "th_pp",
                 "ww_pp", "p_pp", "al_pp"):
        np.testing.assert_allclose(
            cp.asnumpy(getattr(state, name)), ref[name],
            rtol=3.0e-4, atol=2.0e-5, err_msg=name)

    got_w = cp.asnumpy(state.w_pp)
    mask = np.zeros(got_w.shape[-2:], dtype=bool)
    mask[0, :] = mask[-1, :] = True
    mask[1:-1, 0] = mask[1:-1, -1] = True
    expected = before["w_pp"] + dtau * before["rw_t"]
    np.testing.assert_allclose(got_w[:, mask], expected[:, mask],
                               rtol=2.0e-6, atol=2.0e-5)
    # Non-vacuous against the former no-op and the root zero-gradient rule.
    assert np.max(np.abs(got_w[:, mask] - before["w_pp"][:, mask])) > 1.0e-5


@requires_gpu
def test_sound_wave_speed():
    # isothermal 300K atmosphere at rest; tiny p-pulse; acoustic-only stepping
    # must propagate at c = sqrt(GAMMA*RD*T) within 3%
    import cupy as cp
    from gpuwm.verify.npref import build_isothermal_rest_state
    from gpuwm.core.acoustic import run_acoustic_only
    s, cfg, dx = build_isothermal_rest_state(nx=400, nz=16, dx=100.0, T=300.0,
                                             pulse_amp=1.0, pulse_x0=20000.0,
                                             pulse_halfwidth=500.0)
    t = 40.0
    run_acoustic_only(s, cfg, dtau=0.05, n=int(t / 0.05))
    p = cp.asnumpy(s.p_pp)[s.p_pp.shape[0] // 2, 0, :]
    x = (np.arange(400) + 0.5) * dx
    # two fronts moving +-c from x0: find the rightward-moving peak
    xr = x[220 + int(np.argmax(p[220:]))]
    c_measured = (xr - 20000.0) / t
    c_theory = np.sqrt(1.4 * 287.0 * 300.0)    # ~347 m/s
    assert abs(c_measured - c_theory) / c_theory < 0.03
