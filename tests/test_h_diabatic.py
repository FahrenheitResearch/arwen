# tests/test_h_diabatic.py
"""h_diabatic: microphysics latent heating retained as an RK dynamics
tendency (audit finding R7).

Oracle: WRF v4.6.1 --
  compute   moist_physics_prep_em saves pre-mp full theta in h_diabatic
            (module_big_step_utilities_em.F:5503/:5523-5526);
            moist_physics_finish_em clamps the scheme's theta increment,
            applies it ONCE directly, and stores mpten/dt (:5682-5746);
  apply     rk_addtend_dry adds (c1*mut+c2)*h_diabatic/msfty to the theta
            tendency on EVERY RK step (module_em.F:1076-1080);
  removal   small_step_finish subtracts the final RK step's accumulated
            contribution (module_small_step_em.F:408-426), so theta(t+dt)
            is heated exactly once, by the direct update;
  timing    microphysics runs AFTER the RK loop (solve_em.F:3604-3616 /
            :3659-4094), so step N's RK tendencies consume step N-1's
            heating; h_diabatic starts zero (start_em.F:643-644).

CPU tests pin the float64 mirrors (npref) and the config surface; the
GPU tests pin the production path: the direct-update/rate identity, the
no_mp_heating gate, the tendency-felt-but-net-zero step invariant, the
per-step recompute, and the one-step lag.
"""
import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.verify.npref import (MP_TEND_LIM_DEFAULT, NO_MP_HEATING_DEFAULT,
                                np_h_diabatic_tendency,
                                np_moist_physics_finish,
                                np_small_step_finish_theta)


# ---------------------------------------------------------------------------
# float64 mirror behavior (CPU, no GPU required)
# ---------------------------------------------------------------------------

def _random_thetas(seed=3, shape=(12, 4, 5), dt=60.0, amp=0.02):
    """Pre/post-microphysics full-theta pair with heating rates ~amp K/s."""
    rng = np.random.default_rng(seed)
    th_saved = 300.0 + 40.0 * rng.random(shape)
    mpten = amp * dt * rng.standard_normal(shape)
    return th_saved, th_saved + mpten, mpten


def test_finish_mirror_direct_update_equals_dt_times_rate():
    """moist_physics_finish_em bookkeeping identity: below the clamp the
    applied increment IS th_after - th_saved, and the stored rate times dt
    reproduces it exactly -- the pair the no-double-count argument rests
    on (theta gets mpten once; the RK loop gets mpten/dt)."""
    dt = 60.0
    th_saved, th_after, mpten = _random_thetas(dt=dt)
    th_new, hd = np_moist_physics_finish(th_after, th_saved, dt)
    # |mpten| ~ 1.2 K << mp_tend_lim*dt = 600 K: the clamp must be inert.
    np.testing.assert_allclose(th_new - th_saved, mpten, rtol=0, atol=1e-12)
    # dt*(mpten/dt) round-trips to one float64 ulp of the applied increment
    np.testing.assert_allclose(hd * dt, th_new - th_saved, rtol=1e-15)


def test_finish_mirror_hand_derived_two_level_column():
    """Hand-worked instance of moist_physics_finish_em, independent of the
    mirror's own expressions.  Binary-exact inputs so the Fortran arithmetic
    (module_big_step_utilities_em.F:5688 difference, :5743 direct add,
    :5745 rate) has ONE exact answer:

      dt = 32 s;  saved theta [300, 310] K;  post-mp theta [301, 309.5] K
      => mpten = [+1.0, -0.5] K  (:5688, exact)
      => theta_new = [301.0, 309.5] K  (:5743)
      => h_diabatic = mpten/32 = [0.03125, -0.015625] K/s  (:5745, exact
         because 32 is a power of two)

    The clamp (:5706-5707) is inert: |mpten| <= 1 K << 10*32 = 320 K
    (mp_tend_lim default 10 K/s, Registry.EM_COMMON:2642)."""
    th_saved = np.array([300.0, 310.0])
    th_after = np.array([301.0, 309.5])
    th_new, hd = np_moist_physics_finish(th_after, th_saved, 32.0)
    np.testing.assert_array_equal(th_new, np.array([301.0, 309.5]))
    np.testing.assert_array_equal(hd, np.array([0.03125, -0.015625]))


def test_finish_mirror_clamps_at_mp_tend_lim():
    """The theta increment saturates at +/- mp_tend_lim*dt
    (module_big_step_utilities_em.F:5706-5707) and the stored rate at
    +/- mp_tend_lim; the WRF Registry default is 10. K/s
    (Registry.EM_COMMON:2642)."""
    assert MP_TEND_LIM_DEFAULT == 10.0      # Registry.EM_COMMON:2642
    dt = 30.0
    th_saved = np.full((4,), 300.0)
    # increments of -3x, -0.5x, +0.5x, +3x the clamp
    lim = MP_TEND_LIM_DEFAULT * dt
    th_after = th_saved + np.array([-3.0, -0.5, 0.5, 3.0]) * lim
    th_new, hd = np_moist_physics_finish(th_after, th_saved, dt)
    np.testing.assert_array_equal(
        th_new - th_saved, np.array([-lim, -0.5 * lim, 0.5 * lim, lim]))
    np.testing.assert_array_equal(
        hd, np.array([-10.0, -5.0, 5.0, 10.0]))   # K/s at the WRF default
    # a tighter namelist clamp bites earlier
    th_new2, hd2 = np_moist_physics_finish(th_after, th_saved, dt,
                                           mp_tend_lim=1.0)
    assert np.abs(hd2).max() == 1.0
    assert np.abs(th_new2 - th_saved).max() == 1.0 * dt


def test_finish_mirror_no_mp_heating_disables_everything():
    """no_mp_heating=1: theta reverts to the pre-microphysics save (WRF
    leaves t_new untouched, :5775) and h_diabatic is zeroed (:5776); the
    WRF Registry default is 0 = heating ON (Registry.EM_COMMON:2630)."""
    assert NO_MP_HEATING_DEFAULT == 0       # Registry.EM_COMMON:2630
    dt = 60.0
    th_saved, th_after, _ = _random_thetas(seed=7, dt=dt)
    th_new, hd = np_moist_physics_finish(th_after, th_saved, dt,
                                         no_mp_heating=1)
    np.testing.assert_array_equal(th_new, th_saved)
    np.testing.assert_array_equal(hd, 0.0)


@pytest.mark.parametrize("with_msf", [False, True])
def test_rk_final_stage_removal_cancels_the_integrated_tendency(with_msf):
    """The no-double-count invariant, composed from the mirrors exactly as
    the step composes the production pieces: ns acoustic substeps
    integrate msfty*dtau*t_tend (WRF advance_mu_t,
    module_small_step_em.F:1142) with the rk_addtend_dry h_diabatic term
    (module_em.F:1078-1079), and the final-stage fold subtracts
    dt_stage*(c1h*mut+c2h)*h_diabatic (module_small_step_em.F:421).  Net
    theta change: ZERO (map factors and mass coupling cancel exactly);
    without the removal it would be dt_stage*h_diabatic -- which is what a
    naive double-heating implementation would leave behind."""
    rng = np.random.default_rng(11)
    nz, ny, nx = 10, 3, 4
    c1h = np.linspace(1.0, 0.2, nz)          # hybrid-like coefficients
    c2h = np.linspace(0.0, 3000.0, nz)
    thb = 300.0 + 50.0 * np.linspace(0.0, 1.0, nz)
    mu = 9.0e4 + 5.0e3 * rng.random((ny, nx))
    thp0 = rng.standard_normal((nz, ny, nx))
    hd = 0.01 * rng.standard_normal((nz, ny, nx))       # K/s
    msft = 1.0 + 0.2 * rng.random((ny, nx)) if with_msf else None

    ns, dtau = 6, 10.0                       # final RK stage: ns*dtau = dt
    t_tend = np_h_diabatic_tendency(hd, mu, c1h, c2h, msft=msft)
    myy = 1.0 if msft is None else msft[None]
    th_pp = np.zeros((nz, ny, nx))
    for _ in range(ns):                      # advance_mu_t: t += my*dtau*ft
        th_pp = th_pp + myy * dtau * t_tend

    # final stage (removal engaged): net h_diabatic contribution is zero
    thp_final = np_small_step_finish_theta(
        thp0, th_pp, mu, mu, thb, c1h, c2h,
        h_diabatic=hd, dt_stage=ns * dtau)
    np.testing.assert_allclose(thp_final, thp0, rtol=0, atol=1e-9)

    # without the removal the stage retains exactly dt*h_diabatic -- the
    # double-counted heating the mechanism must NOT leave in the state
    thp_no_removal = np_small_step_finish_theta(
        thp0, th_pp, mu, mu, thb, c1h, c2h)
    np.testing.assert_allclose(thp_no_removal - thp0, ns * dtau * hd,
                               rtol=0, atol=1e-9)
    assert np.abs(thp_no_removal - thp_final).max() > 0.1  # non-vacuous


def test_removal_cancels_with_mass_change_msf_and_fp32_inputs():
    """Production-shaped conservation contract (shadow-review hardening):
    non-uniform mu, acoustic mass CHANGE (mu_pp != 0, so the fold divides
    by mu_new != mu_s), active map factors, a non-binary-exact acoustic
    partition (dtau = 10/6 s), and FP32 inputs/arithmetic.

    float64 arm: the H contribution still cancels exactly -- both H terms
    live in the coupled numerator with the SAME stage mass (WRF's shared
    grid%mut, module_small_step_em.F:421-422) and cancel before the
    division by C_new -- while a removal wrongly coupled with mu_new (a
    mut/muts mix-up, the bug class this arm exists to catch) leaves an
    O(dt*c1h*mu_pp*H/C_new) residue.

    FP32 arm: the identical composition in float32 (the production dtype)
    leaves only rounding-level residue (< 1e-3 K), while omitting the
    removal leaves the full double-counted dt_stage*H = 0.1 K -- the
    cancellation is a real-arithmetic invariant that survives FP32 to
    rounding level (it is NOT bitwise, matching WRF's own source-level
    multiply/divide/repeated-add structure).
    """
    rng = np.random.default_rng(29)
    nz, ny, nx = 8, 4, 5
    c1h = np.linspace(1.0, 0.15, nz)
    c2h = np.linspace(0.0, 5000.0, nz)
    thb = 300.0 + 47.0 * np.linspace(0.0, 1.0, nz)
    # FP32-valued inputs shaped like production state fields
    f32 = np.float32
    mu_s = (9.0e4 + 8.0e3 * rng.random((ny, nx))).astype(f32)
    mu_pp = (400.0 * rng.standard_normal((ny, nx))).astype(f32)
    mu_new = mu_s + mu_pp
    msft = (1.0 + 0.25 * rng.random((ny, nx))).astype(f32)
    hd = (0.01 * rng.standard_normal((nz, ny, nx))).astype(f32)
    thp0 = rng.standard_normal((nz, ny, nx)).astype(f32)
    ns, dtau = 6, 10.0 / 6.0                 # dt_stage = 10 s, inexact dtau
    dt_stage = ns * dtau

    # --- float64 arm ------------------------------------------------------
    t_tend = np_h_diabatic_tendency(hd, mu_s, c1h, c2h, msft=msft)
    th_pp = np.zeros((nz, ny, nx))
    for _ in range(ns):                      # advance_mu_t: t += my*dtau*ft
        th_pp = th_pp + msft.astype(np.float64)[None] * dtau * t_tend
    got = np_small_step_finish_theta(thp0, th_pp, mu_s, mu_new, thb,
                                     c1h, c2h, h_diabatic=hd,
                                     dt_stage=dt_stage)
    want = np_small_step_finish_theta(thp0, np.zeros_like(th_pp), mu_s,
                                      mu_new, thb, c1h, c2h)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)

    # mut/muts mix-up arm: removal coupled with mu_new must NOT cancel
    c1h3 = c1h[:, None, None]
    c2h3 = c2h[:, None, None]
    thb3 = thb[:, None, None]
    hd64 = hd.astype(np.float64)
    num_wrong = ((c1h3 * mu_s.astype(np.float64)[None] + c2h3)
                 * (thb3 + thp0.astype(np.float64)) + th_pp
                 - dt_stage * (c1h3 * mu_new.astype(np.float64)[None]
                               + c2h3) * hd64)
    wrong = (num_wrong / (c1h3 * mu_new.astype(np.float64)[None] + c2h3)
             - thb3)
    assert np.abs(wrong - want).max() > 1.0e-5

    # --- FP32 arm ---------------------------------------------------------
    hdc = np.full((nz, ny, nx), 0.01, dtype=f32)   # crisp dt_stage*H = 0.1 K
    c1h32, c2h32 = c1h.astype(f32)[:, None, None], c2h.astype(f32)[:, None,
                                                                   None]
    thb32 = thb.astype(f32)[:, None, None]
    dtau32, dt32 = f32(dtau), f32(dt_stage)
    chm32 = c1h32 * mu_s[None] + c2h32
    tend32 = (chm32 * hdc) / msft[None]
    th_pp32 = np.zeros((nz, ny, nx), dtype=f32)
    for _ in range(ns):
        th_pp32 = th_pp32 + msft[None] * (dtau32 * tend32)
    ref32 = chm32 * (thb32 + thp0)               # H-free coupled reference
    den32 = c1h32 * mu_new[None] + c2h32
    base32 = ref32 / den32 - thb32               # no H anywhere
    got32 = (ref32 + th_pp32 - dt32 * chm32 * hdc) / den32 - thb32
    nr32 = (ref32 + th_pp32) / den32 - thb32     # removal omitted
    assert np.abs(got32 - base32).max() < 1.0e-3          # rounding only
    assert np.abs(nr32 - base32).max() > 0.05             # ~dt_stage*H left
    assert np.abs(nr32 - base32).max() > 50 * np.abs(got32 - base32).max()


def test_intermediate_stages_keep_the_heating():
    """Stages 1-2 (rk_step < rk_order) take WRF's no-removal branch
    (module_small_step_em.F:408-415): their provisional estimates DO carry
    the integrated heating -- that is how the stage-2/3 dynamics feel it."""
    rng = np.random.default_rng(19)
    nz, ny, nx = 6, 2, 3
    c1h, c2h = np.ones(nz), np.zeros(nz)
    thb = np.full(nz, 300.0)
    mu = np.full((ny, nx), 1.0e5)
    thp0 = rng.standard_normal((nz, ny, nx))
    hd = np.full((nz, ny, nx), 0.02)
    dtau, ns = 5.0, 2                        # stage 1: dt/3-style short leg
    t_tend = np_h_diabatic_tendency(hd, mu, c1h, c2h)
    th_pp = ns * dtau * t_tend
    thp_stage = np_small_step_finish_theta(thp0, th_pp, mu, mu, thb,
                                           c1h, c2h)   # dt_stage=0: stages 1-2
    np.testing.assert_allclose(thp_stage - thp0, ns * dtau * hd,
                               rtol=1e-12)


# ---------------------------------------------------------------------------
# config surface
# ---------------------------------------------------------------------------

def test_config_defaults_match_wrf_registry():
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1e3, dy=1e3, ztop=1e4, dt=10.0,
                    run_seconds=0.0)
    assert cfg.no_mp_heating == 0            # Registry.EM_COMMON:2630
    assert cfg.mp_tend_lim == 10.0           # Registry.EM_COMMON:2642 (K/s)


def test_config_validation_rejects_bad_values(tmp_path):
    import textwrap

    from gpuwm.config import load_config

    def write(dynamics):
        f = tmp_path / "run.toml"
        f.write_text(textwrap.dedent(f"""
            [grid]
            nx = 16
            ny = 16
            nz = 8
            dx = 1000.0
            dy = 1000.0
            ztop = 8000.0
            [dynamics]
            dt = 10.0
            {dynamics}
            [run]
            run_seconds = 0.0
        """))
        return f

    with pytest.raises(ValueError, match="no_mp_heating"):
        load_config(write("no_mp_heating = 2"))
    with pytest.raises(ValueError, match="mp_tend_lim"):
        load_config(write("mp_tend_lim = 0.0"))
    with pytest.raises(ValueError, match="mp_tend_lim"):
        load_config(write("mp_tend_lim = -5.0"))
    cfg = load_config(write("no_mp_heating = 1\nmp_tend_lim = 2.5"))
    assert cfg.no_mp_heating == 1 and cfg.mp_tend_lim == 2.5


# ---------------------------------------------------------------------------
# production path (GPU)
# ---------------------------------------------------------------------------

def _saturated_state(**kw):
    from test_kessler import _moist_state
    return _moist_state(**kw)


@pytest.mark.gpu
@requires_gpu
def test_apply_direct_update_equals_dt_times_h_diabatic():
    """Production finish identity on a really-condensing Kessler batch:
    the theta increment applied to the state is dt times the stored
    h_diabatic rate (WRF :5743 vs :5745), heating is nonzero where the
    column condenses, and cells the scheme never touches keep an exactly
    zero rate."""
    import cupy as cp
    from gpuwm.core import microphysics
    s, cfg = _saturated_state()
    thp0 = cp.asnumpy(s.thp).astype(np.float64)
    microphysics.apply(s, cfg, cfg.dt)
    dthp = cp.asnumpy(s.thp).astype(np.float64) - thp0
    hd = cp.asnumpy(s.h_diabatic).astype(np.float64)
    # FP32 state += rounding on theta' ~ O(1): atol 1e-5 K
    np.testing.assert_allclose(dthp, cfg.dt * hd, rtol=0, atol=1e-5)
    # saturated below ~2.5 km: condensational heating must register
    assert hd.max() > 1.0e-4                 # K/s, ~0.5 K adjustment / 30 s
    # dry, condensate-free top cells: bitwise-untouched theta => rate == 0
    z = s.height_half()
    top = np.broadcast_to((np.asarray(z) > 6000.0)[:, None, None], hd.shape)
    np.testing.assert_array_equal(hd[top], 0.0)


@pytest.mark.gpu
@requires_gpu
def test_apply_no_mp_heating_keeps_theta_and_zeroes_rate():
    """no_mp_heating=1 (WRF :5770-5782): theta is bitwise unchanged by the
    microphysics call, h_diabatic is zeroed, and the scheme's MOISTURE
    updates still land (condensation dries the supersaturated cells)."""
    import cupy as cp
    from gpuwm.core import microphysics
    s, cfg = _saturated_state()
    cfg1 = RunConfig(**{**cfg.__dict__, "no_mp_heating": 1})
    thp0 = cp.asnumpy(s.thp).copy()
    qv0 = cp.asnumpy(s.qv).copy()
    s.h_diabatic[...] = 0.123                # stale garbage must be cleared
    microphysics.apply(s, cfg1, cfg1.dt)
    np.testing.assert_array_equal(cp.asnumpy(s.thp), thp0)
    np.testing.assert_array_equal(cp.asnumpy(s.h_diabatic), 0.0)
    assert np.abs(cp.asnumpy(s.qv) - qv0).max() > 1.0e-5   # scheme ran


@pytest.mark.gpu
@requires_gpu
def test_step_feels_tendency_but_heats_net_zero_and_recomputes():
    """One dycore.step with a preset h_diabatic on an otherwise inert moist
    state (qv = 0: Kessler leaves theta bitwise, so the post-step recompute
    must return exactly zero):

    - the dynamics FEEL the heating (buoyancy spins up w vs the control),
    - the net theta increment is NOT dt*h_diabatic (the final-stage
      removal cancels the integrated tendency; only indirect dynamic
      response remains), and
    - h_diabatic is recomputed from THIS step's microphysics (zero here),
      pinning the per-step cadence.
    """
    import cupy as cp
    from gpuwm.core.dycore import step
    H = 0.005                                # K/s preset heating rate
    runs = {}
    for preset in (0.0, H):
        s, cfg = _saturated_state(nx=8, nz=24, dx=2000.0, dt=30.0)
        s.qv[...] = 0.0                      # Kessler inert: no mp heating
        s.qc[...] = 0.0
        z = np.asarray(s.height_half())
        layer = (z > 1000.0) & (z < 5000.0)
        hd = np.where(layer[:, None, None], preset, 0.0)
        s.h_diabatic[...] = cp.asarray(
            np.broadcast_to(hd, s.p.shape), dtype=cp.float32)
        step(s, cfg)
        runs[preset] = (cp.asnumpy(s.thp).astype(np.float64),
                        float(cp.abs(s.w).max()),
                        float(cp.abs(s.h_diabatic).max()))
    thp_c, wmax_c, _ = runs[0.0]
    thp_h, wmax_h, hdmax_h = runs[H]
    dt_h = 30.0 * H                          # 0.15 K if double-counted
    dthp = np.abs(thp_h - thp_c).max()
    # felt: heating excites vertical motion well above the control's noise
    assert wmax_h > 10.0 * wmax_c + 1.0e-4
    # net-zero: removal cancels the integrated tendency (indirect dynamic
    # response only); a no-removal implementation leaves ~dt*H = 0.15 K
    assert dthp < 0.25 * dt_h
    # and the change is NOT the double-counted dt*H in the heated layer
    layer3 = np.broadcast_to(layer[:, None, None], thp_h.shape)
    err_vs_double = np.abs((thp_h - thp_c)[layer3] - dt_h)
    assert err_vs_double.min() > 0.5 * dt_h
    # cadence: this step's (inert) microphysics overwrote the preset rate
    assert hdmax_h == 0.0


@pytest.mark.gpu
@requires_gpu
def test_step_one_step_lag_and_zero_init():
    """h_diabatic starts zero (WRF start_em.F:643-644) and step N's RK
    loop consumes step N-1's heating: zeroing the field between two steps
    changes step 2's result on a condensing state."""
    import cupy as cp
    from gpuwm.core.dycore import step

    def two_steps(zero_between):
        s, cfg = _saturated_state(nx=8, nz=30, dx=2000.0, dt=30.0)
        assert float(cp.abs(s.h_diabatic).max()) == 0.0   # zero init
        step(s, cfg)
        hd1 = float(cp.abs(s.h_diabatic).max())
        assert hd1 > 1.0e-4                  # step 1 diagnosed real heating
        if zero_between:
            s.h_diabatic[...] = 0.0
        step(s, cfg)
        return cp.asnumpy(s.thp)

    thp_lagged = two_steps(zero_between=False)
    thp_cut = two_steps(zero_between=True)
    assert np.abs(thp_lagged - thp_cut).max() > 0.0


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("with_msf", [False, True])
def test_add_h_diabatic_tendency_matches_mirror(with_msf):
    """Production rk_addtend_dry slot vs the float64 mirror, with and
    without map factors (module_em.F:1078-1079's /msfty)."""
    import cupy as cp
    from gpuwm.core.dycore import add_h_diabatic_tendency
    s, cfg = _saturated_state(nx=6, nz=12)
    rng = np.random.default_rng(23)
    if with_msf:
        ny, nx = cfg.ny, cfg.nx
        msfu = 1.0 + 0.2 * rng.random((ny, nx + 1))
        msfv = 1.0 + 0.2 * rng.random((ny + 1, nx))
        msfu[:, -1] = msfu[:, 0]
        msfv[-1, :] = msfv[0, :]
        s.set_map_coriolis(msft=1.0 + 0.2 * rng.random((ny, nx)),
                           msfu=msfu, msfv=msfv)
    hd = 0.01 * rng.standard_normal(s.p.shape)
    s.h_diabatic[...] = cp.asarray(hd, dtype=cp.float32)
    s.rth_t[...] = 0.0
    add_h_diabatic_tendency(s)
    want = np_h_diabatic_tendency(
        cp.asnumpy(s.h_diabatic).astype(np.float64),
        cp.asnumpy(s.total_mu()).astype(np.float64),
        cp.asnumpy(s.c1h).astype(np.float64),
        cp.asnumpy(s.c2h).astype(np.float64),
        msft=(cp.asnumpy(s.msft).astype(np.float64) if with_msf else None))
    np.testing.assert_allclose(cp.asnumpy(s.rth_t).astype(np.float64),
                               want, rtol=2e-5, atol=1e-4)
