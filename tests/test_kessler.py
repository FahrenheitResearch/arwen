# tests/test_kessler.py
"""Kessler warm-rain microphysics (Phase 2 Task 6).

Ported line-faithfully from the bundle's WRF v4.6.1
``phys/module_mp_kessler.F`` (one thread per column, split-fall-speed
sedimentation loop, Teten saturation adjustment, autoconversion above the
FILE's threshold c2 = 1 g/kg, accretion c3 = 2.2, rain evaporation, fall
speed 36.34*(rho*qr/1000)^0.1364*sqrt(rho0/rho)).  Tests: float64 mirror
behavior pins (conservation, latent heating, autoconversion threshold,
fall-speed formula), kernel-vs-mirror at rtol 1e-4, kernel conservation,
the saturated-column-rains-within-20-min gate through the public
``microphysics.apply`` path, the mp_physics=0 bitwise no-op, and the
dycore post-RK3 call-site wiring.
"""
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.core import constants as c


# ---------------------------------------------------------------------------
# helpers (float64 throughout)
# ---------------------------------------------------------------------------

def _teten_qvs(theta, pii):
    """Saturation mixing ratio exactly as module_mp_kessler.F diagnoses it."""
    temp = pii * theta
    pressure = 1.0e5 * pii ** (1004.0 / 287.0)
    es = 1000.0 * c.SVP1 * np.exp(c.SVP2 * (temp - c.SVPT0)
                                  / (temp - c.SVP3))
    return c.EP2 * es / (pressure - es)


def _column_grid(nz=40, ztop=12000.0, stretch=1.0):
    """Full/half-level heights and layer depths for a synthetic column."""
    zf = ztop * np.linspace(0.0, 1.0, nz + 1) ** stretch
    z = 0.5 * (zf[:-1] + zf[1:])
    dz8w = np.diff(zf)
    return z, dz8w


def _column_thermo(z):
    """US-standard-ish temperature/pressure -> (theta, pii, rho), float64."""
    T = np.maximum(300.0 - 0.0065 * z, 215.0)
    p = c.P0 * (T / 300.0) ** (c.G / (c.RD * 0.0065))
    pii = (p / c.P0) ** c.RCP
    theta = T / pii
    rho = p / (c.RD * T)
    return theta, pii, rho


def _dzk(z):
    """Sedimentation cell depths 1/rdzk exactly as the Fortran builds them:
    z(k+1)-z(k) below the top, z(kte)-z(kte-1) at the top."""
    dzk = np.empty_like(z)
    dzk[:-1] = z[1:] - z[:-1]
    dzk[-1] = z[-1] - z[-2]
    return dzk


def _colmass(qv, qc, qr, rho, z):
    """Column water mass (kg/m^2) in the scheme's own discrete measure."""
    return float(np.sum((np.asarray(qv, np.float64) + qc + qr)
                        * rho * _dzk(np.asarray(z, np.float64))))


def _random_columns(nz=40, ny=3, nx=8, seed=12):
    """A batch of plausible moist columns: saturated, cloudy, rainy, and dry
    cells all represented; returns float32 arrays shaped like the kernel's."""
    z1, dz1 = _column_grid(nz, stretch=1.15)
    theta1, pii1, rho1 = _column_thermo(z1)
    rng = np.random.default_rng(seed)
    shp = (nz, ny, nx)
    fac = 1.0 + 0.01 * rng.standard_normal((ny, nx))
    t = theta1[:, None, None] * fac[None]
    pii = pii1[:, None, None] * (1.0 + 0.002 * rng.standard_normal(shp))
    rho = rho1[:, None, None] * (1.0 + 0.01 * rng.standard_normal(shp))
    z = np.broadcast_to(z1[:, None, None], shp).copy()
    dz8w = np.broadcast_to(dz1[:, None, None], shp).copy()
    qvs = _teten_qvs(t, pii)
    qv = qvs * rng.uniform(0.3, 1.15, shp)
    qc = np.where(rng.random(shp) < 0.5, 0.0,
                  rng.uniform(0.0, 2.5e-3, shp))
    qr = np.where(rng.random(shp) < 0.5, 0.0,
                  rng.uniform(0.0, 4.0e-3, shp))
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
    return tuple(map(f32, (t, qv, qc, qr, rho, pii, z, dz8w)))


# ---------------------------------------------------------------------------
# float64 mirror behavior (CPU, no GPU required)
# ---------------------------------------------------------------------------

def test_mirror_heats_exactly_where_condensing():
    """theta increases exactly in supersaturated cells; a subsaturated cell
    with no condensate is bitwise untouched (product = ern = 0)."""
    from gpuwm.verify.npref import np_kessler_column
    z, dz8w = _column_grid(nz=30)
    theta, pii, rho = _column_thermo(z)
    qvs = _teten_qvs(theta, pii)
    rh = np.where(z < 4000.0, 1.2, 0.5)         # supersaturated below 4 km
    qv = rh * qvs
    qc = np.zeros_like(qv)
    qr = np.zeros_like(qv)
    t1, qv1, qc1, qr1, rainnc, rainncv = np_kessler_column(
        theta, qv, qc, qr, rho, pii, z, dz8w, dt=10.0)
    sup = rh > 1.0
    assert (t1[sup] > theta[sup]).all()          # latent heating
    assert (qc1[sup] > 0.0).all()                # condensate formed
    np.testing.assert_array_equal(t1[~sup], theta[~sup])   # untouched
    np.testing.assert_array_equal(qv1[~sup], qv[~sup])
    assert rainnc == 0.0 and rainncv == 0.0      # no rain without qr


def test_mirror_autoconversion_threshold_is_the_files():
    """The LOCAL module_mp_kessler.F has c2 = 0.001 kg/kg (1 g/kg): qc just
    below it produces no rain; qc above it does.  Saturated vapor keeps the
    saturation adjustment out of the way."""
    from gpuwm.verify.npref import np_kessler_column
    z, dz8w = _column_grid(nz=8, ztop=2000.0)
    theta, pii, rho = _column_thermo(z)
    qv = _teten_qvs(theta, pii)                  # exactly saturated
    zero = np.zeros_like(qv)
    for qc0, rains in ((8.0e-4, False), (1.5e-3, True)):
        _, _, _, qr1, _, _ = np_kessler_column(
            theta, qv, qc0 + zero, zero.copy(), rho, pii, z, dz8w, dt=30.0)
        assert (qr1.max() > 0.0) == rains


def test_mirror_fall_speed_formula():
    """First-split-step precip pins V = 36.34*(rho*qr/1000)^0.1364 (rho0 =
    rho = 1 makes the density factor exactly 1)."""
    from gpuwm.verify.npref import np_kessler_column
    nz, dt, qr0 = 6, 5.0, 2.0e-3
    z = 250.0 + 500.0 * np.arange(nz)
    dz8w = np.full(nz, 500.0)
    ones = np.ones(nz)
    qr = np.zeros(nz)
    qr[0] = qr0
    *_, rainnc, rainncv = np_kessler_column(
        300.0 * ones, np.zeros(nz), np.zeros(nz), qr, ones,
        0.9 * ones, z, dz8w, dt=dt)
    vt = 36.34 * (0.001 * qr0) ** 0.1364
    assert vt * dt / 500.0 < 0.75                # single split step
    np.testing.assert_allclose(rainncv, 1000.0 * qr0 * vt * dt / c.RHOWATER,
                               rtol=1e-12)
    np.testing.assert_allclose(rainnc, rainncv, rtol=1e-12)


def test_mirror_conserves_water_and_rains():
    """20 minutes of a saturated cloudy column: total water (column mass +
    rain through the floor, the scheme's own rho*dzk measure) conserved to
    1e-6 relative (float64: ~1e-14), and surface rain appears."""
    from gpuwm.verify.npref import np_kessler_column
    z, dz8w = _column_grid(nz=40, ztop=12000.0, stretch=1.1)
    theta, pii, rho = _column_thermo(z)
    qvs = _teten_qvs(theta, pii)
    qv = np.where(z < 3000.0, 1.02 * qvs, 0.5 * qvs)
    qc = np.where((z > 500.0) & (z < 3000.0), 2.0e-3, 0.0)
    qr = np.zeros_like(qv)
    t = theta.copy()
    m0 = _colmass(qv, qc, qr, rho, z)
    rainnc = 0.0
    dt, nsteps = 30.0, 40                        # 20 min of model time
    for _ in range(nsteps):
        t, qv, qc, qr, rainnc, rainncv = np_kessler_column(
            t, qv, qc, qr, rho, pii, z, dz8w, dt, rainnc=rainnc)
    m1 = _colmass(qv, qc, qr, rho, z) + rainnc * c.RHOWATER / 1000.0
    assert abs(m1 - m0) / m0 <= 1e-6
    assert rainnc > 0.05                         # surface rain fell (mm)
    assert min(qv.min(), qc.min(), qr.min()) >= 0.0


# ---------------------------------------------------------------------------
# CUDA kernel vs mirror
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@requires_gpu
def test_kessler_kernel_matches_mirror():
    """Column tests vs the float64 mirror at rtol 1e-4 (plan acceptance),
    including rain accumulation, over a batch of mixed-regime columns with
    the split-fall-speed loop engaged (nfall >= 2 somewhere)."""
    import cupy as cp
    from gpuwm.core.microphysics import launch_kessler
    from gpuwm.verify.npref import np_kessler_column
    t, qv, qc, qr, rho, pii, z, dz8w = _random_columns()
    nz, ny, nx = t.shape
    dt = 45.0                                    # crmax/0.75 ~ 1.5-2.3
    dev = {n: cp.asarray(a) for n, a in
           (("t", t), ("qv", qv), ("qc", qc), ("qr", qr), ("rho", rho),
            ("pii", pii), ("z", z), ("dz8w", dz8w))}
    rainnc = cp.zeros((ny, nx), cp.float32)
    rainncv = cp.zeros((ny, nx), cp.float32)
    launch_kessler(dev["t"], dev["qv"], dev["qc"], dev["qr"], dev["rho"],
                   dev["pii"], dev["z"], dev["dz8w"], rainnc, rainncv, dt)

    f64 = lambda a: a.astype(np.float64)
    ref = {n: np.zeros((nz, ny, nx)) for n in ("t", "qv", "qc", "qr")}
    ref_rnc = np.zeros((ny, nx))
    ref_rncv = np.zeros((ny, nx))
    split_seen = False
    for j in range(ny):
        for i in range(nx):
            col = lambda a: f64(a[:, j, i])
            crmax = (36.34 * np.maximum(0.0, 0.001 * col(rho) * col(qr))
                     ** 0.1364 * np.sqrt(col(rho)[0] / col(rho))
                     * dt / col(dz8w)).max()
            split_seen |= crmax / 0.75 > 1.0
            (ref["t"][:, j, i], ref["qv"][:, j, i], ref["qc"][:, j, i],
             ref["qr"][:, j, i], ref_rnc[j, i], ref_rncv[j, i]) = \
                np_kessler_column(col(t), col(qv), col(qc), col(qr),
                                  col(rho), col(pii), col(z), col(dz8w), dt)
    assert split_seen                            # nfall >= 2 exercised
    # FP32-FLOOR: absolute tolerance covers float32 cancellation at a
    # zero/near-zero reference; rtol=1e-4 remains the signal-scale gate.
    eps32 = np.finfo(np.float32).eps
    for n, scale in (("t", 300.0), ("qv", 0.02), ("qc", 0.02), ("qr", 0.02)):
        np.testing.assert_allclose(cp.asnumpy(dev[n]), ref[n], rtol=1e-4,
                                   atol=8.0 * eps32 * scale, err_msg=n)
    np.testing.assert_allclose(cp.asnumpy(rainnc), ref_rnc, rtol=1e-4,
                               atol=8.0 * eps32 * max(ref_rnc.max(), 1e-3))
    np.testing.assert_allclose(cp.asnumpy(rainncv), ref_rncv, rtol=1e-4,
                               atol=8.0 * eps32 * max(ref_rncv.max(), 1e-3))
    assert ref_rnc.max() > 0.0                   # rain actually fell


@pytest.mark.gpu
@requires_gpu
def test_kessler_kernel_conserves_water():
    """One kernel application: per-column qv+qc+qr + precip out the bottom
    conserved to 1e-6 relative (FP32 fields, FP64 sums)."""
    import cupy as cp
    from gpuwm.core.microphysics import launch_kessler
    t, qv, qc, qr, rho, pii, z, dz8w = _random_columns(seed=5)
    nz, ny, nx = t.shape
    dt = 45.0
    m0 = np.array([[_colmass(qv[:, j, i], qc[:, j, i], qr[:, j, i],
                             rho[:, j, i].astype(np.float64),
                             z[:, j, i]) for i in range(nx)]
                   for j in range(ny)])
    dt_ = {n: cp.asarray(a) for n, a in
           (("t", t), ("qv", qv), ("qc", qc), ("qr", qr), ("rho", rho),
            ("pii", pii), ("z", z), ("dz8w", dz8w))}
    rainnc = cp.zeros((ny, nx), cp.float32)
    rainncv = cp.zeros((ny, nx), cp.float32)
    launch_kessler(dt_["t"], dt_["qv"], dt_["qc"], dt_["qr"], dt_["rho"],
                   dt_["pii"], dt_["z"], dt_["dz8w"], rainnc, rainncv, dt)
    qv1, qc1, qr1 = (cp.asnumpy(dt_[n]) for n in ("qv", "qc", "qr"))
    m1 = np.array([[_colmass(qv1[:, j, i], qc1[:, j, i], qr1[:, j, i],
                             rho[:, j, i].astype(np.float64),
                             z[:, j, i]) for i in range(nx)]
                   for j in range(ny)])
    rain = cp.asnumpy(rainnc).astype(np.float64) * c.RHOWATER / 1000.0
    drift = np.abs(m1 + rain - m0) / m0
    assert drift.max() <= 1e-6


# ---------------------------------------------------------------------------
# public microphysics.apply path + dycore wiring
# ---------------------------------------------------------------------------

def _moist_state(nx=4, nz=40, mp=1, dx=1000.0, ztop=10000.0, dt=30.0):
    """Balanced moist state with a saturated cloudy lowest 3 km."""
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.state import DTYPE
    cfg = RunConfig(nx=nx, ny=1, nz=nz, dx=dx, dy=dx, ztop=ztop, dt=dt,
                    run_seconds=0.0, moist=True, mp_physics=mp)
    vc = make_vertical_coord(cfg.nz)
    b = make_base_state(vc, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                        p_surf=cfg.p_surf, ztop=cfg.ztop)
    s = init_moist_balanced(cfg, vc, b, lambda z: 0.0 * np.asarray(z, float))
    z = s.height_half()                          # (nz,)
    zc = z[:, None, None]
    qc = np.where((zc > 500.0) & (zc < 3000.0), 2.0e-3, 0.0)
    s.qc[...] = cp.asarray(np.broadcast_to(qc, s.p.shape), dtype=DTYPE)

    def state_qvs():
        th = b.thb[:, None, None] + cp.asnumpy(s.thp).astype(np.float64)
        pii = (cp.asnumpy(s.p).astype(np.float64) / c.P0) ** c.RCP
        return _teten_qvs(th, pii)

    # Self-consistent saturation: filling qv raises p through the theta_m
    # EOS (and with it the scheme's qvs), so iterate qv -> diagnostics to a
    # fixed point instead of using the dry-state qvs.
    for _ in range(6):
        qv = np.where(zc < 3000.0, 1.02 * state_qvs(), 0.3 * state_qvs())
        s.qv[...] = cp.asarray(np.broadcast_to(qv, s.p.shape), dtype=DTYPE)
        update_diagnostics(s)
    rh = cp.asnumpy(s.qv).astype(np.float64) / state_qvs()
    assert rh[np.broadcast_to(zc < 2500.0, rh.shape)].min() > 1.015
    return s, cfg


@pytest.mark.gpu
@requires_gpu
def test_saturated_column_rains_within_20_minutes():
    """Plan gate, through the public microphysics.apply surface: repeated
    application on a saturated cloudy column produces surface rain within
    20 min of model time; theta warms; moisture stays nonnegative."""
    import cupy as cp
    from gpuwm.core import microphysics
    s, cfg = _moist_state()
    th0 = float(s.thp.max())
    for _ in range(40):                          # 40 x 30 s = 20 min
        microphysics.apply(s, cfg, cfg.dt)
    rainnc = s.scratch((cfg.ny, cfg.nx), "mp_rainnc")
    assert float(rainnc.min()) > 0.05            # rain at every column (mm)
    assert float(s.thp.max()) > th0 + 0.1        # latent heating
    for q in (s.qv, s.qc, s.qr):
        arr = cp.asnumpy(q)
        assert np.isfinite(arr).all()
        assert arr.min() >= 0.0


@pytest.mark.gpu
@requires_gpu
def test_mp_physics_0_is_bitwise_noop_and_dispatch_raises():
    import cupy as cp
    from gpuwm.core import microphysics
    s, _ = _moist_state(mp=0)
    cfg0 = RunConfig(nx=4, ny=1, nz=40, dx=1000.0, dy=1000.0, ztop=10000.0,
                     dt=30.0, run_seconds=0.0, moist=True, mp_physics=0)
    fields = ("u", "v", "w", "thp", "php", "mup", "p", "al", "alt",
              "qv", "qc", "qr")
    before = {n: cp.asnumpy(getattr(s, n)).copy() for n in fields}
    microphysics.apply(s, cfg0, cfg0.dt)         # mp_physics=0: no-op
    for n in fields:
        np.testing.assert_array_equal(cp.asnumpy(getattr(s, n)), before[n],
                                      err_msg=n)
    with pytest.raises(ValueError, match="mp_physics"):
        bad = RunConfig(**{**cfg0.__dict__, "mp_physics": 7})
        microphysics.apply(s, bad, bad.dt)
    dry = RunConfig(**{**cfg0.__dict__, "moist": False, "mp_physics": 1})
    from gpuwm.core.state import DomainState
    s_dry = DomainState(dry)
    with pytest.raises(ValueError, match="moist"):
        microphysics.apply(s_dry, dry, dry.dt)


@pytest.mark.gpu
@requires_gpu
def test_dycore_post_rk3_slot_wired():
    """dycore.step runs Kessler after RK3 when mp_physics=1 (cloud forms
    from supersaturated vapor); with mp_physics=0 the same run keeps qc
    identically zero (advection of an all-zero scalar)."""
    import cupy as cp
    from gpuwm.core.dycore import run_steps, stability_report
    qc_max = {}
    for mp in (1, 0):
        s, cfg = _moist_state(nx=16, nz=20, mp=mp, dx=2000.0, dt=10.0)
        s.qc[...] = 0.0                          # vapor only; mp must make qc
        run_steps(s, cfg, n=5)
        r = stability_report(s, cfg)
        assert not r["nan"]
        qc_max[mp] = float(s.qc.max())
    assert qc_max[1] > 1.0e-4                    # condensation happened
    assert qc_max[0] == 0.0                      # slot gated off
