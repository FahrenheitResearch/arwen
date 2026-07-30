# tests/test_hypsometric.py
"""hypsometric_opt keying of the EOS diagnostic and the real-init recurrences.

Oracle: WRF v4.6.1 calc_p_rho_phi (dyn_em/module_big_step_utilities_em.F:
1025-1092, NONHYDROSTATIC) and module_initialize_real.F:3811-3825 /
:3970-3986 / :4002-4010.  The hand-derived references below are direct
per-point loop transcriptions of the Fortran expressions — deliberately NOT
calls into the vectorized mirror — so mirror bugs cannot self-certify.
"""

import textwrap

import numpy as np
import pytest

from gpuwm.config import RunConfig, load_config
from gpuwm.core import constants as c
from gpuwm.core.grid import make_base_state, make_vertical_coord
from gpuwm.verify.npref import np_calc_p_alpha


# ---------------------------------------------------------------------------
# Config key


def test_runconfig_default_is_opt1():
    cfg = RunConfig(nx=8, ny=4, nz=8, dx=100.0, dy=100.0, ztop=6400.0,
                    dt=0.5, run_seconds=1.0)
    assert cfg.hypsometric_opt == 1


def _write_toml(tmp_path, dynamics=""):
    f = tmp_path / "run.toml"
    f.write_text(textwrap.dedent(f"""
        [grid]
        nx = 512
        ny = 1
        nz = 64
        dx = 100.0
        dy = 100.0
        ztop = 6400.0
        [dynamics]
        dt = 0.5
        time_step_sound = 4
        {dynamics}
        [run]
        run_seconds = 900.0
    """))
    return f


@pytest.mark.parametrize("value", [1, 2])
def test_load_config_accepts_wrf_options(tmp_path, value):
    cfg = load_config(_write_toml(tmp_path, f"hypsometric_opt = {value}"))
    assert cfg.hypsometric_opt == value


@pytest.mark.parametrize("value", [0, 3, -1])
def test_load_config_rejects_non_wrf_options(tmp_path, value):
    with pytest.raises(ValueError, match="hypsometric_opt"):
        load_config(_write_toml(tmp_path, f"hypsometric_opt = {value}"))


def test_mirror_rejects_non_wrf_options():
    cfg, coord, base = _flat_setup()
    thp, php, mup, qv = _random_state(cfg, seed=3)
    with pytest.raises(ValueError, match="hypsometric_opt"):
        np_calc_p_alpha(thp, php, mup, base, coord, qv=qv, hypsometric_opt=3)


# ---------------------------------------------------------------------------
# Fixtures: flat sigma column and hybrid/terrain per-column base states.


def _flat_setup(nx=6, ny=4, nz=12):
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=100.0, dy=100.0, ztop=6400.0,
                    dt=0.5, run_seconds=1.0)
    coord = make_vertical_coord(nz)
    base = make_base_state(coord, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    return cfg, coord, base


def _hybrid_terrain_setup(nx=6, ny=4, nz=12):
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=100.0, dy=100.0, ztop=6400.0,
                    dt=0.5, run_seconds=1.0, hybrid_opt=2, terrain_opt=1)
    coord = make_vertical_coord(nz, hybrid_opt=2, etac=cfg.etac)
    x = np.arange(nx)[None, :] + np.zeros((ny, 1))
    terrain = 150.0 + 120.0 * np.sin(2.0 * np.pi * x / nx)
    base = make_base_state(coord, lambda z: 300.0 + 0.003 * np.asarray(z, float),
                           p_surf=cfg.p_surf, ztop=cfg.ztop,
                           terrain_z=terrain)
    return cfg, coord, base


def _random_state(cfg, seed=0, php_amp=25.0):
    rng = np.random.default_rng(seed)
    thp = 0.5 * rng.standard_normal((cfg.nz, cfg.ny, cfg.nx))
    php = php_amp * rng.standard_normal((cfg.nz + 1, cfg.ny, cfg.nx))
    mup = 30.0 * rng.standard_normal((cfg.ny, cfg.nx))
    qv = 1.0e-2 * rng.random((cfg.nz, cfg.ny, cfg.nx))
    return thp, php, mup, qv


def _fortran_calc_p_rho_phi(hypsometric_opt, thp, php, mup, base, coord, qv):
    """Loop transcription of WRF v4.6.1 calc_p_rho_phi (NONHYDROSTATIC).

    hypso_nonhydro branch (module_big_step_utilities_em.F:1025-1051)::

        opt 1:  al = -1/(c1(k)*muts+c2(k))
                     * (alb*(c1(k)*mu) + rdnw(k)*(ph(k+1)-ph(k)))
        opt 2:  pfu = c3f(k+1)*MUTS + c4f(k+1) + ptop
                pfd = c3f(k  )*MUTS + c4f(k  ) + ptop
                phm = c3h(k  )*MUTS + c4h(k  ) + ptop
                al  = (ph(k+1)-ph(k)+phb(k+1)-phb(k))/phm/LOG(pfd/pfu) - alb

    with WRF's ``mu`` the PERTURBATION dry mass, ``muts`` the total, and
    ``ph`` the perturbation geopotential.  Pressure from the moist_nonhydro
    use_theta_m=0 arm (F:1064-1075), returned as FULL pressure
    ``p0*temp**cpovcv`` (WRF stores ``p0*temp**cpovcv - pb``; the base
    ``pb`` cancels when the perturbation is re-totalled).
    """
    nz, ny, nx = thp.shape
    mub = np.asarray(base.mub, dtype=np.float64)

    def prof(arr, k, j, i):
        arr = np.asarray(arr, dtype=np.float64)
        return arr[k] if arr.ndim == 1 else arr[k, j, i]

    p = np.empty((nz, ny, nx))
    al = np.empty((nz, ny, nx))
    alt = np.empty((nz, ny, nx))
    for j in range(ny):
        for i in range(nx):
            mu_pert = mup[j, i]
            muts = (float(mub) if mub.ndim == 0 else mub[j, i]) + mu_pert
            for k in range(nz):
                alb_k = prof(base.alb, k, j, i)
                if hypsometric_opt == 1:
                    al_k = (-1.0 / (coord.c1h[k] * muts + coord.c2h[k])
                            * (alb_k * (coord.c1h[k] * mu_pert)
                               + coord.rdnw[k]
                               * (php[k + 1, j, i] - php[k, j, i])))
                else:
                    pfu = coord.c3f[k + 1] * muts + coord.c4f[k + 1] + base.p_top
                    pfd = coord.c3f[k] * muts + coord.c4f[k] + base.p_top
                    phm = coord.c3h[k] * muts + coord.c4h[k] + base.p_top
                    al_k = ((php[k + 1, j, i] - php[k, j, i]
                             + prof(base.phb, k + 1, j, i)
                             - prof(base.phb, k, j, i))
                            / phm / np.log(pfd / pfu) - alb_k)
                # moist_nonhydro, use_theta_m == 0:
                #   qvf  = 1 + rvovrd*qv
                #   temp = (r_d*(t0+t)*qvf)/(p0*(al+alb));  p = p0*temp**cpovcv
                th_total = prof(base.thb, k, j, i) + thp[k, j, i]
                qvf = 1.0 + c.RVOVRD * qv[k, j, i]
                temp = (c.RD * th_total * qvf) / (c.P0 * (al_k + alb_k))
                p[k, j, i] = c.P0 * temp ** (c.CP / c.CV)
                al[k, j, i] = al_k
                alt[k, j, i] = al_k + alb_k
    return p, al, alt


# ---------------------------------------------------------------------------
# Mirror vs hand-derived Fortran, both branches, flat and hybrid/terrain.


@pytest.mark.parametrize("setup", [_flat_setup, _hybrid_terrain_setup])
@pytest.mark.parametrize("opt", [1, 2])
def test_mirror_matches_hand_derived_fortran(setup, opt):
    cfg, coord, base = setup()
    thp, php, mup, qv = _random_state(cfg, seed=7)
    p_ref, al_ref, alt_ref = _fortran_calc_p_rho_phi(
        opt, thp, php, mup, base, coord, qv)
    p, al, alt = np_calc_p_alpha(thp, php, mup, base, coord, qv=qv,
                                 hypsometric_opt=opt)
    np.testing.assert_allclose(alt, alt_ref, rtol=1e-11)
    np.testing.assert_allclose(al, al_ref, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(p, p_ref, rtol=1e-11)


def test_opt2_actually_differs_from_opt1():
    """The key must switch formulations, not alias them.

    The two hypsometric forms agree only to O((dp/p)^2/12) per layer
    (TKW comment, module_big_step_utilities_em.F:1035-1040), so on a
    perturbed column the pressures must differ measurably but not wildly.
    """
    cfg, coord, base = _flat_setup()
    thp, php, mup, qv = _random_state(cfg, seed=11)
    p1, _, alt1 = np_calc_p_alpha(thp, php, mup, base, coord, qv=qv,
                                  hypsometric_opt=1)
    p2, _, alt2 = np_calc_p_alpha(thp, php, mup, base, coord, qv=qv,
                                  hypsometric_opt=2)
    dp = np.abs(p2 - p1)
    assert dp.max() > 0.1                       # visibly different (Pa)
    assert dp.max() < 500.0                     # same physics, small delta
    np.testing.assert_allclose(alt2, alt1, rtol=5e-3)


def test_mirror_opt1_is_bitwise_the_frozen_expressions():
    """opt=1 (and the default) must remain the exact pre-option mirror."""
    cfg, coord, base = _hybrid_terrain_setup()
    thp, php, mup, qv = _random_state(cfg, seed=5)

    # The frozen pre-hypsometric_opt mirror expressions, verbatim.
    def prof3(a):
        a = np.asarray(a, dtype=np.float64)
        return a[:, None, None] if a.ndim == 1 else a

    thb, phb, alb = prof3(base.thb), prof3(base.phb), prof3(base.alb)
    rdnw = np.asarray(coord.rdnw, dtype=np.float64)[:, None, None]
    c1h = np.asarray(coord.c1h, dtype=np.float64)[:, None, None]
    c2h = np.asarray(coord.c2h, dtype=np.float64)[:, None, None]
    th = thb + thp
    th = th * (1.0 + c.RVOVRD * np.asarray(qv, dtype=np.float64))
    mu = np.asarray(base.mub, dtype=np.float64) + mup
    ph = phb + php
    alt_frozen = -(ph[1:] - ph[:-1]) * rdnw / (c1h * mu + c2h)
    al_frozen = alt_frozen - alb
    p_frozen = c.P0 * ((c.RD * th) / (c.P0 * alt_frozen)) ** c.GAMMA

    for kwargs in ({}, {"hypsometric_opt": 1}):
        p, al, alt = np_calc_p_alpha(thp, php, mup, base, coord, qv=qv,
                                     **kwargs)
        assert np.array_equal(p, p_frozen)
        assert np.array_equal(al, al_frozen)
        assert np.array_equal(alt, alt_frozen)


# ---------------------------------------------------------------------------
# Ingest-side counterparts (module_initialize_real.F), CPU-only helpers.


def _real_base_inputs(nx=5, ny=4, nz=12):
    coord = make_vertical_coord(nz, hybrid_opt=2, etac=0.2)
    rng = np.random.default_rng(2)
    terrain = 200.0 + 150.0 * rng.random((ny, nx))
    return coord, terrain


def test_real_base_opt1_key_is_inert():
    from gpuwm.ingest.real import _make_real_base
    coord, terrain = _real_base_inputs()
    default = _make_real_base(coord, terrain, 10000.0, 290.0)
    keyed = _make_real_base(coord, terrain, 10000.0, 290.0,
                            hypsometric_opt=1)
    for name in ("mub", "pb", "alb", "thb", "phb"):
        assert np.array_equal(np.asarray(getattr(default, name)),
                              np.asarray(getattr(keyed, name))), name


def test_real_base_opt2_matches_hand_derived_fortran():
    """phb recurrence, module_initialize_real.F:3816-3822 verbatim."""
    from gpuwm.ingest.real import _make_real_base
    coord, terrain = _real_base_inputs()
    p_top = 10000.0
    base1 = _make_real_base(coord, terrain, p_top, 290.0, hypsometric_opt=1)
    base2 = _make_real_base(coord, terrain, p_top, 290.0, hypsometric_opt=2)
    # Everything but the geopotential integration is opt-independent.
    for name in ("mub", "pb", "alb", "thb"):
        assert np.array_equal(np.asarray(getattr(base1, name)),
                              np.asarray(getattr(base2, name))), name
    ny, nx = terrain.shape
    nz = coord.dnw.size
    phb_ref = np.empty((nz + 1, ny, nx))
    phb_ref[0] = c.G * terrain
    for j in range(ny):
        for i in range(nx):
            for kk in range(1, nz + 1):      # WRF DO k = 2,kte (1-based)
                pfu = coord.c3f[kk] * base2.mub[j, i] + coord.c4f[kk] + p_top
                pfd = (coord.c3f[kk - 1] * base2.mub[j, i]
                       + coord.c4f[kk - 1] + p_top)
                phm = (coord.c3h[kk - 1] * base2.mub[j, i]
                       + coord.c4h[kk - 1] + p_top)
                phb_ref[kk, j, i] = (phb_ref[kk - 1, j, i]
                                     + base2.alb[kk - 1, j, i] * phm
                                     * np.log(pfd / pfu))
    np.testing.assert_allclose(base2.phb, phb_ref, rtol=1e-12)
    # The two integrations differ (log-pressure vs d(eta)) at the per-layer
    # O((dp/p)^2/12) level; on this deliberately coarse 12-level column the
    # topmost layer has dp/p ~ 0.5, so the accumulated difference is O(1e3)
    # J/kg (~130 m) — measurable, same-physics scale.
    dphb = np.abs(base2.phb - base1.phb)
    assert dphb.max() > 1.0
    assert dphb.max() < 3000.0


def test_fp32_split_opt1_key_is_inert():
    from gpuwm.ingest.real import _fp32_geopotential_split, _make_real_base
    coord, terrain = _real_base_inputs()
    base = _make_real_base(coord, terrain, 10000.0, 290.0)
    rng = np.random.default_rng(4)
    dry_mass = np.asarray(base.mub) + 100.0 * rng.standard_normal(
        terrain.shape)
    alpha = np.asarray(base.alb) * (1.0 + 0.02 * rng.standard_normal(
        base.alb.shape))
    default = _fp32_geopotential_split(base, coord, dry_mass, alpha)
    keyed = _fp32_geopotential_split(base, coord, dry_mass, alpha,
                                     hypsometric_opt=1)
    assert np.array_equal(default, keyed)


def test_fp32_split_opt2_preserves_opt2_diagnostic():
    """The quantizer must optimize the SAME operator the opt-2 EOS runs."""
    from gpuwm.ingest.real import _fp32_geopotential_split, _make_real_base
    coord, terrain = _real_base_inputs()
    p_top = 10000.0
    base = _make_real_base(coord, terrain, p_top, 290.0, hypsometric_opt=2)
    rng = np.random.default_rng(4)
    dry_mass = np.asarray(base.mub) + 100.0 * rng.standard_normal(
        terrain.shape)
    alpha = np.asarray(base.alb) * (1.0 + 0.02 * rng.standard_normal(
        base.alb.shape))
    php = _fp32_geopotential_split(base, coord, dry_mass, alpha,
                                   hypsometric_opt=2)
    # Re-diagnose alt with the kernel's opt-2 FP32 arithmetic.
    phb32 = np.asarray(base.phb, dtype=np.float32)
    c3f = np.asarray(coord.c3f, dtype=np.float32)
    c4f = np.asarray(coord.c4f, dtype=np.float32)
    c3h = np.asarray(coord.c3h, dtype=np.float32)
    c4h = np.asarray(coord.c4h, dtype=np.float32)
    mu32 = np.asarray(dry_mass, dtype=np.float32)
    pt32 = np.float32(p_top)
    total = np.asarray(phb32 + php, dtype=np.float32)
    worst = 0.0
    for k in range(coord.dnw.size):
        pfu = c3f[k + 1] * mu32 + c4f[k + 1] + pt32
        pfd = c3f[k] * mu32 + c4f[k] + pt32
        phm = c3h[k] * mu32 + c4h[k] + pt32
        dphi = np.asarray(total[k + 1] - total[k], dtype=np.float32)
        diagnosed = np.asarray(
            np.asarray(dphi / phm, dtype=np.float32) / np.log(pfd / pfu),
            dtype=np.float32).astype(np.float64)
        err = np.abs(diagnosed - alpha[k]) / alpha[k]
        worst = max(worst, float(err.max()))
    # A few ulps of FP32: the split cannot beat the quantization floor,
    # but it must sit on it.  Had the split optimized the opt-1 operator
    # instead, the opt-2 rediagnosis error would be the inter-operator
    # discrepancy, O((dp/p)^2/12) ~ 5e-4 — three orders larger.
    assert worst < 16.0 * np.finfo(np.float32).eps


# ---------------------------------------------------------------------------
# Device parity (controller-run; requires a healthy GPU).


@pytest.mark.gpu
def test_device_opt1_key_is_bitwise_inert():
    import cupy as cp
    from conftest import requires_gpu  # noqa: F401  (collection guard)
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.state import init_at_rest

    cfg, coord, base = _flat_setup(nx=16, ny=8, nz=16)
    s = init_at_rest(cfg, coord, base)
    thp, php, mup, _ = _random_state(cfg, seed=9)
    s.thp[...] = cp.asarray(thp, dtype=cp.float32)
    s.php[...] = cp.asarray(php, dtype=cp.float32)
    s.mup[...] = cp.asarray(mup, dtype=cp.float32)
    update_diagnostics(s)
    frozen = tuple(cp.asnumpy(a).copy() for a in (s.p, s.al, s.alt))
    update_diagnostics(s, hypsometric_opt=1)
    keyed = tuple(cp.asnumpy(a) for a in (s.p, s.al, s.alt))
    for a, b in zip(frozen, keyed):
        assert np.array_equal(a, b)


@pytest.mark.gpu
@pytest.mark.parametrize("opt", [1, 2])
def test_device_matches_mirror_both_options(opt):
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.state import init_at_rest

    cfg, coord, base = _hybrid_terrain_setup(nx=16, ny=8, nz=16)
    s = init_at_rest(cfg, coord, base)
    thp, php, mup, _ = _random_state(cfg, seed=13)
    s.thp[...] = cp.asarray(thp, dtype=cp.float32)
    s.php[...] = cp.asarray(php, dtype=cp.float32)
    s.mup[...] = cp.asarray(mup, dtype=cp.float32)
    update_diagnostics(s, hypsometric_opt=opt)
    p_ref, al_ref, alt_ref = np_calc_p_alpha(
        cp.asnumpy(s.thp).astype(np.float64),
        cp.asnumpy(s.php).astype(np.float64),
        cp.asnumpy(s.mup).astype(np.float64), base, coord,
        hypsometric_opt=opt)
    np.testing.assert_allclose(cp.asnumpy(s.p), p_ref, rtol=2e-5)
    np.testing.assert_allclose(cp.asnumpy(s.alt), alt_ref, rtol=2e-5)
    np.testing.assert_allclose(cp.asnumpy(s.al), al_ref, rtol=2e-5,
                               atol=2e-5)
