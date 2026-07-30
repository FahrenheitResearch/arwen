"""CPU-only WRF acoustic-column reference tests.

These tests deliberately import no CuPy and carry no ``gpu`` marker.  They
pin both WRF v4.6.1 ``top_lid`` branches in ``gpuwm.verify.npref`` so the
controller can exercise the source transcription without a device.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from gpuwm.config import RunConfig
from gpuwm.core.acoustic import prepare_moist_cq
from gpuwm.core import constants as c
from gpuwm.verify.npref import np_advance_uv, np_advance_w_phi, np_calc_cq


def _cfg(*, top_lid: bool, nx: int = 1, ny: int = 1) -> RunConfig:
    return RunConfig(
        nx=nx, ny=ny, nz=4, dx=1000.0, dy=1000.0, ztop=8000.0,
        dt=3.0, run_seconds=0.0, top_lid=top_lid)


def _zero_column(p_profile) -> tuple[dict, dict, dict]:
    """One flat dry column with every acoustic forcing initially zero."""
    nz = 4
    shape = (nz, 1, 1)
    fshape = (nz + 1, 1, 1)
    meta = {
        "c1h": np.ones(nz), "c2h": np.zeros(nz),
        "c1f": np.ones(nz + 1), "c2f": np.zeros(nz + 1),
        "mub2d": np.full((1, 1), 90000.0), "mup": np.zeros((1, 1)),
        "thb": np.full(nz, 300.0), "thp": np.zeros(shape),
        "p": np.asarray(p_profile, dtype=np.float64)[:, None, None],
        "alt": np.ones(shape),
        "phb": np.zeros(fshape), "php": np.zeros(fshape),
        "rdnw": np.array([-3.8, -4.2, -4.5, -5.0]),
        "rdn": np.array([0.0, -4.0, -4.35, -4.75]),
        "fnm": np.array([0.0, 0.55, 0.52, 0.50]),
        "fnp": np.array([0.0, 0.45, 0.48, 0.50]),
        "msft": np.ones((1, 1)), "ht": np.zeros((1, 1)),
        "cf1": 1.5, "cf2": -0.6, "cf3": 0.1,
        "rw_t": np.zeros(fshape), "rph_t": np.zeros(fshape),
        "w": np.zeros(fshape),
    }
    pp = {
        "mu_pp": np.zeros((1, 1)), "th_pp": np.zeros(shape),
        "ph_pp": np.zeros(fshape), "w_pp": np.zeros(fshape),
    }
    new = {
        "mu_pp": np.zeros((1, 1)), "th_pp": np.zeros(shape),
        "ww_pp": np.zeros(fshape), "u_pp": np.zeros((nz, 1, 2)),
        "v_pp": np.zeros((nz, 2, 1)),
    }
    return pp, new, meta


def test_npref_open_top_keeps_top_rhs_forcing_and_phi_update():
    """WRF advance_w F:1318,1366-1369,1420-1431,1460-1465."""
    dtau = 0.5
    cfg = _cfg(top_lid=False)
    pp, new, meta = _zero_column(np.zeros(cfg.nz))
    pp["w_pp"][-1, 0, 0] = 2.0
    meta["rph_t"][-1, 0, 0] = 3.0
    meta["rw_t"][-1, 0, 0] = 4.0

    w, ph = np_advance_w_phi(pp, new, meta, cfg, dtau)

    mut = meta["mub2d"][0, 0]
    rhs_top = dtau * (
        3.0 + 0.5 * c.G * (1.0 - cfg.epssm) * 2.0) / mut
    expected_w_top = 2.0 + dtau * 4.0
    expected_ph_top = (
        rhs_top
        + 0.5 * dtau * c.G * (1.0 + cfg.epssm)
        * expected_w_top / mut)
    np.testing.assert_allclose(w[-1, 0, 0], expected_w_top, rtol=0, atol=1e-14)
    np.testing.assert_allclose(ph[-1, 0, 0], expected_ph_top,
                               rtol=0, atol=1e-14)


def test_npref_open_top_uses_live_top_tridiagonal_coupling():
    """WRF calc_coef_w F:619-648, including the lid_flag top lower row."""
    dtau = 0.5
    cfg = _cfg(top_lid=False)
    pp, new, meta = _zero_column([90000.0, 70000.0, 50000.0, 30000.0])
    # WRF hybrid_opt=2 factors for nz=4, etac=0.2, p_top=5000 Pa.
    # This is deliberately non-degenerate at the open top: C_f(kde-1)
    # and C_f(kde) differ, so the top lower and diagonal denominators cannot
    # be conflated without this independently assembled solve detecting it.
    meta["c1h"] = np.array([1.400390625, 1.615234375,
                             0.951171875, 0.033203125])
    meta["c2h"] = np.array([-38037.109375, -58447.265625,
                             4638.671875, 91845.703125])
    meta["c1f"] = np.array([1.0, 1.5078125, 1.283203125, 0.4921875, 0.0])
    meta["c2f"] = np.array([0.0, -48242.1875, -26904.296875,
                             48242.1875, 95000.0])
    meta["rw_t"][-1, 0, 0] = 2.0

    w, ph = np_advance_w_phi(pp, new, meta, cfg, dtau)

    nz = cfg.nz
    mut = float(meta["mub2d"][0, 0])
    c2a = c.GAMMA * meta["p"][:, 0, 0] / meta["alt"][:, 0, 0]
    chm = meta["c1h"] * mut + meta["c2h"]
    cfm = meta["c1f"] * mut + meta["c2f"]
    np.testing.assert_array_equal(cfm[-2:], [92539.0625, 95000.0])
    cof = (0.5 * dtau * c.G * (1.0 + cfg.epssm)) ** 2
    matrix = np.zeros((nz, nz))
    for k in range(1, nz):
        row = k - 1
        matrix[row, row] = 1.0 + cof * meta["rdn"][k] * (
            meta["rdnw"][k] * c2a[k] / (chm[k] * cfm[k])
            + meta["rdnw"][k - 1] * c2a[k - 1]
            / (chm[k - 1] * cfm[k]))
        if k > 1:
            matrix[row, row - 1] = (
                -cof * meta["rdn"][k] * meta["rdnw"][k - 1]
                * c2a[k - 1] / (chm[k - 1] * cfm[k - 1]))
        matrix[row, row + 1] = (
            -cof * meta["rdn"][k] * meta["rdnw"][k]
            * c2a[k] / (chm[k] * cfm[k + 1]))
    matrix[-1, -2] = (
        -2.0 * cof * meta["rdnw"][-1] ** 2 * c2a[-1]
        / (chm[-1] * cfm[-2]))
    matrix[-1, -1] = (
        1.0 + 2.0 * cof * meta["rdnw"][-1] ** 2 * c2a[-1]
        / (chm[-1] * cfm[-1]))
    forcing = np.zeros(nz)
    forcing[-1] = dtau * 2.0
    expected_w = np.linalg.solve(matrix, forcing)

    np.testing.assert_allclose(w[1:, 0, 0], expected_w,
                               rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        ph[1:, 0, 0],
        0.5 * dtau * c.G * (1.0 + cfg.epssm) * expected_w / cfm[1:],
        rtol=2e-13, atol=2e-13)
    assert np.any(w[1:-1, 0, 0] != 0.0), "top row must couple into the column"


def _rigid_npref_hash_fixture(*, ny: int = 1):
    """Deterministic rigid-lid NumPy-reference input used for a hash pin."""
    nz, nx = 4, 2
    cfg = _cfg(top_lid=True, nx=nx, ny=ny)
    rng = np.random.default_rng(20260718)
    shape, fshape = (nz, ny, nx), (nz + 1, ny, nx)
    meta = {
        "c1h": np.ones(nz), "c2h": np.zeros(nz),
        "c1f": np.ones(nz + 1), "c2f": np.zeros(nz + 1),
        "mub2d": np.full((ny, nx), 90000.0),
        "mup": rng.normal(0, 10, (ny, nx)),
        "thb": np.linspace(298.0, 310.0, nz),
        "thp": rng.normal(0, 0.1, shape),
        "p": np.broadcast_to(
            np.linspace(90000.0, 30000.0, nz)[:, None, None], shape).copy(),
        "alt": np.broadcast_to(
            np.linspace(0.9, 2.2, nz)[:, None, None], shape).copy(),
        "phb": np.broadcast_to(
            np.linspace(0.0, 80000.0, nz + 1)[:, None, None], fshape).copy(),
        "php": rng.normal(0, 2, fshape),
        "rdnw": np.array([-3.8, -4.2, -4.5, -5.0]),
        "rdn": np.array([0.0, -4.0, -4.35, -4.75]),
        "fnm": np.array([0.0, 0.55, 0.52, 0.50]),
        "fnp": np.array([0.0, 0.45, 0.48, 0.50]),
        "msft": np.ones((ny, nx)), "ht": np.zeros((ny, nx)),
        "cf1": 1.5, "cf2": -0.6, "cf3": 0.1,
        "rw_t": rng.normal(0, 0.2, fshape),
        "rph_t": rng.normal(0, 0.2, fshape), "w": np.zeros(fshape),
    }
    pp = {
        "mu_pp": rng.normal(0, 0.2, (ny, nx)),
        "th_pp": rng.normal(0, 0.2, shape),
        "ph_pp": rng.normal(0, 0.001, fshape),
        "w_pp": rng.normal(0, 0.1, fshape),
    }
    pp["ph_pp"][0] = 0.0
    new = {
        "mu_pp": rng.normal(0, 0.2, (ny, nx)),
        "th_pp": rng.normal(0, 0.2, shape),
        "ww_pp": rng.normal(0, 0.1, fshape),
        "u_pp": rng.normal(0, 0.1, (nz, ny, nx + 1)),
        "v_pp": rng.normal(0, 0.1, (nz, ny + 1, nx)),
    }
    # Additional advance_uv inputs are deterministic and appended after all
    # RNG draws so the rigid w/phi NumPy hash remains stable.
    pp["p_pp"] = np.linspace(-0.003, 0.004, nz * ny * nx).reshape(shape)
    pp["p_pp_old"] = 0.7 * pp["p_pp"]
    pp["al_pp"] = np.zeros(shape)
    pp["u_pp"] = new["u_pp"].copy()
    pp["v_pp"] = new["v_pp"].copy()
    meta["ru_t"] = np.linspace(-0.2, 0.3, nz * ny * (nx + 1)).reshape(
        nz, ny, nx + 1)
    meta["rv_t"] = np.linspace(0.25, -0.15, nz * (ny + 1) * nx).reshape(
        nz, ny + 1, nx)
    meta["pb"] = np.linspace(90000.0, 30000.0, nz)
    return cfg, pp, new, meta


def test_npref_rigid_top_reference_output_hash_is_stable():
    """Pin NumPy output only; this does not execute or byte-pin the device."""
    cfg, pp, new, meta = _rigid_npref_hash_fixture()
    w, ph = np_advance_w_phi(pp, new, meta, cfg, dtau=0.5)

    def digest(array):
        return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()

    assert digest(w) == "f44502b653f60f0c60306989c99c8aa90df53eb9e1ee9ff85fc4041a2ed34a93"
    assert digest(ph) == "80b1e02fd71e4b432e7260a73e63c507ee8a2c9de7c5fa761d06172bdc57d547"
    assert np.array_equal(w[-1], np.zeros_like(w[-1]))
    assert np.array_equal(ph[-1], np.zeros_like(ph[-1]))


def test_npref_calc_cq_sums_every_active_wrf_moist_mass_species():
    """WRF calc_cq F:787-906 plus pg_buoy_w F:2485-2489."""
    nz, ny, nx = 3, 2, 3
    base = np.arange(nz * ny * nx, dtype=np.float64).reshape(nz, ny, nx)
    moisture = {
        name: (index + 1.0) * 1.0e-4 + base * (index + 1.0) * 1.0e-6
        for index, name in enumerate(("qv", "qc", "qr", "qi", "qs", "qg"))
    }
    # Morrison number moments are WRF Registry ``scalar``, not ``moist``;
    # calc_cq must not add their incompatible #/kg values.
    moisture["nr"] = np.full_like(base, 1.0e8)

    cqu, cqv, cqw = np_calc_cq(moisture, mp_physics=10)

    qtot = sum(moisture[name]
               for name in ("qv", "qc", "qr", "qi", "qs", "qg"))
    expect_u_core = 1.0 / (1.0 + 0.5 * (qtot + np.roll(qtot, 1, axis=2)))
    expect_v_core = 1.0 / (1.0 + 0.5 * (qtot + np.roll(qtot, 1, axis=1)))
    expect_u = np.concatenate([expect_u_core, expect_u_core[:, :, :1]], axis=2)
    expect_v = np.concatenate([expect_v_core, expect_v_core[:, :1, :]], axis=1)
    expect_w = np.ones((nz + 1, ny, nx))
    expect_w[1:nz] = 1.0 / (1.0 + 0.5 * (qtot[1:] + qtot[:-1]))
    np.testing.assert_array_equal(cqu, expect_u)
    np.testing.assert_array_equal(cqv, expect_v)
    np.testing.assert_array_equal(cqw, expect_w)


def test_npref_calc_cq_nssl_includes_hail_mass_not_number_moments():
    """NSSL option 18 adds QH to WRF's moist package, but not QNHAIL."""
    shape = (3, 2, 3)
    qv = np.linspace(0.001, 0.018, np.prod(shape)).reshape(shape)
    moisture = {
        name: scale * qv
        for name, scale in zip(
            ("qv", "qc", "qr", "qi", "qs", "qg", "qh"),
            (1.0, 0.20, 0.10, 0.05, 0.03, 0.02, 0.01),
            strict=True)
    }
    moisture["qnh"] = np.full(shape, 1.0e8)

    cqu, cqv, cqw = np_calc_cq(moisture, mp_physics=18)

    qtot = sum(moisture[name]
               for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh"))
    expect_u_core = 1.0 / (1.0 + 0.5 * (qtot + np.roll(qtot, 1, axis=2)))
    expect_v_core = 1.0 / (1.0 + 0.5 * (qtot + np.roll(qtot, 1, axis=1)))
    expect_u = np.concatenate([expect_u_core, expect_u_core[:, :, :1]], axis=2)
    expect_v = np.concatenate([expect_v_core, expect_v_core[:, :1, :]], axis=1)
    expect_w = np.ones((shape[0] + 1, shape[1], shape[2]))
    expect_w[1:-1] = 1.0 / (1.0 + 0.5 * (qtot[1:] + qtot[:-1]))
    np.testing.assert_array_equal(cqu, expect_u)
    np.testing.assert_array_equal(cqv, expect_v)
    np.testing.assert_array_equal(cqw, expect_w)


def test_npref_calc_cq_mp0_uses_passive_vapor_only():
    """WRF Registry package for mp_physics=0 exposes only moist:qv."""
    qv = np.array([[[0.001, 0.002], [0.003, 0.004]],
                   [[0.005, 0.006], [0.007, 0.008]]])
    moisture = {
        "qv": qv,
        # gpuwm retains these transported fields in a moist mp=0 state, but
        # WRF does not expose them through the active Registry moist package.
        "qc": np.full_like(qv, 0.2),
        "qr": np.full_like(qv, 0.1),
    }

    cqu, cqv, cqw = np_calc_cq(moisture, mp_physics=0)

    expect_u_core = 1.0 / (1.0 + 0.5 * (qv + np.roll(qv, 1, axis=2)))
    expect_v_core = 1.0 / (1.0 + 0.5 * (qv + np.roll(qv, 1, axis=1)))
    expect_u = np.concatenate([expect_u_core, expect_u_core[:, :, :1]], axis=2)
    expect_v = np.concatenate([expect_v_core, expect_v_core[:, :1, :]], axis=1)
    expect_w = np.ones((qv.shape[0] + 1, *qv.shape[1:]))
    expect_w[1:-1] = 1.0 / (1.0 + 0.5 * (qv[1:] + qv[:-1]))
    np.testing.assert_array_equal(cqu, expect_u)
    np.testing.assert_array_equal(cqv, expect_v)
    np.testing.assert_array_equal(cqw, expect_w)


def test_prepare_moist_cq_mp0_launches_with_one_registry_species(monkeypatch):
    """The device cq sum receives WRF's passive-vapor n_moist count."""
    cfg = replace(_cfg(top_lid=False, nx=2, ny=2), moist=True,
                  mp_physics=0, moist_cq=True)
    shape = (cfg.nz, cfg.ny, cfg.nx)
    state = SimpleNamespace(
        qv=np.full(shape, 0.01), qc=np.full(shape, 0.2),
        qr=np.full(shape, 0.1), p=np.zeros(shape),
        scratch=lambda scratch_shape, _name: np.empty(scratch_shape),
    )
    launched = {}

    def kernel(_grid, _block, args):
        launched["n_mass"] = int(args[10])

    monkeypatch.setattr("gpuwm.core.acoustic.get_kernel",
                        lambda _module, _name: kernel)

    _cqu, _cqv, _cqw, use_cq = prepare_moist_cq(state, cfg)

    assert use_cq
    assert launched["n_mass"] == 1


def test_npref_moist_cq_scales_horizontal_acoustic_pgf_not_forcing():
    """WRF advance_uv F:868/942 multiplies dpxy, not ru_t/rv_t."""
    cfg, pp, _new, meta = _rigid_npref_hash_fixture(ny=2)
    qv = np.linspace(0.001, 0.018, cfg.nz * cfg.ny * cfg.nx).reshape(
        cfg.nz, cfg.ny, cfg.nx)
    meta.update(qv=qv, qc=0.2 * qv, qr=0.1 * qv,
                qi=0.05 * qv, qs=0.03 * qv, qg=0.02 * qv)
    off = replace(cfg, moist=True, mp_physics=10, moist_cq=False)
    on = replace(off, moist_cq=True)
    dtau = 0.5

    u_off, v_off = np_advance_uv(pp, meta, off, dtau)
    u_on, v_on = np_advance_uv(pp, meta, on, dtau)
    cqu, cqv, _ = np_calc_cq(meta, mp_physics=on.mp_physics)
    du = meta["ru_t"][:, :, :cfg.nx] - (
        u_off[:, :, :cfg.nx] - pp["u_pp"][:, :, :cfg.nx]) / dtau
    dv = meta["rv_t"][:, :cfg.ny, :] - (
        v_off[:, :cfg.ny, :] - pp["v_pp"][:, :cfg.ny, :]) / dtau
    expect_u = (pp["u_pp"][:, :, :cfg.nx]
                + dtau * (meta["ru_t"][:, :, :cfg.nx]
                           - cqu[:, :, :cfg.nx] * du))
    expect_v = (pp["v_pp"][:, :cfg.ny, :]
                + dtau * (meta["rv_t"][:, :cfg.ny, :]
                           - cqv[:, :cfg.ny, :] * dv))
    np.testing.assert_allclose(u_on[:, :, :cfg.nx], expect_u,
                               rtol=0, atol=2e-15)
    np.testing.assert_allclose(v_on[:, :cfg.ny, :], expect_v,
                               rtol=0, atol=2e-15)
    assert np.max(np.abs(u_on - u_off)) > 0.0
    assert np.max(np.abs(v_on - v_off)) > 0.0


def test_npref_moist_cqw_scales_implicit_vertical_rows_and_pgf():
    """WRF calc_coef_w F:632-641 and advance_w F:1405-1417."""
    dtau = 0.5
    cfg = replace(_cfg(top_lid=True), moist=True, mp_physics=1,
                  moist_cq=True)
    pp, new, meta = _zero_column([90000.0, 70000.0, 50000.0, 30000.0])
    pp["ph_pp"][1:, 0, 0] = np.array([0.2, -0.1, 0.15, 0.0])
    qv = np.array([0.002, 0.008, 0.014, 0.020])[:, None, None]
    meta.update(qv=qv, qc=0.25 * qv, qr=0.10 * qv)

    w, ph = np_advance_w_phi(pp, new, meta, cfg, dtau)

    nz = cfg.nz
    mut = float(meta["mub2d"][0, 0])
    c2a = c.GAMMA * meta["p"][:, 0, 0] / meta["alt"][:, 0, 0]
    _, _, cqw = np_calc_cq(meta, mp_physics=cfg.mp_physics)
    cq = cqw[:, 0, 0]
    cof = (0.5 * dtau * c.G * (1.0 + cfg.epssm)) ** 2
    matrix = np.zeros((nz, nz))
    forcing = np.zeros(nz)
    rhs = pp["ph_pp"][:, 0, 0]
    for k in range(1, nz):
        row = k - 1
        matrix[row, row] = 1.0 + cq[k] * cof * meta["rdn"][k] * (
            meta["rdnw"][k] * c2a[k] / mut ** 2
            + meta["rdnw"][k - 1] * c2a[k - 1] / mut ** 2)
        if k > 1:
            matrix[row, row - 1] = (
                -cq[k] * cof * meta["rdn"][k] * meta["rdnw"][k - 1]
                * c2a[k - 1] / mut ** 2)
        matrix[row, row + 1] = (
            -cq[k] * cof * meta["rdn"][k] * meta["rdnw"][k]
            * c2a[k] / mut ** 2)
        dph_up = 2.0 * (rhs[k + 1] - rhs[k])
        dph_dn = 2.0 * (rhs[k] - rhs[k - 1])
        forcing[row] = (
            cq[k] * 0.5 * dtau * c.G * meta["rdn"][k]
            * (c2a[k] * meta["rdnw"][k] / mut * dph_up
               - c2a[k - 1] * meta["rdnw"][k - 1] / mut * dph_dn))
    matrix[-1, -1] = (
        1.0 + 2.0 * cof * meta["rdnw"][-1] ** 2 * c2a[-1] / mut ** 2)
    expected_w = np.linalg.solve(matrix, forcing)
    expected_w[-1] = 0.0

    np.testing.assert_allclose(w[1:, 0, 0], expected_w,
                               rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        ph[1:, 0, 0], rhs[1:]
        + 0.5 * dtau * c.G * (1.0 + cfg.epssm) * expected_w / mut,
        rtol=2e-13, atol=2e-13)
