"""YSU PBL column scheme (Phase 3 Task 11).

The float64 mirror and CUDA kernel follow WRF v4.6.1
``phys/physics_mmm/bl_ysu.F90`` for the non-BEP path, including the
Registry-default top-down cloud mixing option.  These tests cover both surface-stability
branches, countergradient/entrainment mixing, the implicit column solve,
and the plan's six-hour convective-growth and stable-decay cases.
"""

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig, load_config


def _grid(nz=36, ztop=6000.0):
    """Hydrostatic synthetic column in the kernel's ``(nz, ny, nx)`` form."""
    z_ifc = np.linspace(0.0, ztop, nz + 1)
    z = 0.5 * (z_ifc[:-1] + z_ifc[1:])
    dz = np.diff(z_ifc)
    p_ifc = 100000.0 * np.exp(-z_ifc / 8200.0)
    p = 0.5 * (p_ifc[:-1] + p_ifc[1:])
    exner = (p / 100000.0) ** (287.0 / 1004.5)
    return z, dz, p, p_ifc, exner


def _column(theta=None, u=None, v=None, qv=None, *, nz=36, ztop=6000.0,
            hfx=0.0, qfx=0.0, ust=0.0, br=0.0, xland=1.0, dt=60.0):
    z, dz, p, p_ifc, exner = _grid(nz, ztop)
    theta = 300.0 + 0.003 * z if theta is None else np.asarray(theta, float)
    u = np.full(nz, 8.0) if u is None else np.asarray(u, float)
    v = np.zeros(nz) if v is None else np.asarray(v, float)
    qv = 0.010 * np.exp(-z / 2400.0) if qv is None else np.asarray(qv, float)
    zero = np.zeros(nz)
    return dict(
        u=u, v=v, theta=theta, qv=qv, qc=zero.copy(), qi=zero.copy(),
        p=p, p_interface=p_ifc, exner=exner, dz=dz,
        psfc=100000.0, znt=0.10, ust=ust, hfx=hfx, qfx=qfx,
        wspd=max(float(np.hypot(u[0], v[0])), 0.1), br=br,
        psim=np.log(max(z[0] / 0.10, 1.01)),
        psih=np.log(max(z[0] / 0.01, 1.01)), xland=xland,
        u10=float(u[0]), v10=float(v[0]), dt=dt,
    )


def _batched_columns(nz=32, ny=2, nx=5, seed=11):
    """Randomized stable and unstable columns for the CUDA comparison."""
    z, dz1, p1, pi1, ex1 = _grid(nz, 5000.0)
    rng = np.random.default_rng(seed)
    shp = (nz, ny, nx)
    theta = 298.0 + 0.004 * z[:, None, None]
    theta = np.broadcast_to(theta, shp).copy()
    theta += rng.normal(0.0, 0.08, shp)
    u = 5.0 + 0.002 * z[:, None, None] + rng.normal(0.0, 0.15, shp)
    v = -1.0 + 0.001 * z[:, None, None] + rng.normal(0.0, 0.15, shp)
    qv = 0.011 * np.exp(-z[:, None, None] / 2200.0)
    qv = np.broadcast_to(qv, shp).copy() * rng.uniform(0.95, 1.05, shp)
    qc = np.where((z[:, None, None] > 700.0)
                  & (z[:, None, None] < 1400.0), 2.0e-5, 0.0)
    qc = np.broadcast_to(qc, shp).copy()
    qi = np.zeros(shp)
    f3 = lambda a: np.ascontiguousarray(np.broadcast_to(a, shp), np.float32)
    f2 = lambda a: np.ascontiguousarray(a, np.float32)
    br = np.array([[-0.08, 0.12, -0.03, 0.20, -0.15],
                   [0.10, -0.05, 0.18, -0.12, 0.06]])
    hfx = np.where(br <= 0.0, 180.0, -25.0)
    qfx = np.where(br <= 0.0, 8.0e-5, 0.0)
    ust = np.where(br <= 0.0, 0.42, 0.24)
    # Include physics-off coupling in the device batch as well as the
    # dedicated CPU no-op regression.
    hfx[0, 0] = qfx[0, 0] = ust[0, 0] = 0.0
    wspd = np.hypot(u[0], v[0])
    return dict(
        u=f2(u), v=f2(v), theta=f2(theta), qv=f2(qv), qc=f2(qc),
        qi=f2(qi), p=f3(p1[:, None, None]),
        p_interface=np.ascontiguousarray(
            np.broadcast_to(pi1[:, None, None], (nz + 1, ny, nx)),
            np.float32),
        exner=f3(ex1[:, None, None]), dz=f3(dz1[:, None, None]),
        psfc=f2(np.full((ny, nx), 100000.0)),
        znt=f2(np.full((ny, nx), 0.10)), ust=f2(ust), hfx=f2(hfx),
        qfx=f2(qfx), wspd=f2(wspd), br=f2(br),
        psim=f2(np.full((ny, nx), 6.5)),
        psih=f2(np.full((ny, nx), 8.5)),
        xland=f2(np.where(np.indices((ny, nx))[1] == nx - 1, 2.0, 1.0)),
        u10=f2(u[0]), v10=f2(v[0]), dt=45.0,
    )


def test_bl_pbl_config_default_and_toml(tmp_path):
    base = dict(nx=8, ny=2, nz=12, dx=1000.0, dy=1000.0, ztop=6000.0,
                dt=10.0, run_seconds=60.0)
    assert RunConfig(**base).bl_pbl_physics == 0
    assert RunConfig(**base).ysu_topdown_pblmix == 1
    cfg_file = tmp_path / "ysu.toml"
    cfg_file.write_text(
        "[grid]\nnx=8\nny=2\nnz=12\ndx=1000.0\ndy=1000.0\n"
        "ztop=6000.0\n[run]\ndt=10.0\nrun_seconds=60.0\n"
        "[dynamics]\nbl_pbl_physics=1\nysu_topdown_pblmix=0\n"
        # YSU reads UST/HFX/QFX/WSPD/RMOL from the surface layer, so
        # validate_run_config refuses bl_pbl_physics without one -- the same
        # pair gpuwm/core/physics.py initialize_physics has always refused at
        # driver construction.  91 is classic MM5.
        "sf_sfclay_physics=91\n")
    assert load_config(cfg_file).bl_pbl_physics == 1
    assert load_config(cfg_file).ysu_topdown_pblmix == 0
    cfg_file.write_text(cfg_file.read_text().replace("bl_pbl_physics=1",
                                                      "bl_pbl_physics=7"))
    with pytest.raises(ValueError, match="bl_pbl_physics"):
        load_config(cfg_file)


@pytest.mark.gpu
@requires_gpu
def test_ysu_topdown_cloud_radiative_mixing_f90_fixture():
    """Hand fixture for v4.6.1 bl_ysu.F90:839-897 and 943-964."""
    import cupy as cp

    from gpuwm.core import constants as c
    from gpuwm.core.ysu import launch_ysu
    from gpuwm.verify.npref import np_ysu_column

    z, _, _, _, _ = _grid()
    theta = np.where(z < 900.0, 298.0 + 0.001 * z,
                     300.0 + 0.004 * (z - 900.0))
    args = _column(theta=theta, hfx=180.0, qfx=6.0e-5,
                   ust=0.40, br=-0.06, dt=60.0)
    cloud = (z > 400.0) & (z < 1800.0)
    args["qc"][cloud] = 2.0e-4
    args["rthraten"] = np.where(cloud, -2.0e-5, 0.0)

    ref = np_ysu_column(**args, ysu_topdown_pblmix=1)
    off = np_ysu_column(**args, ysu_topdown_pblmix=0)
    assert ref["cloudflg"] and not off["cloudflg"]
    assert off["topdown_radsum"] == 0.0 == off["wstar3_2"]

    # F90's radsum loop converts negative theta heating to W m-2 over
    # levels 1:kpbl-1.  Rebuild it independently from the fixture.
    kt = ref["kpbl"] - 2
    expected_radsum = 0.0
    for k in range(kt + 1):
        radflux = (args["rthraten"][k] * args["exner"][k] * c.CP / c.G
                   * (args["p_interface"][k]
                      - args["p_interface"][k + 1]))
        if radflux < 0.0:
            expected_radsum += abs(radflux)
    rho_top = (args["p"][kt]
               / (c.RD * args["theta"][kt] * args["exner"][kt]
                  * (1.0 + (c.RVOVRD - 1.0) * args["qv"][kt])))
    expected_wstar3_2 = (c.G / (args["theta"][kt]
                                      * (1.0 + (c.RVOVRD - 1.0)
                                         * args["qv"][kt]))
                         * (expected_radsum / rho_top / c.CP)
                         * ref["hpbl"])
    assert ref["topdown_radsum"] == pytest.approx(expected_radsum, rel=1e-13)
    assert ref["wstar3_2"] == pytest.approx(expected_wstar3_2, rel=1e-13)
    assert ref["topdown_radsum"] == pytest.approx(11.398764015571423)
    assert ref["wstar3_2"] == pytest.approx(0.28175264114865023)

    # Independent sign gate: radiative top-down mixing warms the mixed side
    # of the inversion and increases cooling immediately above it.  A shared
    # kernel/mirror sign error must not pass on agreement alone.
    kt = ref["kpbl"] - 2
    assert ref["dtheta"][kt] > off["dtheta"][kt]
    assert ref["dtheta"][kt + 1] < off["dtheta"][kt + 1]

    nz = args["theta"].size
    col3 = {name: cp.asarray(value[:, None, None], dtype=cp.float32)
            for name, value in args.items()
            if isinstance(value, np.ndarray) and value.shape == (nz,)}
    col3["p_interface"] = cp.asarray(
        args["p_interface"][:, None, None], dtype=cp.float32)
    surf = {name: cp.asarray([[args[name]]], dtype=cp.float32)
            for name in ("psfc", "znt", "ust", "hfx", "qfx", "wspd",
                         "br", "psim", "psih", "xland", "u10", "v10")}
    got = launch_ysu(
        **{name: col3[name] for name in
           ("u", "v", "theta", "qv", "qc", "qi", "p", "p_interface",
            "exner", "dz", "rthraten")},
        **surf, dt=args["dt"], ysu_topdown_pblmix=1)
    assert int(got["cloudflg"][0, 0]) == 1
    assert float(got["topdown_radsum"][0, 0]) == pytest.approx(
        expected_radsum, rel=3e-5)
    assert float(got["wstar3_2"][0, 0]) == pytest.approx(
        expected_wstar3_2, rel=3e-5)
    np.testing.assert_allclose(cp.asnumpy(got["dtheta"])[:, 0, 0],
                               ref["dtheta"], rtol=5e-4, atol=5e-7)


@pytest.mark.gpu
@requires_gpu
def test_ysu_topdown_revives_stable_surface_stratocumulus_pbl():
    """F90:732-768 revives a nocturnal liquid-unstable cloud layer."""
    import cupy as cp

    from gpuwm.core.ysu import launch_ysu
    from gpuwm.verify.npref import np_ysu_column

    nz, ztop = 24, 1500.0
    z, _, _, _, _ = _grid(nz, ztop)
    theta = 289.0 + 0.0015 * z
    args = _column(theta=theta, nz=nz, ztop=ztop, hfx=-10.0,
                   qfx=0.0, ust=0.25, br=0.05, dt=60.0)
    cloud = (z >= 350.0) & (z <= 800.0)
    args["qc"][cloud] = 6.0e-4
    args["rthraten"] = np.where(cloud, -2.0e-5, 0.0)

    ref = np_ysu_column(**args, ysu_topdown_pblmix=1)
    assert ref["kpbl"] == 14
    assert 780.0 < ref["hpbl"] < 820.0
    assert ref["cloudflg"]
    assert ref["topdown_radsum"] > 0.0
    assert ref["wstar3_2"] > 0.0

    col3 = {name: cp.asarray(value[:, None, None], dtype=cp.float32)
            for name, value in args.items()
            if isinstance(value, np.ndarray) and value.shape == (nz,)}
    col3["p_interface"] = cp.asarray(
        args["p_interface"][:, None, None], dtype=cp.float32)
    surf = {name: cp.asarray([[args[name]]], dtype=cp.float32)
            for name in ("psfc", "znt", "ust", "hfx", "qfx", "wspd",
                         "br", "psim", "psih", "xland", "u10", "v10")}
    got = launch_ysu(
        **{name: col3[name] for name in
           ("u", "v", "theta", "qv", "qc", "qi", "p", "p_interface",
            "exner", "dz", "rthraten")},
        **surf, dt=args["dt"], ysu_topdown_pblmix=1)
    assert int(got["kpbl"][0, 0]) == 14
    assert int(got["cloudflg"][0, 0]) == 1
    assert float(got["topdown_radsum"][0, 0]) > 0.0
    np.testing.assert_allclose(cp.asnumpy(got["dtheta"])[:, 0, 0],
                               ref["dtheta"], rtol=5e-4, atol=5e-7)


def test_zero_surface_exchange_is_exact_noop():
    """Physics-off surface coupling must not leak background mixing."""
    from gpuwm.verify.npref import np_ysu_column

    args = _column(theta=np.full(36, 300.0), u=np.full(36, 8.0),
                   v=np.full(36, -2.0), qv=np.full(36, 0.008),
                   hfx=0.0, qfx=0.0, ust=0.0, br=0.0)
    out = np_ysu_column(**args)
    for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi",
                 "exch_h", "exch_m"):
        np.testing.assert_array_equal(out[name], 0.0, err_msg=name)
    assert np.isfinite(out["hpbl"])
    assert out["kpbl"] >= 1


def test_convective_surface_heating_grows_pbl_over_six_hours():
    """A capped morning layer deepens into a realistic daytime PBL."""
    from gpuwm.verify.npref import np_ysu_column

    z, *_ = _grid()
    theta = np.where(z < 350.0, 298.0 + 0.001 * z,
                     300.4 + 0.004 * (z - 350.0))
    u = 6.0 + 0.001 * z
    qv = 0.011 * np.exp(-z / 2200.0)
    args = _column(theta=theta, u=u, qv=qv, hfx=240.0, qfx=7.0e-5,
                   ust=0.40, br=-0.08, dt=60.0)
    hpbl = []
    for _ in range(6 * 60):
        out = np_ysu_column(**args)
        hpbl.append(out["hpbl"])
        for field, tend in (("theta", "dtheta"), ("u", "du"),
                            ("v", "dv"), ("qv", "dqv"),
                            ("qc", "dqc"), ("qi", "dqi")):
            args[field] = args[field] + args["dt"] * out[tend]
        args["wspd"] = max(float(np.hypot(args["u"][0], args["v"][0])), 0.1)
    assert hpbl[0] < 800.0
    assert 900.0 <= hpbl[-1] <= 3500.0
    assert hpbl[-1] >= hpbl[0] + 500.0
    assert np.isfinite(args["theta"]).all()
    assert args["qv"].min() >= 0.0


def test_stable_boundary_layer_wind_decays():
    """Stable surface drag and local diffusion reduce low-level kinetic energy."""
    from gpuwm.verify.npref import np_ysu_column

    z, *_ = _grid()
    theta = 296.0 + 0.009 * z
    u = 11.0 + 0.0015 * z
    v = 2.0 - 0.0003 * z
    args = _column(theta=theta, u=u, v=v, hfx=-35.0, qfx=0.0,
                   ust=0.30, br=0.14, dt=30.0)
    ke0 = np.mean(args["u"][:4] ** 2 + args["v"][:4] ** 2)
    hpbl = []
    for _ in range(3 * 120):
        out = np_ysu_column(**args)
        hpbl.append(out["hpbl"])
        for field, tend in (("theta", "dtheta"), ("u", "du"),
                            ("v", "dv"), ("qv", "dqv")):
            args[field] = args[field] + args["dt"] * out[tend]
        args["wspd"] = max(float(np.hypot(args["u"][0], args["v"][0])), 0.1)
    ke1 = np.mean(args["u"][:4] ** 2 + args["v"][:4] ** 2)
    assert ke1 < 0.92 * ke0
    assert 0.0 < hpbl[-1] < 1000.0
    assert np.isfinite(args["theta"]).all()


@pytest.mark.gpu
@requires_gpu
def test_ysu_kernel_matches_float64_mirror():
    """Mixed stable/unstable batch, including cloud and ocean branches."""
    import cupy as cp
    from gpuwm.core.ysu import launch_ysu
    from gpuwm.verify.npref import np_ysu_column

    host = _batched_columns()
    dt = host.pop("dt")
    dev = {name: cp.asarray(value) for name, value in host.items()}
    got = launch_ysu(**dev, dt=dt)

    nz, ny, nx = host["theta"].shape
    ref = {name: np.empty((nz, ny, nx), np.float64)
           for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi",
                        "exch_h", "exch_m")}
    ref.update({name: np.empty((ny, nx), np.float64)
                for name in ("hpbl", "kpbl", "wstar", "delta")})
    column_names = ("u", "v", "theta", "qv", "qc", "qi", "p",
                    "p_interface", "exner", "dz")
    surface_names = ("psfc", "znt", "ust", "hfx", "qfx", "wspd", "br",
                     "psim", "psih", "xland", "u10", "v10")
    for j in range(ny):
        for i in range(nx):
            kw = {name: host[name][:, j, i].astype(np.float64)
                  for name in column_names}
            kw.update({name: float(host[name][j, i]) for name in surface_names})
            col = np_ysu_column(**kw, dt=dt)
            for name in ref:
                if ref[name].ndim == 3:
                    ref[name][:, j, i] = col[name]
                else:
                    ref[name][j, i] = col[name]

    for name in ("du", "dv", "dtheta", "dqv", "dqc", "dqi"):
        np.testing.assert_allclose(cp.asnumpy(got[name]), ref[name],
                                   rtol=3e-4,
                                   atol=5e-7 if name == "dtheta" else 2e-7,
                                   err_msg=name)
    for name in ("exch_h", "exch_m"):
        np.testing.assert_allclose(cp.asnumpy(got[name]), ref[name],
                                   rtol=2e-3, atol=2e-3, err_msg=name)
    for name in ("hpbl", "wstar", "delta"):
        np.testing.assert_allclose(cp.asnumpy(got[name]), ref[name],
                                   rtol=3e-4, atol=3e-2, err_msg=name)
    np.testing.assert_array_equal(cp.asnumpy(got["kpbl"]), ref["kpbl"])
    assert (ref["exch_h"][:, :, :] >= 0.0).all()
    assert (ref["exch_m"][:, :, :] >= 0.0).all()
