"""Column-oracle input states for the km_opt=3 (3-D Smagorinsky) closure.

Two families, both in the engine's array layout (k, j, i; C-grid staggers):

* ``retained_cases``: the fourteen real initialization-state and evolved-wind
  crops already sealed under tests/data/wrf471_diffusion (16 x 12 columns,
  49 levels, dx = 3 km).  Only their ``input__*`` arrays are read; every WRF
  word is recomputed here from WRF 4.6.1.
* ``synthetic_cases``: LES-scale states (dx 50-200 m) built hydrostatically
  from analytic profiles plus seeded turbulence, one per regime: convective
  boundary layer, shallow cumulus, stable night with a low-level jet, cold
  snow/ice cloud, marine stratocumulus over ocean, high steep terrain, a
  coefficient-cap-binding vortex, high-latitude map factors, and four edge
  cases (zero flow, 1e-20 flow, supersaturated everywhere, qc exactly at the
  1e-5 threshold).

Everything is float32 and deterministic (fixed seeds).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

F = np.float32
RD, CP, G, P0, RV = 287.0, 1004.5, 9.81, 1.0e5, 461.6

# ---------------------------------------------------------------------------
# Retained real-state crops
# ---------------------------------------------------------------------------


def retained_cases(fixture_dir):
    """Yield (name, arrays, meta) for the sealed real-state crops."""
    fixture_dir = Path(fixture_dir)
    manifest = json.loads((fixture_dir / "deformation-manifest.json").read_text())
    for case in manifest["cases"]:
        path = fixture_dir / case["file"]
        blob = path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != case["sha256"]:
            raise ValueError(f"retained fixture changed: {case['file']}")
        with np.load(path) as data:
            arrays = {k.removeprefix("input__"): data[k].copy()
                      for k in data.files if k.startswith("input__")}
            meta = json.loads(str(data["meta_json"]))
        nz, ny, nx = arrays["alt"].shape
        arrays.pop("tke", None)
        # Surface forcing for the vertical driver (not in the source crop).
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        arrays["ust"] = (0.25 + 0.02 * ((ii + 2 * jj) % 7)).astype(F)
        arrays["hfx"] = (60.0 + 15.0 * ((3 * ii + jj) % 5)).astype(F)
        arrays["qfx"] = (1.0e-5 + 2.0e-6 * ((ii + jj) % 4)).astype(F)
        meta = {**meta, "dt": 1.0, "c_s": 0.25, "mix_upper_bound": 0.1,
                "family": "retained-real-state",
                "regime": "real-state " + case["name"],
                "source_fixture_sha256": case["sha256"]}
        yield "real_" + case["name"], arrays, meta


# ---------------------------------------------------------------------------
# Synthetic LES-scale states
# ---------------------------------------------------------------------------


def _std_pressure(z):
    """Standard-atmosphere pressure (Pa) for heights in metres."""
    z = np.asarray(z, dtype=np.float64)
    t0, lapse = 288.15, 0.0065
    p = np.where(z < 11000.0,
                 101325.0 * (1.0 - lapse * z / t0) ** (9.80665 / (287.05 * lapse)),
                 22632.06 * np.exp(-9.80665 * (z - 11000.0) / (287.05 * 216.65)))
    return p


def vertical_grid(nz, dz0, stretch, dzmax, terrain_mean=0.0):
    """Eta levels (WRF full levels znw) for a stretched LES grid.

    Levels follow the standard atmosphere above the mean terrain and the
    model top is the pressure of the top level, so the top layer keeps the
    grid's own depth.
    """
    dz = np.minimum(dz0 * stretch ** np.arange(nz), dzmax)
    zw = np.concatenate([[0.0], np.cumsum(dz)])
    pw = _std_pressure(zw + terrain_mean)
    ptop = float(np.round(pw[-1]))
    ps = pw[0]
    znw = (pw - ptop) / (ps - ptop)
    znw[0], znw[-1] = 1.0, 0.0
    znw = znw.astype(F)
    dnw = np.diff(znw).astype(F)                      # nz, negative
    znu = (0.5 * (znw[:-1] + znw[1:])).astype(F)
    dn = np.zeros(nz, F)
    dn[1:] = (0.5 * (dnw[1:] + dnw[:-1])).astype(F)
    fnm = np.zeros(nz, F)
    fnp = np.zeros(nz, F)
    fnm[1:] = (F(0.5) * dnw[:-1] / dn[1:]).astype(F)
    fnp[1:] = (F(0.5) * dnw[1:] / dn[1:]).astype(F)
    cof1 = (F(2.) * dn[1] + dn[2]) / (dn[1] + dn[2]) * dnw[0] / dn[1]
    cof2 = dn[1] / (dn[1] + dn[2]) * dnw[0] / dn[2]
    cf1, cf2, cf3 = F(fnp[1] + cof1), F(fnm[1] - cof1 - cof2), F(cof2)
    return dict(znw=znw, dnw=dnw, znu=znu, dn=dn, fnm=fnm, fnp=fnp, ptop=ptop,
                c1h=np.ones(nz, F), c2h=np.zeros(nz, F),
                c1f=np.ones(nz + 1, F), c2f=np.zeros(nz + 1, F),
                cf=(float(cf1), float(cf2), float(cf3)), zw_guess=zw)


def _qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / (p - es)


def hydrostatic(theta_of_z, qv_of_z, terrain, grid, ptop, dmu=None):
    """Integrate a column set; returns full p, alt, phi (w levels), mu."""
    ny, nx = terrain.shape
    znw, dnw, znu = (grid[k].astype(np.float64) for k in ("znw", "dnw", "znu"))
    nz = dnw.size
    ps = _std_pressure(terrain)
    mu = ps - ptop
    if dmu is not None:
        mu = mu + dmu
    pm = znu[:, None, None] * mu[None] + ptop
    zm = np.broadcast_to(0.5 * (grid["zw_guess"][:-1] + grid["zw_guess"][1:])[:, None, None]
                         + terrain[None], (nz, ny, nx)).copy()
    for _ in range(3):
        theta = theta_of_z(zm, terrain)
        qv = qv_of_z(zm, terrain)
        t = theta * (pm / P0) ** (RD / CP)
        alt = RD * t * (1.0 + 1.61 * qv) / pm / (1.0 + qv)   # inverse dry density
        phi = np.zeros((nz + 1, ny, nx))
        phi[0] = G * terrain
        for k in range(nz):
            phi[k + 1] = phi[k] - alt[k] * mu * dnw[k]
        zm = 0.5 * (phi[:-1] + phi[1:]) / G
    return dict(p=pm, alt=alt, phi=phi, mu=mu, theta=theta, qv=qv, t=t, zm=zm)


def _periodic_faces(a, bx, by):
    if not bx:
        a["u"][:, :, -1] = a["u"][:, :, 0]
        a["msfu"][:, -1] = a["msfu"][:, 0]
    if not by:
        a["v"][:, -1, :] = a["v"][:, 0, :]
        a["msfv"][-1, :] = a["msfv"][0, :]


def _assemble(name, *, nx, ny, nz, dx, dt, bx, by, theta_of_z, qv_of_z,
              wind, terrain=None, qc_of=None, qi_of=None, msf=None,
              ust=0.3, hfx=100.0, qfx=5e-5, seed=0, regime="",
              dz0=20.0, stretch=1.07, dzmax=300.0,
              turbulence=1.0, wscale=1.0, post=None):
    rng = np.random.default_rng(seed)
    terr = np.zeros((ny, nx)) if terrain is None else terrain
    grid = vertical_grid(nz, dz0, stretch, dzmax, float(terr.mean()))
    ptop = grid["ptop"]
    # Base state: dry reference sounding over the same terrain.
    base = hydrostatic(lambda z, h: 290.0 + 0.004 * z,
                       lambda z, h: 0.0 * z, terr, grid, ptop)
    dmu = 30.0 * rng.standard_normal((ny, nx))
    full = hydrostatic(theta_of_z, qv_of_z, terr, grid, ptop, dmu=dmu)
    zm = full["zm"]
    zw = full["phi"] / G
    a = {}
    a["phb"] = base["phi"].astype(F)
    a["php"] = (full["phi"] - base["phi"]).astype(F)
    a["p"] = full["p"].astype(F)
    a["alt"] = full["alt"].astype(F)
    a["mub2d"] = base["mu"].astype(F)
    a["mut"] = full["mu"].astype(F)
    a["thb"] = np.full((nz, ny, nx), 300.0, F)
    a["thp"] = (full["theta"] - 300.0).astype(F)
    qv = full["qv"].copy()
    qc = np.zeros_like(qv) if qc_of is None else qc_of(zm, full)
    qi = np.zeros_like(qv) if qi_of is None else qi_of(zm, full)
    a["qv"], a["qc"], a["qi"] = qv.astype(F), qc.astype(F), qi.astype(F)
    # Winds: mean profile + seeded eddies scaled with height.
    zu = np.concatenate([zm, zm[:, :, -1:]], axis=2)
    zv = np.concatenate([zm, zm[:, -1:, :]], axis=1)
    ub, vb = wind(zu, "u"), wind(zv, "v")
    damp_u = np.exp(-zu / 2500.0)
    damp_v = np.exp(-zv / 2500.0)
    a["u"] = (ub + turbulence * damp_u * rng.standard_normal(ub.shape)).astype(F)
    a["v"] = (vb + turbulence * damp_v * rng.standard_normal(vb.shape)).astype(F)
    w = wscale * np.exp(-zw / 2000.0) * rng.standard_normal(zw.shape)
    w[0] = 0.0 if terrain is None else w[0] * 0.2
    w[-1] = 0.0
    a["w"] = w.astype(F)
    for key, shape in (("msft", (ny, nx)), ("msfu", (ny, nx + 1)), ("msfv", (ny + 1, nx))):
        a[key] = (np.ones(shape) if msf is None else msf(shape)).astype(F)
    a["lat"] = np.full((ny, nx), 40.0, F)
    for key in ("znw", "dnw", "dn", "fnm", "fnp", "c1h", "c2h", "c1f", "c2f"):
        a[key] = grid[key].copy()
    a["ust"] = np.broadcast_to(np.asarray(ust, F), (ny, nx)).astype(F) + F(0.01) * rng.random((ny, nx)).astype(F)
    a["hfx"] = (np.broadcast_to(np.asarray(hfx, F), (ny, nx)) + 5.0 * rng.standard_normal((ny, nx))).astype(F)
    a["qfx"] = np.abs(np.broadcast_to(np.asarray(qfx, F), (ny, nx))
                      * (1.0 + 0.1 * rng.standard_normal((ny, nx)))).astype(F)
    if post is not None:
        post(a, rng)
    _periodic_faces(a, bx, by)
    cf1, cf2, cf3 = grid["cf"]
    meta = dict(nx=nx, ny=ny, nz=nz, dx=float(dx), dy=float(dx), dt=float(dt),
                bx=int(bx), by=int(by), cf1=cf1, cf2=cf2, cf3=cf3,
                p_top=float(ptop), c_s=0.25, c_k=0.15, mix_upper_bound=0.1,
                family="synthetic-les", regime=regime, seed=seed)
    return name, a, meta


def _ml_theta(zi, theta_ml, jump, lapse_above, sfc_excess=0.0, sfc_depth=50.0):
    def f(z, h):
        th = np.where(z < zi, theta_ml, theta_ml + jump * np.clip((z - zi) / 100.0, 0, 1)
                      + lapse_above * np.maximum(z - zi - 100.0, 0.0))
        return th + sfc_excess * np.exp(-z / sfc_depth)
    return f


def synthetic_cases():
    nx, ny, nz = 12, 10, 40
    out = []

    # 1. Dry convective boundary layer, periodic, dx 100 m.
    out.append(_assemble(
        "les_cbl", nx=nx, ny=ny, nz=nz, dx=100.0, dt=0.5, bx=0, by=0, seed=11,
        theta_of_z=_ml_theta(1000.0, 300.0, 5.0, 0.003, sfc_excess=2.0),
        qv_of_z=lambda z, h: 0.010 * np.where(z < 1000.0, 1.0, np.exp(-(z - 1000.0) / 1500.0)),
        wind=lambda z, s: (5.0 if s == "u" else 1.0) + 0.0 * z,
        turbulence=1.0, wscale=1.5, hfx=300.0, ust=0.4, qfx=8e-5,
        regime="convective boundary layer (superadiabatic surface layer, thermals)"))

    # 2. Shallow cumulus: saturated cloud layer with qc, dx 50 m, periodic.
    def qc_cu(z, full):
        return np.where((z > 1200.0) & (z < 1800.0), 8e-4 * np.sin(np.pi * (z - 1200.0) / 600.0), 0.0)

    def qv_cu(z, h):
        return np.where(z < 1200.0, 0.014, 0.0)

    def post_cu(a, rng):
        # Cloud layer at (super)saturation: qv = 1.02 qvs where qc > 0.
        t = (a["thp"] + a["thb"]) * (a["p"] / P0) ** (RD / CP)
        qs = _qsat(t.astype(np.float64), a["p"].astype(np.float64))
        cloud = a["qc"] > 0
        above = ~cloud & (a["qv"] == 0)
        a["qv"] = np.where(cloud, 1.02 * qs, np.where(above, 0.4 * qs, a["qv"])).astype(F)
    out.append(_assemble(
        "les_shallow_cumulus", nx=nx, ny=ny, nz=nz, dx=50.0, dt=0.3, bx=0, by=0, seed=12,
        theta_of_z=_ml_theta(1200.0, 298.0, 1.0, 0.0045, sfc_excess=1.0),
        qv_of_z=qv_cu, qc_of=qc_cu, wind=lambda z, s: (-8.0 + 0.002 * z) if s == "u" else 0.5 + 0.0 * z,
        turbulence=0.8, wscale=2.0, hfx=150.0, ust=0.3, qfx=1.2e-4, post=post_cu, dz0=25.0, stretch=1.05,
        regime="shallow cumulus (saturated cloud layer, qc up to 8e-4)"))

    # 3. Stable night with a low-level jet, open boundaries, dx 100 m.
    def llj(z, s):
        jet = 4.0 + 11.0 * np.exp(-((z - 300.0) / 150.0) ** 2)
        return jet if s == "u" else 0.3 * jet
    out.append(_assemble(
        "les_stable_night", nx=nx, ny=ny, nz=nz, dx=100.0, dt=0.5, bx=1, by=1, seed=13,
        theta_of_z=lambda z, h: 285.0 + 8.0 * (1 - np.exp(-z / 200.0)) + 0.004 * z,
        qv_of_z=lambda z, h: 0.006 * np.exp(-z / 2500.0), wind=llj,
        turbulence=0.25, wscale=0.1, hfx=-40.0, ust=0.15, qfx=1e-6, dz0=8.0, stretch=1.08,
        regime="stable nocturnal boundary layer (8 K inversion, 15 m/s low-level jet)"))

    # 4. Cold snow/ice cloud, x periodic / y open, dx 200 m.
    def qi_cold(z, full):
        return np.where((z > 1000.0) & (z < 4000.0), 1.5e-4 * np.sin(np.pi * (z - 1000.0) / 3000.0), 0.0)

    def qc_cold(z, full):
        return np.where((z > 1500.0) & (z < 1700.0), 2e-5, 0.0)

    def post_cold(a, rng):
        t = (a["thp"] + a["thb"]) * (a["p"] / P0) ** (RD / CP)
        qs = _qsat(t.astype(np.float64), a["p"].astype(np.float64))
        icy = a["qi"] > 0
        a["qv"] = np.where(icy, 0.98 * qs, 0.6 * qs).astype(F)
    out.append(_assemble(
        "les_snow_ice", nx=nx, ny=ny, nz=nz, dx=200.0, dt=1.0, bx=0, by=1, seed=14,
        theta_of_z=lambda z, h: 262.0 + 0.0055 * z, qv_of_z=lambda z, h: 0.002 + 0.0 * z,
        qc_of=qc_cold, qi_of=qi_cold, wind=lambda z, s: (12.0 + 0.003 * z) if s == "u" else -3.0 + 0.0 * z,
        turbulence=0.6, wscale=0.5, hfx=10.0, ust=0.35, qfx=5e-6, post=post_cold, dz0=30.0,
        regime="cold snow/ice cloud (262 K surface, qi to 1.5e-4, supercooled qc layer)"))

    # 5. Marine stratocumulus over ocean, open boundaries, dx 100 m.
    def qc_sc(z, full):
        return np.where((z > 600.0) & (z < 800.0), 4e-4 * (z - 600.0) / 200.0, 0.0)

    def post_sc(a, rng):
        t = (a["thp"] + a["thb"]) * (a["p"] / P0) ** (RD / CP)
        qs = _qsat(t.astype(np.float64), a["p"].astype(np.float64))
        a["qv"] = np.where(a["qc"] > 0, qs, a["qv"]).astype(F)
    out.append(_assemble(
        "les_marine_sc", nx=nx, ny=ny, nz=nz, dx=100.0, dt=0.5, bx=1, by=1, seed=15,
        theta_of_z=_ml_theta(800.0, 289.0, 10.0, 0.005), qc_of=qc_sc, post=post_sc,
        qv_of_z=lambda z, h: np.where(z < 800.0, 0.0105, 0.004),
        wind=lambda z, s: (6.0 if s == "u" else -4.0) + 0.0 * z,
        turbulence=0.5, wscale=0.6, hfx=15.0, ust=0.25, qfx=4e-5, dz0=15.0, stretch=1.07,
        regime="ocean: marine stratocumulus under a 10 K inversion"))

    # 6. High steep terrain: 2000 m plateau with a 3-D ridge, open, dx 100 m.
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    ridge = 2000.0 + 600.0 * np.exp(-((ii - 5.5) / 2.5) ** 2) * (1 + 0.3 * np.cos(2 * np.pi * jj / ny)) \
        + 40.0 * np.sin(1.3 * ii + 0.7 * jj)
    out.append(_assemble(
        "les_high_terrain", nx=nx, ny=ny, nz=nz, dx=100.0, dt=0.5, bx=1, by=1, seed=16,
        terrain=ridge, theta_of_z=lambda z, h: 300.0 + 0.0035 * (z - h) + 0.002 * z,
        qv_of_z=lambda z, h: 0.005 * np.exp(-(z - h) / 2000.0),
        wind=lambda z, s: (15.0 + 0.002 * z) if s == "u" else 2.0 + 0.0 * z,
        turbulence=1.0, wscale=1.0, hfx=200.0, ust=0.6, qfx=3e-5,
        regime="high terrain (2.0-2.6 km ridge, slopes to 0.5)"))

    # 7. Cap-binding vortex: 60 m/s swirl, dx 50 m, dt 1 s, saturated core.
    def vortex(z, s):
        return 0.0 * z

    def post_vortex(a, rng):
        nzv, nyv, nxv = a["alt"].shape
        for key, sx, sy in (("u", 0.0, 0.5), ("v", 0.5, 0.0)):
            f = a[key]
            jy, ix = np.meshgrid(np.arange(f.shape[1]) - sy, np.arange(f.shape[2]) - sx, indexing="ij")
            x, y = (ix - (nxv - 1) / 2.0) * 50.0, (jy - (nyv - 1) / 2.0) * 50.0
            r = np.hypot(x, y) + 1e-3
            vt = 60.0 * (r / 120.0) * np.exp(1.0 - r / 120.0)
            comp = -vt * y / r if key == "u" else vt * x / r
            f += (comp[None] * np.exp(-np.arange(nzv) / 25.0)[:, None, None]).astype(F)
        t = (a["thp"] + a["thb"]) * (a["p"] / P0) ** (RD / CP)
        qs = _qsat(t.astype(np.float64), a["p"].astype(np.float64))
        a["qv"] = (1.01 * qs).astype(F)
        a["qc"] = np.where(np.arange(nzv)[:, None, None] < 20, 5e-4, 0.0).astype(F) + 0 * a["qv"]
    out.append(_assemble(
        "les_vortex_caps", nx=nx, ny=ny, nz=nz, dx=50.0, dt=1.0, bx=0, by=0, seed=17,
        theta_of_z=lambda z, h: 300.0 + 0.003 * z, qv_of_z=lambda z, h: 0.015 + 0.0 * z,
        wind=vortex, turbulence=2.0, wscale=5.0, hfx=50.0, ust=1.2, qfx=1e-4, post=post_vortex,
        regime="tornado-like vortex (60 m/s swirl; mix_upper_bound caps bind)"))

    # 8. High-latitude map factors 1.3-1.6, dx 150 m, x open / y periodic.
    def hilat(shape):
        j, i = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        return 1.3 + 0.3 * (j / max(shape[0] - 1, 1)) + 0.02 * np.sin(i)
    out.append(_assemble(
        "les_map_hilat", nx=nx, ny=ny, nz=nz, dx=150.0, dt=0.6, bx=1, by=0, seed=18, msf=hilat,
        theta_of_z=_ml_theta(600.0, 270.0, 4.0, 0.006, sfc_excess=0.5),
        qv_of_z=lambda z, h: 0.003 * np.exp(-z / 2000.0),
        wind=lambda z, s: (10.0 if s == "u" else 5.0) + 0.001 * z,
        turbulence=0.7, wscale=0.7, hfx=60.0, ust=0.3, qfx=1e-5,
        regime="high latitude (map factors 1.3-1.6)"))

    # 9. Edge: zero flow, dry, open boundaries (deformation 0, floor branch).
    def post_zero(a, rng):
        for k in ("u", "v", "w"):
            a[k][...] = 0.0
        a["qv"][...] = 0.0
    out.append(_assemble(
        "edge_zero_flow", nx=nx, ny=ny, nz=nz, dx=100.0, dt=0.5, bx=1, by=1, seed=19,
        theta_of_z=lambda z, h: 295.0 + 0.004 * z, qv_of_z=lambda z, h: 0.0 * z,
        wind=lambda z, s: 0.0 * z, post=post_zero, hfx=0.0, ust=0.0, qfx=0.0,
        regime="edge: zero flow, dry (K = 1e-6 floor everywhere)"))

    # 10. Edge: 1e-20 flow (deformation products below the normal float32 range), periodic.
    def post_tiny(a, rng):
        for k in ("u", "v", "w"):
            a[k] *= F(1e-20)
    out.append(_assemble(
        "edge_tiny_flow", nx=nx, ny=ny, nz=nz, dx=100.0, dt=0.5, bx=0, by=0, seed=20,
        theta_of_z=lambda z, h: 300.0 + 0.0 * z, qv_of_z=lambda z, h: 0.008 + 0.0 * z,
        wind=lambda z, s: 3.0 + 0.0 * z, post=post_tiny, turbulence=1.0, wscale=1.0,
        regime="edge: 1e-20 m/s flow, neutral (products below the normal float32 range)"))

    # 11. Edge: supersaturated every level incl. the surface (saturated N2 surface branch).
    def post_supersat(a, rng):
        t = (a["thp"] + a["thb"]) * (a["p"] / P0) ** (RD / CP)
        qs = _qsat(t.astype(np.float64), a["p"].astype(np.float64))
        a["qv"] = (1.05 * qs).astype(F)
        a["qc"] = np.full_like(a["qv"], 2e-4)
        a["qi"] = np.where(t < 278.0, 5e-5, 0.0).astype(F)
    out.append(_assemble(
        "edge_supersaturated", nx=nx, ny=ny, nz=nz, dx=100.0, dt=0.5, bx=1, by=0, seed=21,
        theta_of_z=lambda z, h: 293.0 + 0.0045 * z, qv_of_z=lambda z, h: 0.012 + 0.0 * z,
        wind=lambda z, s: (7.0 if s == "u" else 2.0) + 0.0 * z, post=post_supersat,
        turbulence=0.8, wscale=1.0,
        regime="edge: supersaturated at every level (saturated surface N2 branch)"))

    # 12. Edge: qc exactly at the 1e-5 threshold on alternating columns, dry otherwise.
    def post_thresh(a, rng):
        nzv, nyv, nxv = a["alt"].shape
        jj2, ii2 = np.meshgrid(np.arange(nyv), np.arange(nxv), indexing="ij")
        pattern = ((ii2 + jj2) % 3)
        val = np.where(pattern == 0, F(1e-5), np.where(pattern == 1, np.nextafter(F(1e-5), F(0)), F(0)))
        a["qc"] = np.broadcast_to(val[None], (nzv, nyv, nxv)).astype(F).copy()
    out.append(_assemble(
        "edge_qc_threshold", nx=nx, ny=ny, nz=nz, dx=100.0, dt=0.5, bx=0, by=1, seed=22,
        theta_of_z=_ml_theta(900.0, 301.0, 3.0, 0.004, sfc_excess=1.0),
        qv_of_z=lambda z, h: 0.009 * np.exp(-z / 3000.0),
        wind=lambda z, s: (4.0 if s == "u" else -2.0) + 0.0 * z, post=post_thresh,
        regime="edge: qc exactly 1e-5 / one ULP below / zero (saturation switch)"))
    return out


def all_cases(fixture_dir):
    cases = list(retained_cases(fixture_dir)) + synthetic_cases()
    return cases


def columns(meta):
    return int(meta["nx"]) * int(meta["ny"])
