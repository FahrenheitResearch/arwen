"""Realistic column patches for the km_opt=2 (1.5-order TKE) column oracle.

km_opt=2 is not a pure column scheme: the deformation tensors read the
neighbouring columns, so every case is a small horizontal patch of columns
(8 x 8 = 64 by default) that share one regime.  The base state is WOOF's own
discretely hydrostatic base state (gpuwm.core.grid.make_base_state, numpy
only); the perturbation state is built here in float64 and rounded once to
float32.  Both sides of the comparison read the SAME float32 words.

Every array is in WOOF's device layout (k, j, i).  The WRF side converts to
WRF's (i, k, j) layout with halos in wrf_side.py.

Usage: python make_cases.py OUT_DIR   (writes case-<name>.npz and cases.json)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gpuwm.core.grid import make_base_state, make_vertical_coord

F32 = np.float32
G, RD, RV, CP, P0 = 9.81, 287.0, 461.6, 1004.5, 1.0e5
GAMMA = CP / (CP - RD)
NZ = 40

from arms import ARMS  # noqa: E402


def _theta_fn(points):
    """Piecewise-linear theta(z) through (z, theta) points."""
    zs = np.array([p[0] for p in points], dtype=np.float64)
    ts = np.array([p[1] for p in points], dtype=np.float64)

    def theta(z):
        z = np.asarray(z, dtype=np.float64)
        return np.where(z <= zs[-1], np.interp(z, zs, ts),
                        ts[-1] + (z - zs[-1]) * 0.0035)
    return theta


REGIMES = {
    # name: dict(sounding points, qv(z) surface g/kg + scale height,
    # cloud band, winds, tke profile, flux, grid)
    "convective_land": dict(
        dx=250.0, ztop=12000.0, psfc=97000.0, bound=0,
        theta=[(0, 303.0), (60, 300.6), (1500, 300.0), (1700, 305.5),
               (12000, 336.0)],
        qv0=12.0, qv_h=1600.0, qv_top=1700.0, cloud=(1350.0, 1650.0, 2.0e-4, 0.0),
        u=(5.0, 1.0), v=(2.0, 0.5), turb=1.5, wturb=2.0, bl=1500.0,
        tke=(1.5, 0.01), ust=0.45, hfx=320.0, msf=(1.0, 0.0)),
    "stable_night": dict(
        dx=1000.0, ztop=12000.0, psfc=99500.0, bound=0,
        theta=[(0, 281.0), (200, 289.0), (2000, 296.0), (12000, 330.0)],
        qv0=6.0, qv_h=2500.0, qv_top=None, cloud=None,
        u=(2.0, 12.0), v=(0.5, 3.0), jet=250.0, turb=0.15, wturb=0.05,
        bl=250.0, tke=(0.05, 0.0), ust=0.12, hfx=-35.0, msf=(1.01, 0.002)),
    "snow_ice_cold": dict(
        dx=1000.0, ztop=12000.0, psfc=90000.0, bound=0,
        theta=[(0, 256.0), (1000, 260.0), (4000, 272.0), (12000, 310.0)],
        qv0=1.6, qv_h=2500.0, qv_top=None, sat=(800.0, 4000.0, 1.03),
        cloud=(1000.0, 3500.0, 1.0e-4, 6.0e-5),
        u=(8.0, 1.5), v=(-3.0, 0.5), turb=0.6, wturb=0.3, bl=1200.0,
        tke=(0.3, 0.02), ust=0.25, hfx=-12.0, msf=(1.03, 0.003)),
    "ocean_mbl": dict(
        dx=500.0, ztop=12000.0, psfc=101500.0, bound=0,
        theta=[(0, 289.5), (800, 290.5), (900, 300.5), (12000, 335.0)],
        qv0=9.5, qv_h=3000.0, qv_top=900.0, sat=(550.0, 860.0, 1.01),
        cloud=(560.0, 860.0, 3.0e-4, 0.0),
        u=(7.0, 0.5), v=(-2.0, 0.2), turb=0.5, wturb=0.4, bl=850.0,
        tke=(0.5, 0.005), ust=0.25, hfx=15.0, msf=(1.0, 0.0)),
    "high_terrain": dict(
        dx=1000.0, ztop=15000.0, psfc=100000.0, bound=1, terrain=2500.0,
        theta=[(0, 292.0), (3000, 304.0), (15000, 352.0)],
        qv0=5.0, qv_h=2000.0, qv_top=None, cloud=None,
        u=(10.0, 3.0), v=(4.0, 1.0), turb=1.2, wturb=0.8, bl=1200.0,
        tke=(0.8, 0.02), ust=0.6, hfx=110.0, msf=(1.02, 0.01)),
    "tropical_deep": dict(
        dx=3000.0, ztop=20000.0, psfc=100800.0, bound=0,
        theta=[(0, 300.0), (1000, 301.0), (12000, 345.0), (16000, 375.0),
               (20000, 450.0)],
        qv0=18.0, qv_h=2500.0, qv_top=None, sat=(1500.0, 11000.0, 1.005),
        cloud=(2000.0, 11000.0, 5.0e-4, 1.0e-4),
        u=(-6.0, 2.0), v=(3.0, 1.0), turb=1.0, wturb=1.5, bl=1000.0,
        tke=(1.0, 0.05), ust=0.3, hfx=80.0, msf=(0.98, 0.004)),
    "map_extremes_southern": dict(
        dx=3000.0, ztop=16000.0, psfc=100000.0, bound=0,
        theta=[(0, 288.0), (1000, 291.0), (16000, 360.0)],
        qv0=8.0, qv_h=2500.0, qv_top=None, cloud=None,
        u=(9.0, 2.0), v=(-5.0, -1.0), turb=0.8, wturb=0.3, bl=900.0,
        tke=(0.5, 0.01), ust=0.35, hfx=60.0, msf="extreme"),
    "edge_cases": dict(
        dx=100.0, ztop=4000.0, psfc=100000.0, bound=1,
        theta=[(0, 301.0), (1000, 300.0), (1200, 304.0), (4000, 314.0)],
        qv0=10.0, qv_h=2000.0, qv_top=None, cloud=(700.0, 1000.0, 4.0e-4, 0.0),
        u=(3.0, 1.0), v=(1.0, 0.5), turb=1.0, wturb=1.0, bl=1000.0,
        tke=(1.0, 0.0), ust=0.3, hfx=200.0, msf=(1.0, 0.0), edge=True),
    # The em_les-style dry convective boundary layer, WOOF's main km_opt=2
    # use: no moisture species at all on the WOOF side (state.qv is None,
    # the dry branches of calculate_N2 and the surface buoyancy), zero
    # moisture in WRF's moist array, as WRF carries it with mp_physics = 0.
    "dry_cbl_les": dict(
        dx=50.0, ztop=3000.0, psfc=100000.0, bound=0, dry=True,
        theta=[(0, 302.0), (40, 300.2), (1000, 300.0), (1100, 305.0),
               (3000, 311.0)],
        qv0=0.0, qv_h=2000.0, qv_top=None, cloud=None,
        u=(3.0, 0.8), v=(0.5, 0.2), turb=1.0, wturb=1.5, bl=1000.0,
        tke=(1.2, 0.0), ust=0.35, hfx=240.0, msf=(1.0, 0.0)),
}


def _qvs(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.6217504 * es / (p - es)


def build_case(name: str, spec: dict, *, nx: int = 8, ny: int = 8,
               seed: int = 0) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    vc = make_vertical_coord(NZ, stretch=2.2, hybrid_opt=2, etac=0.2)
    theta_fn = _theta_fn(spec["theta"])
    terrain = None
    if spec.get("terrain"):
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        r2 = ((ii - nx / 2 + 0.5) / (nx / 4)) ** 2 + ((jj - ny / 2 + 0.5) / (ny / 3)) ** 2
        terrain = spec["terrain"] * np.exp(-r2) + 50.0 * rng.random((ny, nx))
    base = make_base_state(vc, theta_fn, p_surf=spec["psfc"],
                           ztop=spec["ztop"], terrain_z=terrain)
    nz = NZ
    flat = terrain is None
    phb = np.asarray(base.phb, dtype=np.float64)
    phb3 = phb[:, None, None] * np.ones((1, ny, nx)) if flat else phb
    pb = np.asarray(base.pb, dtype=np.float64)
    pb3 = pb[:, None, None] * np.ones((1, ny, nx)) if flat else pb
    thb = np.asarray(base.thb, dtype=np.float64)
    thb3 = thb[:, None, None] * np.ones((1, ny, nx)) if flat else thb
    mub = np.asarray(base.mub, dtype=np.float64) * np.ones((ny, nx))
    # Heights of the mass levels above ground, from the base geopotential.
    zw = phb3 / G
    zh = 0.5 * (zw[1:] + zw[:-1]) - zw[0][None]
    zabs = 0.5 * (zw[1:] + zw[:-1])

    # theta: the regime sounding at the (perturbed) level heights plus small
    # resolved fluctuations inside the boundary layer.
    bl = spec["bl"]
    inbl = (zh < bl).astype(np.float64)
    theta = theta_fn(zabs if not flat else zh)
    theta = theta + inbl * 0.25 * rng.standard_normal(theta.shape)
    # Moisture.
    qv = 1.0e-3 * spec["qv0"] * np.exp(-zh / spec["qv_h"])
    if spec.get("qv_top"):
        qv = np.where(zh > spec["qv_top"], qv * 0.35, qv)
    qv *= 1.0 + 0.03 * rng.standard_normal(qv.shape)
    p = pb3 * (1.0 + 0.61 * qv) * (1.0 + 2.0e-5 * rng.standard_normal(pb3.shape))
    if spec.get("sat"):
        lo, hi, rh = spec["sat"]
        t = theta * (p / P0) ** (RD / CP)
        band = (zh > lo) & (zh < hi)
        # Columns alternate: half saturated (RH above 1), half just below.
        col = (np.add.outer(np.arange(ny), np.arange(nx)) % 2 == 0)[None]
        target = np.where(col, rh, 0.97) * _qvs(t, p)
        qv = np.where(band, target, qv)
    qc = np.zeros_like(qv)
    qi = np.zeros_like(qv)
    if spec.get("cloud"):
        lo, hi, qcm, qim = spec["cloud"]
        band = (zh > lo) & (zh < hi)
        pick = rng.random((1, ny, nx)) < 0.75
        qc = np.where(band & pick, qcm * (0.5 + rng.random(qv.shape)), 0.0)
        qi = np.where(band & pick & (zh > 0.5 * (lo + hi)),
                      qim * (0.5 + rng.random(qv.shape)), 0.0)
    # Winds (C grid), with resolved turbulence in the boundary layer.
    u0, u1 = spec["u"]
    v0, v1 = spec["v"]
    def prof(a, b, z):
        if spec.get("jet"):
            return a + b * np.exp(-((z - spec["jet"]) / (0.6 * spec["jet"])) ** 2)
        return a + b * np.log1p(np.maximum(z, 0.0) / 10.0)
    zu = np.concatenate([zh, zh[:, :, :1]], axis=2)
    zv = np.concatenate([zh, zh[:, :1, :]], axis=1)
    u = prof(u0, u1, zu) + spec["turb"] * np.concatenate(
        [inbl, inbl[:, :, :1]], axis=2) * rng.standard_normal(zu.shape)
    v = prof(v0, v1, zv) + spec["turb"] * np.concatenate(
        [inbl, inbl[:, :1, :]], axis=1) * rng.standard_normal(zv.shape)
    zwg = zw - zw[0][None]
    w = spec["wturb"] * (zwg < bl) * rng.standard_normal(zwg.shape) * np.sin(
        np.pi * np.clip(zwg / bl, 0.0, 1.0))
    w[0] = 0.0
    w[-1] = 0.0
    tke_bl, tke_free = spec["tke"]
    tke = np.where(zh < bl, tke_bl * (1.0 - 0.8 * zh / bl), tke_free)
    tke = tke * (1.0 + 0.3 * rng.random(tke.shape))
    # Surface fields.
    ust = spec["ust"] * (0.7 + 0.6 * rng.random((ny, nx)))
    hfx = spec["hfx"] * (0.7 + 0.6 * rng.random((ny, nx)))
    # Map factors.
    if spec["msf"] == "extreme":
        def mpat(shape):
            jj, ii = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                                 indexing="ij")
            return 0.25 + 3.75 * (0.5 + 0.5 * np.sin(2 * np.pi * ii / nx)
                                  * np.cos(2 * np.pi * jj / ny))
        msft, msfu, msfv = mpat((ny, nx)), mpat((ny, nx + 1)), mpat((ny + 1, nx))
    else:
        m0, dm = spec["msf"]
        msft = m0 + dm * rng.standard_normal((ny, nx))
        msfu = m0 + dm * rng.standard_normal((ny, nx + 1))
        msfv = m0 + dm * rng.standard_normal((ny + 1, nx))
    if name == "map_extremes_southern":
        v = -v
    if spec.get("edge"):
        # Column-wise edge cases (j, i) on the 8 x 8 patch.
        u[:, 2, 2:4] = 0.0; v[:, 2:4, 2] = 0.0; w[:, 2, 2] = 0.0   # calm
        ust[2, 2] = 0.0; hfx[2, 2] = 0.0
        tke[:, 3, :] = 0.0                       # zero TKE: seed and floors
        tke[:, 4, 1] = -0.01                     # negative TKE after advection
        tke[:, 4, 2] = 1.0e-20                   # vanishing TKE
        tke[:, 4, 3] = 50.0                      # very large TKE
        theta[:, 5, 1] = theta[0, 5, 1]          # isentropic: dthrdn = BN2 = 0
        qv[:, 5, 1] = 0.0; qc[:, 5, 1] = 0.0; qi[:, 5, 1] = 0.0
        qv[:, 5, 2] = 0.0                        # dry column inside moist air
        qc[:, 5, 3] = 1.0e-5                     # exactly the qc_cr threshold
        theta[:3, 6, 1] = theta[3, 6, 1] + np.array([1.5, 1.0, 0.5])  # superadiabatic
        u[:, 6, 3] += np.linspace(0.0, 25.0, nz)  # strong shear
        hfx[6, 4] = -200.0                       # strong surface cooling
        ust[6, 5] = 1.5                          # strong drag
    # Perturbation dry mass and the hydrostatic geopotential (WRF's
    # dphi = -(c1h*mut + c2h) * alt * dnw with alt from the equation of
    # state; mu' a few tens of Pa).
    mup = 30.0 * rng.standard_normal((ny, nx))
    mut = mub + mup
    thm = theta * (1.0 + RV / RD * qv)
    alt = RD * thm / P0 * (p / P0) ** (-1.0 / GAMMA)
    ph = np.empty_like(phb3)
    ph[0] = phb3[0]
    c1h = np.asarray(vc.c1h, dtype=np.float64)[:, None, None]
    c2h = np.asarray(vc.c2h, dtype=np.float64)[:, None, None]
    dnw = np.asarray(vc.dnw, dtype=np.float64)[:, None, None]
    for k in range(nz):
        ph[k + 1] = ph[k] - dnw[k, 0, 0] * (c1h[k, 0, 0] * mut + c2h[k, 0, 0]) * alt[k]
    php = ph - phb3

    f32 = lambda a: np.ascontiguousarray(np.asarray(a, dtype=np.float64).astype(F32))
    arrays = {
        "u": f32(u), "v": f32(v), "w": f32(w),
        "php": f32(php), "phb": f32(phb if flat else phb3),
        "alt": f32(alt), "p": f32(p),
        "thb": f32(thb if flat else thb3),
        "qv": f32(qv), "qc": f32(qc), "qi": f32(qi),
        "tke": f32(tke),
        "mub2d": f32(mub), "mup": f32(mup),
        "msft": f32(msft), "msfu": f32(msfu), "msfv": f32(msfv),
        "ust": f32(ust), "hfx": f32(hfx), "qfx": np.zeros((ny, nx), F32),
    }
    arrays["mup0"] = arrays["mup"].copy()
    # thp is the float32 residual that reproduces the float32 theta through
    # WOOF's thb + thp: the engine stores exactly this split.
    theta32 = f32(theta)
    thb_b = arrays["thb"] if not flat else arrays["thb"][:, None, None]
    arrays["thp"] = np.ascontiguousarray(theta32 - thb_b, dtype=F32)
    arrays["mut"] = np.add(arrays["mub2d"], arrays["mup"], dtype=F32)
    for key in ("fnm", "fnp", "dn", "dnw", "znw", "c1h", "c2h", "c1f", "c2f"):
        arrays[key] = f32(getattr(vc, key))
    # Periodic patches carry the redundant staggered row/column.
    if not spec["bound"]:
        arrays["u"][:, :, -1] = arrays["u"][:, :, 0]
        arrays["v"][:, -1, :] = arrays["v"][:, 0, :]
        arrays["msfu"][:, -1] = arrays["msfu"][:, 0]
        arrays["msfv"][-1, :] = arrays["msfv"][0, :]
    # cf1..cf3 (WRF start_em): surface extrapolation weights.
    dn_, dnw_ = arrays["dn"].astype(np.float64), arrays["dnw"].astype(np.float64)
    fnm_, fnp_ = arrays["fnm"].astype(np.float64), arrays["fnp"].astype(np.float64)
    cof1 = (2.0 * dn_[1] + dn_[2]) / (dn_[1] + dn_[2]) * dnw_[0] / dn_[1]
    cof2 = dn_[1] / (dn_[1] + dn_[2]) * dnw_[0] / dn_[2]
    cf1 = fnp_[1] + cof1
    cf2 = fnm_[1] - cof1 - cof2
    cf3 = cof2
    dx = spec["dx"]
    meta = {
        "name": name, "nx": nx, "ny": ny, "nz": nz,
        "bx": int(spec["bound"]), "by": int(spec["bound"]),
        "dx": dx, "dy": dx, "dt": max(0.5, 6.0 * dx / 1000.0),
        "ztop": spec["ztop"], "p_top": float(base.p_top),
        "cf1": float(F32(cf1)), "cf2": float(F32(cf2)), "cf3": float(F32(cf3)),
        "mix_upper_bound": 0.1, "seed": seed,
        "terrain": not flat, "dry": bool(spec.get("dry", False)),
    }
    assert np.float32(1.0 / dx) == np.float32(1.0) / np.float32(dx)
    return arrays, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for n, (name, spec) in enumerate(REGIMES.items()):
        arrays, meta = build_case(name, spec, seed=1000 + n)
        path = args.out / f"case-{name}.npz"
        np.savez(path, meta_json=np.array(json.dumps(meta, sort_keys=True)),
                 **arrays)
        index.append({"file": path.name, "name": name,
                      "columns": meta["nx"] * meta["ny"],
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(name, meta["nx"] * meta["ny"], "columns")
    (args.out / "cases.json").write_text(json.dumps(
        {"arms": ARMS, "cases": index}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
