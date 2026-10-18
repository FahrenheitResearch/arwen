"""Columns and regimes for the legacy RRTMG column oracle.

Builds 128 independent atmospheric columns (one valid time, 2026-07-15
18:00 UTC, so longitude and hemisphere set day, night, season and grazing
sun) and writes, for each regime, the exact float32 inputs both sides read:

* ``in_<regime>.bin``: the Fortran oracle's input (rad_column_oracle.F90).
* ``columns_<grid>.npz`` + ``regimes.json``: what run_woof.py hands WOOF.

Families (column count): convective afternoon land (16), tropical ocean
convection (8), marine stratocumulus (8), stable clear night (12), snow and
ice surfaces including sea ice, Greenland, Antarctic polar night (14), high
terrain (10), grazing sun on both sides of the terminator (10), edge cases
(22), mixed random columns (28).

Inputs are physical where it matters and pushed to the branch edges where
the code branches: QCLD on both sides of 1e-12, RH at and above 1, radii on
both sides of the wrapper thresholds (liquid 2.5 um, ice 5 um, snow 130 um
clamp), RH on both sides of the aerosol clamps, zero and negative mixing
ratios, polar and equatorial latitudes, the date line.

usage: python make_columns.py OUT_DIR
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

F = np.float32
NZ = 50
NCOL = 128
START = datetime(2026, 7, 15, 12, 0, 0)
ELAPSED = 6 * 3600.0
RADT_MINUTES = 15.0
R_D, CP, G = 287.0, 1004.5, 9.81

#: regime name -> options.  mp flags follow WRF v4.6.1's package tables;
#: has_req follows module_physics_init.F:985-1033 for use_mp_re = 1.
REGIMES = {
    "thompson8":        dict(mp=8,  grid="ptop5000", o3input=2, aer_opt=0),
    "thompson8_o3wrap": dict(mp=8,  grid="ptop5000", o3input=0, aer_opt=0),
    "thompson8_ptop2k": dict(mp=8,  grid="ptop2000", o3input=2, aer_opt=0),
    "wsm6":             dict(mp=6,  grid="ptop5000", o3input=2, aer_opt=0),
    "morrison10":       dict(mp=10, grid="ptop5000", o3input=2, aer_opt=0),
    "p3_50":            dict(mp=50, grid="ptop5000", o3input=2, aer_opt=0),
    "kessler1":         dict(mp=1,  grid="ptop5000", o3input=2, aer_opt=0),
    "thompson28_aer0":  dict(mp=28, grid="ptop5000", o3input=2, aer_opt=0),
    "thompson28_aer3":  dict(mp=28, grid="ptop5000", o3input=2, aer_opt=3),
}
SPECIES = {  # f_qv, f_qc, f_qr, f_qi, f_qs, f_qg
    8: (1, 1, 1, 1, 1, 1), 28: (1, 1, 1, 1, 1, 1), 6: (1, 1, 1, 1, 1, 1),
    10: (1, 1, 1, 1, 1, 1), 50: (1, 1, 1, 1, 0, 0), 1: (1, 1, 1, 0, 0, 0)}
HAS_REQ = {8: (1, 1, 1), 28: (1, 1, 1), 6: (1, 1, 1), 10: (0, 0, 0),
           50: (1, 1, 0), 1: (0, 0, 0)}
P_TOP = {"ptop5000": 5000.0, "ptop2000": 2000.0}


def calendar():
    from datetime import timedelta
    valid = START + timedelta(seconds=ELAPSED)
    julday = int(valid.timetuple().tm_yday)
    hour = valid.hour + valid.minute / 60.0 + valid.second / 3600.0
    julian = F((julday - 1) + hour / 24.0)
    gmt = F(START.hour + START.minute / 60.0 + START.second / 3600.0)
    xtime = F(ELAPSED / 60.0)
    return dict(yr=valid.year, julday=julday, julian=julian, gmt=gmt,
                xtime=xtime, radt=F(RADT_MINUTES))


def coszen_estimate(lat, lon):
    """float64 sun height for choosing longitudes only (never graded)."""
    cal = calendar()
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (cal["julday"] - 80) / 365.0)
    hour = 18.0 + RADT_MINUTES / 120.0
    hr = np.deg2rad(15.0 * (hour + lon / 15.0 - 12.0))
    la = np.deg2rad(lat)
    return np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(hr)


def qsat(p, t):
    tc = t - 273.15
    es = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    return 0.622 * es / np.maximum(p - es, 1.0)


def eta_levels(nz=NZ):
    x = np.linspace(0.0, 1.0, nz + 1)
    return 1.0 - x ** 1.6           # surface 1 -> top 0, fine near the ground


def base_spec(**kw):
    spec = dict(lat=40.0, lon=-95.0, ht=300.0, psfc=97000.0, tsfc=300.0,
                lapse=0.0065, p_trop=15000.0, t_trop=205.0, inversion=0.0,
                rh_sfc=0.7, rh_top=0.1, clouds=[], tsk=None, emiss=0.98,
                albedo=0.18, xland=1.0, xice=0.0, snow=0.0, nwfa=3e9,
                nifa=1e6, rc=None, ri=None, rs=None, hooks=[])
    spec.update(kw)
    return spec


def build_column(spec, p_top, rng):
    eta = eta_levels()
    psfc = spec["psfc"]
    p8w = p_top + eta * (psfc - p_top)                     # (nz+1,) float64
    p = 0.5 * (p8w[:-1] + p8w[1:])
    # temperature on mass levels
    t = spec["tsfc"] * (p / psfc) ** (R_D * spec["lapse"] / G)
    t = np.maximum(t, spec["t_trop"])
    strat = p < spec["p_trop"]
    t = np.where(strat, spec["t_trop"] + 2.5 * np.log(spec["p_trop"] / p), t)
    if spec["inversion"]:
        dp = psfc - p
        t = t - spec["inversion"] * np.exp(-dp / 4000.0)
    t = t + rng.uniform(-0.3, 0.3, NZ)
    # moisture: RH decreasing with height, then cloud layers saturate
    frac = (psfc - p) / (psfc - p_top)
    rh = spec["rh_sfc"] + (spec["rh_top"] - spec["rh_sfc"]) * np.sqrt(frac)
    rh = np.where(strat, 0.02, rh)
    q = {name: np.zeros(NZ) for name in ("qc", "qr", "qi", "qs", "qg")}
    for cloud in spec["clouds"]:
        name, pbot, ptop, amount = cloud[:4]
        rh_in = cloud[4] if len(cloud) > 4 else None
        layer = (p <= pbot) & (p >= ptop)
        shape = np.sin(np.pi * np.clip((pbot - p) / max(pbot - ptop, 1.0), 0, 1))
        q[name] = np.where(layer, q[name] + amount * (0.3 + 0.7 * shape), q[name])
        if rh_in is not None:
            rh = np.where(layer, rh_in + rng.uniform(-0.01, 0.01, NZ), rh)
    qv = rh * qsat(p, t)
    # heights: hypsometric on the interfaces, MSL
    tv = t * (1.0 + 0.608 * qv)
    dz = R_D * tv / G * np.log(p8w[:-1] / p8w[1:])
    zw = spec["ht"] + np.concatenate([[0.0], np.cumsum(dz)])
    col = dict(p3d=p, p8w=p8w, t3d=t, qv=qv, z_at_w=zw, **q)
    col = {k: np.asarray(v, F) for k, v in col.items()}
    col["dz8w"] = (col["z_at_w"][1:] - col["z_at_w"][:-1]).astype(F)
    col["pi3d"] = ((col["p3d"].astype(np.float64) / 1.0e5) ** (R_D / CP)).astype(F)
    col["rho"] = (col["p3d"].astype(np.float64)
                  / (R_D * col["t3d"] * (1.0 + 0.608 * col["qv"]))).astype(F)
    # radii, microns (WOOF state convention); background where no condensate
    rc = spec["rc"] if spec["rc"] is not None else rng.uniform(6.0, 18.0)
    ri = spec["ri"] if spec["ri"] is not None else rng.uniform(15.0, 70.0)
    rs = spec["rs"] if spec["rs"] is not None else rng.uniform(80.0, 400.0)
    col["effc"] = np.where(col["qc"] > 0, rc * rng.uniform(0.8, 1.2, NZ), 2.49).astype(F)
    col["effi"] = np.where(col["qi"] > 0, ri * rng.uniform(0.8, 1.2, NZ), 4.99).astype(F)
    col["effs"] = np.where(col["qs"] > 0, rs * rng.uniform(0.8, 1.2, NZ), 9.99).astype(F)
    col["effc"] = np.clip(col["effc"], 2.49, 50.0).astype(F)
    col["effi"] = np.clip(col["effi"], 4.99, 125.0).astype(F)
    col["effs"] = np.clip(col["effs"], 9.99, 999.0).astype(F)
    col["nwfa"] = (spec["nwfa"] * np.exp(-frac * 3.0) * rng.uniform(0.5, 1.5, NZ)).astype(F)
    col["nifa"] = (spec["nifa"] * np.exp(-frac * 2.0) * rng.uniform(0.5, 1.5, NZ)).astype(F)
    surf = dict(tsk=spec["tsk"] if spec["tsk"] is not None else spec["tsfc"] + 2.0,
                emiss=spec["emiss"], albedo=spec["albedo"], xland=spec["xland"],
                xice=spec["xice"], snow=spec["snow"], xlat=spec["lat"],
                xlong=spec["lon"], ht=spec["ht"])
    col.update({k: F(v) for k, v in surf.items()})
    for hook in spec["hooks"]:
        hook(col)
    return col


def specs(rng):
    out = []
    u = rng.uniform
    # convective afternoon land, US Plains / Southeast (local early afternoon)
    for n in range(16):
        deep = n % 2 == 0
        clouds = [("qc", 85000, 50000, u(2e-4, 1.5e-3), 1.0),
                  ("qi", 45000, 20000, u(5e-5, 6e-4)),
                  ("qs", 60000, 25000, u(1e-4, 1e-3)),
                  ("qr", 95000, 60000, u(1e-4, 2e-3)),
                  ("qg", 70000, 35000, u(1e-5, 2e-3))] if deep else \
                 [("qc", 88000, 80000, u(5e-5, 4e-4), u(0.93, 1.02))]
        out.append(base_spec(lat=u(28, 46), lon=u(-102, -80), ht=u(100, 900),
                             psfc=u(91000, 101000), tsfc=u(298, 309),
                             rh_sfc=u(0.55, 0.85), clouds=clouds,
                             albedo=u(0.15, 0.22), emiss=u(0.95, 0.99),
                             tsk=None, nwfa=u(1e9, 1e10), nifa=u(1e5, 3e6)))
    # tropical ocean convection
    for n in range(8):
        out.append(base_spec(lat=u(-10, 15), lon=u(-150, -110), ht=0.0,
                             psfc=u(100800, 101300), tsfc=u(299, 302),
                             p_trop=u(9000, 11000), t_trop=u(190, 197),
                             rh_sfc=0.85, rh_top=0.3, xland=2.0, emiss=0.98,
                             albedo=u(0.06, 0.08), tsk=u(300, 303),
                             clouds=[("qc", 95000, 55000, u(3e-4, 2e-3), 1.0),
                                     ("qi", 30000, 12000, u(1e-5, 3e-4), 0.95),
                                     ("qs", 50000, 20000, u(5e-5, 5e-4)),
                                     ("qr", 100000, 70000, u(1e-4, 1e-3))],
                             nwfa=u(3e8, 3e9), nifa=u(1e4, 3e5)))
    # marine stratocumulus under an inversion (California, Peru, Namibia)
    for n, (la, lo) in enumerate([(30, -125), (33, -122), (-18, -80), (-20, -78),
                                  (-15, 10), (-20, 8), (25, -130), (35, -128)]):
        out.append(base_spec(lat=la + u(-1, 1), lon=lo + u(-1, 1), ht=0.0,
                             psfc=u(101200, 102000), tsfc=u(287, 292),
                             inversion=-6.0, rh_sfc=0.9, rh_top=0.1,
                             xland=2.0, albedo=0.07, tsk=u(288, 292),
                             clouds=[("qc", 96000, 90000, u(1e-4, 6e-4), u(0.97, 1.03))],
                             rc=u(8, 14), nwfa=u(2e8, 2e9)))
    # stable clear night: Central Asia / China / Africa at local night
    for n in range(12):
        out.append(base_spec(lat=u(25, 55), lon=u(60, 125), ht=u(100, 1500),
                             psfc=u(85000, 101500), tsfc=u(285, 300),
                             inversion=u(3, 10), rh_sfc=u(0.3, 0.8),
                             rh_top=0.05, emiss=u(0.9, 0.98),
                             clouds=([("qi", 30000, 22000, u(1e-6, 4e-5))]
                                     if n % 3 == 0 else []),
                             albedo=u(0.15, 0.3)))
    # snow and ice: Arctic sea ice, Greenland plateau, Antarctic polar night,
    # Southern Ocean ice edge, Alaska/Siberia snow (summer NH, winter SH)
    snowy = [(85, 0, 0, 101000, 271, 2.0, 1.0, 0.8), (80, 120, 0, 101000, 270, 2.0, 0.6, 0.7),
             (72, -38, 3200, 68000, 255, 1.0, 0.0, 0.82), (75, -45, 2600, 73000, 258, 1.0, 0.0, 0.8),
             (-78, 0, 2800, 70000, 210, 1.0, 0.0, 0.85), (-85, 140, 3000, 68000, 205, 1.0, 0.0, 0.85),
             (-70, 60, 50, 99000, 245, 2.0, 1.0, 0.75), (-65, -60, 0, 98500, 250, 2.0, 0.5, 0.6),
             (-60, 170, 0, 99000, 268, 2.0, 0.5, 0.55), (68, 160, 300, 98000, 278, 1.0, 0.0, 0.7),
             (-45, -70, 1200, 87000, 262, 1.0, 0.0, 0.7), (-40, 175, 800, 92000, 270, 1.0, 0.0, 0.6),
             (89.9, 45, 0, 101000, 272, 2.0, 1.0, 0.8), (-89.9, 0, 2800, 68000, 200, 1.0, 0.0, 0.85)]
    for la, lo, ht, ps, ts, xl, xi, alb in snowy:
        out.append(base_spec(lat=la, lon=lo, ht=ht, psfc=ps, tsfc=ts,
                             t_trop=u(190, 215), p_trop=u(22000, 30000),
                             inversion=u(2, 12), rh_sfc=u(0.7, 0.95), rh_top=0.1,
                             xland=xl, xice=xi, snow=u(20, 1200), albedo=alb,
                             emiss=u(0.98, 1.0), tsk=ts - u(0, 6),
                             clouds=[("qi", ps - 5000, ps - 30000, u(5e-6, 1e-4), 1.0),
                                     ("qs", ps - 2000, ps - 25000, u(1e-5, 4e-4)),
                                     ("qc", ps - 2000, ps - 8000,
                                      u(0, 1e-4) if ts > 250 else 0.0)],
                             nwfa=u(5e7, 5e8), nifa=u(1e3, 1e5)))
    # high terrain: Tibet (night), Andes and Rockies (day), Ethiopia, Alps
    terrain = [(32, 90, 4800, 57000, 278), (33, 85, 5000, 55000, 275),
               (-15, -70, 3900, 63000, 280), (-22, -67, 4300, 60000, 276),
               (40, -106, 3000, 70000, 290), (39, -107, 3400, 67500, 287),
               (9, 39, 2400, 76000, 292), (46, 8, 2500, 75000, 285),
               (28, 86, 6000, 49000, 260), (-33, -70, 3500, 66000, 272)]
    for n, (la, lo, ht, ps, ts) in enumerate(terrain):
        out.append(base_spec(lat=la, lon=lo, ht=ht, psfc=ps, tsfc=ts,
                             rh_sfc=u(0.3, 0.7), albedo=u(0.15, 0.4), emiss=u(0.9, 0.97),
                             snow=(u(10, 200) if n % 2 else 0.0),
                             clouds=([("qc", ps - 3000, ps - 15000, u(1e-4, 8e-4), 1.0),
                                      ("qi", 35000, 25000, u(1e-5, 2e-4))]
                                     if n % 2 == 0 else []),
                             nwfa=u(1e8, 1e9)))
    # grazing sun: longitudes on both sides of the terminator
    lons = np.linspace(-180, 180, 72001)
    for la in (45.0, -30.0, 10.0, 60.0, -55.0):
        cz = coszen_estimate(la, lons)
        for target in (0.004, -0.004):
            k = int(np.argmin(np.abs(cz - target)))
            out.append(base_spec(lat=la, lon=float(lons[k]), ht=u(0, 500),
                                 psfc=u(98000, 101500), tsfc=u(280, 300),
                                 clouds=[("qc", 90000, 80000, u(1e-5, 3e-4), 1.0)]))
    nedge = 22
    edges = edge_specs(rng)
    assert len(edges) == nedge, len(edges)
    out += edges
    # mixed random columns anywhere on the globe
    while len(out) < NCOL:
        ps = u(60000, 102500)
        clouds = []
        for name, lo_amt, hi_amt in (("qc", 1e-6, 1e-3), ("qi", 1e-7, 3e-4),
                                     ("qs", 1e-6, 1e-3), ("qr", 1e-6, 1e-3),
                                     ("qg", 1e-7, 5e-4)):
            if u() < 0.6:
                pb = u(30000, ps)
                clouds.append((name, pb, pb - u(5000, 30000), 10 ** u(np.log10(lo_amt), np.log10(hi_amt)),
                               u(0.85, 1.05) if u() < 0.5 else None))
        clouds = [c if c[4] is not None else c[:4] for c in clouds]
        land = u() < 0.6
        out.append(base_spec(lat=u(-70, 70), lon=u(-180, 180), ht=(u(0, 3000) if land else 0.0),
                             psfc=ps, tsfc=u(250, 310), lapse=u(0.004, 0.0085),
                             inversion=(u(-5, 8) if u() < 0.3 else 0.0),
                             rh_sfc=u(0.2, 0.95), rh_top=u(0.02, 0.3),
                             xland=1.0 if land else 2.0, albedo=u(0.06, 0.5),
                             emiss=u(0.88, 1.0), snow=(u(0, 300) if land and u() < 0.3 else 0.0),
                             clouds=clouds, nwfa=10 ** u(7, 10.5), nifa=10 ** u(3, 7)))
    return out[:NCOL]


def edge_specs(rng):
    u = rng.uniform
    e = []

    def at(names, ks, values):
        def hook(col):
            for name, k, v in zip(names, ks, values):
                col[name][k] = F(v)
        return hook

    day = dict(lat=35.0, lon=-90.0)
    # 1 bone-dry clear column
    e.append(base_spec(**day, rh_sfc=0.0, rh_top=0.0,
                       hooks=[lambda c: c.__setitem__("qv", np.zeros(NZ, F))]))
    # 2 negative mixing ratios (advection undershoot)
    e.append(base_spec(**day, clouds=[("qc", 90000, 80000, 2e-4, 1.0)],
                       hooks=[at(["qv", "qc", "qi", "qs", "qr"], [40, 3, 20, 25, 2],
                                 [-1e-7, -1e-10, -2e-9, -1e-8, -3e-9])]))
    # 3 supersaturated cloud (RH > 1, CLDFRA = 1)
    e.append(base_spec(**day, clouds=[("qc", 92000, 70000, 6e-4, 1.05),
                                      ("qi", 40000, 30000, 1e-4, 1.08)]))
    # 4/5 QCLD at the 1e-12 threshold on both sides
    e.append(base_spec(**day, hooks=[at(["qc"] * 4, [5, 6, 7, 8],
                                        [9.99e-13, 1.0e-12, 1.0001e-12, 5e-13])]))
    e.append(base_spec(**day, hooks=[at(["qi", "qs", "qc"], [30, 30, 30],
                                        [4e-13, 4e-13, 2e-13]),
                                     at(["qi", "qs"], [31, 31], [6e-13, 5e-13])]))
    # 6 ice radius at the 5 um threshold and just above
    e.append(base_spec(**day, clouds=[("qi", 50000, 25000, 2e-4, 1.0)], ri=5.0,
                       hooks=[at(["effi"] * 3, [35, 36, 37], [5.0, 5.0001, 4.99])]))
    # 7 snow radius beyond the 130 um clamp
    e.append(base_spec(**day, clouds=[("qs", 70000, 40000, 8e-4)], rs=999.0))
    # 8/9 liquid radius at 2.5 um / background 2.49 with cloud water, land and ocean
    e.append(base_spec(**day, clouds=[("qc", 92000, 85000, 3e-4, 1.0)], rc=2.49,
                       hooks=[at(["effc"] * 2, [4, 5], [2.5, 2.5001])]))
    e.append(base_spec(lat=20.0, lon=-60.0, xland=2.0, albedo=0.07,
                       clouds=[("qc", 95000, 88000, 3e-4, 1.0)], rc=2.49,
                       hooks=[at(["effc"] * 2, [3, 4], [2.5, 2.5001])]))
    # 10 very heavy cloud water and rain, large optical depth
    e.append(base_spec(**day, clouds=[("qc", 95000, 40000, 5e-3, 1.0),
                                      ("qr", 98000, 60000, 6e-3), ("qg", 80000, 40000, 4e-3)]))
    # 11 cirrus at the model top
    e.append(base_spec(**day, clouds=[("qi", 9000, 5000, 2e-5, 1.0)], ri=60.0))
    # 12 sea ice exactly at the 0.5 threshold, ocean point
    e.append(base_spec(lat=72.0, lon=-150.0, xland=2.0, xice=0.5, snow=50.0,
                       tsfc=272.0, albedo=0.5, tsk=271.0))
    # 13 hot desert, low emissivity, dust-like aerosol
    e.append(base_spec(lat=24.0, lon=10.0, tsfc=318.0, tsk=335.0, emiss=0.85,
                       albedo=0.38, rh_sfc=0.05, rh_top=0.02, nwfa=5e10, nifa=5e9))
    # 14 extreme cold surface, deep snow
    e.append(base_spec(lat=-80.0, lon=120.0, ht=3500.0, psfc=64000.0, tsfc=205.0,
                       tsk=198.0, t_trop=185.0, p_trop=25000.0, snow=3000.0,
                       albedo=0.9, emiss=1.0, inversion=20.0))
    # 15/16 equator at the date line, both signs of 180
    e.append(base_spec(lat=0.0, lon=180.0, xland=2.0, albedo=0.06, tsfc=301.0))
    e.append(base_spec(lat=0.0, lon=-180.0, xland=2.0, albedo=0.06, tsfc=301.0,
                       clouds=[("qc", 96000, 85000, 2e-4, 1.0)]))
    # 17 aerosol numbers zero, negative and above the gt_aod caps
    e.append(base_spec(**day, hooks=[at(["nwfa", "nifa", "nwfa", "nifa", "nwfa", "nifa"],
                                        [0, 0, 1, 1, 2, 2],
                                        [0.0, 0.0, -5.0, -1.0, 2e11, 2e10])]))
    # 18-20 RH bands the two aerosol generations clamp differently: < 10.1,
    # between 98 and 99.1, above 99.1
    e.append(base_spec(**day, rh_sfc=0.05, rh_top=0.01))
    e.append(base_spec(**day, clouds=[("qr", 96000, 60000, 1e-6, 0.985)]))
    e.append(base_spec(**day, clouds=[("qr", 96000, 60000, 1e-6, 0.996)]))
    # 21 very low surface pressure, cloud at the ground
    e.append(base_spec(lat=30.0, lon=-75.0, ht=5800.0, psfc=50000.0, tsfc=262.0,
                       clouds=[("qc", 50000, 45000, 2e-4, 1.0), ("qi", 45000, 30000, 1e-4, 1.0)]))
    # 22 every hydrometeor in every layer, mixed phase
    e.append(base_spec(**day, clouds=[("qc", 101000, 5000, 1e-5, 0.99),
                                      ("qi", 101000, 5000, 1e-6), ("qs", 101000, 5000, 1e-5),
                                      ("qr", 101000, 5000, 1e-5), ("qg", 101000, 5000, 1e-6)]))
    return e


FIELDS_3 = ("p3d", "t3d", "pi3d", "dz8w", "rho", "qv", "qc", "qr", "qi", "qs",
            "qg", "effc", "effi", "effs", "nwfa", "nifa")
FIELDS_W = ("p8w", "z_at_w")
FIELDS_2 = ("tsk", "emiss", "albedo", "xland", "xice", "snow", "xlat", "xlong", "ht")


def fnm_fnp(p8w_ref):
    """WRF-like half-to-full weights from the shared eta (FP64 build, FP32 store)."""
    znw = eta_levels()
    dnw = np.diff(znw)
    fnm = np.zeros(NZ)
    fnp = np.zeros(NZ)
    dn = np.zeros(NZ)
    dn[1:] = 0.5 * (dnw[1:] + dnw[:-1])
    fnp[1:] = 0.5 * dnw[1:] / dn[1:]
    fnm[1:] = 0.5 * dnw[:-1] / dn[1:]
    return fnm.astype(F), fnp.astype(F)


def build_grid(p_top, seed=20261007):
    rng = np.random.default_rng(seed)
    cols = [build_column(s, p_top, rng) for s in specs(np.random.default_rng(seed + 1))]
    out = {}
    for name in FIELDS_3 + FIELDS_W:
        out[name] = np.stack([c[name] for c in cols]).astype(F)
    for name in FIELDS_2:
        out[name] = np.array([c[name] for c in cols], F)
    out["fnm"], out["fnp"] = fnm_fnp(out["p8w"][0])
    out["p_top"] = np.array([p_top], F)
    return out


def radius_meters(um):
    return (np.asarray(um, F) * F(1.0e-6)).astype(F)


def write_fortran_input(path, grid, regime):
    cal = calendar()
    mp = regime["mp"]
    sp = SPECIES[mp]
    hr = HAS_REQ[mp]
    hdr = np.zeros(24, np.int32)
    hdr[:22] = [NCOL, NZ, mp, *hr, *sp, 1 if mp == 1 else 0, regime["o3input"],
                regime["aer_opt"], 2, cal["julday"], cal["yr"], 1, 2, 0, 1]
    rhdr = np.zeros(8, F)
    rhdr[:5] = [cal["julian"], cal["xtime"], cal["gmt"], cal["radt"], grid["p_top"][0]]
    zero = np.zeros_like(grid["qv"])
    present = dict(zip(("qv", "qc", "qr", "qi", "qs", "qg"), sp))
    with open(path, "wb") as f:
        f.write(hdr.tobytes()); f.write(rhdr.tobytes())

        def w(a):
            f.write(np.asfortranarray(np.asarray(a, F)).tobytes(order="F"))
        for name in FIELDS_3:
            a = grid[name]
            if name in present and not present[name]:
                a = zero
            if name in ("effc", "effi", "effs"):
                a = radius_meters(a)
            w(a)
        for name in FIELDS_W:
            w(grid[name])
        w(grid["fnm"]); w(grid["fnp"])
        for name in FIELDS_2:
            w(grid[name])


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    grids = {g: build_grid(p) for g, p in P_TOP.items()}
    for g, grid in grids.items():
        np.savez_compressed(out / f"columns_{g}.npz", **grid)
    meta = {"start": START.isoformat(), "elapsed_seconds": ELAPSED,
            "radt_minutes": RADT_MINUTES, "ncol": NCOL, "nz": NZ,
            "calendar": {k: float(v) for k, v in calendar().items()},
            "regimes": {}}
    for name, r in REGIMES.items():
        write_fortran_input(out / f"in_{name}.bin", grids[r["grid"]], r)
        meta["regimes"][name] = dict(r, species=SPECIES[r["mp"]], has_req=HAS_REQ[r["mp"]])
    (out / "regimes.json").write_text(json.dumps(meta, indent=2) + "\n")
    g = grids["ptop5000"]
    cz = coszen_estimate(g["xlat"].astype(float), g["xlong"].astype(float))
    print(f"wrote {len(REGIMES)} regimes x {NCOL} columns; day {int((cz > 0).sum())}, "
          f"night {int((cz <= 0).sum())}")


if __name__ == "__main__":
    main()
