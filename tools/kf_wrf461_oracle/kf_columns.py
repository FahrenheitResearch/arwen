"""Column sets for the WRF v4.6.1 Kain-Fritsch column oracle.

Every column is a hydrostatically consistent sounding built in float64 and
rounded once to float32, then written in the stream layout
``run_kf_oracle.F90`` reads.  The SAME float32 words feed WRF's Fortran and
WOOF's kernel; nothing downstream regenerates them.

Two sets:

* ``structured`` -- named regimes (tropical ocean, midlatitude loaded gun,
  stable night, trade-wind shallow, cold/snow, elevated over a cold dome,
  high terrain, dry desert, tropical-cyclone core, very dry, supersaturated
  layers) crossed with grid length, time step, cumulus period, vertical
  resolution, model top and the four WRF microphysics feedback branches
  (warm_rain, .NOT.F_QS, F_QS without F_QI, F_QS with F_QI) plus F_QR off.
* ``sweep`` -- random draws around the same regimes, in launch-sized blocks
  that share (nz, dx, dt, cudt, branch), so the kernel's multi-column tiling
  and column reordering are exercised too.

Usage::

    python kf_columns.py structured OUT.bin [--seed N]
    python kf_columns.py sweep OUT.bin --blocks B --per-block C [--seed N]

A JSON sidecar ``OUT.bin.json`` records each column's regime label and
parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys

import numpy as np

RD = 287.0
CP = 1004.5
G = 9.81
RV = 461.6

# (warm_rain, f_qr, f_qi, f_qs): WRF's four KF feedback branches and the
# F_QR-off fold (module_cu_kfeta.F:492-505, 2599-2638).
BRANCHES = {
    "warm_rain": (1, 1, 0, 0),        # mp_physics=1 (Kessler)
    "no_qs": (0, 1, 0, 0),            # .NOT. F_QS, e.g. mp_physics=50 (P3)
    "qs_no_qi": (0, 1, 0, 1),         # F_QS without F_QI
    "qs_qi": (0, 1, 1, 1),            # F_QS and F_QI (6, 8, 9, 10, 16, 18, 28)
    "no_qr": (0, 0, 0, 0),            # no QR at all (mp_physics=0)
}

# Temperature (K) and relative-humidity knots against pressure (hPa).
REGIMES = {
    "tropical_ocean": dict(
        psfc=(1006.0, 1012.0),
        t=[(1012, 301.0), (950, 296.5), (850, 291.5), (700, 283.0),
           (500, 267.5), (400, 257.0), (300, 242.5), (200, 221.0),
           (150, 207.0), (100, 192.5), (70, 197.0), (40, 210.0)],
        rh=[(1012, 0.82), (950, 0.88), (850, 0.85), (700, 0.70),
            (500, 0.60), (300, 0.50), (200, 0.35), (120, 0.15),
            (40, 0.02)]),
    "midlat_loaded_gun": dict(
        psfc=(960.0, 985.0),
        t=[(985, 304.0), (850, 292.5), (820, 296.0), (700, 288.0),
           (500, 264.0), (400, 251.0), (300, 236.0), (250, 225.0),
           (200, 219.0), (150, 216.0), (100, 213.0), (40, 218.0)],
        rh=[(985, 0.68), (870, 0.80), (835, 0.35), (700, 0.25),
            (500, 0.30), (300, 0.35), (200, 0.15), (100, 0.04),
            (40, 0.01)]),
    "stable_night": dict(
        psfc=(990.0, 1020.0),
        t=[(1020, 276.0), (985, 283.0), (900, 280.0), (850, 278.5),
           (700, 270.0), (500, 254.0), (300, 228.0), (200, 218.0),
           (100, 213.0), (40, 215.0)],
        rh=[(1020, 0.85), (985, 0.40), (850, 0.35), (700, 0.30),
            (500, 0.25), (300, 0.20), (100, 0.05), (40, 0.01)]),
    "trade_shallow": dict(
        psfc=(1010.0, 1016.0),
        t=[(1016, 299.0), (950, 294.0), (850, 287.0), (800, 289.5),
           (700, 284.0), (500, 267.0), (300, 241.0), (200, 220.0),
           (100, 193.0), (40, 209.0)],
        rh=[(1016, 0.80), (900, 0.92), (815, 0.90), (790, 0.18),
            (500, 0.10), (300, 0.12), (100, 0.04), (40, 0.01)]),
    "cold_snow": dict(
        psfc=(985.0, 1005.0),
        t=[(1005, 268.5), (900, 262.0), (850, 258.0), (700, 247.0),
           (500, 229.0), (400, 220.0), (300, 214.0), (200, 213.0),
           (100, 211.0), (40, 213.0)],
        rh=[(1005, 0.92), (850, 0.95), (700, 0.93), (500, 0.85),
            (400, 0.70), (300, 0.50), (100, 0.10), (40, 0.02)]),
    "polar_low_ocean": dict(
        psfc=(980.0, 1000.0),
        t=[(1000, 274.5), (900, 266.0), (850, 262.0), (700, 250.0),
           (500, 232.0), (400, 222.0), (300, 218.0), (200, 217.0),
           (100, 214.0), (40, 214.0)],
        rh=[(1000, 0.88), (850, 0.92), (700, 0.90), (500, 0.80),
            (300, 0.40), (100, 0.08), (40, 0.02)]),
    "elevated_cold_dome": dict(
        psfc=(990.0, 1010.0),
        t=[(1010, 275.0), (920, 272.0), (860, 281.0), (750, 280.0),
           (600, 270.0), (500, 259.0), (400, 247.0), (300, 233.0),
           (200, 219.0), (100, 212.0), (40, 214.0)],
        rh=[(1010, 0.75), (920, 0.85), (860, 0.97), (750, 0.95),
            (600, 0.80), (500, 0.70), (300, 0.45), (100, 0.05),
            (40, 0.01)]),
    "high_terrain_monsoon": dict(
        psfc=(600.0, 710.0),
        t=[(710, 296.0), (650, 291.5), (550, 281.0), (500, 275.5),
           (400, 262.0), (300, 245.0), (200, 222.0), (150, 210.0),
           (100, 198.0), (40, 210.0)],
        rh=[(710, 0.70), (600, 0.78), (500, 0.70), (400, 0.60),
            (300, 0.45), (200, 0.30), (100, 0.05), (40, 0.01)]),
    "dry_desert": dict(
        psfc=(880.0, 930.0),
        t=[(930, 314.0), (800, 301.5), (700, 291.5), (560, 276.5),
           (500, 270.0), (400, 257.0), (300, 241.0), (200, 220.0),
           (100, 200.0), (40, 212.0)],
        rh=[(930, 0.10), (700, 0.15), (560, 0.30), (500, 0.45),
            (400, 0.40), (300, 0.30), (100, 0.04), (40, 0.01)]),
    "tc_core": dict(
        psfc=(960.0, 1000.0),
        t=[(1000, 300.5), (900, 295.0), (850, 292.5), (700, 285.5),
           (500, 271.0), (400, 261.5), (300, 248.0), (200, 226.0),
           (150, 211.0), (100, 195.0), (40, 209.0)],
        rh=[(1000, 0.93), (850, 0.95), (700, 0.94), (500, 0.92),
            (300, 0.85), (200, 0.60), (100, 0.10), (40, 0.02)]),
    "very_dry": dict(
        psfc=(950.0, 1000.0),
        t=[(1000, 288.0), (850, 280.0), (700, 270.0), (500, 252.0),
           (300, 228.0), (200, 218.0), (100, 210.0), (40, 212.0)],
        rh=[(1000, 0.05), (850, 0.03), (500, 1.0e-4), (300, 1.0e-5),
            (40, 1.0e-6)]),
    "supersaturated_layers": dict(
        psfc=(995.0, 1010.0),
        t=[(1010, 298.0), (900, 291.0), (850, 288.0), (700, 280.0),
           (500, 264.0), (300, 239.0), (200, 219.0), (100, 196.0),
           (40, 209.0)],
        rh=[(1010, 0.95), (920, 1.04), (850, 1.06), (700, 0.97),
            (500, 1.02), (300, 0.70), (100, 0.10), (40, 0.02)]),
}


def _interp_logp(knots, p_hpa):
    ps = np.array([k[0] for k in knots], dtype=np.float64)
    vs = np.array([k[1] for k in knots], dtype=np.float64)
    order = np.argsort(ps)
    return np.interp(np.log(p_hpa), np.log(ps[order]), vs[order])


def _qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    es = np.minimum(es, 0.5 * p)
    return 0.622 * es / (p - es)


def _eta_levels(nz, rng, stretch):
    """WRF-like full-level eta: dense near the ground, stretched aloft."""
    x = np.linspace(0.0, 1.0, nz + 1)
    eta = 1.0 - (x ** stretch) * 0.35 - 0.65 * x ** (1.0 + 0.6 * stretch)
    eta = (eta - eta[-1]) / (eta[0] - eta[-1])
    return eta


def make_column(rng, regime, *, nz, ptop, dx, dt, cudt, branch,
                w_scale, jitter=1.0, rh_scale=1.0, wind=(5.0, 25.0),
                stretch=1.4, nca_in=-100.0, adapt=0):
    spec = REGIMES[regime]
    psfc = rng.uniform(*spec["psfc"]) * 100.0
    eta = _eta_levels(nz, rng, stretch)
    p_int = ptop + eta * (psfc - ptop)                 # Pa, nz+1, decreasing
    p_mid = 0.5 * (p_int[:-1] + p_int[1:])
    p_hpa = p_mid / 100.0
    t = _interp_logp(spec["t"], p_hpa)
    t = t + jitter * (rng.normal(0.0, 0.6) + rng.normal(0.0, 0.25, nz))
    rh = _interp_logp(spec["rh"], p_hpa) * rh_scale
    rh = rh * np.exp(jitter * rng.normal(0.0, 0.05, nz))
    qs = _qsat(t, p_mid)
    qv = np.clip(rh, 0.0, 1.10) * qs
    qv = np.maximum(qv, 1.0e-9)
    tv = t * (1.0 + 0.608 * qv)
    dz = RD * tv / G * np.log(p_int[:-1] / p_int[1:])
    rho = p_mid / (RD * t * (1.0 + qv * RV / RD)) * (1.0 + qv)
    pi = (p_mid / 1.0e5) ** (RD / CP)
    z_int = np.concatenate([[0.0], np.cumsum(dz)])
    ztop = z_int[-1]
    # Wind: a sheared profile with a random direction turn.
    u0, utop = wind
    frac = np.clip(z_int[:-1] / 12000.0, 0.0, 1.0)
    turn = rng.uniform(0.0, 1.2)
    speed = u0 + (utop - u0) * frac + rng.normal(0.0, 1.0, nz) * jitter
    ang = rng.uniform(0, 2 * math.pi) + turn * frac
    u = speed * np.cos(ang)
    v = speed * np.sin(ang)
    # Grid-scale vertical motion: one smooth mode plus noise; W0AVG is an
    # older running mean with its own amplitude so the update matters.
    shape_w = np.sin(np.pi * np.clip(z_int / max(ztop * 0.8, 1.0), 0, 1))
    w = w_scale * shape_w + rng.normal(0.0, 0.02, nz + 1) * jitter
    w[0] = 0.0
    shape_m = 0.5 * (shape_w[:-1] + shape_w[1:])
    w0avg = (w_scale * rng.uniform(0.5, 1.5) * shape_m
             + rng.normal(0.0, 0.01, nz) * jitter)
    stepcu = 1 if cudt <= 0.0 else max(int(math.floor(cudt * 60.0 / dt + 0.5)), 1)
    warm, fqr, fqi, fqs = BRANCHES[branch]
    f32 = np.float32
    return dict(
        regime=regime, branch=branch, nz=nz,
        ints=(nz, warm, fqr, fqi, fqs, stepcu, adapt, 1),
        scal=tuple(f32(x) for x in (dt, dx, cudt, nca_in)),
        u=u.astype(f32), v=v.astype(f32), t=t.astype(f32),
        qv=qv.astype(f32), p=p_mid.astype(f32), dz=dz.astype(f32),
        rho=rho.astype(f32), pi=pi.astype(f32), w0avg=w0avg.astype(f32),
        w=w.astype(f32),
        meta=dict(regime=regime, branch=branch, nz=nz, ptop=ptop, dx=dx,
                  dt=dt, cudt=cudt, stepcu=stepcu, w_scale=w_scale,
                  psfc=float(psfc), adapt=adapt))


def write_columns(path, columns):
    with open(path, "wb") as fh:
        fh.write(b"KFCOL1  ")
        fh.write(struct.pack("<i", len(columns)))
        for col in columns:
            fh.write(struct.pack("<8i", *col["ints"]))
            fh.write(np.array(col["scal"], dtype="<f4").tobytes())
            for name in ("u", "v", "t", "qv", "p", "dz", "rho", "pi",
                         "w0avg", "w"):
                fh.write(np.asarray(col[name], dtype="<f4").tobytes())
    with open(str(path) + ".json", "w", encoding="utf-8") as fh:
        json.dump([c["meta"] for c in columns], fh, indent=0)


def read_columns(path):
    """Inverse of :func:`write_columns` (used by the WOOF-side runner)."""
    data = open(path, "rb").read()
    if data[:8] != b"KFCOL1  ":
        raise ValueError("bad column file magic")
    (ncol,) = struct.unpack_from("<i", data, 8)
    off = 12
    cols = []
    for _ in range(ncol):
        ints = struct.unpack_from("<8i", data, off)
        off += 32
        scal = np.frombuffer(data, "<f4", 4, off)
        off += 16
        nz = ints[0]
        col = dict(ints=ints, scal=scal.copy())
        for name in ("u", "v", "t", "qv", "p", "dz", "rho", "pi", "w0avg"):
            col[name] = np.frombuffer(data, "<f4", nz, off).copy()
            off += 4 * nz
        col["w"] = np.frombuffer(data, "<f4", nz + 1, off).copy()
        off += 4 * (nz + 1)
        cols.append(col)
    return cols


OUT_SCALARS = ("raincv", "pratec", "nca", "cutop", "cubot", "shall")
OUT_ARRAYS = ("rthcuten", "rqvcuten", "rqccuten", "rqrcuten", "rqicuten",
              "rqscuten", "w0avg", "cldfra_dp", "cldfra_sh", "qc_kf",
              "qi_kf")


def read_outputs(path):
    data = open(path, "rb").read()
    if data[:8] != b"KFOUT1  ":
        raise ValueError("bad output file magic")
    (ncol,) = struct.unpack_from("<i", data, 8)
    off = 12
    outs = []
    for _ in range(ncol):
        nz, fatal, msgs, u98 = struct.unpack_from("<4i", data, off)
        off += 16
        rec = dict(nz=nz, fatal=fatal, messages=msgs, unit98=u98)
        sc = np.frombuffer(data, "<f4", 6, off)
        off += 24
        for name, value in zip(OUT_SCALARS, sc):
            rec[name] = np.float32(value)
        for name in OUT_ARRAYS:
            rec[name] = np.frombuffer(data, "<f4", nz, off).copy()
            off += 4 * nz
        outs.append(rec)
    return outs


def structured(seed):
    """The named-regime set: every regime crossed with the axes that move it."""
    rng = np.random.default_rng(seed)
    cols = []
    regimes = list(REGIMES)
    branches = list(BRANCHES)
    # (nz, ptop, dx, dt, cudt): grid configurations, chosen so 2400/dt is
    # sometimes not an integer (the shallow TIMEC reset) and cudt=0 is in.
    grids = [
        (50, 5000.0, 12000.0, 60.0, 0.0),
        (40, 5000.0, 27000.0, 120.0, 5.0),
        (60, 5000.0, 9000.0, 45.0, 0.0),
        (35, 10000.0, 36000.0, 180.0, 10.0),
        (45, 5000.0, 15000.0, 70.0, 5.0),     # 2400/70 is not an integer
        (50, 5000.0, 4000.0, 18.0, 0.0),
    ]
    w_scales = (-0.3, 0.0, 0.05, 0.15, 0.4, 1.2)
    i = 0
    for regime in regimes:
        for gi, (nz, ptop, dx, dt, cudt) in enumerate(grids):
            branch = branches[(i + gi) % len(branches)]
            w_scale = w_scales[(i * 7 + gi * 3) % len(w_scales)]
            cols.append(make_column(
                rng, regime, nz=nz, ptop=ptop, dx=dx, dt=dt, cudt=cudt,
                branch=branch, w_scale=w_scale))
        i += 1
        # Same regime, every branch, one strongly forced grid: the feedback
        # branches only differ when the column convects.
        for branch in branches:
            cols.append(make_column(
                rng, regime, nz=50, ptop=5000.0, dx=12000.0, dt=60.0,
                cudt=0.0, branch=branch, w_scale=0.6))
    # Edge cases.
    edge = [
        dict(regime="tropical_ocean", nz=8, ptop=20000.0, dx=30000.0,
             dt=180.0, cudt=0.0, branch="qs_qi", w_scale=0.8, stretch=1.0),
        dict(regime="tropical_ocean", nz=12, ptop=10000.0, dx=30000.0,
             dt=150.0, cudt=0.0, branch="qs_qi", w_scale=0.8, stretch=1.0),
        dict(regime="tc_core", nz=50, ptop=5000.0, dx=3000.0, dt=15.0,
             cudt=0.0, branch="qs_qi", w_scale=5.0),
        dict(regime="tc_core", nz=50, ptop=5000.0, dx=50000.0, dt=240.0,
             cudt=0.0, branch="qs_qi", w_scale=2.0),
        dict(regime="midlat_loaded_gun", nz=50, ptop=5000.0, dx=12000.0,
             dt=60.0, cudt=0.0, branch="qs_qi", w_scale=-2.0),
        dict(regime="midlat_loaded_gun", nz=80, ptop=5000.0, dx=12000.0,
             dt=60.0, cudt=0.0, branch="qs_qi", w_scale=0.5, stretch=1.2),
        dict(regime="midlat_loaded_gun", nz=128, ptop=5000.0, dx=12000.0,
             dt=60.0, cudt=0.0, branch="qs_qi", w_scale=0.5, stretch=1.2),
        dict(regime="trade_shallow", nz=60, ptop=5000.0, dx=12000.0,
             dt=7.5, cudt=0.0, branch="qs_qi", w_scale=0.1),
        dict(regime="trade_shallow", nz=60, ptop=5000.0, dx=12000.0,
             dt=72.0, cudt=5.0, branch="no_qs", w_scale=0.1),
        dict(regime="very_dry", nz=50, ptop=5000.0, dx=12000.0, dt=60.0,
             cudt=0.0, branch="qs_qi", w_scale=3.0),
        dict(regime="supersaturated_layers", nz=50, ptop=5000.0,
             dx=12000.0, dt=60.0, cudt=0.0, branch="qs_qi", w_scale=0.3,
             rh_scale=1.03),
        dict(regime="high_terrain_monsoon", nz=50, ptop=5000.0,
             dx=9000.0, dt=45.0, cudt=0.0, branch="qs_qi", w_scale=0.4),
        dict(regime="cold_snow", nz=50, ptop=5000.0, dx=12000.0, dt=60.0,
             cudt=0.0, branch="no_qs", w_scale=1.0),
        dict(regime="cold_snow", nz=50, ptop=5000.0, dx=12000.0, dt=60.0,
             cudt=0.0, branch="warm_rain", w_scale=1.0),
        dict(regime="elevated_cold_dome", nz=50, ptop=5000.0, dx=12000.0,
             dt=60.0, cudt=0.0, branch="qs_no_qi", w_scale=1.0),
        dict(regime="stable_night", nz=50, ptop=5000.0, dx=12000.0,
             dt=60.0, cudt=0.0, branch="qs_qi", w_scale=4.0),
    ]
    for kw in edge:
        cols.append(make_column(rng, **kw))
    # Adaptive-time-step W0AVG arm (module_cu_kfeta.F:222-230).
    for regime in ("tropical_ocean", "midlat_loaded_gun"):
        cols.append(make_column(
            rng, regime, nz=50, ptop=5000.0, dx=12000.0, dt=50.0, cudt=5.0,
            branch="qs_qi", w_scale=0.5, adapt=1))
    return cols


def sweep(seed, blocks, per_block):
    rng = np.random.default_rng(seed)
    regimes = list(REGIMES)
    branches = list(BRANCHES)
    cols = []
    for b in range(blocks):
        nz = int(rng.choice([30, 40, 45, 50, 60, 70]))
        ptop = float(rng.choice([5000.0, 5000.0, 10000.0]))
        dx = float(rng.choice([3000.0, 4000.0, 9000.0, 12000.0, 20000.0,
                               36000.0]))
        dt = float(rng.choice([15.0, 18.0, 30.0, 45.0, 60.0, 70.0, 72.0,
                               90.0, 120.0]))
        cudt = float(rng.choice([0.0, 0.0, 5.0, 10.0]))
        branch = branches[b % len(branches)]
        stretch = float(rng.uniform(1.1, 1.8))
        for _ in range(per_block):
            regime = regimes[int(rng.integers(len(regimes)))]
            cols.append(make_column(
                rng, regime, nz=nz, ptop=ptop, dx=dx, dt=dt, cudt=cudt,
                branch=branch, w_scale=float(rng.uniform(-0.5, 1.5)),
                jitter=float(rng.uniform(0.5, 2.0)),
                rh_scale=float(rng.uniform(0.85, 1.08)), stretch=stretch,
                wind=(float(rng.uniform(0, 10)), float(rng.uniform(5, 45)))))
    return cols


def real74(npz_path):
    """The two ERA5-initialised 1974-04-03 12Z d01 columns of
    tests/data/kf_real74_12z_columns.npz (a warm-sector column and a stable
    northern one), crossed with every feedback branch, four running-mean
    vertical velocities and two time steps.  The model's own T, qv, p (49
    levels), dz, exner and winds are used as stored; RHO is formed as in
    :func:`make_column`, the initial w is zero and W0AVG is a uniform
    prior mean."""
    f32 = np.float32
    cols = []
    with np.load(npz_path, allow_pickle=False) as data:
        for label in ("unstable", "stable"):
            t = data[f"{label}_temperature"].astype(np.float64)
            qv = data[f"{label}_qv"].astype(np.float64)
            p = data[f"{label}_pressure"].astype(np.float64)
            nz = t.size
            rho = p / (RD * t * (1.0 + qv * RV / RD)) * (1.0 + qv)
            for branch in BRANCHES:
                for w0 in (0.0, 0.05, 0.2, 0.6):
                    for dt, cudt in ((60.0, 0.0), (90.0, 5.0)):
                        stepcu = (1 if cudt <= 0.0 else
                                  max(int(math.floor(cudt * 60.0 / dt + 0.5)), 1))
                        warm, fqr, fqi, fqs = BRANCHES[branch]
                        cols.append(dict(
                            regime=f"real74_{label}", branch=branch, nz=nz,
                            ints=(nz, warm, fqr, fqi, fqs, stepcu, 0, 1),
                            scal=tuple(f32(x) for x in (dt, 12000.0, cudt,
                                                        -100.0)),
                            u=data[f"{label}_u"].astype(f32),
                            v=data[f"{label}_v"].astype(f32),
                            t=t.astype(f32), qv=qv.astype(f32),
                            p=p.astype(f32),
                            dz=data[f"{label}_dz"].astype(f32),
                            rho=rho.astype(f32),
                            pi=data[f"{label}_exner"].astype(f32),
                            w0avg=np.full(nz, w0, dtype=f32),
                            w=np.zeros(nz + 1, dtype=f32),
                            meta=dict(regime=f"real74_{label}", branch=branch,
                                      nz=nz, ptop=None, dx=12000.0, dt=dt,
                                      cudt=cudt, stepcu=stepcu, w_scale=w0,
                                      psfc=None, adapt=0)))
    return cols


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("structured", "sweep", "real74"))
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--per-block", type=int, default=64)
    ap.add_argument("--npz", help="kf_real74_12z_columns.npz (real74 only)")
    a = ap.parse_args(argv)
    if a.kind == "structured":
        cols = structured(a.seed)
    elif a.kind == "real74":
        cols = real74(a.npz)
    else:
        cols = sweep(a.seed, a.blocks, a.per_block)
    write_columns(a.out, cols)
    print(f"{len(cols)} columns -> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
