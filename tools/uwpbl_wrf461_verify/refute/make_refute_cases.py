"""Adversarial column cases for the UW PBL (bl_pbl_physics=9) WRF v4.6.1 oracle.

    python make_refute_cases.py OUTDIR

Written by the refute check of lane/verify-uw-pbl.  Every column here is
built by this file's own profile generator (not make_cases.build_column),
so none of the regimes, grids or time steps repeats the lane's 370
columns.  The format is run_camuwpbl.F90's case file (header 'UWPB',
version 1, ncol, nk, nsteps, dt, then float32 arrays in the order the
Fortran reads them).

Sets:
  refute-fuzz-*    eight grids, 200 random columns each, 3 steps: standard
                   atmosphere style temperature with a random boundary layer,
                   tropopause and stratosphere, random multi-layer cloud
                   (liquid, mixed, ice), random wind direction and shear,
                   random LW cooling/heating, fluxes over the whole physical
                   range.  Grids reach model tops of 45 and 52 km (pressure
                   near 1 hPa), dt from 6 s to 180 s including dt = 100/3.
  refute-hightop   60 columns, 80 levels to 60 km, dt 20 s, 3 steps, with a
                   warm stratopause (255-295 K at 45-52 km): the saturation
                   lookups' es > p branches (aqsat and vqsatd: qs < 0 -> qs = 1,
                   es = p; qs = min(1, qs); gam = 0 when qs == 1).
  refute-long      48 columns, 60 levels to 25 km, dt 60 s, 60 steps: a long
                   carried history of kvm3d/kvh3d/tauresx/tauresy.
  refute-qncA/B    the same 100 columns twice, A with random liquid number,
                   B with liquid number zero.  The engine passes QNC_CURR =
                   0 (physics.py _run_uwpbl) on the claim that WRF's
                   liquid number cannot reach any camuwpbl output; A and B
                   must give byte-identical WRF outputs.
  refute-nanprobe  calm level-1 wind (u = v = 0 exactly) and ust = 0: WRF
                   divides by the level-1 speed; graded separately.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

MAGIC = 1431785538
R_D = 287.0
CP = 7.0 * R_D / 2.0
RCP = R_D / CP
G = 9.81
P0 = 1.0e5
EP = 0.622
MASS = ("u", "v", "th", "rho", "qv", "qc", "qi", "qnc", "qni", "p", "z", "t",
        "cldfra", "exner", "rthratenlw", "wsedl3d")
FULL = ("p8w", "z_at_w")
SFC = ("hfx", "qfx", "ust", "ht")
SEED = 2026100742


def esat(t):
    """Bolton over water above 253 K, Murphy-Koop style ice fit below."""
    t = np.asarray(t, dtype=np.float64)
    w = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    i = 611.15 * np.exp(22.452 * (t - 273.15) / (t - 0.61))
    return np.where(t >= 253.15, w, i)


def grid(rng, nk, ztop):
    dz0 = rng.uniform(8.0, 80.0)
    fac = rng.uniform(1.04, 1.2)
    cap = rng.uniform(400.0, 2500.0)
    dz = np.empty(nk)
    d = dz0
    for k in range(nk):
        dz[k] = d
        d = min(d * fac, cap)
    zi = np.concatenate([[0.0], np.cumsum(dz)])
    return zi * (ztop / zi[-1])


def temperature(rng, z, warm_stratopause=False):
    t0 = rng.uniform(222.0, 312.0)
    blt = int(rng.integers(0, 4))
    h = rng.uniform(50.0, 2500.0)
    lapse = rng.uniform(4.5e-3, 8.5e-3)
    dtinv = rng.uniform(2.0, 15.0)
    ztp = rng.uniform(8000.0, 17500.0)
    tmin_tp = rng.uniform(188.0, 228.0)
    zsp = rng.uniform(45000.0, 52000.0)
    tsp = rng.uniform(255.0, 295.0) if warm_stratopause else         rng.uniform(245.0, 272.0)
    zbase = ztp + rng.uniform(2000.0, 9000.0)
    out = np.empty_like(z)
    for k, zz in enumerate(z):
        above = max(0.0, zz - h)
        if blt == 0:      # superadiabatic skin under a mixed layer
            tb = t0 + 3.0 * max(0.0, 1.0 - zz / 80.0) - 0.0098 * min(zz, h)                 - lapse * above
        elif blt == 1:    # well mixed
            tb = t0 - 0.0098 * min(zz, h) - lapse * above
        elif blt == 2:    # surface-based inversion
            tb = t0 + dtinv * min(zz, h) / h - lapse * above
        else:             # elevated inversion capping a mixed layer
            tb = t0 - 0.0098 * min(zz, h) + (dtinv if zz > h else 0.0)                 - lapse * above
        tt = max(tb, tmin_tp)
        if zz > zbase:
            frac = min(1.0, (zz - zbase) / (zsp - zbase))
            tt = max(tt, tmin_tp + (tsp - tmin_tp) * frac ** 1.3)
        if zz > zsp:
            tt = tsp - 2.6e-3 * (zz - zsp)
        out[k] = tt
    return out


def column(rng, nk, ztop, *, warm_stratopause=False):
    zi = grid(rng, nk, ztop)
    zm = 0.5 * (zi[1:] + zi[:-1])
    ht = rng.choice([rng.uniform(-80.0, 50.0), rng.uniform(0.0, 1500.0),
                     rng.uniform(1500.0, 4200.0)])
    psfc = 101325.0 * np.exp(-ht / 8000.0) * rng.uniform(0.97, 1.03)
    t = temperature(rng, zm, warm_stratopause)
    # moisture: RH profile, stratospheric floor of a few ppmv
    rh0 = rng.uniform(0.05, 1.0)
    hq = rng.uniform(1500.0, 5000.0)
    # hydrostatic pressure (virtual temperature iterated on qv)
    p8w = np.empty(nk + 1)
    p8w[0] = psfc
    qv = np.zeros(nk)
    for k in range(nk):
        dz = zi[k + 1] - zi[k]
        pm = p8w[k]
        for _ in range(3):
            pmid = 0.5 * (p8w[k] + pm)
            es = float(esat(t[k]))
            rh = rh0 * np.exp(-zm[k] / hq)
            q = EP * rh * es / max(pmid - es, 0.5 * pmid)
            q = max(min(q, 0.04), rng.uniform(2.5e-6, 5e-6))
            qv[k] = q
            tv = t[k] * (1.0 + 0.608 * q)
            pm = p8w[k] * np.exp(-G * dz / (R_D * tv))
        p8w[k + 1] = pm
    p = 0.5 * (p8w[1:] + p8w[:-1])
    if rng.uniform() < 0.05:
        qv[:] = 0.0                       # bone dry, exactly zero
    exner = (p / P0) ** RCP
    th = t / exner
    # clouds: 0-3 random layers below the tropopause
    qc = np.zeros(nk); qi = np.zeros(nk); qnc = np.zeros(nk); qni = np.zeros(nk)
    cf = np.zeros(nk)
    lw = np.full(nk, -rng.uniform(0.5, 3.0) / 86400.0)
    for _ in range(int(rng.integers(0, 4))):
        zb = rng.uniform(0.0, 11000.0)
        dzc = rng.uniform(30.0, 4000.0)
        lay = (zm >= zb) & (zm < zb + dzc)
        if not lay.any():
            continue
        frac = rng.choice([1.0, rng.uniform(0.01, 1.0)])
        cf[lay] = np.maximum(cf[lay], frac)
        for k in np.flatnonzero(lay):
            if t[k] > 253.0:
                qc[k] += frac * rng.uniform(1e-6, 1.2e-3)
                qnc[k] = rng.uniform(1e7, 4e8)
            if t[k] < 273.0:
                qi[k] += frac * rng.uniform(1e-7, 2e-4)
                qni[k] = rng.uniform(1e3, 1e7)
            es = float(esat(t[k]))
            qv[k] = max(qv[k], EP * es / max(p[k] - es, 0.5 * p[k]))
        ktop = int(np.flatnonzero(lay)[-1])
        lw[ktop] = -rng.uniform(1e-5, 1.2e-2)
        if rng.uniform() < 0.3:
            lw[int(np.flatnonzero(lay)[0])] = rng.uniform(1e-5, 5e-4)
    # wind: random direction turning with height, shear, optional jet
    spd1 = rng.choice([rng.uniform(0.05, 1.0), rng.uniform(1.0, 30.0)])
    d0 = rng.uniform(0.0, 2.0 * np.pi)
    turn = rng.uniform(-1.5, 1.5)
    shear = rng.uniform(0.0, 4e-3)
    spd = spd1 + shear * zm
    if rng.uniform() < 0.3:
        zj = rng.uniform(100.0, 1500.0)
        spd += rng.uniform(5.0, 35.0) * np.exp(-((zm - zj) / (0.5 * zj)) ** 2)
    dirn = d0 + turn * np.tanh(zm / 3000.0)
    u = spd * np.cos(dirn) + rng.normal(0.0, 0.1, nk)
    v = spd * np.sin(dirn) + rng.normal(0.0, 0.1, nk)
    if rng.uniform() < 0.1:
        u[0] = 0.0                        # one component exactly zero
    hfx = rng.choice([rng.uniform(-300.0, 0.0), rng.uniform(0.0, 900.0),
                      rng.uniform(-5.0, 5.0)])
    qfx = rng.choice([rng.uniform(-1e-5, 0.0), rng.uniform(0.0, 5e-4)])
    ust = rng.choice([rng.uniform(0.005, 0.1), rng.uniform(0.1, 1.5)])
    wsed = np.zeros(nk)
    if rng.uniform() < 0.15:
        wsed = np.where(qc > 0, rng.uniform(0.0, 0.05, nk), 0.0)
    col = {"u": u, "v": v, "th": th, "qv": qv, "qc": qc, "qi": qi, "qnc": qnc,
           "qni": qni, "p": p, "z": zm + ht, "cldfra": cf, "exner": exner,
           "rthratenlw": lw, "wsedl3d": wsed, "p8w": p8w, "z_at_w": zi + ht,
           "hfx": np.array([hfx]), "qfx": np.array([qfx]),
           "ust": np.array([ust]), "ht": np.array([ht])}
    col = {k: np.asarray(a, dtype=np.float64).astype(np.float32)
           for k, a in col.items()}
    # WRF: t_phy = th_phy * pi_phy in single precision; rho from phy_prep
    col["t"] = (col["th"] * col["exner"]).astype(np.float32)
    col["rho"] = (col["p"].astype(np.float64) /
                  (R_D * col["t"].astype(np.float64) *
                   (1.0 + 0.608 * col["qv"].astype(np.float64)))
                  ).astype(np.float32)
    return col


def write(path, cols, nk, nsteps, dt):
    with open(path, "wb") as fh:
        np.array([MAGIC, 1, len(cols), nk, nsteps], dtype="<i4").tofile(fh)
        np.array([dt], dtype="<f4").tofile(fh)
        for name in MASS + FULL + SFC:
            np.stack([c[name] for c in cols]).astype("<f4").tofile(fh)


FUZZ_GRIDS = ((50, 20000.0, 45.0), (60, 30000.0, 30.0), (80, 45000.0, 20.0),
              (100, 52000.0, 10.0), (41, 15000.0, 90.0), (33, 12000.0, 180.0),
              (150, 25000.0, 6.0), (45, 18000.0, float(np.float32(100.0 / 3.0))))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    index = {"seed": SEED, "sets": {}}

    def emit(name, cols, nk, nsteps, dt, note):
        write(out / f"cases-{name}.bin", cols, nk, nsteps, dt)
        index["sets"][name] = {"nk": nk, "dt": dt, "nsteps": nsteps,
                               "ncol": len(cols), "note": note}

    for nk, ztop, dt in FUZZ_GRIDS:
        cols = [column(rng, nk, ztop) for _ in range(200)]
        emit(f"refute-fuzz-n{nk}", cols, nk, 3, dt,
             f"random columns, top {ztop:g} m")
    cols = [column(rng, 80, 60000.0, warm_stratopause=True) for _ in range(60)]
    emit("refute-hightop", cols, 80, 3, 20.0, "warm stratopause, top 60 km")
    cols = [column(rng, 60, 25000.0) for _ in range(48)]
    emit("refute-long", cols, 60, 60, 60.0, "60-step carried history")
    cols = [column(rng, 50, 20000.0) for _ in range(100)]
    for c in cols:                         # make sure liquid number is live
        live = c["qc"] > 0
        c["qnc"][live] = np.float32(2e8)
        c["qnc"][~live] = np.float32(5e6)
    emit("refute-qncA", cols, 50, 3, 40.0, "liquid number nonzero")
    colsb = []
    for c in cols:
        d = {k: a.copy() for k, a in c.items()}
        d["qnc"][:] = 0.0
        colsb.append(d)
    emit("refute-qncB", colsb, 50, 3, 40.0, "same columns, liquid number 0")
    probes = []
    for kind in ("calm", "ust0", "calm-ust0"):
        for _ in range(4):
            c = column(rng, 44, 16500.0)
            if kind in ("calm", "calm-ust0"):
                c["u"][0] = 0.0; c["v"][0] = 0.0
            if kind in ("ust0", "calm-ust0"):
                c["ust"][:] = 0.0
            probes.append(c)
    emit("refute-nanprobe", probes, 44, 2, 60.0,
         "calm level-1 wind x4, ust 0 x4, both x4")
    (out / "refute-index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(json.dumps({k: v["ncol"] for k, v in index["sets"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
