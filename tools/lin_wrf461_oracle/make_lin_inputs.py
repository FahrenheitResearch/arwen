"""Write the Purdue Lin column oracle's input file, deterministically.

Eight regimes of nine columns each (72 columns, 40 levels): deep
convective, stratiform, cold dry, warm rain only, supersaturated, near-zero
and subnormal species, high terrain, and phase-boundary temperatures.  Every
value is float32 and drawn from a fixed seed, so the file is a pure function
of this script.

Output (little-endian stream, read by run_lin.F90):
  int32 ncol, nz
  float32 dt[ncol], ht[ncol], rainnc[ncol], snownc[ncol], graupelnc[ncol]
  float32 th, qv, qc, qr, qi, qs, qg, rho, pii, p, z, dz8w  each (nz, ncol)
and a JSON sidecar naming each column's regime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NZ = 40
PER_REGIME = 9
REGIMES = ("deep_convective", "stratiform", "cold_dry", "warm_rain_only",
           "supersaturated", "near_zero_subnormal", "high_terrain",
           "phase_boundaries")
RD, RV, CP, G = 287.0, 461.6, 1004.5, 9.81
EP2 = RD / RV
DTS = (10.0, 20.0, 30.0, 45.0, 60.0, 72.0, 90.0, 120.0)

#: Species values at and around every threshold the scheme tests, plus
#: signed zeros and subnormals (run_lin keeps IEEE gradual underflow).
SPECIAL = np.array([
    0.0, -0.0, 1.4e-45, 3.0e-39, -2.0e-40, 1.17549435e-38, 1.0e-30,
    1.0e-20, 1.0e-12, 1.0e-10, np.nextafter(np.float32(1.0e-10),
                                            np.float32(1.0)),
    1.0e-8, np.nextafter(np.float32(1.0e-8), np.float32(1.0)), 9.99e-9,
    1.0e-5, 1.0e-4, np.nextafter(np.float32(1.0e-4), np.float32(0.0)),
    -1.0e-7, 3.0e-6, 2.0e-3], dtype=np.float32)


def qsat_water(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return EP2 * es / (p - es)


def sounding(rng, ht, psfc, tsfc, lapse, ztrop, nz=NZ):
    k = np.arange(nz + 1, dtype=np.float64)
    dz = 55.0 + 720.0 * (k[:nz] / (nz - 1)) ** 1.4
    dz = dz * rng.uniform(0.92, 1.08)
    z8w = ht + np.concatenate([[0.0], np.cumsum(dz)])
    zh = 0.5 * (z8w[:-1] + z8w[1:])
    t = np.where(zh - ht < ztrop - ht,
                 tsfc - lapse * (zh - ht),
                 tsfc - lapse * (ztrop - ht) + 1.5e-3 * (zh - ztrop))
    # hydrostatic pressure on the half levels
    tm = np.concatenate([[tsfc], t])
    p = np.empty(nz)
    pw = psfc
    zprev, tprev = ht, tsfc
    for kk in range(nz):
        tbar = 0.5 * (tprev + t[kk])
        p[kk] = pw * np.exp(-G * (zh[kk] - zprev) / (RD * tbar))
        pw, zprev, tprev = p[kk], zh[kk], t[kk]
    del tm
    return zh, z8w[1:] - z8w[:-1], t, p


def build(seed: int = 20261008):
    cols = []
    for r_index, regime in enumerate(REGIMES):
        rng = np.random.default_rng(seed + 101 * r_index)
        for c in range(PER_REGIME):
            ht = 0.0
            psfc = rng.uniform(98000.0, 102000.0)
            tsfc, lapse, ztrop = 296.0, 6.5e-3, 12000.0
            if regime == "deep_convective":
                tsfc = rng.uniform(299.0, 306.0)
            elif regime == "stratiform":
                tsfc = rng.uniform(280.0, 290.0)
            elif regime == "cold_dry":
                tsfc, lapse, ztrop = rng.uniform(238.0, 256.0), 5.0e-3, 8000.0
            elif regime == "warm_rain_only":
                tsfc = rng.uniform(296.0, 302.0)
            elif regime == "supersaturated":
                tsfc = rng.uniform(270.0, 300.0)
            elif regime == "near_zero_subnormal":
                tsfc = rng.uniform(265.0, 295.0)
            elif regime == "high_terrain":
                ht = rng.uniform(2800.0, 4600.0)
                psfc = 101325.0 * np.exp(-ht / 8000.0) * rng.uniform(0.98, 1.02)
                tsfc = rng.uniform(262.0, 285.0)
            else:   # phase_boundaries
                tsfc = rng.uniform(283.0, 293.0)
            zh, dz, t, p = sounding(rng, ht, psfc, tsfc, lapse, ztrop)
            nz = NZ
            rh = np.clip(rng.uniform(0.55, 0.95) - 0.25 * (zh - ht) / 12000.0
                         + rng.normal(0.0, 0.05, nz), 0.02, 0.99)
            qc = np.zeros(nz); qr = np.zeros(nz); qi = np.zeros(nz)
            qs = np.zeros(nz); qg = np.zeros(nz)
            tc = t - 273.15
            amp = rng.uniform(0.3, 1.6)
            if regime == "deep_convective":
                rh = np.where((zh > 500) & (zh < 11000), rng.uniform(0.97, 1.03, nz), rh)
                qc = np.where((tc > -38) & (zh > 600) & (zh < 9000),
                              amp * rng.uniform(2e-4, 3e-3, nz), 0.0)
                qr = np.where(zh < 6500, amp * rng.uniform(1e-4, 6e-3, nz), 0.0)
                qg = np.where((tc < 2) & (zh < 11500), amp * rng.uniform(1e-4, 8e-3, nz), 0.0)
                qs = np.where((tc < -3), amp * rng.uniform(5e-5, 2e-3, nz), 0.0)
                qi = np.where((tc < -12), amp * rng.uniform(1e-6, 1.5e-3, nz), 0.0)
            elif regime == "stratiform":
                melt = tc > 0
                qs = np.where(~melt & (zh < 9000), amp * rng.uniform(5e-5, 1.2e-3, nz), 0.0)
                qr = np.where(melt, amp * rng.uniform(5e-5, 8e-4, nz), 0.0)
                qi = np.where(tc < -15, amp * rng.uniform(1e-6, 2e-4, nz), 0.0)
                qc = np.where(np.abs(tc) < 6, amp * rng.uniform(0, 3e-4, nz), 0.0)
                qg = np.where((tc < 0) & (tc > -10), amp * rng.uniform(0, 2e-4, nz), 0.0)
                rh = np.where(zh < 8000, rng.uniform(0.85, 1.01, nz), rh)
            elif regime == "cold_dry":
                rh = rng.uniform(0.15, 0.8, nz)
                qi = rng.uniform(0, 1e-5, nz) * (rng.random(nz) < 0.6)
                qs = rng.uniform(0, 6e-5, nz) * (rng.random(nz) < 0.5)
                qc = np.where(tc < -36, rng.uniform(0, 2e-5, nz), 0.0) * (rng.random(nz) < 0.5)
                qg = rng.uniform(0, 2e-5, nz) * (rng.random(nz) < 0.3)
            elif regime == "warm_rain_only":
                warm = (zh < 3500) & (tc > 2)
                qc = np.where(warm, amp * rng.uniform(1e-4, 2.5e-3, nz), 0.0)
                qr = np.where(warm, amp * rng.uniform(0, 3e-3, nz), 0.0)
                rh = np.where(warm, rng.uniform(0.9, 1.0, nz), rh)
            elif regime == "supersaturated":
                rh = rng.uniform(0.98, 1.4, nz)
                pick = rng.random(nz)
                qc = np.where(pick < 0.35, rng.uniform(0, 1e-3, nz), 0.0)
                qi = np.where((pick > 0.25) & (pick < 0.6) & (tc < 0),
                              rng.uniform(0, 4e-4, nz), 0.0)
                qi = np.where((pick > 0.85) & (tc < 0), rng.uniform(0, 9e-6, nz), qi)
                qs = np.where(pick > 0.7, rng.uniform(0, 3e-4, nz), 0.0)
            elif regime == "near_zero_subnormal":
                n = len(SPECIAL)
                off = c * 7
                qc = SPECIAL[(np.arange(nz) + off) % n].astype(np.float64)
                qr = SPECIAL[(np.arange(nz) * 3 + off + 1) % n].astype(np.float64)
                qi = SPECIAL[(np.arange(nz) * 5 + off + 2) % n].astype(np.float64)
                qs = SPECIAL[(np.arange(nz) * 7 + off + 3) % n].astype(np.float64)
                qg = SPECIAL[(np.arange(nz) * 11 + off + 4) % n].astype(np.float64)
            elif regime == "high_terrain":
                qs = np.where(tc < 1, amp * rng.uniform(1e-5, 1.5e-3, nz), 0.0)
                qi = np.where(tc < -10, amp * rng.uniform(1e-6, 3e-4, nz), 0.0)
                qc = np.where((tc > -25) & (zh - ht < 3000), amp * rng.uniform(0, 6e-4, nz), 0.0)
                qg = np.where((tc < 0) & (zh - ht < 4000), amp * rng.uniform(0, 1.5e-3, nz), 0.0)
                qr = np.where(tc > -5, amp * rng.uniform(0, 1e-3, nz), 0.0)
                rh = np.where(zh - ht < 5000, rng.uniform(0.8, 1.05, nz), rh)
            else:   # phase_boundaries: temperatures pinned near the branch points
                targets = np.array([273.15, 273.16, 253.15, 242.15, 233.15,
                                    248.15, 272.0, 274.0])
                lev = np.arange(2, nz - 4, 3)
                for m, kk in enumerate(lev):
                    t[kk] = targets[(m + c) % len(targets)] + rng.choice(
                        [0.0, 1e-3, -1e-3, 0.02, -0.02])
                qr = np.where(tc < 8, rng.choice([0.0, 9.99e-5, 1.0e-4, 1.01e-4, 2e-3], nz), 0.0)
                qs = np.where(tc < 3, rng.choice([0.0, 9.99e-5, 1.0e-4, 1.01e-4, 6e-4, 1.5e-3], nz), 0.0)
                qg = np.where(tc < 4, rng.choice([0.0, 1e-8, 5e-4, 4e-3], nz), 0.0)
                qc = rng.choice([0.0, 1e-10, 1e-6, 4e-4, 1.5e-3], nz)
                qi = np.where(tc < 1, rng.choice([0.0, 9e-6, 1e-5, 2e-4, 1.2e-3], nz), 0.0)
                rh = rng.uniform(0.85, 1.08, nz)
            qv = rh * qsat_water(t, p)
            if regime == "near_zero_subnormal":
                qv = np.where(np.arange(nz) % 9 == 4, SPECIAL[(np.arange(nz) + c) % 8], qv)
            pii = (p / 1.0e5) ** (RD / CP)
            th = t / pii
            rho = p / (RD * t * (1.0 + 0.608 * np.maximum(qv, 0.0)))
            dt = DTS[(r_index * PER_REGIME + c) % len(DTS)]
            cols.append(dict(
                regime=regime, dt=dt, ht=ht,
                rainnc=rng.uniform(0.0, 80.0), snownc=rng.uniform(0.0, 20.0),
                graupelnc=rng.uniform(0.0, 10.0),
                th=th, qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg,
                rho=rho, pii=pii, p=p, z=zh, dz8w=dz))
    return cols


FIELDS3 = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "rho", "pii", "p",
           "z", "dz8w")
FIELDS2 = ("dt", "ht", "rainnc", "snownc", "graupelnc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    cols = build()
    ncol = len(cols)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "lin-inputs.bin", "wb") as f:
        np.array([ncol, NZ], dtype="<i4").tofile(f)
        for name in FIELDS2:
            np.array([c[name] for c in cols], dtype="<f4").tofile(f)
        for name in FIELDS3:
            # (nz, ncol) C order is Fortran's (ncol, nz) column-major order
            arr = np.stack([np.asarray(c[name], dtype=np.float32)
                            for c in cols], axis=1)
            arr.astype("<f4").tofile(f)
    (args.out_dir / "lin-regimes.json").write_text(
        json.dumps([c["regime"] for c in cols]) + "\n", encoding="utf-8")
    print(f"wrote {ncol} columns x {NZ} levels")


if __name__ == "__main__":
    main()
