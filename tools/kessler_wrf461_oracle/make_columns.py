#!/usr/bin/env python3
"""Build the Kessler (mp_physics=1) column-oracle inputs.

Writes one stream-unformatted little-endian file that run_kessler.F90 reads
and one ``.npz`` holding the same float32 words for the GPU comparison:

    int32 ncol, nz, ncase
    float32 dt[ncase]
    int32   nsteps[ncase]
    float32 t, qv, qc, qr, rho, pii, z, dz8w   each (nz, ncol) C order
    float32 rainnc[ncol]

C order (nz, ncol) is Fortran order (ncol, nz), which is WRF's (i, k) with
j = 1, and also the GPU layout (nz, 1, ncol).  The inputs follow WRF's
moist_physics_prep_em contract: full (dry) potential temperature, dry
density, full-pressure Exner, half-level heights and layer depths.

The columns are built in float64 from a hydrostatic profile per regime and
rounded once to float32.  Regimes and their edge cases are listed in
REGIMES; every column carries its regime name in the ``.npz``.  The
committed ``.npz`` is the fixture of record: numpy's float64 libm may round
the last bit of a profile differently on another platform, so regenerating
is not guaranteed to reproduce the same words, and nothing depends on it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NZ = 50
PER_REGIME = 8
SEED = 20261007
# (dt seconds, steps): small, convection-permitting, mesoscale, coarse, and a
# large step that forces many sedimentation splits.
CASES = ((5.0, 4), (18.0, 8), (60.0, 6), (120.0, 4), (300.0, 3))

RD, RV, CP, G = 287.0, 461.6, 1004.5, 9.81
EP2 = RD / RV


def _qvs(T, p):
    es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    return EP2 * es / (p - es)


def _grid(zs, ztop, b):
    s = np.linspace(0.0, 1.0, NZ + 1)
    zf = zs + ztop * np.expm1(b * s) / np.expm1(b)
    return 0.5 * (zf[:-1] + zf[1:]), np.diff(zf), zf


def _thermo(z, zs, ps, tsfc, lapse, ttrop, inv_dt=0.0, inv_h=1.0):
    agl = z - zs
    T = tsfc - lapse * agl
    if inv_dt:
        T = np.where(agl < inv_h, tsfc + inv_dt * agl / inv_h,
                     tsfc + inv_dt - lapse * (agl - inv_h))
    T = np.maximum(T, ttrop)
    p = np.empty_like(z)
    zprev, pprev, tprev = zs, ps, (tsfc if not inv_dt else tsfc)
    for k in range(z.size):
        tm = 0.5 * (tprev + T[k])
        p[k] = pprev * np.exp(-G * (z[k] - zprev) / (RD * tm))
        zprev, pprev, tprev = z[k], p[k], T[k]
    pii = (p / 1.0e5) ** (RD / CP)
    return T, p, pii


def _bump(z, z0, z1):
    """Smooth 0..1..0 bump between z0 and z1."""
    x = (z - z0) / max(z1 - z0, 1.0)
    return np.where((x > 0) & (x < 1), np.sin(np.pi * np.clip(x, 0, 1)) ** 2,
                    0.0)


def build(rng):
    cols, names = [], []

    def add(name, zs, ps, tsfc, lapse, ttrop, rh, qc, qr, *, b=2.5,
            ztop=20000.0, inv=(0.0, 1.0), rainnc=0.0):
        z, dz, _ = _grid(zs, ztop, b)
        T, p, pii = _thermo(z, zs, ps, tsfc, lapse, ttrop, *inv)
        agl = z - zs
        qv = rh(agl) * _qvs(T, p)
        rho = p / (RD * T * (1.0 + qv / EP2))
        cols.append(dict(t=T / pii, qv=qv, qc=qc(agl), qr=qr(agl), rho=rho,
                         pii=pii, z=z, dz8w=dz, rainnc=rainnc))
        names.append(name)

    u = rng.uniform
    for _ in range(PER_REGIME):          # deep convection
        a, r = u(0.5e-3, 3e-3), u(2e-3, 12e-3)
        add("convective", u(0, 300), 100000.0, u(300, 306), 0.0068, 205.0,
            lambda h, s=u(1.01, 1.03): np.where(h < 1200, 0.85,
                                                np.where(h < 10000, s, 0.3)),
            lambda h, a=a: a * _bump(h, 1500, 9500),
            lambda h, r=r: r * _bump(h, -2500, 7000),
            rainnc=u(0, 40))
    for _ in range(PER_REGIME):          # stratiform rain
        add("stratiform", u(0, 500), 100500.0, u(285, 293), 0.0060, 210.0,
            lambda h, s=u(0.95, 1.01): np.where(h < 7000, s, 0.5),
            lambda h, a=u(1e-4, 5e-4): a * _bump(h, 2000, 6000),
            lambda h, r=u(1e-4, 1e-3): r * _bump(h, -5000, 5000))
    for _ in range(PER_REGIME):          # stable night, fog
        add("stable_night", u(0, 800), 101000.0, u(270, 285), 0.0060, 212.0,
            lambda h, s=u(1.0, 1.005): np.where(h < 120, s, 0.45),
            lambda h, a=u(5e-5, 3e-4): np.where(h < 150, a, 0.0),
            lambda h, d=u(0, 2e-6): np.where(h < 100, d, 0.0),
            inv=(u(5, 10), u(200, 400)), b=4.0)
    for _ in range(PER_REGIME):          # cold arctic: supercooled water
        add("cold_arctic", u(0, 400), 101500.0, u(240, 262), 0.0055, 200.0,
            lambda h, s=u(1.0, 1.02): np.where(h < 3500, s, 0.6),
            lambda h, a=u(5e-5, 4e-4): a * _bump(h, 500, 3000),
            lambda h, r=u(1e-5, 2e-4): r * _bump(h, -3000, 3000),
            inv=(u(2, 6), u(150, 300)))
    for _ in range(PER_REGIME):          # tropical ocean: shallow cu, drizzle
        add("ocean", 0.0, u(101000, 101500), u(299, 302), 0.0062, 195.0,
            lambda h, s=u(0.8, 0.9): np.where(h < 600, s,
                                              np.where(h < 2500, 1.012, 0.6)),
            lambda h, a=u(2e-4, 1e-3): a * _bump(h, 600, 2500),
            lambda h, r=u(1e-5, 3e-4): r * _bump(h, -2000, 2000),
            rainnc=u(0, 5))
    for _ in range(PER_REGIME):          # high terrain
        zs = u(2500, 4500)
        ps = 101325.0 * (1 - 2.25577e-5 * zs) ** 5.25588
        add("high_terrain", zs, ps, u(270, 290), 0.0070, 205.0,
            lambda h, s=u(1.0, 1.02): np.where(h < 6000, s, 0.4),
            lambda h, a=u(2e-4, 2e-3): a * _bump(h, 800, 6000),
            lambda h, r=u(5e-4, 6e-3): r * _bump(h, -3000, 5000),
            ztop=16000.0)
    for _ in range(PER_REGIME):          # desert virga
        add("desert_virga", u(500, 1500), u(90000, 96000), u(310, 318),
            0.0095, 210.0,
            lambda h, d=u(0.05, 0.2): np.where(h < 5000, d, 0.9),
            lambda h, a=u(1e-5, 2e-4): a * _bump(h, 6000, 8000),
            lambda h, r=u(2e-4, 2e-3): r * _bump(h, 3500, 7500))
    for _ in range(PER_REGIME):          # clear and dry: the no-op path
        add("clear_dry", u(0, 2000), 100000.0, u(275, 305), 0.0065, 210.0,
            lambda h, d=u(0.2, 0.8): d + 0 * h,
            lambda h: 0 * h, lambda h: 0 * h, rainnc=u(0, 100))
    for _ in range(PER_REGIME):          # torrential rain on thin layers
        add("heavy_rain_thin", u(0, 200), 100000.0, u(298, 304), 0.0065,
            205.0,
            lambda h, s=u(1.0, 1.02): np.where(h < 9000, s, 0.4),
            lambda h, a=u(1e-3, 4e-3): a * _bump(h, 1000, 9000),
            lambda h, r=u(10e-3, 25e-3): r * _bump(h, -4000, 4000),
            b=4.0, rainnc=u(0, 200))
    for _ in range(PER_REGIME):          # supersaturated, no condensate yet
        add("saturated_onset", u(0, 1000), 100000.0, u(280, 300), 0.0060,
            210.0,
            lambda h, s=u(1.0, 1.05): np.where(h < 8000, s, 0.7),
            lambda h: 0 * h, lambda h: 0 * h)
    for n in range(PER_REGIME):          # autoconversion threshold c2 = 1 g/kg
        base = np.float32(0.001)
        lad = np.array([np.nextafter(base, np.float32(-1)), base,
                        np.nextafter(base, np.float32(1))], np.float32)
        def qc_thr(h, n=n, lad=lad):
            q = np.zeros_like(h)
            idx = np.arange(h.size)
            q[:] = lad[(idx + n) % 3]
            q[idx % 5 == 4] = 0.0
            return q
        add("autoconv_threshold", u(0, 500), 100000.0, u(285, 300), 0.0062,
            210.0, lambda h: 1.0 + 0 * h, qc_thr,
            lambda h, n=n: (0.0 if n % 2 else 1e-4) + 0 * h)
    # edge values: built from a mild saturated column then overwritten
    for n in range(PER_REGIME):
        add("edge_values", u(0, 500), 100000.0, u(285, 300), 0.0065, 210.0,
            lambda h: np.where(h < 6000, 1.005, 0.5),
            lambda h: 3e-4 * _bump(h, 1000, 5000),
            lambda h: 1e-3 * _bump(h, -3000, 4000))
        c = cols[-1]
        if n == 0:   # subnormal and normal-boundary words in qc/qr/qv
            sub = np.array([1.4e-45, 1e-40, 1.1754942e-38, 1.17549435e-38,
                            2e-38, 1e-30], np.float32)
            for i, v in enumerate(sub):
                c["qr"][2 * i] = v
                c["qc"][2 * i + 1] = v
                c["qr"][20 + i] = v
        elif n == 1:  # small negatives (advection undershoot)
            c["qc"][::3] = -1e-8
            c["qr"][1::3] = -1e-6
            c["qv"][40:] = -1e-9
        elif n == 2:  # signed zeros
            c["qc"][::2] = -0.0
            c["qr"][1::2] = -0.0
        elif n == 3:  # dry vapour under rain: evaporation limited by qr
            c["qv"][:] = 0.0
        elif n == 4:  # rain only in the top level: the top-flux quirk
            c["qr"][:] = 0.0
            c["qr"][-1] = 5e-3
            c["qc"][-1] = 2e-3
        elif n == 5:  # rain only in the lowest level
            c["qr"][:] = 0.0
            c["qr"][0] = 8e-3
        elif n == 6:  # one extreme rain level: many splits
            c["qr"][5] = 0.05
            c["dz8w"][:3] *= 0.25
        elif n == 7:  # constant density, pii == 1 at the ground
            c["rho"][:] = c["rho"][0]
            c["pii"][0] = 1.0
    for _ in range(PER_REGIME):          # random mixture of everything
        add("random_mix", u(0, 3000), 100000.0, u(260, 310), u(0.004, 0.009),
            u(195, 220),
            lambda h, a=u(0.3, 1.05), b=u(0.3, 1.05):
                np.where(h < u(1000, 8000), a, b),
            lambda h, a=u(0, 3e-3): a * _bump(h, u(0, 3000), u(4000, 12000)),
            lambda h, r=u(0, 15e-3): r * _bump(h, u(-5000, 0), u(2000, 9000)),
            b=u(1.5, 4.5), rainnc=u(0, 300))
    return cols, names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    cols, names = build(np.random.default_rng(SEED))
    ncol = len(cols)
    fields = ("t", "qv", "qc", "qr", "rho", "pii", "z", "dz8w")
    arr = {f: np.stack([np.asarray(c[f], np.float32) for c in cols], axis=1)
           for f in fields}                                 # (nz, ncol)
    rainnc = np.asarray([c["rainnc"] for c in cols], np.float32)
    dts = np.asarray([d for d, _ in CASES], np.float32)
    steps = np.asarray([s for _, s in CASES], np.int32)
    for f in fields:
        assert np.all(np.isfinite(arr[f])), f
    assert np.all(arr["dz8w"] > 0) and np.all(np.diff(arr["z"], axis=0) > 0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "kessler-inputs.bin", "wb") as fh:
        np.asarray([ncol, NZ, len(CASES)], "<i4").tofile(fh)
        dts.astype("<f4").tofile(fh)
        steps.astype("<i4").tofile(fh)
        for f in fields:
            np.ascontiguousarray(arr[f]).astype("<f4").tofile(fh)
        rainnc.astype("<f4").tofile(fh)
    np.savez_compressed(args.out_dir / "kessler-inputs.npz",
                        regime=np.asarray(names), dt=dts, nsteps=steps,
                        rainnc=rainnc, **arr)
    print(f"{ncol} columns x {NZ} levels, {len(CASES)} cases -> {args.out_dir}")


if __name__ == "__main__":
    main()
