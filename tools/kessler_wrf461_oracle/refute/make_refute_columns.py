#!/usr/bin/env python3
"""NEW Kessler column-oracle inputs the kessler-mp1 lane did not use.

Writes the same stream format as tools/kessler_wrf461_oracle/make_columns.py
(read by run_kessler.F90 and gpuwm.verify.kessler_oracle.read_inputs_bin):

    int32 ncol, nz, ncase; float32 dt[ncase]; int32 nsteps[ncase]
    float32 t, qv, qc, qr, rho, pii, z, dz8w  each (nz, ncol) C order
    float32 rainnc[ncol]

    make_refute_columns.py OUT_DIR

writes four sets, each its own input file:
  setA  nz=50,  13 new regimes x 8 columns, 6 dt cases (87 calls)
  setB  nz=256 (the kernel's KESS_KMAX), LES-like and deep grids
  setC  nz=2 and setD nz=3 (smallest columns the routine accepts)
  setE  nz=120, random mixture, 40 free-running calls
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RD, RV, CP, G = 287.0, 461.6, 1004.5, 9.81
EP2 = RD / RV
F32 = np.float32


def qvs_of(T, p):
    es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    return EP2 * es / np.maximum(p - es, 1e-3 * p)


def grid(nz, zs, ztop, b):
    s = np.linspace(0.0, 1.0, nz + 1)
    zf = zs + ztop * np.expm1(b * s) / np.expm1(b)
    return 0.5 * (zf[:-1] + zf[1:]), np.diff(zf)


def hydro(z, zs, ps, T):
    p = np.empty_like(z)
    zp, pp, tp = zs, ps, T[0]
    for k in range(z.size):
        tm = 0.5 * (tp + T[k])
        p[k] = pp * np.exp(-G * (z[k] - zp) / (RD * tm))
        zp, pp, tp = z[k], p[k], T[k]
    return p


def strat_T(agl, tsfc, lapse, ttrop, ztrop, tstrat_top, ztop):
    """Troposphere lapse, isothermal tropopause, warming stratosphere."""
    T = np.maximum(tsfc - lapse * agl, ttrop)
    warm = np.clip((agl - (ztrop + 8000.0)) / max(ztop - ztrop - 8000.0, 1.0),
                   0, 1)
    return np.where(agl > ztrop + 8000.0,
                    ttrop + (tstrat_top - ttrop) * warm, T)


def bump(h, z0, z1):
    x = (h - z0) / max(z1 - z0, 1.0)
    return np.where((x > 0) & (x < 1), np.sin(np.pi * np.clip(x, 0, 1)) ** 2,
                    0.0)


def column(nz, zs, ps, T_of_agl, rh, qc, qr, *, ztop=20000.0, b=2.5,
           rainnc=0.0):
    z, dz = grid(nz, zs, ztop, b)
    agl = z - zs
    T = T_of_agl(agl)
    p = hydro(z, zs, ps, T)
    pii = (p / 1.0e5) ** (RD / CP)
    qv = rh(agl) * qvs_of(T, p)
    rho = p / (RD * T * (1.0 + qv / EP2))
    return dict(t=T / pii, qv=qv, qc=qc(agl), qr=qr(agl), rho=rho, pii=pii,
                z=z, dz8w=dz, rainnc=rainnc)


def to32(c):
    out = {k: (np.asarray(v, np.float64).astype(F32) if k != "rainnc"
               else F32(v)) for k, v in c.items()}
    return out


def set_a(rng, nz=50, per=8):
    u = rng.uniform
    cols, names = [], []

    def add(name, c):
        cols.append(to32(c))
        names.append(name)

    # 1. high model top (~1 hPa): warm stratopause, es > p aloft, so WRF's
    #    qvs = ep2*es/(p-es) goes NEGATIVE in the top layers.
    for _ in range(per):
        ztop = u(46000, 50000)
        tst = u(262, 275)
        add("high_top_1hPa", column(
            nz, u(0, 300), 100000.0,
            lambda h, a=u(288, 300), t=tst, zt=ztop: strat_T(
                h, a, 0.0065, 212.0, 12000.0, t, zt),
            lambda h: np.where(h < 9000, 0.9, np.where(h < 15000, 0.3, 0.05)),
            lambda h, a=u(1e-4, 2e-3): a * bump(h, 1500, 8000),
            lambda h, r=u(1e-4, 6e-3): r * bump(h, -3000, 6000),
            ztop=ztop, b=1.5, rainnc=u(0, 20)))
    # 2. same top, with condensate and rain ABOVE the es > p level and dry
    #    vapour there (qv = 0 against a negative qvs).
    for n in range(per):
        ztop = u(46000, 50000)
        c = column(
            nz, u(0, 300), 100000.0,
            lambda h, a=u(285, 295), t=u(265, 275), zt=ztop: strat_T(
                h, a, 0.0065, 215.0, 11000.0, t, zt),
            lambda h: np.where(h < 9000, 0.95, 0.2),
            lambda h, a=u(1e-6, 1e-4): a * (h > 35000),
            lambda h, r=u(1e-6, 1e-3): r * (h > 38000),
            ztop=ztop, b=1.5)
        if n % 2 == 0:
            c["qv"][-8:] = 0.0
        if n % 4 == 1:
            c["qv"][-8:] = -1e-7
        add("high_top_cond_aloft", c)
    # 3. Exner above one: surface pressure 1050-1085 hPa, cold, fog/drizzle
    for _ in range(per):
        add("pii_above_one", column(
            nz, u(-400, 0), u(105000, 108500),
            lambda h, a=u(225, 262): np.where(
                h < 300, a + 8 * h / 300, a + 8 - 0.006 * (h - 300)),
            lambda h, s=u(1.0, 1.01): np.where(h < 200, s, 0.6),
            lambda h, a=u(5e-5, 5e-4): a * (h < 250),
            lambda h, r=u(1e-6, 1e-4): r * (h < 400)))
    # 4. hot house: 315-330 K surface, very moist, heavy rain
    for _ in range(per):
        add("hothouse", column(
            nz, 0.0, 101000.0,
            lambda h, a=u(315, 330): np.maximum(a - 0.0058 * h, 200.0),
            lambda h, s=u(0.95, 1.03): np.where(h < 12000, s, 0.4),
            lambda h, a=u(1e-3, 6e-3): a * bump(h, 500, 12000),
            lambda h, r=u(5e-3, 30e-3): r * bump(h, -6000, 8000),
            rainnc=u(0, 500)))
    # 5. cold plateau: 185-205 K surface at 2.8-3.5 km
    for _ in range(per):
        zs = u(2800, 3500)
        add("cold_plateau", column(
            nz, zs, 101325.0 * (1 - 2.25577e-5 * zs) ** 5.25588,
            lambda h, a=u(185, 205): np.where(
                h < 400, a + 25 * h / 400, np.maximum(a + 25 - 0.006 * (h - 400),
                                                      180.0)),
            lambda h, s=u(1.0, 1.05): np.where(h < 2000, s, 0.5),
            lambda h, a=u(1e-6, 5e-5): a * bump(h, 0, 2000),
            lambda h, r=u(1e-7, 5e-5): r * bump(h, -2000, 2000),
            ztop=16000.0))
    # 6. large negative undershoots in qv, qc and qr, random levels
    for _ in range(per):
        c = column(nz, u(0, 1000), 100000.0,
                   lambda h, a=u(280, 300): np.maximum(a - 0.0065 * h, 210.0),
                   lambda h: np.where(h < 6000, 1.0, 0.5),
                   lambda h: 5e-4 * bump(h, 1000, 6000),
                   lambda h: 3e-3 * bump(h, -3000, 5000))
        for f, lo, hi in (("qv", 1e-8, 1e-4), ("qc", 1e-8, 1e-3),
                          ("qr", 1e-8, 1e-3)):
            m = rng.random(nz) < 0.3
            c[f][m] = -np.exp(u(np.log(lo), np.log(hi), m.sum()))
        add("neg_undershoot", c)
    # 7. signed zeros everywhere: -0.0/+0.0 in qv, qc and qr at random levels,
    #    two columns entirely +-0 in all three
    for n in range(per):
        c = column(nz, u(0, 500), 100000.0,
                   lambda h, a=u(275, 300): np.maximum(a - 0.0065 * h, 210.0),
                   lambda h: np.where(h < 5000, 1.002, 0.4),
                   lambda h: 2e-4 * bump(h, 500, 4000),
                   lambda h: 1e-3 * bump(h, -2000, 3000))
        c = to32(c)
        for f in ("qv", "qc", "qr"):
            m = rng.random(nz) < (1.0 if n >= per - 2 else 0.35)
            c[f][m] = np.where(rng.random(m.sum()) < 0.5, F32(-0.0), F32(0.0))
        cols.append(c)
        names.append("signed_zero_all")
    # 8. full float32 range: qc, qr log-uniform 1e-45..1e-1 (subnormals
    #    included), qv log-uniform 1e-12..3e-2, 10% negative
    for _ in range(per):
        c = to32(column(nz, u(0, 1000), 100000.0,
                        lambda h, a=u(270, 305): np.maximum(a - 0.0065 * h, 205.0),
                        lambda h: 0.9 + 0 * h, lambda h: 0 * h, lambda h: 0 * h))
        for f, lo, hi in (("qc", -103.0, -2.3), ("qr", -103.0, -2.3),
                          ("qv", -27.6, -3.5)):
            v = np.exp(u(lo, hi, nz))
            v[rng.random(nz) < 0.1] *= -1
            c[f] = v.astype(F32)
        cols.append(c)
        names.append("logrand_fullrange")
    # 9. saturation knife edge: qv on Tetens' qvs to 1e-6 relative, qc 0 or
    #    tiny, so prod sits at +-tiny and the max/min ties are exercised
    for n in range(per):
        c = column(nz, u(0, 500), 100000.0,
                   lambda h, a=u(280, 300): np.maximum(a - 0.0065 * h, 210.0),
                   lambda h: 1.0 + 1e-6 * (2 * rng.random(h.size) - 1),
                   lambda h: (0.0 if n % 2 else 1e-9) + 0 * h,
                   lambda h: (0.0 if n % 3 else 1e-7) + 0 * h)
        add("saturation_edge", c)
    # 10. thin surface layers (1-5 m) under heavy rain: very large nfall
    for _ in range(per):
        c = column(nz, u(0, 200), 100000.0,
                   lambda h, a=u(295, 303): np.maximum(a - 0.0065 * h, 205.0),
                   lambda h: np.where(h < 9000, 1.01, 0.4),
                   lambda h: 2e-3 * bump(h, 1000, 9000),
                   lambda h, r=u(8e-3, 20e-3): r * bump(h, -4000, 4000),
                   b=6.0)
        c["dz8w"][:6] = u(1.0, 5.0, 6)
        add("thin_surface_layers", c)
    # 11. density spikes and inversion of the density profile
    for _ in range(per):
        c = column(nz, u(0, 500), 100000.0,
                   lambda h, a=u(285, 300): np.maximum(a - 0.0065 * h, 210.0),
                   lambda h: np.where(h < 6000, 1.0, 0.5),
                   lambda h: 8e-4 * bump(h, 1000, 6000),
                   lambda h: 6e-3 * bump(h, -3000, 5000))
        c["rho"][0] *= 0.5
        idx = rng.integers(1, nz, 5)
        c["rho"][idx] *= u(0.2, 3.0, 5)
        add("density_spikes", c)
    # 12. large accumulated RAINNC (1e3 to 1e5 mm) under heavy rain
    for _ in range(per):
        add("rainnc_large", column(
            nz, u(0, 200), 100000.0,
            lambda h, a=u(298, 304): np.maximum(a - 0.0065 * h, 205.0),
            lambda h: np.where(h < 9000, 1.01, 0.4),
            lambda h: 2e-3 * bump(h, 1000, 9000),
            lambda h, r=u(5e-3, 20e-3): r * bump(h, -4000, 4000),
            rainnc=np.exp(u(np.log(1e3), np.log(1e5)))))
    # 13. qr/qc on exact powers of two and their float neighbours,
    #     2^-149 .. 2^-3, one word per level
    for n in range(per):
        c = to32(column(nz, u(0, 500), 100000.0,
                        lambda h, a=u(280, 300): np.maximum(a - 0.0065 * h, 210.0),
                        lambda h: np.where(h < 6000, 1.0, 0.5),
                        lambda h: 0 * h, lambda h: 0 * h))
        e = np.linspace(-149, -3, nz).round().astype(int)
        base = np.ldexp(1.0, e).astype(F32)
        step = (n % 3) - 1
        words = np.array([np.nextafter(x, F32(np.inf * step)) if step else x
                          for x in base], F32)
        c["qr"] = np.roll(words, n)
        c["qc"] = np.roll(words, 3 * n + 7)
        cols.append(c)
        names.append("pow2_ladder")
    cases = ((0.5, 6), (7.2, 10), (100.0 / 3.0, 6), (600.0, 3), (900.0, 2),
             (10.0, 60))
    return cols, names, cases


def set_b(rng, nz=256, ncol=32):
    u = rng.uniform
    cols, names = [], []
    for n in range(ncol):
        les = n < ncol // 2
        ztop = u(2500, 4000) if les else u(25000, 32000)
        c = column(nz, u(0, 1500), 100000.0,
                   lambda h, a=u(280, 305): np.maximum(a - 0.0065 * h, 205.0),
                   lambda h, s=u(0.98, 1.02): np.where(h < 2000, s, 0.6),
                   lambda h, a=u(0, 3e-3): a * bump(h, u(0, 800), u(1200, 2500)),
                   lambda h, r=u(0, 15e-3): r * bump(h, -2000, u(500, 2500)),
                   ztop=ztop, b=u(1.0, 3.0), rainnc=u(0, 100))
        cols.append(to32(c))
        names.append("les_256" if les else "deep_256")
    return cols, names, ((0.25, 8), (2.0, 10), (20.0, 5))


def set_small(rng, nz, ncol=32):
    u = rng.uniform
    cols, names = [], []
    for n in range(ncol):
        c = column(nz, u(0, 1000), 100000.0,
                   lambda h, a=u(270, 305): np.maximum(a - 0.0065 * h, 205.0),
                   lambda h, s=u(0.5, 1.05): s + 0 * h,
                   lambda h, a=u(0, 3e-3): a + 0 * h,
                   lambda h, r=u(0, 2e-2): r * (h >= 0),
                   ztop=u(200, 3000), b=1.0, rainnc=u(0, 50))
        cols.append(to32(c))
        names.append(f"nz{nz}")
    return cols, names, ((60.0, 3), (300.0, 3), (5.0, 6))


def set_e(rng, nz=120, ncol=64):
    u = rng.uniform
    cols, names = [], []
    for n in range(ncol):
        c = column(nz, u(0, 3000), 100000.0,
                   lambda h, a=u(260, 310), l=u(0.004, 0.009), tt=u(195, 220):
                   np.maximum(a - l * h, tt),
                   lambda h, a=u(0.3, 1.05), bb=u(0.3, 1.05), z1=u(1000, 8000):
                   np.where(h < z1, a, bb),
                   lambda h, a=u(0, 3e-3): a * bump(h, u(0, 3000), u(4000, 12000)),
                   lambda h, r=u(0, 15e-3): r * bump(h, u(-5000, 0), u(2000, 9000)),
                   ztop=u(15000, 30000), b=u(1.5, 4.5), rainnc=u(0, 300))
        cols.append(to32(c))
        names.append("mix_120")
    return cols, names, ((18.0, 40),)


#: qr words where this box's C-library powf(x, 0.875) (FMA-dispatched on
#: an FMA-capable CPU) and the device's gfk_pow disagree by 1 ULP (found by
#: libm_sweep.py over all 2**32 inputs), and float32 dt values for which that
#: 1 ULP survives into factorn = 1/(1 + c3*dt*qr**c4).
WITNESS_QR = (0x34740e00, 0x30740e00)
WITNESS_DT = {0x34740e00: ("135.5", "223.25", "582.0"),
              0x30740e00: ("20.290653",)}


def set_f(rng, nz=50):
    """Physically plausible high-top columns (stratospheric vapour of a few
    ppmv) and the libm-variant witness columns."""
    u = rng.uniform
    cols, names = [], []
    for n in range(16):
        ztop = u(40000, 50000)
        c = column(
            nz, u(0, 300), 100000.0,
            lambda h, a=u(288, 300), t=u(258, 275), zt=ztop: strat_T(
                h, a, 0.0065, 212.0, 12000.0, t, zt),
            lambda h: np.where(h < 9000, 0.85, 0.3),
            lambda h, a=u(1e-4, 1e-3): a * bump(h, 1500, 8000),
            lambda h, r=u(1e-4, 4e-3): r * bump(h, -3000, 6000),
            ztop=ztop, b=1.5, rainnc=u(0, 20))
        strat = (c["z"] - c["z"][0]) > 12000.0
        c["qv"] = np.where(strat, u(2e-6, 6e-6), np.minimum(c["qv"], 0.03))
        if n % 4 == 3:      # a thin cloud of supercooled drops at the top
            c["qc"][-3:] = 1e-6
        cols.append(to32(c))
        names.append("high_top_real")
    for n in range(8):
        c = to32(column(
            nz, u(0, 500), 100000.0,
            lambda h, a=u(280, 295): np.maximum(a - 0.0065 * h, 210.0),
            lambda h: np.where(h < 6000, 1.0, 0.5),
            lambda h, a=u(2e-4, 1e-3): a * (h < 6000),
            lambda h: 0 * h))
        word = WITNESS_QR[n % 2]
        lev = np.arange(2, 40, 2)
        c["qr"][lev] = np.uint32(word).view(F32)
        cols.append(c)
        names.append(f"witness_{word:08x}")
    dts = sorted({F32(v) for vals in WITNESS_DT.values() for v in vals})
    return cols, names, tuple((float(d), 1) for d in dts) + ((60.0, 4),)


FIELDS = ("t", "qv", "qc", "qr", "rho", "pii", "z", "dz8w")


def write(out_dir: Path, tag, cols, names, cases):
    ncol = len(cols)
    nz = len(cols[0]["t"])
    arr = {f: np.stack([np.asarray(c[f], F32) for c in cols], axis=1)
           for f in FIELDS}
    rainnc = np.asarray([c["rainnc"] for c in cols], F32)
    dts = np.asarray([d for d, _ in cases], F32)
    steps = np.asarray([s for _, s in cases], np.int32)
    for f in FIELDS:
        assert np.all(np.isfinite(arr[f])), (tag, f)
    assert np.all(arr["dz8w"] > 0) and np.all(np.diff(arr["z"], axis=0) > 0)
    d = out_dir / tag
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "kessler-inputs.bin", "wb") as fh:
        np.asarray([ncol, nz, len(cases)], "<i4").tofile(fh)
        dts.astype("<f4").tofile(fh)
        steps.astype("<i4").tofile(fh)
        for f in FIELDS:
            np.ascontiguousarray(arr[f]).astype("<f4").tofile(fh)
        rainnc.astype("<f4").tofile(fh)
    np.savez_compressed(d / "kessler-inputs.npz", regime=np.asarray(names),
                        dt=dts, nsteps=steps, rainnc=rainnc, **arr)
    print(f"{tag}: {ncol} columns x {nz} levels, {len(cases)} cases "
          f"({int(steps.sum())} calls)")


def main():
    out = Path(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "setF":
        write(out, "setF", *set_f(np.random.default_rng(777)))
        return
    rng = np.random.default_rng(424242)
    write(out, "setA", *set_a(rng))
    write(out, "setB", *set_b(rng))
    write(out, "setC", *set_small(rng, 2))
    write(out, "setD", *set_small(rng, 3))
    write(out, "setE", *set_e(rng))


if __name__ == "__main__":
    main()
