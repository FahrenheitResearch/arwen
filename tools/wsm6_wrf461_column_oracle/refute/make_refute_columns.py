"""Adversarial WSM6 column sets the lane's oracle did not use.

Same binary layout as tools/wsm6_wrf461_column_oracle/make_columns.py, so
the same run_wsm6_columns (WRF v4.6.1, gfortran -O0) reads it.

usage: python make_refute_columns.py OUT_DIR

Sets (each its own directory under OUT):
  fuzz45   nz=45: 24 targeted regime columns + 360 random columns (all
           species, random dz incl. 1-5 m layers, random inversions/warm
           noses, random thresholds, small negatives) -- NO subnormal inputs
  deep80   nz=80: real tier-80 launch (no forcing), 96 random columns
  nz64     nz=64: last level of tier 64, 64 random columns
  nz65     nz=65: first level of tier 80, 64 random columns
  nz2      nz=2 : minimum bound, 64 random columns
  nz3      nz=3 : 64 random columns
  sub45    nz=45: 64 random columns with subnormal and negative inputs
           (strict build only claim)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make_columns.py
import make_columns as mc  # noqa: E402

F32 = np.float32
SPECIES = ("qc", "qr", "qi", "qs", "qg")
up = lambda x: float(np.nextafter(F32(x), F32(np.inf)))
dn = lambda x: float(np.nextafter(F32(x), F32(-np.inf)))


def geo_dz(nz, dz0, growth, dzmax):
    dz = np.empty(nz)
    d = dz0
    for k in range(nz):
        dz[k] = d
        d = min(d * growth, dzmax)
    return dz


def rand_column(rng, nz, name, *, subnormal=False):
    kind = rng.integers(0, 4)
    if kind == 0:
        dz = geo_dz(nz, rng.uniform(1.0, 80.0), rng.uniform(1.0, 1.15), rng.uniform(200, 2000))
    elif kind == 1:
        dz = rng.uniform(20.0, 1200.0, nz)
    elif kind == 2:  # thin layers somewhere
        dz = geo_dz(nz, rng.uniform(20.0, 60.0), 1.08, 700.0)
        k0 = rng.integers(0, nz)
        k1 = min(nz, k0 + rng.integers(1, max(2, nz // 3)))
        dz[k0:k1] = rng.uniform(1.0, 5.0, k1 - k0)
    else:
        dz = np.full(nz, rng.uniform(100.0, 22000.0 / nz))
    ts = rng.uniform(225.0, 318.0)
    ps = rng.uniform(50000.0, 105000.0)
    lapse = rng.uniform(0.0, 9.8e-3)
    ztrop = rng.uniform(8000.0, 17000.0)
    inv = dict(inv_depth=rng.uniform(100, 1500), inv_dt=rng.uniform(-3, 12)) if rng.random() < 0.3 else {}
    nose = (rng.uniform(300, 1500), rng.uniform(1600, 4000), rng.uniform(1, 12)) if rng.random() < 0.2 else None
    rh0, rh1, zs = rng.uniform(0.0, 1.15), rng.uniform(0.0, 1.15), rng.uniform(1000, 12000)
    rh = lambda z: rh1 + (rh0 - rh1) * np.exp(-z / zs)
    c = mc.Column(name, "refute: random", ps=ps, ts=ts, lapse=lapse, ztrop=ztrop,
                  warm_nose=nose, rh=rh, dz=dz, seed=int(rng.integers(1 << 30)), **inv)
    ztop = c.z[-1] + 1.0
    for f in SPECIES:
        if rng.random() < 0.75:
            for _ in range(rng.integers(1, 4)):
                z1 = rng.uniform(-0.2, 0.9) * ztop
                z2 = z1 + rng.uniform(0.05, 0.8) * ztop
                c.add(f, z1, z2, 10.0 ** rng.uniform(-9.0, -1.6))
        if rng.random() < 0.2:
            ks = rng.choice(nz, size=max(1, nz // 5), replace=True)
            for k in ks:
                c.set_level(f, int(k), 10.0 ** rng.uniform(-16.0, -2.0))
        if rng.random() < 0.1:   # exact thresholds and neighbours
            for k in rng.choice(nz, size=min(nz, 3), replace=False):
                v = [1.0e-9, 1.0e-15, dn(1.0e-9), up(1.0e-9), dn(1.0e-15), up(1.0e-15)][rng.integers(6)]
                c.set_level(f, int(k), v)
        if rng.random() < 0.06:  # small negatives (normal range)
            for k in rng.choice(nz, size=min(nz, 2), replace=False):
                c.set_level(f, int(k), -(10.0 ** rng.uniform(-14.0, -6.0)))
        if rng.random() < 0.05:
            c.set_level(f, int(rng.integers(nz)), -0.0)
    if rng.random() < 0.05:
        c.set_level("qv", int(rng.integers(nz)), 0.0)
    if subnormal:
        for _ in range(rng.integers(2, 8)):
            f = ("qc", "qr", "qi", "qs", "qg", "qv")[rng.integers(6)]
            v = float(F32(10.0 ** rng.uniform(-44.8, -38.0)))
            c.set_level(f, int(rng.integers(nz)), v if rng.random() < 0.7 else -v)
    c.rain = float(rng.uniform(0, 2000)) if rng.random() < 0.5 else 0.0
    c.snow = float(rng.uniform(0, 300)) if rng.random() < 0.5 else 0.0
    c.graupel = float(rng.uniform(0, 300)) if rng.random() < 0.5 else 0.0
    return c


def targeted(nz=45):
    cols = []
    base_dz = geo_dz(nz, 40.0, 1.08, 700.0)

    def C(name, regime, **kw):
        kw.setdefault("dz", base_dz)
        return mc.Column(name, "refute: " + regime, **kw)

    c = C("t-tropical-312", "tropical, 312 K surface, saturated", ts=312.0, ztrop=16500.0,
          rh=lambda z: np.full_like(z, 1.0), seed=301)
    c.add("qc", 500, 12000, 3e-3).add("qr", 0, 8000, 1e-2).add("qg", 4000, 13000, 8e-3)
    c.add("qs", 7000, 15000, 3e-3).add("qi", 11000, 17000, 1e-3)
    cols.append(c)
    c = C("t-warm-ice", "cloud ice and snow below the freezing level (instant melt)", ts=300.0,
          rh=lambda z: np.full_like(z, 0.9), seed=302)
    c.add("qi", 0, 4000, 2e-3).add("qs", 0, 4000, 3e-3).add("qg", 0, 4000, 3e-3)
    cols.append(c)
    c = C("t-homog", "cloud water straddling -40 C", ts=262.0, lapse=6.5e-3,
          rh=lambda z: np.full_like(z, 1.0), seed=303)
    c.add("qc", 0, 9000, 5e-4)
    temps = [233.15, dn(233.15), up(233.15), 233.0, 233.3, 232.0, 234.0, dn(dn(233.15))]
    for i, tt in enumerate(temps):
        c.t[3 + i] = tt
    cols.append(c)
    hi = geo_dz(nz, 200.0, 1.06, 2500.0)
    c = C("t-high-top", "model top near 1 hPa, ice and snow aloft", ts=290.0, dz=hi,
          rh=lambda z: np.where(z < 10000, 0.8, 1.2), seed=304)
    c.add("qi", 20000, 45000, 5e-5).add("qs", 15000, 45000, 1e-4).add("qg", 18000, 45000, 5e-5)
    cols.append(c)
    c = C("t-massive", "unphysically heavy hydrometeors near the ground", ts=303.0,
          rh=lambda z: np.full_like(z, 0.95), seed=305)
    c.add("qr", 0, 3000, 5e-2).add("qg", 0, 3000, 5e-2).add("qs", 2000, 6000, 3e-2)
    c.add("qc", 0, 6000, 1e-2).add("qi", 6000, 12000, 1e-2)
    cols.append(c)
    c = C("t-isothermal-t0c", "isothermal exactly 273.15 K with all species", ts=273.15,
          lapse=0.0, rh=lambda z: np.full_like(z, 1.0), seed=306)
    c.t[:] = 273.15
    for f, v in (("qr", 3e-4), ("qs", 3e-4), ("qg", 3e-4), ("qi", 1e-4), ("qc", 2e-4)):
        c.add(f, 0, 9000, v)
    cols.append(c)
    c = C("t-supersat-ice", "strong ice supersaturation at -30 C", ts=255.0,
          rh=lambda z: np.full_like(z, 1.15), seed=307)
    c.add("qi", 2000, 6000, 1e-7)
    cols.append(c)
    c = C("t-cold-strat", "polar stratosphere 160-190 K with cloud ice", ts=240.0, lapse=9.0e-3,
          ztrop=9000.0, rh=lambda z: np.full_like(z, 1.0), seed=308)
    c.t[-15:] = np.linspace(190.0, 160.0, 15)
    c.add("qi", 8000, 30000, 1e-5).add("qs", 5000, 30000, 5e-5)
    cols.append(c)
    c = C("t-warm-rain-only", "warm rain, no ice anywhere", ts=300.0, lapse=4.0e-3,
          rh=lambda z: np.full_like(z, 0.99), seed=309)
    c.t[:] = np.maximum(c.t, 276.0)
    c.add("qc", 300, 5000, 2e-3).add("qr", 0, 5000, 3e-3)
    cols.append(c)
    c = C("t-qv-zero", "zero vapour with condensate", ts=280.0,
          rh=lambda z: np.zeros_like(z), seed=310)
    c.add("qr", 0, 6000, 1e-3).add("qs", 3000, 8000, 1e-3).add("qc", 0, 6000, 1e-4)
    c.qv[:] = 0.0
    cols.append(c)
    c = C("t-qv-negative", "negative vapour levels with condensate", ts=285.0, seed=311)
    c.add("qr", 0, 6000, 1e-3).add("qc", 0, 6000, 5e-4)
    for k in range(0, nz, 5):
        c.set_level("qv", k, -1.0e-5)
    cols.append(c)
    c = C("t-onelevel-1m", "1 m layers throughout lower half, heavy graupel", ts=298.0,
          dz=np.concatenate([np.full(nz // 2, 1.0), geo_dz(nz - nz // 2, 100.0, 1.1, 800.0)]),
          rh=lambda z: np.full_like(z, 0.9), seed=312)
    c.add("qg", 0, 3000, 2e-2).add("qr", 0, 3000, 1e-2).add("qs", 0, 3000, 5e-3)
    c.qg[-5:] = 1e-2
    c.qr[-5:] = 1e-2
    cols.append(c)
    c = C("t-ri-pressure", "very low surface pressure 40 kPa", ps=40000.0, ts=255.0,
          rh=lambda z: np.full_like(z, 1.0), seed=313)
    c.add("qs", 0, 6000, 1e-3).add("qi", 0, 8000, 3e-4).add("qg", 0, 4000, 5e-4)
    cols.append(c)
    c = C("t-high-p", "very high surface pressure 108 kPa, fog", ps=108000.0, ts=268.0,
          inv_depth=400.0, inv_dt=8.0, rh=lambda z: np.where(z < 400, 1.02, 0.4), seed=314)
    c.add("qc", 0, 400, 5e-4).add("qi", 0, 400, 5e-5)
    cols.append(c)
    # thresholds of other branches, scanned level by level
    c = C("t-thresh-scan", "1e-9/1e-15 neighbours in every species at -10 C and +5 C",
          ts=278.0, rh=lambda z: np.full_like(z, 0.97), seed=315)
    vals = [dn(1e-9), 1e-9, up(1e-9), dn(1e-15), 1e-15, up(1e-15), 1e-12, 1e-6]
    k = 0
    for f in SPECIES:
        for v in vals:
            if k < nz:
                c.set_level(f, k, v)
            k += 1
    cols.append(c)
    c = C("t-big-accum", "huge accumulated surface totals", ts=300.0,
          rh=lambda z: np.full_like(z, 0.95), seed=316)
    c.add("qr", 0, 4000, 5e-3).add("qs", 3000, 9000, 2e-3).add("qg", 2000, 8000, 3e-3)
    c.rain, c.snow, c.graupel = 98765.4, 54321.0, 12345.6
    cols.append(c)
    c = C("t-only-graupel-top", "graupel only in the top level, cold column", ts=265.0,
          rh=lambda z: np.full_like(z, 0.6), seed=317)
    c.qg[-1] = 2e-2
    cols.append(c)
    c = C("t-psaut-edge", "cloud ice exactly near qi0-type limits", ts=262.0,
          rh=lambda z: np.full_like(z, 1.0), seed=318)
    for i, v in enumerate([1e-3, dn(1e-3), up(1e-3), 2e-4, 8e-5, 5e-4, 1e-4, 3e-3]):
        c.set_level("qi", 10 + i, v)
    cols.append(c)
    return cols


def write(out: Path, cols, nz):
    data = [c.fields() for c in cols]
    for d in data:
        for v in d.values():
            assert v.shape == (nz,), v.shape
    names = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "den", "pii", "p", "dz")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "columns.bin", "wb") as fh:
        np.asarray([len(cols), nz], "<i4").tofile(fh)
        for n in names:
            a = np.stack([d[n] for d in data], axis=1).astype("<f4")
            assert np.all(np.isfinite(a)), n
            a.tofile(fh)
        for n in ("rain", "snow", "graupel"):
            np.asarray([getattr(c, n) for c in cols], "<f4").tofile(fh)
    meta = {"ncol": len(cols), "nz": nz,
            "columns": [{"name": c.name, "regime": c.regime} for c in cols]}
    (out / "columns.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(out.name, len(cols), "x", nz)


def main():
    out = Path(sys.argv[1])
    rng = np.random.default_rng(20261007)
    cols = targeted(45) + [rand_column(rng, 45, f"r45-{i}") for i in range(360)]
    write(out / "fuzz45", cols, 45)
    for nz, n, tag in ((80, 96, "deep80"), (64, 64, "nz64"), (65, 64, "nz65"),
                       (2, 64, "nz2"), (3, 64, "nz3")):
        write(out / tag, [rand_column(rng, nz, f"r{nz}-{i}") for i in range(n)], nz)
    write(out / "sub45", [rand_column(rng, 45, f"s45-{i}", subnormal=True) for i in range(64)], 45)


if __name__ == "__main__":
    main()
