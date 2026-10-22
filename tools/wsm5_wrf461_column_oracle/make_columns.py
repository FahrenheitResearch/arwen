"""Deterministic WSM5 column sets for the WRF v4.6.1 column oracle.

Writes little-endian binary files that BOTH sides read, so the WRF Fortran
(run_wsm5_columns.F90) and the CUDA kernel see the same float32 words.
Nothing here is an expected value: it is the input.

Layout of each ``<set>.bin``::

    int32 ncol, int32 nz
    10 level fields, each nz*ncol float32, k-major with the column index
       fastest (Fortran ``a(i,k)`` order):
       th, qv, qc, qr, qi, qs, den, pii, p, dz
    2 surface fields, each ncol float32: rain, snow (accumulated)

``<set>.json`` names every column and its family.

Two sets:

``main`` (45 levels, 66 columns) spans the families WSM5 meets in a
forecast -- warm rain, mixed phase through a melting layer, deep
convection, deep cold (all-snow winter columns, cirrus, supercooled cloud
below -40 C), saturated (fog, stratocumulus, supersaturated clear air),
dry (stable clear nights, subsaturated cloud that evaporates),
sedimentation-heavy (heavy condensate aloft, thin layers that force the
semi-Lagrangian CFL limiter, condensate in the top level), high terrain --
plus edge columns that sit exactly on the scheme's branch thresholds
(qcrmin, qc0, qmin, t0c, the triple point), carry negative, negative-zero
or subnormal hydrometeors, or hold no condensate at all.

``lev2`` (2 levels, 16 columns): the shallowest grid WSM5 admits, where
the sedimentation remap reaches its last arrival interface on every
column that carries falling condensate.

Usage::

    python make_columns.py OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.wsm6_wrf461_column_oracle.common import (
    _interfaces, _qsat, _bump, profile, driver_fields)

F32 = np.float32
NZ = 45
RD, CP, P0 = 287.0, 1004.5, 1.0e5
LEVEL_FIELDS = ("th", "qv", "qc", "qr", "qi", "qs", "den", "pii", "p", "dz")
SURFACE_FIELDS = ("rain", "snow")


class Column:
    def __init__(self, name: str, family: str, *, ps=100000.0, ts=295.0,
                 lapse=6.5e-3, ztrop=12000.0, inv_depth=0.0, inv_dt=0.0,
                 warm_nose=None, rh=None, dz=None, seed=0):
        self.name, self.family = name, family
        self.dz = _interfaces() if dz is None else np.asarray(dz, float)
        zi, self.z, self.t, self.p, self.qv = profile(
            self.dz, ps=ps, ts=ts, lapse=lapse, ztrop=ztrop,
            inv_depth=inv_depth, inv_dt=inv_dt, warm_nose=warm_nose,
            rh=rh, seed=seed)
        z = self.z
        zero = np.zeros_like(z)
        self.qc, self.qr, self.qi, self.qs = (zero.copy() for _ in range(4))
        self.rain = 0.0
        self.snow = 0.0
        self.overrides = []

    def add(self, field: str, z1: float, z2: float, peak: float):
        setattr(self, field, getattr(self, field) + _bump(self.z, z1, z2, peak))
        return self

    def set_level(self, field: str, k: int, value):
        self.overrides.append((field, k, value))
        return self

    def fields(self) -> dict[str, np.ndarray]:
        out = {"t": self.t.astype(F32), "p": self.p.astype(F32),
               "qv": self.qv.astype(F32),
               "qc": self.qc.astype(F32), "qr": self.qr.astype(F32),
               "qi": self.qi.astype(F32), "qs": self.qs.astype(F32),
               "dz": self.dz.astype(F32)}
        for field, k, value in self.overrides:
            out[field][k] = F32(value)
        # WRF driver quantities, formed in float32.  They are inputs: both
        # sides read these bits.
        return driver_fields(out)


def _main_columns() -> list[Column]:
    cols: list[Column] = []
    up = lambda x: float(np.nextafter(F32(x), F32(np.inf)))
    dn = lambda x: float(np.nextafter(F32(x), F32(-np.inf)))

    # -- warm rain: liquid-only columns, rain falling into drier air ----------
    for n, (ts, qcp, qrp, rhb) in enumerate([
            (300.0, 1.2e-3, 2.0e-3, 0.95), (303.0, 2.0e-3, 4.0e-3, 0.85),
            (297.0, 6.0e-4, 8.0e-4, 0.70), (305.0, 2.5e-3, 6.0e-3, 0.98),
            (299.0, 9.0e-4, 1.5e-3, 0.60), (301.0, 1.6e-3, 3.0e-4, 0.90),
            (296.0, 4.0e-4, 5.0e-3, 0.50), (302.0, 3.0e-3, 1.0e-4, 1.00)]):
        c = Column(f"warm-rain-{n}", "warm rain", ts=ts, lapse=5.5e-3,
                   rh=lambda z, rhb=rhb: np.where(z < 1200, rhb,
                                                  np.where(z < 4500, 0.99, 0.4)),
                   seed=10 + n)
        c.add("qc", 1200.0, 4500.0, qcp).add("qr", 0.0, 4200.0, qrp)
        c.rain = 2.0 * n
        cols.append(c)

    # -- mixed phase through a melting layer (snow melts, then sublimates in
    #    subsaturated air below the freezing level: the psevp process) -------
    for n, (ts, frz, rhlow) in enumerate([(285.0, 3000.0, 0.97), (281.0, 2000.0, 0.60),
                                          (289.0, 4000.0, 0.80), (278.0, 1200.0, 0.50),
                                          (283.0, 2500.0, 0.90), (287.0, 3500.0, 0.70),
                                          (280.0, 1800.0, 0.40), (284.0, 2800.0, 0.99)]):
        c = Column(f"mixed-phase-{n}", "mixed phase", ts=ts, lapse=6.0e-3,
                   rh=lambda z, frz=frz, rhlow=rhlow: np.where(
                       z < frz - 400.0, rhlow, np.where(z < 8000, 0.97, 0.5)),
                   seed=30 + n)
        c.add("qs", frz - 900.0, 9000.0, 6.0e-4 + 1.0e-4 * n)
        c.add("qr", 0.0, frz + 200.0, 4.0e-4 + 1.5e-4 * (n % 4))
        c.add("qc", frz - 1500.0, frz + 1200.0, 1.5e-4)
        c.add("qi", 6000.0, 10000.0, 3.0e-5 + 1.0e-5 * n)
        c.rain, c.snow = 1.5 * n, 0.25 * n
        cols.append(c)

    # -- deep convection ---------------------------------------------------------
    for n, (ts, qcp, qrp, qsp, qip) in enumerate([
            (303.0, 2.5e-3, 6.0e-3, 3.0e-3, 4.0e-4), (300.0, 1.8e-3, 3.0e-3, 4.0e-3, 2.0e-4),
            (305.0, 3.0e-3, 9.0e-3, 2.0e-3, 6.0e-4), (298.0, 1.2e-3, 1.5e-3, 5.0e-3, 1.0e-4),
            (301.0, 2.2e-3, 4.5e-3, 6.0e-3, 8.0e-4), (304.0, 2.8e-3, 1.2e-2, 1.0e-3, 3.0e-4)]):
        c = Column(f"convective-{n}", "deep convection", ts=ts, lapse=6.8e-3,
                   ztrop=13500.0, rh=lambda z: np.where(z < 9000, 0.95, 0.6), seed=50 + n)
        c.add("qc", 800.0, 9500.0, qcp).add("qr", 0.0, 7000.0, qrp)
        c.add("qs", 4000.0, 13000.0, qsp).add("qi", 8000.0, 15000.0, qip)
        c.rain, c.snow = 3.5 * n, 0.25 * n
        cols.append(c)

    # -- deep cold: all-snow winter, cirrus, supercooled below -40 C -----------
    for n in range(6):
        c = Column(f"snow-{n}", "deep cold", ts=262.0 + 1.5 * n, lapse=5.5e-3,
                   ztrop=9000.0, rh=lambda z: np.where(z < 5000, 0.98, 0.6), seed=70 + n)
        c.add("qs", 0.0, 6000.0, 2.0e-4 + 1.0e-4 * n)
        c.add("qi", 2000.0, 8000.0, 2.0e-5 + 1.0e-5 * n)
        c.add("qc", 300.0, 2500.0, 5.0e-5 * (n % 3))
        c.snow = 1.0 + n
        cols.append(c)
    for n in range(2):
        c = Column(f"cirrus-{n}", "deep cold", ts=288.0 + n,
                   rh=lambda z: np.where(z > 9000, 1.05, 0.3), seed=80 + n)
        c.add("qi", 9000.0, 13000.0, 5.0e-5 + 2.0e-4 * n)
        cols.append(c)
    for n in range(2):
        c = Column(f"supercooled-{n}", "deep cold", ts=248.0 - 5.0 * n,
                   lapse=7.0e-3, ztrop=9000.0, rh=lambda z: np.full_like(z, 1.0),
                   seed=85 + n)
        c.add("qc", 0.0, 9000.0, 2.0e-4).add("qr", 0.0, 3000.0, 5.0e-5 * n)
        cols.append(c)

    # -- saturated: fog, stratocumulus, supersaturated clear -------------------
    for n in range(3):
        c = Column(f"fog-{n}", "saturated", ts=271.0 + 6.0 * n, inv_depth=250.0,
                   inv_dt=4.0, rh=lambda z: np.where(z < 300, 1.01, 0.5), seed=90 + n)
        c.add("qc", 0.0, 300.0, 1.2e-4 + 5.0e-5 * n)
        if n % 2:
            c.add("qi", 0.0, 300.0, 1.0e-6)
        cols.append(c)
    for n in range(3):
        c = Column(f"ocean-sc-{n}", "saturated", ps=101500.0, ts=290.0 + n,
                   rh=lambda z: np.where(z < 1100, 0.995, 0.2), seed=95 + n)
        c.add("qc", 500.0, 1150.0, 4.0e-4 + 1.0e-4 * n)
        c.add("qr", 0.0, 1100.0, 2.0e-5 * n)
        cols.append(c)
    c = Column("saturated-clear", "saturated", ts=290.0,
               rh=lambda z: np.full_like(z, 1.03), seed=99)
    cols.append(c)
    c = Column("saturated-cold-clear", "saturated", ts=265.0,
               rh=lambda z: np.full_like(z, 1.02), seed=100)
    cols.append(c)

    # -- dry: stable clear nights, cloud evaporating in dry air ---------------
    for n in range(3):
        c = Column(f"stable-night-{n}", "dry", ts=275.0 + 3.0 * n,
                   inv_depth=300.0 + 100.0 * n, inv_dt=6.0 + n,
                   rh=lambda z: np.full_like(z, 0.35), seed=105 + n)
        cols.append(c)
    c = Column("dry-evaporating", "dry", ts=293.0, rh=lambda z: np.full_like(z, 0.4),
               seed=108)
    c.add("qc", 0.0, 9000.0, 1.0e-3).add("qi", 6000.0, 12000.0, 2.0e-4)
    c.add("qs", 3000.0, 12000.0, 4.0e-4).add("qr", 0.0, 6000.0, 6.0e-4)
    cols.append(c)
    c = Column("dry-warm-snow", "dry", ts=288.0, lapse=7.5e-3,
               rh=lambda z: np.full_like(z, 0.3), seed=109)
    c.add("qs", 0.0, 4000.0, 8.0e-4)
    cols.append(c)

    # -- sedimentation-heavy ------------------------------------------------------
    for n in range(3):
        c = Column(f"sediment-aloft-{n}", "sedimentation-heavy", ts=296.0 - 4.0 * n,
                   rh=lambda z: np.full_like(z, 0.9), seed=110 + n)
        c.add("qr", 4000.0 + 1500.0 * n, 15000.0, 6.0e-3)
        c.add("qs", 8000.0, 22000.0, 3.0e-3)
        c.add("qi", 10000.0, 22000.0, 1.0e-3)
        cols.append(c)
    thin = np.full(NZ, 5.0)
    thin[20:] = _interfaces()[20:]
    c = Column("sediment-thin-layers", "sedimentation-heavy", ts=296.0,
               dz=thin, rh=lambda z: np.full_like(z, 0.9), seed=115)
    c.add("qr", 0.0, 3000.0, 8.0e-3).add("qs", 0.0, 3000.0, 4.0e-3)
    cols.append(c)
    c = Column("sediment-top-only", "sedimentation-heavy", ts=280.0,
               rh=lambda z: np.full_like(z, 0.5), seed=116)
    for f, v in (("qr", 1.0e-4), ("qs", 2.0e-4), ("qi", 5.0e-5), ("qc", 1.0e-5)):
        c.set_level(f, NZ - 1, v)
    # negative zeros: max(q, 0.0) keeps the sign WRF's build gives it
    for k, f in enumerate(("qc", "qr", "qi", "qs")):
        c.set_level(f, 3 + k, -0.0)
    cols.append(c)
    c = Column("sediment-top-two", "sedimentation-heavy", ts=284.0,
               rh=lambda z: np.full_like(z, 0.8), seed=117)
    for k in (NZ - 2, NZ - 1):
        for f, v in (("qr", 3.0e-4), ("qs", 6.0e-4), ("qi", 1.0e-4)):
            c.set_level(f, k, v)
    cols.append(c)
    c = Column("sediment-bottom-only", "sedimentation-heavy", ts=276.0,
               rh=lambda z: np.full_like(z, 0.8), seed=118)
    for f, v in (("qr", 2.0e-3), ("qs", 1.0e-3), ("qi", 1.0e-4), ("qc", 5.0e-4)):
        c.set_level(f, 0, v)
    c.rain, c.snow = 123.456, 7.25
    cols.append(c)

    # -- high terrain ------------------------------------------------------------
    for n in range(4):
        c = Column(f"high-terrain-{n}", "high terrain", ps=62000.0 + 3000.0 * n,
                   ts=270.0 + 3.0 * n, lapse=7.0e-3, ztrop=11000.0,
                   rh=lambda z: np.where(z < 4000, 0.9, 0.3), seed=120 + n)
        c.add("qs", 0.0, 4500.0, 3.0e-4).add("qc", 200.0, 2500.0, 2.0e-4)
        c.add("qr", 0.0, 800.0, 1.0e-4 * (n % 3)).add("qi", 2000.0, 6000.0, 2.0e-5)
        cols.append(c)

    # -- edge columns ------------------------------------------------------------
    base = dict(ts=282.0, lapse=6.5e-3, rh=lambda z: np.where(z < 6000, 0.98, 0.5))
    c = Column("edge-thresholds", "edge: branch thresholds", seed=200, **base)
    qcrmin, qmin = 1.0e-9, 1.0e-15
    qc0 = float(F32(F32(4.0 / 3.0) * F32(np.pi) * F32(1000.0) * F32(F32(0.8e-5) ** 3)
                    * F32(3.0e8) / F32(1.28)))
    vals = [("qr", qcrmin), ("qs", qcrmin), ("qc", qc0), ("qi", qmin), ("qc", qmin),
            ("qs", qmin), ("qr", qmin)]
    k = 2
    for field, v in vals:
        for x in (dn(v), v, up(v)):
            c.set_level(field, k, x)
            k += 1
    cols.append(c)

    c = Column("edge-temperatures", "edge: t0c and triple point", seed=201, **base)
    c.add("qs", 0.0, 6000.0, 3.0e-4).add("qr", 0.0, 6000.0, 3.0e-4)
    c.add("qi", 0.0, 6000.0, 1.0e-5).add("qc", 0.0, 6000.0, 2.0e-4)
    t0c = 273.15
    ttp = float(F32(F32(273.15) + F32(0.01)))
    temps = [dn(t0c), t0c, up(t0c), dn(ttp), ttp, up(ttp), 233.15, dn(233.15), 223.15, 200.0]
    for k, tt in enumerate(temps):
        c.t[3 + 2 * k] = tt
    cols.append(c)

    c = Column("edge-negative", "edge: negative and subnormal inputs", seed=202, **base)
    c.add("qr", 0.0, 3000.0, 2.0e-4).add("qs", 2000.0, 6000.0, 2.0e-4)
    for k, (f, v) in enumerate([("qc", -1.0e-10), ("qr", -3.0e-12), ("qi", -1.0e-14),
                                ("qs", -2.0e-11), ("qc", 1.0e-40), ("qr", 3.0e-41),
                                ("qs", 1.0e-39), ("qi", 2.0e-40), ("qv", 1.0e-16),
                                ("qv", -1.0e-9)]):
        c.set_level(f, 1 + 2 * k, v)
    cols.append(c)

    c = Column("edge-psaut", "edge: cloud ice above the aggregation limit", ts=270.0,
               rh=lambda z: np.full_like(z, 1.0), seed=203)
    c.add("qi", 1000.0, 9000.0, 3.0e-3)
    cols.append(c)

    c = Column("edge-dry-tiny", "edge: near-zero vapour and snow", ts=250.0,
               rh=lambda z: np.full_like(z, 1.0e-6), seed=204)
    for k in range(0, NZ, 4):
        c.set_level("qs", k, 5.0e-16)
    c.set_level("qs", 1, 1.0e-15)
    cols.append(c)

    c = Column("edge-cold-rain", "edge: rain far below freezing (supcol > 50)", ts=225.0,
               lapse=4.0e-3, rh=lambda z: np.full_like(z, 0.9), seed=205)
    c.add("qr", 0.0, 8000.0, 5.0e-4).add("qc", 0.0, 8000.0, 3.0e-4)
    c.add("qs", 0.0, 8000.0, 1.0e-4)
    cols.append(c)

    c = Column("edge-heavy", "edge: very large rain and snow", ts=306.0,
               lapse=7.2e-3, ztrop=14000.0, rh=lambda z: np.where(z < 10000, 0.95, 0.5),
               seed=206)
    c.add("qr", 0.0, 6000.0, 2.0e-2).add("qs", 4000.0, 13000.0, 1.5e-2)
    c.add("qc", 1000.0, 10000.0, 4.0e-3).add("qi", 9000.0, 15000.0, 2.0e-3)
    cols.append(c)

    c = Column("edge-zero", "edge: all hydrometeors zero, saturated", ts=280.0,
               rh=lambda z: np.full_like(z, 1.0), seed=207)
    cols.append(c)

    c = Column("edge-warm-ice", "edge: cloud ice above freezing melts at once", ts=292.0,
               rh=lambda z: np.full_like(z, 0.9), seed=208)
    c.add("qi", 0.0, 2500.0, 1.0e-4).add("qs", 0.0, 2500.0, 2.0e-4)
    cols.append(c)

    c = Column("edge-freezing-rain", "edge: warm nose over a cold surface", ts=268.0,
               lapse=2.0e-3, warm_nose=(600.0, 2500.0, 10.0),
               rh=lambda z: np.full_like(z, 0.96), seed=209)
    c.add("qs", 2200.0, 7000.0, 4.0e-4).add("qr", 0.0, 2600.0, 4.0e-4)
    c.add("qc", 0.0, 1500.0, 1.0e-4)
    cols.append(c)

    return cols


def _lev2_columns() -> list[Column]:
    cols: list[Column] = []
    combos = [
        ("rain", dict(qr=(1.0e-3, 2.0e-3)), 292.0, 0.9),
        ("snow", dict(qs=(1.0e-3, 2.0e-3)), 265.0, 0.9),
        ("ice", dict(qi=(2.0e-4, 4.0e-4)), 250.0, 1.0),
        ("rain-snow", dict(qr=(5.0e-4, 0.0), qs=(5.0e-4, 1.0e-3)), 276.0, 0.95),
        ("top-only-rain", dict(qr=(0.0, 1.0e-3)), 290.0, 0.6),
        ("top-only-snow", dict(qs=(0.0, 1.0e-3)), 270.0, 0.6),
        ("bottom-only", dict(qr=(1.0e-3, 0.0), qs=(1.0e-3, 0.0)), 275.0, 0.8),
        ("all", dict(qc=(3.0e-4, 2.0e-4), qr=(1.0e-3, 5.0e-4),
                     qi=(1.0e-4, 1.0e-4), qs=(8.0e-4, 8.0e-4)), 272.0, 1.0),
    ]
    for dz in (100.0, 600.0):
        for name, fields, ts, rh in combos:
            c = Column(f"lev2-{name}-dz{int(dz)}", "two levels", ts=ts, lapse=6.5e-3,
                       dz=np.array([dz, dz]), rh=lambda z, rh=rh: np.full_like(z, rh),
                       seed=300 + len(cols))
            for f, (lo, hi) in fields.items():
                c.set_level(f, 0, lo).set_level(f, 1, hi)
            cols.append(c)
    return cols


SETS = {"main": _main_columns, "lev2": _lev2_columns}


def write_set(out_dir: Path, name: str) -> dict:
    cols = SETS[name]()
    data = [c.fields() for c in cols]
    nz = len(data[0]["th"])
    with open(out_dir / f"{name}.bin", "wb") as fh:
        np.asarray([len(cols), nz], "<i4").tofile(fh)
        for field in LEVEL_FIELDS:
            np.stack([d[field] for d in data], axis=1).astype("<f4").tofile(fh)
        for field in SURFACE_FIELDS:
            np.asarray([getattr(c, field) for c in cols], "<f4").tofile(fh)
    meta = {"set": name, "ncol": len(cols), "nz": nz,
            "level_fields": list(LEVEL_FIELDS),
            "surface_fields": list(SURFACE_FIELDS),
            "columns": [{"name": c.name, "family": c.family} for c in cols]}
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def build(out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [write_set(out_dir, name) for name in SETS]


if __name__ == "__main__":
    for m in build(Path(sys.argv[1] if len(sys.argv) > 1 else ".")):
        print(f"{m['set']}: {m['ncol']} columns x {m['nz']} levels")
