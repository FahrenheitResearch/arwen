"""Deterministic WSM6 column set for the WRF v4.6.1 bitwise column oracle.

Writes one little-endian binary file that BOTH sides read, so the WRF
Fortran (run_wsm6_columns.F90) and the CUDA kernel see the same float32
words.  Nothing here is an expected value: it is the input.

Layout of ``columns.bin``::

    int32 ncol, int32 nz
    11 level fields, each nz*ncol float32, k-major with the column index
       fastest (Fortran ``a(i,k)`` order):
       th, qv, qc, qr, qi, qs, qg, den, pii, p, dz
    3 surface fields, each ncol float32: rain, snow, graupel (accumulated)

``columns.json`` names every column and its regime.

The set spans the regimes WSM6 meets in a forecast -- deep convection with
graupel/hail, stratiform rain through a melting layer, freezing rain, a
stable clear night, fog, marine stratocumulus and drizzle, all-snow winter
columns, high terrain, cirrus and supercooled cloud below -40 C -- plus
edge columns that sit exactly on the scheme's branch thresholds (qcrmin,
qc0, qmin, t0c, the triple point), carry negative or subnormal
hydrometeors, put all the condensate in the top or bottom level, or force
the semi-Lagrangian CFL limiter with thin layers.

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


class Column:
    def __init__(self, name: str, regime: str, *, ps=100000.0, ts=295.0,
                 lapse=6.5e-3, ztrop=12000.0, inv_depth=0.0, inv_dt=0.0,
                 warm_nose=None, rh=None, dz=None, seed=0):
        self.name, self.regime = name, regime
        self.dz = _interfaces() if dz is None else np.asarray(dz, float)
        zi, self.z, self.t, self.p, self.qv = profile(
            self.dz, ps=ps, ts=ts, lapse=lapse, ztrop=ztrop,
            inv_depth=inv_depth, inv_dt=inv_dt, warm_nose=warm_nose,
            rh=rh, seed=seed)
        z = self.z
        zero = np.zeros_like(z)
        self.qc, self.qr, self.qi, self.qs, self.qg = (zero.copy() for _ in range(5))
        self.rain = 0.0
        self.snow = 0.0
        self.graupel = 0.0
        self.overrides = []

    def add(self, field: str, z1: float, z2: float, peak: float):
        setattr(self, field, getattr(self, field) + _bump(self.z, z1, z2, peak))
        return self

    def set_level(self, field: str, k: int, value):
        self.overrides.append((field, k, value))
        return self

    def fields(self) -> dict[str, np.ndarray]:
        t = self.t.astype(F32)
        p = self.p.astype(F32)
        qv = self.qv.astype(F32)
        out = {"t": t, "p": p, "qv": qv,
               "qc": self.qc.astype(F32), "qr": self.qr.astype(F32),
               "qi": self.qi.astype(F32), "qs": self.qs.astype(F32),
               "qg": self.qg.astype(F32), "dz": self.dz.astype(F32)}
        for field, k, value in self.overrides:
            out[field][k] = F32(value)
        # WRF driver quantities, formed in float32 as the WRF driver forms them
        # in its own arithmetic.  They are inputs: both sides read these bits.
        return driver_fields(out)


def _columns() -> list[Column]:
    cols: list[Column] = []
    up = lambda x: float(np.nextafter(F32(x), F32(np.inf)))
    dn = lambda x: float(np.nextafter(F32(x), F32(-np.inf)))

    # -- deep convection (warm, moist, heavy mixed phase) ---------------------
    for n, (ts, qcp, qrp, qsp, qgp, qip) in enumerate([
            (303.0, 2.5e-3, 6.0e-3, 1.5e-3, 6.0e-3, 4.0e-4),
            (300.0, 1.8e-3, 3.0e-3, 2.0e-3, 2.5e-3, 2.0e-4),
            (305.0, 3.0e-3, 9.0e-3, 1.0e-3, 1.2e-2, 6.0e-4),
            (298.0, 1.2e-3, 1.5e-3, 2.5e-3, 1.0e-3, 1.0e-4),
            (301.0, 2.2e-3, 4.5e-3, 3.0e-3, 4.0e-3, 8.0e-4),
            (296.0, 8.0e-4, 8.0e-4, 1.2e-3, 5.0e-4, 5.0e-5),
            (304.0, 2.8e-3, 1.5e-2, 5.0e-4, 2.0e-2, 3.0e-4),
            (299.0, 1.5e-3, 2.0e-3, 1.8e-3, 3.0e-3, 1.5e-4)]):
        c = Column(f"convective-{n}", "deep convection", ts=ts, lapse=6.8e-3,
                   ztrop=13500.0, rh=lambda z: np.where(z < 9000, 0.95, 0.6), seed=10 + n)
        c.add("qc", 800.0, 9500.0, qcp).add("qr", 0.0, 7000.0, qrp)
        c.add("qs", 5000.0, 13000.0, qsp).add("qg", 2500.0, 11000.0, qgp)
        c.add("qi", 8000.0, 15000.0, qip)
        c.rain, c.snow, c.graupel = 3.5 * n, 0.25 * n, 0.5 * n
        cols.append(c)

    # -- stratiform rain through a melting layer -------------------------------
    for n, (ts, frz) in enumerate([(285.0, 3000.0), (281.0, 2000.0),
                                   (289.0, 4000.0), (278.0, 1200.0),
                                   (283.0, 2500.0), (287.0, 3500.0)]):
        c = Column(f"stratiform-{n}", "stratiform melting layer", ts=ts, lapse=6.0e-3,
                   rh=lambda z: np.where(z < 8000, 0.97, 0.5), seed=30 + n)
        c.add("qs", frz - 300.0, 9000.0, 6.0e-4 + 1.0e-4 * n)
        c.add("qr", 0.0, frz + 200.0, 4.0e-4 + 1.5e-4 * n)
        c.add("qc", frz - 1500.0, frz + 1200.0, 1.5e-4)
        c.add("qg", frz - 200.0, frz + 1500.0, 5.0e-5 * n)
        c.add("qi", 7000.0, 10000.0, 3.0e-5)
        cols.append(c)

    # -- freezing rain / sleet: warm nose over a cold surface ------------------
    for n in range(4):
        c = Column(f"freezing-rain-{n}", "freezing rain", ts=268.0 + n, lapse=2.0e-3,
                   warm_nose=(600.0, 2500.0, 9.0 + n), rh=lambda z: np.full_like(z, 0.96),
                   seed=40 + n)
        c.add("qs", 2200.0, 7000.0, 4.0e-4).add("qr", 0.0, 2600.0, 3.0e-4 + 1.0e-4 * n)
        c.add("qc", 0.0, 1500.0, 1.0e-4 * n).add("qg", 0.0, 800.0, 2.0e-5 * n)
        cols.append(c)

    # -- stable clear night (dry, inversion) and fog ---------------------------
    for n in range(6):
        c = Column(f"stable-night-{n}", "stable clear night", ts=275.0 + 3.0 * n,
                   inv_depth=300.0 + 100.0 * n, inv_dt=6.0 + n,
                   rh=lambda z: np.full_like(z, 0.35), seed=50 + n)
        cols.append(c)
    for n in range(4):
        c = Column(f"fog-{n}", "radiation fog", ts=271.0 + 4.0 * n, inv_depth=250.0,
                   inv_dt=4.0, rh=lambda z: np.where(z < 300, 1.01, 0.5), seed=60 + n)
        c.add("qc", 0.0, 300.0, 1.2e-4 + 5.0e-5 * n)
        if n % 2:
            c.add("qi", 0.0, 300.0, 1.0e-6)
        cols.append(c)

    # -- marine stratocumulus and drizzle over ocean ---------------------------
    for n in range(6):
        c = Column(f"ocean-sc-{n}", "ocean stratocumulus", ps=101500.0, ts=290.0 + n,
                   inv_depth=0.0, rh=lambda z: np.where(z < 1100, 0.99, 0.2), seed=70 + n)
        c.add("qc", 500.0, 1150.0, 4.0e-4 + 1.0e-4 * n)
        c.add("qr", 0.0, 1100.0, 2.0e-5 * n)
        cols.append(c)

    # -- all-snow winter columns -----------------------------------------------
    for n in range(8):
        c = Column(f"snow-{n}", "winter snow", ts=262.0 + 1.5 * n, lapse=5.5e-3,
                   ztrop=9000.0, rh=lambda z: np.where(z < 5000, 0.98, 0.6), seed=80 + n)
        c.add("qs", 0.0, 6000.0, 2.0e-4 + 1.0e-4 * n)
        c.add("qi", 2000.0, 8000.0, 2.0e-5 + 1.0e-5 * n)
        c.add("qc", 300.0, 2500.0, 5.0e-5 * (n % 3))
        c.add("qg", 0.0, 2000.0, 1.0e-5 * (n % 2))
        c.snow = 1.0 + n
        cols.append(c)

    # -- high terrain ----------------------------------------------------------
    for n in range(6):
        c = Column(f"high-terrain-{n}", "high terrain", ps=62000.0 + 3000.0 * n,
                   ts=270.0 + 3.0 * n, lapse=7.0e-3, ztrop=11000.0,
                   rh=lambda z: np.where(z < 4000, 0.9, 0.3), seed=90 + n)
        c.add("qs", 0.0, 4500.0, 3.0e-4).add("qc", 200.0, 2500.0, 2.0e-4)
        c.add("qg", 500.0, 3000.0, 2.0e-4 * (n % 2)).add("qr", 0.0, 800.0, 1.0e-4 * (n % 3))
        cols.append(c)

    # -- cirrus, and supercooled cloud below -40 C ------------------------------
    for n in range(2):
        c = Column(f"cirrus-{n}", "cirrus", ts=288.0 + n, rh=lambda z: np.where(z > 9000, 1.05, 0.3),
                   seed=100 + n)
        c.add("qi", 9000.0, 13000.0, 5.0e-5 + 2.0e-4 * n)
        cols.append(c)
    for n in range(2):
        c = Column(f"supercooled-{n}", "supercooled below -40C", ts=248.0 - 5.0 * n,
                   lapse=7.0e-3, ztrop=9000.0, rh=lambda z: np.full_like(z, 1.0), seed=110 + n)
        c.add("qc", 0.0, 9000.0, 2.0e-4)
        cols.append(c)

    # -- edge columns ------------------------------------------------------------
    base = dict(ts=282.0, lapse=6.5e-3, rh=lambda z: np.where(z < 6000, 0.98, 0.5))
    c = Column("edge-thresholds", "edge: branch thresholds", seed=200, **base)
    qcrmin, qmin = 1.0e-9, 1.0e-15
    qc0 = float(F32(F32(4.0 / 3.0) * F32(np.pi) * F32(1000.0) * F32(F32(0.8e-5) ** 3)
                    * F32(3.0e8) / F32(1.28)))
    vals = [("qr", qcrmin), ("qs", qcrmin), ("qg", qcrmin), ("qc", qc0), ("qi", qmin),
            ("qc", qmin)]
    k = 2
    for field, v in vals:
        for x in (dn(v), v, up(v)):
            c.set_level(field, k, x)
            k += 1
    cols.append(c)

    c = Column("edge-temperatures", "edge: t0c and triple point", seed=201, **base)
    c.add("qs", 0.0, 6000.0, 3.0e-4).add("qr", 0.0, 6000.0, 3.0e-4)
    c.add("qg", 0.0, 6000.0, 1.0e-4).add("qi", 0.0, 6000.0, 1.0e-5)
    c.add("qc", 0.0, 6000.0, 2.0e-4)
    t0c = 273.15
    ttp = float(F32(F32(273.15) + F32(0.01)))
    temps = [dn(t0c), t0c, up(t0c), dn(ttp), ttp, up(ttp), 233.15, dn(233.15), 223.15, 200.0]
    for k, tt in enumerate(temps):
        c.t[3 + 2 * k] = tt
    cols.append(c)

    c = Column("edge-negative", "edge: negative and subnormal inputs", seed=202, **base)
    c.add("qr", 0.0, 3000.0, 2.0e-4).add("qs", 2000.0, 6000.0, 2.0e-4)
    for k, (f, v) in enumerate([("qc", -1.0e-10), ("qr", -3.0e-12), ("qi", -1.0e-14),
                                ("qs", -2.0e-11), ("qg", -5.0e-13), ("qc", 1.0e-40),
                                ("qr", 3.0e-41), ("qs", 1.0e-39), ("qg", 7.0e-42),
                                ("qi", 2.0e-40), ("qv", 1.0e-16), ("qv", -1.0e-9)]):
        c.set_level(f, 1 + 2 * k, v)
    cols.append(c)

    c = Column("edge-top-only", "edge: condensate in the top level only", ts=280.0,
               rh=lambda z: np.full_like(z, 0.5), seed=203)
    for f, v in (("qr", 1.0e-4), ("qs", 2.0e-4), ("qg", 3.0e-4), ("qi", 5.0e-5), ("qc", 1.0e-5)):
        c.set_level(f, NZ - 1, v)
    # negative zeros: WRF's MAX(q, 0.0) keeps -0.0 (gfortran's MAX_EXPR
    # returns its first operand on a tie), fmaxf would not
    for k, f in enumerate(("qc", "qr", "qi", "qs", "qg")):
        c.set_level(f, 3 + k, -0.0)
    cols.append(c)

    c = Column("edge-bottom-only", "edge: condensate in the lowest level only", ts=276.0,
               rh=lambda z: np.full_like(z, 0.8), seed=204)
    for f, v in (("qr", 2.0e-3), ("qs", 1.0e-3), ("qg", 4.0e-3), ("qi", 1.0e-4), ("qc", 5.0e-4)):
        c.set_level(f, 0, v)
    c.rain, c.snow, c.graupel = 123.456, 7.25, 3.0
    cols.append(c)

    thin = np.full(NZ, 5.0)
    thin[20:] = _interfaces()[20:]
    c = Column("edge-thin-layers", "edge: thin layers force the CFL limiter", ts=296.0,
               dz=thin, rh=lambda z: np.full_like(z, 0.9), seed=205)
    c.add("qr", 0.0, 3000.0, 8.0e-3).add("qg", 0.0, 3000.0, 1.0e-2)
    c.add("qs", 0.0, 3000.0, 2.0e-3)
    cols.append(c)

    c = Column("edge-supersaturated-clear", "edge: supersaturated, no condensate",
               ts=290.0, rh=lambda z: np.full_like(z, 1.03), seed=206)
    cols.append(c)

    c = Column("edge-subsaturated-cloud", "edge: cloud in dry air evaporates",
               ts=293.0, rh=lambda z: np.full_like(z, 0.4), seed=207)
    c.add("qc", 0.0, 9000.0, 1.0e-3).add("qi", 6000.0, 12000.0, 2.0e-4)
    c.add("qs", 3000.0, 12000.0, 4.0e-4).add("qr", 0.0, 6000.0, 6.0e-4)
    c.add("qg", 2000.0, 8000.0, 2.0e-4)
    cols.append(c)

    c = Column("edge-heavy-hail", "edge: very large graupel and rain", ts=306.0,
               lapse=7.2e-3, ztrop=14000.0, rh=lambda z: np.where(z < 10000, 0.95, 0.5),
               seed=208)
    c.add("qg", 0.0, 12000.0, 2.5e-2).add("qr", 0.0, 6000.0, 2.0e-2)
    c.add("qc", 1000.0, 10000.0, 4.0e-3).add("qs", 6000.0, 13000.0, 4.0e-3)
    c.add("qi", 9000.0, 15000.0, 2.0e-3)
    cols.append(c)

    c = Column("edge-psaut", "edge: cloud ice above the aggregation limit", ts=270.0,
               rh=lambda z: np.full_like(z, 1.0), seed=209)
    c.add("qi", 1000.0, 9000.0, 3.0e-3)
    cols.append(c)

    c = Column("edge-dry-tiny", "edge: near-zero vapour, qsum at qmin", ts=250.0,
               rh=lambda z: np.full_like(z, 1.0e-6), seed=210)
    for k in range(0, NZ, 4):
        c.set_level("qs", k, 5.0e-16)
        c.set_level("qg", k, 5.0e-16)
    c.set_level("qs", 1, 1.0e-15)
    c.set_level("qg", 2, 1.0e-15)
    cols.append(c)

    c = Column("edge-cold-rain", "edge: rain far below freezing (supcol > 50)", ts=225.0,
               lapse=4.0e-3, rh=lambda z: np.full_like(z, 0.9), seed=211)
    c.add("qr", 0.0, 8000.0, 5.0e-4).add("qc", 0.0, 8000.0, 3.0e-4)
    c.add("qs", 0.0, 8000.0, 1.0e-4)
    cols.append(c)

    c = Column("edge-zero", "edge: all hydrometeors zero, saturated", ts=280.0,
               rh=lambda z: np.full_like(z, 1.0), seed=212)
    cols.append(c)

    return cols


def build(out_dir: Path) -> dict:
    cols = _columns()
    data = [c.fields() for c in cols]
    names = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "den", "pii", "p", "dz")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "columns.bin", "wb") as fh:
        np.asarray([len(cols), NZ], "<i4").tofile(fh)
        for name in names:
            np.stack([d[name] for d in data], axis=1).astype("<f4").tofile(fh)
        for name in ("rain", "snow", "graupel"):
            np.asarray([getattr(c, name) for c in cols], "<f4").tofile(fh)
    meta = {"ncol": len(cols), "nz": NZ,
            "level_fields": list(names),
            "surface_fields": ["rain", "snow", "graupel"],
            "columns": [{"name": c.name, "regime": c.regime} for c in cols]}
    (out_dir / "columns.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


if __name__ == "__main__":
    m = build(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
    print(f"{m['ncol']} columns x {m['nz']} levels")
