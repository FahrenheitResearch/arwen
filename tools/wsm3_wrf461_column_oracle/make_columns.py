"""Deterministic WSM3 column set for the WRF v4.6.1 bitwise column oracle.

Writes one little-endian binary file that BOTH sides read, so the WRF
Fortran (run_wsm3_columns.F90) and the CUDA kernel see the same float32
words.  Nothing here is an expected value: it is the input.

Layout of ``columns.bin``::

    int32 ncol, int32 nz
    8 level fields, each nz*ncol float32, k-major with the column index
       fastest (Fortran ``a(i,k)`` order):
       th, qv, qc, qr, den, pii, p, dz
    1 full-level field, (nz+1)*ncol float32: w (vertical velocity on the
       nz+1 layer faces, bottom face first, as WRF's w_2 is staggered)
    2 surface fields, each ncol float32: rain, snow (accumulated)

``columns.json`` names every column and its regime.

WSM3 carries three water species: vapour, cloud (``qc``, cloud water above
0 C and cloud ice below) and precipitation (``qr``, rain above 0 C and snow
below).  The set spans warm rain, deep convection whose cloud and
precipitation cross the melting level in updrafts and downdrafts (the
freezing/melting step reads w at the melting level), stratiform rain
through a melting layer, freezing rain under a warm nose, all-cold winter
columns, cirrus, dry columns, supersaturated columns (warm condensation and
cold ice nucleation), a high model top reaching the upper stratosphere,
fog and high terrain, plus edge columns that sit exactly on the scheme's
branch thresholds (qcrmin, qc0, qmin, t0c, the triple point, roqimax),
carry negative or subnormal hydrometeors, put all the condensate in the top
or bottom level, or force the semi-Lagrangian CFL limiter with thin layers.

Every column keeps its top level below 0 C: a column warm at its top level
with rising air there would send WRF's freezing step to level kte+1, which
reads the two-dimensional temperature array out of bounds
(module_mp_wsm3.F:572-592) -- undefined behaviour, so not an oracle input.

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
                 lapse=6.5e-3, ztrop=12000.0, strat=1.5e-3, inv_depth=0.0,
                 inv_dt=0.0, warm_nose=None, rh=None, dz=None, seed=0):
        self.name, self.regime = name, regime
        self.dz = _interfaces() if dz is None else np.asarray(dz, float)
        zi, self.z, self.t, self.p, self.qv = profile(
            self.dz, ps=ps, ts=ts, lapse=lapse, ztrop=ztrop,
            inv_depth=inv_depth, inv_dt=inv_dt, warm_nose=warm_nose,
            rh=rh, seed=seed, strat=strat)
        self.zi = zi
        z = self.z
        self.qc = np.zeros_like(z)
        self.qr = np.zeros_like(z)
        self.w = np.zeros_like(zi)
        self.rain = 0.0
        self.snow = 0.0
        self.overrides = []

    def add(self, field: str, z1: float, z2: float, peak: float):
        if field == "w":
            self.w = self.w + _bump(self.zi, z1, z2, peak)
        else:
            setattr(self, field, getattr(self, field) + _bump(self.z, z1, z2, peak))
        return self

    def set_level(self, field: str, k: int, value):
        self.overrides.append((field, k, value))
        return self

    def fields(self) -> dict[str, np.ndarray]:
        out = {"t": self.t.astype(F32), "p": self.p.astype(F32),
               "qv": self.qv.astype(F32), "qc": self.qc.astype(F32),
               "qr": self.qr.astype(F32), "dz": self.dz.astype(F32),
               "w": self.w.astype(F32)}
        for field, k, value in self.overrides:
            out[field][k] = F32(value)
        if out["t"][-1] >= F32(273.15):
            raise ValueError(f"{self.name}: top level is not below 0 C")
        # WRF driver quantities, formed in float32.  They are inputs: both
        # sides read these bits.
        return driver_fields(out)


def _high_top_dz() -> np.ndarray:
    """45 layers to about 47 km: thin at the ground, 2 km at the top."""
    return _interfaces(dz0=50.0, growth=1.105, dzmax=2000.0)


def _columns() -> list[Column]:
    cols: list[Column] = []
    up = lambda x: float(np.nextafter(F32(x), F32(np.inf)))
    dn = lambda x: float(np.nextafter(F32(x), F32(-np.inf)))

    # -- warm rain: maritime cloud and rain wholly below the melting level ---
    for n, (ts, qcp, qrp, wpk) in enumerate([
            (299.0, 8.0e-4, 6.0e-4, 0.5), (301.0, 1.5e-3, 1.2e-3, -0.8),
            (297.0, 4.0e-4, 2.0e-4, 0.0), (303.0, 2.0e-3, 2.5e-3, 1.5),
            (300.0, 1.0e-3, 4.0e-5, -0.2), (298.0, 6.0e-4, 1.5e-3, 0.3),
            (302.0, 2.5e-3, 3.0e-4, 2.0), (296.0, 3.0e-4, 8.0e-4, -1.0)]):
        c = Column(f"warm-rain-{n}", "warm rain", ps=101200.0, ts=ts,
                   rh=lambda z: np.where(z < 3500, 0.97, 0.45), seed=10 + n)
        c.add("qc", 500.0, 3200.0, qcp).add("qr", 0.0, 3000.0, qrp)
        c.add("w", 0.0, 4000.0, wpk)
        c.rain = 2.0 * n
        cols.append(c)

    # -- deep convection: cloud and precipitation across the melting level --
    for n, (ts, qcp, qrp, wpk) in enumerate([
            (303.0, 2.5e-3, 6.0e-3, 12.0), (300.0, 1.8e-3, 3.0e-3, 6.0),
            (305.0, 3.0e-3, 9.0e-3, 20.0), (298.0, 1.2e-3, 1.5e-3, -4.0),
            (301.0, 2.2e-3, 4.5e-3, 8.0), (296.0, 8.0e-4, 8.0e-4, -8.0),
            (304.0, 2.8e-3, 1.5e-2, 15.0), (299.0, 1.5e-3, 2.0e-3, 0.0)]):
        c = Column(f"convective-{n}", "deep convection", ts=ts, lapse=6.8e-3,
                   ztrop=13500.0, rh=lambda z: np.where(z < 9000, 0.95, 0.6),
                   seed=30 + n)
        c.add("qc", 800.0, 11000.0, qcp).add("qr", 0.0, 12000.0, qrp)
        c.add("w", 0.0, 14000.0, wpk)
        c.rain, c.snow = 3.5 * n, 0.25 * n
        cols.append(c)

    # -- stratiform: snow above, rain below a melting layer ------------------
    for n, (ts, frz, wpk) in enumerate([(285.0, 3000.0, 0.05), (281.0, 2000.0, -0.05),
                                        (289.0, 4000.0, 0.2), (278.0, 1200.0, 0.0),
                                        (283.0, 2500.0, -0.3), (287.0, 3500.0, 0.1)]):
        c = Column(f"stratiform-{n}", "stratiform melting layer", ts=ts, lapse=6.0e-3,
                   rh=lambda z: np.where(z < 8000, 0.97, 0.5), seed=50 + n)
        c.add("qr", 0.0, 9000.0, 6.0e-4 + 1.5e-4 * n)
        c.add("qc", frz - 1500.0, frz + 1200.0, 1.5e-4)
        c.add("w", 0.0, 10000.0, wpk)
        cols.append(c)

    # -- freezing rain: a warm nose over a cold surface ----------------------
    for n in range(4):
        c = Column(f"freezing-rain-{n}", "freezing rain", ts=268.0 + n, lapse=2.0e-3,
                   warm_nose=(600.0, 2500.0, 9.0 + n), rh=lambda z: np.full_like(z, 0.96),
                   seed=70 + n)
        c.add("qr", 0.0, 7000.0, 4.0e-4 + 1.0e-4 * n)
        c.add("qc", 0.0, 1500.0, 1.0e-4 * n)
        c.add("w", 0.0, 3000.0, (-0.1, 0.0, 0.1, 0.4)[n])
        cols.append(c)

    # -- cold: winter columns below freezing everywhere ----------------------
    for n in range(8):
        c = Column(f"cold-ice-{n}", "cold ice", ts=255.0 + 2.0 * n, lapse=5.5e-3,
                   ztrop=9000.0, rh=lambda z: np.where(z < 5000, 0.98, 0.6), seed=80 + n)
        c.add("qr", 0.0, 6000.0, 2.0e-4 + 1.0e-4 * n)
        c.add("qc", 300.0, 8000.0, 2.0e-5 + 4.0e-5 * n)
        c.add("w", 0.0, 6000.0, (0.0, 0.3, -0.3, 1.0)[n % 4])
        c.snow = 1.0 + n
        cols.append(c)

    # -- cirrus --------------------------------------------------------------
    for n in range(4):
        c = Column(f"cirrus-{n}", "cirrus", ts=288.0 + n,
                   rh=lambda z, n=n: np.where(z > 9000, 0.9 + 0.05 * n, 0.3), seed=100 + n)
        c.add("qc", 9000.0, 13000.0, 5.0e-6 + 6.0e-5 * n)
        cols.append(c)

    # -- dry -------------------------------------------------------------------
    for n in range(6):
        if n < 3:
            c = Column(f"dry-night-{n}", "dry", ts=275.0 + 4.0 * n,
                       inv_depth=300.0 + 100.0 * n, inv_dt=6.0 + n,
                       rh=lambda z: np.full_like(z, 0.2), seed=110 + n)
        else:
            c = Column(f"dry-desert-{n}", "dry", ts=310.0 + 2.0 * n, lapse=9.0e-3,
                       ztrop=12000.0, rh=lambda z: np.full_like(z, 0.05), seed=110 + n)
            c.add("w", 0.0, 3000.0, 2.0)
        cols.append(c)

    # -- supersaturated --------------------------------------------------------
    for n, (ts, rh, qcp) in enumerate([(290.0, 1.03, 0.0), (290.0, 1.10, 2.0e-4),
                                       (262.0, 1.08, 0.0), (255.0, 1.20, 0.0),
                                       (270.0, 1.05, 5.0e-5), (300.0, 1.02, 1.0e-3)]):
        c = Column(f"supersat-{n}", "supersaturated", ts=ts,
                   rh=lambda z, rh=rh: np.full_like(z, rh), seed=130 + n)
        c.add("qc", 0.0, 8000.0, qcp)
        cols.append(c)

    # -- high model top: 45 levels to about 47 km --------------------------------
    for n in range(6):
        c = Column(f"high-top-{n}", "high model top", ts=285.0 + 3.0 * n,
                   ztrop=11000.0, strat=1.0e-3, dz=_high_top_dz(),
                   rh=lambda z: np.where(z < 9000, 0.9, 0.02), seed=150 + n)
        c.add("qc", 1000.0, 9000.0, 3.0e-4 * (n % 3))
        c.add("qr", 0.0, 8000.0, 4.0e-4 * ((n + 1) % 3))
        c.add("w", 0.0, 12000.0, (0.0, 3.0, -2.0)[n % 3])
        if n == 5:
            # condensate in the thin stratosphere
            c.set_level("qc", NZ - 3, 1.0e-6).set_level("qr", NZ - 2, 1.0e-6)
        cols.append(c)

    # -- fog and high terrain ------------------------------------------------------
    for n in range(2):
        c = Column(f"fog-{n}", "radiation fog", ts=271.0 + 6.0 * n, inv_depth=250.0,
                   inv_dt=4.0, rh=lambda z: np.where(z < 300, 1.01, 0.5), seed=170 + n)
        c.add("qc", 0.0, 300.0, 1.2e-4 + 5.0e-5 * n)
        cols.append(c)
    for n in range(4):
        c = Column(f"high-terrain-{n}", "high terrain", ps=62000.0 + 3000.0 * n,
                   ts=270.0 + 4.0 * n, lapse=7.0e-3, ztrop=11000.0,
                   rh=lambda z: np.where(z < 4000, 0.9, 0.3), seed=180 + n)
        c.add("qr", 0.0, 4500.0, 3.0e-4).add("qc", 200.0, 2500.0, 2.0e-4)
        c.add("w", 0.0, 3000.0, (0.0, 0.5, -0.5, 2.0)[n])
        cols.append(c)

    # -- edge columns ------------------------------------------------------------
    base = dict(ts=282.0, lapse=6.5e-3, rh=lambda z: np.where(z < 6000, 0.98, 0.5))
    c = Column("edge-thresholds", "edge: branch thresholds", seed=200, **base)
    qcrmin, qmin = 1.0e-9, 1.0e-15
    qc0 = float(F32(F32(4.0 / 3.0) * F32(np.pi) * F32(1000.0) * F32(F32(0.8e-5) ** 3)
                    * F32(3.0e8) / F32(1.28)))
    roqimax = 8.125e-5
    vals = [("qr", qcrmin), ("qc", qc0), ("qc", qmin), ("qr", qmin)]
    k = 2
    for field, v in vals:
        for x in (dn(v), v, up(v)):
            c.set_level(field, k, x)
            k += 1
    # the cold half: the aggregation threshold roqimax/den and ice at qmin
    for x in (roqimax, 1.0e-3, 2.0e-3):
        c.set_level("qc", 30 + k - 14, x)
        k += 1
    cols.append(c)

    c = Column("edge-temperatures", "edge: t0c and triple point", seed=201, **base)
    c.add("qr", 0.0, 6000.0, 3.0e-4).add("qc", 0.0, 6000.0, 2.0e-4)
    t0c = 273.15
    ttp = float(F32(F32(273.15) + F32(0.01)))
    temps = [dn(t0c), t0c, up(t0c), dn(ttp), ttp, up(ttp), 233.15, dn(233.15), 223.15, 200.0]
    for k, tt in enumerate(temps):
        c.t[3 + 2 * k] = tt
    cols.append(c)

    c = Column("edge-negative", "edge: negative and subnormal inputs", seed=202, **base)
    c.add("qr", 0.0, 3000.0, 2.0e-4).add("qc", 2000.0, 6000.0, 2.0e-4)
    for k, (f, v) in enumerate([("qc", -1.0e-10), ("qr", -3.0e-12), ("qc", -1.0e-14),
                                ("qr", -2.0e-11), ("qc", 1.0e-40), ("qr", 3.0e-41),
                                ("qr", 1.0e-39), ("qc", 2.0e-40), ("qv", 1.0e-16),
                                ("qv", -1.0e-9)]):
        c.set_level(f, 1 + 2 * k, v)
    cols.append(c)

    c = Column("edge-top-only", "edge: condensate in the top level only", ts=280.0,
               rh=lambda z: np.full_like(z, 0.5), seed=203)
    for f, v in (("qr", 1.0e-4), ("qc", 5.0e-5)):
        c.set_level(f, NZ - 1, v)
    # negative zeros: WRF's MAX(q, 0.0) keeps -0.0 (gfortran's MAX_EXPR
    # returns its first operand on a tie), fmaxf would not
    for k, f in enumerate(("qc", "qr")):
        c.set_level(f, 3 + k, -0.0)
    cols.append(c)

    c = Column("edge-bottom-only", "edge: condensate in the lowest level only", ts=276.0,
               rh=lambda z: np.full_like(z, 0.8), seed=204)
    c.set_level("qr", 0, 2.0e-3).set_level("qc", 0, 5.0e-4)
    c.rain, c.snow = 123.456, 7.25
    cols.append(c)

    thin = np.full(NZ, 5.0)
    thin[20:] = _interfaces()[20:]
    c = Column("edge-thin-layers", "edge: thin layers force the CFL limiter", ts=296.0,
               dz=thin, rh=lambda z: np.full_like(z, 0.9), seed=205)
    c.add("qr", 0.0, 3000.0, 8.0e-3).add("qc", 0.0, 3000.0, 2.0e-3)
    cols.append(c)

    c = Column("edge-subsaturated-cloud", "edge: cloud in dry air evaporates",
               ts=293.0, rh=lambda z: np.full_like(z, 0.4), seed=207)
    c.add("qc", 0.0, 12000.0, 1.0e-3).add("qr", 0.0, 12000.0, 6.0e-4)
    cols.append(c)

    c = Column("edge-heavy-loading", "edge: very large rain and snow", ts=306.0,
               lapse=7.2e-3, ztrop=14000.0, rh=lambda z: np.where(z < 10000, 0.95, 0.5),
               seed=208)
    c.add("qr", 0.0, 13000.0, 2.5e-2).add("qc", 1000.0, 12000.0, 5.0e-3)
    c.add("w", 0.0, 13000.0, 25.0)
    cols.append(c)

    c = Column("edge-aggregation", "edge: cloud ice above the aggregation limit", ts=268.0,
               rh=lambda z: np.full_like(z, 1.0), seed=209)
    c.add("qc", 1000.0, 9000.0, 3.0e-3)
    cols.append(c)

    c = Column("edge-dry-tiny", "edge: near-zero vapour and condensate", ts=250.0,
               rh=lambda z: np.full_like(z, 1.0e-6), seed=210)
    for k in range(0, NZ, 4):
        c.set_level("qr", k, 5.0e-16)
        c.set_level("qc", k, 5.0e-16)
    c.set_level("qr", 1, 1.0e-15)
    c.set_level("qc", 2, 1.0e-15)
    cols.append(c)

    c = Column("edge-zero", "edge: all hydrometeors zero, saturated", ts=280.0,
               rh=lambda z: np.full_like(z, 1.0), seed=212)
    cols.append(c)

    # the melting level at the lowest level only, with rising and sinking air
    for n, wpk in enumerate((1.0, -1.0)):
        c = Column(f"edge-melt-surface-{n}", "edge: melting level at the ground",
                   ts=274.0, lapse=6.5e-3, rh=lambda z: np.full_like(z, 0.97),
                   seed=213 + n)
        c.t[0] = 273.4
        c.t[1:] = np.minimum(c.t[1:], 272.8)
        c.add("qr", 0.0, 5000.0, 6.0e-4).add("qc", 0.0, 3000.0, 2.0e-4)
        c.add("w", 0.0, 2000.0, wpk)
        cols.append(c)

    # w exactly zero at the melting level face, and a strong updraft above it
    c = Column("edge-melt-w-zero", "edge: w = 0 at the melting level", ts=286.0,
               rh=lambda z: np.full_like(z, 0.97), seed=215)
    c.add("qr", 0.0, 8000.0, 8.0e-4).add("qc", 1000.0, 6000.0, 3.0e-4)
    c.add("w", 0.0, 9000.0, 3.0)
    mstep = int(np.nonzero(c.t >= 273.15)[0].max())
    c.set_level("w", mstep, 0.0)
    cols.append(c)

    # sedimentation into a warm lowest level: ice falling out of the column
    c = Column("edge-ice-into-warm-surface", "edge: cold air over a warm lowest level",
               ts=273.6, lapse=8.0e-3, rh=lambda z: np.full_like(z, 0.99), seed=216)
    c.t[0] = 273.3
    c.add("qc", 0.0, 2500.0, 2.0e-3).add("qr", 0.0, 2500.0, 1.0e-3)
    c.dz[0] = 10.0
    cols.append(c)

    return cols


def build(out_dir: Path) -> dict:
    cols = _columns()
    data = [c.fields() for c in cols]
    names = ("th", "qv", "qc", "qr", "den", "pii", "p", "dz")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "columns.bin", "wb") as fh:
        np.asarray([len(cols), NZ], "<i4").tofile(fh)
        for name in names:
            np.stack([d[name] for d in data], axis=1).astype("<f4").tofile(fh)
        np.stack([d["w"] for d in data], axis=1).astype("<f4").tofile(fh)
        for name in ("rain", "snow"):
            np.asarray([getattr(c, name) for c in cols], "<f4").tofile(fh)
    meta = {"ncol": len(cols), "nz": NZ,
            "level_fields": list(names), "face_fields": ["w"],
            "surface_fields": ["rain", "snow"],
            "columns": [{"name": c.name, "regime": c.regime} for c in cols]}
    (out_dir / "columns.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


if __name__ == "__main__":
    m = build(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
    print(f"{m['ncol']} columns x {m['nz']} levels")
