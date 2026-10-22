"""Deterministic WSM7 column sets for the WRF v4.6.1 bitwise column oracle.

Writes little-endian binary files that BOTH sides read, so the WRF Fortran
(run_wsm7_columns.F90) and the CUDA kernel see the same float32 words.
Nothing here is an expected value: it is the input.

Layout of ``columns.bin`` (45 levels) and ``columns72.bin`` (72 levels, the
deeper WSM7_KMAX tier)::

    int32 ncol, int32 nz
    12 level fields, each nz*ncol float32, k-major with the column index
       fastest (Fortran ``a(i,k)`` order):
       th, qv, qc, qr, qi, qs, qg, qh, den, pii, p, dz
    4 surface fields, each ncol float32: rain, snow, graupel, hail
       (accumulated totals)

``columns.json`` / ``columns72.json`` name every column and its regime.

The 45-level set spans the regimes WSM7 meets in a forecast: deep
hail-producing convection, hail melting through a warm sub-cloud layer and
evaporating in dry air, hail aloft sublimating, deep convection without
hail, stratiform rain through a melting layer, freezing rain, stable clear
nights, fog, marine stratocumulus, all-snow winter columns, high terrain,
cirrus and supercooled cloud below -40 C; plus edge columns on the scheme's
branch thresholds (qcrmin for every precipitating class, qc0, qmin, t0c, the
triple point), negative, negative-zero and subnormal inputs, condensate only
in the top or the bottom level (the top-level columns exercise declared
divergence D1), thin layers that force the sedimentation CFL limiter,
hail-graupel-snow mixtures straddling the wet-growth test, and a
supersaturated clear column.

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
    _qsat, _bump, profile, driver_fields)

F32 = np.float32
RD, CP, P0 = 287.0, 1004.5, 1.0e5
LEVEL_FIELDS = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
                "den", "pii", "p", "dz")
SURFACE_FIELDS = ("rain", "snow", "graupel", "hail")
SPECIES = ("qc", "qr", "qi", "qs", "qg", "qh")


def _interfaces(nz: int, dz0: float = 40.0, growth: float = 1.1) -> np.ndarray:
    """Layer depths: 40 m at the ground growing 10 % per layer, capped at
    1 km for 45 levels (top about 20.8 km) and 500 m for 72 (about 27.5 km):
    WRF-like model tops above the clouds."""
    dzmax = 1000.0 if nz <= 50 else 500.0
    dz = np.empty(nz)
    d = dz0
    for k in range(nz):
        dz[k] = d
        d = min(d * growth, dzmax)
    return dz


class Column:
    def __init__(self, nz: int, name: str, regime: str, *, ps=100000.0, ts=295.0,
                 lapse=6.5e-3, ztrop=12000.0, inv_depth=0.0, inv_dt=0.0,
                 warm_nose=None, rh=None, dz=None, seed=0):
        self.name, self.regime = name, regime
        self.nz = nz
        self.dz = _interfaces(nz) if dz is None else np.asarray(dz, float)
        zi, self.z, self.t, self.p, self.qv = profile(
            self.dz, ps=ps, ts=ts, lapse=lapse, ztrop=ztrop,
            inv_depth=inv_depth, inv_dt=inv_dt, warm_nose=warm_nose,
            rh=rh, seed=seed)
        z = self.z
        for f in SPECIES:
            setattr(self, f, np.zeros_like(z))
        self.rain = self.snow = self.graupel = self.hail = 0.0
        self.overrides = []

    def add(self, field: str, z1: float, z2: float, peak: float):
        setattr(self, field, getattr(self, field) + _bump(self.z, z1, z2, peak))
        return self

    def set_level(self, field: str, k: int, value):
        self.overrides.append((field, k, value))
        return self

    def fields(self) -> dict[str, np.ndarray]:
        out = {"t": self.t.astype(F32), "p": self.p.astype(F32),
               "qv": self.qv.astype(F32), "dz": self.dz.astype(F32)}
        for f in SPECIES:
            out[f] = getattr(self, f).astype(F32)
        for field, k, value in self.overrides:
            out[field][k] = F32(value)
        # WRF driver quantities, formed in float32.  They are inputs: both
        # sides read these bits.
        return driver_fields(out)


def _columns(nz: int) -> list[Column]:
    cols: list[Column] = []
    up = lambda x: float(np.nextafter(F32(x), F32(np.inf)))
    dn = lambda x: float(np.nextafter(F32(x), F32(-np.inf)))
    C = lambda *a, **k: Column(nz, *a, **k)

    # -- deep hail-producing convection --------------------------------------
    for n, (ts, qcp, qrp, qsp, qgp, qhp, qip) in enumerate([
            (303.0, 2.5e-3, 6.0e-3, 1.5e-3, 5.0e-3, 4.0e-3, 4.0e-4),
            (300.0, 1.8e-3, 3.0e-3, 2.0e-3, 2.5e-3, 1.5e-3, 2.0e-4),
            (305.0, 3.0e-3, 9.0e-3, 1.0e-3, 8.0e-3, 1.2e-2, 6.0e-4),
            (298.0, 1.2e-3, 1.5e-3, 2.5e-3, 1.0e-3, 6.0e-4, 1.0e-4),
            (301.0, 2.2e-3, 4.5e-3, 3.0e-3, 4.0e-3, 3.0e-3, 8.0e-4),
            (296.0, 8.0e-4, 8.0e-4, 1.2e-3, 5.0e-4, 2.0e-4, 5.0e-5),
            (304.0, 2.8e-3, 1.5e-2, 5.0e-4, 1.2e-2, 2.0e-2, 3.0e-4),
            (299.0, 1.5e-3, 2.0e-3, 1.8e-3, 3.0e-3, 2.5e-3, 1.5e-4),
            (307.0, 3.5e-3, 1.0e-2, 2.0e-3, 6.0e-3, 1.5e-2, 1.0e-3),
            (302.0, 2.0e-3, 5.0e-3, 2.5e-3, 3.5e-3, 8.0e-3, 5.0e-4)]):
        c = C(f"hail-convective-{n}", "deep hail convection", ts=ts, lapse=6.8e-3,
              ztrop=13500.0, rh=lambda z: np.where(z < 9000, 0.95, 0.6), seed=10 + n)
        c.add("qc", 800.0, 9500.0, qcp).add("qr", 0.0, 7000.0, qrp)
        c.add("qs", 5000.0, 13000.0, qsp).add("qg", 2500.0, 11000.0, qgp)
        c.add("qh", 1000.0 + 200.0 * n, 10000.0, qhp).add("qi", 8000.0, 15000.0, qip)
        c.rain, c.snow, c.graupel, c.hail = 3.5 * n, 0.25 * n, 0.5 * n, 0.75 * n
        cols.append(c)

    # -- hail melting through a warm, dry sub-cloud layer --------------------
    for n in range(4):
        c = C(f"hail-melting-{n}", "hail melting sub-cloud", ts=300.0 + 2.0 * n,
              lapse=8.0e-3, ztrop=13000.0,
              rh=lambda z: np.where(z < 2500, 0.35, np.where(z < 9000, 0.9, 0.5)),
              seed=120 + n)
        c.add("qh", 0.0, 7000.0, 3.0e-3 + 2.0e-3 * n).add("qg", 1500.0, 7000.0, 1.0e-3)
        c.add("qs", 4000.0, 10000.0, 8.0e-4).add("qr", 0.0, 3000.0, 5.0e-4 * n)
        c.add("qc", 3000.0, 7000.0, 6.0e-4)
        cols.append(c)

    # -- hail aloft in subsaturated cold air (phdep < 0, phaut, phacs) ------
    for n in range(3):
        c = C(f"hail-aloft-{n}", "hail aloft sublimating", ts=285.0 + 3.0 * n,
              rh=lambda z: np.full_like(z, 0.55), seed=130 + n)
        c.add("qh", 4000.0, 10000.0, 2.0e-3).add("qs", 5000.0, 11000.0, 1.0e-3)
        c.add("qg", 4500.0, 9000.0, 1.5e-3 * (n % 2)).add("qi", 7000.0, 12000.0, 1.0e-4)
        cols.append(c)

    # -- deep convection without hail ----------------------------------------
    for n, (ts, qcp, qrp, qsp, qgp, qip) in enumerate([
            (303.0, 2.5e-3, 6.0e-3, 1.5e-3, 6.0e-3, 4.0e-4),
            (300.0, 1.8e-3, 3.0e-3, 2.0e-3, 2.5e-3, 2.0e-4),
            (298.0, 1.2e-3, 1.5e-3, 2.5e-3, 1.0e-3, 1.0e-4),
            (301.0, 2.2e-3, 4.5e-3, 3.0e-3, 4.0e-3, 8.0e-4)]):
        c = C(f"convective-{n}", "deep convection", ts=ts, lapse=6.8e-3,
              ztrop=13500.0, rh=lambda z: np.where(z < 9000, 0.95, 0.6), seed=150 + n)
        c.add("qc", 800.0, 9500.0, qcp).add("qr", 0.0, 7000.0, qrp)
        c.add("qs", 5000.0, 13000.0, qsp).add("qg", 2500.0, 11000.0, qgp)
        c.add("qi", 8000.0, 15000.0, qip)
        cols.append(c)

    # -- stratiform rain through a melting layer -----------------------------
    for n, (ts, frz) in enumerate([(285.0, 3000.0), (281.0, 2000.0),
                                   (289.0, 4000.0), (278.0, 1200.0),
                                   (283.0, 2500.0), (287.0, 3500.0)]):
        c = C(f"stratiform-{n}", "stratiform melting layer", ts=ts, lapse=6.0e-3,
              rh=lambda z: np.where(z < 8000, 0.97, 0.5), seed=30 + n)
        c.add("qs", frz - 300.0, 9000.0, 6.0e-4 + 1.0e-4 * n)
        c.add("qr", 0.0, frz + 200.0, 4.0e-4 + 1.5e-4 * n)
        c.add("qc", frz - 1500.0, frz + 1200.0, 1.5e-4)
        c.add("qg", frz - 200.0, frz + 1500.0, 5.0e-5 * n)
        c.add("qh", frz - 100.0, frz + 1000.0, 2.0e-5 * (n % 3))
        c.add("qi", 7000.0, 10000.0, 3.0e-5)
        cols.append(c)

    # -- freezing rain / sleet ------------------------------------------------
    for n in range(4):
        c = C(f"freezing-rain-{n}", "freezing rain", ts=268.0 + n, lapse=2.0e-3,
              warm_nose=(600.0, 2500.0, 9.0 + n), rh=lambda z: np.full_like(z, 0.96),
              seed=40 + n)
        c.add("qs", 2200.0, 7000.0, 4.0e-4).add("qr", 0.0, 2600.0, 3.0e-4 + 1.0e-4 * n)
        c.add("qc", 0.0, 1500.0, 1.0e-4 * n).add("qg", 0.0, 800.0, 2.0e-5 * n)
        c.add("qh", 0.0, 600.0, 1.0e-5 * (n % 2))
        cols.append(c)

    # -- stable clear night and fog ------------------------------------------
    for n in range(4):
        cols.append(C(f"stable-night-{n}", "stable clear night", ts=275.0 + 3.0 * n,
                      inv_depth=300.0 + 100.0 * n, inv_dt=6.0 + n,
                      rh=lambda z: np.full_like(z, 0.35), seed=50 + n))
    for n in range(3):
        c = C(f"fog-{n}", "radiation fog", ts=271.0 + 4.0 * n, inv_depth=250.0,
              inv_dt=4.0, rh=lambda z: np.where(z < 300, 1.01, 0.5), seed=60 + n)
        c.add("qc", 0.0, 300.0, 1.2e-4 + 5.0e-5 * n)
        if n % 2:
            c.add("qi", 0.0, 300.0, 1.0e-6)
        cols.append(c)

    # -- marine stratocumulus and drizzle ------------------------------------
    for n in range(4):
        c = C(f"ocean-sc-{n}", "ocean stratocumulus", ps=101500.0, ts=290.0 + n,
              rh=lambda z: np.where(z < 1100, 0.99, 0.2), seed=70 + n)
        c.add("qc", 500.0, 1150.0, 4.0e-4 + 1.0e-4 * n)
        c.add("qr", 0.0, 1100.0, 2.0e-5 * n)
        cols.append(c)

    # -- all-snow winter columns ---------------------------------------------
    for n in range(5):
        c = C(f"snow-{n}", "winter snow", ts=262.0 + 1.5 * n, lapse=5.5e-3,
              ztrop=9000.0, rh=lambda z: np.where(z < 5000, 0.98, 0.6), seed=80 + n)
        c.add("qs", 0.0, 6000.0, 2.0e-4 + 1.0e-4 * n)
        c.add("qi", 2000.0, 8000.0, 2.0e-5 + 1.0e-5 * n)
        c.add("qc", 300.0, 2500.0, 5.0e-5 * (n % 3))
        c.add("qg", 0.0, 2000.0, 1.0e-5 * (n % 2))
        c.snow = 1.0 + n
        cols.append(c)

    # -- high terrain ----------------------------------------------------------
    for n in range(5):
        c = C(f"high-terrain-{n}", "high terrain", ps=62000.0 + 3000.0 * n,
              ts=270.0 + 3.0 * n, lapse=7.0e-3, ztrop=11000.0,
              rh=lambda z: np.where(z < 4000, 0.9, 0.3), seed=90 + n)
        c.add("qs", 0.0, 4500.0, 3.0e-4).add("qc", 200.0, 2500.0, 2.0e-4)
        c.add("qg", 500.0, 3000.0, 2.0e-4 * (n % 2)).add("qr", 0.0, 800.0, 1.0e-4 * (n % 3))
        c.add("qh", 300.0, 2500.0, 3.0e-4 * (n % 2))
        cols.append(c)

    # -- cirrus, and supercooled cloud below -40 C -----------------------------
    for n in range(2):
        c = C(f"cirrus-{n}", "cirrus", ts=288.0 + n,
              rh=lambda z: np.where(z > 9000, 1.05, 0.3), seed=100 + n)
        c.add("qi", 9000.0, 13000.0, 5.0e-5 + 2.0e-4 * n)
        cols.append(c)
    for n in range(2):
        c = C(f"supercooled-{n}", "supercooled below -40C", ts=248.0 - 5.0 * n,
              lapse=7.0e-3, ztrop=9000.0, rh=lambda z: np.full_like(z, 1.0), seed=110 + n)
        c.add("qc", 0.0, 9000.0, 2.0e-4)
        cols.append(c)

    # -- edge columns ----------------------------------------------------------
    base = dict(ts=282.0, lapse=6.5e-3, rh=lambda z: np.where(z < 6000, 0.98, 0.5))
    c = C("edge-thresholds", "edge: branch thresholds", seed=200, **base)
    qcrmin, qmin = 1.0e-9, 1.0e-15
    qc0 = float(F32(F32(4.0 / 3.0) * F32(np.pi) * F32(1000.0) * F32(F32(0.8e-5) ** 3)
                    * F32(3.0e8) / F32(1.28)))
    vals = [("qr", qcrmin), ("qs", qcrmin), ("qg", qcrmin), ("qh", qcrmin),
            ("qc", qc0), ("qi", qmin), ("qc", qmin), ("qh", 6.0e-4)]
    k = 1
    for field, v in vals:
        for x in (dn(v), v, up(v)):
            if k < nz:
                c.set_level(field, k, x)
            k += 1
    cols.append(c)

    c = C("edge-temperatures", "edge: t0c and triple point", seed=201, **base)
    for f, v in (("qs", 3.0e-4), ("qr", 3.0e-4), ("qg", 1.0e-4), ("qh", 2.0e-4),
                 ("qi", 1.0e-5), ("qc", 2.0e-4)):
        c.add(f, 0.0, 6000.0, v)
    t0c = 273.15
    ttp = float(F32(F32(273.15) + F32(0.01)))
    temps = [dn(t0c), t0c, up(t0c), dn(ttp), ttp, up(ttp), 233.15, dn(233.15), 223.15, 200.0]
    for k, tt in enumerate(temps):
        c.t[3 + 2 * k] = tt
    cols.append(c)

    c = C("edge-negative", "edge: negative and subnormal inputs", seed=202, **base)
    c.add("qr", 0.0, 3000.0, 2.0e-4).add("qs", 2000.0, 6000.0, 2.0e-4)
    c.add("qh", 1000.0, 5000.0, 2.0e-4)
    for k, (f, v) in enumerate([("qc", -1.0e-10), ("qr", -3.0e-12), ("qi", -1.0e-14),
                                ("qs", -2.0e-11), ("qg", -5.0e-13), ("qh", -4.0e-12),
                                ("qc", 1.0e-40), ("qr", 3.0e-41), ("qs", 1.0e-39),
                                ("qg", 7.0e-42), ("qh", 5.0e-41), ("qi", 2.0e-40),
                                ("qv", 1.0e-16), ("qv", -1.0e-9)]):
        c.set_level(f, 1 + 2 * k, v)
    cols.append(c)

    c = C("edge-top-only", "edge: condensate in the top level only", ts=280.0,
          rh=lambda z: np.full_like(z, 0.5), seed=203)
    for f, v in (("qr", 1.0e-4), ("qs", 2.0e-4), ("qg", 3.0e-4), ("qh", 4.0e-4),
                 ("qi", 5.0e-5), ("qc", 1.0e-5)):
        c.set_level(f, nz - 1, v)
    # negative zeros: WRF's MAX(q, 0.0) keeps -0.0 (gfortran's MAX_EXPR
    # returns its first operand on a tie), fmaxf would not
    for k, f in enumerate(("qc", "qr", "qi", "qs", "qg", "qh")):
        c.set_level(f, 3 + k, -0.0)
    cols.append(c)

    c = C("edge-top-two", "edge: hail and rain in the two top levels", ts=284.0,
          rh=lambda z: np.full_like(z, 0.6), seed=213)
    for f, v in (("qh", 1.0e-3), ("qr", 5.0e-4), ("qs", 3.0e-4)):
        c.set_level(f, nz - 1, v)
        c.set_level(f, nz - 2, 0.5 * v)
    cols.append(c)

    c = C("edge-bottom-only", "edge: condensate in the lowest level only", ts=276.0,
          rh=lambda z: np.full_like(z, 0.8), seed=204)
    for f, v in (("qr", 2.0e-3), ("qs", 1.0e-3), ("qg", 4.0e-3), ("qh", 6.0e-3),
                 ("qi", 1.0e-4), ("qc", 5.0e-4)):
        c.set_level(f, 0, v)
    c.rain, c.snow, c.graupel, c.hail = 123.456, 7.25, 3.0, 1.5
    cols.append(c)

    thin = np.full(nz, 5.0)
    thin[20:] = _interfaces(nz)[20:]
    c = C("edge-thin-layers", "edge: thin layers force the CFL limiter", ts=296.0,
          dz=thin, rh=lambda z: np.full_like(z, 0.9), seed=205)
    c.add("qr", 0.0, 3000.0, 8.0e-3).add("qg", 0.0, 3000.0, 1.0e-2)
    c.add("qs", 0.0, 3000.0, 2.0e-3).add("qh", 0.0, 3000.0, 1.0e-2)
    cols.append(c)

    c = C("edge-supersaturated-clear", "edge: supersaturated, no condensate",
          ts=290.0, rh=lambda z: np.full_like(z, 1.03), seed=206)
    cols.append(c)

    c = C("edge-subsaturated-cloud", "edge: cloud in dry air evaporates",
          ts=293.0, rh=lambda z: np.full_like(z, 0.4), seed=207)
    c.add("qc", 0.0, 9000.0, 1.0e-3).add("qi", 6000.0, 12000.0, 2.0e-4)
    c.add("qs", 3000.0, 12000.0, 4.0e-4).add("qr", 0.0, 6000.0, 6.0e-4)
    c.add("qg", 2000.0, 8000.0, 2.0e-4).add("qh", 1000.0, 7000.0, 3.0e-4)
    cols.append(c)

    c = C("edge-heavy-hail", "edge: very large hail, graupel and rain", ts=306.0,
          lapse=7.2e-3, ztrop=14000.0, rh=lambda z: np.where(z < 10000, 0.95, 0.5),
          seed=208)
    c.add("qh", 0.0, 12000.0, 3.0e-2).add("qg", 0.0, 12000.0, 2.5e-2)
    c.add("qr", 0.0, 6000.0, 2.0e-2).add("qc", 1000.0, 10000.0, 4.0e-3)
    c.add("qs", 6000.0, 13000.0, 4.0e-3).add("qi", 9000.0, 15000.0, 2.0e-3)
    cols.append(c)

    # wet growth: much cloud water and rain (riming beats wet growth) against
    # little (wet growth wins and phaci/phacs/phacg are zeroed)
    for n, (qcp, qrp) in enumerate([(4.0e-3, 6.0e-3), (5.0e-5, 1.0e-5)]):
        c = C(f"edge-wet-growth-{n}", "edge: hail wet-growth test both ways",
              ts=295.0, lapse=7.0e-3, rh=lambda z: np.full_like(z, 0.9), seed=214 + n)
        c.add("qh", 2000.0, 9000.0, 4.0e-3).add("qs", 3000.0, 9000.0, 1.5e-3)
        c.add("qg", 3000.0, 9000.0, 1.5e-3).add("qi", 5000.0, 9000.0, 3.0e-4)
        c.add("qc", 2000.0, 9000.0, qcp).add("qr", 2000.0, 9000.0, qrp)
        cols.append(c)

    c = C("edge-psaut", "edge: cloud ice above the aggregation limit", ts=270.0,
          rh=lambda z: np.full_like(z, 1.0), seed=209)
    c.add("qi", 1000.0, 9000.0, 3.0e-3)
    cols.append(c)

    c = C("edge-dry-tiny", "edge: near-zero vapour, qsum at qmin", ts=250.0,
          rh=lambda z: np.full_like(z, 1.0e-6), seed=210)
    for k in range(0, nz, 4):
        c.set_level("qs", k, 5.0e-16)
        c.set_level("qg", k, 5.0e-16)
        c.set_level("qh", k, 5.0e-16)
    c.set_level("qs", 1, 1.0e-15)
    c.set_level("qg", 2, 1.0e-15)
    c.set_level("qh", 3, 1.0e-15)
    cols.append(c)

    c = C("edge-cold-rain", "edge: rain far below freezing (supcol > 50)", ts=225.0,
          lapse=4.0e-3, rh=lambda z: np.full_like(z, 0.9), seed=211)
    c.add("qr", 0.0, 8000.0, 5.0e-4).add("qc", 0.0, 8000.0, 3.0e-4)
    c.add("qs", 0.0, 8000.0, 1.0e-4).add("qh", 0.0, 8000.0, 1.0e-4)
    cols.append(c)

    c = C("edge-zero", "edge: all hydrometeors zero, saturated", ts=280.0,
          rh=lambda z: np.full_like(z, 1.0), seed=212)
    cols.append(c)
    return cols


def build_set(out_dir: Path, stem: str, nz: int, keep=None) -> dict:
    cols = _columns(nz)
    if keep is not None:
        cols = [c for c in cols if keep(c)]
    data = [c.fields() for c in cols]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stem}.bin", "wb") as fh:
        np.asarray([len(cols), nz], "<i4").tofile(fh)
        for name in LEVEL_FIELDS:
            np.stack([d[name] for d in data], axis=1).astype("<f4").tofile(fh)
        for name in SURFACE_FIELDS:
            np.asarray([getattr(c, name) for c in cols], "<f4").tofile(fh)
    meta = {"ncol": len(cols), "nz": nz,
            "level_fields": list(LEVEL_FIELDS),
            "surface_fields": list(SURFACE_FIELDS),
            "columns": [{"name": c.name, "regime": c.regime} for c in cols]}
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def build(out_dir: Path) -> list[dict]:
    main = build_set(out_dir, "columns", 45)
    # The deep tier: every hail and edge regime again at 72 levels.
    deep = build_set(out_dir, "columns72", 72,
                     keep=lambda c: c.name.startswith(("hail-", "edge-", "stratiform-0",
                                                       "convective-0", "snow-0")))
    return [main, deep]


if __name__ == "__main__":
    for m in build(Path(sys.argv[1] if len(sys.argv) > 1 else ".")):
        print(f"{m['ncol']} columns x {m['nz']} levels")
