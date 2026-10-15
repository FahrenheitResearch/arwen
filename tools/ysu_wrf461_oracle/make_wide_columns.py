#!/usr/bin/env python3
"""Write the input columns of the wide YSU oracle (WRF v4.6.1 bl_ysu_run).

The narrow fixture next to this file (run_bl_ysu.F90, 24 columns, one grid,
one timestep) was built to reach branches.  This one adds what it lacks:
columns shaped like the regimes YSU actually runs in -- daytime convective
land, stable night land, ocean, high terrain with and without topo_wind,
snow and ice, stratocumulus under top-down mixing, tropical moist -- on
several vertical grids and timesteps, plus every edge probe of the narrow
fixture re-expressed here.  The columns are synthetic: each regime is a
seeded random draw around a hand-shaped profile, so the set is reproducible
from this file alone and needs no model output.

Both sides read the SAME bits.  ``run_bl_ysu_wide.F90`` reads this file and
hands bl_ysu_run exactly what module_bl_ysu.F hands it in a WRF run; the
port harness (gpuwm.verify.ysu_wide_oracle) reads the same file and hands
launch_ysu the model-side fields.  The one place the two sides take
different inputs is temperature, and that is WRF's own chain: the model owns
potential temperature ``theta`` and Exner ``pi``; phy_prep forms
``t_phy = th_phy*pi_phy`` (a float32 multiply), and that ``tx`` is what
bl_ysu_run receives.  This file stores theta, pi and the float32 product, so
the Fortran gets ``tx`` and the port gets ``theta``/``pi`` and forms the
product itself.

Record layout (little endian, one record per column, in order):

    int32[4]       nz, ysu_topdown_pblmix, regime id, topo flag
    float32[18]    dt, psfc, znt, ust, hfx, qfx, wspd, br, psim, psih,
                   xland, u10, v10, ctopo, ctopo2, uox, vox, (pad)
    float32[nz]    u, v, theta, pi, tx, qv, qc, qi, p, dz, rthraten
    float32[nz+1]  p_interface

``topo flag`` 0 means WRF's default (ctopo = ctopo2 = 1, the arm every
default WRF run takes and ysu_column ports); 1 means topo_wind 1/2
coefficients (ysu_column_topo).  uox/vox are WRF's ocean currents, zero
unless a coupled ocean model runs; module_bl_ysu.F passes them always.

    python tools/ysu_wrf461_oracle/make_wide_columns.py OUT.bin
    python tools/ysu_wrf461_oracle/make_wide_columns.py --sweep N SEED OUT.bin

``--sweep`` draws N columns from the regime builders alone (no edge
probes) with its own seed: the large randomized sweep that backs the
committed fixture, reproducible from (N, SEED) and not committed.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

F32 = np.float32
SEED = 20261007

#: regime id -> name.  The id is stored per column so every per-regime
#: number in the report can be recomputed from the fixture alone.
REGIMES = {
    1: "convective-land",
    2: "stable-night-land",
    3: "ocean",
    4: "high-terrain",
    5: "high-terrain-topo-wind",
    6: "snow-ice-cold",
    7: "stratocumulus-topdown",
    8: "tropical-moist",
    9: "edge-probe",
}

G, RD, RV = 9.81, 287.0, 461.6
CP = 7.0 * RD / 2.0
ROVCP = F32(F32(RD) / F32(CP))
SMALLEST_SUB = np.float32(np.finfo(np.float32).smallest_subnormal)
SMALLEST_NORM = np.float32(np.finfo(np.float32).smallest_normal)
NEG_ZERO = np.float32(-0.0)


def stretched_dz(nz: int, dz1: float, growth: float, cap: float) -> np.ndarray:
    dz = np.empty(nz)
    dz[0] = dz1
    for k in range(1, nz):
        dz[k] = min(dz[k - 1] * growth, cap)
    return dz


class Column:
    """One column, built in float64 and frozen to float32 on write."""

    def __init__(self, regime: int, nz: int, dz: np.ndarray, dt: float):
        self.regime = regime
        self.nz = nz
        self.dt = dt
        self.topdown = 1
        self.topo = 0
        self.dz = np.asarray(dz, dtype=np.float64)
        zq = np.concatenate([[0.0], np.cumsum(self.dz)])
        self.zq = zq
        self.za = 0.5 * (zq[:-1] + zq[1:])
        self.u = np.zeros(nz)
        self.v = np.zeros(nz)
        self.theta = np.zeros(nz)
        self.qv = np.zeros(nz)
        self.qc = np.zeros(nz)
        self.qi = np.zeros(nz)
        self.rthraten = np.zeros(nz)
        self.psfc = 100000.0
        self.znt = 0.1
        self.ust = 0.3
        self.hfx = 0.0
        self.qfx = 0.0
        self.br = 0.0
        self.psim = 6.0
        self.psih = 8.0
        self.xland = 1.0
        self.u10 = None
        self.v10 = None
        self.wspd = None
        self.ctopo = 1.0
        self.ctopo2 = 1.0
        # float32 overrides applied verbatim after the build (edge probes)
        self.raw: dict[str, object] = {}
        self.raw_levels: dict[tuple[str, int], np.float32] = {}

    # ---- thermodynamics -------------------------------------------------
    def finish(self) -> dict:
        nz = self.nz
        # Hydrostatic interface pressure from the virtual temperature,
        # iterated twice so p and T agree to well below float32.
        p_i = np.empty(nz + 1)
        p_i[0] = self.psfc
        pmid = np.empty(nz)
        for _ in range(2):
            for k in range(nz):
                guess = p_i[k] if _ == 0 else pmid[k]
                tk = self.theta[k] * (guess / 1.0e5) ** (RD / CP)
                tv = tk * (1.0 + 0.608 * max(self.qv[k], 0.0))
                p_i[k + 1] = p_i[k] * np.exp(-G * self.dz[k] / (RD * tv))
                pmid[k] = np.sqrt(p_i[k] * p_i[k + 1])
        rec = {
            "u": self.u.astype(F32), "v": self.v.astype(F32),
            "theta": self.theta.astype(F32), "qv": self.qv.astype(F32),
            "qc": self.qc.astype(F32), "qi": self.qi.astype(F32),
            "p": pmid.astype(F32), "dz": self.dz.astype(F32),
            "rthraten": self.rthraten.astype(F32),
            "p_interface": p_i.astype(F32),
        }
        # Exner as the model would hold it; the exact formula does not matter
        # to the comparison, only that both sides read the same word.
        rec["pi"] = ((rec["p"].astype(np.float64) / 1.0e5)
                     ** (RD / CP)).astype(F32)
        for (name, k), value in self.raw_levels.items():
            rec[name][k] = value
        # phy_prep: t_phy = th_phy*pi_phy, one float32 multiply.
        rec["tx"] = (rec["theta"] * rec["pi"]).astype(F32)
        u1, v1 = float(rec["u"][0]), float(rec["v"][0])
        if self.u10 is None:
            self.u10 = 0.85 * u1
            self.v10 = 0.85 * v1
        if self.wspd is None:
            self.wspd = max(np.hypot(u1, v1), 0.1)
        scal = {
            "dt": self.dt, "psfc": self.psfc, "znt": self.znt,
            "ust": self.ust, "hfx": self.hfx, "qfx": self.qfx,
            "wspd": self.wspd, "br": self.br, "psim": self.psim,
            "psih": self.psih, "xland": self.xland, "u10": self.u10,
            "v10": self.v10, "ctopo": self.ctopo, "ctopo2": self.ctopo2,
            "uox": 0.0, "vox": 0.0,
        }
        scal = {k: F32(v) for k, v in scal.items()}
        for k, v in self.raw.items():
            scal[k] = F32(v)
        rec["scalars"] = scal
        rec["nz"] = nz
        rec["topdown"] = self.topdown
        rec["regime"] = self.regime
        rec["topo"] = self.topo
        return rec


SCALAR_ORDER = ("dt", "psfc", "znt", "ust", "hfx", "qfx", "wspd", "br",
                "psim", "psih", "xland", "u10", "v10", "ctopo", "ctopo2",
                "uox", "vox")
LEVEL_ORDER = ("u", "v", "theta", "pi", "tx", "qv", "qc", "qi", "p", "dz",
               "rthraten")


def write_records(records: list[dict], path: Path) -> None:
    with path.open("wb") as fh:
        for rec in records:
            fh.write(struct.pack("<4i", rec["nz"], rec["topdown"],
                                 rec["regime"], rec["topo"]))
            vals = [rec["scalars"][k] for k in SCALAR_ORDER] + [F32(rec["scalars"].get("_pad", 0.0))]
            fh.write(np.asarray(vals, dtype="<f4").tobytes())
            for name in LEVEL_ORDER:
                arr = np.asarray(rec[name], dtype="<f4")
                assert arr.shape == (rec["nz"],), (name, arr.shape)
                fh.write(arr.tobytes())
            pin = np.asarray(rec["p_interface"], dtype="<f4")
            assert pin.shape == (rec["nz"] + 1,)
            fh.write(pin.tobytes())


# --------------------------------------------------------------------------
# regime builders
# --------------------------------------------------------------------------

GRIDS = (
    # (nz, dz1, growth, cap) -- WRF-like stretched grids
    (40, 25.0, 1.08, 400.0),
    (50, 16.0, 1.07, 350.0),
    (60, 10.0, 1.06, 300.0),
    (30, 40.0, 1.10, 600.0),
    (45, 8.0, 1.09, 450.0),
)
TIMESTEPS = (10.0, 20.0, 30.0, 45.0, 60.0)


def _grid(rng):
    nz, dz1, growth, cap = GRIDS[rng.integers(len(GRIDS))]
    dz1 = dz1 * rng.uniform(0.8, 1.25)
    return nz, stretched_dz(nz, dz1, growth, cap)


def _winds(c: Column, rng, sfc, shear, veer, jet=None):
    """Log-ish near-surface profile turning with height."""
    z = c.za
    speed = sfc + shear * z / 1000.0
    if jet is not None:
        zj, amp = jet
        speed = speed + amp * np.exp(-((z - zj) / (0.4 * zj)) ** 2)
    speed = speed * np.clip(np.log1p(z / max(c.znt, 1e-4))
                            / np.log1p(z[-1] / max(c.znt, 1e-4)) * 3.0,
                            0.25, 1.0)
    angle = rng.uniform(0, 2 * np.pi) + veer * z / 1000.0
    c.u = speed * np.cos(angle)
    c.v = speed * np.sin(angle)


def _mixed_layer(c, th_sfc, zi, jump, lapse, superadiab=0.0):
    z = c.za
    th = np.where(z <= zi, th_sfc, th_sfc + jump + lapse * (z - zi))
    # a thin superadiabatic surface layer
    th = th + superadiab * np.exp(-z / 40.0)
    c.theta = th


def _moisture(c, qv0, zi, scale_above):
    z = c.za
    c.qv = np.where(z <= zi, qv0 * (1.0 - 0.1 * z / max(zi, 1.0)),
                    0.9 * qv0 * np.exp(-(z - zi) / scale_above))


def convective_land(rng):
    nz, dz = _grid(rng)
    c = Column(1, nz, dz, float(rng.choice(TIMESTEPS)))
    zi = rng.uniform(600, 3200)
    _mixed_layer(c, rng.uniform(296, 312), zi, rng.uniform(0.5, 4.5),
                 rng.uniform(0.003, 0.007), superadiab=rng.uniform(0.3, 2.5))
    _moisture(c, rng.uniform(0.004, 0.016), zi, 2200.0)
    c.znt = rng.uniform(0.02, 0.9)
    _winds(c, rng, rng.uniform(1.0, 9.0), rng.uniform(0.5, 5.0),
           rng.uniform(-0.3, 0.3))
    c.hfx = rng.uniform(40, 560)
    c.qfx = rng.uniform(1e-5, 3e-4)
    c.ust = rng.uniform(0.12, 0.85)
    c.br = -rng.uniform(0.02, 3.0)
    c.psim = rng.uniform(2.0, 6.5)
    c.psih = c.psim + rng.uniform(0.5, 3.0)
    c.topdown = int(rng.random() < 0.8)
    c.wspd = max(np.hypot(c.u[0], c.v[0]), 0.1) + rng.uniform(0.0, 1.5)
    return c


def stable_night(rng):
    nz, dz = _grid(rng)
    c = Column(2, nz, dz, float(rng.choice(TIMESTEPS)))
    z = c.za
    hinv = rng.uniform(80, 450)
    strength = rng.uniform(1.5, 15.0)
    th0 = rng.uniform(278, 300)
    c.theta = (th0 + strength * np.minimum(z / hinv, 1.0)
               + 0.004 * np.maximum(z - hinv, 0.0))
    _moisture(c, rng.uniform(0.002, 0.012), hinv, 1800.0)
    c.znt = rng.uniform(0.005, 0.5)
    _winds(c, rng, rng.uniform(0.3, 4.0), rng.uniform(0.5, 3.0),
           rng.uniform(-0.5, 0.5), jet=(hinv * rng.uniform(0.8, 1.5),
                                        rng.uniform(2.0, 12.0)))
    c.hfx = -rng.uniform(1.0, 110.0)
    c.qfx = rng.choice([0.0, -rng.uniform(0, 5e-6), rng.uniform(0, 5e-6)])
    c.ust = rng.uniform(0.015, 0.4)
    c.br = rng.uniform(0.01, 3.0)
    c.psim = rng.uniform(4.0, 20.0)
    c.psih = c.psim * rng.uniform(1.0, 1.6)
    c.topdown = int(rng.random() < 0.7)
    return c


def ocean(rng):
    nz, dz = _grid(rng)
    c = Column(3, nz, dz, float(rng.choice(TIMESTEPS)))
    c.xland = 2.0
    unstable = rng.random() < 0.55
    zi = rng.uniform(300, 1400)
    th0 = rng.uniform(285, 302)
    if unstable:
        _mixed_layer(c, th0, zi, rng.uniform(0.5, 6.0), 0.005,
                     superadiab=rng.uniform(0.1, 0.8))
        c.hfx = rng.uniform(2.0, 90.0)
        c.qfx = rng.uniform(2e-5, 2e-4)
        c.br = -rng.uniform(0.005, 0.8)
    else:
        z = c.za
        c.theta = th0 + rng.uniform(0.5, 4.0) * np.minimum(z / 150.0, 1.0) \
            + 0.005 * np.maximum(z - 150.0, 0.0)
        c.hfx = -rng.uniform(0.0, 30.0)
        c.qfx = rng.uniform(-2e-6, 4e-5)
        c.br = rng.uniform(0.002, 1.5)
    _moisture(c, rng.uniform(0.006, 0.018), zi, 2000.0)
    c.znt = 10 ** rng.uniform(-5.0, -2.7)
    _winds(c, rng, rng.uniform(1.0, 14.0), rng.uniform(0.2, 3.0),
           rng.uniform(-0.2, 0.2))
    c.ust = rng.uniform(0.03, 0.6)
    c.psim = rng.uniform(5.0, 14.0)
    c.psih = c.psim + rng.uniform(0.0, 4.0)
    sp = np.hypot(c.u[0], c.v[0])
    c.u10 = c.u[0] * rng.uniform(0.85, 0.98)
    c.v10 = c.v[0] * rng.uniform(0.85, 0.98)
    c.wspd = max(sp, 0.1)
    c.topdown = int(rng.random() < 0.8)
    return c


def high_terrain(rng, topo: bool):
    nz, dz = _grid(rng)
    c = Column(5 if topo else 4, nz, dz, float(rng.choice(TIMESTEPS)))
    c.psfc = rng.uniform(58000.0, 80000.0)
    convective = rng.random() < 0.5
    th0 = rng.uniform(298, 322)
    if convective:
        zi = rng.uniform(800, 3500)
        _mixed_layer(c, th0, zi, rng.uniform(0.5, 3.0),
                     rng.uniform(0.004, 0.008), superadiab=rng.uniform(0.5, 3))
        c.hfx = rng.uniform(80, 600)
        c.qfx = rng.uniform(5e-6, 1.2e-4)
        c.br = -rng.uniform(0.05, 4.0)
        c.ust = rng.uniform(0.2, 0.9)
        c.psim = rng.uniform(2.0, 5.0)
    else:
        z = c.za
        zi = rng.uniform(60, 300)
        c.theta = th0 + rng.uniform(3, 18) * np.minimum(z / zi, 1.0) \
            + 0.0045 * np.maximum(z - zi, 0)
        c.hfx = -rng.uniform(5, 120)
        c.qfx = rng.uniform(-3e-6, 3e-6)
        c.br = rng.uniform(0.05, 4.0)
        c.ust = rng.uniform(0.03, 0.5)
        c.psim = rng.uniform(5.0, 18.0)
    c.psih = c.psim + rng.uniform(0.2, 3.0)
    _moisture(c, rng.uniform(0.001, 0.007), zi, 1500.0)
    c.znt = rng.uniform(0.1, 1.5)
    _winds(c, rng, rng.uniform(2.0, 16.0), rng.uniform(1.0, 8.0),
           rng.uniform(-0.4, 0.4))
    c.topdown = int(rng.random() < 0.75)
    if topo:
        c.topo = 1
        # terrain_drag.topo_wind_coefficients range: ctopo scales the drag
        # (above 1 in valleys, below on ridges), ctopo2 blends u10.
        c.ctopo = rng.uniform(0.2, 2.6)
        c.ctopo2 = rng.uniform(0.0, 1.0)
    return c


def snow_ice(rng):
    nz, dz = _grid(rng)
    c = Column(6, nz, dz, float(rng.choice(TIMESTEPS)))
    z = c.za
    th0 = rng.uniform(240.0, 272.0)
    c.psfc = rng.uniform(85000.0, 103000.0)
    if rng.random() < 0.7:          # stable over snow
        hinv = rng.uniform(50, 400)
        c.theta = th0 + rng.uniform(3, 20) * np.minimum(z / hinv, 1.0) \
            + 0.005 * np.maximum(z - hinv, 0)
        c.hfx = -rng.uniform(2, 70)
        c.qfx = rng.uniform(-4e-6, 1e-6)
        c.br = rng.uniform(0.02, 5.0)
        c.ust = rng.uniform(0.02, 0.35)
        c.psim = rng.uniform(6.0, 22.0)
        zi = hinv
    else:                           # cold-air outbreak, weakly convective
        zi = rng.uniform(300, 1500)
        _mixed_layer(c, th0, zi, rng.uniform(1, 5), 0.006,
                     superadiab=rng.uniform(0.2, 1.5))
        c.hfx = rng.uniform(5, 200)
        c.qfx = rng.uniform(2e-6, 5e-5)
        c.br = -rng.uniform(0.01, 1.0)
        c.ust = rng.uniform(0.1, 0.6)
        c.psim = rng.uniform(3.0, 8.0)
    c.psih = c.psim + rng.uniform(0.5, 4.0)
    _moisture(c, rng.uniform(0.0002, 0.0028), zi, 1500.0)
    c.znt = rng.uniform(0.0005, 0.05)          # snow roughness
    _winds(c, rng, rng.uniform(0.5, 10.0), rng.uniform(0.5, 4.0),
           rng.uniform(-0.3, 0.3))
    # an ice or mixed-phase layer, sometimes at the PBL top with cooling
    k_lo = int(np.searchsorted(c.za, zi * rng.uniform(0.6, 1.0)))
    k_hi = min(nz - 3, k_lo + int(rng.integers(1, 5)))
    k_lo = max(1, min(k_lo, nz - 4))
    c.qi[k_lo:k_hi + 1] = rng.uniform(1e-6, 2.5e-4)
    if rng.random() < 0.5:
        c.qc[k_lo:k_hi + 1] = rng.uniform(0.0, 3e-4)
    c.rthraten[k_lo:k_hi + 1] = -rng.uniform(0.0, 1.2e-4)
    c.topdown = int(rng.random() < 0.8)
    return c


def stratocumulus(rng):
    nz, dz = _grid(rng)
    c = Column(7, nz, dz, float(rng.choice(TIMESTEPS)))
    zi = rng.uniform(400, 1600)
    _mixed_layer(c, rng.uniform(283, 298), zi, rng.uniform(3.0, 12.0),
                 0.005, superadiab=rng.uniform(0.0, 0.6))
    _moisture(c, rng.uniform(0.006, 0.012), zi, 1200.0)
    k_top = int(np.searchsorted(c.za, zi)) - 1
    k_top = max(3, min(k_top, nz - 4))
    depth = int(rng.integers(2, 6))
    k_lo = max(1, k_top - depth)
    c.qc[k_lo:k_top + 1] = np.linspace(rng.uniform(5e-5, 2e-4),
                                        rng.uniform(2e-4, 9e-4),
                                        k_top + 1 - k_lo)
    if rng.random() < 0.3:
        c.qi[k_lo:k_top + 1] = rng.uniform(1e-6, 5e-5)
    c.rthraten[k_top - 1:k_top + 1] = -rng.uniform(2e-5, 2.5e-4)
    c.xland = float(rng.choice([1.0, 2.0]))
    c.znt = 2e-4 if c.xland == 2.0 else rng.uniform(0.01, 0.4)
    _winds(c, rng, rng.uniform(2.0, 10.0), rng.uniform(0.2, 2.0),
           rng.uniform(-0.2, 0.2))
    c.hfx = rng.uniform(-10, 60)
    c.qfx = rng.uniform(1e-6, 8e-5)
    c.ust = rng.uniform(0.1, 0.5)
    c.br = -rng.uniform(0.0, 0.3) if c.hfx > 0 else rng.uniform(0.0, 0.3)
    c.psim = rng.uniform(4.0, 10.0)
    c.psih = c.psim + rng.uniform(0.0, 3.0)
    c.topdown = int(rng.random() < 0.85)
    return c


def tropical(rng):
    nz, dz = _grid(rng)
    c = Column(8, nz, dz, float(rng.choice(TIMESTEPS)))
    zi = rng.uniform(400, 1200)
    _mixed_layer(c, rng.uniform(298, 303), zi, rng.uniform(0.3, 1.5), 0.0035,
                 superadiab=rng.uniform(0.1, 0.8))
    _moisture(c, rng.uniform(0.016, 0.022), zi, 2600.0)
    if rng.random() < 0.5:
        k = max(2, int(np.searchsorted(c.za, zi)))
        c.qc[k:k + 3] = rng.uniform(1e-4, 1e-3)
    c.xland = float(rng.choice([1.0, 2.0]))
    c.znt = 1e-4 if c.xland == 2.0 else rng.uniform(0.2, 1.5)
    _winds(c, rng, rng.uniform(0.5, 7.0), rng.uniform(0.2, 3.0),
           rng.uniform(-0.2, 0.2))
    c.hfx = rng.uniform(5, 150)
    c.qfx = rng.uniform(5e-5, 2.5e-4)
    c.ust = rng.uniform(0.08, 0.5)
    c.br = -rng.uniform(0.0, 1.2)
    c.psim = rng.uniform(3.0, 9.0)
    c.psih = c.psim + rng.uniform(0.0, 2.0)
    c.topdown = int(rng.random() < 0.8)
    return c


# --------------------------------------------------------------------------
# edge probes (the narrow fixture's, re-expressed, plus new ones)
# --------------------------------------------------------------------------

def _edge_base(nz=40, dz1=25.0, grow=20.0):
    dz = dz1 + grow * np.arange(nz)
    c = Column(9, nz, dz, 45.0)
    z = c.za
    _mixed_layer(c, 300.0, 900.0, 1.5, 0.006)
    c.qv = 0.012 * np.exp(-z / 2500.0)
    c.u = 6.0 + 0.0015 * z
    c.v = -1.5 + 0.0008 * z
    c.hfx, c.qfx, c.ust, c.br = 0.5, 1e-7, 0.3, 0.0
    c.psim, c.psih = 6.5, 8.5
    return c


def edge_probes() -> list[Column]:
    out = []

    def add(**kw):
        c = _edge_base(**{k: kw.pop(k) for k in ("nz", "dz1", "grow")
                          if k in kw})
        levels = kw.pop("levels", {})
        build = kw.pop("build", None)
        if build:
            build(c)
        for k, v in kw.items():
            if k in SCALAR_ORDER:
                c.raw[k] = v
            else:
                setattr(c, k, v)
        for key, v in levels.items():
            c.raw_levels[key] = np.float32(v)
        out.append(c)

    # br sign probes on a near-neutral column
    for br in (0.0, NEG_ZERO, SMALLEST_SUB, -SMALLEST_SUB, SMALLEST_NORM,
               -SMALLEST_NORM):
        add(br=br)
    # zero surface coupling: stable br (WRF computes a PBL), unstable br
    # (WRF divides 0 by 0), ust subnormal, ust zero with heating
    add(ust=0.0, hfx=0.0, qfx=0.0, br=0.02)
    add(ust=0.0, hfx=0.0, qfx=0.0, br=-0.02)
    add(ust=SMALLEST_SUB, hfx=0.0, qfx=NEG_ZERO, br=-0.02)
    add(ust=SMALLEST_SUB, hfx=0.0, qfx=0.0, br=0.05)
    add(ust=0.0, hfx=120.0, qfx=5e-5, br=-0.3)
    add(ust=1.0e-13, hfx=30.0, qfx=0.0, br=-0.1)      # ust**3 underflows
    add(ust=5.0e-4, hfx=0.0, qfx=0.0, br=0.5)
    # wspd at the sfclay floor, at 1e-10, and huge
    add(wspd=0.1, build=lambda c: (setattr(c, "u", c.u * 0.0 + 0.05),
                                   setattr(c, "v", c.v * 0.0)))
    add(wspd=1.0e-10)
    add(wspd=60.0, ust=2.5, hfx=150.0, br=-0.05)
    # signed zeros and subnormals in the moisture and wind fields
    add(hfx=20.0, qfx=SMALLEST_SUB, ust=0.18, br=-0.01,
        levels={("qv", 0): SMALLEST_SUB, ("qv", 1): NEG_ZERO,
                ("qv", 2): SMALLEST_NORM, ("qc", 3): SMALLEST_SUB,
                ("qi", 4): -SMALLEST_SUB, ("qc", 5): NEG_ZERO,
                ("u", 0): NEG_ZERO, ("v", 0): NEG_ZERO})
    # negative moisture (advection undershoot) aloft
    add(hfx=80.0, qfx=6e-5, ust=0.4, br=-0.2,
        build=lambda c: c.qv.__setitem__(slice(30, 40), -1e-7))
    # cloud test straddle at 0.01e-3
    def cloud(qc):
        def b(c):
            c.qc[7:12] = qc
            c.rthraten[7:12] = -5e-5
        return b
    add(hfx=80.0, qfx=5e-5, ust=0.35, br=-0.1, build=cloud(np.float32(1e-5)))
    add(hfx=80.0, qfx=5e-5, ust=0.35, br=-0.1,
        build=cloud(np.nextafter(np.float32(1e-5), np.float32(1))))
    add(hfx=80.0, qfx=5e-5, ust=0.35, br=-0.1,
        build=cloud(np.nextafter(np.float32(1e-5), np.float32(0))))
    # gamcrt / gamcrq saturation, very large flux
    add(hfx=900.0, qfx=1.2e-3, ust=1.1, br=-4.0,
        build=lambda c: _mixed_layer(c, 305.0, 2500.0, 5.0, 0.006))
    # PBL fills the column
    add(hfx=300.0, qfx=2e-4, ust=0.7, br=-0.9,
        build=lambda c: setattr(c, "theta", np.full(c.nz, 300.0)))
    # coarse 500 m grid (rlamdz cap) and a 2 m first layer
    add(dz1=500.0, grow=0.0, hfx=120.0, qfx=6e-5, ust=0.4, br=-0.2)
    add(dz1=2.0, grow=4.0, hfx=60.0, qfx=3e-5, ust=0.3, br=-0.1)
    # strong shear, unstable free-atmosphere Ri
    add(hfx=60.0, qfx=3e-5, ust=0.6, br=-0.04,
        build=lambda c: (setattr(c, "u", 2.0 + 0.01 * c.za),
                         setattr(c, "v", 0.004 * c.za),
                         _mixed_layer(c, 300.0, 600.0, 0.2, 0.0008)))
    # very stable free atmosphere (prmax clamp), topdown off
    add(hfx=-15.0, qfx=0.0, ust=0.1, br=0.8, topdown=0,
        build=lambda c: _mixed_layer(c, 300.0, 400.0, 1.0, 0.03))
    # minimum vertical extent the launcher accepts, and the KMAX-ish top
    add(nz=4, dz1=40.0, grow=40.0, hfx=50.0, qfx=2e-5, ust=0.3, br=-0.1)
    add(nz=120, dz1=10.0, grow=2.0, hfx=200.0, qfx=1e-4, ust=0.5, br=-0.6)
    # ocean with a subnormal / signed-zero 10 m wind (Rossby brcrit form)
    add(xland=2.0, znt=1e-4, hfx=-8.0, qfx=3e-6, ust=0.05, br=0.6,
        u10=SMALLEST_SUB, v10=NEG_ZERO,
        build=lambda c: _mixed_layer(c, 300.0, 200.0, 0.5, 0.01))
    # timestep extremes
    add(dt=1.0, hfx=150.0, qfx=8e-5, ust=0.4, br=-0.3)
    add(dt=300.0, hfx=150.0, qfx=8e-5, ust=0.4, br=-0.3)
    # wscale**4 below the old 1e-20 floor while WRF stays finite
    add(ust=1.0e-7, hfx=50.0, qfx=2e-5, br=-0.1)
    # zero coupling under a cloud-topped layer, stable and unstable surface
    def sc_top(c):
        c.qc[8:12] = 4e-4
        c.rthraten[10:12] = -1e-4
    add(ust=0.0, hfx=0.0, qfx=0.0, br=0.05, build=sc_top)
    add(ust=0.0, hfx=0.0, qfx=0.0, br=-0.05, build=sc_top)
    # weak wind under a stable surface: get_pblh's hybrid height below
    # 1167 m keeps vconvlim < 1, so the tanh weight reaches the drag
    for zi_scale in (0.55, 0.8, 1.1, 1.4):
        add(hfx=-20.0, qfx=0.0, ust=0.12, br=0.4,
            build=lambda c, s=zi_scale: _mixed_layer(
                c, 290.0, 300.0 * s, 4.0, 0.006))
    return out


def build_columns() -> list[Column]:
    rng = np.random.default_rng(SEED)
    cols: list[Column] = []
    cols += [convective_land(rng) for _ in range(16)]
    cols += [stable_night(rng) for _ in range(14)]
    cols += [ocean(rng) for _ in range(14)]
    cols += [high_terrain(rng, False) for _ in range(8)]
    cols += [high_terrain(rng, True) for _ in range(10)]
    cols += [snow_ice(rng) for _ in range(12)]
    cols += [stratocumulus(rng) for _ in range(10)]
    cols += [tropical(rng) for _ in range(8)]
    cols += edge_probes()
    return cols


def sweep_columns(n: int, seed: int) -> list[Column]:
    rng = np.random.default_rng(seed)
    builders = (convective_land, stable_night, ocean,
                lambda r: high_terrain(r, False),
                lambda r: high_terrain(r, True), snow_ice, stratocumulus,
                tropical)
    return [builders[int(rng.integers(len(builders)))](rng)
            for _ in range(n)]


def main(argv: list[str]) -> int:
    if len(argv) == 5 and argv[1] == "--sweep":
        out = Path(argv[4])
        records = [c.finish() for c in sweep_columns(int(argv[2]),
                                                     int(argv[3]))]
        write_records(records, out)
        print(f"sweep: {len(records)} columns, seed {argv[3]}")
        return 0
    if len(argv) != 2:
        print(__doc__)
        return 2
    out = Path(argv[1])
    records = [c.finish() for c in build_columns()]
    write_records(records, out)
    summary = {
        "seed": SEED,
        "ncol": len(records),
        "regimes": REGIMES,
        "per_regime": {REGIMES[r]: sum(1 for x in records if x["regime"] == r)
                       for r in REGIMES},
        "nz": sorted({x["nz"] for x in records}),
        "dt": sorted({float(x["scalars"]["dt"]) for x in records}),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n",
                                        encoding="ascii")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
