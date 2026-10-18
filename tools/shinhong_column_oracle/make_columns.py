#!/usr/bin/env python3
"""Write the input columns of the wide Shin-Hong column oracle (WRF v4.6.1).

The branch fixture in tools/shinhong_wrf461_oracle (30 cases x 6 dx) was
built to reach branches with hand-made profiles.  This one adds columns
shaped like the regimes bl_pbl_physics=11 actually runs in: daytime
convective land (dry, moist with cumulus, desert), stable night land (weak
and radiative), snow and ice (arctic mixed-phase, ice fog, spring snowmelt),
ocean (tropical trades, stratocumulus, cold-air outbreak, warm-advection fog),
high terrain (day, night, snow), windy near-neutral overcast, and the
southern hemisphere and the equator, plus edge probes.  Every base column is
run at six grid spacings, because Shin-Hong's answer moves with
sqrt(dx*dy)/h, and once more with the TKE diagnostic off.

The columns are synthetic but physically built: hydrostatic pressure from
the virtual temperature, moisture from relative humidity over water or ice,
and a Monin-Obukhov surface layer that makes ust, br, psim (WRF's fm) and
psih (fhh) and u10/v10 consistent with the column's own wind, roughness and
heat flux.  Each family is a seeded random draw, so the set is reproducible
from this file.

Both sides read the SAME bits.  The model owns potential temperature theta
and Exner pi; WRF's phy_prep forms t_phy = th_phy*pi_phy (one float32
multiply) and module_pbl_driver hands that t_phy to shinhong as t3d.  This
file stores theta, pi and that float32 product, so the Fortran driver gets
t3d = theta*pi exactly as a WRF run would, and the WOOF kernel gets theta
and exner as the engine hands them.  psfc is p_interface(1), exactly as the
driver sets PSFC = p8w(kms).

Record layout, little endian, one record per column:

    int32[4]       nz, tke_diag, family id, base column id
    float32[16]    dt, dx, dy, psfc, znt, ust, hfx, qfx, wspd, br, psim,
                   psih, xland, u10, v10, corf
    float32[nz]    u, v, theta, pi, t, qv, qc, qi, p, dz, tke
    float32[nz+1]  p_interface

    python tools/shinhong_column_oracle/make_columns.py OUT.bin [OUT.json]
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

F32 = np.float32
SEED = 20261007

FAMILIES = {
    1: "convective-land-dry",
    2: "convective-land-moist-cumulus",
    3: "convective-desert",
    4: "morning-transition",
    5: "stable-night-land-weak",
    6: "stable-night-land-radiative",
    7: "snow-arctic-mixed-phase",
    8: "snow-ice-fog",
    9: "snow-spring-daytime",
    10: "ocean-tropical-trades",
    11: "ocean-stratocumulus",
    12: "ocean-cold-air-outbreak",
    13: "ocean-warm-advection-fog",
    14: "high-terrain-day",
    15: "high-terrain-night",
    16: "high-terrain-snow",
    17: "windy-neutral-overcast",
    18: "edge-probe",
}

#: grid spacing sweep (m) and the timestep a WRF user would run at each
#: (about 6 s per km, the PBL called every step).
DX_SWEEP = ((100.0, 1.0), (333.0, 2.0), (1000.0, 6.0), (3000.0, 18.0),
            (9000.0, 54.0), (27000.0, 150.0))
#: index into DX_SWEEP of the extra tke_diag = 0 run of every base column.
TKE_OFF_DX = 3

G, RD, RV = 9.81, 287.0, 461.6
CP = 7.0 * RD / 2.0
KARMAN = 0.4
OMEGA = 7.292e-5
SUB = np.float32(np.finfo(np.float32).smallest_subnormal)
MINNORM = np.float32(np.finfo(np.float32).tiny)
NZERO = np.float32(-0.0)

SCALAR_ORDER = ("dt", "dx", "dy", "psfc", "znt", "ust", "hfx", "qfx",
                "wspd", "br", "psim", "psih", "xland", "u10", "v10", "corf")
LEVEL_ORDER = ("u", "v", "theta", "pi", "t", "qv", "qc", "qi", "p", "dz",
               "tke")

GRIDS = (
    # (nz, dz1, growth, cap): WRF-like stretched grids
    (40, 30.0, 1.09, 700.0),
    (50, 20.0, 1.08, 600.0),
    (60, 12.0, 1.07, 500.0),
    (35, 50.0, 1.10, 900.0),
    (45, 8.0, 1.11, 600.0),
)


def stretched_dz(nz, dz1, growth, cap):
    dz = np.empty(nz)
    dz[0] = dz1
    for k in range(1, nz):
        dz[k] = min(dz[k - 1] * growth, cap)
    return dz


def qsat(t, p, ice):
    tc = t - 273.15
    es_w = 611.2 * np.exp(17.67 * tc / (t - 29.65))
    es_i = 611.2 * np.exp(21.8745584 * (t - 273.16) / (t - 7.66))
    es = np.where(ice & (t < 273.15), es_i, es_w)
    es = np.minimum(es, 0.5 * p)
    return 0.622 * es / (p - es)


def psi_unstable(zeta):
    x = (1.0 - 16.0 * zeta) ** 0.25
    psim = (2.0 * np.log((1 + x) / 2) + np.log((1 + x * x) / 2)
            - 2.0 * np.arctan(x) + np.pi / 2)
    psih = 2.0 * np.log((1 + x * x) / 2)
    return psim, psih


class Column:
    """One base column, built in float64 and frozen to float32 on write."""

    def __init__(self, family, rng, grid=None):
        self.family = family
        nz, dz1, growth, cap = GRIDS[rng.integers(len(GRIDS))] if grid is None else grid
        dz1 = dz1 * rng.uniform(0.85, 1.2)
        self.nz = nz
        self.dz = stretched_dz(nz, dz1, growth, cap)
        self.zq = np.concatenate([[0.0], np.cumsum(self.dz)])
        self.za = 0.5 * (self.zq[:-1] + self.zq[1:])
        self.theta = np.full(nz, 300.0)
        self.rh = np.full(nz, 0.5)
        self.ice = False
        self.u = np.zeros(nz)
        self.v = np.zeros(nz)
        self.qc = np.zeros(nz)
        self.qi = np.zeros(nz)
        self.tke = np.full(nz, 0.005)
        self.psfc = 100000.0
        self.znt = 0.1
        self.hfx = 0.0
        self.qfx = 0.0
        self.xland = 1.0
        self.lat = 40.0
        self.tke_profile = None   # ("conv", zi) | ("stable", h) | None
        self.anisotropic = False
        self.dx_only = None       # restrict the dx sweep (edge probes)
        self.raw = {}             # float32 scalar overrides, applied last
        self.raw_levels = {}      # (name, k) -> float32 override
        self.label = FAMILIES[family]

    # ---- profiles -------------------------------------------------------
    def mixed_layer(self, th_sfc, zi, jump, lapse, superadiab=0.0,
                    trop=11000.0):
        z = self.za
        th = np.where(z <= zi, th_sfc, th_sfc + jump + lapse * (z - zi))
        th = th + superadiab * np.exp(-z / 30.0)
        th = th + np.where(z > trop, 0.012 * (z - trop), 0.0)
        self.theta = th

    def stable_layer(self, th0, h, strength, lapse=0.004, trop=11000.0):
        z = self.za
        th = (th0 + strength * (1.0 - np.exp(-z / (0.4 * h)))
              + lapse * np.maximum(z - h, 0.0))
        th = th + np.where(z > trop, 0.012 * (z - trop), 0.0)
        self.theta = th

    def winds(self, rng, ug, shear, veer_deg, jet=None, direction=None):
        z = self.za
        speed = ug + shear * z / 1000.0
        if jet is not None:
            zj, amp = jet
            speed = speed + amp * np.exp(-((z - zj) / (0.45 * zj)) ** 2)
        # log layer below ~200 m
        zs = 200.0
        fac = np.log1p(z / self.znt) / np.log1p(zs / self.znt)
        speed = speed * np.clip(fac, 0.2, 1.0)
        a0 = rng.uniform(0, 2 * np.pi) if direction is None else direction
        ang = a0 + np.deg2rad(veer_deg) * np.clip(1.0 - z / 1500.0, 0.0, 1.0)
        self.u = speed * np.cos(ang)
        self.v = speed * np.sin(ang)

    def rh_profile(self, rh_ml, zi, rh_ft, rh_top=None):
        z = self.za
        top = rh_ml if rh_top is None else rh_top
        rh = np.where(z <= zi, rh_ml + (top - rh_ml) * z / max(zi, 1.0),
                      rh_ft)
        self.rh = np.clip(rh, 0.01, 1.0)

    def cloud(self, kind, zb, zt, peak):
        m = (self.za >= zb) & (self.za <= zt)
        if not m.any():
            k = int(np.argmin(np.abs(self.za - 0.5 * (zb + zt))))
            m = np.zeros(self.nz, bool)
            m[k] = True
        frac = (self.za[m] - zb) / max(zt - zb, 1.0)
        prof = peak * np.clip(0.2 + 0.8 * frac, 0.0, 1.0)
        if kind == "qc":
            self.qc[m] = prof
        else:
            self.qi[m] = prof

    # ---- assemble -------------------------------------------------------
    def build(self):
        nz = self.nz
        qv = np.zeros(nz)
        p_i = np.empty(nz + 1)
        pmid = np.empty(nz)
        for _ in range(3):
            p_i[0] = self.psfc
            for k in range(nz):
                guess = p_i[k] if _ == 0 else pmid[k]
                tk = self.theta[k] * (guess / 1.0e5) ** (RD / CP)
                tv = tk * (1.0 + 0.608 * max(qv[k], 0.0))
                p_i[k + 1] = p_i[k] * np.exp(-G * self.dz[k] / (RD * tv))
                pmid[k] = np.sqrt(p_i[k] * p_i[k + 1])
            tk = self.theta * (pmid / 1.0e5) ** (RD / CP)
            qv = self.rh * qsat(tk, pmid, np.full(nz, self.ice))
        self.qv = qv
        self.p_i = p_i
        self.pmid = pmid
        self.tk = tk

    def surface_layer(self, rng, ust_floor=0.001):
        """Monin-Obukhov: ust, fm (psim), fhh (psih), br and u10/v10."""
        za = self.za[0]
        z0 = self.znt
        z0t = z0 if self.xland > 1.5 else z0 / 7.4
        u1, v1 = self.u[0], self.v[0]
        wspd = max(np.hypot(u1, v1), 0.1)
        th1 = self.theta[0]
        rho = self.psfc / (RD * self.tk[0] * (1.0 + 0.608 * self.qv[0]))
        wth = self.hfx / (rho * CP)
        wq = self.qfx / rho
        wthv = wth + 0.608 * th1 * wq
        ust = KARMAN * wspd / np.log(za / z0)
        fm = fh = None
        zeta = 0.0
        for _ in range(12):
            if wthv != 0.0:
                L = -ust ** 3 * th1 / (KARMAN * G * wthv)
                zeta = float(np.clip(za / L, -5.0, 1.0))
            else:
                zeta = 0.0
            if zeta < 0:
                pm, ph = psi_unstable(zeta)
                pm10, _x = psi_unstable(zeta * 10.0 / za)
            else:
                pm = ph = -5.0 * zeta
                pm10 = -5.0 * zeta * 10.0 / za
            fm = max(np.log(za / z0) - pm, 0.5)
            fh = max(np.log(za / z0t) - ph, 0.5)
            ust = max(KARMAN * wspd / fm, ust_floor)
        fm10 = max(np.log(10.0 / z0) - pm10, 0.1)
        self.wspd = wspd
        self.ust = ust
        self.psim = fm
        self.psih = fh
        self.br = float(np.clip(zeta * fh / (fm * fm), -5.0, 2.0))
        r10 = min(fm10 / fm, 1.0) if za > 10.0 else 1.0
        self.u10 = u1 * r10
        self.v10 = v1 * r10
        self.wstar = (max(G / th1 * wthv, 0.0) * 1000.0) ** (1.0 / 3.0)

    def tke_build(self):
        z = self.za
        e = np.full(self.nz, 0.005)
        if self.tke_profile is not None:
            kind, h = self.tke_profile
            zf = z / h
            if kind == "conv":
                ws = (max(G / self.theta[0] * self.hfx
                          / (1.2 * CP), 0.0) * h) ** (1.0 / 3.0)
                e = e + np.where(zf < 1.1, 0.5 * (1.8 * ws ** 2
                                 * np.maximum(zf, 1e-3) ** (2.0 / 3.0)
                                 * np.maximum(1.0 - 0.8 * zf, 0.0) ** 2
                                 + 3.0 * self.ust ** 2
                                 * np.maximum(1.0 - zf, 0.0)), 0.0)
            else:
                e = e + np.where(zf < 1.0, 3.3 * self.ust ** 2
                                 * np.maximum(1.0 - zf, 0.0) ** 1.75, 0.0)
        self.tke = e

    def finish(self):
        rec = {
            "u": self.u.astype(F32), "v": self.v.astype(F32),
            "theta": self.theta.astype(F32), "qv": self.qv.astype(F32),
            "qc": self.qc.astype(F32), "qi": self.qi.astype(F32),
            "p": self.pmid.astype(F32), "dz": self.dz.astype(F32),
            "tke": self.tke.astype(F32),
            "p_interface": self.p_i.astype(F32),
        }
        # Exner as the model would hold it; both sides read the same word.
        rec["pi"] = ((rec["p"].astype(np.float64) / 1.0e5)
                     ** (RD / CP)).astype(F32)
        for (name, k), value in self.raw_levels.items():
            rec[name][k] = value
        # phy_prep: t_phy = th_phy*pi_phy, one float32 multiply.
        rec["t"] = (rec["theta"] * rec["pi"]).astype(F32)
        scal = {
            "psfc": rec["p_interface"][0], "znt": self.znt, "ust": self.ust,
            "hfx": self.hfx, "qfx": self.qfx, "wspd": self.wspd,
            "br": self.br, "psim": self.psim, "psih": self.psih,
            "xland": self.xland, "u10": self.u10, "v10": self.v10,
            "corf": 2.0 * OMEGA * np.sin(np.deg2rad(self.lat)),
        }
        scal = {k: F32(v) for k, v in scal.items()}
        for k, v in self.raw.items():
            scal[k] = F32(v)
        rec["scalars"] = scal
        rec["nz"] = self.nz
        rec["family"] = self.family
        return rec


# --------------------------------------------------------------------------
# families
# --------------------------------------------------------------------------

def _lat(rng, lo=25.0, hi=55.0, south=0.25):
    lat = rng.uniform(lo, hi)
    return -lat if rng.random() < south else lat


def convective_land_dry(rng):
    c = Column(1, rng)
    zi = rng.uniform(1000, 2800)
    c.mixed_layer(rng.uniform(300, 312), zi, rng.uniform(1.0, 5.0),
                  rng.uniform(0.003, 0.007), superadiab=rng.uniform(0.5, 2.5))
    c.rh_profile(rng.uniform(0.25, 0.5), zi, rng.uniform(0.1, 0.4))
    c.znt = rng.uniform(0.03, 0.5)
    c.winds(rng, rng.uniform(3, 10), rng.uniform(0.5, 4), rng.uniform(5, 30))
    c.hfx = rng.uniform(150, 450)
    c.qfx = rng.uniform(2e-5, 1.5e-4)
    c.psfc = rng.uniform(92000, 101000)
    c.lat = _lat(rng)
    c.tke_profile = ("conv", zi) if rng.random() < 0.6 else None
    return c


def convective_land_moist(rng):
    c = Column(2, rng)
    zi = rng.uniform(800, 1600)
    c.mixed_layer(rng.uniform(298, 304), zi, rng.uniform(0.5, 2.5),
                  rng.uniform(0.003, 0.005), superadiab=rng.uniform(0.3, 1.5))
    c.rh_profile(rng.uniform(0.6, 0.75), zi, rng.uniform(0.5, 0.8),
                 rh_top=rng.uniform(0.92, 1.0))
    c.cloud("qc", zi * rng.uniform(0.9, 1.0), zi + rng.uniform(300, 1200),
            rng.uniform(1e-5, 6e-4))
    c.znt = rng.uniform(0.1, 1.0)
    c.winds(rng, rng.uniform(2, 8), rng.uniform(0.5, 3), rng.uniform(5, 25))
    c.hfx = rng.uniform(80, 250)
    c.qfx = rng.uniform(1.2e-4, 3.2e-4)
    c.psfc = rng.uniform(98000, 101500)
    c.lat = _lat(rng, 15, 40)
    c.tke_profile = ("conv", zi) if rng.random() < 0.6 else None
    return c


def convective_desert(rng):
    c = Column(3, rng)
    zi = rng.uniform(3000, 5000)
    c.mixed_layer(rng.uniform(312, 322), zi, rng.uniform(0.5, 2.0),
                  rng.uniform(0.003, 0.005), superadiab=rng.uniform(2.0, 5.0))
    c.rh_profile(rng.uniform(0.03, 0.12), zi, rng.uniform(0.05, 0.2))
    c.znt = rng.uniform(0.001, 0.05)
    c.winds(rng, rng.uniform(2, 9), rng.uniform(0.5, 3), rng.uniform(5, 20))
    c.hfx = rng.uniform(400, 650)
    c.qfx = rng.uniform(1e-7, 1e-5)
    c.psfc = rng.uniform(93000, 100000)
    c.lat = _lat(rng, 18, 35, south=0.3)
    c.tke_profile = ("conv", zi) if rng.random() < 0.6 else None
    return c


def morning_transition(rng):
    c = Column(4, rng)
    zi = rng.uniform(150, 600)
    c.mixed_layer(rng.uniform(288, 300), zi, rng.uniform(2.0, 7.0),
                  rng.uniform(0.004, 0.008), superadiab=rng.uniform(0.2, 1.0))
    c.rh_profile(rng.uniform(0.6, 0.9), zi, rng.uniform(0.3, 0.6))
    c.znt = rng.uniform(0.03, 0.6)
    c.winds(rng, rng.uniform(1, 6), rng.uniform(1, 5), rng.uniform(10, 35),
            jet=(zi * rng.uniform(1.2, 2.0), rng.uniform(2, 8)))
    c.hfx = rng.uniform(10, 90)
    c.qfx = rng.uniform(5e-6, 8e-5)
    c.psfc = rng.uniform(95000, 101500)
    c.lat = _lat(rng)
    c.tke_profile = ("conv", zi) if rng.random() < 0.5 else None
    return c


def stable_weak(rng):
    c = Column(5, rng)
    h = rng.uniform(150, 500)
    c.stable_layer(rng.uniform(283, 298), h, rng.uniform(1.5, 6.0))
    c.rh_profile(rng.uniform(0.6, 0.9), h, rng.uniform(0.3, 0.6))
    c.znt = rng.uniform(0.03, 0.8)
    c.winds(rng, rng.uniform(3, 9), rng.uniform(0.5, 3), rng.uniform(15, 40),
            jet=(h * rng.uniform(0.8, 1.4), rng.uniform(3, 10)))
    c.hfx = -rng.uniform(5, 45)
    c.qfx = rng.choice([0.0, rng.uniform(-3e-6, 6e-6)])
    c.psfc = rng.uniform(95000, 102000)
    c.lat = _lat(rng)
    c.tke_profile = ("stable", h) if rng.random() < 0.5 else None
    return c


def stable_radiative(rng):
    c = Column(6, rng)
    h = rng.uniform(40, 200)
    c.stable_layer(rng.uniform(270, 292), h, rng.uniform(5.0, 15.0))
    c.rh_profile(rng.uniform(0.75, 0.98), h, rng.uniform(0.2, 0.5))
    c.znt = rng.uniform(0.01, 0.3)
    c.winds(rng, rng.uniform(0.3, 3), rng.uniform(0.5, 2), rng.uniform(10, 50),
            jet=(rng.uniform(150, 450), rng.uniform(4, 14)))
    c.hfx = -rng.uniform(2, 70)
    c.qfx = rng.uniform(-6e-6, 1e-6)
    c.psfc = rng.uniform(96000, 103000)
    c.lat = _lat(rng)
    c.tke_profile = ("stable", h) if rng.random() < 0.5 else None
    return c


def snow_arctic(rng):
    c = Column(7, rng)
    c.ice = True
    h = rng.uniform(80, 400)
    c.stable_layer(rng.uniform(245, 262), h, rng.uniform(4.0, 14.0),
                   lapse=0.005, trop=8500.0)
    c.rh_profile(rng.uniform(0.7, 0.95), h, rng.uniform(0.5, 0.9))
    zb = rng.uniform(300, 1200)
    zt = zb + rng.uniform(200, 700)
    c.cloud("qc", zb, zt, rng.uniform(2e-5, 2.5e-4))   # supercooled liquid
    c.cloud("qi", zb - 150, zt + 200, rng.uniform(5e-6, 1.2e-4))
    c.znt = rng.uniform(0.0005, 0.005)
    c.winds(rng, rng.uniform(2, 10), rng.uniform(0.5, 2), rng.uniform(10, 40))
    c.hfx = rng.uniform(-45, 5)
    c.qfx = rng.uniform(-2e-6, 5e-7)
    c.psfc = rng.uniform(98000, 104000)
    c.lat = rng.uniform(65, 85) * (1 if rng.random() < 0.7 else -1)
    c.tke_profile = ("stable", max(h, 80.0)) if rng.random() < 0.5 else None
    return c


def snow_ice_fog(rng):
    c = Column(8, rng)
    c.ice = True
    h = rng.uniform(30, 150)
    c.stable_layer(rng.uniform(235, 252), h, rng.uniform(8.0, 18.0),
                   lapse=0.004, trop=8000.0)
    c.rh_profile(1.0, h, rng.uniform(0.4, 0.8))
    c.cloud("qi", 0.0, h * rng.uniform(0.8, 1.5), rng.uniform(5e-6, 6e-5))
    c.znt = rng.uniform(0.0005, 0.003)
    c.winds(rng, rng.uniform(0.3, 2.5), rng.uniform(0.2, 2), rng.uniform(0, 30))
    c.hfx = rng.uniform(-25, -1)
    c.qfx = rng.uniform(-1e-6, 0.0)
    c.psfc = rng.uniform(99000, 105000)
    c.lat = rng.uniform(60, 72)
    return c


def snow_spring(rng):
    c = Column(9, rng)
    c.ice = True
    zi = rng.uniform(300, 1200)
    c.mixed_layer(rng.uniform(266, 274), zi, rng.uniform(1.0, 4.0),
                  rng.uniform(0.004, 0.007), superadiab=rng.uniform(0.1, 0.6),
                  trop=9500.0)
    c.rh_profile(rng.uniform(0.6, 0.85), zi, rng.uniform(0.4, 0.8))
    c.cloud("qi", zi + rng.uniform(1000, 2500), zi + rng.uniform(3000, 5000),
            rng.uniform(5e-6, 5e-5))
    if rng.random() < 0.5:
        c.cloud("qc", zi * 0.9, zi + 300, rng.uniform(1e-5, 1.5e-4))
    c.znt = rng.uniform(0.001, 0.01)
    c.winds(rng, rng.uniform(2, 9), rng.uniform(0.5, 3), rng.uniform(5, 25))
    c.hfx = rng.uniform(20, 110)
    c.qfx = rng.uniform(5e-6, 4e-5)
    c.psfc = rng.uniform(90000, 101000)
    c.lat = _lat(rng, 45, 65)
    c.tke_profile = ("conv", zi) if rng.random() < 0.5 else None
    return c


def _charnock(rng, speed):
    ust = 0.035 * speed + 0.05
    return max(0.011 * ust * ust / G + 1.5e-5, 1.0e-5) * rng.uniform(0.8, 1.2)


def ocean_trades(rng):
    c = Column(10, rng)
    c.xland = 2.0
    zi = rng.uniform(500, 900)
    zinv = rng.uniform(1500, 2200)
    c.mixed_layer(rng.uniform(298, 301), zi, 0.5, 0.003,
                  superadiab=rng.uniform(0.2, 0.6))
    z = c.za
    c.theta = c.theta + np.where(z > zinv, rng.uniform(3, 6), 0.0)
    c.rh_profile(rng.uniform(0.75, 0.85), zi, rng.uniform(0.6, 0.9),
                 rh_top=rng.uniform(0.95, 1.0))
    c.cloud("qc", zi, zinv, rng.uniform(5e-5, 5e-4))
    sp = rng.uniform(5, 11)
    c.znt = _charnock(rng, sp)
    c.winds(rng, sp, rng.uniform(-1, 0.5), rng.uniform(3, 15))
    c.hfx = rng.uniform(5, 25)
    c.qfx = rng.uniform(5e-5, 1.3e-4)
    c.lat = _lat(rng, 8, 25, south=0.4)
    c.tke_profile = ("conv", zi) if rng.random() < 0.5 else None
    return c


def ocean_stratocumulus(rng):
    c = Column(11, rng)
    c.xland = 2.0
    zi = rng.uniform(600, 1300)
    c.mixed_layer(rng.uniform(287, 293), zi, rng.uniform(7, 13),
                  rng.uniform(0.004, 0.006), superadiab=rng.uniform(0.0, 0.3))
    c.rh_profile(rng.uniform(0.85, 0.9), zi, rng.uniform(0.1, 0.35),
                 rh_top=1.0)
    c.cloud("qc", zi - rng.uniform(200, 450), zi, rng.uniform(1.5e-4, 7e-4))
    sp = rng.uniform(4, 10)
    c.znt = _charnock(rng, sp)
    c.winds(rng, sp, rng.uniform(-1, 1), rng.uniform(3, 15))
    c.hfx = rng.uniform(0.5, 18)
    c.qfx = rng.uniform(1.5e-5, 6e-5)
    c.lat = _lat(rng, 15, 35, south=0.5)
    c.tke_profile = ("conv", zi) if rng.random() < 0.5 else None
    return c


def ocean_cao(rng):
    c = Column(12, rng)
    c.xland = 2.0
    c.ice = True
    zi = rng.uniform(800, 2200)
    c.mixed_layer(rng.uniform(262, 272), zi, rng.uniform(1.0, 4.0),
                  rng.uniform(0.004, 0.007), superadiab=rng.uniform(1.0, 4.0),
                  trop=9000.0)
    c.rh_profile(rng.uniform(0.7, 0.85), zi, rng.uniform(0.3, 0.6),
                 rh_top=1.0)
    c.cloud("qc", zi * 0.6, zi, rng.uniform(5e-5, 4e-4))
    c.cloud("qi", zi * 0.7, zi + 200, rng.uniform(1e-5, 1.5e-4))
    sp = rng.uniform(12, 22)
    c.znt = _charnock(rng, sp)
    c.winds(rng, sp, rng.uniform(-2, 1), rng.uniform(5, 20))
    c.hfx = rng.uniform(150, 450)
    c.qfx = rng.uniform(4e-5, 1.5e-4)
    c.lat = _lat(rng, 50, 72, south=0.3)
    c.tke_profile = ("conv", zi) if rng.random() < 0.6 else None
    return c


def ocean_fog(rng):
    c = Column(13, rng)
    c.xland = 2.0
    h = rng.uniform(60, 300)
    c.stable_layer(rng.uniform(280, 292), h, rng.uniform(2.0, 8.0))
    c.rh_profile(1.0, h, rng.uniform(0.5, 0.85))
    c.cloud("qc", 0.0, h, rng.uniform(5e-5, 3e-4))
    sp = rng.uniform(3, 9)
    c.znt = _charnock(rng, sp)
    c.winds(rng, sp, rng.uniform(0, 2), rng.uniform(5, 20))
    c.hfx = -rng.uniform(3, 35)
    c.qfx = rng.uniform(-8e-6, 0.0)
    c.lat = _lat(rng, 35, 55)
    c.tke_profile = ("stable", h) if rng.random() < 0.5 else None
    return c


def high_terrain_day(rng):
    c = Column(14, rng)
    zi = rng.uniform(2000, 4500)
    c.psfc = rng.uniform(56000, 75000)
    c.mixed_layer(rng.uniform(310, 325), zi, rng.uniform(0.5, 3.0),
                  rng.uniform(0.004, 0.007), superadiab=rng.uniform(1.0, 4.0),
                  trop=8000.0)
    c.rh_profile(rng.uniform(0.1, 0.4), zi, rng.uniform(0.1, 0.5))
    if rng.random() < 0.4:
        c.cloud("qc", zi, zi + 600, rng.uniform(1e-5, 2e-4))
        c.cloud("qi", zi + 400, zi + 1500, rng.uniform(5e-6, 6e-5))
        c.ice = True
    c.znt = rng.uniform(0.05, 1.0)
    c.winds(rng, rng.uniform(4, 14), rng.uniform(2, 8), rng.uniform(5, 30))
    c.hfx = rng.uniform(200, 600)
    c.qfx = rng.uniform(5e-6, 8e-5)
    c.lat = _lat(rng, 25, 45, south=0.3)
    c.tke_profile = ("conv", zi) if rng.random() < 0.6 else None
    return c


def high_terrain_night(rng):
    c = Column(15, rng)
    h = rng.uniform(60, 400)
    c.psfc = rng.uniform(58000, 80000)
    c.stable_layer(rng.uniform(295, 315), h, rng.uniform(3.0, 15.0),
                   lapse=0.005, trop=8000.0)
    c.rh_profile(rng.uniform(0.3, 0.7), h, rng.uniform(0.1, 0.4))
    c.znt = rng.uniform(0.05, 1.2)
    c.winds(rng, rng.uniform(2, 10), rng.uniform(1, 5), rng.uniform(5, 40))
    c.hfx = -rng.uniform(5, 110)
    c.qfx = rng.uniform(-2e-6, 2e-6)
    c.lat = _lat(rng, 25, 50, south=0.3)
    c.tke_profile = ("stable", h) if rng.random() < 0.5 else None
    return c


def high_terrain_snow(rng):
    c = Column(16, rng)
    c.ice = True
    c.psfc = rng.uniform(55000, 72000)
    if rng.random() < 0.5:
        zi = rng.uniform(300, 1500)
        c.mixed_layer(rng.uniform(270, 285), zi, rng.uniform(1.0, 4.0),
                      rng.uniform(0.004, 0.007), superadiab=0.3, trop=7500.0)
        c.hfx = rng.uniform(10, 150)
        c.tke_profile = ("conv", zi) if rng.random() < 0.5 else None
    else:
        zi = rng.uniform(50, 300)
        c.stable_layer(rng.uniform(262, 280), zi, rng.uniform(4, 15),
                       lapse=0.005, trop=7500.0)
        c.hfx = -rng.uniform(5, 80)
    c.rh_profile(rng.uniform(0.6, 0.95), zi, rng.uniform(0.4, 0.9))
    c.cloud("qi", zi * 0.5, zi + rng.uniform(500, 2000),
            rng.uniform(5e-6, 1.5e-4))
    c.znt = rng.uniform(0.002, 0.05)
    c.winds(rng, rng.uniform(3, 18), rng.uniform(1, 6), rng.uniform(5, 30))
    c.qfx = rng.uniform(-2e-6, 1e-5)
    c.lat = _lat(rng, 30, 50, south=0.3)
    return c


def windy_neutral(rng):
    c = Column(17, rng)
    zi = rng.uniform(800, 2000)
    c.mixed_layer(rng.uniform(280, 295), zi, rng.uniform(0.3, 1.5),
                  rng.uniform(0.004, 0.006), superadiab=0.0)
    c.rh_profile(rng.uniform(0.85, 0.95), zi, rng.uniform(0.85, 0.99),
                 rh_top=1.0)
    c.cloud("qc", zi * 0.5, zi + 1500, rng.uniform(1e-4, 5e-4))
    if rng.random() < 0.5:
        c.cloud("qi", zi + 1000, zi + 4000, rng.uniform(1e-5, 1e-4))
        c.ice = True
    c.znt = rng.uniform(0.05, 0.8)
    if rng.random() < 0.3:
        c.xland = 2.0
        c.znt = _charnock(rng, 20.0)
    c.winds(rng, rng.uniform(12, 25), rng.uniform(2, 8), rng.uniform(10, 30))
    c.hfx = rng.uniform(-8, 8)
    c.qfx = rng.uniform(-1e-6, 1e-5)
    c.lat = _lat(rng, 35, 60, south=0.4)
    c.tke_profile = ("stable", zi) if rng.random() < 0.5 else None
    return c


FAMILY_BUILDERS = (
    (convective_land_dry, 8), (convective_land_moist, 8),
    (convective_desert, 6), (morning_transition, 6),
    (stable_weak, 8), (stable_radiative, 7),
    (snow_arctic, 7), (snow_ice_fog, 5), (snow_spring, 6),
    (ocean_trades, 6), (ocean_stratocumulus, 7), (ocean_cao, 6),
    (ocean_fog, 5),
    (high_terrain_day, 7), (high_terrain_night, 6), (high_terrain_snow, 6),
    (windy_neutral, 6),
)


def finished(c, rng, ust_floor=0.001):
    c.build()
    c.surface_layer(rng, ust_floor=ust_floor)
    c.tke_build()
    return c


# --------------------------------------------------------------------------
# edge probes
# --------------------------------------------------------------------------

def _neutral_base(rng, label):
    c = Column(18, rng, grid=GRIDS[1])
    zi = 900.0
    c.mixed_layer(300.0, zi, 1.5, 0.004, superadiab=0.1)
    c.rh_profile(0.6, zi, 0.4)
    c.znt = 0.1
    c.winds(rng, 6.0, 1.0, 20.0, direction=0.7)
    c.hfx = 0.5
    c.qfx = 1e-7
    c.lat = 40.0
    finished(c, rng)
    c.label = label
    return c


def edge_columns(rng):
    out = []
    # br sign probes on the sfcflg compare (br > 0), signed zero and subnormal
    for label, br in (("br=+0", F32(0.0)), ("br=-0", NZERO),
                      ("br=+subnormal", SUB), ("br=-subnormal", -SUB),
                      ("br=min-normal", MINNORM)):
        c = _neutral_base(rng, label)
        c.raw["br"] = br
        out.append(c)
    # calm: ust = hfx = qfx = 0 with stable br
    c = _neutral_base(rng, "calm-zero-fluxes")
    c.raw.update(ust=0.0, hfx=0.0, qfx=0.0, br=0.02, wspd=0.1)
    out.append(c)
    # WRF's own prfac2 0/0 (module_bl_shinhong.F:1010): unstable br with
    # wstar3 = ust3 = 0.  Unreachable from WRF's surface layers (they floor
    # ust), kept so the NaN geometry is measured, not assumed.
    c = _neutral_base(rng, "nan-probe-ust-subnormal")
    c.raw.update(ust=SUB, hfx=0.0, qfx=NZERO, br=-0.02)
    out.append(c)
    c = _neutral_base(rng, "nan-probe-ust-zero")
    c.raw.update(ust=0.0, hfx=0.0, qfx=0.0, br=-0.02)
    out.append(c)
    # in-cloud imvdif test at exactly qc = 0.01e-3 and one float32 step above
    for label, qcv in (("qc=0.01e-3", F32(0.01e-3)),
                       ("qc=0.01e-3+1ulp",
                        np.nextafter(F32(0.01e-3), F32(1.0)))):
        c = Column(18, rng, grid=GRIDS[1])
        zi = 900.0
        c.mixed_layer(300.0, zi, 6.0, 0.0002, superadiab=0.5)
        c.rh_profile(0.7, zi, 0.6)
        c.winds(rng, 5.0, 1.0, 20.0, direction=1.1)
        c.hfx, c.qfx, c.lat = 80.0, 5e-5, 35.0
        finished(c, rng)
        for k in range(c.nz):
            if 1100.0 <= c.za[k] <= 1500.0:
                c.raw_levels[("qc", k)] = qcv
        c.label = label
        out.append(c)
    # PBL fills the column (kpbl = kte): shallow grid, very deep mixing
    c = Column(18, rng, grid=(35, 30.0, 1.02, 60.0))
    c.mixed_layer(305.0, 1.0e6, 0.0, 0.0, superadiab=0.2, trop=1.0e6)
    c.rh_profile(0.3, 1.0e6, 0.3)
    c.winds(rng, 6.0, 1.0, 20.0, direction=0.3)
    c.hfx, c.qfx, c.lat = 400.0, 1e-4, 30.0
    finished(c, rng)
    c.label = "kpbl=kte"
    out.append(c)
    # uniform wind, equator: dux = dvx = 0 exactly and corf = 0
    c = Column(18, rng, grid=GRIDS[0])
    c.mixed_layer(300.0, 900.0, 2.5, 0.004, superadiab=0.5)
    c.rh_profile(0.6, 900.0, 0.5)
    c.hfx, c.qfx, c.lat = 100.0, 6e-5, 0.0
    finished(c, rng)
    c.u[:] = 5.0
    c.v[:] = -2.0
    c.surface_layer(rng)
    c.label = "uniform-wind-equator"
    out.append(c)
    # subnormal, signed-zero and slightly negative moisture
    c = _neutral_base(rng, "subnormal-and-negative-moisture")
    c.raw.update(hfx=20.0, qfx=SUB, br=-0.01)
    for (name, k), val in ((("qv", 0), SUB), (("qv", 1), NZERO),
                           (("qv", 2), MINNORM), (("qc", 3), SUB),
                           (("qi", 4), -SUB), (("qc", 5), NZERO),
                           (("qv", 6), F32(-1.0e-9)),
                           (("qc", 7), F32(-1.0e-12))):
        c.raw_levels[(name, k)] = val
    out.append(c)
    # extreme heating (wildfire-scale) and extreme winds
    c = Column(18, rng, grid=GRIDS[2])
    c.mixed_layer(318.0, 4500.0, 1.0, 0.004, superadiab=6.0)
    c.rh_profile(0.05, 4500.0, 0.1)
    c.znt = 0.5
    c.winds(rng, 14.0, 4.0, 10.0, direction=2.0)
    c.hfx, c.qfx, c.lat = 900.0, 6e-4, 33.0
    finished(c, rng)
    c.label = "extreme-heating"
    out.append(c)
    # LES-like 4 m first layer
    c = Column(18, rng, grid=(60, 4.0, 1.09, 400.0))
    c.mixed_layer(302.0, 1200.0, 3.0, 0.005, superadiab=1.5)
    c.rh_profile(0.5, 1200.0, 0.3)
    c.znt = 0.05
    c.winds(rng, 6.0, 2.0, 20.0, direction=0.0)
    c.hfx, c.qfx, c.lat = 250.0, 1e-4, 45.0
    finished(c, rng)
    c.label = "thin-first-layer-4m"
    out.append(c)
    # anisotropic grid dy = 4 dx (the sqrt(dx*dy) contract)
    c = Column(18, rng, grid=GRIDS[1])
    c.mixed_layer(304.0, 1500.0, 3.0, 0.005, superadiab=1.0)
    c.rh_profile(0.4, 1500.0, 0.3)
    c.winds(rng, 7.0, 2.0, 20.0, direction=0.9)
    c.hfx, c.qfx, c.lat = 220.0, 1e-4, 38.0
    c.tke_profile = ("conv", 1500.0)
    finished(c, rng)
    c.anisotropic = True
    c.label = "anisotropic-dy=4dx"
    out.append(c)
    # very stable, br at the sfclay cap, near-zero ust
    c = Column(18, rng, grid=GRIDS[1])
    c.stable_layer(265.0, 60.0, 15.0)
    c.rh_profile(0.9, 60.0, 0.4)
    c.winds(rng, 0.4, 0.5, 10.0, direction=4.0)
    c.hfx, c.qfx, c.lat = -3.0, 0.0, 70.0
    finished(c, rng)
    c.raw.update(br=2.0, ust=0.005)
    c.label = "very-stable-br-cap"
    out.append(c)
    # strong shear, Ri < 0 above the PBL, southern hemisphere
    c = Column(18, rng, grid=GRIDS[0])
    c.mixed_layer(300.0, 600.0, 0.2, 0.0008, superadiab=0.5)
    c.rh_profile(0.6, 600.0, 0.5)
    c.winds(rng, 2.0, 12.0, 30.0, direction=0.4)
    c.hfx, c.qfx, c.lat = 60.0, 3e-5, -45.0
    finished(c, rng)
    c.label = "strong-shear-south"
    out.append(c)
    # pole: largest |corf|, ice-fog stable column
    c = Column(18, rng, grid=GRIDS[2])
    c.ice = True
    c.stable_layer(240.0, 100.0, 12.0, trop=8000.0)
    c.rh_profile(1.0, 100.0, 0.6)
    c.cloud("qi", 0.0, 120.0, 3e-5)
    c.znt = 0.001
    c.winds(rng, 1.5, 0.5, 20.0, direction=3.0)
    c.hfx, c.qfx, c.lat = -10.0, -5e-7, 89.9
    finished(c, rng)
    c.label = "pole-ice-fog"
    out.append(c)
    return out


# --------------------------------------------------------------------------

def records():
    rng = np.random.default_rng(SEED)
    base = []
    for builder, count in FAMILY_BUILDERS:
        for _ in range(count):
            base.append(finished(builder(rng), rng))
    base.extend(edge_columns(rng))
    recs = []
    labels = []
    for bid, c in enumerate(base):
        rec = c.finish()
        runs = [(i, 1) for i in range(len(DX_SWEEP))] + [(TKE_OFF_DX, 0)]
        for idx, tke_diag in runs:
            dx, dt = DX_SWEEP[idx]
            r = dict(rec)
            r["scalars"] = dict(rec["scalars"])
            r["scalars"]["dt"] = F32(dt)
            r["scalars"]["dx"] = F32(dx)
            r["scalars"]["dy"] = F32(4.0 * dx if c.anisotropic else dx)
            r["tke_diag"] = tke_diag
            r["base"] = bid
            recs.append(r)
            labels.append({"column": len(labels), "base": bid,
                           "family": FAMILIES[c.family], "label": c.label,
                           "dx": dx, "dt": dt, "tke_diag": tke_diag,
                           "nz": c.nz, "lat": round(float(c.lat), 3),
                           "xland": float(c.xland)})
    return recs, labels


def write_records(recs, path):
    with open(path, "wb") as fh:
        for r in recs:
            fh.write(struct.pack("<4i", r["nz"], r["tke_diag"], r["family"],
                                 r["base"]))
            fh.write(np.asarray([r["scalars"][k] for k in SCALAR_ORDER],
                                dtype="<f4").tobytes())
            for name in LEVEL_ORDER:
                arr = np.asarray(r[name], dtype="<f4")
                assert arr.shape == (r["nz"],), (name, arr.shape)
                fh.write(arr.tobytes())
            pin = np.asarray(r["p_interface"], dtype="<f4")
            assert pin.shape == (r["nz"] + 1,)
            fh.write(pin.tobytes())


def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    recs, labels = records()
    write_records(recs, Path(argv[1]))
    if len(argv) == 3:
        Path(argv[2]).write_text(json.dumps(
            {"seed": SEED, "families": FAMILIES, "columns": labels},
            indent=1) + "\n", encoding="utf-8")
    nbase = len({lab["base"] for lab in labels})
    print(f"wrote {len(recs)} columns ({nbase} base columns x "
          f"{len(DX_SWEEP)} dx + tke_diag=0) to {argv[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
