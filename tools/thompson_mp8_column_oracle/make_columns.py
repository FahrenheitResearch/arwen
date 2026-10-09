#!/usr/bin/env python3
"""Regime columns for the classic Thompson (mp_physics=8) column oracle.

Builds float32 model columns that span the regimes a forecast meets, each a
physically consistent profile (hydrostatic pressure from the temperature and
humidity profile, geopotential on the interfaces, theta from T and p), plus
edge-case columns that sit on the scheme's thresholds.  Deterministic: one
seed, no inputs.

Regimes (counts are the defaults):

* ``convective``    deep moist convection: warm moist boundary layer, cloud
                    water to the homogeneous-freezing level, rain, graupel,
                    snow and ice aloft, updraft and downdraft columns
* ``stable_night``  radiation inversion with fog in the lowest levels, no
                    precipitation, near-zero vertical motion
* ``snow``          winter storm: whole column below 0 C, snow and ice with
                    supercooled cloud and light graupel
* ``cirrus``        dry lower troposphere, ice cloud at 9 to 12 km
* ``ocean``         marine boundary layer at sea level: stratocumulus with
                    drizzle (many small drops)
* ``high_terrain``  surface at 2.8 to 4.8 km (550 to 720 hPa), orographic
                    snow, graupel and cloud
* ``bright_band``   stratiform rain: snow above a 1 to 2 km melting level,
                    melting snow and rain below
* ``freezing_rain`` warm nose aloft over a subfreezing surface layer
* ``edge``          threshold and pathological columns (see ``_edge``)

Output: an ``.npz`` with ``p th qv qc qr qi qs qg ni nr`` as (ncol, nz),
``geop w`` as (ncol, nz+1) and ``labels`` (regime/variant), all float32.
Geopotential is the whole field over a zero base (``php`` over ``phb=0``),
as the committed fixture holds it.

usage: make_columns.py OUT.npz [--nz 50] [--seed 20261007] [--nan | --stress N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

f32 = np.float32
G = 9.81
RD = 287.0
RV = 461.6
CP = 7.0 * RD / 2.0
P0 = 1.0e5
R1 = 1.0e-12

SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr")


def _esw(t):
    """Saturation vapour pressure over water (Pa), Bolton."""
    tc = t - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def _esi(t):
    """Saturation vapour pressure over ice (Pa), Murphy-Koop."""
    return np.exp(9.550426 - 5723.265 / t + 3.53068 * np.log(t)
                  - 0.00728332 * t)


def _qsat(t, p, ice=False):
    e = np.where(ice & (t < 273.15), _esi(t), _esw(t)) if ice else _esw(t)
    e = np.minimum(e, 0.5 * p)
    return 0.622 * e / (p - e)


def _grid(nz, z_sfc, z_top=20500.0, first=40.0, thin=None):
    """Interface heights (nz+1), stretched from ``first`` metres at the
    surface; ``thin`` overrides the lowest layer depths."""
    n = nz
    # geometric stretching solved for the total depth
    depth = z_top - z_sfc
    lo, hi = 1.0, 1.5
    for _ in range(80):
        r = 0.5 * (lo + hi)
        tot = first * (r ** n - 1.0) / (r - 1.0)
        if tot > depth:
            hi = r
        else:
            lo = r
    r = 0.5 * (lo + hi)
    dz = first * r ** np.arange(n)
    if thin is not None:
        thin = np.asarray(thin, dtype=float)
        dz[:thin.size] = thin
    dz *= depth / dz.sum()
    return z_sfc + np.concatenate([[0.0], np.cumsum(dz)])


class _Col:
    """One column under construction, in float64."""

    def __init__(self, nz, z_sfc=0.0, p_sfc=101325.0, thin=None,
                 station=False):
        """``p_sfc`` is the sea-level pressure, reduced to the surface
        height with an 8 km scale height, unless ``station``."""
        self.zf = _grid(nz, z_sfc, thin=thin)
        if not station:
            p_sfc = p_sfc * np.exp(-z_sfc / 8000.0)
        self.zh = 0.5 * (self.zf[1:] + self.zf[:-1])
        self.nz = nz
        self.p_sfc = p_sfc
        self.t = np.zeros(nz)
        self.rh = np.zeros(nz)
        self.ice_rh = False
        self.w = np.zeros(nz + 1)
        for s in SPECIES:
            setattr(self, s, np.zeros(nz))

    def temperature(self, t_sfc, lapse=6.5e-3, z_trop=12000.0, t_strat=None,
                    inversion=None, nose=None):
        z = self.zh - self.zf[0]
        t = t_sfc - lapse * z
        t_trop = t_sfc - lapse * (z_trop - self.zf[0])
        if t_strat is None:
            t_strat = t_trop
        above = self.zh > z_trop
        t = np.where(above, t_trop + 1.5e-3 * (self.zh - z_trop), t)
        t = np.maximum(t, min(t_strat, t_trop) - 2.0)
        if inversion is not None:
            depth, strength = inversion
            t = t + np.where(z < depth, -strength * (1.0 - z / depth), 0.0)
        if nose is not None:
            zc, half, amp = nose
            t = t + amp * np.exp(-((self.zh - zc) / half) ** 2)
        self.t = t

    def humidity(self, rh_low, rh_mid, rh_high, ice=False):
        z = self.zh - self.zf[0]
        self.rh = np.interp(z, [0.0, 2000.0, 6000.0, 20000.0],
                            [rh_low, rh_mid, rh_high, 0.05])
        self.ice_rh = ice

    def layer(self, name, value, z_lo, z_hi, *, shape="hat"):
        z = self.zh
        if shape == "hat":
            mid, half = 0.5 * (z_lo + z_hi), 0.5 * (z_hi - z_lo)
            prof = np.clip(1.0 - np.abs(z - mid) / half, 0.0, 1.0)
        else:
            prof = ((z >= z_lo) & (z <= z_hi)).astype(float)
        cur = getattr(self, name)
        setattr(self, name, cur + value * prof)

    def updraft(self, w_max, z_lo, z_hi):
        z = self.zf
        prof = np.clip(np.sin(np.pi * (z - z_lo) / (z_hi - z_lo)), 0.0, None)
        prof[(z < z_lo) | (z > z_hi)] = 0.0
        self.w = self.w + w_max * prof

    def finish(self):
        """Hydrostatic pressure, theta, qv from RH; float32 arrays."""
        nz = self.nz
        p_f = np.empty(nz + 1)
        p_f[0] = self.p_sfc
        p = np.empty(nz)
        qv = np.empty(nz)
        for k in range(nz):
            # iterate qv and the layer's mean pressure
            pk = p_f[k] * np.exp(-G * (self.zh[k] - self.zf[k])
                                 / (RD * self.t[k]))
            for _ in range(4):
                qs = _qsat(self.t[k], pk, self.ice_rh)
                q = max(self.rh[k] * qs, 0.0)
                tv = self.t[k] * (1.0 + 0.608 * q)
                pk = p_f[k] * np.exp(-G * (self.zh[k] - self.zf[k])
                                     / (RD * tv))
            p[k] = pk
            qv[k] = q
            p_f[k + 1] = pk * np.exp(-G * (self.zf[k + 1] - self.zh[k])
                                     / (RD * tv))
        self.qv = qv + self.qv
        th = self.t * (P0 / p) ** (RD / CP)
        out = {"p": p, "th": th, "geop": G * self.zf, "w": self.w}
        for s in SPECIES:
            out[s] = getattr(self, s)
        return {k: np.asarray(v, dtype=f32) for k, v in out.items()}


def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _lognum(rng, lo, hi):
    return float(10.0 ** rng.uniform(np.log10(lo), np.log10(hi)))


def _convective(rng, nz, variant):
    c = _Col(nz, z_sfc=_u(rng, 0.0, 400.0), p_sfc=_u(rng, 99000, 101500))
    c.temperature(_u(rng, 297, 306), lapse=_u(rng, 6.0e-3, 7.2e-3),
                  z_trop=_u(rng, 13000, 15500))
    c.humidity(_u(rng, 0.8, 0.95), _u(rng, 0.65, 0.95), _u(rng, 0.5, 0.9))
    frz = c.zf[0] + (c.t[0] - 273.15) / 6.5e-3
    c.layer("qc", _u(rng, 0.4e-3, 3.0e-3), 900, frz + 4500)
    c.layer("qr", _u(rng, 0.5e-3, 6.0e-3), 0, frz + 1500, shape="box"
            if variant % 3 == 0 else "hat")
    c.layer("qg", _u(rng, 0.3e-3, 6.0e-3), frz - 1500, frz + 6000)
    c.layer("qs", _u(rng, 0.1e-3, 1.5e-3), frz + 2000, 13000)
    c.layer("qi", _u(rng, 0.03e-3, 0.6e-3), frz + 4000, 14500)
    c.layer("ni", _lognum(rng, 1e4, 2e6), frz + 4000, 14500)
    c.layer("nr", _lognum(rng, 3e2, 2e5), 0, frz + 1500)
    if variant == 1:
        c.updraft(-_u(rng, 3, 12), 0, 6000)          # downdraft core
    else:
        c.updraft(_u(rng, 3, 35), 500, 14000)
    return c.finish()


def _stable_night(rng, nz, variant):
    c = _Col(nz, z_sfc=_u(rng, 0.0, 1500.0), p_sfc=_u(rng, 85000, 102500))
    c.temperature(_u(rng, 268, 288), lapse=_u(rng, 5.0e-3, 6.5e-3),
                  inversion=(_u(rng, 150, 400), _u(rng, 4, 10)))
    c.humidity(_u(rng, 0.97, 1.01) if variant % 2 == 0 else
               _u(rng, 0.5, 0.8), _u(rng, 0.3, 0.6), 0.2)
    if variant % 2 == 0:
        z0 = c.zf[0]
        c.layer("qc", _u(rng, 0.03e-3, 0.3e-3), z0, z0 + _u(rng, 60, 250),
                shape="box")
    c.w = c.w + rng.uniform(-0.02, 0.02, size=nz + 1)
    c.w[0] = 0.0
    return c.finish()


def _snow(rng, nz, variant):
    c = _Col(nz, z_sfc=_u(rng, 0.0, 1200.0), p_sfc=_u(rng, 88000, 103000))
    c.temperature(_u(rng, 255, 271.5), lapse=_u(rng, 4.0e-3, 6.5e-3),
                  z_trop=_u(rng, 8500, 10500))
    c.humidity(_u(rng, 0.9, 1.02), _u(rng, 0.9, 1.05), 0.5, ice=True)
    z0 = c.zf[0]
    c.layer("qs", _u(rng, 0.1e-3, 1.2e-3), z0, z0 + _u(rng, 4000, 7000),
            shape="box" if variant % 2 else "hat")
    c.layer("qi", _u(rng, 0.005e-3, 0.15e-3), z0 + 2000, z0 + 8000)
    c.layer("ni", _lognum(rng, 5e3, 5e5), z0 + 2000, z0 + 8000)
    c.layer("qc", _u(rng, 0.0, 0.25e-3), z0 + 500, z0 + 2500)
    c.layer("qg", _u(rng, 0.0, 0.3e-3), z0, z0 + 3000)
    if variant == 3:                                 # freezing drizzle
        c.layer("qr", _u(rng, 0.01e-3, 0.1e-3), z0, z0 + 1500)
        c.layer("nr", _lognum(rng, 1e4, 1e6), z0, z0 + 1500)
    c.updraft(_u(rng, 0.05, 1.0), z0, z0 + 7000)
    return c.finish()


def _cirrus(rng, nz, variant):
    c = _Col(nz, z_sfc=_u(rng, 0.0, 800.0), p_sfc=_u(rng, 95000, 102000))
    c.temperature(_u(rng, 285, 300), lapse=_u(rng, 6.5e-3, 7.5e-3),
                  z_trop=_u(rng, 11500, 13500))
    c.humidity(_u(rng, 0.2, 0.5), _u(rng, 0.2, 0.4), _u(rng, 0.8, 1.15),
               ice=True)
    c.layer("qi", _u(rng, 0.004e-3, 0.08e-3), 9000, 12500)
    c.layer("ni", _lognum(rng, 1e3, 3e5), 9000, 12500)
    c.layer("qs", _u(rng, 0.0, 0.05e-3), 8500, 12000)
    c.updraft(_u(rng, 0.0, 0.5), 8000, 13000)
    return c.finish()


def _ocean(rng, nz, variant):
    c = _Col(nz, z_sfc=0.0, p_sfc=_u(rng, 100500, 102500))
    sst = _u(rng, 286, 302)
    c.temperature(sst - 0.5, lapse=_u(rng, 6.0e-3, 9.0e-3),
                  inversion=None)
    zi = _u(rng, 600, 1400)
    # capping inversion above the marine boundary layer
    c.t = c.t + np.where(c.zh > zi, _u(rng, 4, 10), 0.0)
    c.humidity(_u(rng, 0.85, 1.0), _u(rng, 0.2, 0.5), 0.3)
    c.layer("qc", _u(rng, 0.1e-3, 0.6e-3), zi - 400, zi)
    c.layer("qr", _u(rng, 0.001e-3, 0.08e-3), 0, zi)
    c.layer("nr", _lognum(rng, 1e4, 2e6), 0, zi)
    if variant == 2:                                 # tropical shower
        c.layer("qr", _u(rng, 0.5e-3, 3e-3), 0, 4000)
        c.layer("nr", _lognum(rng, 1e3, 3e4), 0, 4000)
        c.layer("qc", _u(rng, 0.5e-3, 1.5e-3), 600, 4500)
        c.updraft(_u(rng, 2, 8), 300, 5000)
    else:
        c.updraft(_u(rng, -0.05, 0.3), 0, zi)
    return c.finish()


def _high_terrain(rng, nz, variant):
    z0 = _u(rng, 2800, 4800)
    p0 = 101325.0 * np.exp(-z0 / 8000.0) * _u(rng, 0.98, 1.02)
    c = _Col(nz, z_sfc=z0, p_sfc=p0, station=True)
    c.temperature(_u(rng, 255, 282), lapse=_u(rng, 5.5e-3, 7.5e-3),
                  z_trop=_u(rng, 11000, 15000))
    c.humidity(_u(rng, 0.8, 1.0), _u(rng, 0.7, 1.0), 0.6,
               ice=bool(variant % 2))
    frz = z0 + (c.t[0] - 273.15) / 6.5e-3
    c.layer("qs", _u(rng, 0.2e-3, 2.0e-3), z0, z0 + 5000)
    c.layer("qg", _u(rng, 0.0, 1.5e-3), z0, z0 + 3500)
    c.layer("qc", _u(rng, 0.05e-3, 0.8e-3), z0 + 200, z0 + 2500)
    c.layer("qi", _u(rng, 0.01e-3, 0.2e-3), z0 + 2000, z0 + 7000)
    c.layer("ni", _lognum(rng, 1e4, 1e6), z0 + 2000, z0 + 7000)
    if frz > z0 + 200:                               # rain below a warm base
        c.layer("qr", _u(rng, 0.05e-3, 1.0e-3), z0, frz)
        c.layer("nr", _lognum(rng, 1e3, 1e5), z0, frz)
    c.updraft(_u(rng, 0.5, 6.0), z0, z0 + 6000)
    return c.finish()


def _bright_band(rng, nz, variant):
    c = _Col(nz, z_sfc=_u(rng, 0.0, 600.0), p_sfc=_u(rng, 96000, 102000))
    c.temperature(_u(rng, 276, 284), lapse=_u(rng, 5.5e-3, 6.5e-3),
                  z_trop=_u(rng, 10000, 12000))
    c.humidity(_u(rng, 0.9, 1.0), _u(rng, 0.95, 1.02), 0.7)
    frz = c.zf[0] + (c.t[0] - 273.15) / 6.0e-3
    c.layer("qs", _u(rng, 0.3e-3, 1.5e-3), frz - 400, frz + 5000)
    c.layer("qr", _u(rng, 0.2e-3, 2.0e-3), 0, frz + 200, shape="box")
    c.layer("nr", _lognum(rng, 3e2, 3e4), 0, frz + 200, shape="box")
    c.layer("qi", _u(rng, 0.01e-3, 0.1e-3), frz + 2000, frz + 6000)
    c.layer("ni", _lognum(rng, 1e4, 3e5), frz + 2000, frz + 6000)
    c.layer("qg", _u(rng, 0.0, 0.4e-3), frz - 300, frz + 1500)
    c.layer("qc", _u(rng, 0.0, 0.3e-3), frz - 800, frz + 800)
    c.updraft(_u(rng, 0.05, 0.6), 0, frz + 6000)
    return c.finish()


def _freezing_rain(rng, nz, variant):
    c = _Col(nz, z_sfc=_u(rng, 0.0, 500.0), p_sfc=_u(rng, 98000, 103000))
    c.temperature(_u(rng, 266, 271.5), lapse=_u(rng, 4.0e-3, 6.0e-3),
                  z_trop=_u(rng, 9000, 11000),
                  nose=(_u(rng, 1200, 2200), _u(rng, 400, 800),
                        _u(rng, 6, 11)))
    c.humidity(_u(rng, 0.92, 1.0), _u(rng, 0.95, 1.02), 0.6)
    c.layer("qs", _u(rng, 0.2e-3, 1.0e-3), 2500, 7000)
    c.layer("qr", _u(rng, 0.1e-3, 1.5e-3), 0, 2600, shape="box")
    c.layer("nr", _lognum(rng, 1e3, 1e5), 0, 2600, shape="box")
    c.layer("qg", _u(rng, 0.0, 0.2e-3), 0, 1000)
    c.layer("qi", _u(rng, 0.0, 0.05e-3), 4000, 8000)
    c.layer("ni", _lognum(rng, 1e3, 1e5), 4000, 8000)
    c.updraft(_u(rng, 0.05, 0.5), 0, 6000)
    return c.finish()


def _edge(rng, nz):
    """Threshold and pathological columns, one per named variant."""
    out = []
    base = _convective(np.random.default_rng(7), nz, 0)
    snow = _snow(np.random.default_rng(8), nz, 0)

    def from_(src, name):
        return name, {k: v.copy() for k, v in src.items()}

    # 1 completely dry: no condensate, very low vapour
    name, c = from_(base, "dry")
    for s in SPECIES[1:]:
        c[s][:] = 0
    c["qv"] *= f32(1e-3)
    out.append((name, c))
    # 2 every species exactly R1, one unit above and one below, by level
    name, c = from_(base, "at_R1")
    r1 = f32(R1)
    up, dn = np.nextafter(r1, f32(1)), np.nextafter(r1, f32(0))
    for s in ("qc", "qr", "qi", "qs", "qg"):
        c[s][:] = np.resize(np.array([r1, up, dn, 0], dtype=f32), nz)
    c["ni"][:] = f32(1e3)
    c["nr"][:] = f32(1e2)
    out.append((name, c))
    # 3 negative and signed-zero hydrometeors (advection undershoot)
    name, c = from_(base, "negative_and_minus_zero")
    for s in ("qc", "qr", "qi", "qs", "qg", "ni", "nr"):
        a = c[s]
        a[0::4] = -np.abs(a[0::4]) * f32(1e-3) - f32(1e-14)
        a[1::4] = f32(-0.0)
    out.append((name, c))
    # 4 orphan numbers: n > 0 over q = 0
    name, c = from_(base, "orphan_numbers")
    for q, n in (("qi", "ni"), ("qr", "nr")):
        c[n][:] = f32(5e4)
        c[q][::2] = f32(0)
    out.append((name, c))
    # 5 subnormal mixing ratios and numbers
    name, c = from_(base, "subnormal")
    for s in ("qc", "qr", "qi", "qs", "qg", "ni", "nr"):
        c[s][::2] = f32(1.0e-40)
        c[s][1::4] = f32(3.0e-39)
    out.append((name, c))
    # 6 huge loads (hail-like graupel, torrential rain)
    name, c = from_(base, "huge_loads")
    c["qr"] = np.where(c["qr"] > 0, f32(0.02), c["qr"]).astype(f32)
    c["qg"] = np.where(c["qg"] > 0, f32(0.03), c["qg"]).astype(f32)
    c["qs"] = np.where(c["qs"] > 0, f32(0.01), c["qs"]).astype(f32)
    c["qc"] = np.where(c["qc"] > 0, f32(0.008), c["qc"]).astype(f32)
    out.append((name, c))
    # 7 strongly supersaturated over water and ice
    name, c = from_(base, "supersaturated")
    c["qv"] = (c["qv"] * f32(1.35)).astype(f32)
    out.append((name, c))
    # 8 strongly subsaturated with condensate (fast evaporation/sublimation)
    name, c = from_(base, "subsaturated_condensate")
    c["qv"] = (c["qv"] * f32(0.2)).astype(f32)
    out.append((name, c))
    # 9 tiny rain number over large rain mass (huge drops, clamps)
    name, c = from_(base, "huge_drops")
    c["nr"] = np.where(c["qr"] > 0, f32(1.0), c["nr"]).astype(f32)
    out.append((name, c))
    # 10 huge rain number over small mass (tiny drops, clamps)
    name, c = from_(base, "tiny_drops")
    c["qr"] = np.where(c["qr"] > 0, f32(1e-7), c["qr"]).astype(f32)
    c["nr"] = np.where(c["qr"] > 0, f32(1e8), c["nr"]).astype(f32)
    out.append((name, c))
    # 11 ice number extremes
    name, c = from_(snow, "ice_number_extremes")
    c["qi"] = np.where(c["qi"] > 0, c["qi"], f32(1e-6)).astype(f32)
    c["ni"][0::2] = f32(1e8)
    c["ni"][1::2] = f32(1e-3)
    out.append((name, c))
    # 12 supercooled cloud below -38 C (homogeneous freezing)
    name, c = from_(base, "homogeneous_freeze")
    t = c["th"] * (c["p"] / f32(P0)) ** f32(RD / CP)
    c["qc"] = np.where(t < 236.0, f32(5e-4), c["qc"]).astype(f32)
    c["qr"] = np.where(t < 236.0, f32(2e-4), c["qr"]).astype(f32)
    c["nr"] = np.where(t < 236.0, f32(1e4), c["nr"]).astype(f32)
    out.append((name, c))
    # 13 zero vapour at some levels
    name, c = from_(base, "zero_vapour")
    c["qv"][::3] = f32(0)
    out.append((name, c))
    # 14 very cold column (190 K aloft, 230 K surface)
    name, c = from_(snow, "very_cold")
    c["th"] = (c["th"] - f32(35.0)).astype(f32)
    out.append((name, c))
    # 15 very hot surface (320 K) with rain
    name, c = from_(base, "very_hot")
    c["th"] = (c["th"] + np.linspace(18.0, 0.0, nz)).astype(f32)
    out.append((name, c))
    # 16 thin lowest layers (10 m) with fog and rain
    col = _Col(nz, z_sfc=50.0, p_sfc=101000.0, thin=[10.0] * 6)
    col.temperature(285.0, lapse=6.5e-3)
    col.humidity(1.0, 0.8, 0.5)
    col.layer("qc", 2e-4, 50, 120, shape="box")
    col.layer("qr", 5e-4, 50, 3000, shape="box")
    col.layer("nr", 1e4, 50, 3000, shape="box")
    out.append(("thin_layers", col.finish()))
    # 17 exactly 0 C levels with every species (melting thresholds)
    name, c = from_(base, "at_freezing_point")
    pii = (c["p"] / f32(P0)) ** f32(RD / CP)
    c["th"] = (f32(273.15) / pii).astype(f32)
    for s, v in (("qs", 5e-4), ("qg", 5e-4), ("qi", 5e-5), ("qr", 5e-4),
                 ("qc", 3e-4), ("ni", 1e5), ("nr", 1e4)):
        c[s][:] = f32(v)
    out.append((name, c))
    # 18 large vertical velocity everywhere, alternating sign
    name, c = from_(base, "extreme_w")
    c["w"] = np.resize(np.array([60.0, -40.0], dtype=f32), nz + 1)
    out.append((name, c))
    return out


REGIMES = (("convective", _convective, 12), ("stable_night", _stable_night, 8),
           ("snow", _snow, 8), ("cirrus", _cirrus, 6), ("ocean", _ocean, 8),
           ("high_terrain", _high_terrain, 8),
           ("bright_band", _bright_band, 6),
           ("freezing_rain", _freezing_rain, 6))

#: Edge columns that put NaN into the state.  They are written to a
#: separate set (``--nan``): a NaN state is not a forecast state, and the
#: classic unit's handling of it is graded apart from the main oracle.
def _nan_edge(rng, nz):
    base = _convective(np.random.default_rng(7), nz, 0)
    out = []
    for s in ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th"):
        c = {k: v.copy() for k, v in base.items()}
        c[s][nz // 4] = f32("nan")
        out.append((f"nan_{s}", c))
    return out


def stress(n=1024, nz=50, seed=20261008):
    """``n`` columns perturbed cell by cell from the regime columns: masses
    scaled by 10**U(-1, 1), numbers by 10**U(-2, 2), theta shifted by
    U(-3, 3) K, vapour scaled by U(0.7, 1.3); one in five hydrometeor cells
    zeroed.  Wider coverage of the process thresholds than the regimes
    alone, still physically shaped."""
    base = build(nz)
    rng = np.random.default_rng(seed)
    nb = base["p"].shape[0] - len(_edge(rng, nz))      # regimes only
    pick = rng.integers(0, nb, size=n)
    out = {k: base[k][pick].copy() for k in ("p", "th", "geop", "w")
           + SPECIES}
    shape = out["p"].shape
    for s in ("qc", "qr", "qi", "qs", "qg"):
        out[s] = (out[s] * 10.0 ** rng.uniform(-1, 1, shape)).astype(f32)
        out[s][rng.uniform(size=shape) < 0.2] = 0
    for s in ("ni", "nr"):
        out[s] = (out[s] * 10.0 ** rng.uniform(-2, 2, shape)).astype(f32)
    out["th"] = (out["th"] + rng.uniform(-3, 3, shape)).astype(f32)
    out["qv"] = (out["qv"] * rng.uniform(0.7, 1.3, shape)).astype(f32)
    out["labels"] = np.array([f"stress/{base['labels'][i]}" for i in pick])
    return out


def build(nz=50, seed=20261007, nan=False):
    rng = np.random.default_rng(seed)
    cols, labels = [], []
    if nan:
        for name, c in _nan_edge(rng, nz):
            cols.append(c)
            labels.append(f"nan/{name}")
    else:
        for regime, fn, count in REGIMES:
            for v in range(count):
                cols.append(fn(rng, nz, v))
                labels.append(f"{regime}/{v}")
        for name, c in _edge(rng, nz):
            cols.append(c)
            labels.append(f"edge/{name}")
    keys = ("p", "th", "geop", "w") + SPECIES
    out = {k: np.stack([c[k] for c in cols]).astype(f32) for k in keys}
    for k in SPECIES:
        bad = ~np.isfinite(out[k]) if not nan else np.zeros_like(out[k], bool)
        assert not bad.any(), k
    out["labels"] = np.array(labels)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--nz", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--nan", action="store_true",
                    help="write the NaN-state edge set instead")
    ap.add_argument("--stress", type=int, default=0,
                    help="write N perturbed stress columns instead")
    a = ap.parse_args(argv)
    out = (stress(a.stress, a.nz) if a.stress
           else build(a.nz, a.seed, nan=a.nan))
    np.savez(a.out, **out)
    print(f"wrote {a.out}: {out['p'].shape[0]} columns x {a.nz} levels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
