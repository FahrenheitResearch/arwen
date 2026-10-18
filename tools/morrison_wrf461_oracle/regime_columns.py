#!/usr/bin/env python3
"""Regime column oracle for Morrison (mp_physics=10) against WRF v4.6.1.

The stress sets (stress_columns.py) are random draws aimed at branches.
These columns are weather: each is a hydrostatic sounding on a WRF-like
stretched grid with hydrometeor layers sized from realistic particle
diameters, in twelve regimes (eight seeded variants each) plus ten edge
columns:

  supercell, mcs_stratiform, kf_seeded (cumulus-scheme tendencies),
  night_fog, night_frost, synoptic_snow, freezing_rain, marine_sc (ocean),
  tropical_ocean, high_terrain, cirrus_anvil, desert_virga,
  edge_clear, edge_subnormal, edge_negative, edge_mass_no_number,
  edge_number_no_mass, edge_hail_core, edge_qsmall, edge_stratosphere,
  edge_signed_zero, edge_supersat_homog

``make`` writes run_morrison_regimes.F90 inputs, one file per set (mode,
dt, nz, steps vary by set).  ``compare`` runs WOOF's Morrison over the same
inputs for the same number of calls and grades every returned word against
WRF's: mismatch count, max FP32 total-order ULP and max absolute difference,
per field, per regime.

usage:
  regime_columns.py make OUTDIR
  regime_columns.py compare OUTDIR [--row37-off] [--json F] [--sets a,b]
  regime_columns.py npz OUTDIR OUT.npz --sets a,b   (inputs + WRF outputs)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LEVEL_IN = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "ns", "nr",
            "ng", "rho", "pii", "p", "dz", "qrcuten", "qscuten", "qicuten")
SURF = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
        "graupelncv", "sr")
LEVEL_OUT = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "ns", "nr",
             "ng", "refl_10cm")

REGIMES = ("supercell", "mcs_stratiform", "kf_seeded", "night_fog",
           "night_frost", "synoptic_snow", "freezing_rain", "marine_sc",
           "tropical_ocean", "high_terrain", "cirrus_anvil", "desert_virga")
EDGES = ("edge_clear", "edge_subnormal", "edge_negative",
         "edge_mass_no_number", "edge_number_no_mass", "edge_hail_core",
         "edge_qsmall", "edge_stratosphere", "edge_signed_zero",
         "edge_supersat_homog")
VARIANTS = 8

#: name: (morr_rimed_ice, dt seconds, nz, consecutive calls, seed)
SETS = {
    "m0_dt20_nz50": (0, 20.0, 50, 1, 101),
    "m1_dt20_nz50": (1, 20.0, 50, 1, 102),
    "m0_dt72_nz50": (0, 72.0, 50, 1, 103),
    "m1_dt10_nz80": (1, 10.0, 80, 1, 104),
    "m0_dt60_nz80": (0, 60.0, 80, 1, 105),
    "m1_dt20_nz50_x30": (1, 20.0, 50, 30, 106),
    "m0_dt18_nz50_x30": (0, 18.0, 50, 30, 107),
}

RD, CP, G, EPS = 287.0, 1004.5, 9.81, 0.622
PI = np.pi


def column_labels():
    labels = [r for r in REGIMES for _ in range(VARIANTS)]
    return labels + list(EDGES)


# --------------------------------------------------------------- profiles

def _grid(nz, rng, terrain=False):
    if nz <= 64:
        dz0, dzmax = rng.uniform(35.0, 55.0), rng.uniform(600.0, 750.0)
    else:
        dz0, dzmax = rng.uniform(20.0, 30.0), rng.uniform(350.0, 420.0)
    if terrain:
        dzmax *= 0.85
    dz = np.minimum(dzmax, dz0 * 1.12 ** np.arange(nz))
    zf = np.concatenate([[0.0], np.cumsum(dz)])
    z = 0.5 * (zf[:-1] + zf[1:])
    return z, dz


def _esw(t):
    return 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))


def _esi(t):
    return 611.2 * np.exp(21.875 * (t - 273.15) / (t - 7.66))


def _hydrostatic(z, t, psfc, rh_fn):
    """Mid-level pressure by integrating dlnp/dz = -g/(Rd Tv)."""
    p = np.empty_like(z)
    qv = np.empty_like(z)
    pk = psfc
    zprev = 0.0
    for k in range(z.size):
        # two passes: moisture depends on p, Tv on moisture
        for _ in range(2):
            ptry = pk * np.exp(-G * (z[k] - zprev) / (RD * t[k] * (
                1.0 + 0.608 * (qv[k] if _ else 0.0))))
            es = _esw(t[k]) if t[k] >= 273.15 else _esi(t[k])
            qsat = EPS * es / max(ptry - es, 1.0)
            qv[k] = rh_fn(k) * qsat
        p[k] = ptry
        pk, zprev = ptry, z[k]
    return p, qv


def _bump(z, zb, zt):
    """0 outside [zb, zt], sin^2 peak of 1 inside."""
    s = np.clip((z - zb) / max(zt - zb, 1.0), 0.0, 1.0)
    out = np.sin(PI * s) ** 2
    return np.where((z > zb) & (z < zt), out, 0.0)


def _number(q, rho_x, d):
    """Number per kg for mass mixing ratio q at mean-mass diameter d."""
    return np.where(q > 0.0, q / (PI / 6.0 * rho_x * d ** 3), 0.0)


def _temperature(z, t0, lapse, ztrop, ttrop_floor=190.0,
                 inversion=None, nose=None):
    t = t0 - lapse * z * 1e-3
    trop = t0 - lapse * ztrop * 1e-3
    t = np.where(z > ztrop, trop + 1.0e-3 * (z - ztrop), t)
    if inversion is not None:
        depth, strength = inversion
        t = t + np.where(z < depth, strength * (z / depth) - strength, 0.0)
    if nose is not None:
        zb, zt, amp = nose
        t = t + amp * _bump(z, zb, zt)
    return np.maximum(t, ttrop_floor)


class Column:
    def __init__(self, nz):
        self.f = {name: np.zeros(nz) for name in LEVEL_IN}
        self.s = {name: 0.0 for name in SURF}


def _finish(col, z, dz, t, p, qv):
    pii = (p / 1.0e5) ** (RD / CP)
    col.f["th"] = t / pii
    col.f["qv"] = qv
    col.f["p"] = p
    col.f["pii"] = pii
    col.f["dz"] = dz
    col.f["rho"] = p / (RD * t * (1.0 + 0.608 * qv))
    return col


def _hydro(col, z, name, nname, zb, zt, peak, rho_x, d):
    q = peak * _bump(z, zb, zt)
    q = np.where(q < 1e-12 * peak, 0.0, q)
    col.f[name] = col.f[name] + q
    if nname is not None:
        col.f[nname] = col.f[nname] + _number(q, rho_x, d)


def regime(name, nz, mode, rng):
    """One column of regime ``name``."""
    u = rng.uniform
    terrain = name == "high_terrain"
    z, dz = _grid(nz, rng, terrain)
    col = Column(nz)
    rhog = 900.0 if mode == 1 else 400.0
    if name == "supercell":
        t = _temperature(z, u(300, 306), u(7.0, 8.0), u(13e3, 15e3))
        zcb = u(1200, 1800)
        rh = lambda k: (u(1.0, 1.03) if zcb < z[k] < 12e3 else u(0.65, 0.8))
        p, qv = _hydrostatic(z, t, u(95500, 98000), rh)
        _hydro(col, z, "qc", None, zcb, u(6e3, 8e3), u(0.8e-3, 3e-3), 1, 1)
        _hydro(col, z, "qr", "nr", -500, u(4e3, 6e3), u(1e-3, 6e-3), 1000.,
               u(0.8e-3, 2e-3))
        _hydro(col, z, "qg", "ng", u(2e3, 3.5e3), u(9e3, 11e3),
               u(1e-3, 8e-3), rhog, u(2e-3, 6e-3))
        _hydro(col, z, "qs", "ns", u(5e3, 7e3), u(11e3, 13e3),
               u(0.2e-3, 2e-3), 100., u(1e-3, 3e-3))
        _hydro(col, z, "qi", "ni", u(8e3, 9e3), u(13e3, 15e3),
               u(1e-5, 5e-4), 500., u(3e-5, 1e-4))
    elif name == "mcs_stratiform":
        zm = u(3.8e3, 4.5e3)
        t = _temperature(z, u(296, 301), u(6.2, 6.8), u(12.5e3, 14e3))
        rh = lambda k: u(0.97, 1.01) if z[k] > zm - 500 else u(0.8, 0.92)
        p, qv = _hydrostatic(z, t, u(97000, 100000), rh)
        _hydro(col, z, "qs", "ns", zm - 400, u(9e3, 11e3), u(0.3e-3, 1.5e-3),
               100., u(1.5e-3, 4e-3))
        _hydro(col, z, "qi", "ni", zm + 2e3, u(11e3, 12.5e3),
               u(5e-6, 1e-4), 500., u(3e-5, 8e-5))
        _hydro(col, z, "qr", "nr", -300, zm + 300, u(0.2e-3, 1.2e-3), 1000.,
               u(0.8e-3, 1.8e-3))
        _hydro(col, z, "qg", "ng", zm - 300, zm + 2.5e3, u(1e-5, 2e-4),
               rhog, u(1e-3, 2e-3))
    elif name == "kf_seeded":
        t = _temperature(z, u(297, 303), u(6.5, 7.5), u(13e3, 15e3))
        rh = lambda k: u(0.85, 0.99)
        p, qv = _hydrostatic(z, t, u(97000, 101000), rh)
        _hydro(col, z, "qc", None, u(800, 1500), u(4e3, 7e3),
               u(1e-4, 1e-3), 1, 1)
        _hydro(col, z, "qr", "nr", -200, u(3e3, 5e3), u(1e-5, 8e-4), 1000.,
               u(0.5e-3, 1.5e-3))
        _hydro(col, z, "qs", "ns", u(5e3, 6e3), u(10e3, 12e3),
               u(1e-5, 5e-4), 100., u(1e-3, 2.5e-3))
        for key, zb, zt, peak in (("qrcuten", 0.0, u(4e3, 6e3), u(1e-7, 4e-6)),
                                  ("qscuten", u(5e3, 6e3), u(10e3, 13e3),
                                   u(1e-7, 2e-6)),
                                  ("qicuten", u(7e3, 9e3), u(11e3, 14e3),
                                   u(1e-8, 1e-6))):
            col.f[key] = peak * _bump(z, zb, zt)
    elif name == "night_fog":
        depth = u(150, 400)
        t = _temperature(z, u(268, 283), u(5.5, 6.5), u(10e3, 12e3),
                         inversion=(depth, u(5, 12)))
        zfog = u(80, 260)
        rh = lambda k: 1.0 + u(0.0, 0.004) if z[k] < zfog else u(0.3, 0.55)
        p, qv = _hydrostatic(z, t, u(96000, 102000), rh)
        _hydro(col, z, "qc", None, -zfog, zfog, u(2e-5, 3e-4), 1, 1)
        if rng.uniform() < 0.5:
            _hydro(col, z, "qr", "nr", -zfog, zfog, u(1e-7, 1e-5), 1000.,
                   u(5e-5, 2e-4))
        if col.f["th"].size and t[0] < 273.15:
            _hydro(col, z, "qi", "ni", -zfog, zfog, u(1e-7, 1e-5), 500.,
                   u(1e-5, 4e-5))
    elif name == "night_frost":
        depth = u(100, 300)
        t = _temperature(z, u(255, 271), u(5.5, 6.5), u(9e3, 11e3),
                         inversion=(depth, u(6, 14)))
        rh = lambda k: u(0.95, 1.0) if z[k] < 60 else u(0.2, 0.5)
        p, qv = _hydrostatic(z, t, u(95000, 103000), rh)
        _hydro(col, z, "qi", "ni", -100, u(60, 150), u(1e-8, 2e-6), 500.,
               u(1e-5, 5e-5))
    elif name == "synoptic_snow":
        t = _temperature(z, u(255, 271.5), u(5.0, 6.5), u(8.5e3, 10e3))
        rh = lambda k: u(0.98, 1.02) if z[k] < 6e3 else u(0.5, 0.8)
        p, qv = _hydrostatic(z, t, u(98000, 102500), rh)
        _hydro(col, z, "qs", "ns", -600, u(4e3, 6e3), u(5e-5, 1.5e-3), 100.,
               u(1e-3, 4e-3))
        _hydro(col, z, "qi", "ni", u(1e3, 3e3), u(5e3, 7e3), u(5e-6, 2e-4),
               500., u(3e-5, 1e-4))
        _hydro(col, z, "qc", None, u(300, 800), u(1.2e3, 2.5e3),
               u(0, 2e-4), 1, 1)
        _hydro(col, z, "qg", "ng", -300, u(1e3, 2e3), u(0, 1e-4), rhog,
               u(0.8e-3, 2e-3))
    elif name == "freezing_rain":
        t = _temperature(z, u(266, 271.5), u(5.0, 6.0), u(9e3, 11e3),
                         nose=(u(500, 900), u(2.0e3, 3.0e3), u(8, 13)))
        rh = lambda k: u(0.95, 1.01) if z[k] < 6e3 else u(0.4, 0.7)
        p, qv = _hydrostatic(z, t, u(98000, 102500), rh)
        _hydro(col, z, "qs", "ns", u(2.3e3, 2.8e3), u(6e3, 8e3),
               u(1e-4, 1e-3), 100., u(1e-3, 3e-3))
        _hydro(col, z, "qr", "nr", -800, u(2.5e3, 3e3), u(5e-5, 6e-4),
               1000., u(0.5e-3, 1.2e-3))
        _hydro(col, z, "qg", "ng", -300, u(600, 1000), u(0, 5e-5), rhog,
               u(1e-3, 3e-3))
    elif name == "marine_sc":
        zi = u(600, 1200)
        t0 = u(287, 292)
        t = np.where(z < zi, t0 - 7.5e-3 * z, t0 - 7.5e-3 * zi + u(6, 10)
                     - 5.5e-3 * (z - zi))
        t = np.where(z > 12e3, np.maximum(t, 205.0), t)
        t = np.maximum(t, 200.0)
        zb = zi - u(250, 450)
        rh = lambda k: (1.0 + u(0, 0.003) if zb < z[k] < zi
                        else (u(0.8, 0.92) if z[k] < zi else u(0.08, 0.3)))
        p, qv = _hydrostatic(z, t, u(101500, 102500), rh)
        _hydro(col, z, "qc", None, zb, zi, u(1e-4, 6e-4), 1, 1)
        _hydro(col, z, "qr", "nr", -zi, zi, u(1e-6, 6e-5), 1000.,
               u(6e-5, 2.5e-4))
    elif name == "tropical_ocean":
        t = _temperature(z, u(299, 302), u(5.8, 6.3), u(15e3, 17e3))
        zt = u(2.5e3, 5e3)
        rh = lambda k: u(0.99, 1.01) if 600 < z[k] < zt else u(0.75, 0.88)
        p, qv = _hydrostatic(z, t, u(100800, 101300), rh)
        _hydro(col, z, "qc", None, 600, zt, u(3e-4, 2e-3), 1, 1)
        _hydro(col, z, "qr", "nr", -500, zt, u(5e-5, 2e-3), 1000.,
               u(4e-4, 1.5e-3))
        if zt > 4.8e3:
            _hydro(col, z, "qg", "ng", 4.2e3, zt + 500, u(0, 2e-4), rhog,
                   u(1e-3, 2e-3))
    elif name == "high_terrain":
        t = _temperature(z, u(262, 282), u(6.0, 7.0), u(9e3, 11e3))
        rh = lambda k: u(0.99, 1.02) if z[k] < 5e3 else u(0.3, 0.6)
        p, qv = _hydrostatic(z, t, u(62000, 75000), rh)
        _hydro(col, z, "qc", None, -200, u(1.5e3, 3e3), u(5e-5, 6e-4), 1, 1)
        _hydro(col, z, "qs", "ns", -500, u(4e3, 5.5e3), u(1e-4, 1.2e-3),
               100., u(1e-3, 3e-3))
        _hydro(col, z, "qg", "ng", -300, u(1.5e3, 3e3), u(1e-5, 8e-4), rhog,
               u(1e-3, 3e-3))
        _hydro(col, z, "qi", "ni", u(1e3, 2.5e3), u(4.5e3, 6e3),
               u(1e-6, 1e-4), 500., u(2e-5, 8e-5))
    elif name == "cirrus_anvil":
        t = _temperature(z, u(285, 300), u(6.8, 7.6), u(12e3, 15e3))
        zb, ztop = u(8.5e3, 10e3), u(11.5e3, 13.5e3)
        rh = lambda k: u(1.0, 1.4) if zb < z[k] < ztop else u(0.2, 0.6)
        p, qv = _hydrostatic(z, t, u(98000, 101500), rh)
        _hydro(col, z, "qi", "ni", zb, ztop, u(1e-6, 1e-4), 500.,
               u(1.5e-5, 6e-5))
        _hydro(col, z, "qs", "ns", zb - 1e3, ztop - 1e3, u(0, 1e-4), 100.,
               u(5e-4, 1.5e-3))
        if rng.uniform() < 0.5:
            _hydro(col, z, "qc", None, zb, zb + 800, u(1e-7, 2e-5), 1, 1)
    elif name == "desert_virga":
        t = _temperature(z, u(305, 315), u(9.3, 9.8), u(14e3, 16e3))
        zcb = u(4.3e3, 5.2e3)
        rh = lambda k: u(0.97, 1.0) if zcb < z[k] < zcb + 2e3 else u(0.07, 0.2)
        p, qv = _hydrostatic(z, t, u(84000, 90000), rh)
        _hydro(col, z, "qr", "nr", u(500, 2.5e3), zcb + 1e3, u(5e-5, 1e-3),
               1000., u(0.6e-3, 1.5e-3))
        _hydro(col, z, "qc", None, zcb, zcb + 2e3, u(1e-4, 8e-4), 1, 1)
        _hydro(col, z, "qs", "ns", zcb + 1e3, zcb + 4e3, u(1e-5, 4e-4), 100.,
               u(1e-3, 2.5e-3))
        _hydro(col, z, "qg", "ng", zcb - 500, zcb + 2.5e3, u(0, 5e-4), rhog,
               u(1.5e-3, 4e-3))
    else:
        raise ValueError(name)
    for key in ("rainnc", "snownc", "graupelnc"):
        col.s[key] = float(u(0.0, 25.0))
    return _finish(col, z, dz, t, p, qv)


def edge(name, nz, mode, rng):
    """Edge columns, each on a realistic base sounding."""
    base = regime("supercell" if name not in ("edge_stratosphere",
                                              "edge_supersat_homog",
                                              "edge_clear")
                  else "cirrus_anvil", nz, mode, rng)
    f = base.f
    hydro = ("qc", "qr", "qi", "qs", "qg", "ni", "ns", "nr", "ng")
    if name == "edge_clear":
        for key in hydro + ("qrcuten", "qscuten", "qicuten"):
            f[key][:] = 0.0
        f["qv"] *= 0.5
    elif name == "edge_subnormal":
        tiny = np.float32(1e-39)
        for key in hydro:
            f[key] = np.where(np.arange(nz) % 3 == 0, f[key],
                              tiny * (1 + np.arange(nz) % 5))
    elif name == "edge_negative":
        for key in ("qc", "qr", "qi", "qs", "qg"):
            f[key] = np.where(np.arange(nz) % 4 == 1,
                              -rng.uniform(1e-12, 1e-8, nz), f[key])
        for key in ("ni", "ns", "nr", "ng"):
            f[key] = np.where(np.arange(nz) % 7 == 3,
                              -rng.uniform(1e-3, 10.0, nz), f[key])
    elif name == "edge_mass_no_number":
        for key in ("ni", "ns", "nr", "ng"):
            f[key][:] = 0.0
    elif name == "edge_number_no_mass":
        for q, n in (("qr", "nr"), ("qi", "ni"), ("qs", "ns"), ("qg", "ng")):
            f[n] = np.where(f[q] > 0, f[n], rng.uniform(1.0, 1e5, nz))
            f[q][:] = 0.0
    elif name == "edge_hail_core":
        f["qg"] = f["qg"] * (20e-3 / max(f["qg"].max(), 1e-12))
        f["qr"] = f["qr"] * (10e-3 / max(f["qr"].max(), 1e-12))
        f["ng"] = _number(f["qg"], 900.0 if mode == 1 else 400.0, 8e-3)
    elif name == "edge_qsmall":
        qsmall = np.float32(1e-14)
        lo = np.nextafter(qsmall, np.float32(0))
        hi = np.nextafter(qsmall, np.float32(1))
        pattern = np.asarray([lo, qsmall, hi, 0.0], dtype=np.float32)
        for key in ("qc", "qr", "qi", "qs", "qg"):
            f[key] = np.where(f[key] > 0, f[key],
                              pattern[np.arange(nz) % 4].astype(np.float64))
        for key in ("ni", "ns", "nr", "ng"):
            f[key] = np.where(f[key] > 0, f[key], 1.0)
    elif name == "edge_stratosphere":
        th_floor = 180.0
        t = np.maximum(f["th"] * f["pii"] - 15.0, th_floor)
        f["th"] = t / f["pii"]
        f["qi"] = np.where(t < 205, 1e-7, f["qi"])
        f["ni"] = np.where(t < 205, 1e4, f["ni"])
    elif name == "edge_signed_zero":
        for key in hydro + ("qrcuten", "qscuten", "qicuten"):
            f[key] = np.where(f[key] == 0.0, -0.0, f[key])
    elif name == "edge_supersat_homog":
        t = f["th"] * f["pii"]
        cold = (t < 235.0) & (t > 215.0)
        f["qv"] = np.where(cold, f["qv"] * 1.6, f["qv"])
        f["qc"] = np.where(cold, 5e-6, f["qc"])
    else:
        raise ValueError(name)
    return base


def make_set(nz, mode, seed):
    rng = np.random.default_rng(seed)
    cols = []
    for name in REGIMES:
        for _ in range(VARIANTS):
            cols.append(regime(name, nz, mode, rng))
    for name in EDGES:
        cols.append(edge(name, nz, mode, rng))
    ncol = len(cols)
    fields = {k: np.empty((nz, ncol), np.float32) for k in LEVEL_IN}
    surf = {k: np.zeros(ncol, np.float32) for k in SURF}
    for i, c in enumerate(cols):
        for k in LEVEL_IN:
            fields[k][:, i] = np.asarray(c.f[k], dtype=np.float32)
        for k in SURF:
            surf[k][i] = np.float32(c.s[k])
    # Signed zero must survive the float32 cast
    return fields, surf


def write_input(path, fields, surf, mode, dt):
    nz, ncol = fields["th"].shape
    with open(path, "wb") as out:
        np.asarray([ncol, nz, mode], dtype="<i4").tofile(out)
        np.asarray([dt], dtype="<f4").tofile(out)
        for name in LEVEL_IN:
            fields[name].astype("<f4").tofile(out)
        for name in SURF:
            surf[name].astype("<f4").tofile(out)


def read_input(path):
    raw = np.fromfile(path, dtype="<i4", count=3)
    ncol, nz, mode = (int(x) for x in raw)
    dt = np.float32(np.fromfile(path, dtype="<f4", count=1, offset=12)[0])
    data = np.fromfile(path, dtype="<f4", offset=16)
    fields, pos = {}, 0
    for name in LEVEL_IN:
        fields[name] = data[pos:pos + nz * ncol].reshape(nz, ncol).copy()
        pos += nz * ncol
    surf = {}
    for name in SURF:
        surf[name] = data[pos:pos + ncol].copy()
        pos += ncol
    return ncol, nz, mode, dt, fields, surf


def read_output(path, ncol, nz):
    data = np.fromfile(path, dtype="<f4")
    out, pos = {}, 0
    for name in LEVEL_OUT:
        out[name] = data[pos:pos + nz * ncol].reshape(nz, ncol)
        pos += nz * ncol
    for name in SURF:
        out[name] = data[pos:pos + ncol]
        pos += ncol
    return out


# ------------------------------------------------------------------ GPU side

def run_gpu(ncol, nz, mode, dt, fields, surf, steps, columns_per_call=None):
    import cupy as cp
    from gpuwm.core import morrison
    from gpuwm.core.refl import launch_refl10cm_morrison

    def dev(a):
        return cp.asarray(np.ascontiguousarray(a).reshape(nz, 1, ncol))

    d = {name: dev(fields[name]) for name in LEVEL_IN}
    nc = cp.zeros((nz, 1, ncol), dtype=cp.float32)
    s = {name: cp.asarray(np.ascontiguousarray(surf[name]).reshape(1, ncol))
         for name in SURF}
    work = None
    chunked = columns_per_call is not None and columns_per_call < ncol
    t1d = cp.empty_like(d["th"]) if chunked else None
    extra = ({"columns_per_call": columns_per_call, "t1d_out": t1d}
             if chunked else {})
    for _ in range(steps):
        work = morrison.launch_morrison(
            d["th"], d["qv"], d["qc"], d["qr"], d["qi"], d["qs"], d["qg"],
            nc, d["nr"], d["ni"], d["ns"], d["ng"], d["rho"], d["pii"],
            d["p"], d["dz"], s["rainnc"], s["rainncv"], s["snownc"],
            s["snowncv"], s["graupelnc"], s["graupelncv"], s["sr"],
            float(dt), qrcuten=d["qrcuten"], qscuten=d["qscuten"],
            qicuten=d["qicuten"], morr_rimed_ice=mode, **extra)
    if chunked:
        temperature_source = "t1d_out of the column calls (WRF's t1d)"
    elif work is not None and hasattr(morrison, "WORK_T3D"):
        k = morrison.WORK_T3D
        t1d = cp.ascontiguousarray(work[k * nz:(k + 1) * nz])
        temperature_source = "process T3D (WRF's t1d)"
    else:
        # Pre-transcription API returns nothing; the closest stand-in for
        # WRF's t1d is theta * pii after the call.
        t1d = d["th"] * d["pii"]
        temperature_source = "theta*pii stand-in"
    refl = cp.empty_like(d["th"])
    launch_refl10cm_morrison(d["qv"], d["qr"], d["nr"], d["qs"], d["ns"],
                             d["qg"], d["ng"], t1d, d["p"], refl,
                             morr_rimed_ice=mode)
    out = {name: cp.asnumpy(d[name]).reshape(nz, ncol)
           for name in LEVEL_OUT if name != "refl_10cm"}
    out["refl_10cm"] = cp.asnumpy(refl).reshape(nz, ncol)
    for name in SURF:
        out[name] = cp.asnumpy(s[name]).reshape(ncol)
    return out, temperature_source


def _row37_off():
    import cupy as cp
    from gpuwm.core import kernels
    original = kernels.load_module

    def load(name):
        if name != "morrison":
            return original(name)
        module = cp.RawModule(
            code="#define MORR_ROW37_OFF 1" + chr(10)
            + kernels.module_source("morrison"),
            options=kernels.module_options("morrison"))
        module.compile()
        return module

    kernels.load_module = load
    kernels.get_kernel.cache_clear()


def grade(got, want, labels):
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    report = {"fields": {}, "regimes": {}}
    labels = np.asarray(labels)
    bad_total = total = 0
    regime_bad = {r: 0 for r in dict.fromkeys(labels)}
    for name in LEVEL_OUT + SURF:
        g = np.ascontiguousarray(got[name], np.float32)
        w = np.ascontiguousarray(want[name], np.float32)
        bad = (g.view(np.uint32) != w.view(np.uint32)) & ~(
            np.isnan(g) & np.isnan(w))
        ulp = fp32_ulp_distance(g.ravel(), w.ravel()).reshape(g.shape)
        with np.errstate(invalid="ignore", over="ignore"):
            diff = np.abs(g.astype(np.float64) - w.astype(np.float64))
        diff = np.where(bad, diff, 0.0)
        colbad = bad.any(axis=0) if bad.ndim == 2 else bad
        for r in regime_bad:
            regime_bad[r] += int((bad if bad.ndim == 1 else bad.T)[
                labels == r].sum())
        report["fields"][name] = {
            "mismatches": int(bad.sum()), "values": int(g.size),
            "max_ulp": int(ulp[bad].max()) if bad.any() else 0,
            "max_abs": float(np.nanmax(diff)) if bad.any() else 0.0,
            "columns": sorted({str(labels[i]) for i in
                               np.nonzero(colbad)[0]}),
        }
        bad_total += int(bad.sum())
        total += int(g.size)
    report["regimes"] = regime_bad
    report["overall"] = {"mismatches": bad_total, "values": total,
                         "bitwise": bad_total == 0,
                         "max_ulp": max(v["max_ulp"] for v in
                                        report["fields"].values())}
    return report


def export_npz(outdir, out_path, names):
    """Inputs, WRF -O0 outputs and labels of the named sets, for a test."""
    meta = json.loads((outdir / "sets.json").read_text())
    arrays = {"labels": np.asarray(meta["labels"])}
    for index, name in enumerate(names):
        info = meta["sets"][name]
        ncol, nz, mode, dt, fields, surf = read_input(outdir / f"{name}.in.bin")
        want = read_output(outdir / f"{name}.wrf.bin", ncol, nz)
        arrays[f"s{index}_name"] = np.asarray(name)
        arrays[f"s{index}_meta"] = np.asarray([ncol, nz, mode, info["steps"]],
                                              np.int32)
        arrays[f"s{index}_dt"] = np.asarray([dt], np.float32)
        for key in LEVEL_IN:
            arrays[f"s{index}_in_{key}"] = fields[key]
        for key in SURF:
            arrays[f"s{index}_in_{key}"] = surf[key]
        for key in LEVEL_OUT + SURF:
            arrays[f"s{index}_out_{key}"] = np.ascontiguousarray(want[key])
    np.savez_compressed(out_path, **arrays)


def load_npz(path):
    """Yield (name, ncol, nz, mode, steps, dt, fields, surf, want), labels."""
    data = np.load(path)
    labels = [str(x) for x in data["labels"]]
    sets = []
    index = 0
    while f"s{index}_meta" in data:
        ncol, nz, mode, steps = (int(x) for x in data[f"s{index}_meta"])
        dt = np.float32(data[f"s{index}_dt"][0])
        fields = {k: data[f"s{index}_in_{k}"] for k in LEVEL_IN}
        surf = {k: data[f"s{index}_in_{k}"] for k in SURF}
        want = {k: data[f"s{index}_out_{k}"] for k in LEVEL_OUT + SURF}
        sets.append((str(data[f"s{index}_name"]), ncol, nz, mode, steps, dt,
                     fields, surf, want))
        index += 1
    return sets, labels


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("outdir", type=Path)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("outdir", type=Path)
    cmp_.add_argument("--row37-off", action="store_true")
    cmp_.add_argument("--json", type=Path)
    cmp_.add_argument("--sets", default="")
    cmp_.add_argument("--columns-per-call", type=int, default=None,
                      help="split each set into column calls of this size "
                           "(the forecast path's chunking)")
    cmp_.add_argument("--wrf-suffix", default="wrf",
                      help="read <set>.<suffix>.bin (wrf = -O0 reference, "
                           "wrfO2 = WRF's stock -O2 vectorised flags)")
    npz = sub.add_parser("npz")
    npz.add_argument("outdir", type=Path)
    npz.add_argument("out", type=Path)
    npz.add_argument("--sets", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "npz":
        export_npz(args.outdir, args.out, args.sets.split(","))
        return 0
    if args.cmd == "make":
        args.outdir.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for name, (mode, dt, nz, steps, seed) in SETS.items():
            fields, surf = make_set(nz, mode, seed)
            write_input(args.outdir / f"{name}.in.bin", fields, surf, mode,
                        dt)
            manifest[name] = {"mode": mode, "dt": dt, "nz": nz,
                              "steps": steps, "seed": seed,
                              "columns": fields["th"].shape[1]}
        (args.outdir / "sets.json").write_text(json.dumps(
            {"sets": manifest, "labels": column_labels()}, indent=2))
        print(json.dumps(manifest, indent=2))
        return 0
    if args.row37_off:
        _row37_off()
    meta = json.loads((args.outdir / "sets.json").read_text())
    labels = meta["labels"]
    wanted = [s for s in args.sets.split(",") if s] or list(meta["sets"])
    from gpuwm import wrf_exact
    reports = {"strict": wrf_exact.ENABLED, "row37": not args.row37_off,
               "columns_per_call": args.columns_per_call,
               "wrf_suffix": args.wrf_suffix, "sets": {}}
    for name in wanted:
        steps = meta["sets"][name]["steps"]
        ncol, nz, mode, dt, fields, surf = read_input(
            args.outdir / f"{name}.in.bin")
        want = read_output(args.outdir / f"{name}.{args.wrf_suffix}.bin",
                           ncol, nz)
        got, tsrc = run_gpu(ncol, nz, mode, dt, fields, surf, steps,
                            args.columns_per_call)
        rep = grade(got, want, labels)
        rep["refl_temperature"] = tsrc
        rep.update(meta["sets"][name])
        reports["sets"][name] = rep
        o = rep["overall"]
        worst = sorted(((v["max_ulp"], k) for k, v in rep["fields"].items()
                        if v["mismatches"]), reverse=True)[:4]
        print(f"{name}: {o['mismatches']} of {o['values']} differ, "
              f"max_ulp={o['max_ulp']} bitwise={o['bitwise']} worst={worst}",
              flush=True)
    all_bad = sum(r["overall"]["mismatches"] for r in reports["sets"].values())
    all_n = sum(r["overall"]["values"] for r in reports["sets"].values())
    reports["overall"] = {"mismatches": all_bad, "values": all_n,
                          "bitwise": all_bad == 0,
                          "max_ulp": max(r["overall"]["max_ulp"] for r in
                                         reports["sets"].values())}
    print(f"ALL: {all_bad} of {all_n} differ; max_ulp="
          f"{reports['overall']['max_ulp']}; strict={wrf_exact.ENABLED} "
          f"row37={not args.row37_off}")
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
