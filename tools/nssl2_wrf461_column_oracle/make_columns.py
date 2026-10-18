"""Deterministic NSSL option-18 column set for the WRF/WOOF bitwise oracle.

Writes one ``.npz`` (read by the GPU runner and the comparator) and one
little-endian stream file (read by the Fortran driver) holding identical
FP32 bytes.  Every column is a physically plausible atmosphere with a
hydrometeor population typical of its regime; the regimes and their edge
cases are listed in ``REGIMES`` below and recorded per column in the npz.

Stream layout (all little-endian):
  int32 nz, int32 ncol
  then, for each name in INPUT_FIELDS: ncol*nz float32 (k fastest)
  then w: ncol*(nz+1) float32 (interface, k fastest, w(1)=surface)
  then, for each name in ACCUMULATORS: ncol float32
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

NZ = 50
INPUT_FIELDS = (
    "theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
    "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
    "qvolg", "qvolh", "pressure", "exner", "rho", "dz",
)
ACCUMULATORS = ("rainnc", "snownc", "grplnc", "hailnc")

RD = 287.0
CP = 1004.5
G = 9.81
EPS = 0.622
RHO_W = 1000.0


def _esat_liquid(t):
    return 610.78 * np.exp(17.2693882 * (t - 273.16) / (t - 35.86))


def _esat_ice(t):
    return 610.78 * np.exp(21.8745584 * (t - 273.16) / (t - 7.66))


def _qsat(t, p, ice=False):
    es = np.where((t < 273.15) & ice, _esat_ice(t), _esat_liquid(t))
    es = np.minimum(es, 0.5 * p)
    return EPS * es / (p - es)


def _grid(rng, top=20000.0):
    """Stretched interface heights above ground: thin near the surface."""
    eta = np.linspace(0.0, 1.0, NZ + 1)
    a = rng.uniform(2.2, 2.8)
    z = top * (np.expm1(a * eta) / np.expm1(a)) * (1.0 - 0.15) + top * 0.15 * eta
    z[0] = 0.0
    return z


def _temperature(zm, t_sfc, lapse, z_trop, inversion=None):
    """Troposphere at ``lapse`` K/km (after an optional surface inversion of
    ``strength`` K over ``depth`` m), isothermal-to-warming stratosphere."""
    depth, strength = inversion if inversion is not None else (0.0, 0.0)

    def trop(z):
        if depth <= 0.0:
            return t_sfc - lapse * z / 1000.0
        return np.where(z < depth, t_sfc + strength * z / depth,
                        t_sfc + strength - lapse * (z - depth) / 1000.0)

    t_trop = max(float(trop(np.asarray(z_trop))), 195.0)
    t = np.maximum(trop(zm), t_trop)
    return np.where(zm > z_trop, t_trop + 1.5 * (zm - z_trop) / 1000.0, t)


def _hydrostatic(zi, zm, t, qv, p_sfc):
    """Pressure at mass levels by integrating from the surface."""
    tv = t * (1.0 + 0.608 * qv)
    p = np.empty(NZ)
    p_lower = p_sfc
    z_lower = zi[0]
    for k in range(NZ):
        dz = zm[k] - z_lower
        p[k] = p_lower * np.exp(-G * dz / (RD * tv[k]))
        p_lower, z_lower = p[k], zm[k]
    return p


def _mean_mass(diameter_m, density):
    return density * np.pi / 6.0 * diameter_m ** 3


def _layer(zm, base, top, peak, rng, ramp=600.0):
    shape = np.clip((zm - base) / ramp, 0.0, 1.0) * np.clip((top - zm) / ramp, 0.0, 1.0)
    jitter = rng.uniform(0.7, 1.3, size=zm.shape)
    return peak * shape * jitter


class Column:
    def __init__(self, rng, *, p_sfc, t_sfc, lapse, z_trop, rh, inversion=None,
                 top=20000.0):
        self.rng = rng
        self.zi = _grid(rng, top)
        self.zm = 0.5 * (self.zi[1:] + self.zi[:-1])
        self.t = _temperature(self.zm, t_sfc, lapse, z_trop, inversion)
        self.rh = rh(self.zm) if callable(rh) else np.full(NZ, rh)
        p_guess = p_sfc * np.exp(-self.zm / 8000.0)
        qv = self.rh * _qsat(self.t, p_guess, ice=True)
        self.p = _hydrostatic(self.zi, self.zm, self.t, qv, p_sfc)
        self.qv = self.rh * _qsat(self.t, self.p, ice=True)
        self.f = {name: np.zeros(NZ) for name in INPUT_FIELDS}
        self.w = np.zeros(NZ + 1)
        self.rho_g = rng.uniform(350.0, 800.0, size=NZ)
        self.rho_h = rng.uniform(600.0, 900.0, size=NZ)

    def add(self, name, mass, diameter_m, density):
        mass = np.maximum(mass, 0.0)
        self.f[name] = self.f[name] + mass
        number_name = {"qc": "qndrop", "qr": "qnr", "qi": "qni",
                       "qs": "qns", "qg": "qng", "qh": "qnh"}[name]
        diam = diameter_m * self.rng.uniform(0.6, 1.6, size=NZ)
        number = np.where(mass > 0.0, mass / _mean_mass(diam, density), 0.0)
        self.f[number_name] = self.f[number_name] + number

    def finish(self, ccn_per_kg, w_profile, *, supersat=None):
        t = self.t
        exner = (self.p / 1.0e5) ** (RD / CP)
        theta = t / exner
        qv = self.qv.copy()
        if supersat is not None:
            qv = qv * supersat
        rho = self.p / (RD * t * (1.0 + 0.608 * qv))
        f = self.f
        f["theta"], f["qv"], f["pressure"] = theta, qv, self.p
        f["exner"], f["rho"] = exner, rho
        f["dz"] = np.diff(self.zi)
        f["qvolg"] = f["qg"] / self.rho_g
        f["qvolh"] = f["qh"] / self.rho_h
        activated = f["qndrop"]
        f["qnn"] = np.maximum(ccn_per_kg - activated, 0.02 * ccn_per_kg) * \
            self.rng.uniform(0.85, 1.0, size=NZ)
        self.w = w_profile(self.zi)
        self.w[0] = 0.0
        return self


def _profile_w(peak, z_peak, depth, rng, noise=0.2):
    def w(zi):
        base = peak * np.exp(-((zi - z_peak) / depth) ** 2)
        return base + noise * rng.standard_normal(zi.shape)
    return w


def convective_core(rng):
    t_sfc = rng.uniform(298.0, 305.0)
    c = Column(rng, p_sfc=rng.uniform(96000, 100500), t_sfc=t_sfc,
               lapse=rng.uniform(6.6, 7.8), z_trop=rng.uniform(12500, 14500),
               rh=lambda z: np.where(z < 1500, 0.92, np.where(z < 11000, 0.97, 0.6)))
    zm = c.zm
    base = rng.uniform(800, 1600)
    c.add("qc", _layer(zm, base, 11000, rng.uniform(1.5e-3, 4e-3), rng), 18e-6, RHO_W)
    c.add("qr", _layer(zm, 0, 6500, rng.uniform(1e-3, 6e-3), rng, 1200), 1.2e-3, RHO_W)
    c.add("qg", _layer(zm, 3000, 11000, rng.uniform(1e-3, 5e-3), rng, 1500), 2.5e-3, 500.0)
    c.add("qh", _layer(zm, 2500, 10000, rng.uniform(5e-4, 4e-3), rng, 1500), 9e-3, 850.0)
    c.add("qs", _layer(zm, 6000, 13500, rng.uniform(5e-4, 2e-3), rng, 1500), 1.5e-3, 100.0)
    c.add("qi", _layer(zm, 7000, 14000, rng.uniform(1e-4, 8e-4), rng, 1500), 60e-6, 900.0)
    return c.finish(rng.uniform(3e8, 6e8), _profile_w(rng.uniform(15, 40),
                    rng.uniform(5000, 9000), 3500, rng, 1.0),
                    supersat=np.where(zm > base, 1.0 + rng.uniform(0.0, 0.01, NZ), 1.0))


def precip_shaft(rng):
    t_sfc = rng.uniform(295.0, 303.0)
    c = Column(rng, p_sfc=rng.uniform(95000, 100000), t_sfc=t_sfc,
               lapse=rng.uniform(6.5, 7.5), z_trop=rng.uniform(12000, 14000),
               rh=lambda z: np.where(z < 2000, rng.uniform(0.45, 0.75), 0.9))
    zm = c.zm
    c.add("qr", _layer(zm, 0, 5500, rng.uniform(2e-3, 9e-3), rng, 800), 1.8e-3, RHO_W)
    c.add("qh", _layer(zm, 0, 9000, rng.uniform(1e-3, 6e-3), rng, 800), 15e-3, 900.0)
    c.add("qg", _layer(zm, 1000, 9000, rng.uniform(1e-3, 4e-3), rng, 800), 3e-3, 400.0)
    c.add("qs", _layer(zm, 5000, 11000, rng.uniform(2e-4, 1e-3), rng, 800), 2e-3, 100.0)
    return c.finish(rng.uniform(2e8, 5e8), _profile_w(rng.uniform(-14, -4),
                    rng.uniform(1500, 3500), 2500, rng, 0.5))


def stratiform(rng):
    t_sfc = rng.uniform(280.0, 290.0)
    c = Column(rng, p_sfc=rng.uniform(98000, 102000), t_sfc=t_sfc,
               lapse=rng.uniform(5.5, 6.5), z_trop=rng.uniform(10500, 12000),
               rh=lambda z: np.where(z < 8000, 0.99, 0.7))
    zm = c.zm
    z0 = (t_sfc - 273.15) / 6.0 * 1000.0
    c.add("qs", _layer(zm, z0 - 300, 9000, rng.uniform(3e-4, 1.5e-3), rng, 800), 2.5e-3, 100.0)
    c.add("qr", _layer(zm, 0, z0 + 200, rng.uniform(2e-4, 1.2e-3), rng, 300), 0.9e-3, RHO_W)
    c.add("qi", _layer(zm, z0 + 1500, 10000, rng.uniform(2e-5, 2e-4), rng, 800), 80e-6, 900.0)
    c.add("qc", _layer(zm, 500, z0 + 800, rng.uniform(1e-4, 5e-4), rng, 400), 12e-6, RHO_W)
    c.add("qg", _layer(zm, z0 - 200, z0 + 1500, rng.uniform(0, 3e-4), rng, 300), 1.5e-3, 300.0)
    return c.finish(rng.uniform(1.5e8, 4e8), _profile_w(rng.uniform(0.2, 1.0),
                    rng.uniform(4000, 7000), 3000, rng, 0.05))


def stable_night(rng):
    t_sfc = rng.uniform(268.0, 286.0)
    c = Column(rng, p_sfc=rng.uniform(97000, 103000), t_sfc=t_sfc,
               lapse=rng.uniform(5.0, 6.5), z_trop=rng.uniform(10000, 12000),
               rh=lambda z: np.where(z < 250, 1.0, np.where(z < 3000, 0.6, 0.3)),
               inversion=(rng.uniform(150, 500), rng.uniform(3.0, 10.0)))
    zm = c.zm
    fog = rng.uniform(0, 4e-4)
    c.add("qc", _layer(zm, -100, rng.uniform(150, 300), fog, rng, 80), 8e-6, RHO_W)
    if t_sfc < 273:
        c.add("qi", _layer(zm, -100, 200, rng.uniform(0, 3e-5), rng, 80), 40e-6, 900.0)
    return c.finish(rng.uniform(2e8, 1.2e9), _profile_w(0.0, 1000, 1000, rng, 0.02),
                    supersat=np.where(zm < 200, 1.0 + rng.uniform(0, 0.004), 1.0))


def winter_snow(rng):
    t_sfc = rng.uniform(253.0, 271.0)
    c = Column(rng, p_sfc=rng.uniform(96000, 103000), t_sfc=t_sfc,
               lapse=rng.uniform(4.0, 6.0), z_trop=rng.uniform(8500, 10500),
               rh=lambda z: np.where(z < 6000, 0.98, 0.6))
    zm = c.zm
    c.add("qs", _layer(zm, 0, 7000, rng.uniform(2e-4, 1.5e-3), rng, 500), 2e-3, 100.0)
    c.add("qi", _layer(zm, 1000, 8000, rng.uniform(2e-5, 3e-4), rng, 500), 70e-6, 900.0)
    c.add("qc", _layer(zm, 300, 2500, rng.uniform(0, 3e-4), rng, 300), 14e-6, RHO_W)
    c.add("qg", _layer(zm, 0, 2500, rng.uniform(0, 4e-4), rng, 300), 1.2e-3, 300.0)
    return c.finish(rng.uniform(1e8, 4e8), _profile_w(rng.uniform(0.1, 1.5),
                    rng.uniform(1500, 4000), 2000, rng, 0.1))


def ocean_shallow(rng):
    t_sfc = rng.uniform(298.5, 302.0)
    c = Column(rng, p_sfc=rng.uniform(100800, 101600), t_sfc=t_sfc,
               lapse=rng.uniform(6.0, 6.8), z_trop=rng.uniform(15000, 16500),
               rh=lambda z: np.where(z < 600, 0.85, np.where(z < 2600, 0.98, 0.4)))
    zm = c.zm
    base = rng.uniform(500, 800)
    c.add("qc", _layer(zm, base, rng.uniform(1800, 3000), rng.uniform(3e-4, 1.5e-3), rng, 300), 22e-6, RHO_W)
    c.add("qr", _layer(zm, 0, 2500, rng.uniform(0, 8e-4), rng, 400), 0.6e-3, RHO_W)
    return c.finish(rng.uniform(4e7, 1.5e8), _profile_w(rng.uniform(1, 5),
                    rng.uniform(1000, 2000), 900, rng, 0.2),
                    supersat=np.where(zm > base, 1.0 + rng.uniform(0, 0.006, NZ), 1.0))


def high_terrain(rng):
    t_sfc = rng.uniform(262.0, 285.0)
    c = Column(rng, p_sfc=rng.uniform(62000, 74000), t_sfc=t_sfc,
               lapse=rng.uniform(6.0, 8.0), z_trop=rng.uniform(8500, 11000),
               rh=lambda z: np.where(z < 3500, 0.97, 0.5), top=16000.0)
    zm = c.zm
    c.add("qc", _layer(zm, 200, 2500, rng.uniform(1e-4, 8e-4), rng, 300), 15e-6, RHO_W)
    c.add("qs", _layer(zm, 0, 5000, rng.uniform(1e-4, 1e-3), rng, 400), 1.8e-3, 100.0)
    c.add("qi", _layer(zm, 800, 5500, rng.uniform(1e-5, 2e-4), rng, 400), 60e-6, 900.0)
    c.add("qg", _layer(zm, 0, 3000, rng.uniform(0, 6e-4), rng, 400), 1.6e-3, 450.0)
    if t_sfc > 275:
        c.add("qr", _layer(zm, 0, 800, rng.uniform(0, 5e-4), rng, 200), 0.8e-3, RHO_W)
    return c.finish(rng.uniform(1e8, 5e8), _profile_w(rng.uniform(0.5, 4.0),
                    rng.uniform(800, 2500), 1500, rng, 0.2),
                    supersat=np.where(zm < 2500, 1.0 + rng.uniform(0, 0.005, NZ), 1.0))


def anvil_cirrus(rng):
    t_sfc = rng.uniform(290.0, 300.0)
    c = Column(rng, p_sfc=rng.uniform(97000, 101000), t_sfc=t_sfc,
               lapse=rng.uniform(6.8, 7.6), z_trop=rng.uniform(13000, 15000),
               rh=lambda z: np.where(z > 9000, 1.02, 0.4))
    zm = c.zm
    c.add("qi", _layer(zm, 9000, 14500, rng.uniform(5e-5, 6e-4), rng, 800), 50e-6, 900.0)
    c.add("qs", _layer(zm, 9000, 13500, rng.uniform(5e-5, 5e-4), rng, 800), 1e-3, 100.0)
    tc = c.t - 273.15
    homog = np.where((tc < -34.0) & (tc > -45.0), rng.uniform(0, 2e-4), 0.0)
    c.add("qc", homog, 10e-6, RHO_W)
    return c.finish(rng.uniform(1e8, 4e8), _profile_w(rng.uniform(0.2, 3.0),
                    rng.uniform(10000, 12000), 2000, rng, 0.3))


def clear_air(rng):
    t_sfc = rng.uniform(270.0, 305.0)
    moist = rng.uniform() < 0.5
    c = Column(rng, p_sfc=rng.uniform(85000, 102000), t_sfc=t_sfc,
               lapse=rng.uniform(5.0, 9.0), z_trop=rng.uniform(10000, 15000),
               rh=lambda z: np.full(z.shape, rng.uniform(0.05, 0.5)))
    zm = c.zm
    supersat = None
    if moist:
        lift = rng.uniform(1000, 6000)
        supersat = np.where(np.abs(zm - lift) < 600, (1.0 + rng.uniform(0.002, 0.03)) /
                            np.maximum(c.rh, 1e-3), 1.0)
    return c.finish(rng.uniform(1e8, 8e8), _profile_w(rng.uniform(-2, 6),
                    rng.uniform(1000, 6000), 1500, rng, 0.1), supersat=supersat)


def heavy_warm_rain(rng):
    t_sfc = rng.uniform(299.0, 303.0)
    c = Column(rng, p_sfc=rng.uniform(99000, 101500), t_sfc=t_sfc,
               lapse=rng.uniform(5.5, 6.5), z_trop=rng.uniform(14500, 16000),
               rh=lambda z: np.where(z < 9000, 0.99, 0.5))
    zm = c.zm
    c.add("qr", _layer(zm, 0, 5000, rng.uniform(5e-3, 1.4e-2), rng, 500), 2.4e-3, RHO_W)
    c.add("qc", _layer(zm, 400, 6000, rng.uniform(1e-3, 3e-3), rng, 400), 20e-6, RHO_W)
    c.add("qg", _layer(zm, 5500, 9500, rng.uniform(1e-4, 1e-3), rng, 600), 2e-3, 400.0)
    c.add("qs", _layer(zm, 6500, 11000, rng.uniform(1e-4, 8e-4), rng, 600), 1.5e-3, 100.0)
    return c.finish(rng.uniform(5e7, 3e8), _profile_w(rng.uniform(3, 12),
                    rng.uniform(3000, 6000), 2500, rng, 0.5))


def giant_hail(rng):
    t_sfc = rng.uniform(298.0, 306.0)
    c = Column(rng, p_sfc=rng.uniform(88000, 96000), t_sfc=t_sfc,
               lapse=rng.uniform(7.5, 8.8), z_trop=rng.uniform(13000, 15500),
               rh=lambda z: np.where(z < 1200, 0.75, 0.95))
    zm = c.zm
    c.add("qh", _layer(zm, 0, 12000, rng.uniform(4e-3, 1.5e-2), rng, 1500), 3.5e-2, 900.0)
    c.add("qg", _layer(zm, 2000, 12000, rng.uniform(2e-3, 8e-3), rng, 1500), 4e-3, 800.0)
    c.add("qc", _layer(zm, 1500, 12000, rng.uniform(2e-3, 5e-3), rng, 800), 16e-6, RHO_W)
    c.add("qr", _layer(zm, 0, 7000, rng.uniform(2e-3, 6e-3), rng, 800), 2.2e-3, RHO_W)
    c.add("qi", _layer(zm, 8000, 15000, rng.uniform(2e-4, 1e-3), rng, 800), 40e-6, 900.0)
    c.add("qs", _layer(zm, 7000, 14000, rng.uniform(2e-4, 2e-3), rng, 800), 1e-3, 100.0)
    return c.finish(rng.uniform(4e8, 1.2e9), _profile_w(rng.uniform(35, 55),
                    rng.uniform(6000, 9000), 3000, rng, 2.0),
                    supersat=np.where(zm > 1500, 1.0 + rng.uniform(0, 0.015, NZ), 1.0))


def thresholds(rng):
    """Edge cases: trace amounts at and around the scheme's mass floors,
    mass without number, number without mass, saturation exactly at 1."""
    t_sfc = rng.uniform(275.0, 300.0)
    c = Column(rng, p_sfc=rng.uniform(90000, 101000), t_sfc=t_sfc,
               lapse=rng.uniform(6.0, 7.0), z_trop=rng.uniform(11000, 14000),
               rh=lambda z: np.full(z.shape, 1.0))
    scale = 10.0 ** rng.uniform(-15, -9, size=NZ)
    for name, d, dens in (("qc", 10e-6, RHO_W), ("qr", 0.3e-3, RHO_W),
                          ("qi", 30e-6, 900.0), ("qs", 0.5e-3, 100.0),
                          ("qg", 1e-3, 400.0), ("qh", 5e-3, 900.0)):
        c.add(name, scale * rng.uniform(0.1, 10.0, size=NZ), d, dens)
    c.finish(rng.uniform(1e8, 5e8), _profile_w(rng.uniform(-1, 3), 3000, 2000, rng, 0.3))
    f = c.f
    # Mass with zero number on a third of the levels, number without mass on
    # another third (WRF diagnoses / cleans both).
    k = np.arange(NZ)
    for name, nname in (("qc", "qndrop"), ("qr", "qnr"), ("qi", "qni"),
                        ("qs", "qns"), ("qg", "qng"), ("qh", "qnh")):
        f[nname] = np.where(k % 3 == 0, 0.0, f[nname])
        f[name] = np.where(k % 3 == 1, 0.0, f[name])
        f[nname] = np.where(k % 3 == 1, rng.uniform(0, 1e4, NZ), f[nname])
    f["qvolg"] = np.where(f["qg"] > 0, f["qvolg"], 0.0)
    f["qvolh"] = np.where(f["qh"] > 0, f["qvolh"], 0.0)
    return c


REGIMES = (
    ("convective_core", convective_core, 10),
    ("precip_shaft", precip_shaft, 8),
    ("stratiform_melting", stratiform, 8),
    ("stable_night", stable_night, 8),
    ("winter_snow", winter_snow, 8),
    ("ocean_shallow", ocean_shallow, 8),
    ("high_terrain", high_terrain, 8),
    ("anvil_cirrus", anvil_cirrus, 8),
    ("clear_air", clear_air, 8),
    ("heavy_warm_rain", heavy_warm_rain, 6),
    ("giant_hail", giant_hail, 6),
    ("thresholds", thresholds, 10),
)


def build(seed: int):
    rng = np.random.default_rng(seed)
    columns, labels = [], []
    for name, factory, count in REGIMES:
        for _ in range(count):
            columns.append(factory(rng))
            labels.append(name)
    ncol = len(columns)
    arrays = {}
    for name in INPUT_FIELDS:
        arrays[name] = np.ascontiguousarray(
            np.stack([c.f[name] for c in columns]).astype(np.float32))
    arrays["w"] = np.ascontiguousarray(
        np.stack([c.w for c in columns]).astype(np.float32))
    for name in ACCUMULATORS:
        arrays[name] = (rng.uniform(0.0, 50.0, size=ncol)
                        * (rng.uniform(size=ncol) < 0.7)).astype(np.float32)
    for name in INPUT_FIELDS + ("w",):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"non-finite {name}")
    for name in ("theta", "pressure", "exner", "rho", "dz"):
        if np.any(arrays[name] <= 0):
            raise ValueError(f"non-positive {name}")
    return arrays, labels


def write(arrays, labels, prefix: Path, seed: int):
    ncol, nz = arrays["theta"].shape
    npz_path = prefix.with_suffix(".npz")
    bin_path = prefix.with_suffix(".bin")
    np.savez(npz_path, **arrays, regime=np.asarray(labels), seed=np.int64(seed))
    with bin_path.open("wb") as out:
        out.write(np.asarray([nz, ncol], dtype="<i4").tobytes())
        for name in INPUT_FIELDS:
            out.write(arrays[name].astype("<f4").tobytes())
        out.write(arrays["w"].astype("<f4").tobytes())
        for name in ACCUMULATORS:
            out.write(arrays[name].astype("<f4").tobytes())
    digest = hashlib.sha256(bin_path.read_bytes()).hexdigest()
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    meta = {"seed": seed, "nz": nz, "ncol": ncol, "regimes": counts,
            "bin_sha256": digest}
    prefix.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    arrays, labels = build(args.seed)
    write(arrays, labels, args.prefix, args.seed)


if __name__ == "__main__":
    main()
