"""Column set for the TEMPO column oracle (GPU port against TEMPO's Fortran).

Writes ``columns_in.bin`` (read by tempo_oracle_driver.F90 and by
run_woof_oracle.py) and ``columns_catalog.json`` (one row per column: set,
index, regime, variant, short description).

The columns are built, not sampled from a model: each regime template sets a
temperature profile (surface value, lapse rates, inversions, warm noses,
tropopause), a surface pressure (sea level to high terrain), a relative
humidity profile, a vertical velocity profile, aerosol profiles and
hydrometeor layers, with number concentrations tied to the masses through a
mean particle size.  Pressure is hydrostatic from the virtual temperature.
A seeded generator draws variants of every template, so the set is the same
on every machine.

Binary layout (little-endian):
  int64 nsets
  per set: int64 ncol, int64 nz, int64 nsteps, float64 dt,
           then 17 fields in FIELDS order, each float64[ncol][nz]
           (column-major over columns: column c's nz values are contiguous,
           level 1 is the lowest).

Fields: t p w dz qv qc qr qi qs qg ni nr nc nwfa nifa ng qb, in the units
TEMPO's tempo_run takes (K, Pa, m/s, m, kg/kg, numbers per kg, qb in L/kg).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

FIELDS = ("t", "p", "w", "dz", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc",
          "nwfa", "nifa", "ng", "qb")
NZ = 60
G = 9.80665
RD = 287.04
EPS = 0.622


def levels():
    """dz (m) per level and mid-level heights above ground: 25 m at the
    ground stretching 8 % per level to a 600 m cap, top near 18.6 km."""
    dz = np.empty(NZ)
    d = 25.0
    for k in range(NZ):
        dz[k] = d
        d = min(d * 1.08, 600.0)
    zf = np.concatenate([[0.0], np.cumsum(dz)])
    zm = 0.5 * (zf[1:] + zf[:-1])
    return dz, zm


DZ, ZM = levels()


def esat_w(t):
    """Saturation vapour pressure over water (Pa), Bolton (1980)."""
    tc = t - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def esat_i(t):
    tc = t - 273.15
    return 611.15 * np.exp(22.452 * tc / (tc + 272.55))


def qsat(t, p, ice=False):
    e = np.where((t < 273.15) & ice, esat_i(t), esat_w(t))
    e = np.minimum(e, 0.5 * p)
    return EPS * e / (p - e)


def temperature(ts, lapse_low, lapse_high, z_break, z_tp, inv_depth=0.0, inv_dt=0.0,
                nose_z=None, nose_dt=0.0, nose_w=500.0, t_strat_rate=1.5e-3):
    """Temperature (K) at ZM.  A surface inversion of inv_dt K over inv_depth
    m sits under the lapse-rate profile; a warm nose adds a Gaussian bump."""
    z = ZM
    t = np.where(z < z_break, ts - lapse_low * z,
                 ts - lapse_low * z_break - lapse_high * (z - z_break))
    t_tp = (ts - lapse_low * z_break - lapse_high * (z_tp - z_break)
            if z_tp > z_break else ts - lapse_low * z_tp)
    t = np.where(z > z_tp, t_tp + t_strat_rate * (z - z_tp), t)
    if inv_depth > 0.0:
        # surface is inv_dt colder than the free profile and warms to it at inv_depth
        frac = np.clip(1.0 - z / inv_depth, 0.0, 1.0)
        t = t - inv_dt * frac
    if nose_z is not None:
        t = t + nose_dt * np.exp(-0.5 * ((z - nose_z) / nose_w) ** 2)
    return t


def pressure(t, qv, psfc):
    """Hydrostatic mid-level pressure from psfc with virtual temperature."""
    tv = t * (1.0 + 0.608 * qv)
    p = np.empty(NZ)
    pf = psfc
    for k in range(NZ):
        # half layer to the mid level, then the other half to the next face
        p[k] = pf * math.exp(-G * 0.5 * DZ[k] / (RD * tv[k]))
        pf = p[k] * math.exp(-G * 0.5 * DZ[k] / (RD * tv[k]))
    return p


def layer(z0, z1, peak, taper=300.0):
    """Smooth top-hat between z0 and z1 (m), peak value, cosine edges."""
    z = ZM
    f = np.zeros(NZ)
    inside = (z >= z0) & (z <= z1)
    f[inside] = 1.0
    lo = (z >= z0 - taper) & (z < z0)
    hi = (z > z1) & (z <= z1 + taper)
    f[lo] = 0.5 * (1.0 + np.cos(np.pi * (z0 - z[lo]) / taper))
    f[hi] = 0.5 * (1.0 + np.cos(np.pi * (z[hi] - z1) / taper))
    return peak * f


def number_from_mass(q, d_mean, rho_p):
    """Number per kg for mass q (kg/kg) of spheres of diameter d_mean (m)."""
    m = math.pi / 6.0 * rho_p * d_mean ** 3
    return np.where(q > 0.0, q / m, 0.0)


class Builder:
    def __init__(self, rng):
        self.rng = rng

    def j(self, x, rel=0.15):
        """jitter a template value by a uniform relative factor"""
        return x * (1.0 + rel * (2.0 * self.rng.random() - 1.0))

    def column(self, *, ts, psfc, lapse_low=6.5e-3, lapse_high=7.0e-3, z_break=3000.0,
               z_tp=12000.0, inv_depth=0.0, inv_dt=0.0, nose_z=None, nose_dt=0.0,
               nose_w=500.0, rh_low=0.7, rh_mid=0.5, rh_high=0.3, rh_cloud=None, w=None,
               qc=None, qr=None, qi=None, qs=None, qg=None,
               dc=15e-6, dr=1.0e-3, di=80e-6, dg=3.0e-3, rho_g=400.0,
               nwfa_sfc=1.5e9, nwfa_h=2000.0, nifa_sfc=1.5e6, nifa_h=3000.0,
               ice_rh=False):
        t = temperature(ts, lapse_low, lapse_high, z_break, z_tp, inv_depth, inv_dt,
                        nose_z, nose_dt, nose_w)
        z = ZM
        rh = np.interp(z, [0.0, 2000.0, 6000.0, 12000.0, 20000.0],
                       [rh_low, rh_low, rh_mid, rh_high, 0.05])
        if rh_cloud is not None:
            for (z0, z1, val) in rh_cloud:
                inside = (z >= z0) & (z <= z1)
                rh = np.where(inside, val, rh)
        # first guess pressure with dry air, then moisture, then pressure again
        p = pressure(t, np.zeros(NZ), psfc)
        qv = rh * qsat(t, p, ice=ice_rh)
        p = pressure(t, qv, psfc)
        qv = rh * qsat(t, p, ice=ice_rh)
        zero = np.zeros(NZ)
        qc = zero.copy() if qc is None else qc
        qr = zero.copy() if qr is None else qr
        qi = zero.copy() if qi is None else qi
        qs = zero.copy() if qs is None else qs
        qg = zero.copy() if qg is None else qg
        w = zero.copy() if w is None else w
        nc = number_from_mass(qc, dc, 1000.0)
        nr = number_from_mass(qr, dr, 1000.0)
        ni = number_from_mass(qi, di, 890.0)
        rg = np.full(NZ, rho_g) if np.isscalar(rho_g) else rho_g
        ng = np.where(qg > 0.0, qg / (math.pi / 6.0 * rg * dg ** 3), 0.0)
        qb = np.where(qg > 0.0, qg * 1000.0 / rg, 0.0)
        nwfa = 0.1 * nwfa_sfc + nwfa_sfc * np.exp(-z / nwfa_h)
        nifa = 0.2 * nifa_sfc + nifa_sfc * np.exp(-z / nifa_h)
        return dict(t=t, p=p, w=w, dz=DZ.copy(), qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg,
                    ni=ni, nr=nr, nc=nc, nwfa=nwfa, nifa=nifa, ng=ng, qb=qb)


def updraft(z0, z1, wmax):
    z = ZM
    f = np.where((z > z0) & (z < z1), np.sin(np.pi * (z - z0) / (z1 - z0)), 0.0)
    return wmax * f


def melting_height(t):
    above = np.nonzero(t < 273.15)[0]
    return ZM[above[0]] if above.size else 20000.0


# --------------------------------------------------------------------------
# regime templates: each returns one column for a Builder b
# --------------------------------------------------------------------------
def conv_supercell(b):
    ts = b.j(303.0, 0.01)
    t = temperature(ts, 7.5e-3, 7.5e-3, 3000.0, 13000.0)
    zm = melting_height(t)
    return b.column(ts=ts, psfc=b.j(97000.0, 0.01), lapse_low=7.5e-3, lapse_high=7.5e-3,
                    z_tp=13000.0, rh_low=0.85, rh_mid=0.6, rh_high=0.4,
                    rh_cloud=[(1200.0, 11000.0, b.j(1.005, 0.003))],
                    w=updraft(800.0, 13000.0, b.j(30.0, 0.3)),
                    qc=layer(1200.0, 9000.0, b.j(3.0e-3, 0.3)),
                    qr=layer(0.0, zm + 1500.0, b.j(5.0e-3, 0.3)),
                    qg=layer(2500.0, 10000.0, b.j(8.0e-3, 0.4)),
                    qs=layer(8000.0, 12500.0, b.j(1.0e-3, 0.4)),
                    qi=layer(9000.0, 13000.0, b.j(5.0e-4, 0.4)),
                    dr=b.j(1.8e-3, 0.2), dg=b.j(8.0e-3, 0.4), rho_g=b.j(820.0, 0.08),
                    dc=b.j(18e-6, 0.2), nwfa_sfc=b.j(2.0e9, 0.4))


def conv_mcs_stratiform(b):
    ts = b.j(297.0, 0.01)
    t = temperature(ts, 6.0e-3, 6.8e-3, 3000.0, 12500.0)
    zm = melting_height(t)
    return b.column(ts=ts, psfc=b.j(98500.0, 0.01), lapse_low=6.0e-3, lapse_high=6.8e-3,
                    z_tp=12500.0, rh_low=0.9, rh_mid=0.95, rh_high=0.7,
                    rh_cloud=[(zm, 9000.0, b.j(1.0, 0.002))],
                    w=updraft(zm, 10000.0, b.j(0.6, 0.4)),
                    qr=layer(0.0, zm, b.j(1.2e-3, 0.4)),
                    qs=layer(zm - 200.0, 9000.0, b.j(1.5e-3, 0.4)),
                    qg=layer(zm - 300.0, zm + 1500.0, b.j(3.0e-4, 0.5)),
                    qi=layer(7000.0, 10500.0, b.j(1.5e-4, 0.4)),
                    dr=b.j(1.2e-3, 0.3), rho_g=b.j(300.0, 0.3), dg=b.j(2.0e-3, 0.3))


def conv_pulse(b):
    ts = b.j(305.0, 0.01)
    t = temperature(ts, 8.0e-3, 7.0e-3, 2500.0, 14000.0)
    zm = melting_height(t)
    return b.column(ts=ts, psfc=b.j(100500.0, 0.005), lapse_low=8.0e-3, lapse_high=7.0e-3,
                    z_break=2500.0, z_tp=14000.0, rh_low=0.75, rh_mid=0.55, rh_high=0.3,
                    rh_cloud=[(1500.0, 9000.0, b.j(1.003, 0.003))],
                    w=updraft(1500.0, 10000.0, b.j(12.0, 0.4)),
                    qc=layer(1500.0, 8500.0, b.j(2.0e-3, 0.4)),
                    qr=layer(500.0, zm + 2000.0, b.j(2.5e-3, 0.4)),
                    qg=layer(zm, 9000.0, b.j(2.0e-3, 0.5)),
                    qi=layer(8500.0, 11000.0, b.j(2.0e-4, 0.5)),
                    rho_g=b.j(500.0, 0.3), dg=b.j(3.0e-3, 0.3))


def conv_elevated_hail(b):
    ts = b.j(299.0, 0.01)
    t = temperature(ts, 9.0e-3, 7.5e-3, 2000.0, 12500.0)
    zm = melting_height(t)
    w = updraft(2000.0, 12000.0, b.j(25.0, 0.3)) + updraft(0.0, 2000.0, -b.j(6.0, 0.3))
    return b.column(ts=ts, psfc=b.j(96000.0, 0.01), lapse_low=9.0e-3, lapse_high=7.5e-3,
                    z_break=2000.0, z_tp=12500.0, rh_low=0.35, rh_mid=0.7, rh_high=0.4,
                    rh_cloud=[(2500.0, 10000.0, b.j(1.004, 0.003))], w=w,
                    qc=layer(2500.0, 9000.0, b.j(2.5e-3, 0.3)),
                    qr=layer(0.0, zm + 1000.0, b.j(2.0e-3, 0.4)),
                    qg=layer(0.0, 10000.0, b.j(1.2e-2, 0.4)),
                    dg=b.j(1.5e-2, 0.3), rho_g=b.j(880.0, 0.03), nwfa_sfc=b.j(3.0e9, 0.3))


def stable_night_clear(b):
    ts = b.j(283.0, 0.02)
    return b.column(ts=ts, psfc=b.j(99500.0, 0.01), inv_depth=b.j(300.0, 0.3),
                    inv_dt=b.j(8.0, 0.4), rh_low=b.j(0.6, 0.3), rh_mid=0.35, rh_high=0.2,
                    nwfa_sfc=b.j(2.5e9, 0.4), nwfa_h=600.0)


def stable_night_fog(b):
    ts = b.j(280.0, 0.02)
    return b.column(ts=ts, psfc=b.j(100800.0, 0.005), inv_depth=b.j(250.0, 0.3),
                    inv_dt=b.j(5.0, 0.3), rh_low=0.85, rh_mid=0.4, rh_high=0.2,
                    rh_cloud=[(0.0, b.j(180.0, 0.3), b.j(1.004, 0.003))],
                    qc=layer(0.0, 150.0, b.j(2.5e-4, 0.4), taper=60.0),
                    qr=layer(0.0, 120.0, b.j(1.0e-6, 0.8), taper=60.0),
                    dc=b.j(10e-6, 0.2), dr=b.j(0.15e-3, 0.3), w=np.full(NZ, b.j(0.01, 1.0)),
                    nwfa_sfc=b.j(4.0e9, 0.3), nwfa_h=500.0)


def stable_stratus_drizzle(b):
    ts = b.j(286.0, 0.01)
    top = b.j(1000.0, 0.2)
    return b.column(ts=ts, psfc=b.j(101500.0, 0.005), lapse_low=9.0e-3, z_break=top,
                    lapse_high=6.0e-3, rh_low=0.92, rh_mid=0.3, rh_high=0.2,
                    rh_cloud=[(top - 600.0, top, 1.001)],
                    qc=layer(top - 600.0, top, b.j(3.5e-4, 0.3), taper=100.0),
                    qr=layer(0.0, top - 100.0, b.j(2.0e-5, 0.5), taper=150.0),
                    w=updraft(0.0, top, b.j(0.3, 0.5)),
                    dc=b.j(14e-6, 0.2), dr=b.j(0.2e-3, 0.3), nwfa_sfc=b.j(8.0e8, 0.4))


def winter_snow(b):
    ts = b.j(266.0, 0.01)
    return b.column(ts=ts, psfc=b.j(99000.0, 0.01), lapse_low=5.5e-3, lapse_high=6.5e-3,
                    z_tp=10000.0, rh_low=0.98, rh_mid=0.95, rh_high=0.6, ice_rh=True,
                    rh_cloud=[(500.0, 5000.0, b.j(1.02, 0.01))],
                    w=updraft(0.0, 6000.0, b.j(0.5, 0.5)),
                    qs=layer(0.0, 6000.0, b.j(1.2e-3, 0.4)),
                    qi=layer(2000.0, 7000.0, b.j(1.0e-4, 0.5)),
                    qc=layer(800.0, 1800.0, b.j(1.5e-4, 0.6)),
                    qg=layer(0.0, 2000.0, b.j(1.0e-4, 0.6)), rho_g=b.j(200.0, 0.4),
                    dg=b.j(1.5e-3, 0.3), di=b.j(100e-6, 0.3), nwfa_sfc=b.j(1.0e9, 0.4))


def freezing_rain(b):
    ts = b.j(270.5, 0.003)
    return b.column(ts=ts, psfc=b.j(100000.0, 0.01), inv_depth=b.j(1500.0, 0.2),
                    inv_dt=b.j(10.0, 0.2), lapse_low=5.0e-3, lapse_high=6.5e-3,
                    nose_z=b.j(1600.0, 0.15), nose_dt=b.j(4.0, 0.3), nose_w=400.0,
                    rh_low=0.97, rh_mid=0.95, rh_high=0.6,
                    rh_cloud=[(2500.0, 5000.0, 1.002)],
                    qr=layer(0.0, 2200.0, b.j(6.0e-4, 0.4)),
                    qs=layer(2000.0, 6000.0, b.j(8.0e-4, 0.4)),
                    qc=layer(2500.0, 4000.0, b.j(1.0e-4, 0.5)),
                    dr=b.j(1.0e-3, 0.3), w=updraft(1000.0, 6000.0, b.j(0.4, 0.5)))


def sleet(b):
    ts = b.j(268.0, 0.005)
    return b.column(ts=ts, psfc=b.j(100200.0, 0.01), inv_depth=b.j(2000.0, 0.2),
                    inv_dt=b.j(12.0, 0.2), lapse_low=4.5e-3, lapse_high=6.5e-3,
                    nose_z=b.j(2300.0, 0.1), nose_dt=b.j(2.5, 0.3), nose_w=250.0,
                    rh_low=0.96, rh_mid=0.95, rh_high=0.5,
                    qr=layer(1000.0, 2600.0, b.j(5.0e-4, 0.4)),
                    qg=layer(0.0, 1500.0, b.j(3.0e-4, 0.4)), rho_g=b.j(850.0, 0.05),
                    dg=b.j(1.5e-3, 0.3),
                    qs=layer(2400.0, 6000.0, b.j(6.0e-4, 0.4)))


def lake_effect(b):
    ts = b.j(276.0, 0.005)
    return b.column(ts=ts, psfc=b.j(99800.0, 0.005), lapse_low=9.0e-3, lapse_high=5.0e-3,
                    z_break=2500.0, z_tp=9000.0, rh_low=0.95, rh_mid=0.6, rh_high=0.4,
                    rh_cloud=[(400.0, 2500.0, b.j(1.01, 0.005))], ice_rh=True,
                    w=updraft(0.0, 3000.0, b.j(2.5, 0.4)),
                    qc=layer(400.0, 2000.0, b.j(4.0e-4, 0.4)),
                    qs=layer(0.0, 2800.0, b.j(1.5e-3, 0.4)),
                    qg=layer(0.0, 2000.0, b.j(4.0e-4, 0.5)), rho_g=b.j(250.0, 0.3),
                    qi=layer(1500.0, 3000.0, b.j(1.0e-4, 0.5)), nwfa_sfc=b.j(5.0e8, 0.4))


def arctic_ice_fog(b):
    ts = b.j(236.0, 0.01)
    return b.column(ts=ts, psfc=b.j(102500.0, 0.005), inv_depth=b.j(400.0, 0.3),
                    inv_dt=b.j(15.0, 0.3), lapse_low=5.0e-3, lapse_high=5.0e-3,
                    z_tp=8500.0, rh_low=1.0, rh_mid=0.8, rh_high=0.5, ice_rh=True,
                    rh_cloud=[(0.0, 300.0, b.j(1.05, 0.02))],
                    qi=layer(0.0, 300.0, b.j(2.0e-5, 0.5), taper=80.0),
                    qs=layer(0.0, 1000.0, b.j(5.0e-6, 0.8)),
                    di=b.j(30e-6, 0.3), nwfa_sfc=b.j(3.0e8, 0.4), nifa_sfc=b.j(3.0e6, 0.4))


def ocean_trop_warm_rain(b):
    ts = b.j(301.5, 0.003)
    t = temperature(ts, 6.0e-3, 6.5e-3, 3000.0, 16000.0)
    zm = melting_height(t)
    return b.column(ts=ts, psfc=b.j(101000.0, 0.003), lapse_low=6.0e-3, lapse_high=6.5e-3,
                    z_tp=16000.0, rh_low=0.85, rh_mid=0.7, rh_high=0.4,
                    rh_cloud=[(600.0, 3500.0, b.j(1.003, 0.003))],
                    w=updraft(500.0, 4000.0, b.j(3.0, 0.4)),
                    qc=layer(600.0, 3500.0, b.j(1.5e-3, 0.4)),
                    qr=layer(0.0, min(zm, 4000.0), b.j(1.5e-3, 0.4)),
                    dc=b.j(20e-6, 0.15), dr=b.j(1.0e-3, 0.3),
                    nwfa_sfc=b.j(1.5e8, 0.4), nwfa_h=1500.0, nifa_sfc=b.j(2.0e5, 0.5))


def ocean_marine_sc(b):
    ts = b.j(290.0, 0.005)
    top = b.j(900.0, 0.2)
    return b.column(ts=ts, psfc=b.j(101800.0, 0.003), lapse_low=9.5e-3, z_break=top,
                    lapse_high=5.0e-3, nose_z=top + 100.0, nose_dt=b.j(9.0, 0.2), nose_w=120.0,
                    rh_low=0.95, rh_mid=0.25, rh_high=0.2,
                    rh_cloud=[(top - 450.0, top, b.j(1.002, 0.001))],
                    qc=layer(top - 450.0, top, b.j(4.5e-4, 0.3), taper=80.0),
                    qr=layer(0.0, top, b.j(3.0e-5, 0.6)), dr=b.j(0.25e-3, 0.3),
                    dc=b.j(22e-6, 0.15), w=updraft(0.0, top, b.j(0.4, 0.5)),
                    nwfa_sfc=b.j(1.0e8, 0.4), nwfa_h=1000.0, nifa_sfc=b.j(1.0e5, 0.5))


def ocean_eyewall(b):
    ts = b.j(302.0, 0.003)
    t = temperature(ts, 5.5e-3, 6.5e-3, 3000.0, 16500.0)
    zm = melting_height(t)
    return b.column(ts=ts, psfc=b.j(95000.0, 0.01), lapse_low=5.5e-3, lapse_high=6.5e-3,
                    z_tp=16500.0, rh_low=0.98, rh_mid=0.95, rh_high=0.7,
                    rh_cloud=[(300.0, 14000.0, b.j(1.002, 0.002))],
                    w=updraft(300.0, 15000.0, b.j(12.0, 0.3)),
                    qc=layer(300.0, 8000.0, b.j(2.0e-3, 0.3)),
                    qr=layer(0.0, zm + 500.0, b.j(6.0e-3, 0.3)),
                    qg=layer(zm - 500.0, 10000.0, b.j(2.0e-3, 0.4)), rho_g=b.j(400.0, 0.3),
                    qs=layer(zm + 1000.0, 14000.0, b.j(1.5e-3, 0.4)),
                    qi=layer(10000.0, 15500.0, b.j(3.0e-4, 0.4)),
                    nwfa_sfc=b.j(3.0e8, 0.4), dc=b.j(20e-6, 0.15))


def terrain_orographic_snow(b):
    ts = b.j(265.0, 0.01)
    return b.column(ts=ts, psfc=b.j(70000.0, 0.02), lapse_low=6.0e-3, lapse_high=7.0e-3,
                    z_tp=8500.0, rh_low=0.98, rh_mid=0.9, rh_high=0.5, ice_rh=True,
                    rh_cloud=[(0.0, 3000.0, b.j(1.01, 0.005))],
                    w=updraft(0.0, 4000.0, b.j(1.5, 0.4)),
                    qs=layer(0.0, 4000.0, b.j(1.0e-3, 0.4)),
                    qc=layer(200.0, 1500.0, b.j(2.5e-4, 0.5)),
                    qg=layer(0.0, 1500.0, b.j(2.0e-4, 0.5)), rho_g=b.j(150.0, 0.3),
                    qi=layer(1500.0, 4500.0, b.j(8.0e-5, 0.5)), nwfa_sfc=b.j(6.0e8, 0.4))


def terrain_dry_virga(b):
    ts = b.j(300.0, 0.01)
    t = temperature(ts, 9.5e-3, 7.5e-3, 3500.0, 13000.0)
    zm = melting_height(t)
    return b.column(ts=ts, psfc=b.j(75000.0, 0.02), lapse_low=9.5e-3, lapse_high=7.5e-3,
                    z_break=3500.0, z_tp=13000.0, rh_low=0.15, rh_mid=0.45, rh_high=0.5,
                    rh_cloud=[(zm + 500.0, zm + 2500.0, 1.0)],
                    qr=layer(zm - 2000.0, zm + 300.0, b.j(2.0e-4, 0.5)),
                    qs=layer(zm, zm + 3000.0, b.j(3.0e-4, 0.5)),
                    qc=layer(zm + 500.0, zm + 2500.0, b.j(2.0e-4, 0.5)),
                    w=updraft(0.0, zm, -b.j(3.0, 0.4)), dr=b.j(0.7e-3, 0.3),
                    nwfa_sfc=b.j(1.0e9, 0.4))


def terrain_summer_storm(b):
    ts = b.j(296.0, 0.01)
    t = temperature(ts, 8.5e-3, 7.5e-3, 3000.0, 13000.0)
    zm = melting_height(t)
    return b.column(ts=ts, psfc=b.j(78000.0, 0.02), lapse_low=8.5e-3, lapse_high=7.5e-3,
                    z_tp=13000.0, rh_low=0.5, rh_mid=0.7, rh_high=0.4,
                    rh_cloud=[(1200.0, 9000.0, b.j(1.004, 0.003))],
                    w=updraft(800.0, 10000.0, b.j(15.0, 0.4)),
                    qc=layer(1200.0, 7000.0, b.j(1.5e-3, 0.4)),
                    qr=layer(0.0, zm + 800.0, b.j(1.5e-3, 0.4)),
                    qg=layer(zm - 1000.0, 8000.0, b.j(4.0e-3, 0.4)), rho_g=b.j(650.0, 0.2),
                    dg=b.j(5.0e-3, 0.3),
                    qs=layer(6000.0, 10000.0, b.j(6.0e-4, 0.4)))


REGIMES = {
    "convective": (conv_supercell, conv_mcs_stratiform, conv_pulse, conv_elevated_hail),
    "stable_night": (stable_night_clear, stable_night_fog, stable_stratus_drizzle),
    "snow_ice": (winter_snow, freezing_rain, sleet, lake_effect, arctic_ice_fog),
    "ocean": (ocean_trop_warm_rain, ocean_marine_sc, ocean_eyewall),
    "high_terrain": (terrain_orographic_snow, terrain_dry_virga, terrain_summer_storm),
}


# --------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------
def edge_cases(b):
    """(name, column) pairs.  The first one must be the first column of the
    first set: it carries zero aerosols and graupel without number or
    volume, so TEMPO's first-call aerosol and graupel initialisation runs."""
    out = []
    c = conv_pulse(b)
    c["nwfa"][:] = 0.0
    c["nifa"][:] = 0.0
    c["ng"][:] = 0.0
    c["qb"][:] = 0.0
    out.append(("first_call_init_aerosol_and_graupel", c))

    c = stable_night_clear(b)
    c["qv"] *= 0.3
    out.append(("all_dry_subsaturated_early_return", c))

    c = stable_night_clear(b)
    c["qv"] = 1.03 * qsat(c["t"], c["p"])
    out.append(("supersaturated_no_condensate", c))

    c = winter_snow(b)
    for f in ("qc", "qr", "qi", "qs", "qg"):
        c[f] = np.where(c[f] > 0.0, 1.0e-12 * (1.0 + 1e-3 * np.arange(NZ)), 0.0)
    for f in ("nc", "nr", "ni", "ng"):
        c[f] = np.where(c[f] > 0.0, 1.0e-6, 0.0)
    c["qb"] = np.where(c["qg"] > 0.0, c["qg"] * 1000.0 / 500.0, 0.0)
    out.append(("masses_at_r1_numbers_at_r2", c))

    c = conv_supercell(b)
    c["qr"] = layer(0.0, 5000.0, 1.5e-2)
    c["qg"] = layer(2000.0, 10000.0, 2.5e-2)
    c["qs"] = layer(7000.0, 12000.0, 8.0e-3)
    c["qi"] = layer(9000.0, 13000.0, 3.0e-3)
    c["qc"] = layer(1000.0, 9000.0, 6.0e-3)
    c["nr"] = number_from_mass(c["qr"], 2.5e-3, 1000.0)
    c["ng"] = c["qg"] / (math.pi / 6.0 * 900.0 * (2.0e-2) ** 3)
    c["qb"] = c["qg"] * 1000.0 / 900.0
    c["ni"] = number_from_mass(c["qi"], 150e-6, 890.0)
    c["nc"] = number_from_mass(c["qc"], 25e-6, 1000.0)
    out.append(("extreme_mass_loading", c))

    c = conv_pulse(b)
    for f in ("nc", "nr", "ni", "ng"):
        c[f][:] = 0.0
    out.append(("mass_without_number", c))

    c = conv_pulse(b)
    for f in ("nc", "nr", "ni", "ng"):
        c[f] = np.where(c[f] > 0.0, 0.0, 0.0) + layer(1000.0, 8000.0, 1.0e5)
    for f in ("qc", "qr", "qi", "qs", "qg", "qb"):
        c[f][:] = 0.0
    out.append(("number_without_mass", c))

    c = b.column(ts=322.0, psfc=101000.0, lapse_low=9.8e-3, lapse_high=8.5e-3, z_tp=17500.0,
                 rh_low=0.2, rh_mid=0.5, rh_high=0.9,
                 rh_cloud=[(14000.0, 17000.0, 1.08)],
                 qi=layer(14000.0, 17000.0, 4.0e-5), qs=layer(13000.0, 16000.0, 1.0e-5),
                 di=20e-6)
    out.append(("extreme_hot_surface_cold_tropopause", c))

    c = conv_elevated_hail(b)
    c["qb"] = np.where(c["qg"] > 0.0, c["qg"] * 1000.0 / np.where(ZM < 5000.0, 50.0, 900.0), 0.0)
    out.append(("graupel_density_at_bounds", c))

    c = conv_elevated_hail(b)
    c["qb"] = np.where(c["qg"] > 0.0, c["qg"] * 1000.0 / np.where(ZM < 5000.0, 20.0, 1500.0), 0.0)
    out.append(("graupel_volume_outside_density_bounds", c))

    c = conv_supercell(b)
    c["ng"] = np.where(c["qg"] > 0.0, c["qg"] / (math.pi / 6.0 * 900.0 * (4.0e-2) ** 3), 0.0)
    out.append(("hail_mvd_above_25mm", c))

    c = conv_pulse(b)
    c["ng"] = np.where(c["qg"] > 0.0, c["qg"] / (math.pi / 6.0 * 400.0 * (5.0e-5) ** 3), 0.0)
    c["nr"] = np.where(c["qr"] > 0.0, c["qr"] / (math.pi / 6.0 * 1000.0 * (2.0e-5) ** 3), 0.0)
    out.append(("tiny_drops_and_graupel", c))

    c = conv_supercell(b)
    c["w"] = updraft(500.0, 14000.0, 55.0) - updraft(0.0, 3000.0, 25.0)
    out.append(("extreme_vertical_velocity", c))

    c = ocean_eyewall(b)
    rng = b.rng
    for f in ("qc", "qr", "qi", "qs", "qg"):
        neg = rng.random(NZ) < 0.15
        c[f] = np.where(neg, -1.0e-9 * rng.random(NZ), c[f])
    out.append(("small_negative_mixing_ratios", c))

    c = winter_snow(b)
    c["t"][5:15] = 273.15
    c["qr"] = layer(0.0, 2000.0, 3.0e-4)
    c["nr"] = number_from_mass(c["qr"], 0.8e-3, 1000.0)
    out.append(("temperature_exactly_t0", c))

    c = arctic_ice_fog(b)
    c["qv"][20:] = 1.0e-12
    out.append(("qv_below_min_qv", c))

    c = stable_night_fog(b)
    c["nwfa"][:] = 1.0e11
    c["nifa"][:] = 1.0e9
    out.append(("very_high_aerosol", c))

    c = ocean_marine_sc(b)
    c["nwfa"][:] = 1.0e6
    c["nifa"][:] = 0.0
    out.append(("near_zero_aerosol", c))

    c = conv_mcs_stratiform(b)
    c["ni"] = np.where(c["qi"] > 0.0, 1.0e8, 0.0)
    c["nc"] = np.where(c["qc"] > 0.0, 1.0e10, c["nc"])
    out.append(("ice_and_cloud_number_above_limits", c))
    return out


def build(seed: int = 20261007):
    rng = np.random.default_rng(seed)
    b = Builder(rng)
    sets = []
    # set 1: dt 20 s (RRFS-like), every regime template x 4 variants + the edge cases
    cols, cat = [], []
    edges = edge_cases(b)
    cols.append(edges[0][1])
    cat.append(("edge", edges[0][0]))
    for regime, fns in REGIMES.items():
        for fn in fns:
            for v in range(4):
                cols.append(fn(b))
                cat.append((regime, f"{fn.__name__}_v{v}"))
    for name, c in edges[1:]:
        cols.append(c)
        cat.append(("edge", name))
    sets.append(dict(dt=20.0, nsteps=60, cols=cols, cat=cat))
    # set 2: dt 60 s, one variant of every template
    cols, cat = [], []
    for regime, fns in REGIMES.items():
        for fn in fns:
            cols.append(fn(b))
            cat.append((regime, f"{fn.__name__}_dt60"))
    sets.append(dict(dt=60.0, nsteps=30, cols=cols, cat=cat))
    # set 3: dt 150 s (>= 120 s: TEMPO converts rime above freezing to rain)
    cols, cat = [], []
    for regime, fns in REGIMES.items():
        for fn in fns:
            cols.append(fn(b))
            cat.append((regime, f"{fn.__name__}_dt150"))
    sets.append(dict(dt=150.0, nsteps=12, cols=cols, cat=cat))
    return sets


def write(sets, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = []
    with open(out_dir / "columns_in.bin", "wb") as f:
        np.array([len(sets)], dtype="<i8").tofile(f)
        for si, s in enumerate(sets, start=1):
            ncol = len(s["cols"])
            np.array([ncol, NZ, s["nsteps"]], dtype="<i8").tofile(f)
            np.array([s["dt"]], dtype="<f8").tofile(f)
            for name in FIELDS:
                a = np.stack([np.asarray(c[name], dtype=np.float64) for c in s["cols"]])
                assert a.shape == (ncol, NZ), (name, a.shape)
                assert np.all(np.isfinite(a)), name
                a.astype("<f8").tofile(f)
            for ci, (regime, variant) in enumerate(s["cat"], start=1):
                c = s["cols"][ci - 1]
                catalog.append(dict(set=si, column=ci, dt=s["dt"], nsteps=s["nsteps"],
                                    regime=regime, variant=variant,
                                    t_sfc=float(c["t"][0]), p_sfc=float(c["p"][0]),
                                    t_top=float(c["t"][-1]), w_max=float(np.max(c["w"])),
                                    w_min=float(np.min(c["w"])),
                                    q_total_max=float(np.max(c["qc"] + c["qr"] + c["qi"]
                                                             + c["qs"] + c["qg"]))))
    (out_dir / "columns_catalog.json").write_text(json.dumps(catalog, indent=1) + "\n")
    return catalog


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    cat = write(build(), out)
    by = {}
    for row in cat:
        by[row["regime"]] = by.get(row["regime"], 0) + 1
    print(f"columns: {len(cat)}  by regime: {by}")


if __name__ == "__main__":
    main()
