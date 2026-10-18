#!/usr/bin/env python3
"""Build the gwd_opt = 1 (KIM orographic drag) column-oracle inputs.

    python make_columns.py OUT_DIR

Writes, for each vertical grid G in GRIDS, one stream-unformatted
little-endian file ``gwdo-kim-inputs-G.bin`` that run_gwdo_kim.F90 reads and
one ``gwdo-kim-inputs-G.npz`` holding the same float32 words plus the
per-column regime names:

    int32   ncol, nz, ncase
    float32 dx[ncase], dt[ncase]
    float32 u, v, t, qv, p, pi, z, dz, rublten0, rvblten0   each (nz, ncol)
    float32 p_interface                                       (nz + 1, ncol)
    float32 var, oc1, sina, cosa                              each (ncol,)
    float32 oa, ol                                            each (4, ncol)

C order (nz, ncol) is Fortran order (ncol, nz): WRF's (i, k) with j = 1,
and the GPU layout (nz, 1, ncol).  The words follow what WRF's PBL driver
hands gwdo (module_pbl_driver.F:2136-2151): mass-point winds u_phy/v_phy,
temperature t_phy, vapour qv_curr, full pressure p_phy at mass levels and
p8w at the nz + 1 interfaces, Exner pi_phy and mass-level heights z above
sea level.

The columns are built in float64 and rounded once to float32: terrain-
following dry-hydrostatic eta levels (p = eta*(psfc - p_top) + p_top, p_top
= 50 hPa) on a WRF-like stretched grid, heights by hypsometric integration
of the virtual temperature, a temperature profile per regime and winds per
regime.  The orographic statistics (VAR, CON, OA1-4, OL1-4) span the
ranges WPS's orogwd_10m dataset carries (VAR 0 to about 1,300 m, CON 0 to
about 20, OA in [-1, 1], OL in [0, 1]), with water masked to 0 as GEOGRID
masks it.  The committed ``.npz`` is the fixture of record: numpy's float64
libm may round the last bit of a profile differently elsewhere, so
regeneration is not promised to reproduce the words and nothing depends on
it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SEED = 20261007
P_TOP = 5000.0
#: (name, mass levels): a WRF-default-like 50-level grid and a fine 90-level
#: grid, so the blending height kbl lands on different levels.
GRIDS = (("L50", 50), ("L90", 90))
#: (dx metres, dt seconds): the scheme reads dx for cleff, the along/across
#: wind grid lengths and the blocking area, and dt for the critical-line
#: limiter.  dt follows the usual 6*dx(km) rule.
CASES = ((1000.0, 6.0), (3000.0, 18.0), (9000.0, 54.0), (12000.0, 72.0),
         (30000.0, 180.0), (50000.0, 300.0))
PER_REGIME = 12

RD, RV, G = 287.0, 461.6, 9.81
CP = 7.0 * RD / 2.0
EP_1 = RV / RD - 1.0
EP2 = RD / RV

LEVEL_FIELDS = ("u", "v", "t", "qv", "p", "pi", "z", "dz", "rublten0",
                "rvblten0")
COLUMN_FIELDS = ("var", "oc1", "sina", "cosa")


def _eta_levels(nz: int) -> np.ndarray:
    """nz + 1 interface etas, 1 at the surface to 0 at p_top.

    Built on a reference standard atmosphere (1013.25 hPa, 288.15 K, 6.5 K/km
    to 11 km, isothermal above): layer depths grow from 40 m by a constant
    ratio to a cap, the cap chosen so the grid tops out at p_top.
    """
    def p_std(z):
        t11 = 288.15 - 0.0065 * 11000.0
        p11 = 101325.0 * (t11 / 288.15) ** (G / (RD * 0.0065))
        lo = 101325.0 * ((288.15 - 0.0065 * np.minimum(z, 11000.0))
                         / 288.15) ** (G / (RD * 0.0065))
        hi = p11 * np.exp(-G * (z - 11000.0) / (RD * t11))
        return np.where(z <= 11000.0, lo, hi)

    def z_of_p(p):
        lo, hi = 0.0, 40000.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if p_std(mid) > p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    ztop = z_of_p(P_TOP)
    ratio = 1.12 if nz <= 60 else 1.05

    def top_for(cap):
        dz = np.minimum(40.0 * ratio ** np.arange(nz), cap)
        return dz.sum(), dz

    lo, hi = 50.0, 5000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if top_for(mid)[0] < ztop:
            lo = mid
        else:
            hi = mid
    _, dz = top_for(0.5 * (lo + hi))
    zw = np.concatenate(([0.0], np.cumsum(dz)))
    zw *= ztop / zw[-1]
    eta = (p_std(zw) - P_TOP) / (101325.0 - P_TOP)
    eta[0], eta[-1] = 1.0, 0.0
    return eta


def _qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return EP2 * es / np.maximum(p - es, 1.0)


class Column:
    """One column's controls; ``build`` turns them into WRF's words."""

    def __init__(self, regime, *, ht, t_of_z, qv_sfc, hq, wind, var, oc1,
                 oa, ol, alpha, rh_cap=0.95):
        self.regime = regime
        self.ht = ht
        self.t_of_z = t_of_z        # T(z above ground) in K
        self.qv_sfc = qv_sfc
        self.hq = hq
        self.wind = wind            # (u(z), v(z)) arrays from z above ground
        self.var, self.oc1 = var, oc1
        self.oa, self.ol = np.asarray(oa, float), np.asarray(ol, float)
        self.alpha = alpha
        self.rh_cap = rh_cap

    def build(self, eta):
        nz = eta.size - 1
        # surface pressure: the standard atmosphere at the terrain height,
        # adjusted to the column's own surface temperature
        t0 = self.t_of_z(np.zeros(1))[0]
        psfc = 101325.0 * np.exp(-G * self.ht / (RD * (t0 + 0.0032 * self.ht)))
        pw = eta * (psfc - P_TOP) + P_TOP
        zw = np.zeros(nz + 1)
        t = np.zeros(nz)
        qv = np.zeros(nz)
        for _ in range(3):   # thickness depends on Tv, Tv on height
            zc = 0.5 * (zw[:-1] + zw[1:])
            pc = 0.5 * (pw[:-1] + pw[1:])
            t = self.t_of_z(zc)
            qv = self.qv_sfc * np.exp(-zc / self.hq)
            qv = np.minimum(qv, self.rh_cap * _qsat(t, pc))
            qv = np.maximum(qv, 1.0e-7)
            tv = t * (1.0 + EP_1 * qv)
            dzk = RD * tv / G * np.log(pw[:-1] / pw[1:])
            zw = np.concatenate(([0.0], np.cumsum(dzk)))
        zc = 0.5 * (zw[:-1] + zw[1:])
        pc = 0.5 * (pw[:-1] + pw[1:])
        u, v = self.wind(zc)
        return {
            "u": u, "v": v, "t": t, "qv": qv, "p": pc,
            "pi": (pc / 1.0e5) ** (RD / CP), "z": self.ht + zc,
            "dz": np.diff(zw), "p_interface": pw,
        }


def _piecewise_t(t_sfc, layers, t_floor=200.0, z_trop=None,
                 strat=-0.002):
    """T(z) from a surface value and (depth, lapse K/m) layers.

    A positive lapse cools with height, a negative one warms (inversion).
    The last layer extends to the tropopause; above it the temperature
    rises at ``-strat`` K/m.
    """
    edges = [0.0]
    for depth, _ in layers:
        edges.append(edges[-1] + depth)

    def t_of_z(z):
        z = np.asarray(z, float)
        out = np.full(z.shape, t_sfc)
        for (depth, lapse), z0 in zip(layers, edges[:-1]):
            dz = np.clip(z - z0, 0.0, depth)
            out = out - lapse * dz
        out = np.maximum(out, t_floor)
        if z_trop is not None:
            t_trop = max(t_of_layers(z_trop), t_floor)
            out = np.where(z > z_trop, t_trop - strat * (z - z_trop), out)
        return out

    def t_of_layers(z):
        t = t_sfc
        for (depth, lapse), z0 in zip(layers, edges[:-1]):
            t -= lapse * min(max(z - z0, 0.0), depth)
        return t
    return t_of_z


def _wind(speed, direction_deg, *, shear=0.0, veer=0.0, jet=None,
          reverse_at=None, reverse_factor=-0.6, ekman=0.0):
    """(u, v) from a speed, a meteorological 'towards' direction in degrees
    (0 = towards north, 90 = towards east, as u = s*sin, v = s*cos),
    linear shear (1/s), veering (deg per km), an optional jet (z, peak,
    depth) and an optional reversal height."""
    def f(z):
        z = np.asarray(z, float)
        s = speed + shear * np.minimum(z, 10000.0)   # shear to 10 km only
        if ekman:
            s = s * (1.0 - np.exp(-z / ekman))
        if jet is not None:
            zj, peak, depth = jet
            s = s + peak * np.exp(-((z - zj) / depth) ** 2)
        d = np.radians(direction_deg + veer * z / 1000.0)
        u, v = s * np.sin(d), s * np.cos(d)
        if reverse_at is not None:
            u = np.where(z > reverse_at, reverse_factor * u, u)
            v = np.where(z > reverse_at, reverse_factor * v, v)
        return u, v
    return f


def _stats(rng, var_range, oc_range=(0.0, 12.0)):
    var = rng.uniform(*var_range)
    oc = rng.uniform(*oc_range) ** 1.0
    oa = rng.uniform(-1.0, 1.0, 4)
    ol = rng.uniform(0.0, 1.0, 4)
    return var, oc, oa, ol


def regimes(rng):
    """Every column, as (regime, Column)."""
    out = []
    add = out.append

    # 1. convective: daytime mixed layer under a capping inversion
    for i in range(PER_REGIME):
        zi = rng.uniform(1200.0, 3200.0)
        t_of_z = _piecewise_t(rng.uniform(295.0, 312.0),
                              [(60.0, 0.015), (zi - 60.0, 0.0097),
                               (200.0, -0.008), (12000.0, 0.0065)],
                              z_trop=13000.0)
        var, oc, oa, ol = _stats(rng, (50.0, 700.0))
        add(Column("convective", ht=rng.uniform(100.0, 1800.0), t_of_z=t_of_z,
                   qv_sfc=rng.uniform(0.006, 0.016), hq=2500.0,
                   wind=_wind(rng.uniform(3.0, 14.0), rng.uniform(0, 360),
                              shear=0.002, veer=rng.uniform(-10, 10),
                              ekman=60.0),
                   var=var, oc1=oc, oa=oa, ol=ol,
                   alpha=rng.uniform(-0.3, 0.3)))

    # 2. stable night: surface inversion, low-level jet, mountains
    for i in range(PER_REGIME):
        depth = rng.uniform(120.0, 450.0)
        t_of_z = _piecewise_t(rng.uniform(268.0, 290.0),
                              [(depth, -rng.uniform(0.01, 0.04)),
                               (12000.0, rng.uniform(0.004, 0.0065))],
                              z_trop=12000.0)
        var, oc, oa, ol = _stats(rng, (100.0, 900.0))
        add(Column("stable_night", ht=rng.uniform(200.0, 2200.0),
                   t_of_z=t_of_z, qv_sfc=rng.uniform(0.003, 0.010), hq=2200.0,
                   wind=_wind(rng.uniform(2.0, 8.0), rng.uniform(0, 360),
                              shear=0.0015, veer=rng.uniform(5, 30),
                              jet=(depth * 1.2, rng.uniform(5, 15), 150.0),
                              ekman=30.0),
                   var=var, oc1=oc, oa=oa, ol=ol,
                   alpha=rng.uniform(-0.3, 0.3)))

    # 3. near-neutral windy synoptic flow over ridges
    for i in range(PER_REGIME):
        t_of_z = _piecewise_t(rng.uniform(275.0, 295.0),
                              [(11000.0, rng.uniform(0.0055, 0.0075))],
                              z_trop=11000.0)
        var, oc, oa, ol = _stats(rng, (150.0, 1000.0))
        add(Column("neutral_windy", ht=rng.uniform(300.0, 2500.0),
                   t_of_z=t_of_z, qv_sfc=rng.uniform(0.004, 0.012), hq=2500.0,
                   wind=_wind(rng.uniform(12.0, 28.0), rng.uniform(0, 360),
                              shear=0.0025, veer=rng.uniform(-5, 15),
                              ekman=100.0),
                   var=var, oc1=oc, oa=oa, ol=ol,
                   alpha=rng.uniform(-0.35, 0.35)))

    # 4. blocking: deep stable layer, weak flow, tall rough terrain
    for i in range(PER_REGIME):
        t_of_z = _piecewise_t(rng.uniform(262.0, 285.0),
                              [(rng.uniform(1500.0, 3500.0),
                                rng.uniform(-0.002, 0.003)),
                               (12000.0, 0.006)], z_trop=11500.0)
        var, oc, oa, ol = _stats(rng, (400.0, 1200.0), (2.0, 18.0))
        add(Column("blocking", ht=rng.uniform(500.0, 2500.0), t_of_z=t_of_z,
                   qv_sfc=rng.uniform(0.002, 0.008), hq=2000.0,
                   wind=_wind(rng.uniform(1.5, 6.0), rng.uniform(0, 360),
                              shear=0.002, veer=rng.uniform(-20, 20),
                              ekman=40.0),
                   var=var, oc1=oc, oa=oa, ol=ol,
                   alpha=rng.uniform(-0.3, 0.3)))

    # 5. high terrain: plateau surface 2.5 to 4.6 km, westerlies
    for i in range(PER_REGIME):
        t_of_z = _piecewise_t(rng.uniform(258.0, 285.0),
                              [(rng.uniform(100.0, 300.0),
                                rng.uniform(-0.02, 0.012)),
                               (10000.0, 0.0068)], z_trop=12000.0)
        var, oc, oa, ol = _stats(rng, (250.0, 1250.0), (1.0, 20.0))
        add(Column("high_terrain", ht=rng.uniform(2500.0, 4600.0),
                   t_of_z=t_of_z, qv_sfc=rng.uniform(0.001, 0.006), hq=1800.0,
                   wind=_wind(rng.uniform(6.0, 22.0), rng.uniform(60, 120),
                              shear=0.003, veer=rng.uniform(-10, 10),
                              ekman=80.0),
                   var=var, oc1=oc, oa=oa, ol=ol,
                   alpha=rng.uniform(-0.3, 0.3)))

    # 6. ocean: GEOGRID masks water to zero statistics; two near-coast
    #    columns keep a small residual island variance.
    for i in range(PER_REGIME):
        t_of_z = _piecewise_t(rng.uniform(280.0, 302.0),
                              [(600.0, 0.008), (11000.0, 0.0065)],
                              z_trop=14000.0)
        island = i >= PER_REGIME - 2
        var, oc, oa, ol = ((rng.uniform(5.0, 60.0), rng.uniform(0.0, 3.0),
                            rng.uniform(-1, 1, 4), rng.uniform(0, 1, 4))
                           if island else (0.0, 0.0, np.zeros(4), np.zeros(4)))
        add(Column("ocean", ht=0.0 if not island else rng.uniform(0.0, 40.0),
                   t_of_z=t_of_z, qv_sfc=rng.uniform(0.008, 0.020), hq=2200.0,
                   wind=_wind(rng.uniform(4.0, 18.0), rng.uniform(0, 360),
                              shear=0.002, ekman=30.0),
                   var=var, oc1=oc, oa=oa, ol=ol,
                   alpha=rng.uniform(-0.3, 0.3)))

    # 7. cold snow / arctic: very cold, dry, strong surface inversion
    for i in range(PER_REGIME):
        t_of_z = _piecewise_t(rng.uniform(232.0, 262.0),
                              [(rng.uniform(200.0, 800.0),
                                -rng.uniform(0.01, 0.03)),
                               (9000.0, 0.0055)], z_trop=9000.0,
                              t_floor=195.0)
        var, oc, oa, ol = _stats(rng, (50.0, 800.0))
        add(Column("cold_snow", ht=rng.uniform(0.0, 2000.0), t_of_z=t_of_z,
                   qv_sfc=rng.uniform(1.0e-4, 1.5e-3), hq=1500.0,
                   wind=_wind(rng.uniform(2.0, 15.0), rng.uniform(0, 360),
                              shear=0.002, veer=rng.uniform(-15, 15),
                              ekman=30.0),
                   var=var, oc1=oc, oa=oa, ol=ol,
                   alpha=rng.uniform(-1.0, 1.0)))   # polar projection

    # 8. tropical moist: warm, near-saturated low levels, easterlies
    for i in range(PER_REGIME):
        t_of_z = _piecewise_t(rng.uniform(297.0, 304.0),
                              [(500.0, 0.0095), (15000.0, 0.0062)],
                              z_trop=16000.0)
        var, oc, oa, ol = _stats(rng, (50.0, 900.0))
        add(Column("tropical_moist", ht=rng.uniform(0.0, 1500.0),
                   t_of_z=t_of_z, qv_sfc=rng.uniform(0.016, 0.022), hq=2800.0,
                   wind=_wind(rng.uniform(3.0, 12.0), rng.uniform(240, 300),
                              shear=-0.0005, ekman=50.0),
                   var=var, oc1=oc, oa=oa, ol=ol,
                   alpha=rng.uniform(-0.1, 0.1), rh_cap=0.98))

    # 9. jet: strong shear to a 50-80 m/s jet near 250 hPa, turning flow
    for i in range(PER_REGIME):
        t_of_z = _piecewise_t(rng.uniform(270.0, 292.0),
                              [(11000.0, 0.0065)], z_trop=11000.0)
        var, oc, oa, ol = _stats(rng, (150.0, 1100.0))
        add(Column("jet_shear", ht=rng.uniform(200.0, 3000.0), t_of_z=t_of_z,
                   qv_sfc=rng.uniform(0.003, 0.010), hq=2200.0,
                   wind=_wind(rng.uniform(5.0, 15.0), rng.uniform(0, 360),
                              shear=0.001, veer=rng.uniform(-25, 25),
                              jet=(10500.0, rng.uniform(40.0, 70.0), 2500.0),
                              ekman=60.0),
                   var=var, oc1=oc, oa=oa, ol=ol,
                   alpha=rng.uniform(-0.35, 0.35)))

    # 10. critical level: the flow reverses aloft (inside or above the
    #     low layer), so the wave stress meets velco <= 0
    for i in range(PER_REGIME):
        t_of_z = _piecewise_t(rng.uniform(272.0, 292.0),
                              [(600.0, rng.uniform(0.0, 0.006)),
                               (11000.0, 0.0062)], z_trop=11500.0)
        var, oc, oa, ol = _stats(rng, (150.0, 1000.0))
        add(Column("critical_level", ht=rng.uniform(300.0, 2200.0),
                   t_of_z=t_of_z, qv_sfc=rng.uniform(0.003, 0.010), hq=2200.0,
                   wind=_wind(rng.uniform(6.0, 18.0), rng.uniform(0, 360),
                              shear=0.0, veer=rng.uniform(-10, 10),
                              reverse_at=rng.uniform(800.0, 9000.0),
                              reverse_factor=rng.uniform(-1.0, -0.2),
                              ekman=50.0),
                   var=var, oc1=oc, oa=oa, ol=ol,
                   alpha=rng.uniform(-0.3, 0.3)))

    # 11. sector boundaries: the low-level wind sits on the edge between two
    #     of the scheme's eight direction sectors (atan2 + pi = (n + 1/2) *
    #     2 pi / 8), offset by a few float32 ulps either way, so the OA/OL
    #     direction the scheme picks turns on the last bits of atan2.
    bound = [-157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5]
    offsets = [0.0, 3e-6, -3e-6]
    for i in range(PER_REGIME):
        t_of_z = _piecewise_t(rng.uniform(270.0, 290.0),
                              [(11000.0, rng.uniform(0.004, 0.006))],
                              z_trop=11500.0)
        var, oc, oa, ol = _stats(rng, (200.0, 900.0))
        ang = bound[i % 8] + offsets[i % 3]
        add(Column("sector_boundary", ht=rng.uniform(300.0, 2000.0),
                   t_of_z=t_of_z, qv_sfc=rng.uniform(0.003, 0.010), hq=2200.0,
                   wind=_wind(rng.uniform(5.0, 15.0), ang, shear=0.0),
                   var=var, oc1=oc, oa=oa, ol=ol, alpha=0.0))

    # 12. edge cases, each reaching one branch (labelled in EDGE_NOTES)
    base_t = _piecewise_t(282.0, [(11000.0, 0.005)], z_trop=11500.0)
    stable_t = _piecewise_t(275.0, [(3000.0, 0.001), (9000.0, 0.006)],
                            z_trop=11500.0)
    oa1, ol5 = np.full(4, 0.4), np.full(4, 0.5)

    def edge(note, **kw):
        kw.setdefault("ht", 1200.0)
        kw.setdefault("t_of_z", base_t)
        kw.setdefault("qv_sfc", 0.006)
        kw.setdefault("hq", 2200.0)
        kw.setdefault("wind", _wind(10.0, 45.0, shear=0.001))
        kw.setdefault("var", 500.0)
        kw.setdefault("oc1", 4.0)
        kw.setdefault("oa", oa1)
        kw.setdefault("ol", ol5)
        kw.setdefault("alpha", 0.0)
        EDGE_NOTES.append(note)
        add(Column("edge", **kw))

    edge("calm: zero wind, ulow clamps to 1 -> no drag",
         wind=lambda z: (np.zeros_like(z), np.zeros_like(z)))
    edge("light wind below 1 m/s: ulow == 1 -> no drag",
         wind=_wind(0.6, 200.0))
    edge("var = 0 under drag-favouring flow -> no drag", var=0.0)
    edge("var = 0.5 m: 2*var reached at level 2, kbl = 3", var=0.5)
    edge("var = 15000 m: 2*var above the model top, klowtop = 0, kbl = 2",
         var=15000.0)
    edge("oa = +1, ol = 1 in every direction", oa=np.ones(4), ol=np.ones(4))
    edge("oa = -1, ol = 0: olmin floor, od clamps to odmin",
         oa=-np.ones(4), ol=np.zeros(4))
    edge("ol along 1e-3, across 0.9: od clamps to odmax",
         ol=np.array([1e-3, 0.9, 1e-3, 0.9]),
         wind=_wind(10.0, 270.0, shear=0.001))
    edge("con = 0: no wave stress, flow blocking only", oc1=0.0,
         t_of_z=stable_t, wind=_wind(3.0, 30.0, shear=0.0005))
    edge("con = 40, very stable, weak wind: fr reaches frmax",
         oc1=40.0, var=1250.0, t_of_z=stable_t, wind=_wind(2.0, 120.0))
    edge("wind reversed between the two lowest levels -> no drag",
         wind=lambda z: (np.where(z < 60.0, 5.0, -5.0) * np.ones_like(z),
                         np.zeros_like(z)))
    edge("flow reversed in a layer inside the low layer, not at level 1: "
         "drag off", var=800.0,
         wind=lambda z: (np.where((z > 450.0) & (z < 650.0), -10.0, 10.0)
                         * np.ones_like(z), np.zeros_like(z)))
    edge("velco just above 0: clamped to veleps",
         wind=lambda z: (8.0 * np.cos(np.radians(np.minimum(z / 12.0, 89.5))),
                         8.0 * np.sin(np.radians(np.minimum(z / 12.0, 89.5)))))
    edge("map rotation alpha = 1.2 rad (polar stereographic)", alpha=1.2,
         wind=_wind(12.0, 300.0, shear=0.002))
    edge("deep blocking: kblk well above the surface", var=1100.0,
         oc1=10.0, t_of_z=stable_t, wind=_wind(2.5, 210.0, shear=0.0003))
    edge("strong low-level drag with weak velco: dtfac limiter active",
         var=1200.0, oc1=15.0, t_of_z=stable_t,
         wind=_wind(4.0, 160.0, shear=-0.0003, ekman=20.0))
    edge("subnormal and signed-zero winds: no FTZ anywhere",
         wind=lambda z: (np.where(np.arange(z.size) % 2 == 0, 1.0e-40,
                                  -0.0) * np.ones_like(z),
                         np.full_like(z, -2.0e-39)))
    return out


EDGE_NOTES: list[str] = []


def build_grid(nz: int, rng_cols, rng_pbl):
    eta = _eta_levels(nz)
    cols = regimes(rng_cols)
    ncol = len(cols)
    f = {k: np.zeros((nz, ncol)) for k in LEVEL_FIELDS}
    f["p_interface"] = np.zeros((nz + 1, ncol))
    for k in COLUMN_FIELDS:
        f[k] = np.zeros(ncol)
    f["oa"] = np.zeros((4, ncol))
    f["ol"] = np.zeros((4, ncol))
    names = []
    for i, c in enumerate(cols):
        w = c.build(eta)
        for k in ("u", "v", "t", "qv", "p", "pi", "z", "dz"):
            f[k][:, i] = w[k]
        f["p_interface"][:, i] = w["p_interface"]
        f["var"][i], f["oc1"][i] = c.var, c.oc1
        f["oa"][:, i], f["ol"][:, i] = c.oa, c.ol
        f["sina"][i], f["cosa"][i] = np.sin(c.alpha), np.cos(c.alpha)
        names.append(c.regime)
    # the PBL scheme's tendencies gwdo adds to: a few 1e-4 m/s2
    f["rublten0"] = rng_pbl.normal(0.0, 2.0e-4, (nz, ncol))
    f["rvblten0"] = rng_pbl.normal(0.0, 2.0e-4, (nz, ncol))
    out = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in f.items()}
    # keep the deliberate subnormal / signed-zero words exactly
    sub = names.index("edge") + EDGE_NOTES.index(
        "subnormal and signed-zero winds: no FTZ anywhere")
    lev = np.arange(nz) % 2 == 0
    out["u"][:, sub] = np.where(lev, np.float32(1.0e-40), np.float32(-0.0))
    out["v"][:, sub] = np.float32(-2.0e-39)
    out["regime"] = np.asarray(names)
    out["dx"] = np.asarray([c[0] for c in CASES], np.float32)
    out["dt"] = np.asarray([c[1] for c in CASES], np.float32)
    return out


def write_bin(path: Path, f: dict) -> None:
    nz, ncol = f["u"].shape
    with open(path, "wb") as fh:
        np.asarray([ncol, nz, len(CASES)], "<i4").tofile(fh)
        f["dx"].astype("<f4").tofile(fh)
        f["dt"].astype("<f4").tofile(fh)
        for k in LEVEL_FIELDS:
            f[k].astype("<f4").tofile(fh)
        f["p_interface"].astype("<f4").tofile(fh)
        for k in COLUMN_FIELDS:
            f[k].astype("<f4").tofile(fh)
        f["oa"].astype("<f4").tofile(fh)
        f["ol"].astype("<f4").tofile(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n, (name, nz) in enumerate(GRIDS):
        EDGE_NOTES.clear()
        f = build_grid(nz, np.random.default_rng(SEED),
                       np.random.default_rng(SEED + 1 + n))
        for k, v in f.items():
            if v.dtype == np.float32 and not np.all(np.isfinite(v)):
                raise SystemExit(f"{name}: non-finite {k}")
        write_bin(args.out_dir / f"gwdo-kim-inputs-{name}.bin", f)
        np.savez(args.out_dir / f"gwdo-kim-inputs-{name}.npz",
                 edge_notes=np.asarray(EDGE_NOTES), **f)
        print(f"{name}: {f['u'].shape[1]} columns x {f['u'].shape[0]} levels, "
              f"{len(set(f['regime']))} regimes, {len(CASES)} (dx, dt) cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
