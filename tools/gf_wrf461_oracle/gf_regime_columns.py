"""Realistic Grell-Freitas oracle columns across weather regimes.

The 216-column fixture beside this file (gf_cases.F90) was built to reach
branches.  These columns are built to look like weather: soundings for
the regimes cumulus parameterisation actually meets -- tropical ocean,
continental severe, capped, nocturnal stable, winter snow and arctic ice,
lake-effect over warm water, high terrain, trade cumulus, stratocumulus,
desert -- plus a clos_choice sweep and named edge cases (moisture floor,
the TN < 200 K reset, zero forcing, extreme kpbl, supersaturation, dew,
ground below sea level, sub-km and 100 km spacing, the exact 258 K split).

Every column is hydrostatically consistent: interface heights on a
WRF-like stretched grid, temperature from a lapse-rate profile, moisture
from a relative-humidity profile over water above 0 C and ice below,
pressure integrated with virtual temperature, Exner and density from it.
Forcing lanes are shaped like a PBL scheme's, a radiation scheme's and
advection's, scaled so the PBL lanes integrate to the surface fluxes.
A deterministic seeded jitter (0.15 K, 1.5 % RH) keeps the profiles from
being unrealistically smooth.  Everything is float64 until the single
float32 cast at the end, so the words written are reproducible.

usage:
    python gf_regime_columns.py OUT_DIR
writes OUT_DIR/gf-regime-inputs.bin (run_gf_columns.F90's input format)
and OUT_DIR/gf-regime-manifest.json (one record per column).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

NZ = 50
MAGIC_IN = 47011

RD = 287.04
RV = 461.6
CP = 1004.5
G = 9.81
EPS = RD / RV
P0 = 1.0e5
KAPPA = RD / CP

LEV_FIELDS = ("u", "v", "w", "t", "qv", "p", "pi", "rho", "dz8w",
              "rthften", "rqvften", "rthraten", "rthblten", "rqvblten")
SCA_FIELDS = ("dt", "dx", "ht", "hfx", "qfx", "xland")
INT_FIELDS = ("kpbl", "ishallow", "ichoice")


@dataclass(frozen=True)
class Regime:
    name: str
    description: str
    t_sfc: float                     # K, air temperature at the ground
    ht: float = 0.0                  # m, terrain height
    psfc: float = 101325.0           # Pa, surface pressure
    xland: float = 1.0               # 1 land, 2 water
    hfx: float = 0.0                 # W m-2
    qfx: float = 0.0                 # kg m-2 s-1
    pbl_depth: float = 1000.0        # m
    pbl_kind: str = "mixed"          # mixed | stable
    stable_inv: float = 0.0          # K warming over the stable layer
    lapse: float = 6.5               # K/km, free troposphere
    cap_z: float = 0.0               # m AGL, capping inversion base (0 = none)
    cap_dt: float = 0.0              # K warming across the cap
    cap_depth: float = 300.0         # m
    eml_lapse: float = 0.0           # K/km in the layer above the cap (0 = lapse)
    eml_top: float = 0.0             # m AGL
    z_trop: float = 12000.0          # m AGL tropopause
    strat_lapse: float = -2.0        # K/km above the tropopause (warming)
    t_trop_min: float = 0.0          # K floor at the tropopause (0 = none)
    rh: tuple = ((0.0, 0.7), (3000.0, 0.5), (10000.0, 0.3))
    rh_strat: float = 0.03
    u0: float = 2.0
    u_shear: float = 10.0            # m/s over 6 km
    v0: float = 0.0
    v_shear: float = 2.0
    jet_z: float = 0.0               # m AGL low-level jet height (0 = none)
    jet_speed: float = 0.0
    w_amp: float = 0.02              # m/s, large-scale vertical motion
    w_depth: float = 10000.0         # m, depth of the w half-sine
    w_sfc: float = 0.0               # m/s at the lowest face (terrain upslope)
    th_adv: float = 0.0              # K/s advective theta tendency amplitude
    qv_adv: float = 0.0              # kg/kg/s advective moisture tendency at the ground
    rad_day: float = -1.5            # K/day radiative temperature tendency, troposphere
    rad_sfc_extra: float = 0.0       # K/day extra near-surface (night)
    seed: int = 0


def _es_water(t):
    tc = t - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def _es_ice(t):
    tc = t - 273.15
    return 611.2 * np.exp(22.46 * tc / (tc + 272.62))


def _es(t):
    return np.where(t >= 273.15, _es_water(t), _es_ice(t))


def interface_heights(nz: int = NZ) -> np.ndarray:
    """AGL heights of the nz + 1 w-faces: 40 m at the ground, stretched by
    12 per cent per layer, capped at 650 m -- a WRF-like 50-level grid with
    its top near 21.6 km above the ground."""
    dz = np.minimum(40.0 * 1.12 ** np.arange(nz), 650.0)
    return np.concatenate([[0.0], np.cumsum(dz)])


def _lapse_at(r: Regime, z: np.ndarray) -> np.ndarray:
    """K/m lapse rate (positive = cooling with height) at AGL z."""
    lap = np.full_like(z, r.lapse / 1000.0)
    in_pbl = z < r.pbl_depth
    if r.pbl_kind == "mixed":
        lap = np.where(in_pbl, 9.8e-3, lap)
    else:
        lap = np.where(in_pbl, -r.stable_inv / max(r.pbl_depth, 1.0), lap)
    if r.cap_z > 0.0:
        in_cap = (z >= r.cap_z) & (z < r.cap_z + r.cap_depth)
        lap = np.where(in_cap, -r.cap_dt / r.cap_depth, lap)
        if r.eml_lapse > 0.0:
            in_eml = (z >= r.cap_z + r.cap_depth) & (z < r.eml_top)
            lap = np.where(in_eml, r.eml_lapse / 1000.0, lap)
    lap = np.where(z >= r.z_trop, r.strat_lapse / 1000.0, lap)
    return lap


def _temperature(r: Regime, z: np.ndarray) -> np.ndarray:
    fine = np.linspace(0.0, float(z.max()) + 1.0, 40001)
    lap = _lapse_at(r, fine)
    tf = r.t_sfc - np.concatenate(
        [[0.0], np.cumsum(0.5 * (lap[1:] + lap[:-1]) * np.diff(fine))])
    t = np.interp(z, fine, tf)
    if r.t_trop_min > 0.0:
        t = np.maximum(t, r.t_trop_min)
    return t


def _rh(r: Regime, z: np.ndarray) -> np.ndarray:
    zk = np.array([k[0] for k in r.rh])
    rk = np.array([k[1] for k in r.rh])
    rh = np.interp(z, zk, rk)
    return np.where(z >= r.z_trop, r.rh_strat, rh)


@dataclass
class Column:
    regime: str
    variant: str
    dt: float
    dx: float
    ishallow: int
    ichoice: int
    kpbl: int
    sca: dict
    lev: dict
    p8w: np.ndarray
    notes: list = field(default_factory=list)


def build_column(r: Regime, *, dx: float, ishallow: int, ichoice: int = 0,
                 dt: float | None = None, variant: str = "",
                 kpbl_override: int | None = None,
                 post=None) -> Column:
    rng = np.random.default_rng(r.seed * 1000 + 17)
    zw = interface_heights()
    dz = np.diff(zw)
    zm = 0.5 * (zw[1:] + zw[:-1])
    nz = dz.size

    t = _temperature(r, zm)
    jitter_t = rng.normal(0.0, 0.15, nz)
    jitter_t[zm < r.pbl_depth] *= 0.3
    t = t + jitter_t
    rh = np.clip(_rh(r, zm) + rng.normal(0.0, 0.015, nz), 0.005, 1.05)

    # hydrostatic: iterate twice so qv and the virtual temperature agree
    qv = np.zeros(nz)
    for _ in range(3):
        tv = t * (1.0 + 0.608 * qv)
        pw = np.empty(nz + 1)
        pw[0] = r.psfc
        for k in range(nz):
            pw[k + 1] = pw[k] * math.exp(-G * dz[k] / (RD * tv[k]))
        p = pw[:-1] * np.exp(-G * 0.5 * dz / (RD * tv))
        e = np.minimum(rh * _es(t), 0.5 * p)
        qv = EPS * e / (p - (1.0 - EPS) * e)
    tv = t * (1.0 + 0.608 * qv)
    rho = p / (RD * tv)
    pi = (p / P0) ** KAPPA

    # winds
    frac = np.clip(zm / 6000.0, 0.0, 1.0)
    u = r.u0 + r.u_shear * frac
    v = r.v0 + r.v_shear * frac
    if r.jet_z > 0.0:
        jet = r.jet_speed * np.exp(-((zm - r.jet_z) / 300.0) ** 2)
        v = v + jet
    u = u + rng.normal(0.0, 0.3, nz)
    v = v + rng.normal(0.0, 0.3, nz)

    # w on the faces GFDRV is handed (WRF passes w_2, the face below each
    # mass level), zero at the ground unless the terrain drives an upslope
    zf = zw[:-1]
    w = np.where(zf < r.w_depth, r.w_amp * np.sin(np.pi * zf / r.w_depth), 0.0)
    w = w + r.w_sfc * np.exp(-zf / 1500.0)
    w = w + rng.normal(0.0, 0.1 * max(abs(r.w_amp), 0.005), nz)
    w[0] = r.w_sfc

    # forcing lanes
    th_adv = r.th_adv * np.sin(np.pi * np.clip(zm / 10000.0, 0.0, 1.0))
    qv_adv = r.qv_adv * np.exp(-zm / 2500.0)
    rad_t = np.where(zm < r.z_trop, r.rad_day, 0.0)
    rad_t = rad_t + r.rad_sfc_extra * np.exp(-zm / 150.0)
    rthraten = rad_t / 86400.0 / pi

    h = max(r.pbl_depth, 50.0)
    if r.hfx >= 0.0 and r.pbl_kind == "mixed":
        shape_t = np.where(zm < h, 1.0 - 1.2 * zm / h, 0.0)
    else:
        shape_t = np.exp(-zm / 120.0)
    shape_q = np.where(zm < h, 1.0 - 1.5 * zm / h, 0.0) if r.qfx > 0.0 \
        else np.exp(-zm / 120.0)
    norm_t = float(np.sum(rho * CP * pi * shape_t * dz))
    norm_q = float(np.sum(rho * shape_q * dz))
    rthblten = (r.hfx / norm_t) * shape_t if norm_t != 0.0 else 0.0 * zm
    rqvblten = (r.qfx / norm_q) * shape_q if norm_q != 0.0 else 0.0 * zm

    if kpbl_override is not None:
        kpbl = int(kpbl_override)
    else:
        kpbl = int(np.searchsorted(zm, r.pbl_depth) + 1)
        kpbl = max(2, min(kpbl, nz - 5))

    if dt is None:
        dt = float(min(max(round(6.0 * dx / 1000.0), 3.0), 180.0))

    lev = dict(u=u, v=v, w=w, t=t, qv=qv, p=p, pi=pi, rho=rho, dz8w=dz,
               rthften=th_adv, rqvften=qv_adv, rthraten=rthraten,
               rthblten=rthblten, rqvblten=rqvblten)
    sca = dict(dt=dt, dx=dx, ht=r.ht, hfx=r.hfx, qfx=r.qfx, xland=r.xland)
    col = Column(regime=r.name, variant=variant, dt=dt, dx=dx,
                 ishallow=ishallow, ichoice=ichoice, kpbl=kpbl, sca=sca,
                 lev=lev, p8w=pw)
    if post is not None:
        post(col)
    return col


# ---------------------------------------------------------------------------
# the regimes
# ---------------------------------------------------------------------------
REGIMES = [
    Regime("tropical-ocean-deep", "warm-pool maritime deep convection, "
           "SST-driven fluxes, weak large-scale ascent",
           t_sfc=300.5, xland=2.0, hfx=12.0, qfx=6.0e-5, pbl_depth=600.0,
           lapse=5.6, z_trop=15500.0, t_trop_min=193.0,
           rh=((0, 0.82), (1000, 0.85), (3000, 0.65), (7000, 0.5),
               (12000, 0.45)), u0=-4.0, u_shear=6.0, w_amp=0.03,
           w_depth=13000.0, th_adv=-1.0e-5, qv_adv=1.0e-8, seed=1),
    Regime("tropical-cyclone-core", "moist, near-saturated deep troposphere "
           "under strong ascent over warm ocean",
           t_sfc=300.0, xland=2.0, hfx=60.0, qfx=2.0e-4, pbl_depth=500.0,
           lapse=5.2, z_trop=16000.0, t_trop_min=192.0,
           rh=((0, 0.92), (2000, 0.95), (8000, 0.9), (13000, 0.7)),
           u0=-15.0, u_shear=10.0, w_amp=0.6, w_depth=14000.0,
           th_adv=2.0e-5, qv_adv=4.0e-8, rad_day=-0.5, seed=2),
    Regime("continental-severe", "Weisman-Klemp-like afternoon sounding "
           "over land, strong sensible and latent flux, veering shear",
           t_sfc=303.0, hfx=320.0, qfx=1.4e-4, pbl_depth=1600.0,
           lapse=7.4, z_trop=12000.0, strat_lapse=-1.5,
           rh=((0, 0.6), (1500, 0.7), (3000, 0.45), (9000, 0.25),
               (12000, 0.2)), u0=5.0, u_shear=25.0, v0=8.0, v_shear=-4.0,
           w_amp=0.08, th_adv=1.0e-5, qv_adv=2.0e-8, rad_day=1.0, seed=3),
    Regime("capped-elevated-mixed-layer", "moist boundary layer under a "
           "strong cap and a steep elevated mixed layer (loaded gun)",
           t_sfc=301.0, hfx=250.0, qfx=1.6e-4, pbl_depth=1100.0,
           cap_z=1150.0, cap_dt=4.0, cap_depth=250.0, eml_lapse=8.8,
           eml_top=4500.0, lapse=6.8, z_trop=12500.0,
           rh=((0, 0.75), (1100, 0.85), (1400, 0.3), (4500, 0.2),
               (9000, 0.3)), u0=8.0, u_shear=20.0, v0=12.0, w_amp=0.02,
           seed=4),
    Regime("nocturnal-stable-llj", "radiative surface inversion, low-level "
           "jet, moist elevated layer above it (night)",
           t_sfc=291.0, hfx=-35.0, qfx=5.0e-6, pbl_depth=250.0,
           pbl_kind="stable", stable_inv=4.0, lapse=6.9, z_trop=12500.0,
           rh=((0, 0.85), (300, 0.7), (1200, 0.85), (2500, 0.7),
               (6000, 0.35), (10000, 0.2)), u0=3.0, u_shear=15.0,
           jet_z=600.0, jet_speed=18.0, w_amp=0.05, th_adv=-1.5e-5,
           qv_adv=3.0e-8, rad_day=-2.0, rad_sfc_extra=-6.0, seed=5),
    Regime("stable-night-dry", "clear radiative night over dry land, "
           "strong inversion, subsidence",
           t_sfc=278.0, hfx=-45.0, qfx=0.0, pbl_depth=200.0,
           pbl_kind="stable", stable_inv=7.0, lapse=6.0, z_trop=11000.0,
           rh=((0, 0.6), (500, 0.35), (5000, 0.15), (10000, 0.1)),
           u0=1.0, u_shear=12.0, w_amp=-0.02, rad_day=-2.5,
           rad_sfc_extra=-10.0, seed=6),
    Regime("winter-continental-snow", "cold continental column, saturated "
           "low levels under warm advection (light snow), ice everywhere",
           t_sfc=266.0, hfx=10.0, qfx=1.0e-5, pbl_depth=500.0, lapse=6.0,
           z_trop=9500.0, strat_lapse=0.0,
           rh=((0, 0.95), (2500, 0.98), (4500, 0.8), (8000, 0.5)),
           u0=10.0, u_shear=25.0, w_amp=0.12, w_depth=8000.0,
           th_adv=4.0e-5, qv_adv=1.0e-8, rad_day=-1.0, seed=7),
    Regime("arctic-polar-night", "very cold, very dry column under a deep "
           "surface inversion; every level below 258 K",
           t_sfc=238.0, hfx=-20.0, qfx=0.0, pbl_depth=800.0,
           pbl_kind="stable", stable_inv=12.0, lapse=5.0, z_trop=8500.0,
           strat_lapse=0.5, rh=((0, 0.85), (1000, 0.7), (5000, 0.5)),
           rh_strat=0.1, u0=4.0, u_shear=8.0, w_amp=0.0, rad_day=-1.8,
           seed=8),
    Regime("lake-effect-cold-air-outbreak", "arctic air over open warm "
           "water: huge fluxes, shallow convective boundary layer, snow",
           t_sfc=268.0, xland=2.0, hfx=280.0, qfx=1.1e-4, pbl_depth=1500.0,
           cap_z=1600.0, cap_dt=2.0, cap_depth=200.0, lapse=6.5,
           z_trop=8500.0, strat_lapse=0.0,
           rh=((0, 0.8), (1500, 0.9), (1800, 0.5), (5000, 0.4)),
           u0=12.0, u_shear=6.0, w_amp=0.05, w_depth=3000.0, th_adv=-5.0e-5,
           rad_day=-1.5, seed=9),
    Regime("plateau-monsoon", "high plateau (4.5 km) in the summer "
           "monsoon: thin air, moist, afternoon heating",
           t_sfc=287.0, ht=4500.0, psfc=58500.0, hfx=200.0, qfx=9.0e-5,
           pbl_depth=1500.0, lapse=7.0, z_trop=12000.0, t_trop_min=196.0,
           rh=((0, 0.65), (2000, 0.7), (5000, 0.5), (9000, 0.3)),
           u0=-2.0, u_shear=8.0, w_amp=0.06, w_sfc=0.05, seed=10),
    Regime("high-plains-dry-afternoon", "2.5 km terrain, deep dry "
           "afternoon mixed layer, high-based convection",
           t_sfc=300.0, ht=2500.0, psfc=75000.0, hfx=420.0, qfx=4.0e-5,
           pbl_depth=3500.0, lapse=7.8, z_trop=10500.0,
           rh=((0, 0.2), (3500, 0.55), (5000, 0.45), (8000, 0.3)),
           u0=6.0, u_shear=14.0, w_amp=0.04, w_sfc=0.03, rad_day=1.5,
           seed=11),
    Regime("mountain-winter-orographic", "3 km mountain ridge in winter, "
           "saturated orographic ascent, snow",
           t_sfc=263.0, ht=3000.0, psfc=69500.0, hfx=5.0, qfx=8.0e-6,
           pbl_depth=600.0, lapse=6.2, z_trop=7500.0, strat_lapse=0.0,
           rh=((0, 0.98), (3000, 0.99), (5000, 0.75)), u0=18.0,
           u_shear=20.0, w_amp=0.4, w_depth=5000.0, w_sfc=0.25,
           th_adv=1.0e-5, seed=12),
    Regime("trade-cumulus", "BOMEX-like trade-wind boundary layer under a "
           "subsidence inversion",
           t_sfc=299.0, xland=2.0, hfx=10.0, qfx=6.5e-5, pbl_depth=600.0,
           cap_z=1500.0, cap_dt=3.0, cap_depth=500.0, lapse=5.5,
           z_trop=15500.0, t_trop_min=194.0,
           rh=((0, 0.8), (600, 0.9), (1500, 0.85), (2100, 0.3),
               (8000, 0.2)), u0=-8.0, u_shear=4.0, w_amp=-0.0065,
           w_depth=4000.0, th_adv=-2.3e-5, qv_adv=-1.2e-8, seed=13),
    Regime("marine-stratocumulus", "DYCOMS-like cool ocean, sharp 800 m "
           "inversion, subsidence",
           t_sfc=289.0, xland=2.0, hfx=15.0, qfx=3.0e-5, pbl_depth=800.0,
           cap_z=820.0, cap_dt=8.5, cap_depth=60.0, lapse=5.0,
           z_trop=13000.0, rh=((0, 0.9), (800, 1.0), (900, 0.12),
                               (6000, 0.1)),
           u0=6.0, u_shear=2.0, w_amp=-0.004, w_depth=3000.0,
           rad_day=-1.0, rad_sfc_extra=0.0, seed=14),
    Regime("desert-afternoon", "very hot, very dry deep mixed layer",
           t_sfc=316.0, hfx=480.0, qfx=5.0e-6, pbl_depth=4500.0,
           lapse=7.0, z_trop=14000.0,
           rh=((0, 0.08), (4500, 0.18), (8000, 0.15)), u0=4.0,
           u_shear=6.0, w_amp=-0.01, rad_day=2.0, seed=15),
    Regime("monsoon-land", "humid monsoon trough over land, moderate "
           "instability, broad ascent",
           t_sfc=304.0, hfx=150.0, qfx=1.5e-4, pbl_depth=1200.0,
           lapse=6.0, z_trop=15000.0, t_trop_min=195.0,
           rh=((0, 0.75), (1500, 0.8), (6000, 0.7), (11000, 0.5)),
           u0=-6.0, u_shear=-15.0, w_amp=0.1, w_depth=12000.0,
           qv_adv=3.0e-8, seed=16),
    Regime("midlatitude-cold-front", "strong frontal ascent in a moist, "
           "moderately unstable warm sector",
           t_sfc=295.0, hfx=60.0, qfx=8.0e-5, pbl_depth=900.0, lapse=6.6,
           z_trop=11500.0, rh=((0, 0.85), (2000, 0.9), (6000, 0.75),
                               (10000, 0.5)),
           u0=10.0, u_shear=30.0, v0=15.0, w_amp=0.35, w_depth=10000.0,
           th_adv=-6.0e-5, qv_adv=5.0e-8, seed=17),
    Regime("tropical-ocean-suppressed", "subsiding dry-mid-level tropical "
           "ocean (suppressed phase)",
           t_sfc=300.0, xland=2.0, hfx=8.0, qfx=5.0e-5, pbl_depth=500.0,
           lapse=5.8, z_trop=15500.0, t_trop_min=193.0,
           rh=((0, 0.8), (1000, 0.75), (2500, 0.25), (8000, 0.15)),
           u0=-6.0, u_shear=3.0, w_amp=-0.03, w_depth=8000.0, seed=18),
    Regime("dryline-moist-side", "moist side of a dryline, strong "
           "heating, convergence",
           t_sfc=304.0, hfx=380.0, qfx=1.2e-4, pbl_depth=2000.0, lapse=7.6,
           z_trop=12500.0, rh=((0, 0.5), (2000, 0.75), (3500, 0.3),
                               (9000, 0.2)),
           u0=4.0, u_shear=22.0, v0=10.0, w_amp=0.25, w_depth=4000.0,
           seed=19),
    Regime("elevated-mcs-night", "elevated nocturnal convection above a "
           "stable surface layer, warm moist advection",
           t_sfc=294.0, hfx=-25.0, qfx=8.0e-6, pbl_depth=300.0,
           pbl_kind="stable", stable_inv=3.0, lapse=7.2, z_trop=13000.0,
           rh=((0, 0.8), (400, 0.75), (1000, 0.92), (3000, 0.8),
               (8000, 0.4)), u0=8.0, u_shear=18.0, jet_z=900.0,
           jet_speed=20.0, w_amp=0.15, w_depth=9000.0, th_adv=3.0e-5,
           qv_adv=5.0e-8, rad_day=-2.0, rad_sfc_extra=-4.0, seed=20),
]

_BY_NAME = {r.name: r for r in REGIMES}


# ---------------------------------------------------------------------------
# edge cases: (name, base regime, overrides, build kwargs, post-hook, note)
# ---------------------------------------------------------------------------
def _post_qfloor(col):
    # dry slot and a bone-dry stratosphere: qv below GFDRV's 1e-8 floor,
    # including exact zeros
    qv = col.lev["qv"]
    qv[30:] = 0.0
    qv[22] = 1.0e-12
    qv[23] = 5.0e-9


def _post_tn_reset(col):
    # a cold tropical tropopause (below 200 K) and a strong advective
    # cooling that drives TN = T + (..)*pi*dt below 200 at levels just above
    # 200 K, exercising IF (TN < 200) TN = T2D at both kinds of level
    t = col.lev["t"]
    t[40:44] = np.array([197.0, 195.5, 199.0, 201.0])
    col.lev["rthften"][40:46] = -0.08


def _post_zero_forcing(col):
    for name in ("w", "rthften", "rqvften", "rthraten", "rthblten",
                 "rqvblten"):
        col.lev[name][:] = 0.0
    col.sca["hfx"] = 0.0
    col.sca["qfx"] = 0.0


def _post_supersat(col):
    t = col.lev["t"]
    p = col.lev["p"]
    e = 1.02 * _es(t[:8])
    col.lev["qv"][:8] = EPS * e / (p[:8] - (1.0 - EPS) * e)


def _post_258(col):
    t = col.lev["t"]
    k = int(np.argmin(np.abs(t - 258.0)))
    t[k] = 258.0
    t[k + 1] = np.nextafter(np.float32(258.0), np.float32(0.0))


EDGES = [
    ("moisture-floor", "tropical-ocean-deep", {}, {}, _post_qfloor,
     "qv below the 1e-8 floor (zeros, 1e-12, 5e-9) in a dry slot and aloft"),
    ("tn-reset-below-200k", "tropical-ocean-deep", {"t_trop_min": 0.0},
     {}, _post_tn_reset,
     "tropopause below 200 K plus strong cooling: the TN < 200 reset"),
    ("zero-forcing", "continental-severe", {}, {}, _post_zero_forcing,
     "hfx = qfx = 0, w = 0, every forcing lane 0"),
    ("kpbl-1", "trade-cumulus", {}, {"kpbl_override": 1}, None,
     "kpbl at its floor"),
    ("kpbl-deep", "desert-afternoon", {}, {"kpbl_override": 32}, None,
     "kpbl 32 of 50 under a deep dry mixed layer"),
    ("supersaturated", "midlatitude-cold-front", {}, {}, _post_supersat,
     "102 per cent relative humidity in the lowest 8 levels"),
    ("dew-night-unstable-aloft", "elevated-mcs-night",
     {"qfx": -1.2e-5, "hfx": -60.0}, {}, None,
     "negative latent and sensible flux (dew) under unstable air aloft"),
    ("below-sea-level", "desert-afternoon",
     {"ht": -420.0, "psfc": 106300.0, "t_sfc": 312.0,
      "rh": ((0, 0.25), (4500, 0.2), (8000, 0.15))}, {}, None,
     "terrain below sea level: ter11 = max(0, ht) clamps"),
    ("dx-500m", "continental-severe", {}, {"dx": 500.0, "dt": 3.0}, None,
     "sub-kilometre spacing, sig at its clamp"),
    ("dx-100km", "tropical-ocean-deep", {}, {"dx": 100000.0, "dt": 300.0},
     None, "100 km spacing, sig near 1"),
    ("strong-ascent", "monsoon-land", {"w_amp": 2.0, "w_depth": 9000.0}, {},
     None, "convective-scale ascent, 2 m/s"),
    ("strong-subsidence", "tropical-ocean-deep", {"w_amp": -0.4}, {}, None,
     "strong subsidence, -0.4 m/s"),
    ("extreme-shear", "continental-severe",
     {"u_shear": 45.0, "v_shear": -20.0}, {}, None,
     "45 m/s over 6 km"),
    ("warm-saturated", "tropical-cyclone-core",
     {"t_sfc": 305.0, "rh": ((0, 1.0), (6000, 1.0), (13000, 0.8))}, {},
     None, "saturated lower troposphere at 305 K"),
    ("exactly-258k", "winter-continental-snow", {"t_sfc": 275.0}, {},
     _post_258, "a level at exactly 258 K and one a ULP below it: the "
     "t2d < 258 split on both sides of equality"),
]

#: The convectively ACTIVE state of a regime: the same airmass in the phase
#: where Grell-Freitas actually fires (moister mid-levels, steeper lapse,
#: stronger large-scale ascent, or a shallow-cumulus cap).  The base
#: soundings are climatological and many of them are, correctly, rejected
#: by the trigger; without these the precipitating paths of the scheme would
#: be graded on a handful of columns.  Stable regimes (dry night, arctic,
#: suppressed tropics, dry high plains, desert) have no active state here:
#: they are in the set to be rejected.
_MOIST_MID = ((0, 0.7), (1500, 0.8), (4000, 0.7), (9000, 0.5), (12000, 0.3))
ACTIVE = {
    "tropical-ocean-deep": (dict(rh=((0, 0.82), (1000, 0.85), (3000, 0.85),
                                     (7000, 0.7), (12000, 0.65)),
                                 w_amp=0.09), {}),
    "tropical-cyclone-core": (dict(w_amp=1.0), {}),
    "continental-severe": (dict(lapse=8.5, rh=_MOIST_MID, w_amp=0.3), {}),
    "capped-elevated-mixed-layer": (dict(lapse=7.5, rh=_MOIST_MID,
                                         w_amp=0.3), {}),
    "winter-continental-snow": (dict(t_sfc=273.0, lapse=8.5, w_amp=0.3),
                                {}),
    "lake-effect-cold-air-outbreak": (dict(t_sfc=273.0, lapse=8.5,
                                           cap_z=0.0, w_amp=0.3), {}),
    "plateau-monsoon": (dict(lapse=8.5, rh=_MOIST_MID, w_amp=0.3), {}),
    "trade-cumulus": (dict(rh=((0, 0.9), (1400, 0.97), (2100, 0.3),
                               (8000, 0.2)), cap_z=1300.0, cap_dt=5.0,
                           cap_depth=300.0, pbl_depth=1200.0),
                      {"kpbl_override": 5}),
    "marine-stratocumulus": (dict(cap_z=1300.0, cap_dt=5.0, cap_depth=300.0,
                                  pbl_depth=1200.0), {"kpbl_override": 5}),
    "monsoon-land": (dict(rh=((0, 0.75), (1500, 0.95), (6000, 0.9),
                              (11000, 0.7)), w_amp=0.6), {}),
    "dryline-moist-side": (dict(lapse=8.5, rh=_MOIST_MID, w_amp=0.3), {}),
}

#: regimes for the closure sweep (clos_choice 1..16), in their active state
CLOSURE_SWEEP = ("continental-severe", "tropical-ocean-deep", "monsoon-land",
                 "midlatitude-cold-front")


def _state(name: str):
    """(regime, build kwargs) for a regime's active state, or its base."""
    r = _BY_NAME[name]
    if name in ACTIVE:
        over, kw = ACTIVE[name]
        return replace(r, **over), dict(kw)
    return r, {}


def all_columns() -> list[Column]:
    cols: list[Column] = []
    for r in REGIMES:
        states = [("base", r, {})]
        if r.name in ACTIVE:
            ar, kw = _state(r.name)
            states.append(("active", ar, kw))
        for state, rr, kw in states:
            for dx in (3000.0, 9000.0, 27000.0):
                for ish in (0, 1):
                    cols.append(build_column(
                        rr, dx=dx, ishallow=ish,
                        variant=f"{state}-dx{int(dx)}-sh{ish}", **kw))
    for name, base, over, kw, post, note in EDGES:
        # an edge case perturbs its base regime's ACTIVE state where it has
        # one, so the perturbed path is exercised on a convecting column
        rb, kwb = _state(base)
        r = replace(rb, name=name, **over)
        kw = {**kwb, **kw}
        dx = kw.pop("dx", 9000.0)
        for ish in (0, 1):
            c = build_column(r, dx=dx, ishallow=ish, post=post,
                             variant=f"edge-sh{ish}", **kw)
            c.notes.append(note)
            cols.append(c)
    for name in CLOSURE_SWEEP:
        r, kw = _state(name)
        for ich in range(1, 17):
            cols.append(build_column(r, dx=12000.0, ishallow=1, ichoice=ich,
                                     variant=f"active-clos{ich}", **kw))
    return cols


def pack(cols: list[Column]):
    n = len(cols)
    sc = np.zeros((n, len(SCA_FIELDS)), dtype=np.float32)
    ii = np.zeros((n, len(INT_FIELDS)), dtype=np.int32)
    lev = np.zeros((n, len(LEV_FIELDS), NZ), dtype=np.float32)
    p8w = np.zeros((n, NZ + 1), dtype=np.float32)
    for i, c in enumerate(cols):
        for j, name in enumerate(SCA_FIELDS):
            sc[i, j] = np.float32(c.sca[name])
        ii[i] = (c.kpbl, c.ishallow, c.ichoice)
        for j, name in enumerate(LEV_FIELDS):
            lev[i, j] = np.asarray(c.lev[name], dtype=np.float64).astype(
                np.float32)
        p8w[i] = c.p8w.astype(np.float32)
    return sc, ii, lev, p8w


def write(out_dir: Path) -> list[Column]:
    cols = all_columns()
    sc, ii, lev, p8w = pack(cols)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "gf-regime-inputs.bin").open("wb") as fh:
        np.array([MAGIC_IN, len(cols), NZ], dtype="<i4").tofile(fh)
        sc.astype("<f4").tofile(fh)
        ii.astype("<i4").tofile(fh)
        lev.astype("<f4").tofile(fh)
        p8w.astype("<f4").tofile(fh)
    manifest = []
    for i, c in enumerate(cols):
        t = lev[i, LEV_FIELDS.index("t")]
        manifest.append(dict(
            column=i, regime=c.regime, variant=c.variant, dx=c.dx, dt=c.dt,
            ishallow=c.ishallow, ichoice=c.ichoice, kpbl=c.kpbl,
            xland=c.sca["xland"], ht=c.sca["ht"], hfx=c.sca["hfx"],
            qfx=c.sca["qfx"], psfc_hpa=round(float(c.p8w[0]) / 100.0, 1),
            t_sfc=round(float(t[0]), 2), t_min=round(float(t.min()), 2),
            levels_below_258k=int((t < 258.0).sum()), notes=c.notes))
    (out_dir / "gf-regime-manifest.json").write_text(
        json.dumps(dict(nz=NZ, ncol=len(cols), lev_fields=LEV_FIELDS,
                        sca_fields=SCA_FIELDS, int_fields=INT_FIELDS,
                        columns=manifest), indent=1) + "\n",
        encoding="utf-8")
    return cols


def read_inputs(path: Path):
    raw = path.read_bytes()
    head = np.frombuffer(raw, dtype="<i4", count=3)
    if int(head[0]) != MAGIC_IN:
        raise ValueError(f"{path}: bad magic {int(head[0])}")
    n, nz = int(head[1]), int(head[2])
    off = 12
    sc = np.frombuffer(raw, "<f4", n * len(SCA_FIELDS), off).reshape(n, -1)
    off += sc.nbytes
    ii = np.frombuffer(raw, "<i4", n * len(INT_FIELDS), off).reshape(n, -1)
    off += ii.nbytes
    lev = np.frombuffer(raw, "<f4", n * len(LEV_FIELDS) * nz, off).reshape(
        n, len(LEV_FIELDS), nz)
    off += lev.nbytes
    p8w = np.frombuffer(raw, "<f4", n * (nz + 1), off).reshape(n, nz + 1)
    off += p8w.nbytes
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return dict(sc=sc.copy(), ii=ii.copy(), lev=lev.copy(), p8w=p8w.copy(),
                ncol=n, nz=nz)


OUT_LEV = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "dudt_phy",
           "dvdt_phy", "gdc", "gdc2")
OUT_SCA = ("raincv", "pratec", "htop", "hbot", "xmb_shallow")
OUT_INT = ("ktop_deep", "k22_shallow", "kbcon_shallow", "ktop_shallow")


def read_outputs(path: Path):
    raw = path.read_bytes()
    head = np.frombuffer(raw, dtype="<i4", count=4)
    if int(head[0]) != 47012:
        raise ValueError(f"{path}: bad magic {int(head[0])}")
    n, nz, mode = int(head[1]), int(head[2]), int(head[3])
    off = 16
    lo = np.frombuffer(raw, "<f4", n * len(OUT_LEV) * nz, off).reshape(
        n, len(OUT_LEV), nz)
    off += lo.nbytes
    so = np.frombuffer(raw, "<f4", n * len(OUT_SCA), off).reshape(n, -1)
    off += so.nbytes
    io = np.frombuffer(raw, "<i4", n * len(OUT_INT), off).reshape(n, -1)
    off += io.nbytes
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return dict(lev=lo.copy(), sca=so.copy(), isc=io.copy(), ncol=n, nz=nz,
                kte_mode=mode)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    cs = write(Path(sys.argv[1]))
    print(f"wrote {len(cs)} columns x {NZ} levels")
