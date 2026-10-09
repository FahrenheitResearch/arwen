"""Column set for the WRF 4.6.1 MYNN surface-layer column oracle.

One deterministic list of single columns, each a complete set of inputs to
``SFCLAY1D_mynn`` (module_sf_mynn.F:370) and to WOOF's
``mynn_surface_column`` kernel.  The same float32 bits go to both sides: the
Fortran driver reads ``columns.bin`` written from these arrays, and the GPU
comparison reads the oracle file that echoes them back.

The set is hand-picked edge columns (one per branch or boundary the scheme
has) followed by seeded random columns drawn from six realistic regimes.  It
is pure NumPy so the desktop, the box and the test suite build the identical
set; ``columns.bin`` carries a SHA-256 in the oracle receipt.

Configs run every column through several option sets and several timesteps,
carrying the INOUT state (UST, MOL, QSFC, ZNT, USTM, HFX, QFX) from step to
step exactly as WRF does between calls.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

F = np.float32

#: Static per-column forcing, in the order the Fortran driver reads it.
STATIC_FIELDS = (
    "u1", "v1", "t1", "qv1", "p1", "rho1", "dz1", "u2", "v2", "dz2",
    "psfc", "tsk", "pblh", "mavail", "xland", "snowh",
)
#: INOUT state that enters a call and is carried to the next one.
STATE_FIELDS = ("hfx", "qfx", "znt", "qsfc", "ust", "mol", "ustm")
FIELDS = STATIC_FIELDS + STATE_FIELDS

#: Outputs recorded per (config, step), WOOF's MYNN_SURFACE_OUTPUTS order.
OUTPUT_FIELDS = (
    "regime", "zol", "rmol", "ust", "ustm", "mol", "psim", "psih",
    "chs", "chs2", "cqs2", "ch", "flhc", "flqc", "qgh", "qsfc",
    "hfx", "qfx", "lh", "u10", "v10", "th2", "t2", "q2",
    "gz1oz0", "wspd", "br", "ck", "cka", "cd", "cda", "wstar",
    "qstar", "cpm", "znt",
)

#: (name, isftcflx, isfflx, dx, first itimestep, seeded, steps, spp_pbl).
#: ``seeded`` runs the SFCLAY_mynn wrapper's itimestep==1 block
#: (module_sf_mynn.F:330-337) before the first call; an unseeded config is a
#: restart: it enters with the column's own UST/MOL/QSFC/USTM.  ``spp_pbl``
#: 1 runs WRF's stochastic roughness (ZNTstoch, :665-668 and :715-718, and
#: the rstoch terms of fairall_etal_2003 and zilitinkevich_1995) with the
#: fixed pattern :func:`spp_pattern`.
CONFIGS = (
    ("coare30_dx3km", 0, 1, 3000.0, 1, True, 3, 0),
    ("davis_coare_dx3km", 1, 1, 3000.0, 1, True, 3, 0),
    ("davis_garratt_dx3km", 2, 1, 3000.0, 1, True, 3, 0),
    ("taylor_yelland_dx3km", 3, 1, 3000.0, 1, True, 3, 0),
    ("fluxes_off_dx3km", 0, 0, 3000.0, 1, True, 3, 0),
    ("coare30_dx12km", 0, 1, 12000.0, 1, True, 3, 0),
    ("restart_davis_dx25km", 1, 1, 25000.0, 5, False, 3, 0),
    ("restart_coare30_dx3km", 0, 1, 3000.0, 2, False, 2, 0),
    ("spp_coare30_dx3km", 0, 1, 3000.0, 1, True, 3, 1),
    ("spp_davis_garratt_dx12km", 2, 1, 12000.0, 1, True, 3, 1),
    ("spp_restart_taylor_yelland_dx3km", 3, 1, 3000.0, 4, False, 2, 1),
)


def spp_pattern(ncol: int) -> np.ndarray:
    """The SPP pattern run_columns.F90 builds: 0.15*REAL(MOD(i-1,7)-3).

    One float32 product per column on both sides, so the bits agree."""
    k = (np.arange(ncol) % 7 - 3).astype(F)
    return (F(0.15) * k).astype(F)

R_D, R_V = F(287.0), F(461.6)
EP1 = F(R_V / R_D - F(1.0))


def _rho(p1, t1, qv1):
    qvsh = qv1 / (F(1.0) + qv1)
    return (p1 / (R_D * t1 * (F(1.0) + EP1 * qvsh))).astype(F)


def _column(**kw):
    base = dict(
        u1=5.0, v1=1.0, t1=295.0, qv1=0.008, p1=99000.0, dz1=40.0,
        u2=6.5, v2=1.3, dz2=60.0, psfc=99500.0, tsk=297.0, pblh=800.0,
        mavail=0.5, xland=1.0, snowh=0.0, hfx=50.0, qfx=3.0e-5, znt=0.1,
        qsfc=0.0, ust=0.3, mol=0.0, ustm=0.3,
    )
    unknown = set(kw) - set(base) - {"rho1"}
    if unknown:
        raise KeyError(f"unknown column fields {sorted(unknown)}")
    base.update(kw)
    col = {k: F(v) for k, v in base.items()}
    if "rho1" not in kw:
        col["rho1"] = _rho(col["p1"], col["t1"], col["qv1"])
    if "qsfc" not in kw:
        col["qsfc"] = F(col["qv1"] / (F(1.0) + col["qv1"]))
    return col


def _edge_columns():
    c = []
    add = lambda name, **kw: c.append((name, _column(**kw)))  # noqa: E731
    # Exactly neutral land: T1==TSK, P1==PSFC and (after seeding)
    # QSFC==QVSH, so DTHVDZ cancels to zero and REGIME=3 (:851).
    add("neutral_exact_land", t1=290.0, tsk=290.0, p1=99000.0,
        psfc=99000.0, hfx=0.0, qfx=0.0)
    add("convective_desert", u1=1.0, v1=0.5, t1=305.0, tsk=330.0,
        qv1=0.003, hfx=450.0, qfx=1.0e-5, pblh=3500.0, mavail=0.05,
        znt=0.05)
    add("convective_no_wind", u1=0.0, v1=0.0, u2=0.0, v2=0.0, t1=300.0,
        tsk=318.0, hfx=350.0, pblh=2000.0)
    add("calm_floor_wmin", u1=0.0, v1=0.0, u2=0.0, v2=0.0, t1=290.0,
        tsk=289.5, hfx=0.0, qfx=0.0, pblh=100.0)
    add("stable_night_clear", u1=2.0, v1=0.3, t1=278.0, tsk=270.0,
        qv1=0.004, hfx=-30.0, qfx=-2.0e-6, pblh=150.0, mavail=0.6)
    add("very_stable_clip", u1=0.5, v1=0.2, t1=275.0, tsk=255.0,
        qv1=0.003, hfx=-10.0, qfx=-1.0e-6, pblh=50.0)
    add("damped_stable", u1=8.0, v1=2.0, t1=290.0, tsk=289.0,
        hfx=-15.0, qfx=0.0, pblh=500.0)
    add("snow_stable", u1=4.0, v1=1.0, t1=262.0, tsk=260.0, qv1=0.0015,
        snowh=0.5, znt=0.001, mavail=1.0, hfx=-20.0, qfx=-1.0e-6,
        pblh=200.0)
    add("snow_unstable", u1=3.0, v1=-1.0, t1=266.0, tsk=271.0,
        qv1=0.002, snowh=0.3, znt=0.002, mavail=1.0, hfx=40.0,
        qfx=1.0e-5, pblh=600.0)
    add("snow_depth_exact_0p1", u1=5.0, t1=268.0, tsk=267.0, qv1=0.002,
        snowh=0.1, znt=0.003, mavail=1.0)
    add("snow_depth_below_0p1", u1=5.0, t1=268.0, tsk=267.0, qv1=0.002,
        snowh=0.0999, znt=0.003, mavail=1.0)
    add("snow_low_ustar", u1=0.3, v1=0.1, t1=263.0, tsk=262.5,
        qv1=0.0018, snowh=0.8, znt=1.0e-4, mavail=1.0, hfx=-2.0,
        ust=0.01, ustm=0.01)
    add("snow_high_ustar", u1=25.0, v1=10.0, u2=30.0, v2=12.0, t1=258.0,
        tsk=256.0, qv1=0.001, snowh=1.2, znt=0.005, mavail=1.0,
        hfx=-60.0, ust=1.5, ustm=1.5)
    add("ocean_calm_warm", u1=0.3, v1=0.1, u2=0.4, v2=0.1, t1=297.0,
        tsk=300.0, qv1=0.016, xland=2.0, znt=2.0e-4, mavail=1.0,
        hfx=20.0, qfx=6.0e-5, pblh=600.0)
    add("ocean_hurricane", u1=45.0, v1=20.0, u2=50.0, v2=22.0, t1=298.0,
        tsk=302.0, qv1=0.018, p1=96000.0, psfc=96500.0, xland=2.0,
        znt=2.0e-3, mavail=1.0, hfx=200.0, qfx=4.0e-4, pblh=1500.0,
        ust=2.0, ustm=2.0)
    add("ocean_stable_fog", u1=6.0, v1=2.0, t1=288.0, tsk=283.0,
        qv1=0.0105, xland=2.0, znt=2.0e-4, mavail=1.0, hfx=-25.0,
        qfx=-5.0e-6, pblh=150.0)
    add("ocean_cold_air_outbreak", u1=15.0, v1=-5.0, u2=17.0, v2=-6.0,
        t1=265.0, tsk=285.0, qv1=0.002, xland=2.0, znt=5.0e-4,
        mavail=1.0, hfx=300.0, qfx=1.5e-4, pblh=1200.0)
    add("ocean_moderate", u1=9.0, v1=3.0, t1=292.0, tsk=292.5,
        qv1=0.0110, xland=2.0, znt=2.0e-4, mavail=1.0, hfx=5.0,
        qfx=3.0e-5, pblh=700.0)
    add("sea_ice_snow", u1=7.0, v1=1.0, t1=245.0, tsk=250.0,
        qv1=0.0004, snowh=0.2, znt=0.001, mavail=1.0, hfx=30.0,
        qfx=2.0e-6, pblh=300.0)
    add("xland_exactly_1p5", u1=6.0, v1=2.0, t1=290.0, tsk=292.0,
        xland=1.5, znt=0.01, hfx=33.0, qfx=2.0e-5)
    add("high_terrain_convective", u1=4.0, v1=2.0, t1=270.0, tsk=285.0,
        qv1=0.003, p1=61500.0, psfc=62000.0, znt=0.5, hfx=300.0,
        pblh=2500.0, mavail=0.2)
    add("high_terrain_night", u1=3.0, v1=-1.0, t1=265.0, tsk=255.0,
        qv1=0.0015, p1=64700.0, psfc=65000.0, znt=0.3, hfx=-40.0,
        qfx=-1.0e-6, pblh=100.0, mavail=0.3)
    add("thin_za_level2_wind", dz1=6.0, dz2=10.0, u2=6.0, v2=1.2)
    add("thin_za_log_wind", dz1=4.0, dz2=30.0)
    add("za_exactly_7", dz1=14.0, dz2=20.0)
    add("za_mid_res", dz1=20.0, dz2=30.0)
    add("za_exactly_13", dz1=26.0, dz2=40.0)
    add("za_coarse", dz1=120.0, dz2=150.0, u2=7.0, v2=1.5)
    add("dry_column_qv_zero", qv1=0.0, hfx=200.0, qfx=0.0, t1=300.0,
        tsk=310.0, mavail=0.0)
    add("tall_roughness_urban", u1=6.0, v1=2.0, dz1=20.0, znt=2.5,
        hfx=150.0, t1=296.0, tsk=303.0)
    add("tiny_roughness_land", znt=1.0e-5, u1=10.0, v1=0.0, t1=290.0,
        tsk=291.0)
    add("unstable_full_function", u1=0.5, v1=0.2, dz1=120.0, dz2=150.0,
        t1=295.0, tsk=320.0, hfx=500.0, pblh=3000.0, mavail=0.1,
        znt=0.01)
    add("stable_full_function", u1=1.5, v1=0.0, dz1=120.0, dz2=150.0,
        t1=280.0, tsk=262.0, hfx=-15.0, qfx=-1.0e-6, pblh=60.0,
        znt=0.2)
    add("hfx_floor_minus250", u1=20.0, v1=5.0, u2=24.0, v2=6.0, t1=280.0,
        tsk=262.0, qv1=0.004, znt=1.0, hfx=-200.0, pblh=300.0)
    add("tsk_exactly_27315", t1=273.15, tsk=273.15, qv1=0.0035,
        p1=99000.0, psfc=99500.0, hfx=5.0)
    add("tsk_just_below_freezing", t1=275.0, tsk=273.14999, qv1=0.004,
        hfx=10.0)
    add("negative_winds", u1=-7.0, v1=-4.0, u2=-9.0, v2=-5.0, t1=291.0,
        tsk=294.0)
    add("supersaturated_air_dew", u1=12.0, v1=0.0, t1=285.0, tsk=275.0,
        qv1=0.012, xland=2.0, znt=2.0e-4, mavail=1.0, hfx=-50.0,
        qfx=-2.0e-5, pblh=200.0)
    add("big_negative_qfx_land", u1=15.0, v1=0.0, t1=295.0, tsk=280.0,
        qv1=0.02, znt=0.8, mavail=1.0, hfx=-100.0, qfx=-1.0e-4,
        pblh=200.0)
    # Restart configs enter with this QSFC<=0 land point: :532 recomputes
    # QSFC from saturation and :573 then sees the updated value.
    add("land_qsfc_unset", qsfc=0.0, t1=297.0, tsk=302.0, hfx=150.0,
        qfx=8.0e-5, pblh=900.0)
    # zolrib's 20-pass search fails to converge on these (found by a
    # probe-instrumented copy of the module over 12,000 random stable
    # columns), so :2036-2039 falls back to Li_etal_2010.  Exact float32
    # values; a restart enters them with QSFC=0.
    for k, row in enumerate((
            dict(u1=3.5352330207824707, t1=257.7168273925781, rho1=1.3320444822311401, dz1=21.932498931884766, dz2=28.512248992919922, tsk=230.10214233398438, pblh=2471.10498046875, hfx=180.05886840820312, znt=1.7589380741119385, ust=0.9669567346572876, mol=-0.7062673568725586),
            dict(u1=2.0321884155273438, t1=299.332275390625, rho1=1.1468535661697388, dz1=58.41596603393555, dz2=75.94075775146484, tsk=295.76934814453125, pblh=799.6563720703125, hfx=24.15023422241211, znt=2.380385398864746, ust=0.9843752384185791, mol=0.7474886775016785),
            dict(u1=2.538501739501953, t1=256.6702880859375, rho1=1.3374756574630737, dz1=31.480613708496094, dz2=40.92479705810547, tsk=235.29881286621094, pblh=2207.77880859375, hfx=308.9805908203125, znt=3.3458025455474854, ust=0.8406614661216736, mol=-1.989466905593872),
            dict(u1=1.400272250175476, t1=271.0036926269531, rho1=1.266736388206482, dz1=11.056283950805664, dz2=14.3731689453125, tsk=243.22512817382812, pblh=1983.15234375, hfx=359.2991943359375, znt=1.0823184251785278, ust=0.27918460965156555, mol=-1.0487682819366455),
            dict(u1=4.028586387634277, t1=275.7077941894531, rho1=1.2451235055923462, dz1=16.014554977416992, dz2=20.818920135498047, tsk=238.19647216796875, pblh=2277.134521484375, hfx=367.5329284667969, znt=0.733895480632782, ust=0.9975560903549194, mol=-0.6694930195808411),
            dict(u1=1.3180826902389526, t1=298.88128662109375, rho1=1.148584008216858, dz1=50.38382339477539, dz2=65.49897003173828, tsk=295.8683776855469, pblh=1946.5245361328125, hfx=14.953125, znt=2.845379114151001, ust=0.47797128558158875, mol=0.2702760398387909),
    )):
        add(f"zolrib_nonconvergent_{k}", v1=0.0, qsfc=0.0, ustm=0.3,
            **row)
    return c


#: Six regime samplers for the random columns: (name, count, sampler).
def _random_columns(rng):
    out = []

    def u(lo, hi):
        return float(rng.uniform(lo, hi))

    def wind(lo, hi):
        speed, angle = u(lo, hi), u(0.0, 2.0 * np.pi)
        return speed * np.cos(angle), speed * np.sin(angle)

    def pressure_pair(psfc, dz1):
        return psfc - 0.5 * dz1 * 11.5 * psfc / 100000.0

    for k in range(14):  # convective land, daytime
        dz1 = u(8.0, 60.0); psfc = u(85000.0, 102000.0)
        u1, v1 = wind(0.5, 9.0)
        t1 = u(280.0, 310.0)
        out.append((f"land_day_{k:02d}", _column(
            u1=u1, v1=v1, u2=u1 * 1.2, v2=v1 * 1.2, t1=t1,
            tsk=t1 + u(1.0, 25.0), qv1=u(0.002, 0.020),
            p1=pressure_pair(psfc, dz1), psfc=psfc, dz1=dz1,
            dz2=dz1 * u(1.1, 1.6), pblh=u(500.0, 3500.0),
            mavail=u(0.05, 1.0), znt=10.0 ** u(-3.0, 0.3),
            hfx=u(20.0, 450.0), qfx=u(1.0e-6, 2.0e-4))))
    for k in range(14):  # stable land, night
        dz1 = u(8.0, 60.0); psfc = u(85000.0, 102000.0)
        u1, v1 = wind(0.2, 10.0)
        t1 = u(265.0, 300.0)
        out.append((f"land_night_{k:02d}", _column(
            u1=u1, v1=v1, u2=u1 * 1.3, v2=v1 * 1.3, t1=t1,
            tsk=t1 - u(0.2, 15.0), qv1=u(0.001, 0.015),
            p1=pressure_pair(psfc, dz1), psfc=psfc, dz1=dz1,
            dz2=dz1 * u(1.1, 1.6), pblh=u(30.0, 600.0),
            mavail=u(0.1, 1.0), znt=10.0 ** u(-3.0, 0.0),
            hfx=-u(0.0, 80.0), qfx=-u(0.0, 1.0e-5),
            ust=u(0.02, 0.5), mol=u(0.0, 0.5))))
    for k in range(14):  # open water, mixed stability
        dz1 = u(8.0, 60.0); psfc = u(97000.0, 103000.0)
        u1, v1 = wind(0.3, 35.0)
        t1 = u(270.0, 303.0)
        out.append((f"water_{k:02d}", _column(
            u1=u1, v1=v1, u2=u1 * 1.1, v2=v1 * 1.1, t1=t1,
            tsk=u(271.5, 305.0), qv1=u(0.002, 0.022),
            p1=pressure_pair(psfc, dz1), psfc=psfc, dz1=dz1,
            dz2=dz1 * u(1.1, 1.6), pblh=u(100.0, 1800.0), mavail=1.0,
            xland=2.0, znt=10.0 ** u(-5.0, -2.6), hfx=u(-40.0, 250.0),
            qfx=u(-5.0e-6, 2.5e-4), ust=u(0.01, 1.5),
            mol=u(-1.0, 0.3))))
    for k in range(8):  # snow and ice on land
        dz1 = u(8.0, 40.0); psfc = u(80000.0, 101000.0)
        u1, v1 = wind(0.3, 15.0)
        t1 = u(240.0, 275.0)
        out.append((f"snow_{k:02d}", _column(
            u1=u1, v1=v1, u2=u1 * 1.2, v2=v1 * 1.2, t1=t1,
            tsk=min(t1 + u(-12.0, 6.0), 273.15), qv1=u(2.0e-4, 0.004),
            p1=pressure_pair(psfc, dz1), psfc=psfc, dz1=dz1,
            dz2=dz1 * u(1.1, 1.6), pblh=u(50.0, 800.0), mavail=1.0,
            snowh=u(0.1, 2.0), znt=10.0 ** u(-4.0, -2.0),
            hfx=u(-60.0, 60.0), qfx=u(-3.0e-6, 2.0e-5))))
    for k in range(8):  # high terrain
        dz1 = u(10.0, 50.0); psfc = u(55000.0, 75000.0)
        u1, v1 = wind(0.5, 20.0)
        t1 = u(250.0, 290.0)
        out.append((f"terrain_{k:02d}", _column(
            u1=u1, v1=v1, u2=u1 * 1.25, v2=v1 * 1.25, t1=t1,
            tsk=t1 + u(-12.0, 18.0), qv1=u(5.0e-4, 0.008),
            p1=pressure_pair(psfc, dz1), psfc=psfc, dz1=dz1,
            dz2=dz1 * u(1.1, 1.6), pblh=u(50.0, 3000.0),
            mavail=u(0.05, 0.6), znt=10.0 ** u(-2.0, 0.2),
            hfx=u(-80.0, 350.0), qfx=u(-2.0e-6, 6.0e-5))))
    for k in range(6):  # thin first layer (high vertical resolution)
        dz1 = u(2.0, 16.0); psfc = u(95000.0, 102000.0)
        u1, v1 = wind(0.5, 12.0)
        t1 = u(270.0, 305.0)
        out.append((f"thin_layer_{k:02d}", _column(
            u1=u1, v1=v1, u2=u1 * 1.15, v2=v1 * 1.15, t1=t1,
            tsk=t1 + u(-8.0, 12.0), qv1=u(0.002, 0.016),
            p1=pressure_pair(psfc, dz1), psfc=psfc, dz1=dz1,
            dz2=u(4.0, 30.0), pblh=u(50.0, 2000.0),
            mavail=u(0.1, 1.0), xland=2.0 if k % 3 == 0 else 1.0,
            znt=1.0e-4 if k % 3 == 0 else 10.0 ** u(-2.5, -0.5),
            hfx=u(-40.0, 200.0), qfx=u(-2.0e-6, 8.0e-5))))
    return out


def build_columns():
    """(names, {field: float32[ncol]}) for the whole set, in fixed order."""
    rng = np.random.default_rng(20261007)
    cols = _edge_columns() + _random_columns(rng)
    names = [name for name, _ in cols]
    if len(set(names)) != len(names):
        raise ValueError("duplicate column names")
    arrays = {f: np.array([col[f] for _, col in cols], dtype=F)
              for f in FIELDS}
    return names, arrays


def columns_bytes(arrays) -> bytes:
    """``int32 ncol, int32 nfield`` then float32[ncol][nfield], little-endian."""
    ncol = len(arrays[FIELDS[0]])
    table = np.stack([arrays[f] for f in FIELDS], axis=1).astype("<f4")
    return struct.pack("<ii", ncol, len(FIELDS)) + table.tobytes(order="C")


def configs_text() -> str:
    lines = [str(len(CONFIGS))]
    for name, isftcflx, isfflx, dx, it0, seeded, steps, spp in CONFIGS:
        lines.append(f"{isftcflx} {isfflx} {dx:.1f} {it0} "
                     f"{1 if seeded else 0} {steps} {spp} {name}")
    return "\n".join(lines) + "\n"


def write_inputs(directory: Path) -> dict:
    names, arrays = build_columns()
    directory.mkdir(parents=True, exist_ok=True)
    blob = columns_bytes(arrays)
    (directory / "columns.bin").write_bytes(blob)
    (directory / "configs.txt").write_text(configs_text(), encoding="ascii")
    (directory / "column-names.txt").write_text(
        "\n".join(names) + "\n", encoding="ascii")
    return {"ncol": len(names),
            "columns_sha256": hashlib.sha256(blob).hexdigest()}


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(write_inputs(Path(sys.argv[1]))))
