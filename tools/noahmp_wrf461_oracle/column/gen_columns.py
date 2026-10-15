"""Column-oracle inputs for the WRF v4.6.1 ``noahmplsm`` driver.

Writes one tagged binary file that ``run_column.F90`` reads and that
``compare_column.py`` reads back, so WRF's Fortran and WOOF's device path
start from the same bits.  Every record is

    name    16 bytes, ASCII, blank padded
    kind    int32   1 = float32, 2 = int32
    count   int32
    values  count x 4 bytes, little endian

The column set spans the regimes the land model has distinct branches for:
sunlit convective afternoons, stable nights, shallow (sub-layer) snow,
layered snow, melting snow and rain on snow, frozen soil, glaciers,
deserts, saturated wetlands, high terrain, urban cells, the southern
hemisphere, open water and sea ice, plus edge cases (calm wind, grazing
sun, hail and graupel, a SWE above the 2000 mm cap, soil type 14 at a land
point, wilting-point and above-porosity soil).  A seeded generator varies
values inside each regime, so the file is reproducible from the seed.

Usage:  python gen_columns.py OUT.bin --config {summer60,winter300}
"""

from __future__ import annotations

import argparse
import struct
from datetime import datetime, timedelta

import numpy as np

NSOIL = 4

#: Two step configurations.  ``summer60`` is a July afternoon start at a
#: convective-scale step; ``winter300`` a January start at a coarse step,
#: which moves PHENOLOGY's month interpolation and every flux magnitude.
CONFIGS = {
    "summer60": dict(start=datetime(2026, 7, 1, 18, 0, 0), dt=60.0,
                     nstep=36, dx=3000.0, seed=461),
    "winter300": dict(start=datetime(2026, 1, 15, 15, 0, 0), dt=300.0,
                      nstep=24, dx=12000.0, seed=4610),
}

#: Fields that hold for the whole run (surface identity and initial state).
STATIC_FIELDS = (
    "xland", "xice", "tsk", "tmn", "snow", "snowh", "lai", "xlat", "xlong",
    "vegfra", "shdmax",
)
#: Fields that are new every step (the atmosphere and precipitation the
#: land model is handed).  Stored step-major: (nstep, ncol).
FORCING_FIELDS = (
    "coszen", "t3d", "qv3d", "u", "v", "dz8w", "p8w1", "p8w2", "swdown",
    "glw", "rainbl", "sr", "rainc", "rainnc", "rainsh", "snownc",
    "graupnc", "hailnc",
)
PRECIP_FIELDS = ("rainbl", "rainc", "rainnc", "rainsh", "snownc",
                 "graupnc", "hailnc")
INT_FIELDS = ("ivgtyp", "isltyp")
LAYER_FIELDS = ("tslb", "smois")


def _column(**kw):
    base = dict(
        ivgtyp=10, isltyp=6, xland=1.0, xice=0.0, tsk=300.0, tmn=288.0,
        tslb=(299.0, 297.0, 294.0, 290.0), smois=(0.25, 0.26, 0.27, 0.28),
        snow=0.0, snowh=0.0, lai=2.0, xlat=40.0, xlong=-100.0,
        vegfra=60.0, shdmax=70.0,
        coszen=0.8, t3d=298.0, qv3d=0.012, u=4.0, v=1.0, dz8w=50.0,
        p8w1=98000.0, p8w2=97400.0, swdown=750.0, glw=380.0,
        rainbl=0.0, sr=0.0, rainc=0.0, rainnc=0.0, rainsh=0.0, snownc=0.0,
        graupnc=0.0, hailnc=0.0, regime="")
    base.update(kw)
    return base


def build_columns(seed: int, dt: float):
    rng = np.random.default_rng(seed)
    u = lambda a, b: float(rng.uniform(a, b))  # noqa: E731
    cols = []
    scale = dt / 60.0   # precipitation per step scales with the step

    # --- sunlit convective afternoons, vegetated, north hemisphere --------
    veg_soil = [(1, 6), (2, 9), (4, 4), (5, 6), (8, 3), (9, 7), (10, 6),
                (12, 8), (14, 4), (6, 2), (7, 3), (3, 6), (18, 6), (19, 4)]
    for n, (veg, soil) in enumerate(veg_soil):
        rain = n % 3 == 0
        t_air = u(298.0, 307.0)
        sm = u(0.15, 0.38)
        cols.append(_column(
            regime="convective", ivgtyp=veg, isltyp=soil,
            tsk=t_air + u(3.0, 10.0), tmn=u(285.0, 295.0),
            tslb=(t_air + u(-1, 3), t_air - u(1, 4), t_air - u(4, 8),
                  t_air - u(8, 12)),
            smois=(sm, sm + u(0, 0.03), sm + u(0, 0.05), sm + u(0, 0.06)),
            lai=u(0.5, 5.5), vegfra=u(20, 95), shdmax=u(30, 99),
            xlat=u(25, 55), coszen=u(0.55, 0.97), swdown=u(550, 980),
            glw=u(360, 440), t3d=t_air, qv3d=u(0.008, 0.019),
            u=u(-8, 8), v=u(-8, 8), dz8w=u(30, 80),
            p8w1=u(93000, 101500), p8w2=0.0,
            rainc=(u(0.05, 0.6) * scale if rain else 0.0),
            rainsh=(u(0.0, 0.05) * scale if rain else 0.0),
            rainnc=(u(0.0, 0.2) * scale if rain else 0.0)))

    # --- stable nights -----------------------------------------------------
    for n, (veg, soil) in enumerate([(10, 6), (12, 4), (1, 3), (4, 9),
                                     (16, 1), (8, 6), (14, 8), (11, 12)]):
        t_air = u(276.0, 292.0)
        sm = u(0.12, 0.40)
        cols.append(_column(
            regime="stable_night", ivgtyp=veg, isltyp=soil,
            tsk=t_air - u(2.0, 7.0), tmn=u(282.0, 292.0),
            tslb=(t_air + u(0, 3), t_air + u(2, 5), t_air + u(3, 6),
                  t_air + u(4, 8)),
            smois=(sm, sm, sm + 0.02, sm + 0.04),
            lai=u(0.3, 4.0), vegfra=u(10, 90), shdmax=u(20, 95),
            xlat=u(30, 60), coszen=0.0, swdown=0.0, glw=u(240, 330),
            t3d=t_air, qv3d=u(0.003, 0.010), u=u(-2, 2), v=u(-2, 2),
            dz8w=u(25, 60), p8w1=u(95000, 102000), p8w2=0.0,
            rainnc=(0.02 * scale if n == 3 else 0.0)))

    # --- snow regimes ------------------------------------------------------
    snow_cases = [
        # (regime, veg, soil, swe mm, depth m, t_air, tsk, sr, precip/step)
        ("snow_sublayer", 10, 6, 4.0, 0.02, 268.0, 267.0, 0.0, 0.0),
        ("snow_sublayer", 16, 3, 6.0, 0.022, 270.0, 269.0, 1.0, 0.03),
        ("snow_one_layer", 10, 4, 12.0, 0.05, 265.0, 264.0, 1.0, 0.05),
        ("snow_two_layer", 1, 6, 50.0, 0.20, 263.0, 262.0, 1.0, 0.08),
        ("snow_two_layer", 5, 8, 80.0, 0.32, 258.0, 257.0, 0.0, 0.0),
        ("snow_three_layer", 1, 4, 300.0, 1.1, 255.0, 254.0, 1.0, 0.1),
        ("snow_three_layer", 19, 6, 500.0, 1.6, 250.0, 249.0, 0.0, 0.0),
        ("snow_melt", 10, 6, 30.0, 0.12, 280.0, 274.0, 0.0, 0.0),
        ("snow_melt", 4, 4, 120.0, 0.40, 283.0, 275.0, 0.0, 0.0),
        ("rain_on_snow", 12, 8, 60.0, 0.22, 277.0, 273.5, 0.0, 0.15),
        ("rain_on_snow", 1, 9, 200.0, 0.70, 276.0, 273.0, 0.3, 0.2),
        ("snow_cap_swe", 18, 6, 2600.0, 6.5, 250.0, 248.0, 1.0, 0.05),
        ("snow_cold_clear", 7, 3, 25.0, 0.10, 245.0, 240.0, 0.0, 0.0),
        ("snow_cold_clear", 20, 1, 15.0, 0.06, 248.0, 244.0, 0.0, 0.0),
        ("snowfall_bare", 16, 6, 0.0, 0.0, 268.0, 270.0, 1.0, 0.12),
        ("snowfall_bare", 10, 4, 0.0, 0.0, 271.0, 272.0, 0.8, 0.10),
    ]
    for regime, veg, soil, swe, depth, t_air, tsk, sr, pr in snow_cases:
        sunlit = rng.uniform() < 0.6
        cz = u(0.2, 0.55) if sunlit else 0.0
        tsoil = min(tsk + 1.0, 272.5)
        sm = u(0.20, 0.35)
        precip = pr * scale
        cols.append(_column(
            regime=regime, ivgtyp=veg, isltyp=soil, tsk=tsk,
            tmn=u(272.0, 280.0),
            tslb=(tsoil, tsoil + 0.8, tsoil + 1.6, tsoil + 3.0),
            smois=(sm, sm, sm + 0.02, sm + 0.03), snow=swe, snowh=depth,
            lai=u(0.0, 3.0), vegfra=u(5, 80), shdmax=u(10, 90),
            xlat=u(40, 68), coszen=cz, swdown=(u(150, 450) if sunlit else 0.0),
            glw=u(180, 300), t3d=t_air, qv3d=u(0.0008, 0.0045),
            u=u(-6, 6), v=u(-6, 6), dz8w=u(30, 70),
            p8w1=u(85000, 101000), p8w2=0.0,
            rainbl=precip, sr=sr, rainnc=precip,
            snownc=precip * sr))

    # --- frozen soil without snow -------------------------------------------
    for veg, soil in [(10, 6), (12, 9), (16, 3), (6, 11)]:
        t_air = u(262.0, 270.0)
        cols.append(_column(
            regime="frozen_soil", ivgtyp=veg, isltyp=soil,
            tsk=t_air - 1.0, tmn=u(272.0, 278.0),
            tslb=(t_air + 1.0, t_air + 3.0, 271.5, 273.5),
            smois=(u(0.25, 0.40), u(0.25, 0.40), u(0.25, 0.40), 0.30),
            lai=u(0.2, 1.5), vegfra=u(5, 50), shdmax=u(10, 60),
            xlat=u(45, 65), coszen=u(0.0, 0.4),
            swdown=u(0.0, 300.0), glw=u(200, 280), t3d=t_air,
            qv3d=u(0.001, 0.003), u=u(-5, 5), v=u(-5, 5), dz8w=u(30, 60),
            p8w1=u(95000, 102000), p8w2=0.0))

    # --- glaciers (ISICE_TABLE = 15) -----------------------------------------
    glacier_cases = [
        ("glacier_sunlit", 250.0, 248.0, 0.0, 0.0, 0.7, 700.0, 0.0),
        ("glacier_night", 240.0, 238.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ("glacier_snowfall", 258.0, 257.0, 1.0, 0.1, 0.3, 250.0, 0.0),
        ("glacier_melt", 276.0, 271.0, 0.0, 0.0, 0.8, 850.0, 0.0),
        ("glacier_rain", 275.0, 270.0, 0.0, 0.2, 0.4, 300.0, 0.0),
        ("glacier_deep_snow", 245.0, 243.0, 0.0, 0.0, 0.5, 400.0, 900.0),
    ]
    for regime, t_air, tsk, sr, pr, cz, sw, swe in glacier_cases:
        precip = pr * scale
        cols.append(_column(
            regime=regime, ivgtyp=15, isltyp=16, tsk=tsk,
            tmn=u(250.0, 270.0),
            tslb=(tsk + 1.0, tsk + 3.0, tsk + 5.0, tsk + 7.0),
            smois=(0.3, 0.3, 0.3, 0.3), snow=swe,
            snowh=(swe * 0.004 if swe else 0.0), lai=0.0, vegfra=0.0,
            shdmax=0.0, xlat=u(60, 80), coszen=cz, swdown=sw,
            glw=u(160, 300), t3d=t_air, qv3d=u(0.0003, 0.004),
            u=u(-10, 10), v=u(-10, 10), dz8w=u(30, 60),
            p8w1=u(70000, 90000), p8w2=0.0,
            rainbl=precip, sr=sr, rainnc=precip, snownc=precip * sr))

    # --- desert / barren -----------------------------------------------------
    for veg, soil in [(16, 1), (16, 2), (7, 1), (20, 3)]:
        t_air = u(305.0, 315.0)
        cols.append(_column(
            regime="desert", ivgtyp=veg, isltyp=soil, tsk=t_air + u(12, 25),
            tmn=u(295.0, 300.0),
            tslb=(t_air + 8.0, t_air + 2.0, t_air - 4.0, t_air - 8.0),
            smois=(u(0.01, 0.04), u(0.02, 0.05), 0.06, 0.08),
            lai=u(0.0, 0.3), vegfra=u(0, 10), shdmax=u(0, 15),
            xlat=u(20, 35), coszen=u(0.85, 0.99), swdown=u(900, 1050),
            glw=u(380, 440), t3d=t_air, qv3d=u(0.002, 0.006),
            u=u(-10, 10), v=u(-10, 10), dz8w=u(40, 90),
            p8w1=u(90000, 100000), p8w2=0.0))

    # --- saturated wetland, heavy rain ---------------------------------------
    for veg, soil in [(11, 12), (11, 9), (14, 11), (2, 12)]:
        t_air = u(295.0, 302.0)
        cols.append(_column(
            regime="wet_heavy_rain", ivgtyp=veg, isltyp=soil,
            tsk=t_air + 0.5, tmn=u(290.0, 296.0),
            tslb=(t_air, t_air - 1.0, t_air - 2.0, t_air - 3.0),
            smois=(0.47, 0.48, 0.49, 0.50), lai=u(2.0, 5.0),
            vegfra=u(60, 95), shdmax=u(70, 99), xlat=u(5, 35),
            coszen=u(0.2, 0.6), swdown=u(100, 400), glw=u(400, 450),
            t3d=t_air, qv3d=u(0.016, 0.021), u=u(-6, 6), v=u(-6, 6),
            dz8w=u(30, 60), p8w1=u(99000, 101500), p8w2=0.0,
            rainc=u(0.6, 1.2) * scale, rainnc=u(0.2, 0.5) * scale,
            rainsh=u(0.0, 0.1) * scale))

    # --- high terrain --------------------------------------------------------
    for veg, soil in [(7, 3), (10, 4), (16, 1), (6, 6), (19, 6)]:
        t_air = u(266.0, 285.0)
        cols.append(_column(
            regime="high_terrain", ivgtyp=veg, isltyp=soil,
            tsk=t_air + u(4, 12), tmn=u(270.0, 280.0),
            tslb=(t_air + 3.0, t_air + 1.0, t_air - 1.0, t_air - 2.0),
            smois=(u(0.08, 0.25),) * 2 + (0.2, 0.22),
            lai=u(0.2, 2.0), vegfra=u(5, 50), shdmax=u(10, 60),
            xlat=u(28, 45), coszen=u(0.8, 0.98), swdown=u(950, 1150),
            glw=u(200, 300), t3d=t_air, qv3d=u(0.0015, 0.005),
            u=u(-20, 20), v=u(-15, 15), dz8w=u(40, 90),
            p8w1=u(58000, 70000), p8w2=0.0))

    # --- urban (ISURBAN_TABLE = 13, sf_urban_physics = 0) --------------------
    for n in range(3):
        t_air = u(296.0, 306.0)
        cols.append(_column(
            regime="urban", ivgtyp=13, isltyp=(6, 4, 12)[n],
            tsk=t_air + u(2, 8), tmn=u(288.0, 295.0),
            tslb=(t_air + 1.0, t_air - 1.0, t_air - 3.0, t_air - 6.0),
            smois=(u(0.1, 0.3),) * 4, lai=u(0.0, 1.0), vegfra=u(5, 30),
            shdmax=u(10, 40), xlat=u(30, 45), coszen=(0.7, 0.0, 0.3)[n],
            swdown=(700.0, 0.0, 280.0)[n], glw=u(360, 420), t3d=t_air,
            qv3d=u(0.008, 0.014), u=u(-5, 5), v=u(-5, 5), dz8w=u(30, 60),
            p8w1=u(98000, 101000), p8w2=0.0,
            rainnc=(0.1 * scale if n == 2 else 0.0)))

    # --- southern hemisphere (PHENOLOGY's LAT < 0 branch) ---------------------
    for veg, soil in [(2, 9), (9, 3), (12, 6), (10, 4), (8, 6)]:
        t_air = u(288.0, 300.0)
        cols.append(_column(
            regime="south_hemisphere", ivgtyp=veg, isltyp=soil,
            tsk=t_air + u(-2, 6), tmn=u(285.0, 295.0),
            tslb=(t_air, t_air - 1.0, t_air - 2.0, t_air - 3.0),
            smois=(u(0.15, 0.35),) * 4, lai=u(0.5, 4.0), vegfra=u(20, 90),
            shdmax=u(30, 95), xlat=u(-45, -10), coszen=u(0.0, 0.9),
            swdown=u(0.0, 800.0), glw=u(320, 420), t3d=t_air,
            qv3d=u(0.006, 0.016), u=u(-6, 6), v=u(-6, 6), dz8w=u(30, 60),
            p8w1=u(96000, 101500), p8w2=0.0))

    # --- open water and sea ice ------------------------------------------------
    water_cases = [
        ("ocean", 2.0, 0.0, 17, 14), ("ocean", 2.0, 0.0, 17, 14),
        ("ocean_partial_ice", 2.0, 0.3, 17, 14),
        ("ocean_full_ice_flag", 2.0, 1.0, 17, 14),
        ("sea_ice_full", 1.0, 1.0, 17, 14),
        ("sea_ice_full", 1.0, 1.0, 15, 16),
        ("sea_ice_threshold", 1.0, 0.5, 17, 14),
        ("sea_ice_partial", 1.0, 0.8, 15, 16),
        ("lake_as_water", 2.0, 0.0, 17, 14),
    ]
    for regime, xland, xice, veg, soil in water_cases:
        t_air = u(270.0, 295.0)
        cols.append(_column(
            regime=regime, ivgtyp=veg, isltyp=soil, xland=xland, xice=xice,
            tsk=t_air + u(-2, 2), tmn=u(271.0, 290.0),
            tslb=(t_air,) * 4, smois=(0.5,) * 4, lai=0.0, vegfra=0.0,
            shdmax=0.0, xlat=u(40, 75), coszen=u(0.0, 0.6),
            swdown=u(0.0, 500.0), glw=u(220, 380), t3d=t_air,
            qv3d=u(0.001, 0.012), u=u(-10, 10), v=u(-10, 10),
            dz8w=u(30, 60), p8w1=u(98000, 102500), p8w2=0.0))

    # --- edge cases --------------------------------------------------------------
    t = 300.0
    cols.append(_column(regime="edge_calm_wind", u=0.0, v=0.0, tsk=t + 6,
                        t3d=t, swdown=800.0, coszen=0.85))
    cols.append(_column(regime="edge_calm_night", u=0.0, v=0.0, tsk=283.0,
                        t3d=288.0, swdown=0.0, coszen=0.0, glw=300.0,
                        qv3d=0.006))
    cols.append(_column(regime="edge_grazing_sun", coszen=0.002,
                        swdown=1.5, tsk=292.0, t3d=293.0))
    cols.append(_column(regime="edge_hail_graupel", t3d=285.0, tsk=290.0,
                        rainbl=0.8 * scale, sr=0.4,
                        rainc=0.3 * scale, rainnc=0.4 * scale,
                        snownc=0.05 * scale, graupnc=0.15 * scale,
                        hailnc=0.1 * scale, coszen=0.3, swdown=150.0))
    cols.append(_column(regime="edge_fog_residual", rainbl=0.02 * scale,
                        sr=0.0, rainnc=0.0, coszen=0.1, swdown=40.0,
                        tsk=287.0, t3d=287.5, qv3d=0.0105))
    cols.append(_column(regime="edge_water_soil_on_land", isltyp=14,
                        ivgtyp=10))
    cols.append(_column(regime="edge_wilting_soil", isltyp=4, ivgtyp=12,
                        smois=(0.05, 0.06, 0.07, 0.08), tsk=318.0,
                        t3d=306.0, qv3d=0.004))
    cols.append(_column(regime="edge_above_porosity", isltyp=3, ivgtyp=10,
                        smois=(0.6, 0.6, 0.6, 0.6)))
    cols.append(_column(regime="edge_tslb_at_threshold", isltyp=6,
                        tslb=(273.149, 273.15, 273.16, 274.0), tsk=272.0,
                        t3d=271.0, coszen=0.3, swdown=200.0, qv3d=0.003,
                        glw=260.0))
    cols.append(_column(regime="edge_zero_lai_full_shdmax", lai=0.0,
                        shdmax=100.0, vegfra=100.0))
    cols.append(_column(regime="edge_zero_shdmax", lai=3.0, shdmax=0.0,
                        vegfra=0.0))
    cols.append(_column(regime="edge_equator", xlat=0.0, ivgtyp=2,
                        isltyp=12, tsk=303.0, t3d=300.0, qv3d=0.019))
    cols.append(_column(regime="edge_organic_soil", isltyp=13, ivgtyp=11,
                        smois=(0.4, 0.42, 0.44, 0.45)))
    cols.append(_column(regime="edge_supersaturated_air", t3d=290.0,
                        qv3d=0.0145, tsk=289.0, coszen=0.0, swdown=0.0,
                        glw=360.0))

    # 128 columns: pad with seeded mid-latitude mixtures over the land classes
    land_veg = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 19]
    land_soil = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    while len(cols) < 128:
        t_air = u(272.0, 308.0)
        night = rng.uniform() < 0.35
        sm = u(0.08, 0.42)
        snowy = t_air < 276.0 and rng.uniform() < 0.6
        swe = u(5.0, 150.0) if snowy else 0.0
        precip = (u(0.0, 0.3) * scale) if rng.uniform() < 0.4 else 0.0
        sr = 1.0 if t_air < 273.15 else 0.0
        cols.append(_column(
            regime="mixed", ivgtyp=int(rng.choice(land_veg)),
            isltyp=int(rng.choice(land_soil)),
            tsk=t_air + (u(-6, 0) if night else u(0, 10)),
            tmn=u(275.0, 295.0),
            tslb=(t_air + u(-2, 2), t_air + u(-3, 3), t_air + u(-4, 4),
                  t_air + u(-6, 6)),
            smois=(sm, sm + u(0, 0.04), sm + u(0, 0.06), sm + u(0, 0.08)),
            snow=swe, snowh=(swe / u(150.0, 350.0) if swe else 0.0),
            lai=u(0.0, 6.0), vegfra=u(0, 100), shdmax=u(0, 100),
            xlat=u(-55, 70), coszen=(0.0 if night else u(0.05, 0.98)),
            swdown=(0.0 if night else u(20, 1000)), glw=u(220, 440),
            t3d=t_air, qv3d=u(0.001, 0.02), u=u(-12, 12), v=u(-12, 12),
            dz8w=u(20, 100), p8w1=u(80000, 102000), p8w2=0.0,
            rainbl=precip, sr=sr, rainnc=precip, snownc=precip * sr))

    for c in cols:
        # pressure at the first interface above the surface, from dz8w
        # and a hydrostatic scale: ~1 hPa per 8.5 m near the surface.
        c["p8w2"] = c["p8w1"] - c["dz8w"] * (c["p8w1"] / 8434.0) * 0.98
        liquid = c["rainc"] + c["rainnc"] + c["rainsh"]
        if c["rainbl"] == 0.0 and liquid > 0.0:
            c["rainbl"] = liquid
        c["xlong"] = c["xlong"] if c["xlong"] != -100.0 else u(-170, 170)
    return cols


def julian_and_year(start: datetime, elapsed: float):
    """WOOF's NoahmpSolarGeometry.julian / year for one elapsed time."""
    valid = start + timedelta(seconds=float(elapsed))
    hour = (valid.hour + valid.minute / 60.0 + valid.second / 3600.0
            + valid.microsecond / 3.6e9)
    return float(valid.timetuple().tm_yday - 1.0 + hour / 24.0), valid.year


def _record(name: str, kind: int, values) -> bytes:
    arr = np.ascontiguousarray(values, dtype="<f4" if kind == 1 else "<i4")
    head = name.ljust(16).encode("ascii")[:16] + struct.pack(
        "<ii", kind, arr.size)
    return head + arr.tobytes()


def forcing_at(c, index: int, step: int, nstep: int) -> dict:
    """One column's forcing at one step: the regime's base values carried
    through a smooth drift plus an intermittent precipitation gate, so the
    run exercises rain and snow starting and stopping, a sun that moves,
    and winds and temperatures that change under an evolving state."""
    phase = 2.0 * np.pi * step / max(nstep, 1)
    out = {}
    out["t3d"] = c["t3d"] + 0.6 * np.sin(phase + 0.1 * index)
    out["qv3d"] = c["qv3d"] * (1.0 + 0.03 * np.sin(phase + 0.2 * index))
    out["u"] = c["u"] * (1.0 + 0.15 * np.cos(phase))
    out["v"] = c["v"] * (1.0 - 0.10 * np.cos(phase + 0.3))
    sun = max(0.0, 1.0 - 0.25 * step / max(nstep, 1))
    out["coszen"] = c["coszen"] * sun
    out["swdown"] = c["swdown"] * sun
    out["glw"] = c["glw"] + 4.0 * np.sin(phase)
    out["dz8w"] = c["dz8w"]
    out["p8w1"] = c["p8w1"] + 20.0 * np.sin(phase)
    out["p8w2"] = c["p8w2"] + 20.0 * np.sin(phase)
    out["sr"] = c["sr"]
    gate = (index + step) % 7
    scale = 0.0 if gate in (0, 1) else (1.5 if gate == 3 else 1.0)
    for name in PRECIP_FIELDS:
        out[name] = c[name] * scale
    return out


def write(path: str, config: str):
    cfg = CONFIGS[config]
    cols = build_columns(cfg["seed"], cfg["dt"])
    n = len(cols)
    nstep = cfg["nstep"]
    jul = []
    yrs = []
    for k in range(nstep):
        j, y = julian_and_year(cfg["start"], k * cfg["dt"])
        jul.append(np.float32(j))
        yrs.append(y)
    out = bytearray()
    out += _record("nx", 2, [n])
    out += _record("nstep", 2, [nstep])
    out += _record("dt", 1, [cfg["dt"]])
    out += _record("dx", 1, [cfg["dx"]])
    out += _record("julian", 1, jul)
    out += _record("yr", 2, yrs)
    for name in INT_FIELDS:
        out += _record(name, 2, [c[name] for c in cols])
    for name in LAYER_FIELDS:
        # (NSOIL, n), layer-major: Fortran's X(NX, NSOIL) column-major order
        out += _record(name, 1, np.array([c[name] for c in cols],
                                         dtype=np.float32).T)
    for name in STATIC_FIELDS:
        out += _record(name, 1, [c[name] for c in cols])
    per_step = [[forcing_at(c, i, k, nstep) for i, c in enumerate(cols)]
                for k in range(nstep)]
    for name in FORCING_FIELDS:
        # (nstep, n), step-major: Fortran's F(NX, NSTEP)
        out += _record(name, 1, np.array(
            [[col[name] for col in row] for row in per_step],
            dtype=np.float32))
    with open(path, "wb") as fh:
        fh.write(bytes(out))
    with open(path + ".regimes.txt", "w") as fh:
        for i, c in enumerate(cols):
            fh.write(f"{i + 1} {c['regime']} veg={c['ivgtyp']} "
                     f"soil={c['isltyp']}\n")
    return cols


def read_records(path: str) -> dict:
    """Read every record of an input or output file into numpy arrays.

    Output files repeat names once per snapshot; those come back as a list
    under the same key, in file order."""
    data = open(path, "rb").read()
    pos = 0
    out: dict = {}
    while pos < len(data):
        name = data[pos:pos + 16].decode("ascii").strip()
        kind, count = struct.unpack("<ii", data[pos + 16:pos + 24])
        pos += 24
        dtype = "<f4" if kind == 1 else "<i4"
        values = np.frombuffer(data, dtype=dtype, count=count,
                               offset=pos).copy()
        pos += 4 * count
        out.setdefault(name, []).append(values)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--config", choices=sorted(CONFIGS), required=True)
    args = ap.parse_args()
    cols = write(args.out, args.config)
    print(f"{args.out}: {len(cols)} columns, config {args.config}")


if __name__ == "__main__":
    main()
