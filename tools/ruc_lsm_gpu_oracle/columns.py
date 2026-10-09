"""Column set and per-step forcing for the RUC LSM GPU column oracle.

Every column is a deterministic, physically plausible surface state with a
regime label.  The set spans warm convective land, stable night land, snow
packs (thick two-layer, thin one-layer, melting, fresh snowfall on bare
ground), frozen soil, land ice, sea ice (full and fractional), open water,
lakes, high terrain, dry desert, saturated wetland with drainage, urban and
cropland, plus edge cases (skin exactly at freezing, a trace of snow, zero and
full vegetation).  Nothing here imports the engine: the same arrays feed the
WRF Fortran driver and the WOOF GPU runner.

The forcing for step ``k`` changes with ``k`` so a free-running comparison
exercises a sequence of different calls, not one call repeated.
"""

from __future__ import annotations

import numpy as np

F = np.float32

# MODIFIED_IGBP_MODIS_NOAH (MODI-RUC) categories used below.
ENF, EBF, DBF, MIXF, CSHRUB, OSHRUB, WSAV, SAV, GRASS, WETL, CROP, URBAN, \
    CROPMOS, SNOWICE, BARREN, WATER, WTUNDRA, MTUNDRA, BTUNDRA, LAKE = (
        1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
# STAS-RUC soil categories.
SAND, LSAND, SLOAM, SILOAM, SILT, LOAM, SCLOAM, SICLOAM, CLOAM, SCLAY, \
    SICLAY, CLAY, ORGANIC, WATSOIL, BEDROCK, LANDICE, PLAYA, LAVA, WSAND = range(1, 20)

NLCAT = 21
NSCAT = 19

#: The per-column 2-D state the oracle sets explicitly (WOOF field names).
STATE_2D = (
    "snow", "snowh", "snowc", "canwat", "snoalb", "albedo", "emiss", "lai",
    "mavail", "sfcexc", "z0", "znt", "tsk", "hfx", "qfx", "lh", "sfcevp",
    "sfcrunoff", "udrunoff", "acrunoff", "grdflx", "acsnow", "acsnom", "qvg",
    "qcg", "dew", "qsfc", "qsg", "chklowq", "soilt1", "tsnav", "smstav",
    "smstot", "rhosnf", "precipfr", "snowfallac",
    # seam state
    "albbck", "tsk_save", "t2", "th2", "q2",
)
#: Static per-column forcing (does not change with the step).
STATIC_2D = ("xland", "xice", "tmn", "shdmin", "shdmax", "vegfra", "lakemask")
#: Forcing that changes every step.
STEP_2D = (
    "rainbl", "sr", "glw", "gsw", "chs", "flhc", "flqc", "cpm", "cqs2",
    "chs2", "qgh", "psfc", "surface_rainncv", "surface_snowncv",
    "surface_graupelncv",
    "tsk_sea", "flhc_sea", "flqc_sea", "cpm_sea", "cqs2_sea", "chs2_sea",
    "chs_sea", "qsfc_sea", "qgh_sea", "hfx_sea", "qfx_sea", "lh_sea",
)
#: Lowest model level (WOOF atmosphere keys), changes every step.
ATMOS = ("dz", "pressure", "temperature", "qv", "qc", "rho")
PROFILES = ("smois", "sh2o", "tslb", "smfr3d", "keepfr3dflag")


def _qsat(p, t):
    """A smooth saturation mixing ratio, float64; only for plausible inputs."""
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / np.maximum(p - es, 1.0)


class _Builder:
    def __init__(self, nzs):
        self.nzs = nzs
        self.cols = []

    def add(self, regime, **kw):
        self.cols.append((regime, kw))


def _regimes(nzs):
    b = _Builder(nzs)
    # --- warm convective land: afternoon, strong sun, various veg/soil ---
    for n, (veg, soil, sm) in enumerate([
            (CROP, SILOAM, .32), (GRASS, LOAM, .28), (DBF, CLOAM, .30),
            (ENF, SLOAM, .22), (SAV, SCLOAM, .18), (WSAV, LSAND, .14),
            (MIXF, SICLOAM, .35), (CROPMOS, SILT, .30), (EBF, CLAY, .40),
            (OSHRUB, SAND, .08), (CSHRUB, SCLAY, .20), (WETL, ORGANIC, .45),
            (URBAN, LOAM, .25), (GRASS, SICLAY, .38), (CROP, LOAM, .12),
            (DBF, SILOAM, .26)]):
        b.add("convective", veg=veg, soil=soil, sm=sm, tsk=300.0 + 0.6 * n,
              tair=297.0 + 0.5 * n, rh=0.55, gsw=650.0 + 10 * n, glw=380.0,
              rain=(n % 3 == 0), vegfra=20.0 + 4.5 * n, p=97000.0 - 150 * n)
    # --- stable night land: no sun, cold air over warmer ground ----------
    for n, (veg, soil, sm) in enumerate([
            (GRASS, LOAM, .25), (CROP, SILOAM, .30), (ENF, SLOAM, .20),
            (SAV, SAND, .06), (MIXF, CLOAM, .33), (WSAV, LSAND, .12),
            (BARREN, PLAYA, .05), (URBAN, SICLOAM, .22), (WTUNDRA, SILT, .30),
            (DBF, CLAY, .38), (OSHRUB, SCLOAM, .15), (CROPMOS, LOAM, .27)]):
        b.add("stable_night", veg=veg, soil=soil, sm=sm, tsk=280.0 + 0.7 * n,
              tair=283.0 + 0.5 * n, rh=0.85, gsw=0.0, glw=280.0 + 3 * n,
              vegfra=10.0 + 6 * n, p=99500.0 - 100 * n)
    # --- snow: thick two-layer packs, cold ------------------------------
    for n, (veg, soil) in enumerate([(ENF, LOAM), (GRASS, SILOAM),
                                     (MTUNDRA, SILT), (CROP, CLOAM),
                                     (MIXF, SLOAM), (BTUNDRA, BEDROCK)]):
        b.add("snow_thick", veg=veg, soil=soil, sm=.30, tsk=258.0 + 2 * n,
              tair=257.0 + 2 * n, rh=0.8, gsw=150.0 + 40 * n, glw=220.0,
              snow=60.0 + 25 * n, snowdens=200.0 + 20 * n, frozen=True,
              vegfra=30.0, p=95000.0)
    # --- snow: thin one-layer packs (above snth, below deltsn+snth) -----
    for n, (veg, soil) in enumerate([(GRASS, LOAM), (CROP, SILOAM),
                                     (OSHRUB, SLOAM), (WTUNDRA, CLOAM)]):
        b.add("snow_thin", veg=veg, soil=soil, sm=.28, tsk=266.0 + n,
              tair=265.0 + n, rh=0.8, gsw=200.0, glw=240.0,
              snow=4.0 + 2 * n, snowdens=150.0 + 20 * n, frozen=True,
              vegfra=25.0, p=96000.0)
    # --- snow: trace packs below snth (the ilnb-sensitive regime) -------
    for n, (veg, soil) in enumerate([(GRASS, LOAM), (CROP, SILOAM),
                                     (SAV, SCLOAM)]):
        b.add("snow_trace", veg=veg, soil=soil, sm=.25, tsk=268.0 + n,
              tair=267.5 + n, rh=0.85, gsw=120.0, glw=250.0,
              snow=0.6 + 0.4 * n, snowdens=120.0, frozen=True,
              vegfra=30.0, p=97000.0)
    # --- snow: melting packs on warm ground -----------------------------
    for n, (veg, soil) in enumerate([(GRASS, SILOAM), (CROP, LOAM),
                                     (DBF, CLOAM), (MIXF, SLOAM)]):
        b.add("snow_melt", veg=veg, soil=soil, sm=.32, tsk=273.0 + 0.3 * n,
              tair=279.0 + 1.5 * n, rh=0.9, gsw=450.0 + 50 * n, glw=320.0,
              snow=8.0 + 10 * n, snowdens=300.0, vegfra=40.0, p=97500.0,
              rain=(n % 2 == 0))
    # --- fresh snowfall onto bare cold ground (new-snow density, tanh) ---
    for n, (veg, soil, tair) in enumerate([(GRASS, LOAM, 268.0),
                                           (CROP, SILOAM, 271.0),
                                           (ENF, SLOAM, 274.5),
                                           (SAV, SCLOAM, 262.0),
                                           (MIXF, CLOAM, 270.0)]):
        b.add("snowfall_bare", veg=veg, soil=soil, sm=.27, tsk=tair + 0.5,
              tair=tair, rh=0.95, gsw=60.0, glw=290.0, snowfall=True,
              frozen=tair < 272.0, vegfra=35.0, p=96500.0)
    # --- frozen soil without snow ----------------------------------------
    for n, (veg, soil) in enumerate([(GRASS, SILOAM), (BTUNDRA, LOAM),
                                     (CROP, CLAY)]):
        b.add("frozen_soil", veg=veg, soil=soil, sm=.30, tsk=263.0 + 2 * n,
              tair=262.0 + 2 * n, rh=0.7, gsw=100.0, glw=230.0, frozen=True,
              vegfra=15.0, p=98000.0)
    # --- land ice (glacier, isice vegetation on land) --------------------
    for n in range(2):
        b.add("land_ice", veg=SNOWICE, soil=LANDICE, sm=.95, tsk=250.0 + 5 * n,
              tair=249.0 + 5 * n, rh=0.8, gsw=250.0 * n, glw=200.0,
              snow=200.0, snowdens=350.0, frozen=True, vegfra=0.0,
              p=80000.0)
    # --- sea ice: full cover, with and without snow ----------------------
    for n in range(4):
        b.add("sea_ice_full", veg=SNOWICE, soil=LANDICE, sm=1.0,
              tsk=250.0 + 3 * n, tair=249.0 + 3 * n, rh=0.85,
              gsw=80.0 * n, glw=200.0, xice=1.0,
              snow=(0.0 if n % 2 else 30.0), snowdens=300.0, frozen=True,
              vegfra=0.0, p=100500.0)
    # --- sea ice: fractional cover ---------------------------------------
    for n, frac in enumerate([0.95, 0.80, 0.62, 0.51, 0.30, 0.05]):
        b.add("sea_ice_fractional", veg=SNOWICE, soil=LANDICE, sm=1.0,
              tsk=255.0 + 2 * n, tair=254.0 + 2 * n, rh=0.85,
              gsw=50.0 * n, glw=210.0, xice=frac,
              snow=(10.0 if n % 2 == 0 else 0.0), snowdens=280.0,
              frozen=True, vegfra=0.0, p=100800.0)
    # --- open water -------------------------------------------------------
    for n in range(6):
        b.add("ocean", veg=WATER, soil=WATSOIL, sm=1.0, tsk=275.0 + 5 * n,
              tair=274.0 + 5 * n, rh=0.8, gsw=300.0 * (n % 2), glw=330.0,
              water=True, vegfra=0.0, p=101000.0, rain=(n == 3))
    # --- lakes ------------------------------------------------------------
    for n in range(3):
        b.add("lake", veg=LAKE, soil=WATSOIL, sm=1.0, tsk=285.0 + 4 * n,
              tair=284.0 + 4 * n, rh=0.75, gsw=200.0, glw=320.0,
              water=True, lake=True, vegfra=0.0, p=99000.0)
    # --- high terrain: low pressure, cold, rocky/barren -------------------
    for n, (veg, soil, snow) in enumerate([
            (BARREN, BEDROCK, 0.0), (BTUNDRA, LAVA, 0.0), (ENF, SLOAM, 90.0),
            (GRASS, LOAM, 0.0), (OSHRUB, SCLOAM, 25.0), (MTUNDRA, SILT, 0.0),
            (CSHRUB, SAND, 0.0), (BARREN, WSAND, 5.0)]):
        b.add("high_terrain", veg=veg, soil=soil, sm=.18, tsk=268.0 + 4 * n,
              tair=266.0 + 3.5 * n, rh=0.5, gsw=700.0 - 40 * n, glw=200.0,
              snow=snow, snowdens=250.0, frozen=snow > 0, vegfra=10.0,
              p=62000.0 + 1500 * n)
    # --- hot desert: dry soil near wilting --------------------------------
    for n, (veg, soil) in enumerate([(BARREN, SAND), (OSHRUB, LSAND),
                                     (BARREN, PLAYA), (CSHRUB, SLOAM),
                                     (BARREN, WSAND), (SAV, SCLAY)]):
        b.add("desert", veg=veg, soil=soil, sm=.03 + .01 * n, tsk=318.0 + n,
              tair=309.0 + n, rh=0.12, gsw=850.0, glw=400.0, vegfra=3.0,
              p=92000.0)
    # --- saturated wetland with heavy rain and drainage ------------------
    for n, (veg, soil) in enumerate([(WETL, ORGANIC), (CROP, SICLAY),
                                     (GRASS, CLAY), (WETL, SICLOAM),
                                     (DBF, CLOAM), (CROPMOS, SILT)]):
        b.add("wet_drainage", veg=veg, soil=soil, sm=.47, tsk=293.0 + n,
              tair=292.0 + n, rh=0.98, gsw=150.0, glw=390.0, rain=True,
              heavy=True, vegfra=70.0, p=99800.0, saturated=True)
    # --- edge cases -------------------------------------------------------
    b.add("edge_freezing_skin", veg=GRASS, soil=LOAM, sm=.30, tsk=273.15,
          tair=273.15, rh=0.95, gsw=100.0, glw=300.0, vegfra=50.0,
          p=98000.0)
    b.add("edge_trace_snow", veg=CROP, soil=SILOAM, sm=.30, tsk=272.0,
          tair=271.0, rh=0.9, gsw=50.0, glw=280.0, snow=1.0e-3,
          snowdens=100.0, vegfra=20.0, p=98000.0)
    b.add("edge_zero_veg", veg=GRASS, soil=LOAM, sm=.25, tsk=305.0,
          tair=300.0, rh=0.4, gsw=700.0, glw=380.0, vegfra=0.0, p=97000.0)
    b.add("edge_full_veg", veg=EBF, soil=CLAY, sm=.42, tsk=302.0,
          tair=300.0, rh=0.8, gsw=700.0, glw=410.0, vegfra=100.0,
          p=100000.0, rain=True)
    b.add("edge_wilting", veg=GRASS, soil=CLAY, sm=.14, tsk=310.0,
          tair=304.0, rh=0.3, gsw=800.0, glw=390.0, vegfra=60.0, p=96000.0)
    b.add("edge_canopy_wet", veg=DBF, soil=LOAM, sm=.30, tsk=290.0,
          tair=289.0, rh=0.99, gsw=100.0, glw=360.0, vegfra=90.0,
          p=99000.0, canwat=0.4, rain=True)
    b.add("edge_dew", veg=GRASS, soil=SILOAM, sm=.30, tsk=284.0,
          tair=287.0, rh=1.0, gsw=0.0, glw=300.0, vegfra=80.0, p=99000.0)
    b.add("edge_snow_warm_air_rain", veg=GRASS, soil=LOAM, sm=.30,
          tsk=272.5, tair=276.0, rh=1.0, gsw=20.0, glw=330.0, snow=20.0,
          snowdens=350.0, rain=True, heavy=True, vegfra=30.0, p=99000.0)
    return b.cols


def build_case(nzs: int = 9, nsteps: int = 8, *, dt: float = 20.0,
               fractional_seaice: int = 1, mosaic: bool = False,
               lakemodel: int = 0, rdlai2d: bool = False,
               round_xice: bool = True, seed: int = 20261007) -> dict:
    """The whole case: initial raw inputs, explicit state, per-step forcing."""
    from_regimes = _regimes(nzs)
    ncol = len(from_regimes)
    rng = np.random.default_rng(seed)
    if nzs == 9:
        depth = np.array([0.0, .01, .04, .10, .30, .60, 1.0, 1.6, 3.0])
    else:
        depth = np.array([0.0, .05, .20, .40, 1.6, 3.0])
    regime = []
    s = {name: np.zeros(ncol) for name in STATE_2D + STATIC_2D}
    prof = {name: np.zeros((nzs, ncol)) for name in PROFILES}
    ivgtyp = np.zeros(ncol, np.int32)
    isltyp = np.zeros(ncol, np.int32)
    landusef = np.zeros((NLCAT, ncol))
    soilctop = np.zeros((NSCAT, ncol))
    meta = []
    for i, (name, kw) in enumerate(from_regimes):
        regime.append(name)
        meta.append(kw)
        veg, soil = kw["veg"], kw["soil"]
        ivgtyp[i] = veg
        isltyp[i] = soil
        water = kw.get("water", False)
        xice = kw.get("xice", 0.0)
        if fractional_seaice == 0 and 0.0 < xice < 1.0 and round_xice:
            # fractional_seaice=0: real.exe hands RUC a 0/1 ice mask.
            xice = 1.0 if xice >= 0.5 else 0.0
            if xice == 0.0:
                water = True
                veg, soil = WATER, WATSOIL
                ivgtyp[i], isltyp[i] = veg, soil
        s["xice"][i] = xice
        s["xland"][i] = 2.0 if (water and xice == 0.0) else 1.0
        s["lakemask"][i] = 1.0 if kw.get("lake") else 0.0
        tsk = kw["tsk"]
        s["tsk"][i] = tsk
        s["tsk_save"][i] = tsk - (0.0 if xice in (0.0, 1.0) else 1.5)
        s["tmn"][i] = 271.4 if (water or xice > 0) else tsk - 12.0 + 4 * np.sin(i)
        if kw.get("frozen") and not water:
            s["tmn"][i] = min(s["tmn"][i], 268.0)
        vegfra = kw.get("vegfra", 30.0)
        s["vegfra"][i] = vegfra
        s["shdmin"][i] = max(0.0, vegfra - 15.0)
        s["shdmax"][i] = min(100.0, vegfra + 20.0)
        # soil temperature profile: relaxes from skin to tmn with depth
        frac = np.clip(depth / 1.6, 0, 1)
        prof["tslb"][:, i] = tsk + (s["tmn"][i] - tsk) * frac
        if kw.get("frozen") and not water:
            prof["tslb"][:, i] = np.minimum(prof["tslb"][:, i],
                                            271.0 + 1.5 * frac)
        sm = kw["sm"]
        if water or xice > 0:
            prof["smois"][:, i] = 1.0
        elif kw.get("saturated"):
            prof["smois"][:, i] = sm - 0.01 * frac
        else:
            prof["smois"][:, i] = sm + 0.08 * frac + 0.01 * rng.uniform(
                -1, 1, nzs)
        prof["smois"][:, i] = np.clip(prof["smois"][:, i], 0.02, 1.0)
        prof["sh2o"][:, i] = prof["smois"][:, i]
        snow = kw.get("snow", 0.0)
        s["snow"][i] = snow
        dens = kw.get("snowdens", 250.0)
        s["snowh"][i] = snow / dens if snow > 0 else 0.0
        s["snowc"][i] = (min(1.0, s["snowh"][i] / 0.1) if snow > 0 else 0.0)
        s["canwat"][i] = kw.get("canwat", 0.0 if snow else 0.05 * (i % 4))
        s["snoalb"][i] = 0.55 + 0.05 * (i % 5)
        s["albbck"][i] = 0.12 + 0.01 * (i % 8)
        s["albedo"][i] = (0.6 if snow > 0 else s["albbck"][i])
        if water:
            s["albedo"][i] = 0.08
        if xice > 0:
            s["albedo"][i] = xice * 0.6 + (1 - xice) * 0.08
        s["emiss"][i] = 0.95 + 0.004 * (i % 5)
        if xice > 0:
            s["emiss"][i] = xice * 0.98 + (1 - xice) * 0.98 - 0.003
        s["lai"][i] = 0.5 + 0.1 * (i % 30)
        s["mavail"][i] = 1.0 if water else 0.3 + 0.01 * (i % 50)
        s["sfcexc"][i] = 0.0
        s["z0"][i] = 0.1
        s["znt"][i] = 0.0001 if water else 0.05 + 0.01 * (i % 10)
        s["hfx"][i] = 30.0 + 3 * (i % 20) - 40.0 * (kw["gsw"] == 0)
        s["qfx"][i] = 2.0e-5 + 1.0e-6 * (i % 20)
        s["lh"][i] = s["qfx"][i] * 2.5e6
        s["sfcevp"][i] = 0.5 + 0.01 * i
        s["sfcrunoff"][i] = 0.1 * (i % 3)
        s["udrunoff"][i] = 0.2 * (i % 2)
        s["acrunoff"][i] = 0.3 * (i % 4)
        s["acsnow"][i] = 0.0
        s["acsnom"][i] = 0.0
        s["qvg"][i] = 0.0
        s["qcg"][i] = 0.0
        s["dew"][i] = 0.0
        s["qsfc"][i] = 0.0
        s["qsg"][i] = 0.0
        s["chklowq"][i] = 1.0
        s["soilt1"][i] = 0.0
        s["tsnav"][i] = 0.0
        s["smstav"][i] = 0.0
        s["smstot"][i] = 0.0
        s["rhosnf"][i] = 0.0
        s["precipfr"][i] = 0.0
        s["snowfallac"][i] = 0.0
        s["t2"][i] = kw["tair"]
        s["th2"][i] = kw["tair"]
        s["q2"][i] = 0.005
        # mosaic fractions: dominant category plus up to two neighbours
        if mosaic and not water and xice == 0 and veg != SNOWICE:
            others = [c for c in (GRASS, CROP, MIXF, OSHRUB, URBAN)
                      if c != veg][: 1 + i % 2]
            dom = 0.55 + 0.05 * (i % 5)
            landusef[veg - 1, i] = dom
            for c in others:
                landusef[c - 1, i] += (1 - dom) / len(others)
            sothers = [c for c in (LOAM, SAND, CLAY) if c != soil][: 1 + i % 2]
            sdom = 0.6 + 0.05 * (i % 4)
            soilctop[soil - 1, i] = sdom
            for c in sothers:
                soilctop[c - 1, i] += (1 - sdom) / len(sothers)
        else:
            landusef[veg - 1, i] = 1.0
            soilctop[soil - 1, i] = 1.0

    steps = []
    for k in range(1, nsteps + 1):
        f = {name: np.zeros(ncol) for name in STEP_2D}
        a = {name: np.zeros(ncol) for name in ATMOS}
        for i, kw in enumerate(meta):
            wave = np.sin(0.7 * k + 0.37 * i)
            tair = kw["tair"] + 0.15 * wave
            p = kw.get("p", 98000.0) - 3.0 * k
            q = min(kw["rh"] * _qsat(p, tair), 0.03)
            a["temperature"][i] = tair
            a["qv"][i] = q
            a["qc"][i] = 1.0e-4 if (kw.get("rain") and k % 2 == 0) else 0.0
            a["pressure"][i] = p
            a["dz"][i] = 24.0 + (i % 7) * 4.0
            a["rho"][i] = p / (287.04 * tair * (1 + 0.61 * q))
            f["psfc"][i] = p + a["rho"][i] * 9.81 * a["dz"][i] * 0.5
            f["gsw"][i] = max(0.0, kw["gsw"] * (1 + 0.05 * wave))
            f["glw"][i] = kw["glw"] + 5.0 * wave
            chs = 0.004 + 0.0004 * (i % 25) + 0.001 * wave
            if kw["gsw"] == 0:
                chs *= 0.3
            rho = a["rho"][i]
            cpm = 1004.5 * (1 + 0.8 * q)
            f["chs"][i] = chs
            f["cpm"][i] = cpm
            f["flhc"][i] = cpm * rho * chs
            f["flqc"][i] = rho * chs * s["mavail"][i]
            f["chs2"][i] = 0.6 * chs
            f["cqs2"][i] = 0.6 * chs
            f["qgh"][i] = _qsat(p, s["tsk"][i])
            # precipitation, mm per step
            rain = 0.0
            if kw.get("rain"):
                rain = (2.5 if kw.get("heavy") else 0.4) * (1 + 0.5 * wave)
                if k % 3 == 0:
                    rain = 0.0
            snowfall = 0.0
            if kw.get("snowfall"):
                snowfall = 0.6 * (1 + 0.3 * wave)
            total = rain + snowfall
            f["rainbl"][i] = total
            nonc = 0.7 * total
            f["surface_rainncv"][i] = nonc
            frz = snowfall / total if total > 0 else 0.0
            if kw.get("snowfall") and tair > 273.0:
                frz = 0.5
            f["sr"][i] = frz
            f["surface_snowncv"][i] = nonc * frz * 0.8
            f["surface_graupelncv"][i] = nonc * frz * 0.1
            # open-water components for the fractional sea-ice reblend
            f["tsk_sea"][i] = 271.4
            f["flhc_sea"][i] = f["flhc"][i] * 1.3
            f["flqc_sea"][i] = f["flqc"][i] * 1.2
            f["cpm_sea"][i] = cpm
            f["cqs2_sea"][i] = f["cqs2"][i] * 1.1
            f["chs2_sea"][i] = f["chs2"][i] * 1.1
            f["chs_sea"][i] = chs * 1.2
            f["qsfc_sea"][i] = _qsat(p, 271.4)
            f["qgh_sea"][i] = _qsat(p, 271.4)
            f["hfx_sea"][i] = 40.0 + wave
            f["qfx_sea"][i] = 3.0e-5
            f["lh_sea"][i] = 75.0
        steps.append({"forcing": {n: v.astype(F) for n, v in f.items()},
                      "atmos": {n: v.astype(F) for n, v in a.items()}})

    return {
        "ncol": ncol, "nzs": nzs, "nsteps": nsteps, "dt": dt,
        "fractional_seaice": fractional_seaice, "mosaic": int(bool(mosaic)),
        "lakemodel": lakemodel, "rdlai2d": int(bool(rdlai2d)),
        "regime": regime,
        "state": {n: v.astype(F) for n, v in s.items()},
        "profiles": {n: v.astype(F) for n, v in prof.items()},
        "ivgtyp": ivgtyp, "isltyp": isltyp,
        "landusef": landusef.astype(F), "soilctop": soilctop.astype(F),
        "steps": steps,
    }


if __name__ == "__main__":
    case = build_case()
    from collections import Counter
    print(case["ncol"], Counter(case["regime"]))
