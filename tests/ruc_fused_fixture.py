"""RUC per-call oracle grid for the fused-driver tests.

Builds a RUC physics driver on a heterogeneous grid (warm land, cold snow
band with thick and thin packs, water, lakes, full and fractional sea ice,
every land vegetation and soil category) and drives a RUC step function for
K calls with deterministic forcing that changes per call.  The mixed grid
reaches every sfctmp arm (mosaic land and sea ice, snow land and sea ice,
melt-out, the urban cap, bare land and bare sea ice).
"""

from __future__ import annotations

import numpy as np
import cupy as cp

from gpuwm.config import RunConfig
from gpuwm.core.physics import _prepare_atmosphere, initialize_physics
from gpuwm.core.surface_forcing import SurfacePrecipitationForcing

WATER = 17
ICE = 15
WATER_SOIL = 14
LAND_VEG = [c for c in range(1, 22) if c not in (WATER, ICE)]
LAND_SOIL = [c for c in range(1, 20) if c != WATER_SOIL]


def build(nx, ny, nz, nzs, scenario, seed):
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced

    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0, ztop=16000.0,
                    dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
                    mp_physics=8, sf_sfclay_physics=1, sf_surface_physics=3,
                    num_soil_layers=nzs, bl_pbl_physics=1, bldt=0.0,
                    ra_physics=0, radt_minutes=12.0)

    def theta(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 300.0,
                        np.where(z < 1700.0, 300.0 + 0.030 * (z - 1500.0),
                                 306.0 + 0.0045 * (z - 1700.0)))

    def qvapor(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 0.0110,
                        np.maximum(0.0110 - 5.0e-6 * (z - 1500.0), 1.0e-5))

    coord = make_vertical_coord(cfg.nz, stretch=2.0)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, qvapor)
    state.u[...] = cp.float32(5.0)
    state.v[...] = cp.float32(1.0)

    rng = np.random.default_rng(seed)
    shape = (ny, nx)
    rows = np.arange(ny)[:, None] * np.ones((1, nx))
    cols = np.ones((ny, 1)) * np.arange(nx)[None, :]
    landmask = np.ones(shape)
    xice = np.zeros(shape)
    tsk = np.full(shape, 303.0) + rng.uniform(-4.0, 4.0, shape)
    snow = np.zeros(shape)
    snowh = np.zeros(shape)
    veg = rng.choice(LAND_VEG, size=shape)
    soil = rng.choice(LAND_SOIL, size=shape)
    cold = np.zeros(shape, bool)
    if scenario == "mixed":
        water = cols >= int(0.85 * nx)
        landmask[water] = 0.0
        ice = (rows < int(0.08 * ny)) & ~water
        xice[ice] = 1.0
        frac = ice & (cols % 3 == 0)
        xice[frac] = rng.uniform(0.55, 0.95, shape)[frac]
        veg[ice] = ICE
        cold = (rows >= int(0.08 * ny)) & (rows < int(0.3 * ny)) & ~water
        tsk[cold | ice] = 262.0 + rng.uniform(0.0, 8.0, shape)[cold | ice]
        snow_band = cold & (rng.uniform(size=shape) < 0.8)
        snow[snow_band] = rng.uniform(1.0, 80.0, shape)[snow_band]
        density = rng.uniform(90.0, 350.0, shape)
        snowh[snow_band] = snow[snow_band] / density[snow_band]
        thin = snow_band & (rng.uniform(size=shape) < 0.15)
        snow[thin] = rng.uniform(0.05, 0.4, shape)[thin]
        snowh[thin] = snow[thin] / density[thin]
        snowh[ice & (cols % 2 == 0)] = 0.05
        snow[ice & (cols % 2 == 0)] = 15.0
        # A few thin packs on WARM land, so the melt-out reset (snhei
        # reaching exactly zero inside one call) is exercised.
        warm_snow = (~cold & ~ice & ~water
                     & (rng.uniform(size=shape) < 0.04))
        snow[warm_snow] = rng.uniform(0.02, 2.0, shape)[warm_snow]
        snowh[warm_snow] = snow[warm_snow] / density[warm_snow]
        veg[water] = WATER
        soil[water] = WATER_SOIL
        tsk[water] = 290.0 + rng.uniform(-2.0, 2.0, shape)[water]
    elif scenario == "warm":
        water = cols >= int(0.85 * nx)
        landmask[water] = 0.0
        veg[water] = WATER
        soil[water] = WATER_SOIL
        tsk[water] = 294.0
    else:
        raise SystemExit(f"unknown scenario {scenario}")

    spread = np.where(cold, 7.0, 8.0)
    soil_t = np.stack([tsk - spread * f for f in np.linspace(0.0, 1.0, nzs)])
    soil_m = np.stack([rng.uniform(0.12, 0.40, shape) for _ in range(nzs)])
    soil_m[:, landmask == 0.0] = 1.0
    driver = initialize_physics(
        state, cfg, landmask=landmask, tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m, ivgtyp=veg, isltyp=soil,
        vegfra=rng.uniform(5.0, 95.0, shape), tmn=np.where(cold, 270.0, 288.0),
        swdown=700.0, glw=340.0, pblh=500.0, xice=xice, snow=snow,
        snow_depth=snowh)
    driver.set_forcing(gsw=0.0)
    f = driver.fields
    if scenario == "mixed":
        lake = (rows >= int(0.5 * ny)) & (rows < int(0.55 * ny)) & (
            cols < int(0.2 * nx))
        f["lakemask"][...] = cp.asarray(np.where(lake, 1.0, 0.0),
                                        dtype=f["lakemask"].dtype)
    atmosphere = _prepare_atmosphere(state)
    if scenario == "mixed":
        cold_d = cp.asarray(cold | (xice > 0))
        atmosphere["temperature"][0][...] = cp.where(
            cold_d, cp.float32(266.0), atmosphere["temperature"][0])
        atmosphere["qv"][0][...] = cp.where(
            cold_d, cp.float32(0.0021), atmosphere["qv"][0])
    for name in ("tsk_save", "tsk_sea"):
        if name in f:
            f[name][...] = f["tsk"]
    return state, cfg, driver, atmosphere, cold


def forcing(driver, k, seed, shape, cold):
    rng = np.random.default_rng(seed * 1000 + k)
    f = driver.fields
    wet = rng.uniform(size=shape) < 0.5
    rainbl = np.where(wet, rng.uniform(0.0, 1.5, shape), 0.0)
    nonc = rainbl * rng.uniform(0.0, 1.0, shape)
    frz = np.where(cold, rng.uniform(0.0, 1.0, shape), 0.0)
    snowncv = nonc * frz * rng.uniform(0.3, 1.0, shape)
    graup = (nonc * frz - snowncv) * rng.uniform(0.0, 1.0, shape)

    def put(name, value):
        f[name][...] = cp.asarray(value, dtype=f[name].dtype)

    put("rainbl", rainbl)
    put("surface_rainncv", nonc)
    put("surface_snowncv", snowncv)
    put("surface_graupelncv", graup)
    put("sr", frz)
    put("gsw", rng.uniform(0.0, 750.0, shape))
    put("glw", rng.uniform(230.0, 400.0, shape))
    put("chs", rng.uniform(0.002, 0.03, shape))
    put("flhc", rng.uniform(0.5, 40.0, shape))
    put("flqc", rng.uniform(0.0005, 0.03, shape))
    put("cpm", np.full(shape, 1004.5 * 1.01))
    put("chs2", rng.uniform(0.0, 0.02, shape))
    put("cqs2", rng.uniform(0.0, 0.02, shape))


def call(driver, atmosphere, k, step=None):
    """One RUC call on the fixture's fields with its per-call forcing.

    ``step`` defaults to the forecast entry point; tests pass the array
    orchestration (``_ruc_lsm_step_reference``) to run the oracle.
    """
    if step is None:
        from gpuwm.core.ruc_runtime import ruc_lsm_step as step
    f = driver.fields
    census = step(
        f, atmosphere, params=driver.ruc_params,
        precipitation=SurfacePrecipitationForcing.from_fields(f),
        dt=12.0, itimestep=k, mosaic_lu=0, mosaic_soil=0, flag_sm_adj=0,
        spp_lsm=0)
    f["rainbl"][...] = 0.0
    SurfacePrecipitationForcing.from_fields(f).clear()
    return census
