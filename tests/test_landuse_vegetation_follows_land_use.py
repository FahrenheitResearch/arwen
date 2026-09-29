"""A land column's vegetation class is its land-use index, whatever its soil map says.

The 30-arc-second soil and land-use maps are independent and disagree along
coasts, rivers and reclaimed land, so a land ``LU_INDEX`` over a water
``SCT_DOM`` (14) is an ordinary geogrid output.  real.exe matches the soil to
the land mask before its final consistency pass
(``dyn_em/module_initialize_real.F:3108-3131`` for the Registry default
``surface_input_source = 3``): the column takes silty clay loam (8) and keeps
``IVGTYP = LU_INDEX``.  WRF v4 real.exe output on two 1 km MODIS domains shows
exactly that for every such cell (land-use 10 and 12 over water soil come out
``IVGTYP`` 10 and 12, ``ISLTYP`` 8, ``LANDMASK`` 1).

Every land-surface scheme reads ``IVGTYP`` from the one derivation tested here,
so these cases cover Noah, Noah-MP, RUC and the slab alike.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from gpuwm.core.landuse import initialize_landuse, reconciled_soil_category

_MMINLU = "MODIFIED_IGBP_MODIS_NOAH"
_ISWATER, _ISLAKE, _ISICE = 17, 21, 15
_WATER_SOIL, _SILTY_CLAY_LOAM = 14, 8
_NSOIL = 4

#: Every MODIS land category Noah's VEGPARM carries (1..20 without water).
_LAND_CATEGORIES = tuple(c for c in range(1, 21) if c != _ISWATER)


def _evidence(kind: str, shape):
    """The soil-temperature and SST evidence a door might hold."""

    if kind == "warm-soil":
        return dict(soil_temperature=np.full((_NSOIL,) + shape, 288.0),
                    sst=np.full(shape, 290.0))
    if kind == "cold-soil-warm-sea":
        return dict(soil_temperature=np.zeros((_NSOIL,) + shape),
                    sst=np.full(shape, 290.0))
    if kind == "none":
        return dict(soil_temperature=None, sst=None)
    raise AssertionError(kind)


def _initialize(lu_index, soil_type, **evidence):
    shape = lu_index.shape
    return initialize_landuse(
        lu_index, soil_type=soil_type,
        landmask=(~np.isin(lu_index, (_ISWATER, _ISLAKE))).astype(np.float32),
        snow=np.zeros(shape), xice=np.zeros(shape),
        valid_time=datetime(2026, 9, 27, 12), cen_lat=52.25, mminlu=_MMINLU,
        iswater=_ISWATER, islake=_ISLAKE, isice=_ISICE, **evidence)


@pytest.mark.parametrize("kind", ["warm-soil", "cold-soil-warm-sea", "none"])
def test_land_over_water_soil_keeps_its_land_use_as_vegetation(kind):
    """Every land category, over water soil, with any evidence or none."""

    lu_index = np.array([_LAND_CATEGORIES], np.int32)
    soil_type = np.full(lu_index.shape, _WATER_SOIL, np.int32)

    landuse = _initialize(lu_index, soil_type, **_evidence(kind, lu_index.shape))

    np.testing.assert_array_equal(landuse.ivgtyp, lu_index)
    np.testing.assert_array_equal(landuse.isltyp, _SILTY_CLAY_LOAM)
    np.testing.assert_array_equal(landuse.landmask, 1.0)
    np.testing.assert_array_equal(landuse.xland, 1.0)
    # The soil ingest builds its column from the same category.
    np.testing.assert_array_equal(
        reconciled_soil_category(
            lu_index, soil_type=soil_type, xice=0.0, iswater=_ISWATER,
            islake=_ISLAKE, isice=_ISICE, **_evidence(kind, lu_index.shape)),
        landuse.isltyp)


def test_real_exe_output_for_land_over_water_soil():
    """What WRF v4 real.exe writes for geogrid land over water soil.

    Land-use 10 (grassland) and 12 (cropland) over soil 14 come out of
    real.exe as IVGTYP 10 and 12, ISLTYP 8, LANDMASK 1, beside ordinary
    land, sea and lake columns that it leaves as they were.
    """

    lu_index = np.array([[10, 12, 12, 10, 17, 21, 12]], np.int32)
    soil_type = np.array([[14, 14, 6, 3, 14, 14, 14]], np.int32)

    landuse = _initialize(lu_index, soil_type,
                          **_evidence("warm-soil", lu_index.shape))

    np.testing.assert_array_equal(landuse.ivgtyp,
                                  [[10, 12, 12, 10, 17, 17, 12]])
    np.testing.assert_array_equal(landuse.isltyp, [[8, 8, 6, 3, 14, 14, 8]])
    np.testing.assert_array_equal(landuse.landmask,
                                  [[1, 1, 1, 1, 0, 0, 1]])
    np.testing.assert_array_equal(landuse.lakemask,
                                  [[0, 0, 0, 0, 0, 1, 0]])


def test_a_nest_of_coast_and_polder_keeps_every_land_vegetation():
    """A 160 x 160 child with sea, a lake, a polder and a shoreline.

    A block of land whose soil map is water (reclaimed land the soil map
    predates) and a scatter of shoreline cells: none may leave its land-use
    class, and water and lakes stay water.
    """

    rng = np.random.default_rng(2152)
    shape = (160, 160)
    lu_index = rng.choice(np.array(_LAND_CATEGORIES, np.int32), size=shape)
    soil_type = rng.integers(1, 13, size=shape).astype(np.int32)
    lu_index[:, :40] = _ISWATER                       # sea
    lu_index[100:120, 60:90] = _ISLAKE                # a lake
    water_like = np.isin(lu_index, (_ISWATER, _ISLAKE))
    soil_type[water_like] = _WATER_SOIL
    soil_type[20:60, 40:100] = _WATER_SOIL            # polder under water soil
    shoreline = rng.random(shape) < 0.05
    soil_type[shoreline & ~water_like] = _WATER_SOIL
    land = ~water_like
    disagreeing = land & (soil_type == _WATER_SOIL)
    assert disagreeing.sum() > 2000

    landuse = _initialize(lu_index, soil_type,
                          **_evidence("warm-soil", shape))

    assert int((land & (landuse.ivgtyp != lu_index)).sum()) == 0
    np.testing.assert_array_equal(landuse.isltyp[disagreeing],
                                  _SILTY_CLAY_LOAM)
    np.testing.assert_array_equal(landuse.isltyp[land & ~disagreeing],
                                  soil_type[land & ~disagreeing])
    np.testing.assert_array_equal(landuse.ivgtyp[water_like], _ISWATER)
    np.testing.assert_array_equal(landuse.isltyp[water_like], _WATER_SOIL)
    np.testing.assert_array_equal(landuse.landmask, land.astype(np.float32))


def test_sea_ice_over_a_water_column_is_still_ice():
    """The match runs before the sea-ice categories, as in real.exe."""

    lu_index = np.array([[_ISWATER, 12]], np.int32)
    soil_type = np.array([[_WATER_SOIL, _WATER_SOIL]], np.int32)
    landuse = initialize_landuse(
        lu_index, soil_type=soil_type, landmask=np.array([[0.0, 1.0]]),
        snow=0.0, xice=np.array([[1.0, 0.0]]),
        valid_time=datetime(2026, 1, 15), cen_lat=70.0, mminlu=_MMINLU,
        iswater=_ISWATER, islake=_ISLAKE, isice=_ISICE)

    np.testing.assert_array_equal(landuse.ivgtyp, [[_ISICE, 12]])
    np.testing.assert_array_equal(landuse.isltyp, [[16, _SILTY_CLAY_LOAM]])
