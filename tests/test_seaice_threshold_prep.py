"""real.exe's ``seaice_threshold`` rule reaches the initial surface.

Referee: the HRRR v4.1.21 fork's share/module_soil_pre.F
adjust_for_seaice_post, RUCLSMSCHEME arm (:337-405).  A WATER cell becomes
sea ice where ``XICE >= xice_threshold`` OR ``TSK < seaice_threshold``
(TSK the saved input skin, module_initialize_real.F:2749, :3449).  Such a
cell takes the ice land use and soil class 16, LANDMASK 1, TMN 271.4, the
blended skin ``XICE*min(271.4, TSK) + (1 - XICE)*SST`` and the ice column;
under ``fractional_seaice = 1`` its XICE is left as analysed (possibly 0),
under 0 it is set to 1.  HRRR v4 runs ``seaice_threshold = 271.4``.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from test_soil_node_contract import _node_contract


F = np.float32
SHAPE = (1, 4)
# cold open water (XICE 0) | warm open water | cold water, XICE 0.01 | land
SKIN = [[268.0, 285.0, 269.0, 260.0]]
SST = [[272.0, 286.0, 272.5, 280.0]]
XICE = [[0.0, 0.0, 0.01, 0.0]]
LANDMASK = np.array([[0.0, 0.0, 0.0, 1.0]])


def _fields(shape, xice, *, sst):
    from gpuwm.ingest.soil_contract import (MAPPED_SOIL_MOISTURE,
                                            MAPPED_SOIL_TEMPERATURE)
    return {"SKINTEMP": np.full(shape, 268.0), "LANDSEA": np.zeros(shape),
            "SEAICE": np.asarray(xice, dtype=np.float64),
            "SST": np.asarray(sst, dtype=np.float64),
            "SNOW": np.zeros(shape), "SNOWH": np.zeros(shape),
            MAPPED_SOIL_TEMPERATURE: np.full((9, *shape), 280.0),
            MAPPED_SOIL_MOISTURE: np.full((9, *shape), 0.3)}


def _kw(shape):
    return dict(sf_surface_physics=3, num_soil_layers=9,
                soil_type=np.full(shape, 14),
                deep_soil_temperature=np.full(shape, 280.0),
                soil_layer_contract=_node_contract(),
                landmask=np.zeros(shape), water_temperature_policy="wrf_compat")


def _deck():
    fields = _fields(SHAPE, XICE, sst=SST)
    fields["SKINTEMP"] = np.asarray(SKIN, dtype=np.float64)
    fields["LANDSEA"] = LANDMASK.copy()
    kw = _kw(SHAPE)
    kw["landmask"] = LANDMASK.copy()
    return fields, kw


def _soil(**settings):
    from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
    fields, kw = _deck()
    return preprocess_land_surface_soil(fields, **kw, **settings)


def _ice_column(tsk, tmn, n):
    """module_soil_pre.F:375-385 (fork), scalar float32."""
    total = F(3.0)
    column = [F(tsk)] + [F(0.0)] * (n - 2) + [F(tmn)]
    for loop in range(2, n):
        mid = F(F(F(total / F(n)) / F(4.0)) + F(F(loop - 2) * F(total / F(n))))
        column[loop - 1] = F(F(F(F(total - mid) * F(tsk))
                               + F(mid * F(tmn))) / total)
    return np.array(column, dtype=F)


def test_cold_open_water_becomes_ruc_land_ice_with_xice_kept():
    default = _soil(fractional_seaice=True)
    frac = _soil(fractional_seaice=True, seaice_threshold=271.4)
    binary = _soil(fractional_seaice=False, seaice_threshold=271.4)

    # Without the rule the two cold cells stay open water.
    np.testing.assert_array_equal(default.landmask, [[0, 0, 0, 1]])
    # With it they are ice; warm water and land are untouched.
    for state in (frac, binary):
        np.testing.assert_array_equal(state.landmask, [[1, 0, 1, 1]])
        np.testing.assert_array_equal(state.xland, [[1, 2, 1, 1]])
        np.testing.assert_array_equal(state.sea_ice_cells,
                                      [[True, False, True, False]])
        for name in ("tsk", "soil_temperature", "soil_moisture",
                     "liquid_moisture", "deep_soil_temperature"):
            got = np.asarray(getattr(state, name))
            want = np.asarray(getattr(default, name))
            np.testing.assert_array_equal(got[..., 1], want[..., 1], name)
            np.testing.assert_array_equal(got[..., 3], want[..., 3], name)
    # fractional: XICE stays as analysed, 0 and 0.01 (not raised to 0.02)
    np.testing.assert_array_equal(np.asarray(frac.xice, F),
                                  F([[0.0, 0.0, 0.01, 0.0]]))
    # binary: XICE = 1 on every converted cell
    np.testing.assert_array_equal(binary.xice, [[1, 0, 1, 0]])

    # The fork's RUC column on the fractional cells: TMN 271.4, blended
    # skin XICE*min(271.4, TSK) + (1 - XICE)*SST, ice column, SMOIS 1, SH2O 0.
    n = frac.soil_temperature.shape[0]
    tmn = np.asarray(frac.deep_soil_temperature)
    assert tmn[0, 0] == 271.4 and tmn[0, 2] == 271.4
    for i in (0, 2):
        x = F(XICE[0][i])
        skin = F(F(x * F(min(F(271.4), F(SKIN[0][i]))))
                 + F(F(F(1) - x) * F(SST[0][i])))
        assert F(frac.tsk[0, i]) == skin
        np.testing.assert_array_equal(
            np.asarray(frac.soil_temperature, F)[:, 0, i],
            _ice_column(skin, tmn[0, i], n))
        np.testing.assert_array_equal(np.asarray(frac.soil_moisture)[:, 0, i], 1.0)
        np.testing.assert_array_equal(
            np.asarray(frac.liquid_moisture)[:, 0, i], 0.0)
    # XICE 0 blends to the SST outright.
    assert F(frac.tsk[0, 0]) == F(272.0)


def test_landuse_and_soil_categories_agree_on_threshold_cells():
    from gpuwm.core.landuse import (initialize_landuse, prepared_sea_ice_cells,
                                    reconciled_soil_category)
    from gpuwm.native_wrf_contract import canonical_noah_surface

    frac = _soil(fractional_seaice=True, seaice_threshold=271.4)
    lu = np.array([[17, 17, 17, 10]], np.int32)
    soil_type = np.array([[14, 14, 14, 8]], np.int32)
    common = dict(
        valid_time=datetime(2024, 1, 15, 0), cen_lat=45.0,
        mminlu="MODIFIED_IGBP_MODIS_NOAH", iswater=17, islake=21, isice=15,
        fractional_seaice=True)
    without = initialize_landuse(
        lu, soil_type=soil_type, landmask=LANDMASK, snow=0.0,
        xice=frac.xice, **common)
    landuse = initialize_landuse(
        lu, soil_type=soil_type, landmask=LANDMASK, snow=0.0,
        xice=frac.xice, sea_ice_cells=frac.sea_ice_cells, **common)
    # adjust_for_seaice_post's categories on both converted cells
    np.testing.assert_array_equal(landuse.ivgtyp, [[15, 17, 15, 10]])
    np.testing.assert_array_equal(landuse.isltyp, [[16, 14, 16, 8]])
    # the land use the driver integrates matches the soil state's mask
    np.testing.assert_array_equal(landuse.landmask, frac.landmask)
    np.testing.assert_array_equal(landuse.xland, frac.xland)
    # landuse_init reads XICE for the ice blend (module_physics_init.F:1633):
    # the XICE-0 cell takes the ice category's own row, unblended; the
    # XICE-0.01 cell too (0.01 < 0.02).  Without the mask they were water.
    assert landuse.albedo[0, 0] == landuse.albbck[0, 0]
    assert landuse.albedo[0, 0] != without.albedo[0, 0]
    assert landuse.emiss[0, 2] == landuse.embck[0, 2]
    np.testing.assert_array_equal(landuse.albedo[0, 1], without.albedo[0, 1])
    # the pre-ingest soil category takes the same decision
    np.testing.assert_array_equal(
        reconciled_soil_category(
            lu, soil_type=soil_type, xice=frac.xice, iswater=17, islake=21,
            isice=15, fractional_seaice=True,
            sea_ice_cells=frac.sea_ice_cells),
        landuse.isltyp)
    # a prepared (cached) surface recovers the same cells
    fields, _ = _deck()
    np.testing.assert_array_equal(
        prepared_sea_ice_cells(canonical_noah_surface(frac), fields,
                               LANDMASK),
        frac.sea_ice_cells)


def test_noah_binary_threshold_and_config_door():
    from gpuwm.config import (ConfigurationRefused, RunConfig,
                              validate_run_config)
    from gpuwm.ingest.soil import seaice_threshold_cells

    # 100 K (the Registry default) and anything <= 170 K never trigger
    water = np.ones(SHAPE, bool)
    assert not seaice_threshold_cells(np.full(SHAPE, 171.0), water, 100.0).any()
    # REAL comparison: a skin equal to the threshold is not below it
    np.testing.assert_array_equal(
        seaice_threshold_cells(np.array([[271.4, 271.39, 271.41, 250.0]]),
                               np.array([[True, True, True, False]]), 271.4),
        [[False, True, False, False]])
    base = dict(nx=8, ny=6, nz=10, dx=3000.0, dy=3000.0, ztop=20000.0,
                dt=18.0, run_seconds=36.0)
    cfg = validate_run_config(RunConfig(**base))
    assert cfg.seaice_threshold == 100.0
    with pytest.raises(ConfigurationRefused, match="seaice_threshold"):
        validate_run_config(RunConfig(**base, seaice_threshold=float("nan")))
