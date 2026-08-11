"""Phase 3 Task 7: GPU ERA5 horizontal interpolation and wind rotation.

The numerical reference is WPS v4.6.0 ``geogrid/src/interp_module.F``
(``four_pt``, ``sixteen_pt``, and ``oned``).  Wind rotation follows local
WRF v4.6.1 ``share/wrf_fddaobs_in.F:rotate_vector``.  The acceptance oracle
is the bundle's d01 met_em file.
"""
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.preprocess_backend import _rotate_earth_to_grid_cpu
from gpuwm.ingest.horiz import (
    _canonical_psfc_bilinear,
    _WPS_FULL_CHAIN,
    _WPS_SST_CHAIN,
    HorizontalSnapshot,
    interpolate_lake_skin_temperature,
    interpolate_era5_to_lambert,
    interpolate_regular_gpu,
    lambert_rotation,
    masked_nearest_gpu,
    rotate_earth_to_grid_gpu,
    rotate_grid_to_earth_gpu,
    source_orography_from_catalog,
    wps_masked_field_interpolate,
)
from gpuwm.static.lambert import LambertGrid, grids_from_wps_namelist
from gpuwm.static.projection import MercatorGrid, PolarStereoGrid
from gpuwm.verify.npref import (
    era5_rh_to_water_np,
    interpolate_regular_np,
    masked_nearest_np,
    rotate_earth_to_grid_np,
    rotate_grid_to_earth_np,
)

BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
ERA5_NC = BUNDLE / "era5_grib" / "nc" / "era5_19740403_12.nc"
MET_EM = BUNDLE / "met_em" / "met_em.d01.1974-04-03_12_00_00.nc"
D04_MET_EM = BUNDLE / "met_em" / "met_em.d04.1974-04-03_12_00_00.nc"
NAMELIST_WPS = BUNDLE / "namelists" / "namelist.wps"

# Assembly/ratification runs set GPUWM_REQUIRE_CASE_GATES=1 so the bundle
# oracle gates FAIL instead of silently skipping when the data is absent
# (the tests then run and fail loudly on the missing files).
_REQUIRE_GATES = os.environ.get("GPUWM_REQUIRE_CASE_GATES") == "1"
requires_bundle = pytest.mark.skipif(
    not _REQUIRE_GATES and not (ERA5_NC.is_file() and MET_EM.is_file()),
    reason="WRF_1974_MP55 reference bundle not present",
)
requires_d04_surface_bundle = pytest.mark.skipif(
    not _REQUIRE_GATES and not (ERA5_NC.is_file() and D04_MET_EM.is_file()
                                and NAMELIST_WPS.is_file()),
    reason="WRF real74 d04 surface oracle is not present",
)


def test_canonical_psfc_bilinear_pins_single_rounding_bytes():
    latitude = np.array([0.0, 1.0, 2.0])
    longitude = np.array([10.0, 11.0, 12.0])
    field = np.array([
        [100001.3, 99987.1, 100050.7],
        [100100.4, 100025.2, 99950.9],
        [99875.6, 100200.8, 100010.5],
    ], dtype=np.float32)
    target_latitude = np.array([[0.125, 0.875], [1.25, 1.75]])
    target_longitude = np.array([[10.2, 11.6], [10.75, 11.125]])

    actual = _canonical_psfc_bilinear(
        field, latitude, longitude,
        target_latitude, target_longitude)
    np.testing.assert_array_equal(actual.view(np.uint32), np.array([
        [1203983529, 1203980570],
        [1203990384, 1203999838],
    ], dtype=np.uint32))


def test_era5_z_invariant_provider_is_cpu_bilinear_and_fail_loud():
    grid = LambertGrid(
        ref_lat=0.0, ref_lon=0.0, truelat1=30.0, truelat2=60.0,
        stand_lon=0.0, dx=1000.0, dy=1000.0, e_we=4, e_sn=3)
    latitude = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    longitude = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    lon2, lat2 = np.meshgrid(longitude, latitude)
    height = 150.0 + 12.0 * lat2 - 7.0 * lon2
    snapshots = tuple(Era5Snapshot(
        valid_time=datetime(1999, 5, 3, hour),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=latitude, longitude=longitude,
        fields={"SOILGEO": 9.81 * height})
        for hour in (12, 18))
    catalog = SimpleNamespace(
        snapshots=snapshots, units={"SOILGEO": "m2 s-2"})

    actual = source_orography_from_catalog(catalog, grid)
    target_lat, target_lon = grid.latlon_mass()
    target_lon = np.where(target_lon > 180.0, target_lon - 360.0, target_lon)
    np.testing.assert_allclose(
        actual, 150.0 + 12.0 * target_lat - 7.0 * target_lon,
        rtol=0.0, atol=5.0e-13)

    changed = dict(snapshots[1].fields)
    changed["SOILGEO"] = changed["SOILGEO"].copy()
    changed["SOILGEO"][0, 0] += 1.0
    non_invariant = Era5Snapshot(
        valid_time=snapshots[1].valid_time,
        levels_hpa=snapshots[1].levels_hpa,
        latitude=latitude, longitude=longitude, fields=changed)
    with pytest.raises(ValueError, match="declared invariant but changes"):
        source_orography_from_catalog(
            SimpleNamespace(snapshots=(snapshots[0], non_invariant), units={}),
            grid)
    with pytest.raises(ValueError, match="unknown source-orography provider"):
        source_orography_from_catalog(catalog, grid, provider="artifact_guess")
    with pytest.raises(ValueError, match="units metadata is required"):
        source_orography_from_catalog(
            SimpleNamespace(snapshots=snapshots, units={}), grid)
    missing = Era5Snapshot(
        valid_time=snapshots[1].valid_time,
        levels_hpa=snapshots[1].levels_hpa,
        latitude=latitude, longitude=longitude, fields={})
    with pytest.raises(ValueError, match="at every catalog valid time"):
        source_orography_from_catalog(
            SimpleNamespace(snapshots=(snapshots[0], missing),
                            units={"SOILGEO": "m2 s-2"}), grid)


def _small_grid():
    return LambertGrid(
        ref_lat=40.0,
        ref_lon=-85.0,
        truelat1=30.0,
        truelat2=60.0,
        stand_lon=-85.0,
        dx=100_000.0,
        dy=100_000.0,
        e_we=6,
        e_sn=5,
    )


def test_lake_skin_search_expands_beyond_the_masked_field_radius():
    """Raw GEOG lakes use the globally nearest finite ERA5 water skin."""
    grid = LambertGrid(
        ref_lat=0.0, ref_lon=0.0, truelat1=30.0, truelat2=60.0,
        stand_lon=0.0, dx=1000.0, dy=1000.0, e_we=2, e_sn=2)
    latitude = np.linspace(1.0, -1.0, 101, dtype=np.float64)
    longitude = np.linspace(-1.0, 1.0, 101, dtype=np.float64)
    landsea = np.ones((101, 101), dtype=np.float64)
    skin = np.full((101, 101), np.nan, dtype=np.float64)
    # The nearest valid source-water point is 20 source cells away.  This
    # deliberately exceeds the ordinary masked interpolation radius of 8.
    landsea[50, 70] = 0.0
    skin[50, 70] = 278.25
    landsea[80, 80] = 0.0
    skin[80, 80] = 265.0
    snapshot = Era5Snapshot(
        valid_time=datetime(1999, 5, 3, 12),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=latitude, longitude=longitude,
        fields={"LANDSEA": landsea, "SKINTEMP": skin})

    actual = interpolate_lake_skin_temperature(
        snapshot, grid, np.ones((1, 1), dtype=bool))
    assert actual.shape == (1, 1)
    assert actual.dtype == np.float64
    assert actual[0, 0] == 278.25

    no_water = Era5Snapshot(
        valid_time=snapshot.valid_time, levels_hpa=snapshot.levels_hpa,
        latitude=latitude, longitude=longitude,
        fields={"LANDSEA": np.ones_like(landsea),
                "SKINTEMP": np.full_like(skin, 280.0)})
    with pytest.raises(ValueError, match="finite source-water SKINTEMP"):
        interpolate_lake_skin_temperature(
            no_water, grid, np.ones((1, 1), dtype=bool))


@requires_d04_surface_bundle
def test_d04_raw_lakes_match_metgrid_source_water_skin_temperature():
    """CPU oracle for every real74 d04 raw MODIS lake cell."""
    import netCDF4

    with netCDF4.Dataset(ERA5_NC) as dataset:
        source = Era5Snapshot(
            valid_time=datetime(1974, 4, 3, 12),
            levels_hpa=np.array([1000.0], dtype=np.float64),
            latitude=np.asarray(dataset.variables["latitude"][:],
                                dtype=np.float64),
            longitude=np.asarray(dataset.variables["longitude"][:],
                                 dtype=np.float64),
            fields={
                "LANDSEA": np.asarray(dataset.variables["LSM"][0],
                                      dtype=np.float64),
                "SKINTEMP": np.asarray(dataset.variables["SKT"][0],
                                       dtype=np.float64),
            })
    with netCDF4.Dataset(D04_MET_EM) as dataset:
        raw_lakes = np.asarray(dataset.variables["LU_INDEX"][0]) == 21
        expected = np.asarray(
            dataset.variables["SKINTEMP"][0], dtype=np.float64)

    assert np.count_nonzero(raw_lakes) == 1958
    grid = grids_from_wps_namelist(NAMELIST_WPS)[3]
    actual = interpolate_lake_skin_temperature(source, grid, raw_lakes)
    np.testing.assert_allclose(
        actual[raw_lakes], expected[raw_lakes], rtol=0.0, atol=2.0e-5)


def test_bilinear_reference_is_exact_for_affine_field_and_descending_latitude():
    latitude = np.linspace(45.0, 35.0, 6, dtype=np.float64)
    longitude = np.linspace(265.0, 285.0, 7, dtype=np.float64)
    lon2, lat2 = np.meshgrid(longitude, latitude)
    field = 2.0 * lat2 - 0.25 * lon2 + 7.0
    target_lat = np.array([[36.25, 39.5], [41.0, 43.75]], dtype=np.float64)
    target_lon = np.array([[-94.0, -88.25], [-82.0, -76.5]], dtype=np.float64)

    actual = interpolate_regular_np(
        field, latitude, longitude, target_lat, target_lon, method="bilinear"
    )
    expected = 2.0 * target_lat - 0.25 * (target_lon + 360.0) + 7.0
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_overlapping_parabolic_reference_is_exact_for_quadratic_field():
    latitude = np.arange(20.0, 28.0, dtype=np.float64)
    longitude = np.arange(250.0, 259.0, dtype=np.float64)
    lon2, lat2 = np.meshgrid(longitude, latitude)
    field = 0.5 * lon2**2 + 0.25 * lat2**2 - 0.75 * lon2 * lat2
    target_lat = np.array([[22.2, 23.7], [24.4, 25.8]], dtype=np.float64)
    # All targets retain the full i-1:i+2 WPS stencil (the production edge
    # behavior intentionally clamps and is therefore not polynomial-exact).
    target_lon = np.array([[-108.6, -106.1], [-104.25, -103.4]], dtype=np.float64)

    actual = interpolate_regular_np(
        field, latitude, longitude, target_lat, target_lon, method="parabolic"
    )
    tlon = target_lon + 360.0
    expected = 0.5 * tlon**2 + 0.25 * target_lat**2 - 0.75 * tlon * target_lat
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-11)


def test_overlapping_parabolic_treats_physical_zero_as_data_not_missing():
    latitude = np.arange(6, dtype=np.float64)
    longitude = np.arange(7, dtype=np.float64)
    lon2, lat2 = np.meshgrid(longitude, latitude)
    field = lon2 - 2.0 + 0.25 * lat2  # exact zeros occur in the WPS stencil
    target_lat = np.array([[1.4, 2.6]], dtype=np.float64)
    target_lon = np.array([[1.2, 3.4]], dtype=np.float64)
    actual = interpolate_regular_np(
        field, latitude, longitude, target_lat, target_lon, method="parabolic"
    )
    np.testing.assert_allclose(
        actual, target_lon - 2.0 + 0.25 * target_lat, rtol=0.0, atol=1.0e-14
    )


def test_regular_reference_rejects_nonuniform_axes_and_outside_targets():
    field = np.ones((4, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="uniform"):
        interpolate_regular_np(
            field,
            np.array([0.0, 1.0, 2.1, 3.0]),
            np.arange(4, dtype=np.float64),
            np.array([[1.0]]),
            np.array([[1.0]]),
        )
    with pytest.raises(ValueError, match="outside"):
        interpolate_regular_np(
            field,
            np.arange(4, dtype=np.float64),
            np.arange(4, dtype=np.float64),
            np.array([[4.0]]),
            np.array([[1.0]]),
        )


def test_masked_nearest_reference_selects_requested_surface_and_fill():
    latitude = np.arange(4, dtype=np.float64)
    longitude = np.arange(5, dtype=np.float64)
    field = np.arange(20, dtype=np.float64).reshape(4, 5)
    source_land = np.zeros((4, 5), dtype=bool)
    source_land[1:3, 2:4] = True
    target_lat = np.array([[1.2, 1.2], [2.6, 2.6]], dtype=np.float64)
    target_lon = np.array([[1.2, 2.2], [2.7, 4.0]], dtype=np.float64)
    target_land = np.array([[True, False], [True, False]])

    actual = masked_nearest_np(
        field,
        latitude,
        longitude,
        target_lat,
        target_lon,
        source_land,
        target_land,
        surface="match",
    )
    assert source_land[np.unravel_index(actual.astype(int), field.shape)].tolist() == (
        target_land.tolist()
    )
    water_only = masked_nearest_np(
        field,
        latitude,
        longitude,
        target_lat,
        target_lon,
        source_land,
        target_land,
        surface="water",
        fill_value=-999.0,
    )
    assert np.all(water_only[target_land] == -999.0)
    assert np.all(water_only[~target_land] >= 0.0)


def test_wrf_wind_rotation_reference_round_trip():
    rng = np.random.default_rng(7)
    u = rng.normal(size=(3, 4, 5))
    v = rng.normal(size=(3, 4, 5))
    alpha = np.deg2rad(np.linspace(-20.0, 20.0, 20)).reshape(4, 5)
    sina, cosa = np.sin(alpha), np.cos(alpha)
    ug, vg = rotate_earth_to_grid_np(u, v, sina, cosa)
    ue, ve = rotate_grid_to_earth_np(ug, vg, sina, cosa)
    np.testing.assert_allclose(ue, u, rtol=0.0, atol=5.0e-16)
    np.testing.assert_allclose(ve, v, rtol=0.0, atol=5.0e-16)


@pytest.mark.parametrize(
    "grid",
    [
        MercatorGrid(1.3, 103.8, 1.3, 1.3, 103.8,
                     40_000.0, 40_000.0, 8, 7),
        MercatorGrid(-17.8, 178.5, -17.8, -17.8, 178.5,
                     40_000.0, 40_000.0, 8, 7),
        PolarStereoGrid(64.8, -147.7, 64.8, 64.8, -147.7,
                        40_000.0, 40_000.0, 8, 7),
        PolarStereoGrid(-77.85, 166.7, -71.0, -71.0, 166.7,
                        40_000.0, 40_000.0, 8, 7),
    ],
    ids=("mercator-north", "mercator-south",
         "polar-north", "polar-south"),
)
@pytest.mark.parametrize(
    "earth_wind",
    [(7.25, -3.5), (-4.75, 8.125)],
    ids=("southeastward", "northwestward"),
)
def test_new_projection_wind_rotation_both_directions(grid, earth_wind):
    """GFS/ERA5 earth basis -> model grid basis -> uvmet earth basis.

    The source routes call the CPU/GPU equivalent of
    ``_rotate_earth_to_grid_cpu``.  wrf-rust's uvmet path consumes the
    stored SINALPHA/COSALPHA and applies the inverse below.  Exercise both
    projections, both hemispheres (hence multiple latitude bands), and two
    non-axis-aligned wind directions.
    """
    sina, cosa = grid.rotation_m()
    assert np.unique(grid.latlon_mass()[0]).size > 1
    if isinstance(grid, MercatorGrid):
        assert np.all(sina == 0.0) and np.all(cosa == 1.0)
    else:
        assert np.max(np.abs(sina)) > 0.0

    ue = np.full(sina.shape, earth_wind[0], dtype=np.float32)
    ve = np.full(sina.shape, earth_wind[1], dtype=np.float32)
    ug, vg = _rotate_earth_to_grid_cpu(ue, ve, sina, cosa)

    sina32 = np.asarray(sina, dtype=np.float32)
    cosa32 = np.asarray(cosa, dtype=np.float32)
    np.testing.assert_array_equal(
        ug, np.asarray(ue * cosa32 + ve * sina32, dtype=np.float32))
    np.testing.assert_array_equal(
        vg, np.asarray(ve * cosa32 - ue * sina32, dtype=np.float32))

    # Earth-relative inverse used by WRF diagnostics/uvmet.  Unit-norm
    # rotation fields are stored at FP32 in wrfout, so the two-rounding
    # round trip has a small FP32 tolerance rather than a byte identity.
    ue_back = np.asarray(ug * cosa32 - vg * sina32, dtype=np.float32)
    ve_back = np.asarray(vg * cosa32 + ug * sina32, dtype=np.float32)
    np.testing.assert_allclose(ue_back, ue, rtol=2.0e-7, atol=1.0e-6)
    np.testing.assert_allclose(ve_back, ve, rtol=2.0e-7, atol=1.0e-6)


def test_lambert_rotation_uses_task4_mass_sinalpha_cosalpha():
    grid = _small_grid()
    sina, cosa = lambert_rotation(grid, "mass")
    expected_sina, expected_cosa = grid.rotation_m()
    np.testing.assert_array_equal(sina, expected_sina)
    np.testing.assert_array_equal(cosa, expected_cosa)


def _coastal_sst_source():
    """A 0.25-degree strip: land on the left, a lake, land, then open water.

    SST is defined only over water, as every SST analysis is, so the lake's
    stencils reach across the land gap into missing values.
    """
    lat = np.arange(40.0, 44.01, 0.25)
    lon = np.arange(-84.0, -76.99, 0.25)
    ny, nx = lat.size, lon.size
    landsea = np.ones((ny, nx), dtype=np.float64)
    # A lake narrower than a four-point stencil -- the ordinary case for an
    # inland lake on a 0.25 degree grid, and the one WPS abandons.
    landsea[7, 7] = 0.0
    # Open water on the right edge, wide enough for a full 16-point stencil.
    landsea[:, 20:] = 0.0
    sst = np.full((ny, nx), np.nan, dtype=np.float64)
    water = landsea < 0.5
    # A gentle gradient, so any wrong donor shows up as a wrong value.
    gradient = 280.0 + 0.5 * np.arange(nx)[None, :] + 0.1 * np.arange(ny)[:, None]
    sst[water] = gradient[water]
    return lat, lon, landsea, sst


def test_wps_sst_chain_is_metgrid_tbl_and_abandons_the_coast_to_the_fill():
    """The mapped SST field is WPS's, fill and all, and stays that way.

    ``sixteen_pt`` and ``four_pt`` both demand a fully usable stencil, and an
    SST analysis is missing over land, so a lake narrower than a stencil gets
    no mapped SST at all.  That is METGRID.TBL's own behaviour and the
    reconciler, the ``wrf_compat`` policy and every stock-WRF comparison read
    this field expecting exactly it.  The forecast's water temperature is
    assembled elsewhere (gpuwm/ingest/water_temperature.py), from the source
    analysis, which is why this field no longer has to be repaired here.
    """
    assert _WPS_SST_CHAIN == ("sixteen_pt", "four_pt")

    lat, lon, landsea, sst = _coastal_sst_source()
    # Half-cell offset: a model grid does not land on source nodes, and on a
    # node metgrid's four_pt takes its integer-degeneracy path instead of a
    # real stencil.
    target_lat, target_lon = np.meshgrid(lat[:-1] + 0.125, lon[:-1] + 0.125,
                                         indexing="ij")
    mapped = wps_masked_field_interpolate(
        sst, lat, lon, target_lat, target_lon,
        source_valid=np.isfinite(sst),
        target_active=np.ones(target_lat.shape, dtype=bool),
        chain=_WPS_SST_CHAIN, fill_value=0.0)

    lake = np.zeros(target_lat.shape, dtype=bool)
    lake[7, 7] = True
    # The WPS fill over the lake -- zero, which is not a temperature.
    assert mapped[lake][0] == 0.0
    # Open water wide enough for the stencils still resolves.
    assert np.any(mapped[:, 21:] > 0.0)


def test_sst_chain_excludes_search_so_a_landlocked_lake_keeps_its_own_water():
    """``search`` is unbounded, and SST donors are a different basin.

    ``_wps_search`` expands up to 1200 source cells, so appending it to the
    SST chain would let a lake with no analysis of its own take ocean water
    from far outside its basin -- a confident wrong answer where the WPS fill
    is at least diagnosable downstream.  The chain must stop at the local
    operators.
    """
    assert "search" not in _WPS_SST_CHAIN
    assert _WPS_SST_CHAIN[:2] == ("sixteen_pt", "four_pt")

    lat = np.arange(40.0, 44.01, 0.25)
    lon = np.arange(-84.0, -76.99, 0.25)
    sst = np.full((lat.size, lon.size), np.nan, dtype=np.float64)
    # One distant patch of water, and a target with none of its own.
    sst[:3, -3:] = 300.0
    valid = np.isfinite(sst)
    target_lat = np.array([[41.0]])
    target_lon = np.array([[-83.0]])
    got = wps_masked_field_interpolate(
        sst, lat, lon, target_lat, target_lon,
        source_valid=valid, target_active=np.array([[True]]),
        chain=_WPS_SST_CHAIN, fill_value=0.0)
    # The fill, not the distant basin's 300 K.
    assert got[0, 0] == 0.0


def test_wps_search_is_queue_limited_not_global_nearest():
    """WPS ``search_extrap`` FIFO semantics (interp_module.F:451-615).

    The BFS stops expanding once the first usable point is dequeued and
    compares Euclidean distance only among points still in the queue at
    that moment.  A usable point that is globally nearest but not yet
    enqueued must NOT win.  Adversarial-review counterexample: target
    (4.49, 4.49) with donors at (x=0, y=4) [d2=20.4, Manhattan 4, on the
    x-first BFS arm] and (x=7, y=6) [d2=8.58, Manhattan 5].  The
    Manhattan-4 donor is dequeued while the Manhattan-5 donor is still
    outside the queue, so WPS returns 11 even though 22 is closer; a
    global-nearest shortcut returns 22.
    """
    lat = np.arange(10, dtype=np.float64)
    lon = np.arange(10, dtype=np.float64)
    field = np.zeros((10, 10), dtype=np.float64)
    valid = np.zeros((10, 10), dtype=bool)
    field[4, 0] = 11.0   # (x=0, y=4)
    valid[4, 0] = True
    field[6, 7] = 22.0   # (x=7, y=6)
    valid[6, 7] = True
    target_lat = np.array([[4.49]])
    target_lon = np.array([[4.49]])
    got = wps_masked_field_interpolate(
        field, lat, lon, target_lat, target_lon,
        source_valid=valid, target_active=np.array([[True]]),
        chain=("search",), fill_value=-999.0)
    assert got[0, 0] == 11.0


def test_era5_rh_conversion_preserves_warm_values_and_reduces_cold_values():
    rh = np.array([50.0, 50.0, 50.0], dtype=np.float64)
    temperature = np.array([280.0, 260.0, 230.0], dtype=np.float64)
    converted = era5_rh_to_water_np(rh, temperature)
    assert converted[0] == pytest.approx(50.0)
    assert 0.0 < converted[2] < converted[1] < converted[0]


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("method", ["bilinear", "parabolic"])
def test_gpu_regular_interpolation_matches_float64_reference(method):
    import cupy as cp

    rng = np.random.default_rng(11)
    latitude = np.linspace(25.0, 50.0, 9, dtype=np.float64)
    longitude = np.linspace(250.0, 295.0, 10, dtype=np.float64)
    field = rng.normal(size=(3, 9, 10)).astype(np.float64)
    target_lat = rng.uniform(28.5, 46.0, size=(4, 5))
    target_lon = rng.uniform(-106.0, -68.0, size=(4, 5))
    expected = interpolate_regular_np(
        field, latitude, longitude, target_lat, target_lon, method=method
    )
    actual = interpolate_regular_gpu(
        cp.asarray(field, dtype=cp.float32),
        latitude,
        longitude,
        target_lat,
        target_lon,
        method=method,
    )
    assert isinstance(actual, cp.ndarray)
    assert actual.dtype == cp.float32
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=3.0e-6, atol=2.0e-5)


@requires_gpu
@pytest.mark.gpu
def test_gpu_masked_nearest_and_rotation_match_mirrors():
    import cupy as cp

    latitude = np.arange(4, dtype=np.float64)
    longitude = np.arange(5, dtype=np.float64)
    field = np.arange(20, dtype=np.float64).reshape(4, 5)
    source_land = np.zeros((4, 5), dtype=bool)
    source_land[1:3, 2:4] = True
    target_lat = np.array([[1.2, 1.2], [2.6, 2.6]], dtype=np.float64)
    target_lon = np.array([[1.2, 2.2], [2.7, 4.0]], dtype=np.float64)
    target_land = np.array([[True, False], [True, False]])
    expected = masked_nearest_np(
        field,
        latitude,
        longitude,
        target_lat,
        target_lon,
        source_land,
        target_land,
        surface="match",
    )
    actual = masked_nearest_gpu(
        cp.asarray(field, dtype=cp.float32),
        latitude,
        longitude,
        target_lat,
        target_lon,
        cp.asarray(source_land),
        cp.asarray(target_land),
        surface="match",
    )
    np.testing.assert_array_equal(cp.asnumpy(actual), expected.astype(np.float32))

    u = cp.arange(24, dtype=cp.float32).reshape(2, 3, 4) / 7.0
    v = -0.25 * u
    sina = cp.asarray(np.linspace(-0.2, 0.2, 12).reshape(3, 4), dtype=cp.float32)
    cosa = cp.sqrt(1.0 - sina * sina)
    ug, vg = rotate_earth_to_grid_gpu(u, v, sina, cosa)
    ue, ve = rotate_grid_to_earth_gpu(ug, vg, sina, cosa)
    np.testing.assert_allclose(cp.asnumpy(ue), cp.asnumpy(u), rtol=2.0e-7, atol=2.0e-7)
    np.testing.assert_allclose(cp.asnumpy(ve), cp.asnumpy(v), rtol=2.0e-7, atol=2.0e-7)


@requires_gpu
@pytest.mark.gpu
def test_snapshot_interpolates_every_field_to_expected_stagger_and_fp32():
    import cupy as cp

    grid = _small_grid()
    latitude = np.linspace(34.0, 46.0, 9, dtype=np.float64)
    longitude = np.linspace(267.0, 283.0, 10, dtype=np.float64)
    lon2, lat2 = np.meshgrid(longitude, latitude)
    levels = np.array([500.0, 1000.0], dtype=np.float64)
    base = lat2 + 0.1 * lon2
    landsea = (lon2 >= 275.0).astype(np.float64)
    snapshot = Era5Snapshot(
        valid_time=datetime(1974, 4, 3, 12),
        levels_hpa=levels,
        latitude=latitude,
        longitude=longitude,
        fields={
            "Z": np.stack((9.81 * base, 9.81 * (base + 1.0))),
            "T": np.stack((base + 190.0, base + 200.0)),
            "U": np.stack((base - 60.0, base - 58.0)),
            "V": np.stack((0.5 * base - 20.0, 0.5 * base - 19.0)),
            "RH": np.stack((base, base + 2.0)),
            "PSFC": 90_000.0 + base,
            "T2": 250.0 + base,
            "D2": 245.0 + base,
            "U10": base - 60.0,
            "V10": 0.5 * base - 20.0,
            "LANDSEA": landsea,
            "SKINTEMP": 250.0 + base,
            "SST": np.where(landsea < 0.5, 270.0 + base, np.nan),
            "SEAICE": np.where(
                landsea < 0.5, np.clip((275.0 - lon2) / 8.0, 0.0, 1.0),
                np.nan),
            "SOILGEO": 9.81 * (100.0 + base),
            "ST000007": 260.0 + base,
            "SM000007": 0.2 + 0.001 * base,
            "SNOW_EC": 0.01 * base,
        },
    )
    catalog = SimpleNamespace(
        snapshots=(snapshot,), units={"SOILGEO": "m2 s-2"})
    result = interpolate_era5_to_lambert(
        snapshot, grid, source_orography_catalog=catalog)
    assert isinstance(result, HorizontalSnapshot)
    assert result.valid_time == snapshot.valid_time
    np.testing.assert_array_equal(result.levels_hpa, levels)
    assert set(result.fields) == {
        "GHT", "TT", "UU", "VV", "RH", "PSFC", "T2", "D2",
        "U10", "V10", "LANDSEA", "SKINTEMP", "SST", "ST000007",
        "XICE", "SOURCE_OROGRAPHY", "SM000007", "SNOW_EC",
    }
    ny, nx = grid.e_sn - 1, grid.e_we - 1
    assert result.fields["TT"].shape == (2, ny, nx)
    assert result.fields["UU"].shape == (2, ny, nx + 1)
    assert result.fields["VV"].shape == (2, ny + 1, nx)
    assert result.fields["U10"].shape == (ny, nx + 1)
    assert result.fields["V10"].shape == (ny + 1, nx)
    for name, value in result.fields.items():
        if name == "SOURCE_OROGRAPHY":
            assert isinstance(value, np.ndarray) and value.dtype == np.float64
            continue
        assert isinstance(value, cp.ndarray)
        assert value.dtype == cp.float32
    target_land = cp.asnumpy(result.fields["LANDSEA"]) >= 0.5
    # METGRID.TBL SST: coastal sixteen_pt/four_pt stencils that touch a
    # missing land source collapse to fill 0; interior land is always 0.
    assert np.all(cp.asnumpy(result.fields["SST"])[target_land] == 0.0)
    assert np.all(cp.asnumpy(result.fields["XICE"])[target_land] == 0.0)
    assert np.any(cp.asnumpy(result.fields["XICE"])[~target_land] > 0.0)
    # METGRID.TBL fill_missing on masked-out target cells: ST 285, SM 1,
    # snow family 0 (masked=water fields are undefined over model water).
    assert np.all(cp.asnumpy(result.fields["ST000007"])[~target_land] == 285.0)
    assert np.all(cp.asnumpy(result.fields["SM000007"])[~target_land] == 1.0)
    assert np.all(cp.asnumpy(result.fields["SNOW_EC"])[~target_land] == 0.0)


def _sub_grid_island_grid():
    """A fine tropical mass grid that resolves a source-sub-grid island."""
    return MercatorGrid(13.4, 144.8, 13.4, 13.4, 144.8,
                        6_000.0, 6_000.0, 13, 11)


_ISLAND_SOURCE_LAT = np.linspace(12.5, 14.5, 9, dtype=np.float64)
_ISLAND_SOURCE_LON = np.linspace(143.5, 146.0, 11, dtype=np.float64)


def _island_source_fields(peak_fraction):
    """Source with land ONLY as an area fraction at three cells."""
    shape = (_ISLAND_SOURCE_LAT.size, _ISLAND_SOURCE_LON.size)
    landsea = np.zeros(shape, dtype=np.float64)
    landsea[4, 5] = peak_fraction
    landsea[4, 6] = peak_fraction * 0.75
    landsea[3, 5] = peak_fraction * 0.6
    land = landsea > 0.0
    skin = np.full(shape, 301.0, dtype=np.float64)
    skin[land] = 305.0
    soil_t = np.full(shape, 302.0, dtype=np.float64)
    soil_t[land] = 306.0
    soil_m = np.full(shape, 0.02, dtype=np.float64)
    soil_m[land] = 0.31
    return {"LANDSEA": landsea, "SKINTEMP": skin,
            "ST000007": soil_t, "SM000007": soil_m}


@requires_gpu
@pytest.mark.gpu
def test_island_smaller_than_a_source_cell_takes_the_fractional_land_state():
    """The one case WPS cannot serve: no flagged source land anywhere.

    ``ungrib`` binarizes an ECMWF land-sea mask at the half mark
    (rrpr.F:869-876), so an island whose area is a fraction of a 0.25
    degree cell leaves metgrid with no land donor at all and every land
    target of a domain that RESOLVES the island takes METGRID.TBL
    fill_missing: 0 K skin, 285 K soil, 1.0 soil moisture.  The
    second-chance pass reads the fraction instead.
    """
    import cupy as cp

    grid = _sub_grid_island_grid()
    fields = _island_source_fields(0.15)
    assert not np.any(fields["LANDSEA"] > 0.5), "fixture must starve WPS"
    mass_lat, mass_lon = grid.latlon_mass()
    target_land = np.zeros(mass_lat.shape, dtype=bool)
    target_land[4:7, 5:8] = True

    snapshot = Era5Snapshot(
        valid_time=datetime(2000, 1, 1, 0),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=_ISLAND_SOURCE_LAT, longitude=_ISLAND_SOURCE_LON,
        fields=fields)
    result = interpolate_era5_to_lambert(
        snapshot, grid, target_landmask=target_land)

    got_skin = cp.asnumpy(result.fields["SKINTEMP"])
    got_st = cp.asnumpy(result.fields["ST000007"])
    got_sm = cp.asnumpy(result.fields["SM000007"])
    # No land target carries the fill, and every cell is a temperature the
    # surface preprocessor accepts (its 170..400 K guard).
    assert not np.any(got_skin == 0.0)
    assert np.all(got_skin > 170.0) and np.all(got_skin < 400.0)
    assert np.all(got_st[target_land] > 285.0)
    assert np.all(got_sm[target_land] > 0.02)
    # ... and the masked-out side is untouched METGRID.TBL fill_missing.
    assert np.all(got_st[~target_land] == 285.0)
    assert np.all(got_sm[~target_land] == 1.0)

    # The recovered values are the fractional cells' own land state, read
    # through the unchanged WPS chain.
    expected_skin_land = wps_masked_field_interpolate(
        fields["SKINTEMP"], _ISLAND_SOURCE_LAT, _ISLAND_SOURCE_LON,
        mass_lat, mass_lon,
        source_valid=fields["LANDSEA"] > 0.0, target_active=target_land,
        chain=_WPS_FULL_CHAIN, fill_value=0.0)
    np.testing.assert_array_equal(
        got_skin[target_land],
        expected_skin_land.astype(np.float32)[target_land])


@requires_gpu
@pytest.mark.gpu
def test_a_source_with_flagged_land_never_reaches_the_second_chance():
    """WPS parity is the default: pass two is gated on an EMPTY donor set.

    Add one cell the binarized flag calls land and the whole field must
    come from metgrid's own pass, fractional neighbours excluded exactly
    as ``make_zero_or_one`` excludes them.
    """
    import cupy as cp

    grid = _sub_grid_island_grid()
    fields = _island_source_fields(0.15)
    fields["LANDSEA"] = fields["LANDSEA"].copy()
    fields["LANDSEA"][0, 0] = 1.0
    fields["SKINTEMP"] = fields["SKINTEMP"].copy()
    fields["SKINTEMP"][0, 0] = 288.0
    flagged = fields["LANDSEA"] > 0.5
    assert np.count_nonzero(flagged) == 1

    mass_lat, mass_lon = grid.latlon_mass()
    target_land = np.zeros(mass_lat.shape, dtype=bool)
    target_land[4:7, 5:8] = True
    snapshot = Era5Snapshot(
        valid_time=datetime(2000, 1, 1, 0),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=_ISLAND_SOURCE_LAT, longitude=_ISLAND_SOURCE_LON,
        fields=fields)
    result = interpolate_era5_to_lambert(
        snapshot, grid, target_landmask=target_land)

    wps_only = wps_masked_field_interpolate(
        fields["SKINTEMP"], _ISLAND_SOURCE_LAT, _ISLAND_SOURCE_LON,
        mass_lat, mass_lon,
        source_valid=flagged, target_active=target_land,
        chain=_WPS_FULL_CHAIN, fill_value=0.0)
    got_skin = cp.asnumpy(result.fields["SKINTEMP"])
    np.testing.assert_array_equal(
        got_skin[target_land], wps_only.astype(np.float32)[target_land])
    # The single flagged cell is the only donor, so every land target is
    # its value -- the fractional cells next door never contribute.
    assert np.all(got_skin[target_land] == np.float32(288.0))


@requires_gpu
@pytest.mark.gpu
def test_seaice_without_local_finite_support_zero_fills_open_water():
    import cupy as cp

    grid = _small_grid()
    latitude = np.linspace(34.0, 46.0, 9, dtype=np.float64)
    longitude = np.linspace(267.0, 283.0, 10, dtype=np.float64)
    snapshot = Era5Snapshot(
        valid_time=datetime(1999, 5, 3, 12),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=latitude, longitude=longitude,
        fields={
            "LANDSEA": np.zeros((9, 10), dtype=np.float64),
            "SEAICE": np.full((9, 10), np.nan, dtype=np.float64),
        })
    result = interpolate_era5_to_lambert(snapshot, grid)
    np.testing.assert_array_equal(cp.asnumpy(result.fields["XICE"]), 0.0)


@requires_gpu
@pytest.mark.gpu
def test_snapshot_coastal_mask_mismatch_uses_nearest_requested_surface():
    import cupy as cp

    grid = _small_grid()
    latitude = np.linspace(34.0, 46.0, 9, dtype=np.float64)
    longitude = np.linspace(267.0, 283.0, 10, dtype=np.float64)
    lon_index, lat_index = np.meshgrid(
        np.arange(longitude.size), np.arange(latitude.size)
    )
    source_land = (lat_index + lon_index) % 2 == 0
    source_value = 260.0 + 2.0 * lat_index + lon_index
    mass_lat, mass_lon = grid.latlon_mass()
    rounded_land = interpolate_regular_np(
        source_land.astype(np.float64),
        latitude,
        longitude,
        mass_lat,
        mass_lon,
        method="nearest",
    ).astype(bool)
    target_land = ~rounded_land
    assert np.all(target_land != rounded_land)

    snapshot = Era5Snapshot(
        valid_time=datetime(1974, 4, 3, 12),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=latitude,
        longitude=longitude,
        fields={
            "LANDSEA": source_land.astype(np.float64),
            "SKINTEMP": source_value,
            "SST": np.where(~source_land, source_value + 10.0, np.nan),
            "ST000007": np.where(source_land, source_value + 20.0, np.nan),
            "SNOW_EC": np.where(source_land, source_value + 30.0, np.nan),
        },
    )
    result = interpolate_era5_to_lambert(
        snapshot, grid, target_landmask=target_land
    )

    np.testing.assert_array_equal(cp.asnumpy(result.fields["LANDSEA"]), target_land)
    finite_st = np.where(source_land, source_value + 20.0, 0.0)
    finite_snow = np.where(source_land, source_value + 30.0, 0.0)
    finite_sst = np.where(~source_land, source_value + 10.0, 0.0)
    # WPS chain mirrors: masked=both SKINTEMP combines a land-source pass on
    # land targets with a water-source pass on water targets; ST/SNOW are
    # land-masked with METGRID.TBL fills; SST is unmasked sixteen_pt+four_pt
    # over its finite (water) support with fill 0.
    skin_land = wps_masked_field_interpolate(
        source_value, latitude, longitude, mass_lat, mass_lon,
        source_valid=source_land, target_active=target_land,
        chain=_WPS_FULL_CHAIN, fill_value=0.0)
    skin_water = wps_masked_field_interpolate(
        source_value, latitude, longitude, mass_lat, mass_lon,
        source_valid=~source_land, target_active=~target_land,
        chain=_WPS_FULL_CHAIN, fill_value=0.0)
    expected = {
        "SKINTEMP": np.where(target_land, skin_land, skin_water),
        "ST000007": wps_masked_field_interpolate(
            finite_st, latitude, longitude, mass_lat, mass_lon,
            source_valid=source_land, target_active=target_land,
            chain=_WPS_FULL_CHAIN, fill_value=285.0),
        "SNOW_EC": wps_masked_field_interpolate(
            finite_snow, latitude, longitude, mass_lat, mass_lon,
            source_valid=source_land, target_active=target_land,
            chain=("four_pt", "average_4pt"), fill_value=0.0),
        "SST": wps_masked_field_interpolate(
            np.where(~source_land, source_value + 10.0, np.nan),
            latitude, longitude, mass_lat, mass_lon,
            source_valid=np.ones_like(source_land, dtype=bool),
            target_active=np.ones(target_land.shape, dtype=bool),
            chain=("sixteen_pt", "four_pt"), fill_value=0.0),
    }
    for name, mirror in expected.items():
        np.testing.assert_array_equal(
            cp.asnumpy(result.fields[name]), mirror.astype(np.float32),
            err_msg=name,
        )
    assert np.all(cp.asnumpy(result.fields["SKINTEMP"]) > 0.0)
    # The checkerboard source guarantees every masked stencil is rejected
    # down to the search/average fallbacks; land targets still resolve real
    # land-source values and masked-out cells carry METGRID.TBL fills.
    assert np.all(cp.asnumpy(result.fields["ST000007"])[target_land] > 0.0)
    assert np.all(cp.asnumpy(result.fields["ST000007"])[~target_land] == 285.0)
    assert np.all(cp.asnumpy(result.fields["SNOW_EC"])[target_land] > 0.0)


@lru_cache(maxsize=1)
def _bundle_snapshot():
    import netCDF4

    names = {
        "Z": "Z",
        "T": "T",
        "U": "U",
        "V": "V",
        "RH": "R",
        "PSFC": "SP",
        "SKINTEMP": "SKT",
        "SST": "SSTK",
        "LANDSEA": "LSM",
    }
    with netCDF4.Dataset(ERA5_NC) as ds:
        fields = {
            canonical: np.asarray(
                np.ma.filled(ds.variables[nc_name][0], np.nan), dtype=np.float64
            )
            for canonical, nc_name in names.items()
        }
        return Era5Snapshot(
            valid_time=datetime(1974, 4, 3, 12),
            levels_hpa=np.asarray(ds.variables["level"][:], dtype=np.float64),
            latitude=np.asarray(ds.variables["latitude"][:], dtype=np.float64),
            longitude=np.asarray(ds.variables["longitude"][:], dtype=np.float64),
            fields=fields,
        )


@requires_bundle
@requires_gpu
@pytest.mark.gpu
def test_d01_met_em_oracle_rmse_gates_and_wind_inverse():
    import cupy as cp
    import netCDF4

    grid = grids_from_wps_namelist(NAMELIST_WPS)[0]
    source = _bundle_snapshot()
    result = interpolate_era5_to_lambert(source, grid)
    with netCDF4.Dataset(MET_EM) as ds:
        # met_em level 0 is the surface pseudo-level; pressure levels then run
        # from 1000 to 1 hPa, opposite Era5Snapshot's ascending metadata.
        expected = {
            "TT": np.asarray(ds.variables["TT"][0, 1:][::-1], dtype=np.float64),
            "UU": np.asarray(ds.variables["UU"][0, 1:][::-1], dtype=np.float64),
            "VV": np.asarray(ds.variables["VV"][0, 1:][::-1], dtype=np.float64),
            "RH": np.asarray(ds.variables["RH"][0, 1:][::-1], dtype=np.float64),
            "GHT": np.asarray(ds.variables["GHT"][0, 1:][::-1], dtype=np.float64),
            "PSFC": np.asarray(ds.variables["PSFC"][0], dtype=np.float64),
            "LANDSEA": np.asarray(ds.variables["LANDSEA"][0], dtype=np.float64),
            "SKINTEMP": np.asarray(ds.variables["SKINTEMP"][0], dtype=np.float64),
            "SST": np.asarray(ds.variables["SST"][0], dtype=np.float64),
        }

    actual = {name: cp.asnumpy(value).astype(np.float64)
              for name, value in result.fields.items() if name in expected}
    level_rmse = {
        name: np.sqrt(np.mean((actual[name] - expected[name]) ** 2, axis=(-2, -1)))
        for name in ("TT", "UU", "VV", "RH", "GHT")
    }
    assert level_rmse["TT"].max() <= 0.15
    assert level_rmse["UU"].max() <= 0.2
    assert level_rmse["VV"].max() <= 0.2
    assert level_rmse["RH"].max() <= 1.0
    assert level_rmse["GHT"].max() <= 2.0
    assert np.sqrt(np.mean((actual["PSFC"] - expected["PSFC"]) ** 2)) <= 50.0
    np.testing.assert_array_equal(actual["LANDSEA"], expected["LANDSEA"])
    assert np.sqrt(np.mean((actual["SKINTEMP"] - expected["SKINTEMP"]) ** 2)) <= 0.3
    assert np.mean(np.abs(actual["SKINTEMP"] - expected["SKINTEMP"])) <= 0.05
    # WPS-chain SKINTEMP must not exhibit nearest-neighbour plateaus: the
    # met_em oracle's own adjacent-equal land fraction is ~1.5e-3.
    land2 = (expected["LANDSEA"] >= 0.5)
    pair = land2[:, :-1] & land2[:, 1:]
    plateau = np.sum(
        (actual["SKINTEMP"][:, :-1] == actual["SKINTEMP"][:, 1:]) & pair
    ) / np.sum(pair)
    assert plateau <= 0.005
    # METGRID.TBL SST (sixteen_pt+four_pt over the finite water support,
    # fill 0) reproduces the oracle's support pattern -- including its 63
    # nonzero coastal-land leaks -- to within a single boundary cell, and
    # its values at FP32 rounding.
    expected_support = expected["SST"] != 0.0
    actual_support = actual["SST"] != 0.0
    assert np.sum(expected_support != actual_support) <= 2
    common_sst = expected_support & actual_support
    assert np.sqrt(np.mean(
        (actual["SST"][common_sst] - expected["SST"][common_sst]) ** 2
    )) <= 0.01

    # Destagger the production winds to mass points, independently invert the
    # rotation, and compare with the source interpolated by the float64 mirror.
    # The residual includes the expected stagger/destagger truncation error.
    ug = 0.5 * (actual["UU"][..., :, :-1] + actual["UU"][..., :, 1:])
    vg = 0.5 * (actual["VV"][..., :-1, :] + actual["VV"][..., 1:, :])
    sina, cosa = lambert_rotation(grid, "mass")
    ue_back, ve_back = rotate_grid_to_earth_np(ug, vg, sina, cosa)
    mlat, mlon = grid.latlon_mass()
    ue = interpolate_regular_np(
        source.fields["U"], source.latitude, source.longitude,
        mlat, mlon, method="parabolic",
    )
    ve = interpolate_regular_np(
        source.fields["V"], source.latitude, source.longitude,
        mlat, mlon, method="parabolic",
    )
    inverse_level_rmse = {
        "U": np.sqrt(np.mean((ue_back - ue) ** 2, axis=(-2, -1))),
        "V": np.sqrt(np.mean((ve_back - ve) ** 2, axis=(-2, -1))),
    }
    assert inverse_level_rmse["U"].max() <= 0.025
    assert inverse_level_rmse["V"].max() <= 0.025
