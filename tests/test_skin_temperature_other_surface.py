"""A surface the source crop holds no cell of takes the other surface's skin.

METGRID.TBL maps SKINTEMP ``masked=both``: land targets from source land,
water targets from source water, and a target its own surface cannot
reach keeps fill_missing, 0 K.  The WPS search reaches the whole source
array, so that happens exactly when the crop holds none of the target's
surface: a regional crop of a coarse source over an inland domain has no
water for its lakes, and one over open ocean no land for its islands.
The water-temperature assembly refused such lakes ("N water cells have no
admissible water temperature") and the soil initializer such islands
("TSK contains non-finite or nonphysical values").
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from gpuwm.ingest import water_temperature as wt
from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.horiz import (
    _WPS_FULL_CHAIN,
    _regular_coordinates,
    interpolate_era5_to_lambert,
    wps_masked_field_interpolate,
)
from gpuwm.ingest.water_temperature import WaterTemperatureStatics
from gpuwm.static.lambert import LambertGrid
from gpuwm.verify.npref import (
    interpolate_regular_np,
    masked_nearest_np,
    rotate_earth_to_grid_np,
)
from conftest import requires_wps_masked_chain_bridge

#: The tests here map masked fields through the native chain.
pytestmark = requires_wps_masked_chain_bridge

LAKE = 21
_LATITUDE = 33.0 + 0.25 * np.arange(21, dtype=np.float64)
_LONGITUDE = -100.0 + 0.25 * np.arange(21, dtype=np.float64)


class _NumpyBackend:
    """The preprocessing ABI on NumPy, so the test needs no card or bridge."""

    name = "numpy-reference-test"
    array_module = np

    @staticmethod
    def float32(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def bool_array(value):
        return np.asarray(value, dtype=bool)

    @staticmethod
    def regular_plan(latitude, longitude, target_lat, target_lon):
        class _Plan:
            source_shape = (len(latitude), len(longitude))
            target_shape = np.shape(target_lat)

            @staticmethod
            def apply(field, method="parabolic", *, source_support=False):
                return np.asarray(interpolate_regular_np(
                    field, latitude, longitude, target_lat, target_lon,
                    method=method), dtype=np.float32)

        return _Plan()

    @staticmethod
    def masked_nearest(*args, **kwargs):
        return masked_nearest_np(*args, **kwargs)

    @staticmethod
    def rotate_earth_to_grid(*args):
        return rotate_earth_to_grid_np(*args)

    @staticmethod
    def era5_rh_to_water(*args):
        raise AssertionError("no relative humidity reaches this pass")

    @staticmethod
    def prepare_wrf_vertical(*args):
        raise AssertionError("vertical preprocessing is unused here")

    @staticmethod
    def receipt():
        return {"backend": "numpy-reference-test"}


def _grid():
    """A 3 km grid well inside the source crop."""
    return LambertGrid(
        ref_lat=35.5, ref_lon=-97.5, truelat1=35.5, truelat2=35.5,
        stand_lon=-97.5, dx=3000.0, dy=3000.0, e_we=31, e_sn=29)


def _snapshot(landsea, *, skin=None):
    """A skin temperature that differs everywhere, over a given mask."""
    lon2, lat2 = np.meshgrid(_LONGITUDE, _LATITUDE)
    if skin is None:
        skin = 280.0 + 0.8 * (lat2 - 33.0) + 0.3 * (lon2 + 100.0)
    return Era5Snapshot(
        valid_time=datetime(2026, 9, 27, 13),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=_LATITUDE, longitude=_LONGITUDE,
        fields={"LANDSEA": np.asarray(landsea, dtype=np.float64),
                "SKINTEMP": skin})


def _lakes(shape):
    """Land everywhere but two small lakes."""
    land = np.ones(shape, dtype=bool)
    land[10:14, 12:17] = False
    land[20, 5:7] = False
    return land


def _statics(land):
    return WaterTemperatureStatics.for_route(
        route="the mapped test route", policy=None,
        landmask=land.astype(np.float64),
        lu_index=np.where(land, 10, LAKE), landuse_attrs={"ISLAKE": LAKE})


def _chain(snapshot, grid, donors, active):
    """WPS's full chain on ``donors`` at ``active`` targets, as it maps."""
    target_lat, target_lon = grid.latlon_mass()
    return wps_masked_field_interpolate(
        snapshot.fields["SKINTEMP"], snapshot.latitude, snapshot.longitude,
        target_lat, target_lon, source_valid=donors, target_active=active,
        chain=_WPS_FULL_CHAIN, fill_value=np.nan,
        physical_range=(170.0, 400.0))


def test_a_lake_in_a_crop_with_no_water_takes_the_land_skin_there(capsys):
    grid = _grid()
    shape = grid.latlon_mass()[0].shape
    land = _lakes(shape)
    water = ~land
    snapshot = _snapshot(np.ones((_LATITUDE.size, _LONGITUDE.size)))

    met = interpolate_era5_to_lambert(
        snapshot, grid, target_landmask=land,
        water_temperature_statics=_statics(land), backend=_NumpyBackend())

    skin = np.asarray(met.fields["SKINTEMP"], dtype=np.float64)
    expected = _chain(snapshot, grid, np.ones(snapshot.fields["LANDSEA"].shape,
                                              dtype=bool), water)
    np.testing.assert_allclose(skin[water], expected[water], rtol=0,
                               atol=1e-4)
    assert skin[water].min() > 280.0
    repairs = met.masked_field_repairs["SKINTEMP"]
    assert repairs["other_surface"] == int(water.sum()) == 22
    assert repairs["fill"] == 0
    # The lakes prepare on the component skin, now a temperature.
    np.testing.assert_allclose(met.water_temperature[water], skin[water])
    assert (met.water_temperature_source[water]
            == wt.SOURCE_COMPONENT_SKIN).all()
    assert "water_fill" not in met.water_temperature_receipt
    said = capsys.readouterr().err
    assert ("SKINTEMP: 22 value(s) on a surface the source holds no usable "
            "cell of took the source's value on the other surface") in said


def test_an_island_in_a_crop_with_no_land_takes_the_sea_skin_there(capsys):
    grid = _grid()
    shape = grid.latlon_mass()[0].shape
    island = np.zeros(shape, dtype=bool)
    island[14:16, 15:17] = True
    snapshot = _snapshot(np.zeros((_LATITUDE.size, _LONGITUDE.size)))

    met = interpolate_era5_to_lambert(
        snapshot, grid, target_landmask=island, backend=_NumpyBackend())

    skin = np.asarray(met.fields["SKINTEMP"], dtype=np.float64)
    expected = _chain(snapshot, grid, np.ones(snapshot.fields["LANDSEA"].shape,
                                              dtype=bool), island)
    np.testing.assert_allclose(skin[island], expected[island], rtol=0,
                               atol=1e-4)
    assert skin.min() > 280.0
    repairs = met.masked_field_repairs["SKINTEMP"]
    assert repairs["other_surface"] == 4
    assert repairs["fill"] == 0
    assert "took the source's value on the other surface" in (
        capsys.readouterr().err)


def test_a_crop_holding_both_surfaces_maps_as_wps_does():
    """Far water still reaches a lake through the WPS search, unchanged."""
    grid = _grid()
    shape = grid.latlon_mass()[0].shape
    land = _lakes(shape)
    landsea = np.ones((_LATITUDE.size, _LONGITUDE.size))
    landsea[0, 0] = 0.0             # one water cell, far from every lake
    snapshot = _snapshot(landsea)

    met = interpolate_era5_to_lambert(
        snapshot, grid, target_landmask=land, backend=_NumpyBackend())

    skin = np.asarray(met.fields["SKINTEMP"], dtype=np.float64)
    source_land = landsea > 0.5
    np.testing.assert_allclose(
        skin[~land], _chain(snapshot, grid, ~source_land, ~land)[~land],
        rtol=0, atol=1e-4)
    np.testing.assert_array_equal(skin[~land], snapshot.fields["SKINTEMP"][0, 0]
                                  .astype(np.float32))
    np.testing.assert_allclose(
        skin[land], _chain(snapshot, grid, source_land, land)[land],
        rtol=0, atol=1e-4)
    assert met.masked_field_repairs["SKINTEMP"]["other_surface"] == 0


def test_a_source_with_no_skin_temperature_still_keeps_the_fill():
    """Nothing on either surface: nothing is invented, the fill is counted."""
    grid = _grid()
    shape = grid.latlon_mass()[0].shape
    land = _lakes(shape)
    snapshot = _snapshot(np.ones((_LATITUDE.size, _LONGITUDE.size)),
                         skin=np.full((_LATITUDE.size, _LONGITUDE.size),
                                      np.nan))

    met = interpolate_era5_to_lambert(
        snapshot, grid, target_landmask=land, backend=_NumpyBackend())

    np.testing.assert_array_equal(met.fields["SKINTEMP"], 0.0)
    repairs = met.masked_field_repairs["SKINTEMP"]
    assert repairs["other_surface"] == 0
    assert repairs["fill"] == int(land.sum())


def test_the_lake_targets_really_lie_inside_the_crop():
    """The fixture's lakes sit well inside the source, so only the missing
    surface, not the crop's edge, decides what they take."""
    grid = _grid()
    target_lat, target_lon = grid.latlon_mass()
    y, x = _regular_coordinates(_LATITUDE, _LONGITUDE, target_lat, target_lon)
    assert y.min() > 2 and y.max() < _LATITUDE.size - 3
    assert x.min() > 2 and x.max() < _LONGITUDE.size - 3
