"""Inland water the source does not resolve takes the 2 m air temperature.

Metgrid's search reaches the whole source array, so a basin the source
holds as land took the nearest source water of any other body: the
Charles River basin on a 250 m Boston grid initialized from coastal HRRR
water 4.5 to 6 km away.  The skin temperature's water pass now refuses
source water past ``INLAND_WATER_SOURCE_REACH_M`` for inland water, and
such a target takes WRF's answer for a lake the analysis does not
resolve (real.exe ``use_tavg_for_tsk``): the daily-mean 2 m air
temperature.  Open water touching the domain edge keeps metgrid's search.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from gpuwm.ingest import horiz
from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.horiz import (
    INLAND_WATER_SOURCE_REACH_M,
    DailyMeanAirTemperature,
    inland_water_targets,
    interpolate_era5_to_lambert,
    with_inland_air_temperature,
)
from gpuwm.ingest.water_temperature import WaterTemperatureStatics
from gpuwm.verify import wps_masked_oracle as oracle
from conftest import requires_wps_masked_chain_bridge
from test_skin_temperature_other_surface import (
    LAKE,
    _LATITUDE,
    _LONGITUDE,
    _NumpyBackend,
    _grid,
)
from test_wps_masked_chain_native import (
    _axes,
    _field,
    _land,
    _native,
    _same,
    _targets,
)

pytestmark = requires_wps_masked_chain_bridge

START = datetime(2026, 10, 2, 18)


@pytest.mark.parametrize("cells", [0.4, 1.5, 4.0, np.inf])
def test_the_reach_matches_the_oracle_byte_for_byte(cells):
    native = _native()
    rng = np.random.default_rng(int(7 + 10 * min(cells, 9.0)))
    ny, nx = 23, 31
    lat, lon = _axes(ny, nx)
    field = _field(rng, ny, nx, (170.0, 400.0))
    land, partial = _land(rng, ny, nx)
    ty, tx = _targets(rng, ny, nx, 260)
    target_land = rng.uniform(size=ty.shape) < 0.55
    limited = rng.uniform(size=ty.shape) < 0.6
    beyond = 0
    for donors_land, donors_partial in (
            (land, partial), (np.ones_like(land), partial),
            (np.zeros_like(land), np.zeros_like(land))):
        tallies = ({}, {})
        got, got_recovered = horiz._skin_temperature_on_both_surfaces(
            field, lat, lon, ty, tx, land_donors=donors_land,
            partial_land_donors=donors_partial, target_land=target_land,
            fill_value=0.0, physical_range=(170.0, 400.0), tally=tallies[0],
            native=native, reach=(limited, cells))
        want, want_recovered = oracle._skin_temperature_on_both_surfaces(
            field, lat, lon, ty, tx, land_donors=donors_land,
            partial_land_donors=donors_partial, target_land=target_land,
            fill_value=0.0, physical_range=(170.0, 400.0), tally=tallies[1],
            reach=(limited, cells))
        _same(got, want)
        assert got_recovered == want_recovered
        assert list(tallies[0].items()) == list(tallies[1].items())
        # A target the reach left open is NaN, inland water, and nothing else.
        assert not np.any(np.isnan(got) & ~(limited & ~target_land))
        beyond += tallies[0]["beyond_reach"]
    if np.isinf(cells):
        assert beyond == 0
    elif cells < 1.0:
        assert beyond > 0


def test_no_reach_is_the_original_chain():
    native = _native()
    rng = np.random.default_rng(3)
    ny, nx = 23, 31
    lat, lon = _axes(ny, nx)
    field = _field(rng, ny, nx, (170.0, 400.0))
    land, partial = _land(rng, ny, nx)
    ty, tx = _targets(rng, ny, nx, 200)
    target_land = rng.uniform(size=ty.shape) < 0.5
    kwargs = dict(land_donors=land, partial_land_donors=partial,
                  target_land=target_land, fill_value=0.0,
                  physical_range=(170.0, 400.0), native=native)
    plain, _ = horiz._skin_temperature_on_both_surfaces(
        field, lat, lon, ty, tx, **kwargs)
    unlimited, _ = horiz._skin_temperature_on_both_surfaces(
        field, lat, lon, ty, tx, reach=(np.ones_like(target_land), np.inf),
        **kwargs)
    _same(plain, unlimited)


def test_inland_water_is_water_no_domain_edge_reaches():
    land = np.ones((9, 12), dtype=bool)
    land[:, :2] = False          # open water on the west edge
    land[3:5, 2] = False         # an inlet joined to it (8-connected)
    land[4:6, 6:8] = False       # an enclosed basin
    land[8, 10] = False          # water on the south edge
    lake = np.zeros_like(land)
    lake[0, 0] = True            # a lake cell on the edge stays inland
    inland = inland_water_targets(land)
    assert inland.sum() == 4 and inland[4:6, 6:8].all()
    inland = inland_water_targets(land, lake)
    assert inland.sum() == 5 and inland[0, 0]
    assert not inland_water_targets(np.ones((5, 5), dtype=bool)).any()


def test_the_reach_is_metres_on_the_source_cells():
    snapshot = _snapshot(np.ones((_LATITUDE.size, _LONGITUDE.size)),
                         START, 290.0)
    spacing = horiz.source_cell_spacing_m(snapshot, np.full((2, 2), 35.0))
    # 0.25 degree: the latitude spacing is the coarser one.
    assert spacing == pytest.approx(0.25 * np.pi / 180.0 * 6370000.0)

    class Projected:
        latitude = np.arange(5, dtype=np.float64) * 3.0
        longitude = np.arange(6, dtype=np.float64) * 3.0
        projection = {"family": "lambert",
                      "parameters": {"axis_unit_m": 1000.0}}

    assert horiz.source_cell_spacing_m(Projected, np.zeros(1)) == 3000.0
    assert INLAND_WATER_SOURCE_REACH_M == 3000.0


def _snapshot(landsea, valid_time, air, *, skin_land=305.0, skin_sea=288.0):
    shape = (_LATITUDE.size, _LONGITUDE.size)
    return Era5Snapshot(
        valid_time=valid_time,
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=_LATITUDE, longitude=_LONGITUDE,
        fields={"LANDSEA": np.asarray(landsea, dtype=np.float64),
                "SKINTEMP": np.where(np.asarray(landsea) > 0.5,
                                     skin_land, skin_sea),
                "T2": np.full(shape, float(air))})


def _case():
    """A 3 km grid: two inland lakes, open water on its west edge, and
    source water only on the source's east columns, far from both."""
    grid = _grid()
    shape = grid.latlon_mass()[0].shape
    land = np.ones(shape, dtype=bool)
    land[10:14, 12:17] = False
    land[20, 5:7] = False
    land[:, :2] = False
    lu = np.where(land, 10, LAKE)
    lu[:, :2] = 17
    statics = WaterTemperatureStatics.for_route(
        route="the inland reach test", policy=None,
        landmask=land.astype(np.float64), lu_index=lu,
        landuse_attrs={"ISLAKE": LAKE})
    landsea = np.ones((_LATITUDE.size, _LONGITUDE.size))
    landsea[:, 17:] = 0.0
    lakes = ~land
    lakes[:, :2] = False
    return grid, land, lakes, statics, landsea


def test_an_unresolved_basin_takes_the_air_temperature_not_far_water(capsys):
    grid, land, lakes, statics, landsea = _case()
    snapshot = _snapshot(landsea, START, 291.5)
    met = interpolate_era5_to_lambert(
        snapshot, grid, target_landmask=land,
        water_temperature_statics=statics, backend=_NumpyBackend())
    skin = np.asarray(met.fields["SKINTEMP"], dtype=np.float64)
    # The lakes: this time's 2 m air temperature, not the sea 100 km east.
    assert np.all(skin[lakes] == np.float32(291.5))
    # Open water on the domain edge keeps metgrid's whole-array search.
    edge = ~land & ~lakes
    assert np.all(skin[edge] == np.float32(288.0))
    repairs = met.masked_field_repairs["SKINTEMP"]
    assert repairs["inland_air_temperature"] == int(lakes.sum()) == 22
    assert repairs["other_surface"] == 0
    np.testing.assert_array_equal(met.water_temperature[lakes], skin[lakes])
    said = capsys.readouterr().err
    assert "use_tavg_for_tsk" in said
    assert "this forcing time only" in said


def test_the_route_supplies_the_daily_mean_over_the_first_day(capsys):
    grid, land, lakes, statics, landsea = _case()
    series = tuple(
        _snapshot(landsea, START + timedelta(hours=hour), air)
        for hour, air in ((0, 280.0), (6, 290.0), (12, 300.0), (18, 290.0),
                          (24, 400.0)))
    declared = with_inland_air_temperature(statics, series)
    assert isinstance(declared.inland_air_temperature, DailyMeanAirTemperature)
    # One provider per series, shared by every domain mapped from it.
    again = with_inland_air_temperature(statics, series)
    assert again.inland_air_temperature is declared.inland_air_temperature
    met = interpolate_era5_to_lambert(
        series[2], grid, target_landmask=land,
        water_temperature_statics=declared, backend=_NumpyBackend())
    skin = np.asarray(met.fields["SKINTEMP"], dtype=np.float64)
    assert np.all(skin[lakes] == np.float32(290.0))
    said = capsys.readouterr().err
    assert "daily mean of 4 forcing time(s) over 18 h" in said
    assert with_inland_air_temperature(None, series) is None


def test_a_resolved_lake_keeps_its_own_source_water():
    grid, land, lakes, statics, landsea = _case()
    # The source now holds water under the big lake.
    target_lat, target_lon = grid.latlon_mass()
    rows, columns = np.nonzero(lakes)
    source_landsea = landsea.copy()
    for row, column in zip(rows, columns):
        j = int(np.argmin(abs(_LATITUDE - target_lat[row, column])))
        i = int(np.argmin(abs(_LONGITUDE - target_lon[row, column])))
        source_landsea[j, i] = 0.0
    snapshot = _snapshot(source_landsea, START, 291.5)
    met = interpolate_era5_to_lambert(
        snapshot, grid, target_landmask=land,
        water_temperature_statics=statics, backend=_NumpyBackend())
    skin = np.asarray(met.fields["SKINTEMP"], dtype=np.float64)
    assert np.all(skin[lakes] == np.float32(288.0))
    assert met.masked_field_repairs["SKINTEMP"].get(
        "inland_air_temperature", 0) == 0
