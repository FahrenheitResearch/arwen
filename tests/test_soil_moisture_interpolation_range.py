"""Mapped land-surface fields keep to their physical range and fill every land cell.

The fixture is the 1.6 m soil moisture HRRR carried at 2026-09-27 06Z
under three land cells of a 1 km grid: a block of dry land cells near
0.002 among land cells near 0.30, every one of them land.  WPS's
``sixteen_pt`` operator puts those three cells at -0.051, -0.055 and
-0.052, past the 0.05 margin the mapped route used to repair within, and
the preparation stopped with "declarative mapped soil moisture is missing
or outside 0..1 on land".  Beside it sits a reservoir the source calls
water and the target grid calls land (a land-sea mask disagreement), and
a source land cell that carries no soil value at all.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.ingest.horiz import (
    _WPS_FULL_CHAIN,
    interpolate_era5_to_lambert,
    wps_masked_field_interpolate,
)
from gpuwm.ingest.soil import HRRR_SOIL_NODE_DEPTHS_M, preprocess_noah_soil
from gpuwm.ingest.soil_contract import (
    MAPPED_SOIL_MOISTURE,
    MAPPED_SOIL_TEMPERATURE,
)
from gpuwm.static.lambert import LambertGrid
from gpuwm.verify.npref import (
    interpolate_regular_np,
    masked_nearest_np,
    rotate_earth_to_grid_np,
)
from conftest import requires_wps_masked_chain_bridge

#: The tests here map masked fields through the native chain.
pytestmark = requires_wps_masked_chain_bridge

#: HRRR 1.6 m volumetric soil moisture on the 6 x 6 source cells around the
#: three refused cells, rows south to north, all land.
_MEASURED_BLOCK = np.array([
    [0.041, 0.277, 0.284, 0.286, 0.287, 0.035],
    [0.301, 0.293, 0.300, 0.290, 0.285, 0.035],
    [0.304, 0.314, 0.007, 0.019, 0.304, 0.289],
    [0.309, 0.302, 0.002, 0.002, 0.002, 0.015],
    [0.019, 0.270, 0.288, 0.271, 0.262, 0.267],
    [0.303, 0.299, 0.293, 0.274, 0.264, 0.254],
])
#: Where the three refused 1 km cells sit inside source cell (2, 2) of the
#: block, as (row, column) fractions of a source cell.
_REFUSED_OFFSETS = ((0.279, 0.570), (0.617, 0.228), (0.619, 0.568))
_DEEP_NODE = 7          # 1.6 m in HRRR's nine soil nodes

#: A 3 km-like regular source, large enough to hold the measured block and
#: a reservoir under one 61 km grid.
_LAT0, _LON0, _STEP = 32.0, -98.0, 0.03
_LATITUDE = _LAT0 + _STEP * np.arange(30, dtype=np.float64)
_LONGITUDE = _LON0 + _STEP * np.arange(30, dtype=np.float64)
_BLOCK_ROW, _BLOCK_COLUMN = 12, 12
#: Source water the target calls land: seven source cells square, so the
#: target land in its middle has no source land within two source cells
#: and only the WPS search can answer it.
_RESERVOIR = (slice(16, 23), slice(5, 12))
#: A source land cell whose soil tiling left it without a value.
_SOIL_GAP = (9, 20)


def _source_moisture():
    """Nine soil nodes at 0.30, the measured block in the 1.6 m node."""
    moisture = np.full(
        (HRRR_SOIL_NODE_DEPTHS_M.size, _LATITUDE.size, _LONGITUDE.size),
        0.30, dtype=np.float64)
    moisture[_DEEP_NODE, _BLOCK_ROW:_BLOCK_ROW + 6,
             _BLOCK_COLUMN:_BLOCK_COLUMN + 6] = _MEASURED_BLOCK
    return moisture


def _refused_targets():
    row = _BLOCK_ROW + 2
    column = _BLOCK_COLUMN + 2
    latitude = np.array([[_LAT0 + _STEP * (row + dy)]
                         for dy, _ in _REFUSED_OFFSETS])
    longitude = np.array([[_LON0 + _STEP * (column + dx)]
                          for _, dx in _REFUSED_OFFSETS])
    return latitude, longitude


def test_sixteen_pt_takes_the_measured_cells_below_zero_without_a_range():
    """The fixture reproduces the refusal: WPS's own values, unchanged."""
    target_lat, target_lon = _refused_targets()
    field = _source_moisture()[_DEEP_NODE]
    land = np.ones(field.shape, dtype=bool)
    wps = wps_masked_field_interpolate(
        field, _LATITUDE, _LONGITUDE, target_lat, target_lon,
        source_valid=land, target_active=np.ones(target_lat.shape, bool),
        chain=_WPS_FULL_CHAIN, fill_value=1.0)
    np.testing.assert_allclose(
        wps[:, 0], [-0.0513, -0.0545, -0.0525], atol=5.0e-4)


def test_soil_moisture_range_hands_the_cells_a_mean_of_their_source_land():
    target_lat, target_lon = _refused_targets()
    field = _source_moisture()[_DEEP_NODE]
    land = np.ones(field.shape, dtype=bool)
    active = np.ones(target_lat.shape, bool)
    tally = {}
    bounded = wps_masked_field_interpolate(
        field, _LATITUDE, _LONGITUDE, target_lat, target_lon,
        source_valid=land, target_active=active,
        chain=_WPS_FULL_CHAIN, fill_value=1.0, physical_range=(0.0, 1.0),
        tally=tally)
    four_pt = wps_masked_field_interpolate(
        field, _LATITUDE, _LONGITUDE, target_lat, target_lon,
        source_valid=land, target_active=active,
        chain=("four_pt",), fill_value=np.nan)
    # Each cell takes the bilinear mean of the four dry source cells
    # around it, all of them land, and stays inside what they span.
    np.testing.assert_array_equal(bounded, four_pt)
    corners = _MEASURED_BLOCK[2:4, 2:4]
    assert np.all(bounded >= corners.min())
    assert np.all(bounded <= corners.max())
    assert tally["sixteen_pt_outside_range"] == 3
    assert tally["fill"] == 0


def test_soil_moisture_range_leaves_every_in_range_value_byte_identical():
    """Only the cells WPS takes out of 0..1 move."""
    grid = _one_km_grid()
    mass_lat, mass_lon = grid.latlon_mass()
    field = _source_moisture()[_DEEP_NODE]
    land = np.ones(field.shape, dtype=bool)
    active = np.ones(mass_lat.shape, bool)
    wps = wps_masked_field_interpolate(
        field, _LATITUDE, _LONGITUDE, mass_lat, mass_lon,
        source_valid=land, target_active=active,
        chain=_WPS_FULL_CHAIN, fill_value=1.0)
    bounded = wps_masked_field_interpolate(
        field, _LATITUDE, _LONGITUDE, mass_lat, mass_lon,
        source_valid=land, target_active=active,
        chain=_WPS_FULL_CHAIN, fill_value=1.0, physical_range=(0.0, 1.0))
    outside = (wps < 0.0) | (wps > 1.0)
    assert outside.any(), "the fixture must reach the overshoot"
    np.testing.assert_array_equal(bounded[~outside], wps[~outside])
    assert np.all((bounded >= 0.0) & (bounded <= 1.0))


def test_a_source_fill_value_is_not_a_donor():
    """A numeric fill in the source is a missing value, not data.

    Without the range the bilinear stencil carries -999 straight into the
    target; with it the cell is not a donor and the chain answers from
    the land around it.
    """
    target_lat, target_lon = _refused_targets()
    field = _source_moisture()[_DEEP_NODE]
    field[_BLOCK_ROW + 3, _BLOCK_COLUMN + 2] = -999.0
    land = np.ones(field.shape, dtype=bool)
    active = np.ones(target_lat.shape, bool)
    unbounded = wps_masked_field_interpolate(
        field, _LATITUDE, _LONGITUDE, target_lat, target_lon,
        source_valid=land, target_active=active,
        chain=_WPS_FULL_CHAIN, fill_value=1.0)
    assert unbounded.min() < -1.0
    tally = {}
    bounded = wps_masked_field_interpolate(
        field, _LATITUDE, _LONGITUDE, target_lat, target_lon,
        source_valid=land, target_active=active,
        chain=_WPS_FULL_CHAIN, fill_value=1.0, physical_range=(0.0, 1.0),
        tally=tally)
    assert np.all((bounded >= 0.0) & (bounded <= 0.314))
    assert tally["source_outside_range"] == 1


def test_packing_roundoff_at_a_bound_stays_a_donor_unchanged():
    """ERA5's measured -9.52e-4 soil moisture is roundoff, not a fill."""
    target_lat, target_lon = _refused_targets()
    field = _source_moisture()[_DEEP_NODE]
    field[_BLOCK_ROW + 3, _BLOCK_COLUMN + 2] = -9.52e-4
    land = np.ones(field.shape, dtype=bool)
    active = np.ones(target_lat.shape, bool)
    tally = {}
    bounded = wps_masked_field_interpolate(
        field, _LATITUDE, _LONGITUDE, target_lat, target_lon,
        source_valid=land, target_active=active, chain=("four_pt",),
        fill_value=np.nan, physical_range=(0.0, 1.0), tally=tally)
    plain = wps_masked_field_interpolate(
        field, _LATITUDE, _LONGITUDE, target_lat, target_lon,
        source_valid=land, target_active=active, chain=("four_pt",),
        fill_value=np.nan)
    np.testing.assert_array_equal(bounded, plain)
    assert tally["source_outside_range"] == 0


def test_packing_roundoff_at_a_bound_goes_on_the_bound_not_called_overshoot():
    """A saturated ice sheet the source stores at 1.0003.

    GEFS carries its Antarctic soil moisture that way.  Every answer is
    the donors' own value, a little past 1: not the parabola's overshoot,
    so it neither falls through to four_pt nor is counted as such, and it
    goes on the bound.
    """
    target_lat, target_lon = _refused_targets()
    field = np.full((_LATITUDE.size, _LONGITUDE.size), 1.0003)
    land = np.ones(field.shape, dtype=bool)
    tally = {}
    bounded = wps_masked_field_interpolate(
        field, _LATITUDE, _LONGITUDE, target_lat, target_lon,
        source_valid=land, target_active=np.ones(target_lat.shape, bool),
        chain=_WPS_FULL_CHAIN, fill_value=1.0, physical_range=(0.0, 1.0),
        tally=tally)
    np.testing.assert_array_equal(bounded, 1.0)
    assert tally["sixteen_pt_outside_range"] == 0
    assert tally["source_roundoff_at_bound"] == 3
    assert tally["source_outside_range"] == 0


def test_physical_range_must_be_ordered():
    field = _source_moisture()[_DEEP_NODE]
    target_lat, target_lon = _refused_targets()
    with pytest.raises(ValueError, match="low < high"):
        wps_masked_field_interpolate(
            field, _LATITUDE, _LONGITUDE, target_lat, target_lon,
            source_valid=np.ones(field.shape, bool),
            target_active=np.ones(target_lat.shape, bool),
            chain=_WPS_FULL_CHAIN, fill_value=1.0, physical_range=(1.0, 0.0))


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


def _one_km_grid():
    """A 1 km grid over the dry block and the reservoir."""
    centre_lat = _LAT0 + _STEP * (_BLOCK_ROW + 2.5)
    centre_lon = _LON0 + _STEP * (_BLOCK_COLUMN + 2.5)
    return LambertGrid(
        ref_lat=centre_lat, ref_lon=centre_lon, truelat1=centre_lat,
        truelat2=centre_lat, stand_lon=centre_lon, dx=1000.0, dy=1000.0,
        e_we=61, e_sn=61)


def _node_contract():
    def selector(name, depth):
        return {
            "format": "grib2", "discipline": 2,
            "category": 3 if name == "soil_temperature" else 0,
            "parameter": 18 if name == "soil_temperature" else 192,
            "center": 7, "subcenter": 0, "master_table_version": 2,
            "local_table_version": 1, "level_type": 106,
            "level_value": depth, "second_level_type": 106,
            "second_level_value": depth,
        }

    return {
        "temperature_field": "soil_temperature",
        "moisture_field": "volumetric_soil_moisture",
        "depth_units": "m",
        "source_nodes": [
            {"depth": float(depth), "selectors": {
                "soil_temperature": selector("soil_temperature", float(depth)),
                "volumetric_soil_moisture": selector(
                    "volumetric_soil_moisture", float(depth)),
            }}
            for depth in HRRR_SOIL_NODE_DEPTHS_M
        ],
        "target_layers": [
            {"top": 0.0, "bottom": 0.1}, {"top": 0.1, "bottom": 0.4},
            {"top": 0.4, "bottom": 1.0}, {"top": 1.0, "bottom": 2.0},
        ],
        "remap": {
            "kind": "linear_node_samples",
            "source_value_location": "level_node",
            "target_value_location": "layer_midpoint",
        },
        "missing": {
            "land": "reject",
            "ocean": {
                "stage": "after_horizontal_interpolation",
                "temperature": "skin_temperature",
                "moisture": 1.0,
            },
        },
    }


def _edge_and_reservoir_snapshot(moisture=None):
    """The sharp wet edge, the reservoir and the soil gap, one snapshot."""
    shape = (_LATITUDE.size, _LONGITUDE.size)
    moisture = _source_moisture() if moisture is None else moisture
    temperature = np.full(moisture.shape, 293.0)
    landsea = np.ones(shape)
    landsea[_RESERVOIR] = 0.0
    # Over its water the source carries what HRRR does: saturated soil at
    # the water's temperature, which no land target may take.
    moisture[(slice(None),) + _RESERVOIR] = 1.0
    temperature[(slice(None),) + _RESERVOIR] = 288.0
    moisture[(slice(None),) + _SOIL_GAP] = np.nan
    temperature[(slice(None),) + _SOIL_GAP] = np.nan
    return Era5Snapshot(
        valid_time=datetime(2026, 9, 27, 6),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=_LATITUDE, longitude=_LONGITUDE,
        fields={
            "LANDSEA": landsea,
            "SKINTEMP": np.full(shape, 295.0),
            MAPPED_SOIL_TEMPERATURE: temperature,
            MAPPED_SOIL_MOISTURE: moisture,
        })


def test_mapped_soil_over_a_sharp_edge_and_a_mask_disagreement_prepares(
        capsys):
    """The mapped route end to end: interpolate, then the Noah soil state.

    Refused before: the dry block's overshoot was past the margin.  Now
    every land cell is physical (sixteen_pt still overshoots the donors
    INSIDE 0..1, as WPS does), the reservoir's target land takes source
    LAND values through the WPS search (never the source water's 1.0),
    and the receipt counts both.
    """
    grid = _one_km_grid()
    target_land = np.ones(grid.latlon_mass()[0].shape, dtype=bool)
    met = interpolate_era5_to_lambert(
        _edge_and_reservoir_snapshot(), grid, target_landmask=target_land,
        backend=_NumpyBackend())
    fields = dict(met.fields)
    fields["TMN"] = np.full(target_land.shape, 290.0)
    state = preprocess_noah_soil(
        fields, soil_type=np.full(target_land.shape, 6),
        soil_layer_contract=_node_contract(),
        landmask=target_land.astype(np.float64))

    mapped = np.asarray(met.fields[MAPPED_SOIL_MOISTURE], dtype=np.float64)
    assert mapped.min() >= 0.0 and mapped.max() < 0.5
    assert np.isfinite(state.soil_moisture).all()
    assert state.soil_moisture.min() > 0.0
    assert state.soil_moisture.max() <= 1.0
    soil_temperature = np.asarray(
        met.fields[MAPPED_SOIL_TEMPERATURE], dtype=np.float64)
    np.testing.assert_array_equal(soil_temperature, 293.0)

    repairs = met.masked_field_repairs[MAPPED_SOIL_MOISTURE]
    assert repairs["sixteen_pt_outside_range"] > 0
    assert repairs["search"] > 0
    assert repairs["fill"] == 0
    assert met.masked_field_repairs[MAPPED_SOIL_TEMPERATURE]["search"] \
        == repairs["search"]
    said = capsys.readouterr().err
    assert "took four_pt's weighted mean" in said
    assert "land-sea masks disagree there" in said


def _dry_patch_in_every_node():
    """The measured dry block, in every one of the nine soil nodes."""
    moisture = _source_moisture()
    moisture[:, _BLOCK_ROW:_BLOCK_ROW + 6,
             _BLOCK_COLUMN:_BLOCK_COLUMN + 6] = _MEASURED_BLOCK
    return moisture


def test_soil_moisture_in_percent_refuses_though_its_driest_land_is_in_range():
    """Percent read as a fraction, with a dry patch in every layer.

    The patch's 0.2% and 0.7% cells lie inside 0..1 read as fractions, so
    a source judged only on whether ANY land value is in range passes, and
    those few cells become the only donors the WPS search spreads over the
    domain's land.  Nearly every land value is outside the range, and the
    source is refused for the unit.
    """
    grid = _one_km_grid()
    target_land = np.ones(grid.latlon_mass()[0].shape, dtype=bool)
    snapshot = _edge_and_reservoir_snapshot(
        100.0 * _dry_patch_in_every_node())
    # The patch cells drier than 1%: 0.7% and three at 0.2%.
    inside = int(np.count_nonzero(100.0 * _MEASURED_BLOCK <= 1.01))
    assert inside == 4
    with pytest.raises(ValueError, match=(
            rf"only {inside} of the \d+ soil moisture value\(s\) the source "
            r"carries on its land in source layer 1 lie inside 0..1")) as (
                refusal):
        interpolate_era5_to_lambert(
            snapshot, grid, target_landmask=target_land,
            backend=_NumpyBackend())
    assert "not in the unit its name states" in str(refusal.value)


def test_soil_temperature_in_celsius_refuses_by_name():
    grid = _one_km_grid()
    target_land = np.ones(grid.latlon_mass()[0].shape, dtype=bool)
    snapshot = _edge_and_reservoir_snapshot()
    celsius = snapshot.fields[MAPPED_SOIL_TEMPERATURE] - 273.15
    fields = dict(snapshot.fields)
    fields[MAPPED_SOIL_TEMPERATURE] = celsius
    with pytest.raises(ValueError, match=(
            r"only 0 of the \d+ soil temperature value\(s\) the source "
            r"carries on its land in source layer 1 lie inside 170..400")):
        interpolate_era5_to_lambert(
            replace(snapshot, fields=fields), grid,
            target_landmask=target_land, backend=_NumpyBackend())


def test_a_source_with_no_soil_moisture_on_any_land_cell_refuses_by_name():
    grid = _one_km_grid()
    target_land = np.ones(grid.latlon_mass()[0].shape, dtype=bool)
    moisture = np.full(_source_moisture().shape, np.nan)
    with pytest.raises(ValueError, match=(
            "carries no soil moisture on any of its")):
        interpolate_era5_to_lambert(
            _edge_and_reservoir_snapshot(moisture), grid,
            target_landmask=target_land, backend=_NumpyBackend())


def _part_land_only_snapshot(soil_temperature_on_part_land=np.nan):
    """A window of small islands from a source that keeps soil only on its land.

    No source cell is half land; a few are part land (0.1 to 0.4, the
    fractions a remapped native mesh leaves around small islands), and the
    soil fields are missing everywhere except, when asked, on those
    part-land cells.
    """
    shape = (_LATITUDE.size, _LONGITUDE.size)
    landsea = np.zeros(shape)
    part_land = (slice(13, 16), slice(13, 16))
    landsea[part_land] = 0.3
    temperature = np.full((len(HRRR_SOIL_NODE_DEPTHS_M),) + shape, np.nan)
    moisture = np.full(temperature.shape, np.nan)
    temperature[(slice(None),) + part_land] = soil_temperature_on_part_land
    return Era5Snapshot(
        valid_time=datetime(2026, 9, 27, 18),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=_LATITUDE, longitude=_LONGITUDE,
        fields={
            "LANDSEA": landsea,
            "SKINTEMP": np.full(shape, 300.0),
            MAPPED_SOIL_TEMPERATURE: temperature,
            MAPPED_SOIL_MOISTURE: moisture,
        })


def test_a_window_whose_only_land_is_part_land_without_soil_takes_the_fill():
    """Small islands the source never calls land are not a missing field.

    Refused before as "the source carries no soil temperature on any of
    its 9 land cell(s)"; the target's island land now takes the path
    every island the source cannot resolve takes: it holds METGRID.TBL
    fill_missing here, is counted as ``no_source_land``, and the soil
    initializer builds its column from that mark.
    """
    grid = _one_km_grid()
    target_land = np.zeros(grid.latlon_mass()[0].shape, dtype=bool)
    target_land[28:33, 28:33] = True
    met = interpolate_era5_to_lambert(
        _part_land_only_snapshot(), grid, target_landmask=target_land,
        backend=_NumpyBackend())
    temperature = np.asarray(met.fields[MAPPED_SOIL_TEMPERATURE])
    moisture = np.asarray(met.fields[MAPPED_SOIL_MOISTURE])
    np.testing.assert_array_equal(temperature[:, target_land], 285.0)
    np.testing.assert_array_equal(moisture[:, target_land], 1.0)
    assert met.masked_field_repairs[MAPPED_SOIL_TEMPERATURE]["no_source_land"] > 0


def test_part_land_values_in_the_wrong_unit_are_still_refused():
    grid = _one_km_grid()
    target_land = np.zeros(grid.latlon_mass()[0].shape, dtype=bool)
    target_land[28:33, 28:33] = True
    with pytest.raises(ValueError, match=(
            r"only 0 of the \d+ soil temperature value\(s\) the source "
            r"carries on its land in source layer 1 lie inside 170..400")):
        interpolate_era5_to_lambert(
            _part_land_only_snapshot(soil_temperature_on_part_land=21.0),
            grid, target_landmask=target_land, backend=_NumpyBackend())


def test_fill_values_on_a_block_of_land_are_said_as_fill_not_as_a_mask(
        capsys):
    """Land whose donors were fill values is not a land-sea mask story.

    A 7 x 7 block of source land carries a numeric fill in the 1.6 m node.
    It is a sliver of the source's land, so the field is in its unit and
    prepares; the target land in the block's middle has no usable source
    value within two source cells and takes the WPS search's nearest one,
    and the receipt says it was the values, not the masks.
    """
    grid = _one_km_grid()
    target_land = np.ones(grid.latlon_mass()[0].shape, dtype=bool)
    plain = interpolate_era5_to_lambert(
        _edge_and_reservoir_snapshot(), grid, target_landmask=target_land,
        backend=_NumpyBackend())
    capsys.readouterr()
    moisture = _source_moisture()
    moisture[_DEEP_NODE, 3:10, 18:25] = 9999.0
    filled = interpolate_era5_to_lambert(
        _edge_and_reservoir_snapshot(moisture), grid,
        target_landmask=target_land, backend=_NumpyBackend())
    mapped = np.asarray(filled.fields[MAPPED_SOIL_MOISTURE])
    assert mapped.min() >= 0.0 and mapped.max() <= 1.0
    base = plain.masked_field_repairs[MAPPED_SOIL_MOISTURE]
    repairs = filled.masked_field_repairs[MAPPED_SOIL_MOISTURE]
    assert base["search_past_unusable"] == 0
    assert repairs["search_past_unusable"] > 0
    # The reservoir's mask disagreement is counted as before, apart.
    assert repairs["search"] == base["search"]
    said = capsys.readouterr().err
    assert ("were all missing a value or outside 0..1 took the nearest "
            "usable one") in said
