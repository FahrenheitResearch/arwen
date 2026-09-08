"""Explicit lake water is a source state, never mixed land skin or sea ice."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest.lake_temperature import (
    ICE_DEPTH_NEGATIVE_ZERO_M, map_ice_free_lake_water, source_lake_fields,
)
from gpuwm.ingest.water_temperature import (
    SOURCE_ANALYSIS, SOURCE_LAKE_WATER, assemble_water_temperature,
)


def lake_source(depth=0.0):
    y, x = np.mgrid[:3, :3]
    return SimpleNamespace(
        valid_time=datetime(2013, 6, 1), latitude=np.arange(3.0),
        longitude=np.arange(3.0), fields={
            "LAKE_WATER_TEMP": 280.0 + 2.0 * y + 3.0 * x,
            "LAKE_ICE_TEMP": np.full((3, 3), 260.0),
            "LAKE_ICE_DEPTH": np.full((3, 3), depth),
            "LANDSEA": np.ones((3, 3)),
            "SKINTEMP": np.full((3, 3), 350.0),
        })


def at_middle(source):
    return map_ice_free_lake_water(
        source, np.array([[0.25]]), np.array([[0.75]]), np.ones((1, 1), bool))


def test_local_lake_model_plane_ignores_majority_land_and_mixed_skin():
    source = lake_source()
    lat = np.array([[0.25, 2.0], [1.75, 1.0]])
    lon = np.array([[0.75, 2.0], [0.5, 1.0]])
    active = np.array([[True, True], [True, False]])
    result = map_ice_free_lake_water(source, lat, lon, active)
    np.testing.assert_array_equal(result.values[active],
                                  (280 + 2 * lat + 3 * lon)[active])
    assert np.isnan(result.values[~active]).all()
    assert result.receipt["negative_zero_depth_donors"] == 0
    assert len(result.receipt["source_fields_sha256"]) == 3
    np.testing.assert_array_equal(source.fields["SKINTEMP"], 350.0)


@pytest.mark.parametrize("depth", [0.0, -6.77626680920867e-21,
                                   -1.0842026894733873e-19])
def test_actual_cds_ice_free_encodings_preserve_raw_depth_and_temperature(depth):
    source = lake_source(depth)
    original = source.fields["LAKE_ICE_DEPTH"].copy()
    result = at_middle(source)
    assert result.values[0, 0] == 282.75
    assert result.receipt["negative_zero_depth_donors"] == (4 if depth < 0 else 0)
    assert result.receipt["negative_zero_depth_tolerance_m"] == ICE_DEPTH_NEGATIVE_ZERO_M
    np.testing.assert_array_equal(source.fields["LAKE_ICE_DEPTH"], original)


@pytest.mark.parametrize("depth, match", [
    (-1e-12, "unknown or negative"), (np.nan, "unknown or negative"),
    (np.inf, "unknown or negative"),
    (np.nextafter(0.0, 1.0), "frozen or partially frozen"),
    (1e-19, "frozen or partially frozen"), (0.05, "freshwater-ice"),
    (1.0, "freshwater-ice"),
])
def test_invalid_unknown_and_every_positive_ice_depth_refuses(depth, match):
    with pytest.raises(ValueError, match=match):
        at_middle(lake_source(depth))


def test_zero_weight_ice_and_missing_neighbours_do_not_change_exact_point():
    source = lake_source(1.0)
    source.fields["LAKE_ICE_DEPTH"][2, 2] = 0.0
    source.fields["LAKE_WATER_TEMP"][:2] = np.nan
    result = map_ice_free_lake_water(source, np.array([[2.0]]),
                                   np.array([[2.0]]), np.ones((1, 1), bool))
    assert result.values[0, 0] == 290.0


@pytest.mark.parametrize("temperature", [np.nan, 0.0, 169.999, 400.001])
def test_a_bad_positive_weight_temperature_is_never_filled_from_skin(temperature):
    source = lake_source()
    source.fields["LAKE_WATER_TEMP"][0, 0] = temperature
    with pytest.raises(ValueError, match="lack admissible source lake water"):
        at_middle(source)


def test_absent_provider_is_identity_but_partial_contract_refuses():
    assert source_lake_fields({"SKINTEMP": np.ones((2, 2))}) is None
    with pytest.raises(ValueError, match="provider is incomplete"):
        source_lake_fields({"LAKE_WATER_TEMP": np.ones((2, 2))})
    fields = lake_source().fields
    fields["LAKE_ICE_TEMP"] = np.ones((3, 2))
    with pytest.raises(ValueError, match="share one 2-D source grid"):
        source_lake_fields(fields)


def test_lake_provider_is_component_wide_and_does_not_change_land_or_ocean():
    land = np.array([[False, False, True], [True, True, False]])
    lake = np.array([[True, True, False], [False, False, False]])
    skin = np.array([[0., 0., 305.], [302., 303., 290.]])
    mapped = np.array([[280., 282., np.nan], [np.nan, np.nan, np.nan]])
    values, provider, receipt = assemble_water_temperature(
        mapped_sst=None, mapped_skin=skin, target_land=land,
        target_lake=lake, mapped_lake_water=mapped)
    np.testing.assert_array_equal(values[lake], [280., 282.])
    np.testing.assert_array_equal(values[~lake], skin[~lake])
    assert (provider[lake] == SOURCE_LAKE_WATER).all()
    assert receipt["components_on_lake_water"] == 1
    mapped[0, 1] = np.nan
    with pytest.raises(ValueError, match="no admissible water temperature"):
        assemble_water_temperature(mapped_sst=None, mapped_skin=skin,
            target_land=land, target_lake=lake, mapped_lake_water=mapped)


def test_existing_same_component_sst_keeps_precedence_over_optional_lake_state():
    land = np.zeros((3, 3), bool)
    y, x = np.mgrid[:3, :3]
    values, provider, _ = assemble_water_temperature(
        mapped_sst=np.full((3, 3), 0.0), mapped_skin=np.full((3, 3), 295.0),
        target_land=land, target_lake=~land,
        source_sst=np.full((3, 3), 280.0), source_lat=np.arange(3.0),
        source_lon=np.arange(3.0), target_lat=y, target_lon=x,
        mapped_lake_water=np.full((3, 3), 300.0))
    np.testing.assert_array_equal(values, 280.0)
    assert (provider == SOURCE_ANALYSIS).all()
