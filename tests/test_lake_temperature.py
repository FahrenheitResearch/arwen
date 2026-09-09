"""Explicit lake water is a source state, never mixed land skin or sea ice."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest.lake_temperature import (
    ICE_DEPTH_NEGATIVE_ZERO_M, map_ice_free_lake_water, source_lake_fields,
)
from gpuwm.ingest.water_temperature import (
    SOURCE_ANALYSIS, SOURCE_COMPONENT_SKIN, SOURCE_LAKE_WATER,
    assemble_water_temperature, water_temperature_advisory,
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


@pytest.mark.parametrize("depth, counter", [
    (-1e-12, "unknown_depth_cells"), (np.nan, "unknown_depth_cells"),
    (np.inf, "unknown_depth_cells"),
    (np.nextafter(0.0, 1.0), "frozen_cells"),
    (1e-19, "frozen_cells"), (0.05, "frozen_cells"), (1.0, "frozen_cells"),
])
def test_invalid_unknown_and_every_positive_ice_depth_declines_the_cell_and_counts_it(
        depth, counter):
    """A frozen or unknown donor is DECLINED (NaN) and named, never refused.

    The provider is default-on for every ERA5 run with lake cells; refusing
    the whole preparation after fetch and decode for one frozen
    high-latitude lake was a new blocker on a route that ran in 2.6.5
    (ENG-008).  The assembly falls back per cell to the component skin; the
    mapping itself still invents nothing.
    """
    result = at_middle(lake_source(depth))
    assert np.isnan(result.values[0, 0])
    assert result.receipt["declined_cells"] == 1
    assert result.receipt[counter] == 1
    assert result.receipt["declined_cell_indices"] == [[0, 0]]
    assert result.receipt["ice_phase"] == "ice_free_where_provided"


def test_an_ice_free_mapping_declines_nothing():
    result = at_middle(lake_source(0.0))
    assert result.receipt["declined_cells"] == 0
    assert result.receipt["declined_cell_indices"] == []
    assert result.receipt["ice_phase"] == "ice_free"


def test_zero_weight_ice_and_missing_neighbours_do_not_change_exact_point():
    source = lake_source(1.0)
    source.fields["LAKE_ICE_DEPTH"][2, 2] = 0.0
    source.fields["LAKE_WATER_TEMP"][:2] = np.nan
    result = map_ice_free_lake_water(source, np.array([[2.0]]),
                                   np.array([[2.0]]), np.ones((1, 1), bool))
    assert result.values[0, 0] == 290.0


@pytest.mark.parametrize("temperature", [np.nan, 0.0, 169.999, 400.001])
def test_a_bad_positive_weight_temperature_is_declined_not_filled_by_the_mapping(temperature):
    source = lake_source()
    source.fields["LAKE_WATER_TEMP"][0, 0] = temperature
    result = at_middle(source)
    assert np.isnan(result.values[0, 0])
    assert result.receipt["invalid_temperature_cells"] == 1
    assert result.receipt["declined_cells"] == 1
    # The mapping never reads the grid-box skin: the fallback is the
    # assembly's, per cell, and is reported there.
    np.testing.assert_array_equal(source.fields["SKINTEMP"], 350.0)


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
    assert receipt["lake_fallback_cells"] == 0
    # A cell the provider declined falls back, PER CELL, to the component
    # skin the pre-lake-model route used, and the receipt names it.
    mapped[0, 1] = np.nan
    skin[0, 1] = 281.0
    values, provider, receipt = assemble_water_temperature(
        mapped_sst=None, mapped_skin=skin, target_land=land,
        target_lake=lake, mapped_lake_water=mapped)
    assert values[0, 0] == 280.0 and values[0, 1] == 281.0
    assert provider[0, 0] == SOURCE_LAKE_WATER
    assert provider[0, 1] == SOURCE_COMPONENT_SKIN
    assert receipt["components_on_lake_water"] == 1
    assert receipt["lake_fallback_cells"] == 1
    assert receipt["lake_fallback_cell_indices"] == [[0, 1]]
    advisory = water_temperature_advisory(
        {**receipt, "policy": "era5_class_coherent"})
    assert "1 lake cell(s) had no ice-free lake-model water" in advisory
    assert "(0, 1)" in advisory
    # A declined cell whose skin is itself inadmissible still refuses, by
    # name: nothing invents a temperature.
    skin[0, 1] = 0.0
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


def test_a_wholly_frozen_lake_prepares_and_announces_every_fallback_cell():
    lat, lon = np.mgrid[:3, :3]
    land = np.zeros((3, 3), bool)
    mapped = map_ice_free_lake_water(lake_source(0.05), lat, lon, ~land)
    assert mapped.receipt["frozen_cells"] == 9
    skin = np.full((3, 3), 265.0)
    values, provider, receipt = assemble_water_temperature(
        mapped_sst=None, mapped_skin=skin, target_land=land,
        target_lake=~land, mapped_lake_water=mapped.values)
    np.testing.assert_array_equal(values, skin)
    assert (provider == SOURCE_COMPONENT_SKIN).all()
    assert receipt["components_on_skin"] == 1
    assert receipt["components_on_lake_water"] == 0
    assert receipt["lake_fallback_cells"] == 9
    assert receipt["lake_fallback_cell_indices"] == np.argwhere(~land).tolist()
    advisory = water_temperature_advisory(receipt)
    assert "9 lake cell(s) had no ice-free lake-model water" in advisory
    assert "(2, 2)" in advisory
    skin[1, 1] = 0.0
    with pytest.raises(ValueError, match="no admissible water temperature"):
        assemble_water_temperature(
            mapped_sst=None, mapped_skin=skin, target_land=land,
            target_lake=~land, mapped_lake_water=mapped.values)


@pytest.mark.parametrize("partial", [False, True])
def test_declined_lake_cell_list_is_bounded_across_components(partial):
    # Seventy separate two-cell lakes exercise the cap across components,
    # including both the partially provided and the all-skin branches.
    land = np.ones((1, 210), bool)
    land[:, ::3] = False
    land[:, 1::3] = False
    mapped = np.full(land.shape, np.nan)
    if partial:
        mapped[:, ::3] = 280.0
    values, provider, receipt = assemble_water_temperature(
        mapped_sst=None, mapped_skin=np.full(land.shape, 270.0),
        target_land=land, target_lake=~land, mapped_lake_water=mapped)
    expected = 70 if partial else 140
    assert receipt["lake_fallback_cells"] == expected
    assert len(receipt["lake_fallback_cell_indices"]) == 64
    assert int((provider == SOURCE_COMPONENT_SKIN).sum()) == expected
    assert np.isfinite(values).all()
    advisory = water_temperature_advisory(receipt)
    assert f"{expected} lake cell(s) had no ice-free lake-model water" in advisory
    assert f"and {expected - 64} more" in advisory
