"""Regression for native em_fire's metric FX coordinates labeled degrees."""
import pytest

from gpuwm.sfire_coordinates import fire_coordinate_mode


def test_native_ideal_projection_overrules_misleading_fx_names_and_units():
    fields = {"FXLAT": object(), "FXLONG": object(), "MAP_PROJ": 0}
    assert fire_coordinate_mode(fields) == "metric"


def test_geographic_bundle_and_native_projection_establish_geography():
    fields = {"FXLAT": object(), "FXLONG": object()}
    assert fire_coordinate_mode({**fields, "MAP_PROJ": 1}) == "geographic"
    assert fire_coordinate_mode({**fields, "_METADATA": {"grid_spec": {}}}) == "geographic"


def test_unbound_and_partial_coordinates_refuse_before_ignition():
    with pytest.raises(ValueError, match="authority"):
        fire_coordinate_mode({"FXLAT": object(), "FXLONG": object()})
    with pytest.raises(ValueError, match="both"):
        fire_coordinate_mode({"FXLAT": object(), "MAP_PROJ": 0})
    assert fire_coordinate_mode({}) == "metric"
