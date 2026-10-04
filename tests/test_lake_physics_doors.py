"""Keep the lake selector's optional native inputs on every initialized column."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest.lake_physics import (
    lake_physics_inputs, wrfinput_lake_physics_inputs)


def test_off_does_not_read_any_lake_input():
    cfg = SimpleNamespace(sf_lake_physics=0)
    assert lake_physics_inputs(cfg, None) == {}
    assert wrfinput_lake_physics_inputs(None, cfg) == {}


def test_absent_inputs_preserve_column_model_defaults():
    assert lake_physics_inputs(SimpleNamespace(sf_lake_physics=1), {}) == {
        "lake_depth": None, "lake_depth_flag": None, "lakemask": None,
        "lake_mask_flag": None}


def test_supplied_inputs_keep_their_values_and_identity():
    depth = np.array([[12., 50.], [-2., -3.]], np.float32)
    mask = np.array([[1., 1.], [0., 0.]], np.float32)
    fields = dict(LAKE_DEPTH=depth, LAKE_DEPTH_FLAG=1, LAKEMASK=mask, LAKEFLAG=0)
    result = lake_physics_inputs(SimpleNamespace(sf_lake_physics=1), fields)
    assert result["lake_depth"] is depth
    assert result["lakemask"] is mask
    assert result["lake_depth_flag"] == 1
    assert result["lake_mask_flag"] == 0


def test_wrfinput_reader_keeps_mask_depth_flag_and_refuses_wrong_axes(monkeypatch):
    from gpuwm import netcdf_bridge
    from gpuwm.ingest import wrfinput
    cfg = SimpleNamespace(sf_lake_physics=1)
    values = dict(LAKE_DEPTH=np.full((2, 3), 12., np.float32),
                  LAKE_DEPTH_FLAG=np.int32(1),
                  LAKEMASK=np.ones((2, 3), np.float32), LAKEFLAG=np.int32(0))
    variables = {name: SimpleNamespace(
        dimensions=("Time",) if name in ("LAKE_DEPTH_FLAG", "LAKEFLAG") else (
            "Time", "south_north", "west_east"), data=value)
        for name, value in values.items()}
    monkeypatch.setattr(netcdf_bridge, "open_dataset", lambda path: nullcontext(
        SimpleNamespace(variables=variables)))
    monkeypatch.setattr(wrfinput, "_read_numeric", lambda variable: variable.data)
    restored = SimpleNamespace(raw={}, path="lake.nc")
    result = wrfinput_lake_physics_inputs(restored, cfg)
    for name, value in values.items():
        keyword = "lake_mask_flag" if name == "LAKEFLAG" else name.lower()
        np.testing.assert_array_equal(result[keyword], value)
    variables["LAKE_DEPTH"].dimensions = ("Time", "west_east", "south_north")
    with pytest.raises(ValueError, match="LAKE_DEPTH axes.*different columns"):
        wrfinput_lake_physics_inputs(restored, cfg)
