"""WPS lake depth must use its own average, bounded search, mask and fill."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.static.build import GeogSelection
from gpuwm.static.lake import (
    LAKE_DEPTH_ROW, build_lake_fields, with_lake_statics)


@pytest.mark.parametrize("option,use_depth,wanted", [(0, 1, False),
                                                   (1, 0, False),
                                                   (1, 1, True)])
def test_depth_dataset_is_requested_only_when_consumed(tmp_path, option, use_depth, wanted):
    selection = GeogSelection.fallback(tmp_path)
    result = with_lake_statics(selection, SimpleNamespace(
        sf_lake_physics=option, use_lakedepth=use_depth))
    assert result.lake_depth is wanted
    if not wanted:
        assert result is selection


def test_absent_depth_cannot_silently_become_a_constant_lake(tmp_path):
    with pytest.raises(FileNotFoundError, match="use_lakedepth=1.*lake_depth"):
        build_lake_fields(None, tmp_path, landuse_path=tmp_path / "land")


def test_old_native_library_cannot_ignore_the_new_row_options(monkeypatch):
    from gpuwm.static import rust_bridge
    old = SimpleNamespace(**{rust_bridge.OROGRAPHIC_MARKER: lambda: 2})
    monkeypatch.setattr(rust_bridge, "load", lambda: old)
    with pytest.raises(rust_bridge.StaticBridgeError, match="different bathymetry"):
        rust_bridge.build_orographic(0, {"fields": [dict(LAKE_DEPTH_ROW, path="depth")]})


def _dataset(path, *, categorical=False, value=123):
    path.mkdir(parents=True)
    text = (
        "projection=regular_ll\ndx=.2\ndy=.2\nknown_x=1\nknown_y=1\n"
        "known_lat=33\nknown_lon=-92\nsigned=no\n"
        "tile_x=21\ntile_y=21\ntile_z=1\n")
    if categorical:
        text += ("type=categorical\ncategory_min=1\ncategory_max=21\n"
                 "iswater=17\nislake=21\nisice=15\nwordsize=1\nmissing_value=0\n")
        data = np.full((21, 21), value, np.uint8)
    else:
        text += "type=continuous\nwordsize=2\nscale_factor=.1\nmissing_value=65535\n"
        data = np.full((21, 21), value, ">u2")
    (path / "index").write_text(text)
    (path / "00001-00021.00001-00021").write_bytes(data.tobytes())


@pytest.mark.parametrize("land_category,depth,wanted", [(21, 123, np.float32(123)*np.float32(.1)),
                                                        (1, 123, 10.),
                                                        (21, 65535, 10.)])
def test_native_lake_row_preserves_wps_mask_scale_and_fill(
        tmp_path, land_category, depth, wanted):
    from gpuwm.static.lambert import LambertGrid
    _dataset(tmp_path / "lake_depth", value=depth)
    _dataset(tmp_path / "land", categorical=True, value=land_category)
    grid = LambertGrid(ref_lat=35., ref_lon=-90., truelat1=30., truelat2=60.,
                       stand_lon=-90., dx=50000., dy=50000., e_we=4, e_sn=4)
    report = {}
    result = build_lake_fields(grid, tmp_path, landuse_path=tmp_path / "land",
                               halo=0, coverage_report=report)
    np.testing.assert_array_equal(result["LAKE_DEPTH"], np.full((3, 3), float(wanted)))
    assert report["lake_depth"]["field"] == "lake_depth"
    assert report["lake_depth"]["output_field"] == "LAKE_DEPTH"
    assert report["lake_depth"]["status"] == "PASS"
