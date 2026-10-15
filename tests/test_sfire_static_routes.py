"""Packaged SFIRE static routes speak the protocol their services answer.

The breakage these prevent (measured 2026-10-12 on a real fire, the 2024
Mountain Fire domain): LANDFIRE's GeoServer answers WCS 1.0.0 GetCoverage
with an ExceptionReport ("java.lang.String cannot be cast to ... EObject")
and no longer serves conus_2022, so `gpuwm fire-static` could not acquire
any fuel; and the default 3DEP WCS terrain route returned HTTP 404/504 for
a 66 x 60 km fire domain.  Offline: the tests read the route table and the
query builder, never the network.
"""
from urllib.parse import parse_qs, parse_qsl

import pytest

from gpuwm.static import sfire


def _query(source_id, year=2024, bounds=(-2075025.0, 1514985.0, -2069985.0, 1520025.0)):
    row = sfire.source_routes()[source_id]
    resolution = row.get("resolution", row.get("resolution_degrees"))
    return row, sfire._wcs_query(row, year, bounds, resolution, row.get("crs", "EPSG:4326"))


@pytest.mark.parametrize("source_id,model", [("landfire-fbfm40", "40"), ("landfire-fbfm13", "13")])
def test_landfire_requests_native_pixels_over_wcs_201(source_id, model):
    row, text = _query(source_id)
    pairs = parse_qsl(text)
    query = parse_qs(text)
    assert query["version"] == ["2.0.1"] and query["request"] == ["GetCoverage"]
    assert query["coverageId"] == [f"landfire_wcs__LF2024_FBFM{model}_CONUS"]
    assert [v for k, v in pairs if k == "subset"] == [
        "X(-2075025.0000000000,-2069985.0000000000)",
        "Y(1514985.0000000000,1520025.0000000000)"]
    assert query["format"] == ["image/geotiff"]
    # No resampling knobs: WCS 2.0.1 returns the coverage's own 30 m pixels.
    assert not {"resx", "resy", "interpolation", "bbox"} & set(query)


def test_landfire_years_are_the_ones_its_service_publishes():
    for source_id in ("landfire-fbfm13", "landfire-fbfm40"):
        assert sfire.source_routes()[source_id]["years"] == [2016, 2023, 2024, 2025]


def test_a_row_without_a_version_keeps_its_wcs_100_request():
    row, text = _query("usgs-3dep-wcs", year=None, bounds=(-119.1, 34.2, -119.0, 34.3))
    query = parse_qs(text)
    assert query["version"] == ["1.0.0"] and query["coverage"] == ["DEP3Elevation"]
    assert query["bbox"] == ["-119.1000000000,34.2000000000,-119.0000000000,34.3000000000"]
    assert query["interpolation"] == ["nearest neighbor"]


def test_an_unknown_wcs_version_is_refused_by_name():
    row = dict(sfire.source_routes()["landfire-fbfm40"], wcs_version="1.1.0")
    with pytest.raises(ValueError, match="1.1.0"):
        sfire._wcs_query(row, 2024, (0., 0., 30., 30.), 30.0, "EPSG:5070")


def test_default_terrain_is_the_3dep_tile_route():
    assert dict(sfire.DEFAULT_ROUTES) == {"terrain": "usgs-3dep-13as", "fuel": "landfire-fbfm40"}
    assert sfire.source_routes()["usgs-3dep-13as"]["kind"] == "terrain-tiles"
