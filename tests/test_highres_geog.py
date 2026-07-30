"""Focused gates for provenance-bound high-resolution geography."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpuwm.static.highres import (
    BoundRaster,
    _raster_geometry,
    merge_highres_overrides,
    resample_continuous,
    resample_mapped_categories,
    sha256_file,
    usda_texture_category,
)
from gpuwm.static.lambert import LambertGrid


def _grid() -> LambertGrid:
    return LambertGrid(
        ref_lat=39.66, ref_lon=-84.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-84.0, dx=500.0, dy=500.0, e_we=9, e_sn=9,
    )


def _write_raster(path: Path, values: np.ndarray) -> None:
    rasterio = pytest.importorskip("rasterio")
    transform = rasterio.transform.from_bounds(
        -84.1, 39.57, -83.9, 39.75, values.shape[1], values.shape[0])
    with rasterio.open(
            path, "w", driver="GTiff", height=values.shape[0],
            width=values.shape[1], count=1, dtype=values.dtype,
            crs="EPSG:4326", transform=transform) as dataset:
        dataset.write(values, 1)


def _bound(path: Path, *, role: str = "test") -> BoundRaster:
    return BoundRaster(
        path=path, sha256=sha256_file(path), source_id="synthetic",
        role=role, source_url="https://example.invalid/source",
        license_id="test-only", license_url="https://example.invalid/license",
        nominal_resolution="test",
    )


def test_bound_raster_rejects_hash_mismatch_before_decode(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"payload")
    source = BoundRaster(
        path=path, sha256="0" * 64, source_id="synthetic", role="terrain",
        source_url="https://example.invalid/source", license_id="test-only",
        license_url="https://example.invalid/license",
        nominal_resolution="test",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        source.verify()


def test_bound_raster_rejects_size_mismatch_before_hashing(tmp_path):
    path = tmp_path / "source.bin"
    path.write_bytes(b"payload")
    source = BoundRaster(
        path=path, sha256=sha256_file(path), source_id="synthetic",
        role="terrain", source_url="https://example.invalid/source",
        license_id="test-only", license_url="https://example.invalid/license",
        nominal_resolution="test", expected_bytes=999,
    )
    with pytest.raises(ValueError, match="size mismatch"):
        source.verify()


def test_raster_geometry_matches_wps_lambert_mass_points():
    pyproj = pytest.importorskip("pyproj")
    grid = _grid()
    crs, transform, (ny, nx) = _raster_geometry(grid)
    columns, rows = np.meshgrid(
        np.arange(nx, dtype=np.float64) + 0.5,
        np.arange(ny, dtype=np.float64) + 0.5,
    )
    x, y = transform * (columns, rows)
    longitude, latitude = pyproj.Transformer.from_crs(
        crs, "EPSG:4326", always_xy=True).transform(x, y)
    expected_latitude, expected_longitude = grid.latlon_mass()
    np.testing.assert_allclose(latitude[::-1], expected_latitude, atol=2.0e-12)
    np.testing.assert_allclose(longitude[::-1], expected_longitude, atol=2.0e-12)


def test_lambert_resampling_preserves_constant_and_category_fractions(tmp_path):
    terrain_path = tmp_path / "terrain.tif"
    _write_raster(terrain_path, np.full((180, 200), 247.5, dtype=np.float32))
    terrain = resample_continuous(_bound(terrain_path), _grid())
    np.testing.assert_allclose(terrain, 247.5, rtol=0.0, atol=1.0e-9)

    categories = np.full((180, 200), 82, dtype=np.uint8)
    categories[:, :100] = 11
    category_path = tmp_path / "landcover.tif"
    _write_raster(category_path, categories)
    fractions = resample_mapped_categories(
        _bound(category_path), _grid(), {11: 21, 82: 12},
        category_count=21,
    )
    assert fractions.shape == (21, 8, 8)
    np.testing.assert_allclose(fractions.sum(axis=0), 1.0, atol=1.0e-6)
    assert np.any(fractions[20] > 0.99)
    assert np.any(fractions[11] > 0.99)


def test_usda_texture_rules_cover_all_wrf_mineral_categories():
    samples = np.asarray([
        [92, 5, 3], [82, 12, 6], [65, 25, 10], [20, 65, 15],
        [5, 90, 5], [40, 40, 20], [55, 15, 30], [10, 55, 35],
        [35, 30, 35], [52, 8, 40], [10, 47, 43], [30, 25, 45],
    ], dtype=np.float64)
    actual = usda_texture_category(
        samples[:, 0][None, :], samples[:, 1][None, :],
        samples[:, 2][None, :],
    )[0]
    np.testing.assert_array_equal(actual, np.arange(1, 13))


def test_merge_highres_overrides_keeps_complete_noah_state():
    shape = (4, 4)
    old_land = np.ones(shape, dtype=np.float64)
    old_land[0, 0] = 0.0
    baseline = {
        "HGT_M": np.full(shape, 100.0),
        "LANDMASK": old_land,
        "LU_INDEX": np.where(old_land > 0.5, 12.0, 21.0),
        "LANDUSEF": np.zeros((21, *shape)),
        "SOILCTOP": np.zeros((16, *shape)),
        "SCT_DOM": np.full(shape, 6.0),
        "SOILCBOT": np.zeros((16, *shape)),
        "SCB_DOM": np.full(shape, 6.0),
        "GREENFRAC": np.full((12, *shape), 0.5),
        "LAI12M": np.full((12, *shape), 2.0),
        "ALBEDO12M": np.full((12, *shape), 15.0),
        "SNOALB": np.full(shape, 0.4),
        "SOILTEMP": np.full(shape, 285.0),
        "TMN": np.full(shape, 284.35),
    }
    for name in ("LANDUSEF", "SOILCTOP", "SOILCBOT"):
        baseline[name][0] = 1.0
    new_land = old_land.copy()
    new_land[0, 0] = 1.0
    new_land[3, 3] = 0.0
    overrides = {
        "HGT_M": np.full(shape, 120.0),
        "LANDMASK": new_land,
        "LU_INDEX": np.where(new_land > 0.5, 12.0, 21.0),
        "LANDUSEF": baseline["LANDUSEF"],
        "SOILCTOP": baseline["SOILCTOP"],
        "SCT_DOM": baseline["SCT_DOM"],
        "SOILCBOT": baseline["SOILCBOT"],
        "SCB_DOM": baseline["SCB_DOM"],
    }
    merged, audit = merge_highres_overrides(baseline, overrides)
    assert audit["newly_land_nearest_climatology_fallback_cells"] == 1
    assert audit["newly_water_masked_cells"] == 1
    assert merged["SOILTEMP"][0, 0] == 285.0
    assert merged["SOILTEMP"][3, 3] == 0.0
    assert merged["ALBEDO12M"][0, 3, 3] == 8.0
    assert merged["TMN"][0, 0] == pytest.approx(285.0 - 0.0065 * 120.0)
    assert all(np.isfinite(value).all() for value in merged.values())
