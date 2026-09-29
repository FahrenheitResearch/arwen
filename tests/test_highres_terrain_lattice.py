"""Terrain cut for two footprints agrees on the model cells they share.

A moving nest's statics corridor and the nest itself are two footprints
cut from the same terrain tiles.  The crops started their pixel lattice
at each footprint's own west/north edge, so the same ground sampled a
different sub-pixel position in each; the corridor's terrain then differed
from the nest's on every cell (HGT_M by up to 0.15 m, TMN with it) and the
first move refused with "footprint-rebuilt statics differ from the outgoing
child's on shared ground".  Crops are now cut on one fixed lattice per
source, so the terrain a model cell receives no longer depends on which
footprint's window it came from.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwm.static import rust_bridge
from gpuwm.static.highres import BoundRaster, resample_continuous, sha256_file
from gpuwm.static.highres_fetch import (FetchedFile, FootprintBBox,
                                        derive_global_terrain_window,
                                        derive_terrain_window)
from gpuwm.static.lambert import LambertGrid

#: Two real Copernicus GLO-30 clips either side of 8 E, 46.43 to 46.49 N.
HIGHRES_FIXTURES = (Path(__file__).resolve().parent.parent
                    / "tools" / "rustwx" / "crates" / "static-fields"
                    / "tests" / "fixtures" / "highres")

#: A nest's footprint and a larger corridor around it whose edges sit at
#: other sub-pixel offsets (24.12 and 18.72 source pixels apart).
CHILD = FootprintBBox(lat_min=46.452, lat_max=46.466,
                      lon_min=7.992, lon_max=8.012)
CORRIDOR = FootprintBBox(lat_min=46.4473, lat_max=46.4712,
                         lon_min=7.9853, lon_max=8.0206)


def _tiles():
    if rust_bridge.unavailable_reason() is not None:
        pytest.fail("the Rust static-fields bridge is the default terrain "
                    f"door and must load: {rust_bridge.unavailable_reason()}")
    tiles = []
    for name in ("mosaic_west.tif", "mosaic_east.tif"):
        path = HIGHRES_FIXTURES / name
        if not path.is_file():
            pytest.skip(f"{name} is not in this checkout")
        tiles.append(FetchedFile(
            path=path, url=f"fixture:{name}", sha256=sha256_file(path),
            bytes=path.stat().st_size, fetched_utc="", cache_hit=True))
    return tiles


def _nest_grid() -> LambertGrid:
    """An 11 x 11 cell, 100 m grid inside the nest's footprint."""
    return LambertGrid(ref_lat=46.459, ref_lon=8.002, truelat1=30.0,
                       truelat2=60.0, stand_lon=8.002, dx=100.0, dy=100.0,
                       e_we=12, e_sn=12)


def _terrain(window: FetchedFile, grid: LambertGrid) -> np.ndarray:
    source = BoundRaster(
        path=Path(window.path), sha256=window.sha256, source_id="fixture",
        role="terrain", source_url="https://example.invalid/source",
        license_id="test-only", license_url="https://example.invalid/licence",
        nominal_resolution="1 arc-second")
    return resample_continuous(source, grid, method="average")


def _global(bbox, cache_root):
    window, audit = derive_global_terrain_window(
        _tiles(), bbox, cache_root, sea_level_fill=None)
    assert audit["no_data_pixels_outside_coverage"] == 0, audit
    return window


@pytest.mark.parametrize("derive", ["global-terrain-window",
                                    "terrain-window"])
def test_a_model_grid_gets_the_same_terrain_from_either_footprint(
        tmp_path, derive):
    grid = _nest_grid()
    planes = []
    for bbox in (CHILD, CORRIDOR):
        if derive == "global-terrain-window":
            window = _global(bbox, tmp_path)
        else:
            window = derive_terrain_window(_tiles(), bbox, tmp_path)
        planes.append(_terrain(window, grid))
    child, corridor = planes
    assert np.isfinite(child).all()
    assert float(child.max() - child.min()) > 10.0, "real relief"
    np.testing.assert_array_equal(child, corridor)


def test_a_window_cut_on_the_old_edge_lattice_is_not_reused(tmp_path):
    """The derived-window cache is keyed so an edge-anchored window
    already on disk is rebuilt, not served to a moving nest."""
    tiles = _tiles()
    identity = hashlib.sha256(json.dumps(
        {"tiles": sorted(item.sha256 for item in tiles),
         "bbox": CHILD.as_dict(), "res": 1.0 / 3600.0,
         "fill": None, "src_nodata": None,
         "kind": "global-terrain-window-v1"},
        sort_keys=True).encode("utf-8")).hexdigest()[:20]
    stale = tmp_path / "derived" / f"terrain_global_{identity}.tif"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"an edge-anchored window from an older build")
    stale.with_name(stale.name + ".sha256.json").write_text(json.dumps(
        {"sha256": hashlib.sha256(stale.read_bytes()).hexdigest(),
         "bytes": stale.stat().st_size, "fetched_utc": ""}),
        encoding="utf-8")
    stale.with_name(f"terrain_global_{identity}.audit.json").write_text(
        "{}", encoding="utf-8")
    window, _audit = derive_global_terrain_window(
        tiles, CHILD, tmp_path, sea_level_fill=None)
    assert Path(window.path).resolve() != stale.resolve()
    assert not window.cache_hit
