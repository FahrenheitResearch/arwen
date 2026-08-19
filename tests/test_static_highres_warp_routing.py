"""The high-resolution warp substrate runs on the Rust crate BY DEFAULT.

The breakage these gates prevent, named: through the port-lane
integration the `resample_*` entry points of :mod:`gpuwm.static.highres`
kept their rasterio bodies behind a source comment ("until the port
lanes integrate").  The lanes had landed.  A user who turned
``[static.highres] enabled = true`` on therefore ran GeoTIFF decode,
CRS construction and the warp onto the model grid in Python -- a data
path the project boundary places in Drew's Rust -- and nothing told
them: no registry entry, no doctor row, no receipt field, no console
line.  A silent Python data path is exactly what fixed-means-default
exists to stop.

So the assertion here is not "the Rust path works".  It is stronger and
deliberately brutal: with ``rasterio``, ``pyproj`` and ``affine`` made
UNIMPORTABLE, every default high-resolution call must still produce its
answer.  An import of any of the three raises, so a body that quietly
falls back to Python fails loudly instead of passing quietly.

The Python bodies stay, as the parity reference and as the explicit
``GPUWM_STATIC_PYTHON=1`` workaround, and the last two tests pin that
they announce themselves once per operation.
"""
from __future__ import annotations

import builtins
import contextlib
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwm.static import rust_bridge
from gpuwm.static.highres import (BoundRaster, build_highres_overrides,
                                  build_terrain_override,
                                  resample_continuous,
                                  resample_mapped_categories, sha256_file)
from gpuwm.static.lambert import LambertGrid

#: The Python geography stack the default path must no longer need.
PYTHON_GEOGRAPHY_STACK = ("rasterio", "pyproj", "affine")

#: The lane-3 goldens: real Copernicus DEM / land-cover / SoilGrids
#: windows, decoded by the REAL Python to produce the crate's expected
#: bytes (see the crate's tests/fixtures/highres/generate_goldens.py).
HIGHRES_FIXTURES = (Path(__file__).resolve().parent.parent
                    / "tools" / "rustwx" / "crates" / "static-fields"
                    / "tests" / "fixtures" / "highres")


@contextlib.contextmanager
def no_python_geography_stack():
    """Make rasterio/pyproj/affine unimportable inside the block.

    ``import x`` always calls ``builtins.__import__`` even for a module
    already in ``sys.modules``, so this catches a cached import too --
    which matters, because these three are importable on this box and a
    fallback would otherwise succeed invisibly.
    """
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in PYTHON_GEOGRAPHY_STACK:
            raise AssertionError(
                f"the default high-resolution path imported {name!r}; the "
                "warp substrate is supposed to be the Rust static-fields "
                "bridge, and a Python geography import here is the silent "
                "data path this gate exists to catch")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = guarded
    try:
        yield
    finally:
        builtins.__import__ = real_import


def _bridge_or_fail():
    """A missing bridge FAILS: it is the shipped default, not an option."""
    reason = rust_bridge.unavailable_reason()
    if reason is not None:
        pytest.fail(
            "the Rust static-fields bridge is not loadable, so the default "
            f"high-resolution warp cannot run: {reason}")


def _meta():
    path = HIGHRES_FIXTURES / "meta.json"
    if not path.is_file():
        pytest.skip("lane-3 highres fixtures not present in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


def _grid_from_spec(spec: dict) -> LambertGrid:
    return LambertGrid(
        spec["ref_lat"], spec["ref_lon"], spec["truelat1"],
        spec["truelat2"], spec["stand_lon"], spec["dx"], spec["dy"],
        spec["e_we"], spec["e_sn"], known_x=spec["known_x"],
        known_y=spec["known_y"])


def _bound(path: Path, *, role: str, crs_override: str | None = None,
           nodata_override: float | None = None,
           scale_factor: float = 1.0) -> BoundRaster:
    return BoundRaster(
        path=path, sha256=sha256_file(path), source_id="lane3-golden",
        role=role, source_url="https://example.invalid/source",
        license_id="test-only", license_url="https://example.invalid/license",
        nominal_resolution="test", crs_override=crs_override,
        nodata_override=nodata_override, scale_factor=scale_factor)


# ---------------------------------------------------------------------------
# The default path needs no Python geography stack
# ---------------------------------------------------------------------------

def test_resample_continuous_needs_no_python_geography_stack():
    _bridge_or_fail()
    meta = _meta()
    clip = HIGHRES_FIXTURES / "terrain_clip.tif"
    if not clip.is_file():
        pytest.skip("terrain_clip.tif not present in this checkout")
    grid = _grid_from_spec(meta["terrain_warp"]["grid_spec"])
    source = _bound(clip, role="terrain")

    with no_python_geography_stack():
        plane = resample_continuous(source, grid, method="average")

    assert plane.shape == (grid.e_sn - 1, grid.e_we - 1)
    assert np.isfinite(plane).all()


def test_resample_mapped_categories_needs_no_python_geography_stack():
    _bridge_or_fail()
    meta = _meta()
    landcover = HIGHRES_FIXTURES / "landcover.tif"
    if not landcover.is_file():
        pytest.skip("landcover.tif not present in this checkout")
    case = meta["landcover"]
    grid = _grid_from_spec(case["grid_spec"])
    mapping = {int(raw): int(target) for raw, target in case["mapping"]}
    source = _bound(landcover, role="landcover",
                    nodata_override=float(case["nodata"]))

    with no_python_geography_stack():
        fractions = resample_mapped_categories(
            source, grid, mapping, category_count=21)

    assert fractions.shape == (21, grid.e_sn - 1, grid.e_we - 1)


def test_the_soil_leg_needs_no_python_geography_stack():
    """`build_highres_overrides` reads six SoilGrids GeoTIFFs, takes the
    depth-weighted mean, classifies it and warps the categories.  All of
    that is decode plus transform."""
    _bridge_or_fail()
    meta = _meta()
    case = meta["soil"]
    layers = {}
    for component in ("sand", "silt", "clay"):
        for depth in ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm"):
            path = HIGHRES_FIXTURES / f"soil_{component}_{depth}.tif"
            if not path.is_file():
                pytest.skip("SoilGrids fixtures not present in this checkout")
            layers[(component, depth)] = _bound(
                path, role="soil", crs_override=case["crs_override"],
                nodata_override=float(case["nodata"]),
                scale_factor=float(case["scale_factor"]))

    from gpuwm.static.highres import soilgrids_category_fractions
    grid = _grid_from_spec(case["grid_spec"])
    with no_python_geography_stack():
        fractions, audit = soilgrids_category_fractions(
            layers, {"0-5cm": 5.0, "5-15cm": 10.0, "15-30cm": 15.0},
            grid, category_count=16)

    assert fractions.shape[0] == 16
    assert audit["valid_source_pixels"] > 0


def test_build_terrain_override_needs_no_python_geography_stack():
    _bridge_or_fail()
    meta = _meta()
    clip = HIGHRES_FIXTURES / "terrain_clip.tif"
    if not clip.is_file():
        pytest.skip("terrain_clip.tif not present in this checkout")
    warp_meta = meta["terrain_warp"]
    grid = _grid_from_spec(warp_meta["grid_spec"])
    source = _bound(clip, role="terrain")

    with no_python_geography_stack():
        fields, audit = build_terrain_override(
            grid, terrain=source, halo=int(warp_meta["halo"]))

    assert set(fields) == {"HGT_M"}
    assert fields["HGT_M"].shape == (grid.e_sn - 1, grid.e_we - 1)
    assert np.isfinite(fields["HGT_M"]).all()
    assert audit["sources"][0]["source_id"] == "lane3-golden"


def test_derive_windows_need_no_python_geography_stack(tmp_path):
    """The fetch driver's mosaic/clip/fill derivations are byte work."""
    _bridge_or_fail()
    meta = _meta()
    west = HIGHRES_FIXTURES / "mosaic_west.tif"
    east = HIGHRES_FIXTURES / "mosaic_east.tif"
    if not west.is_file() or not east.is_file():
        pytest.skip("mosaic fixtures not present in this checkout")
    import shutil

    from gpuwm.static.highres_fetch import (FootprintBBox,
                                            derive_global_terrain_window,
                                            record_local_artifact)

    # Copy the tiles out of the committed fixture tree first:
    # record_local_artifact writes a sha256 sidecar beside its argument,
    # and a test does not get to leave artifacts in a golden directory.
    staged = tmp_path / "tiles"
    staged.mkdir()
    tiles = []
    for path in (west, east):
        copy = staged / path.name
        shutil.copyfile(path, copy)
        tiles.append(record_local_artifact(copy, url=f"file:{path.name}"))
    bounds = meta["mosaic"]["bounds_wsen"]
    bbox = FootprintBBox(lat_min=bounds[1] + 0.01, lat_max=bounds[3] - 0.01,
                         lon_min=bounds[0] + 0.01, lon_max=bounds[2] - 0.01)

    with no_python_geography_stack():
        window, audit = derive_global_terrain_window(
            tiles, bbox, tmp_path,
            resolution_deg=meta["mosaic"]["resolution_deg"])

    assert window.path.is_file()
    assert audit["total_pixels"] > 0
    assert audit["output_shape"] == meta["mosaic"]["shape"]


def test_the_soilgrids_window_snap_needs_no_python_geography_stack():
    """The IGH window snap is a CRS transform, not orchestration."""
    _bridge_or_fail()
    from gpuwm.static.highres_fetch import FootprintBBox, _soilgrids_window_m

    bbox = FootprintBBox(lat_min=39.3, lat_max=39.8,
                         lon_min=-84.3, lon_max=-83.7)
    with no_python_geography_stack():
        window = _soilgrids_window_m(bbox)
    assert len(window) == 4
    x0, x1, y0, y1 = window
    assert x1 > x0 and y1 > y0


# ---------------------------------------------------------------------------
# The fallback still exists, and says so
# ---------------------------------------------------------------------------

def test_the_python_fallback_announces_itself_once(monkeypatch, capsys):
    _bridge_or_fail()
    meta = _meta()
    clip = HIGHRES_FIXTURES / "terrain_clip.tif"
    if not clip.is_file():
        pytest.skip("terrain_clip.tif not present in this checkout")
    pytest.importorskip("rasterio")
    monkeypatch.setattr(rust_bridge, "_REPORTED_FALLBACKS", set())
    monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
    grid = _grid_from_spec(meta["terrain_warp"]["grid_spec"])
    source = _bound(clip, role="terrain")

    resample_continuous(source, grid, method="average")
    resample_continuous(source, grid, method="average")

    printed = capsys.readouterr().out
    assert printed.count("WORKAROUND") == 1, printed
    assert "resample_continuous" in printed
    assert rust_bridge.STATIC_PYTHON_ENV in printed


def test_default_and_fallback_warps_agree_within_the_recorded_caps(
        monkeypatch):
    """Rust-vs-rasterio on the pinned real footprint, gated by the caps
    the goldens recorded (the warper is a black box, not a spec)."""
    _bridge_or_fail()
    meta = _meta()
    clip = HIGHRES_FIXTURES / "terrain_clip.tif"
    if not clip.is_file():
        pytest.skip("terrain_clip.tif not present in this checkout")
    pytest.importorskip("rasterio")
    warp_meta = meta["terrain_warp"]
    grid = _grid_from_spec(warp_meta["grid_spec"])
    source = _bound(clip, role="terrain")

    rust = resample_continuous(source, grid, method="average")
    monkeypatch.setenv(rust_bridge.STATIC_PYTHON_ENV, "1")
    python = resample_continuous(source, grid, method="average")

    delta = np.abs(rust - python)
    assert delta.max() <= warp_meta["max_abs_delta_cap_m"], (
        f"max |delta| {delta.max():.3f} m beyond the recorded cap")
    assert delta.mean() <= warp_meta["mean_abs_delta_cap_m"], (
        f"mean |delta| {delta.mean():.4f} m beyond the recorded cap")


# ---------------------------------------------------------------------------
# The georeferencing convention the real terrain source actually uses
# ---------------------------------------------------------------------------

def _index_raster(path: Path, *, ny: int, nx: int, res: float,
                  west: float, north: float, point: bool) -> None:
    """An index-valued GeoTIFF whose every pixel names itself.

    ``point`` writes GTRasterTypeGeoKey = RasterPixelIsPoint, where the
    tiepoint is a pixel CENTRE and the raster's origin is half a pixel
    north-west of it.  Copernicus DEM GLO-30 -- the near-global terrain
    source -- ships exactly that.
    """
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import Affine

    values = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx)
    with rasterio.open(path, "w", driver="GTiff", height=ny, width=nx,
                       count=1, dtype="float32", crs="EPSG:4326",
                       transform=Affine(res, 0.0, west, 0.0, -res, north)
                       ) as dataset:
        dataset.write(values, 1)
        if point:
            # GDAL turns this tag into GTRasterTypeGeoKey = 2 and writes
            # the tiepoint at the pixel CENTRE; it is a tag, not a
            # creation option, so it is set on the open dataset.
            dataset.update_tags(AREA_OR_POINT="Point")


@pytest.mark.parametrize("point", [False, True])
def test_the_mosaic_samples_the_pixel_that_contains_the_cell(tmp_path,
                                                             point):
    """Both registrations, one rule, and the instrument checked both ways.

    A nearest-neighbour mosaic must take the source pixel whose extent
    CONTAINS the destination pixel centre.  Every pixel of the fixture
    carries its own flat index, so the value in a destination cell names
    the source pixel that was sampled and there is nothing to interpret.

    The ``point`` arm is the one that mattered: Copernicus DEM GLO-30
    ships RasterPixelIsPoint with a tiepoint at (8.0, 47.0), and reading
    that as a corner puts the whole tile half a pixel (~15 m) south-east.
    MEASURED on a real 500 m Alpine domain before this was fixed: max
    |delta| 62.2 m and mean |delta| 8.05 m of terrain height, and one
    full 30 m source pixel of displacement wherever the half-pixel
    crossed a boundary.  The committed lane-3 goldens could not catch it
    -- they were written out of the source tiles BY RASTERIO, which
    emits PixelIsArea with the shift already folded in, so the fixtures
    never carried the tag under test.  The ``area`` arm is the control:
    it must pass before and after, or this test is measuring the wrong
    thing.
    """
    _bridge_or_fail()
    rasterio = pytest.importorskip("rasterio")

    res = 1.0 / 3600.0
    west, north, ny, nx = 10.0, 50.0, 200, 200
    tile = tmp_path / ("point.tif" if point else "area.tif")
    _index_raster(tile, ny=ny, nx=nx, res=res, west=west, north=north,
                  point=point)

    # A destination window deliberately off the source grid, so the
    # containing-pixel choice is decided by the registration and not by
    # a lucky alignment.
    bounds = [west + (20 + 0.62) * res, north - (120 + 0.13) * res,
              west + (120 + 0.62) * res, north - (20 + 0.13) * res]
    out = tmp_path / "window.tif"
    rust_bridge.highres_derive_window({
        "kind": "terrain-window", "tiles": [str(tile)],
        "bounds": bounds, "out_path": str(out)})

    with rasterio.open(tile) as source:
        source_transform = source.transform
    with rasterio.open(out) as window:
        got = window.read(1)
        window_transform = window.transform

    rows, cols = got.shape
    checked = 0
    for row in range(2, rows - 2, 7):
        for col in range(2, cols - 2, 7):
            x = window_transform.c + window_transform.a * (col + 0.5)
            y = window_transform.f + window_transform.e * (row + 0.5)
            src_col = int(np.floor((x - source_transform.c)
                                   / source_transform.a))
            src_row = int(np.floor((y - source_transform.f)
                                   / source_transform.e))
            expected = src_row * nx + src_col
            assert int(got[row, col]) == expected, (
                f"destination [{row},{col}] took source index "
                f"{int(got[row, col])} (row {int(got[row, col]) // nx}, col "
                f"{int(got[row, col]) % nx}); the pixel containing its "
                f"centre is {expected} (row {src_row}, col {src_col}). "
                f"registration = {'point' if point else 'area'}")
            checked += 1
    assert checked > 100, "the sweep checked too few cells to mean anything"


def test_the_deferral_comment_is_gone():
    """The scope-out this lane closes must not survive as prose.

    A comment that says the substrate is still Python outlives the
    change it describes and becomes the next reader's ground truth.
    """
    source = (Path(__file__).resolve().parent.parent / "gpuwm" / "static"
              / "highres.py").read_text(encoding="utf-8")
    assert "until the port lanes integrate" not in source
    assert "keep their rasterio" not in source


# ---------------------------------------------------------------------------
# A library staged before these entry points existed
# ---------------------------------------------------------------------------


class _AbiOnlyEntry:
    """One ctypes entry point: settable signature, callable, nothing else."""

    def __init__(self, answer: int) -> None:
        self._answer = answer
        self.argtypes: list = []
        self.restype = None

    def __call__(self, *_args):
        return self._answer


#: The entry points a build from before the warp port does not export.
#: Measured against a real staged `~/.gpuwm/bridges/static_fields.dll`
#: on 2026-08-18: it carried `gpuwm_static_highres_merge` and not these.
ADDED_BY_THE_WARP_PORT = ("gpuwm_static_highres_resample",
                          "gpuwm_static_highres_transform_points")


class _StaleLibrary:
    """A build from before the warp entry points, ABI number unchanged.

    The shape of a real one: `~/.gpuwm/bridges/static_fields.dll` staged
    before those entry points were added still answers
    ``gpuwm_static_abi_version() == 1``, because ADDING an entry point
    did not change that number.  Everything else resolves; the two new
    names do not.
    """

    def __init__(self, abi: int) -> None:
        self.gpuwm_static_abi_version = _AbiOnlyEntry(abi)

    def __getattr__(self, name: str):
        if name in ADDED_BY_THE_WARP_PORT:
            raise AttributeError(f"function '{name}' not found")
        entry = _AbiOnlyEntry(0)
        setattr(self, name, entry)
        return entry


def test_a_library_missing_an_entry_point_is_refused_not_a_traceback(
        monkeypatch, tmp_path):
    """The breakage: `gpuwm doctor` crashed on a stale staged bundle.

    Measured 2026-08-18 at the law-remediation union: a box carrying a
    `static_fields` library built before `gpuwm_static_highres_resample`
    existed answered the ABI handshake, and `load()` then died on a bare
    ``AttributeError`` from ctypes -- which `unavailable_reason` does not
    catch, so `gpuwm doctor` printed a traceback instead of a report on
    exactly the partial install it exists to diagnose.

    A missing entry point is the same fact as a version mismatch, and is
    reported the same way: named, with the remedy, as an unusable
    library.
    """

    import ctypes

    stale = tmp_path / "static_fields.dll"
    stale.write_bytes(b"not read: ctypes is stubbed below")
    monkeypatch.setattr(rust_bridge, "_LIBRARY", None)
    monkeypatch.setattr(rust_bridge, "resolve_static_bridge",
                        lambda: stale)
    monkeypatch.setattr(ctypes, "CDLL",
                        lambda *_a, **_k: _StaleLibrary(rust_bridge.STATIC_ABI))

    with pytest.raises(rust_bridge.StaticBridgeError) as excinfo:
        rust_bridge.load()
    message = str(excinfo.value)
    assert "gpuwm_static_highres_resample" in message, (
        f"the refusal must name the entry point that is missing: {message}")
    assert "fetch-bridges" in message, "the refusal must name the remedy"

    # And the reason a caller reads is that refusal, not an exception
    # class no caller catches.
    monkeypatch.setattr(rust_bridge, "_LIBRARY", None)
    reason = rust_bridge.unavailable_reason()
    assert reason and "gpuwm_static_highres_resample" in reason
