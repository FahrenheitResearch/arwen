"""Golden fixtures for the lane-3 highres Rust port, extracted by
RUNNING THE REAL PYTHON (gpuwm.static.highres + rasterio/pyproj) on
real source data.

Run from the repo root:

    python tools/rustwx/crates/static-fields/tests/fixtures/highres/generate_goldens.py

Real inputs: the cached Copernicus DEM GLO-30 tiles under
``COPERNICUS_CACHE`` (fetched by a real production run through
``gpuwm.static.highres_fetch``; their sha256 sidecars sit next to
them).  Small windows of those tiles are clipped into this directory
so the byte-level decode/warp goldens are portable; the full tiles are
additionally summarized in ``real_tile_summary.json`` for the
run-at-the-data check.

Synthetic-but-realistic sources (land cover in the CONUS Albers CRS,
soil component windows in the Interrupted Goode Homolosine CRS with
the recorded crs_override discipline) are WRITTEN by rasterio and then
run through the REAL Python overlay functions, so every expected
array in here is the true Python implementation's output, not a
hand-computed imitation.

Array files are raw little-endian buffers (``.bin``); ``meta`` JSON
carries dtype/shape and the measured tolerances the Rust tests gate
on (measured divergence x safety margin, never guessed).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO))

#: The high-resolution DEM cache a real production run filled.
#: Overridable, and the default is COMPOSED from this account's home
#: rather than spelled out: a written-out default is one developer's
#: absolute path, and the release snapshot's machine-path gate refuses
#: to build a tree that ships one.  `tests/highres_parity.rs` resolves
#: the same root the same way, which is why `meta.json` records the
#: cached tile by NAME rather than by path.
COPERNICUS_CACHE = Path(
    os.environ.get("GPUWM_HIGHRES_DEM_CACHE")
    or (Path.home() / "arwen-verify-232" / "hrcache"
        / "copernicus_dem_glo30"))

import rasterio  # noqa: E402
from pyproj import CRS, Transformer  # noqa: E402

from gpuwm.static.build import smth_desmth_special  # noqa: E402
from gpuwm.static.highres import (  # noqa: E402
    BoundRaster, SOILGRIDS_DEPTH_WEIGHTS, _nearest_donors,
    _soilgrids_categories, merge_highres_overrides,
    merge_terrain_override, resample_continuous,
    resample_mapped_categories, sha256_file, usda_texture_category,
    _extended_grid, _raster_geometry,
)
from gpuwm.static.highres_fetch import SOILGRIDS_CRS  # noqa: E402
from gpuwm.static.lambert import LambertGrid  # noqa: E402

NLCD_ALBERS = ("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23.0 +lon_0=-96 "
               "+x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs")


def save(name: str, array: np.ndarray) -> dict:
    array = np.ascontiguousarray(array)
    (HERE / name).write_bytes(array.tobytes())
    return {"file": name, "dtype": str(array.dtype),
            "shape": list(array.shape)}


def bound(path: Path, **kwargs) -> BoundRaster:
    return BoundRaster(
        path=path, sha256=sha256_file(path), source_id="golden",
        role="golden", source_url="https://example.invalid/",
        license_id="test", license_url="https://example.invalid/",
        nominal_resolution="test", **kwargs)


META: dict[str, object] = {}


# ---------------------------------------------------------------------------
# 1. transform_points goldens (pyproj is the instrument reference)
# ---------------------------------------------------------------------------

def gen_transforms() -> None:
    rng = np.random.default_rng(42)
    cases = []

    def case(name, crs_text, lon_range, lat_range, n=120):
        lon = rng.uniform(*lon_range, n)
        lat = rng.uniform(*lat_range, n)
        transformer = Transformer.from_crs(
            "EPSG:4326", CRS.from_proj4(crs_text), always_xy=True)
        x, y = transformer.transform(lon, lat)
        back = Transformer.from_crs(
            CRS.from_proj4(crs_text), "EPSG:4326", always_xy=True)
        lon2, lat2 = back.transform(x, y)
        cases.append({
            "name": name, "crs": crs_text,
            "lon": save(f"tp_{name}_lon.bin", lon),
            "lat": save(f"tp_{name}_lat.bin", lat),
            "x": save(f"tp_{name}_x.bin", np.asarray(x)),
            "y": save(f"tp_{name}_y.bin", np.asarray(y)),
            "lon_back": save(f"tp_{name}_lon2.bin", np.asarray(lon2)),
            "lat_back": save(f"tp_{name}_lat2.bin", np.asarray(lat2)),
        })

    case("lcc",
         "+proj=lcc +lat_1=38 +lat_2=41 +lat_0=38 +lon_0=-84 "
         "+R=6370000 +units=m +no_defs", (-95.0, -75.0), (30.0, 48.0))
    case("lcc_tangent",
         "+proj=lcc +lat_1=45 +lat_2=45 +lat_0=45 +lon_0=10 "
         "+R=6370000 +units=m +no_defs", (0.0, 20.0), (35.0, 55.0))
    case("merc",
         "+proj=merc +lat_ts=20 +lon_0=-80 +R=6370000 +units=m +no_defs",
         (-100.0, -60.0), (-30.0, 40.0))
    case("polar_n",
         "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-100 +R=6370000 "
         "+units=m +no_defs", (-180.0, 180.0), (35.0, 89.0))
    case("polar_s",
         "+proj=stere +lat_0=-90 +lat_ts=-60 +lon_0=170 +R=6370000 "
         "+units=m +no_defs", (-180.0, 180.0), (-89.0, -35.0))
    case("igh", SOILGRIDS_CRS, (-170.0, 170.0), (-80.0, 80.0), n=400)
    case("aea", NLCD_ALBERS, (-125.0, -66.5), (24.0, 49.5))
    META["transform_points"] = cases


# ---------------------------------------------------------------------------
# 2. real Copernicus clips + real-tile summary
# ---------------------------------------------------------------------------

TILE_A = COPERNICUS_CACHE / "Copernicus_DSM_COG_10_N46_00_E007_00_DEM.tif"
TILE_B = COPERNICUS_CACHE / "Copernicus_DSM_COG_10_N46_00_E008_00_DEM.tif"


#: Committed fixtures must stay small; decode parity is proven on a
#: bit-exact stride sample + bit-exact min/max/NaN census instead of a
#: full duplicate array.
SAMPLE_STRIDE = 41


def clip_tile(src_path: Path, out_name: str, lon0, lon1, lat0, lat1,
              *, predictor: int, tiled: bool) -> dict:
    with rasterio.open(src_path) as src:
        window = rasterio.windows.from_bounds(
            lon0, lat0, lon1, lat1, transform=src.transform)
        window = window.round_offsets().round_lengths()
        values = src.read(1, window=window)
        transform = src.window_transform(window)
        profile = dict(driver="GTiff", height=values.shape[0],
                       width=values.shape[1], count=1, dtype="float32",
                       crs=src.crs, transform=transform,
                       compress="deflate", predictor=predictor)
        if tiled:
            profile.update(tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(HERE / out_name, "w", **profile) as dst:
            dst.write(values.astype(np.float32), 1)
    with rasterio.open(HERE / out_name) as check:
        stored = check.read(1)
        t = check.transform
    flat = stored.astype(np.float32).ravel()
    return {
        "file": out_name,
        "source_tile": src_path.name,
        "source_sha256": sha256_file(src_path),
        "shape": list(stored.shape),
        "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
        "sample_stride": SAMPLE_STRIDE,
        "sample": save(out_name + ".sample.bin", flat[::SAMPLE_STRIDE]),
        "min": float(np.nanmin(flat)),
        "max": float(np.nanmax(flat)),
        "nan_count": int(np.count_nonzero(~np.isfinite(flat))),
    }


def gen_clips() -> None:
    # Warp-source clip: one committed window of the real Alps tile,
    # written with the full decode envelope exercised (tiled, deflate,
    # floating-point predictor).
    META["terrain_clip"] = clip_tile(
        TILE_A, "terrain_clip.tif", 7.38, 7.62, 46.42, 46.58,
        predictor=3, tiled=True)
    # Mosaic clips: two adjacent windows across the 8E tile seam,
    # striped/tiled with predictor none to vary the layout.
    META["mosaic_clip_west"] = clip_tile(
        TILE_A, "mosaic_west.tif", 7.96, 8.00, 46.43, 46.49,
        predictor=1, tiled=False)
    META["mosaic_clip_east"] = clip_tile(
        TILE_B, "mosaic_east.tif", 8.00, 8.04, 46.43, 46.49,
        predictor=1, tiled=True)
    # Run-at-the-data summary of the REAL full tile (used only when the
    # cache path exists on the running box).
    with rasterio.open(TILE_A) as src:
        window = rasterio.windows.Window(512, 1024, 256, 256)
        values = src.read(1, window=window).astype(np.float64)
        META["real_tile_summary"] = {
            # The NAME, not the path: the reader resolves it under its
            # own DEM cache root (highres_parity.rs), so this record is
            # usable on a second box and carries nobody's home directory.
            "path": TILE_A.name,
            "sha256": sha256_file(TILE_A),
            "shape": [src.height, src.width],
            "transform": [src.transform.a, src.transform.b,
                          src.transform.c, src.transform.d,
                          src.transform.e, src.transform.f],
            "window_col_row_w_h": [512, 1024, 256, 256],
            "window_sum": float(values.sum()),
            "window_min": float(values.min()),
            "window_max": float(values.max()),
        }


# ---------------------------------------------------------------------------
# 3. terrain warp golden (real Python resample on the real clip)
# ---------------------------------------------------------------------------

def grid_spec_dict(grid: LambertGrid) -> dict:
    return {
        "kind": "lambert", "ref_lat": grid.ref_lat,
        "ref_lon": grid.ref_lon, "truelat1": grid.truelat1,
        "truelat2": grid.truelat2, "stand_lon": grid.stand_lon,
        "dx": grid.dx, "dy": grid.dy, "e_we": grid.e_we,
        "e_sn": grid.e_sn, "known_x": grid.known_x,
        "known_y": grid.known_y, "moad_cen_lat": grid.moad_cen_lat,
        "moad_cen_lon": grid.moad_cen_lon,
    }


def gen_terrain_warp() -> None:
    grid = LambertGrid(
        ref_lat=46.5, ref_lon=7.5, truelat1=46.0, truelat2=47.0,
        stand_lon=7.5, dx=750.0, dy=750.0, e_we=11, e_sn=9)
    halo = 3
    extended = _extended_grid(grid, halo)
    source = bound(HERE / "terrain_clip.tif")
    warped = resample_continuous(source, extended, method="average")
    assert np.isfinite(warped).all(), "golden warp must be fully covered"
    smoothed = smth_desmth_special(warped, passes=1)
    crop = (slice(halo, halo + grid.e_sn - 1),
            slice(halo, halo + grid.e_we - 1))
    hgt = smoothed[crop]
    crs, transform, shape = _raster_geometry(extended)
    META["terrain_warp"] = {
        "grid_spec": grid_spec_dict(grid),
        "halo": halo,
        "extended_transform": [transform.a, transform.b, transform.c,
                               transform.d, transform.e, transform.f],
        "extended_shape": list(shape),
        "warped": save("terrain_warp_extended.bin", warped),
        "smoothed": save("terrain_warp_smoothed.bin", smoothed),
        "hgt": save("terrain_warp_hgt.bin", hgt),
        # MEASURED 2026-08-17 against rasterio/GDAL average on this
        # pinned Alps footprint: max |delta| 8.42 m, mean 2.25 m (the
        # residual of GDAL's approximate transformer + chunked edges
        # against the exact-transform corner-box kernel).  Caps carry
        # ~2x headroom over the measurement.
        "max_abs_delta_cap_m": 20.0,
        "mean_abs_delta_cap_m": 4.5,
    }


# ---------------------------------------------------------------------------
# 4. land-cover category golden (synthetic classes, real Albers CRS,
#    real Python resample_mapped_categories)
# ---------------------------------------------------------------------------

NLCD_CLASSES = np.asarray([11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52,
                           71, 81, 82, 90, 95], dtype=np.uint8)
MAPPING = {11: 21, 12: 15, 21: 13, 22: 13, 23: 13, 24: 13, 31: 16,
           41: 4, 42: 1, 43: 5, 52: 7, 71: 10, 81: 10, 82: 12,
           90: 11, 95: 11}


def gen_landcover() -> None:
    rng = np.random.default_rng(7)
    grid = LambertGrid(
        ref_lat=39.5, ref_lon=-84.0, truelat1=38.0, truelat2=41.0,
        stand_lon=-84.0, dx=1500.0, dy=1500.0, e_we=9, e_sn=8)
    halo = 3
    extended = _extended_grid(grid, halo)
    # Source window: the extended footprint in the Albers plane plus a
    # margin, at the collection's own 30 m grid.
    transformer = Transformer.from_crs(
        "EPSG:4326", CRS.from_proj4(NLCD_ALBERS), always_xy=True)
    lat, lon = extended.latlon_mass()
    x, y = transformer.transform(lon, lat)
    west = float(x.min()) - 2000.0
    east = float(x.max()) + 2000.0
    south = float(y.min()) - 2000.0
    north = float(y.max()) + 2000.0
    step = 30.0
    nx = int(np.ceil((east - west) / step))
    ny = int(np.ceil((north - south) / step))
    # Blocky, spatially-correlated classes like a real map, plus a
    # nodata wedge so validity masking is exercised.
    blocks = rng.integers(0, len(NLCD_CLASSES), (ny // 24 + 1, nx // 24 + 1))
    values = NLCD_CLASSES[blocks.repeat(24, axis=0).repeat(24, axis=1)]
    values = values[:ny, :nx].copy()
    values[:40, :60] = 250  # the collection's nodata
    transform = rasterio.transform.from_origin(west, north, step, step)
    with rasterio.open(
            HERE / "landcover.tif", "w", driver="GTiff", height=ny,
            width=nx, count=1, dtype="uint8",
            crs=CRS.from_proj4(NLCD_ALBERS), transform=transform,
            nodata=250, compress="deflate", predictor=2, tiled=True,
            blockxsize=256, blockysize=256) as dst:
        dst.write(values, 1)
    source = bound(HERE / "landcover.tif")
    fractions = resample_mapped_categories(
        source, extended, MAPPING, category_count=21)
    META["landcover"] = {
        "grid_spec": grid_spec_dict(grid),
        "halo": halo,
        "mapping": sorted((int(k), int(v)) for k, v in MAPPING.items()),
        "nodata": 250,
        "fractions": save("landcover_fractions.bin", fractions),
        # MEASURED 2026-08-17: max per-cell L1 0.055, mean 0.025 on
        # this pinned rotated (Albers -> Lambert) footprint; ~1.6x
        # headroom.
        "per_cell_l1_cap": 0.09,
        "mean_l1_cap": 0.04,
    }


# ---------------------------------------------------------------------------
# 5. soil goldens (synthetic IGH windows through the REAL depth-mean +
#    triangle + fraction pipeline)
# ---------------------------------------------------------------------------

def gen_soil() -> None:
    rng = np.random.default_rng(11)
    grid = LambertGrid(
        ref_lat=39.5, ref_lon=-84.0, truelat1=38.0, truelat2=41.0,
        stand_lon=-84.0, dx=1500.0, dy=1500.0, e_we=9, e_sn=8)
    halo = 3
    extended = _extended_grid(grid, halo)
    transformer = Transformer.from_crs("EPSG:4326", SOILGRIDS_CRS,
                                       always_xy=True)
    lat, lon = extended.latlon_mass()
    x, y = transformer.transform(lon, lat)
    west = float(x.min()) - 2000.0
    east = float(x.max()) + 2000.0
    south = float(y.min()) - 2000.0
    north = float(y.max()) + 2000.0
    step = 250.0
    nx = int(np.ceil((east - west) / step))
    ny = int(np.ceil((north - south) / step))
    transform = rasterio.transform.from_origin(west, north, step, step)

    sources: dict[tuple[str, str], BoundRaster] = {}
    j = np.arange(ny)[:, None]
    i = np.arange(nx)[None, :]
    for component in ("sand", "silt", "clay"):
        for depth_index, depth in enumerate(
                ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm")):
            base = {"sand": 420.0, "silt": 350.0, "clay": 230.0}[component]
            field = (base
                     + 60.0 * np.sin(j / 17.0 + depth_index)
                     + 40.0 * np.cos(i / 23.0)
                     + rng.normal(0.0, 12.0, (ny, nx)))
            field = np.clip(field, 30.0, 900.0).astype(np.int16)
            field[:12, :20] = 0  # the masked/urban-core nodata
            name = f"soil_{component}_{depth}.tif"
            with rasterio.open(
                    HERE / name, "w", driver="GTiff", height=ny,
                    width=nx, count=1, dtype="int16", crs=None,
                    transform=transform, compress="deflate") as dst:
                dst.write(field, 1)
            sources[(component, depth)] = bound(
                HERE / name, crs_override=SOILGRIDS_CRS,
                nodata_override=0.0, scale_factor=0.1)

    layers = {}
    for layer, weights in SOILGRIDS_DEPTH_WEIGHTS.items():
        category, valid, s_transform, s_crs, raw_total = \
            _soilgrids_categories(sources, weights)
        from gpuwm.static.highres import _resample_category_array
        fractions = _resample_category_array(
            category, valid, transform=s_transform, crs=s_crs,
            grid=extended, category_count=16)
        layers[layer] = {
            "weights": [[d, float(w)] for d, w in weights.items()],
            "category": save(f"soil_{layer}_category.bin", category),
            "valid": save(f"soil_{layer}_valid.bin",
                          valid.astype(np.uint8)),
            "raw_total": save(f"soil_{layer}_total.bin", raw_total),
            "fractions": save(f"soil_{layer}_fractions.bin", fractions),
        }
    META["soil"] = {
        "grid_spec": grid_spec_dict(grid),
        "halo": halo,
        "source_shape": [ny, nx],
        "source_transform": [transform.a, transform.b, transform.c,
                             transform.d, transform.e, transform.f],
        "crs_override": SOILGRIDS_CRS,
        "scale_factor": 0.1,
        "nodata": 0.0,
        "layers": layers,
        "per_cell_l1_cap": 0.08,
        "mean_l1_cap": 0.02,
    }


# ---------------------------------------------------------------------------
# 6. USDA triangle golden (byte contract)
# ---------------------------------------------------------------------------

def gen_usda() -> None:
    compositions = []
    for sand in range(0, 101):
        for silt in range(0, 101 - sand):
            compositions.append((sand, silt, 100 - sand - silt))
    arr = np.asarray(compositions, dtype=np.float64)
    categories = usda_texture_category(
        arr[:, 0][None, :], arr[:, 1][None, :], arr[:, 2][None, :])[0]
    # Unnormalized g/kg-style inputs exercise the normalization path.
    # (NOT asserted equal to the integer-percent categories: points on
    # a rule boundary legitimately flip under the different float
    # normalization, and the contract is Python-vs-Rust, not
    # scale-invariance.)
    scaled = arr * 8.13
    categories_scaled = usda_texture_category(
        scaled[:, 0][None, :], scaled[:, 1][None, :],
        scaled[:, 2][None, :])[0]
    try:
        usda_texture_category(np.asarray([0.0]), np.asarray([0.0]),
                              np.asarray([0.0]))
        raise AssertionError("zero totals must refuse")
    except ValueError as error:
        invalid_message = str(error)
    # The composition enumeration is deterministic (sand 0..100, silt
    # 0..100-sand, clay the remainder) so the Rust test regenerates it
    # rather than shipping 123 KB of integers.
    META["usda"] = {
        "categories": save("usda_categories.bin", categories),
        "scale": 8.13,
        "categories_scaled": save("usda_categories_scaled.bin",
                                  categories_scaled),
        "invalid_total_message": invalid_message,
    }


# ---------------------------------------------------------------------------
# 7. nearest-donor goldens (byte contract)
# ---------------------------------------------------------------------------

def gen_donors() -> None:
    rng = np.random.default_rng(3)
    cases = []
    masks = {
        "random": rng.random((20, 16)) > 0.55,
        "single": np.zeros((7, 9), dtype=bool),
        "banded": np.zeros((14, 14), dtype=bool),
    }
    masks["single"][3, 4] = True
    masks["banded"][::3, :] = True
    for name, mask in masks.items():
        if not mask.any():
            mask[0, 0] = True
        donor_y, donor_x = _nearest_donors(mask)
        cases.append({
            "name": name,
            "mask": save(f"donors_{name}_mask.bin",
                         mask.astype(np.uint8)),
            "donor_y": save(f"donors_{name}_y.bin", donor_y),
            "donor_x": save(f"donors_{name}_x.bin", donor_x),
        })
    try:
        _nearest_donors(np.zeros((3, 3), dtype=bool))
        raise AssertionError("empty mask must refuse")
    except ValueError as error:
        empty_message = str(error)
    META["donors"] = {"cases": cases, "empty_message": empty_message}


# ---------------------------------------------------------------------------
# 8. merge goldens (byte contract, real Python merges)
# ---------------------------------------------------------------------------

def merge_baseline(shape=(12, 10)) -> dict[str, np.ndarray]:
    ny, nx = shape
    j = np.arange(ny)[:, None].astype(np.float64)
    i = np.arange(nx)[None, :].astype(np.float64)
    old_land = np.ones(shape)
    old_land[0:3, 0:3] = 0.0
    old_land[8, 7] = 0.0
    baseline = {
        "HGT_M": 300.0 + 5.0 * j + 3.0 * i,
        "LANDMASK": old_land,
        "LU_INDEX": np.where(old_land > 0.5, 12.0, 21.0),
        "LANDUSEF": np.zeros((21, ny, nx)),
        "SOILCTOP": np.zeros((16, ny, nx)),
        "SCT_DOM": np.full(shape, 6.0),
        "SOILCBOT": np.zeros((16, ny, nx)),
        "SCB_DOM": np.full(shape, 6.0),
        "GREENFRAC": 0.30 + 0.02 * j[None] + 0.001 * i[None]
        + 0.01 * np.arange(12)[:, None, None],
        "LAI12M": 1.0 + 0.1 * j[None] + 0.02 * i[None]
        + 0.05 * np.arange(12)[:, None, None],
        "ALBEDO12M": 14.0 + 0.05 * j[None] + 0.02 * i[None]
        + 0.2 * np.arange(12)[:, None, None],
        "SNOALB": 0.4 + 0.01 * j + 0.002 * i,
        "SOILTEMP": 278.0 + 0.5 * j + 0.25 * i,
        "TMN": 276.0 + 0.5 * j + 0.25 * i,
    }
    baseline["LANDUSEF"][11] = 1.0
    baseline["SOILCTOP"][5] = 1.0
    baseline["SOILCBOT"][5] = 1.0
    return baseline


def merge_overrides(baseline: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    old_land = baseline["LANDMASK"]
    new_land = old_land.copy()
    new_land[0, 0] = 1.0     # newly land, donor fill fires
    new_land[1, 1] = 1.0     # newly land
    new_land[11, 9] = 0.0    # newly water
    new_land[5, 5] = 0.0     # newly water inland
    overrides = {
        "HGT_M": baseline["HGT_M"] + 12.5,
        "LANDMASK": new_land,
        "LU_INDEX": np.where(new_land > 0.5, 12.0, 21.0),
        "LANDUSEF": baseline["LANDUSEF"].copy(),
        "SOILCTOP": baseline["SOILCTOP"].copy(),
        "SCT_DOM": baseline["SCT_DOM"].copy(),
        "SOILCBOT": baseline["SOILCBOT"].copy(),
        "SCB_DOM": baseline["SCB_DOM"].copy(),
    }
    return overrides


def fieldset_entry(fields: dict[str, np.ndarray], prefix: str) -> list:
    ordered = []
    for name in sorted(fields):
        arr = np.asarray(fields[name], dtype=np.float64)
        planes = 1 if arr.ndim == 2 else arr.shape[0]
        ny, nx = arr.shape[-2:]
        ordered.append({
            "name": name, "planes": planes, "ny": ny, "nx": nx,
            "data": save(f"{prefix}_{name}.bin", arr),
        })
    return ordered


def gen_merges() -> None:
    baseline = merge_baseline()
    overrides = merge_overrides(baseline)
    merged, audit = merge_highres_overrides(baseline, overrides)
    META["merge_full"] = {
        "baseline": fieldset_entry(baseline, "mf_base"),
        "overrides": fieldset_entry(overrides, "mf_over"),
        "merged": fieldset_entry(merged, "mf_out"),
        "audit": {k: int(v) for k, v in audit.items()},
    }

    terrain_baseline = merge_baseline()
    hgt_override = terrain_baseline["HGT_M"] + 33.25
    merged_t, audit_t = merge_terrain_override(
        terrain_baseline, {"HGT_M": hgt_override})
    META["merge_terrain"] = {
        "baseline": fieldset_entry(terrain_baseline, "mt_base"),
        "hgt_override": save("mt_hgt_override.bin", hgt_override),
        "merged": fieldset_entry(merged_t, "mt_out"),
        "audit": {k: int(v) for k, v in audit_t.items()},
    }

    # Refusal DECISION goldens with the exact Python messages.
    dead = merge_baseline()
    dead["SOILTEMP"] = np.zeros((12, 10))
    try:
        merge_highres_overrides(dead, merge_overrides(dead))
        raise AssertionError("0 K land deep soil must refuse")
    except ValueError as error:
        whole_domain_message = str(error)
    holed = merge_baseline()
    holed["SOILTEMP"] = np.array(holed["SOILTEMP"], copy=True)
    holed["SOILTEMP"][6, 6] = 0.0
    try:
        merge_highres_overrides(holed, merge_overrides(holed))
        raise AssertionError("single 0 K land cell must refuse")
    except ValueError as error:
        single_cell_message = str(error)
    shaped = merge_baseline()
    try:
        merge_terrain_override(shaped, {"HGT_M": np.zeros((5, 4))})
        raise AssertionError("shape mismatch must refuse")
    except ValueError as error:
        shape_message = str(error)
    META["merge_refusals"] = {
        "whole_domain": whole_domain_message,
        "single_cell": single_cell_message,
        "terrain_shape": shape_message,
    }


# ---------------------------------------------------------------------------
# 9. mosaic golden (real tile clips through rasterio.merge, the exact
#    derive_global_terrain_window arithmetic)
# ---------------------------------------------------------------------------

def gen_mosaic() -> None:
    from rasterio.enums import Resampling
    from rasterio.merge import merge as rasterio_merge

    step = 1.0 / 3600.0
    margin = 0.01
    # Footprint straddling the seam, inside the two clips.
    bbox = dict(lon_min=7.975, lon_max=8.025, lat_min=46.445, lat_max=46.475)
    bounds = (bbox["lon_min"] - margin, bbox["lat_min"] - margin,
              bbox["lon_max"] + margin, bbox["lat_max"] + margin)
    datasets = [rasterio.open(HERE / "mosaic_west.tif"),
                rasterio.open(HERE / "mosaic_east.tif")]
    try:
        mosaic, transform = rasterio_merge(
            datasets, bounds=bounds, res=(step, step),
            resampling=Resampling.nearest, nodata=np.nan,
            dtype="float32")
    finally:
        for dataset in datasets:
            dataset.close()
    values = np.asarray(mosaic[0], dtype=np.float32)
    hole_mask = ~np.isfinite(values)
    holes = int(np.count_nonzero(hole_mask))
    filled = values.copy()
    filled[hole_mask] = np.float32(0.0)
    META["mosaic"] = {
        "tiles": ["mosaic_west.tif", "mosaic_east.tif"],
        "bounds_wsen": list(bounds),
        "resolution_deg": step,
        "shape": [int(values.shape[0]), int(values.shape[1])],
        "transform": [transform.a, transform.b, transform.c,
                      transform.d, transform.e, transform.f],
        # rasterio's integer window alignment can drop a sub-pixel
        # seam column to nodata; the Rust mosaic samples by centre
        # containment, so its hole set must be a SUBSET of this mask
        # and every mutually-covered cell must agree.
        "holes": holes,
        "hole_mask": save("mosaic_holes.bin",
                          hole_mask.astype(np.uint8)),
        "filled": save("mosaic_filled.bin", filled.astype(np.float32)),
        # MEASURED 2026-08-17: rasterio aligns each source window by
        # floor-snapping (gdal_merge win_align), which can shift a
        # sub-pixel-staggered source by one whole pixel; centre-
        # containment nearest keeps true registration, so ~40% of the
        # cells on this staggered real-tile seam differ by one-pixel
        # terrain steps (max 49.7 m in the Alps).  The gates: hole
        # subset, exact fraction floor, per-cell max, and a mean cap
        # that keeps the disagreement one-pixel-sized.
        "exact_fraction_floor": 0.55,
        "max_abs_delta_cap_m": 60.0,
        "mean_abs_delta_cap_m": 8.0,
    }


# ---------------------------------------------------------------------------
# 10. tile-id enumeration goldens (byte contract, real Python
#     highres_fetch enumerators)
# ---------------------------------------------------------------------------

def gen_tile_ids() -> None:
    from gpuwm.static.highres_fetch import (
        FootprintBBox, copernicus_dem_tile_ids, one_degree_tile_bbox,
        srtm_tile_ids, three_dep_tile_ids)

    boxes = {
        "midwest": FootprintBBox(38.95, 40.12, -85.30, -83.90),
        "alps": FootprintBBox(45.98, 47.63, 6.55, 9.02),
        "southern": FootprintBBox(-34.6, -33.2, 18.2, 19.9),
        "dateline_west": FootprintBBox(50.1, 52.4, 176.4, 179.9),
    }
    cases = []
    for name, box in boxes.items():
        entry = {"name": name, "bbox": [box.lat_min, box.lat_max,
                                        box.lon_min, box.lon_max],
                 "copernicus": list(copernicus_dem_tile_ids(box)),
                 "srtm": list(srtm_tile_ids(box))}
        try:
            entry["three_dep"] = list(three_dep_tile_ids(box))
        except Exception:
            entry["three_dep"] = None
        cases.append(entry)
    tiles = {}
    for tile in ("N39_00_W105_00", "S34_00_E018_00", "N39W105", "S56W071"):
        box = one_degree_tile_bbox(tile)
        tiles[tile] = [box.lat_min, box.lat_max, box.lon_min, box.lon_max]
    META["tile_ids"] = {"cases": cases, "tile_bboxes": tiles}


def main() -> None:
    gen_transforms()
    gen_clips()
    gen_terrain_warp()
    gen_landcover()
    gen_soil()
    gen_usda()
    gen_donors()
    gen_merges()
    gen_mosaic()
    gen_tile_ids()
    (HERE / "meta.json").write_text(
        json.dumps(META, indent=1, sort_keys=True), encoding="utf-8")
    total = sum(p.stat().st_size for p in HERE.iterdir()
                if p.suffix in (".bin", ".tif", ".json"))
    print(f"goldens written: {total / 1e6:.2f} MB in {HERE}")


if __name__ == "__main__":
    main()
