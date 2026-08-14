"""Provenance-bound high-resolution static geography overlays.

This module is deliberately separate from :mod:`gpuwm.static.build`.  The
established builder remains the WPS/geogrid-compatible 30-arc-second path;
this module provides an explicit, auditable opt-in for GeoTIFF sources.  It
never downloads data and it verifies every source hash before opening it.

The first supported pack is intentionally narrow: bare-earth terrain,
categorical land cover, and SoilGrids sand/silt/clay predictions.  Existing
WPS climatologies remain authoritative for LAI, green fraction, albedo, snow
albedo, and deep-soil temperature.  When a higher-resolution water mask turns
an old water cell into land, those climatologies are filled from the nearest
old-land cell and the exact fallback count is reported.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

from .build import (
    HALO,
    dominant_category,
    landmask_from_landusef,
    lu_index_from_landusef,
    smth_desmth_special,
)
from .lambert import EARTH_RADIUS_M, LambertGrid


NLCD_TO_MODIS21_INLAND = {
    11: 21,  # open water -> inland lake for the scoped CONUS pilot
    12: 15,  # perennial snow / ice
    21: 13, 22: 13, 23: 13, 24: 13,  # developed intensity classes
    31: 16,  # barren
    41: 4,   # deciduous broadleaf forest
    42: 1,   # evergreen needleleaf forest
    43: 5,   # mixed forest
    52: 7,   # open shrubland
    71: 10,  # grassland / herbaceous
    81: 10,  # pasture / hay
    82: 12,  # cultivated crops
    90: 11, 95: 11,  # woody and herbaceous wetlands
}

SOILGRIDS_DEPTH_WEIGHTS = {
    "top_0_30cm": {"0-5cm": 5.0, "5-15cm": 10.0, "15-30cm": 15.0},
    "bottom_30_100cm": {"30-60cm": 30.0, "60-100cm": 40.0},
}


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for ``path``."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BoundRaster:
    """One immutable raster source and the provenance needed to use it."""

    path: Path
    sha256: str
    source_id: str
    role: str
    source_url: str
    license_id: str
    license_url: str
    nominal_resolution: str
    expected_bytes: int | None = None
    reference_year: int | None = None
    crs_override: str | None = None
    nodata_override: float | None = None
    scale_factor: float = 1.0

    def verify(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"high-resolution raster missing: {self.path}")
        if (self.expected_bytes is not None
                and self.path.stat().st_size != self.expected_bytes):
            raise ValueError(
                f"high-resolution raster size mismatch for {self.path}: "
                f"expected {self.expected_bytes}, observed "
                f"{self.path.stat().st_size}"
            )
        observed = sha256_file(self.path)
        if observed != self.sha256:
            raise ValueError(
                f"high-resolution raster hash mismatch for {self.path}: "
                f"expected {self.sha256}, observed {observed}"
            )

    def receipt(self) -> dict[str, object]:
        payload = asdict(self)
        payload["path"] = str(self.path.resolve())
        payload["observed_bytes"] = self.path.stat().st_size
        return payload


@contextmanager
def _open_verified(source: BoundRaster):
    source.verify()
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise RuntimeError(
            "high-resolution geography requires the optional 'geog' extra"
        ) from exc
    with rasterio.open(source.path) as dataset:
        yield dataset


def _grid_crs(grid):
    """PROJ CRS for one projected grid on WPS's spherical earth."""
    try:
        from pyproj import CRS
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise RuntimeError(
            "high-resolution geography requires the optional 'geog' extra"
        ) from exc
    map_proj = getattr(grid, "map_proj", "lambert")
    if map_proj == "lambert":
        proj4 = (
            f"+proj=lcc +lat_1={grid.truelat1:.15g} "
            f"+lat_2={grid.truelat2:.15g} +lat_0={grid.truelat1:.15g} "
            f"+lon_0={grid.stand_lon:.15g} +R={EARTH_RADIUS_M:.15g} "
            "+units=m +no_defs")
    elif map_proj == "mercator":
        # module_llxy Mercator is anchored at the known point's
        # longitude, not stand_lon.
        proj4 = (
            f"+proj=merc +lat_ts={grid.truelat1:.15g} "
            f"+lon_0={grid.ref_lon:.15g} +R={EARTH_RADIUS_M:.15g} "
            "+units=m +no_defs")
    elif map_proj == "polar":
        pole = 90.0 if grid.truelat1 >= 0.0 else -90.0
        proj4 = (
            f"+proj=stere +lat_0={pole:.15g} "
            f"+lat_ts={grid.truelat1:.15g} "
            f"+lon_0={grid.stand_lon:.15g} +R={EARTH_RADIUS_M:.15g} "
            "+units=m +no_defs")
    else:
        raise NotImplementedError(
            f"no CRS mapping for map_proj {map_proj!r}")
    return CRS.from_proj4(proj4)


#: Backward-compatible name (predates the worldwide projections).
_lambert_crs = _grid_crs


def _raster_geometry(grid):
    """Return north-first raster geometry for WRF mass points."""

    try:
        from affine import Affine
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise RuntimeError(
            "high-resolution geography requires the optional 'geog' extra"
        ) from exc

    crs = _grid_crs(grid)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    ref_x, ref_y = transformer.transform(grid.ref_lon, grid.ref_lat)
    nx, ny = grid.e_we - 1, grid.e_sn - 1
    west_center = ref_x - (grid.known_x - 1.0) * grid.dx
    south_center = ref_y - (grid.known_y - 1.0) * grid.dy
    west_edge = west_center - 0.5 * grid.dx
    north_edge = south_center + (ny - 0.5) * grid.dy
    transform = Affine(grid.dx, 0.0, west_edge,
                       0.0, -grid.dy, north_edge)
    return crs, transform, (ny, nx)


def _extended_grid(grid, halo: int):
    if halo < 0:
        raise ValueError("halo must be non-negative")
    if halo == 0:
        return grid
    return type(grid)(
        grid.ref_lat, grid.ref_lon, grid.truelat1, grid.truelat2,
        grid.stand_lon, grid.dx, grid.dy,
        grid.e_we + 2 * halo, grid.e_sn + 2 * halo,
        known_x=grid.known_x + halo, known_y=grid.known_y + halo,
        moad_cen_lat=grid.moad_cen_lat, moad_cen_lon=grid.moad_cen_lon,
    )


def _source_crs(dataset, source: BoundRaster):
    try:
        from pyproj import CRS
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "high-resolution geography requires the optional 'geog' extra"
        ) from exc
    value = dataset.crs if dataset.crs is not None else source.crs_override
    if value is None:
        raise ValueError(
            f"raster {source.path} has no CRS and declares no crs_override"
        )
    return CRS.from_user_input(value)


def resample_continuous(source: BoundRaster, grid: LambertGrid, *,
                        method: str = "average") -> np.ndarray:
    """Reproject one continuous raster to mass points in south-north order."""

    try:
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "high-resolution geography requires the optional 'geog' extra"
        ) from exc
    methods = {
        "average": Resampling.average,
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest,
    }
    if method not in methods:
        raise ValueError(f"unsupported continuous resampling method {method!r}")

    dst_crs, dst_transform, shape = _raster_geometry(grid)
    destination = np.full(shape, np.nan, dtype=np.float64)
    with _open_verified(source) as dataset:
        values = dataset.read(1).astype(np.float64) * source.scale_factor
        nodata = (source.nodata_override
                  if source.nodata_override is not None else dataset.nodata)
        reproject(
            source=values,
            destination=destination,
            src_transform=dataset.transform,
            src_crs=_source_crs(dataset, source),
            src_nodata=(None if nodata is None else nodata * source.scale_factor),
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=methods[method],
            init_dest_nodata=True,
        )
    return destination[::-1].copy()


def _resample_category_array(values: np.ndarray, valid: np.ndarray, *,
                             transform, crs, grid: LambertGrid,
                             category_count: int) -> np.ndarray:
    """Area fractions for already-classified source pixels."""

    try:
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "high-resolution geography requires the optional 'geog' extra"
        ) from exc

    values = np.asarray(values, dtype=np.int16)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("category values and validity mask shapes differ")
    dst_crs, dst_transform, shape = _raster_geometry(grid)
    fractions = np.zeros((category_count, *shape), dtype=np.float64)
    coverage = np.full(shape, np.nan, dtype=np.float64)
    valid_float = np.where(valid, 1.0, -9999.0).astype(np.float32)
    reproject(
        source=valid_float,
        destination=coverage,
        src_transform=transform,
        src_crs=crs,
        src_nodata=-9999.0,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.average,
        init_dest_nodata=True,
    )
    for category in np.unique(values[valid]):
        category = int(category)
        if category < 1 or category > category_count:
            raise ValueError(
                f"mapped category {category} is outside 1..{category_count}"
            )
        indicator = np.where(
            valid, values == category, -9999.0).astype(np.float32)
        destination = np.full(shape, np.nan, dtype=np.float64)
        reproject(
            source=indicator,
            destination=destination,
            src_transform=transform,
            src_crs=crs,
            src_nodata=-9999.0,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
            init_dest_nodata=True,
        )
        fractions[category - 1] = np.nan_to_num(
            destination, nan=0.0)
    fractions = fractions[:, ::-1, :].copy()
    coverage = coverage[::-1].copy()
    total = fractions.sum(axis=0)
    covered = np.isfinite(coverage) & (coverage > 0.0) & (total > 0.0)
    fractions[:, covered] /= total[covered]
    fractions[:, ~covered] = np.nan
    return fractions


def resample_mapped_categories(
        source: BoundRaster, grid: LambertGrid,
        mapping: Mapping[int, int], *, category_count: int) -> np.ndarray:
    """Map raw categories then compute target-cell area fractions."""

    with _open_verified(source) as dataset:
        raw = dataset.read(1)
        nodata = (source.nodata_override
                  if source.nodata_override is not None else dataset.nodata)
        valid = np.ones(raw.shape, dtype=bool)
        if nodata is not None:
            valid &= raw != nodata
        valid &= np.isfinite(raw)
        observed = {int(value) for value in np.unique(raw[valid])}
        unknown = sorted(observed - {int(value) for value in mapping})
        if unknown:
            raise ValueError(
                f"raster {source.path} contains unmapped categories {unknown}"
            )
        mapped = np.zeros(raw.shape, dtype=np.int16)
        for source_category, target_category in mapping.items():
            mapped[raw == int(source_category)] = int(target_category)
        valid &= mapped > 0
        return _resample_category_array(
            mapped, valid, transform=dataset.transform,
            crs=_source_crs(dataset, source), grid=grid,
            category_count=category_count,
        )


def usda_texture_category(sand: np.ndarray, silt: np.ndarray,
                          clay: np.ndarray) -> np.ndarray:
    """Map normalized percentages to WRF's USDA soil categories 1..12."""

    sand = np.asarray(sand, dtype=np.float64)
    silt = np.asarray(silt, dtype=np.float64)
    clay = np.asarray(clay, dtype=np.float64)
    if sand.shape != silt.shape or sand.shape != clay.shape:
        raise ValueError("sand, silt, and clay shapes differ")
    total = sand + silt + clay
    if np.any(~np.isfinite(total)) or np.any(total <= 0.0):
        raise ValueError("soil texture contains invalid totals")
    sand, silt, clay = sand / total * 100.0, silt / total * 100.0, clay / total * 100.0
    out = np.zeros(sand.shape, dtype=np.int16)

    rules = (
        (1, silt + 1.5 * clay < 15.0),
        (2, (silt + 1.5 * clay >= 15.0) & (silt + 2.0 * clay < 30.0)),
        (3, (((clay >= 7.0) & (clay < 20.0) & (sand > 52.0)
              & (silt + 2.0 * clay >= 30.0))
             | ((clay < 7.0) & (silt < 50.0)
                & (silt + 2.0 * clay >= 30.0)))),
        (4, (((silt >= 50.0) & (clay >= 12.0) & (clay < 27.0))
             | ((silt >= 50.0) & (silt < 80.0) & (clay < 12.0)))),
        (5, (silt >= 80.0) & (clay < 12.0)),
        (6, (clay >= 7.0) & (clay < 27.0) & (silt >= 28.0)
         & (silt < 50.0) & (sand <= 52.0)),
        (7, (clay >= 20.0) & (clay < 35.0) & (silt < 28.0)
         & (sand > 45.0)),
        (8, (clay >= 27.0) & (clay < 40.0) & (sand <= 20.0)),
        (9, (clay >= 27.0) & (clay < 40.0) & (sand > 20.0)
         & (sand <= 45.0)),
        (10, (clay >= 35.0) & (sand > 45.0)),
        (11, (clay >= 40.0) & (silt >= 40.0)),
        (12, (clay >= 40.0) & (sand <= 45.0) & (silt < 40.0)),
    )
    for category, condition in rules:
        out[(out == 0) & condition] = category
    if np.any(out == 0):
        j, i = np.argwhere(out == 0)[0]
        raise ValueError(
            "USDA texture rules left an unclassified point: "
            f"sand={sand[j, i]:.3f}, silt={silt[j, i]:.3f}, "
            f"clay={clay[j, i]:.3f}"
        )
    return out


def _soilgrids_categories(
        sources: Mapping[tuple[str, str], BoundRaster],
        depth_weights: Mapping[str, float]):
    expected = {
        (component, depth)
        for component in ("sand", "silt", "clay")
        for depth in depth_weights
    }
    missing = sorted(expected - set(sources))
    if missing:
        raise KeyError(f"missing SoilGrids sources: {missing}")

    arrays: dict[tuple[str, str], np.ndarray] = {}
    transform = crs = shape = None
    for key in sorted(expected):
        source = sources[key]
        with _open_verified(source) as dataset:
            if shape is None:
                shape = dataset.shape
                transform = dataset.transform
                crs = _source_crs(dataset, source)
            elif (dataset.shape != shape or dataset.transform != transform
                  or _source_crs(dataset, source) != crs):
                raise ValueError("SoilGrids source rasters are not co-registered")
            value = dataset.read(1).astype(np.float64) * source.scale_factor
            nodata = (source.nodata_override if source.nodata_override is not None
                      else dataset.nodata)
            if nodata is not None:
                value[dataset.read(1) == nodata] = np.nan
            arrays[key] = value

    weights = np.asarray(list(depth_weights.values()), dtype=np.float64)
    means = {}
    for component in ("sand", "silt", "clay"):
        stack = np.stack([
            arrays[(component, depth)] for depth in depth_weights
        ])
        valid = np.all(np.isfinite(stack), axis=0)
        mean = np.full(shape, np.nan, dtype=np.float64)
        mean[valid] = np.average(stack[:, valid], axis=0, weights=weights)
        means[component] = mean
    valid = np.all(np.stack([np.isfinite(value) for value in means.values()]),
                   axis=0)
    category = np.zeros(shape, dtype=np.int16)
    category[valid] = usda_texture_category(
        means["sand"][valid][None, :],
        means["silt"][valid][None, :],
        means["clay"][valid][None, :],
    )[0]
    raw_total = means["sand"] + means["silt"] + means["clay"]
    return category, valid, transform, crs, raw_total


def _require_coverage(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        count = int(np.count_nonzero(~np.isfinite(values)))
        raise ValueError(f"high-resolution {name} lacks {count} target values")


def build_highres_overrides(
        grid: LambertGrid, *, terrain: BoundRaster,
        landcover: BoundRaster,
        soil_sources: Mapping[tuple[str, str], BoundRaster],
        soil_fallback: Mapping[str, np.ndarray] | None = None,
        landcover_mapping: Mapping[int, int] = NLCD_TO_MODIS21_INLAND,
        halo: int = HALO) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build terrain, land-use, and soil fields for one Lambert domain."""

    extended = _extended_grid(grid, halo)
    crop = (slice(halo, halo + grid.e_sn - 1),
            slice(halo, halo + grid.e_we - 1))

    terrain_extended = resample_continuous(terrain, extended, method="average")
    _require_coverage("terrain", terrain_extended)
    terrain_extended = smth_desmth_special(terrain_extended, passes=1)
    hgt = terrain_extended[crop]

    luf_extended = resample_mapped_categories(
        landcover, extended, landcover_mapping, category_count=21)
    _require_coverage("land cover", luf_extended)
    luf = luf_extended[(slice(None),) + crop]
    landmask = landmask_from_landusef(luf, iswater=17, islake=21)
    lu_index = lu_index_from_landusef(
        luf, landmask, iswater=17, islake=21)

    soil_fields: dict[str, np.ndarray] = {}
    soil_audit = {}
    for layer_name, weights in SOILGRIDS_DEPTH_WEIGHTS.items():
        category, valid, transform, crs, raw_total = _soilgrids_categories(
            soil_sources, weights)
        fractions = _resample_category_array(
            category, valid, transform=transform, crs=crs,
            grid=extended, category_count=16,
        )[(slice(None),) + crop]
        water = landmask == 0.0
        fractions[:, water] = 0.0
        fractions[13, water] = 1.0  # WRF soil category 14 = water
        land = ~water
        missing_land = land & (
            ~np.all(np.isfinite(fractions), axis=0)
            | (np.nansum(fractions, axis=0) <= 0.0))
        fallback_name = (
            "SOILCTOP" if layer_name == "top_0_30cm" else "SOILCBOT")
        if np.any(missing_land):
            if soil_fallback is None or fallback_name not in soil_fallback:
                raise ValueError(
                    f"SoilGrids {layer_name} lacks coverage over "
                    f"{int(missing_land.sum())} target land cells and no "
                    f"{fallback_name} fallback was supplied")
            fallback = np.asarray(soil_fallback[fallback_name],
                                  dtype=np.float64)
            if fallback.shape != fractions.shape:
                raise ValueError(
                    f"{fallback_name} fallback shape {fallback.shape} differs "
                    f"from {fractions.shape}")
            fractions[:, missing_land] = fallback[:, missing_land]
        fractions[:, land] /= fractions[:, land].sum(axis=0)
        if layer_name == "top_0_30cm":
            soil_fields["SOILCTOP"] = fractions
            soil_fields["SCT_DOM"] = dominant_category(fractions)
        else:
            soil_fields["SOILCBOT"] = fractions
            soil_fields["SCB_DOM"] = dominant_category(fractions)
        totals = raw_total[valid]
        soil_audit[layer_name] = {
            "raw_component_total_percent_min": float(totals.min()),
            "raw_component_total_percent_max": float(totals.max()),
            "valid_source_pixels": int(np.count_nonzero(valid)),
            "fallback_land_cells": int(missing_land.sum()),
        }

    fields = {
        "HGT_M": hgt,
        "LANDUSEF": luf,
        "LANDMASK": landmask,
        "LU_INDEX": lu_index,
        **soil_fields,
    }
    audit = {
        "method": (
            "hash-bound GeoTIFF reprojection to the WRF spherical "
            f"{getattr(grid, 'map_proj', 'lambert')} grid; "
            "area-average continuous/categorical fractions; one WPS "
            "smooth-desmooth terrain pass"
        ),
        "halo_cells": halo,
        "terrain": {
            "min_m": float(hgt.min()),
            "max_m": float(hgt.max()),
            "mean_m": float(hgt.mean()),
        },
        "land_fraction": float(landmask.mean()),
        "soil": soil_audit,
        "sources": [
            terrain.receipt(), landcover.receipt(),
            *[source.receipt() for _, source in sorted(soil_sources.items())],
        ],
    }
    return fields, audit


def build_terrain_override(grid: LambertGrid, *, terrain: BoundRaster,
                           halo: int = HALO
                           ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build ONLY the terrain field for one domain, from one bound raster.

    This is the same terrain science :func:`build_highres_overrides` runs --
    area-average onto the halo-extended WPS grid, one WPS smooth-desmooth
    pass, crop -- with the land-cover and soil legs left out entirely.  It
    exists because terrain is the one high-resolution field with a
    near-global source: outside the United States there is no wired land
    cover, and a terrain-only replacement is a real, useful product as long
    as nobody is led to believe they also got high-resolution land use.

    Because no land-use rule runs, this path never classifies water and
    therefore never needs the ocean/lake distinction that constrains the
    full overlay.
    """
    extended = _extended_grid(grid, halo)
    crop = (slice(halo, halo + grid.e_sn - 1),
            slice(halo, halo + grid.e_we - 1))
    terrain_extended = resample_continuous(terrain, extended, method="average")
    _require_coverage("terrain", terrain_extended)
    terrain_extended = smth_desmth_special(terrain_extended, passes=1)
    hgt = terrain_extended[crop]
    audit = {
        "method": (
            "hash-bound GeoTIFF reprojection to the WRF spherical "
            f"{getattr(grid, 'map_proj', 'lambert')} grid; area-average; "
            "one WPS smooth-desmooth terrain pass (terrain only -- land "
            "use and soil are untouched)"),
        "halo_cells": halo,
        "terrain": {"min_m": float(hgt.min()), "max_m": float(hgt.max()),
                    "mean_m": float(hgt.mean())},
        "sources": [terrain.receipt()],
    }
    return {"HGT_M": hgt}, audit


def merge_terrain_override(
        baseline: Mapping[str, np.ndarray],
        overrides: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray],
                                                      dict[str, int]]:
    """Replace terrain only; recompute TMN; leave the land/water mask alone.

    TMN is not an independent field: the baseline's deep-soil temperature
    was lapsed to the baseline's terrain height.  Swapping HGT_M without
    recomputing it would leave the two disagreeing by 6.5 K per kilometre
    of terrain change, so the same lapse the full path applies is applied
    here.  Nothing else moves, so there is no newly-land climatology fill
    to do and none is reported.
    """
    required = {"HGT_M", "LANDMASK", "SOILTEMP"}
    missing = sorted(required - set(baseline))
    if missing:
        raise KeyError(f"baseline static fields missing {missing}")
    if "HGT_M" not in overrides:
        raise KeyError("terrain-only overrides missing ['HGT_M']")
    extra = sorted(set(overrides) - {"HGT_M"})
    if extra:
        raise KeyError(
            f"terrain-only overrides carry non-terrain field(s) {extra}; "
            "use merge_highres_overrides for the full overlay")

    out = {name: np.array(value, copy=True)
           for name, value in baseline.items()}
    out["HGT_M"] = np.array(overrides["HGT_M"], copy=True)
    if out["HGT_M"].shape != np.asarray(baseline["HGT_M"]).shape:
        raise ValueError(
            f"terrain override shape {out['HGT_M'].shape} differs from "
            f"baseline {np.asarray(baseline['HGT_M']).shape}")
    land = np.asarray(baseline["LANDMASK"]) > 0.5
    out["TMN"] = np.where(land, out["SOILTEMP"] - 0.0065 * out["HGT_M"],
                          out["SOILTEMP"])
    for name, value in out.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise ValueError(f"merged terrain-only field {name} is non-finite")
    audit = {
        "terrain_cells_changed": int(np.count_nonzero(
            np.asarray(baseline["HGT_M"]) != out["HGT_M"])),
        "land_water_cells_unchanged": int(land.size),
        "newly_land_nearest_climatology_fallback_cells": 0,
        "newly_water_masked_cells": 0,
    }
    return out, audit


def _nearest_donors(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic Manhattan-nearest donor indices for every 2-D cell."""

    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2 or not valid.any():
        raise ValueError("nearest-donor mask must be 2-D with a valid cell")
    ny, nx = valid.shape
    donor_y = np.full((ny, nx), -1, dtype=np.int32)
    donor_x = np.full((ny, nx), -1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()
    for y, x in np.argwhere(valid):
        donor_y[y, x], donor_x[y, x] = y, x
        queue.append((int(y), int(x)))
    while queue:
        y, x = queue.popleft()
        for yy, xx in ((y - 1, x), (y, x - 1),
                       (y, x + 1), (y + 1, x)):
            if (0 <= yy < ny and 0 <= xx < nx and donor_y[yy, xx] < 0):
                donor_y[yy, xx] = donor_y[y, x]
                donor_x[yy, xx] = donor_x[y, x]
                queue.append((yy, xx))
    return donor_y, donor_x


#: The physical envelope a deep-soil temperature must sit in to be a
#: temperature at all.  Same band ``gpuwm/ingest/soil.py`` admits TMN and
#: TSK in, so the producer and the consumer refuse the same thing.
_DEEP_SOIL_KELVIN_RANGE = (170.0, 400.0)

#: Deep-soil fields whose water cells carry a mask, not a measurement.
_LAND_ONLY_DEEP_SOIL = ("SOILTEMP", "TMN")


def _refuse_unusable_merged_statics(out, new_land) -> None:
    """Refuse merged statics that are unusable ON LAND, by name and count.

    The gate this replaces asked ``np.isfinite(value).all()`` of every
    field.  A water fill of 0.0 satisfies it, and so does a whole-domain
    deep-soil decode failure that leaves 0 K on land: the two are the
    same bytes, which is exactly why one hid behind the other.  A
    domain-wide TMN of 0 K removes the seasonal thermal reservoir under
    the soil column and biases the surface energy balance for the whole
    forecast, and it used to arrive with an empty receipt.

    Water cells of the deep-soil pair are allowed to be the masked
    sentinel and nothing else; land cells of every field must be finite;
    land cells of the deep-soil pair must additionally be a temperature.
    """
    land = np.asarray(new_land, dtype=bool)
    for name, value in out.items():
        if not isinstance(value, np.ndarray) or not value.size:
            continue
        if not np.issubdtype(value.dtype, np.floating):
            continue
        broadcast_land = np.broadcast_to(
            land if value.ndim == 2 else land[None, ...], value.shape)
        land_cells = int(np.count_nonzero(broadcast_land))
        holed = broadcast_land & ~np.isfinite(value)
        if holed.any():
            raise ValueError(
                f"merged high-resolution field {name} is non-finite on "
                f"{int(np.count_nonzero(holed))} land cell(s) of "
                f"{land_cells} -- the high-resolution mask resolves land "
                "the source field does not cover")
        if name not in _LAND_ONLY_DEEP_SOIL:
            if not np.isfinite(value).all():
                raise ValueError(
                    f"merged high-resolution field {name} is non-finite")
            continue
        low, high = _DEEP_SOIL_KELVIN_RANGE
        unusable = broadcast_land & ~((value >= low) & (value <= high))
        if unusable.any():
            count = int(np.count_nonzero(unusable))
            sample = np.asarray(value)[unusable]
            raise ValueError(
                f"merged high-resolution {name} is not a temperature on "
                f"{count} land cell(s) of {land_cells}: range "
                f"[{np.nanmin(sample):.6g}, {np.nanmax(sample):.6g}] K "
                f"outside {low:g}..{high:g}.  0 K is the geog "
                "soil_temperature fill, so this is a deep-soil source "
                "that did not decode over the land this domain resolves "
                "-- check the geog soil_temperature tile coverage for "
                "this footprint")


def merge_highres_overrides(
        baseline: Mapping[str, np.ndarray],
        overrides: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray],
                                                       dict[str, int]]:
    """Return a complete Noah-static state with explicit fallback counts."""

    required = {
        "HGT_M", "LANDUSEF", "LANDMASK", "LU_INDEX", "SOILCTOP",
        "SCT_DOM", "SOILCBOT", "SCB_DOM", "GREENFRAC", "LAI12M",
        "ALBEDO12M", "SNOALB", "SOILTEMP",
    }
    missing = sorted(required - set(baseline))
    if missing:
        raise KeyError(f"baseline static fields missing {missing}")
    missing = sorted({
        "HGT_M", "LANDUSEF", "LANDMASK", "LU_INDEX", "SOILCTOP",
        "SCT_DOM", "SOILCBOT", "SCB_DOM",
    } - set(overrides))
    if missing:
        raise KeyError(f"high-resolution overrides missing {missing}")

    out = {name: np.array(value, copy=True)
           for name, value in baseline.items()}
    for name, value in overrides.items():
        out[name] = np.array(value, copy=True)

    old_land = np.asarray(baseline["LANDMASK"]) > 0.5
    new_land = out["LANDMASK"] > 0.5
    newly_land = new_land & ~old_land
    newly_water = ~new_land & old_land
    donor_y, donor_x = _nearest_donors(old_land)
    yy, xx = donor_y[newly_land], donor_x[newly_land]
    fills = {
        "GREENFRAC": 0.0,
        "LAI12M": 0.0,
        "ALBEDO12M": 8.0,
        "SNOALB": 0.0,
        # A MASKED SENTINEL, not a temperature.  Filling water deep-soil
        # with 0.0 made a manufactured 0 K indistinguishable from a real
        # one, so the finiteness gate below -- and every count downstream
        # -- read a whole-domain deep-soil decode failure as ordinary
        # water.  NaN says "no deep-soil information here", which is the
        # truth about a water cell, and lets the gate ask the only
        # question that matters: is every LAND cell a real temperature?
        "SOILTEMP": np.nan,
    }
    for name, water_fill in fills.items():
        value = np.array(baseline[name], copy=True, dtype=np.float64)
        if value.ndim == 3:
            value[:, newly_land] = value[:, yy, xx]
            value[:, ~new_land] = water_fill
        elif value.ndim == 2:
            value[newly_land] = value[yy, xx]
            value[~new_land] = water_fill
        else:
            raise ValueError(f"unsupported Noah static rank for {name}")
        out[name] = value
    out["TMN"] = np.where(
        new_land, out["SOILTEMP"] - 0.0065 * out["HGT_M"],
        out["SOILTEMP"])
    _refuse_unusable_merged_statics(out, new_land)
    # The sentinel does NOT cross this return.  geo_em's on-disk
    # convention for a land-masked field is 0.0 over water, and three
    # consumers read the merged TMN with no land mask of their own --
    # ``gpuwm/runtime.py`` hands it straight to ``initialize_physics``
    # as the Noah driver's ``tmn`` field.  So the sentinel does its work
    # inside the gate, where the question is asked, and the returned
    # arrays keep the convention every reader already expects.  What
    # changed is that a land leak can no longer hide behind it.
    water_masked = 0
    for name in ("SOILTEMP", "TMN"):
        sentinel = ~np.isfinite(out[name])
        water_masked = max(water_masked, int(np.count_nonzero(sentinel)))
        out[name] = np.where(sentinel, 0.0, out[name])
    audit = {
        "newly_land_nearest_climatology_fallback_cells": int(newly_land.sum()),
        "newly_water_masked_cells": int(newly_water.sum()),
        "unchanged_land_water_cells": int((old_land == new_land).sum()),
        # The water cells whose deep-soil temperature is a mask rather
        # than a measurement.  Named so a reader of the receipt can tell
        # the 0.0 in SOILTEMP/TMN from a temperature.
        "deep_soil_water_masked_cells": water_masked,
    }
    return out, audit


__all__ = [
    "BoundRaster", "NLCD_TO_MODIS21_INLAND", "SOILGRIDS_DEPTH_WEIGHTS",
    "build_highres_overrides", "build_terrain_override",
    "merge_highres_overrides", "merge_terrain_override",
    "resample_continuous", "resample_mapped_categories", "sha256_file",
    "usda_texture_category",
]
