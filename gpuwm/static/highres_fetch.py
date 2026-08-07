"""Tiled, cached, provenance-recorded fetch for high-resolution geography.

This module turns one model-domain footprint (plus the geogrid processing
halo) into the concrete source artifacts :mod:`gpuwm.static.highres`
consumes: USGS 3DEP 1/3 arc-second terrain tiles, one Annual NLCD land-cover
year, and SoilGrids v2 sand/silt/clay WCS windows.  It is deliberately
footprint-parametric -- nothing in here knows about any particular case or
place; every geographic number arrives from the caller's grid.

Contract:

- Sources that publish whole artifacts are fetched whole.  3DEP is fetched
  as complete 1x1-degree staged GeoTIFF tiles and Annual NLCD as the
  complete published CONUS year bundle; neither is ever range-subset from
  the network.  SoilGrids is served by ISRIC's own WCS windowing service,
  which is the pilot-proven route for that source.
- Every fetched byte is hashed (SHA-256) at fetch time and the digest is
  recorded in a JSON sidecar next to the cached payload; receipts carry
  those digests.  Arbitrary user windows cannot be pre-pinned, so recorded
  provenance -- not a pinned manifest -- is the contract.
- A cache hit is a payload whose sidecar exists and whose byte count
  matches the sidecar.  Anything else is refetched (resumable ``.partial``
  staging, atomic rename).
- Incomplete tile coverage refuses loudly, naming the missing tiles
  (:class:`CoverageError`).  Transport failures other than the source
  saying "absent" raise plainly; they are infrastructure faults, not
  coverage facts, and must never be converted into a silent fallback.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from .highres import sha256_file

#: Read/stream chunk for downloads and hashing.
_CHUNK = 8 * 1024 * 1024

#: USGS 3DEP seamless 1/3 arc-second staged products (whole 1x1-degree
#: GeoTIFF tiles, public domain).
THREE_DEP_TILE_URL = ("https://prd-tnm.s3.amazonaws.com/StagedProducts/"
                      "Elevation/13/TIFF/current/{tile}/USGS_13_{tile}.tif")
THREE_DEP_SOURCE_URL = "https://www.usgs.gov/3d-elevation-program"
THREE_DEP_LICENSE = ("US-PD",
                     "https://www.usgs.gov/information-policies-and-"
                     "instructions/copyrights-and-credits")

#: Annual NLCD Collection 1 land cover, one whole published CONUS year
#: bundle per fetch (public domain).
ANNUAL_NLCD_URL = ("https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/"
                   "data-bundles/Annual_NLCD_LndCov_{year}_CU_C1V1.zip")
ANNUAL_NLCD_SOURCE_URL = "https://www.mrlc.gov/data"
ANNUAL_NLCD_LICENSE = ("US-PD", "https://www.mrlc.gov/data")
#: Years the Annual NLCD collection publishes.  Cases before the first
#: year take the earliest map and the receipt names the anachronism.
ANNUAL_NLCD_FIRST_YEAR = 1985
ANNUAL_NLCD_LAST_YEAR = 2024

#: SoilGrids v2 250 m WCS (ISRIC; CC-BY-4.0).  The pilot-proven route.
SOILGRIDS_WCS_URL = (
    "https://maps.isric.org/mapserv?map=/map/{component}.map"
    "&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
    "&COVERAGEID={component}_{depth}_Q0.5&FORMAT=GEOTIFF_INT16"
    "&SUBSET=X({x0:.0f},{x1:.0f})&SUBSET=Y({y0:.0f},{y1:.0f})"
    "&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/152160"
    "&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/152160")
SOILGRIDS_SOURCE_URL = "https://www.isric.org/explore/soilgrids"
SOILGRIDS_LICENSE = ("CC-BY-4.0",
                     "https://creativecommons.org/licenses/by/4.0/")
#: Interrupted Goode Homolosine, SoilGrids' native grid ("EPSG" 152160 is
#: ISRIC's own registry entry; the delivered GeoTIFF carries no CRS tag,
#: hence the override recorded on every bound raster).
SOILGRIDS_CRS = "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"
SOILGRIDS_COMPONENTS = ("clay", "sand", "silt")
SOILGRIDS_DEPTHS = ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm")
#: SoilGrids raw units are g/kg; the USDA triangle wants percent.
SOILGRIDS_SCALE = 0.1
#: The WCS delivers masked (water/ice/urban-core) pixels as 0 g/kg, which
#: is not a physical soil composition; treat it as the nodata sentinel.
SOILGRIDS_NODATA = 0.0
#: Fetch margin around the model footprint, metres on the IGH plane.
_SOILGRIDS_MARGIN_M = 2000.0
#: Snap WCS windows to whole kilometres so repeated preparations of the
#: same domain hit the cache instead of minting near-duplicate windows.
_SOILGRIDS_SNAP_M = 1000.0


class CoverageError(RuntimeError):
    """A source does not cover the requested footprint; names what is
    missing."""


@dataclass(frozen=True)
class FootprintBBox:
    """Geographic bounding box of one model domain plus processing halo."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def __post_init__(self):
        if not (math.isfinite(self.lat_min) and math.isfinite(self.lat_max)
                and math.isfinite(self.lon_min)
                and math.isfinite(self.lon_max)):
            raise ValueError(f"footprint bbox is not finite: {self}")
        if self.lat_min >= self.lat_max or self.lon_min >= self.lon_max:
            raise ValueError(f"footprint bbox is degenerate: {self}")

    def padded(self, degrees: float) -> "FootprintBBox":
        return FootprintBBox(self.lat_min - degrees, self.lat_max + degrees,
                             self.lon_min - degrees, self.lon_max + degrees)

    def contains(self, other: "FootprintBBox") -> bool:
        return (self.lat_min <= other.lat_min
                and self.lat_max >= other.lat_max
                and self.lon_min <= other.lon_min
                and self.lon_max >= other.lon_max)

    def as_dict(self) -> dict[str, float]:
        return {"lat_min": self.lat_min, "lat_max": self.lat_max,
                "lon_min": self.lon_min, "lon_max": self.lon_max}


def domain_footprint(grid, halo: int, margin_deg: float = 0.03
                     ) -> FootprintBBox:
    """Geographic bbox of the grid extended by ``halo`` cells + margin.

    Uses the cell-corner mesh of the extended grid (the same support the
    static builder samples) so the fetched sources cover every source pixel
    any halo cell can accumulate.
    """
    if halo < 0:
        raise ValueError("halo must be non-negative")
    nx, ny = grid.e_we - 1, grid.e_sn - 1
    xc, yc = np.meshgrid(
        np.arange(0.5 - halo, nx + halo + 1.0, dtype=np.float64),
        np.arange(0.5 - halo, ny + halo + 1.0, dtype=np.float64))
    lat, lon = grid.ij_to_latlon(xc, yc)
    return FootprintBBox(
        float(np.min(lat)), float(np.max(lat)),
        float(np.min(lon)), float(np.max(lon))).padded(margin_deg)


@dataclass(frozen=True)
class FetchedFile:
    """One cached payload with fetch-time provenance."""

    path: Path
    url: str
    sha256: str
    bytes: int
    fetched_utc: str
    cache_hit: bool

    def receipt(self) -> dict[str, object]:
        return {
            "path": str(Path(self.path).resolve()),
            "url": self.url,
            "sha256": self.sha256,
            "bytes": int(self.bytes),
            "fetched_utc": self.fetched_utc,
            "cache_hit": bool(self.cache_hit),
        }


class SourceAbsent(RuntimeError):
    """The source authoritatively reports the artifact does not exist."""


class RangeExhausted(RuntimeError):
    """A resume offset sits at/after the payload end (HTTP 416)."""


def _default_urlopen(url: str, offset: int):
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        return urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        if error.code in (403, 404):
            # S3 buckets report absent keys as 403 without list permission;
            # both codes mean "this artifact is not published here".
            raise SourceAbsent(f"{url} -> HTTP {error.code}") from error
        if error.code == 416 and offset:
            raise RangeExhausted(f"{url} -> HTTP 416 at offset {offset}") \
                from error
        raise


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256.json")


def _read_sidecar(path: Path) -> dict | None:
    sidecar = _sidecar(path)
    if not path.is_file() or not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("bytes") != path.stat().st_size:
        return None
    if not isinstance(payload.get("sha256"), str):
        return None
    return payload


def _write_sidecar(path: Path, payload: dict) -> None:
    sidecar = _sidecar(path)
    temporary = sidecar.with_name(sidecar.name + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, sidecar)


def record_local_artifact(path: Path, *, url: str,
                          cache_hit: bool = False) -> FetchedFile:
    """Hash one locally produced payload and write its sidecar."""
    path = Path(path)
    digest = sha256_file(path)
    record = FetchedFile(
        path=path, url=url, sha256=digest, bytes=path.stat().st_size,
        fetched_utc=datetime.now(timezone.utc).isoformat(),
        cache_hit=cache_hit)
    _write_sidecar(path, {
        "url": url, "sha256": digest, "bytes": record.bytes,
        "fetched_utc": record.fetched_utc})
    return record


def fetch_file(url: str, path: Path, *, urlopen=None) -> FetchedFile:
    """Fetch ``url`` to ``path`` (cached, resumable, hashed at fetch)."""
    path = Path(path)
    cached = _read_sidecar(path)
    if cached is not None:
        return FetchedFile(
            path=path, url=str(cached.get("url", url)),
            sha256=cached["sha256"], bytes=int(cached["bytes"]),
            fetched_utc=str(cached.get("fetched_utc", "")), cache_hit=True)

    opener = _default_urlopen if urlopen is None else urlopen
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    offset = partial.stat().st_size if partial.is_file() else 0
    try:
        response = opener(url, offset)
    except RangeExhausted:
        # The staged partial already holds the complete payload; promote
        # it.  The recorded digest still reflects the exact local bytes.
        os.replace(partial, path)
        return record_local_artifact(path, url=url)
    status = int(getattr(response, "status", 200) or 200)
    mode = "ab" if (offset and status == 206) else "wb"
    with response, partial.open(mode) as stream:
        while True:
            block = response.read(_CHUNK)
            if not block:
                break
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return record_local_artifact(path, url=url)


# ---------------------------------------------------------------------------
# USGS 3DEP terrain tiles
# ---------------------------------------------------------------------------

def three_dep_tile_ids(bbox: FootprintBBox) -> tuple[str, ...]:
    """1x1-degree staged-tile ids covering ``bbox`` (n..w.. naming).

    Tile ``n40w084`` covers latitudes [39, 40] x longitudes [-84, -83].
    Only the northern/western quadrant is enumerable because that is where
    the jointly declared sources exist; anything else must already have
    been refused by the coverage gate.
    """
    if bbox.lat_min < 0.0 or bbox.lon_max > 0.0:
        raise CoverageError(
            "3DEP staged 1/3 arc-second tiles are enumerated for the "
            f"northern/western quadrant only; footprint {bbox.as_dict()} "
            "leaves it")
    tiles = []
    for north in range(int(math.floor(bbox.lat_min)) + 1,
                       int(math.ceil(bbox.lat_max)) + 1):
        for west_edge in range(int(math.floor(bbox.lon_min)),
                               int(math.ceil(bbox.lon_max))):
            tiles.append(f"n{north:02d}w{-west_edge:03d}")
    return tuple(tiles)


def fetch_three_dep_tiles(bbox: FootprintBBox, cache_root: Path, *,
                          urlopen=None) -> tuple[FetchedFile, ...]:
    """Fetch every whole 3DEP tile covering ``bbox``; refuse on gaps."""
    cache = Path(cache_root) / "usgs3dep_13as"
    fetched: list[FetchedFile] = []
    missing: list[str] = []
    for tile in three_dep_tile_ids(bbox):
        url = THREE_DEP_TILE_URL.format(tile=tile)
        try:
            fetched.append(
                fetch_file(url, cache / f"USGS_13_{tile}.tif",
                           urlopen=urlopen))
        except SourceAbsent:
            missing.append(tile)
    if missing:
        raise CoverageError(
            "USGS 3DEP 1/3 arc-second coverage is incomplete for footprint "
            f"{bbox.as_dict()}: missing staged tile(s) {missing} "
            f"(checked {THREE_DEP_TILE_URL.format(tile='<tile>')})")
    return tuple(fetched)


# ---------------------------------------------------------------------------
# Annual NLCD land cover
# ---------------------------------------------------------------------------

def nlcd_year_for(case_date: date) -> tuple[int, int]:
    """(published year nearest the case date, anachronism in years)."""
    year = min(max(int(case_date.year), ANNUAL_NLCD_FIRST_YEAR),
               ANNUAL_NLCD_LAST_YEAR)
    return year, abs(int(case_date.year) - year)


def fetch_annual_nlcd(year: int, cache_root: Path, *, urlopen=None
                      ) -> tuple[FetchedFile, FetchedFile]:
    """Fetch one whole Annual NLCD year bundle; return (zip, extracted tif).

    The published artifact is a zip around one CONUS GeoTIFF; both the
    bundle exactly as fetched and the extracted raster are hashed and
    sidecar-recorded, so the receipt can bind the raster actually decoded
    back to the bytes actually downloaded.
    """
    if not (ANNUAL_NLCD_FIRST_YEAR <= int(year) <= ANNUAL_NLCD_LAST_YEAR):
        raise CoverageError(
            f"Annual NLCD publishes {ANNUAL_NLCD_FIRST_YEAR}.."
            f"{ANNUAL_NLCD_LAST_YEAR}; there is no year {year}")
    cache = Path(cache_root) / "annual_nlcd"
    url = ANNUAL_NLCD_URL.format(year=int(year))
    try:
        bundle = fetch_file(url, cache / Path(url).name, urlopen=urlopen)
    except SourceAbsent as error:
        raise CoverageError(
            f"Annual NLCD year {year} is not published at {url}") from error

    members: list[str] = []
    with zipfile.ZipFile(bundle.path) as archive:
        members = [name for name in archive.namelist()
                   if name.lower().endswith(".tif")]
        if len(members) != 1:
            raise ValueError(
                f"Annual NLCD bundle {bundle.path} contains "
                f"{len(members)} .tif members ({members}); expected one")
        raster_path = cache / Path(members[0]).name
        cached = _read_sidecar(raster_path)
        if cached is None:
            partial = raster_path.with_name(raster_path.name + ".partial")
            with archive.open(members[0]) as source, \
                    partial.open("wb") as target:
                while True:
                    block = source.read(_CHUNK)
                    if not block:
                        break
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
            os.replace(partial, raster_path)
            raster = record_local_artifact(
                raster_path, url=f"{url}!{members[0]}")
        else:
            raster = FetchedFile(
                path=raster_path, url=str(cached.get("url", url)),
                sha256=cached["sha256"], bytes=int(cached["bytes"]),
                fetched_utc=str(cached.get("fetched_utc", "")),
                cache_hit=True)
    return bundle, raster


# ---------------------------------------------------------------------------
# SoilGrids v2 WCS windows
# ---------------------------------------------------------------------------

def _soilgrids_window_m(bbox: FootprintBBox) -> tuple[float, float,
                                                      float, float]:
    """Snap the footprint to a whole-km window on SoilGrids' IGH plane."""
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise RuntimeError(
            "high-resolution geography requires the optional 'geog' extra"
        ) from exc
    transformer = Transformer.from_crs("EPSG:4326", SOILGRIDS_CRS,
                                       always_xy=True)
    lons = np.linspace(bbox.lon_min, bbox.lon_max, 25)
    lats = np.linspace(bbox.lat_min, bbox.lat_max, 25)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    x, y = transformer.transform(grid_lon, grid_lat)
    snap = _SOILGRIDS_SNAP_M
    x0 = math.floor((float(np.min(x)) - _SOILGRIDS_MARGIN_M) / snap) * snap
    x1 = math.ceil((float(np.max(x)) + _SOILGRIDS_MARGIN_M) / snap) * snap
    y0 = math.floor((float(np.min(y)) - _SOILGRIDS_MARGIN_M) / snap) * snap
    y1 = math.ceil((float(np.max(y)) + _SOILGRIDS_MARGIN_M) / snap) * snap
    return x0, x1, y0, y1


def fetch_soilgrids(bbox: FootprintBBox, cache_root: Path, *, urlopen=None
                    ) -> dict[tuple[str, str], FetchedFile]:
    """Fetch SoilGrids Q0.5 windows for every component x depth."""
    x0, x1, y0, y1 = _soilgrids_window_m(bbox)
    key = f"x{x0:.0f}_{x1:.0f}_y{y0:.0f}_{y1:.0f}"
    cache = Path(cache_root) / "soilgrids_v2"
    out: dict[tuple[str, str], FetchedFile] = {}
    for component in SOILGRIDS_COMPONENTS:
        for depth in SOILGRIDS_DEPTHS:
            url = SOILGRIDS_WCS_URL.format(
                component=component, depth=depth, x0=x0, x1=x1, y0=y0, y1=y1)
            name = f"{component}_{depth}_Q0.5_{key}.tif"
            try:
                out[(component, depth)] = fetch_file(
                    url, cache / name, urlopen=urlopen)
            except SourceAbsent as error:
                raise CoverageError(
                    f"SoilGrids WCS refused {component} {depth} for window "
                    f"{key}: {error}") from error
    return out


# ---------------------------------------------------------------------------
# Derived per-footprint windows (local derivations of fetched payloads)
# ---------------------------------------------------------------------------

def _densified_bounds(bbox: FootprintBBox, dst_crs, margin_m: float
                      ) -> tuple[float, float, float, float]:
    """Footprint bounds in ``dst_crs``, sampled along the perimeter."""
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "high-resolution geography requires the optional 'geog' extra"
        ) from exc
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    lons = np.linspace(bbox.lon_min, bbox.lon_max, 41)
    lats = np.linspace(bbox.lat_min, bbox.lat_max, 41)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    x, y = transformer.transform(grid_lon, grid_lat)
    return (float(np.min(x)) - margin_m, float(np.min(y)) - margin_m,
            float(np.max(x)) + margin_m, float(np.max(y)) + margin_m)


def derive_terrain_window(tiles, bbox: FootprintBBox,
                          cache_root: Path) -> FetchedFile:
    """Mosaic the fetched whole tiles and clip to the footprint.

    The derivation is cached by the SHA-256 of its inputs (tile digests +
    footprint), so re-preparing a domain reuses it byte-identically.
    """
    try:
        import rasterio
        from rasterio.merge import merge as rasterio_merge
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "high-resolution geography requires the optional 'geog' extra"
        ) from exc
    tiles = list(tiles)
    if not tiles:
        raise ValueError("terrain window derivation requires >= 1 tile")
    identity = hashlib.sha256(json.dumps(
        {"tiles": sorted(item.sha256 for item in tiles),
         "bbox": bbox.as_dict(), "kind": "terrain-window-v1"},
        sort_keys=True).encode("utf-8")).hexdigest()[:20]
    out_dir = Path(cache_root) / "derived"
    out_path = out_dir / f"terrain_{identity}.tif"
    cached = _read_sidecar(out_path)
    derivation_url = ("derived:mosaic+clip of "
                      + ",".join(sorted(item.path.name for item in tiles)))
    if cached is not None:
        return FetchedFile(
            path=out_path, url=derivation_url, sha256=cached["sha256"],
            bytes=int(cached["bytes"]),
            fetched_utc=str(cached.get("fetched_utc", "")), cache_hit=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [rasterio.open(item.path) for item in tiles]
    try:
        crs = datasets[0].crs
        margin_deg = 0.01
        mosaic, transform = rasterio_merge(
            datasets, bounds=(bbox.lon_min - margin_deg,
                              bbox.lat_min - margin_deg,
                              bbox.lon_max + margin_deg,
                              bbox.lat_max + margin_deg))
        nodata = datasets[0].nodata
    finally:
        for dataset in datasets:
            dataset.close()
    partial = out_path.with_name(out_path.name + ".partial")
    with rasterio.open(
            partial, "w", driver="GTiff", height=mosaic.shape[1],
            width=mosaic.shape[2], count=1, dtype=mosaic.dtype, crs=crs,
            transform=transform, nodata=nodata, compress="deflate",
            predictor=2 if np.issubdtype(mosaic.dtype, np.integer) else 3,
            tiled=True) as target:
        target.write(mosaic[0], 1)
    os.replace(partial, out_path)
    return record_local_artifact(out_path, url=derivation_url)


def derive_landcover_window(raster: FetchedFile, bbox: FootprintBBox,
                            cache_root: Path) -> FetchedFile:
    """Clip the whole NLCD year raster to the footprint (cached)."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "high-resolution geography requires the optional 'geog' extra"
        ) from exc
    identity = hashlib.sha256(json.dumps(
        {"source": raster.sha256, "bbox": bbox.as_dict(),
         "kind": "landcover-window-v1"},
        sort_keys=True).encode("utf-8")).hexdigest()[:20]
    out_dir = Path(cache_root) / "derived"
    out_path = out_dir / f"landcover_{identity}.tif"
    derivation_url = f"derived:clip of {raster.path.name}"
    cached = _read_sidecar(out_path)
    if cached is not None:
        return FetchedFile(
            path=out_path, url=derivation_url, sha256=cached["sha256"],
            bytes=int(cached["bytes"]),
            fetched_utc=str(cached.get("fetched_utc", "")), cache_hit=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raster.path) as source:
        left, bottom, right, top = _densified_bounds(
            bbox, source.crs, margin_m=2000.0)
        window = from_bounds(left, bottom, right, top,
                             transform=source.transform)
        # Explicit outward rounding (rasterio's round_offsets/round_lengths
        # signatures drifted across 1.x releases; the arithmetic is fixed).
        col_off = math.floor(window.col_off)
        row_off = math.floor(window.row_off)
        window = rasterio.windows.Window(
            col_off, row_off,
            math.ceil(window.width + (window.col_off - col_off)),
            math.ceil(window.height + (window.row_off - row_off)))
        full = rasterio.windows.Window(0, 0, source.width, source.height)
        clipped = window.intersection(full)
        if clipped.width <= 0 or clipped.height <= 0:
            raise CoverageError(
                f"footprint {bbox.as_dict()} lies outside the Annual NLCD "
                f"raster extent of {raster.path.name}")
        if (clipped.width != window.width
                or clipped.height != window.height):
            raise CoverageError(
                f"footprint {bbox.as_dict()} is only partially covered by "
                f"the Annual NLCD raster {raster.path.name}; the source "
                "window would be truncated from "
                f"{int(window.width)}x{int(window.height)} to "
                f"{int(clipped.width)}x{int(clipped.height)} pixels")
        values = source.read(1, window=clipped)
        transform = source.window_transform(clipped)
        partial = out_path.with_name(out_path.name + ".partial")
        with rasterio.open(
                partial, "w", driver="GTiff", height=values.shape[0],
                width=values.shape[1], count=1, dtype=values.dtype,
                crs=source.crs, transform=transform, nodata=source.nodata,
                compress="deflate", predictor=2, tiled=True) as target:
            target.write(values, 1)
    os.replace(partial, out_path)
    return record_local_artifact(out_path, url=derivation_url)


__all__ = [
    "ANNUAL_NLCD_FIRST_YEAR", "ANNUAL_NLCD_LAST_YEAR", "ANNUAL_NLCD_URL",
    "CoverageError", "FetchedFile", "FootprintBBox", "SOILGRIDS_COMPONENTS",
    "SOILGRIDS_CRS", "SOILGRIDS_DEPTHS", "SOILGRIDS_NODATA",
    "SOILGRIDS_SCALE", "SourceAbsent", "THREE_DEP_TILE_URL",
    "derive_landcover_window", "derive_terrain_window", "domain_footprint",
    "fetch_annual_nlcd", "fetch_file", "fetch_soilgrids",
    "fetch_three_dep_tiles", "nlcd_year_for", "record_local_artifact",
    "three_dep_tile_ids",
]
