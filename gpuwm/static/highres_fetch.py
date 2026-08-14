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

# One remedy string for the whole geography stack; see geog_stack.
from .geog_stack import geog_unavailable_detail

import hashlib
import json
import math
import os
import re
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


#: Copernicus DEM GLO-30 (COG GeoTIFF, ~30 m, near-global), AWS Open Data.
#: Anonymous HTTP GET; no account, no token, no signing.  Tile ids name the
#: SOUTH-WEST corner: ``N39_00_W105_00`` spans 39..40 N, 105..104 W.
COPERNICUS_DEM_TILE_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{tile}_DEM/Copernicus_DSM_COG_10_{tile}_DEM.tif")
COPERNICUS_DEM_SOURCE_URL = (
    "https://registry.opendata.aws/copernicus-dem/")
COPERNICUS_DEM_LICENSE = (
    "Copernicus-DEM-EULA-free-open",
    "https://spacedata.copernicus.eu/documents/20123/121286/"
    "CSCDA_ESA_Mission-specific+Annex.pdf")
COPERNICUS_DEM_ATTRIBUTION = (
    "produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) "
    "Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by "
    "the European Union and ESA; all rights reserved")
#: Published tile labels run S90..N83 (south-west corners), so the product
#: reaches 84 N and 90 S -- not the whole globe.  Verified against the
#: bucket's own tileList.txt, not assumed.
COPERNICUS_DEM_LAT_MAX = 84.0
COPERNICUS_DEM_LAT_MIN = -90.0
#: Latitude spacing is 1/3600 degree everywhere.  LONGITUDE spacing is
#: latitude-banded (verified against the published tiles): 3600 columns
#: below 50 deg, 2400 in 50-60, 1800 in 60-70, and coarser toward the
#: poles.  A mosaic therefore has to declare its own output resolution
#: instead of inheriting the first tile's.
COPERNICUS_DEM_LAT_STEP_DEG = 1.0 / 3600.0
#: Elevations are metres above the EGM2008 geoid -- the same vertical sense
#: as WPS terrain, so no datum shift is applied or needed.
COPERNICUS_DEM_VERTICAL_DATUM = "EGM2008 geoid (orthometric metres)"

#: SRTM 1 arc-second (SRTMGL1 v3).  Wired as a named alternative because
#: users ask for it by name.  NASA's own LP DAAC distribution needs an
#: Earthdata login, which this program will not require of anyone; the
#: OpenTopography S3-compatible mirror serves the same raw v3 tiles
#: anonymously and is what is fetched here.  Tile ids name the south-west
#: corner, 3601x3601 int16, nodata -32768, 1/3600 degree in BOTH axes at
#: every latitude (unlike Copernicus).
SRTM_GL1_TILE_URL = ("https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/"
                     "SRTM_GL1_srtm/{tile}.tif")
SRTM_GL1_SOURCE_URL = "https://portal.opentopography.org/raster?opentopoID=OTSRTM.082015.4326.1"
SRTM_GL1_LICENSE = ("US-PD", "https://lpdaac.usgs.gov/data/data-citation-"
                             "and-policies/")
SRTM_GL1_ATTRIBUTION = (
    "NASA Shuttle Radar Topography Mission Global 1 arc second "
    "(SRTMGL1 v3), NASA JPL 2013, doi:10.5067/MEaSUREs/SRTM/SRTMGL1.003; "
    "distributed by OpenTopography, doi:10.5069/G9445JDF")
SRTM_GL1_STEP_DEG = 1.0 / 3600.0
SRTM_GL1_NODATA = -32768.0
#: SRTM heights are metres above the EGM96 geoid; Copernicus uses EGM2008.
#: The two differ by up to a few metres regionally, which is why a run
#: names its terrain source in the receipt instead of calling them
#: interchangeable.
SRTM_GL1_VERTICAL_DATUM = "EGM96 geoid (orthometric metres)"


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


@dataclass(frozen=True)
class SourceCoverage:
    """Where one named source is published, and under what terms.

    Coverage is a property of the source, not of the program: each source
    declares its own envelope and a footprint is checked against the source
    actually selected, so a refusal can say *which* dataset does not reach
    *where*.  ``global_lon`` marks sources published for every longitude,
    which is the normal case outside the US collections.
    """

    source_id: str
    role: str
    envelope: FootprintBBox
    nominal_resolution: str
    source_url: str
    license_id: str
    license_url: str
    attribution: str = ""
    note: str = ""

    def outside(self, bbox: FootprintBBox) -> dict[str, float]:
        """Per-edge overshoot in degrees; empty when fully covered."""
        env, out = self.envelope, {}
        if bbox.lat_min < env.lat_min:
            out["south_by_deg"] = round(env.lat_min - bbox.lat_min, 6)
        if bbox.lat_max > env.lat_max:
            out["north_by_deg"] = round(bbox.lat_max - env.lat_max, 6)
        if bbox.lon_min < env.lon_min:
            out["west_by_deg"] = round(env.lon_min - bbox.lon_min, 6)
        if bbox.lon_max > env.lon_max:
            out["east_by_deg"] = round(bbox.lon_max - env.lon_max, 6)
        return out

    def check(self, bbox: FootprintBBox) -> None:
        """Raise :class:`CoverageError` naming source, footprint, overshoot."""
        overshoot = self.outside(bbox)
        if not overshoot:
            return
        raise CoverageError(
            f"source {self.source_id!r} ({self.role}, "
            f"{self.nominal_resolution}) is published over "
            f"{self.envelope.as_dict()}; the domain+halo footprint "
            f"{bbox.as_dict()} leaves it by {overshoot}. "
            + (f"{self.note} " if self.note else "")
            + f"Publication terms: {self.source_url}")

    def echo(self) -> dict[str, object]:
        return {"source_id": self.source_id, "role": self.role,
                "envelope": self.envelope.as_dict(),
                "nominal_resolution": self.nominal_resolution,
                "source_url": self.source_url,
                "license_id": self.license_id,
                "license_url": self.license_url,
                "attribution": self.attribution, "note": self.note}


#: Envelope where the two US collections are JOINTLY published: 3DEP staged
#: 1/3 arc-second tiles and the Annual NLCD conterminous-US collection.
_US_ENVELOPE = FootprintBBox(lat_min=24.0, lat_max=49.5,
                             lon_min=-125.0, lon_max=-66.5)
#: Copernicus GLO-30's published extent.  Individual tiles can still be
#: absent (the product does not publish all-water tiles); that is a
#: tile-level fact checked at fetch time, not an envelope fact.
_COPERNICUS_ENVELOPE = FootprintBBox(
    lat_min=COPERNICUS_DEM_LAT_MIN, lat_max=COPERNICUS_DEM_LAT_MAX,
    lon_min=-180.0, lon_max=180.0)

#: Terrain sources, keyed by the id users write in ``terrain_source``.
TERRAIN_SOURCES: dict[str, SourceCoverage] = {
    "usgs-3dep-13as": SourceCoverage(
        source_id="usgs-3dep-13as", role="terrain",
        envelope=_US_ENVELOPE,
        nominal_resolution="1/3 arc-second (~10 m)",
        source_url=THREE_DEP_SOURCE_URL,
        license_id=THREE_DEP_LICENSE[0], license_url=THREE_DEP_LICENSE[1],
        attribution="USGS 3D Elevation Program (public domain).",
        note="3DEP staged 1/3 arc-second tiles are a United States "
             "collection; outside the US use terrain_source = "
             "\"copernicus-dem-glo30\"."),
    "copernicus-dem-glo30": SourceCoverage(
        source_id="copernicus-dem-glo30", role="terrain",
        envelope=_COPERNICUS_ENVELOPE,
        nominal_resolution="1 arc-second latitude (~30 m)",
        source_url=COPERNICUS_DEM_SOURCE_URL,
        license_id=COPERNICUS_DEM_LICENSE[0],
        license_url=COPERNICUS_DEM_LICENSE[1],
        attribution=COPERNICUS_DEM_ATTRIBUTION,
        note="Published south-west tile labels run S90..N83, so the "
             "product reaches 84 N and 90 S.  All-water tiles are not "
             "published, which this path treats as sea level only where "
             "the domain's own baseline land mask already says water."),
    "srtm-gl1": SourceCoverage(
        source_id="srtm-gl1", role="terrain",
        envelope=FootprintBBox(lat_min=-56.0, lat_max=60.0,
                               lon_min=-180.0, lon_max=180.0),
        nominal_resolution="1 arc-second (~30 m)",
        source_url=SRTM_GL1_SOURCE_URL,
        license_id=SRTM_GL1_LICENSE[0], license_url=SRTM_GL1_LICENSE[1],
        attribution=SRTM_GL1_ATTRIBUTION,
        note="SRTM stops at 60 N / 56 S: it does not reach Canada, "
             "Scandinavia, Alaska or most of Russia.  Fetched from the "
             "anonymous OpenTopography mirror because NASA's own "
             "distribution requires an Earthdata login."),
}

#: Land-cover sources.  There is exactly one wired, and it is US-only.
LANDCOVER_SOURCES: dict[str, SourceCoverage] = {
    "annual-nlcd": SourceCoverage(
        source_id="annual-nlcd", role="landcover",
        envelope=_US_ENVELOPE, nominal_resolution="30 m",
        source_url=ANNUAL_NLCD_SOURCE_URL,
        license_id=ANNUAL_NLCD_LICENSE[0],
        license_url=ANNUAL_NLCD_LICENSE[1],
        attribution="Annual NLCD Collection 1, MRLC (public domain).",
        note="Annual NLCD is a conterminous-United-States collection; no "
             "global land cover is wired, so outside the US the land-use "
             "fields stay on the 30-arc-second baseline."),
}


def terrain_source_coverage(source_id: str) -> SourceCoverage:
    """Look up one terrain source, refusing an unknown id by name."""
    try:
        return TERRAIN_SOURCES[source_id]
    except KeyError:
        raise CoverageError(
            f"unknown terrain source {source_id!r}; known sources are "
            f"{sorted(TERRAIN_SOURCES)}") from None


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
# Copernicus DEM GLO-30 terrain tiles (near-global)
# ---------------------------------------------------------------------------

def copernicus_dem_tile_ids(bbox: FootprintBBox) -> tuple[str, ...]:
    """1x1-degree GLO-30 tile ids covering ``bbox`` (south-west corner names).

    ``N39_00_W105_00`` spans 39..40 N and 105..104 W; ``S34_00_E018_00``
    spans 34..33 S and 18..19 E.  Unlike the US enumerator this one is
    valid in all four quadrants, which is the whole point of the source.
    """
    if bbox.lon_max - bbox.lon_min > 180.0:
        raise CoverageError(
            f"footprint {bbox.as_dict()} spans "
            f"{bbox.lon_max - bbox.lon_min:.1f} degrees of longitude; a "
            "footprint that wide is an antimeridian wrap, not a domain, "
            "and 1x1-degree tile enumeration cannot express it")
    lat_lo = max(-90, int(math.floor(bbox.lat_min)))
    lat_hi = min(90, int(math.ceil(bbox.lat_max)))
    lon_lo = int(math.floor(bbox.lon_min))
    lon_hi = int(math.ceil(bbox.lon_max))
    tiles: list[str] = []
    for lat_sw in range(lat_lo, lat_hi):
        ns = "N" if lat_sw >= 0 else "S"
        for lon_sw in range(lon_lo, lon_hi):
            wrapped = ((lon_sw + 180) % 360) - 180
            ew = "E" if wrapped >= 0 else "W"
            tiles.append(f"{ns}{abs(lat_sw):02d}_00_"
                         f"{ew}{abs(wrapped):03d}_00")
    if not tiles:
        raise CoverageError(
            f"footprint {bbox.as_dict()} enumerates no Copernicus DEM tile")
    return tuple(tiles)


def srtm_tile_ids(bbox: FootprintBBox) -> tuple[str, ...]:
    """SRTMGL1 tile ids covering ``bbox`` (``N39W105`` style, SW corner)."""
    return tuple(tile.replace("_00_", "").removesuffix("_00")
                 for tile in copernicus_dem_tile_ids(bbox))


def one_degree_tile_bbox(tile: str) -> FootprintBBox:
    """Geographic box of one 1x1-degree tile id, either naming style.

    Accepts ``N39_00_W105_00`` (Copernicus) and ``N39W105`` (SRTM); both
    name the south-west corner.
    """
    match = re.fullmatch(r"([NS])(\d{2})(?:_00)?_?([EW])(\d{3})(?:_00)?",
                         tile)
    if match is None:
        raise ValueError(
            f"tile id {tile!r} is not a 1x1-degree south-west-corner id "
            "(expected N39_00_W105_00 or N39W105)")
    lat = int(match.group(2)) * (1 if match.group(1) == "N" else -1)
    lon = int(match.group(4)) * (1 if match.group(3) == "E" else -1)
    return FootprintBBox(lat_min=float(lat), lat_max=float(lat + 1),
                         lon_min=float(lon), lon_max=float(lon + 1))


#: Backward-compatible alias.
copernicus_tile_bbox = one_degree_tile_bbox


def fetch_srtm_gl1_tiles(bbox: FootprintBBox, cache_root: Path, *,
                         urlopen=None
                         ) -> tuple[tuple[FetchedFile, ...], tuple[str, ...]]:
    """Fetch every published SRTMGL1 tile covering ``bbox``.

    Same ``(fetched, absent)`` contract as
    :func:`fetch_copernicus_dem_tiles`: SRTM publishes no all-water tiles
    either, so absence is handed back for the caller to cross-check rather
    than being read as terrain.
    """
    cache = Path(cache_root) / "srtm_gl1"
    fetched: list[FetchedFile] = []
    absent: list[str] = []
    tiles = srtm_tile_ids(bbox)
    for tile in tiles:
        url = SRTM_GL1_TILE_URL.format(tile=tile)
        try:
            fetched.append(fetch_file(url, cache / f"{tile}.tif",
                                      urlopen=urlopen))
        except SourceAbsent:
            absent.append(tile)
    if not fetched:
        raise CoverageError(
            "SRTM 1 arc-second publishes none of the "
            f"{len(tiles)} tile(s) covering footprint {bbox.as_dict()} "
            f"({list(tiles)}); SRTM omits all-water tiles and stops at "
            "60 N / 56 S, so this footprint is either open water or beyond "
            f"the mission's reach (checked "
            f"{SRTM_GL1_TILE_URL.format(tile='<tile>')})")
    return tuple(fetched), tuple(absent)


def fetch_copernicus_dem_tiles(bbox: FootprintBBox, cache_root: Path, *,
                               urlopen=None
                               ) -> tuple[tuple[FetchedFile, ...],
                                          tuple[str, ...]]:
    """Fetch every published GLO-30 tile covering ``bbox``.

    Returns ``(fetched, absent)``.  Absence is *not* silently equivalent to
    a coverage gap here, because the product deliberately does not publish
    all-water tiles: the absent ids are handed back so the caller can
    cross-check them against the domain's own baseline land mask before
    deciding they mean ocean.  A footprint where *every* tile is absent has
    no terrain to improve and refuses outright.
    """
    cache = Path(cache_root) / "copernicus_dem_glo30"
    fetched: list[FetchedFile] = []
    absent: list[str] = []
    tiles = copernicus_dem_tile_ids(bbox)
    for tile in tiles:
        url = COPERNICUS_DEM_TILE_URL.format(tile=tile)
        try:
            fetched.append(fetch_file(
                url, cache / f"Copernicus_DSM_COG_10_{tile}_DEM.tif",
                urlopen=urlopen))
        except SourceAbsent:
            absent.append(tile)
    if not fetched:
        raise CoverageError(
            "Copernicus DEM GLO-30 publishes none of the "
            f"{len(tiles)} tile(s) covering footprint {bbox.as_dict()} "
            f"({list(tiles)}); the product omits all-water tiles, so this "
            "footprint is open water and has no high-resolution terrain to "
            f"apply (checked "
            f"{COPERNICUS_DEM_TILE_URL.format(tile='<tile>')})")
    return tuple(fetched), tuple(absent)


def derive_global_terrain_window(tiles, bbox: FootprintBBox,
                                 cache_root: Path, *,
                                 sea_level_fill: float = 0.0,
                                 source_nodata: float | None = None,
                                 resolution_deg: float
                                 = COPERNICUS_DEM_LAT_STEP_DEG
                                 ) -> tuple[FetchedFile, dict[str, object]]:
    """Mosaic GLO-30 tiles onto one uniform grid over the whole footprint.

    Two things differ from :func:`derive_terrain_window` and both are
    forced by the source:

    - GLO-30's longitude sampling is latitude-banded (3600 columns per
      degree below 50, 2400 in 50-60, 1800 in 60-70), so tiles from
      different bands do not share a resolution.  The output resolution is
      declared, not inherited, and the resampling is nearest so no
      elevation value is invented -- coarser bands are replicated, and the
      subsequent area-average to the model grid is what actually reduces
      them.
    - Unpublished all-water tiles leave holes (and SRTM additionally
      carries an in-band void sentinel).  They are filled with
      ``sea_level_fill`` (0 m on the EGM2008 geoid, the source's own
      vertical datum) and the filled pixel count is returned so the caller
      can refuse if any of it lands on baseline land.
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.merge import merge as rasterio_merge
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            geog_unavailable_detail()
        ) from exc
    tiles = list(tiles)
    if not tiles:
        raise ValueError("terrain window derivation requires >= 1 tile")
    identity = hashlib.sha256(json.dumps(
        {"tiles": sorted(item.sha256 for item in tiles),
         "bbox": bbox.as_dict(), "res": resolution_deg,
         "fill": sea_level_fill, "src_nodata": source_nodata,
         "kind": "global-terrain-window-v1"},
        sort_keys=True).encode("utf-8")).hexdigest()[:20]
    out_dir = Path(cache_root) / "derived"
    out_path = out_dir / f"terrain_global_{identity}.tif"
    sidecar_path = out_dir / f"terrain_global_{identity}.audit.json"
    derivation_url = ("derived:mosaic+fill+clip of "
                      + ",".join(sorted(item.path.name for item in tiles)))
    cached = _read_sidecar(out_path)
    if cached is not None and sidecar_path.is_file():
        return (FetchedFile(
            path=out_path, url=derivation_url, sha256=cached["sha256"],
            bytes=int(cached["bytes"]),
            fetched_utc=str(cached.get("fetched_utc", "")), cache_hit=True),
            json.loads(sidecar_path.read_text(encoding="utf-8")))

    out_dir.mkdir(parents=True, exist_ok=True)
    margin_deg = 0.01
    bounds = (bbox.lon_min - margin_deg, bbox.lat_min - margin_deg,
              bbox.lon_max + margin_deg, bbox.lat_max + margin_deg)
    datasets = [rasterio.open(item.path) for item in tiles]
    try:
        crs = datasets[0].crs
        mosaic, transform = rasterio_merge(
            datasets, bounds=bounds,
            res=(resolution_deg, resolution_deg),
            resampling=Resampling.nearest,
            nodata=np.nan, dtype="float32")
    finally:
        for dataset in datasets:
            dataset.close()
    values = np.asarray(mosaic[0], dtype=np.float32)
    holes = ~np.isfinite(values)
    if source_nodata is not None:
        # SRTM carries an in-band void sentinel; Copernicus carries none.
        holes |= values == np.float32(source_nodata)
    filled = int(np.count_nonzero(holes))
    values[holes] = np.float32(sea_level_fill)
    partial = out_path.with_name(out_path.name + ".partial")
    with rasterio.open(
            partial, "w", driver="GTiff", height=values.shape[0],
            width=values.shape[1], count=1, dtype="float32", crs=crs,
            transform=transform, nodata=None, compress="deflate",
            predictor=3, tiled=True) as target:
        target.write(values, 1)
    os.replace(partial, out_path)
    audit = {
        "output_resolution_deg": float(resolution_deg),
        "output_shape": [int(values.shape[0]), int(values.shape[1])],
        "sea_level_filled_pixels": filled,
        "total_pixels": int(values.size),
        "sea_level_fill_m": float(sea_level_fill),
        "source_nodata": (None if source_nodata is None
                          else float(source_nodata)),
        "resampling": "nearest (latitude-banded source resolutions)",
        "vertical_datum": COPERNICUS_DEM_VERTICAL_DATUM,
    }
    sidecar_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record_local_artifact(out_path, url=derivation_url), audit


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
            geog_unavailable_detail()
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
            geog_unavailable_detail()
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
            geog_unavailable_detail()
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
            geog_unavailable_detail()
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
    "COPERNICUS_DEM_ATTRIBUTION", "COPERNICUS_DEM_LAT_STEP_DEG",
    "COPERNICUS_DEM_LICENSE", "COPERNICUS_DEM_SOURCE_URL",
    "COPERNICUS_DEM_TILE_URL", "COPERNICUS_DEM_VERTICAL_DATUM",
    "CoverageError", "FetchedFile", "FootprintBBox", "LANDCOVER_SOURCES",
    "SOILGRIDS_COMPONENTS", "SOILGRIDS_CRS", "SOILGRIDS_DEPTHS",
    "SOILGRIDS_NODATA", "SOILGRIDS_SCALE", "SRTM_GL1_LICENSE",
    "SRTM_GL1_ATTRIBUTION", "SRTM_GL1_NODATA", "SRTM_GL1_SOURCE_URL",
    "SRTM_GL1_STEP_DEG", "SRTM_GL1_TILE_URL", "SRTM_GL1_VERTICAL_DATUM",
    "SourceAbsent", "SourceCoverage",
    "TERRAIN_SOURCES", "THREE_DEP_TILE_URL", "copernicus_dem_tile_ids",
    "copernicus_tile_bbox", "derive_global_terrain_window",
    "fetch_srtm_gl1_tiles", "one_degree_tile_bbox", "srtm_tile_ids",
    "derive_landcover_window", "derive_terrain_window", "domain_footprint",
    "fetch_annual_nlcd", "fetch_copernicus_dem_tiles", "fetch_file",
    "fetch_soilgrids", "fetch_three_dep_tiles", "nlcd_year_for",
    "record_local_artifact", "terrain_source_coverage", "three_dep_tile_ids",
]
