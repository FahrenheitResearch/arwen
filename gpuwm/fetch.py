"""Acquire initialization/boundary data: the ``gpuwm fetch`` front door.

This module is transport only.  It downloads (or, for ERA5, templates and
validates) the exact source inventory the fail-closed Rust GRIB bridges
consume downstream; it never decodes scientific payloads itself.  Every
published file is envelope-verified, sha256-summed, and recorded in a
``fetch-manifest.json`` the preparation step can consume.

Per-source transport:

* ``gfs`` -- the NOMADS ``filter_gfs_0p25.pl`` subsetter (spatial subregion
  + the exact 124-record variable/level selection).  Raw AWS Open Data S3
  ``noaa-gfs-bdp-pds`` objects are whole-globe north-to-south grids
  (scan mode 0x00); the certified ``gfs_grib2_bridge`` admits only the
  west-east/south-north (scan 0x40) regular grids the NOMADS subsetter
  emits, so neither whole-file S3 downloads nor ``.idx`` byte-range
  subsets of them can feed the certified pipeline.  S3 is still used for
  anonymous ``--cycle latest`` resolution (HEAD probes, no HTML scraping).
* ``hrrr`` -- NOAA ``.idx`` byte-range subsetting, reusing the proven
  record inventory and range transport in
  :mod:`tools.download_hrrr_native_subset` (native hybrid ``wrfnat``
  atmosphere plus the soil records of ``wrfprs``, the inputs
  ``hrrr_grib2_bridge`` requires), over either of two hosts serving the
  identical production files and indexes: the NOMADS mirror
  (``nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod``, roughly the
  newest 48 h, where each hour publishes first) and the AWS Open Data
  S3 archive ``noaa-hrrr-bdp-pds``.  ``--transport auto`` (default)
  probes NOMADS for the requested window and falls back to S3.  NOMADS
  has no grib-filter route for these products -- its HRRR filter
  scripts cover the 2-D ``wrfsfc`` file only -- so subsetting stays
  ``.idx`` byte ranges on both hosts and the exact 561/18 record
  contracts are unchanged.  ``--wait-for`` polls (at most every 30 s)
  and downloads each forecast hour as it publishes, so preparation can
  start before a live cycle finishes publishing.  HRRR objects are
  CONUS-wide; ``.idx`` subsetting selects records, not areas, so
  ``--area`` is validated against CONUS coverage rather than used to
  crop.
* ``era5`` -- no CDS download is implemented (the CDS API requires a user
  account and key).  ``fetch`` emits the precise ``cdsapi`` request
  template for the variables/levels/times/area gpuwm ingest expects and,
  with ``--validate``, checks a user-supplied GRIB1 file set against that
  expectation (transport envelopes via
  :func:`gpuwm.ingest.grib.inspect_grib1_envelopes` plus a
  parameter/level/valid-time census).  The Rust bridge remains the decode
  authority at ingest time.

No case is named here; sources (GFS/HRRR/ERA5) are public data products,
not cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


FETCH_MANIFEST_SCHEMA = "gpuwm-fetch-manifest-v1"
FETCH_MANIFEST_NAME = "fetch-manifest.json"

#: The GFS front door (``rw-wps --source gfs``) verifies its inputs
#: against this manifest schema (gpuwm/gfs_direct.py
#: ``_verify_input_manifest``).  ``gpuwm fetch --source gfs
#: --author-front-door-manifest`` writes it, so the value is mirrored
#: here to keep fetch importable on a base install (no ingest imports);
#: a test binds the two constants together.
GFS_FRONT_DOOR_MANIFEST_SCHEMA = "gpuwm-gfs-direct-input-manifest-v1"
GFS_INPUT_MANIFEST_NAME = "gfs-input-manifest.json"

#: Margin (degrees) the suggested GFS fetch crop adds beyond the outer
#: domain.  Two terms, both anchored in the front door's donor-coverage
#: proof (gpuwm/gfs_direct.py ``_source_coverage_receipt``):
#:
#: * the deterministic parabolic/masked interpolation stencil reaches
#:   floor-based [-1, +2] source cells -- 2 cells = 0.5 deg at the
#:   0.25-deg GFS resolution -- so the crop needs at least that halo;
#: * lake initialization must *prove* the nearest source-water donor to
#:   every model lake lies inside the crop, i.e. the crop edge must be
#:   farther from each lake than its nearest GFS water cell.  Interior
#:   North-American lakes can sit many degrees from the nearest
#:   GFS-resolved water, so the suggested crop allows
#:   :data:`GFS_LAKE_DONOR_MARGIN_DEG` for that search.
GFS_SOURCE_RESOLUTION_DEG = 0.25
GFS_DONOR_HALO_CELLS = 2
GFS_LAKE_DONOR_MARGIN_DEG = 15.0


def gfs_suggested_fetch_margin_deg() -> float:
    """Fetch-crop margin (deg) sized for the GFS donor-coverage proof.

    Used by ``gpuwm domain`` to compute its suggested ``--area`` so the
    wizard's own hint passes the front door's coverage check instead of
    being rejected downstream.
    """
    return max(GFS_DONOR_HALO_CELLS * GFS_SOURCE_RESOLUTION_DEG,
               GFS_LAKE_DONOR_MARGIN_DEG)

GFS_S3_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
HRRR_S3_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
#: The NCEP production mirror.  It serves byte-identical HRRR files and
#: ``.idx`` indexes (HEAD Content-Length, ``Accept-Ranges: bytes``, and
#: HTTP 206 range responses verified 2026-07-29), publishes each hour
#: before the cloud mirrors, and keeps only about the newest two days.
HRRR_NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod"
#: Approximate NOMADS retention (hours).  Older cycles live on S3 only.
HRRR_NOMADS_RETENTION_HOURS = 48
HRRR_TRANSPORTS = ("auto", "nomads", "s3")
#: ``--wait-for`` polling cadence ceiling (seconds between probe rounds).
HRRR_WAIT_POLL_SECONDS = 30
#: ``--wait-for`` default patience: 90 min covers a live HRRR cycle's
#: full f00..f18 publication spread with margin.
HRRR_WAIT_TIMEOUT_DEFAULT_MINUTES = 90.0

GFS_CYCLE_HOURS = (0, 6, 12, 18)
GFS_MAX_FORECAST_HOUR = 384
#: One NOMADS-subset GFS pgrb2.0p25 file carries exactly the 124 records
#: the fail-closed ``gfs_grib2_bridge`` selects (21 pressure levels x
#: {GHT, T, RH, U, V} + 11 surface/near-surface + 8 soil-layer records).
GFS_SUBSET_RECORD_COUNT = 124

#: Approximate HRRR CONUS domain coverage (Lambert grid corners, outward
#: bound).  Requests outside this box cannot be served by HRRR at all, so
#: they fail closed here instead of at ingest.
HRRR_CONUS_LAT = (21.1, 52.7)
HRRR_CONUS_LON = (-134.2, -60.8)

#: The full ERA5 pressure-level ladder (hPa).  Requesting all 37 levels
#: keeps the template independent of any one experiment's p_top.
ERA5_PRESSURE_LEVELS_HPA = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850,
    875, 900, 925, 950, 975, 1000,
)

ERA5_REQUEST_NAME = "era5-cds-request.json"

# ERA5 GRIB1 parameter expectations, grounded in what ingest consumes
# (gpuwm/ingest/grib.py _CANONICAL_SPECS; gpuwm/ingest/real.py requires
# TT/RH/GHT/UU/VV/PSFC/T2/D2-or-RH2/U10/V10; gpuwm/ingest/soil.py requires
# LANDSEA/SKINTEMP plus the four ST/SM layers and reads SNOW_EC/SST/
# SEAICE; gpuwm/ingest/horiz.py converts invariant geopotential to source
# orography).  Keys are (grib1_parameter, cds short name).
ERA5_REQUIRED_PRESSURE = {
    129: "z", 130: "t", 131: "u", 132: "v", 157: "r",
}
ERA5_REQUIRED_SURFACE = {
    134: "sp", 165: "10u", 166: "10v", 167: "2t", 168: "2d", 172: "lsm",
    235: "skt", 141: "sd",
    139: "stl1", 170: "stl2", 183: "stl3", 236: "stl4",
    39: "swvl1", 40: "swvl2", 41: "swvl3", 42: "swvl4",
}
#: Invariant geopotential doubles as the source orography.  gpuwm's
#: ingest can substitute a per-domain source-orography supplement, so its
#: absence is reported as a failure with that escape hatch named.
ERA5_OROGRAPHY_PARAMETER = 129
ERA5_OPTIONAL_SURFACE = {151: "msl", 31: "ci", 34: "sst"}
#: Native CDS GRIB1 encodes soil layers as level type 112; the
#: CDO-normalized form flattens them to level type 1.  Ingest accepts
#: both (gpuwm/ingest/grib.py _NATIVE_LEVEL_ALIASES), so the census does
#: too.
ERA5_SOIL_PARAMETERS = frozenset({139, 170, 183, 236, 39, 40, 41, 42})

_USER_AGENT = "gpuwm-fetch/1"


# ---------------------------------------------------------------------------
# Area / cycle / hours parsing
# ---------------------------------------------------------------------------

def _wrap_lon(value: float) -> float:
    """Wrap a longitude into [-180, 180) (west-edge convention)."""
    return (value + 180.0) % 360.0 - 180.0


def _wrap_lon_east(value: float) -> float:
    """Wrap a longitude into (-180, 180]: an east edge sitting exactly
    on the antimeridian reads as +180, not -180."""
    wrapped = _wrap_lon(value)
    return 180.0 if wrapped == -180.0 else wrapped


@dataclass(frozen=True)
class Area:
    """A geographic bounding box, south/west/north/east in degrees.

    ``lon_west > lon_east`` (after wrapping into [-180, 180)) denotes a
    box crossing the antimeridian -- the eastward walk from west to
    east passes 180E.
    """

    lat_south: float
    lon_west: float
    lat_north: float
    lon_east: float

    @property
    def crosses_antimeridian(self) -> bool:
        return _wrap_lon(self.lon_west) > _wrap_lon_east(self.lon_east)

    def as_manifest(self) -> dict[str, float]:
        return {
            "lat_south": self.lat_south, "lon_west": self.lon_west,
            "lat_north": self.lat_north, "lon_east": self.lon_east,
        }

    def as_cds(self) -> list[float]:
        """CDS ``area`` convention: [north, west, south, east].

        Longitudes are wrapped into the signed convention; the CDS API
        reads ``west > east`` as an antimeridian-crossing box.
        """
        return [self.lat_north, _wrap_lon(self.lon_west),
                self.lat_south, _wrap_lon_east(self.lon_east)]

    def as_nomads(self) -> dict[str, float]:
        """NOMADS subregion in [0, 360] longitudes with left < right.

        A box crossing the prime meridian cannot be expressed with
        ``0 <= left < right <= 360``; it widens to the full longitude
        band, which downloads more bytes but stays correct.
        """
        left = self.lon_west % 360.0
        right = self.lon_east % 360.0
        if right == 0.0 and self.lon_east != 0.0:
            right = 360.0
        if not left < right:
            left, right = 0.0, 360.0
        return {"left_lon": left, "right_lon": right,
                "bottom_lat": self.lat_south, "top_lat": self.lat_north}


def parse_area(raw: str) -> Area:
    """Parse ``lat0,lon0,lat1,lon1`` (corner order free) into an Area.

    A longitude pair spanning more than 180 degrees is read as the
    complementary box crossing the antimeridian: ``170,-170`` (or
    ``-170,170``) is the 20-degree Pacific box over 180E, never the
    340-degree box that excludes it.  Boxes genuinely wider than 180
    degrees must be requested as the full band (span 360) or split.
    """

    parts = raw.split(",")
    if len(parts) != 4:
        raise ValueError(
            "--area must be lat0,lon0,lat1,lon1 (two corners, degrees)")
    try:
        lat0, lon0, lat1, lon1 = (float(part) for part in parts)
    except ValueError as error:
        raise ValueError("--area corners must be decimal degrees") from error
    lat_south, lat_north = sorted((lat0, lat1))
    lon_west, lon_east = sorted((lon0, lon1))
    if not (-90.0 <= lat_south and lat_north <= 90.0):
        raise ValueError("--area latitudes must lie within [-90, 90]")
    if not (-180.0 <= lon_west and lon_east <= 360.0):
        raise ValueError("--area longitudes must lie within [-180, 360]")
    if lat_south == lat_north or lon_west == lon_east:
        raise ValueError("--area must span a nonzero box")
    span = lon_east - lon_west
    if 180.0 < span < 360.0:
        # The complement crossing the antimeridian is the intended box.
        lon_west, lon_east = lon_east, lon_west
    return Area(lat_south, lon_west, lat_north, lon_east)


def area_from_point(raw_point: str, radius_km: float) -> Area:
    """Convert ``--point lat,lon --radius-km N`` to a bounding Area."""

    parts = raw_point.split(",")
    if len(parts) != 2:
        raise ValueError("--point must be lat,lon in decimal degrees")
    try:
        lat, lon = (float(part) for part in parts)
    except ValueError as error:
        raise ValueError("--point must be lat,lon in decimal degrees") from error
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 360.0:
        raise ValueError("--point lies outside [-90,90] x [-180,360]")
    if not math.isfinite(radius_km) or radius_km <= 0.0:
        raise ValueError("--radius-km must be positive")
    km_per_degree = 111.195  # mean meridional degree
    dlat = radius_km / km_per_degree
    cos_lat = math.cos(math.radians(lat))
    if cos_lat * km_per_degree * 360.0 <= 2.0 * radius_km or cos_lat <= 0.0:
        raise ValueError(
            "--radius-km circles the pole at this latitude; "
            "pass an explicit --area instead")
    dlon = radius_km / (km_per_degree * cos_lat)
    lat_south = max(-90.0, lat - dlat)
    lat_north = min(90.0, lat + dlat)
    # Wrap the walked-out edges into the signed convention; near the
    # antimeridian this produces a lon_west > lon_east crossing box.
    return Area(lat_south, _wrap_lon(lon - dlon),
                lat_north, _wrap_lon_east(lon + dlon))


def parse_cycle(raw: str, source: str) -> datetime:
    """Parse ``YYYY-MM-DDTHH`` and enforce the source's cycle cadence."""

    try:
        cycle = datetime.strptime(raw, "%Y-%m-%dT%H")
    except ValueError as error:
        raise ValueError(
            f"--cycle {raw!r} must be YYYY-MM-DDTHH (UTC) or 'latest'"
        ) from error
    if source == "gfs" and cycle.hour not in GFS_CYCLE_HOURS:
        raise ValueError("GFS cycles run at 00/06/12/18 UTC only")
    return cycle


def gfs_forecast_hours(hours: int, cadence: int) -> tuple[int, ...]:
    """The f000..fNNN ladder the GFS series contract accepts."""

    if cadence not in (1, 3):
        raise ValueError("GFS cadence must be 1 or 3 hours")
    if hours < cadence or hours % cadence:
        raise ValueError(
            f"--hours must be a positive multiple of the {cadence} h cadence")
    if hours > GFS_MAX_FORECAST_HOUR:
        raise ValueError(
            f"--hours exceeds the GFS f{GFS_MAX_FORECAST_HOUR} horizon")
    return tuple(range(0, hours + 1, cadence))


def hrrr_forecast_hours(hours: int, cycle: datetime) -> tuple[int, ...]:
    """The contiguous f00..fNN window, checked against the cycle horizon."""

    from gpuwm.hrrr_forecast import validate_hrrr_source_forecast_hours

    if hours < 1:
        raise ValueError("--hours must be at least 1 (two hourly frames)")
    return validate_hrrr_source_forecast_hours(
        range(0, hours + 1), cycle=cycle)


# ---------------------------------------------------------------------------
# Latest-cycle resolution (anonymous S3 HEAD probes)
# ---------------------------------------------------------------------------

def _head_ok(url: str) -> bool:
    request = Request(url, method="HEAD",
                      headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=60) as response:
            return 200 <= response.status < 300
    except (HTTPError, URLError, OSError):
        return False


def gfs_object_url(cycle: datetime, hour: int) -> str:
    return (f"{GFS_S3_BASE}/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/"
            f"gfs.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}")


def _hrrr_transport_base(transport: str) -> str:
    if transport == "s3":
        return HRRR_S3_BASE
    if transport == "nomads":
        return HRRR_NOMADS_BASE
    raise ValueError(
        f"unknown HRRR transport {transport!r}; expected 'nomads' or 's3'")


def hrrr_object_url(cycle: datetime, hour: int, product: str,
                    transport: str = "s3") -> str:
    if product not in ("wrfnat", "wrfprs"):
        raise ValueError(f"unknown HRRR product {product!r}")
    return (f"{_hrrr_transport_base(transport)}/hrrr.{cycle:%Y%m%d}/conus/"
            f"hrrr.t{cycle:%H}z.{product}f{hour:02d}.grib2")


def resolve_latest_cycle(source: str, last_hour: int, *,
                         now: datetime | None = None,
                         probe=_head_ok) -> datetime:
    """Newest cycle whose final requested objects exist on AWS S3.

    A cycle qualifies only when every probed object for forecast hour
    ``last_hour`` is already published, so a partially uploaded cycle
    never wins and the fetched window is complete by construction.  For
    HRRR that means BOTH the final ``wrfnat`` (atmosphere) and the final
    ``wrfprs`` (soil-record source) objects: fetching needs both per
    hour, and during a live publication ``wrfnat`` can appear before its
    ``wrfprs`` sibling, which must not make the cycle win.
    """

    if now is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
    if source == "gfs":
        step = timedelta(hours=6)
        candidate = now.replace(
            minute=0, second=0, microsecond=0,
            hour=max(h for h in GFS_CYCLE_HOURS if h <= now.hour))
        candidates = tuple(candidate - i * step for i in range(9))
        urls = {cycle: (gfs_object_url(cycle, last_hour),)
                for cycle in candidates}
    elif source == "hrrr":
        candidate = now.replace(minute=0, second=0, microsecond=0)
        candidates = tuple(candidate - timedelta(hours=i) for i in range(13))
        urls = {cycle: (hrrr_object_url(cycle, last_hour, "wrfnat"),
                        hrrr_object_url(cycle, last_hour, "wrfprs"))
                for cycle in candidates}
    else:
        raise ValueError(
            "--cycle latest is only meaningful for gfs/hrrr; ERA5 is a "
            "reanalysis published with a delay of several days")
    if source == "hrrr":
        from gpuwm.hrrr_forecast import hrrr_cycle_horizon
        candidates = tuple(cycle for cycle in candidates
                           if last_hour <= hrrr_cycle_horizon(cycle))
    for cycle in candidates:
        if all(probe(url) for url in urls[cycle]):
            return cycle
    span = "48 h" if source == "gfs" else "12 h"
    raise RuntimeError(
        f"no complete {source.upper()} cycle covering f{last_hour:03d} was "
        f"found on AWS S3 within the last {span}; pass an explicit --cycle")


def resolve_hrrr_transport(cycle: datetime, requested: str, *,
                           last_hour: int, now: datetime | None = None,
                           probe=None, progress=print) -> str:
    """Pick the concrete HRRR transport for one fetch invocation.

    NOMADS serves byte-identical HRRR files and ``.idx`` indexes but
    keeps only about the newest :data:`HRRR_NOMADS_RETENTION_HOURS` of
    cycles, publishing each hour before the cloud mirrors.  ``auto``
    probes NOMADS for the requested window's final hour pair (both the
    ``wrfnat`` and ``wrfprs`` objects, mirroring the latest-cycle
    completeness rule) and falls back to S3 -- one decision per
    invocation, so a fetch never silently mixes hosts; the manifest
    records every file's actual URL and transport either way, and the
    downloaded subset bytes are identical from both.  An explicit
    ``nomads`` that NOMADS cannot serve refuses with the retention
    story rather than failing file by file mid-download.
    """

    if requested == "s3":
        return "s3"
    if requested not in ("auto", "nomads"):
        raise ValueError(
            f"unknown HRRR transport {requested!r}; expected one of "
            f"{HRRR_TRANSPORTS}")
    if probe is None:
        probe = _head_ok
    if now is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
    age_hours = (now - cycle).total_seconds() / 3600.0
    urls = (hrrr_object_url(cycle, last_hour, "wrfnat", transport="nomads"),
            hrrr_object_url(cycle, last_hour, "wrfprs", transport="nomads"))
    if requested == "nomads":
        if all(probe(url) for url in urls):
            return "nomads"
        detail = (
            f" -- the cycle is {age_hours:.0f} h old and NOMADS keeps only "
            f"about the newest {HRRR_NOMADS_RETENTION_HOURS} h"
            if age_hours > HRRR_NOMADS_RETENTION_HOURS else
            " (still publishing, or the window's final hour is not up yet;"
            " --wait-for downloads hours as they appear)")
        raise ValueError(
            f"--transport nomads: NOMADS is not serving cycle "
            f"{cycle:%Y-%m-%dT%H}Z through f{last_hour:02d}{detail}; use "
            "--transport s3 (the full archive) or auto")
    if age_hours > HRRR_NOMADS_RETENTION_HOURS:
        progress(
            f"fetch hrrr: cycle {cycle:%Y-%m-%dT%H}Z is {age_hours:.0f} h "
            f"old, beyond the ~{HRRR_NOMADS_RETENTION_HOURS} h NOMADS "
            "retention -- using the AWS S3 archive")
        return "s3"
    if all(probe(url) for url in urls):
        progress("fetch hrrr: using NOMADS (production mirror, identical "
                 "files and .idx indexes; S3 remains the fallback)")
        return "nomads"
    progress("fetch hrrr: NOMADS does not serve the full requested window "
             "yet -- falling back to the AWS S3 archive")
    return "s3"


# ---------------------------------------------------------------------------
# Shared transport helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_grib2_messages(path: Path) -> int:
    """Walk and validate every GRIB2 envelope; return the message count.

    Fail-closed transport check mirroring the GRIB1 envelope walk in
    :mod:`gpuwm.ingest.grib`: every message must declare edition 2, its
    exact length, and close with ``7777``; the messages must tile the
    file exactly.
    """

    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"GRIB2 file {path} is empty")
    count = 0
    with path.open("rb") as stream:
        offset = 0
        while offset < size:
            stream.seek(offset)
            header = stream.read(16)
            if len(header) != 16 or header[:4] != b"GRIB":
                raise ValueError(
                    f"invalid GRIB2 file {path}: message {count} at byte "
                    f"{offset} lacks a GRIB indicator")
            if header[7] != 2:
                raise ValueError(
                    f"unsupported GRIB edition {header[7]} in {path}, "
                    f"message {count} at byte {offset}")
            length = int.from_bytes(header[8:16], "big")
            end = offset + length
            if length < 20 or end > size:
                raise ValueError(
                    f"truncated GRIB2 file {path}: message {count} at byte "
                    f"{offset} declares {length} bytes, file has {size}")
            stream.seek(end - 4)
            if stream.read(4) != b"7777":
                raise ValueError(
                    f"invalid GRIB2 file {path}: message {count} at byte "
                    f"{offset} lacks the 7777 terminator")
            count += 1
            offset = end
    return count


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_fetch_manifest(out: Path, payload: dict) -> Path:
    path = out / FETCH_MANIFEST_NAME
    _atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _load_fetch_manifest(out: Path) -> dict | None:
    """The prior fetch manifest payload in ``out``, or None.

    Malformed or foreign JSON yields None rather than an error: the
    per-file completeness bars still apply, so an unreadable manifest
    only ever loses the request-identity comparison, never safety.
    """

    path = out / FETCH_MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError, OSError):
        return None
    if (not isinstance(payload, dict)
            or payload.get("schema") != FETCH_MANIFEST_SCHEMA):
        return None
    return payload


def check_prior_request(out: Path, *, source: str, cycle: datetime,
                        area: Area | None) -> None:
    """Refuse resuming into ``out`` unless the recorded request matches.

    The per-file resume check verifies envelopes and record counts, but a
    GFS subset carries the same 124 records for ANY area, so re-fetching
    a different area (or cycle) into the same ``--out`` would silently
    keep the old files.  The fetch manifest records the request; a
    source, cycle, or area difference refuses with the exact mismatch
    and the remedy.  Forecast-hour changes alone stay resumable: files
    are per-hour and byte-identical for the same source/cycle/area, so
    extending the window is safe by construction.

    A nonempty ``out`` WITHOUT a readable manifest refuses too: with no
    recorded request there is nothing to tie the existing files to (an
    interrupted fetch that died before its manifest, a corrupted
    manifest, or a directory some other tool wrote), and the per-file
    bars are area-blind, so resuming would bless files this request
    cannot verify.  Only a directory that is absent or empty may be
    fetched into without a manifest.
    """

    prior = _load_fetch_manifest(out)
    if prior is None:
        if out.is_dir() and any(out.iterdir()):
            raise ValueError(
                f"--out {out} is not empty but carries no readable "
                f"{FETCH_MANIFEST_NAME} (a fetch interrupted before its "
                "manifest was written, a corrupted manifest, or files "
                "another tool put there).  The existing files cannot be "
                "tied to any recorded source/cycle/area, and the "
                "per-file resume check is area-blind, so they are "
                "UNVERIFIED for this request and will not be resumed.\n"
                "  remedy: fetch into a different --out, or pass "
                "--force-refetch to move the existing files aside "
                "(nothing is deleted) and re-download this request.")
        return
    requested = {
        "source": source,
        "cycle": cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "area": None if area is None else area.as_manifest(),
    }
    existing = {key: prior.get(key) for key in requested}
    differences = [
        f"  {key}: requested {requested[key]!r}, but {out} was fetched "
        f"with {existing[key]!r}"
        for key in requested if requested[key] != existing[key]]
    if differences:
        raise ValueError(
            f"--out {out} already holds a fetch for a different request, "
            "and the per-file resume check cannot tell the difference "
            "(a subset file passes its record-count bar for any area):\n"
            + "\n".join(differences)
            + "\n  remedy: fetch into a different --out, or pass "
            "--force-refetch to move the existing files aside (nothing "
            "is deleted) and re-download this request.")


def _manifest_payload(*, source: str, cycle: datetime,
                      hours: tuple[int, ...], area: Area | None,
                      files: list[dict]) -> dict:
    return {
        "schema": FETCH_MANIFEST_SCHEMA,
        "source": source,
        "cycle": cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_hours": list(hours),
        "area": None if area is None else area.as_manifest(),
        "files": files,
        "payload_bytes": sum(item["bytes"] for item in files),
        "created": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    }


# ---------------------------------------------------------------------------
# GFS
# ---------------------------------------------------------------------------

def fetch_gfs(*, cycle: datetime, hours: tuple[int, ...], area: Area,
              out: Path, progress=print, force: bool = False) -> Path:
    """Download the exact GFS pgrb2.0p25 subset series into ``out``.

    Reuses the certified NOMADS query builder and downloader in
    :mod:`tools.download_gfs_native_subset` (single source for the
    124-record selection), adds resumability (a present, envelope-valid
    subset is never re-downloaded), and writes the ``gfs-series.tsv``
    that ``rw-wps --source gfs --gfs-series`` / ``gfs_grib2_bridge``
    consume, plus the fetch manifest.  An existing file must pass the
    envelope walk, the exact 124-record count, AND -- when the prior
    fetch manifest recorded its digest -- that same sha256; a swapped
    file is never re-blessed by the area-blind count alone.  ``force``
    moves every existing subset aside (never deletes) and re-downloads.
    """

    from tools import download_gfs_native_subset as transport

    out.mkdir(parents=True, exist_ok=True)
    prior_digests = _prior_manifest_digests(out)
    box = area.as_nomads()
    files: list[dict] = []
    for hour in hours:
        name = f"gfs.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}.subset.grib2"
        path = out / name
        url = transport.nomads_query(cycle, hour, **box)
        if force and path.exists():
            _quarantine_rejected(path, progress, f"gfs f{hour:03d}")
        if path.exists():
            observed = count_grib2_messages(path)
            if observed != GFS_SUBSET_RECORD_COUNT:
                raise ValueError(
                    f"existing {name} carries {observed} GRIB2 messages, "
                    f"expected {GFS_SUBSET_RECORD_COUNT}; move it aside "
                    "and re-fetch")
            digest = sha256_file(path)
            recorded = prior_digests.get(name)
            if recorded is not None and digest != recorded:
                raise ValueError(
                    f"existing {name} does not match the sha256 recorded "
                    "in the prior fetch manifest, so it cannot be resumed "
                    "for this request; pass --force-refetch to move the "
                    "existing files aside (nothing is deleted) and "
                    "re-download")
            progress(f"fetch gfs f{hour:03d}: {name} exists, "
                     f"{path.stat().st_size:,} B verified -- skipped")
        else:
            started = time.perf_counter()
            transport._download(url, path)
            observed = count_grib2_messages(path)
            if observed != GFS_SUBSET_RECORD_COUNT:
                raise ValueError(
                    f"NOMADS returned {observed} GRIB2 messages for "
                    f"f{hour:03d}, expected {GFS_SUBSET_RECORD_COUNT}; the "
                    "upstream inventory has drifted")
            digest = sha256_file(path)
            progress(f"fetch gfs f{hour:03d}: {name} "
                     f"{path.stat().st_size:,} B in "
                     f"{time.perf_counter() - started:.1f} s")
        files.append({
            "name": name, "role": "gfs-subset", "forecast_hour": hour,
            "bytes": path.stat().st_size, "sha256": digest,
            "url": url,
        })
    # Relative names: gfs_grib2_bridge resolves them against the TSV's
    # own directory, so the fetched directory stays relocatable.
    series = out / "gfs-series.tsv"
    _atomic_write_text(series, "".join(
        f"{hour}\t{item['name']}\n" for hour, item in zip(hours, files)))
    files.append({
        "name": series.name, "role": "series", "forecast_hour": None,
        "bytes": series.stat().st_size, "sha256": sha256_file(series),
        "url": None,
    })
    payload = _manifest_payload(source="gfs", cycle=cycle, hours=hours,
                                area=area, files=files)
    payload["notes"] = (
        "NOMADS filter subsets (south-to-north 0.25-degree grids); raw "
        "noaa-gfs-bdp-pds S3 objects are north-to-south and are rejected "
        "by the fail-closed gfs_grib2_bridge")
    return write_fetch_manifest(out, payload)


def author_gfs_front_door_manifest(
        *, out: Path, bridge: Path, wps_namelist: Path,
        experiment_config: Path, static_input: Path | None = None,
        static_receipt: Path | None = None,
        manifest_out: Path | None = None,
        progress=print) -> tuple[Path, str]:
    """Write the exact input manifest the GFS front door verifies.

    Bridges ``gpuwm fetch`` to ``rw-wps --source gfs``: the front door
    (gpuwm/gfs_direct.py ``_verify_input_manifest``) demands a
    ``gpuwm-gfs-direct-input-manifest-v1`` document binding every input
    role -- ``series``/``bridge``/``wps_namelist``/``experiment_config``
    (plus the optional static pair) and one ``grib-fNNN`` per forecast
    hour -- to its basename and sha256, INCLUDING the bridge
    executable's own hash, plus a ``source`` identity block.  This
    function derives the cycle and hour inventory from the fetch
    manifest in ``out``, hashes every role from disk, writes the
    document, and returns ``(path, sha256-of-the-document)`` -- the pair
    ``--source-manifest``/``--source-manifest-sha256`` wants verbatim.
    """

    prior = _load_fetch_manifest(out)
    if prior is None or prior.get("source") != "gfs":
        raise ValueError(
            f"{out / FETCH_MANIFEST_NAME} is not a completed "
            "`gpuwm fetch --source gfs` output; run the fetch first "
            "(the front-door manifest binds the fetched cycle and "
            "forecast-hour inventory)")
    if (static_input is None) != (static_receipt is None):
        raise ValueError(
            "--static-input and --static-receipt must be supplied "
            "together (or neither, when the front door builds statics "
            "from --geog-root)")
    cycle_text = str(prior.get("cycle", ""))
    try:
        cycle = datetime.strptime(cycle_text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError(
            f"fetch manifest in {out} carries an unreadable cycle "
            f"{cycle_text!r}") from error
    hours = prior.get("forecast_hours")
    if (not isinstance(hours, list) or not hours
            or not all(isinstance(hour, int) for hour in hours)):
        raise ValueError(
            f"fetch manifest in {out} lacks a forecast-hour inventory")
    subset_names = {
        item.get("forecast_hour"): item.get("name")
        for item in prior.get("files", ())
        if isinstance(item, dict) and item.get("role") == "gfs-subset"}
    roles: dict[str, Path] = {
        "series": out / "gfs-series.tsv",
        "bridge": Path(bridge),
        "wps_namelist": Path(wps_namelist),
        "experiment_config": Path(experiment_config),
    }
    if static_input is not None:
        roles["static_input"] = Path(static_input)
        roles["static_receipt"] = Path(static_receipt)
    for hour in hours:
        name = subset_names.get(hour)
        if not isinstance(name, str):
            raise ValueError(
                f"fetch manifest in {out} lists forecast hour {hour} "
                "without a gfs-subset file entry")
        roles[f"grib-f{hour:03d}"] = out / name
    missing = sorted(
        f"{role}: {path}" for role, path in roles.items()
        if not path.is_file())
    if missing:
        raise ValueError(
            "front-door manifest inputs are missing:\n  "
            + "\n  ".join(missing))
    payload = {
        "schema": GFS_FRONT_DOOR_MANIFEST_SCHEMA,
        "source": {
            "model": "GFS",
            "product": "pgrb2.0p25",
            "cycle": cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "files": {
            role: {"name": path.name, "sha256": sha256_file(path)}
            for role, path in roles.items()
        },
    }
    path = (Path(manifest_out) if manifest_out is not None
            else out / GFS_INPUT_MANIFEST_NAME)
    _atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = sha256_file(path)
    static_args = (
        f" --static-input {static_input}"
        f" --static-receipt {static_receipt}"
        if static_input is not None else " --geog-root WPS_GEOG_DIR")
    progress(f"fetch gfs: front-door manifest {path}")
    progress(f"fetch gfs: front-door manifest sha256 {digest}")
    progress(
        "fetch gfs: feed the GFS front door with:\n"
        f"  rw-wps --source gfs --gfs-series {roles['series']}"
        f" --cycle {cycle:%Y-%m-%d_%H:%M:%S}"
        f" --bridge {bridge}"
        f" --wps-namelist {wps_namelist}"
        f" --experiment-config {experiment_config}"
        f" --source-manifest {path}"
        f" --source-manifest-sha256 {digest}"
        f"{static_args} --output-root OUTPUT_DIR")
    return path, digest


# ---------------------------------------------------------------------------
# HRRR
# ---------------------------------------------------------------------------

def _prior_manifest_digests(out: Path) -> dict[str, str]:
    """``name -> sha256`` from an existing fetch manifest, else empty.

    A malformed or foreign manifest yields no digests rather than an
    error: the record-count bar below still applies to every existing
    file, so the digest map only ever *adds* strictness.
    """

    path = out / FETCH_MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {item["name"]: item["sha256"]
                for item in payload.get("files", ())
                if isinstance(item.get("name"), str)
                and isinstance(item.get("sha256"), str)}
    except (ValueError, TypeError, AttributeError):
        return {}


def _existing_hrrr_digest(dest: Path, *, expected_count: int,
                          prior_digest: str | None, progress,
                          label: str) -> str | None:
    """sha256 of a verified existing subset, or None to force re-download.

    An existing file passes the SAME completeness bar as a fresh
    download: the full GRIB2 envelope walk plus the exact expected
    record count (a nonempty prefix truncated at a message boundary
    walks clean but fails the count).  When a prior fetch manifest
    recorded a digest for this name, the file must also still match it;
    a failing file is never digest-blessed -- it is re-downloaded.
    """

    try:
        observed = count_grib2_messages(dest)
    except ValueError as error:
        progress(f"fetch hrrr {label}: existing {dest.name} failed "
                 f"envelope validation ({error}); re-downloading")
        return None
    if observed != expected_count:
        progress(f"fetch hrrr {label}: existing {dest.name} carries "
                 f"{observed} GRIB2 messages, expected {expected_count} "
                 "(truncated or drifted); re-downloading")
        return None
    digest = sha256_file(dest)
    if prior_digest is not None and digest != prior_digest:
        progress(f"fetch hrrr {label}: existing {dest.name} does not "
                 "match the sha256 recorded in the prior fetch manifest; "
                 "re-downloading")
        return None
    return digest


def _quarantine_rejected(dest: Path, progress, label: str) -> None:
    """Move a failed existing file aside (never deleted, never reused)."""

    aside = dest.with_name(f"{dest.name}.rejected-{time.time_ns()}")
    os.replace(dest, aside)
    progress(f"fetch {label}: moved rejected file aside to "
             f"{aside.name}")


def _validate_hrrr_area(area: Area) -> None:
    if (area.lat_south < HRRR_CONUS_LAT[0]
            or area.lat_north > HRRR_CONUS_LAT[1]
            or area.lon_west < HRRR_CONUS_LON[0]
            or area.lon_east > HRRR_CONUS_LON[1]):
        raise ValueError(
            "requested area extends beyond HRRR CONUS coverage "
            f"(lat {HRRR_CONUS_LAT[0]}..{HRRR_CONUS_LAT[1]}, "
            f"lon {HRRR_CONUS_LON[0]}..{HRRR_CONUS_LON[1]}); "
            "use --source gfs for domains outside CONUS")


def _wait_for_hrrr_product(*, cycle: datetime, hour: int, product: str,
                           candidates: tuple[str, ...], probe, clock,
                           deadline: float, sleeper, progress,
                           label: str) -> str | None:
    """Block until a transport serves the object AND its ``.idx``.

    Returns the transport name, or None when the deadline expires.
    Candidates are tried in order each round (NOMADS first under
    ``auto``, because the production mirror publishes first); rounds
    are separated by at most :data:`HRRR_WAIT_POLL_SECONDS`.  Both the
    object and its index must answer: the ``.idx`` can lag its GRIB by
    a moment, and the range transport needs both.
    """

    announced = False
    while True:
        for name in candidates:
            url = hrrr_object_url(cycle, hour, product, transport=name)
            if probe(url) and probe(url + ".idx"):
                if announced:
                    progress(f"fetch hrrr {label}: published on {name}")
                return name
        remaining = deadline - clock()
        if remaining <= 0:
            return None
        if not announced:
            progress(f"fetch hrrr {label}: not yet published on "
                     f"{'/'.join(candidates)}; polling every "
                     f"{HRRR_WAIT_POLL_SECONDS} s (up to "
                     f"{remaining / 60.0:.0f} more min)")
            announced = True
        sleeper(min(HRRR_WAIT_POLL_SECONDS, remaining))


def fetch_hrrr(*, cycle: datetime, hours: tuple[int, ...],
               area: Area | None, out: Path, workers: int = 8,
               retries: int = 5, progress=print,
               force: bool = False, transport: str = "s3",
               wait: bool = False,
               wait_timeout_s: float = HRRR_WAIT_TIMEOUT_DEFAULT_MINUTES * 60,
               probe=None, sleeper=time.sleep,
               clock=time.monotonic) -> Path:
    """Byte-range download the native HRRR subset series into ``out``.

    Reuses the proven ``.idx`` selection/range transport in
    :mod:`tools.download_hrrr_native_subset` per product (atmosphere
    ``wrfnat`` subset + soil records of ``wrfprs``), adds resumability,
    and writes the fetch manifest plus ``SHA256SUMS``.  An existing file
    is skipped only when it passes the same completeness bar as a fresh
    download (envelope walk + the exact 561/18 record counts) and, when
    a prior manifest recorded its digest, still matches that digest;
    anything else is moved aside and re-downloaded, never re-blessed.
    HRRR files are CONUS-wide: ``--area`` is a coverage check, not a
    crop.

    ``transport`` selects the host ('s3' or 'nomads'; both serve
    byte-identical files and indexes, so every bar above is
    host-independent and a directory fetched over one host resumes over
    the other).  ``wait`` is the live-cycle mode: hours are fetched in
    order, each product polled (at most every
    :data:`HRRR_WAIT_POLL_SECONDS` seconds) until it publishes; under
    ``transport='auto'`` each round tries NOMADS first, then S3.  On
    timeout the manifest is still written for the contiguous complete
    prefix -- so a re-run of the same command resumes instead of
    refusing -- and a ``RuntimeError`` reports exactly what was and was
    not fetched.
    """

    from tools import download_hrrr_native_subset as range_transport

    expected_counts = {"atmosphere": range_transport.ATMOSPHERE_RECORD_COUNT,
                       "soil": range_transport.SOIL_RECORD_COUNT}
    candidates = (("nomads", "s3") if transport == "auto"
                  else (transport,))
    for name in candidates:
        _hrrr_transport_base(name)  # unknown transports fail before I/O
    if transport == "auto" and not wait:
        raise ValueError(
            "transport 'auto' reaches fetch_hrrr only in --wait-for mode "
            "(per-file polling); a plain fetch resolves it first via "
            "resolve_hrrr_transport, as the CLI does")
    if wait and (not math.isfinite(wait_timeout_s) or wait_timeout_s <= 0):
        raise ValueError("the --wait-for timeout must be positive")
    if probe is None:
        probe = _head_ok
    if area is not None:
        _validate_hrrr_area(area)
    out.mkdir(parents=True, exist_ok=True)
    prior_digests = _prior_manifest_digests(out)
    deadline = clock() + wait_timeout_s
    files: list[dict] = []
    complete_hours: list[int] = []

    def publish_manifest(recorded_hours: tuple[int, ...]) -> Path:
        sums = out / "SHA256SUMS"
        _atomic_write_text(sums, "".join(
            f"{item['sha256']}  {item['name']}\n"
            for item in sorted(files, key=lambda item: item["name"])))
        entries = files + [{
            "name": sums.name, "role": "checksums", "forecast_hour": None,
            "bytes": sums.stat().st_size, "sha256": sha256_file(sums),
            "url": None, "transport": None,
        }]
        payload = _manifest_payload(source="hrrr", cycle=cycle,
                                    hours=recorded_hours, area=area,
                                    files=entries)
        payload["notes"] = (
            "native hybrid-level wrfnat subsets plus wrfprs soil records, "
            "the exact inventory hrrr_grib2_bridge requires; CONUS-wide "
            "(idx subsetting selects records, not areas); NOMADS and S3 "
            "serve byte-identical files, so per-file transports may mix "
            "across resumed runs without weakening the digest bars")
        return write_fetch_manifest(out, payload)

    for hour in hours:
        atmosphere = f"hrrr.t{cycle:%H}z.wrfnatf{hour:02d}.grib2"
        pressure = f"hrrr.t{cycle:%H}z.wrfprsf{hour:02d}.grib2"
        soil = f"hrrr.t{cycle:%H}z.soilf{hour:02d}.grib2"
        for kind, source_name, dest_name, product in (
                ("atmosphere", atmosphere, atmosphere, "wrfnat"),
                ("soil", pressure, soil, "wrfprs")):
            dest = out / dest_name
            expected = expected_counts[kind]
            label = f"f{hour:02d} {kind}"
            digest = None
            if dest.exists() and not force:
                digest = _existing_hrrr_digest(
                    dest, expected_count=expected,
                    prior_digest=prior_digests.get(dest_name),
                    progress=progress, label=label)
            if digest is not None:
                chosen = candidates[0]
                url = hrrr_object_url(cycle, hour, product,
                                      transport=chosen)
                progress(f"fetch hrrr {label}: {dest_name} exists, "
                         f"{dest.stat().st_size:,} B / {expected} records "
                         "verified -- skipped")
            else:
                if dest.exists():
                    _quarantine_rejected(dest, progress, f"hrrr {label}")
                chosen = candidates[0]
                if wait:
                    found = _wait_for_hrrr_product(
                        cycle=cycle, hour=hour, product=product,
                        candidates=candidates, probe=probe, clock=clock,
                        deadline=deadline, sleeper=sleeper,
                        progress=progress, label=label)
                    if found is None:
                        publish_manifest(tuple(complete_hours))
                        fetched = (
                            f"complete hours f{complete_hours[0]:02d}.."
                            f"f{complete_hours[-1]:02d} are on disk and "
                            "recorded in the fetch manifest"
                            if complete_hours else
                            "no complete forecast hour was fetched")
                        raise RuntimeError(
                            f"--wait-for timed out after "
                            f"{wait_timeout_s / 60.0:.0f} min: {dest_name} "
                            f"(cycle {cycle:%Y-%m-%dT%H}Z) never appeared "
                            f"on {'/'.join(candidates)}.  {fetched}; "
                            "re-running the same command resumes the "
                            "verified files and extends the window.")
                    chosen = found
                url = hrrr_object_url(cycle, hour, product,
                                      transport=chosen)
                request = range_transport.ProductRequest(
                    url=url,
                    index_url=url + ".idx",
                    index_path=out / f"{source_name}.idx",
                    destination=dest,
                    kind=kind,
                )
                started = time.perf_counter()
                range_transport._download_product(
                    request, workers=workers, retries=retries)
                observed = count_grib2_messages(dest)
                if observed != expected:
                    raise ValueError(
                        f"downloaded {dest_name} carries {observed} GRIB2 "
                        f"messages, expected {expected}; the upstream "
                        ".idx inventory has drifted")
                digest = sha256_file(dest)
                progress(f"fetch hrrr {label}: {dest_name} "
                         f"{dest.stat().st_size:,} B in "
                         f"{time.perf_counter() - started:.1f} s "
                         f"({chosen})")
            files.append({
                "name": dest_name, "role": kind, "forecast_hour": hour,
                "bytes": dest.stat().st_size, "sha256": digest,
                "url": url, "transport": chosen,
            })
        complete_hours.append(hour)
    return publish_manifest(hours)


# ---------------------------------------------------------------------------
# ERA5: cdsapi template + user-file validation
# ---------------------------------------------------------------------------

def _era5_times(cycle: datetime, hours: int,
                cadence: int) -> tuple[datetime, ...]:
    if hours < cadence or hours % cadence:
        raise ValueError(
            f"--hours must be a positive multiple of the {cadence} h "
            "cadence")
    return tuple(cycle + timedelta(hours=lead)
                 for lead in range(0, hours + 1, cadence))


def era5_request_template(*, cycle: datetime, hours: int, area: Area,
                          cadence: int = 6) -> dict:
    """The exact two-part cdsapi request gpuwm's ERA5 ingest expects."""

    times = _era5_times(cycle, hours, cadence)
    dates = sorted({when.strftime("%Y-%m-%d") for when in times})
    clock = sorted({when.strftime("%H:00") for when in times})
    shared = {
        "product_type": "reanalysis",
        "data_format": "grib",
        "date": dates,
        "time": clock,
        "area": area.as_cds(),
    }
    pressure = dict(shared)
    pressure["variable"] = [
        "geopotential", "temperature", "u_component_of_wind",
        "v_component_of_wind", "relative_humidity",
    ]
    pressure["pressure_level"] = [
        str(level) for level in ERA5_PRESSURE_LEVELS_HPA]
    single = dict(shared)
    single["variable"] = [
        "geopotential", "surface_pressure", "mean_sea_level_pressure",
        "10m_u_component_of_wind", "10m_v_component_of_wind",
        "2m_temperature", "2m_dewpoint_temperature", "land_sea_mask",
        "skin_temperature", "sea_surface_temperature", "sea_ice_cover",
        "snow_depth",
        "soil_temperature_level_1", "soil_temperature_level_2",
        "soil_temperature_level_3", "soil_temperature_level_4",
        "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3", "volumetric_soil_water_layer_4",
    ]
    return {
        "schema": "gpuwm-era5-cds-request-v1",
        "requires": "CDS account + ~/.cdsapirc key; pip install cdsapi",
        "requests": [
            {"dataset": "reanalysis-era5-pressure-levels",
             "target": "era5-pressure.grib", "request": pressure},
            {"dataset": "reanalysis-era5-single-levels",
             "target": "era5-single.grib", "request": single},
        ],
        "combine": ("concatenate the two GRIB1 targets into one file "
                    "(byte concatenation preserves every message): "
                    "era5-combined.grib"),
        "validate": "gpuwm fetch --source era5 --validate era5-combined.grib",
    }


ERA5_INSTRUCTIONS = """\
ERA5 acquisition is manual: the Copernicus CDS API requires a personal
account and key, which gpuwm will not embed.

1. Create an account at https://cds.climate.copernicus.eu and write your
   key to ~/.cdsapirc as documented there.
2. pip install cdsapi
3. Run each request in {request_file}:
       import cdsapi, json
       spec = json.load(open({request_file_repr}))
       client = cdsapi.Client()
       for item in spec["requests"]:
           client.retrieve(item["dataset"], item["request"], item["target"])
4. Concatenate the two GRIB files into one (plain byte concatenation):
       python -c "open('era5-combined.grib','wb').write(
           open('era5-pressure.grib','rb').read()
           + open('era5-single.grib','rb').read())"
5. Validate the result against what gpuwm ingest expects:
       gpuwm fetch --source era5 --validate era5-combined.grib
"""


def write_era5_request(*, cycle: datetime, hours: int, area: Area,
                       out: Path, cadence: int = 6,
                       progress=print) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    template = era5_request_template(
        cycle=cycle, hours=hours, area=area, cadence=cadence)
    path = out / ERA5_REQUEST_NAME
    _atomic_write_text(
        path, json.dumps(template, indent=2, sort_keys=True) + "\n")
    progress(f"fetch era5: wrote {path}")
    progress(ERA5_INSTRUCTIONS.format(
        request_file=path, request_file_repr=repr(str(path))))
    return path


@dataclass(frozen=True)
class Grib1Record:
    """Transport-level identity of one GRIB1 message."""

    parameter: int
    level_type: int
    level: int
    valid_time: datetime


_GRIB1_TIME_UNITS = {
    0: timedelta(minutes=1), 1: timedelta(hours=1), 2: timedelta(days=1),
    10: timedelta(hours=3), 11: timedelta(hours=6),
    12: timedelta(hours=12), 254: timedelta(seconds=1),
}


def read_grib1_records(path: Path) -> tuple[Grib1Record, ...]:
    """Envelope-validate ``path`` and read each message's PDS identity.

    Reuses :func:`gpuwm.ingest.grib.inspect_grib1_envelopes` for the
    strict transport walk, then reads only Product Definition Section
    header bytes -- parameter, level, and reference/valid time.  No
    scientific payload is decoded.
    """

    from gpuwm.ingest.grib import inspect_grib1_envelopes

    envelopes = inspect_grib1_envelopes(path)
    records: list[Grib1Record] = []
    with path.open("rb") as stream:
        for envelope in envelopes:
            stream.seek(envelope.offset + 8)
            pds = stream.read(28)
            if len(pds) < 28:
                raise ValueError(
                    f"GRIB1 message {envelope.index} in {path} has a "
                    "truncated PDS")
            parameter = pds[8]
            level_type = pds[9]
            level = int.from_bytes(pds[10:12], "big")
            year_of_century, month, day, hour, minute = pds[12:17]
            unit, p1, p2, time_range = pds[17], pds[18], pds[19], pds[20]
            century = pds[24]
            year = (century - 1) * 100 + year_of_century
            reference = datetime(year, month, day, hour, minute)
            if unit not in _GRIB1_TIME_UNITS:
                raise ValueError(
                    f"unsupported GRIB1 forecast time unit {unit} in "
                    f"{path} message {envelope.index}")
            if time_range in (0, 1):
                lead = p1 if time_range == 0 else 0
            elif time_range == 10:
                lead = (p1 << 8) | p2
            else:
                raise ValueError(
                    f"unsupported GRIB1 time range indicator {time_range} "
                    f"in {path} message {envelope.index}; ERA5 analyses "
                    "are instantaneous")
            valid_time = reference + lead * _GRIB1_TIME_UNITS[unit]
            records.append(
                Grib1Record(parameter, level_type, level, valid_time))
    return tuple(records)


@dataclass(frozen=True)
class Era5ValidationReport:
    """Census of a user-supplied ERA5 GRIB1 set vs ingest expectations."""

    failures: tuple[str, ...]
    checks: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures

    def format(self) -> str:
        lines = [f"era5 validation: "
                 f"{'PASS' if self.ok else 'FAIL'}"]
        lines.extend(f"  ok: {check}" for check in self.checks)
        lines.extend(f"  FAIL: {failure}" for failure in self.failures)
        return "\n".join(lines)


def validate_era5_files(
        paths: tuple[Path, ...], *,
        expected_times: tuple[datetime, ...] | None = None,
) -> Era5ValidationReport:
    """Validate a user-supplied ERA5 GRIB1 file set for gpuwm ingest.

    Covers: strict GRIB1 transport envelopes (edition, declared lengths,
    ``7777`` terminators, exact EOF coverage), the required
    pressure-level and surface parameter inventory at every valid time,
    identical pressure-level ladders across variables and times, soil
    encodings at level type 112 or 1, invariant orography presence, and
    (when the caller supplies ``expected_times``) valid-time coverage.
    It does not decode data values -- the Rust GRIB1 bridge re-validates
    and decodes at ingest time.
    """

    failures: list[str] = []
    checks: list[str] = []
    records: list[Grib1Record] = []
    for path in paths:
        if not path.is_file():
            failures.append(f"missing input file {path}")
            continue
        try:
            found = read_grib1_records(path)
        except ValueError as error:
            failures.append(str(error))
            continue
        checks.append(f"{path.name}: {len(found)} valid GRIB1 envelopes")
        records.extend(found)
    if failures:
        return Era5ValidationReport(tuple(failures), tuple(checks))

    times = sorted({record.valid_time for record in records})
    if not times:
        return Era5ValidationReport(
            ("no GRIB1 messages found in the supplied files",),
            tuple(checks))
    checks.append(
        f"{len(times)} valid times {times[0].isoformat()} .. "
        f"{times[-1].isoformat()}")
    if expected_times is not None:
        missing = sorted(set(expected_times) - set(times))
        if missing:
            failures.append(
                "missing valid times: "
                + ", ".join(when.isoformat() for when in missing))
        else:
            checks.append("requested cycle/hours window fully covered")

    # Pressure-level census: required variables at identical ladders.
    ladders: dict[tuple[int, datetime], set[int]] = {}
    for record in records:
        if record.level_type == 100:
            ladders.setdefault(
                (record.parameter, record.valid_time), set()
            ).add(record.level)
    reference_ladder: set[int] | None = None
    for parameter, short in sorted(ERA5_REQUIRED_PRESSURE.items()):
        per_time = [ladders.get((parameter, when)) for when in times]
        if any(levels is None for levels in per_time):
            failures.append(
                f"pressure-level {short} (GRIB1 parameter {parameter}) is "
                "absent at one or more valid times")
            continue
        if len({frozenset(levels) for levels in per_time}) != 1:
            failures.append(
                f"pressure-level {short} ladder differs across valid times")
            continue
        if reference_ladder is None:
            reference_ladder = per_time[0]
        elif per_time[0] != reference_ladder:
            failures.append(
                f"pressure-level {short} ladder differs from the other "
                "variables")
    if reference_ladder is not None:
        checks.append(
            f"pressure ladder: {len(reference_ladder)} levels "
            f"{min(reference_ladder)}..{max(reference_ladder)} hPa, "
            "identical across required variables and times")

    # Surface census.  Soil layers may arrive as level type 112 (native
    # CDS) or 1 (CDO-normalized); everything else must be level type 1.
    surface: dict[int, set[datetime]] = {}
    invariant_orography = False
    for record in records:
        accepted_types = (
            (1, 112) if record.parameter in ERA5_SOIL_PARAMETERS else (1,))
        if record.level_type not in accepted_types:
            continue
        surface.setdefault(record.parameter, set()).add(record.valid_time)
        if (record.parameter == ERA5_OROGRAPHY_PARAMETER
                and record.level_type == 1):
            invariant_orography = True
    for parameter, short in sorted(ERA5_REQUIRED_SURFACE.items()):
        present = surface.get(parameter, set())
        if not present >= set(times):
            failures.append(
                f"surface {short} (GRIB1 parameter {parameter}) is absent "
                "at one or more valid times")
    if not invariant_orography:
        failures.append(
            "invariant geopotential (parameter 129 at the surface) is "
            "absent: request 'geopotential' in the single-levels dataset, "
            "or declare a per-domain source-orography supplement in "
            "[case_data]")
    optional = sorted(
        short for parameter, short in ERA5_OPTIONAL_SURFACE.items()
        if parameter in surface)
    if optional:
        checks.append(f"optional fields present: {', '.join(optional)}")
    required_present = sorted(
        short for parameter, short in ERA5_REQUIRED_SURFACE.items()
        if surface.get(parameter, set()) >= set(times))
    if required_present:
        checks.append(
            f"required surface fields at every time: "
            f"{', '.join(required_present)}")
    return Era5ValidationReport(tuple(failures), tuple(checks))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_area(args) -> Area | None:
    if args.area is not None and args.point is not None:
        raise ValueError("--area and --point are mutually exclusive")
    if args.area is not None:
        if args.radius_km is not None:
            raise ValueError("--radius-km belongs to --point, not --area")
        return parse_area(args.area)
    if args.point is not None:
        if args.radius_km is None:
            raise ValueError("--point requires --radius-km")
        return area_from_point(args.point, args.radius_km)
    if args.radius_km is not None:
        raise ValueError("--radius-km requires --point")
    return None


def fetch_main(args) -> int:
    source = args.source
    area = _resolve_area(args)

    if args.validate is not None and source != "era5":
        raise ValueError("--validate applies to --source era5 only")

    if source != "hrrr":
        hrrr_only = sorted(
            flag for flag, value in (
                ("--transport", args.transport),
                ("--wait-for", args.wait_for or None),
                ("--wait-timeout-minutes", args.wait_timeout_minutes),
            ) if value is not None)
        if hrrr_only:
            raise ValueError(
                f"{', '.join(hrrr_only)}: --source hrrr only (GFS rides "
                "the NOMADS grib filter already; ERA5 is a manual CDS "
                "retrieval)")

    author_roles = {
        "--bridge": args.bridge,
        "--wps-namelist": args.wps_namelist,
        "--experiment-config": args.experiment_config,
        "--static-input": args.static_input,
        "--static-receipt": args.static_receipt,
        "--manifest-out": args.manifest_out,
    }
    if args.author_front_door_manifest:
        if source != "gfs":
            raise ValueError(
                "--author-front-door-manifest applies to --source gfs "
                "only (the HRRR front door consumes the fetched "
                "SHA256SUMS directly; its handoff line is printed after "
                "every HRRR fetch)")
        required = ("--bridge", "--wps-namelist", "--experiment-config")
        absent = [flag for flag in required if author_roles[flag] is None]
        if absent or args.out is None:
            raise ValueError(
                "--author-front-door-manifest requires --out plus "
                + ", ".join(required)
                + " (the front-door manifest binds each file's sha256, "
                "including the bridge executable's)")
    else:
        supplied = sorted(
            flag for flag, value in author_roles.items()
            if value is not None)
        if supplied:
            raise ValueError(
                f"{', '.join(supplied)} belong to "
                "--author-front-door-manifest")

    if (args.author_front_door_manifest and args.cycle is None
            and args.hours is None):
        # Author-only: convert an already-completed fetch directory.
        author_gfs_front_door_manifest(
            out=args.out, bridge=args.bridge,
            wps_namelist=args.wps_namelist,
            experiment_config=args.experiment_config,
            static_input=args.static_input,
            static_receipt=args.static_receipt,
            manifest_out=args.manifest_out)
        return 0
    if source == "era5" and args.validate:
        expected = None
        if args.cycle is not None and args.hours is not None:
            cycle = parse_cycle(args.cycle, source)
            cadence = args.cadence if args.cadence is not None else 6
            expected = _era5_times(cycle, args.hours, cadence)
        report = validate_era5_files(
            tuple(args.validate), expected_times=expected)
        print(report.format())
        return 0 if report.ok else 1

    if args.cycle is None or args.hours is None or args.out is None:
        raise ValueError(
            "fetch requires --cycle, --hours, and --out (the exceptions: "
            "era5 --validate mode, and gfs --author-front-door-manifest "
            "on an already-fetched --out, which needs neither --cycle "
            "nor --hours)")
    if args.hours < 1:
        raise ValueError("--hours must be a positive forecast window")

    if source == "era5":
        if area is None:
            raise ValueError("era5 fetch requires --area or --point")
        if args.cycle == "latest":
            raise ValueError(
                "--cycle latest is not meaningful for ERA5: the reanalysis "
                "is published with a delay of several days; pass an "
                "explicit --cycle")
        cadence = args.cadence if args.cadence is not None else 6
        write_era5_request(
            cycle=parse_cycle(args.cycle, source), hours=args.hours,
            area=area, out=args.out, cadence=cadence)
        return 0

    if source == "gfs":
        if area is None:
            raise ValueError(
                "gfs fetch requires --area or --point --radius-km: the "
                "NOMADS subsetter needs a subregion (fetching the whole "
                "globe is never what an experiment needs)")
        cadence = args.cadence if args.cadence is not None else 3
        hours = gfs_forecast_hours(args.hours, cadence)
        if args.cycle == "latest":
            cycle = resolve_latest_cycle("gfs", hours[-1])
            print(f"fetch gfs: latest complete cycle is "
                  f"{cycle:%Y-%m-%dT%H}Z")
        else:
            cycle = parse_cycle(args.cycle, source)
        if not args.force_refetch:
            check_prior_request(args.out, source="gfs", cycle=cycle,
                                area=area)
        manifest = fetch_gfs(cycle=cycle, hours=hours, area=area,
                             out=args.out, force=args.force_refetch)
    elif source == "hrrr":
        if args.cadence is not None:
            raise ValueError("HRRR is hourly; --cadence does not apply")
        if args.wait_timeout_minutes is not None and not args.wait_for:
            raise ValueError(
                "--wait-timeout-minutes belongs to --wait-for")
        if args.wait_timeout_minutes is not None \
                and args.wait_timeout_minutes <= 0:
            raise ValueError("--wait-timeout-minutes must be positive")
        transport = args.transport if args.transport is not None else "auto"
        if args.cycle == "latest":
            if args.wait_for:
                # Wait mode wants the cycle currently PUBLISHING, so the
                # completeness probe is f00 (has publication begun?), not
                # the final requested hour.
                cycle = resolve_latest_cycle("hrrr", 0)
                print(f"fetch hrrr: latest publishing cycle is "
                      f"{cycle:%Y-%m-%dT%H}Z (f00 probe; --wait-for "
                      "downloads later hours as they appear)")
            else:
                cycle = resolve_latest_cycle("hrrr", args.hours)
                print(f"fetch hrrr: latest complete cycle is "
                      f"{cycle:%Y-%m-%dT%H}Z")
        else:
            cycle = parse_cycle(args.cycle, source)
        hours = hrrr_forecast_hours(args.hours, cycle)
        if not args.force_refetch:
            check_prior_request(args.out, source="hrrr", cycle=cycle,
                                area=area)
        if not args.wait_for:
            # One transport decision per invocation; 'auto' probes
            # NOMADS for the window's final hour pair and falls back S3.
            transport = resolve_hrrr_transport(
                cycle, transport, last_hour=hours[-1])
        timeout_minutes = (args.wait_timeout_minutes
                           if args.wait_timeout_minutes is not None
                           else HRRR_WAIT_TIMEOUT_DEFAULT_MINUTES)
        manifest = fetch_hrrr(cycle=cycle, hours=hours, area=area,
                              out=args.out, force=args.force_refetch,
                              transport=transport, wait=args.wait_for,
                              wait_timeout_s=timeout_minutes * 60.0)
    else:
        raise ValueError(f"unknown fetch source {source!r}")
    print(f"fetch {source}: manifest {manifest}")
    if source == "hrrr":
        sums = args.out / "SHA256SUMS"
        print("fetch hrrr: front door: rw-wps --source hrrr "
              f"--source-root {args.out} --source-manifest {sums} "
              f"--source-manifest-sha256 {sha256_file(sums)} ...")
    elif args.author_front_door_manifest:
        author_gfs_front_door_manifest(
            out=args.out, bridge=args.bridge,
            wps_namelist=args.wps_namelist,
            experiment_config=args.experiment_config,
            static_input=args.static_input,
            static_receipt=args.static_receipt,
            manifest_out=args.manifest_out)
    else:
        print("fetch gfs: next: author the front-door input manifest "
              "with `gpuwm fetch --source gfs "
              f"--author-front-door-manifest --out {args.out} "
              "--bridge GFS_GRIB2_BRIDGE_EXE --wps-namelist NAMELIST_WPS "
              "--experiment-config EXPERIMENT_TOML`")
    return 0


#: Keys an advisory ``[fetch]`` table may carry (mirroring the CLI flags).
#: The table is emitted by ``gpuwm domain`` and validated -- never silently
#: ignored -- by the experiment loaders, which split it off before the
#: strict experiment schema runs.
FETCH_HINT_KEYS = frozenset({
    "source", "cycle", "hours", "area", "point", "radius_km", "out",
    "cadence",
})
_FETCH_HINT_SOURCES = ("gfs", "hrrr", "era5")


def validate_fetch_hints(table: dict, *, source: str) -> None:
    """Fail-loud validation of an advisory ``[fetch]`` hints table.

    ``source`` names the config file for error messages.  The hints are
    documentation for a human (and future fetch sugar); validation keeps
    them from rotting silently: unknown keys, an unknown data source, or
    non-scalar values are hard errors at config load.
    """
    if not isinstance(table, dict):
        raise ValueError(
            f"[fetch] of {source} must be a table of scalar hint keys")
    unknown = sorted(set(table) - FETCH_HINT_KEYS)
    if unknown:
        raise ValueError(
            f"unknown key(s) {unknown} in [fetch] of {source}; known "
            f"keys: {sorted(FETCH_HINT_KEYS)}")
    if "source" not in table:
        raise ValueError(
            f"[fetch] of {source} must carry source = "
            f"{'|'.join(_FETCH_HINT_SOURCES)}")
    if table["source"] not in _FETCH_HINT_SOURCES:
        raise ValueError(
            f"source = {table['source']!r} in [fetch] of {source} is not "
            f"one of {_FETCH_HINT_SOURCES}")
    for key, value in table.items():
        if isinstance(value, bool) or not isinstance(
                value, (str, int, float)):
            raise ValueError(
                f"{key} = {value!r} in [fetch] of {source} must be a "
                "scalar (string or number)")


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "fetch",
        help="download initialization/boundary data (GFS/HRRR), or "
             "template + validate a manual ERA5 CDS retrieval")
    parser.add_argument(
        "--source", required=True, choices=("gfs", "hrrr", "era5"),
        help="public data source")
    parser.add_argument(
        "--cycle", default=None, metavar="YYYY-MM-DDTHH|latest",
        help="model cycle (UTC); 'latest' resolves the newest complete "
             "cycle from the AWS Open Data listing (gfs/hrrr only)")
    parser.add_argument(
        "--hours", type=int, default=None, metavar="N",
        help="forecast window length: hours 0..N are fetched")
    parser.add_argument(
        "--area", default=None, metavar="LAT0,LON0,LAT1,LON1",
        help="bounding box corners in degrees (order free); allow several "
             "degrees of margin beyond the outer domain -- for gfs, "
             f"{GFS_LAKE_DONOR_MARGIN_DEG:g} deg (the front door must "
             "prove every model lake's nearest source-water donor lies "
             "inside the crop; `gpuwm domain` suggests areas with this "
             "margin built in)")
    parser.add_argument(
        "--point", default=None, metavar="LAT,LON",
        help="center point; requires --radius-km")
    parser.add_argument(
        "--radius-km", type=float, default=None, metavar="KM",
        help="half-width of the box around --point")
    parser.add_argument(
        "--out", type=Path, default=None, metavar="DIR",
        help="output directory (created; complete files are skipped on "
             "re-run)")
    parser.add_argument(
        "--cadence", type=int, default=None, choices=(1, 3, 6),
        help="forecast-hour cadence: gfs 1 or 3 (default 3); era5 "
             "template 1, 3, or 6 (default 6); hrrr is hourly")
    parser.add_argument(
        "--validate", type=Path, nargs="+", default=None, metavar="GRIB",
        help="era5 only: validate user-supplied GRIB1 file(s) against "
             "what gpuwm ingest expects instead of fetching")
    parser.add_argument(
        "--transport", default=None, choices=HRRR_TRANSPORTS,
        help="hrrr only: download host.  Both serve byte-identical files "
             "and .idx indexes; 'nomads' (nomads.ncep.noaa.gov, roughly "
             "the newest 48 h) publishes each hour first, 's3' is the "
             "full AWS archive, and 'auto' (default) probes NOMADS for "
             "the requested window and falls back to S3")
    parser.add_argument(
        "--wait-for", action="store_true",
        help="hrrr only: live-cycle mode -- download each forecast hour "
             "as it publishes (polling at most every "
             f"{HRRR_WAIT_POLL_SECONDS} s), so preparation can start "
             "before the cycle finishes publishing; on timeout the "
             "manifest still records the complete fetched prefix and a "
             "re-run resumes")
    parser.add_argument(
        "--wait-timeout-minutes", type=float, default=None, metavar="MIN",
        help="hrrr --wait-for only: give up after this long (default "
             f"{HRRR_WAIT_TIMEOUT_DEFAULT_MINUTES:g} min), reporting "
             "exactly which hours were fetched")
    parser.add_argument(
        "--force-refetch", action="store_true",
        help="move every existing file in --out aside (nothing is "
             "deleted) and re-download; required when re-fetching a "
             "different area/cycle into the same --out")
    front = parser.add_argument_group(
        "GFS front-door manifest authoring",
        "gfs only: write the gpuwm-gfs-direct-input-manifest-v1 document "
        "the rw-wps GFS front door verifies (name + sha256 for every "
        "role, including the bridge executable) and print the "
        "ready-to-run command.  Runs after the download, or standalone "
        "on an already-fetched --out when --cycle/--hours are omitted.")
    front.add_argument(
        "--author-front-door-manifest", action="store_true",
        help="author the front-door input manifest for the fetched "
             "series; requires --bridge, --wps-namelist, and "
             "--experiment-config")
    front.add_argument(
        "--bridge", type=Path, default=None, metavar="EXE",
        help="built gfs_grib2_bridge executable (cargo build --release "
             "--locked --offline in tools/grib1_bridge; see gpuwm "
             "doctor)")
    front.add_argument(
        "--wps-namelist", type=Path, default=None, metavar="WPS",
        help="the namelist.wps the front door will consume (e.g. the "
             "gpuwm domain output)")
    front.add_argument(
        "--experiment-config", type=Path, default=None, metavar="TOML",
        help="the experiment TOML the front door will consume")
    front.add_argument(
        "--static-input", type=Path, default=None, metavar="NPZ",
        help="optional prebuilt static cache (with --static-receipt); "
             "omit when the front door builds statics from --geog-root")
    front.add_argument(
        "--static-receipt", type=Path, default=None, metavar="JSON",
        help="receipt for --static-input")
    front.add_argument(
        "--manifest-out", type=Path, default=None, metavar="JSON",
        help=f"manifest path (default <out>/{GFS_INPUT_MANIFEST_NAME})")
    parser.set_defaults(func=fetch_main)
    return parser


__all__ = [
    "Area", "Era5ValidationReport", "FETCH_HINT_KEYS",
    "FETCH_MANIFEST_SCHEMA", "GFS_FRONT_DOOR_MANIFEST_SCHEMA",
    "GFS_INPUT_MANIFEST_NAME", "GFS_LAKE_DONOR_MARGIN_DEG",
    "HRRR_NOMADS_BASE", "HRRR_NOMADS_RETENTION_HOURS", "HRRR_TRANSPORTS",
    "HRRR_WAIT_POLL_SECONDS", "HRRR_WAIT_TIMEOUT_DEFAULT_MINUTES",
    "validate_fetch_hints",
    "GFS_SUBSET_RECORD_COUNT", "Grib1Record", "area_from_point",
    "author_gfs_front_door_manifest", "check_prior_request",
    "count_grib2_messages", "era5_request_template", "fetch_gfs",
    "fetch_hrrr", "fetch_main", "gfs_forecast_hours", "gfs_object_url",
    "gfs_suggested_fetch_margin_deg",
    "hrrr_forecast_hours", "hrrr_object_url", "parse_area", "parse_cycle",
    "read_grib1_records", "register_cli", "resolve_hrrr_transport",
    "resolve_latest_cycle",
    "sha256_file", "validate_era5_files", "write_era5_request",
    "write_fetch_manifest",
]
