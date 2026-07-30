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
import shlex
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from gpuwm import fetch_bars


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

#: Which downloader moves the bytes.  ``rust`` is the vendored
#: ``rw_fetch`` backbone (16 MiB parallel range GETs, ``.idx``
#: coalescing, the cross-process NOMADS rate governor, a disk cache);
#: ``python`` is the stdlib ``urllib`` transport in :mod:`tools`, which
#: stays as the always-available fallback.  ``auto`` uses the backbone
#: when it is built and the Python transport when it is not.
FETCH_ENGINES = ("auto", "rust", "python")

#: ``--transport`` picks the *host*; ``--mode`` picks the *byte
#: transport*, which is a separate axis: whether to pull the whole
#: object or only the ``.idx``-selected byte ranges out of it.  ``auto``
#: is the probe rule -- object present, and its ``.idx`` absent,
#: malformed, or provably shorter than the object => take the whole
#: file.  No time constants are involved; both named modes are
#: first-class and either can be forced.
FETCH_MODES = ("auto", "full-file", "idx-subset")

#: ``gpuwm fetch --transport`` host names to ``rw_fetch --source``
#: registry names.  Same two hosts, different vocabularies: ArWen has
#: said ``s3`` since before there was a registry, and rustwx calls the
#: same bucket ``aws``.
RW_FETCH_SOURCES = {"nomads": "nomads", "s3": "aws"}

#: ``rw_fetch --model``/``--product`` for each HRRR product ArWen wants.
RW_FETCH_HRRR_PRODUCTS = {"atmosphere": "nat", "soil": "prs"}

#: ``rw_fetch --product`` for the raw GFS object.
RW_FETCH_GFS_PRODUCT = "pgrb2.0p25"

#: ``--wait-for`` polling cadence ceiling (seconds between probe rounds).
HRRR_WAIT_POLL_SECONDS = 30
#: ``--wait-for`` default patience: 90 min covers a live HRRR cycle's
#: full f00..f18 publication spread with margin.
HRRR_WAIT_TIMEOUT_DEFAULT_MINUTES = 90.0

GFS_CYCLE_HOURS = (0, 6, 12, 18)
GFS_MAX_FORECAST_HOUR = 384

#: GDAS is the GFS assimilation cycle's own output, in the *same*
#: pgrb2.0p25 container: same 0.25-degree regular lat/lon grid, same
#: variable and level codes, same 124-record census under the certified
#: selector, same originating centre and table versions.  Verified
#: against a live cycle before this lane was wired -- 124 records, scan
#: 0x40, DRT 5.0, shape 6, centre 7, master table 2, local table 1,
#: PDT 4.0, generating process 81 at f000.  The v1.1 proof corpus adds
#: real f003/f006/f009 subsets with generating process 96; fetch declares
#: the expected process ID per row so the bridge never infers that
#: capability from an hour or a source name.
#:
#: **Fetch and decode, not a front door.**  v1.0.1 scoped this source to
#: f000 because the fail-closed ``gfs_grib2_bridge`` -- which selects by
#: exact field identity and never guesses -- was certified only against
#: the process the analysis carries, and it said that widening the gate
#: would be a re-certification event rather than a flag.  That event has
#: happened: real NOMADS f000/f003/f006/f009 subsets of
#: ``gdas.20260729/12`` are committed under
#: ``tests/fixtures/gdas-process-id/`` -- the last of them the published
#: endpoint itself, so the ceiling rests on bytes and not on this
#: constant -- all 124 messages each frozen at the envelope above, and
#: the bridge now
#: verifies a *declared* process ID against its certified ``{81, 96}``
#: set (each forecast sample is also required to fail under the
#: undeclared analysis-only policy).  So the fetch/decode span is
#: f000..f009 again.
#:
#: What is still not certified is INGEST: the ``gdas`` adapter declares
#: no field/level/cadence mapping and ``rw-wps --source gdas`` refuses.
#: The container is the certified GFS container and the mapping is
#: expected to be reusable wholesale, but that has not been run end to
#: end, so no ``next:`` step here points at it.
GDAS_MAX_FORECAST_HOUR = 9

#: The GDAS forecast-hour ladder this ArWen is certified for: NOMADS
#: publishes the assimilation cycle's short forecast hourly.
GDAS_CERTIFIED_HOURS = tuple(range(GDAS_MAX_FORECAST_HOUR + 1))

#: Sources that ride the certified GFS pgrb2.0p25 container.
GFS_CONTAINER_SOURCES = ("gfs", "gdas")

#: gpuwm source name -> S3 object prefix and directory stem.
GFS_CONTAINER_PREFIX = {"gfs": "gfs", "gdas": "gdas"}
#: One NOMADS-subset GFS pgrb2.0p25 file carries exactly the 124 records
#: the fail-closed ``gfs_grib2_bridge`` selects (21 pressure levels x
#: {GHT, T, RH, U, V} + 11 surface/near-surface + 8 soil-layer records).
#: This is now the **certified tripwire** rather than the bar itself:
#: the bar applied to a download is derived from the live inventory, and
#: a disagreement with this constant is a loud, explicitly acknowledged
#: re-certification event.  See :mod:`gpuwm.fetch_bars`.
GFS_SUBSET_RECORD_COUNT = fetch_bars.CERTIFIED_RECORD_BARS["gfs"]

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

    @property
    def longitude_span_degrees(self) -> float:
        """Eastward longitude width represented by the two stored edges."""

        span = (self.lon_east - self.lon_west) % 360.0
        if span == 0.0 and self.lon_east != self.lon_west:
            return 360.0
        return span

    @property
    def nomads_longitude_amplification(self) -> float | None:
        """Full-band/requested-span ratio when one NOMADS box widens."""

        box = self.as_nomads()
        if (box["left_lon"] == 0.0 and box["right_lon"] == 360.0
                and self.longitude_span_degrees < 360.0):
            return 360.0 / self.longitude_span_degrees
        return None

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

        A box crossing the prime meridian cannot be expressed with one
        ``0 <= left < right <= 360`` request; it widens to the full
        longitude band.  The fetch path must disclose that amplification.
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
    if source in GFS_CONTAINER_SOURCES and cycle.hour not in GFS_CYCLE_HOURS:
        raise ValueError(
            f"{source.upper()} cycles run at 00/06/12/18 UTC only")
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


def gdas_capability_refusal(requested_hour: int) -> str:
    """Why a GDAS request past f009 is refused, and what to do.

    Capability wording on purpose: it names what this ArWen is certified
    to serve, why the boundary is where it is, and the source that does
    cover a full forecast.  It is not a statement about GDAS itself --
    it is a statement about what has been proved here.
    """

    return (
        "GDAS is certified in this ArWen for fetch and decode through "
        f"f{GDAS_MAX_FORECAST_HOUR:03d}: the assimilation cycle's "
        "analysis and its short forecast, and nothing past it.  "
        f"f{requested_hour:03d} was requested.\n"
        "  Why: the certified corpus is real NOMADS f000/f003/f006/f009 "
        "subsets, and the fail-closed gfs_grib2_bridge downstream "
        "selects by exact field identity -- it admits the declared "
        "analysis and forecast generating processes and nothing else.  "
        "Fetching hours the bridge has never seen would just move the "
        "failure later, so the refusal is here.\n"
        "  What to do: stay inside "
        f"--hours 0..{GDAS_MAX_FORECAST_HOUR}, or use --source gfs, "
        f"which is certified through f{GFS_MAX_FORECAST_HOUR}.")


def gdas_forecast_hours(hours: int, cadence: int = 3) -> tuple[int, ...]:
    """The GDAS ladder inside the certified span, or a capability refusal."""

    if hours > GDAS_MAX_FORECAST_HOUR:
        raise ValueError(gdas_capability_refusal(hours))
    if hours == 0:
        return (0,)
    if cadence < 1:
        raise ValueError("--cadence must be a positive number of hours")
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


def gfs_object_url(cycle: datetime, hour: int, source: str = "gfs") -> str:
    """The raw S3 object URL for one ``pgrb2.0p25`` forecast hour.

    Used for availability probes and for reading the live index behind
    the record-count bar -- not for the payload, which comes through the
    NOMADS grib-filter crop (the raw objects are complex-packed; see
    ``tests/fixtures/gfs-scan-order/README.md``).
    """

    prefix = GFS_CONTAINER_PREFIX[source]
    return (f"{GFS_S3_BASE}/{prefix}.{cycle:%Y%m%d}/{cycle:%H}/atmos/"
            f"{prefix}.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}")


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


def cycle_probe_urls(source: str, cycle: datetime,
                     last_hour: int) -> tuple[str, ...]:
    """The objects whose existence proves one cycle covers ``last_hour``."""

    if source in GFS_CONTAINER_SOURCES:
        return (gfs_object_url(cycle, last_hour, source),)
    if source == "hrrr":
        return (hrrr_object_url(cycle, last_hour, "wrfnat"),
                hrrr_object_url(cycle, last_hour, "wrfprs"))
    raise ValueError(
        "cycle completeness probes are only meaningful for gfs/gdas/hrrr")


def require_published_cycle(source: str, cycle: datetime, last_hour: int, *,
                            now: datetime | None = None,
                            probe=_head_ok, progress=print) -> None:
    """Refuse a NAMED cycle the mirrors have not finished publishing.

    ``--cycle 2026-07-30T00`` an hour before that cycle exists used to
    produce twenty lines of ``urllib.error.HTTPError: HTTP Error 404``
    from inside the downloader -- in a product that names the remedy for
    almost everything else.  The probe is the same one ``--cycle latest``
    already runs; the only new thing is running it before we start
    downloading, and saying which cycle IS complete.
    """

    urls = cycle_probe_urls(source, cycle, last_hour)
    if all(probe(url) for url in urls):
        return
    try:
        newest = resolve_latest_cycle(source, last_hour, now=now, probe=probe)
        remedy = (f"the newest complete {source.upper()} cycle covering "
                  f"f{last_hour:03d} is {newest:%Y-%m-%dT%H}Z -- pass that, "
                  "or --cycle latest to resolve it automatically")
    except (RuntimeError, ValueError) as error:
        remedy = f"and no complete cycle could be resolved either ({error})"
    raise RuntimeError(
        f"{source.upper()} cycle {cycle:%Y-%m-%dT%H}Z is not published "
        f"through f{last_hour:03d} yet; {remedy}")


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
    if source in GFS_CONTAINER_SOURCES:
        step = timedelta(hours=6)
        candidate = now.replace(
            minute=0, second=0, microsecond=0,
            hour=max(h for h in GFS_CYCLE_HOURS if h <= now.hour))
        candidates = tuple(candidate - i * step for i in range(9))
        urls = {cycle: cycle_probe_urls(source, cycle, last_hour)
                for cycle in candidates}
    elif source == "hrrr":
        candidate = now.replace(minute=0, second=0, microsecond=0)
        candidates = tuple(candidate - timedelta(hours=i) for i in range(13))
        urls = {cycle: cycle_probe_urls(source, cycle, last_hour)
                for cycle in candidates}
    else:
        raise ValueError(
            "--cycle latest is only meaningful for gfs/gdas/hrrr; ERA5 is "
            "a reanalysis published with a delay of several days")
    if source == "hrrr":
        from gpuwm.hrrr_forecast import hrrr_cycle_horizon
        candidates = tuple(cycle for cycle in candidates
                           if last_hour <= hrrr_cycle_horizon(cycle))
    for cycle in candidates:
        if all(probe(url) for url in urls[cycle]):
            return cycle
    span = "48 h" if source in GFS_CONTAINER_SOURCES else "12 h"
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
    recorded request there is nothing to tie the existing files to (a
    legacy interrupted fetch from before incremental manifests, a
    corrupted manifest, or a directory some other tool wrote), and the
    per-file bars are area-blind, so resuming would bless files this
    request cannot verify.  Only a directory that is absent or empty may
    be fetched into without a manifest.
    """

    prior = _load_fetch_manifest(out)
    if prior is None:
        if out.is_dir() and any(out.iterdir()):
            raise ValueError(
                f"--out {out} is not empty but carries no readable "
                f"{FETCH_MANIFEST_NAME} (a legacy interrupted fetch, a "
                "corrupted manifest, or files another tool put there).  "
                "The existing files cannot be "
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

def gfs_derived_record_bar(cycle: datetime, *, progress=print,
                           opener=None, source: str = "gfs") -> int | None:
    """How many records the GFS selection yields in the live inventory.

    The NOMADS CGI subset has no index of its own -- it *is* the subset
    -- so the live inventory is the ``.idx`` of the corresponding full
    ``pgrb2.0p25`` object on S3, and the bar is the count of records in
    it whose ``(variable, level)`` the CGI query asks for.  Both sides
    come from the same declaration in
    :mod:`tools.download_gfs_native_subset`, so there is no second table
    to drift.

    Returns ``None`` when the index cannot be read; the caller then
    stands the certified constant in and says so.  A transient S3 blip
    must not stop a fetch whose own record count is checked anyway.
    """

    from gpuwm.fetch_bars import count_index_selection, nomads_selector_pairs
    from tools import download_gfs_native_subset as transport

    url = f"{gfs_object_url(cycle, 0, source)}.idx"
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with (opener or urlopen)(request, timeout=120) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, OSError, ValueError) as error:
        progress(f"fetch {source}: could not read the live inventory at "
                 f"{url} ({error}); the certified record bar stands in")
        return None
    return count_index_selection(text, nomads_selector_pairs(
        transport.NOMADS_VARIABLES, transport.NOMADS_LEVELS,
        transport.PRESSURE_LEVELS_HPA))


def fetch_gfs(*, cycle: datetime, hours: tuple[int, ...], area: Area,
              out: Path, progress=print, force: bool = False,
              accept_inventory_change: bool = False,
              derived_bar=gfs_derived_record_bar,
              source: str = "gfs") -> Path:
    """Download the exact GFS pgrb2.0p25 subset series into ``out``.

    Reuses the certified NOMADS query builder and downloader in
    :mod:`tools.download_gfs_native_subset` (single source for the
    124-record selection), adds resumability (a present, envelope-valid
    subset is never re-downloaded), and writes the ``gfs-series.tsv``
    that ``rw-wps --source gfs --gfs-series`` / ``gfs_grib2_bridge``
    consume, plus the fetch manifest.  The series and manifest are
    atomically refreshed after every verified forecast hour, so an
    interrupted fetch records its complete contiguous prefix and the
    same command resumes it.  An existing file must pass the envelope
    walk, the exact 124-record count, AND -- when the prior fetch
    manifest recorded its digest -- that same sha256; a swapped file is
    never re-blessed by the area-blind count alone.  ``force`` moves
    every existing subset aside (never deletes) and re-downloads.
    """

    from gpuwm.fetch_bars import resolve_bar
    from tools import download_gfs_native_subset as transport

    if source not in GFS_CONTAINER_SOURCES:
        raise ValueError(f"fetch_gfs serves {GFS_CONTAINER_SOURCES}, not "
                         f"{source!r}")
    # The capability boundary sits here as well as at the CLI: a caller
    # reaching the library directly must hit the same refusal.
    if source == "gdas":
        beyond = [hour for hour in hours if hour > GDAS_MAX_FORECAST_HOUR]
        if beyond:
            raise ValueError(gdas_capability_refusal(beyond[0]))
    prefix = GFS_CONTAINER_PREFIX[source]
    out.mkdir(parents=True, exist_ok=True)
    # One record bar for the whole request: the selection is
    # instantaneous fields only, so its census does not vary by hour.
    bar = resolve_bar("gfs", derived_bar(cycle, progress=progress,
                                        source=source),
                      accept_inventory_change=accept_inventory_change,
                      progress=progress)
    prior_digests = _prior_manifest_digests(out)
    box = area.as_nomads()
    longitude_amplification = area.nomads_longitude_amplification
    longitude_note = None
    if longitude_amplification is not None:
        requested = (
            f"lat {area.lat_south:g}..{area.lat_north:g}, "
            f"lon {area.lon_west:g}..{area.lon_east:g}")
        fetched = (
            f"lat {box['bottom_lat']:g}..{box['top_lat']:g}, "
            f"lon {box['left_lon']:g}..{box['right_lon']:g}")
        longitude_note = (
            f"requested box {requested} crosses 0 degrees longitude, "
            "which one NOMADS [0,360] subregion cannot express; fetched "
            f"band {fetched}, a {longitude_amplification:g}x "
            "longitude-span amplification (compressed-byte "
            "amplification is data-dependent)")
        progress(f"fetch {source}: NOTE {longitude_note}")
    files: list[dict] = []

    def publish_manifest() -> Path:
        # Relative names: gfs_grib2_bridge resolves them against the
        # TSV's own directory, so the fetched directory stays relocatable.
        series = out / f"{prefix}-series.tsv"
        _atomic_write_text(series, "".join(
            f"{item['forecast_hour']}\t{item['name']}\t"
            f"{81 if item['forecast_hour'] == 0 else 96}\n"
            for item in files))
        entries = files + [{
            "name": series.name, "role": "series", "forecast_hour": None,
            "bytes": series.stat().st_size, "sha256": sha256_file(series),
            "url": None,
        }]
        recorded_hours = tuple(
            int(item["forecast_hour"]) for item in files)
        payload = _manifest_payload(
            source=source, cycle=cycle, hours=recorded_hours,
            area=area, files=entries)
        payload["notes"] = (
            "NOMADS filter subsets (south-to-north 0.25-degree grids); raw "
            "noaa-gfs-bdp-pds S3 objects are north-to-south and are "
            "accepted by gfs_grib2_bridge only after its declared "
            "scan-order flip")
        payload["nomads_area"] = box
        if longitude_note is not None:
            payload["notes"] += f"; {longitude_note}"
            payload["longitude_span_amplification"] = (
                longitude_amplification)
        payload["engine"] = "python"
        payload["mode"] = "nomads-cgi-subset"
        payload["record_bars"] = [bar.as_manifest()]
        return write_fetch_manifest(out, payload)

    def resume_command() -> str:
        cadence = hours[1] - hours[0] if len(hours) > 1 else 3
        area_arg = ",".join(format(value, "g") for value in (
            area.lat_south, area.lon_west,
            area.lat_north, area.lon_east))
        command = [
            "gpuwm", "fetch", "--source", source,
            "--cycle", cycle.strftime("%Y-%m-%dT%H"),
            "--hours", str(hours[-1]), "--cadence", str(cadence),
            "--area", area_arg, "--out", str(out),
        ]
        if accept_inventory_change:
            command.append("--accept-inventory-change")
        return shlex.join(command)

    current: tuple[int, str, Path, str] | None = None
    try:
        for hour in hours:
            name = (f"{prefix}.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}"
                    ".subset.grib2")
            path = out / name
            url = transport.nomads_query(cycle, hour, model=source, **box)
            current = (hour, name, path, url)
            if force and path.exists():
                _quarantine_rejected(path, progress, f"gfs f{hour:03d}")
            if path.exists():
                observed = count_grib2_messages(path)
                if observed != bar.expected:
                    raise ValueError(
                        f"existing {name} carries {observed} GRIB2 "
                        f"messages, expected {bar.expected}; move it "
                        "aside and re-fetch")
                digest = sha256_file(path)
                recorded = prior_digests.get(name)
                if recorded is not None and digest != recorded:
                    raise ValueError(
                        f"existing {name} does not match the sha256 "
                        "recorded in the prior fetch manifest, so it "
                        "cannot be resumed for this request; pass "
                        "--force-refetch to move the existing files aside "
                        "(nothing is deleted) and re-download")
                progress(f"fetch {source} f{hour:03d}: {name} exists, "
                         f"{path.stat().st_size:,} B verified -- skipped")
            else:
                started = time.perf_counter()
                transport._download(url, path)
                observed = count_grib2_messages(path)
                if observed != bar.expected:
                    raise ValueError(
                        f"NOMADS returned {observed} GRIB2 messages for "
                        f"f{hour:03d}, expected {bar.expected}; the "
                        "upstream inventory has drifted")
                digest = sha256_file(path)
                progress(f"fetch {source} f{hour:03d}: {name} "
                         f"{path.stat().st_size:,} B in "
                         f"{time.perf_counter() - started:.1f} s")
            files.append({
                "name": name, "role": f"{source}-subset",
                "forecast_hour": hour, "bytes": path.stat().st_size,
                "sha256": digest, "url": url,
            })
            publish_manifest()
    except KeyboardInterrupt:
        # The downloader atomically promotes .part only after checking the
        # GRIB envelope.  If SIGINT arrived just after that rename but
        # before this loop recorded the hour, apply our own full
        # envelope/count/digest bars before admitting it to the prefix.
        if current is not None:
            hour, name, path, url = current
            if path.is_file() and not any(
                    item["name"] == name for item in files):
                try:
                    observed = count_grib2_messages(path)
                    digest = sha256_file(path)
                    recorded = prior_digests.get(name)
                    if (observed == bar.expected
                            and (recorded is None or digest == recorded)):
                        files.append({
                            "name": name, "role": f"{source}-subset",
                            "forecast_hour": hour,
                            "bytes": path.stat().st_size,
                            "sha256": digest, "url": url,
                        })
                except (OSError, ValueError):
                    pass
        publish_manifest()
        verified = (
            ", ".join(
                f"f{item['forecast_hour']:03d} {item['name']} "
                f"({item['bytes']:,} B, sha256 {item['sha256']})"
                for item in files)
            if files else "none")
        verified_names = {item["name"] for item in files}
        unverified_paths = sorted(
            path for path in out.iterdir()
            if path.is_file()
            and (path.name.endswith(".part")
                 or (path.name.endswith(".grib2")
                     and path.name not in verified_names)))
        unverified = (
            ", ".join(f"{path.name} ({path.stat().st_size:,} B)"
                      for path in unverified_paths)
            if unverified_paths else "none")
        raise RuntimeError(
            f"interrupted. Verified complete GRIB files on disk and "
            f"recorded in {out / FETCH_MANIFEST_NAME}: {verified}. "
            f"Unverified partial/incomplete GRIB files on disk (not "
            f"recorded): {unverified}.\n"
            f"  resume exactly with: {resume_command()}") from None
    return out / FETCH_MANIFEST_NAME


def author_gfs_front_door_manifest(
        *, out: Path, bridge: Path, wps_namelist: Path,
        experiment_config: Path, static_input: Path | None = None,
        static_receipt: Path | None = None,
        manifest_out: Path | None = None,
        source: str = "gfs",
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

    if source not in GFS_CONTAINER_SOURCES:
        raise ValueError(
            f"the GFS front door serves {GFS_CONTAINER_SOURCES}, not "
            f"{source!r}")
    prior = _load_fetch_manifest(out)
    if prior is None or prior.get("source") != source:
        raise ValueError(
            f"{out / FETCH_MANIFEST_NAME} is not a completed "
            f"`gpuwm fetch --source {source}` output; run the fetch first "
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
    prefix = GFS_CONTAINER_PREFIX[source]
    subset_names = {
        item.get("forecast_hour"): item.get("name")
        for item in prior.get("files", ())
        if isinstance(item, dict)
        and item.get("role") == f"{source}-subset"}
    roles: dict[str, Path] = {
        "series": out / f"{prefix}-series.tsv",
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
                f"without a {source}-subset file entry")
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
            # The front door verifies schema, roles and digests, not the
            # model string, so the tag is provenance: which container
            # this series came out of, in one place a receipt can read.
            "model": source.upper(),
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
    # Pasteable exactly as printed: no placeholder a user has to fill
    # in, because every value is already known here.  The geography root
    # is the one gpuwm reads everywhere else (and `gpuwm fetch-geog`
    # stages into by default); the output root is a sibling of the
    # download, so the line runs from any directory and does not write
    # into the inputs.
    from gpuwm.geog_assets import default_geog_root

    def printed(value) -> str:
        """One printed argument, quoted if a shell would split it.

        POSIX display form, as `source_cli._quote_command` renders argv:
        the certified runtime is Linux/CUDA, and forward slashes are
        accepted by every path API on Windows too.  A valid `--out` or
        config path containing a space used to be split the moment this
        command -- whose entire value is that it can be pasted -- was.
        """

        return shlex.quote(str(value).replace("\\", "/"))

    static_args = (
        f" --static-input {printed(static_input)}"
        f" --static-receipt {printed(static_receipt)}"
        if static_input is not None
        else f" --geog-root {printed(default_geog_root())}")
    output_root = out.resolve() / "prepared"
    progress(f"fetch {source}: front-door manifest {path}")
    progress(f"fetch {source}: front-door manifest sha256 {digest}")
    if source != "gfs":
        # A command that ends in a refusal is worse than no command.
        # This container has exactly one certified ingest route and it
        # is named for the product it was certified on; printing it here
        # under another source's series would be a source-identity lie,
        # and printing `--source {source}` would be a dead end.  See
        # GDAS_MAX_FORECAST_HOUR above and docs/public/DATA.md.
        progress(
            f"fetch {source}: no ingest route -- `rw-wps --source "
            f"{source}` refuses, because that adapter declares no "
            "field/level/cadence mapping.  The manifest above is real "
            "and digest-bound; what is missing is the front door, not "
            "the data.  `rw-wps --show-source "
            f"{source}` states the same thing in machine form.")
        return path, digest
    progress(
        "fetch gfs: feed the GFS front door with:\n"
        f"  rw-wps --source gfs --gfs-series {printed(roles['series'])}"
        f" --cycle {cycle:%Y-%m-%d_%H:%M:%S}"
        f" --bridge {printed(bridge)}"
        f" --wps-namelist {printed(wps_namelist)}"
        f" --experiment-config {printed(experiment_config)}"
        f" --source-manifest {printed(path)}"
        f" --source-manifest-sha256 {digest}"
        f"{static_args} --output-root {printed(output_root)}")
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


def _prior_manifest_records(out: Path) -> dict[str, int]:
    """``name -> GRIB2 message count`` from an existing fetch manifest.

    The resume bar has to know what a file was *supposed* to contain,
    and since the probe rule can land either a subset or a whole object
    the certified subset count is no longer that answer on its own.  A
    manifest written before this key existed simply yields nothing here
    and the caller falls back to the certified constant, so old fetch
    directories still resume.
    """

    path = out / FETCH_MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {item["name"]: item["records"]
                for item in payload.get("files", ())
                if isinstance(item.get("name"), str)
                and isinstance(item.get("records"), int)}
    except (ValueError, TypeError, AttributeError):
        return {}


def _prior_manifest_bars(out: Path) -> dict[str, object]:
    """``kind -> RecordBar`` reconstructed from an existing manifest.

    A resume that finds every file already present downloads nothing and
    therefore resolves no bars -- and the manifest it republishes was
    being written from an empty map, so ``record_bars`` came out ``[]``
    and an ``inventory_change_accepted: true`` from the original fetch
    vanished.  DATA promises that acceptance stays recorded, and the
    directory's files really were fetched under that bar, so the prior
    bars seed this run's map and any kind actually re-resolved replaces
    its own entry.

    A malformed or foreign manifest yields nothing, exactly as the digest
    and record maps above do: worst case the provenance is no worse than
    it is today.
    """

    from gpuwm.fetch_bars import RecordBar

    path = out / FETCH_MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        bars: dict[str, object] = {}
        for item in payload.get("record_bars", ()):
            kind = item["kind"]
            bars[kind] = RecordBar(
                kind=kind, expected=int(item["expected"]),
                certified=int(item["certified"]),
                derived=None if item.get("derived") is None
                else int(item["derived"]))
        return bars
    except (ValueError, TypeError, AttributeError, KeyError):
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


def _quarantine_inventory_payload(dest: Path, out: Path, entry: dict,
                                  progress, label: str) -> str:
    """Set aside a payload the record bar is about to refuse.

    Returns the sentence the refusal prints about the disk.  Two things
    have to be true afterwards: no unverified GRIB is left where a
    consumer would read it as a fetch product, and the message says
    where the bytes went -- nothing is ever deleted, so an operator who
    accepts the change keeps the evidence of what arrived.

    The ``.idx`` the backbone kept beside the object goes with it: it is
    the very index whose census disagreed, and it is what a later
    ordinary run would otherwise resume against.
    """

    moved: list[str] = []
    for path in (dest, out / entry["idx_name"] if entry.get("idx_name")
                 else None):
        if path is None or not path.is_file():
            continue
        aside = path.with_name(f"{path.name}.inventory-change-"
                               f"{time.time_ns()}")
        os.replace(path, aside)
        moved.append(aside.name)
    if not moved:
        return "Nothing was left on disk."
    progress(f"fetch {label}: quarantined {', '.join(moved)}")
    return (f"The transfer had already completed, so the payload is on "
            f"disk; it has been moved aside as {', '.join(moved)} in "
            f"{out} and no manifest was written, so nothing downstream "
            f"will read it as a fetch product.  Nothing was deleted.")


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


def resolve_fetch_engine(requested: str, *, progress=print
                         ) -> tuple[str, Path | None]:
    """Resolve ``--engine`` to ``('rust', binary)`` or ``('python', None)``.

    ``rust`` is explicit and fails loudly when the backbone is not
    built; ``auto`` prefers it silently when it is there and falls
    through to the Python transport with one line of explanation when it
    is not.  ``python`` never looks.
    """

    if requested not in FETCH_ENGINES:
        raise ValueError(f"unknown fetch engine {requested!r}; expected one "
                         f"of {FETCH_ENGINES}")
    if requested == "python":
        return "python", None

    from gpuwm import rustwx_fetch

    binary = rustwx_fetch.find_fetch_bin()
    if binary is None:
        if requested == "rust":
            raise ValueError(
                "--engine rust needs the vendored fetch backbone, which is "
                f"not built.\n  {rustwx_fetch.fetch_remedy()}")
        return "python", None
    ok, evidence = rustwx_fetch.probe_fetch_bin(binary)
    if not ok:
        if requested == "rust":
            raise ValueError(f"--engine rust: {binary} -- {evidence}")
        progress(f"fetch: the rust backbone at {binary} is unusable "
                 f"({evidence}); using the Python transport")
        return "python", None
    return "rust", binary


def _rw_fetch_hrrr(*, binary: Path, cycle: datetime, hour: int, kind: str,
                   host: str, mode: str, out: Path,
                   cache_dir: Path | None, progress) -> dict:
    """One HRRR product through the Rust backbone; returns its record.

    The backbone names each object after the URL it came from, so the
    atmosphere lands under the name ArWen already uses and only the soil
    product -- carved out of ``wrfprs`` -- needs renaming afterwards.
    """

    from tools import download_hrrr_native_subset as range_transport
    from gpuwm import rustwx_fetch

    selectors = (range_transport.atmosphere_selectors() if kind == "atmosphere"
                 else range_transport.soil_selectors())
    patterns = out / f".rw-fetch-{kind}-f{hour:02d}.selectors"
    rustwx_fetch.write_pattern_file(patterns, selectors)
    try:
        record = rustwx_fetch.run_fetch(
            binary, model="hrrr", date=f"{cycle:%Y%m%d}", cycle=cycle.hour,
            hours=(hour,), product=RW_FETCH_HRRR_PRODUCTS[kind],
            source=RW_FETCH_SOURCES[host], mode=mode, out=out,
            pattern_file=patterns,
            exclusions=(range_transport.ACCUMULATION_EXCLUSION,),
            cache_dir=cache_dir, keep_idx=True)
    except RuntimeError as error:
        # A selector that matches nothing is this host publishing an
        # inventory we do not recognise, not a network fault; the caller
        # may legitimately try the next host.
        if "matched no index record" in str(error):
            raise range_transport.IndexInventoryError(str(error)) from error
        raise
    finally:
        patterns.unlink(missing_ok=True)
    if len(record["files"]) != 1:
        raise RuntimeError(
            f"rw_fetch returned {len(record['files'])} files for one "
            "forecast hour")
    entry = record["files"][0]
    progress(f"fetch hrrr f{hour:02d} {kind}: {entry['name']} "
             f"{entry['bytes']:,} B in {entry['wall_seconds']:.1f} s "
             f"({entry['source']}, {entry['mode']} -- "
             f"{entry['mode_reason']})")
    return entry


def _expand_selector(selector: str):
    """``A|B:LEVEL`` -> ``[('A', 'LEVEL'), ('B', 'LEVEL')]``.

    The alternation is how one selector covers two provider spellings of
    the same record (``CLMR`` on AWS, ``CLWMR`` on NOMADS).
    """

    variable, _, level = selector.partition(":")
    return [(spelling.strip(), level if _ else None)
            for spelling in variable.split("|")]


def count_selectors_in_index(index_text: str, selectors: tuple[str, ...],
                             exclusion: str | None = None) -> int:
    """How many ``.idx`` records the exact ``VAR:LEVEL`` selectors take.

    Exact on both columns, and skipping any line carrying ``exclusion``
    -- the same rule the Rust backbone applies, so the derived bar and
    the transfer agree by construction.
    """

    wanted = {f"{spelling}:{level}" if level is not None else spelling
              for selector in selectors
              for spelling, level in _expand_selector(selector)}
    matched = 0
    for line in index_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if exclusion is not None and exclusion in stripped:
            continue
        fields = stripped.rstrip(":").split(":")
        if len(fields) < 6:
            continue
        if f"{fields[3]}:{fields[4]}" in wanted:
            matched += 1
    return matched


def _hrrr_derived_bar(entry: dict, *, kind: str, out: Path) -> int | None:
    """The live selection count behind one Rust HRRR transfer.

    In ``idx-subset`` mode the backbone selected the records itself and
    reports how many.  In ``full-file`` mode nothing was selected -- the
    whole object landed -- so the selection census is recomputed from
    the index the backbone kept beside it, but **only** when that index
    was proven to cover the whole object.  A short index is exactly why
    the full file was taken; counting a selection out of it would report
    a deficit that is an artefact of the index, not of the data.  With
    no complete index the bar cannot be derived; the certified constant
    stands in, :func:`gpuwm.fetch_bars.resolve_bar` says so, and the
    fail-closed ``hrrr_grib2_bridge`` remains the completeness gate --
    it selects by exact field identity and refuses a file whose
    inventory it cannot satisfy.
    """

    if entry["mode"] == "idx-subset":
        return entry.get("selected_record_count")
    if entry.get("probe", {}).get("idx_covers_object") is not True:
        return None
    index_name = entry.get("idx_name")
    if not index_name:
        return None
    index_path = out / index_name
    if not index_path.is_file():
        return None
    from tools import download_hrrr_native_subset as range_transport

    selectors = (range_transport.atmosphere_selectors() if kind == "atmosphere"
                 else range_transport.soil_selectors())
    return count_selectors_in_index(
        index_path.read_text(encoding="utf-8"), selectors,
        range_transport.ACCUMULATION_EXCLUSION)


def _download_one_hrrr_product(
        *, engine: str, engine_bin: Path | None, mode: str,
        cycle: datetime, hour: int, kind: str, host: str, url: str,
        dest: Path, dest_name: str, source_name: str, out: Path,
        label: str, cache_dir: Path | None, workers: int, retries: int,
        bar_kind: str, certified: int, accept_inventory_change: bool,
        progress):
    """Download one HRRR product from one host; ``(bar, url)``.

    Raises
    :class:`tools.download_hrrr_native_subset.IndexInventoryError` when
    *this host's* published index does not carry the expected inventory,
    which the caller answers by trying the next host.
    """

    from gpuwm.fetch_bars import resolve_bar
    from tools import download_hrrr_native_subset as range_transport

    if engine == "rust":
        entry = _rw_fetch_hrrr(
            binary=engine_bin, cycle=cycle, hour=hour, kind=kind,
            host=host, mode=mode, out=out, cache_dir=cache_dir,
            progress=progress)
        url = entry["grib_url"]
        landed = out / entry["name"]
        if landed != dest:
            # Only the soil product needs renaming: it is carved out of
            # wrfprs and ArWen files it under its own name.
            os.replace(landed, dest)
        # The census can only be read after the object has landed, so a
        # tripped tripwire here refuses AFTER a payload exists.  Quarantine
        # it first, then let the refusal say what is actually on disk --
        # the alternative is a manifestless directory the next ordinary
        # run also refuses, under a message swearing nothing was
        # downloaded.
        bar = resolve_bar(
            bar_kind, _hrrr_derived_bar(entry, kind=kind, out=out),
            accept_inventory_change=accept_inventory_change,
            progress=progress,
            on_refusal=lambda: _quarantine_inventory_payload(
                dest, out, entry, progress, f"hrrr {label}"))
        if entry["mode"] == "idx-subset":
            expected_messages: int | None = bar.expected
        else:
            # The whole object landed: its census is the index's record
            # count, not the selection's.
            expected_messages = (
                entry.get("probe", {}).get("idx_record_count") or None)
        observed = count_grib2_messages(dest)
        if expected_messages is not None and observed != expected_messages:
            _quarantine_rejected(dest, progress, f"hrrr {label}")
            raise ValueError(
                f"downloaded {dest_name} carries {observed} GRIB2 messages, "
                f"expected {expected_messages} ({entry['mode']} transfer); "
                "the file has been moved aside, nothing was deleted")
        return bar, url

    request = range_transport.ProductRequest(
        url=url,
        index_url=url + ".idx",
        index_path=out / f"{source_name}.idx",
        destination=dest,
        kind=kind,
    )
    # The selection itself is the derivation here: with the count clause
    # left on, an inventory change is refused inside the transport with
    # a message naming exactly what moved; with it accepted, the live
    # selection count becomes the bar.
    range_transport._download_product(
        request, workers=workers, retries=retries,
        expected_count=None if accept_inventory_change else certified)
    observed = count_grib2_messages(dest)
    # An unaccepted change is normally refused inside the transport,
    # before any range GET.  Should one ever reach here the payload has
    # landed, so this refusal quarantines it and tells the truth too.
    bar = resolve_bar(bar_kind, observed,
                      accept_inventory_change=accept_inventory_change,
                      progress=progress,
                      on_refusal=lambda: _quarantine_inventory_payload(
                          dest, out, {"idx_name": f"{source_name}.idx"},
                          progress, f"hrrr {label}"))
    if observed != bar.expected:
        raise ValueError(
            f"downloaded {dest_name} carries {observed} GRIB2 messages, "
            f"expected {bar.expected}; the upstream .idx inventory has "
            "drifted")
    return bar, url


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
               clock=time.monotonic,
               engine: str = "python", engine_bin: Path | None = None,
               mode: str = "auto", cache_dir: Path | None = None,
               accept_inventory_change: bool = False,
               transport_fallback: tuple[str, ...] = ()) -> Path:
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

    ``engine`` selects the downloader: ``'python'`` is the stdlib
    byte-range transport in :mod:`tools.download_hrrr_native_subset`,
    ``'rust'`` the vendored ``rw_fetch`` backbone (parallel range GETs,
    the cross-process NOMADS rate governor, a disk cache).  ``mode`` is
    the byte transport the backbone uses -- under ``'auto'`` a lagging
    or short ``.idx`` lands the **whole** object instead of a subset,
    which is bigger on disk and still exactly what
    ``hrrr_grib2_bridge`` wants, because that bridge selects by field
    identity rather than by file size.

    ``transport_fallback`` lists further hosts to try, in order, when
    the chosen one publishes an index whose inventory this ArWen does
    not recognise.  The two hosts do **not** publish identical index
    vocabularies -- see ``download_hrrr_native_subset.FIELD_ALIASES`` --
    and a host that has genuinely changed something is a reason to move
    on with an explanation rather than to abort a fetch the other host
    can serve.  Network faults remain the transport's own to retry.
    """

    from gpuwm.fetch_bars import resolve_bar
    from tools import download_hrrr_native_subset as range_transport

    if engine not in FETCH_ENGINES or engine == "auto":
        raise ValueError(
            "fetch_hrrr needs a resolved engine ('rust' or 'python'); the "
            "CLI resolves 'auto' first via resolve_fetch_engine")
    if mode not in FETCH_MODES:
        raise ValueError(f"unknown fetch mode {mode!r}; expected one of "
                         f"{FETCH_MODES}")
    if engine == "rust" and engine_bin is None:
        raise ValueError("engine 'rust' needs the resolved rw_fetch binary")
    bar_kinds = {"atmosphere": "hrrr-atmosphere", "soil": "hrrr-soil"}
    expected_counts = {"atmosphere": range_transport.ATMOSPHERE_RECORD_COUNT,
                       "soil": range_transport.SOIL_RECORD_COUNT}
    # Seeded, not empty: a completed resume downloads nothing and would
    # otherwise republish record_bars as [], erasing an accepted
    # inventory change the directory's files really were fetched under.
    bars: dict[str, object] = {}
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
    prior_records = _prior_manifest_records(out)
    bars.update(_prior_manifest_bars(out))
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
        payload["engine"] = engine
        payload["mode"] = mode
        payload["record_bars"] = [
            bar.as_manifest() for bar in bars.values()]
        return write_fetch_manifest(out, payload)

    for hour in hours:
        atmosphere = f"hrrr.t{cycle:%H}z.wrfnatf{hour:02d}.grib2"
        pressure = f"hrrr.t{cycle:%H}z.wrfprsf{hour:02d}.grib2"
        soil = f"hrrr.t{cycle:%H}z.soilf{hour:02d}.grib2"
        for kind, source_name, dest_name, product in (
                ("atmosphere", atmosphere, atmosphere, "wrfnat"),
                ("soil", pressure, soil, "wrfprs")):
            dest = out / dest_name
            # A prior full-file transfer recorded its own census; only
            # fall back to the certified subset count when the manifest
            # predates that key.
            expected = prior_records.get(dest_name, expected_counts[kind])
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
                # Hosts do not publish identical index vocabularies (see
                # download_hrrr_native_subset.FIELD_ALIASES), so a host
                # whose inventory this ArWen does not recognise is a
                # reason to move to the next one with an explanation --
                # not to abort a fetch the other host can serve.
                attempts = (chosen,) + tuple(
                    host for host in transport_fallback if host != chosen)
                started = time.perf_counter()
                for position, host in enumerate(attempts):
                    remaining = attempts[position + 1:]
                    chosen = host
                    url = hrrr_object_url(cycle, hour, product,
                                          transport=host)
                    try:
                        bar, url = _download_one_hrrr_product(
                            engine=engine, engine_bin=engine_bin, mode=mode,
                            cycle=cycle, hour=hour, kind=kind, host=host,
                            url=url, dest=dest, dest_name=dest_name,
                            source_name=source_name, out=out, label=label,
                            cache_dir=cache_dir, workers=workers,
                            retries=retries, bar_kind=bar_kinds[kind],
                            certified=expected_counts[kind],
                            accept_inventory_change=accept_inventory_change,
                            progress=progress)
                    except range_transport.IndexInventoryError as error:
                        if not remaining:
                            raise
                        # The refused host's .idx is already on disk and
                        # the next host's will differ; move it aside so
                        # the byte-identity guard has a clean slate.
                        stale = out / f"{source_name}.idx"
                        if stale.is_file():
                            _quarantine_rejected(
                                stale, progress, f"hrrr {label} index")
                        progress(
                            f"fetch hrrr {label}: {host} publishes an index "
                            f"this ArWen does not recognise ({error}); "
                            f"falling back to {remaining[0]}")
                        continue
                    bars[bar_kinds[kind]] = bar
                    break
                digest = sha256_file(dest)
                progress(f"fetch hrrr {label}: {dest_name} "
                         f"{dest.stat().st_size:,} B in "
                         f"{time.perf_counter() - started:.1f} s "
                         f"({chosen})")
            files.append({
                "name": dest_name, "role": kind, "forecast_hour": hour,
                "bytes": dest.stat().st_size, "sha256": digest,
                "url": url, "transport": chosen,
                "records": count_grib2_messages(dest),
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
                ("--engine", args.engine),
                ("--mode", args.mode),
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
        if source not in GFS_CONTAINER_SOURCES:
            raise ValueError(
                "--author-front-door-manifest applies to --source "
                f"{'/'.join(GFS_CONTAINER_SOURCES)} only (the HRRR front "
                "door consumes the fetched SHA256SUMS directly; its "
                "handoff line is printed after every HRRR fetch)")
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
            manifest_out=args.manifest_out, source=source)
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
    if args.hours < 0:
        raise ValueError("--hours cannot be negative")
    if args.hours < 1 and source != "gdas":
        raise ValueError(
            "--hours must be a positive forecast window; only "
            "--source gdas takes --hours 0, because its f000 is an "
            "analysis and is useful on its own")

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

    if source in GFS_CONTAINER_SOURCES:
        if area is None:
            raise ValueError(
                f"{source} fetch requires --area or --point --radius-km: "
                "the NOMADS subsetter needs a subregion (fetching the "
                "whole globe is never what an experiment needs)")
        if source == "gdas":
            if args.cadence is not None and args.hours == 0:
                raise ValueError(
                    "--hours 0 fetches the f000 analysis alone; --cadence "
                    "does not apply to a single time")
            hours = gdas_forecast_hours(
                args.hours,
                3 if args.cadence is None else args.cadence)
        else:
            cadence = args.cadence if args.cadence is not None else 3
            hours = gfs_forecast_hours(args.hours, cadence)
        if args.cycle == "latest":
            cycle = resolve_latest_cycle(source, hours[-1])
            print(f"fetch {source}: latest complete cycle is "
                  f"{cycle:%Y-%m-%dT%H}Z")
        else:
            cycle = parse_cycle(args.cycle, source)
            require_published_cycle(source, cycle, hours[-1])
        if not args.force_refetch:
            check_prior_request(args.out, source=source, cycle=cycle,
                                area=area)
        manifest = fetch_gfs(
            cycle=cycle, hours=hours, area=area, out=args.out,
            force=args.force_refetch, source=source,
            accept_inventory_change=args.accept_inventory_change)
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
        mode = args.mode if args.mode is not None else "auto"
        # Only an unpinned request may wander between hosts; an operator
        # who named --transport gets that host or an error.
        transport_fallback = (
            tuple(HRRR_TRANSPORTS[1:]) if transport == "auto" else ())
        engine, engine_bin = resolve_fetch_engine(
            args.engine if args.engine is not None else "auto")
        if engine == "python" and mode != "auto":
            raise ValueError(
                f"--mode {mode} needs the rust fetch backbone: the Python "
                "transport only does .idx range subsets.  Build the "
                "backbone (cd tools/rustwx && cargo build --release "
                "--locked --offline) or drop --mode.")
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
            require_published_cycle(
                source, cycle, hrrr_forecast_hours(args.hours, cycle)[-1])
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
        print(f"fetch hrrr: engine {engine}"
              + (f" ({engine_bin})" if engine_bin is not None else "")
              + (f", mode {mode}" if engine == "rust" else ""))
        manifest = fetch_hrrr(
            cycle=cycle, hours=hours, area=area, out=args.out,
            force=args.force_refetch, transport=transport,
            wait=args.wait_for, wait_timeout_s=timeout_minutes * 60.0,
            engine=engine, engine_bin=engine_bin, mode=mode,
            cache_dir=args.cache_dir,
            accept_inventory_change=args.accept_inventory_change,
            transport_fallback=transport_fallback)
    else:
        raise ValueError(f"unknown fetch source {source!r}")
    print(f"fetch {source}: manifest {manifest}")
    if source == "hrrr":
        # The trailing `...` this used to print was on a command line
        # after a "front door:" label, and the consumer refuses it:
        # `gpuwm-wrf-init: error: unrecognized arguments: ...`.  A
        # successful producer must not print a command that fails before
        # it can look at what was just fetched.  So the half this step
        # knows is a bound fragment, and the half it cannot know is a
        # comment naming the flags -- the same shape the 20CRv3
        # authoring step uses.
        sums = args.out / "SHA256SUMS"
        print("fetch hrrr: next: feed the HRRR front door, source "
              "already bound:")
        print(f"  --source-root {args.out} --source-manifest {sums} "
              f"--source-manifest-sha256 {sha256_file(sums)}")
        print("  # fetching cannot bind the run's own flags: "
              "--wps-namelist, --geog-root,\n"
              "  # --experiment-config, --valid-time and --output-root "
              "are yours to supply.\n"
              "  # `rw-wps --show-source hrrr` lists the full argument "
              "contract.")
    elif args.author_front_door_manifest:
        author_gfs_front_door_manifest(
            out=args.out, bridge=args.bridge,
            wps_namelist=args.wps_namelist,
            experiment_config=args.experiment_config,
            static_input=args.static_input,
            static_receipt=args.static_receipt,
            manifest_out=args.manifest_out, source=source)
    elif source == "gdas":
        # No `next:` here on purpose.  Every step past this one ends in
        # `rw-wps --source gdas`, which refuses; a next: that leads to a
        # refusal is a worse experience than an honest full stop.
        print(f"fetch {source}: the files above are verified and "
              "digest-bound, but this ArWen has no GDAS ingest route: "
              "`rw-wps --source gdas` refuses, because that adapter "
              "declares no field/level/cadence mapping.  For a runnable "
              "single-domain front door today use `--source gfs`.  See "
              "`rw-wps --show-source gdas`.")
    else:
        # A template with GFS_GRIB2_BRIDGE_EXE, NAMELIST_WPS and
        # EXPERIMENT_TOML in it was presented as "next" and does not run
        # as printed.  Same shape as the HRRR handoff above: the bound
        # half is a real command, the three values only the user has are
        # named in comments.
        print(f"fetch {source}: next: author the front-door input "
              "manifest.  This half is bound:")
        print(f"  gpuwm fetch --source {source} "
              f"--author-front-door-manifest --out "
              f"{shlex.quote(str(args.out))}")
        print("  # and these three are yours to point at: --bridge (the "
              "built\n"
              "  # gfs_grib2_bridge), --wps-namelist, "
              "--experiment-config.\n"
              "  # `gpuwm doctor` names the bridge path on this machine.")
    return 0


#: Keys an advisory ``[fetch]`` table may carry (mirroring the CLI flags).
#: The table is emitted by ``gpuwm domain`` and validated -- never silently
#: ignored -- by the experiment loaders, which split it off before the
#: strict experiment schema runs.
FETCH_HINT_KEYS = frozenset({
    "source", "cycle", "hours", "area", "point", "radius_km", "out",
    "cadence",
})
_FETCH_HINT_SOURCES = ("gfs", "gdas", "hrrr", "era5")


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
    # A hint table that advertises a fetch this ArWen would refuse is a
    # rotten hint: catch it at config load, in the same words the CLI
    # would use, rather than at the download.
    if table["source"] == "gdas":
        hours = table.get("hours")
        if isinstance(hours, (int, float)) and hours > GDAS_MAX_FORECAST_HOUR:
            raise ValueError(
                f"[fetch] of {source}: {gdas_capability_refusal(int(hours))}")


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "fetch",
        help="download initialization/boundary data (GFS/HRRR), download "
             "and decode only (GDAS -- no ingest route), or template + "
             "validate a manual ERA5 CDS retrieval")
    parser.add_argument(
        "--source", required=True,
        choices=("gfs", "gdas", "hrrr", "era5"),
        help="public data source")
    parser.add_argument(
        "--cycle", default=None, metavar="YYYY-MM-DDTHH|latest",
        help="model cycle (UTC); 'latest' resolves the newest complete "
             "cycle from the AWS Open Data listing (gfs/hrrr only)")
    parser.add_argument(
        "--hours", type=int, default=None, metavar="N",
        help="forecast window length: hours 0..N are fetched.  gdas is "
             "certified for fetch and decode through "
             f"f{GDAS_MAX_FORECAST_HOUR:03d} -- there is no gdas ingest "
             "route, so those files stop at the decoder -- and it is the "
             "one source that also accepts --hours 0, because its f000 "
             "is an analysis")
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
        help="forecast-hour cadence: gfs 1 or 3 (default 3); gdas 1, 3, "
             "or 6 (default 3, and it does not apply to --hours 0, which "
             "is the analysis alone); era5 template 1, 3, or 6 "
             "(default 6); hrrr is hourly")
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
        help="move the existing files this fetch would write aside "
             "(nothing is deleted) and re-download them; unrelated "
             "files, manifests and forecast hours you did not request "
             "stay where they are.  Required when re-fetching a "
             "different area/cycle into the same --out")
    parser.add_argument(
        "--engine", default=None, choices=FETCH_ENGINES,
        help="hrrr only: which downloader moves the bytes.  'rust' is "
             "the vendored rw_fetch backbone (16 MiB parallel range "
             "GETs, .idx coalescing, the cross-process NOMADS rate "
             "governor, a disk cache); 'python' is the stdlib transport "
             "and always works; 'auto' (default) uses the backbone when "
             "it is built")
    parser.add_argument(
        "--mode", default=None, choices=FETCH_MODES,
        help="hrrr --engine rust only: the byte transport, a separate "
             "axis from --transport's choice of host.  'auto' (default) "
             "is the probe rule -- if the object is there and its .idx "
             "is absent, malformed, or provably shorter than the "
             "object, the whole file is taken; there are no time "
             "constants in it.  'full-file' and 'idx-subset' force "
             "either transport, and 'idx-subset' refuses rather than "
             "silently degrading when the index cannot carry it")
    parser.add_argument(
        "--cache-dir", type=Path, default=None, metavar="DIR",
        help="hrrr --engine rust only: wx-core disk cache root, keyed "
             "by URL and byte range, so a re-run or an overlapping "
             "window re-reads bytes instead of re-downloading them")
    parser.add_argument(
        "--accept-inventory-change", action="store_true",
        help="proceed when the live provider inventory yields a "
             "different record count than this ArWen was certified "
             "against.  Without it such a mismatch is a refusal naming "
             "both counts; with it the live count becomes the bar and "
             "the fetch manifest records the acceptance")
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
    "cycle_probe_urls",
    "require_published_cycle",
    "resolve_latest_cycle",
    "sha256_file", "validate_era5_files", "write_era5_request",
    "write_fetch_manifest",
]
