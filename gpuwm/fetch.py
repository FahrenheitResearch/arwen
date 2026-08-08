"""Acquire initialization/boundary data: the ``gpuwm fetch`` front door.

This module is transport only.  It downloads (or, for ERA5, templates and
validates) the exact source inventory the fail-closed Rust GRIB bridges
consume downstream; it never decodes scientific payloads itself.  Every
published file is envelope-verified, sha256-summed, and recorded in a
``fetch-manifest.json`` the preparation step can consume.

Per-source transport:

* ``gfs`` -- two first-class byte transports.  The default is the NOMADS
  ``filter_gfs_0p25.pl`` subsetter (spatial subregion + the exact
  variable/level selection): rate-governed, bandwidth-frugal, re-encoded
  by NOMADS to south-to-north simple packing.  ``--mode full-file``
  takes the whole ``pgrb2.0p25`` objects from the AWS Open Data S3
  archive ``noaa-gfs-bdp-pds`` instead -- whole-globe north-to-south
  (scan 0x00) complex-packed (DRT 5.3) grids, both certified in
  ``gfs_grib2_bridge`` by committed matched pairs (the scan-order flip
  and the SOILW missing-value proof; see
  ``tests/fixtures/gfs-scan-order/README.md``) -- through either the
  Rust backbone's parallel range GETs or the stdlib transport.  ``.idx``
  byte-range subsetting of the raw objects is NOT a certified GFS route.
  S3 is also used for anonymous ``--cycle latest`` resolution (HEAD
  probes, no HTML scraping).
* ``hrrr`` -- NOAA ``.idx`` byte-range subsetting, reusing the proven
  record inventory and range transport in
  :mod:`tools.download_hrrr_native_subset` (native hybrid ``wrfnat``
  atmosphere plus the soil records of ``wrfprs``, the inputs
  ``hrrr_grib2_bridge`` requires), over either of two hosts serving the
  identical production files and indexes: the NOMADS mirror
  (``nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod``, roughly the
  newest 48 h, where each hour publishes first) and the AWS Open Data
  S3 archive ``noaa-hrrr-bdp-pds``.  ``--transport auto`` (default)
  probes S3 for the requested window and uses NOMADS only when S3
  cannot serve it yet: S3 is several times faster for the whole-file
  default, and NOMADS's head start matters only to ``--wait-for``,
  which keeps its own NOMADS-first polling.  NOMADS
  has no grib-filter route for these products -- its HRRR filter
  scripts cover the 2-D ``wrfsfc`` file only -- so subsetting stays
  ``.idx`` byte ranges on both hosts and the exact 561/18 record
  contracts are unchanged.  The bytes move whole by default
  (:data:`HRRR_DEFAULT_MODE`, through the Rust backbone's parallel
  range GETs); ``--mode idx-subset`` is the opt-in bandwidth saver.  ``--wait-for`` polls (at most every 30 s)
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

from gpuwm import explain, fetch_bars, fetch_guard
from gpuwm.explain import layered
from gpuwm.nomads_governor import paced_urlopen


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

#: What ``gpuwm fetch --source hrrr`` does when nobody says otherwise.
#:
#: The whole file, in parallel range GETs.  This was ``auto``, whose
#: probe rule -- take the whole object only when the ``.idx`` cannot
#: carry the selection -- resolves to ``idx-subset`` against every
#: healthy host, because a healthy host publishes a complete index.  So
#: the default was hundreds of small serial range requests: a field
#: report timed one 419 MB HRRR file at **560 s** on a 2 Gbps host,
#: against **27-35 s** for the same class of file taken whole through
#: the Rust backbone.  Roughly a 16x tax, paid by default, to save
#: bandwidth nobody had asked to save.
#:
#: The project ruling this restores is older than the probe rule: full
#: files are the pipeline, and record subsetting is an opt-in bandwidth
#: saver.  ``--mode idx-subset`` still does exactly what it always did
#: and says so in one line when it is chosen; ``--mode auto`` still
#: exists for a caller that genuinely wants the probe to decide.
HRRR_DEFAULT_MODE = "full-file"

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

#: The last GFS 0.25-degree ``pgrb2`` lead published EVERY hour.
#:
#: NCEP publishes that product hourly through f120 and 3-hourly from
#: f120 to f384.  f121, f122 and f124 are not late -- they do not exist
#: and never will, and a HEAD against the archive returns 404 for each
#: while f120 and f123 return 200.  Modelling that break here is what
#: lets a window crossing it be refused for what it is: an availability
#: probe alone reported the permanent gap as "not published yet" and
#: sent a reader off to wait for data no cycle will ever carry.
GFS_HOURLY_MAX_FORECAST_HOUR = 120


def gfs_cadence_break_refusal(start: int, last: int, cadence: int) -> str:
    """Why an hourly GFS window may not cross f120, and what does work."""

    return layered(
        f"--cadence {cadence} reaches f{last:03d}, and the GFS "
        f"0.25-degree pgrb2 product is published every hour only through "
        f"f{GFS_HOURLY_MAX_FORECAST_HOUR:03d}.\n"
        "  What to do: end the window at "
        f"f{GFS_HOURLY_MAX_FORECAST_HOUR:03d} or earlier, or use "
        "--cadence 3, which is published all the way to "
        f"f{GFS_MAX_FORECAST_HOUR}"
        + ("" if start % 3 == 0 else
           f" (--forecast-start-hour {start} is not on the 3 h grid "
           f"either, so a 3 h window would have to begin at "
           f"f{start - start % 3:03d} or f{start + 3 - start % 3:03d})"),
        "  Why: this is NCEP's publication cadence, not a delay.  f121, "
        "f122 and f124 return 404 from the archive today and will still "
        "return 404 tomorrow; f120 and f123 return 200.  Probing for "
        "them would report a permanent structural gap as a cycle that "
        "has not finished uploading.")

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

# HRRR CONUS coverage is NOT a constant here.  It is derived from the
# native Lambert grid definition -- the single source of truth in
# :func:`gpuwm.ingest.hrrr_target.hrrr_coverage_envelope` -- via
# :func:`source_coverage_envelope` below.  The hand-held box this
# replaces (lat 21.1..52.7, lon -134.2..-60.8) was a second definition
# of coverage beside the one `gpuwm domain` sized against, and the two
# disagreed in both directions: its 52.70 cap admitted latitudes north
# of the grid's real 52.6157 top, and refused the wizard's own emitted
# next command for a legal, source-coverable 3 km CONUS root.

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


#: Decimal places of an emitted ``--area`` hint: the wizard's printed
#: command and its [fetch] table both carry this fixed-point form, and
#: :func:`area_bounds_inward` quantizes coverage bounds to the same
#: precision so a formatted hint can never round back out of coverage.
AREA_HINT_DECIMALS = 2


def source_coverage_envelope(source: str
                             ) -> tuple[float, float, float, float] | None:
    """``(south, west, north, east)`` the source's native grid actually
    covers, or ``None`` for a source with no coverage box (global).

    Data, not policy, and ONE definition per source: HRRR's envelope is
    computed from the native Lambert grid itself
    (:func:`gpuwm.ingest.hrrr_target.hrrr_coverage_envelope`), and both
    sides of the area contract -- this module's ``--area`` gate and
    ``gpuwm domain``'s suggested fetch box -- consume it here, so they
    cannot drift apart the way the retired hand-held CONUS box did.
    The import is deferred so a base install stays importable without
    the ingest subpackage loaded.
    """

    if source == "hrrr":
        from gpuwm.ingest.hrrr_target import hrrr_coverage_envelope
        return hrrr_coverage_envelope()
    return None


def area_bounds_inward(envelope: tuple[float, float, float, float],
                       decimals: int = AREA_HINT_DECIMALS
                       ) -> tuple[float, float, float, float]:
    """ENVELOPE rounded INWARD to DECIMALS places.

    The tightest ``(south, west, north, east)`` box that both lies
    inside the envelope and survives fixed-point formatting: a value
    clamped to these bounds and printed at the same precision parses
    back inside the true envelope, whereas clamping to the exact
    envelope and then rounding to the printed form can cross it (e.g.
    52.615653 prints as 52.62, north of the grid).
    """

    south, west, north, east = envelope
    scale = 10.0 ** decimals
    return (math.ceil(south * scale) / scale,
            math.ceil(west * scale) / scale,
            math.floor(north * scale) / scale,
            math.floor(east * scale) / scale)


#: Per-source remedy sentence for a coverage refusal.  Source-specific
#: words live in data, next to the sources this module already names.
_COVERAGE_REMEDY = {
    "hrrr": "use --source gfs for domains outside HRRR's CONUS coverage",
}


def validate_fetch_area(source: str, area: Area) -> None:
    """The per-source ``--area`` coverage gate ``gpuwm fetch`` applies.

    One seam for every front door that wants to prove an area before
    paying for a download (the wizard proves each emitted hint through
    it).  A source with a coverage envelope refuses, fail-closed, any
    box extending beyond what its native grid carries; global sources
    accept every parseable box.  An antimeridian-crossing box cannot
    lie inside a non-crossing envelope, so it is refused too (the old
    corner-order arithmetic waved a Pacific-crossing box through the
    HRRR gate).
    """

    envelope = source_coverage_envelope(source)
    if envelope is None:
        return
    south, west, north, east = envelope
    if (area.crosses_antimeridian
            or area.lat_south < south or area.lat_north > north
            or area.lon_west < west or area.lon_east > east):
        # Printed bounds are quantized INWARD so the remedy box the
        # message names is itself accepted.
        say_s, say_w, say_n, say_e = area_bounds_inward(envelope)
        remedy = _COVERAGE_REMEDY.get(
            source, "choose a source whose coverage includes the request")
        raise ValueError(
            f"requested area extends beyond {source.upper()} coverage: "
            f"the native grid's own lat/lon envelope is "
            f"lat {say_s:.2f}..{say_n:.2f}, lon {say_w:.2f}..{say_e:.2f} "
            f"(derived from the grid definition, not a hand-held box); "
            f"{remedy}")


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


def _forecast_start_hour(start: int | None, cadence: int) -> int:
    """The lead a fetch window begins at: 0, or a checked positive lead.

    A window that starts at f000 is the analysis and its short forecast;
    a window that starts at f{K} is the run whose initial condition is
    GFS's own K-hour forecast.  Both are legitimate; the second is what
    a user wanting the f174..f240 window needs, and fetching f000..f240
    to reach it is the workaround this closes.
    """

    if start is None:
        return 0
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError(
            "--forecast-start-hour must be a nonnegative forecast lead")
    if start % cadence:
        raise ValueError(
            f"--forecast-start-hour {start} is not on the {cadence} h "
            "cadence, so the requested lead is not a time this window "
            "would contain")
    return start


def gfs_forecast_hours(hours: int, cadence: int,
                       start: int | None = None) -> tuple[int, ...]:
    """The f{start}..f{start+NNN} ladder the GFS series contract accepts.

    ``start`` defaults to 0, which is the f000..fNNN ladder every prior
    release fetched, byte for byte.  ``--hours`` stays what it always
    was: the LENGTH of the window, not its final lead, so a window is
    described the same way wherever it begins.
    """

    if cadence not in (1, 3):
        raise ValueError("GFS cadence must be 1 or 3 hours")
    if hours < cadence or hours % cadence:
        raise ValueError(
            f"--hours must be a positive multiple of the {cadence} h cadence")
    start = _forecast_start_hour(start, cadence)
    if start + hours > GFS_MAX_FORECAST_HOUR:
        raise ValueError(
            f"--hours {hours} beginning at f{start:03d} reaches "
            f"f{start + hours:03d}, past the GFS "
            f"f{GFS_MAX_FORECAST_HOUR} horizon")
    if cadence == 1 and start + hours > GFS_HOURLY_MAX_FORECAST_HOUR:
        # Refused HERE, before any probe.  The availability probe cannot
        # tell a permanent publication gap from a cycle still uploading,
        # so it reported f121/f122/f124 as "not published yet" and
        # advised passing an explicit --cycle the caller had already
        # passed.  The cadence break is a property of the product.
        raise ValueError(
            gfs_cadence_break_refusal(start, start + hours, cadence))
    return tuple(range(start, start + hours + 1, cadence))


def gdas_capability_refusal(requested_hour: int) -> str:
    """Why a GDAS request past f009 is refused, and what to do.

    Capability wording on purpose: it names what this ArWen is certified
    to serve, why the boundary is where it is, and the source that does
    cover a full forecast.  It is not a statement about GDAS itself --
    it is a statement about what has been proved here.
    """

    # Already written in the two halves this project now layers
    # everywhere: `What to do` is the action, `Why` is the mechanism.
    # Composing them with `layered` only changes which one prints
    # first and which one waits for the flag.
    return layered(
        "GDAS is certified in this ArWen for fetch and decode through "
        f"f{GDAS_MAX_FORECAST_HOUR:03d}: the assimilation cycle's "
        "analysis and its short forecast, and nothing past it.  "
        f"f{requested_hour:03d} was requested.\n"
        "  What to do: stay inside "
        f"--hours 0..{GDAS_MAX_FORECAST_HOUR}, or use --source gfs, "
        f"which is certified through f{GFS_MAX_FORECAST_HOUR}.",
        "  Why: the certified corpus is real NOMADS f000/f003/f006/f009 "
        "subsets, and the fail-closed gfs_grib2_bridge downstream "
        "selects by exact field identity -- it admits the declared "
        "analysis and forecast generating processes and nothing else.  "
        "Fetching hours the bridge has never seen would just move the "
        "failure later, so the refusal is here.")


#: What NOMADS keeps, roughly, for the 0.25-degree pgrb2 product.
#:
#: Approximate on purpose: it is a rolling window NCEP manages, and this
#: number is only ever used to name the shape of a refusal the transport
#: has already made -- never to predict one.  DATA.md states the same
#: figure.
NOMADS_RETENTION_DAYS = 10


def nomads_reach_refusal(source: str, cycle: datetime, hour: int,
                         error: HTTPError) -> str:
    """Why the NOMADS grib filter would not serve one cycle.

    Two probes disagree in this product, and a user in the gap between
    them met the disagreement as a 42-line ``urllib.error.HTTPError``
    traceback.  Cycle completeness is checked against the AWS S3 archive,
    which holds years; the download is the NOMADS grib-filter crop, which
    holds about :data:`NOMADS_RETENTION_DAYS` days.  Every cycle in
    between passes the check and then dies in the transport -- measured
    on one node: 7 days old fetched, 10 days old 404, 31 days old 403.

    The archive's own answer is the ground truth, so it is translated
    here rather than predicted by a second probe: whatever NCEP's
    retention is today, this is what it just said.
    """

    age = datetime.now(timezone.utc).replace(tzinfo=None) - cycle
    days = age.total_seconds() / 86400.0
    if error.code in (403, 404) and days > 1.0:
        return layered(
            f"{source.upper()} cycle {cycle:%Y-%m-%dT%H}Z is "
            f"{days:.0f} days old and the NOMADS grib filter no longer "
            f"serves it (HTTP {error.code} for f{hour:03d}).\n"
            "  What to do: fetch a cycle inside NOMADS' rolling window "
            f"-- about {NOMADS_RETENTION_DAYS} days -- or use "
            "--source hrrr, whose S3 archive this ArWen reads directly.",
            "  Why: cycle completeness is probed against the AWS S3 "
            "archive, which holds years, while the GFS/GDAS download is "
            "the NOMADS grib-filter crop, which holds days.  A cycle in "
            "between passes the probe and is then refused by the "
            "transport, which is what this is.  Reading the raw S3 "
            "objects instead is not a substitute: they are GRIB2 "
            "template 5.3 (complex packing), and the certified bridge "
            "admits 5.0 only -- see docs/public/DATA.md.")
    return layered(
        f"NOMADS returned HTTP {error.code} for {source.upper()} cycle "
        f"{cycle:%Y-%m-%dT%H}Z f{hour:03d} ({error.reason}).",
        f"  The request URL was {error.url}")


def gdas_forecast_hours(hours: int, cadence: int = 3,
                        start: int | None = None) -> tuple[int, ...]:
    """The GDAS ladder inside the certified span, or a capability refusal."""

    if cadence < 1:
        raise ValueError("--cadence must be a positive number of hours")
    start = _forecast_start_hour(start, cadence)
    if start + hours > GDAS_MAX_FORECAST_HOUR:
        raise ValueError(gdas_capability_refusal(start + hours))
    if hours == 0:
        return (start,)
    return tuple(range(start, start + hours + 1, cadence))


def hrrr_forecast_hours(hours: int, cycle: datetime,
                        start: int | None = None) -> tuple[int, ...]:
    """The contiguous f{start}..f{start+NN} window, checked against the horizon.

    ``start`` defaults to 0, which is the f00..fNN ladder every prior
    release fetched, byte for byte.  ``--hours`` stays what it always
    was -- the LENGTH of the window, not its final lead -- so a window is
    described the same way here as on the GFS and GDAS ladders above.

    HRRR publishes hourly, so the only cadence a lead can be off is one
    hour; ``_forecast_start_hour`` is still what checks the value, so a
    negative or non-integer lead is refused in the same words on every
    source.
    """

    from gpuwm.hrrr_forecast import validate_hrrr_source_forecast_hours

    if hours < 1:
        raise ValueError("--hours must be at least 1 (two hourly frames)")
    start = _forecast_start_hour(start, 1)
    return validate_hrrr_source_forecast_hours(
        range(start, start + hours + 1), cycle=cycle)


# ---------------------------------------------------------------------------
# Latest-cycle resolution (anonymous S3 HEAD probes)
# ---------------------------------------------------------------------------

def _head_ok(url: str) -> bool:
    """Availability probe, governed when it is aimed at NOMADS.

    ``--wait-for`` polls this every 30 s per product per host and
    ``--transport auto`` probes a whole window with it, so it is a real
    request stream and belongs under the same node-wide pacer as the
    payload transfers -- not beside them.  Probes at S3 pass straight
    through, unpaced.
    """

    request = Request(url, method="HEAD",
                      headers={"User-Agent": _USER_AGENT})
    try:
        with paced_urlopen(request, timeout=60) as response:
            return 200 <= response.status < 300
    except (HTTPError, URLError, OSError):
        return False


def gfs_object_url(cycle: datetime, hour: int, source: str = "gfs") -> str:
    """The raw S3 object URL for one ``pgrb2.0p25`` forecast hour.

    Availability probes, the live index behind the record-count bar,
    and -- since the raw complex-packed north-to-south form earned its
    certification (see ``tests/fixtures/gfs-scan-order/README.md``) --
    the payload of ``--mode full-file`` itself.
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

    Both hosts serve byte-identical HRRR files and ``.idx`` indexes, so
    the choice is never about the data.  It is about two things the
    hosts do NOT share: NOMADS publishes each forecast hour before the
    cloud mirrors do and keeps only about the newest
    :data:`HRRR_NOMADS_RETENTION_HOURS`; S3 is the full archive and is
    far faster for whole-file transfers.

    ``auto`` therefore probes **S3** for the requested window's final
    hour pair (both the ``wrfnat`` and ``wrfprs`` objects, mirroring the
    latest-cycle completeness rule) and uses NOMADS only when S3 cannot
    serve that window yet.  It used to be the other way round, and the
    pairing was wrong for the default byte mode: this product's default
    is ``--mode full-file``, and NOMADS paces large transfers hard.
    Measured on one box, one cycle, the same four objects through the
    same backbone: 348/209/418/255 s from NOMADS against 69/34/45/44 s
    from S3, and a three-hour window that took 54 minutes over NOMADS
    took about three over S3.  Nobody chose that; it fell out of
    preferring the freshest host to a caller who was not asking for
    freshness.

    ``--wait-for`` IS asking for freshness, and it does not come through
    here at all -- the live-cycle path keeps ``auto`` unresolved and
    tries NOMADS first on every polling round, which is exactly what
    downloading hours as they publish requires
    (:func:`_fetch_hrrr_locked`).

    One decision per invocation, so a fetch never silently mixes hosts;
    the manifest records every file's actual URL and transport either
    way.  An explicit ``nomads`` that NOMADS cannot serve refuses with
    the retention story rather than failing file by file mid-download.
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
    if requested == "nomads":
        urls = (hrrr_object_url(cycle, last_hour, "wrfnat",
                                transport="nomads"),
                hrrr_object_url(cycle, last_hour, "wrfprs",
                                transport="nomads"))
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
        # Past NOMADS retention there is no second host to probe.
        progress(
            f"fetch hrrr: cycle {cycle:%Y-%m-%dT%H}Z is {age_hours:.0f} h "
            f"old, beyond the ~{HRRR_NOMADS_RETENTION_HOURS} h NOMADS "
            "retention -- using the AWS S3 archive")
        return "s3"
    s3_urls = (hrrr_object_url(cycle, last_hour, "wrfnat", transport="s3"),
               hrrr_object_url(cycle, last_hour, "wrfprs", transport="s3"))
    if all(probe(url) for url in s3_urls):
        progress("fetch hrrr: using the AWS S3 archive (identical files "
                 "and .idx indexes, and far faster for whole-file "
                 "transfers; NOMADS remains the fallback while a cycle "
                 "is still reaching S3)")
        return "s3"
    progress("fetch hrrr: S3 does not serve the full requested window yet "
             "-- using NOMADS, which publishes ahead of the mirrors")
    return "nomads"


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
    """Publish a receipt whole, or leave the previous one alone.

    The staging name used to be a fixed ``<name>.tmp``, which is exactly
    the file two publishers collide on -- one could be renaming the
    other's half-written bytes onto a canonical receipt.  The shared
    helper stages under a per-process, per-call name and fsyncs before
    the rename, so a crash leaves either the old receipt or the new one
    and never a torn or foreign one.
    """

    fetch_guard.atomic_write_text(path, text, tag="fetch")


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
            raise ValueError(layered(
                f"--out {out} is not empty but carries no readable "
                f"{FETCH_MANIFEST_NAME}, so its files are UNVERIFIED for "
                "this request and will not be resumed.\n"
                "  remedy: fetch into a different --out, or pass "
                "--force-refetch to move the existing files aside "
                "(nothing is deleted) and re-download this request.",
                "  why: a missing manifest is a legacy interrupted fetch, "
                "a corrupted manifest, or files another tool put there.  "
                "The existing files cannot be tied to any recorded "
                "source/cycle/area, and the per-file resume check is "
                "area-blind, so resuming onto them would publish a "
                "receipt describing bytes nobody recorded."))
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
        raise ValueError(layered(
            f"--out {out} already holds a fetch for a different request:\n"
            + "\n".join(differences)
            + "\n  remedy: fetch into a different --out, or pass "
            "--force-refetch to move the existing files aside (nothing "
            "is deleted) and re-download this request.",
            "  why: the per-file resume check cannot tell the difference "
            "-- a subset file passes its record-count bar for any area -- "
            "so resuming would silently mix two requests' bytes under one "
            "manifest."))


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

def gfs_live_index(cycle: datetime, *, progress=print, opener=None,
                   source: str = "gfs") -> str | None:
    """The live ``.idx`` behind one GFS/GDAS cycle, or None.

    The NOMADS CGI subset has no index of its own -- it *is* the subset
    -- so the live inventory is the ``.idx`` of the corresponding full
    ``pgrb2.0p25`` object on S3.  Two questions are answered from this
    one document: how many records the selection yields (the record
    bar) and which isobaric levels the product publishes (the ladder a
    requested model top is resolved against).  Reading it once means
    both answers describe the same generation of the same object.

    Returns None when it cannot be read; callers stand the certified
    constants in and say so.  A transient S3 blip must not stop a fetch
    whose own record count is checked anyway.
    """

    url = f"{gfs_object_url(cycle, 0, source)}.idx"
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with (opener or paced_urlopen)(request, timeout=120) as response:
            return response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, OSError, ValueError) as error:
        progress(f"fetch {source}: could not read the live inventory at "
                 f"{url} ({error}); the certified constants stand in")
        return None


def gfs_available_levels(cycle: datetime, *, progress=print, opener=None,
                         source: str = "gfs",
                         index_text: str | None = None
                         ) -> tuple[float, ...]:
    """Which isobaric levels this cycle publishes for every 3-D field.

    The live index first; the captured certified ladder only when it
    cannot be read.  Deciding which levels a requested model top needs
    is a claim about what the product carries, and the product is the
    authority on that -- but a fetch must not fail because S3 hiccuped,
    so the certified fallback is named out loud when it is used.
    """

    from tools import download_gfs_native_subset as transport

    if index_text is None:
        index_text = gfs_live_index(cycle, progress=progress, opener=opener,
                                    source=source)
    if index_text is not None:
        levels = transport.available_levels_from_index(index_text)
        if levels:
            return levels
        progress(f"fetch {source}: the live inventory names no isobaric "
                 "level carrying all of "
                 f"{', '.join(transport.PRESSURE_FIELDS)}; the certified "
                 "ladder stands in")
    return transport.CERTIFIED_AVAILABLE_LEVELS_HPA


def gfs_derived_record_bar(cycle: datetime, *, progress=print,
                           opener=None, source: str = "gfs",
                           levels_hpa: tuple[float, ...] | None = None,
                           index_text: str | None = None) -> int | None:
    """How many records the GFS selection yields in the live inventory.

    The NOMADS CGI subset has no index of its own -- it *is* the subset
    -- so the live inventory is the ``.idx`` of the corresponding full
    ``pgrb2.0p25`` object on S3, and the bar is the count of records in
    it whose ``(variable, level)`` the CGI query asks for.  Both sides
    come from the same declaration in
    :mod:`tools.download_gfs_native_subset`, so there is no second table
    to drift.

    ``levels_hpa`` is the ladder this request actually asks for; the
    count is derived against it, so a run with a deeper model top is
    measured against its own selection rather than the default one.
    ``index_text`` lets the caller hand over an index it has already
    read, so the ladder and the count describe one generation of one
    object rather than two reads that could straddle a publication.

    Returns ``None`` when the index cannot be read; the caller then
    stands the certified constant in and says so.  A transient S3 blip
    must not stop a fetch whose own record count is checked anyway.
    """

    from gpuwm.fetch_bars import count_index_selection, nomads_selector_pairs
    from tools import download_gfs_native_subset as transport

    if index_text is None:
        index_text = gfs_live_index(cycle, progress=progress, opener=opener,
                                    source=source)
    if index_text is None:
        return None
    if levels_hpa is None:
        levels_hpa = transport.PRESSURE_LEVELS_HPA
    return count_index_selection(index_text, nomads_selector_pairs(
        transport.NOMADS_VARIABLES, transport.NOMADS_LEVELS, levels_hpa))


def fetch_gfs(*, cycle: datetime, hours: tuple[int, ...], area: Area,
              out: Path, progress=print, force: bool = False,
              accept_inventory_change: bool = False,
              derived_bar=gfs_derived_record_bar,
              source: str = "gfs",
              top_pressure_pa: float | None = None,
              all_levels: bool = False,
              available_levels=gfs_available_levels) -> Path:
    """Download the exact GFS pgrb2.0p25 subset series into ``out``.

    Single writer per ``--out``: the whole flow -- reading the prior
    receipt, moving files aside under ``force``, transferring, and
    publishing the new receipt -- runs under an exclusive OS lock on the
    output root, so two concurrent fetches cannot interleave into a
    manifest that describes the other one's bytes.  A second run
    announces the wait and then refuses loudly rather than proceed.

    ``top_pressure_pa`` is the model top the fetched atmosphere must
    reach.  Left ``None`` the certified 21-level ladder is fetched
    exactly as before (a 100 hPa / 10000 Pa source top); given a value
    the ladder is extended upward along whatever the live inventory says
    the product publishes, until a level sits at or above it.  A top the
    product genuinely cannot serve refuses, naming the deepest it can.
    ``all_levels`` takes every level the product carries instead.

    See :func:`_fetch_gfs_locked` for the transfer itself.
    """

    with fetch_guard.hold("fetch-out", out, progress=progress):
        return _fetch_gfs_locked(
            cycle=cycle, hours=hours, area=area, out=out, progress=progress,
            force=force, accept_inventory_change=accept_inventory_change,
            derived_bar=derived_bar, source=source,
            top_pressure_pa=top_pressure_pa, all_levels=all_levels,
            available_levels=available_levels)


def _fetch_gfs_locked(*, cycle: datetime, hours: tuple[int, ...], area: Area,
                      out: Path, progress=print, force: bool = False,
                      accept_inventory_change: bool = False,
                      derived_bar=gfs_derived_record_bar,
                      source: str = "gfs",
                      top_pressure_pa: float | None = None,
                      all_levels: bool = False,
                      available_levels=gfs_available_levels) -> Path:
    """The GFS transfer, with the output-root lock already held.

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
    every existing file in ``out`` aside -- receipts first, so an
    interrupted force leaves no manifest claiming replaced bytes -- and
    re-downloads.  Nothing is ever deleted.
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
    if top_pressure_pa is not None and all_levels:
        raise ValueError(
            "--p-top-pa names the model top the ladder must reach and "
            "--all-levels takes every level the product carries; they "
            "are two answers to the same question, so pass one")
    prefix = GFS_CONTAINER_PREFIX[source]
    out.mkdir(parents=True, exist_ok=True)
    # One read of the live index answers both questions below, so the
    # ladder and the record count describe the same generation of the
    # same object rather than two reads that could straddle a
    # publication.
    index_text = gfs_live_index(cycle, progress=progress, source=source)
    available = available_levels(cycle, progress=progress, source=source,
                                 index_text=index_text)
    if all_levels:
        levels = tuple(float(level) for level in available)
        progress(f"fetch {source}: --all-levels takes the whole published "
                 f"ladder, {len(levels)} isobaric levels "
                 f"({min(levels):g}..{max(levels):g} hPa)")
    else:
        levels = transport.levels_for_top(top_pressure_pa,
                                          available=available)
        if top_pressure_pa is not None:
            extra = len(levels) - len(transport.PRESSURE_LEVELS_HPA)
            progress(
                f"fetch {source}: model top {float(top_pressure_pa):g} Pa "
                f"needs {len(levels)} isobaric levels, source top "
                f"{min(levels) * 100.0:g} Pa"
                + (f" ({extra} level(s) above the certified 100 hPa "
                   "ladder)" if extra else " (the certified ladder "
                   "already reaches it)"))
    source_top_pa = min(levels) * 100.0
    # One record bar for the whole request: the selection is
    # instantaneous fields only, so its census does not vary by hour.
    # The certified count is a function of THIS request's ladder --
    # five records per level plus the single-level records -- so a
    # deeper top is not mistaken for an upstream inventory change.
    bar = resolve_bar("gfs", derived_bar(cycle, progress=progress,
                                         source=source, levels_hpa=levels,
                                         index_text=index_text),
                      accept_inventory_change=accept_inventory_change,
                      progress=progress,
                      certified=transport.record_count_for_levels(
                          len(levels)))
    if force:
        # Receipts first, then every other existing file: an interrupted
        # force must never leave a manifest behind that still claims a
        # payload it has already replaced.
        _force_quarantine_output(out, progress, source)
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
            "amplification is data-dependent); informational only -- the "
            "ingest interpolates the domain out of the wider band, so the "
            "only cost is download size and the run continues unchanged")
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
        # The ladder is request state, not a constant, so the receipt
        # carries it: the front-door manifest passes it to the bridge,
        # and the vertical contract needs the source top to decide
        # whether the case's p_top is reachable at all.
        payload["pressure_levels_hpa"] = [
            float(format(float(level), "g")) for level in levels]
        payload["source_top_pressure_pa"] = source_top_pa
        return write_fetch_manifest(out, payload)

    def resume_command() -> str:
        cadence = hours[1] - hours[0] if len(hours) > 1 else 3
        area_arg = ",".join(format(value, "g") for value in (
            area.lat_south, area.lon_west,
            area.lat_north, area.lon_east))
        command = [
            "gpuwm", "fetch", "--source", source,
            "--cycle", cycle.strftime("%Y-%m-%dT%H"),
            # --hours is the window LENGTH, so a window that begins at a
            # forecast lead resumes to the same set it was cut from.
            "--hours", str(hours[-1] - hours[0]),
            "--cadence", str(cadence),
            "--area", area_arg, "--out", str(out),
        ]
        if hours[0]:
            command.extend(("--forecast-start-hour", str(hours[0])))
        if accept_inventory_change:
            command.append("--accept-inventory-change")
        return shlex.join(command)

    current: tuple[int, str, Path, str] | None = None
    try:
        for hour in hours:
            name = (f"{prefix}.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}"
                    ".subset.grib2")
            path = out / name
            url = transport.nomads_query(
                cycle, hour, model=source,
                pressure_levels_hpa=levels, **box)
            current = (hour, name, path, url)
            # No per-hour force quarantine here: the whole directory was
            # swept before the loop, receipts first.  Moving a payload
            # aside mid-loop is what let an old manifest outlive the
            # bytes it claimed.
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
                try:
                    transport._download(url, path)
                except HTTPError as error:
                    raise RuntimeError(nomads_reach_refusal(
                        source, cycle, hour, error)) from None
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


def _gfs_index_record_count(index_url: str, *, progress, label: str
                            ) -> int | None:
    """Message count the live ``.idx`` declares for one whole object.

    The full-file transfer's independent census: every index line names
    one GRIB2 message, so the downloaded object must walk to exactly
    this many envelopes.  ``None`` when the index cannot be read -- the
    envelope walk and the fail-closed bridge remain the completeness
    gates, the same doctrine the HRRR full-file route applies when an
    index cannot vouch for an object.
    """

    request = Request(index_url, headers={"User-Agent": _USER_AGENT})
    try:
        with paced_urlopen(request, timeout=120) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, OSError, ValueError) as error:
        progress(f"fetch {label}: could not read {index_url} ({error}); "
                 "the GRIB2 envelope walk and the fail-closed bridge "
                 "remain the completeness gates")
        return None
    count = sum(1 for line in text.splitlines() if line.strip())
    return count or None


def _rw_fetch_gfs_fullfile(*, binary: Path, cycle: datetime, hour: int,
                           source: str, out: Path,
                           cache_dir: Path | None, progress) -> dict:
    """One whole ``pgrb2.0p25`` object through the Rust backbone.

    No selectors: ``--mode full-file`` takes the object in parallel
    range GETs, and the backbone names it after the URL, which is
    already the name the series records.
    """

    from gpuwm import rustwx_fetch

    record = rustwx_fetch.run_fetch(
        binary, model=source, date=f"{cycle:%Y%m%d}", cycle=cycle.hour,
        hours=(hour,), product=RW_FETCH_GFS_PRODUCT, source="aws",
        mode="full-file", out=out, cache_dir=cache_dir)
    if len(record["files"]) != 1:
        raise RuntimeError(
            f"rw_fetch returned {len(record['files'])} files for one "
            "forecast hour")
    entry = record["files"][0]
    progress(f"fetch {source} f{hour:03d}: {entry['name']} "
             f"{entry['bytes']:,} B in {entry['wall_seconds']:.1f} s "
             f"({entry['source']}, {entry['mode']} -- "
             f"{entry['mode_reason']})")
    return entry


def fetch_gfs_fullfile(*, cycle: datetime, hours: tuple[int, ...],
                       area: Area | None, out: Path, progress=print,
                       force: bool = False, source: str = "gfs",
                       engine: str = "python", engine_bin: Path | None = None,
                       cache_dir: Path | None = None,
                       top_pressure_pa: float | None = None,
                       all_levels: bool = False,
                       available_levels=gfs_available_levels) -> Path:
    """Download whole ``pgrb2.0p25`` objects from the AWS S3 archive.

    The full-file transport for the GFS container sources, holding the
    same output-root lock discipline as :func:`fetch_gfs`.  The raw
    objects are whole-globe north-to-south DRT 5.3 grids; both forms
    are certified in ``gfs_grib2_bridge`` (see
    ``tests/fixtures/gfs-scan-order/README.md``), so no area crop and
    no NOMADS CGI round trip are involved -- ``area``, when given, is
    recorded as request identity only.  The decode ladder is still a
    request property: the manifest records it and the front door
    declares it to the bridge, so a whole-globe object never drags the
    mesosphere into a tornado-scale decode.
    """

    with fetch_guard.hold("fetch-out", out, progress=progress):
        return _fetch_gfs_fullfile_locked(
            cycle=cycle, hours=hours, area=area, out=out, progress=progress,
            force=force, source=source, engine=engine, engine_bin=engine_bin,
            cache_dir=cache_dir, top_pressure_pa=top_pressure_pa,
            all_levels=all_levels, available_levels=available_levels)


def _fetch_gfs_fullfile_locked(*, cycle: datetime, hours: tuple[int, ...],
                               area: Area | None, out: Path, progress,
                               force: bool, source: str, engine: str,
                               engine_bin: Path | None,
                               cache_dir: Path | None,
                               top_pressure_pa: float | None,
                               all_levels: bool,
                               available_levels) -> Path:
    """The whole-object transfer, with the output-root lock held.

    Per object, three independent bars before the manifest admits it:
    the GRIB2 envelope walk (every message declares edition 2 and its
    exact length, and the messages tile the file), the live ``.idx``
    message census when the index can be read, and -- on resume -- the
    sha256 the prior manifest recorded.  The series and manifest are
    refreshed after every verified hour, so an interrupted fetch
    records its complete prefix and the same command resumes it.
    """

    from tools import download_gfs_native_subset as transport

    if source not in GFS_CONTAINER_SOURCES:
        raise ValueError(f"fetch_gfs_fullfile serves {GFS_CONTAINER_SOURCES}, "
                         f"not {source!r}")
    if source == "gdas":
        beyond = [hour for hour in hours if hour > GDAS_MAX_FORECAST_HOUR]
        if beyond:
            raise ValueError(gdas_capability_refusal(beyond[0]))
    if top_pressure_pa is not None and all_levels:
        raise ValueError(
            "--p-top-pa names the model top the ladder must reach and "
            "--all-levels takes every level the product carries; they "
            "are two answers to the same question, so pass one")
    prefix = GFS_CONTAINER_PREFIX[source]
    out.mkdir(parents=True, exist_ok=True)
    # The ladder is a DECODE declaration here, not a transfer selection:
    # the whole object carries every published level either way.  It is
    # resolved exactly as the subset route resolves it, recorded in the
    # manifest, and handed to the bridge by the front door.
    index_text = gfs_live_index(cycle, progress=progress, source=source)
    available = available_levels(cycle, progress=progress, source=source,
                                 index_text=index_text)
    if all_levels:
        levels = tuple(float(level) for level in available)
        progress(f"fetch {source}: --all-levels declares the whole "
                 f"published ladder for the decode, {len(levels)} isobaric "
                 f"levels ({min(levels):g}..{max(levels):g} hPa)")
    else:
        levels = transport.levels_for_top(top_pressure_pa,
                                          available=available)
        if top_pressure_pa is not None:
            extra = len(levels) - len(transport.PRESSURE_LEVELS_HPA)
            progress(
                f"fetch {source}: model top {float(top_pressure_pa):g} Pa "
                f"needs {len(levels)} isobaric levels, source top "
                f"{min(levels) * 100.0:g} Pa"
                + (f" ({extra} level(s) above the certified 100 hPa "
                   "ladder)" if extra else " (the certified ladder "
                   "already reaches it)"))
    source_top_pa = min(levels) * 100.0
    if force:
        _force_quarantine_output(out, progress, source)
    prior_digests = _prior_manifest_digests(out)
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
        payload = _manifest_payload(
            source=source, cycle=cycle,
            hours=tuple(int(item["forecast_hour"]) for item in files),
            area=area, files=entries)
        payload["notes"] = (
            "whole pgrb2.0p25 objects from the AWS S3 archive "
            "(north-to-south DRT 5.3 grids, certified in "
            "gfs_grib2_bridge with its scan-order flip and the SOILW "
            "missing-value matched pair); no area crop is involved")
        payload["engine"] = engine
        payload["mode"] = "full-file"
        payload["transport"] = "s3"
        # The decode ladder this request declares; the front-door
        # manifest passes it to the bridge, and the vertical contract
        # needs the source top to decide whether the case's p_top is
        # reachable at all.
        payload["pressure_levels_hpa"] = [
            float(format(float(level), "g")) for level in levels]
        payload["source_top_pressure_pa"] = source_top_pa
        return write_fetch_manifest(out, payload)

    def resume_command() -> str:
        cadence = hours[1] - hours[0] if len(hours) > 1 else 3
        command = [
            "gpuwm", "fetch", "--source", source,
            "--cycle", cycle.strftime("%Y-%m-%dT%H"),
            "--hours", str(hours[-1] - hours[0]),
            "--cadence", str(cadence),
            "--mode", "full-file", "--out", str(out),
        ]
        if area is not None:
            command.extend(("--area", ",".join(
                format(value, "g") for value in (
                    area.lat_south, area.lon_west,
                    area.lat_north, area.lon_east))))
        if hours[0]:
            command.extend(("--forecast-start-hour", str(hours[0])))
        return shlex.join(command)

    current: tuple[int, str, Path, str] | None = None
    try:
        for hour in hours:
            name = f"{prefix}.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}"
            path = out / name
            url = gfs_object_url(cycle, hour, source)
            current = (hour, name, path, url)
            idx_records = _gfs_index_record_count(
                url + ".idx", progress=progress,
                label=f"{source} f{hour:03d}")
            if path.exists():
                observed = count_grib2_messages(path)
                if idx_records is not None and observed != idx_records:
                    raise ValueError(
                        f"existing {name} carries {observed} GRIB2 "
                        f"messages where the live index lists "
                        f"{idx_records}; move it aside and re-fetch")
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
                if engine == "rust":
                    entry = _rw_fetch_gfs_fullfile(
                        binary=engine_bin, cycle=cycle, hour=hour,
                        source=source, out=out, cache_dir=cache_dir,
                        progress=progress)
                    landed = out / entry["name"]
                    if landed != path:
                        raise RuntimeError(
                            f"rw_fetch landed {entry['name']}, expected "
                            f"{name}")
                else:
                    try:
                        transport._download(url, path)
                    except HTTPError as error:
                        raise RuntimeError(
                            f"the S3 archive did not serve {url} "
                            f"({error.code} {error.reason}); "
                            "`gpuwm fetch --cycle latest` probes the same "
                            "archive, and a permanent 404 here usually "
                            "names a lead this cycle never published"
                        ) from None
                observed = count_grib2_messages(path)
                if idx_records is not None and observed != idx_records:
                    _quarantine_rejected(path, progress,
                                         f"{source} full-file")
                    raise ValueError(
                        f"downloaded {name} carries {observed} GRIB2 "
                        f"messages where its live .idx lists "
                        f"{idx_records}; the file has been moved aside, "
                        "nothing was deleted")
                digest = sha256_file(path)
                if engine != "rust":
                    progress(f"fetch {source} f{hour:03d}: {name} "
                             f"{path.stat().st_size:,} B in "
                             f"{time.perf_counter() - started:.1f} s "
                             "(s3, full-file)")
            files.append({
                "name": name, "role": f"{source}-full-file",
                "forecast_hour": hour, "bytes": path.stat().st_size,
                "sha256": digest, "url": url,
                "grib2_messages": observed, "idx_records": idx_records,
            })
            publish_manifest()
    except KeyboardInterrupt:
        # Same admission bars as the loop: envelope walk, index census
        # when known, and the prior manifest's digest.  A partial file
        # never reaches the manifest.
        if current is not None:
            hour, name, path, url = current
            if path.is_file() and not any(
                    item["name"] == name for item in files):
                try:
                    observed = count_grib2_messages(path)
                    digest = sha256_file(path)
                    recorded = prior_digests.get(name)
                    if recorded is None or digest == recorded:
                        files.append({
                            "name": name, "role": f"{source}-full-file",
                            "forecast_hour": hour,
                            "bytes": path.stat().st_size,
                            "sha256": digest, "url": url,
                            "grib2_messages": observed,
                            "idx_records": None,
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
                 or (".pgrb2." in path.name
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
        forecast_start_hour: int | None = None,
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
    # Author over a TAIL of what was fetched, when asked.  A directory
    # already holding f000..f240 does not have to be re-downloaded for a
    # run that starts at f174: the manifest and its series are cut to
    # f174..f240, so the front door decodes only that window.  The cut is
    # by absolute lead, because that is what the fetch recorded.
    series_name = f"{prefix}-series.tsv"
    if forecast_start_hour is not None:
        if (isinstance(forecast_start_hour, bool)
                or not isinstance(forecast_start_hour, int)
                or forecast_start_hour < 0):
            raise ValueError(
                "--forecast-start-hour must be a nonnegative forecast lead")
        if forecast_start_hour not in hours:
            raise ValueError(
                f"--forecast-start-hour {forecast_start_hour} is not a "
                f"forecast hour {out} carries.  It holds "
                + ", ".join(f"f{hour:03d}" for hour in hours))
        hours = [hour for hour in hours if hour >= forecast_start_hour]
        if len(hours) < 2:
            raise ValueError(
                f"--forecast-start-hour {forecast_start_hour} leaves "
                f"{len(hours)} forecast hour(s) in {out}; a run needs its "
                "initial condition and at least one lateral boundary time")
        series_name = f"{prefix}-series-f{forecast_start_hour:03d}.tsv"
    # Either transport's payload rows: the NOMADS CGI crop and the
    # whole-object S3 route feed the same front door and the same
    # bridge, which selects by exact field identity either way.
    payload_roles = {f"{source}-subset", f"{source}-full-file"}
    subset_names = {
        item.get("forecast_hour"): item.get("name")
        for item in prior.get("files", ())
        if isinstance(item, dict)
        and item.get("role") in payload_roles}
    if forecast_start_hour:
        # A real, hash-bound series over the tail, written beside the
        # fetch's own.  The full series is never edited: both remain
        # readable, and the manifest names exactly one of them.
        _atomic_write_text(out / series_name, "".join(
            f"{hour}\t{subset_names[hour]}\t{81 if hour == 0 else 96}\n"
            for hour in hours
            if isinstance(subset_names.get(hour), str)))
    roles: dict[str, Path] = {
        "series": out / series_name,
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
                f"without a {source}-subset or {source}-full-file entry")
        roles[f"grib-f{hour:03d}"] = out / name
    missing = sorted(
        f"{role}: {path}" for role, path in roles.items()
        if not path.is_file())
    if missing:
        raise ValueError(
            "front-door manifest inputs are missing:\n  "
            + "\n  ".join(missing))
    identity = {
        # The front door verifies schema, roles and digests, not the
        # model string, so the tag is provenance: which container
        # this series came out of, in one place a receipt can read.
        "model": source.upper(),
        "product": "pgrb2.0p25",
        "cycle": cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # The pressure ladder the fetch actually took, and the source top it
    # implies.  The front door validates the case's p_top against the
    # source top BEFORE the bridge runs, so it has to learn it from the
    # receipt rather than from a constant -- a constant is exactly what
    # capped every GFS run at 10000 Pa.  Absent for a directory fetched
    # by an older ArWen, where the certified 100 hPa ladder is the only
    # thing it can have been.
    levels = prior.get("pressure_levels_hpa")
    if isinstance(levels, list) and levels and all(
            isinstance(level, (int, float)) for level in levels):
        identity["pressure_levels_hpa"] = [float(level) for level in levels]
        identity["top_pressure_pa"] = float(min(levels)) * 100.0
    payload = {
        "schema": GFS_FRONT_DOOR_MANIFEST_SCHEMA,
        "source": identity,
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

    # --physics-profile, when this config HAS one the front door names.
    #
    # It is spelled "optional" in rw-wps' own help and is not: absent,
    # `gpuwm.source_cli` substitutes WSM6_PROFILE_ID and then compares
    # the experiment's physics against that, so a pasted command with no
    # profile refuses every config except a wsm6-no-radiation one.  The
    # command this function prints is the one FIRST-LIGHT.md section 3a
    # tells a reader to paste, and its own worked example uses the
    # Morrison profile -- so the documented chain failed at this step
    # for exactly the case it documents.  Found by running it.
    #
    # Derived through the same authority the front door asks
    # (`identify_single_domain_profile`), so this cannot name a profile
    # the runner would then reject.  A config matching no shipped
    # profile adds nothing: the front door prints its own explanation of
    # that case after preparation, and inventing a flag here would only
    # move the refusal earlier without making it truer.
    profile_arg = ""
    corridor_arg = ""
    try:
        from gpuwm.experiment import load_experiment
        from gpuwm.physics_compat import identify_single_domain_profile
        from gpuwm.static.corridor import config_declares_follow_source

        experiment = load_experiment(Path(experiment_config))
        matched = identify_single_domain_profile(experiment.root.run)
        if matched is not None:
            profile_arg = f" --physics-profile {matched}"
        # A config that declares a [relocation] follow source needs the
        # sealed statics corridor prepared, or the tree runner refuses
        # the very bundle this command builds.  The predicate itself
        # lives in the corridor module, so the pasted line, `gpuwm go`'s
        # driven line and run-plan's refusal cannot drift apart on it.
        if config_declares_follow_source(experiment):
            corridor_arg = " --statics-corridor"
    except Exception:
        # A config this process cannot load is not a reason to withhold
        # the rest of a correct command.
        profile_arg = ""
        corridor_arg = ""
    output_root = out.resolve() / "prepared"
    progress(f"fetch {source}: front-door manifest {path}")
    progress(f"fetch {source}: front-door manifest sha256 {digest}")
    if hours[0]:
        lead_start = cycle + timedelta(hours=hours[0])
        progress(
            f"fetch {source}: this manifest binds forecast hours "
            f"f{hours[0]:03d}..f{hours[-1]:03d}.  An experiment whose "
            f"start_time is {lead_start:%Y-%m-%d %H:%M:%S} is initialized "
            f"from f{hours[0]:03d} -- a {hours[0]} h forecast, not an "
            "analysis -- with its lateral boundaries from the hours "
            "after it")
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
        f"{profile_arg}{corridor_arg}{static_args} "
        f"--output-root {printed(output_root)}")
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


def _quarantine_rejected(dest: Path, progress, label: str) -> Path:
    """Move a failed existing file aside (never deleted, never reused).

    The name is proven free first.  ``.rejected-<time_ns>`` alone is
    not: two quarantines inside one clock tick -- or two processes --
    produce the same name, and ``os.replace`` onto it silently destroys
    the earlier evidence, which is the one thing quarantine promises not
    to do.
    """

    aside = fetch_guard.quarantine(dest)
    progress(f"fetch {label}: moved rejected file aside to "
             f"{aside.name}")
    return aside


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
        aside = fetch_guard.quarantine(path, tag="inventory-change")
        moved.append(aside.name)
    if not moved:
        return "Nothing was left on disk."
    progress(f"fetch {label}: quarantined {', '.join(moved)}")
    return (f"The transfer had already completed, so the payload is on "
            f"disk; it has been moved aside as {', '.join(moved)} in "
            f"{out} and no manifest was written, so nothing downstream "
            f"will read it as a fetch product.  Nothing was deleted.")


#: Suffix fragments that mark a file as *already* set aside.  A force
#: sweep leaves these alone: re-quarantining evidence only renames it,
#: and the audited property that quarantine artifacts are never treated
#: as canonical and never recursively quarantined is worth keeping.
_QUARANTINE_MARKS = (".rejected-", ".inventory-change-")


def _force_quarantine_output(out: Path, progress, label: str) -> list[str]:
    """``--force-refetch``: move every existing file in ``out`` aside.

    Two properties, and the order between them is the whole point.

    *The receipt goes first.*  Force used to move each requested payload
    aside one at a time, as its turn came round in the fetch loop, while
    the previous ``fetch-manifest.json`` stayed canonical -- so a kill
    (or a network failure) part-way through left a readable manifest
    claiming a digest for bytes that had just been renamed away.  That
    directory lies until something re-reads it.  Quarantining the
    manifest and the checksum/series receipts *before* touching a single
    payload means an interrupted force leaves a directory with payloads
    and no receipt, which the front door already refuses honestly.

    *Every file, as advertised.*  The CLI has always said force moves
    every existing file in ``--out`` aside; it actually moved only the
    payload paths the new request selected.  Forecast hours outside a
    shortened window, ``.idx`` indexes, selector files, stale ``.part``
    files and unrelated files all stayed canonical -- old payloads that
    the new manifest does not list, and sidecars that can block the
    recovery force was invoked to perform.  Now the sweep matches the
    sentence.

    Nothing is deleted and nothing already set aside is touched again.
    Subdirectories are left alone: fetch writes no directories into
    ``--out``, so anything that is one belongs to the operator.
    """

    if not out.is_dir():
        return []
    receipts = (FETCH_MANIFEST_NAME, "SHA256SUMS")
    def order(path: Path) -> tuple[int, str]:
        if path.name in receipts:
            return 0, path.name
        if path.name.endswith("-series.tsv"):
            return 1, path.name
        return 2, path.name

    candidates = sorted(
        (path for path in out.iterdir()
         if path.is_file()
         and not any(mark in path.name for mark in _QUARANTINE_MARKS)),
        key=order)
    moved: list[str] = []
    for path in candidates:
        aside = fetch_guard.quarantine(path)
        moved.append(aside.name)
    if moved:
        progress(f"fetch {label}: --force-refetch moved {len(moved)} "
                 f"existing file(s) in {out} aside (receipts first, so no "
                 "manifest survives claiming replaced bytes); nothing was "
                 "deleted: " + ", ".join(moved))
    return moved


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
        # The backbone names its output after the URL, so the soil
        # product lands as `wrfprs` and Python renames it to `soil`
        # afterwards.  A kill in that gap leaves a canonical orphan
        # under the source name: the next Rust run refuses because its
        # own destination already exists, and force never selected that
        # name, so the directory was unrecoverable without hand
        # intervention.  Set the orphan aside first -- it is unclaimed
        # by any receipt and nothing is deleted.
        landing = out / source_name
        if landing != dest and landing.exists():
            _quarantine_rejected(
                landing, progress,
                f"hrrr {label} orphaned {source_name}")
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

    Single writer per ``--out``: the prior-receipt read, the ``force``
    sweep, the transfers and every receipt publication run under an
    exclusive OS lock on the output root, so two concurrent HRRR fetches
    cannot publish receipts describing each other's bytes.

    See :func:`_fetch_hrrr_locked` for the transfer itself.
    """

    with fetch_guard.hold("fetch-out", out, progress=progress):
        return _fetch_hrrr_locked(
            cycle=cycle, hours=hours, area=area, out=out, workers=workers,
            retries=retries, progress=progress, force=force,
            transport=transport, wait=wait, wait_timeout_s=wait_timeout_s,
            probe=probe, sleeper=sleeper, clock=clock, engine=engine,
            engine_bin=engine_bin, mode=mode, cache_dir=cache_dir,
            accept_inventory_change=accept_inventory_change,
            transport_fallback=transport_fallback)


def _fetch_hrrr_locked(*, cycle: datetime, hours: tuple[int, ...],
                       area: Area | None, out: Path, workers: int = 8,
                       retries: int = 5, progress=print,
                       force: bool = False, transport: str = "s3",
                       wait: bool = False,
                       wait_timeout_s: float = (
                           HRRR_WAIT_TIMEOUT_DEFAULT_MINUTES * 60),
                       probe=None, sleeper=time.sleep,
                       clock=time.monotonic,
                       engine: str = "python",
                       engine_bin: Path | None = None,
                       mode: str = "auto", cache_dir: Path | None = None,
                       accept_inventory_change: bool = False,
                       transport_fallback: tuple[str, ...] = ()) -> Path:
    """The HRRR transfer, with the output-root lock already held.

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

    The manifest is republished after every **completed hour**, and it
    claims only the files of the hours it declares complete.  Both
    halves matter: without the first, an ordinary kill after many good
    hours leaves a fresh output with valid payloads and no receipt at
    all; without the second, the timeout path publishes the
    half-fetched hour's atmosphere in ``files`` and ``SHA256SUMS``
    while ``forecast_hours`` names only the earlier prefix -- one
    receipt with two definitions of complete.  A half-fetched hour
    stays on disk unclaimed and is re-verified under the ordinary bars
    on the next run.

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
        validate_fetch_area("hrrr", area)
    out.mkdir(parents=True, exist_ok=True)
    if force:
        # Receipts first, then every other existing file -- including the
        # `.idx` indexes, whose byte-identity guard could otherwise block
        # the very host failover force was invoked to unblock.
        _force_quarantine_output(out, progress, "hrrr")
    prior_digests = _prior_manifest_digests(out)
    prior_records = _prior_manifest_records(out)
    bars.update(_prior_manifest_bars(out))
    deadline = clock() + wait_timeout_s
    files: list[dict] = []
    complete_hours: list[int] = []

    def publish_manifest(recorded_hours: tuple[int, ...]) -> Path:
        # A receipt claims only files belonging to a COMPLETE hour.  An
        # hour is complete when both its products landed and verified;
        # `files` grows a product at a time, so publishing it whole
        # against an earlier `forecast_hours` prefix -- which is exactly
        # what the wait-timeout branch did -- produced one receipt with
        # two contradictory definitions of completeness: `forecast_hours`
        # said [0] while `files` and `SHA256SUMS` carried the half-done
        # hour 1.  A half-fetched hour stays on disk, unclaimed, and the
        # next run re-verifies it under the ordinary bars.
        wanted = set(recorded_hours)
        claimed = [item for item in files
                   if item["forecast_hour"] in wanted]
        sums = out / "SHA256SUMS"
        _atomic_write_text(sums, "".join(
            f"{item['sha256']}  {item['name']}\n"
            for item in sorted(claimed, key=lambda item: item["name"])))
        entries = claimed + [{
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
        # Checkpoint per completed hour, as GFS already did.  Publishing
        # only after the whole nested loop meant an ordinary SIGKILL
        # after many good hours left a fresh output with valid payloads
        # and no receipt at all, which the front door then refused --
        # verified data, unusable, and nothing to resume from.
        publish_manifest(tuple(complete_hours))
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


#: Where cdsapi reads the personal CDS key, exactly as
#: :data:`ERA5_INSTRUCTIONS` step 1 tells the reader to write it.  Named
#: once so the instruction and the check cannot come to disagree about
#: which file they are talking about.
CDSAPIRC_NAME = ".cdsapirc"


def cds_credentials_path() -> Path:
    """The ``~/.cdsapirc`` cdsapi would read on this machine."""

    return Path.home() / CDSAPIRC_NAME


def cds_credentials_present() -> bool:
    """Is a CDS key file in place?  Existence only -- never read.

    The ERA5 route is the one front door whose first step happens
    outside gpuwm entirely, and the failure mode it produces is a
    cdsapi exception several commands later with nothing pointing back
    at the missing file.  Answering "is it there" costs a ``stat`` and
    lets the wizard say so while the reader is still deciding what to
    run next.

    Presence, not validity: a key's correctness is the CDS server's
    verdict to give, and guessing at its format here would produce
    confident wrong advice about a file this project does not own.
    """

    try:
        return cds_credentials_path().is_file()
    except OSError:
        return False


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
                f"{', '.join(hrrr_only)}: --source hrrr only (the GFS "
                "container sources have one full-file host, the S3 "
                "archive; ERA5 is a manual CDS retrieval)")
    if source not in ("hrrr",) + GFS_CONTAINER_SOURCES:
        transported = sorted(
            flag for flag, value in (
                ("--engine", args.engine),
                ("--mode", args.mode),
                ("--cache-dir", args.cache_dir),
            ) if value is not None)
        if transported:
            raise ValueError(
                f"{', '.join(transported)}: --source hrrr or "
                f"{'/'.join(GFS_CONTAINER_SOURCES)} only (ERA5 is a "
                "manual CDS retrieval)")
    # The GFS container sources have exactly two byte transports, and
    # --mode is how the second is chosen: the NOMADS grib-filter crop
    # (the default; spatial subregion + exact record selection) and
    # --mode full-file (the whole pgrb2.0p25 objects from the S3
    # archive, the same first-class whole-file doctrine as HRRR).
    # .idx record subsetting of the raw objects is not a certified GFS
    # route, and 'auto' has nothing to probe -- the two transports
    # differ in kind, not in health.
    gfs_fullfile = False
    if source in GFS_CONTAINER_SOURCES:
        if args.mode in ("auto", "idx-subset"):
            raise ValueError(
                f"--mode {args.mode}: --source {source} has two byte "
                "transports -- the NOMADS grib-filter crop (the default, "
                "no --mode needed) and '--mode full-file' (whole "
                "pgrb2.0p25 objects from the S3 archive).  .idx record "
                "subsetting of the raw objects is not a certified GFS "
                "route, and 'auto' has nothing to probe: the two "
                "transports differ in kind, not in health.")
        gfs_fullfile = args.mode == "full-file"
        if not gfs_fullfile:
            cgi_extras = sorted(
                flag for flag, value in (
                    ("--engine", args.engine),
                    ("--cache-dir", args.cache_dir),
                ) if value is not None)
            if cgi_extras:
                raise ValueError(
                    f"{', '.join(cgi_extras)}: these choose how whole "
                    "objects move and belong to '--mode full-file'; the "
                    "default NOMADS grib-filter crop has exactly one "
                    "transport (governed stdlib HTTP)")
    if source not in GFS_CONTAINER_SOURCES:
        gfs_only = sorted(
            flag for flag, value in (
                ("--p-top-pa", args.p_top_pa),
                ("--all-levels", args.all_levels or None),
            ) if value is not None)
        if gfs_only:
            raise ValueError(
                f"{', '.join(gfs_only)}: --source "
                f"{'/'.join(GFS_CONTAINER_SOURCES)} only.  HRRR is fetched "
                "on its native hybrid levels (no isobaric ladder to "
                "choose), and the ERA5 request template carries its own "
                "level list.")
    # A separate refusal, with its own reason.  --forecast-start-hour used
    # to ride the level-ladder bundle above, so `--source hrrr
    # --forecast-start-hour 6` was declined for having asked about an
    # isobaric ladder -- a sentence about a flag the user had not typed,
    # for a source whose whole decode path is already lead-aware.  What
    # the flag actually needs is a source with forecast leads in it.
    if args.forecast_start_hour is not None and source == "era5":
        raise ValueError(
            "--forecast-start-hour: forecast sources only (gfs/gdas/hrrr).  "
            "ERA5 is a reanalysis -- every time in it is an analysis, so "
            "there is no forecast lead for a window to begin at.  Name the "
            "analysis time you want with --cycle instead.")
    if args.p_top_pa is not None and args.p_top_pa <= 0:
        raise ValueError("--p-top-pa must be a positive pressure in Pa")

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
        required = ("--wps-namelist", "--experiment-config")
        absent = [flag for flag in required if author_roles[flag] is None]
        if absent or args.out is None:
            raise ValueError(
                "--author-front-door-manifest requires --out plus "
                + ", ".join(required)
                + " (the front-door manifest binds each file's sha256, "
                "including the bridge executable's)")
        if args.bridge is None:
            # `gpuwm go` has always resolved this through
            # gpuwm.bridges; the stage-by-stage route demanded a path
            # instead, and the one FIRST-LIGHT printed
            # (tools/grib1_bridge/target/release/...) exists only in a
            # checkout -- so a wheel user following the documented long
            # form met "front-door manifest inputs are missing: bridge"
            # after paying for the fetch.  Same resolver, same answer,
            # whichever door they came through.  Resolved after the
            # flags above so a usage mistake still reads as one.
            args.bridge = author_roles["--bridge"] = _resolve_manifest_bridge(
                source)
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
            manifest_out=args.manifest_out, source=source,
            forecast_start_hour=args.forecast_start_hour)
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
        if area is None and not gfs_fullfile:
            raise ValueError(
                f"{source} fetch requires --area or --point --radius-km: "
                "the NOMADS subsetter needs a subregion (--mode full-file "
                "takes the whole-globe objects instead, and there --area "
                "is optional request identity)")
        if source == "gdas":
            if args.cadence is not None and args.hours == 0:
                raise ValueError(
                    "--hours 0 fetches the f000 analysis alone; --cadence "
                    "does not apply to a single time")
            hours = gdas_forecast_hours(
                args.hours,
                3 if args.cadence is None else args.cadence,
                args.forecast_start_hour)
        else:
            cadence = args.cadence if args.cadence is not None else 3
            hours = gfs_forecast_hours(
                args.hours, cadence, args.forecast_start_hour)
        if hours[0]:
            print(f"fetch {source}: window begins at forecast lead "
                  f"f{hours[0]:03d}; a model initialized there starts from "
                  f"a {hours[0]} h forecast, not an analysis")
        if args.cycle == "latest":
            cycle = resolve_latest_cycle(source, hours[-1])
            print(f"fetch {source}: latest complete cycle is "
                  f"{cycle:%Y-%m-%dT%H}Z")
        else:
            cycle = parse_cycle(args.cycle, source)
            require_published_cycle(source, cycle, hours[-1])
        # The request-identity guard and the transfer it authorises are
        # one decision: taking the lock around BOTH is what stops two
        # writers from passing the guard together and then publishing
        # incompatible receipts into the same directory.  The library
        # call re-enters the same lock (it is re-entrant per process).
        requested_gfs_mode = ("full-file" if gfs_fullfile
                              else "nomads-cgi-subset")
        with fetch_guard.hold("fetch-out", args.out):
            if not args.force_refetch:
                check_prior_request(args.out, source=source, cycle=cycle,
                                    area=area)
                prior = _load_fetch_manifest(args.out)
                if prior is not None:
                    recorded_mode = prior.get("mode") or "nomads-cgi-subset"
                    if recorded_mode != requested_gfs_mode:
                        raise ValueError(layered(
                            f"--out {args.out} already holds a "
                            f"{recorded_mode} fetch and this request is "
                            f"{requested_gfs_mode}.\n"
                            "  remedy: fetch into a different --out, or "
                            "pass --force-refetch to move the existing "
                            "files aside (nothing is deleted) and "
                            "re-download this request.",
                            "  why: the two transports name their files "
                            "differently and verify them against "
                            "different bars, so resuming one onto the "
                            "other would publish a manifest mixing two "
                            "requests' bytes."))
            if gfs_fullfile:
                engine, engine_bin = resolve_fetch_engine(
                    args.engine if args.engine is not None else "auto")
                print(f"fetch {source}: engine {engine}"
                      + (f" ({engine_bin})" if engine_bin is not None
                         else "")
                      + ", mode full-file (whole pgrb2.0p25 objects from "
                        "the S3 archive)")
                manifest = fetch_gfs_fullfile(
                    cycle=cycle, hours=hours, area=area, out=args.out,
                    force=args.force_refetch, source=source,
                    engine=engine, engine_bin=engine_bin,
                    cache_dir=args.cache_dir,
                    top_pressure_pa=args.p_top_pa,
                    all_levels=args.all_levels)
            else:
                manifest = fetch_gfs(
                    cycle=cycle, hours=hours, area=area, out=args.out,
                    force=args.force_refetch, source=source,
                    accept_inventory_change=args.accept_inventory_change,
                    top_pressure_pa=args.p_top_pa,
                    all_levels=args.all_levels)
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
        # Only an unpinned request may wander between hosts; an operator
        # who named --transport gets that host or an error.
        transport_fallback = (
            tuple(HRRR_TRANSPORTS[1:]) if transport == "auto" else ())
        engine, engine_bin = resolve_fetch_engine(
            args.engine if args.engine is not None else "auto")
        # The default is the fast path wherever the fast path exists.
        # The Python transport can only do .idx range subsets, so on an
        # install without the backbone the default has to stay 'auto' --
        # and that install is told, in one line, what it is paying.
        if args.mode is not None:
            mode = args.mode
            if mode == "idx-subset":
                print("fetch hrrr: --mode idx-subset selected: record "
                      "subsetting saves bandwidth and costs wall clock "
                      "(hundreds of small range GETs per file instead of "
                      "one parallel whole-file transfer).")
        elif engine == "rust":
            mode = HRRR_DEFAULT_MODE
        else:
            mode = "auto"
            # Only when the slow path was inherited rather than chosen.
            if args.engine != "python":
                explain.warn(
                    "gpuwm fetch is using the Python transport, which can "
                    "only do .idx range subsets; whole-file transfers need "
                    "the rust backbone.  Run `gpuwm setup` (or `gpuwm "
                    "fetch-bridges`) to stage it.",
                    "A field measurement of this exact difference: one "
                    "419 MB HRRR file took 560 s through serial .idx range "
                    "requests on a 2 Gbps host, and 27-35 s for the same "
                    "class of file taken whole through the backbone's "
                    "parallel range GETs.  The Python transport is the "
                    "always-available fallback and is correct -- it is "
                    "simply the slow one, and an install should not "
                    "discover that after the download.")
        if engine == "python" and mode != "auto":
            raise ValueError(
                f"--mode {mode} needs the rust fetch backbone: the Python "
                "transport only does .idx range subsets.  Build the "
                "backbone (cd tools/rustwx && cargo build --release "
                "--locked --offline) or drop --mode.")
        # The lead is checked before any network round trip: `--cycle
        # latest` probes for a cycle complete through the END of the
        # window, and a bad lead should not have to pay for a probe to
        # be refused.  --hours stays the window LENGTH on every source,
        # so the window's final lead is lead + length.
        start_hour = _forecast_start_hour(args.forecast_start_hour, 1)
        if args.hours < 1:
            raise ValueError("--hours must be at least 1 (two hourly frames)")
        last_hour = start_hour + args.hours
        if start_hour:
            print(f"fetch hrrr: window begins at forecast lead "
                  f"f{start_hour:02d}; a model initialized there starts "
                  f"from a {start_hour} h forecast, not an analysis")
        if args.cycle == "latest":
            if args.wait_for:
                # Wait mode wants the cycle currently PUBLISHING, so the
                # completeness probe is f00 (has publication begun?), not
                # the final requested hour.  A lead does not change that
                # question, and the window's own horizon check below is
                # what refuses a lead this cycle cannot reach -- in words
                # that name the horizon, which a failed probe would not.
                cycle = resolve_latest_cycle("hrrr", 0)
                print(f"fetch hrrr: latest publishing cycle is "
                      f"{cycle:%Y-%m-%dT%H}Z (f00 probe; --wait-for "
                      "downloads later hours as they appear)")
            else:
                cycle = resolve_latest_cycle("hrrr", last_hour)
                print(f"fetch hrrr: latest complete cycle is "
                      f"{cycle:%Y-%m-%dT%H}Z")
        else:
            cycle = parse_cycle(args.cycle, source)
            require_published_cycle(
                source, cycle,
                hrrr_forecast_hours(args.hours, cycle, start_hour)[-1])
        hours = hrrr_forecast_hours(args.hours, cycle, start_hour)
        # One lock over the guard and the transfer it authorises; see the
        # GFS branch above.
        with fetch_guard.hold("fetch-out", args.out):
            if not args.force_refetch:
                check_prior_request(args.out, source="hrrr", cycle=cycle,
                                    area=area)
            if not args.wait_for:
                # One transport decision per invocation; 'auto' probes
                # NOMADS for the window's final hour pair, falls back S3.
                transport = resolve_hrrr_transport(
                    cycle, transport, last_hour=hours[-1])
            timeout_minutes = (args.wait_timeout_minutes
                               if args.wait_timeout_minutes is not None
                               else HRRR_WAIT_TIMEOUT_DEFAULT_MINUTES)
            # Name who chose the byte mode.  The backbone cannot: it
            # sees `--mode full-file` on its command line and cannot
            # tell a typed flag from this front door's own default,
            # which is where every unqualified `gpuwm fetch` gets it.
            mode_chooser = "you" if args.mode is not None else "the default"
            print(f"fetch hrrr: engine {engine}"
                  + (f" ({engine_bin})" if engine_bin is not None else "")
                  + (f", mode {mode} ({mode_chooser})"
                     if engine == "rust" else ""))
            if (args.transport is None and transport == "nomads"
                    and mode == "full-file"):
                # Said BEFORE the first byte moves, and only when the
                # host was RESOLVED rather than named: an operator who
                # typed `--transport nomads` made a decision, and a
                # decision does not get advice.  Under the S3-first rule
                # above, `auto` reaches NOMADS only when S3 cannot serve
                # the window yet, so this line now explains a real trade
                # instead of apologising for a default.  Measured on one
                # box, one cycle, the same four objects through the same
                # backbone: 348/209/418/255 s from NOMADS against
                # 69/34/45/44 s from S3.
                print("fetch hrrr: NOMADS paces whole-file transfers -- "
                      "expect several times the wall clock of the S3 "
                      "archive for --mode full-file.  S3 carries this "
                      "cycle once it finishes propagating; --transport "
                      "s3 takes it from there.")
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
              f"--source-manifest-sha256 {sha256_file(sums)} "
              f"--valid-time {cycle:%Y-%m-%d_%H:%M:%S}"
              + (f" --forecast-start-hour {hours[0]}" if hours[0] else ""))
        print("  # fetching cannot bind the run's own flags: "
              "--wps-namelist, --geog-root,\n"
              "  # --experiment-config and --output-root are yours to "
              "supply.  The\n"
              "  # --valid-time above is the CYCLE these files came from; "
              "model time zero\n"
              "  # is cycle + the lead, and every stage derives it.\n"
              "  # `rw-wps --show-source hrrr` lists the full argument "
              "contract.")
    elif args.author_front_door_manifest:
        author_gfs_front_door_manifest(
            out=args.out, bridge=args.bridge,
            wps_namelist=args.wps_namelist,
            experiment_config=args.experiment_config,
            static_input=args.static_input,
            static_receipt=args.static_receipt,
            # No tail cut here: the download that just ran already
            # STARTS at --forecast-start-hour, so its series and manifest
            # are the window.  Cutting again would only be a second,
            # redundant statement of the same lead.
            manifest_out=args.manifest_out, source=source,
            forecast_start_hour=None)
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
        print("  # and these two are yours to point at: --wps-namelist "
              "and\n"
              "  # --experiment-config.  The bridge resolves itself "
              "(--bridge PATH\n"
              "  # overrides); `gpuwm doctor` names the one this "
              "install found.")
    return 0


def _resolve_manifest_bridge(source: str) -> Path:
    """The built decoder ``--author-front-door-manifest`` should bind.

    Through :mod:`gpuwm.bridges`, which is where every other consumer
    looks: the environment override, then a checkout's own build, then
    ``libexec/bridges``, then the ``~/.gpuwm/bridges`` that ``gpuwm
    setup`` / ``gpuwm fetch-bridges`` stage into.  Resolving here rather
    than defaulting in argparse keeps the flag's absence meaningful --
    an explicit ``--bridge`` still wins, and still fails loudly when it
    names a file that is not there.
    """

    from gpuwm import bridges

    found = bridges.find_bridge(bridges.SOURCE_DECODERS[source])
    if found is None:
        raise ValueError(
            f"--author-front-door-manifest needs the built "
            f"{bridges.SOURCE_DECODERS[source]}, and none is resolvable on "
            "this install -- run `gpuwm fetch-bridges` (or pass --bridge "
            "PATH), then re-run this command; `gpuwm doctor` prints the "
            "exact steps for this install")
    return found


#: Keys an advisory ``[fetch]`` table may carry (mirroring the CLI flags).
#: The table is emitted by ``gpuwm domain`` and validated -- never silently
#: ignored -- by the experiment loaders, which split it off before the
#: strict experiment schema runs.
FETCH_HINT_KEYS = frozenset({
    "source", "cycle", "hours", "area", "point", "radius_km", "out",
    "cadence", "forecast_start_hour",
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
    area = table.get("area")
    if area is not None:
        # The hint the fetch reads as a box, through the REAL parser and
        # the per-source coverage gate.  The field defect class this
        # closes: `gpuwm domain --source hrrr` printed an --area its own
        # fetch refused, because the wizard and the guard held two
        # definitions of HRRR coverage.  Both now derive from the native
        # grid (source_coverage_envelope), and every emission round-trips
        # through this check before the file is written.
        try:
            validate_fetch_area(table["source"], parse_area(str(area)))
        except ValueError as error:
            raise ValueError(f"[fetch] of {source}: {error}") from error
    point, radius = table.get("point"), table.get("radius_km")
    if point is not None and radius is not None:
        # The --point/--radius-km spelling of the same box, same gate.
        try:
            validate_fetch_area(
                table["source"], area_from_point(str(point), float(radius)))
        except (TypeError, ValueError) as error:
            raise ValueError(f"[fetch] of {source}: {error}") from error
    start_hour = table.get("forecast_start_hour", 0)
    if not isinstance(start_hour, int) or start_hour < 0:
        raise ValueError(
            f"forecast_start_hour = {start_hour!r} in [fetch] of {source} "
            "must be a nonnegative forecast lead")
    if start_hour and table["source"] not in {"gfs", "gdas", "hrrr"}:
        raise ValueError(
            f"[fetch] of {source}: forecast_start_hour applies to "
            "source = gfs|gdas|hrrr only (era5 is a reanalysis, so every "
            "time in it is an analysis and there is no lead to begin at)")
    if table["source"] == "gdas":
        hours = table.get("hours")
        if isinstance(hours, (int, float))                 and start_hour + hours > GDAS_MAX_FORECAST_HOUR:
            raise ValueError(
                f"[fetch] of {source}: "
                f"{gdas_capability_refusal(int(start_hour + hours))}")
    if table["source"] in GFS_CONTAINER_SOURCES:
        # The window this table describes, planned by the planner that
        # would run it.  A hand-written (or 1.4.0-emitted) table pairing
        # `cadence = 3` with `forecast_start_hour = 4` names a download
        # `gpuwm fetch` refuses -- and a lead that would cross the f120
        # hourly publication break names one NCEP never published.
        hours = table.get("hours")
        cadence = table.get("cadence")
        if (isinstance(hours, int) and hours >= 1 and not isinstance(hours, bool)
                and isinstance(cadence, int) and not isinstance(cadence, bool)
                and cadence in (1, 3)):
            try:
                gfs_forecast_hours(hours, cadence, start_hour)
            except ValueError as error:
                raise ValueError(f"[fetch] of {source}: {error}") from error
    if table["source"] == "hrrr":
        # Same rotten-hint rule as GDAS above, for the bound a lead can
        # now cross: a 13Z cycle stops at f18, so `forecast_start_hour =
        # 12` with `hours = 9` describes a window NOAA never published.
        # Caught at config load, in the words the fetch itself would use.
        hours = table.get("hours")
        raw_cycle = table.get("cycle")
        if isinstance(hours, int) and hours >= 1 and isinstance(raw_cycle, str):
            try:
                cycle = parse_cycle(raw_cycle, "hrrr")
            except ValueError:
                cycle = None
            if cycle is not None:
                try:
                    hrrr_forecast_hours(hours, cycle, start_hour)
                except ValueError as error:
                    raise ValueError(
                        f"[fetch] of {source}: {error}") from error


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
        help="move every existing file in --out aside (nothing is "
             "deleted) and re-download this request.  The receipts go "
             "first -- fetch-manifest.json, SHA256SUMS, the series -- so "
             "an interrupted force can never leave a manifest behind "
             "claiming payloads it has already replaced; then payloads, "
             ".idx indexes, stale parts and anything else in the "
             "directory.  Files already set aside by an earlier "
             "quarantine are left untouched, and subdirectories are "
             "yours.  Required when re-fetching a different area/cycle "
             "into the same --out")
    parser.add_argument(
        "--p-top-pa", type=float, default=None, metavar="PA",
        help="gfs/gdas only: the model top (Pa) the fetched atmosphere "
             "must reach.  The pressure ladder is extended upward along "
             "whatever the live inventory publishes until a level sits "
             "at or above it, so --p-top-pa 5000 fetches the 70 and 50 "
             "hPa levels the certified 100 hPa ladder stops short of.  "
             "Omitted, the certified 21-level ladder is fetched exactly "
             "as before (a 10000 Pa source top).  A top the product "
             "cannot serve refuses and names the deepest it can")
    parser.add_argument(
        "--all-levels", action="store_true",
        help="gfs/gdas only: take every isobaric level the product "
             "publishes instead of choosing a ladder.  On the default "
             "NOMADS grib-filter transport this selects every level; "
             "with --mode full-file the whole object already carries "
             "every level and this declares them all for the decode.  "
             "Either way level subsetting stays an opt-in bandwidth "
             "saver rather than a ceiling on the model top")
    parser.add_argument(
        "--engine", default=None, choices=FETCH_ENGINES,
        help="hrrr, and gfs/gdas --mode full-file: which downloader "
             "moves the bytes.  'rust' is "
             "the vendored rw_fetch backbone (16 MiB parallel range "
             "GETs, .idx coalescing, the cross-process NOMADS rate "
             "governor, a disk cache); 'python' is the stdlib transport "
             "and always works; 'auto' (default) uses the backbone when "
             "it is built")
    parser.add_argument(
        "--mode", default=None, choices=FETCH_MODES,
        help="the byte transport.  hrrr (--engine rust): "
             f"'{HRRR_DEFAULT_MODE}' "
             "is the default -- the whole object in parallel range GETs, "
             "which is the pipeline this product is built on; "
             "'idx-subset' is the opt-in bandwidth saver: it selects "
             "records instead of taking the file, saves transfer volume, "
             "costs wall clock, and refuses rather than silently "
             "degrading when the index cannot carry the selection; "
             "'auto' is the probe rule -- take the whole file when the "
             ".idx is absent, malformed, or provably shorter than the "
             "object -- which is what an install without the rust "
             "backbone falls back to.  gfs/gdas: 'full-file' takes the "
             "whole pgrb2.0p25 objects from the S3 archive (either "
             "engine); omitted, the NOMADS grib-filter crop remains the "
             "default, and 'auto'/'idx-subset' refuse -- .idx record "
             "subsetting of the raw objects is not a certified GFS route")
    parser.add_argument(
        "--cache-dir", type=Path, default=None, metavar="DIR",
        help="--engine rust only (hrrr, gfs/gdas --mode full-file): "
             "wx-core disk cache root, keyed "
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
             "series; requires --wps-namelist and --experiment-config "
             "(--bridge defaults to the built decoder this install "
             "resolves)")
    front.add_argument(
        "--bridge", type=Path, default=None, metavar="EXE",
        help="built gfs_grib2_bridge executable; omit it and the same "
             "resolver `gpuwm go` uses finds the one this install has "
             "(checkout build, libexec, then ~/.gpuwm/bridges -- see "
             "gpuwm doctor)")
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
    parser.add_argument(
        "--forecast-start-hour", type=int, default=None, metavar="K",
        help="gfs/gdas/hrrr: the forecast lead the window BEGINS at "
             "(default f000, the analysis).  --hours stays the window "
             "length, so --forecast-start-hour 174 --hours 66 fetches "
             "f174..f240 and nothing before it; an experiment whose "
             "start_time is cycle+K is then initialized from f{K} with "
             "its boundaries from f{K+i}.  With "
             "--author-front-door-manifest on an already-fetched --out, "
             "this authors the manifest over that tail of the existing "
             "series instead of re-downloading it")
    parser.set_defaults(func=fetch_main)
    return parser


__all__ = [
    "AREA_HINT_DECIMALS", "Area", "Era5ValidationReport", "FETCH_HINT_KEYS",
    "area_bounds_inward", "source_coverage_envelope", "validate_fetch_area",
    "FETCH_MANIFEST_SCHEMA", "GFS_FRONT_DOOR_MANIFEST_SCHEMA",
    "GFS_INPUT_MANIFEST_NAME", "GFS_LAKE_DONOR_MARGIN_DEG",
    "HRRR_DEFAULT_MODE", "FETCH_ENGINES", "FETCH_MODES",
    "HRRR_NOMADS_BASE", "HRRR_NOMADS_RETENTION_HOURS", "HRRR_TRANSPORTS",
    "HRRR_WAIT_POLL_SECONDS", "HRRR_WAIT_TIMEOUT_DEFAULT_MINUTES",
    "validate_fetch_hints",
    "GFS_SUBSET_RECORD_COUNT", "Grib1Record", "area_from_point",
    "author_gfs_front_door_manifest", "check_prior_request",
    "count_grib2_messages", "era5_request_template", "fetch_gfs",
    "fetch_gfs_fullfile",
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
