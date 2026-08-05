"""The data contract at the observation seam.

This module is the *seam*: everything the scorer knows about an observation
arrives through the types declared here, and nothing else.  The ingest lane
writes them, the scorer reads them, and the integration wave diffs exactly
this surface -- which is the point of writing it down before either side is
finished rather than discovering the mismatch in a score table.

Three rules the shapes here exist to keep:

* **The scorer never names a case.**  A case is an argument -- a case id
  string, a set of valid times, a station list, a grid -- carried in data.
  Nothing in this package branches on which case it is scoring, so the same
  scorer that scores a December nocturnal event scores a July derecho with no
  edit.  Same for the observation *source*: MRMS and Stage-IV are
  source-specific ingest concerns, and by the time bytes reach here they are
  "a gridded field on its own latitude/longitude with a validity mask".

* **Provenance travels with the values.**  Every field and every station set
  carries the archive object it came from and that object's SHA-256 at fetch,
  so the scoring pass can re-hash and refuse silently-mutated inputs (the
  promotion rule's integrity clause) without a side channel.

* **A stand-in can never be mistaken for an observation.**  Provenance carries
  ``is_stub``; a score file built from stubbed inputs records that fact, and
  the promotion evaluator refuses such evidence outright.  A quiet synthetic
  field that scores like a real one is the failure mode this flag exists to
  make impossible.

Units are pinned, in SI, at the seam.  A source that publishes Celsius or
inches converts on the ingest side; the scorer performs no unit inference,
because a scorer that guesses units eventually guesses wrong on the one case
nobody re-reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

#: Schema id for the seam.  Bump when a field changes meaning, never in place.
CONTRACT_SCHEMA = "gpuwm.obs-battery-contract/v1"

#: Canonical surface variable ids and their pinned SI units.  The ingest lane
#: converts into these; the scorer never converts.  ``mslp`` is carried
#: because it is a published diagnostic, and it is deliberately absent from
#: :data:`SCORED_SURFACE_VARIABLES` -- altimeter-to-MSLP reduction differs
#: between the observation network and the model, so it is reported, not
#: refereed.
SURFACE_UNITS: Mapping[str, str] = {
    "temperature_2m": "K",
    "dewpoint_2m": "K",
    "wind_speed_10m": "m s-1",
    "mslp": "Pa",
}

#: The surface variables that carry a score.  Order is the published order.
SCORED_SURFACE_VARIABLES: tuple[str, ...] = (
    "temperature_2m", "dewpoint_2m", "wind_speed_10m")

#: Gridded observation quantities the scorer understands, with pinned units.
GRID_UNITS: Mapping[str, str] = {
    "composite_reflectivity": "dBZ",
    "precipitation_accumulation": "mm",
}

#: Physically impossible values, used only to catch a units or endianness
#: accident at the seam.  These are NOT quality control -- the registered
#: gross-error screen lives in :mod:`gpuwm.verify.obs.stations` and is a
#: scoring parameter.  These bounds refuse a field that cannot be the
#: quantity it claims to be, which is a different failure.
SEAM_BOUNDS: Mapping[str, tuple[float, float]] = {
    "temperature_2m": (150.0, 350.0),
    "dewpoint_2m": (150.0, 350.0),
    "wind_speed_10m": (0.0, 150.0),
    "mslp": (80000.0, 115000.0),
    "composite_reflectivity": (-40.0, 100.0),
    "precipitation_accumulation": (0.0, 2000.0),
}

_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


def parse_valid_time(value: str) -> datetime:
    """Parse a seam timestamp: ISO-8601, UTC, second resolution, no zone.

    A zone suffix is refused rather than interpreted.  Everything in this
    battery is UTC by construction, and a string that carries a zone is a
    string somebody built from a local clock.
    """
    text = str(value)
    try:
        parsed = datetime.strptime(text, _TIME_FORMAT)
    except ValueError as exc:
        raise ValueError(
            f"valid time {text!r} is not UTC ISO-8601 'YYYY-MM-DDTHH:MM:SS'"
        ) from exc
    return parsed


def format_valid_time(value: datetime) -> str:
    """The seam spelling of an instant."""
    return value.strftime(_TIME_FORMAT)


def normalize_longitude(values: np.ndarray) -> np.ndarray:
    """Longitudes wrapped into ``[-180, 180)``, as float64."""
    array = np.asarray(values, dtype=np.float64)
    return ((array + 180.0) % 360.0) - 180.0


def _check_geo(latitude: np.ndarray, longitude: np.ndarray,
               shape: tuple[int, ...], label: str) -> None:
    for name, array in (("latitude", latitude), ("longitude", longitude)):
        if array.shape != shape:
            raise ValueError(f"{label}: {name} shape {array.shape} != {shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{label}: {name} carries non-finite values")
    if np.any(latitude < -90.0) or np.any(latitude > 90.0):
        raise ValueError(f"{label}: latitude outside [-90, 90]")
    if np.any(longitude < -180.0) or np.any(longitude >= 180.0):
        raise ValueError(f"{label}: longitude is not wrapped into [-180, 180)")


@dataclass(frozen=True, eq=False)
class ObsProvenance:
    """Where one observation artifact came from, and whether it is real.

    ``sha256`` is the digest of the fetched archive object, taken at fetch.
    The scoring pass re-hashes the same object and refuses a mismatch; that
    round trip is the whole reason this field is on the value rather than in
    a manifest somebody has to remember to consult.
    """

    source: str
    product: str
    uri: str
    sha256: str
    fetched_at: str
    is_stub: bool = False
    stub_reason: str = ""

    def __post_init__(self) -> None:
        for name in ("source", "product", "uri", "fetched_at"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"observation provenance needs a {name}")
        digest = str(self.sha256).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef"
                                    for ch in digest):
            raise ValueError("observation provenance sha256 must be 64 hex")
        parse_valid_time(self.fetched_at)
        if self.is_stub and not str(self.stub_reason).strip():
            raise ValueError("a stub provenance must say why it is a stub")
        if not self.is_stub and str(self.stub_reason).strip():
            raise ValueError("a non-stub provenance may not carry a stub reason")

    def record(self) -> dict[str, object]:
        """The JSON form a receipt carries."""
        return {
            "source": self.source, "product": self.product, "uri": self.uri,
            "sha256": self.sha256.lower(), "fetched_at": self.fetched_at,
            "is_stub": bool(self.is_stub), "stub_reason": self.stub_reason,
        }


@dataclass(frozen=True, eq=False)
class ObsGridField:
    """One observed 2-D field on its own grid at one instant.

    ``valid`` is the usability mask: False wherever the source says missing,
    below-coverage, or quality-flagged.  Values under a False mask are never
    read, so a source may leave them at whatever fill it likes -- but they
    must still be finite, because a NaN that a mask *should* have covered is
    exactly the bug this refuses to let through.
    """

    quantity: str
    valid_time: str
    values: np.ndarray
    valid: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    units: str
    provenance: ObsProvenance

    def __post_init__(self) -> None:
        label = f"observed {self.quantity} at {self.valid_time}"
        if self.quantity not in GRID_UNITS:
            raise ValueError(
                f"{label}: unknown gridded quantity; the seam declares "
                f"{sorted(GRID_UNITS)}")
        if self.units != GRID_UNITS[self.quantity]:
            raise ValueError(
                f"{label}: units {self.units!r}, seam pins "
                f"{GRID_UNITS[self.quantity]!r}")
        parse_valid_time(self.valid_time)
        values = np.asarray(self.values)
        if values.ndim != 2 or values.size == 0:
            raise ValueError(f"{label}: values must be a non-empty 2-D grid")
        if values.dtype != np.float64:
            raise ValueError(f"{label}: values must be float64 at the seam")
        mask = np.asarray(self.valid)
        if mask.shape != values.shape or mask.dtype != np.bool_:
            raise ValueError(f"{label}: valid must be a bool mask of one shape")
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{label}: values carry non-finite entries; a missing "
                f"observation belongs under the valid mask, not in the data")
        _check_geo(np.asarray(self.latitude), np.asarray(self.longitude),
                   values.shape, label)
        low, high = SEAM_BOUNDS[self.quantity]
        if np.any(mask):
            observed = values[mask]
            if float(observed.min()) < low or float(observed.max()) > high:
                raise ValueError(
                    f"{label}: valid values span "
                    f"[{observed.min():g}, {observed.max():g}] "
                    f"{self.units}, outside the seam bound [{low:g}, {high:g}]")

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.values.shape[0]), int(self.values.shape[1]))


@dataclass(frozen=True, eq=False)
class Station:
    """A fixed surface observing site."""

    station_id: str
    latitude: float
    longitude: float
    elevation_m: float

    def __post_init__(self) -> None:
        if not str(self.station_id).strip():
            raise ValueError("a station needs an id")
        if not -90.0 <= float(self.latitude) <= 90.0:
            raise ValueError(f"station {self.station_id}: latitude out of range")
        if not -180.0 <= float(self.longitude) < 180.0:
            raise ValueError(
                f"station {self.station_id}: longitude not wrapped into "
                f"[-180, 180)")
        if not np.isfinite(float(self.elevation_m)):
            raise ValueError(f"station {self.station_id}: elevation non-finite")


@dataclass(frozen=True, eq=False)
class StationReport:
    """One report from one station, in seam units.

    A variable a report does not carry is simply absent from ``values``; the
    scorer treats it as missing for that (station, hour) on every arm, so the
    arms stay paired.  ``flags`` carries the source's own quality flags
    verbatim -- the battery honors them and then applies its own screen on
    top, and both steps are visible in the receipt.
    """

    station_id: str
    valid_time: str
    values: Mapping[str, float]
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        parse_valid_time(self.valid_time)
        for name, value in dict(self.values).items():
            if name not in SURFACE_UNITS:
                raise ValueError(
                    f"station {self.station_id}: unknown surface variable "
                    f"{name!r}; the seam declares {sorted(SURFACE_UNITS)}")
            if not np.isfinite(float(value)):
                raise ValueError(
                    f"station {self.station_id}: {name} is non-finite; a "
                    f"missing value is an absent key, not a NaN")


@dataclass(frozen=True, eq=False)
class StationObsSet:
    """Every report in a scoring window, with the stations that produced them."""

    stations: tuple[Station, ...]
    reports: tuple[StationReport, ...]
    provenance: ObsProvenance

    def __post_init__(self) -> None:
        ids = [station.station_id for station in self.stations]
        if len(set(ids)) != len(ids):
            raise ValueError("station set carries duplicate station ids")
        known = set(ids)
        unknown = sorted({report.station_id for report in self.reports}
                         - known)
        if unknown:
            raise ValueError(
                f"reports reference stations absent from the set: {unknown}")

    def by_station(self) -> dict[str, list[StationReport]]:
        grouped: dict[str, list[StationReport]] = {
            station.station_id: [] for station in self.stations}
        for report in self.reports:
            grouped[report.station_id].append(report)
        for reports in grouped.values():
            reports.sort(key=lambda item: item.valid_time)
        return grouped


@dataclass(frozen=True, eq=False)
class ModelGrid:
    """The forecast grid a case is scored on.

    ``dx_m`` is what turns a physical neighborhood radius into a cell count,
    which is the only reason the scorer needs to know anything about geometry
    at all.  ``terrain_m`` is present because the station screen compares
    model terrain against station elevation.
    """

    latitude: np.ndarray
    longitude: np.ndarray
    dx_m: float
    terrain_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        latitude = np.asarray(self.latitude)
        if latitude.ndim != 2 or latitude.size == 0:
            raise ValueError("model grid latitude must be a non-empty 2-D array")
        _check_geo(latitude, np.asarray(self.longitude), latitude.shape,
                   "model grid")
        if not np.isfinite(float(self.dx_m)) or float(self.dx_m) <= 0.0:
            raise ValueError("model grid dx_m must be positive and finite")
        if self.terrain_m is not None:
            terrain = np.asarray(self.terrain_m)
            if terrain.shape != latitude.shape:
                raise ValueError("model terrain shape differs from the grid")
            if not np.all(np.isfinite(terrain)):
                raise ValueError("model terrain carries non-finite values")

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.latitude.shape[0]), int(self.latitude.shape[1]))


@runtime_checkable
class GriddedObsSource(Protocol):
    """What the scorer asks of a gridded observation archive."""

    def quantity(self) -> str:
        """The seam quantity id this source serves."""

    def field(self, valid_time: str) -> ObsGridField:
        """The observed field nearest ``valid_time`` within the source window.

        Raises when no file falls inside the source's registered matching
        window: a silently substituted distant file would be scored as if it
        were coincident.
        """


class ObservedFractionBelowFloor(LookupError):
    """Every frame inside the registered window is too heavily masked.

    Distinct from the plain ``LookupError`` a source raises when the window
    holds no frame at all, because the two mean different things and the
    scorer answers them differently: an empty window is a hole in the
    archive and fails the arm, while a window whose every frame is mostly
    unobserved is an upstream ingest outage, and the registered amendment
    records that lead as missing-obs and leaves it out of the lead mean
    rather than scoring a mask or killing the case.

    It is a ``LookupError`` subclass so a caller written before the
    amendment -- one that catches the frame-search failure -- still behaves
    as it did.  ``candidates`` carries every frame the window held, with the
    observed fraction that disqualified it, so the exclusion can be named in
    the score record instead of asserted.
    """

    def __init__(self, message: str, *, valid_time: str,
                 minimum_observed_fraction: float,
                 candidates: Sequence[Mapping[str, object]] = ()):
        super().__init__(message)
        self.valid_time = str(valid_time)
        self.minimum_observed_fraction = float(minimum_observed_fraction)
        self.candidates = [dict(candidate) for candidate in candidates]


@runtime_checkable
class StationObsSource(Protocol):
    """What the scorer asks of a surface observation archive."""

    def observations(self, valid_times: Sequence[str]) -> StationObsSet:
        """Every report covering the requested valid times, with its stations."""


@runtime_checkable
class ModelFrameSource(Protocol):
    """What the scorer asks of a forecast, whichever model produced it.

    Both models in the three-way are read through one implementation of this
    protocol, so no score can differ because two readers were used.
    """

    def grid(self) -> ModelGrid:
        """Geometry and terrain of the scored domain."""

    def valid_times(self) -> tuple[str, ...]:
        """Every valid time this run can be scored at, ascending."""

    def composite_reflectivity(self, valid_time: str) -> np.ndarray:
        """Column-max reflectivity on the model grid, dBZ."""

    def surface_field(self, valid_time: str, variable: str) -> np.ndarray:
        """One :data:`SURFACE_UNITS` variable on the model grid, seam units."""

    def precipitation_accumulation(self, valid_time: str) -> np.ndarray:
        """Run-total grid-scale precipitation on the model grid, mm."""


__all__ = [
    "CONTRACT_SCHEMA", "GRID_UNITS", "SCORED_SURFACE_VARIABLES",
    "SEAM_BOUNDS", "SURFACE_UNITS", "GriddedObsSource", "ModelFrameSource",
    "ModelGrid", "ObsGridField", "ObsProvenance", "ObservedFractionBelowFloor",
    "Station", "StationObsSet",
    "StationObsSource", "StationReport", "format_valid_time",
    "normalize_longitude", "parse_valid_time",
]
