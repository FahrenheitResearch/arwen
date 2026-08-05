"""Loud stand-ins for observations the ingest lane has not delivered yet.

The scorer and the observation ingest are built in parallel, so the scorer
needs something shaped like an observation before any real bytes exist.  The
danger in that is not the placeholder; it is a placeholder that behaves well
enough to be forgotten.  Everything here is built to be impossible to forget:

* **Construction requires an explicit acknowledgement.**  A caller passes
  :data:`STUB_ACKNOWLEDGEMENT` by value or gets a refusal.  Nothing
  constructs one of these by default, by fallback, or by an empty argument.
* **Every construction warns**, as an :class:`ObsStubWarning` and on stderr.
* **Every field carries ``is_stub`` in its provenance**, so the score file
  built from it says so, and the promotion evaluator's integrity clause
  refuses that evidence outright.  A stubbed score can be computed, plotted
  and read; it can never become a verdict.

The synthetic fields are deliberately *plausible* rather than trivial -- drifting
blobs with a coverage hole, station series with a diurnal cycle -- because a
placeholder that scores 0 or 1 exercises none of the arithmetic that will run
against real data.  They are still not observations of anything.

**The contract these assume**, and what the integration wave should diff:
the ingest lane returns :class:`~gpuwm.verify.obs.contracts.ObsGridField` and
:class:`~gpuwm.verify.obs.contracts.StationObsSet` values in the seam units,
with a real archive URI, the SHA-256 taken at fetch, and ``is_stub`` false.
When that lands, deleting the stub source from a harness call is the entire
integration: the protocols are identical.
"""

from __future__ import annotations

import hashlib
import sys
import warnings
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np

from gpuwm.verify.obs.contracts import (
    ObsGridField, ObsProvenance, Station, StationObsSet, StationReport,
    format_valid_time, normalize_longitude, parse_valid_time,
)

#: The exact string a caller must pass to build a stand-in.  Spelled as a
#: sentence so it cannot be typed by accident and cannot be read as anything
#: other than what it is.
STUB_ACKNOWLEDGEMENT = "these-are-not-observations-they-are-a-stand-in"

#: Marker that appears in every stubbed provenance's source field.
STUB_SOURCE = "STAND-IN-NOT-AN-OBSERVATION"


class ObsStubWarning(UserWarning):
    """Raised as a warning whenever a stand-in observation is manufactured."""


def _announce(what: str) -> None:
    message = (
        f"{STUB_SOURCE}: {what}. These values are manufactured by "
        f"gpuwm.verify.obs.stubs and are not measurements of anything. Any "
        f"score computed from them is a plumbing check, never a result.")
    warnings.warn(message, ObsStubWarning, stacklevel=3)
    print(f"[{STUB_SOURCE}] {what}", file=sys.stderr, flush=True)


def _acknowledge(acknowledgement: str, what: str) -> None:
    if acknowledgement != STUB_ACKNOWLEDGEMENT:
        raise ValueError(
            f"refusing to manufacture {what}: pass "
            f"acknowledgement={STUB_ACKNOWLEDGEMENT!r} to state in the "
            f"calling code that these are not observations")
    _announce(what)


def _digest_provenance(*, product: str, payload: bytes, valid_time: str,
                       reason: str) -> ObsProvenance:
    """Provenance for manufactured bytes: a real digest, a fake origin.

    The digest is genuine -- it is the SHA-256 of the array bytes -- so the
    scoring pass's re-hash step exercises end to end.  The URI is a
    ``stand-in://`` scheme that resolves to nothing, which is the point.
    """
    return ObsProvenance(
        source=STUB_SOURCE,
        product=str(product),
        uri=f"stand-in://{product}/{valid_time}",
        sha256=hashlib.sha256(payload).hexdigest(),
        fetched_at=format_valid_time(datetime(1970, 1, 1, 0, 0, 0)),
        is_stub=True,
        stub_reason=str(reason),
    )


def regular_grid(*, center_latitude: float, center_longitude: float,
                 shape: tuple[int, int], spacing_deg: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """A plain regular latitude/longitude mesh, for stand-in grids only."""
    ny, nx = shape
    if ny < 2 or nx < 2 or spacing_deg <= 0.0:
        raise ValueError("a stand-in grid needs at least 2x2 cells and a spacing")
    latitudes = (center_latitude
                 + (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0)
                 * float(spacing_deg))
    longitudes = (center_longitude
                  + (np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0)
                  * float(spacing_deg))
    mesh_lon, mesh_lat = np.meshgrid(longitudes, latitudes)
    return mesh_lat, normalize_longitude(mesh_lon)


class StubGriddedObsSource:
    """A stand-in gridded observation archive.

    Produces a drifting field of Gaussian blobs on a regular mesh, with a
    fixed coverage hole so the validity mask is exercised.  The drift makes
    consecutive hours genuinely different fields, which is what a scoring
    pass needs to be exercised at all.
    """

    def __init__(self, *, acknowledgement: str, quantity: str,
                 center_latitude: float, center_longitude: float,
                 shape: tuple[int, int] = (200, 200),
                 spacing_deg: float = 0.02,
                 peak_value: float = 60.0,
                 background_value: float = -10.0,
                 blob_count: int = 4,
                 drift_anchor_hour: float = 12.0,
                 drift_deg_per_hour: float = 0.02,
                 coverage_hole_fraction: float = 0.03,
                 seed: int = 20260803) -> None:
        _acknowledge(acknowledgement,
                     f"a stand-in gridded {quantity} archive")
        self._quantity = str(quantity)
        self._latitude, self._longitude = regular_grid(
            center_latitude=center_latitude,
            center_longitude=center_longitude, shape=shape,
            spacing_deg=spacing_deg)
        self._peak = float(peak_value)
        self._background = float(background_value)
        self._drift = float(drift_deg_per_hour)
        self._anchor_hour = float(drift_anchor_hour)
        generator = np.random.default_rng(int(seed))
        span = float(spacing_deg) * max(shape) / 2.0
        self._blobs = [
            (float(center_latitude + generator.uniform(-span / 3, span / 3)),
             float(center_longitude + generator.uniform(-span / 3, span / 3)),
             float(generator.uniform(0.10, 0.25) * span))
            for _ in range(int(blob_count))]
        self._valid = np.ones(self._latitude.shape, dtype=bool)
        hole = int(round(float(coverage_hole_fraction) * self._valid.size))
        if hole > 0:
            flat = generator.choice(self._valid.size, size=hole, replace=False)
            self._valid.ravel()[flat] = False
        self._units = {"composite_reflectivity": "dBZ",
                       "precipitation_accumulation": "mm"}[self._quantity]

    def quantity(self) -> str:
        return self._quantity

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """The stand-in mesh, so a caller can build a matching model grid."""
        return self._latitude.copy(), self._longitude.copy()

    def field(self, valid_time: str) -> ObsGridField:
        instant = parse_valid_time(valid_time)
        hours = (instant - datetime(instant.year, instant.month, instant.day)
                 ).total_seconds() / 3600.0
        values = np.full(self._latitude.shape, self._background,
                         dtype=np.float64)
        for latitude, longitude, radius in self._blobs:
            drifted_lon = longitude + self._drift * (hours - self._anchor_hour)
            distance = np.hypot(self._latitude - latitude,
                                self._longitude - drifted_lon)
            values = np.maximum(
                values,
                self._background
                + (self._peak - self._background)
                * np.exp(-0.5 * (distance / radius) ** 2))
        values = np.clip(values, self._background, self._peak)
        if self._quantity == "precipitation_accumulation":
            values = np.clip(values - self._background, 0.0, None)
        return ObsGridField(
            quantity=self._quantity,
            valid_time=valid_time,
            values=np.ascontiguousarray(values, dtype=np.float64),
            valid=self._valid.copy(),
            latitude=self._latitude.copy(),
            longitude=self._longitude.copy(),
            units=self._units,
            provenance=_digest_provenance(
                product=f"stand-in-{self._quantity}",
                payload=np.ascontiguousarray(values).tobytes(),
                valid_time=valid_time,
                reason=("the observation ingest lane has not delivered this "
                        "product yet")),
        )


class StubStationObsSource:
    """A stand-in surface observation archive.

    Stations are laid out on a coarse mesh inside the requested box and report
    every half hour with a diurnal temperature cycle, a dewpoint depression
    and a wind speed.  One station is deliberately given an elevation far from
    any plausible model terrain so the admission rules have something to drop.
    """

    def __init__(self, *, acknowledgement: str,
                 center_latitude: float, center_longitude: float,
                 station_count: int = 48, span_deg: float = 3.0,
                 elevation_m: float = 250.0,
                 report_interval_minutes: int = 30,
                 seed: int = 20260803) -> None:
        _acknowledge(acknowledgement, "a stand-in surface observation archive")
        if station_count < 2:
            raise ValueError("a stand-in station archive needs 2+ stations")
        generator = np.random.default_rng(int(seed))
        side = int(np.ceil(np.sqrt(station_count)))
        offsets = np.linspace(-span_deg / 2.0, span_deg / 2.0, side)
        stations: list[Station] = []
        for index in range(int(station_count)):
            row, column = divmod(index, side)
            # One station is placed at an impossible elevation on purpose:
            # the terrain-mismatch rule must have something to fire on.
            elevation = (float(elevation_m) + float(generator.normal(0.0, 20.0))
                         if index else float(elevation_m) + 4000.0)
            stations.append(Station(
                station_id=f"STANDIN{index:03d}",
                latitude=float(center_latitude + offsets[row]),
                longitude=float(center_longitude + offsets[column]),
                elevation_m=elevation))
        self._stations = tuple(stations)
        self._interval = int(report_interval_minutes)
        self._generator = generator

    def stations(self) -> tuple[Station, ...]:
        return self._stations

    def observations(self, valid_times: Sequence[str]) -> StationObsSet:
        if not valid_times:
            raise ValueError("a stand-in archive needs valid times to cover")
        instants = sorted(parse_valid_time(text) for text in valid_times)
        start = instants[0] - timedelta(hours=1)
        stop = instants[-1] + timedelta(hours=1)
        reports: list[StationReport] = []
        payload = bytearray()
        step = timedelta(minutes=self._interval)
        for index, station in enumerate(self._stations):
            instant = start
            while instant <= stop:
                hour = instant.hour + instant.minute / 60.0
                temperature = (289.0 + 6.0 * np.sin((hour - 9.0) / 24.0 * 2 * np.pi)
                               - 0.004 * station.elevation_m + 0.1 * index)
                dewpoint = temperature - 4.0 - 0.5 * np.cos(hour / 24.0 * 2 * np.pi)
                speed = 4.0 + 2.0 * np.sin((hour - 15.0) / 24.0 * 2 * np.pi) + 0.02 * index
                values = {
                    "temperature_2m": float(temperature),
                    "dewpoint_2m": float(dewpoint),
                    "wind_speed_10m": float(max(0.0, speed)),
                    "mslp": float(101300.0 - 3.0 * index),
                }
                reports.append(StationReport(
                    station_id=station.station_id,
                    valid_time=format_valid_time(instant),
                    values=values))
                payload.extend(
                    f"{station.station_id}{instant}{values}".encode("utf-8"))
                instant += step
        return StationObsSet(
            stations=self._stations,
            reports=tuple(reports),
            provenance=_digest_provenance(
                product="stand-in-surface-reports",
                payload=bytes(payload),
                valid_time=format_valid_time(instants[0]),
                reason=("the observation ingest lane has not delivered the "
                        "surface network yet")),
        )


def uses_stub(*provenances: ObsProvenance) -> bool:
    """Whether any of these artifacts is a stand-in."""
    return any(bool(item.is_stub) for item in provenances)


__all__ = [
    "STUB_ACKNOWLEDGEMENT", "STUB_SOURCE", "ObsStubWarning",
    "StubGriddedObsSource", "StubStationObsSource", "regular_grid",
    "uses_stub",
]
