"""Observation sources: what the scorer asks of an archive, answered.

These are the ingest lane's side of the seam declared in
``gpuwm.verify.obs.contracts``. Each class satisfies one of that module's
protocols over packs and records the Rust front doors wrote:

* :class:`MrmsCompositeSource` — ``GriddedObsSource`` for composite
  reflectivity in dBZ;
* :class:`OperaCompositeSource` — the same quantity from the European
  network, for domains MRMS does not cover;
* :class:`Stage4PrecipSource` — ``GriddedObsSource`` for one accumulation
  window of precipitation in mm;
* :class:`AsosSurfaceSource` — ``StationObsSource`` for surface reports in
  SI.

The contracts module is imported lazily, and deliberately. The scorer lane
and this lane land in separate integration waves; a module-level import
would make every ingest test depend on the other lane's tree being present,
which is exactly the coupling the seam exists to avoid. Constructing a
source without the contracts module raises and says so.

**Frame selection.** A source answers a valid time with the nearest frame
inside its registered matching window. Under a registered coverage floor
(``minimum_observed_fraction``, registration v2.1) it answers with the
nearest frame that *meets* the floor, walking outward inside the same
window; the window never widens. When the window holds frames but none of
them clears the floor, the source raises the seam's
``ObservedFractionBelowFloor`` so the scorer can record that lead as
missing-obs by name -- an upstream ingest outage should not be scored as if
it were weather, and it should not kill a case either.

**Re-hashing.** Every source exposes ``verify(provenance)``, which re-reads
the archive object the provenance names and compares its digest to the one
taken at fetch. The promotion rule's integrity clause requires that round
trip on every scored input, and an unperformed check counts as a failed one:
a missing artifact raises (that is a wiring fault) and a changed one returns
False (that is the corruption this exists to catch).
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PureWindowsPath

import numpy as np

from gpuwm.obs.obspack import read_geo_pack, read_grid_pack, sha256_of

#: Seam quantity ids, repeated here so a source can name its own quantity
#: without importing the contracts module at import time.
QUANTITY_COMPOSITE_REFLECTIVITY = "composite_reflectivity"
QUANTITY_PRECIPITATION_ACCUMULATION = "precipitation_accumulation"

_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

#: Default matching window for a gridded frame, in seconds. The battery
#: registers +/- 4 minutes around each hourly valid time for the two-minute
#: reflectivity cadence.
DEFAULT_MATCH_SECONDS = 240


def _contracts():
    """The scorer's contract module, imported at use rather than at import."""

    try:
        from gpuwm.verify.obs import contracts
    except ImportError as error:  # pragma: no cover - exercised at integration
        raise RuntimeError(
            "gpuwm.verify.obs.contracts is not importable; the observation "
            "sources build the scorer's own dataclasses and cannot do so "
            "without it. This module is the scoring lane's; a tree that has "
            "the ingest lane but not the scorer lane can still fetch, decode "
            "and verify, but cannot construct seam objects."
        ) from error
    return contracts


def _parse_time(value: str) -> datetime:
    return datetime.strptime(str(value), _TIME_FORMAT)


@dataclass(frozen=True)
class _Frame:
    """One decoded frame on disk, indexed by its own valid time.

    ``observed_fraction`` is the share of the packed subdomain the analysis
    actually observed, as the front door recorded it. ``None`` means the
    pack does not carry the number, and an unknown coverage is never screened
    on -- that is the behaviour every frame had before the coverage floor
    existed, and inventing a value for it would be worse than not screening.
    """

    valid_time: datetime
    pack_path: Path
    observed_fraction: float | None = None


def _index_packs(paths, quantity: str) -> list[_Frame]:
    """Read each pack's metadata and index it by valid time.

    Only the metadata is read here; the payload is left on disk until a
    frame is actually asked for, because a 24-hour case is seventeen scored
    frames out of seven hundred available ones.
    """

    frames: list[_Frame] = []
    for path in paths:
        pack = read_grid_pack(path)
        found = str(pack.meta.get("quantity", ""))
        if found != quantity:
            raise ValueError(
                f"{path} carries quantity {found!r}, this source serves "
                f"{quantity!r}")
        sentinels = dict(pack.meta.get("sentinels", {}) or {})
        observed = sentinels.get("observed_fraction")
        frames.append(_Frame(valid_time=_parse_time(pack.meta["valid_time"]),
                             pack_path=Path(path),
                             observed_fraction=(None if observed is None
                                                else float(observed))))
    frames.sort(key=lambda frame: frame.valid_time)
    times = [frame.valid_time for frame in frames]
    duplicates = {t for t in times if times.count(t) > 1}
    if duplicates:
        raise ValueError(
            f"two packs claim the same valid time(s): "
            f"{sorted(t.strftime(_TIME_FORMAT) for t in duplicates)}; which "
            f"one is the observation would then be an ordering accident")
    return frames


class _GriddedSource:
    """Shared machinery for a gridded archive read out of packs."""

    def __init__(self, pack_paths, geo_pack, *, quantity: str, units: str,
                 match_seconds: int = DEFAULT_MATCH_SECONDS,
                 source: str = "", product: str = "",
                 minimum_observed_fraction: float | None = None):
        paths = [Path(p) for p in pack_paths]
        if not paths:
            raise ValueError(
                f"a {quantity} source needs at least one pack; an empty "
                f"source scores every arm on nothing and reports a clean zero")
        if match_seconds <= 0:
            raise ValueError("match_seconds must be positive")
        if (minimum_observed_fraction is not None
                and not 0.0 <= float(minimum_observed_fraction) <= 1.0):
            raise ValueError("a coverage floor is a fraction in [0, 1]")
        self.minimum_observed_fraction = (
            None if minimum_observed_fraction is None
            else float(minimum_observed_fraction))
        self._frames = _index_packs(paths, quantity)
        self._times = [frame.valid_time for frame in self._frames]
        self._geo_pack = Path(geo_pack)
        self._quantity = quantity
        self._units = units
        self._match = timedelta(seconds=int(match_seconds))
        self._source = source
        self._product = product
        self._geo: tuple[np.ndarray, np.ndarray] | None = None

    # -- the GriddedObsSource protocol ---------------------------------

    def quantity(self) -> str:
        """The seam quantity id this source serves."""

        return self._quantity

    def field(self, valid_time: str):
        """The observed field nearest ``valid_time`` within the window."""

        contracts = _contracts()
        target = contracts.parse_valid_time(valid_time)
        frame = self._nearest(target)
        pack = read_grid_pack(frame.pack_path)
        values = np.asarray(pack.array("values"), dtype=np.float64)
        valid = np.asarray(pack.array("valid")).astype(bool)
        latitude, longitude = self._geometry()
        if latitude.shape != values.shape:
            raise ValueError(
                f"{frame.pack_path}: values {values.shape} do not match the "
                f"geometry pack {self._geo_pack} {latitude.shape}; a field "
                f"and the coordinates it is scored on must be one grid")
        meta_provenance = dict(pack.meta.get("provenance", {}))
        provenance = contracts.ObsProvenance(
            source=meta_provenance.get("source", self._source),
            product=meta_provenance.get("product", self._product),
            uri=meta_provenance.get("uri", str(frame.pack_path)),
            sha256=meta_provenance.get("sha256", ""),
            fetched_at=meta_provenance.get("fetched_at", ""),
        )
        return contracts.ObsGridField(
            quantity=self._quantity,
            valid_time=contracts.format_valid_time(frame.valid_time),
            values=values,
            valid=valid,
            latitude=latitude,
            longitude=longitude,
            units=self._units,
            provenance=provenance,
        )

    # -- integrity ------------------------------------------------------

    def verify(self, provenance, *, root=None) -> bool:
        """Re-hash the archive object ``provenance`` names.

        Returns True when the bytes still hash to the digest taken at fetch.
        A missing artifact raises rather than returning False: that is a
        wiring fault, and reporting it as a corruption would send the reader
        looking for a disk problem that is not there.

        The front doors record an absolute path, so the common case needs
        nothing else. ``root`` covers the case where the cache was moved
        between fetching and scoring: the object is then looked for by its
        own basename under ``root``. It is an explicit argument rather than a
        fallback the reader applies on its own, because a search that happens
        silently is a search that will one day find a different file with the
        same name and report it verified.

        The basename is taken with the rule that reads *both* separators,
        because the recorded path carries the separators of whichever box
        fetched the bytes. A POSIX ``Path`` does not split a Windows path at
        all -- ``name`` is then the whole string -- so relocating a
        Windows-pulled archive onto a Linux node, which is the direction this
        argument exists for, would otherwise look for one absurdly named file
        and raise. Nothing about the comparison changes: the digest still
        decides, and an object that is not the registered one fails.
        """

        uri = getattr(provenance, "uri", None) or dict(provenance)["uri"]
        expected = (getattr(provenance, "sha256", None)
                    or dict(provenance)["sha256"])
        path = Path(uri)
        if root is not None and not path.is_file():
            path = Path(root) / PureWindowsPath(str(uri)).name
        if not path.is_file():
            raise FileNotFoundError(
                f"cannot re-hash {path}: the archive object this field's "
                f"provenance names is not on disk")
        return sha256_of(path) == str(expected).lower()

    # -- internals ------------------------------------------------------

    def valid_times(self) -> tuple[str, ...]:
        """Every valid time this source holds a frame for, ascending."""

        return tuple(t.strftime(_TIME_FORMAT) for t in self._times)

    def _geometry(self) -> tuple[np.ndarray, np.ndarray]:
        if self._geo is None:
            self._geo = read_geo_pack(self._geo_pack)
        return self._geo

    def _inside_window(self, target: datetime) -> list[_Frame]:
        """Every frame inside the matching window, nearest first.

        The order is the selection rule, written once: ascending ``|dt|``,
        and on an exact tie the earlier frame. Deterministic and independent
        of pack order on disk, because a selection that depends on a
        directory listing is a selection nobody can reproduce.
        """

        return sorted(
            (frame for frame in self._frames
             if abs(frame.valid_time - target) <= self._match),
            key=lambda frame: (abs(frame.valid_time - target),
                               frame.valid_time))

    def _nearest(self, target: datetime) -> _Frame:
        """The frame this source scores at ``target``.

        Nearest inside the matching window, and -- when a coverage floor is
        registered -- the nearest one that *meets* it. The window itself
        never widens: a better-covered frame outside it is not a candidate,
        because a distant frame scored as coincident is a wrong number that
        looks like a right one.

        When the window holds frames but none of them clears the floor, this
        raises the seam's own
        :class:`~gpuwm.verify.obs.contracts.ObservedFractionBelowFloor`
        rather than the plain frame-search failure, so the scorer can record
        the lead as missing-obs by name instead of scoring a mask.
        """

        inside = self._inside_window(target)
        if not inside:
            position = bisect.bisect_left(self._times, target)
            best: _Frame | None = None
            for index in (position - 1, position):
                if 0 <= index < len(self._frames):
                    candidate = self._frames[index]
                    if best is None or abs(candidate.valid_time - target) < abs(
                            best.valid_time - target):
                        best = candidate
            offset = ("none" if best is None
                      else f"{abs(best.valid_time - target).total_seconds():.0f} s")
            raise LookupError(
                f"no {self._quantity} frame within "
                f"{self._match.total_seconds():.0f} s of "
                f"{target.strftime(_TIME_FORMAT)} (nearest: {offset}); "
                f"refusing rather than reaching further, because a distant "
                f"frame scored as coincident is a wrong number that looks "
                f"like a right one")

        floor = self.minimum_observed_fraction
        if floor is None:
            return inside[0]
        for frame in inside:
            # An unrecorded coverage is not screened on; see :class:`_Frame`.
            if frame.observed_fraction is None or frame.observed_fraction >= floor:
                return frame

        contracts = _contracts()
        candidates = [
            {"valid_time": frame.valid_time.strftime(_TIME_FORMAT),
             "offset_seconds": abs(
                 (frame.valid_time - target).total_seconds()),
             "observed_fraction": frame.observed_fraction}
            for frame in inside]
        raise contracts.ObservedFractionBelowFloor(
            f"every {self._quantity} frame within "
            f"{self._match.total_seconds():.0f} s of "
            f"{target.strftime(_TIME_FORMAT)} observes less than "
            f"{floor:.4g} of the packed subdomain "
            f"({', '.join(str(row['observed_fraction']) for row in candidates)}"
            f"); the window is not widened for a better-covered frame, and "
            f"this lead is missing observations rather than mostly mask",
            valid_time=target.strftime(_TIME_FORMAT),
            minimum_observed_fraction=float(floor),
            candidates=candidates)


class MrmsCompositeSource(_GriddedSource):
    """MRMS composite reflectivity, in dBZ, from decoded packs.

    ``minimum_observed_fraction`` is the registered coverage floor for frame
    selection (registration v2.1). Left unset, selection is nearest-frame
    exactly as before.
    """

    def __init__(self, pack_paths, geo_pack, *,
                 match_seconds: int = DEFAULT_MATCH_SECONDS,
                 minimum_observed_fraction: float | None = None):
        super().__init__(
            pack_paths, geo_pack,
            quantity=QUANTITY_COMPOSITE_REFLECTIVITY, units="dBZ",
            match_seconds=match_seconds, source="mrms",
            product="MergedReflectivityQCComposite_00.50",
            minimum_observed_fraction=minimum_observed_fraction)


class OperaCompositeSource(_GriddedSource):
    """EUMETNET OPERA composite reflectivity, in dBZ, from decoded packs.

    The European counterpart of :class:`MrmsCompositeSource`, and deliberately
    the same class of thing: same seam quantity, same units, same pack schema,
    same coverage-floor behaviour. A scorer picks between them by which
    network covers the domain, not by which code path it has to take.

    Two differences are real and both are defaults rather than logic. The
    cadence is five minutes rather than two, so the matching window is 300 s;
    and the product is the OPERA column-maximum composite rather than the
    MRMS one, which is what the provenance says when a pack does not carry
    its own.

    One caveat travels with every frame and is not this class's to resolve:
    the composite's own ``/how/comment`` can declare that part of the field
    was produced by advection extrapolation rather than observed. The front
    door records that note in the pack's ``production`` block. It is a
    statement about what the numbers are, and a verification campaign should
    read it before treating the field as ground truth.
    """

    def __init__(self, pack_paths, geo_pack, *,
                 match_seconds: int = 300,
                 minimum_observed_fraction: float | None = None):
        super().__init__(
            pack_paths, geo_pack,
            quantity=QUANTITY_COMPOSITE_REFLECTIVITY, units="dBZ",
            match_seconds=match_seconds, source="opera",
            product="opera-comp-dbzh-max",
            minimum_observed_fraction=minimum_observed_fraction)


class Stage4PrecipSource(_GriddedSource):
    """Stage-IV precipitation for ONE accumulation window, in mm.

    One instance serves one window; the scorer holds a mapping
    ``{1: hourly, 6: six_hourly}`` because that is how the archive stores
    them and how the spec scores them.
    """

    def __init__(self, pack_paths, geo_pack, *, accumulation_hours: int,
                 match_seconds: int = 1800):
        super().__init__(
            pack_paths, geo_pack,
            quantity=QUANTITY_PRECIPITATION_ACCUMULATION, units="mm",
            match_seconds=match_seconds, source="stage4",
            product=f"ST4-{int(accumulation_hours):02d}h")
        self.accumulation_hours = int(accumulation_hours)
        for path in (frame.pack_path for frame in self._frames):
            pack = read_grid_pack(path)
            found = int(pack.meta.get("accumulation_hours", -1))
            if found != self.accumulation_hours:
                raise ValueError(
                    f"{path} is a {found}-hour accumulation; this source "
                    f"serves {self.accumulation_hours}-hour. Mixing windows "
                    f"would compare one model total against another's obs")


class AsosSurfaceSource:
    """Surface reports from a decoded ``gpuwm-obs.asos-surface.v1`` record."""

    def __init__(self, record_path):
        self.path = Path(record_path)
        self.record = json.loads(self.path.read_text())
        schema = str(self.record.get("schema", ""))
        if schema != "gpuwm-obs.asos-surface.v1":
            raise ValueError(
                f"{self.path} declares schema {schema!r}, expected "
                f"'gpuwm-obs.asos-surface.v1'")

    # -- the StationObsSource protocol ----------------------------------

    def observations(self, valid_times):
        """Every report covering ``valid_times``, with its stations."""

        contracts = _contracts()
        wanted = {str(t) for t in valid_times}
        for text in wanted:
            contracts.parse_valid_time(text)
        stations = tuple(
            contracts.Station(
                station_id=str(row["station_id"]),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                elevation_m=float(row["elevation_m"]),
            )
            for row in self.record.get("stations", ())
        )
        reports = tuple(
            contracts.StationReport(
                station_id=str(row["station_id"]),
                valid_time=str(row["valid_time"]),
                values={str(k): float(v)
                        for k, v in dict(row.get("values", {})).items()},
                flags=tuple(str(f) for f in row.get("flags", ())),
            )
            for row in self.record.get("reports", ())
            if str(row["valid_time"]) in wanted
        )
        meta = dict(self.record.get("provenance", {}))
        provenance = contracts.ObsProvenance(
            source=meta.get("source", "asos"),
            product=meta.get("product", "iem-asos-metar"),
            uri=meta.get("uri", str(self.path)),
            sha256=meta.get("sha256", ""),
            fetched_at=meta.get("fetched_at", ""),
        )
        return contracts.StationObsSet(stations=stations, reports=reports,
                                       provenance=provenance)

    # -- integrity ------------------------------------------------------

    verify = _GriddedSource.verify

    @property
    def station_table_sha256(self) -> str:
        """The digest of the frozen station table this record was built on."""

        return str(self.record.get("station_table_sha256", ""))


__all__ = ["AsosSurfaceSource", "DEFAULT_MATCH_SECONDS",
           "MrmsCompositeSource", "OperaCompositeSource",
           "QUANTITY_COMPOSITE_REFLECTIVITY",
           "QUANTITY_PRECIPITATION_ACCUMULATION", "Stage4PrecipSource"]
