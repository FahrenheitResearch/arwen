"""Receipt writer, input ledger and causality check for nowcast frames.

This module is the producer half of the ``gpuwm-obs.nowcast-frames.v1``
contract (``docs/nowcast-frames.md``).  Any 2D reflectivity source writes
that format; the window adapter (``rw_nexrad grid-composite``) reads only
it.  Everything here is standard library plus numpy so it can be tested on
a CPU without the model package, and so the runner's refusals fire before
any GPU time is spent.

What it does NOT do: decode, regrid or reshape any weather data.  Frames
are written exactly as the model hands them over (a raw float32 dump), and
the lattice block is derived from the model's own coordinate arrays by
projecting their four edges, which is metadata for the receipt, not a
transform of the field.

Refusals raise :class:`NowcastRefusal`, and each one names the breakage it
prevents (gate law):

* an input stamped after the issue time: the run would be a hindcast
  scored as a forecast and would overstate skill;
* two history slots resolving to the same object, or a slot more than the
  tolerance off nominal: the model would be told the storms stood still
  or jumped;
* a weights revision other than the pinned one: the receipt would name
  weights that did not make the frames;
* a satellite that was not GOES-East at the issue time, or an issue time
  inside a handover gap of the period table: the model would be
  conditioned on the wrong scan geometry;
* a lattice whose spacing is not uniform, or whose corners do not
  reproduce from its own projection numbers: the adapter would heat the
  wrong columns;
* a lightning bin the read files do not cover: the model would read an
  outage of the source as no lightning;
* an issue time off the satellite source's 5-minute grid, a history
  request before the input archive or across a GOES-East handover: the
  source would fail mid-fetch on the rented box;
* a card another run holds: both runs would run out of memory.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

SCHEMA = "gpuwm-obs.nowcast-frames.v1"
INPUTS_SCHEMA = "gpuwm-obs.nowcast-inputs.v1"
SOURCES_SCHEMA = "gpuwm-obs.nowcast-sources.v1"

STATUS_RUNNING = "RUNNING"
STATUS_READY = "READY"
STATUS_REFUSED = "REFUSED"
STATUS_FAILED = "FAILED"

DTYPE = "float32-le"
CORNER_TOLERANCE_M = 100.0
# Residual allowed between a lattice's projected edge coordinates and the
# straight line through them.  The model's lat/lon arrays may be stored in
# float32, whose resolution near 100 W is about 1.5e-5 degrees (about 1.3 m),
# so a real uniform 3 km grid fits well inside this; a grid with one
# irregular step of even a few percent of a cell does not.
UNIFORM_TOLERANCE_M = 5.0

_HASH_CHUNK = 1 << 22


class NowcastRefusal(RuntimeError):
    """A refusal that names what was refused and the breakage it prevents."""

    def __init__(self, what: str, breakage: str):
        self.what = what
        self.breakage = breakage
        super().__init__(f"refused: {what}. Breakage prevented: {breakage}")


# ---------------------------------------------------------------- time


def parse_utc(text: str) -> datetime:
    """Parse ``2026-10-01T18:00Z`` (or with seconds, or ``+00:00``) as UTC.

    A string without a zone is refused: a naive time is the classic way an
    issue time ends up shifted by a local offset.
    """
    s = text.strip()
    if s.endswith("Z") or s.endswith("z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(f"not an ISO 8601 time: {text!r}") from exc
    if dt.tzinfo is None:
        raise ValueError(f"time {text!r} carries no zone; write it with a trailing Z")
    return dt.astimezone(timezone.utc)


def fmt_utc(dt: datetime) -> str:
    """``2026-10-01T18:00:00Z``."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def frame_dirname(dt: datetime) -> str:
    """``20261001T1800Z``, the per-frame directory name of the contract."""
    dt = dt.astimezone(timezone.utc)
    if dt.second or dt.microsecond:
        raise NowcastRefusal(
            f"frame time {fmt_utc(dt)} is not on a whole minute",
            "two frames could share one directory name",
        )
    return dt.strftime("%Y%m%dT%H%MZ")


def parse_stamp(text: str, fmt: str) -> datetime:
    """Parse an object-name time stamp.

    ``fmt`` is a :func:`datetime.strptime` format, or ``"goes"`` for the
    GOES ``YYYYJJJHHMMSSt`` form (day of year, then tenths of a second).
    """
    if fmt == "goes":
        if len(text) != 14 or not text.isdigit():
            raise ValueError(f"not a GOES stamp: {text!r}")
        base = datetime.strptime(text[:13], "%Y%j%H%M%S")
        base = base + timedelta(milliseconds=100 * int(text[13]))
    else:
        base = datetime.strptime(text, fmt)
    return base.replace(tzinfo=timezone.utc)


def history_slots(issue: datetime, step_minutes: int, frames: int) -> list[datetime]:
    """Nominal history times, oldest first, ending at the issue time."""
    if frames < 1 or step_minutes < 1:
        raise ValueError("history needs at least one frame and a positive step")
    step = timedelta(minutes=step_minutes)
    return [issue - (frames - 1 - k) * step for k in range(frames)]


# ---------------------------------------------------------------- sources table


def load_sources(path: str | os.PathLike) -> dict[str, Any]:
    """Load ``sources.toml`` and check its schema line."""
    import tomllib

    with open(path, "rb") as fh:
        table = tomllib.load(fh)
    if table.get("schema") != SOURCES_SCHEMA:
        raise ValueError(f"{path}: schema is {table.get('schema')!r}, expected {SOURCES_SCHEMA!r}")
    return table


_ROW_KEYS = (
    "kind", "driver", "package", "revision", "variant", "step_minutes",
    "history_frames", "primary_variable", "output_variables", "units",
    "history_tolerance_seconds", "lattice", "streams", "goes_east",
    "issue_align_minutes", "archive_from", "no_coverage_ceiling_dbz",
)


def source_row(table: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    """One row of the source table, with its required keys checked."""
    rows = table.get("sources", {})
    if source_id not in rows:
        known = ", ".join(sorted(rows)) or "none"
        raise KeyError(f"unknown nowcast source {source_id!r}; the table has: {known}")
    row = dict(rows[source_id])
    missing = [k for k in _ROW_KEYS if k not in row]
    if missing:
        raise ValueError(f"source {source_id!r} is missing keys: {', '.join(missing)}")
    if row["primary_variable"] not in row["output_variables"]:
        raise ValueError(f"source {source_id!r}: primary_variable is not an output variable")
    row["id"] = source_id
    return row


def check_revision(requested: str | None, pinned: str) -> str:
    """Refuse any weights revision other than the pinned one.

    ``requested`` may be a full revision, or a package root of the form
    ``hf://owner/repo@<revision>``.  ``None`` means "use the pin".
    """
    if requested is None:
        return pinned
    rev = requested.rsplit("@", 1)[-1] if "@" in requested else requested
    if rev != pinned:
        raise NowcastRefusal(
            f"weights revision {rev!r} is not the pinned revision {pinned!r}",
            "the receipt would name weights that did not make the frames",
        )
    return rev


def goes_east_satellite(when: datetime, periods: Sequence[Mapping[str, Any]]) -> str:
    """The GOES-East satellite for ``when`` from the period table.

    A time inside no period (before the first, or in a handover gap where
    the satellite sources disagree) is refused.
    """
    for p in periods:
        start = parse_utc(p["from"])
        end = parse_utc(p["until"]) if p.get("until") else None
        if start <= when and (end is None or when < end):
            return str(p["satellite"])
    raise NowcastRefusal(
        f"{fmt_utc(when)} falls in no GOES-East period of the source table "
        "(before the record, or inside a handover gap where the satellite "
        "and lightning sources disagree on which satellite was East)",
        "the model would be conditioned on the wrong scan geometry",
    )


def check_satellite(requested: str | None, when: datetime,
                    periods: Sequence[Mapping[str, Any]]) -> str:
    """Return the GOES-East satellite, refusing a different requested one."""
    expected = goes_east_satellite(when, periods)
    if requested is not None and requested != expected:
        raise NowcastRefusal(
            f"satellite {requested!r} was not GOES-East at {fmt_utc(when)} "
            f"(the period table says {expected!r})",
            "the model would be conditioned on the wrong scan geometry",
        )
    return expected


def check_request_window(issue: datetime, requests: Iterable[datetime], satellite: str,
                         periods: Sequence[Mapping[str, Any]], archive_from: str,
                         align_minutes: int) -> None:
    """Refuse a plan the data sources would reject only after the box has started fetching.

    * an issue time off the ``align_minutes`` grid: the ABI source accepts
      only 5-minute-aligned request times and raises mid-fetch;
    * a request before ``archive_from`` (the start of the radar archive the
      source reads): the source raises mid-fetch;
    * a request in a different GOES-East period than the issue time (a
      history window that straddles a handover): the satellite source refuses
      the early slots, or the history would mix two scan geometries.
    """
    issue = issue.astimezone(timezone.utc)
    minutes = issue.hour * 60 + issue.minute
    if align_minutes < 1 or issue.second or issue.microsecond or minutes % align_minutes:
        raise NowcastRefusal(
            f"issue time {fmt_utc(issue)} is not on the {align_minutes}-minute grid",
            "the satellite source accepts only aligned request times and would fail "
            "after the rented box had already fetched the other streams",
        )
    start = parse_utc(archive_from)
    for when in requests:
        if when < start:
            raise NowcastRefusal(
                f"history request {fmt_utc(when)} is before the input archive starts "
                f"({fmt_utc(start)})",
                "the radar source would fail mid-fetch on the rented box",
            )
        other = goes_east_satellite(when, periods)
        if other != satellite:
            raise NowcastRefusal(
                f"history request {fmt_utc(when)} falls in the {other} period while the "
                f"issue time falls in the {satellite} period",
                "the history would mix two scan geometries, or the satellite source "
                "would refuse the early slots mid-fetch",
            )


# ---------------------------------------------------------------- input ledger


@dataclass
class InputRecord:
    """One object a data source read."""

    uri: str
    bucket: str
    key: str
    stream: str | None
    stamp: datetime | None
    satellite: str | None
    slot: datetime | None
    sha256: str | None = None
    nbytes: int | None = None
    # The span the object covers, when its name carries one (GLM files do).
    start: datetime | None = None
    end: datetime | None = None
    # True when the read call returned but left no local file: the source
    # skipped an object it listed (GLM does this for a file gone from S3).
    missing: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "stream": self.stream,
            "bucket": self.bucket,
            "key": self.key,
            "stamp": fmt_utc(self.stamp) if self.stamp else None,
            "satellite": self.satellite,
            "slot": fmt_utc(self.slot) if self.slot else None,
            "sha256": self.sha256,
            "bytes": self.nbytes,
            "missing": self.missing,
        }


def split_uri(uri: str) -> tuple[str, str]:
    """``s3://bucket/key`` or ``bucket/key`` -> ``(bucket, key)``."""
    s = uri
    for prefix in ("s3://", "s3a://", "https://", "http://"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    bucket, _, key = s.partition("/")
    return bucket, key


def sha256_file(path: str | os.PathLike) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


class InputLedger:
    """Every object the data sources read, with its stamp and history slot.

    Reads are attributed to a slot by running each slot's fetch inside
    :meth:`slot`.  A read whose object name matches no stream of the source
    table is kept with no stamp; :func:`check_history` refuses it, because
    a time that cannot be read cannot be shown to be causal.
    """

    def __init__(self, streams: Sequence[Mapping[str, Any]]):
        self.streams = [dict(s) for s in streams]
        self._patterns = [(s["name"], re.compile(s["key_regex"]), s) for s in self.streams]
        self.records: list[InputRecord] = []
        self._slot: datetime | None = None

    @contextlib.contextmanager
    def slot(self, nominal: datetime) -> Iterator[None]:
        previous = self._slot
        self._slot = nominal
        try:
            yield
        finally:
            self._slot = previous

    @property
    def current_slot(self) -> datetime | None:
        """The history slot whose fetch is running, or None outside any slot."""
        return self._slot

    def classify(self, key: str) -> tuple[str | None, datetime | None, str | None]:
        stream, stamp, sat, _, _ = self._classify(key)
        return stream, stamp, sat

    def _classify(self, key: str) -> tuple[str | None, datetime | None, str | None,
                                           datetime | None, datetime | None]:
        for name, pattern, stream in self._patterns:
            m = pattern.search(key)
            if not m:
                continue
            fmt = stream["stamp_format"]
            stamp = parse_stamp(m.group(stream["stamp_group"]), fmt)
            sat = None
            if "sat" in pattern.groupindex and m.group("sat"):
                sat = f"goes{m.group('sat')}"
            start = end = None
            if "start" in pattern.groupindex and "end" in pattern.groupindex:
                start = parse_stamp(m.group("start"), fmt)
                end = parse_stamp(m.group("end"), fmt)
            return name, stamp, sat, start, end
        return None, None, None, None, None

    def record(self, uri: str, local_path: str | os.PathLike | None = None) -> InputRecord:
        bucket, key = split_uri(str(uri))
        for rec in self.records:
            if rec.key == key and rec.bucket == bucket and rec.slot == self._slot:
                return rec  # the same object read again for the same slot (a cache hit)
        stream, stamp, sat, start, end = self._classify(key)
        digest = nbytes = None
        missing = False
        if local_path is not None:
            if Path(local_path).is_file():
                digest, nbytes = sha256_file(local_path)
            else:
                missing = True
        rec = InputRecord(uri=str(uri), bucket=bucket, key=key, stream=stream,
                          stamp=stamp, satellite=sat, slot=self._slot,
                          sha256=digest, nbytes=nbytes, start=start, end=end,
                          missing=missing)
        self.records.append(rec)
        return rec

    def latest_stamp(self) -> datetime | None:
        stamps = [r.stamp for r in self.records if r.stamp is not None]
        return max(stamps) if stamps else None

    def to_json(self, issue: datetime, resolution: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        latest = self.latest_stamp()
        return {
            "schema": INPUTS_SCHEMA,
            "issue_time": fmt_utc(issue),
            "latest_input_time": fmt_utc(latest) if latest else None,
            "causal": is_causal(self.records, issue),
            "objects": [r.to_json() for r in self.records],
            "slots": list(resolution or []),
        }


def is_causal(records: Iterable[InputRecord], issue: datetime) -> bool:
    """True only when every record has a stamp and none is after the issue time."""
    recs = list(records)
    return bool(recs) and all(r.stamp is not None and r.stamp <= issue for r in recs)


def tap_reads(obj: Any, method_name: str, ledger: InputLedger,
              uri_of: Callable[[tuple, dict], str] | None = None,
              local_of: Callable[[Any, tuple, dict, Any], str | None] | None = None) -> None:
    """Wrap ``obj.method_name`` so every call records the object it reads.

    The wrapped method's first positional argument is taken as the object
    URI unless ``uri_of`` says otherwise.  ``local_of(obj, args, kwargs,
    result)`` names the local file to hash (the method's return value when
    it is a path).  Works for plain and ``async`` methods; the record is
    made after the read returns, so a failed read is not listed as input.
    """
    original = getattr(obj, method_name)

    def _uri(args: tuple, kwargs: dict) -> str:
        if uri_of is not None:
            return uri_of(args, kwargs)
        if args:
            return str(args[0])
        return str(next(iter(kwargs.values())))

    def _local(args: tuple, kwargs: dict, result: Any) -> str | None:
        if local_of is not None:
            return local_of(obj, args, kwargs, result)
        return result if isinstance(result, (str, os.PathLike)) else None

    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = await original(*args, **kwargs)
            ledger.record(_uri(args, kwargs), _local(args, kwargs, result))
            return result
    else:
        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            ledger.record(_uri(args, kwargs), _local(args, kwargs, result))
            return result

    setattr(obj, method_name, wrapped)


def check_history(ledger: InputLedger, issue: datetime, slots: Sequence[datetime],
                  expected_satellite: str | None, tolerance_s: float) -> list[dict[str, Any]]:
    """Check every read against the issue time and the history slots.

    Returns the per-slot resolution table written into ``inputs.json``.
    Raises :class:`NowcastRefusal` on the first violation, most severe
    first: an unreadable stamp, a stamp after the issue time, the wrong
    satellite, then the per-stream slot rules.
    """
    recs = ledger.records
    if not recs:
        raise NowcastRefusal("no input object was read", "the frames would rest on nothing the receipt can name")
    for r in recs:
        if r.stamp is None:
            raise NowcastRefusal(
                f"input {r.bucket}/{r.key} matches no stream of the source table, so its time is unknown",
                "a time that cannot be read cannot be shown to be at or before the issue time",
            )
    for r in recs:
        if r.stamp > issue:
            raise NowcastRefusal(
                f"input {r.bucket}/{r.key} is stamped {fmt_utc(r.stamp)}, after the issue time {fmt_utc(issue)}",
                "the run would be a hindcast scored as a forecast and would overstate skill",
            )
    if expected_satellite is not None:
        for r in recs:
            if r.satellite is not None and r.satellite != expected_satellite:
                raise NowcastRefusal(
                    f"input {r.key} comes from {r.satellite}, but {expected_satellite} was GOES-East at {fmt_utc(issue)}",
                    "the model would be conditioned on the wrong scan geometry",
                )
    tol = timedelta(seconds=tolerance_s)
    table: list[dict[str, Any]] = []
    for stream in ledger.streams:
        name = stream["name"]
        kind = stream.get("kind", "single")
        mine = [r for r in recs if r.stream == name]
        stray = [r for r in mine if r.slot is None or r.slot not in slots]
        if stray:
            raise NowcastRefusal(
                f"stream {name} read {stray[0].key} outside any history slot",
                "an object the slot table cannot place could have fed any frame",
            )
        seen_keys: dict[str, datetime] = {}
        seen_stamps: dict[datetime, datetime] = {}
        for nominal in slots:
            got = [r for r in mine if r.slot == nominal]
            if not got:
                raise NowcastRefusal(
                    f"stream {name} read nothing for the history slot {fmt_utc(nominal)}",
                    "the model would read an empty frame as clear sky or no lightning",
                )
            for r in got:
                if r.key in seen_keys:
                    raise NowcastRefusal(
                        f"history slots {fmt_utc(seen_keys[r.key])} and {fmt_utc(nominal)} "
                        f"of stream {name} resolve to the same object {r.key}",
                        "the model would be told the storms stood still",
                    )
                seen_keys[r.key] = nominal
            if kind == "single":
                if len(got) > 1:
                    raise NowcastRefusal(
                        f"stream {name} read {len(got)} objects for the history slot {fmt_utc(nominal)}",
                        "the ledger could not say which object fed the frame",
                    )
                r = got[0]
                if r.missing:
                    raise NowcastRefusal(
                        f"stream {name} listed {r.key} for the history slot {fmt_utc(nominal)} "
                        "but the read left no file",
                        "the model would read an empty frame as clear sky",
                    )
                if r.stamp in seen_stamps:
                    raise NowcastRefusal(
                        f"history slots {fmt_utc(seen_stamps[r.stamp])} and {fmt_utc(nominal)} "
                        f"of stream {name} carry the same stamp {fmt_utc(r.stamp)}",
                        "the model would be told the storms stood still",
                    )
                seen_stamps[r.stamp] = nominal
                offset = r.stamp - nominal
                if abs(offset) > tol:
                    raise NowcastRefusal(
                        f"stream {name} slot {fmt_utc(nominal)} resolved to {r.key} stamped "
                        f"{fmt_utc(r.stamp)}, {offset.total_seconds():+.0f} s off nominal "
                        f"(tolerance {tolerance_s:.0f} s)",
                        "the model would be told the storms jumped",
                    )
                table.append({"stream": name, "slot": fmt_utc(nominal), "objects": 1,
                              "stamp": fmt_utc(r.stamp), "offset_seconds": offset.total_seconds()})
            elif kind == "binned":
                bin_minutes = int(stream["bin_minutes"])
                request = nominal + timedelta(seconds=float(stream.get("request_offset_seconds", 0)))
                bin_end = request + timedelta(minutes=bin_minutes)
                offset = bin_end - nominal
                if abs(offset) > tol:
                    raise NowcastRefusal(
                        f"stream {name} slot {fmt_utc(nominal)} accumulates the bin ending "
                        f"{fmt_utc(bin_end)}, {offset.total_seconds():+.0f} s off nominal",
                        "the model would be told the storms jumped",
                    )
                if "max_gap_seconds" in stream:
                    largest = bin_coverage_gap(got, request, bin_end)
                    if largest is None or largest > float(stream["max_gap_seconds"]):
                        what = ("no object names its span" if largest is None
                                else f"its largest gap is {largest:.0f} s")
                        raise NowcastRefusal(
                            f"stream {name} slot {fmt_utc(nominal)} does not cover the bin "
                            f"{fmt_utc(request)} to {fmt_utc(bin_end)}: {what} "
                            f"(allowed {float(stream['max_gap_seconds']):.0f} s)",
                            "the model would read an outage of the source as no lightning",
                        )
                latest = max(r.stamp for r in got)
                table.append({"stream": name, "slot": fmt_utc(nominal), "objects": len(got),
                              "objects_missing": sum(1 for r in got if r.missing),
                              "bin_start": fmt_utc(request), "bin_end": fmt_utc(bin_end),
                              "latest_stamp": fmt_utc(latest),
                              "offset_seconds": offset.total_seconds()})
            else:
                raise ValueError(f"stream {name}: unknown kind {kind!r}")
    return table


def bin_coverage_gap(records: Iterable[InputRecord], start: datetime,
                     end: datetime) -> float | None:
    """Largest stretch of ``[start, end]``, in seconds, that no read object covers.

    Only objects that were really read (not ``missing``) and whose names
    carry a start and an end count.  ``None`` when no object names its span.
    """
    spans = sorted((r.start, r.end) for r in records
                   if not r.missing and r.start is not None and r.end is not None)
    if not spans:
        return None
    cursor = start
    largest = 0.0
    for s, e in spans:
        if e <= cursor:
            continue
        if s > cursor:
            largest = max(largest, (min(s, end) - cursor).total_seconds())
        cursor = max(cursor, e)
        if cursor >= end:
            break
    if cursor < end:
        largest = max(largest, (end - cursor).total_seconds())
    return largest


def causal_candidates(candidates: Iterable[tuple[datetime, str]],
                      limit: datetime | None) -> list[tuple[datetime, str]]:
    """Keep the ``(stamp, uri)`` candidates stamped at or before ``limit``, order kept.

    The radar source ranks the objects near a requested time nearest first,
    after as well as before.  Filtering its list makes each history slot read
    the newest object at or before the slot, so one late or missing file
    falls back to the one before it instead of to a file after the slot.
    """
    if limit is None:
        return list(candidates)
    limit = limit.astimezone(timezone.utc)
    out = []
    for stamp, uri in candidates:
        when = stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
        if when <= limit:
            out.append((stamp, uri))
    return out


def unobserved_cells(raw_primary: Any, valid: Any, ceiling_dbz: float) -> tuple[Any, list[int]]:
    """Cells the observed history did not cover, for masking the frames.

    ``raw_primary`` is the observed composite on the model grid, before the
    model's clear-air fill, shaped ``[slots, ny, nx]``; ``valid`` is the
    model's own valid mask ``[ny, nx]``.  A cell is unobserved when any
    history slot holds NaN or a value at or below ``ceiling_dbz`` (MRMS
    writes -999 where no radar sees the column).  The model fills those
    cells with clear air, so its frames there are not a nowcast of anything.

    Works on numpy arrays and torch tensors alike.  Returns the mask of cells
    to keep (``valid`` and observed in every slot) and, per slot, how many
    valid cells were unobserved.
    """
    gap = (raw_primary != raw_primary) | (raw_primary <= ceiling_dbz)
    counts = [int(v) for v in (gap & valid).sum((1, 2))]
    keep = valid & ~gap.any(0)
    return keep, counts


def nvidia_smi_uuid(device_uuid: Any) -> str:
    """A card UUID in nvidia-smi's form (``GPU-xxxxxxxx-...``), the engine's lock key.

    torch prints a device UUID without the ``GPU-`` prefix nvidia-smi and
    the engine use; without this the nowcast and an engine run on the same
    card would take two different lock files.
    """
    text = str(device_uuid).strip()
    if text.startswith(("GPU-", "MIG-")):
        return text
    return "GPU-" + text


def card_lock_path(gpu_uuid: str) -> Path:
    """The engine's machine-wide lock file for one card (``gpuwm.supervisor``).

    The same path the engine's single-card runs take, keyed by the
    nvidia-smi UUID (``GPU-...``) and rooted at ``GPUWM_GPU_LOCK_ROOT`` when
    set, so a nowcast and a forecast arm can never share one card.
    """
    import tempfile

    digest = hashlib.sha256(gpu_uuid.encode("utf-8")).hexdigest()[:24]
    configured = os.environ.get("GPUWM_GPU_LOCK_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
    elif os.name == "nt":
        root = Path(os.environ.get("PROGRAMDATA", tempfile.gettempdir())) / "gpuwm" / "locks"
    else:
        root = Path(tempfile.gettempdir()) / "gpuwm" / "locks"
    return root / f"gpu-{digest}.lock"


@contextlib.contextmanager
def card_lock(gpu_uuid: str, run_id: str) -> Iterator[Path]:
    """Hold the card's lock for the rollout, refusing a card another run holds."""
    path = card_lock_path(gpu_uuid)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise NowcastRefusal(
                f"card {gpu_uuid} is held by another run ({path})",
                "two runs on one card would both run out of memory and waste the card",
            ) from exc
        owner = json.dumps({"gpu_uuid": gpu_uuid, "pid": os.getpid(), "run_id": run_id,
                            "acquired_at_utc": fmt_utc(datetime.now(timezone.utc))},
                           sort_keys=True).encode("utf-8")
        stream.seek(1)
        stream.truncate()
        stream.write(owner)
        stream.flush()
        try:
            yield path
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def check_history_fields(has_finite: Any, variables: Sequence[str],
                         slots: Sequence[datetime], what: str) -> None:
    """Refuse a fetched history field with no finite value.

    ``has_finite[k][c]`` says whether slot ``k`` of variable ``c`` holds any
    finite value.  The radar source returns an all-NaN field when every
    candidate file failed to decode; caught here, on the CPU, it costs no
    card time.
    """
    for k, nominal in enumerate(slots):
        for c, name in enumerate(variables):
            if not bool(has_finite[k][c]):
                raise NowcastRefusal(
                    f"{what} variable {name} has no finite value for the history slot "
                    f"{fmt_utc(nominal)}",
                    "the model would refuse the non-finite input only after the weights "
                    "were on the card",
                )


def request_time(nominal: datetime, stream: Mapping[str, Any]) -> datetime:
    """The time handed to the data source for one history slot of one stream.

    The offset makes the source's own nearest-object search land on the
    newest object complete at or before the slot; :func:`check_history`
    then verifies what was actually read.
    """
    return nominal + timedelta(seconds=float(stream.get("request_offset_seconds", 0)))


# ---------------------------------------------------------------- lattice


def _lcc_constants(p: Mapping[str, Any]) -> tuple[float, float, float, float]:
    phi1 = math.radians(float(p["truelat1"]))
    phi2 = math.radians(float(p["truelat2"]))
    if abs(phi1 - phi2) < 1e-12:
        n = math.sin(phi1)
    else:
        n = (math.log(math.cos(phi1) / math.cos(phi2))
             / math.log(math.tan(math.pi / 4 + phi2 / 2) / math.tan(math.pi / 4 + phi1 / 2)))
    if n <= 0:
        raise ValueError("only northern-hemisphere Lambert conformal lattices are supported")
    radius = float(p["earth_radius_m"])
    f = math.cos(phi1) * math.tan(math.pi / 4 + phi1 / 2) ** n / n
    phi0 = math.radians(float(p["ref_lat"]))
    rho0 = radius * f / math.tan(math.pi / 4 + phi0 / 2) ** n
    return n, f, rho0, radius


def _wrap180(deg: np.ndarray) -> np.ndarray:
    return (np.asarray(deg, dtype=np.float64) + 180.0) % 360.0 - 180.0


def lcc_forward(lat: Any, lon: Any, p: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Spherical Lambert conformal: degrees -> metres from (ref_lat, stand_lon)."""
    n, f, rho0, radius = _lcc_constants(p)
    phi = np.radians(np.asarray(lat, dtype=np.float64))
    dlon = np.radians(_wrap180(np.asarray(lon, dtype=np.float64) - float(p["stand_lon"])))
    rho = radius * f / np.tan(np.pi / 4 + phi / 2) ** n
    theta = n * dlon
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


def lcc_inverse(x: Any, y: Any, p: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`lcc_forward`; longitudes in [-180, 180)."""
    n, f, rho0, radius = _lcc_constants(p)
    x = np.asarray(x, dtype=np.float64)
    dy = rho0 - np.asarray(y, dtype=np.float64)
    rho = np.hypot(x, dy)
    theta = np.arctan2(x, dy)
    lon = _wrap180(float(p["stand_lon"]) + np.degrees(theta / n))
    lat = np.degrees(2 * np.arctan((radius * f / rho) ** (1 / n)) - np.pi / 2)
    return lat, lon


def great_circle_m(lat1: float, lon1: float, lat2: float, lon2: float, radius: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def _fit_uniform(values: np.ndarray, what: str, tol: float) -> tuple[float, float]:
    """Least-squares ``v = v0 + k * d``; refuse a residual over ``tol`` metres."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size < 2:
        raise NowcastRefusal(f"{what} has fewer than two points", "the lattice spacing cannot be known")
    k = np.arange(v.size, dtype=np.float64)
    d, v0 = np.polyfit(k, v, 1)
    resid = float(np.max(np.abs(v - (v0 + d * k))))
    if not np.isfinite(resid) or resid > tol:
        raise NowcastRefusal(
            f"{what} is not uniformly spaced (largest departure from a straight line {resid:.2f} m, "
            f"tolerance {tol:.2f} m)",
            "the adapter would heat the wrong columns",
        )
    if d == 0:
        raise NowcastRefusal(f"{what} has zero spacing", "the adapter would heat the wrong columns")
    return float(v0), float(d)


def _corner_block(proj: Mapping[str, Any], x0: float, y0: float, dx: float, dy: float,
                  nx: int, ny: int) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for name, j, i in (("j0_i0", 0, 0), ("j0_in", 0, nx - 1),
                       ("jn_i0", ny - 1, 0), ("jn_in", ny - 1, nx - 1)):
        lat, lon = lcc_inverse(x0 + i * dx, y0 + j * dy, proj)
        out[name] = [round(float(lat), 6), round(float(lon), 6)]
    return out


def lattice_from_projection_coords(x_m: Sequence[float], y_m: Sequence[float],
                                   proj: Mapping[str, Any],
                                   tol_m: float = UNIFORM_TOLERANCE_M) -> dict[str, Any]:
    """The contract's ``lambert`` lattice block from 1D projection coordinates.

    ``x_m`` runs along a row (index i), ``y_m`` along a column (index j), both
    in metres from the projection origin (ref_lat, stand_lon).  Spacing that
    is not uniform is refused.  ``dx_m``/``dy_m`` are signed: a negative
    ``dy_m`` means row 0 is the northern edge.
    """
    x0, dx = _fit_uniform(np.asarray(x_m), "the x coordinate", tol_m)
    y0, dy = _fit_uniform(np.asarray(y_m), "the y coordinate", tol_m)
    nx, ny = len(x_m), len(y_m)
    return {
        "kind": "lambert",
        "truelat1": float(proj["truelat1"]),
        "truelat2": float(proj["truelat2"]),
        "stand_lon": float(proj["stand_lon"]),
        "ref_lat": float(proj["ref_lat"]),
        "ref_lon": float(proj["stand_lon"]),
        "earth_radius_m": float(proj["earth_radius_m"]),
        "nx": nx,
        "ny": ny,
        "dx_m": round(dx, 4),
        "dy_m": round(dy, 4),
        "x0_m": round(x0, 4),
        "y0_m": round(y0, 4),
        "row_order": "south_to_north" if dy > 0 else "north_to_south",
        "corners_latlon": _corner_block(proj, round(x0, 4), round(y0, 4),
                                        round(dx, 4), round(dy, 4), nx, ny),
    }


def lattice_from_latlon(lat2d: Any, lon2d: Any, proj: Mapping[str, Any],
                        tol_m: float = UNIFORM_TOLERANCE_M) -> dict[str, Any]:
    """The ``lambert`` lattice block from a model's 2D cell-centre lat/lon.

    Only the four edges are projected (metadata, not a transform of any
    field).  The x of the first and last rows must agree, as must the y of
    the first and last columns, so the grid is rectilinear in the
    projection; then the block's corners, recomputed from its own numbers,
    must land within :data:`CORNER_TOLERANCE_M` of the given corners.
    """
    lat = np.asarray(lat2d, dtype=np.float64)
    lon = np.asarray(lon2d, dtype=np.float64)
    if lat.ndim != 2 or lat.shape != lon.shape:
        raise NowcastRefusal("latitude and longitude are not matching 2D arrays",
                             "the adapter would heat the wrong columns")
    ny, nx = lat.shape
    xr0, _ = lcc_forward(lat[0, :], lon[0, :], proj)
    xrn, _ = lcc_forward(lat[-1, :], lon[-1, :], proj)
    _, yc0 = lcc_forward(lat[:, 0], lon[:, 0], proj)
    _, ycn = lcc_forward(lat[:, -1], lon[:, -1], proj)
    for a, b, what in ((xr0, xrn, "x of the first and last rows"),
                       (yc0, ycn, "y of the first and last columns")):
        gap = float(np.max(np.abs(a - b)))
        if not np.isfinite(gap) or gap > tol_m:
            raise NowcastRefusal(
                f"the {what} differ by up to {gap:.2f} m, so the grid is not rectilinear in this projection",
                "the adapter would heat the wrong columns",
            )
    block = lattice_from_projection_coords(xr0, yc0, proj, tol_m)
    radius = float(proj["earth_radius_m"])
    given = {"j0_i0": (lat[0, 0], lon[0, 0]), "j0_in": (lat[0, -1], lon[0, -1]),
             "jn_i0": (lat[-1, 0], lon[-1, 0]), "jn_in": (lat[-1, -1], lon[-1, -1])}
    for name, (glat, glon) in given.items():
        clat, clon = block["corners_latlon"][name]
        miss = great_circle_m(float(glat), float(glon), clat, clon, radius)
        if miss > CORNER_TOLERANCE_M:
            raise NowcastRefusal(
                f"corner {name} reproduces {miss:.0f} m from the model's own coordinate",
                "the adapter would heat the wrong columns",
            )
    return block


def check_lattice_corners(block: Mapping[str, Any]) -> None:
    """Refuse a ``lambert`` block whose corners do not follow from its numbers."""
    if block.get("kind") != "lambert":
        return
    again = _corner_block(block, block["x0_m"], block["y0_m"], block["dx_m"], block["dy_m"],
                          int(block["nx"]), int(block["ny"]))
    radius = float(block["earth_radius_m"])
    for name, (lat, lon) in block["corners_latlon"].items():
        clat, clon = again[name]
        miss = great_circle_m(float(lat), float(lon), clat, clon, radius)
        if miss > CORNER_TOLERANCE_M:
            raise NowcastRefusal(
                f"lattice corner {name} is {miss:.0f} m from where its projection numbers put it",
                "the adapter would heat the wrong columns",
            )


# ---------------------------------------------------------------- frame files


def write_json_atomic(path: str | os.PathLike, obj: Any) -> str:
    """Write JSON through a temporary file and a rename; return its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


class FrameWriter:
    """Writes ``<root>/<YYYYmmddTHHMMZ>/<variable>.f32`` member slab by member slab.

    Each frame file is ``float32`` little-endian, C order, shape
    ``[members, ny, nx]``.  Members may arrive in batches (an ensemble split
    across several rollouts); a file is renamed into place only when every
    member is written, so a reader never sees a half-written frame.
    """

    def __init__(self, root: str | os.PathLike, issue: datetime, step_minutes: int,
                 members: int, ny: int, nx: int, variables: Sequence[str], primary: str):
        if primary not in variables:
            raise ValueError("primary variable must be one of the written variables")
        self.root = Path(root)
        self.issue = issue
        self.step = int(step_minutes)
        self.members, self.ny, self.nx = int(members), int(ny), int(nx)
        self.variables = list(variables)
        self.primary = primary
        self._written: dict[tuple[int, str], set[int]] = {}
        self._frame_bytes = self.members * self.ny * self.nx * 4

    def _path(self, lead: int, variable: str, partial: bool) -> Path:
        valid = self.issue + timedelta(minutes=lead)
        name = f"{variable}.f32" + (".partial" if partial else "")
        return self.root / frame_dirname(valid) / name

    def write(self, lead_minutes: int, variable: str, member_start: int, array: Any) -> None:
        lead = int(lead_minutes)
        if lead <= 0 or lead % self.step:
            raise NowcastRefusal(
                f"lead {lead_minutes} min is not a positive multiple of the {self.step} min step",
                "a frame would be written at a time no window ends on",
            )
        if variable not in self.variables:
            raise ValueError(f"variable {variable!r} is not one this writer was opened for")
        a = np.asarray(array)
        if a.ndim != 3 or a.shape[1:] != (self.ny, self.nx):
            raise NowcastRefusal(
                f"{variable} slab has shape {a.shape}, expected [m, {self.ny}, {self.nx}]",
                "members on different grids would be averaged cell by cell",
            )
        members = range(member_start, member_start + a.shape[0])
        if member_start < 0 or members.stop > self.members:
            raise ValueError(f"members {members.start}..{members.stop - 1} outside 0..{self.members - 1}")
        done = self._written.setdefault((lead, variable), set())
        if done.intersection(members):
            raise NowcastRefusal(
                f"member slab {members.start}..{members.stop - 1} of {variable} at +{lead} min was written twice",
                "two rollouts would claim the same ensemble member",
            )
        path = self._path(lead, variable, partial=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not done:
            # The first slab of this frame in this run starts a fresh file of
            # exactly the frame's size, so a partial file left by a failed
            # earlier run (other members, other grid) cannot leak bytes in.
            with open(path, "wb") as fh:
                fh.truncate(self._frame_bytes)
        raw = np.ascontiguousarray(a, dtype="<f4").tobytes()
        with open(path, "r+b") as fh:
            fh.seek(member_start * self.ny * self.nx * 4)
            fh.write(raw)
        done.update(members)

    def finish(self) -> list[dict[str, Any]]:
        """Rename complete frames into place and return the ``frames[]`` list."""
        leads = sorted({lead for lead, _ in self._written})
        if not leads:
            raise NowcastRefusal("no frame was written", "the receipt would promise frames that do not exist")
        frames: list[dict[str, Any]] = []
        for lead in leads:
            entry: dict[str, Any] = {}
            extra: dict[str, Any] = {}
            for variable in self.variables:
                done = self._written.get((lead, variable), set())
                if len(done) != self.members:
                    raise NowcastRefusal(
                        f"{variable} at +{lead} min has {len(done)} of {self.members} members",
                        "a missing member would be read as an all-zero storm field",
                    )
                part = self._path(lead, variable, partial=True)
                final = self._path(lead, variable, partial=False)
                os.replace(part, final)
                digest, nbytes = sha256_file(final)
                rel = final.relative_to(self.root).as_posix()
                if variable == self.primary:
                    valid = self.issue + timedelta(minutes=lead)
                    entry = {"valid": fmt_utc(valid), "lead_minutes": lead, "file": rel,
                             "shape": [self.members, self.ny, self.nx], "dtype": DTYPE,
                             "sha256": digest}
                else:
                    extra[variable] = {"file": rel, "sha256": digest, "dtype": DTYPE}
            entry["extra"] = extra
            frames.append(entry)
        expected = list(range(self.step, leads[-1] + 1, self.step))
        if [f["lead_minutes"] for f in frames] != expected:
            raise NowcastRefusal(
                f"frame leads {[f['lead_minutes'] for f in frames]} have a gap",
                "part of the forced period would run on the model while the record says forced",
            )
        return frames


# ---------------------------------------------------------------- receipt


def build_receipt(*, status: str, source: Mapping[str, Any], issue: datetime,
                  ledger_json: Mapping[str, Any] | None, members: int, variable: str,
                  units: str, lattice: Mapping[str, Any] | None,
                  frames: Sequence[Mapping[str, Any]], sampler: Mapping[str, Any] | None = None,
                  seed: int | None = None, timing: Mapping[str, Any] | None = None,
                  peak_device_bytes: int | None = None, inputs_sha256: str | None = None,
                  history: Mapping[str, Any] | None = None,
                  refusal: str | None = None) -> dict[str, Any]:
    """Assemble ``nowcast.json``.  Only ``READY`` receipts are consumable."""
    causal = bool(ledger_json and ledger_json.get("causal"))
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "source": dict(source),
        "issue_time": fmt_utc(issue),
        "latest_input_time": (ledger_json or {}).get("latest_input_time"),
        "causal": causal,
        "variable": variable,
        "units": units,
        "members": int(members),
        "lattice": dict(lattice) if lattice else None,
        "frames": [dict(f) for f in frames],
        "inputs": {"file": "inputs.json", "sha256": inputs_sha256},
        "timing": dict(timing or {}),
        "peak_device_bytes": peak_device_bytes,
        "created": fmt_utc(datetime.now(timezone.utc)),
    }
    if sampler is not None:
        receipt["sampler"] = dict(sampler)
    if seed is not None:
        receipt["seed"] = int(seed)
    if history is not None:
        receipt["history"] = dict(history)
    if refusal is not None:
        receipt["refusal"] = refusal
    if status == STATUS_READY:
        if not receipt["frames"]:
            raise NowcastRefusal("a READY receipt with no frames", "the adapter would find nothing to read")
        if lattice is None:
            raise NowcastRefusal("a READY receipt with no lattice", "the adapter could not place any cell")
    return receipt


def write_receipt(root: str | os.PathLike, receipt: Mapping[str, Any]) -> str:
    return write_json_atomic(Path(root) / "nowcast.json", receipt)


def read_receipt(root: str | os.PathLike, verify: bool = True) -> dict[str, Any]:
    """Read ``nowcast.json`` and, by default, check every frame's bytes.

    Refused: a schema other than v1, a status other than READY, a frame file
    whose size or sha256 does not match, a frame whose lead does not match
    its valid time, frames out of order, a lattice whose corners do not
    follow from its numbers, and an inputs ledger whose digest moved.
    """
    root = Path(root)
    receipt = json.loads((root / "nowcast.json").read_text(encoding="utf-8"))
    if receipt.get("schema") != SCHEMA:
        raise NowcastRefusal(f"receipt schema is {receipt.get('schema')!r}", "a reader of v1 would misread the files")
    if receipt.get("status") != STATUS_READY:
        raise NowcastRefusal(
            f"receipt status is {receipt.get('status')!r}, not READY",
            "frames of an unfinished or refused run would drive the heating",
        )
    if receipt.get("lattice"):
        check_lattice_corners(receipt["lattice"])
    if not verify:
        return receipt
    issue = parse_utc(receipt["issue_time"])
    previous = 0
    for frame in receipt["frames"]:
        lead = int(frame["lead_minutes"])
        if parse_utc(frame["valid"]) != issue + timedelta(minutes=lead):
            raise NowcastRefusal(f"frame {frame['file']} lead does not match its valid time",
                                 "a frame would drive the wrong window")
        if lead <= previous:
            raise NowcastRefusal("frames are not in increasing lead order", "a frame would drive the wrong window")
        previous = lead
        files = [(frame["file"], frame["sha256"], frame["shape"])]
        files += [(e["file"], e["sha256"], frame["shape"]) for e in frame.get("extra", {}).values()]
        for rel, digest, shape in files:
            path = root / rel
            want = int(np.prod(shape)) * 4
            if not path.is_file() or path.stat().st_size != want:
                raise NowcastRefusal(f"frame file {rel} is missing or not {want} bytes",
                                     "the adapter would read a truncated field")
            got, _ = sha256_file(path)
            if got != digest:
                raise NowcastRefusal(f"frame file {rel} sha256 {got} does not match the receipt",
                                     "the heating would run on bytes the receipt does not describe")
    inputs = receipt.get("inputs") or {}
    if inputs.get("sha256"):
        got, _ = sha256_file(root / inputs["file"])
        if got != inputs["sha256"]:
            raise NowcastRefusal("inputs.json does not match the receipt's digest",
                                 "the causality record would not be the one the run checked")
    return receipt


def load_frame(root: str | os.PathLike, frame: Mapping[str, Any]) -> np.ndarray:
    """Map one primary frame as ``float32[members, ny, nx]`` (for tests and sheets)."""
    shape = tuple(int(v) for v in frame["shape"])
    return np.fromfile(Path(root) / frame["file"], dtype="<f4").reshape(shape)
