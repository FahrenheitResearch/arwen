"""Hourly emission frames on the model grid, for the chem processes that emit.

DESIGN section 4 names ``SourceFrames`` (``gpuwm/chem_sources.py``, the
foundation lane's): host-side per-hour fields already on the model grid,
read by a process through ``ctx.frames.at(source, field, valid_time)`` and
never by opening a file.  Until that object exists in the build, an emitting
process asks this module instead, through the same call shape, and a
``ctx.frames`` that is present always wins (:func:`frames_for`).

What a frame is.  One source row, one field, one source hour: the row's
extensive quantity per MODEL cell for that hour (RAVE: kg of PM2.5 per cell
per hour, fire radiative power in MW per cell), float64, produced by
:func:`gpuwm.chem_source_netcdf.ingest` -- Rust decode (``rw_netcdf``), the
row's masks, and the Rust cell-sum remap -- with its receipt kept beside it.
Nothing here decodes, remaps or does arithmetic on a field.

Where the files come from.  :func:`gpuwm.chem_source_netcdf.fetch_hour`
resolves the hour's file name from the publisher's listing, downloads it
once into the cache directory and pins its first-fetch sha256; a cached
file whose bytes changed is refused.  The cache directory is
``GPUWM_CHEM_SOURCE_CACHE`` when set, else ``~/.gpuwm/chem-sources``, one
sub-directory per source row.

Which hour.  The caller decides (the fire process's ``fire_emission_mode``);
this module serves exactly the hour it is asked for and refuses one that is
not posted, naming it -- a substituted hour would invent emissions.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path

__all__ = ["EmissionFrames", "EmissionFrameError", "frames_for",
           "frames_for_context", "source_cache_dir"]

#: Environment override for the emission file cache.
CACHE_ENV = "GPUWM_CHEM_SOURCE_CACHE"

#: Frames kept in memory per source: the current hour, the next one and the
#: previous day's matching hour are all a run needs at once.
_KEEP = 6


class EmissionFrameError(ValueError):
    """An emission frame cannot be supplied; the message names why."""


def source_cache_dir(source: str) -> Path:
    root = os.environ.get(CACHE_ENV)
    base = Path(root) if root else Path.home() / ".gpuwm" / "chem-sources"
    return base / source


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def hour_start(valid_time: datetime) -> datetime:
    """The hour an instant belongs to (a frame covers [H, H + 1 h))."""
    valid_time = _utc(valid_time)
    return valid_time.replace(minute=0, second=0, microsecond=0)


class EmissionFrames:
    """Per-hour emission fields on one domain's mass grid.

    ``latitude``/``longitude``/``map_factors`` are the domain's mass-point
    arrays (ny, nx) and ``dx_m`` its grid spacing; the remap plan is built
    once per source grid and reused for every hour
    (:mod:`gpuwm.chem_source_netcdf`).
    """

    def __init__(self, table, *, latitude, longitude, map_factors, dx_m,
                 reference_time: datetime, end_time: datetime | None = None,
                 fetch=None, ingest=None, listing=None, prefetch=True):
        import numpy as np

        from gpuwm import chem_source_netcdf

        #: The simulation start, aware UTC: the fire process's source clock
        #: counts from it (``reference_time + curr_secs``).
        self.reference_time = _utc(reference_time)
        self.end_time = _utc(end_time) if end_time is not None else self.reference_time
        self._listing = listing or chem_source_netcdf.listing
        self._posted: dict[str, frozenset] = {}
        self.table = table
        self.latitude = np.asarray(latitude, dtype=np.float64)
        self.longitude = np.asarray(longitude, dtype=np.float64)
        self.map_factors = np.asarray(map_factors, dtype=np.float64)
        self.dx_m = float(dx_m)
        self._fetch = fetch or chem_source_netcdf.fetch_hour
        self._ingest = ingest or chem_source_netcdf.ingest
        self._frames: dict[str, OrderedDict] = {}
        self.receipts: dict[tuple[str, str, str], dict] = {}
        #: Read the next source hour on a background thread while the model
        #: runs this one (:meth:`_schedule`).  The thread runs the same
        #: fetch and ingest the request would, so a frame's values do not
        #: depend on which thread read them.
        self._prefetch = bool(prefetch)
        self._pending: dict = {}

    def _row(self, source: str):
        row = self.table.sources.get(source)
        if row is None:
            raise EmissionFrameError(
                f"emission source {source!r} is not referenced by any active "
                "chem row, so no frame of it can be asked for")
        if source not in self.table.enabled_sources:
            raise EmissionFrameError(
                f"emission source {source!r} is not enabled in chem_sources; "
                "enable it or the rows naming it emit nothing")
        return row

    def _referenced(self, source: str) -> frozenset:
        """Every field of ``source`` an active row can ask for.

        A row's emission fields, the fire-power field each names in its
        ``fire`` metadata, and any process input binding.  An hour is
        ingested once for all of them: asked for field by field, the hour's
        file was decoded and remapped once for the emission and again for
        the emission plus its fire power.
        """
        row = self.table.sources[source]
        fields = set()
        for species in self.table.rows:
            for ref in species.emissions:
                if ref["source"] != source:
                    continue
                fields.add(ref["field"])
                fire = row.variables[ref["field"]].get("fire") or {}
                if fire.get("frp"):
                    fields.add(fire["frp"])
        fields.update(name for name, binding in row.variables.items()
                      if binding.get("process"))
        return frozenset(fields)

    def hour(self, source: str, hour: datetime, fields) -> dict:
        """``{field: (ny, nx) float64}`` for source hour ``hour``."""
        row = self._row(source)
        hour = hour_start(hour)
        fields = tuple(fields)
        cache = self._frames.setdefault(source, OrderedDict())
        frame = cache.get(hour)
        if frame is None or any(name not in frame for name in fields):
            wanted = tuple(sorted(set(fields) | set(frame or ())
                                  | self._referenced(source)))
            values = receipts = None
            pending = self._pending.pop((source, hour), None)
            if pending is not None:
                try:
                    values, receipts, read = pending.result()
                except Exception:
                    # Read again here, so a failure raises exactly as it
                    # would have without the prefetch.
                    values = receipts = None
                else:
                    if not set(wanted) <= set(read):
                        values = receipts = None
            if values is None:
                values, receipts, _read = self._read(row, source, hour, wanted)
            frame = dict(values)
            for name, receipt in receipts.items():
                self.receipts[(source, hour.isoformat(), name)] = {
                    key: value for key, value in receipt.items()
                    if key != "valid"}
            cache[hour] = frame
            cache.move_to_end(hour)
            while len(cache) > _KEEP:
                cache.popitem(last=False)
            self._schedule(row, source, hour + timedelta(hours=1), wanted)
        return {name: frame[name] for name in fields}

    def _read(self, row, source: str, hour: datetime, wanted) -> tuple:
        """Fetch and ingest one source hour: ``(values, receipts, wanted)``."""
        path, _fetch_receipt = self._fetch(row, hour, source_cache_dir(source))
        values, receipts = self._ingest(
            row, path, wanted, latitude=self.latitude,
            longitude=self.longitude, map_factors=self.map_factors,
            dx_m=self.dx_m, max_distance_m=self.dx_m, valid_hour=hour)
        return values, receipts, tuple(wanted)

    def _schedule(self, row, source: str, hour: datetime, wanted) -> None:
        """Start reading ``hour`` in the background if it is posted.

        Both fire clocks walk their source forward one hour at a time (the
        same hour, or the same hour a day earlier), so the next hour of the
        source is the next one asked for.  An hour that is not posted, or is
        already held or being read, is left alone; a read nobody asks for
        is dropped with the frames.
        """
        if not self._prefetch or hour > self.end_time:
            return
        posted = self._posted.get(source)
        if (posted is None or hour not in posted
                or hour in self._frames.get(source, {})
                or (source, hour) in self._pending):
            return
        import threading
        from concurrent.futures import Future

        future = Future()
        wanted = tuple(wanted)

        def read():
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(self._read(row, source, hour, wanted))
            except BaseException as error:  # handed to the requester
                future.set_exception(error)

        self._pending[(source, hour)] = future
        # A daemon thread: a read the run never asks for (its last hour's
        # successor) does not hold the process open at exit.
        threading.Thread(target=read, name="chem-frames", daemon=True).start()

    def available_hours(self, source: str, field: str) -> frozenset:
        """Every posted hour of ``source`` the run can ask for.

        Read from the publisher's listings for each month from 26 hours
        before the start (the trailing-24 h hours ``trailing_24h_dcycle`` and
        ``daily_mean_dcycle`` read) through the run's end, once per run.
        ``field`` is accepted for the SourceFrames call shape; a source's
        files carry all of its fields.
        """
        del field
        posted = self._posted.get(source)
        if posted is None:
            row = self._row(source)
            months = []
            month = (self.reference_time - timedelta(hours=26)).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0)
            while month <= self.end_time:
                months.append(month)
                month = (month + timedelta(days=32)).replace(day=1)
            hours = set()
            for month in months:
                _url, entries = self._listing(row, month, source_cache_dir(source))
                hours.update(entries)
            posted = self._posted[source] = frozenset(hours)
        return posted

    def at(self, source: str, field: str, valid_time: datetime):
        """DESIGN 4's call: the frame of the hour ``valid_time`` falls in."""
        return self.hour(source, valid_time, (field,))[field]


def frames_for_context(ctx) -> "EmissionFrames":
    """The domain's :class:`EmissionFrames`, built once and kept on its chem state.

    Geometry is the domain's own: ``xlat``/``xlong`` (the physics driver's
    chem export), the mass-point map factor ``msft`` and ``cfg.dx``.  A
    SourceFrames the driver supplies in ``ctx.frames`` wins
    (:func:`frames_for`).
    """
    def build():
        chem = ctx.state.chem
        frames = getattr(chem, "emission_frames", None)
        if frames is None:
            import cupy as cp

            start = (getattr(ctx.cfg, "start_time", None)
                     or getattr(chem, "start_time", None))
            if not isinstance(start, datetime):
                raise EmissionFrameError(
                    "emission frames need the run's start_time; without it a "
                    "fire hour cannot be told from any other")
            run_seconds = float(getattr(ctx.cfg, "run_seconds", 0.0) or 0.0)
            frames = EmissionFrames(
                ctx.table,
                latitude=cp.asnumpy(cp.asarray(ctx.met("xlat"))),
                longitude=cp.asnumpy(cp.asarray(ctx.met("xlong"))),
                map_factors=cp.asnumpy(cp.asarray(ctx.state.msft)),
                dx_m=float(ctx.cfg.dx), reference_time=start,
                end_time=start + timedelta(seconds=run_seconds))
            chem.emission_frames = frames
        return frames

    return frames_for(ctx, build)


def frames_for(ctx, build):
    """``ctx.frames`` when the foundation supplies one, else ``build()``.

    ``build`` constructs (once) the :class:`EmissionFrames` the process keeps
    on its own state; a SourceFrames the driver hands over always wins, so
    the day it lands this fallback stops being used without an edit.
    """
    frames = getattr(ctx, "frames", None)
    return frames if frames is not None else build()


def previous_day(hour: datetime) -> datetime:
    return hour_start(hour) - timedelta(hours=24)
