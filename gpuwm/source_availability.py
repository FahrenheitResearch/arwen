"""Date guidance for source selectors, derived from the acquisition registry.

This is orchestration metadata, not another fetch implementation. A calendar
never promises that an archive has every object: only the existing provider
probe can confirm a latest forecast, and a keyed analysis service has no such
probe. Unknown record bounds remain unknown.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys


@dataclass(frozen=True)
class ArchiveWindow:
    """A documented bound of one acquisition transport's current file layout."""

    transport: str
    start: str
    note: str
    documentation: tuple[str, ...]
    checked_at: str = "2026-09-06"
    user_note: str = ""
    #: A small JSON document the provider itself keeps current with the
    #: archive's bounds, and the keys in it naming the last published
    #: day, the most inclusive first.  Empty when the provider publishes
    #: none; the publication schedule is then the only estimate.
    bounds_url: str = ""
    bounds_stop_keys: tuple[str, ...] = ()
    #: A publisher door that keeps only a rolling window and has no
    #: archive behind it: how many hours back it serves, measured.  Its
    #: ``start`` is then empty, since a rolling door has no first day.
    #: Read by the date guidance only; the fetch ladder's own retention
    #: column decides which host a download asks.
    retention_hours: float | None = None
    #: An archive that began with only some cycle hours of the day: from
    #: ``early_start`` until ``start`` it holds only ``early_hours`` (UTC),
    #: and every hour from ``start`` on.  Empty when the archive began
    #: whole.  Measured from the archive's own listing, like ``start``.
    early_start: str = ""
    early_hours: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.start or self.retention_hours is None:
            parse_cycle(self.start)
        if self.early_start:
            if not self.start or parse_cycle(self.early_start) >= parse_cycle(self.start):
                raise ValueError("A partial early record starts before the whole record does.")
            if not self.early_hours or any(not 0 <= hour < 24 for hour in self.early_hours):
                raise ValueError("A partial early record names the UTC hours it holds.")
        if self.retention_hours is not None and not self.retention_hours > 0:
            raise ValueError("A rolling window keeps a positive number of hours.")
        if not self.transport or not self.documentation:
            raise ValueError("An archive bound needs its transport and evidence.")


def parse_cycle(value: str) -> datetime:
    """Read an exact UTC hour without silently discarding minutes or offsets."""
    try:
        result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Enter a UTC cycle as YYYY-MM-DDTHH, for example 2026-09-05T06.") from error
    if "T" not in value or result.minute or result.second or result.microsecond:
        raise ValueError("Choose an exact UTC hour: YYYY-MM-DDTHH.")
    if result.tzinfo is not None:
        if result.utcoffset() != timedelta(0):
            raise ValueError("Enter UTC rather than a local timezone offset.")
        result = result.replace(tzinfo=None)
    return result


_BOUNDS: dict[str, tuple[float, datetime | None]] = {}
_BOUNDS_TTL_S = 3600.0


def published_stop(window: ArchiveWindow, *, timeout: float = 6.0) -> datetime | None:
    """The last hour the provider says its archive holds, read from its own bounds document.

    Cached for an hour per document.  A document that cannot be read
    gives None, and the caller falls back to the publication schedule:
    a calendar is guidance, and the acquisition still checks the bytes.
    """

    if not window.bounds_url or not window.bounds_stop_keys:
        return None
    import time
    import urllib.request

    hit = _BOUNDS.get(window.bounds_url)
    if hit is not None and time.monotonic() - hit[0] < _BOUNDS_TTL_S:
        return hit[1]
    stop = None
    try:
        with urllib.request.urlopen(window.bounds_url, timeout=timeout) as reply:
            document = json.loads(reply.read(65536).decode("utf-8"))
        for key in window.bounds_stop_keys:
            value = str(document.get(key) or "").strip()
            if value:
                day = datetime.fromisoformat(value[:10])
                stop = day.replace(hour=23)
                break
    except (OSError, ValueError, AttributeError):
        stop = None
    _BOUNDS[window.bounds_url] = (time.monotonic(), stop)
    return stop


def cached_stop(window: ArchiveWindow) -> datetime | None:
    """:func:`published_stop` from its cache alone: never reaches a server.

    None when the document was never read, or was read and said nothing;
    :func:`bounds_unread` tells the two apart.
    """

    hit = _BOUNDS.get(window.bounds_url) if window.bounds_url else None
    return None if hit is None else hit[1]


def bounds_unread(window: ArchiveWindow) -> bool:
    """True when this window has a provider bounds document not read within its cache hour."""

    import time

    if not window.bounds_url or not window.bounds_stop_keys:
        return False
    hit = _BOUNDS.get(window.bounds_url)
    return hit is None or time.monotonic() - hit[0] >= _BOUNDS_TTL_S


def _utc(now: datetime | None) -> datetime:
    result = now or datetime.now(timezone.utc)
    return (result.astimezone(timezone.utc).replace(tzinfo=None)
            if result.tzinfo is not None else result)


def _stamp(value: datetime | None) -> str | None:
    return None if value is None else value.strftime("%Y-%m-%dT%H")


def _context(source: str | None, hours: float | None,
             config: Path | None, transport: str | None) -> tuple[str, float, str | None]:
    def span(value, label):
        result = float(value)
        if isinstance(value, bool) or not math.isfinite(result) or result < 0:
            raise ValueError(f"{label} must be a finite, nonnegative number of hours.")
        return result

    if hours is not None:
        hours = span(hours, "Duration")
    if config is not None:
        import tomllib
        document = tomllib.loads(config.read_text(encoding="utf-8"))
        fetch = document.get("fetch", {})
        source = source or fetch.get("source") or document.get("case_data", {}).get("source")
        if hours is None:
            duration = document.get("experiment", {}).get("run_seconds")
            if duration is not None:
                hours = span(duration, "Run duration in seconds") / 3600.0
        # The saved acquisition can deliberately cover more than the run.
        # Never call f006 complete when its [fetch] request still asks for
        # f048, or forget a forecast window that begins after initialization.
        requested = 6.0 if hours is None else hours
        fetch_span = span(fetch.get("hours", requested), "Fetch duration")
        lead = span(fetch.get("forecast_start_hour", 0), "Forecast start hour")
        hours = max(requested, fetch_span) + lead
        transport = transport or fetch.get("transport")
    if not source or not source.strip():
        raise ValueError("Choose an input source before opening its calendar.")
    duration = 6.0 if hours is None else float(hours)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Duration must be a finite, nonnegative number of hours.")
    return source.strip(), duration, transport


def _config_cadence(config: Path | None) -> int | None:
    """The lead spacing a saved configuration's ``[fetch]`` asks for, or None when it names none."""

    if config is None:
        return None
    import tomllib
    return tomllib.loads(config.read_text(encoding="utf-8")).get("fetch", {}).get("cadence")


def availability(source: str, hours: float, *, now: datetime | None = None,
                 transport: str | None = None, published=None, cadence: int | None = None) -> dict:
    """Describe expected cycles and transport bounds.

    Nothing here reaches a server unless ``published`` is given: a
    callable taking an :class:`ArchiveWindow` and answering the last
    hour its provider says it holds (:func:`published_stop`), or None.

    Every question is asked about the download a run of ``hours`` makes
    (:func:`gpuwm.domain_wizard.fetch_window`): the source's file spacing
    and the length rounded up to whole steps of it.  ``cadence`` is a
    spacing the caller's fetch names (its ``--cadence``); left out, the
    one ``gpuwm domain`` writes.  ``last_hour`` is that download's last
    lead, ``cadence_hours`` its spacing.
    """
    from gpuwm.source_adapters import get_source_adapter
    from gpuwm.source_cycles import cycle_grid_for
    from gpuwm import fetch_endpoints, fetch_routes
    from gpuwm.fetch import analysis_window_reference, cycle_is_probeable

    source, hours, transport = _context(source, hours, None, transport)
    adapter = get_source_adapter(source)
    source = adapter.source_id
    now = _utc(now)
    grid = cycle_grid_for(source)
    # A reanalysis window contains successive analyses. Its final valid time,
    # rather than just its start, must be behind the declared publication lag.
    analysis = adapter.max_forecast_hour == 0
    # A uniform local GRIB analysis series is anchored to the caller's
    # exact first valid time. Its default six-hour forcing interval is
    # spacing between frames, not a synoptic-only initialization rule:
    # fetch._era5_times and era5_request_template both preserve any UTC
    # starting hour. Keep the ordinary resolver's conservative Latest
    # policy separate from which explicit historical starts are valid.
    hourly_start = analysis and adapter.cadence_mapping == "uniform-local-grib-time-series-v1"
    # The download a run of this length makes, as `gpuwm domain` writes it
    # into [fetch]: a 3-hour forecast from files that come every 6 hours
    # downloads hours 0 and 6.  Every question below, the object probe's
    # included, is about that download.  Asking the fetch about the 3-hour
    # window no run requests raised inside a page request and turned the
    # whole source list into an error.
    from gpuwm.domain_wizard import fetch_window
    cadence, last_hour = fetch_window(source, hours, 0, None if cadence is None else int(cadence))
    route = fetch_routes.table_route(source)
    ladder_step = None if route is None else (cadence or route.default_cadence)
    # The same back-off the acquisition resolver applies, from the same
    # function, so the date this calendar PUBLISHES as the boundary and the
    # date its Latest button resolves cannot be two different dates.
    reference = analysis_window_reference(source, grid, last_hour, now) if grid else None
    newest = grid.newest(reference) if grid else None
    allowed_hours = []
    spaced_out = []
    if grid:
        for hour in (range(24) if hourly_start else grid.hours):
            horizon = grid.horizon(now.replace(hour=hour))
            if horizon is None and not analysis:
                horizon = adapter.max_forecast_hour
            if not (analysis or horizon is None or last_hour <= horizon):
                continue
            if route is not None and not _leads_resolve(route, now.replace(hour=hour), last_hour, ladder_step):
                # The fetch's own lead resolver, which asks no server, as the
                # Latest resolver asks it: a cycle whose file ladder leaves an
                # hour of this window out (files every 3 hours only part of
                # the way) is refused by the download at that hour.
                spaced_out.append(hour)
                continue
            allowed_hours.append(hour)
        if newest is not None and allowed_hours:
            while newest.hour not in allowed_hours:
                newest = grid.snap(newest - timedelta(hours=1))
    # When the ladder alone leaves no cycle hour, the longest window at this
    # spacing that some cycle still serves, for the refusal's words.
    spacing_limit = None
    if spaced_out and not allowed_hours:
        spacing_limit = next((length for length in range(last_hour - ladder_step, 0, -ladder_step)
                              if any(_leads_resolve(route, now.replace(hour=hour), length, ladder_step)
                                     for hour in spaced_out)),
                             None)
    probeable = cycle_is_probeable(source)
    declared = {window.transport: window for window in adapter.archive_windows}
    endpoints = fetch_endpoints.ladder(source) if fetch_endpoints.has_ladder(source) else ()
    if transport and transport != "auto":
        endpoints = tuple(row for row in endpoints if row.name == transport)
        if not endpoints and transport not in declared:
            raise ValueError(f"The source registry does not declare transport {transport!r} for {source}.")
    rows = []
    for endpoint in endpoints:
        window = declared.get(endpoint.name)
        # None in the existing route table means UNDECLARED, not all history.
        rows.append({
            "transport": endpoint.name, "window": window,
            "record_start": None if window is None else (window.early_start or window.start or None),
            **_partial(window),
            "retention_hours": (endpoint.retention_hours if endpoint.retention_hours is not None
                                else None if window is None else window.retention_hours),
            "note": window.note if window else endpoint.why,
            "documentation": [] if window is None else list(window.documentation),
        })
    if not endpoints:
        rows = [{"transport": window.transport, "window": window,
                 "record_start": window.early_start or window.start or None, **_partial(window),
                 "retention_hours": window.retention_hours, "note": window.note,
                 "documentation": list(window.documentation)}
                for window in adapter.archive_windows
                if not transport or transport == "auto" or window.transport == transport]
    # The provider's own word on its newest data, where it publishes one.
    stops = []
    for row in rows:
        window = row.pop("window")
        stop = published(window) if published is not None and window is not None else None
        row["record_end"] = _stamp(stop)
        if stop is not None:
            stops.append(stop)
    if stops and grid is not None and allowed_hours:
        # The whole window must be published: an analysis series ends
        # hours after it starts, a forecast needs only its cycle.
        bound = max(stops) - timedelta(hours=last_hour if analysis else 0)
        bound = grid.snap(bound)
        if newest is None or bound < newest:
            newest = bound
            while newest.hour not in allowed_hours:
                newest = newest - timedelta(hours=1)
    # The newest start past the source's usual publication delay (CycleGrid.usual_delay): where a probe decides
    # publication its schedule declares no delay, so ``newest`` is the cycle still being made, and this is the start
    # a page takes as published while no check has answered for it.
    due = None
    if grid is not None and newest is not None and allowed_hours:
        extra = max(0.0, grid.usual_delay - float(grid.delay_hours))
        due = min(newest, grid.newest(reference - timedelta(hours=extra)))
        while due.hour not in allowed_hours:
            due = grid.snap(due - timedelta(hours=1))
    starts = [parse_cycle(row["record_start"]) for row in rows if row["record_start"]]
    # Only a union with no unknown unbounded member has a known earliest date.
    # Rolling retention itself is advisory, so it never creates a hard cutoff.
    unknown = any(not row["record_start"] and row["retention_hours"] is None for row in rows)
    earliest = min(starts) if starts and not unknown else None
    preparable = fetch_routes.first_preparable_cycle(route) if route is not None else None
    if preparable is not None:
        earliest = max(earliest, preparable) if earliest is not None else preparable
    notes = ["Archive dates describe the selected transport; individual files and source compatibility still need acquisition checks."]
    notes.extend(dict.fromkeys(window.user_note for window in adapter.archive_windows
                              if window.user_note and any(row["transport"] == window.transport for row in rows)))
    if starts and not analysis:
        notes.append("Cycle hours and forecast horizons describe current production; historical product versions can differ.")
    if not probeable:
        notes.append("Latest is an estimate from the publication schedule. This service has no public object completeness probe.")
    if grid and grid.delay_hours:
        notes.append(f"Expected publication delay: about {grid.delay_hours:g} hours; the provider can publish later.")
    if analysis:
        notes.append(f"The whole {hours:g}-hour analysis period must be published, including its end.")
    if hourly_start:
        notes.append("This uniform analysis series may start at any UTC hour. The forcing cadence is spacing between frames; Latest retains the registered default schedule.")
    if earliest is None:
        notes.append("The earliest usable archive date is not fully declared; no historical cutoff is assumed.")
    if any(row["retention_hours"] for row in rows):
        notes.append("Rolling retention is guidance, not a guarantee that every listed file exists.")
    if grid is None:
        notes.append("No cycle schedule is declared. Enter the intended UTC date manually.")
    elif not allowed_hours:
        notes.append("No declared cycle covers this duration. Shorten the period before choosing a date.")
    if last_hour != math.ceil(hours):
        notes.append(f"The files come every {cadence} hours, so the download runs through hour {last_hour}.")
    if transport and transport != "auto":
        notes.append("A transport is explicitly selected. Choose an exact date; automatic Latest may use a different endpoint.")
    return {
        "schema": "gpuwm.source-availability.v1", "source_id": source,
        "display_name": adapter.display_title, "hours": hours, "last_hour": last_hour,
        "now_utc": _stamp(now), "earliest": _stamp(earliest),
        "latest_candidate": _stamp(newest), "due_start": _stamp(due),
        "usual_delay_hours": None if grid is None else grid.usual_delay,
        "cycle_hours": allowed_hours,
        "cycle_grid": None if grid is None else grid.declaration(),
        "analysis": analysis, "probeable": probeable,
        "latest_label": "Latest complete" if probeable else "Latest expected",
        "latest_supported": grid is not None and bool(allowed_hours) and (not transport or transport == "auto"),
        "cadence_hours": ladder_step or cadence, "spacing_limit": spacing_limit,
        "transports": rows, "notes": notes,
        "requirements": [credential.display_name for credential in adapter.credentials],
    }


def _leads_resolve(route, cycle: datetime, hours: int, cadence: int) -> bool:
    """Whether the fetch's lead resolver accepts this cycle hour and window (no server is asked)."""

    from gpuwm import fetch_routes

    try:
        fetch_routes.resolve_leads(route, cycle, hours, cadence=cadence)
    except ValueError:
        return False
    return True


def _partial(window: ArchiveWindow | None) -> dict:
    if window is None or not window.early_start:
        return {}
    return {"whole_start": window.start, "early_hours": list(window.early_hours)}


def _early_gap(document: dict, selected: datetime) -> dict | None:
    """The partial record that leaves ``selected`` out, when every dated transport does."""
    rows = [row for row in document["transports"] if row.get("record_start")]
    if not rows or any(not row.get("whole_start") for row in rows):
        return None
    missing = [row for row in rows if selected < parse_cycle(row["whole_start"])
               and selected.hour not in row["early_hours"]]
    return missing[0] if len(missing) == len(rows) else None


def _hour_list(hours) -> str:
    hours = sorted(hours)
    if hours == list(range(hours[0], hours[-1] + 1)) and len(hours) > 2:
        return f"{hours[0]:02d} to {hours[-1]:02d}"
    return ", ".join(f"{hour:02d}" for hour in hours)


def _route_cycle_refusal(document: dict, selected: datetime, *,
                         preparation_only: bool = True) -> str | None:
    """Use the fetch's dated layout and preparation checks before archive guidance."""
    from gpuwm import fetch_routes

    route = fetch_routes.table_route(document["source_id"])
    if route is None or not route.publication_eras:
        return None
    try:
        era = fetch_routes.publication_era(route, selected)
        if preparation_only and not era.prep_refusal:
            return None
        fetch_routes.resolve_request(
            route.source_id, cycle=selected, hours=document["last_hour"],
            cadence=document["cadence_hours"])
    except ValueError as error:
        return str(error)
    return None


def validate_cycle(document: dict, cycle: str) -> tuple[str, list[str]]:
    selected = parse_cycle(cycle)
    refusal = _route_cycle_refusal(document, selected)
    if refusal:
        raise ValueError(refusal)
    allowed = document["cycle_hours"]
    if document["cycle_grid"] is not None and selected.hour not in allowed:
        choices = ", ".join(f"{hour:02d}Z" for hour in allowed) or "none for this duration"
        raise ValueError(f"Choose a cycle that covers this period: {choices}.")
    latest = document["latest_candidate"]
    if latest and selected > parse_cycle(latest):
        raise ValueError(f"This period is later than the expected publication boundary ({latest} UTC).")
    earliest = document["earliest"]
    if earliest and selected < parse_cycle(earliest):
        raise ValueError(f"The declared acquisition layout begins {earliest} UTC. Earlier data require another supported transport or input archive.")
    gap = _early_gap(document, selected)
    if gap is not None:
        raise ValueError(f"Before {gap['whole_start']} UTC the archive holds only the "
                         f"{_hour_list(gap['early_hours'])} UTC runs.")
    refusal = _route_cycle_refusal(document, selected, preparation_only=False)
    if refusal:
        raise ValueError(refusal)
    return _stamp(selected), list(document["notes"])


def _words(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M UTC")


#: How long after its start a cycle can still be uploading.  Inside it the
#: publication schedule is only an estimate, so a start that the schedule
#: allows is also put to the fetch's own object probe; older starts are
#: judged by the archive bounds alone.  Every probeable source registered
#: today finishes its longest forecast well inside a day.
PUBLICATION_FRONTIER_HOURS = 24


def publication_refusal(document: dict, cycle: str, *, now: datetime | None = None,
                        probe=None) -> str | None:
    """The fetch's own answer for a start near the publication frontier.

    None when the start is older than :data:`PUBLICATION_FRONTIER_HOURS`,
    when the source has no public object to probe, or when the objects
    the run needs through its last hour are published.  Otherwise the
    sentence the fetch itself would refuse with, from
    :func:`gpuwm.fetch.cycle_publication_refusal`, so the two can never
    disagree about one start: a host that could not be heard is not a
    refusal there, and is not one here.
    """
    from gpuwm.fetch import _head_answer, cycle_publication_refusal

    if not document.get("probeable"):
        return None
    selected = parse_cycle(cycle)
    now = _utc(now)
    if now - selected >= timedelta(hours=PUBLICATION_FRONTIER_HOURS):
        return None
    try:
        return cycle_publication_refusal(
            document["source_id"], selected, int(document["last_hour"]), now=now,
            probe=_head_answer if probe is None else probe)
    except ValueError as error:
        # The fetch's own resolver refuses this start and length before any object is asked: that is this
        # start's answer, not an error for whoever asked.
        return str(error)


def confirmed_latest(document: dict, *, now: datetime | None = None, probe=None) -> str | None:
    """The newest start the fetch would accept, for a Latest the page offers.

    ``latest_candidate`` is the publication schedule's estimate.  Inside
    :data:`PUBLICATION_FRONTIER_HOURS` of the clock it is put to the same
    resolver ``--cycle latest`` uses, so the start a page opens on is one
    the fetch accepts; outside it, or for a source with no public object
    to probe, the estimate stands.
    """
    from gpuwm.fetch import _head_ok, resolve_latest_cycle

    candidate = document.get("latest_candidate")
    if not candidate or not document.get("probeable") or not document.get("latest_supported"):
        return candidate
    now = _utc(now)
    if now - parse_cycle(candidate) >= timedelta(hours=PUBLICATION_FRONTIER_HOURS):
        return candidate
    try:
        cycle = resolve_latest_cycle(document["source_id"], int(document["last_hour"]), now=now,
                                     probe=_head_ok if probe is None else probe)
    except (RuntimeError, ValueError):
        return candidate
    return _stamp(min(cycle, parse_cycle(candidate)))


def verdict(document: dict, cycle: str, *, now: datetime | None = None,
            confirm=None) -> dict:
    """Whether one source has data for one start, in a sentence a person can act on.

    ``state`` is ``"yes"``, ``"no"`` or ``"unknown"``: unknown when the
    source declares no archive start and the time is older than any
    rolling retention it declares, so only the download can tell.  The
    same bounds :func:`validate_cycle` refuses on, said plainly and never
    raised, so a page can list every source with its own answer.

    ``why`` names the source, for a sentence read on its own; ``reason``
    is the same sentence without the name, for a list whose row already
    shows it.

    ``confirm`` is :func:`publication_refusal` or a stand-in with its
    signature.  Given, a start the schedule allows is also put to the
    question the fetch asks before downloading, and its refusal becomes
    this answer; left out, nothing here reaches a server.

    A start whose run the download would refuse at a missing file (files
    that thin out before the window ends) is answered no with ``fix``
    naming the lengths that download.
    """

    name = document.get("display_name") or document["source_id"]

    def answer(state: str, why: str, reason: str | None = None) -> dict:
        return {"state": state, "why": why, "reason": why if reason is None else reason}

    try:
        selected = parse_cycle(cycle)
    except ValueError as error:
        return answer("no", str(error))
    now = _utc(now)
    if selected > now:
        return answer("no", "That time is still in the future. Pick a past date, or the newest run.")
    refusal = _route_cycle_refusal(document, selected)
    if refusal:
        return answer("no", refusal)
    # The archive's own bounds come before its cycle hours: a date no hour
    # of which the archive holds must not be told to change the hour.
    earliest = document["earliest"]
    if earliest and selected < parse_cycle(earliest):
        start = _words(parse_cycle(earliest))
        return answer("no", f"The {name} archive this program reads starts {start}.",
                      f"Archive starts {start}.")
    gap = _early_gap(document, selected)
    if gap is not None:
        whole = _words(parse_cycle(gap["whole_start"]))
        hours = _hour_list(gap["early_hours"])
        return answer("no", f"Before {whole} the {name} archive holds only the {hours} UTC runs.",
                      f"Before {whole} the archive holds only the {hours} UTC runs.")
    rows = document["transports"]
    unknown = None
    if not earliest and rows:
        kept = [row["retention_hours"] for row in rows if row["retention_hours"]]
        open_ended = [row for row in rows if not row["retention_hours"] and not row["record_start"]]
        age = (now - selected).total_seconds() / 3600.0
        if kept and not open_ended and age > max(kept):
            days = max(kept) / 24
            span = "day" if days == 1 else f"{days:g} days"
            return answer("no", f"{name} keeps only its last {span}.", f"Keeps only its last {span}.")
        if open_ended and (not kept or age > max(kept)):
            unknown = answer("unknown", f"Where the {name} archive starts is not recorded, so the "
                                        "download will say whether this date is there.",
                             "Where its archive starts is not recorded, so the download will say "
                             "whether this date is there.")
    latest = document["latest_candidate"]
    if latest and selected > parse_cycle(latest):
        newest = _words(parse_cycle(latest))
        if document["analysis"]:
            return answer("no", f"{name} is not published this recent yet. Its newest start for "
                                f"a {document['hours']:g}-hour forecast is {newest}.",
                          f"Not published this recent yet. Its newest start for a "
                          f"{document['hours']:g}-hour forecast is {newest}.")
        return answer("no", f"This {name} run is not published yet. The newest expected is {newest}.",
                      f"This run is not published yet. The newest expected is {newest}.")
    allowed = document["cycle_hours"]
    if document["cycle_grid"] is not None and selected.hour not in allowed:
        limit = document.get("spacing_limit")
        if not allowed and limit:
            # Every run reaches this far, but its files thin out before the end, and the download asks for one
            # every cadence_hours all the way.
            rule = (f"files come every {document['cadence_hours']} hours only through hour {limit}, so a "
                    f"{document['hours']:g}-hour forecast cannot be downloaded.")
            return {"state": "no", "why": f"{name} {rule}", "reason": f"Its {rule}",
                    "fix": f"Pick {limit} hours or less, or another source."}
        if not allowed:
            return answer("no", f"No {name} run is {document['hours']:g} hours long. Pick a shorter forecast.",
                          f"No run is {document['hours']:g} hours long. Pick a shorter forecast.")
        hours = ", ".join(f"{hour:02d}" for hour in allowed)
        return answer("no", f"{name} starts at {hours} UTC for a forecast this long, not at {selected.hour:02d}.",
                      f"Starts at {hours} UTC for a forecast this long, not at {selected.hour:02d}.")
    refusal = _route_cycle_refusal(document, selected, preparation_only=False)
    if refusal:
        return answer("no", refusal)
    if unknown:
        return unknown
    if confirm is not None:
        # The schedule allows it; the fetch asks the objects themselves.
        refusal = confirm(document, cycle, now=now)
        if refusal:
            return answer("no", refusal)
    return answer("yes", "")


# ------------------------------------------------------------------ a page's probe

#: One HEAD a page asks waits at most this long for the host's answer.
PAGE_PROBE_TIMEOUT_S = 5.0
#: ... and at most this long for its turn under the node's NOMADS
#: governor.  A running forecast downloading from NOMADS holds most of
#: those turns; a page that queued behind it waited minutes.
PAGE_PACE_BUDGET_S = 3.0
#: One source's whole check for a page ends by this deadline.  Whatever
#: is unanswered by then is said to be unchecked, never "not published".
PAGE_SOURCE_DEADLINE_S = 15.0
#: HEADs asked at once for one source's objects (never NOMADS, which the
#: governor serializes node-wide anyway).
PAGE_PREFETCH_WORKERS = 8
#: How many of the newest candidate starts are asked for, newest first,
#: before the resolver walks them in order.
PAGE_LATEST_PREFETCH = 3


def quick_head(url: str) -> bool | None:
    """One HEAD for a page: True, False (the host says it is not there) or None (no answer in time)."""

    from gpuwm import fetch_endpoints

    return fetch_endpoints.object_answer(url, timeout=PAGE_PROBE_TIMEOUT_S, max_wait_s=PAGE_PACE_BUDGET_S)


class ProbeSession:
    """The fetch's object probe as a page asks it: remembered, bounded, and plain about silence.

    Called with a URL it answers True or False, the signature every
    probe in :mod:`gpuwm.fetch` takes, so the fetch's own functions ask
    their own question through it.  Behind that, each URL is asked once;
    after ``deadline_s`` nothing more is sent; and every time an answer
    was not an answer (a timeout, a throttle, a governor turn too far
    away, the deadline) :attr:`misses` counts it, so the caller can tell
    "the host said no" from "the host was not heard".
    """

    def __init__(self, *, deadline_s: float = PAGE_SOURCE_DEADLINE_S, head=None, clock=None) -> None:
        import threading
        import time

        self._clock = clock or time.monotonic
        self._deadline = self._clock() + deadline_s
        self._head = head
        self._memo: dict[str, bool | None] = {}
        self._lock = threading.Lock()
        self.misses = 0
        self.asked = 0

    def expired(self) -> bool:
        return self._clock() >= self._deadline

    def _ask(self, url: str) -> bool | None:
        # Looked up at call time, so a test's stand-in for quick_head is the one asked.
        head = self._head or globals()["quick_head"]
        return head(url)

    def answer(self, url: str) -> bool | None:
        with self._lock:
            if url in self._memo:
                found = self._memo[url]
                if found is None:
                    self.misses += 1
                return found
        if self.expired():
            with self._lock:
                self.misses += 1
            return None
        found = self._ask(url)
        with self._lock:
            self.asked += 1
            self._memo[url] = found
            if found is None:
                self.misses += 1
        return found

    def __call__(self, url: str) -> bool:
        return self.answer(url) is True

    def known(self, url: str) -> bool | None:
        """What this URL was answered, without asking: None when it was never asked or not heard."""

        with self._lock:
            return self._memo.get(url)

    def prefetch(self, urls) -> None:
        """Ask these URLs at once, NOMADS excepted, so a walk that asks them one by one reads answers."""

        from concurrent.futures import ThreadPoolExecutor
        from gpuwm.nomads_governor import is_nomads_url

        with self._lock:
            todo = list(dict.fromkeys(url for url in urls if url not in self._memo and not is_nomads_url(url)))
        if not todo or self.expired():
            return
        with ThreadPoolExecutor(max_workers=min(PAGE_PREFETCH_WORKERS, len(todo))) as pool:
            list(pool.map(self._prefetch_one, todo))

    def _prefetch_one(self, url: str) -> None:
        if self.expired():
            return
        found = self._ask(url)
        with self._lock:
            self.asked += 1
            self._memo.setdefault(url, found)

    def complete(self, urls) -> bool:
        """Ask one rung's objects side by side, stopping at the first not answered present; True when all were.

        A rung on NOMADS is left to the walk, which asks it one object at a time under the governor.
        """

        from gpuwm.fetch import objects_published
        from gpuwm.nomads_governor import is_nomads_url

        urls = tuple(urls)
        if not urls or self.expired() or any(is_nomads_url(url) for url in urls):
            return False
        return objects_published(urls, self, workers=PAGE_PREFETCH_WORKERS)


def _rung_urls(source: str, cycle: datetime, last_hour: int, now: datetime, ladder_cycle=None):
    """[(rung, urls)] in the fetch's own ladder order for one cycle, as the fetch walks it."""

    from gpuwm import fetch_endpoints
    from gpuwm.fetch import cycle_probe_urls

    ladder = fetch_endpoints.serving_ladder(source, cycle=ladder_cycle or cycle, now=now, pinned=None)
    return [(endpoint.name, tuple(cycle_probe_urls(source, cycle, last_hour, transport=endpoint.name)))
            for endpoint in ladder]


def _answered_complete(source: str, cycles, last_hour: int, now: datetime, session: ProbeSession):
    """The newest of ``cycles`` that one rung already answered every object present for; nothing new is asked.

    The rungs are the ones :func:`page_start_check` asks for that start, the fetch's rule for an explicit start.
    """

    for cycle in sorted(cycles, reverse=True):
        try:
            rungs = _rung_urls(source, cycle, last_hour, now)
        except (RuntimeError, ValueError, KeyError):
            continue
        if any(urls and all(session.known(url) is True for url in urls) for _, urls in rungs):
            return cycle
    return None


def start_needs_probe(document: dict, cycle: str, *, now: datetime | None = None) -> bool:
    """Whether :func:`page_start_check` would ask any server about this start."""

    if not document.get("probeable"):
        return False
    return _utc(now) - parse_cycle(cycle) < timedelta(hours=PUBLICATION_FRONTIER_HOURS)


def latest_needs_probe(document: dict, *, now: datetime | None = None) -> bool:
    """Whether :func:`page_latest` would ask any server for this source's newest start."""

    candidate = document.get("latest_candidate")
    if not candidate or not document.get("probeable") or not document.get("latest_supported"):
        return False
    return _utc(now) - parse_cycle(candidate) < timedelta(hours=PUBLICATION_FRONTIER_HOURS)


def _previous_start(grid, allowed, moment: datetime) -> datetime:
    """The start before ``moment`` on the source's grid and on the hours that serve the length."""

    for _ in range(24 * 14):
        earlier = moment - timedelta(hours=1)
        moment = grid.snap(earlier) if grid is not None else earlier
        if not allowed or moment.hour in allowed:
            return moment
    return moment


def starts_after(document: dict, after: str | None) -> list[str]:
    """The starts newer than ``after`` on the hours that serve the length, through the schedule's newest start.

    These are the starts a check must have found not yet whole before
    ``after`` can be called the newest run: newest first, and empty when
    ``after`` is the schedule's newest start or newer.
    """

    from gpuwm.source_cycles import cycle_grid_for

    candidate = document.get("latest_candidate")
    if not candidate or not after or candidate <= after:
        return []
    grid = cycle_grid_for(document["source_id"])
    allowed = set(document.get("cycle_hours") or ())
    floor = parse_cycle(after)
    moment = parse_cycle(candidate)
    newer = []
    for _ in range(24 * 14):
        if moment <= floor:
            break
        newer.append(_stamp(moment))
        moment = _previous_start(grid, allowed, moment)
    return newer


def start_is_due(document: dict, cycle: str) -> bool:
    """Whether a start is past its source's usual publication delay (``due_start``, from :attr:`CycleGrid.usual_delay`).

    A page takes such a start as published while no check has answered for
    it; a check that found it missing still wins.
    """

    due = document.get("due_start")
    return bool(due) and parse_cycle(cycle) <= parse_cycle(due)


def opening_start(document: dict, confirmed: str | None, missing=()) -> str | None:
    """The newest start Start takes, for one source and length: what a page opens on.

    The newest start a check confirmed whole (``confirmed``), or the newest
    start past the source's usual publication delay that no check found
    missing, whichever is newer.  Until a check confirms one, that is the
    due start itself.  A page calls this start its newest run only when it is
    ``confirmed`` and a check found every start after it
    (:func:`starts_after`) not yet whole.
    """

    from gpuwm.source_cycles import cycle_grid_for

    due = document.get("due_start")
    moment = parse_cycle(due) if due else None
    gone = set(missing or ())
    allowed = set(document.get("cycle_hours") or ())
    grid = cycle_grid_for(document["source_id"]) if moment is not None else None
    # A start found missing is not taken, however old: the newest one before it that no check found missing is.
    for _ in range(len(gone)):
        if moment is None or _stamp(moment) not in gone:
            break
        moment = _previous_start(grid, allowed, moment)
    best = _stamp(moment)
    if confirmed and (best is None or parse_cycle(confirmed) >= parse_cycle(best)):
        return confirmed
    return best


def page_start_check(document: dict, cycle: str, *, now: datetime | None = None,
                     session: ProbeSession | None = None) -> dict:
    """Whether the objects one start needs are published, asked as the fetch asks it.

    The same rungs, the same URLs and the same rule as
    :func:`gpuwm.fetch.cycle_publication_refusal`: one rung holding
    every object through the run's last hour is enough.  The objects are
    asked at once and the whole check ends by the session's deadline.

    ``state``: ``"published"``, ``"not-published"`` (every object the
    answer rests on was answered), ``"unchecked"`` (a host was not heard
    in time, so only the schedule speaks), ``"refused"`` (the fetch
    itself refuses this start and length; ``why`` is its sentence), or
    ``"skipped"`` (older than
    :data:`PUBLICATION_FRONTIER_HOURS`, or no public object to ask).
    """

    session = session or ProbeSession()
    if not start_needs_probe(document, cycle, now=now):
        return {"state": "skipped"}
    selected = parse_cycle(cycle)
    now = _utc(now)
    last_hour = int(document["last_hour"])
    try:
        rungs = _rung_urls(document["source_id"], selected, last_hour, now)
    except ValueError as error:
        # The fetch itself refuses this start and length (a lead this start's file ladder does not publish), so
        # no server needs asking and "the server did not answer" would be the wrong reason.
        return {"state": "refused", "why": str(error)}
    except (RuntimeError, KeyError) as error:
        return {"state": "unchecked", "why": str(error)}
    session.prefetch([url for _, urls in rungs for url in urls])
    before = session.misses
    for _, urls in rungs:
        if urls and all(session(url) for url in urls):
            return {"state": "published"}
    if session.misses > before:
        return {"state": "unchecked"}
    return {"state": "not-published"}


def _answered_missing(source: str, cycles, last_hour: int, now: datetime, session: ProbeSession) -> list[str]:
    """The ``cycles`` every rung already answered incomplete for (one object answered absent on each); nothing new is
    asked.  The rungs and the rule are :func:`page_start_check`'s, the fetch's rule for an explicit start."""

    missing = []
    for cycle in cycles:
        try:
            rungs = _rung_urls(source, cycle, last_hour, now)
        except (RuntimeError, ValueError, KeyError):
            continue
        if rungs and all(urls and any(session.known(url) is False for url in urls) for _, urls in rungs):
            missing.append(_stamp(cycle))
    return missing


def page_latest(document: dict, *, now: datetime | None = None, session: ProbeSession | None = None) -> dict:
    """The newest start the fetch accepts, for a page: :func:`confirmed_latest` asked through a session.

    ``latest`` is only ever a start the download takes (``downloads``
    true), or, where no server is asked (an old estimate, or no public
    object to ask), the schedule's estimate with ``basis`` ``"schedule"``.
    When no start was found whole it is None; the schedule's estimate is
    never put in its place, since for a source whose schedule declares no
    publication delay (the probe decides there) that estimate is the cycle
    still being made.

    ``basis`` is ``"checked"`` when every host asked answered, and
    ``"unchecked"`` when one was not heard in time.  ``missing`` names the
    newer starts the check found not yet whole: every one newer than
    ``latest`` when every host answered, and otherwise those each rung
    already answered incomplete for.  :func:`opening_start` reads both.
    When a host went unheard, a newer start one rung already answered
    complete wins over the resolver's walk.
    """

    from gpuwm.fetch import analysis_window_reference, require_cycle_grid

    session = session or ProbeSession()
    candidate = document.get("latest_candidate")
    if not latest_needs_probe(document, now=now):
        return {"latest": candidate, "basis": "schedule", "downloads": False, "missing": []}
    now = _utc(now)
    source, last_hour = document["source_id"], int(document["last_hour"])
    walk: list = []
    cycles: list = []
    try:
        grid = require_cycle_grid(source)
        reference = analysis_window_reference(source, grid, last_hour, now)
        walk = [cycle for cycle in grid.candidates(reference)
                if grid.horizon(cycle) is None or last_hour <= grid.horizon(cycle)]
        cycles = walk[:PAGE_LATEST_PREFETCH]
        # One start at a time, newest first, each start's objects asked side by side: the first start one rung
        # answers complete ends the asking, since the walk below stops there or on a newer one it asks itself.
        # Asking the three newest starts' objects at once sent a GEM row about 400 HEADs where 174 answer it.
        for cycle in cycles:
            if session.expired() or any(session.complete(rung) for _, rung in
                                        _rung_urls(source, cycle, last_hour, reference, cycles[0])):
                break
    except (RuntimeError, ValueError, KeyError):
        pass
    from gpuwm.fetch import resolve_latest_cycle

    before = session.misses
    try:
        # The resolver --cycle latest runs, as confirmed_latest asks it.  A start it returns had every object
        # answered present on one rung, so it is one the download accepts even when a newer start went unheard.
        cycle = resolve_latest_cycle(source, last_hour, now=now, probe=session)
    except (RuntimeError, ValueError):
        cycle = None
    unheard = session.misses > before
    if unheard:
        # The resolver walks the operational server's starts first, newest to oldest, and asks the archive only
        # when that server yields none.  Under the page's short NOMADS budget, with a forecast downloading on the
        # same machine, the newer starts' HEADs can go unheard while a later HEAD gets a turn, stopping the walk on
        # an older start (up to a day old).  A newer start the archive already answered complete is published, since
        # the archive never runs ahead of the operational server, and the download takes it: it wins, and
        # nothing new is asked for it.
        held = _answered_complete(source, cycles, last_hour, now, session)
        if held is not None and (cycle is None or held > cycle):
            cycle = held
    found = None if cycle is None else min(cycle, parse_cycle(candidate))
    newer = [start for start in walk if found is None or start > found]
    # A host not heard was asked about a start newer than the one found (or no start was found): said as unchecked,
    # so it is asked again.  Every host answered: each newer start was answered not whole.  Otherwise only the ones
    # every rung already answered incomplete for are known missing.
    unchecked = unheard and (found is None or bool(newer))
    missing = ([_stamp(start) for start in newer] if not unchecked
               else _answered_missing(source, newer, last_hour, now, session))
    return {"latest": _stamp(found), "basis": "unchecked" if unchecked else "checked",
            "downloads": found is not None, "missing": missing}


def resolve_latest(source: str, hours: float, *, now: datetime | None = None,
                   probe=None, cadence: int | None = None) -> dict:
    """Resolve through the SAME acquisition resolver used by the ordinary CLI."""
    from gpuwm.fetch import resolve_latest_cycle, _head_ok
    now = _utc(now)
    document = availability(source, hours, now=now, cadence=cadence)
    if not document["latest_supported"]:
        raise ValueError("No declared cycle covers this period. Choose a date manually or shorten the duration.")
    checked = []

    def record_probe(url):
        answer = (probe or _head_ok)(url)
        checked.append({"url": url, "available": bool(answer)})
        return answer

    # The resolver owns the analysis back-off (gpuwm.fetch
    # analysis_window_reference); applying it here as well subtracted the
    # window twice, and the cycle selected for a 240-hour era5 request
    # landed ten days before this document's own latest_candidate.
    cycle = resolve_latest_cycle(document["source_id"], document["last_hour"],
                                 now=now, probe=record_probe, cadence=cadence)
    validate_cycle(document, _stamp(cycle))
    document["selected_cycle"] = _stamp(cycle)
    document["resolution"] = {
        "basis": "provider_object_probe" if document["probeable"] else "estimated_publication_schedule",
        "checked_utc": datetime.now(timezone.utc).isoformat(), "objects": checked,
    }
    return document


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--hours", type=float)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--transport")
    parser.add_argument("--resolve-latest", action="store_true")
    args = parser.parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            source, hours, transport = _context(args.source, args.hours, args.config, args.transport)
            cadence = _config_cadence(args.config)
            if args.resolve_latest:
                # The ordinary resolver chooses across its own endpoint ladder.
                # An explicitly pinned transport cannot borrow a different one.
                if transport and transport != "auto":
                    raise ValueError("Latest probing uses automatic endpoint selection. Keep the configured transport and choose an explicit date.")
                document = resolve_latest(source, hours, cadence=cadence)
            else:
                document = availability(source, hours, transport=transport, cadence=cadence)
        print(json.dumps(document, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
