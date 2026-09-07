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

    def __post_init__(self) -> None:
        parse_cycle(self.start)
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


def _utc(now: datetime | None) -> datetime:
    result = now or datetime.now(timezone.utc)
    return (result.astimezone(timezone.utc).replace(tzinfo=None)
            if result.tzinfo is not None else result)


def _stamp(value: datetime | None) -> str | None:
    return None if value is None else value.strftime("%Y-%m-%dT%H")


def _reference(now: datetime, grid, last_hour: int, analysis: bool) -> datetime:
    if not analysis:
        return now
    reference = now - timedelta(hours=last_hour)
    if grid is not None and grid.record_end is not None:
        reference = min(reference, grid.record_end
                        + timedelta(hours=grid.delay_hours - last_hour))
    return reference


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


def availability(source: str, hours: float, *, now: datetime | None = None,
                 transport: str | None = None) -> dict:
    """Describe expected cycles and transport bounds without accessing a server."""
    from gpuwm.source_adapters import get_source_adapter
    from gpuwm.source_cycles import cycle_grid_for
    from gpuwm import fetch_endpoints
    from gpuwm.fetch import cycle_is_probeable

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
    last_hour = math.ceil(hours)
    reference = _reference(now, grid, last_hour, analysis)
    newest = grid.newest(reference) if grid else None
    allowed_hours = []
    if grid:
        for hour in (range(24) if hourly_start else grid.hours):
            horizon = grid.horizon(now.replace(hour=hour))
            if horizon is None and not analysis:
                horizon = adapter.max_forecast_hour
            if analysis or horizon is None or last_hour <= horizon:
                allowed_hours.append(hour)
        if newest is not None and allowed_hours:
            while newest.hour not in allowed_hours:
                newest = grid.snap(newest - timedelta(hours=1))
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
            "transport": endpoint.name,
            "record_start": None if window is None else window.start,
            "retention_hours": endpoint.retention_hours,
            "note": window.note if window else endpoint.why,
            "documentation": [] if window is None else list(window.documentation),
        })
    if not endpoints:
        rows = [{"transport": window.transport, "record_start": window.start,
                 "retention_hours": None, "note": window.note,
                 "documentation": list(window.documentation)}
                for window in adapter.archive_windows
                if not transport or transport == "auto" or window.transport == transport]
    starts = [parse_cycle(row["record_start"]) for row in rows if row["record_start"]]
    # Only a union with no unknown unbounded member has a known earliest date.
    # Rolling retention itself is advisory, so it never creates a hard cutoff.
    unknown = any(not row["record_start"] and row["retention_hours"] is None for row in rows)
    earliest = min(starts) if starts and not unknown else None
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
    if transport and transport != "auto":
        notes.append("A transport is explicitly selected. Choose an exact date; automatic Latest may use a different endpoint.")
    return {
        "schema": "gpuwm.source-availability.v1", "source_id": source,
        "display_name": adapter.display_title, "hours": hours, "last_hour": last_hour,
        "now_utc": _stamp(now), "earliest": _stamp(earliest),
        "latest_candidate": _stamp(newest), "cycle_hours": allowed_hours,
        "cycle_grid": None if grid is None else grid.declaration(),
        "analysis": analysis, "probeable": probeable,
        "latest_label": "Latest complete" if probeable else "Latest expected",
        "latest_supported": grid is not None and bool(allowed_hours) and (not transport or transport == "auto"),
        "transports": rows, "notes": notes,
        "requirements": [credential.display_name for credential in adapter.credentials],
    }


def validate_cycle(document: dict, cycle: str) -> tuple[str, list[str]]:
    selected = parse_cycle(cycle)
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
    return _stamp(selected), list(document["notes"])


def resolve_latest(source: str, hours: float, *, now: datetime | None = None,
                   probe=None) -> dict:
    """Resolve through the SAME acquisition resolver used by the ordinary CLI."""
    from gpuwm.fetch import resolve_latest_cycle, _head_ok
    now = _utc(now)
    document = availability(source, hours, now=now)
    if not document["latest_supported"]:
        raise ValueError("No declared cycle covers this period. Choose a date manually or shorten the duration.")
    checked = []

    def record_probe(url):
        answer = (probe or _head_ok)(url)
        checked.append({"url": url, "available": bool(answer)})
        return answer

    from gpuwm.source_cycles import cycle_grid_for
    reference = _reference(now, cycle_grid_for(document["source_id"]), document["last_hour"], document["analysis"])
    cycle = resolve_latest_cycle(document["source_id"], document["last_hour"],
                                 now=reference, probe=record_probe)
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
            if args.resolve_latest:
                # The ordinary resolver chooses across its own endpoint ladder.
                # An explicitly pinned transport cannot borrow a different one.
                if transport and transport != "auto":
                    raise ValueError("Latest probing uses automatic endpoint selection. Keep the configured transport and choose an explicit date.")
                document = resolve_latest(source, hours)
            else:
                document = availability(source, hours, transport=transport)
        print(json.dumps(document, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
