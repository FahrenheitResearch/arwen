"""Public HRRR cycle-horizon and contiguous source-lead contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable


HRRR_EXTENDED_CYCLE_HOURS = frozenset({0, 6, 12, 18})
HRRR_STANDARD_HORIZON_HOURS = 18
HRRR_EXTENDED_HORIZON_HOURS = 48


def hrrr_cycle_horizon(cycle: datetime) -> int:
    """Return NOAA's public CONUS forecast horizon for one hourly cycle."""

    if not isinstance(cycle, datetime):
        raise TypeError("HRRR cycle must be a datetime")
    if cycle.minute or cycle.second or cycle.microsecond:
        raise ValueError("HRRR cycle must be an exact UTC hour")
    return (
        HRRR_EXTENDED_HORIZON_HOURS
        if cycle.hour in HRRR_EXTENDED_CYCLE_HOURS
        else HRRR_STANDARD_HORIZON_HOURS
    )


def validate_hrrr_source_forecast_hours(
        forecast_hours: Iterable[int], *, cycle: datetime | None = None,
        ) -> tuple[int, ...]:
    """Validate one inclusive, ordered, contiguous public source window.

    The returned leads retain NOAA's absolute cycle-relative identity.  Model
    forcing offsets are a separate ``0..len(hours)-1`` sequence and must not
    be substituted here.
    """

    hours = tuple(forecast_hours)
    if len(hours) < 2:
        raise ValueError("HRRR source window needs at least two hourly frames")
    if any(isinstance(hour, bool) or not isinstance(hour, int)
           for hour in hours):
        raise TypeError("HRRR forecast hours must be integers")
    if hours[0] < 0 or hours != tuple(range(hours[0], hours[-1] + 1)):
        raise ValueError(
            "HRRR source forecast hours must be contiguous, ordered, and "
            "nonnegative")
    horizon = (
        HRRR_EXTENDED_HORIZON_HOURS
        if cycle is None else hrrr_cycle_horizon(cycle))
    if hours[-1] > horizon:
        context = (
            "the absolute public maximum"
            if cycle is None else f"cycle {cycle:%Y-%m-%d %H}Z")
        raise ValueError(
            f"HRRR source window ends at f{hours[-1]:02d}, beyond "
            f"{context} horizon f{horizon:02d}")
    return hours


def hrrr_source_window(
        *, cycle: datetime, start_hour: int, run_seconds: float,
        end_hour: int | None = None) -> tuple[int, ...]:
    """Resolve source leads for a run and reject duration/window drift."""

    import math

    if (isinstance(start_hour, bool) or not isinstance(start_hour, int)
            or start_hour < 0):
        raise ValueError("forecast-start-hour must be a nonnegative integer")
    if (isinstance(run_seconds, bool) or not isinstance(run_seconds, (int, float))
            or not math.isfinite(float(run_seconds)) or run_seconds <= 0.0):
        raise ValueError("run-seconds must be finite and positive")
    required_end = start_hour + max(1, math.ceil(float(run_seconds) / 3600.0))
    if end_hour is not None:
        if isinstance(end_hour, bool) or not isinstance(end_hour, int):
            raise TypeError("forecast-end-hour must be an integer")
        if end_hour != required_end:
            raise ValueError(
                f"forecast-end-hour must be f{required_end:02d} for "
                f"run-seconds={run_seconds:g} beginning at f{start_hour:02d}")
    return validate_hrrr_source_forecast_hours(
        range(start_hour, required_end + 1), cycle=cycle)


#: WRF's own spelling for an instant, which every HRRR stage parses.
HRRR_TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"


def parse_hrrr_cycle(raw: str, *, flag: str = "--cycle") -> datetime:
    """Parse one exact hourly HRRR cycle in WRF's ``YYYY-MM-DD_HH:MM:SS``."""

    try:
        cycle = datetime.strptime(str(raw), HRRR_TIME_FORMAT)
    except ValueError as error:
        raise ValueError(
            f"{flag} must be YYYY-MM-DD_HH:MM:SS (UTC)") from error
    if cycle.minute or cycle.second or cycle.microsecond:
        raise ValueError(f"{flag} must be an exact hourly HRRR cycle")
    return cycle


def resolve_cycle_flags(cycle: str | None, valid_time: str | None, *,
                        tool: str, legacy_means: str,
                        warn=None) -> tuple[datetime, bool]:
    """Accept ``--cycle`` or the deprecated ``--valid-time``, and say so.

    ``--valid-time`` shipped on four HRRR entry points meaning two
    different instants: the CYCLE on the preparer and the single-domain
    benchmark, the MODEL START on the nested hierarchy.  At lead 0 those
    are the same instant, so nothing ever disagreed; at lead K one
    reading is wrong by K hours, and the wizard printed the same string
    to both.

    The resolution is that the typed time is always the cycle
    (``--cycle``) and model time zero is always derived from it and
    ``--forecast-start-hour``.  ``--valid-time`` is still accepted, with
    exactly the meaning it had in v1.4.0 on the tool it is passed to --
    that is what ``legacy_means`` records -- so a v1.4.0 script keeps
    working unchanged.  Passing both is refused rather than ranked.

    Returns ``(instant, came_from_valid_time)``.  A caller whose legacy
    meaning is the model start uses the flag to decide whether the
    instant it just parsed still needs the lead added to it.
    """

    if cycle is not None and valid_time is not None:
        raise ValueError(
            f"{tool}: pass --cycle or --valid-time, not both.  --cycle is "
            f"the HRRR cycle; --valid-time is its deprecated spelling and "
            f"means {legacy_means} on this command")
    if cycle is None and valid_time is None:
        raise ValueError(f"{tool}: --cycle is required")
    if cycle is not None:
        return parse_hrrr_cycle(cycle), False
    if warn is not None:
        warn(f"{tool}: --valid-time is deprecated and means "
             f"{legacy_means} here; --cycle CYCLE "
             "--forecast-start-hour K says the same thing on every stage "
             "of the HRRR chain, and is the only spelling that is "
             "unambiguous at a nonzero lead")
    return parse_hrrr_cycle(valid_time, flag="--valid-time"), True


__all__ = [
    "HRRR_EXTENDED_CYCLE_HOURS",
    "HRRR_EXTENDED_HORIZON_HOURS",
    "HRRR_STANDARD_HORIZON_HOURS",
    "HRRR_TIME_FORMAT",
    "hrrr_cycle_horizon",
    "hrrr_source_window",
    "parse_hrrr_cycle",
    "resolve_cycle_flags",
    "validate_hrrr_source_forecast_hours",
]
