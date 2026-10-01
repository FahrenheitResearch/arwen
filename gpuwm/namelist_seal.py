"""Strict invariant identity for a monotonically extended WRF namelist."""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
from pathlib import Path

from gpuwm.fortran_namelist import (
    parse_namelist, scan_namelist_text, value_end_through_blanks)


NAMELIST_EXTENSION_INVARIANT_SCHEMA = (
    "gpuwm-namelist-extension-invariant-v1"
)
_TIME_WINDOW_KEYS = (
    "run_days",
    "run_hours",
    "run_minutes",
    "run_seconds",
    "end_year",
    "end_month",
    "end_day",
    "end_hour",
    "end_minute",
    "end_second",
)


def validated_namelist_extension_invariant(
        value, *, context: str = "sealed namelist",
) -> dict[str, str]:
    """Return one canonical extension invariant or refuse malformed input."""

    digest = value.get("sha256") if isinstance(value, dict) else None
    if (not isinstance(value, dict)
            or set(value) != {"schema", "sha256"}
            or value.get("schema") != NAMELIST_EXTENSION_INVARIANT_SCHEMA
            or not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef"
                   for character in digest)):
        raise ValueError(f"{context} has no valid extension invariant")
    return {
        "schema": NAMELIST_EXTENSION_INVARIANT_SCHEMA,
        "sha256": digest,
    }


def _integer(value, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"sealed namelist {label} must contain integers")
    integer = int(value)
    if float(value) != float(integer):
        raise ValueError(f"sealed namelist {label} must contain integers")
    return integer


def _expanded(section, key: str, *, max_dom: int) -> list[int]:
    values = section.get(key)
    if not isinstance(values, list) or len(values) not in (1, max_dom):
        raise ValueError(
            f"sealed namelist time_control.{key} must contain one or "
            f"{max_dom} values"
        )
    integers = [
        _integer(value, label=f"time_control.{key}") for value in values
    ]
    if any(value < 0 for value in integers):
        raise ValueError(
            f"sealed namelist time_control.{key} must be nonnegative"
        )
    return integers * max_dom if len(integers) == 1 else integers


def _validate_window(path: Path, *, cycle: datetime, run_seconds: int) -> None:
    if cycle.tzinfo is not None:
        raise ValueError("sealed namelist cycle must be offset-free UTC")
    if isinstance(run_seconds, bool) or not isinstance(run_seconds, int) \
            or run_seconds <= 0:
        raise ValueError("sealed namelist run_seconds must be a positive integer")
    sections = parse_namelist(path)
    domains = sections.get("domains", {})
    max_dom_values = domains.get("max_dom")
    if not isinstance(max_dom_values, list) or len(max_dom_values) != 1:
        raise ValueError("sealed namelist domains.max_dom must be one integer")
    max_dom = _integer(max_dom_values[0], label="domains.max_dom")
    if max_dom <= 0:
        raise ValueError("sealed namelist domains.max_dom must be positive")
    time_control = sections.get("time_control")
    if not isinstance(time_control, dict):
        raise ValueError("sealed namelist lacks time_control")
    values = {
        key: _expanded(time_control, key, max_dom=max_dom)
        for key in _TIME_WINDOW_KEYS
    }
    for index in range(max_dom):
        duration = (
            values["run_days"][index] * 86_400
            + values["run_hours"][index] * 3_600
            + values["run_minutes"][index] * 60
            + values["run_seconds"][index]
        )
        if duration != run_seconds:
            raise ValueError(
                "sealed namelist run duration differs from the requested "
                f"{run_seconds} seconds"
            )
        try:
            observed_end = datetime(
                values["end_year"][index],
                values["end_month"][index],
                values["end_day"][index],
                values["end_hour"][index],
                values["end_minute"][index],
                values["end_second"][index],
            )
        except ValueError as error:
            raise ValueError(
                f"sealed namelist has an invalid end time: {error}"
            ) from None
        expected_end = cycle + timedelta(seconds=run_seconds)
        if observed_end != expected_end:
            raise ValueError(
                "sealed namelist end time differs from the requested "
                f"{expected_end.isoformat()}"
            )


def _normalized_bytes(path: Path) -> bytes:
    """The namelist's bytes with each time-window VALUE replaced.

    Spans come from :mod:`gpuwm.fortran_namelist`, the reader every
    namelist door shares, so only the value of each window key is
    normalized: a key packed beside it on one line (``run_hours = 6,
    history_interval = 60,``) stays in the identity, and a value
    continued over several lines is normalized whole.  The line matcher
    that stood here replaced the rest of the line after the first key,
    which took history_interval out of the identity, and refused the
    other window keys on that line as missing.
    """

    text = path.read_bytes().decode("utf-8")
    seen: set[str] = set()
    spans: list[tuple[int, int]] = []
    for group in scan_namelist_text(text, source=str(path)):
        if group.name != "time_control":
            continue
        for assignment in group.assignments:
            if assignment.name not in _TIME_WINDOW_KEYS:
                continue
            if assignment.name in seen:
                raise ValueError(
                    f"sealed namelist repeats time_control.{assignment.name}"
                )
            seen.add(assignment.name)
            spans.append((assignment.value_start,
                          value_end_through_blanks(text, assignment)))
    missing = sorted(set(_TIME_WINDOW_KEYS) - seen)
    if missing:
        raise ValueError(
            "sealed namelist does not carry simple time-window assignments: "
            + ", ".join(missing)
        )
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + "<gpuwm-monotonic-window>" + text[end:]
    return text.encode("utf-8")


def namelist_extension_invariant(
        path: str | Path, *, cycle: datetime, run_seconds: int,
) -> dict[str, str]:
    """Bind every byte except a validated, monotonic run/end window.

    The exact namelist digest remains part of every prepared-cache identity.
    This second identity is only the cross-generation invariant: before its
    values are normalized, the current run duration and end timestamp must
    agree exactly with the requested stream horizon.
    """

    source = Path(path)
    _validate_window(source, cycle=cycle, run_seconds=run_seconds)
    return {
        "schema": NAMELIST_EXTENSION_INVARIANT_SCHEMA,
        "sha256": hashlib.sha256(_normalized_bytes(source)).hexdigest(),
    }


__all__ = [
    "NAMELIST_EXTENSION_INVARIANT_SCHEMA",
    "namelist_extension_invariant",
    "validated_namelist_extension_invariant",
]
