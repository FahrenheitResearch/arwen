"""Strict invariant identity for a monotonically extended WRF namelist."""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import re
from pathlib import Path

from gpuwm.namelist_import import parse_namelist


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
_ASSIGNMENT = re.compile(
    r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?P<equal>\s*=\s*)(?P<value>.*?)(?P<comment>\s*!.*)?$"
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
    text = path.read_bytes().decode("utf-8")
    section = None
    seen: set[str] = set()
    output = []
    for raw in text.splitlines(keepends=True):
        body = raw.rstrip("\r\n")
        ending = raw[len(body):]
        stripped = body.strip()
        if stripped.startswith("&"):
            section = stripped[1:].strip().lower()
        elif stripped == "/":
            section = None
        match = _ASSIGNMENT.match(body)
        if match and section == "time_control":
            key = match.group("key").lower()
            if key in _TIME_WINDOW_KEYS:
                if key in seen:
                    raise ValueError(
                        f"sealed namelist repeats time_control.{key}"
                    )
                seen.add(key)
                body = (
                    f"{match.group('indent')}{match.group('key')}"
                    f"{match.group('equal')}<gpuwm-monotonic-window>"
                    f"{match.group('comment') or ''}"
                )
        output.append(body + ending)
    missing = sorted(set(_TIME_WINDOW_KEYS) - seen)
    if missing:
        raise ValueError(
            "sealed namelist does not carry simple time-window assignments: "
            + ", ".join(missing)
        )
    return "".join(output).encode("utf-8")


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
