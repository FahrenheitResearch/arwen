"""Env-gated wall-clock instrumentation for the un-timed subsystems.

``gpuwm/obs/``, ``gpuwm/verify/`` and ``gpuwm/io/`` carried no wall-clock
timing at all, so every cost attributed to them was reconstructed from
committed counts crossed with a throughput measured somewhere else.  That is
enough to rank a candidate and not enough to start a port.  This module is the
missing instrument, and it is deliberately the cheapest thing that can answer
"which stage, how long, how many bytes".

It is off unless ``GPUWM_PERF_TIMING`` is set to something other than an empty
string or ``0``.  While off, :func:`stage` hands back a shared do-nothing
object, so an annotated hot loop pays one module-global read and two method
calls per stage -- tens of nanoseconds against stages that are milliseconds at
their smallest.  Nothing here changes a number the model computes: the
instrument only reads the clock.

Reading the results:

* ``seconds`` is inclusive -- a stage's own work plus everything nested inside
  it.
* ``self_seconds`` subtracts the nested stages, which is what a ranking wants:
  it says where the wall clock actually sat.
* ``counters`` sum whatever the call site passed (``bytes=``, ``gates=``,
  ``sweeps=``), so a rate falls out of the receipt without a second run.

``GPUWM_PERF_TIMING_OUT=<path>`` writes the receipt at interpreter exit, which
is how a subprocess stage reports without its parent having to cooperate.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "gpuwm-perf-timing.v1"
ENV_ENABLE = "GPUWM_PERF_TIMING"
ENV_OUT = "GPUWM_PERF_TIMING_OUT"

__all__ = [
    "SCHEMA", "ENV_ENABLE", "ENV_OUT",
    "enabled", "reset", "stage", "phases", "record", "snapshot", "write",
]


def _truthy(raw: str | None) -> bool:
    return str(raw or "").strip().lower() not in ("", "0", "false", "no", "off")


class _NullStage:
    """The disabled path: enter, exit, ignore counters."""

    __slots__ = ()

    def __enter__(self) -> "_NullStage":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def count(self, **_counters: float) -> None:
        return None


class _NullPhases:
    """The disabled path for a straight-line sequence of phases."""

    __slots__ = ()

    def mark(self, _name: str, **_counters: float) -> None:
        return None

    def close(self, **_counters: float) -> None:
        return None

    def __enter__(self) -> "_NullPhases":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


_NULL = _NullStage()
_NULL_PHASES = _NullPhases()


class _Stage:
    """One timed region; nested regions are subtracted from the parent."""

    __slots__ = ("_name", "_counters", "_started", "_children")

    def __init__(self, name: str, counters: Mapping[str, float]) -> None:
        self._name = name
        self._counters = dict(counters)
        self._started = 0.0
        self._children = 0.0

    def count(self, **counters: float) -> None:
        """Add counter values discovered inside the region."""
        for key, value in counters.items():
            self._counters[key] = self._counters.get(key, 0.0) + float(value)

    def __enter__(self) -> "_Stage":
        _STACK.frames.append(self)
        self._started = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> bool:
        elapsed = time.perf_counter() - self._started
        frames = _STACK.frames
        if frames and frames[-1] is self:
            frames.pop()
        elif self in frames:
            # An exception escaped a nested region.  Drop everything above
            # this frame rather than leave the stack mis-nested for the rest
            # of the process; a wrong attribution is worse than a lost one.
            del frames[frames.index(self):]
        if frames:
            frames[-1]._children += elapsed
        _record(self._name, elapsed, elapsed - self._children, self._counters)
        return False


class _Stack(threading.local):
    def __init__(self) -> None:
        self.frames: list[_Stage] = []


_STACK = _Stack()
_LOCK = threading.Lock()
_TOTALS: dict[str, dict[str, Any]] = {}
_ENABLED = _truthy(os.environ.get(ENV_ENABLE))
_ATEXIT_REGISTERED = False


def _record(name: str, seconds: float, self_seconds: float,
            counters: Mapping[str, float]) -> None:
    with _LOCK:
        entry = _TOTALS.get(name)
        if entry is None:
            entry = {"calls": 0, "seconds": 0.0, "self_seconds": 0.0,
                     "counters": {}}
            _TOTALS[name] = entry
        entry["calls"] += 1
        entry["seconds"] += seconds
        entry["self_seconds"] += self_seconds
        if counters:
            bucket = entry["counters"]
            for key, value in counters.items():
                bucket[key] = bucket.get(key, 0.0) + float(value)


def enabled() -> bool:
    """Say whether timing is being collected in this process."""
    return _ENABLED


def reset(*, from_env: bool = True, on: bool | None = None) -> None:
    """Drop everything collected so far and re-read the environment.

    Tests set the variable and call this; a long-lived process can call it
    between phases so one phase's receipt does not carry the previous one's.
    """
    global _ENABLED
    with _LOCK:
        _TOTALS.clear()
    _STACK.frames = []
    if on is not None:
        _ENABLED = bool(on)
    elif from_env:
        _ENABLED = _truthy(os.environ.get(ENV_ENABLE))


def stage(name: str, **counters: float):
    """Time a named region, or hand back a no-op when timing is off."""
    if not _ENABLED:
        return _NULL
    _register_atexit()
    return _Stage(name, counters)


class _Phases:
    """A straight-line sequence of phases inside one function.

    ``with`` blocks would force a re-indent of code that is already written as
    numbered banner sections, and re-indenting a numerical routine to hang a
    stopwatch on it is how instrumentation introduces bugs.  ``mark`` closes
    whatever phase is open and starts the next, so a probe is one line at the
    boundary the code already draws.
    """

    __slots__ = ("_prefix", "_open")

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._open: _Stage | None = None

    def mark(self, name: str, **counters: float) -> None:
        """Close the open phase, if any, and start ``name``."""
        self.close()
        stage_ = _Stage(f"{self._prefix}.{name}", counters)
        stage_.__enter__()
        self._open = stage_

    def close(self, **counters: float) -> None:
        """Close the open phase, adding any last counters to it."""
        stage_ = self._open
        if stage_ is None:
            return
        if counters:
            stage_.count(**counters)
        self._open = None
        stage_.__exit__()

    def __enter__(self) -> "_Phases":
        return self

    def __exit__(self, *_exc: object) -> bool:
        self.close()
        return False


def phases(prefix: str):
    """Return a phase recorder under ``prefix``, or a no-op when off."""
    if not _ENABLED:
        return _NULL_PHASES
    _register_atexit()
    return _Phases(prefix)


def record(name: str, seconds: float, **counters: float) -> None:
    """Record a region timed by the caller (a subprocess, say)."""
    if not _ENABLED:
        return
    _register_atexit()
    _record(name, float(seconds), float(seconds), counters)


def snapshot() -> dict[str, Any]:
    """Return the receipt, stages ordered by descending self time."""
    with _LOCK:
        rows = []
        for name, entry in _TOTALS.items():
            row = {
                "stage": name,
                "calls": int(entry["calls"]),
                "seconds": round(float(entry["seconds"]), 6),
                "self_seconds": round(float(entry["self_seconds"]), 6),
            }
            if entry["counters"]:
                row["counters"] = {key: (int(value) if float(value).is_integer()
                                         else round(float(value), 6))
                                   for key, value in
                                   sorted(entry["counters"].items())}
            rows.append(row)
    rows.sort(key=lambda row: row["self_seconds"], reverse=True)
    return {
        "schema": SCHEMA,
        "enabled": _ENABLED,
        "stages": rows,
        "total_self_seconds": round(sum(row["self_seconds"] for row in rows), 6),
    }


def write(path: str | Path) -> Path:
    """Write the receipt as JSON and return where it landed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot(), indent=2) + "\n", encoding="utf-8")
    return path


def _register_atexit() -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    destination = os.environ.get(ENV_OUT)
    _ATEXIT_REGISTERED = True
    if not destination:
        return
    atexit.register(_write_at_exit, destination)


def _write_at_exit(destination: str) -> None:
    try:
        write(destination)
    except OSError:
        # A perf receipt is never worth failing a run over.
        pass
