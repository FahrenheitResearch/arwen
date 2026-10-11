"""What a stage receipt's ``timing_seconds`` block has to add up to.

Every long stage in this tree publishes a ``timing_seconds`` mapping in
its receipt: a handful of named children and a ``total``.  The block is
only worth reading if the children account for the total -- a receipt
whose named work is 60% of its own wall clock tells a reader where 60%
of the time went and lies by omission about the rest.

MEASURED, 2026-08-16, which is why this module exists: the GFS prepare
stage at the documented normal size took 36.0 s and its proof.json
attributed decode 1.8 + init 9.9 + cache-write 1.1 + export 7.9 =
20.7 s.  The missing 14.1 s -- 40%, the single largest item -- was the
static-field build, timed only behind an opt-in environment variable.
Under this project's fixed-means-default law that is not instrumentation
at all, and the number a reader needed most was the one they could not
get without knowing to ask for it.

The rule this module states is deliberately loose in one direction and
strict in the other.  Children may overlap or nest slightly and may miss
a few milliseconds of bookkeeping between phases, so the bar is a
FRACTION rather than equality; but a child that is not there at all
cannot be recovered by any reader, so the bar exists.
"""

from __future__ import annotations

from typing import Any, Mapping

import contextlib
import functools
from contextvars import ContextVar
import json
import os
from pathlib import Path
import threading
import time

from gpuwm.timeline.ledger import Ledger, span
from gpuwm.timeline.taxonomy import GO_STAGES

_observers = ContextVar("time_ledger_observers", default=None)
_entry_clock = ContextVar("time_ledger_entry_clock", default=None)
_bound_ledger = ContextVar("time_ledger_bound_ledger", default=None)
_first_binding = True


def register_ledger_observer(observer):
    owners = _observers.get()
    if owners is not None:
        owners.append(observer)


def current_ledger(default):
    return _bound_ledger.get() or default


def end_admission(ledger):
    if getattr(ledger, "_admission_open", False):
        ledger.emit("boot.engine_admission", "end")
        ledger._admission_open = False


@contextlib.contextmanager
def ledger_environment():
    """Restore ledger routing after a main-flow call; children inherit it."""
    keys = ("GPUWM_TIME_LEDGER", "GPUWM_RUN_ID", "GPUWM_PRODUCT")
    before = {key: os.environ.get(key) for key in keys}
    try:
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def ledger_main(function):
    """Scope default routing without changing any arguments or return value."""
    @functools.wraps(function)
    def call(*args, **kwargs):
        with ledger_environment():
            owned = []
            token = _observers.set(owned)
            clock_token = _entry_clock.set((time.monotonic(), time.time()))
            ledger_token = _bound_ledger.set(None)
            try:
                return function(*args, **kwargs)
            finally:
                try:
                    for observer in owned:
                        observer.close()
                    bound = _bound_ledger.get()
                    if bound is not None:
                        end_admission(bound)
                        if not owned:
                            bound.emit("finish", "mark")
                finally:
                    _observers.reset(token)
                    _entry_clock.reset(clock_token)
                    _bound_ledger.reset(ledger_token)
    return call


def bind_ledger(run, *, product=None):
    """Called after the existing run-root claim, before any stages spawn."""
    global _first_binding
    run = Path(run)
    if not os.environ.get("GPUWM_TIME_LEDGER"):
        os.environ["GPUWM_TIME_LEDGER"] = str((run / "timeline.jsonl").resolve())
    os.environ.setdefault("GPUWM_RUN_ID", run.name)
    if product is not None:
        os.environ.setdefault("GPUWM_PRODUCT", product)
    ledger = Ledger(run / "timeline.jsonl", run_id=os.environ["GPUWM_RUN_ID"])
    if _entry_clock.get() is not None:
        start_mono, start_unix = _entry_clock.get()
        if _first_binding:
            from gpuwm import LAUNCH_MONOTONIC, LAUNCH_UNIX_MS
            start_mono, start_unix = LAUNCH_MONOTONIC, LAUNCH_UNIX_MS / 1000.0
            _first_binding = False
        ledger.emit("boot.engine_admission", "begin", t_mono=start_mono, t_unix=start_unix)
        ledger._admission_open = True
        _bound_ledger.set(ledger)
    return ledger


def stage_timer(stage, *, ledger=None, attrs=None):
    """Existing orchestration timers can share the default-on C1 pair."""
    return span(stage, ledger=ledger, attrs=attrs)


class ForecastLedger:
    """Relay existing progress.jsonl metadata, never open forecast fields.

    The runner keeps its process isolation and exact command. Step boundaries
    use its existing per-domain timer and millisecond UTC publication clock.
    This is a retrospective relay, with its basis stated on every boundary.
    Missing/thinned/off progress is reported, never replaced with guessed work.
    """

    def __init__(self, ledger, run):
        self.ledger = ledger
        self.run = Path(run)
        self.stop = threading.Event()
        self.thread = None
        self.offset = 0
        self.first = False
        self.last = None
        self.pending = b""
        self.construct_open = False
        self.identity = None
        self.attempt = 1
        self.next_poll = 0.0

    def start(self, *, watch=True):
        self.ledger.emit("forecast.construct", "begin", attrs={"basis": "runner_startup"})
        self.construct_open = True
        if watch:
            self.thread = threading.Thread(target=self._watch, name="time-ledger", daemon=True)
            self.thread.start()

    def _watch(self):
        while not self.stop.wait(0.25):
            self.read()

    def poll(self):
        """Batch hosted callback metadata reads; final drains bypass this clock."""
        now = time.monotonic()
        if now >= self.next_poll:
            self.next_poll = now + 0.25
            self.read()

    def read(self):
        try:
            with (self.run / "progress.jsonl").open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                if self.identity is not None and (self.identity != identity or stat.st_size < self.offset):
                    self._finish_attempt()
                    self.offset = 0
                    self.pending = b""
                    self.first, self.last = False, None
                    self.attempt += 1
                    self.ledger.emit("forecast.construct", "begin", attrs={"attempt": self.attempt,
                        "basis": "progress_log_replaced", "coverage": "startup_boundary_unknown"})
                    self.construct_open = True
                self.identity = identity
                stream.seek(self.offset)
                raw = stream.read(1024 * 1024)
                self.offset += len(raw)
            lines = (self.pending + raw).split(b"\n")
            self.pending = lines.pop()
            for line in lines:
                try:
                    self.record(json.loads(line))
                except (ValueError, TypeError, KeyError):
                    continue
        except OSError:
            pass

    def record(self, record):
        unix = float(record["emitted_unix_ms"]) / 1000.0
        # Same-host clock conversion; readers sort timestamps, not file order.
        mono = time.monotonic() + unix - time.time()
        attrs = {"basis": "progress.jsonl.step_timer", "domain": record.get("domain"),
                 "attempt": self.attempt}
        if record["event"] == "step":
            duration = record.get("step_wall_seconds")
            if not isinstance(duration, (int, float)) or duration <= 0:
                self.ledger.emit("forecast.integrate", "mark", attrs={
                    "coverage": "unknown", "reason": "step timer absent in thinned progress"})
                return
            if not self.first:
                start_unix, start_mono = unix - duration, mono - duration
                self.ledger.emit("forecast.construct", "end", attrs=attrs,
                                 t_mono=start_mono, t_unix=start_unix)
                self.construct_open = False
                self.ledger.emit("forecast.integrate", "begin", attrs=attrs,
                                 t_mono=start_mono, t_unix=start_unix)
                self.ledger.emit("forecast.first_step", "begin", attrs=attrs,
                                 t_mono=start_mono, t_unix=start_unix)
                self.ledger.emit("forecast.first_step", "end", attrs=attrs,
                                 t_mono=mono, t_unix=unix)
                self.first = True
            self.last = (mono, unix, attrs)
        elif record["event"] in ("output_written", "restart_written"):
            stage = "forecast.output." + ("history" if record["event"] == "output_written" else "restart")
            duration = record.get("wall_seconds")
            if isinstance(duration, (int, float)) and duration > 0:
                self.ledger.emit(stage, "begin", attrs={**attrs, "basis": "progress.jsonl.output_latency"},
                                 t_mono=mono - duration, t_unix=unix - duration)
                self.ledger.emit(stage, "end", attrs={**attrs, "basis": "progress.jsonl.output_latency"},
                                 t_mono=mono, t_unix=unix)
            else:
                self.ledger.emit(stage, "mark", attrs={**attrs, "coverage": "duration_unknown"},
                                 t_mono=mono, t_unix=unix)

    def finish(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        # Drain the remaining metadata, including the last committed step.
        while True:
            before = self.offset
            self.read()
            if self.offset == before:
                break
        self._finish_attempt()

    def _finish_attempt(self):
        if self.last is not None:
            mono, unix, attrs = self.last
            self.ledger.emit("forecast.integrate", "end", attrs=attrs, t_mono=mono, t_unix=unix)
        if self.construct_open:
            self.ledger.emit("forecast.construct", "end", attrs={"coverage": "unknown",
                "reason": "no measured step in progress.jsonl"})
            self.construct_open = False


class LedgerObserver:
    """Add C1 to an existing go observer without changing hosting behavior."""

    def __init__(self, observer, ledger, run):
        self.observer, self.ledger, self.run = observer, ledger, run
        self.hosts_forecast = getattr(observer, "hosts_forecast", True)
        self.forecast = None
        self.open_stages = set()
        self.closed = False
        self.idle = True
        self.ledger.emit("post.orchestration", "begin")
        register_ledger_observer(self)

    def __getattr__(self, name):
        return getattr(self.observer, name)

    def _forward(self, name, **fields):
        hook = getattr(self.observer, name, None)
        if hook is not None:
            return hook(**fields)

    def stage_begin(self, *, label, command):
        end_admission(self.ledger)
        if self.idle:
            self.ledger.emit("post.orchestration", "end")
            self.idle = False
        stage = GO_STAGES.get(label, "prep." + label.replace(" ", "_"))
        self.ledger.emit(stage, "begin", attrs={"label": label})
        self.open_stages.add(label)
        if label == "forecast":
            self.forecast = ForecastLedger(self.ledger, self.run)
            self.forecast.start()
        return self._forward("stage_begin", label=label, command=command)

    def stage_end(self, *, label, **fields):
        self._end(label, fields)
        return self._forward("stage_end", label=label, **fields)

    def stage_secondary_end(self, *, label, **fields):
        self._end(label, fields)
        return self._forward("stage_secondary_end", label=label, **fields)

    def _end(self, label, fields):
        if label == "forecast" and self.forecast is not None:
            self.forecast.finish()
            self.forecast = None
        if label in self.open_stages:
            self.ledger.emit(GO_STAGES.get(label, "prep." + label.replace(" ", "_")), "end",
                             attrs={"label": label, "ok": fields.get("ok"),
                                    "exit_code": fields.get("exit_code")})
            self.open_stages.discard(label)
        if not self.open_stages and not self.idle:
            self.ledger.emit("post.orchestration", "begin")
            self.idle = True

    def __call__(self, **fields):
        return self.observer(**fields)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for label in list(self.open_stages):
            self._end(label, {"ok": False})
        if self.idle:
            self.ledger.emit("post.orchestration", "end")
            self.idle = False
        self.ledger.emit("finish", "mark")

#: The key every receipt's ``timing_seconds`` uses for its own wall.
TOTAL_KEY = "total"

#: The fraction of ``total`` the named children must account for.
#:
#: 0.9 leaves room for the process bookkeeping between phases -- receipt
#: assembly, digests, the atomic publish -- while refusing the case this
#: exists to catch, a whole phase that is simply not named.
MINIMUM_COVERAGE = 0.9


def timing_children(timing: Mapping[str, Any]) -> dict[str, float]:
    """The named children of one ``timing_seconds`` block.

    Everything except ``total`` and anything that is not a number: a
    receipt may carry a nested breakdown or a note beside its numbers,
    and a coverage check that choked on one would push call sites into
    keeping a second, parallel list of "the real keys".
    """

    return {
        str(key): float(value)
        for key, value in timing.items()
        if key != TOTAL_KEY and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def timing_coverage(timing: Mapping[str, Any]) -> float | None:
    """What fraction of ``total`` the named children account for.

    ``None`` when the question cannot be asked -- no ``total``, a
    non-numeric one, or a zero one (a stage that took no measurable time
    has nothing to attribute).  ``None`` is never a failure: a caller
    that cannot tell "uncovered" from "unaskable" would refuse receipts
    from the fastest runs.
    """

    total = timing.get(TOTAL_KEY)
    if not isinstance(total, (int, float)) or isinstance(total, bool):
        return None
    total = float(total)
    if total <= 0.0:
        return None
    return sum(timing_children(timing).values()) / total


def timing_coverage_shortfall(timing: Mapping[str, Any], *, what: str,
                              minimum: float = MINIMUM_COVERAGE
                              ) -> str | None:
    """Why this receipt does not account for itself, or ``None``.

    The sentence names the fraction, the seconds that are unattributed
    and the keys that ARE there, because the reader who meets it is
    about to go looking for the phase nobody timed and the existing
    keys are the map of where they have already looked.
    """

    coverage = timing_coverage(timing)
    if coverage is None or coverage >= minimum:
        return None
    total = float(timing[TOTAL_KEY])
    children = timing_children(timing)
    named = ", ".join(sorted(children)) or "(none)"
    return (
        f"{what} attributes {coverage * 100:.0f}% of its own "
        f"{total:.1f} s: {total - sum(children.values()):.1f} s is not "
        f"named by any key in timing_seconds, which carries {named}.  A "
        "receipt whose largest item is the unattributed remainder cannot "
        "answer the question it exists to answer.")


__all__ = [
    "MINIMUM_COVERAGE", "TOTAL_KEY", "timing_children", "timing_coverage",
    "timing_coverage_shortfall",
]
