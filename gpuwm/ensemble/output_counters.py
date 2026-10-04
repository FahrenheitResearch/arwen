"""Exact precipitation-counter endpoints on the existing domain clock.

Counter observations do not add model alarms, shorten adaptive steps, write
frames or interpolate rainfall. Missing original-clock endpoints remain
explicitly unavailable. Streamed observations select two-dimensional carried
counters only, never a stale resident state or a full history projection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from operator import index
from collections.abc import Mapping

import numpy as np

COUNTER_CALENDAR_CONTRACT = "gpuwm-ensemble-counter-deadlines-v1"
_STORE_KEYS = {"RAINNC": "scratch/mp_rainnc", "RAINC": "scratch/cu_rainc"}
_COUNTERS = ("RAINNC", "RAINC", "RAINSH")


@dataclass
class _CounterProgress:
    last_observed: int | None = None
    captured: set = field(default_factory=set)
    unavailable: dict = field(default_factory=dict)


@dataclass
class CounterDeadlineCalendar:
    grid_id: int
    start_ticks: int
    run_ticks: int
    tick_den: int
    history_ticks: tuple
    windows_hours: tuple
    deadlines: tuple
    _progress: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_node(cls, node, *, windows_hours=(1, 3, 6)):
        return cls.from_clock(node.clock, windows_hours=windows_hours)

    @classmethod
    def from_valid_times(cls, *, grid_id, start_time, run_end_time, history_valid_times,
                         windows_hours=(1, 3, 6)):
        """Observe an existing output calendar without creating a model clock.

        Legacy fixed runners already spell their frame dates with datetime
        arithmetic. This constructor retains exactly those dates on the
        collector's integer-microsecond lattice. It does not schedule frames,
        round a timestep, or prescribe the runner's actual observation times.
        """
        windows_hours = tuple(windows_hours)
        windows = tuple(index(value) for value in windows_hours)
        if (any(isinstance(value, (bool, np.bool_)) for value in windows_hours)
                or any(value <= 0 for value in windows) or len(set(windows)) != len(windows)):
            raise ValueError("counter windows must be distinct positive integer hours")
        def tick(valid):
            delta = valid - start_time
            return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds
        stop = tick(run_end_time)
        history = tuple(tick(valid) for valid in history_valid_times)
        if stop < 0 or any(value < 0 or value > stop for value in history) or any(b <= a for a, b in zip(history, history[1:])):
            raise ValueError("existing history dates must increase within this domain's actual run interval")
        required = {0}
        for valid in history:
            required.add(valid)
            required.update(valid - hours * 3_600_000_000 for hours in windows
                            if valid >= hours * 3_600_000_000)
        return cls(index(grid_id), 0, stop, 1_000_000, history, windows, tuple(sorted(required)))

    @classmethod
    def from_clock(cls, clock, *, windows_hours=(1, 3, 6)):
        windows_hours = tuple(windows_hours)
        windows = tuple(index(value) for value in windows_hours)
        if any(isinstance(value, (bool, np.bool_)) for value in windows_hours) or any(value <= 0 for value in windows):
            raise ValueError("counter windows must be positive integer hours")
        if len(set(windows)) != len(windows):
            raise ValueError("counter windows must not repeat")
        spec = clock.spec
        start, stop, denominator = int(spec.start_ticks), int(clock.run_ticks), int(clock.tick_den)
        if denominator < 1 or spec.history_ticks < 1:
            raise ValueError("counter calendar requires the original positive tick and history intervals")
        first = start + int(spec.history_begin_ticks)
        last = stop if spec.history_end_ticks is None else min(stop, start + int(spec.history_end_ticks))
        history = tuple(range(first, last + 1, int(spec.history_ticks))) if first <= last else ()
        required = {start}
        for valid in history:
            required.add(valid)
            required.update(valid - hours * 3600 * denominator for hours in windows
                            if valid - hours * 3600 * denominator >= start)
        return cls(int(spec.grid_id), start, stop, denominator, history, windows,
                   tuple(sorted(tick for tick in required if tick <= stop)))

    def _member(self, member_id, episode):
        key = (index(member_id), index(episode))
        return self._progress.setdefault(key, _CounterProgress())

    def observe(self, ticks, *, member_id, episode=0):
        ticks = index(ticks)
        progress = self._member(member_id, episode)
        if progress.last_observed is not None and ticks < progress.last_observed:
            raise ValueError("counter observations must follow the original domain clock")
        missed = []
        for deadline in self.deadlines:
            if deadline >= ticks:
                break
            if deadline not in progress.captured and deadline not in progress.unavailable:
                reason = "the original domain clock was not observed at this exact counter endpoint"
                progress.unavailable[deadline] = reason
                missed.append({"ticks": deadline, "reason": reason})
        progress.last_observed = ticks
        due = ticks in self.deadlines and ticks not in progress.captured and ticks not in progress.unavailable
        return due, tuple(missed)

    def mark_captured(self, ticks, *, member_id, episode=0):
        self._member(member_id, episode).captured.add(index(ticks))

    def mark_unavailable(self, ticks, reason, *, member_id, episode=0):
        self._member(member_id, episode).unavailable[index(ticks)] = str(reason)

    def unavailable_endpoints(self, *, member_id, episode=0, through_ticks=None):
        """Unavailable baseline/window endpoints, including activation bounds."""
        progress = self._member(member_id, episode)
        through = (progress.last_observed if progress.last_observed is not None else self.start_ticks
                   ) if through_ticks is None else index(through_ticks)
        rows = []
        for valid in self.history_ticks:
            if valid > through:
                break
            for name, endpoint in (("rain_total", self.start_ticks), *(
                (f"qpf_{hours}h", valid - hours * 3600 * self.tick_den) for hours in self.windows_hours)):
                if endpoint < self.start_ticks:
                    reason = "the requested window starts before this domain's activation"
                elif endpoint in progress.unavailable:
                    reason = progress.unavailable[endpoint]
                elif endpoint <= through and endpoint not in progress.captured:
                    reason = "the required counter endpoint was not captured"
                else:
                    continue
                rows.append({"field": name, "valid_ticks": valid, "endpoint_ticks": endpoint, "reason": reason})
            if valid in progress.unavailable:
                rows.append({"field": "cumulative_rain", "valid_ticks": valid,
                             "endpoint_ticks": valid, "reason": progress.unavailable[valid]})
        return tuple(rows)

    def receipt(self, *, member_id, episode=0, through_ticks=None):
        progress = self._member(member_id, episode)
        return {"contract": COUNTER_CALENDAR_CONTRACT, "grid_id": self.grid_id,
                "member_id": index(member_id), "episode": index(episode),
                "domain_start_ticks": self.start_ticks, "tick_den": self.tick_den,
                "required_ticks": list(self.deadlines), "captured_ticks": sorted(progress.captured),
                "unavailable_ticks": [{"ticks": tick, "reason": reason}
                                      for tick, reason in sorted(progress.unavailable.items())],
                "unavailable_endpoints": list(self.unavailable_endpoints(member_id=member_id,
                    episode=episode, through_ticks=through_ticks)),
                "clock_policy": "observe original ticks without new alarms or interpolation"}


@dataclass(frozen=True)
class CounterCaptureResult:
    status: str
    ticks: int
    missed: tuple = ()
    reason: str | None = None
    fields: tuple = ()


def _surface_fields(values, cfg):
    fields = {}
    for name, value in values.items():
        if name not in _COUNTERS:
            raise ValueError("counter capture may borrow precipitation counters only")
        if value is None:
            continue
        if getattr(value, "shape", None) != (cfg.ny, cfg.nx) or getattr(value, "dtype", None) != np.dtype("float32"):
            raise ValueError(f"{name} must be the original float32 two-dimensional surface counter")
        fields[name] = value
    return fields


def _resident_counters(state, cfg):
    physics = getattr(state, "physics", None)
    zero = {"RAINSH"}  # Original physics_history_fields declares no shallow producer.
    fields = {}
    if not cfg.mp_physics:
        zero.add("RAINNC")
    else:
        value = getattr(getattr(physics, "microphysics", None), "rainnc", None)
        if value is None:
            return None, zero, "the active microphysics has no original precipitation accumulator"
        fields["RAINNC"] = value
    rainc = getattr(physics, "rainc", None)
    if rainc is not None:
        fields["RAINC"] = rainc
    elif not cfg.cu_physics:
        zero.add("RAINC")
    else:
        return None, zero, "the active cumulus has no original precipitation accumulator"
    return _surface_fields(fields, cfg), zero, None


def _streamed_counters(streamed, cfg):
    run = getattr(streamed, "_run", None)
    zero = {"RAINSH"}
    selected = {}
    for name, key in _STORE_KEYS.items():
        active = cfg.mp_physics if name == "RAINNC" else cfg.cu_physics
        if active:
            selected[name] = key
        else:
            zero.add(name)
    if not selected:
        return {}, zero, None
    ranked = bool(getattr(run, "ranked", False) or getattr(streamed, "ranked", False))
    if ranked:
        required = ("download", "pending_downloads", "wait_downloads")
        if run is None or any(not callable(getattr(run, name, None)) for name in required):
            return None, zero, "ranked streaming has no selected-counter download and completion API"
        store = getattr(run, "raw_store", None)
        if not isinstance(store, Mapping):
            return None, zero, "ranked streaming has no authoritative selected-counter host store"
        if any(key not in store for key in selected.values()):
            return None, zero, "the streamed store lacks an active precipitation counter"
        # Only metadata is read before named downloads land.
        _surface_fields({name: store[key] for name, key in selected.items()}, cfg)
        if selected:
            run.download(tuple(selected.values()))
            pending = tuple(run.pending_downloads())
            if any(not isinstance(item, Mapping) or "names" not in item for item in pending):
                return None, zero, "ranked selected downloads have no named completion receipts"
            named = tuple(item for item in pending if set(item["names"]) & set(selected.values()))
            run.wait_downloads(named)
    else:
        drain = getattr(run, "drain", None)
        if callable(drain):
            drain()
        store = getattr(streamed, "store", None)
        if not isinstance(store, Mapping):
            return None, zero, "streaming has no authoritative two-dimensional counter store"
        if any(key not in store for key in selected.values()):
            return None, zero, "the streamed store lacks an active precipitation counter"
    return _surface_fields({name: store[key] for name, key in selected.items()}, cfg), zero, None


def capture_due_counter(node, collector, member_id, *, calendar, start_time, metadata,
                        streamed=None, episode=0, counter_fields=None):
    """Observe one initialized/advanced original clock without writing output.

    Hooks belong after initialization and after the original clock advances,
    before any next step. Native callers can supply explicit borrowed member
    stripe counters. The collector consumes fields synchronously before any
    store or driver can overwrite them.
    """
    clock, cfg = node.clock, node.cfg.run
    ticks = int(clock.ticks)
    if int(node.cfg.grid_id) != calendar.grid_id or int(clock.tick_den) != calendar.tick_den:
        raise ValueError("counter calendar differs from its actual domain clock")
    if not getattr(node, "_started", True) or ticks < calendar.start_ticks:
        return CounterCaptureResult("dormant", ticks)
    due, missed = calendar.observe(ticks, member_id=member_id, episode=episode)
    if not due:
        return CounterCaptureResult("not_due", ticks, missed)
    if counter_fields is not None:
        fields = _surface_fields(counter_fields, cfg)
        zeros = {"RAINSH"}
        if not cfg.mp_physics:
            zeros.add("RAINNC")
        if not cfg.cu_physics:
            zeros.add("RAINC")
        reason = (None if set(_COUNTERS) <= fields.keys() | zeros
                  else "the explicit counter view lacks an active precipitation accumulator")
    elif streamed is not None:
        fields, zeros, reason = _streamed_counters(streamed, cfg)
    else:
        fields, zeros, reason = _resident_counters(node.state, cfg)
    capture = getattr(collector, "capture_rain_counters", None)
    if reason is None and not callable(capture):
        reason = "the product collector has no independent precipitation-counter capture hook"
    if reason is not None:
        calendar.mark_unavailable(ticks, reason, member_id=member_id, episode=episode)
        return CounterCaptureResult("unavailable", ticks, missed, reason)
    valid_time = start_time + timedelta(seconds=clock.elapsed_seconds)
    capture(fields, valid_time=valid_time, grid_id=calendar.grid_id, episode=index(episode),
        member_id=index(member_id), latitude=metadata["XLAT"], longitude=metadata["XLONG"],
        absent_zero_fields=tuple(sorted(zeros - fields.keys())))
    calendar.mark_captured(ticks, member_id=member_id, episode=episode)
    return CounterCaptureResult("captured", ticks, missed, fields=tuple(sorted(fields)))


__all__ = ["COUNTER_CALENDAR_CONTRACT", "CounterDeadlineCalendar", "CounterCaptureResult", "capture_due_counter"]
