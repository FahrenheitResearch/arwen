"""Project original member progress onto one supervised ensemble workload."""
from __future__ import annotations

import math
from operator import index
import sys
import threading
import time
from datetime import datetime, timezone


def progress_host(observer):
    """The observer as a runner progress callback, or None when it is not one.

    A forecast runner calls its observer on every committed step. A stage
    observer that only records stage boundaries (it declares
    ``hosts_forecast = False``, or it is simply not callable) is not that
    callback. Breakage this prevents: handing ``gpuwm go``'s stage observer
    to the member runners ended every ensemble at its first forecast step
    with ``TypeError: 'GoChainEvents' object is not callable``.
    """
    if observer is None or not callable(observer):
        return None
    if not getattr(observer, "hosts_forecast", True):
        return None
    return observer


class MemberTerminalProgress:
    """Member progress in plain lines, for a run no supervising process hosts.

    One line when a member (or a native pack of members) starts, one when it
    finishes, and between them at most one line per ``interval_seconds``.
    The lines go to standard error so a runner's JSON summary on standard
    output stays the only thing there.
    """

    def __init__(self, member_ids, run_seconds, *, stream=None, interval_seconds=20.0,
                 clock=time.monotonic):
        self.member_ids = tuple(member_ids)
        self.run_seconds = float(run_seconds)
        self.interval_seconds = float(interval_seconds)
        self._stream, self._clock = stream, clock
        self._started, self._done = {}, set()
        self._last_line = None

    def _say(self, text):
        stream = sys.stderr if self._stream is None else self._stream
        try:
            print(f"ensemble: {text}", file=stream, flush=True)
        except (OSError, ValueError):
            pass

    def _label(self, ids):
        from gpuwm.ensemble.execution import member_label
        return member_label(ids)

    def report(self, ids, *, elapsed, step):
        """Called under the adapter's lock after one progress event."""
        now = self._clock()
        ids = tuple(ids)
        total = len(self.member_ids)
        fresh = [member for member in ids if member not in self._started]
        if fresh:
            for member in fresh:
                self._started[member] = now
            self._say(f"{self._label(ids)} started ({len(self._started)} of {total} "
                      f"{'member' if total == 1 else 'members'} started)")
            self._last_line = now
        finished = elapsed >= self.run_seconds
        if finished:
            newly = [member for member in ids if member not in self._done]
            if not newly:
                return
            self._done.update(newly)
            wall = now - min(self._started[member] for member in ids)
            self._say(f"{self._label(ids)} finished {self.run_seconds:.0f} model seconds in "
                      f"{wall:.0f} s ({len(self._done)} of {total} done)")
            self._last_line = now
            return
        if self._last_line is None or now - self._last_line >= self.interval_seconds:
            self._say(f"{self._label(ids)} at {elapsed:.0f} of {self.run_seconds:.0f} model "
                      f"seconds, step {step} ({len(self._done)} of {total} done)")
            self._last_line = now


class EnsembleProgressAdapter:
    """Only progress is combined. Original member clocks stay independent.

    Elapsed work is the mean of actual reported member forecast times and
    reported steps are their sum. A member checkpoint is recorded with its owner;
    it cannot serve as a restart of a multiple-member ensemble.

    ``control`` is the run's stop request. Every member reports here at each
    step, so this is the step boundary where a stopped run ends its members.
    ``terminal`` receives each member event for a run that has no progress
    host of its own.
    """
    def __init__(self, callback, *, member_ids, run_seconds, control=None, terminal=None,
                 check_products=None):
        ids = tuple(index(member) for member in member_ids)
        if not ids or len(set(ids)) != len(ids) or any(member < 0 for member in ids):
            raise ValueError("ensemble progress needs distinct nonnegative member IDs")
        duration = float(run_seconds)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("ensemble progress needs a positive finite forecast duration")
        self.callback, self.member_ids, self.run_seconds = progress_host(callback), ids, duration
        self.control, self.terminal = control, terminal
        self.check_products = check_products
        self._lock = threading.RLock()
        self._members = {member: {"elapsed_seconds": 0., "outer_step": 0,
            "last_checkpoint": None, "last_durable_wrfout": None, "attempt": 1,
            "last_forecast_step_at": None, "forecast_finished_at": None}
            for member in ids}

    def callback_for_member(self, member_id):
        member_id = index(member_id)
        if member_id not in self._members:
            raise ValueError("ensemble progress names a member outside the bound roster")
        return _MemberProgress(self, member_id)

    def restore_member(self, member_id, *, elapsed_seconds, outer_step=0, checkpoint=None):
        """Seed durable work before a resumed member reports its next step."""
        member_id, outer_step = index(member_id), index(outer_step)
        elapsed_seconds = float(elapsed_seconds)
        if (member_id not in self._members or not math.isfinite(elapsed_seconds)
                or not 0 <= elapsed_seconds <= self.run_seconds or outer_step < 0):
            raise ValueError("restored ensemble progress has an invalid original member clock")
        with self._lock:
            self._members[member_id].update(elapsed_seconds=elapsed_seconds,
                outer_step=outer_step, last_checkpoint=None if checkpoint is None else str(checkpoint))

    def check_stop(self):
        """Raise the member's stop at this boundary when the run was stopped."""
        if self.control is not None:
            self.control.check()
        if self.check_products is not None:
            self.check_products()

    def _submit(self, default_member, event):
        self.check_stop()
        event = dict(event)
        ids = tuple(index(member) for member in event.get("member_ids", (default_member,)))
        if not ids or len(set(ids)) != len(ids) or any(member not in self._members for member in ids):
            raise ValueError("native progress pack is outside the bound ensemble roster")
        elapsed, step = float(event["model_elapsed_seconds"]), index(event["outer_step"])
        if not math.isfinite(elapsed) or not 0 <= elapsed <= self.run_seconds or step < 0:
            raise ValueError("member progress has invalid forecast time or step")
        with self._lock:
            observed_at = datetime.now(timezone.utc).isoformat()
            for member in ids:
                previous = self._members[member]
                if elapsed < previous["elapsed_seconds"] or step < previous["outer_step"]:
                    raise ValueError("member progress moved backwards without a declared restart")
            for member in ids:
                row = self._members[member]
                if step > row["outer_step"] and (event.get("phase") == "post-d01-sync"
                        or event.get("backend") == "native_member_batched"):
                    row["last_forecast_step_at"] = observed_at
                    if elapsed == self.run_seconds:
                        row["forecast_finished_at"] = observed_at
                row.update(elapsed_seconds=elapsed, outer_step=step)
                for key in ("last_checkpoint", "last_durable_wrfout"):
                    if event.get(key) is not None:
                        row[key] = str(event[key])
            event["ensemble_member_ids"] = ids
            event["member_model_elapsed_seconds"] = elapsed
            event["member_outer_step"] = step
            event["model_elapsed_seconds"] = sum(
                self._members[member]["elapsed_seconds"] for member in self.member_ids) / len(self.member_ids)
            event["outer_step"] = sum(self._members[member]["outer_step"] for member in self.member_ids)
            event.setdefault("last_durable_wrfout", None)
            event.setdefault("last_checkpoint", None)
            if len(self.member_ids) > 1:
                event["last_checkpoint"] = None
            if self.terminal is not None:
                self.terminal.report(ids, elapsed=elapsed, step=step)
            if self.callback is not None:
                self.callback(**event)

    def receipt(self):
        with self._lock:
            return {"schema": "gpuwm-ensemble-progress.v1", "member_order": list(self.member_ids),
                "last_member_forecast_step_at": (max(row["forecast_finished_at"] for row in self._members.values())
                    if all(row["forecast_finished_at"] for row in self._members.values()) else None),
                "elapsed_policy": "mean of original member forecast times",
                "step_policy": "sum of original reported member steps",
                "checkpoint_policy": "member-owned; no partial ensemble checkpoint",
                "members": [{"member_id": member, **self._members[member]} for member in self.member_ids]}


class _MemberProgress:
    def __init__(self, owner, member_id):
        self.owner, self.member_id = owner, member_id

    def __call__(self, **event):
        return self.owner._submit(self.member_id, event)

    def restarting(self, reason):
        with self.owner._lock:
            row = self.owner._members[self.member_id]
            row.update(elapsed_seconds=0., outer_step=0, last_checkpoint=None,
                       last_durable_wrfout=None, attempt=row["attempt"] + 1,
                       last_forecast_step_at=None, forecast_finished_at=None)
            hook = getattr(self.owner.callback, "restarting", None)
            if hook is not None:
                hook(f"ensemble member {self.member_id}: {reason}")

    def complete(self, model_elapsed_seconds):
        # Finished member clocks are work progress. Only the enclosing run
        # may publish terminal success after its products are durable.
        with self.owner._lock:
            step = self.owner._members[self.member_id]["outer_step"]
            return self.owner._submit(self.member_id, {
                "model_elapsed_seconds": model_elapsed_seconds, "outer_step": step})

    def __getattr__(self, name):
        hook = getattr(self.owner.callback, name)
        if not callable(hook):
            return hook
        def forward(*args, **kwargs):
            with self.owner._lock:
                return hook(*args, **kwargs)
        return forward


__all__ = ["EnsembleProgressAdapter", "MemberTerminalProgress", "progress_host"]
