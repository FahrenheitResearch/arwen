"""Execute packing waves with at most one live forecast on each card.

Run control lives here too. A stop (Ctrl-C, or a SystemExit raised by a stop
signal) reaches the thread that waits for the wave, never the member threads,
so the wave owns one ``MemberRunControl`` and every member observes it at its
next step boundary through its progress callback.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
import sys
import threading

#: How long the waiting thread sleeps between looks at its members. A bounded
#: wait is what lets the interpreter deliver an interrupt on every platform:
#: an untimed lock wait is not interruptible on Windows.
WAIT_POLL_SECONDS = 0.2


class MemberStopRequested(KeyboardInterrupt):
    """A running member ends at its step boundary because the run was stopped.

    It is a ``KeyboardInterrupt`` on purpose. Every forecast runner already
    treats that class as the stop the user asked for, and no ``except
    Exception`` handler between a step and the wave can swallow it.
    """


class MemberRunControl:
    """One stop request shared by every member of an ensemble run."""

    def __init__(self):
        self._stop = threading.Event()
        self._reason = None

    @property
    def stop_requested(self):
        return self._stop.is_set()

    @property
    def reason(self):
        return self._reason

    def request_stop(self, reason):
        if not self._stop.is_set():
            self._reason = str(reason)
            self._stop.set()

    def check(self):
        """Called by a member between steps; raises once a stop was asked."""
        if self._stop.is_set():
            raise MemberStopRequested(self._reason or "the ensemble run was stopped")


_CONTROL = ContextVar("gpuwm_ensemble_run_control", default=None)


def current_run_control():
    return _CONTROL.get()


@contextmanager
def member_run_scope(control):
    """Bind one control for every wave, nested or not, of one ensemble run."""
    token = _CONTROL.set(control)
    try:
        yield control
    finally:
        _CONTROL.reset(token)


def member_label(member_ids):
    ids = tuple(member_ids)
    return (f"member {ids[0]}" if len(ids) == 1 else
            "members " + ", ".join(str(member) for member in ids))


def name_failing_members(error, member_ids, *, device_id=None, wave=None, execution_mode=None):
    """Say which member an error belongs to without changing its type.

    Callers above the wave tell refusals apart by exception class (a memory
    refusal exits 2 with one sentence, a late source exits 75), so the error
    object is kept and the member is written into it: structured attributes,
    a traceback note, and the message itself when the message is the plain
    first argument. The first attribution wins, so a health gate that already
    named one member of a pack is not overwritten with the whole pack.
    """
    if getattr(error, "ensemble_member_ids", None) is not None:
        if getattr(error, "ensemble_device_id", None) is None:
            error.ensemble_device_id = device_id
        if getattr(error, "ensemble_wave", None) is None:
            error.ensemble_wave = wave
        return error
    ids = tuple(int(member) for member in member_ids)
    error.ensemble_member_ids = ids
    error.ensemble_device_id = device_id
    error.ensemble_wave = wave
    error.ensemble_execution_mode = execution_mode
    label = member_label(ids)
    where = "" if device_id is None else f" on card {device_id}"
    try:
        error.add_note(f"ensemble {label}{where}")
    except (AttributeError, TypeError):
        pass
    args = getattr(error, "args", ())
    if (len(args) == 1 and isinstance(args[0], str) and str(error) == args[0]
            and not args[0].startswith(("member ", "members "))):
        error.args = (f"{label}: {args[0]}",)
    return error


def failed_member_rows(error):
    """Every failing batch of the wave an error came from, as receipt rows."""
    rows = []
    for failure in getattr(error, "ensemble_wave_errors", None) or (error,):
        ids = getattr(failure, "ensemble_member_ids", None)
        if ids is None:
            continue
        rows.append({"member_ids": list(ids),
                     "device_id": getattr(failure, "ensemble_device_id", None),
                     "wave": getattr(failure, "ensemble_wave", None),
                     "execution_mode": getattr(failure, "ensemble_execution_mode", None),
                     "error_type": type(failure).__name__, "error": str(failure)})
    return rows


def _say(text):
    print(f"ensemble: {text}", file=sys.stderr, flush=True)


def _join_stopped(pending):
    """Wait for running members to reach the step boundary where they stop."""
    while pending:
        _, pending = wait(pending, timeout=WAIT_POLL_SECONDS, return_when=FIRST_COMPLETED)


def execute_concurrent_members(batch, execute_member, *, member_scope, control=None):
    """Run admitted ordinary models in independent member stream scopes.

    Every model retains its own original clock and nesting operations. The
    caller supplies the priced wave and owns all output. Results and errors
    keep roster order even when workers finish in another order.
    """
    from dataclasses import replace
    control = control or current_run_control() or MemberRunControl()
    jobs = []
    pool = ThreadPoolExecutor(max_workers=batch.members, thread_name_prefix="ensemble-member")
    def run(member):
        control.check()
        single = replace(batch, member_indices=(member,), execution_mode="ordinary_member",
                         required_bytes=batch.required_bytes // batch.members)
        try:
            with member_scope(member_id=member, device_id=batch.device_id) as owned:
                result = execute_member(single)
            return {"member_id": member, "result": result, "cuda_scope": owned.receipt()}
        except BaseException as error:
            name_failing_members(error, (member,), device_id=batch.device_id,
                                 wave=batch.wave, execution_mode="ordinary_member")
            raise
    try:
        for member in batch.member_indices:
            context = copy_context()
            jobs.append((member, pool.submit(context.run, run, member)))
        pending = {future for _, future in jobs}
        while pending:
            finished, pending = wait(pending, timeout=WAIT_POLL_SECONDS, return_when=FIRST_COMPLETED)
            for future in finished:
                stop = future.exception()
                if isinstance(stop, (KeyboardInterrupt, SystemExit)):
                    control.request_stop(f"{type(stop).__name__} during the ensemble forecast")
                    _join_stopped(pending)
                    raise stop
        errors = [future.exception() for _, future in jobs if future.exception() is not None]
        if errors:
            errors[0].ensemble_wave_errors = tuple(errors)
            raise errors[0]
        return {"status": "PASS", "backend": "ordinary_concurrent_members",
                "members": [future.result() for _, future in jobs]}
    except (KeyboardInterrupt, SystemExit) as stop:
        control.request_stop(f"{type(stop).__name__} during the ensemble forecast")
        _join_stopped({future for _, future in jobs if not future.cancel() and not future.done()})
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def execute_member_packing(plan, execute_batch, *, control=None):
    """Independent cards run in parallel; waves on one card run serially.

    ``execute_batch`` owns a device context and the admitted state. A member
    error propagates after the other owned jobs in that wave finish, names
    its members, and carries every other error of the wave. No failed state
    is retried, no member is dropped, and no same-card member processes exist.

    A ``KeyboardInterrupt`` or ``SystemExit`` in the waiting thread, or one
    raised by a member itself, stops the run: members not yet started are
    cancelled, running members are asked to stop at their next step
    boundary, the wave joins them, and the original stop is raised again.
    """
    if control is None:
        control = current_run_control() or MemberRunControl()
    results = []
    jobs = []
    pool = ThreadPoolExecutor(max_workers=len(plan.cards), thread_name_prefix="ensemble-card")
    try:
        for wave in range(plan.waves):
            control.check()
            jobs = [(batch, pool.submit(copy_context().run, execute_batch, batch))
                    for batch in plan.batches_in_wave(wave)]
            pending = {future for _, future in jobs}
            while pending:
                done, pending = wait(pending, timeout=WAIT_POLL_SECONDS,
                                     return_when=FIRST_COMPLETED)
                for future in done:
                    stop = future.exception()
                    if (isinstance(stop, (KeyboardInterrupt, SystemExit))
                            and not isinstance(stop, MemberStopRequested)):
                        raise stop
            wave_results, errors = [], []
            for batch, future in jobs:
                error = future.exception()
                if error is None:
                    wave_results.append((batch, future.result()))
                else:
                    errors.append(name_failing_members(error, batch.member_indices,
                        device_id=batch.device_id, wave=batch.wave,
                        execution_mode=batch.execution_mode))
            if errors:
                first = errors[0]
                first.ensemble_wave_errors = tuple(errors)
                for other in errors[1:]:
                    try:
                        first.add_note("also failed in this wave: "
                                       f"{type(other).__name__}: {other}")
                    except (AttributeError, TypeError):
                        pass
                raise first
            results.extend(wave_results)
            jobs = []
    except (KeyboardInterrupt, SystemExit) as stop:
        if isinstance(stop, MemberStopRequested):
            raise
        control.request_stop(f"{type(stop).__name__} during the ensemble forecast")
        running = set()
        for _, future in jobs:
            if not future.cancel() and not future.done():
                running.add(future)
        if running:
            _say(f"stopping: {len(running)} running member "
                 f"{'batch ends' if len(running) == 1 else 'batches end'} at the next "
                 "model step, and members not yet started are cancelled")
        _join_stopped(running)
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return tuple(results)


__all__ = ["WAIT_POLL_SECONDS", "MemberStopRequested", "MemberRunControl",
           "current_run_control", "member_run_scope", "member_label",
           "name_failing_members", "failed_member_rows", "execute_member_packing", "execute_concurrent_members"]
