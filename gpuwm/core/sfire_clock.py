"""Bind fire continuation to the atmosphere's published clock image."""
from __future__ import annotations

import math


def _check_clock(old, prior_atmosphere, step_count):
    if not math.isfinite(old) or not math.isfinite(prior_atmosphere):
        raise ValueError("SFIRE continuation and atmosphere clocks must be finite")
    if old != prior_atmosphere and not (step_count == 0 and old == 0):
        raise ValueError("SFIRE clock differs from the atmosphere before authoritative tick publication")


def _move_source_clock(array, old, new):
    if array is None:
        return
    # A zero source clock before first emission is not a completed step.
    first = float(array[0,0].item())
    if first == old:
        array.fill(new)


def bind_state_clock(state, seconds):
    fire = getattr(getattr(state,"physics",None),"fire",None)
    if fire is None:
        return
    old = float(fire.grid.time_seconds)
    _check_clock(old,float(state.elapsed_seconds),int(fire.grid.step_count))
    source = getattr(state,"chemdiag_sfire_source_lasttime",None)
    if old != seconds:
        if fire.grid.step_count:
            _move_source_clock(source,old,seconds)
        fire.grid.time_seconds = float(seconds)


SOURCE_CLOCK_KEY = "state/chemdiag_sfire_source_lasttime"


def bind_streamed_clock(scalars, store, seconds):
    """Bind a streamed domain's fire clocks to the published tick image.

    ``store`` is the domain store, or a zero-argument callable returning it.
    The store is read only when a completed fire step's smoke-source clock
    must move with the fire clock.  A streamed domain passes the callable:
    reading ``RankedRun.store`` drains every slab to the host and re-gathers
    the whole store before the next sweep.  Breakage this prevents: 67e213588
    passed ``self.store`` eagerly from ``StreamedDomain.impose_clock``, which
    the executor calls before and after every step of every run, fire or not
    (720 full-state drains in 720 steps on M1; a 2-card run at 0.78x of one
    card on 2.8.7).
    """
    clocks = scalars.get("fire_clocks")
    if clocks is None:
        return
    old = float(clocks["time_seconds"])
    _check_clock(old,float(scalars["elapsed_seconds"]),int(clocks["step_count"]))
    if old != seconds:
        if clocks["step_count"]:
            joined = store() if callable(store) else store
            _move_source_clock(joined.get(SOURCE_CLOCK_KEY),old,seconds)
        clocks["time_seconds"] = float(seconds)
        if "fire_header" in scalars:
            scalars["fire_header"]["fire"]["grid"]["time_seconds"] = float(seconds)
