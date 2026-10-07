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


def bind_streamed_clock(scalars, store, seconds):
    clocks = scalars.get("fire_clocks")
    if clocks is None:
        return
    old = float(clocks["time_seconds"])
    _check_clock(old,float(scalars["elapsed_seconds"]),int(clocks["step_count"]))
    if old != seconds:
        if clocks["step_count"]:
            _move_source_clock(store.get("state/chemdiag_sfire_source_lasttime"),old,seconds)
        clocks["time_seconds"] = float(seconds)
        if "fire_header" in scalars:
            scalars["fire_header"]["fire"]["grid"]["time_seconds"] = float(seconds)
