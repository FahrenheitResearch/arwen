"""The cycle clock is the one authority; these tests hold it to that.

Every burn this file guards against is a real one: a dt rounded to the
nearest tick is how a child drifts from its parent over a long cycle
with nothing reporting it, and an analysis time snapped silently onto
the step lattice is how a parent/child tree loses its analysis.
"""

from datetime import datetime, timedelta, timezone

import pytest

from gpuwm.cycle.clock import CycleClock
from gpuwm.cycle.contracts import (CLOCK_SCHEMA, CycleRefusal, TICK_HZ,
                                   TickRatio, seconds_to_ticks,
                                   ticks_to_seconds)

ANCHOR = datetime(2026, 8, 14, 18, 0, tzinfo=timezone.utc)


def _clock(**kwargs):
    args = dict(epoch_anchor=ANCHOR, parent_dt_seconds=120.0,
                cycle_seconds=960.0, n_cycles=3)
    args.update(kwargs)
    return CycleClock.build(**args)


def test_ticks_are_exact_for_common_dts():
    for seconds in (120.0, 2.5, 0.05, 30.0):
        ticks = seconds_to_ticks(seconds, label="dt")
        assert ticks == round(seconds * TICK_HZ)
        assert ticks_to_seconds(ticks) == seconds


def test_non_millisecond_dt_refuses_by_name():
    with pytest.raises(CycleRefusal) as excinfo:
        seconds_to_ticks(1.0 / 3.0, label="child dt")
    assert excinfo.value.observed["label"] == "child dt"
    assert excinfo.value.observed["seconds"] == pytest.approx(1.0 / 3.0)


def test_child_step_must_divide_parent():
    with pytest.raises(CycleRefusal) as excinfo:
        TickRatio(120000, 7000)
    assert excinfo.value.observed["remainder"] == 120000 % 7000


def test_require_on_lattice_refuses_off_lattice_analysis_time():
    clock = _clock()
    when = clock.valid_time(1) + timedelta(seconds=37)
    with pytest.raises(CycleRefusal) as excinfo:
        clock.require_on_lattice(when, label="analysis time")
    observed = excinfo.value.observed
    assert observed["label"] == "analysis time"
    assert observed["offset_seconds"] == pytest.approx(37.0)
    ticks = clock.require_on_lattice(when, label="analysis time",
                                     accept_offset_ticks=37000)
    assert ticks == clock.snap(when)[0]
    assert clock.snap(when)[1] == 37000


def test_boundary_lattice_and_schema():
    clock = _clock()
    assert clock.parent_step_ticks == 120000
    assert clock.cycle_ticks == 960000
    assert clock.steps_per_cycle() == 8
    assert [clock.boundary_ticks(i) for i in range(4)] == [
        0, 960000, 1920000, 2880000]
    assert clock.valid_time(2) == ANCHOR + timedelta(seconds=1920)
    payload = clock.to_json()
    assert payload["schema"] == CLOCK_SCHEMA
    assert payload["tick_hz"] == TICK_HZ
    assert payload["epoch_anchor"].startswith("2026-08-14T18:00:00")
    assert payload["n_cycles"] == 3


def test_boundary_index_out_of_range_names_both_ends():
    clock = _clock()
    with pytest.raises(CycleRefusal) as excinfo:
        clock.boundary_ticks(-1)
    assert excinfo.value.observed["cycle_index"] == -1
    with pytest.raises(CycleRefusal) as excinfo:
        clock.boundary_ticks(4)
    assert excinfo.value.observed["n_cycles"] == 3


def test_cycle_must_be_a_whole_number_of_parent_steps():
    # 900 s of cycle on a 120 s parent step is 7.5 steps: the combination
    # named in the build order is itself a refusal case, and stays one.
    with pytest.raises(CycleRefusal) as excinfo:
        _clock(cycle_seconds=900.0)
    assert excinfo.value.observed["remainder_ticks"] == 900000 % 120000


def test_child_ratio_builds_and_refuses():
    clock = _clock()
    assert clock.child_ratio(2.5).ratio == 48
    assert clock.child_ratio(30.0).ratio == 4
    with pytest.raises(CycleRefusal) as excinfo:
        clock.child_ratio(7.0)
    assert excinfo.value.observed["remainder"] == 120000 % 7000
