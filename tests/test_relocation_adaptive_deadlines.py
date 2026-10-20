"""A varying root step must not skip the moving nest's exact deadlines."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gpuwm.core.adaptive_clock import AdaptiveClockDriver
from gpuwm.core.clock import build_schedule, execute_schedule, resolve_clock
from gpuwm.core.relocation_runner import RelocationRunner, RelocationRunnerCollection
from gpuwm.experiment import RelocationConfig, ScheduledRelocationMove
from test_adaptive_history_timing import _experiment, _model


def _case():
    exp = _experiment()
    exp = replace(exp, run_seconds=3600., restart_interval_s=0., domains=tuple(
        replace(dc, history_interval_s=3600., run=replace(dc.run,
            starting_time_step=24 if dc.parent_id == 0 else 8,
            max_time_step=24 if dc.parent_id == 0 else 8))
        for dc in exp.domains))
    clock = resolve_clock(exp)
    schedule = build_schedule(exp, clock)
    return clock, schedule, _model(exp, clock)


def _runner(schedule, cadence=900., moves=()):
    return RelocationRunner(config=RelocationConfig(enabled=True, grid_id=2,
        cadence_seconds=cadence, moves=moves or ((ScheduledRelocationMove(7200., di_parent_cells=1),) if cadence else ())), schedule=schedule,
        provider=lambda *args: None, on_child_built=lambda *args: None)


@pytest.mark.parametrize("respect_deadlines", [False, True])
def test_adaptive_root_lands_on_every_fifteen_minute_consultation(respect_deadlines):
    clock, schedule, model = _case()
    runner = _runner(schedule)
    driver = AdaptiveClockDriver(model, cfl_source=lambda gid: (0., 0.),
        tick_den=clock.tick_den,
        deadline_source=runner.next_deadline_ticks if respect_deadlines else None)
    boundaries = []
    execute_schedule(schedule, clocks={gid: model.node(gid).clock for gid in (1, 2)},
        on_period_steps=driver,
        on_period_begin=lambda period, clocks: boundaries.append(clocks[1].ticks))
    expected = {s * clock.tick_den for s in (900, 1800, 2700)}
    if respect_deadlines:
        assert expected <= set(boundaries)
    else:
        assert not expected <= set(boundaries)  # witnessed production defect


def test_scripted_and_track_deadlines_survive_an_absent_mover_cadence():
    clock, schedule, model = _case()
    clocks = {gid: model.node(gid).clock for gid in (1, 2)}
    runner = _runner(schedule, cadence=None)
    assert runner.next_deadline_ticks(clocks) is None
    runner.track_writer = SimpleNamespace(stream=SimpleNamespace(
        config=SimpleNamespace(interval_seconds=600.)))
    assert runner.next_deadline_ticks(clocks) == 600 * clock.tick_den
    scripted = _runner(schedule, cadence=None,
        moves=(ScheduledRelocationMove(300., di_parent_cells=1),))
    assert scripted.next_deadline_ticks(clocks) == 300 * clock.tick_den
    collection = RelocationRunnerCollection((runner,))
    assert collection.next_deadline_ticks(clocks) == 600 * clock.tick_den
    clocks[1].ticks = clocks[2].ticks = 300 * clock.tick_den
    assert scripted.next_deadline_ticks(clocks) is None
    assert runner.next_deadline_ticks(clocks) == 600 * clock.tick_den
