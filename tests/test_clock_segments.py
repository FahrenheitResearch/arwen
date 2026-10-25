"""Segment boundaries retain the ordinary integer schedule and alarm order."""
import pytest
from gpuwm.core.clock import build_schedule, execute_schedule
from test_clock import _chain_experiment


def collect(schedule, *, boundaries):
    clocks = schedule.clock.clocks()
    events = []
    callbacks = {}
    for name in ('step', 'force', 'feedback_prepare', 'feedback_commit',
                 'feedback_finalize', 'history', 'restart', 'lbc_reset',
                 'domain_start', 'period_begin', 'period_end', 'period_commit'):
        def record(*args, name=name):
            event = []
            for value in args:
                if isinstance(value, dict):
                    value = tuple((gid, c.ticks) for gid, c in value.items())
                elif hasattr(value, 'ticks'):
                    value = (value.ticks, value.step_count, value.dtbc_launch_fp32)
                event.append(value)
            events.append((name, tuple(event)))
        callbacks['on_' + name] = record
    start = 0
    for boundary in boundaries:
        execute_schedule(schedule, clocks=clocks, start_period=start,
            committed_initial_history_grid_ids=tuple(clocks) if start else (),
            stop_ticks=boundary, **callbacks)
        start = boundary // schedule.period_ticks
    return events, [(c.ticks, c.step_count, c.dtbc_launch_fp32) for c in clocks.values()]


@pytest.mark.parametrize('ratios', [(1,), (1, 3, 3), (1, 4, 3, 3)])
def test_segmented_schedule_equals_unbroken_including_output_and_feedback(ratios):
    exp = _chain_experiment(ratios, run_seconds=300, history_s=60)
    schedule = build_schedule(exp)
    stop = schedule.clock.run_ticks
    assert collect(schedule, boundaries=[stop]) == collect(schedule,
        boundaries=[stop//5, 2*stop//5, 3*stop//5, 4*stop//5, stop])


@pytest.mark.parametrize('stop,error', [(True, TypeError), (1.5, TypeError),
                                     (0, ValueError), (301, ValueError), (1, ValueError)])
def test_bad_segment_stops_refuse_before_forecast(stop, error):
    schedule = build_schedule(_chain_experiment((1,), run_seconds=300))
    with pytest.raises(error):
        execute_schedule(schedule, stop_ticks=stop,
            on_step=lambda *_: pytest.fail('invalid target stepped'))
