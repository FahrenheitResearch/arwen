"""A nested tree lands on frequent output times without extra nest steps.

THE BREAKAGE THIS PREVENTS (WOOF run controls, one-minute output):
the same 3 -> 1 km forecast took 477 d02 steps an hour at hourly output and
552 at one-minute output. The root lands on every nest frame, and upstream's
halving of a remainder between one and two root steps hands the nest two
short parent steps, each divided into ceil(parent / nest) nest steps.
"""
from dataclasses import dataclass, replace
from fractions import Fraction

from gpuwm.core.adaptive_clock import AdaptiveClockDriver
from test_adaptive_clock_driver import (
    FakeRun, FakeCfg, FakeNode, FakeClock, FakeSpec, FakeModel)

TICK_DEN = 100


@dataclass
class SizedRun(FakeRun):
    nx: int = 100
    ny: int = 100
    nz: int = 49


def _tree(*, nest_history_s, root_history_s=3600, root_step=24, ratio=3):
    nest_step = Fraction(root_step, ratio)
    root_run = SizedRun(1, float(root_step), dx=3000.0, dy=3000.0,
                        nx=210, ny=204, min_time_step=1,
                        max_time_step=root_step, max_time_step_den=0)
    nest_run = SizedRun(2, float(nest_step), dx=1000.0, dy=1000.0,
                        nx=630, ny=612, min_time_step=1,
                        max_time_step=int(nest_step.numerator),
                        max_time_step_den=int(nest_step.denominator))
    root = FakeNode(FakeCfg(1, root_run, 1), FakeClock(FakeSpec(
        1, root_step * TICK_DEN, history_ticks=root_history_s * TICK_DEN)))
    nest = FakeNode(FakeCfg(2, nest_run, ratio), FakeClock(FakeSpec(
        2, int(nest_step * TICK_DEN), history_ticks=nest_history_s * TICK_DEN)),
        parent=root)
    root.children = [nest]
    return FakeModel(root)


def _hour(model, *, seconds=3600, upstream=False):
    """Drive the tree for ``seconds``; return (root steps, nest steps,
    nest frames landed, largest nest step in seconds). ``upstream`` keeps
    upstream's halves at every landing (the clock before the fix)."""
    # Low CFL: each controller sits at its max_time_step (24 s and 8 s), the
    # steady state of the measured run (d02 median step 7.5-8 s).
    drv = AdaptiveClockDriver(model, cfl_source=lambda gid: (0.1, 0.1),
                              tick_den=TICK_DEN)
    if upstream:
        drv._landing_split = lambda full, to_alarm, halved, props: (halved, True)
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    for clock in clocks.values():
        clock.run_ticks = seconds * TICK_DEN
    hist = clocks[2].spec.history_ticks
    root_steps = nest_steps = frames = 0
    widest = Fraction(0)
    period = 0
    while clocks[1].ticks < seconds * TICK_DEN:
        drv(period, clocks)
        parent, own = clocks[1].step_ticks, clocks[2].step_ticks
        assert parent % own == 0
        widest = max(widest, Fraction(own, TICK_DEN))
        start = clocks[1].ticks
        for k in range(1, parent // own + 1):
            if (start + k * own) % hist == 0:
                frames += 1
        nest_steps += parent // own
        root_steps += 1
        clocks[1].ticks += parent
        clocks[2].ticks = clocks[1].ticks
        period += 1
    return root_steps, nest_steps, frames, widest


def test_one_minute_nest_output_costs_no_extra_nest_steps():
    _, hourly, hourly_frames, _ = _hour(_tree(nest_history_s=3600))
    root, minute, frames, widest = _hour(_tree(nest_history_s=60))
    assert hourly_frames == 1 and frames == 60   # every frame still lands
    assert widest <= Fraction(8)                 # no nest step above its own
    assert hourly == 450                         # 3600 s / 8 s
    assert _hour(_tree(nest_history_s=60), upstream=True)[1] == 540
    # Upstream halving: 24 + 18 + 18 a minute -> 3 + 3 + 3 = 540 (+20 %).
    # Least-work landing: 24 + 24 + 12 -> 3 + 3 + 2 = 480, the fewest whole
    # 8 s steps that land on every minute (ceil(60 / 8) = 8 a minute).
    assert minute == 480, minute
    assert root == 180


def test_quarter_hour_output_lands_every_frame_and_never_costs_more():
    root, quarter, frames, widest = _hour(_tree(nest_history_s=900))
    before = _hour(_tree(nest_history_s=900), upstream=True)
    assert frames == before[2] == 4 and widest <= 8
    assert quarter == 452 and before[1] == 456


def test_a_finer_nest_also_takes_the_fewest_whole_steps():
    # A 6 s nest under a 24 s root: ceil(60 / 6) = 10 a minute.
    root, minute, frames, widest = _hour(_tree(nest_history_s=60, ratio=4))
    assert frames == 60 and widest <= 6 and minute == 600


def test_a_root_without_nests_keeps_upstream_halving():
    model = _tree(nest_history_s=60)
    model.root.children = []
    model = FakeModel(model.root)
    drv = AdaptiveClockDriver(model, cfl_source=lambda gid: (0.1, 0.1),
                              tick_den=TICK_DEN)
    drv._landing_split = lambda *a: (_ for _ in ()).throw(
        AssertionError("a root alone took the nested landing"))
    clock = model.node(1).clock
    clock.spec.history_ticks = 60 * TICK_DEN
    steps = []
    for period in range(4):
        drv(period, {1: clock})
        steps.append(clock.step_ticks)
        clock.ticks += clock.step_ticks
    assert steps[:3] == [2400, 1800, 1800]
