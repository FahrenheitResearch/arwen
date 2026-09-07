"""The schedule executor walking a VARYING step_ticks.

Nothing sets an adaptive dt in production yet.  What is gated here is the
executor's half of it: that ``execute_schedule`` can be driven by a period
source which re-expands one root step at a time at whatever step the
controller just chose, and that the invariants the fixed walk asserts --
parent/child tick-exact sync, the FORCE lead, boundary alignment -- still
hold when the step changes underneath it.

The arms are compared directly: with a controller that changes nothing,
the adaptive walk must produce EXACTLY the fixed walk's op sequence and
counts.  That equivalence is what makes the varying cases evidence about
the step rather than about the walk.
"""

from __future__ import annotations

import pytest

from gpuwm.core.clock import (build_schedule, execute_schedule,
                              resolve_clock)
from test_clock import _chain_experiment


def _fixture(tratios=(1, 3), run_seconds=1800.0, history_s=180.0):
    exp = _chain_experiment(tratios, run_seconds=run_seconds,
                            history_s=history_s)
    clock = resolve_clock(exp, lbc_interval_s=1800.0)
    return exp, build_schedule(exp, clock)


def _walk(schedule, on_period_steps=None):
    seen: list[tuple[str, int]] = []
    report = execute_schedule(
        schedule,
        on_step=lambda gid, dom: seen.append(("STEP", gid)),
        on_force=lambda gid, pid, c, p: seen.append(("FORCE", gid)),
        on_period_steps=on_period_steps)
    return report, seen


def _steps_on(seen, grid_id):
    return sum(1 for kind, gid in seen if kind == "STEP" and gid == grid_id)


# ------------------------------------------------------- the expander

def test_the_schedule_exposes_its_expander():
    """Single-sourced: the adaptive walk reuses the fixed walk's recursion.

    A second transcription of frame/module_integrate.F is how the two
    would drift, so the expander is exposed rather than reimplemented.
    """
    _, schedule = _fixture()
    assert schedule.expander is not None
    assert schedule.expander(0, None) == schedule.interior_period


def test_expander_at_the_configured_steps_reproduces_the_table():
    """Passing the configured steps explicitly must change nothing."""
    _, schedule = _fixture()
    steps = {spec.grid_id: spec.step_ticks
             for spec in schedule.clock.domains}
    assert schedule.expander(0, steps) == schedule.interior_period


# -------------------------------------------------- the equivalence

def test_an_inert_controller_reproduces_the_fixed_walk_exactly():
    """The equivalence every varying case below rests on.

    If this diverges, a difference in the varying cases says nothing
    about the varying step -- it could be the walk itself.
    """
    fixed_report, fixed_seen = _walk(_fixture()[1])
    adaptive_report, adaptive_seen = _walk(
        _fixture()[1], on_period_steps=lambda period, clocks: None)
    assert adaptive_seen == fixed_seen
    for field in ("steps", "forces", "feedback_calls", "restarts",
                  "lbc_resets"):
        assert getattr(adaptive_report, field) == getattr(
            fixed_report, field), field
    assert fixed_report.steps > 0, "an empty walk proves nothing"


# ------------------------------------------------------ varying steps

def test_halving_both_steps_costs_more_steps_and_still_lands_square():
    """The child is halved with the parent so it keeps dividing it exactly.

    That is WRF's ``num_small_steps = CEILING(parent%dt / dt)``
    requirement; the expander's tick-exact assertion is what enforces it.
    """
    def controller(period, clocks):
        if period >= 4:
            for dom in clocks.values():
                dom.step_ticks = dom.spec.step_ticks // 2

    report, seen = _walk(_fixture()[1], on_period_steps=controller)
    fixed_report, fixed_seen = _walk(_fixture()[1])
    assert report.steps > fixed_report.steps, (
        f"halving the step must cost MORE steps: {report.steps} vs "
        f"{fixed_report.steps}")
    assert _steps_on(seen, 1) > _steps_on(fixed_seen, 1)


def test_growing_the_root_alone_is_safe_because_the_child_absorbs_it():
    """Asserted the wrong way round first, which is why it is written down.

    Doubling the ROOT step without touching the child looks like a
    desynchronisation and is not: ``integrate`` runs the child
    ``while ticks[kid] < ticks[parent]``, so the child simply takes twice
    as many steps inside the longer parent step and still lands on the
    boundary.  That is WRF's nest contract -- the child derives its step
    COUNT from the parent, not the reverse -- and it means a controller
    may raise the parent without having to touch the child at all.
    """
    def grow_root(period, clocks):
        clocks[1].step_ticks = clocks[1].spec.step_ticks * 2

    _, seen = _walk(_fixture()[1], on_period_steps=grow_root)
    _, fixed_seen = _walk(_fixture()[1])
    assert _steps_on(seen, 1) == _steps_on(fixed_seen, 1) // 2
    assert _steps_on(seen, 2) == _steps_on(fixed_seen, 2)


# ---------------------------------------------------------- refusals

def test_a_child_step_that_does_not_divide_the_parent_is_REFUSED():
    """The safety property, stated as a failure rather than assumed.

    A child step that does not divide its parent's leaves the child off
    the period boundary.  WRF prevents this by construction; here the
    expander's own tick-exact assertion catches it, so a controller bug
    surfaces as a named error on the first period rather than as a
    silently desynchronised nest hours in.
    """
    def bad(period, clocks):
        for gid, dom in clocks.items():
            if gid != 1:
                dom.step_ticks = dom.spec.step_ticks - 1

    with pytest.raises(RuntimeError, match="tick-exact sync violated"):
        _walk(_fixture()[1], on_period_steps=bad)


def test_a_zero_step_does_not_hang_the_walk():
    """A controller that returns nonsense must fail, not spin forever.

    `while ticks[gid] < stop_subtime` with a zero step is an infinite
    loop that would look like a wedged run rather than a bug.
    """
    def zero(period, clocks):
        clocks[1].step_ticks = 0

    with pytest.raises((RuntimeError, ValueError, ZeroDivisionError)):
        _walk(_fixture()[1], on_period_steps=zero)
