"""``AdaptiveTimestepController`` -- adapt_timestep's sequence over time.

SCOPE OF EVIDENCE, kept explicit because it is not uniform across this
campaign.  ``calc_dt`` is graded bit-exact against a captured oracle
(``test_adaptive_timestep_oracle.py``, 1728 rows).  The SEQUENCE this file
covers is not: ``adapt_timestep`` USEs module_domain / module_configure /
module_dm / module_bc_em, so a standalone Fortran harness for it is a
different order of work.  These gates pin the transcription's ORDER and
its behaviour over a trajectory, which is weaker than the oracle and is
labelled so rather than blurred into it.

The behaviours worth pinning are the ones that only appear over a
SEQUENCE -- a single call cannot show a ratchet.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from gpuwm.core.adaptive_timestep import AdaptiveTimestepController


def _controller(**over):
    kw = dict(target_cfl=1.2, target_hcfl=0.84, max_step_increase_pct=5,
              starting_dt=Fraction(30))
    kw.update(over)
    return AdaptiveTimestepController(**kw)


def _run(ctl, cfls, *, horiz=0.2):
    """Drive a controller through a CFL series; return the dt taken each step."""
    taken = [ctl.first_step()]
    for cfl in cfls:
        dt = ctl.next_dt(max_vert_cfl=cfl, max_horiz_cfl=horiz)
        ctl.accept(dt, max_vert_cfl=cfl, max_horiz_cfl=horiz,
                   stepping_to_time=False)
        taken.append(dt)
    return taken


# ------------------------------------------------------------ direction

def test_a_cfl_over_target_shrinks_the_step():
    ctl = _controller()
    ctl.first_step()
    dt = ctl.next_dt(max_vert_cfl=2.4, max_horiz_cfl=0.2)
    assert dt < 30


def test_a_cfl_under_target_grows_the_step():
    ctl = _controller()
    ctl.first_step()
    dt = ctl.next_dt(max_vert_cfl=0.6, max_horiz_cfl=0.2)
    assert dt > 30


def test_the_growth_bound_binds_on_a_tiny_cfl():
    """calc_dt would multiply by 1.2/0.001; the bound holds it to +5%."""
    ctl = _controller(max_step_increase_pct=5)
    ctl.first_step()
    dt = ctl.next_dt(max_vert_cfl=0.001, max_horiz_cfl=0.001)
    assert dt == Fraction(315, 10), dt          # 30 * 1.05
    assert dt <= 30 * Fraction(105, 100)


def test_the_growth_bound_does_NOT_bind_on_the_very_first_step():
    """:174's `currentTime /= startTime` guard.

    Without it the first step is clamped against a last_dt that describes
    nothing, and the run starts by shrinking for no reason.
    """
    fresh = _controller()
    assert fresh.started is False
    first = fresh.first_step()
    assert first == 30


def test_the_more_restrictive_of_the_two_CFLs_wins():
    """Two calc_dt calls, min taken (:161-167)."""
    ctl = _controller()
    ctl.first_step()
    # horizontal far over ITS target, vertical comfortable
    dt = ctl.next_dt(max_vert_cfl=0.3, max_horiz_cfl=3.0)
    ctl_v = _controller()
    ctl_v.first_step()
    dt_v_only = ctl_v.next_dt(max_vert_cfl=0.3, max_horiz_cfl=0.2)
    assert dt < dt_v_only, "the horizontal constraint was ignored"


# ------------------------------------------------------------- clamps

def test_max_time_step_clamps_growth():
    ctl = _controller(max_dt=Fraction(31), max_step_increase_pct=100)
    ctl.first_step()
    dt = ctl.next_dt(max_vert_cfl=0.01, max_horiz_cfl=0.01)
    assert dt == 31


def test_min_time_step_clamps_reduction():
    ctl = _controller(min_dt=Fraction(29))
    ctl.first_step()
    dt = ctl.next_dt(max_vert_cfl=50.0, max_horiz_cfl=50.0)
    assert dt == 29


def test_every_step_lands_on_a_hundredth_of_a_second():
    """The property that keeps an integer-tick clock exact."""
    ctl = _controller()
    for dt in _run(ctl, [0.4, 0.9, 1.9, 0.05, 3.0, 1.2, 0.7]):
        assert (dt * 100).denominator == 1, dt


# ------------------------------------- behaviours that need a sequence

def test_stepping_to_a_time_does_NOT_ratchet_the_clock_down():
    """`use_last2` (:392-402), and the reason it exists.

    A step shortened to land on an output time must not become the
    baseline the next step grows from.  Without this the clock walks
    downward every time it hits a frame, and a run with frequent output
    slowly grinds to the minimum step for no physical reason.
    """
    ctl = _controller()
    ctl.first_step()
    dt = ctl.next_dt(max_vert_cfl=1.2, max_horiz_cfl=0.2)
    ctl.accept(dt, max_vert_cfl=1.2, max_horiz_cfl=0.2,
               stepping_to_time=False)
    baseline = ctl.last_dt

    # now take a deliberately shortened step to land on a frame
    short = Fraction(1, 10)
    ctl.accept(short, max_vert_cfl=1.2, max_horiz_cfl=0.2,
               stepping_to_time=True)
    assert ctl.last_dt == baseline, (
        f"last_dt fell to the shortened step {ctl.last_dt}; the next step "
        f"would grow from 0.1 s instead of {baseline} s")


def test_after_stepping_to_a_time_only_the_HORIZONTAL_cfl_reaches_back():
    """:154-159 intends to reach back; :405 defeats half of it.

    The intent is clear -- a step shortened to land on a frame has an
    artificially small CFL, so reusing it would over-grow the next step,
    and upstream reaches back to `last_max_*_cfl` instead.

    It only works for the HORIZONTAL.  The `use_last2` branch preserves
    both with no-op self-assignments, and then :405 assigns
    `grid%last_max_vert_cfl = grid%max_vert_cfl` UNCONDITIONALLY, two
    lines below the branch that just protected it.  So the vertical
    memory is overwritten every step and the protection never applies to
    it.

    This test asserts what the code DOES, not what it means to do.  A
    port that "fixed" the asymmetry would diverge from WRF on every run
    that steps to an output time -- which, with step_to_output_time
    defaulting true, is every adaptive run.  Written the other way round
    first, and it failed: the vertical arm produced dt=3 where the
    protected reading predicts 30+.
    """
    ctl = _controller()
    ctl.first_step()
    ctl.accept(Fraction(30), max_vert_cfl=0.6, max_horiz_cfl=0.2,
               stepping_to_time=False)
    ctl.accept(Fraction(1, 10), max_vert_cfl=99.0, max_horiz_cfl=99.0,
               stepping_to_time=True)

    # horizontal DID reach back to the remembered 0.2 ...
    assert ctl.last_max_horiz_cfl == 0.2
    # ... vertical did NOT: :405 clobbered it with the shortened step's.
    assert ctl.last_max_vert_cfl == 99.0

    # and the consequence: the vertical arm drives the next step down.
    dt = ctl.next_dt(max_vert_cfl=99.0, max_horiz_cfl=99.0)
    assert dt < 30, (
        f"dt={dt}: expected the clobbered vertical memory to shrink the "
        f"step, matching WRF")


def test_upstreams_asymmetric_cfl_memory_is_reproduced_not_tidied():
    """:392-405.  The vertical memory IS clobbered when stepping to a time.

    Upstream writes `last_max_vert_cfl = last_max_vert_cfl` (a no-op) in
    the use_last2 branch and then unconditionally assigns
    `last_max_vert_cfl = max_vert_cfl` two lines later, OUTSIDE it -- so
    vertical and horizontal behave differently.  A port that "fixed" the
    asymmetry would diverge from WRF, so it is pinned as-is.
    """
    ctl = _controller()
    ctl.first_step()
    ctl.accept(Fraction(30), max_vert_cfl=0.5, max_horiz_cfl=0.3,
               stepping_to_time=False)
    ctl.accept(Fraction(1), max_vert_cfl=7.0, max_horiz_cfl=8.0,
               stepping_to_time=True)
    assert ctl.last_max_vert_cfl == 7.0, "vertical should be clobbered"
    assert ctl.last_max_horiz_cfl == 0.3, "horizontal should be preserved"


def test_a_cfl_spike_cannot_collapse_the_step_below_the_floor():
    """calc_dt's 0.1 factor floor, over a sequence rather than one call."""
    ctl = _controller()
    taken = _run(ctl, [50.0, 50.0, 50.0], horiz=50.0)
    for a, b in zip(taken, taken[1:]):
        assert b >= a * Fraction(1, 10), (a, b)


def test_the_controller_converges_toward_target_on_a_steady_cfl_rate():
    """The property that makes it a controller and not just a limiter.

    Hold the physical rate fixed (CFL scales with dt) and the step should
    settle where CFL ~ target rather than oscillating or drifting.
    """
    rate = 0.02           # per-second CFL rate: cfl = rate * dt
    ctl = _controller(max_step_increase_pct=50)
    dt = ctl.first_step()
    for _ in range(60):
        cfl = rate * float(dt)
        nxt = ctl.next_dt(max_vert_cfl=cfl, max_horiz_cfl=0.01)
        ctl.accept(nxt, max_vert_cfl=cfl, max_horiz_cfl=0.01,
                   stepping_to_time=False)
        dt = nxt
    settled = rate * float(dt)
    assert settled == pytest.approx(1.2, abs=0.08), (
        f"settled at CFL {settled}, not near target 1.2 (dt={float(dt)})")


# --------------------------------------------------- step_to_output_time

@pytest.mark.parametrize("dt,remaining,expect_dt,expect_flag", [
    (Fraction(30), Fraction(0), Fraction(30), False),      # nothing due
    (Fraction(30), Fraction(100), Fraction(30), False),    # far away
    (Fraction(30), Fraction(45), Fraction(45, 2), True),   # halve
    (Fraction(30), Fraction(20), Fraction(20), True),      # take remainder
    (Fraction(30), Fraction(30), Fraction(30), True),      # exact
])
def test_step_to_time_looks_two_steps_ahead(dt, remaining, expect_dt,
                                            expect_flag):
    """:299-374.  Halving at one-to-two steps out avoids a tiny step.

    Upstream's own comment: "We look out two time steps to avoid having a
    very short time step.  Very short time steps can cause model
    instability."
    """
    ctl = _controller()
    got, flag = ctl.step_to_time(dt, remaining)
    assert got == expect_dt
    assert flag is expect_flag


def test_step_to_time_never_manufactures_a_step_under_half():
    """The point of the two-step lookahead, as a property."""
    ctl = _controller()
    dt = Fraction(30)
    for numerator in range(1, 121):
        remaining = Fraction(numerator, 2)
        got, flag = ctl.step_to_time(dt, remaining)
        if flag and remaining > dt:
            assert got >= dt / 2, (remaining, got)
