"""The nest divide must not collapse a child's step to arithmetic.

WHAT THIS PINS, and it was found the expensive way.  A nest's step is the
largest divisor of its parent's tick count at or below what its CFL asks
for.  When the parent's tick count is poor in factors, the smallest
divisor at or above the ceiling is enormous, and the child gets a step
hundreds of times smaller than it requested.

MEASURED on a 10 / 2 / 0.667 km tree with lattice 15: a root of 5595
ticks is 15 x 373, so d02 took 1119 = 3 x 373 -- whose only divisors are
1, 3, 373 and 1119.  d03 asked for about 1 s, the smallest usable divisor
was 373, and it ran at 0.03 s taking 373 substeps inside one parent step.

The failure mode is what makes this worth a file of its own: the run does
NOT fail.  It slows by the factor -- 11.5 model-seconds per wall-second
fell to 0.23 -- and from outside that is indistinguishable from a
controller responding to a violent flow.  Every field was healthy while
it happened: max |w| FALLING 38.7 -> 23.0 m/s, cells over 20 m/s falling
1145 -> 62, surface pressure moving 1.5 mb in two hours.  It was read as
a diverging storm before the fields were checked.

Two defences, and they are different in kind.  The root is snapped to a
SMOOTH tick count so the quotients have interior divisors to adapt along;
and a substep count far past the grid ratio is REFUSED, because slow and
silent is the worst way for this to fail.
"""

from __future__ import annotations

import pytest

from gpuwm.core.adaptive_clock import (          # noqa: E402
    MAX_NEST_SUBSTEP_FACTOR, NestDivideRefusal, _ROOT_SMOOTH_FACTOR,
    nest_ticks_from_parent,
)


# ------------------------------------------------ the measured collapse

def test_the_parent_tick_count_that_actually_collapsed():
    """1119 ticks: divisors 1, 3, 373, 1119.  Nothing usable between."""
    assert [n for n in range(1, 1120) if 1119 % n == 0] == [1, 3, 373, 1119]


def test_a_divisor_poor_parent_is_refused_not_run_slowly():
    """The guard that turns a 373x slowdown into a message.

    Refusing is the point.  Returning 3 ticks here is arithmetically
    correct and operationally useless, and nothing downstream could tell
    the difference between that and a CFL that genuinely wanted 0.03 s.
    """
    with pytest.raises(NestDivideRefusal, match="substeps inside one parent"):
        nest_ticks_from_parent(1119, 100, subtree_lattice=3,
                               max_substeps=24, grid_id=3)


def test_the_refusal_names_the_remedy():
    try:
        nest_ticks_from_parent(1119, 100, subtree_lattice=3,
                               max_substeps=24, grid_id=3)
    except NestDivideRefusal as exc:
        text = str(exc)
    assert "min_time_step" in text, text
    assert "grid_id=3" in text, text
    assert "373" in text, text


# ------------------------------------------- the smooth root that fixes it

def test_a_smooth_root_gives_the_nest_a_ladder_to_adapt_along():
    """The same tree, one root tick count apart.

    5595 is on the lattice (15 x 373) and still collapses; 5580 is on the
    lattice AND smooth, and every request lands within the ratio band.
    """
    poor, smooth = 5595 // 5, 5580 // 5           # d02 at ratio 5
    assert poor == 1119 and smooth == 1116
    assert len([n for n in range(1, poor + 1) if poor % n == 0]) == 4
    assert len([n for n in range(1, smooth + 1) if smooth % n == 0]) > 12

    for wanted_ticks, expect_n in ((100, 12), (200, 6), (300, 4)):
        got = nest_ticks_from_parent(smooth, wanted_ticks, subtree_lattice=3,
                                     max_substeps=24, grid_id=3)
        assert smooth // got == expect_n, (wanted_ticks, got, smooth // got)


def test_the_smooth_factor_is_a_power_of_two():
    """Its divisors must COMBINE with the lattice's, not duplicate them."""
    assert _ROOT_SMOOTH_FACTOR in (2, 4, 8)
    assert _ROOT_SMOOTH_FACTOR & (_ROOT_SMOOTH_FACTOR - 1) == 0


def test_the_smooth_factor_is_the_smallest_one_that_clears_the_collapse():
    """MEASURED, not asserted.

    The constant carried the claim that larger values coarsen the root
    for no further benefit, with nothing behind it.  This walks the
    three-level 10 / 2 / 0.667 km shape (lattice 15) the way the driver
    does -- snap the root, divide d02 off it with the lattice preference,
    divide d03 off d02 -- and counts the d03 requests the divide cannot
    meet inside the ceiling.  Below 4 there are always some; at 4 there
    are none, and every larger multiplier costs root resolution in
    proportion for none.
    """
    lattice, ratio_d02, ratio_d03 = 15, 5, 3
    ceil_d02 = MAX_NEST_SUBSTEP_FACTOR * ratio_d02
    ceil_d03 = MAX_NEST_SUBSTEP_FACTOR * ratio_d03

    def refusals(factor):
        step = lattice * factor
        count = 0
        for want in range(2000, 9001, 7):        # 20 s to 90 s of ticks
            root = (want // step) * step
            if root <= 0:
                continue
            d02 = nest_ticks_from_parent(root, max(1, root // ratio_d02),
                                         subtree_lattice=ratio_d03,
                                         max_substeps=ceil_d02)
            for frac in (15, 20, 30, 45, 60):
                try:
                    nest_ticks_from_parent(d02, max(1, root // frac),
                                           max_substeps=ceil_d03)
                except NestDivideRefusal:
                    count += 1
        return count

    below = {factor: refusals(factor)
             for factor in range(1, _ROOT_SMOOTH_FACTOR)}
    assert all(below.values()), below
    assert refusals(_ROOT_SMOOTH_FACTOR) == 0
    assert refusals(_ROOT_SMOOTH_FACTOR * 2) == 0


# ------------------------------------------------------ the ordinary path

def test_a_healthy_divide_is_untouched():
    """Positive control: this is a guard, not a new policy.

    A parent rich in factors hands down exactly what it always did.
    """
    parent = 6000                                  # 60 s at 100 ticks/s
    # 400 is a divisor of 6000 and is still NOT taken: 400 % 3 != 0, so it
    # would leave this domain's own children with no exact divide.  375 is
    # the next one down that does, and stepping DOWN is the safe
    # direction -- a smaller step is always admissible where a larger one
    # was asked for, never the reverse.
    for wanted_ticks, expect in ((1200, 1200), (600, 600), (400, 375)):
        got = nest_ticks_from_parent(parent, wanted_ticks, subtree_lattice=3,
                                     max_substeps=24, grid_id=3)
        assert got == expect, (wanted_ticks, got)
        assert parent % got == 0 and got % 3 == 0


def test_the_quotient_stays_divisible_for_the_NEXT_level_down():
    """Why the lattice is passed at all.

    A middle domain is a parent AND adapts, so its own tick count has to
    stay divisible by what ITS children need -- the guarantee
    `_quantise_root` gives the root and nothing gave the middle.
    """
    got = nest_ticks_from_parent(6000, 900, subtree_lattice=3,
                                 max_substeps=24, grid_id=2)
    assert 6000 % got == 0
    assert got % 3 == 0, f"{got} leaves the next level down with no divide"


def test_the_bound_scales_with_the_grid_ratio():
    """A 5:1 nest may legitimately need more substeps than a 3:1 one."""
    assert MAX_NEST_SUBSTEP_FACTOR >= 4
    generous = MAX_NEST_SUBSTEP_FACTOR * 5
    got = nest_ticks_from_parent(6000, 200, subtree_lattice=1,
                                 max_substeps=generous, grid_id=2)
    assert 6000 // got <= generous


def test_no_bound_means_no_refusal():
    """The bound is opt-in, so existing callers keep the old behaviour."""
    got = nest_ticks_from_parent(1119, 100, subtree_lattice=3)
    assert got == 3, got


# ---------------------------------------------------------------------------
# the lattice preference is a PREFERENCE
# ---------------------------------------------------------------------------

def test_an_unaffordable_lattice_divisor_degrades_to_the_plain_divide():
    """It must not become a refusal.

    The lattice preference keeps a MIDDLE domain divisor-rich for its own
    children, and it is the only case where ``subtree_lattice > 1``.
    Returning through the substep check meant that when the smallest
    lattice-friendly divisor was unaffordable, the run died on a message
    naming an arithmetic collapse -- while the plain divide had a small,
    legal answer sitting right there.  MEASURED on this tree: parent 246
    ticks, child asking 62, lattice 3 -- the smallest lattice-friendly
    divisor is 41 substeps, past a ceiling of 24, while the plain divide
    answers 6 substeps and hands the child 41 ticks.
    """
    from gpuwm.core.adaptive_clock import nest_ticks_from_parent

    plain = nest_ticks_from_parent(246, 62, subtree_lattice=1,
                                   max_substeps=24, grid_id=3)
    preferred = nest_ticks_from_parent(246, 62, subtree_lattice=3,
                                       max_substeps=24, grid_id=3)
    assert plain == preferred == 41, (plain, preferred)
    # 246 // 41 == 6 substeps, well inside the ceiling
    assert 246 % preferred == 0


def test_the_preference_still_wins_when_it_is_affordable():
    """The positive control: degrading is not the same as ignoring."""
    from gpuwm.core.adaptive_clock import nest_ticks_from_parent

    # 240 ticks, child asking 47: the plain divide takes 6 substeps and
    # hands the child 40 ticks, which its own 3:1 children cannot divide;
    # the lattice-friendly answer is 8 substeps and 30 ticks, still well
    # inside the ceiling, so the preference is taken.
    assert nest_ticks_from_parent(240, 47, subtree_lattice=1,
                                  max_substeps=24) == 40
    assert nest_ticks_from_parent(240, 47, subtree_lattice=3,
                                  max_substeps=24) == 30


def test_the_refusal_still_fires_when_no_divisor_is_affordable():
    """Degrading must not become a way to never refuse."""
    import pytest
    from gpuwm.core.adaptive_clock import (
        NestDivideRefusal, nest_ticks_from_parent)

    # 1119 = 3 x 373: above 3 the next divisor is 373 substeps
    with pytest.raises(NestDivideRefusal, match="arithmetic collapse"):
        nest_ticks_from_parent(1119, 3, subtree_lattice=3,
                               max_substeps=24, grid_id=3)


# ---------------------------------------------------------------------------
# the root smoothing is for trees that HAVE nests
# ---------------------------------------------------------------------------

def test_a_single_domain_root_is_not_coarsened():
    """There is no divide to keep rich in factors, so it costs only.

    MEASURED before this: a single-domain adaptive run asking 22.53 s was
    handed 22.52 s, and an explicit ``starting_time_step`` was honoured
    only to within four ticks -- a step the flow did not ask for, in
    service of children that do not exist.
    """
    from fractions import Fraction
    from gpuwm.core.adaptive_clock import AdaptiveClockDriver

    drv = AdaptiveClockDriver.__new__(AdaptiveClockDriver)
    drv.tick_den = 100
    drv.nest_lattice = 1
    drv.root_smooth_factor = 1
    assert drv._quantise_root(Fraction(2253, 100)) == Fraction(2253, 100)


def test_a_tree_with_nests_keeps_the_smoothing():
    """The positive control, on the same helper."""
    from fractions import Fraction
    from gpuwm.core.adaptive_clock import (
        _ROOT_SMOOTH_FACTOR, AdaptiveClockDriver)

    drv = AdaptiveClockDriver.__new__(AdaptiveClockDriver)
    drv.tick_den = 100
    drv.nest_lattice = 3
    drv.root_smooth_factor = _ROOT_SMOOTH_FACTOR
    snapped = drv._quantise_root(Fraction(2253, 100))
    ticks = int(snapped * 100)
    assert ticks % (3 * _ROOT_SMOOTH_FACTOR) == 0, ticks
    assert snapped <= Fraction(2253, 100)
