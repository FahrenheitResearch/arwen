"""``AdaptiveClockDriver`` -- the wire between CFL, controller and clock.

Driven with an injected CFL source and a minimal fake tree, so the three
things that must move together can be gated without a GPU.  Two of those
three fail SILENTLY in production if they are missed, which is exactly
why they are asserted here rather than left to a forecast to reveal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import pytest

from gpuwm.core.adaptive_clock import AdaptiveClockDriver, ticks_of

TICK_DEN = 100          # hundredths: what an adaptive clock needs


# --------------------------------------------------------------- fakes

@dataclass
class FakeSpec:
    grid_id: int
    step_ticks: int
    start_ticks: int = 0
    history_ticks: int = 0
    # The other two exact-modulo alarms the root must also land on.
    restart_ticks: int | None = None
    lbc_interval_ticks: int | None = None


@dataclass
class FakeClock:
    spec: FakeSpec
    ticks: int = 0
    step_ticks: int = 0
    # DomainClock always carries this; the driver clamps the last step to
    # it so the run ends ON the stop boundary rather than past it.
    run_ticks: int = 10 ** 9

    def __post_init__(self):
        self.step_ticks = self.spec.step_ticks


@dataclass
class FakeRun:
    grid_id: int
    dt: float
    dx: float = 10000.0
    dy: float = 10000.0
    time_step_sound: int = 4
    target_cfl: float = 1.2
    target_hcfl: float = 0.84
    max_step_increase_pct: int = 5
    min_time_step: int = -1
    min_time_step_den: int = 0
    max_time_step: int = -1
    max_time_step_den: int = 0
    starting_time_step: int = -1
    starting_time_step_den: int = 0
    cu_physics: int = 0
    bldt: float = 0.0


@dataclass
class FakeCfg:
    grid_id: int
    run: FakeRun
    parent_time_step_ratio: int = 1


@dataclass
class FakeCarrierRecord:
    last_update_model_time: float = 0.0


@dataclass
class FakeCarriers:
    records: dict = field(default_factory=dict)


@dataclass
class FakePhysics:
    """Only the attributes the cadence refresh and the two drivers touch."""

    stepra: int = 1
    stepcu: int = 1
    stepbl: int = 1
    radt_minutes: float = 6.0
    cudt_minutes: float = 5.0
    radt_seconds: float = 360.0
    cudt_seconds: float = 300.0
    bldt_seconds: float = 30.0
    radiation_due_override: object = None
    cumulus_due_override: object = None
    carriers: object = None


@dataclass
class FakeState:
    physics: object | None = None


@dataclass
class FakeNode:
    cfg: FakeCfg
    clock: FakeClock
    parent: object | None = None
    children: list = field(default_factory=list)
    state: FakeState = field(default_factory=FakeState)


class FakeModel:
    def __init__(self, tree):
        self.root = tree
        self._by_id = {}

        def walk(n):
            self._by_id[int(n.cfg.grid_id)] = n
            for k in n.children:
                walk(k)
        walk(tree)

    def node(self, gid):
        return self._by_id[int(gid)]


def _tree(root_dt_s=30, ratio=5, history_s=0):
    root_ticks = root_dt_s * TICK_DEN
    root = FakeNode(
        cfg=FakeCfg(1, FakeRun(1, float(root_dt_s), dx=10000.0, dy=10000.0)),
        clock=FakeClock(FakeSpec(1, root_ticks,
                                 history_ticks=history_s * TICK_DEN)))
    child = FakeNode(
        cfg=FakeCfg(2, FakeRun(2, root_dt_s / ratio,
                              dx=10000.0 / ratio, dy=10000.0 / ratio), ratio),
        clock=FakeClock(FakeSpec(2, root_ticks // ratio)),
        parent=root)
    root.children = [child]
    return FakeModel(root)


def _driver(model, cfls, **kw):
    return AdaptiveClockDriver(model, cfl_source=lambda gid: cfls[gid],
                               tick_den=TICK_DEN, **kw)


def _late_nest_tree(root_dt_s=30, ratio=5, start_s=60):
    """A nest whose first period is NOT period 0."""
    root_ticks = root_dt_s * TICK_DEN
    root = FakeNode(
        cfg=FakeCfg(1, FakeRun(1, float(root_dt_s))),
        clock=FakeClock(FakeSpec(1, root_ticks)))
    child = FakeNode(
        cfg=FakeCfg(2, FakeRun(2, root_dt_s / ratio,
                               dx=10000.0 / ratio, dy=10000.0 / ratio),
                    ratio),
        clock=FakeClock(FakeSpec(2, root_ticks // ratio,
                                 start_ticks=start_s * TICK_DEN)),
        parent=root)
    root.children = [child]
    return FakeModel(root)


# ------------------------------------------------------------- ticks_of

def test_ticks_of_refuses_a_dt_that_is_not_a_whole_tick():
    """Rounding here would desync the nest and surface far from the cause."""
    with pytest.raises(ValueError, match="whole number of"):
        ticks_of(Fraction(1, 3), TICK_DEN)


def test_ticks_of_accepts_the_hundredths_the_controller_emits():
    assert ticks_of(Fraction(3017, 100), 100) == 3017


def test_a_clock_without_a_hundredths_denominator_is_refused_by_name():
    """tick_den must carry a factor of 100 for an adaptive run."""
    with pytest.raises(ValueError, match="multiple"):
        ticks_of(Fraction(1, 100), 3)


# ------------------------------------------------------------ the wire

def test_the_first_period_takes_the_configured_step():
    model = _tree()
    d = _driver(model, {1: (0.0, 0.0), 2: (0.0, 0.0)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    d(0, clocks)
    assert clocks[1].step_ticks == 30 * TICK_DEN
    assert model.node(1).cfg.run.dt == 30.0


def test_all_three_move_together():
    """clock.step_ticks, cfg.run.dt, and the physics cadence.

    Missing either of the last two is silent in production: the kernels
    would integrate on a dt the calendar disagrees with, or radiation
    would keep the interval it was built with.
    """
    model = _tree()
    d = _driver(model, {1: (0.6, 0.1), 2: (0.6, 0.1)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    d(0, clocks)
    d(1, clocks)
    node = model.node(1)
    assert node.cfg.run.dt == pytest.approx(
        node.clock.step_ticks / TICK_DEN), (
        "cfg.dt and the tick calendar disagree; the kernels and the clock "
        "would be integrating different timesteps")


def test_a_low_cfl_grows_the_step_and_a_high_one_shrinks_it():
    model = _tree()
    grow = _driver(model, {1: (0.3, 0.05), 2: (0.3, 0.05)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    grow(0, clocks)
    before = clocks[1].step_ticks
    grow(1, clocks)
    assert clocks[1].step_ticks > before

    # Shrinking needs room below the floor.  WRF's default
    # min_time_step is 3*dx(km) = 30 s here, which IS the configured
    # step, so without an explicit floor there is nowhere to go -- that
    # is upstream's design, not a limitation of the test.
    model2 = _tree()
    for gid in (1, 2):
        model2.node(gid).cfg.run.min_time_step = 1
    shrink = _driver(model2, {1: (3.0, 0.05), 2: (3.0, 0.05)})
    clocks2 = {1: model2.node(1).clock, 2: model2.node(2).clock}
    shrink(0, clocks2)
    before2 = clocks2[1].step_ticks
    shrink(1, clocks2)
    assert clocks2[1].step_ticks < before2


# -------------------------------------------------------------- nests

@pytest.mark.parametrize("cfl", [0.1, 0.3, 0.6, 1.2, 2.0, 5.0])
def test_the_nest_step_always_divides_its_parents_exactly(cfl):
    """The invariant the expander's tick-exact assertion enforces.

    If this ever fails the run does not drift -- it refuses on the first
    period -- but it refuses, so the controller must never propose it.
    """
    model = _tree()
    for gid in (1, 2):                       # a realistic run floors the step
        model.node(gid).cfg.run.min_time_step = 1
    d = _driver(model, {1: (cfl, 0.1), 2: (cfl, 0.1)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    for period in range(6):
        d(period, clocks)
        parent, child = clocks[1].step_ticks, clocks[2].step_ticks
        assert child > 0
        assert parent % child == 0, (
            f"period {period}: parent {parent} ticks is not a whole "
            f"number of child steps of {child}")


def test_the_parent_is_chosen_before_the_child():
    """Order matters: the child derives its step from the parent's."""
    model = _tree()
    d = _driver(model, {1: (0.5, 0.1), 2: (0.5, 0.1)})
    assert d.order == [1, 2]


# --------------------------------------------------- relocation carry

def test_the_controller_survives_the_state_object_being_replaced():
    """Invariant (e), closed by construction.

    Relocation sets ``state.physics = None`` and rebuilds the state.  The
    controller is keyed by grid_id on the driver and the live step is on
    the CLOCK, which relocation only reads -- so neither is lost.  Before
    this was arranged, the probe keyed on id(state) and a 4-hour run split
    one domain's history into seven segments.
    """
    model = _tree()
    d = _driver(model, {1: (0.4, 0.1), 2: (0.4, 0.1)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    d(0, clocks)
    d(1, clocks)
    remembered = d.controllers[1].last_dt
    grown = clocks[1].step_ticks

    # a relocation: the state object is REPLACED
    model.node(1).state = FakeState()
    model.node(2).state = FakeState()

    d(2, clocks)
    assert d.controllers[1].last_dt != Fraction(30), (
        "the controller reset to its configured start after a relocation")
    assert clocks[1].step_ticks >= grown, (
        "the live step fell back after a relocation")


# ------------------------------------------------- step_to_output_time

def test_the_root_lands_exactly_on_its_history_time():
    """Otherwise the frame is not written LATE -- it is never written.

    Every executor alarm is `ticks % interval == 0` with no at-or-past
    arm, so a history time the clock steps over is silently skipped.
    """
    model = _tree(root_dt_s=30, history_s=100)     # 100 is NOT a multiple of 30
    d = _driver(model, {1: (0.9, 0.1), 2: (0.9, 0.1)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    hist = clocks[1].spec.history_ticks

    landed = False
    for period in range(12):
        d(period, clocks)
        clocks[1].ticks += clocks[1].step_ticks
        clocks[2].ticks = clocks[1].ticks
        if clocks[1].ticks % hist == 0:
            landed = True
            break
    assert landed, (
        "the root never landed on a history tick; with an exact-modulo "
        "alarm that frame is silently skipped")


def test_a_nest_is_NOT_step_to_timed():
    """Upstream gates step_to_output_time `.not. grid%nested` (:325).

    A nest lands on its parent's boundary by DIVIDING it, not by
    shortening itself -- shortening a child would break the division the
    expander asserts.
    """
    model = _tree(root_dt_s=30, history_s=100)
    d = _driver(model, {1: (0.9, 0.1), 2: (0.9, 0.1)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    for period in range(8):
        d(period, clocks)
        assert clocks[1].step_ticks % clocks[2].step_ticks == 0


def test_a_step_driven_below_the_tick_resolution_is_REFUSED_by_name():
    """The tick lattice has a floor; WRF's rational clock does not.

    A silent clamp would let a diverging run grind along at the smallest
    representable step looking healthy.  The message names min_time_step,
    which is the deliberate way to floor it.
    """
    model = _tree()
    # WRF's own default min_time_step (3*dx = 30 s here) already prevents
    # this, so reaching the floor at all takes a deliberately tiny one --
    # which is the case the refusal exists for.
    for gid in (1, 2):
        model.node(gid).cfg.run.min_time_step = 0
        model.node(gid).cfg.run.min_time_step_den = 1
    d = _driver(model, {1: (500.0, 500.0), 2: (500.0, 500.0)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    with pytest.raises(ValueError, match="min_time_step"):
        for period in range(40):
            d(period, clocks)


def test_min_time_step_prevents_that_refusal():
    """The remedy the message names actually works."""
    model = _tree()
    for gid in (1, 2):
        model.node(gid).cfg.run.min_time_step = 1
    d = _driver(model, {1: (500.0, 500.0), 2: (500.0, 500.0)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    for period in range(40):
        d(period, clocks)
    assert clocks[1].step_ticks >= TICK_DEN


# ------------------------------- the bug a live run found, as a gate

def test_a_nearly_prime_parent_tick_count_does_not_collapse_the_nest():
    """The regression for the failure that killed the first live run.

    33.08 s is 3308 ticks, whose divisors are 1, 2, 4, 827, 1654, 3308.
    Raising n from 5 until it divides reaches 827 -- 827 child steps
    inside one parent step, a 6.6 s nest step collapsed to 0.04 s.  The
    live run died at outer step 4 with "MYJ returned non-finite rublten".

    Two things prevent it now: the root is snapped to a multiple of the
    ratio lattice so a divisor near the ceiling always exists, and the
    search takes the smallest DIVISOR at or above the ceiling rather than
    raising the ceiling until it happens to divide.
    """
    from gpuwm.core.adaptive_clock import nest_ticks_from_parent
    # the exact numbers from the failure, unsnapped
    collapsed = nest_ticks_from_parent(3308, 662)
    assert collapsed == 4, "documents the raw hazard: 3308/827"

    # snapped to the ratio lattice (5), the nest keeps a sane step
    snapped_parent = (3308 // 5) * 5          # 3305
    got = nest_ticks_from_parent(snapped_parent, 662)
    assert got == 661, got
    assert snapped_parent % got == 0


def test_the_root_never_leaves_a_nest_without_a_near_divisor():
    """The property, over a trajectory rather than one number.

    The nest's step must stay within a factor of two of what its own CFL
    asked for; a collapse is what the live failure looked like.
    """
    model = _tree(root_dt_s=30, ratio=5)
    d = _driver(model, {1: (0.55, 0.1), 2: (0.35, 0.1)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    for period in range(25):
        d(period, clocks)
        parent, child = clocks[1].step_ticks, clocks[2].step_ticks
        assert parent % child == 0
        assert child * 12 >= parent, (
            f"period {period}: nest step {child} ticks inside a parent of "
            f"{parent} is {parent // child} substeps -- a collapse")


def test_the_last_step_lands_ON_the_stop_boundary():
    """The regression for a run that integrated fully and then overshot.

    execute_schedule loops `while root.ticks < run_ticks` and takes
    whatever step the controller offers, so without a clamp the final
    step walks past the finish -- measured at 183385 ticks against a
    run_ticks of 180000, refused by the tick-exact stop check after the
    whole forecast had been computed.
    """
    model = _tree(root_dt_s=30)
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    run_ticks = 100 * TICK_DEN                      # 100 s
    for c in clocks.values():
        c.run_ticks = run_ticks
    d = _driver(model, {1: (0.5, 0.1), 2: (0.5, 0.1)})
    for period in range(40):
        if clocks[1].ticks >= run_ticks:
            break
        d(period, clocks)
        clocks[1].ticks += clocks[1].step_ticks
        clocks[2].ticks = clocks[1].ticks
        assert clocks[1].ticks <= run_ticks, (
            f"period {period}: stepped past the stop boundary to "
            f"{clocks[1].ticks} of {run_ticks}")
    assert clocks[1].ticks == run_ticks, (
        f"ended at {clocks[1].ticks}, not exactly {run_ticks}")


# ------------------------------------------- starting_time_step is read

def test_an_explicit_starting_time_step_is_the_first_step():
    """The value was validated, filed and checkpointed -- and read by nothing.

    Measured on a 3 km case before this: `starting_time_step = 20`
    produced a run identical to the unset one in step count, first dt and
    every output frame, because the driver fell straight through to the
    configured `time_step`.
    """
    model = _tree(root_dt_s=30)
    for gid in (1, 2):
        model.node(gid).cfg.run.starting_time_step = 20
    d = _driver(model, {1: (0.0, 0.0), 2: (0.0, 0.0)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    d(0, clocks)
    assert clocks[1].step_ticks == 20 * TICK_DEN
    assert model.node(1).cfg.run.dt == 20.0


def test_a_fractional_starting_time_step_takes_its_den_companion():
    """22.5 s asked for, 22.4 s taken: the SMOOTH-root cost, priced here.

    The root's step is snapped down to a multiple of the ratio lattice
    times ``_ROOT_SMOOTH_FACTOR``, not to the bare lattice.  On this tree
    that is 20 ticks rather than 5, so 2250 rounds to 2240 -- one tenth
    of a second of root step resolution.  That is the whole price of the
    smoothing, and what it buys is bounded and measured: the root's
    quotients carry interior divisors, so a lattice-exact root can no
    longer leave a prime cofactor -- which handed a 3:1 nest 373 substeps
    in one parent step (tests/test_nest_divide_collapse.py).  It is not a
    guarantee about every descendant at every depth; the divide's own
    lattice preference is what carries it further down, and that
    preference degrades to the plain divide rather than refusing.  The
    price is charged only to a tree that HAS nests.  The DEN companion is
    still
    what is being read -- an integer starting step would land on the
    lattice exactly.
    """
    model = _tree(root_dt_s=30)
    for gid in (1, 2):
        model.node(gid).cfg.run.starting_time_step = 45
        model.node(gid).cfg.run.starting_time_step_den = 2      # 22.5 s
    d = _driver(model, {1: (0.0, 0.0), 2: (0.0, 0.0)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    d(0, clocks)
    assert clocks[1].step_ticks == 2240
    # ... and it really is the smoothing, not a lost fraction: the bare
    # lattice would have taken the request exactly.
    assert 2250 % (d.nest_lattice) == 0


def test_an_unset_starting_time_step_keeps_the_configured_step():
    """gpuwm's named divergence: -1 is the CONFIGURED step, not 4*dx.

    Upstream substitutes NINT(4*dx km) = 40 s on this 10 km root
    (start_em.F:939-941).  A prepared tree has its output alarms, nest
    ratios and boundary interval built around the configured step, so the
    port keeps it and says so in docs/ADAPTIVE-TIMESTEP.md.
    """
    model = _tree(root_dt_s=30)
    d = _driver(model, {1: (0.0, 0.0), 2: (0.0, 0.0)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    d(0, clocks)
    assert clocks[1].step_ticks == 30 * TICK_DEN


# ------------------------------------------------------- the late nest

def _executor_boundary(clocks, root_gid=1):
    """What execute_schedule does between periods: every clock at the boundary.

    The root advanced by its step; an unstarted nest is PARKED at the same
    boundary (clock.py: ``if gid not in started: dom.ticks = boundary``).
    """
    boundary = clocks[root_gid].ticks + clocks[root_gid].step_ticks
    for clock in clocks.values():
        clock.ticks = boundary


def test_a_late_nest_is_not_driven_until_it_starts_and_enters_on_its_ratio_step():
    """The executor hands the driver EVERY clock; an unstarted nest is parked.

    Two defects met here.  `period == 0` used to gate `first_step`, so a
    late nest never took it and its growth bound was never armed.  Then the
    `clocks.get(gid) is None` skip was dead -- the executor never omits a
    domain -- so from period 1 the nest was driven on cfl_source's (0, 0)
    for a domain that had folded nothing, ratcheted to max_time_step
    (16 s on this 10 km/5 nest) before its first solve, and had its
    node.cfg.run.dt rewritten to that value for initialize_child to read.
    """
    model = _late_nest_tree(root_dt_s=30, ratio=5, start_s=60)
    # The root sits exactly at target_cfl, so its step holds at 30 s and
    # the second boundary lands on the nest's 60 s start.
    d = _driver(model, {1: (1.2, 0.5), 2: (0.0, 0.0)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    child = model.node(2)
    child_ctl = d.controllers[2]
    configured_child_ticks = clocks[2].step_ticks
    # Periods 0 and 1: the root only.  The nest joins at period 2 (60 s).
    for period in (0, 1):
        d(period, clocks)
        assert not child_ctl.started, period
        assert clocks[2].step_ticks == configured_child_ticks, (
            "an unstarted nest's clock was written by the controller")
        assert child.cfg.run.dt == 6.0, (
            "an unstarted nest's cfg.run.dt was rewritten before "
            "initialize_child could read the configured step")
        assert child_ctl.last_dt == Fraction(6), (
            "an unstarted nest's controller was driven on a (0, 0) CFL")
        _executor_boundary(clocks)
    assert clocks[2].ticks == 60 * TICK_DEN == clocks[2].spec.start_ticks
    d(2, clocks)
    assert child_ctl.started, (
        "the nest was driven without ever taking its first step, so its "
        "growth bound was never armed")
    # 6 s configured; the unbounded branch would have taken it to the
    # 10 km/5 nest's max_time_step of 8*2 = 16 s in one period -- the
    # parent's own step.  The entry step is the configured ratio step,
    # divided into whatever the parent is taking now.
    assert clocks[2].step_ticks <= 6 * TICK_DEN
    assert clocks[1].step_ticks % clocks[2].step_ticks == 0
    assert clocks[1].step_ticks // clocks[2].step_ticks >= 5
    assert child_ctl.last_dt == Fraction(6)


def test_a_late_nest_under_a_grown_root_still_enters_below_its_ratio_step():
    """Calm flow lets the root grow before the nest starts; the nest must not
    inherit the growth."""
    model = _late_nest_tree(root_dt_s=30, ratio=5, start_s=120)
    d = _driver(model, {1: (0.0, 0.0), 2: (0.0, 0.0)})     # calm: root grows
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    period = 0
    while clocks[2].ticks < clocks[2].spec.start_ticks:
        d(period, clocks)
        assert not d.controllers[2].started
        _executor_boundary(clocks)
        period += 1
    assert clocks[1].step_ticks > 30 * TICK_DEN, "the root did grow"
    d(period, clocks)
    assert d.controllers[2].started
    assert clocks[2].step_ticks <= 6 * TICK_DEN
    assert clocks[1].step_ticks % clocks[2].step_ticks == 0


# -------------------------------------------- the physics cadence wire

def _stepped(model, gid, physics, seconds):
    node = model.node(gid)
    node.state.physics = physics
    node.clock.elapsed_seconds = seconds


def test_the_measured_radiation_interval_reaches_radt_seconds():
    """The observation used to be computed and dropped on the floor.

    Without it `radt_seconds` is a PREDICTION (stepra * dt), and under a
    falling dt the true gap exceeds it -- 675.65 s of gap against a
    668.56 s tolerance, which refused a run in which nothing had stopped.
    """
    model = _tree(root_dt_s=30)
    record = FakeCarrierRecord(0.0)
    physics = FakePhysics(radt_minutes=6.0,
                          carriers=FakeCarriers({"lw": record}))
    d = _driver(model, {1: (0.5, 0.1), 2: (0.5, 0.1)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    d(0, clocks)
    _stepped(model, 1, physics, 0.0)
    d.before_step(1)
    # A gap far longer than stepra * dt, the way a shrinking dt produces.
    record.last_update_model_time = 900.0
    _stepped(model, 1, physics, 900.0)
    d.before_step(1)
    assert physics.radt_seconds >= 900.0, (
        "radt_seconds ignored the interval radiation was measured to "
        "take, so the freshness contract is grading against a prediction")


def test_cumulus_is_driven_on_time_like_radiation():
    """The same defect the patch fixes for radiation, on the same predicate.

    `_cumulus_step_due` is `itimestep % stepcu == 0`, and under an
    adaptive clock neither number counts what it says.
    """
    model = _tree(root_dt_s=30)
    physics = FakePhysics(cudt_minutes=5.0, cudt_seconds=300.0)
    model.node(1).cfg.run.cu_physics = 1
    d = _driver(model, {1: (0.5, 0.1), 2: (0.5, 0.1)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    d(0, clocks)
    _stepped(model, 1, physics, 0.0)
    d.before_step(1)
    assert physics.cumulus_due_override is True, "the first call fires"
    _stepped(model, 1, physics, 30.0)
    d.before_step(1)
    assert physics.cumulus_due_override is False
    _stepped(model, 1, physics, 290.0)
    d.before_step(1)
    assert physics.cumulus_due_override is True, (
        "the next step would carry the run past cudt and cumulus did not "
        "fire")


def test_cumulus_is_left_alone_when_the_scheme_is_off():
    model = _tree(root_dt_s=30)
    physics = FakePhysics()
    d = _driver(model, {1: (0.5, 0.1), 2: (0.5, 0.1)})
    d(0, {1: model.node(1).clock, 2: model.node(2).clock})
    _stepped(model, 1, physics, 0.0)
    d.before_step(1)
    assert physics.cumulus_due_override is None


# ------------------------------------------ the quantiser is not a ratchet

def _three_level_tree(root_dt_s=30, ratios=(5, 3), min_dt_s=3):
    """10/2/0.667 km: ratio lattice 15, smoothed x4 -> a 0.6 s root grid."""
    root_ticks = root_dt_s * TICK_DEN
    root = FakeNode(
        cfg=FakeCfg(1, FakeRun(1, float(root_dt_s), min_time_step=min_dt_s)),
        clock=FakeClock(FakeSpec(1, root_ticks)))
    r1, r2 = ratios
    child = FakeNode(
        cfg=FakeCfg(2, FakeRun(2, root_dt_s / r1, dx=10000.0 / r1,
                               dy=10000.0 / r1), r1),
        clock=FakeClock(FakeSpec(2, root_ticks // r1)), parent=root)
    grandchild = FakeNode(
        cfg=FakeCfg(3, FakeRun(3, root_dt_s / (r1 * r2),
                               dx=10000.0 / (r1 * r2), dy=10000.0 / (r1 * r2)),
                    r2),
        clock=FakeClock(FakeSpec(3, root_ticks // (r1 * r2))), parent=child)
    root.children = [child]
    child.children = [grandchild]
    return FakeModel(root)


def test_one_cfl_spike_does_not_park_the_root_at_min_time_step():
    """The floor is a per-step presentation, not a permanent debit.

    `_quantise_root` floors the root's step to the smoothed lattice (0.6 s
    on this tree).  Fed back into the controller as the baseline, that
    floor erased every 5 % growth request smaller than one lattice unit --
    everything below 12 s here -- so a transient that drove the root down
    left it integrating at min_time_step for the rest of the run, with
    every field healthy.  The controller now sees its own proposal, with
    the measured CFL rescaled to it, and recovers at the documented rate:
    one max_step_increase_pct growth per root step, ln(30/3)/ln(1.05) = 48
    periods from the floor back to 30 s.
    """
    model = _three_level_tree()
    clocks = {gid: model.node(gid).clock for gid in (1, 2, 3)}
    # A PHYSICAL CFL source: flow speed k per domain times the step the
    # model actually took last period.  Calm k puts the root at exactly
    # target_cfl for a 30 s step, so 30 s is the steady state.
    k = {1: 1.2 / 30.0, 2: 0.01, 3: 0.01}

    def cfl(gid):
        value = k[gid] * clocks[gid].step_ticks / TICK_DEN
        return value, value * 0.5

    d = AdaptiveClockDriver(model, cfl_source=cfl, tick_den=TICK_DEN)
    root_dt, root_ticks = [], []
    for period in range(80):
        # Six periods of a violent column, then calm forever.
        k[1] = 6.0 / 30.0 if 1 <= period <= 6 else 1.2 / 30.0
        d(period, clocks)
        root_ticks.append(int(clocks[1].step_ticks))
        root_dt.append(clocks[1].step_ticks / TICK_DEN)
    assert root_dt[0] == 30.0
    assert min(root_dt[1:8]) == 3.0, "the spike must reach the floor"
    # Recovery: back at the pre-spike step within the growth horizon.
    recovered = [i for i, dt in enumerate(root_dt) if i > 7 and dt == 30.0]
    assert recovered and recovered[0] <= 7 + 48, root_dt
    assert all(dt == 30.0 for dt in root_dt[recovered[0]:]), root_dt
    # Every applied root step sits on the smoothed lattice, as before.
    assert all(ticks % 60 == 0 for ticks in root_ticks), root_ticks


def test_the_controller_baseline_is_the_proposal_not_the_floored_step():
    """During recovery the applied step is the floor of a larger proposal."""
    model = _three_level_tree()
    clocks = {gid: model.node(gid).clock for gid in (1, 2, 3)}
    k = {1: 1.2 / 30.0, 2: 0.01, 3: 0.01}

    def cfl(gid):
        value = k[gid] * clocks[gid].step_ticks / TICK_DEN
        return value, value * 0.5

    d = AdaptiveClockDriver(model, cfl_source=cfl, tick_den=TICK_DEN)
    for period in range(12):
        k[1] = 6.0 / 30.0 if 1 <= period <= 6 else 1.2 / 30.0
        d(period, clocks)
    ctl = d.controllers[1]
    applied = Fraction(clocks[1].step_ticks, TICK_DEN)
    assert ctl.last_dt > applied, (ctl.last_dt, applied)
    assert ctl.last_dt - applied < Fraction(60, TICK_DEN)


# ------------------------------------ the cumulus memory rides the checkpoint

def test_the_cumulus_cadence_memory_is_published_and_restored():
    """A resume must NOT re-phase cumulus to the resume instant.

    `_cumulus_fired` was the one piece of driver memory absent from
    `adaptive_state`; on resume `_drive_cumulus_on_time` took its
    `last is None` arm, fired on the first step and re-phased the whole
    cudt cadence -- silently, for every cu_physics = 1 run with the default
    cudt_minutes = 5.  The radiation twin was carried for exactly this
    reason.  The memory is also re-published after before_step, because
    the checkpoint is written at the period's END and the fire happens
    inside the period.
    """
    model = _tree(root_dt_s=30)
    physics = FakePhysics(cudt_minutes=5.0, cudt_seconds=300.0)
    model.node(1).cfg.run.cu_physics = 1
    d = _driver(model, {1: (0.5, 0.1), 2: (0.5, 0.1)})
    clocks = {1: model.node(1).clock, 2: model.node(2).clock}
    d(0, clocks)
    assert clocks[1].adaptive_state["cumulus_fired"] is None
    _stepped(model, 1, physics, 0.0)
    d.before_step(1)
    assert physics.cumulus_due_override is True, "the first call fires"
    # Published INSIDE the period, where the checkpoint writer reads it.
    checkpoint = dict(clocks[1].adaptive_state)
    assert checkpoint["cumulus_fired"] == 0.0
    assert checkpoint["radiation_seen"] is None    # nothing else moved

    # The resume: a fresh driver built from the checkpointed state.
    resumed = _tree(root_dt_s=30)
    resumed.node(1).cfg.run.cu_physics = 1
    resumed.node(1).clock.adaptive_state = checkpoint
    resumed.node(1).clock.step_ticks = clocks[1].step_ticks
    r = _driver(resumed, {1: (0.5, 0.1), 2: (0.5, 0.1)})
    rclocks = {1: resumed.node(1).clock, 2: resumed.node(2).clock}
    r(1, rclocks)
    after = FakePhysics(cudt_minutes=5.0, cudt_seconds=300.0)
    _stepped(resumed, 1, after, 30.0)
    r.before_step(1)
    assert after.cumulus_due_override is False, (
        "the resumed leg re-fired cumulus at the resume instant")
    _stepped(resumed, 1, after, 290.0)
    r.before_step(1)
    assert after.cumulus_due_override is True, (
        "the next cumulus call time moved across the resume")
    assert rclocks[1].adaptive_state["cumulus_fired"] == 290.0


def test_a_checkpoint_without_the_cumulus_key_still_resumes():
    """The key is optional on read: absent means "not yet fired"."""
    model = _tree(root_dt_s=30)
    model.node(1).cfg.run.cu_physics = 1
    model.node(1).clock.adaptive_state = {
        "last_dt_num": 30, "last_dt_den": 1, "last_max_vert_cfl": 0.5,
        "last_max_horiz_cfl": 0.1, "stepping_to_time": False,
        "started": True, "radiation_seen": None, "radiation_actual": None}
    d = _driver(model, {1: (0.5, 0.1), 2: (0.5, 0.1)})
    assert 1 not in d._cumulus_fired
