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

def test_a_late_nest_takes_its_first_step_through_first_step():
    """`period == 0` was the gate, and a late nest never sees period 0.

    Left that way the nest's controller stayed unstarted for the life of
    the run, so max_step_increase_pct (adapt_timestep_em.F:168-177) was
    never applied to it -- and it was driven from its first period with a
    CFL of zero, which is calc_dt's negligible-CFL branch, so it ratcheted
    straight to max_dt before its first solve.
    """
    model = _late_nest_tree(root_dt_s=30, ratio=5, start_s=60)
    d = _driver(model, {1: (0.5, 0.1), 2: (0.0, 0.0)})
    root_clock = model.node(1).clock
    child_clock = model.node(2).clock
    # Periods 0 and 1: the root only.  The nest joins at period 2.
    d(0, {1: root_clock})
    d(1, {1: root_clock})
    child_ctl = d.controllers[2]
    assert not child_ctl.started
    d(2, {1: root_clock, 2: child_clock})
    assert child_ctl.started, (
        "the nest was driven without ever taking its first step, so its "
        "growth bound was never armed")
    # 6 s configured; the unbounded branch would have taken it to the
    # 10 km/5 nest's max_time_step of 8*2 = 16 s in one period.
    assert child_clock.step_ticks <= 6 * TICK_DEN


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
