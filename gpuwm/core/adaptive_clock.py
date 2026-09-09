"""Drive a live ``dt`` from measured CFL -- the executor's ``on_period_steps``.

This is the wire between three things that already exist and were built
separately on purpose: the CFL reduction (``kernels/openbc.cu``
``w_cfl_stat``), the ported controller
(:mod:`gpuwm.core.adaptive_timestep`), and the executor's varying-step
walk (``gpuwm.core.clock.adaptive_periods``).

WHERE THE STATE LIVES, and why it is not on ``DomainState``.  Relocation
replaces the state object -- measured, a 4-hour run split one domain's
CFL series into seven segments -- so a controller kept there would reset
its ``last_dt`` on every move and silently re-run the ramp.  It is kept
here, keyed by ``grid_id``, and the LIVE step lives on
:class:`~gpuwm.core.clock.DomainClock`, which ``nest_relocation`` only
ever reads.  That is the relocation invariant, closed
by construction rather than by remembering to copy something.

THREE THINGS MUST MOVE TOGETHER when a step changes, and two of them fail
silently if they do not:

1. ``clock.step_ticks`` -- the executor's integer calendar.
2. ``node.cfg.run.dt`` -- what the KERNELS read.  All 55 ``cfg.dt``
   readers take ``cfg`` as a parameter and read ``.dt`` at call time, so
   replacing the frozen ``RunConfig`` reaches every one of them.
3. the physics driver's ``stepra``/``stepcu`` -- computed ONCE in its
   ``__init__`` from ``cfg.dt`` (``core/physics.py:2022``).  Left alone,
   radiation and cumulus keep whatever interval they were built with: no
   error, no refusal, a slowly wrong answer, and INTERMITTENT on a moving
   nest because relocation rebuilds the driver.

The float arithmetic lives here and not in the executor: ``clock.py``'s
walk is AST-audited to be integer-only, so what crosses that boundary is
an integer number of ticks and nothing else.
"""

from __future__ import annotations

from dataclasses import replace as dataclass_replace
from fractions import Fraction
from math import lcm
import os
import math


def env_flag(name: str) -> bool:
    """An environment switch that reads ``0``/``false``/``off`` as OFF.

    ``bool(os.environ.get(name))`` is true for the string ``"0"``, so a
    user who turns a probe off the obvious way turns it on.  Shared with
    :mod:`gpuwm.core.dycore`, which carries three of these.
    """
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in ("", "0", "false", "no", "off")


#: One line per domain per period; diagnosis only.
_TRACE = env_flag("GPUWM_ADAPTIVE_TRACE")


from gpuwm.core.adaptive_timestep import (AdaptiveTimestepController,
                                          nint, real_time_fp32)


#: The ``RunConfig`` fields the adaptive clock DERIVES at run time and
#: overwrites on the live config every root step.
#:
#: One authority, read by two ends that must agree:
#:
#: * :meth:`AdaptiveClockDriver._derive_run_field` computes them, and
#: * ``gpuwm.io.restart``'s identity walk exempts them from the
#:   stored-versus-live comparison when BOTH sides are adaptive.
#:
#: They were written independently once and drifted.  ``dt`` was exempted
#: and ``time_step_sound`` was not, so a checkpoint stored the DERIVED
#: sound-step count and the walk compared it against the config's
#: declared one and refused -- making every checkpoint an adaptive run
#: wrote unrestartable the moment its step grew enough to move that
#: count.  On a sub-km nest that is almost immediately:
#: :func:`wrf_num_sound_steps` leaves 4 as soon as ``300*dt/spacing >= 2``.
#:
#: Adding a field here without teaching ``_derive_run_field`` raises; the
#: reverse is impossible, because ``_apply`` iterates this set.  Under a
#: FIXED clock none of this applies and every one of these is compared
#: exactly, which is correct -- there they are real model differences.
ADAPTIVE_DERIVED_RUN_FIELDS = frozenset({"dt", "time_step_sound"})


def ticks_of(dt: Fraction, tick_den: int) -> int:
    """A timestep as an exact integer number of clock ticks.

    Refused rather than rounded when it does not land: the controller
    emits ``n/100`` s (see :func:`~gpuwm.core.adaptive_timestep.requantise`),
    so ``tick_den`` must carry a factor of 100 for an adaptive run.  A
    silent round here would put the domains a fraction of a tick apart and
    surface later as a nest-sync failure with no obvious cause.
    """
    scaled = dt * tick_den
    if scaled.denominator != 1:
        raise ValueError(
            f"adaptive dt {dt} s ({float(dt):g}) is not a whole number of "
            f"1/{tick_den} s ticks.  The controller emits hundredths, so "
            f"an adaptive run needs a tick denominator that is a multiple "
            f"of {PRECISION_TICKS}; this clock resolved {tick_den}.")
    return int(scaled)


#: The tick denominator an adaptive clock needs as a factor, because
#: every interval the controller produces is a whole number of these.
PRECISION_TICKS = 100


def wrf_default_clamps(dx: float, dy: float) -> tuple[int, int, int]:
    """WRF's ``-1`` fill-ins (``dyn_em/start_em.F:939-953``).

        starting_time_step = NINT(4 * min(dx,dy)/1000)
        max_time_step      = NINT(8 * min(dx,dy)/1000)
        min_time_step      = NINT(3 * min(dx,dy)/1000)

    Note the starting step is **4*dx(km)**, not the 6*dx often quoted.
    6*dx is the fixed-clock rule of thumb; the adaptive path has its own
    numbers and they are wider on both sides.

    Returns ``(starting, max, min)`` in whole seconds.
    """
    km = min(dx, dy) / 1000.0
    return (nint(4 * km), nint(8 * km), nint(3 * km))


def wrf_num_sound_steps(dt: float, dx: float, dy: float,
                        max_msft: float = 1.0) -> int:
    """WRF's DYNAMIC acoustic substep count (``dyn_em/solve_em.F:451-463``).

        num_sound_steps = max(2*(INT(300*dt/(spacing/max_msft) - 0.01)+1), 4)

    Upstream sets ``time_step_sound = 0`` whenever the adaptive clock is
    on (``start_em.F:966``) precisely so this runs: a fixed substep count
    under a growing dt walks the ACOUSTIC Courant up until the split
    scheme fails, and that failure does not look like a timestep problem.

    MEASURED, and this is why it is here: with time_step_sound pinned at
    4, a live adaptive run grew dt to ~76 s on a 10 km grid -- an acoustic
    substep of 19 s -- and died at outer step 20 with "MYJ returned
    non-finite rublten".  The formula gives 6 there, for a 12.7 s substep.
    Upstream's own comment: "gives 4 for 6*dx and 6 for 10*dx, and returns
    even numbers only".
    """
    spacing = min(dx, dy) / max(max_msft, 1.0e-6)
    return max(2 * (int(300.0 * dt / spacing - 0.01) + 1), 4)


def maximum_map_factor(state=None, geography=None) -> float:
    """The static full-domain factor used by WRF's acoustic count."""
    best = 1.0
    for name in ("msfu", "msfv"):
        arr = (geography.get("setup/" + name, geography.get(name))
               if geography is not None else getattr(state, name, None))
        if arr is not None:
            value = float(arr.max())
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must have a finite positive maximum")
            best = max(best, value)
    return best


def acoustic_step_ceiling(cfg, max_map_factor: float = 1.0) -> int:
    """Reserve the acoustic reach of every configured adaptive timestep.

    This is geometry planning only: the live clock still computes its actual
    count at every step. A cold caller without geography uses unit map factors;
    the live domain planner resolves the full static maximum before allocation.
    """
    if not bool(getattr(cfg, "use_adaptive_time_step", False)):
        return int(cfg.time_step_sound)
    upper = _interval(cfg.max_time_step, cfg.max_time_step_den)
    if upper is None:
        upper = Fraction(wrf_default_clamps(cfg.dx, cfg.dy)[1])
    start = _interval(cfg.starting_time_step, cfg.starting_time_step_den)
    largest = max(float(cfg.dt), float(upper), float(start or 0))
    return max(int(cfg.time_step_sound),
               wrf_num_sound_steps(largest, cfg.dx, cfg.dy, max_map_factor))


#: How many substeps a nest may take inside one parent step before the
#: divide is treated as broken rather than demanding.
#:
#: A nest's nominal count is its ``parent_time_step_ratio``.  Needing a
#: few times that is a real CFL response; needing hundreds is the divisor
#: collapse this module already documents, and the difference matters
#: because the two look identical from outside -- the run does not fail,
#: it silently costs the factor in throughput.  MEASURED on a 10/2/0.667
#: km tree: a 3:1 nest reached **373** substeps in one parent step, at
#: which point the run made 0.23 model-seconds per wall-second against
#: 11.5 an hour earlier, and every field was healthy.  Read as a CFL
#: response it looks like a diverging storm; it was arithmetic.
MAX_NEST_SUBSTEP_FACTOR = 8

#: Extra factor the ROOT's tick count carries beyond the ratio lattice,
#: so the quotients handed down are divisible by more than the ratios.
#:
#: A power of two, because the divisors it contributes (2 and 4) combine
#: with the lattice's own factors to give the nests a ladder to adapt
#: along instead of a single rung.  4 costs the root 0.6 s of step
#: resolution on a lattice-15 tree and removes the collapse; larger
#: values coarsen the root for no further benefit, since what matters is
#: having SOME interior divisor near each ceiling, not many.
_ROOT_SMOOTH_FACTOR = 4


class NestDivideRefusal(RuntimeError):
    """The nest divide cannot represent the step the controller asked for."""


def nest_ticks_from_parent(parent_ticks: int, own_ticks: int, *,
                           subtree_lattice: int = 1,
                           max_substeps: int | None = None,
                           grid_id: int | None = None) -> int:
    """A nest's step in TICKS, dividing its parent's exactly.

    WRF takes ``num_small_steps = CEILING(parent%dt / dt)`` and then
    ``parent_dt / num_small_steps`` as an exact ESMF rational
    (adapt_timestep_em.F:232-238).  Fine there; not here.  ArWen's clock
    is an integer tick lattice, so the quotient has to land on it.

    ``n`` is therefore the smallest DIVISOR of ``parent_ticks`` that is at
    least the ceiling -- not the ceiling raised until it happens to
    divide.  Those are the same thing only when the parent's tick count is
    rich in factors, and the difference is not subtle:

      a 33.08 s parent is 3308 ticks, whose divisors are
      1, 2, 4, 827, 1654, 3308.  Raising n from 5 gets to 827, which
      collapses a 6.6 s nest step to 0.04 s -- 827 child steps inside one
      parent step.  MEASURED: that killed a live run at outer step 4 with
      "MYJ returned non-finite rublten".

    The real fix is upstream of this, in
    :meth:`AdaptiveClockDriver._quantise_root`: the ROOT's step is snapped
    to a multiple of the tree's ratio lattice, so a divisor near the
    ceiling always exists.  This function stays exact about what it can
    do alone, and the caller guarantees it is asked a reasonable question.
    """
    if own_ticks <= 0:
        raise ValueError(f"nest step must be positive, got {own_ticks}")
    if parent_ticks <= 0:
        raise ValueError(f"parent step must be positive, got {parent_ticks}")
    want = max(1, -((-parent_ticks) // own_ticks))        # CEILING

    # PREFER a divisor that leaves THIS nest divisor-rich for its OWN
    # children.  ``_quantise_root`` gives that guarantee to the root, and
    # on a two-level tree that is enough.  On three levels the middle
    # domain is a parent AND adapts, so its tick count is whatever the
    # divide happened to produce -- and when that lands somewhere poor in
    # factors, the smallest divisor at or above the next ceiling jumps
    # enormously.  Measured: d02 at 11.19 s handing d03 0.03 s, a ratio of
    # 373 where the grid ratio is 3.
    #
    # A PREFERENCE, NEVER A REFUSAL.  The lattice-friendly divisor is the
    # one that leaves this nest's OWN children a divide; when it is the
    # only one considered and it happens to be huge, the run dies on a
    # message naming a collapse that is not happening -- the plain divide
    # had a small, legal answer sitting right there.  MEASURED on this
    # tree: parent 246 ticks, child asking 62, lattice 3 -- the smallest
    # lattice-friendly divisor is 41, past a ceiling of 24, while the
    # plain divide answers 6.  So the preference is taken only while it
    # is affordable, and the plain loop below decides the rest.  Because
    # ``n`` only grows, the FIRST lattice-friendly divisor is the
    # smallest, and if that one is unaffordable every later one is too.
    lattice = max(1, int(subtree_lattice))
    if lattice > 1:
        for n in range(want, parent_ticks + 1):
            if parent_ticks % n:
                continue
            child = parent_ticks // n
            if child % lattice == 0:
                if max_substeps is not None and n > max_substeps:
                    break
                return _checked_substeps(n, max_substeps, grid_id,
                                         parent_ticks, child)
    # No lattice-friendly divisor exists; take the smallest one rather
    # than fail, and let the substep bound below decide whether what it
    # produced is usable.
    for n in range(want, parent_ticks + 1):
        if parent_ticks % n == 0:
            return _checked_substeps(n, max_substeps, grid_id,
                                     parent_ticks, parent_ticks // n)
    return 1                                              # n == parent_ticks


def _checked_substeps(n: int, max_substeps, grid_id, parent_ticks: int,
                      child_ticks: int) -> int:
    """Refuse a substep count that means the divide has collapsed.

    Loud rather than slow.  Without this the run continues and merely
    costs the factor, which reads from outside as the controller
    responding to a violent flow -- the failure mode that cost a whole
    diagnosis, because every field was healthy while the clock was not.
    """
    if max_substeps is not None and n > max_substeps:
        where = "" if grid_id is None else f" for grid_id={grid_id}"
        raise NestDivideRefusal(
            f"nest divide{where} needs {n} substeps inside one parent step "
            f"(parent {parent_ticks} ticks -> child {child_ticks}), past the "
            f"limit of {max_substeps}.  The parent's tick count is poor in "
            f"factors, so the smallest usable divisor is far above the "
            f"ceiling the CFL asked for -- this is an arithmetic collapse, "
            f"not a timestep the flow requires.  THE BREAKAGE THIS "
            f"PREVENTS, measured on a 10/2/0.667 km tree: a parent of 1119 "
            f"ticks (3 x 373) had no divisor between 3 and 373, so a 3:1 "
            f"nest asking for about 1 s took 373 substeps at 0.03 s and the "
            f"run fell from 11.5 model-seconds per wall-second to 0.23 -- "
            f"without failing, and with every field healthy, so it read "
            f"from outside as a diverging storm for hours.  Raise "
            f"min_time_step, or give the tree ratios whose lattice divides "
            f"the parent's step.")
    return child_ticks


class AdaptiveClockDriver:
    """``on_period_steps(period, clocks)`` for :func:`execute_schedule`.

    One period is one root step, so this runs once per root step and sets
    every domain's step for the period about to be expanded.
    """

    def __init__(self, model, *, cfl_source, tick_den: int,
                 restart_dt: dict[int, Fraction] | None = None,
                 carrier_source=None, map_factor_source=None):
        self.model = model
        self.cfl_source = cfl_source
        self.carrier_source = carrier_source
        self.tick_den = int(tick_den)
        self.controllers: dict[int, AdaptiveTimestepController] = {}
        self.order: list[int] = []
        restart_dt = restart_dt or {}

        # Parent before child: a nest's step is derived from its parent's,
        # so the parent must have chosen before the child asks.
        def walk(node):
            self.order.append(int(node.cfg.grid_id))
            for kid in node.children:
                walk(kid)

        walk(model.root)

        # THE RATIO LATTICE.  The root's step is snapped to a multiple of
        # this so every nest can find a divisor near the step IT wants.
        # Without it the root is free to land on a tick count with no
        # useful factors -- 3308 has none between 4 and 827 -- and the
        # nest's step collapses by two orders of magnitude.  Built from
        # the CONFIGURED ratios, which is what the tree was designed
        # around, and floored at 1 for a single domain.
        ratios = []

        def gather(node):
            for kid in node.children:
                ratios.append(max(1, int(getattr(
                    kid.cfg, 'parent_time_step_ratio', 1) or 1)))
                gather(kid)

        gather(model.root)
        self.nest_lattice = lcm(*ratios) if ratios else 1
        #: The extra smoothing :meth:`_quantise_root` applies, and the
        #: reason it is a field rather than the constant: a tree with NO
        #: nests has no divide to keep rich in factors, so coarsening its
        #: root buys nothing and costs resolution.  MEASURED: a
        #: single-domain adaptive run asking 22.53 s was handed 22.52 s,
        #: and an explicit ``starting_time_step`` was honoured only to
        #: within four ticks -- a step the flow did not ask for, to serve
        #: children that do not exist.
        self.root_smooth_factor = _ROOT_SMOOTH_FACTOR if ratios else 1

        #: grid_id -> the lattice this domain's DESCENDANTS need it to
        #: land on.  ``_quantise_root`` gives the root a divisor-rich tick
        #: count; on a two-level tree that is the whole guarantee, because
        #: the only other domain has no children.  Add a third level and
        #: the middle domain is a parent AND adapts, so its ticks are
        #: whatever the divide produced -- and nothing was keeping THOSE
        #: rich in factors.  This is what lets the divide prefer a
        #: quotient its own children can still divide.
        self.subtree_lattice: dict[int, int] = {}

        def below(node) -> int:
            acc = 1
            for kid in node.children:
                ratio = max(1, int(getattr(
                    kid.cfg, "parent_time_step_ratio", 1) or 1))
                acc = lcm(acc, ratio, below(kid))
            self.subtree_lattice[int(node.cfg.grid_id)] = acc
            return acc

        below(model.root)

        # max_msftx/max_msfty, which upstream resolves once in start_em
        # (:984-989) because the map factors are static.
        self.max_msft: dict[int, float] = {}
        for gid in self.order:
            actual = None if map_factor_source is None else map_factor_source(gid)
            self.max_msft[gid] = (maximum_map_factor(model.node(gid).state)
                                  if actual is None else float(actual))

        #: grid_id -> the model second radiation was last seen to write,
        #: and the interval it actually ran at.  See _observe_radiation.
        self._radiation_seen: dict[int, float] = {}
        self._radiation_actual: dict[int, float] = {}
        #: grid_id -> the model second this driver last told cumulus to
        #: fire.  Cumulus gets the same TIME cadence radiation does, for
        #: the same reason; see _drive_cumulus_on_time.
        self._cumulus_fired: dict[int, float] = {}
        #: grid_id -> the step the MODEL actually took last period, as
        #: distinct from the step the CONTROLLER proposed.  The two differ
        #: by the root quantiser's floor and the nest divide's ceiling, and
        #: the controller must only ever see its own proposal -- see the
        #: rescale in __call__ (ENG-017).
        self._last_applied: dict[int, Fraction] = {}

        for gid in self.order:
            node = model.node(gid)
            run = node.cfg.run
            wrf_max, wrf_min = wrf_default_clamps(run.dx, run.dy)[1:]
            resumed = getattr(node.clock, "adaptive_state", None)
            if _TRACE:
                print(f"ADTINIT d{gid:02d} resumed={resumed is not None} "
                      f"clock.step_ticks={node.clock.step_ticks} "
                      f"spec.step_ticks={node.clock.spec.step_ticks}",
                      flush=True)
            start = restart_dt.get(gid)
            if start is None and resumed:
                start = Fraction(int(resumed["last_dt_num"]),
                                 int(resumed["last_dt_den"]))
            if start is None:
                # AN EXPLICIT starting_time_step WINS.  Until this line
                # existed it was read by NOTHING: the value was imported
                # from the namelist, validated against min/max, filed in
                # all three identity tables and written into the
                # checkpoint header, and the run then took the configured
                # time_step and said nothing.  Measured on a 3 km case --
                # starting_time_step = 20 produced a run identical to -1
                # in step count, first dt and every output frame.
                start = _interval(run.starting_time_step,
                                  run.starting_time_step_den)
            if start is None:
                # -1 IS A NAMED DIVERGENCE (docs/ADAPTIVE-TIMESTEP.md,
                # section "What is ruled out"): upstream substitutes
                # NINT(4*dx km) here (start_em.F:939-941); gpuwm keeps
                # the CONFIGURED time_step.  A prepared tree has its
                # whole cadence lattice -- output alarms, nest step
                # ratios, the boundary interval -- built around that step
                # and bound into its prepared identity, so a first period
                # taken at some other step is the one period nothing was
                # prepared for.  min and max keep WRF's -1 fill-ins
                # below, where no lattice is involved.
                start = Fraction(node.clock.spec.step_ticks, self.tick_den)
            min_dt = _interval(run.min_time_step, run.min_time_step_den)
            max_dt = _interval(run.max_time_step, run.max_time_step_den)
            if min_dt is None:
                min_dt = Fraction(wrf_min)
            if max_dt is None:
                max_dt = Fraction(wrf_max)
            ctl = AdaptiveTimestepController(
                target_cfl=run.target_cfl, target_hcfl=run.target_hcfl,
                max_step_increase_pct=run.max_step_increase_pct,
                starting_dt=start, min_dt=min_dt, max_dt=max_dt)
            if resumed:
                if resumed.get("radiation_seen") is not None:
                    self._radiation_seen[gid] = float(
                        resumed["radiation_seen"])
                if resumed.get("radiation_actual") is not None:
                    self._radiation_actual[gid] = float(
                        resumed["radiation_actual"])
                # THE CUMULUS CADENCE MEMORY.  Absent, the first resumed
                # step takes _drive_cumulus_on_time's `last is None` arm,
                # fires cumulus NOW and re-phases the whole cudt cadence to
                # the resume instant -- a silent trajectory divergence for
                # every cu_physics = 1 run with cudt_minutes > 0 (the
                # default 5.0).  Carried beside the radiation twin, which
                # was carried for exactly this reason (ENG-019).
                if resumed.get("cumulus_fired") is not None:
                    self._cumulus_fired[gid] = float(resumed["cumulus_fired"])
                # THE CFL MEMORY, which WRF does not checkpoint.  Without
                # it the first step after a resume reads cfl ~ 0, takes
                # calc_dt's `max_cfl < 0.001` branch, grows by the full
                # increase factor and overshoots -- the failure the MOVING
                # tree patches by reusing last_dtInterval for one step.
                # Carrying it makes that patch unnecessary here: the
                # controller simply continues.
                ctl.last_max_vert_cfl = float(resumed["last_max_vert_cfl"])
                ctl.last_max_horiz_cfl = float(resumed["last_max_horiz_cfl"])
                ctl.stepping_to_time = bool(resumed["stepping_to_time"])
                ctl.started = bool(resumed["started"])
                # The step the checkpoint's last period actually took, so
                # the first resumed CFL is rescaled the way every other
                # period's is.
                if int(node.clock.step_ticks) > 0:
                    self._last_applied[gid] = Fraction(
                        int(node.clock.step_ticks), self.tick_den)
            self.controllers[gid] = ctl

    # -- the executor's hook ------------------------------------------

    def __call__(self, period: int, clocks) -> None:
        # Whether the ROOT shortened this period to land on a time.  A
        # nest's step is parent_dt/n, so when the parent shortens the
        # nest shortens WITH it -- for the parent's reason, not its own.
        # Recording that as a normal step lets the nest's baseline
        # ratchet down, which is the exact `use_last2` failure the
        # controller exists to prevent, arriving through the back door.
        #
        # MEASURED: the root shortened to hit the 720 s frame, d02 went
        # 9.44 -> 2.23 s, and d02's last_dt BECAME 2.23.  Its radiation
        # cadence was then re-derived as 161 steps, and gpuwm's carrier
        # gate refused a 361.91 s gap against a 359.03 s tolerance.
        #
        # WRF has the same ratchet -- its nest takes parent_dt/n and
        # stores it -- and does not notice, because its stepra is frozen
        # at init and it has no freshness contract.  gpuwm has both, so
        # the parent's flag is propagated.
        root_stepping = False
        for gid in self.order:
            node = self.model.node(gid)
            clock = clocks[gid]
            # A NEST THAT HAS NOT STARTED IS NOT DRIVEN.  The executor
            # hands this hook EVERY domain's clock (execute_schedule
            # validates the dict against the whole schedule), and parks an
            # unstarted nest's clock at each period boundary until the
            # boundary reaches its start_ticks -- so "not yet started" is
            # `ticks < spec.start_ticks`, never a missing key.  The
            # previous `clocks.get(gid) is None` test was a branch the
            # executor could not reach: from period 1 the nest's controller
            # was fed cfl_source's (0, 0) for a domain that had folded
            # nothing, took calc_dt's negligible-CFL branch every period,
            # and reached its first solve at max_time_step -- its parent's
            # step, up to parent_time_step_ratio times its own -- with
            # node.cfg.run.dt already rewritten to that value for
            # initialize_child to read (ENG-018).  Skipped here, the nest
            # enters through `first_step` on its activation period with its
            # configured ratio step and is divided into its parent's step
            # like any other period.
            if int(clock.ticks) < int(clock.spec.start_ticks):
                continue
            ctl = self.controllers[gid]
            vert, horiz = self.cfl_source(gid)
            # THE CONTROLLER SEES ITS OWN PROPOSAL, NEVER THE QUANTISED
            # STEP.  The CFL just measured was measured on the step the
            # model TOOK -- the proposal floored to the root lattice, or
            # the parent's step divided by a ceiling -- while `last_dt` is
            # what the controller PROPOSED.  CFL is linear in dt, so the
            # measurement is rescaled to the proposal and the pair
            # (last_dt, cfl) the controller reasons from is consistent.
            #
            # Feeding it the applied step instead made the floor a
            # permanent debit: below 0.8 * lattice seconds the 5 % growth
            # the controller asked for was smaller than one lattice unit,
            # the floor erased it, and `accept` stored the floored value
            # as the new baseline -- so one CFL spike parked dt at
            # min_time_step for the rest of the run, with every field
            # healthy (ENG-017).  Skipped after a step shortened to land
            # on a time: next_dt reaches back to the previous memory there
            # and the vertical clobber keeps upstream's asymmetry as
            # transcribed.
            applied_before = self._last_applied.get(gid)
            if (ctl.started and not ctl.stepping_to_time
                    and applied_before is not None and applied_before > 0
                    and applied_before != ctl.last_dt):
                scale = float(ctl.last_dt / applied_before)
                vert, horiz = vert * scale, horiz * scale

            # NOT `period == 0 and ...`: a nest whose first period is
            # not zero -- any domain with start_ticks > 0 -- would never
            # take its first step through this arm, so `started` stayed
            # false for the life of the run and `limit_increase`
            # (adapt_timestep_em.F:168-177) was never applied to it.  It
            # was also driven from its first period with cfl_source
            # returning (0, 0), which is calc_dt's negligible-CFL branch,
            # so it ratcheted straight to max_dt before its first solve.
            if not ctl.started:
                dt = ctl.first_step()
                stepping = False
            else:
                dt = ctl.next_dt(max_vert_cfl=vert, max_horiz_cfl=horiz)
                stepping = False
                if node.parent is None:
                    # step_to_output_time is upstream's, and upstream
                    # gates it `.not. grid%nested` (:325) -- a nest lands
                    # on its parent's boundary by dividing it, not by
                    # shortening itself.
                    dt, stepping = ctl.step_to_time(
                        dt, self._to_next_alarm_anywhere(clocks),
                        quantise=self._quantise_root)
                    root_stepping = stepping
                    # THE RUN END IS A TIME TO LAND ON TOO.  The executor
                    # loops `while root.ticks < run_ticks` and takes
                    # whatever step the controller offers, so without this
                    # the last one steps straight over the finish: a run
                    # ended at 183385 ticks against a run_ticks of 180000
                    # and was refused, correctly, by the tick-exact stop
                    # check.  WRF gets this from its stop-time alarm; here
                    # it is the same clamp step_to_time already applies to
                    # an output frame, on a different deadline.
                    left = Fraction(
                        max(0, clock.run_ticks - clock.ticks),
                        self.tick_den)
                    if 0 < left < dt:
                        dt = left
                        stepping = True
                        root_stepping = True

            # The floor is checked on the CONTROLLER's own step, before
            # the nest divide, so a diverging run is named here rather
            # than surfacing as "nest step must be positive" two calls
            # deeper with no mention of the remedy.
            self._refuse_sub_tick(node, dt)
            # What the controller asked for, before the lattice and the
            # divide have their say.  This, not the applied step, is what
            # the controller is handed back below.
            proposed = dt

            if node.parent is None:
                dt = self._quantise_root(dt)

            if node.parent is not None:
                parent_ticks = clocks[
                    int(node.parent.cfg.grid_id)].step_ticks
                own_gid = int(node.cfg.grid_id)
                ticks = nest_ticks_from_parent(
                    parent_ticks, ticks_of(dt, self.tick_den),
                    subtree_lattice=self.subtree_lattice.get(own_gid, 1),
                    max_substeps=MAX_NEST_SUBSTEP_FACTOR * max(1, int(
                        getattr(node.cfg, "parent_time_step_ratio", 1) or 1)),
                    grid_id=own_gid)
                dt = Fraction(ticks, self.tick_den)

            self._apply(node, clock, dt, baseline=ctl.last_dt)
            self._last_applied[gid] = dt
            if _TRACE:
                print(f"ADT p{period:04d} d{gid:02d} cfl_v={vert:.4f} "
                      f"cfl_h={horiz:.4f} dt={float(dt):.4f} "
                      f"proposed={float(proposed):.4f} "
                      f"ticks={clock.step_ticks}", flush=True)
            # The PROPOSAL is committed as the baseline; the model took
            # `dt`.  See the rescale at the top of this loop for why the
            # two are kept apart.
            ctl.accept(proposed, max_vert_cfl=vert, max_horiz_cfl=horiz,
                       stepping_to_time=(stepping or (
                           root_stepping and node.parent is not None)))
            # PUBLISH the controller's memory where the checkpoint writer
            # can see it.  WRF stores last_dtInterval and NOT the CFL
            # memory, which is why its MOVING tree needs a special first
            # step after a restart (docs/ADAPTIVE-TIMESTEP.md section 5).
            # Storing both makes the resume ordinary instead of special.
            clock.adaptive_state = self._published_state(gid, ctl)

    def _published_state(self, gid: int, ctl) -> dict:
        """The controller's AND the driver's memory, for the checkpoint.

        Written at the top of every period and again after every
        ``before_step``: the driver's own memory (radiation seen, cumulus
        fired) moves INSIDE the period, and a checkpoint is written at the
        period's end -- a state published only at the period's start
        carried the previous period's memory (ENG-019).
        """
        return {
            "last_dt_num": ctl.last_dt.numerator,
            "last_dt_den": ctl.last_dt.denominator,
            "last_max_vert_cfl": float(ctl.last_max_vert_cfl),
            "last_max_horiz_cfl": float(ctl.last_max_horiz_cfl),
            "stepping_to_time": bool(ctl.stepping_to_time),
            "started": bool(ctl.started),
            # THE DRIVER'S OWN MEMORY, not the controller's.  Without
            # it a resume cannot drive radiation on time -- it has no
            # idea when the producer last ran -- and falls back to the
            # step-count predicate, which fires at a different instant
            # and diverges the trajectory.  Measured: dt reproduced
            # EXACTLY across a resume and the fields still differed,
            # which is what pointed here.
            "radiation_seen": self._radiation_seen.get(gid),
            "radiation_actual": self._radiation_actual.get(gid),
            # The cumulus twin, for the same reason; see __init__.
            "cumulus_fired": self._cumulus_fired.get(gid),
        }

    def before_step(self, grid_id: int) -> None:
        """Re-assert the physics cadence immediately before a solve.

        The period-start refresh is not enough on its own: RELOCATION
        rebuilds the physics driver MID-PERIOD (``state.physics = None``),
        and the rebuilt driver derives ``stepra`` from whatever
        ``cfg.dt`` happens to be -- which, on a step shortened to land on
        an output frame, is a step nothing else will ever take.  MEASURED:
        d02 dropped 9.44 -> 2.23 s at the 720 s frame, a relocation
        rebuilt there, and stepra came out 161 instead of ~38.

        Called from ``on_step``, which runs after any relocation for that
        domain and before its solve, so it is the last word.  Idempotent
        and cheap -- two integer divisions on a Python object.
        """
        ctl = self.controllers.get(int(grid_id))
        if ctl is None:
            return
        node = self.model.node(grid_id)
        # THE RETURN VALUE IS THE POINT.  _observe_radiation measures
        # the interval the producer actually took, and
        # _refresh_physics_cadence takes the larger of that and its own
        # prediction -- the jitter cover its docstring argues for.  The
        # observation used to be computed here and dropped on the floor,
        # which left the `radt_seconds = max(predicted, observed)` arm
        # unreachable and the carrier gate refusing runs in which nothing
        # had stopped.
        observed = self._observe_radiation(int(grid_id), node)
        _refresh_physics_cadence(node, ctl.last_dt, observed=observed)
        self._drive_radiation_on_time(int(grid_id), node)
        self._drive_cumulus_on_time(int(grid_id), node)
        # The driver's memory just moved; re-publish so a checkpoint
        # written at this period's end carries it (ENG-019).
        node.clock.adaptive_state = self._published_state(int(grid_id), ctl)

    def _drive_radiation_on_time(self, grid_id: int, node) -> None:
        """Fire radiation on a TIME cadence, not a step count.

        ``_radiation_step_due`` is ``itimestep % stepra == 1``.  Under a
        varying dt that phase MOVES whenever stepra is re-derived, so the
        interval jitters: measured 373.06 s where the target was 358, a
        15 s overshoot that a one-step tolerance cannot cover.  Freezing
        stepra removes the jitter and replaces it with a worse problem --
        the cadence in seconds then drifts with dt, and radiation went
        from every 360 s to every 660 s.

        Neither step-count arrangement holds a cadence the sun cares
        about.  So under an adaptive clock the decision is made HERE, on
        elapsed model time, and handed to the existing predicate in the
        only language it speaks: ``stepra = 1`` means "fire now" (the
        predicate short-circuits on it) and a large value means "not
        yet".  No change to the physics driver, no second firing path,
        and the interval becomes target + at most one step -- which is
        exactly the tolerance the carrier contract already allows.

        A DELIBERATE DIVERGENCE FROM WRF, and the better answer: WRF's
        frozen STEPRA lets its radiation cadence stretch with dt, which
        it does not notice because it has no freshness contract.  gpuwm
        does, and radiation should follow the sun rather than the step
        counter.
        """
        physics = getattr(node.state, "physics", None)
        if physics is None or not hasattr(physics, "stepra"):
            return
        target = float(getattr(physics, "radt_seconds", 0.0) or 0.0)
        if target <= 0.0:
            return                      # radt = 0: every step, leave alone
        now = float(node.clock.elapsed_seconds)
        last = self._radiation_seen.get(grid_id)
        if last is None:
            return                      # before the first write: WRF fires
        step = float(node.cfg.run.dt)
        # Due when the NEXT step would carry us past the target, so the
        # interval lands at or just under it rather than just over.
        # Set the OVERRIDE, not stepra: compute() re-derives stepra from
        # cfg.dt on every call (physics.py:4290), so anything written to
        # it here is gone before the predicate reads it.  Found by setting
        # stepra and watching nothing change.
        physics.radiation_due_override = (now + step - last) >= target

    def _drive_cumulus_on_time(self, grid_id: int, node) -> None:
        """Fire cumulus on a TIME cadence too, for radiation's reason.

        ``_cumulus_step_due`` is ``itimestep % stepcu == 0``, and under an
        adaptive clock BOTH of its inputs stop meaning what they say:
        ``itimestep`` is reconstructed in ``compute`` as
        ``elapsed/cfg.dt``, which is not the step count when the steps
        were different sizes, and ``stepcu`` is re-derived from the
        momentary dt on every call (``core/physics.py:4422``, which
        overwrites anything ``_refresh_physics_cadence`` wrote).  A phase
        condition on two drifting numbers fires irregularly.

        That is the SAME defect this patch fixes for radiation, on the
        same predicate, and leaving cumulus out of the fix would have
        shipped it half done.  The decision is made here on elapsed model
        time and handed to the driver as an override, exactly as for
        radiation: no second firing path, and the interval becomes the
        target plus at most one step.

        The first call fires, which is what WRF's ``itimestep == 1`` arm
        does; from then on the driver holds ``cudt``.
        """
        physics = getattr(node.state, "physics", None)
        if physics is None or not hasattr(physics, "stepcu"):
            return
        if not bool(node.cfg.run.cu_physics):
            return
        target = float(getattr(physics, "cudt_seconds", 0.0) or 0.0)
        if target <= 0.0:
            return                      # cudt = 0: every step, leave alone
        now = float(node.clock.elapsed_seconds)
        step = float(node.cfg.run.dt)
        last = self._cumulus_fired.get(grid_id)
        if last is None:
            physics.cumulus_due_override = True
            self._cumulus_fired[grid_id] = now
            return
        due = (now + step - last) >= target
        physics.cumulus_due_override = due
        if due:
            self._cumulus_fired[grid_id] = now

    def _observe_radiation(self, grid_id: int, node) -> float | None:
        """The interval radiation ACTUALLY ran at, from the carrier record.

        gpuwm's carrier gate asks whether a produced field is older than
        one radiation cadence plus a step.  It is handed ``stepra * dt``
        as that cadence, which IS the exact gap under a fixed clock and is
        only a PREDICTION under a varying one: the true gap is the SUM of
        dt over the interval, so whenever dt fell during it the sum
        exceeds the prediction.  Measured: 675.65 s of gap against a
        668.56 s tolerance -- a 1% miss that refused a run in which
        nothing had stopped.

        The record already knows the answer.  Watching
        ``last_update_model_time`` change gives the interval the producer
        is really running at, with no new bookkeeping and no change to the
        contract's semantics: a producer that STOPS still fails, because
        its observed interval stays at the last good value while the age
        grows past it.  That is the property the gate exists for, and it
        is preserved exactly.

        Returns None until two writes have been seen.
        """
        source = getattr(self, "carrier_source", None)
        live = None if source is None else source(grid_id)
        if live is None:
            carriers = getattr(getattr(node.state, "physics", None),
                               "carriers", None)
            if carriers is None:
                return self._radiation_actual.get(grid_id)
            records = getattr(carriers, "records", {}).values()
        else:
            records = live["records"].values()
        latest = None
        for rec in records:
            t = (rec.get("last_update_model_time") if isinstance(rec, dict)
                 else getattr(rec, "last_update_model_time", None))
            if t is not None and (latest is None or t > latest):
                latest = float(t)
        if latest is not None:
            previous = self._radiation_seen.get(grid_id)
            if previous is not None and latest > previous:
                self._radiation_actual[grid_id] = latest - previous
            if previous is None or latest > previous:
                self._radiation_seen[grid_id] = latest
        return self._radiation_actual.get(grid_id)

    # -- the three things that must move together ---------------------

    def _refuse_sub_tick(self, node, dt: Fraction) -> None:
        if ticks_of(dt, self.tick_den) > 0:
            return
        # THE TICK LATTICE HAS A FLOOR AND WRF'S RATIONAL CLOCK DOES NOT.
        # Upstream can shrink dt without bound; here the smallest
        # representable step is 1/tick_den s, and a controller asking for
        # less has to be refused rather than clamped -- a silent clamp
        # would let a diverging run grind along at the floor looking
        # healthy.  min_time_step is the user-facing guard and the
        # message names it.
        raise ValueError(
            f"adaptive dt for grid_id={node.cfg.grid_id} fell to "
            f"{float(dt):g} s, below this clock's resolution of "
            f"1/{self.tick_den} s.  The CFL is driving the controller "
            f"past what the tick lattice can represent, which usually "
            f"means the run is diverging; set min_time_step to floor it "
            f"deliberately if that is not what is happening.")

    def _derive_run_field(self, name: str, dt: Fraction, run):
        """One runtime-derived ``RunConfig`` value, by name.

        The dispatch exists so :data:`ADAPTIVE_DERIVED_RUN_FIELDS` can be
        the single authority on WHICH fields the clock overwrites, while
        this says HOW.  An unknown name raises rather than silently
        skipping: a field listed as derived but never computed would be
        exempted from the restart comparison while still holding its
        stale configured value, which is a worse failure than the one
        this replaced.
        """
        if name == "dt":
            return float(dt)
        if name == "time_step_sound":
            return wrf_num_sound_steps(
                float(dt), run.dx, run.dy,
                self.max_msft.get(int(run.grid_id), 1.0))
        raise KeyError(
            f"ADAPTIVE_DERIVED_RUN_FIELDS names {name!r} but "
            f"_derive_run_field does not compute it; add it here or "
            f"remove it from the set -- the restart walk exempts exactly "
            f"this set and must not exempt a field nothing derives")

    def _apply(self, node, clock, dt: Fraction,
               baseline: Fraction | None = None) -> None:
        # ONE copy of the refusal, in _refuse_sub_tick.  This function
        # used to carry a verbatim second copy, unreachable because
        # on_period_steps calls the guard on every path before reaching
        # here -- and a second copy of a message is a second copy to keep
        # true.
        self._refuse_sub_tick(node, dt)
        ticks = ticks_of(dt, self.tick_den)
        clock.step_ticks = ticks
        # The LIVE kernel dt, by WRF's own real_time construction.  This
        # drives dtbc, the nest boundary weight and the Davies relaxation,
        # so leaving it behind forces the boundaries on a different clock
        # from the interior -- silently.
        clock.dt_fp32 = real_time_fp32(dt)
        run = node.cfg.run
        # Built by iterating ADAPTIVE_DERIVED_RUN_FIELDS rather than by
        # writing the names again here.  That is not style: the restart
        # identity walk has to exempt exactly this set, and when the two
        # were written independently they drifted -- `dt` was exempted and
        # `time_step_sound` was not, which made every checkpoint an
        # adaptive run wrote unrestartable as soon as its step grew enough
        # to move the sound-step count.  Deriving a field the constant does
        # not name is now impossible, and naming one without teaching the
        # deriver raises here instead of failing a resume hours later.
        node.cfg = dataclass_replace(node.cfg, run=dataclass_replace(
            run, **{name: self._derive_run_field(name, dt, run)
                    for name in ADAPTIVE_DERIVED_RUN_FIELDS}))
        _refresh_physics_cadence(
            node, baseline or dt,
            observed=self._radiation_actual.get(int(run.grid_id)))

    def _quantise_root(self, dt: Fraction) -> Fraction:
        """Snap the root's step DOWN to a SMOOTH multiple of the lattice.

        Down, never up: rounding up would hand the nests a step the
        parent's CFL did not sanction.  Floored at one lattice unit so a
        shrinking root cannot reach zero here -- the sub-tick refusal in
        :meth:`_apply` is what catches a genuinely diverging run.

        Smooth, not merely on the ratio lattice: being on the lattice
        makes the nominal divide exact and leaves the cofactor free to be
        prime, which is the collapse :data:`MAX_NEST_SUBSTEP_FACTOR`
        refuses.  The lattice is multiplied by :data:`_ROOT_SMOOTH_FACTOR`
        so the quotients handed down carry interior divisors.
        """
        # FLOOR, not ticks_of: this is the one place a fractional tick
        # count legitimately arrives -- step_to_time halves an odd
        # remainder -- and refusing it here would refuse the very value
        # this function exists to fix.  ticks_of stays strict everywhere
        # else, which is what makes it a useful gate.
        ticks = (dt * self.tick_den).__floor__()
        # SMOOTH, not merely on the lattice.  Landing on a multiple of the
        # ratio lattice makes the NOMINAL divide exact and nothing more:
        # the cofactor can still be prime, and then the nests have no
        # interior divisor to move to when the CFL asks for a different
        # step.  Measured on a 10/2/0.667 km tree (lattice 15): a root of
        # 5595 ticks is 15 x 373, so d02 took 1119 = 3 x 373 -- whose only
        # divisors are 1, 3, 373 and 1119.  d03 asked for about 1 s and the
        # smallest divisor at or above the ceiling was 373, handing it
        # 0.03 s and 373 substeps per parent step.  Every field was
        # healthy; the arithmetic was not.
        #
        # Multiplying the lattice by a small power of two costs resolution
        # in the root's step -- 0.6 s here instead of 0.15 s -- and buys
        # every nest a ladder of interior divisors to adapt along.
        lattice = max(1, self.nest_lattice) * self.root_smooth_factor
        snapped = (ticks // lattice) * lattice
        if snapped <= 0:
            snapped = lattice
        return Fraction(snapped, self.tick_den)

    def _to_next_alarm_anywhere(self, clocks) -> Fraction:
        """Seconds to the next history frame due ANYWHERE in the tree.

        The root is the only domain that steps to a time -- upstream
        gates step_to_output_time `.not. grid%nested` (:325) -- so if it
        lands only on its OWN frames, a nest whose interval is finer
        loses every frame that falls between two of the parent's.

        MEASURED: d01 at 720 s and d02 at 360 s, adaptive, produced 7 of
        the 9 expected frames.  All three d01 frames landed; d02's
        18:06 and 18:18 -- its own times, sitting mid-parent-step --
        were silently skipped, because every executor alarm is
        `ticks % interval == 0` with no at-or-past arm.  The run PASSED
        and quietly wrote less output than it was asked for, which is
        the worst way for this to fail.

        Under a FIXED clock the nest lands on those times by dividing the
        parent's step, so this question never comes up.  It is a
        tree-shaped consequence of a varying dt, and the fix is for the
        root to respect the whole tree's calendar.

        All domains are tick-synchronised at a period boundary -- the
        executor asserts it -- so a single minimum over their next due
        ticks is well defined here.
        """
        now = None
        best = None
        for clock in clocks.values():
            if now is None:
                now = clock.ticks
            # EVERY exact-modulo alarm, not just history.  restart_due and
            # lbc_reset_due are `ticks % interval == 0` on the same terms
            # (clock.py:441-455), so a checkpoint or a boundary seam the
            # clock steps over is not late -- it never happens.  A resume
            # cannot be tested at all if the checkpoint is never written.
            for interval, phased in (
                    (clock.spec.history_ticks, True),
                    (clock.spec.restart_ticks, False),
                    (clock.spec.lbc_interval_ticks, False)):
                if not interval:
                    continue
                origin = clock.spec.start_ticks if phased else 0
                elapsed = clock.ticks - origin
                if elapsed < 0:
                    continue
                remaining = (-elapsed) % interval
                if remaining == 0:
                    remaining = interval
                due = clock.ticks + remaining
                if best is None or due < best:
                    best = due
        if best is None or now is None:
            return Fraction(0)
        return Fraction(best - now, self.tick_den)


def _interval(whole: int, den: int):
    """``min``/``max_time_step`` as a Fraction, or None when unset."""
    if whole == -1 and den == 0:
        return None
    return Fraction(whole) if den == 0 else Fraction(whole, den)


def _refresh_physics_cadence(node, baseline: Fraction, *,
                             observed: float | None = None) -> None:
    """Hold the physics cadence in TIME, and tell the contract the truth.

    WRF freezes STEPRA in phys_init and never revisits it, so under an
    adaptive clock its radiation cadence in SECONDS drifts with dt.  This
    port does NOT copy that, and the divergence is deliberate:

      MEASURED with STEPRA frozen -- d02's step grew 6 -> 11 s, stepra
      stayed 60, and radiation went from firing every 360 s to every
      660 s.  gpuwm's carrier gate refused it, and the gate was RIGHT:
      radiation really had become half as frequent, and a longwave flux
      that old no longer answers to the sun.  WRF does not notice because
      it has no such contract.

    So ``stepra``/``stepcu`` are re-derived from the BASELINE step, which
    holds the cadence near ``radt`` however dt moves.  Baseline rather
    than the momentary dt because a step shortened to land on an output
    frame is not a step anything else will take -- deriving a cadence
    from one gave stepra = 161 instead of 38.

    Re-asserted before EVERY solve rather than once a period, because
    relocation rebuilds the physics driver mid-period and the rebuilt one
    derives its own counts from whatever dt is current.

    THE COST is phase jitter: ``_radiation_step_due`` is
    ``itimestep % stepra == 1``, so a stepra that moves shifts when the
    next call lands, and an interval can run a few seconds long.  That is
    what ``observed`` is for -- the interval the producer was last
    measured to take, which covers the jitter exactly where a prediction
    cannot.  The caller must actually pass it: the driver used to compute
    the observation and discard it, which left the ``max(predicted,
    observed)`` arm below unreachable.
    """
    physics = getattr(node.state, "physics", None)
    if physics is None:
        return
    from gpuwm.core.physics import (_physics_interval_seconds,
                                    _physics_interval_steps)

    # bldt_seconds IS THE TIMESTEP, not a cadence, and it must track the
    # MOMENTARY dt rather than the baseline.  physics.py:2017 sets it once
    # in __init__ as _physics_interval_seconds(cfg.bldt, cfg.dt), and with
    # bldt = 0 -- the default, and what every config in this tree uses --
    # that IS cfg.dt.  It is then handed to MYJ as dtturbl (:3089), to
    # Noah (:3139), to the surface layer (:3175, :3215) and to the 10 m
    # diagnostics (:3369, :3436).
    #
    # Frozen at construction, the PBL and the land surface integrate on
    # whatever dt the driver happened to be BUILT with, for the whole run
    # -- and because relocation rebuilds the driver, they silently jump to
    # a new one at each move.  That is a correctness bug in the RUNNING
    # model, not only across a resume; the resume is merely where it
    # became visible, because a continuous run and a resumed one build the
    # driver at different instants and so freeze different steps.
    #
    # FOUND by digesting every field before every solve: at the resume
    # instant all 153 tracked arrays matched on both domains, and one
    # nest step later the first things to differ were exactly Noah's and
    # MYJ's outputs -- smois, tslb, tke_myj, el_myj, hfx, lh -- with the
    # dynamics following.  Fixing it made the PARENT byte-identical
    # across a resume on its own.
    current = float(node.cfg.run.dt)
    if current > 0.0:
        physics.bldt_seconds = _physics_interval_seconds(
            node.cfg.run.bldt, current)
        if hasattr(physics, "stepbl"):
            physics.stepbl = max(
                int(round(physics.bldt_seconds / current)), 1)

    dt = float(baseline)
    if dt <= 0.0:
        return
    for steps_attr, minutes_attr, seconds_attr in (
            ("stepra", "radt_minutes", "radt_seconds"),
            ("stepcu", "cudt_minutes", "cudt_seconds")):
        if not hasattr(physics, steps_attr):
            continue
        minutes = getattr(physics, minutes_attr, None)
        if minutes is None:
            continue
        steps = _physics_interval_steps(minutes, dt)
        setattr(physics, steps_attr, steps)
        if hasattr(physics, seconds_attr):
            predicted = steps * dt
            # The LARGER of the prediction and what the producer was last
            # measured to do.  The prediction rises with the baseline and
            # so covers a growing dt; the observation remembers the longer
            # interval just taken and so covers the phase jitter.  Neither
            # alone covers both.  A producer that STOPS still fails: its
            # observed interval stays put while the age grows past it.
            if seconds_attr == "radt_seconds" and observed:
                predicted = max(predicted, float(observed))
            setattr(physics, seconds_attr, predicted)
