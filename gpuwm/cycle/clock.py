"""The cycling spine's single clock authority.

One rule shapes this module: **derive from the authority, never from a
peer**.  A child's birth tick is minted here, from the cycle boundary,
not copied off the parent's ``DomainClock``; an analysis time is checked
against the lattice here, and the caller either accepts the observed
offset by name or takes a refusal.  Nothing in the spine snaps silently,
because a silent snap is indistinguishable from a correct time right up
until hour three, when the tree has drifted and no receipt says so.

Within-leg parent/child stepping is NOT reimplemented here.
:func:`gpuwm.core.clock.build_schedule` owns that -- it is transcribed
op-for-op from WRF ``module_integrate.F`` and already asserts tick-exact
sync.  The spine hands it a boundary and checks the answer came back on
that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from gpuwm.cycle.contracts import (CLOCK_SCHEMA, CycleRefusal, TICK_HZ,
                                   TickRatio, seconds_to_ticks,
                                   ticks_to_seconds)


@dataclass(frozen=True, slots=True)
class CycleClock:
    """Cycle boundaries and analysis times, in integer model ticks.

    ``epoch_anchor`` is the only :class:`datetime` in the spine -- it is
    the MPAS init's ``config_start_time``, which the port already treats
    as authoritative.  Everything below it is an integer.
    """

    epoch_anchor: datetime
    parent_step_ticks: int
    cycle_ticks: int
    n_cycles: int

    @classmethod
    def build(cls, *, epoch_anchor: datetime, parent_dt_seconds: float,
              cycle_seconds: float, n_cycles: int) -> "CycleClock":
        """Build a clock, refusing anything that cannot be exact."""
        anchor = _as_utc(epoch_anchor)
        parent_ticks = seconds_to_ticks(parent_dt_seconds,
                                        label="parent dt")
        cycle_ticks = seconds_to_ticks(cycle_seconds, label="cycle length")
        if cycle_ticks % parent_ticks:
            raise CycleRefusal(
                "cycle length is not a whole number of parent steps; "
                "choose a cycle length divisible by the parent dt",
                cycle_seconds=float(cycle_seconds),
                parent_dt_seconds=float(parent_dt_seconds),
                cycle_ticks=cycle_ticks,
                parent_step_ticks=parent_ticks,
                remainder_ticks=cycle_ticks % parent_ticks)
        if int(n_cycles) <= 0:
            raise CycleRefusal("a cycle run needs at least one cycle",
                               n_cycles=int(n_cycles))
        return cls(epoch_anchor=anchor, parent_step_ticks=parent_ticks,
                   cycle_ticks=cycle_ticks, n_cycles=int(n_cycles))

    # -- the lattice -----------------------------------------------------

    def boundary_ticks(self, cycle_index: int) -> int:
        """Ticks since the epoch anchor at boundary ``cycle_index``."""
        index = int(cycle_index)
        if index < 0 or index > self.n_cycles:
            raise CycleRefusal(
                "cycle index is outside this clock's lattice",
                cycle_index=index, n_cycles=self.n_cycles,
                highest_boundary_index=self.n_cycles)
        return index * self.cycle_ticks

    def valid_time(self, cycle_index: int) -> datetime:
        """Wall time of boundary ``cycle_index`` (derived, never stored)."""
        seconds = ticks_to_seconds(self.boundary_ticks(cycle_index))
        return self.epoch_anchor + timedelta(seconds=seconds)

    def steps_per_cycle(self) -> int:
        return self.cycle_ticks // self.parent_step_ticks

    def lattice(self) -> list[tuple[int, int, datetime]]:
        """Every boundary as ``(cycle_index, ticks, valid_time)``."""
        return [(i, self.boundary_ticks(i), self.valid_time(i))
                for i in range(self.n_cycles + 1)]

    # -- analysis times --------------------------------------------------

    def snap(self, when: datetime) -> tuple[int, int]:
        """``(ticks_on_lattice, offset_ticks)`` for a wall time.

        This NEVER snaps silently downstream: it hands back the offset
        and the caller decides.  ``offset_ticks`` is signed -- positive
        when ``when`` is later than the lattice point it snapped to.
        """
        delta = _as_utc(when) - self.epoch_anchor
        exact_ticks = int(round(delta.total_seconds() * TICK_HZ))
        step = self.parent_step_ticks
        # Round-half-away-from-zero onto the parent-step lattice so the
        # reported offset is the smallest one, in either direction.
        quotient, remainder = divmod(exact_ticks, step)
        if remainder * 2 >= step:
            quotient += 1
        on_lattice = quotient * step
        return on_lattice, exact_ticks - on_lattice

    def require_on_lattice(self, when: datetime, *, label: str,
                           accept_offset_ticks: int = 0) -> int:
        """Ticks for ``when``, or a refusal naming the observed offset."""
        on_lattice, offset = self.snap(when)
        if abs(offset) > int(accept_offset_ticks):
            raise CycleRefusal(
                "time does not land on the parent-step lattice; pass "
                "--accept-snap-offset-seconds to accept the offset by name "
                "or move the time onto a step boundary",
                label=label,
                requested=_as_utc(when).isoformat(),
                nearest_lattice_time=(
                    self.epoch_anchor
                    + timedelta(seconds=ticks_to_seconds(on_lattice))
                ).isoformat(),
                offset_seconds=ticks_to_seconds(offset),
                offset_ticks=offset,
                accepted_offset_ticks=int(accept_offset_ticks),
                parent_step_ticks=self.parent_step_ticks)
        return on_lattice

    # -- children --------------------------------------------------------

    def child_ratio(self, child_dt_seconds: float) -> TickRatio:
        """A child's step against the parent's, exactly or not at all."""
        child_ticks = seconds_to_ticks(child_dt_seconds, label="child dt")
        return TickRatio(self.parent_step_ticks, child_ticks)

    def birth_ticks(self, cycle_index: int) -> int:
        """A child born at a boundary derives its birth from the clock.

        Never copy this off the parent's ``DomainClock``: that is the
        streamed-LBC-clock burn, generalized.
        """
        return self.boundary_ticks(cycle_index)

    # -- receipts --------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "schema": CLOCK_SCHEMA,
            "epoch_anchor": self.epoch_anchor.isoformat(),
            "tick_hz": TICK_HZ,
            "parent_step_ticks": self.parent_step_ticks,
            "cycle_ticks": self.cycle_ticks,
            "n_cycles": self.n_cycles,
        }


def _as_utc(when: datetime) -> datetime:
    """UTC-aware, or a refusal -- a naive time is an unanswered question."""
    if not isinstance(when, datetime):
        raise CycleRefusal("epoch anchor must be a datetime",
                           got_type=type(when).__name__, got=repr(when))
    if when.tzinfo is None:
        return when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc)
