"""Child domains across a cycle boundary: re-place, retire, reclaim.

**A child cannot be allocated mid-run on this line, and that is a design
decision, not a gap.**  :func:`gpuwm.experiment.active_experiment` refuses
any grid_id that was not declared dormant, :func:`build_experiment` prices
the whole experiment at t=0, and :func:`gpuwm.core.clock.build_schedule`
bakes activation ticks into precomputed op tables.  The result is that a
spawn is an ACTIVATION of a reservation rather than an allocation that can
OOM after three hours of integration, and VRAM is deterministic from the
first second of the run.  Fighting that is an allocator project.

So this module makes the RESERVATION the fixed thing and the PLACEMENT the
arbitrary thing, and uses the cycle boundary as the allocation point --
which is free, because a cycle boundary is already a process boundary.

    within a cycle    spawn is activation of a pre-declared dormant slot,
                      through unchanged 2.3.0 machinery
    across cycles     the placement plan is recomputed from the analysis;
                      a live child is re-declared at its moved placement
                      and restored from its anchor, a child whose storm
                      died is RETIRED and its slot returns to the pool,
                      and a new storm claims a free slot

The honest cost, stated here because it is the one thing a user must know
before choosing ``--child-slots N``: **N slots cost N nests' worth of VRAM
for the whole run, filled or not.**  That is what buys determinism.

**A child may die; the cycle may not.**  :func:`run_child_leg` catches a
child's divergence and returns ``DIVERGED`` with the field, the cell and
the value.  It does not re-raise.  That asymmetry is what makes unattended
arbitrary spawning safe: a bad placement or a blown-up 500 m child costs
one slot, not the run.  It does NOT catch failures that are not the
child's -- an allocator error is the run's problem and propagates.

No case names, no station names (standing owner rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gpuwm.cycle.contracts import (CHILD_STATES, CycleRefusal, TickRatio,
                                   seconds_to_ticks)
from gpuwm.cycle.placement import (DEFAULT_KEEPOUT_PARENT_CELLS,
                                   ParentGeometry, PlacementRequest,
                                   ResolvedPlacement, rank_and_assign,
                                   resolve_placement, separation_km)

#: The trigger a spine-declared dormant slot carries.  ``time`` with an
#: ``at_s`` the spine supplies at the cycle boundary, because the spine --
#: not a field threshold inside one leg -- is what decides placement now.
#: The 2.3.0 spawn machinery executes it unchanged.
SPINE_SPAWN_TRIGGER = "time"


@dataclass(frozen=True, slots=True)
class ChildSlot:
    """A reservation: shape and step, but no place.

    Identically shaped slots are the point.  A pool of differently shaped
    reservations would price the worst case of every one of them, and the
    determinism that justifies reserving at t=0 would go with it.
    """

    grid_id: int
    nx: int
    ny: int
    dx_m: float
    dt_seconds: float
    parent_grid_ratio: int


class SlotPool:
    """Which reservations are free, which are live, and why one came back.

    Deliberately not a bare set: a release carries a REASON, and the
    reason is what a receipt needs to distinguish "its storm decayed"
    from "it blew up" from "it left the parent" months later.
    """

    def __init__(self, slots: Sequence[ChildSlot]) -> None:
        self._slots: dict[int, ChildSlot] = {}
        for item in slots:
            if item.grid_id in self._slots:
                raise CycleRefusal("two slots claim one grid_id",
                                   grid_id=int(item.grid_id))
            self._slots[int(item.grid_id)] = item
        self._live: dict[int, ResolvedPlacement] = {}
        self._released: list[dict] = []

    def slot(self, grid_id: int) -> ChildSlot:
        try:
            return self._slots[int(grid_id)]
        except KeyError:
            raise CycleRefusal(
                "no such reservation in the slot pool",
                grid_id=int(grid_id),
                declared=tuple(sorted(self._slots))) from None

    def free(self) -> tuple[int, ...]:
        return tuple(sorted(set(self._slots) - set(self._live)))

    def claim(self, grid_id: int, placement: ResolvedPlacement) -> None:
        self.slot(grid_id)
        if int(grid_id) in self._live:
            raise CycleRefusal(
                "that reservation is already carrying a child",
                grid_id=int(grid_id),
                held_by=self._live[int(grid_id)].request.source)
        self._live[int(grid_id)] = placement

    def release(self, grid_id: int, *, reason: str) -> None:
        """The despawn.  Free at a cycle boundary; impossible mid-leg."""

        if int(grid_id) not in self._live:
            raise CycleRefusal("that reservation is not carrying a child",
                               grid_id=int(grid_id), live=tuple(self._live))
        self._live.pop(int(grid_id))
        self._released.append({"grid_id": int(grid_id), "reason": str(reason)})

    def live(self) -> dict[int, ResolvedPlacement]:
        return dict(self._live)

    def released(self) -> list[dict]:
        return list(self._released)


def child_tick_alignment(*, parent_step_ticks: int,
                         child_dt_seconds: float) -> TickRatio:
    """A child's step against its parent's, exactly, or a named refusal.

    A thin wrapper so every caller in the spine gets the SAME refusal for
    a non-dividing child dt, and gets it at plan time -- before a single
    byte is allocated -- rather than at hour three when the child has
    quietly drifted off its parent's step lattice.
    """

    return TickRatio(
        parent_ticks=int(parent_step_ticks),
        child_ticks=seconds_to_ticks(float(child_dt_seconds),
                                     label="child dt"))


def _moved_parent_cells(before: ResolvedPlacement,
                        after: ResolvedPlacement) -> int:
    """Chebyshev distance in parent cells between two placements."""

    return int(max(abs(after.i_parent_start - before.i_parent_start),
                   abs(after.j_parent_start - before.j_parent_start)))


def _record(*, grid_id: int, state: str, placement: ResolvedPlacement | None,
            moved: int, evidence: Mapping, refusal: str | None) -> dict:
    if state not in CHILD_STATES:
        raise CycleRefusal("child state is not in the shared vocabulary",
                           state=str(state), allowed=CHILD_STATES)
    return {"grid_id": int(grid_id), "state": str(state),
            "placement": None if placement is None else placement.to_payload(),
            "moved_parent_cells": int(moved), "evidence": dict(evidence),
            "refusal": refusal}


def plan_children(*, cycle_index: int, pool: SlotPool,
                  requests: Sequence[PlacementRequest],
                  previous_children: Sequence[Mapping],
                  retire_below_strength: float, min_separation_km: float,
                  parent_geometry: ParentGeometry,
                  allow_clamp: bool = False,
                  keepout_parent_cells: int = DEFAULT_KEEPOUT_PARENT_CELLS
                  ) -> list[dict]:
    """One cycle boundary's child plan: who moves, who dies, who is born.

    Returns one record per child, and **every input request appears in the
    output** -- placed, or REFUSED with the reason named.  A planner that
    dropped a request because there was no slot would write a receipt
    indistinguishable from one where the storm was never detected, and
    those two mean opposite things about the run.
    """

    parent = parent_geometry
    live = {int(item["grid_id"]): item
            for item in previous_children
            if item.get("state") in ("LIVE", "PLANNED")
            and item.get("placement") is not None}
    remaining = list(requests)
    out: list[dict] = []

    # 1. Existing children first: each one keeps its slot if its storm is
    #    still there, so a moving storm never costs a second reservation.
    for grid_id in sorted(live):
        record = live[grid_id]
        held = record["placement"]["request"]
        follower = _closest(remaining, lat=held["lat"], lon=held["lon"],
                            within_km=float(min_separation_km))
        if follower is None or float(follower.strength) < float(
                retire_below_strength):
            observed = (None if follower is None
                        else float(follower.strength))
            pool.release(grid_id, reason=(
                "no signal above the retirement threshold"
                if follower is None else "storm decayed"))
            out.append(_record(
                grid_id=grid_id, state="RETIRED", placement=None, moved=0,
                evidence={"observed_strength": observed,
                          "retire_below_strength": float(
                              retire_below_strength),
                          "cycle_index": int(cycle_index)},
                refusal=None))
            if follower is not None:
                remaining.remove(follower)
            continue

        remaining.remove(follower)
        moved_to = resolve_placement(
            follower, parent_geometry=parent, grid_id=grid_id,
            keepout_parent_cells=keepout_parent_cells,
            allow_clamp=allow_clamp)
        if moved_to.state == "REFUSED":
            # The storm walked out of the parent.  The slot comes back
            # rather than the child being clamped into a place it is not:
            # clamping is opt-in, and it is opt-in here too.
            pool.release(grid_id, reason="storm left the parent domain")
            out.append(_record(
                grid_id=grid_id, state="RETIRED", placement=None, moved=0,
                evidence={"observed_strength": float(follower.strength),
                          "cycle_index": int(cycle_index)},
                refusal=moved_to.refusal))
            continue

        before = _placement_from_payload(record["placement"])
        pool.release(grid_id, reason="re-placed at this cycle boundary")
        pool.claim(grid_id, moved_to)
        out.append(_record(
            grid_id=grid_id, state="LIVE", placement=moved_to,
            moved=_moved_parent_cells(before, moved_to),
            evidence=dict(follower.evidence), refusal=None))

    # 2. Whatever is left is a new storm, ranked against the slots that
    #    are actually free after the retirements above.
    for resolved in rank_and_assign(
            remaining, free_slots=pool.free(),
            min_separation_km=float(min_separation_km),
            parent_geometry=parent,
            keepout_parent_cells=keepout_parent_cells,
            allow_clamp=allow_clamp):
        if resolved.state == "REFUSED":
            out.append(_record(
                grid_id=resolved.grid_id, state="REFUSED", placement=None,
                moved=0, evidence=dict(resolved.request.evidence),
                refusal=resolved.refusal))
            continue
        pool.claim(resolved.grid_id, resolved)
        out.append(_record(
            grid_id=resolved.grid_id, state="PLANNED", placement=resolved,
            moved=0, evidence=dict(resolved.request.evidence), refusal=None))
    return out


def _closest(requests: Sequence[PlacementRequest], *, lat: float, lon: float,
             within_km: float) -> PlacementRequest | None:
    """The request nearest a live child's current place, if any is near."""

    scored = [(separation_km(lat, lon, item.lat, item.lon), item)
              for item in requests]
    scored = [row for row in scored if row[0] <= float(within_km)]
    if not scored:
        return None
    return min(scored, key=lambda row: (row[0], -float(row[1].strength)))[1]


def _placement_from_payload(payload: Mapping) -> ResolvedPlacement:
    """Rebuild a resolution from its receipt form, for a move comparison."""

    ask = payload["request"]
    return ResolvedPlacement(
        grid_id=int(payload["grid_id"]),
        i_parent_start=int(payload["i_parent_start"]),
        j_parent_start=int(payload["j_parent_start"]),
        parent_grid_ratio=int(payload["parent_grid_ratio"]),
        request=PlacementRequest(
            lat=float(ask["lat"]), lon=float(ask["lon"]),
            dx_m=float(ask["dx_m"]), nx=int(ask["nx"]), ny=int(ask["ny"]),
            source=str(ask["source"]), strength=float(ask["strength"]),
            evidence=dict(ask["evidence"])),
        state=str(payload["state"]), clamped=bool(payload["clamped"]),
        refusal=payload["refusal"])


def emit_dormant_domains(pool: SlotPool, *, template_domain: Mapping
                         ) -> list[dict]:
    """The slot pool as ``[[domain]]`` blocks the 2.3.0 loader can eat.

    This is the hinge of the whole design: an arbitrary geographic plan
    becomes a DECLARED experiment, and every piece of spawn machinery
    downstream (``build_spawn_config`` -> ``SpawnController`` ->
    ``SpawnRunner.on_leg_boundary`` -> ``walk_spawn_legs``) runs unchanged.

    The placeholder placement is deliberate and matches what the loader
    already expects of a dormant nest: the real one arrives at the cycle
    boundary, from the analysis, through ``active_experiment``.
    """

    blocks = []
    for grid_id in sorted(set(pool.free()) | set(pool.live())):
        slot = pool.slot(grid_id)
        block = dict(template_domain)
        block.update({
            "grid_id": int(slot.grid_id),
            "nx": int(slot.nx), "ny": int(slot.ny),
            "dx": float(slot.dx_m), "dy": float(slot.dx_m),
            "dt": float(slot.dt_seconds),
            "parent_grid_ratio": int(slot.parent_grid_ratio),
            # A spine-declared slot is triggered by the SPINE, at the
            # boundary, which is what ``time`` with an at_s the supervisor
            # supplies means.  A field threshold here would put the
            # placement decision back inside one leg, where it cannot see
            # the analysis that just landed.
            "spawn": {"trigger": SPINE_SPAWN_TRIGGER, "at_s": 0.0},
        })
        blocks.append(block)
    return blocks


# ---------------------------------------------------------------------------
# one child's leg -- where a child is allowed to die
# ---------------------------------------------------------------------------
def first_nonfinite(state: Mapping) -> tuple[str, list[int], float] | None:
    """The first field, cell and value that stopped being a number."""

    for name in sorted(state):
        array = np.asarray(state[name])
        if not np.issubdtype(array.dtype, np.floating):
            continue
        bad = ~np.isfinite(array)
        if not bool(bad.any()):
            continue
        cell = [int(axis[0]) for axis in np.nonzero(bad)]
        return str(name), cell, float(array[tuple(cell)])
    return None


def run_child_leg(*, grid_id: int, placement: ResolvedPlacement,
                  cycle_index: int,
                  advance: Callable[..., Mapping],
                  **advance_kwargs: Any) -> dict:
    """Advance one child through one cycle, absorbing only ITS failure.

    ``advance`` is injected so the whole path is testable without a GPU;
    the shipped caller passes the forecast driver's own leg runner.

    A non-finite state or a :class:`CycleRefusal` out of the runner is
    this child's death and returns ``DIVERGED`` with the field, cell and
    value.  Anything else propagates: an allocator failure or a bug in the
    supervisor is the RUN's problem, and swallowing it here would turn a
    halt into a silently thinner forecast.
    """

    try:
        state = advance(grid_id=int(grid_id), placement=placement,
                        cycle_index=int(cycle_index), **advance_kwargs)
    except CycleRefusal as refusal:
        return _record(
            grid_id=grid_id, state="DIVERGED", placement=placement, moved=0,
            evidence={"cycle_index": int(cycle_index),
                      **{key: repr(value)
                         for key, value in refusal.observed.items()}},
            refusal=(f"d{int(grid_id):02d} refused at cycle "
                     f"{int(cycle_index)}: {refusal}"))

    found = first_nonfinite(state or {})
    if found is not None:
        name, cell, value = found
        return _record(
            grid_id=grid_id, state="DIVERGED", placement=placement, moved=0,
            evidence={"field": name, "cell": cell, "value": value,
                      "cycle_index": int(cycle_index)},
            refusal=(f"d{int(grid_id):02d} diverged at cycle "
                     f"{int(cycle_index)}: {name} is {value} at cell "
                     f"{cell}; the child is retired and the cycle "
                     "continues"))

    return _record(grid_id=grid_id, state="LIVE", placement=placement,
                   moved=0, evidence={"cycle_index": int(cycle_index)},
                   refusal=None)
