# SPDX-License-Identifier: Apache-2.0
"""How many ensemble members stay resident on one card, from measured bytes.

Resident members share one card-wide workspace (the step scratch arena and the
restart-rebuilt state arrays, :mod:`gpuwm.ensemble.sequential_workspace`) and
take serial turns on one stream.  Each member's own arrays -- prognostic state,
carried scratch, the physics driver, its lateral boundary tables and, inside an
analysis window, its incremental-update forcing -- are private.  Members over
the card's budget are parked: their private pages move to host memory between
turns (:mod:`gpuwm.ensemble.managed_storage`) and come back for their turn.

:func:`plan_residency` turns measured bytes into a resident count and says
whether members must also park while an analysis holds the card.  Every figure
carries its source, so a plan built on any estimated figure is labelled
estimated, and every refusal names the out-of-memory it prevents.

Parking modes:

``managed``
    Private allocations live in a per-member managed pool and migrate whole
    between the host and the card (the measured mode).
``slot-restore``
    The named fallback when managed memory is unavailable: one retained model
    slot per resident position, and a member restored into it from its own
    full checkpoint (state, clock, physics carriers) at each turn.  Its
    measured basis is the in-process branch restore of the mass-ensemble
    runner (exact restores into one retained model, about 2.3 s at
    480x420x49), which restores a canonical parent and is not yet a
    multi-cycle park/resume: a parking adapter must save and restore each
    member's current full checkpoint and prove continuation against door legs
    before it is used.  Not built here; planned exactly like ``managed``.
``none``
    Nothing parks: a roster over the resident count is refused.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

MEASURED = "measured"
ESTIMATED = "estimated"
PARKING_MODES = ("managed", "slot-restore", "none")
GIB = 1024 ** 3


class MemberBudgetRefused(ValueError):
    """A roster or card the planner cannot place, refused before anything allocates."""


@dataclass(frozen=True)
class Figure:
    """One byte (or second) figure and where it came from."""

    value: int | float
    source: str = MEASURED
    evidence: str = ""

    def __post_init__(self):
        if self.source not in (MEASURED, ESTIMATED):
            raise ValueError(f"figure source {self.source!r} is neither measured nor estimated")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("a figure is a number")
        if not math.isfinite(float(self.value)) or self.value < 0:
            raise ValueError(f"figure {self.value!r} must be finite and non-negative")

    @property
    def measured(self) -> bool:
        return self.source == MEASURED

    def receipt(self):
        return {"value": self.value, "source": self.source, "evidence": self.evidence}


def _figure(value, source=MEASURED, evidence=""):
    return value if isinstance(value, Figure) else Figure(value, source, evidence)


@dataclass(frozen=True)
class MemberBytes:
    """One member's private device bytes.

    ``state`` is the model state and carried scratch beyond the shared
    workspace's views; ``physics`` the physics driver's arrays; ``forcing`` the
    member's lateral boundary tables; ``iau`` the incremental-update forcing
    attached during an analysis window (zero outside one); ``residue`` device
    bytes the member holds outside its parkable owner (allocations made
    outside an allocation scope), which stay on the card when it parks.
    """

    state: Figure
    physics: Figure
    forcing: Figure
    iau: Figure = Figure(0)
    residue: Figure = Figure(0)

    def __post_init__(self):
        for name in ("state", "physics", "forcing", "iau", "residue"):
            object.__setattr__(self, name, _figure(getattr(self, name)))

    @property
    def parkable(self) -> int:
        return int(self.state.value + self.physics.value + self.forcing.value + self.iau.value)

    @property
    def parkable_without_iau(self) -> int:
        return self.parkable - int(self.iau.value)

    @property
    def device(self) -> int:
        return self.parkable + int(self.residue.value)

    @property
    def measured(self) -> bool:
        return all(getattr(self, n).measured for n in ("state", "physics", "forcing", "iau", "residue"))

    def receipt(self):
        return {name: getattr(self, name).receipt()
                for name in ("state", "physics", "forcing", "iau", "residue")} | {
                    "parkable_bytes": self.parkable, "device_bytes": self.device}


@dataclass(frozen=True)
class CardBudget:
    """One card's fixed costs and phase peaks.

    ``total`` the card's memory; ``other`` bytes other processes hold;
    ``context`` this process's CUDA context, kernels and library handles;
    ``arena`` the shared workspace; ``transient`` the stepping member's
    largest transient (its radiation call); ``analysis`` the analysis's device
    peak on this card (zero when it runs elsewhere); ``export`` device staging
    for exporting member states to the analysis; ``reserve`` allocator
    fragmentation and safety margin; ``host`` host memory available to parked
    pages.
    """

    total: Figure
    context: Figure
    arena: Figure
    transient: Figure
    reserve: Figure
    other: Figure = Figure(0)
    analysis: Figure = Figure(0)
    export: Figure = Figure(0)
    host: Figure | None = None
    label: str = ""

    def __post_init__(self):
        for name in ("total", "context", "arena", "transient", "reserve", "other", "analysis", "export"):
            object.__setattr__(self, name, _figure(getattr(self, name)))
        if self.host is not None:
            object.__setattr__(self, "host", _figure(self.host))

    @property
    def fixed(self) -> int:
        return int(self.other.value + self.context.value + self.arena.value + self.reserve.value)

    @property
    def measured(self) -> bool:
        names = ["total", "context", "arena", "transient", "reserve", "other", "analysis", "export"]
        return all(getattr(self, n).measured for n in names) and (self.host is None or self.host.measured)

    def receipt(self):
        out = {name: getattr(self, name).receipt()
               for name in ("total", "other", "context", "arena", "transient", "reserve", "analysis", "export")}
        out["host"] = None if self.host is None else self.host.receipt()
        out["label"] = self.label
        return out


@dataclass(frozen=True)
class ResidencyPlan:
    """Which members stay on the card, and the peak bytes of each phase."""

    card: CardBudget
    members: Mapping[int, MemberBytes]
    parking: str
    pinned: tuple[int, ...]
    rotating: tuple[int, ...]
    analysis_pinned: tuple[int, ...]
    peaks: Mapping[str, int]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def resident_count(self) -> int:
        """Members on the card at once while the forecast steps."""
        return len(self.pinned) + (1 if self.rotating else 0)

    @property
    def parks(self) -> bool:
        return bool(self.rotating)

    @property
    def park_during_analysis(self) -> bool:
        return len(self.analysis_pinned) < len(self.pinned)

    @property
    def measured(self) -> bool:
        return self.card.measured and all(m.measured for m in self.members.values())

    def keeps_resident(self, member_id) -> bool:
        return member_id in self.pinned

    def receipt(self):
        return {
            "schema": "gpuwm-member-residency-plan/1",
            "label": self.card.label,
            "source": MEASURED if self.measured else ESTIMATED,
            "members": len(self.members),
            "resident_count": self.resident_count,
            "pinned": list(self.pinned),
            "rotating": list(self.rotating),
            "parking": self.parking,
            "parks_during_forecast": self.parks,
            "park_during_analysis": self.park_during_analysis,
            "analysis_resident": list(self.analysis_pinned),
            "peaks_bytes": dict(self.peaks),
            "peaks_gib": {k: round(v / GIB, 3) for k, v in self.peaks.items()},
            "card": self.card.receipt(),
            "member_bytes": {str(k): v.receipt() for k, v in sorted(self.members.items())},
            "notes": list(self.notes),
        }


def _gib(value) -> str:
    return f"{value / GIB:.2f} GiB"


def plan_residency(members: Mapping[int, MemberBytes], card: CardBudget, *,
                   parking: str = "managed", order=None) -> ResidencyPlan:
    """Place ``members`` on ``card``: a pinned resident set and a rotating slot.

    While the forecast steps, the pinned members never leave the card and the
    rest take turns through one more slot (each resumed for its turn and parked
    after it), so a round of member-major turns moves only the rotating
    members.  Members are pinned in ``order`` (default: ascending private
    bytes, which pins the most).  During the analysis the card also holds the
    analysis peak and the export staging; pinned members that would not fit
    beside them park for the analysis (``park_during_analysis``).

    Refused by name: a card on which one member cannot step at all; a roster
    over the resident count with ``parking="none"``; parked pages over the
    host budget; an analysis that does not fit even with every member parked.
    """
    if parking not in PARKING_MODES:
        raise ValueError(f"parking {parking!r} is not one of {PARKING_MODES}")
    members = {int(k): (v if isinstance(v, MemberBytes) else MemberBytes(**v)) for k, v in members.items()}
    if not members:
        raise ValueError("a residency plan needs at least one member")
    ids = list(order) if order is not None else sorted(members, key=lambda m: (members[m].parkable, m))
    if sorted(ids) != sorted(members):
        raise ValueError("order must name every member exactly once")
    total = int(card.total.value)
    residue = sum(int(m.residue.value) for m in members.values())
    fixed = card.fixed + residue
    transient = int(card.transient.value)
    notes = []

    def forecast_peak(pinned, rotating, *, iau=True):
        size = ((lambda m: members[m].parkable) if iau
                else (lambda m: members[m].parkable_without_iau))
        slot = max((size(m) for m in rotating), default=0)
        return fixed + sum(size(m) for m in pinned) + slot + transient

    largest = max(m.parkable for m in members.values())
    if fixed + largest + transient > total:
        raise MemberBudgetRefused(
            f"one member cannot step on this card ({card.label or 'card'}): its private arrays "
            f"({_gib(largest)}) plus the fixed bytes ({_gib(fixed)}: workspace, CUDA context, reserve, "
            f"other tenants and every member's unparkable residue) and the stepping transient "
            f"({_gib(transient)}) exceed the card's {_gib(total)}; its first radiation step would "
            "stop in a CUDA out-of-memory")
    pinned = []
    for index, member in enumerate(ids):
        rest = ids[index + 1:]
        candidate = pinned + [member]
        if forecast_peak(candidate, rest) <= total:
            pinned = candidate
        else:
            break
    rotating = [m for m in ids if m not in pinned]
    if len(pinned) == len(ids) - 1 and rotating:
        # The last member fits as the slot itself: everyone is resident.
        pinned, rotating = pinned + rotating, []
    if rotating and parking == "none":
        raise MemberBudgetRefused(
            f"{len(ids)} members need {_gib(sum(m.parkable for m in members.values()))} of private "
            f"arrays and this card holds {len(pinned) + 1} at once; with parking off, constructing "
            f"member {rotating[1] if len(rotating) > 1 else rotating[0]} would stop in a CUDA "
            "out-of-memory (park members, or run fewer per card)")
    if rotating and card.host is not None:
        parked = sum(members[m].parkable for m in rotating)
        if parked > int(card.host.value):
            raise MemberBudgetRefused(
                f"the parked members' pages ({_gib(parked)}) exceed the host memory available to "
                f"them ({_gib(card.host.value)}): parking would fail or swap the host and stall "
                "every turn (run fewer members per card)")
    analysis = int(card.analysis.value) + int(card.export.value)
    analysis_pinned = list(pinned)
    if analysis:
        def analysis_peak(kept):
            return fixed + sum(members[m].parkable for m in kept) + analysis
        if fixed + analysis > total:
            raise MemberBudgetRefused(
                f"the analysis ({_gib(card.analysis.value)} device peak plus "
                f"{_gib(card.export.value)} export staging) and the fixed bytes ({_gib(fixed)}) exceed "
                f"the card's {_gib(total)} even with every member parked: the analysis would stop in a "
                "CUDA out-of-memory (run it on another card)")
        while analysis_pinned and analysis_peak(analysis_pinned) > total:
            analysis_pinned.pop()
        if parking == "none" and len(analysis_pinned) < len(pinned):
            raise MemberBudgetRefused(
                f"the analysis needs {_gib(analysis)} beside {len(pinned)} resident members and parking "
                "is off: the analysis would stop in a CUDA out-of-memory")
    # The plan holds every member's IAU forcing attached (a cycle whose window
    # is its length always carries one); "forecast" is the same placement
    # outside a window.
    peaks = {"forecast": forecast_peak(pinned, rotating, iau=False),
             "iau": forecast_peak(pinned, rotating),
             "analysis": fixed + sum(members[m].parkable for m in analysis_pinned) + analysis,
             "export": fixed + sum(members[m].parkable for m in analysis_pinned) + int(card.export.value)}
    if any(m.iau.value == 0 for m in members.values()):
        notes.append("a member with no IAU figure is planned without forcing bytes")
    if not all(m.measured for m in members.values()) or not card.measured:
        notes.append("built on at least one estimated figure: the plan is an estimate")
    return ResidencyPlan(card=card, members=members, parking=parking, pinned=tuple(pinned),
                         rotating=tuple(rotating), analysis_pinned=tuple(analysis_pinned),
                         peaks=peaks, notes=tuple(notes))


def member_bytes_from_estimate(domain_estimate, *, iau_fields=0, source=ESTIMATED,
                               evidence="gpuwm.core.preflight.estimate_experiment") -> MemberBytes:
    """A member's private bytes from the preflight itemization of its domain.

    The workspace's views (arena-admitted scratch and rebuilt state symbols)
    are excluded; ``iau_fields`` 3-D float32 fields of the domain's mass grid
    price the analysis window's forcing.
    """
    from gpuwm.core.preflight import scratch_slot_lifetime, scratch_slot_uses_arena

    def shared(name):
        # A slot with no lifetime row is never in the arena: priced private.
        return scratch_slot_lifetime(name) is not None and scratch_slot_uses_arena(name)

    private = (int(domain_estimate.resident_bytes) - int(domain_estimate.arena_scratch_bytes)
               - int(domain_estimate.rebuilt_state_bytes))
    physics = int(domain_estimate.category_bytes("physics"))
    forcing = sum(int(item.nbytes) for item in domain_estimate.items
                  if item.category == "lbc" and not shared(item.name))
    state = private - physics - forcing
    return MemberBytes(state=Figure(max(0, state), source, evidence),
                       physics=Figure(physics, source, evidence),
                       forcing=Figure(forcing, source, evidence),
                       iau=Figure(int(iau_fields), source, evidence))


def timing_rows(plan, *, park_seconds=None, resume_seconds=None, members_per_cycle=None):
    """Seconds parking adds per cycle: each rotating member parks and resumes once."""
    rotating = len(plan.rotating)
    if park_seconds is None or resume_seconds is None:
        return {"rotating_members": rotating, "seconds_added_per_cycle": None}
    return {"rotating_members": rotating,
            "seconds_added_per_cycle": rotating * (float(park_seconds) + float(resume_seconds))}


__all__ = ["CardBudget", "ESTIMATED", "Figure", "MEASURED", "MemberBudgetRefused", "MemberBytes",
           "PARKING_MODES", "ResidencyPlan", "member_bytes_from_estimate",
           "plan_residency", "timing_rows"]
