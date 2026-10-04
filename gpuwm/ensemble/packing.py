"""Member-to-card packing using measured budgets and sequential waves.

Member splits have no halo exchange. Each member keeps its own ordinary clock
and physics. A missing batched adapter selects ordinary member execution; a
resident shape that does not fit selects the existing streamed admission. The
planner never changes a configuration to make an ensemble fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Literal

from gpuwm.ensemble.admission import EnsembleMemoryModel, _integer

ExecutionMode = Literal["member_batched", "ordinary_member", "ordinary_streamed_member"]


@dataclass(frozen=True)
class CardBudget:
    device_id: int
    available_bytes: int
    total_bytes: int | None = None
    name: str = ""
    fingerprint: str = ""

    def __post_init__(self):
        object.__setattr__(self, "device_id", _integer(self.device_id, "device_id"))
        object.__setattr__(self, "available_bytes", _integer(self.available_bytes, "available_bytes"))
        if self.total_bytes is not None:
            object.__setattr__(self, "total_bytes", _integer(self.total_bytes, "total_bytes", positive=True))
            if self.available_bytes > self.total_bytes:
                raise ValueError("available device bytes exceed physical memory")


@dataclass(frozen=True)
class MemberBatch:
    wave: int
    device_id: int
    member_indices: tuple[int, ...]
    execution_mode: ExecutionMode
    required_bytes: int | None
    available_bytes: int
    reason: str = ""

    def __post_init__(self):
        for name in ("wave", "device_id", "available_bytes"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        members = tuple(_integer(member, "member index") for member in self.member_indices)
        if not members or len(set(members)) != len(members):
            raise ValueError("one batch needs distinct member indices")
        object.__setattr__(self, "member_indices", members)
        if self.execution_mode not in ("member_batched", "ordinary_member", "ordinary_streamed_member"):
            raise ValueError("unknown ensemble execution mode")
        if self.execution_mode != "member_batched" and len(members) != 1:
            raise ValueError("ordinary execution advances one member at a time")
        if self.required_bytes is not None:
            object.__setattr__(self, "required_bytes", _integer(self.required_bytes, "required_bytes"))
        if self.execution_mode != "ordinary_streamed_member":
            if self.required_bytes is None or self.required_bytes > self.available_bytes:
                raise ValueError("resident member batch exceeds its sampled memory budget")

    @property
    def members(self):
        return len(self.member_indices)

    def receipt(self):
        return {"wave": self.wave, "device_id": self.device_id,
                "member_indices": list(self.member_indices), "members": self.members,
                "execution_mode": self.execution_mode, "required_bytes": self.required_bytes,
                "available_bytes": self.available_bytes, "reason": self.reason}


@dataclass(frozen=True)
class MemberPackingPlan:
    requested_members: int
    cards: tuple[CardBudget, ...]
    capacities: tuple[int, ...]
    batches: tuple[MemberBatch, ...]
    inventory_id: str = ""
    inventory_ids: tuple[str, ...] = ()

    def __post_init__(self):
        requested = _integer(self.requested_members, "requested_members", positive=True)
        members = tuple(member for batch in self.batches for member in batch.member_indices)
        if tuple(sorted(members)) != tuple(range(requested)):
            raise ValueError("packing must execute every requested member exactly once")
        if len(self.cards) != len(self.capacities):
            raise ValueError("one member capacity is required per physical card")
        if self.inventory_ids and len(self.inventory_ids) != len(self.cards):
            raise ValueError("one inventory ID is required per physical card")
        if any(batch.device_id not in {card.device_id for card in self.cards} for batch in self.batches):
            raise ValueError("member batch names a card outside its sampled budgets")
        if tuple(sorted({batch.wave for batch in self.batches})) != tuple(range(self.waves)):
            raise ValueError("member packing waves must be contiguous")
        placements = tuple((batch.wave, batch.device_id) for batch in self.batches)
        if len(set(placements)) != len(placements):
            raise ValueError("a physical card can own only one member batch per wave")

    @property
    def waves(self):
        return max(batch.wave for batch in self.batches) + 1

    def batches_in_wave(self, wave):
        wave = _integer(wave, "wave")
        return tuple(batch for batch in self.batches if batch.wave == wave)

    def receipt(self):
        cards = [{"device_id": card.device_id, "name": card.name,
                  "fingerprint": card.fingerprint, "available_bytes": card.available_bytes,
                  "total_bytes": card.total_bytes, "resident_member_capacity": capacity}
                 for card, capacity in zip(self.cards, self.capacities)]
        for card, inventory_id in zip(cards, self.inventory_ids):
            card["inventory_id"] = inventory_id
        return {"requested_members": self.requested_members, "scheduled_members": self.requested_members,
                "inventory_id": self.inventory_id, "waves": self.waves,
                "cards": cards,
                "batches": [batch.receipt() for batch in self.batches],
                "physics_changed": False, "halo_exchange": False}


def card_budgets_from_readings(options, readings):
    """Resolve the existing ``[devices]`` ids against already sampled budgets.

    Repeated ids name one physical card for member packing. No CUDA call is
    made here. Readings must provide ``available_bytes`` after reusable pool
    bytes and device-wide occupancy have been resolved by the ordinary door.
    """
    ids = tuple(dict.fromkeys(options.device_ids()))
    result = []
    for device_id in ids:
        if device_id not in readings:
            raise ValueError(f"device {device_id} has no sampled memory budget")
        reading = readings[device_id]
        if isinstance(reading, CardBudget):
            if reading.device_id != device_id:
                raise ValueError("device budget is bound to a different card")
            result.append(reading)
        else:
            result.append(CardBudget(device_id=device_id, **dict(reading)))
    return tuple(result)


def _models_for_cards(model, cards, name):
    if isinstance(model, EnsembleMemoryModel):
        return {card.device_id: model for card in cards}, False
    if isinstance(model, Mapping):
        for device_id in model:
            _integer(device_id, "model device_id")
        if set(model) != {card.device_id for card in cards}:
            raise ValueError(f"{name} mapping must name every sampled card exactly")
        if any(not isinstance(value, EnsembleMemoryModel) for value in model.values()):
            raise TypeError(f"{name} mapping needs an execution-adapter memory inventory per card")
        return dict(model), True
    raise TypeError(f"{name} needs an execution-adapter memory inventory")


def pack_members(requested_members, cards, model, *, batched=True, reason="", streamed_model=None):
    """Pack all members across physical cards, then sequential waves.

    ``batched=False`` keeps each member in the ordinary runner, including its
    adaptive clock. If no resident member fits, one ordinary streamed member
    is scheduled per wave. An omitted ``streamed_model`` means that the
    ordinary tile door must resolve its tile size, not that zero memory is
    required. Its own configuration and physical allocation checks still run.

    ``model`` may be one common model or ``{device_id: model}``. A mapping
    binds each card's actual context, kernel and physics workspace envelope;
    every sampled card needs its own entry, including a card with zero fit.
    """
    requested = _integer(requested_members, "requested_members", positive=True)
    cards = tuple(cards)
    if not cards or any(not isinstance(card, CardBudget) for card in cards):
        raise ValueError("member packing needs sampled physical card budgets")
    if len({card.device_id for card in cards}) != len(cards):
        raise ValueError("member packing must price each physical card once")
    models, device_bound = _models_for_cards(model, cards, "member packing")
    if not isinstance(batched, bool):
        raise TypeError("batched must be a boolean")
    streamed_models = ({card.device_id: None for card in cards} if streamed_model is None else
                       _models_for_cards(streamed_model, cards, "streamed_model")[0])
    capacities = tuple(models[card.device_id].largest_that_fits(card.available_bytes, max_members=requested if batched else 1)
                       for card in cards)
    # A resident-capable card gets the first members before a card that
    # needs tile fallback. Preserve the selected order within either group.
    assignments = tuple(sorted(zip(cards, capacities), key=lambda row: row[1] == 0))
    batches = []
    next_member = wave = 0
    while next_member < requested:
        for card, capacity in assignments:
            if next_member == requested:
                break
            card_model = models[card.device_id]
            if capacity:
                count = min(capacity if batched else 1, requested - next_member)
                mode = "member_batched" if batched and count > 1 else "ordinary_member"
                need = card_model.required_bytes(count)
                explanation = reason
            else:
                count, mode = 1, "ordinary_streamed_member"
                streamed = streamed_models[card.device_id]
                need = None if streamed is None else streamed.required_bytes(1)
                explanation = (f"one resident member needs {card_model.required_bytes(1)} bytes; "
                               f"card budget is {card.available_bytes} bytes; ordinary tile admission "
                               "selects a streamed member without changing its physics")
                if reason:
                    explanation = reason + "; " + explanation
            batch = MemberBatch(wave, card.device_id, tuple(range(next_member, next_member + count)),
                                mode, need, card.available_bytes, explanation)
            batches.append(batch)
            next_member += count
        wave += 1
    inventory_ids = tuple(models[card.device_id].inventory_id for card in cards) if device_bound else ()
    return MemberPackingPlan(requested, cards, capacities, tuple(batches),
                             "per-card" if device_bound else model.inventory_id, inventory_ids)


__all__ = ["CardBudget", "MemberBatch", "MemberPackingPlan", "card_budgets_from_readings", "pack_members"]
