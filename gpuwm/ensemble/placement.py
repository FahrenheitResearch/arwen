"""Source-independent member placement and completion-driven card queues.

A card consumes a new member as soon as it finishes. There is no cross-card
wave barrier. Slots represent explicitly admitted concurrent member workers;
the default is one slot per card (not one forecast spread across cards).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class CardPlacement:
    card: str
    slots: int = 1

    def __post_init__(self):
        if not self.card or type(self.slots) is not int or self.slots < 1:
            raise ValueError("a card needs an identity and a positive member capacity")


def place_members(members, cards):
    """Stable initial assignment; no source, perturbation or provenance policy."""
    members, cards = tuple(members), tuple(cards)
    if not cards or len({card.card for card in cards}) != len(cards):
        raise ValueError("placement needs distinct cards")
    if len(set(members)) != len(members):
        raise ValueError("member IDs must be unique; otherwise an output would be overwritten")
    slots = tuple(card.card for card in cards for _ in range(card.slots))
    result = {card.card: [] for card in cards}
    for index, member in enumerate(members):
        result[slots[index % len(slots)]].append(member)
    return result


class CardQueue:
    """One FIFO per card with idle-card work stealing and exact completion.

    Slow physics on one card cannot strand work while another card is idle.
    A failed member is accounted for explicitly and never silently retried as
    another member. The caller chooses restart policy from its durable receipt.
    """
    def __init__(self, members, cards):
        cards = tuple(cards)
        self.pending = {card: deque(items) for card, items in place_members(members, cards).items()}
        self.capacity = {card.card: card.slots for card in cards}
        self.active = {card.card: set() for card in cards}
        self.finished = {}
        self._lock = Lock()

    def take(self, card):
        with self._lock:
            if len(self.active[card]) >= self.capacity[card]:
                return None
            source = card
            if not self.pending[source]:
                source = max(self.pending, key=lambda key: len(self.pending[key]))
            if not self.pending[source]:
                return None
            member = self.pending[source].popleft()
            self.active[card].add(member)
            return member

    def complete(self, card, member, result):
        with self._lock:
            if member not in self.active[card] or member in self.finished:
                raise ValueError("completion must belong to one active member on this card")
            self.active[card].remove(member)
            self.finished[member] = result

    @property
    def done(self):
        with self._lock:
            return not any(self.pending.values()) and not any(self.active.values())
