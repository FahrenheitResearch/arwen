from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from gpuwm.ensemble.placement import CardPlacement, CardQueue, place_members


def test_thousand_members_across_eight_cards():
    placed = place_members(range(1000), [CardPlacement(str(i)) for i in range(8)])
    assert [len(items) for items in placed.values()] == [125] * 8
    assert sorted(member for items in placed.values() for member in items) == list(range(1000))


def test_memory_admitted_slots_are_respected():
    placed = place_members(range(9), [CardPlacement('large', 2), CardPlacement('small')])
    assert len(placed['large']) == 6
    queue = CardQueue(range(9), [CardPlacement('large', 2), CardPlacement('small')])
    first, second = queue.take('large'), queue.take('large')
    assert queue.take('large') is None
    queue.complete('large', second, {'ok': True})
    assert queue.take('large') is not None
    with pytest.raises(ValueError, match='active'):
        queue.complete('small', first, {})


def test_fast_card_steals_pending_work_without_a_wave_barrier():
    queue = CardQueue(range(6), [CardPlacement('slow'), CardPlacement('fast')])
    slow = queue.take('slow')
    seen = []
    for _ in range(5):
        member = queue.take('fast')
        assert member is not None
        seen.append(member)
        queue.complete('fast', member, True)
    assert len(set(seen + [slow])) == 6
    assert not queue.done
    queue.complete('slow', slow, False)
    assert queue.done


def test_concurrent_queue_completes_every_member_exactly_once():
    cards = [CardPlacement(str(i)) for i in range(8)]
    queue = CardQueue(range(1000), cards)
    barrier = Barrier(len(cards))
    def drain(card):
        member = queue.take(card.card)
        assert member is not None
        barrier.wait(timeout=10)
        queue.complete(card.card, member, card.card)
        while (member := queue.take(card.card)) is not None:
            queue.complete(card.card, member, card.card)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(drain, cards))
    assert queue.done and len(queue.finished) == 1000
    assert set(queue.finished.values()) == {card.card for card in cards}


def test_duplicate_members_cannot_overwrite_output():
    with pytest.raises(ValueError, match='unique'):
        place_members([1, 1], [CardPlacement('0')])
