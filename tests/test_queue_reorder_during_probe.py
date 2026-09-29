"""A queue reorder that lands while the scheduler reads the card decides which forecast starts next.

The defect: a tick read the line, then read the card outside the lock, then
started the forecast that had been first when it read the line.  A Move made
while the card was being read had already answered ``["second", "first"]``,
yet ``first`` started, so a long earlier forecast delayed the one the user had
just put first.  The line is now compared in the same locked step that marks a
forecast starting, and a changed line sends the scheduler to look again.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

from gpuwm.gui import runs
from gpuwm.gui.files import write_json
from gpuwm.gui.machines import LOCAL
from gpuwm.gui.queue import MARKER_SCHEMA, QUEUE_SCHEMA, ForecastQueue


def _queue(tmp_path, launched):
    # A stand-in engine, which runs on any install (the real one needs CuPy in this Python).
    api = SimpleNamespace(root=tmp_path, _launch=lambda run_id, *a, **kw: launched.append(run_id),
                          runner=SimpleNamespace(runtime_gap=lambda: None))
    queue = ForecastQueue(api, owner_file="", disk=lambda: 100.0)
    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        write_json(directory / runs.QUEUED, {
            "schema": MARKER_SCHEMA, "machine": LOCAL, "need_gib": 2.0,
            "queued_utc": "2026-09-27T00:00:00Z", "held": None, "waiting": None})
    queue._save({"schema": QUEUE_SCHEMA, "order": ["first", "second"], "owned": []})
    return queue


CARD = {"machine": LOCAL, "busy": False, "total_gib": 16.0, "free_gib": 16.0}


def test_a_move_made_while_the_card_is_read_starts_the_forecast_moved_to_the_front(tmp_path):
    launched: list[str] = []
    queue = _queue(tmp_path, launched)
    reading, proceed = Event(), Event()

    def held_card(fresh=False):
        reading.set()
        assert proceed.wait(5), "the test never let the card reading finish"
        return CARD

    queue.local_card = held_card
    with ThreadPoolExecutor(max_workers=1) as pool:
        tick = pool.submit(queue.tick)
        try:
            assert reading.wait(5), "the scheduler never read the card"
            assert queue.move("second", -1) == ["second", "first"]
        finally:
            proceed.set()
        tick.result(timeout=5)
    # The tick that saw the old line starts nothing; the next look (the loop wakes for it at once) starts the
    # forecast now first.
    assert launched in ([], ["second"]), f"launched {launched} after moving second to the front"
    if not launched:
        queue.local_card = lambda fresh=False: CARD
        queue.tick()
    assert launched == ["second"], f"launched {launched} after moving second to the front"
    assert queue.order() == ["first"]


def test_an_unchanged_line_starts_its_first_forecast_in_one_look(tmp_path):
    launched: list[str] = []
    queue = _queue(tmp_path, launched)
    queue.local_card = lambda fresh=False: CARD
    assert queue.tick() == ["first"]
    assert launched == ["first"]
    assert queue.order() == ["second"]
