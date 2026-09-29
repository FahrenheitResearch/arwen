"""My forecasts in a real browser: a list reply read before a Move up never puts the old order back.

The defect: the page reads the list every few seconds and again after each
Move up, Move down or Remove, and drew whichever reply arrived, in arrival
order. A slow reply read before a Move up, arriving after the move had been
drawn, put the old order back on the page while the server kept the new one.

The shipped page runs in headless Chromium against the real page server with
two forecasts queued behind a busy card; only the delivery of one real reply
is held back.
"""

from __future__ import annotations

import json
import time

import pytest

from gpuwm.gui import runs
from gpuwm.gui.queue import MARKER_SCHEMA
from gpuwm.gui.server import build_server, serve_in_thread
from test_gui_create_draft import PORT, PORT_TRIES, Page, PageRunner

playwright_api = pytest.importorskip("playwright.sync_api")
expect = playwright_api.expect


class BusyCard(PageRunner):
    """The page's engine with the card held by another job, so the queued forecasts wait in line."""

    def card_holder(self):
        return {"job_id": "job-busy", "started_utc": "2026-09-28T01:00:00+00:00", "kind": "gui:run-plan",
                "folder": None}


@pytest.fixture()
def served(tmp_path):
    root = tmp_path / "runs"
    for minute, name in enumerate(("first", "second")):
        folder = root / name
        folder.mkdir(parents=True)
        (folder / runs.PLAN).write_text(json.dumps({"schema": "gpuwm.run-plan.v1", "name": name, "route": "prepared",
                                                    "config": {"intent": {"source": "gfs", "hours": 3}}}),
                                        encoding="utf-8")
        (folder / runs.QUEUED).write_text(json.dumps({"schema": MARKER_SCHEMA, "machine": "this-computer",
                                                      "queued_utc": f"2026-09-28T01:0{minute}:00Z"}),
                                          encoding="utf-8")
    runner = BusyCard()
    server = build_server(root, port=PORT, port_tries=PORT_TRIES, runner=runner, token="t" * 43)
    serve_in_thread(server)
    assert server.api.queue.order(scan=True) == ["first", "second"]
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture()
def browser():
    with playwright_api.sync_playwright() as p:
        launched = None
        for channel in ("chrome", "msedge", None):
            try:
                launched = p.chromium.launch(channel=channel, headless=True) if channel else p.chromium.launch(headless=True)
                break
            except Exception:  # noqa: BLE001 - try the next browser this computer has
                continue
        if launched is None:
            pytest.skip("No Chromium-family browser is installed")
        yield launched
        launched.close()


def test_a_reply_read_before_move_up_cannot_put_the_old_order_back(browser, served):
    tab = Page(browser, served, viewport={"width": 1366, "height": 800})
    try:
        page = tab.page
        page.goto(f"http://127.0.0.1:{served.port}/#/runs")
        names = page.locator("table.queuelist tbody tr td.name")
        expect(names).to_have_text(["first", "second"])
        held = []

        def hold_first(route):
            # The first list read after this point is answered by the server now and delivered later.
            if not held:
                held.append((route, route.fetch()))
            else:
                route.continue_()

        page.route("**/api/queue", hold_first)
        deadline = time.monotonic() + 15
        while not held and time.monotonic() < deadline:
            page.wait_for_timeout(100)
        assert held, "the page did not read its list again"
        page.locator("table.queuelist tbody tr").nth(1).get_by_role("button", name="Move up", exact=True).click()
        expect(names).to_have_text(["second", "first"])
        assert served.api.queue.order() == ["second", "first"]
        route, reply = held[0]
        route.fulfill(response=reply)
        # Watched for a second: the old reply lands at once, and must change nothing.
        seen = []
        for _ in range(20):
            page.wait_for_timeout(50)
            seen.append(names.all_text_contents())
        assert all(order == ["second", "first"] for order in seen), seen
        assert not tab.errors, tab.errors
    finally:
        tab.close()
