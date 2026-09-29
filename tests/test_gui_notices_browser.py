"""The notice line under the header says what just happened, on the page the person is looking at.

The defects this file holds shut:

- My forecasts put up "The page lost touch with the server." when a read
  of the forecasts folder failed and never took it down once the server
  answered again, so a page back in touch kept saying it was not. Its
  reads once cleared the whole line after every good answer, which also
  cleared the line a start had just left there.
- A read of the forecasts folder that failed after the person had left My
  forecasts put that same line on the page they had gone to, where nothing
  takes it down.
- Queue it on Review set "Queued in place 1. ..." and then opened My
  forecasts, and drawing that page cleared the line before anyone saw it.
  The same happened to the line that says a queued forecast waits for its
  start to be published, the only place that says so when it is queued,
  and to "Started. Opening the Watching page." after Start. The start's
  line is now set once the page it opens is drawn, and My forecasts' own
  reads must leave it there.

The page runs in headless Chrome (or Edge) against the real page server;
the engine is a stand-in, so no forecast runs.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from gpuwm.gui.availability import Board
from gpuwm.gui.server import build_server, serve_in_thread
from test_gui_create_draft import PORT, PORT_TRIES, Page, PageRunner
from test_gui_server import request

playwright_api = pytest.importorskip("playwright.sync_api")

LOST = "The page lost touch with the server."
#: How long the server is out of reach, as when the page server is restarted, in milliseconds.
AWAY_MS = 10000
QUEUED = "Queued in place 1. It starts by itself when the card is free."
QUEUED_DATA = "Queued in place 1. It starts by itself once its start is published and the card is free."
STARTED = "Started. Opening the Watching page."
#: My forecasts reads the forecasts folder this often (runs.js), in milliseconds.
READ_MS = 3000


class HeldRunner(PageRunner):
    """The page's engine, with a card the test can say is busy running another forecast."""

    def __init__(self) -> None:
        super().__init__()
        self.holder: str | None = None

    def card_holder(self):
        return self.holder


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    # What the data hosts answer to each object the page asks about: True there, False not published yet.
    hosts = {"answer": True}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("gpuwm.fetch._head_ok", lambda url: True)
        patch.setattr("gpuwm.source_availability.quick_head", lambda url: hosts["answer"])
        patch.setattr("gpuwm.source_availability.published_stop", lambda window, **_: None)
        patch.setattr("gpuwm.gui.api.disk_free_gib", lambda path: 4096.0)
        runner = HeldRunner()
        root = tmp_path_factory.mktemp("notices") / "runs"
        server = build_server(root, port=PORT, port_tries=PORT_TRIES, runner=runner, token="t" * 43)
        # The card and the disk as the queue reads them, so no test asks this machine's own.
        card = {"devices": [{"name": "Test card", "memory_total_mib": 16 * 1024, "memory_used_mib": 512}],
                "processes": []}
        server.api.queue._cards = lambda: card
        server.api.queue._disk = lambda: 4096.0
        serve_in_thread(server)
        try:
            yield server, runner, hosts
        finally:
            server.shutdown()
            server.server_close()


@pytest.fixture(scope="module")
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


@pytest.fixture()
def tab(browser, served):
    server, _runner, _hosts = served
    tab = Page(browser, server)
    yield tab
    tab.close()
    assert not tab.errors, tab.errors


def drawn(tab, screen):
    tab.page.wait_for_function("(s) => document.body.dataset.screen === s && document.body.dataset.ready === '1'",
                               arg=screen, timeout=20000)


def open_runs(tab):
    tab.page.goto(f"http://127.0.0.1:{tab.server.port}/#/runs")
    drawn(tab, "runs")


def line(tab):
    """The notice line's text, or "" when there is none."""

    return tab.page.evaluate("() => { const p = document.querySelector('.noticebox .notice'); return p ? p.textContent : ''; }")


def to_review(tab, route):
    tab.open(route)
    tab.step("When")
    tab.next()
    tab.step("How fine")
    tab.next()
    tab.step("Physics")
    tab.next()
    tab.step("Review")


def click_when_on(tab, name):
    button = tab.page.get_by_role("button", name=name, exact=True)
    button.wait_for(timeout=20000)
    tab.page.wait_for_function(
        "(n) => [...document.querySelectorAll('button')].some((b) => b.textContent === n && !b.disabled)",
        arg=name, timeout=20000)
    button.click()


def stays(tab, text, ms):
    """The line still says ``text`` after ``ms`` of the page's own reads."""

    tab.page.wait_for_timeout(ms)
    assert line(tab).startswith(text), line(tab)


def remove_queued(server):
    _, listing = request(server, "GET", "/api/queue")
    for item in listing.get("items", []):
        reply, _ = request(server, "POST", f"/api/queue/{item['run']}/remove")
        assert reply.status == 200


# ---------------------------------------------------------------- losing touch and getting it back

def test_my_forecasts_takes_the_lost_line_down_once_the_server_answers_again(tab):
    open_runs(tab)
    assert line(tab) == ""
    # Every request to the server fails for ten seconds, as when the page server is restarted.
    away = time.monotonic()
    tab.page.route("**/api/**", lambda route: route.abort())
    tab.page.wait_for_function("(t) => (document.querySelector('.noticebox .notice') || {}).textContent"
                               "?.startsWith(t)", arg=LOST, timeout=2 * READ_MS + 4000)
    stays(tab, LOST, max(0, AWAY_MS - int((time.monotonic() - away) * 1000)))
    tab.page.unroute("**/api/**")
    # The first good read takes it down: the next read is at most one interval away.
    tab.page.wait_for_function("() => !document.querySelector('.noticebox .notice')", timeout=READ_MS + 3000)
    assert tab.page.locator(".dtable.runlist, .panel.pad").count() >= 1


def test_a_read_that_fails_after_leaving_my_forecasts_says_nothing_on_the_next_page(tab):
    # Every read of the forecasts folder waits: the sidebar's count and My forecasts' own list.
    held = tab.hold("**/api/runs")
    open_runs(tab)
    tab.wait_held(held, 2)
    # Settings draws without the forecasts folder.
    tab.page.evaluate("location.hash = '#/settings'")
    drawn(tab, "settings")
    # The reads sent before My forecasts was left fail now.
    for route in held:
        route.abort()
    tab.page.unroute("**/api/runs")
    tab.page.wait_for_timeout(1000)
    assert not line(tab).startswith(LOST), line(tab)


# ---------------------------------------------------------------- the line a start leaves

def test_queue_it_says_the_forecasts_place_on_my_forecasts_and_it_stays_through_the_pages_reads(tab, served):
    server, runner, _hosts = served
    runner.holder = "The GPU is held by running job job-x."
    try:
        to_review(tab, "create/40,-100,600x600/when/at/2020-06-01T12/src/era5")
        click_when_on(tab, "Queue it")
        tab.page.wait_for_function("() => location.hash === '#/runs'", timeout=20000)
        drawn(tab, "runs")
        assert line(tab) == QUEUED
        # My forecasts reads the folder every few seconds; its good reads leave the line as it is.
        stays(tab, QUEUED, 2 * READ_MS + 1000)
        assert tab.page.locator(".dtable.queuelist tbody tr").count() == 1
    finally:
        runner.holder = None
        remove_queued(server)


def test_queue_it_on_a_start_not_published_yet_says_it_waits_for_it_on_my_forecasts(tab, served):
    server, _runner, hosts = served
    moment = datetime.now(timezone.utc)
    # The newest six-hourly start by this clock: the one still being made, which only Queue it takes.
    frontier = f"{moment:%Y-%m-%d}T{moment.hour // 6 * 6:02d}"
    hosts["answer"] = False
    # Asked afresh: an earlier test's Create page on this module's server asks every source's newest start while
    # the hosts answer yes, and the board keeps that answer, so this start read as published and Queue it said
    # nothing about waiting for it.
    server.api.board.close()
    server.api.board = Board()
    try:
        to_review(tab, f"create/40,-100,600x600/when/at/{frontier}/src/gfs")
        click_when_on(tab, "Queue it")
        tab.page.wait_for_function("() => location.hash === '#/runs'", timeout=40000)
        drawn(tab, "runs")
        assert line(tab) == QUEUED_DATA
        stays(tab, QUEUED_DATA, 2 * READ_MS + 1000)
    finally:
        hosts["answer"] = True
        remove_queued(server)


def test_start_says_started_on_the_watching_page(tab, served):
    _server, runner, _hosts = served
    runner.holder = None
    to_review(tab, "create/40,-100,600x600/when/at/2020-06-01T12/src/era5")
    click_when_on(tab, "Start forecast")
    tab.page.wait_for_function("() => location.hash.startsWith('#/watch/')", timeout=20000)
    drawn(tab, "watch")
    assert line(tab).startswith(STARTED), line(tab)
    stays(tab, STARTED, 2000)
