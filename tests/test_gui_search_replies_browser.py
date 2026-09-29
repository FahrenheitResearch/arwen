"""A search's answer belongs to the words it was asked for, in a real browser.

The defects this file holds shut, each seen on the unmodified page:

- New forecast's Find an event: clearing the field, pressing Escape or
  typing other words while a search was on its way did not cancel it, so
  its answer brought back the choices just cleared, and Enter then opened
  one of them, even from a list that was not shown.
- The header search (Ctrl K): changing "tornado" to "x" left tornado's
  answers under "x", and an answer for "tornado" that came late was shown
  for "x", so Enter opened the tornado page instead of searching for x.

The page runs in headless Chrome (or Edge) against the real page server;
each search answer is held by the test and let go when the test says, so
the late answer is the page's own request, answered late.
"""

from __future__ import annotations

import pytest

from gpuwm.gui.server import build_server, serve_in_thread
from test_gui_create_draft import PORT, PORT_TRIES, Page, PageRunner

playwright_api = pytest.importorskip("playwright.sync_api")

EVENT = {"results": [{"kind": "event", "id": "stale-event", "title": "Obsolete event", "start": "2013-05-20"}]}
KIND = {"results": [{"kind": "phenomenon", "id": "tornado", "title": "Obsolete storm"}]}


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("gpuwm.fetch._head_ok", lambda url: True)
        patch.setattr("gpuwm.source_availability.quick_head", lambda url: True)
        server = build_server(tmp_path_factory.mktemp("search") / "runs", port=PORT, port_tries=PORT_TRIES,
                              runner=PageRunner(), token="t" * 43)
        serve_in_thread(server)
        try:
            yield server
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
    tab = Page(browser, served)
    yield tab
    tab.close()
    assert not tab.errors, tab.errors


def _held_search(tab, words):
    return tab.hold(f"**/api/wiki/search?q={words}&sort=score")


def _answer(held, document):
    for route in held:
        route.fulfill(json=document)


def _record_moves(tab):
    """Every address the page goes to from now on: a made-up event's recipe fails and the page comes back to
    New forecast, so the address afterwards does not say whether Enter opened anything."""

    tab.page.evaluate("() => { window.moves = []; addEventListener('hashchange', () => moves.push(location.hash)); }")


def _moves(tab):
    return tab.page.evaluate("window.moves")


# ---------------------------------------------------------------- Find an event

@pytest.mark.parametrize("leave", ["clear", "escape", "other words", "click away"])
def test_an_event_search_left_before_its_answer_brings_nothing_back(tab, leave):
    tab.page.goto(f"http://127.0.0.1:{tab.server.port}/#/create")
    field = tab.page.locator(".finder input")
    field.wait_for()
    held = _held_search(tab, "tornado")
    field.fill("tornado")
    tab.wait_held(held, 1)
    if leave == "clear":
        field.fill("")
    elif leave == "escape":
        field.press("Escape")
    elif leave == "other words":
        field.fill("x")
    else:
        tab.page.locator("h1, h2").first.click()
    tab.page.wait_for_timeout(400)          # past the field's pause before it asks
    _answer(held, EVENT)
    tab.page.wait_for_timeout(200)
    assert not tab.page.locator(".finderlist").is_visible()
    assert tab.page.locator(".finderrow").count() == 0
    _record_moves(tab)
    field.press("Enter")
    tab.page.wait_for_timeout(200)
    assert _moves(tab) == []


def test_enter_opens_an_event_only_from_the_list_on_screen(tab):
    tab.page.goto(f"http://127.0.0.1:{tab.server.port}/#/create")
    field = tab.page.locator(".finder input")
    field.wait_for()
    held = _held_search(tab, "tornado")
    field.fill("tornado")
    tab.wait_held(held, 1)
    _answer(held, EVENT)
    tab.page.locator(".finderrow", has_text="Obsolete event").wait_for()
    # A click elsewhere puts the list away and keeps the words.
    tab.page.locator("h1, h2").first.click()
    assert not tab.page.locator(".finderlist").is_visible()
    _record_moves(tab)
    tab.page.evaluate("document.querySelector('.finder input').dispatchEvent("
                      "new KeyboardEvent('keydown', {key: 'Enter', bubbles: true}))")
    tab.page.wait_for_timeout(200)
    assert _moves(tab) == []
    # Back in the field the same words' choices are shown again, and Enter opens the first.  (The event's
    # recipe is never answered: only where Enter goes is under test here.)
    tab.page.route("**/api/wiki/recipe/**", lambda route: None)
    field.focus()
    tab.page.locator(".finderrow", has_text="Obsolete event").wait_for()
    field.press("Enter")
    tab.page.wait_for_function("() => moves.some((m) => m.startsWith('#/create/recipe/stale-event'))",
                               timeout=5000)


# ---------------------------------------------------------------- the header search

@pytest.mark.parametrize("answered", [False, True], ids=["answer-late", "answer-in"])
def test_the_header_search_shows_and_opens_only_answers_for_the_words_typed(tab, answered):
    tab.page.goto(f"http://127.0.0.1:{tab.server.port}/#/wiki")
    tab.page.wait_for_function("() => !!document.body.dataset.ready", timeout=20000)
    held = _held_search(tab, "tornado")
    tab.page.keyboard.press("Control+k")
    field = tab.page.locator(".pal input")
    field.wait_for()
    field.fill("tornado")
    tab.wait_held(held, 1)
    if answered:
        _answer(held, KIND)
        tab.page.get_by_role("option", name="Obsolete storm").wait_for()
    field.fill("x")
    if not answered:
        tab.page.wait_for_timeout(300)      # past the field's pause before it asks
        _answer(held, KIND)
        tab.page.wait_for_timeout(200)
    assert tab.page.get_by_role("option", name="Obsolete storm").count() == 0
    field.press("Enter")
    tab.page.wait_for_function("() => location.hash === '#/search/x'", timeout=5000)
