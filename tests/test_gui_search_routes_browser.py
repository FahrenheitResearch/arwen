"""Search keeps every word typed, a "?" and what follows it included, and its filters as they were picked.

The defect: the page decoded every part of the link before the search page
split its words from its filters at the "?".  A search for "tornado?
Oklahoma" is carried as ``#/search/tornado%3F%20Oklahoma``; decoded first,
its own "?" was read as the start of the filters, so the page searched for
"tornado" and Oklahoma was lost.  A filter value holding "&" or "+" was
decoded twice the same way and split or changed.

The page runs in headless Chrome (or Edge) against the real page server;
the engine is a stand-in.
"""

from __future__ import annotations

from urllib.parse import parse_qs, quote, urlparse

import pytest

from gpuwm.gui.server import build_server, serve_in_thread
from test_gui_create_draft import PORT, PORT_TRIES, Page, PageRunner

playwright_api = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
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
    tab.asked = []
    tab.page.on("request", lambda req: tab.asked.append(req.url) if "/api/library/search" in req.url else None)
    yield tab
    tab.close()
    assert not tab.errors, tab.errors


def drawn(tab, screen):
    tab.page.wait_for_function("(s) => document.body.dataset.screen === s && document.body.dataset.ready === '1'",
                               arg=screen, timeout=20000)


def last_search(tab) -> dict[str, str]:
    """The query of the page's last request to the search API, one value per key."""

    assert tab.asked, "the page never asked the search API"
    return {key: values[0] for key, values in parse_qs(urlparse(tab.asked[-1]).query).items()}


@pytest.mark.parametrize("term", ["tornado? Oklahoma", "rain 100%", "wind/rain? 50% & snow"])
def test_the_words_typed_into_search_are_the_words_searched(tab, term):
    tab.page.goto(f"http://127.0.0.1:{tab.server.port}/#/browse")
    drawn(tab, "browse")
    tab.page.locator("form.bigsearch input").fill(term)
    tab.page.locator("form.bigsearch button").click()
    drawn(tab, "search")
    assert tab.page.locator("form.bigsearch input").input_value() == term
    assert tab.page.locator(".article h1").inner_text() == f"“{term}”"
    assert last_search(tab).get("q") == term


def test_a_search_link_keeps_its_words_and_its_filters(tab):
    words = "tornado? Oklahoma"
    tab.page.goto(f"http://127.0.0.1:{tab.server.port}/#/search/{quote(words, safe='')}?region=Plains+%26+Prairie")
    drawn(tab, "search")
    asked = last_search(tab)
    assert asked.get("q") == words, asked
    assert asked.get("region") == "Plains & Prairie", asked


def test_a_hand_typed_percent_sign_still_draws_the_search(tab):
    tab.page.goto(f"http://127.0.0.1:{tab.server.port}/#/search/100%")
    drawn(tab, "search")
    assert last_search(tab).get("q") == "100%"


@pytest.mark.parametrize("old, new", [
    ("#/wiki", "#/library"),
    ("#wiki", "#/library"),
    ("#/wiki?q=a%3Fb%26c&sort=score", "#/library?q=a%3Fb%26c&sort=score"),
    ("#/wiki/extra%2Fpart?q=a%2Fb", "#/library/extra%2Fpart?q=a%2Fb"),
    ("#/wiki/", "#/library/"),
])
def test_old_library_bookmarks_forward_with_their_suffix_and_query_unchanged(tab, old, new):
    tab.page.goto(f"http://127.0.0.1:{tab.server.port}/{old}")
    tab.page.wait_for_function("(hash) => location.hash === hash", arg=new, timeout=20000)
    drawn(tab, "library")
    assert tab.page.locator(".logo").inner_text() == "Weather Library"
    assert tab.page.title() == "Weather Library"
