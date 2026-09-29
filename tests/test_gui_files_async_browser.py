"""Files shows the pictures of the folder picked last, and says so when a folder's list cannot be read.

The defects: picking a folder asked the server for its picture list and drew
whatever answer came back, so a slow answer for the folder picked first
replaced the pictures of the folder picked after it while the second row
stayed marked as picked.  A list that failed drew nothing and said nothing;
the page kept the last folder's pictures, or none, with no reason and no way
to ask again.

The shipped Files page runs in headless Chrome (or Edge) against the real
page server; the forecast and its picture lists are answered by the test.
"""

from __future__ import annotations

import pytest

from gpuwm.gui.server import build_server, serve_in_thread
from test_gui_create_draft import PORT, PORT_TRIES, Page, PageRunner

playwright_api = pytest.importorskip("playwright.sync_api")

GROUPS = [{"domain": "d01-12km", "episode": "", "product": product, "day": "2026-09-27", "count": 1}
          for product in ("composite_reflectivity", "2m_temperature")]
FAILED = {"ok": False, "message": "The picture list could not be read.", "fix": "Try again in a moment."}


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    server = build_server(tmp_path_factory.mktemp("files") / "runs", port=PORT, port_tries=PORT_TRIES,
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


def open_files(tab):
    """The Files page of the test's forecast; the list of held picture-list requests (the first is held)."""

    page = tab.page
    held = []

    def listing(route):
        if held:
            route.fulfill(json={"count": 0, "pictures": []})
        else:
            held.append(route)

    page.route("**/api/runs/fixture", lambda route: route.fulfill(json={"title": "Fixture", "folder": "fixture"}))
    page.route("**/api/runs/fixture/pictures", lambda route: route.fulfill(json={"count": 2, "groups": GROUPS}))
    page.route("**/api/runs/fixture/pictures/list?*", listing)
    page.goto(f"http://127.0.0.1:{tab.server.port}/#/explore/fixture")
    page.locator("button.rowlink").nth(1).wait_for()
    return held


def test_a_late_list_of_the_folder_picked_first_never_replaces_the_folder_picked_since(tab):
    page = tab.page
    held = open_files(tab)
    rows = page.locator("button.rowlink")
    rows.nth(0).click()
    tab.wait_held(held, 1)
    second = rows.nth(1).inner_text()
    rows.nth(1).click()
    heading = page.locator(".thumbs").locator("..").locator(".phd b")
    heading.wait_for()
    assert heading.inner_text() == second
    # The first folder's list lands now, after the second folder's pictures are drawn.
    held[0].fulfill(json={"count": 0, "pictures": []})
    page.wait_for_timeout(400)
    assert heading.inner_text() == second, "the first folder's late list replaced the folder picked since"
    assert rows.nth(1).get_attribute("aria-pressed") == "true"


def test_a_list_that_fails_says_why_and_offers_try_again(tab):
    page = tab.page
    held = open_files(tab)
    page.locator("button.rowlink").first.click()
    tab.wait_held(held, 1)
    held[0].fulfill(status=503, json=FAILED)
    page.get_by_text(FAILED["message"], exact=False).wait_for(timeout=5000)
    page.get_by_role("button", name="Try again", exact=True).click()
    page.locator(".thumbs").wait_for(state="attached", timeout=5000)
    assert FAILED["message"] not in page.locator("main").inner_text()
