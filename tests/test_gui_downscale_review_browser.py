"""A downscale review approves the settings it was asked about, never an edit made while it ran.

The defect: Review recorded the form as it stood when the engine's answer
came back, not as it was sent.  Picking 4 times finer while a review of 3
times finer was running bound the ratio-3 answer to the ratio-4 form, so
Start was enabled and the page said "This is the grid Start runs" over a
grid it had never planned.  Start must stay off and the page must ask for a
new review until the edited settings have their own.

The shipped downscale panel runs in headless Chrome (or Edge) inside the real
page server's page; its requests to the downscale route are answered by the
test, so no engine runs.
"""

from __future__ import annotations

import pytest

from gpuwm.gui.server import build_server, serve_in_thread
from test_gui_create_draft import PORT, PORT_TRIES, Page, PageRunner

playwright_api = pytest.importorskip("playwright.sync_api")

FACTS = {"domains": [{"id": 1, "dx_km": 12}], "domain": 1, "ratio": 3, "ratio_range": [2, 20],
         "cards": ["32gb"], "name": "child", "eligible": True}
OPEN = """async () => {
  const { openDownscale } = await import('/static/js/downscale.js');
  const words = (await (await fetch('/api/copy')).json()).screens.downscale;
  const map = { on(kind, fn) { window.chooseCentre = fn; return () => {}; }, add() { return () => {}; },
                redraw() {}, drawBox() {}, screen() { return [0, 0]; } };
  const stage = document.createElement('div');
  document.body.append(stage);
  await openDownscale({ map, stage, runId: 'parent', words });
  window.chooseCentre(-98, 39);
}"""


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    server = build_server(tmp_path_factory.mktemp("downscale") / "runs", port=PORT, port_tries=PORT_TRIES,
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


def test_an_edit_made_while_the_review_runs_keeps_start_off_until_it_is_reviewed(tab):
    page = tab.page
    reviews = []

    def answer(route):
        if route.request.method == "GET":
            route.fulfill(json=FACTS)
        else:
            reviews.append(route)  # answered below, once the form has been edited

    page.route("**/api/runs/parent/downscale", answer)
    page.evaluate(OPEN)
    panel = page.locator(".dspanel")
    review = panel.get_by_role("button", name="Review", exact=True)
    start = panel.get_by_role("button", name="Start", exact=True)
    fine = panel.locator("label", has_text="How fine").locator("select")

    review.click()
    tab.wait_held(reviews, 1)
    asked = reviews[0].request.post_data_json
    assert asked["ratio"] == 3 and asked["mode"] == "plan", asked
    # The person picks 4 times finer while the engine is still planning 3 times finer.
    fine.select_option("4")
    reviews[0].fulfill(json={"review": {"ratio": 3, "words": "Reviewed ratio 3"}})
    panel.get_by_text("Reviewed ratio 3", exact=True).wait_for()
    assert fine.input_value() == "4"
    assert start.is_disabled(), "Start was enabled for settings the review never checked"
    assert "Review again before Start" in panel.inner_text()

    # Reviewed as they now stand, the new settings may start.
    review.click()
    tab.wait_held(reviews, 2)
    assert reviews[1].request.post_data_json["ratio"] == 4
    reviews[1].fulfill(json={"review": {"ratio": 4, "words": "Reviewed ratio 4"}})
    panel.get_by_text("Reviewed ratio 4", exact=True).wait_for()
    page.wait_for_function("() => { const b = [...document.querySelectorAll('.dspanel button')]"
                           ".find((x) => x.textContent === 'Start'); return b && !b.disabled; }", timeout=10000)
    assert "Review again before Start" not in panel.inner_text()
