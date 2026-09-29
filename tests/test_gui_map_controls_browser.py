"""A forecast's map and its time bar, in a real browser: what a person can pick is shown, or said.

The defects this file holds shut, each seen on the unmodified page:

- A nested run draws its parent hourly and its nest every quarter hour.
  With only the parent turned on, a quarter-hour time showed grid outlines
  and nothing else, with no word of why; Play and the step buttons went
  through those empty times as well.
- On a phone-wide window (390 px) New forecast ran off the header, the
  map's Picture, Files and Downscale ran off its top bar, and the time
  bar's later times ran off its end, out of reach.
- On a window narrower than 860 px the run's area was framed for the
  bars' heights before they settled (the top bar wrapping once the grids
  and the map were named, the time bar once its times were drawn), so it
  sat off centre in the space the bars leave: 67 px above it and 90 px
  below it at 390 px wide.

The page runs in headless Chrome (or Edge) against the real page server;
the run's records are answered by the test.  No weather picture is drawn:
the picture files are refused, and only the map's controls are under test.
"""

from __future__ import annotations

import pytest

from gpuwm.gui.server import build_server, serve_in_thread
from test_gui_create_draft import PORT, PORT_TRIES, Page, PageRunner

playwright_api = pytest.importorskip("playwright.sync_api")

DOMAINS = ["d01-12km", "d02-3km"]
#: The parent drawn at 06:00 and 07:00, its nest every quarter hour between.
TIMES = ["2026-09-27T06:00:00Z", "2026-09-27T06:15:00Z", "2026-09-27T06:30:00Z", "2026-09-27T06:45:00Z",
         "2026-09-27T07:00:00Z"]
DRAWN = [(DOMAINS[0], TIMES[0]), (DOMAINS[0], TIMES[4])] + [(DOMAINS[1], valid) for valid in TIMES]
PICTURES = [{"domain": domain, "valid": valid, "name": f"{i}.png", "path": f"{i}.png", "geo": None}
            for i, (domain, valid) in enumerate(DRAWN)]
NOT_AT_THIS_TIME = "The grids turned on have no picture at this time. Pick another time, or turn on a grid that has one."
ALL_OFF = "Every grid is turned off. Turn one on to see its map."


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("gpuwm.fetch._head_ok", lambda url: True)
        patch.setattr("gpuwm.source_availability.quick_head", lambda url: True)
        server = build_server(tmp_path_factory.mktemp("mapcontrols") / "runs", port=PORT, port_tries=PORT_TRIES,
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


def open_map(browser, served, *, width=1280, name="nested", region=None):
    tab = Page(browser, served, viewport={"width": width, "height": 844})
    page = tab.page
    page.route("**/api/runs/fixture", lambda r: r.fulfill(json={"id": "fixture", "name": name, "title": name,
                                                                 "status": {"state": "finished"}}))
    page.route("**/api/runs/fixture/map", lambda r: r.fulfill(json={
        "projection": None, "domains": [], "moves": [], "start_time": TIMES[0], "state": "finished",
        **({"region": region} if region else {})}))
    page.route("**/api/runs/fixture/pictures", lambda r: r.fulfill(json={
        "count": len(PICTURES), "domains": DOMAINS, "products": ["2m_temperature"],
        "favourites": ["2m_temperature"], "by_domain": {}}))
    page.route("**/api/runs/fixture/pictures/list?*", lambda r: r.fulfill(json={"pictures": PICTURES,
                                                                                 "georefs": []}))
    page.route("**/api/runs/fixture/files/**", lambda r: r.abort())
    page.goto(f"http://127.0.0.1:{served.port}/#/results/fixture")
    page.locator("body[data-screen=results][data-ready]").wait_for()
    page.locator(".track .tick").nth(len(TIMES) - 1).wait_for()
    return tab


def at(tab) -> int:
    """The time shown, as its place on the time bar."""

    return tab.page.evaluate("() => [...document.querySelectorAll('.track .tick')].findIndex((t) => "
                             "t.classList.contains('on'))")


@pytest.fixture()
def nested(browser, served):
    tab = open_map(browser, served)
    yield tab
    tab.close()
    assert not tab.errors, tab.errors


def test_a_time_no_grid_turned_on_has_a_picture_at_says_so_and_offers_the_nearest_one(nested):
    page = nested.page
    page.locator(".domains button").nth(1).click()          # the nest off: the parent alone
    page.locator(".track .tick").nth(1).click()             # 06:15, a time only the nest drew
    card = page.locator(".statecard")
    card.get_by_text(NOT_AT_THIS_TIME, exact=True).wait_for(timeout=5000)
    card.get_by_role("button", name="Show Sun 27 Sep, 06:00 UTC").click()
    page.wait_for_function("() => document.querySelector('.statecard').hidden", timeout=5000)
    assert at(nested) == 0
    # The same time with the nest turned back on has a picture, and the card goes.
    page.locator(".track .tick").nth(1).click()
    card.get_by_text(NOT_AT_THIS_TIME, exact=True).wait_for(timeout=5000)
    page.locator(".domains button").nth(1).click()
    page.wait_for_function("() => document.querySelector('.statecard').hidden", timeout=5000)


def test_the_step_buttons_go_to_the_times_a_grid_turned_on_has_a_picture_at(nested):
    page = nested.page
    page.locator(".domains button").nth(1).click()
    page.locator(".track .tick").nth(0).click()
    page.get_by_role("button", name="Forward one step").click()
    page.wait_for_function("() => document.querySelectorAll('.track .tick')[4].classList.contains('on')",
                           timeout=5000)
    assert page.locator(".statecard").is_hidden()
    page.get_by_role("button", name="Back one step").click()
    page.wait_for_function("() => document.querySelectorAll('.track .tick')[0].classList.contains('on')",
                           timeout=5000)
    # With every grid on, every time is stepped through.
    page.locator(".domains button").nth(1).click()
    page.get_by_role("button", name="Forward one step").click()
    page.wait_for_function("() => document.querySelectorAll('.track .tick')[1].classList.contains('on')",
                           timeout=5000)


def test_every_grid_turned_off_says_so(nested):
    page = nested.page
    page.locator(".domains button").nth(0).click()
    page.locator(".domains button").nth(1).click()
    page.locator(".statecard").get_by_text(ALL_OFF, exact=True).wait_for(timeout=5000)


#: Every control a person needs on the map page that is drawn and not where they can reach it: past either side
#: of the window or below it.
OFF_SCREEN = """() => [...document.querySelectorAll(
    '.hdr a, .hdr button, .mvtop a, .mvtop button, .mvtop input, .timebar button, .track .tick, .zoom button')]
  .filter((e) => e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden')
  .map((e) => { const r = e.getBoundingClientRect();
    return { what: e.getAttribute('aria-label') || e.textContent.trim(), left: Math.round(r.left),
             right: Math.round(r.right), bottom: Math.round(r.bottom) }; })
  .filter((r) => r.left < 0 || r.right > innerWidth + 0.5 || r.bottom > innerHeight + 0.5)"""


@pytest.mark.parametrize("width", [320, 390, 600, 860, 1440])
def test_every_control_and_time_of_the_map_is_on_screen_at_this_width(browser, served, width):
    tab = open_map(browser, served, width=width, name="a nested forecast with a long name to fit")
    try:
        page = tab.page
        page.wait_for_timeout(300)          # the bars measure themselves once drawn
        out_of_reach = page.evaluate(OFF_SCREEN)
        assert not out_of_reach, out_of_reach
        assert page.evaluate("document.documentElement.scrollWidth") <= width
        assert page.locator(".track").bounding_box()["width"] > 100
        for name in ("Picture", "Files", "Downscale"):
            assert page.locator(".mvtop").get_by_role("button" if name != "Files" else "link", name=name,
                                                      exact=True).is_visible(), name
    finally:
        tab.close()
    assert not tab.errors, tab.errors


#: A run's area as the map reply gives it before any grid is on record, across the equator so that the frame's
#: padding is as tall to the south as to the north.
REGION = [[10.0, -12.0], [16.0, -12.0], [16.0, 12.0], [10.0, 12.0], [10.0, -12.0]]
#: Where the area's outline is drawn on the map, in page pixels, beside the bars above and below it.
FRAMED = """() => {
  const canvas = document.querySelector('.mv canvas.mapcanvas');
  const box = canvas.getBoundingClientRect();
  const k = canvas.width / box.width;
  const px = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
  let top = Infinity; let bottom = -Infinity;
  for (let y = 0; y < canvas.height; y++) for (let x = 0; x < canvas.width; x++) {
    const i = (y * canvas.width + x) * 4;     // the outline's blue, which nothing else on this map is drawn in
    if (px[i] < 90 && px[i + 1] > 60 && px[i + 1] < 150 && px[i + 2] > 190) { top = Math.min(top, y); bottom = Math.max(bottom, y); }
  }
  return { top: box.top + top / k, bottom: box.top + bottom / k,
           above: document.querySelector('.mv .mvtop').getBoundingClientRect().bottom,
           below: document.querySelector('.mv .timebar').getBoundingClientRect().top };
}"""


@pytest.mark.parametrize("width", [390, 860, 1440])
def test_a_runs_area_is_framed_between_the_top_bar_and_the_time_bar(browser, served, width):
    tab = open_map(browser, served, width=width, region=REGION)
    try:
        page = tab.page
        page.wait_for_timeout(300)
        framed = page.evaluate(FRAMED)
    finally:
        tab.close()
    assert framed["above"] < framed["top"] < framed["bottom"] < framed["below"], framed
    # centred in the part of the map the bars leave free, not in the whole map behind the time bar
    assert abs((framed["top"] - framed["above"]) - (framed["below"] - framed["bottom"])) <= 4, framed
