"""New forecast sends what the person chose, from one draft, in a real browser.

The defects this file holds shut, each seen on the unmodified page:

- The grid choices' memory lines priced other nests than a click gave: the
  3 km preview kept the event's 4,3 chain (three domains to 250 m) while
  clicking 3 km sent one 3 km domain, because the payload read the chosen
  spacing, not the one it was asked about.
- Picking a nest ladder in a customised event kept the event's chain and
  buffers beside it, and the plan was refused.
- Customise of a storm-following event sent one 12 km domain: the
  following 3 km nest and its cyclone setup were dropped.
- Reloading Review kept the box, date and source and put every other
  choice back to its default.
- While the source list was on its way the page took the date and source
  out of its own link, so the same link drawn twice opened on the newest
  start, and a reload in that wait lost the kept draft.
- Customise could start a storm-following run the disk cannot hold, which
  the event page's button refuses.
- Review said the source cycle was the forecast start when a forecast hour
  was set to start from.
- Picking physics scheme A, then B within the check's wait, then A again,
  never checked A: the cancelled wait left an entry that was never sent.
- The assistant's settings dropped a saved choice-server address on the
  next save.
- The Create map let a phone's pan gesture cancel the box drag, and the
  physics and Files rows could not be reached by keyboard.
- Show command beside Start and Queue it asked without the machine picked
  under Run it on, so it printed this computer's interpreter and plan path
  for a forecast that runs on the node, and picking another machine closed
  an open Show command instead of showing that machine's line.
- A settings Save answered after the person edited a field again put the
  value it had sent back over the newer edit.
- Customise of a 12, 3 and 1 km best run showed one grid: the event's nests
  rode on the 12 km Overview choice, which named its outer grid alone, and
  the plan it built left out the rest of the best run (how often each grid
  writes), so it was not the run the event page's button starts.

The page runs in headless Chrome (or Edge) against the real page server;
the engine is a stand-in, so no forecast runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuwm.gui import runs
from gpuwm.gui.machines import LOCAL, Registry
from gpuwm.gui.server import build_server, serve_in_thread
from test_gui_machines import FakeMachine
from test_gui_server import FakeRunner, make_run, request

playwright_api = pytest.importorskip("playwright.sync_api")

#: The page server binds in this range: the other ports on the test machines belong to other services.
PORT = 8770
PORT_TRIES = 30

SOURCES = {"sources": [
    {"source_id": "gfs", "display_name": "GFS", "max_forecast_hour": 384,
     "run_plan": {"intent_supported": True, "intent_routes": ["prepared"], "requires_source_root": False}},
    {"source_id": "era5", "display_name": "ERA5", "max_forecast_hour": 0, "record_kind": "reanalysis",
     "run_plan": {"intent_supported": True, "intent_routes": ["config"], "requires_source_root": False}},
]}
PROFILES = {"sources": [{"source_id": sid, "default_profile_id": "p1",
                         "profiles": [{"profile_id": "p1", "admissible": True, "is_default": True}]}
                        for sid in ("gfs", "era5")],
            "profiles": [{"profile_id": "p1", "summary": "p1: words"}]}
CATALOG = {"default_suite": "p1", "suites": [{"id": "p1", "label": "Set one"}], "families": [
    {"id": "microphysics", "name": "Microphysics", "what": "Cloud and rain.", "schemes": [
        {"id": "a", "choice": "a", "label": "Scheme A", "description": "The first.", "cost": None, "is_default": True},
        {"id": "b", "choice": "b", "label": "Scheme B", "description": "The second.", "cost": None}]}]}
CYCLONE = {"schema": "gpuwm.cyclone-setup.v2", "kind": "configuration", "created": False,
           "forecast_start_hour": 0, "start_time": "2004-03-26T18:00:00",
           "domains": [{"grid_id": 1, "parent_id": 0, "nx": 200, "ny": 160, "nz": 49, "dx_m": 12000.0,
                        "dy_m": 12000.0, "following": False},
                       {"grid_id": 2, "parent_id": 1, "nx": 240, "ny": 240, "nz": 49, "dx_m": 3000.0,
                        "dy_m": 3000.0, "following": True}],
           "memory": {"peak_envelope_bytes": int(5.98 * 2**30), "budget_bytes": int(6.75 * 2**30)},
           "fitting": {"review_required": False, "notice": None}}


def nested(answer, argv):
    """The stand-in's fit with the nests its plan asks for: the outer grid, then one grid per ratio of the chain."""

    plan = json.loads(Path(argv[argv.index("run-plan") + 1]).read_text(encoding="utf-8"))
    intent = (plan.get("config") or {}).get("intent") or {}
    ratios = [int(r) for r in str(intent.get("chain") or "").split(",") if r]
    if not ratios:
        return answer
    experiment = answer["configuration"]["experiment"]
    root = experiment["domains"][0]
    dx = float(intent.get("root_dx_km") or 12) * 1000.0
    domains = [{**root, "run": {**root["run"], "dx": dx}}]
    for grid, ratio in enumerate(ratios, start=2):
        dx /= ratio
        domains.append({"grid_id": grid, "parent_id": grid - 1, "i_parent_start": 5, "j_parent_start": 5,
                        "parent_grid_ratio": ratio, "run": {"nx": 41, "ny": 37, "nz": root["run"]["nz"], "dx": dx}})
    return {**answer, "configuration": {"experiment": {**experiment, "domains": domains}}}


class PageRunner(FakeRunner):
    """The engine as the page sees it: two sources, one physics family with two schemes, answers at once."""

    def __init__(self) -> None:
        super().__init__()
        self.checks: list[dict] = []
        self.cyclone: list[list[str]] = []
        # What the engine projects a run writes: a storm-following Customise of Catarina on an 8 GB card.
        self.disk_bytes = int(30.2 * 2**30)

    def query(self, argv, *, cwd=None, timeout=0, log=None):
        if "--resolve" in argv:
            return nested({**super().query(argv, cwd=cwd, timeout=timeout, log=log),
                           "disk": {"total_bytes": self.disk_bytes}}, argv)
        if "--sources" in argv:
            self.queries.append(list(argv))
            return SOURCES
        if "--physics-profiles" in argv:
            self.queries.append(list(argv))
            return PROFILES
        if "physics-catalog" in argv and "--check" in argv:
            body = json.loads(argv[argv.index("--check") + 1])
            self.checks.append(body)
            pick = (body.get("choices") or {}).get("microphysics", "a")
            return {"valid": True, "words": "These run together.", "named_suite": "p1",
                    "named_suite_label": "Set one", "resolved": {"microphysics": pick}}
        if "physics-catalog" in argv:
            return CATALOG
        if "cyclone-setup" in argv:
            self.cyclone.append(list(argv))
            out = next((a.split("=", 1)[1] for a in argv if a.startswith("--out=")), None)
            if out is None and "--out" in argv:
                out = argv[argv.index("--out") + 1]
            if out:
                Path(out).write_text("# cyclone configuration\n", encoding="utf-8")
                return {**CYCLONE, "created": True, "config_path": out}
            return CYCLONE
        return super().query(argv, cwd=cwd, timeout=timeout, log=log)


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("gpuwm.fetch._head_ok", lambda url: True)
        patch.setattr("gpuwm.source_availability.quick_head", lambda url: True)
        # The event page's start refuses a run the disk cannot hold; the test machine's own free space is not
        # what these tests are about, and a dry run writes nothing.
        patch.setattr("gpuwm.gui.api.disk_free_gib", lambda path: 4096.0)
        runner = PageRunner()
        root = tmp_path_factory.mktemp("create") / "runs"
        server = build_server(root, port=PORT, port_tries=PORT_TRIES, runner=runner, token="t" * 43)
        serve_in_thread(server)
        try:
            yield server, runner, root
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


class Page:
    """One tab on the page server, with every request body the page sent."""

    def __init__(self, browser, server, **context):
        self.server = server
        self.context = browser.new_context(**context)
        self.page = self.context.new_page()
        self.sent: list[tuple[str, dict]] = []
        self.errors: list[str] = []
        self.page.on("pageerror", lambda err: self.errors.append(str(err)))
        self.page.on("request", self._request)
        self.page.goto(server.url)
        # The first page drawn, so the app shell is listening for a changed link: a link opened before that is
        # drawn once by the shell and once more by the hash change.
        self.page.wait_for_function("() => !!document.body.dataset.ready", timeout=20000)

    def _request(self, req):
        if req.method == "POST" and "/api/" in req.url:
            try:
                body = json.loads(req.post_data or "{}")
            except ValueError:
                body = {}
            self.sent.append((req.url.split(str(self.server.port), 1)[1], body))

    def open(self, route):
        self.page.goto(f"http://127.0.0.1:{self.server.port}/#/{route}")
        self.page.wait_for_selector(".steppanel h2")

    def hold(self, pattern):
        """Hold every request to ``pattern`` until :meth:`release`; the list of held requests."""

        held = []
        self.page.route(pattern, lambda route: held.append(route))
        self._held = (pattern, held)
        return held

    def wait_held(self, held, count, timeout=10000):
        for _ in range(timeout // 50):
            if len(held) >= count:
                return
            self.page.wait_for_timeout(50)
        raise AssertionError(f"only {len(held)} of {count} requests were asked")

    def release(self):
        pattern, held = self._held
        done = 0
        # Each held request is let through before the hold goes, so none is left waiting on a handler.
        for final in (False, True):
            if final:
                self.page.unroute(pattern)
            while done < len(held):
                try:
                    held[done].continue_()
                except Exception:  # noqa: BLE001 - a request of a page since reloaded has nothing to continue
                    pass
                done += 1

    def hash(self):
        return self.page.evaluate("location.hash")

    def bodies(self, path):
        return [body for url, body in self.sent if url == path]

    def form(self):
        return self.page.evaluate("import('/static/js/bridge.js').then((m) => m.readForm())")

    def step(self, name, timeout=15000):
        self.page.wait_for_function(
            "(n) => { const b = document.querySelector('nav.steps button.on'); return b && b.textContent.includes(n); }",
            arg=name, timeout=timeout)

    def next(self):
        button = self.page.locator(".steppanel .nav2 button.primary", has_text="Next")
        button.wait_for()
        self.page.wait_for_function("() => { const b = [...document.querySelectorAll('.steppanel .nav2 button')]"
                                    ".find((x) => x.textContent === 'Next'); return b && !b.disabled; }", timeout=20000)
        button.click()

    def close(self):
        self.context.close()


def plan_shape(body):
    """What a fit or a start is sized by: everything but the run's name and the pictures."""

    return {k: v for k, v in body.items() if k not in ("name", "products", "queue", "need_gib")}


@pytest.fixture()
def tab(browser, served):
    server, runner, _root = served
    tab = Page(browser, server)
    yield tab
    tab.close()
    assert not tab.errors, tab.errors


def to_fine(tab, route):
    tab.open(route)
    tab.step("When")
    tab.next()
    tab.step("How fine")


# ---------------------------------------------------------------- one draft, one plan

def test_a_grid_preview_prices_the_same_nests_a_click_on_it_gives(tab):
    to_fine(tab, "create/recipe/tornado-1999-466/8")
    tab.page.wait_for_function("() => document.querySelectorAll('.steppanel .option .fit.ok').length >= 4", timeout=20000)
    previews = {body["dx_km"]: plan_shape(body) for body in tab.bodies("/api/create/fit") if body.get("nz") is None}
    assert set(previews) >= {12, 3, 1.5, 0.75}
    # The event's own 12 km keeps its nests; any other spacing is one domain, in the preview as in the click.
    assert previews[12]["chain"] == "4,3" and previews[3]["chain"] is None
    for name, dx in (("Storms", 3), ("Overview", 12), ("Detail", 1.5)):
        tab.page.locator(".steppanel .option", has_text=name).first.click()
        assert plan_shape(tab.form()) == previews[dx], name


def test_a_nest_ladder_replaces_the_events_chain_and_buffers(tab, served):
    server, _runner, _root = served
    to_fine(tab, "create/recipe/tornado-1999-466/8")
    tab.page.wait_for_function("() => !!document.querySelector('.steppanel .option.on .fit.ok')", timeout=20000)
    tab.next()
    tab.step("Physics")
    tab.next()
    tab.step("Review")
    tab.page.locator(".steppanel details.fold > summary", has_text="More settings").click()
    tab.page.locator(".steppanel select:has(option[value='12-3'])").select_option("12-3")
    form = tab.form()
    assert form["ladder"] == "12-3"
    assert form["chain"] is None and form["buffer_km"] is None and form["dx_km"] is None
    reply, body = request(server, "POST", "/api/create/start", body={**form, "dry_run": True})
    assert reply.status == 200, body
    intent = body["plan"]["config"]["intent"]
    assert intent["ladder"] == "12-3" and "chain" not in intent and "buffer_km" not in intent


def test_a_ladder_beside_a_chain_is_refused_with_the_conflict_named(served):
    server, _runner, _root = served
    body = {"name": "both", "source": "era5", "cycle": "1999-08-11T12", "lat": 40.73, "lon": -111.87,
            "width_km": 200, "height_km": 200, "hours": 12, "card": "8gb", "ladder": "12-3",
            "chain": "4,3", "buffer_km": "680,180,0", "dry_run": True}
    reply, answer = request(server, "POST", "/api/create/start", body=body)
    assert reply.status == 400
    assert "ladder" in answer["message"] and "chain" in answer["message"]


def test_customise_keeps_a_storm_following_nest(tab, served):
    server, runner, _root = served
    to_fine(tab, "create/recipe/tc-2004086s29318/8")
    form = tab.form()
    # The page names the layout (its event and card size); the server reads its cyclone setup from the storm wiki.
    assert form["following"] is True and form["event"] == "tc-2004086s29318" and form["recipe_card_gb"] == 8
    assert "cyclone_setup" not in form
    assert form["dx_km"] == 12
    # The Fine step says the event's following nest stays at this grid.
    assert tab.page.locator(".steppanel", has_text="follows the storm").count() == 1
    reply, body = request(server, "POST", "/api/create/start", body={**form, "name": "catarina-custom", "dry_run": True})
    assert reply.status == 200, body
    assert body["plan"]["route"] == "experiment"
    prepare = body["prepare"]
    assert "cyclone-setup" in prepare and "--advisory-position=-28.90,-44.30" in prepare
    assert "--source=era5" in prepare and "--hours=42" in prepare and "--card=8gb" in prepare
    # The same cyclone setup the event page's button runs, history intervals included: they keep the run inside
    # the card's disk budget, and the setup's own default writes the nest four times as often.
    reply, event = request(server, "POST", "/api/wiki/simulate",
                           body={"event": "tc-2004086s29318", "card_gb": 8, "name": "catarina-event", "dry_run": True})
    assert reply.status == 200, event
    options = lambda argv: sorted(a for a in argv if a.startswith("--") and a not in ("--out", "--json"))
    assert options(prepare) == options(event["prepare"])
    # Any other grid is the plain grid, and the page says the following nest goes with it.
    tab.page.locator(".steppanel .option", has_text="Storms").first.click()
    moved = tab.form()
    assert moved["following"] is False and moved["event"] is None
    assert tab.page.locator(".steppanel", has_text="drops the storm-following nest").count() == 1


JOPLIN = "tornado-2011-1105221634-01"


def test_an_untouched_customise_shows_every_grid_and_plans_what_the_events_button_plans(tab, served):
    server, _runner, _root = served
    _, recipe = request(server, "GET", f"/api/wiki/recipe/{JOPLIN}?card=16")
    tab.open(f"create/recipe/{JOPLIN}/16")
    tab.step("When")
    panel = tab.page.locator(".steppanel")
    # It opens on the event's own start data, start and length, says so, and the start data is in view.
    form = tab.form()
    assert (form["source"], form["cycle"], form["hours"]) == (recipe["start_source"], recipe["start_cycle"], 15)
    assert panel.locator(".fromevent", has_text="are the event's").count() == 1
    chosen = panel.locator(".option.src.on")
    chosen.wait_for()
    box = chosen.bounding_box()
    assert box["y"] + box["height"] <= tab.page.viewport_size["height"], box
    tab.next()
    tab.step("How fine")
    # The chosen grid is the event's best run, first in the list, every grid of it named.
    grid = ".steppanel .options:not(.levels) .option"
    tab.page.wait_for_function("(g) => { const o = document.querySelector(g + '.on');"
                               " return !!(o && o.querySelector('.fit.ok') && o.textContent.includes('1 km')); }",
                               arg=grid, timeout=20000)
    shown = tab.page.locator(f"{grid}.on").inner_text()
    assert "The event's best run" in shown and "12 km / 3 km / 1 km" in shown, shown
    assert tab.page.locator(grid).first.inner_text() == shown
    tab.next()
    tab.step("Physics")
    assert panel.locator(".fromevent", has_text="The event's best run uses").count() == 1
    tab.next()
    tab.step("Review")
    review = panel.locator("table.review").inner_text()
    assert "The event's best run: 12 km, 3 km, 1 km" in review, review
    assert "every 30 minutes on the nests" in review and "The standard set" in review, review
    form = tab.form()
    reply, custom = request(server, "POST", "/api/create/start", body={**form, "name": "joplin-same", "dry_run": True})
    assert reply.status == 200, custom
    reply, button = request(server, "POST", "/api/wiki/simulate",
                            body={"event": JOPLIN, "card_gb": 16, "name": "joplin-same", "dry_run": True})
    assert reply.status == 200, button
    assert custom["plan"] == button["plan"]


def test_customise_takes_another_listed_source_and_length(tab, served):
    server, _runner, _root = served
    _, recipe = request(server, "GET", f"/api/wiki/recipe/{JOPLIN}?card=16")
    tab.open(f"create/recipe/{JOPLIN}/16")
    tab.step("When")
    # A recent start, which GFS holds, then GFS from the list and another length.
    tab.page.locator(".steppanel .choices button", has_text="Yesterday").click()
    gfs = tab.page.locator(".steppanel .option.src", has_text="GFS")
    gfs.wait_for()
    gfs.click()
    tab.page.locator(".steppanel .choices button", has_text="24 h").click()
    form = tab.form()
    assert form["source"] == "gfs" and form["hours"] == 24 and form["cycle"] != recipe["start_cycle"]
    reply, body = request(server, "POST", "/api/create/start", body={**form, "name": "joplin-gfs", "dry_run": True})
    assert reply.status == 200, body
    intent = body["plan"]["config"]["intent"]
    assert (intent["source"], intent["cycle"], intent["hours"]) == ("gfs", form["cycle"], 24)
    # The event's grids and the rest of its best run stay; the ERA5 copy the event read from does not.
    assert intent["chain"] == recipe["chain"] and intent["nest_history_interval_s"] == 1800
    assert "era5_provider" not in intent


def test_the_event_page_puts_customise_beside_its_button_and_says_the_start(tab):
    tab.page.goto(f"http://127.0.0.1:{tab.server.port}/#/event/{JOPLIN}")
    tab.page.locator(".seg button", has_text="16 GB").click()
    route = f"#/create/recipe/{JOPLIN}/16"
    custom = tab.page.locator(".runacts a.btn", has_text="Customise this run")
    custom.wait_for()
    assert custom.get_attribute("href") == route
    starts = tab.page.locator(".runp .note.starts")
    assert "Starts from ERA5, 2011-05-22 12:00 UTC, 15 h." in starts.inner_text()
    assert starts.locator("a", has_text="Change").get_attribute("href") == route
    custom.click()
    tab.step("When")
    tab.page.locator(".steppanel .option.src.on").wait_for()
    assert tab.form()["source"] == "era5"


CATARINA = "tc-2004086s29318"


def test_a_kept_following_layout_names_its_physics_and_intervals(tab):
    # The 8 GB best run of this cyclone is its cyclone setup, which sets the physics and how often each grid
    # writes; kept whole, Review and the Physics step say both.
    to_fine(tab, f"create/recipe/{CATARINA}/8")
    assert tab.form()["following"] is True
    tab.next()
    tab.step("Physics")
    panel = tab.page.locator(".steppanel")
    assert panel.locator(".fromevent", has_text="The event's best run uses").count() == 1
    tab.next()
    tab.step("Review")
    review = panel.locator("table.review").inner_text()
    assert "Written" in review and "Every hour on the outer grid" in review, review


def test_a_following_layout_left_behind_takes_its_physics_and_intervals_with_it(tab, served):
    # Levels of its own drop the storm-following nest, and with it the cyclone setup that sets the best run's
    # physics and intervals: Review said "Written: every hour, as the event's best run writes" and the Physics step
    # named the best run's set, for a plan that runs neither.
    server, _runner, _root = served
    to_fine(tab, f"create/recipe/{CATARINA}/8")
    tab.page.locator(".steppanel .options.levels .option", has_text="More").first.click()
    form = tab.form()
    assert form["following"] is False and form["event"] == CATARINA
    tab.next()
    tab.step("Physics")
    panel = tab.page.locator(".steppanel")
    physics_note = panel.locator(".fromevent", has_text="The event's best run uses").count()
    tab.next()
    tab.step("Review")
    review = panel.locator("table.review").inner_text()
    assert ("Written" in review, physics_note) == (False, 0), review
    reply, body = request(server, "POST", "/api/create/start",
                          body={**tab.form(), "name": "catarina-levels", "dry_run": True})
    assert reply.status == 200, body
    assert "history_interval_s" not in body["plan"]["config"]["intent"]


def test_another_start_data_keeps_the_intervals_and_drops_the_best_runs_physics_note(tab, served):
    # From GFS the plan still writes as the best run does (the server carries its intervals with its grids), but
    # it runs GFS's own physics set, not the one the event's ERA5 run used.
    server, _runner, _root = served
    tab.open(f"create/recipe/{JOPLIN}/16")
    tab.step("When")
    tab.page.locator(".steppanel .choices button", has_text="Yesterday").click()
    gfs = tab.page.locator(".steppanel .option.src", has_text="GFS")
    gfs.wait_for()
    gfs.click()
    assert tab.form()["source"] == "gfs"
    tab.next()
    tab.step("How fine")
    tab.page.wait_for_function("() => !!document.querySelector('.steppanel .option.on .fit.ok')", timeout=20000)
    tab.next()
    tab.step("Physics")
    panel = tab.page.locator(".steppanel")
    assert panel.locator(".fromevent", has_text="The event's best run uses").count() == 0
    tab.next()
    tab.step("Review")
    review = panel.locator("table.review").inner_text()
    assert "every 30 minutes on the nests" in review, review
    reply, body = request(server, "POST", "/api/create/start",
                          body={**tab.form(), "name": "joplin-gfs-review", "dry_run": True})
    assert reply.status == 200, body
    assert body["plan"]["config"]["intent"]["nest_history_interval_s"] == 1800


def test_a_following_draft_is_priced_by_the_cyclone_setup_and_starts_through_it(served):
    server, runner, root = served
    base = {"name": "follow-start", "source": "era5", "cycle": "2004-03-26T18", "lat": -28.9, "lon": -44.3,
            "width_km": 2388, "height_km": 1908, "hours": 42, "card": "8gb", "dx_km": 12, "following": True,
            "event": "tc-2004086s29318", "recipe_card_gb": 8}
    reply, fit = request(server, "POST", "/api/create/fit", body=base)
    assert reply.status == 200, fit
    assert [d["dx_km"] for d in fit["fit"]["domains"]] == [12.0, 3.0]
    assert fit["fit"]["memory"] == {"need_gib": 5.98, "budget_gib": 6.75, "fits": True}
    reply, started = request(server, "POST", "/api/create/start", body=base)
    assert reply.status == 200, started
    plan = json.loads((root / "follow-start" / "plan.json").read_text(encoding="utf-8"))
    assert plan["route"] == "experiment" and plan["config"]["path"].endswith("cyclone.toml")
    assert (root / "follow-start" / "cyclone.toml").is_file()
    # A layout the following setup cannot honour is refused, not silently run without the nest.
    reply, answer = request(server, "POST", "/api/create/start",
                            body={**base, "name": "follow-ladder", "dx_km": None, "ladder": "12-3", "dry_run": True})
    assert reply.status == 400 and "follow" in answer["message"]


def test_a_following_start_the_disk_cannot_hold_is_refused_and_a_queued_one_waits(served, monkeypatch):
    server, runner, root = served
    # The event page's button refuses this same run on this disk (HTTP 507); Customise now carries its following
    # nest, so it refuses it too, on the engine's own projection of the plan the cyclone setup wrote.
    monkeypatch.setattr("gpuwm.gui.api.disk_free_gib", lambda path: 25.0)
    body = {"name": "follow-full", "source": "era5", "cycle": "2004-03-26T18", "lat": -28.9, "lon": -44.3,
            "width_km": 2388, "height_km": 1908, "hours": 42, "card": "8gb", "dx_km": 12, "following": True,
            "event": "tc-2004086s29318", "recipe_card_gb": 8}
    launched = len(runner.launched)
    reply, answer = request(server, "POST", "/api/create/start", body=body)
    assert reply.status == 507, answer
    assert "about 30 GiB" in answer["message"] and "25 GiB free" in answer["message"]
    assert not (root / "follow-full").exists() and len(runner.launched) == launched
    # Queued, it waits with its projection, and the queue holds it until the disk has the room.
    reply, answer = request(server, "POST", "/api/create/start", body={**body, "name": "follow-later", "queue": True})
    assert reply.status == 200 and answer["queued"], answer
    marker = json.loads((root / "follow-later" / runs.QUEUED).read_text(encoding="utf-8"))
    assert marker["disk_gib"] == 30.2
    held = server.api.queue.hold_reason(marker, {}, LOCAL)
    assert held and "about 30 GiB" in held and "25.0 GiB free" in held
    reply, _ = request(server, "POST", "/api/queue/follow-later/remove")
    assert reply.status == 200


def test_reloading_review_keeps_the_whole_draft(tab):
    tab.open("create/40,-100,600x600/when/at/2020-06-01T12/src/era5")
    tab.step("When")
    tab.page.locator(".steppanel .choices button", has_text="24 h").click()
    tab.next()
    tab.step("How fine")
    tab.page.locator(".steppanel .choices button", has_text="8 GB").click()
    tab.page.locator(".steppanel .option", has_text="Overview").first.click()
    tab.next()
    tab.step("Physics")
    tab.next()
    tab.step("Review")
    before = tab.form()
    assert (before["hours"], before["dx_km"], before["card"]) == (24, 12, "8gb")
    tab.page.reload()
    tab.step("Review")
    after = tab.form()
    assert plan_shape(after) == plan_shape(before)
    assert after["name"] == before["name"]


def test_named_section_survives_review_reload_and_requires_a_line(tab, served):
    server, _runner, _root = served
    tab.open("create/35,-100,600x600/when/at/2020-06-01T12/src/era5")
    tab.step("When")
    tab.next()
    tab.step("How fine")
    tab.page.locator(".steppanel .option", has_text="Overview").first.click()
    tab.next()
    tab.step("Physics")
    tab.next()
    tab.step("Review")
    tab.page.locator(".steppanel details.fold > summary", has_text="More settings").click()
    tab.page.locator(".steppanel select:has(option[value='custom'])").select_option("custom")
    tab.page.get_by_label("Product names", exact=True).fill("xsec:wa")
    tab.page.get_by_label("Cross-section line", exact=False).fill("35,-100,36,-99")
    form = tab.form()
    assert form["products"] == "xsec:wa"
    assert form["render_section"] == "35,-100,36,-99"
    tab.page.reload()
    tab.step("Review")
    assert tab.form()["render_section"] == form["render_section"]
    response, body = request(server, "POST", "/api/create/start", body={**form, "dry_run": True})
    assert response.status == 200, body
    assert body["plan"]["run_options"]["render_section"] == form["render_section"]
    response, body = request(server, "POST", "/api/create/start",
                             body={**form, "render_section": None, "dry_run": True})
    assert response.status == 400
    assert "xsec:wa" in body["message"] and "line" in body["message"]


def test_an_emptied_named_list_stays_named_through_a_reload_and_is_refused(tab, served):
    """Clearing Product names sent the standard set while the page showed Named pictures."""
    server, _runner, _root = served
    tab.open("create/35,-100,600x600/when/at/2020-06-01T12/src/era5")
    tab.step("When")
    tab.next()
    tab.step("How fine")
    tab.page.locator(".steppanel .option", has_text="Overview").first.click()
    tab.next()
    tab.step("Physics")
    tab.next()
    tab.step("Review")
    tab.page.locator(".steppanel details.fold > summary", has_text="More settings").click()
    pick = tab.page.locator(".steppanel select:has(option[value='custom'])")
    pick.select_option("custom")
    tab.page.get_by_label("Product names", exact=True).fill("")
    form = tab.form()
    assert form["products"] is None and form["products_named"] is True
    tab.page.reload()
    tab.step("Review")
    tab.page.locator(".steppanel details.fold > summary", has_text="More settings").click()
    assert pick.input_value() == "custom"
    assert tab.form()["products_named"] is True
    response, body = request(server, "POST", "/api/create/start", body={**form, "dry_run": True})
    assert response.status == 400, body
    assert "standard set" in body["message"] and "named" in body["message"]


def test_a_reload_while_the_sources_load_keeps_the_whole_draft(tab):
    link = "#/create/40,-100,600x600/review/at/2020-06-01T12/src/era5"
    tab.open("create/40,-100,600x600/when/at/2020-06-01T12/src/era5")
    tab.step("When")
    tab.page.locator(".steppanel .choices button", has_text="24 h").click()
    tab.next()
    tab.step("How fine")
    tab.page.locator(".steppanel .choices button", has_text="8 GB").click()
    tab.page.locator(".steppanel .option", has_text="Overview").first.click()
    tab.next()
    tab.step("Physics")
    tab.next()
    tab.step("Review")
    before = tab.form()
    assert tab.hash() == link
    held = tab.hold("**/api/sources")
    tab.page.reload()
    tab.wait_held(held, 1)
    # The link keeps its date and source while the list is on its way, so the kept draft is still found by it.
    assert tab.hash() == link
    tab.page.reload()
    tab.wait_held(held, 2)
    assert tab.hash() == link
    tab.release()
    tab.step("Review")
    after = tab.form()
    assert plan_shape(after) == plan_shape(before)
    assert tab.page.locator(".noticebox .notice", has_text="start again").count() == 0


def test_a_link_drawn_twice_while_the_sources_load_keeps_its_date_and_source(tab):
    # A link opened while the app shell was still drawing its first page is drawn twice; the first drawing took the
    # date and source out of the link, and the second opened on the newest start of the first source.
    link = "#/create/40,-100,600x600/when/at/2026-09-24T12/src/gfs"
    held = tab.hold("**/api/sources")
    tab.open(link[2:])
    tab.wait_held(held, 1)
    assert tab.hash() == link
    tab.page.evaluate("window.dispatchEvent(new HashChangeEvent('hashchange'))")
    tab.wait_held(held, 2)
    assert tab.hash() == link
    tab.release()
    tab.step("When")
    tab.next()
    tab.step("How fine")
    form = tab.form()
    assert (form["cycle"], form["source"]) == ("2026-09-24T12", "gfs")


def test_a_link_to_review_without_its_draft_starts_again_at_where_and_says_so(browser, served):
    server, _runner, _root = served
    fresh = Page(browser, server)
    try:
        fresh.open("create/40,-100,600x600/review/at/2020-06-01T12/src/era5")
        fresh.step("Where")
        assert fresh.page.locator(".noticebox .notice", has_text="start again").count() == 1
    finally:
        fresh.close()


def test_review_shows_the_forecast_start_after_the_start_hour(tab):
    tab.open("create/40,-100,600x600/when/at/2026-09-24T12/src/gfs")
    tab.step("When")
    tab.next()
    tab.step("How fine")
    tab.next()
    tab.step("Physics")
    tab.next()
    tab.step("Review")
    tab.page.locator(".steppanel details.fold > summary", has_text="More settings").click()
    hour = tab.page.locator(".steppanel input[type=number][max='384']")
    hour.fill("30")
    hour.dispatch_event("change")
    table = tab.page.locator(".steppanel table.review")
    start = table.locator("tr", has_text="Start").first.inner_text()
    assert "2026-09-25 18:00 UTC" in start
    assert "2026-09-24 12:00 UTC" in table.inner_text() and "forecast hour 30" in table.inner_text()


# ---------------------------------------------------------------- Start and Queue it on a Machines node

class TwoNodes(Registry):
    """Two Machines nodes answered here, with no SSH; a node named in ``running`` has a forecast on its card."""

    def __init__(self, path, running=()):
        super().__init__(path)
        self.nodes = {name: FakeMachine({"name": name, "kind": "ssh", "host": f"me@{name}", "workspace": f"/w/{name}"})
                      for name in ("worker-a", "worker-b")}
        self.running = set(running)

    def rows(self):
        return [node.row for node in self.nodes.values()]

    def get(self, name):
        return self.nodes[name] if name in self.nodes else super().get(name)

    def probe(self, name, *, fresh=False):
        return {"name": name, "state": "running" if name in self.running else "idle", "card": "16gb",
                "version_here": "9.9", "version_there": "9.9", "version_matches": True}

    def listing(self, *, fresh=False):
        return [self.probe(name) for name in self.nodes]


@pytest.fixture()
def nodes(tmp_path, monkeypatch):
    monkeypatch.setattr("gpuwm.fetch._head_ok", lambda url: True)
    monkeypatch.setattr("gpuwm.source_availability.quick_head", lambda url: True)
    server = build_server(tmp_path / "runs", port=PORT, port_tries=PORT_TRIES, runner=PageRunner(), token="t" * 43,
                          machines=TwoNodes(tmp_path / "machines.toml", running=("worker-b",)))
    serve_in_thread(server)
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def shows_command(tab, button, text):
    """Wait until the bar's ``button`` has its Show command open on the page, drawn from a reply whose exact line
    (the one Copy takes) holds ``text``."""

    try:
        tab.page.wait_for_function(
            """([label, text]) => [...document.querySelectorAll('.steppanel .nav2 .btns .action')].some((a) => {
                 const b = a.querySelector('button.big'); const box = a.querySelector('.showcmd.live');
                 const out = box && box.querySelector('.livecmd');
                 return b && b.textContent === label && out && !out.hidden && !!out.querySelector('.copyline code')
                   && box.dataset.command.includes(text); })""",
            arg=[button, text], timeout=20000)
    except playwright_api.TimeoutError:
        bar = tab.page.evaluate("() => [...document.querySelectorAll('.steppanel .nav2 .btns .action')].map((a) =>"
                                " [a.innerText, (a.querySelector('.showcmd.live') || {dataset: {}}).dataset.command])")
        raise AssertionError(f"{button} never showed {text!r}; the button bar shows {bar!r}") from None


def test_show_command_runs_on_the_machine_picked_and_follows_a_new_pick(browser, nodes):
    tab = Page(browser, nodes)
    try:
        tab.open("create/40,-100,600x600/when/at/2020-06-01T12/src/era5")
        tab.step("When")
        tab.next()
        tab.step("How fine")
        tab.next()
        tab.step("Physics")
        tab.next()
        tab.step("Review")
        pick = tab.page.locator(".steppanel select[aria-label='Run it on']")
        pick.wait_for()
        pick.select_option("worker-a")
        # Once worker-a's first check is in, its card is free and the bar offers Start.
        tab.page.wait_for_function("() => [...document.querySelectorAll('.steppanel table.review tr')].some((r) =>"
                                   " r.cells[0].textContent === 'Starts' && r.cells[1].textContent === 'Now.')",
                                   timeout=20000)
        tab.page.locator(".steppanel .nav2 .btns .showcmd.live > button").click()
        # Start's line is worker-a's own interpreter and run folder, not this computer's.
        shows_command(tab, "Start forecast", "/w/worker-a/venv/bin/python -m gpuwm run-plan /w/worker-a/runs/")
        # Picking worker-b, whose card is taken, draws Queue it in Start's place with the Show command still open,
        # and its line is worker-b's.
        pick.select_option("worker-b")
        shows_command(tab, "Queue it", "/w/worker-b/venv/bin/python -m gpuwm run-plan /w/worker-b/runs/")
        asked = tab.bodies("/api/create/start")
        assert asked and all(body["dry_run"] is True for body in asked)
        assert all(body.get("machine") in ("worker-a", "worker-b") for body in asked), asked
        assert any(body["machine"] == "worker-a" and body["queue"] is False for body in asked), asked
        assert asked[-1]["machine"] == "worker-b" and asked[-1]["queue"] is True, asked
    finally:
        tab.close()
    assert not tab.errors, tab.errors


# ---------------------------------------------------------------- physics

def test_a_set_picked_again_after_a_cancelled_wait_is_checked(tab, served):
    _server, runner, _root = served
    to_fine(tab, "create/40,-100,600x600/when/at/2020-06-01T12/src/era5")
    tab.next()
    tab.step("Physics")
    tab.page.locator(".steppanel details.fold > summary", has_text="Microphysics").click()
    a = tab.page.locator(".steppanel tr.row", has_text="Scheme A")
    b = tab.page.locator(".steppanel tr.row", has_text="Scheme B")
    a.click()
    b.click()
    tab.page.wait_for_function("() => document.querySelector('.steppanel').textContent.includes('Set one')", timeout=10000)
    before = len([c for c in runner.checks if (c.get("choices") or {}).get("microphysics") == "a"])
    tab.page.locator(".steppanel tr.row", has_text="Scheme A").click()
    tab.page.wait_for_timeout(1200)
    after = [c for c in runner.checks if (c.get("choices") or {}).get("microphysics") == "a"]
    assert len(after) == before + 1
    assert tab.page.locator(".steppanel", has_text="The engine is checking this set").count() == 0


def test_physics_schemes_are_picked_by_keyboard(tab):
    to_fine(tab, "create/40,-100,600x600/when/at/2020-06-01T12/src/era5")
    tab.next()
    tab.step("Physics")
    tab.page.locator(".steppanel details.fold > summary", has_text="Microphysics").click()
    radio = tab.page.locator(".steppanel input[type=radio]")
    assert radio.count() == 2
    radio.first.focus()
    tab.page.keyboard.press("ArrowDown")
    tab.page.wait_for_function("() => { const r = document.querySelectorAll('.steppanel input[type=radio]');"
                               " return r.length === 2 && r[1].checked; }", timeout=5000)
    assert tab.form()["physics_choices"] == {"microphysics": "b"}


# ---------------------------------------------------------------- input

def test_the_create_map_keeps_a_touch_drag(browser, served):
    server, _runner, _root = served
    phone = Page(browser, server, viewport={"width": 375, "height": 812}, has_touch=True, is_mobile=True)
    try:
        phone.open("create/35,-100,600x600/where")
        canvas = phone.page.locator(".mapwrap canvas.map")
        phone.page.evaluate("""() => { window.moves = []; const c = document.querySelector('.mapwrap canvas.map');
          for (const k of ['pointermove', 'pointercancel', 'pointerup']) c.addEventListener(k, () => moves.push(k)); }""")
        box = canvas.bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        cdp = phone.context.new_cdp_session(phone.page)
        touch = lambda kind, points: cdp.send("Input.dispatchTouchEvent", {"type": kind, "touchPoints": points})
        touch("touchStart", [{"x": x, "y": y}])
        for i in range(1, 9):
            touch("touchMove", [{"x": x, "y": y - 15 * i}])
        touch("touchEnd", [])
        moves = phone.page.evaluate("window.moves")
        # Every move reaches the box and the drag ends as a lift, not as a cancel taken over by a page pan.
        assert "pointercancel" not in moves and moves.count("pointermove") >= 8 and moves[-1] == "pointerup"
        assert phone.form()["lat"] > 35.5
        assert canvas.evaluate("(el) => getComputedStyle(el).touchAction") == "none"
    finally:
        phone.close()


def test_files_rows_open_by_keyboard(tab, served):
    _server, _runner, root = served
    make_run(root, "keyboard-run", plan=True, frames=4)
    tab.page.goto(f"http://127.0.0.1:{tab.server.port}/#/explore")
    link = tab.page.locator("table.dtable a", has_text="keyboard-run")
    link.wait_for()
    link.focus()
    tab.page.keyboard.press("Enter")
    tab.page.wait_for_function("() => location.hash.startsWith('#/explore/keyboard-run')")
    group = tab.page.locator("table.dtable tbody button").first
    group.wait_for()
    group.focus()
    tab.page.keyboard.press("Enter")
    tab.page.wait_for_selector(".thumbs img")
    assert group.get_attribute("aria-pressed") == "true"


# ---------------------------------------------------------------- assistant settings

def test_saving_settings_twice_keeps_a_custom_decision_address(tab, served):
    server, _runner, _root = served
    reply, _ = request(server, "POST", "/api/assistant/settings",
                       body={"decisions": "jev", "decision_url": "https://choices.example/v1"})
    assert reply.status == 200
    reply, status = request(server, "GET", "/api/assistant")
    assert status["decision_url"] == "https://choices.example/v1"
    tab.page.goto(f"http://127.0.0.1:{tab.server.port}/#/runs")
    tab.page.evaluate("import('/static/js/assistant.js').then((m) => m.openAssistant())")
    drawer = tab.page.locator("aside.drawer")
    drawer.locator("details.fold > summary").first.click()
    address = drawer.locator("label", has_text="Choice server address").locator("input")
    tab.page.wait_for_function("(el) => el.value === 'https://choices.example/v1'", arg=address.element_handle())
    for _ in range(2):
        drawer.locator("button", has_text="Save").click()
        tab.page.wait_for_timeout(300)
    reply, status = request(server, "GET", "/api/assistant")
    assert status["decision_url"] == "https://choices.example/v1"
    saves = tab.bodies("/api/assistant/settings")
    assert saves and all("decision_url" not in body for body in saves)


def test_settings_opened_before_the_status_arrives_keep_what_was_saved(tab, served):
    # Seen on the real page server: Settings opened while the status was still on its way was never filled in,
    # and its Save then sent the blank address and the first "who answers" option over the saved ones.
    server, _runner, _root = served
    reply, _ = request(server, "POST", "/api/assistant/settings",
                       body={"decisions": "jev", "decision_url": "https://choices.example/v2"})
    assert reply.status == 200
    held = []
    tab.page.route("**/api/assistant", lambda route: held.append(route))
    tab.page.goto(f"http://127.0.0.1:{tab.server.port}/#/runs")
    tab.page.evaluate("import('/static/js/assistant.js').then((m) => m.openAssistant())")
    drawer = tab.page.locator("aside.drawer")
    drawer.locator("details.fold > summary").first.click()
    for _ in range(100):
        if held:
            break
        tab.page.wait_for_timeout(50)
    assert held, "the panel never asked for its status"
    for route in held:
        route.continue_()
    tab.page.unroute("**/api/assistant")
    address = drawer.locator("label", has_text="Choice server address").locator("input")
    tab.page.wait_for_function("(el) => el.value === 'https://choices.example/v2'", arg=address.element_handle())
    assert drawer.locator("label", has_text="Who answers the choices").locator("select").input_value() == "jev"
    # A field the person edits is kept through a later status, and it is the only one a save sends.
    model = drawer.locator("label", has_text="Model name on that server").locator("input")
    model.fill("typed-model")
    tab.page.evaluate("import('/static/js/assistant.js').then((m) => m.openAssistant())")
    tab.page.wait_for_timeout(300)
    assert model.input_value() == "typed-model"
    drawer.locator("button", has_text="Save").click()
    tab.page.wait_for_timeout(300)
    reply, status = request(server, "GET", "/api/assistant")
    assert (status["decision_url"], status["decisions"]) == ("https://choices.example/v2", "jev")
    assert status["endpoint_model"] == "typed-model"
    saves = tab.bodies("/api/assistant/settings")
    assert saves == [{"endpoint_model": "typed-model"}]


def test_a_field_edited_while_its_save_is_on_its_way_keeps_the_newer_value(tab, served):
    server, _runner, _root = served
    tab.page.goto(f"http://127.0.0.1:{tab.server.port}/#/runs")
    status_read = lambda r: r.url.endswith("/api/assistant") and r.request.method == "GET"
    with tab.page.expect_response(status_read):
        tab.page.evaluate("import('/static/js/assistant.js').then((m) => m.openAssistant())")
    drawer = tab.page.locator("aside.drawer")
    drawer.locator("details.fold > summary").first.click()
    model = drawer.locator("label", has_text="Model name on that server").locator("input")
    model.fill("sent-model")
    held = tab.hold("**/api/assistant/settings")
    drawer.locator("button", has_text="Save").click()
    tab.wait_held(held, 1)
    model.fill("newer-model")
    with tab.page.expect_response(status_read):
        tab.release()
    tab.page.wait_for_timeout(300)
    reply, status = request(server, "GET", "/api/assistant")
    assert status["endpoint_model"] == "sent-model"
    # The status read after the save fills the fields nobody edited since; the newer edit stays on screen.
    assert model.input_value() == "newer-model"
    with tab.page.expect_response(status_read):
        drawer.locator("button", has_text="Save").click()
    reply, status = request(server, "GET", "/api/assistant")
    assert status["endpoint_model"] == "newer-model"
    saves = tab.bodies("/api/assistant/settings")
    assert [body.get("endpoint_model") for body in saves] == ["sent-model", "newer-model"]
