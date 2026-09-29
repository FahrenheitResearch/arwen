"""The run map in a real browser keeps a machine's run, its drawing and its warnings up to date, or says why not.

Three defects, each on the shipped map page against the real page server and
real run folders (the follower and the engine are the test's; no forecast runs
and no weather picture is drawn):

- a machine's run whose follower ended showed the last state received
  (running, with Stop) and nothing more; the map now says its updates stopped
  and Resume updates follows it again without starting anything;
- a forecast that had finished stopped reading while a render worker still
  drew its pictures, so the open map said "No maps yet" after they had all
  arrived;
- the engine's library warning (a warm bubble above 10 K) was in the run's
  events and never on the map, live or after a reload.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gpuwm import proc_identity
from gpuwm.gui import runs
from gpuwm.gui.machines import FOLLOW
from gpuwm.gui.server import build_server, serve_in_thread
from test_gui_create_draft import PORT, PORT_TRIES, Page, PageRunner

playwright_api = pytest.importorskip("playwright.sync_api")
expect = playwright_api.expect

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
WARM = ("perturbation bubble amplitude_k = 15 K is above 10 K and 5.0 times WRF's idealized warm bubble "
        "(3 K, em_quarter_ss): it replaces the analysis near its center rather than nudging it. It runs as "
        "configured, and this warning is recorded in the perturbation receipt.")


def live_job(folder: Path) -> dict:
    return {"job_id": "job-follow", "jobs_dir": str(folder), "wrapper_pid": os.getpid(),
            "wrapper_process": proc_identity.identify(os.getpid())}


class FollowRunner(PageRunner):
    """The page's engine, with a follower that is this test's own process (so it reads as running)."""

    def __init__(self) -> None:
        super().__init__()
        self.helpers: list[list[str]] = []

    def launch_helper(self, rundir, argv, kind):
        self.helpers.append(list(argv))
        return live_job(Path(rundir) / f".follow-{len(self.helpers)}")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    server = build_server(tmp_path_factory.mktemp("updates") / "runs", port=PORT, port_tries=PORT_TRIES,
                          runner=FollowRunner(), token="t" * 43)
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
    tab = Page(browser, served, viewport={"width": 1366, "height": 800})
    yield tab
    tab.close()
    assert not tab.errors, tab.errors


def write_events(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def open_map(tab, route: str) -> None:
    tab.page.goto(f"http://127.0.0.1:{tab.server.port}/#/{route}")
    tab.page.wait_for_function("() => document.body.dataset.ready === '1'", timeout=20000)


def test_a_machine_run_whose_updates_stopped_says_so_and_resumes(tab):
    server = tab.server
    rundir = server.root / "worker-run"
    rundir.mkdir()
    (rundir / runs.REMOTE).write_text(json.dumps({"machine": "worker", "alive": True, "ended": False,
                                                  "checked_utc": "2026-09-28T00:00:00Z",
                                                  "job": {"state": "running"}}), encoding="utf-8")
    write_events(rundir / runs.EVENTS, [{"event": "plan_accepted", "sequence": 1},
                                        {"event": "stage_started", "sequence": 2, "stage": "forecast"}])
    ended = rundir / ".follow-gave-up"
    ended.mkdir()
    (ended / "result.json").write_text(json.dumps({"exit_code": 3}), encoding="utf-8")
    (rundir / FOLLOW).write_text(json.dumps(live_job(ended)), encoding="utf-8")

    open_map(tab, "watch/worker-run")
    page = tab.page
    note = page.locator(".mvnotes .loose", has_text="Updates from the machine stopped")
    expect(note).to_be_visible()
    # The live line keeps the last state received, without the pulse that says it is live.
    expect(page.locator(".liveline .pulse")).to_have_count(0)
    helpers = len(server.api.runner.helpers)
    note.get_by_role("button", name="Resume updates").click()
    expect(note).to_be_hidden(timeout=15000)
    assert len(server.api.runner.helpers) == helpers + 1 and "follow" in server.api.runner.helpers[-1]
    # Nothing was launched but the follower: no forecast, no drawing.
    assert server.api.runner.launched == []
    assert not any(path.endswith("/render") for path, _ in tab.sent)
    expect(page.locator(".liveline .pulse")).to_have_count(1)


def test_a_finished_forecast_keeps_reading_while_its_machine_still_draws(tab):
    server = tab.server
    rundir = server.root / "drawn-run"
    rundir.mkdir()
    write_events(rundir / runs.EVENTS, [{"event": "plan_accepted", "sequence": 1},
                                        {"event": "completed", "sequence": 2}])
    (rundir / runs.RENDER).write_text(json.dumps({"machine": "worker", "state": "rendering", "pictures": 0}),
                                      encoding="utf-8")
    (rundir / FOLLOW).write_text(json.dumps(live_job(rundir / ".follow-live")), encoding="utf-8")
    # The map of a drawn run fetches its picture files; this timing test has none worth drawing.
    tab.page.route("**/api/runs/drawn-run/files/**", lambda route: route.abort())

    open_map(tab, "results/drawn-run")
    page = tab.page
    expect(page.locator(".statecard")).to_contain_text("No maps yet")
    folder = rundir / "render-worker" / "d01-12km" / "2m_temperature" / "2026-09-27"
    folder.mkdir(parents=True)
    (folder / "rustwx_wrf_20260927_12z_f001_d01-12km_2m_temperature.png").write_bytes(PNG)
    (rundir / runs.RENDER).write_text(json.dumps({"machine": "worker", "state": "finished", "pictures": 1}),
                                      encoding="utf-8")
    # No reload and no navigation: the open map reads the worker's pictures as they land.
    expect(page.locator(".picker-btn")).to_contain_text("Temperature", timeout=20000)
    expect(page.locator(".statecard")).to_be_hidden()


def test_a_library_warning_is_on_the_map_live_and_after_a_reload(tab):
    server = tab.server
    rundir = server.root / "warm-run"
    rundir.mkdir()
    write_events(rundir / runs.EVENTS, [
        {"event": "plan_accepted", "sequence": 1},
        {"event": "warning", "sequence": 2, "code": "library_warning", "message": WARM, "detail": ""},
        {"event": "stage_started", "sequence": 3, "stage": "forecast"}])
    # Running here: its job is this test's own process.
    (rundir / runs.JOB).write_text(json.dumps(live_job(rundir / ".job")), encoding="utf-8")

    open_map(tab, "watch/warm-run")
    page = tab.page
    head = page.locator(".mvnotes .warnnote summary")
    expect(head).to_have_text("Run warnings (1)")
    head.click()
    expect(page.locator(".mvnotes .runwarnings")).to_contain_text("amplitude_k = 15 K is above 10 K")
    # A warning that lands while the map is open is shown without a reload.
    write_events(rundir / runs.EVENTS, [
        {"event": "warning", "sequence": 4, "code": "library_warning", "message": "Another setting needs review.",
         "detail": "It runs as configured."}])
    expect(head).to_have_text("Run warnings (2)", timeout=15000)
    expect(page.locator(".mvnotes .runwarnings")).to_contain_text("It runs as configured.")
    write_events(rundir / runs.EVENTS, [{"event": "completed", "sequence": 5}])
    (rundir / runs.JOB).unlink()
    page.reload()
    page.wait_for_function("() => document.body.dataset.ready === '1'", timeout=20000)
    expect(page.locator(".mvnotes .warnnote summary")).to_have_text("Run warnings (2)")
