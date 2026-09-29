"""The Machines page's This computer row says which forecast holds the card, not a launch refusal.

The defect this file holds shut: while a forecast ran, the Doing column of
This computer read the job manager's launch refusal, "the GPU is held by
running job job-... so this launch is refused -- poll job_status job-...,
or job_cancel it, then relaunch."  Nothing was being launched, and
job_status and job_cancel are tool names a page user cannot use.  The row
now names the running forecast, when it took the card and links to its
page; the refusal stays with launches, and its tool names stay with the
tools that have them.

Every card lock here is taken by a real launch through the job manager
(a sleeping Python process holds it); no engine and no GPU is used.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from gpuwm.gui.jobs import Refused, Runner
from gpuwm.gui.machines import Registry
from gpuwm.gui.server import build_server, serve_in_thread
from gpuwm.mcp.doors import ArwenRefusal
from gpuwm.mcp.jobs import JobManager
from test_gui_server import FakeRunner, request

SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]
MCP_WORDS = ("refused", "job_status", "job_cancel", "relaunch", "this launch")


class CardRunner(FakeRunner):
    """The real job manager and its card lock; the engine's own answers are the fake ones (no engine is asked)."""

    launch = Runner.launch
    # The fake's own card_holder says the card is free; this page reads the real card lock.
    card_holder = Runner.card_holder

    def __init__(self, jobs_root: Path) -> None:
        super().__init__()
        self.jobs_root = jobs_root


@pytest.fixture()
def page(tmp_path, monkeypatch):
    monkeypatch.setattr("gpuwm.fetch._head_ok", lambda url: True)
    monkeypatch.setattr("gpuwm.source_availability.quick_head", lambda url: True)
    jobs = tmp_path / "jobs"
    runner = CardRunner(jobs)
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43,
                          machines=Registry(tmp_path / "machines.toml"))
    # The scheduler reads this card instead of asking nvidia-smi.
    server.api.queue._cards = lambda: {"devices": [{"name": "Test card", "memory_total_mib": 16 * 1024,
                                                    "memory_used_mib": 512}], "processes": []}
    server.api.queue._disk = lambda: 200.0
    serve_in_thread(server)
    try:
        yield server, runner, jobs
    finally:
        server.shutdown()
        server.server_close()
        manager = JobManager(jobs)
        for row in manager.list()["jobs"]:
            if row["state"] == "running":
                manager.cancel(row["job_id"])


def a_forecast(root: Path, name: str, title: str | None = None) -> Path:
    rundir = root / name
    rundir.mkdir(parents=True)
    (rundir / "plan.json").write_text(json.dumps({"schema": "test"}), encoding="utf-8")
    if title:
        (rundir / "wiki-run.json").write_text(json.dumps({"title": title}), encoding="utf-8")
    return rundir


def this_computer(server) -> dict:
    response, body = request(server, "GET", "/api/machines")
    assert response.status == 200, body
    row = body["machines"][0]
    assert row["name"] == "this-computer"
    return row


def test_the_row_names_the_running_forecast_links_to_it_and_says_nothing_of_a_launch(page):
    server, runner, jobs = page
    assert this_computer(server)["detail"] == "idle"
    rundir = a_forecast(server.root, "storm-a", title="Storm A, 16 GB layout")
    job = runner.launch(rundir, SLEEP)

    row = this_computer(server)
    held = json.loads((jobs / "gpu.lock").read_text(encoding="utf-8"))
    assert held["job_id"] == job["job_id"]
    assert row["state"] == "running"
    assert row["detail"].startswith("running Storm A, 16 GB layout, started ")
    assert row["detail"].endswith(" UTC")
    for word in MCP_WORDS:
        assert word not in row["detail"].lower(), row["detail"]
    assert row["run"] == "storm-a" and row["url"] == "storm-a" and row["title"] == "Storm A, 16 GB layout"
    assert row["started_utc"] == held["created_utc"] and row["job_id"] == job["job_id"]
    # GET /api/machines/this-computer answers the same row.
    response, one = request(server, "GET", "/api/machines/this-computer")
    assert response.status == 200 and one["detail"] == row["detail"] and one["run"] == "storm-a"

    # The queue reads the same holder and names the same forecast.
    response, body = request(server, "GET", "/api/queue?need_gib=5.3")
    assert response.status == 200 and body["expect"]["busy"]
    assert server.api.queue.local_card(fresh=True)["why"] == "The card is running storm-a."

    # A second start from the page is refused in the page's words, naming the running forecast.
    second = a_forecast(server.root, "storm-b")
    with pytest.raises(Refused) as refused:
        runner.launch(second, SLEEP)
    words = str(refused.value)
    assert words.startswith("The card is running storm-a, started ") and "was not started" in words
    for word in ("job_status", "job_cancel", "relaunch"):
        assert word not in words, words

    # The MCP tools keep their own refusal, with the tool names an MCP client can use.
    with pytest.raises(ArwenRefusal, match="poll job_status"):
        JobManager(jobs).launch("forecast", SLEEP, cwd=second, gpu=True)


def test_a_holder_that_is_not_one_of_the_pages_forecasts_is_named_without_a_link(page, tmp_path):
    server, _, jobs = page
    elsewhere = tmp_path / "elsewhere" / "case-one"
    elsewhere.mkdir(parents=True)
    # An MCP client's run: its job works in the source tree and writes its declared folder.
    JobManager(jobs).launch("forecast", SLEEP, cwd=tmp_path, gpu=True, outputs={"outdir": str(elsewhere)})
    row = this_computer(server)
    assert row["state"] == "running" and row["run"] is None and row["url"] is None
    assert row["detail"].startswith("running case-one, started ")
    assert "not one of the forecasts on this page" in row["detail"]
    for word in MCP_WORDS:
        assert word not in row["detail"].lower(), row["detail"]


def test_a_card_lock_that_cannot_be_read_reads_as_in_use_in_plain_words(page):
    server, _, jobs = page
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / "gpu.lock").write_text("{not json", encoding="utf-8")
    row = this_computer(server)
    assert row["state"] == "busy" and row["run"] is None
    assert row["detail"] == "the card counts as in use: its lock file cannot be read"
    assert str(jobs / "gpu.lock") in row["fix"]
    for word in MCP_WORDS:
        assert word not in (row["detail"] + row["fix"]).lower()
