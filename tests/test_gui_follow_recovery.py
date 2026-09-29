"""A machine's run whose follower ended says so, and Resume updates follows it again without starting anything.

The defect: the follower that mirrors a machine's forecast (and its render
worker) here can end before the forecast does (it gives up after about ten
minutes of failed calls, or its process is stopped). The run folder then
kept the last state the machine sent, so the page showed a running forecast
with a Stop button after the machine had finished, with nothing saying its
updates had stopped and no way to start them again short of drawing the
run a second time.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
import time

from gpuwm import proc_identity
from gpuwm.gui import runs, server as gui_server
from gpuwm.gui.machines import FOLLOW
from gpuwm.gui.remote_runs import Follower
from test_gui_machines import FakeMachine, FakeRegistry
from test_gui_server import gui, request  # noqa: F401 - the fixture is used by name


def live_job(folder):
    """A follower's job record whose process is this test's own, so it reads as running."""

    return {"job_id": "job-follow", "jobs_dir": str(folder), "wrapper_pid": os.getpid(),
            "wrapper_process": proc_identity.identify(os.getpid())}


def ended_job(folder):
    """A follower's job record whose job wrote its result: it gave up (exit 3 after its failed calls)."""

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "result.json").write_text(json.dumps({"exit_code": 3}), encoding="utf-8")
    return live_job(folder)


def machine_run(root, name="worker-run", *, ended=False, follower=None):
    rundir = root / name
    rundir.mkdir(parents=True)
    job = {"state": "finished" if ended else "running"}
    (rundir / runs.REMOTE).write_text(json.dumps({"machine": "worker", "alive": not ended, "ended": ended,
                                                  "checked_utc": "2026-09-28T00:00:00Z", "job": job}),
                                      encoding="utf-8")
    (rundir / runs.EVENTS).write_text(json.dumps({"event": "stage_started", "sequence": 1, "stage": "forecast"})
                                      + "\n", encoding="utf-8")
    if follower is not None:
        (rundir / FOLLOW).write_text(json.dumps(follower), encoding="utf-8")
    return rundir


def test_the_follow_file_name_is_the_machines_one():
    assert runs.FOLLOW == FOLLOW


def test_a_follower_that_ended_while_the_machine_runs_is_said(tmp_path):
    rundir = machine_run(tmp_path, follower=ended_job(tmp_path / "jobs" / "follow"))
    info = runs.status(rundir)
    # The state is the last one the machine sent; the flag says it is no longer being kept up to date.
    assert info["state"] == "running" and info["follow_lost"] is True
    (rundir / FOLLOW).write_text(json.dumps(live_job(tmp_path / "jobs" / "live")), encoding="utf-8")
    assert runs.status(rundir)["follow_lost"] is False


def test_a_run_with_no_follower_at_all_is_said_too(tmp_path):
    assert runs.status(machine_run(tmp_path))["follow_lost"] is True


def test_nothing_is_lost_once_the_machine_ended_and_nothing_is_drawing(tmp_path):
    rundir = machine_run(tmp_path, ended=True, follower=ended_job(tmp_path / "jobs" / "follow"))
    assert runs.status(rundir)["follow_lost"] is False
    local = tmp_path / "local-run"
    local.mkdir()
    (local / runs.EVENTS).write_text(json.dumps({"event": "completed", "sequence": 1}) + "\n", encoding="utf-8")
    assert runs.status(local)["follow_lost"] is False


def test_a_render_worker_still_drawing_needs_its_follower(tmp_path):
    # A finished forecast sent to a machine to draw: its pictures come back only through the follower.
    rundir = tmp_path / "drawn"
    rundir.mkdir()
    (rundir / runs.EVENTS).write_text(json.dumps({"event": "completed", "sequence": 1}) + "\n", encoding="utf-8")
    (rundir / runs.RENDER).write_text(json.dumps({"machine": "worker", "state": "rendering"}), encoding="utf-8")
    assert runs.status(rundir)["follow_lost"] is True
    (rundir / runs.RENDER).write_text(json.dumps({"machine": "worker", "state": "finished"}), encoding="utf-8")
    assert runs.status(rundir)["follow_lost"] is False


def test_the_follower_keeps_going_exactly_while_the_page_needs_it(tmp_path):
    machine = FakeMachine({"name": "worker", "host": "me@worker", "workspace": "/w"})
    rundir = machine_run(tmp_path / "runs")
    machine.snapshots = [{"ok": True, "alive": False, "job": {"state": "finished"}, "events_b64": "",
                          "events_size": 0, "heartbeat": None, "manifest": None}]
    follower = Follower(tmp_path / "runs", "worker-run", FakeRegistry(tmp_path / "m.toml", machine))
    assert follower.tick() is False
    assert runs.follow_needed(json.loads((rundir / runs.REMOTE).read_text()), None) is False


def test_resume_updates_starts_one_follower_and_no_forecast(gui, monkeypatch):  # noqa: F811
    server, runner = gui
    rundir = machine_run(server.root, follower=ended_job(server.root / ".jobs" / "follow"))
    helpers = []
    lock = threading.Lock()

    def launch_helper(folder, argv, kind):
        # Slow enough that two requests that both found no follower would both get here.
        time.sleep(0.2)
        with lock:
            helpers.append((list(argv), kind))
        return live_job(server.root / ".jobs" / f"live-{len(helpers)}")

    monkeypatch.setattr(runner, "launch_helper", launch_helper, raising=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(lambda _: request(server, "POST", "/api/runs/worker-run/follow", body={}), range(2)))
    assert [response.status for response, _ in replies] == [200, 200], replies
    assert len(helpers) == 1
    argv, kind = helpers[0]
    assert argv[-1] == "worker-run" and "follow" in argv and kind == "gui:machines-follow"
    assert runner.launched == []
    assert runs.status(rundir)["follow_lost"] is False
    status, detail = request(server, "GET", "/api/runs/worker-run")
    assert status.status == 200 and detail["status"]["follow_lost"] is False


def test_resume_updates_dry_run_shows_the_follow_command(gui):  # noqa: F811
    server, runner = gui
    machine_run(server.root, follower=ended_job(server.root / ".jobs" / "follow"))
    response, body = request(server, "POST", "/api/runs/worker-run/follow", body={"dry_run": True})
    assert response.status == 200 and body["dry_run"] and "follow" in body["argv"]
    assert json.loads((server.root / "worker-run" / FOLLOW).read_text())["job_id"] == "job-follow"


def test_resume_updates_on_a_run_of_this_computer_says_there_is_nothing_to_resume(gui):  # noqa: F811
    server, runner = gui
    local = server.root / "here"
    local.mkdir()
    (local / runs.EVENTS).write_text(json.dumps({"event": "completed", "sequence": 1}) + "\n", encoding="utf-8")
    response, body = request(server, "POST", "/api/runs/here/follow", body={})
    assert response.status == 409 and "another machine" in body["message"]


def test_the_event_stream_says_when_updates_stop_and_start_again(tmp_path):
    rundir = machine_run(tmp_path, follower=ended_job(tmp_path / "jobs" / "follow"))
    gone, stopping = threading.Event(), threading.Event()
    timer = threading.Timer(10.0, stopping.set)
    timer.start()
    seen = []
    try:
        for event, _, data in gui_server.follow(rundir, None, gone, stopping):
            if event != "status":
                continue
            seen.append(data["follow_lost"])
            if len(seen) == 1:
                (rundir / FOLLOW).write_text(json.dumps(live_job(tmp_path / "jobs" / "live")), encoding="utf-8")
            if len(seen) == 2:
                break
    finally:
        timer.cancel()
    assert seen == [True, False]
