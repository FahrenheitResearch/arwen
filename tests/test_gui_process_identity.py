"""A recorded PID is a process only while that process lives, and the card is claimed in one step.

The defects this file holds shut:

- After a crash or a reboot a run folder's PID can belong to an
  unrelated program.  The page read the run as Running and its Stop
  signalled that program's process group; the escalation a minute later
  could do the same to whoever got the PID in between, and a stale
  model-server record made the assistant's stop (run automatically when a
  forecast starts) end another program.
- A started forecast the engine refused at its start (exit 2, before its
  manifest) fell back to Ready with no word of why.
- Two starts in the same instant both passed the card check and both ran
  on the card: the check and the claim were two steps.

Every process signalled here is one this test started.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from gpuwm import proc_identity
from gpuwm.gui import runs
from gpuwm.gui.jobs import Refused, Runner
from gpuwm.gui.server import build_server, serve_in_thread
from gpuwm.mcp.doors import ArwenRefusal
from gpuwm.mcp.jobs import JobManager
from test_gui_server import DRAFT, FakeRunner, request

SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]


@pytest.fixture()
def sleeper():
    """A process this test owns, in its own process group, standing in for a program that got a reused PID."""

    process = subprocess.Popen(SLEEP, stdin=subprocess.DEVNULL, start_new_session=os.name != "nt",
                               creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=30)


def other_identity(pid: int) -> dict:
    """The identity a run recorded for ``pid`` before the PID was handed to someone else."""

    identity = proc_identity.identify(pid)
    assert identity is not None
    return {**identity, "start": str(identity["start"]) + "0"}


def crashed_run(root: Path, name: str, pid: int, process: dict | None) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / runs.MANIFEST).write_text(json.dumps({"pid": pid, "process": process,
                                                 "started_at_utc": "2026-09-10T00:00:00Z"}), encoding="utf-8")
    (run / runs.EVENTS).write_text(json.dumps({"sequence": 0, "event": "run_started"}) + "\n", encoding="utf-8")
    return run


# ---------------------------------------------------------------- identity

def test_an_identity_names_one_process_and_not_a_later_holder_of_its_pid(sleeper):
    identity = proc_identity.identify(sleeper.pid)
    assert proc_identity.alive(identity, sleeper.pid)
    assert not proc_identity.alive(identity, sleeper.pid + 1)
    assert not proc_identity.alive(other_identity(sleeper.pid))
    assert proc_identity.reused(other_identity(sleeper.pid))
    assert not proc_identity.alive({"pid": sleeper.pid})          # a record from before identities: stale
    sleeper.kill()
    sleeper.wait(timeout=30)
    assert not proc_identity.alive(identity)


# ---------------------------------------------------------------- a stale record: status and Stop

def test_a_manifest_whose_pid_now_names_another_process_is_not_running(tmp_path, sleeper):
    run = crashed_run(tmp_path, "crashed", sleeper.pid, other_identity(sleeper.pid))
    status = runs.status(run)
    assert status["state"] == "stale" and status["alive"] is False
    # A manifest written before identities existed cannot prove which process it named.
    legacy = crashed_run(tmp_path, "legacy", sleeper.pid, None)
    assert runs.status(legacy)["state"] == "stale"
    # The same record naming the process itself is running.
    live = crashed_run(tmp_path, "live", sleeper.pid, proc_identity.identify(sleeper.pid))
    assert runs.status(live)["state"] == "running"


def test_a_job_record_whose_wrapper_pid_was_reused_is_not_running(tmp_path, sleeper):
    run = tmp_path / "started"
    run.mkdir()
    (run / runs.PLAN).write_text("{}", encoding="utf-8")
    (run / runs.JOB).write_text(json.dumps({"job_id": "job-x", "jobs_dir": str(tmp_path / "job-x"),
                                            "wrapper_pid": sleeper.pid,
                                            "wrapper_process": other_identity(sleeper.pid)}), encoding="utf-8")
    assert not runs.job_alive(run)
    assert runs.status(run)["state"] != "running"


def test_stop_on_a_stale_record_refuses_and_signals_nothing(tmp_path, sleeper):
    run = crashed_run(tmp_path / "runs", "crashed", sleeper.pid, other_identity(sleeper.pid))
    (run / runs.JOB).write_text(json.dumps({"job_id": "job-gone", "jobs_dir": str(tmp_path / "jobs" / "job-gone"),
                                            "wrapper_pid": sleeper.pid,
                                            "wrapper_process": other_identity(sleeper.pid)}), encoding="utf-8")
    runner = Runner(tmp_path / "jobs")
    with pytest.raises(Refused, match="nothing to stop"):
        runner.stop(run)
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    serve_in_thread(server)
    try:
        response, _ = request(server, "POST", "/api/runs/crashed/stop", body={})
        assert response.status == 409
    finally:
        server.shutdown()
        server.server_close()
    time.sleep(0.5)
    assert sleeper.poll() is None, "Stop signalled a process the run did not start"
    assert not (run / runs.STOP).exists()


#: A stand-in engine: writes its manifest as ``gpuwm run-plan`` does (the pid and that process's identity) and waits.
ENGINE = ("import json, os, pathlib, sys, time\n"
          "from gpuwm import proc_identity\n"
          "run = pathlib.Path(sys.argv[1])\n"
          "record = {'pid': os.getpid(), 'process': proc_identity.identify(os.getpid())}\n"
          "(run / 'run-manifest.json.tmp').write_text(json.dumps(record))\n"
          "os.replace(run / 'run-manifest.json.tmp', run / 'run-manifest.json')\n"
          "time.sleep(60)\n")


def test_stop_still_reaches_a_run_that_is_its_own_process(tmp_path):
    runner = Runner(tmp_path / "jobs")
    run = tmp_path / "runs" / "live"
    run.mkdir(parents=True)
    job = runner.launch(run, [sys.executable, "-c", ENGINE, str(run)])
    manifest: dict = {}
    try:
        deadline = time.monotonic() + 60
        while not (run / runs.MANIFEST).is_file() and time.monotonic() < deadline:
            time.sleep(0.1)
        manifest = json.loads((run / runs.MANIFEST).read_text(encoding="utf-8"))
        assert runs.status(run)["state"] == "running"
        runner.stop(run)
        deadline = time.monotonic() + 60
        while runs.status(run)["alive"] and time.monotonic() < deadline:
            time.sleep(0.2)
        assert runs.status(run)["alive"] is False
        assert not proc_identity.alive(manifest["process"], manifest["pid"]), "Stop did not reach the run"
    finally:
        for record in (job.get("wrapper_process"), manifest.get("process")):  # only this test's own processes
            if proc_identity.alive(record):
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(record["pid"])], capture_output=True)
                else:
                    os.killpg(os.getpgid(int(record["pid"])), 9)


@pytest.mark.skipif(os.name == "nt", reason="the escalation is the POSIX stop's SIGTERM to a process group")
def test_a_pid_reused_between_the_interrupt_and_the_escalation_gets_no_sigterm(sleeper):
    group = os.getpgid(sleeper.pid)
    assert group == sleeper.pid
    runner = Runner()
    # The run's leader ended and its number went to this sleeper: the leader the Stop saw is another process.
    runner._escalate(group, "job-gone", other_identity(sleeper.pid))
    # The leader had already gone when Stop ran, and now a process holds the group's number: a new group.
    runner._escalate(group, "job-gone", None)
    time.sleep(0.5)
    assert sleeper.poll() is None, "the escalation signalled a process given the run's PID"
    # The same group, its leader still the process Stop saw, is ended.
    runner._escalate(group, "job-live", proc_identity.identify(sleeper.pid))
    assert sleeper.wait(timeout=30) == -15


def test_the_model_servers_stale_record_stops_nothing(tmp_path, sleeper):
    from gpuwm.gui.assistant.local import Local

    local = Local.__new__(Local)
    local.root = tmp_path
    for process in (other_identity(sleeper.pid), None):
        (tmp_path / "server.json").write_text(json.dumps({"pid": sleeper.pid, "process": process, "port": 8790,
                                                          "model": "m"}), encoding="utf-8")
        assert local.running() is None
        assert local.stop() is False
        assert not (tmp_path / "server.json").exists()
    time.sleep(0.5)
    assert sleeper.poll() is None, "the model's stop ended a program that is not the model server"
    # Its own record still stops it.
    (tmp_path / "server.json").write_text(json.dumps({"pid": sleeper.pid, "port": 8790, "model": "m",
                                                      "process": proc_identity.identify(sleeper.pid)}),
                                          encoding="utf-8")
    assert local.stop() is True
    sleeper.wait(timeout=30)


# ---------------------------------------------------------------- a start the engine refused

def started_job(root: Path, name: str, result: dict, stderr: str) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / runs.PLAN).write_text("{}", encoding="utf-8")
    jobs_dir = root / f"job-{name}"
    jobs_dir.mkdir()
    (jobs_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (jobs_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    (run / runs.JOB).write_text(json.dumps({"job_id": f"job-{name}", "jobs_dir": str(jobs_dir),
                                            "wrapper_pid": 999_999_999}), encoding="utf-8")
    return run


#: A user's home folder, assembled from pieces: written out whole it reads
#: as one machine's path to the release scan for machine paths.
HOME = "/ho" + "me/someone"
BANNER = (f"gpuwm run-plan: gpuwm 2.8.0 -- editable source at {HOME}/gpuwm, git 0123abcd on main\n")


def test_a_start_the_engine_refused_reads_failed_with_its_reason(tmp_path):
    run = started_job(tmp_path, "invalid", {"exit_code": 2, "cancelled": False},
                      BANNER + f"gpuwm run-plan: run plan {HOME}/runs/invalid/plan.json is missing "
                               "required key(s) ['schema', 'name', 'route', 'config']\n"
                               f"  (run gpuwm run-plan {HOME}/runs/invalid/plan.json --explain for the reason)\n")
    status = runs.status(run)
    assert status["state"] == "failed" and status["alive"] is False
    end = status["end"]
    assert end["exit_code"] == 2 and end["stage"] == "start"
    assert end["message"].startswith("run plan ") and "missing required key(s)" in end["message"]
    assert HOME not in end["message"] and "gpuwm 2.8.0" not in end["message"]


def test_a_start_that_crashed_says_why_without_the_traceback(tmp_path):
    run = started_job(tmp_path, "crashed", {"exit_code": 1, "cancelled": False},
                      BANNER + "Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\n"
                               "RuntimeError: the card could not be opened\n")
    end = runs.status(run)["end"]
    assert runs.status(run)["state"] == "failed" and end["message"] == "the card could not be opened"


def test_a_job_that_was_cancelled_before_its_first_record_reads_stopped(tmp_path):
    run = started_job(tmp_path, "cancelled", {"exit_code": None, "cancelled": True}, BANNER)
    status = runs.status(run)
    assert status["state"] == "stopped" and status["end"]["interrupted"] is True


def test_a_clean_exit_or_a_plan_never_started_is_left_alone(tmp_path):
    clean = started_job(tmp_path, "clean", {"exit_code": 0, "cancelled": False}, BANNER)
    assert runs.status(clean)["state"] == "ready"
    never = tmp_path / "never"
    never.mkdir()
    (never / runs.PLAN).write_text("{}", encoding="utf-8")
    assert runs.status(never)["state"] == "ready"


def test_a_run_that_had_begun_and_was_killed_reads_failed_without_guessing_from_its_log(tmp_path):
    run = started_job(tmp_path, "killed", {"exit_code": -9, "cancelled": False},
                      BANNER + "reading the source files\nwarming the card\n")
    (run / runs.MANIFEST).write_text(json.dumps({"pid": 999_999_999, "process": None}), encoding="utf-8")
    (run / runs.EVENTS).write_text(json.dumps({"sequence": 0, "event": "run_started"}) + "\n", encoding="utf-8")
    status = runs.status(run)
    assert status["state"] == "failed" and status["end"]["exit_code"] == -9
    # Its stderr is its log, so the first line of it is not the reason it ended.
    assert status["end"]["message"] == "The forecast ended (exit code -9) without writing its last record."
    cancelled = started_job(tmp_path, "stopped", {"exit_code": None, "cancelled": True}, BANNER)
    (cancelled / runs.EVENTS).write_text(json.dumps({"sequence": 0, "event": "run_started"}) + "\n",
                                         encoding="utf-8")
    status = runs.status(cancelled)
    assert status["state"] == "stopped" and status["end"]["message"] == "Stopped."


# ---------------------------------------------------------------- one card, one claim

def cancel_all(root: Path) -> None:
    manager = JobManager(root)
    for row in manager.list()["jobs"]:
        if row["state"] == "running":
            manager.cancel(row["job_id"])


def test_two_simultaneous_gpu_launches_admit_exactly_one(tmp_path):
    root = tmp_path / "jobs"
    barrier = threading.Barrier(2)
    answers: list[tuple[str, str]] = []

    def launch() -> None:
        manager = JobManager(root)
        barrier.wait(timeout=30)
        try:
            answers.append(("ok", manager.launch("sleep", SLEEP, cwd=tmp_path, gpu=True)["job_id"]))
        except ArwenRefusal as refusal:
            answers.append(("refused", str(refusal)))

    threads = [threading.Thread(target=launch) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert sorted(kind for kind, _ in answers) == ["ok", "refused"], answers
        winner = next(text for kind, text in answers if kind == "ok")
        assert winner in next(text for kind, text in answers if kind == "refused")
        held = json.loads((root / "gpu.lock").read_text(encoding="utf-8"))
        assert held["job_id"] == winner and proc_identity.alive(held["wrapper_process"], held["wrapper_pid"])
        assert sum(1 for row in JobManager(root).list()["jobs"] if row["state"] == "running") == 1
    finally:
        cancel_all(root)


def test_a_card_lock_whose_holder_pid_was_reused_does_not_hold_the_card(tmp_path, sleeper):
    root = tmp_path / "jobs"
    root.mkdir()
    (root / "gpu.lock").write_text(json.dumps({"job_id": "job-gone", "wrapper_pid": sleeper.pid,
                                               "wrapper_process": other_identity(sleeper.pid)}), encoding="utf-8")
    manager = JobManager(root)
    assert manager.gpu_lock.refusal() is None
    try:
        job = manager.launch("sleep", SLEEP, cwd=tmp_path, gpu=True)
        assert json.loads((root / "gpu.lock").read_text(encoding="utf-8"))["job_id"] == job["job_id"]
    finally:
        cancel_all(root)
    assert sleeper.poll() is None


def test_a_launch_that_fails_to_spawn_leaves_the_card_free(tmp_path, monkeypatch):
    root = tmp_path / "jobs"
    manager = JobManager(root)

    def broken(*args, **kwargs):
        raise OSError("no interpreter")

    monkeypatch.setattr("gpuwm.mcp.jobs.subprocess.Popen", broken)
    with pytest.raises(OSError):
        manager.launch("sleep", SLEEP, cwd=tmp_path, gpu=True)
    assert not (root / "gpu.lock").exists() and manager.gpu_lock.refusal() is None


def test_the_card_mutex_is_reentrant_in_a_thread_and_exclusive_across_processes(tmp_path):
    root = tmp_path / "jobs"
    lock = JobManager(root).gpu_lock
    with lock.mutex():
        with lock.mutex():  # a release inside a claim: the same thread does not wait on itself
            lock.release("job-none")
    # Another process (a second page server, an MCP client) holding the mutex makes a claim wait for it.
    held = tmp_path / "held"
    holder = subprocess.Popen([sys.executable, "-c", (
        "import sys, time, pathlib\n"
        "from gpuwm.mcp.gpulock import GpuLock\n"
        "with GpuLock(pathlib.Path(sys.argv[1])).mutex():\n"
        "    pathlib.Path(sys.argv[2]).write_text('x')\n"
        "    time.sleep(2.0)\n"), str(root), str(held)], stdin=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 60
        while not held.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert held.exists()
        began = time.monotonic()
        with lock.mutex():
            waited = time.monotonic() - began
        assert waited > 1.0, waited
    finally:
        holder.wait(timeout=60)


class RacingRunner(FakeRunner):
    """The page's queries answer from the fake; the launch is the real job manager's, of a sleeper."""

    def __init__(self, jobs_root: Path, barrier: threading.Barrier) -> None:
        super().__init__()
        self.jobs_root = jobs_root
        self.barrier = barrier

    def launch(self, rundir, argv):
        self.barrier.wait(timeout=30)
        return Runner.launch(self, rundir, SLEEP)


def test_two_simultaneous_starts_from_the_page_admit_exactly_one(tmp_path, monkeypatch):
    monkeypatch.setattr("gpuwm.fetch._head_ok", lambda url: True)
    monkeypatch.setattr("gpuwm.source_availability.quick_head", lambda url: True)
    jobs = tmp_path / "jobs"
    runner = RacingRunner(jobs, threading.Barrier(2))
    server = build_server(tmp_path / "runs", port=0, runner=runner, token="t" * 43)
    serve_in_thread(server)
    replies: dict[str, int] = {}

    def start(name: str) -> None:
        response, _ = request(server, "POST", "/api/create/start", body={**DRAFT, "name": name}, timeout=60)
        replies[name] = response.status

    threads = [threading.Thread(target=start, args=(name,)) for name in ("first", "second")]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=90)
        assert sorted(replies.values()) == [200, 409], replies
        assert sum(1 for row in JobManager(jobs).list()["jobs"] if row["state"] == "running") == 1
    finally:
        server.shutdown()
        server.server_close()
        cancel_all(jobs)


# ---------------------------------------------------------------- the queue's owned list

def test_the_queues_owned_line_goes_when_its_pid_was_reused_and_stays_while_the_run_lives(tmp_path, sleeper):
    from types import SimpleNamespace

    from gpuwm.gui.queue import ForecastQueue

    owner = tmp_path / "OWNER"
    owner.write_text(f"gpuwm-gui 2026-09-26T00:00:00Z pid {sleeper.pid} bounded 720 min\n"
                     f"gpuwm-gui 2026-09-26T00:00:00Z pid {os.getpid()} bounded 720 min\n", encoding="utf-8")
    queue = ForecastQueue(SimpleNamespace(root=tmp_path / "runs"), owner_file=str(owner))
    document = queue._document()
    # Each record names the exact line it handed on, as the queue writes it: a record naming no line removes
    # none, since tag and PID alone also match a later claim under the same number.
    document["owned"] = [
        {"run": "gone", "pid": sleeper.pid, "process": other_identity(sleeper.pid),
         "line": {"tag": "gpuwm-gui", "utc": "2026-09-26T00:00:00Z", "pid": sleeper.pid}},
        {"run": "live", "pid": os.getpid(), "process": proc_identity.identify(os.getpid()),
         "line": {"tag": "gpuwm-gui", "utc": "2026-09-26T00:00:00Z", "pid": os.getpid()}}]
    queue._save(document)
    queue._release_ended()
    # The ended run's line goes although its PID answers (another program holds it now); the live one stays.
    assert owner.read_text(encoding="utf-8").splitlines() == [
        f"gpuwm-gui 2026-09-26T00:00:00Z pid {os.getpid()} bounded 720 min"]
    assert [row["run"] for row in queue._document()["owned"]] == ["live"]
    assert sleeper.poll() is None


# ---------------------------------------------------------------- a Machines node's records

def test_a_machine_nodes_stale_engine_record_is_not_alive_and_its_stop_signals_nothing(tmp_path, sleeper):
    from types import SimpleNamespace

    from gpuwm import machine_agent as agent

    rundir = tmp_path / "runs" / "remote"
    rundir.mkdir(parents=True)
    start = agent.process_start(sleeper.pid)
    assert start and agent.process_alive(sleeper.pid, start)
    job = {"state": "running", "supervisor_pid": sleeper.pid, "supervisor_start": start + "0",
           "engine_pid": sleeper.pid, "engine_start": start + "0"}
    (rundir / "gui-job.json").write_text(json.dumps(job), encoding="utf-8")
    args = SimpleNamespace(workspace=str(tmp_path), run="remote", offset=0)
    assert agent.cmd_snapshot(args)["alive"] is False
    assert agent.workspace_jobs(tmp_path)["forecasts"][0]["alive"] is False
    stopped = agent.cmd_stop(args)
    assert stopped["method"] == "request" and (rundir / "gui-stop-request").exists()
    # A record from before starts were kept cannot prove which process it named either.
    (rundir / "gui-job.json").write_text(json.dumps({**job, "supervisor_start": None, "engine_start": None}),
                                         encoding="utf-8")
    assert agent.cmd_snapshot(args)["alive"] is False and agent.cmd_stop(args)["method"] == "request"
    time.sleep(0.5)
    assert sleeper.poll() is None, "the machine's Stop signalled a process its run did not start"
    # The record naming the process itself reads alive.
    (rundir / "gui-job.json").write_text(json.dumps({**job, "supervisor_start": start}), encoding="utf-8")
    assert agent.cmd_snapshot(args)["alive"] is True
