"""A job or a remote forecast whose program cannot start ends with a result, and a stop ends what it stops.

The defects this file holds shut:

- The job wrapper raised when the command it runs could not be started (its
  program gone, its folder not there) and wrote no result: the job read as
  running until its wrapper was found dead, with no reason anywhere.
- A Machines node's supervisor died with a traceback when the engine's
  Python had been removed: the job kept its "waiting" state and the page's
  follower waited on a supervisor that was gone.
- An engine that answered neither Ctrl+C nor SIGTERM after its bound held
  the card for another hour; it now gets SIGKILL after a second grace.
- A run told to stop whose engine then exited 0 read Finished.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gpuwm import machine_agent as agent
from gpuwm.gui import runs

ROOT = Path(__file__).resolve().parents[1]


def _wrap(tmp_path: Path, argv: list[str], cwd: Path) -> Path:
    jobdir = tmp_path / "job"
    jobdir.mkdir()
    (jobdir / "receipt.json").write_text(json.dumps({"argv": argv, "cwd": str(cwd), "env_additions": {}}),
                                         encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    done = subprocess.run([sys.executable, "-m", "gpuwm.mcp._jobwrap", str(jobdir)], env=env, cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return jobdir


@pytest.mark.parametrize("case", ["missing program", "missing folder"])
def test_a_job_whose_command_cannot_start_ends_with_a_result_and_a_reason(tmp_path, case):
    missing = str(tmp_path / "removed-venv" / "bin" / "python")
    argv, cwd = ([missing, "-V"], tmp_path) if case == "missing program" else ([sys.executable, "-V"],
                                                                                 tmp_path / "gone")
    jobdir = _wrap(tmp_path, argv, cwd)
    result = json.loads((jobdir / "result.json").read_text(encoding="utf-8"))
    assert result["exit_code"] == 127 and result["cancelled"] is False
    assert "could not start" in (jobdir / "stderr.log").read_text(encoding="utf-8")
    # The page reads the run as failed with that reason, in words, without the machine path.
    rundir = tmp_path / "run"
    rundir.mkdir()
    (rundir / runs.PLAN).write_text("{}", encoding="utf-8")
    (rundir / runs.JOB).write_text(json.dumps({"jobs_dir": str(jobdir)}), encoding="utf-8")
    info = runs.status(rundir)
    assert info["state"] == "failed" and info["end"]["exit_code"] == 127
    assert "could not start" in info["end"]["message"] and str(tmp_path) not in info["end"]["message"]


def _supervised(tmp_path: Path, **job) -> Path:
    rundir = tmp_path / "runs" / "forecast"
    rundir.mkdir(parents=True)
    agent.write_json(rundir / "gui-job.json", {"bound_min": 1, "wait_min": 0, **job})
    return rundir


def test_a_machine_forecast_whose_engine_is_gone_fails_and_gives_the_card_back(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "cards", lambda: {"devices": [], "processes": []})
    owner = tmp_path / "OWNER"
    missing = str(tmp_path / "removed-venv" / "bin" / "python")
    rundir = _supervised(tmp_path, argv=[missing, "-m", "gpuwm", "run-plan", "plan.json"], python=missing,
                         owner_file=str(owner))
    assert agent.cmd_supervise(SimpleNamespace(workspace=str(tmp_path), run="forecast")) == 1
    job = agent.read_json(rundir / "gui-job.json")
    assert job["state"] == "failed" and job["ended_utc"] and job["exit_code"] == 127
    assert "could not start" in job["message"] and missing not in job["message"]
    assert not owner.exists() or str(os.getpid()) not in owner.read_text(encoding="utf-8")
    assert "failed" in runs.REMOTE_END_STATES


def test_a_machine_forecast_whose_engine_is_gone_fails_without_waiting_for_a_busy_card(tmp_path, monkeypatch):
    """A card held by another program must not keep a job that cannot start waiting for its turn."""

    monkeypatch.setattr(agent, "cards", lambda: {"devices": [], "processes": [{"pid": 1, "used_mib": 900}]})
    owner = tmp_path / "OWNER"
    held = f"other-program 2026-09-27T00:00:00Z pid {os.getpid()} bounded 240 min"
    owner.write_text(held + "\n", encoding="utf-8")
    monkeypatch.setattr(agent, "claim_card", lambda *a, **k: pytest.fail("a job that cannot start waits for no card"))
    missing = str(tmp_path / "removed-venv" / "bin" / "python")
    rundir = _supervised(tmp_path, argv=[missing, "-m", "gpuwm", "run-plan", "plan.json"], python=missing,
                         owner_file=str(owner), wait_min=30)
    assert agent.cmd_supervise(SimpleNamespace(workspace=str(tmp_path), run="forecast")) == 1
    job = agent.read_json(rundir / "gui-job.json")
    assert job["state"] == "failed" and job["exit_code"] == 127 and "could not start" in job["message"]
    assert owner.read_text(encoding="utf-8") == held + "\n"


class _Engine:
    """An engine that ends only on the signal named, standing in for a stubborn one."""

    pid = 424242

    def __init__(self, ends_on, code):
        self.ends_on, self.code, self.returncode = ends_on, code, None
        self.sent: list[int] = []

    def poll(self):
        if self.ends_on in self.sent:
            self.returncode = self.code
        return self.returncode


def _drive(monkeypatch, engine, *, stop_request_at=None, rundir=None):
    clock = [0.0]
    monkeypatch.setattr(agent, "cards", lambda: {"devices": [], "processes": []})
    monkeypatch.setattr(agent.subprocess, "Popen", lambda *a, **k: engine)
    monkeypatch.setattr(agent, "process_start", lambda pid: "fixture")
    monkeypatch.setattr(agent.os, "killpg", lambda pid, sig: engine.sent.append(sig), raising=False)
    monkeypatch.setattr(agent.time, "monotonic", lambda: clock[0])

    def sleep(seconds):
        clock[0] += 61.0
        if stop_request_at is not None and clock[0] >= stop_request_at:
            (rundir / "gui-stop-request").touch()
        assert clock[0] < 5 * 61.0, f"the engine is still running at {clock[0]} s with signals {engine.sent}"

    monkeypatch.setattr(agent.time, "sleep", sleep)


def test_an_engine_that_answers_no_signal_is_killed_after_two_graces(tmp_path, monkeypatch):
    import signal

    kill = getattr(signal, "SIGKILL", 9)
    rundir = _supervised(tmp_path, argv=[sys.executable])
    engine = _Engine(kill, -kill)
    _drive(monkeypatch, engine)
    agent.cmd_supervise(SimpleNamespace(workspace=str(tmp_path), run="forecast"))
    assert engine.sent == [signal.SIGINT, signal.SIGTERM, kill]
    job = agent.read_json(rundir / "gui-job.json")
    assert job["state"] == "stopped" and job["stop_reason"] == "bound"


def test_a_stopped_run_whose_engine_then_exits_0_reads_stopped(tmp_path, monkeypatch):
    import signal

    rundir = _supervised(tmp_path, argv=[sys.executable], bound_min=60)
    engine = _Engine(signal.SIGINT, 0)
    _drive(monkeypatch, engine, stop_request_at=61.0, rundir=rundir)
    agent.cmd_supervise(SimpleNamespace(workspace=str(tmp_path), run="forecast"))
    job = agent.read_json(rundir / "gui-job.json")
    assert job["state"] == "stopped" and job["stop_reason"] == "stop" and job["exit_code"] == 0


@pytest.mark.skipif(os.name == "nt", reason="the supervisor runs on Linux machines")
def test_a_real_stubborn_engine_is_ended_and_its_card_line_goes(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "cards", lambda: {"devices": [], "processes": []})
    monkeypatch.setattr(agent, "STOP_ESCALATE_S", 0.5)
    owner = tmp_path / "OWNER"
    stubborn = ("import signal, time\n"
                "signal.signal(signal.SIGINT, signal.SIG_IGN)\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "time.sleep(120)\n")
    rundir = _supervised(tmp_path, argv=[sys.executable, "-c", stubborn], owner_file=str(owner), bound_min=60)
    import threading

    def stop_soon():
        import time as clock
        for _ in range(100):
            if (agent.read_json(rundir / "gui-job.json", default={}) or {}).get("state") == "running":
                break
            clock.sleep(0.05)
        (rundir / "gui-stop-request").touch()

    threading.Thread(target=stop_soon, daemon=True).start()
    agent.cmd_supervise(SimpleNamespace(workspace=str(tmp_path), run="forecast"))
    job = agent.read_json(rundir / "gui-job.json")
    assert job["state"] == "stopped" and job["exit_code"] == -9
    assert not agent.pid_alive(job["engine_pid"])
    assert f"pid {os.getpid()} " not in owner.read_text(encoding="utf-8")
