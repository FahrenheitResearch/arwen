"""A Machines node writes a forecast's plan and its companion files as sent, and one launch of a name at a time.

The defects this file holds shut:

- A storm-following layout's configuration, Vtable and WPS namelist could
  not reach a node: the launch took JSON documents only, and a text file
  passed through it would have been written as one quoted JSON string.
- A sent file name was checked only by its pattern: a Windows device name
  or one of the files the supervisor keeps (its job record, its logs)
  could be written over.
- Two launches of one name at once both wrote the folder and both started.
- The claim must not refuse what a queued forecast needs: a launch that did
  not start (the card was busy, its supervisor could not start) leaves a
  folder the next launch of that name takes.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from gpuwm import machine_agent as agent


def _args(tmp_path: Path, run: str = "following") -> SimpleNamespace:
    return SimpleNamespace(workspace=str(tmp_path), run=run, python=str(tmp_path / "missing-engine"),
                           owner_file=None, owner_tag="gpuwm-gui")


def _request(text_files=None, **extra):
    return {"files": {"plan.json": {"schema": "test"}, "region.geojson": {"type": "Polygon"}},
            "text_files": text_files or {}, "wait_min": 1, **extra}


def _launch(tmp_path, monkeypatch, request, run="following"):
    monkeypatch.setattr(agent.sys, "stdin", io.StringIO(json.dumps(request)))
    return agent.cmd_launch(_args(tmp_path, run))


@pytest.mark.parametrize("text_files", [
    {"../outside.toml": "bad"}, {"x/y.toml": "bad"}, {"C:\\outside.toml": "bad"},
    {"plan.json": "overwrite"}, {"gui-job.json": "receipt"}, {"engine.log": "log"}, {"Supervisor.log": "log"},
    {"new.toml": {"not": "text"}}, {"new.toml": "bad\x00text"}, {"new.toml": "\ud800"},
    {"trailing.toml.": "alias"}, {"NUL.toml": "device"}, {"com1": "device"},
])
def test_a_file_that_cannot_be_written_as_sent_is_refused_before_anything_is(tmp_path, monkeypatch, text_files):
    monkeypatch.setattr(agent, "spawn", lambda *a, **k: pytest.fail("nothing starts"))
    answer = _launch(tmp_path, monkeypatch, _request(text_files))
    assert answer["ok"] is False and answer["message"]
    assert not (tmp_path / "runs" / "following").exists()
    assert not (tmp_path / "outside.toml").exists()


def test_companions_are_written_as_their_own_text(tmp_path, monkeypatch):
    text = '[experiment]\nname = "René forecast"\n'
    monkeypatch.setattr(agent, "spawn", lambda *a, **k: os.getpid())
    answer = _launch(tmp_path, monkeypatch, _request({"cyclone.toml": text, "cyclone.Vtable": "table\n",
                                                       "cyclone.namelist.wps": "&share\n/\n"}))
    assert answer["ok"], answer
    run = tmp_path / "runs" / "following"
    assert (run / "cyclone.toml").read_bytes() == text.encode("utf-8")
    assert (run / "cyclone.namelist.wps").read_text(encoding="utf-8") == "&share\n/\n"
    assert json.loads((run / "plan.json").read_text(encoding="utf-8")) == {"schema": "test"}
    assert not list(run.glob("*.tmp"))


def test_a_run_going_or_finished_there_is_never_written_over(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "following"
    run.mkdir(parents=True)
    (run / "cyclone.toml").write_text("original", encoding="utf-8")
    for job in ({"state": "finished"}, {"state": "running"},
                {"state": "waiting-for-card", "supervisor_pid": os.getpid(),
                 "supervisor_start": agent.process_start(os.getpid())}):
        agent.write_json(run / "gui-job.json", job)
        answer = _launch(tmp_path, monkeypatch, _request({"cyclone.toml": "new"}))
        assert answer["ok"] is False and "already exists" in answer["message"]
        assert (run / "cyclone.toml").read_text(encoding="utf-8") == "original"


def test_a_launch_that_did_not_start_leaves_a_folder_the_next_launch_takes(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "following"
    run.mkdir(parents=True)
    (run / "cyclone.toml").write_text("the last attempt's", encoding="utf-8")
    agent.write_json(run / "gui-job.json", {"state": "failed", "supervisor_pid": 999999999,
                                            "supervisor_start": "long gone"})
    monkeypatch.setattr(agent, "spawn", lambda *a, **k: os.getpid())
    answer = _launch(tmp_path, monkeypatch, _request({"cyclone.toml": "this attempt's"}))
    assert answer["ok"], answer
    assert (run / "cyclone.toml").read_text(encoding="utf-8") == "this attempt's"


@pytest.mark.skipif(os.name == "nt", reason="links need a privilege on Windows")
def test_a_run_name_that_is_a_link_elsewhere_is_not_followed(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "following").symlink_to(outside, target_is_directory=True)
    answer = _launch(tmp_path, monkeypatch, _request({"cyclone.toml": "overwrite"}))
    assert answer["ok"] is False
    assert list(outside.iterdir()) == []


def test_two_launches_of_one_name_at_once_start_one(tmp_path):
    """The standalone agent, as a Machines node runs it: two launches in two processes at the same moment."""

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "machine_agent.py").write_bytes(Path(agent.__file__).read_bytes())
    workspace = tmp_path / "workspace with spaces"
    command = [sys.executable, "-S", str(bundle / "machine_agent.py"), "launch", "--workspace", str(workspace),
               "--python", str(tmp_path / "missing-engine"), "--run", "following"]
    env = {**os.environ, "PYTHONPATH": "", "GPUWM_NO_LOCAL_GPU": "1", "CUDA_VISIBLE_DEVICES": "-1"}
    gate = threading.Barrier(2)
    answers = []

    def launch(label):
        payload = _request({"cyclone.toml": f'name = "{label}"\n'})
        gate.wait(timeout=10)
        done = subprocess.run(command, input=json.dumps(payload), text=True, capture_output=True, cwd=bundle,
                              env=env, timeout=60)
        assert done.returncode == 0, done.stderr
        answers.append((label, json.loads(done.stdout)))

    threads = [threading.Thread(target=launch, args=(label,)) for label in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=90)
    accepted = [label for label, answer in answers if answer["ok"]]
    assert len(answers) == 2 and len(accepted) == 1, answers
    run = workspace / "runs" / "following"
    assert (run / "cyclone.toml").read_text(encoding="utf-8") == f'name = "{accepted[0]}"\n'
    # Its engine is missing, so its supervisor ends the job failed rather than leaving it waiting.
    import time

    for _ in range(200):
        job = agent.read_json(run / "gui-job.json", default={}) or {}
        if job.get("state") == "failed":
            break
        time.sleep(0.05)
    assert job.get("state") == "failed" and "could not start" in job.get("message", ""), job
