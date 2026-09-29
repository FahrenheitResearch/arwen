"""A job cancel that did not stop the job keeps its card claim and writes no cancelled result.

The defect: ``JobManager.cancel`` ended the job with a bare ``taskkill`` on
the recorded PID (Windows) and ignored its answer, then wrote a cancelled
result and freed the card claim.  The GUI's Stop on Windows goes through it,
so a kill the system refused left the forecast running on the card while
the page said it had stopped and the next GPU job could claim the card
beside it.  Cancel now signals the recorded wrapper through
``proc_identity.signal_process`` and only a signal that went through
publishes the result and frees the card.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from gpuwm import proc_identity
from gpuwm.mcp import jobs
from gpuwm.mcp.doors import ArwenRefusal


def _job(root, job_id, pid, identity):
    manager = jobs.JobManager(root)
    directory = root / job_id
    directory.mkdir(parents=True)
    (directory / "receipt.json").write_text(json.dumps(
        {"wrapper_pid": pid, "wrapper_process": identity, "gpu": True}), encoding="utf-8")
    manager.gpu_lock.acquire(job_id, pid, identity)
    return manager, directory


@pytest.fixture
def no_real_signals(monkeypatch):
    """Nothing in these cases may reach a real process by number: every bare route out is recorded instead."""

    bare: list = []

    def refuse(*args, **kwargs):
        bare.append(args)
        return subprocess.CompletedProcess(args[0] if args else [], 1, b"", b"termination denied")

    def record(*args):
        bare.append(args)
        raise ProcessLookupError(args)

    monkeypatch.setattr(subprocess, "run", refuse)
    for name in ("kill", "killpg"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, record)
    if hasattr(os, "getpgid"):
        monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    return bare


@pytest.mark.parametrize("outcome", ["refused", "ended"])
def test_a_stop_that_did_not_go_through_keeps_the_claim_and_writes_no_result(tmp_path, monkeypatch,
                                                                               no_real_signals, outcome):
    job_id = "job-20260927-000000-test"
    identity = {"pid": 424242, "start": "original"}
    manager, directory = _job(tmp_path, job_id, 424242, identity)
    # The wrapper answered as itself when the status was read; the signal then fails (the system refuses to end
    # it) or finds it gone (it ended between the two).
    monkeypatch.setattr(proc_identity, "alive", lambda *a, **kw: True)
    sent = []

    def signal_process(record, sig, *, tree=False):
        sent.append((record, int(sig), tree))
        if outcome == "refused":
            raise OSError("termination denied")
        return False

    monkeypatch.setattr(proc_identity, "signal_process", signal_process)
    with pytest.raises(ArwenRefusal) as refused:
        manager.cancel(job_id)
    assert sent == [(identity, 15, True)]
    assert no_real_signals == [], "cancel reached a process by bare PID"
    assert not (directory / "result.json").exists()
    assert manager.gpu_lock.holder()["job_id"] == job_id
    if outcome == "refused":
        assert "termination denied" in str(refused.value)
        assert "card claim" in str(refused.value)


def test_a_stop_that_went_through_writes_cancelled_and_frees_the_card(tmp_path, monkeypatch, no_real_signals):
    job_id = "job-20260927-000001-test"
    identity = {"pid": 424243, "start": "original"}
    manager, directory = _job(tmp_path, job_id, 424243, identity)
    monkeypatch.setattr(proc_identity, "alive", lambda *a, **kw: True)
    monkeypatch.setattr(proc_identity, "signal_process", lambda record, sig, *, tree=False: True)
    assert manager.cancel(job_id)["state"] == "cancelled"
    assert json.loads((directory / "result.json").read_text(encoding="utf-8"))["cancelled"] is True
    assert manager.gpu_lock.holder() is None


def _sleeper() -> subprocess.Popen:
    options: dict = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        options["start_new_session"] = True  # the wrapper leads its own group, as JobManager.launch starts it
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], **options)


def test_cancel_ends_a_real_wrapper_and_frees_the_card(tmp_path):
    process = _sleeper()
    try:
        identity = proc_identity.identify(process.pid)
        manager, directory = _job(tmp_path, "job-20260927-000002-test", process.pid, identity)
        assert manager.cancel("job-20260927-000002-test")["state"] == "cancelled"
        process.wait(timeout=20)
        assert manager.gpu_lock.holder() is None
        assert (directory / "result.json").is_file()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=20)


@pytest.mark.skipif(os.name != "nt", reason="taskkill is the Windows route")
def test_a_refused_taskkill_leaves_the_wrapper_running_and_the_card_claimed(tmp_path, monkeypatch):
    process = _sleeper()
    try:
        identity = proc_identity.identify(process.pid)
        manager, directory = _job(tmp_path, "job-20260927-000003-test", process.pid, identity)
        monkeypatch.setattr(subprocess, "run", lambda argv, **kw:
                            subprocess.CompletedProcess(argv, 1, b"", b"termination denied"))
        with pytest.raises(ArwenRefusal, match="termination denied"):
            manager.cancel("job-20260927-000003-test")
        assert process.poll() is None
        assert not (directory / "result.json").exists()
        assert manager.gpu_lock.holder()["job_id"] == "job-20260927-000003-test"
    finally:
        monkeypatch.undo()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=20)
