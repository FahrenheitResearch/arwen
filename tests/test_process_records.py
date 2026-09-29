"""A recorded PID names one process, and one card is claimed by one launcher, across processes too.

The defects this file holds shut:

- A record's PID of ``true``, ``1.5``, ``NaN`` or a number past 32 bits
  was read as a process number: ``true`` became 1 (the system's first
  process), and Windows cut a large number to 32 bits, naming some other
  process that a stop could then reach.
- Two launchers in two processes checked the card, spawned and recorded
  the holder apart, and both ran on the card; the claim is one step under
  an operating-system lock, which this holds across real processes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from gpuwm import proc_identity


@pytest.mark.parametrize("pid", [True, False, 1.5, float("inf"), float("nan"), -1, 0, 2**32, 2**64, [], {},
                                 "not-a-pid"])
def test_a_value_that_is_not_a_pid_names_no_process(pid):
    own = proc_identity.identify(os.getpid())
    assert proc_identity.pid_number(pid) is None
    assert proc_identity.identify(pid) is None
    assert not proc_identity.running(pid)
    assert not proc_identity.alive({**own, "pid": pid})
    assert not proc_identity.alive(own, pid)
    assert not proc_identity.signal_process({**own, "pid": pid}, 0)


def test_two_launchers_in_two_processes_do_not_share_the_card(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "launch.py"
    script.write_text(
        "import json, pathlib, sys, time\n"
        "from gpuwm.mcp.jobs import JobManager\n"
        "from gpuwm.mcp.doors import ArwenRefusal\n"
        "root = pathlib.Path(sys.argv[1]); label = sys.argv[2]\n"
        "(root / (label + '.ready')).touch()\n"
        "while not (root / 'go').exists():\n"
        "    time.sleep(0.005)\n"
        "try:\n"
        "    result = JobManager(root / 'jobs').launch('owned-test', [sys.executable, '-c', 'import time; "
        "time.sleep(30)'], cwd=root, gpu=True)\n"
        "except ArwenRefusal as error:\n"
        "    result = {'refused': str(error)}\n"
        "(root / (label + '.json')).write_text(json.dumps(result, default=str))\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(root)}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    launchers = [subprocess.Popen([sys.executable, str(script), str(tmp_path), label], env=env, creationflags=flags)
                 for label in ("a", "b")]
    try:
        deadline = time.monotonic() + 30
        while not all((tmp_path / f"{label}.ready").exists() for label in ("a", "b")):
            assert time.monotonic() < deadline, "a launcher did not start"
            time.sleep(0.01)
        (tmp_path / "go").touch()
        for process in launchers:
            assert process.wait(timeout=60) == 0
        answers = [json.loads((tmp_path / f"{label}.json").read_text(encoding="utf-8")) for label in ("a", "b")]
        assert sorted("job_id" in answer for answer in answers) == [False, True], answers
        assert sum("refused" in answer for answer in answers) == 1, answers
    finally:
        for process in launchers:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
        from gpuwm.mcp.doors import ArwenRefusal
        from gpuwm.mcp.jobs import JobManager

        manager = JobManager(tmp_path / "jobs")
        for receipt in (tmp_path / "jobs").glob("job-*/receipt.json"):
            try:
                manager.cancel(receipt.parent.name)
            except ArwenRefusal:
                pass


def test_a_card_holder_written_before_holders_carried_an_identity_holds_while_its_pid_answers(tmp_path):
    """A run started by an earlier version and still going across the upgrade kept its card.

    Its lock names a PID and no identity, so it cannot prove which process it
    named; read as free, a second run started on a card the first still used.
    It holds while that PID answers, is never signalled, and frees once no
    process has the number.
    """

    from gpuwm.mcp.doors import ArwenRefusal
    from gpuwm.mcp.jobs import JobManager

    root = tmp_path / "jobs"
    root.mkdir()
    running = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        (root / "gpu.lock").write_text(json.dumps({"job_id": "job-before", "wrapper_pid": running.pid}),
                                       encoding="utf-8")
        manager = JobManager(root)
        assert manager.gpu_lock.refusal() is not None
        with pytest.raises(ArwenRefusal):
            manager.launch("sleep", [sys.executable, "-c", "pass"], cwd=tmp_path, gpu=True)
        assert running.poll() is None
    finally:
        running.kill()
        running.wait(timeout=10)
    assert JobManager(root).gpu_lock.refusal() is None
