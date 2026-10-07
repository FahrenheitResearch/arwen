"""A dead packed DA controller takes its member workers down with it.

Breakage these tests pin: a controller stopped during a free leg left its
eight member forecast workers running as orphans (ppid 1) on every card, so
the next job queued on cards that looked free but were not.  The fake
controller here drives the real ``run_wave`` launcher (the CPU scheduler
seam ``check_mps=False``) with CPU-only sleeping workers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from gpuwm.da import member_wave as wave

ROOT = Path(__file__).resolve().parents[1]
LINUX = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="PR_SET_PDEATHSIG, process groups and POSIX signal delivery are Linux-only; "
           "packed DA member waves run on Linux GPU hosts")

WORKER = """\
import json, os, sys, time
path = sys.argv[1]
with open(path + ".tmp", "w") as stream:
    json.dump({"pid": os.getpid(), "ppid": os.getppid(), "sid": os.getsid(0)}, stream)
os.replace(path + ".tmp", path)
time.sleep(300)
"""

CONTROLLER = """\
import sys
from pathlib import Path
from gpuwm.da import member_wave as wave
root, worker, count = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
jobs = [{"member": m, "argv": [sys.executable, worker, str(root/f"started-{m}.json")],
         "result_path": str(root/f"m{m:03d}"/"result.json"), "output_root": str(root/f"m{m:03d}"),
         "request_hash": f"{m:064x}", "t_start": 0.0, "t_end": 60.0,
         "expected_domain_ids": [1], "ensemble_members": count} for m in range(count)]
cards = [f"GPU-{i:08x}-0000-0000-0000-000000000000" for i in range(2)]
wave.run_wave(jobs, cards, 4, root/"wave", 600, check_mps=False,
              host_available_bytes=251*wave.GIB)
"""


def _alive(pid):
    """True while ``pid`` exists and is not a zombie awaiting its reaper."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return fields[0] not in ("Z", "X")


def _start_controller(tmp_path, count=8):
    (tmp_path/"worker.py").write_text(WORKER)
    (tmp_path/"controller.py").write_text(CONTROLLER)
    environment = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="",
                       GPUWM_NO_LOCAL_GPU="1")
    controller = subprocess.Popen(
        [sys.executable, str(tmp_path/"controller.py"), str(tmp_path),
         str(tmp_path/"worker.py"), str(count)],
        env=environment, cwd=str(ROOT), stdin=subprocess.DEVNULL)
    deadline = time.monotonic()+60
    started = []
    while time.monotonic() < deadline:
        paths = [tmp_path/f"started-{m}.json" for m in range(count)]
        if all(path.exists() for path in paths):
            started = [json.loads(path.read_text()) for path in paths]
            break
        if controller.poll() is not None:
            break
        time.sleep(0.05)
    return controller, started


def _wait_gone(pids, seconds=5.0):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        survivors = [pid for pid in pids if _alive(pid)]
        if not survivors:
            return []
        time.sleep(0.05)
    return [pid for pid in pids if _alive(pid)]


def _cleanup(controller, started):
    # Only processes this test started, by explicit PID.
    for pid in [controller.pid, *(row["pid"] for row in started)]:
        if _alive(pid):
            try:
                os.kill(pid, getattr(signal, "SIGKILL"))
            except ProcessLookupError:
                pass
    if controller.poll() is None:
        controller.wait(timeout=10)


@LINUX
@pytest.mark.parametrize("name", ["SIGTERM", "SIGKILL", "SIGHUP"])
def test_killed_controller_takes_every_member_worker_down(tmp_path, name):
    signum = getattr(signal, name)
    controller, started = _start_controller(tmp_path)
    try:
        assert len(started) == 8, "fake controller did not start its eight workers"
        for row in started:
            # The launcher replaces itself with the worker: the controller is
            # still the direct parent and the worker still leads its own session.
            assert row["ppid"] == controller.pid
            assert row["sid"] == row["pid"]
        os.kill(controller.pid, signum)
        assert controller.wait(timeout=20) == -signum
        survivors = _wait_gone([row["pid"] for row in started])
        assert survivors == [], f"member workers outlived their controller: {survivors}"
        receipt = tmp_path/"wave"/"forecast-leg-receipt.json"
        if signum == signal.SIGKILL:
            assert not receipt.exists()
        else:
            document = json.loads(receipt.read_text())
            assert document["analysis_permitted"] is False and document["complete"] is False
            assert document["failure"].startswith("ControllerSignalled")
            assert document["stop_errors"] == []
    finally:
        _cleanup(controller, started)


@LINUX
def test_controller_signal_handlers_are_restored_after_a_wave(tmp_path):
    before = {number: signal.getsignal(number)
              for number in (getattr(signal, "SIGTERM"), getattr(signal, "SIGHUP"))}
    roster = [{"member": m, "argv": [sys.executable, "-c", "pass"],
               "result_path": str(tmp_path/f"m{m:03d}"/"result.json"),
               "output_root": str(tmp_path/f"m{m:03d}"), "request_hash": f"{m:064x}",
               "t_start": 0.0, "t_end": 60.0} for m in range(2)]
    with pytest.raises(wave.MemberWaveFailed):
        wave.run_wave(roster, ["GPU-00000000-0000-0000-0000-000000000000"], 2,
                      tmp_path/"wave", 10, host_available_bytes=251*wave.GIB)
    assert {number: signal.getsignal(number) for number in before} == before


@LINUX
def test_launcher_refuses_when_its_controller_is_already_gone(tmp_path):
    marker = tmp_path/"ran"
    argv = wave._launch_argv([sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"])
    # A parent PID that is not this child's parent stands in for a controller
    # that died between fork and prctl.
    argv[argv.index(str(os.getpid()))] = "1"
    finished = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert finished.returncode != 0
    assert "controller already exited" in finished.stderr
    assert not marker.exists()


@LINUX
def test_launcher_execs_the_worker_in_place(tmp_path):
    out = tmp_path/"ids.json"
    code = ("import json, os, sys; json.dump({'pid': os.getpid(), 'ppid': os.getppid()}, "
            f"open({str(out)!r}, 'w'))")
    process = subprocess.Popen(wave._launch_argv([sys.executable, "-c", code]))
    assert process.wait(timeout=30) == 0
    ids = json.loads(out.read_text())
    assert ids == {"pid": process.pid, "ppid": os.getpid()}


def test_launcher_leaves_the_vector_unchanged_off_linux(monkeypatch):
    monkeypatch.setattr(wave.sys, "platform", "win32")
    assert wave._launch_argv(["worker", "--flag"]) == ["worker", "--flag"]
    monkeypatch.setattr(wave.sys, "platform", "linux")
    launched = wave._launch_argv(["worker", "--flag"])
    assert launched[-2:] == ["worker", "--flag"] and launched[-3] == str(os.getpid())
