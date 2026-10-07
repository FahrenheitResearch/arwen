"""A stage dies with the ``gpuwm go`` that launched it (gpuwm.parent_death).

THE BREAKAGE: on box B (2026-10-07) a stopped ``gpuwm go`` -- its mutex
wrapper and the ``gpuwm.cli go`` process killed by pid -- left its
``gpuwm.forecast_supervisor`` child reparented to init, holding 63.9 GB on
a card for 43 minutes with no owner file.

CPU only.  The chain here is the real one -- ``go_cli._run_stage`` wrapping
the stage in ``gpuwm.forecast_supervisor`` -- around a stub forecast that
records its pid and sleeps.  The launcher is then SIGKILLed, the hardest
case: no handler in it can run, so only the child's own binding can end
the child.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from gpuwm import go_cli, parent_death

REPO = Path(go_cli.__file__).resolve().parents[1]

#: "Within a few seconds": PR_SET_PDEATHSIG is immediate, and the watchdog
#: polls once a second.
EXIT_BOUND_SECONDS = 5.0

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="PR_SET_PDEATHSIG and /proc are Linux")

_STUB_FORECAST = '''
import os
import sys
import time
from pathlib import Path

def main(argv=None, *, observer=None):
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[argv.index("--outdir") + 1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "stub.pid").write_text(f"{os.getpid()} {os.getppid()}")
    time.sleep(600)
    return 0
'''


def _alive(pid: int) -> bool:
    """Running, not merely a zombie waiting for init to reap it."""

    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _wait_for(path: Path, *, timeout: float = 120.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.read_text().strip():
            return path.read_text()
        time.sleep(0.05)
    raise AssertionError(f"{path} never appeared")


def _gone_within(pid: int, seconds: float) -> float | None:
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        if not _alive(pid):
            return time.monotonic() - start
        time.sleep(0.05)
    return None


def _env(tmp_path: Path) -> dict:
    env = dict(os.environ)
    env.pop(parent_death.PARENT_ENV, None)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(REPO)])
    return env


@linux_only
@pytest.mark.parametrize("kill_with", ["SIGKILL", "SIGTERM"])
def test_forecast_supervisor_dies_when_go_is_killed(tmp_path, kill_with):
    (tmp_path / "stub_forecast.py").write_text(_STUB_FORECAST, encoding="utf-8")
    out = tmp_path / "run"
    launcher = textwrap.dedent(f'''
        import sys
        from gpuwm import go_cli
        go_cli._run_stage("forecast", [sys.executable, "-m", "stub_forecast",
                                       "--outdir", {str(out)!r}],
                          explain=False, heartbeat_seconds=3600)
        ''')
    go = subprocess.Popen([sys.executable, "-c", launcher], cwd=tmp_path,
                          env=_env(tmp_path), stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        stage_pid, stage_parent = map(int, _wait_for(out / "stub.pid").split())
        assert stage_parent == go.pid
        cmdline = Path(f"/proc/{stage_pid}/cmdline").read_bytes().split(b"\0")
        assert b"gpuwm.forecast_supervisor" in cmdline
        os.kill(go.pid, getattr(signal, kill_with))
        go.wait(10)
        took = _gone_within(stage_pid, EXIT_BOUND_SECONDS)
        assert took is not None, (
            f"forecast supervisor pid {stage_pid} outlived its killed launcher "
            f"by more than {EXIT_BOUND_SECONDS} s")
    finally:
        for pid in (go.pid,):
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass


@linux_only
def test_watchdog_alone_ends_an_orphaned_stage(tmp_path):
    """No prctl at all: the parent-pid watchdog by itself ends the orphan."""

    marker = tmp_path / "child.pid"
    child = (f"import os, time, gpuwm; open({str(marker)!r}, 'w').write(str(os.getpid())); "
             "time.sleep(600)")
    middle = textwrap.dedent(f'''
        import os, subprocess, sys, time
        from gpuwm.parent_death import PARENT_ENV
        env = dict(os.environ, **{{PARENT_ENV: str(os.getpid())}})
        subprocess.Popen([sys.executable, "-c", {child!r}], env=env)
        time.sleep(600)
        ''')
    launcher = subprocess.Popen([sys.executable, "-c", middle], cwd=tmp_path,
                                env=_env(tmp_path), start_new_session=True)
    try:
        pid = int(_wait_for(marker))
        os.kill(launcher.pid, signal.SIGKILL)
        launcher.wait(10)
        assert _gone_within(pid, EXIT_BOUND_SECONDS) is not None
    finally:
        try:
            os.killpg(launcher.pid, signal.SIGKILL)
        except OSError:
            pass


@linux_only
def test_preexec_refuses_to_start_once_the_launcher_is_gone(tmp_path, monkeypatch):
    """The fork-to-prctl race: a launcher already gone means no start."""

    monkeypatch.setattr(os, "getpid", lambda: 1)   # "the launcher" is not our pid
    options = parent_death.popen_options()
    monkeypatch.undo()
    completed = subprocess.run([sys.executable, "-c", "print('started')"],
                               capture_output=True, text=True, **options)
    assert completed.returncode == 128 + signal.SIGTERM
    assert "started" not in completed.stdout


def test_watchdog_arms_only_in_a_direct_child():
    """An inherited pid that is not this process's parent arms nothing.

    A native binary or shell between the launcher and a Python stage would
    otherwise read as "the launcher died" and stop a healthy run.
    """

    environ = {parent_death.PARENT_ENV: str(os.getpid())}
    assert parent_death.arm_from_environment(environ) is None
    assert parent_death.PARENT_ENV not in environ   # never handed down
    assert parent_death.arm_from_environment({}) is None
    assert parent_death.arm_from_environment(
        {parent_death.PARENT_ENV: "not-a-pid"}) is None


def test_stage_environment_names_this_process():
    env = go_cli._stage_env()
    if os.name == "posix":
        assert env[parent_death.PARENT_ENV] == str(os.getpid())
    else:
        assert parent_death.PARENT_ENV not in env


def test_interrupted_stage_wait_is_bounded():
    class Stubborn:
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("stage", timeout)

    started = time.monotonic()
    go_cli._await_interrupted_stage(Stubborn(), timeout=0.1)
    go_cli._await_interrupted_stage(None)
    assert time.monotonic() - started < 5
