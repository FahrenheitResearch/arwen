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


def _dead_pid() -> int:
    """A pid that existed a moment ago and is now reaped."""

    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    return gone.pid


@linux_only
def test_child_refuses_to_start_once_the_launcher_is_gone(tmp_path):
    """The fork-to-arm race: a launcher already gone means no start.

    The child's own check after exec, where 2.8.7 checked inside a
    preexec_fn (D-03).
    """

    env = _env(tmp_path)
    env[parent_death.PARENT_ENV] = str(_dead_pid())
    completed = subprocess.run(
        [sys.executable, "-c", "import gpuwm; print('started')"],
        capture_output=True, text=True, env=env, cwd=tmp_path)
    assert completed.returncode == 128 + signal.SIGTERM, completed.stderr
    assert "started" not in completed.stdout


@linux_only
def test_child_binds_its_death_signal_after_exec(tmp_path):
    """PR_SET_PDEATHSIG(SIGTERM) is set by the child itself at import."""

    probe = ("import ctypes, gpuwm, sys; v = ctypes.c_int(0); "
             "ctypes.CDLL(None).prctl(2, ctypes.byref(v), 0, 0, 0); print(v.value)")
    env = _env(tmp_path)
    env.update(parent_death.child_environment())
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                               text=True, env=env, cwd=tmp_path, check=True)
    assert int(completed.stdout.split()[-1]) == signal.SIGTERM
    env.pop(parent_death.PARENT_ENV)
    unbound = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                             text=True, env=env, cwd=tmp_path, check=True)
    assert int(unbound.stdout.split()[-1]) == 0


_NO_PYTHON_IN_FORKED_CHILD = '''
import os, sys
marker = os.open(sys.argv[1], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
# Runs in a forked child only when subprocess runs Python between fork
# and exec -- which a preexec_fn makes it do.
os.register_at_fork(after_in_child=lambda: os.write(marker, b"python ran in a forked child\\n"))
from gpuwm import first_products, go_cli
first_products._run_render([sys.executable, "-c", "pass"])
first_products._run_render([sys.executable, "-c", "pass"], own_group=True)
go_cli._run_stage("stage", [sys.executable, "-c", "pass"], explain=False,
                  heartbeat_seconds=3600)
print("spawned")
'''


@linux_only
def test_launchers_run_no_python_between_fork_and_exec(tmp_path):
    """D-03: the render and stage spawns take subprocess's exec-only path.

    THE BREAKAGE (2.8.8 acceptance, c8f95278): 2.8.7's preexec_fn made
    subprocess run Python in every forked render, stage and worker child
    before exec; on the free-threaded build that Python ran CuPy
    ``Event.__del__`` finalizers with no CUDA context, and 163 of 400
    spawns printed cudaErrorInitializationError tracebacks.  An at-fork
    hook runs in the child exactly when Python does, so any marker line
    is the defect, whatever the interpreter or the card.
    """

    marker = tmp_path / "forked-python.log"
    script = tmp_path / "launcher.py"
    script.write_text(_NO_PYTHON_IN_FORKED_CHILD, encoding="utf-8")
    completed = subprocess.run([sys.executable, str(script), str(marker)],
                               capture_output=True, text=True, cwd=tmp_path,
                               env=_env(tmp_path), timeout=300)
    assert completed.returncode == 0, completed.stderr
    assert "spawned" in completed.stdout
    assert not marker.exists() or marker.read_text() == "", marker.read_text()


def test_worker_spawn_passes_no_preexec_fn():
    """D-03 for the supervisor's worker spawn, which needs a card to run:
    no ``Popen`` in a CUDA launcher passes ``preexec_fn``."""

    import ast

    for module in ("first_products", "go_cli", "supervisor"):
        path = REPO / "gpuwm" / f"{module}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                names = {keyword.arg for keyword in node.keywords}
                assert "preexec_fn" not in names, f"{path}:{node.lineno}"
    assert not hasattr(parent_death, "popen_options")


def test_reexec_keeps_the_binding(monkeypatch):
    """The free-threading re-exec hands the launcher pid to its new image."""

    from gpuwm import free_threading

    monkeypatch.setattr(parent_death, "_ARMED_PARENT", None)
    assert parent_death.reexec_environment() == {}
    monkeypatch.setattr(parent_death, "_ARMED_PARENT", 4242)
    assert parent_death.reexec_environment() == {parent_death.PARENT_ENV: "4242"}
    captured = {}
    monkeypatch.setattr(free_threading, "reexec_command", lambda: ["python", "x"])
    monkeypatch.setattr(free_threading.os, "name", "posix")
    monkeypatch.setattr(free_threading.os, "execve",
                        lambda exe, argv, env: captured.update(env))
    free_threading.keep_gil_disabled()
    assert captured[parent_death.PARENT_ENV] == "4242"
    assert captured["PYTHON_GIL"] == "0"


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
