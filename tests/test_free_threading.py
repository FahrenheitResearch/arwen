"""The free-threaded interpreter seam (gpuwm.free_threading).

THE BREAKAGE THESE GUARD: on a free-threaded python3.14t, importing netCDF4
(gpuwm.io.wrfout, gpuwm.core.rrtmgp) re-enables the interpreter lock, and
every [devices] rank thread goes back to taking turns on the host -- the
multi-card slowdown this seam exists to remove.  A command-line run must
re-execute itself once with PYTHON_GIL=0 (on Windows as a child bound to the
caller's life); an explicit PYTHON_GIL must win; a GIL build and the
re-executed child must never re-exec; and both prepared runners, the
single-domain one and the domain-tree one, must take the seam.
"""
import os
import sys

import pytest

from gpuwm import free_threading as ft


@pytest.fixture
def free_threaded(monkeypatch):
    monkeypatch.setattr(ft, "free_threaded_build", lambda: True)
    monkeypatch.delenv("PYTHON_GIL", raising=False)
    monkeypatch.delenv(ft.REEXEC_MARKER, raising=False)
    monkeypatch.setattr(ft.os, "name", "posix")
    monkeypatch.setattr(ft.sys, "orig_argv",
                        ["/venv/bin/python3.14t", "-m",
                         "gpuwm.prepared_single_domain_forecast", "--io-mode", "history"])


def test_gil_build_never_reexecutes(monkeypatch):
    monkeypatch.setattr(ft, "free_threaded_build", lambda: False)
    monkeypatch.delenv("PYTHON_GIL", raising=False)
    assert ft.reexec_command() is None


def test_free_threaded_command_line_reexecutes_with_its_own_arguments(free_threaded):
    command = ft.reexec_command()
    assert command == [sys.executable, "-m", "gpuwm.prepared_single_domain_forecast",
                       "--io-mode", "history"]


@pytest.mark.parametrize("value", ["0", "1"])
def test_an_explicit_python_gil_is_respected(free_threaded, monkeypatch, value):
    monkeypatch.setenv("PYTHON_GIL", value)
    assert ft.reexec_command() is None


def test_the_reexecuted_child_cannot_loop(free_threaded, monkeypatch):
    monkeypatch.setenv(ft.REEXEC_MARKER, "1")
    assert ft.reexec_command() is None


def test_windows_reruns_as_a_bound_child_and_exits_with_its_status(free_threaded, monkeypatch):
    # os.execve on Windows spawns a new process and orphans the caller's pid,
    # so Windows runs the command as a child held in a kill-on-close job.
    monkeypatch.setattr(ft.os, "name", "nt")
    monkeypatch.setattr(ft.os, "execve", lambda *a: pytest.fail("execve on Windows"))
    job = object()
    monkeypatch.setattr(ft, "_kill_on_close_job", lambda: job)
    seen = {}

    def fake_child(command, env, held):
        seen.update(command=command, env=env, held=held)
        return 7

    monkeypatch.setattr(ft, "run_bound_child", fake_child)
    with pytest.raises(SystemExit) as stop:
        ft.keep_gil_disabled()
    assert stop.value.code == 7
    assert seen["held"] is job
    assert seen["command"] == [sys.executable, "-m", "gpuwm.prepared_single_domain_forecast",
                               "--io-mode", "history"]
    assert seen["env"]["PYTHON_GIL"] == "0" and seen["env"][ft.REEXEC_MARKER] == "1"


def test_windows_without_a_job_runs_on_with_the_lock_and_says_so(free_threaded, monkeypatch, capsys):
    # Without the job a killed caller would orphan the forecast, so it is not re-run.
    monkeypatch.setattr(ft.os, "name", "nt")
    monkeypatch.setattr(ft, "_kill_on_close_job", lambda: None)
    monkeypatch.setattr(ft, "run_bound_child", lambda *a: pytest.fail("ran an unbound child"))
    ft.keep_gil_disabled()
    assert ft.NO_JOB_WARNING in capsys.readouterr().err


def test_keep_gil_disabled_execs_with_python_gil_zero(free_threaded, monkeypatch):
    seen = {}

    def fake_execve(path, args, env):
        seen.update(path=path, args=args, env=env)
        raise SystemExit(0)

    monkeypatch.setattr(ft.os, "execve", fake_execve)
    with pytest.raises(SystemExit):
        ft.keep_gil_disabled()
    assert seen["path"] == sys.executable
    assert seen["env"]["PYTHON_GIL"] == "0"
    assert seen["env"][ft.REEXEC_MARKER] == "1"
    assert seen["args"][1:] == ["-m", "gpuwm.prepared_single_domain_forecast",
                                "--io-mode", "history"]


def test_keep_gil_disabled_is_inert_on_a_gil_build(monkeypatch):
    monkeypatch.setattr(ft, "free_threaded_build", lambda: False)
    monkeypatch.setattr(ft.os, "execve", lambda *a: pytest.fail("re-executed"))
    ft.keep_gil_disabled()


def test_host_threads_report_names_the_interpreter():
    report = ft.host_threads_report()
    assert report["python"] == sys.version.split()[0]
    assert isinstance(report["gil_enabled"], bool)
    assert isinstance(report["free_threaded_build"], bool)
    assert report["python_gil_env"] == os.environ.get("PYTHON_GIL")


@pytest.mark.parametrize("windows, rebuild", [
    (False, "rm -rf .venv && bash install.sh"),
    (True, "Remove-Item -Recurse -Force .venv; .\\install.ps1"),
])
def test_doctor_names_a_locked_interpreter_without_blocking(monkeypatch, windows, rebuild):
    # The installers make .venv on 3.14t by default since 2.8.8, so the remedy
    # is to let them make it again -- not to export GPUWM_PYTHON by hand.  It
    # is printed as comments only: doctor's closing line says every remedy line
    # runs "as printed, in the order printed", and a pasted report must not
    # delete the environment it runs in part-way through the other gaps'
    # commands (tests/test_doctor.py holds every remedy to that claim).  The
    # spelling follows the shell the report is forced to, not the host.
    from gpuwm import bridges, doctor
    monkeypatch.setattr(ft, "free_threaded_build", lambda: False)
    monkeypatch.setattr(bridges, "WINDOWS_SHELL", windows)
    check = doctor._host_threads_check()
    assert check.status == "info" and check.blocking is False
    assert "take turns" in check.detail and "2x" in check.detail
    lines = [line.strip() for line in check.remedy.splitlines() if line.strip()]
    assert all(line.startswith("#") for line in lines), lines
    assert any(line.lstrip("# ") == rebuild for line in lines), lines
    assert "GPUWM_PYTHON" not in check.remedy


@pytest.mark.parametrize("locked", [False, True])
def test_doctor_verifies_a_free_threaded_interpreter(monkeypatch, locked):
    from gpuwm import doctor
    monkeypatch.setattr(ft, "free_threaded_build", lambda: True)
    monkeypatch.setattr(ft, "gil_enabled", lambda: locked)
    check = doctor._host_threads_check()
    assert check.status == "verified" and check.brief == "free-threaded"
    assert ("lock is on in this process" in check.detail) is locked


class _Reexecuted(Exception):
    pass


@pytest.mark.parametrize("module", ["gpuwm.prepared_single_domain_forecast",
                                    "gpuwm.prepared_domain_tree_forecast"])
def test_both_prepared_runners_take_the_seam_first_from_the_command_line(monkeypatch, module):
    # The tree runner had no re-exec of its own: a multi-card tree run on
    # 3.14t stepped its ranks with the lock that netCDF4's import switched on.
    import importlib
    runner = importlib.import_module(module)
    calls = []

    def seam():
        calls.append("seam")
        raise _Reexecuted

    monkeypatch.setattr(ft, "keep_gil_disabled", seam)
    monkeypatch.setattr(sys, "argv", [module, "--show-capabilities"])
    with pytest.raises(_Reexecuted):
        runner.main()
    assert calls == ["seam"]


@pytest.mark.parametrize("module", ["gpuwm.prepared_single_domain_forecast",
                                    "gpuwm.prepared_domain_tree_forecast"])
def test_a_runner_hosted_in_process_never_reexecutes(monkeypatch, capsys, module):
    import importlib
    runner = importlib.import_module(module)
    monkeypatch.setattr(ft, "keep_gil_disabled", lambda: pytest.fail("re-executed a hosted runner"))
    assert runner.main(["--show-capabilities"]) == 0
    capsys.readouterr()
