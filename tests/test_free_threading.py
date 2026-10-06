"""The free-threaded interpreter seam (gpuwm.free_threading).

THE BREAKAGE THESE GUARD: on a free-threaded python3.14t, importing netCDF4
(gpuwm.io.wrfout, gpuwm.core.rrtmgp) re-enables the interpreter lock, and
every [devices] rank thread goes back to taking turns on the host -- the
multi-card slowdown this seam exists to remove.  A command-line run must
re-execute itself once with PYTHON_GIL=0; an explicit PYTHON_GIL must win; a
GIL build, Windows and the re-executed child must never re-exec.
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


def test_windows_does_not_reexec(free_threaded, monkeypatch):
    # os.execve on Windows spawns a new process and orphans the caller's pid.
    monkeypatch.setattr(ft.os, "name", "nt")
    assert ft.reexec_command() is None


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


def test_doctor_names_a_locked_interpreter_without_blocking(monkeypatch):
    from gpuwm import doctor
    monkeypatch.setattr(ft, "free_threaded_build", lambda: False)
    check = doctor._host_threads_check()
    assert check.status == "info" and check.blocking is False
    assert "take turns" in check.detail
    lines = [line.strip() for line in check.remedy.splitlines()]
    assert lines[1:] == ["export GPUWM_PYTHON=python3.14t", "bash install.sh"]


def test_doctor_verifies_a_free_threaded_interpreter(monkeypatch):
    from gpuwm import doctor
    monkeypatch.setattr(ft, "free_threaded_build", lambda: True)
    check = doctor._host_threads_check()
    assert check.status == "verified" and check.brief == "free-threaded"
