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
from pathlib import Path
import sys

import pytest

from gpuwm import free_threading as ft
from gpuwm.free_threading import multi_card_lock_reason as _real_lock_reason


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


@pytest.fixture
def real_lock_gate(monkeypatch):
    """Undo the suite-wide pin (tests/conftest.py) so the gate answers for real."""
    monkeypatch.setattr(ft, "multi_card_lock_reason", _real_lock_reason)
    monkeypatch.delenv("PYTHON_GIL", raising=False)


def _interpreter(monkeypatch, *, free_threaded, locked):
    monkeypatch.setattr(ft, "free_threaded_build", lambda: free_threaded)
    monkeypatch.setattr(ft, "gil_enabled", lambda: locked)


@pytest.mark.parametrize("free_threaded, locked", [(False, True), (True, True)])
def test_multi_card_split_is_refused_while_the_lock_is_on(real_lock_gate, monkeypatch,
                                                          free_threaded, locked):
    # An 8-card box got the wheel by hand into a venv on its GIL Python and
    # --devices 8 was about to run with nothing said: 2 cards measured 0.78x of 1.
    from gpuwm.core.devices import DevicesRefused
    _interpreter(monkeypatch, free_threaded=free_threaded, locked=locked)
    with pytest.raises(DevicesRefused) as caught:
        ft.refuse_multi_card_under_gil(8, "gpuwm go")
    assert ft.MULTI_CARD_GIL_REFUSAL in str(caught.value)
    assert "serialised by the GIL" in str(caught.value) and "8-card" in str(caught.value)


@pytest.mark.parametrize("free_threaded, locked", [(False, True), (True, True), (True, False)])
def test_one_card_is_never_refused(real_lock_gate, monkeypatch, free_threaded, locked):
    _interpreter(monkeypatch, free_threaded=free_threaded, locked=locked)
    ft.refuse_multi_card_under_gil(1, "gpuwm go")


def test_free_threaded_lock_off_runs_multi_card(real_lock_gate, monkeypatch):
    _interpreter(monkeypatch, free_threaded=True, locked=False)
    assert ft.multi_card_lock_reason() is None
    ft.refuse_multi_card_under_gil(8, "gpuwm go")


def _go_plan(tmp_path, devices):
    from gpuwm import go_cli
    root = Path(__file__).resolve().parents[1]
    return go_cli.plan_from_config(root / "configs" / "gfs_12km_quickstart.toml",
                                   outdir=tmp_path / "run", run_stamp=False,
                                   devices=devices)


def test_go_door_refuses_devices_two_on_a_gil_build(real_lock_gate, monkeypatch, tmp_path):
    from gpuwm.core.devices import DevicesRefused
    _interpreter(monkeypatch, free_threaded=False, locked=True)
    with pytest.raises(DevicesRefused, match="serialised by the GIL"):
        _go_plan(tmp_path, 2)


def test_go_door_passes_devices_one_on_a_gil_build(real_lock_gate, monkeypatch, tmp_path):
    _interpreter(monkeypatch, free_threaded=False, locked=True)
    plan = _go_plan(tmp_path, 1)
    assert plan["devices_count_override"] == 1


def test_go_door_passes_devices_two_free_threaded(real_lock_gate, monkeypatch, tmp_path):
    _interpreter(monkeypatch, free_threaded=True, locked=False)
    plan = _go_plan(tmp_path, 2)
    assert plan["devices_count_override"] == 2


@pytest.mark.parametrize("windows, rebuild", [
    (False, "rm -rf .venv && bash install.sh"),
    (True, "Remove-Item -Recurse -Force .venv; .\\install.ps1"),
])
def test_doctor_names_a_locked_interpreter_without_blocking(real_lock_gate, monkeypatch,
                                                            windows, rebuild):
    # The installers make .venv on 3.14t by default since 2.8.8, so the remedy
    # is to let them make it again -- not to export GPUWM_PYTHON by hand.  It
    # is printed as comments only: doctor's closing line says every remedy line
    # runs "as printed, in the order printed", and a pasted report must not
    # delete the environment it runs in part-way through the other gaps'
    # commands (tests/test_doctor.py holds every remedy to that claim).  The
    # spelling follows the shell the report is forced to, not the host.
    from gpuwm import bridges, doctor
    _interpreter(monkeypatch, free_threaded=False, locked=True)
    monkeypatch.setattr(bridges, "WINDOWS_SHELL", windows)
    check = doctor._host_threads_check()
    assert check.status == "info" and check.blocking is False
    assert sys.executable in check.detail and "GIL build" in check.detail
    assert "--devices N>1" in check.detail and "refuse" in check.detail
    assert "0.78x" in check.detail
    lines = [line.strip() for line in check.remedy.splitlines() if line.strip()]
    assert all(line.startswith("#") for line in lines), lines
    assert any(line.lstrip("# ") == rebuild for line in lines), lines
    assert any(line.lstrip("# ") == ft.MULTI_CARD_GIL_REFUSAL for line in lines), lines
    assert "GPUWM_PYTHON" not in check.remedy


def test_doctor_names_a_free_threaded_build_with_the_lock_forced_on(real_lock_gate, monkeypatch):
    from gpuwm import doctor
    _interpreter(monkeypatch, free_threaded=True, locked=True)
    monkeypatch.setenv("PYTHON_GIL", "1")
    check = doctor._host_threads_check()
    assert check.status == "info" and check.blocking is False
    assert "PYTHON_GIL=1" in check.detail and "--devices N>1" in check.detail
    assert ft.MULTI_CARD_GIL_REFUSAL in check.remedy


def test_doctor_verifies_a_free_threaded_interpreter(real_lock_gate, monkeypatch):
    from gpuwm import doctor
    _interpreter(monkeypatch, free_threaded=True, locked=False)
    check = doctor._host_threads_check()
    assert check.status == "verified" and "GIL off" in check.brief
    assert "free-threaded build" in check.detail


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


_POINT = {"gfs": "--point=39,-98", "hrrr": "--point=39,-98"}


def _front_door_go(tmp_path, source, devices):
    # gfs takes the prepared stage composer, hrrr the registered run-plan
    # launch: both branches of `gpuwm go --dry-run` reach the refusal.
    from gpuwm.cli import main
    from _staged_geog import staged_geog_tree
    config = tmp_path / f"{source}.toml"
    assert main(["domain", _POINT[source], "--card", "16gb", "--ladder", "12",
                 "--source", source, "--cycle", "2026-08-18T06",
                 "--hours", "3", "--out", str(config)]) == 0
    geog = staged_geog_tree(tmp_path / "geog" / "WPS_GEOG")
    return main(["go", str(config), "--dry-run", "--devices", str(devices),
                 "--geog-root", str(geog), "--outdir", str(tmp_path / "out"),
                 "--no-memory-gate"])


@pytest.mark.parametrize("source", ["gfs", "hrrr"])
def test_go_dry_run_refuses_devices_two_on_a_gil_build(real_lock_gate, monkeypatch,
                                                       tmp_path, capsys, source):
    _interpreter(monkeypatch, free_threaded=False, locked=True)
    rc = _front_door_go(tmp_path, source, 2)
    captured = capsys.readouterr()
    assert rc == 2, captured.out + captured.err
    assert ft.MULTI_CARD_GIL_REFUSAL in captured.err, captured.err


# The gfs branch's one-card pass is test_go_door_passes_devices_one_on_a_gil_build
# (its dry run also prices the local card, which a CI card may not fit).
@pytest.mark.parametrize("source", ["hrrr"])
def test_go_dry_run_passes_devices_one_on_a_gil_build(real_lock_gate, monkeypatch,
                                                      tmp_path, capsys, source):
    _interpreter(monkeypatch, free_threaded=False, locked=True)
    rc = _front_door_go(tmp_path, source, 1)
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    assert "serialised by the GIL" not in captured.err
