"""Actual subprocess witnesses for a readable, live preparation boundary."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from gpuwm import prep_output, source_cli
from gpuwm.command_output import AdapterOutputError, run_streamed
from test_source_adapters import _mapped_args
from test_stage_seams import _authority, _single_domain_bundle, _tree_bundle


def test_flushed_output_reaches_host_before_child_can_finish(tmp_path):
    acknowledgement = tmp_path / "ack"
    class LiveSink(io.StringIO):
        def write(self, text):
            result = super().write(text)
            if "ready" in self.getvalue():
                acknowledgement.touch()
            return result

    script = """
import os, pathlib, sys, time
ack = pathlib.Path(sys.argv[1])
print('ready', flush=True)
deadline = time.monotonic() + 10
while not ack.exists() and time.monotonic() < deadline:
    time.sleep(.01)
assert ack.exists(), 'host held progress until exit'
os.write(1, b'\\xe2')
os.write(1, b'\\x82\\xac')
os.write(2, b'E' * (2 * 1024 * 1024))
os.write(1, b'O' * (2 * 1024 * 1024))
"""
    stdout, stderr = LiveSink(), io.StringIO()
    result = run_streamed([sys.executable, "-c", script, str(acknowledgement)],
                          stdout, stderr)
    assert result.returncode == 0, stderr.getvalue()
    assert stdout.getvalue() == "ready\n\u20ac" + "O" * (2 * 1024 * 1024)
    assert stderr.getvalue() == "E" * (2 * 1024 * 1024)


def test_failed_output_destination_does_not_deadlock_child():
    class BrokenSink(io.StringIO):
        def write(self, text):
            raise OSError("diagnostic disk is full")
    with pytest.raises(AdapterOutputError, match="diagnostic disk is full"):
        run_streamed([sys.executable, "-c",
                      "import os; os.write(1,b'x'*2000000); os.write(2,b'y'*2000000)"],
                     BrokenSink(), io.StringIO())


@pytest.mark.parametrize("explain", [False, True])
@pytest.mark.parametrize("entry", ["rw-wps", "gpuwm prep"])
def test_real_preparation_failure_keeps_diagnostics_and_exit_code(
        tmp_path, monkeypatch, capsys, explain, entry):
    root = tmp_path / "prepared"
    argv = _mapped_args()
    argv[argv.index("--output-root") + 1] = str(root)
    if explain:
        argv.append("--explain")
    script = (
        "import sys; print('INTERNAL PROOF ' + chr(120)*200000); "
        "print('warning: source interval ends before requested forecast', file=sys.stderr); "
        "raise ValueError('Missing boundary time 2026-09-04T03:00Z; supply that input file.')"
    )
    monkeypatch.setattr(source_cli, "_mapped_command",
                        lambda args: [sys.executable, "-c", script])
    if entry == "rw-wps":
        code = source_cli.main(argv)
    else:
        from gpuwm.cli import main
        code = main(["prep", *argv])
    assert code == 1
    output = capsys.readouterr()
    assert "Missing boundary time" in output.err
    assert "warning: source interval" in output.err
    assert ("Traceback" in output.err) == explain
    assert ("INTERNAL PROOF" in output.out) == explain
    assert "Run the forecast" not in output.out
    assert not root.exists(), "logging must not pre-create the atomic output root"
    logs = list(tmp_path.glob("prepared-prep-*.log"))
    assert len(logs) == 1
    log = logs[0].read_text(encoding="utf-8")
    # Independent stdout/stderr pipes may interleave at chunk boundaries.
    # chr(120) also keeps Python3.13+ traceback source excerpts from adding
    # another literal payload character to the count.
    # Assert all emitted payload bytes survived, without imposing an ordering
    # that the operating system and live merged log do not promise.
    assert log.count("x") == 200000 and "Traceback" in log
    assert source_cli._ADAPTER_OUTPUT.get() is None


@pytest.mark.parametrize("traceback", [False, True])
@pytest.mark.parametrize("explain", [False, True])
def test_real_child_preserves_multiline_refusal_and_remedy(
        tmp_path, monkeypatch, capsys, traceback, explain):
    argv = _mapped_args()
    argv[argv.index("--output-root") + 1] = str(tmp_path / "prepared")
    if explain:
        argv.append("--explain")
    reason = "The source manifest names a different input directory."
    remedy = "Next: create a manifest for these input files."
    script = ("raise ValueError(" + repr(reason + "\n" + remedy) + ")" if traceback
              else "import sys; print(" + repr(reason + "\n" + remedy)
              + ", file=sys.stderr); sys.exit(78)")
    monkeypatch.setattr(source_cli, "_mapped_command",
                        lambda args: [sys.executable, "-c", script])
    from gpuwm.cli import main
    assert main(["prep", *argv]) == (1 if traceback else 78)
    terminal = capsys.readouterr()
    assert reason in terminal.err and remedy in terminal.err
    assert ("Traceback" in terminal.err) == (traceback and explain)
    assert "Details:" in terminal.err
    log = next(tmp_path.glob("prepared-prep-*.log")).read_text()
    assert reason in log and remedy in log


@pytest.mark.parametrize("tree", [False, True])
def test_next_command_uses_existing_sim_with_copied_authorities(tmp_path, tree):
    root = tmp_path / "prepared case"
    (_tree_bundle if tree else _single_domain_bundle)(root)
    config, wps = _authority(root)
    args = SimpleNamespace(output_root=root, experiment_config=None, wps_namelist=None)
    command = prep_output.forecast_command(args)
    assert "gpuwm sim" in command
    assert str(config) in command
    assert ("--wps-namelist" in command) != tree
    assert "sha256" not in command
    assert not root.with_name(root.name + "-forecast").exists()


def test_successful_prep_prints_handoff_without_receipt_wall(tmp_path, capsys):
    root = tmp_path / "prepared"
    args = SimpleNamespace(output_root=root, explain=False)
    def prepare():
        _single_domain_bundle(root)
        _authority(root)
        print(json.dumps({"proof_sha256": "e" * 64, "status": "READY"}))
        return 0
    assert prep_output.run_preparation(args, prepare) == 0
    output = capsys.readouterr()
    assert "prep: complete" in output.out and "gpuwm sim" in output.out
    assert "proof_sha256" not in output.out
    assert not output.err


def test_nested_host_keeps_ownership_of_adapter_output(tmp_path, monkeypatch):
    argv = _mapped_args()
    argv[argv.index("--output-root") + 1] = str(tmp_path / "prepared")
    monkeypatch.setattr(source_cli, "_mapped_command",
                        lambda args: [sys.executable, "-c", "print('raw host receipt')"])
    out, err = io.StringIO(), io.StringIO()
    with source_cli.redirect_adapter_output(out, err):
        assert source_cli.main(argv) == 0
    assert out.getvalue() == "raw host receipt\n"
    assert not list(tmp_path.iterdir())


def test_dry_run_keeps_exact_command_and_creates_no_log(tmp_path, monkeypatch, capsys):
    argv = _mapped_args() + ["--dry-run"]
    argv[argv.index("--output-root") + 1] = str(tmp_path / "prepared")
    monkeypatch.setattr(source_cli, "_mapped_command", lambda args: ["bridge", "--input", "a"])
    assert source_cli.main(argv) == 0
    assert capsys.readouterr().out == "bridge --input a\n"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("outcome, expected", [
    ("produced", "Companion WRF files: written"),
    ("not_requested", "Companion WRF files: not requested"),
    ("refused", "Companion WRF files: not produced (native preparation continues)"),
])
def test_real_child_stage_events_show_backend_and_actual_export_outcome(
        tmp_path, capsys, outcome, expected):
    root = tmp_path / "prepared"
    args = SimpleNamespace(output_root=root, explain=False)
    script = """
import sys
from gpuwm.progress import prep_stage
with prep_stage('root_initialize', label='Initialize forcing states', backend='cpu'):
    print('large internal receipt')
with prep_stage('wrf_export', label='Companion WRF files') as event:
    event['outcome'] = sys.argv[1]
"""
    def prepare():
        _single_domain_bundle(root)
        _authority(root)
        return source_cli._run_native_adapter([sys.executable, "-c", script, outcome])
    assert prep_output.run_preparation(args, prepare) == 0
    output = capsys.readouterr()
    assert "prep: Initialize forcing states (cpu)" in output.out
    assert expected in output.out
    assert "GPUWM_PREP_EVENT" not in output.out + output.err
    assert "large internal receipt" not in output.out
    log = next(tmp_path.glob("prepared-prep-*.log")).read_text(encoding="utf-8")
    assert "GPUWM_PREP_EVENT" in log and "large internal receipt" in log


def test_nested_progress_restores_parent_and_ignores_unrecognized_records():
    from gpuwm.prep_progress import PrepProgress
    from gpuwm.progress import PREP_EVENT_PREFIX, PREP_EVENT_SCHEMA
    progress = PrepProgress()
    def emit(stage, event):
        return progress.line(PREP_EVENT_PREFIX + json.dumps({
            "schema": PREP_EVENT_SCHEMA, "stage": stage, "label": stage,
            "event": event}))
    emit("parent", "started")
    emit("child", "started")
    assert progress.label == "child"
    assert progress.line(PREP_EVENT_PREFIX + "not json") is None
    assert progress.line(PREP_EVENT_PREFIX + "[]") is None
    emit("child", "finished")
    assert progress.label == "parent"
    emit("parent", "failed")
    assert not progress.active


def test_log_error_after_launch_never_relaunches_the_preparation(tmp_path, capsys):
    import errno
    counter = tmp_path / "attempts"
    class BrokenSink(io.StringIO):
        def write(self, text):
            raise OSError(errno.E2BIG, "output destination failed")
    script = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.open('a').write('ran\\n'); print('receipt')"
    with source_cli.redirect_adapter_output(BrokenSink(), io.StringIO()):
        code = source_cli._run_native_adapter(
            [sys.executable, "-c", script, str(counter), "--input", "unused"])
    assert code == 74
    assert counter.read_text().splitlines() == ["ran"]
    assert "output could not be saved" in capsys.readouterr().err


def test_child_unicode_survives_a_non_utf8_parent_setting(monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    stdout, stderr = io.StringIO(), io.StringIO()
    result = run_streamed([sys.executable, "-c", "print('\\u6e29\\u5ea6 20\\xb0C')"], stdout, stderr)
    assert result.returncode == 0, stderr.getvalue()
    assert stdout.getvalue() == "\u6e29\u5ea6 20\u00b0C\n"


@pytest.mark.parametrize("failure", ["write", "flush", "close"])
def test_real_prep_log_failure_reports_once_without_replaying_child(
        tmp_path, monkeypatch, capsys, failure):
    import errno
    real_fdopen = prep_output.os.fdopen
    class FailedLog:
        def __init__(self, stream):
            self.stream = stream
        @property
        def closed(self):
            return self.stream.closed
        def write(self, text):
            if failure == "write":
                raise OSError(errno.ENOSPC, "diagnostic log is full")
            return self.stream.write(text)
        def flush(self):
            if failure == "flush":
                raise OSError(errno.ENOSPC, "diagnostic log is full")
            return self.stream.flush()
        def close(self):
            self.stream.close()
            if failure == "close":
                raise OSError(errno.ENOSPC, "diagnostic log is full")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
    monkeypatch.setattr(prep_output.os, "fdopen",
                        lambda *args, **kwargs: FailedLog(real_fdopen(*args, **kwargs)))
    counter = tmp_path / "attempts"
    script = "import pathlib,sys; pathlib.Path(sys.argv[1]).open('a').write('ran\\n'); print('child receipt')"
    args = SimpleNamespace(output_root=tmp_path / "prepared", explain=False)
    code = prep_output.run_preparation(
        args, lambda: source_cli._run_native_adapter([sys.executable, "-c", script, str(counter)]))
    output = capsys.readouterr()
    assert code == 74
    assert counter.read_text().splitlines() == ["ran"]
    assert "diagnostic log is full" in output.err
    assert "Traceback" not in output.err
    assert "Run the forecast" not in output.out


def test_large_in_process_receipt_does_not_duplicate_it_in_host_ram(tmp_path, capsys):
    import tracemalloc
    root = tmp_path / "prepared"
    _single_domain_bundle(root)
    _authority(root)
    payload = "x" * (8 * 1024 * 1024) + "\n"
    args = SimpleNamespace(output_root=root, explain=False)
    def prepare():
        sys.stdout.write(payload)
        return 0
    tracemalloc.start()
    try:
        assert prep_output.run_preparation(args, prepare) == 0
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024 * 1024, f"host copied a large receipt: {peak} bytes"
    assert next(tmp_path.glob("prepared-prep-*.log")).read_text() == payload
    assert payload[:200] not in capsys.readouterr().out


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell handoff requires Windows")
def test_windows_handoff_preserves_literal_paths_in_actual_powershell():
    import subprocess
    paths = ["#prepared", "@prepared", "{prepared}", "owner's case", "", "normal-case"]
    commands = [prep_output.shell_command(["gpuwm", "sim", path, "--outdir", "forecast"])
                for path in paths]
    script = "function gpuwm { ConvertTo-Json -Compress -InputObject @($args) }\n"
    script += "\n".join(commands)
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive",
                             "-Command", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert [json.loads(line) for line in result.stdout.splitlines()] == [
        ["sim", path, "--outdir", "forecast"] for path in paths]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX handoff requires a POSIX shell")
def test_posix_handoff_preserves_literal_paths_in_actual_shell():
    import subprocess
    paths = ["#prepared", "@prepared", "{prepared}", "owner's case", "", "normal-case"]
    script = "gpuwm() { printf '%s\\n' \"$@\"; }\n"
    script += "\n".join(prep_output.shell_command(["gpuwm", "sim", path, "--outdir", "forecast"])
                        for path in paths)
    result = subprocess.run(["sh", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [word for path in paths
                                          for word in ("sim", path, "--outdir", "forecast")]
