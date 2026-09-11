"""Real worker subprocesses: handshake, logs, argv and durable outcomes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, code, args=("probe",)):
    package = tmp_path / "gpuwm"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "cli.py").write_text(
        "import json\ndef main(argv):\n"
        "    print('CLI ENTERED ' + json.dumps(argv), flush=True)\n" +
        "\n".join("    " + line for line in code.splitlines()) + "\n")
    directory = tmp_path / "job"
    directory.mkdir()
    environment = dict(os.environ, PYTHONPATH=str(tmp_path))
    with (directory / "job.log").open("wb") as output:
        process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "gpuwm/tui_worker.py"),
             "--job-dir", str(directory), "--", *args],
            cwd=tmp_path, env=environment, stdin=subprocess.DEVNULL,
            stdout=output, stderr=output)
    try:
        deadline = time.monotonic() + 10
        while not (directory / "ready").exists():
            assert process.poll() is None, (directory / "job.log").read_text()
            assert time.monotonic() < deadline
            time.sleep(.02)
        assert process.poll() is None
        assert "CLI ENTERED" not in (directory / "job.log").read_text()
        assert not (directory / "result.json").exists()
        (directory / "start").touch()
        status = process.wait(timeout=10)
        return status, json.loads((directory / "result.json").read_text()), \
            (directory / "job.log").read_text()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


@pytest.mark.parametrize("code,expected,state", [
    ("return None", 0, "completed"),
    ("return 7", 7, "failed"),
    ("raise SystemExit(2)", 2, "failed"),
    ("raise KeyboardInterrupt", 130, "interrupted"),
    ("raise ValueError('deliberate process fixture')", 1, "failed"),
])
def test_worker_outcomes_are_durable_and_only_start_after_marker(tmp_path, code, expected, state):
    status, record, log = _run(tmp_path, code)
    assert status == expected
    assert record["exit_code"] == expected
    assert record["status"] == state
    assert record["ended_at"] >= record["started_at"]
    assert "CLI ENTERED" in log
    if "ValueError" in code:
        assert record["error"]["type"] == "ValueError"
        assert "Traceback" in log


def test_worker_records_a_printed_refusal_as_the_jobs_error(tmp_path):
    """A refusal is a sentence at exit 2, and the receipt carries it.

    The CLI's refusal boundary prints ``gpuwm <command>: <sentence>`` and
    returns 2 instead of raising, so ``result.json`` used to hold only the
    exit code and a front door reading it could show the user nothing but
    "failed".  The receipt now records the sentence the way it already
    records an uncaught exception; an advisory printed before it is not
    mistaken for it, and a layered refusal keeps its follow-on lines.
    """
    status, record, log = _run(tmp_path, "\n".join([
        "import sys",
        "print('warning: parent cadence 3600 s is coarser than the 900 s guidance', file=sys.stderr)",
        "print('gpuwm probe: no complete gpuwmrst checkpoint set in /runs/parent; the parent needs restart_interval_s inside its window to be downscalable', file=sys.stderr)",
        "print('  Re-run the parent with restart_interval_s set.', file=sys.stderr)",
        "return 2"]))
    assert status == 2 and record["status"] == "failed"
    assert record["error"] == {
        "type": "Refusal",
        "message": "no complete gpuwmrst checkpoint set in /runs/parent; the "
                   "parent needs restart_interval_s inside its window to be "
                   "downscalable\n  Re-run the parent with restart_interval_s set."}
    # The log still carries everything, warning included.
    assert "warning: parent cadence" in log and "gpuwm probe: no complete" in log

    # A nonzero exit that printed no refusal sentence records no error: the
    # receipt never invents a reason.
    (tmp_path / "silent").mkdir()
    _, record, _ = _run(tmp_path / "silent", "\n".join([
        "import sys", "print('warning: only an advisory', file=sys.stderr)", "return 7"]))
    assert record["exit_code"] == 7 and "error" not in record

    # argparse's usage errors open the same way and are recorded too.
    (tmp_path / "usage").mkdir()
    _, record, _ = _run(tmp_path / "usage", "\n".join([
        "import sys", "print('gpuwm downscale: error: argument --ratio: invalid int value', file=sys.stderr)",
        "raise SystemExit(2)"]))
    assert record["error"] == {"type": "Refusal",
                               "message": "error: argument --ratio: invalid int value"}


def test_worker_preserves_literal_argv(tmp_path):
    args = ("probe", "a path with spaces", 'literal "quote"', "$(echo private)", "日本語")
    status, record, _ = _run(tmp_path, "return 0", args)
    assert status == 0
    assert record["cli_args"] == list(args)
