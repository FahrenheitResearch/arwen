"""The commit gate must FIRE, and nothing ever checked that it could.

``tools/ntiedtke_wrf461_oracle/check_no_forecast.sh`` refuses a commit
while a forecast is running, because any commit changes ``git_commit`` in
the run's identity and kills it at completion.

On 2026-08-29 it reported ``safe to commit`` over a live 14-hour run at
87%.  It matched only the command line, and the run had been launched
through a wrapper script that imports the entry point rather than naming
it in argv -- so none of its three patterns appeared.  The process was
``python.exe`` and WAS returned by the query; only the match missed.

That is the sixth matcher-matches-nothing in this tree and the first with
a live cost.  It is also the only one where nothing ever asserted the
matcher FIRES: every test here is that missing control.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

GATE = (Path(__file__).resolve().parents[1]
        / "tools" / "ntiedtke_wrf461_oracle" / "check_no_forecast.sh")

#: RESOLVE THE INTERPRETER, do not let CreateProcess pick one.
#: subprocess.run(["bash", ...]) searches PATH through CreateProcess, which
#: on this box finds System32ash.exe -- the WSL launcher -- before Git's.
#: WSL then reports "No such file or directory" for a Git-style `/c/` path
#: because it mounts that drive as /mnt/c, and every test in this file fails for a reason
#: unrelated to the gate.  shutil.which reports Git's bash; CreateProcess
#: chooses differently, and only the second one runs the script.
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(
    BASH is None or not GATE.is_file(),
    reason="needs bash and the gate script")


def msys(path) -> str:
    """Windows path -> the form MSYS bash accepts as an ARGUMENT.

    Two conversions, and both were needed before any test here ran:

    * backslashes are eaten by bash, so a native Windows path arrives
      with its separators gone -- ``C:UsersNameDesktop...``;
    * MSYS translates paths inside its own shell, but NOT a ``C:/...``
      argument handed to ``bash.exe`` by an outside process -- that is
      reported as "No such file or directory" for a file that plainly
      exists.

    Both failure modes make every assertion here pass or fail for a reason
    that has nothing to do with the gate, which is the shape of bug this
    whole file exists to catch.
    """
    text = Path(path).as_posix()
    if len(text) > 1 and text[1] == ":":
        text = f"/{text[0].lower()}{text[2:]}"
    return text


#: A pattern nothing on any box carries except this file's own fixtures.
#: The tests that assert the gate PASSES hand it to the script so a real
#: forecast running on the test box (a node is rarely idle) cannot turn a
#: test about the gate's logic red; the tests that assert it FIRES keep the
#: production pattern, since a fixture shaped to match its own matcher
#: would prove nothing.
SELFTEST_PATTERN = f"gate-selftest-{os.getpid()}-no-forecast-carries-this"


def run_gate(runs_root: Path | None = None, patterns: str | None = None):
    env = dict(os.environ)
    if runs_root is not None:
        env["GPUWM_RUNS_ROOT"] = msys(runs_root)
    if patterns is not None:
        env["GPUWM_GATE_PATTERNS"] = patterns
    return subprocess.run(
        [BASH, msys(GATE)], capture_output=True, text=True, env=env,
        timeout=120)


@pytest.fixture
def empty_runs(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    return root


# -- the control that was missing --------------------------------------------


def test_the_gate_fires_on_the_shape_that_defeated_it(tmp_path, empty_runs):
    """A wrapper script, invoked the way the live run actually was.

    NOT a synthesised command line built to satisfy the pattern -- a
    fixture shaped to match the matcher is a fixture that agrees with
    itself.  This spawns a real script from a temp directory with the
    forecast flags the old gate never looked at.
    """
    wrapper = tmp_path / "vram_timeline.py"
    wrapper.write_text(textwrap.dedent("""
        import time
        # stands in for: from gpuwm.prepared_domain_tree_forecast import main
        time.sleep(30)
    """), encoding="utf-8")

    proc = subprocess.Popen(
        [sys.executable, str(wrapper), "out.json",
         "--prepared-root", str(tmp_path / "prepared"),
         "--preparation-receipt-sha256", "0" * 64,
         "--io-mode", "history"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(3.0)                       # let it appear in the table
        result = run_gate(empty_runs)
        assert result.returncode == 1, (
            "the gate PASSED over a running wrapper-launched forecast -- "
            "this is the exact 2026-08-29 blind spot, reopened.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}")
        assert "REFUSING" in result.stderr
    finally:
        proc.kill()
        proc.wait(timeout=30)


def test_the_gate_fires_on_a_freshly_written_progress_file(empty_runs):
    """The behavioural arm, with no matching process anywhere.

    This is the arm that survives an invocation shape nobody predicted --
    it observes a forecast doing something rather than inferring it from
    how it was spelled.
    """
    d = empty_runs / "somecycle" / "output" / "somerun"
    d.mkdir(parents=True)
    (d / "progress.jsonl").write_text('{"event": "step"}\n', encoding="utf-8")

    result = run_gate(empty_runs)
    assert result.returncode == 1, (
        "a progress.jsonl written seconds ago did not stop a commit; the "
        f"behavioural arm is dead.\nstdout: {result.stdout}")
    assert "progress written" in result.stderr


# -- and the other direction, without which "always refuse" would pass -------


def test_the_gate_passes_when_nothing_is_running(empty_runs):
    """A gate that always refuses gets disabled by whoever it inconveniences
    first, so the permissive case is as load-bearing as the refusing one.
    """
    result = run_gate(empty_runs, patterns=SELFTEST_PATTERN)
    assert result.returncode == 0, (
        f"the gate refuses an idle box: {result.stderr}")
    assert "safe to commit" in result.stdout


def test_a_stale_progress_file_does_not_refuse_forever(empty_runs):
    """Every finished run leaves a progress.jsonl behind.  If age were not
    checked, the first completed forecast would block commits permanently
    and the gate would be removed within the day.
    """
    d = empty_runs / "oldcycle" / "output" / "oldrun"
    d.mkdir(parents=True)
    p = d / "progress.jsonl"
    p.write_text('{"event": "step"}\n', encoding="utf-8")
    old = time.time() - 3600
    os.utime(p, (old, old))

    result = run_gate(empty_runs, patterns=SELFTEST_PATTERN)
    assert result.returncode == 0, (
        f"an hour-old progress file still refuses: {result.stderr}")


def test_an_ordinary_python_process_is_not_a_forecast(empty_runs):
    """The widened pattern must not match python itself.

    The syntactic arm was widened to `--prepared-root` and
    `--preparation-receipt-sha256`; if it had been widened to something
    that matches any python, the gate would refuse constantly.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(3.0)
        result = run_gate(empty_runs, patterns=SELFTEST_PATTERN)
        assert result.returncode == 0, (
            "an unrelated python process is read as a forecast; the gate "
            f"will be disabled by the first person it blocks: {result.stderr}")
    finally:
        proc.kill()
        proc.wait(timeout=30)
