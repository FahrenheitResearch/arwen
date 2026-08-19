"""The parity sweep's log is readable WHILE the sweep is running.

``tools/mapped_engine_parity_sweep.py`` is a twenty-minute job that
prints one verdict per row, and the only reason it prints per row is so
that somebody -- a person at a terminal, or a watcher grepping the
redirect a queued leg set up -- can see where it is.

Python line-buffers ``sys.stdout`` only when it is a terminal.  Redirect
it to a file, which is exactly what a queued leg does, and it becomes an
8 KiB block buffer: the 2.5.0 RC verify measured the log at 0 bytes for
the whole run, with every row appearing at once at exit.

That is the recorded buffered-log silent-green trap, and it is worse
than losing progress information.  A watcher reading that file sees no
FAIL rows because it sees NO rows, so it cannot tell "still running, all
good" from "died on row 3" until the process is over -- which is after
the only moment the answer was worth having.

The gate below drives the SHIPPED :func:`report` under the exact
condition that broke: stdout redirected to a file, read by another
process while the writer is still alive and has not exited.  A buffered
implementation cannot pass it, and it does not need the staged bytes or
the twenty minutes.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SWEEP = REPOSITORY_ROOT / "tools" / "mapped_engine_parity_sweep.py"

#: A child that emits rows through the sweep's own reporter and then
#: waits, so the parent can read the log at a moment the child is
#: provably still running.  It signals readiness on stderr, which is
#: unbuffered enough for a handshake and is not the stream under test.
_CHILD = """
import sys, time
from pathlib import Path
sys.path.insert(0, {tools!r})
import mapped_engine_parity_sweep as sweep

for index in range(3):
    sweep.report(f"row {{index}}: PASS")
Path({flag!r}).write_text("emitted", encoding="utf-8")
# Alive, and staying alive: anything the parent reads now was flushed
# during the run rather than at exit.
time.sleep(30)
"""


def test_a_row_reaches_the_redirected_log_before_the_sweep_exits(tmp_path):
    log = tmp_path / "sweep.log"
    flag = tmp_path / "emitted.txt"
    source = _CHILD.format(
        tools=str(REPOSITORY_ROOT / "tools"), flag=str(flag))

    with log.open("wb") as handle:
        child = subprocess.Popen(
            [sys.executable, "-c", source],
            stdout=handle, stderr=subprocess.PIPE, cwd=str(REPOSITORY_ROOT))
        try:
            deadline = time.time() + 60.0
            while not flag.is_file() and time.time() < deadline:
                if child.poll() is not None:
                    raise AssertionError(
                        "the reporter child exited before it signalled: "
                        + child.stderr.read().decode("utf-8", "replace"))
                time.sleep(0.05)
            assert flag.is_file(), "the reporter child never emitted its rows"

            # THE measurement: the child is still running, and the log
            # already carries the rows.
            assert child.poll() is None, (
                "the child exited on its own, so this would prove nothing "
                "about flushing DURING a run")
            text = log.read_text(encoding="utf-8", errors="replace")
        finally:
            child.kill()
            child.wait(timeout=30)
            if child.stderr is not None:
                child.stderr.close()

    assert "row 0: PASS" in text and "row 2: PASS" in text, (
        "the sweep's row verdicts did not reach a REDIRECTED stdout while "
        "the process was still alive; a watcher grepping this log would "
        f"stay silent through a crash. Log held {len(text)} bytes: {text!r}")


def test_the_sweep_reports_rows_through_the_flushing_helper():
    """The row prints go through :func:`report`, not through ``print``.

    Behaviour is gated above; this reads the source so that a new row
    added with a bare ``print`` is caught where it is written rather
    than only on the one path the gate above happens to drive.
    """

    source = SWEEP.read_text(encoding="utf-8")
    body = source.split("def main(", 1)[1]
    bare = [line.strip() for line in body.splitlines()
            if line.strip().startswith("print(")
            and "file=sys.stderr" not in line]
    assert bare == [], (
        "these rows print to stdout without flushing, so they vanish into "
        f"the block buffer of a redirected log: {bare}")
