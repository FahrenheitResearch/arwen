"""The detached job wrapper: run one recorded argv, write the result.

``python -m gpuwm.mcp._jobwrap JOBDIR`` is what the MCP server actually
detaches.  It exists because a detached child cannot report its exit
code back to a server that may since have restarted: the wrapper reads
the launch receipt, runs the REAL command as its own child with stdout
and stderr appended to the job's logs, waits, and publishes
``result.json`` atomically.  The receipt on disk plus this file's two
documents (``started.json``, ``result.json``) are the whole job state,
so a restarted server reconstructs every job from the jobs directory
alone and no receipt is ever lost with a process.

Deliberately dependency-free (stdlib only) and import-light: it must
start fast and must not drag the engine into the wrapper process --
the child is the one that pays the engine's import bill.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _publish(path: Path, document: dict) -> None:
    """Atomic JSON publication: a crash never leaves a truncated file."""

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(document, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m gpuwm.mcp._jobwrap JOBDIR", file=sys.stderr)
        return 2
    jobdir = Path(args[0])
    receipt = json.loads((jobdir / "receipt.json").read_text(
        encoding="utf-8"))

    env = dict(os.environ)
    env.update(receipt.get("env_additions") or {})

    stdout_log = open(jobdir / "stdout.log", "ab")
    stderr_log = open(jobdir / "stderr.log", "ab")
    try:
        child = subprocess.Popen(
            receipt["argv"], cwd=receipt["cwd"], env=env,
            stdout=stdout_log, stderr=stderr_log,
            stdin=subprocess.DEVNULL)
        _publish(jobdir / "started.json", {
            "child_pid": child.pid,
            "started_utc": _utc_now(),
        })
        exit_code = child.wait()
    finally:
        stdout_log.close()
        stderr_log.close()
    _publish(jobdir / "result.json", {
        "exit_code": exit_code,
        "cancelled": False,
        "ended_utc": _utc_now(),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
