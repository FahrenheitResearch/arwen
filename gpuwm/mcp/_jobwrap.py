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

A job launched on a card shared through an OWNER file carries the file
in its receipt (``owner_file``); the launcher hands its line to this
wrapper's pid, and the wrapper removes that line the moment its child
has ended, before it writes ``result.json``.  The line so lives exactly
as long as the run, whether or not the page server that started it is
still running.  (Measured before this: a forecast started through
``gpuwm gui --owner-file`` kept its line after it ended, until a page
server on the same folder next looked, and programs that take the card
only when OWNER is empty waited on it.)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from gpuwm import proc_identity


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _publish(path: Path, document: dict) -> None:
    """Atomic JSON publication: a crash never leaves a truncated file.

    The temporary file is named for this process, since the job manager
    publishes result.json on a cancel at the same moment as this wrapper.
    """

    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(document, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _release_owner(owner_file: str, jobdir: Path) -> None:
    """Remove this wrapper's own OWNER line, and nobody else's (``gpuwm.machine_agent`` is stdlib only).

    The line names the pid the launcher recorded as ``wrapper_pid``.  That
    is this process, except under a Windows virtual environment, whose
    ``python.exe`` starts the real interpreter as its child and waits for
    it: there the recorded pid is that parent, which lives exactly as long
    as this process does.
    """

    try:
        from gpuwm.machine_agent import release_card

        pids = {os.getpid()}
        try:
            pids.add(int(json.loads((jobdir / "receipt.json").read_text(encoding="utf-8")).get("wrapper_pid") or 0))
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        for pid in sorted(pid for pid in pids if pid > 0):
            release_card(owner_file, pid)
    except Exception:  # noqa: BLE001 - the page server's sweep removes a line whose process has ended
        pass


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

    # A stop is a signal to the whole process group, this wrapper included.
    # The child answers it (a run-plan run writes its stopped event and
    # exits 130); the wrapper's job is to outlive it and record that exit.
    # Two breakages this prevents, both measured on a Linux node through
    # `gpuwm gui`: the wrapper died on the same Ctrl+C, wrote no
    # result.json and the job read as "lost"; and a server started in the
    # background (nohup ... &) hands its children SIGINT already IGNORED,
    # so the run printed "Ctrl-C cannot stop gpuwm run-plan" and ran on.
    # A SIGTERM is the job manager's cancel (MCP cancel_job, or the gui's
    # escalation after a Ctrl+C the run did not answer).  The manager has
    # already published result.json with cancelled true; the wrapper
    # still outlives the child, and when it writes its own result it must
    # keep that verdict, or a cancelled job reads as exited -15 seconds
    # later (measured on a Linux node at efeeea1e5).
    terminated = {"flag": False}
    preexec = None
    if os.name != "nt":
        import signal

        def preexec() -> None:
            signal.signal(signal.SIGINT, signal.SIG_DFL)

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        def on_term(*_: object) -> None:
            terminated["flag"] = True

        signal.signal(signal.SIGTERM, on_term)

    stdout_log = open(jobdir / "stdout.log", "ab")
    stderr_log = open(jobdir / "stderr.log", "ab")
    owner_file = receipt.get("owner_file")
    try:
        child = subprocess.Popen(
            receipt["argv"], cwd=receipt["cwd"], env=env,
            stdout=stdout_log, stderr=stderr_log,
            stdin=subprocess.DEVNULL, preexec_fn=preexec)
        _publish(jobdir / "started.json", {
            "child_pid": child.pid,
            # The child's identity, so a reader can tell it from a later process given the same pid.
            "child_process": proc_identity.identify(child.pid),
            "started_utc": _utc_now(),
        })
        while True:
            try:
                exit_code = child.wait()
                break
            except KeyboardInterrupt:
                continue
    except OSError as error:
        # A command that cannot be started (its program is gone, its folder cannot be entered) ends the job
        # here with a result like any command that exits nonzero; without one the job read as running until
        # the wrapper was found dead, and its reason was nowhere.
        exit_code = 127
        stderr_log.write(f"The job could not start: {error.strerror or error}"
                         f"{f' ({error.filename})' if error.filename else ''}.\n".encode("utf-8", "replace"))
    finally:
        stdout_log.close()
        stderr_log.close()
        if owner_file:
            _release_owner(str(owner_file), jobdir)
    cancelled = terminated["flag"]
    try:
        earlier = json.loads((jobdir / "result.json").read_text(
            encoding="utf-8"))
        cancelled = cancelled or bool(earlier.get("cancelled"))
    except (OSError, ValueError, AttributeError):
        pass
    _publish(jobdir / "result.json", {
        "exit_code": exit_code,
        "cancelled": cancelled,
        "ended_utc": _utc_now(),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
