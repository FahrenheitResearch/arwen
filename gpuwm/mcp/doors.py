"""Shelling out to the real gpuwm doors, and carrying refusals verbatim.

The MCP server never re-implements logic a door owns: a synchronous
tool runs ``python -m gpuwm.cli <door> ...`` as a subprocess against
THIS tree's own package (the interpreter running the server, cwd at the
tree root that provides :mod:`gpuwm`), waits, and hands back what the
door said.  The gold of the CLI contract is that every documented
refusal is ONE SENTENCE on stderr at exit 2; :func:`refusal_text`
extracts that sentence -- dropping only the provenance banner and the
``--explain`` pointer, which are about the invocation rather than the
refusal -- and :class:`ArwenRefusal` carries it verbatim to the MCP
client as the tool error, so a driving agent self-corrects off the same
words a person reads.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

#: The tree root that provides the running gpuwm package: parent of the
#: package directory.  In a checkout this is the repository root (which
#: also puts ``configs/`` and ``tools/`` where doors expect them); from
#: a wheel it is site-packages, where ``-m gpuwm.cli`` resolves anyway.
TREE_ROOT = Path(__file__).resolve().parents[2]

#: Default wall-clock ceiling for a synchronous door, seconds.  Long
#: work does not belong on a synchronous tool at all -- it goes through
#: :mod:`gpuwm.mcp.jobs` -- so this bounds only the quick JSON doors.
DEFAULT_TIMEOUT_S = 300.0


class ArwenRefusal(Exception):
    """A door refused (exit 2); ``str(self)`` is the sentence, verbatim."""


class DoorFailure(Exception):
    """A door failed for a non-refusal reason (crash, timeout, bad JSON)."""


def engine_argv(*door_args: str) -> list[str]:
    """The subprocess argv for one CLI invocation of this tree's gpuwm."""

    return [sys.executable, "-m", "gpuwm.cli", *door_args]


def _door_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    if extra:
        env.update(extra)
    return env


def _is_banner_or_pointer(line: str) -> bool:
    """Provenance banner / --explain pointer lines, not the refusal.

    The provenance gate prints ``gpuwm <cmd>: gpuwm <version> -- ...``
    before anything runs, and the refusal boundary appends an indented
    ``(run <invocation> --explain for the reason)`` pointer.  Both are
    about the invocation; the sentence between them is the refusal.
    """

    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("(run ") and "--explain" in stripped:
        return True
    if stripped.startswith("provenance:"):
        return True
    # The banner is the only stderr line whose message part restates the
    # program's own name and version: "gpuwm X: gpuwm 2.6.0 -- ...".
    parts = stripped.split(": ", 1)
    if len(parts) == 2 and parts[0].startswith("gpuwm") and \
            parts[1].startswith("gpuwm ") and " -- " in parts[1]:
        return True
    return False


def refusal_text(stderr: str) -> str:
    """The refusal sentence(s) out of a door's stderr, verbatim.

    Everything that is not the provenance banner or the ``--explain``
    pointer is kept unchanged -- a layered refusal can be more than one
    line, and paraphrasing is exactly what this transport must never do.
    """

    kept = [line for line in stderr.splitlines()
            if not _is_banner_or_pointer(line)]
    text = "\n".join(kept).strip()
    return text or stderr.strip()


def run_door(door_args: list[str], *, timeout_s: float = DEFAULT_TIMEOUT_S,
             env: dict[str, str] | None = None,
             cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    """Run one door to completion; refusals and crashes become exceptions.

    Exit 0 and exit 1 both RETURN (1 is a graded outcome on doors like
    ``doctor`` and ``verify`` -- the caller reads the payload and the
    code); exit 2 raises :class:`ArwenRefusal` with the sentence; any
    other exit raises :class:`DoorFailure` with the stderr tail.
    """

    command = f"gpuwm {door_args[0]}" if door_args else "gpuwm"
    try:
        # stdin=DEVNULL is load-bearing: this process's stdin is the MCP
        # protocol pipe, and a child that inherited it could read (and
        # eat) protocol frames addressed to the server.
        proc = subprocess.run(
            engine_argv(*door_args), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=timeout_s, cwd=str(cwd or TREE_ROOT),
            env=_door_env(env))
    except subprocess.TimeoutExpired as error:
        raise DoorFailure(
            f"{command} exceeded this tool's {timeout_s:.0f}s timeout and "
            "was stopped; a door that legitimately runs this long belongs "
            "on a job tool (arwen_fetch/arwen_prep/arwen_forecast/"
            "arwen_render), which does not hold the call open."
        ) from error
    if proc.returncode == 2:
        raise ArwenRefusal(refusal_text(proc.stderr))
    if proc.returncode not in (0, 1):
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-12:]
        raise DoorFailure(
            f"{command} exited {proc.returncode} (not a documented "
            "refusal); stderr tail:\n" + "\n".join(tail))
    return proc


def door_json(door_args: list[str], *,
              timeout_s: float = DEFAULT_TIMEOUT_S,
              env: dict[str, str] | None = None) -> tuple[object, int]:
    """Run a ``--json`` door and parse its stdout document.

    Returns ``(document, exit_code)`` -- exit 1 is a real outcome on the
    graded doors and the payload still parses there.
    """

    proc = run_door(door_args, timeout_s=timeout_s, env=env)
    try:
        return json.loads(proc.stdout), proc.returncode
    except json.JSONDecodeError as error:
        command = f"gpuwm {door_args[0]}"
        raise DoorFailure(
            f"{command} exited {proc.returncode} but its stdout is not "
            f"JSON ({error}); first 400 chars: {proc.stdout[:400]!r}"
        ) from error
