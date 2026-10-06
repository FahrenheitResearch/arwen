"""Rank threads that do not take turns: the free-threaded interpreter seam.

THE BREAKAGE THIS PREVENTS, measured 2026-10-03 on the HRRR grid
(1797 x 1057 x 50, full HRRR physics): every ``[devices]`` rank steps its
own card from its own Python thread (:class:`tilestream.ranks.RankedRun`),
and on an ordinary CPython build those threads share one interpreter lock.
CuPy gives the lock up and takes it back around every CUDA call, so one
ordinary step costs each rank about 3,900 hand-offs and a radiation step
about 140,000.  The hand-offs do not shrink when a rank's slab does, so
adding cards adds lock traffic faster than it removes GPU work: on a
two-socket 4 x RTX PRO 6000 box the four rank threads used about 1.2 cores
between them, the cards sat 28-59 % busy, and four cards were slower than
two (333.8 against 257.9 s per forecast hour).  On one build, four cards
took 289 s per forecast hour under the lock (radiation steps 17-40 s) and
121 s without it (4.4 s), with byte-identical histories.

A free-threaded build (``python3.14t``, PEP 703) has no such lock and the
rank threads run at once.  It re-enables the lock the moment any extension
module that has not declared itself free-threading safe is imported, and
netCDF4 is one: :mod:`gpuwm.io.wrfout` and :mod:`gpuwm.core.rrtmgp` import
it, so without this module every rank would quietly go back to taking
turns.  The lock is not what keeps netCDF4 safe in this process: every
netCDF4 session runs under :data:`gpuwm.io.netcdf_serialization.NETCDF4_IO_LOCK`
(netCDF4 releases the interpreter lock inside HDF5 anyway, which is why
that lock exists).  So a command-line run re-executes itself once with
``PYTHON_GIL=0`` before it imports anything heavy, and that import can no
longer serialize the ranks.  An explicit ``PYTHON_GIL`` in the environment
is always respected, including ``PYTHON_GIL=1`` to force the lock on.

Nothing here changes arithmetic: the same kernels run in the same order on
each card's streams, and cards still order against each other only through
CUDA events.  Byte identity across GIL and free-threaded runs is a measured
receipt, not an assumption (CHANGELOG 2.8.6).
"""
from __future__ import annotations

import os
import sys
import sysconfig

#: Set in the re-executed process so a second re-exec can never loop.
REEXEC_MARKER = "GPUWM_FREE_THREADING_REEXEC"


def free_threaded_build() -> bool:
    """True when this interpreter was built without the GIL (PEP 703)."""
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def gil_enabled() -> bool:
    """Whether the interpreter lock is active right now.

    A GIL build always answers True.  A free-threaded build answers False
    until an extension that has not declared free-threading support is
    imported without ``PYTHON_GIL=0``.
    """
    probe = getattr(sys, "_is_gil_enabled", None)
    return True if probe is None else bool(probe())


def host_threads_report() -> dict:
    """What the ranked receipts record about the host threads."""
    return dict(python=sys.version.split()[0],
                free_threaded_build=free_threaded_build(),
                gil_enabled=gil_enabled(),
                python_gil_env=os.environ.get("PYTHON_GIL"))


def reexec_command() -> list[str] | None:
    """The command line that re-runs this process with the lock kept off.

    ``None`` when no re-exec is needed or possible: a GIL build, an explicit
    ``PYTHON_GIL`` already in the environment, a process that is itself the
    re-exec, a platform whose ``exec`` does not replace the process (Windows
    spawns a new one, which would orphan the caller's pid), or an
    interpreter that does not record its own command line.
    """
    if not free_threaded_build():
        return None
    if os.environ.get("PYTHON_GIL") is not None or os.environ.get(REEXEC_MARKER):
        return None
    if os.name != "posix":
        return None
    original = list(getattr(sys, "orig_argv", None) or [])
    if len(original) < 2 or not sys.executable:
        return None
    return [sys.executable, *original[1:]]


def keep_gil_disabled() -> None:
    """Re-execute a command-line run once with ``PYTHON_GIL=0``.

    Call only from a command-line front door, before any heavy import and
    before any output that a re-exec would duplicate.  ``exec`` keeps the
    process id, so a supervisor that launched the command keeps tracking it.
    """
    command = reexec_command()
    if command is None:
        return
    env = dict(os.environ)
    env["PYTHON_GIL"] = "0"
    env[REEXEC_MARKER] = "1"
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass
    os.execve(sys.executable, command, env)


__all__ = ["REEXEC_MARKER", "free_threaded_build", "gil_enabled",
           "host_threads_report", "keep_gil_disabled", "reexec_command"]
