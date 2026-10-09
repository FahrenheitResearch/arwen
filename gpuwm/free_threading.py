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
netCDF4 is one: the RRTMGP coefficient tables (:mod:`gpuwm.core.rrtmgp`)
and the netCDF4 writer workarounds still open it during a forecast, so
without this module every rank would quietly go back to taking turns.
No module imports netCDF4 at import time any more
(:data:`gpuwm.io.netcdf_serialization.netCDF4`, D-10): a process that
never opens a netCDF4 dataset -- a preparation hosted in another
program's process, which this re-exec cannot reach -- keeps the lock off.  The lock is not what keeps netCDF4 safe in this process: every
netCDF4 session runs under :data:`gpuwm.io.netcdf_serialization.NETCDF4_IO_LOCK`
(netCDF4 releases the interpreter lock inside HDF5 anyway, which is why
that lock exists).  So a command-line run re-executes itself once with
``PYTHON_GIL=0`` before it imports anything heavy, and that import can no
longer serialize the ranks.  An explicit ``PYTHON_GIL`` in the environment
is always respected, including ``PYTHON_GIL=1`` to force the lock on.

Windows has no ``exec`` that keeps the process: ``os.execve`` there starts a
new process and lets the caller's pid exit, which orphans whatever a
supervisor was tracking.  So on Windows the command line runs once more as a
child with ``PYTHON_GIL=0``, and this process waits for it and exits with its
status.  The child is held in a job object that kills it when this process's
handle closes, so a supervisor that terminates the pid it started also ends
the forecast; the job lets the child's own children break away, so anything
the forecast starts on purpose keeps its own lifetime.  Ctrl+C reaches both
processes on one console; the child owns the interruption and this process
only waits.  ``PYTHON_GIL`` is never set for the user's whole environment:
every Python built with the lock refuses to start when it sees
``PYTHON_GIL=0`` ("Disabling the GIL is not supported by this build").

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
    re-exec, or an interpreter that does not record its own command line.
    """
    if not free_threaded_build():
        return None
    if os.environ.get("PYTHON_GIL") is not None or os.environ.get(REEXEC_MARKER):
        return None
    original = list(getattr(sys, "orig_argv", None) or [])
    if len(original) < 2 or not sys.executable:
        return None
    return [sys.executable, *original[1:]]


#: ``JOBOBJECT_BASIC_LIMIT_INFORMATION.LimitFlags``: the job kills its
#: processes when its last handle closes, and the processes they start are
#: not put in it.
_JOB_KILL_ON_JOB_CLOSE = 0x2000
_JOB_SILENT_BREAKAWAY_OK = 0x1000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

#: What :func:`keep_gil_disabled` says when Windows will not make the job.
NO_JOB_WARNING = (
    "gpuwm: Windows would not create the job object that ties a re-run to "
    "this process, so this run keeps the interpreter lock: the slab threads of "
    "a multi-card [devices] forecast will take turns (set PYTHON_GIL=0 for "
    "this command to avoid it)")


def _kill_on_close_job():
    """A Windows job object that ends its processes when this one ends.

    Returns ``(kernel32, handle)``, or ``None`` when the job cannot be made.
    The handle is not inheritable and stays open for this process's life.
    """
    import ctypes
    from ctypes import wintypes

    class _Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _Io(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _Basic),
                    ("IoInfo", _Io),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    except (AttributeError, OSError):
        return None
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = _Extended()
    info.BasicLimitInformation.LimitFlags = (_JOB_KILL_ON_JOB_CLOSE
                                             | _JOB_SILENT_BREAKAWAY_OK)
    if not kernel32.SetInformationJobObject(
            job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info), ctypes.sizeof(info)):
        kernel32.CloseHandle(job)
        return None
    return kernel32, job


def run_bound_child(command: list[str], env: dict, job) -> int:
    """Run ``command`` as a child held in ``job``; return its exit status.

    ``job`` is what :func:`_kill_on_close_job` returned.  The child inherits
    this process's console and standard streams.
    """
    import signal
    import subprocess

    # Ctrl+C and Ctrl+Break go to every process on the console: the child
    # owns the interruption, this process only waits for its status.
    for name in ("SIGINT", "SIGBREAK"):
        number = getattr(signal, name, None)
        if number is not None:
            signal.signal(number, signal.SIG_IGN)
    child = subprocess.Popen(command, env=env)
    kernel32, handle = job
    kernel32.AssignProcessToJobObject(handle, int(child._handle))
    return child.wait()


def keep_gil_disabled() -> None:
    """Re-execute a command-line run once with ``PYTHON_GIL=0``.

    Call only from a command-line front door, before any heavy import and
    before any output that a re-exec would duplicate.  On POSIX ``exec``
    keeps the process id, so a supervisor that launched the command keeps
    tracking it; on Windows the run becomes a child this process waits on
    and is bound to (see the module docstring), and this process exits
    with the child's status.
    """
    command = reexec_command()
    if command is None:
        return
    env = dict(os.environ)
    env["PYTHON_GIL"] = "0"
    env[REEXEC_MARKER] = "1"
    # A stage bound to its launcher stays bound in its new image: same pid,
    # same parent, so it re-arms (gpuwm.parent_death).
    from gpuwm.parent_death import reexec_environment
    env.update(reexec_environment())
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass
    if os.name == "posix":
        os.execve(sys.executable, command, env)
    job = _kill_on_close_job()
    if job is None:
        print(NO_JOB_WARNING, file=sys.stderr)
        return
    raise SystemExit(run_bound_child(command, env, job))


__all__ = ["NO_JOB_WARNING", "REEXEC_MARKER", "free_threaded_build", "gil_enabled",
           "host_threads_report", "keep_gil_disabled", "reexec_command",
           "run_bound_child"]
