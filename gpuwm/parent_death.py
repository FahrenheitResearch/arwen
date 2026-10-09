"""A stage process dies with the process that launched it.

THE BREAKAGE (box B, 2026-10-07): a stopped ``gpuwm go`` -- its mutex
wrapper and the ``gpuwm.cli go`` process killed by pid -- left its
``gpuwm.forecast_supervisor`` child alive.  The child was reparented to
init and held 63.9 GB on a card for 43 minutes with no owner file,
starving the jobs queued behind that card and slowing a benchmark that
shared the host.  ``go`` had no SIGTERM handler, so it died at the default
disposition and nothing ever told its stage it was gone.

Two layers, so either one alone ends the orphan:

* **PR_SET_PDEATHSIG(SIGTERM)**, set by the child itself the moment it
  imports ``gpuwm`` (:func:`arm_from_environment`, armed from
  ``gpuwm/__init__``), and preserved by the kernel across the CLI's
  free-threading re-exec.  The kernel delivers SIGTERM the moment the
  launcher dies of anything -- SIGKILL included, which no handler in the
  launcher could ever answer.  Linux only.  Note the kernel's meaning of
  "parent": the THREAD that forked the child.  Every caller here forks
  from a thread that stays blocked on that child's pipes or exit status
  for as long as the child lives, so the thread outlives the child unless
  the whole launcher dies, which is exactly when the child must die.
* **A parent-pid watchdog** in the child (same function): the launcher's
  pid travels in :data:`PARENT_ENV`; a daemon thread polls ``getppid()``
  once a second and, once the child has been reparented, sends itself
  SIGTERM and exits outright :data:`WATCHDOG_GRACE_SECONDS` later if that
  did not end it.  It covers a kernel or container where prctl is
  refused, and any POSIX system without prctl.

A launcher that died between the fork and the child's arming is caught
in the child too: its pid is gone, so the child exits there instead of
starting.

NO PYTHON RUNS BETWEEN THE FORK AND THE EXEC (D-03, 2.8.8 acceptance).
2.8.7 set the death signal from a ``preexec_fn``.  That makes
``subprocess`` fork the whole interpreter and run Python in the child
before exec, and on the free-threaded build (2.8.8's default) that Python
ran the CuPy ``Event.__del__`` finalizers of objects the launcher's other
threads had just dropped -- in a child with no CUDA context.  From a
CUDA-live 3.14t launcher, 163 of 400 spawns printed
"Exception ignored ... cudaErrorInitializationError" tracebacks into the
render, stage and worker logs; ``preexec_fn`` with threads running can
also deadlock.  So the launcher passes only an environment entry
(:func:`child_environment`) and ``subprocess`` takes its exec-only fast
path, and every binding happens after exec in the child.

Nothing here detaches anything.  A process that is MEANT to outlive its
launcher (``gpuwm remote``'s durable job worker, which an ssh session
starts and then leaves) does not use these options and is unchanged.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time

#: The launcher's pid, handed to a stage so its watchdog knows whom to
#: outlive by no more than a poll.  Removed from the child's own
#: environment when the watchdog arms, so a grandchild does not inherit
#: a pid that was never its parent.
PARENT_ENV = "GPUWM_PARENT_DEATH_PID"

#: How often the watchdog asks whether its launcher is still its parent.
WATCHDOG_POLL_SECONDS = 1.0

#: How long a stage gets to answer the SIGTERM its watchdog sent before
#: it exits outright: the forecast's own stop grace
#: (``forecast_supervisor.STOP_GRACE_SECONDS``) is ten seconds.
WATCHDOG_GRACE_SECONDS = 10.0

_PR_SET_PDEATHSIG = 1
_EXIT_ON_TERM = 128 + int(signal.SIGTERM)

#: The launcher pid this process armed on, for :func:`reexec_environment`.
_ARMED_PARENT: int | None = None


def _bind_death_signal() -> bool:
    """PR_SET_PDEATHSIG(SIGTERM) on this process; False when unavailable.

    Called in the child after exec (never between a fork and an exec, see
    the module docstring).  A refused prctl (a seccomp profile) is not
    fatal: the watchdog still covers it.
    """

    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        prctl = ctypes.CDLL(None, use_errno=True).prctl
        return prctl(_PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0) == 0
    except (OSError, AttributeError):
        return False


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:   # EPERM: it exists, it is just not ours to signal
        return True
    return True


def child_environment() -> dict:
    """The environment entry that binds a child's life to this process.

    The whole of what a launcher passes: the child binds itself after
    exec (:func:`arm_from_environment`), so ``Popen`` needs no
    ``preexec_fn`` and runs no Python between fork and exec (D-03).
    """

    if os.name != "posix":
        return {}
    return {PARENT_ENV: str(os.getpid())}


def _watch(parent: int, poll: float, grace: float) -> None:
    while os.getppid() == parent:
        time.sleep(poll)
    try:
        sys.stderr.write(f"gpuwm: launcher pid {parent} is gone; this process "
                         f"(pid {os.getpid()}) is stopping so it cannot hold "
                         "its GPU as an orphan\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(grace)
    os._exit(_EXIT_ON_TERM)


def arm_from_environment(environ=None, *, poll: float = WATCHDOG_POLL_SECONDS,
                         grace: float = WATCHDOG_GRACE_SECONDS):
    """Bind this process's life to the launcher that asked for it.

    Sets PR_SET_PDEATHSIG(SIGTERM) and starts the parent-pid watchdog.
    Returns the watchdog thread, or ``None`` when :data:`PARENT_ENV` is
    absent, unusable or names a live process that is not this one's
    parent, or off POSIX.  When it names a launcher that has already
    died, the process exits here (status 128 + SIGTERM) instead of
    starting.  Never raises: a package import must not fail on a
    malformed variable.
    """

    environ = os.environ if environ is None else environ
    value = environ.pop(PARENT_ENV, None)
    if not value or os.name != "posix":
        return None
    try:
        parent = int(value)
    except ValueError:
        return None
    if parent <= 1:
        return None
    # Only a DIRECT child watches.  A process that inherited the variable
    # through something else (a shell, a native binary that starts Python)
    # has a different parent from birth while that launcher still runs,
    # and reading that as "the launcher died" would stop a healthy run.
    # A launcher that is GONE already died between the fork and this
    # line: the child must not start (the race prctl alone cannot see).
    if os.getppid() != parent:
        if not _process_exists(parent):
            os._exit(_EXIT_ON_TERM)
        return None
    _bind_death_signal()
    # The launcher may have died between the check above and the prctl,
    # in which case the kernel will never send the signal.
    if os.getppid() != parent:
        os._exit(_EXIT_ON_TERM)
    global _ARMED_PARENT
    _ARMED_PARENT = parent
    thread = threading.Thread(target=_watch, args=(parent, poll, grace),
                              name="gpuwm-parent-watchdog", daemon=True)
    thread.start()
    return thread


def reexec_environment() -> dict:
    """The entry that re-arms this process after it re-executes itself.

    :func:`arm_from_environment` removes :data:`PARENT_ENV` so a grandchild
    never inherits it, and the watchdog thread does not survive an exec.
    A stage that re-executes itself (the free-threading re-exec,
    :func:`gpuwm.free_threading.keep_gil_disabled`) keeps its pid and its
    parent, so it hands the entry back to its new image and is watched
    again.  Empty when this process was never armed.
    """

    if _ARMED_PARENT is None:
        return {}
    return {PARENT_ENV: str(_ARMED_PARENT)}


__all__ = ["PARENT_ENV", "WATCHDOG_GRACE_SECONDS", "WATCHDOG_POLL_SECONDS",
           "arm_from_environment", "child_environment", "reexec_environment"]
