"""A stage process dies with the process that launched it.

THE BREAKAGE (box B, 2026-10-07): a stopped ``gpuwm go`` -- its mutex
wrapper and the ``gpuwm.cli go`` process killed by pid -- left its
``gpuwm.forecast_supervisor`` child alive.  The child was reparented to
init and held 63.9 GB on a card for 43 minutes with no owner file,
starving the jobs queued behind that card and slowing a benchmark that
shared the host.  ``go`` had no SIGTERM handler, so it died at the default
disposition and nothing ever told its stage it was gone.

Two layers, so either one alone ends the orphan:

* **PR_SET_PDEATHSIG(SIGTERM)**, set in the forked child before it execs
  (:func:`popen_options`).  The kernel delivers SIGTERM the moment the
  launcher dies of anything -- SIGKILL included, which no handler in the
  launcher could ever answer.  Linux only.  Note the kernel's meaning of
  "parent": the THREAD that forked the child.  Every caller here forks
  from a thread that stays blocked on that child's pipes or exit status
  for as long as the child lives, so the thread outlives the child unless
  the whole launcher dies, which is exactly when the child must die.
* **A parent-pid watchdog** in the child (:func:`arm_from_environment`,
  armed from ``gpuwm/__init__``): the launcher's pid travels in
  :data:`PARENT_ENV`; a daemon thread polls ``getppid()`` once a second
  and, once the child has been reparented, sends itself SIGTERM and exits
  outright :data:`WATCHDOG_GRACE_SECONDS` later if that did not end it.
  It covers a kernel or container where prctl is refused, and any POSIX
  system without prctl.

The preexec hook calls one libc function resolved before the fork, plus
``getppid`` and ``_exit``: no lock, no import, no I/O, which is what makes
a ``preexec_fn`` safe in a launcher that has other threads running.

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


def _prctl():
    """libc ``prctl``, resolved in the launcher (never after a fork)."""

    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes

        return ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None


def popen_options() -> dict:
    """``Popen`` keyword options that bind a child's life to this process.

    Empty off Linux.  The child sets PR_SET_PDEATHSIG(SIGTERM) before it
    execs, then checks that this process is still its parent: a launcher
    that died between the fork and the prctl would otherwise never send
    the signal, so the child exits there instead of starting.  A refused
    prctl (a seccomp profile) is not fatal; the watchdog still covers it.
    """

    prctl = _prctl()
    if prctl is None:
        return {}
    parent = os.getpid()
    term = int(signal.SIGTERM)

    def bind_to_launcher() -> None:
        prctl(_PR_SET_PDEATHSIG, term, 0, 0, 0)
        if os.getppid() != parent:
            os._exit(_EXIT_ON_TERM)

    return {"preexec_fn": bind_to_launcher}


def child_environment() -> dict:
    """The environment entry that arms a child's watchdog on this process."""

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
    """Start the parent-pid watchdog when a launcher asked for one.

    Returns the watchdog thread, or ``None`` when :data:`PARENT_ENV` is
    absent, unusable or names a process that is not this one's parent, or
    off POSIX.  Never raises: a package import must
    not fail on a malformed variable.
    """

    environ = os.environ if environ is None else environ
    value = environ.pop(PARENT_ENV, None)
    if not value or os.name != "posix":
        return None
    try:
        parent = int(value)
    except ValueError:
        return None
    # Only a DIRECT child watches.  A process that inherited the variable
    # through something else (a shell, a native binary that starts Python)
    # has a different parent from birth, and reading that as "the launcher
    # died" would stop a healthy run.  The death that happens before this
    # line is the preexec check's to catch (:func:`popen_options`).
    if parent <= 1 or os.getppid() != parent:
        return None
    thread = threading.Thread(target=_watch, args=(parent, poll, grace),
                              name="gpuwm-parent-watchdog", daemon=True)
    thread.start()
    return thread


__all__ = ["PARENT_ENV", "WATCHDOG_GRACE_SECONDS", "WATCHDOG_POLL_SECONDS",
           "arm_from_environment", "child_environment", "popen_options"]
