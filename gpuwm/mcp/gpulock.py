"""One GPU job at a time, arbitrated by a lockfile the server owns.

Two CUDA compute contexts on one card contend for VRAM and can OOM or
slow each other into uselessness -- ``gpuwm run`` itself documents
shared-GPU operation as UNSUPPORTED -- so the MCP server admits ONE
GPU-touching job at a time and REFUSES the second by sentence, naming
the running job id.  Refusing rather than queueing is deliberate v1
policy: a silent queue is a launch tool that lies about having
launched.

The lock is a JSON file in the jobs root.  It is advisory and scoped to
this server's own launches (a person at the terminal is outside it);
one server per machine is the v1 assumption.  Staleness is resolved by
liveness: a lock whose holder process is gone releases itself at the
next acquire, so a crashed job cannot wedge the card forever.

The holder is named by process identity (:mod:`gpuwm.proc_identity`),
not by PID alone: a PID reused after a crash or a reboot is an
unrelated program, not the job, so it neither holds the card nor keeps
it held.

Checking the card and taking it is one step.  :meth:`GpuLock.claim`
holds an operating-system lock on ``gpu.lock.mutex`` across the check,
the caller's spawn and the write of the holder, so two starts in the
same instant (two browser tabs, the page and an MCP client) cannot both
pass the check: two simultaneous starts were both accepted and both
wrappers ran on one card.  The OS drops that lock when its process
dies, so a crash mid-start never wedges the card either.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from gpuwm import proc_identity

LOCK_NAME = "gpu.lock"
MUTEX_NAME = "gpu.lock.mutex"

#: In-process guard beside the OS lock: one thread of this process holds the mutex file at a time.
_THREADS = threading.RLock()
#: How deep the current thread is inside :func:`_os_lock`, per file.  Only the outermost entry takes the OS
#: lock: a second handle of the same file would wait on the first one's lock forever, in this very thread.
_DEPTH = threading.local()


@contextmanager
def _os_lock(path: Path) -> Iterator[None]:
    """An exclusive OS lock on ``path`` for the block; released by the OS if this process dies.

    Re-entrant in one thread (a release inside a claim is not a deadlock).
    """

    key = os.path.abspath(path)
    with _THREADS:
        held = getattr(_DEPTH, "held", None)
        if held is None:
            held = _DEPTH.held = {}
        depth = held.get(key, 0)
        held[key] = depth + 1
        try:
            if depth:
                yield
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a+b") as handle:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    while True:
                        try:
                            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                            break
                        except OSError:  # LK_LOCK gives up after ten tries a second apart; keep waiting
                            time.sleep(0.05)
                    try:
                        yield
                    finally:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - windows dev box
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            held[key] = depth


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness for a pid this user owns, both platforms."""

    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # SOMETHING answers to the pid; treat a process we cannot signal
        # as alive rather than reclaiming a lock out from under it.
        return True
    except OSError:
        return False
    return True


class GpuLock:
    """The jobs root's ``gpu.lock``: acquire, holder, release."""

    def __init__(self, jobs_root: Path):
        self.path = Path(jobs_root) / LOCK_NAME
        self.mutex_path = Path(jobs_root) / MUTEX_NAME

    def mutex(self):
        """The lock every write of ``gpu.lock`` happens under (see :meth:`claim`)."""

        return _os_lock(self.mutex_path)

    @staticmethod
    def _holder_alive(held: dict) -> bool:
        pid = proc_identity.pid_number(held.get("wrapper_pid"))
        if pid is None:
            return False
        if not isinstance(held.get("wrapper_process"), dict):
            # Written before holders carried an identity (a run started by an earlier version, still going across
            # the upgrade): it cannot prove which process it named, so while its PID answers it holds the card.
            # Freeing it put a second run on a card the first still used; it is never signalled on this reading.
            return proc_identity.running(pid)
        return proc_identity.alive(held.get("wrapper_process"), pid)

    def holder(self) -> dict | None:
        """The current lock document, or None (absent or unreadable)."""

        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def held_by(self) -> dict | None:
        """The record of the job that holds the card, or None when the card is free.

        This is what the refusal sentence is built from, for a reader that
        says who holds the card instead of refusing a launch (the page's
        Machines row): the lock document, with ``job_id``, ``wrapper_pid``
        and ``created_utc`` (when the job took the card).  A lock file that
        exists but cannot be read holds the card too, as ``{"unreadable":
        True, "path": ...}``: nothing can show the card is free beside a
        holder no one can name.

        A holder whose recorded process is gone is stale -- the job
        crashed or the machine rebooted, or its PID now names another
        program -- and does not hold the card.  Reading never deletes the
        file: a reader outside the mutex could otherwise remove a holder
        written a moment after it read the stale one.  :meth:`claim`
        reclaims a stale file under the mutex.
        """

        held = self.holder()
        if not isinstance(held, dict):
            if self.path.exists():
                return {"unreadable": True, "path": str(self.path)}
            return None
        return held if self._holder_alive(held) else None

    def refusal(self) -> str | None:
        """The launch refusal sentence when the card is held (see :meth:`held_by`), else None."""

        held = self.held_by()
        return None if held is None else self._refusal_for(held)

    def _refusal_for(self, held: dict) -> str:
        if held.get("unreadable"):
            # Unreadable lock: refuse rather than run beside an
            # unknown holder.
            return (f"the GPU lockfile {self.path} exists but cannot "
                    "be read, so this launch cannot prove the card is "
                    "free; inspect or remove the file and retry.")
        pid = held.get("wrapper_pid")
        job_id = held.get("job_id", "<unknown>")
        started = held.get("created_utc", "<unknown time>")
        return (f"the GPU is held by running job {job_id} (pid {pid}, "
                f"started {started}); a second CUDA run on the same card "
                "contends for VRAM and can OOM or corrupt the timing of "
                "both, so this launch is refused -- poll job_status "
                f"{job_id}, or job_cancel it, then relaunch.")

    def claim(self, start: Callable[[], tuple[str, int, dict | None]],
              before: Callable[[], None] | None = None) -> tuple[str, int, dict | None]:
        """Check the card, run ``start`` and record its job as the holder, as one step.

        ``start`` spawns the job and returns ``(job_id, wrapper_pid,
        wrapper_identity)``; it runs only when the card is free, and the
        mutex is held from the check until the holder is written, so a
        second claim waits and then sees this one.  ``before`` runs first
        under the mutex (the manager's release of a holder whose job has
        ended).  A refused claim raises :class:`CardHeld`; a ``start``
        that raises leaves the card free.
        """

        with self.mutex():
            if before is not None:
                before()
            held = self.held_by()
            if held is not None:
                raise CardHeld(self._refusal_for(held), held)
            if self.path.exists():
                # A stale holder (its process is gone): reclaimed here, under the mutex.
                try:
                    self.path.unlink()
                except OSError:
                    pass
            started = start()
            job_id, wrapper_pid, identity = started
            self._write(job_id, wrapper_pid, identity)
            return started

    def acquire(self, job_id: str, wrapper_pid: int, identity: dict | None = None) -> None:
        """Record this job as the holder (use :meth:`claim`, which checks and records as one step)."""

        with self.mutex():
            self._write(job_id, wrapper_pid, identity)

    def _write(self, job_id: str, wrapper_pid: int, identity: dict | None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "job_id": job_id,
            "wrapper_pid": wrapper_pid,
            "wrapper_process": identity,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(document, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def release(self, job_id: str) -> None:
        """Drop the lock iff this job holds it (never someone else's)."""

        with self.mutex():
            self.release_locked(job_id)

    def release_locked(self, job_id: str) -> None:
        """:meth:`release` for a caller already inside :meth:`mutex`."""

        held = self.holder()
        if held is not None and held.get("job_id") == job_id:
            try:
                self.path.unlink()
            except OSError:
                pass


class CardHeld(Exception):
    """The card is held: the message is the refusal sentence, ``holder`` the lock record it names."""

    def __init__(self, sentence: str, holder: dict | None = None) -> None:
        super().__init__(sentence)
        self.holder = holder
