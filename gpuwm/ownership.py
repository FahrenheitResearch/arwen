"""One live owner for a shared file or folder.

Several writers in this package used to assume they were alone: the run
event stream, the downscale output folder and the bridge download.  Two
processes aimed at the same path then interleaved or deleted each
other's bytes.  This module is the one exclusive claim they all take.

A claim is an owner file created with ``O_CREAT | O_EXCL``.  It records
the owning process (pid, the process creation time and, on Linux, the
boot id), the host, what the claim is for and a random token.  A second
claimant that finds a live owner waits or refuses; one that finds an
owner whose process is gone, or whose pid now names a different process,
takes the claim over.  Release and every cleanup that depends on the
claim check the token first, so a writer never removes something another
owner now holds.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from gpuwm import proc_identity

OWNER_SCHEMA = "gpuwm.owner.v1"

#: An owner file that exists but cannot be parsed is normally one that its
#: creator has not finished writing.  Past this age it is a leftover from
#: a writer that died between create and write, and is taken over.
UNREADABLE_GRACE_SECONDS = 10.0

#: The takeover of a stale owner is itself serialised by a short-lived
#: breaker file.  One older than this was left by a process that died
#: during the takeover and is removed.
BREAKER_STALE_SECONDS = 30.0

#: States that pass on their own (an owner file its holder is deleting
#: right now, a stale one another claimant is removing) are retried for
#: this long even by a caller that does not wait for a live owner.
TRANSIENT_GRACE_SECONDS = 2.0

#: Windows answers a create over a file that is being deleted with access
#: denied, the same answer a folder that refuses writes gives.  Only this
#: platform needs the two told apart.
_WINDOWS = os.name == "nt"


class OwnershipError(RuntimeError):
    """Another process owns the path.  ``holder`` is its record.

    ``stale`` is True when that process has ended but its owner file
    could not be removed, so nothing will free the path on its own.
    """

    def __init__(self, message: str, *, path: Path, holder: dict | None,
                 stale: bool = False):
        super().__init__(message)
        self.path = path
        self.holder = holder or {}
        self.stale = stale


# ---------------------------------------------------------------------------
# Process identity
# ---------------------------------------------------------------------------

def process_created(pid: int) -> str | None:
    """An opaque creation stamp for ``pid``, or None when it is unknowable.

    The same stamp the page server and the job manager record beside every
    PID they keep (:mod:`gpuwm.proc_identity`): the process start time in
    clock ticks since boot on Linux, its creation FILETIME on Windows, and
    the start ``ps`` reports elsewhere.
    """

    found = proc_identity.identify(pid)
    return None if found is None else str(found["start"])


_KERNEL32 = None


def _kernel32():
    """A private kernel32 that keeps the last error, for the access check below."""

    global _KERNEL32
    if _KERNEL32 is None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL,
                                         wintypes.DWORD)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        _KERNEL32 = kernel32
    return _KERNEL32


def _present_but_hidden(pid: int) -> bool:
    """Whether something runs under ``pid`` that this user may not inspect.

    :func:`gpuwm.proc_identity.running` reads such a process as absent.  An
    owner claim cannot: taking over a claim whose holder is merely another
    user's live process is the mistake this module exists to prevent.
    """

    if os.name == "nt":
        import ctypes

        kernel32 = _kernel32()
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if handle:
            kernel32.CloseHandle(handle)
            return False
        return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return False


def pid_alive(pid: Any) -> bool:
    """Whether ``pid`` names a running process on this host."""

    number = proc_identity.pid_number(pid)
    if number is None:
        return False
    return proc_identity.running(number) or _present_but_hidden(number)


def process_identity(pid: int | None = None) -> dict[str, Any]:
    """Who a process is: pid, creation stamp, host and boot."""

    pid = os.getpid() if pid is None else int(pid)
    return {"pid": pid, "created": process_created(pid),
            "host": socket.gethostname(), "boot": proc_identity.boot_id()}


def identity_alive(record: dict) -> bool:
    """Whether the process a record names is still that same process.

    A record from another host cannot be checked from here and counts as
    alive: taking over a claim that may be live on another machine is the
    one mistake this module exists to prevent.
    """

    if record.get("host") not in (None, socket.gethostname()):
        return True
    boot = proc_identity.boot_id()
    if record.get("boot") and boot and record["boot"] != boot:
        return False
    pid = record.get("pid")
    if not pid_alive(pid):
        return False
    recorded = record.get("created")
    if recorded is not None:
        current = process_created(int(pid))
        if current is not None and str(current) != str(recorded):
            return False
    return True


# ---------------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """One held owner file."""

    path: Path
    token: str
    record: dict = field(default_factory=dict)

    def held(self) -> bool:
        """True while the owner file on disk still carries this token."""

        current = _read_record(self.path)
        return bool(current) and current.get("token") == self.token

    def release(self) -> bool:
        """Remove the owner file if it is still ours.  True when removed.

        Never raises for a file that will not go: this runs in ``close``
        and ``finally`` blocks.  Windows refuses to delete a file another
        process has open, and a claimant reading the owner file to learn
        who holds it has it open for a moment, so the delete is retried
        briefly.  One that still fails is marked released instead, which
        the next claimant takes over at once rather than waiting on this
        process, which is still alive.
        """

        _forget(self)
        if not self.held():
            return False
        for delay in (0.01, 0.03, 0.1, 0.3, None):
            try:
                os.unlink(self.path)
                return True
            except FileNotFoundError:
                return False
            except PermissionError:
                if delay is None:
                    break
                time.sleep(delay)
            except OSError:
                break
        try:
            with open(self.path, "r+b") as handle:
                handle.truncate(0)
                handle.write(json.dumps({**self.record, "released": True})
                             .encode("utf-8"))
        except OSError:
            pass
        return False


_HELD: dict[str, list] = {}
_HELD_LOCK = threading.Lock()


def _key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _forget(claim: Claim) -> None:
    with _HELD_LOCK:
        entry = _HELD.get(_key(claim.path))
        if entry is not None and entry[0] is claim:
            _HELD.pop(_key(claim.path), None)


def _release_all_at_exit() -> None:
    with _HELD_LOCK:
        claims = [entry[0] for entry in _HELD.values()]
    for claim in claims:
        try:
            claim.release()
        except OSError:
            pass


atexit.register(_release_all_at_exit)


def _read_bytes(path: Path) -> bytes | None:
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return None


def _read_record(path: Path) -> dict | None:
    return _parse_record(_read_bytes(path))


def _parse_record(raw: bytes | None) -> dict | None:
    if not raw:
        return None
    try:
        record = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _age(path: Path) -> float | None:
    try:
        return time.time() - os.stat(path).st_mtime
    except OSError:
        return None


def _breaker(path: Path) -> Path:
    return Path(str(path) + ".break")


def _folder_refusal(error: PermissionError, blocker: Path,
                    since: float | None) -> float | None:
    """Tell a Windows create refused over a file being deleted from a
    folder that refuses writes.

    With ``blocker`` still there, a delete is in progress: that is
    contention, and None is returned so the caller keeps its own limit.
    With nothing in the way the folder itself refused.  That is retried
    for :data:`TRANSIENT_GRACE_SECONDS` in case a delete finished
    between the two looks, then ``error`` is raised, however long the
    caller would wait for a live owner: a setup that waits an hour for
    another one must not wait an hour on a folder it cannot write.
    Returns when the refusal started.
    """

    if os.path.lexists(blocker):
        return None
    now = time.monotonic()
    if since is None:
        return now
    if now - since >= TRANSIENT_GRACE_SECONDS:
        raise error
    return since


def _break_stale(path: Path, seen: bytes) -> bool:
    """Remove a stale owner file, unless someone replaced it meanwhile.

    True when the caller should try its create again at once: the file
    is gone, or it no longer holds the bytes judged stale.  False when
    another claimant is removing it right now, or it would not go.  A
    breaker create refused for permission is raised for the caller to
    judge.
    """

    breaker = _breaker(path)
    try:
        handle = os.open(breaker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        age = _age(breaker)
        if age is not None and age > BREAKER_STALE_SECONDS:
            try:
                os.unlink(breaker)
            except OSError:
                pass
        return False
    os.close(handle)
    try:
        if _read_bytes(path) != seen:
            return True
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            return False
        return True
    finally:
        try:
            os.unlink(breaker)
        except OSError:
            pass


def describe_holder(record: dict | None) -> str:
    """The owner in plain words, for a refusal."""

    if not record:
        return "another process"
    words = f"process {record.get('pid')}"
    if record.get("host") and record.get("host") != socket.gethostname():
        words += f" on {record['host']}"
    if record.get("purpose"):
        words += f" ({record['purpose']})"
    if record.get("claimed_utc"):
        words += f", since {record['claimed_utc']}"
    return words


def recovery_words(error: OwnershipError) -> str:
    """How to free a claim this host cannot judge, or "" when it can.

    A holder on this machine that has gone is taken over on its own.  One
    on another machine (a shared folder) cannot be checked from here, so
    the person who knows it has stopped is told what to remove.
    """

    if error.stale:
        return (" That process has ended, but its claim could not be "
                f"removed: delete {error.path.name} in that folder and "
                "start again.")
    host = error.holder.get("host")
    if not host or host == socket.gethostname():
        return ""
    return (f" If nothing is running on {host} any more, delete "
            f"{error.path.name} in that folder and start again.")


def claim(path, *, purpose: str, wait: float = 0.0, poll: float = 0.25,
          on_wait: Callable[[dict | None], None] | None = None) -> Claim:
    """Take the owner file at ``path``, or raise :class:`OwnershipError`.

    ``wait`` is how long to wait for a live owner to finish before
    refusing; 0 refuses at once.  ``on_wait`` is told once, with the
    holder's record, when a wait starts.  An OSError that is not
    contention (a folder that cannot be written, for instance) is raised
    as it is, so the caller can say what is wrong with the folder.  It is
    raised within :data:`TRANSIENT_GRACE_SECONDS` whatever ``wait`` is.
    """

    path = Path(path)
    started = time.monotonic()
    deadline = started + max(0.0, float(wait))
    transient = max(deadline, started + TRANSIENT_GRACE_SECONDS)
    told = False
    # When a Windows create of the owner file, or of the breaker, was
    # first refused with nothing in its way.
    refused_since = breaker_refused_since = None
    while True:
        token = uuid.uuid4().hex
        record = {"schema": OWNER_SCHEMA, **process_identity(),
                  "purpose": purpose, "token": token,
                  "claimed_utc": datetime.now(timezone.utc).strftime(
                      "%Y-%m-%d %H:%M:%S UTC")}
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            refused_since = None
        except PermissionError as error:
            # Windows answers a create over a file that is being deleted
            # with access denied; that passes in moments, and is waited
            # out.  Elsewhere it means the folder refuses writes.
            if not _WINDOWS:
                raise
            refused_since = _folder_refusal(error, path, refused_since)
            if refused_since is None and time.monotonic() >= transient:
                raise
            time.sleep(0.02)
            continue
        else:
            try:
                os.write(handle, json.dumps(record).encode("utf-8"))
                os.fsync(handle)
            except OSError:
                os.close(handle)
                try:
                    os.unlink(path)
                except OSError:
                    pass
                raise
            os.close(handle)
            held = Claim(path=path, token=token, record=record)
            with _HELD_LOCK:
                _HELD[_key(path)] = [held, 1, threading.get_ident()]
            return held

        seen = _read_bytes(path)
        if seen is None:
            # Gone between the create and the read (its holder released
            # it), or being deleted: try the create again.
            if time.monotonic() >= transient:
                raise OwnershipError(
                    f"{path.parent} is in use by another process.",
                    path=path, holder=None)
            time.sleep(0.02)
            continue
        holder = _parse_record(seen)
        if holder is None:
            age = _age(path)
            stale = age is not None and age > UNREADABLE_GRACE_SECONDS
        else:
            stale = bool(holder.get("released")) or not identity_alive(holder)
        if stale:
            try:
                broken = _break_stale(path, seen)
                breaker_refused_since = None
            except PermissionError as error:
                if not _WINDOWS:
                    raise
                breaker_refused_since = _folder_refusal(
                    error, _breaker(path), breaker_refused_since)
                broken = False
            if broken:
                continue
            if time.monotonic() >= transient:
                raise OwnershipError(
                    f"{path.parent} holds a claim left by "
                    f"{describe_holder(holder)}, which has ended.",
                    path=path, holder=holder, stale=True)
            time.sleep(0.05)
            continue
        if time.monotonic() >= deadline:
            raise OwnershipError(
                f"{path.parent} is in use by {describe_holder(holder)}.",
                path=path, holder=holder)
        if not told and on_wait is not None:
            on_wait(holder)
        told = True
        time.sleep(poll)


@contextmanager
def owned(path, *, purpose: str, wait: float = 0.0, poll: float = 0.25,
          on_wait: Callable[[dict | None], None] | None = None
          ) -> Iterator[Claim]:
    """Hold the claim for a block.  Re-entrant within one thread."""

    key = _key(Path(path))
    with _HELD_LOCK:
        entry = _HELD.get(key)
        if entry is not None and entry[2] == threading.get_ident():
            entry[1] += 1
            nested = entry[0]
        else:
            nested = None
    if nested is not None:
        try:
            yield nested
        finally:
            with _HELD_LOCK:
                entry = _HELD.get(key)
                if entry is not None and entry[0] is nested:
                    entry[1] -= 1
        return
    held = claim(path, purpose=purpose, wait=wait, poll=poll, on_wait=on_wait)
    try:
        yield held
    finally:
        held.release()


def held_claim(path) -> Claim | None:
    """The claim this process holds on ``path``, if any."""

    with _HELD_LOCK:
        entry = _HELD.get(_key(Path(path)))
    return None if entry is None else entry[0]


__all__ = [
    "Claim", "OWNER_SCHEMA", "OwnershipError", "claim", "describe_holder",
    "held_claim", "identity_alive", "owned", "pid_alive", "process_created",
    "process_identity", "recovery_words",
]
