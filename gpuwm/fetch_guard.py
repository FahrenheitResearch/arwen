"""Single-writer discipline and atomic publication for the fetch layer.

Every ``gpuwm fetch*`` flow writes into a directory that a later run --
or a *concurrent* run -- reads back as authority.  Two properties have
to hold for that to be safe, and neither is free:

**One writer at a time.**  The request-identity guard, the mutation of
the payloads, and the publication of the receipt that blesses them are
three separate steps.  Interleave two processes across them and the
final receipt can describe the other process's bytes.  Every mutating
flow therefore takes an exclusive, OS-enforced lock on its output root
first: a Windows byte-range lock / POSIX ``flock`` on a lock file kept
outside the output tree (so it never lands in a fetched directory and
never confuses the nonempty-output guard).  The kernel releases it when
the holder dies, which a cooperative sentinel file cannot promise --
a crashed fetch must not leave a directory permanently unfetchable.
The loser announces the wait, waits, and then refuses loudly naming the
holder; it never proceeds in parallel and never silently doubles work.

**Nothing is published half-written.**  Text receipts are written to a
temp that is unique per process *and* per call -- a fixed ``.tmp`` is
exactly the file two writers collide on -- flushed, fsynced, atomically
renamed, and the containing directory is fsynced too where the platform
allows it.  Quarantine names are proven absent before the rename, so
moving evidence aside can never overwrite older evidence.

Nothing here deletes anything.  Quarantine moves aside; the lock file
is the only file this module creates on its own, and it lives in the
per-user lock root, not in the fetched output.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

from gpuwm.explain import layered

#: How long a losing writer waits for the holder before refusing.
DEFAULT_LOCK_TIMEOUT_S = 600.0

#: Override for the wait budget (seconds).  A test or a batch driver
#: that would rather fail fast than queue sets this.
LOCK_TIMEOUT_ENV = "GPUWM_FETCH_LOCK_TIMEOUT_S"

#: Override for the lock root, for tests and for sandboxes where the
#: default per-user root is not writable.
LOCK_ROOT_ENV = "GPUWM_FETCH_LOCK_ROOT"

_POLL_S = 0.25

#: Re-entrancy and in-process exclusion.  The same *thread* may nest
#: ``hold()`` for one target -- the CLI takes it around the request
#: guard and the library function takes it again around the transfer --
#: and a second OS lock on one file from one process would deadlock
#: against itself on Windows, so nesting is counted rather than
#: re-locked.  A *different* thread is a different writer and has to
#: queue: the per-key ``RLock`` gives exactly that pair of behaviours,
#: and the OS lock underneath it excludes other processes.
_KEY_LOCKS: dict[str, threading.RLock] = {}
_HELD: dict[str, "_Entry"] = {}
_REGISTRY_GUARD = threading.Lock()


class FetchLockBusy(RuntimeError):
    """Another writer holds the output root; this one refuses."""


class _Entry:
    __slots__ = ("stream", "depth")

    def __init__(self, stream) -> None:
        self.stream = stream
        self.depth = 0


def _key_lock(key: str) -> threading.RLock:
    with _REGISTRY_GUARD:
        lock = _KEY_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _KEY_LOCKS[key] = lock
        return lock


def lock_root() -> Path:
    """Where lock files live: outside every fetched output tree."""

    override = os.environ.get(LOCK_ROOT_ENV)
    if override:
        return Path(override)
    if os.name == "nt":
        base = Path(os.environ.get("PROGRAMDATA", tempfile.gettempdir()))
    else:
        base = Path(tempfile.gettempdir())
    return base / "gpuwm" / "locks"


def lock_path(kind: str, target: str | Path) -> Path:
    """The lock file for ``kind`` over ``target``.

    Keyed by the *resolved* path, so ``--out .\\run`` and an absolute
    spelling of the same directory take the same lock, and a junction
    or symlink cannot split one directory into two writers.  The
    directory need not exist yet: ``Path.resolve()`` is non-strict.
    """

    resolved = Path(target).expanduser().resolve()
    key = str(resolved)
    if os.name == "nt":
        key = key.lower()  # NTFS is case-insensitive; the key must be too
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return lock_root() / f"{kind}-{digest}.lock"


def _is_contention(error: OSError) -> bool:
    """A held lock, as opposed to a disk, ACL, or descriptor failure."""

    if os.name == "nt":
        return error.errno in {errno.EACCES, errno.EDEADLK}
    return error.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}


def _try_lock(stream) -> bool:
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if _is_contention(error):
            return False
        raise
    return True


def _unlock(stream) -> None:
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except OSError:
        # Closing the stream releases the lock either way; a failure to
        # unlock explicitly must not mask the caller's own exception.
        pass


def describe_holder(path: Path) -> str:
    """Whatever the current holder recorded about itself, for refusals.

    The owner record sits past the locked byte, so it stays readable
    while the lock is held.  An unreadable or empty record is not an
    error -- the refusal simply says less.
    """

    try:
        with path.open("rb") as stream:
            stream.seek(1)
            raw = stream.read(4096)
    except OSError:
        return "an unidentified process"
    try:
        owner = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return "an unidentified process"
    if not isinstance(owner, dict):
        return "an unidentified process"
    pid = owner.get("pid")
    since = owner.get("acquired_at_utc")
    target = owner.get("target")
    parts = [f"pid {pid}" if pid is not None else "an unidentified process"]
    if since:
        parts.append(f"since {since}")
    if target:
        parts.append(f"on {target}")
    return " ".join(parts)


def _timeout(explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    raw = os.environ.get(LOCK_TIMEOUT_ENV)
    if raw is None:
        return DEFAULT_LOCK_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            f"{LOCK_TIMEOUT_ENV}={raw!r} is not a number of seconds"
        ) from error
    if value < 0:
        raise ValueError(f"{LOCK_TIMEOUT_ENV} cannot be negative")
    return value


class OutputLock:
    """Exclusive cross-process lock over one fetch output root."""

    def __init__(self, kind: str, target: str | Path, *,
                 timeout_s: float | None = None, progress=print,
                 clock=time.monotonic, sleeper=time.sleep) -> None:
        self.kind = kind
        self.target = Path(target)
        self.path = lock_path(kind, target)
        self.timeout_s = _timeout(timeout_s)
        self._progress = progress
        self._clock = clock
        self._sleeper = sleeper
        self._key = str(self.path).lower() if os.name == "nt" \
            else str(self.path)
        self._held = False

    def _busy(self, waited: float) -> FetchLockBusy:
        return FetchLockBusy(layered(
            f"another gpuwm fetch is writing {self.target} "
            f"({describe_holder(self.path)}) and this run waited "
            f"{waited:g} s for it.\n"
            "  remedy: wait for the other run to finish, fetch into a "
            f"different --out, or raise {LOCK_TIMEOUT_ENV} if the other "
            "run is expected to take longer.",
            "  why: two writers in one output directory can publish a "
            "receipt that describes the other one's bytes, so this run "
            "refuses rather than interleave."))

    def acquire(self) -> "OutputLock":
        key_lock = _key_lock(self._key)
        # Same thread: re-entrant, so nesting counts.  Another thread in
        # this process: a genuine second writer, so it queues here.
        if self.timeout_s <= 0:
            acquired = key_lock.acquire(blocking=False)
        else:
            acquired = key_lock.acquire(timeout=self.timeout_s)
        if not acquired:
            raise self._busy(self.timeout_s)
        self._held = True
        try:
            with _REGISTRY_GUARD:
                entry = _HELD.get(self._key)
            if entry is not None and entry.depth > 0:
                entry.depth += 1
                return self
            self._take_os_lock()
        except BaseException:
            self._held = False
            key_lock.release()
            raise
        return self

    def _take_os_lock(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            deadline = self._clock() + self.timeout_s
            announced = False
            while not _try_lock(stream):
                if self._clock() >= deadline:
                    raise self._busy(self.timeout_s)
                if not announced:
                    self._progress(
                        f"fetch: {self.target} is locked by "
                        f"{describe_holder(self.path)}; waiting up to "
                        f"{self.timeout_s:g} s for it to finish")
                    announced = True
                self._sleeper(_POLL_S)
        except BaseException:
            stream.close()
            raise
        owner = json.dumps({
            "pid": os.getpid(),
            "kind": self.kind,
            "target": str(self.target),
            "acquired_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, sort_keys=True).encode("utf-8")
        try:
            stream.seek(1)
            stream.truncate()
            stream.write(owner)
            stream.flush()
        except OSError:
            pass  # the lock is what matters; the record is a courtesy
        entry = _Entry(stream)
        entry.depth = 1
        with _REGISTRY_GUARD:
            _HELD[self._key] = entry

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            with _REGISTRY_GUARD:
                entry = _HELD.get(self._key)
            if entry is not None:
                entry.depth -= 1
                if entry.depth <= 0:
                    with _REGISTRY_GUARD:
                        _HELD.pop(self._key, None)
                    _unlock(entry.stream)
                    entry.stream.close()
        finally:
            _key_lock(self._key).release()

    def __enter__(self) -> "OutputLock":
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


def hold(kind: str, target: str | Path, *, timeout_s: float | None = None,
         progress=print) -> OutputLock:
    """``with hold('fetch-out', out):`` -- the single-writer contract."""

    return OutputLock(kind, target, timeout_s=timeout_s, progress=progress)


# ---------------------------------------------------------------------------
# Atomic publication
# ---------------------------------------------------------------------------

def _fsync_dir(directory: Path) -> None:
    """Best-effort durability for the rename itself.

    POSIX needs the containing directory fsynced before a rename is
    durable.  Windows has no directory handle to fsync through the
    stdlib; there the file's own flush plus ``os.replace`` is what the
    platform offers, and this is a no-op rather than a pretence.
    """

    if os.name == "nt":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _staging_path(path: Path, tag: str) -> Path:
    return path.with_name(
        f"{path.name}.{tag}-{os.getpid()}-{time.time_ns()}.tmp")


def atomic_write_bytes(path: Path, payload: bytes, *,
                       tag: str = "publish") -> Path:
    """Publish ``payload`` at ``path`` or leave the old bytes alone.

    The staging name carries the pid and a nanosecond stamp, so two
    publishers never share it -- the fixed ``<name>.tmp`` this replaces
    is the one file concurrent writers were guaranteed to collide on.
    """

    tmp = _staging_path(path, tag)
    try:
        with tmp.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)
    return path


def atomic_write_text(path: Path, text: str, *,
                      tag: str = "publish") -> Path:
    """:func:`atomic_write_bytes` for UTF-8 text with LF newlines."""

    return atomic_write_bytes(path, text.encode("utf-8"), tag=tag)


# ---------------------------------------------------------------------------
# Quarantine (never deletes, never overwrites older evidence)
# ---------------------------------------------------------------------------

def aside_path(path: Path, tag: str = "rejected") -> Path:
    """A free ``<name>.<tag>-<stamp>`` beside ``path``.

    ``time.time_ns()`` alone is not free: two quarantines inside one
    clock tick, or two processes, can generate the same name, and
    ``os.replace`` onto an existing file overwrites it -- turning
    "nothing is ever deleted" into a lie.  This proves the name absent
    and disambiguates with a counter when it is not.
    """

    stamp = time.time_ns()
    candidate = path.with_name(f"{path.name}.{tag}-{stamp}")
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = path.with_name(f"{path.name}.{tag}-{stamp}-{counter}")
    return candidate


def quarantine(path: Path, *, tag: str = "rejected") -> Path:
    """Move ``path`` aside to a proven-free name; returns the new path."""

    aside = aside_path(path, tag)
    os.replace(path, aside)
    return aside


__all__ = [
    "DEFAULT_LOCK_TIMEOUT_S",
    "FetchLockBusy",
    "LOCK_ROOT_ENV",
    "LOCK_TIMEOUT_ENV",
    "OutputLock",
    "aside_path",
    "atomic_write_bytes",
    "atomic_write_text",
    "describe_holder",
    "hold",
    "lock_path",
    "lock_root",
    "quarantine",
]
