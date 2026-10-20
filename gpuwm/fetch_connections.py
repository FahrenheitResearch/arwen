"""Connection leases shared by engine fetch processes on one box.

Locks last for the response body (or the entire Rust subprocess), not
just the request headers. Kernel locks are released on process death.
Use the same cap and state directory for all engine processes on a box.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
import tempfile

DEFAULT_MAX_CONNECTIONS = 30
MAX_CONNECTIONS_ENV = "GPUWM_FETCH_MAX_CONNECTIONS"
DIRECTORY_ENV = "GPUWM_FETCH_CONNECTION_DIR"


def max_connections(value=None) -> int:
    raw = os.environ.get(MAX_CONNECTIONS_ENV, DEFAULT_MAX_CONNECTIONS) if value is None else value
    try:
        count = int(raw)
    except (TypeError, ValueError):
        count = 0
    if isinstance(raw, bool) or str(count) != str(raw) or count < 1:
        raise ValueError(f"fetch max connections {raw!r} must be a positive integer; an invalid cap cannot bound outbound sockets")
    return count


def _try_lock(handle) -> bool:
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
    else:
        import fcntl
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True



def _open_slot(path):
    """Open one box slot so every user of the machine can open it after us (mode 0666)."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o666)
    try:
        os.chmod(path, 0o666)
    except OSError:
        pass  # not ours: whoever created it already opened it to everyone
    return os.fdopen(fd, "a+b")


def _usable_slot_directory(directory):
    """The box-wide slot directory, shared by every user; per-user when another user's copy is closed to us.

    The slots live under /tmp so concurrent jobs contend for one box cap.  The first process to create the
    directory opens it to everyone (sticky, world-writable); a directory another user left closed (an older
    install, a stricter umask) refused every fetch on a shared machine with "[Errno 13] Permission denied:
    .../slot-0" (2.9.3 release battery on a host where root had created it), so this process then keeps its
    own per-user slots instead of failing.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(directory, 0o1777)
        except OSError:
            pass
        probe = directory / "slot-0"
        _open_slot(probe).close()
        return directory
    except OSError:
        uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
        own = directory.with_name(f"{directory.name}-{uid}")
        own.mkdir(parents=True, exist_ok=True)
        return own

@contextlib.contextmanager
def lease(count=1):
    """Acquire a bundle atomically, releasing partial bundles before waiting."""
    from gpuwm import fetch_pool
    cap = max_connections()
    if not 1 <= count <= cap:
        raise ValueError(f"{count} connections exceed the box fetch cap {cap}")
    # TMPDIR often differs between concurrent jobs. Unix processes must
    # still contend for the same box slots.
    shared_temp = Path(tempfile.gettempdir()) if os.name == "nt" else Path("/tmp")
    directory = Path(os.environ.get(DIRECTORY_ENV) or
                     (shared_temp / "gpuwm-fetch-connections"))
    directory = _usable_slot_directory(directory)
    handles = []
    try:
        while True:
            fetch_pool.raise_if_stopped()
            for index in range(cap):
                handle = _open_slot(directory / f"slot-{index}")
                if _try_lock(handle):
                    handles.append(handle)
                    if len(handles) == count:
                        break
                else:
                    handle.close()
            if len(handles) == count:
                break
            for handle in handles:
                handle.close()
            handles.clear()
            fetch_pool.sleep_unless_stopped(0.05)
        yield
    finally:
        for handle in handles:
            handle.close()


class LeasedResponse:
    """A urllib response whose socket slot is held until it closes."""
    def __init__(self, response, reservation):
        self._response = response
        self._reservation = reservation

    def __getattr__(self, name):
        return getattr(self._response, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            close = getattr(self._response, "close", None)
            if close is not None:
                close()
        finally:
            if self._reservation is not None:
                reservation, self._reservation = self._reservation, None
                reservation.__exit__(None, None, None)

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()
