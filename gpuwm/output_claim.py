"""Live ownership of one output path, including supervised worker handoff.

The supervisor and its worker hold shared OS leases for the same random token.
A new launch needs an exclusive lease before it may adopt an empty directory.
Kernel-released locks make an interrupted launch recoverable without trusting a
PID, estimating its lifetime, or deleting any forecast files.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import os
from pathlib import Path
import time
from typing import BinaryIO, Callable
import uuid

from gpuwm.filesystem_paths import canonical_path, io_path


class OutputInUse(FileExistsError):
    """Another live launch owns this output path."""


def _lock(stream: BinaryIO, *, exclusive: bool) -> bool:
    """Try one byte-range lock on Windows, or one flock on POSIX."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        import msvcrt

        class Overlapped(ctypes.Structure):
            _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                        ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                        ("hEvent", wintypes.HANDLE)]

        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.LockFileEx.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped))
        api.LockFileEx.restype = wintypes.BOOL
        overlap = Overlapped()
        flags = 1 | (2 if exclusive else 0)  # immediate; exclusive or shared
        if api.LockFileEx(msvcrt.get_osfhandle(stream.fileno()), flags, 0, 1, 0,
                          ctypes.byref(overlap)):
            return True
        error = ctypes.get_last_error()
        if error == 33:  # ERROR_LOCK_VIOLATION
            return False
        raise ctypes.WinError(error)
    import fcntl
    try:
        fcntl.flock(stream.fileno(), (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        return True
    except OSError as error:
        if error.errno in (errno.EACCES, errno.EAGAIN):
            return False
        raise


def _unlock(stream: BinaryIO) -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        import msvcrt

        class Overlapped(ctypes.Structure):
            _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                        ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                        ("hEvent", wintypes.HANDLE)]

        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.UnlockFileEx.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.DWORD, ctypes.POINTER(Overlapped))
        api.UnlockFileEx.restype = wintypes.BOOL
        overlap = Overlapped()
        if not api.UnlockFileEx(msvcrt.get_osfhandle(stream.fileno()), 0, 1, 0,
                                ctypes.byref(overlap)):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _open_lock(path: Path) -> BinaryIO:
    # Sidecars are permanent: unlinking a locked file would let another caller
    # create and lock a different inode for the same output address.
    fd = os.open(io_path(path), os.O_RDWR | os.O_CREAT, 0o600)
    return os.fdopen(fd, "r+b", buffering=0)


@contextmanager
def _gate(path: Path):
    with _open_lock(path) as stream:
        deadline = time.monotonic() + 15
        while not _lock(stream, exclusive=True):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Output ownership update did not finish: {path}")
            time.sleep(.01)
        try:
            yield
        finally:
            _unlock(stream)


@dataclass
class OutputClaim:
    path: Path
    token: str
    _lease: BinaryIO

    def close(self) -> None:
        if not self._lease.closed:
            try:
                _unlock(self._lease)
            finally:
                self._lease.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def acquire_output(path: Path, *, prepare: Callable[[], Path], token: str | None = None) -> OutputClaim:
    """Reserve a new run, or join only the supervisor's exact output token.

    ``prepare`` applies the ordinary protected-path and prior-content checks
    while the output address is exclusively owned. Lock sidecars live beside
    outputs so an accepted precreated empty directory stays empty for existing
    prepared runners. Both leases remain held until their respective callers
    exit their complete launch/worker scope.
    """
    path = canonical_path(path)
    io_path(path.parent).mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(os.path.normcase(str(path)).encode("utf-8")).hexdigest()
    locks = path.parent / ".arwen-output-owners"
    io_path(locks).mkdir(exist_ok=True)
    with _gate(locks / (key + ".gate")):
        lease = _open_lock(locks / (key + ".lease"))
        try:
            if token is not None:
                if _lock(lease, exclusive=True):
                    raise ValueError("The supervisor's output ownership is no longer active")
                if not _lock(lease, exclusive=False):
                    raise OutputInUse(f"Another launch is reserving {path}")
                lease.seek(1)  # byte zero is the reserved Windows lock range
                stored = lease.read().decode("ascii", errors="replace")
                if stored != "arwen-output-owner-v1:" + token:
                    raise ValueError("The worker output ownership token no longer matches its supervisor")
                if not io_path(path).is_dir():
                    raise ValueError("The supervisor's claimed output directory is missing")
            else:
                if not _lock(lease, exclusive=True):
                    raise OutputInUse(f"Another running forecast owns {path}")
                path = prepare()
                token = uuid.uuid4().hex
                lease.seek(1)
                lease.truncate()
                lease.write(("arwen-output-owner-v1:" + token).encode("ascii"))
                lease.flush()
                # The gate prevents any competing acquisition during the
                # Windows unlock/relock transition. POSIX takes the same path.
                _unlock(lease)
                if not _lock(lease, exclusive=False):
                    raise RuntimeError("Could not retain the reserved output lease")
            return OutputClaim(path, token, lease)
        except BaseException:
            # Closing releases any acquired OS lock, including on token errors.
            lease.close()
            raise
