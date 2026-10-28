"""SHA-256 of a file, computed once per process for each unchanged file.

THE COST THIS REMOVES.  One WRF-input leg hashed its 438 MB wrfinput four
times (before the read, after it to prove nothing changed underneath, for
the preparation receipt, and again at the end to prove the inputs did not
change during the run), and the classic-Thompson tables three times plus
once more at every checkpoint (the table validation and the checkpoint's
physics identity both hash every table byte).  Measured on a 300 x 300 x 49
member: about 0.8 s of input hashing and 0.7 s of table hashing per leg
on the stepping thread, and 0.11 to 0.14 s per checkpoint.

THE RULE.  A digest is reused only for the same file in the same state:
the same device and inode, size, modification time and change time (to the
nanosecond).  Any write to the file moves its change time, which user
space cannot set back, so a file modified in place, replaced, truncated or
renamed over is hashed again -- every check that compared a digest before
and after a read, or before and after a run, still sees the change.  A
file whose state moved while it was being hashed is not remembered.
"""

from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import Future
from pathlib import Path

_LOCK = threading.Lock()
_MEMO: dict[tuple, str] = {}
#: Hashes running on a :func:`prefetch` thread, by the file state they started from.
_PENDING: dict[tuple, Future] = {}
#: Bound on remembered files; the oldest are dropped past it.
_LIMIT = 4096


def _state(path) -> tuple:
    st = os.stat(path)
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _hash(path, before, block_bytes) -> tuple[str, bool]:
    """``(digest, stable)``: the digest is remembered only when the file kept ``before`` throughout."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    value = digest.hexdigest()
    stable = _state(path) == before
    if stable:
        with _LOCK:
            if len(_MEMO) >= _LIMIT:
                _MEMO.pop(next(iter(_MEMO)))
            _MEMO[before] = value
    return value, stable


def sha256_file(path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    """Hex SHA-256 of the bytes of ``path``."""
    path = os.fspath(path)
    before = _state(path)
    with _LOCK:
        known = _MEMO.get(before)
        pending = _PENDING.get(before) if known is None else None
    if known is not None:
        return known
    if pending is not None:
        # A prefetch of the file in this same state is under way: its bytes, not a second pass over them.
        value, stable = pending.result()
        if stable:
            return value
    return _hash(path, before, block_bytes)[0]


def prefetch(path, *, block_bytes: int = 8 * 1024 * 1024) -> None:
    """Start hashing ``path`` on a background thread.

    A later :func:`sha256_file` of the file in the same state waits for that hash instead of reading the
    file again, so a digest the caller needs later (a checkpoint's, a bundle's) is computed beside the
    work in between rather than on the caller's thread.  The rule above is unchanged: a file whose state
    moved is hashed again by whoever asks.  A path that cannot be read is left to the caller, which then
    meets the error as it would have.
    """
    path = os.fspath(path)
    try:
        before = _state(path)
    except OSError:
        return
    with _LOCK:
        if before in _MEMO or before in _PENDING:
            return
        future = Future()
        _PENDING[before] = future
    hasher = _hash

    def run():
        try:
            outcome = hasher(path, before, block_bytes)
        except BaseException:  # noqa: BLE001 - the caller reads the file itself and meets the error
            outcome = (None, False)
        with _LOCK:
            _PENDING.pop(before, None)
        future.set_result(outcome)

    threading.Thread(target=run, name="file-digest-prefetch", daemon=True).start()


def forget() -> None:
    """Drop every remembered digest (tests)."""
    with _LOCK:
        _MEMO.clear()


__all__ = ["forget", "prefetch", "sha256_file"]
