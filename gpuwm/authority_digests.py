"""Input-file digests a preparation already holds, and its full re-reads moved beside the work.

Named breakage (WOOF fast start, 2026-10-23, Oklahoma City 48 h HRRR, 32 vCPU): one
preparation read the ~22 GB of GRIB it had just fetched four more times, each time on one
core while the other cores waited. The input manifest's authoring hashed every file and
re-hashed it after publishing (38 s); the composed decode hashed every file before its
parallel decode and again after it (about 20 s at each end). The fetch had already hashed
every byte it wrote, and records each digest in its route manifest
(``gpuwm.fetch_routes.MANIFEST_NAME``) beside the files.

Inside :func:`deferred_checks` (the preparation door, ``gpuwm.source_cli.dispatch``):

- a snapshot of a file this process already hashed, with the same device, inode, size,
  mtime and ctime, takes that digest without reading the file again;
- a snapshot of a file the fetch's route manifest records, with the same size and not
  modified or relinked since the manifest was written, takes the fetch's digest;
- every digest taken without a read, and every re-check of a snapshot after validation,
  is still a full re-read and compare. The re-check compares the file's identity on the
  caller (cheap, and it catches any change made through the file system). The full
  re-read runs on a background pool across the cores, beside the decode, and
  :func:`join` waits for it before the prepared cache is sealed
  (``gpuwm.mapped_direct``) and again when the scope ends. Any difference fails the
  preparation.

Outside the scope every snapshot reads and every re-check re-reads on the caller, as
before. The digests are the same values either way, so every manifest and prepared
byte is identical.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

_BLOCK = 8 * 1024 * 1024
_ROUTE_MANIFEST_NAME = "fetch-manifest.json"  # gpuwm.fetch_routes.MANIFEST_NAME
#: How far above a file the route manifest of the fetch that wrote it is looked for.
_ROUTE_MANIFEST_DEPTH = 3

_lock = threading.Lock()
_depth = 0
_pool: ThreadPoolExecutor | None = None
_memo: dict[tuple, str] = {}
_queued: dict[tuple, Future] = {}
_futures: list[Future] = []
_route_manifests: dict[str, tuple[int, dict[str, tuple[int, str]]]] = {}
_counts = {"held": 0, "fetch_recorded": 0, "hashed": 0, "deferred": 0,
           "deferred_bytes": 0}


def active() -> bool:
    """True inside :func:`deferred_checks`."""
    return _depth > 0


def identity(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (int(status.st_dev), int(status.st_ino), int(status.st_size),
            int(status.st_mtime_ns), int(status.st_ctime_ns))


def _key(path: Path, status: os.stat_result) -> tuple:
    return (str(path), *identity(status))


def _workers() -> int:
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    return max(1, min(32, count))


def _executor() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=_workers(),
                                   thread_name_prefix="input-digest")
    return _pool


_foreground: ThreadPoolExecutor | None = None


def map_parallel(function, items) -> list:
    """``[function(item) for item in items]`` across the cores, results in order.

    For the snapshot and re-check loops over a preparation's input files: each call
    reads one file (hashlib and file reads release the interpreter), so the loop is
    bounded by the disk and the cores, not by one thread. The first error, in item
    order, is raised after every call has finished. Its own pool, so a call never
    waits behind the deferred re-reads.
    """
    global _foreground
    items = list(items)
    if len(items) < 2:
        return [function(item) for item in items]
    with _lock:
        if _foreground is None:
            _foreground = ThreadPoolExecutor(max_workers=_workers(),
                                             thread_name_prefix="input-snapshot")
        pool = _foreground
    futures = [pool.submit(function, item) for item in items]
    results, error = [], None
    for future in futures:
        try:
            results.append(future.result())
        except BaseException as caught:  # noqa: BLE001 - re-raised below
            results.append(None)
            if error is None:
                error = caught
    if error is not None:
        raise error
    return results


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def remember(path: Path, status: os.stat_result, sha256: str) -> None:
    """A digest this process computed from the file's bytes at ``status``."""
    if not active():
        return
    with _lock:
        _memo[_key(path, status)] = sha256
        _counts["hashed"] += 1


def _route_rows(manifest: Path) -> tuple[int, dict[str, tuple[int, str]]] | None:
    try:
        status = manifest.stat()
    except OSError:
        return None
    cached = _route_manifests.get(str(manifest))
    if cached is not None and cached[0] == status.st_mtime_ns:
        return cached
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rows: dict[str, tuple[int, str]] = {}
    for entry in document.get("files") or ():
        if isinstance(entry, dict) and entry.get("relpath") and entry.get("sha256"):
            rows[str(entry["relpath"])] = (int(entry.get("bytes", -1)), str(entry["sha256"]))
    for entry in document.get("composed") or ():
        if isinstance(entry, dict) and entry.get("name") and entry.get("sha256"):
            rows[str(entry["name"])] = (int(entry.get("bytes", -1)), str(entry["sha256"]))
    held = (int(status.st_mtime_ns), rows)
    _route_manifests[str(manifest)] = held
    return held


def _fetch_recorded(path: Path, status: os.stat_result) -> str | None:
    """The fetch's digest of ``path``, when the file is still the one it wrote."""
    folder = path.parent
    for _ in range(_ROUTE_MANIFEST_DEPTH):
        manifest = folder / _ROUTE_MANIFEST_NAME
        if manifest.is_file():
            held = _route_rows(manifest)
            if held is None:
                return None
            written_ns, rows = held
            try:
                relpath = path.relative_to(folder).as_posix()
            except ValueError:
                return None
            row = rows.get(relpath)
            # The nearest route manifest decides. The file must have the recorded
            # size and must not have been written, renamed or relinked after the
            # manifest recorded it (ctime moves on every one of those).
            if (row is None or row[0] != int(status.st_size)
                    or int(status.st_mtime_ns) > written_ns
                    or int(status.st_ctime_ns) > written_ns):
                return None
            return row[1]
        if folder.parent == folder:
            return None
        folder = folder.parent
    return None


def known_digest(path: Path, status: os.stat_result) -> tuple[str, str] | None:
    """``(sha256, basis)`` for a file this scope need not read now, else None."""
    if not active() or not stat.S_ISREG(status.st_mode):
        return None
    with _lock:
        held = _memo.get(_key(path, status))
    if held is not None:
        with _lock:
            _counts["held"] += 1
        return held, "hashed earlier in this preparation"
    recorded = _fetch_recorded(path, status)
    if recorded is not None:
        with _lock:
            _memo[_key(path, status)] = recorded
            _counts["fetch_recorded"] += 1
        return recorded, "recorded by the fetch that wrote it"
    return None


def _verify(path: Path, expected: tuple, sha256: str, why: str) -> None:
    before = os.stat(path)
    if identity(before) != expected[1:]:
        raise ValueError(f"authority changed after validation: {path}")
    actual = hash_file(path)
    after = os.stat(path)
    if identity(after) != expected[1:]:
        raise ValueError(f"authority changed while it was re-read: {path}")
    if actual != sha256:
        raise ValueError(
            f"authority bytes differ from the digest the preparation used "
            f"({why}): {path} hashes to {actual}, not {sha256}")


def defer_verify(path: Path, status: os.stat_result, sha256: str, why: str) -> None:
    """Re-read ``path`` in full beside the work and compare it with ``sha256``.

    A request is satisfied by any re-read that starts after it, so a request for a
    file whose re-read is queued and not yet started joins that one.
    """
    key = _key(path, status)
    with _lock:
        queued = _queued.get(key)
        if queued is not None and not queued.running() and not queued.done():
            return
        future = _executor().submit(_verify, path, key, sha256, why)
        _queued[key] = future
        _futures.append(future)
        _counts["deferred"] += 1
        _counts["deferred_bytes"] += int(status.st_size)


def require_unchanged(path: Path, *, sha256: str, size: int, mtime_ns: int,
                      device: int, inode: int, mode: int) -> bool:
    """The re-check of a snapshot after validation, inside the scope.

    Returns False outside the scope (the caller re-reads, as before). Inside it the
    file's identity is compared now and its bytes are re-read beside the work.
    """
    if not active():
        return False
    status = path.stat()
    if (int(status.st_size), int(status.st_mtime_ns), int(status.st_dev),
            int(status.st_ino), int(status.st_mode)) != (size, mtime_ns, device, inode, mode):
        raise ValueError(f"authority changed after validation: {path}")
    defer_verify(path, status, sha256, "re-check after validation")
    return True


def join() -> None:
    """Wait for every deferred re-read so far; raise the first difference found."""
    error = None
    while True:
        with _lock:
            pending = list(_futures)
            _futures.clear()
        if not pending:
            break
        for future in pending:
            try:
                future.result()
            except BaseException as caught:  # noqa: BLE001 - re-raised below
                if error is None:
                    error = caught
    with _lock:
        for key in [key for key, future in _queued.items() if future.done()]:
            del _queued[key]
    if error is not None:
        raise error


def receipt() -> dict[str, int]:
    """How the scope's digests were taken (for logs and receipts)."""
    with _lock:
        return dict(_counts)


@contextmanager
def deferred_checks():
    """The scope in which snapshots reuse held digests and re-reads run beside the work."""
    global _depth
    with _lock:
        _depth += 1
    ok = False
    try:
        yield
        ok = True
    finally:
        try:
            if ok:
                join()
            else:
                try:
                    join()
                except BaseException:  # noqa: BLE001 - the original error wins
                    pass
        finally:
            with _lock:
                _depth -= 1
                if _depth == 0:
                    _memo.clear()
                    _queued.clear()
                    _route_manifests.clear()
                    for key in _counts:
                        _counts[key] = 0
