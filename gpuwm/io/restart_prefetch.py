"""Owned host checkpoint payloads, prefetched beside other preparation work.

Only archive I/O is cached. Every restore still runs the native complete-set,
physics, setup, inventory and forcing validation against its fresh model.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import copy
import hashlib

_ACTIVE = ContextVar('gpuwm_restart_payloads', default={})


def _identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@dataclass(frozen=True)
class RestartPayload:
    path: Path
    identity: tuple
    header: dict
    arrays: dict
    sha256: str


def prefetch_restart(path):
    """Read once on a host worker; immutable arrays stay owned by the set."""
    from gpuwm.io.restart import _load_restart, RestartMismatchError
    path = Path(path).resolve()
    before = _identity(path)
    header, arrays = _load_restart(path, with_arrays=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if _identity(path) != before:
        raise RestartMismatchError(f'{path}: checkpoint changed while being prefetched')
    for value in arrays.values():
        value.flags.writeable = False
    return RestartPayload(path, before, header, arrays, digest)


@contextmanager
def restart_payload_scope(payloads):
    token = _ACTIVE.set({payload.path: payload for payload in payloads})
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def cached_restart(path, *, with_arrays):
    from gpuwm.io.restart import RestartMismatchError
    path = Path(path).resolve()
    payload = _ACTIVE.get().get(path)
    if payload is None:
        return None
    if _identity(path) != payload.identity:
        raise RestartMismatchError(f'{path}: checkpoint changed after prefetch; start a fresh ensemble set')
    # A validator may annotate metadata; no member may mutate another's input.
    return copy.deepcopy(payload.header), dict(payload.arrays) if with_arrays else None


@contextmanager
def restart_archive(path):
    """Manifest reader over prefetched arrays, or the normal lazy archive."""
    cached = cached_restart(path, with_arrays=True)
    if cached is None:
        import numpy as np
        with np.load(path, allow_pickle=False) as archive:
            yield archive
    else:
        class Archive:
            files = tuple(cached[1])
            def __getitem__(self, name):
                return cached[1][name]
        yield Archive()
