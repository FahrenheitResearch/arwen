"""Owned raw-frame cache with OS leases shared by the native Rust readers.

The lock inode is permanent. Readers retain a shared lock for every clone's
lifetime; eviction and integrity recovery require its exclusive lock.
"""
from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import re
import stat
import time

SHA = re.compile(r"[0-9a-f]{64}\Z")
OBJECT = re.compile(r"([0-9a-f]{64})\.wrf\Z")
PART = re.compile(r"\.[0-9a-f]{64}\.wrf\.[0-9a-f]{16}\.part\Z")
MAX_OBJECTS = 4096


class CacheRecovery(ValueError):
    def __init__(self, digest):
        self.digest = digest
        super().__init__(f"Cached committed frame {digest} failed integrity while retained by a reader; release that frame and retry synchronization")


def _owned_directory(path):
    if not path.is_absolute() or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("Artifact cache requires an absolute owned directory without a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or (hasattr(os, "getuid") and info.st_uid != os.getuid()):
        raise ValueError("Artifact cache directory is not owned by this account")
    return path.resolve(strict=True)


class Lease:
    """One permanent lock inode, interoperable with Rust File::lock_shared."""
    def __init__(self, path, *, timeout=0):
        self.path, self.file = path, None
        if path.is_symlink():
            raise ValueError("Artifact lease must not be a symlink")
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        file = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            info = os.fstat(file.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 1
                    or (hasattr(os, "getuid") and info.st_uid != os.getuid())):
                raise ValueError("Artifact lease is not one owned lock inode")
            deadline = time.monotonic() + timeout
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt
                        file.seek(0)
                        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if info.st_size == 0:
                        file.seek(0)
                        file.write(b"\0")
                        file.flush()
                    self.file = file
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if time.monotonic() >= deadline:
                        file.close()
                        break
                    time.sleep(.025)
        except BaseException:
            file.close()
            raise

    def close(self):
        if self.file is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    self.file.seek(0)
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            finally:
                self.file.close()
                self.file = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class Cache:
    def __init__(self, root, *, target_bytes=2 * 1024**3, max_frame_bytes=512 * 1024**2, reader_leases=True):
        self.root = _owned_directory(Path(root))
        self.objects = _owned_directory(self.root / "objects")
        self.leases = _owned_directory(self.root / "leases")
        self.target, self.headroom = target_bytes, max_frame_bytes
        self.reader_leases = reader_leases
        self.writer = None
        self.evicted = 0

    def __enter__(self):
        self.writer = Lease(self.root / ".writer.lock", timeout=10)
        if self.writer.file is None:
            raise ValueError("A committed frame cache update is still in progress; refresh to retry")
        # The writer lock proves no live download owns a leftover partial.
        try:
            for count, path in enumerate(self.objects.iterdir()):
                if count >= MAX_OBJECTS or path.is_symlink() or not path.is_file():
                    raise ValueError("Committed frame cache has unexpected or excessive objects")
                if PART.fullmatch(path.name):
                    path.unlink()
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_args):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def _entries(self):
        if self.writer is None or self.writer.file is None:
            raise ValueError("Cache pruning requires its exclusive writer transaction")
        result = []
        for count, path in enumerate(self.objects.iterdir()):
            match = OBJECT.fullmatch(path.name)
            if count >= MAX_OBJECTS or not match or path.is_symlink() or not path.is_file():
                raise ValueError("Committed frame cache has unexpected or excessive objects")
            lock = self.leases / (match[1] + ".lock")
            if lock.is_symlink():
                raise ValueError("Committed frame lease must not be a symlink")
            used_at = (lock if lock.exists() else path).stat().st_mtime_ns
            result.append((used_at, path, match[1], path.stat().st_size))
        return sorted(result)

    def prune(self, *, incoming=0, protected=None):
        entries = self._entries()
        used = sum(row[3] for row in entries)
        if not self.reader_leases:
            if used + incoming > self.target:
                raise ValueError("Update the visual workspace to enable leased cache rotation before retrieving more frames")
            return used
        for _age, path, digest, _size in entries:
            if used + incoming <= self.target:
                break
            if digest == protected:
                continue
            with Lease(self.leases / (digest + ".lock")) as lease:
                if lease.file is None:
                    continue
                # No active reader can start until after this unlink. A reader
                # takes the shared lock first, then rechecks existence/hash.
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Committed frame cache changed during eviction")
                size = path.stat().st_size
                path.unlink()
                used -= size
                self.evicted += size
        if used + incoming > self.target + self.headroom:
            raise ValueError("Active frame readers exceed the bounded cache replacement allowance; release obsolete frames and retry")
        return used

    def obtain(self, frame, download):
        if self.writer is None or self.writer.file is None:
            raise ValueError("Cache publication requires its exclusive writer transaction")
        digest, size = frame["sha256"], frame["size_bytes"]
        if not SHA.fullmatch(str(digest)) or type(size) is not int or not 0 < size <= self.headroom:
            raise ValueError("Committed frame has an invalid cache size or SHA-256")
        path = self.objects / (digest + ".wrf")
        lock_path = self.leases / (digest + ".lock")
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("Committed frame object must not be a symlink")
        transferred = 0
        if path.exists() and (path.stat().st_size != size or _hash(path) != digest):
            if not self.reader_leases:
                raise ValueError("Update the visual workspace to enable safe repair of this changed cached frame")
            with Lease(lock_path) as lease:
                if lease.file is None:
                    raise CacheRecovery(digest)
                path.unlink()
        if not path.exists():
            self.prune(incoming=size, protected=digest)
            with Lease(lock_path, timeout=10) as lease:
                if lease.file is None:
                    raise ValueError("A native reader is refreshing this frame; retry synchronization")
                download(path)
                if path.is_symlink() or path.stat().st_size != size or _hash(path) != digest:
                    path.unlink(missing_ok=True)
                    raise ValueError("Downloaded frame did not satisfy its immutable cache identity")
            transferred = size
        # Update access on the lock inode, never the immutable raw WRF file:
        # native readers bind the object's original file metadata as well.
        if not lock_path.exists():
            with Lease(lock_path):
                pass
        os.utime(lock_path, None)
        used = self.prune(protected=digest)
        return path, transferred, {"schema": "arwen.artifact-cache.v1", "used_bytes": used,
                                   "target_bytes": self.target, "replacement_headroom_bytes": self.headroom,
                                   "evicted_bytes": self.evicted}
