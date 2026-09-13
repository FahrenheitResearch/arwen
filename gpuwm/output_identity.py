"""Bind an output receipt to one stable file and its literal address."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from contextlib import nullcontext


@dataclass(frozen=True)
class FileRevision:
    """A descriptor's file identity and observed content-change metadata."""

    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int
    stat_ctime_ns: int
    reusable: bool

    @property
    def file_id(self):
        return self.device, self.inode


@dataclass(frozen=True)
class CompletedFileRecord:
    """Process-local proof from a completed write, never loaded from JSON."""

    path: str
    size: int
    sha256: str
    revision: FileRevision
    address: str

    def record(self):
        return {"path": self.path, "bytes": self.size, "sha256": self.sha256}


@lru_cache(maxsize=1)
def _windows_basic_info():
    import ctypes
    from ctypes import wintypes

    class BasicInfo(ctypes.Structure):
        _fields_ = [("creation", ctypes.c_longlong),
                    ("access", ctypes.c_longlong),
                    ("written", ctypes.c_longlong),
                    ("changed", ctypes.c_longlong),
                    ("attributes", wintypes.DWORD)]

    function = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandleEx
    function.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                         wintypes.DWORD)
    function.restype = wintypes.BOOL
    return function, BasicInfo


def _revision(handle) -> FileRevision:
    info = os.fstat(handle.fileno())
    changed = info.st_ctime_ns
    reusable = bool(info.st_ino)
    if os.name == "nt":
        # st_ctime is creation time on Windows. FILE_BASIC_INFO.ChangeTime
        # observes metadata changes, including restoration of LastWriteTime.
        # Its clock can give a rapid content edit the same token as completion,
        # even after both write handles close. It still detects many changes,
        # but cannot authorize reusing a digest without reading current bytes.
        reusable = False
        import ctypes
        import msvcrt
        try:
            function, record_type = _windows_basic_info()
        except (OSError, AttributeError):
            reusable = False
        else:
            record = record_type()
            if function(msvcrt.get_osfhandle(handle.fileno()), 0,
                        ctypes.byref(record), ctypes.sizeof(record)):
                changed = int(record.changed) * 100
                reusable = reusable and changed > 0
            else:
                reusable = False
    return FileRevision(info.st_dev, info.st_ino, info.st_size,
                        info.st_mtime_ns, changed, info.st_ctime_ns, reusable)


def _address_matches(path, revision):
    try:
        if os.name == "nt":
            # Python's path stat may report creation time as st_ctime while
            # descriptor stat reports change time. Compare two descriptors
            # through the same owner instead of mixing those clocks.
            with path.open("rb") as addressed:
                return _revision(addressed) == revision
        info = path.stat()
    except OSError as exc:
        raise RuntimeError(
            f"Output {path} disappeared while its receipt was hashed. "
            "Keep the completed output stable and retry finalization.") from exc
    return ((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
             info.st_ctime_ns)
            == (revision.device, revision.inode, revision.size,
                revision.modified_ns, revision.stat_ctime_ns))


def _changed(path):
    return RuntimeError(
        f"Output {path} changed while its receipt was hashed or after "
        "writer completion. Keep the completed output stable and retry finalization.")


def open_publication_file(path):
    """Retain the written file across rename, including Windows sharing."""
    if os.name != "nt":
        return Path(path).open("rb")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                       wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    # GENERIC_READ; share read/write/delete; OPEN_EXISTING; normal file.
    # Python's ordinary rb handle does not promise FILE_SHARE_DELETE.
    raw = create(str(Path(path).absolute()), 0x80000000, 7, None, 3, 0x80, None)
    if raw == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(raw, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close(raw)
        raise
    try:
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


class PublicationFile:
    """Own the validated file until its publication and digest complete."""

    def __init__(self, path):
        self.path = Path(path)
        self.handle = open_publication_file(path)
        try:
            self.written = _revision(self.handle)
            self.check_validated()
        except BaseException:
            self.close()
            raise

    def check_validated(self):
        if (_revision(self.handle) != self.written
                or not _address_matches(self.path, self.written)):
            raise _changed(self.path)

    def published(self, path):
        current = _revision(self.handle)
        # A rename legitimately changes ctime/ChangeTime. It cannot change
        # the retained file identity, length, or last content-write time.
        if ((current.file_id, current.size, current.modified_ns)
                != (self.written.file_id, self.written.size,
                    self.written.modified_ns)):
            raise _changed(path)
        return publication_revision(path, expected=current)

    def close(self):
        self.handle.close()


def publication_revision(path, *, expected=None) -> FileRevision:
    """Capture the published writer revision without reading its payload."""
    path = Path(path).resolve(strict=True)
    with path.open("rb") as handle:
        revision = _revision(handle)
        if (expected is not None and revision != expected
                or not _address_matches(path, revision)
                or _revision(handle) != revision):
            raise _changed(path)
    return revision


def completed_file_record(path, *, published=None, cancel_event=None,
                          handle=None) -> CompletedFileRecord:
    """Hash a closed output on its completion thread, bound to its writer."""
    address = os.path.abspath(path)
    path = Path(path).resolve(strict=True)
    with (path.open("rb") if handle is None else nullcontext(handle)) as stream:
        before = _revision(stream)
        if published is not None and before != published:
            raise _changed(path)
        return _hash_open_file(path, stream, before, address=address,
                               cancel_event=cancel_event)


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Output identity capture was cancelled before completion.")


def _hash_open_file(path, handle, before, *, address=None, cancel_event=None):
    digest = hashlib.sha256()
    count = 0
    _check_cancel(cancel_event)
    for block in iter(lambda: handle.read(1 << 20), b""):
        _check_cancel(cancel_event)
        count += len(block)
        digest.update(block)
    _check_cancel(cancel_event)
    after = _revision(handle)
    if (before != after or not _address_matches(path, after)
            or count != after.size):
        raise _changed(path)
    return CompletedFileRecord(str(path), count, digest.hexdigest(), after,
                               str(path) if address is None else address)


def file_record(path, *, completed=None) -> dict[str, object]:
    """Verify a fresh writer record, or hash a stable legacy output.

    Reuse observes the full file revision again, not size/mtime alone. A
    changed revision requires reading and comparing the exact SHA-256; an
    address replacement cannot inherit the original writer's record. The
    observation ends at the descriptor/address checks in this call, not at
    an arbitrary later external mutation.
    """
    path = Path(path).resolve(strict=True)
    with path.open("rb") as handle:
        before = _revision(handle)
        if completed is None:
            return _hash_open_file(path, handle, before).record()
        if not isinstance(completed, CompletedFileRecord):
            raise TypeError("completed output proof must come from the writer")
        if completed.path != str(path) or completed.revision.file_id != before.file_id:
            raise _changed(path)
        if before.reusable and before == completed.revision:
            if not _address_matches(path, before) or _revision(handle) != before:
                raise _changed(path)
            return completed.record()
        # Attribute-only changes need not invalidate unchanged bytes. Read
        # the same descriptor and require the original digest to survive.
        observed = _hash_open_file(path, handle, before)
        if observed.size != completed.size or observed.sha256 != completed.sha256:
            raise _changed(path)
        return observed.record()


def file_records(paths, *, completed=(), before_record=None):
    """One receipt owner for every ordinary and prepared output inventory."""
    by_path = {}
    for proof in completed:
        if not isinstance(proof, CompletedFileRecord):
            raise TypeError("completed output proof must come from the writer")
        for address in (proof.path, proof.address):
            if address in by_path and proof != by_path[address]:
                raise RuntimeError(f"Two writer revisions claim output {address}")
            by_path[address] = proof
    records = []
    for index, path in enumerate(paths, start=1):
        if before_record is not None:
            before_record(index, path)
        key = os.path.abspath(path)
        records.append(file_record(path, completed=by_path.get(key)))
    return records
