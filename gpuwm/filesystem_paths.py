"""Separate filesystem I/O spelling from resolved path identity on Windows."""
from __future__ import annotations

import errno
import os
from pathlib import Path
import sys
import time


_REPLACE_BACKOFF_SECONDS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.19)


def io_path(path: str | os.PathLike[str]) -> Path:
    """Use an absolute extended Windows path without changing POSIX paths.

    Python and native filesystem calls can otherwise disagree at MAX_PATH.
    This spelling is for I/O; use ``canonical_path`` for comparisons, keys,
    ownership checks and user-facing logical paths.
    """
    path = Path(path)
    if os.name != "nt":
        return path
    text = os.path.abspath(os.fspath(path))
    if text.startswith("\\\\?\\"):
        return Path(text)
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text[2:])
    return Path("\\\\?\\" + text)


def canonical_path(path: str | os.PathLike[str]) -> Path:
    """Resolve links, then equate ordinary and extended Windows spellings.

    Resolve errors intentionally propagate: callers retain their existing
    invalid-path and symlink-loop diagnostics instead of treating failed
    resolution as proof that two locations differ.
    """
    resolved = io_path(path).resolve()
    if os.name == "nt":
        text = os.fspath(resolved)
        if text[:8].upper() == "\\\\?\\UNC\\":
            return Path("\\\\" + text[8:])
        if (text.startswith("\\\\?\\") and len(text) >= 7
                and "a" <= text[4].lower() <= "z" and text[5:7] == ":\\"):
            return Path(text[4:])
    return resolved


def replace_file_with_retry(source: str | os.PathLike[str],
                            destination: str | os.PathLike[str]) -> Path:
    """Atomically replace after at most 0.50 s of permission-error backoff.

    Windows readers commonly hold a publication briefly without delete sharing.
    The source stays complete and the prior destination stays intact until one
    replace succeeds. Persistent permission failures and all other I/O failures
    remain visible to the caller; this helper never discards either revision.
    """
    target = Path(destination)
    source, destination = io_path(source), io_path(destination)
    for delay in (*_REPLACE_BACKOFF_SECONDS, None):
        try:
            os.replace(source, destination)
            return target
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def publish_new(source: str | os.PathLike[str],
                destination: str | os.PathLike[str]) -> None:
    """Publish the complete file ``source`` at ``destination``, never over a file.

    A same-directory hard link does this atomically and leaves ``source`` in
    place. exFAT and FAT32 have no hard links (Windows answers "Incorrect
    function" on exFAT, Linux EPERM on both), so every create-only
    publication on such a drive failed. There ``source`` is renamed
    instead, by a rename that refuses an existing destination just as the
    link does: Windows' own rename, and ``renameat2`` with RENAME_NOREPLACE
    on Linux. ``source`` is then gone, so callers treat it as spent either
    way and remove it with ``missing_ok``. ``FileExistsError`` means
    ``destination`` already existed and nothing was published.
    """
    try:
        os.link(source, destination)
        return
    except FileExistsError:
        raise
    except OSError as error:
        if not _no_hard_links(error):
            raise
        unsupported = error
    if not _rename_no_replace(source, destination):
        raise unsupported


def has_hard_links(directory: str | os.PathLike[str]) -> bool:
    """Whether the drive holding ``directory`` can hard-link a file.

    Links one new empty file to a second name in ``directory`` and removes
    both. False only for the answer ``publish_new`` falls back on (a drive
    with no hard links, such as exFAT); any other failure is raised.
    """
    import tempfile

    handle, name = tempfile.mkstemp(prefix=".gpuwm-link-probe-",
                                    dir=io_path(directory))
    os.close(handle)
    source = Path(name)
    linked = source.with_name(source.name + ".link")
    try:
        os.link(source, linked)
    except OSError as error:
        if _no_hard_links(error):
            return False
        raise
    finally:
        linked.unlink(missing_ok=True)
        source.unlink(missing_ok=True)
    return True


def _no_hard_links(error: OSError) -> bool:
    """Whether ``error`` says the filesystem has no hard links at all."""
    if os.name == "nt":
        # ERROR_INVALID_FUNCTION (exFAT) and ERROR_NOT_SUPPORTED.
        return getattr(error, "winerror", None) in (1, 50)
    return error.errno in (errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP)


def links_unavailable(error: OSError) -> bool:
    """Whether a failed ``os.link`` means these two folders cannot share a file.

    True for a drive with no hard links (exFAT, FAT32), for two folders on
    different drives, and for a file already at its link limit: in each a
    copy still works.  Anything else (a missing source, no permission to
    write the folder, a full disk) is a failure a copy would meet too, and
    stays the caller's to report.
    """
    if error.errno in (errno.EXDEV, errno.EMLINK):
        return True
    if os.name == "nt" and getattr(error, "winerror", None) in (17, 1142):
        # ERROR_NOT_SAME_DEVICE and ERROR_TOO_MANY_LINKS.
        return True
    return _no_hard_links(error)


class CopyWouldNotFitError(OSError):
    """A copy standing in for a hard link, refused before its first byte.

    The breakage it prevents: the disk filling partway through the copy,
    which leaves a half-written file and fails whatever runs next on that
    disk.  Its bound is the measured free space against the copy's size.
    """


def _free_bytes(folder: Path) -> int | None:
    probe = Path(folder)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        import shutil

        return int(shutil.disk_usage(io_path(probe)).free)
    except OSError:
        return None


def _size_text(count: int) -> str:
    for unit, scale in (("GiB", 1024 ** 3), ("MiB", 1024 ** 2),
                        ("KiB", 1024)):
        if count >= scale:
            return f"{count / scale:.1f} {unit}"
    return f"{count} bytes"


def require_room_to_copy(nbytes: int, folder: str | os.PathLike[str],
                         what: str) -> None:
    """Refuse, by name, a copy of ``nbytes`` the disk under ``folder`` cannot hold.

    Unknown free space (the probe itself failed) does not refuse: the copy
    then meets the disk's own answer, which is still a named OSError.
    """
    free = _free_bytes(Path(folder))
    if free is None or int(nbytes) <= free:
        return
    raise CopyWouldNotFitError(
        errno.ENOSPC,
        f"{what} cannot be hard-linked on this drive, so it has to be "
        f"copied, and the copy needs {_size_text(int(nbytes))} where the "
        f"disk holding {folder} has {_size_text(free)} free. Free some "
        f"disk there, or keep these folders on one drive that supports "
        f"hard links (NTFS, ext4), where the copy is a link and costs no "
        f"space")


def copy_verified(source: str | os.PathLike[str],
                  destination: str | os.PathLike[str]) -> int:
    """Copy ``source`` to a NEW ``destination`` and prove the bytes match.

    The stand-in for a hard link on a drive that has none.  It never
    replaces an existing file (``FileExistsError``), checks the disk has
    room first (:func:`require_room_to_copy`), hashes every byte it
    writes, then reads the written file back and requires the same size
    and SHA-256.  A copy that does not match is removed and raised as an
    ``OSError``, so a torn or altered copy never stands in for the
    original.  Returns the number of bytes copied.
    """
    import hashlib

    source, destination = Path(source), Path(destination)
    size = source.stat().st_size
    require_room_to_copy(size, destination.parent, str(source))
    written = hashlib.sha256()
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        try:
            for chunk in iter(lambda: incoming.read(1 << 20), b""):
                written.update(chunk)
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        except BaseException:
            outgoing.close()
            destination.unlink(missing_ok=True)
            raise
    reread = hashlib.sha256()
    with destination.open("rb") as copied:
        for chunk in iter(lambda: copied.read(1 << 20), b""):
            reread.update(chunk)
    copied_size = destination.stat().st_size
    if (copied_size != size or source.stat().st_size != size
            or reread.digest() != written.digest()):
        destination.unlink(missing_ok=True)
        raise OSError(
            errno.EIO,
            f"the copy of {source} at {destination} does not match it "
            f"({copied_size} of {size} bytes, or different bytes); nothing "
            f"was kept")
    return size


#: The one line a process prints the first time a copy stands in for a
#: hard link.  Constant, so ``warn(once=True)`` prints it once however
#: many files follow.
COPY_INSTEAD_OF_LINK_NOTE = (
    "this drive cannot hard-link files, so earlier prepared files are "
    "copied rather than linked, and each copy costs its own size in disk "
    "space")


def link_or_copy_verified(source: str | os.PathLike[str],
                          destination: str | os.PathLike[str]) -> str:
    """Give ``destination`` the bytes of ``source``: a hard link, else a verified copy.

    A link is atomic and costs no space, so it is always tried first.  A
    drive or folder pair that cannot link gets :func:`copy_verified` and
    the one-line note that the copy costs disk.  ``FileExistsError`` and
    every other link failure propagate unchanged.  Returns ``"link"`` or
    ``"copy"``.
    """
    try:
        os.link(source, destination)
        return "link"
    except FileExistsError:
        raise
    except OSError as error:
        if not links_unavailable(error):
            raise
    note_copy_instead_of_link()
    copy_verified(source, destination)
    return "copy"


def note_copy_instead_of_link() -> None:
    """Say once per process that copies are standing in for hard links."""
    from gpuwm.explain import warn

    warn(COPY_INSTEAD_OF_LINK_NOTE, once=True)


def _rename_no_replace(source, destination) -> bool:
    """Rename without replacing; False where no such rename exists here."""
    if os.name == "nt":
        # MoveFileExW without MOVEFILE_REPLACE_EXISTING is atomic and raises
        # FileExistsError for an existing destination, on every volume.
        os.rename(source, destination)
        return True
    if not sys.platform.startswith("linux"):
        return False
    import ctypes

    try:
        rename = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError):
        return False
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                       ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    at_cwd, no_replace = -100, 1
    if rename(at_cwd, os.fsencode(source), at_cwd, os.fsencode(destination),
              no_replace) == 0:
        return True
    code = ctypes.get_errno()
    if code in (errno.EINVAL, errno.ENOSYS):
        return False
    # OSError maps EEXIST to FileExistsError, as os.link raises it.
    raise OSError(code, os.strerror(code), os.fspath(source), None,
                  os.fspath(destination))


#: Windows' classic full-path ceiling.  A path at or past it fails with
#: ERROR_PATH_NOT_FOUND, which Python reports as ``FileNotFoundError``,
#: unless every call that touches it is long-path aware.
WINDOWS_MAX_PATH = 260

#: Characters a forecast chain adds below the root it is handed, measured
#: on a GFS `gpuwm go` run.  A chain that downloads writes deepest in its
#: request cache, 142 for a staging file
#: (``downloads/<64-hex key>/<object>.<pid>-<ns>.part``); one that does
#: not writes at most 81 in its run tree
#: (``run-<stamp>/run/ready/<wrfout name>.json``).  Both carry margin.  The
#: picture tree places itself: :mod:`gpuwm.render_layout` measures its
#: own paths against the limit.
CHAIN_DEPTH_BUDGET = 150
RUN_TREE_DEPTH_BUDGET = 100

#: Characters a download adds below its output folder: the longest
#: object name a route writes (``gfs.t00z.pgrb2.0p25.f003.subset.grib2``,
#: 37) plus a ``.<pid>-<ns>.part`` staging suffix (up to 32), with margin.
DOWNLOAD_DEPTH_BUDGET = 80


def deep_io_path(path: str | os.PathLike[str], depth: int) -> Path:
    """``path``, or its extended spelling when ``depth`` more characters would not fit.

    For a caller about to write up to ``depth`` characters below
    ``path``: on Windows, when that would pass :data:`WINDOWS_MAX_PATH`,
    the extended spelling is returned, which Python and native calls open
    at any length.  Otherwise ``path`` comes back unchanged, so the paths
    a command prints stay the ones its reader typed.
    """
    if os.name != "nt":
        return Path(path)
    absolute = os.path.abspath(os.fspath(path))
    if is_extended(absolute) or len(absolute) + depth < WINDOWS_MAX_PATH:
        return Path(path)
    return io_path(absolute)


def is_extended(path: str | os.PathLike[str]) -> bool:
    """Whether ``path`` is written in the Windows extended spelling."""
    return os.name == "nt" and os.fspath(path).startswith("\\\\?\\")


def keep_spelling(given: str | os.PathLike[str], resolved: str | os.PathLike[str]) -> Path:
    """``resolved`` in the spelling its caller chose for ``given``.

    :func:`canonical_path` answers in the plain spelling, which is the
    identity receipts record.  A caller that handed over an extended
    spelling did so because the files it will write below that folder
    pass the Windows path limit; answering it in the plain spelling
    would have it write through a path that fails as a missing file.
    """
    return io_path(resolved) if is_extended(given) else Path(resolved)
