"""Separate filesystem I/O spelling from resolved path identity on Windows."""
from __future__ import annotations

import os
from pathlib import Path
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
