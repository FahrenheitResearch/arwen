"""One honest sentence before the first run's invisible kernel compile.

The first GPU forecast on a machine pays on the order of two minutes of
NVRTC compilation -- every physics module's ``cupy.RawModule`` and the
direct-NVRTC radiation translation units compile for the local card
before a single model step runs.  CuPy writes the compiled binaries to
its on-disk kernel cache, so every later run loads them in seconds; but
on the run that pays, nothing said so.  The first field run of the
published wheel watched the forecast phase sit at a stale status for a
hundred seconds and reasonably wondered whether it had hung.

This module is the one place that knows how to ask "is that about to
happen?": the CuPy kernel cache directory (``CUPY_CACHE_DIR`` when set,
else ``~/.cupy/kernel_cache`` -- CuPy's own documented resolution) with
no compiled entries in it means every kernel this run needs is about to
be built from source.  The check is a heuristic in one direction only:
a warm cache can still recompile (a gpuwm upgrade changes kernel
sources, a new card changes the architecture), but a cold cache never
loads anything, so the notice never fires on the runs that are fast.

No CUDA import happens here; the question is answered from the
filesystem alone, so callers may ask it before touching the device.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Environment variable CuPy consults for its kernel cache location.
CUPY_CACHE_ENV = "CUPY_CACHE_DIR"

#: progress.json status published while the first-run compile is likely
#: under way; ``gpuwm go`` relays it verbatim in its heartbeat lines.
COMPILING_STATUS = "COMPILING_GPU_KERNELS"


def cupy_kernel_cache_dir() -> Path:
    """Where CuPy keeps compiled kernels, by CuPy's own resolution order."""

    override = os.environ.get(CUPY_CACHE_ENV, "").strip()
    if override:
        return Path(override)
    return Path.home() / ".cupy" / "kernel_cache"


def kernel_cache_is_cold(cache_dir: Path | None = None) -> bool:
    """True when the CuPy kernel cache holds no compiled entries at all.

    A missing directory, an unreadable one, and an existing-but-empty
    one all count as cold: in every one of those states the coming run
    compiles from source.  Any regular file in the directory counts as
    warmth -- the entries are content-hashed ``.cubin`` blobs and this
    predicate has no business knowing their naming scheme.
    """

    directory = cupy_kernel_cache_dir() if cache_dir is None else cache_dir
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    return False
    except OSError:
        return True
    return True


def kernel_compile_notice(cache_dir: Path | None = None) -> str | None:
    """The one line to print before first-run compilation, or ``None``.

    ``None`` means the cache is warm and saying anything would be noise.
    """

    if not kernel_cache_is_cold(cache_dir):
        return None
    return (
        "forecast: first run on this machine -- compiling GPU kernels "
        "for the local card (typically 1-3 minutes with no visible "
        "progress; the compiled kernels are cached, so later runs skip "
        "this)")


__all__ = [
    "COMPILING_STATUS", "CUPY_CACHE_ENV", "cupy_kernel_cache_dir",
    "kernel_cache_is_cold", "kernel_compile_notice",
]
