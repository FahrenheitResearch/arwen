"""Native wrapped-cost surfaces for the dealiaser's coarse VAD search.

:func:`gpuwm.obs.dealias.vad_reference` seeds its fold-aware harmonic fit
from an exhaustive search over a ``speed x direction`` grid of environmental
winds.  That search is the dominant cost of the whole observation front door
-- 1.13 million cosines per range band, twelve hundred bands in a volume --
and it is the one part of the dealiaser with no sequential dependence: every
band's search stands alone, so the whole sweep's worth can go to the parallel
Rust CPU backend in one call.

This module is the thin ctypes seam onto ``gpuwm_dealias_coarse_cost_f64``.
It deliberately reuses :func:`gpuwm.ingest.cpu_backend.resolve_cpu_bridge`
rather than growing a second search order: there is one native preprocessing
library and it is found one way.

What the seam promises, and what it does not
--------------------------------------------
It returns a cost surface accurate to about 1e-11 absolute, not one that is
bit-identical to NumPy's.  The kernel rides a rotation recurrence along the
speed axis, which is a different summation of a different set of roundings.
So the caller must not rank candidates with these numbers alone -- and
:mod:`gpuwm.obs.dealias` does not.  It shortlists with them, proves the
shortlist is sufficient, and ranks the shortlist with the same NumPy
expression the pure-Python path uses.  See ``_coarse_shortlisted`` there.

``GPUWM_DEALIAS_NATIVE=0`` forces the pure NumPy path.  That is a diagnostic
and a workaround, not a supported mode: the native path is the default
wherever the library is present, and the parity test drives both.
"""

from __future__ import annotations

import ctypes
import os
from typing import Final

import numpy as np

#: The coarse-cost entry point's own version, independent of the
#: interpolation ABI so that adding it does not invalidate preparation
#: receipts that quote ``CPU_BACKEND_ABI``.
DEALIAS_COST_ABI: Final[int] = 1

#: Set to ``0`` to refuse the native path.  A workaround, and reported as
#: one: the counted stat says which path a sweep actually took.
NATIVE_ENV: Final[str] = "GPUWM_DEALIAS_NATIVE"

_ERRORS = {
    1: "null buffer",
    2: "invalid dimensions",
    3: "non-finite input",
    127: "native coarse-cost kernel panicked",
}

_LIBRARY: ctypes.CDLL | None = None
_UNAVAILABLE: str | None = None


def unavailable_reason() -> str | None:
    """Why the native path is not in use, or None when it is."""

    _load()
    return _UNAVAILABLE


def _load() -> ctypes.CDLL | None:
    global _LIBRARY, _UNAVAILABLE
    if _LIBRARY is not None or _UNAVAILABLE is not None:
        return _LIBRARY
    if os.environ.get(NATIVE_ENV, "1") == "0":
        _UNAVAILABLE = f"{NATIVE_ENV}=0"
        return None
    try:
        from gpuwm.ingest.cpu_backend import resolve_cpu_bridge

        path = resolve_cpu_bridge()
        library = ctypes.CDLL(str(path))
        library.gpuwm_dealias_cost_abi_version.argtypes = []
        library.gpuwm_dealias_cost_abi_version.restype = ctypes.c_uint32
        observed = int(library.gpuwm_dealias_cost_abi_version())
        if observed != DEALIAS_COST_ABI:
            _UNAVAILABLE = (f"coarse-cost ABI {observed} != required "
                            f"{DEALIAS_COST_ABI}")
            return None
        pointer = ctypes.c_void_p
        size = ctypes.c_size_t
        library.gpuwm_dealias_coarse_cost_f64.argtypes = [
            pointer, pointer, pointer, pointer, size,
            pointer, size, pointer, size,
            ctypes.c_double, pointer, size,
        ]
        library.gpuwm_dealias_coarse_cost_f64.restype = ctypes.c_int32
    except (OSError, AttributeError, FileNotFoundError, RuntimeError) as error:
        # A library that predates this entry point has no such symbol, which
        # is a fact about the build and not a failure of this volume.
        _UNAVAILABLE = f"{type(error).__name__}: {error}"
        return None
    _LIBRARY = library
    return _LIBRARY


def _workers(value: int | None, units: int) -> int:
    if value is None:
        value = os.cpu_count() or 1
    value = int(value)
    if value < 1:
        raise ValueError("workers must be positive")
    return max(1, min(value, units))


def coarse_cost_batch(bands, speeds, directions, nyquist: float, *,
                      workers: int | None = None) -> np.ndarray | None:
    """Wrapped cost over the wind grid for every band of one sweep.

    ``bands`` is a sequence of ``(values, sin_az, cos_az)`` triples, one per
    range band, each already reduced to the samples the search looks at.
    Returns ``(len(bands), len(speeds) * len(directions))`` in the raveled
    order of ``speed[:, None]`` against ``direction[None, :]``, or None when
    no native library is available -- in which case the caller owns its own
    arithmetic and nothing has been decided here.
    """

    library = _load()
    if library is None or not bands:
        return None
    speeds = np.ascontiguousarray(speeds, dtype=np.float64)
    directions = np.ascontiguousarray(directions, dtype=np.float64)
    counts = [int(np.asarray(band[0]).size) for band in bands]
    total = sum(counts)
    if total == 0 or speeds.size == 0 or directions.size == 0:
        return None
    values = np.empty(total, dtype=np.float64)
    sin_az = np.empty(total, dtype=np.float64)
    cos_az = np.empty(total, dtype=np.float64)
    offsets = np.zeros(len(bands) + 1, dtype=np.uintp)
    cursor = 0
    for index, (band_values, band_sin, band_cos) in enumerate(bands):
        stop = cursor + counts[index]
        values[cursor:stop] = band_values
        sin_az[cursor:stop] = band_sin
        cos_az[cursor:stop] = band_cos
        offsets[index + 1] = stop
        cursor = stop
    cost = np.empty((len(bands), speeds.size * directions.size),
                    dtype=np.float64)
    code = int(library.gpuwm_dealias_coarse_cost_f64(
        ctypes.c_void_p(values.ctypes.data),
        ctypes.c_void_p(sin_az.ctypes.data),
        ctypes.c_void_p(cos_az.ctypes.data),
        ctypes.c_void_p(offsets.ctypes.data),
        len(bands),
        ctypes.c_void_p(speeds.ctypes.data),
        speeds.size,
        ctypes.c_void_p(directions.ctypes.data),
        directions.size,
        float(nyquist),
        ctypes.c_void_p(cost.ctypes.data),
        _workers(workers, len(bands) * directions.size),
    ))
    if code:
        detail = _ERRORS.get(code, f"unknown native error {code}")
        raise ValueError(f"native coarse VAD cost failed: {detail}")
    return cost


__all__ = [
    "DEALIAS_COST_ABI",
    "NATIVE_ENV",
    "coarse_cost_batch",
    "unavailable_reason",
]
