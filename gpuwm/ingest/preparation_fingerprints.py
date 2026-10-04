"""Native exact receipt hashes over immutable contiguous host buffers."""
from __future__ import annotations

import ctypes
import numpy as np


class _Input(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("data_bytes", ctypes.c_size_t),
                ("length", ctypes.c_size_t), ("kind", ctypes.c_uint32),
                ("metadata", ctypes.c_uint32)]


class _Result(ctypes.Structure):
    _fields_ = [("digest", ctypes.c_ubyte * 32), ("nonzero_mask_digest", ctypes.c_ubyte * 32),
                ("nonzero_count", ctypes.c_uint64), ("minimum", ctypes.c_double),
                ("maximum", ctypes.c_double), ("numpy_extrema", ctypes.c_uint32)]


_KINDS = {np.dtype(name): code for code, name in enumerate(
    ("bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
     "int64", "uint64", "float32", "float64"), start=1)}


def _native_entry():
    from gpuwm.core import portable_math as pm
    try:
        library = pm._load()
    except FileNotFoundError:
        return None
    entry = getattr(library, "gpuwm_preparation_fingerprints", None)
    if entry is not None:
        entry.argtypes = [ctypes.POINTER(_Input), ctypes.c_size_t, ctypes.c_size_t,
                          ctypes.POINTER(_Result)]
        entry.restype = ctypes.c_int32
    return entry


def _run(arrays, *, metadata, workers):
    entry = _native_entry()
    if entry is None:
        return None
    jobs = (_Input * len(arrays))()
    for index, array in enumerate(arrays):
        if not isinstance(array, np.ndarray) or not array.flags.c_contiguous or array.dtype.hasobject:
            return None
        kind = _KINDS.get(array.dtype)
        if metadata and (kind is None or not array.dtype.isnative or not array.size):
            return None
        jobs[index] = _Input(array.ctypes.data, array.nbytes, array.size, kind or 0, int(metadata))
    results = (_Result * len(arrays))()
    code = entry(jobs, len(arrays), workers, results)
    if code:
        raise RuntimeError(f"native preparation fingerprint failed with code {code}")
    return results


def array_sha256(array):
    """Exact C-order payload hash, or None for an older bridge/layout.

    This does not copy a contiguous array or build an intermediate bytes object.
    The caller must keep it immutable until the function returns.
    """
    result = _run((array,), metadata=False, workers=1)
    return None if result is None else bytes(result[0].digest).hex()


def array_fingerprints(arrays, *, workers=None):
    """Full correspondence receipts in order, or None for unsupported input.

    Independent arrays run on separate native workers. The SHA-256 state of
    any one array remains a single ordered stream. Scratch is 8 KiB per worker.
    """
    arrays = tuple(arrays)
    if workers is None:
        from gpuwm.core import portable_math as pm
        workers = pm._workers(None)
    if not arrays:
        return []
    result = _run(arrays, metadata=True, workers=int(workers))
    if result is None:
        return None
    receipts = []
    for array, row in zip(arrays, result):
        # NumPy's SIMD reduction order owns NaN payload and signed-zero ties.
        # Finite nonzero extrema have one representation, so native reductions
        # are exact there regardless of worker scheduling.
        minimum = float(np.min(array)) if row.numpy_extrema & 1 else row.minimum
        maximum = float(np.max(array)) if row.numpy_extrema & 2 else row.maximum
        receipts.append({"shape": list(array.shape), "dtype": str(array.dtype),
                         "sha256": bytes(row.digest).hex(),
                         "nonzero_mask_sha256": bytes(row.nonzero_mask_digest).hex(),
                         "nonzero_count": int(row.nonzero_count),
                         "minimum": minimum, "maximum": maximum})
    return receipts


def array_fingerprint(array):
    """One full correspondence receipt, or None for the existing fallback."""
    result = array_fingerprints((array,), workers=1)
    return None if result is None else result[0]
