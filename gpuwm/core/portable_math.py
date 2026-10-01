"""Portable elementwise transcendentals for the host preparation.

NumPy's ``exp``, ``log``, ``log1p``, ``power``, ``sin``, ``cos``, ``tan``,
``arcsin``, ``arctan`` and ``arctan2`` take the C library on most hosts and
their own vector loops on an AVX-512 Linux host, and the C library itself
differs between glibc releases and between glibc and the MSVC runtime.  A
prepared state, receipt hash or pin built from them therefore changes in
its last bits with the machine that prepared it.

These take the CPU preprocessing library ``tools/grib1_bridge`` builds
(``gpuwm_preprocess_cpu``, the one every preparation already loads), whose
``gpuwm_portable_*`` entries run the vendored ``libm`` crate: a pure Rust
port of musl's libm, the same instructions on every x86-64 host whatever
its vector unit or C library.  The work is split over fixed contiguous
ranges, so the thread count never changes an element.

Where that library, or a library new enough to hold the entries, is
absent, every function here falls back to :mod:`math` one element at a
time (the host C library, and a Python call per element) and says so once
per process on stderr.  That fallback is a workaround: its answers are the
host's, not the portable ones.

float32 operands (a float32 array, alone or with a Python scalar) take the
single-precision entries (``expf``, ``powf`` and so on); anything else is
computed in float64.  The answers are the libm crate's, not glibc's, so
the transcriptions that must equal a Fortran oracle's glibc calls stay in
:mod:`gpuwm.core.noahmp_libm` and :mod:`gpuwm.core.host_libm`.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Final

import numpy as np

__all__ = ["IMPLEMENTATION", "FALLBACK_IMPLEMENTATION", "implementation",
           "worker_limit", "exp", "log", "log1p", "log10", "sin", "cos",
           "tan", "arcsin", "arccos", "arctan", "arctan2", "power"]

#: The implementation name a receipt records for the library route.
IMPLEMENTATION: Final[str] = "rust-libm-0.2.16-v1"
#: The implementation name of the element-by-element fallback.
FALLBACK_IMPLEMENTATION: Final[str] = "python-math-host-libc"
#: The generation of the library entries this module was written against.
PORTABLE_MATH_VERSION: Final[int] = 1

_UNARY_CODES: Final[dict[str, int]] = {
    "exp": 0, "log": 1, "log1p": 2, "sin": 3, "cos": 4, "tan": 5,
    "asin": 6, "atan": 7, "log10": 8, "acos": 9,
}
_BINARY_CODES: Final[dict[str, int]] = {"pow": 0, "atan2": 1}
_ENTRIES: Final[tuple[str, ...]] = (
    "gpuwm_portable_unary_f64", "gpuwm_portable_unary_f32",
    "gpuwm_portable_binary_f64", "gpuwm_portable_binary_f32")

_lock = threading.Lock()
_worker_limit: ContextVar[int | None] = ContextVar(
    "gpuwm_portable_math_worker_limit", default=None)
_library = None
_resolved = False
_absent_reason: str | None = None
_warned = False


def _load():
    """The configured library, or None with the reason kept for the warning."""
    global _library, _resolved, _absent_reason
    if _resolved:
        return _library
    with _lock:
        if _resolved:
            return _library
        library = None
        from gpuwm.ingest.cpu_backend import CPU_BRIDGE_ENV, resolve_cpu_bridge

        try:
            path = resolve_cpu_bridge()
            candidate = ctypes.CDLL(str(path))
        except OSError as error:
            # An explicit GPUWM_CPU_PREPROCESS_BRIDGE that names a missing
            # or unloadable file is the user's configuration and fails
            # loudly, as every other consumer of that variable does.
            if os.environ.get(CPU_BRIDGE_ENV):
                raise
            _absent_reason = (
                "the CPU preprocessing library is not available ("
                + str(error).splitlines()[0] + ")")
        else:
            missing = [name for name in
                       _ENTRIES + ("gpuwm_portable_math_version",)
                       if not hasattr(candidate, name)]
            if missing:
                _absent_reason = (
                    f"the CPU preprocessing library at {path} predates the "
                    "portable math entries (" + ", ".join(missing) + ")")
            else:
                candidate.gpuwm_portable_math_version.argtypes = []
                candidate.gpuwm_portable_math_version.restype = ctypes.c_uint32
                version = int(candidate.gpuwm_portable_math_version())
                if version != PORTABLE_MATH_VERSION:
                    _absent_reason = (
                        f"the CPU preprocessing library at {path} carries "
                        f"portable math generation {version}, this package "
                        f"reads {PORTABLE_MATH_VERSION}")
                else:
                    pointer, size = ctypes.c_void_p, ctypes.c_size_t
                    code = ctypes.c_uint32
                    for name in ("gpuwm_portable_unary_f64",
                                 "gpuwm_portable_unary_f32"):
                        entry = getattr(candidate, name)
                        entry.argtypes = [code, pointer, pointer, size, size]
                        entry.restype = ctypes.c_int32
                    for name in ("gpuwm_portable_binary_f64",
                                 "gpuwm_portable_binary_f32"):
                        entry = getattr(candidate, name)
                        entry.argtypes = [code, pointer, size, pointer, size,
                                          pointer, size, size]
                        entry.restype = ctypes.c_int32
                    library = candidate
        _library = library
        _resolved = True
        return _library


def implementation() -> str:
    """The implementation the functions here run on in this process."""
    return IMPLEMENTATION if _load() is not None else FALLBACK_IMPLEMENTATION


def _warn_fallback() -> None:
    global _warned
    if _warned:
        return
    _warned = True
    sys.stderr.write(
        "[portable-math] WORKAROUND: " + str(_absent_reason) + "; the host "
        "preparation's exp, log, pow and trigonometric functions take the "
        "C library through math.* one element at a time, so the prepared "
        "state's last bits follow this host.  Build tools/grib1_bridge or "
        "run `gpuwm fetch-bridges` for the portable ones.\n")


def _positive_workers(workers) -> int:
    if isinstance(workers, (bool, np.bool_)) or not isinstance(
            workers, (int, np.integer)):
        raise TypeError("workers must be an integer")
    value = int(workers)
    if value < 1:
        raise ValueError("workers must be positive")
    return value


@contextmanager
def worker_limit(workers: int):
    """Run the calls inside on ``workers`` threads when they name none.

    A preparation that was given a thread count (``--preprocess-workers``,
    its column workers) sets it here around the steps it runs on its own
    thread, so the library does not go above the count the user chose.
    The count never changes an element, only the wall time.
    """
    token = _worker_limit.set(_positive_workers(workers))
    try:
        yield
    finally:
        _worker_limit.reset(token)


def _workers(workers: int | None) -> int:
    if workers is not None:
        return _positive_workers(workers)
    limit = _worker_limit.get()
    if limit is not None:
        return limit
    # A call from a worker thread belongs to a caller that already split
    # the work over threads; it runs on that thread alone.
    if threading.current_thread() is not threading.main_thread():
        return 1
    from gpuwm.ingest.cpu_backend import automatic_workers

    return automatic_workers()


def _is_single(*operands) -> bool:
    arrays = [value for value in operands
              if isinstance(value, (np.ndarray, np.generic))]
    if not arrays:
        return False
    return np.result_type(*operands) == np.float32


def _finish(result: np.ndarray):
    return result[()] if result.ndim == 0 else result


# ---- the element-by-element fallback (NumPy's answers for the edges) ----

def _exp(value):
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _log(value):
    if value > 0.0:
        return math.log(value)
    if value == 0.0:
        return -math.inf
    return math.nan


def _log10(value):
    if value > 0.0:
        return math.log10(value)
    if value == 0.0:
        return -math.inf
    return math.nan


def _log1p(value):
    if value > -1.0:
        return math.log1p(value)
    if value == -1.0:
        return -math.inf
    return math.nan


def _periodic(function):
    def guarded(value):
        if math.isinf(value):
            return math.nan
        return function(value)
    return guarded


def _bounded(function):
    def guarded(value):
        if -1.0 <= value <= 1.0:
            return function(value)
        return math.nan
    return guarded


def _pow(base, exponent):
    try:
        return math.pow(base, exponent)
    except OverflowError:
        odd = exponent.is_integer() and math.fmod(exponent, 2.0) != 0.0
        return -math.inf if base < 0.0 and odd else math.inf
    except ValueError:
        return math.inf if base == 0.0 else math.nan


_FALLBACK_UNARY = {
    "exp": _exp, "log": _log, "log1p": _log1p, "log10": _log10,
    "sin": _periodic(math.sin), "cos": _periodic(math.cos),
    "tan": _periodic(math.tan), "asin": _bounded(math.asin),
    "acos": _bounded(math.acos), "atan": math.atan,
}
_FALLBACK_BINARY = {"pow": _pow, "atan2": math.atan2}


def _unary(name: str, values, out, workers):
    single = _is_single(values)
    dtype = np.float32 if single else np.float64
    # order="C" keeps a 0-d operand 0-d (ascontiguousarray would not).
    array = np.asarray(values, dtype=dtype, order="C")
    if out is None:
        result = np.empty(array.shape, dtype=dtype)
    else:
        result = out
        if (not isinstance(result, np.ndarray) or result.dtype != dtype
                or result.shape != array.shape
                or not result.flags.c_contiguous):
            raise ValueError(
                f"out must be a C-contiguous {np.dtype(dtype).name} array of "
                f"shape {array.shape}")
    library = _load()
    if library is None:
        _warn_fallback()
        function = _FALLBACK_UNARY[name]
        flat = np.fromiter((function(float(v)) for v in array.reshape(-1)),
                           dtype=np.float64, count=array.size)
        result[...] = flat.reshape(array.shape).astype(dtype)
        return _finish(result)
    entry = (library.gpuwm_portable_unary_f32 if single
             else library.gpuwm_portable_unary_f64)
    code = entry(_UNARY_CODES[name], array.ctypes.data, result.ctypes.data,
                 array.size, _workers(workers))
    if code != 0:
        raise RuntimeError(
            f"portable {name} failed in the CPU preprocessing library "
            f"(code {code})")
    return _finish(result)


def _binary(name: str, left, right, out, workers):
    single = _is_single(left, right)
    dtype = np.float32 if single else np.float64
    a = np.asarray(left, dtype=dtype)
    b = np.asarray(right, dtype=dtype)
    shape = np.broadcast_shapes(a.shape, b.shape)
    if out is None:
        result = np.empty(shape, dtype=dtype)
    else:
        result = out
        if (not isinstance(result, np.ndarray) or result.dtype != dtype
                or result.shape != shape or not result.flags.c_contiguous):
            raise ValueError(
                f"out must be a C-contiguous {np.dtype(dtype).name} array of "
                f"shape {shape}")
    library = _load()
    if library is None:
        _warn_fallback()
        function = _FALLBACK_BINARY[name]
        x, y = np.broadcast_arrays(a, b)
        flat = np.fromiter(
            (function(float(u), float(v))
             for u, v in zip(x.reshape(-1), y.reshape(-1))),
            dtype=np.float64, count=result.size)
        result[...] = flat.reshape(shape).astype(dtype)
        return _finish(result)
    # A one-element operand travels once and is broadcast in the library;
    # any other broadcast is materialized here.
    operands = []
    for operand in (a, b):
        if operand.size == 1 and result.size != 1:
            operands.append(np.ascontiguousarray(operand.reshape(1)))
        else:
            operands.append(np.ascontiguousarray(
                np.broadcast_to(operand, shape)))
    entry = (library.gpuwm_portable_binary_f32 if single
             else library.gpuwm_portable_binary_f64)
    code = entry(_BINARY_CODES[name], operands[0].ctypes.data,
                 operands[0].size, operands[1].ctypes.data, operands[1].size,
                 result.ctypes.data, result.size, _workers(workers))
    if code != 0:
        raise RuntimeError(
            f"portable {name} failed in the CPU preprocessing library "
            f"(code {code})")
    return _finish(result)


def exp(values, *, out=None, workers=None):
    return _unary("exp", values, out, workers)


def log(values, *, out=None, workers=None):
    return _unary("log", values, out, workers)


def log1p(values, *, out=None, workers=None):
    return _unary("log1p", values, out, workers)


def log10(values, *, out=None, workers=None):
    return _unary("log10", values, out, workers)


def sin(values, *, out=None, workers=None):
    return _unary("sin", values, out, workers)


def cos(values, *, out=None, workers=None):
    return _unary("cos", values, out, workers)


def tan(values, *, out=None, workers=None):
    return _unary("tan", values, out, workers)


def arcsin(values, *, out=None, workers=None):
    return _unary("asin", values, out, workers)


def arccos(values, *, out=None, workers=None):
    return _unary("acos", values, out, workers)


def arctan(values, *, out=None, workers=None):
    return _unary("atan", values, out, workers)


def arctan2(y, x, *, out=None, workers=None):
    """``atan2(y, x)`` over the broadcast operands."""
    return _binary("atan2", y, x, out, workers)


def power(base, exponent, *, out=None, workers=None):
    """``base ** exponent`` through ``pow`` (``powf`` for float32).

    NumPy's own ``**`` takes square or square root for a scalar exponent of
    2 or 0.5; callers that write those keep ``**`` (both are exact).
    """
    return _binary("pow", base, exponent, out, workers)
