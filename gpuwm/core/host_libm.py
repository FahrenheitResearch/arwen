"""Binary64 transcendentals through the C library, element by element.

NumPy's float64 ``exp``, ``log``, ``log10``, ``tan``, ``arctan``,
``arcsin``, ``arccos`` and ``power`` call the C library on most hosts, but
NumPy 2.5 on an AVX-512 Linux machine runs its own vector loops for them
instead, and those round the last bit differently on a few percent of
arguments (scalars included: a 0-d call takes the same loop).  A value that
must come out the same on every machine, or equal what the native Rust
code computes through the same C library, cannot take them.  These take
:mod:`math` one element at a time, which is the C library on every host and
the answer NumPy gives wherever its vector loops are not in use.

``sin``, ``cos``, ``sqrt`` and ``x ** 2`` are not here: NumPy gives the C
library's (or the correctly rounded) answer for them with and without its
AVX-512 loops.  ``arctan2`` is: NumPy 2.5.3's AVX-512 float64 loop differs
from the C library's ``atan2`` on 32,883 of 400,000 random argument pairs
and on 6,096 of 400,000 Lambert-inverse-shaped ones (WSL Ubuntu 24.04,
glibc 2.39), and on none with ``NPY_DISABLE_CPU_FEATURES="X86_V4
AVX512_ICL"``.

The CPU preprocessing bridge calls those same C-library functions over
disjoint ranges on its explicit Rust worker pool. This preserves the host
math contract, including the domain handling below; it does not use the
different portable musl implementation. An older installed bridge retains
the scalar Python reference. These are for initialization and setup, not
per-step fields.
"""

from __future__ import annotations

import math
import ctypes

import numpy as np

__all__ = ["exp", "log", "log10", "tan", "arctan", "arctan2", "arcsin",
           "arccos", "power"]


def _host_entry(pm, name):
    try:
        library = pm._load()
    except FileNotFoundError:
        # Geometry/configuration callers also use this module before
        # preparation's native assets have been installed. Keep their
        # original scalar math path. Other load or ABI errors are real.
        return None
    return getattr(library, name, None)


def _unary(function, values, name):
    from gpuwm.core import portable_math as pm
    entry = _host_entry(pm, "gpuwm_host_unary_f64")
    array = np.require(values, dtype=np.float64, requirements=["C", "A"])
    if entry is not None:
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        entry.argtypes = [ctypes.c_uint32, pointer, pointer, size, size]
        entry.restype = ctypes.c_int32
        out = np.empty(array.shape, dtype=np.float64)
        code = entry(pm._UNARY_CODES[name], array.ctypes.data, out.ctypes.data,
                     array.size, pm._workers(None))
        if code:
            raise RuntimeError(f"native host {name} failed with code {code}")
        return out
    flat = np.fromiter((function(float(v)) for v in array.reshape(-1)),
                       dtype=np.float64, count=array.size)
    return flat.reshape(array.shape)


def _binary(function, left, right, name):
    from gpuwm.core import portable_math as pm
    entry = _host_entry(pm, "gpuwm_host_binary_f64")
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    shape = np.broadcast_shapes(a.shape, b.shape)
    if entry is None:
        x, y = np.broadcast_arrays(a, b)
        flat = np.fromiter((function(float(u), float(v)) for u, v in
                            zip(x.reshape(-1), y.reshape(-1))),
                           dtype=np.float64, count=x.size)
        return flat.reshape(shape)
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    entry.argtypes = [ctypes.c_uint32, pointer, size, pointer, size,
                     pointer, size, size]
    entry.restype = ctypes.c_int32
    out = np.empty(shape, dtype=np.float64)
    if not out.size:
        return out
    workers = pm._workers(None)

    def apply(x, y, target):
        x, y = (np.require(value, dtype=np.float64, requirements=["C", "A"])
                for value in (x, y))
        code = entry(pm._BINARY_CODES[name], x.ctypes.data, x.size,
                     y.ctypes.data, y.size, target.ctypes.data, target.size, workers)
        if code:
            raise RuntimeError(f"native host {name} failed with code {code}")

    if (a.size == 1 or (a.shape == shape and a.flags.c_contiguous)) and (
            b.size == 1 or (b.shape == shape and b.flags.c_contiguous)):
        apply(a, b, out)
    else:
        # General broadcasting marshals at most one bounded block per
        # operand, rather than allocating copies of a whole 3-D field.
        for x, y, target in np.nditer(
                [a, b, out], flags=["external_loop", "buffered", "zerosize_ok"],
                op_flags=[["readonly"], ["readonly"], ["writeonly"]],
                order="C", buffersize=1 << 16):
            apply(x, y, target)
    return out


def _nan_outside(function, lower, upper):
    """``function`` with NumPy's NaN, not :mod:`math`'s ValueError, outside
    ``[lower, upper]``."""
    def guarded(value):
        if lower <= value <= upper:
            return function(value)
        return math.nan
    return guarded


def _log_domain(function):
    def guarded(value):
        if value > 0.0:
            return function(value)
        if value == 0.0:
            return -math.inf
        return math.nan
    return guarded


_LOG = _log_domain(math.log)
_LOG10 = _log_domain(math.log10)
_ASIN = _nan_outside(math.asin, -1.0, 1.0)
_ACOS = _nan_outside(math.acos, -1.0, 1.0)


def _exp(value):
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _tan(value):
    if math.isinf(value):
        return math.nan
    return math.tan(value)


def exp(values) -> np.ndarray:
    return _unary(_exp, values, "exp")


def log(values) -> np.ndarray:
    return _unary(_LOG, values, "log")


def log10(values) -> np.ndarray:
    return _unary(_LOG10, values, "log10")


def tan(values) -> np.ndarray:
    return _unary(_tan, values, "tan")


def arctan(values) -> np.ndarray:
    return _unary(math.atan, values, "atan")


def arctan2(y, x) -> np.ndarray:
    """``atan2(y, x)`` through the C library, over the broadcast operands.

    :func:`math.atan2` raises for no argument, and its answers for zeros,
    infinities and NaNs are the C library's, which are NumPy's.
    """
    return _binary(math.atan2, y, x, "atan2")


def arcsin(values) -> np.ndarray:
    return _unary(_ASIN, values, "asin")


def arccos(values) -> np.ndarray:
    return _unary(_ACOS, values, "acos")


def _pow(base, exponent):
    try:
        return math.pow(base, exponent)
    except OverflowError:
        odd = exponent.is_integer() and math.fmod(exponent, 2.0) != 0.0
        return -math.inf if base < 0.0 and odd else math.inf
    except ValueError:
        # math.pow refuses 0 ** negative (NumPy: inf) and a negative base
        # with a non-integer exponent (NumPy: nan).
        return math.inf if base == 0.0 else math.nan


def power(base, exponent) -> np.ndarray:
    """``base ** exponent`` through the C library's ``pow``.

    A scalar exponent of 2 or 0.5 is NumPy's square or square root, not
    ``pow``, in NumPy's own ``**``; callers that write those keep ``**``.
    """
    return _binary(_pow, base, exponent, "pow")
