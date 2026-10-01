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

These cost a Python call per element.  They are for setup-scale arrays
(projection transforms, coefficient ladders), not per-step fields.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["exp", "log", "log10", "tan", "arctan", "arctan2", "arcsin",
           "arccos", "power"]


def _unary(function, values):
    array = np.asarray(values, dtype=np.float64)
    flat = np.fromiter((function(float(v)) for v in array.reshape(-1)),
                       dtype=np.float64, count=array.size)
    return flat.reshape(array.shape)


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
    return _unary(_exp, values)


def log(values) -> np.ndarray:
    return _unary(_LOG, values)


def log10(values) -> np.ndarray:
    return _unary(_LOG10, values)


def tan(values) -> np.ndarray:
    return _unary(_tan, values)


def arctan(values) -> np.ndarray:
    return _unary(math.atan, values)


def arctan2(y, x) -> np.ndarray:
    """``atan2(y, x)`` through the C library, over the broadcast operands.

    :func:`math.atan2` raises for no argument, and its answers for zeros,
    infinities and NaNs are the C library's, which are NumPy's.
    """
    a, b = np.broadcast_arrays(np.asarray(y, dtype=np.float64),
                               np.asarray(x, dtype=np.float64))
    flat = np.fromiter(
        (math.atan2(float(u), float(v)) for u, v in zip(a.reshape(-1),
                                                        b.reshape(-1))),
        dtype=np.float64, count=a.size)
    return flat.reshape(a.shape)


def arcsin(values) -> np.ndarray:
    return _unary(_ASIN, values)


def arccos(values) -> np.ndarray:
    return _unary(_ACOS, values)


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
    b, e = np.broadcast_arrays(np.asarray(base, dtype=np.float64),
                               np.asarray(exponent, dtype=np.float64))
    flat = np.fromiter(
        (_pow(float(x), float(y)) for x, y in zip(b.reshape(-1),
                                                  e.reshape(-1))),
        dtype=np.float64, count=b.size)
    return flat.reshape(b.shape)
