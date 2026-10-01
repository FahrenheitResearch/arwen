"""Fortran semantics the UW PBL reference needs, as measured on the oracle.

gfortran 15.2.0 at -O0 (tools/uwpbl_wrf471_oracle/README.md):

* ``MAX(a, b)`` returns ``a`` when ``a > b`` or ``a`` is NaN, else ``b`` (ties
  return the second argument, so max(+0.0, -0.0) is -0.0); ``MIN`` mirrors
  it.  Python's builtin ``max``/``min`` differ on ties and NaN and are never
  used for real values.
* ``x**n`` with an integer constant ``n`` is ``x*x`` for n = 2 and libgcc's
  ``__powidf2`` otherwise; a real exponent is the C library's ``pow``.
* A literal with no kind suffix is single precision.
"""
from __future__ import annotations

import math
import struct


def fmax(a: float, b: float, *more: float) -> float:
    r = a if (a > b or a != a) else b
    for c in more:
        r = r if (r > c or r != r) else c
    return r


def fmin(a: float, b: float, *more: float) -> float:
    r = a if (a < b or a != a) else b
    for c in more:
        r = r if (r < c or r != r) else c
    return r


def powi(x: float, m: int) -> float:
    """libgcc ``__powidf2``: square and multiply, reciprocal for m < 0."""
    n = -m if m < 0 else m
    y = x if n % 2 else 1.0
    n >>= 1
    while n:
        x = x * x
        if n % 2:
            y = y * x
        n >>= 1
    return 1.0 / y if m < 0 else y


def F32(x: float) -> float:
    """A default-REAL (float32) literal widened to binary64."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def narrow(x: float) -> float:
    """binary64 -> default REAL, round to nearest even (value as a float)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def fint(x: float) -> int:
    """``INT(x)``: truncation toward zero."""
    return int(x)


def aint(x: float) -> float:
    """``AINT(x)``: truncation toward zero, kept in binary64."""
    return float(math.trunc(x)) if math.isfinite(x) else x


def sign(a: float, b: float) -> float:
    """``SIGN(a, b)``: |a| carrying the sign bit of b."""
    return math.copysign(abs(a), b)


def zeros(n: int) -> list:
    """A 1-based binary64 array of length n (index 0 unused)."""
    return [0.0] * (n + 1)
