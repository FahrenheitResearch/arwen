"""Platform-independent, correctly rounded FP64 ``exp`` and ``log``.

Not scheme-specific: any lane that derives a *pinned* constant from a
transcendental should call these rather than :mod:`math`.

Why this exists
---------------
``math.exp`` and ``math.log`` are thin wrappers over the host C library, and
the host C library is not the same program on every box.  glibc's FP64 ``exp``
(the ARM optimized-routines core, glibc >= 2.28) returns the correctly rounded
double across the ranges this project derives constants over; the Microsoft
UCRT does not.  They disagree by one ulp, and one ulp is enough to move a
SHA-256.

The concrete failure this module was written for: the aerosol-aware Thompson
droplet-number ladder evaluates ``EXP(n/100 * LOG(3000))`` for ``n`` in
1..99.  At ``n = 61`` the exact value is

    132.142940103429296889...

so the correctly rounded double is ``0x1.08492f71fb081p+7``.  glibc returns
it; the UCRT returns ``0x1.08492f71fb080p+7``, one ulp low.  That single bin
changed ``DERIVED_CONSTANT_SHA256`` and took the whole mp=28 CPU tier down on
Windows -- a pin firing on the vendor of the host libm rather than on any
change to the port.

The fix is not to loosen the pin.  A derived-constant pin exists to catch a
gamma-implementation drift, and it has to keep firing on one.  The fix is to
make the *derivation* independent of the host, so the pinned bytes are a
property of the algorithm alone.

How
---
:mod:`decimal` is a correctly rounded, software-only, fixed-precision
implementation that is part of the standard library, so it behaves identically
on every platform CPython runs on -- unlike the host libm, and unlike
``mpmath``, which is not a declared dependency of this project.
:meth:`decimal.Context.exp` and :meth:`decimal.Context.ln` are documented as
correctly rounded to the context precision.  Evaluating at 60 significant
decimal digits (~199 bits) and rounding once to binary64 (53 bits) reproduces
the correctly rounded double: the intermediate carries ~146 bits of headroom
past the rounding boundary, so the double rounding is only observable for an
argument whose exact result sits within 2**-146 relative of a tie, which does
not occur in the derivations here.  That claim is checked rather than asserted
-- ``tests/test_correctly_rounded_libm.py`` sweeps the arguments these
constants are built from against an independent arbitrary-precision reference.

These are ~12 microseconds per call, a thousand times slower than the libm
they replace.  They are for **import-time derivation of pinned constants**,
where the whole ladder costs a millisecond once.  Do not put them in a kernel
or a per-gridpoint loop; runtime physics must keep using the float32 paths in
``noahmp_libm`` and friends, which exist to reproduce a specific libm's error
rather than to avoid it.
"""

from __future__ import annotations

import decimal

__all__ = ["exp_cr", "log_cr"]

#: 60 significant decimal digits ~= 199 bits, against the 53 a double needs.
_PRECISION = 60

_CONTEXT = decimal.Context(prec=_PRECISION)


def exp_cr(x: float) -> float:
    """``exp(x)`` correctly rounded to binary64, identically on every host."""
    return float(_CONTEXT.exp(decimal.Decimal(x)))


def log_cr(x: float) -> float:
    """``log(x)`` correctly rounded to binary64, identically on every host."""
    return float(_CONTEXT.ln(decimal.Decimal(x)))
