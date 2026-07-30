"""One total-order FP32 ULP metric, so oracle gates stop re-deriving it.

Every oracle-parity gate in this repository measures the same quantity: how
many representable float32 values separate a port's answer from the word the
unmodified WRF module wrote.  That measurement is a two-line bit trick, which
is exactly why it was copy-pasted into sixteen modules -- and why thirteen of
those copies carried the same sign error for as long as they existed:

    np.where(bits < 0, np.int64(0x80000000) - bits, bits)     # wrong
    np.where(bits < 0, np.int64(-0x80000000) - bits, bits)    # right

The widened ``bits`` is a *signed* int64, so subtracting from ``+2**31``
lands the negative half of the number line 2**32 above the positive half
instead of below it.  Measured consequence: ``-0.0`` reads as 4294967296 ULP
from ``+0.0``, and ``-1.4e-45`` as 4294967294 ULP from ``+1.4e-45`` instead
of 2.  The error only ever over-reports, so no gate was ever loosened by it,
but it turns a passing comparison into a nonsense failure message and it
cannot survive a field that legitimately holds signed zero.  That case is
live, not hypothetical: MYNN's ``gh`` is ``-0.0`` on all seven neutral-shear
pairs in both the CUDA result and the oracle.

``monotone_fp32_key`` maps float32 bit patterns onto integers that increase
with the value they encode, so adjacent representable floats get adjacent
keys across the whole line, the two signed zeros collapse to one key, and
plain integer subtraction is the ULP distance.  Comparing bit patterns rather
than magnitudes also means a NaN opposite a finite value can never read as
agreement.

Nothing here is case- or scheme-specific and nothing here imports anything
but numpy: oracle validators under ``tools/`` are run as standalone scripts
and must not have to drag a physics package in to measure a difference.

NaN, and why the distance is a sentinel
---------------------------------------
A second derivation of this map arrived with the five Noah-MP leaf lanes,
written in uint32 rather than widened int32.  It computes *the same map*:
for a negative float with signed pattern ``b`` and unsigned pattern
``u = b + 2**32``, ``2**31 - u == -2**31 - b``, so the two spellings agree
bit for bit over the whole line, and both fold the signed zeros onto one key.
It was folded in here rather than kept, and everything below is built on
``monotone_fp32_key`` so that the map itself still exists exactly once.

It did carry one behaviour this module did not, and that behaviour is the
stricter one, so it won: **any NaN that is not bit-identical to its
counterpart is reported as** :data:`MISMATCH`, not as a bit distance.  Under a
plain bit distance two NaNs with adjacent payloads -- ``0x7FC00000`` against
``0x7FC00001`` -- read as 1 ULP apart, which a gate stated as ``<= 1`` would
accept.  ``max_ulp 0`` gates were never exposed to that, but gates with a
budget are, and a NaN must not be able to buy its way under a budget.  Both
spellings already refused to report 0 for NaN-against-a-number; only the
sentinel also refuses to report *small*.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "monotone_fp32_key",
    "fp32_ulp_distance",
    "MISMATCH",
    "ulp_index",
    "ulp_diff",
    "max_ulp",
    "bitwise_identical",
    "assert_bit_exact",
]

#: Reported instead of a distance when either side is NaN and the two bit
#: patterns are not identical.  Larger than the largest real distance (2**32)
#: so it can never be mistaken for one, and still an int64 a caller can print.
MISMATCH: int = 1 << 40

_EXP_MASK = np.uint32(0x7F800000)
_MANT_MASK = np.uint32(0x007FFFFF)


def monotone_fp32_key(values) -> np.ndarray:
    """Return int64 keys that increase monotonically with the float32 value.

    The sign-magnitude float32 layout orders the negative half backwards, so
    the negative branch is reflected through ``INT32_MIN``.  Subtracting from
    ``INT32_MIN`` -- rather than from ``+2**31`` -- is what folds ``-0.0``
    onto ``+0.0`` at distance 0 and keeps the two halves of the number line
    contiguous.
    """
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.int32)
    bits = bits.astype(np.int64)
    return np.where(bits < 0, np.int64(-0x80000000) - bits, bits)


def _as_bits(values) -> np.ndarray:
    """The raw uint32 patterns, for the two questions the key cannot answer.

    ``monotone_fp32_key`` deliberately loses the difference between ``-0.0``
    and ``+0.0`` and orders NaN by its payload.  Bit-level identity and NaN
    screening therefore read the pattern directly -- which is not a second
    ordering, only a view.
    """
    return np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)


def _is_nan_bits(bits: np.ndarray) -> np.ndarray:
    return ((bits & _EXP_MASK) == _EXP_MASK) & ((bits & _MANT_MASK) != 0)


def fp32_ulp_distance(got, want) -> np.ndarray:
    """Return the elementwise ULP distance between two float32 arrays.

    Shapes must already agree; a gate comparing differently shaped arrays is
    measuring the wrong thing and should say so rather than broadcast.

    A lane whose bit patterns are both NaN and identical is 0 apart.  Every
    other NaN -- different payloads, or NaN against a number -- is
    :data:`MISMATCH` rather than a small number a budgeted gate could accept.
    """
    got = np.ascontiguousarray(got, dtype=np.float32)
    want = np.ascontiguousarray(want, dtype=np.float32)
    if got.shape != want.shape:
        raise ValueError(f"shape mismatch: {got.shape} vs {want.shape}")
    got_bits, want_bits = _as_bits(got), _as_bits(want)
    distance = np.abs(monotone_fp32_key(got) - monotone_fp32_key(want))
    nan = _is_nan_bits(got_bits) | _is_nan_bits(want_bits)
    distance = np.where(nan, np.int64(MISMATCH), distance)
    return np.where(nan & (got_bits == want_bits), np.int64(0), distance)


#: The same map under the name the leaf lanes' gates import it by.
ulp_index = monotone_fp32_key

#: The same measurement under the name the leaf lanes' gates import it by.
ulp_diff = fp32_ulp_distance


def max_ulp(got, want) -> int:
    """Worst ULP distance over the whole array pair; 0 if both are empty."""
    distance = fp32_ulp_distance(got, want)
    return int(distance.max()) if distance.size else 0


def bitwise_identical(got, want) -> bool:
    """True iff every lane carries the identical 32-bit pattern.

    Strictly stronger than ``max_ulp(...) == 0``, which also passes ``-0.0``
    against ``+0.0``.  A gate that means bit-for-bit must say so with this.
    """
    got_bits, want_bits = _as_bits(got), _as_bits(want)
    return got_bits.shape == want_bits.shape and bool(
        np.array_equal(got_bits, want_bits)
    )


def assert_bit_exact(got, want, label: str = "") -> None:
    """Raise AssertionError unless ``got`` is bit-identical to ``want``.

    Reports the worst ULP distance and the first offending lane, so a failure
    says what the residue is instead of only that one exists.
    """
    got_bits, want_bits = _as_bits(got), _as_bits(want)
    if got_bits.shape != want_bits.shape:
        raise AssertionError(f"{label}: shape {got_bits.shape} != {want_bits.shape}")
    if np.array_equal(got_bits, want_bits):
        return
    distance = fp32_ulp_distance(got, want)
    differing = np.flatnonzero((got_bits != want_bits).ravel())
    first = int(differing[0])
    got_value = np.ascontiguousarray(got, dtype=np.float32).ravel()[first]
    want_value = np.ascontiguousarray(want, dtype=np.float32).ravel()[first]
    raise AssertionError(
        f"{label}: {differing.size}/{got_bits.size} lanes differ,"
        f" max_ulp={int(distance.max())}; first at flat index {first}:"
        f" got {got_value!r} (0x{got_bits.ravel()[first]:08x})"
        f" want {want_value!r} (0x{want_bits.ravel()[first]:08x})"
    )
