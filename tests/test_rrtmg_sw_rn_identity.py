"""The RRTMG SW FP64-emulation armor computes ordinary IEEE FP32 arithmetic.

``gpuwm/core/kernels/rrtmg_sw.cu`` used to route all 679 FP32 add/sub/mul
sites through ``rsw_add``/``rsw_sub``/``rsw_mul``: decode a possibly
subnormal operand from its bits, do the operation in FP64, re-encode a
possibly subnormal result.  It exists because a compiler-emitted FP32
instruction in this unit may or may not flush subnormals depending on how
the module was built: through ``cupy.RawModule`` CuPy appends ``-ftz=true``
and it does (``tools/ftz_receipt/receipt/receipt.json`` route R2, where the
module's own ``--ftz=false`` is overridden and the cubin hashes equal to a
plain ``-ftz=true`` build), while through ``compile_using_nvrtc``, the
route ``gpuwm/core/rrtmg_sw.py`` takes today, it does not (route R3).
Meanwhile the gfortran oracle always keeps subnormals, and this chain
really makes them: exp_tbl floors at 1e-20, so a two-layer transmittance
product is 1e-40.

The same receipt measures route R5 -- inline PTX written without the
``.ftz`` modifier, inside a module compiled with the very same appended
``-ftz=true`` -- as ``ieee-agreement`` on every mechanism it can express.
The flush is the flag's, not the silicon's, and PTX is the only mechanism
that is right under every route rather than under the current one.

So the armor can be paid for with one instruction instead of ~20 IF the
emulation is exactly IEEE FP32 round-to-nearest.  This module proves that
half of the claim on the CPU, exhaustively where exhaustion is possible
and by dense structured + random sweeps elsewhere: ``rsw_*`` transcribed
faithfully from the CUDA source, against NumPy's IEEE FP32 operators
(x86-64 SSE with FTZ/DAZ clear -- the same semantics ``add.rn.f32`` /
``sub.rn.f32`` / ``mul.rn.f32`` carry).  Together with receipt route R5,
that makes the macro swap bit-identical rather than a numerics change,
which is why the max_ulp-0 oracle and batched CUDA gates are expected to
pass byte-for-byte across it.

Red-on-revert: :func:`test_the_check_catches_an_unarmored_transcription`
shows the comparison failing the moment either half of the armor is
dropped, so a green run here is evidence and not a vacuous pass.
"""

from __future__ import annotations

import numpy as np
import pytest

F = np.float32

MIN_NORMAL_F32 = np.float32(np.finfo(np.float32).tiny)      # 2**-126
TWO_POW_M149 = float(2.0 ** -149)
TWO_POW_149 = float(2.0 ** 149)


# ---------------------------------------------------------------------------
# Faithful transcription of the CUDA armor helpers (rrtmg_sw.cu:95-122).
#
# The CUDA source, verbatim:
#
#   __device__ double rsw_f2d(float x)
#   {
#       unsigned int ix = __float_as_uint(x);
#       if (((ix >> 23) & 0xffu) == 0u) {     // zero or subnormal
#           double v = (double)(ix & 0x7fffffu) * 0x1p-149;
#           return (ix & 0x80000000u) ? -v : v;
#       }
#       return (double)x;                     // normal / inf / nan
#   }
#
#   __device__ float rsw_d2f_rn(double y)
#   {
#       double a = fabs(y);
#       if (a > 0.0 && a < 1.1754943508222875e-38) {   /* 0x1p-126 */
#           double scaled = rint(a * 7.1362384635297994e+44);   /* 2^149 */
#           unsigned int m = (unsigned int)scaled;
#           unsigned int s = (__double_as_longlong(y) < 0LL) ? 0x80000000u : 0u;
#           return __uint_as_float(s | m);
#       }
#       return __double2float_rn(y);
#   }
#
# Both are transcribed below on scalar Python floats (which are FP64) so
# the emulation is reproduced exactly, including its two-step rounding.
# ---------------------------------------------------------------------------


def rsw_f2d(x: np.float32) -> float:
    """FP32 -> FP64 that decodes subnormals from bits instead of cvt."""
    ix = int(np.float32(x).view(np.uint32))
    if ((ix >> 23) & 0xFF) == 0:                       # zero or subnormal
        v = float(ix & 0x7FFFFF) * TWO_POW_M149
        return -v if (ix & 0x80000000) else v
    return float(np.float64(np.float32(x)))            # normal / inf / nan


def _rint_half_even(a: float) -> float:
    """C ``rint`` under the default rounding mode: half-to-even."""
    return float(np.rint(a))


def rsw_d2f_rn(y: float) -> np.float32:
    """FP64 -> FP32 that encodes subnormal results instead of cvt."""
    a = abs(y)
    if a > 0.0 and a < 1.1754943508222875e-38:         # 0x1p-126
        scaled = _rint_half_even(a * TWO_POW_149)
        m = int(scaled) & 0xFFFFFFFF
        s = 0x80000000 if np.float64(y).view(np.int64) < 0 else 0
        return np.uint32(s | m).view(np.float32)
    return np.float64(y).astype(np.float32)            # __double2float_rn


def rsw_add(a: np.float32, b: np.float32) -> np.float32:
    return rsw_d2f_rn(rsw_f2d(a) + rsw_f2d(b))


def rsw_sub(a: np.float32, b: np.float32) -> np.float32:
    return rsw_d2f_rn(rsw_f2d(a) - rsw_f2d(b))


def rsw_mul(a: np.float32, b: np.float32) -> np.float32:
    return rsw_d2f_rn(rsw_f2d(a) * rsw_f2d(b))


EMULATED = {"add": rsw_add, "sub": rsw_sub, "mul": rsw_mul}


# ---------------------------------------------------------------------------
# Vectorised IEEE FP32 reference.  NumPy on x86-64 runs SSE with FTZ and DAZ
# clear, i.e. full IEEE-754 binary32 round-to-nearest-even with gradual
# underflow -- the semantics PTX ``add.rn.f32``/``sub.rn.f32``/``mul.rn.f32``
# (no ``.ftz`` modifier) implement.  The guard below refuses to certify
# anything if this build is not actually keeping subnormals.
# ---------------------------------------------------------------------------

IEEE = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
}


def _bits(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def _same_bits(x, y) -> np.ndarray:
    """Bitwise equality, with all NaN payloads treated as one value.

    The CUDA armor and the hardware instruction agree on quiet-NaN-ness but
    the emulation can route a NaN through FP64 and back, which is allowed to
    change the payload; the oracle contract is on numbers, and NaN never
    reaches a stored SW field (the err_flag paths fail closed first).
    """
    xa = np.asarray(x, dtype=np.float32)
    ya = np.asarray(y, dtype=np.float32)
    both_nan = np.isnan(xa) & np.isnan(ya)
    return (_bits(xa) == _bits(ya)) | both_nan


def test_host_float32_keeps_subnormals():
    """Instrument check: this interpreter must not be flushing."""
    tiny = np.float32(1e-20)
    assert tiny * tiny != np.float32(0.0), (
        "NumPy FP32 multiply flushed 1e-20*1e-20 to zero; this build cannot "
        "serve as the IEEE reference (FTZ/DAZ set?)")
    assert np.float32(1e-40) != np.float32(0.0)
    assert np.float32(np.uint32(1).view(np.float32)) != np.float32(0.0)


# ---------------------------------------------------------------------------
# The value sets.  Each is a 1-D float32 array; tests take outer products.
# ---------------------------------------------------------------------------

def _structured_values() -> np.ndarray:
    """Every value class the SW chain and the armor's branches care about."""
    u = []
    # zeros and the smallest subnormals from both ends
    u += [0x00000000, 0x80000000]
    u += [i for i in range(1, 40)]                       # +0x1..0x27 subnormal
    u += [0x80000000 | i for i in range(1, 40)]          # negatives
    # the whole subnormal exponent ladder, both signs
    u += [1 << k for k in range(0, 23)]
    u += [0x80000000 | (1 << k) for k in range(0, 23)]
    u += [(1 << k) - 1 for k in range(1, 24)]            # all-ones mantissas
    u += [0x007FFFFF, 0x807FFFFF]                        # largest subnormals
    # the normal/subnormal boundary and its neighbours
    for base in (0x00800000, 0x00800001, 0x007FFFFF, 0x01000000, 0x00FFFFFF):
        u += [base, 0x80000000 | base]
    # exp_tbl's floor 1e-20 and its square 1e-40 (the proven SW subnormal)
    for v in (1e-20, 1e-40, 1e-45, 5e-45, 1.4e-45, 1e-38, 1.1754944e-38,
              5.877472e-39, 1e-30, 1e-25, 3e-39):
        b = int(np.float32(v).view(np.uint32))
        u += [b, 0x80000000 | b]
    # ordinary physics magnitudes, powers of two, and awkward mantissas
    for v in (1.0, 0.5, 2.0, 3.0, 0.25, 1e-3, 1e3, 1366.0, 0.9999995,
              1.0000001, 8388608.0, 16777215.0, 1e20, 1e30, 3.4028235e38,
              1.1920929e-7, 296.0 / 1013.0):
        b = int(np.float32(v).view(np.uint32))
        u += [b, 0x80000000 | b]
    # infinities and a quiet NaN
    u += [0x7F800000, 0xFF800000, 0x7FC00000]
    return np.unique(np.asarray(u, dtype=np.uint32)).view(np.float32)


def _random_values(rng, n: int) -> np.ndarray:
    """Uniform random bit patterns, biased hard into the subnormal band.

    Uniform-over-bits alone would put ~1 in 2^8 draws in the subnormal
    exponent; the extra draws below make the underflow region carry real
    weight instead of being sampled incidentally.
    """
    uniform = rng.integers(0, 1 << 32, size=n, dtype=np.uint64).astype(np.uint32)
    # dense subnormals: exponent field zero, random mantissa and sign
    sub = (rng.integers(0, 1 << 23, size=n, dtype=np.uint64).astype(np.uint32)
           | (rng.integers(0, 2, size=n, dtype=np.uint64).astype(np.uint32) << 31))
    # small normals just above the boundary (exponent 1..40): these are the
    # operands whose products and cancellations land in the subnormal band
    small = ((rng.integers(1, 41, size=n, dtype=np.uint64).astype(np.uint32) << 23)
             | rng.integers(0, 1 << 23, size=n, dtype=np.uint64).astype(np.uint32)
             | (rng.integers(0, 2, size=n, dtype=np.uint64).astype(np.uint32) << 31))
    return np.concatenate([uniform, sub, small]).view(np.float32)


def _emulate_vec(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Scalar emulation over flattened operand pairs."""
    fn = EMULATED[op]
    out = np.empty(a.size, dtype=np.float32)
    af = a.ravel()
    bf = b.ravel()
    for i in range(af.size):
        out[i] = fn(af[i], bf[i])
    return out.reshape(a.shape)


def _check(op: str, a: np.ndarray, b: np.ndarray) -> None:
    got = _emulate_vec(op, a, b)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        want = np.asarray(IEEE[op](a, b), dtype=np.float32)
    ok = _same_bits(got, want)
    if not ok.all():
        bad = np.argwhere(~ok)[:8]
        lines = []
        for idx in bad:
            i = tuple(idx)
            lines.append(
                f"  {op}(0x{int(_bits(a[i])):08x}, 0x{int(_bits(b[i])):08x}) "
                f"emulated=0x{int(_bits(got[i])):08x} "
                f"ieee=0x{int(_bits(want[i])):08x}")
        raise AssertionError(
            f"{int((~ok).sum())} of {ok.size} {op} pairs differ between the "
            f"FP64-emulation armor and IEEE FP32:\n" + "\n".join(lines))


@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_structured_pairs_are_bit_identical(op):
    """Every structured value against every structured value."""
    v = _structured_values()
    a, b = np.meshgrid(v, v, indexing="ij")
    _check(op, a, b)


@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_random_pairs_are_bit_identical(op):
    """Random bit patterns, subnormal-weighted, fixed seed."""
    rng = np.random.default_rng(20260813)
    a = _random_values(rng, 40_000)
    b = _random_values(rng, 40_000)
    _check(op, a, b)


@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_exhaustive_low_order_pairs_are_bit_identical(op):
    """Exhaustive over the bottom of the subnormal range, both signs.

    These are the patterns where a second rounding could in principle
    disagree with a single one: results at 1, 2, 3 significant bits, where
    round-half-even has real ties to break.
    """
    lo = np.arange(0, 512, dtype=np.uint32)
    vals = np.concatenate([lo, lo | np.uint32(0x80000000)]).view(np.float32)
    a, b = np.meshgrid(vals, vals, indexing="ij")
    _check(op, a, b)


def test_exhaustive_scaled_mantissa_sweep_mul():
    """Products that land exactly on FP32 subnormal rounding ties.

    ``2**-149 * k`` times a power of two walks the result across the whole
    subnormal ladder with exact half-way cases at every step -- the classic
    double-rounding hazard, if the armor had one.
    """
    ks = np.arange(1, 2048, dtype=np.uint32).view(np.float32)     # k * 2**-149
    scales = np.asarray(
        [2.0 ** e for e in range(-30, 31)], dtype=np.float32)
    a, b = np.meshgrid(ks, scales, indexing="ij")
    _check("mul", a, b)


def test_exhaustive_cancellation_sweep_sub():
    """Near-cancellation of small normals, the other route into subnormals.

    ``a - b`` with a and b a few ulps apart just above 2**-126 produces
    every subnormal magnitude by Sterbenz-exact subtraction.
    """
    base = int(np.float32(MIN_NORMAL_F32).view(np.uint32))
    span = np.arange(0, 4096, dtype=np.uint32)
    vals = (base + span).astype(np.uint32).view(np.float32)
    a, b = np.meshgrid(vals[:256], vals, indexing="ij")
    _check("sub", a, b)
    _check("add", a, -b)


def test_the_proven_shortwave_subnormal_is_reproduced():
    """The site the armor was built for: exp_tbl floor squared.

    ``rsw_etbl`` bottoms out at the exp_tbl floor 1e-20; two optically thick
    layers multiply their transmittances in ``ztdbt[J+1] = MU(zdbt[J],
    ztdbt[J])`` and the product is 1e-40, a subnormal.  Under the shipped
    compile route a compiler-emitted ``mul`` returns +0 there; the armor and
    an unflushed ``mul.rn.f32`` both return the subnormal.
    """
    floor = np.float32(1e-20)
    product = rsw_mul(floor, floor)
    assert product != np.float32(0.0)
    assert abs(float(product)) < float(MIN_NORMAL_F32)
    assert _bits(product) == _bits(floor * floor)
    # and one more layer takes it under the smallest subnormal, to a true zero
    assert rsw_mul(product, floor) == np.float32(0.0)


def _flushing_d2f(y: float) -> np.float32:
    """What ``__double2float_rn`` actually does on the shipped compile route.

    Receipt cell (R1/R2, mechanism ``__double2float_rn``) = ``flush-to-zero``:
    NVRTC synthesises a flush around ``cvt.rn.f32.f64``, which has none of
    its own.  NumPy's cast keeps the subnormal, so the defect has to be
    modelled explicitly to drive the red arm.
    """
    out = np.float64(y).astype(np.float32)
    if out != np.float32(0.0) and abs(float(out)) < float(MIN_NORMAL_F32):
        return np.float32(0.0) if out > 0 else np.float32(-0.0)
    return out


def _flushing_f2d(x: np.float32) -> float:
    """What a plain ``(double)x`` cast does on the shipped route: DAZ."""
    xf = np.float32(x)
    if xf != np.float32(0.0) and abs(float(xf)) < float(MIN_NORMAL_F32):
        return 0.0
    return float(np.float64(xf))


@pytest.mark.parametrize("defect", ["flushed_result", "flushed_operand"])
def test_the_check_catches_an_unarmored_transcription(defect):
    """Red-on-revert: the comparison is not vacuous.

    Drop either half of the armor -- the subnormal decode on the way in
    (``rsw_f2d`` -> plain cast, i.e. DAZ) or the subnormal encode on the way
    out (``rsw_d2f_rn`` -> ``__double2float_rn``, i.e. FTZ) -- and the very
    same check the passing tests run must go red.  This is what a
    compiler-emitted ``mul.rn.ftz.f32`` would score, and it is the state the
    unit would be in if the macros were swapped to bare ``__fmul_rn``.
    """
    if defect == "flushed_result":
        def broken_mul(a, b):
            return _flushing_d2f(rsw_f2d(a) * rsw_f2d(b))
        # 1e-20 * 1e-20 = 1e-40: a subnormal RESULT from normal operands.
        a = np.asarray([np.float32(1e-20)] * 4)
        b = np.asarray([np.float32(1e-20)] * 4)
    else:
        def broken_mul(a, b):
            return rsw_d2f_rn(_flushing_f2d(a) * _flushing_f2d(b))
        # 1e-40 * 1e10 = 1e-30: a NORMAL result from a subnormal operand,
        # which only a decode-side defect can get wrong.
        a = np.asarray([np.float32(1e-40)] * 4)
        b = np.asarray([np.float32(1e10)] * 4)

    saved = EMULATED["mul"]
    EMULATED["mul"] = broken_mul
    try:
        with pytest.raises(AssertionError, match="differ between"):
            _check("mul", a, b)
    finally:
        EMULATED["mul"] = saved
    # ... and with the real transcription restored, the same call passes.
    _check("mul", a, b)
