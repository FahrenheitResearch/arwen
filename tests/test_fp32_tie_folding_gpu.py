"""Toolchain guard: the PTX->SASS constant folder must round ties to even.

This is a property of the CUDA toolchain, not of any scheme in this
repository, which is why nothing here is named after a physics module.

Why the guard exists
--------------------
``__fadd_rn`` / ``__fsub_rn`` / ``__fmul_rn`` pin the *hardware* rounding
mode: NVVM lowers them to ``add.rn.f32`` / ``sub.rn.f32`` / ``mul.rn.f32``
and does not fold them itself, so a kernel whose operands are compile-time
constants reaches ptxas as an arithmetic instruction with two immediate
operands.  ptxas then folds it -- and that fold is a *different*
implementation of the operation from the one the SM executes.  Measured on
sm_120: ptxas 12.8.93 and 12.9.86 resolve some exact halfway cases by
rounding toward the smaller-magnitude neighbour instead of to the even
significand; 13.0.88 and 13.1.115 do not.  The surfaced symptom was a 1-ULP
mismatch against a pinned oracle six kernels deep; this file turns it into a
sub-second failure that names the operation and both operand words.

Why the sweep has to be a sweep
-------------------------------
The defect is *not* reproducible from a single hand-picked pair, which is
what makes a one-case guard worthless here.  Measured on ptxas 12.8.93, all
three of these are ties and only the last one folds wrong:

    __fsub_rn(0x3CCCCCCD, 0x3BA3D70A)   folds to the even word (correct)
    __fsub_rn(0x3E99999A, 0x3DCCCCCD)   folds to the even word (correct)
    __fsub_rn(0x3CCCCCCC, 0x3BA3D70A)   folds to the odd word  (wrong)

So the guard sweeps instead of asserting one case, and it keeps a small set
of anchor pairs taken from constants this repository's own kernels hand the
folder, so the exact production exposure can never regress silently.

What is swept
-------------
``sub``, ``add`` and ``mul`` FP32 pairs whose *exact* result is an exact
halfway case, constructed (not sampled) so that both tie classes are
covered: ties whose round-to-nearest-even answer is the larger neighbour and
ties whose answer is the smaller one.  Only the first class has ever been
observed to fold wrong, so a sweep that omits it is blind.

Each case is evaluated three ways inside one kernel:

``literal``    both operands are compile-time literals -- the constant folder;
``global``     both operands are read from device global memory -- the SM;
``constant``   both operands are read from ``__constant__`` memory -- the
               barrier this repository uses to keep a foldable constant table
               away from the folder.

All three must equal the round-to-nearest-even reference.  A toolchain that
declines to fold passes the ``literal`` arm trivially; that is the correct
verdict, because a fold that never happens cannot round the wrong way.

FP32 division and square root are deliberately absent, and that is a proof,
not an omission -- see ``test_fp32_division_and_sqrt_cannot_produce_a_tie``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import requires_gpu

# --------------------------------------------------------------------------
# exact-arithmetic helpers (host side, integer/float64 only)
# --------------------------------------------------------------------------

_F32_SIGNIFICAND_BITS = 24


def _as_odd_significand(value: float) -> tuple[int, int]:
    """Return ``(m, e)`` with ``value == m * 2**e`` exactly and ``m`` odd."""
    if value == 0.0:
        return 0, 0
    fraction, exponent = math.frexp(abs(value))
    m = int(fraction * (1 << 53))
    e = exponent - 53
    while m % 2 == 0:
        m //= 2
        e += 1
    return m, e


def _is_float32_tie(value: float) -> bool:
    """True when ``value`` sits exactly halfway between two float32 values.

    A float32 in binade ``[2**n, 2**(n+1))`` is a multiple of ``2**(n-23)``,
    so a halfway case is an *odd* multiple of ``2**(n-24)``: its odd
    significand has exactly 25 bits.  Subnormal halfway cases are excluded
    here because none of the constructions below reach them.
    """
    if value == 0.0 or not math.isfinite(value):
        return False
    m, _ = _as_odd_significand(value)
    return m.bit_length() == _F32_SIGNIFICAND_BITS + 1


def _round_to_nearest_even_f32(value: float) -> np.float32:
    """RNE of an exactly representable float64 down to float32.

    Every value handed to this function is exact in float64 (the sum,
    difference and product of two float32 values always are: 24+24 <= 53
    significand bits and the exponent range is not close), so the float64 ->
    float32 conversion is a single correctly rounded step -- no double
    rounding.
    """
    return np.float32(value)


def _bits(value) -> int:
    return int(np.float32(value).view(np.uint32))


def _cuda_literal(value: np.float32) -> str:
    """A C float literal that parses back to exactly ``value``."""
    text = "%.*e" % (17, float(value))
    assert np.float32(float(text)) == value
    return text + "f"


# --------------------------------------------------------------------------
# tie construction
# --------------------------------------------------------------------------

_TIE_EXPONENTS = (-30, -20, -6, -1, 0, 3, 12, 40)

# Odd offsets used as the second operand's significand.  Odd is required:
# an even significand would move the tie out of the last representable bit.
_ODD_OFFSETS = (1, 3, 5, 11, 4097, 8191, 65537, 1048577, 4194303, 8388607,
                12345679, 16777213)


def _tie_targets() -> list[tuple[int, int]]:
    """``(M, e)`` pairs describing tie values ``M * 2**(e-24)``.

    ``M`` is odd with 25 bits, so the value lies in ``[2**e, 2**(e+1))`` and
    is exactly halfway between ``((M-1)//2) * 2**(e-23)`` and
    ``((M+1)//2) * 2**(e-23)``.  ``(M-1)//2`` takes both parities across the
    list, so both tie classes are present.
    """
    out: list[tuple[int, int]] = []
    m = (1 << 24) + 1
    for _ in range(64):
        for e in _TIE_EXPONENTS:
            out.append((m, e))
        m += 16382                # even stride keeps M odd
        if m >= (1 << 25):
            m = (1 << 24) + 3
    return out


def _exact_float32(significand: int, exponent: int) -> np.float32:
    value = np.float32(math.ldexp(significand, exponent))
    assert float(value) == math.ldexp(significand, exponent), \
        (significand, exponent)
    return value


def _sub_cases(count: int) -> list[tuple[np.float32, np.float32]]:
    """``(a, b)`` with ``a - b`` exactly a tie.

    With ``r = M * 2**(e-24)`` and ``b = B * 2**(e-24)`` for odd ``B``,
    ``a = r + b = (M+B) * 2**(e-24)`` and ``M+B`` is even, so ``a`` is
    representable as long as ``(M+B)//2`` fits in 24 bits.
    """
    cases = []
    for i, (M, e) in enumerate(_tie_targets()):
        if len(cases) >= count:
            break
        B = _ODD_OFFSETS[i % len(_ODD_OFFSETS)]
        if (M + B) >= (1 << 25):
            continue
        cases.append((_exact_float32((M + B) // 2, e - 23),
                      _exact_float32(B, e - 24)))
    assert len(cases) == count
    return cases


def _add_cases(count: int) -> list[tuple[np.float32, np.float32]]:
    """``(a, b)`` with ``a + b`` exactly a tie."""
    cases = []
    for i, (M, e) in enumerate(_tie_targets()):
        if len(cases) >= count:
            break
        B = _ODD_OFFSETS[i % len(_ODD_OFFSETS)]
        if B >= M:
            continue
        cases.append((_exact_float32((M - B) // 2, e - 23),
                      _exact_float32(B, e - 24)))
    assert len(cases) == count
    return cases


def _mul_cases(count: int) -> list[tuple[np.float32, np.float32]]:
    """``(a, b)`` with ``a * b`` exactly a tie.

    A 25-bit odd product needs two odd factors, so the tie targets here are
    built from the factors rather than picked first.
    """
    cases = []
    u = (1 << 12) + 1
    v = (1 << 12) + 3
    exponents = (-12, 0, 5, -30)
    idx = 0
    while len(cases) < count:
        product = u * v
        if (1 << 24) <= product < (1 << 25):
            e = exponents[idx % len(exponents)]
            idx += 1
            cases.append((np.float32(math.ldexp(u, e)),
                          np.float32(math.ldexp(v, -12))))
        v += 2
        if v >= (1 << 13):
            v = (1 << 12) + 1
            u += 2
            if u >= (1 << 13):
                u = (1 << 12) + 1
    return cases


_OPS = {
    "sub": ("__fsub_rn", lambda a, b: float(a) - float(b), _sub_cases),
    "add": ("__fadd_rn", lambda a, b: float(a) + float(b), _add_cases),
    "mul": ("__fmul_rn", lambda a, b: float(a) * float(b), _mul_cases),
}

_PER_OP = 48

# Operand words that this repository's compiled CUDA modules actually hand
# the constant folder, harvested by scanning the NVRTC PTX of every kernel
# module for arithmetic with two compile-time operands and keeping the pairs
# whose exact result is a halfway case.  They are anchors, not the sweep: the
# third one is the pair ptxas 12.8.93 and 12.9.86 get wrong.
_ANCHOR_WORDS = (
    ("sub", 0x3CCCCCCD, 0x3BA3D70A),
    ("sub", 0x3E99999A, 0x3DCCCCCD),
    ("sub", 0x3CCCCCCC, 0x3BA3D70A),
    ("add", 0x3FCCCCCD, 0x3FA66666),
)


def _word(value: int) -> np.float32:
    return np.uint32(value).view(np.float32)


def _build_sweep():
    """Return ``(cases, source)`` for the whole sweep.

    ``cases`` is a list of ``(op, a, b, exact, want)``; ``source`` is the
    CUDA module evaluating every case through all three operand sources.
    """
    cases = []
    for op, a_word, b_word in _ANCHOR_WORDS:
        a, b = _word(a_word), _word(b_word)
        exact = _OPS[op][1](a, b)
        assert _is_float32_tie(exact), ("anchor", op, hex(a_word), hex(b_word))
        cases.append((op, a, b, exact, _round_to_nearest_even_f32(exact)))
    for op, (_intrinsic, exact_op, generator) in _OPS.items():
        for a, b in generator(_PER_OP):
            exact = exact_op(a, b)
            assert _is_float32_tie(exact), (op, float(a), float(b), exact)
            cases.append((op, a, b, exact, _round_to_nearest_even_f32(exact)))

    n = len(cases)
    literal_lines = []
    memory_lines = []
    for i, (op, a, b, _exact, _want) in enumerate(cases):
        intrinsic = _OPS[op][0]
        literal_lines.append(
            "    out[%d] = __float_as_uint(%s(%s, %s));"
            % (i, intrinsic, _cuda_literal(a), _cuda_literal(b)))
        memory_lines.append(
            "    out[%d] = __float_as_uint(%s(src[%d], src[%d]));"
            % (i, intrinsic, 2 * i, 2 * i + 1))

    operands = []
    for _op, a, b, _exact, _want in cases:
        operands.extend([_cuda_literal(a), _cuda_literal(b)])

    source = "\n".join([
        "// generated by tests/test_fp32_tie_folding_gpu.py",
        "__constant__ float tie_operands[%d] = {" % (2 * n),
        "    " + ", ".join(operands),
        "};",
        "",
        'extern "C" __global__ void fold_literal(unsigned* out) {',
        "\n".join(literal_lines),
        "}",
        "",
        'extern "C" __global__ void fold_memory('
        "unsigned* out, const float* src) {",
        "\n".join(memory_lines),
        "}",
        "",
        'extern "C" __global__ void fold_constant(unsigned* out) {',
        "    const float* src = tie_operands;",
        "\n".join(memory_lines),
        "}",
        "",
    ])
    return cases, source


@pytest.mark.gpu
@requires_gpu
def test_fp32_constant_folding_rounds_ties_to_even():
    """Every FP32 tie must round to even, however the operands arrive.

    The ``literal`` arm is the guard: it is the only one the PTX->SASS
    constant folder can answer, and a folder that resolves halfway cases by
    truncation instead of ties-to-even fails here in well under a second
    instead of surfacing as a 1-ULP oracle mismatch inside a physics kernel.
    """
    import cupy as cp

    cases, source = _build_sweep()
    n = len(cases)
    module = cp.RawModule(code=source, options=("-std=c++17",))

    operand_host = np.empty(2 * n, dtype=np.float32)
    for i, (_op, a, b, _exact, _want) in enumerate(cases):
        operand_host[2 * i] = a
        operand_host[2 * i + 1] = b
    operands = cp.asarray(operand_host)

    results = {}
    for arm, kernel, args in (
            ("literal", "fold_literal", ()),
            ("global", "fold_memory", (operands,)),
            ("constant", "fold_constant", ())):
        out = cp.zeros(n, dtype=cp.uint32)
        module.get_function(kernel)((1,), (1,), (out,) + args)
        cp.cuda.Stream.null.synchronize()
        results[arm] = out.get()

    failures = []
    for arm, got in results.items():
        for i, (op, a, b, exact, want) in enumerate(cases):
            if int(got[i]) != _bits(want):
                failures.append(
                    "%s %s(0x%08X, 0x%08X): exact %.20g is an FP32 tie; "
                    "got 0x%08X, round-to-nearest-even is 0x%08X"
                    % (arm, op, _bits(a), _bits(b), exact,
                       int(got[i]), _bits(want)))

    assert not failures, (
        "%d of %d FP32 tie evaluations did not round to even.\n"
        "A 'literal' failure is the PTX->SASS constant folder; a 'global' or "
        "'constant' failure would be the SM itself.\n%s"
        % (len(failures), 3 * n, "\n".join(failures[:24])))


@pytest.mark.gpu
@requires_gpu
def test_fp32_tie_sweep_covers_both_rounding_directions():
    """Negative control on the sweep, not on the toolchain.

    A round-half-down folder is only visible on ties whose even neighbour is
    the *larger* one; a sweep that happened to contain only the other class
    would pass on a defective toolchain and prove nothing.  Assert both
    classes are present, for every operation.
    """
    cases, _source = _build_sweep()
    seen = {}
    for op, _a, _b, exact, want in cases:
        seen.setdefault(op, set()).add(float(want) > exact)
    for op in _OPS:
        assert seen.get(op) == {True, False}, (
            "%s ties do not cover both rounding directions: %s"
            % (op, sorted(seen.get(op, ()))))


@pytest.mark.gpu
@requires_gpu
def test_fp32_division_and_sqrt_cannot_produce_a_tie():
    """Bound the guard: div and sqrt folds have no halfway case to get wrong.

    ``a / b``: write ``a = A*2**p`` and ``b = B*2**q`` with ``A``, ``B`` odd
    and below ``2**24``.  The quotient is a halfway case only if it is a
    dyadic rational, which forces ``B | A``; then ``A/B`` is an odd integer
    below ``2**24``, so the quotient is *exactly* representable and is not a
    halfway case.  ``sqrt(x)``: a 25-bit odd significand squares to a 49- or
    50-bit odd significand, which no float32 argument has.  So neither
    operation can hand the folder a tie -- the exposure is confined to
    ``add``, ``sub`` and ``mul``, which is structure, not luck.

    The assertion below is the computational half of that argument: a search
    over the same value space the constructions above use finds no quotient
    and no square root that is a tie.
    """
    rng = np.random.default_rng(20260725)
    checked = 0
    for _ in range(20000):
        a = np.float32(rng.uniform(1.0, 2.0) * 2.0 ** int(rng.integers(-8, 9)))
        b = np.float32(rng.uniform(1.0, 2.0) * 2.0 ** int(rng.integers(-8, 9)))
        if b == 0:
            continue
        checked += 1
        # float64 division of two float32 values is exact whenever the
        # quotient is dyadic at all, which is the only case that could be a
        # tie; a non-dyadic quotient is irrational-in-binary and cannot be.
        A, p = _as_odd_significand(float(a))
        B, q = _as_odd_significand(float(b))
        if A % B == 0:
            assert not _is_float32_tie(math.ldexp(A // B, p - q))
        root = float(a) ** 0.5
        if root == int(root) or (root * root == float(a)):
            assert not _is_float32_tie(root)
    assert checked > 19000, checked
