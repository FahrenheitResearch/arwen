"""Bit-faithful reproductions of the glibc 2.39 FP32 libm calls WRF makes.

Not Noah-MP-specific despite the module name (the Noah-MP lane owns the
``noahmp_*`` namespace; promote this to a shared module when another lane needs
it).

Why this exists
---------------
The oracle is gfortran 13.3.0 linked against glibc 2.39, so ``EXP``, ``LOG``,
``LOG10`` and ``**`` on ``REAL(4)`` become calls to glibc's ``expf``, ``logf``,
``log10f`` and ``powf``.  **None of those four is correctly rounded.**  Rounding
the FP64 result once -- the shim the MYNN lane uses, and the obvious thing to
do -- disagrees with glibc on a large fraction of the domain:

| call      | domain swept                    | disagreement with FP64-then-round |
|-----------|---------------------------------|----------------------------------:|
| ``log10f``| every FP32 in (0.1, 1.0]        |            5,266,896 / 28,521,268 = **18.47%** |
| ``expf``  | FP32 grid over \\|x\\| <= 25      |               21,750 / 34,902,602 = 0.062% |
| ``powf``  | 4e6 samples over Noah-MP shapes |                       2,618 / 4e6 = 0.065% |

So TDFCND's ``LOG10(SATRATIO)`` cannot reach ``max_ulp 0`` through an FP64
shim: roughly one soil column in five lands on the wrong side of the rounding
boundary.  That is not a tolerance to loosen, it is the wrong function.

What this is
------------
Direct transcriptions of the glibc 2.39 implementations:

* ``logf``   -- ``sysdeps/ieee754/flt-32/e_logf.c`` + ``e_logf_data.c``
  (ARM optimized-routines: 16-entry table, cubic in ``r``, FP64 intermediate,
  one rounding to FP32 at the end).
* ``log10f`` -- ``sysdeps/ieee754/flt-32/e_log10f.c``, which is still the 1993
  SunPro reduction ``log10(x) = k*log10_2 + log10(e)*logf(z)`` evaluated in
  **FP32** on top of ``logf``.  This is why ``log10f`` is so much less accurate
  than ``logf``.
* ``expf``   -- ``sysdeps/ieee754/flt-32/e_expf.c`` + ``e_exp2f_data.c``
  (32-entry ``2^(i/32)`` table, the shift-trick round-to-int, cubic in ``r``).
* ``powf``   -- ``sysdeps/ieee754/flt-32/e_powf.c`` + ``e_powf_log2_data.c``
  (a 16-entry ``log2`` table feeding the same ``exp2`` core).

``TOINT_INTRINSICS`` is 0 on x86-64, so ``POWF_SCALE`` is 1.0 and the shift
trick is used in both ``exp2`` paths; both are reproduced as glibc compiles
them for this target.

Verification (all against the live glibc 2.39 on the oracle host, C
transcription compiled with ``-ffp-contract=off`` and again with
``-mfma -ffp-contract=fast``, identical results):

* ``logf``  : 222,414,918 FP32 inputs swept over (0.1,1], [1,100], [1e-6,0.1]
  -- **0 mismatches**.
* ``log10f``: the same 222,414,918 inputs -- **0 mismatches**.
* ``powf``  : 6,000,000 samples over the Noah-MP bases (7.7, 2.0, 2.2, 0.57)
  and broad random ``(base, exponent)`` -- **0 mismatches**.
* ``expf``  : 146,800,642 FP32 inputs over \\|x\\| <= 80 -- **2 mismatches**,
  both 1 ULP: ``x = 0x4202422F`` (32.5646324) and ``x = 0xC27C65D9``
  (-63.0994606).  Residual rate 1.4e-8, a 4,400x improvement on the FP64
  shim.  Diagnosis: glibc's x86-64 multiarch ``expf`` is selected at run time
  and is not byte-identical to the generic C at those two points; enabling FMA
  contraction in the transcription does not move either, so it is not a
  contraction artefact.  Any Noah-MP column that lands on one of those two
  exact FP32 arguments will miss ``max_ulp 0`` by 1 ULP and must be reported,
  not absorbed.

The CUDA halves of these functions live in
``gpuwm/core/kernels/noahmp_leaves.cu`` and must stay in step with this file.

Two independent transcriptions, and what they measured about each other
-----------------------------------------------------------------------
The five Noah-MP leaf lanes (vegprecip, radiation, bareflux, vegeflux, snow)
arrived carrying a second transcription of ``logf``/``expf``/``powf``/``atanf``
and a third of ``logf``/``atanf``, written independently from the same glibc
2.39 sources.  They differed in one interesting way: the second transcription
put an explicit :func:`math.fma` at every ``a*b + c`` site in ``logf``,
``expf`` and ``powf``'s ``log2``, on the argument that glibc ifunc-selects the
``-mfma`` rebuild (``__logf_fma``, ``__powf_fma``) on an FMA host, and that
``logc + k*Ln2`` in particular is a single FMA in the disassembly.  This file
contracts nothing.

Both were run against each other over 10,500,000 argument pairs and both
against the 30,000 pinned glibc words in
``gpuwm/data/noahmp/oracle/glibc-libm-fp32.csv`` and the 4,256 in
``glibc-atanf-fp32.csv``: **zero disagreements, on every function, in every
sweep**.  The FMA sites are real in the disassembly and they do change the
binary64 intermediate, but not by enough to move the single rounding to
binary32 at the end on any argument either lane could find.  That is a
measured null result, not a proof; it is recorded here because it is the only
independent check these transcriptions will ever get.  This file keeps the
non-FMA form for a second reason as well: :func:`math.fma` is CPython 3.13+,
and the GPU host this project validates on runs 3.12.

Only this file survived the merge.  ``sqrtf`` and :func:`f32` came across from
the leaf lanes; ``log10f`` exists only here.
"""

from __future__ import annotations

import math
import struct

import numpy as np

__all__ = [
    "GLIBC_VERSION",
    "f32",
    "logf",
    "log10f",
    "expf",
    "powf",
    "atanf",
    "sqrtf",
    "expm1f",
    "tanhf",
]

#: The glibc these transcriptions reproduce.  Gates that pin a libm answer
#: assert against this so a host upgrade cannot silently move the reference.
GLIBC_VERSION = "2.39"

F = np.float32
_H = float.fromhex

_U32 = 0xFFFFFFFF
_U64 = 0xFFFFFFFFFFFFFFFF


def _asuint(value) -> int:
    return struct.unpack("<I", struct.pack("<f", F(value)))[0]


def _asfloat(bits: int) -> np.float32:
    return F(struct.unpack("<f", struct.pack("<I", bits & _U32))[0])


def _asuint64(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def _asdouble(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits & _U64))[0]


def _as_int32(bits: int) -> int:
    bits &= _U32
    return bits - 0x100000000 if bits & 0x80000000 else bits


#: ``struct.Struct("<f")``, built once.  ``struct.pack("<f", x)`` re-resolves
#: the module global and looks the format up in struct's internal cache on
#: every call; binding the compiled struct's methods below skips both.  This
#: is the hottest function in the Noah-MP host column -- 5,888 calls per land
#: column, 282,634 per 48-column LSM step -- so the two lookups are worth
#: removing.
_F32_STRUCT = struct.Struct("<f")

#: The two halves of :func:`f32`, published so a caller in an inner loop can
#: inline the rounding instead of paying a Python call frame for it.  The only
#: caller that does is :class:`gpuwm.core.noahmp_vegeflux.R4`, whose operators
#: are the hottest code in the host column; ``_unpack(_pack(x))[0]`` there is
#: character for character what ``f32`` evaluates, so inlining cannot move an
#: answer.  Do not reach for these anywhere a function call is not measurably
#: the cost.
F32_PACK = _F32_STRUCT.pack
F32_UNPACK = _F32_STRUCT.unpack


def f32(x, _pack=_F32_STRUCT.pack, _unpack=_F32_STRUCT.unpack) -> float:
    """Round to float32 and hand it back as a Python float.

    Round-to-nearest-even, which is what a C ``(float)`` cast and a Fortran
    ``REAL(4)`` assignment both do.  The leaf lanes use this rather than
    ``np.float32`` to keep every intermediate a Python float: a chain of
    Python-float operations rounded once per Fortran statement reproduces
    float32 arithmetic exactly, because binary32 -> binary64 -> binary32 is an
    innocuous double rounding for ``+ - * /`` (53 >= 2*24 + 2).

    An argument outside the float32 range raises ``OverflowError`` rather than
    saturating to an infinity.  That is a guard, not a hazard: a Fortran
    ``REAL(4)`` overflow is a physics bug, and none of the leaves reach one.

    ``_pack``/``_unpack`` are default arguments, not globals, so the body does
    no name lookup outside the frame.  Both are bound methods of the module
    ``Struct`` above; a caller that passes them explicitly gets the same
    function, and a caller that rebinds them is out of contract.

    Four alternative carriers were measured on the reference box against a
    21-value probe set (both zeros, the smallest subnormal, the largest finite
    binary32, the first value that overflows it, both infinities and a NaN).
    Two of them are faster and none of them is this function:

    ======================  ========  =========================================
    carrier                 ns/call   how it differs
    ======================  ========  =========================================
    ``struct`` (this one)     229.8   --
    ``ctypes.c_float``        133.7   saturates an overflow to an infinity
    ``array('f')``            162.2   saturates an overflow to an infinity
    ``array('f')`` + guard    194.2   identical, but reads and writes one
                                      module-level buffer, so two threads in
                                      ``f32`` can interleave and return each
                                      other's value
    ``float(np.float32(x))``  481.4   saturates, and warns while doing it
    ======================  ========  =========================================

    The guarded ``array`` variant is the only bit-identical one, and 15% is
    not worth a carrier that is silently wrong under concurrency.  What was
    taken instead is :data:`F32_PACK`/:data:`F32_UNPACK`: the same two calls,
    inlined into the one caller hot enough to notice the frame.
    """
    return _unpack(_pack(x))[0]


def sqrtf(x) -> np.float32:
    """``SQRT`` on ``REAL(4)``, which is not a libm call at all.

    gfortran emits ``sqrtss`` inline, and the hardware square root is
    correctly rounded, so unlike its neighbours in this file this one needs no
    transcription -- computing in binary64 and rounding once is the same
    function (again, 53 >= 2*24 + 2).  It lives here so that a leaf reaching
    for a float32 elementary function has exactly one place to look.
    """
    value = F(x)
    if value != value:                                  # NaN, payload intact
        return value
    if value < F(0.0):
        return F(np.nan)
    return F(math.sqrt(float(value)))


# ---------------------------------------------------------------------------
# e_logf_data.c -- 16-entry table, ln2, and the three polynomial coefficients.
# ---------------------------------------------------------------------------
_LOGF_TAB = (
    (_H("0x1.661ec79f8f3bep+0"), _H("-0x1.57bf7808caadep-2")),
    (_H("0x1.571ed4aaf883dp+0"), _H("-0x1.2bef0a7c06ddbp-2")),
    (_H("0x1.49539f0f010bp+0"), _H("-0x1.01eae7f513a67p-2")),
    (_H("0x1.3c995b0b80385p+0"), _H("-0x1.b31d8a68224e9p-3")),
    (_H("0x1.30d190c8864a5p+0"), _H("-0x1.6574f0ac07758p-3")),
    (_H("0x1.25e227b0b8eap+0"), _H("-0x1.1aa2bc79c81p-3")),
    (_H("0x1.1bb4a4a1a343fp+0"), _H("-0x1.a4e76ce8c0e5ep-4")),
    (_H("0x1.12358f08ae5bap+0"), _H("-0x1.1973c5a611cccp-4")),
    (_H("0x1.0953f419900a7p+0"), _H("-0x1.252f438e10c1ep-5")),
    (_H("0x1p+0"), _H("0x0p+0")),
    (_H("0x1.e608cfd9a47acp-1"), _H("0x1.aa5aa5df25984p-5")),
    (_H("0x1.ca4b31f026aap-1"), _H("0x1.c5e53aa362eb4p-4")),
    (_H("0x1.b2036576afce6p-1"), _H("0x1.526e57720db08p-3")),
    (_H("0x1.9c2d163a1aa2dp-1"), _H("0x1.bc2860d22477p-3")),
    (_H("0x1.886e6037841edp-1"), _H("0x1.1058bc8a07ee1p-2")),
    (_H("0x1.767dcf5534862p-1"), _H("0x1.4043057b6ee09p-2")),
)
_LOGF_LN2 = _H("0x1.62e42fefa39efp-1")
_LOGF_A = (
    _H("-0x1.00ea348b88334p-2"),
    _H("0x1.5575b0be00b6ap-2"),
    _H("-0x1.ffffef20a4123p-2"),
)
_LOGF_OFF = 0x3F330000

# ---------------------------------------------------------------------------
# e_exp2f_data.c -- shared by expf, exp2f and powf.  EXP2F_TABLE_BITS = 5.
# ---------------------------------------------------------------------------
_EXP2F_N = 32
_EXP2F_TAB = (
    0x3FF0000000000000, 0x3FEFD9B0D3158574, 0x3FEFB5586CF9890F,
    0x3FEF9301D0125B51, 0x3FEF72B83C7D517B, 0x3FEF54873168B9AA,
    0x3FEF387A6E756238, 0x3FEF1E9DF51FDEE1, 0x3FEF06FE0A31B715,
    0x3FEEF1A7373AA9CB, 0x3FEEDEA64C123422, 0x3FEECE086061892D,
    0x3FEEBFDAD5362A27, 0x3FEEB42B569D4F82, 0x3FEEAB07DD485429,
    0x3FEEA47EB03A5585, 0x3FEEA09E667F3BCD, 0x3FEE9F75E8EC5F74,
    0x3FEEA11473EB0187, 0x3FEEA589994CCE13, 0x3FEEACE5422AA0DB,
    0x3FEEB737B0CDC5E5, 0x3FEEC49182A3F090, 0x3FEED503B23E255D,
    0x3FEEE89F995AD3AD, 0x3FEEFF76F2FB5E47, 0x3FEF199BDD85529C,
    0x3FEF3720DCEF9069, 0x3FEF5818DCFBA487, 0x3FEF7C97337B9B5F,
    0x3FEFA4AFA2A490DA, 0x3FEFD0765B6E4540,
)
_EXP2F_POLY = (
    _H("0x1.c6af84b912394p-5"),
    _H("0x1.ebfce50fac4f3p-3"),
    _H("0x1.62e42ff0c52d6p-1"),
)
_EXP2F_SHIFT = _H("0x1.8p+52")
_EXP2F_SHIFT_SCALED = _H("0x1.8p+52") / _EXP2F_N
_EXP2F_POLY_SCALED = (
    _EXP2F_POLY[0] / _EXP2F_N / _EXP2F_N / _EXP2F_N,
    _EXP2F_POLY[1] / _EXP2F_N / _EXP2F_N,
    _EXP2F_POLY[2] / _EXP2F_N,
)
_EXP2F_INVLN2_SCALED = _H("0x1.71547652b82fep+0") * _EXP2F_N

# ---------------------------------------------------------------------------
# e_powf_log2_data.c -- POWF_SCALE is 1.0 because TOINT_INTRINSICS is 0.
# ---------------------------------------------------------------------------
_POWF_TAB = (
    (_H("0x1.661ec79f8f3bep+0"), _H("-0x1.efec65b963019p-2")),
    (_H("0x1.571ed4aaf883dp+0"), _H("-0x1.b0b6832d4fca4p-2")),
    (_H("0x1.49539f0f010bp+0"), _H("-0x1.7418b0a1fb77bp-2")),
    (_H("0x1.3c995b0b80385p+0"), _H("-0x1.39de91a6dcf7bp-2")),
    (_H("0x1.30d190c8864a5p+0"), _H("-0x1.01d9bf3f2b631p-2")),
    (_H("0x1.25e227b0b8eap+0"), _H("-0x1.97c1d1b3b7afp-3")),
    (_H("0x1.1bb4a4a1a343fp+0"), _H("-0x1.2f9e393af3c9fp-3")),
    (_H("0x1.12358f08ae5bap+0"), _H("-0x1.960cbbf788d5cp-4")),
    (_H("0x1.0953f419900a7p+0"), _H("-0x1.a6f9db6475fcep-5")),
    (_H("0x1p+0"), _H("0x0p+0")),
    (_H("0x1.e608cfd9a47acp-1"), _H("0x1.338ca9f24f53dp-4")),
    (_H("0x1.ca4b31f026aap-1"), _H("0x1.476a9543891bap-3")),
    (_H("0x1.b2036576afce6p-1"), _H("0x1.e840b4ac4e4d2p-3")),
    (_H("0x1.9c2d163a1aa2dp-1"), _H("0x1.40645f0c6651cp-2")),
    (_H("0x1.886e6037841edp-1"), _H("0x1.88e9c2c1b9ff8p-2")),
    (_H("0x1.767dcf5534862p-1"), _H("0x1.ce0a44eb17bccp-2")),
)
_POWF_A = (
    _H("0x1.27616c9496e0bp-2"),
    _H("-0x1.71969a075c67ap-2"),
    _H("0x1.ec70a6ca7baddp-2"),
    _H("-0x1.7154748bef6c8p-1"),
    _H("0x1.71547652ab82bp0"),
)
_POWF_OFF = 0x3F330000
_POWF_SIGN_BIAS = 1 << (5 + 11)

# e_log10f.c FP32 constants.
_IVLN10 = _asfloat(0x3EDE5BD9)
_LOG10_2HI = _asfloat(0x3E9A2080)
_LOG10_2LO = _asfloat(0x355427DB)
_TWO25 = _asfloat(0x4C000000)


def logf(x) -> np.float32:
    """glibc 2.39 ``logf`` (sysdeps/ieee754/flt-32/e_logf.c)."""
    ix = _asuint(x)
    if ix == 0x3F800000:
        return F(0.0)
    if (ix - 0x00800000) & _U32 >= 0x7F800000 - 0x00800000:
        if ix * 2 & _U32 == 0:
            return F(-np.inf)
        if ix == 0x7F800000:
            return F(x)
        if (ix & 0x80000000) or (ix * 2) & _U32 >= 0xFF000000:
            return F(np.nan)
        ix = _asuint(F(F(x) * F(_H("0x1p23"))))
        ix = (ix - (23 << 23)) & _U32
    tmp = (ix - _LOGF_OFF) & _U32
    i = (tmp >> (23 - 4)) % 16
    k = _as_int32(tmp) >> 23
    iz = (ix - (tmp & 0xFF800000)) & _U32
    invc, logc = _LOGF_TAB[i]
    z = float(_asfloat(iz))
    r = z * invc - 1.0
    y0 = logc + float(k) * _LOGF_LN2
    r2 = r * r
    y = _LOGF_A[1] * r + _LOGF_A[2]
    y = _LOGF_A[0] * r2 + y
    y = y * r2 + (y0 + r)
    return F(y)


def log10f(x) -> np.float32:
    """glibc 2.39 ``log10f`` (still the 1993 SunPro FP32 reduction)."""
    value = F(x)
    hx = _asuint(value)
    k = 0
    if _as_int32(hx) < 0x00800000:
        if hx & 0x7FFFFFFF == 0:
            return F(-_TWO25 / abs(value))
        if _as_int32(hx) < 0:
            return F(np.nan)
        k -= 25
        value = F(value * _TWO25)
        hx = _asuint(value)
    if hx >= 0x7F800000:
        return F(value + value)
    k += (hx >> 23) - 127
    i = 1 if k < 0 else 0
    hx = (hx & 0x007FFFFF) | ((0x7F - i) << 23)
    y = F(k + i)
    value = _asfloat(hx)
    z = F(F(y * _LOG10_2LO) + F(_IVLN10 * logf(value)))
    return F(z + F(y * _LOG10_2HI))


def _exp2_core(xd: float, shift: float, poly, sign_bias: int) -> float:
    """The shared 32-entry exp2 core of glibc's expf / exp2f / powf."""
    kd = xd + shift
    ki = _asuint64(kd)
    kd -= shift
    r = xd - kd
    t = _EXP2F_TAB[ki % _EXP2F_N]
    t = (t + (((ki + sign_bias) & _U64) << (52 - 5))) & _U64
    s = _asdouble(t)
    z = poly[0] * r + poly[1]
    r2 = r * r
    y = poly[2] * r + 1.0
    y = z * r2 + y
    return y * s


def expf(x) -> np.float32:
    """glibc 2.39 ``expf`` (sysdeps/ieee754/flt-32/e_expf.c)."""
    value = F(x)
    abstop = (_asuint(value) >> 20) & 0x7FF
    if abstop >= (_asuint(F(88.0)) >> 20):
        if _asuint(value) == _asuint(F(-np.inf)):
            return F(0.0)
        if abstop >= (_asuint(F(np.inf)) >> 20):
            return F(value + value)
        if value > _asfloat(0x42B17218):        # 0x1.62e42ep6f
            return F(np.inf)
        if value < -_asfloat(0x42CFF1B4):       # -0x1.9fe368p6f
            return F(0.0)
    xd = float(value)
    z = _EXP2F_INVLN2_SCALED * xd
    return F(_exp2_core(z, _EXP2F_SHIFT, _EXP2F_POLY_SCALED, 0))


def _powf_log2(ix: int) -> float:
    tmp = (ix - _POWF_OFF) & _U32
    i = (tmp >> (23 - 4)) % 16
    top = tmp & 0xFF800000
    iz = (ix - top) & _U32
    k = _as_int32(top) >> 23
    invc, logc = _POWF_TAB[i]
    z = float(_asfloat(iz))
    r = z * invc - 1.0
    y0 = logc + float(k)
    r2 = r * r
    y = _POWF_A[0] * r + _POWF_A[1]
    p = _POWF_A[2] * r + _POWF_A[3]
    r4 = r2 * r2
    q = _POWF_A[4] * r + y0
    q = p * r2 + q
    return y * r4 + q


def _checkint(iy: int) -> int:
    exponent = (iy >> 23) & 0xFF
    if exponent < 0x7F:
        return 0
    if exponent > 0x7F + 23:
        return 2
    if iy & ((1 << (0x7F + 23 - exponent)) - 1):
        return 0
    if iy & (1 << (0x7F + 23 - exponent)):
        return 1
    return 2


def _zeroinfnan(ix: int) -> bool:
    return ((2 * ix - 1) & _U32) >= (2 * 0x7F800000 - 1) & _U32


def powf(x, y) -> np.float32:
    """glibc 2.39 ``powf`` (sysdeps/ieee754/flt-32/e_powf.c)."""
    base = F(x)
    exponent = F(y)
    sign_bias = 0
    ix = _asuint(base)
    iy = _asuint(exponent)
    if (ix - 0x00800000) & _U32 >= 0x7F800000 - 0x00800000 or _zeroinfnan(iy):
        if _zeroinfnan(iy):
            if (2 * iy) & _U32 == 0:
                return F(1.0)
            if ix == 0x3F800000:
                return F(1.0)
            if (2 * ix) & _U32 > (2 * 0x7F800000) & _U32 \
                    or (2 * iy) & _U32 > (2 * 0x7F800000) & _U32:
                return F(base + exponent)
            if (2 * ix) & _U32 == 2 * 0x3F800000:
                return F(1.0)
            if ((2 * ix) & _U32 < 2 * 0x3F800000) == (not (iy & 0x80000000)):
                return F(0.0)
            return F(exponent * exponent)
        if _zeroinfnan(ix):
            squared = F(base * base)
            if ix & 0x80000000 and _checkint(iy) == 1:
                squared = F(-squared)
            return F(F(1.0) / squared) if iy & 0x80000000 else squared
        if ix & 0x80000000:
            yint = _checkint(iy)
            if yint == 0:
                return F(np.nan)
            if yint == 1:
                sign_bias = _POWF_SIGN_BIAS
            ix &= 0x7FFFFFFF
        if ix < 0x00800000:
            ix = _asuint(F(base * F(_H("0x1p23")))) & 0x7FFFFFFF
            ix = (ix - (23 << 23)) & _U32
    logx = _powf_log2(ix)
    ylogx = float(exponent) * logx
    if (_asuint64(ylogx) >> 47) & 0xFFFF >= (_asuint64(126.0) >> 47):
        if ylogx > _H("0x1.fffffffd1d571p+6"):
            return F(-np.inf) if sign_bias else F(np.inf)
        if ylogx <= -150.0:
            return F(-0.0) if sign_bias else F(0.0)
    return F(_exp2_core(ylogx, _EXP2F_SHIFT_SCALED, _EXP2F_POLY, sign_bias))


# ---------------------------------------------------------------------------
# atanf -- glibc 2.39 sysdeps/ieee754/flt-32/s_atanf.c, which is still the
# fdlibm reduction: five argument ranges, one 11-term odd/even polynomial, and
# a hi/lo pair per range.  Every operation is FP32; there is no FP64
# intermediate anywhere, which is why an FP64-then-round shim is a different
# function here as well.  SFCDIF1 (module_sf_noahmplsm.F:4691, 4698) is the
# only Noah-MP caller.
#
# Two facts measured against the live glibc 2.39 on the oracle host, not
# assumed:
#
#   * the large-argument shortcut fires at |x| >= 2**25 (0x4C000000), not at
#     the 2**26 or 2**34 thresholds other fdlibm descendants use.  Above it
#     glibc returns the constant 0x3FC90FDB for every argument;
#   * with that threshold, this transcription reproduces glibc on **all
#     4,278,190,082 non-NaN FP32 inputs** -- the entire domain, swept
#     exhaustively -- with 0 mismatches.  Rounding the FP64 result once instead
#     disagrees on 823,767 of the 16,777,216 inputs in [0.5, 1) alone.
# ---------------------------------------------------------------------------
_ATANF_HI = tuple(_asfloat(b) for b in
                  (0x3EED6338, 0x3F490FDA, 0x3F7B985E, 0x3FC90FDA))
_ATANF_LO = tuple(_asfloat(b) for b in
                  (0x31AC3769, 0x33222168, 0x33140FB4, 0x33A22168))
_ATANF_T = tuple(_asfloat(b) for b in (
    0x3EAAAAAB, 0xBE4CCCCD, 0x3E124925, 0xBDE38E38, 0x3DBA2E6E, 0xBD9D8795,
    0x3D886B35, 0xBD6EF16B, 0x3D4BDA59, 0xBD15A221, 0x3C8569D7))


def atanf(x) -> np.float32:
    """glibc 2.39 ``atanf`` (sysdeps/ieee754/flt-32/s_atanf.c)."""
    value = F(x)
    hx = _as_int32(_asuint(value))
    ix = hx & 0x7FFFFFFF
    if ix >= 0x4C000000:                                # |x| >= 2**25
        if ix > 0x7F800000:
            return F(value + value)                     # NaN
        if hx > 0:
            return F(_ATANF_HI[3] + _ATANF_LO[3])
        return F(F(-_ATANF_HI[3]) - _ATANF_LO[3])
    if ix < 0x3EE00000:                                 # |x| < 0.4375
        if ix < 0x31000000:                             # |x| < 2**-29
            return value                                # huge + x > 1 always
        idx = -1
    else:
        value = F(abs(value))
        if ix < 0x3F980000:                             # |x| < 1.1875
            if ix < 0x3F300000:                         # 0.4375 <= |x| < 0.6875
                idx = 0
                value = F(F(F(F(2.0) * value) - F(1.0))
                          / F(F(2.0) + value))
            else:                                       # 0.6875 <= |x| < 1.1875
                idx = 1
                value = F(F(value - F(1.0)) / F(value + F(1.0)))
        elif ix < 0x401C0000:                           # |x| < 2.4375
            idx = 2
            value = F(F(value - F(1.5))
                      / F(F(1.0) + F(F(1.5) * value)))
        else:                                           # 2.4375 <= |x| < 2**25
            idx = 3
            value = F(F(-1.0) / value)
    z = F(value * value)
    w = F(z * z)
    s1 = F(z * F(_ATANF_T[0] + F(w * F(_ATANF_T[2] + F(w * F(
        _ATANF_T[4] + F(w * F(_ATANF_T[6] + F(w * F(
            _ATANF_T[8] + F(w * _ATANF_T[10])))))))))))
    s2 = F(w * F(_ATANF_T[1] + F(w * F(_ATANF_T[3] + F(w * F(
        _ATANF_T[5] + F(w * F(_ATANF_T[7] + F(w * _ATANF_T[9])))))))))
    if idx < 0:
        return F(value - F(value * F(s1 + s2)))
    z = F(_ATANF_HI[idx]
          - F(F(F(value * F(s1 + s2)) - _ATANF_LO[idx]) - value))
    return F(-z) if hx < 0 else z


# ---------------------------------------------------------------------------
# expm1f / tanhf -- sysdeps/ieee754/flt-32/s_expm1f.c and s_tanhf.c
#
# Noah-MP's ENERGY reaches TANH exactly once, in the snow-cover fraction
#   FSNO = TANH( SNOWH / (parameters%SCFFAC * FMELT) )       (:2072)
# and that single call is enough to put every snow column's answer through
# glibc's tanhf.  It is *not* interchangeable with an FP64 shim: over the
# 204,064,836 FP32 inputs in [1e-6, 22], `(float)tanh((double)x)` disagrees
# with glibc's `tanhf` on 48,501,304 of them -- 23.8%, three orders of
# magnitude worse than expf's 0.062%.  These are still the 1993 SunPro
# routines in glibc 2.39, unlike logf/expf/powf above, which is why they are
# so much less accurate.
#
# Verified against the live glibc 2.39 on the oracle host (gcc -O2
# -ffp-contract=off): 1,106,247,680 FP32 inputs over (0, 30] -- **0
# mismatches**.  The CUDA half lives in gpuwm/core/kernels/noahmp_energy.cu
# and must stay in step with this.
# ---------------------------------------------------------------------------

_EXPM1F_ONE = F(1.0)
_EXPM1F_HUGE = F(1.0e30)
_EXPM1F_TINY = F(1.0e-30)
_EXPM1F_O_THRESHOLD = F(8.8721679688e+01)
_EXPM1F_LN2_HI = F(6.9313812256e-01)
_EXPM1F_LN2_LO = F(9.0580006145e-06)
_EXPM1F_INVLN2 = F(1.4426950216e+00)
_EXPM1F_Q = (
    F(-3.3333335072e-02),
    F(1.5873016091e-03),
    F(-7.9365076090e-05),
    F(4.0082177293e-06),
    F(-2.0109921195e-07),
)


def expm1f(x) -> np.float32:
    """glibc 2.39 ``expm1f``.  Only reached through :func:`tanhf`."""

    x = F(x)
    one = _EXPM1F_ONE
    hx = _asuint(x)
    xsb = hx & 0x80000000
    hx &= 0x7FFFFFFF
    c = F(0.0)

    if hx >= 0x4195B844:                                # |x| >= 27*ln2
        if hx >= 0x42B17218:                            # |x| >= 88.721...
            if hx > 0x7F800000:
                return F(x + x)                         # NaN
            if hx == 0x7F800000:
                return x if xsb == 0 else F(-1.0)
            if x > _EXPM1F_O_THRESHOLD:
                # glibc returns huge*huge, i.e. +inf with the overflow flag
                # raised.  numpy would warn about the multiply; the value is
                # the point, not the warning.
                with np.errstate(over="ignore"):
                    return F(_EXPM1F_HUGE * _EXPM1F_HUGE)
        if xsb != 0:
            return F(_EXPM1F_TINY - one)                # x < -27*ln2 -> -1

    if hx > 0x3EB17218:                                 # |x| > 0.5*ln2
        if hx < 0x3F851592:                             # |x| < 1.5*ln2
            if xsb == 0:
                hi = F(x - _EXPM1F_LN2_HI)
                lo = _EXPM1F_LN2_LO
                k = 1
            else:
                hi = F(x + _EXPM1F_LN2_HI)
                lo = F(-_EXPM1F_LN2_LO)
                k = -1
        else:
            k = int(F(F(_EXPM1F_INVLN2 * x)
                      + (F(0.5) if xsb == 0 else F(-0.5))))
            t = F(k)
            hi = F(x - F(t * _EXPM1F_LN2_HI))           # t*ln2_hi is exact
            lo = F(t * _EXPM1F_LN2_LO)
        x = F(hi - lo)
        c = F(F(hi - x) - lo)
    elif hx < 0x33000000:                               # |x| < 2**-25
        t = F(_EXPM1F_HUGE + x)
        return F(x - F(t - F(_EXPM1F_HUGE + x)))
    else:
        k = 0

    hfx = F(F(0.5) * x)
    hxs = F(x * hfx)
    q1, q2, q3, q4, q5 = _EXPM1F_Q
    r1 = F(hxs * q5)
    r1 = F(hxs * F(q4 + r1))
    r1 = F(hxs * F(q3 + r1))
    r1 = F(hxs * F(q2 + r1))
    r1 = F(hxs * F(q1 + r1))
    r1 = F(one + r1)
    t = F(F(3.0) - F(r1 * hfx))
    e = F(hxs * F(F(r1 - t) / F(F(6.0) - F(x * t))))
    if k == 0:
        return F(x - F(F(x * e) - hxs))

    e = F(F(x * F(e - c)) - c)
    e = F(e - hxs)
    if k == -1:
        return F(F(F(0.5) * F(x - e)) - F(0.5))
    if k == 1:
        if x < F(-0.25):
            return F(F(-2.0) * F(e - F(x + F(0.5))))
        return F(one + F(F(2.0) * F(x - e)))
    if k <= -2 or k > 56:                               # exp(x)-1 suffices
        y = F(one - F(e - x))
        y = _asfloat((_asuint(y) + (k << 23)) & _U32)
        return F(y - one)
    if k < 23:
        t = _asfloat(0x3F800000 - (0x1000000 >> k))     # t = 1 - 2**-k
        y = F(t - F(e - x))
    else:
        t = _asfloat(((0x7F - k) << 23) & _U32)         # t = 2**-k
        y = F(x - F(e + t))
        y = F(y + one)
    return _asfloat((_asuint(y) + (k << 23)) & _U32)


def tanhf(x) -> np.float32:
    """glibc 2.39 ``tanhf``.  Noah-MP reaches it only through ENERGY's FSNO."""

    x = F(x)
    one = F(1.0)
    jx = _as_int32(_asuint(x))
    ix = jx & 0x7FFFFFFF
    if ix >= 0x7F800000:                                # inf or NaN
        return F(F(one / x) + one) if jx >= 0 else F(F(one / x) - one)
    if ix < 0x41B00000:                                 # |x| < 22
        if ix == 0:
            return x
        if ix < 0x24000000:                             # |x| < 2**-55
            return F(x * F(one + x))
        ax = F(abs(x))
        if ix >= 0x3F800000:                            # |x| >= 1
            t = expm1f(F(F(2.0) * ax))
            z = F(one - F(F(2.0) / F(t + F(2.0))))
        else:
            t = expm1f(F(F(-2.0) * ax))
            z = F(F(-t) / F(t + F(2.0)))
    else:
        z = F(one - _EXPM1F_TINY)
    return z if jx >= 0 else F(-z)
