"""The primitives the vectorised Noah-MP composition stands on.

Every claim in :mod:`gpuwm.core.noahmp_slab_libm`'s docstring is a claim about
*bit patterns*, and each one is either measured here against
:mod:`gpuwm.core.noahmp_libm` -- the scalar CPython transcription the
unmodified-WRF fixtures pin -- or it is not made.

Three of these tests carry a negative control, because the whole file is one
long assertion that array arithmetic can stand in for scalar arithmetic, and
a gate that has never been observed to fail is not evidence:

* ``cupy.minimum`` is shown FAILING the signed-zero comparison that
  :func:`~gpuwm.core.noahmp_slab_libm.fmn` passes;
* numpy's own float32 ``power`` is shown FAILING the same sweep the glibc
  ``powf`` wrapper passes;
* CuPy's ``cupy.exp`` is shown FAILING the ``expf`` sweep.

Without those three, "the wrapper matches" would be consistent with the gate
comparing nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

from gpuwm.core import noahmp_libm as scalar
from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.core.noahmp_energy import _mn, _mx


def _differing_bits(a, b):
    """Boolean mask of lanes whose 32-bit patterns differ.

    The pass gates in this file mean *bit-for-bit* -- ``-0.0`` against
    ``+0.0`` must fail them -- so they compare patterns directly rather than
    through a ULP distance, which folds the two zeros onto one key.  The
    distance itself comes from :mod:`gpuwm.core.fp32_ulp`, the one module
    allowed to derive the float32 total ordering; this file used to carry its
    own copy, with the exact sign error that module's docstring documents.
    """
    return a.view(np.int32) != b.view(np.int32)


# ---------------------------------------------------------------------------
# the arguments the composition actually evaluates
# ---------------------------------------------------------------------------

def _powf_arguments():
    """(base, exponent) pairs spanning ENERGY's and SFLX's live powf calls.

    :2057 ``UU**2.0`` and ``VV**2.0``; :2071 ``(BDSNO/100)**MFSNO``; :2186
    ``(1-SAT)**RSURF_EXP``; :2189 ``(1-SMCWLT/SMCMAX)**(2+3/BEXP)``; :2201
    ``(MAX(0.01,SH2O)/SMCMAX)**(-BEXP)``; :2338 ``(...)**0.25``.
    """
    rng = np.random.default_rng(20260727)
    bases = np.concatenate([
        rng.uniform(-30.0, 30.0, 4096),          # UU, VV
        rng.uniform(0.0, 6.0, 4096),             # BDSNO/100
        rng.uniform(0.0, 1.0, 4096),             # 1 - SAT
        rng.uniform(0.0, 1.0, 4096),             # 1 - SMCWLT/SMCMAX
        rng.uniform(0.01, 1.0, 4096),            # MAX(0.01,SH2O)/SMCMAX
        rng.uniform(1.0e4, 1.0e6, 4096),         # the TRAD quotient
        np.array([0.0, 1.0, 2.0, 0.5, 1.0e-6, 1.0e6]),
    ]).astype(np.float32)
    exponents = np.concatenate([
        np.full(4096, 2.0),
        rng.uniform(1.0, 5.0, 4096),             # MFSNO
        rng.uniform(1.0, 8.0, 4096),             # RSURF_EXP
        rng.uniform(2.1, 3.5, 4096),             # 2 + 3/BEXP
        rng.uniform(-12.0, -2.0, 4096),          # -BEXP
        np.full(4096, 0.25),
        np.array([2.0, 2.0, 0.25, 3.0, 0.25, 0.25]),
    ]).astype(np.float32)
    assert bases.shape == exponents.shape
    return bases, exponents


def _expf_arguments():
    """:2140 ``-(ELAI+ESAI)``, :2186's exponent, :2203's ``PSI*GRAV/(RW*TG)``."""
    rng = np.random.default_rng(20260728)
    return np.concatenate([
        rng.uniform(-8.0, 0.0, 8192),
        rng.uniform(-40.0, 4.0, 8192),
        np.array([0.0, -0.0, 1.0, -1.0, 88.0, -88.0]),
    ]).astype(np.float32)


def _tanhf_arguments():
    """:2072's ``SNOWH / (SCFFAC*FMELT)``, which is non-negative and unbounded."""
    rng = np.random.default_rng(20260729)
    return np.concatenate([
        rng.uniform(0.0, 25.0, 8192),
        rng.uniform(0.0, 1.0, 8192),
        np.array([0.0, 1.75, 1.7499999, 1.7500001, 22.0]),
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# the transcendentals
# ---------------------------------------------------------------------------

@requires_gpu
def test_powf_reproduces_the_scalar_glibc_transcription():
    import cupy as cp
    from gpuwm.core import noahmp_slab_libm as slab

    bases, exponents = _powf_arguments()
    want = np.array([scalar.powf(b, e) for b, e in zip(bases, exponents)],
                    dtype=np.float32)
    got = cp.asnumpy(slab.slab_powf(cp.asarray(bases), cp.asarray(exponents)))
    bad = np.argwhere(_differing_bits(got, want)).ravel()
    assert bad.size == 0, (
        f"{bad.size}/{bases.size} powf arguments differ; first is "
        f"powf({bases[bad[0]]!r}, {exponents[bad[0]]!r})")


@requires_gpu
def test_numpys_own_float32_power_fails_the_powf_sweep():
    """The negative control for the test above.

    numpy's float32 ``power`` is not glibc's ``powf``.  If it passed, the
    sweep would not be observing the transcription at all.
    """
    bases, exponents = _powf_arguments()
    want = np.array([scalar.powf(b, e) for b, e in zip(bases, exponents)],
                    dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        naive = np.power(bases, exponents, dtype=np.float32)
    finite = np.isfinite(want) & np.isfinite(naive)
    assert (fp32_ulp_distance(naive[finite], want[finite]) != 0).any(), (
        "numpy.power agreed with glibc powf everywhere in this sweep, so the "
        "sweep cannot tell the two apart and proves nothing")


@requires_gpu
def test_expf_reproduces_the_scalar_glibc_transcription():
    import cupy as cp
    from gpuwm.core import noahmp_slab_libm as slab

    x = _expf_arguments()
    want = np.array([scalar.expf(v) for v in x], dtype=np.float32)
    got = cp.asnumpy(slab.slab_expf(cp.asarray(x)))
    bad = np.argwhere(_differing_bits(got, want)).ravel()
    assert bad.size == 0, (
        f"{bad.size}/{x.size} expf arguments differ; first is {x[bad[0]]!r}")


@requires_gpu
def test_cupys_own_exp_fails_the_expf_sweep():
    """The negative control for the test above."""
    import cupy as cp
    x = _expf_arguments()
    want = np.array([scalar.expf(v) for v in x], dtype=np.float32)
    naive = cp.asnumpy(cp.exp(cp.asarray(x)))
    assert (fp32_ulp_distance(naive, want) != 0).any(), (
        "cupy.exp agreed with glibc expf everywhere in this sweep, so the "
        "sweep cannot tell the two apart and proves nothing")


@requires_gpu
def test_tanhf_reproduces_the_scalar_glibc_transcription():
    import cupy as cp
    from gpuwm.core import noahmp_slab_libm as slab

    x = _tanhf_arguments()
    want = np.array([scalar.tanhf(v) for v in x], dtype=np.float32)
    got = cp.asnumpy(slab.slab_tanhf(cp.asarray(x)))
    bad = np.argwhere(_differing_bits(got, want)).ravel()
    assert bad.size == 0, (
        f"{bad.size}/{x.size} tanhf arguments differ; first is {x[bad[0]]!r}")


@requires_gpu
def test_logf_reproduces_the_scalar_glibc_transcription():
    import cupy as cp
    from gpuwm.core import noahmp_slab_libm as slab

    rng = np.random.default_rng(20260730)
    x = np.concatenate([rng.uniform(1e-6, 1e4, 8192),
                        np.array([1.0, 2.0, 0.5, 1e-6])]).astype(np.float32)
    want = np.array([scalar.logf(v) for v in x], dtype=np.float32)
    got = cp.asnumpy(slab.slab_logf(cp.asarray(x)))
    assert _differing_bits(got, want).sum() == 0


# ---------------------------------------------------------------------------
# the four operations that need no wrapper, and the two that do
# ---------------------------------------------------------------------------

@requires_gpu
@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_ieee_agreement(op):
    """``+ - * /`` in CuPy float32 ARE ``f32(a op b)`` in CPython.

    This is the claim the whole vectorised composition rests on: if it fails,
    no amount of care in the transcription can hold, and if it holds then the
    only operations needing a wrapper are the ones this module wraps.
    """
    import cupy as cp

    rng = np.random.default_rng(20260731)
    a = np.concatenate([
        rng.uniform(-1e4, 1e4, 20000),
        rng.uniform(-1e-4, 1e-4, 20000),
        np.array([0.0, -0.0, 1.0, -1.0, 3.0, 1e-30]),
    ]).astype(np.float32)
    b = np.concatenate([
        rng.uniform(-1e4, 1e4, 20000),
        rng.uniform(-1e-4, 1e-4, 20000),
        np.array([-0.0, 0.0, 3.0, 7.0, 7.0, 3.0]),
    ]).astype(np.float32)
    f = scalar.f32
    if op == "add":
        want = np.array([f(float(x) + float(y)) for x, y in zip(a, b)], np.float32)
        got = cp.asnumpy(cp.asarray(a) + cp.asarray(b))
    elif op == "sub":
        want = np.array([f(float(x) - float(y)) for x, y in zip(a, b)], np.float32)
        got = cp.asnumpy(cp.asarray(a) - cp.asarray(b))
    elif op == "mul":
        want = np.array([f(float(x) * float(y)) for x, y in zip(a, b)], np.float32)
        got = cp.asnumpy(cp.asarray(a) * cp.asarray(b))
    else:
        keep = b != 0.0
        a, b = a[keep], b[keep]
        want = np.array([f(float(x) / float(y)) for x, y in zip(a, b)], np.float32)
        got = cp.asnumpy(cp.asarray(a) / cp.asarray(b))
    np.testing.assert_array_equal(got.view(np.int32), want.view(np.int32))


@requires_gpu
def test_sqrt_agreement():
    import cupy as cp
    from gpuwm.core import noahmp_slab_libm as slab

    rng = np.random.default_rng(20260801)
    x = np.concatenate([rng.uniform(0.0, 1e6, 20000),
                        np.array([0.0, 1.0, 2.0, 1e-30])]).astype(np.float32)
    want = np.array([scalar.sqrtf(v) for v in x], dtype=np.float32)
    got = cp.asnumpy(slab.slab_sqrtf(cp.asarray(x)))
    np.testing.assert_array_equal(got.view(np.int32), want.view(np.int32))


#: The tie inputs that separate ``a if a < b else b`` from ``cupy.minimum``.
_SIGNED_ZERO_PAIRS = ((0.0, -0.0), (-0.0, 0.0), (0.0, 0.0), (-0.0, -0.0))


@requires_gpu
def test_fmn_and_fmx_reproduce_the_scalar_tie_behaviour():
    import cupy as cp
    from gpuwm.core import noahmp_slab_libm as slab

    rng = np.random.default_rng(20260802)
    pairs = [(float(x), float(y)) for x, y in
             zip(rng.uniform(-5, 5, 4000).astype(np.float32),
                 rng.uniform(-5, 5, 4000).astype(np.float32))]
    pairs += list(_SIGNED_ZERO_PAIRS)
    a = np.array([p[0] for p in pairs], np.float32)
    b = np.array([p[1] for p in pairs], np.float32)
    want_mn = np.array([_mn(x, y) for x, y in pairs], np.float32)
    want_mx = np.array([_mx(x, y) for x, y in pairs], np.float32)
    got_mn = cp.asnumpy(slab.fmn(cp.asarray(a), cp.asarray(b)))
    got_mx = cp.asnumpy(slab.fmx(cp.asarray(a), cp.asarray(b)))
    np.testing.assert_array_equal(got_mn.view(np.int32), want_mn.view(np.int32))
    np.testing.assert_array_equal(got_mx.view(np.int32), want_mx.view(np.int32))


# ---------------------------------------------------------------------------
# the subnormal band, which is where the device libm was wrong
# ---------------------------------------------------------------------------

#: glibc's ``expf`` returns a binary32 **subnormal** for arguments in
#: ``[-103.616, -87.337)`` and zero below that.  ``__double2float_rn`` flushes
#: the whole band to zero on this toolchain, so ``r_exp`` and ``r_pow`` used to
#: disagree with the CPython authority there; ``nmp_d2f_rn`` in
#: ``noahmp_leaves.cu`` is the fix.  ENERGY's RHSUR (:2203) reaches this band
#: on very dry soil.
_SUBNORMAL_EXPF_BAND = (-103.61632918473205, -87.33654475125263)


@requires_gpu
def test_expf_is_bitwise_across_the_whole_subnormal_band():
    """Every FP32 argument whose ``expf`` is subnormal, not a sample of them."""
    import cupy as cp
    from gpuwm.core import noahmp_slab_libm as slab

    lo, hi = _SUBNORMAL_EXPF_BAND
    lo_bits = np.float32(lo).view(np.int32)
    hi_bits = np.float32(hi).view(np.int32)
    # Negative floats: a larger magnitude is a larger unsigned bit pattern, so
    # walk the exponent/mantissa ladder between the two ends inclusive.
    bits = np.arange(min(lo_bits, hi_bits), max(lo_bits, hi_bits) + 1,
                     dtype=np.int32)
    x = bits.view(np.float32)
    assert x.size > 100000, f"the band collapsed to {x.size} arguments"
    want = np.array([scalar.expf(v) for v in x], dtype=np.float32)
    assert (want != 0.0).any() and (want < np.float32(1.1754944e-38)).all(), (
        "this band is supposed to be the subnormal results; it is not")
    got = cp.asnumpy(slab.slab_expf(cp.asarray(x)))
    bad = np.argwhere(got.view(np.int32) != want.view(np.int32)).ravel()
    assert bad.size == 0, (
        f"{bad.size}/{x.size} subnormal expf results differ; first is "
        f"expf({x[bad[0]]!r}): device {got[bad[0]]!r} vs glibc "
        f"{want[bad[0]]!r}")


@requires_gpu
def test_the_hardware_conversion_still_flushes_which_is_why_the_fix_exists():
    """The negative control for the fix, kept as a live measurement.

    The name says "hardware"; the flush is cupy's appended ``-ftz=true``.
    ``nmp_d2f_rn`` is worth its lines only while this route flushes; if a
    future toolchain stops, this test fails and the helper can be retired
    -- the outcome to want, and not something a comment could tell anyone.
    """
    import cupy as cp

    source = """
    extern "C" __global__
    void probe(const double *d, float *hardware, float *fixed, int n)
    {
        int t = blockIdx.x * blockDim.x + threadIdx.x;
        if (t >= n) return;
        double a = fabs(d[t]);
        hardware[t] = __double2float_rn(d[t]);
        if (a > 0.0 && a < 1.1754943508222875e-38) {
            double scaled = rint(a * 7.1362384635297994e+44);
            unsigned int m = (unsigned int)scaled;
            unsigned int s = (__double_as_longlong(d[t]) < 0LL)
                             ? 0x80000000u : 0u;
            fixed[t] = __uint_as_float(s | m);
        } else {
            fixed[t] = __double2float_rn(d[t]);
        }
    }
    """
    module = cp.RawModule(code=source, options=("-std=c++17",))
    module.compile()
    kernel = module.get_function("probe")
    values = np.array([6.0546e-39, 1.0e-45, -6.0546e-39, 1.6458e-38],
                      dtype=np.float64)
    want = values.astype(np.float32)
    device = cp.asarray(values)
    hardware = cp.empty(values.size, cp.float32)
    fixed = cp.empty(values.size, cp.float32)
    kernel((1,), (32,), (device, hardware, fixed, np.int32(values.size)))
    hardware = cp.asnumpy(hardware)
    fixed = cp.asnumpy(fixed)
    assert (hardware.view(np.int32) != want.view(np.int32)).any(), (
        "__double2float_rn no longer flushes subnormals on this toolchain; "
        "nmp_d2f_rn in noahmp_leaves.cu can be retired")
    np.testing.assert_array_equal(fixed.view(np.int32), want.view(np.int32))


@requires_gpu
def test_cupy_minimum_fails_the_signed_zero_comparison():
    """The negative control that makes :func:`fmn` a decision, not a spelling.

    ``cupy.minimum`` is the obvious thing to write and it is wrong on exactly
    one of the four tie inputs.  Observed here so the ``where`` form is not
    mistaken for a stylistic preference.
    """
    import cupy as cp

    a = np.array([p[0] for p in _SIGNED_ZERO_PAIRS], np.float32)
    b = np.array([p[1] for p in _SIGNED_ZERO_PAIRS], np.float32)
    want = np.array([_mn(x, y) for x, y in _SIGNED_ZERO_PAIRS], np.float32)
    naive = cp.asnumpy(cp.minimum(cp.asarray(a), cp.asarray(b)))
    differing = naive.view(np.int32) != want.view(np.int32)
    assert differing.any(), (
        "cupy.minimum matched the scalar _mn on every tie, so this control "
        "proves nothing")
    # And name it: (-0.0, +0.0) is the pair that separates them.
    assert bool(differing[1]), (
        f"expected _mn(-0.0, +0.0) = {want[1]!r} to differ from "
        f"cupy.minimum's {naive[1]!r}")
