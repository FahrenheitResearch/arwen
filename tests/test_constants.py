"""The WRF constant block, and the one place its spelling was a third function.

``share/module_model_constants.F`` declares its derived constants as ``REAL
PARAMETER``.  gfortran folds a parameter expression in the *declared kind*, so
every operation in the chain is rounded to float32; it does not evaluate the
chain in double and round once at the end.  Those are two different functions,
and for ``rvovrd`` they land on two different floats.

Verified rather than assumed.  A probe compiled at ``-O0`` against the
byte-unmodified module on the pinned v4.6.1 tree
(``d66e442fccc04111067e29274c9f9eaccc3cef28``) prints ``transfer(rvovrd, i)``
as ``3FCDDED2``.  The same program routes ``r_v`` and ``r_d`` through an opaque
subroutine, so that divide must happen at runtime on the hardware, and prints
``3FCDDED2`` again -- the fold and the hardware agree.  It also evaluates
``real(461.6d0/287.0d0)`` and prints ``3FCDDED1``.  So gfortran folds in the
declared kind, and the double-then-round spelling this module used to carry was
one ULP low at 1.608 -- two ULP once ``EP_1 = rvovrd - 1`` drops the exponent.
"""

import struct

import numpy as np

from conftest import requires_gpu
from gpuwm.core import constants as c

#: gfortran 13.3.0, ``-O0``, byte-unmodified module_model_constants.F.
WRF_RVOVRD_BITS = 0x3FCDDED2
#: ``real(461.6d0/287.0d0)`` from the same probe: the spelling this replaced.
DOUBLE_ROUNDED_RVOVRD_BITS = 0x3FCDDED1
#: WRF ``EP_1 = R_v/R_d - 1.`` from the same probe, and what the double-rounded
#: quotient gives instead once one is subtracted.
WRF_EP1_BITS = 0x3F1BBDA4
DOUBLE_ROUNDED_EP1_BITS = 0x3F1BBDA2


def _bits(value) -> int:
    return struct.unpack("<I", struct.pack("<f", np.float32(value)))[0]


def test_wrf_constant_values():
    assert c.G == 9.81
    assert c.RD == 287.0
    assert c.RV == 461.6
    assert c.CP == 7.0 * c.RD / 2.0
    assert c.CV == c.CP - c.RD
    assert c.P0 == 1.0e5
    assert c.T0 == 300.0
    assert abs(c.GAMMA - c.CP / c.CV) < 1e-12
    assert set(c.CUDA_DEFINES) >= {"G", "RD", "RV", "CP", "CV", "P0", "T0",
                                   "GAMMA", "RCP", "RCV"}


def test_rvovrd_is_wrfs_float32_quotient():
    assert _bits(c.RVOVRD) == WRF_RVOVRD_BITS


def test_the_spelling_this_replaced_lands_on_a_different_float():
    """The failing form, shown rather than asserted away.

    ``RV / RD`` here is a Python *double* divide of two doubles, and 461.6 is
    not exactly representable in either format, so the quotient is not the one
    gfortran folds.  A gate whose failing form has never been observed is not
    evidence, so the old spelling is evaluated and pinned beside the new one.
    """
    old = np.float32(c.RV / c.RD)
    assert _bits(old) == DOUBLE_ROUNDED_RVOVRD_BITS
    assert _bits(old) != _bits(c.RVOVRD)
    # EP_1 keeps the absolute error while the exponent drops, so the gap
    # doubles in ULP.  This is the number YSU, MYNN and the surface layer all
    # form; ysu.cu measured hpbl 112 -> 1 and exch_h 283 -> 7 when it stopped
    # reading the double-rounded constant.
    assert _bits(np.float32(np.float32(c.RVOVRD) - np.float32(1.0))) \
        == WRF_EP1_BITS
    assert _bits(np.float32(old - np.float32(1.0))) == DOUBLE_ROUNDED_EP1_BITS


def test_every_derived_constant_matches_its_float32_chain():
    """The sweep that found ``RVOVRD``, kept as a gate over the whole block.

    Each right-hand side is WRF's own expression from
    module_model_constants.F, evaluated with a float32 rounding after every
    operation.  Seven of the eight already agreed with the double spelling by
    luck; the eighth did not, and nothing but this assertion would notice the
    next one.
    """
    f = np.float32
    rd, rv = f(c.RD), f(c.RV)
    cp = f(f(f(7.0) * rd) / f(2.0))          # cp = 7.*r_d/2.
    cv = f(cp - rd)                          # cv = cp-r_d
    chain = {
        "CP": cp,
        "CV": cv,
        "GAMMA": f(cp / f(cp - rd)),         # cpovcv = cp/(cp-r_d)
        "RCP": f(rd / cp),                   # rcp = r_d/cp
        "RCV": f(rd / cv),                   # rcv = r_d/cv
        "RVOVRD": f(rv / rd),                # rvovrd = r_v/r_d
        "RERADIUS": f(f(1.0) / f(6370.0e3)),  # reradius = 1./6370.0e03
        "EP2": f(rd / rv),                   # EP_2 = R_d/R_v
    }
    for name, want in chain.items():
        assert _bits(getattr(c, name)) == _bits(want), name


@requires_gpu
def test_the_kernel_preamble_delivers_wrfs_word_to_the_device():
    """What ``#define`` actually compiles to, not what Python holds.

    Two separate things could still go wrong between the constant and the
    kernel: ``repr`` could lose a digit on the way into the macro, and ptxas
    12.x is known to mis-fold FP32 at compile time.  This reads the value back
    out of a running kernel.  The truncated control is there so the assertion
    is known to be capable of failing.
    """
    import cupy as cp

    from gpuwm.core.kernels import CUDA_DEFINES

    preamble = "\n".join(f"#define {k} {float(v)!r}f"
                         for k, v in CUDA_DEFINES.items())
    src = preamble + r"""
#define TRUNCATED_CONTROL 1.60836f
extern "C" __global__ void read_back(unsigned int *out) {
    out[0] = __float_as_uint(RVOVRD);
    out[1] = __float_as_uint(RV / RD);            // nvcc folds the divide
    out[2] = __float_as_uint(RVOVRD - 1.0f);
    out[3] = __float_as_uint(TRUNCATED_CONTROL);
}
"""
    module = cp.RawModule(code=src, options=("-std=c++17",))
    out = cp.zeros(4, cp.uint32)
    module.get_function("read_back")((1,), (1,), (out,))
    got = [int(v) for v in cp.asnumpy(out)]
    assert got[0] == WRF_RVOVRD_BITS
    # nvcc's compile-time fold of the float32 divide agrees with the macro,
    # so ysu.cu's ``RV / RD - 1.0f`` and the macro are now the same number.
    assert got[1] == WRF_RVOVRD_BITS
    assert got[2] == WRF_EP1_BITS
    # ...and the control proves the read-back can tell values apart at all.
    assert got[3] != WRF_RVOVRD_BITS
