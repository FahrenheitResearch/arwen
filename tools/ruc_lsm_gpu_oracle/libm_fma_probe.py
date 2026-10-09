#!/usr/bin/env python3
"""Diagnostic: WOOF's shared float32 expf against the C library's FMA build.

``libm_sweep.py`` finds WOOF's ``gfk_exp`` (glibc_flt32.cuh, shared by every
bitwise-graded unit) one word off the C library at one argument of
2,218,786,818 in [-40, 40] (x = 32.564632, 1 ULP).  This evaluates WOOF's
expf core twice on the device -- as written, every product rounded, and with
the products contracted into fused multiply-adds, which is how the C
library's x86-64 build for FMA-capable CPUs (the one gfortran's EXP reaches on
box W1) evaluates the same algorithm -- over the same arguments.  Measured on
W1: as written 1 miss, contracted 0.  Diagnostic only; nothing imports it.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import libm_sweep as S  # noqa: E402

KERNEL = r"""
__device__ double fma_exp2_core(double xd_unused, float x)
{
    double xd = (double)x;
    const double invln2n = 0x1.71547652b82fep+0 * 32.0;
    double kd = __fma_rn(invln2n, xd, GFK_EXP2F_SHIFT);
    unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd = DSUB(kd, GFK_EXP2F_SHIFT);
    double r = __fma_rn(invln2n, xd, -kd);
    unsigned long long t = GFK_EXP2F_TAB[ki & 31ULL];
    t += ki << (52 - 5);
    double s = __longlong_as_double((long long)t);
    double z = __fma_rn(GFK_EXP2F_P0 / 32.0 / 32.0 / 32.0, r, GFK_EXP2F_P1 / 32.0 / 32.0);
    double r2 = DMUL(r, r);
    double y = __fma_rn(GFK_EXP2F_P2 / 32.0, r, 1.0);
    y = __fma_rn(z, r2, y);
    return DMUL(y, s);
}
extern "C" __global__ void probe(const float* x, float* a, float* b, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = gfk_exp(x[i]);
    float v = x[i];
    unsigned int abstop = (__float_as_uint(v) >> 20) & 0x7ffu;
    if (abstop >= ((__float_as_uint(88.0f)) >> 20)) { b[i] = gfk_exp(v); return; }
    b[i] = gfk_d2f_rn(fma_exp2_core(0.0, v));
}
"""


def main():
    import cupy as cp
    from gpuwm.core.kernels import module_source
    mod = cp.RawModule(code=module_source("ruc") + KERNEL, options=("-std=c++17",))
    k = mod.get_function("probe")
    lib = S.libc()
    tot = da = db = 0
    ex = []
    for x in S.chunks(S.float_range(-40.0, 40.0)):
        ref = S.call_c(lib, "expf", x).view(np.uint32)
        dx = cp.asarray(x); a = cp.empty_like(dx); b = cp.empty_like(dx)
        n = x.size
        k(((n + 255) // 256,), (256,), (dx, a, b, np.int64(n)))
        a = cp.asnumpy(a).view(np.uint32); b = cp.asnumpy(b).view(np.uint32)
        tot += n
        da += int((a != ref).sum()); db += int((b != ref).sum())
        for i in np.nonzero((a != ref) | (b != ref))[0][:4]:
            ex.append((float(x[i]), hex(ref[i]), hex(a[i]), hex(b[i])))
    print(f"expf over {tot}: as-written misses {da}, fma-contracted misses {db}")
    print(ex[:10])


if __name__ == "__main__":
    main()
