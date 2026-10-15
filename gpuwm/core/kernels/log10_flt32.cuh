// gpuwm/core/kernels/log10_flt32.cuh
//
// WOOF's own float32 base-10 logarithm, gfk_log10, for units graded bitwise
// against a gfortran column oracle that calls LOG10 on default REAL.
//
// It is the same arithmetic WOOF already carries privately in mynn_pbl.cu
// (mynn_glibc_log10f): split x into 2^k * m with m in [1, 2) (or [0.5, 1)
// for k < 0), take the natural log of m with WOOF's gfk_log, and fold
//     z = y*log10_2lo + ivln10*ln(m);   result = z + y*log10_2hi
// with every product and sum pinned to one IEEE rounding (FMUL/FADD from
// glibc_flt32.cuh, which this header follows in the loader's order), so no
// contraction can move a bit under default or strict compile options.
//
// WHY IT EXISTS.  CUDA's log10f is a different function, and so is a
// correctly rounded log10: the oracle box's LOG10 (gfortran 13.3, Ubuntu
// 24.04) is NOT correctly rounded -- a sweep of every positive finite
// float measured 29,787,060 of 2,139,095,039 values differing from the
// correctly rounded result -- so neither CUDA's nor a correctly rounded
// log10 can reproduce WRF's ZET or its NccnFNC/SxFNC fits.  This function
// was swept against the oracle box's LOG10 over every positive finite float
// on an RTX PRO 6000 and matches all of them
// (tools/milbrandt2_wrf461_oracle/sweep_log10.py).
//
// Shared by milbrandt2.cu and milbrandt2_zet.cu (mp_physics = 9), which
// must stay bit-identical to each other.

__device__ float gfk_log10(float x)
{
    const float ivln10 = __int_as_float(0x3ede5bd9);
    const float log10_2hi = __int_as_float(0x3e9a2080);
    const float log10_2lo = __int_as_float(0x355427db);
    const float two25 = __int_as_float(0x4c000000);
    unsigned int hx = __float_as_uint(x);
    int k = 0;
    if ((int)hx < 0x00800000) {
        if ((hx & 0x7fffffffu) == 0u) return __int_as_float(0xff800000);
        if ((int)hx < 0) return __int_as_float(0x7fc00000);
        k -= 25;
        x = FMUL(x, two25);
        hx = __float_as_uint(x);
    }
    if (hx >= 0x7f800000u) return FADD(x, x);
    k += (int)(hx >> 23) - 127;
    int i = (k < 0) ? 1 : 0;
    hx = (hx & 0x007fffffu) | ((unsigned int)(0x7f - i) << 23);
    float y = (float)(k + i);
    x = __uint_as_float(hx);
    float z = FADD(FMUL(y, log10_2lo), FMUL(ivln10, gfk_log(x)));
    return FADD(z, FMUL(y, log10_2hi));
}
