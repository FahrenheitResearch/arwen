// ======================================================================
// WOOF's own float32 log10f (gfk_log10) and cbrtf (gfk_cbrt), the
// companions of glibc_flt32.cuh's routines for units graded word for word
// against a gfortran column oracle.  Both are graded against the oracle
// host's C library over every float32 argument by
// tools/glibc_flt32_extra_proof.  Include AFTER glibc_flt32.cuh (gfk_log,
// FMUL/FADD/DMUL/DADD).  Only units that name this header compile it, so
// no other unit's source changes.
// ======================================================================
#pragma once

// The exact binary32 word of m * 2^-149 * 2^25 for a subnormal magnitude
// m = 1 .. 0x7fffff, as an integer: a host scales a subnormal argument by
// 2^25 with a multiply, which is exact, but the card flushes the
// subnormal operand of that multiply to zero under CuPy's -ftz=true.
// The leading one of m sits at bit p (0..22); the product is
// 1.f * 2^(p - 124), a normal number with biased exponent p + 3.
__device__ __forceinline__ unsigned int gfk_subnormal_times_2p25(
    unsigned int m)
{
    const int p = 31 - __clz((int)m);
    return ((unsigned int)(p + 3) << 23) | ((m << (23 - p)) & 0x7fffffu);
}

// log10f: z = y*log10_2lo + ivln10*logf(x); return z + y*log10_2hi,
// with gfk_log as logf.
__device__ float gfk_log10(float x)
{
    const float two25 = __int_as_float(0x4c000000);
    const float ivln10 = __int_as_float(0x3ede5bd9);
    const float log10_2hi = __int_as_float(0x3e9a2080);
    const float log10_2lo = __int_as_float(0x355427db);
    int hx = __float_as_int(x);
    int k = 0;
    if (hx < 0x00800000) {
        if ((hx & 0x7fffffff) == 0) return -two25 / fabsf(x);
        if (hx < 0) return (x - x) / (x - x);
        k -= 25;
        hx = (int)gfk_subnormal_times_2p25((unsigned int)hx);
        x = __int_as_float(hx);
    }
    if (hx >= 0x7f800000) return FADD(x, x);
    k += (hx >> 23) - 127;
    const int i = (int)(((unsigned int)k & 0x80000000u) >> 31);
    hx = (hx & 0x007fffff) | ((0x7f - i) << 23);
    const float y = (float)(k + i);
    x = __int_as_float(hx);
    const float z = FADD(FMUL(y, log10_2lo), FMUL(ivln10, gfk_log(x)));
    return FADD(z, FMUL(y, log10_2hi));
}

// cbrtf.  frexpf/ldexpf by exponent arithmetic on the bits (exact for
// normal arguments; a subnormal argument is first scaled by 2^25 exactly,
// by integer arithmetic so the card's flush of subnormal operands cannot
// touch it).
__device__ float gfk_cbrt(float x)
{
    const double factor[5] = {
        1.0 / 1.5874010519681994748, 1.0 / 1.2599210498948731648, 1.0,
        1.2599210498948731648, 1.5874010519681994748};
    const unsigned int ax = __float_as_uint(x) & 0x7fffffffu;
    if (ax == 0u || ax >= 0x7f800000u) return FADD(x, x);
    // frexpf(|x|): |x| = xm * 2^xe with xm in [0.5, 1).
    int xe;
    float xm;
    if (ax < 0x00800000u) {
        const unsigned int bs = gfk_subnormal_times_2p25(ax);
        xe = (int)(bs >> 23) - 126 - 25;
        xm = __uint_as_float((bs & 0x807fffffu) | 0x3f000000u);
    } else {
        xe = (int)(ax >> 23) - 126;
        xm = __uint_as_float((ax & 0x807fffffu) | 0x3f000000u);
    }
    const double xmd = (double)xm;
    const float u = (float)DADD(0.492659620528969547,
        DMUL(DSUB(0.697570460207922770,
                  DMUL(0.191502161678719066, xmd)), xmd));
    const float t2 = FMUL(FMUL(u, u), u);
    const double ud = (double)u, t2d = (double)t2;
    const float ym = (float)DMUL(DDIV(DMUL(ud, DADD(t2d, DMUL(2.0, xmd))),
                                      DADD(DMUL(2.0, t2d), xmd)),
                                 factor[2 + xe % 3]);
    // ldexpf(+-ym, xe/3): ym is normal and the result stays in range for
    // every finite float argument except subnormal results, handled by
    // scaling in two exact steps.
    // x > 0.0, read from the sign bit: the card compares a subnormal x as
    // zero under -ftz=true.
    const float sym = (__float_as_uint(x) >> 31) ? -ym : ym;
    const int e = xe / 3;
    if (e >= -126) {
        return FMUL(sym, __int_as_float((e + 127) << 23));
    }
    return FMUL(FMUL(sym, __int_as_float((e + 127 + 64) << 23)),
                __int_as_float((127 - 64) << 23));
}
