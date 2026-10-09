// thompson_aerosol_libm.cuh -- WOOF's own float32 and binary64 libm words
// for the mp=28 (aerosol-aware Thompson) units.  Prepended to every
// thompson_aerosol_* unit after glibc_flt32.cuh and glibc_flt64.cuh and
// before thompson_aerosol_common.cuh (gpuwm/core/kernels/__init__.py
// _EXTRA_HEADERS).  Self-contained apart from those two headers, so
// tools/thompson_aerosol_column_oracle/libm_check.cpp compiles it for the
// host and compares every word with the oracle host's C library.
#pragma once

// ---------------------------------------------------------------------------
// WOOF's own libm words for WRF's EXP, LOG, LOG10 and ** (the 0 ULP column
// oracle, tools/thompson_aerosol_column_oracle).
// ---------------------------------------------------------------------------
//
// THE BREAKAGE THIS PREVENTS.  WRF's module_mp_thompson.F is compiled by
// gfortran, which lowers a REAL(4) EXP/ALOG/ALOG10/** to the C library's
// expf/logf/log10f/powf and a DOUBLE PRECISION one to exp/log/log10/pow.
// On the x86-64 oracle host (W1: glibc 2.39 on a CPU with FMA, so the
// library takes its FMA code path) those words are NOT correctly rounded:
// expf and powf differ from the correctly rounded result on some
// arguments, and so did this header's earlier "evaluate in double and
// round once" helpers.  CUDA's expf/logf/powf/exp/log/pow/log10 are
// different functions again.  The column oracle measured the result: no
// column of 153 was bit-identical to WRF, under strict or default
// arithmetic.
//
// So every mp=28 unit evaluates these through the words below, in every
// arithmetic mode (a correctness remedy is default-on):
//
//   thompson_aa_expf(x)     float32 exp: the FMA path's operation order on
//                           the 32-entry exp2 table GFK_EXP2F_TAB
//                           (glibc_flt32.cuh, prepended to these units)
//   thompson_aa_logf(x)     gfk_log
//   thompson_aa_powf(x, y)  float32 pow: the FMA path's log2 and exp2 cores
//                           on GFK_POWF_INVC/LOGC/A and GFK_EXP2F_TAB
//   thompson_aa_log10f(x)   k*log10_2lo + ivln10*logf(m), then + k*log10_2hi
//   thompson_aa_exp/log/pow (binary64)  glibc_exp/glibc_log/glibc_pow
//                           (glibc_flt64.cuh, prepended to these units)
//   thompson_aa_log10(x)    the same composition in binary64 on
//                           thompson_aa_log
//
// MEASURED on W1 against the host's own libm, compiled for the host with
// tools/thompson_real_column_parity/cuda_host_shim.h (-ffp-contract=off):
// expf, logf and log10f over all 2^32 float32 inputs; powf, exp, log, pow
// and log10 on 2e8 sampled arguments each; zero mismatches
// (tools/thompson_aerosol_column_oracle/libm_check.cpp).  The device
// evaluates the same operations (every binary64 product-sum below is an
// explicit __fma_rn or a pinned __dmul_rn/__dadd_rn), so it returns the
// same words.  CUDA's own expf/logf/powf/log10f/exp/log/pow/log10 must not
// be called from an mp=28 unit; tests/test_thompson_aerosol_libm.py reads
// the sources for them.
__device__ __forceinline__ float thompson_aa_d2f(double y)
{
    return gfk_d2f_rn(y);
}

__device__ __forceinline__ float thompson_aa_expf(float x)
{
    const unsigned int abstop = (__float_as_uint(x) >> 20) & 0x7ffu;
    if (abstop >= ((__float_as_uint(88.0f)) >> 20)) {
        if (__float_as_uint(x) == 0xff800000u) return 0.0f;
        if (abstop >= (0x7f800000u >> 20)) return __fadd_rn(x, x);
        if (x > __int_as_float(0x42b17218)) return __int_as_float(0x7f800000);
        if (x < -__int_as_float(0x42cff1b4)) return 0.0f;
    }
    const double xd = (double)x;
    const double inv = 0x1.71547652b82fep+0 * 32.0;
    double kd = __fma_rn(inv, xd, GFK_EXP2F_SHIFT);
    const unsigned long long ki =
        (unsigned long long)__double_as_longlong(kd);
    kd = __dsub_rn(kd, GFK_EXP2F_SHIFT);
    const double r = __fma_rn(inv, xd, -kd);
    unsigned long long t = GFK_EXP2F_TAB[ki & 31ULL];
    t += ki << (52 - 5);
    const double s = __longlong_as_double((long long)t);
    const double z = __fma_rn(GFK_EXP2F_P0 / 32.0 / 32.0 / 32.0, r,
                              GFK_EXP2F_P1 / 32.0 / 32.0);
    const double r2 = __dmul_rn(r, r);
    double y = __fma_rn(GFK_EXP2F_P2 / 32.0, r, 1.0);
    y = __fma_rn(z, r2, y);
    return thompson_aa_d2f(__dmul_rn(y, s));
}

__device__ __forceinline__ float thompson_aa_logf(float x)
{
    return gfk_log(x);
}

__device__ __forceinline__ double thompson_aa_powf_log2(unsigned int ix)
{
    const unsigned int tmp = ix - 0x3f330000u;
    const int i = (int)((tmp >> 19) & 15u);
    const unsigned int top = tmp & 0xff800000u;
    const unsigned int iz = ix - top;
    const int k = (int)top >> 23;
    const double z = (double)__uint_as_float(iz);
    const double r = __fma_rn(z, GFK_POWF_INVC[i], -1.0);
    const double y0 = __dadd_rn(GFK_POWF_LOGC[i], (double)k);
    const double r2 = __dmul_rn(r, r);
    const double y = __fma_rn(GFK_POWF_A[0], r, GFK_POWF_A[1]);
    const double p = __fma_rn(GFK_POWF_A[2], r, GFK_POWF_A[3]);
    const double r4 = __dmul_rn(r2, r2);
    double q = __fma_rn(GFK_POWF_A[4], r, y0);
    q = __fma_rn(p, r2, q);
    return __fma_rn(y, r4, q);
}

__device__ __forceinline__ double thompson_aa_powf_exp2(
        double xd, unsigned long long sign_bias)
{
    double kd = __dadd_rn(xd, GFK_EXP2F_SHIFT_SCALED);
    const unsigned long long ki =
        (unsigned long long)__double_as_longlong(kd);
    kd = __dsub_rn(kd, GFK_EXP2F_SHIFT_SCALED);
    const double r = __dsub_rn(xd, kd);
    unsigned long long t = GFK_EXP2F_TAB[ki & 31ULL];
    t += (ki + sign_bias) << (52 - 5);
    const double s = __longlong_as_double((long long)t);
    const double z = __fma_rn(GFK_EXP2F_P0, r, GFK_EXP2F_P1);
    const double r2 = __dmul_rn(r, r);
    double y = __fma_rn(GFK_EXP2F_P2, r, 1.0);
    y = __fma_rn(z, r2, y);
    return __dmul_rn(y, s);
}

__device__ __forceinline__ float thompson_aa_powf(float x, float y)
{
    unsigned int sign_bias = 0u;
    unsigned int ix = __float_as_uint(x);
    const unsigned int iy = __float_as_uint(y);
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u || gfk_zeroinfnan(iy)) {
        if (gfk_zeroinfnan(iy)) {
            if (2u * iy == 0u) return 1.0f;
            if (ix == 0x3f800000u) return 1.0f;
            if (2u * ix > 2u * 0x7f800000u || 2u * iy > 2u * 0x7f800000u)
                return __fadd_rn(x, y);
            if (2u * ix == 2u * 0x3f800000u) return 1.0f;
            if ((2u * ix < 2u * 0x3f800000u) == !(iy & 0x80000000u))
                return 0.0f;
            return __fmul_rn(y, y);
        }
        if (gfk_zeroinfnan(ix)) {
            float x2 = __fmul_rn(x, x);
            if ((ix & 0x80000000u) && gfk_checkint(iy) == 1) x2 = -x2;
            return (iy & 0x80000000u) ? __fdiv_rn(1.0f, x2) : x2;
        }
        if (ix & 0x80000000u) {
            const int yint = gfk_checkint(iy);
            if (yint == 0) return __int_as_float(0x7fc00000);
            if (yint == 1) sign_bias = 1u << (5 + 11);
            ix &= 0x7fffffffu;
        }
        if (ix < 0x00800000u) {
            ix = __float_as_uint(__fmul_rn(x, 8388608.0f)) & 0x7fffffffu;
            ix -= 23u << 23;
        }
    }
    const double logx = thompson_aa_powf_log2(ix);
    const double ylogx = __dmul_rn((double)y, logx);
    const unsigned int hi = (unsigned int)
        (((unsigned long long)__double_as_longlong(ylogx) >> 47) & 0xffffULL);
    if (hi >= (unsigned int)
            (((unsigned long long)__double_as_longlong(126.0) >> 47)
             & 0xffffULL)) {
        if (ylogx > 0x1.fffffffd1d571p+6)
            return sign_bias ? __int_as_float(0xff800000)
                             : __int_as_float(0x7f800000);
        if (ylogx <= -150.0) return sign_bias ? -0.0f : 0.0f;
    }
    return thompson_aa_d2f(
        thompson_aa_powf_exp2(ylogx, (unsigned long long)sign_bias));
}

__device__ __forceinline__ float thompson_aa_log10f(float x)
{
    unsigned int hx = __float_as_uint(x);
    int k = 0;
    if (hx < 0x00800000u || (hx & 0x80000000u)) {
        if ((hx & 0x7fffffffu) == 0u) return __int_as_float(0xff800000);
        if (hx & 0x80000000u) return __int_as_float(0x7fc00000);
        k -= 25;                                  // subnormal: scale by 2^25
        hx = __float_as_uint(__fmul_rn(x, 33554432.0f));
    }
    if (hx > 0x7f7fffffu) return __fadd_rn(x, x);
    k += (int)(hx >> 23) - 127;
    const int i = (int)(((unsigned int)k & 0x80000000u) >> 31);
    hx = (hx & 0x007fffffu) | ((unsigned int)(0x7f - i) << 23);
    const float y = (float)(k + i);
    const float z = __fadd_rn(
        __fmul_rn(__uint_as_float(0x3ede5bd9u),                 // ivln10
                  thompson_aa_logf(__uint_as_float(hx))),
        __fmul_rn(y, __uint_as_float(0x355427dbu)));            // log10_2lo
    return __fadd_rn(z, __fmul_rn(y, __uint_as_float(0x3e9a2080u)));
}

__device__ __forceinline__ double thompson_aa_exp(double x)
{
    return glibc_exp(x);
}

__device__ __forceinline__ double thompson_aa_log(double x)
{
    return glibc_log(x);
}

__device__ __forceinline__ double thompson_aa_pow(double x, double y)
{
    return glibc_pow(x, y);
}

__device__ __forceinline__ double thompson_aa_log10(double x)
{
    unsigned long long hx = (unsigned long long)__double_as_longlong(x);
    long long k;
    if (hx <= 0x000fffffffffffffULL || (hx >> 63)) {
        if ((hx & 0x7fffffffffffffffULL) == 0ULL)
            return __longlong_as_double((long long)0xfff0000000000000ULL);
        if (hx >> 63)
            return __longlong_as_double((long long)0x7ff8000000000000ULL);
        k = -1023 - 54;                           // subnormal: scale by 2^54
        hx = (unsigned long long)__double_as_longlong(__dmul_rn(x, 0x1p54));
    } else {
        k = -1023;
    }
    if (hx > 0x7fefffffffffffffULL) return __dadd_rn(x, x);
    k += (long long)(hx >> 52);
    const long long i = (long long)(((unsigned long long)k) >> 63);
    k += i;
    hx = (hx & 0x000fffffffffffffULL)
        | ((unsigned long long)(0x3ff - i) << 52);
    const double y = (double)k;
    const double z = __dadd_rn(
        __dmul_rn(0x1.bcb7b1526e50ep-2,                          // ivln10
                  thompson_aa_log(__longlong_as_double((long long)hx))),
        __dmul_rn(y, 0x1.9fef311f12b36p-42));                    // log10_2lo
    return __dadd_rn(z, __dmul_rn(y, 0x1.34413509f6000p-2));     // log10_2hi
}

// ---------------------------------------------------------------------------
// hypot, log1p and atan2 for the complex arithmetic of WRF's melting-snow
// reflectivity.  module_mp_radar.F takes ABS, LOG and SQRT of COMPLEX*16
// values, which gfortran lowers to the C library's cabs, clog and csqrt; on
// the oracle host those call hypot, log1p, log and atan2.  Only the mp=28
// reflectivity (thompson_aa_refl10cm in thompson_aerosol_state.cu) reaches
// these three words.
//
//   thompson_aa_hypot   a correction step on sqrt(x*x + y*y) with no fused
//                       multiply-add, the operation order of the oracle
//                       host's hypot: zero mismatches on the sampled
//                       moderate range (libm_check.cpp).  Arguments whose
//                       ratio exceeds 2^54 return the larger plus the
//                       smaller, as the host does; arguments beyond
//                       1e308 are outside the reflectivity's domain.
//   thompson_aa_log1p   the oracle host runs its log1p fused-multiply-add
//                       path (W1's CPU has FMA); every fusion of that path
//                       is an explicit __fma_rn here, the rest pinned
//                       __dmul_rn/__dadd_rn: zero mismatches on the sampled
//                       range (libm_check.cpp).
//   thompson_aa_atan2   a reduction to |t| <= 7/16 on four breakpoints and
//                       an odd minimax polynomial, under 1.5 ulp.  The
//                       host's atan2 is a different and more accurate
//                       routine, so the two agree on most arguments but not
//                       all; libm_check.cpp reports the agreement rate and
//                       the worst error against a 113-bit reference.  It
//                       feeds only the argument of a complex logarithm in a
//                       binary64 chain that WRF rounds to REAL(4) at the end
//                       (ze_snow = SNGL(...) at module_mp_thompson.F:5950),
//                       which a few binary64 ulp do not move; the column
//                       oracle's melting-snow cells are the measurement.
// ---------------------------------------------------------------------------
__device__ __forceinline__ double thompson_aa_hypot(double x, double y)
{
    const double fx = fabs(x), fy = fabs(y);
    const double ax = fx < fy ? fy : fx;
    const double ay = fx < fy ? fx : fy;
    if (!(ax < 1.0e308) || ax == 0.0) return __dadd_rn(ax, ay);
    if (ay <= __dmul_rn(ax, 0x1p-54)) return __dadd_rn(ax, ay);
    const double h = __dsqrt_rn(__dadd_rn(__dmul_rn(ax, ax),
                                          __dmul_rn(ay, ay)));
    double t1, t2;
    if (h <= __dmul_rn(2.0, ay)) {
        const double delta = __dsub_rn(h, ay);
        t1 = __dmul_rn(ax, __dsub_rn(__dmul_rn(2.0, delta), ax));
        t2 = __dmul_rn(__dsub_rn(delta, __dmul_rn(2.0, __dsub_rn(ax, ay))),
                       delta);
    } else {
        const double delta = __dsub_rn(h, ax);
        t1 = __dmul_rn(__dmul_rn(2.0, delta),
                       __dsub_rn(ax, __dmul_rn(2.0, ay)));
        t2 = __dadd_rn(__dmul_rn(__dsub_rn(__dmul_rn(4.0, delta), ay), ay),
                       __dmul_rn(delta, delta));
    }
    return __dsub_rn(h, __ddiv_rn(__dadd_rn(t1, t2), __dmul_rn(2.0, h)));
}

__device__ __forceinline__ double thompson_aa_log1p(double x)
{
    const double ln2_hi = 6.93147180369123816490e-01;
    const double ln2_lo = 1.90821492927058770002e-10;
    const double lp1 = 6.666666666666735130e-01;
    const double lp2 = 3.999999999940941908e-01;
    const double lp3 = 2.857142874366239149e-01;
    const double lp4 = 2.222219843214978396e-01;
    const double lp5 = 1.818357216161805012e-01;
    const double lp6 = 1.531383769920937332e-01;
    const double lp7 = 1.479819860511658591e-01;
    const int hx = (int)((unsigned long long)__double_as_longlong(x) >> 32);
    const int ax = hx & 0x7fffffff;
    int k = 1, hu = 0;
    double f = 0.0, c = 0.0;
    if (hx < 0x3FDA827A) {
        if (ax >= 0x3ff00000) {
            if (x == -1.0)
                return __longlong_as_double((long long)0xfff0000000000000ULL);
            return __longlong_as_double((long long)0x7ff8000000000000ULL);
        }
        if (ax < 0x3e200000) {
            if (ax < 0x3c900000) return x;
            return __fma_rn(-__dmul_rn(x, x), 0.5, x);
        }
        if (hx > 0 || hx <= (int)0xbfd2bec4) { k = 0; f = x; hu = 1; }
    } else if (hx >= 0x7ff00000) {
        return __dadd_rn(x, x);
    }
    if (k != 0) {
        double u;
        if (hx < 0x43400000) {
            u = __dadd_rn(1.0, x);
            hu = (int)((unsigned long long)__double_as_longlong(u) >> 32);
            k = (hu >> 20) - 1023;
            c = k > 0 ? __dsub_rn(1.0, __dsub_rn(u, x))
                      : __dsub_rn(x, __dsub_rn(u, 1.0));
            c = __ddiv_rn(c, u);
        } else {
            u = x;
            hu = (int)((unsigned long long)__double_as_longlong(u) >> 32);
            k = (hu >> 20) - 1023;
            c = 0.0;
        }
        hu &= 0x000fffff;
        const unsigned long long low =
            (unsigned long long)__double_as_longlong(u) & 0xffffffffULL;
        if (hu < 0x6a09e) {
            u = __longlong_as_double((long long)(
                ((unsigned long long)(hu | 0x3ff00000) << 32) | low));
        } else {
            k += 1;
            u = __longlong_as_double((long long)(
                ((unsigned long long)(hu | 0x3fe00000) << 32) | low));
            hu = (0x00100000 - hu) >> 2;
        }
        f = __dsub_rn(u, 1.0);
    }
    const double dk = (double)k;
    const double hfsq = __dmul_rn(__dmul_rn(0.5, f), f);
    if (hu == 0) {
        if (f == 0.0) {
            if (k == 0) return 0.0;
            return __fma_rn(dk, ln2_hi, __fma_rn(dk, ln2_lo, c));
        }
        const double r = __dmul_rn(__fma_rn(-0.66666666666666666, f, 1.0),
                                   hfsq);
        if (k == 0) return __dsub_rn(f, r);
        return __fma_rn(dk, ln2_hi, -__dsub_rn(
            __dsub_rn(r, __fma_rn(dk, ln2_lo, c)), f));
    }
    const double s = __ddiv_rn(f, __dadd_rn(2.0, f));
    const double z = __dmul_rn(s, s);
    const double r2 = __fma_rn(z, lp3, lp2);
    const double r3 = __fma_rn(z, lp5, lp4);
    const double r4 = __fma_rn(z, lp7, lp6);
    const double z2 = __dmul_rn(z, z);
    const double z4 = __dmul_rn(z2, z2);
    const double z6 = __dmul_rn(z4, z2);
    double r = __fma_rn(z, lp1, __dmul_rn(z2, r2));
    r = __fma_rn(z4, r3, r);
    r = __fma_rn(z6, r4, r);
    const double t = __dmul_rn(__dadd_rn(hfsq, r), s);
    if (k == 0) return __dsub_rn(f, __dsub_rn(hfsq, t));
    return __fma_rn(dk, ln2_hi, -__dsub_rn(__dsub_rn(hfsq,
        __dadd_rn(__fma_rn(dk, ln2_lo, c), t)), f));
}

__device__ __forceinline__ double thompson_aa_atan_reduced(double x)
{
    const double at0 = 3.33333333333329318027e-01;
    const double at1 = -1.99999999998764832476e-01;
    const double at2 = 1.42857142725034663711e-01;
    const double at3 = -1.11111104054623557880e-01;
    const double at4 = 9.09088713343650656196e-02;
    const double at5 = -7.69187620504482999495e-02;
    const double at6 = 6.66107313738753120669e-02;
    const double at7 = -5.83357013379057348645e-02;
    const double at8 = 4.97687799461593236017e-02;
    const double at9 = -3.65315727442169155270e-02;
    const double at10 = 1.62858201153657823623e-02;
    // atan(0.5), atan(1), atan(1.5) and pi/2, each as a high and low part.
    const double hi0 = 4.63647609000806093515e-01;
    const double hi1 = 7.85398163397448278999e-01;
    const double hi2 = 9.82793723247329054082e-01;
    const double hi3 = 1.57079632679489655800e+00;
    const double lo0 = 2.26987774529616870924e-17;
    const double lo1 = 3.06161699786838301793e-17;
    const double lo2 = 1.39033110312309984516e-17;
    const double lo3 = 6.12323399573676603587e-17;
    const int hx = (int)((unsigned long long)__double_as_longlong(x) >> 32);
    const int ix = hx & 0x7fffffff;
    double hi = 0.0, lo = 0.0;
    bool reduced = true;
    if (ix >= 0x44100000) {
        const double r = __dadd_rn(hi3, lo3);
        return hx > 0 ? r : -r;
    }
    if (ix < 0x3fdc0000) {
        if (ix < 0x3e400000) return x;
        reduced = false;
    } else {
        x = fabs(x);
        if (ix < 0x3ff30000) {
            if (ix < 0x3fe60000) {
                hi = hi0; lo = lo0;
                x = __ddiv_rn(__dsub_rn(__dmul_rn(2.0, x), 1.0),
                              __dadd_rn(2.0, x));
            } else {
                hi = hi1; lo = lo1;
                x = __ddiv_rn(__dsub_rn(x, 1.0), __dadd_rn(x, 1.0));
            }
        } else if (ix < 0x40038000) {
            hi = hi2; lo = lo2;
            x = __ddiv_rn(__dsub_rn(x, 1.5),
                          __dadd_rn(1.0, __dmul_rn(1.5, x)));
        } else {
            hi = hi3; lo = lo3;
            x = __ddiv_rn(-1.0, x);
        }
    }
    const double z = __dmul_rn(x, x);
    const double w = __dmul_rn(z, z);
    const double s1 = __dmul_rn(z, __dadd_rn(at0, __dmul_rn(w, __dadd_rn(at2,
        __dmul_rn(w, __dadd_rn(at4, __dmul_rn(w, __dadd_rn(at6,
        __dmul_rn(w, __dadd_rn(at8, __dmul_rn(w, at10))))))))))) ;
    const double s2 = __dmul_rn(w, __dadd_rn(at1, __dmul_rn(w, __dadd_rn(at3,
        __dmul_rn(w, __dadd_rn(at5, __dmul_rn(w, __dadd_rn(at7,
        __dmul_rn(w, at9)))))))));
    const double xs = __dmul_rn(x, __dadd_rn(s1, s2));
    if (!reduced) return __dsub_rn(x, xs);
    const double r = __dsub_rn(hi, __dsub_rn(__dsub_rn(xs, lo), x));
    return hx < 0 ? -r : r;
}

__device__ __forceinline__ double thompson_aa_atan2(double y, double x)
{
    const double pi_o_2 = 1.5707963267948965580e+00;
    const double pi = 3.1415926535897931160e+00;
    const double pi_lo = 1.2246467991473531772e-16;
    if (isnan(x) || isnan(y)) return __dadd_rn(x, y);
    const unsigned long long ux = (unsigned long long)__double_as_longlong(x);
    const unsigned long long uy = (unsigned long long)__double_as_longlong(y);
    if (x == 1.0) return thompson_aa_atan_reduced(y);
    // m = 2*sign(x) + sign(y).
    const int m = (int)((uy >> 63) & 1ULL) | (int)(((ux >> 63) & 1ULL) << 1);
    if (y == 0.0) {
        if (m < 2) return y;
        return m == 2 ? pi : -pi;
    }
    if (x == 0.0) return (uy >> 63) ? -pi_o_2 : pi_o_2;
    if (isinf(y)) return (uy >> 63) ? -pi_o_2 : pi_o_2;
    if (isinf(x)) {
        if (m == 0) return 0.0;
        if (m == 1) return -0.0;
        return m == 2 ? pi : -pi;
    }
    const int ix = (int)((ux >> 52) & 0x7ffULL);
    const int iy = (int)((uy >> 52) & 0x7ffULL);
    const int k = iy - ix;
    double z;
    int mm = m;
    if (k > 60) {
        z = __dadd_rn(pi_o_2, __dmul_rn(0.5, pi_lo));
        mm = m & 1;
    } else if ((ux >> 63) && k < -60) {
        z = 0.0;
    } else {
        z = thompson_aa_atan_reduced(fabs(__ddiv_rn(y, x)));
    }
    switch (mm) {
    case 0: return z;
    case 1: return -z;
    case 2: return __dsub_rn(pi, __dsub_rn(z, pi_lo));
    default: return __dsub_rn(__dsub_rn(z, pi_lo), pi);
    }
}
