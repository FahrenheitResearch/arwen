/* Correctly rounded float32 tgammaf and log10f for the CR oracle arm.
 *
 * module_mp_p3.F calls exactly six libm entries (nm -D on the arm-A
 * executable): expf, logf, powf, exp, log10f and tgammaf (gfortran binds
 * GAMMA() and ALOG10() of a REAL to the last two).  glibc 2.39, the C
 * library on the campaign box, rounds tgammaf incorrectly on 39.4 % of
 * [0.25, 36] (worst 6 ULP, docs/gf_gamma_known_delta.md) and log10f
 * incorrectly on part of its domain; the glibc 2.43 build behind the
 * committed 2026-08-28 dumps agrees with this arm on every word of all
 * thirteen p3-fortref fixtures.  Linking this object ahead of libm makes
 * the arm compute what WRF 4.6.1 computes on a C library whose tgammaf/log10f are correctly
 * rounded.  expf/logf/powf/exp are left to the C library.
 *
 * Method: the double result (error far below 2^-40 relative) is rounded to
 * float32 directly unless it lies within 2^-40 relative of a float32
 * rounding boundary; only then is the 113-bit libquadmath value computed
 * and rounded once.  The result is the correctly rounded float32 either
 * way, at double speed.  This file is the project's own; it consults no
 * libm implementation. */
#include <math.h>
#include <quadmath.h>
#include <stdint.h>
#include <string.h>

static int near_boundary(double d)
{
    if (!isfinite(d) || d == 0.0) return 1;
    /* float32 keeps 24 significand bits; double has 53.  The 29 dropped bits
     * sit at the bottom of the double's mantissa; a rounding boundary is the
     * pattern 1000...0 there.  Near means within 2^13 double ulps of it,
     * i.e. within 2^-40 relative. */
    uint64_t u; memcpy(&u, &d, 8);
    uint64_t low = u & ((1ULL << 29) - 1);
    int64_t dist = (int64_t)low - (int64_t)(1ULL << 28);
    if (dist < 0) dist = -dist;
    if (dist < (1 << 13)) return 1;
    /* also near an exactly representable float (low ~ 0 or ~ 2^29) */
    if (low < (1 << 13) || low > (1ULL << 29) - (1 << 13)) return 1;
    return 0;
}

float tgammaf(float x)
{
    double d = tgamma((double)x);
    if (!near_boundary(d) || fabs(d) < 1e-300 || fabs(d) > 1e300)
        return (float)d;
    return (float)tgammaq((__float128)x);
}

float log10f(float x)
{
    double d = log10((double)x);
    if (!near_boundary(d))
        return (float)d;
    return (float)log10q((__float128)x);
}
