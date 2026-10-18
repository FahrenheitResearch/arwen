// WRF swint_opt = 1: the surface shortwave between radiation calls.
//
// Statement for statement the routines the radiation driver runs for
// swint_opt = 1, as WRF v4.6.1 phys/module_radiation_driver.F has them
// (commit d66e442, sha256 77987565...):
//
//   swint_coszen_loc   radconst (:3469-3511, the declination half) and
//                      calc_coszen (:3514-3541) at the CURRENT xtime: the
//                      per-step block at :1158-1170 (the radiation step
//                      keeps its own xtime + radt*0.5 call at :1204-1206)
//   swint_update       update_swinterp_parameters (:3543-3624), once per
//                      radiation call (:2932-2938), after the shortwave
//                      scheme wrote SWDOWN and SWDDIR
//   swint_interp       interp_sw_radiation (:3626-3670), on EVERY step,
//                      radiation steps included (:2964-2980 sits after
//                      ENDIF Radiation_step)
//
// The fit is a power law per column: SWDDIR = Bx * coszen**bb and
// SWDOWN = Gx * coszen**gg, the exponents from the log ratio of the two
// most recent radiation results, clamped to [-0.5, 2.5]; at a clamp, and
// at an exponent of exactly zero, the evaluation falls back to the ratio
// coszen_loc/coszen_ref.
//
// Two upstream WRF bug fixes are part of the scheme here, both absent from
// the operational HRRR fork (NOAA-EMC/HRRR v4.1.21, WRFV3.9 base, commit
// 40ee6058c), which predates them:
//   * WRF cc02f989f (v3.9.1, "fix for sunrise/sunset swint_opt calcs"):
//     an exponent of exactly zero takes the ratio rule.  Without it a
//     column whose fit had no sun-angle information (the ratio was one, or
//     both fluxes were at or under 1 W m-2) held its flux flat until the
//     next radiation call after sunrise or sunset.
//   * WRF 650be0c7d (v4.3): calc_coszen bounds the cosine to [-1, 1]; the
//     float32 sum can round one spacing past one at the subsolar point.
// tools/swint_wrf461_oracle grades these kernels against v4.6.1's Fortran.
//
// Numerics.  Every add, subtract, multiply and divide is an explicit IEEE
// single-rounded intrinsic (FADD and friends from glibc_flt32.cuh, which
// the loader prepends for this module), so NVRTC neither contracts a
// multiply-add into an FMA nor rewrites a constant division as a
// reciprocal multiply (A146).  The transcendentals are WOOF's own float32
// libm routines, not CUDA's: gfk_log / gfk_pow (glibc_flt32.cuh) and
// glibc_sinf / glibc_cosf / glibc_asinf (glibc_trig_flt32.cuh), which
// return the same words as the C library gfortran calls for LOG, **, SIN,
// COS and ASIN on REAL(4), so the kernels are bitwise WRF's Fortran
// compiled with gfortran at -O0.

// calc_coszen per point at the current xtime, with radconst's
// declination formed in the same thread (a per-call scalar every thread
// computes identically; it is two sinf and one asinf).  The cosine is
// bounded to [-1, 1] as v4.6.1 calc_coszen bounds it (WRF 650be0c7d).
extern "C" __global__ void swint_coszen_loc(
    long long n, const float* __restrict__ xlat,
    const float* __restrict__ xlon, float julian, float xtime, float gmt,
    float degrad, float dpd, int solar_480, float* __restrict__ coszen_loc)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // radconst: OBECL, SINOB, SXLONG, ARG, DECLIN
    float obecl = FMUL(23.5f, degrad);
    float sinob = glibc_sinf(obecl);
    float sxlong;
    if (julian >= 80.0f) {
        sxlong = FMUL(dpd, FSUB(julian, 80.0f));
    } else {
        sxlong = FMUL(dpd, FADD(julian, 285.0f));
    }
    sxlong = FMUL(sxlong, degrad);
    float arg = FMUL(sinob, glibc_sinf(sxlong));
    float declin = glibc_asinf(arg);
    // calc_coszen: da, eot, xt24 (per call), then the point
    float da = FDIV(FMUL(6.2831853071795862f, FSUB(julian, solar_480 ? 0.5f : 1.0f)), 365.0f);
    float two_da = FMUL(2.0f, da);
    float eot = FADD(0.000075f, FMUL(0.001868f, glibc_cosf(da)));
    eot = FSUB(eot, FMUL(0.032077f, glibc_sinf(da)));
    eot = FSUB(eot, FMUL(0.014615f, glibc_cosf(two_da)));
    eot = FSUB(eot, FMUL(solar_480 ? 0.040849f : 0.04089f, glibc_sinf(two_da)));
    eot = FMUL(eot, 229.18f);
    float xt24 = FADD(fmodf(xtime, 1440.0f), eot);
    float tloctm = FADD(FADD(gmt, FDIV(xt24, 60.0f)), FDIV(xlon[i], 15.0f));
    float hrang = FMUL(FMUL(15.0f, FSUB(tloctm, 12.0f)), degrad);
    float xxlat = FMUL(xlat[i], degrad);
    float cz = FADD(FMUL(glibc_sinf(xxlat), glibc_sinf(declin)),
                    FMUL(FMUL(glibc_cosf(xxlat), glibc_cosf(declin)),
                         glibc_cosf(hrang)));
    // coszen = min(max(coszen, -1.0), 1.0)
    coszen_loc[i] = fminf(fmaxf(cz, -1.0f), 1.0f);
}

// The exponent of one power-law fit (the DIR and GHI halves of
// update_swinterp_parameters are the same statements on different
// arrays).  flux is this call's radiation result, flux_0 and coszen_0 the
// anchor the Fortran picks: the linear first guess on a fresh column, the
// stored reference otherwise.  The two branch tests divide afresh, as the
// Fortran does (the quotient is the same correctly rounded word).
static __device__ __forceinline__ float swi_exponent(
    float flux, float flux_0, float coszen, float coszen_0)
{
    float b;
    if (FDIV(coszen, coszen_0) < 1.0f) {
        b = FDIV(gfk_log(FDIV(fmaxf(1.0f, flux), fmaxf(1.0f, flux_0))),
                 gfk_log(fminf(FSUB(1.0f, 1e-4f), FDIV(coszen, coszen_0))));
    } else if (FDIV(coszen, coszen_0) > 1.0f) {
        b = FDIV(gfk_log(FDIV(fmaxf(1.0f, flux), fmaxf(1.0f, flux_0))),
                 gfk_log(fmaxf(FADD(1.0f, 1e-4f), FDIV(coszen, coszen_0))));
    } else {
        b = 0.0f;
    }
    return fmaxf(-0.5f, fminf(2.5f, b));
}

// update_swinterp_parameters: fit coefficients from this radiation call's
// SWDDIR/SWDOWN at the radiation-step coszen and the stored reference,
// then store this call as the reference.
extern "C" __global__ void swint_update(
    long long n, const float* __restrict__ coszen,
    const float* __restrict__ coszen_loc, const float* __restrict__ swddir,
    const float* __restrict__ swdown, float* __restrict__ swddir_ref,
    float* __restrict__ bb, float* __restrict__ bx,
    float* __restrict__ swdown_ref, float* __restrict__ gg,
    float* __restrict__ gx, float* __restrict__ coszen_ref)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float coszen_min = 1e-4f;
    float cz = coszen[i];
    float czl = coszen_loc[i];
    if (cz > coszen_min && czl > coszen_min) {
        float swddir_0, coszen_0;
        if (bx[i] <= 0.0f) {
            swddir_0 = FMUL(FDIV(czl, cz), swddir[i]);
            coszen_0 = czl;
        } else {
            swddir_0 = swddir_ref[i];
            coszen_0 = coszen_ref[i];
        }
        float b = swi_exponent(swddir[i], swddir_0, cz, coszen_0);
        bb[i] = b;
        bx[i] = FDIV(swddir[i], gfk_pow(cz, b));

        float swdown_0;
        if (gx[i] <= 0.0f) {
            swdown_0 = FMUL(FDIV(czl, cz), swdown[i]);
            coszen_0 = czl;
        } else {
            swdown_0 = swdown_ref[i];
            coszen_0 = coszen_ref[i];
        }
        float g = swi_exponent(swdown[i], swdown_0, cz, coszen_0);
        gg[i] = g;
        gx[i] = FDIV(swdown[i], gfk_pow(cz, g));
    } else {
        bx[i] = 0.0f;
        bb[i] = 0.0f;
        gx[i] = 0.0f;
        gg[i] = 0.0f;
    }
    coszen_ref[i] = cz;
    swdown_ref[i] = swdown[i];
    swddir_ref[i] = swddir[i];
}

// interp_sw_radiation: the surface shortwave at the current coszen.  A
// clamped exponent or an exponent of exactly zero takes the ratio rule
// (WRF cc02f989f).
extern "C" __global__ void swint_interp(
    long long n, const float* __restrict__ coszen_ref,
    const float* __restrict__ coszen_loc,
    const float* __restrict__ swddir_ref, const float* __restrict__ bb,
    const float* __restrict__ bx, const float* __restrict__ swdown_ref,
    const float* __restrict__ gg, const float* __restrict__ gx,
    const float* __restrict__ albedo, float* __restrict__ swdown,
    float* __restrict__ swddir, float* __restrict__ swddni,
    float* __restrict__ swddif, float* __restrict__ gsw)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float coszen_min = 1e-4f;
    float czr = coszen_ref[i];
    float czl = coszen_loc[i];
    if (czr > coszen_min && czl > coszen_min) {
        float dir, dn;
        if (bb[i] == -0.5f || bb[i] == 2.5f || bb[i] == 0.0f) {
            dir = FMUL(FDIV(czl, czr), swddir_ref[i]);
        } else {
            dir = FMUL(bx[i], gfk_pow(czl, bb[i]));
        }
        if (gg[i] == -0.5f || gg[i] == 2.5f || gg[i] == 0.0f) {
            dn = FMUL(FDIV(czl, czr), swdown_ref[i]);
        } else {
            dn = FMUL(gx[i], gfk_pow(czl, gg[i]));
        }
        swddir[i] = dir;
        swdown[i] = dn;
        swddif[i] = FSUB(dn, dir);
        swddni[i] = FDIV(dir, czl);
        gsw[i] = FMUL(dn, FSUB(1.0f, albedo[i]));
    } else {
        swddir[i] = 0.0f;
        swdown[i] = 0.0f;
        swddif[i] = 0.0f;
        swddni[i] = 0.0f;
        gsw[i] = 0.0f;
    }
}
