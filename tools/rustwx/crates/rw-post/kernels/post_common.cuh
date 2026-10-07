// SPDX-License-Identifier: Apache-2.0
//
// WOOF post-processor, shared device code (revision 1).
//
// Clean-room implementation written from the WOOF post specification
// (sections 3.2 to 3.5 and 4.1) and these public sources only:
//   [R1] A Description of the Advanced Research WRF Model Version 4,
//        NCAR/TN-556+STR (2019): constants, hybrid coordinate, staggering.
//   [R2] AMS Glossary of Meteorology: virtual temperature, hypsometric
//        equation, latent heat.
//   [R4] Solar Energy 40 (1988) 227-235: approximate solar position.
//   [R6] J. Appl. Meteor. 35 (1996) 601-609: improved Magnus forms.
//   [R8] WMO-No. 8: relative humidity with respect to water.
//   [R9] Mon. Wea. Rev. 108 (1980) 1046-1053: equivalent potential
//        temperature (eqs. 21 and 43).
//
// Every expression here has a Rust twin in src/thermo.rs and src/math.rs
// with the same operations in the same order.  Built with --fmad=false,
// --prec-div=true, --prec-sqrt=true and --ftz=false, so the GPU and CPU
// results are the same bits.
//
// Elementary functions come from woof_math.cuh, the twin of src/math.rs.

#pragma once

typedef unsigned int u32;
typedef unsigned long long u64;
typedef int i32;
typedef long long i64;

// ---------------------------------------------------------------------------
// Constants (section 3.2).  Hex literals so both compilers see the same bits;
// src/consts.rs carries the identical bit patterns.
#define WP_G        0x1.39eb86p+3f   /* 9.81 m s-2          */
#define WP_RD       0x1.1fp+8f       /* 287 J kg-1 K-1      */
#define WP_RV       0x1.cd999ap+8f   /* 461.6 J kg-1 K-1    */
#define WP_P0       0x1.86ap+16f     /* 100000 Pa           */
#define WP_EPS      0x1.3e5612p-1f   /* Rd / Rv             */
#define WP_ONE_EPS  0x1.8353dcp-2f   /* 1 - Rd / Rv         */
#define WP_KAPPA    0x1.24924ap-2f   /* Rd / cp, cp = 7 Rd / 2 */
#define WP_LV       0x1.314c4p+21f   /* 2.501e6 J kg-1      */
#define WP_T0C      0x1.112666p+8f   /* 273.15 K            */
#define WP_ESW_A    0x1.317852p+9f   /* 610.94 Pa           */
#define WP_ESW_B    0x1.1ap+4f       /* 17.625              */
#define WP_ESW_C    0x1.e6147ap+7f   /* 243.04 C            */
#define WP_ESI_A    0x1.319ae2p+9f   /* 611.21 Pa           */
#define WP_ESI_B    0x1.69645ap+4f   /* 22.587              */
#define WP_ESI_C    0x1.11dc28p+8f   /* 273.86 C            */
#define WP_BL_A     0x1.63p+11f      /* 2840                */
#define WP_BL_B     0x1.cp+1f        /* 3.5                 */
#define WP_BL_C     0x1.33851ep+2f   /* 4.805               */
#define WP_BL_D     0x1.b8p+5f       /* 55                  */
#define WP_BK       0x1.243fe6p-2f   /* 0.2854              */
#define WP_BK_R     0x1.2599eep-12f  /* 0.28e-3             */
#define WP_BE_A     0x1.b020c4p+1f   /* 3.376               */
#define WP_BE_B     0x1.4cec42p-9f   /* 0.00254             */
#define WP_BE_C     0x1.a8ac5cp-11f  /* 0.81e-3             */
#define WP_HPA      0x1.47ae14p-7f   /* 0.01 hPa per Pa     */
#define WP_DEG2RAD  0x1.1df46ap-6f   /* pi / 180            */

#define WP_NAN __int_as_float(0x7fc00000)

// ---------------------------------------------------------------------------
// Elementary functions: the shared maths library (woof_expf, woof_logf,
// woof_sinf, woof_cosf, woof_powpos and the binary64 wm_* family).

#include "woof_math.cuh"

// ---------------------------------------------------------------------------
// Thermodynamic helpers (section 4.1).  Temperatures in K, pressures in Pa.

// Saturation vapour pressure over liquid water, AERK form [R6].
__device__ __forceinline__ float wp_es_water(float tk) {
    float tc = tk - WP_T0C;
    return WP_ESW_A * woof_expf(WP_ESW_B * tc / (tc + WP_ESW_C));
}

// Saturation vapour pressure over ice, AERKi form [R6].
__device__ __forceinline__ float wp_es_ice(float tk) {
    float tc = tk - WP_T0C;
    return WP_ESI_A * woof_expf(WP_ESI_B * tc / (tc + WP_ESI_C));
}

// Vapour pressure from specific humidity and pressure.
__device__ __forceinline__ float wp_vapour_pressure(float q, float p) {
    return q * p / (WP_EPS + WP_ONE_EPS * q);
}

// Dewpoint (K) from vapour pressure, inverse of the liquid AERK form [R6].
// e <= 0 has no dewpoint: NaN.
__device__ __forceinline__ float wp_dewpoint(float e) {
    if (!(e > 0.0f)) return WP_NAN;
    float l = woof_logf(e / WP_ESW_A);
    return WP_ESW_C * l / (WP_ESW_B - l) + WP_T0C;
}

// Relative humidity (%) w.r.t. liquid water [R8], clipped to [0, 100];
// NaN stays NaN.
__device__ __forceinline__ float wp_rh(float e, float tk) {
    float rh = 100.0f * e / wp_es_water(tk);
    if (rh > 100.0f) rh = 100.0f;
    if (rh < 0.0f) rh = 0.0f;
    return rh;
}

// Virtual temperature from temperature and mixing ratio [R2].
__device__ __forceinline__ float wp_virtual_temperature(float tk, float r) {
    return tk * (1.0f + r / WP_EPS) / (1.0f + r);
}

// Pseudo-equivalent potential temperature, [R9] eq. 43 with T_L from eq. 21.
// tk in K, p in Pa, r mixing ratio (kg/kg).  e <= 0: NaN.
__device__ __forceinline__ float wp_theta_e(float tk, float p, float r) {
    float q = r / (1.0f + r);
    float e_hpa = wp_vapour_pressure(q, p) * WP_HPA;
    if (!(e_hpa > 0.0f)) return WP_NAN;
    float tl = WP_BL_A / (WP_BL_B * woof_logf(tk) - woof_logf(e_hpa) - WP_BL_C) + WP_BL_D;
    float rg = r * 1000.0f;
    float expo = WP_BK * (1.0f - WP_BK_R * rg);
    float theta = tk * woof_powpos(1000.0f / (p * WP_HPA), expo);
    return theta * woof_expf((WP_BE_A / tl - WP_BE_B) * rg * (1.0f + WP_BE_C * rg));
}

// Solar zenith cosine from per-frame scalars (computed on the host by
// src/solar.rs from [R4]): sin and cos of the declination and the hour
// angle at longitude 0 in degrees.
__device__ __forceinline__ float wp_cosz(float lat_deg, float lon_deg, float sin_dec,
                                         float cos_dec, float ha0_deg) {
    float ha = ha0_deg + lon_deg;
    if (ha > 180.0f) ha = ha - 360.0f;
    if (ha < -180.0f) ha = ha + 360.0f;
    float lat = lat_deg * WP_DEG2RAD;
    float har = ha * WP_DEG2RAD;
    return sin_dec * woof_sinf(lat) + cos_dec * woof_cosf(lat) * woof_cosf(har);
}
