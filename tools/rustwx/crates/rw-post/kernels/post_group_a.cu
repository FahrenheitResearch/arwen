// SPDX-License-Identifier: Apache-2.0
//
// WOOF post-processor, Group A kernels (revision 1): the shared column state,
// shelter and surface fields, wind rotation, and probes that let the tests
// compare every device helper with its CPU twin bit for bit.
//
// Clean-room implementation written from the WOOF post specification
// (sections 3.4, 3.5, 4.1, 4.2 and decisions D1 to D5, D26 to D28) and the
// public sources cited in post_common.cuh plus:
//   [R3]  ARW Version 4 Modeling System User's Guide (SINALPHA/COSALPHA).
//   [R11] J. Geophys. Res. 108 (2003) 8851: Noah land model snow cover.
// Rust twins: src/state.rs, src/surface.rs, src/wind.rs.
//
// Layout: structure of arrays, level-major [k][cell], cell = j * nx + i.
// One thread per column (state) or per cell (surface).  No atomics, no
// cross-thread reductions: each output word depends only on its own column.

#include "post_common.cuh"

// Slots of the packed mass-level input buffer (each nz * ncell words).
#define IN3_T      0
#define IN3_P      1
#define IN3_PB     2
#define IN3_QV     3
#define IN3_QC     4
#define IN3_QR     5
#define IN3_QI     6
#define IN3_QS     7
#define IN3_QG     8
#define IN3_QH     9
#define IN3_PHYD  10
#define IN3_COUNT 11

// Slots of the packed mass-level state output (each nz * ncell words).
#define ST_THETA   0
#define ST_Q       1
#define ST_P       2
#define ST_PFULL   3
#define ST_TK      4
#define ST_TV      5
#define ST_ZMASS   6
#define ST_U       7
#define ST_V       8
#define ST_R       9
#define ST_THETAV 10
#define ST_COUNT  11

// Slots of the packed interface state output (each (nz + 1) * ncell words).
#define SI_PINT    0
#define SI_ZINT    1

extern "C" __global__ void woof_post_state_v1(
    const float* __restrict__ in3,        // IN3_COUNT * nz * ncell
    u32 present3,                         // bit s set: slot s carries data
    const float* __restrict__ ph,         // (nz + 1) * ncell
    const float* __restrict__ phb,        // (nz + 1) * ncell
    const float* __restrict__ u_stag,     // nz * ny * (nx + 1)
    const float* __restrict__ v_stag,     // nz * (ny + 1) * nx
    const float* __restrict__ mu,         // ncell
    const float* __restrict__ mub,        // ncell
    const float* __restrict__ c3f,        // nz + 1
    const float* __restrict__ c4f,        // nz + 1
    float p_top,
    u32 nx, u32 ny, u32 nz,
    float* __restrict__ st,               // ST_COUNT * nz * ncell
    float* __restrict__ si) {             // 2 * (nz + 1) * ncell
    const u32 ncell = nx * ny;
    const u32 cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= ncell) return;
    const u32 i = cell % nx;
    const u32 j = cell / nx;
    const u64 vol = (u64)nz * ncell;
    const u64 ivol = (u64)(nz + 1) * ncell;

    // Interface heights (PH + PHB) / g, metres MSL [R1].
    for (u32 k = 0; k <= nz; ++k) {
        const u64 at = (u64)k * ncell + cell;
        si[SI_ZINT * ivol + at] = (ph[at] + phb[at]) / WP_G;
    }

    // Hydrostatic interface and mass pressure, integrated down from P_TOP
    // with each layer's dry weight times (1 + total water) [R1 2.1-2.3].
    const float mud = mu[cell] + mub[cell];
    float p_above = p_top;
    float pd_above = c3f[nz] * mud + c4f[nz] + p_top;
    si[SI_PINT * ivol + (u64)nz * ncell + cell] = p_above;
    for (i32 k = (i32)nz - 1; k >= 0; --k) {
        const u64 at = (u64)k * ncell + cell;
        const float pd = c3f[k] * mud + c4f[k] + p_top;
        float qt = 0.0f;
        for (u32 s = IN3_QV; s <= IN3_QH; ++s) {
            if ((present3 >> s) & 1u) {
                const float w = in3[s * vol + at];
                if (w > 0.0f) qt = qt + w;
            }
        }
        const float weight = (pd - pd_above) * (1.0f + qt);
        const float pmass = p_above + 0.5f * weight;
        p_above = p_above + weight;
        pd_above = pd;
        si[SI_PINT * ivol + at] = p_above;
        st[ST_P * vol + at] = ((present3 >> IN3_PHYD) & 1u) ? in3[IN3_PHYD * vol + at] : pmass;
    }

    for (u32 k = 0; k < nz; ++k) {
        const u64 at = (u64)k * ncell + cell;
        const float theta = in3[IN3_T * vol + at] + 300.0f;
        const float rraw = in3[IN3_QV * vol + at];
        const float r = rraw > 0.0f ? rraw : 0.0f;
        const float pfull = in3[IN3_P * vol + at] + in3[IN3_PB * vol + at];
        const float tk = theta * woof_powpos(pfull / WP_P0, WP_KAPPA);
        st[ST_THETA * vol + at] = theta;
        st[ST_Q * vol + at] = r / (1.0f + r);
        st[ST_PFULL * vol + at] = pfull;
        st[ST_TK * vol + at] = tk;
        st[ST_TV * vol + at] = wp_virtual_temperature(tk, r);
        st[ST_ZMASS * vol + at] =
            0.5f * (si[SI_ZINT * ivol + at] + si[SI_ZINT * ivol + at + ncell]);
        const u64 urow = ((u64)k * ny + j) * (nx + 1);
        st[ST_U * vol + at] = 0.5f * (u_stag[urow + i] + u_stag[urow + i + 1]);
        const u64 vrow = ((u64)k * (ny + 1) + j) * nx;
        st[ST_V * vol + at] = 0.5f * (v_stag[vrow + i] + v_stag[vrow + nx + i]);
        st[ST_R * vol + at] = r;
        st[ST_THETAV * vol + at] = wp_virtual_temperature(theta, r);
    }
}

// Slots of the packed 2D input buffer (each ncell words).
#define IN2_HGT     0
#define IN2_PSFC    1
#define IN2_T2      2
#define IN2_TH2     3
#define IN2_Q2      4
#define IN2_TSK     5
#define IN2_SNOW    6
#define IN2_SNOWH   7
#define IN2_SNOWC   8
#define IN2_HFX     9
#define IN2_LH     10
#define IN2_QFX    11
#define IN2_GRDFLX 12
#define IN2_UST    13
#define IN2_GLW    14
#define IN2_SWDOWN 15
#define IN2_VEGFRA 16
#define IN2_IVGTYP 17
#define IN2_SEAICE 18
#define IN2_ZNT    19
#define IN2_SHDMIN 20
#define IN2_SHDMAX 21
#define IN2_LAI    22
#define IN2_XLAT   23
#define IN2_XLONG  24
#define IN2_SNUP   25   /* snow-depletion threshold, m water; 0 = water/ice, NaN = unknown */
#define IN2_TVLOW  26   /* lowest mass-level virtual temperature (from the state) */
#define IN2_COSZ   27   /* solar zenith cosine carried by the input, used as given */
#define IN2_COUNT  28

// Output planes, in the order of the specification's section 4.2.
#define OUT_TERRAIN   0
#define OUT_PSFC      1
#define OUT_T2        2
#define OUT_TD2       3
#define OUT_RH2       4
#define OUT_TH2       5
#define OUT_Q2        6
#define OUT_TSK       7
#define OUT_SNOW      8
#define OUT_SNOWH     9
#define OUT_SNOWC    10
#define OUT_HFX      11
#define OUT_LH       12
#define OUT_GRDFLX   13
#define OUT_UST      14
#define OUT_GLW      15
#define OUT_SWDOWN   16
#define OUT_VEGFRA   17
#define OUT_IVGTYP   18
#define OUT_SEAICE   19
#define OUT_ZNT      20
#define OUT_SHDMIN   21
#define OUT_SHDMAX   22
#define OUT_LAI      23
#define OUT_COSZ     24
#define OUT_P2       25
#define OUT_COUNT    26

// Scalar options of the surface kernel.
struct WpSurfaceOptions {
    float sin_dec, cos_dec, ha0_deg;  // solar scalars, src/solar.rs
    float vegfra_scale;               // 1 (percent) or 100 (fraction)
    float shd_scale;                  // same for SHDMIN/SHDMAX
    u32 latent_from_qfx;              // 1: RUC land model, LH = QFX * Lv (D28)
    u32 snow_cover_derived;           // 1: no SNOWC, Noah depletion curve (D26)
    float snow_salp;                  // Noah depletion-curve shape parameter
};

__device__ __forceinline__ float wp_floor0(float x) { return x < 0.0f ? 0.0f : x; }

extern "C" __global__ void woof_post_surface_v1(
    const float* __restrict__ in2,        // IN2_COUNT * ncell
    u32 present2,                         // bit s set: slot s carries data
    WpSurfaceOptions opt,
    u32 ncell,
    float* __restrict__ out) {            // OUT_COUNT * ncell
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncell) return;
#define IN(s) in2[(u64)(s) * ncell + c]
#define HAS(s) (((present2 >> (s)) & 1u) != 0u)
#define OUT(f) out[(u64)(f) * ncell + c]
    const float nan = WP_NAN;

    OUT(OUT_TERRAIN) = HAS(IN2_HGT) ? IN(IN2_HGT) : nan;
    const float psfc = HAS(IN2_PSFC) ? IN(IN2_PSFC) : nan;
    OUT(OUT_PSFC) = psfc;

    // Shelter pressure, hypsometric 2 m above the ground (D4, [R2]).
    float p2 = nan;
    if (HAS(IN2_TVLOW)) {
        p2 = psfc * woof_expf(-(WP_G * 2.0f) / (WP_RD * IN(IN2_TVLOW)));
    }
    OUT(OUT_P2) = p2;

    // 2 m temperature: the surface scheme's T2, else TH2 at shelter pressure.
    float t2 = nan;
    if (HAS(IN2_T2)) {
        t2 = IN(IN2_T2);
    } else if (HAS(IN2_TH2)) {
        t2 = IN(IN2_TH2) * woof_powpos(p2 / WP_P0, WP_KAPPA);
    }
    OUT(OUT_T2) = t2;

    // Shelter humidity: Q2 is a mixing ratio; negative values are no vapour.
    float q2 = nan;
    float r2 = nan;
    if (HAS(IN2_Q2)) {
        r2 = wp_floor0(IN(IN2_Q2));
        q2 = r2 / (1.0f + r2);
    }
    OUT(OUT_Q2) = q2;
    const float e2 = wp_vapour_pressure(q2, p2);
    float td2 = wp_dewpoint(e2);
    if (td2 > t2) td2 = t2;
    OUT(OUT_TD2) = td2;
    OUT(OUT_RH2) = wp_rh(e2, t2);
    OUT(OUT_TH2) = HAS(IN2_TH2) ? IN(IN2_TH2) : nan;
    OUT(OUT_TSK) = HAS(IN2_TSK) ? IN(IN2_TSK) : nan;
    OUT(OUT_SNOW) = HAS(IN2_SNOW) ? wp_floor0(IN(IN2_SNOW)) : nan;
    OUT(OUT_SNOWH) = HAS(IN2_SNOWH) ? wp_floor0(IN(IN2_SNOWH)) : nan;

    // Snow cover in percent (D26).
    float snowc = nan;
    if (!opt.snow_cover_derived) {
        if (HAS(IN2_SNOWC)) snowc = IN(IN2_SNOWC) * 100.0f;
    } else if (HAS(IN2_SNOW) && HAS(IN2_SNUP)) {
        const float snup = IN(IN2_SNUP);
        const float swe_m = wp_floor0(IN(IN2_SNOW)) / 1000.0f;
        if (snup != snup) {
            snowc = nan;   // category without a threshold: unknown
        } else if (snup == 0.0f) {
            snowc = 0.0f;  // water, lake or ice category
        } else if (swe_m >= snup) {
            snowc = 100.0f;
        } else {
            const float rs = swe_m / snup;
            const float frac =
                1.0f - (woof_expf(-opt.snow_salp * rs) - rs * woof_expf(-opt.snow_salp));
            snowc = 100.0f * frac;
        }
    }
    OUT(OUT_SNOWC) = snowc;

    OUT(OUT_HFX) = HAS(IN2_HFX) ? IN(IN2_HFX) : nan;
    float lh = nan;
    if (opt.latent_from_qfx) {
        if (HAS(IN2_QFX)) lh = IN(IN2_QFX) * WP_LV;
    } else if (HAS(IN2_LH)) {
        lh = IN(IN2_LH);
    }
    OUT(OUT_LH) = lh;
    OUT(OUT_GRDFLX) = HAS(IN2_GRDFLX) ? IN(IN2_GRDFLX) : nan;
    OUT(OUT_UST) = HAS(IN2_UST) ? IN(IN2_UST) : nan;
    OUT(OUT_GLW) = HAS(IN2_GLW) ? IN(IN2_GLW) : nan;

    float cosz = nan;
    if (HAS(IN2_COSZ)) {
        cosz = IN(IN2_COSZ);
    } else if (HAS(IN2_XLAT) && HAS(IN2_XLONG)) {
        cosz = wp_cosz(IN(IN2_XLAT), IN(IN2_XLONG), opt.sin_dec, opt.cos_dec, opt.ha0_deg);
    }
    OUT(OUT_COSZ) = cosz;
    float sw = nan;
    if (HAS(IN2_SWDOWN)) {
        sw = IN(IN2_SWDOWN);
        if (cosz <= 0.0f) sw = 0.0f;  // night guard (D27); NaN cosz keeps SWDOWN
    }
    OUT(OUT_SWDOWN) = sw;
    OUT(OUT_VEGFRA) = HAS(IN2_VEGFRA) ? IN(IN2_VEGFRA) * opt.vegfra_scale : nan;
    OUT(OUT_IVGTYP) = HAS(IN2_IVGTYP) ? IN(IN2_IVGTYP) : nan;
    OUT(OUT_SEAICE) = HAS(IN2_SEAICE) ? IN(IN2_SEAICE) : nan;
    OUT(OUT_ZNT) = HAS(IN2_ZNT) ? IN(IN2_ZNT) : nan;
    OUT(OUT_SHDMIN) = HAS(IN2_SHDMIN) ? IN(IN2_SHDMIN) * opt.shd_scale : nan;
    OUT(OUT_SHDMAX) = HAS(IN2_SHDMAX) ? IN(IN2_SHDMAX) * opt.shd_scale : nan;
    OUT(OUT_LAI) = HAS(IN2_LAI) ? IN(IN2_LAI) : nan;
#undef IN
#undef HAS
#undef OUT
}

// Grid-relative to earth-relative wind rotation [R1], [R3]:
// u_e = u cos a - v sin a, v_e = v cos a + u sin a.  One thread per word;
// `planes` levels share the 2D sin/cos.
extern "C" __global__ void woof_post_rotate_v1(
    const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ sina, const float* __restrict__ cosa,
    u32 ncell, u32 planes,
    float* __restrict__ ue, float* __restrict__ ve) {
    const u64 n = (u64)ncell * planes;
    const u64 at = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= n) return;
    const u32 c = (u32)(at % ncell);
    const float sa = sina[c];
    const float ca = cosa[c];
    ue[at] = u[at] * ca - v[at] * sa;
    ve[at] = v[at] * ca + u[at] * sa;
}

// Probe of the elementary functions: out = [expf, logf, sinf, cosf] per x.
extern "C" __global__ void woof_post_libm_probe_v1(
    const float* __restrict__ x, u32 n, float* __restrict__ out) {
    const u32 at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= n) return;
    const float v = x[at];
    out[4u * at + 0u] = woof_expf(v);
    out[4u * at + 1u] = woof_logf(v);
    out[4u * at + 2u] = woof_sinf(v);
    out[4u * at + 3u] = woof_cosf(v);
}

// Probe of the binary64 maths library: out = [ln, exp, pow(|x|, y), cbrt,
// sin, cos] per (x, y) pair, for the identity tests.
extern "C" __global__ void woof_post_math64_probe_v1(
    const double* __restrict__ x, const double* __restrict__ y, u32 n, double* __restrict__ out) {
    const u32 at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= n) return;
    const double v = x[at];
    const double a = v < 0.0 ? -v : v;
    out[6u * at + 0u] = wm_ln(v);
    out[6u * at + 1u] = wm_exp(v);
    out[6u * at + 2u] = wm_pow(a, y[at]);
    out[6u * at + 3u] = wm_cbrt(v);
    out[6u * at + 4u] = wm_sin(v);
    out[6u * at + 5u] = wm_cos(v);
}

// Probe of the thermodynamic helpers on (T K, p Pa, r kg/kg) triples:
// out = [es_water, es_ice, e, dewpoint, rh, tv, theta_e] per triple.
extern "C" __global__ void woof_post_thermo_probe_v1(
    const float* __restrict__ tk, const float* __restrict__ p,
    const float* __restrict__ r, u32 n, float* __restrict__ out) {
    const u32 at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= n) return;
    const float t = tk[at];
    const float pp = p[at];
    const float rr = r[at];
    const float q = rr / (1.0f + rr);
    const float e = wp_vapour_pressure(q, pp);
    out[7u * at + 0u] = wp_es_water(t);
    out[7u * at + 1u] = wp_es_ice(t);
    out[7u * at + 2u] = e;
    out[7u * at + 3u] = wp_dewpoint(e);
    out[7u * at + 4u] = wp_rh(e, t);
    out[7u * at + 5u] = wp_virtual_temperature(t, rr);
    out[7u * at + 6u] = wp_theta_e(t, pp, rr);
}
