// SPDX-License-Identifier: Apache-2.0
//
// WOOF clean-room post, Group B kernels: pressure levels, sea-level
// pressure and column water.  Written from public sources only (see
// src/lib.rs for the titles): NCAR/TN-556 (WRF ARW v4), NCAR/TN-396
// (1993), J. Appl. Meteor. 35 (1996), Mon. Wea. Rev. 123 (1995),
// Wea. Forecasting 13 (1998), Mon. Wea. Rev. 118 (1990), Rev. Geophys. 8
// (1970), the AMS Glossary and WMO-No. 8.
//
// Every device function is the twin of a Rust function in src/thermo.rs,
// src/group_b.rs, src/membrane.rs or src/smooth.rs, with the same float64
// operations in the same order.  Built with --fmad=false and IEEE division
// and square root; transcendentals come from the
// shared maths library (woof_math.cuh, the twin of src/math.rs) the CPU path
// calls.  The two devices therefore agree bit for bit.  Keep the twins in
// step.  The column state (pressure, interfaces, temperature, winds) is
// Group A's shared state (post_group_a.cu, ST_* and SI_* slots).
//
// No atomics; every thread owns its outputs; fixed loop counts.

#include "woof_math.cuh"

#define G 9.81
#define RD 287.0
#define RV 461.6
#define CP (7.0 * RD / 2.0)
#define P0 100000.0
#define EPS (RD / RV)
#define KAPPA (RD / CP)
#define THETA_BASE 300.0
#define GAMMA 0.0065
#define ALPHA (GAMMA * RD / G)
#define T_MELT 273.15
#define MAGNUS_A 610.94
#define MAGNUS_B 17.625
#define MAGNUS_C 243.04
#define Q_FLOOR 1.0e-10
#define TN396_Z_LOW 2000.0
#define TN396_Z_HIGH 2500.0
#define TN396_T_PLATEAU 298.0
#define TN396_T_WARM 290.5
#define TN396_T_COLD 255.0
#define NMC_T_WARM 290.66
#define NMC_WARM_COEF 0.005
#define MAPS_P_REF 70000.0

#define NZ_MAX 160
#define P2_PWAT 0
#define P2_MSLP 1
#define P2_T700 2
#define P2_MAPS 3
#define P2_MEMBRANE_SLP 4
#define N2D 5
#define Q_PGROUND 0
#define Q_TVGROUND 1
#define Q_MAPS_RAW 2
#define N64_EXTRA 3

#define M_UNKNOWN 0
#define M_FIXED 1
#define M_MISSING 2

__device__ __forceinline__ double dnan() { return __longlong_as_double(0x7ff8000000000000LL); }
__device__ __forceinline__ bool disnan(double x) { return x != x; }
__device__ __forceinline__ float to_f32(double v) {
    return disnan(v) ? __int_as_float(0x7fc00000) : __double2float_rn(v);
}
__device__ __forceinline__ double clamp0(double x) { return x > 0.0 ? x : 0.0; }

// ---- thermo.rs twins -------------------------------------------------------

__device__ __forceinline__ double esat_water(double t_k) {
    double tc = t_k - T_MELT;
    return MAGNUS_A * wm_exp(MAGNUS_B * tc / (tc + MAGNUS_C));
}

__device__ __forceinline__ double vapour_pressure(double q, double p) {
    double qf = q < Q_FLOOR ? Q_FLOOR : q;
    return qf * p / (EPS + (1.0 - EPS) * qf);
}

__device__ __forceinline__ double dewpoint(double e) {
    double l = wm_ln(e / MAGNUS_A);
    return MAGNUS_C * l / (MAGNUS_B - l) + T_MELT;
}

__device__ __forceinline__ double rh_percent(double e, double t_k) {
    double rh = 100.0 * e / esat_water(t_k);
    if (rh > 100.0) return 100.0;
    if (rh < 0.0) return 0.0;
    return rh;
}

__device__ __forceinline__ double tn396_temperature(double t_star, double ps, double zs, double p) {
    double lr = wm_ln(p / ps);
    double y;
    if (zs < TN396_Z_LOW) {
        y = ALPHA * lr;
    } else {
        double t0 = t_star + GAMMA * zs;
        double tplat = t0 < TN396_T_PLATEAU ? t0 : TN396_T_PLATEAU;
        double tprime0;
        if (zs <= TN396_Z_HIGH) {
            tprime0 = 0.002 * ((TN396_Z_HIGH - zs) * t0 + (zs - TN396_Z_LOW) * tplat);
        } else {
            tprime0 = tplat;
        }
        if (tprime0 < t_star) {
            y = 0.0;
        } else {
            y = RD * (tprime0 - t_star) / (G * zs) * lr;
        }
    }
    return t_star * (1.0 + y + 0.5 * y * y + y * y * y / 6.0);
}

__device__ __forceinline__ double tn396_height(double tv_star, double ps, double zs, double p) {
    double phis = G * zs;
    double ts = tv_star;
    double t0 = ts + GAMMA * zs;
    double alpha;
    if (ts <= TN396_T_WARM && t0 > TN396_T_WARM) {
        alpha = RD * (TN396_T_WARM - ts) / phis;
    } else if (ts > TN396_T_WARM && t0 > TN396_T_WARM) {
        alpha = 0.0;
        ts = 0.5 * (TN396_T_WARM + ts);
    } else {
        alpha = ALPHA;
    }
    if (ts < TN396_T_COLD) {
        ts = 0.5 * (TN396_T_COLD + ts);
    }
    double lr = wm_ln(p / ps);
    double y = alpha * lr;
    return (phis - RD * ts * lr * (1.0 + 0.5 * y + y * y / 6.0)) / G;
}

// ---- group_b.rs twins ------------------------------------------------------

struct Col {
    int nz;
    double pi[NZ_MAX + 1];
    double lnpi[NZ_MAX + 1];
    double zi[NZ_MAX + 1];
    double pm[NZ_MAX];
    double lnp[NZ_MAX];
    double tk[NZ_MAX];
    double tv[NZ_MAX];
    double q[NZ_MAX];
    double uu[NZ_MAX];
    double vv[NZ_MAX];
};

__device__ __forceinline__ double interp_mass(const Col& s, double p_t, double lnp_t, const double* a) {
    int k = 0;
    while (k + 1 < s.nz) {
        if (p_t <= s.pm[k] && p_t >= s.pm[k + 1]) {
            double w = (lnp_t - s.lnp[k]) / (s.lnp[k + 1] - s.lnp[k]);
            return a[k] + w * (a[k + 1] - a[k]);
        }
        k += 1;
    }
    return dnan();
}

__device__ __forceinline__ double temperature_at(const Col& s, double p_t, double lnp_t, double t_star, double pg, double zs) {
    int nz = s.nz;
    if (p_t < s.pm[nz - 1]) {
        return dnan();
    } else if (p_t <= s.pm[0]) {
        return interp_mass(s, p_t, lnp_t, s.tk);
    } else if (p_t <= pg) {
        double w = (lnp_t - s.lnp[0]) / (s.lnpi[0] - s.lnp[0]);
        return s.tk[0] + w * (t_star - s.tk[0]);
    } else {
        return tn396_temperature(t_star, pg, zs, p_t);
    }
}

__device__ __forceinline__ double virtual_temperature_at(const Col& s, double p_t, double lnp_t, double tv_star, double pg, double zs) {
    int nz = s.nz;
    if (p_t < s.pm[nz - 1]) {
        return dnan();
    } else if (p_t <= s.pm[0]) {
        return interp_mass(s, p_t, lnp_t, s.tv);
    } else if (p_t <= pg) {
        double w = (lnp_t - s.lnp[0]) / (s.lnpi[0] - s.lnp[0]);
        return s.tv[0] + w * (tv_star - s.tv[0]);
    } else {
        return tn396_temperature(tv_star, pg, zs, p_t);
    }
}

// Inputs: Group A's shared column state (st: ST_COUNT mass slots of
// nz * n2 words; si: p_int then z_int, (nz + 1) * n2 words each), PSFC and
// HGT.
extern "C" __global__ void woof_post_b_column(
    int nx, int ny, int nz,
    const float* __restrict__ st, const float* __restrict__ si,
    const float* __restrict__ psfc, const float* __restrict__ hgt,
    const double* __restrict__ levels, int nl,
    const double* __restrict__ mem_levels, int nm,
    float* __restrict__ out32, double* __restrict__ out64)
{
    long long n2 = (long long)nx * ny;
    long long cell = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n2) return;
    long long base2 = 6LL * nl;
    long long vol = (long long)nz * n2;
    long long ivol = (long long)(nz + 1) * n2;
    const float* sp = st + ST_P * vol;
    const float* stk = st + ST_TK * vol;
    const float* stv = st + ST_TV * vol;
    const float* sq = st + ST_Q * vol;
    const float* su = st + ST_U * vol;
    const float* sv = st + ST_V * vol;
    const float* pint = si + SI_PINT * ivol;
    const float* zint = si + SI_ZINT * ivol;

    bool ok = isfinite(psfc[cell]) && isfinite(hgt[cell]);
    for (int k = 0; k < nz; ++k) {
        long long c = k * n2 + cell;
        ok = ok && isfinite(sp[c]) && isfinite(stk[c]) && isfinite(stv[c]) && isfinite(sq[c])
             && isfinite(su[c]) && isfinite(sv[c]);
    }
    for (int k = 0; k <= nz; ++k) {
        ok = ok && isfinite(pint[k * n2 + cell]) && isfinite(zint[k * n2 + cell]);
    }
    if (!ok) {
        for (long long q = 0; q < base2 + N2D; ++q) {
            if (q != base2 + P2_MAPS && q != base2 + P2_MEMBRANE_SLP) out32[q * n2 + cell] = __int_as_float(0x7fc00000);
        }
        for (long long q = 0; q < nm + N64_EXTRA; ++q) out64[q * n2 + cell] = dnan();
        return;
    }

    Col s;
    s.nz = nz;
    for (int k = 0; k <= nz; ++k) {
        s.pi[k] = (double)pint[k * n2 + cell];
        s.lnpi[k] = wm_ln(s.pi[k]);
        s.zi[k] = (double)zint[k * n2 + cell];
    }
    for (int k = 0; k < nz; ++k) {
        long long c = k * n2 + cell;
        s.pm[k] = (double)sp[c];
        s.tk[k] = (double)stk[c];
        s.tv[k] = (double)stv[c];
        s.q[k] = (double)sq[c];
        s.lnp[k] = wm_ln(s.pm[k]);
        s.uu[k] = (double)su[c];
        s.vv[k] = (double)sv[c];
    }

    double pg = s.pi[0];
    double zs = (double)hgt[cell];
    double ratio = pg / s.pm[0] - 1.0;
    double t_star = s.tk[0] * (1.0 + ALPHA * ratio);
    double tv_star = s.tv[0] * (1.0 + ALPHA * ratio);
    double e_bot = vapour_pressure(s.q[0], s.pm[0]);
    double rh_bot = e_bot / esat_water(s.tk[0]);
    if (rh_bot > 1.0) rh_bot = 1.0;

    for (int l = 0; l < nl; ++l) {
        double pt = levels[l];
        double lnpt = wm_ln(pt);
        double tt, e, uo, vo;
        if (pt < s.pm[nz - 1]) {
            tt = dnan(); e = dnan(); uo = dnan(); vo = dnan();
        } else if (pt <= s.pm[0]) {
            tt = interp_mass(s, pt, lnpt, s.tk);
            double qq = interp_mass(s, pt, lnpt, s.q);
            e = vapour_pressure(qq, pt);
            uo = interp_mass(s, pt, lnpt, s.uu);
            vo = interp_mass(s, pt, lnpt, s.vv);
        } else {
            tt = temperature_at(s, pt, lnpt, t_star, pg, zs);
            e = rh_bot * esat_water(tt);
            uo = s.uu[0];
            vo = s.vv[0];
        }
        double td, rh;
        if (disnan(tt)) {
            td = dnan(); rh = dnan();
        } else {
            double td0 = dewpoint(e);
            td = td0 > tt ? tt : td0;
            rh = rh_percent(e, tt);
        }
        double z;
        if (pt < s.pi[nz]) {
            z = dnan();
        } else if (pt <= pg) {
            int kk = 0;
            while (kk + 1 < nz + 1 && !(pt <= s.pi[kk] && pt >= s.pi[kk + 1])) kk += 1;
            double down = s.lnpi[kk] - lnpt;
            double up = lnpt - s.lnpi[kk + 1];
            double a = RD * s.tv[kk] / G;
            z = down <= up ? s.zi[kk] + a * down : s.zi[kk + 1] - a * up;
        } else {
            z = tn396_height(tv_star, pg, zs, pt);
        }
        long long o = 6LL * l;
        out32[(o + 0) * n2 + cell] = to_f32(z);
        out32[(o + 1) * n2 + cell] = to_f32(tt);
        out32[(o + 2) * n2 + cell] = to_f32(td);
        out32[(o + 3) * n2 + cell] = to_f32(rh);
        out32[(o + 4) * n2 + cell] = to_f32(uo);
        out32[(o + 5) * n2 + cell] = to_f32(vo);
    }

    double pw = 0.0;
    for (int k = 0; k < nz; ++k) pw = pw + s.q[k] * (s.pi[k] - s.pi[k + 1]);
    out32[(base2 + P2_PWAT) * n2 + cell] = to_f32(pw / G);

    double ps = (double)psfc[cell];
    double tsv = s.tv[0] * wm_pow(ps / s.pm[0], ALPHA);
    double t0 = tsv + GAMMA * zs;
    if (t0 > NMC_T_WARM) {
        if (tsv <= NMC_T_WARM) {
            t0 = NMC_T_WARM;
        } else {
            double d = tsv - NMC_T_WARM;
            t0 = NMC_T_WARM - NMC_WARM_COEF * d * d;
        }
    }
    double tm = 0.5 * (tsv + t0);
    out32[(base2 + P2_MSLP) * n2 + cell] = to_f32(ps * wm_exp(G * zs / (RD * tm)));

    double t700 = temperature_at(s, MAPS_P_REF, wm_ln(MAPS_P_REF), t_star, pg, zs);
    out32[(base2 + P2_T700) * n2 + cell] = to_f32(t700);
    double tsm = t700 * wm_pow(ps / MAPS_P_REF, ALPHA);
    double maps = ps * wm_pow(1.0 + GAMMA * zs / tsm, G / (RD * GAMMA));
    out64[(nm + Q_MAPS_RAW) * n2 + cell] = maps;

    for (int m = 0; m < nm; ++m) {
        double pt = mem_levels[m];
        out64[(long long)m * n2 + cell] = virtual_temperature_at(s, pt, wm_ln(pt), tv_star, pg, zs);
    }
    out64[(nm + Q_PGROUND) * n2 + cell] = pg;
    out64[(nm + Q_TVGROUND) * n2 + cell] = tv_star;
}

// ---- membrane.rs twins -----------------------------------------------------

// Fill the active levels' values and masks from the column pass.
extern "C" __global__ void woof_post_b_mem_inputs(
    long long n2, int nact, const int* __restrict__ act_m, const double* __restrict__ mem_levels, int nm,
    const double* __restrict__ out64, double* __restrict__ vals, unsigned char* __restrict__ mask)
{
    long long cell = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int a = blockIdx.y;
    if (cell >= n2 || a >= nact) return;
    int m = act_m[a];
    double tv = out64[(long long)m * n2 + cell];
    double g = out64[(nm + Q_PGROUND) * n2 + cell];
    unsigned char mk;
    if (disnan(tv) || disnan(g)) mk = M_MISSING;
    else if (mem_levels[m] > g) mk = M_UNKNOWN;
    else mk = M_FIXED;
    vals[a * n2 + cell] = tv;
    mask[a * n2 + cell] = mk;
}

extern "C" __global__ void woof_post_b_restrict(
    int nxf, int nyf, int nxc, int nyc, int nact,
    const double* __restrict__ vf, const unsigned char* __restrict__ mf,
    double* __restrict__ vcs, unsigned char* __restrict__ mcs)
{
    long long nc = (long long)nxc * nyc;
    long long nf = (long long)nxf * nyf;
    long long cell = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int a = blockIdx.y;
    if (cell >= nc || a >= nact) return;
    int ic = (int)(cell % nxc);
    int jc = (int)(cell / nxc);
    const double* v = vf + a * nf;
    const unsigned char* m = mf + a * nf;
    double sum_fixed = 0.0;
    unsigned n_fixed = 0;
    double sum_all = 0.0;
    unsigned n_all = 0;
    for (int t = 0; t < 4; ++t) {
        int i = 2 * ic + (t & 1);
        int j = 2 * jc + (t >> 1);
        if (i < nxf && j < nyf) {
            long long c = (long long)j * nxf + i;
            unsigned char mk = m[c];
            if (mk != M_MISSING) {
                sum_all = sum_all + v[c];
                n_all += 1;
                if (mk == M_FIXED) {
                    sum_fixed = sum_fixed + v[c];
                    n_fixed += 1;
                }
            }
        }
    }
    double outv;
    unsigned char outm;
    if (n_all == 0) { outv = dnan(); outm = M_MISSING; }
    else if (n_fixed > 0) { outv = sum_fixed / (double)n_fixed; outm = M_FIXED; }
    else { outv = sum_all / (double)n_all; outm = M_UNKNOWN; }
    vcs[a * nc + cell] = outv;
    mcs[a * nc + cell] = outm;
}

extern "C" __global__ void woof_post_b_jacobi(
    int nx, int ny, int nact,
    const double* __restrict__ src, const unsigned char* __restrict__ mask, double* __restrict__ dst)
{
    long long n = (long long)nx * ny;
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int a = blockIdx.y;
    if (c >= n || a >= nact) return;
    const double* s = src + a * n;
    const unsigned char* m = mask + a * n;
    int i = (int)(c % nx);
    int j = (int)(c / nx);
    double r;
    if (m[c] != M_UNKNOWN) {
        r = s[c];
    } else {
        double sum = 0.0;
        unsigned k = 0;
        if (i > 0 && m[c - 1] != M_MISSING) { sum = sum + s[c - 1]; k += 1; }
        if (i + 1 < nx && m[c + 1] != M_MISSING) { sum = sum + s[c + 1]; k += 1; }
        if (j > 0 && m[c - nx] != M_MISSING) { sum = sum + s[c - nx]; k += 1; }
        if (j + 1 < ny && m[c + nx] != M_MISSING) { sum = sum + s[c + nx]; k += 1; }
        r = k == 0 ? s[c] : sum / (double)k;
    }
    dst[a * n + c] = r;
}

extern "C" __global__ void woof_post_b_prolong(
    int nxf, int nyf, int nxc, int nyc, int nact,
    const double* __restrict__ vcs, const unsigned char* __restrict__ mf, double* __restrict__ vf)
{
    long long nf = (long long)nxf * nyf;
    long long nc = (long long)nxc * nyc;
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int a = blockIdx.y;
    if (c >= nf || a >= nact) return;
    int i = (int)(c % nxf);
    int j = (int)(c / nxf);
    if (mf[a * nf + c] == M_UNKNOWN) {
        vf[a * nf + c] = vcs[a * nc + (long long)(j / 2) * nxc + i / 2];
    }
}

// Underground heights and membrane sea-level pressure; 1000 hPa height
// replaced where 1000 hPa is underground (z_plane >= 0).
extern "C" __global__ void woof_post_b_mem_column(
    long long n2, int nm, const double* __restrict__ mem_levels, const int* __restrict__ act_of_m,
    const double* __restrict__ solved, const double* __restrict__ out64,
    const float* __restrict__ hgt, long long slp_plane, long long z_plane, float* __restrict__ out32)
{
    long long cell = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n2) return;
    double pg = out64[(nm + Q_PGROUND) * n2 + cell];
    if (disnan(pg)) {
        out32[slp_plane * n2 + cell] = __int_as_float(0x7fc00000);
        return;
    }
    double tvg = out64[(nm + Q_TVGROUND) * n2 + cell];
    double zs = (double)hgt[cell];
    double p_prev = pg;
    double z_prev = zs;
    double tv_prev = tvg;
    double slp = dnan();
    double z1000 = dnan();
    if (zs <= 0.0) {
        slp = pg * wm_exp(G * zs / (RD * tvg));
    }
    for (int m = 0; m < nm; ++m) {
        double pt = mem_levels[m];
        if (pt <= pg) continue;
        double tvm = solved[(long long)act_of_m[m] * n2 + cell];
        double tbar = 0.5 * (tv_prev + tvm);
        double z = z_prev - RD * tbar * wm_ln(pt / p_prev) / G;
        if (disnan(slp) && z <= 0.0) {
            slp = p_prev * wm_exp(G * z_prev / (RD * tbar));
        }
        if (pt == 100000.0) z1000 = z;
        p_prev = pt;
        z_prev = z;
        tv_prev = tvm;
    }
    if (disnan(slp)) {
        slp = p_prev * wm_exp(G * z_prev / (RD * tv_prev));
    }
    out32[slp_plane * n2 + cell] = to_f32(slp);
    if (z_plane >= 0 && !disnan(z1000)) out32[z_plane * n2 + cell] = to_f32(z1000);
}

// ---- smooth.rs twins -------------------------------------------------------

__device__ __forceinline__ double pass_cell(const double* src, long long c, bool has_lo, long long lo, bool has_hi, long long hi) {
    double x = src[c];
    if (disnan(x)) return x;
    double sum = 0.5 * x;
    double w = 0.5;
    if (has_lo && !disnan(src[lo])) { sum = sum + 0.25 * src[lo]; w = w + 0.25; }
    if (has_hi && !disnan(src[hi])) { sum = sum + 0.25 * src[hi]; w = w + 0.25; }
    return sum / w;
}

extern "C" __global__ void woof_post_b_smooth(int nx, int ny, int along_y, const double* __restrict__ src, double* __restrict__ dst)
{
    long long n = (long long)nx * ny;
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    int i = (int)(c % nx);
    int j = (int)(c / nx);
    if (along_y) dst[c] = pass_cell(src, c, j > 0, c - nx, j + 1 < ny, c + nx);
    else dst[c] = pass_cell(src, c, i > 0, c - 1, i + 1 < nx, c + 1);
}

extern "C" __global__ void woof_post_b_copy64(long long n, const double* __restrict__ src, long long src_off, double* __restrict__ dst)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    dst[c] = src[src_off + c];
}

extern "C" __global__ void woof_post_b_store_f32(long long n, const double* __restrict__ src, float* __restrict__ dst, long long dst_off)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    dst[dst_off + c] = to_f32(src[c]);
}
