// SPDX-License-Identifier: Apache-2.0
//
// WOOF post-processor, Group C (parcels, shear, helicity, severe indices),
// CUDA kernels, revision 1.  Written clean-room from public science:
//
// - Constants: NCAR/TN-556+STR (2019), WRF ARW v4 technical note.
// - Saturation vapour pressure: J. Appl. Meteor. 35 (1996) 601-609 (AERK).
// - LCL temperature (eq. 21), theta-e (eq. 43): Mon. Wea. Rev. 108 (1980)
//   1046-1053.
// - Pseudoadiabat by Newton iteration: Mon. Wea. Rev. 136 (2008) 2764-2785.
// - Virtual temperature buoyancy: Wea. Forecasting 9 (1994) 625-629.
// - CAPE/CIN: AMS Glossary of Meteorology; parcels: SPC Mesoanalysis help,
//   Wea. Forecasting 17 (2002) 885-890.
// - Storm motion: Wea. Forecasting 15 (2000) 61-79.  SRH: 16th Conf. on
//   Severe Local Storms (1990) 588-592.  STP: Wea. Forecasting 18 (2003)
//   1243-1261 and 27 (2012) 1136-1154.  EHI: SHARP user guide (1991).
//
// Every device function mirrors the Rust CPU reference in
// src/severe/{thermo,parcel,wind}.rs statement for statement, in double
// precision.  One thread owns one column (and one parcel type in the
// parcel kernel); there are no atomics, no shared-memory reductions and the
// Newton iteration count is fixed, so the result of a launch does not depend
// on scheduling.  Built with --fmad=false and precise division/sqrt.
// Elementary functions come from the shared maths library woof_math.cuh
// (wm_ln, wm_exp, wm_pow, wm_sqrt), the twin of src/math.rs, never from
// libdevice, so the CPU and GPU paths return the same bits.

#include "woof_math.cuh"

#define NPLANES 22
#define G_ 9.81
#define RD_ 287.0
#define CP_ (3.5 * 287.0)
#define RV_ 461.6
#define P0_ 100000.0
#define EPS_ (287.0 / 461.6)
#define KAPPA_ (287.0 / (3.5 * 287.0))
#define R_FLOOR_ 1.0e-10
#define NEWTON_ITERS_ 10
#define Z10_ 10.0
#define BUNKERS_D_ 7.5
// Method switches, RULED 2026-10-05 (RULINGS.md items 3 and 4); twins of the
// Rust consts in src/severe/wind.rs, change both together.
// 0: 0-6 km shear from the 10 m and 6 km endpoints (SPEC D14); 1: from the
// 0-0.5 km and 5.5-6 km layer means.
#define SHEAR06_LAYER_MEANS 0
// 0: STP as published (negative with negative SRH); 1: floored at 0.
#define STP_FLOOR_ZERO 1

__device__ __forceinline__ double fmax_(double a, double b) { return a > b ? a : b; }
__device__ __forceinline__ double fmin_(double a, double b) { return a < b ? a : b; }

__device__ __forceinline__ double es_liquid(double t) {
    double tc = t - 273.15;
    return 610.94 * wm_exp(17.625 * tc / (tc + 243.04));
}
__device__ __forceinline__ double vapour_pressure(double r, double p) { return r * p / (EPS_ + r); }
__device__ __forceinline__ double rsat(double t, double p) {
    double es = es_liquid(t);
    return EPS_ * es / fmax_(p - es, 1.0e-3 * p);
}
__device__ __forceinline__ double tv_(double t, double r) { return t * (1.0 + r / EPS_) / (1.0 + r); }
__device__ __forceinline__ double theta_(double t, double p) { return t * wm_pow(P0_ / p, KAPPA_); }
__device__ __forceinline__ double t_from_theta(double th, double p) { return th * wm_pow(p / P0_, KAPPA_); }
__device__ __forceinline__ double lcl_temperature(double t, double r, double p) {
    r = fmax_(r, R_FLOOR_);
    double u = vapour_pressure(r, p) / es_liquid(t);
    return 1.0 / (1.0 / (t - 55.0) - wm_ln(u) / 2840.0) + 55.0;
}
__device__ __forceinline__ double theta_e(double t, double p, double r, double tl) {
    double rg = 1000.0 * fmax_(r, R_FLOOR_);
    double ex = 0.2854 * (1.0 - 0.28e-3 * rg);
    return t * wm_pow(P0_ / p, ex) * wm_exp((3.376 / tl - 0.00254) * rg * (1.0 + 0.81e-3 * rg));
}
__device__ __forceinline__ double theta_e_parcel(double t, double p, double r) {
    r = fmin_(fmax_(r, R_FLOOR_), rsat(t, p));
    return theta_e(t, p, r, lcl_temperature(t, r, p));
}
__device__ double pseudoadiabat_t(double ln_te, double p, double t) {
    double lpr = wm_ln(P0_ / p);
    for (int i = 0; i < NEWTON_ITERS_; ++i) {
        double tc = t - 273.15;
        double es = 610.94 * wm_exp(17.625 * tc / (tc + 243.04));
        double dlnes = 17.625 * 243.04 / ((tc + 243.04) * (tc + 243.04));
        double denom = fmax_(p - es, 1.0e-3 * p);
        double r = EPS_ * es / denom;
        double dr = r * (p / denom) * dlnes;
        double rg = 1000.0 * r;
        double drg = 1000.0 * dr;
        double kap = 0.2854 * (1.0 - 0.28e-3 * rg);
        double dkap = -0.2854 * 0.28e-3 * drg;
        double a = 3.376 / t - 0.00254;
        double da = -3.376 / (t * t);
        double b = rg * (1.0 + 0.81e-3 * rg);
        double db = drg * (1.0 + 1.62e-3 * rg);
        double f = wm_ln(t) + kap * lpr + a * b - ln_te;
        double df = 1.0 / t + dkap * lpr + da * b + a * db;
        t -= f / df;
        t = fmin_(fmax_(t, 100.0), 400.0);
    }
    return t;
}

// ---- column accessors (level-major [k][cell]) ----
struct Col {
    int nz, n, c;
    const float *p, *tk, *r, *p_int, *z_int;
    double zsfc, psfc, t2, q2, p2;
    __device__ double P(int k) const { return (double)p[(long long)k * n + c]; }
    __device__ double T(int k) const { return (double)tk[(long long)k * n + c]; }
    __device__ double R(int k) const { return (double)r[(long long)k * n + c]; }
    __device__ double PI(int k) const { return (double)p_int[(long long)k * n + c]; }
    __device__ double ZI(int k) const { return (double)z_int[(long long)k * n + c]; }
    __device__ double tv_env(int k) const { return tv_(T(k), fmax_(R(k), 0.0)); }
};

__device__ double tv_env_at(const Col& col, double px) {
    double tv2 = tv_(col.t2, fmax_(col.q2, 0.0));
    if (px >= col.p2) return tv2;
    double pa = col.p2, ta = tv2;
    for (int k = 0; k < col.nz; ++k) {
        double pb = col.P(k);
        double tb = col.tv_env(k);
        if (px >= pb) {
            double w = wm_ln(pa / px) / wm_ln(pa / pb);
            return ta + (tb - ta) * w;
        }
        pa = pb;
        ta = tb;
    }
    return __longlong_as_double(0x7ff8000000000000LL);
}

__device__ void layer_mean(const Col& col, double p_top, double p_bot, double* th_out, double* r_out) {
    double w = 0.0, sth = 0.0, sr = 0.0;
    for (int k = 0; k < col.nz; ++k) {
        double lo = fmin_(col.PI(k), p_bot);
        double hi = fmax_(col.PI(k + 1), p_top);
        double overlap = lo - hi;
        if (overlap > 0.0) {
            w += overlap;
            sth += overlap * theta_(col.T(k), col.P(k));
            sr += overlap * fmax_(col.R(k), 0.0);
        }
    }
    if (w > 0.0) {
        *th_out = sth / w;
        *r_out = sr / w;
    } else {
        *th_out = __longlong_as_double(0x7ff8000000000000LL);
        *r_out = __longlong_as_double(0x7ff8000000000000LL);
    }
}

struct Acc {
    bool lfc;
    double neg_below, run_pos, cape, cin;
};

__device__ void acc_add(Acc& a, double xa, double ba, double xb, double bb, bool above) {
    double dx = xa - xb;
    double pos, neg;
    if (ba > 0.0 && bb > 0.0) {
        pos = 0.5 * (ba + bb) * dx;
        neg = 0.0;
    } else if (ba <= 0.0 && bb <= 0.0) {
        pos = 0.0;
        neg = 0.5 * (ba + bb) * dx;
    } else if (ba > 0.0) {
        double t = ba / (ba - bb);
        pos = 0.5 * ba * t * dx;
        neg = 0.5 * bb * (1.0 - t) * dx;
    } else {
        double t = ba / (ba - bb);
        pos = 0.5 * bb * (1.0 - t) * dx;
        neg = 0.5 * ba * t * dx;
    }
    if (a.lfc) {
        a.cape += pos;
        return;
    }
    bool pa = ba > 0.0, pb = bb > 0.0;
    if (pa && pb) {
        if (above) {
            a.lfc = true;
            a.cin = a.neg_below;
            a.cape = a.run_pos + pos;
        } else {
            a.run_pos += pos;
        }
    } else if (!pa && !pb) {
        a.neg_below += neg;
        a.run_pos = 0.0;
    } else if (pa) {
        if (above) {
            a.lfc = true;
            a.cin = a.neg_below;
            a.cape = a.run_pos + pos;
        } else {
            a.run_pos = 0.0;
            a.neg_below += neg;
        }
    } else if (above) {
        a.lfc = true;
        a.cin = a.neg_below + neg;
        a.cape = pos;
    } else {
        a.neg_below += neg;
        a.run_pos = pos;
    }
}

__device__ void lcl_tp(double p_o, double t_o, double r_o, double* t_l, double* p_l) {
    double tl = lcl_temperature(t_o, r_o, p_o);
    if (tl >= t_o) {
        *t_l = t_o;
        *p_l = p_o;
    } else {
        *t_l = tl;
        *p_l = p_o * wm_pow(tl / t_o, 1.0 / KAPPA_);
    }
}

__device__ void lift(const Col& col, double p_o, double t_o, double r_o, double tv_env_o, double* cape, double* cin) {
    r_o = fmin_(fmax_(r_o, R_FLOOR_), rsat(t_o, p_o));
    double t_l, p_lcl;
    lcl_tp(p_o, t_o, r_o, &t_l, &p_lcl);
    double th_o = theta_(t_o, p_o);
    double ln_te = wm_ln(theta_e(t_o, p_o, r_o, t_l));
    Acc acc = {false, 0.0, 0.0, 0.0, 0.0};
    double p_prev = p_o;
    double x_prev = wm_ln(p_o);
    double b_prev = tv_(t_o, r_o) - tv_env_o;
    double tve_prev = tv_env_o;
    double t_moist = t_l;
    for (int k = 0; k < col.nz; ++k) {
        double pk = col.P(k);
        if (pk >= p_prev) continue;
        double xk = wm_ln(pk);
        double tve_k = col.tv_env(k);
        if (p_lcl < p_prev && p_lcl > pk) {
            double xl = wm_ln(p_lcl);
            double tve_l = tve_prev + (tve_k - tve_prev) * (x_prev - xl) / (x_prev - xk);
            double b_l = tv_(t_l, r_o) - tve_l;
            acc_add(acc, x_prev, b_prev, xl, b_l, p_prev <= p_lcl);
            p_prev = p_lcl;
            x_prev = xl;
            b_prev = b_l;
        }
        double tvp;
        if (pk >= p_lcl) {
            tvp = tv_(t_from_theta(th_o, pk), r_o);
        } else {
            t_moist = pseudoadiabat_t(ln_te, pk, t_moist);
            tvp = tv_(t_moist, rsat(t_moist, pk));
        }
        double b_k = tvp - tve_k;
        acc_add(acc, x_prev, b_prev, xk, b_k, p_prev <= p_lcl);
        p_prev = pk;
        x_prev = xk;
        b_prev = b_k;
        tve_prev = tve_k;
    }
    if (acc.lfc) {
        *cape = RD_ * acc.cape;
        *cin = RD_ * acc.cin;
    } else {
        *cape = 0.0;
        *cin = 0.0;
    }
}

__device__ double lcl_height_agl(const Col& col, double p_o, double t_o, double r_o) {
    if (!(isfinite(p_o) && isfinite(t_o) && r_o > 0.0)) return __longlong_as_double(0x7ff8000000000000LL);
    double t_l, p_lcl;
    lcl_tp(p_o, t_o, r_o, &t_l, &p_lcl);
    if (p_lcl >= p_o || p_lcl >= col.PI(0)) return 0.0;
    int nz = col.nz;
    if (p_lcl < col.PI(nz)) return __longlong_as_double(0x7ff8000000000000LL);
    for (int k = 0; k < nz; ++k) {
        double pa = col.PI(k), pb = col.PI(k + 1);
        if (p_lcl <= pa && p_lcl > pb) {
            double w = wm_ln(pa / p_lcl) / wm_ln(pa / pb);
            double z = col.ZI(k) + (col.ZI(k + 1) - col.ZI(k)) * w;
            return fmax_(z - col.zsfc, 0.0);
        }
    }
    return fmax_(col.ZI(nz) - col.zsfc, 0.0);
}

__device__ __forceinline__ bool finite_(double v) { return isfinite(v); }

// Parcel kernel: blockIdx.y selects the parcel type
// 0 = surface-based (+ its LCL height), 1 = mixed layer, 2 = most unstable,
// 3 = best of six 30 hPa layers (+ lcl_height from layer 0).
extern "C" __global__ void woof_severe_parcels_v1(
    int nz, int n,
    const float* __restrict__ p, const float* __restrict__ tk, const float* __restrict__ r,
    const float* __restrict__ p_int, const float* __restrict__ z_int,
    const float* __restrict__ zsfc, const float* __restrict__ psfc,
    const float* __restrict__ t2, const float* __restrict__ q2, const float* __restrict__ p2,
    float* __restrict__ out, double* __restrict__ keep)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int kind = blockIdx.y;
    if (c >= n) return;
    const double nan_ = __longlong_as_double(0x7ff8000000000000LL);
    Col col;
    col.nz = nz; col.n = n; col.c = c;
    col.p = p; col.tk = tk; col.r = r; col.p_int = p_int; col.z_int = z_int;
    col.zsfc = (double)zsfc[c]; col.psfc = (double)psfc[c];
    col.t2 = (double)t2[c]; col.q2 = (double)q2[c]; col.p2 = (double)p2[c];
    bool ok = finite_(col.zsfc) && finite_(col.psfc) && finite_(col.t2) && finite_(col.q2) && finite_(col.p2);
    for (int k = 0; k < nz; ++k) ok = ok && finite_(col.P(k)) && finite_(col.T(k)) && finite_(col.R(k));
    for (int k = 0; k <= nz; ++k) ok = ok && finite_(col.PI(k)) && finite_(col.ZI(k));
    long long N = n;
    if (!ok) {
        if (kind == 0) {
            out[0 * N + c] = nan_; out[1 * N + c] = nan_; out[21 * N + c] = nan_;
            keep[0 * N + c] = nan_; keep[1 * N + c] = nan_; keep[2 * N + c] = nan_;
        }
        if (kind == 1) { out[2 * N + c] = nan_; out[3 * N + c] = nan_; }
        if (kind == 2) { out[4 * N + c] = nan_; out[5 * N + c] = nan_; }
        if (kind == 3) { out[6 * N + c] = nan_; out[7 * N + c] = nan_; out[8 * N + c] = nan_; }
        return;
    }
    double cape = nan_, cin = nan_;
    if (kind == 0) {
        double r2 = fmax_(col.q2, R_FLOOR_);
        double tv2 = tv_(col.t2, r2);
        lift(col, col.p2, col.t2, r2, tv2, &cape, &cin);
        double sblcl = lcl_height_agl(col, col.p2, col.t2, col.q2);
        out[0 * N + c] = (float)cape;
        out[1 * N + c] = (float)cin;
        out[21 * N + c] = (float)sblcl;
        keep[0 * N + c] = cape;
        keep[1 * N + c] = cin;
        keep[2 * N + c] = sblcl;
    } else if (kind == 1) {
        double pg = col.PI(0);
        double th_ml, r_ml;
        layer_mean(col, pg - 9000.0, pg, &th_ml, &r_ml);
        double t_ml = t_from_theta(th_ml, col.psfc);
        lift(col, col.psfc, t_ml, r_ml, tv_env_at(col, col.psfc), &cape, &cin);
        out[2 * N + c] = (float)cape;
        out[3 * N + c] = (float)cin;
    } else if (kind == 2) {
        int best_k = -1;
        double best_te = -1.0 / 0.0;
        for (int k = 0; k < nz; ++k) {
            if (col.P(k) < col.psfc - 30000.0) break;
            double te = theta_e_parcel(col.T(k), col.P(k), fmax_(col.R(k), R_FLOOR_));
            if (te > best_te) {
                best_te = te;
                best_k = k;
            }
        }
        if (best_k >= 0) {
            lift(col, col.P(best_k), col.T(best_k), col.R(best_k), col.tv_env(best_k), &cape, &cin);
        }
        out[4 * N + c] = (float)cape;
        out[5 * N + c] = (float)cin;
    } else {
        double pg = col.PI(0);
        double b_te = -1.0 / 0.0, b_p = 0.0, b_t = 0.0, b_r = 0.0;
        double lclh = nan_;
        for (int i = 0; i < 6; ++i) {
            double p_bot = pg - 3000.0 * (double)i;
            double th, rr;
            layer_mean(col, p_bot - 3000.0, p_bot, &th, &rr);
            double pm = p_bot - 1500.0;
            double t = t_from_theta(th, pm);
            double te = theta_e_parcel(t, pm, fmax_(rr, R_FLOOR_));
            if (i == 0) lclh = lcl_height_agl(col, col.psfc, t_from_theta(th, col.psfc), rr);
            if (te > b_te) {
                b_te = te; b_p = pm; b_t = t; b_r = rr;
            }
        }
        if (isfinite(b_te)) {
            lift(col, b_p, b_t, b_r, tv_env_at(col, b_p), &cape, &cin);
        }
        out[6 * N + c] = (float)cape;
        out[7 * N + c] = (float)cin;
        out[8 * N + c] = (float)lclh;
    }
}

// ---- wind profile: points (0, w10), (10, w10), mass levels above 10 m ----
struct Wind {
    int nz, n, c;
    const float *zm, *u, *v;
    double zsfc, u10, v10;
    __device__ double Z(int k) const { return (double)zm[(long long)k * n + c] - zsfc; }
    __device__ double U(int k) const { return (double)u[(long long)k * n + c]; }
    __device__ double V(int k) const { return (double)v[(long long)k * n + c]; }
};

__device__ bool wind_at(const Wind& w, double h, double* uo, double* vo) {
    if (h <= Z10_) { *uo = w.u10; *vo = w.v10; return true; }
    // first segment (0 -> 10 m) cannot contain h > 10.
    double za = Z10_, ua = w.u10, va = w.v10;
    for (int k = 0; k < w.nz; ++k) {
        double zb = w.Z(k);
        if (!(zb > Z10_)) continue;
        double ub = w.U(k), vb = w.V(k);
        if (h >= za && h <= zb) {
            double t = (h - za) / (zb - za);
            *uo = ua + (ub - ua) * t;
            *vo = va + (vb - va) * t;
            return true;
        }
        za = zb; ua = ub; va = vb;
    }
    return false;
}

__device__ __forceinline__ void seg_mean(double za, double ua, double va, double zb, double ub, double vb,
                                         double h1, double h2, double* su, double* sv) {
    double lo = fmax_(za, h1);
    double hi = fmin_(zb, h2);
    if (hi > lo) {
        double tl = (lo - za) / (zb - za);
        double th = (hi - za) / (zb - za);
        double ul = ua + (ub - ua) * tl, vl = va + (vb - va) * tl;
        double uh = ua + (ub - ua) * th, vh = va + (vb - va) * th;
        *su += 0.5 * (ul + uh) * (hi - lo);
        *sv += 0.5 * (vl + vh) * (hi - lo);
    }
}

__device__ bool wind_layer_mean(const Wind& w, double h1, double h2, double* uo, double* vo) {
    double ztop = Z10_;
    for (int k = 0; k < w.nz; ++k) {
        double z = w.Z(k);
        if (z > Z10_) ztop = z;
    }
    if (ztop < h2) return false;
    double su = 0.0, sv = 0.0;
    seg_mean(0.0, w.u10, w.v10, Z10_, w.u10, w.v10, h1, h2, &su, &sv);
    double za = Z10_, ua = w.u10, va = w.v10;
    for (int k = 0; k < w.nz; ++k) {
        double zb = w.Z(k);
        if (!(zb > Z10_)) continue;
        double ub = w.U(k), vb = w.V(k);
        seg_mean(za, ua, va, zb, ub, vb, h1, h2, &su, &sv);
        za = zb; ua = ub; va = vb;
    }
    *uo = su / (h2 - h1);
    *vo = sv / (h2 - h1);
    return true;
}

__device__ bool wind_srh(const Wind& w, double h, double cu, double cv, double* out) {
    double pu = w.u10, pv = w.v10, pz = Z10_;
    double sum = 0.0;
    for (int k = 0; k < w.nz; ++k) {
        double z = w.Z(k);
        if (!(z > Z10_)) continue;
        double u = w.U(k), v = w.V(k);
        double nu, nv;
        bool done;
        if (z < h) {
            nu = u; nv = v; done = false;
        } else {
            double t = (h - pz) / (z - pz);
            nu = pu + (u - pu) * t;
            nv = pv + (v - pv) * t;
            done = true;
        }
        sum += (nu - cu) * (pv - cv) - (pu - cu) * (nv - cv);
        if (done) { *out = sum; return true; }
        pu = nu; pv = nv; pz = z;
    }
    return false;
}

extern "C" __global__ void woof_severe_winds_v1(
    int nz, int n,
    const float* __restrict__ z_mass, const float* __restrict__ u, const float* __restrict__ v,
    const float* __restrict__ zsfc, const float* __restrict__ u10, const float* __restrict__ v10,
    float* __restrict__ out, double* __restrict__ keep)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const double nan_ = __longlong_as_double(0x7ff8000000000000LL);
    long long N = n;
    Wind w;
    w.nz = nz; w.n = n; w.c = c; w.zm = z_mass; w.u = u; w.v = v;
    w.zsfc = (double)zsfc[c]; w.u10 = (double)u10[c]; w.v10 = (double)v10[c];
    bool ok = finite_(w.zsfc) && finite_(w.u10) && finite_(w.v10);
    for (int k = 0; k < nz; ++k) ok = ok && finite_(w.Z(k)) && finite_(w.U(k)) && finite_(w.V(k));
    double o[10];
    for (int i = 0; i < 10; ++i) o[i] = nan_;
    if (ok) {
        double um, vm, ul, vl, uh, vh;
        if (wind_layer_mean(w, 0.0, 6000.0, &um, &vm) && wind_layer_mean(w, 0.0, 500.0, &ul, &vl) &&
            wind_layer_mean(w, 5500.0, 6000.0, &uh, &vh)) {
            double su = uh - ul, sv = vh - vl;
            double mag = wm_sqrt(su * su + sv * sv);
            double cu, cv;
            if (mag > 0.0) {
                cu = um + BUNKERS_D_ * sv / mag;
                cv = vm - BUNKERS_D_ * su / mag;
            } else {
                cu = um;
                cv = vm;
            }
            o[0] = cu;
            o[1] = cv;
            double s;
            if (wind_srh(w, 1000.0, cu, cv, &s)) o[2] = s;
            if (wind_srh(w, 3000.0, cu, cv, &s)) o[3] = s;
        }
        double ua, va;
        if (wind_at(w, 1000.0, &ua, &va)) {
            o[4] = ua - w.u10;
            o[5] = va - w.v10;
            o[8] = wm_sqrt(o[4] * o[4] + o[5] * o[5]);
        }
        double du = 0.0, dv = 0.0;
        bool have06 = false;
#if SHEAR06_LAYER_MEANS
        double uh6, vh6, ul6, vl6;
        if (wind_layer_mean(w, 5500.0, 6000.0, &uh6, &vh6) && wind_layer_mean(w, 0.0, 500.0, &ul6, &vl6)) {
            du = uh6 - ul6;
            dv = vh6 - vl6;
            have06 = true;
        }
#else
        if (wind_at(w, 6000.0, &ua, &va)) {
            du = ua - w.u10;
            dv = va - w.v10;
            have06 = true;
        }
#endif
        if (have06) {
            o[6] = du;
            o[7] = dv;
            o[9] = wm_sqrt(o[6] * o[6] + o[7] * o[7]);
        }
    }
    for (int i = 0; i < 10; ++i) out[(9 + i) * N + c] = (float)o[i];
    keep[3 * N + c] = o[2];
    keep[4 * N + c] = o[9];
}

// STP and EHI.  The CPU path combines its inputs in f64 before rounding, so
// this kernel reads the f64 copies the other two kernels keep (sbcape, sbcin,
// surface-parcel LCL height, SRH 0-1 km, 0-6 km bulk shear), not the rounded
// f32 planes.
extern "C" __global__ void woof_severe_indices_v1(
    int n, const double* __restrict__ keep, float* __restrict__ out)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    long long N = n;
    const double nan_ = __longlong_as_double(0x7ff8000000000000LL);
    double sbcape = keep[0 * N + c], sbcin = keep[1 * N + c], sblcl = keep[2 * N + c];
    double srh01 = keep[3 * N + c], bwd06 = keep[4 * N + c];
    double stp = nan_;
    if (finite_(sbcape) && finite_(sbcin) && finite_(sblcl) && finite_(srh01) && finite_(bwd06)) {
        double lcl_term = sblcl < 1000.0 ? 1.0 : (sblcl > 2000.0 ? 0.0 : (2000.0 - sblcl) / 1000.0);
        double shear_term = bwd06 < 12.5 ? 0.0 : (bwd06 > 30.0 ? 1.5 : bwd06 / 20.0);
        double cin_term = sbcin > -50.0 ? 1.0 : (sbcin < -200.0 ? 0.0 : (200.0 + sbcin) / 150.0);
        stp = (sbcape / 1500.0) * lcl_term * (srh01 / 150.0) * shear_term * cin_term;
#if STP_FLOOR_ZERO
        if (stp < 0.0) stp = 0.0;
#endif
    }
    double ehi = (finite_(sbcape) && finite_(srh01)) ? sbcape * srh01 / 160000.0 : nan_;
    out[19 * N + c] = (float)stp;
    out[20 * N + c] = (float)ehi;
}
