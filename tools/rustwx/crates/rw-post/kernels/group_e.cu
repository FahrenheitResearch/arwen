// SPDX-License-Identifier: Apache-2.0
//
// WOOF post-processor, group E: boundary layer, near-surface winds and
// isotherm levels.  CUDA device code.
//
// Clean-room implementation written only from the WOOF clean-room post
// specification (section 8, "Group E") and the public sources it cites:
//   - A Description of the Advanced Research WRF Model Version 4,
//     NCAR/TN-556+STR (2019): staggering, constants, theta and pressure.
//   - Vogelezang and Holtslag, Boundary-Layer Meteorol. 81 (1996) 245-269:
//     bulk Richardson boundary-layer height with the b u*^2 term (b = 100).
//   - Troen and Mahrt, Boundary-Layer Meteorol. 37 (1986) 129-148, and
//     Seidel et al., J. Geophys. Res. 117 (2012) D17106: critical Ri = 0.25.
//   - ECMWF IFS Documentation Part IV (wind gusts) and Panofsky et al.,
//     Boundary-Layer Meteorol. 11 (1977) 355-361: similarity gust.
//   - Brasseur, Mon. Wea. Rev. 129 (2001) 5-25: wind gust estimate.
//   - AMS Glossary of Meteorology: freezing level, tropopause (WMO 1957
//     lapse-rate definition), virtual temperature, hypsometric equation.
//   - WMO-No. 306 Manual on Codes: GRIB2 level types.
//
// Every value is computed in binary64 with only + - * / sqrt and the shared
// maths library (woof_math.cuh), in exactly the operation order of the
// Rust CPU path (src/group_e/column.rs).  Built with --fmad=false,
// --prec-div=true, --prec-sqrt=true, --ftz=false, so a frame is bitwise the
// same on the CPU and on every GPU.  No atomics; one thread owns one column.

typedef unsigned long long u64;
typedef long long i64;

#define W_G 9.81
#define W_RD 287.0
#define W_CP (3.5 * W_RD)
#define W_RV 461.6
#define W_P0 100000.0
#define W_EPS (W_RD / W_RV)
#define W_KAPPA (W_RD / W_CP)
#define W_LV 2.501e6
#define W_KARMAN 0.4
#define W_RI_CRIT 0.25
#define W_B_UST 100.0
#define W_C_UGN 7.71
#define W_DEN_FLOOR 1.0e-10
#define W_TROP_LAPSE 0.002
#define W_TROP_DEPTH 2000.0
// RULED (RULINGS.md item 8): tropopause searched from 500 to 50 hPa, twins
// of TROP_P_MAX / TROP_P_MIN in src/group_e/column.rs.
#define W_TROP_P_MAX 50000.0
#define W_TROP_P_MIN 5000.0

#define W_SQRT2 1.4142135623730951
#define W_LN2_HI 6.93147180369123816490e-01
#define W_LN2_LO 1.90821492927058770002e-10
#define W_INV_LN2 1.44269504088896338700e+00

__device__ __forceinline__ double w_nan() { return __longlong_as_double(0x7ff8000000000000LL); }
__device__ __forceinline__ bool w_isnan(double x) { return x != x; }
__device__ __forceinline__ float w_out(double x) {
    if (x != x) return __int_as_float(0x7fc00000);
    return (float)x;
}

// Elementary functions: the shared maths library (woof_math.cuh, twin of
// src/math.rs): wm_ln, wm_exp, wm_cbrt, wm_sqrt.
#include "woof_math.cuh"

__device__ __forceinline__ double w_ld(const float* a, u64 i) { return (double)a[i]; }

// ---------------------------------------------------------------------------

struct WCross {
    double z;
    double p;
};

// Height and pressure where the profile crosses tc going from warm (below)
// to cold (above) between point a (lower) and point b (upper).
__device__ __forceinline__ WCross w_cross(double za, double ta, double pa,
                                          double zb, double tb, double pb, double tc) {
    WCross c;
    double f = (ta - tc) / (ta - tb);
    c.z = za + f * (zb - za);
    double la = wm_ln(pa);
    double lb = wm_ln(pb);
    c.p = wm_exp(la + f * (lb - la));
    return c;
}

// Profile point n of the isotherm search: point 0 is the shelter point when
// it exists, the mass levels follow.
#define W_PT(n, Z, T, P)                                                     \
    do {                                                                     \
        if (has_sh) {                                                        \
            if ((n) == 0) { Z = z_sh; T = t_sh; P = p_sh; }                  \
            else { u64 mm_ = (u64)((n) - 1) * ncell + cell;                  \
                   Z = w_ld(z_mass, mm_); T = w_ld(tk, mm_); P = w_ld(pcoord, mm_); } \
        } else {                                                             \
            u64 mm_ = (u64)(n) * ncell + cell;                               \
            Z = w_ld(z_mass, mm_); T = w_ld(tk, mm_); P = w_ld(pcoord, mm_); \
        }                                                                    \
    } while (0)

// Flags (bit set = carrier present / option on).
#define WF_UST 1
#define WF_T2 2
#define WF_TH2 4
#define WF_Q2 8
#define WF_HFX 16
#define WF_QFX 32
#define WF_LH 64
#define WF_TKE 128
#define WF_EARTH 256
#define WF_ROT 512
#define WF_GUST_B 1024

extern "C" __global__ void woof_post_e_fields_v1(
    const float* __restrict__ theta_v,  // [nz][cell]
    const float* __restrict__ tk,
    const float* __restrict__ pcoord,
    const float* __restrict__ z_mass,
    const float* __restrict__ u_mass,
    const float* __restrict__ v_mass,
    const float* __restrict__ tke,      // [nz][cell] or null
    const float* __restrict__ hgt,      // [cell]
    const float* __restrict__ psfc,
    const float* __restrict__ u10,
    const float* __restrict__ v10,
    const float* __restrict__ ust,      // null when absent
    const float* __restrict__ t2,
    const float* __restrict__ th2,
    const float* __restrict__ q2,
    const float* __restrict__ hfx,
    const float* __restrict__ qfx,
    const float* __restrict__ lh,
    const float* __restrict__ sinalpha,
    const float* __restrict__ cosalpha,
    int ncell_i, int nz, int flags,
    float* __restrict__ out)            // [10][cell]
{
    u64 cell = (u64)blockIdx.x * (u64)blockDim.x + (u64)threadIdx.x;
    u64 ncell = (u64)ncell_i;
    if (cell >= ncell) return;

    double zsfc = w_ld(hgt, cell);
    double ps = w_ld(psfc, cell);

    // ---------------- pbl_height (bulk Richardson) ----------------
    double pbl = w_nan();
    {
        double thv_s = w_ld(theta_v, cell);
        double u_s = w_ld(u_mass, cell);
        double v_s = w_ld(v_mass, cell);
        double z_s = w_ld(z_mass, cell);
        double us2 = 0.0;
        bool ok = !(w_isnan(thv_s) || w_isnan(u_s) || w_isnan(v_s) || w_isnan(z_s) || w_isnan(zsfc));
        if (flags & WF_UST) {
            double u_star = w_ld(ust, cell);
            if (w_isnan(u_star)) ok = false;
            us2 = W_B_UST * u_star * u_star;
        } else {
            ok = false;
        }
        if (ok) {
            double ri_prev = 0.0;
            double z_prev = z_s;
            for (int k = 1; k < nz; ++k) {
                u64 m = (u64)k * ncell + cell;
                double thv = w_ld(theta_v, m);
                double zk = w_ld(z_mass, m);
                double du = w_ld(u_mass, m) - u_s;
                double dv = w_ld(v_mass, m) - v_s;
                if (w_isnan(thv) || w_isnan(zk) || w_isnan(du) || w_isnan(dv)) break;
                double den = du * du + dv * dv + us2;
                if (den < W_DEN_FLOOR) den = W_DEN_FLOOR;
                double ri = (W_G / thv_s) * (thv - thv_s) * (zk - z_s) / den;
                if (ri >= W_RI_CRIT) {
                    double f = (W_RI_CRIT - ri_prev) / (ri - ri_prev);
                    double h = z_prev + f * (zk - z_prev);
                    pbl = h - zsfc;
                    break;
                }
                ri_prev = ri;
                z_prev = zk;
            }
        }
    }

    // ---------------- u80 / v80 ----------------
    double u80 = w_nan();
    double v80 = w_nan();
    {
        double zt = zsfc + 80.0;
        double z0 = w_ld(z_mass, cell);
        if (!w_isnan(zt) && !w_isnan(z0)) {
            if (z0 > zt) {
                double a = w_ld(u10, cell);
                double b = w_ld(v10, cell);
                double zb = zsfc + 10.0;
                if (!w_isnan(a) && !w_isnan(b) && z0 > zb) {
                    double f = (zt - zb) / (z0 - zb);
                    u80 = a + f * (w_ld(u_mass, cell) - a);
                    v80 = b + f * (w_ld(v_mass, cell) - b);
                }
            } else {
                for (int k = 0; k + 1 < nz; ++k) {
                    u64 m0 = (u64)k * ncell + cell;
                    u64 m1 = m0 + ncell;
                    double za = w_ld(z_mass, m0);
                    double zb = w_ld(z_mass, m1);
                    if (w_isnan(za) || w_isnan(zb)) break;
                    if (zb >= zt) {
                        double f = (zt - za) / (zb - za);
                        double ua = w_ld(u_mass, m0);
                        double va = w_ld(v_mass, m0);
                        u80 = ua + f * (w_ld(u_mass, m1) - ua);
                        v80 = va + f * (w_ld(v_mass, m1) - va);
                        break;
                    }
                }
            }
        }
        if ((flags & WF_EARTH) && (flags & WF_ROT)) {
            double sa = w_ld(sinalpha, cell);
            double ca = w_ld(cosalpha, cell);
            double ue = u80 * ca - v80 * sa;
            double ve = v80 * ca + u80 * sa;
            u80 = ue;
            v80 = ve;
        }
    }

    // ---------------- shelter point ----------------
    double t_sh = w_nan();
    double th_sh = w_nan();
    double r_sh = w_nan();
    if (flags & WF_Q2) r_sh = w_ld(q2, cell);
    if (!w_isnan(r_sh) && r_sh < 0.0) r_sh = 0.0;
    {
        double ex = wm_exp(W_KAPPA * wm_ln(ps / W_P0));
        if (flags & WF_T2) t_sh = w_ld(t2, cell);
        if (flags & WF_TH2) th_sh = w_ld(th2, cell);
        if (w_isnan(t_sh) && !w_isnan(th_sh)) t_sh = th_sh * ex;
        if (w_isnan(th_sh) && !w_isnan(t_sh)) th_sh = t_sh / ex;
    }
    bool has_sh = !w_isnan(t_sh) && !w_isnan(ps) && !w_isnan(zsfc);
    double z_sh = zsfc + 2.0;
    double p_sh = w_nan();
    if (has_sh) {
        double tv = w_isnan(r_sh) ? t_sh : t_sh * (1.0 + r_sh / W_EPS) / (1.0 + r_sh);
        p_sh = ps * wm_exp(-(W_G * 2.0) / (W_RD * tv));
    }
    int npt = nz + (has_sh ? 1 : 0);

    // ---------------- tropopause (WMO lapse-rate) ----------------
    int ktrop = nz - 1;
    for (int k = 0; k + 1 < nz; ++k) {
        u64 m = (u64)k * ncell + cell;
        double pk = w_ld(pcoord, m);
        if (!(pk <= W_TROP_P_MAX && pk >= W_TROP_P_MIN)) continue;
        double zk = w_ld(z_mass, m);
        double tkk = w_ld(tk, m);
        double gam = (tkk - w_ld(tk, m + ncell)) / (w_ld(z_mass, m + ncell) - zk);
        if (!(gam <= W_TROP_LAPSE)) continue;
        bool ok = true;
        for (int jj = k + 1; jj < nz; ++jj) {
            u64 mj = (u64)jj * ncell + cell;
            double dz = w_ld(z_mass, mj) - zk;
            if (dz > W_TROP_DEPTH) break;
            double avg = (tkk - w_ld(tk, mj)) / dz;
            if (!(avg <= W_TROP_LAPSE)) { ok = false; break; }
        }
        if (ok) { ktrop = k; break; }
    }
    int ntop = ktrop + (has_sh ? 1 : 0);  // highest profile point allowed

    // ---------------- isotherms ----------------
    double out_fz = w_nan(), out_fp = w_nan();
    double out_hz = w_nan(), out_hp = w_nan();
    double out_m10 = w_nan(), out_m20 = w_nan();
    bool col_ok = !w_isnan(zsfc) && !w_isnan(ps);
    for (int n = 0; n < npt && col_ok; ++n) {
        double zz, tt, pp;
        W_PT(n, zz, tt, pp);
        if (w_isnan(zz) || w_isnan(tt) || w_isnan(pp)) col_ok = false;
    }
    if (col_ok) {
        double tground;
        { double zz, pp; W_PT(0, zz, tground, pp); }
        // freezing level: lowest warm-to-cold crossing from the ground up
        const double tf = 273.15;
        if (tground <= tf) {
            out_fz = zsfc;
            out_fp = ps;
        } else {
            for (int n = 0; n + 1 < npt; ++n) {
                double za, ta, pa, zb, tb, pb;
                W_PT(n, za, ta, pa);
                W_PT(n + 1, zb, tb, pb);
                if (ta > tf && tb <= tf) {
                    WCross c = w_cross(za, ta, pa, zb, tb, pb, tf);
                    out_fz = c.z;
                    out_fp = c.p;
                    break;
                }
            }
        }
        // highest crossings below the tropopause, for 0, -10 and -20 C
        for (int which = 0; which < 3; ++which) {
            double tc = which == 0 ? 273.15 : (which == 1 ? 263.15 : 253.15);
            double hz = w_nan(), hp = w_nan();
            bool found = false;
            for (int n = ntop - 1; n >= 0; --n) {
                double za, ta, pa, zb, tb, pb;
                W_PT(n, za, ta, pa);
                W_PT(n + 1, zb, tb, pb);
                if (ta > tc && tb <= tc) {
                    WCross c = w_cross(za, ta, pa, zb, tb, pb, tc);
                    hz = c.z;
                    hp = c.p;
                    found = true;
                    break;
                }
            }
            if (!found && tground <= tc) {
                hz = zsfc;
                hp = ps;
            }
            if (which == 0) { out_hz = hz; out_hp = hp; }
            else if (which == 1) { out_m10 = hz; }
            else { out_m20 = hz; }
        }
    }

    // ---------------- gust ----------------
    double gust = w_nan();
    {
        double a = w_ld(u10, cell);
        double b = w_ld(v10, cell);
        double spd = wm_sqrt(a * a + b * b);
        if (flags & WF_GUST_B) {
            // Brasseur (2001): the fastest wind in the boundary layer whose
            // parcel the mean turbulent kinetic energy below it can bring
            // down against the buoyancy it meets on the way.
            if ((flags & WF_TKE) && !w_isnan(spd) && !w_isnan(pbl) && !w_isnan(th_sh)) {
                double thv_g = w_isnan(r_sh) ? th_sh : th_sh * (1.0 + r_sh / W_EPS) / (1.0 + r_sh);
                gust = spd;
                double e0 = w_ld(tke, cell);
                for (int kp = 0; kp < nz; ++kp) {
                    u64 mp = (u64)kp * ncell + cell;
                    double zp = w_ld(z_mass, mp) - zsfc;
                    if (zp > pbl) break;
                    double thp = w_ld(theta_v, mp);
                    // trapezoid sums from the ground (z = 0) to zp
                    double e_int = 0.0;
                    double b_int = 0.0;
                    double z_lo = 0.0;
                    double e_lo = e0;
                    double f_lo = W_G * (thp - thv_g) / thv_g;
                    for (int k = 0; k <= kp; ++k) {
                        u64 mk = (u64)k * ncell + cell;
                        double z_hi = w_ld(z_mass, mk) - zsfc;
                        double e_hi = w_ld(tke, mk);
                        double thk = w_ld(theta_v, mk);
                        double f_hi = W_G * (thp - thk) / thk;
                        double dz = z_hi - z_lo;
                        e_int = e_int + 0.5 * (e_lo + e_hi) * dz;
                        b_int = b_int + 0.5 * (f_lo + f_hi) * dz;
                        z_lo = z_hi;
                        e_lo = e_hi;
                        f_lo = f_hi;
                    }
                    if (zp > 0.0 && e_int / zp >= b_int) {
                        double uk = w_ld(u_mass, mp);
                        double vk = w_ld(v_mass, mp);
                        double sk = wm_sqrt(uk * uk + vk * vk);
                        if (sk > gust) gust = sk;
                    }
                }
            }
        } else {
            // IFS similarity gust: 10 m wind plus C u* f(zi/L), with the
            // convective factor written as (u*^3 + zi kappa g w'thv' / (24 thv))^(1/3).
            bool ok = (flags & WF_UST) && (flags & WF_HFX) && ((flags & WF_QFX) || (flags & WF_LH));
            if (ok && !w_isnan(spd) && !w_isnan(t_sh) && !w_isnan(th_sh) && !w_isnan(r_sh) && !w_isnan(ps)) {
                double u_star = w_ld(ust, cell);
                double h = w_ld(hfx, cell);
                double wq_src = (flags & WF_QFX) ? w_ld(qfx, cell) : w_ld(lh, cell);
                if (!w_isnan(u_star) && !w_isnan(h) && !w_isnan(wq_src)) {
                    if (u_star < 0.0) u_star = 0.0;
                    double vf = (1.0 + r_sh / W_EPS) / (1.0 + r_sh);
                    double tv = t_sh * vf;
                    double thv = th_sh * vf;
                    double rho = ps / (W_RD * tv);
                    double wth = h / (rho * W_CP);
                    double wq = (flags & WF_QFX) ? wq_src / rho : wq_src / (rho * W_LV);
                    double qs = r_sh / (1.0 + r_sh);
                    double c61 = 1.0 / W_EPS - 1.0;
                    double wthv = wth * (1.0 + c61 * qs) + c61 * th_sh * wq;
                    double sig = u_star;
                    if (wthv > 0.0 && !w_isnan(pbl) && pbl > 0.0) {
                        sig = wm_cbrt(u_star * u_star * u_star + pbl * W_KARMAN * W_G * wthv / (24.0 * thv));
                    }
                    gust = spd + W_C_UGN * sig;
                }
            }
        }
    }

    out[0 * ncell + cell] = w_out(pbl);
    out[1 * ncell + cell] = w_out(gust);
    out[2 * ncell + cell] = w_out(u80);
    out[3 * ncell + cell] = w_out(v80);
    out[4 * ncell + cell] = w_out(out_fz);
    out[5 * ncell + cell] = w_out(out_fp);
    out[6 * ncell + cell] = w_out(out_hz);
    out[7 * ncell + cell] = w_out(out_hp);
    out[8 * ncell + cell] = w_out(out_m10);
    out[9 * ncell + cell] = w_out(out_m20);
}
