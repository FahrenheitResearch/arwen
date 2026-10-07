// SPDX-License-Identifier: Apache-2.0
//
// WOOF post-processor, group D: reflectivity, cloud and visibility
// (specification section 7).  Statement-for-statement twin of
// src/group_d/column.rs; included by woof_post.cu inside namespace woof_d
// after post_group_a.cu (ST_* and IN3_* slot numbers) and woof_math.cuh
// (the one maths library, D30).
//
// One thread per column, a fixed loop over levels, binary64 arithmetic from
// binary32 carriers, one rounding per output.  No atomics, no libdevice
// transcendental: pow, exp and ln are wm_pow, wm_exp and wm_ln.
//
// Public sources (spec section 12): WRF ARW technical note [R1] (state,
// constants); Smith, J. Climate Appl. Meteor. 23 (1984) 1258 [R29] (ice
// dielectric ratio 0.224); RIP simulated-reflectivity note [R30]
// (exponential distributions, fixed intercepts); ISCCP, Bull. Amer. Meteor.
// Soc. 80 (1999) 2261 [R34] (680 and 440 hPa); maximum overlap [R33]; ECMWF
// parameter database [R35] (cloud base thresholds, ceiling 50 %); FMH-1
// [R36]; Stoelinga and Warner, J. Appl. Meteor. 38 (1999) 385 [R37] and
// Kunkel, J. Climate Appl. Meteor. 23 (1984) 34 [R38] (extinction); FRAM,
// Bull. Amer. Meteor. Soc. 90 (2009) 341 [R40] (RH haze); WMO-No. 8 [R8]
// (MOR); Alduchov and Eskridge, J. Appl. Meteor. 35 (1996) 601 [R6]
// (saturation); Xu and Randall, J. Atmos. Sci. 53 (1996) 3084 (diagnosed
// cloud fraction, used only when the history carries no CLDFRA).

// Flag bits, shared with src/group_d/column.rs (DF_*).
static constexpr int D_REFL = 1;        // REFL_10CM carrier given
static constexpr int D_QR = 2;          // QRAIN present (species reflectivity)
static constexpr int D_QS = 4;          // QSNOW present
static constexpr int D_QG = 8;          // QGRAUP present
static constexpr int D_QC = 16;         // QCLOUD present
static constexpr int D_QI = 32;         // QICE present
static constexpr int D_CF_MODEL = 64;   // cloud fraction from the CLDFRA carrier
static constexpr int D_CF_DIAG = 128;   // cloud fraction diagnosed (no CLDFRA)
static constexpr int D_CF_BINARY = 256; // diagnosed method: condensate yes/no instead of Xu-Randall
static constexpr int D_AER = 512;       // aerosol extinction plane given (km-1)
static constexpr int D_HGT = 1024;      // HGT plane given
static constexpr int D_INTERP_DBZ = 2048; // reflectivity_1km interpolated in dBZ (alternative to linear Z)
static constexpr int D_OLR = 4096;      // OLR plane given (simulated_ir)

static constexpr int D_NOUT = 9;
static constexpr int D_COMP = 0, D_R1KM = 1, D_LOW = 2, D_MID = 3, D_HIGH = 4,
                     D_BASE = 5, D_TOP = 6, D_CEIL = 7, D_VIS = 8;

static constexpr double D_RD = 287.0;
static constexpr double D_RV = 461.6;
static constexpr double D_EPS = 287.0 / 461.6;
static constexpr double D_PI = 3.141592653589793;
static constexpr double D_LN10 = 2.302585092994046;
// Reflectivity (spec 7.1, D17).
static constexpr double D_DBZ_FLOOR = -20.0;
static constexpr double D_Z_TINY = 1.0e-10;
static constexpr double D_REFL_AGL = 1000.0;
static constexpr double D_ICE_RATIO = 0.224;
static constexpr double D_RHO_W = 1000.0;
static constexpr double D_N0R = 8.0e6, D_RHOR = 1000.0;
static constexpr double D_N0S = 2.0e7, D_RHOS = 100.0;
static constexpr double D_N0G = 4.0e6, D_RHOG = 400.0;
// Cloud layers (spec 7.2, D18) and geometry (7.3, D19, D20).
static constexpr double D_P_LOW_TOP = 68000.0;
static constexpr double D_P_MID_TOP = 44000.0;
static constexpr double D_CF_MIN = 0.01;
static constexpr double D_COND_MIN = 1.0e-6;
static constexpr double D_CEIL_CF = 0.5;
// Diagnosed cloud fraction, Xu and Randall (1996): p, alpha0, gamma.
static constexpr double D_XR_P = 0.25;
static constexpr double D_XR_ALPHA0 = 100.0;
static constexpr double D_XR_GAMMA = 0.49;
// Visibility (spec 7.4, D21).
static constexpr double D_VIS_CAP = 20000.0;
static constexpr double D_EXT_C_A = 144.7, D_EXT_C_B = 0.88;
static constexpr double D_EXT_R_A = 2.24, D_EXT_R_B = 0.75;
static constexpr double D_EXT_I_A = 327.8, D_EXT_I_B = 1.0;
static constexpr double D_EXT_S_A = 10.36, D_EXT_S_B = 0.7776;
static constexpr double D_EXT_G_A = 2.24, D_EXT_G_B = 0.75;
static constexpr double D_HAZE_A = 40.10, D_HAZE_B = 5.19e-10, D_HAZE_C = 5.44;
static constexpr double D_HAZE_RH_MIN = 30.0;
static constexpr double D_MAG_A = 610.94, D_MAG_B = 17.625, D_MAG_C = 243.04;
static constexpr double D_MAG_ICE_A = 611.21, D_MAG_ICE_B = 22.587, D_MAG_ICE_C = 273.86;
// Simulated infrared brightness temperature (twin of the IR constants in
// src/group_d/column.rs): Stefan-Boltzmann constant (CODATA 2018), the
// Ohring, Gruber and Ellingson (1984) relation with the Yang and Slingo
// (2001) constants, the PROVISIONAL window absorption per gram of
// condensate (m2 g-1) and the emission optical depth.
static constexpr double D_SIGMA_SB = 5.670374419e-8;
static constexpr double D_OGE_A = 1.228, D_OGE_B = -1.106e-3;
static constexpr double D_IR_K_LIQUID = 0.145, D_IR_K_ICE = 0.272;
static constexpr double D_IR_TAU_EMIT = 1.0;
static constexpr double D_G_IR = 9.81;

struct DCol {
    const float* st;   // Group A mass state, [ST_COUNT][nz][ncell]
    const float* in3;  // packed carriers, [IN3_COUNT][nz][ncell]
    const float* refl; // [nz][ncell] or null
    const float* cf;   // [nz][ncell] or null
    long long vol;
    long long n;
    long long i;
    int flags;
};

__device__ __forceinline__ double d_st(const DCol& c, int slot, int k) {
    return (double)c.st[(long long)slot * c.vol + (long long)k * c.n + c.i];
}

__device__ __forceinline__ double d_in(const DCol& c, int slot, int bit, int k) {
    if (!(c.flags & bit)) return 0.0;
    return (double)c.in3[(long long)slot * c.vol + (long long)k * c.n + c.i];
}

__device__ __forceinline__ double d_max(double a, double b) { return a > b ? a : b; }
__device__ __forceinline__ double d_min(double a, double b) { return a < b ? a : b; }
__device__ __forceinline__ double d_clamp01(double w) { return w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w); }
__device__ __forceinline__ bool d_nan(double x) { return x != x; }

__device__ __forceinline__ double d_rho_air(double p, double tk, double r) {
    double tv = tk * (1.0 + r / D_EPS) / (1.0 + r);
    return p / (D_RD * tv);
}

__device__ __forceinline__ double d_log10(double x) { return wm_ln(x) / D_LN10; }

__device__ __forceinline__ double d_z_species(double rho_q, double n0, double rho_x) {
    if (!(rho_q > 0.0)) return 0.0;
    double lam = wm_pow(D_PI * rho_x * n0 / rho_q, 0.25);
    return 720.0 * n0 * wm_pow(lam, -7.0) * 1.0e18;
}

// dBZ at level k: the REFL_10CM carrier, else the RIP-note volume rounded
// once to binary32 (the precision a stored volume has).
__device__ double d_dbz(const DCol& c, int k) {
    if (c.flags & D_REFL) return (double)c.refl[(long long)k * c.n + c.i];
    double p = d_st(c, ST_PFULL, k);
    double tk = d_st(c, ST_TK, k);
    double r = d_st(c, ST_R, k);
    double qr = d_in(c, IN3_QR, D_QR, k);
    double qs = d_in(c, IN3_QS, D_QS, k);
    double qg = d_in(c, IN3_QG, D_QG, k);
    if (d_nan(p) || d_nan(tk) || d_nan(r) || d_nan(qr) || d_nan(qs) || d_nan(qg)) return wm_nan();
    double rho = d_rho_air(p, tk, r);
    double fs = D_ICE_RATIO * ((D_RHOS / D_RHO_W) * (D_RHOS / D_RHO_W));
    double fg = D_ICE_RATIO * ((D_RHOG / D_RHO_W) * (D_RHOG / D_RHO_W));
    double z = d_z_species(rho * qr, D_N0R, D_RHOR);
    z = z + fs * d_z_species(rho * qs, D_N0S, D_RHOS);
    z = z + fg * d_z_species(rho * qg, D_N0G, D_RHOG);
    z = d_max(z, D_Z_TINY);
    return (double)__double2float_rn(10.0 * d_log10(z));
}

__device__ __forceinline__ double d_condensate(const DCol& c, int k) {
    double cond = d_in(c, IN3_QC, D_QC, k) + d_in(c, IN3_QI, D_QI, k);
    if (c.flags & D_QS) cond = cond + d_in(c, IN3_QS, D_QS, k);
    return cond;
}

// Saturation vapour pressure (Pa): water at or above 0 C, ice below [R6].
__device__ __forceinline__ double d_esat_phase(double tk) {
    double tc = tk - 273.15;
    if (tc >= 0.0) return D_MAG_A * wm_exp(D_MAG_B * tc / (tc + D_MAG_C));
    return D_MAG_ICE_A * wm_exp(D_MAG_ICE_B * tc / (tc + D_MAG_ICE_C));
}

// Cloud fraction (0..1) at level k.
__device__ double d_cloud_fraction(const DCol& c, int k) {
    if (c.flags & D_CF_MODEL) return (double)c.cf[(long long)k * c.n + c.i];
    double qc = d_in(c, IN3_QC, D_QC, k);
    double qi = d_in(c, IN3_QI, D_QI, k);
    if (c.flags & D_CF_BINARY) {
        double cond = d_condensate(c, k);
        if (d_nan(cond)) return wm_nan();
        return cond > D_COND_MIN ? 1.0 : 0.0;
    }
    double p = d_st(c, ST_PFULL, k);
    double tk = d_st(c, ST_TK, k);
    double r = d_st(c, ST_R, k);
    if (d_nan(p) || d_nan(tk) || d_nan(r) || d_nan(qc) || d_nan(qi)) return wm_nan();
    double l = d_max(qc, 0.0) + d_max(qi, 0.0);
    double es = d_esat_phase(tk);
    if (!(p > es)) return l > 0.0 ? 1.0 : 0.0;
    double rs = D_EPS * es / (p - es);
    double rh = r / rs;
    if (rh >= 1.0) return 1.0;
    if (!(rh > 0.0) || !(l > 0.0)) return 0.0;
    double arg = D_XR_ALPHA0 * l / wm_pow((1.0 - rh) * rs, D_XR_GAMMA);
    double f = wm_pow(rh, D_XR_P) * (1.0 - wm_exp(-arg));
    return d_clamp01(f);
}

__device__ __forceinline__ double d_cross(double cf, double cond, double fthr) {
    return d_min(cf / fthr, cond / D_COND_MIN);
}

// All nine group D planes of one column.  `avail` bits say which planes the
// carriers support; the host blanks the others (omissions) identically.
__device__ void d_column(const DCol& c, int nz, const float* aer, const float* hgt, double contrast, double* o) {
    for (int f = 0; f < D_NOUT; ++f) o[f] = wm_nan();
    bool has_refl = (c.flags & D_REFL) || (c.flags & D_QR);
    bool has_cf = (c.flags & D_CF_MODEL) || (c.flags & D_CF_DIAG);

    if (has_refl) {
        double m = -1.0e300;
        bool bad = false;
        for (int k = 0; k < nz; ++k) {
            double d = d_dbz(c, k);
            if (d_nan(d)) bad = true;
            else if (d > m) m = d;
        }
        if (!bad) o[D_COMP] = d_max(m, D_DBZ_FLOOR);
        if (c.flags & D_HGT) {
            double h = (double)hgt[c.i];
            bool bad1 = d_nan(h);
            int kf = -1;
            for (int k = 0; k < nz; ++k) {
                double z = d_st(c, ST_ZMASS, k);
                if (d_nan(z) || d_nan(d_dbz(c, k))) bad1 = true;
                if (kf < 0 && (z - h) >= D_REFL_AGL) kf = k;
            }
            if (!bad1 && kf >= 0) {
                int lo = kf > 0 ? kf - 1 : 0;
                double z_hi = d_st(c, ST_ZMASS, kf) - h;
                double z_lo = d_st(c, ST_ZMASS, lo) - h;
                double w = kf > 0 ? (D_REFL_AGL - z_lo) / (z_hi - z_lo) : 1.0;
                double d_hi = d_dbz(c, kf);
                double d_lo = d_dbz(c, lo);
                double d;
                if (c.flags & D_INTERP_DBZ) {
                    d = d_lo + w * (d_hi - d_lo);
                } else {
                    double zh = wm_pow(10.0, d_hi / 10.0);
                    double zl = wm_pow(10.0, d_lo / 10.0);
                    double zz = zl + w * (zh - zl);
                    d = 10.0 * d_log10(d_max(zz, D_Z_TINY));
                }
                o[D_R1KM] = d_max(d, D_DBZ_FLOOR);
            }
        }
    }

    if (has_cf) {
        double l = 0.0, m = 0.0, h = 0.0;
        bool bad = false;
        int kb = -1, kt = -1, kc = -1;
        for (int k = 0; k < nz; ++k) {
            double p = d_st(c, ST_PFULL, k);
            double cf = d_cloud_fraction(c, k);
            double cond = d_condensate(c, k);
            double z = d_st(c, ST_ZMASS, k);
            if (d_nan(p) || d_nan(cf) || d_nan(cond) || d_nan(z)) { bad = true; continue; }
            if (p > D_P_LOW_TOP) l = d_max(l, cf);
            else if (p > D_P_MID_TOP) m = d_max(m, cf);
            else h = d_max(h, cf);
            bool cloudy = (cf > D_CF_MIN) && (cond > D_COND_MIN);
            bool ceil_cloudy = (cf >= D_CEIL_CF) && (cond > D_COND_MIN);
            if (cloudy) { if (kb < 0) kb = k; kt = k; }
            if (ceil_cloudy && kc < 0) kc = k;
        }
        if (!bad) {
            o[D_LOW] = d_min(d_max(100.0 * l, 0.0), 100.0);
            o[D_MID] = d_min(d_max(100.0 * m, 0.0), 100.0);
            o[D_HIGH] = d_min(d_max(100.0 * h, 0.0), 100.0);
            // Lowest crossings: base (D_CF_MIN) and ceiling (D_CEIL_CF).
            for (int pass = 0; pass < 2; ++pass) {
                int k = pass == 0 ? kb : kc;
                double fthr = pass == 0 ? D_CF_MIN : D_CEIL_CF;
                int dst = pass == 0 ? D_BASE : D_CEIL;
                if (k < 0) continue;
                double z_k = d_st(c, ST_ZMASS, k);
                if (k == 0) { o[dst] = z_k; continue; }
                double z_lo = d_st(c, ST_ZMASS, k - 1);
                double s_k = d_cross(d_cloud_fraction(c, k), d_condensate(c, k), fthr);
                double s_lo = d_cross(d_cloud_fraction(c, k - 1), d_condensate(c, k - 1), fthr);
                double den = s_k - s_lo;
                double w = den > 0.0 ? (1.0 - s_lo) / den : 1.0;
                w = d_clamp01(w);
                o[dst] = z_lo + w * (z_k - z_lo);
            }
            // Highest crossing: top.
            if (kt >= 0) {
                double z_k = d_st(c, ST_ZMASS, kt);
                if (kt == nz - 1) {
                    o[D_TOP] = z_k;
                } else {
                    double z_hi = d_st(c, ST_ZMASS, kt + 1);
                    double s_k = d_cross(d_cloud_fraction(c, kt), d_condensate(c, kt), D_CF_MIN);
                    double s_hi = d_cross(d_cloud_fraction(c, kt + 1), d_condensate(c, kt + 1), D_CF_MIN);
                    double den = s_k - s_hi;
                    double w = den > 0.0 ? (s_k - 1.0) / den : 0.0;
                    w = d_clamp01(w);
                    o[D_TOP] = z_k + w * (z_hi - z_k);
                }
            }
        }
    }

    // Visibility at the lowest mass level.
    {
        double p = d_st(c, ST_PFULL, 0);
        double tk = d_st(c, ST_TK, 0);
        double r = d_st(c, ST_R, 0);
        double qc = d_in(c, IN3_QC, D_QC, 0);
        double qr = d_in(c, IN3_QR, D_QR, 0);
        double qi = d_in(c, IN3_QI, D_QI, 0);
        double qs = d_in(c, IN3_QS, D_QS, 0);
        double qg = d_in(c, IN3_QG, D_QG, 0);
        double a = (c.flags & D_AER) ? (double)aer[c.i] : 0.0;
        bool bad = d_nan(p) || d_nan(tk) || d_nan(r) || d_nan(qc) || d_nan(qr) || d_nan(qi)
                   || d_nan(qs) || d_nan(qg) || d_nan(a);
        if (!bad) {
            double rho = d_rho_air(p, tk, r);
            double beta = D_EXT_C_A * wm_pow(1000.0 * rho * d_max(qc, 0.0), D_EXT_C_B);
            beta = beta + D_EXT_R_A * wm_pow(1000.0 * rho * d_max(qr, 0.0), D_EXT_R_B);
            beta = beta + D_EXT_I_A * wm_pow(1000.0 * rho * d_max(qi, 0.0), D_EXT_I_B);
            beta = beta + D_EXT_S_A * wm_pow(1000.0 * rho * d_max(qs, 0.0), D_EXT_S_B);
            beta = beta + D_EXT_G_A * wm_pow(1000.0 * rho * d_max(qg, 0.0), D_EXT_G_B);
            double q = r / (1.0 + r);
            double e = q * p / (D_EPS + (1.0 - D_EPS) * q);
            double tc = tk - 273.15;
            double es = D_MAG_A * wm_exp(D_MAG_B * tc / (tc + D_MAG_C));
            double rh = d_min(d_max(100.0 * e / es, 0.0), 100.0);
            double rh_h = d_max(rh, D_HAZE_RH_MIN);
            double vis_haze_km = D_HAZE_A - D_HAZE_B * wm_pow(rh_h, D_HAZE_C);
            beta = beta + contrast / vis_haze_km;
            beta = beta + a;
            o[D_VIS] = d_min(1000.0 * contrast / beta, D_VIS_CAP);
        }
    }
}

extern "C" __global__ void woof_post_d_fields_v1(
        const float* __restrict__ st, const float* __restrict__ in3,
        const float* __restrict__ refl, const float* __restrict__ cf,
        const float* __restrict__ aer, const float* __restrict__ hgt,
        int ncell, int nz, int flags, double contrast, float* __restrict__ out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncell) return;
    DCol c;
    c.st = st;
    c.in3 = in3;
    c.refl = refl;
    c.cf = cf;
    c.n = ncell;
    c.vol = (long long)nz * ncell;
    c.i = i;
    c.flags = flags;
    double o[D_NOUT];
    d_column(c, nz, aer, hgt, contrast, o);
    for (int f = 0; f < D_NOUT; ++f) out[(long long)f * ncell + i] = wm_to_f32(o[f]);
}

// Window brightness temperature from the model's top-of-atmosphere OLR
// (W m-2): T_f = (OLR / sigma)^(1/4), root of T_f = T_b (a + b T_b)
// (Ohring, Gruber and Ellingson, J. Climate Appl. Meteor. 23 (1984) 416;
// constants of Yang and Slingo, Mon. Wea. Rev. 129 (2001) 784).
__device__ __forceinline__ double d_olr_tb(double olr) {
    if (!(olr > 0.0)) return wm_nan();
    double tf = wm_sqrt(wm_sqrt(olr / D_SIGMA_SB));
    double disc = D_OGE_A * D_OGE_A + 4.0 * D_OGE_B * tf;
    if (!(disc >= 0.0)) return wm_nan();
    return (wm_sqrt(disc) - D_OGE_A) / (2.0 * D_OGE_B);
}

// simulated_ir of one column: temperature at unit window absorption optical
// depth from the model top where the column is opaque (Eddington-Barbier),
// else the OLR relation.  Statement-for-statement twin of ir_column in
// src/group_d/column.rs.
__device__ double d_ir_column(const DCol& c, const float* si, const float* olr, int nz) {
    if (!(c.flags & D_OLR)) return wm_nan();
    if ((c.flags & D_QC) && (c.flags & D_QI)) {
        long long ivol = (long long)(nz + 1) * c.n;
        double cum = 0.0;
        for (int k = nz - 1; k >= 0; --k) {
            double qc = d_in(c, IN3_QC, D_QC, k);
            double qi = d_in(c, IN3_QI, D_QI, k);
            double qs = d_in(c, IN3_QS, D_QS, k);
            double dp = (double)si[SI_PINT * ivol + (long long)k * c.n + c.i]
                      - (double)si[SI_PINT * ivol + (long long)(k + 1) * c.n + c.i];
            double tk = d_st(c, ST_TK, k);
            if (d_nan(qc) || d_nan(qi) || d_nan(qs) || d_nan(dp) || d_nan(tk)) return wm_nan();
            double cond = D_IR_K_LIQUID * d_max(qc, 0.0) + D_IR_K_ICE * (d_max(qi, 0.0) + d_max(qs, 0.0));
            double dtau = cond * 1000.0 * d_max(dp, 0.0) / D_G_IR;
            if (cum + dtau >= D_IR_TAU_EMIT) {
                double w = (D_IR_TAU_EMIT - cum) / dtau;
                double t_above = (k + 1 < nz) ? d_st(c, ST_TK, k + 1) : tk;
                if (d_nan(t_above)) return wm_nan();
                return t_above + w * (tk - t_above);
            }
            cum = cum + dtau;
        }
    }
    double o = (double)olr[c.i];
    if (d_nan(o)) return wm_nan();
    return d_olr_tb(o);
}

extern "C" __global__ void woof_post_d_ir_v1(
        const float* __restrict__ st, const float* __restrict__ si,
        const float* __restrict__ in3, const float* __restrict__ olr,
        int ncell, int nz, int flags, float* __restrict__ out) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncell) return;
    DCol c;
    c.st = st;
    c.in3 = in3;
    c.refl = nullptr;
    c.cf = nullptr;
    c.n = ncell;
    c.vol = (long long)nz * ncell;
    c.i = i;
    c.flags = flags;
    out[i] = wm_to_f32(d_ir_column(c, si, olr, nz));
}
