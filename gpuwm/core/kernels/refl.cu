// gpuwm/core/kernels/refl.cu
//
// Radar reflectivity diagnostic (WRF do_radar_ref=1, REFL_10CM).
// Transcription authority (local WRF v4.6.1):
//   phys/module_mp_morr_two_moment.F: refl10cm_hm (:4502-4675), the
//     wrapper floor MAX(-35., dBZ) (:913-917), and the m(D) parameters
//     the scheme hands radar_init (:532-542; morr_rimed_ice selects CG,
//     with WRF Registry default 1 = hail/RHOG 900).
//   phys/module_mp_radar.F: the Blahak melting-particle backscatter
//     (rayleigh_soak_wetgraupel :265-358, get_m_mix_nested :362-489,
//     get_m_mix :493-541, m_complex_maxwellgarnett :544-590).
//
// One CUDA thread per (j, i) column (the melting-level scan k_0 is a
// column-serial dependency).  radar_init PRODUCTS (size bins, Simpson
// weights, K_w, m_w_0, m_i_0) arrive from gpuwm.core.refl.radar_init,
// mirroring WRF's split: radar_init runs once at scheme init on the
// host, the column routine only consumes its module state.  Like the
// Fortran's DOUBLE PRECISION locals, lambda/N0/backscatter chains are
// binary64 while loaded species densities and the ze arrays stay FP32
// (the Fortran's REAL storage); the mirror
// gpuwm.verify.npref.np_refl10cm_morrison_column is float64 throughout.
//
// The Kessler fallback kernel is per-cell (rain-only, no column scan):
// Smith et al. (1975) exponential-PSD rain reflectivity with the fixed
// Marshall-Palmer intercept N0r = 8e6 m-4 and rho_w = 1000 kg m-3 that
// Kessler's own fall-speed closure assumes (WRF's Kessler carries no
// refl_10cm diagnostic; derivation in gpuwm/core/refl.py).  Float64
// mirror: np_refl10cm_kessler_column.

// Compile-time bound on the three column kernels' per-thread arrays.  The
// launchers (gpuwm/core/refl.py) specialize it to the field's own nz through
// get_kernel_int_defines; 256 is the unspecialized ceiling this source keeps
// when nothing overrides it, and the value gpuwm/core/refl.py validates nz
// against.  Nothing here reads the bound as a value -- every loop runs to
// the runtime nz -- so specializing it moves no arithmetic.  It costs the
// widest kernel (refl10cm_morrison_column) 72 B of local frame per level,
// which the driver answers with (frame - 1024) * 1536 * 170 bytes of backing
// store at first launch: 4,335 MiB at 256 against 624 MiB at 49.
#ifndef REFL_KMAX
#define REFL_KMAX 256
#endif
#define REFL_NRBINS 50

// Morrison module PI (module_mp_morr_two_moment.F:99), also the radar
// module PIx (module_mp_radar.F:283).
#define RPI 3.1415926535897932384626434
// module_model_constants.F:19 r_d = 287., Morrison's R (F:92).
#define R_D 287.0
// m(D) prefactors set before "call radar_init" (F:532-542): rain
// PI*RHOW/6 (RHOW = 997), snow CS = 100*PI/6.  Dense-ice CG is a
// runtime scalar because morr_rimed_ice selects hail or graupel.
#define RXAM_R (RPI * 997.0 / 6.0)
#define RXAM_S (100.0 * RPI / 6.0)
// radar_init gamma moments for xbm = 3, xmu = 0 (module_mp_radar.F:
// 150-175): xc?g(3) = Gamma(4), xc?g(4) = Gamma(7); the 1/Gamma(1+mu)
// factors xorg2/xosg2/xogg2 are exactly 1.
#define RGAMMA_4 6.0
#define RGAMMA_7 720.0
// PI5 with WRF's truncated pi, and lamda_radar = 0.10 m (:41/:76-77).
#define RPI5 (3.14159 * 3.14159 * 3.14159 * 3.14159 * 3.14159)
#define RLAMDA4 (0.10 * 0.10 * 0.10 * 0.10)
// melt_outside_s == melt_outside_g = 0.9 (:61-62).
#define RMELT_OUTSIDE 0.9

// ---- binary64 complex helpers (COMPLEX*16 arithmetic) --------------------

struct cd { double re, im; };

__device__ __forceinline__ cd cd_make(double re, double im)
{
    cd c; c.re = re; c.im = im; return c;
}

__device__ __forceinline__ cd cd_add(cd a, cd b)
{
    return cd_make(a.re + b.re, a.im + b.im);
}

__device__ __forceinline__ cd cd_sub(cd a, cd b)
{
    return cd_make(a.re - b.re, a.im - b.im);
}

__device__ __forceinline__ cd cd_mul(cd a, cd b)
{
    return cd_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

__device__ __forceinline__ cd cd_div(cd a, cd b)
{
    double d = b.re * b.re + b.im * b.im;
    return cd_make((a.re * b.re + a.im * b.im) / d,
                   (a.im * b.re - a.re * b.im) / d);
}

__device__ __forceinline__ double cd_abs(cd a)
{
    return hypot(a.re, a.im);
}

// Principal square root (Fortran SQRT on COMPLEX*16).
__device__ __forceinline__ cd cd_sqrt(cd a)
{
    double m = cd_abs(a);
    double re = sqrt(fmax(0.5 * (m + a.re), 0.0));
    double im = sqrt(fmax(0.5 * (m - a.re), 0.0));
    return cd_make(re, a.im < 0.0 ? -im : im);
}

// Principal logarithm (Fortran LOG on COMPLEX*16).
__device__ __forceinline__ cd cd_log(cd a)
{
    return cd_make(log(cd_abs(a)), atan2(a.im, a.re));
}

__device__ __forceinline__ cd cd_scale(double s, cd a)
{
    return cd_make(s * a.re, s * a.im);
}

// ---- module_mp_radar.F per-particle machinery -----------------------------

// m_complex_maxwellgarnett (:544-590) on the only inclusion string
// radar_init configures ('spheroidal', :106-118; betas :574-576).  The
// Fortran's volume-closure/-999 error path (:558) is unreachable here:
// every call below passes fractions that sum to 1 by construction.
__device__ cd refl_maxwellgarnett(double vol1, double vol2, double vol3,
                                  cd m1, cd m2, cd m3)
{
    cd m1t = cd_mul(m1, m1);
    cd m2t = cd_mul(m2, m2);
    cd m3t = cd_mul(m3, m3);
    cd d2 = cd_sub(m2t, m1t);
    cd d3 = cd_sub(m3t, m1t);
    cd one = cd_make(1.0, 0.0);
    cd beta2 = cd_mul(cd_div(cd_scale(2.0, m1t), d2),
                      cd_sub(cd_mul(cd_div(m2t, d2),
                                    cd_log(cd_div(m2t, m1t))), one));
    cd beta3 = cd_mul(cd_div(cd_scale(2.0, m1t), d3),
                      cd_sub(cd_mul(cd_div(m3t, d3),
                                    cd_log(cd_div(m3t, m1t))), one));
    cd num = cd_add(cd_scale(1.0 - vol2 - vol3, m1t),
                    cd_add(cd_scale(vol2, cd_mul(beta2, m2t)),
                           cd_scale(vol3, cd_mul(beta3, m3t))));
    cd den = cd_add(cd_make(1.0 - vol2 - vol3, 0.0),
                    cd_add(cd_scale(vol2, beta2), cd_scale(vol3, beta3)));
    return cd_sqrt(cd_div(num, den));
}

// get_m_mix_nested (:362-489) on radar_init's Morrison string set --
// host='air', matrix='water', hostmatrix='icewater' (:106-118) -- which
// resolves to exactly two nested Maxwell-Garnett evaluations along the
// host='air' branch (:384-413): the ice-in-water inclusion mix
// (get_m_mix matrix='water', :517-519), then that mixture as 'ice'
// inclusions in the air host (get_m_mix matrix='ice', :514-516).
__device__ cd refl_m_mix_nested(cd m_a, cd m_i, cd m_w, double volair,
                                double volice, double volwater)
{
    double vol1 = volice / fmax(volice + volwater, 1.0e-10);   // :391
    double vol2 = 1.0 - vol1;                                  // :392
    cd mtmp = refl_maxwellgarnett(vol2, 0.0, vol1, m_w, m_a, m_i);
    return refl_maxwellgarnett(1.0 - volair, volair, 0.0,
                               mtmp, m_a, cd_scale(2.0, m_a));
}

// rayleigh_soak_wetgraupel (:265-358): Rayleigh backscattering cross
// section (m2) of one melting, water-coated ice particle of mass x_g.
__device__ double refl_rayleigh_soak_wetgraupel(
    double x_g, double a_geo, double b_geo, double fmelt,
    double meltratio_outside, cd m_w, cd m_i)
{
    double fm = fmin(fmax(fmelt, 0.0), 1.0);                   // :290
    double mra = fmin(fmax(meltratio_outside, 0.0), 1.0);      // :292
    mra = mra + (1.0 - mra) * fm;                              // :299
    double x_w = x_g * fm;                                     // :301
    double d_g = a_geo * pow(x_g, b_geo);                      // :303
    if (d_g < 1.0e-12) return 0.0;                             // :305/:354-356
    double vg = RPI / 6.0 * d_g * d_g * d_g;                   // :307
    double rhog = fmin(fmax(x_g / vg, 10.0), 900.0);           // :308
    vg = x_g / rhog;                                           // :309
    double grenz = 1.0 - rhog / 1000.0;                        // :311
    double volg;
    if (mra <= grenz) {                                        // :313-317
        volg = vg * (1.0 - mra * fm);
    } else {                                                   // :319-331
        double fmgrenz = (900.0 - rhog)
                       / (mra * 900.0 - rhog + 900.0 * rhog / 1000.0);
        volg = (fm <= fmgrenz) ? (1.0 - mra * fm) * vg
                               : (x_g - x_w) / 900.0 + x_w / 1000.0;
    }
    double d_large = pow(6.0 / RPI * volg, 1.0 / 3.0);         // :335
    double volice = (x_g - x_w) / (volg * 900.0);              // :336
    double volwater = x_w / (1000.0 * volg);                   // :337
    double volair = 1.0 - volice - volwater;                   // :338
    cd m_core = refl_m_mix_nested(cd_make(1.0, 0.0), m_i, m_w,
                                  volair, volice, volwater);   // :342
    cd m2 = cd_mul(m_core, m_core);
    double kfac = cd_abs(cd_div(cd_sub(m2, cd_make(1.0, 0.0)),
                                cd_add(m2, cd_make(2.0, 0.0))));
    double d3 = d_large * d_large * d_large;
    return kfac * kfac * RPI5 * d3 * d3 / RLAMDA4;             // :351-352
}

// ---- Morrison refl10cm_hm column kernel -----------------------------------

extern "C" __global__ void refl10cm_morrison_column(
    const real* __restrict__ qv, const real* __restrict__ qr,
    const real* __restrict__ nr, const real* __restrict__ qs,
    const real* __restrict__ ns, const real* __restrict__ qg,
    const real* __restrict__ ng, const real* __restrict__ t,
    const real* __restrict__ p, const double* __restrict__ tables,
    const double k_w, const double m_w0_re, const double m_w0_im,
    const double m_i0_re, const double m_i0_im,
    const double xam_g,
    real* __restrict__ refl, const int nz, const int ny, const int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    int j = col / nx;
    int i = col - j * nx;

    // radar_init bin tables, packed [xxDs | xdts | xxDg | xdtg | simpson].
    const double* xxds = tables;
    const double* xdts = tables + REFL_NRBINS;
    const double* xxdg = tables + 2 * REFL_NRBINS;
    const double* xdtg = tables + 3 * REFL_NRBINS;
    const double* simpson = tables + 4 * REFL_NRBINS;
    const cd m_w_0 = cd_make(m_w0_re, m_w0_im);
    const cd m_i_0 = cd_make(m_i0_re, m_i0_im);
    // radar_init soak geometry factors (module_mp_radar.F:177-183):
    // xocms = (1/xam_s)**(1/xbm_s), xocmg likewise.
    const double xobm = 1.0 / 3.0;
    const double xocms = pow(1.0 / RXAM_S, xobm);
    const double xocmg = pow(1.0 / xam_g, xobm);

    float rr[REFL_KMAX], rs[REFL_KMAX], rg[REFL_KMAX];
    double ilamr[REFL_KMAX], ilams[REFL_KMAX], ilamg[REFL_KMAX];
    double n0r[REFL_KMAX], n0s[REFL_KMAX], n0g[REFL_KMAX];
    bool lqr[REFL_KMAX], lqs[REFL_KMAX], lqg[REFL_KMAX];
    bool invalid_moment[REFL_KMAX];
    float zer[REFL_KMAX], zes[REFL_KMAX], zeg[REFL_KMAX];

    // Load the column and diagnose PSD slopes/intercepts from the
    // scheme's prognostic moments (:4536-4584).  Species densities keep
    // the Fortran's REAL storage; slope/intercept locals are its
    // DOUBLE PRECISION.
    for (int k = 0; k < nz; ++k) {
        size_t idx = I3(k, j, i, ny, nx);
        double temp = (double)t[idx];
        double qvk = fmax(1.0e-10, (double)qv[idx]);           // :4540
        double rho = 0.622 * (double)p[idx]
                   / (R_D * temp * (qvk + 0.622));             // :4542
        rr[k] = 1.0e-12f; rs[k] = 1.0e-12f; rg[k] = 1.0e-12f;
        lqr[k] = false; lqs[k] = false; lqg[k] = false;
        invalid_moment[k] = false;
        ilamr[k] = 0.0; ilams[k] = 0.0; ilamg[k] = 0.0;        // unset in
        n0r[k] = 0.0; n0s[k] = 0.0; n0g[k] = 0.0;              // WRF too
        if (qr[idx] > 1.0e-9f) {                               // :4544-4556
            rr[k] = (float)((double)qr[idx] * rho);
            lqr[k] = true;
            if (!(nr[idx] > 0.0f) || !isfinite((double)nr[idx])) {
                invalid_moment[k] = true;
            } else {
                double nrk = (double)nr[idx] * rho;
                double lamr = pow(RXAM_R * RGAMMA_4 * 1.0 * nrk
                                  / (double)rr[k], xobm);
                ilamr[k] = 1.0 / lamr;
                n0r[k] = nrk * 1.0 * lamr;  // xorg2 * lamr**xcre(2)
            }
        }
        if (qs[idx] > 1.0e-9f) {                               // :4558-4570
            rs[k] = (float)((double)qs[idx] * rho);
            lqs[k] = true;
            if (!(ns[idx] > 0.0f) || !isfinite((double)ns[idx])) {
                invalid_moment[k] = true;
            } else {
                double nsk = (double)ns[idx] * rho;
                double lams = pow(RXAM_S * RGAMMA_4 * 1.0 * nsk
                                  / (double)rs[k], xobm);
                ilams[k] = 1.0 / lams;
                n0s[k] = nsk * 1.0 * lams;
            }
        }
        if (qg[idx] > 1.0e-9f) {                               // :4572-4584
            rg[k] = (float)((double)qg[idx] * rho);
            lqg[k] = true;
            if (!(ng[idx] > 0.0f) || !isfinite((double)ng[idx])) {
                invalid_moment[k] = true;
            } else {
                double ngk = (double)ng[idx] * rho;
                double lamg = pow(xam_g * RGAMMA_4 * 1.0 * ngk
                                  / (double)rg[k], xobm);
                ilamg[k] = 1.0 / lamg;
                n0g[k] = ngk * 1.0 * lamg;
            }
        }
    }

    // Highest warm rainy level with frozen species just above; k_0 is
    // the level above it (:4586-4599).
    bool melti = false;
    int k_0 = 0;
    for (int k = nz - 2; k >= 0; --k) {
        if ((double)t[I3(k, j, i, ny, nx)] > 273.15 && lqr[k]
            && (lqs[k + 1] || lqg[k + 1])) {
            k_0 = max(k + 1, k_0);
            melti = true;
            break;
        }
    }

    // Rayleigh integrals; dry snow/graupel carry the
    // (0.176/0.93)*(6/pi)^2*(am/900)^2 adjustment (:4601-4620).
    for (int k = 0; k < nz; ++k) {
        zer[k] = 1.0e-22f; zes[k] = 1.0e-22f; zeg[k] = 1.0e-22f;
        if (lqr[k])
            zer[k] = (float)(n0r[k] * RGAMMA_7 * pow(ilamr[k], 7.0));
        if (lqs[k])
            zes[k] = (float)((0.176 / 0.93) * (6.0 / RPI) * (6.0 / RPI)
                             * (RXAM_S / 900.0) * (RXAM_S / 900.0)
                             * n0s[k] * RGAMMA_7 * pow(ilams[k], 7.0));
        if (lqg[k])
            zeg[k] = (float)((0.176 / 0.93) * (6.0 / RPI) * (6.0 / RPI)
                             * (xam_g / 900.0) * (xam_g / 900.0)
                             * n0g[k] * RGAMMA_7 * pow(ilamg[k], 7.0));
    }

    // Melting snow/graupel: 50-bin Simpson integration of the soaked
    // particle backscatter below k_0, meltwater fraction from the mass
    // ratio against the k_0 reference level (:4626-4667).
    if (melti && k_0 >= 1) {
        for (int k = k_0 - 1; k >= 0; --k) {
            if (lqs[k] && lqs[k_0]) {                          // :4629-4645
                double fmelt_s = fmax(0.005,
                    fmin(1.0 - (double)rs[k] / (double)rs[k_0], 0.99));
                double eta = 0.0;
                double lams = 1.0 / ilams[k];
                for (int n = 0; n < REFL_NRBINS; ++n) {
                    double x = RXAM_S * xxds[n] * xxds[n] * xxds[n];
                    double cback = refl_rayleigh_soak_wetgraupel(
                        x, xocms, xobm, fmelt_s, RMELT_OUTSIDE,
                        m_w_0, m_i_0);
                    double f_d = n0s[k] * exp(-lams * xxds[n]); // xmu_s = 0
                    eta += f_d * cback * simpson[n] * xdts[n];
                }
                zes[k] = (float)(RLAMDA4 / (RPI5 * k_w) * eta);
            }
            if (lqg[k] && lqg[k_0]) {                          // :4648-4665
                double fmelt_g = fmax(0.005,
                    fmin(1.0 - (double)rg[k] / (double)rg[k_0], 0.99));
                double eta = 0.0;
                double lamg = 1.0 / ilamg[k];
                for (int n = 0; n < REFL_NRBINS; ++n) {
                    double x = xam_g * xxdg[n] * xxdg[n] * xxdg[n];
                    double cback = refl_rayleigh_soak_wetgraupel(
                        x, xocmg, xobm, fmelt_g, RMELT_OUTSIDE,
                        m_w_0, m_i_0);
                    double f_d = n0g[k] * exp(-lamg * xxdg[n]); // xmu_g = 0
                    eta += f_d * cback * simpson[n] * xdtg[n];
                }
                zeg[k] = (float)(RLAMDA4 / (RPI5 * k_w) * eta);
            }
        }
    }

    // dBZ from the REAL ze sum (:4670-4671) with the wrapper's floor
    // (:916).
    for (int k = 0; k < nz; ++k) {
        float zsum = zer[k] + zes[k] + zeg[k];
        double dbz = 10.0 * log10((double)zsum * 1.0e18);
        // WRF has no guard for an active mass with a zero/negative/nonfinite
        // number moment (:4544-4584); its slope arithmetic is invalid before
        // the wrapper floor.  Preserve that invalidity explicitly instead of
        // letting CUDA fmaxf turn NaN into meteorological clear air.
        refl[I3(k, j, i, ny, nx)] = (invalid_moment[k]
            ? nanf("") : fmaxf(-35.0f, (float)dbz));
    }
}

// ---- Classic Thompson (mp_physics=8) calc_refl10cm -----------------------

__device__ __forceinline__ float thompson_snow_field_a(
    const float tc, const float moment)
{
    const float sa0 = 5.065339f, sa1 = -0.062659f;
    const float sa2 = -3.032362f, sa3 = 0.029469f;
    const float sa4 = -0.000285f, sa5 = 0.31255f;
    const float sa6 = 0.000204f, sa7 = 0.003199f;
    const float sa8 = 0.0f, sa9 = -0.015952f;
    const float tc2 = tc * tc, m2 = moment * moment;
    const float loga = sa0 + sa1 * tc + sa2 * moment
        + sa3 * tc * moment + sa4 * tc2 + sa5 * m2
        + sa6 * tc2 * moment + sa7 * tc * m2
        + sa8 * tc2 * tc + sa9 * m2 * moment;
    return powf(10.0f, loga);
}

__device__ __forceinline__ float thompson_snow_field_b(
    const float tc, const float moment)
{
    const float sb0 = 0.476221f, sb1 = -0.015896f;
    const float sb2 = 0.165977f, sb3 = 0.007468f;
    const float sb4 = -0.000141f, sb5 = 0.060366f;
    const float sb6 = 0.000079f, sb7 = 0.000594f;
    const float sb8 = 0.0f, sb9 = -0.003577f;
    const float tc2 = tc * tc, m2 = moment * moment;
    return sb0 + sb1 * tc + sb2 * moment + sb3 * tc * moment
        + sb4 * tc2 + sb5 * m2 + sb6 * tc2 * moment
        + sb7 * tc * m2 + sb8 * tc2 * tc + sb9 * m2 * moment;
}

extern "C" __global__ void refl10cm_thompson_column(
    const real* __restrict__ qv, const real* __restrict__ qr,
    const real* __restrict__ nr, const real* __restrict__ qs,
    const real* __restrict__ qg,
    const real* __restrict__ graupel_number_shadow,
    const real* __restrict__ t, const real* __restrict__ p,
    const double* __restrict__ tables,
    const double k_w, const double m_w0_re, const double m_w0_im,
    const double m_i0_re, const double m_i0_im,
    real* __restrict__ refl, const int nz, const int ny, const int nx)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    const int j = col / nx;
    const int i = col - j * nx;

    const double* xxds = tables;
    const double* xdts = tables + REFL_NRBINS;
    const double* simpson = tables + 4 * REFL_NRBINS;
    const cd mw = cd_make(m_w0_re, m_w0_im);
    const cd mi = cd_make(m_i0_re, m_i0_im);

    // module_mp_thompson.F classic mp=8 constants.  Unlike Morrison/WSM6,
    // Thompson snow is the Field et al. sum of two gamma distributions and
    // has m(D)=0.069 D**2.  Classic graupel is the fixed 400 kg m-3 member.
    const float tpi = 3.1415926536f;
    const float am_r = tpi * 1000.0f / 6.0f;
    const float am_s = 0.069f;
    const float am_g = tpi * 400.0f / 6.0f;
    const double obm_r = 1.0 / 3.0;
    const double obm_g = 1.0 / 3.0;
    const double ocms = sqrt(1.0 / (double)am_s);

    float rr[REFL_KMAX], rs[REFL_KMAX], rg[REFL_KMAX];
    float smob[REFL_KMAX], smoc[REFL_KMAX], smoz[REFL_KMAX];
    double ilamr[REFL_KMAX], n0r[REFL_KMAX];
    double ilamg[REFL_KMAX], n0g[REFL_KMAX];
    bool lqr[REFL_KMAX], lqs[REFL_KMAX], lqg[REFL_KMAX];
    float zer[REFL_KMAX], zes[REFL_KMAX], zeg[REFL_KMAX];

    for (int k = 0; k < nz; ++k) {
        const size_t idx = I3(k, j, i, ny, nx);
        const float qvk = fmaxf(1.0e-10f, qv[idx]);
        const float rho = 0.622f * p[idx]
            / (287.04f * t[idx] * (qvk + 0.622f));
        rr[k] = rs[k] = rg[k] = 1.0e-12f;
        smob[k] = smoc[k] = smoz[k] = 0.0f;
        ilamr[k] = n0r[k] = ilamg[k] = n0g[k] = 0.0;
        lqr[k] = lqs[k] = lqg[k] = false;

        if (qr[idx] > 1.0e-12f) {
            rr[k] = qr[idx] * rho;
            const float nr_vol = fmaxf(1.0e-6f, nr[idx] * rho);
            const double lamr = pow(
                (double)(am_r * 6.0f * nr_vol / rr[k]), obm_r);
            ilamr[k] = 1.0 / lamr;
            n0r[k] = (double)nr_vol * lamr;
            lqr[k] = true;
        }
        if (qs[idx] > 1.0e-6f) {
            rs[k] = qs[idx] * rho;
            const float tc0 = fminf(-0.1f, t[idx] - 273.15f);
            smob[k] = rs[k] / am_s;
            const float smo2 = smob[k];
            smoc[k] = thompson_snow_field_a(tc0, 3.0f)
                * powf(smo2, thompson_snow_field_b(tc0, 3.0f));
            smoz[k] = thompson_snow_field_a(tc0, 4.0f)
                * powf(smo2, thompson_snow_field_b(tc0, 4.0f));
            lqs[k] = true;
        }
        if (qg[idx] > 1.0e-6f) {
            rg[k] = qg[idx] * rho;
            const float ng_vol = fmaxf(
                1.0e-6f, graupel_number_shadow[idx] * rho);
            const double lamg = pow(
                (double)(am_g * 6.0f * ng_vol / rg[k]), obm_g);
            ilamg[k] = 1.0 / lamg;
            n0g[k] = (double)ng_vol * lamg;
            lqg[k] = true;
        }
    }

    bool melti = false;
    int k0 = 0;
    for (int k = nz - 2; k >= 0; --k) {
        if (t[I3(k, j, i, ny, nx)] > 273.15f && lqr[k]
            && (lqs[k + 1] || lqg[k + 1])) {
            k0 = max(k + 1, k0);
            melti = true;
            break;
        }
    }

    for (int k = 0; k < nz; ++k) {
        zer[k] = zes[k] = zeg[k] = 1.0e-22f;
        if (lqr[k])
            zer[k] = (float)(n0r[k] * 720.0 * pow(ilamr[k], 7.0));
        if (lqs[k])
            zes[k] = (0.176f / 0.93f) * (6.0f / tpi) * (6.0f / tpi)
                * (am_s / 900.0f) * (am_s / 900.0f) * smoz[k];
        if (lqg[k])
            zeg[k] = (float)((0.176 / 0.93) * (6.0 / (double)tpi)
                * (6.0 / (double)tpi) * ((double)am_g / 900.0)
                * ((double)am_g / 900.0) * n0g[k] * 720.0
                * pow(ilamg[k], 7.0));
    }

    // Thompson's graupel bright-band block is commented out in WRF v4.6.1;
    // only the native Field-snow distribution enters the Blahak soak.
    if (melti && k0 >= 1) {
        for (int k = k0 - 1; k >= 0; --k) {
            if (lqs[k] && lqs[k0]) {
                const double fmelt = fmax(
                    0.05, fmin(1.0 - (double)rs[k] / (double)rs[k0], 0.99));
                const float m0f = smob[k] / smoc[k];
                const float mratf = smob[k] * m0f * m0f * m0f;
                const double m0 = (double)m0f;
                const double mrat = (double)mratf;
                const double slam1 = m0 * 20.78;
                const double slam2 = m0 * 3.29;
                double eta = 0.0;
                for (int n = 0; n < REFL_NRBINS; ++n) {
                    const double d = xxds[n];
                    const double x = (double)am_s * d * d;
                    const double cback = refl_rayleigh_soak_wetgraupel(
                        x, ocms, 0.5, fmelt, RMELT_OUTSIDE, mw, mi);
                    const double fd = mrat * (
                        490.6 * exp(-slam1 * d)
                        + 17.46 * pow(m0 * d, 0.6357)
                            * exp(-slam2 * d));
                    eta += fd * cback * simpson[n] * xdts[n];
                }
                zes[k] = (float)(RLAMDA4 / (RPI5 * k_w) * eta);
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        const float zsum = zer[k] + zes[k] + zeg[k];
        const float dbz = (float)(10.0 * log10((double)zsum * 1.0e18));
        refl[I3(k, j, i, ny, nx)] = fmaxf(-35.0f, dbz);
    }
}

// ---- Kessler (mp_physics=1) rain-only fallback ----------------------------

extern "C" __global__ void refl10cm_kessler_cell(
    const real* __restrict__ qv, const real* __restrict__ qr,
    const real* __restrict__ t, const real* __restrict__ p,
    real* __restrict__ refl, const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // Same density diagnosis, thresholds, and floors as refl10cm_hm so
    // both schemes share one output convention.
    double qvk = fmax(1.0e-10, (double)qv[idx]);
    double rho = 0.622 * (double)p[idx]
               / (R_D * (double)t[idx] * (qvk + 0.622));
    float ze = 1.0e-22f;
    if (qr[idx] > 1.0e-9f) {
        // Smith 1975 exponential-PSD rain: lambda = (pi*rho_w*N0r /
        // (rho*qr))**0.25, Ze = Gamma(7)*N0r/lambda**7 (m6 m-3).
        double lam = pow(RPI * 1000.0 * 8.0e6 / (rho * (double)qr[idx]),
                         0.25);
        ze = (float)(720.0 * 8.0e6 / pow(lam, 7.0));
    }
    double dbz = 10.0 * log10((double)ze * 1.0e18);
    refl[idx] = fmaxf(-35.0f, (float)dbz);
}

// ---- WSM6 refl10cm_wsm6 column kernel -----------------------------------
// Fixed-intercept PSDs and density choices follow mp_wsm6.F90:2275-2444.
// The melting-particle integration reuses the exact mp_radar machinery above.
extern "C" __global__ void refl10cm_wsm6_column(
    const real* __restrict__ qv, const real* __restrict__ qr,
    const real* __restrict__ qs, const real* __restrict__ qg,
    const real* __restrict__ t, const real* __restrict__ p,
    const double* __restrict__ tables,
    const double k_w, const double m_w0_re, const double m_w0_im,
    const double m_i0_re, const double m_i0_im, const double xam_g,
    const double n0_g, real* __restrict__ refl,
    const int nz, const int ny, const int nx)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ny * nx) return;
    int j = col / nx, i = col - j * nx;
    const double* xxds = tables;
    const double* xdts = tables + REFL_NRBINS;
    const double* xxdg = tables + 2 * REFL_NRBINS;
    const double* xdtg = tables + 3 * REFL_NRBINS;
    const double* simpson = tables + 4 * REFL_NRBINS;
    const cd mw = cd_make(m_w0_re, m_w0_im);
    const cd mi = cd_make(m_i0_re, m_i0_im);
    const double xam_r = RPI * 1000.0 / 6.0;
    const double xam_s = RPI * 100.0 / 6.0;
    const double xobm = 1.0 / 3.0;
    const double xocms = pow(1.0 / xam_s, xobm);
    const double xocmg = pow(1.0 / xam_g, xobm);
    float rr[REFL_KMAX], rs[REFL_KMAX], rg[REFL_KMAX];
    double ilamr[REFL_KMAX], ilams[REFL_KMAX], ilamg[REFL_KMAX];
    double n0s[REFL_KMAX];
    bool lqr[REFL_KMAX], lqs[REFL_KMAX], lqg[REFL_KMAX];
    float zer[REFL_KMAX], zes[REFL_KMAX], zeg[REFL_KMAX];
    for (int k = 0; k < nz; ++k) {
        size_t idx = I3(k, j, i, ny, nx);
        double tk = (double)t[idx];
        double qvk = fmax(1.0e-10, (double)qv[idx]);
        double rho = 0.622 * (double)p[idx] / (R_D * tk * (qvk + 0.622));
        rr[k] = rs[k] = rg[k] = 1.0e-12f;
        lqr[k] = lqs[k] = lqg[k] = false;
        if (qr[idx] > 1.0e-9f) {
            rr[k] = (float)((double)qr[idx] * rho);
            double lam = pow(xam_r * RGAMMA_4 * 8.0e6 / rr[k], 0.25);
            ilamr[k] = 1.0 / lam; lqr[k] = true;
        }
        double tc = fmin(-0.001, tk - 273.15);
        n0s[k] = fmin(1.0e11, 2.0e6 * exp(-0.12 * tc));
        if (qs[idx] > 1.0e-9f) {
            rs[k] = (float)((double)qs[idx] * rho);
            double lam = pow(xam_s * RGAMMA_4 * n0s[k] / rs[k], 0.25);
            ilams[k] = 1.0 / lam; lqs[k] = true;
        }
        if (qg[idx] > 1.0e-9f) {
            rg[k] = (float)((double)qg[idx] * rho);
            double lam = pow(xam_g * RGAMMA_4 * n0_g / rg[k], 0.25);
            ilamg[k] = 1.0 / lam; lqg[k] = true;
        }
    }
    bool melti = false; int k0 = 0;
    for (int k = nz - 2; k >= 0; --k) {
        if (t[I3(k,j,i,ny,nx)] > 273.15f && lqr[k]
            && (lqs[k+1] || lqg[k+1])) {
            k0 = max(k + 1, k0); melti = true; break;
        }
    }
    for (int k = 0; k < nz; ++k) {
        zer[k] = zes[k] = zeg[k] = 1.0e-22f;
        if (lqr[k]) zer[k] = (float)(8.0e6 * RGAMMA_7 * pow(ilamr[k],7.0));
        if (lqs[k]) zes[k] = (float)((0.176/0.93)*pow(6.0/RPI,2.0)
            *pow(xam_s/900.0,2.0)*n0s[k]*RGAMMA_7*pow(ilams[k],7.0));
        if (lqg[k]) zeg[k] = (float)((0.176/0.93)*pow(6.0/RPI,2.0)
            *pow(xam_g/900.0,2.0)*n0_g*RGAMMA_7*pow(ilamg[k],7.0));
    }
    if (melti && k0 >= 1) {
        for (int k = k0 - 1; k >= 0; --k) {
            if (lqs[k] && lqs[k0]) {
                double fm = fmax(0.005, fmin(1.0-(double)rs[k]/rs[k0],0.99));
                double eta=0.0, lam=1.0/ilams[k];
                for(int n=0;n<REFL_NRBINS;++n){
                    double x=xam_s*xxds[n]*xxds[n]*xxds[n];
                    double cb=refl_rayleigh_soak_wetgraupel(
                        x,xocms,xobm,fm,RMELT_OUTSIDE,mw,mi);
                    eta += n0s[k]*exp(-lam*xxds[n])*cb*simpson[n]*xdts[n];
                }
                zes[k]=(float)(RLAMDA4/(RPI5*k_w)*eta);
            }
            if (lqg[k] && lqg[k0]) {
                double fm=fmax(0.005,fmin(1.0-(double)rg[k]/rg[k0],0.99));
                double eta=0.0, lam=1.0/ilamg[k];
                for(int n=0;n<REFL_NRBINS;++n){
                    double x=xam_g*xxdg[n]*xxdg[n]*xxdg[n];
                    double cb=refl_rayleigh_soak_wetgraupel(
                        x,xocmg,xobm,fm,RMELT_OUTSIDE,mw,mi);
                    eta += n0_g*exp(-lam*xxdg[n])*cb*simpson[n]*xdtg[n];
                }
                zeg[k]=(float)(RLAMDA4/(RPI5*k_w)*eta);
            }
        }
    }
    for(int k=0;k<nz;++k){
        float zsum=zer[k]+zes[k]+zeg[k];
        refl[I3(k,j,i,ny,nx)]=fmaxf(-35.0f,(float)(10.0*log10((double)zsum*1e18)));
    }
}
