// gpuwm/core/kernels/wdm6_refl.cu
//
// WDM6 radar reflectivity (WRF do_radar_ref=1, REFL_10CM at mp_physics=16).
// Transcription authority (byte-frozen WRF v4.6.1 reference bundle):
//   phys/module_mp_wdm6.F: refl10cm_wdm6 (:2957-3131) and the m(D)
//     parameters wdm6init hands radar_init (:2196-2208; xmu_r = 1 for rain,
//     0 for snow and graupel).
//   phys/module_mp_radar.F: the Blahak melting-particle backscatter
//     (rayleigh_soak_wetgraupel, get_m_mix_nested, get_m_mix,
//     m_complex_maxwellgarnett).
//
// WHY THIS IS A SEPARATE TRANSLATION UNIT.  Everything below the header is
// refl.cu's device prologue, copied verbatim.  It is copied rather than
// shared because refl.cu is BYTE-FROZEN: tests/test_mp8_frozen.py pins its
// file and compiled-source sha256 and its own docstring states the rule --
// a new scheme gets a new .cu file, so that no scheme's compiled numerics
// can move because another scheme was added.  gpuwm/core/kernels/wdm6.cu
// makes the same trade against wsm6.cu, for the same reason.
//
// The duplication is guarded rather than trusted: tests/test_wdm6.py
// asserts this file's copy of the prologue is byte-identical to refl.cu's,
// so the two cannot drift apart silently.  Because the copy is verbatim it
// carries Morrison constants WDM6 does not use; the two that would MISLEAD
// (RXAM_R at RHOW = 997, RXAM_S) are #undef'd immediately after the
// prologue ends, so this file cannot reach them.
//
// One CUDA thread per (j, i) column; the melting-level scan k_0 is a
// column-serial dependency.  radar_init products (size bins, Simpson
// weights, K_w, m_w_0, m_i_0) arrive from gpuwm.core.refl.radar_init_wdm6.

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

// ^^ that heading is the last line of refl.cu's copied prologue, and this is
// where the copy ends.  RXAM_R and RXAM_S are DEAD below it and are removed
// here rather than left lying around: any later use is now a compile error,
// not a wrong number.  They cannot be deleted at their definition -- the
// block above is refl.cu's prologue byte-for-byte and
// tests/test_wdm6.py::test_the_wdm6_refl_prologue_is_byte_identical_to_
// refl_cu_s pins it that way -- and they are Morrison's constants, not
// WDM6's: RXAM_R uses RHOW = 997 (module_mp_morr_two_moment.F:532-542),
// while wdm6init calls radar_init with denr = rhowater = 1000
// (module_mp_wdm6.F:2196-2208).  The xam_r this kernel uses arrives from
// gpuwm.core.refl.radar_init_wdm6 and is PI*1000/6.  A reader checking rain
// density against WRF must not find 997 answering for WDM6.
#undef RXAM_R
#undef RXAM_S

// WDM6 refl10cm_wdm6 (module_mp_wdm6.F:2957-3131).  Structurally WSM6's
// routine with ONE substitution: rain is a gamma PSD with xmu_r = 1 and a
// PROGNOSTIC number, so the fixed Marshall-Palmer intercept is replaced by
// the pair
//     lamr = (xam_r*xcrg(3)*xorg2*nr/rr)**xobmr        (:3005)
//     N0_r = nr*xorg2*lamr**xcre(2)                    (:3007)
// with wdm6init's radar_init inputs xbm_r = 3, xmu_r = 1 (:2198-2200) and
// therefore xcre = (4, 2, 5, 8), xcrg = (6, 1, 24, 5040), xorg2 = 1/G(2) = 1,
// xobmr = 1/3.  Snow and graupel keep xmu = 0 and are WSM6's expressions
// verbatim.
//
// INHERITED INCONSISTENCY, transcribed as written (not a bug in the sense
// the "never bit-exact to a bug" rule covers -- nothing here is undefined,
// the two routines simply disagree): wdm62D consumes ncr as a VOLUMETRIC
// number, dividing by (q*den) at :2251, while this routine forms
// nr = nr1d*rho and rr = qr1d*rho so the densities cancel and nr1d enters
// as a PER-MASS number.  The two lamr definitions therefore differ by a
// factor rho inside the cube root (about 3 per cent in lamr near the
// surface).  WRF does this; reproducing it is the port's job, and the
// divergence is recorded here rather than silently repaired.
extern "C" __global__ void refl10cm_wdm6_column(
    const real* __restrict__ qv, const real* __restrict__ qr,
    const real* __restrict__ nr_in, const real* __restrict__ qs,
    const real* __restrict__ qg,
    const real* __restrict__ t, const real* __restrict__ p,
    const double* __restrict__ tables,
    const double k_w, const double m_w0_re, const double m_w0_im,
    const double m_i0_re, const double m_i0_im, const double xam_g,
    const double n0_g,
    // ONE SOURCE OF TRUTH for the rain m(D)/PSD moments.  These used to be
    // literals here AND a derivation in gpuwm.core.refl.radar_init_wdm6
    // that nothing consumed, so changing xmu_r on the host would have left
    // the kernel silently on the old exponents.  They now arrive from that
    // derivation; the launcher refuses if xcre is not (4, 2, 5, 8), which
    // is what lets the two exponents below stay compile-time.
    const double xam_r, const double xcrg3, const double xcrg4,
    const double xorg2, real* __restrict__ refl,
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
    // xam_r, xcrg3, xcrg4 and xorg2 are arguments now (radar_init with
    // xmu_r = 1: xcrg(3) = G(5) = 24, xcrg(4) = G(8) = 5040,
    // xorg2 = 1/G(2) = 1, and xam_r = PI*denr/6 with WDM6's denr = 1000).
    // Only the two INTEGER exponents xcre(2) = 2 and xcre(4) = 8 stay
    // compile-time, written as lam*lam and pow(ilamr, 8.0); the launcher
    // refuses any xcre that is not (4, 2, 5, 8), so they cannot go stale.
    const double xam_s = RPI * 100.0 / 6.0;
    const double xobm = 1.0 / 3.0;
    const double xocms = pow(1.0 / xam_s, xobm);
    const double xocmg = pow(1.0 / xam_g, xobm);
    float rr[REFL_KMAX], rs[REFL_KMAX], rg[REFL_KMAX];
    double ilamr[REFL_KMAX], ilams[REFL_KMAX], ilamg[REFL_KMAX];
    double n0r[REFL_KMAX], n0s[REFL_KMAX];
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
            double nrk = (double)nr_in[idx] * rho;
            double lam = pow(xam_r * xcrg3 * xorg2 * nrk / rr[k], xobm);
            ilamr[k] = 1.0 / lam;
            n0r[k] = nrk * xorg2 * lam * lam;   // lam**xcre(2), xcre(2) = 2
            lqr[k] = true;
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
        if (lqr[k]) zer[k] = (float)(n0r[k] * xcrg4 * pow(ilamr[k], 8.0));
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
