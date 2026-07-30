// gpuwm/core/kernels/sfclay.cu
//
// WRF v4.6.1 MM5 surface-layer schemes, transcribed from
//   phys/module_sf_sfclay.F          SFCLAY1D (option 91), and
//   phys/physics_mmm/sf_sfclayrev.F90 sf_sfclayrev_run (option 1).
// module_sf_sfclayrev.F is only the WRF/CCPP array wrapper.  One CUDA thread
// handles one (j,i) surface column; inputs are the lowest mass-level values
// already interpolated to mass points exactly as the WRF wrappers do.
//
// The incoming znt, ust, mol, hfx, qfx, qsfc, and zol are previous-step/inout
// fields.  This matters: classic unstable z/L uses old MOL, its strong-stable
// branch leaves ZOL untouched, and both schemes average newly diagnosed u*
// with old UST.  Outputs remain FP32 model state;
// gpuwm.verify.npref.np_sfclay is the float64 transcription mirror.

__device__ __forceinline__ real sf_psim_classic_full(real z)
{
    real x = powf(1.0f - 16.0f * z, 0.25f);
    return 2.0f * logf(0.5f * (1.0f + x))
         + logf(0.5f * (1.0f + x * x)) - 2.0f * atanf(x)
         + 2.0f * atanf(1.0f);
}

__device__ __forceinline__ real sf_psih_classic_full(real z)
{
    real y = sqrtf(1.0f - 16.0f * z);
    return 2.0f * logf(0.5f * (1.0f + y));
}

__device__ __forceinline__ real sf_classic_table(real z, bool heat)
{
    real x = fminf(fmaxf(-z, 0.0f), 9.9999f) * 100.0f;
    int n = (int)x;
    real r = x - (real)n;
    real z0 = -0.01f * (real)n, z1 = -0.01f * (real)(n + 1);
    real f0 = heat ? sf_psih_classic_full(z0) : sf_psim_classic_full(z0);
    real f1 = heat ? sf_psih_classic_full(z1) : sf_psim_classic_full(z1);
    return f0 + r * (f1 - f0);
}

__device__ __forceinline__ real sf_psim_stable_full(real z)
{
    return -6.1f * logf(z + powf(1.0f + powf(z, 2.5f), 1.0f / 2.5f));
}

__device__ __forceinline__ real sf_psih_stable_full(real z)
{
    return -5.3f * logf(z + powf(1.0f + powf(z, 1.1f), 1.0f / 1.1f));
}

__device__ __forceinline__ real sf_psim_unstable_full(real z)
{
    real x = powf(1.0f - 16.0f * z, 0.25f);
    real psimk = 2.0f * logf(0.5f * (1.0f + x))
               + logf(0.5f * (1.0f + x * x)) - 2.0f * atanf(x)
               + 2.0f * atanf(1.0f);
    real ym = powf(1.0f - 10.0f * z, 0.33f); // file literal .33
    real rt3 = sqrtf(3.0f);
    real psimc = 1.5f * logf((ym * ym + ym + 1.0f) / 3.0f)
                - rt3 * atanf((2.0f * ym + 1.0f) / rt3)
                + 4.0f * atanf(1.0f) / rt3;
    return (psimk + z * z * psimc) / (1.0f + z * z);
}

__device__ __forceinline__ real sf_psih_unstable_full(real z)
{
    real y = sqrtf(1.0f - 16.0f * z);
    real psihk = 2.0f * logf((1.0f + y) / 2.0f);
    real yh = powf(1.0f - 34.0f * z, 0.33f);
    real rt3 = sqrtf(3.0f);
    real psihc = 1.5f * logf((yh * yh + yh + 1.0f) / 3.0f)
                - rt3 * atanf((2.0f * yh + 1.0f) / rt3)
                + 4.0f * atanf(1.0f) / rt3;
    return (psihk + z * z * psihc) / (1.0f + z * z);
}

__device__ __forceinline__ real sf_rev_table(real z, int which)
{
    // which: 0 psim stable, 1 psih stable, 2 psim unstable, 3 psih unstable
    bool unstable = which >= 2;
    z = unstable ? fminf(z, 0.0f) : fmaxf(z, 0.0f);
    real x = (unstable ? -z : z) * 100.0f;
    int n = (int)x;
    if (n + 1 >= 1000) {
        if (which == 0) return sf_psim_stable_full(z);
        if (which == 1) return sf_psih_stable_full(z);
        if (which == 2) return sf_psim_unstable_full(z);
        return sf_psih_unstable_full(z);
    }
    real r = x - (real)n;
    real sign = unstable ? -1.0f : 1.0f;
    real z0 = sign * 0.01f * (real)n;
    real z1 = sign * 0.01f * (real)(n + 1);
    real f0, f1;
    if (which == 0) { f0 = sf_psim_stable_full(z0); f1 = sf_psim_stable_full(z1); }
    else if (which == 1) { f0 = sf_psih_stable_full(z0); f1 = sf_psih_stable_full(z1); }
    else if (which == 2) { f0 = sf_psim_unstable_full(z0); f1 = sf_psim_unstable_full(z1); }
    else { f0 = sf_psih_unstable_full(z0); f1 = sf_psih_unstable_full(z1); }
    return f0 + r * (f1 - f0);
}

__device__ __forceinline__ real sf_psim_stable(real z) { return sf_rev_table(z, 0); }
__device__ __forceinline__ real sf_psih_stable(real z) { return sf_rev_table(z, 1); }
__device__ __forceinline__ real sf_psim_unstable(real z) { return sf_rev_table(z, 2); }
__device__ __forceinline__ real sf_psih_unstable(real z) { return sf_rev_table(z, 3); }

__device__ __forceinline__ real sf_zolri_residual(real &zeta, real ri,
                                                   real z, real z0)
{
    if (zeta * ri < 0.0f) zeta = 0.0f;
    real zeta0 = zeta * z0 / z;
    real zeta3 = zeta + zeta0;
    real fm, fh;
    if (ri < 0.0f) {
        fm = logf((z + z0) / z0)
           - (sf_psim_unstable(zeta3) - sf_psim_unstable(zeta0));
        fh = logf((z + z0) / z0)
           - (sf_psih_unstable(zeta3) - sf_psih_unstable(zeta0));
    } else {
        fm = logf((z + z0) / z0)
           - (sf_psim_stable(zeta3) - sf_psim_stable(zeta0));
        fh = logf((z + z0) / z0)
           - (sf_psih_stable(zeta3) - sf_psih_stable(zeta0));
    }
    return zeta * fh / (fm * fm) - ri;
}

__device__ __forceinline__ real sf_zolri(real ri, real z, real z0)
{
    real x1 = ri < 0.0f ? -5.0f : 0.0f;
    real x2 = ri < 0.0f ? 0.0f : 5.0f;
    real fx1 = sf_zolri_residual(x1, ri, z, z0);
    real fx2 = sf_zolri_residual(x2, ri, z, z0);
    real result = fabsf(fx1) < fabsf(fx2) ? x1 : x2;
    int iter = 0;
    while (fabsf(x1 - x2) > 0.01f) {
        if (iter == 10 || fx1 == fx2) return result;
        if (fabsf(fx2) < fabsf(fx1)) {
            x1 = x1 - fx1 / (fx2 - fx1) * (x2 - x1);
            fx1 = sf_zolri_residual(x1, ri, z, z0);
            result = x1;
        } else {
            x2 = x2 - fx2 / (fx2 - fx1) * (x2 - x1);
            fx2 = sf_zolri_residual(x2, ri, z, z0);
            result = x2;
        }
        ++iter;
    }
    return result;
}

__device__ __forceinline__ real sf_rev_heat_psi(real zol, real za,
                                                 real rough, real height)
{
    real zh = zol * (height + rough) / za;
    real z0 = zol * rough / za;
    if (zol > 0.0f) return sf_psih_stable(zh) - sf_psih_stable(z0);
    if (zol < 0.0f) return sf_psih_unstable(zh) - sf_psih_unstable(z0);
    return 0.0f;
}

extern "C" __global__
void sfclay_column(
    const real* __restrict__ u, const real* __restrict__ v,
    const real* __restrict__ t, const real* __restrict__ qv,
    const real* __restrict__ p, const real* __restrict__ dz8w,
    const real* __restrict__ psfc, const real* __restrict__ tsk,
    const real* __restrict__ pblh, const real* __restrict__ mavail,
    const real* __restrict__ xland, const real* __restrict__ lakemask,
    real* __restrict__ znt, real* __restrict__ ust, real* __restrict__ mol,
    real* __restrict__ hfx, real* __restrict__ qfx, real* __restrict__ qsfc,
    real* __restrict__ zol_o, real* __restrict__ regime_o,
    real* __restrict__ psim_o, real* __restrict__ psih_o,
    real* __restrict__ fm_o, real* __restrict__ fh_o,
    real* __restrict__ lh_o, real* __restrict__ u10_o,
    real* __restrict__ v10_o, real* __restrict__ th2_o,
    real* __restrict__ t2_o, real* __restrict__ q2_o,
    real* __restrict__ chs_o, real* __restrict__ chs2_o,
    real* __restrict__ cqs2_o, real* __restrict__ flhc_o,
    real* __restrict__ flqc_o, real* __restrict__ qgh_o,
    real* __restrict__ rmol_o, real* __restrict__ wspd_o,
    real* __restrict__ br_o, real* __restrict__ gz1_o,
    real* __restrict__ cpm_o, real* __restrict__ ck_o,
    real* __restrict__ cka_o, real* __restrict__ cd_o,
    real* __restrict__ cda_o,
    real dx, int option, int isfflx, int isftcflx, int iz0tlnd, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const real karman = 0.4f, ep1 = RV / RD - 1.0f, xka = 2.4e-5f;
    bool land = xland[idx] < 1.5f;
    real uu = u[idx], vv = v[idx], temp = t[idx], qvx = qv[idx];
    real press = p[idx], ps = psfc[idx], ground_t = tsk[idx];
    real z0 = znt[idx], old_ust = ust[idx], old_mol = mol[idx];
    real old_zol = zol_o[idx];
    real old_hfx = hfx[idx], old_qfx = qfx[idx], qs = qsfc[idx];

    real thgb = ground_t * powf(P0 / ps, RCP);
    real thx = temp * powf(P0 / press, RCP);
    real thvx = thx * (1.0f + ep1 * qvx);
    real tv = temp * (1.0f + ep1 * qvx);
    real cpm = CP * (1.0f + 0.8f * qvx);
    real es = SVP1 * expf(SVP2 * (ground_t - SVPT0) / (ground_t - SVP3));
    if (!land && lakemask[idx] == 0.0f) es *= 0.98f;
    if (!land || qs <= 0.0f) qs = EP2 * es / (ps / 1000.0f - es);
    real es_air = SVP1 * expf(SVP2 * (temp - SVPT0) / (temp - SVP3));
    real qgh = EP2 * es_air / (press / 1000.0f - es_air);
    real rho = ps / (RD * tv);
    real za = 0.5f * dz8w[idx];
    real gz1, gz2, gz10;
    if (option == 91) {
        gz1 = logf(za / z0); gz2 = logf(2.0f / z0); gz10 = logf(10.0f / z0);
    } else {
        gz1 = logf((za + z0) / z0); gz2 = logf((2.0f + z0) / z0);
        gz10 = logf((10.0f + z0) / z0);
    }

    real tskv = thgb * (1.0f + ep1 * qs);
    real dthv = thvx - tskv;
    real wspd0 = hypotf(uu, vv), vconv;
    if (land) {
        real fluxc = fmaxf(old_hfx / rho / CP + ep1 * tskv * old_qfx / rho,
                           0.0f);
        vconv = powf(G / ground_t * pblh[idx] * fluxc, 0.33f);
    } else {
        vconv = sqrtf(fmaxf(-dthv, 0.0f));
    }
    real vsgd = 0.32f * powf(fmaxf(dx / 5000.0f - 1.0f, 0.0f), 0.33f);
    real wspd = fmaxf(sqrtf(wspd0 * wspd0 + vconv * vconv + vsgd * vsgd),
                       0.1f);
    real br = G / thx * za * dthv / (wspd * wspd);
    if (old_mol < 0.0f) br = fminf(br, 0.0f);

    real psim = 0.0f, psih = 0.0f, psim10 = 0.0f, psih10 = 0.0f;
    real psim2 = 0.0f, psih2 = 0.0f, pq = 0.0f, pq2 = 0.0f, pq10 = 0.0f;
    real zol = old_zol, regime, rmol;
    if (option == 91) {
        if (br >= 0.2f) {
            regime = 1.0f;
            psim = fmaxf(-10.0f * gz1, -10.0f); psih = psim;
            psim10 = fmaxf(10.0f / za * psim, -10.0f); psih10 = psim10;
            psim2 = fmaxf(2.0f / za * psim, -10.0f); psih2 = psim2;
            real za_over_l = old_ust < 0.01f ? br * gz1
                  : karman * G / thx * za * old_mol / (old_ust * old_ust);
            rmol = fminf(za_over_l, 9.999f) / za;
        } else if (br > 0.0f) {
            regime = 2.0f;
            psim = fmaxf(-5.0f * br * gz1 / (1.1f - 5.0f * br), -10.0f);
            psih = psim;
            psim10 = fmaxf(10.0f / za * psim, -10.0f); psih10 = psim10;
            psim2 = fmaxf(2.0f / za * psim, -10.0f); psih2 = psim2;
            zol = br * gz1 / (1.00001f - 5.0f * br);
            if (zol > 0.5f) {
                zol = (1.89f * gz1 + 44.2f) * br * br
                    + (1.18f * gz1 - 1.37f) * br;
                zol = fminf(zol, 9.999f);
            }
            rmol = zol / za;
        } else if (br == 0.0f) {
            regime = 3.0f;
            zol = old_ust < 0.01f ? br * gz1
                  : karman * G / thx * za * old_mol / (old_ust * old_ust);
            rmol = zol / za;
        } else {
            regime = 4.0f;
            zol = old_ust < 0.01f ? br * gz1
                  : karman * G / thx * za * old_mol / (old_ust * old_ust);
            real zol10 = fmaxf(fminf(10.0f / za * zol, 0.0f), -9.9999f);
            real zol2 = fmaxf(fminf(2.0f / za * zol, 0.0f), -9.9999f);
            zol = fmaxf(fminf(zol, 0.0f), -9.9999f);
            psim = sf_classic_table(zol, false); psih = sf_classic_table(zol, true);
            psim10 = sf_classic_table(zol10, false); psih10 = sf_classic_table(zol10, true);
            psim2 = sf_classic_table(zol2, false); psih2 = sf_classic_table(zol2, true);
            psih = fminf(psih, 0.9f * gz1); psim = fminf(psim, 0.9f * gz1);
            psih2 = fminf(psih2, 0.9f * gz2);
            psim10 = fminf(psim10, 0.9f * gz10);
            psih10 = fminf(psih10, 0.9f * gz10);
            rmol = zol / za;
        }
    } else {
        zol = 0.0f;
        if (br > 0.0f) zol = sf_zolri(fminf(br, 250.0f), za, z0);
        else if (br < 0.0f)
            zol = old_ust < 0.001f ? br * gz1
                  : sf_zolri(fmaxf(br, -250.0f), za, z0);
        real zz = zol * (za + z0) / za, z10 = zol * (10.0f + z0) / za;
        real z2 = zol * (2.0f + z0) / za, zz0 = zol * z0 / za;
        real scalar_z = land ? zol * 0.01f / za : zz0;
        if (br > 0.0f) {
            regime = 1.0f;
            psim = sf_psim_stable(zz) - sf_psim_stable(zz0);
            psih = sf_psih_stable(zz) - sf_psih_stable(zz0);
            psim10 = sf_psim_stable(z10) - sf_psim_stable(zz0);
            psih10 = sf_psih_stable(z10) - sf_psih_stable(zz0);
            psim2 = sf_psim_stable(z2) - sf_psim_stable(zz0);
            psih2 = sf_psih_stable(z2) - sf_psih_stable(zz0);
            pq = sf_psih_stable(zol) - sf_psih_stable(scalar_z);
            pq2 = sf_psih_stable(2.0f / za * zol) - sf_psih_stable(scalar_z);
            pq10 = sf_psih_stable(10.0f / za * zol) - sf_psih_stable(scalar_z);
        } else if (br == 0.0f) {
            regime = 3.0f; zol = 0.0f;
        } else {
            regime = 4.0f;
            psim = sf_psim_unstable(zz) - sf_psim_unstable(zz0);
            psih = sf_psih_unstable(zz) - sf_psih_unstable(zz0);
            psim10 = sf_psim_unstable(z10) - sf_psim_unstable(zz0);
            psih10 = sf_psih_unstable(z10) - sf_psih_unstable(zz0);
            psim2 = sf_psim_unstable(z2) - sf_psim_unstable(zz0);
            psih2 = sf_psih_unstable(z2) - sf_psih_unstable(zz0);
            pq = sf_psih_unstable(zol) - sf_psih_unstable(scalar_z);
            pq2 = sf_psih_unstable(2.0f / za * zol) - sf_psih_unstable(scalar_z);
            pq10 = sf_psih_unstable(10.0f / za * zol) - sf_psih_unstable(scalar_z);
            psih = fminf(psih, 0.9f * gz1); psim = fminf(psim, 0.9f * gz1);
            psih2 = fminf(psih2, 0.9f * gz2);
            psim10 = fminf(psim10, 0.9f * gz10);
            psih10 = fminf(psih10, 0.9f * gz10);
        }
        rmol = zol / za;
    }

    real dtg = thx - thgb;
    real psix = gz1 - psim, psix10 = gz10 - psim10;
    real psit = option == 91 ? fmaxf(gz1 - psih, 2.0f) : gz1 - psih;
    real psit2 = gz2 - psih2;
    real zl = land ? 0.01f : z0;
    real psiq = logf(karman * old_ust * za / xka + za / zl)
              - (option == 91 ? psih : pq);
    real psiq2 = logf(karman * old_ust * 2.0f / xka + 2.0f / zl)
               - (option == 91 ? psih2 : pq2);
    real psiq10 = logf(karman * old_ust * 10.0f / xka + 10.0f / zl)
                - (option == 91 ? psih10 : pq10);

    if (!land) {
        real visc = (1.32f + 0.009f * (temp - 273.15f)) * 1.0e-5f;
        real restar = old_ust * z0 / visc;
        real z0t = fminf(fmaxf(5.5e-5f * powf(restar, -0.60f), 2.0e-9f),
                         1.0e-4f);
        if (option == 91) {
            psiq = fmaxf(logf((za + z0t) / z0t) - psih, 2.0f);
            psit = fmaxf(logf((za + z0t) / z0t) - psih, 2.0f);
            psiq2 = fmaxf(logf((2.0f + z0t) / z0t) - psih2, 2.0f);
            psit2 = fmaxf(logf((2.0f + z0t) / z0t) - psih2, 2.0f);
            psiq10 = fmaxf(logf((10.0f + z0t) / z0t) - psih10, 2.0f);
        } else {
            psih = sf_rev_heat_psi(zol, za, z0t, za);
            psih2 = sf_rev_heat_psi(zol, za, z0t, 2.0f);
            psih10 = sf_rev_heat_psi(zol, za, z0t, 10.0f);
            psit = logf((za + z0t) / z0t) - psih;
            psit2 = logf((2.0f + z0t) / z0t) - psih2;
            psiq = psit; psiq2 = psit2;
            psiq10 = logf((10.0f + z0t) / z0t) - psih10;
        }
    }

    if (isftcflx == 1 && !land) {
        real z0q = 1.0e-4f;
        if (option == 91) {
            psiq = logf(za / z0q) - psih;
            psiq2 = logf(2.0f / z0q) - psih2;
            psiq10 = logf(10.0f / z0q) - psih10;
        } else {
            psih = sf_rev_heat_psi(zol, za, z0q, za);
            psih2 = sf_rev_heat_psi(zol, za, z0q, 2.0f);
            psih10 = sf_rev_heat_psi(zol, za, z0q, 10.0f);
            psiq = logf((za + z0q) / z0q) - psih;
            psiq2 = logf((2.0f + z0q) / z0q) - psih2;
            psiq10 = logf((10.0f + z0q) / z0q) - psih10;
        }
        psit = psiq; psit2 = psiq2;
    } else if (isftcflx == 2 && !land) {
        real visc = (1.32f + 0.009f * (temp - 273.15f)) * 1.0e-5f;
        real restar = old_ust * z0 / visc;
        real gz0t = 0.4f * (7.3f * powf(restar, 0.25f) * sqrtf(0.71f) - 5.0f);
        real gz0q = 0.4f * (7.3f * powf(restar, 0.25f) * sqrtf(0.60f) - 5.0f);
        if (option == 91) {
            psit = gz1 - psih + gz0t; psiq = gz1 - psih + gz0q;
            psit2 = gz2 - psih2 + gz0t; psiq2 = gz2 - psih2 + gz0q;
            psiq10 = gz10 - psih + gz0q;
        } else {
            real z0t = z0 / expf(gz0t), z0q = z0 / expf(gz0q);
            real pht = sf_rev_heat_psi(zol, za, z0t, za);
            real pht2 = sf_rev_heat_psi(zol, za, z0t, 2.0f);
            psit = logf((za + z0t) / z0t) - pht;
            psit2 = logf((2.0f + z0t) / z0t) - pht2;
            psih = sf_rev_heat_psi(zol, za, z0q, za);
            psih2 = sf_rev_heat_psi(zol, za, z0q, 2.0f);
            psih10 = sf_rev_heat_psi(zol, za, z0q, 10.0f);
            psiq = logf((za + z0q) / z0q) - psih;
            psiq2 = logf((2.0f + z0q) / z0q) - psih2;
            psiq10 = logf((10.0f + z0q) / z0q) - psih10;
        }
    }

    real ck = (karman / psix10) * (karman / psiq10);
    real cd = (karman / psix10) * (karman / psix10);
    real cka = (karman / psix) * (karman / psiq);
    real cda = (karman / psix) * (karman / psix);
    if (iz0tlnd >= 1 && land) {
        real visc = (1.32f + 0.009f * (temp - 273.15f)) * 1.0e-5f;
        real restar = old_ust * z0 / visc;
        real czil = iz0tlnd == 1 ? powf(10.0f, -0.40f * z0 / 0.07f) : 0.1f;
        if (option == 91) {
            real add = czil * karman * sqrtf(restar);
            psit = psiq = gz1 - psih + add;
            psit2 = psiq2 = gz2 - psih2 + add;
        } else {
            real z0t = z0 / expf(czil * karman * sqrtf(restar));
            psih = sf_rev_heat_psi(zol, za, z0t, za);
            psih2 = sf_rev_heat_psi(zol, za, z0t, 2.0f);
            psih10 = sf_rev_heat_psi(zol, za, z0t, 10.0f);
            psit = psiq = logf((za + z0t) / z0t) - psih;
            psit2 = psiq2 = logf((2.0f + z0t) / z0t) - psih2;
        }
    }

    real new_ust = 0.5f * old_ust + 0.5f * karman * wspd / psix;
    real u10 = uu * psix10 / psix, v10 = vv * psix10 / psix;
    real th2 = thgb + (thx - thgb) * psit2 / psit;
    real q2 = qs + (qvx - qs) * psiq2 / psiq;
    real t2 = th2 * powf(ps / P0, RCP);
    if (land) new_ust = fmaxf(new_ust, option == 91 ? 0.1f : 0.001f);
    real new_mol = karman * (thx - thgb) / psit;

    real z0out = z0;
    if (isfflx && !land) {
        z0out = fminf(0.0185f * new_ust * new_ust / G
                      + 0.11f * 1.5e-5f / new_ust, 2.85e-3f);
        if (isftcflx != 0) {
            real zw = fminf(powf(new_ust / 1.06f, 0.3f), 1.0f);
            real zn1 = 0.011f * new_ust * new_ust / G + 1.59e-5f;
            real zn2 = 10.0f * expf(-9.5f * powf(new_ust, -0.3333f))
                       + 0.11f * 1.5e-5f / fmaxf(new_ust, 0.01f);
            z0out = fminf(fmaxf((1.0f - zw) * zn1 + zw * zn2, 1.27e-7f),
                           2.85e-3f);
        }
    }

    real flhc = 0.0f, flqc = 0.0f, new_hfx = 0.0f, new_qfx = 0.0f;
    real lh = 0.0f, chs = 0.0f, chs2 = 0.0f, cqs2 = 0.0f;
    if (isfflx) {
        flqc = rho * mavail[idx] * new_ust * karman / psiq;
        if (fabsf(thx - thgb) > 1.0e-5f)
            flhc = cpm * rho * new_ust * new_mol / (thx - thgb);
        new_qfx = flqc * (qs - qvx); lh = XLV * new_qfx;
        new_hfx = flhc * (thgb - thx);
        chs = new_ust * karman / psiq;
        cqs2 = new_ust * karman / psiq2;
        chs2 = new_ust * karman / psit2;
    }

    znt[idx] = z0out; ust[idx] = new_ust; mol[idx] = new_mol;
    hfx[idx] = new_hfx; qfx[idx] = new_qfx; qsfc[idx] = qs;
    zol_o[idx] = zol; regime_o[idx] = regime; psim_o[idx] = psim;
    psih_o[idx] = psih; fm_o[idx] = psix; fh_o[idx] = psit; lh_o[idx] = lh;
    u10_o[idx] = u10; v10_o[idx] = v10; th2_o[idx] = th2; t2_o[idx] = t2;
    q2_o[idx] = q2; chs_o[idx] = chs; chs2_o[idx] = chs2;
    cqs2_o[idx] = cqs2; flhc_o[idx] = flhc; flqc_o[idx] = flqc;
    qgh_o[idx] = qgh; rmol_o[idx] = rmol; wspd_o[idx] = wspd;
    br_o[idx] = br; gz1_o[idx] = gz1; cpm_o[idx] = cpm;
    ck_o[idx] = ck; cka_o[idx] = cka; cd_o[idx] = cd; cda_o[idx] = cda;
}
