// WRF v4.6.1 MYNN surface layer (sf_sfclay_physics=5).
//
// This is the CUDA transcription of module_sf_mynn.F:SFCLAY1D_mynn for the
// pinned option identities: every defined isftcflx over water, with
// iz0tlnd=0, spp_pbl=0 and psi_opt=0.  One thread handles one independent
// horizontal column.  The MYNN surface-layer runtime selector remains
// fail-closed until the PBL kernel and full driver coupling pass their
// separate WRF oracles.

__device__ __forceinline__ real mynn_psim_stable_full(real z)
{
    return -6.1f * logf(z + powf(1.0f + powf(z, 2.5f), 1.0f / 2.5f));
}

__device__ __forceinline__ real mynn_psih_stable_full(real z)
{
    return -5.3f * logf(z + powf(1.0f + powf(z, 1.1f), 1.0f / 1.1f));
}

__device__ __forceinline__ real mynn_psim_unstable_full(real z)
{
    real x = powf(1.0f - 16.0f * z, 0.25f);
    real psimk = 2.0f * logf(0.5f * (1.0f + x))
               + logf(0.5f * (1.0f + x * x))
               - 2.0f * atanf(x) + 2.0f * atanf(1.0f);
    real ym = powf(1.0f - 10.0f * z, 0.33f);
    real rt3 = sqrtf(3.0f);
    real psimc = 1.5f * logf((ym * ym + ym + 1.0f) / 3.0f)
               - rt3 * atanf((2.0f * ym + 1.0f) / rt3)
               + 4.0f * atanf(1.0f) / rt3;
    return (psimk + z * z * psimc) / (1.0f + z * z);
}

__device__ __forceinline__ real mynn_psih_unstable_full(real z)
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

__device__ __forceinline__ real mynn_table(real z, int which)
{
    // which: 0 psim stable, 1 psih stable, 2 psim unstable, 3 psih unstable.
    bool unstable = which >= 2;
    real scaled = (unstable ? -z : z) * 100.0f;
    int index = (int)scaled;
    if (index + 1 > 1000) {
        if (which == 0) return mynn_psim_stable_full(z);
        if (which == 1) return mynn_psih_stable_full(z);
        if (which == 2) return mynn_psim_unstable_full(z);
        return mynn_psih_unstable_full(z);
    }
    real fraction = scaled - (real)index;
    real sign = unstable ? -1.0f : 1.0f;
    real z0 = sign * 0.01f * (real)index;
    real z1 = sign * 0.01f * (real)(index + 1);
    real f0, f1;
    if (which == 0) {
        f0 = mynn_psim_stable_full(z0); f1 = mynn_psim_stable_full(z1);
    } else if (which == 1) {
        f0 = mynn_psih_stable_full(z0); f1 = mynn_psih_stable_full(z1);
    } else if (which == 2) {
        f0 = mynn_psim_unstable_full(z0); f1 = mynn_psim_unstable_full(z1);
    } else {
        f0 = mynn_psih_unstable_full(z0); f1 = mynn_psih_unstable_full(z1);
    }
    return f0 + fraction * (f1 - f0);
}

__device__ __forceinline__ real mynn_psim_stable(real z)
{ return mynn_table(z, 0); }
__device__ __forceinline__ real mynn_psih_stable(real z)
{ return mynn_table(z, 1); }
__device__ __forceinline__ real mynn_psim_unstable(real z)
{ return mynn_table(z, 2); }
__device__ __forceinline__ real mynn_psih_unstable(real z)
{ return mynn_table(z, 3); }

__device__ __forceinline__ real mynn_li_etal_2010(
    real rib, real zaz0, real z0zt)
{
    real zaz02 = fminf(fmaxf(zaz0, 100.0f), 100000.0f);
    real z0zt2 = fminf(fmaxf(z0zt, 0.5f), 100.0f);
    real alfa = logf(zaz02), beta = logf(z0zt2), zl;
    if (rib <= 0.0f) {
        zl = 0.045f * alfa * rib * rib
           + ((0.003f * beta + 0.0059f) * alfa * alfa
              + (-0.0828f * beta + 0.8845f) * alfa
              + (0.1739f * beta * beta - 0.9213f * beta - 0.1057f)) * rib;
        return fminf(fmaxf(zl, -15.0f), 0.0f);
    }
    if (rib <= 0.2f) {
        zl = ((0.5738f * beta - 0.4399f) * alfa
              + (-4.901f * beta + 52.50f)) * rib * rib
           + ((-0.0539f * beta + 1.540f) * alfa
              + (-0.669f * beta - 3.282f)) * rib;
        return fminf(fmaxf(zl, 0.0f), 4.0f);
    }
    zl = (0.7529f * alfa + 14.94f) * rib
       + 0.1569f * alfa - 0.3091f * beta - 1.303f;
    return fminf(fmaxf(zl, 1.0f), 20.0f);
}

__device__ __forceinline__ real mynn_zolrib(
    real ri, real za, real z0, real zt, real logz0, real logzt, real zol1)
{
    if (zol1 * ri < 0.0f) zol1 = 0.0f;
    real zolold = ri < 0.0f ? -99999.0f : 99999.0f;
    real result = ri < 0.0f ? -66666.0f : 66666.0f;
    int iteration = 1;
    while (fabsf(zolold - result) > 0.01f && iteration < 20) {
        zolold = iteration == 1 ? zol1 : result;
        real zol20 = zolold * z0 / za;
        real zol3 = zolold + zol20;
        real zolt = zolold * zt / za;
        real psit2, psix2;
        if (ri < 0.0f) {
            psit2 = fmaxf(logzt
                - (mynn_psih_unstable(zol3) - mynn_psih_unstable(zolt)), 1.0f);
            psix2 = fmaxf(logz0
                - (mynn_psim_unstable(zol3) - mynn_psim_unstable(zol20)), 1.0f);
        } else {
            psit2 = fmaxf(logzt
                - (mynn_psih_stable(zol3) - mynn_psih_stable(zolt)), 1.0f);
            psix2 = fmaxf(logz0
                - (mynn_psim_stable(zol3) - mynn_psim_stable(zol20)), 1.0f);
        }
        result = ri * psix2 * psix2 / psit2;
        ++iteration;
    }
    if (iteration == 20 && fabsf(zolold - result) > 0.01f)
        result = mynn_li_etal_2010(ri, za / z0, z0 / zt);
    return result;
}

__device__ __forceinline__ void mynn_zilitinkevich_land(
    real z0, real restar, real &zt, real &zq)
{
    zt = z0 * expf(-0.4f * 0.085f * sqrtf(restar));
    zt = fminf(zt, 0.75f * z0);
    zq = zt;
}

__device__ __forceinline__ real mynn_charnock_1955(
    real ustar, real wsp, real visc, real zu)
{
    real wsp10 = wsp * logf(10.0f / 1.0e-4f) / logf(zu / 1.0e-4f);
    real czc = 0.011f + 0.007f
        * fminf(fmaxf((wsp10 - 10.0f) / 8.0f, 0.0f), 1.0f);
    real z0 = czc * ustar * ustar / 9.81f
            + 0.11f * visc / fmaxf(ustar, 0.05f);
    return fminf(fmaxf(z0, 1.27e-7f), 2.85e-3f);
}

// module_sf_mynn.F:1281.  ISFTCFLX=1 and 2.  A function of u* alone: the
// Fortran takes no wind speed.
__device__ __forceinline__ real mynn_davis_etal_2008(real ustar)
{
    real zw = fminf(powf(ustar / 1.06f, 0.3f), 1.0f);
    real zn1 = 0.011f * ustar * ustar / 9.81f + 1.59e-5f;
    real zn2 = 10.0f * expf(-9.5f * powf(ustar, -0.3333f))
             + 0.11f * 1.5e-5f / fmaxf(ustar, 0.01f);
    real z0 = (1.0f - zw) * zn1 + zw * zn2;
    return fminf(fmaxf(z0, 1.27e-7f), 2.85e-3f);
}

// module_sf_mynn.F:1311.  ISFTCFLX=3.  ustar is an argument of the Fortran
// subroutine and is never read in its body, so it is not taken here either.
__device__ __forceinline__ real mynn_taylor_yelland_2001(real wsp)
{
    real hs = 0.0248f * powf(wsp, 2.0f);
    real tp = 0.729f * fmaxf(wsp, 0.1f);
    real lp = 9.81f * (tp * tp) / (2.0f * 3.14159265f);
    real z0 = 1200.0f * hs * powf(hs / lp, 4.5f);
    return fminf(fmaxf(z0, 1.27e-7f), 2.85e-3f);
}

// module_sf_mynn.F:1425.  COARE 3.0 zt/zq.  The Ren<=2 test at :1442 selects
// between two arms that compute the identical expression.  Zq is assigned
// from the already-clamped Zt (:1466-1467), so zq == zt.
__device__ __forceinline__ void mynn_fairall_etal_2003(
    real restar, real &zt, real &zq)
{
    zt = 5.5e-5f * powf(restar, -0.60f);
    zt = fminf(fmaxf(zt, 2.0e-9f), 1.0e-4f);
    zq = zt;
}

// module_sf_mynn.F:1392.  ISFTCFLX=2 zt/zq.  This leaf makes its own
// landsea-1.5 .GT. 0 test, which is not the .GE. 0 at :625 that selected the
// water roughness, so a column at exactly XLAND=1.5 arrives here and takes
// the LAND arm.
__device__ __forceinline__ void mynn_garratt_1992(
    real z0, real restar, real xland, real &zt, real &zq)
{
    if (xland - 1.5f > 0.0f) {
        real quarter = powf(restar, 0.25f);
        zt = z0 * expf(2.0f - 2.48f * quarter);
        zq = z0 * expf(2.0f - 2.28f * quarter);
        zq = fmaxf(fminf(zq, 5.5e-5f), 2.0e-9f);
        zt = fmaxf(fminf(zt, 5.5e-5f), 2.0e-9f);
    } else {
        zq = z0 / powf(2.71828183f, 2.0f);
        zt = zq;
    }
}

// module_sf_mynn.F:631-710.  ISFTCFLX picks the over-water z0 and then, from
// the restar that new z0 implies (:675), the zt/zq pair.  COARE_OPT is a REAL
// PARAMETER fixed at 3.0 (:85), so edson_etal_2013 and fairall_etal_2014 are
// unreachable through SFCLAY1D_mynn and are deliberately absent from this
// kernel: they exist in the CPU reference only because the leaf oracle can
// call them at their own entry points.  ISFTCFLX=4 sets a z0 and then leaves
// z_t/z_q unassigned (:680-702 has no arm for it), so the host rejects it
// before launch and this switch has no arm for it either.
__device__ __forceinline__ void mynn_water_roughness(
    int isftcflx, real ustar, real wsp, real visc, real za, real xland,
    real &z0, real &restar, real &zt, real &zq)
{
    if (isftcflx == 1 || isftcflx == 2) z0 = mynn_davis_etal_2008(ustar);
    else if (isftcflx == 3) z0 = mynn_taylor_yelland_2001(wsp);
    else z0 = mynn_charnock_1955(ustar, wsp, visc, za);
    restar = fmaxf(ustar * z0 / visc, 0.1f);
    if (isftcflx == 2) mynn_garratt_1992(z0, restar, xland, zt, zq);
    else mynn_fairall_etal_2003(restar, zt, zq);
}

__device__ __forceinline__ void mynn_andreas_snow(
    real visc, real ustar, real &zt, real &zq)
{
    real zntsno = 0.135f * visc / ustar
        + 0.035f * ustar * ustar / 9.8f
        * (5.0f * expf(-powf((ustar - 0.18f) / 0.1f, 2.0f)) + 1.0f);
    real ren = fminf(ustar * zntsno / visc, 1000.0f);
    real log_ren = logf(ren);
    real bt0, bt1, bt2, bq0, bq1, bq2;
    if (ren <= 0.135f) {
        bt0 = 1.25f; bt1 = 0.0f; bt2 = 0.0f;
        bq0 = 1.61f; bq1 = 0.0f; bq2 = 0.0f;
    } else if (ren < 2.5f) {
        bt0 = 0.149f; bt1 = -0.55f; bt2 = 0.0f;
        bq0 = 0.351f; bq1 = -0.628f; bq2 = 0.0f;
    } else {
        bt0 = 0.317f; bt1 = -0.565f; bt2 = -0.183f;
        bq0 = 0.396f; bq1 = -0.512f; bq2 = -0.180f;
    }
    zt = zntsno * expf(bt0 + bt1 * log_ren + bt2 * log_ren * log_ren);
    zq = zntsno * expf(bq0 + bq1 * log_ren + bq2 * log_ren * log_ren);
}

extern "C" __global__
void mynn_surface_column(
    const real* __restrict__ u1_a, const real* __restrict__ v1_a,
    const real* __restrict__ t1_a, const real* __restrict__ qv1_a,
    const real* __restrict__ p1_a, const real* __restrict__ rho1_a,
    const real* __restrict__ dz1_a, const real* __restrict__ u2_a,
    const real* __restrict__ v2_a, const real* __restrict__ dz2_a,
    const real* __restrict__ psfc_a, const real* __restrict__ tsk_a,
    const real* __restrict__ pblh_a, const real* __restrict__ mavail_a,
    const real* __restrict__ hfx_a, const real* __restrict__ qfx_a,
    const real* __restrict__ znt_a, const real* __restrict__ qsfc_a,
    const real* __restrict__ ust_a, const real* __restrict__ xland_a,
    const real* __restrict__ snowh_a, const real* __restrict__ mol_a,
    const real* __restrict__ ustm_a,
    real* __restrict__ regime_o, real* __restrict__ zol_o,
    real* __restrict__ rmol_o, real* __restrict__ ust_o,
    real* __restrict__ ustm_o, real* __restrict__ mol_o,
    real* __restrict__ psim_o, real* __restrict__ psih_o,
    real* __restrict__ chs_o, real* __restrict__ chs2_o,
    real* __restrict__ cqs2_o, real* __restrict__ ch_o,
    real* __restrict__ flhc_o, real* __restrict__ flqc_o,
    real* __restrict__ qgh_o, real* __restrict__ qsfc_o,
    real* __restrict__ hfx_o, real* __restrict__ qfx_o,
    real* __restrict__ lh_o, real* __restrict__ u10_o,
    real* __restrict__ v10_o, real* __restrict__ th2_o,
    real* __restrict__ t2_o, real* __restrict__ q2_o,
    real* __restrict__ gz1oz0_o, real* __restrict__ wspd_o,
    real* __restrict__ br_o, real* __restrict__ ck_o,
    real* __restrict__ cka_o, real* __restrict__ cd_o,
    real* __restrict__ cda_o, real* __restrict__ wstar_o,
    real* __restrict__ qstar_o, real* __restrict__ cpm_o,
    real* __restrict__ znt_o,
    real dx, int itimestep, int isfflx, int isftcflx, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const real cp = 1004.5f, grav = 9.81f, r = 287.0f, rv = 461.6f;
    const real rovcp = r / cp, xlv = 2.5e6f;
    const real svp1 = 0.6112f, svp2 = 17.67f, svp3 = 29.65f;
    const real svpt0 = 273.15f, ep1 = rv / r - 1.0f;
    const real ep2 = r / rv, ep3 = 1.0f - ep2;
    const real karman = 0.4f, prt = 1.0f, wmin = 0.1f, vconvc = 1.25f;

    real u1 = u1_a[idx], v1 = v1_a[idx], t1 = t1_a[idx];
    real qv1 = qv1_a[idx], p1 = p1_a[idx], rho1 = rho1_a[idx];
    real dz1 = dz1_a[idx], u2 = u2_a[idx], v2 = v2_a[idx];
    real dz2 = dz2_a[idx], psfcpa = psfc_a[idx], tsk = tsk_a[idx];
    real pblh = pblh_a[idx], mavail = mavail_a[idx];
    real hfx = hfx_a[idx], qfx = qfx_a[idx], z0 = znt_a[idx];
    real qsfc = qsfc_a[idx], ust = ust_a[idx], xland = xland_a[idx];
    real snowh = snowh_a[idx], mol = mol_a[idx], ustm = ustm_a[idx];

    real psfc = psfcpa / 1000.0f;
    real thgb = tsk * powf(100.0f / psfc, rovcp);
    real pl = p1 / 1000.0f;
    real th1 = t1 * powf(100.0f / pl, rovcp);
    real tc1 = t1 - 273.15f;
    real qvsh = qv1 / (1.0f + qv1);
    // See the THVGB comment below: TVCON/THV1D are pinned to unfused
    // round-to-nearest so they contract identically to their surface twins.
    real tvcon = __fadd_rn(1.0f, __fmul_rn(ep1, qvsh));
    real thv1 = __fmul_rn(th1, tvcon);
    real za = 0.5f * dz1;
    real za2 = dz1 + 0.5f * dz2;
    real govrth = grav / th1;

    real e1;
    if (tsk < 273.15f) {
        e1 = svp1 * expf(4648.0f * (1.0f / 273.15f - 1.0f / tsk)
             - 11.64f * logf(273.15f / tsk) + 0.02265f * (273.15f - tsk));
    } else {
        e1 = svp1 * expf(svp2 * (tsk - svpt0) / (tsk - svp3));
    }
    // module_sf_mynn.F:532 tests the INCOMING QSFC: a land column whose LSM
    // never set QSFC is recomputed from saturation here.
    real qsfcmr;
    if (xland > 1.5f || qsfc <= 0.0f) {
        qsfc = ep2 * e1 / (psfc - ep3 * e1);
        qsfcmr = ep2 * e1 / (psfc - e1);
    } else {
        qsfcmr = qsfc / (1.0f - qsfc);
    }

    if (tsk < 273.15f) {
        e1 = svp1 * expf(4648.0f * (1.0f / 273.15f - 1.0f / t1)
             - 11.64f * logf(273.15f / t1) + 0.02265f * (273.15f - t1));
    } else {
        e1 = svp1 * expf(svp2 * (t1 - svpt0) / (t1 - svp3));
    }
    real qgh = ep2 * e1 / (pl - e1);
    real cpm = cp * (1.0f + 0.84f * qv1);

    // module_sf_mynn.F:556 -- WSPD is SQRT(u*u+v*v), not hypot().
    real wsp = sqrtf(u1 * u1 + v1 * v1);
    // THV1D (:513) and THVGB (:559) are the same product of a temperature and
    // (1+EP1*q).  gfortran evaluates both without contraction, so an exactly
    // saturated neutral column cancels to zero.  NVRTC is free to contract one
    // and not the other; pin both to unfused round-to-nearest so the identity
    // holds for the same reason it holds in WRF instead of by special case.
    real tvcon_gb = __fadd_rn(1.0f, __fmul_rn(ep1, qsfc));
    real thvgb = __fmul_rn(thgb, tvcon_gb);
    real dthvdz = thv1 - thvgb;
    real fluxc = fmaxf(hfx / rho1 / cp + ep1 * thvgb * qfx / rho1, 0.0f);
    // module_sf_mynn.F:573 spells the :532 predicate a second time, but it runs
    // after the :533 saturation update, so a land column that entered with
    // QSFC<=0 now takes the LAND height scale.
    bool wstar_water = xland > 1.5f || qsfc <= 0.0f;
    real height = wstar_water ? pblh : fminf(1.5f * pblh, 4000.0f);
    real wstar = vconvc * powf(grav / tsk * height * fluxc, 0.33f);
    real vsgd = 0.32f * powf(fmaxf(dx / 5000.0f - 1.0f, 0.0f), 0.33f);
    wsp = fmaxf(sqrtf(wsp * wsp + wstar * wstar + vsgd * vsgd), wmin);
    real br = govrth * za * dthvdz / (wsp * wsp);
    real limit = itimestep == 1 ? 2.0f : 4.0f;
    br = fminf(fmaxf(br, -limit), limit);

    real visc = 1.326e-5f * (1.0f + 6.542e-3f * tc1
              + 8.301e-6f * tc1 * tc1 - 4.84e-9f * tc1 * tc1 * tc1);
    real restar, zt, zq;
    if (xland >= 1.5f) {
        mynn_water_roughness(isftcflx, ust, wsp, visc, za, xland,
                             z0, restar, zt, zq);
    } else {
        restar = fmaxf(ust * z0 / visc, 0.1f);
        if (snowh >= 0.1f) mynn_andreas_snow(visc, ust, zt, zq);
        else mynn_zilitinkevich_land(z0, restar, zt, zq);
    }

    real zratio = z0 / zt;
    real gz1oz0 = logf((za + z0) / z0);
    real gz1ozt = logf((za + z0) / zt);
    real gz2ozt = logf((2.0f + z0) / zt);
    real gz10oz0 = logf((10.0f + z0) / z0);
    real gz10ozt = logf((10.0f + z0) / zt);

    real regime, zol, psim, psih, psim10, psih10, psih2;
    if (br > 0.0f) {
        regime = br > 0.2f ? 1.0f : 2.0f;
        if (itimestep <= 1) zol = mynn_li_etal_2010(br, za / z0, zratio);
        else {
            zol = za * karman * grav * mol / (th1 * fmaxf(ust * ust, 0.0001f));
            zol = fminf(fmaxf(zol, 0.0f), 20.0f);
        }
        zol = fminf(fmaxf(mynn_zolrib(br, za, z0, zt, gz1oz0, gz1ozt, zol),
                           0.0f), 20.0f);
        real zolzt = zol * zt / za, zolz0 = zol * z0 / za;
        real zolza = zol * (za + z0) / za;
        real zol10 = zol * (10.0f + z0) / za;
        real zol2 = zol * (2.0f + z0) / za;
        psim = mynn_psim_stable(zolza) - mynn_psim_stable(zolz0);
        psih = mynn_psih_stable(zolza) - mynn_psih_stable(zolzt);
        psim10 = mynn_psim_stable(zol10) - mynn_psim_stable(zolz0);
        psih10 = mynn_psih_stable(zol10) - mynn_psih_stable(zolz0);
        psih2 = mynn_psih_stable(zol2) - mynn_psih_stable(zolz0);
    } else if (br == 0.0f) {
        regime = 3.0f; zol = 0.0f;
        psim = 0.0f; psih = 0.0f; psim10 = 0.0f; psih10 = 0.0f; psih2 = 0.0f;
    } else {
        regime = 4.0f;
        if (itimestep <= 1) zol = mynn_li_etal_2010(br, za / z0, zratio);
        else {
            zol = za * karman * grav * mol / (th1 * fmaxf(ust * ust, 0.001f));
            zol = fminf(fmaxf(zol, -20.0f), 0.0f);
        }
        zol = fminf(fmaxf(mynn_zolrib(br, za, z0, zt, gz1oz0, gz1ozt, zol),
                           -20.0f), 0.0f);
        real zolzt = zol * zt / za, zolz0 = zol * z0 / za;
        real zolza = zol * (za + z0) / za;
        real zol10 = zol * (10.0f + z0) / za;
        real zol2 = zol * (2.0f + z0) / za;
        psim = mynn_psim_unstable(zolza) - mynn_psim_unstable(zolz0);
        psih = mynn_psih_unstable(zolza) - mynn_psih_unstable(zolzt);
        psim10 = mynn_psim_unstable(zol10) - mynn_psim_unstable(zolz0);
        psih10 = mynn_psih_unstable(zol10) - mynn_psih_unstable(zolz0);
        psih2 = mynn_psih_unstable(zol2) - mynn_psih_unstable(zolz0);
        psih = fminf(psih, 0.9f * gz1ozt);
        psim = fminf(psim, 0.9f * gz1oz0);
        psih2 = fminf(psih2, 0.9f * gz2ozt);
        psim10 = fminf(psim10, 0.9f * gz10oz0);
        psih10 = fminf(psih10, 0.9f * gz10ozt);
    }
    real rmol = zol / za;

    real psix = gz1oz0 - psim;
    real psix10 = gz10oz0 - psim10;
    ust = 0.5f * ust + 0.5f * karman * wsp / psix;
    // module_sf_mynn.F:953 -- WSPDI is SQRT(u*u+v*v), not hypot().
    real wspdi = fmaxf(sqrtf(u1 * u1 + v1 * v1), wmin);
    ustm = 0.5f * ustm + 0.5f * karman * wspdi / psix;
    if (xland < 1.5f) {
        ust = fmaxf(ust, 0.005f);
        ustm = ust;
    }

    gz1ozt = logf((za + z0) / zt);
    gz2ozt = logf((2.0f + z0) / zt);
    real psit = fmaxf(gz1ozt - psih, 1.0f);
    real psit2 = fmaxf(gz2ozt - psih2, 1.0f);
    real psiq = fmaxf(logf((za + z0) / zq) - psih, 1.0f);
    real psiq2 = fmaxf(logf((2.0f + z0) / zq) - psih2, 1.0f);
    real psiq10 = fmaxf(logf((10.0f + z0) / zq) - psih10, 1.0f);
    mol = karman * (thv1 - thvgb) / psit / prt;
    real qstar = karman * (qvsh - qsfc) * 1000.0f / psiq / prt;

    // WRF deliberately recomputes moisture resistance with zq in numerator.
    psiq = fmaxf(logf((za + zq) / zq) - psih, 1.0f);
    psiq2 = fmaxf(logf((2.0f + zq) / zq) - psih2, 1.0f);
    psiq10 = fmaxf(logf((10.0f + zq) / zq) - psih10, 1.0f);
    real flhc, flqc, lh, chs, ch, cqs2, chs2, ck, cka, cd, cda;
    if (isfflx < 1) {
        qfx = 0.0f; hfx = 0.0f; flhc = 0.0f; flqc = 0.0f; lh = 0.0f;
        chs = 0.0f; ch = 0.0f; chs2 = 0.0f; cqs2 = 0.0f;
        ck = 0.0f; cka = 0.0f; cd = 0.0f; cda = 0.0f;
    } else {
        flqc = rho1 * mavail * ust * karman / psiq;
        flhc = rho1 * cpm * ust * karman / psit;
        qfx = fmaxf(flqc * (qsfcmr - qv1), -0.02f);
        lh = xlv * qfx;
        // module_sf_mynn.F:1065-1076.  The arms test XLAND-1.5 for .GT. and
        // .LT. and nothing else, so a column at exactly XLAND=1.5 takes
        // neither and HFX keeps the value it entered with.
        if (xland > 1.5f) {
            hfx = flhc * (thgb - th1);
            // :1067-1072, the AHW dissipative-heating term -- the third and
            // last thing ISFTCFLX selects, water-only, and keyed on
            // ISFTCFLX.NE.0, so every non-default roughness identity adds it.
            // USTM is the u* computed without the vconv/vsgd gust (:953-956).
            if (isftcflx != 0) hfx = hfx + rho1 * ustm * ustm * wspdi;
        } else if (xland < 1.5f) {
            hfx = flhc * (thgb - th1);
            hfx = fmaxf(hfx, -250.0f);
        }
        chs = ust * karman / psit;
        ch = flhc / (cpm * rho1);
        cqs2 = ust * karman / psiq2;
        chs2 = ust * karman / psit2;
        ck = (karman / psix10) * (karman / psiq10);
        cd = (karman / psix10) * (karman / psix10);
        cka = (karman / psix) * (karman / psiq);
        cda = (karman / psix) * (karman / psix);
    }

    real u10, v10;
    if (za <= 7.0f) {
        if (za2 > 7.0f && za2 < 13.0f) { u10 = u2; v10 = v2; }
        else {
            real ratio = logf(10.0f / z0) / logf(za / z0);
            u10 = u1 * ratio; v10 = v1 * ratio;
        }
    } else if (za < 13.0f) {
        real ratio = logf(10.0f / z0) / logf(za / z0);
        u10 = u1 * ratio; v10 = v1 * ratio;
    } else {
        u10 = u1 * psix10 / psix; v10 = v1 * psix10 / psix;
    }

    real th2 = thgb + (th1 - thgb) * psit2 / psit;
    if ((th1 > thgb && !(th2 >= thgb && th2 <= th1))
        || (th1 < thgb && !(th2 >= th1 && th2 <= thgb)))
        th2 = thgb + 2.0f * (th1 - thgb) / za;
    real t2 = th2 * powf(psfc / 100.0f, rovcp);
    real q2 = qsfcmr + (qv1 - qsfcmr) * psiq2 / psiq;
    q2 = fmaxf(q2, fminf(qsfcmr, qv1));
    q2 = fminf(q2, 1.05f * qv1);

    regime_o[idx] = regime; zol_o[idx] = zol; rmol_o[idx] = rmol;
    ust_o[idx] = ust; ustm_o[idx] = ustm; mol_o[idx] = mol;
    psim_o[idx] = psim; psih_o[idx] = psih; chs_o[idx] = chs;
    chs2_o[idx] = chs2; cqs2_o[idx] = cqs2; ch_o[idx] = ch;
    flhc_o[idx] = flhc; flqc_o[idx] = flqc; qgh_o[idx] = qgh;
    qsfc_o[idx] = qsfc; hfx_o[idx] = hfx; qfx_o[idx] = qfx;
    lh_o[idx] = lh; u10_o[idx] = u10; v10_o[idx] = v10;
    th2_o[idx] = th2; t2_o[idx] = t2; q2_o[idx] = q2;
    gz1oz0_o[idx] = gz1oz0; wspd_o[idx] = wsp; br_o[idx] = br;
    ck_o[idx] = ck; cka_o[idx] = cka; cd_o[idx] = cd; cda_o[idx] = cda;
    wstar_o[idx] = wstar; qstar_o[idx] = qstar; cpm_o[idx] = cpm;
    // module_sf_mynn.F:436 declares ZNT INTENT(INOUT); :657 (charnock_1955)
    // mutates it in place over water and the updated value persists into the
    // next step's ZNTstoch/restar/z_t/z_q chain.
    znt_o[idx] = z0;
}
