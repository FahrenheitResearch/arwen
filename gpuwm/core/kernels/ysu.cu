// YSU planetary-boundary-layer column scheme.
//
// Transcribed from WRF v4.6.1 phys/module_bl_ysu.F and the scheme body in
// phys/physics_mmm/bl_ysu.F90 (bl_ysu_run), for the non-BEP,
// non-BEP path, including v4.6.1 ysu_topdown_pblmix.  One CUDA thread owns a
// complete surface-to-top column.
// All implicit heat/moisture/momentum systems use the same in-thread Thomas
// pattern as kernels/acoustic.cu.  Float64 verification authority:
// gpuwm.verify.npref.np_ysu_column.

#define YSU_KMAX 128

__device__ __forceinline__ real ysu_min(real a, real b) { return a < b ? a : b; }
__device__ __forceinline__ real ysu_max(real a, real b) { return a > b ? a : b; }
__device__ __forceinline__ real ysu_clip(real x, real lo, real hi) {
    return ysu_min(ysu_max(x, lo), hi);
}

__device__ void ysu_thomas(const real *lower, const real *diag,
                           const real *upper, real *rhs, real *gamma, int nz) {
    real inv = 1.0f / diag[0];
    gamma[0] = upper[0] * inv;
    rhs[0] *= inv;
    for (int k = 1; k < nz - 1; ++k) {
        inv = 1.0f / (diag[k] - lower[k] * gamma[k - 1]);
        gamma[k] = upper[k] * inv;
        rhs[k] = (rhs[k] - lower[k] * rhs[k - 1]) * inv;
    }
    inv = 1.0f / (diag[nz - 1] - lower[nz - 1] * gamma[nz - 2]);
    rhs[nz - 1] = (rhs[nz - 1] - lower[nz - 1] * rhs[nz - 2]) * inv;
    for (int k = nz - 2; k >= 0; --k)
        rhs[k] -= gamma[k] * rhs[k + 1];
}

__device__ void ysu_diagnose(const real *u, const real *v, const real *thv,
                             const real *za, real thermal, real br0,
                             real brcrit, int nz, int st, int col,
                             real *hpbl, int *kpbl,
                             real *brdn_out, real *brup_out) {
    real brup = br0, brdn = brup;
    int kp = 1;
    bool crossed = false;
    for (int k = 1; k < nz; ++k) {
        if (!crossed) {
            brdn = brup;
            int q = k * st + col;
            real spdk2 = ysu_max(u[q] * u[q] + v[q] * v[q], 1.0f);
            brup = (thv[k] - thermal) * (G * za[k] / thv[0]) / spdk2;
            kp = k + 1;
            crossed = brup > brcrit;
        }
    }
    real frac;
    if (brdn >= brcrit) frac = 0.0f;
    else if (brup <= brcrit) frac = 1.0f;
    else frac = (brcrit - brdn) / (brup - brdn);
    int kh = kp - 1;
    *hpbl = za[kh - 1] + frac * (za[kh] - za[kh - 1]);
    *kpbl = kp;
    *brdn_out = brdn;
    *brup_out = brup;
}

extern "C" __global__
void ysu_column(const real *u, const real *v, const real *theta,
                const real *qv, const real *qc, const real *qi,
                const real *p, const real *p_interface, const real *exner,
                const real *dz, const real *rthraten,
                const real *psfc, const real *znt,
                const real *ust, const real *hfx, const real *qfx,
                const real *wspd, const real *br, const real *psim,
                const real *psih, const real *xland, const real *u10,
                const real *v10, real *du, real *dv, real *dtheta,
                real *dqv, real *dqc, real *dqi, real *hpbl_out,
                int *kpbl_out, real *exch_h, real *exch_m,
                real *wstar_out, real *delta_out,
                real dt, real *topdown_radsum_out,
                real *wstar3_2_out, int *cloudflg_out,
                int ysu_topdown_pblmix, int nz, int ny, int nx) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int st = ny * nx;
    if (col >= st || nz > YSU_KMAX) return;

    real thv[YSU_KMAX], thli[YSU_KMAX];
    real zq[YSU_KMAX + 1], za[YSU_KMAX];
    real dza[YSU_KMAX], delp[YSU_KMAX];
    real xkzm[YSU_KMAX], xkzh[YSU_KMAX], xkzq[YSU_KMAX];
    real xkzml[YSU_KMAX], xkzhl[YSU_KMAX];
    real zfacent[YSU_KMAX], entfac[YSU_KMAX];
    real lower[YSU_KMAX], diag[YSU_KMAX], upper[YSU_KMAX];
    real rhs[YSU_KMAX], gamma[YSU_KMAX];

    real us = ust[col], hf = hfx[col], qf = qfx[col];
    if (us == 0.0f && hf == 0.0f && qf == 0.0f) {
        for (int k = 0; k < nz; ++k) {
            int q = k * st + col;
            du[q] = dv[q] = dtheta[q] = dqv[q] = dqc[q] = dqi[q] = 0.0f;
            exch_h[q] = exch_m[q] = 0.0f;
        }
        hpbl_out[col] = dz[col];
        kpbl_out[col] = 1;
        wstar_out[col] = delta_out[col] = 0.0f;
        topdown_radsum_out[col] = wstar3_2_out[col] = 0.0f;
        cloudflg_out[col] = 0;
        return;
    }

    const real xkzminm = 0.1f, xkzminh = 0.01f, xkzmax = 1000.0f;
    const real rimin = -100.0f, rlam = 30.0f, prmin = 0.25f, prmax = 4.0f;
    const real brcr_ub = 0.0f, brcr_sb = 0.25f, cori = 1.0e-4f;
    const real afac = 6.8f, bfac = 6.8f, phifac = 8.0f, sfcfrac = 0.1f;
    const real d1 = 0.02f, d2 = 0.05f, d3 = 0.001f;
    const real h1 = 0.333333333333f, h2 = 0.666666666667f;
    const real zfmin = 1.0e-8f, aphi5 = 5.0f, aphi16 = 16.0f;
    const real tmin = 0.01f, gamcrt = 3.0f, gamcrq = 0.002f;
    // ep1 is WRF's EP_1, and WRF forms it in float32: module_model_constants
    // declares `REAL, PARAMETER :: EP_1 = R_v/R_d - 1.` so the quotient is a
    // float32 divide of two float32 values.  CUDA_DEFINES["RVOVRD"] is RV/RD
    // computed in Python doubles and rounded once to float32, which lands 1
    // ULP below WRF's quotient at 1.608 and therefore 2 ULP below it here at
    // 0.608.  Spelling the divide keeps the whole thv/Richardson/PBL-diagnosis
    // chain on WRF's constant; measured worth on the oracle fixture: hpbl
    // 112 -> 1 ULP, exch_h 283 -> 7, exch_m 48 -> 7, dv 46604 -> 23302.
    const real karman = 0.4f, ep1 = RV / RD - 1.0f;
    real dt2 = 2.0f * dt, rdt = 1.0f / dt2;

    zq[0] = 0.0f;
    for (int k = 0; k < nz; ++k) {
        int q = k * st + col;
        thv[k] = theta[q] * (1.0f + ep1 * qv[q]);
        thli[k] = (theta[q] * exner[q] - XLV * qc[q] / CP
                   - 2.834e6f * qi[q] / CP) / exner[q];
        zq[k + 1] = zq[k] + dz[q];
        delp[k] = p_interface[k * st + col] - p_interface[(k + 1) * st + col];
    }
    for (int k = 0; k < nz; ++k) za[k] = 0.5f * (zq[k] + zq[k + 1]);
    dza[0] = za[0];
    for (int k = 1; k < nz; ++k) dza[k] = za[k] - za[k - 1];

    real theta0 = theta[col], qv0 = qv[col];
    real temp0 = theta0 * exner[col];
    real rho = psfc[col] / (RD * temp0 * (1.0f + ep1 * qv0));
    real govrth = G / theta0;
    real u0 = u[col], v0 = v[col];
    real wspd1 = sqrtf(u0 * u0 + v0 * v0) + 1.0e-9f;
    real sflux = hf / rho / CP + qf / rho * ep1 * theta0;
    real thermal = thv[0], thermalli = thli[0];
    real hpbl;
    int kpbl;
    real brdn, brup;
    ysu_diagnose(u, v, thv, za, thermal, br[col], brcr_ub, nz, st, col,
                 &hpbl, &kpbl, &brdn, &brup);
    if (hpbl < zq[1]) kpbl = 1;
    bool pblflg = kpbl > 1;
    bool sfcflg = br[col] <= 0.0f;
    real zol1 = ysu_max(br[col] * psim[col] * psim[col] / psih[col], rimin);
    zol1 = sfcflg ? ysu_min(zol1, -zfmin) : ysu_max(zol1, zfmin);
    real hol1 = zol1 * hpbl / za[0] * sfcfrac;
    real phim, phih, wstar3, wstar;
    if (sfcflg) {
        phim = powf(1.0f - aphi16 * hol1, -0.25f);
        phih = powf(1.0f - aphi16 * hol1, -0.5f);
        wstar3 = govrth * ysu_max(sflux, 0.0f) * hpbl;
        wstar = cbrtf(ysu_max(wstar3, 0.0f));
    } else {
        phim = phih = 1.0f + aphi5 * hol1;
        wstar3 = wstar = 0.0f;
    }
    real ust3 = us * us * us;
    real wscale = cbrtf(ysu_max(ust3 + phifac * karman * wstar3 * 0.5f, 0.0f));
    wscale = ysu_min(wscale, us * aphi16);
    wscale = ysu_max(wscale, us / aphi5);
    real hgamt = 0.0f, hgamq = 0.0f, hgamu = 0.0f, hgamv = 0.0f;
    if (sfcflg && sflux > 0.0f) {
        real gamfac = bfac / rho / wscale;
        hgamt = ysu_min(gamfac * hf / CP, gamcrt);
        hgamq = ysu_min(gamfac * qf, gamcrq);
        real vpert = (hgamt + ep1 * theta0 * hgamq) / bfac * afac;
        thermal += ysu_max(vpert, 0.0f)
                 * ysu_min(za[0] / (sfcfrac * hpbl), 1.0f);
        thermalli += ysu_max(vpert, 0.0f)
                   * ysu_min(za[0] / (sfcfrac * hpbl), 1.0f);
        hgamt = ysu_max(hgamt, 0.0f);
        hgamq = ysu_max(hgamq, 0.0f);
        real cg = -15.9f * us * us / ysu_max(wspd[col], 1.0e-9f) * wstar3
                / ysu_max(wscale * wscale * wscale * wscale, 1.0e-20f);
        hgamu = cg * u0;
        hgamv = cg * v0;
        ysu_diagnose(u, v, thv, za, thermal, br[col], brcr_ub, nz, st, col,
                     &hpbl, &kpbl, &brdn, &brup);
        if (hpbl < zq[1]) kpbl = 1;
        pblflg = kpbl > 1;
    } else pblflg = false;

    // WRF v4.6.1 bl_ysu.F90:732-768 runs the theta-li extension for every
    // column.  In-cloud liquid loading can therefore revive pblflg even
    // when the surface is stable and wstar3 remains zero.
    if (ysu_topdown_pblmix) {
        bool definebrup = false;
        int kpblold = kpbl;
        for (int fk = kpblold; fk <= nz - 1; ++fk) {
            int k = fk - 1;
            int q = k * st + col;
            real spdk2 = ysu_max(u[q] * u[q] + v[q] * v[q], 1.0f);
            real bruptmp = (thli[k] - thermalli)
                           * (G * za[k] / thli[0]) / spdk2;
            bool stable_li = bruptmp >= brcr_ub;
            if (definebrup) {
                kpbl = fk;
                brup = bruptmp;
                definebrup = false;
            }
            if (!stable_li) {
                brdn = bruptmp;
                definebrup = true;
                pblflg = true;
            }
        }
    }
    if (pblflg) {
        int kh = kpbl - 1;
        real frac;
        if (brdn >= brcr_ub) frac = 0.0f;
        else if (brup <= brcr_ub) frac = 1.0f;
        else frac = (brcr_ub - brdn) / (brup - brdn);
        hpbl = za[kh - 1] + frac * (za[kh] - za[kh - 1]);
        if (hpbl < zq[1]) {
            kpbl = 1;
            pblflg = false;
        }
    }

    if (!sfcflg && hpbl < zq[1]) {
        real brcrit = brcr_sb;
        if (xland[col] >= 1.5f) {
            real ross = ysu_max(hypotf(u10[col], v10[col]), 1.0e-9f)
                      / (cori * znt[col]);
            brcrit = ysu_min(0.16f * powf(1.0e-7f * ross, -0.18f), 0.3f);
        }
        ysu_diagnose(u, v, thv, za, thermal, br[col], brcrit, nz, st, col,
                     &hpbl, &kpbl, &brdn, &brup);
        if (hpbl < zq[1]) kpbl = 1;
        // Keep pblflg false: WRF uses kpbl for the stable K profile but
        // reserves countergradient and entrainment terms for convection.
    }

    real wm2 = 0.0f, we = 0.0f, hfxpbl = 0.0f, qfxpbl = 0.0f;
    real ufxpbl = 0.0f, vfxpbl = 0.0f, delta = 0.0f;
    real wstar3_2 = 0.0f, topdown_radsum = 0.0f;
    bool cloudflg = false;
    if (pblflg) {
        int kt = kpbl - 2;
        real wm3 = wstar3 + 5.0f * ust3;
        wm2 = powf(ysu_max(wm3, 0.0f), h2);
        real bfxpbl = -0.15f * thv[0] / G * wm3 / hpbl;
        real dthv = ysu_max(thv[kt + 1] - thv[kt], tmin);
        we = ysu_max(bfxpbl / dthv, -sqrtf(wm2));
        // F90 839-897.  The kpbl<nz guard makes the source's k+2 access
        // explicit: a top-level PBL has no above-cloud comparison layer.
        if (ysu_topdown_pblmix && kpbl < nz
                && qc[kt * st + col] + qi[kt * st + col] > 1.0e-5f) {
            cloudflg = true;
            real ptop = p_interface[(kt + 1) * st + col];
            real templ = thli[kt] * powf(ptop / 100000.0f, RCP);
            real rvls = 100.0f * 6.112f
                      * expf(17.67f * (templ - 273.16f)
                             / (templ - 29.65f)) * (EP2 / ptop);
            real qsum = qv[kt * st + col] + qc[kt * st + col];
            real temps = templ + (qsum - rvls)
                       / (CP / XLV + EP2 * XLV * rvls
                          / (RD * templ * templ));
            rvls = 100.0f * 6.112f
                 * expf(17.67f * (temps - 273.15f)
                        / (temps - 29.65f)) * (EP2 / ptop);
            real rcldb = ysu_max(qsum - rvls, 0.0f);
            int kabove = kt + 2;
            real dthv_cloud = (thli[kabove]
                    + theta[kabove * st + col] * ep1
                    * (qv[kabove * st + col] + qc[kabove * st + col]))
                    - (thli[kt] + theta[kt * st + col] * ep1 * qsum);
            dthv_cloud = ysu_max(dthv_cloud, 0.1f);
            real tmp1 = XLV / CP * rcldb
                      / (exner[kt * st + col] * dthv_cloud);
            real ent_eff = 0.2f * 8.0f * tmp1 + 0.2f;
            for (int kk = 0; kk <= kt; ++kk) {
                real radflux = rthraten[kk * st + col]
                             * exner[kk * st + col];
                radflux *= CP / G
                         * (p_interface[kk * st + col]
                            - p_interface[(kk + 1) * st + col]);
                if (radflux < 0.0f) topdown_radsum += fabsf(radflux);
            }
            topdown_radsum = ysu_max(topdown_radsum, 0.0f);
            real tk = theta[kt * st + col] * exner[kt * st + col];
            real rho2 = p[kt * st + col]
                      / (RD * tk * (1.0f + ep1 * qv[kt * st + col]));

            // Preserve the two overwritten bfx0 assignments in v4.6.1:
            // the executable values are max(sflux,0) and radsum/rho/cp.
            real bfx0 = ysu_max(sflux, 0.0f);
            wm3 = govrth * bfx0 * hpbl + 5.0f * ust3;
            wm2 = powf(wm3, h2);
            bfxpbl = -0.15f * thv[0] / G * wm3 / hpbl;
            dthv = ysu_max(thv[kt + 1] - thv[kt], tmin);
            we = ysu_max(bfxpbl / dthv, -sqrtf(wm2));

            bfx0 = ysu_max(topdown_radsum / rho2 / CP, 0.0f);
            real wm3_top = G / thv[kt] * bfx0 * hpbl;
            real wm2_top = powf(wm3_top, h2);
            wm2 += wm2_top;
            bfxpbl = -ent_eff * bfx0;
            dthv = ysu_max(thv[kt + 1] - thv[kt], 0.1f);
            we += ysu_max(bfxpbl / dthv, -sqrtf(wm2_top));
            wstar3_2 = G / thv[kt] * bfx0 * hpbl;
            wscale = cbrtf(ust3 + phifac * karman
                           * (wstar3 + wstar3_2) * 0.5f);
            wscale = ysu_min(wscale, us * aphi16);
            wscale = ysu_max(wscale, us / aphi5);
            real gamfac = bfac / rho / wscale;
            hgamt = ysu_min(gamfac * hf / CP, gamcrt);
            hgamq = ysu_min(gamfac * qf, gamcrq);
            gamfac = bfac / rho2 / wscale;
            real hgamt2 = ysu_min(gamfac * topdown_radsum / CP, gamcrt);
            hgamt = ysu_max(hgamt, 0.0f) + ysu_max(hgamt2, 0.0f);
            real cg = -15.9f * us * us / ysu_max(wspd[col], 1.0e-9f)
                    * (wstar3 + wstar3_2)
                    / ysu_max(wscale * wscale * wscale * wscale, 1.0e-20f);
            hgamu = cg * u0;
            hgamv = cg * v0;
        }
        real dth = ysu_max(theta[(kt + 1) * st + col] - theta[kt * st + col], tmin);
        real dqq = ysu_min(qv[(kt + 1) * st + col] - qv[kt * st + col], 0.0f);
        hfxpbl = we * dth;
        qfxpbl = we * dqq;
        real dux = u[(kt + 1) * st + col] - u[kt * st + col];
        real dvx = v[(kt + 1) * st + col] - v[kt * st + col];
        if (dux > tmin) ufxpbl = ysu_max(we * dux, -us * us);
        else if (dux < -tmin) ufxpbl = ysu_min(we * dux, us * us);
        if (dvx > tmin) vfxpbl = ysu_max(we * dvx, -us * us);
        else if (dvx < -tmin) vfxpbl = ysu_min(we * dvx, us * us);
        real delb = govrth * d3 * hpbl;
        delta = ysu_min(d1 * hpbl + d2 * wm2 / delb, 100.0f);
    }
    for (int k = 0; k < nz; ++k) {
        entfac[k] = 1.0e30f;
        if (pblflg && delta > 0.0f && k + 1 >= kpbl) {
            real e = (zq[k + 1] - hpbl) / delta;
            entfac[k] = e * e;
        }
        xkzm[k] = xkzh[k] = xkzq[k] = 0.0f;
        if (k < nz - 1) {
            xkzm[k] = xkzminm;
            xkzh[k] = xkzq[k] = xkzminh;
        }
        xkzml[k] = xkzhl[k] = zfacent[k] = 0.0f;
    }

    for (int k = 0; k < nz; ++k) {
        if (k + 1 < kpbl) {
            real zfac = ysu_clip(1.0f - (zq[k + 1] - za[0]) / (hpbl - za[0]),
                                  zfmin, 1.0f);
            zfacent[k] = powf(1.0f - zfac, 3.0f);
            real wsk = cbrtf(ysu_max(ust3 + phifac * karman * wstar3
                                     * (1.0f - zfac), 0.0f));
            real wsk2 = cbrtf(ysu_max(phifac * karman * wstar3_2 * zfac,
                                      0.0f));
            real prfac, prfac2, prnumfac;
            if (sfcflg) {
                prfac = bfac * karman * sfcfrac;
                prfac2 = 15.9f * (wstar3 + wstar3_2) / ust3
                       / (1.0f + 4.0f * karman
                          * (wstar3 + wstar3_2) / ust3);
                real zz = ysu_max(zq[k + 1] - sfcfrac * hpbl, 0.0f);
                prnumfac = -3.0f * zz * zz / (hpbl * hpbl);
            } else {
                prfac = prfac2 = prnumfac = 0.0f;
                wsk = ysu_max(us / (1.0f + aphi5 * zol1 * zq[k + 1] / za[0]),
                              0.001f);
            }
            real prnum0 = ysu_clip(phih / phim + prfac, prmin, prmax);
            real km = wsk * karman * zq[k + 1] * zfac * zfac
                    + wsk2 * karman * (hpbl - zq[k + 1])
                    * (1.0f - zfac) * (1.0f - zfac);
            if (k == kpbl - 2 && cloudflg && we < 0.0f) km = 0.0f;
            real prnum = 1.0f + (prnum0 - 1.0f) * expf(prnumfac);
            real kqq = km / prnum;
            prnum0 /= 1.0f + prfac2 * karman * sfcfrac;
            prnum = 1.0f + (prnum0 - 1.0f) * expf(prnumfac);
            real kh = km / prnum;
            xkzm[k] = ysu_min(km + xkzminm, xkzmax);
            xkzh[k] = ysu_min(kh + xkzminh, xkzmax);
            xkzq[k] = ysu_min(kqq + xkzminh, xkzmax);
        }
    }

    for (int k = 0; k < nz - 1; ++k) {
        if (k + 1 >= kpbl) {
            int q0i = k * st + col, q1i = (k + 1) * st + col;
            real ud = u[q1i] - u[q0i], vd = v[q1i] - v[q0i];
            real ss = (ud * ud + vd * vd) / (dza[k + 1] * dza[k + 1]) + 1.0e-9f;
            real ri = (G / (0.5f * (thv[k + 1] + thv[k])))
                    * (thv[k + 1] - thv[k]) / (ss * dza[k + 1]);
            if (qc[q0i] + qi[q0i] > 1.0e-5f && qc[q1i] + qi[q1i] > 1.0e-5f) {
                real qmean = 0.5f * (qv[q0i] + qv[q1i]);
                real tmean = 0.5f * (theta[q0i] * exner[q0i]
                                     + theta[q1i] * exner[q1i]);
                real alph = XLV * qmean / RD / tmean;
                real chi = XLV * XLV * qmean / CP / RV / (tmean * tmean);
                ri = (1.0f + alph) * (ri - G * G / ss / tmean / CP
                                      * ((chi - alph) / (1.0f + chi)));
            }
            real zk = karman * zq[k + 1];
            real rlamdz = ysu_min(ysu_max(0.1f * dza[k + 1], rlam), 300.0f);
            rlamdz = ysu_min(dza[k + 1], rlamdz);
            real rl = zk * rlamdz / (rlamdz + zk);
            real dk = rl * rl * sqrtf(ss), km, kh;
            if (ri < 0.0f) {
                ri = ysu_max(ri, rimin);
                real sri = sqrtf(-ri);
                km = dk * (1.0f + 8.0f * (-ri) / (1.0f + 1.746f * sri));
                kh = dk * (1.0f + 8.0f * (-ri) / (1.0f + 1.286f * sri));
            } else {
                kh = dk / ((1.0f + 5.0f * ri) * (1.0f + 5.0f * ri));
                km = kh * ysu_min(1.0f + 2.1f * ri, prmax);
            }
            xkzm[k] = ysu_min(km + xkzminm, xkzmax);
            xkzh[k] = ysu_min(kh + xkzminh, xkzmax);
            xkzml[k] = xkzm[k];
            xkzhl[k] = xkzh[k];
        }
    }
    for (int k = kpbl - 1; k < nz - 1; ++k) xkzq[k] = xkzh[k];

    // Heat matrix, including the countergradient and entrainment fluxes.
    for (int k = 0; k < nz; ++k) lower[k] = diag[k] = upper[k] = rhs[k] = 0.0f;
    diag[0] = 1.0f;
    rhs[0] = theta0 - 300.0f + hf / (CP / G) / delp[0] * dt2;
    for (int k = 0; k < nz - 1; ++k) {
        real dtodsd = dt2 / delp[k], dtodsu = dt2 / delp[k + 1];
        real dsig = p[k * st + col] - p[(k + 1) * st + col];
        real rdz = 1.0f / dza[k + 1];
        real tem1 = dsig * xkzh[k] * rdz;
        if (pblflg && k + 1 < kpbl) {
            real flux = tem1 * (-hgamt / hpbl - hfxpbl * zfacent[k] / xkzh[k]);
            rhs[k] += dtodsd * flux;
            rhs[k + 1] = theta[(k + 1) * st + col] - 300.0f - dtodsu * flux;
        } else if (pblflg && k + 1 >= kpbl && entfac[k] < 4.6f) {
            real knew = -we * dza[kpbl - 1] * expf(-entfac[k]);
            xkzh[k] = ysu_clip(sqrtf(ysu_max(knew * xkzhl[k], 0.0f)),
                               xkzminh, xkzmax);
            rhs[k + 1] = theta[(k + 1) * st + col] - 300.0f;
        } else rhs[k + 1] = theta[(k + 1) * st + col] - 300.0f;
        real dsdz2 = dsig * xkzh[k] * rdz * rdz;
        upper[k] = -dtodsd * dsdz2;
        lower[k + 1] = -dtodsu * dsdz2;
        diag[k] -= upper[k];
        diag[k + 1] = 1.0f - lower[k + 1];
    }
    ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    for (int k = 0; k < nz; ++k) {
        int q = k * st + col;
        // WRF's own association, bl_ysu.F90:1103: (f1(i,k) - thx(i,k) + 300.).
        // Not cosmetic.  The solve leaves rhs[k] ~ theta - 300, so
        // (rhs - theta) is exactly -300 and adding 300 gives exactly zero,
        // whereas (rhs + 300) rounds at magnitude ~theta -- one ULP coarser
        // than rhs itself -- before the subtract, and the residue survives.
        // Measured: 3.4e-07 K/s where WRF writes exactly 0.0, which is
        // 884345697 ULP of that zero.
        dtheta[q] = (rhs[k] - theta[q] + 300.0f) * rdt;
    }

    // Vapor matrix (Kq); cloud and ice reuse its matrix.
    for (int k = 0; k < nz; ++k) lower[k] = diag[k] = upper[k] = rhs[k] = 0.0f;
    diag[0] = 1.0f;
    rhs[0] = qv0 + qf * G / delp[0] * dt2;
    for (int k = 0; k < nz - 1; ++k) {
        real dtodsd = dt2 / delp[k], dtodsu = dt2 / delp[k + 1];
        real dsig = p[k * st + col] - p[(k + 1) * st + col];
        real rdz = 1.0f / dza[k + 1];
        real tem1 = dsig * xkzq[k] * rdz;
        if (pblflg && k + 1 < kpbl) {
            real flux = tem1 * (-qfxpbl * zfacent[k] / xkzq[k]);
            rhs[k] += dtodsd * flux;
            rhs[k + 1] = qv[(k + 1) * st + col] - dtodsu * flux;
        } else if (pblflg && k + 1 >= kpbl && entfac[k] < 4.6f) {
            real knew = -we * dza[kpbl - 1] * expf(-entfac[k]);
            xkzq[k] = ysu_clip(sqrtf(ysu_max(knew * xkzhl[k], 0.0f)),
                               xkzminh, xkzmax);
            rhs[k + 1] = qv[(k + 1) * st + col];
        } else rhs[k + 1] = qv[(k + 1) * st + col];
        real dsdz2 = dsig * xkzq[k] * rdz * rdz;
        upper[k] = -dtodsd * dsdz2;
        lower[k + 1] = -dtodsu * dsdz2;
        diag[k] -= upper[k];
        diag[k + 1] = 1.0f - lower[k + 1];
    }
    ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    for (int k = 0; k < nz; ++k) dqv[k * st + col] = (rhs[k] - qv[k * st + col]) * rdt;
    for (int k = 0; k < nz; ++k) rhs[k] = qc[k * st + col];
    ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    for (int k = 0; k < nz; ++k) dqc[k * st + col] = (rhs[k] - qc[k * st + col]) * rdt;
    for (int k = 0; k < nz; ++k) rhs[k] = qi[k * st + col];
    ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    for (int k = 0; k < nz; ++k) dqi[k * st + col] = (rhs[k] - qi[k * st + col]) * rdt;

    // Momentum matrix and implicit surface stress.
    for (int k = 0; k < nz; ++k) lower[k] = diag[k] = upper[k] = rhs[k] = 0.0f;
    real fric = us * us / wspd1 * rho * G / delp[0] * dt2
              * (wspd1 / ysu_max(wspd[col], 1.0e-9f))
              * (wspd1 / ysu_max(wspd[col], 1.0e-9f));
    diag[0] = 1.0f + fric;
    rhs[0] = u0;
    for (int k = 0; k < nz - 1; ++k) {
        real dtodsd = dt2 / delp[k], dtodsu = dt2 / delp[k + 1];
        real dsig = p[k * st + col] - p[(k + 1) * st + col];
        real rdz = 1.0f / dza[k + 1];
        real tem1 = dsig * xkzm[k] * rdz;
        if (pblflg && k + 1 < kpbl) {
            real flux = tem1 * (-hgamu / hpbl - ufxpbl * zfacent[k] / xkzm[k]);
            rhs[k] += dtodsd * flux;
            rhs[k + 1] = u[(k + 1) * st + col] - dtodsu * flux;
        } else if (pblflg && k + 1 >= kpbl && entfac[k] < 4.6f) {
            xkzm[k] = ysu_clip(sqrtf(ysu_max(xkzh[k] * xkzml[k], 0.0f)),
                               xkzminm, xkzmax);
            rhs[k + 1] = u[(k + 1) * st + col];
        } else rhs[k + 1] = u[(k + 1) * st + col];
        real dsdz2 = dsig * xkzm[k] * rdz * rdz;
        upper[k] = -dtodsd * dsdz2;
        lower[k + 1] = -dtodsu * dsdz2;
        diag[k] -= upper[k];
        diag[k + 1] = 1.0f - lower[k + 1];
    }
    ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    for (int k = 0; k < nz; ++k) du[k * st + col] = (rhs[k] - u[k * st + col]) * rdt;
    for (int k = 0; k < nz; ++k) rhs[k] = v[k * st + col];
    // tridi2n assembles distinct u/v countergradient right-hand sides while
    // sharing one matrix.  Recreate the v RHS after the u solve.
    for (int k = 0; k < nz - 1; ++k) {
        if (pblflg && k + 1 < kpbl) {
            real dtodsd = dt2 / delp[k], dtodsu = dt2 / delp[k + 1];
            real dsig = p[k * st + col] - p[(k + 1) * st + col];
            real rdz = 1.0f / dza[k + 1];
            real tem1 = dsig * xkzm[k] * rdz;
            real flux = tem1 * (-hgamv / hpbl - vfxpbl * zfacent[k] / xkzm[k]);
            rhs[k] += dtodsd * flux;
            rhs[k + 1] = v[(k + 1) * st + col] - dtodsu * flux;
        }
    }
    ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    for (int k = 0; k < nz; ++k) dv[k * st + col] = (rhs[k] - v[k * st + col]) * rdt;

    exch_h[col] = exch_m[col] = 0.0f;
    for (int k = 1; k < nz; ++k) {
        exch_h[k * st + col] = xkzh[k - 1];
        exch_m[k * st + col] = xkzm[k - 1];
    }
    hpbl_out[col] = hpbl;
    kpbl_out[col] = kpbl;
    wstar_out[col] = wstar;
    delta_out[col] = delta;
    topdown_radsum_out[col] = topdown_radsum;
    wstar3_2_out[col] = wstar3_2;
    cloudflg_out[col] = cloudflg ? 1 : 0;
}
