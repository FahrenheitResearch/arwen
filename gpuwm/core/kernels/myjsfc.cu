// MYJ (Eta similarity) surface layer, sf_sfclay_physics=2, WRF v4.6.1.
//
// CUDA mirror of the float32 CPU authority
// gpuwm/verify/myj_ref.py::np_myjsfc_column, itself transcribed line by
// line from the byte-frozen phys/module_sf_myjsfc.F (MYJSFC + SFCDIF).
// The :NNN line anchors below are that Fortran file's.  One thread owns
// one surface column: the full dz/tke columns feed the PBLH scan
// (:246,:263-277), everything else is lowest-model-level state.  The
// PSIM/PSIH tables are BUILT ON THE HOST by
// gpuwm/core/myjsfc_tables.build_psi_tables (the same float32 words the
// CPU authority interpolates) and passed in as device arrays, so the two
// halves of the port share one table byte-for-byte.
//
// Conformance status: CPU-vs-CUDA agreement is asserted by
// tests/test_myj_port.py within a documented tolerance; NO oracle
// comparison against the WRF Fortran has been run yet (that campaign is
// the declared next stage, as it was for Shin-Hong and Grell-Freitas).
// libm identity (logf/expf/powf vs glibc) is deliberately unpinned until
// that campaign.
//
// Faithful quirks: CT is identically zero (:206-211, :816-825); the
// Chen-Zhang CZIL block is commented out in the source so CZIL=0.1
// unconditionally (:689-697) and IVGTYP/ISURBAN/IZ0TLND are never read;
// the Zilitinkevich fix uses Z0BASE, not the working ZNT (:733).

#ifndef MYJSFC_KZTM
#define MYJSFC_KZTM 10001
#endif
#define MYJSFC_KZTM2 (MYJSFC_KZTM - 2)

// ---- shared model constants (share/module_model_constants.F) ----
#define MYJ_G        9.81f
#define MYJ_RD       287.0f
#define MYJ_RV       461.6f
#define MYJ_CP       (7.0f * 287.0f / 2.0f)
#define MYJ_XLV      2.5e6f
#define MYJ_P608     (461.6f / 287.0f - 1.0f)
#define MYJ_PQ0      379.90516f
#define MYJ_A2       17.2693882f
#define MYJ_A3       273.16f
#define MYJ_A4       35.86f
#define MYJ_EPSQ2    0.2f
#define MYJ_P1000MB  1.0e5f

// ---- module_sf_myjsfc.F:23-55 ----
#define SFC_ITRMX    5
#define SFC_VKARMAN  0.4f
#define SFC_CAPA     (MYJ_RD / MYJ_CP)
#define SFC_ELOCP    (2.72e6f / MYJ_CP)
#define SFC_RCAP     (1.0f / SFC_CAPA)
#define SFC_GOCP02   (MYJ_G / MYJ_CP * 2.0f)
#define SFC_GOCP10   (MYJ_G / MYJ_CP * 10.0f)
#define SFC_EPSU2    1.0e-6f
#define SFC_EPSUST   1.0e-9f
#define SFC_EPSZT    1.0e-28f
#define SFC_A2S      17.2693882f
#define SFC_A3S      273.16f
#define SFC_A4S      35.86f
#define SFC_SEAFC    0.98f
#define SFC_PQ0SEA   (MYJ_PQ0 * SFC_SEAFC)
#define SFC_EXCML    0.0001f
#define SFC_EXCMS    0.0001f
#define SFC_GLKBR    10.0f
#define SFC_GLKBS    30.0f
#define SFC_QVISC    2.1e-5f
#define SFC_RIC      0.505f
#define SFC_SMALL    0.35f
#define SFC_SQPR     0.84f
#define SFC_SQSC     0.84f
#define SFC_SQVISC   258.2f
#define SFC_TVISC    2.1e-5f
#define SFC_USTC     0.7f
#define SFC_USTR     0.225f
#define SFC_VISC     1.5e-5f
#define SFC_FH       1.01f
#define SFC_WWST     1.2f
#define SFC_ZTFC     1.0f
#define SFC_CZIV     (SFC_SMALL * SFC_GLKBS)
#define SFC_GRRS     (SFC_GLKBR / SFC_GLKBS)
#define SFC_RTVISC   (1.0f / SFC_TVISC)
#define SFC_RVISC    (1.0f / SFC_VISC)
#define SFC_ZQRZT    (SFC_SQSC / SFC_SQPR)
#define SFC_FZQ1     (SFC_RTVISC * SFC_QVISC * SFC_ZQRZT)
#define SFC_FZQ2     (SFC_RTVISC * SFC_QVISC * SFC_ZQRZT)
#define SFC_FZT1     (SFC_RVISC * SFC_TVISC * SFC_SQPR)
#define SFC_FZT2     (SFC_CZIV * SFC_GRRS * SFC_TVISC * SFC_SQPR)
#define SFC_FZU1     (SFC_CZIV * SFC_VISC)
#define SFC_RQVISC   (1.0f / SFC_QVISC)
#define SFC_USTFC    (0.018f / MYJ_G)
#define SFC_WWST2    (SFC_WWST * SFC_WWST)

// One PSIM/PSIH table lookup (module_sf_myjsfc.F:583-588 pattern).
__device__ static real myjsfc_table(const real* __restrict__ psi,
                                    real zeta, real ztmin, real dzeta)
{
    real rz = (zeta - ztmin) / dzeta;
    int k = (int)rz;
    real rdzt = rz - (real)k;
    k = min(k, MYJSFC_KZTM2);
    k = max(k, 0);
    return (psi[k + 1] - psi[k]) * rdzt + psi[k];
}

extern "C" __global__ void myjsfc_column(
    // full columns, gpuwm bottom-up (nz, n)
    const real* __restrict__ dz_a,
    const real* __restrict__ tke_a,
    // lowest-model-level state (n)
    const real* __restrict__ u1_a, const real* __restrict__ v1_a,
    const real* __restrict__ t1_a, const real* __restrict__ th1_a,
    const real* __restrict__ qv1_a, const real* __restrict__ qc1_a,
    const real* __restrict__ p1_a, const real* __restrict__ psfc_a,
    // static surface (n)
    const real* __restrict__ tsk_a, const real* __restrict__ xland_a,
    const real* __restrict__ mavail_a, const real* __restrict__ z0base_a,
    // WRF INOUT state (n)
    real* __restrict__ ust_a, real* __restrict__ znt_a,
    real* __restrict__ thz0_a, real* __restrict__ qz0_a,
    real* __restrict__ uz0_a, real* __restrict__ vz0_a,
    real* __restrict__ qsfc_a, real* __restrict__ akhs_a,
    real* __restrict__ akms_a,
    // outputs (n)
    real* __restrict__ rmol_a, real* __restrict__ ct_a,
    real* __restrict__ pblh_a, real* __restrict__ rib_a,
    real* __restrict__ chs_a, real* __restrict__ chs2_a,
    real* __restrict__ cqs2_a, real* __restrict__ hfx_a,
    real* __restrict__ qfx_a, real* __restrict__ lh_a,
    real* __restrict__ flhc_a, real* __restrict__ flqc_a,
    real* __restrict__ qgh_a, real* __restrict__ cpm_a,
    real* __restrict__ u10_a, real* __restrict__ v10_a,
    real* __restrict__ t2_a, real* __restrict__ th2_a,
    real* __restrict__ tshltr_a, real* __restrict__ th10_a,
    real* __restrict__ q2_a, real* __restrict__ qshltr_a,
    real* __restrict__ q10_a, real* __restrict__ pshltr_a,
    real* __restrict__ u10e_a, real* __restrict__ v10e_a,
    // similarity tables (MYJSFC_KZTM each) + their scalars
    const real* __restrict__ psim1, const real* __restrict__ psih1,
    const real* __restrict__ psim2, const real* __restrict__ psih2,
    real dzeta1, real dzeta2, real ztmin1, real ztmax1,
    real ztmin2, real ztmax2, real fh01, real fh02,
    int itimestep, int nz, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    size_t st = (size_t)n;

    int ntsd = itimestep;
    real ust = ust_a[col];
    if (ntsd == 1) ust = 0.1f;            // :186-203 (ARW branch)
    real seamask = xland_a[col] - 1.0f;   // :231
    real psfc = psfc_a[col];
    real thsk = tsk_a[col] / powf(psfc / MYJ_P1000MB, SFC_CAPA);  // :227

    // PBLH scan (:263-277): lpbl is the myj (top-down, 1-based) index of
    // the first weak-TKE level above the lowest layer; PBLH=ZHK(LPBL).
    //
    // DELIBERATE DIVERGENCE, gpuwm goes its own way (float32 precision).
    // WRF builds ZHK upward from ZINT(I,KTE+1,J)=HT(I,J) (:162) and then
    // subtracts ZHK(LMH+1) back off (:277), so over terrain it differences
    // two sea-level heights; this kernel accumulates dz directly, which is
    // the same value in real arithmetic and a strictly better-conditioned
    // one in float32.  Same declaration as the PBL kernel's header block;
    // measured by
    // tests/test_myj_port.py::test_the_dropped_terrain_height_cancels_in_float32.
    int lmh = nz;
    int lpbl = lmh;
    real zcum = dz_a[0];                  // interface height above layer 0
    real pblh = 0.0f;
    bool found = false;
    for (int iz = 1; iz < nz && !found; ++iz) {
        zcum += dz_a[(size_t)iz * st + col];
        if (2.0f * tke_a[(size_t)iz * st + col] <= MYJ_EPSQ2 * SFC_FH) {
            lpbl = nz - iz;
            pblh = zcum;                  // ZHK(LPBL) - ZHK(LMH+1)
            found = true;
        }
    }
    if (!found) { lpbl = 1; pblh = zcum; }
    (void)lpbl;

    // Lowest-layer state (:284-294).
    real plow = p1_a[col];
    real tlow = t1_a[col];
    real thlow = th1_a[col];
    real qv1 = qv1_a[col];
    real qlow = qv1 / (1.0f + qv1);
    real cwmlow = qc1_a[col];
    real thelow = (cwmlow * (-SFC_ELOCP / tlow) + 1.0f) * thlow;
    real ulow = u1_a[col];
    real vlow = v1_a[col];
    real zsl = dz_a[col] * 0.5f;
    real apesfc = powf(psfc / MYJ_P1000MB, SFC_CAPA);
    real tz0 = (ntsd == 1) ? tsk_a[col] : thz0_a[col] * apesfc;  // :296-305

    // ---- SFCDIF (:361-1056) ----
    real qs = qsfc_a[col];
    real thz0 = thz0_a[col];
    real qz0 = qz0_a[col];
    real uz0 = uz0_a[col];
    real vz0 = vz0_a[col];
    real z0 = znt_a[col];
    real akhs = akhs_a[col];
    real akms = akms_a[col];
    real z0base = z0base_a[col];
    real mavail = mavail_a[col];

    real rdz = 1.0f / zsl;
    real cxchl = SFC_EXCML * rdz;
    real cxchs = SFC_EXCMS * rdz;
    real btgx = MYJ_G / thlow;
    real elfc = SFC_VKARMAN * btgx;
    real btgh = (pblh > 1000.0f) ? btgx * pblh : btgx * 1000.0f;

    real wstar2 = 0.0f;
    real rlmo = 0.0f;
    real rib = 0.0f, du2 = SFC_EPSU2, dthv = 0.0f;
    real zu = z0, zt = z0;
    real psmz = 0.0f, pshz = 0.0f, ustark = ust * SFC_VKARMAN;

    if (seamask > 0.5f) {
        // ---- sea points (:443-636) ----
        for (int itr = 0; itr < SFC_ITRMX; ++itr) {
            z0 = fmaxf(SFC_USTFC * ust * ust, 1.59e-5f);
            if (ust < SFC_USTC) {
                if (ust < SFC_USTR) {
                    if (ntsd == 1) {
                        akms = cxchs;
                        akhs = cxchs;
                        qs = qlow;
                    }
                    zu = SFC_FZU1 * sqrtf(sqrtf(z0 * ust * SFC_RVISC)) / ust;
                    real wght = akms * zu * SFC_RVISC;
                    real rwgh = wght / (wght + 1.0f);
                    uz0 = (ulow * rwgh + uz0) * 0.5f;
                    vz0 = (vlow * rwgh + vz0) * 0.5f;
                    zt = SFC_FZT1 * zu;
                    real zq = SFC_FZQ1 * zt;
                    real wghtt = akhs * zt * SFC_RTVISC;
                    real wghtq = akhs * zq * SFC_RQVISC;
                    if (ntsd > 1) {
                        thz0 = ((wghtt * thlow + thsk) / (wghtt + 1.0f)
                                + thz0) * 0.5f;
                        qz0 = ((wghtq * qlow + qs) / (wghtq + 1.0f)
                               + qz0) * 0.5f;
                    } else {
                        thz0 = (wghtt * thlow + thsk) / (wghtt + 1.0f);
                        qz0 = (wghtq * qlow + qs) / (wghtq + 1.0f);
                    }
                }
                if (ust >= SFC_USTR && ust < SFC_USTC) {
                    zu = z0;
                    uz0 = 0.0f;
                    vz0 = 0.0f;
                    zt = SFC_FZT2 * sqrtf(sqrtf(z0 * ust * SFC_RVISC)) / ust;
                    real zq = SFC_FZQ2 * zt;
                    real wghtt = akhs * zt * SFC_RTVISC;
                    real wghtq = akhs * zq * SFC_RQVISC;
                    if (ntsd > 1) {
                        thz0 = ((wghtt * thlow + thsk) / (wghtt + 1.0f)
                                + thz0) * 0.5f;
                        qz0 = ((wghtq * qlow + qs) / (wghtq + 1.0f)
                               + qz0) * 0.5f;
                    } else {
                        thz0 = (wghtt * thlow + thsk) / (wghtt + 1.0f);
                        qz0 = (wghtq * qlow + qs) / (wghtq + 1.0f);
                    }
                }
            } else {
                zu = z0;
                uz0 = 0.0f;
                vz0 = 0.0f;
                zt = z0;
                thz0 = thsk;
                qz0 = qs;
            }
            real tem = (tlow + tz0) * 0.5f;
            real thm = (thelow + thz0) * 0.5f;
            real a = thm * MYJ_P608;
            real b = (SFC_ELOCP / tem - 1.0f - MYJ_P608) * thm;
            dthv = ((thelow - thz0)
                    * ((qlow + qz0 + cwmlow) * (0.5f * MYJ_P608) + 1.0f)
                    + (qlow - qz0 + cwmlow) * a + cwmlow * b);
            du2 = fmaxf((ulow - uz0) * (ulow - uz0)
                        + (vlow - vz0) * (vlow - vz0), SFC_EPSU2);
            rib = btgx * dthv * zsl / du2;
            real zslu = zsl + zu;
            real zslt = zsl + zt;
            real rzsu = zslu / zu;
            real rzst = zslt / zt;
            real rlogu = logf(rzsu);
            real rlogt = logf(rzst);
            rlmo = elfc * akhs * dthv / (ust * ust * ust);
            real zetalu = zslu * rlmo;
            real zetalt = zslt * rlmo;
            real zetau = zu * rlmo;
            real zetat = zt * rlmo;
            zetalu = fminf(fmaxf(zetalu, ztmin1), ztmax1);
            zetalt = fminf(fmaxf(zetalt, ztmin1), ztmax1);
            zetau = fminf(fmaxf(zetau, ztmin1 / rzsu), ztmax1 / rzsu);
            zetat = fminf(fmaxf(zetat, ztmin1 / rzst), ztmax1 / rzst);
            psmz = myjsfc_table(psim1, zetau, ztmin1, dzeta1);
            real psmzl = myjsfc_table(psim1, zetalu, ztmin1, dzeta1);
            real simm = psmzl - psmz + rlogu;
            pshz = myjsfc_table(psih1, zetat, ztmin1, dzeta1);
            real pshzl = myjsfc_table(psih1, zetalt, ztmin1, dzeta1);
            real simh = (pshzl - pshz + rlogt) * fh01;
            ustark = ust * SFC_VKARMAN;
            akms = fmaxf(ustark / simm, cxchs);
            akhs = fmaxf(ustark / simh, cxchs);
            if (dthv <= 0.0f) {
                wstar2 = SFC_WWST2
                    * powf(fabsf(btgh * akhs * dthv), 2.0f / 3.0f);
            } else {
                wstar2 = 0.0f;
            }
            ust = fmaxf(sqrtf(akms * sqrtf(du2 + wstar2)), SFC_EPSUST);
        }
    } else {
        // ---- land points (:644-806) ----
        if (ntsd == 1) qs = qlow;
        zu = z0;
        uz0 = 0.0f;
        vz0 = 0.0f;
        zt = zu * SFC_ZTFC;
        thz0 = thsk;
        qz0 = qs;
        real tem = (tlow + tz0) * 0.5f;
        real thm = (thelow + thz0) * 0.5f;
        real a = thm * MYJ_P608;
        real b = (SFC_ELOCP / tem - 1.0f - MYJ_P608) * thm;
        dthv = ((thelow - thz0)
                * ((qlow + qz0 + cwmlow) * (0.5f * MYJ_P608) + 1.0f)
                + (qlow - qz0 + cwmlow) * a + cwmlow * b);
        du2 = fmaxf((ulow - uz0) * (ulow - uz0)
                    + (vlow - vz0) * (vlow - vz0), SFC_EPSU2);
        rib = btgx * dthv * zsl / du2;
        real zslu = zsl + zu;
        real rzsu = zslu / zu;
        real rlogu = logf(rzsu);
        real zslt = zsl + zu;       // u,v and t are at the same level (:685)
        const real czil = 0.1f;     // :697 (Chen-Zhang block commented out)
        real zilfc = -czil * SFC_VKARMAN * SFC_SQVISC;
        const real czetmax = 10.0f;
        real zzil;
        if (dthv > 0.0f) {
            if (rib < SFC_RIC) {
                zzil = zilfc * (1.0f + (rib / SFC_RIC) * (rib / SFC_RIC)
                                * czetmax);
            } else {
                zzil = zilfc * (1.0f + czetmax);
            }
        } else {
            zzil = zilfc;
        }
        for (int itr = 0; itr < SFC_ITRMX; ++itr) {
            zt = fmaxf(expf(zzil * sqrtf(ust * z0base)) * z0base,
                       SFC_EPSZT);                       // :733
            real rzst = zslt / zt;
            real rlogt = logf(rzst);
            rlmo = elfc * akhs * dthv / (ust * ust * ust);
            real zetalu = zslu * rlmo;
            real zetalt = zslt * rlmo;
            real zetau = zu * rlmo;
            real zetat = zt * rlmo;
            zetalu = fminf(fmaxf(zetalu, ztmin2), ztmax2);
            zetalt = fminf(fmaxf(zetalt, ztmin2), ztmax2);
            zetau = fminf(fmaxf(zetau, ztmin2 / rzsu), ztmax2 / rzsu);
            zetat = fminf(fmaxf(zetat, ztmin2 / rzst), ztmax2 / rzst);
            psmz = myjsfc_table(psim2, zetau, ztmin2, dzeta2);
            real psmzl = myjsfc_table(psim2, zetalu, ztmin2, dzeta2);
            real simm = psmzl - psmz + rlogu;
            pshz = myjsfc_table(psih2, zetat, ztmin2, dzeta2);
            real pshzl = myjsfc_table(psih2, zetalt, ztmin2, dzeta2);
            real simh = (pshzl - pshz + rlogt) * fh02;
            ustark = ust * SFC_VKARMAN;
            akms = fmaxf(ustark / simm, cxchl);
            akhs = fmaxf(ustark / simh, cxchl);
            if (dthv <= 0.0f) {
                wstar2 = SFC_WWST2
                    * powf(fabsf(btgh * akhs * dthv), 2.0f / 3.0f);
            } else {
                wstar2 = 0.0f;
            }
            ust = fmaxf(sqrtf(akms * sqrtf(du2 + wstar2)), SFC_EPSUST);
        }
    }
    real ct = 0.0f;                 // :816-825, countergradient commented out

    // ---- 2m / 10m diagnostics (:829-1023) ----
    real umflx = akms * (ulow - uz0);
    real vmflx = akms * (vlow - vz0);
    real hsflx = akhs * (thlow - thz0);
    real hlflx = akhs * (qlow - qz0);
    real zu10 = zu + 10.0f;
    real zt02 = zt + 2.0f;
    real zt10 = zt + 10.0f;
    real rlnu10 = logf(zu10 / zu);
    real rlnt02 = logf(zt02 / zt);
    real rlnt10 = logf(zt10 / zt);
    real ztau10 = zu10 * rlmo;
    real ztat02 = zt02 * rlmo;
    real ztat10 = zt10 * rlmo;
    real akms10, akhs02, akhs10;
    if (seamask > 0.5f) {
        ztau10 = fminf(fmaxf(ztau10, ztmin1), ztmax1);
        ztat02 = fminf(fmaxf(ztat02, ztmin1), ztmax1);
        ztat10 = fminf(fmaxf(ztat10, ztmin1), ztmax1);
        real psm10 = myjsfc_table(psim1, ztau10, ztmin1, dzeta1);
        real simm10 = psm10 - psmz + rlnu10;
        real psh02 = myjsfc_table(psih1, ztat02, ztmin1, dzeta1);
        real simh02 = (psh02 - pshz + rlnt02) * fh01;
        real psh10 = myjsfc_table(psih1, ztat10, ztmin1, dzeta1);
        real simh10 = (psh10 - pshz + rlnt10) * fh01;
        akms10 = fmaxf(ustark / simm10, cxchs);
        akhs02 = fmaxf(ustark / simh02, cxchs);
        akhs10 = fmaxf(ustark / simh10, cxchs);
    } else {
        ztau10 = fminf(fmaxf(ztau10, ztmin2), ztmax2);
        ztat02 = fminf(fmaxf(ztat02, ztmin2), ztmax2);
        ztat10 = fminf(fmaxf(ztat10, ztmin2), ztmax2);
        real psm10 = myjsfc_table(psim2, ztau10, ztmin2, dzeta2);
        real simm10 = psm10 - psmz + rlnu10;
        real psh02 = myjsfc_table(psih2, ztat02, ztmin2, dzeta2);
        real simh02 = (psh02 - pshz + rlnt02) * fh02;
        real psh10 = myjsfc_table(psih2, ztat10, ztmin2, dzeta2);
        real simh10 = (psh10 - pshz + rlnt10) * fh02;
        akms10 = fmaxf(ustark / simm10, cxchl);
        akhs02 = fmaxf(ustark / simh02, cxchl);
        akhs10 = fmaxf(ustark / simh10, cxchl);
    }
    real u10 = umflx / akms10 + uz0;
    real v10 = vmflx / akms10 + vz0;
    real th02 = hsflx / akhs02 + thz0;
    if ((thlow > thz0 && (th02 < thz0 || th02 > thlow))
        || (thlow < thz0 && (th02 > thz0 || th02 < thlow))) {
        th02 = thz0 + 2.0f * rdz * (thlow - thz0);
    }
    real th10 = hsflx / akhs10 + thz0;
    if ((thlow > thz0 && (th10 < thz0 || th10 > thlow))
        || (thlow < thz0 && (th10 > thz0 || th10 < thlow))) {
        th10 = thz0 + 10.0f * rdz * (thlow - thz0);
    }
    real q02 = hlflx / akhs02 + qz0;
    real q10 = hlflx / akhs10 + qz0;
    real pshltr = psfc * expf(-0.068283f / tlow);
    real u10e = u10;
    real v10e = v10;
    if (seamask < 0.5f) {
        // Equivalent z0 shelter winds over land (:988-1020).
        real zuuz = fminf(zu * 0.50f, 0.18f);
        real zu_l = fmaxf(zu * 0.35f, zuuz);
        real zu10_l = zu_l + 10.0f;
        real rzsu_l = zu10_l / zu_l;
        real rlnu10_l = logf(rzsu_l);
        real ztau10_l = zu10_l * rlmo;
        ztau10_l = fminf(fmaxf(ztau10_l, ztmin2), ztmax2);
        real psm10 = myjsfc_table(psim2, ztau10_l, ztmin2, dzeta2);
        real simm10 = psm10 - psmz + rlnu10_l;
        real ekms10 = fmaxf(ustark / simm10, cxchl);
        u10e = umflx / ekms10 + uz0;
        v10e = vmflx / ekms10 + vz0;
    }
    // ---- WRF driver arrays (:1029-1053) ----
    real rlow = plow / (MYJ_RD * tlow);
    real chs = akhs;
    real chs2 = akhs02;
    real cqs2 = akhs02;
    real hfx = -rlow * MYJ_CP * hsflx;
    real qfx = -rlow * hlflx * mavail;
    real flx_lh = MYJ_XLV * qfx;
    real flhc = rlow * MYJ_CP * akhs;
    real flqc = rlow * akhs * mavail;
    real qgh = ((1.0f - seamask) * MYJ_PQ0 + seamask * SFC_PQ0SEA) / plow
        * expf(SFC_A2S * (tlow - SFC_A3S) / (tlow - SFC_A4S));
    qgh = qgh / (1.0f - qgh);              // convert to mixing ratio (:1041)
    real cpm = MYJ_CP * (1.0f + 0.8f * qlow);
    if (seamask > 0.5f) {
        real tskl = tsk_a[col];
        qs = SFC_PQ0SEA / psfc
            * expf(SFC_A2S * (tskl - SFC_A3S) / (tskl - SFC_A4S));
        qs = qs / (1.0f - qs);
    }
    // ---- shelter supersaturation removal (module_sf_myjsfc.F:326-349) ----
    real tshltr = th02;      // SFCDIF's TH02 dummy is MYJSFC's TSHLTR
    real qshltr = q02;       // SFCDIF's Q02 dummy is MYJSFC's QSHLTR
    real rapa = apesfc;
    real th02p = tshltr;
    real th10p = th10;
    real th2_grid = tshltr;
    real rapa02 = rapa - SFC_GOCP02 / th02p;
    real rapa10 = rapa - SFC_GOCP10 / th10p;
    real t02p = th02p * rapa02;
    real t10p = th10p * rapa10;
    real t2_grid = th2_grid * apesfc;      // :337
    real p02p = powf(rapa02, SFC_RCAP) * MYJ_P1000MB;
    real p10p = powf(rapa10, SFC_RCAP) * MYJ_P1000MB;
    real qs02 = MYJ_PQ0 / p02p
        * expf(MYJ_A2 * (t02p - MYJ_A3) / (t02p - MYJ_A4));
    real qs10 = MYJ_PQ0 / p10p
        * expf(MYJ_A2 * (t10p - MYJ_A3) / (t10p - MYJ_A4));
    if (qshltr > qs02) qshltr = qs02;
    if (q10 > qs10) q10 = qs10;
    real q02_grid = qshltr / (1.0f - qshltr);   // :349

    // ---- writeback ----
    ust_a[col] = ust;
    znt_a[col] = z0;
    thz0_a[col] = thz0;
    qz0_a[col] = qz0;
    uz0_a[col] = uz0;
    vz0_a[col] = vz0;
    qsfc_a[col] = qs;
    akhs_a[col] = akhs;
    akms_a[col] = akms;
    rmol_a[col] = rlmo;
    ct_a[col] = ct;
    pblh_a[col] = pblh;
    rib_a[col] = rib;
    chs_a[col] = chs;
    chs2_a[col] = chs2;
    cqs2_a[col] = cqs2;
    hfx_a[col] = hfx;
    qfx_a[col] = qfx;
    lh_a[col] = flx_lh;
    flhc_a[col] = flhc;
    flqc_a[col] = flqc;
    qgh_a[col] = qgh;
    cpm_a[col] = cpm;
    u10_a[col] = u10;
    v10_a[col] = v10;
    t2_a[col] = t2_grid;
    th2_a[col] = th2_grid;
    tshltr_a[col] = tshltr;
    th10_a[col] = th10;
    q2_a[col] = q02_grid;
    qshltr_a[col] = qshltr;
    q10_a[col] = q10;
    pshltr_a[col] = pshltr;
    u10e_a[col] = u10e;
    v10e_a[col] = v10e;
}
