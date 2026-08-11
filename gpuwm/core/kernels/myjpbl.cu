// MYJ (Mellor-Yamada-Janjic level 2.5) PBL, bl_pbl_physics=2, WRF v4.6.1.
//
// CUDA mirror of the float32 CPU authority
// gpuwm/verify/myj_ref.py::np_myjpbl_column, itself transcribed line by
// line from the byte-frozen phys/module_bl_myjpbl.F.  The :NNN anchors
// below are that Fortran file's.  One thread owns one column and runs
// MIXLEN, PRODQ2, DIFCOF, VDIFQ, VDIFH and VDIFV in the Fortran's own
// order.
//
// LAYOUT.  gpuwm columns are bottom-up (index 0 = lowest layer); MYJ
// counts DOWNWARD from the domain top.  The kernel flips on load exactly
// as the Fortran driver flips WRF's arrays (:364-383) and flips back when
// it writes tendencies (:636-652, :751-757).  Inside, ``m`` is the
// zero-based MYJ index (Fortran K = m+1) and ``zh[m]`` is Fortran
// ``ZHK(m+1)``, so ``zh[nz]`` is ZHK(LMH+1) -- the ground.
//
// DELIBERATE DIVERGENCE, gpuwm goes its own way (float32 precision).  WRF
// seeds that column with the TERRAIN HEIGHT, ZINT(I,KTE+1,J)=HT(I,J)
// (:312), so its interface heights are above SEA LEVEL; this kernel sets
// zh[nz] = 0 and carries heights ABOVE GROUND (the CPU authority does the
// same).  Every consumer reads only DIFFERENCES of these heights, so HT
// cancels EXACTLY in real arithmetic -- but not in float32, where
// differencing numbers offset by 1-3 km drops low-order bits the
// ground-relative column keeps.  gpuwm's column is the more accurate one,
// which is why it ships.  Declared, not assumed: the cancellation is
// MEASURED in float32 over 1500 m and 3000 m terrain by
// tests/test_myj_port.py::test_the_dropped_terrain_height_cancels_in_float32,
// and gpuwm/verify/myj_ref.py::_interface_heights carries the full
// declaration and the ``ht`` instrument that test drives.
//
// SPECIES LOOP.  VDIFH's tridiagonal coefficients CM/CR/DTOZ are
// species-INDEPENDENT (:1465-1495), so the kernel builds them once and
// then runs one species at a time through a single RSS scratch row.  That
// is the same arithmetic in the same order as the Fortran's inner
// DO M=MSS,NSPEC, and it is what keeps the per-thread local footprint at
// YSU's scale instead of four times it.
//
// Conformance status: CPU-vs-CUDA agreement is asserted by
// tests/test_myj_port.py within a documented tolerance; NO oracle
// comparison against the WRF Fortran has been run yet (that campaign is
// the declared next stage, as it was for Shin-Hong and Grell-Freitas).
//
// Faithful quirks: CT is identically zero in ARW (MYJSFC zeroes it every
// call, module_sf_myjsfc.F:206-211, and SFCDIF's countergradient block is
// commented out at :816-825), so MIXLEN's DTH+CT fix (:845-850) and
// VDIFH's RKCT term add exactly zero; both are transcribed anyway so the
// seam is WRF's.  DIFCOF's T argument feeds only its commented-out
// inversion block (:1275-1322) and is therefore not passed.

#ifndef MYJ_KMAX
#define MYJ_KMAX 128
#endif

// ---- share/module_model_constants.F ----
#define MYJP_G       9.81f
#define MYJP_RD      287.0f
#define MYJP_CP      (7.0f * 287.0f / 2.0f)
#define MYJP_XLV     2.5e6f
#define MYJP_XLS     2.85e6f
#define MYJP_P608    (461.6f / 287.0f - 1.0f)
#define MYJP_PQ0     379.90516f
#define MYJP_A2      17.2693882f
#define MYJP_A3      273.16f
#define MYJP_A4      35.86f
#define MYJP_EPSQ2   0.2f

// ---- module_bl_myjpbl.F:25-124 ----
#define PBL_VKARMAN  0.4f
#define PBL_CAPA     (MYJP_RD / MYJP_CP)
#define PBL_RLIVWV   (MYJP_XLS / MYJP_XLV)
#define PBL_ELOCP    (2.72e6f / MYJP_CP)
#define PBL_EPS1     1.0e-12f
#define PBL_EPS2     0.0f
#define PBL_EPSL     0.32f
#define PBL_EPSRU    1.0e-7f
#define PBL_EPSRS    1.0e-7f
#define PBL_EPSTRB   1.0e-24f
#define PBL_FH       1.01f
#define PBL_ALPH     0.30f
#define PBL_BETA     (1.0f / 273.0f)
#define PBL_EL0MAX   1000.0f
#define PBL_EL0MIN   1.0f
#define PBL_ELFC     (0.23f * 0.5f)
#define PBL_A1       0.659888514560862645f
#define PBL_A2X      0.6574209922667784586f
#define PBL_B1       11.87799326209552761f
#define PBL_B2       7.226971804046074028f
#define PBL_C1       0.000830955950095854396f
#define PBL_ELZ0     0.0f
#define PBL_ESQ      5.0f
#define PBL_SEAFC    0.98f
#define PBL_PQ0SEA   (MYJP_PQ0 * PBL_SEAFC)
#define PBL_BTG      (PBL_BETA * MYJP_G)
#define PBL_RB1      (1.0f / PBL_B1)

// :62-92
#define PBL_ADNH (9.0f*PBL_A1*PBL_A2X*PBL_A2X*(12.0f*PBL_A1+3.0f*PBL_B2)*PBL_BTG*PBL_BTG)
#define PBL_ADNM (18.0f*PBL_A1*PBL_A1*PBL_A2X*(PBL_B2-3.0f*PBL_A2X)*PBL_BTG)
#define PBL_ANMH (-9.0f*PBL_A1*PBL_A2X*PBL_A2X*PBL_BTG*PBL_BTG)
#define PBL_ANMM (-3.0f*PBL_A1*PBL_A2X*(3.0f*PBL_A2X+3.0f*PBL_B2*PBL_C1+18.0f*PBL_A1*PBL_C1-PBL_B2)*PBL_BTG)
#define PBL_BDNH (3.0f*PBL_A2X*(7.0f*PBL_A1+PBL_B2)*PBL_BTG)
#define PBL_BDNM (6.0f*PBL_A1*PBL_A1)
#define PBL_BEQH (PBL_A2X*PBL_B1*PBL_BTG+3.0f*PBL_A2X*(7.0f*PBL_A1+PBL_B2)*PBL_BTG)
#define PBL_BEQM (-PBL_A1*PBL_B1*(1.0f-3.0f*PBL_C1)+6.0f*PBL_A1*PBL_A1)
#define PBL_BNMH (-PBL_A2X*PBL_BTG)
#define PBL_BNMM (PBL_A1*(1.0f-3.0f*PBL_C1))
#define PBL_BSHH (9.0f*PBL_A1*PBL_A2X*PBL_A2X*PBL_BTG)
#define PBL_BSHM (18.0f*PBL_A1*PBL_A1*PBL_A2X*PBL_C1)
#define PBL_BSMH (-3.0f*PBL_A1*PBL_A2X*(3.0f*PBL_A2X+3.0f*PBL_B2*PBL_C1+12.0f*PBL_A1*PBL_C1-PBL_B2)*PBL_BTG)
#define PBL_CESH PBL_A2X
#define PBL_CESM (PBL_A1*(1.0f-3.0f*PBL_C1))

// :98-101 free term in the equilibrium equation for (L/Q)**2
#define PBL_AEQH (9.0f*PBL_A1*PBL_A2X*PBL_A2X*PBL_B1*PBL_BTG*PBL_BTG \
                  + 9.0f*PBL_A1*PBL_A2X*PBL_A2X*(12.0f*PBL_A1+3.0f*PBL_B2)*PBL_BTG*PBL_BTG)
#define PBL_AEQM (3.0f*PBL_A1*PBL_A2X*PBL_B1*(3.0f*PBL_A2X+3.0f*PBL_B2*PBL_C1+18.0f*PBL_A1*PBL_C1-PBL_B2)*PBL_BTG \
                  + 18.0f*PBL_A1*PBL_A1*PBL_A2X*(PBL_B2-3.0f*PBL_A2X)*PBL_BTG)

// :107-124 forbidden turbulence area / near isotropy
#define PBL_REQU  (-PBL_AEQH/PBL_AEQM)
#define PBL_EPSGH 1.0e-9f
#define PBL_EPSGM (PBL_REQU*PBL_EPSGH)
#define PBL_UBRYL ((18.0f*PBL_REQU*PBL_A1*PBL_A1*PBL_A2X*PBL_B2*PBL_C1*PBL_BTG \
                    + 9.0f*PBL_A1*PBL_A2X*PBL_A2X*PBL_B2*PBL_BTG*PBL_BTG) \
                   / (PBL_REQU*PBL_ADNM+PBL_ADNH))
#define PBL_UBRY  ((1.0f+PBL_EPSRS)*PBL_UBRYL)
#define PBL_UBRY3 (3.0f*PBL_UBRY)
#define PBL_AUBH  (27.0f*PBL_A1*PBL_A2X*PBL_A2X*PBL_B2*PBL_BTG*PBL_BTG - PBL_ADNH*PBL_UBRY3)
#define PBL_AUBM  (54.0f*PBL_A1*PBL_A1*PBL_A2X*PBL_B2*PBL_C1*PBL_BTG - PBL_ADNM*PBL_UBRY3)
#define PBL_BUBH  ((9.0f*PBL_A1*PBL_A2X+3.0f*PBL_A2X*PBL_B2)*PBL_BTG - PBL_BDNH*PBL_UBRY3)
#define PBL_BUBM  (18.0f*PBL_A1*PBL_A1*PBL_C1 - PBL_BDNM*PBL_UBRY3)
#define PBL_CUBR  (1.0f - PBL_UBRY3)
#define PBL_RCUBR (1.0f/PBL_CUBR)

extern "C" __global__ void myjpbl_column(
    // full bottom-up columns, (nz, n)
    const real* __restrict__ dz_a,
    const real* __restrict__ u_a, const real* __restrict__ v_a,
    const real* __restrict__ t_a, const real* __restrict__ th_a,
    const real* __restrict__ exner_a, const real* __restrict__ qv_a,
    const real* __restrict__ qc_a, const real* __restrict__ qi_a,
    const real* __restrict__ p_a,
    real* __restrict__ tke_a,            // TKE_MYJ, inout (nz, n)
    // surface (n)
    const real* __restrict__ psfc_a, const real* __restrict__ ust_a,
    const real* __restrict__ tsk_a, const real* __restrict__ chklowq_a,
    const real* __restrict__ xland_a, const real* __restrict__ sice_a,
    const real* __restrict__ snow_a, const real* __restrict__ akhs_a,
    const real* __restrict__ akms_a, const real* __restrict__ elflx_a,
    const real* __restrict__ uz0_a, const real* __restrict__ vz0_a,
    // inout surface (n)
    real* __restrict__ thz0_a, real* __restrict__ qz0_a,
    real* __restrict__ qsfc_a, real* __restrict__ ct_a,
    // outputs, bottom-up (nz, n)
    real* __restrict__ rublten_a, real* __restrict__ rvblten_a,
    real* __restrict__ rthblten_a, real* __restrict__ rqvblten_a,
    real* __restrict__ rqcblten_a, real* __restrict__ rqiblten_a,
    real* __restrict__ el_myj_a, real* __restrict__ exch_h_a,
    // outputs (n)
    real* __restrict__ pblh_a, int* __restrict__ kpbl_a,
    real* __restrict__ mixht_a,
    real dtturbl, int flqi, int nz, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    if (nz > MYJ_KMAX || nz < 4) return;      // launcher refuses these
    size_t st = (size_t)n;
#define MYJ_UP(a, kup) a[(size_t)(kup) * st + col]

    const int lmh = nz;
    const int nzm = nz - 1;
    real rdtturbl = 1.0f / dtturbl;
    real dtdif = dtturbl;

    // Persistent MYJ-layout columns.
    real zh[MYJ_KMAX + 1];
    real uk[MYJ_KMAX], vk[MYJ_KMAX], tk[MYJ_KMAX];
    real the[MYJ_KMAX], qk[MYJ_KMAX], cwm[MYJ_KMAX], qcik[MYJ_KMAX];
    real q2[MYJ_KMAX], rhok[MYJ_KMAX];
    real gm[MYJ_KMAX], gh[MYJ_KMAX], el[MYJ_KMAX];
    real akm[MYJ_KMAX], akh[MYJ_KMAX];
    // Scratch, reused between phases (documented at each use).
    real s1[MYJ_KMAX], s2[MYJ_KMAX], s3[MYJ_KMAX], s4[MYJ_KMAX];

    // ---- flip on load (:364-383) --------------------------------------
    zh[nz] = 0.0f;                             // ZHK(LMH+1) = HT, cancels
    for (int m = nz - 1; m >= 0; --m) {
        zh[m] = zh[m + 1] + MYJ_UP(dz_a, nz - 1 - m);
    }
    for (int m = 0; m < nz; ++m) {
        int kup = nz - 1 - m;                  // KFLIP
        uk[m] = MYJ_UP(u_a, kup);
        vk[m] = MYJ_UP(v_a, kup);
        tk[m] = MYJ_UP(t_a, kup);
        real thx = MYJ_UP(th_a, kup);
        real ratiomx = MYJ_UP(qv_a, kup);
        qk[m] = ratiomx / (1.0f + ratiomx);
        real cw = MYJ_UP(qc_a, kup);
        qcik[m] = flqi ? MYJ_UP(qi_a, kup) : 0.0f;
        if (flqi) cw = cw + qcik[m];           // :331-332
        cwm[m] = cw;
        the[m] = (cw * (-PBL_ELOCP / tk[m]) + 1.0f) * thx;   // :336
        q2[m] = 2.0f * MYJ_UP(tke_a, kup);     // :377
    }

    // ================= MIXLEN (:771-963) ================================
    int lpbl = lmh;
    for (int kf = lmh - 1; kf >= 1; --kf) {    // K=LMH-1,1,-1
        if (q2[kf - 1] <= MYJP_EPSQ2 * PBL_FH) { lpbl = kf; goto mixlen_110; }
    }
    lpbl = 1;
mixlen_110:
    {
        real pblh = zh[lpbl] - zh[lmh];        // :834 Z(LPBL+1)-Z(LMH+1)
        pblh_a[col] = pblh;
    }
    // DTH lives in s1 (:841-850); q1 lives in s2 (:837-839, :914-916).
    for (int m = 0; m < nzm; ++m) s1[m] = the[m] - the[m + 1];
    {
        real ct = ct_a[col];
        for (int kf = lmh - 2; kf >= 1; --kf) {
            if (s1[kf - 1] > 0.0f && s1[kf] <= 0.0f) {
                s1[kf - 1] = s1[kf - 1] + ct;
                break;
            }
        }
        ct_a[col] = 0.0f;                      // :852
    }
    for (int m = 0; m < nzm; ++m) {
        real rdz = 2.0f / (zh[m] - zh[m + 2]);
        real gml = ((uk[m] - uk[m + 1]) * (uk[m] - uk[m + 1])
                    + (vk[m] - vk[m + 1]) * (vk[m] - vk[m + 1])) * rdz * rdz;
        gm[m] = fmaxf(gml, PBL_EPSGM);
        real tem = (tk[m] + tk[m + 1]) * 0.5f;
        real thm = (the[m] + the[m + 1]) * 0.5f;
        real a = thm * MYJP_P608;
        real b = (PBL_ELOCP / tem - 1.0f - MYJP_P608) * thm;
        real ghl = (s1[m] * ((qk[m] + qk[m + 1] + cwm[m] + cwm[m + 1])
                             * (0.5f * MYJP_P608) + 1.0f)
                    + (qk[m] - qk[m + 1] + cwm[m] - cwm[m + 1]) * a
                    + (cwm[m] - cwm[m + 1]) * b) * rdz;
        if (fabsf(ghl) <= PBL_EPSGH) ghl = PBL_EPSGH;
        gh[m] = ghl;
    }
    // ELM lives in akm, REL in akh: both are dead until DIFCOF (:1216).
    int lmxl = lmh;
    for (int m = 0; m < nzm; ++m) {
        real gml = gm[m], ghl = gh[m], eloq2x;
        if (ghl >= PBL_EPSGH) {
            if (gml / ghl <= PBL_REQU) {
                akm[m] = PBL_EPSL;
                lmxl = m + 1;
                continue;
            }
            real aubr = (PBL_AUBM * gml + PBL_AUBH * ghl) * ghl;
            real bubr = PBL_BUBM * gml + PBL_BUBH * ghl;
            real qol2st = (-0.5f * bubr
                           + sqrtf(bubr * bubr * 0.25f - aubr * PBL_CUBR))
                          * PBL_RCUBR;
            eloq2x = 1.0f / qol2st;
        } else {
            real aden = (PBL_ADNM * gml + PBL_ADNH * ghl) * ghl;
            real bden = PBL_BDNM * gml + PBL_BDNH * ghl;
            real qol2un = -0.5f * bden + sqrtf(bden * bden * 0.25f - aden);
            eloq2x = 1.0f / (qol2un + PBL_EPSRU);
        }
        akm[m] = fmaxf(sqrtf(eloq2x * q2[m]), PBL_EPSL);
    }
    if (akm[lmh - 2] == PBL_EPSL) lmxl = lmh;  // :904
    mixht_a[col] = zh[lmxl - 1] - zh[lmh];     // BLMX (:910-911)
    for (int m = 0; m < lmh; ++m) s2[m] = 0.0f;
    for (int m = lpbl - 1; m < lmh; ++m) s2[m] = sqrtf(q2[m]);
    {
        real szq = 0.0f, sq = 0.0f;
        for (int m = 0; m < nzm; ++m) {
            real qdzl = (s2[m] + s2[m + 1]) * (zh[m + 1] - zh[m + 2]);
            szq = (zh[m + 1] + zh[m + 2] - zh[lmh] - zh[lmh]) * qdzl + szq;
            sq = qdzl + sq;
        }
        real el0 = fminf(PBL_ALPH * szq * 0.5f / sq, PBL_EL0MAX);
        el0 = fmaxf(el0, PBL_EL0MIN);
        int lpblm = max(lpbl - 1, 1);
        for (int m = 0; m < lpblm; ++m) {      // :940-943 above the PBL top
            el[m] = fminf((zh[m] - zh[m + 2]) * PBL_ELFC, akm[m]);
            akh[m] = el[m] / akm[m];           // REL
        }
        if (lpbl < lmh) {                      // :949-955 inside the PBL
            for (int m = lpbl - 1; m < lmh - 1; ++m) {
                real vkrmz = (zh[m + 1] - zh[lmh]) * PBL_VKARMAN;
                el[m] = fminf(vkrmz / (vkrmz / el0 + 1.0f), akm[m]);
                akh[m] = el[m] / akm[m];
            }
        }
        for (int m = lpbl; m < lmh - 2; ++m) { // :957-960 K=LPBL+1,LMH-2
            real srel = fminf(((akh[m - 1] + akh[m + 1]) * 0.5f + akh[m])
                              * 0.5f, akh[m]);
            el[m] = fmaxf(srel * akm[m], PBL_EPSL);
        }
    }

    // ================= PRODQ2 (:967-1167) ===============================
    {
        real ustar = ust_a[col];
        for (int m = 0; m < lmh - 1; ++m) {
            real gml = gm[m], ghl = gh[m];
            real aequ = (PBL_AEQM * gml + PBL_AEQH * ghl) * ghl;
            real bequ = PBL_BEQM * gml + PBL_BEQH * ghl;
            real eqol2 = -0.5f * bequ + sqrtf(bequ * bequ * 0.25f - aequ);
            if ((gml + ghl * ghl <= PBL_EPSTRB)
                || (ghl >= PBL_EPSGH && gml / ghl <= PBL_REQU)
                || (eqol2 <= PBL_EPS2)) {
                q2[m] = MYJP_EPSQ2;
                el[m] = PBL_EPSL;
                continue;
            }
            real anum = (PBL_ANMM * gml + PBL_ANMH * ghl) * ghl;
            real bnum = PBL_BNMM * gml + PBL_BNMH * ghl;
            real aden = (PBL_ADNM * gml + PBL_ADNH * ghl) * ghl;
            real bden = PBL_BDNM * gml + PBL_BDNH * ghl;
            real cden = 1.0f;
            real arhs = -(anum * bden - bnum * aden) * 2.0f;
            real brhs = -anum * 4.0f;
            real crhs = -bnum * 2.0f;
            real dloq1 = el[m] / sqrtf(q2[m]);
            real eloq21 = 1.0f / eqol2;
            real eloq11 = sqrtf(eloq21);
            real eloq31 = eloq21 * eloq11;
            real eloq41 = eloq21 * eloq21;
            real eloq51 = eloq21 * eloq31;
            real rden1 = 1.0f / (aden * eloq41 + bden * eloq21 + cden);
            real rhsp1 = (arhs * eloq51 + brhs * eloq31 + crhs * eloq11)
                         * rden1 * rden1;
            real eloq12 = eloq11 + (dloq1 - eloq11) * expf(rhsp1 * dtturbl);
            eloq12 = fmaxf(eloq12, PBL_EPS1);
            real eloq22 = eloq12 * eloq12;
            real eloq32 = eloq22 * eloq12;
            real eloq42 = eloq22 * eloq22;
            real eloq52 = eloq22 * eloq32;
            real rden2 = 1.0f / (aden * eloq42 + bden * eloq22 + cden);
            real rhs2 = -(anum * eloq42 + bnum * eloq22) * rden2 + PBL_RB1;
            real rhsp2 = (arhs * eloq52 + brhs * eloq32 + crhs * eloq12)
                         * rden2 * rden2;
            real rhst2 = rhs2 / rhsp2;
            real eloq13 = eloq12 - rhst2
                          + (rhst2 + dloq1 - eloq12) * expf(rhsp2 * dtturbl);
            eloq13 = fmaxf(eloq13, PBL_EPS1);
            if (eloq13 > PBL_EPS1) {
                q2[m] = el[m] * el[m] / (eloq13 * eloq13);
                q2[m] = fmaxf(q2[m], MYJP_EPSQ2);
                if (q2[m] == MYJP_EPSQ2) el[m] = PBL_EPSL;
            } else {
                q2[m] = MYJP_EPSQ2;
                el[m] = PBL_EPSL;
            }
        }
        // :1164 lower boundary, left-to-right (B1**(2./3.)*USTAR)*USTAR.
        q2[lmh - 1] = fmaxf(powf(PBL_B1, 2.0f / 3.0f) * ustar * ustar,
                            MYJP_EPSQ2);
    }
    kpbl_a[col] = nz - lpbl + 1;               // :421 KPBL=KTE-LPBL+1

    // ================= DIFCOF (:1172-1325) ==============================
    for (int m = 0; m < lmh - 1; ++m) {
        real ell = el[m];
        real eloq2 = ell * ell / q2[m];
        real eloq4 = eloq2 * eloq2;
        real gml = gm[m], ghl = gh[m];
        real aden = (PBL_ADNM * gml + PBL_ADNH * ghl) * ghl;
        real bden = PBL_BDNM * gml + PBL_BDNH * ghl;
        real cden = 1.0f;
        real besm = PBL_BSMH * ghl;
        real besh = PBL_BSHM * gml + PBL_BSHH * ghl;
        real rden = 1.0f / (aden * eloq4 + bden * eloq2 + cden);
        real esm = (besm * eloq2 + PBL_CESM) * rden;
        real esh = (besh * eloq2 + PBL_CESH) * rden;
        real rdz = 2.0f / (zh[m] - zh[m + 2]);
        real q1l = sqrtf(q2[m]);
        real elqdz = ell * q1l * rdz;
        akm[m] = elqdz * esm;
        akh[m] = elqdz * esh;
    }
    // EXCH_H publication (:438-444), written in WRF-up order.
    for (int kup = 0; kup < nz - 1; ++kup) {
        int kflip = nz - 1 - kup;              // Fortran KFLIP=KTE-K
        real deltaz = 0.5f * (zh[kflip - 1] - zh[kflip + 1]);
        MYJ_UP(exch_h_a, kup) = akh[kflip - 1] * deltaz;
    }
    MYJ_UP(exch_h_a, nz - 1) = 0.0f;           // AKH is undefined at KTE

    // ================= VDIFQ (:1330-1406) ===============================
    // s1=DTOZ, s2=AKQ, s3=CR, s4=CM.  AKQ is still read by the bottom-level
    // block, so RSQ2 cannot share it; rhok is not loaded until the heat
    // half (:527), so it carries RSQ2 here.
    {
        const real esqhf = 0.5f * PBL_ESQ;
        int nq = lmh - 2;
        for (int m = 0; m < nq; ++m) {
            s1[m] = (dtdif + dtdif) / (zh[m] - zh[m + 2]);
            s2[m] = sqrtf((q2[m] + q2[m + 1]) * 0.5f) * (el[m] + el[m + 1])
                    * esqhf / (zh[m + 1] - zh[m + 2]);
            s3[m] = -s1[m] * s2[m];
        }
        s4[0] = s1[0] * s2[0] + 1.0f;
        rhok[0] = q2[0];
        for (int m = 1; m < nq; ++m) {
            real cf = -s1[m] * s2[m - 1] / s4[m - 1];
            s4[m] = -s3[m - 1] * cf + (s2[m - 1] + s2[m]) * s1[m] + 1.0f;
            rhok[m] = -rhok[m - 1] * cf + q2[m];
        }
        real dtozs = (dtdif + dtdif) / (zh[lmh - 2] - zh[lmh]);
        real akqs = sqrtf((q2[lmh - 2] + q2[lmh - 1]) * 0.5f)
                    * (el[lmh - 2] + PBL_ELZ0) * esqhf
                    / (zh[lmh - 1] - zh[lmh]);
        real cf = -dtozs * s2[nq - 1] / s4[nq - 1];
        q2[lmh - 2] = (dtozs * akqs * q2[lmh - 1] - rhok[nq - 1] * cf
                       + q2[lmh - 2])
                      / ((s2[nq - 1] + akqs) * dtozs - s3[nq - 1] * cf + 1.0f);
        for (int m = nq - 1; m >= 0; --m) {
            q2[m] = (-s3[m] * q2[m + 1] + rhok[m]) / s4[m];
        }
    }
    // Save the new TKE and mixing length (:459-464).
    for (int kup = 0; kup < nz; ++kup) {
        int kflip = nz - kup;                  // Fortran KFLIP=KTE+1-K
        q2[kflip - 1] = fmaxf(q2[kflip - 1], MYJP_EPSQ2);
        MYJ_UP(tke_a, kup) = 0.5f * q2[kflip - 1];
        MYJ_UP(el_myj_a, kup) = (kflip < nz) ? el[kflip - 1] : 0.0f;
    }

    // ================= main_integration, heat/moisture (:474-652) =======
    real psfc = psfc_a[col];
    real thsk = tsk_a[col] * powf(1.0e5f / psfc, PBL_CAPA);      // :477
    for (int m = 0; m < nz; ++m) {
        real pkm = MYJ_UP(p_a, nz - 1 - m);
        rhok[m] = pkm / (MYJP_RD * tk[m]
                         * (1.0f + MYJP_P608 * qk[m] - cwm[m]));  // :527-528
    }
    // AKHK (:535-537) overwrites AKH in place; AKH is not read again.
    for (int m = 0; m < nz - 1; ++m) {
        akh[m] = akh[m] * 0.5f * (rhok[m] + rhok[m + 1]);
    }
    real seamask = xland_a[col] - 1.0f;
    real thz0 = (1.0f - seamask) * thsk + seamask * thz0_a[col];  // :542
    thz0_a[col] = thz0;
    real akhs_dens = akhs_a[col] * rhok[nz - 1];                  // :545
    real qsfc = qsfc_a[col];
    if (seamask < 0.5f) {
        real qfc1 = MYJP_XLV * chklowq_a[col] * akhs_dens;
        if (snow_a[col] > 0.0f || sice_a[col] > 0.5f) qfc1 = qfc1 * PBL_RLIVWV;
        if (qfc1 > 0.0f) qsfc = qk[nz - 1] + elflx_a[col] / qfc1;
    } else {
        real exnsfc = powf(1.0e5f / psfc, PBL_CAPA);
        qsfc = PBL_PQ0SEA / psfc
               * expf(MYJP_A2 * (thsk - MYJP_A3 * exnsfc)
                      / (thsk - MYJP_A4 * exnsfc));
    }
    qsfc_a[col] = qsfc;
    real qz0 = (1.0f - seamask) * qsfc + seamask * qz0_a[col];    // :566
    qz0_a[col] = qz0;

    // ---- VDIFH (:1411-1519), coefficients once, then species by species.
    // s1=DTOZ, s3=CR, s4=CM, s2=RKHZ (the RKCT operand), rs in "the"'s row?
    // No -- the species rows ARE the unknowns.  RSS rides gm, which is dead
    // after DIFCOF.
    {
        // SPECIES(3,K) is QCW ALONE (:499,:503) while CWMK is the TOTAL
        // condensate (:498).  Both rode ``cwm`` up to here because RHOK
        // (:528) and MIXLEN want the total; the row that gets MIXED is the
        // cloud-water one, so reload it now that the total's last reader is
        // behind us.
        for (int m = 0; m < nz; ++m) cwm[m] = MYJ_UP(qc_a, nz - 1 - m);
        int nh = lmh - 1;
        for (int m = 0; m < nh; ++m) {
            s1[m] = dtdif / (zh[m] - zh[m + 1]);
            s3[m] = -s1[m] * akh[m];
            s2[m] = ((m + 1) < lpbl) ? 0.0f : akh[m] * (zh[m] - zh[m + 2]);
        }
        s4[0] = s1[0] * akh[0] + rhok[0];
        for (int m = 1; m < nh; ++m) {
            real cf = -s1[m] * akh[m - 1] / s4[m - 1];
            s4[m] = -s3[m - 1] * cf + (akh[m - 1] + akh[m]) * s1[m] + rhok[m];
        }
        real dtozs = dtdif / (zh[lmh - 1] - zh[lmh]);
        real rkhh = akh[nh - 1];
        real cfb = -dtozs * rkhh / s4[nh - 1];
        real cmb = s3[nh - 1] * cfb;
        int nspec = flqi ? 4 : 3;
        real ct_species0 = 0.0f;    // CTS(1)=CT, zeroed by MIXLEN (:852)
        for (int s = 0; s < nspec; ++s) {
            real* var = (s == 0) ? the : (s == 1) ? qk
                        : (s == 2) ? cwm : qcik;
            // SZ0/CLOW/CTS (:569-585): only rows 1 and 2 are nonzero.
            real sz0 = (s == 0) ? thz0 : (s == 1) ? qz0 : 0.0f;
            real clow = (s == 0) ? 1.0f : (s == 1) ? chklowq_a[col] : 0.0f;
            real cts = (s == 0) ? ct_species0 : 0.0f;
            // Forward sweep, RSS in gm (dead since DIFCOF).
            gm[0] = -((s2[0] * cts) * 0.5f) * s1[0] + var[0] * rhok[0];
            for (int m = 1; m < nh; ++m) {
                real cf = -s1[m] * akh[m - 1] / s4[m - 1];
                real rkct_prev = (s2[m - 1] * cts) * 0.5f;
                real rkct_cur = (s2[m] * cts) * 0.5f;
                gm[m] = -gm[m - 1] * cf + (rkct_prev - rkct_cur) * s1[m]
                        + var[m] * rhok[m];
            }
            real rkss = akhs_dens * clow;
            real cmsb = -cmb + (rkhh + rkss) * dtozs + rhok[lmh - 1];
            real rssb = -gm[nh - 1] * cfb + ((s2[nh - 1] * cts) * 0.5f) * dtozs
                        + var[lmh - 1] * rhok[lmh - 1];
            var[lmh - 1] = (dtozs * rkss * sz0 + rssb) / cmsb;
            for (int m = nh - 1; m >= 0; --m) {
                // RCML=1./CM(K) then MULTIPLY (:1512-1514): a float32
                // reciprocal-then-multiply is not a float32 divide.
                real rcml = 1.0f / s4[m];
                var[m] = (-s3[m] * var[m + 1] + gm[m]) * rcml;
            }
        }
    }
    // Primary variable tendencies (:607-652).  CWMK is rebuilt from the
    // MIXED species exactly as :610-617 does, and it is the total that
    // enters THNEW while RQCBLTEN sees the cloud-water row alone.
    for (int kup = 0; kup < nz; ++kup) {
        int m = nz - 1 - kup;
        real qci = flqi ? qcik[m] : 0.0f;
        real cwmk = cwm[m] + qci;
        real ape = 1.0f / MYJ_UP(exner_a, kup);
        real thold = MYJ_UP(th_a, kup);
        real thnew = the[m] + cwmk * PBL_ELOCP * ape;
        MYJ_UP(rthblten_a, kup) = (thnew - thold) * rdtturbl;
        real qvup = MYJ_UP(qv_a, kup);
        real qold = qvup / (1.0f + qvup);
        real dqdt = (qk[m] - qold) * rdtturbl;
        MYJ_UP(rqvblten_a, kup) = dqdt / ((1.0f - qk[m]) * (1.0f - qk[m]));
        MYJ_UP(rqcblten_a, kup) = (cwm[m] - MYJ_UP(qc_a, kup)) * rdtturbl;
        MYJ_UP(rqiblten_a, kup) = flqi
            ? (qci - MYJ_UP(qi_a, kup)) * rdtturbl : 0.0f;
    }

    // ================= main_integration, momentum (:714-757) ============
    {
        for (int m = 0; m < nz - 1; ++m) {
            akm[m] = akm[m] * (rhok[m] + rhok[m + 1]) * 0.5f;   // :722
        }
        real akms_dens = akms_a[col] * rhok[nz - 1];
        real uz0 = uz0_a[col], vz0 = vz0_a[col];
        int nv = lmh - 1;
        for (int m = 0; m < nv; ++m) {
            s1[m] = dtdif / (zh[m] - zh[m + 1]);
            s3[m] = -s1[m] * akm[m];
        }
        s4[0] = s1[0] * akm[0] + rhok[0];
        for (int m = 1; m < nv; ++m) {
            real cf = -s1[m] * akm[m - 1] / s4[m - 1];
            s4[m] = -s3[m - 1] * cf + (akm[m - 1] + akm[m]) * s1[m] + rhok[m];
        }
        real dtozs = dtdif / (zh[lmh - 1] - zh[lmh]);
        real rkmh = akm[nv - 1];
        real cfb = -dtozs * rkmh / s4[nv - 1];
        real rcmvb = 1.0f / ((rkmh + akms_dens) * dtozs - s3[nv - 1] * cfb
                             + rhok[lmh - 1]);
        real dtozak = dtozs * akms_dens;
        for (int comp = 0; comp < 2; ++comp) {
            real* var = comp ? vk : uk;
            real sz0 = comp ? vz0 : uz0;
            gm[0] = var[0] * rhok[0];                            // RSU/RSV
            for (int m = 1; m < nv; ++m) {
                real cf = -s1[m] * akm[m - 1] / s4[m - 1];
                gm[m] = -gm[m - 1] * cf + var[m] * rhok[m];
            }
            var[lmh - 1] = (dtozak * sz0 - gm[nv - 1] * cfb
                            + var[lmh - 1] * rhok[lmh - 1]) * rcmvb;
            for (int m = nv - 1; m >= 0; --m) {
                real rcml = 1.0f / s4[m];      // :1683-1685, as VDIFH
                var[m] = (-s3[m] * var[m + 1] + gm[m]) * rcml;
            }
        }
        for (int kup = 0; kup < nz; ++kup) {
            int m = nz - 1 - kup;
            MYJ_UP(rublten_a, kup) = (uk[m] - MYJ_UP(u_a, kup)) * rdtturbl;
            MYJ_UP(rvblten_a, kup) = (vk[m] - MYJ_UP(v_a, kup)) * rdtturbl;
        }
    }
#undef MYJ_UP
}
