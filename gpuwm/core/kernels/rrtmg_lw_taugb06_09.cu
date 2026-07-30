// RRTMG LW bands 6-9 -- CUDA FP32 twins of gpuwm/core/rrtmg_lw.py
// _taugb6.._taugb9 (and _nine_major), themselves bitwise-gated against
// WRF v4.6.1 module_ra_rrtmg_lw.F taugb6..taugb9 (lines 6090-6835).
// Helpers (RLW_* macros, rlw_pow/rlw_pow4, FS/IS/WXS/A2/TAUG/FRACS,
// TAUGB_PROLOGUE, SL_/SI_ slots) come from kernels/rrtmg_lw.cu; this
// file only adds the band kernels plus the 9-species spec-fac helpers
// shared by bands 7 and 9 (names suffixed _0609 to stay unique in the
// assembled translation unit).

// 3-D minor-gas table (9,19,ng), Fortran order: element (j,i,g).
#define KA3_0609(tab, j, i, g) \
    tab[((j) - 1) + 9 * ((i) - 1) + 171 * ((g) - 1)]

// specparm branch triplet shared by the 9-species bands (7 and 9).
// facp/fact are fac00/fac10 (ind0 half) or fac01/fac11 (ind1 half).
// Fortran e.g. lines 6278-6307 (band 7) / 6674-6703 (band 9).
struct RlwSpecFac0609 {
    float f0p, f1p, f2p, f0t, f1t, f2t;
};

__device__ RlwSpecFac0609 rlw_specfac_0609(float specparm, float fsl,
                                           float facp, float fact)
{
    RlwSpecFac0609 r;
    if (specparm < 0.125f) {
        float p = RLW_SU(fsl, 1.0f);              // p = fs - 1
        float p4 = rlw_pow4(p);                   // p**4
        float fk0 = p4;
        float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
        float fk2 = RLW_AD(p, p4);
        r.f0p = RLW_MU(fk0, facp);
        r.f1p = RLW_MU(fk1, facp);
        r.f2p = RLW_MU(fk2, facp);
        r.f0t = RLW_MU(fk0, fact);
        r.f1t = RLW_MU(fk1, fact);
        r.f2t = RLW_MU(fk2, fact);
    } else if (specparm > 0.875f) {
        float p = -fsl;                           // p = -fs
        float p4 = rlw_pow4(p);
        float fk0 = p4;
        float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
        float fk2 = RLW_AD(p, p4);
        r.f0p = RLW_MU(fk0, facp);
        r.f1p = RLW_MU(fk1, facp);
        r.f2p = RLW_MU(fk2, facp);
        r.f0t = RLW_MU(fk0, fact);
        r.f1t = RLW_MU(fk1, fact);
        r.f2t = RLW_MU(fk2, fact);
    } else {
        r.f0p = RLW_MU(RLW_SU(1.0f, fsl), facp);
        r.f0t = RLW_MU(RLW_SU(1.0f, fsl), fact);
        r.f1p = RLW_MU(fsl, facp);
        r.f1t = RLW_MU(fsl, fact);
        r.f2p = 0.0f;                             // unused in else branch
        r.f2t = 0.0f;
    }
    return r;
}

// tau_major sum for one g-point; absa leading dimension 585 (9-species).
// Term ORDER matches the Fortran statements (e.g. 6350-6372) exactly.
__device__ float rlw_ninemajor_0609(float specparm, float speccomb,
                                    RlwSpecFac0609 f,
                                    const float* absa, int ind, int ig)
{
    float s;
    if (specparm < 0.125f) {
        s = RLW_MU(f.f0p, A2(absa, 585, ind, ig));
        s = RLW_AD(s, RLW_MU(f.f1p, A2(absa, 585, ind + 1, ig)));
        s = RLW_AD(s, RLW_MU(f.f2p, A2(absa, 585, ind + 2, ig)));
        s = RLW_AD(s, RLW_MU(f.f0t, A2(absa, 585, ind + 9, ig)));
        s = RLW_AD(s, RLW_MU(f.f1t, A2(absa, 585, ind + 10, ig)));
        s = RLW_AD(s, RLW_MU(f.f2t, A2(absa, 585, ind + 11, ig)));
    } else if (specparm > 0.875f) {
        s = RLW_MU(f.f2p, A2(absa, 585, ind - 1, ig));
        s = RLW_AD(s, RLW_MU(f.f1p, A2(absa, 585, ind, ig)));
        s = RLW_AD(s, RLW_MU(f.f0p, A2(absa, 585, ind + 1, ig)));
        s = RLW_AD(s, RLW_MU(f.f2t, A2(absa, 585, ind + 8, ig)));
        s = RLW_AD(s, RLW_MU(f.f1t, A2(absa, 585, ind + 9, ig)));
        s = RLW_AD(s, RLW_MU(f.f0t, A2(absa, 585, ind + 10, ig)));
    } else {
        s = RLW_MU(f.f0p, A2(absa, 585, ind, ig));
        s = RLW_AD(s, RLW_MU(f.f1p, A2(absa, 585, ind + 1, ig)));
        s = RLW_AD(s, RLW_MU(f.f0t, A2(absa, 585, ind + 9, ig)));
        s = RLW_AD(s, RLW_MU(f.f1t, A2(absa, 585, ind + 10, ig)));
    }
    return RLW_MU(speccomb, s);
}

// ---------------------------------------------------------------------
// Band 6: 820-980 cm-1 (low key h2o; low minor co2; upper NOTHING but
// the cfc11adj/cfc12 cross-section terms).  Fortran 6090-6175.
// ---------------------------------------------------------------------

extern "C" __global__ void rlw_taugb6(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    // GPU_BAND_TABS[6] = absa(65,ng), selfref(10,ng), forref(4,ng),
    //                    ka_mco2(19,ng), cfc11adj(ng), cfc12(ng),
    //                    fracrefa(ng)
    const float* absa = tabs[0];
    const float* selfref = tabs[1];
    const float* forref = tabs[2];
    const float* ka_mco2 = tabs[3];
    const float* cfc11adj = tabs[4];
    const float* cfc12 = tabs[5];
    const float* fracrefa = tabs[6];
    const int nspa6 = 1, ng6 = 8, gs = 68;

    if (lay <= laytrop) {
        int jpv = IS(SI_JP);
        float chi_co2 = RLW_DV(FS(SL_COLCO2), FS(SL_COLDRY));
        // ratco2 = 1.e20_rb*chi_co2/chi_mls(2,jp(lay)+1)
        float ratco2 = RLW_DV(RLW_MU(1.e20f, chi_co2),
                              chi_mls[1 + 7 * jpv]);
        float adjcolco2;
        if (ratco2 > 3.0f) {
            float adjfac = RLW_AD(2.0f,
                                  rlw_pow(RLW_SU(ratco2, 2.0f), 0.77f));
            adjcolco2 = RLW_MU(RLW_MU(RLW_MU(adjfac, chi_mls[1 + 7 * jpv]),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcolco2 = FS(SL_COLCO2);
        }

        int ind0 = ((jpv - 1) * 5 + (IS(SI_JT) - 1)) * nspa6 + 1;
        int ind1 = (jpv * 5 + (IS(SI_JT1) - 1)) * nspa6 + 1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        for (int ig = 1; ig <= ng6; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float absco2 = RLW_AD(A2(ka_mco2, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(ka_mco2, 19, indm + 1, ig),
                              A2(ka_mco2, 19, indm, ig))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absa, 65, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10),
                                       A2(absa, 65, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01),
                                       A2(absa, 65, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11),
                                       A2(absa, 65, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, RLW_MU(adjcolco2, absco2));
            t = RLW_AD(t, RLW_MU(WXS(2), cfc11adj[ig - 1]));
            t = RLW_AD(t, RLW_MU(WXS(3), cfc12[ig - 1]));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefa[ig - 1];
        }
    } else {
        // Nothing important goes on above laytrop in this band.
        for (int ig = 1; ig <= ng6; ++ig) {
            float t = RLW_AD(0.0f, RLW_MU(WXS(2), cfc11adj[ig - 1]));
            t = RLW_AD(t, RLW_MU(WXS(3), cfc12[ig - 1]));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefa[ig - 1];
        }
    }
}

// ---------------------------------------------------------------------
// Band 7: 980-1080 cm-1 (low key h2o,o3 + co2 minor; high key o3 + co2
// minor + empirical g-point rescale).  Fortran 6178-6449.
// ---------------------------------------------------------------------

extern "C" __global__ void rlw_taugb7(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    // GPU_BAND_TABS[7] = absa(585,ng), absb(235,ng), selfref(10,ng),
    //                    forref(4,ng), ka_mco2(9,19,ng), kb_mco2(19,ng),
    //                    fracrefa(ng,9), fracrefb(ng)
    const float* absa = tabs[0];
    const float* absb = tabs[1];
    const float* selfref = tabs[2];
    const float* forref = tabs[3];
    const float* ka_mco2 = tabs[4];
    const float* kb_mco2 = tabs[5];
    const float* fracrefa = tabs[6];
    const float* fracrefb = tabs[7];
    const int nspa7 = 9, nspb7 = 1, ng7 = 12, gs = 76;

    if (lay <= laytrop) {
        // P = 706.2620 mb / 706.2720 mb (identical expressions in this
        // source): refrat_planck_a = refrat_m_a = chi_mls(1,3)/chi_mls(3,3)
        float refrat_planck_a = RLW_DV(chi_mls[0 + 7 * 2],
                                       chi_mls[2 + 7 * 2]);
        float refrat_m_a = RLW_DV(chi_mls[0 + 7 * 2], chi_mls[2 + 7 * 2]);

        int jpv = IS(SI_JP);
        float colh2o = FS(SL_COLH2O);
        float colo3 = FS(SL_COLO3);

        float speccomb = RLW_AD(colh2o, RLW_MU(FS(SL_RAT_H2OO3), colo3));
        float specparm = RLW_DV(colh2o, speccomb);
        if (specparm >= oneminus) specparm = oneminus;
        float specmult = RLW_MU(8.0f, specparm);
        int js = 1 + (int)specmult;
        float fsv = fmodf(specmult, 1.0f);

        float speccomb1 = RLW_AD(colh2o,
                                 RLW_MU(FS(SL_RAT_H2OO3_1), colo3));
        float specparm1 = RLW_DV(colh2o, speccomb1);
        if (specparm1 >= oneminus) specparm1 = oneminus;
        float specmult1 = RLW_MU(8.0f, specparm1);
        int js1 = 1 + (int)specmult1;
        float fs1v = fmodf(specmult1, 1.0f);

        float speccomb_mco2 = RLW_AD(colh2o, RLW_MU(refrat_m_a, colo3));
        float specparm_mco2 = RLW_DV(colh2o, speccomb_mco2);
        if (specparm_mco2 >= oneminus) specparm_mco2 = oneminus;
        float specmult_mco2 = RLW_MU(8.0f, specparm_mco2);
        int jmco2 = 1 + (int)specmult_mco2;
        float fmco2 = fmodf(specmult_mco2, 1.0f);

        float chi_co2 = RLW_DV(FS(SL_COLCO2), FS(SL_COLDRY));
        // ratco2 = 1.e20*chi_co2/chi_mls(2,jp(lay)+1)  (no _rb in source)
        float ratco2 = RLW_DV(RLW_MU(1.e20f, chi_co2),
                              chi_mls[1 + 7 * jpv]);
        float adjcolco2;
        if (ratco2 > 3.0f) {
            float adjfac = RLW_AD(3.0f,
                                  rlw_pow(RLW_SU(ratco2, 3.0f), 0.79f));
            adjcolco2 = RLW_MU(RLW_MU(RLW_MU(adjfac, chi_mls[1 + 7 * jpv]),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcolco2 = FS(SL_COLCO2);
        }

        float speccomb_planck = RLW_AD(colh2o,
                                       RLW_MU(refrat_planck_a, colo3));
        float specparm_planck = RLW_DV(colh2o, speccomb_planck);
        if (specparm_planck >= oneminus) specparm_planck = oneminus;
        float specmult_planck = RLW_MU(8.0f, specparm_planck);
        int jpl = 1 + (int)specmult_planck;
        float fpl = fmodf(specmult_planck, 1.0f);

        int ind0 = ((jpv - 1) * 5 + (IS(SI_JT) - 1)) * nspa7 + js;
        int ind1 = (jpv * 5 + (IS(SI_JT1) - 1)) * nspa7 + js1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        RlwSpecFac0609 f0 = rlw_specfac_0609(specparm, fsv,
                                             FS(SL_FAC00), FS(SL_FAC10));
        RlwSpecFac0609 f1 = rlw_specfac_0609(specparm1, fs1v,
                                             FS(SL_FAC01), FS(SL_FAC11));

        for (int ig = 1; ig <= ng7; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float co2m1 = RLW_AD(KA3_0609(ka_mco2, jmco2, indm, ig),
                RLW_MU(fmco2,
                       RLW_SU(KA3_0609(ka_mco2, jmco2 + 1, indm, ig),
                              KA3_0609(ka_mco2, jmco2, indm, ig))));
            float co2m2 = RLW_AD(KA3_0609(ka_mco2, jmco2, indm + 1, ig),
                RLW_MU(fmco2,
                       RLW_SU(KA3_0609(ka_mco2, jmco2 + 1, indm + 1, ig),
                              KA3_0609(ka_mco2, jmco2, indm + 1, ig))));
            float absco2 = RLW_AD(co2m1, RLW_MU(FS(SL_MINORFRAC),
                                                RLW_SU(co2m2, co2m1)));

            float tau_major = rlw_ninemajor_0609(specparm, speccomb, f0,
                                                 absa, ind0, ig);
            float tau_major1 = rlw_ninemajor_0609(specparm1, speccomb1, f1,
                                                  absa, ind1, ig);

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, RLW_MU(adjcolco2, absco2));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefa, 12, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefa, 12, ig, jpl + 1),
                                   A2(fracrefa, 12, ig, jpl))));
        }
    } else {
        int jpv = IS(SI_JP);
        float chi_co2 = RLW_DV(FS(SL_COLCO2), FS(SL_COLDRY));
        float ratco2 = RLW_DV(RLW_MU(1.e20f, chi_co2),
                              chi_mls[1 + 7 * jpv]);
        float adjcolco2;
        if (ratco2 > 3.0f) {
            float adjfac = RLW_AD(2.0f,
                                  rlw_pow(RLW_SU(ratco2, 2.0f), 0.79f));
            adjcolco2 = RLW_MU(RLW_MU(RLW_MU(adjfac, chi_mls[1 + 7 * jpv]),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcolco2 = FS(SL_COLCO2);
        }

        int ind0 = ((jpv - 13) * 5 + (IS(SI_JT) - 1)) * nspb7 + 1;
        int ind1 = ((jpv - 12) * 5 + (IS(SI_JT1) - 1)) * nspb7 + 1;
        int indm = IS(SI_INDMINOR);

        for (int ig = 1; ig <= ng7; ++ig) {
            float absco2 = RLW_AD(A2(kb_mco2, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(kb_mco2, 19, indm + 1, ig),
                              A2(kb_mco2, 19, indm, ig))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10),
                                       A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01),
                                       A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11),
                                       A2(absb, 235, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLO3), tmaj);
            t = RLW_AD(t, RLW_MU(adjcolco2, absco2));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefb[ig - 1];
        }

        // Empirical modification to improve stratospheric cooling rates
        // for o3 (reduced g-points 6..11 of this band).
        TAUG(gs + 6) = RLW_MU(TAUG(gs + 6), 0.92f);
        TAUG(gs + 7) = RLW_MU(TAUG(gs + 7), 0.88f);
        TAUG(gs + 8) = RLW_MU(TAUG(gs + 8), 1.07f);
        TAUG(gs + 9) = RLW_MU(TAUG(gs + 9), 1.1f);
        TAUG(gs + 10) = RLW_MU(TAUG(gs + 10), 0.99f);
        TAUG(gs + 11) = RLW_MU(TAUG(gs + 11), 0.855f);
    }
}

// ---------------------------------------------------------------------
// Band 8: 1080-1180 cm-1 (low key h2o + co2/o3/n2o minors + cfc12/
// cfc22adj; high key o3 + co2/n2o minors + cfcs).  Fortran 6452-6572.
// ---------------------------------------------------------------------

extern "C" __global__ void rlw_taugb8(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    // GPU_BAND_TABS[8] = absa(65,ng), absb(235,ng), selfref(10,ng),
    //                    forref(4,ng), ka_mco2(19,ng), kb_mco2(19,ng),
    //                    ka_mn2o(19,ng), kb_mn2o(19,ng), ka_mo3(19,ng),
    //                    cfc12(ng), cfc22adj(ng), fracrefa(ng),
    //                    fracrefb(ng)
    const float* absa = tabs[0];
    const float* absb = tabs[1];
    const float* selfref = tabs[2];
    const float* forref = tabs[3];
    const float* ka_mco2 = tabs[4];
    const float* kb_mco2 = tabs[5];
    const float* ka_mn2o = tabs[6];
    const float* kb_mn2o = tabs[7];
    const float* ka_mo3 = tabs[8];
    const float* cfc12 = tabs[9];
    const float* cfc22adj = tabs[10];
    const float* fracrefa = tabs[11];
    const float* fracrefb = tabs[12];
    const int nspa8 = 1, nspb8 = 1, ng8 = 8, gs = 88;

    if (lay <= laytrop) {
        int jpv = IS(SI_JP);
        float chi_co2 = RLW_DV(FS(SL_COLCO2), FS(SL_COLDRY));
        float ratco2 = RLW_DV(RLW_MU(1.e20f, chi_co2),
                              chi_mls[1 + 7 * jpv]);
        float adjcolco2;
        if (ratco2 > 3.0f) {
            float adjfac = RLW_AD(2.0f,
                                  rlw_pow(RLW_SU(ratco2, 2.0f), 0.65f));
            adjcolco2 = RLW_MU(RLW_MU(RLW_MU(adjfac, chi_mls[1 + 7 * jpv]),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcolco2 = FS(SL_COLCO2);
        }

        int ind0 = ((jpv - 1) * 5 + (IS(SI_JT) - 1)) * nspa8 + 1;
        int ind1 = (jpv * 5 + (IS(SI_JT1) - 1)) * nspa8 + 1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        for (int ig = 1; ig <= ng8; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float absco2 = RLW_AD(A2(ka_mco2, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(ka_mco2, 19, indm + 1, ig),
                              A2(ka_mco2, 19, indm, ig))));
            float abso3 = RLW_AD(A2(ka_mo3, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(ka_mo3, 19, indm + 1, ig),
                              A2(ka_mo3, 19, indm, ig))));
            float absn2o = RLW_AD(A2(ka_mn2o, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(ka_mn2o, 19, indm + 1, ig),
                              A2(ka_mn2o, 19, indm, ig))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absa, 65, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10),
                                       A2(absa, 65, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01),
                                       A2(absa, 65, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11),
                                       A2(absa, 65, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, RLW_MU(adjcolco2, absco2));
            t = RLW_AD(t, RLW_MU(FS(SL_COLO3), abso3));
            t = RLW_AD(t, RLW_MU(FS(SL_COLN2O), absn2o));
            t = RLW_AD(t, RLW_MU(WXS(3), cfc12[ig - 1]));
            t = RLW_AD(t, RLW_MU(WXS(4), cfc22adj[ig - 1]));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefa[ig - 1];
        }
    } else {
        int jpv = IS(SI_JP);
        float chi_co2 = RLW_DV(FS(SL_COLCO2), FS(SL_COLDRY));
        float ratco2 = RLW_DV(RLW_MU(1.e20f, chi_co2),
                              chi_mls[1 + 7 * jpv]);
        float adjcolco2;
        if (ratco2 > 3.0f) {
            float adjfac = RLW_AD(2.0f,
                                  rlw_pow(RLW_SU(ratco2, 2.0f), 0.65f));
            adjcolco2 = RLW_MU(RLW_MU(RLW_MU(adjfac, chi_mls[1 + 7 * jpv]),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcolco2 = FS(SL_COLCO2);
        }

        int ind0 = ((jpv - 13) * 5 + (IS(SI_JT) - 1)) * nspb8 + 1;
        int ind1 = ((jpv - 12) * 5 + (IS(SI_JT1) - 1)) * nspb8 + 1;
        int indm = IS(SI_INDMINOR);

        for (int ig = 1; ig <= ng8; ++ig) {
            float absco2 = RLW_AD(A2(kb_mco2, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(kb_mco2, 19, indm + 1, ig),
                              A2(kb_mco2, 19, indm, ig))));
            float absn2o = RLW_AD(A2(kb_mn2o, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(kb_mn2o, 19, indm + 1, ig),
                              A2(kb_mn2o, 19, indm, ig))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10),
                                       A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01),
                                       A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11),
                                       A2(absb, 235, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLO3), tmaj);
            t = RLW_AD(t, RLW_MU(adjcolco2, absco2));
            t = RLW_AD(t, RLW_MU(FS(SL_COLN2O), absn2o));
            t = RLW_AD(t, RLW_MU(WXS(3), cfc12[ig - 1]));
            t = RLW_AD(t, RLW_MU(WXS(4), cfc22adj[ig - 1]));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }
}

// ---------------------------------------------------------------------
// Band 9: 1180-1390 cm-1 (low key h2o,ch4 + n2o minor; high key ch4 +
// n2o minor).  Fortran 6575-6835.
// ---------------------------------------------------------------------

extern "C" __global__ void rlw_taugb9(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    // GPU_BAND_TABS[9] = absa(585,ng), absb(235,ng), selfref(10,ng),
    //                    forref(4,ng), ka_mn2o(9,19,ng), kb_mn2o(19,ng),
    //                    fracrefa(ng,9), fracrefb(ng)
    const float* absa = tabs[0];
    const float* absb = tabs[1];
    const float* selfref = tabs[2];
    const float* forref = tabs[3];
    const float* ka_mn2o = tabs[4];
    const float* kb_mn2o = tabs[5];
    const float* fracrefa = tabs[6];
    const float* fracrefb = tabs[7];
    const int nspa9 = 9, nspb9 = 1, ng9 = 12, gs = 96;

    if (lay <= laytrop) {
        // P = 212 mb: refrat_planck_a = chi_mls(1,9)/chi_mls(6,9)
        float refrat_planck_a = RLW_DV(chi_mls[0 + 7 * 8],
                                       chi_mls[5 + 7 * 8]);
        // P = 706.272 mb: refrat_m_a = chi_mls(1,3)/chi_mls(6,3)
        float refrat_m_a = RLW_DV(chi_mls[0 + 7 * 2], chi_mls[5 + 7 * 2]);

        int jpv = IS(SI_JP);
        float colh2o = FS(SL_COLH2O);
        float colch4 = FS(SL_COLCH4);

        float speccomb = RLW_AD(colh2o, RLW_MU(FS(SL_RAT_H2OCH4), colch4));
        float specparm = RLW_DV(colh2o, speccomb);
        if (specparm >= oneminus) specparm = oneminus;
        float specmult = RLW_MU(8.0f, specparm);
        int js = 1 + (int)specmult;
        float fsv = fmodf(specmult, 1.0f);

        float speccomb1 = RLW_AD(colh2o,
                                 RLW_MU(FS(SL_RAT_H2OCH4_1), colch4));
        float specparm1 = RLW_DV(colh2o, speccomb1);
        if (specparm1 >= oneminus) specparm1 = oneminus;
        float specmult1 = RLW_MU(8.0f, specparm1);
        int js1 = 1 + (int)specmult1;
        float fs1v = fmodf(specmult1, 1.0f);

        float speccomb_mn2o = RLW_AD(colh2o, RLW_MU(refrat_m_a, colch4));
        float specparm_mn2o = RLW_DV(colh2o, speccomb_mn2o);
        if (specparm_mn2o >= oneminus) specparm_mn2o = oneminus;
        float specmult_mn2o = RLW_MU(8.0f, specparm_mn2o);
        int jmn2o = 1 + (int)specmult_mn2o;
        float fmn2o = fmodf(specmult_mn2o, 1.0f);

        float chi_n2o = RLW_DV(FS(SL_COLN2O), FS(SL_COLDRY));
        // ratn2o = 1.e20_rb*chi_n2o/chi_mls(4,jp(lay)+1)
        float ratn2o = RLW_DV(RLW_MU(1.e20f, chi_n2o),
                              chi_mls[3 + 7 * jpv]);
        float adjcoln2o;
        if (ratn2o > 1.5f) {
            float adjfac = RLW_AD(0.5f,
                                  rlw_pow(RLW_SU(ratn2o, 0.5f), 0.65f));
            adjcoln2o = RLW_MU(RLW_MU(RLW_MU(adjfac, chi_mls[3 + 7 * jpv]),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcoln2o = FS(SL_COLN2O);
        }

        float speccomb_planck = RLW_AD(colh2o,
                                       RLW_MU(refrat_planck_a, colch4));
        float specparm_planck = RLW_DV(colh2o, speccomb_planck);
        if (specparm_planck >= oneminus) specparm_planck = oneminus;
        float specmult_planck = RLW_MU(8.0f, specparm_planck);
        int jpl = 1 + (int)specmult_planck;
        float fpl = fmodf(specmult_planck, 1.0f);

        int ind0 = ((jpv - 1) * 5 + (IS(SI_JT) - 1)) * nspa9 + js;
        int ind1 = (jpv * 5 + (IS(SI_JT1) - 1)) * nspa9 + js1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        RlwSpecFac0609 f0 = rlw_specfac_0609(specparm, fsv,
                                             FS(SL_FAC00), FS(SL_FAC10));
        RlwSpecFac0609 f1 = rlw_specfac_0609(specparm1, fs1v,
                                             FS(SL_FAC01), FS(SL_FAC11));

        for (int ig = 1; ig <= ng9; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float n2om1 = RLW_AD(KA3_0609(ka_mn2o, jmn2o, indm, ig),
                RLW_MU(fmn2o,
                       RLW_SU(KA3_0609(ka_mn2o, jmn2o + 1, indm, ig),
                              KA3_0609(ka_mn2o, jmn2o, indm, ig))));
            float n2om2 = RLW_AD(KA3_0609(ka_mn2o, jmn2o, indm + 1, ig),
                RLW_MU(fmn2o,
                       RLW_SU(KA3_0609(ka_mn2o, jmn2o + 1, indm + 1, ig),
                              KA3_0609(ka_mn2o, jmn2o, indm + 1, ig))));
            float absn2o = RLW_AD(n2om1, RLW_MU(FS(SL_MINORFRAC),
                                                RLW_SU(n2om2, n2om1)));

            float tau_major = rlw_ninemajor_0609(specparm, speccomb, f0,
                                                 absa, ind0, ig);
            float tau_major1 = rlw_ninemajor_0609(specparm1, speccomb1, f1,
                                                  absa, ind1, ig);

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, RLW_MU(adjcoln2o, absn2o));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefa, 12, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefa, 12, ig, jpl + 1),
                                   A2(fracrefa, 12, ig, jpl))));
        }
    } else {
        int jpv = IS(SI_JP);
        float chi_n2o = RLW_DV(FS(SL_COLN2O), FS(SL_COLDRY));
        float ratn2o = RLW_DV(RLW_MU(1.e20f, chi_n2o),
                              chi_mls[3 + 7 * jpv]);
        float adjcoln2o;
        if (ratn2o > 1.5f) {
            float adjfac = RLW_AD(0.5f,
                                  rlw_pow(RLW_SU(ratn2o, 0.5f), 0.65f));
            adjcoln2o = RLW_MU(RLW_MU(RLW_MU(adjfac, chi_mls[3 + 7 * jpv]),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcoln2o = FS(SL_COLN2O);
        }

        int ind0 = ((jpv - 13) * 5 + (IS(SI_JT) - 1)) * nspb9 + 1;
        int ind1 = ((jpv - 12) * 5 + (IS(SI_JT1) - 1)) * nspb9 + 1;
        int indm = IS(SI_INDMINOR);

        for (int ig = 1; ig <= ng9; ++ig) {
            float absn2o = RLW_AD(A2(kb_mn2o, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(kb_mn2o, 19, indm + 1, ig),
                              A2(kb_mn2o, 19, indm, ig))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10),
                                       A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01),
                                       A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11),
                                       A2(absb, 235, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLCH4), tmaj);
            t = RLW_AD(t, RLW_MU(adjcoln2o, absn2o));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }
}

#undef KA3_0609
