// WRF v4.6.1 legacy RRTMG longwave -- CUDA FP32 twins of the NumPy
// references _taugb2/_taugb10/_taugb11/_taugb12 in gpuwm/core/rrtmg_lw.py,
// gated at max_ulp 0 against the unmodified-Fortran fixtures.
//
// Fortran authority: module_ra_rrtmg_lw.F lines 5169-5238 (taugb2) and
// 6838-7186 (taugb10/11/12).  This file is assembled after
// kernels/rrtmg_lw.cu into one translation unit; it reuses (never
// redefines) RLW_AD/SU/MU/DV, rlw_pow4, FS/IS/A2/TAUG/FRACS,
// TAUGB_PROLOGUE and the SL_/SI_ slot defines from that file.
//
// Band 2:  350-500 cm-1, low/high key h2o; lower region has
//          corradj = 1 - .05*(pp - 100)/900, upper region none.
// Band 10: 1390-1480 cm-1, low/high key h2o; no corradj, no minors.
// Band 11: 1480-1800 cm-1, low/high key h2o; o2 minor
//          (scaleo2 = colo2*scaleminor) in BOTH regions.
// Band 12: 1800-2080 cm-1, lower-only 9-species h2o/co2 (absa ld 585,
//          nspa(12)=9) with the specparm branch triplet, chi_mls Planck
//          ratio and oneminus clamp; the upper region writes exact 0.0f
//          to taug and fracs, transcribing the Fortran loop literally.

extern "C" __global__ void rlw_taugb2(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    // GPU_BAND_TABS[2] = absa(65,ng), absb(235,ng), selfref(10,ng),
    //                    forref(4,ng), fracrefa(ng), fracrefb(ng)
    const float* absa = tabs[0];
    const float* absb = tabs[1];
    const float* selfref = tabs[2];
    const float* forref = tabs[3];
    const float* fracrefa = tabs[4];
    const float* fracrefb = tabs[5];
    const int nspa2 = 1, nspb2 = 1, ng2 = 12, gs = 10;

    if (lay <= laytrop) {
        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa2 + 1;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa2 + 1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        float pp = FS(SL_PAVEL);
        // corradj = 1._rb - .05_rb * (pp - 100._rb) / 900._rb
        float corradj = RLW_SU(1.0f,
            RLW_DV(RLW_MU(0.05f, RLW_SU(pp, 100.0f)), 900.0f));
        for (int ig = 1; ig <= ng2; ++ig) {
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
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absa, 65, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10), A2(absa, 65, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01), A2(absa, 65, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11), A2(absa, 65, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            TAUG(gs + ig) = RLW_MU(corradj, t);
            FRACS(gs + ig) = fracrefa[ig - 1];
        }
    } else {
        int ind0 = ((IS(SI_JP) - 13) * 5 + (IS(SI_JT) - 1)) * nspb2 + 1;
        int ind1 = ((IS(SI_JP) - 12) * 5 + (IS(SI_JT1) - 1)) * nspb2 + 1;
        int indf = IS(SI_INDFOR);
        for (int ig = 1; ig <= ng2; ++ig) {
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10), A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01), A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11), A2(absb, 235, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, taufor);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }
}

extern "C" __global__ void rlw_taugb10(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    // GPU_BAND_TABS[10] = absa(65,ng), absb(235,ng), selfref(10,ng),
    //                     forref(4,ng), fracrefa(ng), fracrefb(ng)
    const float* absa = tabs[0];
    const float* absb = tabs[1];
    const float* selfref = tabs[2];
    const float* forref = tabs[3];
    const float* fracrefa = tabs[4];
    const float* fracrefb = tabs[5];
    const int nspa10 = 1, nspb10 = 1, ng10 = 6, gs = 108;

    if (lay <= laytrop) {
        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa10 + 1;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa10 + 1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        for (int ig = 1; ig <= ng10; ++ig) {
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
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absa, 65, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10), A2(absa, 65, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01), A2(absa, 65, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11), A2(absa, 65, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefa[ig - 1];
        }
    } else {
        int ind0 = ((IS(SI_JP) - 13) * 5 + (IS(SI_JT) - 1)) * nspb10 + 1;
        int ind1 = ((IS(SI_JP) - 12) * 5 + (IS(SI_JT1) - 1)) * nspb10 + 1;
        int indf = IS(SI_INDFOR);
        for (int ig = 1; ig <= ng10; ++ig) {
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10), A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01), A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11), A2(absb, 235, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, taufor);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }
}

extern "C" __global__ void rlw_taugb11(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    // GPU_BAND_TABS[11] = absa(65,ng), absb(235,ng), selfref(10,ng),
    //                     forref(4,ng), ka_mo2(19,ng), kb_mo2(19,ng),
    //                     fracrefa(ng), fracrefb(ng)
    const float* absa = tabs[0];
    const float* absb = tabs[1];
    const float* selfref = tabs[2];
    const float* forref = tabs[3];
    const float* ka_mo2 = tabs[4];
    const float* kb_mo2 = tabs[5];
    const float* fracrefa = tabs[6];
    const float* fracrefb = tabs[7];
    const int nspa11 = 1, nspb11 = 1, ng11 = 8, gs = 114;

    if (lay <= laytrop) {
        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa11 + 1;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa11 + 1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);
        float scaleo2 = RLW_MU(FS(SL_COLO2), FS(SL_SCALEMINOR));
        for (int ig = 1; ig <= ng11; ++ig) {
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
            float tauo2 = RLW_MU(scaleo2,
                RLW_AD(A2(ka_mo2, 19, indm, ig),
                       RLW_MU(FS(SL_MINORFRAC),
                              RLW_SU(A2(ka_mo2, 19, indm + 1, ig),
                                     A2(ka_mo2, 19, indm, ig)))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absa, 65, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10), A2(absa, 65, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01), A2(absa, 65, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11), A2(absa, 65, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, tauo2);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefa[ig - 1];
        }
    } else {
        int ind0 = ((IS(SI_JP) - 13) * 5 + (IS(SI_JT) - 1)) * nspb11 + 1;
        int ind1 = ((IS(SI_JP) - 12) * 5 + (IS(SI_JT1) - 1)) * nspb11 + 1;
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);
        float scaleo2 = RLW_MU(FS(SL_COLO2), FS(SL_SCALEMINOR));
        for (int ig = 1; ig <= ng11; ++ig) {
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float tauo2 = RLW_MU(scaleo2,
                RLW_AD(A2(kb_mo2, 19, indm, ig),
                       RLW_MU(FS(SL_MINORFRAC),
                              RLW_SU(A2(kb_mo2, 19, indm + 1, ig),
                                     A2(kb_mo2, 19, indm, ig)))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10), A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01), A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11), A2(absb, 235, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, tauo2);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }
}

extern "C" __global__ void rlw_taugb12(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    // GPU_BAND_TABS[12] = absa(585,ng), selfref(10,ng), forref(4,ng),
    //                     fracrefa(ng12,9)
    const float* absa = tabs[0];
    const float* selfref = tabs[1];
    const float* forref = tabs[2];
    const float* fracrefa = tabs[3];
    const int nspa12 = 9, ng12 = 8, gs = 122;

    if (lay <= laytrop) {
        // refrat_planck_a = chi_mls(1,10)/chi_mls(2,10)   (P = 174.164 mb)
        float refrat_planck_a = RLW_DV(chi_mls[0 + 7 * 9],
                                       chi_mls[1 + 7 * 9]);
        float colh2o = FS(SL_COLH2O);
        float colco2 = FS(SL_COLCO2);

        float speccomb = RLW_AD(colh2o, RLW_MU(FS(SL_RAT_H2OCO2), colco2));
        float specparm = RLW_DV(colh2o, speccomb);
        if (specparm >= oneminus) specparm = oneminus;
        float specmult = RLW_MU(8.0f, specparm);
        int js = 1 + (int)specmult;
        float fs0 = fmodf(specmult, 1.0f);

        float speccomb1 = RLW_AD(colh2o, RLW_MU(FS(SL_RAT_H2OCO2_1), colco2));
        float specparm1 = RLW_DV(colh2o, speccomb1);
        if (specparm1 >= oneminus) specparm1 = oneminus;
        float specmult1 = RLW_MU(8.0f, specparm1);
        int js1 = 1 + (int)specmult1;
        float fs1 = fmodf(specmult1, 1.0f);

        float speccomb_planck = RLW_AD(colh2o,
                                       RLW_MU(refrat_planck_a, colco2));
        float specparm_planck = RLW_DV(colh2o, speccomb_planck);
        if (specparm_planck >= oneminus) specparm_planck = oneminus;
        float specmult_planck = RLW_MU(8.0f, specparm_planck);
        int jpl = 1 + (int)specmult_planck;
        float fpl = fmodf(specmult_planck, 1.0f);

        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa12 + js;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa12 + js1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);

        float fac000 = 0.0f, fac100 = 0.0f, fac200 = 0.0f;
        float fac010 = 0.0f, fac110 = 0.0f, fac210 = 0.0f;
        if (specparm < 0.125f) {
            float p = RLW_SU(fs0, 1.0f);
            float p4 = rlw_pow4(p);
            float fk0 = p4;
            float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
            float fk2 = RLW_AD(p, p4);
            fac000 = RLW_MU(fk0, FS(SL_FAC00));
            fac100 = RLW_MU(fk1, FS(SL_FAC00));
            fac200 = RLW_MU(fk2, FS(SL_FAC00));
            fac010 = RLW_MU(fk0, FS(SL_FAC10));
            fac110 = RLW_MU(fk1, FS(SL_FAC10));
            fac210 = RLW_MU(fk2, FS(SL_FAC10));
        } else if (specparm > 0.875f) {
            float p = -fs0;
            float p4 = rlw_pow4(p);
            float fk0 = p4;
            float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
            float fk2 = RLW_AD(p, p4);
            fac000 = RLW_MU(fk0, FS(SL_FAC00));
            fac100 = RLW_MU(fk1, FS(SL_FAC00));
            fac200 = RLW_MU(fk2, FS(SL_FAC00));
            fac010 = RLW_MU(fk0, FS(SL_FAC10));
            fac110 = RLW_MU(fk1, FS(SL_FAC10));
            fac210 = RLW_MU(fk2, FS(SL_FAC10));
        } else {
            fac000 = RLW_MU(RLW_SU(1.0f, fs0), FS(SL_FAC00));
            fac010 = RLW_MU(RLW_SU(1.0f, fs0), FS(SL_FAC10));
            fac100 = RLW_MU(fs0, FS(SL_FAC00));
            fac110 = RLW_MU(fs0, FS(SL_FAC10));
        }

        float fac001 = 0.0f, fac101 = 0.0f, fac201 = 0.0f;
        float fac011 = 0.0f, fac111 = 0.0f, fac211 = 0.0f;
        if (specparm1 < 0.125f) {
            float p = RLW_SU(fs1, 1.0f);
            float p4 = rlw_pow4(p);
            float fk0 = p4;
            float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
            float fk2 = RLW_AD(p, p4);
            fac001 = RLW_MU(fk0, FS(SL_FAC01));
            fac101 = RLW_MU(fk1, FS(SL_FAC01));
            fac201 = RLW_MU(fk2, FS(SL_FAC01));
            fac011 = RLW_MU(fk0, FS(SL_FAC11));
            fac111 = RLW_MU(fk1, FS(SL_FAC11));
            fac211 = RLW_MU(fk2, FS(SL_FAC11));
        } else if (specparm1 > 0.875f) {
            float p = -fs1;
            float p4 = rlw_pow4(p);
            float fk0 = p4;
            float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
            float fk2 = RLW_AD(p, p4);
            fac001 = RLW_MU(fk0, FS(SL_FAC01));
            fac101 = RLW_MU(fk1, FS(SL_FAC01));
            fac201 = RLW_MU(fk2, FS(SL_FAC01));
            fac011 = RLW_MU(fk0, FS(SL_FAC11));
            fac111 = RLW_MU(fk1, FS(SL_FAC11));
            fac211 = RLW_MU(fk2, FS(SL_FAC11));
        } else {
            fac001 = RLW_MU(RLW_SU(1.0f, fs1), FS(SL_FAC01));
            fac011 = RLW_MU(RLW_SU(1.0f, fs1), FS(SL_FAC11));
            fac101 = RLW_MU(fs1, FS(SL_FAC01));
            fac111 = RLW_MU(fs1, FS(SL_FAC11));
        }

        for (int ig = 1; ig <= ng12; ++ig) {
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

            float tau_major;
            if (specparm < 0.125f) {
                float t = RLW_MU(fac000, A2(absa, 585, ind0, ig));
                t = RLW_AD(t, RLW_MU(fac100, A2(absa, 585, ind0 + 1, ig)));
                t = RLW_AD(t, RLW_MU(fac200, A2(absa, 585, ind0 + 2, ig)));
                t = RLW_AD(t, RLW_MU(fac010, A2(absa, 585, ind0 + 9, ig)));
                t = RLW_AD(t, RLW_MU(fac110, A2(absa, 585, ind0 + 10, ig)));
                t = RLW_AD(t, RLW_MU(fac210, A2(absa, 585, ind0 + 11, ig)));
                tau_major = RLW_MU(speccomb, t);
            } else if (specparm > 0.875f) {
                float t = RLW_MU(fac200, A2(absa, 585, ind0 - 1, ig));
                t = RLW_AD(t, RLW_MU(fac100, A2(absa, 585, ind0, ig)));
                t = RLW_AD(t, RLW_MU(fac000, A2(absa, 585, ind0 + 1, ig)));
                t = RLW_AD(t, RLW_MU(fac210, A2(absa, 585, ind0 + 8, ig)));
                t = RLW_AD(t, RLW_MU(fac110, A2(absa, 585, ind0 + 9, ig)));
                t = RLW_AD(t, RLW_MU(fac010, A2(absa, 585, ind0 + 10, ig)));
                tau_major = RLW_MU(speccomb, t);
            } else {
                float t = RLW_MU(fac000, A2(absa, 585, ind0, ig));
                t = RLW_AD(t, RLW_MU(fac100, A2(absa, 585, ind0 + 1, ig)));
                t = RLW_AD(t, RLW_MU(fac010, A2(absa, 585, ind0 + 9, ig)));
                t = RLW_AD(t, RLW_MU(fac110, A2(absa, 585, ind0 + 10, ig)));
                tau_major = RLW_MU(speccomb, t);
            }

            float tau_major1;
            if (specparm1 < 0.125f) {
                float t = RLW_MU(fac001, A2(absa, 585, ind1, ig));
                t = RLW_AD(t, RLW_MU(fac101, A2(absa, 585, ind1 + 1, ig)));
                t = RLW_AD(t, RLW_MU(fac201, A2(absa, 585, ind1 + 2, ig)));
                t = RLW_AD(t, RLW_MU(fac011, A2(absa, 585, ind1 + 9, ig)));
                t = RLW_AD(t, RLW_MU(fac111, A2(absa, 585, ind1 + 10, ig)));
                t = RLW_AD(t, RLW_MU(fac211, A2(absa, 585, ind1 + 11, ig)));
                tau_major1 = RLW_MU(speccomb1, t);
            } else if (specparm1 > 0.875f) {
                float t = RLW_MU(fac201, A2(absa, 585, ind1 - 1, ig));
                t = RLW_AD(t, RLW_MU(fac101, A2(absa, 585, ind1, ig)));
                t = RLW_AD(t, RLW_MU(fac001, A2(absa, 585, ind1 + 1, ig)));
                t = RLW_AD(t, RLW_MU(fac211, A2(absa, 585, ind1 + 8, ig)));
                t = RLW_AD(t, RLW_MU(fac111, A2(absa, 585, ind1 + 9, ig)));
                t = RLW_AD(t, RLW_MU(fac011, A2(absa, 585, ind1 + 10, ig)));
                tau_major1 = RLW_MU(speccomb1, t);
            } else {
                float t = RLW_MU(fac001, A2(absa, 585, ind1, ig));
                t = RLW_AD(t, RLW_MU(fac101, A2(absa, 585, ind1 + 1, ig)));
                t = RLW_AD(t, RLW_MU(fac011, A2(absa, 585, ind1 + 9, ig)));
                t = RLW_AD(t, RLW_MU(fac111, A2(absa, 585, ind1 + 10, ig)));
                tau_major1 = RLW_MU(speccomb1, t);
            }

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            TAUG(gs + ig) = t;
            // fracs = fracrefa(ig,jpl) + fpl*(fracrefa(ig,jpl+1)-fracrefa(ig,jpl))
            float fr0 = A2(fracrefa, 8, ig, jpl);
            float fr1 = A2(fracrefa, 8, ig, jpl + 1);
            FRACS(gs + ig) = RLW_AD(fr0, RLW_MU(fpl, RLW_SU(fr1, fr0)));
        }
    } else {
        // Upper atmosphere loop: taug = 0.0_rb ; fracs = 0.0_rb, literally.
        for (int ig = 1; ig <= ng12; ++ig) {
            TAUG(gs + ig) = 0.0f;
            FRACS(gs + ig) = 0.0f;
        }
    }
}
