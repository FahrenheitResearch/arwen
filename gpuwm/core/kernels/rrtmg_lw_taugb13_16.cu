// WRF v4.6.1 legacy RRTMG longwave, bands 13-16 -- CUDA twins of the
// gated NumPy ports _taugb13.._taugb16 in gpuwm/core/rrtmg_lw.py
// (Fortran authority module_ra_rrtmg_lw.F lines 7189-7940).
//
// Helpers (RLW_AD/SU/MU/DV, rlw_pow, rlw_pow4, FS/IS/A2/TAUG/FRACS,
// TAUGB_PROLOGUE, SL_/SI_ slot defines) come from kernels/rrtmg_lw.cu;
// the host assembles one translation unit.  Every float op is routed
// through the _rn intrinsics -- no bare arithmetic, no FMA.
//
// Band 13: 9-species lower (h2o,n2o) with co2 AND co minors; the co2
//   "too major" adjustment uses the FIXED 3.55e-4 reference (not
//   chi_mls(2,jp+1) as in bands 3/5/7/9) and a real exponent 0.68 ->
//   rlw_pow.  Upper is the kb_mo3 o3 term only.
// Band 14: 1-species (co2 low and high), no minors.
// Band 15: 9-species lower (n2o,co2) with n2 minor scaled by
//   scalen2 = colbrd*scaleminor (NOT scaleminorn2); upper writes
//   taug = fracs = 0 exactly as the Fortran does.
// Band 16: 9-species lower (h2o,ch4), no minors; upper key ch4 with
//   nspb(16) = 0 in this WRF copy, so ind0/ind1 collapse to 1 --
//   transcribed exactly, not "fixed".

// Fortran-order 3-D minor tables (9,19,ng): element (j,i,g).
// Name-spaced to this file (sibling band files build their own).
#define RLW1316_A3(tab, j, i, g) \
    tab[((j) - 1) + 9 * ((i) - 1) + 171 * ((g) - 1)]

// The specparm branch triplet + tau_major sum shared by the 9-species
// bands in this file (13, 15, 16; absa leading dimension 585 for all).
// Mirrors _nine_major in gpuwm/core/rrtmg_lw.py; facp/fact are
// fac00/fac10 (ind0 half) or fac01/fac11 (ind1 half); ind/ig 1-based.
__device__ float rlw_nine_major_b1316(float specparm, float fsv,
                                      float facp, float fact,
                                      const float* absa, int ind, int ig,
                                      float speccomb)
{
    if (specparm < 0.125f) {
        float p = RLW_SU(fsv, 1.0f);
        float p4 = rlw_pow4(p);
        float fk0 = p4;
        float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
        float fk2 = RLW_AD(p, p4);
        float fac0p = RLW_MU(fk0, facp);
        float fac1p = RLW_MU(fk1, facp);
        float fac2p = RLW_MU(fk2, facp);
        float fac0t = RLW_MU(fk0, fact);
        float fac1t = RLW_MU(fk1, fact);
        float fac2t = RLW_MU(fk2, fact);
        float s = RLW_MU(fac0p, A2(absa, 585, ind, ig));
        s = RLW_AD(s, RLW_MU(fac1p, A2(absa, 585, ind + 1, ig)));
        s = RLW_AD(s, RLW_MU(fac2p, A2(absa, 585, ind + 2, ig)));
        s = RLW_AD(s, RLW_MU(fac0t, A2(absa, 585, ind + 9, ig)));
        s = RLW_AD(s, RLW_MU(fac1t, A2(absa, 585, ind + 10, ig)));
        s = RLW_AD(s, RLW_MU(fac2t, A2(absa, 585, ind + 11, ig)));
        return RLW_MU(speccomb, s);
    } else if (specparm > 0.875f) {
        float p = -fsv;
        float p4 = rlw_pow4(p);
        float fk0 = p4;
        float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
        float fk2 = RLW_AD(p, p4);
        float fac0p = RLW_MU(fk0, facp);
        float fac1p = RLW_MU(fk1, facp);
        float fac2p = RLW_MU(fk2, facp);
        float fac0t = RLW_MU(fk0, fact);
        float fac1t = RLW_MU(fk1, fact);
        float fac2t = RLW_MU(fk2, fact);
        float s = RLW_MU(fac2p, A2(absa, 585, ind - 1, ig));
        s = RLW_AD(s, RLW_MU(fac1p, A2(absa, 585, ind, ig)));
        s = RLW_AD(s, RLW_MU(fac0p, A2(absa, 585, ind + 1, ig)));
        s = RLW_AD(s, RLW_MU(fac2t, A2(absa, 585, ind + 8, ig)));
        s = RLW_AD(s, RLW_MU(fac1t, A2(absa, 585, ind + 9, ig)));
        s = RLW_AD(s, RLW_MU(fac0t, A2(absa, 585, ind + 10, ig)));
        return RLW_MU(speccomb, s);
    } else {
        float fac0p = RLW_MU(RLW_SU(1.0f, fsv), facp);
        float fac0t = RLW_MU(RLW_SU(1.0f, fsv), fact);
        float fac1p = RLW_MU(fsv, facp);
        float fac1t = RLW_MU(fsv, fact);
        float s = RLW_MU(fac0p, A2(absa, 585, ind, ig));
        s = RLW_AD(s, RLW_MU(fac1p, A2(absa, 585, ind + 1, ig)));
        s = RLW_AD(s, RLW_MU(fac0t, A2(absa, 585, ind + 9, ig)));
        s = RLW_AD(s, RLW_MU(fac1t, A2(absa, 585, ind + 10, ig)));
        return RLW_MU(speccomb, s);
    }
}

// ---------------------------------------------------------------------
// Band 13: 2080-2250 cm-1 (low key h2o,n2o; low minors co2, co; high
// minor o3 only).  Fortran lines 7189-7445.
// ---------------------------------------------------------------------

extern "C" __global__ void rlw_taugb13(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    // GPU_BAND_TABS[13] = absa(585,ng), selfref(10,ng), forref(4,ng),
    //                     ka_mco2(9,19,ng), ka_mco(9,19,ng),
    //                     kb_mo3(19,ng), fracrefa(ng,9), fracrefb(ng)
    const float* absa = tabs[0];
    const float* selfref = tabs[1];
    const float* forref = tabs[2];
    const float* ka_mco2 = tabs[3];
    const float* ka_mco = tabs[4];
    const float* kb_mo3 = tabs[5];
    const float* fracrefa = tabs[6];
    const float* fracrefb = tabs[7];
    const int nspa13 = 9, ng13 = 4, gs = 130;

    if (lay <= laytrop) {
        // P = 473.420 mb (Level 5): chi_mls(1,5)/chi_mls(4,5)
        float refrat_planck_a = RLW_DV(chi_mls[0 + 7 * 4],
                                       chi_mls[3 + 7 * 4]);
        // P = 1053. (Level 1): chi_mls(1,1)/chi_mls(4,1)
        float refrat_m_a = RLW_DV(chi_mls[0 + 7 * 0], chi_mls[3 + 7 * 0]);
        // P = 706. (Level 3): chi_mls(1,3)/chi_mls(4,3)
        float refrat_m_a3 = RLW_DV(chi_mls[0 + 7 * 2], chi_mls[3 + 7 * 2]);

        float colh2o = FS(SL_COLH2O);
        float coln2o = FS(SL_COLN2O);

        float speccomb = RLW_AD(colh2o, RLW_MU(FS(SL_RAT_H2ON2O), coln2o));
        float specparm = RLW_DV(colh2o, speccomb);
        if (specparm >= oneminus) specparm = oneminus;
        float specmult = RLW_MU(8.0f, specparm);
        int js = 1 + (int)specmult;
        float fsv = fmodf(specmult, 1.0f);

        float speccomb1 = RLW_AD(colh2o,
                                 RLW_MU(FS(SL_RAT_H2ON2O_1), coln2o));
        float specparm1 = RLW_DV(colh2o, speccomb1);
        if (specparm1 >= oneminus) specparm1 = oneminus;
        float specmult1 = RLW_MU(8.0f, specparm1);
        int js1 = 1 + (int)specmult1;
        float fs1 = fmodf(specmult1, 1.0f);

        float speccomb_mco2 = RLW_AD(colh2o, RLW_MU(refrat_m_a, coln2o));
        float specparm_mco2 = RLW_DV(colh2o, speccomb_mco2);
        if (specparm_mco2 >= oneminus) specparm_mco2 = oneminus;
        float specmult_mco2 = RLW_MU(8.0f, specparm_mco2);
        int jmco2 = 1 + (int)specmult_mco2;
        float fmco2 = fmodf(specmult_mco2, 1.0f);

        // CO2-too-major empirical adjustment (FIXED 3.55e-4 reference;
        // real exponent 0.68 -> glibc powf transcription).
        float chi_co2 = RLW_DV(FS(SL_COLCO2), FS(SL_COLDRY));
        float ratco2 = RLW_DV(RLW_MU(1.e20f, chi_co2), 3.55e-4f);
        float adjcolco2;
        if (ratco2 > 3.0f) {
            float adjfac = RLW_AD(2.0f,
                                  rlw_pow(RLW_SU(ratco2, 2.0f), 0.68f));
            adjcolco2 = RLW_MU(RLW_MU(RLW_MU(adjfac, 3.55e-4f),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcolco2 = FS(SL_COLCO2);
        }

        float speccomb_mco = RLW_AD(colh2o, RLW_MU(refrat_m_a3, coln2o));
        float specparm_mco = RLW_DV(colh2o, speccomb_mco);
        if (specparm_mco >= oneminus) specparm_mco = oneminus;
        float specmult_mco = RLW_MU(8.0f, specparm_mco);
        int jmco = 1 + (int)specmult_mco;
        float fmco = fmodf(specmult_mco, 1.0f);

        float speccomb_planck = RLW_AD(colh2o,
                                       RLW_MU(refrat_planck_a, coln2o));
        float specparm_planck = RLW_DV(colh2o, speccomb_planck);
        if (specparm_planck >= oneminus) specparm_planck = oneminus;
        float specmult_planck = RLW_MU(8.0f, specparm_planck);
        int jpl = 1 + (int)specmult_planck;
        float fpl = fmodf(specmult_planck, 1.0f);

        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa13 + js;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa13 + js1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        for (int ig = 1; ig <= ng13; ++ig) {
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
            float co2m1 = RLW_AD(RLW1316_A3(ka_mco2, jmco2, indm, ig),
                RLW_MU(fmco2,
                       RLW_SU(RLW1316_A3(ka_mco2, jmco2 + 1, indm, ig),
                              RLW1316_A3(ka_mco2, jmco2, indm, ig))));
            float co2m2 = RLW_AD(RLW1316_A3(ka_mco2, jmco2, indm + 1, ig),
                RLW_MU(fmco2,
                       RLW_SU(RLW1316_A3(ka_mco2, jmco2 + 1, indm + 1, ig),
                              RLW1316_A3(ka_mco2, jmco2, indm + 1, ig))));
            float absco2 = RLW_AD(co2m1, RLW_MU(FS(SL_MINORFRAC),
                                                RLW_SU(co2m2, co2m1)));
            float com1 = RLW_AD(RLW1316_A3(ka_mco, jmco, indm, ig),
                RLW_MU(fmco,
                       RLW_SU(RLW1316_A3(ka_mco, jmco + 1, indm, ig),
                              RLW1316_A3(ka_mco, jmco, indm, ig))));
            float com2 = RLW_AD(RLW1316_A3(ka_mco, jmco, indm + 1, ig),
                RLW_MU(fmco,
                       RLW_SU(RLW1316_A3(ka_mco, jmco + 1, indm + 1, ig),
                              RLW1316_A3(ka_mco, jmco, indm + 1, ig))));
            float absco = RLW_AD(com1, RLW_MU(FS(SL_MINORFRAC),
                                              RLW_SU(com2, com1)));

            float tau_major = rlw_nine_major_b1316(
                specparm, fsv, FS(SL_FAC00), FS(SL_FAC10),
                absa, ind0, ig, speccomb);
            float tau_major1 = rlw_nine_major_b1316(
                specparm1, fs1, FS(SL_FAC01), FS(SL_FAC11),
                absa, ind1, ig, speccomb1);

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, RLW_MU(adjcolco2, absco2));
            t = RLW_AD(t, RLW_MU(FS(SL_COLCO), absco));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefa, 4, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefa, 4, ig, jpl + 1),
                                   A2(fracrefa, 4, ig, jpl))));
        }
    } else {
        // Upper atmosphere: o3 minor ONLY (no foreign continuum in
        // this WRF copy).
        int indm = IS(SI_INDMINOR);
        for (int ig = 1; ig <= ng13; ++ig) {
            float abso3 = RLW_AD(A2(kb_mo3, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(kb_mo3, 19, indm + 1, ig),
                              A2(kb_mo3, 19, indm, ig))));
            TAUG(gs + ig) = RLW_MU(FS(SL_COLO3), abso3);
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }
}

// ---------------------------------------------------------------------
// Band 14: 2250-2380 cm-1 (low - co2; high - co2).  Fortran lines
// 7448-7506.
// ---------------------------------------------------------------------

extern "C" __global__ void rlw_taugb14(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    // GPU_BAND_TABS[14] = absa(65,ng), absb(235,ng), selfref(10,ng),
    //                     forref(4,ng), fracrefa(ng), fracrefb(ng)
    const float* absa = tabs[0];
    const float* absb = tabs[1];
    const float* selfref = tabs[2];
    const float* forref = tabs[3];
    const float* fracrefa = tabs[4];
    const float* fracrefb = tabs[5];
    const int nspa14 = 1, nspb14 = 1, ng14 = 2, gs = 134;

    if (lay <= laytrop) {
        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa14 + 1;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa14 + 1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        for (int ig = 1; ig <= ng14; ++ig) {
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
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC10), A2(absa, 65, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC01), A2(absa, 65, ind1, ig)));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC11), A2(absa, 65, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLCO2), tmaj);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefa[ig - 1];
        }
    } else {
        int ind0 = ((IS(SI_JP) - 13) * 5 + (IS(SI_JT) - 1)) * nspb14 + 1;
        int ind1 = ((IS(SI_JP) - 12) * 5 + (IS(SI_JT1) - 1)) * nspb14 + 1;
        for (int ig = 1; ig <= ng14; ++ig) {
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC10), A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC01), A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC11), A2(absb, 235, ind1 + 1, ig)));
            TAUG(gs + ig) = RLW_MU(FS(SL_COLCO2), tmaj);
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }
}

// ---------------------------------------------------------------------
// Band 15: 2380-2600 cm-1 (low key n2o,co2; low minor n2 via
// colbrd*scaleminor; high - NOTHING: taug = fracs = 0).  Fortran lines
// 7509-7731.
// ---------------------------------------------------------------------

extern "C" __global__ void rlw_taugb15(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    // GPU_BAND_TABS[15] = absa(585,ng), selfref(10,ng), forref(4,ng),
    //                     ka_mn2(9,19,ng), fracrefa(ng,9)
    const float* absa = tabs[0];
    const float* selfref = tabs[1];
    const float* forref = tabs[2];
    const float* ka_mn2 = tabs[3];
    const float* fracrefa = tabs[4];
    const int nspa15 = 9, ng15 = 2, gs = 136;

    if (lay <= laytrop) {
        // P = 1053. mb (Level 1): chi_mls(4,1)/chi_mls(2,1) -- Planck
        // and minor use the same ratio.
        float refrat_planck_a = RLW_DV(chi_mls[3 + 7 * 0],
                                       chi_mls[1 + 7 * 0]);
        float refrat_m_a = RLW_DV(chi_mls[3 + 7 * 0], chi_mls[1 + 7 * 0]);

        float coln2o = FS(SL_COLN2O);
        float colco2 = FS(SL_COLCO2);

        float speccomb = RLW_AD(coln2o, RLW_MU(FS(SL_RAT_N2OCO2), colco2));
        float specparm = RLW_DV(coln2o, speccomb);
        if (specparm >= oneminus) specparm = oneminus;
        float specmult = RLW_MU(8.0f, specparm);
        int js = 1 + (int)specmult;
        float fsv = fmodf(specmult, 1.0f);

        float speccomb1 = RLW_AD(coln2o,
                                 RLW_MU(FS(SL_RAT_N2OCO2_1), colco2));
        float specparm1 = RLW_DV(coln2o, speccomb1);
        if (specparm1 >= oneminus) specparm1 = oneminus;
        float specmult1 = RLW_MU(8.0f, specparm1);
        int js1 = 1 + (int)specmult1;
        float fs1 = fmodf(specmult1, 1.0f);

        float speccomb_mn2 = RLW_AD(coln2o, RLW_MU(refrat_m_a, colco2));
        float specparm_mn2 = RLW_DV(coln2o, speccomb_mn2);
        if (specparm_mn2 >= oneminus) specparm_mn2 = oneminus;
        float specmult_mn2 = RLW_MU(8.0f, specparm_mn2);
        int jmn2 = 1 + (int)specmult_mn2;
        float fmn2 = fmodf(specmult_mn2, 1.0f);

        float speccomb_planck = RLW_AD(coln2o,
                                       RLW_MU(refrat_planck_a, colco2));
        float specparm_planck = RLW_DV(coln2o, speccomb_planck);
        if (specparm_planck >= oneminus) specparm_planck = oneminus;
        float specmult_planck = RLW_MU(8.0f, specparm_planck);
        int jpl = 1 + (int)specmult_planck;
        float fpl = fmodf(specmult_planck, 1.0f);

        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa15 + js;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa15 + js1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        // scalen2 = colbrd*scaleminor -- NOT scaleminorn2 (band 15 is
        // the one lower-minor band that scales by plain scaleminor).
        float scalen2 = RLW_MU(FS(SL_COLBRD), FS(SL_SCALEMINOR));

        for (int ig = 1; ig <= ng15; ++ig) {
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
            float n2m1 = RLW_AD(RLW1316_A3(ka_mn2, jmn2, indm, ig),
                RLW_MU(fmn2,
                       RLW_SU(RLW1316_A3(ka_mn2, jmn2 + 1, indm, ig),
                              RLW1316_A3(ka_mn2, jmn2, indm, ig))));
            float n2m2 = RLW_AD(RLW1316_A3(ka_mn2, jmn2, indm + 1, ig),
                RLW_MU(fmn2,
                       RLW_SU(RLW1316_A3(ka_mn2, jmn2 + 1, indm + 1, ig),
                              RLW1316_A3(ka_mn2, jmn2, indm + 1, ig))));
            float taun2 = RLW_MU(scalen2,
                RLW_AD(n2m1, RLW_MU(FS(SL_MINORFRAC),
                                    RLW_SU(n2m2, n2m1))));

            float tau_major = rlw_nine_major_b1316(
                specparm, fsv, FS(SL_FAC00), FS(SL_FAC10),
                absa, ind0, ig, speccomb);
            float tau_major1 = rlw_nine_major_b1316(
                specparm1, fs1, FS(SL_FAC01), FS(SL_FAC11),
                absa, ind1, ig, speccomb1);

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, taun2);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefa, 2, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefa, 2, ig, jpl + 1),
                                   A2(fracrefa, 2, ig, jpl))));
        }
    } else {
        // Upper atmosphere: nothing -- taug and fracs BOTH zeroed.
        for (int ig = 1; ig <= ng15; ++ig) {
            TAUG(gs + ig) = 0.0f;
            FRACS(gs + ig) = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------
// Band 16: 2600-3250 cm-1 (low key h2o,ch4; high key ch4).  Fortran
// lines 7734-7940.  nspb(16) = 0 in this WRF copy, so the upper-loop
// ind0/ind1 are always 1 (transcribed as-is, matching the Fortran and
// the gated NumPy port -- do NOT "fix").
// ---------------------------------------------------------------------

extern "C" __global__ void rlw_taugb16(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    // GPU_BAND_TABS[16] = absa(585,ng), absb(235,ng), selfref(10,ng),
    //                     forref(4,ng), fracrefa(ng,9), fracrefb(ng)
    const float* absa = tabs[0];
    const float* absb = tabs[1];
    const float* selfref = tabs[2];
    const float* forref = tabs[3];
    const float* fracrefa = tabs[4];
    const float* fracrefb = tabs[5];
    const int nspa16 = 9, nspb16 = 0, ng16 = 2, gs = 138;

    if (lay <= laytrop) {
        // P = 387. mb (Level 6): chi_mls(1,6)/chi_mls(6,6)
        float refrat_planck_a = RLW_DV(chi_mls[0 + 7 * 5],
                                       chi_mls[5 + 7 * 5]);

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
        float fs1 = fmodf(specmult1, 1.0f);

        float speccomb_planck = RLW_AD(colh2o,
                                       RLW_MU(refrat_planck_a, colch4));
        float specparm_planck = RLW_DV(colh2o, speccomb_planck);
        if (specparm_planck >= oneminus) specparm_planck = oneminus;
        float specmult_planck = RLW_MU(8.0f, specparm_planck);
        int jpl = 1 + (int)specmult_planck;
        float fpl = fmodf(specmult_planck, 1.0f);

        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa16 + js;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa16 + js1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);

        for (int ig = 1; ig <= ng16; ++ig) {
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

            float tau_major = rlw_nine_major_b1316(
                specparm, fsv, FS(SL_FAC00), FS(SL_FAC10),
                absa, ind0, ig, speccomb);
            float tau_major1 = rlw_nine_major_b1316(
                specparm1, fs1, FS(SL_FAC01), FS(SL_FAC11),
                absa, ind1, ig, speccomb1);

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefa, 2, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefa, 2, ig, jpl + 1),
                                   A2(fracrefa, 2, ig, jpl))));
        }
    } else {
        // nspb(16) = 0: both indices collapse to 1 -> absb rows 1 and 2.
        int ind0 = ((IS(SI_JP) - 13) * 5 + (IS(SI_JT) - 1)) * nspb16 + 1;
        int ind1 = ((IS(SI_JP) - 12) * 5 + (IS(SI_JT1) - 1)) * nspb16 + 1;
        for (int ig = 1; ig <= ng16; ++ig) {
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC10), A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC01), A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC11), A2(absb, 235, ind1 + 1, ig)));
            TAUG(gs + ig) = RLW_MU(FS(SL_COLCH4), tmaj);
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }
}

#undef RLW1316_A3
