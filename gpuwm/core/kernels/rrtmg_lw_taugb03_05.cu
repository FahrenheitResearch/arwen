// WRF v4.6.1 legacy RRTMG longwave -- CUDA FP32 twins of the NumPy band
// ports _taugb3/_taugb4/_taugb5 in gpuwm/core/rrtmg_lw.py (Fortran
// authority module_ra_rrtmg_lw.F lines 5241-6087), gated at max_ulp 0.
//
// Assembled into the rrtmg_lw.cu translation unit; RLW_AD/SU/MU/DV,
// rlw_pow, rlw_pow4, FS/IS/WXS/A2/TAUG/FRACS and the slot defines come
// from that file and are NOT redefined here.  Every float op is routed
// through the __f*_rn intrinsics; real-exponent ** goes through rlw_pow;
// p**4 through rlw_pow4; Fortran mod(x,1.0) is fmodf (exact).
//
// Bands 3-5 share the two-major-species (9-point spectral) lower-
// atmosphere interpolation of _spec_major and the 4-term upper-
// atmosphere absb form (ld 1175); the b35-prefixed helpers below keep
// that shared op sequence in one place.  absa leading dimension is 585
// (9,5,13,ng), absb 1175 (5,5,47,ng).

#define CHI35(s, j) chi_mls[((s) - 1) + 7 * ((j) - 1)]
// 3-D minor tables, Fortran (9,19,ng) / (5,19,ng): element (j,i,g).
#define KA9X19(tab, j, i, g) \
    tab[((j) - 1) + 9 * ((i) - 1) + 171 * ((g) - 1)]
#define KB5X19(tab, j, i, g) \
    tab[((j) - 1) + 5 * ((i) - 1) + 95 * ((g) - 1)]

// speccomb/specparm/js/fs block (Fortran e.g. lines 5297-5302).
// mult = 8. below laytrop, 4. above.
struct RlwSpecB35 {
    float comb;   // speccomb
    float parm;   // specparm (clamped at oneminus)
    int j;        // js (1-based)
    float f;      // fs
};

__device__ RlwSpecB35 rlw_b35_spec(float col1, float rat, float col2,
                                   float oneminus, float mult)
{
    RlwSpecB35 s;
    s.comb = RLW_AD(col1, RLW_MU(rat, col2));
    s.parm = RLW_DV(col1, s.comb);
    if (s.parm >= oneminus) s.parm = oneminus;
    float specmult = RLW_MU(mult, s.parm);
    s.j = 1 + (int)specmult;
    s.f = fmodf(specmult, 1.0f);
    return s;
}

// The specparm branch triplet factors (Fortran e.g. lines 5343-5372);
// facp/fact are fac00/fac10 for the ind0 half, fac01/fac11 for ind1.
// branch: 0 = specparm < 0.125, 1 = specparm > 0.875, 2 = else.
struct RlwFacB35 {
    int branch;
    float f0p, f1p, f2p, f0t, f1t, f2t;
};

__device__ RlwFacB35 rlw_b35_facs(float specparm, float sf,
                                  float facp, float fact)
{
    RlwFacB35 r;
    if (specparm < 0.125f) {
        float p = RLW_SU(sf, 1.0f);
        float p4 = rlw_pow4(p);
        float fk0 = p4;
        float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
        float fk2 = RLW_AD(p, p4);
        r.branch = 0;
        r.f0p = RLW_MU(fk0, facp);
        r.f1p = RLW_MU(fk1, facp);
        r.f2p = RLW_MU(fk2, facp);
        r.f0t = RLW_MU(fk0, fact);
        r.f1t = RLW_MU(fk1, fact);
        r.f2t = RLW_MU(fk2, fact);
    } else if (specparm > 0.875f) {
        float p = -sf;
        float p4 = rlw_pow4(p);
        float fk0 = p4;
        float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
        float fk2 = RLW_AD(p, p4);
        r.branch = 1;
        r.f0p = RLW_MU(fk0, facp);
        r.f1p = RLW_MU(fk1, facp);
        r.f2p = RLW_MU(fk2, facp);
        r.f0t = RLW_MU(fk0, fact);
        r.f1t = RLW_MU(fk1, fact);
        r.f2t = RLW_MU(fk2, fact);
    } else {
        r.branch = 2;
        r.f0p = RLW_MU(RLW_SU(1.0f, sf), facp);
        r.f0t = RLW_MU(RLW_SU(1.0f, sf), fact);
        r.f1p = RLW_MU(sf, facp);
        r.f1t = RLW_MU(sf, fact);
        r.f2p = 0.0f;
        r.f2t = 0.0f;
    }
    return r;
}

// tau_major for one g-point, absa ld 585 (Fortran e.g. lines 5415-5437).
__device__ float rlw_b35_major(const RlwFacB35 f, const float* absa,
                               int ind, int ig, float comb)
{
    float t;
    if (f.branch == 0) {
        t = RLW_MU(f.f0p, A2(absa, 585, ind, ig));
        t = RLW_AD(t, RLW_MU(f.f1p, A2(absa, 585, ind + 1, ig)));
        t = RLW_AD(t, RLW_MU(f.f2p, A2(absa, 585, ind + 2, ig)));
        t = RLW_AD(t, RLW_MU(f.f0t, A2(absa, 585, ind + 9, ig)));
        t = RLW_AD(t, RLW_MU(f.f1t, A2(absa, 585, ind + 10, ig)));
        t = RLW_AD(t, RLW_MU(f.f2t, A2(absa, 585, ind + 11, ig)));
    } else if (f.branch == 1) {
        t = RLW_MU(f.f2p, A2(absa, 585, ind - 1, ig));
        t = RLW_AD(t, RLW_MU(f.f1p, A2(absa, 585, ind, ig)));
        t = RLW_AD(t, RLW_MU(f.f0p, A2(absa, 585, ind + 1, ig)));
        t = RLW_AD(t, RLW_MU(f.f2t, A2(absa, 585, ind + 8, ig)));
        t = RLW_AD(t, RLW_MU(f.f1t, A2(absa, 585, ind + 9, ig)));
        t = RLW_AD(t, RLW_MU(f.f0t, A2(absa, 585, ind + 10, ig)));
    } else {
        t = RLW_MU(f.f0p, A2(absa, 585, ind, ig));
        t = RLW_AD(t, RLW_MU(f.f1p, A2(absa, 585, ind + 1, ig)));
        t = RLW_AD(t, RLW_MU(f.f0t, A2(absa, 585, ind + 9, ig)));
        t = RLW_AD(t, RLW_MU(f.f1t, A2(absa, 585, ind + 10, ig)));
    }
    return RLW_MU(comb, t);
}

// Upper-atmosphere 4-term absb sum, ld 1175 (Fortran e.g. 5536-5545):
// comb * (f0p*absb(ind) + f1p*absb(ind+1) + f0t*absb(ind+5)
//         + f1t*absb(ind+6)).
__device__ float rlw_b35_bmajor(float f0p, float f1p, float f0t, float f1t,
                                const float* absb, int ind, int ig,
                                float comb)
{
    float t = RLW_MU(f0p, A2(absb, 1175, ind, ig));
    t = RLW_AD(t, RLW_MU(f1p, A2(absb, 1175, ind + 1, ig)));
    t = RLW_AD(t, RLW_MU(f0t, A2(absb, 1175, ind + 5, ig)));
    t = RLW_AD(t, RLW_MU(f1t, A2(absb, 1175, ind + 6, ig)));
    return RLW_MU(comb, t);
}

// ---------------------------------------------------------------------
// Band 3: 500-630 cm-1 (low key h2o,co2 + n2o minor; high key h2o,co2
// + n2o minor).  Fortran lines 5241-5553.
// ---------------------------------------------------------------------

extern "C" __global__ void rlw_taugb3(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    // GPU_BAND_TABS[3] = absa(585,ng), absb(1175,ng), selfref(10,ng),
    //                    forref(4,ng), ka_mn2o(9,19,ng),
    //                    kb_mn2o(5,19,ng), fracrefa(ng,9), fracrefb(ng,5)
    const float* absa = tabs[0];
    const float* absb = tabs[1];
    const float* selfref = tabs[2];
    const float* forref = tabs[3];
    const float* ka_mn2o = tabs[4];
    const float* kb_mn2o = tabs[5];
    const float* fracrefa = tabs[6];
    const float* fracrefb = tabs[7];
    const int nspa3 = 9, nspb3 = 5, ng3 = 16, gs = 22;

    // P = 212.725 mb                       chi_mls(1,9)/chi_mls(2,9)
    float refrat_planck_a = RLW_DV(CHI35(1, 9), CHI35(2, 9));
    // P = 95.58 mb                         chi_mls(1,13)/chi_mls(2,13)
    float refrat_planck_b = RLW_DV(CHI35(1, 13), CHI35(2, 13));
    // P = 706.270 mb                       chi_mls(1,3)/chi_mls(2,3)
    float refrat_m_a = RLW_DV(CHI35(1, 3), CHI35(2, 3));
    // P = 95.58 mb                         chi_mls(1,13)/chi_mls(2,13)
    float refrat_m_b = RLW_DV(CHI35(1, 13), CHI35(2, 13));

    float colh2o = FS(SL_COLH2O);
    float colco2 = FS(SL_COLCO2);
    float coln2o = FS(SL_COLN2O);
    float coldry = FS(SL_COLDRY);
    int jpv = IS(SI_JP);

    if (lay <= laytrop) {
        RlwSpecB35 s0 = rlw_b35_spec(colh2o, FS(SL_RAT_H2OCO2), colco2,
                                     oneminus, 8.0f);
        RlwSpecB35 s1 = rlw_b35_spec(colh2o, FS(SL_RAT_H2OCO2_1), colco2,
                                     oneminus, 8.0f);
        RlwSpecB35 smn = rlw_b35_spec(colh2o, refrat_m_a, colco2,
                                      oneminus, 8.0f);
        int jmn2o = smn.j;
        float fmn2o = smn.f;
        // fmn2omf = minorfrac*fmn2o is a dead statement in the Fortran
        // (never read anywhere in taugb3); omitted, as in the NumPy port.

        float chi_n2o = RLW_DV(coln2o, coldry);
        float ratn2o = RLW_DV(RLW_MU(1.e20f, chi_n2o), CHI35(4, jpv + 1));
        float adjcoln2o;
        if (ratn2o > 1.5f) {
            float adjfac = RLW_AD(0.5f,
                rlw_pow(RLW_SU(ratn2o, 0.5f), 0.65f));
            adjcoln2o = RLW_MU(RLW_MU(RLW_MU(adjfac, CHI35(4, jpv + 1)),
                                      coldry), 1.e-20f);
        } else {
            adjcoln2o = coln2o;
        }

        RlwSpecB35 spl = rlw_b35_spec(colh2o, refrat_planck_a, colco2,
                                      oneminus, 8.0f);
        int jpl = spl.j;
        float fpl = spl.f;

        int ind0 = ((jpv - 1) * 5 + (IS(SI_JT) - 1)) * nspa3 + s0.j;
        int ind1 = (jpv * 5 + (IS(SI_JT1) - 1)) * nspa3 + s1.j;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        RlwFacB35 f0 = rlw_b35_facs(s0.parm, s0.f, FS(SL_FAC00),
                                    FS(SL_FAC10));
        RlwFacB35 f1 = rlw_b35_facs(s1.parm, s1.f, FS(SL_FAC01),
                                    FS(SL_FAC11));

        for (int ig = 1; ig <= ng3; ++ig) {
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
            float n2om1 = RLW_AD(KA9X19(ka_mn2o, jmn2o, indm, ig),
                RLW_MU(fmn2o,
                       RLW_SU(KA9X19(ka_mn2o, jmn2o + 1, indm, ig),
                              KA9X19(ka_mn2o, jmn2o, indm, ig))));
            float n2om2 = RLW_AD(KA9X19(ka_mn2o, jmn2o, indm + 1, ig),
                RLW_MU(fmn2o,
                       RLW_SU(KA9X19(ka_mn2o, jmn2o + 1, indm + 1, ig),
                              KA9X19(ka_mn2o, jmn2o, indm + 1, ig))));
            float absn2o = RLW_AD(n2om1, RLW_MU(FS(SL_MINORFRAC),
                                                RLW_SU(n2om2, n2om1)));

            float tau_major = rlw_b35_major(f0, absa, ind0, ig, s0.comb);
            float tau_major1 = rlw_b35_major(f1, absa, ind1, ig, s1.comb);

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, RLW_MU(adjcoln2o, absn2o));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefa, 16, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefa, 16, ig, jpl + 1),
                                   A2(fracrefa, 16, ig, jpl))));
        }
    } else {
        RlwSpecB35 s0 = rlw_b35_spec(colh2o, FS(SL_RAT_H2OCO2), colco2,
                                     oneminus, 4.0f);
        RlwSpecB35 s1 = rlw_b35_spec(colh2o, FS(SL_RAT_H2OCO2_1), colco2,
                                     oneminus, 4.0f);

        float fac000 = RLW_MU(RLW_SU(1.0f, s0.f), FS(SL_FAC00));
        float fac010 = RLW_MU(RLW_SU(1.0f, s0.f), FS(SL_FAC10));
        float fac100 = RLW_MU(s0.f, FS(SL_FAC00));
        float fac110 = RLW_MU(s0.f, FS(SL_FAC10));
        float fac001 = RLW_MU(RLW_SU(1.0f, s1.f), FS(SL_FAC01));
        float fac011 = RLW_MU(RLW_SU(1.0f, s1.f), FS(SL_FAC11));
        float fac101 = RLW_MU(s1.f, FS(SL_FAC01));
        float fac111 = RLW_MU(s1.f, FS(SL_FAC11));

        RlwSpecB35 smn = rlw_b35_spec(colh2o, refrat_m_b, colco2,
                                      oneminus, 4.0f);
        int jmn2o = smn.j;
        float fmn2o = smn.f;
        // fmn2omf: dead statement in the Fortran here too; omitted.

        float chi_n2o = RLW_DV(coln2o, coldry);
        // Fortran line 5508 writes the literal as 1.e20 (default real,
        // same f32 value as the 1.e20_rb of the lower loop).
        float ratn2o = RLW_DV(RLW_MU(1.e20f, chi_n2o), CHI35(4, jpv + 1));
        float adjcoln2o;
        if (ratn2o > 1.5f) {
            float adjfac = RLW_AD(0.5f,
                rlw_pow(RLW_SU(ratn2o, 0.5f), 0.65f));
            adjcoln2o = RLW_MU(RLW_MU(RLW_MU(adjfac, CHI35(4, jpv + 1)),
                                      coldry), 1.e-20f);
        } else {
            adjcoln2o = coln2o;
        }

        RlwSpecB35 spl = rlw_b35_spec(colh2o, refrat_planck_b, colco2,
                                      oneminus, 4.0f);
        int jpl = spl.j;
        float fpl = spl.f;

        int ind0 = ((jpv - 13) * 5 + (IS(SI_JT) - 1)) * nspb3 + s0.j;
        int ind1 = ((jpv - 12) * 5 + (IS(SI_JT1) - 1)) * nspb3 + s1.j;
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        for (int ig = 1; ig <= ng3; ++ig) {
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float n2om1 = RLW_AD(KB5X19(kb_mn2o, jmn2o, indm, ig),
                RLW_MU(fmn2o,
                       RLW_SU(KB5X19(kb_mn2o, jmn2o + 1, indm, ig),
                              KB5X19(kb_mn2o, jmn2o, indm, ig))));
            float n2om2 = RLW_AD(KB5X19(kb_mn2o, jmn2o, indm + 1, ig),
                RLW_MU(fmn2o,
                       RLW_SU(KB5X19(kb_mn2o, jmn2o + 1, indm + 1, ig),
                              KB5X19(kb_mn2o, jmn2o, indm + 1, ig))));
            float absn2o = RLW_AD(n2om1, RLW_MU(FS(SL_MINORFRAC),
                                                RLW_SU(n2om2, n2om1)));

            float t = rlw_b35_bmajor(fac000, fac100, fac010, fac110,
                                     absb, ind0, ig, s0.comb);
            t = RLW_AD(t, rlw_b35_bmajor(fac001, fac101, fac011, fac111,
                                         absb, ind1, ig, s1.comb));
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, RLW_MU(adjcoln2o, absn2o));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefb, 16, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefb, 16, ig, jpl + 1),
                                   A2(fracrefb, 16, ig, jpl))));
        }
    }
}

// ---------------------------------------------------------------------
// Band 4: 630-700 cm-1 (low key h2o,co2; high key o3,co2 + empirical
// stratospheric-cooling g-point rescale).  Fortran lines 5556-5812.
// ---------------------------------------------------------------------

extern "C" __global__ void rlw_taugb4(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    // GPU_BAND_TABS[4] = absa(585,ng), absb(1175,ng), selfref(10,ng),
    //                    forref(4,ng), fracrefa(ng,9), fracrefb(ng,5)
    const float* absa = tabs[0];
    const float* absb = tabs[1];
    const float* selfref = tabs[2];
    const float* forref = tabs[3];
    const float* fracrefa = tabs[4];
    const float* fracrefb = tabs[5];
    const int nspa4 = 9, nspb4 = 5, ng4 = 14, gs = 38;

    // P = 142.5940 mb                      chi_mls(1,11)/chi_mls(2,11)
    float refrat_planck_a = RLW_DV(CHI35(1, 11), CHI35(2, 11));
    // P = 95.58350 mb                      chi_mls(3,13)/chi_mls(2,13)
    float refrat_planck_b = RLW_DV(CHI35(3, 13), CHI35(2, 13));

    float colco2 = FS(SL_COLCO2);

    if (lay <= laytrop) {
        float colh2o = FS(SL_COLH2O);
        int jpv = IS(SI_JP);

        RlwSpecB35 s0 = rlw_b35_spec(colh2o, FS(SL_RAT_H2OCO2), colco2,
                                     oneminus, 8.0f);
        RlwSpecB35 s1 = rlw_b35_spec(colh2o, FS(SL_RAT_H2OCO2_1), colco2,
                                     oneminus, 8.0f);
        RlwSpecB35 spl = rlw_b35_spec(colh2o, refrat_planck_a, colco2,
                                      oneminus, 8.0f);
        int jpl = spl.j;
        float fpl = spl.f;

        int ind0 = ((jpv - 1) * 5 + (IS(SI_JT) - 1)) * nspa4 + s0.j;
        int ind1 = (jpv * 5 + (IS(SI_JT1) - 1)) * nspa4 + s1.j;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);

        RlwFacB35 f0 = rlw_b35_facs(s0.parm, s0.f, FS(SL_FAC00),
                                    FS(SL_FAC10));
        RlwFacB35 f1 = rlw_b35_facs(s1.parm, s1.f, FS(SL_FAC01),
                                    FS(SL_FAC11));

        for (int ig = 1; ig <= ng4; ++ig) {
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

            float tau_major = rlw_b35_major(f0, absa, ind0, ig, s0.comb);
            float tau_major1 = rlw_b35_major(f1, absa, ind1, ig, s1.comb);

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefa, 14, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefa, 14, ig, jpl + 1),
                                   A2(fracrefa, 14, ig, jpl))));
        }
    } else {
        float colo3 = FS(SL_COLO3);

        RlwSpecB35 s0 = rlw_b35_spec(colo3, FS(SL_RAT_O3CO2), colco2,
                                     oneminus, 4.0f);
        RlwSpecB35 s1 = rlw_b35_spec(colo3, FS(SL_RAT_O3CO2_1), colco2,
                                     oneminus, 4.0f);

        float fac000 = RLW_MU(RLW_SU(1.0f, s0.f), FS(SL_FAC00));
        float fac010 = RLW_MU(RLW_SU(1.0f, s0.f), FS(SL_FAC10));
        float fac100 = RLW_MU(s0.f, FS(SL_FAC00));
        float fac110 = RLW_MU(s0.f, FS(SL_FAC10));
        float fac001 = RLW_MU(RLW_SU(1.0f, s1.f), FS(SL_FAC01));
        float fac011 = RLW_MU(RLW_SU(1.0f, s1.f), FS(SL_FAC11));
        float fac101 = RLW_MU(s1.f, FS(SL_FAC01));
        float fac111 = RLW_MU(s1.f, FS(SL_FAC11));

        RlwSpecB35 spl = rlw_b35_spec(colo3, refrat_planck_b, colco2,
                                      oneminus, 4.0f);
        int jpl = spl.j;
        float fpl = spl.f;

        int ind0 = ((IS(SI_JP) - 13) * 5 + (IS(SI_JT) - 1)) * nspb4 + s0.j;
        int ind1 = ((IS(SI_JP) - 12) * 5 + (IS(SI_JT1) - 1)) * nspb4 + s1.j;

        for (int ig = 1; ig <= ng4; ++ig) {
            float t = rlw_b35_bmajor(fac000, fac100, fac010, fac110,
                                     absb, ind0, ig, s0.comb);
            t = RLW_AD(t, rlw_b35_bmajor(fac001, fac101, fac011, fac111,
                                         absb, ind1, ig, s1.comb));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefb, 14, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefb, 14, ig, jpl + 1),
                                   A2(fracrefb, 14, ig, jpl))));
        }

        // Empirical modification to improve stratospheric cooling rates
        // for co2 (Fortran lines 5802-5808; literals are default real).
        TAUG(gs + 8) = RLW_MU(TAUG(gs + 8), 0.92f);
        TAUG(gs + 9) = RLW_MU(TAUG(gs + 9), 0.88f);
        TAUG(gs + 10) = RLW_MU(TAUG(gs + 10), 1.07f);
        TAUG(gs + 11) = RLW_MU(TAUG(gs + 11), 1.1f);
        TAUG(gs + 12) = RLW_MU(TAUG(gs + 12), 0.99f);
        TAUG(gs + 13) = RLW_MU(TAUG(gs + 13), 0.88f);
        TAUG(gs + 14) = RLW_MU(TAUG(gs + 14), 0.943f);
    }
}

// ---------------------------------------------------------------------
// Band 5: 700-820 cm-1 (low key h2o,co2 + o3 minor + ccl4 xsec;
// high key o3,co2 + ccl4 xsec).  Fortran lines 5815-6087.
// ---------------------------------------------------------------------

extern "C" __global__ void rlw_taugb5(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    // GPU_BAND_TABS[5] = absa(585,ng), absb(1175,ng), selfref(10,ng),
    //                    forref(4,ng), ka_mo3(9,19,ng), ccl4(ng),
    //                    fracrefa(ng,9), fracrefb(ng,5)
    const float* absa = tabs[0];
    const float* absb = tabs[1];
    const float* selfref = tabs[2];
    const float* forref = tabs[3];
    const float* ka_mo3 = tabs[4];
    const float* ccl4 = tabs[5];
    const float* fracrefa = tabs[6];
    const float* fracrefb = tabs[7];
    const int nspa5 = 9, nspb5 = 5, ng5 = 16, gs = 52;

    // P = 473.420 mb                       chi_mls(1,5)/chi_mls(2,5)
    float refrat_planck_a = RLW_DV(CHI35(1, 5), CHI35(2, 5));
    // P = 0.2369 mb                        chi_mls(3,43)/chi_mls(2,43)
    float refrat_planck_b = RLW_DV(CHI35(3, 43), CHI35(2, 43));
    // P = 317.3480 mb                      chi_mls(1,7)/chi_mls(2,7)
    float refrat_m_a = RLW_DV(CHI35(1, 7), CHI35(2, 7));

    float colco2 = FS(SL_COLCO2);

    if (lay <= laytrop) {
        float colh2o = FS(SL_COLH2O);
        int jpv = IS(SI_JP);

        RlwSpecB35 s0 = rlw_b35_spec(colh2o, FS(SL_RAT_H2OCO2), colco2,
                                     oneminus, 8.0f);
        RlwSpecB35 s1 = rlw_b35_spec(colh2o, FS(SL_RAT_H2OCO2_1), colco2,
                                     oneminus, 8.0f);
        RlwSpecB35 smo = rlw_b35_spec(colh2o, refrat_m_a, colco2,
                                      oneminus, 8.0f);
        int jmo3 = smo.j;
        float fmo3 = smo.f;
        RlwSpecB35 spl = rlw_b35_spec(colh2o, refrat_planck_a, colco2,
                                      oneminus, 8.0f);
        int jpl = spl.j;
        float fpl = spl.f;

        int ind0 = ((jpv - 1) * 5 + (IS(SI_JT) - 1)) * nspa5 + s0.j;
        int ind1 = (jpv * 5 + (IS(SI_JT1) - 1)) * nspa5 + s1.j;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        RlwFacB35 f0 = rlw_b35_facs(s0.parm, s0.f, FS(SL_FAC00),
                                    FS(SL_FAC10));
        RlwFacB35 f1 = rlw_b35_facs(s1.parm, s1.f, FS(SL_FAC01),
                                    FS(SL_FAC11));

        for (int ig = 1; ig <= ng5; ++ig) {
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
            float o3m1 = RLW_AD(KA9X19(ka_mo3, jmo3, indm, ig),
                RLW_MU(fmo3,
                       RLW_SU(KA9X19(ka_mo3, jmo3 + 1, indm, ig),
                              KA9X19(ka_mo3, jmo3, indm, ig))));
            float o3m2 = RLW_AD(KA9X19(ka_mo3, jmo3, indm + 1, ig),
                RLW_MU(fmo3,
                       RLW_SU(KA9X19(ka_mo3, jmo3 + 1, indm + 1, ig),
                              KA9X19(ka_mo3, jmo3, indm + 1, ig))));
            float abso3 = RLW_AD(o3m1, RLW_MU(FS(SL_MINORFRAC),
                                              RLW_SU(o3m2, o3m1)));

            float tau_major = rlw_b35_major(f0, absa, ind0, ig, s0.comb);
            float tau_major1 = rlw_b35_major(f1, absa, ind1, ig, s1.comb);

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, RLW_MU(abso3, FS(SL_COLO3)));
            t = RLW_AD(t, RLW_MU(WXS(1), ccl4[ig - 1]));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefa, 16, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefa, 16, ig, jpl + 1),
                                   A2(fracrefa, 16, ig, jpl))));
        }
    } else {
        float colo3 = FS(SL_COLO3);

        RlwSpecB35 s0 = rlw_b35_spec(colo3, FS(SL_RAT_O3CO2), colco2,
                                     oneminus, 4.0f);
        RlwSpecB35 s1 = rlw_b35_spec(colo3, FS(SL_RAT_O3CO2_1), colco2,
                                     oneminus, 4.0f);

        float fac000 = RLW_MU(RLW_SU(1.0f, s0.f), FS(SL_FAC00));
        float fac010 = RLW_MU(RLW_SU(1.0f, s0.f), FS(SL_FAC10));
        float fac100 = RLW_MU(s0.f, FS(SL_FAC00));
        float fac110 = RLW_MU(s0.f, FS(SL_FAC10));
        float fac001 = RLW_MU(RLW_SU(1.0f, s1.f), FS(SL_FAC01));
        float fac011 = RLW_MU(RLW_SU(1.0f, s1.f), FS(SL_FAC11));
        float fac101 = RLW_MU(s1.f, FS(SL_FAC01));
        float fac111 = RLW_MU(s1.f, FS(SL_FAC11));

        RlwSpecB35 spl = rlw_b35_spec(colo3, refrat_planck_b, colco2,
                                      oneminus, 4.0f);
        int jpl = spl.j;
        float fpl = spl.f;

        int ind0 = ((IS(SI_JP) - 13) * 5 + (IS(SI_JT) - 1)) * nspb5 + s0.j;
        int ind1 = ((IS(SI_JP) - 12) * 5 + (IS(SI_JT1) - 1)) * nspb5 + s1.j;

        for (int ig = 1; ig <= ng5; ++ig) {
            float t = rlw_b35_bmajor(fac000, fac100, fac010, fac110,
                                     absb, ind0, ig, s0.comb);
            t = RLW_AD(t, rlw_b35_bmajor(fac001, fac101, fac011, fac111,
                                         absb, ind1, ig, s1.comb));
            t = RLW_AD(t, RLW_MU(WXS(1), ccl4[ig - 1]));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefb, 16, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefb, 16, ig, jpl + 1),
                                   A2(fracrefb, 16, ig, jpl))));
        }
    }
}

#undef CHI35
#undef KA9X19
#undef KB5X19
