// RRTMG LW chain kernels beyond taumol: inatm, cldprmc, and the rtrnmc
// pipeline.  Compiled after kernels/rrtmg_lw.cu in the same translation
// unit (helpers RLW_*, rlw_exp, ... visible).  Bitwise discipline as in
// that file's header; NumPy reference: gpuwm/core/rrtmg_lw.py.

// rtrnmc module data rec_6 (0.166667_rb).
#define RLW_REC6 0.166667f
// rlw_rtrn_march keeps per-layer profiles in thread-local arrays.
#define RLW_MAXLAY 128

// ---------------------------------------------------------------------
// inatm -- one thread per column, serial layers (sequential amttl/wvttl
// accumulations).  Gas/mcica transfers are plain copies done host-side;
// this kernel computes the derived quantities: coldry, wbrodl, wkl
// scaling, wx scaling, pwvcm.  Inputs wkl_vmr holds h2o..o2 vmr already
// installed in Fortran slot order (ncol, MXMOL, nl); wx_vmr likewise.
// ---------------------------------------------------------------------

extern "C" __global__ void rlw_inatm(
    int ncol, int nl,
    const float* __restrict__ plev,     // (ncol, nl+1)
    float grav, float avogad,
    float* __restrict__ wkl,            // (ncol, MXMOL, nl) in: vmr, out: molec/cm2
    float* __restrict__ wx,             // (ncol, MAXXSEC, nl) in: vmr, out: scaled
    float* __restrict__ coldry,         // (ncol, nl)
    float* __restrict__ wbrodl,         // (ncol, nl)
    float* __restrict__ pwvcm_v)        // (ncol)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const float amd = 28.9660f;
    const float amw = 18.0160f;
#define WKLC(m, l) wkl[((long long)col * MXMOL + ((m) - 1)) * nl + ((l) - 1)]
#define WXC(m, l) wx[((long long)col * MAXXSEC + ((m) - 1)) * nl + ((l) - 1)]
    const float* pz = plev + (long long)col * (nl + 1);

    float amttl = 0.0f;
    float wvttl = 0.0f;
    for (int l = 1; l <= nl; ++l) {
        float w1 = WKLC(1, l);
        float amm = RLW_AD(RLW_MU(RLW_SU(1.0f, w1), amd), RLW_MU(w1, amw));
        float num = RLW_MU(RLW_MU(RLW_SU(pz[l - 1], pz[l]), 1.e3f), avogad);
        float den = RLW_MU(RLW_MU(RLW_MU(1.e2f, grav), amm),
                           RLW_AD(1.0f, w1));
        coldry[(long long)col * nl + (l - 1)] = RLW_DV(num, den);
    }
    for (int l = 1; l <= nl; ++l) {
        float cd = coldry[(long long)col * nl + (l - 1)];
        float summol = 0.0f;
        for (int imol = 2; imol <= 7; ++imol)
            summol = RLW_AD(summol, WKLC(imol, l));
        wbrodl[(long long)col * nl + (l - 1)] =
            RLW_MU(cd, RLW_SU(1.0f, summol));
        for (int imol = 1; imol <= 7; ++imol)
            WKLC(imol, l) = RLW_MU(cd, WKLC(imol, l));
        amttl = RLW_AD(RLW_AD(amttl, cd), WKLC(1, l));
        wvttl = RLW_AD(wvttl, WKLC(1, l));
        for (int ix = 1; ix <= MAXXSEC; ++ix) {
            // ixindx = identity (1,2,3,4) in lwdatinit
            WXC(ix, l) = RLW_MU(RLW_MU(cd, WXC(ix, l)), 1.e-20f);
        }
    }
    float wvsh = RLW_DV(RLW_MU(amw, wvttl), RLW_MU(amd, amttl));
    pwvcm_v[col] = RLW_DV(RLW_MU(wvsh, RLW_MU(1.e3f, pz[0])),
                          RLW_MU(1.e2f, grav));
#undef WKLC
#undef WXC
}

// ---------------------------------------------------------------------
// cldprmc -- one thread per (col, ig); serial layers.  Error conditions
// set err_flag (host raises); ncb_flag records that an ncbands-writing
// branch fired (host maps to 16/5/1 exactly as the sequential Fortran
// scalar would end up, since all writes within one iceflag are equal).
// ---------------------------------------------------------------------

extern "C" __global__ void rlw_cldprmc(
    int ncol, int nl, int inflag, int iceflag, int liqflag,
    const float* __restrict__ cldfmc,   // (ncol, NGPTLW, nl)
    const float* __restrict__ ciwpmc,
    const float* __restrict__ clwpmc,
    const float* __restrict__ cswpmc,
    const float* __restrict__ reicmc,   // (ncol, nl)
    const float* __restrict__ relqmc,
    const float* __restrict__ resnmc,
    const float* __restrict__ absice1,  // (2,5) F-order
    const float* __restrict__ absice2,  // (43,16)
    const float* __restrict__ absice3,  // (46,16)
    const float* __restrict__ absice0,  // (2)
    const float* __restrict__ absliq1,  // (58,16)
    float absliq0,
    const int* __restrict__ ngb,        // (NGPTLW) 1-based band ids
    float* __restrict__ taucmc,         // (ncol, NGPTLW, nl) in/out
    int* __restrict__ ncb_flag,         // (ncol)
    int* __restrict__ err_flag)         // (1)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)ncol * NGPTLW) return;
    int col = (int)(tid / NGPTLW);
    int ig = (int)(tid % NGPTLW) + 1;
    const float cldmin = 1.e-20f;
    const int icb[16] = {1,2,3,3,3,4,4,4,5,5,5,5,5,5,5,5};
#define MC(a, l) a[((long long)col * NGPTLW + (ig - 1)) * nl + ((l) - 1)]
#define AB2(tab, ld, i, b) tab[((i) - 1) + ((b) - 1) * (ld)]

    for (int lay = 1; lay <= nl; ++lay) {
        float cwp = RLW_AD(RLW_AD(MC(ciwpmc, lay), MC(clwpmc, lay)),
                           MC(cswpmc, lay));
        if (!(MC(cldfmc, lay) >= cldmin
              && (cwp >= cldmin || MC(taucmc, lay) >= cldmin)))
            continue;
        if (inflag == 0) return;             // taucmc already set
        if (inflag == 1) { atomicExch(err_flag, 1); return; }

        float radice = reicmc[(long long)col * nl + (lay - 1)];
        float abscoice, abscosno;
        float ice_plus_snow = RLW_AD(MC(ciwpmc, lay), MC(cswpmc, lay));
        if (ice_plus_snow == 0.0f) {
            abscoice = 0.0f;
            abscosno = 0.0f;
        } else if (iceflag == 0) {
            if (radice < 10.0f) { atomicExch(err_flag, 2); return; }
            abscoice = RLW_AD(absice0[0], RLW_DV(absice0[1], radice));
            abscosno = 0.0f;
        } else if (iceflag == 1) {
            if (radice < 13.0f || radice > 130.0f) {
                atomicExch(err_flag, 3); return;
            }
            atomicExch(&ncb_flag[col], 1);
            int ib = icb[ngb[ig - 1] - 1];
            abscoice = RLW_AD(AB2(absice1, 2, 1, ib),
                              RLW_DV(AB2(absice1, 2, 2, ib), radice));
            abscosno = 0.0f;
        } else if (iceflag == 2) {
            if (radice < 5.0f || radice > 131.0f) {
                atomicExch(err_flag, 4); return;
            }
            atomicExch(&ncb_flag[col], 1);
            float factor = RLW_DV(RLW_SU(radice, 2.0f), 3.0f);
            int index = (int)factor;
            if (index == 43) index = 42;
            float fint = RLW_SU(factor, (float)index);
            int ib = ngb[ig - 1];
            abscoice = RLW_AD(AB2(absice2, 43, index, ib),
                RLW_MU(fint, RLW_SU(AB2(absice2, 43, index + 1, ib),
                                    AB2(absice2, 43, index, ib))));
            abscosno = 0.0f;
        } else {                              // iceflag >= 3
            if (radice < 5.0f || radice > 140.0f) {
                atomicExch(err_flag, 5); return;
            }
            atomicExch(&ncb_flag[col], 1);
            float factor = RLW_DV(RLW_SU(radice, 2.0f), 3.0f);
            int index = (int)factor;
            if (index == 46) index = 45;
            float fint = RLW_SU(factor, (float)index);
            int ib = ngb[ig - 1];
            abscoice = RLW_AD(AB2(absice3, 46, index, ib),
                RLW_MU(fint, RLW_SU(AB2(absice3, 46, index + 1, ib),
                                    AB2(absice3, 46, index, ib))));
            abscosno = 0.0f;
        }

        if (MC(cswpmc, lay) > 0.0f && iceflag == 5) {
            float radsno = resnmc[(long long)col * nl + (lay - 1)];
            if (radsno < 5.0f || radsno > 140.0f) {
                atomicExch(err_flag, 6); return;
            }
            atomicExch(&ncb_flag[col], 1);
            float factor = RLW_DV(RLW_SU(radsno, 2.0f), 3.0f);
            int index = (int)factor;
            if (index == 46) index = 45;
            float fint = RLW_SU(factor, (float)index);
            int ib = ngb[ig - 1];
            abscosno = RLW_AD(AB2(absice3, 46, index, ib),
                RLW_MU(fint, RLW_SU(AB2(absice3, 46, index + 1, ib),
                                    AB2(absice3, 46, index, ib))));
        }

        float abscoliq;
        if (MC(clwpmc, lay) == 0.0f) {
            abscoliq = 0.0f;
        } else if (liqflag == 0) {
            abscoliq = absliq0;
        } else {                              // liqflag == 1
            float radliq = relqmc[(long long)col * nl + (lay - 1)];
            if (radliq < 2.5f || radliq > 60.0f) {
                atomicExch(err_flag, 7); return;
            }
            int index = (int)RLW_SU(radliq, 1.5f);
            if (index == 0) index = 1;
            if (index == 58) index = 57;
            float fint = RLW_SU(RLW_SU(radliq, 1.5f), (float)index);
            int ib = ngb[ig - 1];
            abscoliq = RLW_AD(AB2(absliq1, 58, index, ib),
                RLW_MU(fint, RLW_SU(AB2(absliq1, 58, index + 1, ib),
                                    AB2(absliq1, 58, index, ib))));
        }

        MC(taucmc, lay) = RLW_AD(
            RLW_AD(RLW_MU(MC(ciwpmc, lay), abscoice),
                   RLW_MU(MC(clwpmc, lay), abscoliq)),
            RLW_MU(MC(cswpmc, lay), abscosno));
    }
#undef MC
#undef AB2
}

// ---------------------------------------------------------------------
// rtrnmc pipeline.
//   k1 rlw_rtrn_secdiff : per col, 16 bands (rlw_exp)
//   k2 rlw_rtrn_prol    : per (col, lay), cloud optics + icldlyr
//   k3 rlw_rtrn_march   : per (col, igc), full down+up march, profiles out
//   k4 rlw_rtrn_accum   : per (col, lev), ordered lane/band accumulation
//   k5 rlw_rtrn_final   : per col, fluxfac/fnet/htr
// Layouts: profiles (ncol, NGPTLW, nl) C-order; lev vectors (ncol, nl+1).
// ---------------------------------------------------------------------

// taut = taug + taua(band): trivial, but compiled HERE so it inherits the
// no-ftz translation unit (CuPy's own ufunc kernels compile with
// -ftz=true and are off-limits for chain arithmetic).
extern "C" __global__ void rlw_taut(
    int ncol, int nl,
    const float* __restrict__ taug,   // (ncol, nl, NGPTLW)
    const float* __restrict__ taua,   // (ncol, nl, 16)
    const int* __restrict__ ngb,
    float* __restrict__ taut)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)ncol * nl * NGPTLW) return;
    int ig = (int)(tid % NGPTLW);
    long long collay = tid / NGPTLW;
    taut[tid] = RLW_AD(taug[tid],
                       taua[collay * NBNDLW + (ngb[ig] - 1)]);
}

extern "C" __global__ void rlw_rtrn_secdiff(
    int ncol, const float* __restrict__ pwvcm_v,
    const float* __restrict__ a0, const float* __restrict__ a1,
    const float* __restrict__ a2,
    float* __restrict__ secdiff)        // (ncol, 16)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    float pwvcm = pwvcm_v[col];
    for (int ibnd = 1; ibnd <= NBNDLW; ++ibnd) {
        float sd;
        if (ibnd == 1 || ibnd == 4 || ibnd >= 10) {
            sd = 1.66f;
        } else {
            sd = RLW_AD(a0[ibnd - 1],
                        RLW_MU(a1[ibnd - 1],
                               rlw_exp(RLW_MU(a2[ibnd - 1], pwvcm))));
            if (sd > 1.80f) sd = 1.80f;
            if (sd < 1.50f) sd = 1.50f;
        }
        secdiff[(long long)col * NBNDLW + (ibnd - 1)] = sd;
    }
}

extern "C" __global__ void rlw_rtrn_prol(
    int ncol, int nl,
    const float* __restrict__ cldfmc,   // (ncol, NGPTLW, nl)
    const float* __restrict__ taucmc,   // (ncol, NGPTLW, nl)
    const float* __restrict__ secdiff,  // (ncol, 16)
    const int* __restrict__ ngb,
    float* __restrict__ odcld,          // (ncol, NGPTLW, nl)
    float* __restrict__ efclfrac,       // (ncol, NGPTLW, nl)
    int* __restrict__ icldlyr)          // (ncol, nl)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)ncol * nl) return;
    int col = (int)(tid / nl);
    int lay = (int)(tid % nl) + 1;
#define MC2(a) a[((long long)col * NGPTLW + (ig - 1)) * nl + (lay - 1)]
    int flag = 0;
    for (int ig = 1; ig <= NGPTLW; ++ig) {
        if (MC2(cldfmc) == 1.0f) {
            int ib = ngb[ig - 1];
            float od = RLW_MU(secdiff[(long long)col * NBNDLW + (ib - 1)],
                              MC2(taucmc));
            MC2(odcld) = od;
            float transcld = rlw_exp(-od);
            float abscld = RLW_SU(1.0f, transcld);
            MC2(efclfrac) = RLW_MU(abscld, MC2(cldfmc));
            flag = 1;
        } else {
            MC2(odcld) = 0.0f;
            MC2(efclfrac) = 0.0f;
        }
    }
    icldlyr[(long long)col * nl + (lay - 1)] = flag;
#undef MC2
}

extern "C" __global__ void rlw_rtrn_march(
    int ncol, int nl,
    const float* __restrict__ cldfmc,   // (ncol, NGPTLW, nl)
    const float* __restrict__ odcld,
    const float* __restrict__ efclfrac,
    const int* __restrict__ icldlyr,    // (ncol, nl)
    const float* __restrict__ secdiff,  // (ncol, 16)
    const float* __restrict__ semiss,   // (ncol, 16)
    const float* __restrict__ planklay, // (ncol, nl, 16)
    const float* __restrict__ planklev, // (ncol, nl+1, 16)
    const float* __restrict__ plankbnd, // (ncol, 16)
    const float* __restrict__ fracs,    // (ncol, nl, NGPTLW)
    const float* __restrict__ taut,     // (ncol, nl, NGPTLW)
    const float* __restrict__ tau_tbl,  // (10001)
    const float* __restrict__ exp_tbl,
    const float* __restrict__ tfn_tbl,
    float bpade,
    const int* __restrict__ ngb,
    float* __restrict__ radld_p,        // (ncol, NGPTLW, nl)
    float* __restrict__ radclrd_p,
    float* __restrict__ radlu_p,
    float* __restrict__ radclru_p,
    unsigned char* __restrict__ iclddn_p,  // (ncol, NGPTLW, nl)
    float* __restrict__ radlu_sfc,      // (ncol, NGPTLW)
    float* __restrict__ radclru_sfc)    // (ncol, NGPTLW)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)ncol * NGPTLW) return;
    if (nl > RLW_MAXLAY) return;   // host asserts nl <= RLW_MAXLAY
    int col = (int)(tid / NGPTLW);
    int igc = (int)(tid % NGPTLW) + 1;
    int iband = ngb[igc - 1];
    float sd = secdiff[(long long)col * NBNDLW + (iband - 1)];

    float atrans[RLW_MAXLAY], atot[RLW_MAXLAY], bbugas[RLW_MAXLAY],
          bbutot[RLW_MAXLAY];

#define PRO(a, l) a[((long long)col * NGPTLW + (igc - 1)) * nl + ((l) - 1)]
#define PLLAY2(l) planklay[((long long)col * nl + ((l) - 1)) * NBNDLW + (iband - 1)]
#define PLLEV2(l) planklev[((long long)col * (nl + 1) + (l)) * NBNDLW + (iband - 1)]
#define FR(l) fracs[((long long)col * nl + ((l) - 1)) * NGPTLW + (igc - 1)]
#define TT(l) taut[((long long)col * nl + ((l) - 1)) * NGPTLW + (igc - 1)]

    float radld = 0.0f;
    float radclrd = 0.0f;
    int iclddn = 0;

    for (int lev = nl; lev >= 1; --lev) {
        float plfrac = FR(lev);
        float blay = PLLAY2(lev);
        float dplankup = RLW_SU(PLLEV2(lev), blay);
        float dplankdn = RLW_SU(PLLEV2(lev - 1), blay);
        float odepth = RLW_MU(sd, TT(lev));
        if (odepth < 0.0f) odepth = 0.0f;
        float bbd;
        if (icldlyr[(long long)col * nl + (lev - 1)] == 1) {
            iclddn = 1;
            float oc = PRO(odcld, lev);
            float odtot = RLW_AD(odepth, oc);
            float gassrc, bbdtot, at, ao, bug, but;
            if (odtot < 0.06f) {
                at = RLW_SU(odepth, RLW_MU(0.5f, RLW_MU(odepth, odepth)));
                float od_rec = RLW_MU(RLW_REC6, odepth);
                gassrc = RLW_MU(RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(dplankdn, od_rec))), at);
                ao = RLW_SU(odtot, RLW_MU(0.5f, RLW_MU(odtot, odtot)));
                float odtot_rec = RLW_MU(RLW_REC6, odtot);
                bbdtot = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(dplankdn, odtot_rec)));
                bbd = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(dplankdn, od_rec)));
                bug = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(dplankup, od_rec)));
                but = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(dplankup, odtot_rec)));
            } else if (odepth <= 0.06f) {
                at = RLW_SU(odepth, RLW_MU(0.5f, RLW_MU(odepth, odepth)));
                float od_rec = RLW_MU(RLW_REC6, odepth);
                gassrc = RLW_MU(RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(dplankdn, od_rec))), at);
                float tblind = RLW_DV(odtot, RLW_AD(bpade, odtot));
                int ittot = (int)RLW_AD(RLW_MU(RLW_TBLINT, tblind), 0.5f);
                float tfactot = tfn_tbl[ittot];
                bbdtot = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(tfactot, dplankdn)));
                bbd = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(dplankdn, od_rec)));
                ao = RLW_SU(1.0f, exp_tbl[ittot]);
                bug = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(dplankup, od_rec)));
                but = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(tfactot, dplankup)));
            } else {
                float tblind = RLW_DV(odepth, RLW_AD(bpade, odepth));
                int itgas = (int)RLW_AD(RLW_MU(RLW_TBLINT, tblind), 0.5f);
                odepth = tau_tbl[itgas];
                at = RLW_SU(1.0f, exp_tbl[itgas]);
                float tfacgas = tfn_tbl[itgas];
                gassrc = RLW_MU(RLW_MU(at, plfrac),
                    RLW_AD(blay, RLW_MU(tfacgas, dplankdn)));
                float odtot2 = RLW_AD(odepth, oc);
                float tblind2 = RLW_DV(odtot2, RLW_AD(bpade, odtot2));
                int ittot = (int)RLW_AD(RLW_MU(RLW_TBLINT, tblind2), 0.5f);
                float tfactot = tfn_tbl[ittot];
                bbdtot = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(tfactot, dplankdn)));
                bbd = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(tfacgas, dplankdn)));
                ao = RLW_SU(1.0f, exp_tbl[ittot]);
                bug = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(tfacgas, dplankup)));
                but = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(tfactot, dplankup)));
            }
            float cf = cldfmc[((long long)col * NGPTLW + (igc - 1)) * nl
                              + (lev - 1)];
            float ef = PRO(efclfrac, lev);
            radld = RLW_SU(radld,
                RLW_MU(radld, RLW_AD(at, RLW_MU(ef, RLW_SU(1.0f, at)))));
            radld = RLW_AD(radld, gassrc);
            radld = RLW_AD(radld,
                RLW_MU(cf, RLW_SU(RLW_MU(bbdtot, ao), gassrc)));
            atrans[lev - 1] = at;
            atot[lev - 1] = ao;
            bbugas[lev - 1] = bug;
            bbutot[lev - 1] = but;
        } else {
            float at, bug;
            if (odepth <= 0.06f) {
                at = RLW_SU(odepth, RLW_MU(0.5f, RLW_MU(odepth, odepth)));
                float od2 = RLW_MU(RLW_REC6, odepth);
                bbd = RLW_MU(plfrac, RLW_AD(blay, RLW_MU(dplankdn, od2)));
                bug = RLW_MU(plfrac, RLW_AD(blay, RLW_MU(dplankup, od2)));
            } else {
                float tblind = RLW_DV(odepth, RLW_AD(bpade, odepth));
                int itr = (int)RLW_AD(RLW_MU(RLW_TBLINT, tblind), 0.5f);
                float transc = exp_tbl[itr];
                at = RLW_SU(1.0f, transc);
                float tausfac = tfn_tbl[itr];
                bbd = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(tausfac, dplankdn)));
                bug = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(tausfac, dplankup)));
            }
            radld = RLW_AD(radld, RLW_MU(RLW_SU(bbd, radld), at));
            atrans[lev - 1] = at;
            bbugas[lev - 1] = bug;
        }
        PRO(radld_p, lev) = radld;
        if (iclddn == 1) {
            radclrd = RLW_AD(radclrd,
                             RLW_MU(RLW_SU(bbd, radclrd), atrans[lev - 1]));
        } else {
            radclrd = radld;
        }
        PRO(radclrd_p, lev) = radclrd;
        iclddn_p[((long long)col * NGPTLW + (igc - 1)) * nl + (lev - 1)] =
            (unsigned char)iclddn;
    }

    float rad0 = RLW_MU(FR(1),
                        plankbnd[(long long)col * NBNDLW + (iband - 1)]);
    float reflect = RLW_SU(1.0f,
                           semiss[(long long)col * NBNDLW + (iband - 1)]);
    float radlu = RLW_AD(rad0, RLW_MU(reflect, radld));
    float radclru = RLW_AD(rad0, RLW_MU(reflect, radclrd));
    radlu_sfc[(long long)col * NGPTLW + (igc - 1)] = radlu;
    radclru_sfc[(long long)col * NGPTLW + (igc - 1)] = radclru;

    for (int lev = 1; lev <= nl; ++lev) {
        if (icldlyr[(long long)col * nl + (lev - 1)] == 1) {
            float gassrc = RLW_MU(bbugas[lev - 1], atrans[lev - 1]);
            float cf = cldfmc[((long long)col * NGPTLW + (igc - 1)) * nl
                              + (lev - 1)];
            float ef = PRO(efclfrac, lev);
            radlu = RLW_SU(radlu,
                RLW_MU(radlu, RLW_AD(atrans[lev - 1],
                    RLW_MU(ef, RLW_SU(1.0f, atrans[lev - 1])))));
            radlu = RLW_AD(radlu, gassrc);
            radlu = RLW_AD(radlu,
                RLW_MU(cf, RLW_SU(RLW_MU(bbutot[lev - 1], atot[lev - 1]),
                                  gassrc)));
        } else {
            radlu = RLW_AD(radlu,
                RLW_MU(RLW_SU(bbugas[lev - 1], radlu), atrans[lev - 1]));
        }
        PRO(radlu_p, lev) = radlu;
        if (iclddn == 1) {
            radclru = RLW_AD(radclru,
                RLW_MU(RLW_SU(bbugas[lev - 1], radclru), atrans[lev - 1]));
        } else {
            radclru = radlu;
        }
        PRO(radclru_p, lev) = radclru;
    }
#undef PRO
#undef PLLAY2
#undef PLLEV2
#undef FR
#undef TT
}

extern "C" __global__ void rlw_rtrn_accum(
    int ncol, int nl,
    const float* __restrict__ radld_p,     // (ncol, NGPTLW, nl)
    const float* __restrict__ radclrd_p,
    const float* __restrict__ radlu_p,
    const float* __restrict__ radclru_p,
    const unsigned char* __restrict__ iclddn_p,
    const float* __restrict__ radlu_sfc,   // (ncol, NGPTLW)
    const float* __restrict__ radclru_sfc,
    const int* __restrict__ ngs,           // (16) cumulative g-points
    const float* __restrict__ delwave,     // (16)
    float wtdiff,
    float* __restrict__ totuflux,          // (ncol, nl+1)
    float* __restrict__ totdflux,
    float* __restrict__ totuclfl,
    float* __restrict__ totdclfl)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)ncol * (nl + 1)) return;
    int col = (int)(tid / (nl + 1));
    int lev = (int)(tid % (nl + 1));   // 0..nl

    float totu = 0.0f, totd = 0.0f, totuc = 0.0f, totdc = 0.0f;
#define LP(a, lane) a[((long long)col * NGPTLW + (lane)) * nl + (lev - 1)]
#define LP0(a, lane) a[((long long)col * NGPTLW + (lane)) * nl + (lev)]
    for (int iband = 1; iband <= NBNDLW; ++iband) {
        int g_lo = (iband == 1) ? 0 : ngs[iband - 2];
        int g_hi = ngs[iband - 1];
        float urad = 0.0f, drad = 0.0f, clru = 0.0f, clrd = 0.0f;
        for (int lane = g_lo; lane < g_hi; ++lane) {
            // downward: drad/clrdrad live at levels 0..nl-1
            if (lev <= nl - 1) {
                drad = RLW_AD(drad, LP0(radld_p, lane));
                if (iclddn_p[((long long)col * NGPTLW + lane) * nl + lev])
                    clrd = RLW_AD(clrd, LP0(radclrd_p, lane));
                else
                    clrd = drad;
            }
            // upward: urad/clrurad at levels 0 (sfc) and 1..nl
            if (lev == 0) {
                urad = RLW_AD(urad,
                    radlu_sfc[(long long)col * NGPTLW + lane]);
                if (iclddn_p[((long long)col * NGPTLW + lane) * nl + 0])
                    clru = RLW_AD(clru,
                        radclru_sfc[(long long)col * NGPTLW + lane]);
                else
                    clru = urad;
            } else {
                urad = RLW_AD(urad, LP(radlu_p, lane));
                if (iclddn_p[((long long)col * NGPTLW + lane) * nl + 0])
                    clru = RLW_AD(clru, LP(radclru_p, lane));
                else
                    clru = urad;
            }
        }
        float dw = delwave[iband - 1];
        totu = RLW_AD(totu, RLW_MU(RLW_MU(urad, wtdiff), dw));
        totd = RLW_AD(totd, RLW_MU(RLW_MU(drad, wtdiff), dw));
        totuc = RLW_AD(totuc, RLW_MU(RLW_MU(clru, wtdiff), dw));
        totdc = RLW_AD(totdc, RLW_MU(RLW_MU(clrd, wtdiff), dw));
    }
    totuflux[(long long)col * (nl + 1) + lev] = totu;
    totdflux[(long long)col * (nl + 1) + lev] = totd;
    totuclfl[(long long)col * (nl + 1) + lev] = totuc;
    totdclfl[(long long)col * (nl + 1) + lev] = totdc;
#undef LP
#undef LP0
}

extern "C" __global__ void rlw_rtrn_final(
    int ncol, int nl,
    const float* __restrict__ pz,       // (ncol, nl+1)
    float fluxfac, float heatfac,
    float* __restrict__ totuflux,       // (ncol, nl+1) in/out
    float* __restrict__ totdflux,
    float* __restrict__ totuclfl,
    float* __restrict__ totdclfl,
    float* __restrict__ fnet,           // (ncol, nl+1)
    float* __restrict__ fnetc,
    float* __restrict__ htr,            // (ncol, nl+1)
    float* __restrict__ htrc)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
#define LV(a, l) a[(long long)col * (nl + 1) + (l)]
    LV(totuflux, 0) = RLW_MU(LV(totuflux, 0), fluxfac);
    LV(totdflux, 0) = RLW_MU(LV(totdflux, 0), fluxfac);
    LV(fnet, 0) = RLW_SU(LV(totuflux, 0), LV(totdflux, 0));
    LV(totuclfl, 0) = RLW_MU(LV(totuclfl, 0), fluxfac);
    LV(totdclfl, 0) = RLW_MU(LV(totdclfl, 0), fluxfac);
    LV(fnetc, 0) = RLW_SU(LV(totuclfl, 0), LV(totdclfl, 0));
    for (int lev = 1; lev <= nl; ++lev) {
        LV(totuflux, lev) = RLW_MU(LV(totuflux, lev), fluxfac);
        LV(totdflux, lev) = RLW_MU(LV(totdflux, lev), fluxfac);
        LV(fnet, lev) = RLW_SU(LV(totuflux, lev), LV(totdflux, lev));
        LV(totuclfl, lev) = RLW_MU(LV(totuclfl, lev), fluxfac);
        LV(totdclfl, lev) = RLW_MU(LV(totdclfl, lev), fluxfac);
        LV(fnetc, lev) = RLW_SU(LV(totuclfl, lev), LV(totdclfl, lev));
        int l0 = lev - 1;
        LV(htr, l0) = RLW_DV(
            RLW_MU(heatfac, RLW_SU(LV(fnet, l0), LV(fnet, lev))),
            RLW_SU(LV(pz, l0), LV(pz, lev)));
        LV(htrc, l0) = RLW_DV(
            RLW_MU(heatfac, RLW_SU(LV(fnetc, l0), LV(fnetc, lev))),
            RLW_SU(LV(pz, l0), LV(pz, lev)));
    }
    LV(htr, nl) = 0.0f;
    LV(htrc, nl) = 0.0f;
#undef LV
}
