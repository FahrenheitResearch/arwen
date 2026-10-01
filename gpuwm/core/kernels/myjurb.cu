// WRF v4.7.1 MYJURB, one thread per bottom-up gpuwm column.
// Internal m is Fortran K-1. Inputs flip on load, outputs flip on store.
// Unlike myjpbl.cu, ZINT starts at input HT exactly as WRF does. Terrain
// offsets affect float32 rounding and must be retained for oracle identity.
// Source arithmetic uses FMUL to prevent FMA contraction and glibc gfk_*.
// IDIFF defaults to 0; IDIFF=1 skips tendencies and THZ0/QZ0/QSFC, but MIXLEN still zeros CT.
// LOWLYR is 1, as in the existing engine launcher; full columns only.
// EXCH_H/M level 0 is INOUT and remains untouched, as in the Fortran.
// A_E and DLG_BEP are accepted but unused by the active Fortran equations.
// VDIFX is not called by MYJURB. MYJURBINIT is initialization, not a step.
// Oracle comparison is pending; compilation alone does not prove identity.
// module_bl_myjurb.F:130-771
#ifndef MYJ_KMAX
#define MYJ_KMAX 128
#endif

// module_bl_myjurb.F:28-124 MYJP_G = 9.81f
#define MYJP_G 9.8100004196166992f
// module_bl_myjurb.F:28-124 MYJP_RD = 287.0f
#define MYJP_RD 287.0f
// module_bl_myjurb.F:28-124 MYJP_CP = (7.0f * 287.0f / 2.0f)
#define MYJP_CP 1004.5f
// module_bl_myjurb.F:28-124 MYJP_XLV = 2.5e6f
#define MYJP_XLV 2500000.0f
// module_bl_myjurb.F:28-124 MYJP_XLS = 2.85e6f
#define MYJP_XLS 2850000.0f
// module_bl_myjurb.F:28-124 MYJP_P608 = (461.6f / 287.0f - 1.0f)
#define MYJP_P608 0.60836243629455566f
// module_bl_myjurb.F:28-124 MYJP_PQ0 = 379.90516f
#define MYJP_PQ0 379.9051513671875f
// module_bl_myjurb.F:28-124 MYJP_A2 = 17.2693882f
#define MYJP_A2 17.269388198852539f
// module_bl_myjurb.F:28-124 MYJP_A3 = 273.16f
#define MYJP_A3 273.16000366210938f
// module_bl_myjurb.F:28-124 MYJP_A4 = 35.86f
#define MYJP_A4 35.860000610351562f
// module_bl_myjurb.F:28-124 MYJP_EPSQ2 = 0.2f
#define MYJP_EPSQ2 0.20000000298023224f

// module_bl_myjurb.F:28-124 PBL_VKARMAN = 0.4f
#define PBL_VKARMAN 0.40000000596046448f
// module_bl_myjurb.F:28-124 PBL_CAPA = (MYJP_RD / MYJP_CP)
#define PBL_CAPA 0.28571429848670959f
// module_bl_myjurb.F:28-124 PBL_RLIVWV = (MYJP_XLS / MYJP_XLV)
#define PBL_RLIVWV 1.1399999856948853f
// module_bl_myjurb.F:28-124 PBL_ELOCP = (2.72e6f / MYJP_CP)
#define PBL_ELOCP 2707.81494140625f
// module_bl_myjurb.F:28-124 PBL_EPS1 = 1.0e-12f
#define PBL_EPS1 9.999999960041972e-13f
// module_bl_myjurb.F:28-124 PBL_EPS2 = 0.0f
#define PBL_EPS2 0.0f
// module_bl_myjurb.F:28-124 PBL_EPSL = 0.32f
#define PBL_EPSL 0.31999999284744263f
// module_bl_myjurb.F:28-124 PBL_EPSRU = 1.0e-7f
#define PBL_EPSRU 1.0000000116860974e-07f
// module_bl_myjurb.F:28-124 PBL_EPSRS = 1.0e-7f
#define PBL_EPSRS 1.0000000116860974e-07f
// module_bl_myjurb.F:28-124 PBL_EPSTRB = 1.0e-24f
#define PBL_EPSTRB 1.0000000195414814e-24f
// module_bl_myjurb.F:28-124 PBL_FH = 1.01f
#define PBL_FH 1.0099999904632568f
// module_bl_myjurb.F:28-124 PBL_ALPH = 0.30f
#define PBL_ALPH 0.30000001192092896f
// module_bl_myjurb.F:28-124 PBL_BETA = (1.0f / 273.0f)
#define PBL_BETA 0.0036630036775022745f
// module_bl_myjurb.F:28-124 PBL_EL0MAX = 1000.0f
#define PBL_EL0MAX 1000.0f
// module_bl_myjurb.F:28-124 PBL_EL0MIN = 1.0f
#define PBL_EL0MIN 1.0f
// module_bl_myjurb.F:28-124 PBL_ELFC = (0.23f * 0.5f)
#define PBL_ELFC 0.11500000208616257f
// module_bl_myjurb.F:28-124 PBL_A1 = 0.659888514560862645f
#define PBL_A1 0.65988850593566895f
// module_bl_myjurb.F:28-124 PBL_A2X = 0.6574209922667784586f
#define PBL_A2X 0.65742099285125732f
// module_bl_myjurb.F:28-124 PBL_B1 = 11.87799326209552761f
#define PBL_B1 11.877993583679199f
// module_bl_myjurb.F:28-124 PBL_B2 = 7.226971804046074028f
#define PBL_B2 7.2269716262817383f
// module_bl_myjurb.F:28-124 PBL_C1 = 0.000830955950095854396f
#define PBL_C1 0.00083095597801730037f
// module_bl_myjurb.F:28-124 PBL_ELZ0 = 0.0f
#define PBL_ELZ0 0.0f
// module_bl_myjurb.F:28-124 PBL_ESQ = 5.0f
#define PBL_ESQ 5.0f
// module_bl_myjurb.F:28-124 PBL_SEAFC = 0.98f
#define PBL_SEAFC 0.98000001907348633f
// module_bl_myjurb.F:28-124 PBL_PQ0SEA = (MYJP_PQ0 * PBL_SEAFC)
#define PBL_PQ0SEA 372.30706787109375f
// module_bl_myjurb.F:28-124 PBL_BTG = (PBL_BETA * MYJP_G)
#define PBL_BTG 0.035934068262577057f
// module_bl_myjurb.F:28-124 PBL_RB1 = (1.0f / PBL_B1)
#define PBL_RB1 0.08418930321931839f

// module_bl_myjurb.F:28-124 PBL_ADNH = (9.0f*PBL_A1*PBL_A2X*PBL_A2X*(12.0f*PBL_A1+3.0f*PBL_B2)*PBL_BTG*PBL_BTG)
#define PBL_ADNH 0.098106667399406433f
// module_bl_myjurb.F:28-124 PBL_ADNM = (18.0f*PBL_A1*PBL_A1*PBL_A2X*(PBL_B2-3.0f*PBL_A2X)*PBL_BTG)
#define PBL_ADNM 0.97299867868423462f
// module_bl_myjurb.F:28-124 PBL_ANMH = (-9.0f*PBL_A1*PBL_A2X*PBL_A2X*PBL_BTG*PBL_BTG)
#define PBL_ANMH -0.0033144615590572357f
// module_bl_myjurb.F:28-124 PBL_ANMM = (-3.0f*PBL_A1*PBL_A2X*(3.0f*PBL_A2X+3.0f*PBL_B2*PBL_C1+18.0f*PBL_A1*PBL_C1-PBL_B2)*PBL_BTG)
#define PBL_ANMM 0.2444441020488739f
// module_bl_myjurb.F:28-124 PBL_BDNH = (3.0f*PBL_A2X*(7.0f*PBL_A1+PBL_B2)*PBL_BTG)
#define PBL_BDNH 0.83955651521682739f
// module_bl_myjurb.F:28-124 PBL_BDNM = (6.0f*PBL_A1*PBL_A1)
#define PBL_BDNM 2.6127171516418457f
// module_bl_myjurb.F:28-124 PBL_BEQH = (PBL_A2X*PBL_B1*PBL_BTG+3.0f*PBL_A2X*(7.0f*PBL_A1+PBL_B2)*PBL_BTG)
#define PBL_BEQH 1.1201599836349487f
// module_bl_myjurb.F:28-124 PBL_BEQM = (-PBL_A1*PBL_B1*(1.0f-3.0f*PBL_C1)+6.0f*PBL_A1*PBL_A1)
#define PBL_BEQM -5.205894947052002f
// module_bl_myjurb.F:28-124 PBL_BNMH = (-PBL_A2X*PBL_BTG)
#define PBL_BNMH -0.023623811081051826f
// module_bl_myjurb.F:28-124 PBL_BNMM = (PBL_A1*(1.0f-3.0f*PBL_C1))
#define PBL_BNMM 0.65824347734451294f
// module_bl_myjurb.F:28-124 PBL_BSHH = (9.0f*PBL_A1*PBL_A2X*PBL_A2X*PBL_BTG)
#define PBL_BSHH 0.092237301170825958f
// module_bl_myjurb.F:28-124 PBL_BSHM = (18.0f*PBL_A1*PBL_A1*PBL_A2X*PBL_C1)
#define PBL_BSHM 0.0042818873189389706f
// module_bl_myjurb.F:28-124 PBL_BSMH = (-3.0f*PBL_A1*PBL_A2X*(3.0f*PBL_A2X+3.0f*PBL_B2*PBL_C1+12.0f*PBL_A1*PBL_C1-PBL_B2)*PBL_BTG)
#define PBL_BSMH 0.24459792673587799f
// module_bl_myjurb.F:28-124 PBL_CESH = PBL_A2X
#define PBL_CESH 0.65742099285125732f
// module_bl_myjurb.F:28-124 PBL_CESM = (PBL_A1*(1.0f-3.0f*PBL_C1))
#define PBL_CESM 0.65824347734451294f

// module_bl_myjurb.F:28-124 PBL_AEQH = (9.0f*PBL_A1*PBL_A2X*PBL_A2X*PBL_B1*PBL_BTG*PBL_BTG                    + 9.0f*PBL_A1*PBL_A2X*PBL_A2X*(12.0f*PBL_A1+3.0f*PBL_B2)*PBL_BTG*PBL_BTG)
#define PBL_AEQH 0.13747581839561462f
// module_bl_myjurb.F:28-124 PBL_AEQM = (3.0f*PBL_A1*PBL_A2X*PBL_B1*(3.0f*PBL_A2X+3.0f*PBL_B2*PBL_C1+18.0f*PBL_A1*PBL_C1-PBL_B2)*PBL_BTG                    + 18.0f*PBL_A1*PBL_A1*PBL_A2X*(PBL_B2-3.0f*PBL_A2X)*PBL_BTG)
#define PBL_AEQM -1.9305069446563721f

// module_bl_myjurb.F:28-124 PBL_REQU = (-PBL_AEQH/PBL_AEQM)
#define PBL_REQU 0.071212291717529297f
// module_bl_myjurb.F:28-124 PBL_EPSGH = 1.0e-9f
#define PBL_EPSGH 9.9999997171806854e-10f
// module_bl_myjurb.F:28-124 PBL_EPSGM = (PBL_REQU*PBL_EPSGH)
#define PBL_EPSGM 7.1212293006883698e-11f
// module_bl_myjurb.F:28-124 PBL_UBRYL = ((18.0f*PBL_REQU*PBL_A1*PBL_A1*PBL_A2X*PBL_B2*PBL_C1*PBL_BTG                      + 9.0f*PBL_A1*PBL_A2X*PBL_A2X*PBL_B2*PBL_BTG*PBL_BTG)                     / (PBL_REQU*PBL_ADNM+PBL_ADNH))
#define PBL_UBRYL 0.14356787502765656f
// module_bl_myjurb.F:28-124 PBL_UBRY = ((1.0f+PBL_EPSRS)*PBL_UBRYL)
#define PBL_UBRY 0.14356788992881775f
// module_bl_myjurb.F:28-124 PBL_UBRY3 = (3.0f*PBL_UBRY)
#define PBL_UBRY3 0.43070366978645325f
// module_bl_myjurb.F:28-124 PBL_AUBH = (27.0f*PBL_A1*PBL_A2X*PBL_A2X*PBL_B2*PBL_BTG*PBL_BTG - PBL_ADNH*PBL_UBRY3)
#define PBL_AUBH 0.029605656862258911f
// module_bl_myjurb.F:28-124 PBL_AUBM = (54.0f*PBL_A1*PBL_A1*PBL_A2X*PBL_B2*PBL_C1*PBL_BTG - PBL_ADNM*PBL_UBRY3)
#define PBL_AUBM -0.41573813557624817f
// module_bl_myjurb.F:28-124 PBL_BUBH = ((9.0f*PBL_A1*PBL_A2X+3.0f*PBL_A2X*PBL_B2)*PBL_BTG - PBL_BDNH*PBL_UBRY3)
#define PBL_BUBH 0.2908875048160553f
// module_bl_myjurb.F:28-124 PBL_BUBM = (18.0f*PBL_A1*PBL_A1*PBL_C1 - PBL_BDNM*PBL_UBRY3)
#define PBL_BUBM -1.1187937259674072f
// module_bl_myjurb.F:28-124 PBL_CUBR = (1.0f - PBL_UBRY3)
#define PBL_CUBR 0.56929636001586914f
// module_bl_myjurb.F:28-124 PBL_RCUBR = (1.0f/PBL_CUBR)
#define PBL_RCUBR 1.7565543651580811f

extern "C" __global__ void myjurb_column(
    const real* __restrict__ dz_a,
    const real* __restrict__ u_a, const real* __restrict__ v_a,
    const real* __restrict__ t_a, const real* __restrict__ th_a,
    const real* __restrict__ exner_a, const real* __restrict__ qv_a,
    const real* __restrict__ qc_a,
    const real* __restrict__ p_a,
    real* __restrict__ tke_a,
    const real* __restrict__ psfc_a, const real* __restrict__ ust_a,
    const real* __restrict__ tsk_a, const real* __restrict__ chklowq_a,
    const real* __restrict__ xland_a, const real* __restrict__ sice_a,
    const real* __restrict__ snow_a, const real* __restrict__ akhs_a,
    const real* __restrict__ akms_a, const real* __restrict__ elflx_a,
    const real* __restrict__ uz0_a, const real* __restrict__ vz0_a,
    real* __restrict__ thz0_a, real* __restrict__ qz0_a,
    real* __restrict__ qsfc_a, real* __restrict__ ct_a,
    real* __restrict__ rublten_a, real* __restrict__ rvblten_a,
    real* __restrict__ rthblten_a, real* __restrict__ rqvblten_a,
    real* __restrict__ rqcblten_a,
    real* __restrict__ el_myj_a, real* __restrict__ exch_h_a,
    real* __restrict__ exch_m_a,
    real* __restrict__ pblh_a, int* __restrict__ kpbl_a,
    real* __restrict__ mixht_a,
    const real* __restrict__ a_u_bep,
    const real* __restrict__ a_v_bep,
    const real* __restrict__ a_t_bep,
    const real* __restrict__ a_q_bep,
    const real* __restrict__ a_e_bep,
    const real* __restrict__ b_u_bep,
    const real* __restrict__ b_v_bep,
    const real* __restrict__ b_t_bep,
    const real* __restrict__ b_q_bep,
    const real* __restrict__ b_e_bep,
    const real* __restrict__ dlg_bep,
    const real* __restrict__ dl_u_bep,
    const real* __restrict__ vl_bep,
    const real* __restrict__ sf_bep,
    const real* __restrict__ frc_urb2d, const real* __restrict__ ht,
    real dtturbl, int flag_bep, int idiff, int nz, int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    if (nz > MYJ_KMAX || nz < 4) return;
    size_t st = (size_t)n;
#define MYJ_UP(a, kup) a[(size_t)(kup) * st + col]

    const int lmh = nz;
    const int nzm = nz - 1;
    real rdtturbl = (1.0f / dtturbl);
    real dtdif = dtturbl;

    real zh[MYJ_KMAX + 1];
    real uk[MYJ_KMAX], vk[MYJ_KMAX], tk[MYJ_KMAX];
    real the[MYJ_KMAX], qk[MYJ_KMAX], cwm[MYJ_KMAX];
    real q2[MYJ_KMAX], rhok[MYJ_KMAX];
    real gm[MYJ_KMAX], gh[MYJ_KMAX], el[MYJ_KMAX];
    real akm[MYJ_KMAX], akh[MYJ_KMAX];
    real sf[MYJ_KMAX], vl[MYJ_KMAX], dl[MYJ_KMAX];
    real be[MYJ_KMAX], beface[MYJ_KMAX], dzk[MYJ_KMAX];
    real s1[MYJ_KMAX], s2[MYJ_KMAX], s3[MYJ_KMAX], s4[MYJ_KMAX];

    // module_bl_myjurb.F:283-351
    zh[nz] = ht[col];
    for (int m = (nz - 1); m >= 0; --m) {
        zh[m] = (zh[(m + 1)] + MYJ_UP(dz_a, ((nz - 1) - m)));
    }
    for (int m = 0; m < nz; ++m) {
        int kup = ((nz - 1) - m);
        uk[m] = MYJ_UP(u_a, kup);
        vk[m] = MYJ_UP(v_a, kup);
        tk[m] = MYJ_UP(t_a, kup);
        real thx = MYJ_UP(th_a, kup);
        real ratiomx = MYJ_UP(qv_a, kup);
        qk[m] = (ratiomx / ((1.0f + ratiomx)));
        real cw = MYJ_UP(qc_a, kup);
        cwm[m] = cw;
        the[m] = FMUL(((FMUL(cw, ((-PBL_ELOCP / tk[m]))) + 1.0f)), thx);
        q2[m] = FMUL(2.0f, MYJ_UP(tke_a, kup));
    }

    // module_bl_myjurb.F:354-391 BEP flip and face interpolation.
    for (int m = (nz - 1); m >= 0; --m) {
        int kup = ((nz - 1) - m);
        dzk[m] = MYJ_UP(dz_a, kup);
        be[m] = (flag_bep ? MYJ_UP(b_e_bep, kup) : 0.0f);
        sf[m] = (flag_bep ? MYJ_UP(sf_bep, kup) : 1.0f);
        vl[m] = (flag_bep ? MYJ_UP(vl_bep, kup) : 1.0f);
        dl[m] = (flag_bep ? MYJ_UP(dl_u_bep, kup) : 0.0f);
        beface[m] = 0.0f;
    }
    if (flag_bep) {
        for (int m = (nz - 2); m >= 1; --m) {
            real slope = ((((be[(m - 1)] - be[m])) / 0.5f) / ((dzk[(m - 1)] + dzk[m])));
            real icept = (be[m] - FMUL(slope, ((zh[m] - FMUL(0.5f, dzk[m])))));
            beface[m] = FMUL(2.0f, ((FMUL(slope, zh[m]) + icept)));
        }
    }

    // module_bl_myjurb.F:831-861
    int lpbl = lmh;
    for (int kf = (lmh - 1); kf >= 1; --kf) {
        if (q2[kf - 1] <= FMUL(MYJP_EPSQ2, PBL_FH)) { lpbl = kf; goto mixlen_110; }
    }
    lpbl = 1;
mixlen_110:
    {
        real pblh = (zh[lpbl] - zh[lmh]);
        pblh_a[col] = pblh;
    }
    for (int m = 0; m < nzm; ++m) s1[m] = (the[m] - the[(m + 1)]);
    {
        real ct = ct_a[col];
        for (int kf = (lmh - 2); kf >= 1; --kf) {
            if (s1[kf - 1] > 0.0f && s1[kf] <= 0.0f) {
                s1[kf - 1] = (s1[(kf - 1)] + ct);
                break;
            }
        }
        ct_a[col] = 0.0f;
    }
    // module_bl_myjurb.F:866-972
    for (int m = 0; m < nzm; ++m) {
        real rdz = (2.0f / ((zh[m] - zh[(m + 2)])));
        real gml = FMUL(FMUL(((FMUL(((uk[m] - uk[(m + 1)])), ((uk[m] - uk[(m + 1)]))) + FMUL(((vk[m] - vk[(m + 1)])), ((vk[m] - vk[(m + 1)]))))), rdz), rdz);
        gm[m] = fmaxf(gml, PBL_EPSGM);
        real tem = FMUL(((tk[m] + tk[(m + 1)])), 0.5f);
        real thm = FMUL(((the[m] + the[(m + 1)])), 0.5f);
        real a = FMUL(thm, MYJP_P608);
        real b = FMUL(((((PBL_ELOCP / tem) - 1.0f) - MYJP_P608)), thm);
        real ghl = FMUL((((FMUL(s1[m], ((FMUL(((((qk[m] + qk[(m + 1)]) + cwm[m]) + cwm[(m + 1)])), (FMUL(0.5f, MYJP_P608))) + 1.0f))) + FMUL(((((qk[m] - qk[(m + 1)]) + cwm[m]) - cwm[(m + 1)])), a)) + FMUL(((cwm[m] - cwm[(m + 1)])), b))), rdz);
        if (fabsf(ghl) <= PBL_EPSGH) ghl = PBL_EPSGH;
        gh[m] = ghl;
    }
    // module_bl_myjurb.F:890-929
    int lmxl = lmh;
    for (int m = 0; m < nzm; ++m) {
        real gml = gm[m]; real ghl = gh[m]; real eloq2x;
        if (ghl >= PBL_EPSGH) {
            if (gml / ghl <= PBL_REQU) {
                akm[m] = PBL_EPSL;
                lmxl = (m + 1);
                continue;
            }
            real aubr = FMUL(((FMUL(PBL_AUBM, gml) + FMUL(PBL_AUBH, ghl))), ghl);
            real bubr = (FMUL(PBL_BUBM, gml) + FMUL(PBL_BUBH, ghl));
            real qol2st = FMUL(((FMUL(-0.5f, bubr) + sqrtf((FMUL(FMUL(bubr, bubr), 0.25f) - FMUL(aubr, PBL_CUBR))))), PBL_RCUBR);
            eloq2x = (1.0f / qol2st);
        } else {
            real aden = FMUL(((FMUL(PBL_ADNM, gml) + FMUL(PBL_ADNH, ghl))), ghl);
            real bden = (FMUL(PBL_BDNM, gml) + FMUL(PBL_BDNH, ghl));
            real qol2un = (FMUL(-0.5f, bden) + sqrtf((FMUL(FMUL(bden, bden), 0.25f) - aden)));
            eloq2x = (1.0f / ((qol2un + PBL_EPSRU)));
        }
        akm[m] = fmaxf(sqrtf(FMUL(eloq2x, q2[m])), PBL_EPSL);
    }
    if (akm[lmh - 2] == PBL_EPSL) lmxl = lmh;
    mixht_a[col] = (zh[(lmxl - 1)] - zh[lmh]);
    // module_bl_myjurb.F:930-979
    for (int m = 0; m < lmh; ++m) s2[m] = 0.0f;
    for (int m = (lpbl - 1); m < lmh; ++m) s2[m] = sqrtf(q2[m]);
    {
        real szq = 0.0f; real sq = 0.0f;
        for (int m = 0; m < nzm; ++m) {
            real qdzl = FMUL(((s2[m] + s2[(m + 1)])), ((zh[(m + 1)] - zh[(m + 2)])));
            szq = (FMUL(((((zh[(m + 1)] + zh[(m + 2)]) - zh[lmh]) - zh[lmh])), qdzl) + szq);
            sq = (qdzl + sq);
        }
        real el0 = fminf((FMUL(FMUL(PBL_ALPH, szq), 0.5f) / sq), PBL_EL0MAX);
        el0 = fmaxf(el0, PBL_EL0MIN);
        int lpblm = max((lpbl - 1), 1);
        for (int m = 0; m < lpblm; ++m) {
            el[m] = fminf(FMUL(((zh[m] - zh[(m + 2)])), PBL_ELFC), akm[m]);
            akh[m] = (el[m] / akm[m]);
        }
        if (lpbl < lmh) {
            for (int m = (lpbl - 1); m < lmh - 1; ++m) {
                real vkrmz = FMUL(((zh[(m + 1)] - zh[lmh])), PBL_VKARMAN);
                el[m] = fminf((vkrmz / (((vkrmz / el0) + 1.0f))), akm[m]);
                // module_bl_myjurb.F:960-969 urban mixing length.
                if (dl[m] > 0.0f) el[m] = (1.0f / (((1.0f / el[m]) + (1.0f / dl[m]))));
                akh[m] = (el[m] / akm[m]);
            }
        }
        for (int m = lpbl; m < lmh - 2; ++m) {
            real srel = fminf(FMUL(((FMUL(((akh[(m - 1)] + akh[(m + 1)])), 0.5f) + akh[m])), 0.5f), akh[m]);
            el[m] = fmaxf(FMUL(srel, akm[m]), PBL_EPSL);
        }
    }

    {
        // module_bl_myjurb.F:1027-1180
        real ustar = ust_a[col];
        for (int m = 0; m < lmh - 1; ++m) {
            real gml = gm[m]; real ghl = gh[m];
            real aequ = FMUL(((FMUL(PBL_AEQM, gml) + FMUL(PBL_AEQH, ghl))), ghl);
            real bequ = (FMUL(PBL_BEQM, gml) + FMUL(PBL_BEQH, ghl));
            real eqol2 = (FMUL(-0.5f, bequ) + sqrtf((FMUL(FMUL(bequ, bequ), 0.25f) - aequ)));
            if ((gml + FMUL(ghl, ghl) <= PBL_EPSTRB)
                || (ghl >= PBL_EPSGH && gml / ghl <= PBL_REQU)
                || (eqol2 <= PBL_EPS2)) {
                q2[m] = MYJP_EPSQ2;
                el[m] = PBL_EPSL;
                continue;
            }
            real anum = FMUL(((FMUL(PBL_ANMM, gml) + FMUL(PBL_ANMH, ghl))), ghl);
            real bnum = (FMUL(PBL_BNMM, gml) + FMUL(PBL_BNMH, ghl));
            real aden = FMUL(((FMUL(PBL_ADNM, gml) + FMUL(PBL_ADNH, ghl))), ghl);
            real bden = (FMUL(PBL_BDNM, gml) + FMUL(PBL_BDNH, ghl));
            real cden = 1.0f;
            real arhs = FMUL(-((FMUL(anum, bden) - FMUL(bnum, aden))), 2.0f);
            real brhs = FMUL(-anum, 4.0f);
            real crhs = FMUL(-bnum, 2.0f);
            real dloq1 = (el[m] / sqrtf(q2[m]));
            real eloq21 = (1.0f / eqol2);
            real eloq11 = sqrtf(eloq21);
            real eloq31 = FMUL(eloq21, eloq11);
            real eloq41 = FMUL(eloq21, eloq21);
            real eloq51 = FMUL(eloq21, eloq31);
            real rden1 = (1.0f / (((FMUL(aden, eloq41) + FMUL(bden, eloq21)) + cden)));
            real rhsp1 = FMUL(FMUL((((FMUL(arhs, eloq51) + FMUL(brhs, eloq31)) + FMUL(crhs, eloq11))), rden1), rden1);
            real eloq12 = (eloq11 + FMUL(((dloq1 - eloq11)), gfk_exp(FMUL(rhsp1, dtturbl))));
            eloq12 = fmaxf(eloq12, PBL_EPS1);
            real eloq22 = FMUL(eloq12, eloq12);
            real eloq32 = FMUL(eloq22, eloq12);
            real eloq42 = FMUL(eloq22, eloq22);
            real eloq52 = FMUL(eloq22, eloq32);
            real rden2 = (1.0f / (((FMUL(aden, eloq42) + FMUL(bden, eloq22)) + cden)));
            real rhs2 = (FMUL(-((FMUL(anum, eloq42) + FMUL(bnum, eloq22))), rden2) + PBL_RB1);
            real rhsp2 = FMUL(FMUL((((FMUL(arhs, eloq52) + FMUL(brhs, eloq32)) + FMUL(crhs, eloq12))), rden2), rden2);
            real rhst2 = (rhs2 / rhsp2);
            real eloq13 = ((eloq12 - rhst2) + FMUL((((rhst2 + dloq1) - eloq12)), gfk_exp(FMUL(rhsp2, dtturbl))));
            eloq13 = fmaxf(eloq13, PBL_EPS1);
            if (eloq13 > PBL_EPS1) {
                q2[m] = (FMUL(el[m], el[m]) / (FMUL(eloq13, eloq13)));
                q2[m] = fmaxf(q2[m], MYJP_EPSQ2);
                if (q2[m] == MYJP_EPSQ2) el[m] = PBL_EPSL;
            } else {
                q2[m] = MYJP_EPSQ2;
                el[m] = PBL_EPSL;
            }
        }
        q2[lmh - 1] = fmaxf(FMUL(FMUL(gfk_pow(PBL_B1, (2.0f / 3.0f)), ustar), ustar), MYJP_EPSQ2);
    }
    // module_bl_myjurb.F:426-429 lower TKE BEP correction.
    if (flag_bep) {
        q2[lmh-1] = (FMUL(q2[(lmh - 1)], ((1.0f - frc_urb2d[col]))) + gfk_pow((__fdiv_rn(FMUL(FMUL(beface[(lmh - 1)], 0.5f), el[(lmh - 2)]), 11.788f) / 2.0f), (2.0f / 3.0f)));
    }
    // module_bl_myjurb.F:437 KPBL publication
    kpbl_a[col] = ((nz - lpbl) + 1);

    // module_bl_myjurb.F:1230-1289 DIFCOF
    for (int m = 0; m < lmh - 1; ++m) {
        real ell = el[m];
        real eloq2 = (FMUL(ell, ell) / q2[m]);
        real eloq4 = FMUL(eloq2, eloq2);
        real gml = gm[m]; real ghl = gh[m];
        real aden = FMUL(((FMUL(PBL_ADNM, gml) + FMUL(PBL_ADNH, ghl))), ghl);
        real bden = (FMUL(PBL_BDNM, gml) + FMUL(PBL_BDNH, ghl));
        real cden = 1.0f;
        real besm = FMUL(PBL_BSMH, ghl);
        real besh = (FMUL(PBL_BSHM, gml) + FMUL(PBL_BSHH, ghl));
        real rden = (1.0f / (((FMUL(aden, eloq4) + FMUL(bden, eloq2)) + cden)));
        real esm = FMUL(((FMUL(besm, eloq2) + PBL_CESM)), rden);
        real esh = FMUL(((FMUL(besh, eloq2) + PBL_CESH)), rden);
        real rdz = (2.0f / ((zh[m] - zh[(m + 2)])));
        real q1l = sqrtf(q2[m]);
        real elqdz = FMUL(FMUL(ell, q1l), rdz);
        akm[m] = FMUL(elqdz, esm);
        akh[m] = FMUL(elqdz, esh);
    }
    // module_bl_myjurb.F:453-463
    for (int kup = 0; kup < nz - 1; ++kup) {
        int kflip = ((nz - 1) - kup);
        real deltaz = FMUL(0.5f, ((zh[(kflip - 1)] - zh[(kflip + 1)])));
        MYJ_UP(exch_h_a, kup+1) = FMUL(akh[(kflip - 1)], deltaz);
        MYJ_UP(exch_m_a, kup+1) = FMUL(akm[(kflip - 1)], deltaz);
    }


    {
        // module_bl_myjurb.F:1393-1426
        const real esqhf = FMUL(0.5f, PBL_ESQ);
        int nq = (lmh - 2);
        for (int m = 0; m < nq; ++m) {
            s1[m] = ((((dtdif + dtdif)) / ((zh[m] - zh[(m + 2)]))) / vl[m]);
            s2[m] = (FMUL(FMUL(FMUL(sf[m], sqrtf(FMUL(((q2[m] + q2[(m + 1)])), 0.5f))), ((el[m] + el[(m + 1)]))), esqhf) / ((zh[(m + 1)] - zh[(m + 2)])));
            s3[m] = FMUL(-s1[m], s2[m]);
        }
        s4[0] = (FMUL(s1[0], s2[0]) + 1.0f);
        rhok[0] = (q2[0] + FMUL(dtdif, beface[0]));
        for (int m = 1; m < nq; ++m) {
            real cf = (FMUL(-s1[m], s2[(m - 1)]) / s4[(m - 1)]);
            s4[m] = ((FMUL(-s3[(m - 1)], cf) + FMUL(((s2[(m - 1)] + s2[m])), s1[m])) + 1.0f);
            s4[m] = ((FMUL(-s3[(m - 1)], cf) + FMUL(((s2[(m - 1)] + s2[m])), s1[m])) + 1.0f);
            rhok[m] = ((FMUL(-rhok[(m - 1)], cf) + q2[m]) + FMUL(dtdif, beface[m]));
        }
        // module_bl_myjurb.F:1412-1415 K after DO is LMH-1, not LMH.
        real dtozs = ((((dtdif + dtdif)) / ((zh[(lmh - 2)] - zh[lmh]))) / vl[(lmh - 2)]);
        real akqs = (FMUL(FMUL(FMUL(sf[(lmh - 2)], sqrtf(FMUL(((q2[(lmh - 2)] + q2[(lmh - 1)])), 0.5f))), ((el[(lmh - 2)] + PBL_ELZ0))), esqhf) / ((zh[(lmh - 1)] - zh[lmh])));
        real cf = (FMUL(-dtozs, s2[(nq - 1)]) / s4[(nq - 1)]);
        q2[lmh - 2] = (((((FMUL(FMUL(dtozs, akqs), q2[(lmh - 1)]) - FMUL(rhok[(nq - 1)], cf)) + q2[(lmh - 2)]) + FMUL(dtdif, beface[(lmh - 2)]))) / (((FMUL(((s2[(nq - 1)] + akqs)), dtozs) - FMUL(s3[(nq - 1)], cf)) + 1.0f)));
        for (int m = (nq - 1); m >= 0; --m) {
            q2[m] = (((FMUL(-s3[m], q2[(m + 1)]) + rhok[m])) / s4[m]);
        }
    }
    // module_bl_myjurb.F:480-485
    for (int kup = 0; kup < nz; ++kup) {
        int kflip = (nz - kup);
        q2[kflip - 1] = fmaxf(q2[(kflip - 1)], MYJP_EPSQ2);
        MYJ_UP(tke_a, kup) = FMUL(0.5f, q2[(kflip - 1)]);
        MYJ_UP(el_myj_a, kup) = (((kflip < nz)) ? el[(kflip - 1)] : 0.0f);
    }

    if (idiff == 1) return; // module_bl_myjurb.F:491
    // module_bl_myjurb.F:500-587
    real psfc = psfc_a[col];
    real thsk = FMUL(tsk_a[col], gfk_pow((1.0e5f / psfc), PBL_CAPA));
    for (int m = 0; m < nz; ++m) {
        real pkm = MYJ_UP(p_a, ((nz - 1) - m));
        rhok[m] = (pkm / (FMUL(FMUL(MYJP_RD, tk[m]), (((1.0f + FMUL(MYJP_P608, qk[m])) - cwm[m])))));
    }
    for (int m = 0; m < nz - 1; ++m) {
        akh[m] = FMUL(FMUL(akh[m], 0.5f), ((rhok[m] + rhok[(m + 1)])));
    }
    real seamask = (xland_a[col] - 1.0f);
    real thz0 = (FMUL(((1.0f - seamask)), thsk) + FMUL(seamask, thz0_a[col]));
    thz0_a[col] = thz0;
    real akhs_dens = FMUL(akhs_a[col], rhok[(nz - 1)]);
    real qsfc = qsfc_a[col];
    if (seamask < 0.5f) {
        real qfc1 = FMUL(FMUL(MYJP_XLV, chklowq_a[col]), akhs_dens);
        if (snow_a[col] > 0.0f || sice_a[col] > 0.5f) qfc1 = FMUL(qfc1, PBL_RLIVWV);
        if (qfc1 > 0.0f) qsfc = (qk[(nz - 1)] + (elflx_a[col] / qfc1));
    } else {
        real exnsfc = gfk_pow((1.0e5f / psfc), PBL_CAPA);
        qsfc = FMUL((PBL_PQ0SEA / psfc), gfk_exp((FMUL(MYJP_A2, ((thsk - FMUL(MYJP_A3, exnsfc)))) / ((thsk - FMUL(MYJP_A4, exnsfc))))));
    }
    qsfc_a[col] = qsfc;
    real qz0 = (FMUL(((1.0f - seamask)), qsfc) + FMUL(seamask, qz0_a[col]));
    qz0_a[col] = qz0;

    // module_bl_myjurb.F:1436-1562 VDIFH, distinct T/Q/CWM diagonals.
    {
        int nh = (lmh - 1);
        for (int m = 0; m < nh; ++m) {
            akh[m] = FMUL(akh[m], sf[m]);
            s1[m] = ((dtdif / ((zh[m] - zh[(m + 1)]))) / vl[m]);
            s3[m] = FMUL(-s1[m], akh[m]);
            s2[m] = FMUL(FMUL(akh[m], ((zh[m] - zh[(m + 2)]))), (FMUL(0.5f, ct_a[col])));
        }
        for (int species = 0; species < 3; ++species) {
            real* var = ((species == 0) ? the : ((species == 1) ? qk : cwm));
            // module_bl_myjurb.F:1499-1526 forward sweep.
            for (int m = 0; m < nh; ++m) {
                int kup = ((nz - 1) - m);
                real a = 0.0f; real b = 0.0f;
                if (flag_bep && species < 2) {
                    a = ((species == 0) ? MYJ_UP(a_t_bep, kup) : MYJ_UP(a_q_bep, kup));
                    b = ((species == 0) ? MYJ_UP(b_t_bep, kup) : MYJ_UP(b_q_bep, kup));
                }
                real diag = ((species < 2) ? FMUL(rhok[m], ((1.0f - FMUL(dtdif, a)))) : rhok[m]);
                real rhs = ((species < 2) ? FMUL(rhok[m], ((var[m] + FMUL(dtdif, b)))) : FMUL(rhok[m], var[m]));
                if (m == 0) {
                    s4[m] = (FMUL(s1[m], akh[m]) + diag);
                    gm[m] = ((species == 0) ? (FMUL(-s2[m], s1[m]) + rhs) : rhs);
                } else {
                    real cf = (FMUL(-s1[m], akh[(m - 1)]) / s4[(m - 1)]);
                    s4[m] = ((FMUL(-s3[(m - 1)], cf) + FMUL(((akh[(m - 1)] + akh[m])), s1[m])) + diag);
                    gm[m] = ((species == 0) ? ((FMUL(-gm[(m - 1)], cf) + FMUL(((s2[(m - 1)] - s2[m])), s1[m])) + rhs) : (FMUL(-gm[(m - 1)], cf) + rhs));
                }
            }
            // module_bl_myjurb.F:1528-1549 flux lower boundary.
            real dtozs = ((dtdif / ((zh[(lmh - 1)] - zh[lmh]))) / vl[(lmh - 1)]);
            real rkhh = akh[(nh - 1)];
            real cf = (FMUL(-dtozs, rkhh) / s4[(nh - 1)]);
            real cmb = FMUL(s3[(nh - 1)], cf);
            real a = 0.0f; real b = 0.0f;
            if (flag_bep && species < 2) {
                a = ((species == 0) ? MYJ_UP(a_t_bep, 0) : MYJ_UP(a_q_bep, 0));
                b = ((species == 0) ? MYJ_UP(b_t_bep, 0) : MYJ_UP(b_q_bep, 0));
            }
            real diag = ((species < 2) ? FMUL(rhok[(lmh - 1)], ((1.0f - FMUL(dtdif, a)))) : rhok[(lmh - 1)]);
            real cm = ((-cmb + FMUL(rkhh, dtozs)) + diag);
            real rhs = ((species < 2) ? FMUL(rhok[(lmh - 1)], ((var[(lmh - 1)] + FMUL(dtdif, b)))) : FMUL(var[(lmh - 1)], rhok[(lmh - 1)]));
            real rs = (FMUL(-gm[(nh - 1)], cf) + rhs);
            if (species == 0) rs = (rs + FMUL(s2[(nh - 1)], dtozs));
            var[lmh-1] = (rs / cm);
            // module_bl_myjurb.F:1552-1559 back substitution.
            for (int m = (nh - 1); m >= 0; --m) {
                real rcml = (1.0f / s4[m]);
                var[m] = FMUL(((FMUL(-s3[m], var[(m + 1)]) + gm[m])), rcml);
            }
        }
    }
    for (int kup = 0; kup < nz; ++kup) {
        int m = ((nz - 1) - kup);
        // module_bl_myjurb.F:613-627
        real cwmk = cwm[m];
        real ape = (1.0f / MYJ_UP(exner_a, kup));
        real thold = MYJ_UP(th_a, kup);
        real thnew = (the[m] + FMUL(FMUL(cwmk, PBL_ELOCP), ape));
        MYJ_UP(rthblten_a, kup) = FMUL(((thnew - thold)), rdtturbl);
        real qvup = MYJ_UP(qv_a, kup);
        real qold = (qvup / ((1.0f + qvup)));
        real dqdt = FMUL(((qk[m] - qold)), rdtturbl);
        MYJ_UP(rqvblten_a, kup) = (dqdt / (FMUL(((1.0f - qk[m])), ((1.0f - qk[m])))));
        MYJ_UP(rqcblten_a, kup) = FMUL(((cwm[m] - MYJ_UP(qc_a, kup))), rdtturbl);
    }

    {
        // module_bl_myjurb.F:694-697
        for (int m = 0; m < nz - 1; ++m) {
            akm[m] = FMUL(FMUL(akm[m], ((rhok[m] + rhok[(m + 1)]))), 0.5f);
        }
        // module_bl_myjurb.F:1656-1751 VDIFV independent U/V solves.
        int nv = (lmh - 1);
        for (int m = 0; m < nv; ++m) {
            akm[m] = FMUL(akm[m], sf[m]);
            s1[m] = ((dtdif / ((zh[m] - zh[(m + 1)]))) / vl[m]);
            s3[m] = FMUL(-s1[m], akm[m]);
        }
        for (int comp = 0; comp < 2; ++comp) {
            real* var = (comp ? vk : uk);
            // module_bl_myjurb.F:1706-1722 forward sweep.
            for (int m = 0; m < nv; ++m) {
                int kup = ((nz - 1) - m);
                real a = (flag_bep ? ((comp ? MYJ_UP(a_v_bep, kup) : MYJ_UP(a_u_bep, kup))) : 0.0f);
                real b = (flag_bep ? ((comp ? MYJ_UP(b_v_bep, kup) : MYJ_UP(b_u_bep, kup))) : 0.0f);
                real diag = FMUL(rhok[m], ((1.0f - FMUL(dtdif, a))));
                real rhs = FMUL(rhok[m], ((var[m] + FMUL(dtdif, b))));
                if (m == 0) {
                    s4[m] = (FMUL(s1[m], akm[m]) + diag);
                    gm[m] = rhs;
                } else {
                    real cf = (FMUL(-s1[m], akm[(m - 1)]) / s4[(m - 1)]);
                    s4[m] = ((FMUL(-s3[(m - 1)], cf) + FMUL(((akm[(m - 1)] + akm[m])), s1[m])) + diag);
                    gm[m] = (FMUL(-gm[(m - 1)], cf) + rhs);
                }
            }
            // module_bl_myjurb.F:1726-1746 flux boundary and back solve.
            real dtozs = ((dtdif / ((zh[(lmh - 1)] - zh[lmh]))) / vl[(lmh - 1)]);
            real rkmh = akm[(nv - 1)];
            real cf = (FMUL(-dtozs, rkmh) / s4[(nv - 1)]);
            real a = (flag_bep ? ((comp ? MYJ_UP(a_v_bep, 0) : MYJ_UP(a_u_bep, 0))) : 0.0f);
            real b = (flag_bep ? ((comp ? MYJ_UP(b_v_bep, 0) : MYJ_UP(b_u_bep, 0))) : 0.0f);
            real rcmvb = (1.0f / (((FMUL(rkmh, dtozs) - FMUL(s3[(nv - 1)], cf)) + FMUL(rhok[(lmh - 1)], ((1.0f - FMUL(dtdif, a)))))));
            var[lmh-1] = FMUL(((FMUL(-gm[(nv - 1)], cf) + FMUL(rhok[(lmh - 1)], ((var[(lmh - 1)] + FMUL(dtdif, b)))))), rcmvb);
            for (int m = (nv - 1); m >= 0; --m) {
                real rcml = (1.0f / s4[m]);
                var[m] = FMUL(((FMUL(-s3[m], var[(m + 1)]) + gm[m])), rcml);
            }
        }
        // module_bl_myjurb.F:754-760 momentum tendency publication
        for (int kup = 0; kup < nz; ++kup) {
            int m = ((nz - 1) - kup);
            MYJ_UP(rublten_a, kup) = FMUL(((uk[m] - MYJ_UP(u_a, kup))), rdtturbl);
            MYJ_UP(rvblten_a, kup) = FMUL(((vk[m] - MYJ_UP(v_a, kup))), rdtturbl);
        }
    }
#undef MYJ_UP
}
