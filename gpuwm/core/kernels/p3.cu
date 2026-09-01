// P3 (Predicted Particle Properties) microphysics, WRF mp_physics=50/51.
//
// Transcription authority: WRF phys/module_mp_p3.F, sha256
// 716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6
// (7,391 lines, version_p3 = 4.5.2; byte-identical in WRF v4.6.1, v4.7.1
// and v4.8.0).  Every ":NNNN" is a line in that file.  The CPU float32
// authority gpuwm/core/p3.py carries the prose for each transcribed quirk;
// this file is the same statement order on the device and does not repeat it.
//
// LAYOUT -- decided before a line of this file was written.
//   * One THREAD per COLUMN.  k is a loop index inside the thread, exactly
//     as the authority's kloop runs inside its i-loop.  Sedimentation's
//     substep while-loop, its Courant reduction, the top/bottom searches
//     and the two column-scope logical flags are then plain sequential
//     code rather than warp reductions, so the arithmetic ORDER is the
//     authority's by construction and not by care.
//   * Every field is LEVEL-MAJOR: element (k, i) sits at k*ncol + i, which
//     is gpuwm's native (nz, ny, nx) storage read as (nk, ncol).  At a
//     fixed k a warp touches 32 consecutive floats, so every access is
//     coalesced AND no transpose is needed anywhere.  The opposite choice
//     (column-contiguous, the shape the host code used) gives every thread
//     its own cache line per load and forces a transpose per step.
//   * itab keeps the authority (densize, rimsize, isize, tabsize) shape
//     with tabsize FASTEST, so the 7..14 quantities one cell needs sit in
//     one 56-byte run.  itabcoll keeps (.., rcollsize, 2) with the two
//     collection quantities adjacent, for the same reason.
//
// PRECISION.  float32 throughout, with the authority two explicit double
// excursions kept double: the dexp saturation-relaxation factor (:3202,
// :3204, :3231) and the rain-freezing log-space products (:3025-3029).
// Contraction is OFF (-fmad=false), matching the Fortran reference arm
// -ffp-contract=off, so plain infix operators ARE the _rn intrinsics;
// tests/test_p3_cuda.py proves that equivalence instead of assuming it.
//
// TRANSCENDENTALS.  r_exp / r_log / r_pow come from the tree single glibc
// transcription in noahmp_leaves.cu, which this unit is compiled after.
// Measured 2026-08-29 on node-1 (glibc 2.43, gcc 15.2.0): all three are
// BIT-IDENTICAL to the running glibc over P3 argument ranges, 0 differing
// in 4096 samples per exponent.  alog10 is (float)log10((double)x), also
// bit-identical to glibc log10f over 600,000 samples, and deliberately NOT
// r_log10 -- that routine transcribes the pre-glibc-2.28 SunPro algorithm
// and disagrees with glibc 2.43 on 25% to 34% of P3 log10 domains, up to
// 2 ULP.  gamma() is (float)tgamma((double)x), bit-identical to glibc
// tgammaf on [3,22]; CUDA own tgammaf is 2 ULP off on 37% of that range.
//
// INTEGER POWERS.  Where the authority writes x**2, x**3 or x**6 with an
// INTEGER exponent, gfortran expands to multiplications and never calls
// powf.  Measured on gfortran 15.2.0 -O0 over 200,000 samples:
// x**2 == x*x, x**3 == (x*x)*x, x**6 == ((x*x)*(x*x))*(x*x), all exactly,
// 0 differing.  A REAL exponent (x**0.54, x**thrd, lamc**bcn where bcn is
// a REAL variable) does go through powf.  Both forms appear below and they
// are not interchangeable: glibc powf(x,6.0f) differs from the
// multiplication chain on 66.6% of arguments, up to 3 ULP.
//
// SCOPE.  nCat = 1, 2-moment ice, WRF orientation (kbot=kts, kdir=+1).
// The multi-category blocks, the 3-moment (qzi) blocks, SCPF, the
// precipitation-type partition and the visibility diagnostics are NOT
// here; mp_physics 52 and 53 are refused by name in gpuwm/config.py.
// BOTH log_predictNc paths ARE here (specified Nc and prognostic Nc),
// driven by the runtime flag the authority derives from whether the
// caller passes nc (:819-820).

// ---------------------------------------------------------------------
// p3_init constants (:177-294), emitted as exact float32 hex literals from
// gpuwm.core.p3 -- the CPU authority whose 52 comparable p3_init constants
// the p3-fortref lane measured bit-identical to the running Fortran.  They
// are NOT recomputed here, so no compile-time folding difference can move
// them.  tests/test_p3_cuda.py re-derives this block and asserts equality.
// ---------------------------------------------------------------------
#define P3_PI                  0x1.921fb60000000p+1f  // 3.1415927410125732       :178
#define P3_THRD                0x1.5555560000000p-2f  // 0.3333333432674408       :180
#define P3_SXTH                0x1.5555560000000p-3f  // 0.1666666716337204       :181
#define P3_PIOV3               0x1.0c15240000000p+0f  // 1.0471975803375244       :182
#define P3_PIOV6               0x1.0c15240000000p-1f  // 0.5235987901687622       :183
#define P3_MAX_TOTAL_NI        0x1.e848000000000p+20f // 2000000.0                :186
#define P3_NCCNST              0x1.7d78400000000p+27f // 200000000.0              :196
#define P3_CP                  0x1.f680000000000p+9f  // 1005.0                   :203
#define P3_INV_CP              0x1.04d6fe0000000p-10f // 0.0009950249223038554    :204
#define P3_G                   0x1.3a1cac0000000p+3f  // 9.815999984741211        :205
#define P3_RD                  0x1.1f26660000000p+8f  // 287.1499938964844        :206
#define P3_RV                  0x1.cd82900000000p+8f  // 461.510009765625         :207
#define P3_EP_2                0x1.3e76c80000000p-1f  // 0.621999979019165        :208
#define P3_RHOSUR              0x1.4662840000000p+0f  // 1.2749407291412354       :209
#define P3_RHOSUI              0x1.a69ad80000000p-1f  // 0.8254001140594482       :210
#define P3_F1R                 0x1.8f5c280000000p-1f  // 0.7799999713897705       :213
#define P3_F2R                 0x1.47ae140000000p-2f  // 0.3199999928474426       :214
#define P3_RHOW                0x1.f400000000000p+9f  // 1000.0                   :216
#define P3_CPW                 0x1.07a0000000000p+12f // 4218.0                   :217
#define P3_INV_RHOW            0x1.0624de0000000p-10f // 0.0010000000474974513    :218
#define P3_MU_R_CONSTANT       0.0f                   // 0.0                      :219
#define P3_INV_DRMAX           0x1.f3fffe0000000p+8f  // 499.9999694824219        :222
#define P3_RHO_RIMEMIN         0x1.9000000000000p+5f  // 50.0                     :225
#define P3_RHO_RIMEMAX         0x1.c200000000000p+9f  // 900.0                    :226
#define P3_INV_RHO_RIMEMAX     0x1.2345680000000p-10f // 0.0011111111380159855    :227
#define P3_QSMALL              0x1.6849b80000000p-47f // 9.9999998245167e-15      :230
#define P3_NSMALL              0x1.cd2b2a0000000p-54f // 1.0000000168623835e-16   :231
#define P3_BSMALL              0x1.99ed7a0000000p-57f // 1.1111111022744057e-17   :232
#define P3_BIMM                0x1.0000000000000p+1f  // 2.0                      :239
#define P3_AIMM                0x1.4ccccc0000000p-1f  // 0.6499999761581421       :240
#define P3_MI0                 0x1.0fa6980000000p-48f // 3.769911561201343e-15    :242
#define P3_ECI                 0x1.0000000000000p-1f  // 0.5                      :244
#define P3_ERI                 0x1.0000000000000p+0f  // 1.0                      :245
#define P3_BCN                 0x1.0000000000000p+1f  // 2.0                      :246
#define P3_NMLTRATIO           0x1.0000000000000p+0f  // 1.0                      :251
#define P3_CONS1               0x1.05cca60000000p+9f  // 523.5988159179688        :259
#define P3_CONS2               0x1.05cca60000000p+12f // 4188.79052734375         :260
#define P3_CONS3               0x1.c758900000000p+33f // 15278874624.0            :261
#define P3_CONS5               0x1.0c15240000000p+0f  // 1.0471975803375244       :263
#define P3_CONS6               0x1.1227dc0000000p+9f  // 548.3114013671875        :264
#define P3_CONS7               0x1.2dd58a0000000p-48f // 4.188790105925802e-15    :265
#define P3_E0                  0x1.319eb60000000p+9f  // 611.2399291992188        :257 polysvp1(273.15,0)
#define P3_XXLV                0x1.314d540000000p+21f // 2501034.5                :2325
#define P3_XXLS                0x1.5a09740000000p+21f // 2834734.5                :2326
#define P3_XLF                 0x1.45e1000000000p+18f // 333700.0                 :2327 xxls-xxlv
// --- SEAM 2: aerosol activation, the two PRESCRIBED lognormal modes as
// --- DATA (:269-294).  A prognostic aerosol species replaces these values
// --- and p3_activate_droplets below, and touches nothing else.
#define P3_MW                  0x1.26e9780000000p-6f  // 0.017999999225139618     :269
#define P3_RR                  0x1.0a32ca0000000p+3f  // 8.318699836730957        :275
#define P3_BACT                0x1.4efb0a0000000p-1f  // 0.6542590260505676       :276
#define P3_INV_RM1             0x1.312d000000000p+24f // 20000000.0               :281
#define P3_SIG1                0x1.0000000000000p+1f  // 2.0                      :282
#define P3_NANEW1              0x1.1e1a300000000p+28f // 300000000.0              :283
#define P3_INV_RM2             0x1.7799d80000000p+19f // 769230.75                :290
#define P3_SIG2                0x1.4000000000000p+1f  // 2.5                      :291
#define P3_NANEW2              0.0f                   // 0.0                      :292

// Lookup-table dimensions (:48-57).
#define P3_ISIZE     50
#define P3_DENSIZE    5
#define P3_RIMSIZE    4
#define P3_RCOLLSIZE 30
#define P3_TABSIZE   14

// Slots in the pointer arrays the launcher uploads.  Mirrored by the
// F_*/S_*/W_*/D_*/P_*/T_* tables in gpuwm/core/p3_device.py; a test pins
// the two lists against each other so a reorder cannot go unnoticed.
#define F_QC 0
#define F_NC 1
#define F_QR 2
#define F_NR 3
#define F_QI 4
#define F_QIR 5
#define F_NI 6
#define F_QIB 7
#define F_TH 8
#define F_QV 9
#define F_THOLD 10
#define F_QVOLD 11
#define F_SSAT 12
#define F_PRES 13
#define F_DZ 14

#define S_RHO 0
#define S_INVRHO 1
#define S_QVS 2
#define S_QVI 3
#define S_SUP 4
#define S_SUPI 5
#define S_RHOFACR 6
#define S_RHOFACI 7
#define S_ACN 8
#define S_T 9
#define S_TMPARR1 10
#define S_QVCLD 11

#define W_VQ 0
#define W_VN 1
#define W_FQ 2
#define W_FN 3
#define W_FQIR 4
#define W_FBIR 5

#define D_ZDBZ 0
#define D_EFFC 1
#define D_EFFI 2
#define D_VMI 3
#define D_DI 4
#define D_RHOPO 5

#define P_PRTLIQ 0
#define P_PRTSOL 1
#define P_RAINNC 2
#define P_RAINNCV 3
#define P_SR 4
#define P_SNOWNC 5
#define P_SNOWNCV 6

#define T_ITAB 0
#define T_ICOLL 1
#define T_VN 2
#define T_VM 3
#define T_REVAP 4

// Level-major addressing: element (k, column i) at k*ncol + i.
#define AT(a, kk) (a)[(size_t)(kk) * (size_t)ncol + (size_t)i]

// itab(jj, ii, i2, idx), zero-based, tabsize fastest.
#define ITAB(jj, ii, i2, idx) \
    itab[((((size_t)(jj) * P3_RIMSIZE + (size_t)(ii)) * P3_ISIZE \
          + (size_t)(i2)) * P3_TABSIZE) + (size_t)(idx)]
// itabcoll(jj, ii, i2, j2, idx), zero-based, the 2 quantities fastest.
#define ICOLL(jj, ii, i2, j2, idx) \
    icoll[(((((size_t)(jj) * P3_RIMSIZE + (size_t)(ii)) * P3_ISIZE \
           + (size_t)(i2)) * P3_RCOLLSIZE + (size_t)(j2)) * 2) + (size_t)(idx)]
// the three generated rain tables are (300, 10), mu_r column fastest.
#define RTAB(tb, a, b) (tb)[(size_t)(a) * 10 + (size_t)(b)]

// ---------------------------------------------------------------------
// Transcendental leaves.  r_exp / r_log / r_pow come from noahmp_leaves.cu,
// which is prepended to this translation unit by p3_device.py.
// ---------------------------------------------------------------------
__device__ __forceinline__ float p3_log10(float x)
{
    return (float)log10((double)x);
}

__device__ __forceinline__ float p3_gam(float x)
{
    return (float)tgamma((double)x);
}

// Fortran int(): truncation toward zero, which is what a C float->int cast
// does.  Named so the transcription reads like the authority.
__device__ __forceinline__ int p3_int(float x) { return (int)x; }

// ---------------------------------------------------------------------
// polysvp1 (:6017-6086) -- saturation vapour pressure [Pa].
// ---------------------------------------------------------------------
__device__ float p3_polysvp1(float t, int i_type)
{
    if (i_type == 1 && t < 273.15f) {
        if (t >= 195.8f) {
            float dt = t - 273.15f;
            float a0i = 6.11147274f,      a1i = 0.503160820f,
                  a2i = 0.188439774e-1f,  a3i = 0.420895665e-3f,
                  a4i = 0.615021634e-5f,  a5i = 0.602588177e-7f,
                  a6i = 0.385852041e-9f,  a7i = 0.146898966e-11f,
                  a8i = 0.252751365e-14f;
            float out = a0i + dt * (a1i + dt * (a2i + dt * (a3i + dt * (a4i
                        + dt * (a5i + dt * (a6i + dt * (a7i + a8i * dt)))))));
            return out * 100.0f;
        }
        return r_pow(10.0f, -9.09718f * (273.16f / t - 1.0f)
                            - 3.56654f * p3_log10(273.16f / t)
                            + 0.876793f * (1.0f - t / 273.16f)
                            + p3_log10(6.1071f)) * 100.0f;
    }
    if (t >= 202.0f) {
        float dt = t - 273.15f;
        float a0 = 6.11239921f,       a1 = 0.443987641f,
              a2 = 0.142986287e-1f,   a3 = 0.264847430e-3f,
              a4 = 0.302950461e-5f,   a5 = 0.206739458e-7f,
              a6 = 0.640689451e-10f,  a7 = -0.952447341e-13f,
              a8 = -0.976195544e-15f;
        float out = a0 + dt * (a1 + dt * (a2 + dt * (a3 + dt * (a4
                    + dt * (a5 + dt * (a6 + dt * (a7 + a8 * dt)))))));
        return out * 100.0f;
    }
    return r_pow(10.0f, -7.90298f * (373.16f / t - 1.0f)
                        + 5.02808f * p3_log10(373.16f / t)
                        - 1.3816e-7f * (r_pow(10.0f, 11.344f
                                        * (1.0f - t / 373.16f)) - 1.0f)
                        + 8.1328e-3f * (r_pow(10.0f, -3.49149f
                                        * (373.16f / t - 1.0f)) - 1.0f)
                        + p3_log10(1013.246f)) * 100.0f;
}

// qv_sat (:6859-6889).
__device__ __forceinline__ float p3_qv_sat(float t_atm, float p_atm, int i_wrt)
{
    float e_pres = p3_polysvp1(t_atm, i_wrt);
    return P3_EP_2 * e_pres / fmaxf(1.0e-3f, p_atm - e_pres);
}

// ---------------------------------------------------------------------
// Lookup-table access (:5205-5313) and index helpers (:6316-6410, :6589-6632).
// The Fortran 1-based index VALUES are kept; the -1 shift happens at the
// array access only, exactly as gpuwm/core/p3.py does it.
// ---------------------------------------------------------------------
__device__ float p3_access_lookup_table(const float* __restrict__ itab,
                                        int dumjj, int dumii, int dumi,
                                        int index, float dum1, float dum4,
                                        float dum5)
{
    float iproc1, gproc1, tmp1, tmp2;
    iproc1 = ITAB(dumjj-1, dumii-1, dumi-1, index-1)
        + (dum1 - (float)dumi) * (ITAB(dumjj-1, dumii-1, dumi, index-1)
                                  - ITAB(dumjj-1, dumii-1, dumi-1, index-1));
    gproc1 = ITAB(dumjj-1, dumii, dumi-1, index-1)
        + (dum1 - (float)dumi) * (ITAB(dumjj-1, dumii, dumi, index-1)
                                  - ITAB(dumjj-1, dumii, dumi-1, index-1));
    tmp1 = iproc1 + (dum4 - (float)dumii) * (gproc1 - iproc1);
    iproc1 = ITAB(dumjj, dumii-1, dumi-1, index-1)
        + (dum1 - (float)dumi) * (ITAB(dumjj, dumii-1, dumi, index-1)
                                  - ITAB(dumjj, dumii-1, dumi-1, index-1));
    gproc1 = ITAB(dumjj, dumii, dumi-1, index-1)
        + (dum1 - (float)dumi) * (ITAB(dumjj, dumii, dumi, index-1)
                                  - ITAB(dumjj, dumii, dumi-1, index-1));
    tmp2 = iproc1 + (dum4 - (float)dumii) * (gproc1 - iproc1);
    return tmp1 + (dum5 - (float)dumjj) * (tmp2 - tmp1);
}

__device__ __forceinline__ float p3_coll_pair(const float* __restrict__ icoll,
                                              int jj, int ii, int dumi,
                                              int dumj, int index,
                                              float dum1, float dum3)
{
    float dproc1 = ICOLL(jj, ii, dumi-1, dumj-1, index-1)
        + (dum1 - (float)dumi) * (ICOLL(jj, ii, dumi, dumj-1, index-1)
                                  - ICOLL(jj, ii, dumi-1, dumj-1, index-1));
    float dproc2 = ICOLL(jj, ii, dumi-1, dumj, index-1)
        + (dum1 - (float)dumi) * (ICOLL(jj, ii, dumi, dumj, index-1)
                                  - ICOLL(jj, ii, dumi-1, dumj, index-1));
    return dproc1 + (dum3 - (float)dumj) * (dproc2 - dproc1);
}

__device__ float p3_access_lookup_table_coll(const float* __restrict__ icoll,
                                             int dumjj, int dumii, int dumj,
                                             int dumi, int index, float dum1,
                                             float dum3, float dum4, float dum5)
{
    float iproc1 = p3_coll_pair(icoll, dumjj-1, dumii-1, dumi, dumj, index,
                                dum1, dum3);
    float gproc1 = p3_coll_pair(icoll, dumjj-1, dumii, dumi, dumj, index,
                                dum1, dum3);
    float tmp1 = iproc1 + (dum4 - (float)dumii) * (gproc1 - iproc1);
    iproc1 = p3_coll_pair(icoll, dumjj, dumii-1, dumi, dumj, index,
                          dum1, dum3);
    gproc1 = p3_coll_pair(icoll, dumjj, dumii, dumi, dumj, index, dum1, dum3);
    float tmp2 = iproc1 + (dum4 - (float)dumii) * (gproc1 - iproc1);
    return tmp1 + (dum5 - (float)dumjj) * (tmp2 - tmp1);
}

// find_lookupTable_indices_1a (:6316-6370).  Note the authority takes the
// INTEGER before clamping the real, then clamps both: the interpolation
// weight saturates to exactly 0 below the axis and 1 above it, which is
// constant extrapolation at both ends, never linear.
__device__ void p3_find_lt_1a(float qitot, float nitot, float qirim, float rhop,
                              int* dumi, int* dumjj, int* dumii,
                              float* dum1, float* dum4, float* dum5)
{
    float d1 = (p3_log10(qitot / nitot) + 18.0f) * 3.444606f - 10.0f;
    int di = p3_int(d1);
    d1 = fminf(d1, (float)P3_ISIZE);
    d1 = fmaxf(d1, 1.0f);
    di = max(1, di);
    di = min(P3_ISIZE - 1, di);

    float d4 = (qirim / qitot) * 3.0f + 1.0f;
    int dii = p3_int(d4);
    d4 = fminf(d4, (float)P3_RIMSIZE);
    d4 = fmaxf(d4, 1.0f);
    dii = max(1, dii);
    dii = min(P3_RIMSIZE - 1, dii);

    float d5;
    if (rhop <= 650.0f) d5 = (rhop - 50.0f) * 0.005f + 1.0f;
    else                d5 = (rhop - 650.0f) * 0.004f + 4.0f;
    int djj = p3_int(d5);
    d5 = fminf(d5, (float)P3_DENSIZE);
    d5 = fmaxf(d5, 1.0f);
    djj = max(1, djj);
    djj = min(P3_DENSIZE - 1, djj);

    *dumi = di; *dumjj = djj; *dumii = dii;
    *dum1 = d1; *dum4 = d4; *dum5 = d5;
}

// find_lookupTable_indices_1b (:6374-6410).
__device__ void p3_find_lt_1b(float qr, float nr, int* dumj, float* dum3)
{
    if (qr >= P3_QSMALL && nr > 0.0f) {
        float dumlr = r_pow(qr / (P3_PI * P3_RHOW * nr), P3_THRD);
        float d3 = (p3_log10(1.0f * dumlr) + 5.0f) * 10.70415f;
        int dj = p3_int(d3);
        d3 = fminf(d3, (float)P3_RCOLLSIZE);
        d3 = fmaxf(d3, 1.0f);
        dj = max(1, dj);
        dj = min(P3_RCOLLSIZE - 1, dj);
        *dumj = dj; *dum3 = d3;
    } else {
        *dumj = 1; *dum3 = 1.0f;
    }
}

// find_lookupTable_indices_3 (:6589-6632).
__device__ void p3_find_lt_3(float mu_r, float lamr, int* dumii, int* dumjj,
                             float* rdumii, float* rdumjj)
{
    float dum1 = (mu_r + 1.0f) / lamr;
    float rii, inv_dum3;
    int dii;
    if (dum1 <= 195.0e-6f) {
        inv_dum3 = 0.1f;
        rii = (dum1 * 1.0e6f + 5.0f) * inv_dum3;
        rii = fmaxf(rii, 1.0f);
        rii = fminf(rii, 20.0f);
        dii = p3_int(rii);
        dii = max(dii, 1);
        dii = min(dii, 20);
    } else {
        inv_dum3 = P3_THRD * 0.1f;
        rii = (dum1 * 1.0e6f - 195.0f) * inv_dum3 + 20.0f;
        rii = fmaxf(rii, 20.0f);
        rii = fminf(rii, 300.0f);
        dii = p3_int(rii);
        dii = max(dii, 20);
        dii = min(dii, 299);
    }
    (void)inv_dum3;
    float rjj = mu_r + 1.0f;
    rjj = fmaxf(rjj, 1.0f);
    rjj = fminf(rjj, 10.0f);
    int djj = p3_int(rjj);
    djj = max(djj, 1);
    djj = min(djj, 9);
    *dumii = dii; *dumjj = djj; *rdumii = rii; *rdumjj = rjj;
}

// Bilinear read of one generated (300, 10) rain table -- the shape the
// authority repeats at :3117-3122, :4150-4160 and :4166-4176.
__device__ __forceinline__ float p3_rain_table(const float* __restrict__ tb,
                                               int dumii, int dumjj,
                                               float rdumii, float rdumjj)
{
    float dum1 = RTAB(tb, dumii-1, dumjj-1)
        + (rdumii - (float)dumii) * (RTAB(tb, dumii, dumjj-1)
                                     - RTAB(tb, dumii-1, dumjj-1));
    float dum2 = RTAB(tb, dumii-1, dumjj)
        + (rdumii - (float)dumii) * (RTAB(tb, dumii, dumjj)
                                     - RTAB(tb, dumii-1, dumjj));
    return dum1 + (rdumjj - (float)dumjj) * (dum2 - dum1);
}

// get_cloud_dsd2 (:6635-6701).  nc_grd is INOUT in the authority.
__device__ void p3_get_cloud_dsd2(float* nc_grd, float qc_grd, float rho,
                                  float iscf, float* mu_c_o, float* lamc_o,
                                  float* cdist_o, float* cdist1_o)
{
    float qc = qc_grd * iscf;
    if (qc >= P3_QSMALL) {
        float nc = *nc_grd * iscf;
        nc = fmaxf(nc, P3_NSMALL);
        float mu_c = 0.0005714f * (nc * 1.0e-6f * rho) + 0.2714f;
        mu_c = 1.0f / (mu_c * mu_c) - 1.0f;            // :6662, integer **2
        mu_c = fmaxf(mu_c, 2.0f);
        mu_c = fminf(mu_c, 15.0f);
        // iparam = 3 (:193), so nu is 0 and dnu is unread (:6666-6670).
        float lamc = r_pow(P3_CONS1 * nc * (mu_c + 3.0f) * (mu_c + 2.0f)
                           * (mu_c + 1.0f) / qc, P3_THRD);
        float lammin = (mu_c + 1.0f) * 2.5e4f;
        float lammax = (mu_c + 1.0f) * 1.0e6f;
        if (lamc < lammin) {
            lamc = lammin;
            nc = 6.0f * ((lamc * lamc) * lamc) * qc
                 / (P3_PI * P3_RHOW * (mu_c + 3.0f) * (mu_c + 2.0f)
                    * (mu_c + 1.0f));                  // :6681, integer **3
        } else if (lamc > lammax) {
            lamc = lammax;
            nc = 6.0f * ((lamc * lamc) * lamc) * qc
                 / (P3_PI * P3_RHOW * (mu_c + 3.0f) * (mu_c + 2.0f)
                    * (mu_c + 1.0f));                  // :6684, integer **3
        }
        *cdist_o = nc * (mu_c + 1.0f) / lamc;
        *cdist1_o = nc / p3_gam(mu_c + 1.0f);
        *nc_grd = nc / iscf;
        *mu_c_o = mu_c; *lamc_o = lamc;
    } else {
        *mu_c_o = 0.0f; *lamc_o = 0.0f; *cdist_o = 0.0f; *cdist1_o = 0.0f;
    }
}

// get_rain_dsd2 (:6705-6780).  nr_grd is INOUT in the authority.
__device__ void p3_get_rain_dsd2(float* nr_grd, float qr_grd, float ispf,
                                 float* mu_r_o, float* lamr_o,
                                 float* cdistr_o, float* logn0r_o)
{
    float qr = qr_grd * ispf;
    if (qr >= P3_QSMALL) {
        float nr = *nr_grd * ispf;
        nr = fmaxf(nr, P3_NSMALL);
        float mu_r = P3_MU_R_CONSTANT;
        float lamr = r_pow(P3_CONS1 * nr * (mu_r + 3.0f) * (mu_r + 2.0f)
                           * (mu_r + 1.0f) / qr, P3_THRD);
        float lammax = (mu_r + 1.0f) * 1.0e5f;
        float lammin = (mu_r + 1.0f) * P3_INV_DRMAX;
        if (lamr < lammin) {
            lamr = lammin;
            nr = r_exp(3.0f * r_log(lamr) + r_log(qr)
                       + r_log(p3_gam(mu_r + 1.0f))
                       - r_log(p3_gam(mu_r + 4.0f))) / P3_CONS1;
        } else if (lamr > lammax) {
            lamr = lammax;
            nr = r_exp(3.0f * r_log(lamr) + r_log(qr)
                       + r_log(p3_gam(mu_r + 1.0f))
                       - r_log(p3_gam(mu_r + 4.0f))) / P3_CONS1;
        }
        *logn0r_o = p3_log10(nr) + (mu_r + 1.0f) * p3_log10(lamr)
                    - p3_log10(p3_gam(mu_r + 1.0f));
        *cdistr_o = nr / p3_gam(mu_r + 1.0f);
        *nr_grd = nr / ispf;
        *mu_r_o = mu_r; *lamr_o = lamr;
    } else {
        *mu_r_o = 0.0f; *lamr_o = 0.0f; *cdistr_o = 0.0f; *logn0r_o = 0.0f;
    }
}

// calc_bulkRhoRime (:6784-6830).
__device__ void p3_calc_bulk_rho_rime(float qi_tot, float* qi_rim,
                                      float* bi_rim, float* rho_rime_o)
{
    float qir = *qi_rim, bir = *bi_rim, rho_rime;
    if (bir >= 1.0e-15f) {
        rho_rime = qir / bir;
        if (rho_rime < P3_RHO_RIMEMIN) {
            rho_rime = P3_RHO_RIMEMIN;
            bir = qir / rho_rime;
        } else if (rho_rime > P3_RHO_RIMEMAX) {
            rho_rime = P3_RHO_RIMEMAX;
            bir = qir / rho_rime;
        }
    } else {
        qir = 0.0f; bir = 0.0f; rho_rime = 0.0f;
    }
    if (qir > qi_tot && rho_rime > 0.0f) {
        qir = qi_tot;
        bir = qir / rho_rime;
    }
    if (qir < P3_QSMALL) { qir = 0.0f; bir = 0.0f; }
    *qi_rim = qir; *bi_rim = bir; *rho_rime_o = rho_rime;
}

// impose_max_total_Ni, nCat = 1 form (:6833-6855).
__device__ __forceinline__ float p3_impose_max_ni(float nitot, float inv_rho)
{
    if (nitot >= 1.0e-20f) {
        float dum = P3_MAX_TOTAL_NI * inv_rho / nitot;
        nitot = nitot * fminf(dum, 1.0f);
    }
    return nitot;
}

// ---------------------------------------------------------------------
// SEAM 2 -- droplet activation as a SUBSTITUTABLE STEP.
//
// The whole of the authority activation section (:3303-3339), BOTH
// branches, lifted into one callable device function whose aerosol inputs
// are the P3_NANEW*/P3_INV_RM*/P3_SIG* DATA above.  A prognostic aerosol
// species replaces this function and those constants and touches nothing
// else in the port.  It stays a call rather than arithmetic inlined into a
// fused kernel for exactly that reason; the measured cost is in the
// receipt, not assumed to be zero.
// ---------------------------------------------------------------------
__device__ void p3_activate_droplets(int log_predictNc, float sup_cld, int it,
                                     float t, float pres, float qc, float nc,
                                     float qv_cld, float inv_rho, float iscf,
                                     float scf, float odt,
                                     float* qcnuc, float* ncnuc)
{
    if (!log_predictNc && sup_cld > 1.0e-6f && it > 1) {
        float dum = P3_NCCNST * inv_rho * P3_CONS7 - qc;
        dum = fmaxf(0.0f, dum * iscf);
        float dumqvs = p3_qv_sat(t, pres, 0);
        float dqsdT = P3_XXLV * dumqvs / (P3_RV * t * t);
        float ab = 1.0f + dqsdT * P3_XXLV * P3_INV_CP;
        dum = fmaxf(0.0f, fminf(dum, (qv_cld - dumqvs) / ab));
        *qcnuc = dum * odt * scf;
    }
    if (log_predictNc) {
        if (sup_cld > 1.0e-6f) {
            float dum1 = 1.0f / r_pow(P3_BACT, 0.5f);
            float sigvl = 0.0761f - 1.55e-4f * (t - 273.15f);
            float aact = 2.0f * P3_MW / (P3_RHOW * P3_RR * t) * sigvl;
            float sm1 = 2.0f * dum1 * r_pow(aact * P3_THRD * P3_INV_RM1, 1.5f);
            float sm2 = 2.0f * dum1 * r_pow(aact * P3_THRD * P3_INV_RM2, 1.5f);
            float uu1 = 2.0f * r_log(sm1 / sup_cld) / (4.242f * r_log(P3_SIG1));
            float uu2 = 2.0f * r_log(sm2 / sup_cld) / (4.242f * r_log(P3_SIG2));
            // derf is the DOUBLE erf in the authority (:3324-3325).
            dum1 = P3_NANEW1 * 0.5f * (1.0f - (float)erf((double)uu1));
            float dum2 = P3_NANEW2 * 0.5f * (1.0f - (float)erf((double)uu2));
            dum2 = fminf(P3_NANEW1 + P3_NANEW2, dum1 + dum2);
            dum2 = (dum2 - nc * iscf) * odt * scf;
            dum2 = fmaxf(0.0f, dum2);
            *ncnuc = dum2;
            if (it <= 1) *qcnuc = 0.0f;
            else         *qcnuc = *ncnuc * P3_CONS7;
        }
    }
}

// =====================================================================
// The column steps.  Each is the authority's corresponding loop, run by
// ONE thread for ONE column.  The __global__ wrappers below compose them
// two ways -- one step per launch (the UNFUSED reference arm) and several
// per launch (the FUSED arm) -- from these same function bodies, so a
// difference between the arms can only come from the compiler, which is
// exactly what the byte gate in tests/test_p3_cuda.py is there to catch.
// =====================================================================

#define P3_ARGS \
    float* const* __restrict__ F, float* const* __restrict__ S, \
    float* const* __restrict__ W, float* const* __restrict__ D, \
    float* const* __restrict__ P, const float* const* __restrict__ T, \
    float* __restrict__ FLG, int ncol, int nk, float dt, int it, \
    int log_predictNc, float clbfact_dep, float clbfact_sub, int i
#define P3_PASS \
    F, S, W, D, P, T, FLG, ncol, nk, dt, it, log_predictNc, \
    clbfact_dep, clbfact_sub, i

// ---------------------------------------------------------------------
// Slab diagnostics at entry (:2293-2297) plus the diagnostic pre-sets
// (:2276-2288) and the surface accumulators.  Level-local, so this is the
// authority's whole-array section written as a k-loop.
// ---------------------------------------------------------------------
__device__ void p3_step_prep_col(P3_ARGS)
{
    float* __restrict__ th = F[F_TH];
    float* __restrict__ qv = F[F_QV];
    const float* __restrict__ pres = F[F_PRES];
    for (int k = 0; k < nk; ++k) {
        float tm = r_pow(AT(pres, k) * 1.0e-5f, P3_RD * P3_INV_CP);  // :2293
        AT(S[S_TMPARR1], k) = tm;
        AT(S[S_T], k) = AT(th, k) * tm;                              // :2295
        AT(qv, k) = fmaxf(AT(qv, k), 0.0f);                          // :2297
        AT(D[D_ZDBZ], k) = -99.0f;                                   // :2278
        AT(D[D_EFFC], k) = 10.0e-6f;                                 // :2279
        AT(D[D_EFFI], k) = 25.0e-6f;                                 // :2280
        AT(D[D_VMI], k) = 0.0f;                                      // :2282
        AT(D[D_DI], k) = 0.0f;                                       // :2283
        AT(D[D_RHOPO], k) = 0.0f;                                    // :2284
    }
    P[P_PRTLIQ][i] = 0.0f;                                           // :2270
    P[P_PRTSOL][i] = 0.0f;                                           // :2271
    // The two column-scope logical flags.  Float 0/1 rather than int so
    // the allocation gate prices them alongside every other companion
    // array; 0.0f and 1.0f are exact and compare exactly.
    FLG[i] = 0.0f;                                                   // nucleation
    FLG[ncol + i] = 0.0f;                                            // hydrometeors
}

// ---------------------------------------------------------------------
// k_loop_1 (:2320-2411): atmospheric variables, the two column-scope
// logical flags, and mass clipping in dry air.
// ---------------------------------------------------------------------
__device__ void p3_step_kloop1_col(P3_ARGS)
{
    float* __restrict__ qc = F[F_QC];
    float* __restrict__ nc = F[F_NC];
    float* __restrict__ qr = F[F_QR];
    float* __restrict__ nr = F[F_NR];
    float* __restrict__ qi = F[F_QI];
    float* __restrict__ qir = F[F_QIR];
    float* __restrict__ ni = F[F_NI];
    float* __restrict__ qib = F[F_QIB];
    float* __restrict__ th = F[F_TH];
    float* __restrict__ qv = F[F_QV];
    const float* __restrict__ th_old = F[F_THOLD];
    const float* __restrict__ qv_old = F[F_QVOLD];
    float* __restrict__ ssat = F[F_SSAT];      // SEAM 1: live, threaded
    const float* __restrict__ pres = F[F_PRES];

    int nucleation_possible = 0;
    int hydrometeors_present = 0;

    for (int k = 0; k < nk; ++k) {
        float t = AT(S[S_T], k);
        float tm = AT(S[S_TMPARR1], k);
        float invexn = 1.0f / tm;                                    // :2294
        float rho = AT(pres, k) / (P3_RD * t);                       // :2322
        float inv_rho = 1.0f / rho;                                  // :2323
        AT(S[S_RHO], k) = rho;
        AT(S[S_INVRHO], k) = inv_rho;
        // The first-call guard (:2329-2330): th_old/qv_old are the zero the
        // allocation left, and max(t_old,1.) is what keeps polysvp1 finite.
        float t_old = AT(th_old, k) * tm;
        float qvs = p3_qv_sat(fmaxf(t_old, 1.0f), AT(pres, k), 0);
        float qvi = p3_qv_sat(fmaxf(t_old, 1.0f), AT(pres, k), 1);
        // Cold-start qvs/qvi floor, DEFAULT-ON in all three arms
        // (2026-08-31): polysvp1(1 K) underflows, so the stock step-1
        // sup/supi diagnoses below were 0/0 NaN; the floor pins them at
        // exactly -1 and is inert from step 2 on.  The full rationale,
        // the declared step-1 clip delta and the flipped pin test live
        // at the CPU authority's floor site (gpuwm/core/p3.py); the
        // three arms move together or not at all.
        qvs = fmaxf(qvs, 1.0e-20f);
        qvi = fmaxf(qvi, 1.0e-20f);
        AT(S[S_QVS], k) = qvs;
        AT(S[S_QVI], k) = qvi;
        // SEAM 1 -- log_predictSsat is .false. (:2252) so ssat is DIAGNOSED
        // here, but it stays a real threaded array with both branches
        // present.  The authority's own note is that the prediction code is
        // absent; this is the documented attach point and it is not
        // optimised away just because it_le_1 makes the second branch
        // unreachable today.
        float sup, supi;
        if (1 /* .not. log_predictSsat .or. it <= 1 */) {
            AT(ssat, k) = AT(qv_old, k) - qvs;                       // :2334
            sup = AT(qv_old, k) / qvs - 1.0f;                        // :2335
            supi = AT(qv_old, k) / qvi - 1.0f;                       // :2336
        } else {
            sup = AT(ssat, k) / qvs;                                 // :2339
            supi = (AT(ssat, k) + qvs - qvi) / qvi;                  // :2340
        }
        AT(S[S_SUP], k) = sup;
        AT(S[S_SUPI], k) = supi;
        AT(S[S_RHOFACR], k) = r_pow(P3_RHOSUR * inv_rho, 0.54f);     // :2343
        AT(S[S_RHOFACI], k) = r_pow(P3_RHOSUI * inv_rho, 0.54f);     // :2344
        float dum = 1.496e-6f * r_pow(t, 1.5f) / (t + 120.0f);       // :2345
        AT(S[S_ACN], k) = P3_G * P3_RHOW / (18.0f * dum);            // :2346
        if (!log_predictNc) AT(nc, k) = P3_NCCNST * inv_rho;         // :2349-2351

        // Fortran precedence preserved: A .or. (B .and. .not.SCPF_on)
        if ((t < 273.15f && supi >= -0.05f)
            || (t >= 273.15f && sup >= -0.05f)) {
            nucleation_possible = 1;                                 // :2358-2360
        }

        if (AT(qc, k) < P3_QSMALL
            || (AT(qc, k) < 1.0e-8f && sup < -0.1f)) {               // :2365
            AT(qv, k) = AT(qv, k) + AT(qc, k);
            AT(th, k) = AT(th, k) - invexn * AT(qc, k) * P3_XXLV * P3_INV_CP;
            AT(qc, k) = 0.0f;
            AT(nc, k) = 0.0f;
        } else {
            hydrometeors_present = 1;
        }

        if (AT(qr, k) < P3_QSMALL
            || (AT(qr, k) < 1.0e-8f && sup < -0.1f)) {               // :2375
            AT(qv, k) = AT(qv, k) + AT(qr, k);
            AT(th, k) = AT(th, k) - invexn * AT(qr, k) * P3_XXLV * P3_INV_CP;
            AT(qr, k) = 0.0f;
            AT(nr, k) = 0.0f;
        } else {
            hydrometeors_present = 1;
        }

        if (AT(qi, k) < P3_QSMALL
            || (AT(qi, k) < 1.0e-8f && supi < -0.1f)) {              // :2385
            AT(qv, k) = AT(qv, k) + AT(qi, k);
            AT(th, k) = AT(th, k) - invexn * AT(qi, k) * P3_XXLS * P3_INV_CP;
            AT(qi, k) = 0.0f;
            AT(ni, k) = 0.0f;
            AT(qir, k) = 0.0f;
            AT(qib, k) = 0.0f;
        } else {
            hydrometeors_present = 1;
        }

        if (AT(qi, k) >= P3_QSMALL && AT(qi, k) < 1.0e-8f
            && t >= 273.15f) {                                       // :2399
            AT(qr, k) = AT(qr, k) + AT(qi, k);
            AT(nr, k) = AT(nr, k) + AT(ni, k);
            AT(th, k) = AT(th, k) - invexn * AT(qi, k) * P3_XLF * P3_INV_CP;
            AT(qi, k) = 0.0f;
            AT(ni, k) = 0.0f;
            AT(qir, k) = 0.0f;
            AT(qib, k) = 0.0f;
        }
    }
    // first compute_SCPF (:2432-2434) with SCPF off (:1889-1899): the qv
    // snapshot is the only product the WRF call shape reads.
    for (int k = 0; k < nk; ++k) AT(S[S_QVCLD], k) = AT(F[F_QV], k);
    FLG[i] = nucleation_possible ? 1.0f : 0.0f;
    FLG[ncol + i] = hydrometeors_present ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------
// k_loop_main (:2449-3927): process rates, conservation, prognostic
// update, clipping.  The single largest step; run by one thread over the
// whole column so the statement order is the authority's.
// ---------------------------------------------------------------------
__device__ void p3_step_kloopmain_col(P3_ARGS)
{
    // goto 333 (:2439) -- nothing to do in this column.
    if (FLG[i] == 0.0f && FLG[ncol + i] == 0.0f) return;
    FLG[ncol + i] = 0.0f;                                            // :2441

    float* __restrict__ qc = F[F_QC];
    float* __restrict__ nc = F[F_NC];
    float* __restrict__ qr = F[F_QR];
    float* __restrict__ nr = F[F_NR];
    float* __restrict__ qi = F[F_QI];
    float* __restrict__ qir = F[F_QIR];
    float* __restrict__ ni = F[F_NI];
    float* __restrict__ qib = F[F_QIB];
    float* __restrict__ th = F[F_TH];
    float* __restrict__ qv = F[F_QV];
    const float* __restrict__ qv_old = F[F_QVOLD];
    const float* __restrict__ th_old = F[F_THOLD];
    float* __restrict__ ssat = F[F_SSAT];
    const float* __restrict__ pres = F[F_PRES];
    const float* __restrict__ itab = T[T_ITAB];
    const float* __restrict__ icoll = T[T_ICOLL];
    const float* __restrict__ revap = T[T_REVAP];

    const float odt = 1.0f / dt;                                     // :2261
    // SCPF off-branch (:1889-1899) -- constant 1 / 0, kept named so the
    // Sundqvist branch has somewhere to attach.
    const float iscf = 1.0f, scf = 1.0f, spf = 1.0f, ispf = 1.0f;
    const float spf_clr = 0.0f;
    // :2290, assigned once outside the loops.  It is only ever READ when
    // qccol > 0, which implies the if/else below just wrote it, so the
    // authority's carry across columns is unobservable and this per-column
    // initialisation is equivalent.
    float rhorime_c = 400.0f;
    int hydrometeors_present = 0;

    for (int k = 0; k < nk; ++k) {
        const float t = AT(S[S_T], k);
        const float rho = AT(S[S_RHO], k);
        const float inv_rho = AT(S[S_INVRHO], k);
        const float qvs = AT(S[S_QVS], k);
        const float qvi = AT(S[S_QVI], k);
        const float sup = AT(S[S_SUP], k);
        const float supi = AT(S[S_SUPI], k);
        const float rhofaci = AT(S[S_RHOFACI], k);
        const float acn = AT(S[S_ACN], k);
        const float qv_cld = AT(S[S_QVCLD], k);
        const float invexn = 1.0f / AT(S[S_TMPARR1], k);
        const float t_old = AT(th_old, k) * AT(S[S_TMPARR1], k);

        // goto 555 (:2462-2464): dry, subsaturated, no hydrometeors.
        int log_exitlevel = !(AT(qc, k) >= P3_QSMALL || AT(qr, k) >= P3_QSMALL
                              || AT(qi, k) >= P3_QSMALL);
        if (log_exitlevel && ((t < 273.15f && supi < -0.05f)
                              || (t >= 273.15f && sup < -0.05f))) continue;

        // process rates (:2467-2485), nCat = 1 so all are scalars
        float qcacc = 0.0f, qrevp = 0.0f, qccon = 0.0f;
        float qcaut = 0.0f, qcevp = 0.0f, qrcon = 0.0f;
        float ncacc = 0.0f, ncnuc = 0.0f, ncslf = 0.0f;
        float ncautc = 0.0f, qcnuc = 0.0f, nrslf = 0.0f;
        float nrevp = 0.0f, ncautr = 0.0f;
        float qchetc = 0.0f, qisub = 0.0f, nrshdr = 0.0f;
        float qcheti = 0.0f, qrcol = 0.0f, qcshd = 0.0f;
        float qrhetc = 0.0f, qimlt = 0.0f, qccol = 0.0f;
        float qrheti = 0.0f, qinuc = 0.0f, nimlt = 0.0f;
        float nchetc = 0.0f, nccol = 0.0f, ncshdc = 0.0f;
        float ncheti = 0.0f, nrcol = 0.0f, nislf = 0.0f;
        float nrhetc = 0.0f, ninuc = 0.0f, qidep = 0.0f;
        float nrheti = 0.0f, nisub = 0.0f, qwgrth = 0.0f;
        float qrmul = 0.0f, nimul = 0.0f;
        int log_wetgrowth = 0;

        // goto 444 (:2523-2528)
        log_exitlevel = !(AT(qc, k) >= P3_QSMALL || AT(qr, k) >= P3_QSMALL
                          || AT(qi, k) >= P3_QSMALL);

        float dqsdT = 0.0f, epsi = 0.0f, epsi_tot = 0.0f;
        float f1pr04 = 0.0f, f1pr05 = 0.0f, f1pr09 = 0.0f;
        float f1pr10 = 0.0f, f1pr14 = 0.0f;
        float f1pr02 = 0.0f, f1pr03 = 0.0f;
        float f1pr07 = -99.0f, f1pr08 = -99.0f;
        float mu_c = 0.0f, lamc = 0.0f, cdist = 0.0f, cdist1 = 0.0f;
        float mu_r = 0.0f, lamr = 0.0f, cdistr = 0.0f, logn0r = 0.0f;
        float abi = 1.0f, ab = 1.0f, oabi = 1.0f, oxx = 0.0f, aaa = 0.0f;
        float dum, dum1, dum2, tmp1, tmp2;

        if (!log_exitlevel) {
            float mu = 1.496e-6f * r_pow(t, 1.5f) / (t + 120.0f);    // :2531
            float dv = 8.794e-5f * r_pow(t, 1.81f) / AT(pres, k);    // :2532
            float sc = mu / (rho * dv);                              // :2533
            dum = 1.0f / (P3_RV * (t * t));                          // :2534 int **2
            dqsdT = P3_XXLV * qvs * dum;                             // :2535
            float dqsidT = P3_XXLS * qvi * dum;                      // :2536
            ab = 1.0f + dqsdT * P3_XXLV * P3_INV_CP;                 // :2537
            abi = 1.0f + dqsidT * P3_XXLS * P3_INV_CP;               // :2538
            float kap = 1.414e3f * mu;                               // :2539
            float eii;
            if (t < 253.15f)      eii = 0.001f;                      // :2547
            else if (t < 273.15f) eii = 0.001f + (t - 253.15f)
                                         * (0.3f - 0.001f) / 20.0f;
            else                  eii = 0.3f;

            {   float ncg = AT(nc, k);
                p3_get_cloud_dsd2(&ncg, AT(qc, k), rho, iscf,
                                  &mu_c, &lamc, &cdist, &cdist1);
                AT(nc, k) = ncg; }
            {   float nrg = AT(nr, k);
                p3_get_rain_dsd2(&nrg, AT(qr, k), ispf,
                                 &mu_r, &lamr, &cdistr, &logn0r);
                AT(nr, k) = nrg; }

            epsi_tot = 0.0f;                                         // :2564
            AT(ni, k) = p3_impose_max_ni(AT(ni, k), inv_rho);        // :2566

            float eii_fact = 1.0f;
            if (AT(qi, k) >= P3_QSMALL) {                            // :2570
                AT(ni, k) = fmaxf(AT(ni, k), P3_NSMALL);
                AT(nr, k) = fmaxf(AT(nr, k), P3_NSMALL);
                float rhop;
                {   float qq = AT(qir, k), bb = AT(qib, k);
                    p3_calc_bulk_rho_rime(AT(qi, k), &qq, &bb, &rhop);
                    AT(qir, k) = qq; AT(qib, k) = bb; }
                int dumi, dumjj, dumii, dumj;
                float d1, d4, d5, d3;
                p3_find_lt_1a(AT(qi, k), AT(ni, k), AT(qir, k), rhop,
                              &dumi, &dumjj, &dumii, &d1, &d4, &d5);
                p3_find_lt_1b(AT(qr, k), AT(nr, k), &dumj, &d3);
                f1pr02 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 2,
                                                d1, d4, d5);
                f1pr03 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 3,
                                                d1, d4, d5);
                f1pr04 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 4,
                                                d1, d4, d5);
                f1pr05 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 5,
                                                d1, d4, d5);
                f1pr09 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 7,
                                                d1, d4, d5);
                f1pr10 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 8,
                                                d1, d4, d5);
                f1pr14 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 10,
                                                d1, d4, d5);
                if (AT(qr, k) >= P3_QSMALL) {
                    f1pr07 = p3_access_lookup_table_coll(icoll, dumjj, dumii,
                                                         dumj, dumi, 1,
                                                         d1, d3, d4, d5);
                    f1pr08 = p3_access_lookup_table_coll(icoll, dumjj, dumii,
                                                         dumj, dumi, 2,
                                                         d1, d3, d4, d5);
                } else {
                    f1pr07 = -99.0f;
                    f1pr08 = -99.0f;
                }
                AT(ni, k) = fminf(AT(ni, k), f1pr09 * AT(qi, k));     // :2649
                AT(ni, k) = fmaxf(AT(ni, k), f1pr10 * AT(qi, k));     // :2650
                if (AT(qir, k) > 0.0f) {                              // :2665
                    tmp1 = AT(qir, k) / AT(qi, k);
                    if (tmp1 < 0.6f)      eii_fact = 1.0f;
                    else if (tmp1 < 0.9f) eii_fact = 1.0f
                                                     - (tmp1 - 0.6f) / 0.3f;
                    else                  eii_fact = 0.0f;
                } else {
                    eii_fact = 1.0f;
                }
            }

            // collection of droplets (:2697-2710)
            if (AT(qi, k) >= P3_QSMALL && AT(qc, k) >= P3_QSMALL
                && t <= 273.15f) {
                qccol = rhofaci * f1pr04 * AT(qc, k) * P3_ECI * rho
                        * AT(ni, k) * iscf;
                nccol = rhofaci * f1pr04 * AT(nc, k) * P3_ECI * rho
                        * AT(ni, k) * iscf;
            }
            if (AT(qi, k) >= P3_QSMALL && AT(qc, k) >= P3_QSMALL
                && t > 273.15f) {
                qcshd = rhofaci * f1pr04 * AT(qc, k) * P3_ECI * rho
                        * AT(ni, k) * iscf;
                nccol = rhofaci * f1pr04 * AT(nc, k) * P3_ECI * rho
                        * AT(ni, k) * iscf;
                ncshdc = qcshd * 1.923e6f;
            }

            // collection of rain (:2725-2748)
            if (AT(qi, k) >= P3_QSMALL && AT(qr, k) >= P3_QSMALL
                && t <= 273.15f) {
                qrcol = r_pow(10.0f, f1pr08 + logn0r) * rho * rhofaci
                        * P3_ERI * AT(ni, k) * iscf * (spf - spf_clr);
                nrcol = r_pow(10.0f, f1pr07 + logn0r) * rho * rhofaci
                        * P3_ERI * AT(ni, k) * iscf * (spf - spf_clr);
            }
            if (AT(qi, k) >= P3_QSMALL && AT(qr, k) >= P3_QSMALL
                && t > 273.15f) {
                nrcol = r_pow(10.0f, f1pr07 + logn0r) * rho * rhofaci
                        * P3_ERI * AT(ni, k) * iscf * (spf - spf_clr);
            }

            // self-collection of ice (:2840-2842)
            if (AT(qi, k) >= P3_QSMALL) {
                nislf = f1pr03 * rho * eii * eii_fact * rhofaci
                        * AT(ni, k) * AT(ni, k) * iscf;
            }

            // melting (:2851-2865)
            if (AT(qi, k) >= P3_QSMALL && t > 273.15f) {
                float qsat0 = 0.622f * P3_E0 / (AT(pres, k) - P3_E0);
                dum = 0.0f;
                qimlt = ((f1pr05 + f1pr14 * r_pow(sc, P3_THRD)
                          * r_pow(rhofaci * rho / mu, 0.5f))
                         * ((t - 273.15f) * kap
                            - rho * P3_XXLV * dv * (qsat0 - qv_cld))
                         * 2.0f * P3_PI / P3_XLF + dum) * AT(ni, k);
                qimlt = fmaxf(qimlt, 0.0f);
                nimlt = qimlt * (AT(ni, k) / AT(qi, k));
            }

            // wet growth (:2873-2894)
            if (AT(qi, k) >= P3_QSMALL
                && (AT(qc, k) + AT(qr, k)) >= 1.0e-6f && t < 273.15f) {
                float qsat0 = 0.622f * P3_E0 / (AT(pres, k) - P3_E0);
                qwgrth = ((f1pr05 + f1pr14 * r_pow(sc, P3_THRD)
                           * r_pow(rhofaci * rho / mu, 0.5f))
                          * 2.0f * P3_PI
                          * (rho * P3_XXLV * dv * (qsat0 - qv_cld)
                             - (t - 273.15f) * kap)
                          / (P3_XLF + P3_CPW * (t - 273.15f))) * AT(ni, k);
                qwgrth = fmaxf(qwgrth, 0.0f);
                dum = fmaxf(0.0f, (qccol + qrcol) - qwgrth);
                if (dum >= 1.0e-10f) {
                    nrshdr = nrshdr + dum * 1.923e6f;
                    if ((qccol + qrcol) >= 1.0e-10f) {
                        dum1 = 1.0f / (qccol + qrcol);
                        qcshd = qcshd + dum * qccol * dum1;
                        qccol = qccol - dum * qccol * dum1;
                        qrcol = qrcol - dum * qrcol * dum1;
                    }
                    log_wetgrowth = 1;
                }
            }

            // inverse supersaturation relaxation timescale (:2900-2906)
            if (AT(qi, k) >= P3_QSMALL && t < 273.15f) {
                epsi = ((f1pr05 + f1pr14 * r_pow(sc, P3_THRD)
                         * r_pow(rhofaci * rho / mu, 0.5f))
                        * 2.0f * P3_PI * rho * dv) * AT(ni, k);
                epsi_tot = epsi_tot + epsi;
            } else {
                epsi = 0.0f;
            }

            // rime density, Cober and List 1993 (:2925-2969)
            if (qccol >= P3_QSMALL && t < 273.15f) {
                float vtrmi1 = f1pr02 * rhofaci;
                float iTc = 1.0f / fminf(-0.001f, t - 273.15f);
                if (AT(qc, k) >= P3_QSMALL) {
                    float Vt_qc = acn * p3_gam(4.0f + P3_BCN + mu_c)
                                  / (r_pow(lamc, P3_BCN)
                                     * p3_gam(mu_c + 4.0f));
                    float D_c = (mu_c + 4.0f) / lamc;
                    float V_impact = fabsf(vtrmi1 - Vt_qc);
                    float Ri = -(0.5e6f * D_c) * V_impact * iTc;
                    Ri = fmaxf(1.0f, fminf(Ri, 12.0f));
                    if (Ri <= 8.0f) {
                        rhorime_c = (0.051f + 0.114f * Ri
                                     - 0.0055f * (Ri * Ri)) * 1000.0f;
                    } else {
                        rhorime_c = 611.0f + 72.25f * (Ri - 8.0f);
                    }
                }
            } else {
                rhorime_c = 400.0f;
            }

            // immersion freezing of droplets (:2984-3015)
            //
            // OVERFLOW RESCUE, and the divergence is exactly the overflowing
            // case.  WRF's left-to-right single-precision chain
            // cons6*gamma*tmp1*dum**2 reaches 4.8e45 on the partial product
            // at t = 198.99 K / cdist1 = 3.394e5 / mu_c = 12.576 /
            // lamc = 1.6846e4 -- MEASURED, in the cell that took a real 6 h
            // forecast non-finite at step 284 -- while the mathematical
            // result 4.4e19 is an ordinary float.  The +Inf then meets the
            // conservation limiter at :3571-3583, ratio = sources/sinks is
            // 0, and Inf*0 is the NaN that reached theta and vapour.
            // Fortran does not fix the association of a*b*c*d, so WRF is
            // UNDEFINED here rather than authoritative; P3's own authors
            // left the double form commented out at :2999-3002.  The float
            // chain runs first and unchanged, so every representable value
            // is still WRF's bit for bit.  Mirrors
            // gpuwm/core/p3.py _rescue_overflowed_product exactly.
            if (AT(qc, k) >= P3_QSMALL && t <= 269.15f) {
                float d = 1.0f / lamc;
                dum = (d * d) * d;                                   // :2988 int **3
                tmp1 = cdist1 * r_exp(P3_AIMM * (273.15f - t));
                float gam_q = p3_gam(7.0f + mu_c);
                float gam_n = p3_gam(mu_c + 4.0f);
                float Q_nuc = P3_CONS6 * gam_q * tmp1
                              * (dum * dum);                         // :2995 int **2
                float N_nuc = P3_CONS5 * gam_n * tmp1 * dum;
                if (!isfinite(Q_nuc) || !isfinite(N_nuc)) {
                    // The argument is built in FLOAT32 and only the exp is
                    // double -- exactly the authority's own commented-out
                    // form, dexp(dble(aimm*(273.15-t(i,k)))) at :2999, and
                    // exactly what the rain branch below already does.  A
                    // bare `273.15` here would also be an unsuffixed double
                    // literal, which tests/test_p3_cuda.py refuses in this
                    // file on purpose.
                    double exp_aimm =
                        exp((double)(P3_AIMM * (273.15f - t)));
                    double dcdist1 = (double)cdist1;
                    double ddum = (double)dum;
                    if (!isfinite(Q_nuc)) {
                        Q_nuc = (float)((double)P3_CONS6 * (double)gam_q
                                        * dcdist1 * exp_aimm * ddum * ddum);
                    }
                    if (!isfinite(N_nuc)) {
                        N_nuc = (float)((double)P3_CONS5 * (double)gam_n
                                        * dcdist1 * exp_aimm * ddum);
                    }
                }
                qcheti = Q_nuc;
                ncheti = N_nuc;
            }

            // immersion freezing of rain (:3021-3043) -- the authority's
            // own double excursion, kept double.
            if (AT(qr, k) * ispf >= P3_QSMALL && t <= 269.15f) {
                double tmpdbl1 = exp((double)(r_log(cdistr)
                                     + r_log(p3_gam(7.0f + mu_r))
                                     - 6.0f * r_log(lamr)));
                double tmpdbl2 = exp((double)(r_log(cdistr)
                                     + r_log(p3_gam(mu_r + 4.0f))
                                     - 3.0f * r_log(lamr)));
                double tmpdbl3 = exp((double)(P3_AIMM * (273.15f - t)));
                qrheti = P3_CONS6 * (float)(tmpdbl1 * tmpdbl3) * spf;
                nrheti = P3_CONS5 * (float)(tmpdbl2 * tmpdbl3) * spf;
                // Same class as the droplet branch above: WRF's
                // sngl(tmpdbl1*tmpdbl3) (:3037-3038) can round a double to
                // +Inf for the same cold-cloud reason and reach the same
                // limiter.  The double product is in hand, so the rescue is
                // one more multiply.
                if (!isfinite(qrheti)) {
                    qrheti = (float)((double)P3_CONS6 * tmpdbl1 * tmpdbl3
                                     * (double)spf);
                }
                if (!isfinite(nrheti)) {
                    nrheti = (float)((double)P3_CONS5 * tmpdbl2 * tmpdbl3
                                     * (double)spf);
                }
            }

            // condensation / evaporation / deposition / sublimation
            float epsr;
            if (AT(qr, k) * ispf >= P3_QSMALL) {                     // :3114
                int dumii_r, dumjj_r;
                float rdumii, rdumjj;
                p3_find_lt_3(mu_r, lamr, &dumii_r, &dumjj_r,
                             &rdumii, &rdumjj);
                dum = p3_rain_table(revap, dumii_r, dumjj_r, rdumii, rdumjj);
                epsr = 2.0f * P3_PI * cdistr * rho * dv
                       * (P3_F1R * p3_gam(mu_r + 2.0f) / lamr
                          + P3_F2R * r_pow(rho / mu, 0.5f)
                            * r_pow(sc, P3_THRD) * dum);
            } else {
                epsr = 0.0f;
            }
            float epsc = (AT(qc, k) >= P3_QSMALL)
                         ? 2.0f * P3_PI * rho * dv * cdist : 0.0f;

            float xx;
            if (t < 273.15f) {
                oabi = 1.0f / abi;
                xx = epsc + epsr + epsi_tot
                     * (1.0f + P3_XXLS * P3_INV_CP * dqsdT) * oabi;
            } else {
                oabi = 1.0f / abi;
                xx = epsc + epsr;
            }

            float dumqvi = qvi;                                      // :3144

            dum = -P3_CP / P3_G * (t - t_old) * odt;                 // :3171
            if (t < 273.15f) {
                aaa = (AT(qv, k) - AT(qv_old, k)) * odt
                      - dqsdT * (-dum * P3_G * P3_INV_CP)
                      - (qvs - dumqvi) * (1.0f + P3_XXLS * P3_INV_CP * dqsdT)
                        * oabi * epsi_tot;
            } else {
                aaa = (AT(qv, k) - AT(qv_old, k)) * odt
                      - dqsdT * (-dum * P3_G * P3_INV_CP);
            }

            xx = fmaxf(1.0e-20f, xx);                                // :3182
            oxx = 1.0f / xx;

            float ssat_cld = AT(ssat, k);                            // :3186
            float ssat_r = AT(ssat, k);
            float sup_cld = sup, sup_r = sup, supi_cld = supi;

            if (AT(qc, k) >= P3_QSMALL) {                            // :3202
                qccon = (aaa * epsc * oxx
                         + (ssat_cld * scf - aaa * oxx) * odt * epsc * oxx
                           * (1.0f - (float)exp(-(double)(xx * dt)))) / ab;
            }
            if (AT(qr, k) >= P3_QSMALL) {                            // :3204
                qrcon = (aaa * epsr * oxx
                         + (ssat_r * spf - aaa * oxx) * odt * epsr * oxx
                           * (1.0f - (float)exp(-(double)(xx * dt)))) / ab;
            }

            if (sup_cld < -0.001f && AT(qc, k) < 1.0e-12f)
                qccon = -AT(qc, k) * odt;
            if (sup_r < -0.001f && AT(qr, k) < 1.0e-12f)
                qrcon = -AT(qr, k) * odt;

            if (qccon < 0.0f) { qcevp = -qccon; qccon = 0.0f; }
            else              { qccon = fminf(qccon, AT(qv, k) * odt); }

            if (qrcon < 0.0f) {
                qrevp = -qrcon;
                nrevp = qrevp * (AT(nr, k) / AT(qr, k));
                qrcon = 0.0f;
            } else {
                qrcon = fminf(qrcon, AT(qv, k) * odt);
            }

            if (AT(qi, k) >= P3_QSMALL && t < 273.15f) {             // :3231
                qidep = (aaa * epsi * oxx
                         + (ssat_cld * scf - aaa * oxx) * odt * epsi * oxx
                           * (1.0f - (float)exp(-(double)(xx * dt)))) * oabi
                        + (qvs - dumqvi) * epsi * oabi;
            }
            if (supi_cld < -0.001f && AT(qi, k) < 1.0e-12f)
                qidep = -AT(qi, k) * odt;
            if (qidep < 0.0f) {
                qisub = -qidep;
                qisub = qisub * clbfact_sub;
                qisub = fminf(qisub, AT(qi, k) * odt);
                nisub = qisub * (AT(ni, k) / AT(qi, k));
                qidep = 0.0f;
            } else {
                qidep = qidep * clbfact_dep;
                qidep = fminf(qidep, AT(qv, k) * odt);
            }
        }
        // ---- 444 continue (:3257) ----------------------------------

        float sup_cld = sup;
        float supi_cld = supi;

        // deposition/condensation-freezing nucleation, Cooper 1986 (:3264)
        if (t < 258.15f && supi_cld >= 0.05f) {
            dum = 0.005f * r_exp(0.304f * (273.15f - t)) * 1000.0f * inv_rho;
            dum = fminf(dum, 100.0e3f * inv_rho * scf);
            float N_nuc = fmaxf(0.0f, (dum - AT(ni, k)) * odt);
            if (N_nuc >= 1.0e-20f) {
                float Q_nuc = fmaxf(0.0f, (dum - AT(ni, k)) * P3_MI0 * odt);
                qinuc = Q_nuc;
                ninuc = N_nuc;
            }
        }

        // SEAM 2 -- droplet activation, both log_predictNc branches.
        p3_activate_droplets(log_predictNc, sup_cld, it, t, AT(pres, k),
                             AT(qc, k), AT(nc, k), qv_cld, inv_rho, iscf, scf,
                             odt, &qcnuc, &ncnuc);

        // first-step saturation adjustment (:3348-3356)
        if (it <= 1) {
            float dumt = AT(th, k) * r_pow(AT(pres, k) * 1.0e-5f,
                                           P3_RD * P3_INV_CP);
            float dumqv = qv_cld;
            float dumqvs = p3_qv_sat(dumt, AT(pres, k), 0);
            float dums = dumqv - dumqvs;
            qccon = dums / (1.0f + (P3_XXLV * P3_XXLV) * dumqvs
                            / (P3_CP * P3_RV * (dumt * dumt)))
                    * odt * scf;                         // :3353, integer **2
            qccon = fmaxf(0.0f, qccon);
            if (qccon <= 1.0e-7f) qccon = 0.0f;
        }

        // autoconversion, iparam = 3, KK2000 (:3396-3402)
        if (AT(qc, k) * iscf >= 1.0e-8f) {
            dum = AT(qc, k) * iscf;
            qcaut = 1350.0f * r_pow(dum, 2.47f)
                    * r_pow(AT(nc, k) * iscf * 1.0e-6f * rho, -1.79f) * scf;
            ncautr = qcaut * P3_CONS3;
            ncautc = qcaut * AT(nc, k) / AT(qc, k);
            if (qcaut == 0.0f) ncautc = 0.0f;
            if (ncautc == 0.0f) qcaut = 0.0f;
        }

        // self-collection of droplets, iparam = 3 (:3421-3435)
        if (AT(qc, k) >= P3_QSMALL) ncslf = 0.0f;

        // accretion, iparam = 3 (:3458-3462)
        if (AT(qr, k) >= P3_QSMALL && AT(qc, k) >= P3_QSMALL) {
            dum2 = spf - spf_clr;
            qcacc = 67.0f * r_pow(AT(qc, k) * iscf * AT(qr, k) * ispf, 1.15f)
                    * dum2;
            ncacc = qcacc * AT(nc, k) / AT(qc, k);
            if (qcacc == 0.0f) ncacc = 0.0f;
            if (ncacc == 0.0f) qcacc = 0.0f;
        }

        // rain self-collection / breakup, iparam = 3 (:3478-3504)
        if (AT(qr, k) >= P3_QSMALL) {
            dum1 = 280.0e-6f;
            AT(nr, k) = fmaxf(AT(nr, k), P3_NSMALL);
            dum2 = r_pow(AT(qr, k) / (P3_PI * P3_RHOW * AT(nr, k)), P3_THRD);
            if (dum2 < dum1) dum = 1.0f;
            else             dum = 2.0f - r_exp(2300.0f * (dum2 - dum1));
            nrslf = dum * 5.78f * AT(nr, k) * ispf * AT(qr, k) * ispf
                    * rho * spf;
        }

        // ---- conservation of mass (:3515-3644) ---------------------
        float ratio;
        {
            float dumqvs = p3_qv_sat(t, AT(pres, k), 0);
            float qcon_satadj = (qv_cld - dumqvs)
                / (1.0f + (P3_XXLV * P3_XXLV) * dumqvs
                   / (P3_CP * P3_RV * (t * t))) * odt * scf;  // :3516 int **2
            tmp1 = qccon + qrcon + qcnuc;
            if (tmp1 > 0.0f && qcon_satadj < 0.0f) {
                qccon = 0.0f; qrcon = 0.0f; qcnuc = 0.0f; ncnuc = 0.0f;
            } else {
                if (tmp1 > 0.0f && tmp1 > qcon_satadj) {
                    ratio = fmaxf(0.0f, qcon_satadj) / tmp1;
                    ratio = fminf(1.0f, ratio);
                    qccon = qccon * ratio;
                    qrcon = qrcon * ratio;
                    qcnuc = qcnuc * ratio;
                    ncnuc = ncnuc * ratio;
                } else if (qcevp + qrevp > 0.0f) {
                    ratio = fmaxf(0.0f, -qcon_satadj) / (qcevp + qrevp);
                    ratio = fminf(1.0f, ratio);
                    qcevp = qcevp * ratio;
                    qrevp = qrevp * ratio;
                    nrevp = nrevp * ratio;
                }
            }

            float qv_tmp = qv_cld + (-qcnuc - qccon - qrcon + qcevp + qrevp)
                           * dt;
            float t_tmp = t + (qcnuc + qccon + qrcon - qcevp - qrevp)
                          * P3_XXLV * P3_INV_CP * dt;
            float dumqvi = p3_qv_sat(t_tmp, AT(pres, k), 1);
            float qdep_satadj = (qv_tmp - dumqvi)
                / (1.0f + (P3_XXLS * P3_XXLS) * dumqvi
                   / (P3_CP * P3_RV * (t_tmp * t_tmp))) * odt * scf;
            tmp1 = qidep + qinuc;
            if (tmp1 > 0.0f && qdep_satadj < 0.0f) {
                qidep = 0.0f; qinuc = 0.0f; ninuc = 0.0f;
            } else {
                if (tmp1 > 0.0f && tmp1 > qdep_satadj) {
                    ratio = fmaxf(0.0f, qdep_satadj) / tmp1;
                    ratio = fminf(1.0f, ratio);
                    qidep = qidep * ratio;
                    qinuc = qinuc * ratio;
                    ninuc = ninuc * ratio;
                }
                dum = fmaxf(qisub, 1.0e-20f);
                qisub = qisub * fminf(1.0f, fmaxf(0.0f, -qdep_satadj)
                                            / fmaxf(qisub, 1.0e-20f));
                nisub = nisub * fminf(1.0f, qisub / dum);
            }
        }

        float sinks, sources;
        sinks = (qcaut + qcacc + qccol + qcevp + qchetc + qcheti + qcshd) * dt;
        sources = AT(qc, k) + (qccon + qcnuc) * dt;
        if (sinks > sources && sinks >= 1.0e-20f) {                  // :3571
            ratio = sources / sinks;
            qcaut *= ratio; qcacc *= ratio; qcevp *= ratio; qccol *= ratio;
            qcheti *= ratio; qcshd *= ratio; ncautc *= ratio; ncacc *= ratio;
            nccol *= ratio; ncheti *= ratio;
        }

        sinks = (qrevp + qrcol + qrhetc + qrheti + qrmul) * dt;
        sources = AT(qr, k) + (qrcon + qcaut + qcacc + qimlt + qcshd) * dt;
        if (sinks > sources && sinks >= 1.0e-20f) {                  // :3593
            ratio = sources / sinks;
            qrevp *= ratio; qrcol *= ratio; qrheti *= ratio; qrmul *= ratio;
            nrevp *= ratio; nrcol *= ratio; nrheti *= ratio;
        }

        sinks = (qisub + qimlt) * dt;
        sources = AT(qi, k) + (qidep + qinuc + qrcol + qccol + qrhetc
                               + qrheti + qchetc + qcheti + qrmul) * dt;
        if (sinks > sources && sinks >= 1.0e-20f) {                  // :3609
            ratio = sources / sinks;
            qisub *= ratio; qimlt *= ratio; nisub *= ratio; nimlt *= ratio;
        }

        sinks = (qccon + qrcon + qcnuc + qidep + qinuc) * dt;
        sources = AT(qv, k) + (qcevp + qrevp + qisub) * dt;
        if (sinks > sources && sinks >= 1.0e-20f) {                  // :3633
            ratio = sources / sinks;
            qccon *= ratio; qrcon *= ratio; qcnuc *= ratio; qidep *= ratio;
            qinuc *= ratio; ninuc *= ratio; ncnuc *= ratio;
        }

        // ---- update prognostic variables (:3756-3921) ---------------
        float rimevolume = 0.0f, rimefraction = 0.0f;
        if (AT(qi, k) >= P3_QSMALL) {                                // :3756
            tmp1 = 1.0f / AT(qi, k);
            rimevolume = AT(qib, k) * tmp1;
            rimefraction = AT(qir, k) * tmp1;
        }

        AT(qc, k) = AT(qc, k) + (-qchetc - qcheti - qccol - qcshd) * dt;
        if (log_predictNc) {                                         // :3770
            AT(nc, k) = AT(nc, k) + (-nccol - nchetc - ncheti) * dt;
        }
        AT(qr, k) = AT(qr, k) + (-qrcol + qimlt - qrhetc - qrheti + qcshd
                                 - qrmul) * dt;
        AT(nr, k) = AT(nr, k) + (-nrcol - nrhetc - nrheti
                                 + P3_NMLTRATIO * nimlt + nrshdr + ncshdc)
                                * dt;

        AT(qib, k) = AT(qib, k) - (qisub + qimlt) * dt * rimevolume;
        AT(qir, k) = AT(qir, k) - (qisub + qimlt) * dt * rimefraction;
        AT(qi, k) = AT(qi, k) - (qisub + qimlt) * dt;

        dum = (qrcol + qccol + qrhetc + qrheti + qchetc + qcheti + qrmul) * dt;
        AT(qi, k) = AT(qi, k) + (qidep + qinuc) * dt + dum;
        AT(qir, k) = AT(qir, k) + dum;
        AT(qib, k) = AT(qib, k) + (qrcol * P3_INV_RHO_RIMEMAX
                                   + qccol / rhorime_c
                                   + (qrhetc + qrheti + qchetc + qcheti
                                      + qrmul) * P3_INV_RHO_RIMEMAX) * dt;
        AT(ni, k) = AT(ni, k) + (ninuc - nimlt - nisub - nislf
                                 + nrhetc + nrheti + nchetc + ncheti + nimul)
                                * dt;

        if (AT(qir, k) < 0.0f) { AT(qir, k) = 0.0f; AT(qib, k) = 0.0f; }

        if (log_wetgrowth) {                                         // :3835
            AT(qir, k) = AT(qi, k);
            AT(qib, k) = AT(qir, k) * P3_INV_RHO_RIMEMAX;
        }

        if (AT(qi, k) >= P3_QSMALL && AT(qib, k) >= P3_BSMALL
            && qimlt > 0.0f) {                                       // :3841
            tmp1 = AT(qir, k) / AT(qib, k);
            tmp2 = AT(qi, k) + qimlt * dt;
            AT(qib, k) = AT(qir, k) / (tmp1 + (917.0f - tmp1) * qimlt
                                       * dt / tmp2);
        }

        AT(qv, k) = AT(qv, k) + (-qidep + qisub - qinuc) * dt;
        AT(th, k) = AT(th, k) + invexn * ((qidep - qisub + qinuc)
                                          * P3_XXLS * P3_INV_CP
                                          + (qrcol + qccol + qchetc + qcheti
                                             + qrhetc + qrheti + qrmul - qimlt)
                                            * P3_XLF * P3_INV_CP) * dt;

        // warm-phase updates (:3869-3885)
        AT(qc, k) = AT(qc, k) + (-qcacc - qcaut + qcnuc + qccon - qcevp) * dt;
        AT(qr, k) = AT(qr, k) + (qcacc + qcaut + qrcon - qrevp) * dt;
        if (log_predictNc) {                                         // :3872
            AT(nc, k) = AT(nc, k) + (-ncacc - ncautc + ncslf + ncnuc) * dt;
        } else {
            AT(nc, k) = P3_NCCNST * inv_rho;                         // :3875
        }
        AT(nr, k) = AT(nr, k) + (ncautr - nrslf - nrevp) * dt;       // iparam=3
        AT(qv, k) = AT(qv, k) + (-qcnuc - qccon - qrcon + qcevp + qrevp) * dt;
        AT(th, k) = AT(th, k) + invexn * ((qcnuc + qccon + qrcon - qcevp
                                           - qrevp) * P3_XXLV * P3_INV_CP)
                                * dt;

        // clipping (:3889-3918)
        if (AT(qc, k) < P3_QSMALL) {
            AT(qv, k) = AT(qv, k) + AT(qc, k);
            AT(th, k) = AT(th, k) - invexn * AT(qc, k) * P3_XXLV * P3_INV_CP;
            AT(qc, k) = 0.0f;
            AT(nc, k) = 0.0f;
        } else {
            hydrometeors_present = 1;
        }

        if (AT(qr, k) < P3_QSMALL) {
            AT(qv, k) = AT(qv, k) + AT(qr, k);
            AT(th, k) = AT(th, k) - invexn * AT(qr, k) * P3_XXLV * P3_INV_CP;
            AT(qr, k) = 0.0f;
            AT(nr, k) = 0.0f;
        } else {
            hydrometeors_present = 1;
        }

        if (AT(qi, k) < P3_QSMALL) {
            AT(qv, k) = AT(qv, k) + AT(qi, k);
            AT(th, k) = AT(th, k) - invexn * AT(qi, k) * P3_XXLS * P3_INV_CP;
            AT(qi, k) = 0.0f;
            AT(ni, k) = 0.0f;
            AT(qir, k) = 0.0f;
            AT(qib, k) = 0.0f;
        } else {
            hydrometeors_present = 1;
        }

        AT(qv, k) = fmaxf(0.0f, AT(qv, k));
        AT(ni, k) = p3_impose_max_ni(AT(ni, k), inv_rho);
        // 555 continue
    }

    // second compute_SCPF (:3968-3970): refresh the qv snapshot.
    for (int k = 0; k < nk; ++k) AT(S[S_QVCLD], k) = AT(F[F_QV], k);
    FLG[ncol + i] = hydrometeors_present ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------
// Sedimentation (:3984-4495).  Three species, each with the authority's
// adaptive Courant substepping.  kbot = 0, ktop = nk-1, kdir = +1 (the
// WRF orientation, :2211-2215).  inv_dzq is 1/dz taken where the
// authority takes it (:2260); it is the same value every time, so it is
// recomputed rather than stored.
// ---------------------------------------------------------------------
__device__ void p3_step_sed_cloud_col(P3_ARGS)
{
    if (FLG[ncol + i] == 0.0f) return;                                      // goto 333
    float* __restrict__ qc = F[F_QC];
    float* __restrict__ nc = F[F_NC];
    const float* __restrict__ dz = F[F_DZ];
    const float iscf = 1.0f;
    const float odt = 1.0f / dt;
    const int kbot = 0, ktop = nk - 1;

    int log_qxpresent = 0, k_qxtop = kbot;
    for (int k = ktop; k >= kbot; --k) {                             // :4064
        if (AT(qc, k) * iscf >= P3_QSMALL) {
            log_qxpresent = 1; k_qxtop = k; break;
        }
    }
    if (!log_qxpresent) return;

    float dt_left = dt, prt_accum = 0.0f;
    int k_qxbot = kbot;
    for (int k = kbot; k <= k_qxtop; ++k) {                          // :4002
        if (AT(qc, k) * iscf >= P3_QSMALL) { k_qxbot = k; break; }
    }
    for (int k = 0; k < nk; ++k) {
        AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
        AT(W[W_FQ], k) = 0.0f; AT(W[W_FN], k) = 0.0f;
    }

    while (dt_left > 1.0e-4f) {
        float Co_max = 0.0f;
        for (int k = 0; k < nk; ++k) {
            AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
        }
        for (int k = k_qxtop; k >= k_qxbot; --k) {
            if (AT(qc, k) * iscf > P3_QSMALL) {
                float mu_c, lamc, t1, t2, ncg = AT(nc, k);
                p3_get_cloud_dsd2(&ncg, AT(qc, k), AT(S[S_RHO], k), iscf,
                                  &mu_c, &lamc, &t1, &t2);
                AT(nc, k) = ncg;
                float dum = 1.0f / r_pow(lamc, P3_BCN);              // :4022
                AT(W[W_VQ], k) = AT(S[S_ACN], k)
                                 * p3_gam(4.0f + P3_BCN + mu_c) * dum
                                 / p3_gam(mu_c + 4.0f);
                if (log_predictNc) {                                 // :4025
                    AT(W[W_VN], k) = AT(S[S_ACN], k)
                                     * p3_gam(1.0f + P3_BCN + mu_c) * dum
                                     / p3_gam(mu_c + 1.0f);
                }
            }
            Co_max = fmaxf(Co_max, AT(W[W_VQ], k) * dt_left
                                   * (1.0f / AT(dz, k)));
        }
        int tmpint1 = (int)(Co_max + 1.0f);
        float dt_sub = fminf(dt_left, dt_left / (float)tmpint1);
        int k_temp = (k_qxbot == kbot) ? k_qxbot : k_qxbot - 1;
        for (int k = k_temp; k <= k_qxtop; ++k) {
            AT(W[W_FQ], k) = AT(W[W_VQ], k) * AT(qc, k) * AT(S[S_RHO], k);
            if (log_predictNc) {
                AT(W[W_FN], k) = AT(W[W_VN], k) * AT(nc, k) * AT(S[S_RHO], k);
            }
        }
        if (k_qxbot == kbot) prt_accum = prt_accum + AT(W[W_FQ], kbot) * dt_sub;
        {
            int k = k_qxtop;
            float idz = 1.0f / AT(dz, k);
            float fdq = -AT(W[W_FQ], k) * idz;
            AT(qc, k) = AT(qc, k) + fdq * dt_sub * AT(S[S_INVRHO], k);
            if (log_predictNc) {
                float fdn = -AT(W[W_FN], k) * idz;
                AT(nc, k) = AT(nc, k) + fdn * dt_sub * AT(S[S_INVRHO], k);
            }
        }
        for (int k = k_qxtop - 1; k >= k_temp; --k) {
            float idz = 1.0f / AT(dz, k);
            float fdq = (AT(W[W_FQ], k + 1) - AT(W[W_FQ], k)) * idz;
            AT(qc, k) = AT(qc, k) + fdq * dt_sub * AT(S[S_INVRHO], k);
            if (log_predictNc) {
                float fdn = (AT(W[W_FN], k + 1) - AT(W[W_FN], k)) * idz;
                AT(nc, k) = AT(nc, k) + fdn * dt_sub * AT(S[S_INVRHO], k);
            }
        }
        dt_left = dt_left - dt_sub;
        if (k_qxbot != kbot) k_qxbot = k_qxbot - 1;
    }
    P[P_PRTLIQ][i] = prt_accum * P3_INV_RHOW * odt;
}

__device__ void p3_step_sed_rain_col(P3_ARGS)
{
    if (FLG[ncol + i] == 0.0f) return;
    float* __restrict__ qr = F[F_QR];
    float* __restrict__ nr = F[F_NR];
    const float* __restrict__ dz = F[F_DZ];
    const float* __restrict__ vn_tab = T[T_VN];
    const float* __restrict__ vm_tab = T[T_VM];
    const float ispf = 1.0f;
    const float odt = 1.0f / dt;
    const int kbot = 0, ktop = nk - 1;

    int log_qxpresent = 0, k_qxtop = kbot;
    for (int k = ktop; k >= kbot; --k) {
        if (AT(qr, k) * ispf >= P3_QSMALL) {
            log_qxpresent = 1; k_qxtop = k; break;
        }
    }
    if (!log_qxpresent) return;

    float dt_left = dt, prt_accum = 0.0f;
    int k_qxbot = kbot;
    for (int k = kbot; k <= k_qxtop; ++k) {
        if (AT(qr, k) * ispf >= P3_QSMALL) { k_qxbot = k; break; }
    }
    for (int k = 0; k < nk; ++k) {
        AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
        AT(W[W_FQ], k) = 0.0f; AT(W[W_FN], k) = 0.0f;
    }

    while (dt_left > 1.0e-4f) {
        float Co_max = 0.0f;
        for (int k = 0; k < nk; ++k) {
            AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
        }
        for (int k = k_qxtop; k >= k_qxbot; --k) {
            if (AT(qr, k) * ispf > P3_QSMALL) {
                AT(nr, k) = fmaxf(AT(nr, k), P3_NSMALL);
                float mu_r, lamr, cdistr, logn0r, nrg = AT(nr, k);
                p3_get_rain_dsd2(&nrg, AT(qr, k), ispf,
                                 &mu_r, &lamr, &cdistr, &logn0r);
                AT(nr, k) = nrg;
                int dumii_r, dumjj_r;
                float rdumii, rdumjj;
                p3_find_lt_3(mu_r, lamr, &dumii_r, &dumjj_r,
                             &rdumii, &rdumjj);
                float vq = p3_rain_table(vm_tab, dumii_r, dumjj_r,
                                         rdumii, rdumjj);
                AT(W[W_VQ], k) = vq * AT(S[S_RHOFACR], k);
                float vn = p3_rain_table(vn_tab, dumii_r, dumjj_r,
                                         rdumii, rdumjj);
                AT(W[W_VN], k) = vn * AT(S[S_RHOFACR], k);
            }
            Co_max = fmaxf(Co_max, AT(W[W_VQ], k) * dt_left
                                   * (1.0f / AT(dz, k)));
        }
        int tmpint1 = (int)(Co_max + 1.0f);
        float dt_sub = fminf(dt_left, dt_left / (float)tmpint1);
        int k_temp = (k_qxbot == kbot) ? k_qxbot : k_qxbot - 1;
        for (int k = k_temp; k <= k_qxtop; ++k) {
            AT(W[W_FQ], k) = AT(W[W_VQ], k) * AT(qr, k) * AT(S[S_RHO], k);
            AT(W[W_FN], k) = AT(W[W_VN], k) * AT(nr, k) * AT(S[S_RHO], k);
        }
        if (k_qxbot == kbot) prt_accum = prt_accum + AT(W[W_FQ], kbot) * dt_sub;
        {
            int k = k_qxtop;
            float idz = 1.0f / AT(dz, k);
            float fdq = -AT(W[W_FQ], k) * idz;
            float fdn = -AT(W[W_FN], k) * idz;
            AT(qr, k) = AT(qr, k) + fdq * dt_sub * AT(S[S_INVRHO], k);
            AT(nr, k) = AT(nr, k) + fdn * dt_sub * AT(S[S_INVRHO], k);
        }
        for (int k = k_qxtop - 1; k >= k_temp; --k) {
            float idz = 1.0f / AT(dz, k);
            float fdq = (AT(W[W_FQ], k + 1) - AT(W[W_FQ], k)) * idz;
            float fdn = (AT(W[W_FN], k + 1) - AT(W[W_FN], k)) * idz;
            AT(qr, k) = AT(qr, k) + fdq * dt_sub * AT(S[S_INVRHO], k);
            AT(nr, k) = AT(nr, k) + fdn * dt_sub * AT(S[S_INVRHO], k);
        }
        dt_left = dt_left - dt_sub;
        if (k_qxbot != kbot) k_qxbot = k_qxbot - 1;
    }
    P[P_PRTLIQ][i] = P[P_PRTLIQ][i] + prt_accum * P3_INV_RHOW * odt;
}

__device__ void p3_step_sed_ice_col(P3_ARGS)
{
    if (FLG[ncol + i] == 0.0f) return;
    float* __restrict__ qi = F[F_QI];
    float* __restrict__ qir = F[F_QIR];
    float* __restrict__ ni = F[F_NI];
    float* __restrict__ qib = F[F_QIB];
    const float* __restrict__ dz = F[F_DZ];
    const float* __restrict__ itab = T[T_ITAB];
    const float odt = 1.0f / dt;
    const int kbot = 0, ktop = nk - 1;

    int log_qxpresent = 0, k_qxtop = kbot;
    for (int k = ktop; k >= kbot; --k) {
        if (AT(qi, k) >= P3_QSMALL) { log_qxpresent = 1; k_qxtop = k; break; }
    }
    if (!log_qxpresent) return;

    float dt_left = dt, prt_accum = 0.0f;
    int k_qxbot = kbot;
    for (int k = kbot; k <= k_qxtop; ++k) {
        if (AT(qi, k) >= P3_QSMALL) { k_qxbot = k; break; }
    }
    for (int k = 0; k < nk; ++k) {
        AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
        AT(W[W_FQ], k) = 0.0f; AT(W[W_FN], k) = 0.0f;
        AT(W[W_FQIR], k) = 0.0f; AT(W[W_FBIR], k) = 0.0f;
    }

    while (dt_left > 1.0e-4f) {
        float Co_max = 0.0f;
        for (int k = 0; k < nk; ++k) {
            AT(W[W_VQ], k) = 0.0f; AT(W[W_VN], k) = 0.0f;
        }
        for (int k = k_qxtop; k >= k_qxbot; --k) {
            if (AT(qi, k) >= P3_QSMALL) {
                AT(ni, k) = fmaxf(AT(ni, k), P3_NSMALL);
                float rhop;
                {   float qq = AT(qir, k), bb = AT(qib, k);
                    p3_calc_bulk_rho_rime(AT(qi, k), &qq, &bb, &rhop);
                    AT(qir, k) = qq; AT(qib, k) = bb; }
                int dumi, dumjj, dumii;
                float d1, d4, d5;
                p3_find_lt_1a(AT(qi, k), AT(ni, k), AT(qir, k), rhop,
                              &dumi, &dumjj, &dumii, &d1, &d4, &d5);
                float f1pr01 = p3_access_lookup_table(itab, dumjj, dumii,
                                                      dumi, 1, d1, d4, d5);
                float f1pr02 = p3_access_lookup_table(itab, dumjj, dumii,
                                                      dumi, 2, d1, d4, d5);
                float f1pr09 = p3_access_lookup_table(itab, dumjj, dumii,
                                                      dumi, 7, d1, d4, d5);
                float f1pr10 = p3_access_lookup_table(itab, dumjj, dumii,
                                                      dumi, 8, d1, d4, d5);
                AT(ni, k) = fminf(AT(ni, k), f1pr09 * AT(qi, k));
                AT(ni, k) = fmaxf(AT(ni, k), f1pr10 * AT(qi, k));
                AT(W[W_VQ], k) = f1pr02 * AT(S[S_RHOFACI], k);
                AT(W[W_VN], k) = f1pr01 * AT(S[S_RHOFACI], k);
            }
            Co_max = fmaxf(Co_max, AT(W[W_VQ], k) * dt_left
                                   * (1.0f / AT(dz, k)));
        }
        int tmpint1 = (int)(Co_max + 1.0f);
        float dt_sub = fminf(dt_left, dt_left / (float)tmpint1);
        int k_temp = (k_qxbot == kbot) ? k_qxbot : k_qxbot - 1;
        for (int k = k_temp; k <= k_qxtop; ++k) {
            float vq = AT(W[W_VQ], k), rho = AT(S[S_RHO], k);
            AT(W[W_FQ], k) = vq * AT(qi, k) * rho;
            AT(W[W_FN], k) = AT(W[W_VN], k) * AT(ni, k) * rho;
            AT(W[W_FQIR], k) = vq * AT(qir, k) * rho;
            AT(W[W_FBIR], k) = vq * AT(qib, k) * rho;
        }
        if (k_qxbot == kbot) prt_accum = prt_accum + AT(W[W_FQ], kbot) * dt_sub;
        {
            int k = k_qxtop;
            float idz = 1.0f / AT(dz, k), irho = AT(S[S_INVRHO], k);
            AT(qi, k) += (-AT(W[W_FQ], k) * idz) * dt_sub * irho;
            AT(qir, k) += (-AT(W[W_FQIR], k) * idz) * dt_sub * irho;
            AT(qib, k) += (-AT(W[W_FBIR], k) * idz) * dt_sub * irho;
            AT(ni, k) += (-AT(W[W_FN], k) * idz) * dt_sub * irho;
        }
        for (int k = k_qxtop - 1; k >= k_temp; --k) {
            float idz = 1.0f / AT(dz, k), irho = AT(S[S_INVRHO], k);
            AT(qi, k) += ((AT(W[W_FQ], k + 1) - AT(W[W_FQ], k)) * idz)
                         * dt_sub * irho;
            AT(qir, k) += ((AT(W[W_FQIR], k + 1) - AT(W[W_FQIR], k)) * idz)
                          * dt_sub * irho;
            AT(qib, k) += ((AT(W[W_FBIR], k + 1) - AT(W[W_FBIR], k)) * idz)
                          * dt_sub * irho;
            AT(ni, k) += ((AT(W[W_FN], k + 1) - AT(W[W_FN], k)) * idz)
                         * dt_sub * irho;
        }
        dt_left = dt_left - dt_sub;
        if (k_qxbot != kbot) k_qxbot = k_qxbot - 1;
    }
    P[P_PRTSOL][i] = P[P_PRTSOL][i] + prt_accum * P3_INV_RHOW * odt;
}

// ---------------------------------------------------------------------
// Level bodies for the last two loops.  Both are level-local, so the
// unfused arm runs them as two k-loops and the fused arm runs them as one;
// these functions are what makes those two arms the same arithmetic.
// ---------------------------------------------------------------------
__device__ __forceinline__ void p3_homofreeze_level(P3_ARGS, int k)
{
    float* __restrict__ qc = F[F_QC];
    float* __restrict__ nc = F[F_NC];
    float* __restrict__ qr = F[F_QR];
    float* __restrict__ nr = F[F_NR];
    float* __restrict__ qi = F[F_QI];
    float* __restrict__ qir = F[F_QIR];
    float* __restrict__ ni = F[F_NI];
    float* __restrict__ qib = F[F_QIB];
    float* __restrict__ th = F[F_TH];
    const float t = AT(S[S_T], k);
    const float invexn = 1.0f / AT(S[S_TMPARR1], k);

    if (AT(qc, k) >= P3_QSMALL && t < 233.15f) {                     // :4529
        float Q_nuc = AT(qc, k);
        AT(nc, k) = fmaxf(AT(nc, k), P3_NSMALL);
        float N_nuc = AT(nc, k);
        AT(qir, k) = AT(qir, k) + Q_nuc;
        AT(qi, k) = AT(qi, k) + Q_nuc;
        AT(qib, k) = AT(qib, k) + Q_nuc * P3_INV_RHO_RIMEMAX;
        AT(ni, k) = AT(ni, k) + N_nuc;
        AT(th, k) = AT(th, k) + invexn * Q_nuc * P3_XLF * P3_INV_CP;
        AT(qc, k) = 0.0f;
        AT(nc, k) = 0.0f;
    }
    if (AT(qr, k) >= P3_QSMALL && t < 233.15f) {                     // :4574
        float Q_nuc = AT(qr, k);
        AT(nr, k) = fmaxf(AT(nr, k), P3_NSMALL);
        float N_nuc = AT(nr, k);
        AT(qir, k) = AT(qir, k) + Q_nuc;
        AT(qi, k) = AT(qi, k) + Q_nuc;
        AT(qib, k) = AT(qib, k) + Q_nuc * P3_INV_RHO_RIMEMAX;
        AT(ni, k) = AT(ni, k) + N_nuc;
        AT(th, k) = AT(th, k) + invexn * Q_nuc * P3_XLF * P3_INV_CP;
        AT(qr, k) = 0.0f;
        AT(nr, k) = 0.0f;
    }
}

// final checks + diagnostics (:4722-4895).  ze_ice/ze_rain are the
// authority's (i,k) arrays but are written and read only at this level, so
// they live in registers instead of a scratch field.
__device__ __forceinline__ void p3_final_level(P3_ARGS, int k)
{
    float* __restrict__ qc = F[F_QC];
    float* __restrict__ nc = F[F_NC];
    float* __restrict__ qr = F[F_QR];
    float* __restrict__ nr = F[F_NR];
    float* __restrict__ qi = F[F_QI];
    float* __restrict__ qir = F[F_QIR];
    float* __restrict__ ni = F[F_NI];
    float* __restrict__ qib = F[F_QIB];
    float* __restrict__ th = F[F_TH];
    float* __restrict__ qv = F[F_QV];
    const float* __restrict__ itab = T[T_ITAB];
    const float iscf = 1.0f;
    const float rho = AT(S[S_RHO], k);
    const float inv_rho = AT(S[S_INVRHO], k);
    const float invexn = 1.0f / AT(S[S_TMPARR1], k);
    float ze_ice = 1.0e-22f, ze_rain = 1.0e-22f;                     // :2286-2287

    if (AT(qc, k) * iscf >= P3_QSMALL) {
        float mu_c, lamc, t1, t2, ncg = AT(nc, k);
        p3_get_cloud_dsd2(&ncg, AT(qc, k), rho, iscf, &mu_c, &lamc, &t1, &t2);
        AT(nc, k) = ncg;
        AT(D[D_EFFC], k) = 0.5f * (mu_c + 3.0f) / lamc;              // :4732
    } else {
        AT(qv, k) = AT(qv, k) + AT(qc, k);
        AT(th, k) = AT(th, k) - invexn * AT(qc, k) * P3_XXLV * P3_INV_CP;
        AT(qc, k) = 0.0f;
        AT(nc, k) = 0.0f;
    }

    if (AT(qr, k) >= P3_QSMALL) {
        float mu_r, lamr, cdistr, logn0r, nrg = AT(nr, k);
        // the authority passes the LITERAL iSPF = 1. here (:4739)
        p3_get_rain_dsd2(&nrg, AT(qr, k), 1.0f,
                         &mu_r, &lamr, &cdistr, &logn0r);
        AT(nr, k) = nrg;
        float l2 = lamr * lamr;
        float lam6 = (l2 * l2) * l2;                     // :4756, integer **6
        ze_rain = rho * AT(nr, k) * (mu_r + 6.0f) * (mu_r + 5.0f)
                  * (mu_r + 4.0f) * (mu_r + 3.0f) * (mu_r + 2.0f)
                  * (mu_r + 1.0f) / lam6;
        ze_rain = fmaxf(ze_rain, 1.0e-22f);
    } else {
        AT(qv, k) = AT(qv, k) + AT(qr, k);
        AT(th, k) = AT(th, k) - invexn * AT(qr, k) * P3_XXLV * P3_INV_CP;
        AT(qr, k) = 0.0f;
        AT(nr, k) = 0.0f;
    }

    AT(ni, k) = p3_impose_max_ni(AT(ni, k), inv_rho);

    if (AT(qi, k) >= P3_QSMALL) {
        AT(ni, k) = fmaxf(AT(ni, k), P3_NSMALL);
        AT(nr, k) = fmaxf(AT(nr, k), P3_NSMALL);
        float rhop;
        {   float qq = AT(qir, k), bb = AT(qib, k);
            p3_calc_bulk_rho_rime(AT(qi, k), &qq, &bb, &rhop);
            AT(qir, k) = qq; AT(qib, k) = bb; }
        int dumi, dumjj, dumii;
        float d1, d4, d5;
        p3_find_lt_1a(AT(qi, k), AT(ni, k), AT(qir, k), rhop,
                      &dumi, &dumjj, &dumii, &d1, &d4, &d5);
        float f1pr02 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 2,
                                              d1, d4, d5);
        float f1pr06 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 6,
                                              d1, d4, d5);
        float f1pr09 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 7,
                                              d1, d4, d5);
        float f1pr10 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 8,
                                              d1, d4, d5);
        float f1pr13 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 9,
                                              d1, d4, d5);
        float f1pr15 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 11,
                                              d1, d4, d5);
        float f1pr16 = p3_access_lookup_table(itab, dumjj, dumii, dumi, 12,
                                              d1, d4, d5);
        AT(ni, k) = fminf(AT(ni, k), f1pr09 * AT(qi, k));
        AT(ni, k) = fmaxf(AT(ni, k), f1pr10 * AT(qi, k));
        if (AT(qir, k) < P3_QSMALL) { AT(qir, k) = 0.0f; AT(qib, k) = 0.0f; }
        AT(D[D_VMI], k) = f1pr02 * AT(S[S_RHOFACI], k);
        AT(D[D_EFFI], k) = f1pr06;
        AT(D[D_DI], k) = f1pr15;
        AT(D[D_RHOPO], k) = f1pr16;
        ze_ice = ze_ice + 0.1892f * f1pr13 * AT(ni, k) * rho;
        ze_ice = fmaxf(ze_ice, 1.0e-22f);
    } else {
        AT(qv, k) = AT(qv, k) + AT(qi, k);
        AT(th, k) = AT(th, k) - invexn * AT(qi, k) * P3_XXLS * P3_INV_CP;
        AT(qi, k) = 0.0f;
        AT(ni, k) = 0.0f;
        AT(qir, k) = 0.0f;
        AT(qib, k) = 0.0f;
        AT(D[D_DI], k) = 0.0f;
    }

    AT(D[D_ZDBZ], k) = 10.0f * p3_log10((ze_rain + ze_ice) * 1.0e18f);

    if (AT(qr, k) < P3_QSMALL) AT(nr, k) = 0.0f;
}

__device__ void p3_step_homofreeze_col(P3_ARGS)
{
    if (FLG[ncol + i] == 0.0f) return;
    // third compute_SCPF (:4521-4523): refresh the qv snapshot.  Nothing in
    // the ported call shape reads it again, but it is where the Sundqvist
    // branch attaches, so the refresh stays.
    for (int k = 0; k < nk; ++k) AT(S[S_QVCLD], k) = AT(F[F_QV], k);
    for (int k = 0; k < nk; ++k) p3_homofreeze_level(P3_PASS, k);
}

__device__ void p3_step_final_col(P3_ARGS)
{
    if (FLG[ncol + i] == 0.0f) return;
    for (int k = 0; k < nk; ++k) p3_final_level(P3_PASS, k);
}

// The fused form of the two above: one k-loop instead of two.  Valid
// because both bodies read and write level k only; the byte gate is what
// proves the compiler agrees.
__device__ void p3_step_homofreeze_final_col(P3_ARGS)
{
    if (FLG[ncol + i] == 0.0f) return;
    for (int k = 0; k < nk; ++k) AT(S[S_QVCLD], k) = AT(F[F_QV], k);
    for (int k = 0; k < nk; ++k) {
        p3_homofreeze_level(P3_PASS, k);
        p3_final_level(P3_PASS, k);
    }
}

// Save end-of-microphysics theta/qv for the next step (:5018-5021).  These
// are the cross-step carriers the diagnosed-ssat branch depends on, and
// they are why gpuwm/io/restart.py serializes th_old/qv_old.
__device__ void p3_step_saveold_col(P3_ARGS)
{
    for (int k = 0; k < nk; ++k) {
        AT(F[F_THOLD], k) = AT(F[F_TH], k);
        AT(F[F_QVOLD], k) = AT(F[F_QV], k);
    }
}

// mp_p3_wrapper_wrf's precipitation conversions (:892-898).
__device__ void p3_step_precip_col(P3_ARGS)
{
    float dum1 = 1000.0f * dt;
    float prt_liq = P[P_PRTLIQ][i];
    float prt_sol = P[P_PRTSOL][i];
    float total = prt_liq + prt_sol;
    P[P_RAINNC][i] = P[P_RAINNC][i] + total * dum1;
    P[P_RAINNCV][i] = total * dum1;
    P[P_SNOWNC][i] = P[P_SNOWNC][i] + prt_sol * dum1;
    P[P_SNOWNCV][i] = prt_sol * dum1;
    P[P_SR][i] = prt_sol / (prt_liq + prt_sol + 1.0e-12f);
}

// =====================================================================
// The two arms.
//
// UNFUSED (9 launches) is the REFERENCE.  One step per kernel, in the
// authority's order.  It is what the per-field agreement against the
// Fortran is measured on, and it is what any fused variant must reproduce
// byte for byte.
//
// FUSED (3 launches) composes the SAME __device__ step functions.  Two of
// its three kernels are pure launch elimination; the third additionally
// merges the homogeneous-freezing and final-diagnostics k-loops into one,
// which halves the loads of eight prognostic fields.  Because both bodies
// are level-local that is arithmetically the same computation -- and
// "arithmetically the same" is a claim the byte gate in
// tests/test_p3_cuda.py checks rather than a claim this comment makes.
// =====================================================================

#define P3_TID \
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= ncol) return

#define P3_KERNEL_ARGS \
    float* const* __restrict__ F, float* const* __restrict__ S, \
    float* const* __restrict__ W, float* const* __restrict__ D, \
    float* const* __restrict__ P, const float* const* __restrict__ T, \
    float* __restrict__ FLG, int ncol, int nk, float dt, int it, \
    int log_predictNc, float clbfact_dep, float clbfact_sub

extern "C" __global__ void p3k_prep(P3_KERNEL_ARGS)
{ P3_TID; p3_step_prep_col(P3_PASS); }

extern "C" __global__ void p3k_kloop1(P3_KERNEL_ARGS)
{ P3_TID; p3_step_kloop1_col(P3_PASS); }

extern "C" __global__ void p3k_kloopmain(P3_KERNEL_ARGS)
{ P3_TID; p3_step_kloopmain_col(P3_PASS); }

extern "C" __global__ void p3k_sed_cloud(P3_KERNEL_ARGS)
{ P3_TID; p3_step_sed_cloud_col(P3_PASS); }

extern "C" __global__ void p3k_sed_rain(P3_KERNEL_ARGS)
{ P3_TID; p3_step_sed_rain_col(P3_PASS); }

extern "C" __global__ void p3k_sed_ice(P3_KERNEL_ARGS)
{ P3_TID; p3_step_sed_ice_col(P3_PASS); }

extern "C" __global__ void p3k_homofreeze(P3_KERNEL_ARGS)
{ P3_TID; p3_step_homofreeze_col(P3_PASS); }

extern "C" __global__ void p3k_final(P3_KERNEL_ARGS)
{ P3_TID; p3_step_final_col(P3_PASS); }

extern "C" __global__ void p3k_saveold_precip(P3_KERNEL_ARGS)
{ P3_TID; p3_step_saveold_col(P3_PASS); p3_step_precip_col(P3_PASS); }

extern "C" __global__ void p3k_fused_process(P3_KERNEL_ARGS)
{
    P3_TID;
    p3_step_prep_col(P3_PASS);
    p3_step_kloop1_col(P3_PASS);
    p3_step_kloopmain_col(P3_PASS);
}

extern "C" __global__ void p3k_fused_sed(P3_KERNEL_ARGS)
{
    P3_TID;
    p3_step_sed_cloud_col(P3_PASS);
    p3_step_sed_rain_col(P3_PASS);
    p3_step_sed_ice_col(P3_PASS);
}

extern "C" __global__ void p3k_fused_finish(P3_KERNEL_ARGS)
{
    P3_TID;
    p3_step_homofreeze_final_col(P3_PASS);
    p3_step_saveold_col(P3_PASS);
    p3_step_precip_col(P3_PASS);
}

// ---------------------------------------------------------------------
// Contraction control-probe.  tests/test_p3_cuda.py compiles the module
// twice and compares these two kernels: under -fmad=false the infix form
// this file is written in must equal the explicit _rn intrinsics bit for
// bit, and under -fmad=true it must NOT.  That is what makes "plain
// operators are safe here" a measurement instead of an assumption.
// ---------------------------------------------------------------------
extern "C" __global__ void p3k_probe_infix(const float* __restrict__ a,
                                           const float* __restrict__ b,
                                           const float* __restrict__ c,
                                           float* __restrict__ o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    o[i] = a[i] * b[i] + c[i];
}

extern "C" __global__ void p3k_probe_rn(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        const float* __restrict__ c,
                                        float* __restrict__ o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    o[i] = __fadd_rn(__fmul_rn(a[i], b[i]), c[i]);
}
