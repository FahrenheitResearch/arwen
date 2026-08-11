// gpuwm/core/kernels/milbrandt2.cu
//
// Milbrandt-Yau double-moment bulk microphysics, WRF mp_physics = 9.
//
// TRANSCRIPTION AUTHORITY (read line by line, never from memory of WRF):
//   C:/Users/drew/Downloads/WRF_1974_MP55_reference_bundle/
//     WRF_source_v4.6.1_group/phys/module_mp_milbrandt2mom.F
//   - my2_mod helper functions           :31-433, :3489-3525
//   - sedi_wrapper_2 / sedi_1D / count_columns  :564-836
//   - mp_milbrandt2mom_main              :841-3485
//   - mp_milbrandt2mom_driver (the WRF-facing 3-D wrapper) :3559-3703
//   - microphysics_driver CASE(MILBRANDT2MOM)  module_microphysics_driver.F:1830-1886
//
// STAGING.  WRF runs one i,k double loop per part.  Every part except
// sedimentation is a pure per-CELL function of the fields the previous part
// left behind, so this file splits the scheme the way morrison.cu does --
// cell-parallel process kernels around one column-parallel sedimentation
// kernel -- rather than giving every thread a 28-array column.  The split
// points are exactly WRF's own part boundaries (:1578, :2830, :3128, :3340),
// so no statement crosses a kernel and the arithmetic order inside each
// statement is the Fortran's.
//
//   milbrandt2_prelim      Part 1  (:1215-1547)   cell
//   milbrandt2_geometry    Part 1  (:1549-1558)   cell (needs pres[k-1])
//   milbrandt2_cold        Part 2  (:1586-2828)   cell
//   milbrandt2_warm        Part 3  (:2836-3126)   cell
//   milbrandt2_sediment_*  Part 4  (:3134-3334)   column
//   milbrandt2_diagnostics        (:3336-3473)    cell
//
// TRANSCENDENTALS ARE IEEE, NOT FAST-MATH.  powf/expf/logf/log10f
// throughout, never the __powf/__expf/__logf/__log10f device intrinsics.
// This file briefly shipped 90 intrinsic sites and was the only
// microphysics kernel in the tree using any (thompson.cu, nssl2.cu,
// morrison.cu, wsm6.cu, kessler.cu and shinhong.cu are all IEEE).
// The lane's adversarial review measured the gap on an RTX 5090 over this
// scheme's own argument ranges and reported up to 85 ULP -- 5.1e-6
// relative on iLAMr**cexr6 at :3056, 36 ULP on the snow-aggregation
// exponent -- which would put a reduced-precision floor under the WRF
// oracle campaign before it starts.  That figure is cited from the
// review, not re-measured here; what this lane measured directly is the
// end-to-end move the swap caused, recorded in the commit that made it.
// The shared loader compiles with
// options=("-std=c++17",) and no --use_fast_math
// (gpuwm/core/kernels/__init__.py), so this is a source-level choice and
// nothing else can reintroduce it.
//
// FIXED SWITCHES.  The WRF wrapper hard-codes CCNtype=2 (continental,
// :3615), precipDiag_ON/sedi_ON/warmphase_ON/autoconv_ON/icephase_ON/
// snow_ON all .true. (:3618-3623) and nk_BOTTOM=.false. (:3591), and the
// scheme body hard-codes snowSpherical=.false., primIceNucl=1, grpl_ON,
// hail_ON, rainAccr_ON, iceDep_ON all .true. (:1168-1176).  Those are the
// only mp=9 identity WRF can run, so they are compiled in here and
// gpuwm/config.py refuses any request to change them.  k=1 is the bottom
// level in WRF's wrapper, which is gpuwm's k=0, so kdir=+1, kbot=0,
// ktop=nz-1 throughout and every Fortran ``k+kdir`` reads ``k+1``.
//
// DIVERGENCES FROM THE FORTRAN, all documented at their site below:
//   1. :2819-2823 -- a live (not DEBUG_ON) ``print``/``stop`` on
//      T<173 K or T>323 K.  A CUDA kernel cannot abort the run, and gpuwm's
//      health gate owns that decision at the step level.
//   2. count_columns :821 would read QX(i,0) when ktop_sedi==kbot; clamped.
//   3. :1733/:1754/:1771 zero ``iLAMsB1``/``iLAMgB1``/``iLAMhB1`` twice and
//      never zero ``iLAMsB2``/``iLAMgB2``/``iLAMhB2``.  Every read of the
//      B2 terms is inside the matching ``Qx>epsQ`` branch that also writes
//      them, so the stale value is unreachable; the defined behaviour
//      (zeroing all three) is implemented.
//   4. RT_snd (:3224-3262) and RT_peL (:3321-3327) are computed by the
//      scheme and DISCARDED by mp_milbrandt2mom_driver -- they are locals
//      of the wrapper with no WRF output binding -- so they are not
//      computed here.  Every rate the driver does consume is.

#define MY2_KMAX_GENERIC 256
#define MY2_KMAX_SHALLOW 64

// --------------------------------------------------------------------------
// The read-only constant vector.  Built once at import by
// gpuwm/core/milbrandt2_constants.py from WRF's :1257-1438 SAVE block; the
// index order is CK_ORDER there and tests/test_milbrandt2.py asserts this
// block still matches ``cuda_define_block()`` verbatim.
// --------------------------------------------------------------------------
#define PI2 ck[0]
#define PIov4 ck[1]
#define PIov6 ck[2]
#define CHLS ck[3]
#define LCP ck[4]
#define LFP ck[5]
#define iCHLF ck[6]
#define LSP ck[7]
#define ck5 ck[8]
#define ck6 ck[9]
#define imgo ck[10]
#define idew ck[11]
#define idei ck[12]
#define ideg ck[13]
#define ideh ck[14]
#define cmr ck[15]
#define icmr ck[16]
#define cmi ck[17]
#define icmi ck[18]
#define cmg ck[19]
#define icmg ck[20]
#define cmh ck[21]
#define icmh ck[22]
#define cms_D3 ck[23]
#define cms ck[24]
#define dms ck[25]
#define icms ck[26]
#define idms ck[27]
#define mso ck[28]
#define imso ck[29]
#define eds ck[30]
#define fds ck[31]
#define GS50 ck[32]
#define iMUc ck[33]
#define GC1 ck[34]
#define iGC1 ck[35]
#define GC2 ck[36]
#define GC3 ck[37]
#define GC4 ck[38]
#define GC11 ck[39]
#define GC12 ck[40]
#define GC5 ck[41]
#define iGC5 ck[42]
#define GC6 ck[43]
#define GC7 ck[44]
#define GC8 ck[45]
#define GC13 ck[46]
#define GC14 ck[47]
#define GC15 ck[48]
#define icexc9 ck[49]
#define N_c_SM ck[50]
#define cexr1 ck[51]
#define cexr2 ck[52]
#define GR17 ck[53]
#define GR31 ck[54]
#define iGR31 ck[55]
#define GR32 ck[56]
#define GR33 ck[57]
#define GR34 ck[58]
#define iGR34 ck[59]
#define GR35 ck[60]
#define GR36 ck[61]
#define GR37 ck[62]
#define GR50 ck[63]
#define cexr5 ck[64]
#define cexr6 ck[65]
#define cexr9 ck[66]
#define icexr9 ck[67]
#define cexr3 ck[68]
#define cexr4 ck[69]
#define ckQr1 ck[70]
#define ckQr2 ck[71]
#define ckQr3 ck[72]
#define GI4 ck[73]
#define GI6 ck[74]
#define GI11 ck[75]
#define GI20 ck[76]
#define GI21 ck[77]
#define GI22 ck[78]
#define GI31 ck[79]
#define iGI31 ck[80]
#define GI32 ck[81]
#define GI33 ck[82]
#define GI34 ck[83]
#define GI35 ck[84]
#define GI36 ck[85]
#define GI40 ck[86]
#define icexi9 ck[87]
#define ckQi1 ck[88]
#define ckQi2 ck[89]
#define ckQi4 ck[90]
#define cexs1 ck[91]
#define cexs2 ck[92]
#define icexs2 ck[93]
#define GS09 ck[94]
#define GS11 ck[95]
#define GS12 ck[96]
#define GS13 ck[97]
#define GS31 ck[98]
#define iGS31 ck[99]
#define GS32 ck[100]
#define GS33 ck[101]
#define GS34 ck[102]
#define iGS34 ck[103]
#define GS35 ck[104]
#define GS36 ck[105]
#define GS40 ck[106]
#define iGS40 ck[107]
#define iGS20 ck[108]
#define ckQs1 ck[109]
#define ckQs2 ck[110]
#define GS40_D3 ck[111]
#define iGS20_D3 ck[112]
#define rfact_FvFm ck[113]
#define GG09 ck[114]
#define GG11 ck[115]
#define GG12 ck[116]
#define GG13 ck[117]
#define GG31 ck[118]
#define iGG31 ck[119]
#define GG32 ck[120]
#define GG33 ck[121]
#define GG34 ck[122]
#define iGG34 ck[123]
#define GG35 ck[124]
#define GG36 ck[125]
#define GG40 ck[126]
#define iGG99 ck[127]
#define GG50 ck[128]
#define ckQg1 ck[129]
#define ckQg2 ck[130]
#define ckQg4 ck[131]
#define GH09 ck[132]
#define GH11 ck[133]
#define GH12 ck[134]
#define GH13 ck[135]
#define GH31 ck[136]
#define iGH31 ck[137]
#define GH32 ck[138]
#define GH33 ck[139]
#define iGH34 ck[140]
#define GH40 ck[141]
#define iGH99 ck[142]
#define GH50 ck[143]
#define ckQh1 ck[144]
#define ckQh2 ck[145]
#define ckQh4 ck[146]
#define cxr ck[147]
#define cxi ck[148]
#define Gzr ck[149]
#define Gzi ck[150]
#define Gzs ck[151]
#define Gzg ck[152]
#define Gzh ck[153]

// --------------------------------------------------------------------------
// Scheme parameters (:1026-1176).  Named exactly as the Fortran except MY2_T0
// (the contact-nucleation reference temperature, :1161) which would collide
// with the kernel preamble's base-state T0.
// --------------------------------------------------------------------------
#define MY2_alpha_c   1.0f
#define MY2_alpha_r   0.0f
#define MY2_alpha_i   0.0f
#define MY2_alpha_s   0.0f
#define MY2_alpha_g   0.0f
#define MY2_alpha_h   0.0f
#define MY2_No_s_max  1.0e+8f
#define MY2_lamdas_min 500.0f
#define MY2_No_r_SM   1.0e+7f
#define MY2_No_g_SM   4.0e+6f
#define MY2_No_h_SM   1.0e+5f
#define MY2_afr 149.100f
#define MY2_bfr 0.5000f
#define MY2_afi 71.340f
#define MY2_bfi 0.6635f
#define MY2_afs 11.720f
#define MY2_bfs 0.4100f
#define MY2_afg 19.300f
#define MY2_bfg 0.3700f
#define MY2_afh 206.890f
#define MY2_bfh 0.6384f
#define MY2_epsQ  1.0e-14f
#define MY2_epsN  1.0e-3f
#define MY2_epsQ2 1.0e-6f
#define MY2_iLAMmin1 1.0e-6f
#define MY2_iLAMmin2 1.0e-10f
#define MY2_deg 400.0f
#define MY2_mgo 1.6e-10f
#define MY2_deh 900.0f
#define MY2_dei 500.0f
#define MY2_mio 1.0e-12f
#define MY2_dew 1000.0f
#define MY2_desFix 100.0f
#define MY2_desMax 500.0f
#define MY2_Dso 125.0e-6f
#define MY2_dmr 3.0f
#define MY2_dmi 3.0f
#define MY2_dmg 3.0f
#define MY2_dmh 3.0f
#define MY2_DrMax 5.0e-3f
#define MY2_VrMax 16.0f
#define MY2_epsQr_sedi 1.0e-8f
#define MY2_DiMax 5.0e-3f
#define MY2_ViMax 2.0f
#define MY2_epsQi_sedi 1.0e-10f
#define MY2_DsMax 5.0e-3f
#define MY2_VsMax 4.0f
#define MY2_epsQs_sedi 1.0e-8f
#define MY2_DgMax 2.0e-3f
#define MY2_VgMax 6.0f
#define MY2_epsQg_sedi 1.0e-8f
#define MY2_DhMax 80.0e-3f
#define MY2_VhMax 25.0f
#define MY2_epsQh_sedi 1.0e-10f
#define MY2_CPW 4218.0f
#define MY2_DEo 1.225f
#define MY2_thrd (1.0f / 3.0f)
#define MY2_Ers 1.0f
#define MY2_Eci 1.0f
#define MY2_Eri 1.0f
#define MY2_Erh 1.0f
#define MY2_Avx 0.78f
#define MY2_Bvx 0.30f
#define MY2_Abigg 0.66f
#define MY2_Bbigg 100.0f
#define MY2_fdielec 4.464f
#define MY2_zfact 1.0e+18f
#define MY2_minZET (-99.0f)
#define MY2_Drshed 0.001f
#define MY2_SIGcTHRS 15.0e-6f
#define MY2_KK1 3.03e3f
#define MY2_KK2 2.59e15f
#define MY2_Dhh 82.0e-6f
#define MY2_zMax_sedi 20000.0f
#define MY2_Dr_large 200.0e-6f
#define MY2_Dh_large 1.0e-2f
#define MY2_Dh_min 1.0e-3f
#define MY2_Dr_3cmpThrs 2.5e-3f
#define MY2_Ngh_crit 1.0e+0f
#define MY2_Tc_FZrh (-10.0f)
#define MY2_CNsgThres 1.0f
#define MY2_capFact_i 0.5f
#define MY2_capFact_s 0.5f
#define MY2_Ni_max 1.0e+7f
#define MY2_satw_peak 1.01f
// WRF + kin_1d thermodynamic constants (:1130-1155).  These are the
// scheme's OWN values and deliberately differ from gpuwm's module
// constants (e.g. GRAV=9.80616 here, not the dycore's g).
#define MY2_CPI 0.21153e+4f
#define MY2_TRPL 0.27316e+3f
#define MY2_PI 0.314159265359e+1f
#define MY2_CHLC 0.2501e+7f
#define MY2_CHLF 0.334e+6f
#define MY2_CPD 0.100546e+4f
#define MY2_RGASD 0.28705e+3f
#define MY2_RGASV 0.46151e+3f
#define MY2_EPS1 0.62194800221014f
#define MY2_GRAV 0.980616e+1f
// Contact-nucleation constants (:1160-1165).
#define MY2_LAMa0 6.6e-8f
#define MY2_T0 293.15f
#define MY2_p0 101325.0f
#define MY2_Ra 1.0e-6f
#define MY2_kBoltz 1.381e-23f
#define MY2_KAPa 5.39e5f
// sedi_1D locals (:637-641)
#define MY2_CoMAX 0.8f

// Fortran intrinsic DIM(x, y) = MAX(x - y, 0).
__device__ __forceinline__ float my2_dim(float x, float y)
{
    float d = x - y;
    return d > 0.0f ? d : 0.0f;
}

// polysvp (:339-407): Flatau et al. (1992) Table 4 right-hand column,
// evaluated in the Fortran's Horner order.  TYPE 1 = ice, 0 = liquid.
__device__ __forceinline__ float my2_polysvp(float t, int wtype)
{
    float dt = t - 273.16f;
    if (dt < -80.0f) dt = -80.0f;
    float p;
    if (wtype == 1) {
        p = 6.11147274f + dt * (0.503160820f + dt * (0.188439774e-1f
            + dt * (0.420895665e-3f + dt * (0.615021634e-5f
            + dt * (0.602588177e-7f + dt * (0.385852041e-9f
            + dt * (0.146898966e-11f + 0.252751365e-14f * dt)))))));
    } else {
        p = 6.11239921f + dt * (0.443987641f + dt * (0.142986287e-1f
            + dt * (0.264847430e-3f + dt * (0.302950461e-5f
            + dt * (0.206739458e-7f + dt * (0.640689451e-10f
            + dt * (-0.952447341e-13f + -0.976195544e-15f * dt)))))));
    }
    return p * 100.0f;
}

// qsat (:410-433)
__device__ __forceinline__ float my2_qsat(float temp, float pres, int wtype)
{
    float e = my2_polysvp(temp, wtype);
    return 0.622f * e / (pres - e);
}

// des_OF_Ds (:3489-3494), Dm_x (:3497-3502), iLAMDA_x (:3505-3510).  All
// three use the source's ``exp(a*log(b))`` IBM optimisation, not powf: the
// substitution is the Fortran's own and changes the last bits.
__device__ __forceinline__ float my2_des_OF_Ds(float Ds, float desMax,
                                               float eds_, float fds_)
{
    float v = eds_ * expf(fds_ * logf(Ds));
    return fminf(desMax, v);
}

__device__ __forceinline__ float my2_Dm_x(float DE, float QX, float iNX,
                                          float icmx, float idmx)
{
    return expf(idmx * logf(DE * QX * iNX * icmx));
}

__device__ __forceinline__ float my2_iLAMDA_x(float DE, float QX, float iNX,
                                              float icex, float idmx)
{
    return expf(idmx * logf(DE * QX * iNX * icex));
}

// N_Cooper (:3513-3518), Nos_Thompson (:3520-3525)
__device__ __forceinline__ float my2_N_Cooper(float T)
{
    return 5.0f * expf(0.304f * (MY2_TRPL - fmaxf(233.0f, T)));
}

__device__ __forceinline__ float my2_Nos_Thompson(float T)
{
    return fminf(2.0e+8f,
                 2.0e+6f * expf(-0.12f * fminf(-0.001f, T - MY2_TRPL)));
}

// NccnFNC (:31-86), CCNtype == 2 (continental) only.  The WRF wrapper
// hard-codes CCNtype=2 at :3615 and the maritime/polluted branches would
// need a different N_c_SM in the constant table, so gpuwm pins the
// continental identity and config.py refuses the others by name.
__device__ __forceinline__ float my2_NccnFNC(float Win, float Tin, float Pin)
{
    float x = log10f(Win * 100.0f);
    float x2 = x * x, x3 = x2 * x, x4 = x2 * x2;
    float T = Tin - 273.15f;
    float T2 = T * T, T3 = T2 * T, T4 = T2 * T2;
    float p = Pin * 0.01f;
    float p2 = p * p;
    float a = 0.0f;
    float b = 0.0f;
    float c = -2.112e-9f * T4 + 3.9836e-8f * T3 + 2.3703e-6f * T2
              - 1.4542e-4f * T - 0.0698f;
    float d = -4.210e-8f * T4 + 5.5745e-7f * T3 + 1.8460e-5f * T2
              + 9.6078e-4f * T + 0.7120f;
    float e = 1.434e-7f * T4 - 1.6455e-6f * T3 - 4.3334e-5f * T2
              - 7.6720e-3f * T + 1.0056f;
    float f = 1.340e-6f * p2 - 3.5114e-3f * p + 1.9453f;
    float g = 4.226e-3f * x4 - 5.6012e-3f * x3 - 8.7846e-2f * x2
              + 2.7435e-2f * x + 0.9932f;
    float h = 5.811e-9f * T4 + 1.5589e-7f * T3 - 3.8623e-5f * T2
              + 1.4471e-3f * T + 0.1496f;
    float y = a * x4 + b * x3 + c * x2 + d * x + e + (f * g * h);
    return powf(10.0f, fmaxf(0.0f, y)) * 1.0e6f;
}

// SxFNC (:89-157) with WRT=1 (peak supersaturation w.r.t. water), CCNtype 2.
// The caller passes Tc (Celsius) as Tin -- :2157 -- and the continental fit
// is stated for -35 < T < -5 C, so that is the intended argument.
__device__ __forceinline__ float my2_SxFNC_w(float Win, float Tin, float Pin)
{
    float x = log10f(fmaxf(Win, 1.0e-20f) * 100.0f);
    float x2 = x * x, x3 = x2 * x, x4 = x2 * x2;
    float T = Tin;
    float T2 = T * T;
    float p = Pin * 0.01f;
    float p2 = p * p;
    float a = 3.80e-5f * T2 + 1.65e-4f * T + 9.88e-2f;
    float b = -7.38e-5f * T2 - 2.53e-3f * T - 3.23e-1f;
    float c = 8.39e-5f * T2 + 3.96e-3f * T + 3.50e-1f;
    float d = -1.88e-6f * T2 - 1.33e-3f * T - 3.73e-2f;
    float f = -1.9761e-6f * p2 + 4.1473e-3f * p - 1.771e0f;
    float g = 0.1539f * x4 - 0.5575f * x3 + 0.9262f * x2 - 0.3498f * x
              - 0.1293f;
    float h = -8.035e-9f * (T2 * T2) + 3.162e-7f * (T2 * T) + 1.029e-5f * T2
              - 5.931e-4f * T + 5.62e-2f;
    float Pcorr = f * g * h;
    float Sw = (a * x3 + b * x2 + c * x + d) + Pcorr;
    Sw = 1.0f + 0.01f * Sw;
    if (Win <= 0.0f) return 1.0f;          // :155
    return Sw;
}

// ==========================================================================
// PART 1 -- preliminary calculations (:1215-1547)
// ==========================================================================
extern "C" __global__
void milbrandt2_prelim(
        float* __restrict__ T, float* __restrict__ Q,
        float* __restrict__ QC, float* __restrict__ QR,
        float* __restrict__ QI, float* __restrict__ QN,
        float* __restrict__ QG, float* __restrict__ QH,
        float* __restrict__ NC, float* __restrict__ NR,
        float* __restrict__ NY, float* __restrict__ NN,
        float* __restrict__ NG, float* __restrict__ NH,
        const float* __restrict__ p_eos,
        const float* __restrict__ psfc,
        float* __restrict__ pres, float* __restrict__ DE,
        float* __restrict__ iDE, float* __restrict__ gamfact,
        float* __restrict__ QSW, float* __restrict__ QSI,
        float* __restrict__ QC_in, float* __restrict__ QR_in,
        float* __restrict__ NC_in, float* __restrict__ NR_in,
        const float* __restrict__ ck,
        int nz, int ny, int nx)
{
    long long n = (long long)nz * ny * nx;
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    int col = (int)(gid % ((long long)ny * nx));

    // The wrapper builds sigma = p/p_sfc (:3648) and the scheme rebuilds
    // pres = PS*sigma (:1216).  The round trip is NOT the identity in FP32
    // and this is the pressure every later statement uses, so it is
    // reproduced rather than short-circuited to p.
    float ps = psfc[col];
    float sigma = p_eos[gid] / ps;
    float pr = ps * sigma;
    pres[gid] = pr;

    float t = T[gid];
    float qsw = my2_qsat(t, pr, 0);        // :1218
    float qsi = my2_qsat(t, pr, 1);        // :1219
    QSW[gid] = qsw;
    QSI[gid] = qsi;

    float de = pr / (MY2_RGASD * t);       // :1224
    float ide = 1.0f / de;                 // :1225
    DE[gid] = de;
    iDE[gid] = ide;

    // :1228-1233 -- N from #/kg to #/m3
    float nc = NC[gid] * de;
    float nr = NR[gid] * de;
    float ny_ = NY[gid] * de;
    float nn = NN[gid] * de;
    float ng = NG[gid] * de;
    float nh = NH[gid] * de;

    float q = Q[gid];
    float qc = QC[gid], qr = QR[gid], qi = QI[gid];
    float qn = QN[gid], qg = QG[gid], qh = QH[gid];

    // --- Ensure consistency between moments (:1445-1531) ---
    float tmp1 = qsw / fmaxf(q, 1.0e-20f);
    float tmp2 = qsi / fmaxf(q, 1.0e-20f);
    float tmp3;

    // cloud (:1459-1467)
    if (qc > MY2_epsQ && nc < MY2_epsN) {
        nc = N_c_SM;
    } else if (qc <= MY2_epsQ || (qc < MY2_epsQ2 && tmp1 < 0.90f)) {
        tmp3 = fmaxf(0.0f, qc);
        q = q + tmp3;
        t = t - LCP * tmp3;
        qc = 0.0f;
        nc = 0.0f;
    }
    // rain (:1470-1479)
    if (qr > MY2_epsQ && nr < MY2_epsN) {
        nr = powf(MY2_No_r_SM * GR31, 3.0f / (4.0f + MY2_alpha_r))
             * powf(GR31 * iGR34 * de * qr * icmr,
                      (1.0f + MY2_alpha_r) / (4.0f + MY2_alpha_r));
    } else if (qr <= MY2_epsQ || (qr < MY2_epsQ2 && tmp1 < 0.90f)) {
        tmp3 = fmaxf(0.0f, qr);
        q = q + tmp3;
        t = t - LCP * tmp3;
        qr = 0.0f;
        nr = 0.0f;
    }
    // ice (:1482-1491)
    if (qi > MY2_epsQ && ny_ < MY2_epsN) {
        ny_ = fmaxf(2.0f * MY2_epsN, my2_N_Cooper(t));
    } else if (qi <= MY2_epsQ || (qi < MY2_epsQ2 && tmp2 < 0.80f)) {
        tmp3 = fmaxf(0.0f, qi);
        q = q + tmp3;
        t = t - LSP * tmp3;
        qi = 0.0f;
        ny_ = 0.0f;
    }
    // snow (:1494-1504)
    if (qn > MY2_epsQ && nn < MY2_epsN) {
        float No_s = my2_Nos_Thompson(t);
        nn = powf(No_s * GS31, dms * icexs2)
             * powf(GS31 * iGS40 * icms * de * qn,
                      (1.0f + MY2_alpha_s) * icexs2);
    } else if (qn <= MY2_epsQ || (qn < MY2_epsQ2 && tmp2 < 0.80f)) {
        tmp3 = fmaxf(0.0f, qn);
        q = q + tmp3;
        t = t - LSP * tmp3;
        qn = 0.0f;
        nn = 0.0f;
    }
    // graupel (:1507-1516)
    if (qg > MY2_epsQ && ng < MY2_epsN) {
        ng = powf(MY2_No_g_SM * GG31, 3.0f / (4.0f + MY2_alpha_g))
             * powf(GG31 * iGG34 * de * qg * icmg,
                      (1.0f + MY2_alpha_g) / (4.0f + MY2_alpha_g));
    } else if (qg <= MY2_epsQ || (qg < MY2_epsQ2 && tmp2 < 0.80f)) {
        tmp3 = fmaxf(0.0f, qg);
        q = q + tmp3;
        t = t - LSP * tmp3;
        qg = 0.0f;
        ng = 0.0f;
    }
    // hail (:1519-1528)
    if (qh > MY2_epsQ && nh < MY2_epsN) {
        nh = powf(MY2_No_h_SM * GH31, 3.0f / (4.0f + MY2_alpha_h))
             * powf(GH31 * iGH34 * de * qh * icmh,
                      (1.0f + MY2_alpha_h) / (4.0f + MY2_alpha_h));
    } else if (qh <= MY2_epsQ || (qh < MY2_epsQ2 && tmp2 < 0.80f)) {
        tmp3 = fmaxf(0.0f, qh);
        q = q + tmp3;
        t = t - LSP * tmp3;
        qh = 0.0f;
        nh = 0.0f;
    }

    T[gid] = t;   Q[gid] = q;
    QC[gid] = qc; QR[gid] = qr; QI[gid] = qi;
    QN[gid] = qn; QG[gid] = qg; QH[gid] = qh;
    NC[gid] = nc; NR[gid] = nr; NY[gid] = ny_;
    NN[gid] = nn; NG[gid] = ng; NH[gid] = nh;

    // :1538-1541 -- time-(t*) copies for the Part 3a coalescence equations,
    // taken AFTER the consistency clip.
    QC_in[gid] = qc;
    QR_in[gid] = qr;
    NC_in[gid] = nc;
    NR_in[gid] = nr;

    // :1546 -- air-density factor for fall speeds
    gamfact[gid] = sqrtf(MY2_DEo / de);
}

// --------------------------------------------------------------------------
// Part 1 layer geometry (:1549-1558).  Split out because iDP reads pres at
// k-1.  DZ/iDZ are frozen here on the PART 1 density and are NOT refreshed
// when Part 3b recomputes DE (:3046) -- that asymmetry is the Fortran's.
// --------------------------------------------------------------------------
extern "C" __global__
void milbrandt2_geometry(
        const float* __restrict__ pres, const float* __restrict__ psfc,
        const float* __restrict__ DE,
        float* __restrict__ DZ, float* __restrict__ iDZ,
        int nz, int ny, int nx)
{
    long long n = (long long)nz * ny * nx;
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    long long plane = (long long)ny * nx;
    int k = (int)(gid / plane);
    int col = (int)(gid % plane);

    float iDP;
    if (k == 0) {
        iDP = 1.0f / (psfc[col] - pres[gid]);              // :1550
    } else {
        iDP = 1.0f / (pres[gid - plane] - pres[gid]);      // :1552
    }
    float idz = DE[gid] * MY2_GRAV * iDP;                  // :1557
    iDZ[gid] = idz;
    DZ[gid] = 1.0f / idz;                                  // :1558
}

// ==========================================================================
// PART 2 -- cold (ice-phase) microphysics (:1586-2828)
// ==========================================================================
extern "C" __global__
void milbrandt2_cold(
        float* __restrict__ T, float* __restrict__ Q,
        float* __restrict__ QC, float* __restrict__ QR,
        float* __restrict__ QI, float* __restrict__ QN,
        float* __restrict__ QG, float* __restrict__ QH,
        float* __restrict__ NC, float* __restrict__ NR,
        float* __restrict__ NY, float* __restrict__ NN,
        float* __restrict__ NG, float* __restrict__ NH,
        const float* __restrict__ WZ,
        const float* __restrict__ pres, const float* __restrict__ DE,
        const float* __restrict__ iDE, const float* __restrict__ gamfact,
        const float* __restrict__ QSW, const float* __restrict__ QSI,
        const float* __restrict__ ck, float dt,
        int nz, int ny, int nx)
{
    long long n = (long long)nz * ny * nx;
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    long long plane = (long long)ny * nx;
    int k = (int)(gid / plane);
    // :1592/:1610 -- the active-point and process loops run
    // ``k = ktop-kdir, kbot, -kdir``: the TOP level is excluded.
    if (k >= nz - 1) return;

    float t = T[gid], q = Q[gid];
    float qc = QC[gid], qr = QR[gid], qi = QI[gid];
    float qn = QN[gid], qg = QG[gid], qh = QH[gid];
    float nc = NC[gid], nr = NR[gid], ny_ = NY[gid];
    float nn = NN[gid], ng = NG[gid], nh = NH[gid];
    float qsw = QSW[gid], qsi = QSI[gid];
    float de = DE[gid], ide = iDE[gid], pr = pres[gid];
    float wz = WZ[gid];

    // :1594-1600 -- active grid points
    bool log1 = (qi + qg + qn + qh) < MY2_epsQ;
    bool log2 = (qc + qr) < MY2_epsQ;
    bool log3 = (t > MY2_TRPL) && log1;
    bool log4 = log1 && log2 && (q < qsi);
    if (log3 || log4) return;                            // icephase_ON = .T.

    float Tc = t - MY2_TRPL;                             // :1614
    // :1615-1618 -- WRF prints a warning for |Tc| outside [-120, 50] and does
    // not act on it (the ``stop`` is commented out).  Nothing to reproduce.
    float Cdiff = (2.2157e-5f + 0.0155e-5f * Tc) * 1.0e5f / pr;
    float MUdyn = 1.72e-5f * (393.0f / (t + 120.0f))
                  * powf(t / MY2_TRPL, 1.5f);
    float MUkin = MUdyn * ide;
    float iMUkin = 1.0f / MUkin;
    float ScTHRD = powf(MUkin / Cdiff, MY2_thrd);
    float Ka = 2.3971e-2f + 0.0078e-2f * Tc;
    float Kdiff = (9.1018e-11f * t * t + 8.8197e-8f * t - (1.0654e-5f));
    float gam = gamfact[gid];

    // :1629-1634 -- collection efficiencies
    float Eis = fminf(0.05f * expf(0.1f * Tc), 1.0f);
    float Eig = fminf(0.01f * expf(0.1f * Tc), 1.0f);
    float Eii = 0.1f * Eis;
    float Ess = Eis, Eih = Eig, Esh = Eig;
    float iEih = 1.0f / Eih;
    float iEsh = 1.0f / Esh;
    (void)Eii;

    float qvs0 = my2_qsat(MY2_TRPL, pr, 0);              // :1641
    float DELqvs = qvs0 - q;                             // :1642

    float tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp8, tmp9, tmp10;
    float iQC = 0.0f, iQR = 0.0f, iQI = 0.0f, iQN = 0.0f, iQG = 0.0f;
    float iQH = 0.0f;
    float iNC = 0.0f, iNR = 0.0f, iNY = 0.0f, iNN = 0.0f, iNG = 0.0f;
    float iNH = 0.0f;

    // Cloud (:1645-1659)
    float Dc, iLAMc, iLAMc2, iLAMc3, iLAMc4, iLAMc5;
    if (qc > MY2_epsQ) {
        iQC = 1.0f / qc;
        iNC = 1.0f / nc;
        Dc = my2_Dm_x(de, qc, iNC, icmr, MY2_thrd);
        iLAMc = my2_iLAMDA_x(de, qc, iNC, icexc9, MY2_thrd);
        iLAMc2 = iLAMc * iLAMc;
        iLAMc3 = iLAMc2 * iLAMc;
        iLAMc4 = iLAMc2 * iLAMc2;
        iLAMc5 = iLAMc3 * iLAMc2;
    } else {
        Dc = 0.0f; iLAMc3 = 0.0f;
        iLAMc = 0.0f; iLAMc4 = 0.0f;
        iLAMc2 = 0.0f; iLAMc5 = 0.0f;
    }

    // Rain (:1662-1680)
    float Dr, iLAMr, iLAMr2, iLAMr3, iLAMr4, iLAMr5, vr0;
    if (qr > MY2_epsQ) {
        iQR = 1.0f / qr;
        iNR = 1.0f / nr;
        Dr = my2_Dm_x(de, qr, iNR, icmr, MY2_thrd);
        iLAMr = fmaxf(MY2_iLAMmin1,
                      my2_iLAMDA_x(de, qr, iNR, icexr9, MY2_thrd));
        iLAMr2 = iLAMr * iLAMr;
        iLAMr3 = iLAMr2 * iLAMr;
        iLAMr4 = iLAMr2 * iLAMr2;
        iLAMr5 = iLAMr4 * iLAMr;
        vr0 = (Dr > 40.0e-6f)
              ? gam * ckQr1 * powf(iLAMr, MY2_bfr) : 0.0f;
    } else {
        iLAMr = 0.0f; Dr = 0.0f; vr0 = 0.0f;
        iLAMr2 = 0.0f; iLAMr3 = 0.0f; iLAMr4 = 0.0f; iLAMr5 = 0.0f;
    }

    // Ice (:1683-1700)
    float Di, iLAMi, iLAMi2, iLAMi3, iLAMi4, iLAMi5;
    float iLAMiB0, iLAMiB1, iLAMiB2, vi0;
    if (qi > MY2_epsQ) {
        iQI = 1.0f / qi;
        iNY = 1.0f / ny_;
        iLAMi = fmaxf(MY2_iLAMmin2,
                      my2_iLAMDA_x(de, qi, iNY, icexi9, MY2_thrd));
        iLAMi2 = iLAMi * iLAMi;
        iLAMi3 = iLAMi2 * iLAMi;
        iLAMi4 = iLAMi2 * iLAMi2;
        iLAMi5 = iLAMi4 * iLAMi;
        iLAMiB0 = powf(iLAMi, MY2_bfi);
        iLAMiB1 = powf(iLAMi, MY2_bfi + 1.0f);
        iLAMiB2 = powf(iLAMi, MY2_bfi + 2.0f);
        vi0 = gam * ckQi1 * iLAMiB0;
        Di = my2_Dm_x(de, qi, iNY, icmi, MY2_thrd);
    } else {
        iLAMi = 0.0f; vi0 = 0.0f; Di = 0.0f;
        iLAMi2 = 0.0f; iLAMi3 = 0.0f; iLAMi4 = 0.0f; iLAMi5 = 0.0f;
        iLAMiB0 = 0.0f; iLAMiB1 = 0.0f; iLAMiB2 = 0.0f;
    }
    (void)iLAMiB1; (void)iLAMiB2;

    // Snow (:1703-1736)
    float Ds, iLAMs, iLAMs2, iLAMs_D3, iLAMsB0, iLAMsB1, iLAMsB2, vs0;
    float des, No_s = 0.0f, VENTs = 0.0f;
    if (qn > MY2_epsQ) {
        iQN = 1.0f / qn;
        iNN = 1.0f / nn;
        iLAMs = fmaxf(MY2_iLAMmin2,
                      my2_iLAMDA_x(de, qn, iNN, iGS20, idms));
        iLAMs_D3 = fmaxf(MY2_iLAMmin2,
                         my2_iLAMDA_x(de, qn, iNN, iGS20_D3, MY2_thrd));
        iLAMs2 = iLAMs * iLAMs;
        iLAMsB0 = powf(iLAMs, MY2_bfs);
        iLAMsB1 = powf(iLAMs, MY2_bfs + 1.0f);
        iLAMsB2 = powf(iLAMs, MY2_bfs + 2.0f);
        vs0 = gam * ckQs1 * iLAMsB0;
        Ds = fminf(MY2_DsMax, my2_Dm_x(de, qn, iNN, icms, idms));
        // snowSpherical = .false. (:1174) -> :1717
        des = my2_des_OF_Ds(Ds, MY2_desMax, eds, fds);
        No_s = nn * iGS31 / iLAMs_D3;                       // :1728
        VENTs = MY2_Avx * GS32 * (iLAMs_D3 * iLAMs_D3)
                + MY2_Bvx * ScTHRD * sqrtf(gam * MY2_afs * iMUkin) * GS09
                  * powf(iLAMs_D3, cexs1);                // :1729
    } else {
        iLAMs = 0.0f; vs0 = 0.0f; Ds = 0.0f; iLAMs2 = 0.0f;
        iLAMs_D3 = 0.0f;
        // DIVERGENCE 3: the Fortran :1733 writes iLAMsB1 twice and never
        // zeroes iLAMsB2.  Every iLAMsB2 read is inside the Qs>epsQ arm
        // that also writes it, so the stale value is unreachable; the
        // defined behaviour is implemented.
        iLAMsB0 = 0.0f; iLAMsB1 = 0.0f; iLAMsB2 = 0.0f;
        des = MY2_desFix;                                   // :1734
    }
    float ides = 1.0f / des;                                // :1736
    (void)ides;

    // Graupel (:1740-1755)
    float Dg, iLAMg, iLAMg2, iLAMgB0, iLAMgB1, iLAMgB2, vg0, No_g;
    if (qg > MY2_epsQ) {
        iQG = 1.0f / qg;
        iNG = 1.0f / ng;
        iLAMg = fmaxf(MY2_iLAMmin1,
                      my2_iLAMDA_x(de, qg, iNG, iGG99, MY2_thrd));
        iLAMg2 = iLAMg * iLAMg;
        iLAMgB0 = powf(iLAMg, MY2_bfg);
        iLAMgB1 = powf(iLAMg, MY2_bfg + 1.0f);
        iLAMgB2 = powf(iLAMg, MY2_bfg + 2.0f);
        No_g = ng * iGG31 / iLAMg;                          // :1749
        vg0 = gam * ckQg1 * iLAMgB0;
        Dg = my2_Dm_x(de, qg, iNG, icmg, MY2_thrd);
    } else {
        iLAMg = 0.0f; vg0 = 0.0f; Dg = 0.0f; No_g = 0.0f;
        iLAMg2 = 0.0f; iLAMgB0 = 0.0f; iLAMgB1 = 0.0f; iLAMgB2 = 0.0f;
    }

    // Hail (:1758-1772)
    float Dh, iLAMh, iLAMh2, iLAMhB0, iLAMhB1, iLAMhB2, vh0, No_h;
    if (qh > MY2_epsQ) {
        iQH = 1.0f / qh;
        iNH = 1.0f / nh;
        iLAMh = fmaxf(MY2_iLAMmin1,
                      my2_iLAMDA_x(de, qh, iNH, iGH99, MY2_thrd));
        iLAMh2 = iLAMh * iLAMh;
        iLAMhB0 = powf(iLAMh, MY2_bfh);
        iLAMhB1 = powf(iLAMh, MY2_bfh + 1.0f);
        iLAMhB2 = powf(iLAMh, MY2_bfh + 2.0f);
        No_h = nh * iGH31 / powf(iLAMh, 1.0f + MY2_alpha_h);   // :1766
        vh0 = gam * ckQh1 * iLAMhB0;
        Dh = my2_Dm_x(de, qh, iNH, icmh, MY2_thrd);
    } else {
        iLAMh = 0.0f; vh0 = 0.0f; Dh = 0.0f; No_h = 0.0f;
        iLAMh2 = 0.0f; iLAMhB0 = 0.0f; iLAMhB1 = 0.0f; iLAMhB2 = 0.0f;
    }
    (void)iQC; (void)iQH; (void)iNH;

    // :1778-1794 -- initialise every source/sink term
    float QNUvi = 0.f, QVDvi = 0.f, QVDvs = 0.f, QVDvg = 0.f, QVDvh = 0.f;
    float QCLcs = 0.f, QCLcg = 0.f, QCLch = 0.f, QFZci = 0.f, QCLri = 0.f;
    float QMLsr = 0.f, QCLrs = 0.f, QCLrg = 0.f, QMLgr = 0.f, QCLrh = 0.f;
    float QMLhr = 0.f, QFZrh = 0.f, QMLir = 0.f, QCLsr = 0.f, QCLsh = 0.f;
    float QCLgr = 0.f, QCNgh = 0.f, QCNis = 0.f, QCLir = 0.f, QCLis = 0.f;
    float QCLih = 0.f, QIMsi = 0.f, QIMgi = 0.f, QCNsg = 0.f, QHwet = 0.f;
    float QCLig = 0.f, QSHhr = 0.f;

    float NCLcs = 0.f, NCLcg = 0.f, NCLch = 0.f, NFZci = 0.f, NMLhr = 0.f;
    float NhCNgh = 0.f, NCLri = 0.f, NCLrs = 0.f, NCLrg = 0.f, NCLrh = 0.f;
    float NMLsr = 0.f, NMLgr = 0.f, NMLir = 0.f, NSHhr = 0.f, NNUvi = 0.f;
    float NVDvi = 0.f, NVDvh = 0.f, NCLir = 0.f, NCLis = 0.f, NCLig = 0.f;
    float NCLih = 0.f, NIMsi = 0.f, NIMgi = 0.f, NiCNis = 0.f;
    float NsCNis = 0.f, NVDvs = 0.f, NCNsg = 0.f, NCLgr = 0.f;
    float NCLsrh = 0.f, NCLss = 0.f, NCLsr = 0.f, NCLsh = 0.f;
    float NCLsrs = 0.f, NCLgrg = 0.f, NgCNgh = 0.f, NVDvg = 0.f;
    float NCLirg = 0.f, NCLsrg = 0.f, NCLgrh = 0.f, NrFZrh = 0.f;
    float NhFZrh = 0.f, NCLirh = 0.f;
    float QCNis1 = 0.f, QCNis2 = 0.f;
    (void)QCNis1; (void)QCNis2;

    float Dirg = 0.f, Dirh = 0.f, Dsrs = 0.f, Dsrg = 0.f, Dsrh = 0.f;
    float Dgrg = 0.f, Dgrh = 0.f;                            // :1794

    float Si = q / qsi;                                      // :1798
    float iABi = 1.0f / (CHLS * CHLS / (Ka * MY2_RGASV * t * t)
                         + 1.0f / (de * qsi * Cdiff));       // :1799
    float VDmax;

    // ---- Collection by SNOW (:1805-1866) ----
    if (qn > MY2_epsQ) {
        if (qc > MY2_epsQ) {
            float Ecs = fminf(Dc, 30.0e-6f) * 3.333e+4f
                        * sqrtf(fminf(Ds, 1.0e-3f) * 1.0e+3f);
            QCLcs = dt * gam * MY2_afs * cmr * Ecs * PIov4 * ide
                    * (nc * nn) * iGC5 * iGS31
                    * (GC13 * GS13 * iLAMc3 * iLAMsB2
                       + 2.0f * GC14 * GS12 * iLAMc4 * iLAMsB1
                       + GC15 * GS11 * iLAMc5 * iLAMsB0);
            NCLcs = dt * gam * MY2_afs * PIov4 * Ecs * (nc * nn)
                    * iGC5 * iGS31
                    * (GC5 * GS13 * iLAMsB2
                       + 2.0f * GC11 * GS12 * iLAMc * iLAMsB1
                       + GC12 * GS11 * iLAMc2 * iLAMsB0);
            // snowSpherical = .false. -> :1826-1830
            tmp1 = 0.6366f;
            QCLcs = tmp1 * QCLcs;
            NCLcs = tmp1 * NCLcs;
            QCLcs = fminf(QCLcs, qc);
            NCLcs = fminf(NCLcs, nc);
        } else {
            QCLcs = 0.0f; NCLcs = 0.0f;
        }
        if (qi > MY2_epsQ) {
            tmp1 = vs0 - vi0;
            tmp3 = sqrtf(tmp1 * tmp1 + 0.04f * vs0 * vi0);
            QCLis = dt * cmi * ide * MY2_PI * 6.0f * Eis * (ny_ * nn) * tmp3
                    * iGI31 * iGS31
                    * (0.5f * iLAMs2 * iLAMi3 + 2.0f * iLAMs * iLAMi4
                       + 5.0f * iLAMi5);
            NCLis = dt * PIov4 * Eis * (ny_ * nn) * GI31 * GS31 * tmp3
                    * (GI33 * GS31 * iLAMi2
                       + 2.0f * GI32 * GS32 * iLAMi * iLAMs
                       + GI31 * GS33 * iLAMs2);
            QCLis = fminf(QCLis, qi);
            NCLis = fminf(QCLis * (ny_ * iQI), NCLis);
        } else {
            QCLis = 0.0f; NCLis = 0.0f;
        }
        // snow self-collection / aggregation (:1856-1862)
        NCLss = dt * 0.93952f * Ess
                * powf(de * qn, (2.0f + MY2_bfs) * MY2_thrd)
                * powf(nn, (4.0f - MY2_bfs) * MY2_thrd);
        NCLss = fminf(NCLss, 0.5f * nn);
    } else {
        QCLcs = 0.0f; NCLcs = 0.0f; QCLis = 0.0f; NCLis = 0.0f;
        NCLss = 0.0f;
    }

    // ---- Collection by GRAUPEL (:1869-1928) ----
    float VENTg = 0.0f;
    if (qg > MY2_epsQ) {
        if (qc > MY2_epsQ) {
            float Kstoke = MY2_dew * vg0 * Dc * Dc / (9.0f * MUdyn * Dg);
            Kstoke = fmaxf(1.5f, fminf(10.0f, Kstoke));
            float Ecg = 0.55f * log10f(2.51f * Kstoke);
            QCLcg = dt * gam * MY2_afg * cmr * Ecg * PIov4 * ide
                    * (nc * ng) * iGC5 * iGG31
                    * (GC13 * GG13 * iLAMc3 * iLAMgB2
                       + 2.0f * GC14 * GG12 * iLAMc4 * iLAMgB1
                       + GC15 * GG11 * iLAMc5 * iLAMgB0);
            NCLcg = dt * gam * MY2_afg * PIov4 * Ecg * (nc * ng)
                    * iGC5 * iGG31
                    * (GC5 * GG13 * iLAMgB2
                       + 2.0f * GC11 * GG12 * iLAMc * iLAMgB1
                       + GC12 * GG11 * iLAMc2 * iLAMgB0);
            QCLcg = fminf(QCLcg, qc);
            NCLcg = fminf(NCLcg, nc);
        } else {
            QCLcg = 0.0f; NCLcg = 0.0f;
        }
        if (qi > MY2_epsQ) {
            tmp1 = vg0 - vi0;
            tmp3 = sqrtf(tmp1 * tmp1 + 0.04f * vg0 * vi0);
            QCLig = dt * cmi * ide * MY2_PI * 6.0f * Eig * (ny_ * ng) * tmp3
                    * iGI31 * iGG31
                    * (0.5f * iLAMg2 * iLAMi3 + 2.0f * iLAMg * iLAMi4
                       + 5.0f * iLAMi5);
            NCLig = dt * PIov4 * Eig * (ny_ * ng) * GI31 * GG31 * tmp3
                    * (GI33 * GG31 * iLAMi2
                       + 2.0f * GI32 * GG32 * iLAMi * iLAMg
                       + GI31 * GG33 * iLAMg2);
            QCLig = fminf(QCLig, qi);
            NCLig = fminf(QCLig * (ny_ * iQI), NCLig);
        } else {
            QCLig = 0.0f; NCLig = 0.0f;
        }
        VENTg = MY2_Avx * GG32 * iLAMg * iLAMg
                + MY2_Bvx * ScTHRD * sqrtf(gam * MY2_afg * iMUkin) * GG09
                  * powf(iLAMg, 2.5f + 0.5f * MY2_bfg + MY2_alpha_g);
        QVDvg = dt * ide * iABi * (PI2 * (Si - 1.0f) * No_g * VENTg);
        VDmax = (q - qsi) / (1.0f + ck6 * qsi / ((t - 7.66f) * (t - 7.66f)));
        if (Si >= 1.0f) {
            QVDvg = fminf(fmaxf(QVDvg, 0.0f), VDmax);
        } else {
            if (VDmax < 0.0f) QVDvg = fmaxf(QVDvg, VDmax);
        }
        NVDvg = 0.0f;                                        // :1923
    } else {
        QCLcg = 0.0f; QCLrg = 0.0f; QCLig = 0.0f;
        NCLcg = 0.0f; NCLrg = 0.0f; NCLig = 0.0f;
    }

    // ---- Collection by HAIL (:1931-2032) ----
    float VENTh = 0.0f;
    if (qh > MY2_epsQ) {
        if (qc > MY2_epsQ) {
            float Ech = expf(-8.68e-7f * powf(Dc, -1.6f) * Dh);
            QCLch = dt * gam * MY2_afh * cmr * Ech * PIov4 * ide
                    * (nc * nh) * iGC5 * iGH31
                    * (GC13 * GH13 * iLAMc3 * iLAMhB2
                       + 2.0f * GC14 * GH12 * iLAMc4 * iLAMhB1
                       + GC15 * GH11 * iLAMc5 * iLAMhB0);
            NCLch = dt * gam * MY2_afh * PIov4 * Ech * (nc * nh)
                    * iGC5 * iGH31
                    * (GC5 * GH13 * iLAMhB2
                       + 2.0f * GC11 * GH12 * iLAMc * iLAMhB1
                       + GC12 * GH11 * iLAMc2 * iLAMhB0);
            QCLch = fminf(QCLch, qc);
            NCLch = fminf(NCLch, nc);
        } else {
            QCLch = 0.0f; NCLch = 0.0f;
        }
        if (qr > MY2_epsQ) {
            tmp1 = vh0 - vr0;
            tmp3 = sqrtf(tmp1 * tmp1 + 0.04f * vh0 * vr0);
            QCLrh = dt * cmr * MY2_Erh * PIov4 * ide * (nh * nr)
                    * iGR31 * iGH31 * tmp3
                    * (GR36 * GH31 * iLAMr5
                       + 2.0f * GR35 * GH32 * iLAMr4 * iLAMh
                       + GR34 * GH33 * iLAMr3 * iLAMh2);
            NCLrh = dt * PIov4 * MY2_Erh * (nh * nr) * iGR31 * iGH31 * tmp3
                    * (GR33 * GH31 * iLAMr2
                       + 2.0f * GR32 * GH32 * iLAMr * iLAMh
                       + GR31 * GH33 * iLAMh2);
            QCLrh = fminf(QCLrh, qr);
            NCLrh = fminf(NCLrh, QCLrh * (nr * iQR));
        } else {
            QCLrh = 0.0f; NCLrh = 0.0f;
        }
        if (qi > MY2_epsQ) {
            tmp1 = vh0 - vi0;
            tmp3 = sqrtf(tmp1 * tmp1 + 0.04f * vh0 * vi0);
            QCLih = dt * cmi * ide * MY2_PI * 6.0f * Eih * (ny_ * nh) * tmp3
                    * iGI31 * iGH31
                    * (0.5f * iLAMh2 * iLAMi3 + 2.0f * iLAMh * iLAMi4
                       + 5.0f * iLAMi5);
            NCLih = dt * PIov4 * Eih * (ny_ * nh) * GI31 * GH31 * tmp3
                    * (GI33 * GH31 * iLAMi2
                       + 2.0f * GI32 * GH32 * iLAMi * iLAMh
                       + GI31 * GH33 * iLAMh2);
            QCLih = fminf(QCLih, qi);
            NCLih = fminf(QCLih * (ny_ * iQI), NCLih);
        } else {
            QCLih = 0.0f; NCLih = 0.0f;
        }
        if (qn > MY2_epsQ) {
            tmp1 = vh0 - vs0;
            tmp3 = sqrtf(tmp1 * tmp1 + 0.04f * vh0 * vs0);
            tmp4 = iLAMs2 * iLAMs2;
            (void)tmp4;
            // snowSpherical = .false. -> the dms=2 form (:1994-1997)
            QCLsh = dt * cms * ide * MY2_PI * 0.25f * Esh * tmp3 * nn * nh
                    * iGS31 * iGH31
                    * (GH33 * GS33 * (iLAMh * iLAMh) * (iLAMs * iLAMs)
                       + 2.0f * GH32 * GS34 * iLAMh
                         * (iLAMs * iLAMs * iLAMs)
                       + GH31 * GS35 * (iLAMs * iLAMs * iLAMs * iLAMs));
            NCLsh = dt * PIov4 * Esh * (nn * nh) * GS31 * GH31 * tmp3
                    * (GS33 * GH31 * iLAMs2
                       + 2.0f * GS32 * GH32 * iLAMs * iLAMh
                       + GS31 * GH33 * iLAMh2);
            QCLsh = fminf(QCLsh, qn);
            NCLsh = fminf(fminf((nn * iQN) * QCLsh, NCLsh), nn);
        } else {
            QCLsh = 0.0f; NCLsh = 0.0f;
        }
        VENTh = MY2_Avx * GH32 * powf(iLAMh, 2.0f + MY2_alpha_h)
                + MY2_Bvx * ScTHRD * sqrtf(gam * MY2_afh * iMUkin) * GH09
                  * powf(iLAMh, 2.5f + 0.5f * MY2_bfh + MY2_alpha_h);
        QHwet = fmaxf(0.0f,
                      dt * PI2 * (de * MY2_CHLC * Cdiff * DELqvs - Ka * Tc)
                      * No_h * ide / (MY2_CHLF + MY2_CPW * Tc) * VENTh
                      + (QCLih * iEih + QCLsh * iEsh)
                        * (1.0f - MY2_CPI * Tc / (MY2_CHLF + MY2_CPW * Tc)));
        QVDvh = dt * ide * iABi * (PI2 * (Si - 1.0f) * No_h * VENTh);
        VDmax = (q - qsi) / (1.0f + ck6 * qsi / ((t - 7.66f) * (t - 7.66f)));
        if (Si >= 1.0f) {
            QVDvh = fminf(fmaxf(QVDvh, 0.0f), VDmax);
        } else {
            if (VDmax < 0.0f) QVDvh = fmaxf(QVDvh, VDmax);
        }
        NVDvh = 0.0f;                                        // :2027
    } else {
        QCLch = 0.0f; QCLrh = 0.0f; QCLih = 0.0f; QCLsh = 0.0f;
        QHwet = 0.0f;
        NCLch = 0.0f; NCLrh = 0.0f; NCLsh = 0.0f; NCLih = 0.0f;
    }

    if (t > MY2_TRPL) {
        // ------------------------- T > To (:2034-2088) -------------------
        QMLir = qi;                                          // :2041
        qi = 0.0f;
        NMLir = ny_;
        if (qn > MY2_epsQ) {
            QMLsr = dt * (PI2 * ide * iCHLF * No_s * VENTs
                          * (Ka * Tc - MY2_CHLC * Cdiff * DELqvs)
                          + MY2_CPW * iCHLF * Tc * (QCLcs + QCLrs) / dt);
            QMLsr = fminf(fmaxf(QMLsr, 0.0f), qn);
            NMLsr = nn * iQN * QMLsr;
        } else {
            QMLsr = 0.0f; NMLsr = 0.0f;
        }
        if (qg > MY2_epsQ) {
            QMLgr = dt * (PI2 * ide * iCHLF * No_g * VENTg
                          * (Ka * Tc - MY2_CHLC * Cdiff * DELqvs)
                          + MY2_CPW * iCHLF * Tc * (QCLcg + QCLrg) / dt);
            QMLgr = fminf(fmaxf(QMLgr, 0.0f), qg);
            NMLgr = ng * iQG * QMLgr;
        } else {
            QMLgr = 0.0f; NMLgr = 0.0f;
        }
        if (qh > MY2_epsQ && Tc > 5.0f) {
            VENTh = MY2_Avx * GH32 * powf(iLAMh, 2.0f + MY2_alpha_h)
                    + MY2_Bvx * ScTHRD * sqrtf(gam * MY2_afh * iMUkin) * GH09
                      * powf(iLAMh, 2.5f + 0.5f * MY2_bfh + MY2_alpha_h);
            QMLhr = dt * (PI2 * ide * iCHLF * No_h * VENTh
                          * (Ka * Tc - MY2_CHLC * Cdiff * DELqvs)
                          + MY2_CPW / MY2_CHLF * Tc * (QCLch + QCLrh) / dt);
            QMLhr = fminf(fmaxf(QMLhr, 0.0f), qh);
            NMLhr = nh * iQH * QMLhr;
            if (QCLrh > 0.0f) NMLhr = NMLhr * 0.1f;
        } else {
            QMLhr = 0.0f; NMLhr = 0.0f;
        }
        // :2079-2088 -- every cold term is reset
        QNUvi = 0.f; QFZci = 0.f; QVDvi = 0.f; QVDvs = 0.f;
        QCLis = 0.f; QCLri = 0.f;
        QCNgh = 0.f; QIMsi = 0.f; QIMgi = 0.f; QCLir = 0.f;
        QCLrs = 0.f; QCLgr = 0.f; QCLrg = 0.f; QCNis = 0.f;
        QCNsg = 0.f; QCLsr = 0.f;
        NNUvi = 0.f; NFZci = 0.f; NCLgr = 0.f; NCLrg = 0.f; NgCNgh = 0.f;
        NCLis = 0.f; NVDvi = 0.f; NVDvs = 0.f; NCLri = 0.f; NCLsr = 0.f;
        NCNsg = 0.f; NhCNgh = 0.f; NiCNis = 0.f; NsCNis = 0.f;
        NIMsi = 0.f; NIMgi = 0.f; NCLir = 0.f; NCLrs = 0.f;
    } else {
        // ------------------------- T < To (:2090-2505) -------------------
        QMLir = 0.f; QMLsr = 0.f; QMLgr = 0.f; QMLhr = 0.f;
        NMLir = 0.f; NMLsr = 0.f; NMLgr = 0.f; NMLhr = 0.f;

        // Probabilistic (Bigg) freezing of rain (:2099-2116)
        if (Tc < MY2_Tc_FZrh && qr > MY2_epsQ) {
            NrFZrh = -dt * MY2_Bbigg * (expf(MY2_Abigg * Tc) - 1.0f)
                     * de * qr * idew;
            float Rz = 1.0f;
            NhFZrh = Rz * NrFZrh;
            QFZrh = NrFZrh * (qr * iNR);
        } else {
            QFZrh = 0.0f; NrFZrh = 0.0f; NhFZrh = 0.0f;
        }

        // Homogeneous freezing of cloud to ice (:2122-2134)
        if (qc > MY2_epsQ) {
            tmp2 = Tc * Tc; tmp3 = tmp2 * Tc; tmp4 = tmp2 * tmp2;
            float JJ = powf(10.0f,
                              fmaxf(-20.0f, (-606.3952f - 52.6611f * Tc
                                             - 1.7439f * tmp2
                                             - 0.0265f * tmp3
                                             - 1.536e-4f * tmp4)));
            tmp1 = 1.0e6f * (de * (qc * iNC) * icmr);
            float FRAC = 1.0f - expf(-JJ * PIov6 * tmp1 * dt);
            if (Tc > -30.0f) FRAC = 0.0f;
            if (Tc < -50.0f) FRAC = 1.0f;
            QFZci = FRAC * qc;
            NFZci = FRAC * nc;
        } else {
            QFZci = 0.0f; NFZci = 0.0f;
        }

        // Primary ice nucleation, primIceNucl = 1 (:2136-2193)
        NNUvi = 0.0f; QNUvi = 0.0f;
        {
            float NuDEPSOR = 0.0f, NuCONT = 0.0f;
            float Simax;
            if (qsi > 1.0e-20f) {
                Simax = fminf(Si, MY2_satw_peak * qsw / qsi);
            } else {
                Simax = 0.0f;
            }
            tmp1 = t - 7.66f;
            float NNUmax = fmaxf(0.0f,
                                 de / MY2_mio * (q - qsi)
                                 / (1.0f + ck6 * (qsi / (tmp1 * tmp1))));
            if (Tc < -5.0f && Si > 1.0f) {
                NuDEPSOR = fmaxf(0.0f,
                                 1.0e3f * expf(12.96f * (Simax - 1.0f)
                                                 - 0.639f) - ny_);
            }
            if (qc > MY2_epsQ && Tc < -2.0f && wz > 0.001f) {
                float GG = 1.0f * idew
                           / (MY2_RGASV * t / ((qsw * pr) / MY2_EPS1) / Cdiff
                              + MY2_CHLC / Ka / t
                                * (MY2_CHLC / MY2_RGASV / t - 1.0f));
                float Swmax = my2_SxFNC_w(wz, Tc, pr);
                float ssat;
                if (qsw > 1.0e-20f) {
                    ssat = fminf(q / qsw, Swmax) - 1.0f;
                } else {
                    ssat = 0.0f;
                }
                float Tcc = Tc + GG * ssat * MY2_CHLC / Kdiff;
                float Na = expf(4.11f - 0.262f * Tcc);
                float Kn = MY2_LAMa0 * t * MY2_p0 / (MY2_T0 * pr * MY2_Ra);
                float PSIa = -MY2_kBoltz * Tcc
                             / (6.0f * MY2_PI * MY2_Ra * MUdyn) * (1.0f + Kn);
                float ft = 0.4f * (1.0f + 1.45f * Kn
                                   + 0.4f * Kn * expf(-1.0f / Kn))
                           * (Ka + 2.5f * Kn * MY2_KAPa)
                           / (1.0f + 3.0f * Kn)
                           / (2.0f * Ka + 5.0f * MY2_KAPa * Kn + MY2_KAPa);
                // :2169 -- Dc is RECOMPUTED here without the icmr->thrd
                // helper and overwrites the size-distribution Dc above.
                Dc = powf(de * (qc * iNC) * icmr, MY2_thrd);
                float F1 = PI2 * Dc * Na * nc;
                float F2 = Ka / pr * (Tc - Tcc);
                float NuCONTA = -F1 * F2 * MY2_RGASV * t / MY2_CHLC * ide;
                float NuCONTB = F1 * F2 * ft * ide;
                float NuCONTC = F1 * PSIa;
                NuCONT = fmaxf(0.0f, (NuCONTA + NuCONTB + NuCONTC) * dt);
            }
            NNUvi = fminf(NNUmax, NuDEPSOR + NuCONT);        // icephase_ON
            QNUvi = MY2_mio * ide * NNUvi;
            QNUvi = fminf(QNUvi, q);
        }

        // ------------------------------ ICE (:2197-2300) -----------------
        if (qi > MY2_epsQ) {
            float No_i = ny_ * iGI31 / iLAMi;                // :2203
            float VENTi = MY2_Avx * GI32 * iLAMi * iLAMi
                          + MY2_Bvx * ScTHRD * sqrtf(gam * MY2_afi * iMUkin)
                            * GI6 * powf(iLAMi, 2.5f + 0.5f * MY2_bfi
                                                  + MY2_alpha_i);
            QVDvi = dt * ide * MY2_capFact_i * iABi
                    * (PI2 * (Si - 1.0f) * No_i * VENTi);
            VDmax = (q - qsi)
                    / (1.0f + ck6 * qsi / ((t - 7.66f) * (t - 7.66f)));
            if (Si >= 1.0f) {
                QVDvi = fminf(fmaxf(QVDvi, 0.0f), VDmax);
            } else {
                if (VDmax < 0.0f) QVDvi = fmaxf(QVDvi, VDmax);
            }
            NVDvi = fminf(0.0f, (ny_ * iQI) * QVDvi);        // :2219

            // Conversion to snow (:2222-2240)
            if (qi + QVDvi > MY2_epsQ && ny_ + NVDvi > MY2_epsN) {
                tmp5 = iLAMi;
                tmp6 = No_i;
                tmp1 = qi + QVDvi;
                tmp2 = ny_ + NVDvi;
                tmp3 = 1.0f / tmp2;
                float iLAMi_h = fmaxf(MY2_iLAMmin2,
                                      my2_iLAMDA_x(de, tmp1, tmp3, icexi9,
                                                   MY2_thrd));
                float No_i_h = tmp2 * iGI31 / iLAMi_h;
                tmp4 = expf(-MY2_Dso / iLAMi_h);
                NiCNis = No_i_h * iLAMi_h * tmp4;
                NsCNis = NiCNis;
                QCNis = cmi * No_i_h * tmp4
                        * (MY2_Dso * MY2_Dso * MY2_Dso * iLAMi_h
                           + 3.0f * MY2_Dso * MY2_Dso * iLAMi_h * iLAMi_h
                           + 6.0f * MY2_Dso * iLAMi_h * iLAMi_h * iLAMi_h
                           + 6.0f * iLAMi_h * iLAMi_h * iLAMi_h * iLAMi_h);
                iLAMi = tmp5;
                No_i = tmp6;
                (void)No_i;
            }

            // 3-component freezing, ice + rain (:2249-2282)
            if (qr > MY2_epsQ && qi > MY2_epsQ) {
                tmp1 = vr0 - vi0;
                tmp3 = sqrtf(tmp1 * tmp1 + 0.04f * vr0 * vi0);
                QCLir = dt * cmi * MY2_Eri * PIov4 * ide * (nr * ny_)
                        * iGI31 * iGR31 * tmp3
                        * (GI36 * GR31 * iLAMi5
                           + 2.0f * GI35 * GR32 * iLAMi4 * iLAMr
                           + GI34 * GR33 * iLAMi3 * iLAMr2);
                NCLri = dt * PIov4 * MY2_Eri * (nr * ny_) * iGI31 * iGR31
                        * tmp3
                        * (GI33 * GR31 * iLAMi2
                           + 2.0f * GI32 * GR32 * iLAMi * iLAMr
                           + GI31 * GR33 * iLAMr2);
                QCLri = dt * cmr * MY2_Eri * PIov4 * ide * (ny_ * nr)
                        * iGR31 * iGI31 * tmp3
                        * (GR36 * GI31 * iLAMr5
                           + 2.0f * GR35 * GI32 * iLAMr4 * iLAMi
                           + GR34 * GI33 * iLAMr3 * iLAMi2);
                NCLir = fminf(QCLir * (ny_ * iQI), NCLri);
                QCLri = fminf(QCLri, qr);
                QCLir = fminf(QCLir, qi);
                NCLri = fminf(NCLri, nr);
                NCLir = fminf(NCLir, ny_);
                tmp1 = fmaxf(Di, Dr);
                float dey = (MY2_dei * Di * Di * Di + MY2_dew * Dr * Dr * Dr)
                            / (tmp1 * tmp1 * tmp1);
                if (dey > 0.5f * (MY2_deg + MY2_deh)
                        && Dr > MY2_Dr_3cmpThrs) {
                    Dirg = 0.0f; Dirh = 1.0f;
                } else {
                    Dirg = 1.0f; Dirh = 0.0f;
                }
            } else {
                QCLir = 0.0f; NCLir = 0.0f; QCLri = 0.0f;
                NCLri = 0.0f; Dirh = 0.0f; Dirg = 0.0f;
            }

            // Rime-splintering (:2285-2291)
            float ff = 0.0f;
            if (Tc >= -8.0f && Tc <= -5.0f) ff = 3.5e8f * (Tc + 8.0f)
                                                 * MY2_thrd;
            if (Tc > -5.0f && Tc < -3.0f) ff = 3.5e8f * (-3.0f - Tc) * 0.5f;
            NIMsi = de * ff * QCLcs;
            NIMgi = de * ff * QCLcg;
            QIMsi = MY2_mio * ide * NIMsi;
            QIMgi = MY2_mio * ide * NIMgi;
        } else {
            QVDvi = 0.f; QCNis = 0.f;
            QIMsi = 0.f; QIMgi = 0.f; QCLri = 0.f; QCLir = 0.f;
            NVDvi = 0.f; NCLir = 0.f; NIMsi = 0.f;
            NiCNis = 0.f; NsCNis = 0.f; NIMgi = 0.f; NCLri = 0.f;
        }

        // ----------------------------- SNOW (:2304-2394) -----------------
        if (qn > MY2_epsQ) {
            QVDvs = dt * ide * MY2_capFact_s * iABi
                    * (PI2 * (Si - 1.0f) * No_s * VENTs
                       - CHLS * MY2_CHLF / (Ka * MY2_RGASV * t * t)
                         * QCLcs / dt);
            VDmax = (q - qsi)
                    / (1.0f + ck6 * qsi / ((t - 7.66f) * (t - 7.66f)));
            if (Si >= 1.0f) {
                QVDvs = fminf(fmaxf(QVDvs, 0.0f), VDmax);
            } else {
                if (VDmax < 0.0f) QVDvs = fmaxf(QVDvs, VDmax);
            }
            NVDvs = -fminf(0.0f, (nn * iQN) * QVDvs);

            // Conversion to graupel (:2325-2338)
            if (QCLcs > 0.0f && QCLcs > MY2_CNsgThres * QVDvs) {
                tmp1 = 100.0f;
                QCNsg = fminf(qn + QCLcs, QCLcs * (tmp1 * QCLcs / qn));
                NCNsg = de * QCNsg / (qn + QCLcs);
            } else {
                QCNsg = 0.0f;
                NCNsg = 0.0f;
            }

            // 3-component freezing, snow + rain (:2341-2387)
            if (qr > MY2_epsQ && qn > MY2_epsQ && Tc < -5.0f) {
                tmp1 = vs0 - vr0;
                tmp2 = sqrtf(tmp1 * tmp1 + 0.04f * vs0 * vr0);
                tmp6 = iLAMs2 * iLAMs2 * iLAMs;
                (void)tmp6;
                QCLrs = dt * cmr * MY2_Ers * PIov4 * ide * nn * nr
                        * iGR31 * iGS31 * tmp2
                        * (GR36 * GS31 * iLAMr5
                           + 2.0f * GR35 * GS32 * iLAMr4 * iLAMs
                           + GR34 * GS33 * iLAMr3 * iLAMs2);
                NCLrs = dt * 0.25e0f * MY2_PI * MY2_Ers * (nn * nr)
                        * iGR31 * iGS31 * tmp2
                        * (GR33 * GS31 * iLAMr2
                           + 2.0f * GR32 * GS32 * iLAMr * iLAMs
                           + GR31 * GS33 * iLAMs2);
                // snowSpherical = .false. -> the dms=2 form (:2360-2362)
                QCLsr = dt * cms * ide * MY2_PI * 0.25f * MY2_Ers * tmp2
                        * nn * nr * iGS31 * iGR31
                        * (GR33 * GS33 * (iLAMr * iLAMr) * (iLAMs * iLAMs)
                           + 2.0f * GR32 * GS34 * iLAMr
                             * (iLAMs * iLAMs * iLAMs)
                           + GR31 * GS35 * (iLAMs * iLAMs * iLAMs * iLAMs));
                NCLsr = fminf(QCLsr * (nn * iQN), NCLrs);
                QCLrs = fminf(QCLrs, qr);
                QCLsr = fminf(QCLsr, qn);
                NCLrs = fminf(NCLrs, nr);
                NCLsr = fminf(NCLsr, nn);
                Dsrs = 0.0f; Dsrg = 0.0f; Dsrh = 0.0f;
                tmp1 = fmaxf(Ds, Dr);
                tmp2 = tmp1 * tmp1 * tmp1;
                float dey = (des * Ds * Ds * Ds + MY2_dew * Dr * Dr * Dr)
                            / tmp2;
                if (dey <= 0.5f * (des + MY2_deg)) Dsrs = 1.0f;
                if (dey > 0.5f * (des + MY2_deg)
                        && dey < 0.5f * (MY2_deg + MY2_deh)) Dsrg = 1.0f;
                if (dey >= 0.5f * (MY2_deg + MY2_deh)) {
                    Dsrh = 1.0f;
                    if (Dr < MY2_Dr_3cmpThrs) {
                        Dsrg = 1.0f; Dsrh = 0.0f;
                    }
                }
            } else {
                QCLrs = 0.f; QCLsr = 0.f; NCLrs = 0.f; NCLsr = 0.f;
            }
        } else {
            QVDvs = 0.f; QCLcs = 0.f; QCNsg = 0.f; QCLsr = 0.f;
            QCLrs = 0.f;
            NVDvs = 0.f; NCLcs = 0.f; NCLsr = 0.f; NCLrs = 0.f;
            NCNsg = 0.f;
        }

        // --------------------------- GRAUPEL (:2398-2482) ----------------
        if (qg > MY2_epsQ && ng > MY2_epsN) {
            if ((QCLcg + QCLrg) > 0.0f) {
                tmp1 = 1.1e4f * de * (qc + qr) + 1.0f;
                tmp1 = fmaxf(1.0f, tmp1);
                float D_sll = 2.0f * 0.01f
                              * (expf(fminf(20.0f, -Tc / tmp1)) - 1.0f);
                D_sll = fminf(1.0f, fmaxf(0.0001f, D_sll));
                tmp1 = iLAMg;
                tmp2 = No_g;
                tmp3 = qg + QCLcg + QCLrg;
                float iLAMg_h = expf(MY2_thrd
                                       * logf(de * tmp3
                                                / (ng * 6.0f * cmg)));
                float No_g_h = ng / iLAMg_h;
                tmp4 = expf(-D_sll / iLAMg_h);
                float Ng_tail = No_g_h * iLAMg_h * tmp4;
                if (Ng_tail > MY2_Ngh_crit) {
                    NgCNgh = fminf(ng, Ng_tail);
                    QCNgh = fminf(qg,
                                  cmg * No_g_h * tmp4
                                  * (D_sll * D_sll * D_sll * iLAMg_h
                                     + 3.0f * D_sll * D_sll * iLAMg_h
                                       * iLAMg_h
                                     + 6.0f * D_sll * iLAMg_h * iLAMg_h
                                       * iLAMg_h
                                     + 6.0f * iLAMg_h * iLAMg_h * iLAMg_h
                                       * iLAMg_h));
                    float Rz = 1.0f;
                    NhCNgh = Rz * NgCNgh;
                } else {
                    QCNgh = 0.0f; NgCNgh = 0.0f; NhCNgh = 0.0f;
                }
                iLAMg = tmp1;
                No_g = tmp2;
            }
            // 3-component freezing, graupel + rain (:2437-2475)
            if (qr > MY2_epsQ && Tc < -5.0f) {
                tmp1 = vg0 - vr0;
                tmp2 = sqrtf(tmp1 * tmp1 + 0.04f * vg0 * vr0);
                tmp8 = iLAMg2 * iLAMg;
                tmp9 = tmp8 * iLAMg;
                tmp10 = tmp9 * iLAMg;
                float Kstoke = MY2_dew * fabsf(vg0 - vr0) * Dr * Dr
                               / (9.0f * MUdyn * Dg);
                Kstoke = fmaxf(1.5f, fminf(10.0f, Kstoke));
                float Erg = 0.55f * log10f(2.51f * Kstoke);
                QCLrg = dt * cmr * Erg * PIov4 * ide * (ng * nr)
                        * iGR31 * iGG31 * tmp2
                        * (GR36 * GG31 * iLAMr5
                           + 2.0f * GR35 * GG32 * iLAMr4 * iLAMg
                           + GR34 * GG33 * iLAMr3 * iLAMg2);
                NCLrg = dt * PIov4 * Erg * (ng * nr) * iGR31 * iGG31 * tmp2
                        * (GR33 * GG31 * iLAMr2
                           + 2.0f * GR32 * GG32 * iLAMr * iLAMg
                           + GR31 * GG33 * iLAMg2);
                QCLgr = dt * cmg * Erg * PIov4 * ide * (nr * ng)
                        * iGG31 * iGR31 * tmp2
                        * (GG36 * GR31 * tmp10
                           + 2.0f * GG35 * GR32 * tmp9 * iLAMr
                           + GG34 * GR33 * tmp8 * iLAMr2);
                NCLgr = fminf(NCLrg, QCLgr * (ng * iQG));
                QCLrg = fminf(QCLrg, qr);
                QCLgr = fminf(QCLgr, qg);
                NCLrg = fminf(NCLrg, nr);
                NCLgr = fminf(NCLgr, ng);
                tmp1 = fmaxf(Dg, Dr);
                tmp2 = tmp1 * tmp1 * tmp1;
                float dey = (MY2_deg * Dg * Dg * Dg
                             + MY2_dew * Dr * Dr * Dr) / tmp2;
                if (dey > 0.5f * (MY2_deg + MY2_deh)
                        && Dr > MY2_Dr_3cmpThrs) {
                    Dgrg = 0.0f; Dgrh = 1.0f;
                } else {
                    Dgrg = 1.0f; Dgrh = 0.0f;
                }
            } else {
                QCLgr = 0.f; QCLrg = 0.f; NCLgr = 0.f; NCLrg = 0.f;
            }
        } else {
            QCNgh = 0.f; QCLgr = 0.f; QCLrg = 0.f; NgCNgh = 0.f;
            NhCNgh = 0.f; NCLgr = 0.f; NCLrg = 0.f;
        }

        // ----------------------------- HAIL (:2486-2503) -----------------
        if (qh > MY2_epsQ) {
            if (QHwet < (QCLch + QCLrh + QCLih + QCLsh) && Tc > -40.0f) {
                QCLih = fminf(QCLih * iEih, qi);
                NCLih = fminf(NCLih * iEih, ny_);
                QCLsh = fminf(QCLsh * iEsh, qn);
                NCLsh = fminf(NCLsh * iEsh, nn);
                tmp3 = QCLrh;
                QCLrh = QHwet - (QCLch + QCLih + QCLsh);
                QSHhr = tmp3 - QCLrh;
                NSHhr = de * QSHhr
                        / (cmr * MY2_Drshed * MY2_Drshed * MY2_Drshed);
            } else {
                NSHhr = 0.0f;
            }
        } else {
            NSHhr = 0.0f;
        }
    }  // end T<To block

    // --- Prevent mass transfer from accretion during melting (:2513-2556) -
    tmp1 = nn + NsCNis - NVDvs - NCNsg - NMLsr - NCLss - NCLsr - NCLsh
           + NCLsrs;
    if (tmp1 < MY2_epsN) {
        QCLcs = 0.0f; NCLcs = 0.0f; QCLrs = 0.0f; NCLrs = 0.0f;
        if (Dsrs == 1.0f) { QCLrs = 0.0f; QCLsr = 0.0f; }
    }
    tmp2 = ng + NCNsg - NCLgr - NVDvg - NMLgr + NCLirg + NCLsrg + NCLgrg
           - NgCNgh;
    if (tmp2 < MY2_epsN) {
        QCLcg = 0.0f; NCLcg = 0.0f; QCLrg = 0.0f; NCLrg = 0.0f;
        if (Dirg == 1.0f) { QCLri = 0.0f; QCLir = 0.0f; }
        if (Dsrg == 1.0f) { QCLrs = 0.0f; QCLsr = 0.0f; }
    }
    tmp3 = nh + NhFZrh + NhCNgh - NMLhr - NVDvh + NCLirh + NCLsrh + NCLgrh;
    if (tmp3 < MY2_epsN) {
        QCLch = 0.0f; NCLch = 0.0f; QCLrh = 0.0f; NCLrh = 0.0f;
        if (Dirh == 1.0f) { QCLri = 0.0f; QCLir = 0.0f; }
        if (Dsrh == 1.0f) { QCLrs = 0.0f; QCLsr = 0.0f; }
    }

    // --- Overdepletion adjustment, two iterations (:2562-2671) -----------
    for (int niter = 0; niter < 2; ++niter) {
        float source, sink, sour, ratio;
        // (1) Vapour
        source = q + my2_dim(-QVDvi, 0.f) + my2_dim(-QVDvs, 0.f)
                 + my2_dim(-QVDvg, 0.f) + my2_dim(-QVDvh, 0.f);
        sink = QNUvi + my2_dim(QVDvi, 0.f) + my2_dim(QVDvs, 0.f);
        sour = fmaxf(source, 0.0f);
        if (sink > sour) {
            ratio = sour / sink;
            QNUvi = ratio * QNUvi; NNUvi = ratio * NNUvi;
            if (QVDvi > 0.f) { QVDvi = ratio * QVDvi; NVDvi = ratio * NVDvi; }
            if (QVDvs > 0.f) { QVDvs = ratio * QVDvs; NVDvs = ratio * NVDvs; }
            QVDvg = ratio * QVDvg; NVDvg = ratio * NVDvg;
            QVDvh = ratio * QVDvh; NVDvh = ratio * NVDvh;
        }
        // (2) Cloud
        source = qc;
        sink = QCLcs + QCLcg + QCLch + QFZci;
        sour = fmaxf(source, 0.0f);
        if (sink > sour) {
            ratio = sour / sink;
            QFZci = ratio * QFZci; NFZci = ratio * NFZci;
            QCLcs = ratio * QCLcs; NCLcs = ratio * NCLcs;
            QCLcg = ratio * QCLcg; NCLcg = ratio * NCLcg;
            QCLch = ratio * QCLch; NCLch = ratio * NCLch;
        }
        // (3) Rain
        source = qr + QMLsr + QMLgr + QMLhr + QMLir;
        sink = QCLri + QCLrs + QCLrg + QCLrh + QFZrh;
        sour = fmaxf(source, 0.0f);
        if (sink > sour) {
            ratio = sour / sink;
            QCLrg = ratio * QCLrg; QCLri = ratio * QCLri;
            NCLri = ratio * NCLri;
            QCLrs = ratio * QCLrs; NCLrs = ratio * NCLrs;
            NCLrg = ratio * NCLrg; QCLrh = ratio * QCLrh;
            NCLrh = ratio * NCLrh;
            QFZrh = ratio * QFZrh; NrFZrh = ratio * NrFZrh;
            NhFZrh = ratio * NhFZrh;
            if (ratio == 0.0f) {
                Dirg = 0.f; Dirh = 0.f; Dgrg = 0.f; Dgrh = 0.f;
                Dsrs = 0.f; Dsrg = 0.f; Dsrh = 0.f;
            }
        }
        // (4) Ice
        source = qi + QNUvi + my2_dim(QVDvi, 0.f) + QFZci;
        sink = QCNis + QCLir + my2_dim(-QVDvi, 0.f) + QCLis + QCLig + QCLih
               + QMLir;
        sour = fmaxf(source, 0.0f);
        if (sink > sour) {
            ratio = sour / sink;
            QMLir = ratio * QMLir; NMLir = ratio * NMLir;
            if (QVDvi < 0.f) { QVDvi = ratio * QVDvi; NVDvi = ratio * NVDvi; }
            QCNis = ratio * QCNis; NiCNis = ratio * NiCNis;
            NsCNis = ratio * NsCNis;
            QCLir = ratio * QCLir; NCLir = ratio * NCLir;
            QCLig = ratio * QCLig;
            QCLis = ratio * QCLis; NCLis = ratio * NCLis;
            QCLih = ratio * QCLih; NCLih = ratio * NCLih;
            if (ratio == 0.0f) { Dirg = 0.f; Dirh = 0.f; }
        }
        // (5) Snow
        source = qn + QCNis + my2_dim(QVDvs, 0.f) + QCLis
                 + Dsrs * (QCLrs + QCLsr) + QCLcs;
        sink = my2_dim(-QVDvs, 0.f) + QCNsg + QMLsr + QCLsr + QCLsh;
        sour = fmaxf(source, 0.0f);
        if (sink > sour) {
            ratio = sour / sink;
            if (QVDvs <= 0.f) { QVDvs = ratio * QVDvs; NVDvs = ratio * NVDvs; }
            QCNsg = ratio * QCNsg; NCNsg = ratio * NCNsg;
            QMLsr = ratio * QMLsr;
            NMLsr = ratio * NMLsr; QCLsr = ratio * QCLsr;
            NCLsr = ratio * NCLsr;
            QCLsh = ratio * QCLsh; NCLsh = ratio * NCLsh;
            if (ratio == 0.0f) { Dsrs = 0.f; Dsrg = 0.f; Dsrh = 0.f; }
        }
        // (6) Graupel
        source = qg + QCNsg + my2_dim(QVDvg, 0.f) + Dirg * (QCLri + QCLir)
                 + Dgrg * (QCLrg + QCLgr) + QCLcg + Dsrg * (QCLrs + QCLsr)
                 + QCLig;
        sink = my2_dim(-QVDvg, 0.f) + QMLgr + QCNgh + QCLgr;
        sour = fmaxf(source, 0.0f);
        if (sink > sour) {
            ratio = sour / sink;
            QVDvg = ratio * QVDvg; NVDvg = ratio * NVDvg;
            QMLgr = ratio * QMLgr;
            NMLgr = ratio * NMLgr; QCNgh = ratio * QCNgh;
            NgCNgh = ratio * NgCNgh;
            QCLgr = ratio * QCLgr; NCLgr = ratio * NCLgr;
            NhCNgh = ratio * NhCNgh;
            if (ratio == 0.0f) { Dgrg = 0.f; Dgrh = 0.f; }
        }
        // (7) Hail
        source = qh + my2_dim(QVDvh, 0.f) + QCLch + QCLrh
                 + Dirh * (QCLri + QCLir) + QCLih + QCLsh
                 + Dsrh * (QCLrs + QCLsr) + QCNgh + Dgrh * (QCLrg + QCLgr)
                 + QFZrh;
        sink = my2_dim(-QVDvh, 0.f) + QMLhr;
        sour = fmaxf(source, 0.0f);
        if (sink > sour) {
            ratio = sour / sink;
            QVDvh = ratio * QVDvh; NVDvh = ratio * NVDvh;
            QMLhr = ratio * QMLhr; NMLhr = ratio * NMLhr;
        }
    }

    // --- N tendencies for 3-comp-freezing destinations (:2675-2698) ------
    NCLirg = 0.f; NCLirh = 0.f; NCLsrs = 0.f; NCLsrg = 0.f;
    NCLsrh = 0.f; NCLgrg = 0.f; NCLgrh = 0.f;
    if (QCLir + QCLri > 0.0f) {
        tmp1 = fmaxf(Dr, Di);
        tmp2 = tmp1 * tmp1 * tmp1 * PIov6;
        NCLirg = Dirg * de * (QCLir + QCLri) / (MY2_deg * tmp2);
        NCLirh = Dirh * de * (QCLir + QCLri) / (MY2_deh * tmp2);
    }
    if (QCLsr + QCLrs > 0.0f) {
        tmp1 = fmaxf(Dr, Ds);
        tmp2 = tmp1 * tmp1 * tmp1 * PIov6;
        NCLsrs = Dsrs * de * (QCLsr + QCLrs) / (des * tmp2);
        NCLsrg = Dsrg * de * (QCLsr + QCLrs) / (MY2_deg * tmp2);
        NCLsrh = Dsrh * de * (QCLsr + QCLrs) / (MY2_deh * tmp2);
    }
    if (QCLgr + QCLrg > 0.0f) {
        tmp1 = fmaxf(Dr, Dg);
        tmp2 = tmp1 * tmp1 * tmp1 * PIov6;
        NCLgrg = Dgrg * de * (QCLgr + QCLrg) / (MY2_deg * tmp2);
        NCLgrh = Dgrh * de * (QCLgr + QCLrg) / (MY2_deh * tmp2);
    }

    // --- Apply all source/sink terms (:2708-2731) ------------------------
    q = q - QNUvi - QVDvi - QVDvs - QVDvg - QVDvh;
    qc = qc - QCLcs - QCLcg - QCLch - QFZci;
    qr = qr - QCLri + QMLsr - QCLrs - QCLrg + QMLgr - QCLrh + QMLhr - QFZrh
         + QMLir;
    qi = qi + QNUvi + QVDvi + QFZci - QCNis - QCLir - QCLis - QCLig
         - QMLir - QCLih + QIMsi + QIMgi;
    qg = qg + QCNsg + QVDvg + QCLcg - QCLgr - QMLgr - QCNgh - QIMgi + QCLig
         + Dirg * (QCLri + QCLir) + Dgrg * (QCLrg + QCLgr)
         + Dsrg * (QCLrs + QCLsr);
    qn = qn + QCNis + QVDvs + QCLcs - QCNsg - QMLsr - QIMsi - QCLsr + QCLis
         - QCLsh + Dsrs * (QCLrs + QCLsr);
    qh = qh + Dirh * (QCLri + QCLir) - QMLhr + QVDvh + QCLch
         + Dsrh * (QCLrs + QCLsr) + QCLih + QCLsh + QFZrh + QCLrh + QCNgh
         + Dgrh * (QCLrg + QCLgr);

    nc = nc - NCLcs - NCLcg - NCLch - NFZci;
    nr = nr - NCLri - NCLrs - NCLrg - NCLrh + NMLsr + NMLgr + NMLhr - NrFZrh
         + NMLir + NSHhr;
    ny_ = ny_ + NNUvi + NVDvi + NFZci - NCLir - NCLis - NCLig - NCLih
          - NMLir + NIMsi + NIMgi - NiCNis;
    nn = nn + NsCNis - NVDvs - NCNsg - NMLsr - NCLss - NCLsr - NCLsh
         + NCLsrs;
    ng = ng + NCNsg - NCLgr - NVDvg - NMLgr + NCLirg + NCLsrg + NCLgrg
         - NgCNgh;
    nh = nh + NhFZrh + NhCNgh - NMLhr - NVDvh + NCLirh + NCLsrh + NCLgrh;

    t = t + LFP * (QCLri + QCLcs + QCLrs + QFZci - QMLsr + QCLcg + QCLrg
                   - QMLir - QMLgr - QMLhr + QCLch + QCLrh + QFZrh)
        + LSP * (QNUvi + QVDvi + QVDvs + QVDvg + QVDvh);

    // --- Ensure consistency between moments (:2733-2816) -----------------
    tmp1 = pr / (MY2_RGASD * t);       // :2734 updated air density
    if (qc > MY2_epsQ && nc < MY2_epsN) {
        nc = N_c_SM;
    } else if (qc <= MY2_epsQ) {
        q = q + qc; t = t - LCP * qc; qc = 0.0f; nc = 0.0f;
    }
    if (qr > MY2_epsQ && nr < MY2_epsN) {
        nr = powf(MY2_No_r_SM * GR31, 3.0f / (4.0f + MY2_alpha_r))
             * powf(GR31 * iGR34 * tmp1 * qr * icmr,
                      (1.0f + MY2_alpha_r) / (4.0f + MY2_alpha_r));
    } else if (qr <= MY2_epsQ) {
        q = q + qr; t = t - LCP * qr; qr = 0.0f; nr = 0.0f;
    }
    if (qi > MY2_epsQ && ny_ < MY2_epsN) {
        ny_ = fmaxf(2.0f * MY2_epsN, my2_N_Cooper(t));
    } else if (qi <= MY2_epsQ) {
        q = q + qi; t = t - LSP * qi; qi = 0.0f; ny_ = 0.0f;
    }
    if (qn > MY2_epsQ && nn < MY2_epsN) {
        float Nos = my2_Nos_Thompson(t);
        nn = powf(Nos * GS31, dms * icexs2)
             * powf(GS31 * iGS40 * icms * tmp1 * qn,
                      (1.0f + MY2_alpha_s) * icexs2);
    } else if (qn <= MY2_epsQ) {
        q = q + qn; t = t - LSP * qn; qn = 0.0f; nn = 0.0f;
    }
    if (qg > MY2_epsQ && ng < MY2_epsN) {
        ng = powf(MY2_No_g_SM * GG31, 3.0f / (4.0f + MY2_alpha_g))
             * powf(GG31 * iGG34 * tmp1 * qg * icmg,
                      (1.0f + MY2_alpha_g) / (4.0f + MY2_alpha_g));
    } else if (qg <= MY2_epsQ) {
        q = q + qg; t = t - LSP * qg; qg = 0.0f; ng = 0.0f;
    }
    if (qh > MY2_epsQ && nh < MY2_epsN) {
        nh = powf(MY2_No_h_SM * GH31, 3.0f / (4.0f + MY2_alpha_h))
             * powf(GH31 * iGH34 * tmp1 * qh * icmh,
                      (1.0f + MY2_alpha_h) / (4.0f + MY2_alpha_h));
    } else if (qh <= MY2_epsQ) {
        q = q + qh; t = t - LSP * qh; qh = 0.0f; nh = 0.0f;
    }
    if (qh > MY2_epsQ && nh > MY2_epsN) {
        // :2803-2813 -- transfer small hail to graupel.  NOTE the Fortran
        // uses DE(i,k) (the PART 1 density), not the tmp1 recomputed above.
        tmp1 = 1.0f / nh;
        float Dh_n = my2_Dm_x(de, qh, tmp1, icmh, MY2_thrd);
        if (Dh_n < MY2_Dh_min) {
            qg = qg + qh;
            ng = ng + nh;
            qh = 0.0f;
            nh = 0.0f;
        }
    }
    q = fmaxf(q, 0.0f);                        // :2815
    ny_ = fminf(ny_, MY2_Ni_max);              // :2816
    // DIVERGENCE 1: :2819-2823 prints and STOPs on T<173 or T>323 K.  A
    // device kernel cannot abort the run; gpuwm's step-level health gate
    // owns that decision, so the value is left as computed.

    T[gid] = t;   Q[gid] = q;
    QC[gid] = qc; QR[gid] = qr; QI[gid] = qi;
    QN[gid] = qn; QG[gid] = qg; QH[gid] = qh;
    NC[gid] = nc; NR[gid] = nr; NY[gid] = ny_;
    NN[gid] = nn; NG[gid] = ng; NH[gid] = nh;
}

// ==========================================================================
// PART 3 -- warm microphysics (:2836-3126)
// ==========================================================================
extern "C" __global__
void milbrandt2_warm(
        float* __restrict__ T, float* __restrict__ Q,
        float* __restrict__ QC, float* __restrict__ QR,
        float* __restrict__ QI,
        float* __restrict__ NC, float* __restrict__ NR,
        float* __restrict__ NY,
        const float* __restrict__ WZ,
        const float* __restrict__ pres,
        float* __restrict__ DE, float* __restrict__ iDE,
        float* __restrict__ QSW,
        float* __restrict__ QC_in, float* __restrict__ QR_in,
        float* __restrict__ NC_in, float* __restrict__ NR_in,
        const float* __restrict__ ck, float dt,
        int nz, int ny, int nx)
{
    long long n = (long long)nz * ny * nx;
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    long long plane = (long long)ny * nx;
    int k = (int)(gid / plane);
    // :2848 -- ``do k = ktop-kdir, kbot, -kdir``: the top level is excluded.
    if (k >= nz - 1) return;

    float idt = 1.0f / dt;                                   // :1269
    float t = T[gid], q = Q[gid];
    float qc = QC[gid], qr = QR[gid], qi = QI[gid];
    float nc = NC[gid], nr = NR[gid], ny_ = NY[gid];
    float de = DE[gid], ide = iDE[gid], pr = pres[gid];
    float wz = WZ[gid];
    float qc_in = QC_in[gid], qr_in = QR_in[gid];
    float nc_in = NC_in[gid], nr_in = NR_in[gid];

    // :2851-2854 -- per-point initialisation
    float RCAUTR = 0.f, CCACCR = 0.f, Dc = 0.f, iLAMc = 0.f, L = 0.f;
    float RCACCR = 0.f, CCSCOC = 0.f, Dr = 0.f, iLAMr = 0.f, TAU = 0.f;
    float CCAUTR = 0.f, CRSCOR = 0.f, SIGc = 0.f, DrINIT = 0.f;
    float iLAMc3 = 0.f, iLAMc6 = 0.f, iLAMr3 = 0.f, iLAMr6 = 0.f;
    float tmp1, tmp2, tmp3, tmp4;
    float iNC, iNR;

    bool rainPresent = (qr_in > MY2_epsQ && nr_in > MY2_epsN);

    if (qc_in > MY2_epsQ && nc_in > MY2_epsN) {
        iNC = 1.0f / nc_in;
        iLAMc = my2_iLAMDA_x(de, qc_in, iNC, icexc9, MY2_thrd);
        iLAMc3 = iLAMc * iLAMc * iLAMc;
        iLAMc6 = iLAMc3 * iLAMc3;
        Dc = iLAMc * powf(GC2 * iGC1, MY2_thrd);
        SIGc = iLAMc * powf(GC3 * iGC1 - (GC2 * iGC1) * (GC2 * iGC1),
                              0.5f * MY2_thrd);
        L = 0.027f * de * qc_in * (6.25e18f * SIGc * SIGc * SIGc * Dc - 0.4f);
        if (SIGc > MY2_SIGcTHRS)
            TAU = 3.7f / (de * qc_in * (0.5e6f * SIGc - 7.5f));
    }

    if (rainPresent) {
        iNR = 1.0f / nr_in;
        Dr = my2_Dm_x(de, qr_in, iNR, icmr, MY2_thrd);
        if (Dr > 3.e-3f) {                                   // :2874-2881
            tmp1 = (Dr - 3.e-3f);
            tmp2 = (Dr / MY2_DrMax);
            tmp3 = tmp2 * tmp2 * tmp2;
            nr_in = nr_in * fmaxf((1.0f + 2.e4f * tmp1 * tmp1), tmp3);
            iNR = 1.0f / nr_in;
            Dr = my2_Dm_x(de, qr_in, iNR, icmr, MY2_thrd);
        }
        iLAMr = my2_iLAMDA_x(de, qr_in, iNR, icexr9, MY2_thrd);
        iLAMr3 = iLAMr * iLAMr * iLAMr;
        iLAMr6 = iLAMr3 * iLAMr3;
    }

    // Autoconversion (:2888-2908), autoconv_ON = .true.
    if (qc_in > MY2_epsQ && SIGc > MY2_SIGcTHRS) {
        RCAUTR = fminf(fmaxf(L / TAU, 0.0f), qc * idt);
        DrINIT = fmaxf(83.e-6f, 12.6e-4f / (0.5e6f * SIGc - 3.5f));
        float DrAUT = fmaxf(DrINIT, Dr);
        CCAUTR = RCAUTR * de / (cmr * DrAUT * DrAUT * DrAUT);
        CCSCOC = fminf(MY2_KK2 * nc_in * nc_in * GC3 * iGC1 * iLAMc6,
                       nc_in * idt);
    }

    // Accretion, self-collection, breakup (:2911-2955), rainAccr_ON = .true.
    if ((qr_in > 1.2f * fmaxf(L, 0.0f) * ide
         || Dr > fmaxf(5.e-6f, DrINIT)) && rainPresent) {
        if (qc_in > MY2_epsQ && L > 0.0f) {
            if (Dr >= 100.e-6f) {
                CCACCR = MY2_KK1 * (nc_in * nr_in)
                         * (GC2 * iGC1 * iLAMc3 + GR34 * iGR31 * iLAMr3);
                RCACCR = cmr * ide * MY2_KK1 * (nc_in * nr_in) * iLAMc3
                         * (GC3 * iGC1 * iLAMc3
                            + GC2 * iGC1 * GR34 * iGR31 * iLAMr3);
            } else {
                CCACCR = MY2_KK2 * (nc_in * nr_in)
                         * (GC3 * iGC1 * iLAMc6 + GR37 * iGR31 * iLAMr6);
                // :2925-2935 -- the source's own overflow-avoiding grouping
                tmp1 = cmr * ide;
                tmp2 = MY2_KK2 * (nc_in * nr_in) * iLAMc3;
                RCACCR = tmp1 * tmp2;
                tmp1 = GC4 * iGR31;
                tmp1 = (tmp1) * iLAMc6;
                tmp2 = GC2 * iGC1;
                tmp2 = tmp2 * GR37 * iGR31;
                tmp2 = (tmp2) * iLAMr6;
                RCACCR = RCACCR * (tmp1 + tmp2);
            }
            CCACCR = fminf(CCACCR, nc * idt);
            RCACCR = fminf(RCACCR, qc * idt);
        }
        tmp1 = nr_in * nr_in;
        if (Dr >= 100.e-6f) {
            CRSCOR = MY2_KK1 * tmp1 * GR34 * iGR31 * iLAMr3;
        } else {
            CRSCOR = MY2_KK2 * tmp1 * GR37 * iGR31 * iLAMr6;
        }
        float Ec = 1.0f;
        // :2951 -- the breakup test is on iLAMr, not Dr (the source's own
        // substitution, valid for alpha_r = 0).  Ec goes NEGATIVE for large
        // iLAMr, which turns self-collection into a number source; that is
        // WRF's behaviour and the min() below is its only bound.
        if (iLAMr > 300.e-6f) Ec = 2.0f - expf(2300.0f * (iLAMr - 300.e-6f));
        CRSCOR = fminf(Ec * CRSCOR, (0.5f * nr) * idt);
    }

    // Prevent overdepletion of cloud (:2958-2965)
    {
        float source = qc;
        float sink = (RCAUTR + RCACCR) * dt;
        if (sink > source) {
            float ratio = source / sink;
            RCAUTR = ratio * RCAUTR;
            RCACCR = ratio * RCACCR;
            CCACCR = ratio * CCACCR;
        }
    }

    // Apply tendencies (:2968-2971)
    qc = fmaxf(0.0f, qc + (-RCAUTR - RCACCR) * dt);
    qr = fmaxf(0.0f, qr + (RCAUTR + RCACCR) * dt);
    nc = fmaxf(0.0f, nc + (-CCACCR - CCSCOC) * dt);
    nr = fmaxf(0.0f, nr + (CCAUTR - CRSCOR) * dt);

    if (qr > MY2_epsQ && nr > MY2_epsN) {                    // :2973-2988
        iNR = 1.0f / nr;
        Dr = my2_Dm_x(de, qr, iNR, icmr, MY2_thrd);
        if (Dr > 3.e-3f) {
            tmp1 = (Dr - 3.e-3f); tmp2 = tmp1 * tmp1;
            tmp3 = (Dr / MY2_DrMax); tmp4 = tmp3 * tmp3 * tmp3;
            nr = nr * (fmaxf((1.0f + 2.e4f * tmp2), tmp4));
        } else if (Dr < MY2_Dhh) {
            qc = qc + qr;
            nc = nc + nr;
            qr = 0.0f; nr = 0.0f;
        }
    } else {
        qr = 0.0f; nr = 0.0f;
    }

    // ---- Part 3b: condensation / evaporation (:2990-3071) ---------------
    float qsw = my2_qsat(t, pr, 0);
    QSW[gid] = qsw;
    float X = q - qsw;
    // :3001-3002 -- the morr2mom latent-heat denominator, not KY97's ck5
    X = X / (1.0f + ((3.1484e6f - 2370.0f * t) * (3.1484e6f - 2370.0f * t)
                     * qsw)
                    / ((1005.0f * (1.0f + 0.887f * q)) * 461.5f * t * t));
    X = fmaxf(X, -qc);
    qc = qc + X;
    q = q - X;
    t = t + LCP * X;

    if (X > 0.0f) {
        if (wz > 0.001f) {
            nc = fmaxf(nc, my2_NccnFNC(wz, t, pr));
        } else {
            nc = fmaxf(nc, N_c_SM);
        }
    } else {
        if (qc > MY2_epsQ) {
            nc = fmaxf(0.0f, nc + X * nc / fmaxf(qc, MY2_epsQ));
        } else {
            nc = 0.0f;
        }
    }
    if (qc > MY2_epsQ && nc < MY2_epsN) nc = N_c_SM;

    // rain evaporation (:3029-3071)
    qsw = my2_qsat(t, pr, 0);
    QSW[gid] = qsw;
    if (q < qsw && qr > MY2_epsQ && nr > MY2_epsN) {
        float Tc = t - MY2_TRPL;
        float Cdiff = fmaxf(1.62e-5f, (2.2157e-5f + 0.0155e-5f * Tc))
                      * 1.0e5f / pr;
        float MUdyn = fmaxf(1.51e-5f, (1.7153e-5f + 0.0050e-5f * Tc));
        float Ka = fmaxf(2.07e-2f, (2.3971e-2f + 0.0078e-2f * Tc));
        float MUkin = MUdyn * ide;
        float iMUkin = 1.0f / MUkin;
        float ScTHRD = powf(MUkin / Cdiff, MY2_thrd);
        X = qsw - q;
        X = X / (1.0f + ((3.1484e6f - 2370.0f * t)
                         * (3.1484e6f - 2370.0f * t) * qsw)
                        / ((1005.0f * (1.0f + 0.887f * q)) * 461.5f * t * t));
        (void)X;   // :3041-3045 computes X and never uses it again
        de = pr / (MY2_RGASD * t);            // :3046 in-place DE refresh
        ide = 1.0f / de;
        DE[gid] = de;
        iDE[gid] = ide;
        float gam = sqrtf(MY2_DEo * ide);
        tmp1 = 1.0f / nr;
        iLAMr = my2_iLAMDA_x(de, qr, tmp1, icexr9, MY2_thrd);
        float LAMr = 1.0f / iLAMr;
        // :3052-3054 -- WRF evaluates the No_r EXPRESSION in double for one
        // stated reason and one only: its own comment at :3052 says "the
        // following coding of 'No_r=...' prevents overflow", because the
        // single-precision LAMr**(1.+alpha_r) overflows FP32 for small
        // drops.  No_r itself is declared `real, save` at :1013, so the
        // double result is ROUNDED BACK TO REAL(4) on assignment and
        // nothing downstream sees the extra bits.
        float No_r = (float)((double)nr
                             * pow((double)LAMr,
                                   (double)(1.0f + MY2_alpha_r))
                             * (double)iGR31);
        float VENTr = MY2_Avx * GR32 * powf(iLAMr, cexr5)
                      + MY2_Bvx * ScTHRD * sqrtf(gam * MY2_afr * iMUkin)
                        * GR17 * powf(iLAMr, cexr6);
        float ABw = MY2_CHLC * MY2_CHLC / (Ka * MY2_RGASV * t * t)
                    + 1.0f / (de * qsw * Cdiff);
        float ssat = q / qsw - 1.0f;
        // :3058 -- entirely single precision in the Fortran.  Every factor
        // is REAL(4): iDE and QSW are in the `real ::` local block, PI2 and
        // No_r are `real, save` (:1013, :1016), ssat is `real` (:994) and
        // dt is `real, intent(in)` (:850).  There is no promotion here to
        // reproduce -- the double at :3054 is an overflow guard whose
        // result was already rounded to REAL(4) above.  A double-precision
        // chain here would make the port MORE accurate than WRF and would
        // show up in the oracle as an unexplained divergence.
        float rate = -(dt * (ide * PI2 * ssat * No_r * VENTr / ABw));
        float QREVP = fminf(qr, rate);
        tmp1 = qr;
        t = t - LCP * QREVP;
        q = q + QREVP;
        qr = qr - QREVP;
        nr = fmaxf(0.0f, nr - QREVP * nr / tmp1);
        if (qr < MY2_epsQ || nr < MY2_epsN) {
            q = q + qr;
            t = t - qr * LCP;
            qr = 0.0f;
            nr = 0.0f;
        }
    }

    // homogeneous freezing of cloud (:3073-3119), icephase_ON = .true.
    {
        float Tc = t - MY2_TRPL;
        if (qc > MY2_epsQ && Tc < -30.0f) {
            float FRAC = (Tc < -35.0f) ? 1.0f : 0.0f;        // :3086-3090
            float QFZci = FRAC * qc;
            float NFZci = FRAC * nc;
            qc = qc - QFZci;
            nc = nc - NFZci;
            qi = qi + QFZci;
            ny_ = ny_ + NFZci;
            t = t + LFP * QFZci;
            if (qc > MY2_epsQ && nc < MY2_epsN) {
                nc = N_c_SM;
            } else if (qc <= MY2_epsQ) {
                q = q + qc; t = t - LCP * qc; qc = 0.0f; nc = 0.0f;
            }
            if (qi > MY2_epsQ && ny_ < MY2_epsN) {
                ny_ = fmaxf(2.0f * MY2_epsN, my2_N_Cooper(t));
            } else if (qi <= MY2_epsQ) {
                q = q + qi; t = t - LSP * qi; qi = 0.0f; ny_ = 0.0f;
            }
        }
    }

    T[gid] = t; Q[gid] = q;
    QC[gid] = qc; QR[gid] = qr; QI[gid] = qi;
    NC[gid] = nc; NR[gid] = nr; NY[gid] = ny_;
    // The Part 3a drop-size limiter writes NR_in in place (:2878); WRF's
    // NR_in array is a live local for the rest of the routine, so the
    // updated value is stored back even though nothing reads it again.
    NR_in[gid] = nr_in;
    (void)qc_in; (void)nc_in;
}

// ==========================================================================
// PART 4 -- sedimentation (:3134-3334), one thread per column
// ==========================================================================

// sedi_1D (:608-785).  ``QX``/``NX`` are column views with stride ``st``.
// kbot = 0, kdir = +1, so every Fortran ``k+kdir`` is ``k+1``.
template <int KMAX>
__device__ void my2_sedi_1D(
        float* QX, float* NX, int cat,
        const float* DE, const float* iDE, const float* gamfact,
        float epsQ, float epsN, float dmx, float VxMax, float DxMax,
        float dt, const float* DZ, const float* iDZ,
        float* massFlux_bot, int ktop, int nz, int st,
        float afx, float bfx, float cmx,
        float ckQx1, float ckQx2, float ckQx4)
{
    float VVQ[KMAX];
    float VVN[KMAX];

    float icmx = 1.0f / cmx;
    (void)icmx; (void)afx;
    float ratio_Vn2Vq = ckQx2 / ckQx1;                       // :655
    float flux = 0.0f;
    float iDxMax = 1.0f / DxMax;
    float idmx = 1.0f / dmx;
    for (int k = 0; k < nz; ++k) { VVQ[k] = 0.0f; VVN[k] = 0.0f; }

    // :667-670 -- first VV_Q pass (positive values)
    for (int k = 0; k <= ktop; ++k) {
        float qx = QX[(size_t)k * st], nxv = NX[(size_t)k * st];
        if (qx > epsQ && nxv > epsN) {
            float iLAMx = powf((qx * DE[(size_t)k * st] / nxv) * ckQx4,
                                 idmx);
            float iLAMxB0 = powf(iLAMx, bfx);
            VVQ[k] = gamfact[(size_t)k * st] * iLAMxB0 * ckQx1;
        }
    }
    float vmax = 0.0f;
    for (int k = 0; k < nz; ++k) vmax = fmaxf(vmax, VVQ[k]);
    float Vxmaxx = fminf(VxMax, vmax);                       // :672
    float dzMIN = DZ[0];                                     // :674, kdir==1
    for (int k = 1; k < nz; ++k) dzMIN = fminf(dzMIN, DZ[(size_t)k * st]);
    int npassx = (int)(dt * Vxmaxx / (MY2_CoMAX * dzMIN) + 0.5f);  // NINT
    if (npassx < 1) npassx = 1;
    float dtx = dt / (float)npassx;

    for (int nnn = 1; nnn <= npassx; ++nnn) {
        bool firstPass = (nnn == 1);
        for (int k = 0; k <= ktop; ++k) {
            size_t o = (size_t)k * st;
            float qx = QX[o], nxv = NX[o];
            if (qx > epsQ && nxv > epsN) {
                if (firstPass) {
                    VVQ[k] = -VVQ[k];
                } else {
                    float iLAMx = powf((qx * DE[o] / nxv) * ckQx4, idmx);
                    float iLAMxB0 = powf(iLAMx, bfx);
                    VVQ[k] = -(gamfact[o] * iLAMxB0 * ckQx1);
                }
                if (cat == 5) {
                    // :697-702 -- hail size-sorting control.  NOTE the
                    // Fortran omits the DE factor here that Dm_x carries.
                    float t1 = powf(icmx * qx / nxv, MY2_thrd);
                    float t2 = fminf(50.0f, 0.1f * (1000.0f * t1));
                    ratio_Vn2Vq = ((3.0f + t2) * (2.0f + t2) * (1.0f + t2))
                                  / ((3.0f + bfx + t2) * (2.0f + bfx + t2)
                                     * (1.0f + bfx + t2));
                }
                VVN[k] = VVQ[k] * ratio_Vn2Vq;
            } else {
                VVQ[k] = 0.0f;
                VVN[k] = 0.0f;
            }
        }
        flux = flux - VVQ[0] * DE[0] * QX[0];                // :715
        for (int k = 0; k <= ktop; ++k) {
            size_t o = (size_t)k * st, o1 = (size_t)(k + 1) * st;
            QX[o] = QX[o] + dtx * iDE[o]
                    * (-DE[o1] * QX[o1] * VVQ[k + 1] + DE[o] * QX[o] * VVQ[k])
                    * iDZ[o1];
            NX[o] = NX[o] + dtx
                    * (-NX[o1] * VVN[k + 1] + NX[o] * VVN[k]) * iDZ[o1];
            QX[o] = fmaxf(QX[o], 0.0f);
            NX[o] = fmaxf(NX[o], 0.0f);
        }
        for (int k = 0; k <= ktop; ++k) {
            size_t o = (size_t)k * st;
            if (QX[o] > epsQ && NX[o] < epsN) {
                // :737-746 -- find the first level above with NX >= epsN.
                // A Fortran DO that completes without EXIT leaves the index
                // at ktop+kdir, and NX1d(ktop+1) is what gets read; ktop is
                // at most nz-2 (count_columns starts one level down), so
                // that read is in bounds and is reproduced.
                int kk = k + 1;
                while (kk <= ktop && NX[(size_t)kk * st] < epsN) ++kk;
                NX[o] = fmaxf(epsN, NX[(size_t)kk * st]);
            }
            if (QX[o] > epsQ && NX[o] > epsN) {
                float Dx = powf(DE[o] * QX[o] / (NX[o] * cmx), idmx);
                if (cat == 1 && Dx > 3.e-3f) {
                    NX[o] = NX[o] * fmaxf((1.0f + 2.e4f
                                           * (Dx - 3.e-3f) * (Dx - 3.e-3f)),
                                          powf(Dx * iDxMax, 3.0f));
                } else {
                    NX[o] = NX[o] * powf(fmaxf(Dx, DxMax) * iDxMax, dmx);
                }
            }
        }
    }
    *massFlux_bot = flux / (float)npassx;                    // :765
}

// count_columns (:788-836) for one column: returns -1 when the column has
// no QX above ``minQX``, else the highest level with QX > minQX.
__device__ __forceinline__ int my2_count_column(
        const float* QX, float minQX, int ktop_sedi, int st)
{
    int k = ktop_sedi;
    while (true) {
        k -= 1;
        // DIVERGENCE 2: :821-822 would read QX(i,0) when ktop_sedi == kbot.
        // That needs the lowest two layers to span 20 km, so it is
        // unreachable in practice; the defined behaviour is "inactive".
        if (k < 0) return -1;
        if (QX[(size_t)k * st] > minQX) return k;
        if (k == 0) return -1;
    }
}

#define MY2_SEDIMENT_PARAMETERS                                              \
        float* __restrict__ T, float* __restrict__ Q,                        \
        float* __restrict__ QC, float* __restrict__ QR,                      \
        float* __restrict__ QI, float* __restrict__ QN,                      \
        float* __restrict__ QG, float* __restrict__ QH,                      \
        float* __restrict__ NC, float* __restrict__ NR,                      \
        float* __restrict__ NY, float* __restrict__ NN,                      \
        float* __restrict__ NG, float* __restrict__ NH,                      \
        const float* __restrict__ DE, const float* __restrict__ iDE,         \
        const float* __restrict__ DZ, const float* __restrict__ iDZ,         \
        const float* __restrict__ gamfact,                                   \
        float* __restrict__ rainnc, float* __restrict__ rainncv,             \
        float* __restrict__ snownc, float* __restrict__ snowncv,             \
        float* __restrict__ graupelnc, float* __restrict__ graupelncv,       \
        float* __restrict__ hailnc, float* __restrict__ hailncv,             \
        float* __restrict__ sr,                                              \
        const float* __restrict__ ck, float dt,                              \
        int nz, int ny, int nx

#define MY2_SEDIMENT_ARGUMENTS                                               \
        T, Q, QC, QR, QI, QN, QG, QH, NC, NR, NY, NN, NG, NH,                \
        DE, iDE, DZ, iDZ, gamfact,                                           \
        rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv,             \
        hailnc, hailncv, sr, ck, dt, nz, ny, nx

template <int KMAX>
__device__ void my2_sediment_impl(MY2_SEDIMENT_PARAMETERS)
{
    long long ncol = (long long)ny * nx;
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    int st = (int)ncol;

    const float* DEc = DE + col;
    const float* iDEc = iDE + col;
    const float* DZc = DZ + col;
    const float* iDZc = iDZ + col;
    const float* gfc = gamfact + col;
    float* QRc = QR + col; float* NRc = NR + col;
    float* QIc = QI + col; float* NYc = NY + col;
    float* QNc = QN + col; float* NNc = NN + col;
    float* QGc = QG + col; float* NGc = NG + col;
    float* QHc = QH + col; float* NHc = NH + col;

    // :1561-1576 -- zheight and the upper-most sedimentation level.  DZ is
    // strictly positive (iDP inverts a downward pressure difference), so
    // zheight increases with k and the Fortran's top-down search for the
    // first level under zMax_sedi is the LARGEST k satisfying it -- with 0
    // when even the lowest level is above, which is the same value the
    // Fortran loop leaves behind when it never EXITs.  Computed as a running
    // sum so no per-thread height column is stored.
    int ktop_sedi = 0;
    {
        float z = 0.0f;
        for (int k = 0; k < nz; ++k) {
            z = (k == 0) ? DZc[0] : z + DZc[(size_t)k * st];
            if (z < MY2_zMax_sedi) ktop_sedi = k;
        }
    }

    float fluxM_r = 0.f, fluxM_i = 0.f, fluxM_s = 0.f;
    float fluxM_g = 0.f, fluxM_h = 0.f;

    int kt;
    kt = my2_count_column(QRc, MY2_epsQr_sedi, ktop_sedi, st);
    if (kt >= 0)
        my2_sedi_1D<KMAX>(QRc, NRc, 1, DEc, iDEc, gfc, MY2_epsQ, MY2_epsN,
                          MY2_dmr, MY2_VrMax, MY2_DrMax, dt, DZc, iDZc,
                          &fluxM_r, kt, nz, st,
                          MY2_afr, MY2_bfr, cmr, ckQr1, ckQr2, icexr9);
    kt = my2_count_column(QIc, MY2_epsQi_sedi, ktop_sedi, st);
    if (kt >= 0)
        my2_sedi_1D<KMAX>(QIc, NYc, 2, DEc, iDEc, gfc, MY2_epsQ, MY2_epsN,
                          MY2_dmi, MY2_ViMax, MY2_DiMax, dt, DZc, iDZc,
                          &fluxM_i, kt, nz, st,
                          MY2_afi, MY2_bfi, cmi, ckQi1, ckQi2, ckQi4);
    kt = my2_count_column(QNc, MY2_epsQs_sedi, ktop_sedi, st);
    if (kt >= 0)
        my2_sedi_1D<KMAX>(QNc, NNc, 3, DEc, iDEc, gfc, MY2_epsQ, MY2_epsN,
                          dms, MY2_VsMax, MY2_DsMax, dt, DZc, iDZc,
                          &fluxM_s, kt, nz, st,
                          MY2_afs, MY2_bfs, cms, ckQs1, ckQs2, iGS20);
    kt = my2_count_column(QGc, MY2_epsQg_sedi, ktop_sedi, st);
    if (kt >= 0)
        my2_sedi_1D<KMAX>(QGc, NGc, 4, DEc, iDEc, gfc, MY2_epsQ, MY2_epsN,
                          MY2_dmg, MY2_VgMax, MY2_DgMax, dt, DZc, iDZc,
                          &fluxM_g, kt, nz, st,
                          MY2_afg, MY2_bfg, cmg, ckQg1, ckQg2, ckQg4);
    kt = my2_count_column(QHc, MY2_epsQh_sedi, ktop_sedi, st);
    if (kt >= 0)
        my2_sedi_1D<KMAX>(QHc, NHc, 5, DEc, iDEc, gfc, MY2_epsQ, MY2_epsN,
                          MY2_dmh, MY2_VhMax, MY2_DhMax, dt, DZc, iDZc,
                          &fluxM_h, kt, nz, st,
                          MY2_afh, MY2_bfh, cmh, ckQh1, ckQh2, ckQh4);

    // Constraints on the snow size distribution (:3177-3200)
    for (int k = nz - 1; k >= 0; --k) {
        size_t o = (size_t)k * st;
        float qn = QNc[o], nn = NNc[o];
        if (qn > MY2_epsQ && nn > MY2_epsN) {
            float t1 = 1.0f / nn;
            float iLAMs = fmaxf(MY2_iLAMmin2,
                                my2_iLAMDA_x(DEc[o], qn, t1, iGS20, idms));
            t1 = fminf(nn / iLAMs, MY2_No_s_max);
            nn = powf(t1, dms / (1.0f + dms))
                 * powf(iGS20 * DEc[o] * qn, 1.0f / (1.0f + dms));
            t1 = 1.0f / nn;
            iLAMs = fmaxf(MY2_iLAMmin2,
                          my2_iLAMDA_x(DEc[o], qn, t1, iGS20, idms));
            float t2 = 1.0f / iLAMs;
            float t4 = 0.6f * MY2_lamdas_min;
            float t5 = 2.0f * t4;
            float t3 = t2 + t4 * powf(fmaxf(0.0f, t5 - t2) / t5, 2.0f);
            t3 = fmaxf(t3, MY2_lamdas_min);
            nn = nn * powf(t3 * iLAMs, dms);
            NNc[o] = nn;
        }
    }

    // Liquid-equivalent volume fluxes (:3205-3209)
    float RT_rn1 = fluxM_r * idew;
    float RT_rn2 = 0.0f;
    float RT_fr1 = 0.0f, RT_fr2 = 0.0f;
    float RT_sn1 = fluxM_i * idew;
    float RT_sn2 = fluxM_s * idew;
    float RT_sn3 = fluxM_g * idew;
    float RT_pe1 = fluxM_h * idew;
    float RT_pe2 = 0.0f;

    // precipDiag_ON = .true. (:3292-3330).  RT_peL (large hail) is computed
    // by WRF and discarded by the wrapper, so it is not reproduced.
    {
        size_t obot = 0;
        size_t otop = (size_t)(nz - 1) * st;
        // :3298-3304 -- N_r and DE are read at the TOP level while QR is
        // read at the BOTTOM.  That mixing is the Fortran's (nk vs kbot) and
        // is defined behaviour, so it is transcribed rather than "fixed".
        float N_r = NRc[otop];
        if (QRc[obot] > MY2_epsQ && N_r > MY2_epsN) {
            float Dm_r = powf(DEc[otop] * icmr * QRc[obot] / N_r,
                                MY2_thrd);
            if (Dm_r > MY2_Dr_large) { RT_rn2 = RT_rn1; RT_rn1 = 0.0f; }
        }
        if (T[col + otop] < MY2_TRPL) {                      // :3307
            RT_fr1 = RT_rn1; RT_rn1 = 0.0f;
            RT_fr2 = RT_rn2; RT_rn2 = 0.0f;
        }
        if (T[col + otop] > (MY2_TRPL + 5.0f)) {             // :3313
            RT_pe2 = RT_pe1; RT_pe1 = 0.0f;
        }
    }

    // mp_milbrandt2mom_driver :3676-3690 -- rates (m s-1) to mm/step
    float ms2mmstp = 1.0e+3f * dt;
    float rncv = (RT_rn1 + RT_rn2 + RT_fr1 + RT_fr2 + RT_sn1 + RT_sn2
                  + RT_sn3 + RT_pe1 + RT_pe2) * ms2mmstp;
    float sncv = (RT_sn1 + RT_sn2) * ms2mmstp;
    float hlcv = (RT_pe1 + RT_pe2) * ms2mmstp;
    float grcv = RT_sn3 * ms2mmstp;
    rainncv[col] = rncv;
    snowncv[col] = sncv;
    hailncv[col] = hlcv;
    graupelncv[col] = grcv;
    rainnc[col] = rainnc[col] + rncv;
    snownc[col] = snownc[col] + sncv;
    hailnc[col] = hailnc[col] + hlcv;
    graupelnc[col] = graupelnc[col] + grcv;
    sr[col] = (sncv + hlcv + grcv) / (rncv + 1.0e-12f);

    (void)Q; (void)QC; (void)NC;
}

extern "C" __global__
void milbrandt2_sediment_64(MY2_SEDIMENT_PARAMETERS)
{
    my2_sediment_impl<MY2_KMAX_SHALLOW>(MY2_SEDIMENT_ARGUMENTS);
}

extern "C" __global__
void milbrandt2_sediment_256(MY2_SEDIMENT_PARAMETERS)
{
    my2_sediment_impl<MY2_KMAX_GENERIC>(MY2_SEDIMENT_ARGUMENTS);
}

// ==========================================================================
// Diagnostics and the #/m3 -> #/kg conversion (:3336-3473)
// ==========================================================================
extern "C" __global__
void milbrandt2_diagnostics(
        const float* __restrict__ T, float* __restrict__ Q,
        const float* __restrict__ QC, const float* __restrict__ QR,
        const float* __restrict__ QI, const float* __restrict__ QN,
        const float* __restrict__ QG, const float* __restrict__ QH,
        float* __restrict__ NC, float* __restrict__ NR,
        float* __restrict__ NY, float* __restrict__ NN,
        float* __restrict__ NG, float* __restrict__ NH,
        const float* __restrict__ pres,
        float* __restrict__ ZET,
        const float* __restrict__ ck,
        int nz, int ny, int nx)
{
    long long n = (long long)nz * ny * nx;
    long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    float q = Q[gid];
    if (q < 0.0f) q = 0.0f;                                  // :3336
    Q[gid] = q;

    float t = T[gid], pr = pres[gid];
    float de = pr / (MY2_RGASD * t);                         // :3400
    float tmp9 = de * de;

    float N_c = NC[gid], N_r = NR[gid], N_i = NY[gid];
    float N_s = NN[gid], N_g = NG[gid], N_h = NH[gid];
    float qr = QR[gid], qi = QI[gid], qn = QN[gid];
    float qg = QG[gid], qh = QH[gid];

    float tmp1 = 0.f, tmp2 = 0.f, tmp3 = 0.f, tmp4 = 0.f, tmp5 = 0.f;
    if (qr > MY2_epsQ && N_r > MY2_epsN) tmp1 = cxr * Gzr * tmp9 * qr * qr
                                                / N_r;
    if (qi > MY2_epsQ && N_i > MY2_epsN) tmp2 = cxi * Gzi * tmp9 * qi * qi
                                                / N_i;
    if (qn > MY2_epsQ && N_s > MY2_epsN) tmp3 = cxi * Gzs * tmp9 * qn * qn
                                                / N_s;
    if (qg > MY2_epsQ && N_g > MY2_epsN) tmp4 = cxi * Gzg * tmp9 * qg * qg
                                                / N_g;
    if (qh > MY2_epsQ && N_h > MY2_epsN) tmp5 = cxi * Gzh * tmp9 * qh * qh
                                                / N_h;
    if (t > MY2_TRPL) {
        tmp2 = tmp2 * MY2_fdielec;
        tmp3 = tmp3 * MY2_fdielec;
        tmp4 = tmp4 * MY2_fdielec;
        tmp5 = tmp5 * MY2_fdielec;
    }
    float zet = tmp1 + tmp2 + tmp3 + tmp4 + tmp5;
    if (zet > 0.0f) {
        zet = 10.0f * log10f(zet * MY2_zfact);
    } else {
        zet = MY2_minZET;
    }
    ZET[gid] = fmaxf(zet, MY2_minZET);

    // :3467-3473 -- N back to #/kg on the FINAL temperature
    float ide = (MY2_RGASD * t) / pr;
    NC[gid] = N_c * ide;
    NR[gid] = N_r * ide;
    NY[gid] = N_i * ide;
    NN[gid] = N_s * ide;
    NG[gid] = N_g * ide;
    NH[gid] = N_h * ide;
    (void)QC;
}
