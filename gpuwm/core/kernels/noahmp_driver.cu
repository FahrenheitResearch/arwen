// gpuwm/core/kernels/noahmp_driver.cu
//
// CUDA half of the Noah-MP driver cold start.  Bitwise-equal to
// gpuwm/core/noahmp_driver.py and to the WRF v4.6.1 oracle
// (tree d66e442fccc04111067e29274c9f9eaccc3cef28,
//  sha256(phys/module_sf_noahmpdrv.F) =
//  9010a757da994ed8796c63ca97da354eaf60c5c732df4ea9acad5bc62a973890).
//
// Covers module_sf_noahmpdrv::SNOW_INIT (2340-2440) and
// module_sf_noahmpdrv::NOAHMP_INIT (1828-2335), one thread per column.
//
// Three rules make bitwise agreement possible and none is optional:
//
// 1. Every float32 operation uses an explicit rounding intrinsic
//    (AD/SU/MU/DV -> __fadd_rn/__fsub_rn/__fmul_rn/__fdiv_rn).  NVRTC defaults
//    to --fmad=true and gfortran on x86-64 emits no FMA at -O0, so a
//    contracted a*b+c on the device is a different number.  Do not
//    "simplify" these back to infix operators.
// 2. Every constant lives in __constant__ memory as a bit pattern.  ptxas
//    12.x's constant folder does not honour round-to-nearest-even when it
//    folds FP32 literals, so a table of literals can be mis-folded at compile
//    time; __fsub_rn pins the hardware, not the folder.
//    tests/test_fp32_tie_folding_gpu.py guards that.
// 3. `x**y` in NOAHMP_INIT's supercooled-liquid guess (2095-2096) is a glibc
//    powf call in the oracle.  CUDA's powf, __powf and exp2f are all different
//    functions and none can hold a max_ulp-0 gate, so this file uses r_pow --
//    the single glibc 2.39 transcription that lives in noahmp_leaves.cu.
//
// COMPOSITION: compiled after noahmp_leaves.cu (see noahmp_driver_gpu.py).
// There must be exactly one transcription of glibc's powf in the tree; two
// copies could drift and only one of them would be audited against
// glibc-libm-fp32.csv.
//
// Pinned option identity: iopt_run=3, iopt_crop=0, iopt_irr=0, iopt_irrm=0,
// sf_urban_physics=0, restart=.false.  Everything they kill is absent, not
// stubbed: the groundwater cold start (2299-2331), the Liu and gecros crop
// blocks (2201-2260), the irrigation cold start (2263-2278).  The host wrapper
// refuses any other option value, so this file does not branch on them.
//
// The two INTENT(OUT)-without-assignment behaviours are reproduced, not
// tidied: SNOW_INIT's ZSNSOXY snow slots above ISNOW keep the caller's values
// (2432-2435 never writes them), and NOAHMP_INIT's cropcat keeps the caller's
// value on a vegetated column (with iopt_crop=0 no statement writes it).

#ifndef AD
#define AD(a, b) __fadd_rn((a), (b))
#define SU(a, b) __fsub_rn((a), (b))
#define MU(a, b) __fmul_rn((a), (b))
#define DV(a, b) __fdiv_rn((a), (b))
#endif

#define DRV_NSNOW    3
#define DRV_NSOIL_MAX 9
#define DRV_NLAY_MAX (DRV_NSNOW + DRV_NSOIL_MAX)

// --------------------------------------------------------------------------
// Constants, as IEEE binary32 bit patterns.  See rule 2 above.
// --------------------------------------------------------------------------
__constant__ unsigned int C_DRV[35] = {
    0x00000000u,  // [ 0] 0.0
    0x3F800000u,  // [ 1] 1.0
    0xBF800000u,  // [ 2] -1.0
    0x40000000u,  // [ 3] 2.0
    0x3F000000u,  // [ 4] 0.5
    0x3CCCCCCDu,  // [ 5] 0.025          2381, 2385
    0x3D4CCCCDu,  // [ 6] 0.05           2385, 2388, 2394, 2398, 2403
    0x3DCCCCCDu,  // [ 7] 0.10           2388, 2392
    0x3E800000u,  // [ 8] 0.25           2392, 2396
    0x3EE66666u,  // [ 9] 0.45           2396, 2401
    0x3E4CCCCDu,  // [10] 0.20           2404
    0x48A2D780u,  // [11] HLICE  3.335e5 1989
    0x411CF5C3u,  // [12] GRAV   9.81    1990
    0x43889333u,  // [13] T0     273.15  1991
    0x43889312u,  // [14] 273.149        2094
    0x43839333u,  // [15] 263.15         2078
    0x44FA0000u,  // [16] 2000.0         2037-2039
    0x41200000u,  // [17] 10.0           2081
    0x3BA3D70Au,  // [18] 0.005          2022
    0x3C23D70Au,  // [19] 0.01           2082
    0x3CA3D70Au,  // [20] 0.02           2097
    0x3D4CCCCDu,  // [21] 0.05           2182, 2183
    0x3DCCCCCDu,  // [22] 0.1            2183
    0x447A0000u,  // [23] 1000.0         2185, 2187, 2190
    0x40400000u,  // [24] 3.0            2190
    0x43A6AAABu,  // [25] 1000./3.0      2190, folded here so ptxas cannot
    0x45992000u,  // [26] 4900.0         2146
    0x41C80000u,  // [27] 25.0           2148
    0x3E4CCCCDu,  // [28] 0.2            2148
    0x44FA0000u,  // [29] 2000.0         2124
    0x3F266666u,  // [30] 0.65           2140
    0x3DCCCCCDu,  // [31] 0.1            2133
    0x2EDBE6FFu,  // [32] 1E-10          2176, 2196
    0x43FA0000u,  // [33] 500.0          2192-2193
    0x447A0000u,  // [34] 1000.0         2194-2195
};

__device__ __forceinline__ float kd(int i)
{
    return __int_as_float(C_DRV[i]);
}

#define K_ZERO         kd(0)
#define K_ONE          kd(1)
#define K_NEG_ONE      kd(2)
#define K_TWO          kd(3)
#define K_HALF         kd(4)
#define K_D025         kd(5)
#define K_D05          kd(6)
#define K_D10          kd(7)
#define K_D25          kd(8)
#define K_D45          kd(9)
#define K_D20          kd(10)
#define K_HLICE        kd(11)
#define K_GRAV         kd(12)
#define K_T0           kd(13)
#define K_FREEZE       kd(14)
#define K_GLAC_TCAP    kd(15)
#define K_SWE_CAP      kd(16)
#define K_GLAC_SWE     kd(17)
#define K_SNOWH_PER_SWE kd(18)
#define K_GLAC_SNOWH   kd(19)
#define K_FK_FLOOR     kd(20)
#define K_LAI_FLOOR    kd(21)
#define K_SAI_PER_LAI  kd(22)
#define K_THOUSAND     kd(23)
#define K_MASSSAI      kd(25)
#define K_WA_COLD      kd(26)
#define K_TWENTYFIVE   kd(27)
#define K_PT2          kd(28)
#define K_EAH          kd(29)
#define K_ALBOLD       kd(30)
#define K_CHSTAR       kd(31)
#define K_GRAIN        kd(32)
#define K_ROOT         kd(33)
#define K_CARBON       kd(34)

// Fortran MIN/MAX on REAL(4).  Written out rather than fminf/fmaxf so the
// NaN handling is the comparison the Fortran performs, not libdevice's.
__device__ __forceinline__ float d_min(float a, float b)
{
    return (a < b) ? a : b;
}

__device__ __forceinline__ float d_max(float a, float b)
{
    return (a > b) ? a : b;
}

// --------------------------------------------------------------------------
// Flat host<->device layout.  One row per column; identical to the packing
// gpuwm/core/noahmp_driver_gpu.py builds from the oracle fixture.
// --------------------------------------------------------------------------

// SNOW_INIT --------------------------------------------------------------
#define SI_IX_NSNOW 0
#define SI_IX_NSOIL 1
#define SI_IX_STRIDE 2

#define SI_IN_SWE     0
#define SI_IN_TGXY    1
#define SI_IN_SNODEP  2
#define SI_IN_ZSOIL   3                          /* DRV_NSOIL_MAX */
#define SI_IN_ZSNSO   (SI_IN_ZSOIL + DRV_NSOIL_MAX)  /* DRV_NLAY_MAX */
#define SI_IN_TSNO    (SI_IN_ZSNSO + DRV_NLAY_MAX)   /* DRV_NSNOW */
#define SI_IN_SNICE   (SI_IN_TSNO + DRV_NSNOW)       /* DRV_NSNOW */
#define SI_IN_SNLIQ   (SI_IN_SNICE + DRV_NSNOW)      /* DRV_NSNOW */
#define SI_IN_STRIDE  (SI_IN_SNLIQ + DRV_NSNOW)

#define SI_OUT_ISNOW  0
#define SI_OUT_ZSNSO  1                          /* DRV_NLAY_MAX */
#define SI_OUT_TSNO   (SI_OUT_ZSNSO + DRV_NLAY_MAX)
#define SI_OUT_SNICE  (SI_OUT_TSNO + DRV_NSNOW)
#define SI_OUT_SNLIQ  (SI_OUT_SNICE + DRV_NSNOW)
#define SI_OUT_STRIDE (SI_OUT_SNLIQ + DRV_NSNOW)

// NOAHMP_INIT ------------------------------------------------------------
#define NI_IX_NSOIL    0
#define NI_IX_NSNOW    1
#define NI_IX_FNDSNOWH 2
#define NI_IX_VEGTYP   3
#define NI_IX_CROPCAT  4
#define NI_IX_SFURBAN  5
#define NI_IX_ISICE    6
#define NI_IX_ISURBAN  7
#define NI_IX_ISWATER  8
#define NI_IX_ISBARREN 9
#define NI_IX_LCZ      10                        /* 11 classes */
#define NI_IX_STRIDE   (NI_IX_LCZ + 11)

#define NI_IN_XICE     0
#define NI_IN_TSK      1
#define NI_IN_LAI      2
#define NI_IN_BEXP     3
#define NI_IN_SMCMAX   4
#define NI_IN_PSISAT   5
#define NI_IN_SLA      6
#define NI_IN_SLANAT   7
#define NI_IN_SNOW     8
#define NI_IN_SNOWH    9
#define NI_IN_DZS      10                        /* DRV_NSOIL_MAX */
#define NI_IN_TSLB     (NI_IN_DZS + DRV_NSOIL_MAX)
#define NI_IN_SMOIS    (NI_IN_TSLB + DRV_NSOIL_MAX)
#define NI_IN_ZSNSO    (NI_IN_SMOIS + DRV_NSOIL_MAX)  /* DRV_NLAY_MAX */
#define NI_IN_TSNO     (NI_IN_ZSNSO + DRV_NLAY_MAX)
#define NI_IN_SNICE    (NI_IN_TSNO + DRV_NSNOW)
#define NI_IN_SNLIQ    (NI_IN_SNICE + DRV_NSNOW)
#define NI_IN_STRIDE   (NI_IN_SNLIQ + DRV_NSNOW)

#define NI_OUT_SNOW    0
#define NI_OUT_SNOWH   1
#define NI_OUT_CANWAT  2
#define NI_OUT_TV      3
#define NI_OUT_TG      4
#define NI_OUT_CANICE  5
#define NI_OUT_CANLIQ  6
#define NI_OUT_EAH     7
#define NI_OUT_TAH     8
#define NI_OUT_CM      9
#define NI_OUT_CH      10
#define NI_OUT_FWET    11
#define NI_OUT_SNEQVO  12
#define NI_OUT_ALBOLD  13
#define NI_OUT_QSNOW   14
#define NI_OUT_QRAIN   15
#define NI_OUT_WSLAKE  16
#define NI_OUT_ZWT     17
#define NI_OUT_WA      18
#define NI_OUT_WT      19
#define NI_OUT_LAI     20
#define NI_OUT_XSAI    21
#define NI_OUT_LFMASS  22
#define NI_OUT_RTMASS  23
#define NI_OUT_STMASS  24
#define NI_OUT_WOOD    25
#define NI_OUT_STBLCP  26
#define NI_OUT_FASTCP  27
#define NI_OUT_GRAIN   28
#define NI_OUT_GDD     29
#define NI_OUT_T2MV    30
#define NI_OUT_T2MB    31
#define NI_OUT_CHSTAR  32
#define NI_OUT_QTDRAIN 33
#define NI_OUT_ISNOW   34
#define NI_OUT_CROPCAT 35
#define NI_OUT_TSLB    36                        /* DRV_NSOIL_MAX */
#define NI_OUT_SMOIS   (NI_OUT_TSLB + DRV_NSOIL_MAX)
#define NI_OUT_SH2O    (NI_OUT_SMOIS + DRV_NSOIL_MAX)
#define NI_OUT_ZSOIL   (NI_OUT_SH2O + DRV_NSOIL_MAX)
#define NI_OUT_ZSNSO   (NI_OUT_ZSOIL + DRV_NSOIL_MAX) /* DRV_NLAY_MAX */
#define NI_OUT_TSNO    (NI_OUT_ZSNSO + DRV_NLAY_MAX)
#define NI_OUT_SNICE   (NI_OUT_TSNO + DRV_NSNOW)
#define NI_OUT_SNLIQ   (NI_OUT_SNICE + DRV_NSNOW)
#define NI_OUT_STRIDE  (NI_OUT_SNLIQ + DRV_NSNOW)

// --------------------------------------------------------------------------
// SNOW_INIT core.  Layer subscripts are WRF's -NSNOW+1..NSOIL, mapped to
// 0..NSNOW+NSOIL-1 by `+ nsnow - 1`, so the body diffs against the Fortran.
// --------------------------------------------------------------------------
__device__ void noahmp_snow_init_core(
    int nsnow, int nsoil,
    const float *zsoil,     /* 1..nsoil        */
    float swe, float tgxy, float snodep,
    float *zsnso,           /* -nsnow+1..nsoil, in and out */
    float *tsno, float *snice, float *snliq,  /* -nsnow+1..0 */
    int *isnow_out)
{
#define ZS(k)  zsoil[(k) - 1]
#define ZSN(k) zsnso[(k) + nsnow - 1]
#define TSN(k) tsno[(k) + nsnow - 1]
#define SNI(k) snice[(k) + nsnow - 1]
#define SNL(k) snliq[(k) + nsnow - 1]

    // DZSNO is a local that only the SNODEP < 0.025 branch zeroes (2383);
    // the NaN seed makes a read of a slot WRF leaves undefined visible.
    float dzsno[DRV_NSNOW];
    float dzsnso[DRV_NLAY_MAX];
    for (int k = 0; k < DRV_NSNOW; ++k) {
        dzsno[k] = __int_as_float(0x7FC00000u);
    }
    for (int k = 0; k < DRV_NLAY_MAX; ++k) {
        dzsnso[k] = __int_as_float(0x7FC00000u);
    }
#define DZS_(k) dzsno[(k) + nsnow - 1]
#define DZSNSO_(k) dzsnso[(k) + nsnow - 1]

    int isnow;
    if (snodep < K_D025) {                                   // 2381
        isnow = 0;
        for (int k = 0; k < DRV_NSNOW; ++k) {
            dzsno[k] = K_ZERO;                               // 2383
        }
    } else if (snodep >= K_D025 && snodep <= K_D05) {        // 2385
        isnow = -1;
        DZS_(0) = snodep;                                    // 2387
    } else if (snodep > K_D05 && snodep <= K_D10) {          // 2388
        isnow = -2;
        DZS_(-1) = DV(snodep, K_TWO);                        // 2390
        DZS_(0) = DV(snodep, K_TWO);                         // 2391
    } else if (snodep > K_D10 && snodep <= K_D25) {          // 2392
        isnow = -2;
        DZS_(-1) = K_D05;                                    // 2394
        DZS_(0) = SU(snodep, DZS_(-1));                      // 2395
    } else if (snodep > K_D25 && snodep <= K_D45) {          // 2396
        isnow = -3;
        DZS_(-2) = K_D05;                                    // 2398
        DZS_(-1) = MU(K_HALF, SU(snodep, DZS_(-2)));         // 2399
        DZS_(0) = MU(K_HALF, SU(snodep, DZS_(-2)));          // 2400
    } else if (snodep > K_D45) {                             // 2401
        isnow = -3;
        DZS_(-2) = K_D05;                                    // 2403
        DZS_(-1) = K_D20;                                    // 2404
        DZS_(0) = SU(SU(snodep, DZS_(-1)), DZS_(-2));        // 2405
    } else {
        // 2407: wrf_error_fatal.  Unreachable for any non-NaN SNODEP.
        isnow = 0;
        for (int k = 0; k < DRV_NSNOW; ++k) {
            dzsno[k] = __int_as_float(0x7FC00000u);
        }
    }

    for (int k = -nsnow + 1; k <= 0; ++k) {                  // 2411-2413
        TSN(k) = K_ZERO;
        SNI(k) = K_ZERO;
        SNL(k) = K_ZERO;
    }
    for (int iz = isnow + 1; iz <= 0; ++iz) {                // 2414-2418
        TSN(iz) = tgxy;
        SNL(iz) = K_ZERO;
        // `1.00 * DZSNO(IZ) * (SWE/SNODEP)` associates left to right and
        // 1.0*x is exact for every finite binary32 x.
        SNI(iz) = MU(DZS_(iz), DV(swe, snodep));
    }

    for (int iz = isnow + 1; iz <= 0; ++iz) {                // 2421-2423
        DZSNSO_(iz) = SU(K_ZERO, DZS_(iz));
    }
    DZSNSO_(1) = ZS(1);                                      // 2426
    for (int iz = 2; iz <= nsoil; ++iz) {                    // 2427-2429
        DZSNSO_(iz) = SU(ZS(iz), ZS(iz - 1));
    }

    // 2432-2435.  Slots below ISNOW+1 are never written and keep the values
    // the caller handed in.
    ZSN(isnow + 1) = DZSNSO_(isnow + 1);
    for (int iz = isnow + 2; iz <= nsoil; ++iz) {
        ZSN(iz) = AD(ZSN(iz - 1), DZSNSO_(iz));
    }

    *isnow_out = isnow;

#undef DZSNSO_
#undef DZS_
#undef SNL
#undef SNI
#undef TSN
#undef ZSN
#undef ZS
}

// --------------------------------------------------------------------------
// 2095-2098: the frozen-soil supercooled-liquid initial guess.
//
// `-1/BEXP` is REAL division -- `-1` is a default INTEGER that Fortran
// converts to REAL before dividing -- so the exponent is -1.0/BEXP and the
// power is a scalar powf call, which is what `nm -u` shows on the pinned
// object.  glibc's powf is not correctly rounded, so r_pow (the single glibc
// 2.39 transcription, in noahmp_leaves.cu) is the only admissible source.
// --------------------------------------------------------------------------
__device__ float noahmp_supercooled_guess(float tslb, float bexp,
                                          float smcmax, float psisat,
                                          float smois)
{
    float base = MU(DV(K_HLICE, MU(K_GRAV, SU(K_ZERO, psisat))),
                    DV(SU(tslb, K_T0), tslb));
    float fk = MU(r_pow(base, DV(K_NEG_ONE, bexp)), smcmax);
    fk = d_max(fk, K_FK_FLOOR);
    return d_min(fk, smois);
}

// --------------------------------------------------------------------------
// NOAHMP_INIT core.
// --------------------------------------------------------------------------
__device__ void noahmp_init_core(
    int nsoil, int nsnow, int fndsnowh,
    int vegtyp, int cropcat_in, int sf_urban_physics,
    int isice, int isurban, int iswater, int isbarren, const int *lcz,
    const float *dzs,
    float xice, float tsk, float lai_in,
    float bexp, float smcmax, float psisat, float sla, float sla_natural,
    float snow_in, float snowh_in,
    const float *tslb_in, const float *smois_in,
    float *tslb, float *smois, float *sh2o, float *zsoil,
    float *zsnso, float *tsno, float *snice, float *snliq,
    float *out)
{
    for (int k = 0; k < nsoil; ++k) {
        tslb[k] = tslb_in[k];
        smois[k] = smois_in[k];
    }

    float snow = snow_in;
    float snowh = snowh_in;

    if (fndsnowh == 0) {                                     // 2017-2025
        snowh = MU(snow, K_SNOWH_PER_SWE);
    }
    if (snow > K_SWE_CAP) {                                  // 2037-2040
        snowh = DV(MU(snowh, K_SWE_CAP), snow);
        snow = K_SWE_CAP;
    }

    if (vegtyp == isice && xice <= K_ZERO) {                 // 2074
        for (int k = 0; k < nsoil; ++k) {
            smois[k] = K_ONE;                                // 2076
            sh2o[k] = K_ZERO;                                // 2077
            tslb[k] = d_min(tslb[k], K_GLAC_TCAP);           // 2078
        }
        snow = d_max(snow, K_GLAC_SWE);                      // 2081
        snowh = MU(snow, K_GLAC_SNOWH);                      // 2082
    } else {
        for (int k = 0; k < nsoil; ++k) {                    // 2089-2091
            if (smois[k] > smcmax) {
                smois[k] = smcmax;
            }
        }
        if (bexp > K_ZERO && smcmax > K_ZERO && psisat > K_ZERO) {  // 2092
            for (int k = 0; k < nsoil; ++k) {
                if (tslb[k] < K_FREEZE) {                    // 2094
                    sh2o[k] = noahmp_supercooled_guess(tslb[k], bexp, smcmax,
                                                       psisat, smois[k]);
                } else {
                    sh2o[k] = smois[k];                      // 2100
                }
            }
        } else {
            for (int k = 0; k < nsoil; ++k) {
                sh2o[k] = smois[k];                          // 2105
            }
        }
    }

    // 2114-2153.  One predicate drives all five 273.15 clamps.
    float cold_skin = tsk;
    if (snow > K_ZERO && tsk > K_T0) {
        cold_skin = K_T0;
    }
    float canwat = K_ZERO;                                   // 2121
    float wa = K_WA_COLD;                                    // 2146
    float zwt = SU(AD(K_TWENTYFIVE, K_TWO),
                   DV(DV(wa, K_THOUSAND), K_PT2));           // 2148

    // 2156-2178.  urbanpt_flag, then the land-use split.
    bool urbanpt = (vegtyp == isurban);
    for (int k = 0; k < 11; ++k) {
        if (vegtyp == lcz[k]) {
            urbanpt = true;
        }
    }
    bool bare = (vegtyp == isbarren) || (vegtyp == isice)
                || (sf_urban_physics == 0 && urbanpt)
                || (vegtyp == iswater);

    float lai, xsai, lfmass, stmass, rtmass, wood, stblcp, fastcp;
    int cropcat;
    if (bare) {
        lai = K_ZERO;                                        // 2168
        xsai = K_ZERO;                                       // 2169
        lfmass = K_ZERO;                                     // 2170
        stmass = K_ZERO;                                     // 2171
        rtmass = K_ZERO;                                     // 2172
        wood = K_ZERO;                                       // 2173
        stblcp = K_ZERO;                                     // 2174
        fastcp = K_ZERO;                                     // 2175
        cropcat = 0;                                         // 2178
    } else {
        lai = d_max(lai_in, K_LAI_FLOOR);                    // 2182
        xsai = d_max(MU(K_SAI_PER_LAI, lai), K_LAI_FLOOR);   // 2183
        float sla_used = urbanpt ? sla_natural : sla;        // 2184-2188
        float masslai = DV(K_THOUSAND, d_max(sla_used, K_ONE));
        lfmass = MU(lai, masslai);                           // 2189
        stmass = MU(xsai, K_MASSSAI);                        // 2191
        rtmass = K_ROOT;                                     // 2192
        wood = K_ROOT;                                       // 2193
        stblcp = K_CARBON;                                   // 2194
        fastcp = K_CARBON;                                   // 2195
        // iopt_crop = 0: nothing writes cropcat here, and the INTENT(OUT)
        // dummy is passed by reference, so the caller's value stands.
        cropcat = cropcat_in;
    }

    zsoil[0] = SU(K_ZERO, dzs[0]);                           // 2288
    for (int k = 1; k < nsoil; ++k) {                        // 2289-2291
        zsoil[k] = SU(zsoil[k - 1], dzs[k]);
    }

    int isnow = 0;
    noahmp_snow_init_core(nsnow, nsoil, zsoil, snow, cold_skin, snowh,
                          zsnso, tsno, snice, snliq, &isnow);  // 2295-2297

    out[NI_OUT_SNOW] = snow;
    out[NI_OUT_SNOWH] = snowh;
    out[NI_OUT_CANWAT] = canwat;
    out[NI_OUT_TV] = cold_skin;
    out[NI_OUT_TG] = cold_skin;
    out[NI_OUT_CANICE] = K_ZERO;
    out[NI_OUT_CANLIQ] = canwat;
    out[NI_OUT_EAH] = K_EAH;
    out[NI_OUT_TAH] = cold_skin;
    out[NI_OUT_CM] = K_ZERO;
    out[NI_OUT_CH] = K_ZERO;
    out[NI_OUT_FWET] = K_ZERO;
    out[NI_OUT_SNEQVO] = K_ZERO;
    out[NI_OUT_ALBOLD] = K_ALBOLD;
    out[NI_OUT_QSNOW] = K_ZERO;
    out[NI_OUT_QRAIN] = K_ZERO;
    out[NI_OUT_WSLAKE] = K_ZERO;
    out[NI_OUT_ZWT] = zwt;
    out[NI_OUT_WA] = wa;
    out[NI_OUT_WT] = wa;
    out[NI_OUT_LAI] = lai;
    out[NI_OUT_XSAI] = xsai;
    out[NI_OUT_LFMASS] = lfmass;
    out[NI_OUT_RTMASS] = rtmass;
    out[NI_OUT_STMASS] = stmass;
    out[NI_OUT_WOOD] = wood;
    out[NI_OUT_STBLCP] = stblcp;
    out[NI_OUT_FASTCP] = fastcp;
    out[NI_OUT_GRAIN] = K_GRAIN;
    out[NI_OUT_GDD] = K_ZERO;
    out[NI_OUT_T2MV] = cold_skin;
    out[NI_OUT_T2MB] = cold_skin;
    out[NI_OUT_CHSTAR] = K_CHSTAR;
    out[NI_OUT_QTDRAIN] = K_ZERO;
    out[NI_OUT_ISNOW] = (float)isnow;
    out[NI_OUT_CROPCAT] = (float)cropcat;
    for (int k = 0; k < nsoil; ++k) {
        out[NI_OUT_TSLB + k] = tslb[k];
        out[NI_OUT_SMOIS + k] = smois[k];
        out[NI_OUT_SH2O + k] = sh2o[k];
        out[NI_OUT_ZSOIL + k] = zsoil[k];
    }
    for (int k = 0; k < nsnow + nsoil; ++k) {
        out[NI_OUT_ZSNSO + k] = zsnso[k];
    }
    for (int k = 0; k < nsnow; ++k) {
        out[NI_OUT_TSNO + k] = tsno[k];
        out[NI_OUT_SNICE + k] = snice[k];
        out[NI_OUT_SNLIQ + k] = snliq[k];
    }
}

// --------------------------------------------------------------------------
// Entry points.  One thread per column.
// --------------------------------------------------------------------------
extern "C" __global__ void noahmp_driver_snow_init(
    const float *x, const int *ix, float *y, int ncase)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncase) {
        return;
    }
    const float *xin = x + (size_t)i * SI_IN_STRIDE;
    const int *ixin = ix + (size_t)i * SI_IX_STRIDE;
    float *out = y + (size_t)i * SI_OUT_STRIDE;

    int nsnow = ixin[SI_IX_NSNOW];
    int nsoil = ixin[SI_IX_NSOIL];

    float zsoil[DRV_NSOIL_MAX];
    float zsnso[DRV_NLAY_MAX];
    float tsno[DRV_NSNOW], snice[DRV_NSNOW], snliq[DRV_NSNOW];
    for (int k = 0; k < nsoil; ++k) {
        zsoil[k] = xin[SI_IN_ZSOIL + k];
    }
    for (int k = 0; k < nsnow + nsoil; ++k) {
        zsnso[k] = xin[SI_IN_ZSNSO + k];
    }
    for (int k = 0; k < nsnow; ++k) {
        tsno[k] = xin[SI_IN_TSNO + k];
        snice[k] = xin[SI_IN_SNICE + k];
        snliq[k] = xin[SI_IN_SNLIQ + k];
    }

    int isnow = 0;
    noahmp_snow_init_core(nsnow, nsoil, zsoil,
                          xin[SI_IN_SWE], xin[SI_IN_TGXY], xin[SI_IN_SNODEP],
                          zsnso, tsno, snice, snliq, &isnow);

    out[SI_OUT_ISNOW] = (float)isnow;
    for (int k = 0; k < nsnow + nsoil; ++k) {
        out[SI_OUT_ZSNSO + k] = zsnso[k];
    }
    for (int k = 0; k < nsnow; ++k) {
        out[SI_OUT_TSNO + k] = tsno[k];
        out[SI_OUT_SNICE + k] = snice[k];
        out[SI_OUT_SNLIQ + k] = snliq[k];
    }
}

extern "C" __global__ void noahmp_driver_noahmp_init(
    const float *x, const int *ix, float *y, int ncase)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ncase) {
        return;
    }
    const float *xin = x + (size_t)i * NI_IN_STRIDE;
    const int *ixin = ix + (size_t)i * NI_IX_STRIDE;
    float *out = y + (size_t)i * NI_OUT_STRIDE;

    int nsoil = ixin[NI_IX_NSOIL];
    int nsnow = ixin[NI_IX_NSNOW];

    float tslb[DRV_NSOIL_MAX], smois[DRV_NSOIL_MAX], sh2o[DRV_NSOIL_MAX];
    float zsoil[DRV_NSOIL_MAX], dzs[DRV_NSOIL_MAX];
    float zsnso[DRV_NLAY_MAX];
    float tsno[DRV_NSNOW], snice[DRV_NSNOW], snliq[DRV_NSNOW];
    for (int k = 0; k < nsoil; ++k) {
        dzs[k] = xin[NI_IN_DZS + k];
    }
    for (int k = 0; k < nsnow + nsoil; ++k) {
        zsnso[k] = xin[NI_IN_ZSNSO + k];
    }
    for (int k = 0; k < nsnow; ++k) {
        tsno[k] = xin[NI_IN_TSNO + k];
        snice[k] = xin[NI_IN_SNICE + k];
        snliq[k] = xin[NI_IN_SNLIQ + k];
    }

    noahmp_init_core(
        nsoil, nsnow, ixin[NI_IX_FNDSNOWH], ixin[NI_IX_VEGTYP],
        ixin[NI_IX_CROPCAT], ixin[NI_IX_SFURBAN],
        ixin[NI_IX_ISICE], ixin[NI_IX_ISURBAN], ixin[NI_IX_ISWATER],
        ixin[NI_IX_ISBARREN], ixin + NI_IX_LCZ,
        dzs,
        xin[NI_IN_XICE], xin[NI_IN_TSK], xin[NI_IN_LAI],
        xin[NI_IN_BEXP], xin[NI_IN_SMCMAX], xin[NI_IN_PSISAT],
        xin[NI_IN_SLA], xin[NI_IN_SLANAT],
        xin[NI_IN_SNOW], xin[NI_IN_SNOWH],
        xin + NI_IN_TSLB, xin + NI_IN_SMOIS,
        tslb, smois, sh2o, zsoil, zsnso, tsno, snice, snliq, out);
}
