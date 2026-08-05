// gpuwm/core/kernels/gf.cu
//
// CUDA half of the Grell-Freitas cumulus scheme (cu_physics = 3), WRF
// v4.6.1 -- ONE translation unit for the whole of GFDRV: the deep cloud
// model (CUP_gf), the shallow one (CUP_gf_sh), neg_check, and the driver's
// own mixed-precision preparation and output algebra, so GFDRV in gives
// GFDRV out on the device.  The deep half is documented here; the shallow
// and driver halves carry their own headers further down.
//
// The deep cloud model is transcribed statement for statement from the
// float32 CPU authority
// gpuwm/verify/gf_deep_ref.py + gf_deep_body.py::cup_gf_column, which is
// itself bitwise (max_ulp 0 with fzu pinned; see below for why the pin
// exists and why this kernel does not need it) against the byte-frozen
// module_cu_gf_deep.F / module_cu_gf_sh.F capture in
// gpuwm/data/gf/oracle/gf-deep-levels.csv / gf-deep-surface.csv.  One CUDA
// thread owns one complete column; column independence is measured, not
// assumed (gf-isolation.csv: 0 differing output words, packed vs solo, all
// 144 combinations), which is what makes one-column-per-thread the kernel
// shape rather than a guess.
//
// The three rules of every bitwise kernel in this tree, none optional:
//
// 1. Every float32 operation goes through __fadd_rn/__fsub_rn/__fmul_rn/
//    __fdiv_rn/__fsqrt_rn, every float64 one through __dadd_rn/__dsub_rn/
//    __dmul_rn/__ddiv_rn.  That pins the hardware rounding mode AND makes
//    nvcc's contraction pass a no-op, so --fmad=true cannot fuse a site the
//    Fortran did not fuse.
// 2. Every scheme constant lives in __constant__ memory as a bit pattern
//    (table GFC below, generated from the CPU reference's own constant set;
//    the gate re-derives every word from gpuwm.verify.gf_deep_ref at test
//    time and compares against gf_deep_const_dump).  ptxas 12.x's constant
//    folder does not honour round-to-nearest-even on FP32 constant-constant
//    arithmetic (tests/test_fp32_tie_folding_gpu.py), so no such arithmetic
//    may reach it: the three expressions WRF's front end folds -- log(10.),
//    log(6.1071)/log(10.), log(1013.246)/log(10.) -- plus xlv/cp, 1./xlv and
//    (1.-frh_thresh)**2 are pinned words, not spelled arithmetic.
// 3. Transcendentals are glibc 2.39's own algorithms, not CUDA's.  gfk_log/
//    gfk_exp/gfk_pow are the audited transcriptions this tree already holds
//    at max_ulp 0 (gpuwm/core/kernels/noahmp_leaves.cu r_log/r_exp/r_pow;
//    renamed here because translation units cannot share device code).
//    gfk_exp2/gfk_expm1/gfk_lgamma/gfk_gamma_product/gfk_tgamma are NEW
//    transcriptions -- sysdeps/ieee754/flt-32/e_exp2f.c, s_expm1f.c,
//    e_lgammaf_r.c (positive arm), dbl-64/gamma_productf.c, e_gammaf_r.c --
//    graded bitwise against the live glibc 2.39 sweep fixtures
//    gpuwm/data/gf/oracle/gf-libm-*.csv.
//
// WHY tgammaf IS TRANSCRIBED AT ALL (the one place this kernel is BETTER
// than the CPU reference): get_zu_zd_pdf_fim normalises the beta-function
// mass-flux shape with fzu = gamma(alpha+beta)/(gamma(alpha)*gamma(beta)).
// Perturb fzu by ONE ULP and xmb moves by up to 7.3 per cent through the
// xk = (xaa0-aa1)/mbdt cancellation (measured;
// tests/test_gf_deep_parity.py::
// test_a_one_ulp_massflux_shape_perturbation_moves_xmb_by_seven_percent).
// So gamma cannot be a tolerance question.  The CPU reference models
// tgammaf in float64 (0-4 ULP off glibc on the live arguments) and pins the
// oracle's captured fzu for its bitwise gate; this kernel computes glibc's
// own words with gfk_tgamma and needs no pin.  CUDA's builtin tgammaf is a
// DIFFERENT function and must not be used; the gate keeps a negative
// control proving the difference is real on the live argument set.
//
// Same story for powf: the CPU reference computes the CORRECTLY ROUNDED
// power and carries a measured 10-lane / 1-ULP zu divergence where glibc's
// powf (0.82 ULP worst case) lands on the far side of a rounding boundary
// (all 10 on ierr == 6 columns WRF rejects).  gfk_pow IS glibc, so those 10
// lanes are bitwise here and the divergence does not exist on the CUDA path.
//
// OWNER RULING (no inherited WRF bugs) as it applies to the deep body: the
// only defined-behaviour divergence on this path is get_inversion_layers,
// whose first-derivative loop in WRF reads t_cup(kend+8) out of bounds at
// kend > ktf-8.  Both the CPU reference and the oracle capture clamp kend to
// ktf-8 and COUNT the clamps; the count is 0 on the whole fixture, so the
// divergence is recorded and not exercised.  This kernel clamps identically
// and reports the count (ISCA_kinv_clamped).  An unclamped arm is not
// implementable on the GPU (the read is off the end of a local array) and
// is not offered.  The corrected-k22 ruling lives in the SHALLOW scheme
// (module_cu_gf_sh.F:373 MAXLOC section offset) and lands with that kernel,
// not this one.
//
// Precision inventory, carried from the reference docstrings:
//   * both constant sets are live in one call and are NOT harmonised: the
//     deep module's g=9.81/cp=1004./xlv=2.5e6/r_v=461. (K_G/K_CP/K_XLV/K_RV)
//     and the bare 9.81/1004./2.5e06 literals cup_env spells out itself
//     (K_LIT_*, same words, kept separate on purpose);
//   * satvap is base-10 powers through powf, NOT exp/log10;
//   * x**2/x**3 with integer literal exponents fold to multiply chains;
//     x**.3333 and the beta-shape powers stay powf;
//   * mconv is rebuilt on the cloud grid with the DEEP g, not the driver's
//     GFS 9.80665 (that one never enters this kernel).
//
// Capture layout: this is the stage-instrumented kernel, the CUDA analogue
// of tools/gf_wrf461_oracle/run_gf_stages.F90.  Field order in the three
// output slabs is the enum order below; tests/test_gf_deep_cuda.py carries
// the same lists and grades every slot against gf-deep-levels.csv /
// gf-deep-surface.csv (plus cupclw/cnvwt against gf-stage-levels.csv, the
// cross-fixture seam the CPU gate also checks).

#ifndef GF_KMAX
#define GF_KMAX 40
#endif
// 1-based level arrays: slot 0 exists and is never read (the reference's
// _a(nz)).  The +8 pad keeps get_inversion_layers' k_inv bookkeeping and
// the kbmax+2 argmax scan inside defined memory; the pad is zero-filled and
// no in-bounds word ever depends on it.
#define GF_KP (GF_KMAX + 9)

#define FADD(a, b) __fadd_rn((a), (b))
#define FSUB(a, b) __fsub_rn((a), (b))
#define FMUL(a, b) __fmul_rn((a), (b))
#define FDIV(a, b) __fdiv_rn((a), (b))
#define FSQRT(a)   __fsqrt_rn(a)
#define DADD(a, b) __dadd_rn((a), (b))
#define DSUB(a, b) __dsub_rn((a), (b))
#define DMUL(a, b) __dmul_rn((a), (b))
#define DDIV(a, b) __ddiv_rn((a), (b))

// Python max/min semantics, as the reference spells them: max(a, b) returns
// b only when b > a, so NaN in the SECOND slot never wins and +0/-0 keep
// the first operand.  The fixture has no NaN on any lane this kernel
// touches; the spelling is kept so that if one ever appears the kernel
// fails against the oracle instead of silently disagreeing with the CPU
// reference.
#define GMAX(a, b) (((b) > (a)) ? (b) : (a))
#define GMIN(a, b) (((b) < (a)) ? (b) : (a))
#define IMAX(a, b) (((b) > (a)) ? (b) : (a))
#define IMIN(a, b) (((b) < (a)) ? (b) : (a))

// ==========================================================================
// scheme constants (rule 2)
// ==========================================================================
// Generated by tools/gf_wrf461_oracle/gen_gf_const_table.py from the
// CPU reference's own constant set; the CUDA gate re-derives every
// word from gpuwm.verify.gf_deep_ref at test time and compares it
// against gf_deep_const_dump's output, so a wrong word here cannot
// survive the suite.
#define GF_NCONST 130
__constant__ unsigned int GFC[GF_NCONST] = {
    0x00000000u,  // [  0] ZERO = 0.
    0x3F800000u,  // [  1] ONE = 1.
    0x40000000u,  // [  2] TWO = 2.
    0x3F000000u,  // [  3] HALF = .5
    0xBF000000u,  // [  4] NEG_HALF = -.5
    0x411CF5C3u,  // [  5] G = g = 9.81
    0x447B0000u,  // [  6] CP = cp = 1004.
    0x4A189680u,  // [  7] XLV = xlv = 2.5e6
    0x43E68000u,  // [  8] RV = r_v = 461.
    0x3A83126Fu,  // [  9] C1 = c1 = .001
    0x3F666666u,  // [ 10] FRH_THRESH = frh_thresh = .9
    0x3F7851ECu,  // [ 11] RH_THRESH = rh_thresh = .97
    0x3C23D70Fu,  // [ 12] SIG_THRESH = (1.-frh_thresh)**2, folded
    0x3FC00000u,  // [ 13] BETAJB = betajb = 1.5
    0x3FC00000u,  // [ 14] FLUXTUNE = fluxtune = 1.5
    0x3F800000u,  // [ 15] PGCD = pgcd = 1.
    0x00800000u,  // [ 16] TINY32 = tiny(zws)
    0x438893D7u,  // [ 17] T273_155 = 273.155
    0xC1A00000u,  // [ 18] NEG20 = -20.
    0x4388947Bu,  // [ 19] T273_16 = 273.16
    0x43BA947Bu,  // [ 20] T373_16 = 373.16
    0xC1118E0Du,  // [ 21] SAT_A = -9.09718
    0x40644231u,  // [ 22] SAT_B = 3.56654
    0x3F607582u,  // [ 23] SAT_C = .876793
    0xC0FCE536u,  // [ 24] SAT_D = -7.90298
    0x40A0E608u,  // [ 25] SAT_E = 5.02808
    0x34145922u,  // [ 26] SAT_F = 1.3816E-07
    0x41358106u,  // [ 27] SAT_G = 11.344
    0x3C053F70u,  // [ 28] SAT_H = .0081328
    0xC05F7492u,  // [ 29] SAT_I = -3.49149
    0x41200000u,  // [ 30] TEN = 10.
    0x40135D8Eu,  // [ 31] LOG10 = log(10.), front-end fold
    0x3F492C7Cu,  // [ 32] LOG6_OVER_LOG10 = log(6.1071)/log(10.), fold
    0x40405DA2u,  // [ 33] LOG1013_OVER_LOG10 = log(1013.246)/log(10.), fold
    0x3F1F3B64u,  // [ 34] P622 = .622
    0x322BCC77u,  // [ 35] E1M8 = 1.e-8
    0x24E69595u,  // [ 36] E1M16 = 1.e-16
    0x411CF5C3u,  // [ 37] LIT_G = bare 9.81 literal in cup_env
    0x447B0000u,  // [ 38] LIT_CP = bare 1004. literal in cup_env
    0x4A189680u,  // [ 39] LIT_XLV = bare 2.5e06 literal in cup_env
    0x451BA0A3u,  // [ 40] XLV_OVER_CP = xlv/cp, front-end fold
    0x34D6BF95u,  // [ 41] ONE_OVER_XLV = 1./xlv, front-end fold
    0x3F1BA5E3u,  // [ 42] P608 = .608
    0x3ED1EB85u,  // [ 43] P41 = .41
    0x3F99999Au,  // [ 44] ONE_P2 = 1.2
    0x3EAAA64Cu,  // [ 45] P3333 = .3333 (NOT 1/3)
    0x42960000u,  // [ 46] CAP_MAXS = cap_maxs = 75.
    0x41A00000u,  // [ 47] TWENTY = 20.
    0x41C80000u,  // [ 48] TWENTYFIVE = 25.
    0x41800000u,  // [ 49] SIXTEEN = 16.
    0x3FC00000u,  // [ 50] ONE_P5 = 1.5
    0x38D1B717u,  // [ 51] P0001 = .0001 xland1 round
    0x3892CCF7u,  // [ 52] ENTR_BASE = 7.e-5
    0x3649539Cu,  // [ 53] ENTR_CSUM = 3.e-6
    0x3E4CCCCDu,  // [ 54] P2 = .2
    0x4048F5C3u,  // [ 55] PI314 = 3.14
    0x457A0000u,  // [ 56] ZKBMAX = zkbmax = 4000.
    0x447A0000u,  // [ 57] Z_DETR = z_detr = 1000.
    0x447A0000u,  // [ 58] DEPTH_MIN = depth_min = 1000.
    0x3DCCCCCDu,  // [ 59] P1 = .1
    0x41A00000u,  // [ 60] CAP_INC_DEEP = cap_max_increment = 20.
    0x43160000u,  // [ 61] P150 = 150.
    0x43480000u,  // [ 62] P200 = 200.
    0x3FA66666u,  // [ 63] ONE_P3 = 1.3
    0x3089705Fu,  // [ 64] E1M9 = 1.e-9
    0x358637BDu,  // [ 65] E1M6 = 1.e-6
    0x3F666666u,  // [ 66] P9 = .9
    0x3ECCCCCDu,  // [ 67] P4 = .4
    0x3C54FDF4u,  // [ 68] P013 = .013
    0x3F4CCCCDu,  // [ 69] P8 = .8
    0x40200000u,  // [ 70] TWO_P5 = 2.5
    0x40800000u,  // [ 71] FOUR = 4.
    0x40000000u,  // [ 72] LAMBAU = lambau = 2.
    0x457A0000u,  // [ 73] ZCUTDOWN = zcutdown = 4000.
    0x3F19999Au,  // [ 74] P6 = .6
    0x40A00000u,  // [ 75] FIVE = 5. (jmini floor is int)
    0x3D4CCCCDu,  // [ 76] BETA_DN_A = .05
    0x3AC49BA6u,  // [ 77] BETA_DN_B = .0015
    0x3CA3D70Au,  // [ 78] BETA_DN_MIN = .02
    0x3ECCCCCDu,  // [ 79] EDTMAX_LAND_A = .4
    0x3C75C28Fu,  // [ 80] EDTMAX_LAND_B = .015
    0x3B83126Fu,  // [ 81] C0_UP = c0 = .004
    0x43889333u,  // [ 82] T273_15 = 273.15
    0x3D8F5C29u,  // [ 83] P07 = .07
    0x40E00000u,  // [ 84] WMEAN = wmean = 7.
    0x3F80C7E3u,  // [ 85] TAU_A = 1.0061
    0x3C4985F0u,  // [ 86] TAU_B = 1.23e-2
    0x447A0000u,  // [ 87] THOUSAND = 1000.
    0x40800000u,  // [ 88] T_STAR = t_star = 4.
    0x3DCCCCCDu,  // [ 89] MBDT = mbdt = .1
    0x433E0000u,  // [ 90] T190 = 190.
    0x42C80000u,  // [ 91] HUNDRED = 100.
    0x447A0000u,  // [ 92] E1P3 = 1.e3
    0x3FCBA5E3u,  // [ 93] PEF_A = 1.591
    0x3F239581u,  // [ 94] PEF_B = .639
    0x3DC32CA5u,  // [ 95] PEF_C = .0953
    0x3BA2877Fu,  // [ 96] PEF_D = .00496
    0x3B57060Cu,  // [ 97] ZKBC_SCALE = 3.281e-3
    0x3CA3D70Au,  // [ 98] PREZK0 = .02
    0x40400000u,  // [ 99] THREE = 3.
    0x3F77A08Cu,  // [100] PZ_A = .96729352
    0xBF334997u,  // [101] PZ_B = -.70034167
    0x3E26127Du,  // [102] PZ_C = .162179896
    0xBC4DF18Eu,  // [103] PZ_D = -1.2569798E-2
    0x39E03F9Bu,  // [104] PZ_E = 4.2772E-4
    0xB6B6893Fu,  // [105] PZ_F = -5.44E-6
    0x4019999Au,  // [106] PREZK25 = 2.4
    0x3E4CCCCDu,  // [107] EDT_EINC = .2 einc factor
    0x3DCCCCCDu,  // [108] EDTMIN = edtmin = .1
    0xBC23D70Au,  // [109] NEG_P01 = -.01
    0x3C23D70Au,  // [110] E1M2 = 1.e-2
    0x3727C5ACu,  // [111] E1M5 = 1.e-5
    0x3A83126Fu,  // [112] E1M3 = 1.e-3
    0x41400000u,  // [113] TWELVE = closure_n = 12.
    0x43960148u,  // [114] THRESH_DEEP = 300.01
    0x47A8C000u,  // [115] SECONDS_DAY = 86400.
    0x4E6E6B28u,  // [116] BIG1E9 = 1.e9
    0x42C80000u,  // [117] P100 = 100. (800 hPa slot)
    0x43960000u,  // [118] P300 = 300. (550 hPa slot)
    0x322BCC77u,  // [119] QMIN = 1.e-08 moisture floor
    0x43480000u,  // [120] TMIN = 200. temperature floor
    0x3C23D70Au,  // [121] MB = mb conversion
    0x43160000u,  // [122] CCN150 = ccn = 150.
    0x4314028Fu,  // [123] T148_01 = 148.01 (neg_check shallow threshold)
    0x453B8000u,  // [124] ZKBMAX_SH = zkbmax = 3000. (shallow)
    0x42FA0000u,  // [125] CAP_MAXS_SH = cap_maxs = 125. (shallow)
    0x38BCBE62u,  // [126] ENTR_SH = entr_rate = 9.e-5 (shallow)
    0x40133333u,  // [127] TWO_P3 = 2.3 (shallow entrainment profile)
    0x3CF5C28Fu,  // [128] P03 = .03 (shallow w* closure member)
    0x43810000u,  // [129] TCRIT = 258. (driver's ice/water split)
};
#define K_ZERO __uint_as_float(GFC[0])
#define K_ONE __uint_as_float(GFC[1])
#define K_TWO __uint_as_float(GFC[2])
#define K_HALF __uint_as_float(GFC[3])
#define K_NEG_HALF __uint_as_float(GFC[4])
#define K_G __uint_as_float(GFC[5])
#define K_CP __uint_as_float(GFC[6])
#define K_XLV __uint_as_float(GFC[7])
#define K_RV __uint_as_float(GFC[8])
#define K_C1 __uint_as_float(GFC[9])
#define K_FRH_THRESH __uint_as_float(GFC[10])
#define K_RH_THRESH __uint_as_float(GFC[11])
#define K_SIG_THRESH __uint_as_float(GFC[12])
#define K_BETAJB __uint_as_float(GFC[13])
#define K_FLUXTUNE __uint_as_float(GFC[14])
#define K_PGCD __uint_as_float(GFC[15])
#define K_TINY32 __uint_as_float(GFC[16])
#define K_T273_155 __uint_as_float(GFC[17])
#define K_NEG20 __uint_as_float(GFC[18])
#define K_T273_16 __uint_as_float(GFC[19])
#define K_T373_16 __uint_as_float(GFC[20])
#define K_SAT_A __uint_as_float(GFC[21])
#define K_SAT_B __uint_as_float(GFC[22])
#define K_SAT_C __uint_as_float(GFC[23])
#define K_SAT_D __uint_as_float(GFC[24])
#define K_SAT_E __uint_as_float(GFC[25])
#define K_SAT_F __uint_as_float(GFC[26])
#define K_SAT_G __uint_as_float(GFC[27])
#define K_SAT_H __uint_as_float(GFC[28])
#define K_SAT_I __uint_as_float(GFC[29])
#define K_TEN __uint_as_float(GFC[30])
#define K_LOG10 __uint_as_float(GFC[31])
#define K_LOG6_OVER_LOG10 __uint_as_float(GFC[32])
#define K_LOG1013_OVER_LOG10 __uint_as_float(GFC[33])
#define K_P622 __uint_as_float(GFC[34])
#define K_E1M8 __uint_as_float(GFC[35])
#define K_E1M16 __uint_as_float(GFC[36])
#define K_LIT_G __uint_as_float(GFC[37])
#define K_LIT_CP __uint_as_float(GFC[38])
#define K_LIT_XLV __uint_as_float(GFC[39])
#define K_XLV_OVER_CP __uint_as_float(GFC[40])
#define K_ONE_OVER_XLV __uint_as_float(GFC[41])
#define K_P608 __uint_as_float(GFC[42])
#define K_P41 __uint_as_float(GFC[43])
#define K_ONE_P2 __uint_as_float(GFC[44])
#define K_P3333 __uint_as_float(GFC[45])
#define K_CAP_MAXS __uint_as_float(GFC[46])
#define K_TWENTY __uint_as_float(GFC[47])
#define K_TWENTYFIVE __uint_as_float(GFC[48])
#define K_SIXTEEN __uint_as_float(GFC[49])
#define K_ONE_P5 __uint_as_float(GFC[50])
#define K_P0001 __uint_as_float(GFC[51])
#define K_ENTR_BASE __uint_as_float(GFC[52])
#define K_ENTR_CSUM __uint_as_float(GFC[53])
#define K_P2 __uint_as_float(GFC[54])
#define K_PI314 __uint_as_float(GFC[55])
#define K_ZKBMAX __uint_as_float(GFC[56])
#define K_Z_DETR __uint_as_float(GFC[57])
#define K_DEPTH_MIN __uint_as_float(GFC[58])
#define K_P1 __uint_as_float(GFC[59])
#define K_CAP_INC_DEEP __uint_as_float(GFC[60])
#define K_P150 __uint_as_float(GFC[61])
#define K_P200 __uint_as_float(GFC[62])
#define K_ONE_P3 __uint_as_float(GFC[63])
#define K_E1M9 __uint_as_float(GFC[64])
#define K_E1M6 __uint_as_float(GFC[65])
#define K_P9 __uint_as_float(GFC[66])
#define K_P4 __uint_as_float(GFC[67])
#define K_P013 __uint_as_float(GFC[68])
#define K_P8 __uint_as_float(GFC[69])
#define K_TWO_P5 __uint_as_float(GFC[70])
#define K_FOUR __uint_as_float(GFC[71])
#define K_LAMBAU __uint_as_float(GFC[72])
#define K_ZCUTDOWN __uint_as_float(GFC[73])
#define K_P6 __uint_as_float(GFC[74])
#define K_FIVE __uint_as_float(GFC[75])
#define K_BETA_DN_A __uint_as_float(GFC[76])
#define K_BETA_DN_B __uint_as_float(GFC[77])
#define K_BETA_DN_MIN __uint_as_float(GFC[78])
#define K_EDTMAX_LAND_A __uint_as_float(GFC[79])
#define K_EDTMAX_LAND_B __uint_as_float(GFC[80])
#define K_C0_UP __uint_as_float(GFC[81])
#define K_T273_15 __uint_as_float(GFC[82])
#define K_P07 __uint_as_float(GFC[83])
#define K_WMEAN __uint_as_float(GFC[84])
#define K_TAU_A __uint_as_float(GFC[85])
#define K_TAU_B __uint_as_float(GFC[86])
#define K_THOUSAND __uint_as_float(GFC[87])
#define K_T_STAR __uint_as_float(GFC[88])
#define K_MBDT __uint_as_float(GFC[89])
#define K_T190 __uint_as_float(GFC[90])
#define K_HUNDRED __uint_as_float(GFC[91])
#define K_E1P3 __uint_as_float(GFC[92])
#define K_PEF_A __uint_as_float(GFC[93])
#define K_PEF_B __uint_as_float(GFC[94])
#define K_PEF_C __uint_as_float(GFC[95])
#define K_PEF_D __uint_as_float(GFC[96])
#define K_ZKBC_SCALE __uint_as_float(GFC[97])
#define K_PREZK0 __uint_as_float(GFC[98])
#define K_THREE __uint_as_float(GFC[99])
#define K_PZ_A __uint_as_float(GFC[100])
#define K_PZ_B __uint_as_float(GFC[101])
#define K_PZ_C __uint_as_float(GFC[102])
#define K_PZ_D __uint_as_float(GFC[103])
#define K_PZ_E __uint_as_float(GFC[104])
#define K_PZ_F __uint_as_float(GFC[105])
#define K_PREZK25 __uint_as_float(GFC[106])
#define K_EDT_EINC __uint_as_float(GFC[107])
#define K_EDTMIN __uint_as_float(GFC[108])
#define K_NEG_P01 __uint_as_float(GFC[109])
#define K_E1M2 __uint_as_float(GFC[110])
#define K_E1M5 __uint_as_float(GFC[111])
#define K_E1M3 __uint_as_float(GFC[112])
#define K_TWELVE __uint_as_float(GFC[113])
#define K_THRESH_DEEP __uint_as_float(GFC[114])
#define K_SECONDS_DAY __uint_as_float(GFC[115])
#define K_BIG1E9 __uint_as_float(GFC[116])
#define K_P100 __uint_as_float(GFC[117])
#define K_P300 __uint_as_float(GFC[118])
#define K_QMIN __uint_as_float(GFC[119])
#define K_TMIN __uint_as_float(GFC[120])
#define K_MB __uint_as_float(GFC[121])
#define K_CCN150 __uint_as_float(GFC[122])
#define K_T148_01 __uint_as_float(GFC[123])
#define K_ZKBMAX_SH __uint_as_float(GFC[124])
#define K_CAP_MAXS_SH __uint_as_float(GFC[125])
#define K_ENTR_SH __uint_as_float(GFC[126])
#define K_TWO_P3 __uint_as_float(GFC[127])
#define K_P03 __uint_as_float(GFC[128])
#define K_TCRIT __uint_as_float(GFC[129])

// GFDRV's own constants are real(8) initialised from default-real literals;
// the fixture measured (gf_ref.py's docstring table) that the arithmetic
// behaves as the HONEST doubles -- float64(9.80665), not
// float64(float32(9.80665)), whose stored-word reading is the trap
// gf-pow-probe.txt's gfsconst rows record.  Double literals parse correctly
// rounded, so these are those words: 0x40239D013A92A305 etc.
#define GFS_G_D  9.80665e0
#define GFS_CP_D 1.0046e3
#define GFS_XLV_D 2.5e6

// ==========================================================================
// glibc 2.39 float32 transcendentals (rule 3)
// ==========================================================================
// e_logf_data.c
__device__ const double GFK_LOGF_INVC[16] = {
    0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0,
    0x1.3c995b0b80385p+0, 0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0,
    0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0, 0x1.0953f419900a7p+0,
    0x1p+0,               0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
    0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1,
    0x1.767dcf5534862p-1 };
__device__ const double GFK_LOGF_LOGC[16] = {
    -0x1.57bf7808caadep-2, -0x1.2bef0a7c06ddbp-2, -0x1.01eae7f513a67p-2,
    -0x1.b31d8a68224e9p-3, -0x1.6574f0ac07758p-3, -0x1.1aa2bc79c81p-3,
    -0x1.a4e76ce8c0e5ep-4, -0x1.1973c5a611cccp-4, -0x1.252f438e10c1ep-5,
     0x0p+0,                0x1.aa5aa5df25984p-5,  0x1.c5e53aa362eb4p-4,
     0x1.526e57720db08p-3,  0x1.bc2860d22477p-3,   0x1.1058bc8a07ee1p-2,
     0x1.4043057b6ee09p-2 };
#define GFK_LOGF_LN2 0x1.62e42fefa39efp-1
#define GFK_LOGF_A0 (-0x1.00ea348b88334p-2)
#define GFK_LOGF_A1 (0x1.5575b0be00b6ap-2)
#define GFK_LOGF_A2 (-0x1.ffffef20a4123p-2)

// e_exp2f_data.c, shared by expf / exp2f / powf.  EXP2F_TABLE_BITS = 5.
__device__ const unsigned long long GFK_EXP2F_TAB[32] = {
    0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL,
    0x3fef9301d0125b51ULL, 0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL,
    0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL, 0x3fef06fe0a31b715ULL,
    0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
    0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL,
    0x3feea47eb03a5585ULL, 0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL,
    0x3feea11473eb0187ULL, 0x3feea589994cce13ULL, 0x3feeace5422aa0dbULL,
    0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL,
    0x3fef3720dcef9069ULL, 0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL,
    0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL };
#define GFK_EXP2F_P0 0x1.c6af84b912394p-5
#define GFK_EXP2F_P1 0x1.ebfce50fac4f3p-3
#define GFK_EXP2F_P2 0x1.62e42ff0c52d6p-1
#define GFK_EXP2F_SHIFT 0x1.8p+52
#define GFK_EXP2F_SHIFT_SCALED (0x1.8p+52 / 32.0)

// e_powf_log2_data.c.  POWF_SCALE is 1.0 (TOINT_INTRINSICS = 0 on x86-64).
__device__ const double GFK_POWF_INVC[16] = {
    0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0,
    0x1.3c995b0b80385p+0, 0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0,
    0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0, 0x1.0953f419900a7p+0,
    0x1p+0,               0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
    0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1,
    0x1.767dcf5534862p-1 };
__device__ const double GFK_POWF_LOGC[16] = {
    -0x1.efec65b963019p-2, -0x1.b0b6832d4fca4p-2, -0x1.7418b0a1fb77bp-2,
    -0x1.39de91a6dcf7bp-2, -0x1.01d9bf3f2b631p-2, -0x1.97c1d1b3b7afp-3,
    -0x1.2f9e393af3c9fp-3, -0x1.960cbbf788d5cp-4, -0x1.a6f9db6475fcep-5,
     0x0p+0,                0x1.338ca9f24f53dp-4,  0x1.476a9543891bap-3,
     0x1.e840b4ac4e4d2p-3,  0x1.40645f0c6651cp-2,  0x1.88e9c2c1b9ff8p-2,
     0x1.ce0a44eb17bccp-2 };
__device__ const double GFK_POWF_A[5] = {
     0x1.27616c9496e0bp-2, -0x1.71969a075c67ap-2,  0x1.ec70a6ca7baddp-2,
    -0x1.7154748bef6c8p-1,  0x1.71547652ab82bp+0 };

// glibc 2.39 sysdeps/ieee754/flt-32/e_logf.c
__device__ float gfk_log(float x)
{
    unsigned int ix = __float_as_uint(x);
    if (ix == 0x3f800000u) return 0.0f;
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u) {
        if (ix * 2u == 0u) return __int_as_float(0xff800000);
        if (ix == 0x7f800000u) return x;
        if ((ix & 0x80000000u) || ix * 2u >= 0xff000000u)
            return __int_as_float(0x7fc00000);
        ix = __float_as_uint(FMUL(x, 8388608.0f));   /* 0x1p23f */
        ix -= 23u << 23;
    }
    unsigned int tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    int k = (int)tmp >> 23;
    unsigned int iz = ix - (tmp & 0xff800000u);
    double z = (double)__uint_as_float(iz);
    double r = DSUB(DMUL(z, GFK_LOGF_INVC[i]), 1.0);
    double y0 = DADD(GFK_LOGF_LOGC[i], DMUL((double)k, GFK_LOGF_LN2));
    double r2 = DMUL(r, r);
    double y = DADD(DMUL(GFK_LOGF_A1, r), GFK_LOGF_A2);
    y = DADD(DMUL(GFK_LOGF_A0, r2), y);
    y = DADD(DMUL(y, r2), DADD(y0, r));
    return __double2float_rn(y);
}

// Round a double to binary32, INCLUDING into the subnormal range.  On this
// toolchain `__double2float_rn` flushes a subnormal result to zero (CuPy
// appends -ftz=true and the compiler emits the flush after the conversion),
// while glibc's expf/powf do produce subnormals.  The correctly rounded
// subnormal is recovered exactly: m * 2^-149 scaling is exact in binary64
// over this band and rint rounds ties to even.  Same function, same
// reasoning, as noahmp_leaves.cu::nmp_d2f_rn -- the sm_120 FP32-DAZ
// countermeasure this repo has already proven.
__device__ float gfk_d2f_rn(double y)
{
    double a = fabs(y);
    if (a > 0.0 && a < 1.1754943508222875e-38) {   /* 0x1p-126 */
        double scaled = rint(a * 7.1362384635297994e+44);   /* 2^149 */
        unsigned int m = (unsigned int)scaled;
        unsigned int s = (__double_as_longlong(y) < 0LL) ? 0x80000000u : 0u;
        return __uint_as_float(s | m);
    }
    return __double2float_rn(y);
}

// The 32-entry exp2 core shared by glibc's expf, exp2f and powf.
__device__ double gfk_exp2_core(double xd, double shift,
                                double p0, double p1, double p2,
                                unsigned long long sign_bias)
{
    double kd = DADD(xd, shift);
    unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd = DSUB(kd, shift);
    double r = DSUB(xd, kd);
    unsigned long long t = GFK_EXP2F_TAB[ki & 31ULL];
    t += (ki + sign_bias) << (52 - 5);
    double s = __longlong_as_double((long long)t);
    double z = DADD(DMUL(p0, r), p1);
    double r2 = DMUL(r, r);
    double y = DADD(DMUL(p2, r), 1.0);
    y = DADD(DMUL(z, r2), y);
    return DMUL(y, s);
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_expf.c
__device__ float gfk_exp(float x)
{
    unsigned int abstop = (__float_as_uint(x) >> 20) & 0x7ffu;
    if (abstop >= ((__float_as_uint(88.0f)) >> 20)) {
        if (__float_as_uint(x) == 0xff800000u) return 0.0f;
        if (abstop >= (0x7f800000u >> 20)) return FADD(x, x);
        if (x > __int_as_float(0x42b17218)) return __int_as_float(0x7f800000);
        if (x < -__int_as_float(0x42cff1b4)) return 0.0f;
    }
    double xd = (double)x;
    double z = DMUL(0x1.71547652b82fep+0 * 32.0, xd);
    return gfk_d2f_rn(gfk_exp2_core(
        z, GFK_EXP2F_SHIFT,
        GFK_EXP2F_P0 / 32.0 / 32.0 / 32.0,
        GFK_EXP2F_P1 / 32.0 / 32.0,
        GFK_EXP2F_P2 / 32.0, 0ULL));
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_exp2f.c -- the identical core with
// the pre-scaled shift and unscaled polynomial.  tgammaf's Stirling arm is
// the only caller in this kernel and hands it |x| <= ~2.6, but the special
// cases are transcribed anyway so the sweep can grade the whole function.
__device__ float gfk_exp2(float x)
{
    unsigned int abstop = (__float_as_uint(x) >> 20) & 0x7ffu;
    if (abstop >= ((__float_as_uint(128.0f)) >> 20)) {
        if (__float_as_uint(x) == 0xff800000u) return 0.0f;
        if (abstop >= (0x7f800000u >> 20)) return FADD(x, x);
        if (x > 0.0f) return __int_as_float(0x7f800000);
        if (x <= -150.0f) return 0.0f;
    }
    double xd = (double)x;
    return gfk_d2f_rn(gfk_exp2_core(
        xd, GFK_EXP2F_SHIFT_SCALED,
        GFK_EXP2F_P0, GFK_EXP2F_P1, GFK_EXP2F_P2, 0ULL));
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_powf.c log2_inline
__device__ double gfk_powf_log2(unsigned int ix)
{
    unsigned int tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    unsigned int top = tmp & 0xff800000u;
    unsigned int iz = ix - top;
    int k = (int)top >> 23;
    double z = (double)__uint_as_float(iz);
    double r = DSUB(DMUL(z, GFK_POWF_INVC[i]), 1.0);
    double y0 = DADD(GFK_POWF_LOGC[i], (double)k);
    double r2 = DMUL(r, r);
    double y = DADD(DMUL(GFK_POWF_A[0], r), GFK_POWF_A[1]);
    double p = DADD(DMUL(GFK_POWF_A[2], r), GFK_POWF_A[3]);
    double r4 = DMUL(r2, r2);
    double q = DADD(DMUL(GFK_POWF_A[4], r), y0);
    q = DADD(DMUL(p, r2), q);
    return DADD(DMUL(y, r4), q);
}

__device__ int gfk_checkint(unsigned int iy)
{
    int e = (int)(iy >> 23 & 0xffu);
    if (e < 0x7f) return 0;
    if (e > 0x7f + 23) return 2;
    if (iy & ((1u << (0x7f + 23 - e)) - 1u)) return 0;
    if (iy & (1u << (0x7f + 23 - e))) return 1;
    return 2;
}

__device__ bool gfk_zeroinfnan(unsigned int ix)
{
    return 2u * ix - 1u >= 2u * 0x7f800000u - 1u;
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_powf.c, full special-case surface:
// the beta-shape powers reach kratio == 0 and kratio == 1 on every column
// (powf(0, +y) and powf(+0-adjacent bases), so the zero/int paths are live.
__device__ float gfk_pow(float x, float y)
{
    unsigned int sign_bias = 0u;
    unsigned int ix = __float_as_uint(x);
    unsigned int iy = __float_as_uint(y);
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u || gfk_zeroinfnan(iy)) {
        if (gfk_zeroinfnan(iy)) {
            if (2u * iy == 0u) return 1.0f;
            if (ix == 0x3f800000u) return 1.0f;
            if (2u * ix > 2u * 0x7f800000u || 2u * iy > 2u * 0x7f800000u)
                return FADD(x, y);
            if (2u * ix == 2u * 0x3f800000u) return 1.0f;
            if ((2u * ix < 2u * 0x3f800000u) == !(iy & 0x80000000u))
                return 0.0f;
            return FMUL(y, y);
        }
        if (gfk_zeroinfnan(ix)) {
            float x2 = FMUL(x, x);
            if ((ix & 0x80000000u) && gfk_checkint(iy) == 1) x2 = -x2;
            return (iy & 0x80000000u) ? FDIV(1.0f, x2) : x2;
        }
        if (ix & 0x80000000u) {
            int yint = gfk_checkint(iy);
            if (yint == 0) return __int_as_float(0x7fc00000);
            if (yint == 1) sign_bias = 1u << (5 + 11);
            ix &= 0x7fffffffu;
        }
        if (ix < 0x00800000u) {
            ix = __float_as_uint(FMUL(x, 8388608.0f)) & 0x7fffffffu;
            ix -= 23u << 23;
        }
    }
    double logx = gfk_powf_log2(ix);
    double ylogx = DMUL((double)y, logx);
    unsigned int hi = (unsigned int)
        (((unsigned long long)__double_as_longlong(ylogx) >> 47) & 0xffffULL);
    if (hi >= (unsigned int)
            (((unsigned long long)__double_as_longlong(126.0) >> 47) & 0xffffULL)) {
        if (ylogx > 0x1.fffffffd1d571p+6)
            return sign_bias ? __int_as_float(0xff800000)
                             : __int_as_float(0x7f800000);
        if (ylogx <= -150.0) return sign_bias ? -0.0f : 0.0f;
    }
    return gfk_d2f_rn(
        gfk_exp2_core(ylogx, GFK_EXP2F_SHIFT_SCALED,
                      GFK_EXP2F_P0, GFK_EXP2F_P1, GFK_EXP2F_P2,
                      (unsigned long long)sign_bias));
}

// --------------------------------------------------------------------------
// glibc 2.39 sysdeps/ieee754/flt-32/s_expm1f.c (SunPro FP32 kernel).  No
// ifunc variant exists on x86-64, so every operation is a plain float32
// op in the written association order -- FMUL/FADD/FSUB/FDIV, never FMA.
// Constant words verified against the decimal literals, not the source
// comments (the C_ATAN precedent: glibc comments have lied before).
// --------------------------------------------------------------------------
#define EM1_HUGE   __uint_as_float(0x7149F2CAu)   /* 1.0e+30 */
#define EM1_OTHR   __uint_as_float(0x42B17180u)   /* o_threshold */
#define EM1_LN2HI  __uint_as_float(0x3F317180u)
#define EM1_LN2LO  __uint_as_float(0x3717F7D1u)
#define EM1_IVLN2  __uint_as_float(0x3FB8AA3Bu)
#define EM1_Q1     __uint_as_float(0xBD088889u)
#define EM1_Q2     __uint_as_float(0x3AD00D01u)
#define EM1_Q3     __uint_as_float(0xB8A670CDu)
#define EM1_Q4     __uint_as_float(0x36867E54u)
#define EM1_Q5     __uint_as_float(0xB457EDBBu)
#define EM1_TINYM1 __uint_as_float(0xBF800000u)   /* tiny - one == -1.0f */

__device__ float gfk_expm1(float x)
{
    float y, hi, lo, c, t, e, hxs, hfx, r1;
    int k, xsb;
    unsigned int hx = __float_as_uint(x);
    xsb = (int)(hx & 0x80000000u);
    hx &= 0x7fffffffu;
    c = 0.0f;

    if (hx >= 0x4195b844u) {                 /* |x| >= 27*ln2 */
        if (hx >= 0x42b17218u) {             /* |x| >= 88.721... */
            if (hx > 0x7f800000u) return FADD(x, x);            /* NaN */
            if (hx == 0x7f800000u)
                return (xsb == 0) ? x : -1.0f;                  /* +-inf */
            if (x > EM1_OTHR) return FMUL(EM1_HUGE, EM1_HUGE);  /* oflow */
        }
        if (xsb != 0) return EM1_TINYM1;     /* x < -27*ln2: -1 */
    }

    if (hx > 0x3eb17218u) {                  /* |x| > 0.5 ln2 */
        if (hx < 0x3F851592u) {              /* |x| < 1.5 ln2 */
            if (xsb == 0) { hi = FSUB(x, EM1_LN2HI); lo = EM1_LN2LO;  k = 1; }
            else          { hi = FADD(x, EM1_LN2HI); lo = -EM1_LN2LO; k = -1; }
        } else {
            float kf = FADD(FMUL(EM1_IVLN2, x), (xsb == 0) ? 0.5f : -0.5f);
            k  = (int)kf;
            t  = (float)k;
            hi = FSUB(x, FMUL(t, EM1_LN2HI));
            lo = FMUL(t, EM1_LN2LO);
        }
        x = FSUB(hi, lo);
        c = FSUB(FSUB(hi, x), lo);
    } else if (hx < 0x33000000u) {           /* |x| < 2**-25 */
        t = FADD(EM1_HUGE, x);
        return FSUB(x, FSUB(t, FADD(EM1_HUGE, x)));
    } else {
        k = 0;
    }

    hfx = FMUL(0.5f, x);
    hxs = FMUL(x, hfx);
    r1 = FADD(1.0f, FMUL(hxs, FADD(EM1_Q1, FMUL(hxs, FADD(EM1_Q2,
             FMUL(hxs, FADD(EM1_Q3, FMUL(hxs, FADD(EM1_Q4,
             FMUL(hxs, EM1_Q5))))))))));
    t = FSUB(3.0f, FMUL(r1, hfx));
    e = FMUL(hxs, FDIV(FSUB(r1, t), FSUB(6.0f, FMUL(x, t))));
    if (k == 0) return FSUB(x, FSUB(FMUL(x, e), hxs));
    e = FSUB(FMUL(x, FSUB(e, c)), c);
    e = FSUB(e, hxs);
    if (k == -1) return FSUB(FMUL(0.5f, FSUB(x, e)), 0.5f);
    if (k == 1) {
        if (x < -0.25f) return FMUL(-2.0f, FSUB(e, FADD(x, 0.5f)));
        return FADD(1.0f, FMUL(2.0f, FSUB(x, e)));
    }
    if (k <= -2 || k > 56) {
        y = FSUB(1.0f, FSUB(e, x));
        y = __uint_as_float(__float_as_uint(y) + ((unsigned int)k << 23));
        return FSUB(y, 1.0f);
    }
    if (k < 23) {
        t = __uint_as_float(0x3f800000u - (0x1000000u >> k)); /* 1-2^-k */
        y = FSUB(t, FSUB(e, x));
        y = __uint_as_float(__float_as_uint(y) + ((unsigned int)k << 23));
    } else {
        t = __uint_as_float((unsigned int)(0x7f - k) << 23);  /* 2^-k */
        y = FSUB(x, FADD(e, t));
        y = FADD(y, 1.0f);
        y = __uint_as_float(__float_as_uint(y) + ((unsigned int)k << 23));
    }
    return y;
}

// --------------------------------------------------------------------------
// glibc 2.39 sysdeps/ieee754/flt-32/e_lgammaf_r.c, POSITIVE arm only.  The
// negative-x machinery (sin_pif, __lgamma_negf) is deliberately absent:
// tgammaf's callers in this kernel hand it x in (0.5, 2.5) and the sweep
// grades (0.4, 2.6) plus the (2, 8) tail; a negative or non-finite argument
// returns NaN rather than a value this kernel cannot vouch for.  No ifunc
// variant exists on x86-64: plain float32 ops, written association order.
// Every word below was verified against the decimal literal.
// --------------------------------------------------------------------------
#define LG_A0  __uint_as_float(0x3D9E233Fu)
#define LG_A1  __uint_as_float(0x3EA51A66u)
#define LG_A2  __uint_as_float(0x3D89F001u)
#define LG_A3  __uint_as_float(0x3CA89915u)
#define LG_A4  __uint_as_float(0x3BF2027Eu)
#define LG_A5  __uint_as_float(0x3B3D6EC6u)
#define LG_A6  __uint_as_float(0x3A9C54A1u)
#define LG_A7  __uint_as_float(0x3A05B634u)
#define LG_A8  __uint_as_float(0x39679767u)
#define LG_A9  __uint_as_float(0x38E28445u)
#define LG_A10 __uint_as_float(0x37D383A2u)
#define LG_A11 __uint_as_float(0x383C2C75u)
#define LG_TC  __uint_as_float(0x3FBB16C3u)
#define LG_TF  __uint_as_float(0xBDF8CDCDu)
#define LG_TT  __uint_as_float(0x31E61C52u)
// tc - one, folded on the host: FP32 constant-constant subtraction must not
// reach ptxas (rule 2).  Exact: 1.4616321325 - 1 loses no mantissa bits.
#define LG_TCM1 __uint_as_float(0x3EEC5B0Cu)
#define LG_T0  __uint_as_float(0x3EF7B95Eu)
#define LG_T1  __uint_as_float(0xBE17213Cu)
#define LG_T2  __uint_as_float(0x3D845A15u)
#define LG_T3  __uint_as_float(0xBD064D47u)
#define LG_T4  __uint_as_float(0x3C93373Du)
#define LG_T5  __uint_as_float(0xBC28FCFEu)
#define LG_T6  __uint_as_float(0x3BC7E707u)
#define LG_T7  __uint_as_float(0xBB7177FEu)
#define LG_T8  __uint_as_float(0x3B141699u)
#define LG_T9  __uint_as_float(0xBAB7F476u)
#define LG_T10 __uint_as_float(0x3A66F867u)
#define LG_T11 __uint_as_float(0xBA0D3085u)
#define LG_T12 __uint_as_float(0x39A57B6Bu)
#define LG_T13 __uint_as_float(0xB9A3F927u)
#define LG_T14 __uint_as_float(0x39AFE9F7u)
#define LG_U0  __uint_as_float(0xBD9E233Fu)
#define LG_U1  __uint_as_float(0x3F2200F4u)
#define LG_U2  __uint_as_float(0x3FBA3AE7u)
#define LG_U3  __uint_as_float(0x3F7A4BB2u)
#define LG_U4  __uint_as_float(0x3E6A7578u)
#define LG_U5  __uint_as_float(0x3C5B3C5Eu)
#define LG_V1  __uint_as_float(0x401D2EBEu)
#define LG_V2  __uint_as_float(0x4008392Du)
#define LG_V3  __uint_as_float(0x3F44EFDFu)
#define LG_V4  __uint_as_float(0x3DD572AFu)
#define LG_V5  __uint_as_float(0x3B52D5DBu)
#define LG_S0  __uint_as_float(0xBD9E233Fu)
#define LG_S1  __uint_as_float(0x3E5C245Au)
#define LG_S2  __uint_as_float(0x3EA6CC7Au)
#define LG_S3  __uint_as_float(0x3E15DCE6u)
#define LG_S4  __uint_as_float(0x3CDA40E4u)
#define LG_S5  __uint_as_float(0x3AF135B4u)
#define LG_S6  __uint_as_float(0x3805FF67u)
#define LG_R1  __uint_as_float(0x3FB22D3Bu)
#define LG_R2  __uint_as_float(0x3F38D0C5u)
#define LG_R3  __uint_as_float(0x3E300F6Eu)
#define LG_R4  __uint_as_float(0x3C98BF54u)
#define LG_R5  __uint_as_float(0x3A4BEED6u)
#define LG_R6  __uint_as_float(0x36F5D7BDu)
#define LG_W0  __uint_as_float(0x3ED67F1Du)
#define LG_W1  __uint_as_float(0x3DAAAAABu)
#define LG_W2  __uint_as_float(0xBB360B61u)
#define LG_W3  __uint_as_float(0x3A500CFDu)
#define LG_W4  __uint_as_float(0xBA1C065Cu)
#define LG_W5  __uint_as_float(0x3A5B3DD2u)
#define LG_W6  __uint_as_float(0xBAD5C4E8u)

__device__ float gfk_lgamma_pos(float x)
{
    unsigned int hx = __float_as_uint(x);
    int ix = (int)(hx & 0x7fffffffu);
    if ((int)hx < 0 ) return __int_as_float(0x7fc00000);  /* not ported */
    if (ix >= 0x7f800000) return FMUL(x, x);
    if (ix == 0) return FDIV(1.0f, fabsf(x));
    if (ix < 0x30800000) return -gfk_log(x);   /* |x| < 2**-30 */

    float t, y, z, p, p1, p2, p3, q, r, w;
    int i;
    y = 0.0f; i = 0;
    if (ix == 0x3f800000 || ix == 0x40000000) {
        r = 0.0f;
    } else if (ix < 0x40000000) {            /* x < 2.0 */
        if (ix <= 0x3f666666) {              /* lgamma(x) = lgamma(x+1)-log(x) */
            r = -gfk_log(x);
            if (ix >= 0x3f3b4a20)      { y = FSUB(1.0f, x); i = 0; }
            else if (ix >= 0x3e6d3308) { y = FSUB(x, LG_TCM1); i = 1; }
            else                       { y = x; i = 2; }
        } else {
            r = 0.0f;
            if (ix >= 0x3fdda618)      { y = FSUB(2.0f, x); i = 0; }
            else if (ix >= 0x3F9da620) { y = FSUB(x, LG_TC); i = 1; }
            else                       { y = FSUB(x, 1.0f); i = 2; }
        }
        switch (i) {
        case 0:
            z = FMUL(y, y);
            p1 = FADD(LG_A0, FMUL(z, FADD(LG_A2, FMUL(z, FADD(LG_A4,
                 FMUL(z, FADD(LG_A6, FMUL(z, FADD(LG_A8,
                 FMUL(z, LG_A10))))))))));
            p2 = FMUL(z, FADD(LG_A1, FMUL(z, FADD(LG_A3, FMUL(z, FADD(LG_A5,
                 FMUL(z, FADD(LG_A7, FMUL(z, FADD(LG_A9,
                 FMUL(z, LG_A11)))))))))));
            p = FADD(FMUL(y, p1), p2);
            r = FADD(r, FSUB(p, FMUL(0.5f, y)));
            break;
        case 1:
            z = FMUL(y, y);
            w = FMUL(z, y);
            p1 = FADD(LG_T0, FMUL(w, FADD(LG_T3, FMUL(w, FADD(LG_T6,
                 FMUL(w, FADD(LG_T9, FMUL(w, LG_T12))))))));
            p2 = FADD(LG_T1, FMUL(w, FADD(LG_T4, FMUL(w, FADD(LG_T7,
                 FMUL(w, FADD(LG_T10, FMUL(w, LG_T13))))))));
            p3 = FADD(LG_T2, FMUL(w, FADD(LG_T5, FMUL(w, FADD(LG_T8,
                 FMUL(w, FADD(LG_T11, FMUL(w, LG_T14))))))));
            p = FSUB(FMUL(z, p1), FSUB(LG_TT, FMUL(w, FADD(p2, FMUL(y, p3)))));
            r = FADD(r, FADD(LG_TF, p));
            break;
        case 2:
            p1 = FMUL(y, FADD(LG_U0, FMUL(y, FADD(LG_U1, FMUL(y, FADD(LG_U2,
                 FMUL(y, FADD(LG_U3, FMUL(y, FADD(LG_U4,
                 FMUL(y, LG_U5)))))))))));
            p2 = FADD(1.0f, FMUL(y, FADD(LG_V1, FMUL(y, FADD(LG_V2,
                 FMUL(y, FADD(LG_V3, FMUL(y, FADD(LG_V4,
                 FMUL(y, LG_V5))))))))));
            r = FADD(r, FADD(FMUL(-0.5f, y), FDIV(p1, p2)));
            break;
        }
    } else if (ix < 0x41000000) {            /* x < 8.0 */
        i = (int)x;
        y = FSUB(x, (float)i);
        p = FMUL(y, FADD(LG_S0, FMUL(y, FADD(LG_S1, FMUL(y, FADD(LG_S2,
            FMUL(y, FADD(LG_S3, FMUL(y, FADD(LG_S4, FMUL(y, FADD(LG_S5,
            FMUL(y, LG_S6)))))))))))));
        q = FADD(1.0f, FMUL(y, FADD(LG_R1, FMUL(y, FADD(LG_R2, FMUL(y,
            FADD(LG_R3, FMUL(y, FADD(LG_R4, FMUL(y, FADD(LG_R5,
            FMUL(y, LG_R6))))))))))));
        r = FADD(FMUL(0.5f, y), FDIV(p, q));
        z = 1.0f;
        switch (i) {                          /* lgamma(1+s) = log(s)+lgamma(s) */
        case 7: z = FMUL(z, FADD(y, 6.0f));   /* FALLTHRU */
        case 6: z = FMUL(z, FADD(y, 5.0f));   /* FALLTHRU */
        case 5: z = FMUL(z, FADD(y, 4.0f));   /* FALLTHRU */
        case 4: z = FMUL(z, FADD(y, 3.0f));   /* FALLTHRU */
        case 3: z = FMUL(z, FADD(y, 2.0f));
                r = FADD(r, gfk_log(z));
                break;
        }
    } else if (ix < 0x4c800000) {            /* 8.0 <= x < 2**26 */
        t = gfk_log(x);
        z = FDIV(1.0f, x);
        y = FMUL(z, z);
        w = FADD(LG_W0, FMUL(z, FADD(LG_W1, FMUL(y, FADD(LG_W2, FMUL(y,
            FADD(LG_W3, FMUL(y, FADD(LG_W4, FMUL(y, FADD(LG_W5,
            FMUL(y, LG_W6))))))))))));
        r = FADD(FMUL(FSUB(x, 0.5f), FSUB(t, 1.0f)), w);
    } else {
        r = FMUL(x, FSUB(gfk_log(x), 1.0f));
    }
    return r;
}

// --------------------------------------------------------------------------
// glibc 2.39 sysdeps/ieee754/dbl-64/gamma_productf.c: the float
// __gamma_productf computed in double, which is what x86-64 links.
// --------------------------------------------------------------------------
__device__ float gfk_gamma_product(float x, float x_eps, int n, float *eps)
{
    double x_full = DADD((double)x, (double)x_eps);
    double ret = x_full;
    for (int i = 1; i < n; i++)
        ret = DMUL(ret, DADD(x_full, (double)i));
    float fret = __double2float_rn(ret);
    *eps = __double2float_rn(DDIV(DSUB(ret, (double)fret), (double)fret));
    return fret;
}

// --------------------------------------------------------------------------
// glibc 2.39 sysdeps/ieee754/flt-32/e_gammaf_r.c, positive arm.  x <= 0,
// NaN and inf return NaN (not ported -- the scheme cannot produce them:
// alpha = (tunning*(beta-2)+1)/(1-tunning) with tunning clamped to
// [.2, .9] and beta in {1.3, 2.5, 4.} keeps every argument in
// [1.06, 32.2)).  x >= 36 overflows to +inf exactly as glibc's
// FLT_MAX*FLT_MAX does.
// --------------------------------------------------------------------------
#define GAM_C0     __uint_as_float(0x3DAAAAABu)   /* 0x1.555556p-4 */
#define GAM_C1     __uint_as_float(0xBB360B61u)   /* -0xb.60b61p-12 */
#define GAM_C2     __uint_as_float(0x3A500D01u)   /* 0x3.403404p-12 */
#define GAM_SQRT12 __uint_as_float(0x3F3504F3u)   /* M_SQRT1_2f */
#define GAM_TWOPI  __uint_as_float(0x40C90FDBu)   /* 2*M_PIf, host-folded */

__device__ float gfk_gammaf_positive(float x, int *exp2_adj)
{
    if (x < 0.5f) {
        *exp2_adj = 0;
        return FDIV(gfk_exp(gfk_lgamma_pos(FADD(x, 1.0f))), x);
    } else if (x <= 1.5f) {
        *exp2_adj = 0;
        return gfk_exp(gfk_lgamma_pos(x));
    } else if (x < 2.5f) {
        *exp2_adj = 0;
        float x_adj = FSUB(x, 1.0f);
        return FMUL(gfk_exp(gfk_lgamma_pos(x_adj)), x_adj);
    } else {
        float eps = 0.0f;
        float x_eps = 0.0f;
        float x_adj = x;
        float prod = 1.0f;
        if (x < 4.0f) {
            float n = ceilf(FSUB(4.0f, x));
            x_adj = FADD(x, n);
            x_eps = FSUB(x, FSUB(x_adj, n));
            prod = gfk_gamma_product(FSUB(x_adj, n), x_eps, (int)n, &eps);
        }
        float exp_adj = -eps;
        float x_adj_int = roundf(x_adj);
        float x_adj_frac = FSUB(x_adj, x_adj_int);
        int x_adj_log2;
        float x_adj_mant = frexpf(x_adj, &x_adj_log2);
        if (x_adj_mant < GAM_SQRT12) {
            x_adj_log2--;
            x_adj_mant = FMUL(x_adj_mant, 2.0f);
        }
        *exp2_adj = x_adj_log2 * (int)x_adj_int;
        float ret = FDIV(FMUL(FMUL(FMUL(
            gfk_pow(x_adj_mant, x_adj),
            gfk_exp2(FMUL((float)x_adj_log2, x_adj_frac))),
            gfk_exp(-x_adj)),
            FSQRT(FDIV(GAM_TWOPI, x_adj))),
            prod);
        exp_adj = FADD(exp_adj, FMUL(x_eps, gfk_log(x_adj)));
        float bsum = GAM_C2;
        float x_adj2 = FMUL(x_adj, x_adj);
        bsum = FADD(FDIV(bsum, x_adj2), GAM_C1);
        bsum = FADD(FDIV(bsum, x_adj2), GAM_C0);
        exp_adj = FADD(exp_adj, FDIV(bsum, x_adj));
        return FADD(ret, FMUL(ret, gfk_expm1(exp_adj)));
    }
}

__device__ float gfk_tgamma(float x)
{
    unsigned int hx = __float_as_uint(x);
    if ((hx & 0x80000000u) || (hx & 0x7fffffffu) >= 0x7f800000u
        || (hx & 0x7fffffffu) == 0u)
        return __int_as_float(0x7fc00000);   /* outside the ported domain */
    if (x >= 36.0f)
        return __int_as_float(0x7f800000);   /* FLT_MAX*FLT_MAX overflow */
    int exp2_adj;
    float tret = gfk_gammaf_positive(x, &exp2_adj);
    float ret = scalbnf(tret, exp2_adj);
    // glibc's isinf/iszero fixups return the same +inf / +0 words for a
    // positive argument; nothing further to do.
    return ret;
}

// ==========================================================================
// the deep cloud model, module_cu_gf_deep.F, one procedure per function
// ==========================================================================

// module_cu_gf_deep.F:3646-3668.  Goff-Gratch in base-10 POWERS: 10**x is
// powf(10., x) and log(x)/log(10.) is a runtime logf over a folded constant,
// never exp/log10f.
__device__ float gfd_satvap(float temp2)
{
    float temp = FSUB(temp2, K_T273_155);
    if (temp < K_NEG20) {
        float toot = FDIV(K_T273_16, temp2);
        float toto = FDIV(K_ONE, toot);
        float e = FMUL(K_SAT_A, FSUB(toot, K_ONE));
        e = FSUB(e, FMUL(K_SAT_B, FDIV(gfk_log(toot), K_LOG10)));
        e = FADD(e, FMUL(K_SAT_C, FSUB(K_ONE, toto)));
        e = FADD(e, K_LOG6_OVER_LOG10);
        return gfk_pow(K_TEN, e);
    }
    float tsot = FDIV(K_T373_16, temp2);
    float ewlog = FMUL(K_SAT_D, FSUB(tsot, K_ONE));
    ewlog = FADD(ewlog, FMUL(K_SAT_E, FDIV(gfk_log(tsot), K_LOG10)));
    float ewlog2 = FSUB(ewlog,
        FMUL(K_SAT_F,
             FSUB(gfk_pow(K_TEN, FMUL(K_SAT_G, FSUB(K_ONE, FDIV(K_ONE, tsot)))),
                  K_ONE)));
    float ewlog3 = FADD(ewlog2,
        FMUL(K_SAT_H, FSUB(gfk_pow(K_TEN, FMUL(K_SAT_I, FSUB(tsot, K_ONE))),
                           K_ONE)));
    float ewlog4 = FADD(ewlog3, K_LOG1013_OVER_LOG10);
    return gfk_pow(K_TEN, ewlog4);
}

// module_cu_gf_deep.F:2141-2269 with itest = -1: the height stack passes
// through untouched and he is still assigned (guard is itest .le. 0).
__device__ void gfd_cup_env(const float *z, const float *t, const float *q,
                            const float *p, float *qes, float *he, float *hes,
                            int nz)
{
    for (int k = 1; k <= nz; k++) {
        float e = gfd_satvap(t[k]);
        qes[k] = FDIV(FMUL(K_P622, e), GMAX(K_E1M8, FSUB(p[k], e)));
        if (qes[k] <= K_E1M16) qes[k] = K_E1M16;
        if (qes[k] < q[k]) qes[k] = q[k];
    }
    for (int k = 1; k <= nz; k++) {
        he[k] = FADD(FADD(FMUL(K_LIT_G, z[k]), FMUL(K_LIT_CP, t[k])),
                     FMUL(K_LIT_XLV, q[k]));
        hes[k] = FADD(FADD(FMUL(K_LIT_G, z[k]), FMUL(K_LIT_CP, t[k])),
                      FMUL(K_LIT_XLV, qes[k]));
        if (he[k] >= hes[k]) he[k] = hes[k];
    }
}

// module_cu_gf_deep.F:2272-2371.  z_cup(1)/p_cup(1) are assigned twice and
// the second assignment -- z1 and psur -- wins.
__device__ void gfd_cup_env_clev(const float *t, const float *qes,
    const float *q, const float *he, const float *hes, const float *z,
    const float *p, float psur, float z1, int nz,
    float *qes_cup, float *q_cup, float *he_cup, float *hes_cup,
    float *z_cup, float *p_cup, float *gamma_cup, float *t_cup)
{
    for (int k = 0; k < GF_KP; k++) {
        qes_cup[k] = K_ZERO; q_cup[k] = K_ZERO; he_cup[k] = K_ZERO;
        hes_cup[k] = K_ZERO; z_cup[k] = K_ZERO; p_cup[k] = K_ZERO;
        gamma_cup[k] = K_ZERO; t_cup[k] = K_ZERO;
    }
    for (int k = 2; k <= nz; k++) {
        qes_cup[k] = FMUL(K_HALF, FADD(qes[k - 1], qes[k]));
        q_cup[k] = FMUL(K_HALF, FADD(q[k - 1], q[k]));
        hes_cup[k] = FMUL(K_HALF, FADD(hes[k - 1], hes[k]));
        he_cup[k] = FMUL(K_HALF, FADD(he[k - 1], he[k]));
        if (he_cup[k] > hes_cup[k]) he_cup[k] = hes_cup[k];
        z_cup[k] = FMUL(K_HALF, FADD(z[k - 1], z[k]));
        p_cup[k] = FMUL(K_HALF, FADD(p[k - 1], p[k]));
        t_cup[k] = FMUL(K_HALF, FADD(t[k - 1], t[k]));
        gamma_cup[k] = FMUL(
            FMUL(K_XLV_OVER_CP,
                 FDIV(K_XLV, FMUL(FMUL(K_RV, t_cup[k]), t_cup[k]))),
            qes_cup[k]);
    }
    qes_cup[1] = qes[1];
    q_cup[1] = q[1];
    hes_cup[1] = FADD(FADD(FMUL(K_LIT_G, z1), FMUL(K_LIT_CP, t[1])),
                      FMUL(K_LIT_XLV, qes[1]));
    he_cup[1] = FADD(FADD(FMUL(K_LIT_G, z1), FMUL(K_LIT_CP, t[1])),
                     FMUL(K_LIT_XLV, q[1]));
    z_cup[1] = z1;
    p_cup[1] = psur;
    t_cup[1] = t[1];
    gamma_cup[1] = FMUL(
        FMUL(K_XLV_OVER_CP,
             FDIV(K_XLV, FMUL(FMUL(K_RV, t_cup[1]), t_cup[1]))),
        qes_cup[1]);
}

// module_cu_gf_deep.F:3670-3693.  A 3-point mean below k22.
__device__ float gfd_get_cloud_bc(const float *arr, int k22, float add,
                                  int has_add)
{
    int local_order = IMIN(k22, 3);
    float x = K_ZERO;
    for (int i = 1; i <= local_order; i++)
        x = FADD(x, arr[k22 - i + 1]);
    x = FDIV(x, (float)local_order);
    if (has_add) x = FADD(x, add);
    return x;
}

// Fortran MAXLOC over a section: the FIRST maximum, strict >.
__device__ int gfd_maxloc(const float *arr, int lo, int hi)
{
    int best = lo;
    float bv = arr[lo];
    for (int k = lo + 1; k <= hi; k++)
        if (arr[k] > bv) { bv = arr[k]; best = k; }
    return best;
}

__device__ int gfd_maxloc_int(const int *arr, int lo, int hi)
{
    int best = lo;
    int bv = arr[lo];
    for (int k = lo + 1; k <= hi; k++)
        if (arr[k] > bv) { bv = arr[k]; best = k; }
    return best;
}

// _minloc over |arr|, as get_inversion_layers uses it.
__device__ int gfd_minloc_abs(const float *arr, int lo, int hi)
{
    int best = lo;
    float bv = fabsf(arr[lo]);
    for (int k = lo + 1; k <= hi; k++) {
        float v = fabsf(arr[k]);
        if (v < bv) { bv = v; best = k; }
    }
    return best;
}

// module_cu_gf_deep.F:2861-2914.  .GE. -- the LAST maximum, unlike MAXLOC.
__device__ int gfd_cup_maximi(const float *arr, int ks, int ke, int ierr)
{
    int maxx = ks;
    if (ierr != 0) return maxx;
    float x = arr[ks];
    for (int k = ks; k <= ke; k++)
        if (arr[k] >= x) { x = arr[k]; maxx = k; }
    return maxx;
}

// module_cu_gf_deep.F:2917-2965.
__device__ int gfd_cup_minimi(const float *arr, int ks, int kend, int ierr)
{
    int kt = ks;
    if (ierr != 0) return kt;
    float x = arr[ks];
    int kstop = IMAX(ks + 1, kend);
    for (int k = ks + 1; k <= kstop; k++)
        if (arr[k] < x) { x = arr[k]; kt = k; }
    return kt;
}

// The hcot integration cup_kbcon rebuilds every time k22 moves
// (module_cu_gf_deep.F:2778-2792).
__device__ void gfd_kbcon_fill_hcot(float *hcot, int start, float hk,
    int kbmax, int nz, const float *z_cup, float entr_rate, const float *heo)
{
    for (int k = 1; k <= start; k++) hcot[k] = hk;
    int hi = IMIN(kbmax + 3, nz);
    for (int k = start + 1; k <= hi; k++) {
        float dz = FSUB(z_cup[k], z_cup[k - 1]);
        hcot[k] = FDIV(
            FADD(FMUL(FSUB(K_ONE, FMUL(FMUL(K_HALF, entr_rate), dz)),
                      hcot[k - 1]),
                 FMUL(FMUL(entr_rate, dz), heo[k - 1])),
            FADD(K_ONE, FMUL(FMUL(K_HALF, entr_rate), dz)));
    }
}

// module_cu_gf_deep.F:2722-2858, imid = 0, transcribed from the GO TO graph.
// k22 and hkb are both INOUT and both move when the cap test fails.
__device__ void gfd_cup_kbcon(float cap_inc, int iloop, int *k22_io,
    const float *he_cup, const float *hes_cup, float *hkb_io, int *ierr_io,
    int kbmax, const float *p_cup, float cap_max, float ztexec, float zqexec,
    const float *z_cup, float entr_rate, const float *heo, int nz,
    float *hcot, int *kbcon_out)
{
    int kbcon = 1;
    if (*ierr_io != 0) { *kbcon_out = kbcon; return; }
    int k22 = *k22_io;
    float hkb = *hkb_io;
    int start_level = k22;
    kbcon = k22 + 1;
    if (iloop == 5) kbcon = k22;
    gfd_kbcon_fill_hcot(hcot, start_level, hkb, kbmax, nz, z_cup, entr_rate,
                        heo);
    for (;;) {
        float hetest = hcot[kbcon];
        if (hetest < hes_cup[kbcon]) {
            kbcon += 1;
            if (kbcon > kbmax + 2) { *ierr_io = 3; break; }
            continue;
        }
        if (kbcon - k22 == 1) break;
        if (iloop == 5 && (kbcon - k22) <= 2) break;
        float pbcdif = FADD(-p_cup[kbcon], p_cup[k22]);
        float plus = GMAX(K_TWENTYFIVE,
                          FSUB(cap_max, FMUL((float)(iloop - 1), cap_inc)));
        if (iloop == 5) {
            plus = K_P150;
            if (cap_max > K_P200) pbcdif = FADD(-p_cup[kbcon], cap_max);
        }
        if (pbcdif <= plus) break;
        k22 += 1;
        kbcon = k22 + 1;
        float x_add = FADD(FMUL(K_XLV, zqexec), FMUL(K_CP, ztexec));
        hkb = gfd_get_cloud_bc(he_cup, k22, x_add, 1);
        start_level = k22;
        gfd_kbcon_fill_hcot(hcot, start_level, hkb, kbmax, nz, z_cup,
                            entr_rate, heo);
        if (iloop == 5) kbcon = k22;
        if (kbcon > kbmax + 2) { *ierr_io = 3; break; }
    }
    *k22_io = k22;
    *hkb_io = hkb;
    *kbcon_out = kbcon;
}

// module_cu_gf_deep.F:3825-3987, drafts UP (0), SH2 (1) and DOWN (2).
// fzu_override <= 0 means "compute fzu with gfk_tgamma"; the parity suite
// can pin the oracle's captured word instead, which is how a residual would
// be attributed to gamma alone if one ever appeared.
struct GfdPdf { float tunning, alpha, beta, fzu; int kb_adj; };

__device__ void gfd_get_zu_zd_pdf(int draft, const float *p, int kb, int kt,
    int kpbli, int csum, float zubeg, int nz, int ktf, float fzu_override,
    float *zu, GfdPdf *info)
{
    for (int k = 0; k < GF_KP; k++) zu[k] = K_ZERO;
    int kb_adj = IMAX(kb, 2);
    float tunning, beta;
    if (draft == 0) {
        float lev_start = GMIN(K_P9, FADD(K_P4, FMUL((float)csum, K_P013)));
        kb_adj = IMAX(kb, 2);
        tunning = FADD(p[kt], FMUL(FSUB(p[kpbli], p[kt]), lev_start));
        tunning = GMIN(K_P9,
                       FDIV(FSUB(tunning, p[kb_adj]), FSUB(p[kt], p[kb_adj])));
        tunning = GMAX(K_P2, tunning);
        beta = K_ONE_P3;
    } else if (draft == 1) {
        tunning = GMIN(K_P8,
                       FDIV(FSUB(p[kpbli], p[kb_adj]), FSUB(p[kt], p[kb_adj])));
        tunning = GMAX(K_P2, tunning);
        beta = K_TWO_P5;
    } else {
        tunning = p[kb];
        tunning = GMIN(K_P9, FDIV(FSUB(tunning, p[1]), FSUB(p[kt], p[1])));
        tunning = GMAX(K_P2, tunning);
        beta = K_FOUR;
    }
    float alpha = FDIV(FADD(FMUL(tunning, FSUB(beta, K_TWO)), K_ONE),
                       FSUB(K_ONE, tunning));
    float fzu;
    if (fzu_override > K_ZERO) {
        fzu = fzu_override;
    } else {
        fzu = FDIV(gfk_tgamma(FADD(alpha, beta)),
                   FMUL(gfk_tgamma(alpha), gfk_tgamma(beta)));
    }
    float ea = FSUB(alpha, K_ONE);
    float eb = FSUB(beta, K_ONE);
    int klo, khi;
    float pbase;
    if (draft == 0 || draft == 1) {
        klo = kb_adj; khi = IMIN(nz, kt); pbase = p[kb_adj];
    } else {
        klo = 2; khi = IMIN(kt, ktf); pbase = p[1];
    }
    for (int k = klo; k <= khi; k++) {
        float kratio = FDIV(FSUB(p[k], pbase), FSUB(p[kt], pbase));
        zu[k] = FADD(zubeg,
                     FMUL(FMUL(fzu, gfk_pow(kratio, ea)),
                          gfk_pow(FSUB(K_ONE, kratio), eb)));
    }
    int hi = IMIN(ktf, kt + 1);
    float peak = zu[1];
    for (int k = 2; k <= hi; k++)
        if (zu[k] > peak) peak = zu[k];
    if (peak > K_ZERO)
        for (int k = 1; k <= hi; k++) zu[k] = FDIV(zu[k], peak);
    if (draft == 0 || draft == 1) {
        for (int k = gfd_maxloc(zu, 1, nz); k >= 1; k--)
            if (zu[k] < K_E1M6) { kb_adj = k + 1; break; }
        if (draft == 0) {
            kb_adj = IMAX(2, kb_adj);
            for (int k = 1; k < kb_adj; k++) zu[k] = K_ZERO;
        }
    } else {
        zu[1] = K_ZERO;
    }
    info->tunning = tunning; info->alpha = alpha; info->beta = beta;
    info->fzu = fzu; info->kb_adj = kb_adj;
}

// module_cu_gf_deep.F:3697-3823, name == 'deep'.  kbcon is raised to at
// least 2 for EVERY column, ierr or not; the k22..kbcon ramp is built and
// then thrown away by the pdf.
__device__ void gfd_rates_up_pdf_deep(int *ktop_io, int *ierr_io,
    const float *p_cup, const float *entr_rate_2d, float hkbo,
    const float *heo, const float *heso_cup, const float *z_cup, int kstabi,
    int k22, int *kbcon_io, int csum, int nz, int ktf, float fzu_override,
    float *zuo, int *ktopdby_out, GfdPdf *pdf, int *kklev_out,
    int *kfinal_out, float *dby, float *dbm, float *hcot)
{
    for (int k = 0; k < GF_KP; k++) zuo[k] = K_ZERO;
    int kbcon = IMAX(*kbcon_io, 2);
    *kbcon_io = kbcon;
    *kklev_out = 0; *kfinal_out = 0; *ktopdby_out = 0;
    if (*ierr_io != 0) return;
    for (int k = 0; k < GF_KP; k++) {
        dby[k] = K_ZERO; dbm[k] = K_ZERO; hcot[k] = K_ZERO;
    }
    int start_level = k22;
    zuo[start_level] = K_P1;
    for (int k = start_level + 1; k <= kbcon; k++) {
        float dz = FSUB(z_cup[k], z_cup[k - 1]);
        float massent = FMUL(FMUL(dz, entr_rate_2d[k - 1]), zuo[k - 1]);
        float massdetr = FMUL(FMUL(dz, K_E1M9), zuo[k - 1]);
        zuo[k] = FSUB(FADD(zuo[k - 1], massent), massdetr);
    }
    int ktop = 0;
    hcot[start_level] = hkbo;
    for (int k = start_level + 1; k <= ktf - 2; k++) {
        float dz = FSUB(z_cup[k], z_cup[k - 1]);
        hcot[k] = FDIV(
            FADD(FMUL(FSUB(K_ONE, FMUL(FMUL(K_HALF, entr_rate_2d[k - 1]), dz)),
                      hcot[k - 1]),
                 FMUL(FMUL(entr_rate_2d[k - 1], dz), heo[k - 1])),
            FADD(K_ONE, FMUL(FMUL(K_HALF, entr_rate_2d[k - 1]), dz)));
        if (k >= kbcon) {
            dby[k] = FADD(dby[k - 1], FMUL(FSUB(hcot[k], heso_cup[k]), dz));
            dbm[k] = FSUB(hcot[k], heso_cup[k]);
        }
    }
    int ktopdby = gfd_maxloc(dby, 1, nz);
    int kklev = gfd_maxloc(dbm, 1, nz);
    float dbymax = dby[1];
    for (int k = 2; k <= nz; k++)
        if (dby[k] > dbymax) dbymax = dby[k];
    int kfinalzu = ktf - 2;
    ktop = kfinalzu;
    for (int k = ktopdby + 1; k <= ktf - 2; k++) {
        if (dby[k] < FMUL(K_ONE, dbymax)) {
            kfinalzu = k - 1;
            ktop = kfinalzu;
            break;
        }
    }
    *kklev_out = kklev; *kfinal_out = kfinalzu; *ktopdby_out = ktopdby;
    if (kfinalzu <= kbcon + 2) {
        *ierr_io = 41;
        *ktop_io = 0;
        return;
    }
    gfd_get_zu_zd_pdf(0, p_cup, k22, kfinalzu, kstabi, csum, K_P1, nz, ktf,
                      fzu_override, zuo, pdf);
    *ktop_io = ktop;
}

// module_cu_gf_deep.F:4239-4334.  Writes cd and entr_rate_2d back.  The
// deep call site passes lambau, so the momentum limb runs.
__device__ void gfd_get_lateral_massflux(int ierr, int ktop,
    const float *zo_cup, const float *zuo, float *cd, float *entr_rate_2d,
    int kbcon, int k22, int nz, int ktf, float lambau, int has_lambau,
    float *upme, float *upmd, float *upmeu, float *upmdu)
{
    for (int k = 0; k < GF_KP; k++) {
        upme[k] = K_ZERO; upmd[k] = K_ZERO;
        upmeu[k] = K_ZERO; upmdu[k] = K_ZERO;
    }
    if (ierr != 0) return;
    int kpeak = gfd_maxloc(zuo, 1, nz);
    for (int k = IMAX(2, k22 + 1); k <= kpeak; k++) {
        float dz = FSUB(zo_cup[k], zo_cup[k - 1]);
        upmd[k - 1] = FMUL(FMUL(cd[k - 1], dz), zuo[k - 1]);
        upme[k - 1] = FADD(FSUB(zuo[k], zuo[k - 1]), upmd[k - 1]);
        if (upme[k - 1] < K_ZERO) {
            upme[k - 1] = K_ZERO;
            upmd[k - 1] = FSUB(zuo[k - 1], zuo[k]);
            if (zuo[k - 1] > K_ZERO)
                cd[k - 1] = FDIV(upmd[k - 1], FMUL(dz, zuo[k - 1]));
        }
        if (zuo[k - 1] > K_ZERO)
            entr_rate_2d[k - 1] = FDIV(upme[k - 1], FMUL(dz, zuo[k - 1]));
    }
    for (int k = kpeak + 1; k <= ktop; k++) {
        float dz = FSUB(zo_cup[k], zo_cup[k - 1]);
        upme[k - 1] = FMUL(FMUL(entr_rate_2d[k - 1], dz), zuo[k - 1]);
        upmd[k - 1] = FSUB(FADD(zuo[k - 1], upme[k - 1]), zuo[k]);
        if (upmd[k - 1] < K_ZERO) {
            upmd[k - 1] = K_ZERO;
            upme[k - 1] = FSUB(zuo[k], zuo[k - 1]);
            if (zuo[k - 1] > K_ZERO)
                entr_rate_2d[k - 1] = FDIV(upme[k - 1], FMUL(dz, zuo[k - 1]));
        }
        if (zuo[k - 1] > K_ZERO)
            cd[k - 1] = FDIV(upmd[k - 1], FMUL(dz, zuo[k - 1]));
    }
    upmd[ktop] = zuo[ktop];
    upme[ktop] = K_ZERO;
    for (int k = ktop + 1; k <= ktf; k++) {
        cd[k] = K_ZERO;
        entr_rate_2d[k] = K_ZERO;
        upme[k] = K_ZERO;
        upmd[k] = K_ZERO;
    }
    if (has_lambau) {
        for (int k = 2; k <= ktf - 1; k++) {
            upmeu[k - 1] = FADD(upme[k - 1], FMUL(lambau, upmd[k - 1]));
            upmdu[k - 1] = FADD(upmd[k - 1], FMUL(lambau, upmd[k - 1]));
        }
    }
}

// module_cu_gf_deep.F:3355-3642, autoconv = 1.  c0 compounds CUMULATIVELY
// across sub-freezing levels below kbcon and resets every level above it.
__device__ void gfd_cup_up_moisture(int *ierr_io, const float *z_cup,
    const float *p_cup, int kbcon, int ktop, const float *dby, int xland1,
    const float *q, const float *gamma_cup, const float *zu,
    const float *qes_cup, int k22, const float *qe_cup, float zqexec,
    float ccn, const float *rho, const float *c1d, const float *t,
    const float *up_massentr, const float *up_massdetr, int nz,
    float *qc, float *qrc, float *pw, float *clw_all, float *qch,
    float *pwav_out, float *psum_out, float *psumh_out)
{
    for (int k = 0; k < GF_KP; k++) {
        qc[k] = K_ZERO; qrc[k] = K_ZERO; pw[k] = K_ZERO;
        clw_all[k] = K_ZERO; qch[k] = K_ZERO;
    }
    float pwav = K_ZERO, psum = K_ZERO, psumh = K_ZERO;
    *pwav_out = pwav; *psum_out = psum; *psumh_out = psumh;
    if (*ierr_io != 0) return;
    for (int k = 1; k <= nz; k++) qc[k] = qe_cup[k];
    for (int k = 0; k < GF_KP; k++) qch[k] = qc[k];
    int start_level = k22;
    float qaver = gfd_get_cloud_bc(qe_cup, k22, K_ZERO, 0);
    qc[start_level] = qaver;
    qch[start_level] = qaver;
    for (int k = 1; k < start_level; k++) {
        qc[k] = qe_cup[k];
        qch[k] = qe_cup[k];
    }

    float c0 = K_C0_UP;
    for (int k = k22 + 1; k <= kbcon; k++) {
        if (t[k] < K_T273_15)
            c0 = FMUL(c0, gfk_exp(FMUL(K_P07, FSUB(t[k], K_T273_15))));
        qc[k] = FDIV(
            FADD(FSUB(FMUL(qc[k - 1], zu[k - 1]),
                      FMUL(FMUL(K_HALF, up_massdetr[k - 1]), qc[k - 1])),
                 FMUL(up_massentr[k - 1], q[k - 1])),
            FADD(FSUB(zu[k - 1], FMUL(K_HALF, up_massdetr[k - 1])),
                 up_massentr[k - 1]));
        float qrch = FADD(
            qes_cup[k],
            FMUL(FMUL(K_ONE_OVER_XLV,
                      FDIV(gamma_cup[k], FADD(K_ONE, gamma_cup[k]))),
                 dby[k]));
        if (k < kbcon) qrch = qc[k];
        if (qc[k] > qrch) {
            float dz = FSUB(z_cup[k], z_cup[k - 1]);
            qrc[k] = FDIV(FSUB(qc[k], qrch), FADD(K_ONE, FMUL(c0, dz)));
            pw[k] = FMUL(FMUL(FMUL(c0, dz), qrc[k]), zu[k]);
            qc[k] = FADD(qrch, qrc[k]);
            clw_all[k] = qrc[k];
        }
    }

    for (int k = kbcon + 1; k <= ktop; k++) {
        c0 = K_C0_UP;
        if (t[k] < K_T273_15)
            c0 = FMUL(c0, gfk_exp(FMUL(K_P07, FSUB(t[k], K_T273_15))));
        float denom = FADD(FSUB(zu[k - 1], FMUL(K_HALF, up_massdetr[k - 1])),
                           up_massentr[k - 1]);
        if (denom < K_E1M8) { *ierr_io = 51; break; }
        float dz = FSUB(z_cup[k], z_cup[k - 1]);
        float qrch = FADD(
            qes_cup[k],
            FMUL(FMUL(K_ONE_OVER_XLV,
                      FDIV(gamma_cup[k], FADD(K_ONE, gamma_cup[k]))),
                 dby[k]));
        qc[k] = FDIV(
            FADD(FSUB(FMUL(qc[k - 1], zu[k - 1]),
                      FMUL(FMUL(K_HALF, up_massdetr[k - 1]), qc[k - 1])),
                 FMUL(up_massentr[k - 1], q[k - 1])),
            denom);
        qch[k] = FDIV(
            FADD(FSUB(FMUL(qch[k - 1], zu[k - 1]),
                      FMUL(FMUL(K_HALF, up_massdetr[k - 1]), qch[k - 1])),
                 FMUL(up_massentr[k - 1], q[k - 1])),
            denom);
        if (qc[k] <= qrch) qc[k] = qrch;
        if (qch[k] <= qrch) qch[k] = qrch;
        clw_all[k] = GMAX(K_ZERO, FSUB(qc[k], qrch));
        qrc[k] = GMAX(K_ZERO, FSUB(qc[k], qrch));
        qrc[k] = FDIV(FSUB(qc[k], qrch),
                      FADD(K_ONE, FMUL(FADD(c1d[k], c0), dz)));
        pw[k] = FMUL(FMUL(FMUL(c0, dz), qrc[k]), zu[k]);
        if (qrc[k] < K_ZERO) {
            qrc[k] = K_ZERO;
            pw[k] = K_ZERO;
        }
        qc[k] = FADD(qrc[k], qrch);
        pwav = FADD(pwav, pw[k]);
        psum = FADD(psum, FMUL(FMUL(clw_all[k], zu[k]), dz));
    }
    // The ierr = 51 exit leaves the k loop but NOT the if(ierr.eq.0) block,
    // so this sweep runs on both paths.
    for (int k = k22 + 1; k <= ktop; k++)
        qc[k] = FSUB(qc[k], qrc[k]);
    *pwav_out = pwav;
    *psum_out = psum;
    *psumh_out = psumh;
    (void)xland1; (void)zqexec; (void)ccn; (void)rho; (void)p_cup;
}

// module_cu_gf_deep.F:1996-2139, iloop = 1.
__device__ void gfd_cup_dd_moisture(int *ierr_io, const float *zd,
    const float *hcd, const float *hes_cup, const float *qes_cup,
    const float *q_cup, const float *z_cup, const float *dd_massentr,
    const float *dd_massdetr, int jmin, const float *gamma_cup,
    const float *q, int nz, float *qcd, float *qrcd, float *pwd,
    float *pwev_out, float *bu_out)
{
    for (int k = 0; k < GF_KP; k++) {
        qcd[k] = K_ZERO; qrcd[k] = K_ZERO; pwd[k] = K_ZERO;
    }
    float pwev = K_ZERO, bu = K_ZERO;
    *pwev_out = pwev; *bu_out = bu;
    if (*ierr_io != 0) return;
    int k = jmin;
    float dz = FSUB(z_cup[k + 1], z_cup[k]);
    qcd[k] = q_cup[k];
    float dh = FSUB(hcd[k], hes_cup[k]);
    if (dh < K_ZERO) {
        qrcd[k] = FADD(
            qes_cup[k],
            FMUL(FMUL(K_ONE_OVER_XLV,
                      FDIV(gamma_cup[k], FADD(K_ONE, gamma_cup[k]))),
                 dh));
    } else {
        qrcd[k] = qes_cup[k];
    }
    pwd[jmin] = FMUL(zd[jmin], GMIN(K_ZERO, FSUB(qcd[k], qrcd[k])));
    qcd[k] = qrcd[k];
    pwev = FADD(pwev, pwd[jmin]);
    bu = FMUL(dz, dh);
    for (int ki = jmin - 1; ki >= 1; ki--) {
        dz = FSUB(z_cup[ki + 1], z_cup[ki]);
        float denom = FADD(FSUB(zd[ki + 1], FMUL(K_HALF, dd_massdetr[ki])),
                           dd_massentr[ki]);
        if (denom < K_E1M8) { *ierr_io = 51; break; }
        qcd[ki] = FDIV(
            FADD(FSUB(FMUL(qcd[ki + 1], zd[ki + 1]),
                      FMUL(FMUL(K_HALF, dd_massdetr[ki]), qcd[ki + 1])),
                 FMUL(dd_massentr[ki], q[ki])),
            denom);
        dh = FSUB(hcd[ki], hes_cup[ki]);
        bu = FADD(bu, FMUL(dz, dh));
        qrcd[ki] = FADD(
            qes_cup[ki],
            FMUL(FMUL(K_ONE_OVER_XLV,
                      FDIV(gamma_cup[ki], FADD(K_ONE, gamma_cup[ki]))),
                 dh));
        float dqeva = FSUB(qcd[ki], qrcd[ki]);
        if (dqeva > K_ZERO) {
            dqeva = K_ZERO;
            qrcd[ki] = qcd[ki];
        }
        pwd[ki] = FMUL(zd[ki], dqeva);
        qcd[ki] = qrcd[ki];
        pwev = FADD(pwev, pwd[ki]);
    }
    if (pwev == K_ZERO) *ierr_io = 7;
    if (bu >= K_ZERO) *ierr_io = 7;
    *pwev_out = pwev;
    *bu_out = bu;
}

// module_cu_gf_deep.F:2968-3035.  dby is read one level BELOW k.
__device__ float gfd_cup_up_aa0(const float *z, const float *zu,
    const float *dby, const float *gamma_cup, const float *t_cup, int kbcon,
    int ktop, int ierr, int ktf)
{
    float aa0 = K_ZERO;
    if (ierr != 0) return aa0;
    for (int k = 2; k <= ktf; k++) {
        if (k <= kbcon || k > ktop) continue;
        float dz = FSUB(z[k], z[k - 1]);
        float da = FDIV(
            FMUL(FMUL(FMUL(zu[k], dz),
                      FDIV(K_LIT_G, FMUL(K_LIT_CP, t_cup[k]))),
                 dby[k - 1]),
            FADD(K_ONE, gamma_cup[k]));
        aa0 = FADD(aa0, GMAX(K_ZERO, da));
        if (aa0 < K_ZERO) aa0 = K_ZERO;
    }
    return aa0;
}

// module_cu_gf_deep.F:3990-4061.
__device__ float gfd_cup_up_aa1bl(const float *t, const float *tn,
    const float *q, const float *qo, float dtime, const float *z, int kbcon,
    int ierr, int ktf)
{
    float aa0 = K_ZERO;
    if (ierr != 0) return aa0;
    for (int k = 2; k <= ktf; k++) {
        if (k > kbcon) continue;
        float dz = FSUB(z[k], z[k - 1]);
        float da = FDIV(
            FMUL(FMUL(dz, K_LIT_G),
                 FADD(FSUB(tn[k], t[k]), FMUL(K_P608, FSUB(qo[k], q[k])))),
            dtime);
        aa0 = FADD(aa0, da);
    }
    return aa0;
}

// module_cu_gf_deep.F:1871-1993, aeroevap = 1.  VSHEAR**2 / **3 are integer
// literal exponents and fold to multiply chains, bitwise, per the probe.
__device__ void gfd_cup_dd_edt(int ierr, const float *us, const float *vs,
    const float *z, int ktop, int kbcon, const float *p, float pwav,
    float pwev, float edtmax, float edtmin, int ktf, float *edt_out,
    float *edtc_out)
{
    *edt_out = K_ZERO;
    *edtc_out = K_ZERO;
    if (ierr != 0) return;
    float vws = K_ZERO, sdp = K_ZERO, vshear = K_ZERO;
    for (int kk = 1; kk <= ktf - 1; kk++) {
        if (kk >= kbcon && kk <= IMIN(ktop, ktf)) {
            vws = FADD(vws,
                FMUL(FADD(fabsf(FDIV(FSUB(us[kk + 1], us[kk]),
                                     FSUB(z[kk + 1], z[kk]))),
                          fabsf(FDIV(FSUB(vs[kk + 1], vs[kk]),
                                     FSUB(z[kk + 1], z[kk])))),
                     FSUB(p[kk], p[kk + 1])));
            // (sdp + p(kk)) - p(kk+1), NOT sdp + (p(kk) - p(kk+1)).
            sdp = FSUB(FADD(sdp, p[kk]), p[kk + 1]);
        }
        if (kk == ktf - 1)
            vshear = FDIV(FMUL(K_E1P3, vws), sdp);
    }
    float pef = FSUB(
        FADD(FSUB(K_PEF_A, FMUL(K_PEF_B, vshear)),
             FMUL(K_PEF_C, FMUL(vshear, vshear))),
        FMUL(K_PEF_D, FMUL(FMUL(vshear, vshear), vshear)));
    if (pef > K_P9) pef = K_P9;
    if (pef < K_P1) pef = K_P1;
    float zkbc = FMUL(z[kbcon], K_ZKBC_SCALE);
    float prezk = K_PREZK0;
    if (zkbc > K_THREE) {
        prezk = FADD(K_PZ_A,
            FMUL(zkbc, FADD(K_PZ_B,
                FMUL(zkbc, FADD(K_PZ_C,
                    FMUL(zkbc, FADD(K_PZ_D,
                        FMUL(zkbc, FSUB(K_PZ_E, FMUL(zkbc, -K_PZ_F))))))))));
    }
    if (zkbc > K_TWENTYFIVE) prezk = K_PREZK25;
    float pefb = FDIV(K_ONE, FADD(K_ONE, prezk));
    if (pefb > K_P9) pefb = K_P9;
    if (pefb < K_P1) pefb = K_P1;
    float edt = FSUB(K_ONE, FMUL(K_HALF, FADD(pefb, pef)));
    float einc = FMUL(K_EDT_EINC, edt);
    float edtc = FSUB(edt, einc);
    edtc = FDIV(FMUL(-edtc, pwav), pwev);
    if (edtc > edtmax) edtc = edtmax;
    if (edtc < edtmin) edtc = edtmin;
    *edt_out = edt;
    *edtc_out = edtc;
}

// module_cu_gf_deep.F:2373-2720.  16 members in four families; ens_adj is 1
// on every path and the xland block is an identity multiply WRF still runs.
#define GF_MAXENS3 16
__device__ void gfd_cup_forcing_ens_3d(float *closure_n_io, int xland1,
    float aa0, float aa1, float xaa0, float mbdt, float dtime, int ierr,
    int ierr2, int ierr3, float axx, float mconv, const float *p_cup,
    int ktop, const float *omeg, const float *zd, int k22, const float *zu,
    const float *pr_ens, float edt, int kbcon, int ichoice, int dicycle,
    float tau_ecmwf, float aa1_bl, int nz, float *xf_ens, float *forcing,
    float *xf_dicycle_out)
{
    for (int i = 0; i <= GF_MAXENS3; i++) xf_ens[i] = K_ZERO;
    for (int i = 0; i < 11; i++) forcing[i] = K_ZERO;
    *xf_dicycle_out = K_ZERO;
    if (ierr != 0) return;
    float xff[GF_MAXENS3 + 1];
    for (int i = 0; i <= GF_MAXENS3; i++) xff[i] = K_ZERO;
    float ens_adj = K_ONE;
    int kloc = gfd_maxloc(zu, 1, nz);
    float a_ave = axx;
    a_ave = GMAX(K_ZERO, a_ave);
    a_ave = GMIN(a_ave, aa1);
    a_ave = GMAX(K_ZERO, a_ave);
    float xff0 = FDIV(FSUB(aa1, aa0), dtime);
    xff[1] = GMAX(K_ZERO, FDIV(FSUB(aa1, aa0), dtime));
    xff[2] = xff[1];
    xff[3] = xff[1];
    xff[16] = xff[1];
    {
        // spelled per member in the reference; the four are the same word
        const float v = GMAX(K_ZERO, FDIV(FSUB(aa1, aa0), dtime));
        xff[1] = v; xff[2] = v; xff[3] = v; xff[16] = v;
    }
    forcing[1] = xff[2];

    float xomg = K_ZERO;
    int kk = 0;
    for (int k = kbcon - 1; k <= kbcon + 1; k++) {
        if (zu[k] > K_ZERO) {
            xomg = FSUB(xomg,
                FDIV(FDIV(omeg[k], K_LIT_G),
                     GMAX(K_HALF,
                          FSUB(K_ONE, FDIV(FMUL(edt, zd[k]), zu[k])))));
            kk += 1;
        }
    }
    if (kk > 0) xff[4] = FDIV(xomg, (float)kk);
    xff[4] = FMUL(K_BETAJB, xff[4]);
    xff[5] = xff[4];
    xff[6] = xff[4];
    for (int nn = 4; nn <= 6; nn++)
        if (xff[nn] < K_ZERO) xff[nn] = K_ZERO;
    xff[14] = FMUL(K_BETAJB, xff[4]);
    forcing[2] = xff[4];

    float den = GMAX(K_HALF,
                     FSUB(K_ONE, FDIV(FMUL(edt, zd[kbcon]), zu[kloc])));
    xff[7] = FDIV(mconv, den);
    xff[8] = xff[7];
    xff[9] = xff[7];
    xff[15] = xff[7];
    {
        const float v = FDIV(mconv, den);
        xff[7] = v; xff[8] = v; xff[9] = v; xff[15] = v;
    }
    forcing[3] = xff[8];

    {
        const float v = FDIV(aa1, tau_ecmwf);
        xff[10] = v; xff[11] = v; xff[12] = v; xff[13] = v;
    }
    float xff_dicycle;
    if (dicycle == 1) xff_dicycle = GMAX(K_ZERO, FDIV(aa1_bl, tau_ecmwf));
    else xff_dicycle = K_ZERO;

    if (ichoice == 0 && xff0 < K_ZERO) {
        xff[1] = K_ZERO; xff[2] = K_ZERO; xff[3] = K_ZERO;
        xff[10] = K_ZERO; xff[11] = K_ZERO; xff[12] = K_ZERO;
        xff[13] = K_ZERO; xff[16] = K_ZERO;
        *closure_n_io = K_TWELVE;
    }

    float xk = FDIV(FSUB(xaa0, aa1), mbdt);
    forcing[4] = aa0;
    forcing[5] = aa1;
    forcing[6] = xaa0;
    forcing[7] = xk;
    if (xk <= K_ZERO && xk > FMUL(K_NEG_P01, mbdt))
        xk = FMUL(K_NEG_P01, mbdt);
    if (xk > K_ZERO && xk < K_E1M2)
        xk = K_E1M2;

    if (xland1 < 1) {
        if (ierr2 > 0 || ierr3 > 0) {
            for (int nn = 1; nn <= GF_MAXENS3; nn++)
                xff[nn] = FMUL(ens_adj, xff[nn]);
            xff_dicycle = FMUL(ens_adj, xff_dicycle);
        }
    }

    if (xk < K_ZERO) {
        const int mem[4] = {1, 2, 3, 16};
        for (int m = 0; m < 4; m++) {
            int nn = mem[m];
            if (xff[nn] > K_ZERO)
                xf_ens[nn] = GMAX(K_ZERO, FDIV(-xff[nn], xk));
        }
    } else {
        xff[1] = K_ZERO; xff[2] = K_ZERO; xff[3] = K_ZERO; xff[16] = K_ZERO;
    }

    {
        const int mem[4] = {4, 5, 6, 14};
        for (int m = 0; m < 4; m++)
            xf_ens[mem[m]] = GMAX(K_ZERO, xff[mem[m]]);
    }
    {
        const int mem[4] = {7, 8, 9, 15};
        for (int m = 0; m < 4; m++) {
            int nn = mem[m];
            float floorv = (nn == 15) ? K_E1M3 : K_E1M5;
            float a1 = GMAX(floorv, pr_ens[nn]);
            xf_ens[nn] = GMAX(K_ZERO, FDIV(xff[nn], a1));
        }
    }

    if (xk < K_ZERO) {
        for (int nn = 10; nn <= 13; nn++)
            xf_ens[nn] = GMAX(K_ZERO, FDIV(-xff[nn], xk));
        forcing[8] = xf_ens[11];
    } else {
        for (int nn = 10; nn <= 13; nn++) xf_ens[nn] = K_ZERO;
        forcing[8] = K_ZERO;
    }

    if (xk < K_ZERO)
        *xf_dicycle_out = GMAX(K_ZERO, FDIV(-xff_dicycle, xk));
    else
        *xf_dicycle_out = K_ZERO;

    if (ichoice >= 1)
        for (int nn = 1; nn <= GF_MAXENS3; nn++)
            xf_ens[nn] = xf_ens[ichoice];
}

// module_cu_gf_deep.F:3142-3353, imid = 0.  xf_ens is scaled by sig once
// per level INSIDE the k loop -- sig**ktop, not sig -- and the diurnal
// subtraction is written as a min.
__device__ void gfd_cup_output_ens_3d(float *xf_ens, int *ierr_io,
    const float *dellat, const float *dellaq, const float *dellaqc,
    const float *zu, const float *pw, int ktop, float edt, const float *pwd,
    const float *p_cup, const float *pr_ens, float sig, float closure_n,
    float xmbs_in, int dicycle, float xf_dicycle, int nz, float *outt,
    float *outq, float *outqc, float *pre_out, float *xmb_out)
{
    for (int k = 0; k < GF_KP; k++) {
        outt[k] = K_ZERO; outq[k] = K_ZERO; outqc[k] = K_ZERO;
    }
    float pre = K_ZERO, xmb = K_ZERO;
    *pre_out = pre; *xmb_out = xmb;
    if (*ierr_io != 0) return;
    for (int nn = 1; nn <= GF_MAXENS3; nn++)
        if (pr_ens[nn] <= K_ZERO) xf_ens[nn] = K_ZERO;
    float xmb_ave = K_ZERO;
    for (int nn = 1; nn <= GF_MAXENS3; nn++)
        xmb_ave = FADD(xmb_ave, xf_ens[nn]);
    xmb_ave = FDIV(xmb_ave, (float)GF_MAXENS3);
    if (dicycle == 2) {
        xmb_ave = FSUB(xmb_ave, GMAX(K_ZERO, xmbs_in));
        xmb_ave = GMAX(K_ZERO, xmb_ave);
    } else if (dicycle == 1) {
        xmb_ave = GMIN(xmb_ave, FSUB(xmb_ave, xf_dicycle));
        xmb_ave = GMAX(K_ZERO, xmb_ave);
    }
    float clos_wei = FDIV(K_SIXTEEN, GMAX(K_ONE, closure_n));
    xmb_ave = GMIN(xmb_ave, K_HUNDRED);
    xmb = FMUL(FMUL(clos_wei, sig), xmb_ave);
    if (xmb < K_E1M16) *ierr_io = 19;
    float pwtot = K_ZERO;
    *xmb_out = xmb;
    if (*ierr_io != 0) return;
    for (int k = 1; k <= ktop; k++)
        pwtot = FADD(pwtot, pw[k]);
    for (int k = 1; k <= ktop; k++) {
        float dp = FDIV(FMUL(K_HUNDRED, FSUB(p_cup[k], p_cup[k + 1])), K_G);
        float dtt = dellat[k];
        float dtq = dellaq[k];
        float dtpwd = -FMUL(pwd[k], edt);
        float dtqc = FSUB(FMUL(dellaqc[k], dp), dtpwd);
        if (dtqc < K_ZERO) {
            dtpwd = FSUB(dtpwd, FMUL(dellaqc[k], dp));
            dtqc = K_ZERO;
        } else {
            dtpwd = K_ZERO;
            dtqc = FDIV(dtqc, dp);
        }
        outt[k] = FMUL(xmb, dtt);
        outq[k] = FMUL(xmb, dtq);
        outqc[k] = FMUL(xmb, dtqc);
        for (int nn = 1; nn <= GF_MAXENS3; nn++)
            xf_ens[nn] = FMUL(sig, xf_ens[nn]);
        pre = FSUB(pre, FMUL(xmb, dtpwd));
    }
    pre = FADD(-pre, FMUL(xmb, pwtot));
    *pre_out = pre;
}

// module_cu_gf_deep.F:4063-4159, reached from CUP_gf_sh and captured here.
// DIVERGENCE, deliberate and counted: WRF's first-derivative loop reads
// t_cup(kend+8) out of bounds when kend > ktf-8; this port clamps kend to
// ktf-8 exactly as the CPU reference and the oracle capture do, and
// reports the clamp.  The count is 0 on the whole committed fixture.
__device__ void gfd_get_inversion_layers(int ierr, const float *p_cup,
    const float *t_cup, const float *z_cup, int kstart, int kend, int nz,
    int ktf, float *dtempdz, int *k_inv, int *clamped_out,
    float *first, float *sec, float *sd)
{
    for (int k = 0; k < GF_KP; k++) {
        dtempdz[k] = K_ZERO;
        k_inv[k] = 1;
        first[k] = K_ZERO;
        sec[k] = K_ZERO;
        sd[k] = K_ZERO;
    }
    k_inv[0] = 1;
    *clamped_out = 0;
    if (ierr != 0) return;
    if (kend > ktf - 8) {
        kend = ktf - 8;
        *clamped_out = 1;
    }
    int kend_p3 = kend + 3;
    for (int k = 2; k <= kend_p3 + 4; k++) {
        first[k] = FDIV(FSUB(t_cup[k + 1], t_cup[k - 1]),
                        FSUB(z_cup[k + 1], z_cup[k - 1]));
        dtempdz[k] = first[k];
    }
    for (int k = 3; k <= kend_p3 + 3; k++) {
        sec[k] = FDIV(FSUB(first[k + 1], first[k - 1]),
                      FSUB(z_cup[k + 1], z_cup[k - 1]));
        sec[k] = fabsf(sec[k]);
    }
    int ilev = IMAX(3, kstart + 1);
    int ix = 1;
    int k = ilev;
    while (ilev < kend_p3) {
        for (int kk = k; kk <= kend_p3 + 2; kk++) {
            if (sec[kk] < sec[kk + 1] && sec[kk] < sec[kk - 1]) {
                k_inv[ix] = kk;
                ix = IMIN(5, ix + 1);
                ilev = kk + 1;
                break;
            }
            ilev = kk + 1;
        }
        k = ilev;
    }
    int kadd = 0;
    int ken = gfd_maxloc_int(k_inv, 1, nz);
    for (int kx = 1; kx <= ken; kx++) {
        int kk = k_inv[kx + kadd];
        if (kk == 1) break;
        if (dtempdz[kk] < dtempdz[kk - 1] && dtempdz[kk] < dtempdz[kk + 1]) {
            kadd += 1;
            for (int kj = kx; kj <= ken; kj++) {
                if (k_inv[kj + kadd] > 1) k_inv[kj] = k_inv[kj + kadd];
                if (k_inv[kj + kadd] == 1) k_inv[kj] = 1;
            }
        }
    }
    // the 800 / 550 hPa slots
    int top = gfd_maxloc_int(k_inv, 1, nz);
    for (int kx = 0; kx < GF_KP; kx++) sd[kx] = K_BIG1E9;
    for (int kx = 1; kx <= top; kx++) {
        float dp = FSUB(p_cup[k_inv[kx]], p_cup[kstart]);
        sd[kx] = FSUB(fabsf(dp), K_P100);
    }
    int k800 = gfd_minloc_abs(sd, 1, nz);
    for (int kx = 0; kx < GF_KP; kx++) sd[kx] = K_BIG1E9;
    for (int kx = 1; kx <= top; kx++) {
        float dp = FSUB(p_cup[k_inv[kx]], p_cup[kstart]);
        sd[kx] = FSUB(fabsf(dp), K_P300);
    }
    int k550 = gfd_minloc_abs(sd, 1, nz);
    int shal = k_inv[k800];
    int mid = k_inv[k550];
    k_inv[1] = shal;
    k_inv[2] = mid;
    for (int kx = 3; kx < GF_KP; kx++) k_inv[kx] = -1;
}

// ==========================================================================
// capture field order.  tests/test_gf_deep_cuda.py carries the same lists;
// a wrong index here cannot round-trip the gate.
// ==========================================================================
enum {
    LEV_qes, LEV_he, LEV_hes, LEV_qeso, LEV_heo, LEV_heso,
    LEV_qes_cup, LEV_q_cup, LEV_he_cup, LEV_hes_cup, LEV_gamma_cup1,
    LEV_t_cup, LEV_qeso_cup, LEV_qo_cup, LEV_heo_cup, LEV_heso_cup,
    LEV_zo_cup, LEV_po_cup, LEV_gammao_cup, LEV_tn_cup,
    LEV_u_cup, LEV_v_cup, LEV_entr2d_a, LEV_zu_pdf,
    LEV_cd, LEV_entr2d_b, LEV_upme, LEV_upmd, LEV_upmeu, LEV_upmdu,
    LEV_hc, LEV_uc, LEV_vc, LEV_hco, LEV_dby, LEV_dbyo, LEV_dbyt,
    LEV_cdd, LEV_ddme, LEV_ddmd, LEV_ddmeu, LEV_ddmdu, LEV_mentrd2d,
    LEV_hcdo, LEV_ucd, LEV_vcd, LEV_dbydo, LEV_c1d,
    LEV_qcdo, LEV_qrcdo, LEV_pwdo,
    LEV_qco, LEV_qrco, LEV_pwo, LEV_clw_all,
    LEV_cupclw, LEV_cnvwt,
    LEV_dellu, LEV_dellv, LEV_dellah, LEV_dellaq, LEV_dellaqc, LEV_dellat,
    LEV_xhe, LEV_xq, LEV_xt, LEV_xqes, LEV_xhes,
    LEV_xqes_cup, LEV_xq_cup, LEV_xhe_cup, LEV_xhes_cup, LEV_gamma_cupx,
    LEV_xt_cup, LEV_xhc, LEV_xdby,
    LEV_outt_o, LEV_outq_o, LEV_outqc_o,
    LEV_outt_ke, LEV_outu_f, LEV_outv_f,
    LEV_dtempdz,
    GF_NLEV
};

enum {
    SCA_zws, SCA_ztexec, SCA_zqexec, SCA_cap_max, SCA_entr_rate, SCA_sig,
    SCA_sig_thresh, SCA_hkb0, SCA_hkbo0, SCA_frh_kb,
    SCA_up_tun, SCA_up_alpha, SCA_up_beta, SCA_up_fzu,
    SCA_dn_tun, SCA_dn_alpha, SCA_dn_beta, SCA_dn_fzu,
    SCA_bud, SCA_beta, SCA_edtmax, SCA_pwevo, SCA_bu,
    SCA_pwavo, SCA_psum, SCA_psumh,
    SCA_aa0, SCA_aa1, SCA_aa1_bl, SCA_tau_ecmwf, SCA_tau_bl, SCA_umean,
    SCA_edt, SCA_edtc1, SCA_edto, SCA_xhkb, SCA_xaa0, SCA_pr7, SCA_mconv2,
    SCA_xf1, SCA_xf2, SCA_xf3, SCA_xf4, SCA_xf5, SCA_xf6, SCA_xf7, SCA_xf8,
    SCA_xf9, SCA_xf10, SCA_xf11, SCA_xf12, SCA_xf13, SCA_xf14, SCA_xf15,
    SCA_xf16,
    SCA_f1, SCA_f2, SCA_f3, SCA_f4, SCA_f5, SCA_f6, SCA_f7, SCA_f8, SCA_f9,
    SCA_f10,
    SCA_xf_dicycle, SCA_closure_n, SCA_xmb, SCA_pre,
    GF_NSCA
};

enum {
    ISCA_xland1, ISCA_kbmax, ISCA_k22_0, ISCA_k22_1, ISCA_kbcon_1,
    ISCA_ierr_1, ISCA_kstabi, ISCA_kstabm, ISCA_pmin_lev, ISCA_start_level,
    ISCA_ktop_pdf, ISCA_ktopdby, ISCA_kbcon_2, ISCA_ierr_2,
    ISCA_up_kbadj, ISCA_up_kklev, ISCA_up_kfinal, ISCA_dn_kbadj,
    ISCA_ktop_dbyt, ISCA_ierr_3, ISCA_kzdown, ISCA_jmin, ISCA_kdet_2,
    ISCA_ierr_4, ISCA_ierr_5, ISCA_ierr_6, ISCA_ierr_7,
    ISCA_k22x, ISCA_kbconx, ISCA_ierr2, ISCA_ierr3, ISCA_ktop, ISCA_ierr,
    ISCA_kinv1, ISCA_kinv2, ISCA_kinv3, ISCA_kinv4, ISCA_kinv5,
    ISCA_kinv_clamped,
    GF_NISCA
};

// input packing (tests/test_gf_deep_cuda.py must agree)
enum {
    IN_zo, IN_t, IN_q, IN_tn, IN_qo, IN_po, IN_us, IN_vs, IN_rho, IN_omeg,
    GF_NIN_LEV
};
enum {
    INS_z1, INS_psur, INS_hfx, INS_qfx, INS_xland, INS_dx, INS_ccn,
    INS_dtime, INS_xmbs, INS_fzu_up, INS_fzu_dn,
    GF_NIN_SCA
};

// ==========================================================================
// CUP_gf itself: module_cu_gf_deep.F:39-1868, one column, with the
// per-stage capture the CPU gate grades.  WRF's reachable identity is fixed
// as parameters: imid_gf = 0, dicycle = 1, csum = 0 (ichoice is an argument
// because GFDRV forwards the namelist value; it is 0 on the whole fixture).
// The capture pointers are always valid -- gf_deep_stage passes slices of
// the output slabs, gf_gfdrv_stage passes per-thread scratch -- so the
// numeric path is one path, captured or not.
// ==========================================================================
__device__ void gfd_deep_column(
    const float *zo, const float *t, const float *q, const float *tn,
    const float *qo, const float *po, const float *us, const float *vs,
    const float *rho, const float *omeg,       // 1-based [GF_KP]
    float z1, float psur, float hfx, float qfx, float xland, float dx,
    float ccn, float dtime, float xmbs_in, float fzu_up, float fzu_dn,
    int kpbl, int nz, int ichoice,
    float *LEVB, float *SCAB, int *ISCB,       // capture
    float *outt, float *outq, float *outqc, float *outu, float *outv,
    float *cupclw,                             // 1-based [GF_KP] outputs
    float *pre_out, int *ktop_out, int *kbcon_out, int *k22_out,
    int *ierr_out)
{
    const int ktf = nz;
    const int kte = nz;
    const int csum = 0;
    const int dicycle = 1;

#define CAPL(FIDX, ARR) do { \
        for (int _k = 1; _k <= nz; _k++) \
            LEVB[(size_t)(FIDX) * (size_t)nz + (_k - 1)] = (ARR)[_k]; \
    } while (0)
    for (int i = 0; i < GF_NSCA; i++) SCAB[i] = K_ZERO;
    for (int i = 0; i < GF_NISCA; i++) ISCB[i] = 0;

    int ierr = 0, kbcon = 0, ktop = 0, k22 = 0, jmin = 0;

    // ---- :359-406 : w*, and the temperature/moisture excesses -------------
    float flux_tun = K_FLUXTUNE;
    float lambau = K_LAMBAU;
    float pgcon = K_ZERO;
    float ztexec = K_ZERO, zqexec = K_ZERO;
    float buo_flux = FDIV(
        FADD(FDIV(hfx, K_CP), FDIV(FMUL(FMUL(K_P608, t[1]), qfx), K_XLV)),
        rho[1]);
    float zws = GMAX(K_ZERO,
        FDIV(FMUL(FMUL(FMUL(FMUL(flux_tun, K_P41), buo_flux), zo[2]), K_G),
             t[1]));
    if (zws > K_TINY32) {
        zws = FMUL(K_ONE_P2, gfk_pow(zws, K_P3333));
        ztexec = GMAX(FDIV(FMUL(flux_tun, hfx),
                           FMUL(FMUL(rho[1], zws), K_CP)), K_ZERO);
        zqexec = GMAX(FDIV(FDIV(FMUL(flux_tun, qfx), K_XLV),
                           FMUL(rho[1], zws)), K_ZERO);
    }
    zws = GMAX(K_ZERO,
        FDIV(FMUL(FMUL(FMUL(FMUL(flux_tun, K_P41), buo_flux), zo[kpbl]),
                  K_G),
             t[kpbl]));
    zws = FMUL(K_ONE_P2, gfk_pow(zws, K_P3333));
    zws = FMUL(zws, rho[kpbl]);
    SCAB[SCA_zws] = zws;
    SCAB[SCA_ztexec] = ztexec;
    SCAB[SCA_zqexec] = zqexec;

    // ---- :409-433 ---------------------------------------------------------
    float cap_maxs = K_CAP_MAXS;
    float closure_n = K_SIXTEEN;
    float cap_max = cap_maxs;
    float cap_max_increment = K_CAP_INC_DEEP;
    int xland1 = (int)FADD(xland, K_P0001);
    if (xland > K_ONE_P5 || xland < K_HALF) {
        xland1 = 0;
        cap_max_increment = K_CAP_INC_DEEP;
    } else {
        if (ztexec > K_ZERO) cap_max = FADD(cap_max, K_TWENTYFIVE);
        if (ztexec < K_ZERO) cap_max = FSUB(cap_max, K_TWENTYFIVE);
    }
    SCAB[SCA_cap_max] = cap_max;
    ISCB[ISCA_xland1] = xland1;

    // ---- :455-471 : entrainment rate, radius, and sig ---------------------
    float c1d[GF_KP];
    for (int k = 0; k < GF_KP; k++) c1d[k] = K_ZERO;
    float entr_rate = FSUB(K_ENTR_BASE,
                           FMUL(GMIN(K_TWENTY, (float)csum), K_ENTR_CSUM));
    if (xland1 == 0) entr_rate = K_ENTR_BASE;
    float radius = FDIV(K_P2, entr_rate);
    float frh = GMIN(K_ONE,
                     FDIV(FDIV(FMUL(FMUL(K_PI314, radius), radius), dx), dx));
    if (frh > K_FRH_THRESH) {
        frh = K_FRH_THRESH;
        radius = FSQRT(FDIV(FMUL(FMUL(frh, dx), dx), K_PI314));
        entr_rate = FDIV(K_P2, radius);
    }
    float sig = FMUL(FSUB(K_ONE, frh), FSUB(K_ONE, frh));
    float sig_thresh = K_SIG_THRESH;
    SCAB[SCA_entr_rate] = entr_rate;
    SCAB[SCA_sig] = sig;
    SCAB[SCA_sig_thresh] = sig_thresh;

    // ---- :480-556 ---------------------------------------------------------
    float cnvwt[GF_KP], zuo[GF_KP], zdo[GF_KP];
    float z[GF_KP], xz[GF_KP], cd[GF_KP], cdd[GF_KP];
    float hcdo[GF_KP], qrcdo[GF_KP], dellaqc[GF_KP];
    for (int k = 0; k < GF_KP; k++) {
        cnvwt[k] = K_ZERO; zuo[k] = K_ZERO; zdo[k] = K_ZERO;
        cupclw[k] = K_ZERO;
        z[k] = zo[k]; xz[k] = zo[k];
        cd[k] = (k == 0) ? K_ZERO : K_E1M9;
        cdd[k] = (k == 0) ? K_ZERO : K_E1M9;
        hcdo[k] = K_ZERO; qrcdo[k] = K_ZERO; dellaqc[k] = K_ZERO;
    }
    float edtmax = K_ONE;
    float edtmin = K_EDTMIN;
    float depth_min = K_DEPTH_MIN;
    int kbmax = 1, kdet = 1;
    float aa0 = K_ZERO, aa1 = K_ZERO;
    int kstabm = ktf - 1;
    int ierr2 = 0, ierr3 = 0;
    float zkbmax = K_ZKBMAX;
    float zcutdown = K_ZCUTDOWN;
    float z_detr = K_Z_DETR;
    float pr_ens[GF_MAXENS3 + 1];
    for (int i = 0; i <= GF_MAXENS3; i++) pr_ens[i] = K_ZERO;
    int start_level = kte;
    int pmin_lev = 1;

    // ---- :561-582 : cup_env x2, cup_env_clev x2 ---------------------------
    float qes[GF_KP], he[GF_KP], hes[GF_KP];
    float qeso[GF_KP], heo[GF_KP], heso[GF_KP];
    gfd_cup_env(z, t, q, po, qes, he, hes, nz);
    gfd_cup_env(zo, tn, qo, po, qeso, heo, heso, nz);
    float qes_cup[GF_KP], q_cup[GF_KP], he_cup[GF_KP], hes_cup[GF_KP];
    float z_cup[GF_KP], p_cup[GF_KP], gamma_cup[GF_KP], t_cup[GF_KP];
    gfd_cup_env_clev(t, qes, q, he, hes, z, po, psur, z1, nz, qes_cup, q_cup,
                     he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup);
    float qeso_cup[GF_KP], qo_cup[GF_KP], heo_cup[GF_KP], heso_cup[GF_KP];
    float zo_cup[GF_KP], po_cup[GF_KP], gammao_cup[GF_KP], tn_cup[GF_KP];
    gfd_cup_env_clev(tn, qeso, qo, heo, heso, zo, po, psur, z1, nz, qeso_cup,
                     qo_cup, heo_cup, heso_cup, zo_cup, po_cup, gammao_cup,
                     tn_cup);
    CAPL(LEV_qes, qes); CAPL(LEV_he, he); CAPL(LEV_hes, hes);
    CAPL(LEV_qeso, qeso); CAPL(LEV_heo, heo); CAPL(LEV_heso, heso);
    CAPL(LEV_qes_cup, qes_cup); CAPL(LEV_q_cup, q_cup);
    CAPL(LEV_he_cup, he_cup); CAPL(LEV_hes_cup, hes_cup);
    CAPL(LEV_gamma_cup1, gamma_cup); CAPL(LEV_t_cup, t_cup);
    CAPL(LEV_qeso_cup, qeso_cup); CAPL(LEV_qo_cup, qo_cup);
    CAPL(LEV_heo_cup, heo_cup); CAPL(LEV_heso_cup, heso_cup);
    CAPL(LEV_zo_cup, zo_cup); CAPL(LEV_gammao_cup, gammao_cup);
    CAPL(LEV_tn_cup, tn_cup);
    // LEV_po_cup is captured at the END: the perturbed-state cup_env_clev
    // overwrites po_cup in place and the oracle records the overwritten
    // words (zeros on rejected columns).

    // ---- :583-615 ---------------------------------------------------------
    float u_cup[GF_KP], v_cup[GF_KP];
    for (int k = 0; k < GF_KP; k++) { u_cup[k] = K_ZERO; v_cup[k] = K_ZERO; }
    u_cup[1] = us[1];
    v_cup[1] = vs[1];
    for (int k = 2; k <= ktf; k++) {
        u_cup[k] = FMUL(K_HALF, FADD(us[k - 1], us[k]));
        v_cup[k] = FMUL(K_HALF, FADD(vs[k - 1], vs[k]));
    }
    CAPL(LEV_u_cup, u_cup); CAPL(LEV_v_cup, v_cup);
    for (int k = 1; k <= ktf; k++)
        if (zo_cup[k] > FADD(zkbmax, z1)) { kbmax = k; break; }
    for (int k = 1; k <= ktf; k++)
        if (zo_cup[k] > FADD(z_detr, z1)) { kdet = k; break; }
    ISCB[ISCA_kbmax] = kbmax;

    // ---- :621-633 : k22 ---------------------------------------------------
    k22 = gfd_maxloc(heo_cup, 2, kbmax + 2);
    if (k22 >= kbmax) {
        ierr = 2;
        ktop = 0;
        k22 = 0;
        kbcon = 0;
    }
    ISCB[ISCA_k22_0] = k22;

    // ---- :638-644 ---------------------------------------------------------
    float hkb = K_ZERO, hkbo = K_ZERO;
    if (ierr == 0) {
        float x_add = FADD(FMUL(K_XLV, zqexec), FMUL(K_CP, ztexec));
        hkb = gfd_get_cloud_bc(he_cup, k22, x_add, 1);
        hkbo = gfd_get_cloud_bc(heo_cup, k22, x_add, 1);
    }
    SCAB[SCA_hkb0] = hkb;
    SCAB[SCA_hkbo0] = hkbo;

    // ---- :648-653 : cup_kbcon --------------------------------------------
    float hcot_s[GF_KP];
    gfd_cup_kbcon(cap_max_increment, 1, &k22, heo_cup, heso_cup, &hkbo,
                  &ierr, kbmax, po_cup, cap_max, ztexec, zqexec, z_cup,
                  entr_rate, heo, nz, hcot_s, &kbcon);
    ISCB[ISCA_kbcon_1] = kbcon;
    ISCB[ISCA_k22_1] = k22;
    ISCB[ISCA_ierr_1] = ierr;

    // ---- :657-659 ---------------------------------------------------------
    int kstabi = gfd_cup_minimi(heso_cup, kbcon, kstabm, ierr);
    ISCB[ISCA_kstabi] = kstabi;
    ISCB[ISCA_kstabm] = kstabm;

    // ---- :660-685 ---------------------------------------------------------
    float frh_kb = K_ZERO;
    if (ierr == 0) {
        frh_kb = GMIN(FDIV(qo_cup[kbcon], qeso_cup[kbcon]), K_ONE);
        if (frh_kb >= K_RH_THRESH && sig <= sig_thresh) {
            ierr = 231;
        } else {
            for (int k = kbcon + 1; k <= ktf; k++) {
                if (FSUB(po[kbcon], po[k]) > K_P150) {
                    pmin_lev = k;
                    break;
                }
            }
            start_level = k22;
            float x_add = FADD(FMUL(K_XLV, zqexec), FMUL(K_CP, ztexec));
            hkb = gfd_get_cloud_bc(he_cup, k22, x_add, 1);
        }
    }
    SCAB[SCA_frh_kb] = frh_kb;
    ISCB[ISCA_pmin_lev] = pmin_lev;

    // ---- :693-726 ---------------------------------------------------------
    if (kstabi < kbcon) {
        kbcon = 1;
        ierr = 42;
    }
    float entr_rate_2d[GF_KP];
    for (int k = 0; k < GF_KP; k++)
        entr_rate_2d[k] = (k == 0) ? K_ZERO : entr_rate;
    if (ierr == 0) {
        kbcon = IMAX(2, kbcon);
        for (int k = 1; k <= ktf; k++) {
            float f = GMIN(FDIV(qo_cup[k], qeso_cup[k]), K_ONE);
            entr_rate_2d[k] = FMUL(entr_rate, FSUB(K_ONE_P3, f));
        }
    }
    CAPL(LEV_entr2d_a, entr_rate_2d);
    ISCB[ISCA_start_level] = start_level;

    // ---- :737-738 : rates_up_pdf -----------------------------------------
    GfdPdf up_pdf;
    up_pdf.tunning = K_ZERO; up_pdf.alpha = K_ZERO; up_pdf.beta = K_ZERO;
    up_pdf.fzu = K_ZERO; up_pdf.kb_adj = 0;
    int ktopdby = 0, up_kklev = 0, up_kfinal = 0;
    {
        float dby_s[GF_KP], dbm_s[GF_KP];
        gfd_rates_up_pdf_deep(&ktop, &ierr, po_cup, entr_rate_2d, hkbo, heo,
                              heso_cup, zo_cup, kstabi, k22, &kbcon, csum,
                              nz, ktf, fzu_up, zuo, &ktopdby, &up_pdf,
                              &up_kklev, &up_kfinal, dby_s, dbm_s, hcot_s);
    }
    CAPL(LEV_zu_pdf, zuo);
    ISCB[ISCA_ktop_pdf] = ktop;
    ISCB[ISCA_ktopdby] = ktopdby;
    ISCB[ISCA_kbcon_2] = kbcon;
    ISCB[ISCA_ierr_2] = ierr;
    SCAB[SCA_up_tun] = up_pdf.tunning;
    SCAB[SCA_up_alpha] = up_pdf.alpha;
    SCAB[SCA_up_beta] = up_pdf.beta;
    SCAB[SCA_up_fzu] = up_pdf.fzu;
    ISCB[ISCA_up_kbadj] = up_pdf.kb_adj;
    ISCB[ISCA_up_kklev] = up_kklev;
    ISCB[ISCA_up_kfinal] = up_kfinal;

    // ---- :743-763 ---------------------------------------------------------
    float zu[GF_KP], xzu[GF_KP];
    for (int k = 0; k < GF_KP; k++) { zu[k] = K_ZERO; xzu[k] = K_ZERO; }
    if (ierr == 0) {
        if (k22 > 1)
            for (int k = 1; k < k22; k++) zuo[k] = K_ZERO;
        for (int k = k22; k <= ktop; k++) {
            xzu[k] = zuo[k];
            zu[k] = zuo[k];
        }
        for (int k = ktop + 1; k < GF_KP; k++) zuo[k] = K_ZERO;
    }

    // ---- :767-770 : get_lateral_massflux ---------------------------------
    float upme[GF_KP], upmd[GF_KP], upmeu[GF_KP], upmdu[GF_KP];
    gfd_get_lateral_massflux(ierr, ktop, zo_cup, zuo, cd, entr_rate_2d,
                             kbcon, k22, nz, ktf, lambau, 1, upme, upmd,
                             upmeu, upmdu);

    // ---- :777-852 : the in-cloud updraft ---------------------------------
    float uc[GF_KP], vc[GF_KP], hc[GF_KP], dby[GF_KP];
    float hco[GF_KP], dbyo[GF_KP], dbyt[GF_KP];
    for (int k = 0; k < GF_KP; k++) {
        uc[k] = K_ZERO; vc[k] = K_ZERO; hc[k] = K_ZERO; dby[k] = K_ZERO;
        hco[k] = K_ZERO; dbyo[k] = K_ZERO; dbyt[k] = K_ZERO;
    }
    if (ierr == 0) {
        for (int k = 1; k <= start_level; k++) {
            uc[k] = u_cup[k];
            vc[k] = v_cup[k];
        }
        for (int k = 1; k < start_level; k++) {
            hc[k] = he_cup[k];
            hco[k] = heo_cup[k];
        }
        hc[start_level] = hkb;
        hco[start_level] = hkbo;
    }
    int ktopkeep = 0;
    if (ierr == 0) {
        ktopkeep = ktop;
        for (int k = start_level + 1; k <= ktop; k++) {
            float denom = FADD(FSUB(zuo[k - 1], FMUL(K_HALF, upmd[k - 1])),
                               upme[k - 1]);
            if (denom < K_E1M8) { ierr = 51; break; }
            float du = FADD(FSUB(zu[k - 1], FMUL(K_HALF, upmd[k - 1])),
                            upme[k - 1]);
            float duu = FADD(FSUB(zu[k - 1], FMUL(K_HALF, upmdu[k - 1])),
                             upmeu[k - 1]);
            hc[k] = FDIV(
                FADD(FSUB(FMUL(hc[k - 1], zu[k - 1]),
                          FMUL(FMUL(K_HALF, upmd[k - 1]), hc[k - 1])),
                     FMUL(upme[k - 1], he[k - 1])),
                du);
            uc[k] = FDIV(
                FSUB(FADD(FSUB(FMUL(uc[k - 1], zu[k - 1]),
                               FMUL(FMUL(K_HALF, upmdu[k - 1]), uc[k - 1])),
                          FMUL(upmeu[k - 1], us[k - 1])),
                     FMUL(FMUL(FMUL(pgcon, K_HALF), FADD(zu[k], zu[k - 1])),
                          FSUB(u_cup[k], u_cup[k - 1]))),
                duu);
            vc[k] = FDIV(
                FSUB(FADD(FSUB(FMUL(vc[k - 1], zu[k - 1]),
                               FMUL(FMUL(K_HALF, upmdu[k - 1]), vc[k - 1])),
                          FMUL(upmeu[k - 1], vs[k - 1])),
                     FMUL(FMUL(FMUL(pgcon, K_HALF), FADD(zu[k], zu[k - 1])),
                          FSUB(v_cup[k], v_cup[k - 1]))),
                duu);
            dby[k] = FSUB(hc[k], hes_cup[k]);
            hco[k] = FDIV(
                FADD(FSUB(FMUL(hco[k - 1], zuo[k - 1]),
                          FMUL(FMUL(K_HALF, upmd[k - 1]), hco[k - 1])),
                     FMUL(upme[k - 1], heo[k - 1])),
                denom);
            dbyo[k] = FSUB(hco[k], heso_cup[k]);
            float dz = FSUB(zo_cup[k + 1], zo_cup[k]);
            dbyt[k] = FADD(dbyt[k - 1], FMUL(dbyo[k], dz));
        }
        for (int k = ktop - 1; k >= kbcon; k--) {
            if (dbyo[k] > K_ZERO) {
                ktopkeep = k + 1;
                break;
            }
        }
        ktop = ktopkeep;
    }
    ISCB[ISCA_ktop_dbyt] = ktop;
    ISCB[ISCA_ierr_3] = ierr;

    // ---- :854-881 ---------------------------------------------------------
    if (ierr == 0) {
        for (int k = ktop + 1; k <= ktf; k++) {
            hc[k] = hes_cup[k];
            uc[k] = u_cup[k];
            vc[k] = v_cup[k];
            hco[k] = heso_cup[k];
            dby[k] = K_ZERO;
            dbyo[k] = K_ZERO;
            zu[k] = K_ZERO;
            zuo[k] = K_ZERO;
            cd[k] = K_ZERO;
            entr_rate_2d[k] = K_ZERO;
            upme[k] = K_ZERO;
            upmd[k] = K_ZERO;
        }
        if (ktop < kbcon + 2) {
            ierr = 5;
            ktop = 0;
        }
    }
    CAPL(LEV_entr2d_b, entr_rate_2d);
    CAPL(LEV_cd, cd);
    CAPL(LEV_upme, upme);
    CAPL(LEV_upmd, upmd);
    CAPL(LEV_upmeu, upmeu);
    CAPL(LEV_upmdu, upmdu);
    CAPL(LEV_hc, hc);
    CAPL(LEV_uc, uc);
    CAPL(LEV_vc, vc);
    CAPL(LEV_hco, hco);
    CAPL(LEV_dby, dby);
    CAPL(LEV_dbyo, dbyo);
    CAPL(LEV_dbyt, dbyt);

    // ---- :882-896 : kzdown -----------------------------------------------
    int kzdown = 0;
    if (ierr == 0) {
        float zktop = FMUL(FSUB(zo_cup[ktop], z1), K_P6);
        zktop = GMIN(FADD(zktop, z1), FADD(zcutdown, z1));
        for (int k = 1; k <= ktf; k++) {
            if (zo_cup[k] > zktop) {
                kzdown = IMIN(k, kstabi - 1);
                break;
            }
        }
    }
    ISCB[ISCA_kzdown] = kzdown;

    // ---- :900-941 : jmin -------------------------------------------------
    jmin = gfd_cup_minimi(heso_cup, k22, kzdown, ierr);
    if (ierr == 0) {
        int jmini = jmin;
        bool keep_going = true;
        while (keep_going) {
            keep_going = false;
            if (jmini - 1 < kdet) kdet = jmini - 1;
            if (jmini >= ktop - 1) jmini = ktop - 2;
            int ki = jmini;
            hcdo[ki] = heso_cup[ki];
            float dh = K_ZERO;
            for (int k = ki - 1; k >= 1; k--) {
                hcdo[k] = heso_cup[jmini];
                float dz = FSUB(zo_cup[k + 1], zo_cup[k]);
                dh = FADD(dh, FMUL(dz, FSUB(hcdo[k], heso_cup[k])));
                if (dh > K_ZERO) {
                    jmini -= 1;
                    if (jmini > 5) {
                        keep_going = true;
                    } else {
                        ierr = 9;
                    }
                    break;
                }
            }
        }
        jmin = jmini;
        if (jmini <= 5) ierr = 4;
    }

    // ---- :946-954 ---------------------------------------------------------
    if (ierr == 0) {
        if (jmin - 1 < kdet) kdet = jmin - 1;
        if (FADD(-zo_cup[kbcon], zo_cup[ktop]) < depth_min) ierr = 6;
    }
    ISCB[ISCA_kdet_2] = kdet;
    ISCB[ISCA_ierr_4] = ierr;

    // ---- :960-1082 : the downdraft ---------------------------------------
    float ddme[GF_KP], ddmd[GF_KP], ddmeu[GF_KP], ddmdu[GF_KP];
    float mentrd_rate_2d[GF_KP], ucd[GF_KP], vcd[GF_KP], dbydo[GF_KP];
    for (int k = 0; k < GF_KP; k++) {
        zdo[k] = K_ZERO;
        cdd[k] = K_ZERO;
        ddme[k] = K_ZERO; ddmd[k] = K_ZERO;
        ddmeu[k] = K_ZERO; ddmdu[k] = K_ZERO;
        mentrd_rate_2d[k] = (k == 0) ? K_ZERO : entr_rate;
        hcdo[k] = heso_cup[k];
        ucd[k] = u_cup[k];
        vcd[k] = v_cup[k];
        dbydo[k] = K_ZERO;
    }
    float beta = GMAX(K_BETA_DN_MIN,
                      FSUB(K_BETA_DN_A, FMUL((float)csum, K_BETA_DN_B)));
    if (xland1 == 0)
        edtmax = GMAX(K_P1, FSUB(K_EDTMAX_LAND_A,
                                 FMUL((float)csum, K_EDTMAX_LAND_B)));
    float bud = K_ZERO;
    GfdPdf dn_pdf;
    dn_pdf.tunning = K_ZERO; dn_pdf.alpha = K_ZERO; dn_pdf.beta = K_ZERO;
    dn_pdf.fzu = K_ZERO; dn_pdf.kb_adj = 0;
    if (ierr == 0) {
        for (int k = 1; k <= jmin; k++) cdd[k] = K_E1M9;
        cdd[jmin] = K_ZERO;
        gfd_get_zu_zd_pdf(2, po_cup, kdet, jmin, kpbl, csum, K_ZERO, nz, ktf,
                          fzu_dn, zdo, &dn_pdf);
        bool skip = false;
        if (zdo[jmin] < K_E1M8) {
            zdo[jmin] = K_ZERO;
            jmin -= 1;
            if (zdo[jmin] < K_E1M8) {
                ierr = 876;
                skip = true;
            }
        }
        if (!skip) {
            int kpeak = gfd_maxloc(zdo, 1, nz);
            for (int ki = jmin; ki >= kpeak; ki--) {
                float dzo = FSUB(zo_cup[ki + 1], zo_cup[ki]);
                ddmd[ki] = FMUL(FMUL(cdd[ki], dzo), zdo[ki + 1]);
                ddme[ki] = FADD(FSUB(zdo[ki], zdo[ki + 1]), ddmd[ki]);
                if (ddme[ki] < K_ZERO) {
                    ddme[ki] = K_ZERO;
                    ddmd[ki] = FSUB(zdo[ki + 1], zdo[ki]);
                    if (zdo[ki + 1] > K_ZERO)
                        cdd[ki] = FDIV(ddmd[ki], FMUL(dzo, zdo[ki + 1]));
                }
                if (zdo[ki + 1] > K_ZERO)
                    mentrd_rate_2d[ki] = FDIV(ddme[ki],
                                              FMUL(dzo, zdo[ki + 1]));
            }
            mentrd_rate_2d[1] = K_ZERO;
            for (int ki = kpeak - 1; ki >= 1; ki--) {
                float dzo = FSUB(zo_cup[ki + 1], zo_cup[ki]);
                ddme[ki] = FMUL(FMUL(mentrd_rate_2d[ki], dzo), zdo[ki + 1]);
                ddmd[ki] = FSUB(FADD(zdo[ki + 1], ddme[ki]), zdo[ki]);
                if (ddmd[ki] < K_ZERO) {
                    ddmd[ki] = K_ZERO;
                    ddme[ki] = FSUB(zdo[ki], zdo[ki + 1]);
                    if (zdo[ki + 1] > K_ZERO)
                        mentrd_rate_2d[ki] = FDIV(ddme[ki],
                                                  FMUL(dzo, zdo[ki + 1]));
                }
                if (zdo[ki + 1] > K_ZERO)
                    cdd[ki] = FDIV(ddmd[ki], FMUL(dzo, zdo[ki + 1]));
            }
            for (int k = kbcon + 1; k <= ktop - 1; k++)
                c1d[k] = K_C1;
            for (int k = 2; k <= jmin + 1; k++) {
                ddmeu[k - 1] = FADD(ddme[k - 1], FMUL(lambau, ddmd[k - 1]));
                ddmdu[k - 1] = FADD(ddmd[k - 1], FMUL(lambau, ddmd[k - 1]));
            }
            dbydo[jmin] = FSUB(hcdo[jmin], heso_cup[jmin]);
            bud = FMUL(dbydo[jmin], FSUB(zo_cup[jmin + 1], zo_cup[jmin]));
            for (int ki = jmin; ki >= 1; ki--) {
                float dzo = FSUB(zo_cup[ki + 1], zo_cup[ki]);
                float h_entr = FMUL(K_HALF,
                    FADD(heo[ki], FMUL(K_HALF, FADD(hco[ki], hco[ki + 1]))));
                float denu = FADD(FSUB(zdo[ki + 1],
                                       FMUL(K_HALF, ddmdu[ki])), ddmeu[ki]);
                float deno = FADD(FSUB(zdo[ki + 1],
                                       FMUL(K_HALF, ddmd[ki])), ddme[ki]);
                ucd[ki] = FDIV(
                    FSUB(FADD(FSUB(FMUL(ucd[ki + 1], zdo[ki + 1]),
                                   FMUL(FMUL(K_HALF, ddmdu[ki]),
                                        ucd[ki + 1])),
                              FMUL(ddmeu[ki], us[ki])),
                         FMUL(FMUL(pgcon, zdo[ki + 1]),
                              FSUB(us[ki + 1], us[ki]))),
                    denu);
                vcd[ki] = FDIV(
                    FSUB(FADD(FSUB(FMUL(vcd[ki + 1], zdo[ki + 1]),
                                   FMUL(FMUL(K_HALF, ddmdu[ki]),
                                        vcd[ki + 1])),
                              FMUL(ddmeu[ki], vs[ki])),
                         FMUL(FMUL(pgcon, zdo[ki + 1]),
                              FSUB(vs[ki + 1], vs[ki]))),
                    denu);
                hcdo[ki] = FDIV(
                    FADD(FSUB(FMUL(hcdo[ki + 1], zdo[ki + 1]),
                              FMUL(FMUL(K_HALF, ddmd[ki]), hcdo[ki + 1])),
                         FMUL(ddme[ki], h_entr)),
                    deno);
                dbydo[ki] = FSUB(hcdo[ki], heso_cup[ki]);
                bud = FADD(bud, FMUL(dbydo[ki], dzo));
            }
        }
    }
    if (bud > K_ZERO) ierr = 7;
    ISCB[ISCA_jmin] = jmin;
    CAPL(LEV_cdd, cdd);
    CAPL(LEV_ddme, ddme);
    CAPL(LEV_ddmd, ddmd);
    CAPL(LEV_ddmeu, ddmeu);
    CAPL(LEV_ddmdu, ddmdu);
    CAPL(LEV_mentrd2d, mentrd_rate_2d);
    CAPL(LEV_hcdo, hcdo);
    CAPL(LEV_ucd, ucd);
    CAPL(LEV_vcd, vcd);
    CAPL(LEV_dbydo, dbydo);
    CAPL(LEV_c1d, c1d);
    SCAB[SCA_bud] = bud;
    SCAB[SCA_beta] = beta;
    SCAB[SCA_edtmax] = edtmax;
    ISCB[ISCA_ierr_5] = ierr;
    SCAB[SCA_dn_tun] = dn_pdf.tunning;
    SCAB[SCA_dn_alpha] = dn_pdf.alpha;
    SCAB[SCA_dn_beta] = dn_pdf.beta;
    SCAB[SCA_dn_fzu] = dn_pdf.fzu;
    ISCB[ISCA_dn_kbadj] = dn_pdf.kb_adj;

    // ---- :1086-1090 : cup_dd_moisture ------------------------------------
    float qcdo[GF_KP], pwdo[GF_KP];
    float pwevo = K_ZERO, bu = K_ZERO;
    gfd_cup_dd_moisture(&ierr, zdo, hcdo, heso_cup, qeso_cup, qo_cup, zo_cup,
                        ddme, ddmd, jmin, gammao_cup, qo, nz, qcdo, qrcdo,
                        pwdo, &pwevo, &bu);
    CAPL(LEV_qcdo, qcdo);
    CAPL(LEV_qrcdo, qrcdo);
    CAPL(LEV_pwdo, pwdo);
    SCAB[SCA_pwevo] = pwevo;
    SCAB[SCA_bu] = bu;

    // ---- :1102-1107 : cup_up_moisture ------------------------------------
    float qco[GF_KP], qrco[GF_KP], pwo[GF_KP], clw_all[GF_KP], qch_s[GF_KP];
    float pwavo = K_ZERO, psum = K_ZERO, psumh = K_ZERO;
    gfd_cup_up_moisture(&ierr, zo_cup, p_cup, kbcon, ktop, dbyo, xland1, qo,
                        gammao_cup, zuo, qeso_cup, k22, qo_cup, zqexec, ccn,
                        rho, c1d, tn_cup, upme, upmd, nz, qco, qrco, pwo,
                        clw_all, qch_s, &pwavo, &psum, &psumh);
    CAPL(LEV_qco, qco);
    CAPL(LEV_qrco, qrco);
    CAPL(LEV_pwo, pwo);
    CAPL(LEV_clw_all, clw_all);
    SCAB[SCA_pwavo] = pwavo;
    SCAB[SCA_psum] = psum;
    SCAB[SCA_psumh] = psumh;

    // ---- :1109-1117 -------------------------------------------------------
    if (ierr == 0) {
        float dp = FMUL(K_HUNDRED, FSUB(po_cup[1], po_cup[2]));
        for (int k = 2; k <= ktop; k++) {
            cupclw[k] = qrco[k];
            cnvwt[k] = FDIV(FMUL(FMUL(zuo[k], cupclw[k]), K_G), dp);
        }
    }
    CAPL(LEV_cupclw, cupclw);
    CAPL(LEV_cnvwt, cnvwt);

    // ---- :1121-1136 : cup_up_aa0 x2 --------------------------------------
    aa0 = gfd_cup_up_aa0(z, zu, dby, gamma_cup, t_cup, kbcon, ktop, ierr,
                         ktf);
    aa1 = gfd_cup_up_aa0(zo, zuo, dbyo, gammao_cup, tn_cup, kbcon, ktop,
                         ierr, ktf);
    if (ierr == 0 && aa1 == K_ZERO) ierr = 17;
    SCAB[SCA_aa0] = aa0;
    SCAB[SCA_aa1] = aa1;
    ISCB[ISCA_ierr_6] = ierr;
    int ierr6_saved = ierr;

    // ---- :1141-1203 : the diurnal-cycle closure --------------------------
    float aa1_bl = K_ZERO;
    float xf_dicycle = K_ZERO;
    float tau_ecmwf = K_ZERO;
    float tau_bl = K_ZERO;
    float umean = K_ZERO;
    float wmean = K_WMEAN;
    if (ierr == 0) {
        tau_ecmwf = FDIV(FSUB(zo_cup[ktopdby], zo_cup[kbcon]), wmean);
        tau_ecmwf = FMUL(tau_ecmwf,
                         FADD(K_TAU_A,
                              FMUL(K_TAU_B, FDIV(dx, K_THOUSAND))));
        if (xland1 == 0) {
            umean = FADD(K_TWO,
                FSQRT(FMUL(K_TWO,
                    FADD(FADD(FADD(FMUL(us[1], us[1]), FMUL(vs[1], vs[1])),
                              FMUL(us[kbcon], us[kbcon])),
                         FMUL(vs[kbcon], vs[kbcon])))));
            tau_bl = FDIV(FSUB(zo_cup[kbcon], z1), umean);
        } else {
            tau_bl = FDIV(FSUB(zo_cup[ktopdby], zo_cup[kbcon]), wmean);
        }
    }
    float t_star = K_T_STAR;
    aa1_bl = gfd_cup_up_aa1bl(t, tn, q, qo, dtime, zo_cup, kbcon, ierr, ktf);
    if (ierr == 0) {
        if (FSUB(zo_cup[kbcon], z1) > zo[IMIN(kte, kpbl + 1)]) {
            aa1_bl = K_ZERO;
        } else {
            aa1_bl = GMAX(K_ZERO, FMUL(FDIV(aa1_bl, t_star), tau_bl));
        }
    }
    float axx = aa1;
    SCAB[SCA_tau_ecmwf] = tau_ecmwf;
    SCAB[SCA_tau_bl] = tau_bl;
    SCAB[SCA_aa1_bl] = aa1_bl;
    SCAB[SCA_umean] = umean;

    // ---- :1297-1305 : cup_dd_edt -----------------------------------------
    float edt = K_ZERO, edtc = K_ZERO;
    gfd_cup_dd_edt(ierr, us, vs, zo, ktop, kbcon, po, pwavo, pwevo, edtmax,
                   edtmin, ktf, &edt, &edtc);
    float edto = (ierr == 0) ? edtc : K_ZERO;
    SCAB[SCA_edt] = edt;
    SCAB[SCA_edtc1] = edtc;
    SCAB[SCA_edto] = edto;

    // ---- :1369-1495 : the della fields -----------------------------------
    float dellu[GF_KP], dellv[GF_KP], dellah[GF_KP], dellaq[GF_KP];
    float dellat[GF_KP];
    for (int k = 0; k < GF_KP; k++) {
        dellu[k] = K_ZERO; dellv[k] = K_ZERO; dellah[k] = K_ZERO;
        dellaq[k] = K_ZERO; dellat[k] = K_ZERO;
    }
    if (ierr == 0) {
        float dp = FMUL(K_HUNDRED, FSUB(po_cup[1], po_cup[2]));
        dellu[1] = FDIV(
            FMUL(FMUL(K_PGCD,
                      FSUB(FMUL(FMUL(edto, zdo[2]), ucd[2]),
                           FMUL(FMUL(edto, zdo[2]), u_cup[2]))),
                 K_G),
            dp);
        dellv[1] = FDIV(
            FMUL(FMUL(K_PGCD,
                      FSUB(FMUL(FMUL(edto, zdo[2]), vcd[2]),
                           FMUL(FMUL(edto, zdo[2]), v_cup[2]))),
                 K_G),
            dp);
        for (int k = 2; k <= ktop; k++) {
            dp = FMUL(K_HUNDRED, FSUB(po_cup[k], po_cup[k + 1]));
            dellu[k] = FADD(
                FDIV(-FMUL(FSUB(FMUL(zuo[k + 1],
                                     FSUB(uc[k + 1], u_cup[k + 1])),
                                FMUL(zuo[k], FSUB(uc[k], u_cup[k]))),
                           K_G),
                     dp),
                FMUL(FDIV(FMUL(FSUB(FMUL(zdo[k + 1],
                                         FSUB(ucd[k + 1], u_cup[k + 1])),
                                    FMUL(zdo[k], FSUB(ucd[k], u_cup[k]))),
                               K_G),
                          dp),
                     FMUL(edto, K_PGCD)));
            dellv[k] = FADD(
                FDIV(-FMUL(FSUB(FMUL(zuo[k + 1],
                                     FSUB(vc[k + 1], v_cup[k + 1])),
                                FMUL(zuo[k], FSUB(vc[k], v_cup[k]))),
                           K_G),
                     dp),
                FMUL(FDIV(FMUL(FSUB(FMUL(zdo[k + 1],
                                         FSUB(vcd[k + 1], v_cup[k + 1])),
                                    FMUL(zdo[k], FSUB(vcd[k], v_cup[k]))),
                               K_G),
                          dp),
                     FMUL(edto, K_PGCD)));
        }
    }
    if (ierr == 0) {
        float dp = FMUL(K_HUNDRED, FSUB(po_cup[1], po_cup[2]));
        dellah[1] = FDIV(
            FMUL(FSUB(FMUL(FMUL(edto, zdo[2]), hcdo[2]),
                      FMUL(FMUL(edto, zdo[2]), heo_cup[2])),
                 K_G),
            dp);
        dellaq[1] = FDIV(
            FMUL(FSUB(FMUL(FMUL(edto, zdo[2]), qcdo[2]),
                      FMUL(FMUL(edto, zdo[2]), qo_cup[2])),
                 K_G),
            dp);
        float g_rain = FDIV(FMUL(FMUL(K_HALF, FADD(pwo[1], pwo[2])), K_G),
                            dp);
        float e_dn = FMUL(
            FDIV(FMUL(FMUL(K_NEG_HALF, FADD(pwdo[1], pwdo[2])), K_G), dp),
            edto);
        dellaq[1] = FSUB(FADD(dellaq[1], e_dn), g_rain);
        for (int k = 2; k <= ktop; k++) {
            dp = FMUL(K_HUNDRED, FSUB(po_cup[k], po_cup[k + 1]));
            dellah[k] = FADD(
                FDIV(-FMUL(FSUB(FMUL(zuo[k + 1],
                                     FSUB(hco[k + 1], heo_cup[k + 1])),
                                FMUL(zuo[k], FSUB(hco[k], heo_cup[k]))),
                           K_G),
                     dp),
                FMUL(FDIV(FMUL(FSUB(FMUL(zdo[k + 1],
                                         FSUB(hcdo[k + 1], heo_cup[k + 1])),
                                    FMUL(zdo[k],
                                         FSUB(hcdo[k], heo_cup[k]))),
                               K_G),
                          dp),
                     edto));
            float detup = upmd[k];
            float dz = FSUB(zo_cup[k], zo_cup[k - 1]);
            if (k < ktop)
                dellaqc[k] = FMUL(
                    FDIV(FMUL(FMUL(FMUL(zuo[k], c1d[k]), qrco[k]), dz), dp),
                    K_G);
            if (k == ktop)
                dellaqc[k] = FDIV(
                    FMUL(FMUL(FMUL(detup, K_HALF),
                              FADD(qrco[k + 1], qrco[k])),
                         K_G),
                    dp);
            float g_rain2 = FDIV(FMUL(FMUL(K_HALF, FADD(pwo[k], pwo[k + 1])),
                                      K_G),
                                 dp);
            float e_dn2 = FMUL(
                FDIV(FMUL(FMUL(K_NEG_HALF, FADD(pwdo[k], pwdo[k + 1])), K_G),
                     dp),
                edto);
            float c_up = FADD(
                FADD(dellaqc[k],
                     FDIV(FMUL(FSUB(FMUL(zuo[k + 1], qrco[k + 1]),
                                    FMUL(zuo[k], qrco[k])),
                               K_G),
                          dp)),
                g_rain2);
            dellaq[k] = FADD(
                FSUB(FADD(
                    FDIV(-FMUL(FSUB(FMUL(zuo[k + 1],
                                         FSUB(qco[k + 1], qo_cup[k + 1])),
                                    FMUL(zuo[k], FSUB(qco[k], qo_cup[k]))),
                               K_G),
                         dp),
                    FMUL(FDIV(FMUL(FSUB(FMUL(zdo[k + 1],
                                             FSUB(qcdo[k + 1],
                                                  qo_cup[k + 1])),
                                        FMUL(zdo[k],
                                             FSUB(qcdo[k], qo_cup[k]))),
                                   K_G),
                              dp),
                         edto)),
                    c_up),
                e_dn2);
        }
    }
    CAPL(LEV_dellu, dellu);
    CAPL(LEV_dellv, dellv);
    CAPL(LEV_dellah, dellah);
    CAPL(LEV_dellaq, dellaq);
    CAPL(LEV_dellaqc, dellaqc);

    // ---- :1500-1524 : the mbdt-perturbed state ---------------------------
    float mbdt = K_MBDT;
    float xaa0_ens = K_ZERO;
    float xhe[GF_KP], xq[GF_KP], xt[GF_KP];
    for (int k = 0; k < GF_KP; k++) {
        xhe[k] = K_ZERO; xq[k] = K_ZERO; xt[k] = K_ZERO;
    }
    if (ierr == 0) {
        for (int k = 1; k <= ktf; k++) {
            xhe[k] = FADD(FMUL(dellah[k], mbdt), heo[k]);
            xq[k] = GMAX(K_E1M16, FADD(FMUL(dellaq[k], mbdt), qo[k]));
            dellat[k] = FMUL(FDIV(K_ONE, K_CP),
                             FSUB(dellah[k], FMUL(K_XLV, dellaq[k])));
            xt[k] = FADD(FMUL(dellat[k], mbdt), tn[k]);
            xt[k] = GMAX(K_T190, xt[k]);
        }
        xhe[ktf] = heo[ktf];
        xq[ktf] = qo[ktf];
        xt[ktf] = tn[ktf];
    }
    CAPL(LEV_dellat, dellat);
    CAPL(LEV_xq, xq);
    CAPL(LEV_xt, xt);

    // ---- :1528-1539 -------------------------------------------------------
    // cup_env OVERWRITES xhe on the perturbed state, and the third
    // cup_env_clev writes po_cup and gamma_cup in place -- zeros on rejected
    // columns, identical words where ierr == 0.
    float xqes[GF_KP], xhes[GF_KP];
    float xqes_cup[GF_KP], xq_cup[GF_KP], xhe_cup[GF_KP], xhes_cup[GF_KP];
    float xt_cup[GF_KP], xz_cup_scratch[GF_KP];
    if (ierr == 0) {
        gfd_cup_env(xz, xt, xq, po, xqes, xhe, xhes, nz);
        gfd_cup_env_clev(xt, xqes, xq, xhe, xhes, xz, po, psur, z1, nz,
                         xqes_cup, xq_cup, xhe_cup, xhes_cup, xz_cup_scratch,
                         po_cup, gamma_cup, xt_cup);
    } else {
        for (int k = 0; k < GF_KP; k++) {
            xqes[k] = K_ZERO; xhes[k] = K_ZERO;
            xqes_cup[k] = K_ZERO; xq_cup[k] = K_ZERO;
            xhe_cup[k] = K_ZERO; xhes_cup[k] = K_ZERO;
            gamma_cup[k] = K_ZERO; xt_cup[k] = K_ZERO;
            po_cup[k] = K_ZERO;
        }
    }
    CAPL(LEV_xhe, xhe);
    CAPL(LEV_xqes, xqes);
    CAPL(LEV_xhes, xhes);
    CAPL(LEV_xqes_cup, xqes_cup);
    CAPL(LEV_xq_cup, xq_cup);
    CAPL(LEV_xhe_cup, xhe_cup);
    CAPL(LEV_xhes_cup, xhes_cup);
    CAPL(LEV_gamma_cupx, gamma_cup);
    CAPL(LEV_xt_cup, xt_cup);
    CAPL(LEV_po_cup, po_cup);

    // ---- :1546-1578 -------------------------------------------------------
    float xhc[GF_KP], xdby[GF_KP];
    for (int k = 0; k < GF_KP; k++) { xhc[k] = K_ZERO; xdby[k] = K_ZERO; }
    float xhkb = K_ZERO;
    if (ierr == 0) {
        float x_add = FADD(FMUL(K_XLV, zqexec), FMUL(K_CP, ztexec));
        xhkb = gfd_get_cloud_bc(xhe_cup, k22, x_add, 1);
        for (int k = 1; k < start_level; k++)
            xhc[k] = xhe_cup[k];
        xhc[start_level] = xhkb;
        for (int k = start_level + 1; k <= ktop; k++) {
            xhc[k] = FDIV(
                FADD(FSUB(FMUL(xhc[k - 1], xzu[k - 1]),
                          FMUL(FMUL(K_HALF, upmd[k - 1]), xhc[k - 1])),
                     FMUL(upme[k - 1], xhe[k - 1])),
                FADD(FSUB(xzu[k - 1], FMUL(K_HALF, upmd[k - 1])),
                     upme[k - 1]));
            xdby[k] = FSUB(xhc[k], xhes_cup[k]);
        }
        for (int k = ktop + 1; k <= ktf; k++) {
            xhc[k] = xhes_cup[k];
            xdby[k] = K_ZERO;
        }
    }
    CAPL(LEV_xhc, xhc);
    CAPL(LEV_xdby, xdby);
    SCAB[SCA_xhkb] = xhkb;

    // ---- :1583-1623 -------------------------------------------------------
    float xaa0 = gfd_cup_up_aa0(xz, xzu, xdby, gamma_cup, xt_cup, kbcon,
                                ktop, ierr, ktf);
    if (ierr == 0) {
        xaa0_ens = xaa0;
        for (int k = 1; k <= ktop; k++) {
            for (int nn = 1; nn <= GF_MAXENS3; nn++) {
                // (pr_ens + pwo) + edto*pwdo -- the accumulator absorbs pwo
                // first and the downdraft term second.
                pr_ens[nn] = FADD(FADD(pr_ens[nn], pwo[k]),
                                  FMUL(edto, pwdo[k]));
            }
        }
        if (pr_ens[7] < K_E1M6) {
            ierr = 18;
            for (int nn = 0; nn <= GF_MAXENS3; nn++) pr_ens[nn] = K_ZERO;
        }
        for (int nn = 1; nn <= GF_MAXENS3; nn++)
            if (pr_ens[nn] < K_E1M5) pr_ens[nn] = K_ZERO;
    }
    SCAB[SCA_xaa0] = xaa0;
    SCAB[SCA_pr7] = pr_ens[7];
    ISCB[ISCA_ierr_7] = ierr;

    // ---- :1633-1654 : the ierr2 / ierr3 cap probes -----------------------
    ierr2 = ierr;
    ierr3 = ierr;
    int k22x = gfd_cup_maximi(heo_cup, 2, kbmax, ierr);
    int kbconx = 0;
    int k22x2 = k22x;
    gfd_cup_kbcon(cap_max_increment, 2, &k22x2, heo_cup, heso_cup, &hkbo,
                  &ierr2, kbmax, po_cup, cap_max, ztexec, zqexec, z_cup,
                  entr_rate, heo, nz, hcot_s, &kbconx);
    int k22x_final = k22x2;
    gfd_cup_kbcon(cap_max_increment, 3, &k22x_final, heo_cup, heso_cup,
                  &hkbo, &ierr3, kbmax, po_cup, cap_max, ztexec, zqexec,
                  z_cup, entr_rate, heo, nz, hcot_s, &kbconx);
    k22x = k22x_final;
    ISCB[ISCA_k22x] = k22x;
    ISCB[ISCA_kbconx] = kbconx;
    ISCB[ISCA_ierr2] = ierr2;
    ISCB[ISCA_ierr3] = ierr3;

    // ---- :1659-1666 : mconv on the cloud grid, with the DEEP g -----------
    float mconv2 = K_ZERO;
    if (ierr == 0) {
        for (int k = 1; k <= ktop; k++) {
            float dq = FSUB(qo_cup[k + 1], qo_cup[k]);
            mconv2 = FADD(mconv2, FDIV(FMUL(omeg[k], dq), K_G));
        }
    }
    SCAB[SCA_mconv2] = mconv2;

    // ---- :1667-1674 -------------------------------------------------------
    float xf_ens[GF_MAXENS3 + 1];
    float forcing[11];
    gfd_cup_forcing_ens_3d(&closure_n, xland1, aa0, aa1, xaa0_ens, mbdt,
                           dtime, ierr, ierr2, ierr3, axx, mconv2, po_cup,
                           ktop, omeg, zdo, k22, zuo, pr_ens, edto, kbcon,
                           ichoice, dicycle, tau_ecmwf, aa1_bl, nz, xf_ens,
                           forcing, &xf_dicycle);
    for (int i = 1; i <= 10; i++) SCAB[SCA_f1 + (i - 1)] = forcing[i];
    SCAB[SCA_xf_dicycle] = xf_dicycle;
    SCAB[SCA_closure_n] = closure_n;

    // ---- :1715-1723 -------------------------------------------------------
    float pre = K_ZERO, xmb = K_ZERO;
    gfd_cup_output_ens_3d(xf_ens, &ierr, dellat, dellaq, dellaqc, zuo, pwo,
                          ktop, edto, pwdo, po_cup, pr_ens, sig, closure_n,
                          xmbs_in, dicycle, xf_dicycle, nz, outt, outq,
                          outqc, &pre, &xmb);
    CAPL(LEV_outt_o, outt);
    CAPL(LEV_outq_o, outq);
    CAPL(LEV_outqc_o, outqc);
    for (int i = 1; i <= GF_MAXENS3; i++) SCAB[SCA_xf1 + (i - 1)] = xf_ens[i];
    SCAB[SCA_xmb] = xmb;

    // ---- :1724-1743 -------------------------------------------------------
    for (int k = 0; k < GF_KP; k++) { outu[k] = K_ZERO; outv[k] = K_ZERO; }
    float xmb_out = K_ZERO;
    if (ierr == 0 && pre > K_ZERO) {
        pre = GMAX(pre, K_ZERO);
        xmb_out = xmb;
        for (int k = 1; k <= ktop; k++) {
            outu[k] = FMUL(dellu[k], xmb);
            outv[k] = FMUL(dellv[k], xmb);
        }
    } else if (ierr != 0 || pre == K_ZERO) {
        ktop = 0;
        for (int k = 0; k < GF_KP; k++) {
            outt[k] = K_ZERO;
            outq[k] = K_ZERO;
            outqc[k] = K_ZERO;
            outu[k] = K_ZERO;
            outv[k] = K_ZERO;
        }
    }
    (void)xmb_out;

    // ---- :1803-1821 : dissipative heating --------------------------------
    if (ierr == 0) {
        float dts = K_ZERO;
        float fpi = K_ZERO;
        for (int k = 1; k <= ktop; k++) {
            float dp = FMUL(FSUB(po_cup[k], po_cup[k + 1]), K_HUNDRED);
            dts = FSUB(dts,
                FDIV(FMUL(FADD(FMUL(outu[k], us[k]), FMUL(outv[k], vs[k])),
                          dp),
                     K_G));
            fpi = FADD(fpi,
                FMUL(FSQRT(FADD(FMUL(outu[k], outu[k]),
                                FMUL(outv[k], outv[k]))),
                     dp));
        }
        if (fpi > K_ZERO) {
            for (int k = 1; k <= ktop; k++) {
                float fp = FDIV(FSQRT(FADD(FMUL(outu[k], outu[k]),
                                           FMUL(outv[k], outv[k]))),
                                fpi);
                outt[k] = FADD(outt[k],
                               FDIV(FMUL(FMUL(fp, dts), K_G), K_CP));
            }
        }
    }
    CAPL(LEV_outt_ke, outt);
    CAPL(LEV_outu_f, outu);
    CAPL(LEV_outv_f, outv);
    SCAB[SCA_pre] = pre;
    ISCB[ISCA_ktop] = ktop;
    ISCB[ISCA_ierr] = ierr;

    // ---- the get_inversion_layers capture, for the shallow port ----------
    {
        float dtempdz[GF_KP], first_s[GF_KP], sec_s[GF_KP], sd_s[GF_KP];
        int k_inv[GF_KP];
        int clamped = 0;
        gfd_get_inversion_layers(ierr6_saved, p_cup, t_cup, z_cup, kbcon,
                                 kstabi, nz, ktf, dtempdz, k_inv, &clamped,
                                 first_s, sec_s, sd_s);
        CAPL(LEV_dtempdz, dtempdz);
        ISCB[ISCA_kinv1] = k_inv[1];
        ISCB[ISCA_kinv2] = k_inv[2];
        ISCB[ISCA_kinv3] = k_inv[3];
        ISCB[ISCA_kinv4] = k_inv[4];
        ISCB[ISCA_kinv5] = k_inv[5];
        ISCB[ISCA_kinv_clamped] = clamped;
    }
#undef CAPL

    *pre_out = pre;
    *ktop_out = ktop;
    *kbcon_out = kbcon;
    *k22_out = k22;
    *ierr_out = ierr;
}

extern "C" __global__ void gf_deep_stage(
    const float *__restrict__ lvin,   // (n, GF_NIN_LEV, nz)
    const float *__restrict__ scin,   // (n, GF_NIN_SCA)
    const int *__restrict__ iin,      // (n,) kpbl
    float *__restrict__ lev,          // (n, GF_NLEV, nz)
    float *__restrict__ sca,          // (n, GF_NSCA)
    int *__restrict__ isca,           // (n, GF_NISCA)
    int n, int nz)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    const float *inb = lvin + (size_t)col * GF_NIN_LEV * (size_t)nz;
    float zo[GF_KP], t[GF_KP], q[GF_KP], tn[GF_KP], qo[GF_KP], po[GF_KP];
    float us[GF_KP], vs[GF_KP], rho[GF_KP], omeg[GF_KP];
    for (int k = 0; k < GF_KP; k++) {
        zo[k] = K_ZERO; t[k] = K_ZERO; q[k] = K_ZERO; tn[k] = K_ZERO;
        qo[k] = K_ZERO; po[k] = K_ZERO; us[k] = K_ZERO; vs[k] = K_ZERO;
        rho[k] = K_ZERO; omeg[k] = K_ZERO;
    }
    for (int k = 1; k <= nz; k++) {
        zo[k] = inb[(size_t)IN_zo * nz + (k - 1)];
        t[k] = inb[(size_t)IN_t * nz + (k - 1)];
        q[k] = inb[(size_t)IN_q * nz + (k - 1)];
        tn[k] = inb[(size_t)IN_tn * nz + (k - 1)];
        qo[k] = inb[(size_t)IN_qo * nz + (k - 1)];
        po[k] = inb[(size_t)IN_po * nz + (k - 1)];
        us[k] = inb[(size_t)IN_us * nz + (k - 1)];
        vs[k] = inb[(size_t)IN_vs * nz + (k - 1)];
        rho[k] = inb[(size_t)IN_rho * nz + (k - 1)];
        omeg[k] = inb[(size_t)IN_omeg * nz + (k - 1)];
    }
    const float *sci = scin + (size_t)col * GF_NIN_SCA;
    float outt[GF_KP], outq[GF_KP], outqc[GF_KP], outu[GF_KP], outv[GF_KP];
    float cupclw[GF_KP];
    float pre;
    int ktop_o, kbcon_o, k22_o, ierr_o;
    gfd_deep_column(
        zo, t, q, tn, qo, po, us, vs, rho, omeg,
        sci[INS_z1], sci[INS_psur], sci[INS_hfx], sci[INS_qfx],
        sci[INS_xland], sci[INS_dx], sci[INS_ccn], sci[INS_dtime],
        sci[INS_xmbs], sci[INS_fzu_up], sci[INS_fzu_dn],
        iin[col], nz, /*ichoice=*/0,
        lev + (size_t)col * GF_NLEV * (size_t)nz,
        sca + (size_t)col * GF_NSCA,
        isca + (size_t)col * GF_NISCA,
        outt, outq, outqc, outu, outv, cupclw,
        &pre, &ktop_o, &kbcon_o, &k22_o, &ierr_o);
}

// ==========================================================================
// probes
// ==========================================================================
// Word dump of the scheme constant table, for the gate that re-derives
// every word from gpuwm.verify.gf_deep_ref.
extern "C" __global__ void gf_deep_const_dump(unsigned int *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < GF_NCONST) out[i] = GFC[i];
}

// The libm surface tgammaf stands on, plus the negative control: CUDA's
// builtin tgammaf is a DIFFERENT function from glibc's and slot 1 proves
// it on the live argument set.
extern "C" __global__ void gf_libm_unary_probe(const float *__restrict__ x,
                                               float *__restrict__ out,
                                               int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    out[7 * (size_t)i + 0] = gfk_tgamma(v);
    out[7 * (size_t)i + 1] = tgammaf(v);      // CUDA builtin: negative control
    out[7 * (size_t)i + 2] = gfk_lgamma_pos(v);
    out[7 * (size_t)i + 3] = gfk_expm1(v);
    out[7 * (size_t)i + 4] = gfk_exp2(v);
    out[7 * (size_t)i + 5] = gfk_exp(v);
    out[7 * (size_t)i + 6] = gfk_log(v);
}

extern "C" __global__ void gf_libm_pow_probe(const float *__restrict__ x,
                                             const float *__restrict__ y,
                                             float *__restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[2 * (size_t)i + 0] = gfk_pow(x[i], y[i]);
    out[2 * (size_t)i + 1] = powf(x[i], y[i]); // CUDA builtin: negative control
}

// fzu exactly as get_zu_zd_pdf_fim spells it, over (alpha, beta) pairs:
// gamma(alpha+beta) / (gamma(alpha) * gamma(beta)), one float32 rounding
// per operation.
extern "C" __global__ void gf_fzu_probe(const float *__restrict__ alpha,
                                        const float *__restrict__ beta,
                                        float *__restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = alpha[i], b = beta[i];
    out[i] = FDIV(gfk_tgamma(FADD(a, b)),
                  FMUL(gfk_tgamma(a), gfk_tgamma(b)));
}

// ==========================================================================
// the shallow cloud model: CUP_gf_sh, module_cu_gf_sh.F:58-936, one column.
// Shares seven procedures with the deep module above and everything else
// about it is different -- no downdraft, three closures averaged instead of
// a 16-member ensemble, no scale awareness (no dx anywhere), mbdt = .5,
// entr_rate = 9.e-5 flat, ktop from the inversion slot, cup_kbcon at
// iloop = 5, and the SH2 profile with beta = 2.5.
//
// OWNER RULING (no inherited WRF bugs), applied here.  WRF's
//     k22(i) = maxloc(HEO_CUP(i,2:kbmax(i)), 1)
// (module_cu_gf_sh.F:373) is a MAXLOC over an array SECTION: it returns the
// position WITHIN 2:kbmax and WRF uses it as an absolute level index
// without adding the section offset, so WRF's k22 sits one level BELOW the
// argmax of heo_cup wherever the argmax is above level 2 (case 13 of the
// committed fixture is the witness: 8 where the argmax is 9).  The SHIPPED
// default of this kernel is the CORRECTED indexing -- k22 IS the argmax --
// and the WRF-faithful off-by-one lives behind `k22_wrf_faithful`, which
// only the parity suites set.  The measured delta between the two modes on
// the committed fixture is recorded in
// tests/test_gf_shallow_cuda.py::test_the_corrected_k22_ledger_entry.
// ==========================================================================

// module_cu_gf_deep.F:3697-3823, name == 'shallow'.  ktop arrives already
// decided; the k22..kbcon ramp is built and survives ONLY on the ierr = 41
// path (the pdf zeroes zu on entry otherwise).
__device__ void gfd_rates_up_pdf_shallow(int *ktop_io, int *ierr_io,
    const float *p_cup, const float *entr_rate_2d, const float *z_cup,
    int kpbl, int k22, int *kbcon_io, int nz, int ktf, float fzu_override,
    float *zuo, GfdPdf *pdf, int *kfinal_out)
{
    for (int k = 0; k < GF_KP; k++) zuo[k] = K_ZERO;
    int kbcon = IMAX(*kbcon_io, 2);
    *kbcon_io = kbcon;
    pdf->tunning = K_ZERO; pdf->alpha = K_ZERO; pdf->beta = K_ZERO;
    pdf->fzu = K_ZERO; pdf->kb_adj = 0;
    *kfinal_out = 0;
    if (*ierr_io != 0) return;
    int start_level = k22;
    zuo[start_level] = K_P1;
    for (int k = start_level + 1; k <= kbcon; k++) {
        float dz = FSUB(z_cup[k], z_cup[k - 1]);
        float massent = FMUL(FMUL(dz, entr_rate_2d[k - 1]), zuo[k - 1]);
        float massdetr = FMUL(FMUL(dz, K_E1M9), zuo[k - 1]);
        zuo[k] = FSUB(FADD(zuo[k - 1], massent), massdetr);
    }
    if (*ktop_io <= kbcon + 2) {
        *ierr_io = 41;
        *ktop_io = 0;
        return;
    }
    int kfinalzu = *ktop_io;
    gfd_get_zu_zd_pdf(1, p_cup, k22, kfinalzu, kpbl, 0, K_P1, nz, ktf,
                      fzu_override, zuo, pdf);
    *kfinal_out = kfinalzu;
}

// capture layout for the shallow stage -- gf-shallow-levels.csv /
// gf-shallow-surface.csv order; tools/gf_wrf461_oracle/gf_field_lists.py
// carries the same lists.
enum {
    SHL_qes, SHL_he, SHL_hes, SHL_qeso, SHL_heo, SHL_heso,
    SHL_qes_cup, SHL_q_cup, SHL_he_cup, SHL_hes_cup, SHL_z_cup, SHL_p_cup,
    SHL_gamma_cup0, SHL_t_cup, SHL_qeso_cup, SHL_qo_cup, SHL_heo_cup,
    SHL_heso_cup, SHL_zo_cup, SHL_po_cup0, SHL_gammao_cup, SHL_tn_cup,
    SHL_dtempdz, SHL_entr2d_a, SHL_cd_a, SHL_zu_pdf, SHL_zuo_b,
    SHL_upme, SHL_upmd, SHL_cd_b, SHL_entr2d_b,
    SHL_hc, SHL_hco, SHL_dby, SHL_dbyo, SHL_dbyt, SHL_qco_a, SHL_qrco,
    SHL_pwo, SHL_cupclw, SHL_qco, SHL_cnvwt,
    SHL_dellah, SHL_dellaq, SHL_dellaqc, SHL_dellat,
    SHL_xhe, SHL_xq, SHL_xt, SHL_xqes, SHL_xhes,
    SHL_xqes_cup, SHL_xq_cup, SHL_xhe_cup, SHL_xhes_cup, SHL_gamma_cupx,
    SHL_xt_cup, SHL_po_cupx,
    SHL_xhc, SHL_xdby, SHL_xzu,
    SHL_zuo, SHL_outt, SHL_outq, SHL_outqc,
    GF_SH_NLEV
};

enum {
    SHS_buo_flux, SHS_zws, SHS_ztexec, SHS_zqexec, SHS_cap_max,
    SHS_entr_rate, SHS_hkb0, SHS_hkbo0, SHS_hkbo_1, SHS_hkb_2,
    SHS_sh_tun, SHS_sh_alpha, SHS_sh_beta, SHS_sh_fzu, SHS_qaver,
    SHS_aa0, SHS_aa1, SHS_xhkb, SHS_xaa0, SHS_xkshal,
    SHS_xff1, SHS_xff2, SHS_xff3, SHS_blqe, SHS_trash_kb, SHS_xmbmax,
    SHS_xmb, SHS_xmb_out, SHS_pre,
    GF_SH_NSCA
};

enum {
    SHI_xland1, SHI_kbmax, SHI_k22_0, SHI_k22_1, SHI_kbcon_1, SHI_ierr_1,
    SHI_kstabi, SHI_kinv1, SHI_kinv2, SHI_kinv3, SHI_kinv4, SHI_kinv5,
    SHI_kstabi_oob, SHI_start_level, SHI_ierr_231, SHI_ktop_0,
    SHI_ktop_pdf, SHI_kbcon_2, SHI_ierr_2, SHI_sh_kbadj, SHI_sh_kfinal,
    SHI_ktop_3, SHI_k22_3, SHI_ki_dbyt, SHI_ktop_4, SHI_ierr_4, SHI_ierr_5,
    SHI_ierr_6, SHI_k22, SHI_kbcon, SHI_ktop, SHI_ierr,
    GF_SH_NISCA
};

// input packing for gf_shallow_stage
enum {
    SIN_zo, SIN_t, SIN_q, SIN_tn, SIN_qo, SIN_po, SIN_dhdt, SIN_rho,
    GF_SH_NIN_LEV
};
enum {
    SINS_z1, SINS_psur, SINS_hfx, SINS_qfx, SINS_xland, SINS_dtime,
    SINS_fzu_sh,
    GF_SH_NIN_SCA
};

__device__ void gfd_shallow_column(
    const float *zo, const float *t, const float *q, const float *tn,
    const float *qo, const float *po, const float *dhdt, const float *rho,
    float z1, float psur, float hfx, float qfx, float xland, float dtime,
    int kpbl, int nz, int ichoice, int k22_wrf_faithful, float fzu_override,
    float *LEVB, float *SCAB, int *ISCB,
    float *outt, float *outq, float *outqc, float *cupclw,
    float *pre_out, float *xmb_out_p, int *k22_out, int *kbcon_out,
    int *ktop_out, int *ierr_out)
{
    const int ktf = nz;
    const int kte = nz;
    (void)kte;

#define SCAPL(FIDX, ARR) do { \
        for (int _k = 1; _k <= nz; _k++) \
            LEVB[(size_t)(FIDX) * (size_t)nz + (_k - 1)] = (ARR)[_k]; \
    } while (0)
    for (int i = 0; i < GF_SH_NSCA; i++) SCAB[i] = K_ZERO;
    for (int i = 0; i < GF_SH_NISCA; i++) ISCB[i] = 0;

    int ierr = 0, kbcon = 0, ktop = 0, k22 = 0;

    // ---- :241-256 ---------------------------------------------------------
    int start_level = 0;
    float flux_tun = K_FLUXTUNE;
    int ktopx = 0;
    int xland1 = (int)FADD(xland, K_E1M3);   // .001, not the deep arm's .0001
    if (xland > K_ONE_P5 || xland < K_HALF) xland1 = 0;
    float pre = K_ZERO;
    float xmb_out = K_ZERO;
    float cap_max_increment = K_TWENTYFIVE;
    float entr_rate = K_ENTR_SH;

    // ---- :265-277 ---------------------------------------------------------
    float upme[GF_KP], upmd[GF_KP], upmeu_s[GF_KP], upmdu_s[GF_KP];
    float z[GF_KP], xz[GF_KP], qrco[GF_KP], pwo[GF_KP], cd[GF_KP];
    float dellaqc[GF_KP], cnvwt[GF_KP];
    float zuo[GF_KP], zu[GF_KP], xzu[GF_KP];
    for (int k = 0; k < GF_KP; k++) {
        upme[k] = K_ZERO; upmd[k] = K_ZERO;
        upmeu_s[k] = K_ZERO; upmdu_s[k] = K_ZERO;
        z[k] = zo[k]; xz[k] = zo[k];
        qrco[k] = K_ZERO; pwo[k] = K_ZERO;
        cd[k] = (k == 0) ? K_ZERO : FMUL(K_ONE, entr_rate);
        dellaqc[k] = K_ZERO; cupclw[k] = K_ZERO; cnvwt[k] = K_ZERO;
        zuo[k] = K_ZERO; zu[k] = K_ZERO; xzu[k] = K_ZERO;
        outt[k] = K_ZERO; outq[k] = K_ZERO; outqc[k] = K_ZERO;
    }

    // ---- :287-298 ---------------------------------------------------------
    float cap_maxs = K_CAP_MAXS_SH;
    int kbmax = 1;
    float aa0 = K_ZERO, aa1 = K_ZERO;
    float cap_max = cap_maxs;
    float ztexec = K_ZERO, zqexec = K_ZERO, zws = K_ZERO;

    // ---- :299-319 : the convective-scale velocity -------------------------
    // The shallow module declares its own g/cp/xlv/r_v and they are
    // numerically EQUAL to the deep module's set (unlike the driver's GFS
    // set), so the K_ words serve both; SH_C0 = .001 shares C1's word and
    // c1_shal = 0.
    float buo_flux = FDIV(
        FADD(FDIV(hfx, K_CP), FDIV(FMUL(FMUL(K_P608, t[1]), qfx), K_XLV)),
        rho[1]);
    zws = GMAX(K_ZERO,
        FDIV(FMUL(FMUL(FMUL(FMUL(flux_tun, K_P41), buo_flux), zo[2]), K_G),
             t[1]));
    if (zws > K_TINY32) {
        zws = FMUL(K_ONE_P2, gfk_pow(zws, K_P3333));
        ztexec = GMAX(FDIV(FMUL(flux_tun, hfx),
                           FMUL(FMUL(rho[1], zws), K_CP)), K_ZERO);
        zqexec = GMAX(FDIV(FDIV(FMUL(flux_tun, qfx), K_XLV),
                           FMUL(rho[1], zws)), K_ZERO);
    }
    zws = GMAX(K_ZERO,
        FDIV(FMUL(FMUL(FMUL(FMUL(flux_tun, K_P41), buo_flux), zo[kpbl]),
                  K_G),
             t[kpbl]));
    zws = FMUL(K_ONE_P2, gfk_pow(zws, K_P3333));
    zws = FMUL(zws, rho[kpbl]);
    SCAB[SHS_buo_flux] = buo_flux;
    SCAB[SHS_zws] = zws;
    SCAB[SHS_ztexec] = ztexec;
    SCAB[SHS_zqexec] = zqexec;
    ISCB[SHI_xland1] = xland1;
    SCAB[SHS_entr_rate] = entr_rate;

    float zkbmax = K_ZKBMAX_SH;

    // ---- :328-349 : the two environments ----------------------------------
    float qes[GF_KP], he[GF_KP], hes[GF_KP];
    float qeso[GF_KP], heo[GF_KP], heso[GF_KP];
    gfd_cup_env(z, t, q, po, qes, he, hes, nz);
    gfd_cup_env(zo, tn, qo, po, qeso, heo, heso, nz);
    float qes_cup[GF_KP], q_cup[GF_KP], he_cup[GF_KP], hes_cup[GF_KP];
    float z_cup[GF_KP], p_cup[GF_KP], gamma_cup[GF_KP], t_cup[GF_KP];
    gfd_cup_env_clev(t, qes, q, he, hes, z, po, psur, z1, nz, qes_cup, q_cup,
                     he_cup, hes_cup, z_cup, p_cup, gamma_cup, t_cup);
    float qeso_cup[GF_KP], qo_cup[GF_KP], heo_cup[GF_KP], heso_cup[GF_KP];
    float zo_cup[GF_KP], po_cup[GF_KP], gammao_cup[GF_KP], tn_cup[GF_KP];
    gfd_cup_env_clev(tn, qeso, qo, heo, heso, zo, po, psur, z1, nz, qeso_cup,
                     qo_cup, heo_cup, heso_cup, zo_cup, po_cup, gammao_cup,
                     tn_cup);
    SCAPL(SHL_qes, qes); SCAPL(SHL_he, he); SCAPL(SHL_hes, hes);
    SCAPL(SHL_qeso, qeso); SCAPL(SHL_heo, heo); SCAPL(SHL_heso, heso);
    SCAPL(SHL_qes_cup, qes_cup); SCAPL(SHL_q_cup, q_cup);
    SCAPL(SHL_he_cup, he_cup); SCAPL(SHL_hes_cup, hes_cup);
    SCAPL(SHL_z_cup, z_cup); SCAPL(SHL_p_cup, p_cup);
    SCAPL(SHL_gamma_cup0, gamma_cup); SCAPL(SHL_t_cup, t_cup);
    SCAPL(SHL_qeso_cup, qeso_cup); SCAPL(SHL_qo_cup, qo_cup);
    SCAPL(SHL_heo_cup, heo_cup); SCAPL(SHL_heso_cup, heso_cup);
    SCAPL(SHL_zo_cup, zo_cup); SCAPL(SHL_po_cup0, po_cup);
    SCAPL(SHL_gammao_cup, gammao_cup); SCAPL(SHL_tn_cup, tn_cup);

    // ---- :350-363 : kbmax --------------------------------------------------
    if (ierr == 0) {
        for (int k = 1; k <= ktf; k++) {
            if (zo_cup[k] > FADD(zkbmax, z1)) {
                kbmax = k;
                break;
            }
        }
        kbmax = IMIN(kbmax, ktf / 2);
    }
    ISCB[SHI_kbmax] = kbmax;

    // ---- :370-383 : cap_max and k22 ---------------------------------------
    // The cap collapse is OUTSIDE the ierr guard.  The k22 MAXLOC is where
    // the owner ruling lands: shipped default is the CORRECTED absolute
    // argmax; k22_wrf_faithful reproduces WRF's missing section offset for
    // the parity suites.
    if (kpbl > 3) cap_max = po_cup[kpbl];
    if (ierr == 0) {
        int am = (kbmax >= 2) ? gfd_maxloc(heo_cup, 2, kbmax) : 0;
        if (kbmax >= 2)
            k22 = k22_wrf_faithful ? (am - 1) : am;
        else
            k22 = 0;
        k22 = IMAX(2, k22);
        if (k22 > kbmax) {
            ierr = 2;
            ktop = 0;
            k22 = 0;
            kbcon = 0;
        }
    }
    SCAB[SHS_cap_max] = cap_max;
    ISCB[SHI_k22_0] = k22;

    // ---- :387-393 ----------------------------------------------------------
    float hkb = K_ZERO, hkbo = K_ZERO;
    if (ierr == 0) {
        float x_add = FADD(FMUL(K_XLV, zqexec), FMUL(K_CP, ztexec));
        hkb = gfd_get_cloud_bc(he_cup, k22, x_add, 1);
        hkbo = gfd_get_cloud_bc(heo_cup, k22, x_add, 1);
    }
    SCAB[SHS_hkb0] = hkb;
    SCAB[SHS_hkbo0] = hkbo;

    // ---- :402-407 : cup_kbcon at iloop = 5 --------------------------------
    float hcot_s[GF_KP];
    gfd_cup_kbcon(cap_max_increment, 5, &k22, heo_cup, heso_cup, &hkbo,
                  &ierr, kbmax, po_cup, cap_max, ztexec, zqexec, z_cup,
                  entr_rate, heo, nz, hcot_s, &kbcon);
    ISCB[SHI_kbcon_1] = kbcon;
    ISCB[SHI_k22_1] = k22;
    SCAB[SHS_hkbo_1] = hkbo;
    ISCB[SHI_ierr_1] = ierr;

    // ---- :409-414 ----------------------------------------------------------
    int kstabi = gfd_cup_minimi(heso_cup, kbcon, kbmax, ierr);
    int k_inv[GF_KP];
    int kinv_clamped = 0;
    {
        float dtempdz[GF_KP], first_s[GF_KP], sec_s[GF_KP], sd_s[GF_KP];
        gfd_get_inversion_layers(ierr, p_cup, t_cup, z_cup, kbcon, kstabi,
                                 nz, ktf, dtempdz, k_inv, &kinv_clamped,
                                 first_s, sec_s, sd_s);
        SCAPL(SHL_dtempdz, dtempdz);
    }
    ISCB[SHI_kstabi] = kstabi;
    ISCB[SHI_kinv1] = k_inv[1];
    ISCB[SHI_kinv2] = k_inv[2];
    ISCB[SHI_kinv3] = k_inv[3];
    ISCB[SHI_kinv4] = k_inv[4];
    ISCB[SHI_kinv5] = k_inv[5];
    ISCB[SHI_kstabi_oob] = kinv_clamped;

    // ---- :417-449 : the entrainment profile and the first ktop ------------
    float entr_rate_2d[GF_KP];
    for (int k = 0; k < GF_KP; k++)
        entr_rate_2d[k] = (k == 0) ? K_ZERO : entr_rate;
    if (ierr == 0) {
        start_level = k22;
        float x_add = FADD(FMUL(K_XLV, zqexec), FMUL(K_CP, ztexec));
        hkb = gfd_get_cloud_bc(he_cup, k22, x_add, 1);
        if (kbcon > ktf - 4) ierr = 231;
        for (int k = 1; k <= ktf; k++) {
            float frh = FMUL(K_TWO, GMIN(FDIV(qo_cup[k], qeso_cup[k]),
                                         K_ONE));
            entr_rate_2d[k] = FMUL(entr_rate, FSUB(K_TWO_P3, frh));
            cd[k] = entr_rate_2d[k];
        }
        ktop = 1;
        if (k_inv[1] > 0
            && FSUB(po_cup[kbcon], po_cup[k_inv[1]]) < K_P200) {
            ktop = k_inv[1];
        } else {
            for (int k = kbcon + 1; k <= ktf; k++) {
                if (FSUB(po_cup[kbcon], po_cup[k]) > K_P200) {
                    ktop = k;
                    break;
                }
            }
        }
    }
    ISCB[SHI_start_level] = start_level;
    SCAB[SHS_hkb_2] = hkb;
    ISCB[SHI_ierr_231] = ierr;
    ISCB[SHI_ktop_0] = ktop;
    SCAPL(SHL_entr2d_a, entr_rate_2d);
    SCAPL(SHL_cd_a, cd);

    // ---- :451-452 : the normalised mass-flux profile ----------------------
    GfdPdf sh_pdf;
    int sh_kfinal = 0;
    gfd_rates_up_pdf_shallow(&ktop, &ierr, po_cup, entr_rate_2d, zo_cup,
                             kpbl, k22, &kbcon, nz, ktf, fzu_override, zuo,
                             &sh_pdf, &sh_kfinal);
    if (ierr == 0) ktopx = ktop;
    (void)ktopx;
    SCAPL(SHL_zu_pdf, zuo);
    ISCB[SHI_ktop_pdf] = ktop;
    ISCB[SHI_kbcon_2] = kbcon;
    ISCB[SHI_ierr_2] = ierr;
    SCAB[SHS_sh_tun] = sh_pdf.tunning;
    SCAB[SHS_sh_alpha] = sh_pdf.alpha;
    SCAB[SHS_sh_beta] = sh_pdf.beta;
    SCAB[SHS_sh_fzu] = sh_pdf.fzu;
    ISCB[SHI_sh_kbadj] = sh_pdf.kb_adj;
    ISCB[SHI_sh_kfinal] = sh_kfinal;

    // ---- :453-486 ----------------------------------------------------------
    if (ierr == 0) {
        if (k22 > 1) {
            for (int k = 1; k < k22; k++) {
                zuo[k] = K_ZERO;
                zu[k] = K_ZERO;
                xzu[k] = K_ZERO;
            }
        }
        for (int k = gfd_maxloc(zuo, 1, nz); k <= ktop; k++) {
            if (zuo[k] < K_E1M6) {
                ktop = k - 1;
                break;
            }
        }
        for (int k = k22; k <= ktop; k++) {
            xzu[k] = zuo[k];
            zu[k] = zuo[k];
        }
        for (int k = ktop + 1; k <= ktf; k++) {
            zuo[k] = K_ZERO;
            zu[k] = K_ZERO;
            xzu[k] = K_ZERO;
        }
        k22 = IMAX(2, k22);
    }
    SCAPL(SHL_zuo_b, zuo);
    ISCB[SHI_ktop_3] = ktop;
    ISCB[SHI_k22_3] = k22;

    // ---- :490-493 : lateral mass flux, without the momentum limb ----------
    gfd_get_lateral_massflux(ierr, ktop, zo_cup, zuo, cd, entr_rate_2d,
                             kbcon, k22, nz, ktf, K_ZERO, 0, upme, upmd,
                             upmeu_s, upmdu_s);

    // ---- :495-611 : the in-cloud updraft ----------------------------------
    float hc[GF_KP], qco[GF_KP], dby[GF_KP], hco[GF_KP];
    float dbyo[GF_KP], dbyt[GF_KP];
    for (int k = 0; k < GF_KP; k++) {
        hc[k] = K_ZERO; qco[k] = K_ZERO; dby[k] = K_ZERO; hco[k] = K_ZERO;
        dbyo[k] = K_ZERO; dbyt[k] = K_ZERO;
    }
    float qaver = K_ZERO;
    int ki = 0;
    if (ierr == 0) {
        for (int k = 1; k < start_level; k++) {
            hc[k] = he_cup[k];
            hco[k] = heo_cup[k];
        }
        hc[start_level] = hkb;
        hco[start_level] = hkbo;
    }
    if (ierr == 0) {
        for (int k = start_level + 1; k <= ktop; k++) {
            hc[k] = FDIV(
                FADD(FSUB(FMUL(hc[k - 1], zu[k - 1]),
                          FMUL(FMUL(K_HALF, upmd[k - 1]), hc[k - 1])),
                     FMUL(upme[k - 1], he[k - 1])),
                FADD(FSUB(zu[k - 1], FMUL(K_HALF, upmd[k - 1])),
                     upme[k - 1]));
            dby[k] = GMAX(K_ZERO, FSUB(hc[k], hes_cup[k]));
            hco[k] = FDIV(
                FADD(FSUB(FMUL(hco[k - 1], zuo[k - 1]),
                          FMUL(FMUL(K_HALF, upmd[k - 1]), hco[k - 1])),
                     FMUL(upme[k - 1], heo[k - 1])),
                FADD(FSUB(zuo[k - 1], FMUL(K_HALF, upmd[k - 1])),
                     upme[k - 1]));
            dbyo[k] = FSUB(hco[k], heso_cup[k]);
            float dz = FSUB(zo_cup[k + 1], zo_cup[k]);
            dbyt[k] = FADD(dbyt[k - 1], FMUL(dbyo[k], dz));
        }
        ki = gfd_maxloc(dbyt, 1, nz);
        if (ktop > ki + 1) {
            ktop = ki + 1;
            for (int k = ktop + 1; k <= ktf; k++) {
                zuo[k] = K_ZERO;
                zu[k] = K_ZERO;
                cd[k] = K_ZERO;
            }
            upmd[ktop] = zuo[ktop];
            for (int k = ktop; k <= ktf; k++) upme[k] = K_ZERO;
            for (int k = ktop + 1; k <= ktf; k++) upmd[k] = K_ZERO;
            for (int k = ktop + 1; k <= ktf; k++) entr_rate_2d[k] = K_ZERO;
        }
        if (ktop < kbcon + 1) {
            ierr = 5;
        } else if (ktop > ktf - 2) {
            ierr = 5;
        }
    }
    ISCB[SHI_ki_dbyt] = ki;

    if (ierr == 0) {
        qaver = gfd_get_cloud_bc(qo_cup, k22, K_ZERO, 0);
        qaver = FADD(qaver, zqexec);
        for (int k = 1; k < start_level; k++) qco[k] = qo_cup[k];
        qco[start_level] = qaver;
        for (int k = start_level + 1; k <= ktop; k++) {
            float trash = FADD(
                qeso_cup[k],
                FMUL(FMUL(K_ONE_OVER_XLV,
                          FDIV(gammao_cup[k], FADD(K_ONE, gammao_cup[k]))),
                     dbyo[k]));
            float trash2 = qco[k - 1];
            qco[k] = FDIV(
                FADD(FMUL(trash2,
                          FSUB(zuo[k - 1], FMUL(K_HALF, upmd[k - 1]))),
                     FMUL(upme[k - 1], qo[k - 1])),
                FADD(FSUB(zuo[k - 1], FMUL(K_HALF, upmd[k - 1])),
                     upme[k - 1]));
            if (qco[k] >= trash) {
                float dz = FSUB(z_cup[k], z_cup[k - 1]);
                // c0_shal + c1_shal: .001 + 0., spelled as WRF adds them
                qrco[k] = FDIV(FSUB(qco[k], trash),
                               FADD(K_ONE,
                                    FMUL(FADD(K_E1M3, K_ZERO), dz)));
                pwo[k] = FMUL(FMUL(FMUL(K_E1M3, dz), qrco[k]), zuo[k]);
                qco[k] = FADD(trash, qrco[k]);
            } else {
                qrco[k] = K_ZERO;
            }
            cupclw[k] = qrco[k];
        }
        SCAPL(SHL_qco_a, qco);
        for (int k = k22 + 1; k <= ktop; k++) {
            float dp = FMUL(K_HUNDRED, FSUB(po_cup[k], po_cup[k + 1]));
            cnvwt[k] = FDIV(FMUL(FMUL(zuo[k], cupclw[k]), K_G), dp);
            qco[k] = FSUB(qco[k], qrco[k]);
        }
        for (int k = ktop + 1; k <= ktf - 1; k++) {
            hc[k] = hes_cup[k];
            hco[k] = heso_cup[k];
            qco[k] = qeso_cup[k];
            qrco[k] = K_ZERO;
            dby[k] = K_ZERO;
            dbyo[k] = K_ZERO;
            zu[k] = K_ZERO;
            xzu[k] = K_ZERO;
            zuo[k] = K_ZERO;
        }
    } else {
        SCAPL(SHL_qco_a, qco);
    }
    ISCB[SHI_ktop_4] = ktop;
    ISCB[SHI_ierr_4] = ierr;
    SCAB[SHS_qaver] = qaver;
    SCAPL(SHL_hc, hc);
    SCAPL(SHL_hco, hco);
    SCAPL(SHL_dby, dby);
    SCAPL(SHL_dbyo, dbyo);
    SCAPL(SHL_dbyt, dbyt);
    SCAPL(SHL_qrco, qrco);
    SCAPL(SHL_pwo, pwo);
    SCAPL(SHL_cupclw, cupclw);
    SCAPL(SHL_qco, qco);
    SCAPL(SHL_cnvwt, cnvwt);
    SCAPL(SHL_upme, upme);
    SCAPL(SHL_upmd, upmd);
    SCAPL(SHL_cd_b, cd);
    SCAPL(SHL_entr2d_b, entr_rate_2d);

    // ---- :615-630 : the cloud work functions ------------------------------
    aa0 = gfd_cup_up_aa0(z, zu, dby, gamma_cup, t_cup, kbcon, ktop, ierr,
                         ktf);
    aa1 = gfd_cup_up_aa0(zo, zuo, dbyo, gammao_cup, tn_cup, kbcon, ktop,
                         ierr, ktf);
    if (ierr == 0 && aa1 <= K_ZERO) ierr = 17;
    SCAB[SHS_aa0] = aa0;
    SCAB[SHS_aa1] = aa1;
    ISCB[SHI_ierr_5] = ierr;

    // ---- :639-720 : the dellas --------------------------------------------
    float dellah[GF_KP], dellaq[GF_KP];
    for (int k = 0; k < GF_KP; k++) {
        dellah[k] = K_ZERO;
        dellaq[k] = K_ZERO;
    }
    if (ierr == 0) {
        for (int k = k22; k <= ktop; k++) {
            float detup = upmd[k];
            float dp = FMUL(K_HUNDRED, FSUB(po_cup[k], po_cup[k + 1]));
            dellah[k] = FDIV(
                -FMUL(FSUB(FMUL(zuo[k + 1],
                                FSUB(hco[k + 1], heo_cup[k + 1])),
                           FMUL(zuo[k], FSUB(hco[k], heo_cup[k]))),
                      K_G),
                dp);
            float dz = FSUB(zo_cup[k + 1], zo_cup[k]);
            if (k < ktop) {
                // c1_shal = 0., kept spelled: zuo*c1*qrco*dz/dp*g
                dellaqc[k] = FMUL(
                    FDIV(FMUL(FMUL(FMUL(zuo[k], K_ZERO), qrco[k]), dz), dp),
                    K_G);
            } else {
                dellaqc[k] = FDIV(FMUL(FMUL(detup, qrco[k]), K_G), dp);
            }
            float c_up = FADD(
                dellaqc[k],
                FDIV(FMUL(FSUB(FMUL(zuo[k + 1], qrco[k + 1]),
                               FMUL(zuo[k], qrco[k])),
                          K_G),
                     dp));
            dellaq[k] = FSUB(
                FSUB(FDIV(-FMUL(FSUB(FMUL(zuo[k + 1],
                                          FSUB(qco[k + 1], qo_cup[k + 1])),
                                     FMUL(zuo[k],
                                          FSUB(qco[k], qo_cup[k]))),
                                K_G),
                          dp),
                     c_up),
                FDIV(FMUL(FMUL(K_HALF, FADD(pwo[k], pwo[k + 1])), K_G),
                     dp));
        }
    }
    SCAPL(SHL_dellah, dellah);
    SCAPL(SHL_dellaq, dellaq);
    SCAPL(SHL_dellaqc, dellaqc);

    // ---- :725-746 : the mbdt-perturbed state ------------------------------
    float mbdt = K_HALF;
    float dellat[GF_KP], xhe[GF_KP], xq[GF_KP], xt[GF_KP];
    for (int k = 0; k < GF_KP; k++) {
        dellat[k] = K_ZERO; xhe[k] = K_ZERO; xq[k] = K_ZERO; xt[k] = K_ZERO;
    }
    if (ierr == 0) {
        for (int k = 1; k <= ktf; k++) {
            xhe[k] = FADD(FMUL(dellah[k], mbdt), heo[k]);
            xq[k] = GMAX(K_E1M16,
                         FADD(FMUL(FADD(dellaq[k], dellaqc[k]), mbdt),
                              qo[k]));
            dellat[k] = FMUL(FDIV(K_ONE, K_CP),
                             FSUB(dellah[k], FMUL(K_XLV, dellaq[k])));
            xt[k] = FADD(FMUL(FADD(FDIV(FMUL(-dellaqc[k], K_XLV), K_CP),
                                   dellat[k]),
                              mbdt),
                         tn[k]);
            xt[k] = GMAX(K_T190, xt[k]);
        }
        xhe[ktf] = heo[ktf];
        xq[ktf] = qo[ktf];
        xt[ktf] = tn[ktf];
    }
    SCAPL(SHL_dellat, dellat);
    SCAPL(SHL_xq, xq);
    SCAPL(SHL_xt, xt);

    // ---- :749-810 : the perturbed static control --------------------------
    float xqes[GF_KP], xhes[GF_KP];
    float xqes_cup[GF_KP], xq_cup[GF_KP], xhe_cup[GF_KP], xhes_cup[GF_KP];
    float xz_cup[GF_KP], xt_cup[GF_KP];
    for (int k = 0; k < GF_KP; k++) {
        xqes[k] = K_ZERO; xhes[k] = K_ZERO;
        xqes_cup[k] = K_ZERO; xq_cup[k] = K_ZERO;
        xhe_cup[k] = K_ZERO; xhes_cup[k] = K_ZERO;
        xz_cup[k] = K_ZERO; xt_cup[k] = K_ZERO;
    }
    if (ierr == 0) {
        gfd_cup_env(xz, xt, xq, po, xqes, xhe, xhes, nz);
        gfd_cup_env_clev(xt, xqes, xq, xhe, xhes, xz, po, psur, z1, nz,
                         xqes_cup, xq_cup, xhe_cup, xhes_cup, xz_cup,
                         po_cup, gamma_cup, xt_cup);
    } else {
        for (int k = 0; k < GF_KP; k++) {
            po_cup[k] = K_ZERO;
            gamma_cup[k] = K_ZERO;
        }
    }
    SCAPL(SHL_xhe, xhe);
    SCAPL(SHL_xqes, xqes);
    SCAPL(SHL_xhes, xhes);
    SCAPL(SHL_xqes_cup, xqes_cup);
    SCAPL(SHL_xq_cup, xq_cup);
    SCAPL(SHL_xhe_cup, xhe_cup);
    SCAPL(SHL_xhes_cup, xhes_cup);
    SCAPL(SHL_gamma_cupx, gamma_cup);
    SCAPL(SHL_xt_cup, xt_cup);
    SCAPL(SHL_po_cupx, po_cup);

    float xhc[GF_KP], xdby[GF_KP];
    for (int k = 0; k < GF_KP; k++) { xhc[k] = K_ZERO; xdby[k] = K_ZERO; }
    float xhkb = K_ZERO;
    if (ierr == 0) {
        float x_add = FADD(FMUL(K_XLV, zqexec), FMUL(K_CP, ztexec));
        xhkb = gfd_get_cloud_bc(xhe_cup, k22, x_add, 1);
        for (int k = 1; k < start_level; k++) xhc[k] = xhe_cup[k];
        xhc[start_level] = xhkb;
        for (int k = 1; k <= ktf; k++) xzu[k] = zuo[k];
        for (int k = start_level + 1; k <= ktop; k++) {
            xhc[k] = FDIV(
                FADD(FSUB(FMUL(xhc[k - 1], xzu[k - 1]),
                          FMUL(FMUL(K_HALF, upmd[k - 1]), xhc[k - 1])),
                     FMUL(upme[k - 1], xhe[k - 1])),
                FADD(FSUB(xzu[k - 1], FMUL(K_HALF, upmd[k - 1])),
                     upme[k - 1]));
            xdby[k] = FSUB(xhc[k], xhes_cup[k]);
        }
        for (int k = ktop + 1; k <= ktf; k++) {
            xhc[k] = xhes_cup[k];
            xdby[k] = K_ZERO;
            xzu[k] = K_ZERO;
        }
    }
    float xaa0 = gfd_cup_up_aa0(xz, xzu, xdby, gamma_cup, xt_cup, kbcon,
                                ktop, ierr, ktf);
    SCAB[SHS_xhkb] = xhkb;
    SCAB[SHS_xaa0] = xaa0;
    SCAPL(SHL_xhc, xhc);
    SCAPL(SHL_xdby, xdby);
    SCAPL(SHL_xzu, xzu);

    // ---- :817-874 : the shallow closure and the tendencies ----------------
    float xmb = K_ZERO;
    float xmbmax = K_ZERO;
    float xkshal = K_ZERO;
    float blqe = K_ZERO;
    float trash = K_ZERO;
    float xff0 = K_ZERO, xff1 = K_ZERO, xff2 = K_ZERO;
    if (ierr == 0) {
        xmbmax = K_ONE;
        xkshal = FDIV(FSUB(xaa0, aa1), mbdt);
        if (xkshal <= K_ZERO && xkshal > FMUL(K_NEG_P01, mbdt))
            xkshal = FMUL(K_NEG_P01, mbdt);
        if (xkshal > K_ZERO && xkshal < K_E1M2)
            xkshal = K_E1M2;
        xff0 = GMAX(K_ZERO,
                    FDIV(-FSUB(aa1, aa0), FMUL(xkshal, dtime)));
        xff1 = FMUL(K_P03, zws);
        for (int k = 1; k <= kpbl; k++) {
            blqe = FADD(blqe,
                FDIV(FMUL(FMUL(K_HUNDRED, dhdt[k]),
                          FSUB(po_cup[k], po_cup[k + 1])),
                     K_G));
        }
        trash = GMAX(FSUB(hc[kbcon], he_cup[kbcon]), K_TEN);
        xff2 = GMAX(K_ZERO, FDIV(blqe, trash));
        xff2 = GMIN(xmbmax, xff2);
        xmb = FDIV(FADD(FADD(xff0, xff1), xff2), K_THREE);
        xmb = GMIN(xmbmax, xmb);
        if (ichoice > 0) {
            float pick = (ichoice == 1) ? xff0
                         : (ichoice == 2) ? xff1 : xff2;
            xmb = GMIN(xmbmax, pick);
        }
        if (xmb <= K_ZERO) ierr = 21;
    }
    if (ierr != 0) {
        k22 = 0;
        kbcon = 0;
        ktop = 0;
        xmb = K_ZERO;
        for (int k = 0; k < GF_KP; k++) {
            outt[k] = K_ZERO;
            outq[k] = K_ZERO;
            outqc[k] = K_ZERO;
        }
    } else {
        xmb_out = xmb;
        pre = K_ZERO;
        for (int k = 2; k <= ktop; k++) {
            outt[k] = FMUL(dellat[k], xmb);
            outq[k] = FMUL(dellaq[k], xmb);
            outqc[k] = FMUL(dellaqc[k], xmb);
            pre = FADD(pre, FMUL(pwo[k], xmb));
        }
    }
    SCAB[SHS_xkshal] = xkshal;
    SCAB[SHS_xff1] = xff0;
    SCAB[SHS_xff2] = xff1;
    SCAB[SHS_xff3] = xff2;
    SCAB[SHS_blqe] = blqe;
    SCAB[SHS_trash_kb] = trash;
    SCAB[SHS_xmbmax] = xmbmax;
    SCAB[SHS_xmb] = xmb;
    ISCB[SHI_ierr_6] = ierr;
    ISCB[SHI_k22] = k22;
    ISCB[SHI_kbcon] = kbcon;
    ISCB[SHI_ktop] = ktop;
    ISCB[SHI_ierr] = ierr;
    SCAB[SHS_xmb_out] = xmb_out;
    SCAB[SHS_pre] = pre;
    SCAPL(SHL_zuo, zuo);
    SCAPL(SHL_outt, outt);
    SCAPL(SHL_outq, outq);
    SCAPL(SHL_outqc, outqc);
#undef SCAPL

    *pre_out = pre;
    *xmb_out_p = xmb_out;
    *k22_out = k22;
    *kbcon_out = kbcon;
    *ktop_out = ktop;
    *ierr_out = ierr;
}

extern "C" __global__ void gf_shallow_stage(
    const float *__restrict__ lvin,   // (n, GF_SH_NIN_LEV, nz)
    const float *__restrict__ scin,   // (n, GF_SH_NIN_SCA)
    const int *__restrict__ iin,      // (n,) kpbl
    float *__restrict__ lev,          // (n, GF_SH_NLEV, nz)
    float *__restrict__ sca,          // (n, GF_SH_NSCA)
    int *__restrict__ isca,           // (n, GF_SH_NISCA)
    int k22_wrf_faithful, int n, int nz)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    const float *inb = lvin + (size_t)col * GF_SH_NIN_LEV * (size_t)nz;
    float zo[GF_KP], t[GF_KP], q[GF_KP], tn[GF_KP], qo[GF_KP], po[GF_KP];
    float dhdt[GF_KP], rho[GF_KP];
    for (int k = 0; k < GF_KP; k++) {
        zo[k] = K_ZERO; t[k] = K_ZERO; q[k] = K_ZERO; tn[k] = K_ZERO;
        qo[k] = K_ZERO; po[k] = K_ZERO; dhdt[k] = K_ZERO; rho[k] = K_ZERO;
    }
    for (int k = 1; k <= nz; k++) {
        zo[k] = inb[(size_t)SIN_zo * nz + (k - 1)];
        t[k] = inb[(size_t)SIN_t * nz + (k - 1)];
        q[k] = inb[(size_t)SIN_q * nz + (k - 1)];
        tn[k] = inb[(size_t)SIN_tn * nz + (k - 1)];
        qo[k] = inb[(size_t)SIN_qo * nz + (k - 1)];
        po[k] = inb[(size_t)SIN_po * nz + (k - 1)];
        dhdt[k] = inb[(size_t)SIN_dhdt * nz + (k - 1)];
        rho[k] = inb[(size_t)SIN_rho * nz + (k - 1)];
    }
    const float *sci = scin + (size_t)col * GF_SH_NIN_SCA;
    float outt[GF_KP], outq[GF_KP], outqc[GF_KP], cupclw[GF_KP];
    float pre, xmbs;
    int k22_o, kbcon_o, ktop_o, ierr_o;
    gfd_shallow_column(
        zo, t, q, tn, qo, po, dhdt, rho,
        sci[SINS_z1], sci[SINS_psur], sci[SINS_hfx], sci[SINS_qfx],
        sci[SINS_xland], sci[SINS_dtime],
        iin[col], nz, /*ichoice=*/0, k22_wrf_faithful, sci[SINS_fzu_sh],
        lev + (size_t)col * GF_SH_NLEV * (size_t)nz,
        sca + (size_t)col * GF_SH_NSCA,
        isca + (size_t)col * GF_SH_NISCA,
        outt, outq, outqc, cupclw,
        &pre, &xmbs, &k22_o, &kbcon_o, &ktop_o, &ierr_o);
}

// ==========================================================================
// GFDRV's own halves: the mixed-precision column preparation
// (module_cu_gf_wrfdrv.F:383-492) and the output algebra (:713-840), plus
// neg_check between them.  The driver's g/cp/xlv come from
// module_gfs_physcons: real(8) parameters initialised from default-real
// literals, measured (gf_ref.py docstring table) to behave as the HONEST
// doubles -- so omeg, dhdt and mconv evaluate in double and round once.
// mconv itself is prepared by GFDRV and then IGNORED: cup_gf zeroes it and
// rebuilds it on the cloud grid, so this kernel does not compute it.
// ==========================================================================

// module_cu_gf_deep.F:3038-3139.  Only the heating-rate cap runs; the
// negative-q rescale below the early RETURN is unreachable.
__device__ float gfd_neg_check(int is_shallow, float dt, float *outq,
    float *outt, float *outu, float *outv, float *outqc, float pret,
    int ktf)
{
    (void)dt;
    float thresh = K_THRESH_DEEP;
    float names = K_ONE;
    if (is_shallow) {
        thresh = K_T148_01;
        names = K_TWO;
    }
    float qmemf = K_ONE;
    for (int k = 1; k <= ktf; k++) {
        float qmem = FMUL(outt[k], K_SECONDS_DAY);
        if (qmem > thresh)
            qmemf = GMIN(qmemf, FDIV(thresh, qmem));
        if (qmem < FMUL(FMUL(K_NEG_HALF, thresh), names))
            qmemf = GMIN(qmemf,
                         FDIV(FMUL(FMUL(K_NEG_HALF, names), thresh), qmem));
    }
    for (int k = 1; k <= ktf; k++) {
        outq[k] = FMUL(outq[k], qmemf);
        outt[k] = FMUL(outt[k], qmemf);
        outu[k] = FMUL(outu[k], qmemf);
        outv[k] = FMUL(outv[k], qmemf);
        outqc[k] = FMUL(outqc[k], qmemf);
    }
    return FMUL(pret, qmemf);
}

// input/output packing for gf_gfdrv_stage
enum {
    DIN_u, DIN_v, DIN_w, DIN_t, DIN_qv, DIN_p, DIN_pi, DIN_rho, DIN_dz8w,
    DIN_p8w, DIN_rthften, DIN_rqvften, DIN_rthraten, DIN_rthblten,
    DIN_rqvblten,
    GF_DRV_NIN_LEV
};
enum {
    DINS_ht, DINS_hfx, DINS_qfx, DINS_xland, DINS_dt, DINS_dx,
    GF_DRV_NIN_SCA
};
// iin is (n, 3): kpbl, ishallow, ichoice
enum {
    DL_rthcuten, DL_rqvcuten, DL_rqccuten, DL_rqicuten, DL_dudt, DL_dvdt,
    DL_gdc, DL_gdc2,
    DL_outt, DL_outq, DL_outqc, DL_outu, DL_outv,
    DL_outts, DL_outqs, DL_outqcs,
    GF_DRV_NLEV
};
enum {
    DS_raincv, DS_pratec, DS_htop, DS_hbot, DS_xmb_shallow, DS_pret,
    DS_prets, DS_cuten, DS_cutens,
    GF_DRV_NSCA
};
enum {
    DI_ktop_deep, DI_k22_shallow, DI_kbcon_shallow, DI_ktop_shallow,
    DI_kbcon, DI_ktop,
    GF_DRV_NISCA
};

extern "C" __global__ void gf_gfdrv_stage(
    const float *__restrict__ lvin,   // (n, GF_DRV_NIN_LEV, nz)
    const float *__restrict__ scin,   // (n, GF_DRV_NIN_SCA)
    const int *__restrict__ iin,      // (n, 3)
    float *__restrict__ lev,          // (n, GF_DRV_NLEV, nz)
    float *__restrict__ sca,          // (n, GF_DRV_NSCA)
    int *__restrict__ isca,           // (n, GF_DRV_NISCA)
    int k22_wrf_faithful, int n, int nz)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    const float *inb = lvin + (size_t)col * GF_DRV_NIN_LEV * (size_t)nz;
    const float *sci = scin + (size_t)col * GF_DRV_NIN_SCA;
    int kpbl = iin[3 * (size_t)col + 0];
    int ishallow = iin[3 * (size_t)col + 1];
    int ichoice = iin[3 * (size_t)col + 2];
    float ht = sci[DINS_ht];
    float hfx = sci[DINS_hfx];
    float qfx = sci[DINS_qfx];
    float xland = sci[DINS_xland];
    float dt = sci[DINS_dt];
    float dx = sci[DINS_dx];

    float u[GF_KP], v[GF_KP], w[GF_KP], t[GF_KP], qv[GF_KP], p[GF_KP];
    float pi_a[GF_KP], rho[GF_KP], dz8w[GF_KP], p8w[GF_KP];
    float rthften[GF_KP], rqvften[GF_KP], rthraten[GF_KP];
    float rthblten[GF_KP], rqvblten[GF_KP];
    for (int k = 0; k < GF_KP; k++) {
        u[k] = K_ZERO; v[k] = K_ZERO; w[k] = K_ZERO; t[k] = K_ZERO;
        qv[k] = K_ZERO; p[k] = K_ZERO; pi_a[k] = K_ZERO; rho[k] = K_ZERO;
        dz8w[k] = K_ZERO; p8w[k] = K_ZERO; rthften[k] = K_ZERO;
        rqvften[k] = K_ZERO; rthraten[k] = K_ZERO; rthblten[k] = K_ZERO;
        rqvblten[k] = K_ZERO;
    }
    for (int k = 1; k <= nz; k++) {
        u[k] = inb[(size_t)DIN_u * nz + (k - 1)];
        v[k] = inb[(size_t)DIN_v * nz + (k - 1)];
        w[k] = inb[(size_t)DIN_w * nz + (k - 1)];
        t[k] = inb[(size_t)DIN_t * nz + (k - 1)];
        qv[k] = inb[(size_t)DIN_qv * nz + (k - 1)];
        p[k] = inb[(size_t)DIN_p * nz + (k - 1)];
        pi_a[k] = inb[(size_t)DIN_pi * nz + (k - 1)];
        rho[k] = inb[(size_t)DIN_rho * nz + (k - 1)];
        dz8w[k] = inb[(size_t)DIN_dz8w * nz + (k - 1)];
        p8w[k] = inb[(size_t)DIN_p8w * nz + (k - 1)];
        rthften[k] = inb[(size_t)DIN_rthften * nz + (k - 1)];
        rqvften[k] = inb[(size_t)DIN_rqvften * nz + (k - 1)];
        rthraten[k] = inb[(size_t)DIN_rthraten * nz + (k - 1)];
        rthblten[k] = inb[(size_t)DIN_rthblten * nz + (k - 1)];
        rqvblten[k] = inb[(size_t)DIN_rqvblten * nz + (k - 1)];
    }

    // ---- the column preparation, module_cu_gf_wrfdrv.F:383-492 ------------
    float psur = FMUL(p8w[1], K_MB);
    float ter11 = GMAX(K_ZERO, ht);
    float dtf = dt;
    float zo[GF_KP], po[GF_KP], q2d[GF_KP], tn_f[GF_KP], qo_f[GF_KP];
    float tshall[GF_KP], qshall[GF_KP], dhdt[GF_KP], omeg[GF_KP];
    for (int k = 0; k < GF_KP; k++) {
        zo[k] = K_ZERO; po[k] = K_ZERO; q2d[k] = K_ZERO; tn_f[k] = K_ZERO;
        qo_f[k] = K_ZERO; tshall[k] = K_ZERO; qshall[k] = K_ZERO;
        dhdt[k] = K_ZERO; omeg[k] = K_ZERO;
    }
    zo[1] = FADD(ter11, FMUL(K_HALF, dz8w[1]));
    for (int k = 2; k <= nz; k++)
        zo[k] = FADD(zo[k - 1], FMUL(K_HALF, FADD(dz8w[k - 1], dz8w[k])));
    for (int k = 1; k <= nz; k++) {
        po[k] = FMUL(p[k], K_MB);
        // IF(Q2d.LT.1.E-08) Q2d = 1.E-08 -- a floor, not a max(): NaN
        // passes through the .LT. as false and survives.
        q2d[k] = (qv[k] < K_QMIN) ? K_QMIN : qv[k];
        tn_f[k] = FADD(t[k],
            FMUL(FMUL(FADD(FADD(rthften[k], rthraten[k]), rthblten[k]),
                      pi_a[k]),
                 dtf));
        qo_f[k] = FADD(q2d[k], FMUL(FADD(rqvften[k], rqvblten[k]), dtf));
        tshall[k] = FADD(t[k], FMUL(FMUL(rthblten[k], pi_a[k]), dtf));
        qshall[k] = FADD(q2d[k], FMUL(rqvblten[k], dtf));
        // dhdt and omeg: real(8) constants, whole right-hand side in
        // double, one rounding on the store (:416, :471).
        dhdt[k] = gfk_d2f_rn(
            DADD(DMUL(DMUL(GFS_CP_D, (double)rthblten[k]),
                      (double)pi_a[k]),
                 DMUL(GFS_XLV_D, (double)rqvblten[k])));
        omeg[k] = gfk_d2f_rn(
            DMUL(DMUL(-GFS_G_D, (double)rho[k]), (double)w[k]));
        if (tn_f[k] < K_TMIN) tn_f[k] = t[k];
        if (qo_f[k] < K_QMIN) qo_f[k] = K_QMIN;
    }
    float ccn = K_CCN150;

    // ---- the shallow arm (:504-530) ---------------------------------------
    float outts[GF_KP], outqs[GF_KP], outqcs[GF_KP];
    float outus[GF_KP], outvs[GF_KP], cupclws[GF_KP];
    for (int k = 0; k < GF_KP; k++) {
        outts[k] = K_ZERO; outqs[k] = K_ZERO; outqcs[k] = K_ZERO;
        outus[k] = K_ZERO; outvs[k] = K_ZERO; cupclws[k] = K_ZERO;
    }
    float prets = K_ZERO, xmbs = K_ZERO;
    int k22s = 0, kbcons = 0, ktops = 0, ierrs = 0;
    if (ishallow == 1) {
        float sh_lev[GF_SH_NLEV * GF_KMAX];
        float sh_sca[GF_SH_NSCA];
        int sh_isc[GF_SH_NISCA];
        gfd_shallow_column(
            zo, t, q2d, tshall, qshall, po, dhdt, rho,
            ter11, psur, hfx, qfx, xland, dt,
            kpbl, nz, /*ichoice=*/0, k22_wrf_faithful, K_ZERO,
            sh_lev, sh_sca, sh_isc,
            outts, outqs, outqcs, cupclws,
            &prets, &xmbs, &k22s, &kbcons, &ktops, &ierrs);
        prets = gfd_neg_check(1, dt, outqs, outts, outus, outvs, outqcs,
                              prets, nz);
    }
    (void)ierrs;

    // ---- the deep arm (:626-711) ------------------------------------------
    float outt[GF_KP], outq[GF_KP], outqc[GF_KP], outu[GF_KP], outv[GF_KP];
    float cupclw[GF_KP];
    float pret;
    int ktop_d, kbcon_d, k22_d, ierr_d;
    {
        float dp_lev[GF_NLEV * GF_KMAX];
        float dp_sca[GF_NSCA];
        int dp_isc[GF_NISCA];
        gfd_deep_column(
            zo, t, q2d, tn_f, qo_f, po, u, v, rho, omeg,
            ter11, psur, hfx, qfx, xland, dx, ccn, dt, xmbs,
            K_ZERO, K_ZERO,
            kpbl, nz, ichoice,
            dp_lev, dp_sca, dp_isc,
            outt, outq, outqc, outu, outv, cupclw,
            &pret, &ktop_d, &kbcon_d, &k22_d, &ierr_d);
    }
    (void)k22_d;
    (void)ierr_d;
    pret = gfd_neg_check(0, dt, outq, outt, outu, outv, outqc, pret, nz);

    // ---- capture the post-neg_check seam ----------------------------------
    float *LEVB = lev + (size_t)col * GF_DRV_NLEV * (size_t)nz;
    float *SCAB = sca + (size_t)col * GF_DRV_NSCA;
    int *ISCB = isca + (size_t)col * GF_DRV_NISCA;
    for (int k = 1; k <= nz; k++) {
        LEVB[(size_t)DL_outt * nz + (k - 1)] = outt[k];
        LEVB[(size_t)DL_outq * nz + (k - 1)] = outq[k];
        LEVB[(size_t)DL_outqc * nz + (k - 1)] = outqc[k];
        LEVB[(size_t)DL_outu * nz + (k - 1)] = outu[k];
        LEVB[(size_t)DL_outv * nz + (k - 1)] = outv[k];
        LEVB[(size_t)DL_outts * nz + (k - 1)] = outts[k];
        LEVB[(size_t)DL_outqs * nz + (k - 1)] = outqs[k];
        LEVB[(size_t)DL_outqcs * nz + (k - 1)] = outqcs[k];
    }

    // ---- the output algebra (:713-840) ------------------------------------
    int ktop = ktop_d;
    int kbcon = kbcon_d;
    int ktop_deep = ktop;
    // The mid-level arm is a parameter 0: its tendencies are identically
    // zero and cutenm never leaves zero.  The terms are KEPT because
    // `x + 0.0` is not the identity on a negative zero, and outts can be
    // one.
    float cutenm = K_ZERO;
    float pretm = K_ZERO;
    float zl = K_ZERO;

    // :328-331, :525 -- cutens, decided from xmbs BEFORE neg_check ran; the
    // call order above already honoured that (xmbs was read pre-neg_check).
    float cutens = K_ONE;
    if (ishallow == 0) cutens = K_ZERO;
    if (ishallow == 1 && xmbs <= K_ZERO) cutens = K_ZERO;

    // :724-742
    float cuten;
    if (pret > K_ZERO) {
        cuten = K_ONE;
    } else {
        cuten = K_ZERO;
        kbcon = 0;
        ktop = 0;
    }

    // :743-754
    for (int k = 1; k <= nz; k++) {
        float rth = FDIV(
            FADD(FADD(FMUL(cutens, outts[k]), FMUL(cutenm, zl)),
                 FMUL(cuten, outt[k])),
            pi_a[k]);
        float rqv = FADD(FADD(FMUL(cuten, outq[k]),
                              FMUL(cutens, outqs[k])),
                         FMUL(cutenm, zl));
        float du = FADD(FMUL(zl, cutenm), FMUL(outu[k], cuten));
        float dv = FADD(FMUL(zl, cutenm), FMUL(outv[k], cuten));
        LEVB[(size_t)DL_rthcuten * nz + (k - 1)] = rth;
        LEVB[(size_t)DL_rqvcuten * nz + (k - 1)] = rqv;
        LEVB[(size_t)DL_dudt * nz + (k - 1)] = du;
        LEVB[(size_t)DL_dvdt * nz + (k - 1)] = dv;
    }

    // :799-808.  HBOT starts at the TOP index and HTOP at the bottom one.
    float hbot = (float)nz;
    float htop = K_ONE;
    float pratec = K_ZERO, raincv = K_ZERO;
    if (pret > K_ZERO || pretm > K_ZERO || prets > K_ZERO) {
        pratec = FADD(FADD(FMUL(cuten, pret), FMUL(cutenm, pretm)),
                      FMUL(cutens, prets));
        raincv = FMUL(pratec, dt);
        if ((float)ktop > htop) htop = FADD((float)ktop, K_E1M3);
        if ((float)kbcon < hbot) hbot = FADD((float)kbcon, K_E1M3);
    }

    // :810-840.  RQCCUTEN and GDC/GDC2 are written twice in WRF; only the
    // 258 K split's second write survives, and the split is on t2d.
    for (int k = 1; k <= nz; k++) {
        float qc = FADD(FADD(zl, outqcs[k]), FMUL(outqc[k], cuten));
        float cw = FADD(FADD(zl, cupclws[k]), FMUL(cupclw[k], cuten));
        float rqi, rqc, g1, g2;
        if (t[k] < K_TCRIT) {
            rqi = qc; rqc = K_ZERO; g2 = cw; g1 = K_ZERO;
        } else {
            rqi = K_ZERO; rqc = qc; g1 = cw; g2 = K_ZERO;
        }
        LEVB[(size_t)DL_rqccuten * nz + (k - 1)] = rqc;
        LEVB[(size_t)DL_rqicuten * nz + (k - 1)] = rqi;
        LEVB[(size_t)DL_gdc * nz + (k - 1)] = g1;
        LEVB[(size_t)DL_gdc2 * nz + (k - 1)] = g2;
    }

    SCAB[DS_raincv] = raincv;
    SCAB[DS_pratec] = pratec;
    SCAB[DS_htop] = htop;
    SCAB[DS_hbot] = hbot;
    SCAB[DS_xmb_shallow] = (ishallow == 1) ? xmbs : K_ZERO;
    SCAB[DS_pret] = pret;
    SCAB[DS_prets] = prets;
    SCAB[DS_cuten] = cuten;
    SCAB[DS_cutens] = cutens;
    ISCB[DI_ktop_deep] = ktop_deep;
    ISCB[DI_k22_shallow] = (ishallow == 1) ? k22s : 0;
    ISCB[DI_kbcon_shallow] = (ishallow == 1) ? kbcons : 0;
    ISCB[DI_ktop_shallow] = (ishallow == 1) ? ktops : 0;
    ISCB[DI_kbcon] = kbcon;
    ISCB[DI_ktop] = ktop;
}
