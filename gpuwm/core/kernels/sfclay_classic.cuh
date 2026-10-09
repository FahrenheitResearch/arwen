// gpuwm/core/kernels/sfclay_classic.cuh
//
// Classic MM5 surface layer, sf_sfclay_physics = 91: WRF v4.6.1
// phys/module_sf_sfclay.F, SFCLAY1D and SFCLAYINIT, one CUDA thread per
// surface column.  Prepended to sfclay.cu (gpuwm/core/kernels/__init__.py
// _EXTRA_HEADERS) after glibc_flt32.cuh, whose gfk_log / gfk_exp / gfk_pow
// this file calls for every ALOG, EXP and REAL**REAL.
//
// THE STANDARD.  Graded word for word against the byte-unmodified Fortran by
// tools/sfclay_classic_wrf461_oracle and tests/test_sfclay_classic_wrf461_parity.py.
// So every statement keeps the Fortran's own operand order (Fortran
// evaluates a*b*c/d as ((a*b)*c)/d, and so does C), the module "sfclay"
// compiles without FMA contraction (_NO_FMAD_MODULES), and every
// transcendental is WOOF's own float32 routine rather than CUDA's builtin,
// which is a different function (up to 2 ULP on logf/expf/powf/atanf).
//
// Why this is its own function and not more `option == 91` arms in
// sfclay_column: the two MM5 options are two WRF modules with two
// spellings, and the shared body had drifted to the revised scheme's
// spelling in seven places the oracle found (see
// tools/sfclay_classic_wrf461_oracle/README.md for the list).
//
// One deliberate difference from WRF, documented and not copied:
//   module_sf_sfclay.F:767  PSIQ10=GZ10OZ0(I)-PSIH(I)+GZ0OZQ   (ISFTCFLX=2)
// pairs the 10 m log contrast with the LOWEST-LEVEL stability term.  Every
// other 10 m quantity in the file (PSIM10, PSIH10, and PSIQ10 in the
// default and ISFTCFLX=1 paths, :717 and :749) uses PSIH10, as do the 2 m
// lines of this same block (:765-766, PSIH2).  It reaches only CK, the
// 10 m enthalpy exchange coefficient, on water columns with isftcflx=2.
// WOOF uses PSIH10 there.
//
// Two ranges where WRF is undefined and WOOF is not (unchanged from the
// shared kernel): ZNT <= 0 or ZNT so small that ZA/ZNT overflows (WRF:
// ALOG(Inf) and U10 = Inf/Inf), and NaN ZNT (kept NaN).  See
// sf_log_zratio in sfclay.cu; sfc_log_zratio below is the same rescue
// around WOOF's own logf.

// ---------------------------------------------------------------------------
// WOOF's own float32 atanf.  The same routine and the same nineteen words as
// mynn_dmp_sibling.cu's mynn_glibc_atanf (tests/test_sfclay_classic_wrf461_parity.py
// holds the words equal and sweeps the two functions against each other).
// A correctly rounded atanf is a different function on 114 of SFCLAYINIT's
// 1001 table arguments, so the table needs exactly this one.  Its names carry
// the CLASSIC prefix because this header is compiled into the same object as
// sfclay.cu, which may declare an atanf of its own for option 1.
// ---------------------------------------------------------------------------
// Integer zero predicates are shared by the two MM5 entry points. Their
// bits preserve positive QSFC and the sign of negative previous MOL.
__device__ __forceinline__ bool surface_le_zero(float x) {
    unsigned int b = __float_as_uint(x), m = b & 0x7fffffffu;
    return m <= 0x7f800000u && (m == 0u || (b & 0x80000000u));
}
__device__ __forceinline__ bool surface_lt_zero(float x) {
    unsigned int b = __float_as_uint(x), m = b & 0x7fffffffu;
    return (b & 0x80000000u) && m != 0u && m <= 0x7f800000u;
}
__device__ const unsigned int SFC_CLASSIC_ATANF_TAB[19] = {
    0x3EED6338u, 0x3F490FDAu, 0x3F7B985Eu, 0x3FC90FDAu,
    0x31AC3769u, 0x33222168u, 0x33140FB4u, 0x33A22168u,
    0x3EAAAAABu, 0xBE4CCCCDu, 0x3E124925u, 0xBDE38E38u,
    0x3DBA2E6Eu, 0xBD9D8795u, 0x3D886B35u, 0xBD6EF16Bu,
    0x3D4BDA59u, 0xBD15A221u, 0x3C8569D7u,
};

__device__ float sfc_classic_atanf(float x)
{
    unsigned int hx = __float_as_uint(x);
    unsigned int ix = hx & 0x7FFFFFFFu;
    int signed_hx = (int)hx;
    int id;
#define SFCC_ATAN_HI(i) __uint_as_float(SFC_CLASSIC_ATANF_TAB[(i)])
#define SFCC_ATAN_LO(i) __uint_as_float(SFC_CLASSIC_ATANF_TAB[4 + (i)])
#define SFCC_ATAN_T(i)  __uint_as_float(SFC_CLASSIC_ATANF_TAB[8 + (i)])
    if (ix >= 0x4C000000u) {                     // |x| >= 2**25
        if (ix > 0x7F800000u) return __fadd_rn(x, x);
        if (signed_hx > 0) return __fadd_rn(SFCC_ATAN_HI(3), SFCC_ATAN_LO(3));
        return __fsub_rn(-SFCC_ATAN_HI(3), SFCC_ATAN_LO(3));
    }
    if (ix < 0x3EE00000u) {                      // |x| < 0.4375
        if (ix < 0x31000000u) return x;          // |x| < 2**-29
        id = -1;
    } else {
        x = __uint_as_float(ix);
        if (ix < 0x3F980000u) {                  // |x| < 1.1875
            if (ix < 0x3F300000u) {              // 7/16 <= |x| < 11/16
                id = 0;
                x = __fdiv_rn(__fsub_rn(__fmul_rn(2.0f, x), 1.0f),
                              __fadd_rn(2.0f, x));
            } else {                             // 11/16 <= |x| < 19/16
                id = 1;
                x = __fdiv_rn(__fsub_rn(x, 1.0f), __fadd_rn(x, 1.0f));
            }
        } else if (ix < 0x401C0000u) {           // |x| < 2.4375
            id = 2;
            x = __fdiv_rn(__fsub_rn(x, 1.5f),
                          __fadd_rn(1.0f, __fmul_rn(1.5f, x)));
        } else {                                 // 2.4375 <= |x| < 2**25
            id = 3;
            x = __fdiv_rn(-1.0f, x);
        }
    }
    float z = __fmul_rn(x, x);
    float w = __fmul_rn(z, z);
    float s1 = __fmul_rn(w, SFCC_ATAN_T(10));
    s1 = __fmul_rn(w, __fadd_rn(SFCC_ATAN_T(8), s1));
    s1 = __fmul_rn(w, __fadd_rn(SFCC_ATAN_T(6), s1));
    s1 = __fmul_rn(w, __fadd_rn(SFCC_ATAN_T(4), s1));
    s1 = __fmul_rn(w, __fadd_rn(SFCC_ATAN_T(2), s1));
    s1 = __fmul_rn(z, __fadd_rn(SFCC_ATAN_T(0), s1));
    float s2 = __fmul_rn(w, SFCC_ATAN_T(9));
    s2 = __fmul_rn(w, __fadd_rn(SFCC_ATAN_T(7), s2));
    s2 = __fmul_rn(w, __fadd_rn(SFCC_ATAN_T(5), s2));
    s2 = __fmul_rn(w, __fadd_rn(SFCC_ATAN_T(3), s2));
    s2 = __fmul_rn(w, __fadd_rn(SFCC_ATAN_T(1), s2));
    float s = __fadd_rn(s1, s2);
    if (id < 0) return __fsub_rn(x, __fmul_rn(x, s));
    float r = __fsub_rn(SFCC_ATAN_HI(id),
                        __fsub_rn(__fsub_rn(__fmul_rn(x, s), SFCC_ATAN_LO(id)), x));
    return (signed_hx < 0) ? -r : r;
#undef SFCC_ATAN_HI
#undef SFCC_ATAN_LO
#undef SFCC_ATAN_T
}

// ---------------------------------------------------------------------------
// SFCLAYINIT (module_sf_sfclay.F:954-969), one entry at a time.  WRF fills
// PSIMTB/PSIHTB(0:1000) once; evaluating entry n here gives the same word.
//   ZOLN = -FLOAT(N)*0.01                  -(n*0.01)
//   X    = (1-16.*ZOLN)**0.25
//   PSIMTB = 2*ALOG(0.5*(1+X)) + ALOG(0.5*(1+X*X)) - 2.*ATAN(X) + 2.*ATAN(1.)
//   Y    = (1-16*ZOLN)**0.5                a REAL power: powf, not sqrtf
//   PSIHTB = 2*ALOG(0.5*(1+Y))
// gfortran folds 2.*ATAN(1.) at compile time to the correctly rounded
// float32 2*atan(1) = 0x3FC90FDB, so that is a literal here.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sfc_zoln(int n)
{
    return -__fmul_rn((float)n, 0.01f);
}

__device__ float sfc_psimtb(int n)
{
    float x = gfk_pow(__fsub_rn(1.0f, __fmul_rn(16.0f, sfc_zoln(n))), 0.25f);
    float a = __fmul_rn(2.0f, gfk_log(__fmul_rn(0.5f, __fadd_rn(1.0f, x))));
    float b = gfk_log(__fmul_rn(0.5f, __fadd_rn(1.0f, __fmul_rn(x, x))));
    float c = __fmul_rn(2.0f, sfc_classic_atanf(x));
    return __fadd_rn(__fsub_rn(__fadd_rn(a, b), c), __uint_as_float(0x3FC90FDBu));
}

__device__ float sfc_psihtb(int n)
{
    float y = gfk_pow(__fsub_rn(1.0f, __fmul_rn(16.0f, sfc_zoln(n))), 0.5f);
    return __fmul_rn(2.0f, gfk_log(__fmul_rn(0.5f, __fadd_rn(1.0f, y))));
}

// SFCLAY1D:669-680, one (ZOL, PSIM, PSIH) lookup.
//   NZOL = INT(-ZOL*100.);  RZOL = -ZOL*100. - NZOL
//   PSI  = TB(NZOL) + RZOL*(TB(NZOL+1) - TB(NZOL))
__device__ __forceinline__ void sfc_table(float zol, float &psim, float &psih)
{
    float x = -__fmul_rn(zol, 100.0f);
    int n = (int)x;
    float r = __fsub_rn(x, (float)n);
    float m0 = sfc_psimtb(n), m1 = sfc_psimtb(n + 1);
    float h0 = sfc_psihtb(n), h1 = sfc_psihtb(n + 1);
    psim = __fadd_rn(m0, __fmul_rn(r, __fsub_rn(m1, m0)));
    psih = __fadd_rn(h0, __fmul_rn(r, __fsub_rn(h1, h0)));
}

// ALOG(num/z0) with sfclay.cu's defined rescue for a quotient that
// overflows (z0 <= 0, or z0 below num/FLT_MAX) and NaN propagation.
__device__ float sfc_log_zratio(float num, float z0)
{
    if (isnan(num) || isnan(z0)) return num + z0;
    float r = gfk_log(__fdiv_rn(num, z0));
    if (isfinite(r)) return r;
    unsigned int iz = __float_as_uint(z0);
    double zd;
    if (((iz >> 23) & 0xffu) == 0u) {           // zero or subnormal, DAZ-proof
        zd = (double)(iz & 0x7fffffu) * 0x1p-149;
        if (iz & 0x80000000u) zd = -zd;
    } else {
        zd = (double)z0;
    }
    if (!(zd > 0.0)) zd = 1.1754943508222875e-38;   // FLT_MIN
    return (float)(log((double)num) - log(zd));
}

__device__ __forceinline__ float sfc_max(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ float sfc_min(float a, float b) { return a < b ? a : b; }

// SFCLAY1D for one column.  The twelve read-only inputs arrive as values
// the kernel loaded with `name[idx]` (the ensemble batch remaps exactly that
// spelling for member-shared fields, gpuwm/ensemble/batch_physics.py); the
// inout and output fields are written at index i.  Argument names are WRF's;
// statement numbers in comments are module_sf_sfclay.F v4.6.1 lines.
__device__ void sfclay_classic_point(
    float ux, float vx, float t1d, float qx, float p1d, float dz8w,
    float psfcpa, float tsk, float pblh, float mavail, float xland,
    float lakemask, int i,
    float* __restrict__ znt_a, float* __restrict__ ust_a,
    float* __restrict__ ustm_a, float* __restrict__ mol_a,
    float* __restrict__ hfx_a, float* __restrict__ qfx_a,
    float* __restrict__ qsfc_a, float* __restrict__ zol_a,
    float* __restrict__ regime_a, float* __restrict__ psim_a,
    float* __restrict__ psih_a, float* __restrict__ fm_a,
    float* __restrict__ fh_a, float* __restrict__ lh_a,
    float* __restrict__ u10_a, float* __restrict__ v10_a,
    float* __restrict__ th2_a, float* __restrict__ t2_a,
    float* __restrict__ q2_a, float* __restrict__ chs_a,
    float* __restrict__ chs2_a, float* __restrict__ cqs2_a,
    float* __restrict__ flhc_a, float* __restrict__ flqc_a,
    float* __restrict__ qgh_a, float* __restrict__ rmol_a,
    float* __restrict__ wspd_a, float* __restrict__ br_a,
    float* __restrict__ gz1oz0_a, float* __restrict__ cpm_a,
    float* __restrict__ ck_a, float* __restrict__ cka_a,
    float* __restrict__ cd_a, float* __restrict__ cda_a,
    float dx, int isfflx, int isftcflx, int iz0tlnd)
{
    // Constants as module_surface_driver.F passes them: CP, G, ROVCP=rcp,
    // R=r_d, XLV, SVP1-3, SVPT0, EP1=ep_1, EP2=ep_2, KARMAN, P1000mb.
    const float karman = 0.4f, xka = 2.4e-5f, prt = 1.0f;
    const float salinity_factor = 0.98f, vconvc = 1.0f;
    const float czo = 0.0185f, ozo = 1.59e-5f;
    const float p1000mb = P0, rovcp = RCP, r = RD;
    const float ep1 = __fsub_rn(__fdiv_rn(RV, RD), 1.0f);
    const float g = G, cp = CP;

    const float znt = znt_a[i], ust0 = ust_a[i], ustm0 = ustm_a[i];
    const float mol0 = mol_a[i], hfx0 = hfx_a[i], qfx0 = qfx_a[i];
    float qsfc = qsfc_a[i], zol = zol_a[i];

    float psfc = __fdiv_rn(psfcpa, 1000.0f);                          // :396
    float thgb = __fmul_rn(tsk, gfk_pow(__fdiv_rn(p1000mb, psfcpa), rovcp));  // :405
    float pl = __fdiv_rn(p1d, 1000.0f);                               // :430
    float thcon = gfk_pow(__fdiv_rn(__fmul_rn(p1000mb, 0.001f), pl), rovcp);  // :433
    float scr3 = t1d;
    float thx = __fmul_rn(scr3, thcon);                               // :434
    float tvcon = __fadd_rn(1.0f, __fmul_rn(ep1, qx));                // :450
    float thvx = __fmul_rn(thx, tvcon);
    float scr4 = __fmul_rn(scr3, tvcon);
    float e1 = __fmul_rn(SVP1, gfk_exp(__fdiv_rn(__fmul_rn(SVP2, __fsub_rn(tsk, SVPT0)),
                                                 __fsub_rn(tsk, SVP3))));     // :456
    if (xland > 1.5f && lakemask == 0.0f) e1 = __fmul_rn(e1, salinity_factor);
    if (xland > 1.5f || surface_le_zero(qsfc))
        qsfc = __fdiv_rn(__fmul_rn(EP2, e1), __fsub_rn(psfc, e1));      // :460
    e1 = __fmul_rn(SVP1, gfk_exp(__fdiv_rn(__fmul_rn(SVP2, __fsub_rn(t1d, SVPT0)),
                                           __fsub_rn(t1d, SVP3))));           // :463
    pl = __fdiv_rn(p1d, 1000.0f);
    float qgh = __fdiv_rn(__fmul_rn(EP2, e1), __fsub_rn(pl, e1));      // :465
    float cpm = __fmul_rn(cp, __fadd_rn(1.0f, __fmul_rn(0.8f, qx)));   // :466
    float rhox = __fdiv_rn(__fmul_rn(psfc, 1000.0f), __fmul_rn(r, scr4));     // :475
    float zqkl = __fadd_rn(dz8w, 0.0f);                          // :479
    float za = __fmul_rn(0.5f, __fadd_rn(zqkl, 0.0f));                // :483
    float govrth = __fdiv_rn(g, thx);                                 // :487

    float gz1oz0 = sfc_log_zratio(za, znt);                           // :494
    float gz2oz0 = sfc_log_zratio(2.0f, znt);
    float gz10oz0 = sfc_log_zratio(10.0f, znt);
    float wspd = __fsqrt_rn(__fadd_rn(__fmul_rn(ux, ux), __fmul_rn(vx, vx)));  // :502
    float tskv = __fmul_rn(thgb, __fadd_rn(1.0f, __fmul_rn(ep1, qsfc)));      // :504
    float dthvdz = __fsub_rn(thvx, tskv);
    float vconv;
    if (xland < 1.5f) {                                               // :512
        float fluxc = sfc_max(__fadd_rn(__fdiv_rn(__fdiv_rn(hfx0, rhox), cp),
                                        __fdiv_rn(__fmul_rn(__fmul_rn(ep1, tskv), qfx0), rhox)),
                              0.0f);
        vconv = __fmul_rn(vconvc, gfk_pow(__fmul_rn(__fmul_rn(__fdiv_rn(g, tsk),
                                                    pblh), fluxc), 0.33f));
    } else {
        float dthvm = (-dthvdz >= 0.0f) ? -dthvdz : 0.0f;
        vconv = __fsqrt_rn(dthvm);                                    // :524
    }
    float vsgd = __fmul_rn(0.32f, gfk_pow(sfc_max(__fsub_rn(__fdiv_rn(dx, 5000.0f), 1.0f),
                                                  0.0f), 0.33f));     // :527
    wspd = __fsqrt_rn(__fadd_rn(__fadd_rn(__fmul_rn(wspd, wspd),
                                          __fmul_rn(vconv, vconv)),
                                __fmul_rn(vsgd, vsgd)));              // :528
    wspd = sfc_max(wspd, 0.1f);
    float br = __fdiv_rn(__fmul_rn(__fmul_rn(govrth, za), dthvdz),
                         __fmul_rn(wspd, wspd));                      // :530
    if (surface_lt_zero(mol0)) br = sfc_min(br, 0.0f);

    // z/L from the previous step's MOL and UST (:588, :645, :659).
    float zol_mo = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(karman, govrth), za), mol0),
                             __fmul_rn(ust0, ust0));
    float regime, rmol;
    float psim, psih, psim10, psih10, psim2, psih2;
    if (br < 0.0f) {                                                  // :654 class 4
        regime = 4.0f;
        zol = (ust0 < 0.01f) ? __fmul_rn(br, gz1oz0) : zol_mo;
        float zol10 = __fmul_rn(__fdiv_rn(10.0f, za), zol);
        float zol2 = __fmul_rn(__fdiv_rn(2.0f, za), zol);
        zol = sfc_max(sfc_min(zol, 0.0f), -9.9999f);
        zol10 = sfc_max(sfc_min(zol10, 0.0f), -9.9999f);
        zol2 = sfc_max(sfc_min(zol2, 0.0f), -9.9999f);
        sfc_table(zol, psim, psih);
        sfc_table(zol10, psim10, psih10);
        sfc_table(zol2, psim2, psih2);
        psih = sfc_min(psih, __fmul_rn(0.9f, gz1oz0));                // :686
        psim = sfc_min(psim, __fmul_rn(0.9f, gz1oz0));
        psih2 = sfc_min(psih2, __fmul_rn(0.9f, gz2oz0));
        psim10 = sfc_min(psim10, __fmul_rn(0.9f, gz10oz0));
        psih10 = sfc_min(psih10, __fmul_rn(0.9f, gz10oz0));
        rmol = __fdiv_rn(zol, za);
    } else if (!(br < 0.2f)) {                                        // :571 class 1
        regime = 1.0f;
        psim = sfc_max(__fmul_rn(-10.0f, gz1oz0), -10.0f);
        psih = psim;
        psim10 = sfc_max(__fmul_rn(__fdiv_rn(10.0f, za), psim), -10.0f);
        psih10 = psim10;
        psim2 = sfc_max(__fmul_rn(__fdiv_rn(2.0f, za), psim), -10.0f);
        psih2 = psim2;
        rmol = (ust0 < 0.01f) ? __fmul_rn(br, gz1oz0) : zol_mo;
        rmol = __fdiv_rn(sfc_min(rmol, 9.999f), za);
        // ZOL is not written in class 1: it leaves as it came in.
    } else if (!(br == 0.0f)) {                                       // :597 class 2
        regime = 2.0f;
        psim = sfc_max(__fdiv_rn(__fmul_rn(__fmul_rn(-5.0f, br), gz1oz0),
                                 __fsub_rn(1.1f, __fmul_rn(5.0f, br))), -10.0f);
        psih = psim;
        psim10 = sfc_max(__fmul_rn(__fdiv_rn(10.0f, za), psim), -10.0f);
        psih10 = psim10;
        psim2 = sfc_max(__fmul_rn(__fdiv_rn(2.0f, za), psim), -10.0f);
        psih2 = psim2;
        zol = __fdiv_rn(__fmul_rn(br, gz1oz0), __fsub_rn(1.00001f, __fmul_rn(5.0f, br)));
        if (zol > 0.5f) {
            zol = __fadd_rn(
                __fmul_rn(__fmul_rn(__fadd_rn(__fmul_rn(1.89f, gz1oz0), 44.2f), br), br),
                __fmul_rn(__fsub_rn(__fmul_rn(1.18f, gz1oz0), 1.37f), br));
            zol = sfc_min(zol, 9.999f);
        }
        rmol = __fdiv_rn(zol, za);
    } else {                                                          // :633 class 3
        regime = 3.0f;
        psim = 0.0f; psih = 0.0f; psim10 = 0.0f; psih10 = 0.0f;
        psim2 = 0.0f; psih2 = 0.0f;
        zol = (ust0 < 0.01f) ? __fmul_rn(br, gz1oz0) : zol_mo;
        rmol = __fdiv_rn(zol, za);
    }

    // ---- loop 330 -------------------------------------------------------
    bool water = xland >= 1.5f;                       // (XLAND-1.5).GE.0
    float dtg = __fsub_rn(thx, thgb);
    float psix = __fsub_rn(gz1oz0, psim);
    float psix10 = __fsub_rn(gz10oz0, psim10);
    float psit = sfc_max(__fsub_rn(gz1oz0, psih), 2.0f);
    float zl = water ? znt : 0.01f;
    float kust = __fmul_rn(karman, ust0);
    float psiq = __fsub_rn(gfk_log(__fadd_rn(__fdiv_rn(__fmul_rn(kust, za), xka),
                                             __fdiv_rn(za, zl))), psih);      // :713
    float psit2 = __fsub_rn(gz2oz0, psih2);
    float psiq2 = __fsub_rn(gfk_log(__fadd_rn(__fdiv_rn(__fmul_rn(kust, 2.0f), xka),
                                              __fdiv_rn(2.0f, zl))), psih2);
    float psiq10 = __fsub_rn(gfk_log(__fadd_rn(__fdiv_rn(__fmul_rn(kust, 10.0f), xka),
                                               __fdiv_rn(10.0f, zl))), psih10);
    if (water) {                                                      // :721
        float visc = __fmul_rn(__fadd_rn(1.32f, __fmul_rn(0.009f, __fsub_rn(scr3, 273.15f))),
                               1.0e-5f);
        float restar = __fdiv_rn(__fmul_rn(ust0, znt), visc);
        float z0t = __fmul_rn(5.5e-5f, gfk_pow(restar, -0.60f));
        z0t = sfc_min(z0t, 1.0e-4f);
        z0t = sfc_max(z0t, 2.0e-9f);
        float z0q = z0t;
        psiq = sfc_max(__fsub_rn(gfk_log(__fdiv_rn(__fadd_rn(za, z0q), z0q)), psih), 2.0f);
        psit = sfc_max(__fsub_rn(gfk_log(__fdiv_rn(__fadd_rn(za, z0t), z0t)), psih), 2.0f);
        psiq2 = sfc_max(__fsub_rn(gfk_log(__fdiv_rn(__fadd_rn(2.0f, z0q), z0q)), psih2), 2.0f);
        psit2 = sfc_max(__fsub_rn(gfk_log(__fdiv_rn(__fadd_rn(2.0f, z0t), z0t)), psih2), 2.0f);
        psiq10 = sfc_max(__fsub_rn(gfk_log(__fdiv_rn(__fadd_rn(10.0f, z0q), z0q)), psih10), 2.0f);
    }
    if (isftcflx == 1 && water) {                                     // :739
        float z0q = 1.0e-4f;
        psiq = __fsub_rn(gfk_log(__fdiv_rn(za, z0q)), psih);
        psit = psiq;
        psiq2 = __fsub_rn(gfk_log(__fdiv_rn(2.0f, z0q)), psih2);
        psiq10 = __fsub_rn(gfk_log(__fdiv_rn(10.0f, z0q)), psih10);
        psit2 = psiq2;
    }
    if (isftcflx == 2 && water) {                                     // :752
        float visc = __fmul_rn(__fadd_rn(1.32f, __fmul_rn(0.009f, __fsub_rn(scr3, 273.15f))),
                               1.0e-5f);
        float restar = __fdiv_rn(__fmul_rn(ust0, znt), visc);
        float rr = __fsqrt_rn(__fsqrt_rn(restar));                    // SQRT(SQRT(RESTAR))
        float gz0ozt = __fmul_rn(0.40f, __fsub_rn(__fmul_rn(__fmul_rn(7.3f, rr),
                                                            __fsqrt_rn(0.71f)), 5.0f));
        float gz0ozq = __fmul_rn(0.40f, __fsub_rn(__fmul_rn(__fmul_rn(7.3f, rr),
                                                            __fsqrt_rn(0.60f)), 5.0f));
        psit = __fadd_rn(__fsub_rn(gz1oz0, psih), gz0ozt);
        psiq = __fadd_rn(__fsub_rn(gz1oz0, psih), gz0ozq);
        psit2 = __fadd_rn(__fsub_rn(gz2oz0, psih2), gz0ozt);
        psiq2 = __fadd_rn(__fsub_rn(gz2oz0, psih2), gz0ozq);
        // :767 writes PSIH here; the 10 m term is PSIH10 (header note).
        psiq10 = __fadd_rn(__fsub_rn(gz10oz0, psih10), gz0ozq);
    }
    float ck = __fmul_rn(__fdiv_rn(karman, psix10), __fdiv_rn(karman, psiq10));   // :771
    float cd = __fmul_rn(__fdiv_rn(karman, psix10), __fdiv_rn(karman, psix10));
    float cka = __fmul_rn(__fdiv_rn(karman, psix), __fdiv_rn(karman, psiq));
    float cda = __fmul_rn(__fdiv_rn(karman, psix), __fdiv_rn(karman, psix));
    if (iz0tlnd >= 1 && xland <= 1.5f) {                             // :777
        float zlc = znt;
        float visc = __fmul_rn(__fadd_rn(1.32f, __fmul_rn(0.009f, __fsub_rn(scr3, 273.15f))),
                               1.0e-5f);
        float restar = __fdiv_rn(__fmul_rn(ust0, zlc), visc);
        float czil = (iz0tlnd == 1)
            ? gfk_pow(10.0f, -__fmul_rn(0.40f, __fdiv_rn(zlc, 0.07f)))   // :786
            : 0.1f;
        float add = __fmul_rn(__fmul_rn(czil, karman), __fsqrt_rn(restar));
        psit = __fadd_rn(__fsub_rn(gz1oz0, psih), add);
        psiq = __fadd_rn(__fsub_rn(gz1oz0, psih), add);
        psit2 = __fadd_rn(__fsub_rn(gz2oz0, psih2), add);
        psiq2 = __fadd_rn(__fsub_rn(gz2oz0, psih2), add);
    }
    float ust = __fadd_rn(__fmul_rn(0.5f, ust0),
                          __fdiv_rn(__fmul_rn(__fmul_rn(0.5f, karman), wspd), psix));  // :799
    float wspdi = __fsqrt_rn(__fadd_rn(__fmul_rn(ux, ux), __fmul_rn(vx, vx)));
    float ustm = __fadd_rn(__fmul_rn(0.5f, ustm0),
                           __fdiv_rn(__fmul_rn(__fmul_rn(0.5f, karman), wspdi), psix));
    float u10 = __fdiv_rn(__fmul_rn(ux, psix10), psix);
    float v10 = __fdiv_rn(__fmul_rn(vx, psix10), psix);
    float th2 = __fadd_rn(thgb, __fdiv_rn(__fmul_rn(dtg, psit2), psit));
    float q2 = __fadd_rn(qsfc, __fdiv_rn(__fmul_rn(__fsub_rn(qx, qsfc), psiq2), psiq));
    float t2 = __fmul_rn(th2, gfk_pow(__fdiv_rn(psfcpa, p1000mb), rovcp));   // :810
    if (xland < 1.5f) ust = sfc_max(ust, 0.1f);                       // :817
    float mol = __fdiv_rn(__fdiv_rn(__fmul_rn(karman, dtg), psit), prt);
    float denomq = psiq, denomq2 = psiq2, denomt2 = psit2;

    // ---- fluxes (:834-937) --------------------------------------------------
    float qfx = 0.0f, hfx = 0.0f, flhc = 0.0f, flqc = 0.0f;
    float znt_out = znt;
    // ISFFLX=0 jumps to statement 410, past every write of LH, CHS, CHS2
    // and CQS2, so WRF leaves them as its state arrays hold them.  WOOF's
    // launcher carries CHS/CHS2/CQS2 between calls (gpuwm.core.sfclay
    // launch_sfclay, zero-seeded by its allocating API, the zero WRF
    // initialises them to), so flux off keeps the incoming words, signed
    // zeros and subnormals included, as the revised scheme does.  LH is
    // unassigned in WRF on this path; WOOF defines it as zero.
    float lh = 0.0f;
    float chs = isfflx ? 0.0f : chs_a[i];
    float chs2 = isfflx ? 0.0f : chs2_a[i];
    float cqs2 = isfflx ? 0.0f : cqs2_a[i];
    if (isfflx != 0) {
        if (water) {                                                  // :845
            znt_out = __fadd_rn(__fdiv_rn(__fmul_rn(__fmul_rn(czo, ust), ust), g),
                                __fdiv_rn(__fmul_rn(0.11f, 1.5e-5f), ust));
            znt_out = sfc_min(znt_out, 2.85e-3f);
            if (isftcflx != 0) {
                float zw = sfc_min(gfk_pow(__fdiv_rn(ust, 1.06f), 0.3f), 1.0f);
                float zn1 = __fadd_rn(__fdiv_rn(__fmul_rn(__fmul_rn(0.011f, ust), ust), g), ozo);
                float zn2 = __fadd_rn(
                    __fmul_rn(10.0f, gfk_exp(-__fmul_rn(9.5f, gfk_pow(ust, -0.3333f)))),
                    __fdiv_rn(__fmul_rn(0.11f, 1.5e-5f), sfc_max(ust, 0.01f)));
                znt_out = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, zw), zn1), __fmul_rn(zw, zn2));
                znt_out = sfc_min(znt_out, 2.85e-3f);
                znt_out = sfc_max(znt_out, 1.27e-7f);
            }
        }
        flqc = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(rhox, mavail), ust), karman),
                         denomq);                                     // :877
        float dtthx = fabsf(__fsub_rn(thx, thgb));
        if (dtthx > 1.0e-5f)
            flhc = __fdiv_rn(__fmul_rn(__fmul_rn(__fmul_rn(cpm, rhox), ust), mol),
                             __fsub_rn(thx, thgb));
        qfx = __fmul_rn(flqc, __fsub_rn(qsfc, qx));                   // :899
        lh = __fmul_rn(XLV, qfx);
        if (xland > 1.5f || xland < 1.5f)                             // :908, :916
            hfx = __fmul_rn(flhc, __fsub_rn(thgb, thx));
        chs = __fdiv_rn(__fmul_rn(ust, karman), denomq);              // :930
        cqs2 = __fdiv_rn(__fmul_rn(ust, karman), denomq2);
        chs2 = __fdiv_rn(__fmul_rn(ust, karman), denomt2);
    }

    znt_a[i] = znt_out; ust_a[i] = ust; ustm_a[i] = ustm; mol_a[i] = mol;
    hfx_a[i] = hfx; qfx_a[i] = qfx; qsfc_a[i] = qsfc; zol_a[i] = zol;
    regime_a[i] = regime; psim_a[i] = psim; psih_a[i] = psih;
    fm_a[i] = psix; fh_a[i] = psit; lh_a[i] = lh;
    u10_a[i] = u10; v10_a[i] = v10; th2_a[i] = th2; t2_a[i] = t2; q2_a[i] = q2;
    chs_a[i] = chs; chs2_a[i] = chs2; cqs2_a[i] = cqs2;
    flhc_a[i] = flhc; flqc_a[i] = flqc; qgh_a[i] = qgh; rmol_a[i] = rmol;
    wspd_a[i] = wspd; br_a[i] = br; gz1oz0_a[i] = gz1oz0; cpm_a[i] = cpm;
    ck_a[i] = ck; cka_a[i] = cka; cd_a[i] = cd; cda_a[i] = cda;
}

// SFCLAYINIT's two tables, entry by entry, for the parity test.
extern "C" __global__
void sfclay_classic_table(float* __restrict__ psimtb,
                          float* __restrict__ psihtb, int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    psimtb[k] = sfc_psimtb(k);
    psihtb[k] = sfc_psihtb(k);
}

// The atanf above, for the equality sweep against mynn_glibc_atanf.
extern "C" __global__
void sfclay_classic_atanf_probe(const float* __restrict__ x,
                                float* __restrict__ out, int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    out[k] = sfc_classic_atanf(x[k]);
}
