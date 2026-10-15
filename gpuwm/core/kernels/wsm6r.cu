// WRF v4.6.1 WSM6R, the revised WSM6 (mp_physics = 106).
//
// A word-for-word port of phys/module_mp_wsm6r.F for one column: the
// wrapper wsm6r (t = th*pii in, th = t/pii out), wsm62D's minor loop and
// every routine it calls in its order -- fallk (rain/snow/graupel fallout
// with sub-steps, then snow and graupel melting), fallkc (cloud-ice
// fallout), rainsc (surface rain, then pimlt, pihmf, pihtf, pgfrz), warmr
// (praut, pracw, prevp), accret1 (praci, piacr, psaci, pgaci, psacw,
// pgacw), accret2 (pracs, psacr, pgacr, pgacs, pseml, pgeml), accret3
// (pidep, psdep, pgdep, pigen, psaut, pgaut, psevp, pgevp) and pconadd
// (pcond).  Line citations are into module_mp_wsm6r.F.
//
// WHY A TRANSLATION UNIT OF ITS OWN.  WSM6R is not WSM6 with edits.  It is
// a different scheme with WSM6's species set: every process is applied and
// committed one at a time (WSM6 forms all rates, then limits and applies
// them together), branch thresholds are replaced by the smoothif logistic
// switch evaluated in binary64, the slope/intercept formulation is replaced
// by precomputed power laws (the _a/_b/_c/_d words of wsm6rinit), and
// sedimentation is an explicit upstream flux with per-column sub-steps
// instead of WSM6's semi-Lagrangian PLM remap.  No line of mp_wsm6.F90 is
// textually shared, so there is no WSM6 device code to share; what IS
// shared is WOOF's own float32 and binary64 libm (glibc_flt32.cuh,
// glibc_flt64.cuh), the WSM6 family's host module (gpuwm/core/wsm6.py and
// wsm6_constants.py hold this scheme's launcher and frozen words beside
// WSM6's) and the WSM6 oracle harness layout.  kernels/wsm6.cu is a frozen
// unit (tests/test_mp8_frozen.py) and is not touched.
//
// BITWISE CONTRACT.  Graded word for word against module_mp_wsm6r.F
// compiled by gfortran 13.3.0 -O0 (tools/wsm6r_wrf461_column_oracle,
// tests/test_wsm6r_wrf461_column_exact.py).  What that takes:
//   * Every expression keeps Fortran's left-to-right association.
//     Parameter subexpressions gfortran folds at compile time ((bvtr-2.)/4.,
//     3.*alpha, (3-bvts), t0c+0.01, ...) are folded here the same way, one
//     float32 rounding per operation.
//   * EXP, LOG and REAL**REAL are WOOF's own float32 routines gfk_exp,
//     gfk_log and gfk_pow; smoothif's binary64 EXP is WOOF's own binary64
//     glibc_exp (glibc_flt64.cuh).  No exponent in this module is 2., 1. or
//     -1., so every ** is a pow call, as gfortran emits it.
//   * NINT is lroundf (round half away from zero), the libm call gfortran
//     emits (tools/.../undefined-O0.txt).
//   * Fortran MAX/MIN are gfortran's MAX_EXPR/MIN_EXPR on two operands in
//     argument order: MAX(a,b) is `b > a ? b : a`, MIN(a,b) `b < a ? b : a`
//     (rmax/rmin).  They keep a signed zero that fmaxf/fminf would not.
//   * Every float division is __fdiv_rn (NVRTC may turn `x / C` into a
//     multiply by the rounded reciprocal under -ftz=true, A146).
//   * The unit compiles with --fmad=false (kernels/__init__.py): gfortran on
//     x86-64 contracts nothing.
//   * wsm6rinit's derived words are frozen from WRF's own float32 init
//     (gpuwm/core/wsm6_constants.py WSM6R_INIT_F32) and passed in `cf`.
//
// LOOP FUSION.  After the two fallout routines, every routine wsm62D calls
// reads and writes level k only (rainsc's surface sum reads fall at kts,
// which fallk has finished), so they run here as one k loop, each level
// taking WRF's routines in WRF's order.  Within fallk, the snow-melting
// sweep and the graupel-melting sweep are per-level and run back to back
// at each level, which is the order WRF's two full sweeps give that level.
//
// DECLARED DIVERGENCE (rainncv).  WRF's rainsc sets RAINNCV to the CURRENT
// minor loop's fallout (overwriting the previous loop's) and only when that
// fallout exceeds qmin; module_microphysics_driver.F never zeroes it, so a
// multi-loop step reports its last loop only and a dry step reports the
// previous step's rain again.  RAINNCV is the per-step precipitation the
// land surface accumulates (RAINBL), so this port zeroes it at entry and
// ADDS each loop's fallout, the same word WRF adds to RAIN.  RAIN itself is
// WRF's, word for word.  Single-loop steps (dt <= 180 s) are therefore
// RAINNCV-identical to WRF with a zeroed RAINNCV.
//
// One CUDA thread owns a complete vertical column.  gpuwm stores arrays as
// (k,j,i), x fastest.  The column state lives in the caller's global
// arrays (the theta array holds the temperature during the call) plus one
// (3,nz,ny,nx) scratch for fallk's per-level fall speeds, so the thread
// frame is a handful of scalars at any nz.

// wsm6rinit's derived words, in WSM6R_INIT_F32 order.
enum Wsm6rCoef {
    R_PI, R_QC0, R_QCK1, R_ROQIMAX, R_XLV1,
    R_VT2R_A, R_VT2S_A, R_VT2G_A, R_VT2I_A, R_FALLI_A,
    R_PREVP_A, R_PREVP_B, R_PSDEP_A, R_PSDEP_B, R_PSEVP_A, R_PSEVP_B,
    R_PGDEP_A, R_PGDEP_B, R_PGEVP_A, R_PGEVP_B,
    R_PSMLT_A, R_PSMLT_B, R_PGMLT_A, R_PGMLT_B,
    R_PRACI_A, R_PRACI_B, R_PRACI_C, R_PRACI_D,
    R_PSACI_A, R_PSACI_B, R_PSACI_C, R_PSACI_D,
    R_PGACI_A, R_PGACI_B, R_PGACI_C, R_PGACI_D,
    R_PRACS_A, R_PRACS_B, R_PRACS_C, R_PRACS_D,
    R_PSACR_A, R_PSACR_B, R_PSACR_C, R_PSACR_D,
    R_PGACR_A, R_PGACR_B, R_PGACR_C, R_PGACR_D,
    R_PGACS_A, R_PGACS_B, R_PGACS_C, R_PGACS_D,
    R_PIDEP_A, R_DIFFAC_A, R_DIFFAC_B, R_PGFRZ_A,
    R_PIACR_A, R_PRACW_A, R_PSACW_A, R_PGACW_A,
    WSM6R_NCOEF
};

// module_mp_wsm6r PARAMETERs (:5-51).
#define R6_DTCLDCR 120.0f
#define R6_BVTR    0.8f
#define R6_XNCR    3.0e8f
#define R6_BVTS    0.41f
#define R6_BVTG    0.8f
#define R6_ALPHA   0.12f
#define R6_PFRZ1   100.0f
#define R6_PFRZ2   0.66f
#define R6_QS0     6.0e-4f
#define R6_RV      461.6f
#define R6_T0C     273.15f
#define R6_CPD     1004.5f
#define R6_CPV     1846.4f
#define R6_EP2     0.6217504f
#define R6_QCRMIN  1.0e-9f
#define R6_QMIN    1.0e-15f
#define R6_XLS     2.85e6f
#define R6_XLV0    2.5e6f
#define R6_XLF0    3.5e5f
#define R6_CLIQ    4190.0f
#define R6_CICE    2106.0f
#define R6_PSAT    610.78f
#define R6_DENR    1000.0f

#define R6DIV(a, b) __fdiv_rn((a), (b))

// gfortran's MAX_EXPR/MIN_EXPR operand order after frontend lowering.
// A computed second argument is moved ahead of a constant first argument.
// Keep that order at call sites, including unordered comparisons.
__device__ __forceinline__ float rmax(float a, float b) { return (b > a) ? b : a; }
__device__ __forceinline__ float rmin(float a, float b) { return (b < a) ? b : a; }

// Exponents and coefficients gfortran folds from PARAMETERs, one float32
// rounding per operation, in the source's own spelling.
struct Wsm6rFold {
    float e_vtr, e_qr4, e_vts, e_qs4, e_vtg, e_qg4;   // (bv-2.)/4., bv/4.
    float e_i131, e_i931;                             // 1.31/8., 9.31/8.
    float e_r1, e_r3, e_r6;                           // (1.+bvtr)/4., (3.+bvtr)/4., (6.+bvtr)/4.
    float e_s1, e_s3, e_g1, e_g3;                     // (1.+bvts)/4. ...
    float e_dr, e_ds, e_dg;                           // (13.+3.*bv)/24.
    float e_qr58, e_qs58, e_qg58;                     // (5.+bv)/8.
    float e_16, e_512, e_13, e_388, e_56;             // 1./6., 5.12/6., 1./3., 3.88/6., 5./6.
    float e_34, e_74, e_73, e_78, e_98, e_54, e_32, e_58;
    float c_bvts_alpha, c_3alpha, c_5alpha, c_1mbvts_alpha, c_3mbvts_alpha;
    float c_3mbvts, c_cpvmcpd;
    float ttp, xa, xb, xai, xbi;
};

__device__ __forceinline__ Wsm6rFold wsm6r_fold() {
    Wsm6rFold f;
    f.e_vtr = R6DIV(R6_BVTR - 2.0f, 4.0f);  f.e_qr4 = R6DIV(R6_BVTR, 4.0f);
    f.e_vts = R6DIV(R6_BVTS - 2.0f, 4.0f);  f.e_qs4 = R6DIV(R6_BVTS, 4.0f);
    f.e_vtg = R6DIV(R6_BVTG - 2.0f, 4.0f);  f.e_qg4 = R6DIV(R6_BVTG, 4.0f);
    f.e_i131 = R6DIV(1.31f, 8.0f);          f.e_i931 = R6DIV(9.31f, 8.0f);
    f.e_r1 = R6DIV(1.0f + R6_BVTR, 4.0f);   f.e_r3 = R6DIV(3.0f + R6_BVTR, 4.0f);
    f.e_r6 = R6DIV(6.0f + R6_BVTR, 4.0f);
    f.e_s1 = R6DIV(1.0f + R6_BVTS, 4.0f);   f.e_s3 = R6DIV(3.0f + R6_BVTS, 4.0f);
    f.e_g1 = R6DIV(1.0f + R6_BVTG, 4.0f);   f.e_g3 = R6DIV(3.0f + R6_BVTG, 4.0f);
    f.e_dr = R6DIV(13.0f + 3.0f * R6_BVTR, 24.0f);
    f.e_ds = R6DIV(13.0f + 3.0f * R6_BVTS, 24.0f);
    f.e_dg = R6DIV(13.0f + 3.0f * R6_BVTG, 24.0f);
    f.e_qr58 = R6DIV(5.0f + R6_BVTR, 8.0f);
    f.e_qs58 = R6DIV(5.0f + R6_BVTS, 8.0f);
    f.e_qg58 = R6DIV(5.0f + R6_BVTG, 8.0f);
    f.e_16 = R6DIV(1.0f, 6.0f);  f.e_512 = R6DIV(5.12f, 6.0f);
    f.e_13 = R6DIV(1.0f, 3.0f);  f.e_388 = R6DIV(3.88f, 6.0f);
    f.e_56 = R6DIV(5.0f, 6.0f);
    f.e_34 = R6DIV(3.0f, 4.0f);  f.e_74 = R6DIV(7.0f, 4.0f);
    f.e_73 = R6DIV(7.0f, 3.0f);  f.e_78 = R6DIV(7.0f, 8.0f);
    f.e_98 = R6DIV(9.0f, 8.0f);  f.e_54 = R6DIV(5.0f, 4.0f);
    f.e_32 = R6DIV(3.0f, 2.0f);  f.e_58 = R6DIV(5.0f, 8.0f);
    f.c_bvts_alpha = R6_BVTS * R6_ALPHA;            // bvts*alpha == alpha*bvts
    f.c_3alpha = 3.0f * R6_ALPHA;
    f.c_5alpha = 5.0f * R6_ALPHA;
    f.c_1mbvts_alpha = (1.0f - R6_BVTS) * R6_ALPHA;
    f.c_3mbvts_alpha = (3.0f - R6_BVTS) * R6_ALPHA;
    f.c_3mbvts = 3.0f - R6_BVTS;                    // (3-bvts), integer 3
    f.c_cpvmcpd = R6_CPV - R6_CPD;
    // calcrh's PARAMETERs (:306-315).
    f.ttp = R6_T0C + 0.01f;
    const float dldt = R6_CPV - R6_CLIQ;
    f.xa = -R6DIV(dldt, R6_RV);
    f.xb = f.xa + R6DIV(R6_XLV0, R6_RV * f.ttp);
    const float dldti = R6_CPV - R6_CICE;
    f.xai = -R6DIV(dldti, R6_RV);
    f.xbi = f.xai + R6DIV(R6_XLS, R6_RV * f.ttp);
    return f;
}

// cpmcal / xlcal (:1786-1799).
__device__ __forceinline__ float r6_cpm(const Wsm6rFold &f, float x) {
    return R6_CPD + x * f.c_cpvmcpd;
}
__device__ __forceinline__ float r6_xl(const float *cf, float x) {
    return R6_XLV0 - cf[R_XLV1] * (x - R6_T0C);
}

// smoothif (:1729-1753): a logistic switch in binary64, centred on a, with
// c1 = 1e-15 ('q') or 1e-9 ('t') and the '+' shift of 710/k1.  The real(4)
// literals are widened to real(8) exactly as gfortran widens them.
__device__ __forceinline__ float r6_smoothif(float x, float a, bool q, bool plus) {
    const double c1 = q ? (double)1.0e-15f : (double)1.0e-9f;
    const double k1 = __ddiv_rn(747.0, c1);
    const double b = plus ? __dadd_rn((double)a, __ddiv_rn(710.0, k1)) : (double)a;
    const double k = -__dmul_rn(k1, __dsub_rn((double)x, b));
    const double f1 = __ddiv_rn(1.0, __dadd_rn(1.0, glibc_exp(k)));
    return gfk_d2f_rn(f1);
}

// calcrh (:300-327).
struct Wsm6rRh { float rh1, rh2, qs1, qs2; };
__device__ __forceinline__ Wsm6rRh r6_calcrh(const Wsm6rFold &f, float t, float p, float q) {
    Wsm6rRh r;
    const float tr = R6DIV(f.ttp, t);
    const float qs10 = (R6_PSAT * gfk_exp(gfk_log(tr) * f.xa)) * gfk_exp(f.xb * (1.0f - tr));
    const float qs11 = R6DIV(R6_EP2 * qs10, p - qs10);
    r.qs1 = qs11;
    r.rh1 = R6DIV(q, rmax(r.qs1, R6_QMIN));
    const float qs20 = (R6_PSAT * gfk_exp(gfk_log(tr) * f.xai)) * gfk_exp(f.xbi * (1.0f - tr));
    const float qs21 = R6DIV(R6_EP2 * qs20, p - qs20);
    r.qs2 = qs21;
    r.rh2 = R6DIV(q, rmax(r.qs2, R6_QMIN));
    return r;
}

// max(0.,min(90.,t0c-t)) -- the capped supercooling of the n0s laws.
__device__ __forceinline__ float r6_cap90(float t) {
    return rmax(rmin(R6_T0C - t, 90.0f), 0.0f);
}

// The diffusion/conduction denominators of prevp/pidep/psdep/... :
// c = diffac_a*den*x*x*(t+120.)/rv/(t**3.5), d = diffac_b*p/(t**1.81)/qs.
__device__ __forceinline__ float r6_diffc(const float *cf, float den, float x, float t) {
    return R6DIV(R6DIV(((((cf[R_DIFFAC_A] * den) * x) * x) * (t + 120.0f)), R6_RV),
                 gfk_pow(t, 3.5f));
}
__device__ __forceinline__ float r6_diffd(const float *cf, float p, float t, float qs) {
    return R6DIV(R6DIV(cf[R_DIFFAC_B] * p, gfk_pow(t, 1.81f)), qs);
}
// ((t+120.)**(1./6.))/(t**(5.12/6.))*(p**(1./3.))*(den**e_den)*(max(q,qcrmin)**e_q)
__device__ __forceinline__ float r6_ventb(const Wsm6rFold &f, float t, float p, float den,
                                          float qq, float e_den, float e_q) {
    return ((R6DIV(gfk_pow(t + 120.0f, f.e_16), gfk_pow(t, f.e_512))
             * gfk_pow(p, f.e_13)) * gfk_pow(den, e_den)) * gfk_pow(rmax(qq, R6_QCRMIN), e_q);
}

// One level's prognostic words during the fused process loop.
struct Wsm6rLevel {
    float t, q, qc, qi, qr, qs, qg;
    float den, p;
};

// rainsc's per-level half (:766-852): pimlt, pihmf, pihtf, pgfrz.
__device__ __forceinline__ void r6_rainsc_level(const float *cf, const Wsm6rFold &f,
                                                Wsm6rLevel &v, float cpm, float dtcld) {
    float xl, xlf, supcol, qtmp, ft0, ft40;
    // pimlt
    xl = r6_xl(cf, v.t);
    xlf = R6_XLS - xl;
    supcol = R6_T0C - v.t;
    if (supcol < 0.0f) xlf = R6_XLF0;
    ft0 = r6_smoothif(v.t, R6_T0C, false, false);
    qtmp = ft0 * rmax(v.qi, 0.0f);
    if (fabsf(qtmp) < R6_QMIN) qtmp = 0.0f;
    v.qc = rmax(v.qc + qtmp, 0.0f);
    v.qi = rmax(v.qi - qtmp, 0.0f);
    v.t = v.t - R6DIV(xlf, cpm) * qtmp;
    // pihmf
    xl = r6_xl(cf, v.t);
    xlf = R6_XLS - xl;
    supcol = R6_T0C - v.t;
    if (supcol < 0.0f) xlf = R6_XLF0;
    ft40 = r6_smoothif(supcol, 40.0f, false, false);
    qtmp = rmax(ft40 * v.qc, 0.0f);
    if (fabsf(qtmp) < R6_QMIN) qtmp = 0.0f;
    v.qi = rmax(v.qi + qtmp, 0.0f);
    v.qc = rmax(v.qc - qtmp, 0.0f);
    v.t = v.t + R6DIV(xlf, cpm) * qtmp;
    // pihtf
    xl = r6_xl(cf, v.t);
    xlf = R6_XLS - xl;
    supcol = R6_T0C - v.t;
    if (supcol < 0.0f) xlf = R6_XLF0;
    ft40 = r6_smoothif(-supcol, -40.0f, false, false);
    {
        // gfortran's scalar MIN lowering selects the donor word when the
        // computed freezing rate is unordered (overflow times zero).
        const float freezing = ((R6DIV(R6DIV(
            (R6_PFRZ1 * (gfk_exp(R6_PFRZ2 * supcol) - 1.0f)) * v.den,
            R6_DENR), R6_XNCR) * v.qc) * v.qc) * dtcld;
        const float pfrzdtc = rmin(v.qc, freezing);
        qtmp = rmax(ft40 * pfrzdtc, 0.0f);
    }
    if (fabsf(qtmp) < R6_QMIN) qtmp = 0.0f;
    v.qi = rmax(v.qi + qtmp, 0.0f);
    v.qc = rmax(v.qc - qtmp, 0.0f);
    v.t = v.t + R6DIV(xlf, cpm) * qtmp;
    // pgfrz
    xl = r6_xl(cf, v.t);
    xlf = R6_XLS - xl;
    supcol = R6_T0C - v.t;
    if (supcol < 0.0f) xlf = R6_XLF0;
    {
        float temp;
        if (v.qr > 0.0f)
            temp = ((cf[R_PGFRZ_A] * (gfk_exp(R6_PFRZ2 * supcol) - 1.0f))
                    * gfk_pow(v.den, f.e_34)) * gfk_pow(v.qr, f.e_74);
        else
            temp = 0.0f;
        const float pfrzdtr = rmin(v.qr, temp * dtcld);
        qtmp = rmax(pfrzdtr, 0.0f);
    }
    if (fabsf(qtmp) < R6_QMIN) qtmp = 0.0f;
    v.qg = rmax(v.qg + qtmp, 0.0f);
    v.qr = rmax(v.qr - qtmp, 0.0f);
    v.t = v.t + R6DIV(xlf, cpm) * qtmp;
}

// warmr (:874-951): praut, pracw, prevp.
__device__ __forceinline__ void r6_warmr_level(const float *cf, const Wsm6rFold &f,
                                               Wsm6rLevel &v, float dtcld) {
    const float qcut = R6DIV(R6_QMIN, dtcld);
    // praut
    const float fqc0 = r6_smoothif(v.qc, cf[R_QC0], true, false);
    float praut1;
    if (v.qc > 0.0f)
        praut1 = (fqc0 * cf[R_QCK1]) * gfk_exp(gfk_log(v.qc) * f.e_73);
    else
        praut1 = 0.0f;
    float praut = rmin(praut1, R6DIV(v.qc, dtcld));
    if (fabsf(praut) < qcut) praut = 0.0f;
    v.qc = rmax(v.qc - praut * dtcld, 0.0f);
    v.qr = rmax(v.qr + praut * dtcld, 0.0f);
    // pracw
    float pracw;
    if (v.qr > 0.0f && v.qc > 0.0f)
        pracw = ((cf[R_PRACW_A] * gfk_pow(v.den, f.e_r1)) * gfk_pow(v.qr, f.e_r3)) * v.qc;
    else
        pracw = 0.0f;
    pracw = rmax(rmin(pracw, R6DIV(v.qc, dtcld)), 0.0f);
    if (fabsf(pracw) < qcut) pracw = 0.0f;
    v.qc = rmax(v.qc - pracw * dtcld, 0.0f);
    v.qr = rmax(v.qr + pracw * dtcld, 0.0f);
    // prevp
    const Wsm6rRh rh = r6_calcrh(f, v.t, v.p, v.q);
    const float xl = r6_xl(cf, v.t);
    const float cpm = r6_cpm(f, v.q);
    const float supsat = v.q - rh.qs1;
    const float satdt = R6DIV(supsat, dtcld);
    const float a = sqrtf(v.den * rmax(v.qr, R6_QCRMIN));
    const float b = r6_ventb(f, v.t, v.p, v.den, v.qr, f.e_dr, f.e_qr58);
    const float c = r6_diffc(cf, v.den, xl, v.t);
    const float d = r6_diffd(cf, v.p, v.t, rh.qs1);
    const float e = R6DIV(rh.rh1 - 1.0f, c + d);
    float prevp = (cf[R_PREVP_A] * a + cf[R_PREVP_B] * b) * e;
    if (prevp < 0.0f)
        prevp = rmin(rmax(prevp, R6DIV(-v.qr, dtcld)), 0.0f);
    else
        prevp = rmax(rmin(prevp, satdt), 0.0f);
    if (fabsf(prevp) < qcut) prevp = 0.0f;
    v.q = rmax(v.q - prevp * dtcld, 0.0f);
    v.qr = rmax(v.qr + prevp * dtcld, 0.0f);
    v.t = v.t + R6DIV((prevp * dtcld) * xl, cpm);
}

__device__ __forceinline__ float r6_vt2r(const float *cf, const Wsm6rFold &f, float den, float qr) {
    return (cf[R_VT2R_A] * gfk_pow(den, f.e_vtr)) * gfk_pow(rmax(qr, R6_QCRMIN), f.e_qr4);
}
__device__ __forceinline__ float r6_vt2g(const float *cf, const Wsm6rFold &f, float den, float qg) {
    return (cf[R_VT2G_A] * gfk_pow(den, f.e_vtg)) * gfk_pow(rmax(qg, R6_QCRMIN), f.e_qg4);
}
__device__ __forceinline__ float r6_vt2s(const float *cf, const Wsm6rFold &f, float den, float qs,
                                         float t) {
    return ((cf[R_VT2S_A] * gfk_pow(den, f.e_vts)) * gfk_pow(rmax(qs, R6_QCRMIN), f.e_qs4))
         * gfk_exp(-R6DIV(f.c_bvts_alpha * r6_cap90(t), 4.0f));
}
__device__ __forceinline__ float r6_vt2i(const float *cf, const Wsm6rFold &f, float den, float qi) {
    return cf[R_VT2I_A] * gfk_pow(den * rmax(qi, R6_QMIN), f.e_i131);
}

// accret1 (:970-1163).  psacw/pgacw leave holding their T >= T0 parts.
__device__ __forceinline__ void r6_accret1_level(const float *cf, const Wsm6rFold &f,
                                                 Wsm6rLevel &v, float dtcld,
                                                 float &psacw, float &pgacw) {
    const float qcut = R6DIV(R6_QMIN, dtcld);
    float supcol, fsupcol, a, b, c, d, delta3, cpm, xl, xlf;
    // praci
    supcol = R6_T0C - v.t;
    fsupcol = r6_smoothif(supcol, 0.0f, false, false);
    {
        const float vt2r = r6_vt2r(cf, f, v.den, v.qr);
        const float vt2i = r6_vt2i(cf, f, v.den, v.qi);
        b = gfk_pow(v.den * rmax(v.qr, R6_QCRMIN), f.e_34) * rmax(v.qi, R6_QMIN);
        c = (gfk_pow(v.den, f.e_58) * sqrtf(rmax(v.qr, R6_QCRMIN)))
          * gfk_pow(rmax(v.qi, R6_QMIN), f.e_98);
        d = (sqrtf(v.den) * sqrtf(sqrtf(rmax(v.qr, R6_QCRMIN))))
          * gfk_pow(rmax(v.qi, R6_QMIN), f.e_54);
        const float praci1 = (cf[R_PRACI_A] * fabsf(vt2r - vt2i))
            * ((cf[R_PRACI_B] * b + cf[R_PRACI_C] * c) + cf[R_PRACI_D] * d);
        float praci = rmin(praci1, R6DIV(v.qi, dtcld));
        praci = fsupcol * praci;
        delta3 = (v.qr < 1.0e-4f) ? 1.0f : 0.0f;
        if (fabsf(praci) < qcut) praci = 0.0f;
        v.qi = rmax(v.qi - praci * dtcld, 0.0f);
        v.qs = rmax(v.qs + (praci * delta3) * dtcld, 0.0f);
        v.qg = rmax(v.qg + (praci * (1.0f - delta3)) * dtcld, 0.0f);
    }
    // piacr (supcol not changed)
    fsupcol = r6_smoothif(supcol, 0.0f, false, false);
    cpm = r6_cpm(f, v.q);
    xl = r6_xl(cf, v.t);
    xlf = R6_XLS - xl;
    if (supcol < 0.0f) xlf = R6_XLF0;
    {
        float piacr1;
        if (v.qi > 0.0f && v.qr > 0.0f)
            piacr1 = ((cf[R_PIACR_A] * gfk_pow(v.den, f.e_r3)) * gfk_pow(v.qi, 0.75f))
                   * gfk_pow(v.qr, f.e_r6);
        else
            piacr1 = 0.0f;
        float piacr = rmin(piacr1, R6DIV(v.qr, dtcld));
        piacr = fsupcol * piacr;
        delta3 = (v.qr < 1.0e-4f) ? 1.0f : 0.0f;
        if (fabsf(piacr) < qcut) piacr = 0.0f;
        v.qr = rmax(v.qr - piacr * dtcld, 0.0f);
        v.qs = rmax(v.qs + (piacr * delta3) * dtcld, 0.0f);
        v.qg = rmax(v.qg + (piacr * (1.0f - delta3)) * dtcld, 0.0f);
        v.t = v.t + R6DIV((piacr * dtcld) * xlf, cpm);
    }
    // psaci
    supcol = R6_T0C - v.t;
    fsupcol = r6_smoothif(supcol, 0.0f, false, false);
    const float eacrs = rmin(gfk_exp(0.07f * (-supcol)), 1.0f);
    {
        const float m = r6_cap90(v.t);
        const float vt2s = r6_vt2s(cf, f, v.den, v.qs, v.t);
        const float vt2i = r6_vt2i(cf, f, v.den, v.qi);
        a = gfk_exp(R6_ALPHA * m);
        b = (gfk_exp(-R6DIV(f.c_3alpha * m, 4.0f))
             * gfk_pow(v.den * rmax(v.qs, R6_QCRMIN), f.e_34)) * rmax(v.qi, R6_QMIN);
        c = ((gfk_exp(-R6DIV(R6_ALPHA * m, 2.0f)) * gfk_pow(v.den, f.e_58))
             * sqrtf(rmax(v.qs, R6_QCRMIN))) * gfk_pow(rmax(v.qi, R6_QMIN), f.e_98);
        d = ((gfk_exp(-R6DIV(R6_ALPHA * m, 4.0f)) * sqrtf(v.den))
             * sqrtf(sqrtf(rmax(v.qs, R6_QCRMIN)))) * gfk_pow(rmax(v.qi, R6_QMIN), f.e_54);
        const float psaci1 = (((cf[R_PSACI_A] * a) * fabsf(vt2s - vt2i))
            * ((cf[R_PSACI_B] * b + cf[R_PSACI_C] * c) + cf[R_PSACI_D] * d)) * eacrs;
        float psaci = rmin(psaci1, R6DIV(v.qi, dtcld));
        psaci = fsupcol * psaci;
        if (fabsf(psaci) < qcut) psaci = 0.0f;
        v.qi = rmax(v.qi - psaci * dtcld, 0.0f);
        v.qs = rmax(v.qs + psaci * dtcld, 0.0f);
    }
    // pgaci (supcol, fsupcol not changed)
    {
        const float egi = eacrs;
        const float vt2g = r6_vt2g(cf, f, v.den, v.qg);
        const float vt2i = r6_vt2i(cf, f, v.den, v.qi);
        b = gfk_pow(v.den * rmax(v.qg, R6_QCRMIN), f.e_34) * rmax(v.qi, R6_QMIN);
        c = (gfk_pow(v.den, f.e_58) * sqrtf(rmax(v.qg, R6_QCRMIN)))
          * gfk_pow(rmax(v.qi, R6_QMIN), f.e_98);
        d = (sqrtf(v.den) * sqrtf(sqrtf(rmax(v.qg, R6_QCRMIN))))
          * gfk_pow(rmax(v.qi, R6_QMIN), f.e_54);
        const float pgaci1 = ((cf[R_PGACI_A] * fabsf(vt2g - vt2i))
            * ((cf[R_PGACI_B] * b + cf[R_PGACI_C] * c) + cf[R_PGACI_D] * d)) * egi;
        float pgaci = rmin(pgaci1, R6DIV(v.qi, dtcld));
        pgaci = fsupcol * pgaci;
        if (fabsf(pgaci) < qcut) pgaci = 0.0f;
        v.qi = rmax(v.qi - pgaci * dtcld, 0.0f);
        v.qg = rmax(v.qg + pgaci * dtcld, 0.0f);
    }
    // psacw (supcol, fsupcol not changed; cpm from piacr)
    xl = r6_xl(cf, v.t);
    xlf = R6_XLS - xl;
    if (supcol < 0.0f) xlf = R6_XLF0;
    if (v.qs > 0.0f && v.qc > 0.0f) {
        a = gfk_exp(R6DIV(f.c_1mbvts_alpha * r6_cap90(v.t), 4.0f));
        psacw = (((cf[R_PSACW_A] * a) * gfk_pow(v.den, f.e_s1)) * gfk_pow(v.qs, f.e_s3)) * v.qc;
    } else {
        psacw = 0.0f;
    }
    psacw = rmax(rmin(psacw, R6DIV(v.qc, dtcld)), 0.0f);
    psacw = fsupcol * psacw;
    if (fabsf(psacw) < qcut) psacw = 0.0f;
    v.qc = rmax(v.qc - psacw * dtcld, 0.0f);
    v.qr = rmax(v.qr + ((1.0f - fsupcol) * psacw) * dtcld, 0.0f);
    v.qg = rmax(v.qg + (fsupcol * psacw) * dtcld, 0.0f);
    v.t = v.t + R6DIV(((fsupcol * psacw) * dtcld) * xlf, cpm);
    psacw = (1.0f - fsupcol) * psacw;
    // pgacw
    supcol = R6_T0C - v.t;
    fsupcol = r6_smoothif(supcol, 0.0f, false, false);
    xl = r6_xl(cf, v.t);
    xlf = R6_XLS - xl;
    if (supcol < 0.0f) xlf = R6_XLF0;
    if (v.qg > 0.0f && v.qc > 0.0f)
        pgacw = ((cf[R_PGACW_A] * gfk_pow(v.den, f.e_g1)) * gfk_pow(v.qg, f.e_g3)) * v.qc;
    else
        pgacw = 0.0f;
    pgacw = rmax(rmin(pgacw, R6DIV(v.qc, dtcld)), 0.0f);
    if (fabsf(pgacw) < qcut) pgacw = 0.0f;
    v.qc = rmax(v.qc - pgacw * dtcld, 0.0f);
    v.qr = rmax(v.qr + ((1.0f - fsupcol) * pgacw) * dtcld, 0.0f);
    v.qg = rmax(v.qg + (fsupcol * pgacw) * dtcld, 0.0f);
    v.t = v.t + R6DIV(((fsupcol * pgacw) * dtcld) * xlf, cpm);
    pgacw = (1.0f - fsupcol) * pgacw;
}

// accret2 (:1185-1387).
__device__ __forceinline__ void r6_accret2_level(const float *cf, const Wsm6rFold &f,
                                                 Wsm6rLevel &v, float dtcld,
                                                 float psacw, float pgacw) {
    const float qcut = R6DIV(R6_QMIN, dtcld);
    float supcol, fsupcol, a, b, c, d, cpm, xl, xlf, m;
    // pracs
    supcol = R6_T0C - v.t;
    fsupcol = r6_smoothif(supcol, 0.0f, false, false);
    {
        m = r6_cap90(v.t);
        const float vt2r = r6_vt2r(cf, f, v.den, v.qr);
        const float vt2s = r6_vt2s(cf, f, v.den, v.qs, v.t);
        a = gfk_exp(R6_ALPHA * m);
        b = ((gfk_exp(-R6DIV(f.c_3alpha * m, 2.0f)) * gfk_pow(v.den, f.e_34))
             * gfk_pow(rmax(v.qs, R6_QCRMIN), f.e_32)) * sqrtf(sqrtf(rmax(v.qr, R6_QCRMIN)));
        c = ((gfk_exp(-R6DIV(f.c_5alpha * m, 4.0f)) * gfk_pow(v.den, f.e_34))
             * gfk_pow(rmax(v.qs, R6_QCRMIN), f.e_54)) * sqrtf(rmax(v.qr, R6_QCRMIN));
        d = ((gfk_exp(-(R6_ALPHA * m)) * gfk_pow(v.den, f.e_34))
             * rmax(v.qs, R6_QCRMIN)) * gfk_pow(rmax(v.qr, R6_QCRMIN), f.e_34);
        const float pracs1 = ((cf[R_PRACS_A] * a) * fabsf(vt2r - vt2s))
            * ((cf[R_PRACS_B] * b + cf[R_PRACS_C] * c) + cf[R_PRACS_D] * d);
        float pracs = rmin(pracs1, R6DIV(v.qs, dtcld));
        pracs = fsupcol * pracs;
        if (fabsf(pracs) < qcut) pracs = 0.0f;
        v.qs = rmax(v.qs - pracs * dtcld, 0.0f);
        v.qg = rmax(v.qg + pracs * dtcld, 0.0f);
    }
    // psacr (supcol, fsupcol not changed)
    cpm = r6_cpm(f, v.q);
    xl = r6_xl(cf, v.t);
    xlf = R6_XLS - xl;
    if (supcol < 0.0f) xlf = R6_XLF0;
    float psacr;
    {
        m = r6_cap90(v.t);
        const float vt2r = r6_vt2r(cf, f, v.den, v.qr);
        const float vt2s = r6_vt2s(cf, f, v.den, v.qs, v.t);
        a = gfk_exp(R6_ALPHA * m);
        b = ((gfk_exp(-R6DIV(R6_ALPHA * m, 4.0f)) * gfk_pow(v.den, f.e_34))
             * gfk_pow(rmax(v.qr, R6_QCRMIN), f.e_32)) * sqrtf(sqrtf(rmax(v.qs, R6_QCRMIN)));
        c = ((gfk_exp(-R6DIV(R6_ALPHA * m, 2.0f)) * gfk_pow(v.den, f.e_34))
             * gfk_pow(rmax(v.qr, R6_QCRMIN), f.e_54)) * sqrtf(rmax(v.qs, R6_QCRMIN));
        d = ((gfk_exp(-R6DIV(f.c_3alpha * m, 4.0f)) * gfk_pow(v.den, f.e_34))
             * rmax(v.qr, R6_QCRMIN)) * gfk_pow(rmax(v.qs, R6_QCRMIN), f.e_34);
        const float psacr1 = ((cf[R_PSACR_A] * a) * fabsf(vt2r - vt2s))
            * ((cf[R_PSACR_B] * b + cf[R_PSACR_C] * c) + cf[R_PSACR_D] * d);
        if (supcol > 0.0f)
            psacr = rmin(psacr1, R6DIV(v.qr, dtcld));
        else
            psacr = rmin(psacr1, R6DIV(v.qs, dtcld));
        if (fabsf(psacr) < qcut) psacr = 0.0f;
        const float delta2 = (v.qr < 1.0e-4f && v.qs < 1.0e-4f) ? 1.0f : 0.0f;
        v.qr = rmax(v.qr - (fsupcol * psacr) * dtcld, 0.0f);
        v.qs = rmax(v.qs + ((fsupcol * delta2) * psacr) * dtcld, 0.0f);
        v.qg = rmax(v.qg + ((fsupcol * (1.0f - delta2)) * psacr) * dtcld, 0.0f);
        v.t = v.t + R6DIV(((fsupcol * psacr) * dtcld) * xlf, cpm);
        psacr = (1.0f - fsupcol) * psacr;
    }
    // pgacr
    supcol = R6_T0C - v.t;
    fsupcol = r6_smoothif(supcol, 0.0f, false, false);
    xl = r6_xl(cf, v.t);
    xlf = R6_XLS - xl;
    if (supcol < 0.0f) xlf = R6_XLF0;
    float pgacr;
    {
        const float vt2r = r6_vt2r(cf, f, v.den, v.qr);
        const float vt2g = r6_vt2g(cf, f, v.den, v.qg);
        b = (gfk_pow(v.den, f.e_34) * gfk_pow(rmax(v.qr, R6_QCRMIN), f.e_32))
          * sqrtf(sqrtf(rmax(v.qg, R6_QCRMIN)));
        c = (gfk_pow(v.den, f.e_34) * gfk_pow(rmax(v.qr, R6_QCRMIN), f.e_54))
          * sqrtf(rmax(v.qg, R6_QCRMIN));
        d = (gfk_pow(v.den, f.e_34) * rmax(v.qr, R6_QCRMIN))
          * gfk_pow(rmax(v.qg, R6_QCRMIN), f.e_34);
        const float pgacr1 = (cf[R_PGACR_A] * fabsf(vt2r - vt2g))
            * ((cf[R_PGACR_B] * b + cf[R_PGACR_C] * c) + cf[R_PGACR_D] * d);
        if (supcol > 0.0f)
            pgacr = rmin(pgacr1, R6DIV(v.qr, dtcld));
        else
            pgacr = rmin(pgacr1, R6DIV(v.qg, dtcld));
        if (fabsf(pgacr) < qcut) pgacr = 0.0f;
        v.qr = rmax(v.qr - (fsupcol * pgacr) * dtcld, 0.0f);
        v.qg = rmax(v.qg + (fsupcol * pgacr) * dtcld, 0.0f);
        v.t = v.t + R6DIV(((fsupcol * pgacr) * dtcld) * xlf, cpm);
        pgacr = (1.0f - fsupcol) * pgacr;
    }
    // pgacs
    supcol = R6_T0C - v.t;
    fsupcol = r6_smoothif(supcol, 0.0f, false, false);
    {
        const float egs = rmin(gfk_exp(-(0.09f * supcol)), 1.0f);
        m = r6_cap90(v.t);
        const float vt2g = r6_vt2g(cf, f, v.den, v.qg);
        const float vt2s = r6_vt2s(cf, f, v.den, v.qs, v.t);
        a = gfk_exp(R6_ALPHA * m);
        b = ((gfk_exp(-R6DIV(f.c_3alpha * m, 2.0f)) * gfk_pow(v.den, f.e_34))
             * gfk_pow(rmax(v.qs, R6_QCRMIN), f.e_32)) * sqrtf(sqrtf(rmax(v.qg, R6_QCRMIN)));
        c = ((gfk_exp(-R6DIV(f.c_5alpha * m, 4.0f)) * gfk_pow(v.den, f.e_34))
             * gfk_pow(rmax(v.qs, R6_QCRMIN), f.e_54)) * sqrtf(rmax(v.qg, R6_QCRMIN));
        d = ((gfk_exp(-(R6_ALPHA * m)) * gfk_pow(v.den, f.e_34))
             * rmax(v.qs, R6_QCRMIN)) * gfk_pow(rmax(v.qg, R6_QCRMIN), f.e_34);
        const float pgacs1 = (((cf[R_PGACS_A] * a) * fabsf(vt2g - vt2s))
            * ((cf[R_PGACS_B] * b + cf[R_PGACS_C] * c) + cf[R_PGACS_D] * d)) * egs;
        float pgacs = rmin(pgacs1, R6DIV(v.qs, dtcld));
        if (fabsf(pgacs) < qcut) pgacs = 0.0f;
        v.qs = rmax(v.qs - pgacs * dtcld, 0.0f);
        v.qg = rmax(v.qg + pgacs * dtcld, 0.0f);
    }
    // pseml (supcol not changed; cpm from psacr)
    xl = r6_xl(cf, v.t);
    xlf = R6_XLS - xl;
    if (supcol < 0.0f) xlf = R6_XLF0;
    {
        const float ft0 = r6_smoothif(v.t, R6_T0C, false, false);
        const float fqs = r6_smoothif(v.qs, 0.0f, true, true);
        float pseml = rmin(rmax(R6DIV((R6_CLIQ * supcol) * (psacw + psacr), xlf),
                                R6DIV(-v.qs, dtcld)), 0.0f);
        pseml = (ft0 * fqs) * pseml;
        if (fabsf(pseml) < qcut) pseml = 0.0f;
        v.qr = rmax(v.qr - pseml * dtcld, 0.0f);
        v.qs = rmax(v.qs + pseml * dtcld, 0.0f);
        v.t = v.t + R6DIV((pseml * dtcld) * xlf, cpm);
    }
    // pgeml
    supcol = R6_T0C - v.t;
    xl = r6_xl(cf, v.t);
    xlf = R6_XLS - xl;
    if (supcol < 0.0f) xlf = R6_XLF0;
    {
        const float ft0 = r6_smoothif(v.t, R6_T0C, false, false);
        const float fqg = r6_smoothif(v.qg, 0.0f, true, true);
        float pgeml = rmin(rmax(R6DIV((R6_CLIQ * supcol) * (pgacw + pgacr), xlf),
                                R6DIV(-v.qg, dtcld)), 0.0f);
        pgeml = (ft0 * fqg) * pgeml;
        if (fabsf(pgeml) < qcut) pgeml = 0.0f;
        v.qr = rmax(v.qr - pgeml * dtcld, 0.0f);
        v.qg = rmax(v.qg + pgeml * dtcld, 0.0f);
        v.t = v.t + R6DIV((pgeml * dtcld) * xlf, cpm);
    }
}

// The deposition/sublimation limiter shared by pidep, psdep and pgdep.
__device__ __forceinline__ float r6_deplimit(float rate, float qsrc, float satdt, float dtcld) {
    if (rate < 0.0f)
        return rmin(rmax(rate, R6DIV(-qsrc, dtcld)), 0.0f);
    return rmax(rmin(rate, satdt), 0.0f);
}

// accret3 (:1412-1674).
__device__ __forceinline__ void r6_accret3_level(const float *cf, const Wsm6rFold &f,
                                                 Wsm6rLevel &v, float dtcld) {
    const float qcut = R6DIV(R6_QMIN, dtcld);
    float supcol, supsat, satdt, cpm, xl, fsupcol, a, b, c, d, e;
    Wsm6rRh rh;
    // pidep
    supcol = R6_T0C - v.t;
    rh = r6_calcrh(f, v.t, v.p, v.q);
    supsat = v.q - rh.qs2;
    satdt = R6DIV(supsat, dtcld);
    xl = r6_xl(cf, v.t);
    cpm = r6_cpm(f, v.q);
    fsupcol = r6_smoothif(supcol, 0.0f, false, true);
    {
        float pidep0;
        if (v.qi > 0.0f) {
            b = r6_diffc(cf, v.den, R6_XLS, v.t);
            c = r6_diffd(cf, v.p, v.t, rh.qs2);
            a = R6DIV(rh.rh2 - 1.0f, b + c);
            pidep0 = (cf[R_PIDEP_A] * a) * gfk_pow(v.den * v.qi, f.e_78);
        } else {
            pidep0 = 0.0f;
        }
        float pidep = r6_deplimit(pidep0, v.qi, satdt, dtcld);
        pidep = fsupcol * pidep;
        if (fabsf(pidep) < qcut) pidep = 0.0f;
        v.q = rmax(v.q - pidep * dtcld, 0.0f);
        v.qi = rmax(v.qi + pidep * dtcld, 0.0f);
        v.t = v.t + R6DIV((pidep * dtcld) * R6_XLS, cpm);
    }
    // psdep
    supcol = R6_T0C - v.t;
    rh = r6_calcrh(f, v.t, v.p, v.q);
    supsat = v.q - rh.qs2;
    satdt = R6DIV(supsat, dtcld);
    xl = r6_xl(cf, v.t);
    cpm = r6_cpm(f, v.q);
    fsupcol = r6_smoothif(supcol, 0.0f, false, true);
    {
        const float m = r6_cap90(v.t);
        a = gfk_exp(R6DIV(R6_ALPHA * m, 2.0f)) * sqrtf(v.den * rmax(v.qs, R6_QCRMIN));
        b = ((((R6DIV(gfk_exp(R6DIV(f.c_3mbvts_alpha * m, 8.0f)) * gfk_pow(v.t + 120.0f, f.e_16),
                      gfk_pow(v.t, f.e_512)))
               * gfk_pow(v.p, f.e_13)) * gfk_pow(v.den, f.e_ds))
             * gfk_pow(rmax(v.qs, R6_QCRMIN), f.e_qs58));
        c = r6_diffc(cf, v.den, R6_XLS, v.t);
        d = r6_diffd(cf, v.p, v.t, rh.qs2);
        e = R6DIV(rh.rh2 - 1.0f, c + d);
        const float psdep0 = e * (cf[R_PSDEP_A] * a + cf[R_PSDEP_B] * b);
        float psdep = r6_deplimit(psdep0, v.qs, satdt, dtcld);
        psdep = fsupcol * psdep;
        if (fabsf(psdep) < qcut) psdep = 0.0f;
        v.q = rmax(v.q - psdep * dtcld, 0.0f);
        v.qs = rmax(v.qs + psdep * dtcld, 0.0f);
        v.t = v.t + R6DIV((psdep * dtcld) * R6_XLS, cpm);
    }
    // pgdep
    supcol = R6_T0C - v.t;
    rh = r6_calcrh(f, v.t, v.p, v.q);
    supsat = v.q - rh.qs2;
    satdt = R6DIV(supsat, dtcld);
    xl = r6_xl(cf, v.t);
    cpm = r6_cpm(f, v.q);
    fsupcol = r6_smoothif(supcol, 0.0f, false, true);
    {
        a = sqrtf(v.den * rmax(v.qg, R6_QCRMIN));
        b = r6_ventb(f, v.t, v.p, v.den, v.qg, f.e_dg, f.e_qg58);
        c = r6_diffc(cf, v.den, R6_XLS, v.t);
        d = r6_diffd(cf, v.p, v.t, rh.qs2);
        e = R6DIV(rh.rh2 - 1.0f, c + d);
        const float pgdep3 = e * (cf[R_PGDEP_A] * a + cf[R_PGDEP_B] * b);
        float pgdep = r6_deplimit(pgdep3, v.qg, satdt, dtcld);
        pgdep = fsupcol * pgdep;
        if (fabsf(pgdep) < qcut) pgdep = 0.0f;
        v.q = rmax(v.q - pgdep * dtcld, 0.0f);
        v.qg = rmax(v.qg + pgdep * dtcld, 0.0f);
        v.t = v.t + R6DIV((pgdep * dtcld) * R6_XLS, cpm);
    }
    // pigen
    supcol = R6_T0C - v.t;
    cpm = r6_cpm(f, v.q);
    rh = r6_calcrh(f, v.t, v.p, v.q);
    supsat = v.q - rh.qs2;
    satdt = R6DIV(supsat, dtcld);
    {
        const float fsupsat = r6_smoothif(supsat, 0.0f, true, true);
        fsupcol = r6_smoothif(supcol, 0.0f, false, true);
        const float xni0 = 1.0e3f * gfk_exp(0.1f * supcol);
        const float roqi0 = 4.92e-11f * gfk_pow(xni0, 1.33f);
        const float pigen0 = rmin(R6DIV(R6DIV(roqi0, v.den) - rmax(v.qi, 0.0f), dtcld), satdt);
        float pigen = rmax(pigen0, 0.0f);
        pigen = (fsupcol * fsupsat) * pigen;
        if (fabsf(pigen) < qcut) pigen = 0.0f;
        v.q = rmax(v.q - pigen * dtcld, 0.0f);
        v.qi = rmax(v.qi + pigen * dtcld, 0.0f);
        v.t = v.t + R6DIV((pigen * dtcld) * R6_XLS, cpm);
    }
    // psaut
    supcol = R6_T0C - v.t;
    fsupcol = r6_smoothif(supcol, 0.0f, false, true);
    {
        const float qimax = R6DIV(cf[R_ROQIMAX], v.den);
        float psaut = rmax(0.0f, R6DIV(v.qi - qimax, dtcld));
        psaut = fsupcol * psaut;
        if (fabsf(psaut) < qcut) psaut = 0.0f;
        v.qi = rmax(v.qi - psaut * dtcld, 0.0f);
        v.qs = rmax(v.qs + psaut * dtcld, 0.0f);
    }
    // pgaut (supcol, fsupcol not changed)
    {
        const float alpha2 = 1.0e-3f * gfk_exp(0.09f * (-supcol));
        float pgaut = rmin(rmax(0.0f, alpha2 * (v.qs - R6_QS0)), R6DIV(v.qs, dtcld));
        pgaut = fsupcol * pgaut;
        if (fabsf(pgaut) < qcut) pgaut = 0.0f;
        v.qs = rmax(v.qs - pgaut * dtcld, 0.0f);
        v.qg = rmax(v.qg + pgaut * dtcld, 0.0f);
    }
    // psevp (supcol not changed)
    rh = r6_calcrh(f, v.t, v.p, v.q);
    xl = r6_xl(cf, v.t);
    cpm = r6_cpm(f, v.q);
    {
        const float ft0 = r6_smoothif(v.t, R6_T0C, false, true);
        const float m = r6_cap90(v.t);
        a = gfk_exp(R6DIV(R6_ALPHA * m, 2.0f)) * sqrtf(v.den * rmax(v.qs, R6_QCRMIN));
        b = ((((R6DIV(gfk_exp(R6DIV(f.c_3mbvts_alpha * m, 8.0f)) * gfk_pow(v.t + 120.0f, f.e_16),
                      gfk_pow(v.t, f.e_512)))
               * gfk_pow(v.p, f.e_13)) * gfk_pow(v.den, f.e_ds))
             * gfk_pow(rmax(v.qs, R6_QCRMIN), f.e_qs58));
        c = r6_diffc(cf, v.den, xl, v.t);
        d = r6_diffd(cf, v.p, v.t, rh.qs1);
        e = R6DIV(rh.rh1 - 1.0f, c + d);
        const float psevp0 = e * (cf[R_PSEVP_A] * a + cf[R_PSEVP_B] * b);
        float psevp = rmin(rmax(psevp0, R6DIV(-v.qs, dtcld)), 0.0f);
        psevp = ft0 * psevp;
        if (fabsf(psevp) < qcut) psevp = 0.0f;
        v.q = rmax(v.q - psevp * dtcld, 0.0f);
        v.qs = rmax(v.qs + psevp * dtcld, 0.0f);
        v.t = v.t + R6DIV((psevp * dtcld) * R6_XLS, cpm);
    }
    // pgevp
    supcol = R6_T0C - v.t;
    rh = r6_calcrh(f, v.t, v.p, v.q);
    xl = r6_xl(cf, v.t);
    cpm = r6_cpm(f, v.q);
    {
        const float ft0 = r6_smoothif(v.t, R6_T0C, false, false);
        a = sqrtf(v.den * rmax(v.qg, R6_QCRMIN));
        b = r6_ventb(f, v.t, v.p, v.den, v.qg, f.e_dg, f.e_qg58);
        c = r6_diffc(cf, v.den, xl, v.t);
        d = r6_diffd(cf, v.p, v.t, rh.qs1);
        e = R6DIV(rh.rh1 - 1.0f, c + d);
        const float pgevp0 = e * (cf[R_PGEVP_A] * a + cf[R_PGEVP_B] * b);
        float pgevp = rmin(rmax(pgevp0, R6DIV(-v.qg, dtcld)), 0.0f);
        pgevp = ft0 * pgevp;
        if (fabsf(pgevp) < qcut) pgevp = 0.0f;
        v.q = rmax(v.q - pgevp * dtcld, 0.0f);
        v.qg = rmax(v.qg + pgevp * dtcld, 0.0f);
        v.t = v.t + R6DIV((pgevp * dtcld) * R6_XLS, cpm);
    }
    (void)supcol;
}

// pconadd (:1693-1721) with conden (:1805-1810).
__device__ __forceinline__ void r6_pconadd_level(const float *cf, const Wsm6rFold &f,
                                                 Wsm6rLevel &v, float dtcld) {
    const Wsm6rRh rh = r6_calcrh(f, v.t, v.p, v.q);
    const float xl = r6_xl(cf, v.t);
    const float cpm = r6_cpm(f, v.q);
    const float work1 = R6DIV(v.q - rh.qs1,
        1.0f + R6DIV(R6DIV(xl * xl, R6_RV * cpm) * rh.qs1, v.t * v.t));
    float pcond;
    if (work1 > 0.0f)
        pcond = R6DIV(rmin(work1, rmax(v.q, 0.0f)), dtcld);
    else
        pcond = R6DIV(rmax(work1, -v.qc), dtcld);
    if (fabsf(pcond) < R6DIV(R6_QMIN, dtcld)) pcond = 0.0f;
    v.q = rmax(v.q - pcond * dtcld, 0.0f);
    v.qc = rmax(v.qc + pcond * dtcld, 0.0f);
    v.t = v.t + R6DIV((pcond * dtcld) * xl, cpm);
}

extern "C" __global__ void wsm6r_column(
    float *theta, float *qv, float *qc, float *qi, float *qr,
    float *qs, float *qg, const float *den, const float *p,
    const float *pii, const float *dz, float *rainnc, float *rainncv,
    float *work, const float *cf, float delt, int nz, int ny, int nx)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = ny * nx;
    if (col >= stride || nz < 1) return;
    const Wsm6rFold f = wsm6r_fold();
    float *t = theta;   // the wrapper's t(i,k) lives in theta during the call
#define R6ID(k) (col + (k) * stride)
#define R6W(j, k) (col + ((j) * nz + (k)) * stride)

    // wsm6r wrapper (:129-145): t = th*pii.  wsm62D (:228-237): padding.
    for (int k = 0; k < nz; ++k) {
        const int id = R6ID(k);
        t[id] = theta[id] * pii[id];
        qv[id] = rmax(qv[id], 0.0f);
        qc[id] = rmax(qc[id], 0.0f);
        qr[id] = rmax(qr[id], 0.0f);
        qi[id] = rmax(qi[id], 0.0f);
        qs[id] = rmax(qs[id], 0.0f);
        qg[id] = rmax(qg[id], 0.0f);
    }
    // Declared divergence: per-step RAINNCV, zeroed and accumulated.
    float rain = rainnc[col];
    float rcv = 0.0f;

    // :242-244
    int loops = (int)lroundf(R6DIV(delt, R6_DTCLDCR));
    if (loops < 1) loops = 1;
    float dtcld = R6DIV(delt, (float)loops);
    if (delt <= R6_DTCLDCR) dtcld = delt;

    for (int loop = 0; loop < loops; ++loop) {
        // ---- fallk (:513-678) -------------------------------------------
        int mstep = 1;
        for (int k = nz - 1; k >= 0; --k) {
            const int id = R6ID(k);
            const float dk = den[id], zk = dz[id];
            const float w_r = R6DIV((cf[R_VT2R_A] * gfk_pow(dk, f.e_vtr))
                                    * gfk_pow(rmax(qr[id], R6_QCRMIN), f.e_qr4), zk);
            const float w_s = R6DIV((cf[R_VT2S_A] * gfk_pow(dk, f.e_vts))
                                    * gfk_pow(rmax(qs[id], R6_QCRMIN), f.e_qs4), zk)
                            * gfk_exp(-R6DIV(f.c_bvts_alpha * r6_cap90(t[id]), 4.0f));
            const float w_g = R6DIV((cf[R_VT2G_A] * gfk_pow(dk, f.e_vtg))
                                    * gfk_pow(rmax(qg[id], R6_QCRMIN), f.e_qg4), zk);
            work[R6W(0, k)] = w_r;
            work[R6W(1, k)] = w_s;
            work[R6W(2, k)] = w_g;
            float w1;
            if (w_r >= w_s && w_r >= w_g) w1 = w_r;
            else if (w_s >= w_r && w_s >= w_g) w1 = w_s;
            else w1 = w_g;
            const int nw = (int)lroundf(w1 * dtcld + 0.5f);
            const int numdt = (nw > 1) ? nw : 1;
            if (numdt >= mstep) mstep = numdt;
        }
        const float fm = (float)mstep;
        float fall_r = 0.0f, fall_s = 0.0f, fall_g = 0.0f;
        for (int n = 1; n <= mstep; ++n) {
            float fk_r = 0.0f, fk_s = 0.0f, fk_g = 0.0f;   // falk(k+1)
            for (int k = nz - 1; k >= 0; --k) {
                const int id = R6ID(k);
                const float dk = den[id];
                float q_r = qr[id], q_s = qs[id], q_g = qg[id];
                const float a_r = R6DIV((dk * q_r) * work[R6W(0, k)], fm);
                const float a_s = R6DIV((dk * q_s) * work[R6W(1, k)], fm);
                const float a_g = R6DIV((dk * q_g) * work[R6W(2, k)], fm);
                if (k == 0) {
                    fall_r = fall_r + a_r;
                    fall_s = fall_s + a_s;
                    fall_g = fall_g + a_g;
                }
                float d_r, d_s, d_g;
                if (k == nz - 1) {
                    d_r = rmin(q_r, R6DIV(a_r * dtcld, dk));
                    d_s = rmin(q_s, R6DIV(a_s * dtcld, dk));
                    d_g = rmin(q_g, R6DIV(a_g * dtcld, dk));
                } else {
                    const float zr = dz[R6ID(k + 1)];   // delz(k+1)
                    const float zk = dz[id];
                    d_r = rmin(q_r, R6DIV((a_r - R6DIV(fk_r * zr, zk)) * dtcld, dk));
                    d_s = rmin(q_s, R6DIV((a_s - R6DIV(fk_s * zr, zk)) * dtcld, dk));
                    d_g = rmin(q_g, R6DIV((a_g - R6DIV(fk_g * zr, zk)) * dtcld, dk));
                }
                if (fabsf(d_r) < R6_QMIN) d_r = 0.0f;
                q_r = q_r - d_r;
                if (fabsf(d_s) < R6_QMIN) d_s = 0.0f;
                q_s = q_s - d_s;
                if (fabsf(d_g) < R6_QMIN) d_g = 0.0f;
                q_g = q_g - d_g;
                qr[id] = q_r; qs[id] = q_s; qg[id] = q_g;
                fk_r = a_r; fk_s = a_s; fk_g = a_g;
            }
            // psmlt then pgmlt, per level (:601-675)
            for (int k = nz - 1; k >= 0; --k) {
                const int id = R6ID(k);
                const float dk = den[id], pk = p[id];
                float tk = t[id], q_r = qr[id], q_s = qs[id], q_g = qg[id];
                const float cpm = r6_cpm(f, qv[id]);
                const float xlf = R6_XLF0;
                {
                    const float m = r6_cap90(tk);
                    const float a = gfk_exp(R6DIV(R6_ALPHA * m, 2.0f));
                    const float b = gfk_exp(R6DIV((R6_ALPHA * m) * f.c_3mbvts, 8.0f));
                    const float c = R6DIV(gfk_pow(tk, 1.5f) * (R6_T0C - tk), tk + 120.0f);
                    const float d = R6DIV(gfk_pow(tk, f.e_388) * (R6_T0C - tk),
                                          gfk_pow(tk + 120.0f, f.e_56));
                    const float psmlt0 = ((cf[R_PSMLT_A] * a) * c) * sqrtf(dk * rmax(q_s, R6_QCRMIN))
                        + ((((cf[R_PSMLT_B] * b) * d) * gfk_pow(pk, f.e_13))
                           * gfk_pow(dk, f.e_ds)) * gfk_pow(rmax(q_s, R6_QCRMIN), f.e_qs58);
                    const float tmp3 = R6DIV(psmlt0 * dtcld, fm);
                    const float tmp4 = R6DIV(-q_s, fm);
                    const float tmp5 = (tmp3 > tmp4) ? tmp3 : tmp4;
                    float psmlt = (tmp5 < 0.0f) ? tmp5 : 0.0f;
                    if (fabsf(psmlt) < R6_QMIN) psmlt = 0.0f;
                    q_s = rmax(q_s + psmlt, 0.0f);
                    q_r = rmax(q_r - psmlt, 0.0f);
                    tk = tk + R6DIV(xlf, cpm) * psmlt;
                }
                {
                    const float c = R6DIV(gfk_pow(tk, 1.5f) * (R6_T0C - tk), tk + 120.0f);
                    const float d = R6DIV(gfk_pow(tk, f.e_388) * (R6_T0C - tk),
                                          gfk_pow(tk + 120.0f, f.e_56));
                    const float pgmlt0 = (cf[R_PGMLT_A] * c) * sqrtf(dk * rmax(q_g, R6_QCRMIN))
                        + (((cf[R_PGMLT_B] * d) * gfk_pow(pk, f.e_13))
                           * gfk_pow(dk, f.e_dg)) * gfk_pow(rmax(q_g, R6_QCRMIN), f.e_qg58);
                    const float tmp6 = R6DIV(pgmlt0 * dtcld, fm);
                    const float tmp7 = R6DIV(-q_g, fm);
                    const float tmp8 = (tmp6 > tmp7) ? tmp6 : tmp7;
                    float pgmlt = (tmp8 < 0.0f) ? tmp8 : 0.0f;
                    if (fabsf(pgmlt) < R6_QMIN) pgmlt = 0.0f;
                    q_g = rmax(q_g + pgmlt, 0.0f);
                    q_r = rmax(q_r - pgmlt, 0.0f);
                    tk = tk + R6DIV(xlf, cpm) * pgmlt;
                }
                t[id] = tk; qr[id] = q_r; qs[id] = q_s; qg[id] = q_g;
            }
        }

        // ---- fallkc (:683-739) ------------------------------------------
        int mstepc = 1;
        for (int k = nz - 1; k >= 0; --k) {
            const int id = R6ID(k);
            const float work1c = cf[R_VT2I_A] * gfk_pow(den[id] * qi[id], f.e_i131);
            const float work2c = R6DIV(work1c, dz[id]);
            int numdt = (int)lroundf(work2c * dtcld + 0.5f);
            if (numdt < 1) numdt = 1;
            if (numdt >= mstepc) mstepc = numdt;
        }
        const float fmc = (float)mstepc;
        float fall_i = 0.0f;
        for (int n = 1; n <= mstepc; ++n) {
            float fk_i = 0.0f;
            for (int k = nz - 1; k >= 0; --k) {
                const int id = R6ID(k);
                const float dk = den[id], zk = dz[id];
                float q_i = qi[id];
                const float a_i = R6DIV(R6DIV(cf[R_FALLI_A] * gfk_pow(dk * q_i, f.e_i931), zk), fmc);
                if (k == 0) fall_i = fall_i + a_i;
                float d_i;
                if (k == nz - 1)
                    d_i = rmin(q_i, R6DIV(a_i * dtcld, dk));
                else
                    d_i = rmin(q_i, R6DIV((a_i - R6DIV(fk_i * dz[R6ID(k + 1)], zk)) * dtcld, dk));
                if (fabsf(d_i) < R6_QMIN) d_i = 0.0f;
                qi[id] = q_i - d_i;
                fk_i = a_i;
            }
        }

        // ---- rainsc surface sum (:757-764) --------------------------------
        {
            const float fallsum = ((fall_r + fall_s) + fall_g) + fall_i;
            if (fallsum > R6_QMIN) {
                const float amt = ((R6DIV(fallsum * dz[R6ID(0)], R6_DENR)) * dtcld) * 1000.0f;
                rcv = rcv + amt;
                rain = amt + rain;
            }
        }

        // ---- per-level processes: rainsc, warmr, accret1-3, pconadd -------
        for (int k = 0; k < nz; ++k) {
            const int id = R6ID(k);
            Wsm6rLevel v;
            v.t = t[id]; v.q = qv[id]; v.qc = qc[id]; v.qi = qi[id];
            v.qr = qr[id]; v.qs = qs[id]; v.qg = qg[id];
            v.den = den[id]; v.p = p[id];
            // rainsc keeps fallk's cpm(i,k) = cpmcal(q): q is unchanged since.
            r6_rainsc_level(cf, f, v, r6_cpm(f, v.q), dtcld);
            r6_warmr_level(cf, f, v, dtcld);
            float psacw = 0.0f, pgacw = 0.0f;
            r6_accret1_level(cf, f, v, dtcld, psacw, pgacw);
            r6_accret2_level(cf, f, v, dtcld, psacw, pgacw);
            r6_accret3_level(cf, f, v, dtcld);
            r6_pconadd_level(cf, f, v, dtcld);
            t[id] = v.t; qv[id] = v.q; qc[id] = v.qc; qi[id] = v.qi;
            qr[id] = v.qr; qs[id] = v.qs; qg[id] = v.qg;
        }
    }

    // wsm6r wrapper (:156-167): th = t/pii.
    for (int k = 0; k < nz; ++k) {
        const int id = R6ID(k);
        theta[id] = R6DIV(t[id], pii[id]);
    }
    rainnc[col] = rain;
    rainncv[col] = rcv;
#undef R6ID
#undef R6W
}

// RADIATION RADII (a declared gpuwm addition).  WRF's wsm6r computes no
// effective radius: WSM6RSCHEME is not in module_physics_init.F's use_mp_re
// disjunction (:1004-1023), so WRF's RRTMG computes its own radii, which is
// what gpuwm's legacy RRTMG arm does for mp=106 too.  RTE+RRTMGP has no WRF
// referent and reads scheme radii, so for it gpuwm diagnoses WSM6's radii
// from the WSM6R state: phys/physics_mmm/mp_wsm6_effectRad.F90
// mp_wsm6_effectRad_run, word for word, at t = th*pii as module_mp_wsm6.F
// :215-226 calls it.  The two schemes share every particle-size PARAMETER
// the routine reads (xncr = nc0 = 3e8, n0s = 2e6, n0smax = 1e11,
// alpha = 0.12, the 5.38e7*(rho*qi)**0.75 ice number and 11.9 diameter
// law), so these are the radii of WSM6R's own size distributions.  The
// routine's tile-wide has_qc/has_qi/has_qs flags only skip loops whose
// every cell would CYCLE anyway, so a column evaluates the same words.
// Graded against WRF's own routine by the same column oracle.  Radii are
// stored in gpuwm's micron convention: WRF's metre word times 1.0e6f.
// pidnc and pidn0s are mp_wsm6_init's words (WSM6R_RADII_F32).
extern "C" __global__ void wsm6r_effective_radii(
    const float *theta, const float *pii, const float *qc, const float *qi,
    const float *qs, const float *den, float *effc, float *effi, float *effs,
    float pidnc, float pidn0s, int nz, int ny, int nx)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = ny * nx;
    if (col >= stride) return;
    // module_model_constants.F:62-67 (re_q*_bg, re_q*_max); mp_wsm6.F90
    // :41-43 (n0smax, n0s, alpha) and the routine's own PARAMETERs.
    const float bg_c = 2.49e-6f, bg_i = 4.99e-6f, bg_s = 9.99e-6f;
    const float mx_c = 50.0e-6f, mx_i = 125.0e-6f, mx_s = 999.0e-6f;
    const float r1 = 1.0e-12f, r2 = 1.0e-6f, nc0 = 3.0e8f;
    const float obmr = R6DIV(1.0f, 3.0f);                 // 1.0/bm_r
    const float n0ratio = R6DIV(1.0e11f, 2.0e6f);         // n0smax/n0s
    const float c_ice = 0.75f * 0.163f;
    for (int k = 0; k < nz; ++k) {
        const int id = col + k * stride;
        const float rho = den[id];
        const float t = theta[id] * pii[id];
        float re_c = bg_c, re_i = bg_i, re_s = bg_s;
        const float rqc = rmax(qc[id] * rho, r1);
        const float rqi = rmax(qi[id] * rho, r1);
        float temp = rho * rmax(qi[id], R6_QMIN);
        temp = sqrtf(sqrtf((temp * temp) * temp));
        const float ni = rmin(rmax(5.38e7f * temp, 1.0e3f), 1.0e6f);
        const float rni = rmax(ni * rho, r2);
        const float rqs = rmax(qs[id] * rho, r1);
        if (!(rqc <= r1)) {
            const float lamdac = gfk_pow(R6DIV(pidnc * nc0, rqc), obmr);
            re_c = rmax(rmin(1.5f * R6DIV(1.0f, lamdac), mx_c), 2.51e-6f);
        }
        if (!(rqi <= r1 || rni <= r2)) {
            const float diai = 11.9f * sqrtf(R6DIV(rqi, ni));
            re_i = rmax(rmin(c_ice * diai, mx_i), 10.01e-6f);
        }
        if (!(rqs <= r1)) {
            const float supcol = R6_T0C - t;
            const float n0sfac = rmax(rmin(gfk_exp(R6_ALPHA * supcol), n0ratio), 1.0f);
            const float lamdas = sqrtf(sqrtf(R6DIV(pidn0s * n0sfac, rqs)));
            re_s = rmax(rmin(0.5f * R6DIV(1.0f, lamdas), mx_s), 25.0e-6f);
        }
        re_c = rmax(rmin(re_c, mx_c), bg_c);
        re_i = rmax(rmin(re_i, mx_i), bg_i);
        re_s = rmax(rmin(re_s, mx_s), bg_s);
        effc[id] = re_c * 1.0e6f;
        effi[id] = re_i * 1.0e6f;
        effs[id] = re_s * 1.0e6f;
    }
}
