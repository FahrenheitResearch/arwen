// Legacy RRTMG SW (WRF v4.6.1, ra_sw_physics = 4) - CUDA twin of the
// NumPy FP32 reference in gpuwm/core/rrtmg_sw.py, held at max_ulp 0
// against the same Fortran fixtures.
//
// Numerics discipline (follows the Noah-MP lanes of this project):
//   * every FP32 operation goes through __fadd_rn/__fsub_rn/__fmul_rn/
//     __fdiv_rn (and __fsqrt_rn), so neither NVRTC's default --fmad=true
//     nor CuPy's appended -ftz can change a bit;
//   * FP64 intermediates inside the glibc transcriptions go through
//     __dadd_rn/__dsub_rn/__dmul_rn for the same reason;
//   * double -> float conversions in the libm use rsw_d2f_rn, because
//     __double2float_rn flushes subnormal results on this toolchain
//     (see tests/test_noahmp_kernel_subnormals.py on the Noah-MP lane);
//   * the glibc 2.39 logf/expf transcriptions below are the same audited
//     tables as gpuwm/core/rrtmg_sw.py SECTION 1 / noahmp_leaves.cu,
//     carried under the rsw_ prefix so this unit never collides with the
//     Noah-MP translation units.
//
// The host driver (gpuwm/core/rrtmg_sw.py, SECTION 10) compiles this file
// with a generated preamble of #defines: exact-bit scalar constants
// (RSW_C_* via __uint_as_float) and offsets into one packed FP32 table
// buffer (RSW_T_*), both derived from the verified SWTables.
//
// Batched multi-column path (SECTION 11 of the host driver): the kernels
// in this unit were written single-column (no ncol axis), so each one is
// factored into a __device__ body (all the arithmetic, verbatim) plus two
// thin __global__ wrappers: the original single-column entry point with
// its original thread mapping, and an rsw_*_b twin that adds a leading
// ncol, maps thread -> (col, work-item), and offsets the array pointers to
// the column base before calling the SAME body.  Per-column scalars
// (laytrop, prmu0) become per-column arrays indexed by col in the _b
// wrappers.  This is pure re-indexing: integer pointer arithmetic and
// launch guards only; no FP statement moved, added, or reordered, so a
// thread computing column i produces bitwise the results of the
// single-column launch on column i (proved by the batched gates in
// tests/test_rrtmg_sw_cuda.py at any chunk size and batch width).
// In particular the bitwise-sensitive SEQUENTIAL reductions are
// untouched: the g-point flux accumulation (rsw_spc_accum_body's
// ordered iw loop) and the ordered vrtqdr/reftra vertical sweeps stay
// per-thread and in original order -- batching parallelizes COLUMNS
// (and, as before, levels/g-points as independent work items), never a
// reduction.

// sm_120 (GeForce Blackwell) flushes FP32 subnormals in EVERY arithmetic
// instruction - even __fadd_rn(x, 0.0f) and cvt.f64.f32 DAZ their inputs -
// while gfortran on x86-64 (MXCSR FTZ/DAZ clear) keeps full IEEE subnormal
// semantics, and this chain really produces them (exp_tbl floors at 1e-20,
// so two clamped layers give a 1e-40 transmittance product).  FP32 add/sub/
// mul are therefore emulated through exact FP64: the product of two
// binary32 values is exact in binary64 (48 <= 53 bits), and for sums the
// double result converts to the same binary32 as single rounding would for
// every operand pair (when the exact sum does not fit binary64 the operand
// magnitudes differ by > 2^53, far beyond the 2^25 half-ulp threshold, so
// both roundings return the larger operand).  rsw_f2d decodes subnormal
// inputs from bits (cvt DAZes them); rsw_d2f_rn encodes subnormal results
// (from the Noah-MP lane; __double2float_rn flushes them).  Division and
// sqrt keep the hardware correctly-rounded instructions: no division in
// this chain sees or produces a subnormal (inputs are floored by exp_tbl's
// 1e-20 clamp or normal-scale physics), and FP64 emulation of division
// would introduce a genuine double-rounding hazard.  Equality-with-zero
// tests on flush-sensitive quantities use bitwise EQ0.
//
// INVARIANT (SW-audit item, load-bearing for the raw div/sqrt choice):
// because division and sqrt stay on the hardware __fdiv_rn/__fsqrt_rn,
// every operand reaching them MUST be non-subnormal.  The two audited
// exposure sites are the zgcc asymmetry-combination division in the
// spcvmc prep (a subnormal AEROSOL numerator could reach it) and the
// reftra discriminant's __fsqrt_rn.  The host entries close the only
// known route by REJECTING aer_opt != 0 (the composition builds
// taua/ssaa/asma = 0/1/0), and the exp_tbl 1e-20 floor plus
// normal-scale physics bound the remaining inputs.  Any future change
// that lets caller-supplied aerosol optics, or any new quantity, reach
// a raw division or sqrt must either prove the non-subnormal range or
// move that site onto the FP64-emulation helpers.  (No in-kernel debug
// assert is shipped: editing this translation unit would force a full
// dual-run re-gate of every SW CUDA gate, and the route is closed
// fail-closed at the API instead.)

typedef float real;

__device__ float rsw_d2f_rn(double y);

__device__ double rsw_f2d(float x)
{
    unsigned int ix = __float_as_uint(x);
    if (((ix >> 23) & 0xffu) == 0u) {     // zero or subnormal
        double v = (double)(ix & 0x7fffffu) * 0x1p-149;
        return (ix & 0x80000000u) ? -v : v;
    }
    return (double)x;                     // normal / inf / nan
}

__device__ float rsw_d2f_rn(double y)
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

__device__ real rsw_add(real a, real b)
{ return rsw_d2f_rn(__dadd_rn(rsw_f2d(a), rsw_f2d(b))); }
__device__ real rsw_sub(real a, real b)
{ return rsw_d2f_rn(__dsub_rn(rsw_f2d(a), rsw_f2d(b))); }
__device__ real rsw_mul(real a, real b)
{ return rsw_d2f_rn(__dmul_rn(rsw_f2d(a), rsw_f2d(b))); }

#define AD(a, b) rsw_add((a), (b))
#define SU(a, b) rsw_sub((a), (b))
#define MU(a, b) rsw_mul((a), (b))
#define DV(a, b) __fdiv_rn((a), (b))
#define DAD(a, b) __dadd_rn((a), (b))
#define DSU(a, b) __dsub_rn((a), (b))
#define DMU(a, b) __dmul_rn((a), (b))
#define EQ0(x) ((__float_as_uint(x) & 0x7fffffffu) == 0u)

#define RSW_NGPT 112
#define RSW_NBND 14
#define RSW_MXMOL 38
#define RSW_NMOL 7

// ---------------------------------------------------------------------------
// glibc 2.39 FP32 libm subset (logf, expf), rsw_ prefix.
// ---------------------------------------------------------------------------

__device__ const double RSW_LOGF_INVC[16] = {
    0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0,
    0x1.3c995b0b80385p+0, 0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0,
    0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0, 0x1.0953f419900a7p+0,
    0x1p+0,               0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
    0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1,
    0x1.767dcf5534862p-1 };
__device__ const double RSW_LOGF_LOGC[16] = {
    -0x1.57bf7808caadep-2, -0x1.2bef0a7c06ddbp-2, -0x1.01eae7f513a67p-2,
    -0x1.b31d8a68224e9p-3, -0x1.6574f0ac07758p-3, -0x1.1aa2bc79c81p-3,
    -0x1.a4e76ce8c0e5ep-4, -0x1.1973c5a611cccp-4, -0x1.252f438e10c1ep-5,
     0x0p+0,                0x1.aa5aa5df25984p-5,  0x1.c5e53aa362eb4p-4,
     0x1.526e57720db08p-3,  0x1.bc2860d22477p-3,   0x1.1058bc8a07ee1p-2,
     0x1.4043057b6ee09p-2 };
#define RSW_LOGF_LN2 0x1.62e42fefa39efp-1
#define RSW_LOGF_A0 (-0x1.00ea348b88334p-2)
#define RSW_LOGF_A1 (0x1.5575b0be00b6ap-2)
#define RSW_LOGF_A2 (-0x1.ffffef20a4123p-2)

__device__ const unsigned long long RSW_EXP2F_TAB[32] = {
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
#define RSW_EXP2F_P0 0x1.c6af84b912394p-5
#define RSW_EXP2F_P1 0x1.ebfce50fac4f3p-3
#define RSW_EXP2F_P2 0x1.62e42ff0c52d6p-1
#define RSW_EXP2F_SHIFT 0x1.8p+52

__device__ real rsw_log(real x)
{
    unsigned int ix = __float_as_uint(x);
    if (ix == 0x3f800000u) return 0.0f;
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u) {
        if (ix * 2u == 0u) return __int_as_float(0xff800000);
        if (ix == 0x7f800000u) return x;
        if ((ix & 0x80000000u) || ix * 2u >= 0xff000000u)
            return __int_as_float(0x7fc00000);
        ix = __float_as_uint(MU(x, 8388608.0f));   /* 0x1p23f */
        ix -= 23u << 23;
    }
    unsigned int tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    int k = (int)tmp >> 23;
    unsigned int iz = ix - (tmp & 0xff800000u);
    double z = (double)__uint_as_float(iz);
    double r = DSU(DMU(z, RSW_LOGF_INVC[i]), 1.0);
    double y0 = DAD(RSW_LOGF_LOGC[i], DMU((double)k, RSW_LOGF_LN2));
    double r2 = DMU(r, r);
    double y = DAD(DMU(RSW_LOGF_A1, r), RSW_LOGF_A2);
    y = DAD(DMU(RSW_LOGF_A0, r2), y);
    y = DAD(DMU(y, r2), DAD(y0, r));
    return rsw_d2f_rn(y);
}

__device__ double rsw_exp2_core(double xd)
{
    double kd = DAD(xd, RSW_EXP2F_SHIFT);
    unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd = DSU(kd, RSW_EXP2F_SHIFT);
    double r = DSU(xd, kd);
    unsigned long long t = RSW_EXP2F_TAB[ki & 31ULL];
    t += ki << (52 - 5);
    double s = __longlong_as_double((long long)t);
    double z = DAD(DMU(RSW_EXP2F_P0 / 32.0 / 32.0 / 32.0, r),
                   RSW_EXP2F_P1 / 32.0 / 32.0);
    double r2 = DMU(r, r);
    double y = DAD(DMU(RSW_EXP2F_P2 / 32.0, r), 1.0);
    y = DAD(DMU(z, r2), y);
    return DMU(y, s);
}

__device__ real rsw_exp(real x)
{
    unsigned int abstop = (__float_as_uint(x) >> 20) & 0x7ffu;
    if (abstop >= ((__float_as_uint(88.0f)) >> 20)) {
        if (__float_as_uint(x) == 0xff800000u) return 0.0f;
        if (abstop >= (0x7f800000u >> 20)) return AD(x, x);
        if (x > __int_as_float(0x42b17218)) return __int_as_float(0x7f800000);
        if (x < -__int_as_float(0x42cff1b4)) return 0.0f;
    }
    double xd = (double)x;
    double z = DMU(0x1.71547652b82fep+0 * 32.0, xd);
    return rsw_d2f_rn(rsw_exp2_core(z));
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

// The exp_tbl lookup idiom of reftra_sw / spcvmc_sw.
__device__ real rsw_etbl(const real* __restrict__ exp_tbl, real ze1)
{
    if (ze1 <= RSW_C_OD_LO)
        return AD(SU(1.0f, ze1), MU(MU(0.5f, ze1), ze1));
    real tblind = DV(ze1, AD(RSW_C_BPADE, ze1));
    int itind = (int)AD(MU(RSW_C_TBLINT, tblind), 0.5f);
    return exp_tbl[itind];
}

// ---------------------------------------------------------------------------
// setcoef_sw: one thread per layer.  laytrop/laylow flags summed on host.
// ---------------------------------------------------------------------------

__device__ void rsw_setcoef_body(int lay, int nlayers,
                 const real* __restrict__ tab,   // packed table buffer
                 const real* __restrict__ pavel,
                 const real* __restrict__ tavel,
                 const real* __restrict__ coldry,
                 const real* __restrict__ wkl,   // (MXMOL, nlayers) F-order
                 int* jp, int* jt, int* jt1, int* indself, int* indfor,
                 int* trop_flag, int* low_flag,
                 real* colh2o, real* colco2, real* colo3, real* coln2o,
                 real* colch4, real* colo2, real* colmol, real* co2mult,
                 real* selffac, real* selffrac, real* forfac, real* forfrac,
                 real* fac00, real* fac01, real* fac10, real* fac11)
{
    const real* preflog = tab + RSW_T_REF_PREFLOG;
    const real* tref = tab + RSW_T_REF_TREF;
    const real stpfac = DV(296.0f, 1013.0f);

    real plog = rsw_log(pavel[lay]);
    int jpv = (int)SU(36.0f, MU(5.0f, AD(plog, 0.04f)));
    if (jpv < 1) jpv = 1; else if (jpv > 58) jpv = 58;
    jp[lay] = jpv;
    int jp1 = jpv + 1;
    real fp = MU(5.0f, SU(preflog[jpv - 1], plog));

    int jtv = (int)AD(3.0f, DV(SU(tavel[lay], tref[jpv - 1]), 15.0f));
    if (jtv < 1) jtv = 1; else if (jtv > 4) jtv = 4;
    jt[lay] = jtv;
    real ft = SU(DV(SU(tavel[lay], tref[jpv - 1]), 15.0f),
                 (real)(jtv - 3));
    int jt1v = (int)AD(3.0f, DV(SU(tavel[lay], tref[jp1 - 1]), 15.0f));
    if (jt1v < 1) jt1v = 1; else if (jt1v > 4) jt1v = 4;
    jt1[lay] = jt1v;
    real ft1 = SU(DV(SU(tavel[lay], tref[jp1 - 1]), 15.0f),
                  (real)(jt1v - 3));

    real water = DV(wkl[0 + RSW_MXMOL * lay], coldry[lay]);
    real scalefac = DV(MU(pavel[lay], stpfac), tavel[lay]);

    bool trop = !(plog <= 4.56f);
    trop_flag[lay] = trop ? 1 : 0;
    low_flag[lay] = (trop && plog >= 6.62f) ? 1 : 0;

    if (!trop) {
        forfac[lay] = DV(scalefac, AD(1.0f, water));
        real factor = DV(SU(tavel[lay], 188.0f), 36.0f);
        indfor[lay] = 3;
        forfrac[lay] = SU(factor, 1.0f);
        selffac[lay] = 0.0f;
        selffrac[lay] = 0.0f;
        indself[lay] = 0;
    } else {
        forfac[lay] = DV(scalefac, AD(1.0f, water));
        real factor = DV(SU(332.0f, tavel[lay]), 36.0f);
        int idf = (int)factor;
        idf = idf < 1 ? 1 : idf; idf = idf > 2 ? 2 : idf;
        indfor[lay] = idf;
        forfrac[lay] = SU(factor, (real)idf);

        selffac[lay] = MU(water, forfac[lay]);
        factor = DV(SU(tavel[lay], 188.0f), 7.2f);
        int ids = (int)factor - 7;
        ids = ids < 1 ? 1 : ids; ids = ids > 9 ? 9 : ids;
        indself[lay] = ids;
        selffrac[lay] = SU(factor, (real)(ids + 7));
    }

    colh2o[lay] = MU(1.0e-20f, wkl[0 + RSW_MXMOL * lay]);
    colco2[lay] = MU(1.0e-20f, wkl[1 + RSW_MXMOL * lay]);
    colo3[lay] = MU(1.0e-20f, wkl[2 + RSW_MXMOL * lay]);
    coln2o[lay] = MU(1.0e-20f, wkl[3 + RSW_MXMOL * lay]);
    colch4[lay] = MU(1.0e-20f, wkl[5 + RSW_MXMOL * lay]);
    colo2[lay] = MU(1.0e-20f, wkl[6 + RSW_MXMOL * lay]);
    colmol[lay] = AD(MU(1.0e-20f, coldry[lay]), colh2o[lay]);
    if (EQ0(colco2[lay])) colco2[lay] = MU(1.0e-32f, coldry[lay]);
    if (EQ0(coln2o[lay])) coln2o[lay] = MU(1.0e-32f, coldry[lay]);
    if (EQ0(colch4[lay])) colch4[lay] = MU(1.0e-32f, coldry[lay]);
    if (EQ0(colo2[lay])) colo2[lay] = MU(1.0e-32f, coldry[lay]);
    real co2reg = MU(3.55e-24f, coldry[lay]);
    co2mult[lay] = DV(MU(MU(SU(colco2[lay], co2reg), 272.63f),
                         rsw_exp(DV(-1919.4f, tavel[lay]))),
                      MU(8.7604e-4f, tavel[lay]));

    real compfp = SU(1.0f, fp);
    fac10[lay] = MU(compfp, ft);
    fac00[lay] = MU(compfp, SU(1.0f, ft));
    fac11[lay] = MU(fp, ft1);
    fac01[lay] = MU(fp, SU(1.0f, ft1));
}

extern "C" __global__
void rsw_setcoef(int nlayers,
                 const real* __restrict__ tab,
                 const real* __restrict__ pavel,
                 const real* __restrict__ tavel,
                 const real* __restrict__ coldry,
                 const real* __restrict__ wkl,
                 int* jp, int* jt, int* jt1, int* indself, int* indfor,
                 int* trop_flag, int* low_flag,
                 real* colh2o, real* colco2, real* colo3, real* coln2o,
                 real* colch4, real* colo2, real* colmol, real* co2mult,
                 real* selffac, real* selffrac, real* forfac, real* forfrac,
                 real* fac00, real* fac01, real* fac10, real* fac11)
{
    int lay = blockIdx.x * blockDim.x + threadIdx.x;
    if (lay >= nlayers) return;
    rsw_setcoef_body(lay, nlayers, tab, pavel, tavel, coldry, wkl,
                     jp, jt, jt1, indself, indfor, trop_flag, low_flag,
                     colh2o, colco2, colo3, coln2o, colch4, colo2, colmol,
                     co2mult, selffac, selffrac, forfac, forfrac,
                     fac00, fac01, fac10, fac11);
}

// Batched twin: thread -> (col, lay); every array pointer moved to the
// column base (pure integer re-indexing; the body is byte-identical).
extern "C" __global__
void rsw_setcoef_b(int ncol, int nlayers,
                   const real* __restrict__ tab,
                   const real* __restrict__ pavel,     // (ncol, nlayers)
                   const real* __restrict__ tavel,
                   const real* __restrict__ coldry,
                   const real* __restrict__ wkl,       // (ncol, nl, MXMOL)
                   int* jp, int* jt, int* jt1, int* indself, int* indfor,
                   int* trop_flag, int* low_flag,
                   real* colh2o, real* colco2, real* colo3, real* coln2o,
                   real* colch4, real* colo2, real* colmol, real* co2mult,
                   real* selffac, real* selffrac, real* forfac,
                   real* forfrac,
                   real* fac00, real* fac01, real* fac10, real* fac11)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * nlayers) return;
    int col = (int)(idx / nlayers);
    int lay = (int)(idx % nlayers);
    size_t o = (size_t)col * nlayers;
    size_t om = (size_t)col * RSW_MXMOL * nlayers;
    rsw_setcoef_body(lay, nlayers, tab, pavel + o, tavel + o, coldry + o,
                     wkl + om, jp + o, jt + o, jt1 + o, indself + o,
                     indfor + o, trop_flag + o, low_flag + o,
                     colh2o + o, colco2 + o, colo3 + o, coln2o + o,
                     colch4 + o, colo2 + o, colmol + o, co2mult + o,
                     selffac + o, selffrac + o, forfac + o, forfrac + o,
                     fac00 + o, fac01 + o, fac10 + o, fac11 + o);
}

// ---------------------------------------------------------------------------
// taumol_sw: one thread per (lay, iw).  laysolfr handled by rsw_sfluxzen.
// ---------------------------------------------------------------------------

struct RswSpec {
    real speccomb, fs;
    int js;
    real f000, f010, f100, f110, f001, f011, f101, f111;
};

__device__ RswSpec rsw_spec(real colA, real colB, real strrat, real mult,
                            real fac00, real fac01, real fac10, real fac11)
{
    RswSpec s;
    s.speccomb = AD(colA, MU(strrat, colB));
    real specparm = DV(colA, s.speccomb);
    if (specparm >= RSW_C_ONEMINUS) specparm = RSW_C_ONEMINUS;
    real specmult = MU(mult, specparm);
    s.js = 1 + (int)specmult;
    s.fs = SU(specmult, truncf(specmult));
    s.f000 = MU(SU(1.0f, s.fs), fac00);
    s.f010 = MU(SU(1.0f, s.fs), fac10);
    s.f100 = MU(s.fs, fac00);
    s.f110 = MU(s.fs, fac10);
    s.f001 = MU(SU(1.0f, s.fs), fac01);
    s.f011 = MU(SU(1.0f, s.fs), fac11);
    s.f101 = MU(s.fs, fac01);
    s.f111 = MU(s.fs, fac11);
    return s;
}

// Fortran left-to-right 8-term interpolation sum; a is column ig of absa/absb
// (stride = rows in the table), ind0/ind1 are Fortran row indices.
__device__ real rsw_acc8(const real* __restrict__ a, int rows, int ig,
                         int ind0, int ind1, int dind, const RswSpec& s)
{
    const real* c = a + (size_t)rows * ig;
    real acc = MU(s.f000, c[ind0 - 1]);
    acc = AD(acc, MU(s.f100, c[ind0]));
    acc = AD(acc, MU(s.f010, c[ind0 - 1 + dind]));
    acc = AD(acc, MU(s.f110, c[ind0 + dind]));
    acc = AD(acc, MU(s.f001, c[ind1 - 1]));
    acc = AD(acc, MU(s.f101, c[ind1]));
    acc = AD(acc, MU(s.f011, c[ind1 - 1 + dind]));
    acc = AD(acc, MU(s.f111, c[ind1 + dind]));
    return acc;
}

__device__ real rsw_acc4(const real* __restrict__ a, int rows, int ig,
                         int ind0, int ind1,
                         real f00, real f10, real f01, real f11)
{
    const real* c = a + (size_t)rows * ig;
    real acc = MU(f00, c[ind0 - 1]);
    acc = AD(acc, MU(f10, c[ind0]));
    acc = AD(acc, MU(f01, c[ind1 - 1]));
    acc = AD(acc, MU(f11, c[ind1]));
    return acc;
}

// colh2o*(selffac*(selfref...) + forfac*(forref...)); tables are (rows, ng)
__device__ real rsw_selffor(const real* __restrict__ selfref, int srows,
                            const real* __restrict__ forref, int frows,
                            int ig, real colh2o, real selffac, real selffrac,
                            int inds, real forfac, real forfrac, int indf)
{
    const real* sc = selfref + (size_t)srows * ig;
    const real* fc = forref + (size_t)frows * ig;
    real sr = AD(sc[inds - 1], MU(selffrac, SU(sc[inds], sc[inds - 1])));
    real fr = AD(fc[indf - 1], MU(forfrac, SU(fc[indf], fc[indf - 1])));
    return MU(colh2o, AD(MU(selffac, sr), MU(forfac, fr)));
}

// colh2o * forfac * (forref...) with WRF's ((colh2o*forfac)*fr) association
__device__ real rsw_forr(const real* __restrict__ forref, int frows, int ig,
                         real colh2o, real forfac, real forfrac, int indf)
{
    const real* fc = forref + (size_t)frows * ig;
    real fr = AD(fc[indf - 1], MU(forfrac, SU(fc[indf], fc[indf - 1])));
    return MU(MU(colh2o, forfac), fr);
}

__device__ void rsw_taumol_body(int L, int iw, int nlayers, int laytrop,
                const real* __restrict__ tab,
                const int* __restrict__ ngb,       // 112, Fortran band ids
                const int* __restrict__ jp, const int* __restrict__ jt,
                const int* __restrict__ jt1,
                const real* __restrict__ colh2o, const real* __restrict__ colco2,
                const real* __restrict__ colch4, const real* __restrict__ colo2,
                const real* __restrict__ colo3, const real* __restrict__ colmol,
                const real* __restrict__ fac00, const real* __restrict__ fac01,
                const real* __restrict__ fac10, const real* __restrict__ fac11,
                const real* __restrict__ selffac, const real* __restrict__ selffrac,
                const int* __restrict__ indself,
                const real* __restrict__ forfac, const real* __restrict__ forfrac,
                const int* __restrict__ indfor,
                real* taug, real* taur)   // (nlayers, 112) F-order
{
    int jb = ngb[iw];               // 16..29
    int lay = L + 1;
    bool lower = (lay <= laytrop);
    size_t out = (size_t)L + (size_t)nlayers * iw;

    real tg = 0.0f, tr = 0.0f;

    switch (jb) {
    case 16: {
        int ig = iw - 0;
        tr = MU(colmol[L], RSW_C_KG16_RAYL);
        if (lower) {
            RswSpec s = rsw_spec(colh2o[L], colch4[L], RSW_C_KG16_STRRAT1,
                                 8.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 9 + s.js;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 9 + s.js;
            tg = AD(MU(s.speccomb,
                       rsw_acc8(tab + RSW_T_KG16_ABSA, 585, ig, ind0, ind1, 9, s)),
                    rsw_selffor(tab + RSW_T_KG16_SELFREF, 10,
                                tab + RSW_T_KG16_FORREF, 3, ig, colh2o[L],
                                selffac[L], selffrac[L], indself[L],
                                forfac[L], forfrac[L], indfor[L]));
        } else {
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = MU(colch4[L], rsw_acc4(tab + RSW_T_KG16_ABSB, 235, ig,
                                        ind0, ind1, fac00[L], fac10[L],
                                        fac01[L], fac11[L]));
        }
        break;
    }
    case 17: {
        int ig = iw - 6;
        tr = MU(colmol[L], RSW_C_KG17_RAYL);
        if (lower) {
            RswSpec s = rsw_spec(colh2o[L], colco2[L], RSW_C_KG17_STRRAT,
                                 8.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 9 + s.js;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 9 + s.js;
            tg = AD(MU(s.speccomb,
                       rsw_acc8(tab + RSW_T_KG17_ABSA, 585, ig, ind0, ind1, 9, s)),
                    rsw_selffor(tab + RSW_T_KG17_SELFREF, 10,
                                tab + RSW_T_KG17_FORREF, 4, ig, colh2o[L],
                                selffac[L], selffrac[L], indself[L],
                                forfac[L], forfrac[L], indfor[L]));
        } else {
            RswSpec s = rsw_spec(colh2o[L], colco2[L], RSW_C_KG17_STRRAT,
                                 4.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 5 + s.js;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 5 + s.js;
            tg = AD(MU(s.speccomb,
                       rsw_acc8(tab + RSW_T_KG17_ABSB, 1175, ig, ind0, ind1, 5, s)),
                    rsw_forr(tab + RSW_T_KG17_FORREF, 4, ig, colh2o[L],
                             forfac[L], forfrac[L], indfor[L]));
        }
        break;
    }
    case 18: {
        int ig = iw - 18;
        tr = MU(colmol[L], RSW_C_KG18_RAYL);
        if (lower) {
            RswSpec s = rsw_spec(colh2o[L], colch4[L], RSW_C_KG18_STRRAT,
                                 8.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 9 + s.js;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 9 + s.js;
            tg = AD(MU(s.speccomb,
                       rsw_acc8(tab + RSW_T_KG18_ABSA, 585, ig, ind0, ind1, 9, s)),
                    rsw_selffor(tab + RSW_T_KG18_SELFREF, 10,
                                tab + RSW_T_KG18_FORREF, 3, ig, colh2o[L],
                                selffac[L], selffrac[L], indself[L],
                                forfac[L], forfrac[L], indfor[L]));
        } else {
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = MU(colch4[L], rsw_acc4(tab + RSW_T_KG18_ABSB, 235, ig,
                                        ind0, ind1, fac00[L], fac10[L],
                                        fac01[L], fac11[L]));
        }
        break;
    }
    case 19: {
        int ig = iw - 26;
        tr = MU(colmol[L], RSW_C_KG19_RAYL);
        if (lower) {
            RswSpec s = rsw_spec(colh2o[L], colco2[L], RSW_C_KG19_STRRAT,
                                 8.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 9 + s.js;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 9 + s.js;
            tg = AD(MU(s.speccomb,
                       rsw_acc8(tab + RSW_T_KG19_ABSA, 585, ig, ind0, ind1, 9, s)),
                    rsw_selffor(tab + RSW_T_KG19_SELFREF, 10,
                                tab + RSW_T_KG19_FORREF, 3, ig, colh2o[L],
                                selffac[L], selffrac[L], indself[L],
                                forfac[L], forfrac[L], indfor[L]));
        } else {
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = MU(colco2[L], rsw_acc4(tab + RSW_T_KG19_ABSB, 235, ig,
                                        ind0, ind1, fac00[L], fac10[L],
                                        fac01[L], fac11[L]));
        }
        break;
    }
    case 20: {
        int ig = iw - 34;
        tr = MU(colmol[L], RSW_C_KG20_RAYL);
        const real* absch4 = tab + RSW_T_KG20_ABSCH4;
        if (lower) {
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 1 + 1;
            const real* sc = tab + RSW_T_KG20_SELFREF + (size_t)10 * ig;
            const real* fc = tab + RSW_T_KG20_FORREF + (size_t)4 * ig;
            int inds = indself[L], indf = indfor[L];
            real core = rsw_acc4(tab + RSW_T_KG20_ABSA, 65, ig, ind0, ind1,
                                 fac00[L], fac10[L], fac01[L], fac11[L]);
            real sr = AD(sc[inds - 1], MU(selffrac[L], SU(sc[inds], sc[inds - 1])));
            real fr = AD(fc[indf - 1], MU(forfrac[L], SU(fc[indf], fc[indf - 1])));
            real inner = AD(AD(core, MU(selffac[L], sr)), MU(forfac[L], fr));
            tg = AD(MU(colh2o[L], inner), MU(colch4[L], absch4[ig]));
        } else {
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 1 + 1;
            const real* fc = tab + RSW_T_KG20_FORREF + (size_t)4 * ig;
            int indf = indfor[L];
            real core = rsw_acc4(tab + RSW_T_KG20_ABSB, 235, ig, ind0, ind1,
                                 fac00[L], fac10[L], fac01[L], fac11[L]);
            real fr = AD(fc[indf - 1], MU(forfrac[L], SU(fc[indf], fc[indf - 1])));
            real inner = AD(core, MU(forfac[L], fr));
            tg = AD(MU(colh2o[L], inner), MU(colch4[L], absch4[ig]));
        }
        break;
    }
    case 21: {
        int ig = iw - 44;
        tr = MU(colmol[L], RSW_C_KG21_RAYL);
        if (lower) {
            RswSpec s = rsw_spec(colh2o[L], colco2[L], RSW_C_KG21_STRRAT,
                                 8.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 9 + s.js;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 9 + s.js;
            tg = AD(MU(s.speccomb,
                       rsw_acc8(tab + RSW_T_KG21_ABSA, 585, ig, ind0, ind1, 9, s)),
                    rsw_selffor(tab + RSW_T_KG21_SELFREF, 10,
                                tab + RSW_T_KG21_FORREF, 4, ig, colh2o[L],
                                selffac[L], selffrac[L], indself[L],
                                forfac[L], forfrac[L], indfor[L]));
        } else {
            RswSpec s = rsw_spec(colh2o[L], colco2[L], RSW_C_KG21_STRRAT,
                                 4.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 5 + s.js;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 5 + s.js;
            tg = AD(MU(s.speccomb,
                       rsw_acc8(tab + RSW_T_KG21_ABSB, 1175, ig, ind0, ind1, 5, s)),
                    rsw_forr(tab + RSW_T_KG21_FORREF, 4, ig, colh2o[L],
                             forfac[L], forfrac[L], indfor[L]));
        }
        break;
    }
    case 22: {
        int ig = iw - 54;
        tr = MU(colmol[L], RSW_C_KG22_RAYL);
        real o2adj = 1.6f;
        real o2cont = DV(MU(4.35e-4f, colo2[L]), MU(350.0f, 2.0f));
        if (lower) {
            RswSpec s = rsw_spec(colh2o[L], colo2[L],
                                 MU(o2adj, RSW_C_KG22_STRRAT),
                                 8.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 9 + s.js;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 9 + s.js;
            tg = AD(AD(MU(s.speccomb,
                          rsw_acc8(tab + RSW_T_KG22_ABSA, 585, ig, ind0, ind1, 9, s)),
                       rsw_selffor(tab + RSW_T_KG22_SELFREF, 10,
                                   tab + RSW_T_KG22_FORREF, 3, ig, colh2o[L],
                                   selffac[L], selffrac[L], indself[L],
                                   forfac[L], forfrac[L], indfor[L])),
                    o2cont);
        } else {
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = AD(MU(MU(colo2[L], o2adj),
                       rsw_acc4(tab + RSW_T_KG22_ABSB, 235, ig, ind0, ind1,
                                fac00[L], fac10[L], fac01[L], fac11[L])),
                    o2cont);
        }
        break;
    }
    case 23: {
        int ig = iw - 56;
        tr = MU(colmol[L], (tab + RSW_T_KG23_RAYL)[ig]);
        if (lower) {
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 1 + 1;
            const real* sc = tab + RSW_T_KG23_SELFREF + (size_t)10 * ig;
            const real* fc = tab + RSW_T_KG23_FORREF + (size_t)3 * ig;
            int inds = indself[L], indf = indfor[L];
            real core = rsw_acc4(tab + RSW_T_KG23_ABSA, 65, ig, ind0, ind1,
                                 fac00[L], fac10[L], fac01[L], fac11[L]);
            real sr = AD(sc[inds - 1], MU(selffrac[L], SU(sc[inds], sc[inds - 1])));
            real fr = AD(fc[indf - 1], MU(forfrac[L], SU(fc[indf], fc[indf - 1])));
            real inner = AD(AD(MU(RSW_C_KG23_GIVFAC, core),
                               MU(selffac[L], sr)), MU(forfac[L], fr));
            tg = MU(colh2o[L], inner);
        } else {
            tg = 0.0f;
        }
        break;
    }
    case 24: {
        int ig = iw - 66;
        if (lower) {
            RswSpec s = rsw_spec(colh2o[L], colo2[L], RSW_C_KG24_STRRAT,
                                 8.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            const real* ra = tab + RSW_T_KG24_RAYLA;   // (8, 9) F-order
            real r0 = ra[ig + 8 * (s.js - 1)];
            real r1 = ra[ig + 8 * s.js];
            tr = MU(colmol[L], AD(r0, MU(s.fs, SU(r1, r0))));
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 9 + s.js;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 9 + s.js;
            tg = AD(AD(MU(s.speccomb,
                          rsw_acc8(tab + RSW_T_KG24_ABSA, 585, ig, ind0, ind1, 9, s)),
                       MU(colo3[L], (tab + RSW_T_KG24_ABSO3A)[ig])),
                    rsw_selffor(tab + RSW_T_KG24_SELFREF, 10,
                                tab + RSW_T_KG24_FORREF, 3, ig, colh2o[L],
                                selffac[L], selffrac[L], indself[L],
                                forfac[L], forfrac[L], indfor[L]));
        } else {
            tr = MU(colmol[L], (tab + RSW_T_KG24_RAYLB)[ig]);
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = AD(MU(colo2[L],
                       rsw_acc4(tab + RSW_T_KG24_ABSB, 235, ig, ind0, ind1,
                                fac00[L], fac10[L], fac01[L], fac11[L])),
                    MU(colo3[L], (tab + RSW_T_KG24_ABSO3B)[ig]));
        }
        break;
    }
    case 25: {
        int ig = iw - 74;
        tr = MU(colmol[L], (tab + RSW_T_KG25_RAYL)[ig]);
        if (lower) {
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = AD(MU(colh2o[L],
                       rsw_acc4(tab + RSW_T_KG25_ABSA, 65, ig, ind0, ind1,
                                fac00[L], fac10[L], fac01[L], fac11[L])),
                    MU(colo3[L], (tab + RSW_T_KG25_ABSO3A)[ig]));
        } else {
            tg = MU(colo3[L], (tab + RSW_T_KG25_ABSO3B)[ig]);
        }
        break;
    }
    case 26: {
        int ig = iw - 80;
        tg = 0.0f;
        tr = MU(colmol[L], (tab + RSW_T_KG26_RAYL)[ig]);
        break;
    }
    case 27: {
        int ig = iw - 86;
        tr = MU(colmol[L], (tab + RSW_T_KG27_RAYL)[ig]);
        if (lower) {
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = MU(colo3[L], rsw_acc4(tab + RSW_T_KG27_ABSA, 65, ig,
                                       ind0, ind1, fac00[L], fac10[L],
                                       fac01[L], fac11[L]));
        } else {
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = MU(colo3[L], rsw_acc4(tab + RSW_T_KG27_ABSB, 235, ig,
                                       ind0, ind1, fac00[L], fac10[L],
                                       fac01[L], fac11[L]));
        }
        break;
    }
    case 28: {
        int ig = iw - 94;
        tr = MU(colmol[L], RSW_C_KG28_RAYL);
        if (lower) {
            RswSpec s = rsw_spec(colo3[L], colo2[L], RSW_C_KG28_STRRAT,
                                 8.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 9 + s.js;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 9 + s.js;
            tg = MU(s.speccomb,
                    rsw_acc8(tab + RSW_T_KG28_ABSA, 585, ig, ind0, ind1, 9, s));
        } else {
            RswSpec s = rsw_spec(colo3[L], colo2[L], RSW_C_KG28_STRRAT,
                                 4.0f, fac00[L], fac01[L], fac10[L], fac11[L]);
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 5 + s.js;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 5 + s.js;
            tg = MU(s.speccomb,
                    rsw_acc8(tab + RSW_T_KG28_ABSB, 1175, ig, ind0, ind1, 5, s));
        }
        break;
    }
    default: {   // 29
        int ig = iw - 100;
        tr = MU(colmol[L], RSW_C_KG29_RAYL);
        if (lower) {
            int ind0 = ((jp[L] - 1) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = (jp[L] * 5 + (jt1[L] - 1)) * 1 + 1;
            const real* sc = tab + RSW_T_KG29_SELFREF + (size_t)10 * ig;
            const real* fc = tab + RSW_T_KG29_FORREF + (size_t)4 * ig;
            int inds = indself[L], indf = indfor[L];
            real core = rsw_acc4(tab + RSW_T_KG29_ABSA, 65, ig, ind0, ind1,
                                 fac00[L], fac10[L], fac01[L], fac11[L]);
            real sr = AD(sc[inds - 1], MU(selffrac[L], SU(sc[inds], sc[inds - 1])));
            real fr = AD(fc[indf - 1], MU(forfrac[L], SU(fc[indf], fc[indf - 1])));
            real inner = AD(AD(core, MU(selffac[L], sr)), MU(forfac[L], fr));
            tg = AD(MU(colh2o[L], inner),
                    MU(colco2[L], (tab + RSW_T_KG29_ABSCO2)[ig]));
        } else {
            int ind0 = ((jp[L] - 13) * 5 + (jt[L] - 1)) * 1 + 1;
            int ind1 = ((jp[L] - 12) * 5 + (jt1[L] - 1)) * 1 + 1;
            tg = AD(MU(colco2[L],
                       rsw_acc4(tab + RSW_T_KG29_ABSB, 235, ig, ind0, ind1,
                                fac00[L], fac10[L], fac01[L], fac11[L])),
                    MU(colh2o[L], (tab + RSW_T_KG29_ABSH2O)[ig]));
        }
        break;
    }
    }

    taug[out] = tg;
    taur[out] = tr;
}

extern "C" __global__
void rsw_taumol(int nlayers, int laytrop,
                const real* __restrict__ tab,
                const int* __restrict__ ngb,
                const int* __restrict__ jp, const int* __restrict__ jt,
                const int* __restrict__ jt1,
                const real* __restrict__ colh2o, const real* __restrict__ colco2,
                const real* __restrict__ colch4, const real* __restrict__ colo2,
                const real* __restrict__ colo3, const real* __restrict__ colmol,
                const real* __restrict__ fac00, const real* __restrict__ fac01,
                const real* __restrict__ fac10, const real* __restrict__ fac11,
                const real* __restrict__ selffac, const real* __restrict__ selffrac,
                const int* __restrict__ indself,
                const real* __restrict__ forfac, const real* __restrict__ forfrac,
                const int* __restrict__ indfor,
                real* taug, real* taur)   // (nlayers, 112) F-order
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nlayers * RSW_NGPT) return;
    int L = idx % nlayers;          // 0-based layer
    int iw = idx / nlayers;         // 0-based g-point
    rsw_taumol_body(L, iw, nlayers, laytrop, tab, ngb, jp, jt, jt1,
                    colh2o, colco2, colch4, colo2, colo3, colmol,
                    fac00, fac01, fac10, fac11, selffac, selffrac,
                    indself, forfac, forfrac, indfor, taug, taur);
}

// Batched twin: thread -> (col, L, iw) with the same within-column
// (L, iw) split; laytrop becomes a per-column array (pure re-indexing).
extern "C" __global__
void rsw_taumol_b(int ncol, int nlayers,
                  const int* __restrict__ laytrop,     // (ncol,)
                  const real* __restrict__ tab,
                  const int* __restrict__ ngb,
                  const int* __restrict__ jp, const int* __restrict__ jt,
                  const int* __restrict__ jt1,
                  const real* __restrict__ colh2o, const real* __restrict__ colco2,
                  const real* __restrict__ colch4, const real* __restrict__ colo2,
                  const real* __restrict__ colo3, const real* __restrict__ colmol,
                  const real* __restrict__ fac00, const real* __restrict__ fac01,
                  const real* __restrict__ fac10, const real* __restrict__ fac11,
                  const real* __restrict__ selffac, const real* __restrict__ selffrac,
                  const int* __restrict__ indself,
                  const real* __restrict__ forfac, const real* __restrict__ forfrac,
                  const int* __restrict__ indfor,
                  real* taug, real* taur)   // (ncol, 112, nlayers) C-order
{
    long long W = (long long)nlayers * RSW_NGPT;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * W) return;
    int col = (int)(idx / W);
    int r = (int)(idx % W);
    int L = r % nlayers;
    int iw = r / nlayers;
    size_t o = (size_t)col * nlayers;
    size_t og = (size_t)col * nlayers * RSW_NGPT;
    rsw_taumol_body(L, iw, nlayers, laytrop[col], tab, ngb,
                    jp + o, jt + o, jt1 + o,
                    colh2o + o, colco2 + o, colch4 + o, colo2 + o,
                    colo3 + o, colmol + o,
                    fac00 + o, fac01 + o, fac10 + o, fac11 + o,
                    selffac + o, selffrac + o, indself + o,
                    forfac + o, forfrac + o, indfor + o,
                    taug + og, taur + og);
}

// sfluxzen: one thread per g-point; laysolfr per band precomputed on host
// (pure integer scans of jp, identical to the gated NumPy port).
__device__ void rsw_sfluxzen_body(int iw, int nlayers, int laytrop,
                  const real* __restrict__ tab,
                  const int* __restrict__ ngb,
                  const int* __restrict__ laysolfr,   // 14, Fortran layer idx
                  const int* __restrict__ jp, const int* __restrict__ jt,
                  const int* __restrict__ jt1,
                  const real* __restrict__ colh2o,
                  const real* __restrict__ colco2,
                  const real* __restrict__ colch4,
                  const real* __restrict__ colo2,
                  const real* __restrict__ colo3,
                  const real* __restrict__ fac00, const real* __restrict__ fac01,
                  const real* __restrict__ fac10, const real* __restrict__ fac11,
                  real* sfluxzen)
{
    int jb = ngb[iw];
    int lsf = laysolfr[jb - 16];
    int L = lsf - 1;
    real v = 0.0f;

    switch (jb) {
    case 16: {
        if (lsf > laytrop)   // set only when the upper loop runs
            v = (tab + RSW_T_KG16_SFLUXREF)[iw - 0];
        else
            return;          // sfluxzen stays 0 (pre-zeroed)
        break;
    }
    case 17: {
        int ig = iw - 6;
        RswSpec s = rsw_spec(colh2o[L], colco2[L], RSW_C_KG17_STRRAT, 4.0f,
                             fac00[L], fac01[L], fac10[L], fac11[L]);
        const real* sf = tab + RSW_T_KG17_SFLUXREF;    // (12, 5)
        real a = sf[ig + 12 * (s.js - 1)], b = sf[ig + 12 * s.js];
        v = AD(a, MU(s.fs, SU(b, a)));
        break;
    }
    case 18: {
        int ig = iw - 18;
        RswSpec s = rsw_spec(colh2o[L], colch4[L], RSW_C_KG18_STRRAT, 8.0f,
                             fac00[L], fac01[L], fac10[L], fac11[L]);
        const real* sf = tab + RSW_T_KG18_SFLUXREF;    // (8, 9)
        real a = sf[ig + 8 * (s.js - 1)], b = sf[ig + 8 * s.js];
        v = AD(a, MU(s.fs, SU(b, a)));
        break;
    }
    case 19: {
        int ig = iw - 26;
        RswSpec s = rsw_spec(colh2o[L], colco2[L], RSW_C_KG19_STRRAT, 8.0f,
                             fac00[L], fac01[L], fac10[L], fac11[L]);
        const real* sf = tab + RSW_T_KG19_SFLUXREF;
        real a = sf[ig + 8 * (s.js - 1)], b = sf[ig + 8 * s.js];
        v = AD(a, MU(s.fs, SU(b, a)));
        break;
    }
    case 20:
        v = (tab + RSW_T_KG20_SFLUXREF)[iw - 34];
        break;
    case 21: {
        int ig = iw - 44;
        RswSpec s = rsw_spec(colh2o[L], colco2[L], RSW_C_KG21_STRRAT, 8.0f,
                             fac00[L], fac01[L], fac10[L], fac11[L]);
        const real* sf = tab + RSW_T_KG21_SFLUXREF;    // (10, 9)
        real a = sf[ig + 10 * (s.js - 1)], b = sf[ig + 10 * s.js];
        v = AD(a, MU(s.fs, SU(b, a)));
        break;
    }
    case 22: {
        int ig = iw - 54;
        RswSpec s = rsw_spec(colh2o[L], colo2[L],
                             MU(1.6f, RSW_C_KG22_STRRAT), 8.0f,
                             fac00[L], fac01[L], fac10[L], fac11[L]);
        const real* sf = tab + RSW_T_KG22_SFLUXREF;    // (2, 9)
        real a = sf[ig + 2 * (s.js - 1)], b = sf[ig + 2 * s.js];
        v = AD(a, MU(s.fs, SU(b, a)));
        break;
    }
    case 23:
        v = (tab + RSW_T_KG23_SFLUXREF)[iw - 56];
        break;
    case 24: {
        int ig = iw - 66;
        RswSpec s = rsw_spec(colh2o[L], colo2[L], RSW_C_KG24_STRRAT, 8.0f,
                             fac00[L], fac01[L], fac10[L], fac11[L]);
        const real* sf = tab + RSW_T_KG24_SFLUXREF;    // (8, 9)
        real a = sf[ig + 8 * (s.js - 1)], b = sf[ig + 8 * s.js];
        v = AD(a, MU(s.fs, SU(b, a)));
        break;
    }
    case 25:
        v = (tab + RSW_T_KG25_SFLUXREF)[iw - 74];
        break;
    case 26:
        v = (tab + RSW_T_KG26_SFLUXREF)[iw - 80];
        break;
    case 27:
        if (lsf > laytrop)
            v = MU(RSW_C_KG27_SCALEKUR, (tab + RSW_T_KG27_SFLUXREF)[iw - 86]);
        else
            return;
        break;
    case 28: {
        int ig = iw - 94;
        RswSpec s = rsw_spec(colo3[L], colo2[L], RSW_C_KG28_STRRAT, 4.0f,
                             fac00[L], fac01[L], fac10[L], fac11[L]);
        const real* sf = tab + RSW_T_KG28_SFLUXREF;    // (6, 5)
        real a = sf[ig + 6 * (s.js - 1)], b = sf[ig + 6 * s.js];
        v = AD(a, MU(s.fs, SU(b, a)));
        break;
    }
    default:   // 29
        v = (tab + RSW_T_KG29_SFLUXREF)[iw - 100];
        break;
    }
    sfluxzen[iw] = v;
}

extern "C" __global__
void rsw_sfluxzen(int nlayers, int laytrop,
                  const real* __restrict__ tab,
                  const int* __restrict__ ngb,
                  const int* __restrict__ laysolfr,
                  const int* __restrict__ jp, const int* __restrict__ jt,
                  const int* __restrict__ jt1,
                  const real* __restrict__ colh2o,
                  const real* __restrict__ colco2,
                  const real* __restrict__ colch4,
                  const real* __restrict__ colo2,
                  const real* __restrict__ colo3,
                  const real* __restrict__ fac00, const real* __restrict__ fac01,
                  const real* __restrict__ fac10, const real* __restrict__ fac11,
                  real* sfluxzen)
{
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    if (iw >= RSW_NGPT) return;
    rsw_sfluxzen_body(iw, nlayers, laytrop, tab, ngb, laysolfr,
                      jp, jt, jt1, colh2o, colco2, colch4, colo2, colo3,
                      fac00, fac01, fac10, fac11, sfluxzen);
}

// Batched twin: thread -> (col, iw); laytrop and the 14-entry laysolfr
// table become per-column (pure re-indexing).
extern "C" __global__
void rsw_sfluxzen_b(int ncol, int nlayers,
                    const int* __restrict__ laytrop,    // (ncol,)
                    const real* __restrict__ tab,
                    const int* __restrict__ ngb,
                    const int* __restrict__ laysolfr,   // (ncol, 14)
                    const int* __restrict__ jp, const int* __restrict__ jt,
                    const int* __restrict__ jt1,
                    const real* __restrict__ colh2o,
                    const real* __restrict__ colco2,
                    const real* __restrict__ colch4,
                    const real* __restrict__ colo2,
                    const real* __restrict__ colo3,
                    const real* __restrict__ fac00, const real* __restrict__ fac01,
                    const real* __restrict__ fac10, const real* __restrict__ fac11,
                    real* sfluxzen)                     // (ncol, 112)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * RSW_NGPT) return;
    int col = (int)(idx / RSW_NGPT);
    int iw = (int)(idx % RSW_NGPT);
    size_t o = (size_t)col * nlayers;
    rsw_sfluxzen_body(iw, nlayers, laytrop[col], tab, ngb,
                      laysolfr + (size_t)col * RSW_NBND,
                      jp + o, jt + o, jt1 + o,
                      colh2o + o, colco2 + o, colch4 + o, colo2 + o,
                      colo3 + o, fac00 + o, fac01 + o, fac10 + o,
                      fac11 + o, sfluxzen + (size_t)col * RSW_NGPT);
}

// ---------------------------------------------------------------------------
// cldprmc_sw: one thread per (ig, lay).  Fatal Fortran STOPs set err_flag.
// ---------------------------------------------------------------------------

__device__ void rsw_cldprmc_body(int ig, int lay,
                 int nlayers, int inflag, int iceflag, int liqflag,
                 const real* __restrict__ tab,
                 const int* __restrict__ ngb,
                 const real* __restrict__ cldfmc,
                 const real* __restrict__ ciwpmc,
                 const real* __restrict__ clwpmc,
                 const real* __restrict__ cswpmc,
                 const real* __restrict__ reicmc,
                 const real* __restrict__ relqmc,
                 const real* __restrict__ resnmc,
                 real* taormc, real* taucmc, real* ssacmc, real* asmcmc,
                 const real* __restrict__ fsfcmc,
                 int* err_flag)
{
    size_t p = (size_t)ig + (size_t)RSW_NGPT * lay;

    taormc[p] = taucmc[p];

    real cwp = AD(AD(ciwpmc[p], clwpmc[p]), cswpmc[p]);
    const real cldmin = 1.0e-20f;
    if (!(cldfmc[p] >= cldmin && (cwp >= cldmin || taucmc[p] >= cldmin)))
        return;

    if (inflag == 0) {
        real taucldorig_a = taucmc[p];
        real ffp = fsfcmc[p];
        real ffp1 = SU(1.0f, ffp);
        real ffpssa = SU(1.0f, MU(ffp, ssacmc[p]));
        real ssacloud_a = DV(MU(ffp1, ssacmc[p]), ffpssa);
        real taucloud_a = MU(ffpssa, taucldorig_a);
        taormc[p] = taucldorig_a;
        ssacmc[p] = ssacloud_a;
        taucmc[p] = taucloud_a;
        asmcmc[p] = DV(SU(asmcmc[p], ffp), ffp1);
        return;
    }
    if (inflag == 1 || liqflag != 1 || iceflag < 2 || iceflag > 5) {
        // inflag=1 is a Fortran STOP; iceflag 1 needs wavenum bounds the
        // WRF option-4 driver cannot produce; fail closed.
        atomicOr(err_flag, 1);
        return;
    }

    real radice = reicmc[lay];
    real extcoice = 0.0f, ssacoice = 0.0f, gice = 0.0f, forwice = 0.0f;
    real extcosno = 0.0f, ssacosno = 0.0f, gsno = 0.0f, forwsno = 0.0f;
    int c = ngb[ig] - 16;

    if (EQ0(AD(ciwpmc[p], cswpmc[p]))) {
        // all four ice coefficients stay zero
    } else if (iceflag == 2) {
        if (radice < 5.0f || radice > 131.0f) { atomicOr(err_flag, 2); return; }
        real factor = DV(SU(radice, 2.0f), 3.0f);
        int index = (int)factor;
        if (index == 43) index = 42;
        real fint = SU(factor, (real)index);
        const real* e2 = tab + RSW_T_CLD_EXTICE2 + (size_t)43 * c;
        const real* s2 = tab + RSW_T_CLD_SSAICE2 + (size_t)43 * c;
        const real* a2 = tab + RSW_T_CLD_ASYICE2 + (size_t)43 * c;
        extcoice = AD(e2[index - 1], MU(fint, SU(e2[index], e2[index - 1])));
        ssacoice = AD(s2[index - 1], MU(fint, SU(s2[index], s2[index - 1])));
        gice = AD(a2[index - 1], MU(fint, SU(a2[index], a2[index - 1])));
        forwice = MU(gice, gice);
        if (extcoice < 0.0f || ssacoice > 1.0f || ssacoice < 0.0f ||
            gice > 1.0f || gice < 0.0f) { atomicOr(err_flag, 2); return; }
    } else {   // iceflag >= 3
        if (radice < 5.0f || radice > 140.0f) { atomicOr(err_flag, 2); return; }
        real factor = DV(SU(radice, 2.0f), 3.0f);
        int index = (int)factor;
        if (index == 46) index = 45;
        real fint = SU(factor, (real)index);
        const real* e3 = tab + RSW_T_CLD_EXTICE3 + (size_t)46 * c;
        const real* s3 = tab + RSW_T_CLD_SSAICE3 + (size_t)46 * c;
        const real* a3 = tab + RSW_T_CLD_ASYICE3 + (size_t)46 * c;
        const real* f3 = tab + RSW_T_CLD_FDLICE3 + (size_t)46 * c;
        extcoice = AD(e3[index - 1], MU(fint, SU(e3[index], e3[index - 1])));
        ssacoice = AD(s3[index - 1], MU(fint, SU(s3[index], s3[index - 1])));
        gice = AD(a3[index - 1], MU(fint, SU(a3[index], a3[index - 1])));
        real fdelta = AD(f3[index - 1], MU(fint, SU(f3[index], f3[index - 1])));
        if (fdelta < 0.0f || fdelta > 1.0f) { atomicOr(err_flag, 2); return; }
        forwice = AD(fdelta, DV(0.5f, ssacoice));
        if (forwice > gice) forwice = gice;
        if (extcoice < 0.0f || ssacoice > 1.0f || ssacoice < 0.0f ||
            gice > 1.0f || gice < 0.0f) { atomicOr(err_flag, 2); return; }
    }

    if (cswpmc[p] > 0.0f && iceflag == 5) {
        real radsno = resnmc[lay];
        if (radsno < 5.0f || radsno > 140.0f) { atomicOr(err_flag, 4); return; }
        real factor = DV(SU(radsno, 2.0f), 3.0f);
        int index = (int)factor;
        if (index == 46) index = 45;
        real fint = SU(factor, (real)index);
        const real* e3 = tab + RSW_T_CLD_EXTICE3 + (size_t)46 * c;
        const real* s3 = tab + RSW_T_CLD_SSAICE3 + (size_t)46 * c;
        const real* a3 = tab + RSW_T_CLD_ASYICE3 + (size_t)46 * c;
        const real* f3 = tab + RSW_T_CLD_FDLICE3 + (size_t)46 * c;
        extcosno = AD(e3[index - 1], MU(fint, SU(e3[index], e3[index - 1])));
        ssacosno = AD(s3[index - 1], MU(fint, SU(s3[index], s3[index - 1])));
        gsno = AD(a3[index - 1], MU(fint, SU(a3[index], a3[index - 1])));
        real fdelta = AD(f3[index - 1], MU(fint, SU(f3[index], f3[index - 1])));
        if (fdelta < 0.0f || fdelta > 1.0f) { atomicOr(err_flag, 4); return; }
        forwsno = AD(fdelta, DV(0.5f, ssacosno));
        if (forwsno > gsno) forwsno = gsno;
        if (extcosno < 0.0f || ssacosno > 1.0f || ssacosno < 0.0f ||
            gsno > 1.0f || gsno < 0.0f) { atomicOr(err_flag, 4); return; }
    }

    real extcoliq = 0.0f, ssacoliq = 0.0f, gliq = 0.0f, forwliq = 0.0f;
    if (!EQ0(clwpmc[p])) {   // liqflag == 1
        real radliq = relqmc[lay];
        if (radliq < 1.5f || radliq > 60.0f) { atomicOr(err_flag, 8); return; }
        int index = (int)SU(radliq, 1.5f);
        if (index == 0) index = 1;
        if (index == 58) index = 57;
        real fint = SU(SU(radliq, 1.5f), (real)index);
        const real* el = tab + RSW_T_CLD_EXTLIQ1 + (size_t)58 * c;
        const real* sl = tab + RSW_T_CLD_SSALIQ1 + (size_t)58 * c;
        const real* al = tab + RSW_T_CLD_ASYLIQ1 + (size_t)58 * c;
        extcoliq = AD(el[index - 1], MU(fint, SU(el[index], el[index - 1])));
        ssacoliq = AD(sl[index - 1], MU(fint, SU(sl[index], sl[index - 1])));
        if (fint < 0.0f && ssacoliq > 1.0f) ssacoliq = sl[index - 1];
        gliq = AD(al[index - 1], MU(fint, SU(al[index], al[index - 1])));
        forwliq = MU(gliq, gliq);
        if (extcoliq < 0.0f || ssacoliq > 1.0f || ssacoliq < 0.0f ||
            gliq > 1.0f || gliq < 0.0f) { atomicOr(err_flag, 8); return; }
    }

    real scatliq, scatice, scatsno;
    if (iceflag < 5) {
        real tauliqorig = MU(clwpmc[p], extcoliq);
        real tauiceorig = MU(ciwpmc[p], extcoice);
        taormc[p] = AD(tauliqorig, tauiceorig);
        real ssaliq = DV(MU(ssacoliq, SU(1.0f, forwliq)),
                         SU(1.0f, MU(forwliq, ssacoliq)));
        real tauliq = MU(SU(1.0f, MU(forwliq, ssacoliq)), tauliqorig);
        real ssaice = DV(MU(ssacoice, SU(1.0f, forwice)),
                         SU(1.0f, MU(forwice, ssacoice)));
        real tauice = MU(SU(1.0f, MU(forwice, ssacoice)), tauiceorig);
        scatliq = MU(ssaliq, tauliq);
        scatice = MU(ssaice, tauice);
        scatsno = 0.0f;
        taucmc[p] = AD(tauliq, tauice);
    } else {
        real tauliqorig = MU(clwpmc[p], extcoliq);
        real tauiceorig = MU(ciwpmc[p], extcoice);
        real tausnoorig = MU(cswpmc[p], extcosno);
        taormc[p] = AD(AD(tauliqorig, tauiceorig), tausnoorig);
        real ssaliq = DV(MU(ssacoliq, SU(1.0f, forwliq)),
                         SU(1.0f, MU(forwliq, ssacoliq)));
        real tauliq = MU(SU(1.0f, MU(forwliq, ssacoliq)), tauliqorig);
        real ssaice = DV(MU(ssacoice, SU(1.0f, forwice)),
                         SU(1.0f, MU(forwice, ssacoice)));
        real tauice = MU(SU(1.0f, MU(forwice, ssacoice)), tauiceorig);
        real ssasno = DV(MU(ssacosno, SU(1.0f, forwsno)),
                         SU(1.0f, MU(forwsno, ssacosno)));
        real tausno = MU(SU(1.0f, MU(forwsno, ssacosno)), tausnoorig);
        scatliq = MU(ssaliq, tauliq);
        scatice = MU(ssaice, tauice);
        scatsno = MU(ssasno, tausno);
        taucmc[p] = AD(AD(tauliq, tauice), tausno);
    }

    if (EQ0(taucmc[p])) taucmc[p] = cldmin;
    if (EQ0(scatice)) scatice = cldmin;
    if (EQ0(scatsno)) scatsno = cldmin;

    if (iceflag < 5)
        ssacmc[p] = DV(AD(scatliq, scatice), taucmc[p]);
    else
        ssacmc[p] = DV(AD(AD(scatliq, scatice), scatsno), taucmc[p]);

    if (iceflag == 3 || iceflag == 4) {
        asmcmc[p] = MU(DV(1.0f, AD(scatliq, scatice)),
            AD(DV(MU(scatliq, SU(gliq, forwliq)), SU(1.0f, forwliq)),
               MU(scatice, DV(SU(gice, forwice), SU(1.0f, forwice)))));
    } else if (iceflag == 5) {
        asmcmc[p] = MU(DV(1.0f, AD(AD(scatliq, scatice), scatsno)),
            AD(AD(DV(MU(scatliq, SU(gliq, forwliq)), SU(1.0f, forwliq)),
                  MU(scatice, DV(SU(gice, forwice), SU(1.0f, forwice)))),
               MU(scatsno, DV(SU(gsno, forwsno), SU(1.0f, forwsno)))));
    } else {
        asmcmc[p] = DV(
            AD(DV(MU(scatliq, SU(gliq, forwliq)), SU(1.0f, forwliq)),
               DV(MU(scatice, SU(gice, forwice)), SU(1.0f, forwice))),
            AD(scatliq, scatice));
    }
}

extern "C" __global__
void rsw_cldprmc(int nlayers, int inflag, int iceflag, int liqflag,
                 const real* __restrict__ tab,
                 const int* __restrict__ ngb,
                 const real* __restrict__ cldfmc,
                 const real* __restrict__ ciwpmc,
                 const real* __restrict__ clwpmc,
                 const real* __restrict__ cswpmc,
                 const real* __restrict__ reicmc,
                 const real* __restrict__ relqmc,
                 const real* __restrict__ resnmc,
                 real* taormc, real* taucmc, real* ssacmc, real* asmcmc,
                 const real* __restrict__ fsfcmc,
                 int* err_flag)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nlayers * RSW_NGPT) return;
    int ig = idx % RSW_NGPT;
    int lay = idx / RSW_NGPT;
    rsw_cldprmc_body(ig, lay, nlayers, inflag, iceflag, liqflag, tab, ngb,
                     cldfmc, ciwpmc, clwpmc, cswpmc, reicmc, relqmc,
                     resnmc, taormc, taucmc, ssacmc, asmcmc, fsfcmc,
                     err_flag);
}

// Batched twin: thread -> (col, ig, lay) with the same within-column
// (ig, lay) split; McICA slabs move to the column base, the per-layer
// radii by col*nlayers; err_flag stays one shared word for the whole
// batch (atomicOr, exactly as before).  Pure re-indexing.
extern "C" __global__
void rsw_cldprmc_b(int ncol, int nlayers,
                   int inflag, int iceflag, int liqflag,
                   const real* __restrict__ tab,
                   const int* __restrict__ ngb,
                   const real* __restrict__ cldfmc,   // (ncol, nl, 112)
                   const real* __restrict__ ciwpmc,
                   const real* __restrict__ clwpmc,
                   const real* __restrict__ cswpmc,
                   const real* __restrict__ reicmc,   // (ncol, nl)
                   const real* __restrict__ relqmc,
                   const real* __restrict__ resnmc,
                   real* taormc, real* taucmc, real* ssacmc, real* asmcmc,
                   const real* __restrict__ fsfcmc,
                   int* err_flag)
{
    long long W = (long long)nlayers * RSW_NGPT;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * W) return;
    int col = (int)(idx / W);
    int r = (int)(idx % W);
    int ig = r % RSW_NGPT;
    int lay = r / RSW_NGPT;
    size_t og = (size_t)col * nlayers * RSW_NGPT;
    size_t o = (size_t)col * nlayers;
    rsw_cldprmc_body(ig, lay, nlayers, inflag, iceflag, liqflag, tab, ngb,
                     cldfmc + og, ciwpmc + og, clwpmc + og, cswpmc + og,
                     reicmc + o, relqmc + o, resnmc + o,
                     taormc + og, taucmc + og, ssacmc + og, asmcmc + og,
                     fsfcmc + og, err_flag);
}

// ---------------------------------------------------------------------------
// reftra_sw as a device function over thread-local arrays (kmodts = 2).
// ---------------------------------------------------------------------------

#define RSW_MAXLAY 64   // >= nlayers+1; asserted host-side

__device__ void rsw_reftra(int nlayers, const unsigned char* lrtchk,
                           const real* pgg, real prmuz, const real* ptau,
                           const real* pw, const real* __restrict__ exp_tbl,
                           real* pref, real* prefd, real* ptra, real* ptrad)
{
    const real eps = 1.0e-08f;
    const real zwcrit = 0.9999995f;
    for (int jk = 0; jk < nlayers; ++jk) {
        if (!lrtchk[jk]) {
            pref[jk] = 0.0f;
            ptra[jk] = 1.0f;
            prefd[jk] = 0.0f;
            ptrad[jk] = 1.0f;
            continue;
        }
        real zto1 = ptau[jk];
        real zw = pw[jk];
        real zg = pgg[jk];

        real zg3 = MU(3.0f, zg);
        real zgamma1 = MU(SU(8.0f, MU(zw, AD(5.0f, zg3))), 0.25f);
        real zgamma2 = MU(MU(3.0f, MU(zw, SU(1.0f, zg))), 0.25f);
        real zgamma3 = MU(SU(2.0f, MU(zg3, prmuz)), 0.25f);
        real zgamma4 = SU(1.0f, zgamma3);

        real zwo = 0.0f;
        real denom = 1.0f;
        if (zg != 1.0f) {
            real q = DV(zg, SU(1.0f, zg));
            denom = SU(1.0f, MU(SU(1.0f, zw), MU(q, q)));
        }
        if (zw > 0.0f && denom != 0.0f) zwo = DV(zw, denom);

        if (zwo >= zwcrit) {
            real za = MU(zgamma1, prmuz);
            real za1 = SU(za, zgamma3);
            real zgt = MU(zgamma1, zto1);
            real ze1 = DV(zto1, prmuz);
            if (!(ze1 < 500.0f)) ze1 = 500.0f;   // min(x, 500)
            real ze2 = rsw_etbl(exp_tbl, ze1);
            pref[jk] = DV(SU(zgt, MU(za1, SU(1.0f, ze2))), AD(1.0f, zgt));
            ptra[jk] = SU(1.0f, pref[jk]);
            prefd[jk] = DV(zgt, AD(1.0f, zgt));
            ptrad[jk] = SU(1.0f, prefd[jk]);
            if (ze2 == 1.0f) {
                pref[jk] = 0.0f;
                ptra[jk] = 1.0f;
                prefd[jk] = 0.0f;
                ptrad[jk] = 1.0f;
            }
        } else {
            real za1 = AD(MU(zgamma1, zgamma4), MU(zgamma2, zgamma3));
            real za2 = AD(MU(zgamma1, zgamma3), MU(zgamma2, zgamma4));
            real zrk = __fsqrt_rn(SU(MU(zgamma1, zgamma1),
                                     MU(zgamma2, zgamma2)));
            real zrp = MU(zrk, prmuz);
            real zrp1 = AD(1.0f, zrp);
            real zrm1 = SU(1.0f, zrp);
            real zrk2 = MU(2.0f, zrk);
            real zrpp = SU(1.0f, MU(zrp, zrp));
            real zrkg = AD(zrk, zgamma1);
            real zr1 = MU(zrm1, AD(za2, MU(zrk, zgamma3)));
            real zr2 = MU(zrp1, SU(za2, MU(zrk, zgamma3)));
            real zr3 = MU(zrk2, SU(zgamma3, MU(za2, prmuz)));
            real zr4 = MU(zrpp, zrkg);
            real zr5 = MU(zrpp, SU(zrk, zgamma1));
            real zt1 = MU(zrp1, AD(za1, MU(zrk, zgamma4)));
            real zt2 = MU(zrm1, SU(za1, MU(zrk, zgamma4)));
            real zt3 = MU(zrk2, AD(zgamma4, MU(za1, prmuz)));
            real zt4 = zr4;
            real zt5 = zr5;
            real zbeta = DV(SU(zgamma1, zrk), zrkg);

            real ze1 = MU(zrk, zto1);
            if (!(ze1 < 500.0f)) ze1 = 500.0f;
            real ze2 = DV(zto1, prmuz);
            if (!(ze2 < 500.0f)) ze2 = 500.0f;
            real zem1, zep1, zem2, zep2;
            if (ze1 <= RSW_C_OD_LO) {
                zem1 = AD(SU(1.0f, ze1), MU(MU(0.5f, ze1), ze1));
                zep1 = DV(1.0f, zem1);
            } else {
                real tblind = DV(ze1, AD(RSW_C_BPADE, ze1));
                int itind = (int)AD(MU(RSW_C_TBLINT, tblind), 0.5f);
                zem1 = exp_tbl[itind];
                zep1 = DV(1.0f, zem1);
            }
            if (ze2 <= RSW_C_OD_LO) {
                zem2 = AD(SU(1.0f, ze2), MU(MU(0.5f, ze2), ze2));
                zep2 = DV(1.0f, zem2);
            } else {
                real tblind = DV(ze2, AD(RSW_C_BPADE, ze2));
                int itind = (int)AD(MU(RSW_C_TBLINT, tblind), 0.5f);
                zem2 = exp_tbl[itind];
                zep2 = DV(1.0f, zem2);
            }

            real zdenr = AD(MU(zr4, zep1), MU(zr5, zem1));
            real zdent = AD(MU(zt4, zep1), MU(zt5, zem1));
            if (zdenr >= -eps && zdenr <= eps) {
                pref[jk] = eps;
                ptra[jk] = zem2;
            } else {
                pref[jk] = DV(MU(zw, SU(SU(MU(zr1, zep1), MU(zr2, zem1)),
                                        MU(zr3, zem2))), zdenr);
                ptra[jk] = SU(zem2,
                    DV(MU(MU(zem2, zw), SU(SU(MU(zt1, zep1), MU(zt2, zem1)),
                                           MU(zt3, zep2))), zdent));
            }

            real zemm = MU(zem1, zem1);
            real zdend = DV(1.0f, MU(SU(1.0f, MU(zbeta, zemm)), zrkg));
            prefd[jk] = MU(MU(zgamma2, SU(1.0f, zemm)), zdend);
            ptrad[jk] = MU(MU(zrk2, zem1), zdend);
        }
    }
}

// ztdn is caller-provided scratch (>= klev+1 entries): it was a
// thread-local array until the batched work moved every per-thread array
// of this pipeline into an explicit workspace (see rsw_spcvmc_body);
// loads/stores only, no arithmetic difference.
__device__ void rsw_vrtqdr(int klev, const real* pref, const real* prefd,
                           const real* ptra, const real* ptrad,
                           const real* pdbt, real* prdnd, real* prup,
                           real* prupd, const real* ptdbt,
                           real* pfd_col, real* pfu_col, real* ztdn)
{
    real zreflect = DV(1.0f, SU(1.0f, MU(prefd[klev], prefd[klev - 1])));
    prup[klev - 1] = AD(pref[klev - 1],
        MU(MU(ptrad[klev - 1],
              AD(MU(SU(ptra[klev - 1], pdbt[klev - 1]), prefd[klev]),
                 MU(pdbt[klev - 1], pref[klev]))), zreflect));
    prupd[klev - 1] = AD(prefd[klev - 1],
        MU(MU(MU(ptrad[klev - 1], ptrad[klev - 1]), prefd[klev]), zreflect));

    for (int jk = 1; jk < klev; ++jk) {
        int ikp = klev + 1 - jk;
        int ikx = ikp - 1;
        zreflect = DV(1.0f, SU(1.0f, MU(prupd[ikp - 1], prefd[ikx - 1])));
        prup[ikx - 1] = AD(pref[ikx - 1],
            MU(MU(ptrad[ikx - 1],
                  AD(MU(SU(ptra[ikx - 1], pdbt[ikx - 1]), prupd[ikp - 1]),
                     MU(pdbt[ikx - 1], prup[ikp - 1]))), zreflect));
        prupd[ikx - 1] = AD(prefd[ikx - 1],
            MU(MU(MU(ptrad[ikx - 1], ptrad[ikx - 1]), prupd[ikp - 1]),
               zreflect));
    }

    ztdn[0] = 1.0f;
    prdnd[0] = 0.0f;
    ztdn[1] = ptra[0];
    prdnd[1] = prefd[0];

    for (int jk = 2; jk <= klev; ++jk) {
        int ikp = jk + 1;
        zreflect = DV(1.0f, SU(1.0f, MU(prefd[jk - 1], prdnd[jk - 1])));
        ztdn[ikp - 1] = AD(MU(ptdbt[jk - 1], ptra[jk - 1]),
            MU(MU(ptrad[jk - 1],
                  AD(SU(ztdn[jk - 1], ptdbt[jk - 1]),
                     MU(MU(ptdbt[jk - 1], pref[jk - 1]), prdnd[jk - 1]))),
               zreflect));
        prdnd[ikp - 1] = AD(prefd[jk - 1],
            MU(MU(MU(ptrad[jk - 1], ptrad[jk - 1]), prdnd[jk - 1]),
               zreflect));
    }

    for (int jk = 0; jk <= klev; ++jk) {
        zreflect = DV(1.0f, SU(1.0f, MU(prdnd[jk], prupd[jk])));
        pfu_col[jk] = MU(AD(MU(ptdbt[jk], prup[jk]),
                            MU(SU(ztdn[jk], ptdbt[jk]), prupd[jk])),
                         zreflect);
        pfd_col[jk] = AD(ptdbt[jk],
            MU(AD(SU(ztdn[jk], ptdbt[jk]),
                  MU(MU(ptdbt[jk], prup[jk]), prdnd[jk])), zreflect));
    }
}

// ---------------------------------------------------------------------------
// spcvmc_sw per-g-point pipeline: one thread per g-point.
// Inputs are the same arrays the NumPy port takes.  Per-thread state
// lived in ~9 KiB of thread-local arrays; on this device a local frame
// of F bytes reserves ~F x 1536 threads x 170 SMs machine-wide at first
// launch (~2.3 GiB here), so the arrays now live in a caller-provided
// workspace: wk carries RSW_SPCVMC_WK real arrays and wkc RSW_SPCVMC_WKC
// flag arrays of (nlayers + 1) entries per thread, carved below in a
// fixed order.  Every statement that touches them is unchanged -- the
// arrays merely moved from local to global memory (loads/stores round
// nothing), which the oracle and batched gates re-prove bitwise.
// ---------------------------------------------------------------------------

#define RSW_SPCVMC_WK 35    // real arrays carved from wk, in this order
#define RSW_SPCVMC_WKC 2    // unsigned char arrays carved from wkc

__device__ void rsw_spcvmc_body(int iw1, int nlayers,
                    const real* __restrict__ tab,
                    const int* __restrict__ ngb,
                    const real* __restrict__ palbd,    // (14)
                    const real* __restrict__ palbp,
                    const real* __restrict__ pcldfmc,  // (nlayers, 112) F
                    const real* __restrict__ ptaucmc,
                    const real* __restrict__ pasycmc,
                    const real* __restrict__ pomgcmc,
                    const real* __restrict__ ptaormc,
                    const real* __restrict__ ptaua,    // (nlayers, 14) F
                    const real* __restrict__ pasya,
                    const real* __restrict__ pomga,
                    real prmu0,
                    const real* __restrict__ adjflux,  // (14)
                    const real* __restrict__ sfluxzen, // (112)
                    const real* __restrict__ taug,     // (nlayers, 112) F
                    const real* __restrict__ taur,
                    real* zincflx,                     // (112)
                    real* zcd, real* zcu,              // (nlayers+1, 112) F
                    real* zfd, real* zfu,
                    real* ztdbt_nodel_out,             // (nlayers+1, 112) F
                    real* ztdbtc_nodel_out,
                    real* wk,                          // RSW_SPCVMC_WK x n1
                    unsigned char* wkc)                // RSW_SPCVMC_WKC x n1
{
    int klev = nlayers;
    int ibm = ngb[iw1] - 15;                            // 1..14
    const real* exp_tbl = tab + RSW_T_EXP_TBL;
    const real repclc = 1.0e-12f;

    const int n1 = nlayers + 1;
    real* ztauc = wk;                 real* zomcc = wk + n1;
    real* zgcc = wk + 2 * n1;         real* ztauo = wk + 3 * n1;
    real* zomco = wk + 4 * n1;        real* zgco = wk + 5 * n1;
    real* zrefc = wk + 6 * n1;        real* zrefdc = wk + 7 * n1;
    real* ztrac = wk + 8 * n1;        real* ztradc = wk + 9 * n1;
    real* zrefo = wk + 10 * n1;       real* zrefdo = wk + 11 * n1;
    real* ztrao = wk + 12 * n1;       real* ztrado = wk + 13 * n1;
    real* zref = wk + 14 * n1;        real* zrefd = wk + 15 * n1;
    real* ztra = wk + 16 * n1;        real* ztrad = wk + 17 * n1;
    real* zdbtc = wk + 18 * n1;       real* ztdbtc = wk + 19 * n1;
    real* zdbt = wk + 20 * n1;        real* ztdbt = wk + 21 * n1;
    real* zdbtc_nodel = wk + 22 * n1; real* ztdbtc_nodel = wk + 23 * n1;
    real* zdbt_nodel = wk + 24 * n1;  real* ztdbt_nodel = wk + 25 * n1;
    real* zrdnd = wk + 26 * n1;       real* zrdndc = wk + 27 * n1;
    real* zrup = wk + 28 * n1;        real* zrupd = wk + 29 * n1;
    real* zrupc = wk + 30 * n1;       real* zrupdc = wk + 31 * n1;
    real* fd_col = wk + 32 * n1;      real* fu_col = wk + 33 * n1;
    real* ztdn = wk + 34 * n1;
    unsigned char* lrtchkclr = wkc;
    unsigned char* lrtchkcld = wkc + n1;

    zincflx[iw1] = MU(MU(adjflux[ibm - 1], sfluxzen[iw1]), prmu0);

    ztdbtc[0] = 1.0f;
    ztdbtc_nodel[0] = 1.0f;
    zdbtc[klev] = 0.0f;
    ztrac[klev] = 0.0f;
    ztradc[klev] = 0.0f;
    zrefc[klev] = palbp[ibm - 1];
    zrefdc[klev] = palbd[ibm - 1];
    zrupc[klev] = palbp[ibm - 1];
    zrupdc[klev] = palbd[ibm - 1];
    ztdbt[0] = 1.0f;
    ztdbt_nodel[0] = 1.0f;
    zdbt[klev] = 0.0f;
    ztra[klev] = 0.0f;
    ztrad[klev] = 0.0f;
    zref[klev] = palbp[ibm - 1];
    zrefd[klev] = palbd[ibm - 1];
    zrup[klev] = palbp[ibm - 1];
    zrupd[klev] = palbd[ibm - 1];

    for (int jk = 1; jk <= klev; ++jk) {
        int ikl = klev + 1 - jk;
        int J = jk - 1, K = ikl - 1;
        size_t pKw = (size_t)K + (size_t)nlayers * iw1;
        size_t pKb = (size_t)K + (size_t)nlayers * (ibm - 1);
        lrtchkclr[J] = 1;
        lrtchkcld[J] = (pcldfmc[pKw] > repclc) ? 1 : 0;

        ztauc[J] = AD(AD(taur[pKw], taug[pKw]), ptaua[pKb]);
        zomcc[J] = AD(MU(taur[pKw], 1.0f), MU(ptaua[pKb], pomga[pKb]));
        zgcc[J] = DV(MU(MU(pasya[pKb], pomga[pKb]), ptaua[pKb]), zomcc[J]);
        zomcc[J] = DV(zomcc[J], ztauc[J]);

        real zclear = SU(1.0f, pcldfmc[pKw]);
        real zcloud = pcldfmc[pKw];

        real ze1 = DV(ztauc[J], prmu0);
        real zdbtmc = rsw_etbl(exp_tbl, ze1);
        zdbtc_nodel[J] = zdbtmc;
        ztdbtc_nodel[J + 1] = MU(zdbtc_nodel[J], ztdbtc_nodel[J]);

        real tauorig = AD(ztauc[J], ptaormc[pKw]);
        ze1 = DV(tauorig, prmu0);
        real zdbtmo = rsw_etbl(exp_tbl, ze1);
        zdbt_nodel[J] = AD(MU(zclear, zdbtmc), MU(zcloud, zdbtmo));
        ztdbt_nodel[J + 1] = MU(zdbt_nodel[J], ztdbt_nodel[J]);
    }

    for (int J = 0; J < klev; ++J) {
        real zf = MU(zgcc[J], zgcc[J]);
        real zwf = MU(zomcc[J], zf);
        ztauc[J] = MU(SU(1.0f, zwf), ztauc[J]);
        zomcc[J] = DV(SU(zomcc[J], zwf), SU(1.0f, zwf));
        zgcc[J] = DV(SU(zgcc[J], zf), SU(1.0f, zf));
    }

    for (int jk = 1; jk <= klev; ++jk) {
        int ikl = klev + 1 - jk;
        int J = jk - 1, K = ikl - 1;
        size_t pKw = (size_t)K + (size_t)nlayers * iw1;
        ztauo[J] = AD(ztauc[J], ptaucmc[pKw]);
        zomco[J] = AD(MU(ztauc[J], zomcc[J]), MU(ptaucmc[pKw], pomgcmc[pKw]));
        zgco[J] = DV(AD(MU(MU(ptaucmc[pKw], pomgcmc[pKw]), pasycmc[pKw]),
                        MU(MU(ztauc[J], zomcc[J]), zgcc[J])), zomco[J]);
        zomco[J] = DV(zomco[J], ztauo[J]);
    }

    rsw_reftra(klev, lrtchkclr, zgcc, prmu0, ztauc, zomcc, exp_tbl,
               zrefc, zrefdc, ztrac, ztradc);
    rsw_reftra(klev, lrtchkcld, zgco, prmu0, ztauo, zomco, exp_tbl,
               zrefo, zrefdo, ztrao, ztrado);

    for (int jk = 1; jk <= klev; ++jk) {
        int ikl = klev + 1 - jk;
        int J = jk - 1, K = ikl - 1;
        size_t pKw = (size_t)K + (size_t)nlayers * iw1;
        real zclear = SU(1.0f, pcldfmc[pKw]);
        real zcloud = pcldfmc[pKw];
        zref[J] = AD(MU(zclear, zrefc[J]), MU(zcloud, zrefo[J]));
        zrefd[J] = AD(MU(zclear, zrefdc[J]), MU(zcloud, zrefdo[J]));
        ztra[J] = AD(MU(zclear, ztrac[J]), MU(zcloud, ztrao[J]));
        ztrad[J] = AD(MU(zclear, ztradc[J]), MU(zcloud, ztrado[J]));

        real ze1 = DV(ztauc[J], prmu0);
        real zdbtmc = rsw_etbl(exp_tbl, ze1);
        zdbtc[J] = zdbtmc;
        ztdbtc[J + 1] = MU(zdbtc[J], ztdbtc[J]);

        ze1 = DV(ztauo[J], prmu0);
        real zdbtmo = rsw_etbl(exp_tbl, ze1);
        zdbt[J] = AD(MU(zclear, zdbtmc), MU(zcloud, zdbtmo));
        ztdbt[J + 1] = MU(zdbt[J], ztdbt[J]);
    }

    rsw_vrtqdr(klev, zrefc, zrefdc, ztrac, ztradc, zdbtc, zrdndc, zrupc,
               zrupdc, ztdbtc, fd_col, fu_col, ztdn);
    for (int jk = 0; jk <= klev; ++jk) {
        size_t q = (size_t)jk + (size_t)(klev + 1) * iw1;
        zcd[q] = fd_col[jk];
        zcu[q] = fu_col[jk];
    }
    rsw_vrtqdr(klev, zref, zrefd, ztra, ztrad, zdbt, zrdnd, zrup,
               zrupd, ztdbt, fd_col, fu_col, ztdn);
    for (int jk = 0; jk <= klev; ++jk) {
        size_t q = (size_t)jk + (size_t)(klev + 1) * iw1;
        zfd[q] = fd_col[jk];
        zfu[q] = fu_col[jk];
        ztdbt_nodel_out[q] = ztdbt_nodel[jk];
        ztdbtc_nodel_out[q] = ztdbtc_nodel[jk];
    }
}

extern "C" __global__
void rsw_spcvmc_gpt(int nlayers,
                    const real* __restrict__ tab,
                    const int* __restrict__ ngb,
                    const real* __restrict__ palbd,
                    const real* __restrict__ palbp,
                    const real* __restrict__ pcldfmc,
                    const real* __restrict__ ptaucmc,
                    const real* __restrict__ pasycmc,
                    const real* __restrict__ pomgcmc,
                    const real* __restrict__ ptaormc,
                    const real* __restrict__ ptaua,
                    const real* __restrict__ pasya,
                    const real* __restrict__ pomga,
                    real prmu0,
                    const real* __restrict__ adjflux,
                    const real* __restrict__ sfluxzen,
                    const real* __restrict__ taug,
                    const real* __restrict__ taur,
                    real* zincflx,
                    real* zcd, real* zcu,
                    real* zfd, real* zfu,
                    real* ztdbt_nodel_out,
                    real* ztdbtc_nodel_out,
                    real* wk,               // (112, RSW_SPCVMC_WK * n1)
                    unsigned char* wkc)     // (112, RSW_SPCVMC_WKC * n1)
{
    int iw1 = blockIdx.x * blockDim.x + threadIdx.x;   // 0-based g-point
    if (iw1 >= RSW_NGPT) return;
    size_t n1 = (size_t)nlayers + 1;
    rsw_spcvmc_body(iw1, nlayers, tab, ngb, palbd, palbp,
                    pcldfmc, ptaucmc, pasycmc, pomgcmc, ptaormc,
                    ptaua, pasya, pomga, prmu0, adjflux, sfluxzen,
                    taug, taur, zincflx, zcd, zcu, zfd, zfu,
                    ztdbt_nodel_out, ztdbtc_nodel_out,
                    wk + (size_t)iw1 * (RSW_SPCVMC_WK * n1),
                    wkc + (size_t)iw1 * (RSW_SPCVMC_WKC * n1));
}

// Batched twin: thread -> (col, iw1); prmu0 becomes per-column, the
// column slabs move to their bases, and each thread gets its own
// workspace slice (pure re-indexing around the identical body).
extern "C" __global__
void rsw_spcvmc_gpt_b(int ncol, int nlayers,
                      const real* __restrict__ tab,
                      const int* __restrict__ ngb,
                      const real* __restrict__ palbd,    // (ncol, 14)
                      const real* __restrict__ palbp,
                      const real* __restrict__ pcldfmc,  // (ncol, 112, nl)
                      const real* __restrict__ ptaucmc,
                      const real* __restrict__ pasycmc,
                      const real* __restrict__ pomgcmc,
                      const real* __restrict__ ptaormc,
                      const real* __restrict__ ptaua,    // (ncol, 14, nl)
                      const real* __restrict__ pasya,
                      const real* __restrict__ pomga,
                      const real* __restrict__ prmu0,    // (ncol,)
                      const real* __restrict__ adjflux,  // (ncol, 14)
                      const real* __restrict__ sfluxzen, // (ncol, 112)
                      const real* __restrict__ taug,     // (ncol, 112, nl)
                      const real* __restrict__ taur,
                      real* zincflx,                     // (ncol, 112)
                      real* zcd, real* zcu,              // (ncol, 112, n1)
                      real* zfd, real* zfu,
                      real* ztdbt_nodel_out,
                      real* ztdbtc_nodel_out,
                      real* wk,           // (ncol*112, RSW_SPCVMC_WK * n1)
                      unsigned char* wkc)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * RSW_NGPT) return;
    int col = (int)(idx / RSW_NGPT);
    int iw1 = (int)(idx % RSW_NGPT);
    size_t n1 = (size_t)nlayers + 1;
    size_t ob = (size_t)col * RSW_NBND;
    size_t og = (size_t)col * RSW_NGPT;
    size_t ogl = (size_t)col * nlayers * RSW_NGPT;
    size_t obl = (size_t)col * nlayers * RSW_NBND;
    size_t og1 = (size_t)col * n1 * RSW_NGPT;
    rsw_spcvmc_body(iw1, nlayers, tab, ngb, palbd + ob, palbp + ob,
                    pcldfmc + ogl, ptaucmc + ogl, pasycmc + ogl,
                    pomgcmc + ogl, ptaormc + ogl,
                    ptaua + obl, pasya + obl, pomga + obl,
                    prmu0[col], adjflux + ob, sfluxzen + og,
                    taug + ogl, taur + ogl, zincflx + og,
                    zcd + og1, zcu + og1, zfd + og1, zfu + og1,
                    ztdbt_nodel_out + og1, ztdbtc_nodel_out + og1,
                    wk + (size_t)idx * (RSW_SPCVMC_WK * n1),
                    wkc + (size_t)idx * (RSW_SPCVMC_WKC * n1));
}

// Accumulate the 14 flux arrays over g-points, sequentially in iw exactly
// like the Fortran band/g loop: one thread per level.
__device__ void rsw_spc_accum_body(int K, int nlayers,
                   const int* __restrict__ ngb,
                   const real* __restrict__ zincflx,
                   const real* __restrict__ zcd, const real* __restrict__ zcu,
                   const real* __restrict__ zfd, const real* __restrict__ zfu,
                   const real* __restrict__ ztdbt_nodel,
                   const real* __restrict__ ztdbtc_nodel,
                   real* pbbfd, real* pbbfu, real* pbbcd, real* pbbcu,
                   real* pbbfddir, real* pbbcddir,
                   real* puvfd, real* puvcd, real* puvfddir, real* puvcddir,
                   real* pnifd, real* pnicd, real* pnifddir, real* pnicddir)
{
    int klev = nlayers;
    // output level K corresponds to two-stream level J = klev - K
    int J = klev - K;

    real bbfd = 0.0f, bbfu = 0.0f, bbcd = 0.0f, bbcu = 0.0f;
    real bbfddir = 0.0f, bbcddir = 0.0f;
    real uvfd = 0.0f, uvcd = 0.0f, uvfddir = 0.0f, uvcddir = 0.0f;
    real nifd = 0.0f, nicd = 0.0f, nifddir = 0.0f, nicddir = 0.0f;

    for (int iw = 0; iw < RSW_NGPT; ++iw) {
        size_t q = (size_t)J + (size_t)(klev + 1) * iw;
        int ibm = ngb[iw] - 15;
        real w = zincflx[iw];
        bbfu = AD(bbfu, MU(w, zfu[q]));
        bbfd = AD(bbfd, MU(w, zfd[q]));
        bbcu = AD(bbcu, MU(w, zcu[q]));
        bbcd = AD(bbcd, MU(w, zcd[q]));
        bbfddir = AD(bbfddir, MU(w, ztdbt_nodel[q]));
        bbcddir = AD(bbcddir, MU(w, ztdbtc_nodel[q]));
        if (ibm >= 10 && ibm <= 13) {
            uvcd = AD(uvcd, MU(w, zcd[q]));
            uvfd = AD(uvfd, MU(w, zfd[q]));
            uvcddir = AD(uvcddir, MU(w, ztdbtc_nodel[q]));
            uvfddir = AD(uvfddir, MU(w, ztdbt_nodel[q]));
        } else if (ibm == 14 || ibm <= 9) {
            nicd = AD(nicd, MU(w, zcd[q]));
            nifd = AD(nifd, MU(w, zfd[q]));
            nicddir = AD(nicddir, MU(w, ztdbtc_nodel[q]));
            nifddir = AD(nifddir, MU(w, ztdbt_nodel[q]));
        }
    }
    pbbfd[K] = bbfd;  pbbfu[K] = bbfu;
    pbbcd[K] = bbcd;  pbbcu[K] = bbcu;
    pbbfddir[K] = bbfddir;  pbbcddir[K] = bbcddir;
    puvfd[K] = uvfd;  puvcd[K] = uvcd;
    puvfddir[K] = uvfddir;  puvcddir[K] = uvcddir;
    pnifd[K] = nifd;  pnicd[K] = nicd;
    pnifddir[K] = nifddir;  pnicddir[K] = nicddir;
}

extern "C" __global__
void rsw_spc_accum(int nlayers,
                   const int* __restrict__ ngb,
                   const real* __restrict__ zincflx,
                   const real* __restrict__ zcd, const real* __restrict__ zcu,
                   const real* __restrict__ zfd, const real* __restrict__ zfu,
                   const real* __restrict__ ztdbt_nodel,
                   const real* __restrict__ ztdbtc_nodel,
                   real* pbbfd, real* pbbfu, real* pbbcd, real* pbbcu,
                   real* pbbfddir, real* pbbcddir,
                   real* puvfd, real* puvcd, real* puvfddir, real* puvcddir,
                   real* pnifd, real* pnicd, real* pnifddir, real* pnicddir)
{
    int K = blockIdx.x * blockDim.x + threadIdx.x;   // 0-based output level
    if (K > nlayers) return;
    rsw_spc_accum_body(K, nlayers, ngb, zincflx, zcd, zcu, zfd, zfu,
                       ztdbt_nodel, ztdbtc_nodel,
                       pbbfd, pbbfu, pbbcd, pbbcu, pbbfddir, pbbcddir,
                       puvfd, puvcd, puvfddir, puvcddir,
                       pnifd, pnicd, pnifddir, pnicddir);
}

// Batched twin: thread -> (col, K); column bases only (pure re-indexing).
extern "C" __global__
void rsw_spc_accum_b(int ncol, int nlayers,
                     const int* __restrict__ ngb,
                     const real* __restrict__ zincflx,   // (ncol, 112)
                     const real* __restrict__ zcd,       // (ncol, 112, n1)
                     const real* __restrict__ zcu,
                     const real* __restrict__ zfd, const real* __restrict__ zfu,
                     const real* __restrict__ ztdbt_nodel,
                     const real* __restrict__ ztdbtc_nodel,
                     real* pbbfd, real* pbbfu, real* pbbcd, real* pbbcu,
                     real* pbbfddir, real* pbbcddir,     // (ncol, n1)
                     real* puvfd, real* puvcd, real* puvfddir, real* puvcddir,
                     real* pnifd, real* pnicd, real* pnifddir, real* pnicddir)
{
    long long W = (long long)nlayers + 1;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * W) return;
    int col = (int)(idx / W);
    int K = (int)(idx % W);
    size_t og = (size_t)col * RSW_NGPT;
    size_t og1 = (size_t)col * (nlayers + 1) * RSW_NGPT;
    size_t o1 = (size_t)col * (nlayers + 1);
    rsw_spc_accum_body(K, nlayers, ngb, zincflx + og,
                       zcd + og1, zcu + og1, zfd + og1, zfu + og1,
                       ztdbt_nodel + og1, ztdbtc_nodel + og1,
                       pbbfd + o1, pbbfu + o1, pbbcd + o1, pbbcu + o1,
                       pbbfddir + o1, pbbcddir + o1,
                       puvfd + o1, puvcd + o1, puvfddir + o1, puvcddir + o1,
                       pnifd + o1, pnicd + o1, pnifddir + o1, pnicddir + o1);
}

// ---------------------------------------------------------------------------
// inatm coldry/wkl arithmetic: one thread per layer.
// ---------------------------------------------------------------------------

__device__ void rsw_inatm_layers_body(int L, int nlayers, real grav,
                      real avogad,
                      const real* __restrict__ plev,     // (nlayers+1)
                      const real* __restrict__ h2ovmr,
                      const real* __restrict__ co2vmr,
                      const real* __restrict__ o3vmr,
                      const real* __restrict__ n2ovmr,
                      const real* __restrict__ ch4vmr,
                      const real* __restrict__ o2vmr,
                      real* pdp, real* coldry, real* wkl)  // wkl (MXMOL, nl) F
{
    const real amd = 28.9660f;
    const real amw = 18.0160f;
    real w1 = h2ovmr[L];
    pdp[L] = SU(plev[L], plev[L + 1]);
    wkl[0 + RSW_MXMOL * L] = w1;
    wkl[1 + RSW_MXMOL * L] = co2vmr[L];
    wkl[2 + RSW_MXMOL * L] = o3vmr[L];
    wkl[3 + RSW_MXMOL * L] = n2ovmr[L];
    wkl[5 + RSW_MXMOL * L] = ch4vmr[L];
    wkl[6 + RSW_MXMOL * L] = o2vmr[L];
    real amm = AD(MU(SU(1.0f, w1), amd), MU(w1, amw));
    coldry[L] = DV(MU(MU(SU(plev[L], plev[L + 1]), 1.0e3f), avogad),
                   MU(MU(MU(1.0e2f, grav), amm), AD(1.0f, w1)));
    for (int imol = 0; imol < RSW_NMOL; ++imol)
        wkl[imol + RSW_MXMOL * L] = MU(coldry[L], wkl[imol + RSW_MXMOL * L]);
}

extern "C" __global__
void rsw_inatm_layers(int nlayers, real grav, real avogad,
                      const real* __restrict__ plev,
                      const real* __restrict__ h2ovmr,
                      const real* __restrict__ co2vmr,
                      const real* __restrict__ o3vmr,
                      const real* __restrict__ n2ovmr,
                      const real* __restrict__ ch4vmr,
                      const real* __restrict__ o2vmr,
                      real* pdp, real* coldry, real* wkl)
{
    int L = blockIdx.x * blockDim.x + threadIdx.x;
    if (L >= nlayers) return;
    rsw_inatm_layers_body(L, nlayers, grav, avogad, plev, h2ovmr, co2vmr,
                          o3vmr, n2ovmr, ch4vmr, o2vmr, pdp, coldry, wkl);
}

// Batched twin: thread -> (col, L); column bases only (pure re-indexing).
extern "C" __global__
void rsw_inatm_layers_b(int ncol, int nlayers, real grav, real avogad,
                        const real* __restrict__ plev,    // (ncol, nl+1)
                        const real* __restrict__ h2ovmr,  // (ncol, nl)
                        const real* __restrict__ co2vmr,
                        const real* __restrict__ o3vmr,
                        const real* __restrict__ n2ovmr,
                        const real* __restrict__ ch4vmr,
                        const real* __restrict__ o2vmr,
                        real* pdp, real* coldry,
                        real* wkl)                        // (ncol, nl, MXMOL)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * nlayers) return;
    int col = (int)(idx / nlayers);
    int L = (int)(idx % nlayers);
    size_t o = (size_t)col * nlayers;
    rsw_inatm_layers_body(L, nlayers, grav, avogad,
                          plev + (size_t)col * (nlayers + 1),
                          h2ovmr + o, co2vmr + o, o3vmr + o, n2ovmr + o,
                          ch4vmr + o, o2vmr + o, pdp + o, coldry + o,
                          wkl + (size_t)col * RSW_MXMOL * nlayers);
}

// Final flux/heating assembly of rrtmg_sw: single block, thread per level.
__device__ void rsw_post_body(int i, int nlayers, real heatfac,
              const real* __restrict__ pdp,
              const real* __restrict__ pbbfu, const real* __restrict__ pbbfd,
              const real* __restrict__ pbbcu, const real* __restrict__ pbbcd,
              real* swhr, real* swhrc)
{
    if (i == nlayers - 1) {
        swhr[i] = 0.0f;
        swhrc[i] = 0.0f;
        return;
    }
    real swnflxc0 = SU(pbbcd[i], pbbcu[i]);
    real swnflxc1 = SU(pbbcd[i + 1], pbbcu[i + 1]);
    real swnflx0 = SU(pbbfd[i], pbbfu[i]);
    real swnflx1 = SU(pbbfd[i + 1], pbbfu[i + 1]);
    real zdpgcp = DV(heatfac, pdp[i]);
    swhrc[i] = MU(SU(swnflxc1, swnflxc0), zdpgcp);
    swhr[i] = MU(SU(swnflx1, swnflx0), zdpgcp);
}

extern "C" __global__
void rsw_post(int nlayers, real heatfac,
              const real* __restrict__ pdp,
              const real* __restrict__ pbbfu, const real* __restrict__ pbbfd,
              const real* __restrict__ pbbcu, const real* __restrict__ pbbcd,
              real* swhr, real* swhrc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nlayers) return;
    rsw_post_body(i, nlayers, heatfac, pdp, pbbfu, pbbfd, pbbcu, pbbcd,
                  swhr, swhrc);
}

// Batched twin: thread -> (col, i); column bases only (pure re-indexing).
extern "C" __global__
void rsw_post_b(int ncol, int nlayers, real heatfac,
                const real* __restrict__ pdp,     // (ncol, nl)
                const real* __restrict__ pbbfu,   // (ncol, nl+1)
                const real* __restrict__ pbbfd,
                const real* __restrict__ pbbcu,
                const real* __restrict__ pbbcd,
                real* swhr, real* swhrc)          // (ncol, nl)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)ncol * nlayers) return;
    int col = (int)(idx / nlayers);
    int i = (int)(idx % nlayers);
    size_t o = (size_t)col * nlayers;
    size_t o1 = (size_t)col * (nlayers + 1);
    rsw_post_body(i, nlayers, heatfac, pdp + o, pbbfu + o1, pbbfd + o1,
                  pbbcu + o1, pbbcd + o1, swhr + o, swhrc + o);
}
