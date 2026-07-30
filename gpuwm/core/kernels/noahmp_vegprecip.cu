// gpuwm/core/kernels/noahmp_vegprecip.cu
//
// Noah-MP PHENOLOGY and PRECIP_HEAT, transcribed from WRF v4.6.1
// phys/module_sf_noahmplsm.F (lines 1255-1358 and 1362-1556) of the pinned
// tree d66e442fccc04111067e29274c9f9eaccc3cef28,
// sha256(module_sf_noahmplsm.F) =
//   bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282
//
// One thread per column.  The acceptance bar is bitwise identity with the
// gfortran build of that module, so:
//
//   * every FP32 operation is written as an explicit round-to-nearest-even
//     intrinsic (__fadd_rn / __fsub_rn / __fmul_rn / __fdiv_rn / __fsqrt_rn).
//     The compiler is therefore never free to contract a multiply and an add
//     into an FMA, which gfortran on x86-64 does not do either.
//
//   * MIN/MAX are written as ternaries with gfortran's tie rule -- minss and
//     maxss return their *second* operand when the operands compare equal, so
//     MAX(a,b) is (a > b) ? a : b.  fmaxf/fminf are not used: they are
//     commutative on signed zeros, and the sign of a zero is observable here.
//
//   * EXP and ** 0.667 are glibc 2.39's expf and powf, not CUDA's.  gfortran
//     emits calls to the C library, and neither CUDA's device expf/powf nor a
//     correctly-rounded expf agrees with glibc often enough to hold a
//     max_ulp-0 gate.  Both are reproduced below in double precision with
//     __dadd_rn / __dsub_rn / __dmul_rn / __fma_rn, matching the FMA-contracted
//     x86-64 ifunc variant that glibc selects on both hosts in this project.
//
//   * the two glibc constant tables live in __constant__ memory and are stored
//     as raw IEEE-754 bit patterns.  ptxas 12.8's constant folder does not
//     honour round-to-nearest-even, so a literal FP array inside a kernel can
//     have its differences mis-folded at compile time; bit patterns in
//     __constant__ memory cannot be folded into arithmetic at all.
//
// Pinned option identity: dveg = 4 and opt_crop = 0.  The DVEG 7/8/9 block and
// every CROPTYPE > 0 disjunct are dead and are not present below.  The host
// must not call these kernels under any other identity; the Python side
// enforces it.
//
// Float32 reference: gpuwm/core/noahmp_vegprecip.py
// Fixtures:          gpuwm/data/noahmp/oracle/noahmp-vegprecip-*.csv

// --------------------------------------------------------------------------
// bit-level helpers
// --------------------------------------------------------------------------
__device__ __forceinline__ double u2d(unsigned long long u)
{
    return __longlong_as_double((long long)u);
}
__device__ __forceinline__ unsigned long long d2u(double d)
{
    return (unsigned long long)__double_as_longlong(d);
}

// --------------------------------------------------------------------------
// glibc 2.39 __exp2f_data / __powf_log2_data, as raw bit patterns.
// POWF_SCALE == 1 because TOINT_INTRINSICS is 0 on x86-64.
// --------------------------------------------------------------------------
__constant__ unsigned long long VP_EXP2F_TAB[32] = {
    0x3FF0000000000000ULL, 0x3FEFD9B0D3158574ULL, 0x3FEFB5586CF9890FULL,
    0x3FEF9301D0125B51ULL, 0x3FEF72B83C7D517BULL, 0x3FEF54873168B9AAULL,
    0x3FEF387A6E756238ULL, 0x3FEF1E9DF51FDEE1ULL, 0x3FEF06FE0A31B715ULL,
    0x3FEEF1A7373AA9CBULL, 0x3FEEDEA64C123422ULL, 0x3FEECE086061892DULL,
    0x3FEEBFDAD5362A27ULL, 0x3FEEB42B569D4F82ULL, 0x3FEEAB07DD485429ULL,
    0x3FEEA47EB03A5585ULL, 0x3FEEA09E667F3BCDULL, 0x3FEE9F75E8EC5F74ULL,
    0x3FEEA11473EB0187ULL, 0x3FEEA589994CCE13ULL, 0x3FEEACE5422AA0DBULL,
    0x3FEEB737B0CDC5E5ULL, 0x3FEEC49182A3F090ULL, 0x3FEED503B23E255DULL,
    0x3FEEE89F995AD3ADULL, 0x3FEEFF76F2FB5E47ULL, 0x3FEF199BDD85529CULL,
    0x3FEF3720DCEF9069ULL, 0x3FEF5818DCFBA487ULL, 0x3FEF7C97337B9B5FULL,
    0x3FEFA4AFA2A490DAULL, 0x3FEFD0765B6E4540ULL
};

// poly[3], poly_scaled[3], shift, shift_scaled, invln2_scaled
__constant__ unsigned long long VP_EXP2F_POLY[3] = {
    0x3FAC6AF84B912394ULL, 0x3FCEBFCE50FAC4F3ULL, 0x3FE62E42FF0C52D6ULL
};
__constant__ unsigned long long VP_EXP2F_POLY_SCALED[3] = {
    0x3EBC6AF84B912394ULL, 0x3F2EBFCE50FAC4F3ULL, 0x3F962E42FF0C52D6ULL
};
__constant__ unsigned long long VP_EXP2F_SHIFT        = 0x4338000000000000ULL;
__constant__ unsigned long long VP_EXP2F_SHIFT_SCALED = 0x42E8000000000000ULL;
__constant__ unsigned long long VP_EXP2F_INVLN2_SCALED = 0x40471547652B82FEULL;

// powf log2 table: 16 pairs of (invc, logc)
__constant__ unsigned long long VP_POWF_INVC[16] = {
    0x3FF661EC79F8F3BEULL, 0x3FF571ED4AAF883DULL, 0x3FF49539F0F010B0ULL,
    0x3FF3C995B0B80385ULL, 0x3FF30D190C8864A5ULL, 0x3FF25E227B0B8EA0ULL,
    0x3FF1BB4A4A1A343FULL, 0x3FF12358F08AE5BAULL, 0x3FF0953F419900A7ULL,
    0x3FF0000000000000ULL, 0x3FEE608CFD9A47ACULL, 0x3FECA4B31F026AA0ULL,
    0x3FEB2036576AFCE6ULL, 0x3FE9C2D163A1AA2DULL, 0x3FE886E6037841EDULL,
    0x3FE767DCF5534862ULL
};
__constant__ unsigned long long VP_POWF_LOGC[16] = {
    0xBFDEFEC65B963019ULL, 0xBFDB0B6832D4FCA4ULL, 0xBFD7418B0A1FB77BULL,
    0xBFD39DE91A6DCF7BULL, 0xBFD01D9BF3F2B631ULL, 0xBFC97C1D1B3B7AF0ULL,
    0xBFC2F9E393AF3C9FULL, 0xBFB960CBBF788D5CULL, 0xBFAA6F9DB6475FCEULL,
    0x0000000000000000ULL, 0x3FB338CA9F24F53DULL, 0x3FC476A9543891BAULL,
    0x3FCE840B4AC4E4D2ULL, 0x3FD40645F0C6651CULL, 0x3FD88E9C2C1B9FF8ULL,
    0x3FDCE0A44EB17BCCULL
};
__constant__ unsigned long long VP_POWF_POLY[5] = {
    0x3FD27616C9496E0BULL, 0xBFD71969A075C67AULL, 0x3FDEC70A6CA7BADDULL,
    0xBFE7154748BEF6C8ULL, 0x3FF71547652AB82BULL
};

// Round a double to binary32, INCLUDING into the subnormal range.
//
// `__double2float_rn` flushes a subnormal result to zero on this toolchain --
// measured on sm_120 with CUDA 13.0, and not changed by `--ftz=false`, which
// only governs FP32 arithmetic and not the double->float conversion.  glibc's
// expf and powf do produce subnormals, gfortran at -O0 leaves MXCSR's FTZ/DAZ
// clear, and `gpuwm.core.noahmp_libm` -- verified against the live glibc 2.39
// over 1,106,247,680 inputs -- produces them too.  So the hardware conversion
// is a divergence from the authority in a narrow but reachable band:
// expf(x) for x in [-103.616, -87.337), and powf wherever y*log2(x) lands in
// the same place.  ENERGY's RHSUR (:2203) evaluates exp(PSI*GRAV/(RW*TG)) on
// very dry soil and can reach it.
//
// The correctly rounded subnormal is recovered exactly rather than
// approximated: a binary32 subnormal with mantissa bits `m` is exactly
// m * 2^-149, `y * 2^149` is an exact scaling in binary64 for any y in this
// band, and `rint` rounds to nearest with ties to even -- which is the same
// rounding the conversion owes.  m == 2^23 is the carry into the smallest
// normal and its bit pattern 0x00800000 is already that number, so the carry
// needs no special case.
__device__ float nmp_d2f_rn(double y)
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

// --------------------------------------------------------------------------
// glibc 2.39 expf  (sysdeps/ieee754/flt-32/e_expf.c)
// --------------------------------------------------------------------------
__device__ float vp_glibc_expf(float x)
{
    unsigned int ix = __float_as_uint(x);
    unsigned int abstop = (ix >> 20) & 0x7FFu;

    if (abstop >= 0x42Bu) {                       // top12(88.0f)
        if (ix == 0xFF800000u) return 0.0f;       // -inf
        if (abstop >= 0x7F8u) return __fadd_rn(x, x);   // +-inf or NaN
        if (x >  0x1.62e42ep6f) return __uint_as_float(0x7F800000u);  // log(2^128)
        if (x < -0x1.9fe368p6f) return 0.0f;                          // log(2^-150)
        // the remaining tail falls through to the main path, as in glibc
    }

    const double SHIFT   = u2d(VP_EXP2F_SHIFT);
    const double INVLN2N = u2d(VP_EXP2F_INVLN2_SCALED);

    double xd = (double)x;                        // exact
    double z  = __dmul_rn(INVLN2N, xd);
    double kd = __dadd_rn(z, SHIFT);
    unsigned long long ki = d2u(kd);
    kd = __dsub_rn(kd, SHIFT);
    double r = __dsub_rn(z, kd);

    unsigned long long t = VP_EXP2F_TAB[ki & 31ULL];
    t += (ki << 47);
    double s = u2d(t);

    double c0 = u2d(VP_EXP2F_POLY_SCALED[0]);
    double c1 = u2d(VP_EXP2F_POLY_SCALED[1]);
    double c2 = u2d(VP_EXP2F_POLY_SCALED[2]);

    double z2 = __fma_rn(c0, r, c1);
    double r2 = __dmul_rn(r, r);
    double y  = __fma_rn(c2, r, 1.0);
    y = __fma_rn(z2, r2, y);
    y = __dmul_rn(y, s);
    return nmp_d2f_rn(y);
}

// --------------------------------------------------------------------------
// glibc 2.39 powf  (sysdeps/ieee754/flt-32/e_powf.c)
//
// Only the domain Noah-MP reaches is transcribed: a finite non-negative base
// and a finite positive exponent.  A negative base would need the
// checkint/sign_bias path, which cannot arise from
// ``MIN(FWET,1.0) ** 0.667`` because FWET is a ratio of two MAX(...,0)
// quantities; a subnormal base would need the normalization path.  Both
// return NaN here rather than a value this kernel cannot vouch for -- the
// Python side raises on the same inputs, and a NaN can never pass the gate.
// --------------------------------------------------------------------------
__device__ __forceinline__ double vp_log2_inline(unsigned int ix)
{
    unsigned int tmp = ix - 0x3F330000u;
    int i = (int)((tmp >> 19) & 15u);
    unsigned int top = tmp & 0xFF800000u;
    unsigned int iz = ix - top;
    int k = ((int)top) >> 23;                     // arithmetic shift

    double invc = u2d(VP_POWF_INVC[i]);
    double logc = u2d(VP_POWF_LOGC[i]);
    double z = (double)__uint_as_float(iz);

    double r  = __fma_rn(z, invc, -1.0);
    double y0 = __dadd_rn(logc, (double)k);

    double a0 = u2d(VP_POWF_POLY[0]);
    double a1 = u2d(VP_POWF_POLY[1]);
    double a2 = u2d(VP_POWF_POLY[2]);
    double a3 = u2d(VP_POWF_POLY[3]);
    double a4 = u2d(VP_POWF_POLY[4]);

    double r2 = __dmul_rn(r, r);
    double y  = __fma_rn(a0, r, a1);
    double p  = __fma_rn(a2, r, a3);
    double r4 = __dmul_rn(r2, r2);
    double q  = __fma_rn(a4, r, y0);
    q = __fma_rn(p, r2, q);
    y = __fma_rn(y, r4, q);
    return y;
}

__device__ __forceinline__ float vp_exp2_inline(double xd)
{
    const double SHIFT = u2d(VP_EXP2F_SHIFT_SCALED);
    double kd = __dadd_rn(xd, SHIFT);
    unsigned long long ki = d2u(kd);
    kd = __dsub_rn(kd, SHIFT);
    double r = __dsub_rn(xd, kd);

    unsigned long long t = VP_EXP2F_TAB[ki & 31ULL];
    t += (ki << 47);
    double s = u2d(t);

    double c0 = u2d(VP_EXP2F_POLY[0]);
    double c1 = u2d(VP_EXP2F_POLY[1]);
    double c2 = u2d(VP_EXP2F_POLY[2]);

    double z = __fma_rn(c0, r, c1);
    double r2 = __dmul_rn(r, r);
    double y = __fma_rn(c2, r, 1.0);
    y = __fma_rn(z, r2, y);
    y = __dmul_rn(y, s);
    return nmp_d2f_rn(y);
}

__device__ __forceinline__ bool vp_zeroinfnan(unsigned int i)
{
    return (2u * i - 1u) >= (2u * 0x7F800000u - 1u);
}

__device__ float vp_glibc_powf(float x, float y)
{
    unsigned int ix = __float_as_uint(x);
    unsigned int iy = __float_as_uint(y);

    if ((ix - 0x00800000u) >= (0x7F800000u - 0x00800000u) || vp_zeroinfnan(iy)) {
        if (vp_zeroinfnan(iy)) {
            if (2u * iy == 0u) return 1.0f;
            if (ix == 0x3F800000u) return 1.0f;
            if (2u * ix > 2u * 0x7F800000u || 2u * iy > 2u * 0x7F800000u)
                return __fadd_rn(x, y);
            if (2u * ix == 2u * 0x3F800000u) return 1.0f;
            if ((2u * ix < 2u * 0x3F800000u) == !(iy & 0x80000000u)) return 0.0f;
            return __fmul_rn(y, y);
        }
        if (vp_zeroinfnan(ix)) {
            if (ix & 0x80000000u) return __uint_as_float(0x7FC00000u);  // out of domain
            if (2u * ix == 0u && (iy & 0x80000000u))
                return __uint_as_float(0x7FC00000u);                    // out of domain
            float x2 = __fmul_rn(x, x);
            return (iy & 0x80000000u) ? __fdiv_rn(1.0f, x2) : x2;
        }
        // finite negative base, or subnormal base: outside the transcribed
        // domain, and unreachable from these leaves.
        return __uint_as_float(0x7FC00000u);
    }

    double logx  = vp_log2_inline(ix);
    double ylogx = __dmul_rn((double)y, logx);
    // |y*log2(x)| >= 126 would need glibc's overflow/underflow shims; FWET is
    // in [0,1] and the exponent is 0.667, so ylogx is in [-126, 0] by
    // construction for every reachable argument.
    return vp_exp2_inline(ylogx);
}

// --------------------------------------------------------------------------
// gfortran MIN / MAX: minss / maxss return the second operand on a tie
// --------------------------------------------------------------------------
__device__ __forceinline__ float vp_fmax(float a, float b) { return (a > b) ? a : b; }
__device__ __forceinline__ float vp_fmin(float a, float b) { return (a < b) ? a : b; }

// --------------------------------------------------------------------------
// PHENOLOGY, module_sf_noahmplsm.F:1255, at dveg = 4 and croptype = 0
// --------------------------------------------------------------------------
extern "C" __global__ void noahmp_phenology(
    int n,
    const int   *__restrict__ vegtyp,
    const int   *__restrict__ yearlen,
    const int   *__restrict__ iswater,
    const int   *__restrict__ isbarren,
    const int   *__restrict__ isice,
    const int   *__restrict__ urban_flag,
    const float *__restrict__ snowh_a,
    const float *__restrict__ tv_a,
    const float *__restrict__ lat_a,
    const float *__restrict__ julian_a,
    const float *__restrict__ hvt_a,
    const float *__restrict__ hvb_a,
    const float *__restrict__ tmin_a,
    const float *__restrict__ laim_a,      // n x 12, row major
    const float *__restrict__ saim_a,
    float *__restrict__ lai_o,
    float *__restrict__ sai_o,
    float *__restrict__ elai_o,
    float *__restrict__ esai_o,
    float *__restrict__ igs_o,
    float *__restrict__ fb_o)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;

    const float snowh  = snowh_a[c];
    const float tv     = tv_a[c];
    const float lat    = lat_a[c];
    const float julian = julian_a[c];
    const float hvt    = hvt_a[c];
    const float hvb    = hvb_a[c];
    const float tmin   = tmin_a[c];
    const int   ylen   = yearlen[c];

    float rylen = __int2float_rn(ylen);

    float day;
    if (lat >= 0.0f) {
        day = julian;
    } else {
        float half = __fmul_rn(0.5f, rylen);
        day = fmodf(__fadd_rn(julian, half), rylen);
    }

    float t = __fdiv_rn(__fmul_rn(12.0f, day), rylen);
    int it1 = __float2int_rz(__fadd_rn(t, 0.5f));
    int it2 = it1 + 1;
    float wt1 = __fsub_rn(__fadd_rn(__int2float_rn(it1), 0.5f), t);  // unclamped IT1
    float wt2 = __fsub_rn(1.0f, wt1);
    if (it1 < 1)  it1 = 12;
    if (it2 > 12) it2 = 1;

    const float *laim = laim_a + (size_t)c * 12;
    const float *saim = saim_a + (size_t)c * 12;

    float lai = __fadd_rn(__fmul_rn(wt1, laim[it1 - 1]),
                          __fmul_rn(wt2, laim[it2 - 1]));
    float sai = __fadd_rn(__fmul_rn(wt1, saim[it1 - 1]),
                          __fmul_rn(wt2, saim[it2 - 1]));

    if (sai < 0.05f) sai = 0.0f;
    if (lai < 0.05f || sai == 0.0f) lai = 0.0f;

    if (vegtyp[c] == iswater[c] || vegtyp[c] == isbarren[c] ||
        vegtyp[c] == isice[c]   || urban_flag[c] != 0) {
        lai = 0.0f;
        sai = 0.0f;
    }

    float span = __fsub_rn(hvt, hvb);
    float db = vp_fmin(vp_fmax(__fsub_rn(snowh, hvb), 0.0f), span);
    float fb = __fdiv_rn(db, vp_fmax(1.0e-06f, span));

    if (hvt > 0.0f && hvt <= 1.0f) {
        float snowhc = __fmul_rn(hvt, vp_glibc_expf(-__fdiv_rn(snowh, 0.2f)));
        fb = __fdiv_rn(vp_fmin(snowh, snowhc), snowhc);
    }

    float elai = __fmul_rn(lai, __fsub_rn(1.0f, fb));
    float esai = __fmul_rn(sai, __fsub_rn(1.0f, fb));
    if (esai < 0.05f) esai = 0.0f;
    if (elai < 0.05f || esai == 0.0f) elai = 0.0f;

    float igs = (tv > tmin) ? 1.0f : 0.0f;

    lai_o[c]  = lai;
    sai_o[c]  = sai;
    elai_o[c] = elai;
    esai_o[c] = esai;
    igs_o[c]  = igs;
    fb_o[c]   = fb;
}

// --------------------------------------------------------------------------
// PRECIP_HEAT, module_sf_noahmplsm.F:1362
// --------------------------------------------------------------------------
#define VP_CWAT_PER_1000 4188.0f     // CWAT  = 4.188e6, folded exactly
#define VP_CICE_PER_1000 2094.0f     // CICE  = 2.094e6, folded exactly
#define VP_TFRZ          273.16f

extern "C" __global__ void noahmp_precip_heat(
    int n,
    const int   *__restrict__ ist_a,
    const float *__restrict__ dt_a,
    const float *__restrict__ uu_a,
    const float *__restrict__ vv_a,
    const float *__restrict__ elai_a,
    const float *__restrict__ esai_a,
    const float *__restrict__ fveg_a,
    const float *__restrict__ bdfall_a,
    const float *__restrict__ rain_a,
    const float *__restrict__ snow_a,
    const float *__restrict__ fp_a,
    const float *__restrict__ canliq_a,
    const float *__restrict__ canice_a,
    const float *__restrict__ tv_a,
    const float *__restrict__ sfctmp_a,
    const float *__restrict__ tg_a,
    const float *__restrict__ ch2op_a,
    float *__restrict__ canliq_o,
    float *__restrict__ canice_o,
    float *__restrict__ qintr_o,
    float *__restrict__ qdripr_o,
    float *__restrict__ qthror_o,
    float *__restrict__ qints_o,
    float *__restrict__ qdrips_o,
    float *__restrict__ qthros_o,
    float *__restrict__ pahv_o,
    float *__restrict__ pahg_o,
    float *__restrict__ pahb_o,
    float *__restrict__ qrain_o,
    float *__restrict__ qsnow_o,
    float *__restrict__ snowhin_o,
    float *__restrict__ fwet_o,
    float *__restrict__ cmc_o)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;

    const int   ist    = ist_a[c];
    const float dt     = dt_a[c];
    const float uu     = uu_a[c];
    const float vv     = vv_a[c];
    const float elai   = elai_a[c];
    const float esai   = esai_a[c];
    const float fveg   = fveg_a[c];
    const float bdfall = bdfall_a[c];
    const float rain   = rain_a[c];
    const float snow   = snow_a[c];
    const float fp     = fp_a[c];
    const float tv     = tv_a[c];
    const float sfctmp = sfctmp_a[c];
    const float tg     = tg_a[c];
    const float ch2op  = ch2op_a[c];

    float canliq = canliq_a[c];
    float canice = canice_a[c];

    float qintr  = 0.0f;
    float qdripr = 0.0f;
    float qthror = 0.0f;
    float qints  = 0.0f;
    float qdrips = 0.0f;
    float qthros = 0.0f;
    float icedrip = 0.0f;

    float lsai = __fadd_rn(elai, esai);

    // ------------------------- liquid water ------------------------------
    float maxliq = __fmul_rn(__fmul_rn(fveg, ch2op), lsai);

    if (lsai > 0.0f) {
        qintr = __fmul_rn(__fmul_rn(fveg, rain), fp);
        float cap = __fmul_rn(
            __fdiv_rn(__fsub_rn(maxliq, canliq), dt),
            __fsub_rn(1.0f, vp_glibc_expf(-__fdiv_rn(__fmul_rn(rain, dt), maxliq))));
        qintr = vp_fmin(qintr, cap);
        qintr = vp_fmax(qintr, 0.0f);
        qdripr = __fsub_rn(__fmul_rn(fveg, rain), qintr);
        qthror = __fmul_rn(__fsub_rn(1.0f, fveg), rain);
        canliq = vp_fmax(0.0f, __fadd_rn(canliq, __fmul_rn(qintr, dt)));
    } else {
        qintr = 0.0f;
        qdripr = 0.0f;
        qthror = rain;
        if (canliq > 0.0f) {
            qdripr = __fadd_rn(qdripr, __fdiv_rn(canliq, dt));
            canliq = 0.0f;
        }
    }

    float pah_ac = __fmul_rn(__fmul_rn(__fmul_rn(fveg, rain), VP_CWAT_PER_1000),
                             __fsub_rn(sfctmp, tv));
    float pah_cg = __fmul_rn(__fmul_rn(qdripr, VP_CWAT_PER_1000),
                             __fsub_rn(tv, tg));
    float pah_ag = __fmul_rn(__fmul_rn(qthror, VP_CWAT_PER_1000),
                             __fsub_rn(sfctmp, tg));

    // --------------------------- canopy ice ------------------------------
    float maxsno = __fmul_rn(
        __fmul_rn(__fmul_rn(fveg, 6.6f),
                  __fadd_rn(0.27f, __fdiv_rn(46.0f, bdfall))),
        lsai);

    if (lsai > 0.0f) {
        qints = __fmul_rn(__fmul_rn(fveg, snow), fp);
        float cap = __fmul_rn(
            __fdiv_rn(__fsub_rn(maxsno, canice), dt),
            __fsub_rn(1.0f, vp_glibc_expf(-__fdiv_rn(__fmul_rn(snow, dt), maxsno))));
        qints = vp_fmin(qints, cap);
        qints = vp_fmax(qints, 0.0f);
        float ft = vp_fmax(0.0f, __fdiv_rn(__fsub_rn(tv, 270.15f), 1.87e5f));
        float fv = __fdiv_rn(
            __fsqrt_rn(__fadd_rn(__fmul_rn(uu, uu), __fmul_rn(vv, vv))), 1.56e5f);
        icedrip = __fmul_rn(vp_fmax(0.0f, canice), __fadd_rn(fv, ft));
        icedrip = vp_fmin(__fadd_rn(__fdiv_rn(canice, dt), qints), icedrip);
        qdrips = __fadd_rn(__fsub_rn(__fmul_rn(fveg, snow), qints), icedrip);
        qthros = __fmul_rn(__fsub_rn(1.0f, fveg), snow);
        canice = vp_fmax(0.0f,
                         __fadd_rn(canice, __fmul_rn(__fsub_rn(qints, icedrip), dt)));
    } else {
        qints = 0.0f;
        qdrips = 0.0f;
        qthros = snow;
        if (canice > 0.0f) {
            qdrips = __fadd_rn(qdrips, __fdiv_rn(canice, dt));
            canice = 0.0f;
        }
    }

    float fwet;
    if (canice > 0.0f)
        fwet = __fdiv_rn(vp_fmax(0.0f, canice), vp_fmax(maxsno, 1.0e-06f));
    else
        fwet = __fdiv_rn(vp_fmax(0.0f, canliq), vp_fmax(maxliq, 1.0e-06f));
    fwet = vp_glibc_powf(vp_fmin(fwet, 1.0f), 0.667f);

    float cmc = __fadd_rn(canliq, canice);

    pah_ac = __fadd_rn(pah_ac,
        __fmul_rn(__fmul_rn(__fmul_rn(fveg, snow), VP_CICE_PER_1000),
                  __fsub_rn(sfctmp, tv)));
    pah_cg = __fadd_rn(pah_cg,
        __fmul_rn(__fmul_rn(qdrips, VP_CICE_PER_1000), __fsub_rn(tv, tg)));
    pah_ag = __fadd_rn(pah_ag,
        __fmul_rn(__fmul_rn(qthros, VP_CICE_PER_1000), __fsub_rn(sfctmp, tg)));

    float pahv = __fsub_rn(pah_ac, pah_cg);
    float pahg = pah_cg;
    float pahb = pah_ag;

    if (fveg > 0.0f && fveg < 1.0f) {
        pahg = __fdiv_rn(pahg, fveg);
        pahb = __fdiv_rn(pahb, __fsub_rn(1.0f, fveg));
    } else if (fveg <= 0.0f) {
        pahb = __fadd_rn(pahg, pahb);
        pahg = 0.0f;
        pahv = 0.0f;
    } else if (fveg >= 1.0f) {
        pahb = 0.0f;
    }

    pahv = vp_fmax(pahv, -20.0f);
    pahv = vp_fmin(pahv,  20.0f);
    pahg = vp_fmax(pahg, -20.0f);
    pahg = vp_fmin(pahg,  20.0f);
    pahb = vp_fmax(pahb, -20.0f);
    pahb = vp_fmin(pahb,  20.0f);

    float qrain = __fadd_rn(qdripr, qthror);
    float qsnow = __fadd_rn(qdrips, qthros);
    float snowhin = __fdiv_rn(qsnow, bdfall);

    if (ist == 2 && tg > VP_TFRZ) {
        qsnow = 0.0f;
        snowhin = 0.0f;
    }

    canliq_o[c]  = canliq;
    canice_o[c]  = canice;
    qintr_o[c]   = qintr;
    qdripr_o[c]  = qdripr;
    qthror_o[c]  = qthror;
    qints_o[c]   = qints;
    qdrips_o[c]  = qdrips;
    qthros_o[c]  = qthros;
    pahv_o[c]    = pahv;
    pahg_o[c]    = pahg;
    pahb_o[c]    = pahb;
    qrain_o[c]   = qrain;
    qsnow_o[c]   = qsnow;
    snowhin_o[c] = snowhin;
    fwet_o[c]    = fwet;
    cmc_o[c]     = cmc;
}
