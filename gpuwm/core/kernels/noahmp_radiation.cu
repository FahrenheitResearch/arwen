// gpuwm/core/kernels/noahmp_radiation.cu
//
// CUDA half of the Noah-MP shortwave-radiation port.  Bitwise-equal to
// gpuwm/core/noahmp_radiation.py and to the WRF-4.6.1 oracle
// (tree d66e442fccc04111067e29274c9f9eaccc3cef28,
//  sha256(module_sf_noahmplsm.F) = bd592a5b7db29000e715250e3a7c779ffb5e0dcc
//  356f6b5a7d9e1c9f69c55282), option identity dveg=4 opt_rad=3 opt_alb=2.
//
// Three rules make that possible and none of them is optional:
//
// 1. Every float32 and float64 operation uses an explicit rounding intrinsic
//    (__fadd_rn/__fsub_rn/__fmul_rn/__fdiv_rn/__fsqrt_rn, __dadd_rn/__dsub_rn/
//    __dmul_rn/__fma_rn).  That pins the hardware rounding mode AND makes
//    nvcc's contraction pass a no-op, so -fmad=true cannot fuse a site the
//    Fortran did not fuse.
// 2. Every constant lives in __constant__ memory as a bit pattern.  ptxas
//    12.8's constant folder does not honour round-to-nearest-even on literal
//    arrays, so a table of FP32 literals can have its differences mis-folded
//    at compile time.  __fsub_rn pins the hardware, not the folder.
// 3. EXP/LOG/** are glibc calls in the oracle, so glibc_expf/glibc_logf/
//    glibc_powf below transcribe glibc 2.39's own algorithms.  CUDA's expf,
//    logf, powf, __expf and __logf are all different functions and none of
//    them can hold a max_ulp-0 gate.
//
// Dead under the pinned identity and deliberately absent: SNOWALB_BATS,
// TWOSTREAM's OPT_RAD=1 crown geometry and OPT_RAD=2.

__constant__ unsigned long long C_EXP2F_TAB[32] = {
    0x3FF0000000000000ULL, 0x3FEFD9B0D3158574ULL, 0x3FEFB5586CF9890FULL, 0x3FEF9301D0125B51ULL,
    0x3FEF72B83C7D517BULL, 0x3FEF54873168B9AAULL, 0x3FEF387A6E756238ULL, 0x3FEF1E9DF51FDEE1ULL,
    0x3FEF06FE0A31B715ULL, 0x3FEEF1A7373AA9CBULL, 0x3FEEDEA64C123422ULL, 0x3FEECE086061892DULL,
    0x3FEEBFDAD5362A27ULL, 0x3FEEB42B569D4F82ULL, 0x3FEEAB07DD485429ULL, 0x3FEEA47EB03A5585ULL,
    0x3FEEA09E667F3BCDULL, 0x3FEE9F75E8EC5F74ULL, 0x3FEEA11473EB0187ULL, 0x3FEEA589994CCE13ULL,
    0x3FEEACE5422AA0DBULL, 0x3FEEB737B0CDC5E5ULL, 0x3FEEC49182A3F090ULL, 0x3FEED503B23E255DULL,
    0x3FEEE89F995AD3ADULL, 0x3FEEFF76F2FB5E47ULL, 0x3FEF199BDD85529CULL, 0x3FEF3720DCEF9069ULL,
    0x3FEF5818DCFBA487ULL, 0x3FEF7C97337B9B5FULL, 0x3FEFA4AFA2A490DAULL, 0x3FEFD0765B6E4540ULL,
};
__constant__ unsigned long long C_EXP2F_MISC[3] = {
    0x42E8000000000000ULL, 0x4338000000000000ULL, 0x40471547652B82FEULL,
};
__constant__ unsigned long long C_EXP2F_POLY[3] = {
    0x3FAC6AF84B912394ULL, 0x3FCEBFCE50FAC4F3ULL, 0x3FE62E42FF0C52D6ULL,
};
__constant__ unsigned long long C_EXP2F_POLY_SCALED[3] = {
    0x3EBC6AF84B912394ULL, 0x3F2EBFCE50FAC4F3ULL, 0x3F962E42FF0C52D6ULL,
};
__constant__ unsigned long long C_POWF_LOG2_TAB[32] = {
    0x3FF661EC79F8F3BEULL, 0xBFDEFEC65B963019ULL,
    0x3FF571ED4AAF883DULL, 0xBFDB0B6832D4FCA4ULL,
    0x3FF49539F0F010B0ULL, 0xBFD7418B0A1FB77BULL,
    0x3FF3C995B0B80385ULL, 0xBFD39DE91A6DCF7BULL,
    0x3FF30D190C8864A5ULL, 0xBFD01D9BF3F2B631ULL,
    0x3FF25E227B0B8EA0ULL, 0xBFC97C1D1B3B7AF0ULL,
    0x3FF1BB4A4A1A343FULL, 0xBFC2F9E393AF3C9FULL,
    0x3FF12358F08AE5BAULL, 0xBFB960CBBF788D5CULL,
    0x3FF0953F419900A7ULL, 0xBFAA6F9DB6475FCEULL,
    0x3FF0000000000000ULL, 0x0000000000000000ULL,
    0x3FEE608CFD9A47ACULL, 0x3FB338CA9F24F53DULL,
    0x3FECA4B31F026AA0ULL, 0x3FC476A9543891BAULL,
    0x3FEB2036576AFCE6ULL, 0x3FCE840B4AC4E4D2ULL,
    0x3FE9C2D163A1AA2DULL, 0x3FD40645F0C6651CULL,
    0x3FE886E6037841EDULL, 0x3FD88E9C2C1B9FF8ULL,
    0x3FE767DCF5534862ULL, 0x3FDCE0A44EB17BCCULL,
};
__constant__ unsigned long long C_POWF_LOG2_POLY[5] = {
    0x3FD27616C9496E0BULL, 0xBFD71969A075C67AULL, 0x3FDEC70A6CA7BADDULL,
    0xBFE7154748BEF6C8ULL, 0x3FF71547652AB82BULL,
};
__constant__ unsigned long long C_LOGF_TAB[32] = {
    0x3FF661EC79F8F3BEULL, 0xBFD57BF7808CAADEULL,
    0x3FF571ED4AAF883DULL, 0xBFD2BEF0A7C06DDBULL,
    0x3FF49539F0F010B0ULL, 0xBFD01EAE7F513A67ULL,
    0x3FF3C995B0B80385ULL, 0xBFCB31D8A68224E9ULL,
    0x3FF30D190C8864A5ULL, 0xBFC6574F0AC07758ULL,
    0x3FF25E227B0B8EA0ULL, 0xBFC1AA2BC79C8100ULL,
    0x3FF1BB4A4A1A343FULL, 0xBFBA4E76CE8C0E5EULL,
    0x3FF12358F08AE5BAULL, 0xBFB1973C5A611CCCULL,
    0x3FF0953F419900A7ULL, 0xBFA252F438E10C1EULL,
    0x3FF0000000000000ULL, 0x0000000000000000ULL,
    0x3FEE608CFD9A47ACULL, 0x3FAAA5AA5DF25984ULL,
    0x3FECA4B31F026AA0ULL, 0x3FBC5E53AA362EB4ULL,
    0x3FEB2036576AFCE6ULL, 0x3FC526E57720DB08ULL,
    0x3FE9C2D163A1AA2DULL, 0x3FCBC2860D224770ULL,
    0x3FE886E6037841EDULL, 0x3FD1058BC8A07EE1ULL,
    0x3FE767DCF5534862ULL, 0x3FD4043057B6EE09ULL,
};
__constant__ unsigned long long C_LOGF_MISC[4] = {
    0x3FE62E42FEFA39EFULL, 0xBFD00EA348B88334ULL, 0x3FD5575B0BE00B6AULL, 0xBFDFFFFEF20A4123ULL,
};
__constant__ unsigned long long C_LIMITS[4] = {
    0x40562E42E0000000ULL, 0xC059FE3680000000ULL, 0x405FFFFFFFD1D571ULL, 0x405F800000000000ULL,
};
// float32 constants, as bit patterns: ptxas 12.8's constant folder
// does not honour round-to-nearest-even on literal arrays.
__constant__ unsigned int C_F32[24] = {
    0x00000000u, 0x3F800000u, 0x3F000000u, 0x40000000u,
    0x4388947Bu, 0x3F0CCCCDu, 0x3C23D70Au, 0x45610000u,
    0x3F570A3Du, 0x3DE147AEu, 0x3ECCCCCDu, 0x3D75C28Fu,
    0x3E19999Au, 0x3FD9999Au, 0x3A83126Fu, 0xBECCCCCDu,
    0x3F19999Au, 0x3F220C4Au, 0x3EA8F5C3u, 0x3F608312u,
    0x358637BDu, 0xBC23D70Au, 0x4B000000u, 0x80000000u,
};
#define K_ZERO __uint_as_float(C_F32[0])
#define K_ONE __uint_as_float(C_F32[1])
#define K_HALF __uint_as_float(C_F32[2])
#define K_TWO __uint_as_float(C_F32[3])
#define K_TFRZ __uint_as_float(C_F32[4])
#define K_P55 __uint_as_float(C_F32[5])
#define K_P01 __uint_as_float(C_F32[6])
#define K_T3600 __uint_as_float(C_F32[7])
#define K_P84 __uint_as_float(C_F32[8])
#define K_P11 __uint_as_float(C_F32[9])
#define K_P40 __uint_as_float(C_F32[10])
#define K_P06 __uint_as_float(C_F32[11])
#define K_P15 __uint_as_float(C_F32[12])
#define K_P17 __uint_as_float(C_F32[13])
#define K_P001 __uint_as_float(C_F32[14])
#define K_NP4 __uint_as_float(C_F32[15])
#define K_P6 __uint_as_float(C_F32[16])
#define K_P633 __uint_as_float(C_F32[17])
#define K_P330 __uint_as_float(C_F32[18])
#define K_P877 __uint_as_float(C_F32[19])
#define K_MPE __uint_as_float(C_F32[20])
#define K_NP01 __uint_as_float(C_F32[21])
#define K_EIGHT388608 __uint_as_float(C_F32[22])
#define K_NEG0 __uint_as_float(C_F32[23])

// --------------------------------------------------------------------------
// rounding-pinned primitives
// --------------------------------------------------------------------------
#define FADD(a, b) __fadd_rn((a), (b))
#define FSUB(a, b) __fsub_rn((a), (b))
#define FMUL(a, b) __fmul_rn((a), (b))
#define FDIV(a, b) __fdiv_rn((a), (b))
#define FSQRT(a)   __fsqrt_rn(a)

#define DADD(a, b) __dadd_rn((a), (b))
#define DSUB(a, b) __dsub_rn((a), (b))
#define DMUL(a, b) __dmul_rn((a), (b))
#define DFMA(a, b, c) __fma_rn((a), (b), (c))

// Python's builtin max/min, which the CPU transcription uses, return the
// FIRST operand on a tie.  Fortran MAX/MIN may return either, and the values
// are equal, so this is the tie-break to mirror.
__device__ __forceinline__ float f_max(float a, float b) { return (b > a) ? b : a; }
__device__ __forceinline__ float f_min(float a, float b) { return (b < a) ? b : a; }

__device__ __forceinline__ float f_sign(float a, float b)
{
    // Fortran SIGN(A, B): gfortran lowers this to a sign-bit splice, so the
    // sign of a negative zero is honoured.
    unsigned int ua = __float_as_uint(a) & 0x7FFFFFFFu;
    unsigned int sb = __float_as_uint(b) & 0x80000000u;
    return __uint_as_float(ua | sb);
}

// Round a double to binary32, INCLUDING into the subnormal range.
//
// `__double2float_rn` flushes a subnormal result to zero on this toolchain --
// measured on sm_120 with CUDA 13.0, and asking for `--ftz=false` does not
// change it, because CuPy appends `-ftz=true` after the caller's options and
// the compiler honours the last occurrence.  The flush is the flag, not the
// SM: the conversion instruction has no flush of its own, so under `-ftz=true`
// the compiler emits an extra multiply after it to produce one, and built
// without the append the same conversion keeps the subnormal.  glibc's
// expf and powf do produce subnormals, gfortran at -O0 leaves MXCSR's FTZ/DAZ
// clear, and `gpuwm.core.noahmp_libm` -- verified against the live glibc 2.39
// over 1,106,247,680 inputs -- produces them too.  So the conversion as
// compiled here is a divergence from the authority in a narrow but reachable
// band:
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
// glibc 2.39 expf  --  sysdeps/ieee754/flt-32/e_expf.c, FMA variant
// --------------------------------------------------------------------------
__device__ float glibc_expf(float x)
{
    unsigned int ix = __float_as_uint(x);
    unsigned int abstop = (ix >> 20) & 0x7FFu;

    if (abstop >= 0x42Bu) {                       // top12(88.0f) & 0x7ff
        if (ix == 0xFF800000u) return 0.0f;       // -inf
        if (abstop >= 0x7F8u) return FADD(x, x);  // +-inf or NaN
        if ((double) x > __longlong_as_double(C_LIMITS[0]))
            return __uint_as_float(0x7F800000u);  // overflow
        if ((double) x < __longlong_as_double(C_LIMITS[1]))
            return 0.0f;                          // underflow
    }

    double shift = __longlong_as_double(C_EXP2F_MISC[1]);
    double xd = (double) x;
    double z = DMUL(__longlong_as_double(C_EXP2F_MISC[2]), xd);
    double kd = DADD(z, shift);
    unsigned long long ki = (unsigned long long) __double_as_longlong(kd);
    kd = DSUB(kd, shift);
    double r = DSUB(z, kd);

    unsigned long long t = C_EXP2F_TAB[ki & 31ULL];
    t += (ki << (52 - 5));
    double s = __longlong_as_double((long long) t);
    double c0 = __longlong_as_double(C_EXP2F_POLY_SCALED[0]);
    double c1 = __longlong_as_double(C_EXP2F_POLY_SCALED[1]);
    double c2 = __longlong_as_double(C_EXP2F_POLY_SCALED[2]);
    double z2 = DFMA(c0, r, c1);
    double r2 = DMUL(r, r);
    double y = DFMA(c2, r, 1.0);
    y = DFMA(z2, r2, y);
    y = DMUL(y, s);
    return nmp_d2f_rn(y);
}

// --------------------------------------------------------------------------
// glibc 2.39 logf  --  sysdeps/ieee754/flt-32/e_logf.c, FMA variant.
// Operation order read off the disassembly of __logf_fma, including the
// single FMA that computes logc + k*Ln2.
// --------------------------------------------------------------------------
__device__ float glibc_logf(float x)
{
    unsigned int ix = __float_as_uint(x);
    if (ix == 0x3F800000u) return 0.0f;           // log(1) == +0

    if ((ix - 0x00800000u) >= (0x7F800000u - 0x00800000u)) {
        if ((ix * 2u) == 0u) return __uint_as_float(0xFF800000u);   // log(0)
        if (ix == 0x7F800000u) return __uint_as_float(0x7F800000u); // log(inf)
        if ((ix & 0x80000000u) || (ix * 2u) >= 0xFF000000u)
            return __uint_as_float(0x7FC00000u);  // negative or NaN -> NaN
        ix = __float_as_uint(FMUL(x, K_EIGHT388608));   // subnormal: * 0x1p23
        ix -= (23u << 23);
    }

    unsigned int tmp = ix - 0x3F330000u;
    unsigned int i = (tmp >> 19) & 15u;
    int k = ((int) tmp) >> 23;                    // arithmetic shift
    unsigned int iz = ix - (tmp & 0xFF800000u);
    double invc = __longlong_as_double(C_LOGF_TAB[2 * i]);
    double logc = __longlong_as_double(C_LOGF_TAB[2 * i + 1]);
    double z = (double) __uint_as_float(iz);

    double ln2 = __longlong_as_double(C_LOGF_MISC[0]);
    double a0 = __longlong_as_double(C_LOGF_MISC[1]);
    double a1 = __longlong_as_double(C_LOGF_MISC[2]);
    double a2 = __longlong_as_double(C_LOGF_MISC[3]);

    double r = DFMA(z, invc, -1.0);
    double y0 = DFMA((double) k, ln2, logc);
    double r2 = DMUL(r, r);
    double y = DFMA(a1, r, a2);
    y = DFMA(r2, a0, y);
    y = DFMA(r2, y, DADD(y0, r));
    return __double2float_rn(y);
}

// --------------------------------------------------------------------------
// glibc 2.39 powf  --  sysdeps/ieee754/flt-32/e_powf.c, FMA variant.
// Only the domain the radiation leaves reach is transcribed: a strictly
// positive normal base and a finite exponent (GROUNDALB's
// MAX(0.01, COSZ) ** 1.7).  Anything else returns NaN rather than a value
// this kernel cannot vouch for.
// --------------------------------------------------------------------------
__device__ double powf_log2_inline(unsigned int ix)
{
    unsigned int tmp = ix - 0x3F330000u;
    unsigned int i = (tmp >> 19) & 15u;
    unsigned int top = tmp & 0xFF800000u;
    unsigned int iz = ix - top;
    int k = ((int) top) >> 23;
    double invc = __longlong_as_double(C_POWF_LOG2_TAB[2 * i]);
    double logc = __longlong_as_double(C_POWF_LOG2_TAB[2 * i + 1]);
    double z = (double) __uint_as_float(iz);

    double r = DFMA(z, invc, -1.0);
    double y0 = DADD(logc, (double) k);

    double a0 = __longlong_as_double(C_POWF_LOG2_POLY[0]);
    double a1 = __longlong_as_double(C_POWF_LOG2_POLY[1]);
    double a2 = __longlong_as_double(C_POWF_LOG2_POLY[2]);
    double a3 = __longlong_as_double(C_POWF_LOG2_POLY[3]);
    double a4 = __longlong_as_double(C_POWF_LOG2_POLY[4]);

    double r2 = DMUL(r, r);
    double y = DFMA(a0, r, a1);
    double p = DFMA(a2, r, a3);
    double r4 = DMUL(r2, r2);
    double q = DFMA(a4, r, y0);
    q = DFMA(p, r2, q);
    y = DFMA(y, r4, q);
    return y;
}

__device__ float powf_exp2_inline(double xd, unsigned long long sign_bias)
{
    double shift = __longlong_as_double(C_EXP2F_MISC[0]);
    double kd = DADD(xd, shift);
    unsigned long long ki = (unsigned long long) __double_as_longlong(kd);
    kd = DSUB(kd, shift);
    double r = DSUB(xd, kd);

    unsigned long long t = C_EXP2F_TAB[ki & 31ULL];
    unsigned long long ski = ki + sign_bias;
    t += (ski << (52 - 5));
    double s = __longlong_as_double((long long) t);
    double c0 = __longlong_as_double(C_EXP2F_POLY[0]);
    double c1 = __longlong_as_double(C_EXP2F_POLY[1]);
    double c2 = __longlong_as_double(C_EXP2F_POLY[2]);
    double z = DFMA(c0, r, c1);
    double r2 = DMUL(r, r);
    double y = DFMA(c2, r, 1.0);
    y = DFMA(z, r2, y);
    y = DMUL(y, s);
    return nmp_d2f_rn(y);
}

__device__ float glibc_powf(float x, float y)
{
    unsigned int ix = __float_as_uint(x);
    unsigned int iy = __float_as_uint(y);
    if ((ix - 0x00800000u) >= 0x7F000000u || (((2u * iy) - 1u) >= 0xFEFFFFFFu))
        return __uint_as_float(0x7FC00000u);   // outside the ported domain

    double logx = powf_log2_inline(ix);
    double ylogx = DMUL((double) y, logx);
    unsigned long long ab = (unsigned long long) __double_as_longlong(ylogx);
    unsigned long long lim = ((unsigned long long) __double_as_longlong(126.0)) >> 47;
    if (((ab >> 47) & 0xFFFFULL) >= lim) {
        if (ylogx > __longlong_as_double(C_LIMITS[2]))
            return __uint_as_float(0x7F800000u);
        if (ylogx <= -150.0) return 0.0f;
    }
    return powf_exp2_inline(ylogx, 0ULL);
}

// --------------------------------------------------------------------------
// SNOW_AGE -- module_sf_noahmplsm.F:3119-3167
// in : tau0 grain_growth extra_growth dirt_soot swemx dt tg sneqvo sneqv tauss
// out: tauss fage
// --------------------------------------------------------------------------
__device__ void d_snow_age(const float *p, float *o)
{
    float tau0 = p[0], gg = p[1], eg = p[2], ds = p[3], swemx = p[4];
    float dt = p[5], tg = p[6], sneqvo = p[7], sneqv = p[8], tauss = p[9];

    if (sneqv <= K_ZERO) {
        tauss = K_ZERO;
    } else {
        float dela0 = FDIV(dt, tau0);
        float arg = FMUL(gg, FSUB(FDIV(K_ONE, K_TFRZ), FDIV(K_ONE, tg)));
        float age1 = glibc_expf(arg);
        float age2 = glibc_expf(f_min(K_ZERO, FMUL(eg, arg)));
        float tage = FADD(FADD(age1, age2), ds);
        float dela = FMUL(dela0, tage);
        float dels = FDIV(f_max(K_ZERO, FSUB(sneqv, sneqvo)), swemx);
        float sge = FMUL(FADD(tauss, dela), FSUB(K_ONE, dels));
        tauss = f_max(K_ZERO, sge);
    }
    o[0] = tauss;
    o[1] = FDIV(tauss, FADD(tauss, K_ONE));
}

// --------------------------------------------------------------------------
// SNOWALB_CLASS -- module_sf_noahmplsm.F:3226-3275
// in : swemx qsnow dt albold          out: alb albsnd(2) albsni(2)
// --------------------------------------------------------------------------
__device__ void d_snowalb_class(const float *p, float *o)
{
    float swemx = p[0], qsnow = p[1], dt = p[2], albold = p[3];
    float alb = FADD(K_P55,
                     FMUL(FSUB(albold, K_P55),
                          glibc_expf(FDIV(FMUL(K_NP01, dt), K_T3600))));
    if (qsnow > K_ZERO) {
        float cap = FDIV(swemx, dt);
        alb = FADD(alb, FDIV(FMUL(f_min(qsnow, cap), FSUB(K_P84, alb)), cap));
    }
    o[0] = alb; o[1] = alb; o[2] = alb; o[3] = alb; o[4] = alb;
}

// --------------------------------------------------------------------------
// GROUNDALB -- module_sf_noahmplsm.F:3279-3332
// in : albsat(2) albdry(2) alblak(2) fsno smc1 albsnd(2) albsni(2) cosz tg
// int: ist          out: albgrd(2) albgri(2)
// ICE is declared INTENT(IN) but is never referenced in the pinned body.
// --------------------------------------------------------------------------
__device__ void d_groundalb(const float *p, int ist, float *o)
{
    float albsat[2] = { p[0], p[1] };
    float albdry[2] = { p[2], p[3] };
    float alblak[2] = { p[4], p[5] };
    float fsno = p[6], smc1 = p[7];
    float albsnd[2] = { p[8], p[9] };
    float albsni[2] = { p[10], p[11] };
    float cosz = p[12], tg = p[13];

    for (int ib = 0; ib < 2; ++ib) {
        float inc = f_max(FSUB(K_P11, FMUL(K_P40, smc1)), K_ZERO);
        float albsod, albsoi;
        if (ist == 1) {
            albsod = f_min(FADD(albsat[ib], inc), albdry[ib]);
            albsoi = albsod;
        } else if (tg > K_TFRZ) {
            albsod = FDIV(K_P06,
                          FADD(glibc_powf(f_max(K_P01, cosz), K_P17), K_P15));
            albsoi = K_P06;
        } else {
            albsod = alblak[ib];
            albsoi = albsod;
        }
        o[ib]     = FADD(FMUL(albsod, FSUB(K_ONE, fsno)), FMUL(albsnd[ib], fsno));
        o[2 + ib] = FADD(FMUL(albsoi, FSUB(K_ONE, fsno)), FMUL(albsni[ib], fsno));
    }
}

// --------------------------------------------------------------------------
// SURRAD -- module_sf_noahmplsm.F:2994-3115
// in (37): mpe fsun fsha elai vai laisun laisha solad(2) solai(2) fabd(2)
//          fabi(2) ftdd(2) ftid(2) ftii(2) albgrd(2) albgri(2) albd(2)
//          albi(2) frevd(2) frevi(2) fregd(2) fregi(2)
// out(8): parsun parsha sav sag fsa fsr fsrv fsrg
// --------------------------------------------------------------------------
__device__ void d_surrad(const float *p, float *o)
{
    float mpe = p[0], fsun = p[1], fsha = p[2], elai = p[3], vai = p[4];
    float laisun = p[5], laisha = p[6];
    const float *solad = p + 7,  *solai = p + 9,  *fabd = p + 11;
    const float *fabi  = p + 13, *ftdd  = p + 15, *ftid = p + 17;
    const float *ftii  = p + 19, *albgrd = p + 21, *albgri = p + 23;
    const float *albd  = p + 25, *albi  = p + 27, *frevd = p + 29;
    const float *frevi = p + 31, *fregd = p + 33, *fregi = p + 35;

    float sag = K_ZERO, sav = K_ZERO, fsa = K_ZERO;
    float cad[2], cai[2];
    for (int ib = 0; ib < 2; ++ib) {
        cad[ib] = FMUL(solad[ib], fabd[ib]);
        cai[ib] = FMUL(solai[ib], fabi[ib]);
        sav = FADD(FADD(sav, cad[ib]), cai[ib]);
        fsa = FADD(FADD(fsa, cad[ib]), cai[ib]);
        float trd = FMUL(solad[ib], ftdd[ib]);
        float tri = FADD(FMUL(solad[ib], ftid[ib]), FMUL(solai[ib], ftii[ib]));
        float ab = FADD(FMUL(trd, FSUB(K_ONE, albgrd[ib])),
                        FMUL(tri, FSUB(K_ONE, albgri[ib])));
        sag = FADD(sag, ab);
        fsa = FADD(fsa, ab);
    }

    float laifra = FDIV(elai, f_max(vai, mpe));
    float parsun, parsha;
    if (fsun > K_ZERO) {
        parsun = FDIV(FMUL(FADD(cad[0], FMUL(fsun, cai[0])), laifra),
                      f_max(laisun, mpe));
        parsha = FDIV(FMUL(FMUL(fsha, cai[0]), laifra), f_max(laisha, mpe));
    } else {
        parsun = K_ZERO;
        parsha = FDIV(FMUL(FADD(cad[0], cai[0]), laifra), f_max(laisha, mpe));
    }

    float rvis = FADD(FMUL(albd[0], solad[0]), FMUL(albi[0], solai[0]));
    float rnir = FADD(FMUL(albd[1], solad[1]), FMUL(albi[1], solai[1]));

    o[0] = parsun; o[1] = parsha; o[2] = sav; o[3] = sag; o[4] = fsa;
    o[5] = FADD(rvis, rnir);
    o[6] = FADD(FADD(FADD(FMUL(frevd[0], solad[0]), FMUL(frevi[0], solai[0])),
                     FMUL(frevd[1], solad[1])), FMUL(frevi[1], solai[1]));
    o[7] = FADD(FADD(FADD(FMUL(fregd[0], solad[0]), FMUL(fregi[0], solai[0])),
                     FMUL(fregd[1], solad[1])), FMUL(fregi[1], solai[1]));
}

// --------------------------------------------------------------------------
// TWOSTREAM -- module_sf_noahmplsm.F:3336-3574, OPT_RAD = 3 only
// in (33): xl omegas(2) betads betais cosz vai fwet t fveg albgrd(2)
//          albgri(2) rho(2) tau(2) fab_in(2) fre_in(2) ftd_in(2) fti_in(2)
//          gdir_in frev_in(2) freg_in(2) bgap_in wgap_in
// int: ib ic
// out(15): fab(2) fre(2) ftd(2) fti(2) gdir frev(2) freg(2) bgap wgap
// --------------------------------------------------------------------------
__device__ void d_twostream(const float *p, int ib1, int ic, float *o)
{
    int ib = ib1 - 1;
    float xl = p[0];
    float omegas[2] = { p[1], p[2] };
    float betads = p[3], betais = p[4];
    float cosz = p[5], vai = p[6], fwet = p[7], t = p[8], fveg = p[9];
    float albgrd[2] = { p[10], p[11] };
    float albgri[2] = { p[12], p[13] };
    float rho[2] = { p[14], p[15] };
    float tau[2] = { p[16], p[17] };
    float fab[2] = { p[18], p[19] };
    float fre[2] = { p[20], p[21] };
    float ftd[2] = { p[22], p[23] };
    float fti[2] = { p[24], p[25] };
    float gdir  = p[26];
    float frev[2] = { p[27], p[28] };
    float freg[2] = { p[29], p[30] };
    float bgap = p[31], wgap = p[32];

    float gap, kopen;
    if (vai == K_ZERO) { gap = K_ONE; kopen = K_ONE; }
    else               { gap = FSUB(K_ONE, fveg); kopen = FSUB(K_ONE, fveg); }

    float coszi = f_max(K_P001, cosz);
    float chil = f_min(f_max(xl, K_NP4), K_P6);
    if (fabsf(chil) <= K_P01) chil = K_P01;
    float phi1 = FSUB(FSUB(K_HALF, FMUL(K_P633, chil)),
                      FMUL(FMUL(K_P330, chil), chil));
    float phi2 = FMUL(K_P877, FSUB(K_ONE, FMUL(K_TWO, phi1)));
    gdir = FADD(phi1, FMUL(phi2, coszi));
    float ext = FDIV(gdir, coszi);
    float avmu = FDIV(FSUB(K_ONE,
                           FMUL(FDIV(phi1, phi2),
                                glibc_logf(FDIV(FADD(phi1, phi2), phi1)))),
                      phi2);
    float omegal = FADD(rho[ib], tau[ib]);
    float tmp0 = FADD(gdir, FMUL(phi2, coszi));
    float tmp1 = FMUL(phi1, coszi);
    float asu = FMUL(FDIV(FMUL(FMUL(K_HALF, omegal), gdir), tmp0),
                     FSUB(K_ONE,
                          FMUL(FDIV(tmp1, tmp0),
                               glibc_logf(FDIV(FADD(tmp1, tmp0), tmp1)))));
    float betadl = FMUL(FDIV(FADD(K_ONE, FMUL(avmu, ext)),
                             FMUL(FMUL(omegal, avmu), ext)), asu);
    float qh = FDIV(FADD(K_ONE, chil), K_TWO);
    float q = FMUL(qh, qh);
    float betail = FDIV(FMUL(K_HALF,
                             FADD(FADD(rho[ib], tau[ib]),
                                  FMUL(FSUB(rho[ib], tau[ib]), q))), omegal);

    float tmp2;
    if (t > K_TFRZ) {
        tmp0 = omegal; tmp1 = betadl; tmp2 = betail;
    } else {
        tmp0 = FADD(FMUL(FSUB(K_ONE, fwet), omegal), FMUL(fwet, omegas[ib]));
        tmp1 = FDIV(FADD(FMUL(FMUL(FSUB(K_ONE, fwet), omegal), betadl),
                         FMUL(FMUL(fwet, omegas[ib]), betads)), tmp0);
        tmp2 = FDIV(FADD(FMUL(FMUL(FSUB(K_ONE, fwet), omegal), betail),
                         FMUL(FMUL(fwet, omegas[ib]), betais)), tmp0);
    }
    float omega = tmp0, betad = tmp1, betai = tmp2;

    float b = FADD(FSUB(K_ONE, omega), FMUL(omega, betai));
    float c = FMUL(omega, betai);
    tmp0 = FMUL(avmu, ext);
    float d = FMUL(FMUL(tmp0, omega), betad);
    float f = FMUL(FMUL(tmp0, omega), FSUB(K_ONE, betad));
    tmp1 = FSUB(FMUL(b, b), FMUL(c, c));
    float h = FDIV(FSQRT(tmp1), avmu);
    float sigma = FSUB(FMUL(tmp0, tmp0), tmp1);
    if (fabsf(sigma) < K_MPE) sigma = f_sign(K_MPE, sigma);
    float p1 = FADD(b, FMUL(avmu, h));
    float p2 = FSUB(b, FMUL(avmu, h));
    float p3 = FADD(b, tmp0);
    float p4 = FSUB(b, tmp0);
    float s1 = glibc_expf(FMUL(-h, vai));
    float s2 = glibc_expf(FMUL(-ext, vai));
    float u1, u2, u3;
    if (ic == 0) {
        u1 = FSUB(b, FDIV(c, albgrd[ib]));
        u2 = FSUB(b, FMUL(c, albgrd[ib]));
        u3 = FADD(f, FMUL(c, albgrd[ib]));
    } else {
        u1 = FSUB(b, FDIV(c, albgri[ib]));
        u2 = FSUB(b, FMUL(c, albgri[ib]));
        u3 = FADD(f, FMUL(c, albgri[ib]));
    }
    tmp2 = FSUB(u1, FMUL(avmu, h));
    float tmp3 = FADD(u1, FMUL(avmu, h));
    float d1 = FSUB(FDIV(FMUL(p1, tmp2), s1), FMUL(FMUL(p2, tmp3), s1));
    float tmp4 = FADD(u2, FMUL(avmu, h));
    float tmp5 = FSUB(u2, FMUL(avmu, h));
    float d2 = FSUB(FDIV(tmp4, s1), FMUL(tmp5, s1));
    float h1 = FSUB(FMUL(-d, p4), FMUL(c, f));
    float tmp6 = FSUB(d, FDIV(FMUL(h1, p3), sigma));
    float tmp7 = FMUL(FSUB(FSUB(d, c), FMUL(FDIV(h1, sigma), FADD(u1, tmp0))), s2);
    float h2 = FDIV(FSUB(FDIV(FMUL(tmp6, tmp2), s1), FMUL(p2, tmp7)), d1);
    float h3 = -FDIV(FSUB(FMUL(FMUL(tmp6, tmp3), s1), FMUL(p1, tmp7)), d1);
    float h4 = FSUB(FMUL(-f, p3), FMUL(c, d));
    float tmp8 = FDIV(h4, sigma);
    float tmp9 = FMUL(FSUB(u3, FMUL(tmp8, FSUB(u2, tmp0))), s2);
    float h5 = -FDIV(FADD(FDIV(FMUL(tmp8, tmp4), s1), tmp9), d2);
    float h6 = FDIV(FADD(FMUL(FMUL(tmp8, tmp5), s1), tmp9), d2);
    float h7 = FDIV(FMUL(c, tmp2), FMUL(d1, s1));
    float h8 = FDIV(FMUL(FMUL(-c, tmp3), s1), d1);
    float h9 = FDIV(tmp4, FMUL(d2, s1));
    float h10 = FDIV(FMUL(-tmp5, s1), d2);

    float ftds, ftis;
    if (ic == 0) {
        ftds = FADD(FMUL(s2, FSUB(K_ONE, gap)), gap);
        ftis = FMUL(FADD(FADD(FDIV(FMUL(h4, s2), sigma), FMUL(h5, s1)),
                         FDIV(h6, s1)), FSUB(K_ONE, gap));
    } else {
        ftds = K_ZERO;
        ftis = FADD(FMUL(FADD(FMUL(h9, s1), FDIV(h10, s1)),
                         FSUB(K_ONE, kopen)), kopen);
    }
    ftd[ib] = ftds;
    fti[ib] = ftis;

    float fres, freveg, frebar;
    if (ic == 0) {
        fres = FADD(FMUL(FADD(FADD(FDIV(h1, sigma), h2), h3),
                         FSUB(K_ONE, gap)), FMUL(albgrd[ib], gap));
        freveg = FMUL(FADD(FADD(FDIV(h1, sigma), h2), h3), FSUB(K_ONE, gap));
        frebar = FMUL(albgrd[ib], gap);
    } else {
        fres = FADD(FMUL(FADD(h7, h8), FSUB(K_ONE, kopen)),
                    FMUL(albgri[ib], kopen));
        freveg = FADD(FMUL(FADD(h7, h8), FSUB(K_ONE, kopen)),
                      FMUL(albgri[ib], kopen));
        frebar = K_ZERO;
    }
    fre[ib] = fres;
    frev[ib] = freveg;
    freg[ib] = frebar;

    fab[ib] = FSUB(FSUB(FSUB(K_ONE, fre[ib]),
                        FMUL(FSUB(K_ONE, albgrd[ib]), ftd[ib])),
                   FMUL(FSUB(K_ONE, albgri[ib]), fti[ib]));

    o[0] = fab[0];  o[1] = fab[1];  o[2] = fre[0];  o[3] = fre[1];
    o[4] = ftd[0];  o[5] = ftd[1];  o[6] = fti[0];  o[7] = fti[1];
    o[8] = gdir;
    o[9] = frev[0]; o[10] = frev[1]; o[11] = freg[0]; o[12] = freg[1];
    o[13] = bgap;   o[14] = wgap;
}

// --------------------------------------------------------------------------
// ALBEDO -- module_sf_noahmplsm.F:2810-2990, OPT_ALB = 2 / OPT_RAD = 3
// in (52): tau0 grain_growth extra_growth dirt_soot swemx albsat(2) albdry(2)
//          alblak(2) rhol(2) rhos(2) taul(2) taus(2) xl omegas(2) betads
//          betais dt cosz elai esai tg tv snowh fsno fwet sneqvo sneqv qsnow
//          fveg smc(4) albold_in tauss_in fage_in frevd_in(2) frevi_in(2)
//          fregd_in(2) fregi_in(2)
// int: ist
// out(36): fage albold tauss fsun bgap wgap albgrd(2) albgri(2) albd(2)
//          albi(2) fabd(2) fabi(2) ftdd(2) ftid(2) ftii(2) frevd(2) frevi(2)
//          fregd(2) fregi(2) albsnd(2) albsni(2)
// --------------------------------------------------------------------------
__device__ void d_albedo(const float *p, int ist, float *o)
{
    float swemx = p[4];
    float dt = p[24], cosz = p[25], elai = p[26], esai = p[27];
    float tg = p[28], tv = p[29];
    float fsno = p[31], fwet = p[32], sneqvo = p[33], sneqv = p[34];
    float qsnow = p[35], fveg = p[36];
    float albold = p[41], tauss = p[42], fage = p[43];

    float mpe = K_MPE;
    float bgap = K_ZERO, wgap = K_ZERO;
    float albd[2] = { K_ZERO, K_ZERO }, albi[2] = { K_ZERO, K_ZERO };
    float albgrd[2] = { K_ZERO, K_ZERO }, albgri[2] = { K_ZERO, K_ZERO };
    float albsnd[2] = { K_ZERO, K_ZERO }, albsni[2] = { K_ZERO, K_ZERO };
    float fabd[2] = { K_ZERO, K_ZERO }, fabi[2] = { K_ZERO, K_ZERO };
    float ftdd[2] = { K_ZERO, K_ZERO }, ftid[2] = { K_ZERO, K_ZERO };
    float ftii[2] = { K_ZERO, K_ZERO };
    float fsun = K_ZERO;
    // Not zeroed by ALBEDO's init loop (lines 2829-2842): undefined on the
    // COSZ <= 0 exit, which in the reference build means the caller's bytes.
    float frevd[2] = { p[44], p[45] }, frevi[2] = { p[46], p[47] };
    float fregd[2] = { p[48], p[49] }, fregi[2] = { p[50], p[51] };
    float ftdi[2] = { K_ZERO, K_ZERO };
    float gdir = K_ZERO;

    if (cosz > K_ZERO) {
        float rho[2], tau[2], vai = K_ZERO;
        for (int ib = 0; ib < 2; ++ib) {
            vai = FADD(elai, esai);
            float wl = FDIV(elai, f_max(vai, mpe));
            float ws = FDIV(esai, f_max(vai, mpe));
            rho[ib] = f_max(FADD(FMUL(p[11 + ib], wl), FMUL(p[13 + ib], ws)), mpe);
            tau[ib] = f_max(FADD(FMUL(p[15 + ib], wl), FMUL(p[17 + ib], ws)), mpe);
        }

        float sa_in[10] = { p[0], p[1], p[2], p[3], swemx,
                            dt, tg, sneqvo, sneqv, tauss };
        float sa_out[2];
        d_snow_age(sa_in, sa_out);
        tauss = sa_out[0];
        fage = sa_out[1];

        float sc_in[4] = { swemx, qsnow, dt, albold };
        float sc_out[5];
        d_snowalb_class(sc_in, sc_out);
        albsnd[0] = sc_out[1]; albsnd[1] = sc_out[2];
        albsni[0] = sc_out[3]; albsni[1] = sc_out[4];
        albold = sc_out[0];

        float ga_in[14] = { p[5], p[6], p[7], p[8], p[9], p[10],
                            fsno, p[37], albsnd[0], albsnd[1],
                            albsni[0], albsni[1], cosz, tg };
        float ga_out[4];
        d_groundalb(ga_in, ist, ga_out);
        albgrd[0] = ga_out[0]; albgrd[1] = ga_out[1];
        albgri[0] = ga_out[2]; albgri[1] = ga_out[3];

        for (int ib = 1; ib <= 2; ++ib) {
            float ts_in[33] = {
                p[19], p[20], p[21], p[22], p[23],
                cosz, vai, fwet, tv, fveg,
                albgrd[0], albgrd[1], albgri[0], albgri[1],
                rho[0], rho[1], tau[0], tau[1],
                fabd[0], fabd[1], albd[0], albd[1],
                ftdd[0], ftdd[1], ftid[0], ftid[1],
                gdir, frevd[0], frevd[1], fregd[0], fregd[1], bgap, wgap };
            float ts_out[15];
            d_twostream(ts_in, ib, 0, ts_out);
            fabd[0] = ts_out[0]; fabd[1] = ts_out[1];
            albd[0] = ts_out[2]; albd[1] = ts_out[3];
            ftdd[0] = ts_out[4]; ftdd[1] = ts_out[5];
            ftid[0] = ts_out[6]; ftid[1] = ts_out[7];
            gdir = ts_out[8];
            frevd[0] = ts_out[9];  frevd[1] = ts_out[10];
            fregd[0] = ts_out[11]; fregd[1] = ts_out[12];
            bgap = ts_out[13]; wgap = ts_out[14];

            float ti_in[33] = {
                p[19], p[20], p[21], p[22], p[23],
                cosz, vai, fwet, tv, fveg,
                albgrd[0], albgrd[1], albgri[0], albgri[1],
                rho[0], rho[1], tau[0], tau[1],
                fabi[0], fabi[1], albi[0], albi[1],
                ftdi[0], ftdi[1], ftii[0], ftii[1],
                gdir, frevi[0], frevi[1], fregi[0], fregi[1], bgap, wgap };
            d_twostream(ti_in, ib, 1, ts_out);
            fabi[0] = ts_out[0]; fabi[1] = ts_out[1];
            albi[0] = ts_out[2]; albi[1] = ts_out[3];
            ftdi[0] = ts_out[4]; ftdi[1] = ts_out[5];
            ftii[0] = ts_out[6]; ftii[1] = ts_out[7];
            gdir = ts_out[8];
            frevi[0] = ts_out[9];  frevi[1] = ts_out[10];
            fregi[0] = ts_out[11]; fregi[1] = ts_out[12];
            bgap = ts_out[13]; wgap = ts_out[14];
        }

        float ext = FMUL(FDIV(gdir, cosz),
                         FSQRT(FSUB(FSUB(K_ONE, rho[0]), tau[0])));
        fsun = FDIV(FSUB(K_ONE, glibc_expf(FMUL(-ext, vai))),
                    f_max(FMUL(ext, vai), mpe));
        ext = fsun;
        fsun = (ext < K_P01) ? K_ZERO : ext;
    }

    o[0] = fage; o[1] = albold; o[2] = tauss; o[3] = fsun;
    o[4] = bgap; o[5] = wgap;
    const float *v[15] = { albgrd, albgri, albd, albi, fabd, fabi,
                           ftdd, ftid, ftii, frevd, frevi, fregd, fregi,
                           albsnd, albsni };
    for (int k = 0; k < 15; ++k) { o[6 + 2 * k] = v[k][0]; o[7 + 2 * k] = v[k][1]; }
}

// --------------------------------------------------------------------------
// one kernel per leaf; row-major [n, nin] in, [n, nout] out
// --------------------------------------------------------------------------
extern "C" __global__ void noahmp_rad_snow_age(const float *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_snow_age(in + (size_t) i * 10, out + (size_t) i * 2);
}

extern "C" __global__ void noahmp_rad_snowalb_class(const float *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_snowalb_class(in + (size_t) i * 4, out + (size_t) i * 5);
}

extern "C" __global__ void noahmp_rad_groundalb(const float *in, const int *ist,
                                                float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_groundalb(in + (size_t) i * 14, ist[i], out + (size_t) i * 4);
}

extern "C" __global__ void noahmp_rad_surrad(const float *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_surrad(in + (size_t) i * 37, out + (size_t) i * 8);
}

extern "C" __global__ void noahmp_rad_twostream(const float *in, const int *ibic,
                                                float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d_twostream(in + (size_t) i * 33, ibic[2 * i], ibic[2 * i + 1],
                    out + (size_t) i * 15);
}

extern "C" __global__ void noahmp_rad_albedo(const float *in, const int *ist,
                                             float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_albedo(in + (size_t) i * 52, ist[i], out + (size_t) i * 36);
}

// --------------------------------------------------------------------------
// RADIATION -- module_sf_noahmplsm.F:2691-2806, whole.
//
// ALBEDO, five assignments, SURRAD.  Nothing else is in that routine, and
// nothing here is new arithmetic: d_albedo and d_surrad are the same bodies
// tests/test_noahmp_radiation_cuda.py holds at max_ulp 0 against
// noahmp-radiation-albedo.csv and noahmp-radiation-surrad.csv, and the five
// statements between them are :2787-2790 transcribed with the same
// rounding-pinned primitives as the rest of the file.
//
// This exists so the *runtime* can answer one paused RADIATION call per land
// column with one launch.  Splitting it into two launches with the five
// assignments on the host would put per-column CPython back on the path,
// which is the cost the whole device column exists to remove.
//
// MPE: RADIATION declares its own PARAMETER MPE = 1.E-6 at :2767, a different
// declaration from ALBEDO's at :2823 and numerically the same value.  K_MPE is
// that bit pattern (0x358637BD) and serves both, exactly as the CPU
// transcription's _RADIATION_MPE and _MPE_ALBEDO do.
//
// in (56): the 52 ALBEDO slots (see noahmp_rad_albedo), then SOLAD(2),
//          SOLAI(2).
// int:     IST.
// out (19): FSUN LAISUN LAISHA PARSUN PARSHA SAV SAG FSA FSR FSRV FSRG
//           ALBSND(2) ALBSNI(2) BGAP WGAP ALBOLD TAUSS
//
// The COSZ <= 0 reset of FSRV/FSRG/BGAP/WGAP is NOT here: it is ENERGY's, at
// the end of the column (:2350 region in the port), not RADIATION's, and
// putting it here would move a statement across a routine boundary.
// --------------------------------------------------------------------------
#define R_NIN 56
#define R_NOUT 19

__device__ void d_radiation(const float *p, int ist, float *o)
{
    float alb[36];
    d_albedo(p, ist, alb);

    const float elai = p[26], esai = p[27];
    const float *solad = p + 52, *solai = p + 54;

    // :2786-2790
    const float fsun = alb[3];
    const float fsha = FSUB(K_ONE, fsun);
    const float laisun = FMUL(elai, fsun);
    const float laisha = FMUL(elai, fsha);
    const float vai = FADD(elai, esai);

    // :2793-2799.  ALBEDO's outputs are laid out
    // fage albold tauss fsun bgap wgap then 15 two-element vectors in the
    // order albgrd albgri albd albi fabd fabi ftdd ftid ftii frevd frevi
    // fregd fregi albsnd albsni, so vector k occupies alb[6+2k], alb[7+2k].
    float s[37];
    s[0] = K_MPE; s[1] = fsun; s[2] = fsha; s[3] = elai; s[4] = vai;
    s[5] = laisun; s[6] = laisha;
    s[7] = solad[0]; s[8] = solad[1];
    s[9] = solai[0]; s[10] = solai[1];
    // SURRAD's order is fabd fabi ftdd ftid ftii albgrd albgri albd albi
    // frevd frevi fregd fregi -- ALBEDO's vectors 4,5,6,7,8,0,1,2,3,9,10,11,12.
    const int order[13] = { 4, 5, 6, 7, 8, 0, 1, 2, 3, 9, 10, 11, 12 };
    for (int k = 0; k < 13; ++k) {
        s[11 + 2 * k] = alb[6 + 2 * order[k]];
        s[12 + 2 * k] = alb[7 + 2 * order[k]];
    }

    float sr[8];
    d_surrad(s, sr);

    o[0] = fsun; o[1] = laisun; o[2] = laisha;
    o[3] = sr[0];  // PARSUN
    o[4] = sr[1];  // PARSHA
    o[5] = sr[2];  // SAV
    o[6] = sr[3];  // SAG
    o[7] = sr[4];  // FSA
    o[8] = sr[5];  // FSR
    o[9] = sr[6];  // FSRV
    o[10] = sr[7]; // FSRG
    o[11] = alb[32]; o[12] = alb[33];   // ALBSND
    o[13] = alb[34]; o[14] = alb[35];   // ALBSNI
    o[15] = alb[4];  o[16] = alb[5];    // BGAP, WGAP
    o[17] = alb[1];  o[18] = alb[2];    // ALBOLD, TAUSS
}

extern "C" __global__ void noahmp_radiation_batch(const float *in,
                                                  const int *ist,
                                                  float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d_radiation(in + (size_t) i * R_NIN, ist[i], out + (size_t) i * R_NOUT);
}

// --------------------------------------------------------------------------
// Device-libm probe.
//
// This file's glibc_powf is one of the five narrow copies in the tree: outside
// [FLT_MIN, inf) it returns NaN instead of taking glibc's sign/zero/inf path.
// Until now nothing measured what that costs, because the guard had never been
// observed to fire.  Both halves of the question are now measurable here:
//
//   * drive glibc_powf with the subnormal / signed-zero / negative
//     neighbourhood and the guard is *seen* returning NaN, so the guard is
//     real rather than asserted; and
//   * sweep COSZ over every float and the one live call site,
//     d_groundalb:384, never presents such a base -- WRF itself clamps it with
//     MAX(0.01, COSZ) -- so RADIATION's answer does not depend on the guard.
//
// out stride 3: expf(x), logf(x), powf(x, y)
// --------------------------------------------------------------------------
extern "C" __global__ void noahmp_rad_libm_probe(const float *x,
                                                 const float *y,
                                                 float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[3 * i + 0] = glibc_expf(x[i]);
    out[3 * i + 1] = glibc_logf(x[i]);
    out[3 * i + 2] = glibc_powf(x[i], y[i]);
}
