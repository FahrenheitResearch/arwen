// WOOF's own float32 libm words for the MYNN surface layer
// (sf_sfclay_physics=5, gpuwm/core/kernels/mynn_surface.cu).
//
// WHY.  The surface layer is graded bit for bit against WRF 4.6.1's
// SFCLAY1D_mynn compiled by gfortran on an x86-64 FMA host
// (tools/mynn_sfclay_wrf461_column_oracle).  Every EXP, LOG, ATAN and
// REAL**REAL there is a host libm call, and CUDA's expf/logf/powf/atanf are
// different functions: before this header the kernel missed the oracle by up
// to 423 ULP (HFX), all of it seeded by one-ULP libm differences that the
// DTHVDZ cancellation and the z/L solve then amplify.  These are the words
// mynn_pbl.cu already proves bitwise for the MYNN PBL (lane/mynn-exact).
//
// ONE TEXT.  The code below is mynn_pbl.cu's operator helpers and libm block
// with the comments taken out, character for character otherwise; NVRTC
// compiles each unit alone, so the surface unit cannot see mynn_pbl.cu.
// tests/test_mynn_sfclay_wrf461_column_oracle.py fails if the two drift,
// and tests/test_cuda_libm_table_copies.py holds every __constant__ table
// here to the same values as every other copy.  mynn_pbl.cu's comments are
// the documentation of record for each function.
//
// Listed for mynn_surface only in gpuwm/core/kernels/__init__.py
// _EXTRA_HEADERS; no other unit's assembled source changes.

__device__ __forceinline__ real mynn_add(real a, real b)
{
    real r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ real mynn_sub(real a, real b)
{
    real r;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ real mynn_mul(real a, real b)
{
    real r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ real mynn_div(real a, real b)
{
    real r;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

#define MYNN_ADD(x, y) mynn_add((x), (y))
#define MYNN_SUB(x, y) mynn_sub((x), (y))
#define MYNN_MUL(x, y) mynn_mul((x), (y))
#define MYNN_DIV(x, y) mynn_div((x), (y))

#define MYNN_DADD(x, y) __dadd_rn((x), (y))
#define MYNN_DSUB(x, y) __dsub_rn((x), (y))
#define MYNN_DMUL(x, y) __dmul_rn((x), (y))

__device__ __forceinline__ bool mynn_gt(real a, real b)
{
    unsigned int p;
    asm("{ .reg .pred q; setp.gt.f32 q, %1, %2; selp.u32 %0, 1, 0, q; }"
        : "=r"(p) : "f"(a), "f"(b));
    return p != 0u;
}

__device__ __forceinline__ real mynn_max2(real a, real b)
{
    return mynn_gt(b, a) ? b : a;
}

__device__ __forceinline__ real mynn_min2(real a, real b)
{
    return mynn_gt(a, b) ? b : a;
}

__constant__ unsigned long long MYNN_LOGF_TAB[32] = {
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
__constant__ unsigned long long MYNN_LOGF_MISC[4] = {
    0x3FE62E42FEFA39EFULL, 0xBFD00EA348B88334ULL,
    0x3FD5575B0BE00B6AULL, 0xBFDFFFFEF20A4123ULL,
};

__constant__ unsigned long long MYNN_POWF_LOG2_TAB[32] = {
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
__constant__ unsigned long long MYNN_POWF_LOG2_POLY[5] = {
    0x3FD27616C9496E0BULL, 0xBFD71969A075C67AULL, 0x3FDEC70A6CA7BADDULL,
    0xBFE7154748BEF6C8ULL, 0x3FF71547652AB82BULL,
};

__constant__ unsigned long long MYNN_EXP2F_TAB[32] = {
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
    0x3FEFA4AFA2A490DAULL, 0x3FEFD0765B6E4540ULL,
};
__constant__ unsigned long long MYNN_EXP2F_POLY[3] = {
    0x3FAC6AF84B912394ULL, 0x3FCEBFCE50FAC4F3ULL, 0x3FE62E42FF0C52D6ULL,
};

__constant__ unsigned long long MYNN_POWF_MISC[4] = {
    0x42E8000000000000ULL, 0x405FFFFFFFD1D571ULL,
    0x405F800000000000ULL, 0xC062C00000000000ULL,
};

__constant__ unsigned int MYNN_ATANF_TAB[19] = {
    0x3EED6338u, 0x3F490FDAu, 0x3F7B985Eu, 0x3FC90FDAu,
    0x31AC3769u, 0x33222168u, 0x33140FB4u, 0x33A22168u,
    0x3EAAAAABu, 0xBE4CCCCDu, 0x3E124925u, 0xBDE38E38u,
    0x3DBA2E6Eu, 0xBD9D8795u, 0x3D886B35u, 0xBD6EF16Bu,
    0x3D4BDA59u, 0xBD15A221u, 0x3C8569D7u,
};

__constant__ unsigned int MYNN_LIBM_F32[1] = { 0x4B000000u };

#define MYNN_DFMA(a, b, c) __fma_rn((a), (b), (c))

__device__ __forceinline__ real mynn_d2f_rn(double y)
{
    double a = fabs(y);
    if (a > 0.0 && a < 1.1754943508222875e-38) {
        double scaled = rint(MYNN_DMUL(a, 7.1362384635297994e+44));
        unsigned int s = (__double_as_longlong(y) < 0LL) ? 0x80000000u : 0u;
        return __uint_as_float(s | (unsigned int) scaled);
    }
    return __double2float_rn(y);
}

__device__ real mynn_glibc_logf(real x)
{
    unsigned int ix = __float_as_uint(x);
    if (ix == 0x3F800000u) return 0.0f;
    if ((ix - 0x00800000u) >= (0x7F800000u - 0x00800000u)) {
        if ((ix * 2u) == 0u) return __uint_as_float(0xFF800000u);
        if (ix == 0x7F800000u) return x;
        if ((ix & 0x80000000u) || (ix * 2u) >= 0xFF000000u)
            return __uint_as_float(0x7FC00000u);
        ix = __float_as_uint(
            MYNN_MUL(x, __uint_as_float(MYNN_LIBM_F32[0])));
        ix -= (23u << 23);
    }
    unsigned int tmp = ix - 0x3F330000u;
    unsigned int i = (tmp >> 19) & 15u;
    int k = ((int) tmp) >> 23;
    unsigned int iz = ix - (tmp & 0xFF800000u);
    double invc = __longlong_as_double(MYNN_LOGF_TAB[2 * i]);
    double logc = __longlong_as_double(MYNN_LOGF_TAB[2 * i + 1]);
    double ln2 = __longlong_as_double(MYNN_LOGF_MISC[0]);
    double a0 = __longlong_as_double(MYNN_LOGF_MISC[1]);
    double a1 = __longlong_as_double(MYNN_LOGF_MISC[2]);
    double a2 = __longlong_as_double(MYNN_LOGF_MISC[3]);
    double z = (double) __uint_as_float(iz);
    double r = MYNN_DFMA(z, invc, -1.0);
    double y0 = MYNN_DFMA((double) k, ln2, logc);
    double r2 = MYNN_DMUL(r, r);
    double y = MYNN_DFMA(a1, r, a2);
    y = MYNN_DFMA(r2, a0, y);
    y = MYNN_DFMA(r2, y, MYNN_DADD(y0, r));
    return mynn_d2f_rn(y);
}

__device__ double mynn_glibc_powf_log2(unsigned int ix)
{
    unsigned int tmp = ix - 0x3F330000u;
    unsigned int i = (tmp >> 19) & 15u;
    unsigned int top = tmp & 0xFF800000u;
    unsigned int iz = ix - top;
    int k = ((int) top) >> 23;
    double invc = __longlong_as_double(MYNN_POWF_LOG2_TAB[2 * i]);
    double logc = __longlong_as_double(MYNN_POWF_LOG2_TAB[2 * i + 1]);
    double a0 = __longlong_as_double(MYNN_POWF_LOG2_POLY[0]);
    double a1 = __longlong_as_double(MYNN_POWF_LOG2_POLY[1]);
    double a2 = __longlong_as_double(MYNN_POWF_LOG2_POLY[2]);
    double a3 = __longlong_as_double(MYNN_POWF_LOG2_POLY[3]);
    double a4 = __longlong_as_double(MYNN_POWF_LOG2_POLY[4]);
    double z = (double) __uint_as_float(iz);
    double r = MYNN_DFMA(z, invc, -1.0);
    double y0 = MYNN_DADD(logc, (double) k);
    double r2 = MYNN_DMUL(r, r);
    double y = MYNN_DFMA(a0, r, a1);
    double p = MYNN_DFMA(a2, r, a3);
    double r4 = MYNN_DMUL(r2, r2);
    double q = MYNN_DFMA(a4, r, y0);
    q = MYNN_DFMA(p, r2, q);
    return MYNN_DFMA(y, r4, q);
}

__device__ double mynn_glibc_powf_exp2(double xd, unsigned long long bias)
{
    double shift = __longlong_as_double(MYNN_POWF_MISC[0]);
    double kd = MYNN_DADD(xd, shift);
    unsigned long long ki = (unsigned long long) __double_as_longlong(kd);
    kd = MYNN_DSUB(kd, shift);
    double r = MYNN_DSUB(xd, kd);
    unsigned long long t = MYNN_EXP2F_TAB[ki & 31ULL];
    t += ((ki + bias) << (52 - 5));
    double s = __longlong_as_double((long long) t);
    double c0 = __longlong_as_double(MYNN_EXP2F_POLY[0]);
    double c1 = __longlong_as_double(MYNN_EXP2F_POLY[1]);
    double c2 = __longlong_as_double(MYNN_EXP2F_POLY[2]);
    double z = MYNN_DFMA(c0, r, c1);
    double r2 = MYNN_DMUL(r, r);
    double y = MYNN_DFMA(c2, r, 1.0);
    y = MYNN_DFMA(z, r2, y);
    return MYNN_DMUL(y, s);
}

__constant__ unsigned long long MYNN_EXPF_MISC[5] = {
    0x40471547652B82FEULL, 0x4338000000000000ULL,
    0x3EBC6AF84B912394ULL, 0x3F2EBFCE50FAC4F3ULL, 0x3F962E42FF0C52D6ULL,
};

__device__ real mynn_glibc_expf(real x)
{
    unsigned int ix = __float_as_uint(x);
    unsigned int abstop = (ix >> 20) & 0x7FFu;
    if (abstop >= 0x42Bu) {
        if (ix == 0xFF800000u) return 0.0f;
        if (abstop >= 0x7F8u) return MYNN_ADD(x, x);
        if (mynn_gt(x, __uint_as_float(0x42B17218u)))
            return __uint_as_float(0x7F800000u);
        if (mynn_gt(__uint_as_float(0xC2CFF1B4u), x)) return 0.0f;
    }
    double inv = __longlong_as_double(MYNN_EXPF_MISC[0]);
    double xd = (double) x;
    double shift = __longlong_as_double(MYNN_EXPF_MISC[1]);
    double kd = MYNN_DFMA(inv, xd, shift);
    unsigned long long ki = (unsigned long long) __double_as_longlong(kd);
    kd = MYNN_DSUB(kd, shift);
    double r = MYNN_DFMA(inv, xd, -kd);
    unsigned long long t = MYNN_EXP2F_TAB[ki & 31ULL];
    t += (ki << (52 - 5));
    double s = __longlong_as_double((long long) t);
    double c0 = __longlong_as_double(MYNN_EXPF_MISC[2]);
    double c1 = __longlong_as_double(MYNN_EXPF_MISC[3]);
    double c2 = __longlong_as_double(MYNN_EXPF_MISC[4]);
    double p = MYNN_DFMA(c0, r, c1);
    double r2 = MYNN_DMUL(r, r);
    double y = MYNN_DFMA(c2, r, 1.0);
    y = MYNN_DMUL(MYNN_DFMA(p, r2, y), s);
    return mynn_d2f_rn(y);
}

__device__ __forceinline__ int mynn_glibc_powf_checkint(unsigned int iy)
{
    int e = (int) ((iy >> 23) & 0xFFu);
    if (e < 0x7F) return 0;
    if (e > 0x7F + 23) return 2;
    if (iy & ((1u << (0x7F + 23 - e)) - 1u)) return 0;
    if (iy & (1u << (0x7F + 23 - e))) return 1;
    return 2;
}

__device__ __forceinline__ bool mynn_glibc_zeroinfnan(unsigned int ix)
{
    return (2u * ix - 1u) >= (2u * 0x7F800000u - 1u);
}

__device__ real mynn_glibc_powf(real x, real y)
{
    unsigned int ix = __float_as_uint(x);
    unsigned int iy = __float_as_uint(y);
    unsigned long long bias = 0ULL;
    if ((ix - 0x00800000u) >= (0x7F800000u - 0x00800000u)
            || mynn_glibc_zeroinfnan(iy)) {
        if (mynn_glibc_zeroinfnan(iy)) {
            if ((2u * iy) == 0u) return 1.0f;
            if (ix == 0x3F800000u) return 1.0f;
            if ((2u * ix) > (2u * 0x7F800000u)
                    || (2u * iy) > (2u * 0x7F800000u))
                return MYNN_ADD(x, y);
            if ((2u * ix) == (2u * 0x3F800000u)) return 1.0f;
            if (((2u * ix) < (2u * 0x3F800000u))
                    == ((iy & 0x80000000u) == 0u))
                return 0.0f;
            return MYNN_MUL(y, y);
        }
        if (mynn_glibc_zeroinfnan(ix)) {
            real squared = MYNN_MUL(x, x);
            if ((ix & 0x80000000u) && mynn_glibc_powf_checkint(iy) == 1)
                squared = -squared;
            return (iy & 0x80000000u) ? MYNN_DIV(1.0f, squared) : squared;
        }
        if (ix & 0x80000000u) {
            int yint = mynn_glibc_powf_checkint(iy);
            if (yint == 0) return __uint_as_float(0x7FC00000u);
            if (yint == 1) bias = (unsigned long long) (1u << (5 + 11));
            ix &= 0x7FFFFFFFu;
        }
        if (ix < 0x00800000u) {
            ix = __float_as_uint(MYNN_MUL(
                x, __uint_as_float(MYNN_LIBM_F32[0]))) & 0x7FFFFFFFu;
            ix -= (23u << 23);
        }
    }
    double logx = mynn_glibc_powf_log2(ix);
    double ylogx = MYNN_DMUL((double) y, logx);
    unsigned long long ab = (unsigned long long) __double_as_longlong(ylogx);
    unsigned long long lim =
        ((unsigned long long) MYNN_POWF_MISC[2]) >> 47;
    if (((ab >> 47) & 0xFFFFULL) >= lim) {
        if (ylogx > __longlong_as_double(MYNN_POWF_MISC[1]))
            return bias ? __uint_as_float(0xFF800000u)
                        : __uint_as_float(0x7F800000u);
        if (ylogx <= __longlong_as_double(MYNN_POWF_MISC[3]))
            return bias ? -0.0f : 0.0f;
    }
    return mynn_d2f_rn(mynn_glibc_powf_exp2(ylogx, bias));
}

__device__ real mynn_glibc_atanf(real x)
{
    unsigned int hx = __float_as_uint(x);
    unsigned int ix = hx & 0x7FFFFFFFu;
    int signed_hx = (int) hx;
    int id;
#define MYNN_ATAN_HI(i) __uint_as_float(MYNN_ATANF_TAB[(i)])
#define MYNN_ATAN_LO(i) __uint_as_float(MYNN_ATANF_TAB[4 + (i)])
#define MYNN_ATAN_T(i)  __uint_as_float(MYNN_ATANF_TAB[8 + (i)])
    if (ix >= 0x4C000000u) {
        if (ix > 0x7F800000u) return MYNN_ADD(x, x);
        if (signed_hx > 0)
            return MYNN_ADD(MYNN_ATAN_HI(3), MYNN_ATAN_LO(3));
        return MYNN_SUB(-MYNN_ATAN_HI(3), MYNN_ATAN_LO(3));
    }
    if (ix < 0x3EE00000u) {
        if (ix < 0x31000000u) return x;
        id = -1;
    } else {
        x = __uint_as_float(ix);
        if (ix < 0x3F980000u) {
            if (ix < 0x3F300000u) {
                id = 0;
                x = MYNN_DIV(MYNN_SUB(MYNN_MUL(2.0f, x), 1.0f),
                             MYNN_ADD(2.0f, x));
            } else {
                id = 1;
                x = MYNN_DIV(MYNN_SUB(x, 1.0f), MYNN_ADD(x, 1.0f));
            }
        } else if (ix < 0x401C0000u) {
            id = 2;
            x = MYNN_DIV(MYNN_SUB(x, 1.5f),
                         MYNN_ADD(1.0f, MYNN_MUL(1.5f, x)));
        } else {
            id = 3;
            x = MYNN_DIV(-1.0f, x);
        }
    }
    real z = MYNN_MUL(x, x);
    real w = MYNN_MUL(z, z);
    real s1 = MYNN_MUL(w, MYNN_ATAN_T(10));
    s1 = MYNN_MUL(w, MYNN_ADD(MYNN_ATAN_T(8), s1));
    s1 = MYNN_MUL(w, MYNN_ADD(MYNN_ATAN_T(6), s1));
    s1 = MYNN_MUL(w, MYNN_ADD(MYNN_ATAN_T(4), s1));
    s1 = MYNN_MUL(w, MYNN_ADD(MYNN_ATAN_T(2), s1));
    s1 = MYNN_MUL(z, MYNN_ADD(MYNN_ATAN_T(0), s1));
    real s2 = MYNN_MUL(w, MYNN_ATAN_T(9));
    s2 = MYNN_MUL(w, MYNN_ADD(MYNN_ATAN_T(7), s2));
    s2 = MYNN_MUL(w, MYNN_ADD(MYNN_ATAN_T(5), s2));
    s2 = MYNN_MUL(w, MYNN_ADD(MYNN_ATAN_T(3), s2));
    s2 = MYNN_MUL(w, MYNN_ADD(MYNN_ATAN_T(1), s2));
    real s = MYNN_ADD(s1, s2);
    if (id < 0) return MYNN_SUB(x, MYNN_MUL(x, s));
    real r = MYNN_SUB(MYNN_ATAN_HI(id),
                      MYNN_SUB(MYNN_SUB(MYNN_MUL(x, s), MYNN_ATAN_LO(id)), x));
    return (signed_hx < 0) ? -r : r;
#undef MYNN_ATAN_HI
#undef MYNN_ATAN_LO
#undef MYNN_ATAN_T
}
