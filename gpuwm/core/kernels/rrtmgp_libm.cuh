// WOOF's own float32 expf and logf for the RTE+RRTMGP units.
//
// rrtmgp_gas (the interpolation prepass: LOG of the layer and reference
// pressures, EXP(LOG(press_ref_trop))) and rrtmgp_rte (EXP of the LW optical
// path and of the SW two-stream exponents) are graded bit for bit against
// the upstream RTE+RRTMGP Fortran built single precision by gfortran on the
// oracle host (tools/rrtmgp_rte_upstream_oracle).  There a real-argument EXP
// or LOG is the host C library's expf/logf; CUDA's expf/logf are different
// functions (CUDA's expf alone is off by up to 2 ULP), so these units call the
// words below instead.
//
// The words are the ones gpuwm/core/kernels/mynn_pbl.cu carries
// (mynn_glibc_expf / mynn_glibc_logf, the forms whose double multiply-adds
// are single fused operations): tools/mynn_pbl_wrf461_oracle/libm_sweep.py
// grades that code against the oracle host over every float32 input, and
// tests/test_cuda_libm_table_copies.py holds the tables below to the same
// bit patterns as every other copy in the tree (prefix RRTMGP_).  An NVRTC
// module cannot include another .cu file, which is why the unit carries its
// own copy.  Every double operation is an explicit rounded intrinsic, so the
// words are the same under --fmad=false, the default build and the strict
// build.

// Logarithm table: 16 (invc, logc) pairs, then ln2 and three coefficients.
__constant__ unsigned long long RRTMGP_LOGF_TAB[32] = {
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
__constant__ unsigned long long RRTMGP_LOGF_MISC[4] = {
    0x3FE62E42FEFA39EFULL, 0xBFD00EA348B88334ULL,
    0x3FD5575B0BE00B6AULL, 0xBFDFFFFEF20A4123ULL,
};

// 32-entry exp2 table, EXP2F_TABLE_BITS = 5.
__constant__ unsigned long long RRTMGP_EXP2F_TAB[32] = {
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
// Scaled 1/ln2, the 0x1.8p52 shift, and the three scaled polynomial terms.
__constant__ unsigned long long RRTMGP_EXPF_MISC[5] = {
    0x40471547652B82FEULL, 0x4338000000000000ULL,
    0x3EBC6AF84B912394ULL, 0x3F2EBFCE50FAC4F3ULL, 0x3F962E42FF0C52D6ULL,
};

// The binary32 value of a binary64 result, subnormals included (a plain
// __double2float_rn flushes them when the unit is compiled with FTZ).
__device__ __forceinline__ float rrtmgp_d2f_rn(double y)
{
    const double a = fabs(y);
    if (a > 0.0 && a < 1.1754943508222875e-38) {
        const double scaled = rint(__dmul_rn(a, 7.1362384635297994e+44));
        const unsigned int s =
            (__double_as_longlong(y) < 0LL) ? 0x80000000u : 0u;
        return __uint_as_float(s | (unsigned int) scaled);
    }
    return __double2float_rn(y);
}

__device__ float rrtmgp_logf(float x)
{
    unsigned int ix = __float_as_uint(x);
    if (ix == 0x3F800000u) return 0.0f;
    if ((ix - 0x00800000u) >= (0x7F800000u - 0x00800000u)) {
        if ((ix * 2u) == 0u) return __uint_as_float(0xFF800000u);
        if (ix == 0x7F800000u) return x;
        if ((ix & 0x80000000u) || (ix * 2u) >= 0xFF000000u)
            return __uint_as_float(0x7FC00000u);
        ix = __float_as_uint(__fmul_rn(x, __uint_as_float(0x4B000000u)));
        ix -= (23u << 23);
    }
    const unsigned int tmp = ix - 0x3F330000u;
    const unsigned int i = (tmp >> 19) & 15u;
    const int k = ((int) tmp) >> 23;
    const unsigned int iz = ix - (tmp & 0xFF800000u);
    const double invc = __longlong_as_double(RRTMGP_LOGF_TAB[2 * i]);
    const double logc = __longlong_as_double(RRTMGP_LOGF_TAB[2 * i + 1]);
    const double ln2 = __longlong_as_double(RRTMGP_LOGF_MISC[0]);
    const double a0 = __longlong_as_double(RRTMGP_LOGF_MISC[1]);
    const double a1 = __longlong_as_double(RRTMGP_LOGF_MISC[2]);
    const double a2 = __longlong_as_double(RRTMGP_LOGF_MISC[3]);
    const double z = (double) __uint_as_float(iz);
    const double r = __fma_rn(z, invc, -1.0);
    const double y0 = __fma_rn((double) k, ln2, logc);
    const double r2 = __dmul_rn(r, r);
    double y = __fma_rn(a1, r, a2);
    y = __fma_rn(r2, a0, y);
    y = __fma_rn(r2, y, __dadd_rn(y0, r));
    return rrtmgp_d2f_rn(y);
}

__device__ float rrtmgp_expf(float x)
{
    const unsigned int ix = __float_as_uint(x);
    const unsigned int abstop = (ix >> 20) & 0x7FFu;
    if (abstop >= 0x42Bu) {                      // |x| >= 88, NaN or inf
        if (ix == 0xFF800000u) return 0.0f;
        if (abstop >= 0x7F8u) return __fadd_rn(x, x);
        if (x > __uint_as_float(0x42B17218u))
            return __uint_as_float(0x7F800000u);
        if (__uint_as_float(0xC2CFF1B4u) > x) return 0.0f;
    }
    const double inv = __longlong_as_double(RRTMGP_EXPF_MISC[0]);
    const double xd = (double) x;
    const double shift = __longlong_as_double(RRTMGP_EXPF_MISC[1]);
    double kd = __fma_rn(inv, xd, shift);
    const unsigned long long ki =
        (unsigned long long) __double_as_longlong(kd);
    kd = __dsub_rn(kd, shift);
    const double r = __fma_rn(inv, xd, -kd);
    unsigned long long t = RRTMGP_EXP2F_TAB[ki & 31ULL];
    t += (ki << (52 - 5));
    const double s = __longlong_as_double((long long) t);
    const double c0 = __longlong_as_double(RRTMGP_EXPF_MISC[2]);
    const double c1 = __longlong_as_double(RRTMGP_EXPF_MISC[3]);
    const double c2 = __longlong_as_double(RRTMGP_EXPF_MISC[4]);
    const double p = __fma_rn(c0, r, c1);
    const double r2 = __dmul_rn(r, r);
    double y = __fma_rn(c2, r, 1.0);
    y = __dmul_rn(__fma_rn(p, r2, y), s);
    return rrtmgp_d2f_rn(y);
}
