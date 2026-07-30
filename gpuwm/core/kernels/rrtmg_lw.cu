// WRF v4.6.1 legacy RRTMG longwave -- CUDA FP32 twin of the NumPy
// reference in gpuwm/core/rrtmg_lw.py, gated at max_ulp 0 against the
// same unmodified-Fortran fixtures.
//
// EVERY float arithmetic operation goes through RLW_AD/SU/MU/DV
// (__fadd_rn etc.): NVRTC defaults to --fmad=true and contraction is the
// dominant bitwise hazard; gfortran -O0 on x86-64 emits no FMA.  The
// FP64 intermediates of the libm transcriptions use RLW_DAD/DSU/DMU for
// the same reason.  Compile with --ftz=false; the host preflight in
// tests/test_rrtmg_lw_cuda.py PROVES subnormal survival and the d2f
// behaviour on the live toolchain rather than assuming them.
//
// rlw_log / rlw_exp / rlw_pow are direct transcriptions of glibc 2.39
// sysdeps/ieee754/flt-32, kept statement for statement in step with
// gpuwm/core/kernels/noahmp_leaves.cu (r_log/r_exp/r_pow) and with the
// host copies in gpuwm/core/rrtmg_lw.py.  rlw_d2f_rn recovers the
// subnormal double->float rounding that __double2float_rn flushes on
// this toolchain (measured on sm_120 / CUDA 12-13, not changed by
// --ftz=false).
//
// Band kernels live in sibling files rrtmg_lw_taugb*.cu; the host
// assembles one translation unit (gpuwm/core/rrtmg_lw.py::_gpu_source),
// so device helpers defined here are visible there without re-definition.

#define RLW_AD(a, b) __fadd_rn((a), (b))
#define RLW_SU(a, b) __fsub_rn((a), (b))
#define RLW_MU(a, b) __fmul_rn((a), (b))
#define RLW_DV(a, b) __fdiv_rn((a), (b))
#define RLW_DAD(a, b) __dadd_rn((a), (b))
#define RLW_DSU(a, b) __dsub_rn((a), (b))
#define RLW_DMU(a, b) __dmul_rn((a), (b))

#define NBNDLW 16
#define NGPTLW 140
#define MXMOL 38
#define MAXXSEC 4
#define RLW_TBLINT 10000.0f

// ---------------------------------------------------------------------
// glibc 2.39 FP32 libm transcriptions (see file header).
// ---------------------------------------------------------------------

__device__ const double RLW_LOGF_INVC[16] = {
    0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0,
    0x1.3c995b0b80385p+0, 0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0,
    0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0, 0x1.0953f419900a7p+0,
    0x1p+0,               0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
    0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1,
    0x1.767dcf5534862p-1 };
__device__ const double RLW_LOGF_LOGC[16] = {
    -0x1.57bf7808caadep-2, -0x1.2bef0a7c06ddbp-2, -0x1.01eae7f513a67p-2,
    -0x1.b31d8a68224e9p-3, -0x1.6574f0ac07758p-3, -0x1.1aa2bc79c81p-3,
    -0x1.a4e76ce8c0e5ep-4, -0x1.1973c5a611cccp-4, -0x1.252f438e10c1ep-5,
     0x0p+0,                0x1.aa5aa5df25984p-5,  0x1.c5e53aa362eb4p-4,
     0x1.526e57720db08p-3,  0x1.bc2860d22477p-3,   0x1.1058bc8a07ee1p-2,
     0x1.4043057b6ee09p-2 };
#define RLW_LOGF_LN2 0x1.62e42fefa39efp-1
#define RLW_LOGF_A0 (-0x1.00ea348b88334p-2)
#define RLW_LOGF_A1 (0x1.5575b0be00b6ap-2)
#define RLW_LOGF_A2 (-0x1.ffffef20a4123p-2)

__device__ const unsigned long long RLW_EXP2F_TAB[32] = {
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
#define RLW_EXP2F_P0 0x1.c6af84b912394p-5
#define RLW_EXP2F_P1 0x1.ebfce50fac4f3p-3
#define RLW_EXP2F_P2 0x1.62e42ff0c52d6p-1
#define RLW_EXP2F_SHIFT 0x1.8p+52
#define RLW_EXP2F_SHIFT_SCALED (0x1.8p+52 / 32.0)

__device__ const double RLW_POWF_INVC[16] = {
    0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0,
    0x1.3c995b0b80385p+0, 0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0,
    0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0, 0x1.0953f419900a7p+0,
    0x1p+0,               0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
    0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1,
    0x1.767dcf5534862p-1 };
__device__ const double RLW_POWF_LOGC[16] = {
    -0x1.efec65b963019p-2, -0x1.b0b6832d4fca4p-2, -0x1.7418b0a1fb77bp-2,
    -0x1.39de91a6dcf7bp-2, -0x1.01d9bf3f2b631p-2, -0x1.97c1d1b3b7afp-3,
    -0x1.2f9e393af3c9fp-3, -0x1.960cbbf788d5cp-4, -0x1.a6f9db6475fcep-5,
     0x0p+0,                0x1.338ca9f24f53dp-4,  0x1.476a9543891bap-3,
     0x1.e840b4ac4e4d2p-3,  0x1.40645f0c6651cp-2,  0x1.88e9c2c1b9ff8p-2,
     0x1.ce0a44eb17bccp-2 };
__device__ const double RLW_POWF_A[5] = {
     0x1.27616c9496e0bp-2, -0x1.71969a075c67ap-2,  0x1.ec70a6ca7baddp-2,
    -0x1.7154748bef6c8p-1,  0x1.71547652ab82bp+0 };

__device__ float rlw_log(float x)
{
    unsigned int ix = __float_as_uint(x);
    if (ix == 0x3f800000u) return 0.0f;
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u) {
        if (ix * 2u == 0u) return __int_as_float(0xff800000);
        if (ix == 0x7f800000u) return x;
        if ((ix & 0x80000000u) || ix * 2u >= 0xff000000u)
            return __int_as_float(0x7fc00000);
        ix = __float_as_uint(RLW_MU(x, 8388608.0f));
        ix -= 23u << 23;
    }
    unsigned int tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    int k = (int)tmp >> 23;
    unsigned int iz = ix - (tmp & 0xff800000u);
    double z = (double)__uint_as_float(iz);
    double r = RLW_DSU(RLW_DMU(z, RLW_LOGF_INVC[i]), 1.0);
    double y0 = RLW_DAD(RLW_LOGF_LOGC[i], RLW_DMU((double)k, RLW_LOGF_LN2));
    double r2 = RLW_DMU(r, r);
    double y = RLW_DAD(RLW_DMU(RLW_LOGF_A1, r), RLW_LOGF_A2);
    y = RLW_DAD(RLW_DMU(RLW_LOGF_A0, r2), y);
    y = RLW_DAD(RLW_DMU(y, r2), RLW_DAD(y0, r));
    return __double2float_rn(y);
}

__device__ float rlw_d2f_rn(double y)
{
    double a = fabs(y);
    if (a > 0.0 && a < 1.1754943508222875e-38) {
        double scaled = rint(a * 7.1362384635297994e+44);
        unsigned int m = (unsigned int)scaled;
        unsigned int s = (__double_as_longlong(y) < 0LL) ? 0x80000000u : 0u;
        return __uint_as_float(s | m);
    }
    return __double2float_rn(y);
}

__device__ double rlw_exp2_core(double xd, double shift,
                                double p0, double p1, double p2,
                                unsigned int sign_bias)
{
    double kd = RLW_DAD(xd, shift);
    unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd = RLW_DSU(kd, shift);
    double r = RLW_DSU(xd, kd);
    unsigned long long t = RLW_EXP2F_TAB[ki & 31ULL];
    t += (ki + (unsigned long long)sign_bias) << (52 - 5);
    double s = __longlong_as_double((long long)t);
    double z = RLW_DAD(RLW_DMU(p0, r), p1);
    double r2 = RLW_DMU(r, r);
    double y = RLW_DAD(RLW_DMU(p2, r), 1.0);
    y = RLW_DAD(RLW_DMU(z, r2), y);
    return RLW_DMU(y, s);
}

__device__ float rlw_exp(float x)
{
    unsigned int abstop = (__float_as_uint(x) >> 20) & 0x7ffu;
    if (abstop >= ((__float_as_uint(88.0f)) >> 20)) {
        if (__float_as_uint(x) == 0xff800000u) return 0.0f;
        if (abstop >= (0x7f800000u >> 20)) return RLW_AD(x, x);
        if (x > __int_as_float(0x42b17218)) return __int_as_float(0x7f800000);
        if (x < -__int_as_float(0x42cff1b4)) return 0.0f;
    }
    double xd = (double)x;
    double z = RLW_DMU(0x1.71547652b82fep+0 * 32.0, xd);
    return rlw_d2f_rn(rlw_exp2_core(
        z, RLW_EXP2F_SHIFT,
        RLW_EXP2F_P0 / 32.0 / 32.0 / 32.0,
        RLW_EXP2F_P1 / 32.0 / 32.0,
        RLW_EXP2F_P2 / 32.0, 0u));
}

__device__ double rlw_powf_log2(unsigned int ix)
{
    unsigned int tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    unsigned int top = tmp & 0xff800000u;
    unsigned int iz = ix - top;
    int k = (int)top >> 23;
    double z = (double)__uint_as_float(iz);
    double r = RLW_DSU(RLW_DMU(z, RLW_POWF_INVC[i]), 1.0);
    double y0 = RLW_DAD(RLW_POWF_LOGC[i], (double)k);
    double r2 = RLW_DMU(r, r);
    double y = RLW_DAD(RLW_DMU(RLW_POWF_A[0], r), RLW_POWF_A[1]);
    double p = RLW_DAD(RLW_DMU(RLW_POWF_A[2], r), RLW_POWF_A[3]);
    double r4 = RLW_DMU(r2, r2);
    double q = RLW_DAD(RLW_DMU(RLW_POWF_A[4], r), y0);
    q = RLW_DAD(RLW_DMU(p, r2), q);
    return RLW_DAD(RLW_DMU(y, r4), q);
}

__device__ int rlw_checkint(unsigned int iy)
{
    int e = (int)(iy >> 23 & 0xffu);
    if (e < 0x7f) return 0;
    if (e > 0x7f + 23) return 2;
    if (iy & ((1u << (0x7f + 23 - e)) - 1u)) return 0;
    if (iy & (1u << (0x7f + 23 - e))) return 1;
    return 2;
}

__device__ bool rlw_zeroinfnan(unsigned int ix)
{
    return 2u * ix - 1u >= 2u * 0x7f800000u - 1u;
}

__device__ float rlw_pow(float x, float y)
{
    unsigned int sign_bias = 0u;
    unsigned int ix = __float_as_uint(x);
    unsigned int iy = __float_as_uint(y);
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u || rlw_zeroinfnan(iy)) {
        if (rlw_zeroinfnan(iy)) {
            if (2u * iy == 0u) return 1.0f;
            if (ix == 0x3f800000u) return 1.0f;
            if (2u * ix > 2u * 0x7f800000u || 2u * iy > 2u * 0x7f800000u)
                return RLW_AD(x, y);
            if (2u * ix == 2u * 0x3f800000u) return 1.0f;
            if ((2u * ix < 2u * 0x3f800000u) == !(iy & 0x80000000u))
                return 0.0f;
            return RLW_MU(y, y);
        }
        if (rlw_zeroinfnan(ix)) {
            float x2 = RLW_MU(x, x);
            if ((ix & 0x80000000u) && rlw_checkint(iy) == 1) x2 = -x2;
            return (iy & 0x80000000u) ? RLW_DV(1.0f, x2) : x2;
        }
        if (ix & 0x80000000u) {
            int yint = rlw_checkint(iy);
            if (yint == 0) return __int_as_float(0x7fc00000);
            if (yint == 1) sign_bias = 1u << (5 + 11);
            ix &= 0x7fffffffu;
        }
        if (ix < 0x00800000u) {
            ix = __float_as_uint(RLW_MU(x, 8388608.0f)) & 0x7fffffffu;
            ix -= 23u << 23;
        }
    }
    double logx = rlw_powf_log2(ix);
    double ylogx = RLW_DMU((double)y, logx);
    unsigned int hi = (unsigned int)
        (((unsigned long long)__double_as_longlong(ylogx) >> 47) & 0xffffULL);
    if (hi >= (unsigned int)
            (((unsigned long long)__double_as_longlong(126.0) >> 47) & 0xffffULL)) {
        if (ylogx > 0x1.fffffffd1d571p+6)
            return sign_bias ? __int_as_float(0xff800000)
                             : __int_as_float(0x7f800000);
        if (ylogx <= -150.0) return sign_bias ? -0.0f : 0.0f;
    }
    return rlw_d2f_rn(
        rlw_exp2_core(ylogx, RLW_EXP2F_SHIFT_SCALED,
                      RLW_EXP2F_P0, RLW_EXP2F_P1, RLW_EXP2F_P2, sign_bias));
}

// Fortran x**4 with integer literal exponent: gfortran square-and-multiply.
__device__ float rlw_pow4(float x)
{
    float x2 = RLW_MU(x, x);
    return RLW_MU(x2, x2);
}

// ---------------------------------------------------------------------
// Toolchain probes (compiled with everything else; run by the preflight)
// ---------------------------------------------------------------------

extern "C" __global__ void rlw_probe(const float* x, float* o, int n)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    if (t == 0) o[0] = RLW_MU(x[0], x[1]);          // subnormal product
    if (t == 1) o[1] = rlw_exp(x[2]);               // deep-negative expf
    if (t == 2) o[2] = rlw_log(x[3]);
    if (t == 3) o[3] = rlw_pow(x[4], x[5]);
}

// ---------------------------------------------------------------------
// setcoef -- one thread per column; serial 74-layer loop matching the
// Fortran statement order (see gpuwm/core/rrtmg_lw.py::setcoef).
// Inputs are packed per column; index arrays keep 1-based values.
// ---------------------------------------------------------------------

extern "C" __global__ void rlw_setcoef(
    int ncol, int nlayers, int istart,
    const float* __restrict__ pavel,     // (ncol, nl)
    const float* __restrict__ tavel,     // (ncol, nl)
    const float* __restrict__ tz,        // (ncol, nl+1)
    const float* __restrict__ tbound_v,  // (ncol)
    const float* __restrict__ semiss,    // (ncol, 16)
    const float* __restrict__ coldry,    // (ncol, nl)
    const float* __restrict__ wkl,       // (ncol, MXMOL, nl) C-order
    const float* __restrict__ wbroad,    // (ncol, nl)
    const float* __restrict__ totplnk,   // (181, 16) Fortran order
    const float* __restrict__ totplk16,  // (181)
    const float* __restrict__ preflog,   // (59)
    const float* __restrict__ tref,      // (59)
    const float* __restrict__ chi_mls,   // (7, 59) Fortran order
    int* __restrict__ laytrop_v,         // (ncol)
    int* __restrict__ jp, int* __restrict__ jt, int* __restrict__ jt1,
    float* __restrict__ planklay,        // (ncol, nl, 16)
    float* __restrict__ planklev,        // (ncol, nl+1, 16)
    float* __restrict__ plankbnd,        // (ncol, 16)
    float* __restrict__ colh2o, float* __restrict__ colco2,
    float* __restrict__ colo3, float* __restrict__ coln2o,
    float* __restrict__ colco, float* __restrict__ colch4,
    float* __restrict__ colo2, float* __restrict__ colbrd,
    float* __restrict__ fac00, float* __restrict__ fac01,
    float* __restrict__ fac10, float* __restrict__ fac11,
    float* __restrict__ rat_h2oco2, float* __restrict__ rat_h2oco2_1,
    float* __restrict__ rat_h2oo3, float* __restrict__ rat_h2oo3_1,
    float* __restrict__ rat_h2on2o, float* __restrict__ rat_h2on2o_1,
    float* __restrict__ rat_h2och4, float* __restrict__ rat_h2och4_1,
    float* __restrict__ rat_n2oco2, float* __restrict__ rat_n2oco2_1,
    float* __restrict__ rat_o3co2, float* __restrict__ rat_o3co2_1,
    float* __restrict__ selffac, float* __restrict__ selffrac,
    int* __restrict__ indself,
    float* __restrict__ forfac, float* __restrict__ forfrac,
    int* __restrict__ indfor,
    float* __restrict__ minorfrac, float* __restrict__ scaleminor,
    float* __restrict__ scaleminorn2, int* __restrict__ indminor)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    const int nl = nlayers;
    const float* pav = pavel + (long long)col * nl;
    const float* tav_ = tavel + (long long)col * nl;
    const float* tzc = tz + (long long)col * (nl + 1);
    const float* sem = semiss + (long long)col * NBNDLW;
    const float* cdry = coldry + (long long)col * nl;
    const float* wklc = wkl + (long long)col * MXMOL * nl;
    const float* wbr = wbroad + (long long)col * nl;
#define TOTPLNK(i, b) totplnk[(i - 1) + 181 * (b - 1)]
#define CHI(s, j) chi_mls[(s - 1) + 7 * (j - 1)]
#define OUT1(a) a[(long long)col * nl + (lay - 1)]
#define PLLAY(lay, b) planklay[((long long)col * nl + (lay - 1)) * NBNDLW + (b - 1)]
#define PLLEV(lev, b) planklev[((long long)col * (nl + 1) + (lev)) * NBNDLW + (b - 1)]

    float stpfac = RLW_DV(296.0f, 1013.0f);
    float tbound = tbound_v[col];

    int indbound = (int)RLW_SU(tbound, 159.0f);
    if (indbound < 1) indbound = 1;
    else if (indbound > 180) indbound = 180;
    float tbndfrac = RLW_SU(RLW_SU(tbound, 159.0f), (float)indbound);
    int indlev0 = (int)RLW_SU(tzc[0], 159.0f);
    if (indlev0 < 1) indlev0 = 1;
    else if (indlev0 > 180) indlev0 = 180;
    float t0frac = RLW_SU(RLW_SU(tzc[0], 159.0f), (float)indlev0);
    int laytrop = 0;

    for (int lay = 1; lay <= nl; ++lay) {
        float tavl = tav_[lay - 1];
        int indlay = (int)RLW_SU(tavl, 159.0f);
        if (indlay < 1) indlay = 1;
        else if (indlay > 180) indlay = 180;
        float tlayfrac = RLW_SU(RLW_SU(tavl, 159.0f), (float)indlay);
        float tzl = tzc[lay];
        int indlev = (int)RLW_SU(tzl, 159.0f);
        if (indlev < 1) indlev = 1;
        else if (indlev > 180) indlev = 180;
        float tlevfrac = RLW_SU(RLW_SU(tzl, 159.0f), (float)indlev);

        for (int ib = 1; ib <= 15; ++ib) {
            if (lay == 1) {
                float dbdtlev = RLW_SU(TOTPLNK(indbound + 1, ib),
                                       TOTPLNK(indbound, ib));
                plankbnd[(long long)col * NBNDLW + (ib - 1)] =
                    RLW_MU(sem[ib - 1],
                           RLW_AD(TOTPLNK(indbound, ib),
                                  RLW_MU(tbndfrac, dbdtlev)));
                dbdtlev = RLW_SU(TOTPLNK(indlev0 + 1, ib),
                                 TOTPLNK(indlev0, ib));
                PLLEV(0, ib) = RLW_AD(TOTPLNK(indlev0, ib),
                                      RLW_MU(t0frac, dbdtlev));
            }
            float dbdtlev = RLW_SU(TOTPLNK(indlev + 1, ib),
                                   TOTPLNK(indlev, ib));
            float dbdtlay = RLW_SU(TOTPLNK(indlay + 1, ib),
                                   TOTPLNK(indlay, ib));
            PLLAY(lay, ib) = RLW_AD(TOTPLNK(indlay, ib),
                                    RLW_MU(tlayfrac, dbdtlay));
            PLLEV(lay, ib) = RLW_AD(TOTPLNK(indlev, ib),
                                    RLW_MU(tlevfrac, dbdtlev));
        }
        {
            const int ib = 16;
            if (istart == 16) {
                if (lay == 1) {
                    float dbdtlev = RLW_SU(totplk16[indbound],
                                           totplk16[indbound - 1]);
                    plankbnd[(long long)col * NBNDLW + (ib - 1)] =
                        RLW_MU(sem[ib - 1],
                               RLW_AD(totplk16[indbound - 1],
                                      RLW_MU(tbndfrac, dbdtlev)));
                    dbdtlev = RLW_SU(TOTPLNK(indlev0 + 1, ib),
                                     TOTPLNK(indlev0, ib));
                    PLLEV(0, ib) = RLW_AD(totplk16[indlev0 - 1],
                                          RLW_MU(t0frac, dbdtlev));
                }
                float dbdtlev = RLW_SU(totplk16[indlev],
                                       totplk16[indlev - 1]);
                float dbdtlay = RLW_SU(totplk16[indlay],
                                       totplk16[indlay - 1]);
                PLLAY(lay, ib) = RLW_AD(totplk16[indlay - 1],
                                        RLW_MU(tlayfrac, dbdtlay));
                PLLEV(lay, ib) = RLW_AD(totplk16[indlev - 1],
                                        RLW_MU(tlevfrac, dbdtlev));
            } else {
                if (lay == 1) {
                    float dbdtlev = RLW_SU(TOTPLNK(indbound + 1, ib),
                                           TOTPLNK(indbound, ib));
                    plankbnd[(long long)col * NBNDLW + (ib - 1)] =
                        RLW_MU(sem[ib - 1],
                               RLW_AD(TOTPLNK(indbound, ib),
                                      RLW_MU(tbndfrac, dbdtlev)));
                    dbdtlev = RLW_SU(TOTPLNK(indlev0 + 1, ib),
                                     TOTPLNK(indlev0, ib));
                    PLLEV(0, ib) = RLW_AD(TOTPLNK(indlev0, ib),
                                          RLW_MU(t0frac, dbdtlev));
                }
                float dbdtlev = RLW_SU(TOTPLNK(indlev + 1, ib),
                                       TOTPLNK(indlev, ib));
                float dbdtlay = RLW_SU(TOTPLNK(indlay + 1, ib),
                                       TOTPLNK(indlay, ib));
                PLLAY(lay, ib) = RLW_AD(TOTPLNK(indlay, ib),
                                        RLW_MU(tlayfrac, dbdtlay));
                PLLEV(lay, ib) = RLW_AD(TOTPLNK(indlev, ib),
                                        RLW_MU(tlevfrac, dbdtlev));
            }
        }

        float plog = rlw_log(pav[lay - 1]);
        int jpl = (int)RLW_SU(36.0f, RLW_MU(5.0f, RLW_AD(plog, 0.04f)));
        if (jpl < 1) jpl = 1;
        else if (jpl > 58) jpl = 58;
        OUT1(jp) = jpl;
        int jp1 = jpl + 1;
        float fp = RLW_MU(5.0f, RLW_SU(preflog[jpl - 1], plog));

        int jtl = (int)RLW_AD(3.0f, RLW_DV(RLW_SU(tavl, tref[jpl - 1]),
                                           15.0f));
        if (jtl < 1) jtl = 1;
        else if (jtl > 4) jtl = 4;
        OUT1(jt) = jtl;
        float ft = RLW_SU(RLW_DV(RLW_SU(tavl, tref[jpl - 1]), 15.0f),
                          (float)(jtl - 3));
        int jt1l = (int)RLW_AD(3.0f, RLW_DV(RLW_SU(tavl, tref[jp1 - 1]),
                                            15.0f));
        if (jt1l < 1) jt1l = 1;
        else if (jt1l > 4) jt1l = 4;
        OUT1(jt1) = jt1l;
        float ft1 = RLW_SU(RLW_DV(RLW_SU(tavl, tref[jp1 - 1]), 15.0f),
                           (float)(jt1l - 3));
        float water = RLW_DV(wklc[0 * nl + (lay - 1)], cdry[lay - 1]);
        float scalefac = RLW_DV(RLW_MU(pav[lay - 1], stpfac), tavl);

        if (!(plog <= 4.56f)) {
            laytrop = laytrop + 1;

            OUT1(forfac) = RLW_DV(scalefac, RLW_AD(1.0f, water));
            float factor = RLW_DV(RLW_SU(332.0f, tavl), 36.0f);
            int indforl = min(2, max(1, (int)factor));
            OUT1(indfor) = indforl;
            OUT1(forfrac) = RLW_SU(factor, (float)indforl);

            OUT1(selffac) = RLW_MU(water, OUT1(forfac));
            factor = RLW_DV(RLW_SU(tavl, 188.0f), 7.2f);
            int indselfl = min(9, max(1, (int)factor - 7));
            OUT1(indself) = indselfl;
            OUT1(selffrac) = RLW_SU(factor, (float)(indselfl + 7));

            OUT1(scaleminor) = RLW_DV(pav[lay - 1], tavl);
            OUT1(scaleminorn2) = RLW_MU(
                RLW_DV(pav[lay - 1], tavl),
                RLW_DV(wbr[lay - 1],
                       RLW_AD(cdry[lay - 1], wklc[0 * nl + (lay - 1)])));
            factor = RLW_DV(RLW_SU(tavl, 180.8f), 7.2f);
            int indminorl = min(18, max(1, (int)factor));
            OUT1(indminor) = indminorl;
            OUT1(minorfrac) = RLW_SU(factor, (float)indminorl);

            OUT1(rat_h2oco2) = RLW_DV(CHI(1, jpl), CHI(2, jpl));
            OUT1(rat_h2oco2_1) = RLW_DV(CHI(1, jpl + 1), CHI(2, jpl + 1));
            OUT1(rat_h2oo3) = RLW_DV(CHI(1, jpl), CHI(3, jpl));
            OUT1(rat_h2oo3_1) = RLW_DV(CHI(1, jpl + 1), CHI(3, jpl + 1));
            OUT1(rat_h2on2o) = RLW_DV(CHI(1, jpl), CHI(4, jpl));
            OUT1(rat_h2on2o_1) = RLW_DV(CHI(1, jpl + 1), CHI(4, jpl + 1));
            OUT1(rat_h2och4) = RLW_DV(CHI(1, jpl), CHI(6, jpl));
            OUT1(rat_h2och4_1) = RLW_DV(CHI(1, jpl + 1), CHI(6, jpl + 1));
            OUT1(rat_n2oco2) = RLW_DV(CHI(4, jpl), CHI(2, jpl));
            OUT1(rat_n2oco2_1) = RLW_DV(CHI(4, jpl + 1), CHI(2, jpl + 1));
        } else {
            OUT1(forfac) = RLW_DV(scalefac, RLW_AD(1.0f, water));
            float factor = RLW_DV(RLW_SU(tavl, 188.0f), 36.0f);
            OUT1(indfor) = 3;
            OUT1(forfrac) = RLW_SU(factor, 1.0f);

            OUT1(selffac) = RLW_MU(water, OUT1(forfac));

            OUT1(scaleminor) = RLW_DV(pav[lay - 1], tavl);
            OUT1(scaleminorn2) = RLW_MU(
                RLW_DV(pav[lay - 1], tavl),
                RLW_DV(wbr[lay - 1],
                       RLW_AD(cdry[lay - 1], wklc[0 * nl + (lay - 1)])));
            factor = RLW_DV(RLW_SU(tavl, 180.8f), 7.2f);
            int indminorl = min(18, max(1, (int)factor));
            OUT1(indminor) = indminorl;
            OUT1(minorfrac) = RLW_SU(factor, (float)indminorl);

            OUT1(rat_h2oco2) = RLW_DV(CHI(1, jpl), CHI(2, jpl));
            OUT1(rat_h2oco2_1) = RLW_DV(CHI(1, jpl + 1), CHI(2, jpl + 1));
            OUT1(rat_o3co2) = RLW_DV(CHI(3, jpl), CHI(2, jpl));
            OUT1(rat_o3co2_1) = RLW_DV(CHI(3, jpl + 1), CHI(2, jpl + 1));
        }

        OUT1(colh2o) = RLW_MU(1.e-20f, wklc[0 * nl + (lay - 1)]);
        OUT1(colco2) = RLW_MU(1.e-20f, wklc[1 * nl + (lay - 1)]);
        OUT1(colo3) = RLW_MU(1.e-20f, wklc[2 * nl + (lay - 1)]);
        OUT1(coln2o) = RLW_MU(1.e-20f, wklc[3 * nl + (lay - 1)]);
        OUT1(colco) = RLW_MU(1.e-20f, wklc[4 * nl + (lay - 1)]);
        OUT1(colch4) = RLW_MU(1.e-20f, wklc[5 * nl + (lay - 1)]);
        OUT1(colo2) = RLW_MU(1.e-20f, wklc[6 * nl + (lay - 1)]);
        if (OUT1(colco2) == 0.0f) OUT1(colco2) = RLW_MU(1.e-32f, cdry[lay - 1]);
        if (OUT1(colo3) == 0.0f) OUT1(colo3) = RLW_MU(1.e-32f, cdry[lay - 1]);
        if (OUT1(coln2o) == 0.0f) OUT1(coln2o) = RLW_MU(1.e-32f, cdry[lay - 1]);
        if (OUT1(colco) == 0.0f) OUT1(colco) = RLW_MU(1.e-32f, cdry[lay - 1]);
        if (OUT1(colch4) == 0.0f) OUT1(colch4) = RLW_MU(1.e-32f, cdry[lay - 1]);
        OUT1(colbrd) = RLW_MU(1.e-20f, wbr[lay - 1]);

        float compfp = RLW_SU(1.0f, fp);
        OUT1(fac10) = RLW_MU(compfp, ft);
        OUT1(fac00) = RLW_MU(compfp, RLW_SU(1.0f, ft));
        OUT1(fac11) = RLW_MU(fp, ft1);
        OUT1(fac01) = RLW_MU(fp, RLW_SU(1.0f, ft1));

        OUT1(selffac) = RLW_MU(OUT1(colh2o), OUT1(selffac));
        OUT1(forfac) = RLW_MU(OUT1(colh2o), OUT1(forfac));
    }
    laytrop_v[col] = laytrop;
#undef TOTPLNK
#undef CHI
#undef OUT1
#undef PLLAY
#undef PLLEV
}

// ---------------------------------------------------------------------
// taumol state pack.  The host (gpuwm/core/rrtmg_lw.py) packs the
// setcoef outputs into one float slab FS (NFSLOT, ncol, nl), one int
// slab IS (NISLOT, ncol, nl), a per-column laytrop vector, and wx
// (ncol, MAXXSEC, nl).  Slot order is FROZEN here and mirrored by the
// host packer; band kernels in sibling files use the same macros.
// One thread per (col, lay); inner loop over the band's g-points.
// ---------------------------------------------------------------------

#define SL_PAVEL 0
#define SL_COLDRY 1
#define SL_COLH2O 2
#define SL_COLCO2 3
#define SL_COLO3 4
#define SL_COLN2O 5
#define SL_COLCO 6
#define SL_COLCH4 7
#define SL_COLO2 8
#define SL_COLBRD 9
#define SL_FAC00 10
#define SL_FAC01 11
#define SL_FAC10 12
#define SL_FAC11 13
#define SL_RAT_H2OCO2 14
#define SL_RAT_H2OCO2_1 15
#define SL_RAT_H2OO3 16
#define SL_RAT_H2OO3_1 17
#define SL_RAT_H2ON2O 18
#define SL_RAT_H2ON2O_1 19
#define SL_RAT_H2OCH4 20
#define SL_RAT_H2OCH4_1 21
#define SL_RAT_N2OCO2 22
#define SL_RAT_N2OCO2_1 23
#define SL_RAT_O3CO2 24
#define SL_RAT_O3CO2_1 25
#define SL_SELFFAC 26
#define SL_SELFFRAC 27
#define SL_FORFAC 28
#define SL_FORFRAC 29
#define SL_MINORFRAC 30
#define SL_SCALEMINOR 31
#define SL_SCALEMINORN2 32
#define NFSLOT 33

#define SI_JP 0
#define SI_JT 1
#define SI_JT1 2
#define SI_INDSELF 3
#define SI_INDFOR 4
#define SI_INDMINOR 5
#define NISLOT 6

// 1-based lay; col/nl from enclosing kernel scope.
#define FS(slot) fs[((long long)(slot) * ncol + col) * nl + ((lay) - 1)]
#define IS(slot) isv[((long long)(slot) * ncol + col) * nl + ((lay) - 1)]
#define WXS(ix) wx[((long long)col * MAXXSEC + ((ix) - 1)) * nl + ((lay) - 1)]
// Fortran-order 2-D coefficient table with leading dimension ld.
#define A2(tab, ld, i, g) tab[((i) - 1) + ((g) - 1) * (ld)]
// taug/fracs (ncol, nl, NGPTLW) C-order; ig1 is the 1-based GLOBAL g-point.
#define TAUG(ig1) taug[((long long)col * nl + ((lay) - 1)) * NGPTLW + ((ig1) - 1)]
#define FRACS(ig1) fracs[((long long)col * nl + ((lay) - 1)) * NGPTLW + ((ig1) - 1)]

// UNIVERSAL band-kernel signature (every rlw_taugbN uses exactly this):
//   rlw_taugbN(int ncol, int nl, const int* laytrop_v,
//              const float* fs, const int* isv, const float* wx,
//              const float* chi_mls /* (7,59) Fortran order */,
//              float oneminus,
//              const float* const* tabs /* per-band coefficient table
//                pointers, order fixed by GPU_BAND_TABS in
//                gpuwm/core/rrtmg_lw.py */,
//              float* taug, float* fracs)
// Coefficient tables keep Fortran storage order; index with A2/strides.

#define TAUGB_PROLOGUE \
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (tid >= (long long)ncol * nl) return; \
    int col = (int)(tid / nl); \
    int lay = (int)(tid % nl) + 1; \
    int laytrop = laytrop_v[col]; \
    (void)laytrop; (void)wx; (void)chi_mls; (void)oneminus;

extern "C" __global__ void rlw_taugb1(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    // GPU_BAND_TABS[1] = absa(65,ng), absb(235,ng), selfref(10,ng),
    //                    forref(4,ng), ka_mn2(19,ng), kb_mn2(19,ng),
    //                    fracrefa(ng), fracrefb(ng)
    const float* absa = tabs[0];
    const float* absb = tabs[1];
    const float* selfref = tabs[2];
    const float* forref = tabs[3];
    const float* ka_mn2 = tabs[4];
    const float* kb_mn2 = tabs[5];
    const float* fracrefa = tabs[6];
    const float* fracrefb = tabs[7];
    const int nspa1 = 1, nspb1 = 1, ng1 = 10, gs = 0;

    if (lay <= laytrop) {
        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa1 + 1;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa1 + 1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);
        float pp = FS(SL_PAVEL);
        float corradj = 1.0f;
        if (pp < 250.0f)
            corradj = RLW_SU(1.0f, RLW_DV(RLW_MU(0.15f, RLW_SU(250.0f, pp)),
                                          154.4f));
        float scalen2 = RLW_MU(FS(SL_COLBRD), FS(SL_SCALEMINORN2));
        for (int ig = 1; ig <= ng1; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float taun2 = RLW_MU(scalen2,
                RLW_AD(A2(ka_mn2, 19, indm, ig),
                       RLW_MU(FS(SL_MINORFRAC),
                              RLW_SU(A2(ka_mn2, 19, indm + 1, ig),
                                     A2(ka_mn2, 19, indm, ig)))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absa, 65, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10), A2(absa, 65, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01), A2(absa, 65, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11), A2(absa, 65, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, taun2);
            TAUG(gs + ig) = RLW_MU(corradj, t);
            FRACS(gs + ig) = fracrefa[ig - 1];
        }
    } else {
        int ind0 = ((IS(SI_JP) - 13) * 5 + (IS(SI_JT) - 1)) * nspb1 + 1;
        int ind1 = ((IS(SI_JP) - 12) * 5 + (IS(SI_JT1) - 1)) * nspb1 + 1;
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);
        float pp = FS(SL_PAVEL);
        float corradj = RLW_SU(1.0f, RLW_MU(0.15f, RLW_DV(pp, 95.6f)));
        float scalen2 = RLW_MU(FS(SL_COLBRD), FS(SL_SCALEMINORN2));
        for (int ig = 1; ig <= ng1; ++ig) {
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float taun2 = RLW_MU(scalen2,
                RLW_AD(A2(kb_mn2, 19, indm, ig),
                       RLW_MU(FS(SL_MINORFRAC),
                              RLW_SU(A2(kb_mn2, 19, indm + 1, ig),
                                     A2(kb_mn2, 19, indm, ig)))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10), A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01), A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11), A2(absb, 235, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, taun2);
            TAUG(gs + ig) = RLW_MU(corradj, t);
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }
}
