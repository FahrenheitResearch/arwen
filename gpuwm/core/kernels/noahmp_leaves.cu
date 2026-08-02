// WRF v4.6.1 Noah-MP leaf kernels, pinned at commit
// d66e442fccc04111067e29274c9f9eaccc3cef28.
//
// One thread evaluates one column/case.  Each kernel takes the same flat FP32
// input vector the oracle harness packs (tools/noahmp_wrf461_oracle/
// run_leaves.F90) plus an integer topology vector, and writes the same flat
// output vector, so parity is checked slot for slot with no repacking.
//
// EVERY arithmetic operation goes through __fadd_rn / __fsub_rn / __fmul_rn /
// __fdiv_rn.  NVRTC defaults to --fmad=true, and contraction is the dominant
// bitwise hazard here: gfortran on x86-64 emits no FMA at -O0 without -mfma,
// so a contracted a*b+c on the device is a different number.  Do not "simplify"
// these back to infix operators.
//
// Transcendentals are NOT CUDA's expf/powf/log10f and are NOT "evaluate in
// FP64 and round once" either.  gfortran calls glibc, and none of glibc's FP32
// transcendentals is correctly rounded -- its log10f is still the 1993 SunPro
// FP32 reduction and disagrees with the correctly-rounded result on 18.47% of
// the FP32 domain (0.1,1.0], which is exactly TDFCND's LOG10(SATRATIO) domain.
// r_exp / r_pow / r_log10 below are direct transcriptions of glibc 2.39
// sysdeps/ieee754/flt-32, matching gpuwm/core/noahmp_libm.py statement for
// statement; see that module for the measured verification.  Keep the two in
// step.
//
// gfortran also routes a real constant exponent -- x**2.0, x**3.0 -- through
// powf rather than repeated multiplication, which for the cube differs from
// x*x*x; r_pow reproduces that.

#define AD(a, b) __fadd_rn((a), (b))
#define SU(a, b) __fsub_rn((a), (b))
#define MU(a, b) __fmul_rn((a), (b))
#define DV(a, b) __fdiv_rn((a), (b))
// The FP64 intermediates need the same treatment: NVRTC contracts fma.rn.f64
// just as happily as the FP32 form, and glibc's polynomials are evaluated
// without contraction on x86-64.
#define DAD(a, b) __dadd_rn((a), (b))
#define DSU(a, b) __dsub_rn((a), (b))
#define DMU(a, b) __dmul_rn((a), (b))

// e_logf_data.c
__device__ const double NMP_LOGF_INVC[16] = {
    0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0,
    0x1.3c995b0b80385p+0, 0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0,
    0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0, 0x1.0953f419900a7p+0,
    0x1p+0,               0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
    0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1,
    0x1.767dcf5534862p-1 };
__device__ const double NMP_LOGF_LOGC[16] = {
    -0x1.57bf7808caadep-2, -0x1.2bef0a7c06ddbp-2, -0x1.01eae7f513a67p-2,
    -0x1.b31d8a68224e9p-3, -0x1.6574f0ac07758p-3, -0x1.1aa2bc79c81p-3,
    -0x1.a4e76ce8c0e5ep-4, -0x1.1973c5a611cccp-4, -0x1.252f438e10c1ep-5,
     0x0p+0,                0x1.aa5aa5df25984p-5,  0x1.c5e53aa362eb4p-4,
     0x1.526e57720db08p-3,  0x1.bc2860d22477p-3,   0x1.1058bc8a07ee1p-2,
     0x1.4043057b6ee09p-2 };
#define NMP_LOGF_LN2 0x1.62e42fefa39efp-1
#define NMP_LOGF_A0 (-0x1.00ea348b88334p-2)
#define NMP_LOGF_A1 (0x1.5575b0be00b6ap-2)
#define NMP_LOGF_A2 (-0x1.ffffef20a4123p-2)

// e_exp2f_data.c, shared by expf / exp2f / powf.  EXP2F_TABLE_BITS = 5.
__device__ const unsigned long long NMP_EXP2F_TAB[32] = {
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
#define NMP_EXP2F_P0 0x1.c6af84b912394p-5
#define NMP_EXP2F_P1 0x1.ebfce50fac4f3p-3
#define NMP_EXP2F_P2 0x1.62e42ff0c52d6p-1
#define NMP_EXP2F_SHIFT 0x1.8p+52
#define NMP_EXP2F_SHIFT_SCALED (0x1.8p+52 / 32.0)

// e_powf_log2_data.c.  POWF_SCALE is 1.0 because TOINT_INTRINSICS is 0 on
// x86-64, so every "* POWF_SCALE" in glibc's table is a no-op.
__device__ const double NMP_POWF_INVC[16] = {
    0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0,
    0x1.3c995b0b80385p+0, 0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0,
    0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0, 0x1.0953f419900a7p+0,
    0x1p+0,               0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
    0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1,
    0x1.767dcf5534862p-1 };
__device__ const double NMP_POWF_LOGC[16] = {
    -0x1.efec65b963019p-2, -0x1.b0b6832d4fca4p-2, -0x1.7418b0a1fb77bp-2,
    -0x1.39de91a6dcf7bp-2, -0x1.01d9bf3f2b631p-2, -0x1.97c1d1b3b7afp-3,
    -0x1.2f9e393af3c9fp-3, -0x1.960cbbf788d5cp-4, -0x1.a6f9db6475fcep-5,
     0x0p+0,                0x1.338ca9f24f53dp-4,  0x1.476a9543891bap-3,
     0x1.e840b4ac4e4d2p-3,  0x1.40645f0c6651cp-2,  0x1.88e9c2c1b9ff8p-2,
     0x1.ce0a44eb17bccp-2 };
__device__ const double NMP_POWF_A[5] = {
     0x1.27616c9496e0bp-2, -0x1.71969a075c67ap-2,  0x1.ec70a6ca7baddp-2,
    -0x1.7154748bef6c8p-1,  0x1.71547652ab82bp+0 };

// glibc 2.39 sysdeps/ieee754/flt-32/e_logf.c
__device__ real r_log(real x)
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
    double r = DSU(DMU(z, NMP_LOGF_INVC[i]), 1.0);
    double y0 = DAD(NMP_LOGF_LOGC[i], DMU((double)k, NMP_LOGF_LN2));
    double r2 = DMU(r, r);
    double y = DAD(DMU(NMP_LOGF_A1, r), NMP_LOGF_A2);
    y = DAD(DMU(NMP_LOGF_A0, r2), y);
    y = DAD(DMU(y, r2), DAD(y0, r));
    return __double2float_rn(y);
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_log10f.c -- FP32 arithmetic on top of
// r_log, which is why log10f is much less accurate than logf.
__device__ real r_log10(real x)
{
    const real ivln10 = __int_as_float(0x3ede5bd9);
    const real log10_2hi = __int_as_float(0x3e9a2080);
    const real log10_2lo = __int_as_float(0x355427db);
    const real two25 = __int_as_float(0x4c000000);
    unsigned int hx = __float_as_uint(x);
    int k = 0;
    if ((int)hx < 0x00800000) {
        if ((hx & 0x7fffffffu) == 0u) return DV(-two25, fabsf(x));
        if ((int)hx < 0) return __int_as_float(0x7fc00000);
        k -= 25;
        x = MU(x, two25);
        hx = __float_as_uint(x);
    }
    if (hx >= 0x7f800000u) return AD(x, x);
    k += (int)(hx >> 23) - 127;
    int i = (k < 0) ? 1 : 0;
    hx = (hx & 0x007fffffu) | ((unsigned int)(0x7f - i) << 23);
    real y = (real)(k + i);
    x = __uint_as_float(hx);
    real z = AD(MU(y, log10_2lo), MU(ivln10, r_log(x)));
    return AD(z, MU(y, log10_2hi));
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

// The 32-entry exp2 core shared by glibc's expf and powf.
__device__ double nmp_exp2_core(double xd, double shift,
                               double p0, double p1, double p2,
                               unsigned int sign_bias)
{
    double kd = DAD(xd, shift);
    unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd = DSU(kd, shift);
    double r = DSU(xd, kd);
    unsigned long long t = NMP_EXP2F_TAB[ki & 31ULL];
    t += (ki + (unsigned long long)sign_bias) << (52 - 5);
    double s = __longlong_as_double((long long)t);
    double z = DAD(DMU(p0, r), p1);
    double r2 = DMU(r, r);
    double y = DAD(DMU(p2, r), 1.0);
    y = DAD(DMU(z, r2), y);
    return DMU(y, s);
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_expf.c
__device__ real r_exp(real x)
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
    return nmp_d2f_rn(nmp_exp2_core(
        z, NMP_EXP2F_SHIFT,
        NMP_EXP2F_P0 / 32.0 / 32.0 / 32.0,
        NMP_EXP2F_P1 / 32.0 / 32.0,
        NMP_EXP2F_P2 / 32.0, 0u));
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_powf.c log2_inline
__device__ double nmp_powf_log2(unsigned int ix)
{
    unsigned int tmp = ix - 0x3f330000u;
    int i = (int)((tmp >> 19) & 15u);
    unsigned int top = tmp & 0xff800000u;
    unsigned int iz = ix - top;
    int k = (int)top >> 23;
    double z = (double)__uint_as_float(iz);
    double r = DSU(DMU(z, NMP_POWF_INVC[i]), 1.0);
    double y0 = DAD(NMP_POWF_LOGC[i], (double)k);
    double r2 = DMU(r, r);
    double y = DAD(DMU(NMP_POWF_A[0], r), NMP_POWF_A[1]);
    double p = DAD(DMU(NMP_POWF_A[2], r), NMP_POWF_A[3]);
    double r4 = DMU(r2, r2);
    double q = DAD(DMU(NMP_POWF_A[4], r), y0);
    q = DAD(DMU(p, r2), q);
    return DAD(DMU(y, r4), q);
}

__device__ int nmp_checkint(unsigned int iy)
{
    int e = (int)(iy >> 23 & 0xffu);
    if (e < 0x7f) return 0;
    if (e > 0x7f + 23) return 2;
    if (iy & ((1u << (0x7f + 23 - e)) - 1u)) return 0;
    if (iy & (1u << (0x7f + 23 - e))) return 1;
    return 2;
}

__device__ bool nmp_zeroinfnan(unsigned int ix)
{
    return 2u * ix - 1u >= 2u * 0x7f800000u - 1u;
}

// glibc 2.39 sysdeps/ieee754/flt-32/e_powf.c
__device__ real r_pow(real x, real y)
{
    unsigned int sign_bias = 0u;
    unsigned int ix = __float_as_uint(x);
    unsigned int iy = __float_as_uint(y);
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u || nmp_zeroinfnan(iy)) {
        if (nmp_zeroinfnan(iy)) {
            if (2u * iy == 0u) return 1.0f;
            if (ix == 0x3f800000u) return 1.0f;
            if (2u * ix > 2u * 0x7f800000u || 2u * iy > 2u * 0x7f800000u)
                return AD(x, y);
            if (2u * ix == 2u * 0x3f800000u) return 1.0f;
            if ((2u * ix < 2u * 0x3f800000u) == !(iy & 0x80000000u))
                return 0.0f;
            return MU(y, y);
        }
        if (nmp_zeroinfnan(ix)) {
            real x2 = MU(x, x);
            if ((ix & 0x80000000u) && nmp_checkint(iy) == 1) x2 = -x2;
            return (iy & 0x80000000u) ? DV(1.0f, x2) : x2;
        }
        if (ix & 0x80000000u) {
            int yint = nmp_checkint(iy);
            if (yint == 0) return __int_as_float(0x7fc00000);
            if (yint == 1) sign_bias = 1u << (5 + 11);
            ix &= 0x7fffffffu;
        }
        if (ix < 0x00800000u) {
            ix = __float_as_uint(MU(x, 8388608.0f)) & 0x7fffffffu;
            ix -= 23u << 23;
        }
    }
    double logx = nmp_powf_log2(ix);
    double ylogx = DMU((double)y, logx);
    unsigned int hi = (unsigned int)
        (((unsigned long long)__double_as_longlong(ylogx) >> 47) & 0xffffULL);
    if (hi >= (unsigned int)
            (((unsigned long long)__double_as_longlong(126.0) >> 47) & 0xffffULL)) {
        if (ylogx > 0x1.fffffffd1d571p+6)
            return sign_bias ? __int_as_float(0xff800000)
                             : __int_as_float(0x7f800000);
        if (ylogx <= -150.0) return sign_bias ? -0.0f : 0.0f;
    }
    return nmp_d2f_rn(
        nmp_exp2_core(ylogx, NMP_EXP2F_SHIFT_SCALED,
                      NMP_EXP2F_P0, NMP_EXP2F_P1, NMP_EXP2F_P2, sign_bias));
}

// module_sf_noahmplsm.F:204-220
#define NMP_TFRZ   273.16f
#define NMP_CWAT   4.188e06f
#define NMP_CICE   2.094e06f
#define NMP_CPAIR  1004.64f
#define NMP_TKWAT  0.6f
#define NMP_TKICE  2.2f
#define NMP_RAIR   287.04f
#define NMP_DENH2O 1000.0f
#define NMP_DENICE 917.0f

// Pinned topology: three snow layers over four soil layers.  Layer index k in
// WRF's -NSNOW+1:NSOIL lives at array position k + NMP_OFF.
#define NMP_NSNOW 3
#define NMP_NSOIL 4
#define NMP_NLAY  7
#define NMP_OFF   2

// ------------------------------------------------------------------- ATM ----
// module_sf_noahmplsm.F:1083-1251 under the pinned OPT_SNF = 1.

extern "C" __global__
void noahmp_leaf_atm(const real* __restrict__ xs, const int* __restrict__ ixs,
                     real* __restrict__ ys, int ncase)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncase) return;
    (void)ixs;
    const real* x = xs + (size_t)idx * 11;
    real* y = ys + (size_t)idx * 17;

    real sfcprs = x[0], sfctmp = x[1], q2 = x[2];
    real prcpconv = x[3], prcpnonc = x[4], prcpshcv = x[5];
    // x[6], x[7], x[8] are PRCPSNOW / PRCPGRPL / PRCPHAIL, read only inside
    // the IF(OPT_SNF == 4) block at :1217-1228.
    real soldn = x[9], cosz = x[10];

    real pair = sfcprs;                                            // :1144
    real thair = MU(sfctmp, r_pow(DV(sfcprs, pair),
                                  DV(NMP_RAIR, NMP_CPAIR)));       // :1145
    real qair = q2;                                                // :1147
    real eair = DV(MU(qair, sfcprs),
                   AD(0.622f, MU(0.378f, qair)));                  // :1149
    real rhoair = DV(SU(sfcprs, MU(0.378f, eair)),
                     MU(NMP_RAIR, sfctmp));                        // :1150

    real swdown = (cosz <= 0.0f) ? 0.0f : soldn;                   // :1152-1156
    real solad1 = MU(MU(swdown, 0.7f), 0.5f);                      // :1158
    real solai1 = MU(MU(swdown, 0.3f), 0.5f);                      // :1160

    real prcp = AD(AD(prcpconv, prcpnonc), prcpshcv);              // :1163
    real qprecc = MU(0.10f, prcp);                                 // :1169
    real qprecl = MU(0.90f, prcp);                                 // :1170

    real fp = 0.0f;                                                // :1175
    if (AD(qprecc, qprecl) > 0.0f) {                               // :1176-1177
        fp = DV(AD(qprecc, qprecl), AD(MU(10.0f, qprecc), qprecl));
    }

    real fpice;                                                    // :1183-1195
    if (sfctmp > AD(NMP_TFRZ, 2.5f)) {
        fpice = 0.0f;
    } else if (sfctmp <= AD(NMP_TFRZ, 0.5f)) {
        fpice = 1.0f;
    } else if (sfctmp <= AD(NMP_TFRZ, 2.0f)) {
        fpice = SU(1.0f, AD(-54.632f, MU(0.2f, sfctmp)));
    } else {
        fpice = 0.6f;
    }

    real bdfall = fminf(120.0f,
        AD(67.92f, MU(51.25f, r_exp(DV(SU(sfctmp, NMP_TFRZ), 2.59f)))));  // :1216

    y[0] = thair;
    y[1] = qair;
    y[2] = eair;
    y[3] = rhoair;
    y[4] = qprecc;
    y[5] = qprecl;
    y[6] = solad1;
    y[7] = solad1;                                                 // :1159
    y[8] = solai1;
    y[9] = solai1;                                                 // :1161
    y[10] = swdown;
    y[11] = bdfall;
    y[12] = MU(prcp, SU(1.0f, fpice));                             // :1247
    y[13] = MU(prcp, fpice);                                       // :1248
    y[14] = fp;
    y[15] = fpice;
    y[16] = prcp;
}

// ------------------------------------------------------------------ ESAT ----
// module_sf_noahmplsm.F:4952-5001.  Argument is degrees Celsius.

#define ESAT_POLY(t, k0, k1, k2, k3, k4, k5, k6) \
    MU(100.0f, AD(k0, MU((t), AD(k1, MU((t), AD(k2, MU((t), AD(k3, \
    MU((t), AD(k4, MU((t), AD(k5, MU((t), k6)))))))))))))

extern "C" __global__
void noahmp_leaf_esat(const real* __restrict__ xs, const int* __restrict__ ixs,
                      real* __restrict__ ys, int ncase)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncase) return;
    (void)ixs;
    real t = xs[(size_t)idx * 1];
    real* y = ys + (size_t)idx * 4;

    y[0] = ESAT_POLY(t, 6.107799961f, 4.436518521e-01f, 1.428945805e-02f,
                     2.650648471e-04f, 3.031240396e-06f, 2.034080948e-08f,
                     6.136820929e-11f);
    y[1] = ESAT_POLY(t, 6.109177956f, 5.034698970e-01f, 1.886013408e-02f,
                     4.176223716e-04f, 5.824720280e-06f, 4.838803174e-08f,
                     1.838826904e-10f);
    y[2] = ESAT_POLY(t, 4.438099984e-01f, 2.857002636e-02f, 7.938054040e-04f,
                     1.215215065e-05f, 1.036561403e-07f, 3.532421810e-10f,
                     -7.090244804e-13f);
    y[3] = ESAT_POLY(t, 5.030305237e-01f, 3.773255020e-02f, 1.267995369e-03f,
                     2.477563108e-05f, 3.005693132e-07f, 2.158542548e-09f,
                     7.131097725e-12f);
}

// ---------------------------------------------------------------- ROSR12 ----
// module_sf_noahmplsm.F:5534-5591.

extern "C" __global__
void noahmp_leaf_rosr12(const real* __restrict__ xs,
                        const int* __restrict__ ixs,
                        real* __restrict__ ys, int ncase)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncase) return;
    const real* x = xs + (size_t)idx * 28;
    real* y = ys + (size_t)idx * 21;
    int ntop = ixs[(size_t)idx * 1] + 1;

    real a[NMP_NLAY], b[NMP_NLAY], c[NMP_NLAY], d[NMP_NLAY];
    real p[NMP_NLAY], del[NMP_NLAY];
    for (int i = 0; i < NMP_NLAY; ++i) {
        a[i] = x[i];
        b[i] = x[NMP_NLAY + i];
        c[i] = x[2 * NMP_NLAY + i];
        d[i] = x[3 * NMP_NLAY + i];
        p[i] = 0.0f;
        del[i] = 0.0f;
    }

    c[NMP_NSOIL + NMP_OFF] = 0.0f;                                 // :5565
    int t = ntop + NMP_OFF;
    p[t] = DV(-c[t], b[t]);                                        // :5566
    del[t] = DV(d[t], b[t]);                                       // :5570
    for (int k = ntop + 1; k <= NMP_NSOIL; ++k) {                  // :5574-5578
        int i = k + NMP_OFF;
        real recip = DV(1.0f, AD(b[i], MU(a[i], p[i - 1])));
        p[i] = MU(-c[i], recip);
        del[i] = MU(SU(d[i], MU(a[i], del[i - 1])), recip);
    }
    p[NMP_NSOIL + NMP_OFF] = del[NMP_NSOIL + NMP_OFF];             // :5582
    for (int k = ntop + 1; k <= NMP_NSOIL; ++k) {                  // :5586-5589
        int kk = NMP_NSOIL - k + (ntop - 1) + 1;
        int i = kk + NMP_OFF;
        p[i] = AD(MU(p[i], p[i + 1]), del[i]);
    }

    for (int i = 0; i < NMP_NLAY; ++i) {
        y[i] = p[i];
        y[NMP_NLAY + i] = del[i];
        y[2 * NMP_NLAY + i] = c[i];
    }
}

// ----------------------------------------------------------------- CSNOW ----
// module_sf_noahmplsm.F:2514-2569.  snice/snliq are length NMP_NSNOW covering
// layers -NSNOW+1..0; dzsnso spans the full stack, and only its snow half is
// read.

__device__ void noahmp_csnow_core(
    int isnow, const real* snice, const real* snliq, const real* dzsnso,
    real* tksno, real* cvsno, real* snicev, real* snliqv, real* epore)
{
    real bdsnoi[NMP_NSNOW];
    for (int i = 0; i < NMP_NSNOW; ++i) {
        tksno[i] = 0.0f;
        cvsno[i] = 0.0f;
        snicev[i] = 0.0f;
        snliqv[i] = 0.0f;
        epore[i] = 0.0f;
        bdsnoi[i] = 0.0f;
    }
    for (int k = isnow + 1; k <= 0; ++k) {                         // :2547-2551
        int j = k + NMP_OFF;
        snicev[j] = fminf(1.0f, DV(snice[j], MU(dzsnso[j], NMP_DENICE)));
        epore[j] = SU(1.0f, snicev[j]);
        snliqv[j] = fminf(epore[j], DV(snliq[j], MU(dzsnso[j], NMP_DENH2O)));
    }
    for (int k = isnow + 1; k <= 0; ++k) {                         // :2553-2557
        int j = k + NMP_OFF;
        bdsnoi[j] = DV(AD(snice[j], snliq[j]), dzsnso[j]);
        cvsno[j] = AD(MU(NMP_CICE, snicev[j]), MU(NMP_CWAT, snliqv[j]));
    }
    for (int k = isnow + 1; k <= 0; ++k) {                         // :2561-2567
        int j = k + NMP_OFF;
        tksno[j] = MU(3.2217e-6f, r_pow(bdsnoi[j], 2.0f));
    }
}

extern "C" __global__
void noahmp_leaf_csnow(const real* __restrict__ xs,
                       const int* __restrict__ ixs,
                       real* __restrict__ ys, int ncase)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncase) return;
    const real* x = xs + (size_t)idx * 9;
    real* y = ys + (size_t)idx * 15;
    int isnow = ixs[(size_t)idx * 1];

    real snice[NMP_NSNOW], snliq[NMP_NSNOW], dzsnso[NMP_NLAY];
    for (int i = 0; i < NMP_NSNOW; ++i) {
        snice[i] = x[i];
        snliq[i] = x[NMP_NSNOW + i];
        dzsnso[i] = x[2 * NMP_NSNOW + i];
    }
    // The soil half is poisoned in the oracle too: CSNOW must never read it.
    for (int i = NMP_NSNOW; i < NMP_NLAY; ++i) dzsnso[i] = -9999.0f;

    real tksno[NMP_NSNOW], cvsno[NMP_NSNOW];
    real snicev[NMP_NSNOW], snliqv[NMP_NSNOW], epore[NMP_NSNOW];
    noahmp_csnow_core(isnow, snice, snliq, dzsnso,
                      tksno, cvsno, snicev, snliqv, epore);
    for (int i = 0; i < NMP_NSNOW; ++i) {
        y[i] = tksno[i];
        y[NMP_NSNOW + i] = cvsno[i];
        y[2 * NMP_NSNOW + i] = snicev[i];
        y[3 * NMP_NSNOW + i] = snliqv[i];
        y[4 * NMP_NSNOW + i] = epore[i];
    }
}

// ---------------------------------------------------------------- TDFCND ----
// module_sf_noahmplsm.F:2573-2680.

__device__ real noahmp_tdfcnd_core(real smc, real sh2o, real smcmax,
                                  real quartz)
{
    real satratio = DV(smc, smcmax);                               // :2628
    real thkw = 0.57f;                                             // :2629
    real thko = 2.0f;                                              // :2631
    real thkqtz = 7.7f;                                            // :2634
    real thks = MU(r_pow(thkqtz, quartz),
                   r_pow(thko, SU(1.0f, quartz)));                 // :2637
    real xunfroz = 1.0f;                                           // :2640
    if (smc > 0.0f) xunfroz = DV(sh2o, smc);                       // :2641
    real xu = MU(xunfroz, smcmax);                                 // :2643
    real thksat = MU(MU(r_pow(thks, SU(1.0f, smcmax)),
                        r_pow(NMP_TKICE, SU(smcmax, xu))),
                     r_pow(thkw, xu));                             // :2646-2647
    real gammd = MU(SU(1.0f, smcmax), 2700.0f);                    // :2650
    real thkdry = DV(AD(MU(0.135f, gammd), 64.7f),
                     SU(2700.0f, MU(0.947f, gammd)));              // :2652
    real ake;
    if (AD(sh2o, 0.0005f) < smc) {                                 // :2654
        ake = satratio;
    } else if (satratio > 0.1f) {                                  // :2664
        ake = AD(r_log10(satratio), 1.0f);                         // :2666
    } else {
        ake = 0.0f;                                                // :2671
    }
    return AD(MU(ake, SU(thksat, thkdry)), thkdry);                // :2677
}

extern "C" __global__
void noahmp_leaf_tdfcnd(const real* __restrict__ xs,
                        const int* __restrict__ ixs,
                        real* __restrict__ ys, int ncase)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncase) return;
    (void)ixs;
    const real* x = xs + (size_t)idx * 4;
    ys[(size_t)idx * 1] = noahmp_tdfcnd_core(x[0], x[1], x[2], x[3]);
}

// ------------------------------------------------------------ THERMOPROP ----
// module_sf_noahmplsm.F:2400-2510.  TG, UR, LAT, Z0M, ZLVL and VEGTYP are in
// WRF's argument list but the body never references them, so the kernel does
// not read those slots.

extern "C" __global__
void noahmp_leaf_thermoprop(const real* __restrict__ xs,
                            const int* __restrict__ ixs,
                            real* __restrict__ ys, int ncase)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncase) return;
    const real* x = xs + (size_t)idx * 44;
    real* y = ys + (size_t)idx * 30;
    const int* ix = ixs + (size_t)idx * 4;
    int isnow = ix[0];
    int ist = ix[1];
    // ix[2] is VEGTYP: declared INTENT(IN) at :2428, never referenced.
    bool urban_flag = (ix[3] != 0);

    real dzsnso[NMP_NLAY], stc[NMP_NLAY];
    real snice[NMP_NSNOW], snliq[NMP_NSNOW];
    real smc[NMP_NSOIL], sh2o[NMP_NSOIL];
    real smcmax[NMP_NSOIL], quartz[NMP_NSOIL];
    for (int i = 0; i < NMP_NLAY; ++i) dzsnso[i] = x[i];
    for (int i = 0; i < NMP_NSNOW; ++i) {
        snice[i] = x[NMP_NLAY + i];
        snliq[i] = x[NMP_NLAY + NMP_NSNOW + i];
    }
    int base = NMP_NLAY + 2 * NMP_NSNOW;
    for (int i = 0; i < NMP_NSOIL; ++i) {
        smc[i] = x[base + i];
        sh2o[i] = x[base + NMP_NSOIL + i];
    }
    base += 2 * NMP_NSOIL;
    for (int i = 0; i < NMP_NLAY; ++i) stc[i] = x[base + i];
    base += NMP_NLAY;
    real snowh = x[base];
    real dt = x[base + 1];
    base += 7;                    // snowh, dt, tg, ur, lat, z0m, zlvl
    for (int i = 0; i < NMP_NSOIL; ++i) smcmax[i] = x[base + i];
    real csoil = x[base + NMP_NSOIL];
    for (int i = 0; i < NMP_NSOIL; ++i)
        quartz[i] = x[base + NMP_NSOIL + 1 + i];

    real df[NMP_NLAY], hcpct[NMP_NLAY], fact[NMP_NLAY];
    for (int i = 0; i < NMP_NLAY; ++i) {
        hcpct[i] = 0.0f;                                           // :2446
        df[i] = 0.0f;                                              // :2447
        fact[i] = 0.0f;
    }

    real tksno[NMP_NSNOW], cvsno[NMP_NSNOW];
    real snicev[NMP_NSNOW], snliqv[NMP_NSNOW], epore[NMP_NSNOW];
    noahmp_csnow_core(isnow, snice, snliq, dzsnso,
                      tksno, cvsno, snicev, snliqv, epore);        // :2451

    for (int k = isnow + 1; k <= 0; ++k) {                         // :2454-2457
        int j = k + NMP_OFF;
        df[j] = tksno[j];
        hcpct[j] = cvsno[j];
    }

    for (int k = 1; k <= NMP_NSOIL; ++k) {                         // :2461-2466
        int j = k + NMP_OFF;
        int s = k - 1;
        real sice = SU(smc[s], sh2o[s]);
        real acc = MU(sh2o[s], NMP_CWAT);
        acc = AD(acc, MU(SU(1.0f, smcmax[s]), csoil));
        acc = AD(acc, MU(SU(smcmax[s], smc[s]), NMP_CPAIR));
        acc = AD(acc, MU(sice, NMP_CICE));
        hcpct[j] = acc;
        df[j] = noahmp_tdfcnd_core(smc[s], sh2o[s], smcmax[s], quartz[s]);
    }

    if (urban_flag) {                                              // :2468-2472
        for (int k = 1; k <= NMP_NSOIL; ++k) df[k + NMP_OFF] = 3.24f;
    }

    if (ist == 2) {                                                // :2483-2493
        for (int k = 1; k <= NMP_NSOIL; ++k) {
            int j = k + NMP_OFF;
            if (stc[j] > NMP_TFRZ) {
                hcpct[j] = NMP_CWAT;
                df[j] = NMP_TKWAT;
            } else {
                hcpct[j] = NMP_CICE;
                df[j] = NMP_TKICE;
            }
        }
    }

    for (int k = isnow + 1; k <= NMP_NSOIL; ++k) {                 // :2497-2499
        int j = k + NMP_OFF;
        fact[j] = DV(dt, MU(hcpct[j], dzsnso[j]));
    }

    int one = 1 + NMP_OFF;
    int zero = 0 + NMP_OFF;
    if (isnow == 0) {                                              // :2504
        df[one] = DV(AD(MU(df[one], dzsnso[one]), MU(0.35f, snowh)),
                     AD(snowh, dzsnso[one]));
    } else {                                                       // :2506
        df[one] = DV(AD(MU(df[one], dzsnso[one]),
                        MU(df[zero], dzsnso[zero])),
                     AD(dzsnso[zero], dzsnso[one]));
    }

    for (int i = 0; i < NMP_NLAY; ++i) {
        y[i] = df[i];
        y[NMP_NLAY + i] = hcpct[i];
        y[2 * NMP_NLAY + 3 * NMP_NSNOW + i] = fact[i];
    }
    for (int i = 0; i < NMP_NSNOW; ++i) {
        y[2 * NMP_NLAY + i] = snicev[i];
        y[2 * NMP_NLAY + NMP_NSNOW + i] = snliqv[i];
        y[2 * NMP_NLAY + 2 * NMP_NSNOW + i] = epore[i];
    }
}

// --------------------------------------------------------------- WDFCND1 ----
// module_sf_noahmplsm.F:9153-9188.

extern "C" __global__
void noahmp_leaf_wdfcnd1(const real* __restrict__ xs,
                         const int* __restrict__ ixs,
                         real* __restrict__ ys, int ncase)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncase) return;
    (void)ixs;
    const real* x = xs + (size_t)idx * 6;
    real* y = ys + (size_t)idx * 2;
    real smc = x[0], fcr = x[1], smcmax = x[2];
    real bexp = x[3], dwsat = x[4], dksat = x[5];

    real factr = fmaxf(0.01f, DV(smc, smcmax));                    // :9177
    real expon = AD(bexp, 2.0f);                                   // :9178
    real wdf = MU(dwsat, r_pow(factr, expon));                     // :9179
    wdf = MU(wdf, SU(1.0f, fcr));                                  // :9180
    expon = AD(MU(2.0f, bexp), 3.0f);                              // :9184
    real wcnd = MU(dksat, r_pow(factr, expon));                    // :9185
    wcnd = MU(wcnd, SU(1.0f, fcr));                                // :9186
    y[0] = wdf;
    y[1] = wcnd;
}

// --------------------------------------------------------------- WDFCND2 ----
// module_sf_noahmplsm.F:9192-9232.

extern "C" __global__
void noahmp_leaf_wdfcnd2(const real* __restrict__ xs,
                         const int* __restrict__ ixs,
                         real* __restrict__ ys, int ncase)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncase) return;
    (void)ixs;
    const real* x = xs + (size_t)idx * 6;
    real* y = ys + (size_t)idx * 2;
    real smc = x[0], sice = x[1], smcmax = x[2];
    real bexp = x[3], dwsat = x[4], dksat = x[5];

    real factr1 = DV(0.05f, smcmax);                               // :9216
    real factr2 = fmaxf(0.01f, DV(smc, smcmax));                   // :9217
    factr1 = fminf(factr1, factr2);                                // :9218
    real expon = AD(bexp, 2.0f);                                   // :9219
    real wdf = MU(dwsat, r_pow(factr2, expon));                    // :9220
    if (sice > 0.0f) {                                             // :9222
        // (500*SICE)**3.0 is a real constant exponent: gfortran calls powf,
        // so this rounds once, not three times.
        real vkwgt = DV(1.0f, AD(1.0f, r_pow(MU(500.0f, sice), 3.0f)));
        wdf = AD(MU(vkwgt, wdf),
                 MU(MU(SU(1.0f, vkwgt), dwsat), r_pow(factr1, expon)));
    }
    expon = AD(MU(2.0f, bexp), 3.0f);                              // :9229
    real wcnd = MU(dksat, r_pow(factr2, expon));                   // :9230
    y[0] = wdf;
    y[1] = wcnd;
}
