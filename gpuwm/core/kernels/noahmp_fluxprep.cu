// WRF v4.6.1 Noah-MP flux-preparation leaf kernels -- SFCDIF1, RAGRB, STOMATA
// -- pinned at commit d66e442fccc04111067e29274c9f9eaccc3cef28.
//
// One thread evaluates one column/case.  Each kernel takes the same flat FP32
// input vector the oracle harness packs (tools/noahmp_wrf461_oracle/
// run_fluxprep.F90) plus an integer topology vector, and writes the same flat
// output vector, so parity is checked slot for slot with no repacking.
//
// This is a sibling of noahmp_leaves.cu, not an extension of it: the porting
// lanes run in parallel and none of them may edit one .cu.  The glibc
// transcendental block below is duplicated verbatim from that file and must
// stay in step with it and with gpuwm/core/noahmp_libm.py.  r_atan is new
// here; SFCDIF1 (:4691, :4698) is Noah-MP's only ATAN caller.
//
// EVERY arithmetic operation goes through __fadd_rn / __fsub_rn / __fmul_rn /
// __fdiv_rn / __fsqrt_rn.  NVRTC defaults to --fmad=true, and contraction is
// the dominant bitwise hazard here: gfortran on x86-64 emits no FMA at -O0
// without -mfma, so a contracted a*b+c on the device is a different number.
// Do not "simplify" these back to infix operators.
//
// Transcendentals are NOT CUDA's expf/powf/logf/atanf and are NOT "evaluate in
// FP64 and round once" either.  gfortran calls glibc, and none of glibc's FP32
// transcendentals is correctly rounded.  For atanf specifically, rounding the
// FP64 result once disagrees with glibc on 823,767 of the 16,777,216 FP32
// inputs in [0.5, 1) alone.
//
// gfortran lowers a REAL constant exponent -- x**0.5, x**0.25, x**(-0.25) --
// to powf, but an INTEGER constant exponent to libgcc's __powisf2, which is
// binary exponentiation: FV**3 is fl(x * fl(x*x)), two roundings, not
// powf(x, 3.0)'s one.  nmp_powi3 below is that expansion.
//
// The FP32 atan tables live in __constant__ memory rather than in a local
// literal array: ptxas 12.8's constant folder does not honour
// round-to-nearest-even when it folds differences of FP32 literals, and
// __fsub_rn pins the hardware rounding mode, not the compiler's folder.  The
// entries are written as exact C99 hex-float literals so no decimal rounding
// step sits between this file and the pinned bit patterns.

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


// The 32-entry exp2 core shared by glibc's expf and powf.
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
#define NMP_GRAV   9.80616f
#define NMP_VKC    0.40f
#define NMP_TFRZ   273.16f
#define NMP_CPAIR  1004.64f

// glibc 2.39 sysdeps/ieee754/flt-32/s_atanf.c.  Exhaustively verified against
// the live glibc over all 4,278,190,082 non-NaN FP32 inputs: 0 mismatches.
// The large-argument shortcut fires at |x| >= 2**25, which was measured on the
// oracle host rather than taken from a source comment.
__constant__ float NMP_ATANF_HI[4] = {
    0x1.dac670p-2f, 0x1.921fb4p-1f, 0x1.f730bcp-1f, 0x1.921fb4p+0f };
__constant__ float NMP_ATANF_LO[4] = {
    0x1.586ed2p-28f, 0x1.4442d0p-25f, 0x1.281f68p-25f, 0x1.4442d0p-24f };
__constant__ float NMP_ATANF_T[11] = {
     0x1.555556p-2f, -0x1.99999ap-3f,  0x1.24924ap-3f, -0x1.c71c70p-4f,
     0x1.745cdcp-4f, -0x1.3b0f2ap-4f,  0x1.10d66ap-4f, -0x1.dde2d6p-5f,
     0x1.97b4b2p-5f, -0x1.2b4442p-5f,  0x1.0ad3aep-6f };

__device__ real r_atan(real x)
{
    int hx = (int)__float_as_uint(x);
    int ix = hx & 0x7fffffff;
    if (ix >= 0x4c000000) {                        /* |x| >= 2**25 */
        if (ix > 0x7f800000) return AD(x, x);      /* NaN */
        if (hx > 0) return AD(NMP_ATANF_HI[3], NMP_ATANF_LO[3]);
        return SU(-NMP_ATANF_HI[3], NMP_ATANF_LO[3]);
    }
    int id;
    if (ix < 0x3ee00000) {                         /* |x| < 0.4375 */
        if (ix < 0x31000000) return x;             /* |x| < 2**-29 */
        id = -1;
    } else {
        x = fabsf(x);
        if (ix < 0x3f980000) {                     /* |x| < 1.1875 */
            if (ix < 0x3f300000) {                 /* 0.4375 <= |x| < 0.6875 */
                id = 0;
                x = DV(SU(MU(2.0f, x), 1.0f), AD(2.0f, x));
            } else {                               /* 0.6875 <= |x| < 1.1875 */
                id = 1;
                x = DV(SU(x, 1.0f), AD(x, 1.0f));
            }
        } else if (ix < 0x401c0000) {              /* |x| < 2.4375 */
            id = 2;
            x = DV(SU(x, 1.5f), AD(1.0f, MU(1.5f, x)));
        } else {                                   /* 2.4375 <= |x| < 2**25 */
            id = 3;
            x = DV(-1.0f, x);
        }
    }
    real z = MU(x, x);
    real w = MU(z, z);
    real s1 = MU(z, AD(NMP_ATANF_T[0], MU(w, AD(NMP_ATANF_T[2],
               MU(w, AD(NMP_ATANF_T[4], MU(w, AD(NMP_ATANF_T[6],
               MU(w, AD(NMP_ATANF_T[8], MU(w, NMP_ATANF_T[10])))))))))));
    real s2 = MU(w, AD(NMP_ATANF_T[1], MU(w, AD(NMP_ATANF_T[3],
               MU(w, AD(NMP_ATANF_T[5], MU(w, AD(NMP_ATANF_T[7],
               MU(w, NMP_ATANF_T[9])))))))));
    if (id < 0) return SU(x, MU(x, AD(s1, s2)));
    z = SU(NMP_ATANF_HI[id], SU(SU(MU(x, AD(s1, s2)), NMP_ATANF_LO[id]), x));
    return (hx < 0) ? -z : z;
}

// libgcc __powisf2 for n == 3: n is odd so y = x, and the single loop pass
// squares x and multiplies.  Two roundings, unlike r_pow(x, 3.0f)'s one.
__device__ real nmp_powi3(real x) { return MU(x, MU(x, x)); }

// ----------------------------------------------------------------- RAGRB ----
// module_sf_noahmplsm.F:4483-4579.  TV, VEGTYP, ILOC and JLOC are in WRF's
// argument list and never referenced; MOZG is INTENT(INOUT) but :4536 assigns
// it 0.0 before any read.  Their slots stay in the input vector so the layout
// is identical to the oracle's.

extern "C" __global__
void noahmp_fluxprep_ragrb(const real* __restrict__ xs,
                           const int* __restrict__ ixs,
                           real* __restrict__ ys, int ncase)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncase) return;
    const real* x = xs + (size_t)idx * 17;
    real* y = ys + (size_t)idx * 6;
    int iter = ixs[(size_t)idx * 4];

    real vai = x[0], rhoair = x[1], hg = x[2];
    real tah = x[4], zpd = x[5], z0mg = x[6], z0hg = x[7], hcan = x[8];
    real uc = x[9], z0h = x[10], fv = x[11], cwp = x[12], mpe = x[13];
    real fhg = x[15], dleaf = x[16];

    real mozg = 0.0f;                                              // :4536
    if (iter > 1) {                                                // :4540
        real tmp1 = DV(MU(MU(NMP_VKC, DV(NMP_GRAV, tah)), hg),
                       MU(rhoair, NMP_CPAIR));
        if (fabsf(tmp1) <= mpe) tmp1 = mpe;                        // :4542
        real molg = -DV(MU(1.0f, nmp_powi3(fv)), tmp1);            // :4543
        mozg = fminf(DV(SU(zpd, z0mg), molg), 1.0f);               // :4544
    }

    real fhgnew;
    if (mozg < 0.0f)                                               // :4547
        fhgnew = r_pow(SU(1.0f, MU(15.0f, mozg)), -0.25f);
    else
        fhgnew = AD(1.0f, MU(4.7f, mozg));                         // :4550
    fhg = (iter == 1) ? fhgnew : MU(0.5f, AD(fhg, fhgnew));        // :4552-4556

    real cwpc = r_pow(MU(MU(MU(cwp, vai), hcan), fhg), 0.5f);      // :4557
    real t1 = r_exp(-DV(MU(cwpc, z0hg), hcan));                    // :4560
    real t2 = r_exp(-DV(MU(cwpc, AD(z0h, zpd)), hcan));            // :4561
    real tmprah2 = MU(DV(MU(hcan, r_exp(cwpc)), cwpc), SU(t1, t2));// :4562

    real kh = fmaxf(MU(MU(NMP_VKC, fv), SU(hcan, zpd)), mpe);      // :4566
    real rahg = DV(tmprah2, kh);                                   // :4568

    real tmprb = DV(MU(cwpc, 50.0f),
                    SU(1.0f, r_exp(-DV(cwpc, 2.0f))));             // :4573
    real rb = MU(tmprb, __fsqrt_rn(DV(dleaf, uc)));                // :4574
    rb = fminf(fmaxf(rb, 5.0f), 50.0f);                            // :4575

    y[0] = mozg;
    y[1] = fhg;
    y[2] = 0.0f;                                                   // RAMG :4567
    y[3] = rahg;
    y[4] = rahg;                                                   // RAWG :4569
    y[5] = rb;
}

// --------------------------------------------------------------- SFCDIF1 ----
// module_sf_noahmplsm.F:4583-4743.  ILOC and JLOC are never referenced.
// ZLVL <= ZPD calls wrf_error_fatal at :4651 and stops the model; a device
// thread cannot, so it writes quiet NaN to every output rather than a value
// that could be mistaken for one.  No fixture case takes that path, and the
// CPU reference raises SfcdifDomainError there.

extern "C" __global__
void noahmp_fluxprep_sfcdif1(const real* __restrict__ xs,
                             const int* __restrict__ ixs,
                             real* __restrict__ ys, int ncase)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncase) return;
    const real* x = xs + (size_t)idx * 16;
    real* y = ys + (size_t)idx * 10;
    int iter = ixs[(size_t)idx * 4];
    int mozsgn = ixs[(size_t)idx * 4 + 1];

    real sfctmp = x[0], rhoair = x[1], h = x[2], qair = x[3];
    real zlvl = x[4], zpd = x[5], z0m = x[6], z0h = x[7];
    real ur = x[8], mpe = x[9], moz = x[10];
    real fm = x[11], fh = x[12], fm2 = x[13], fh2 = x[14], fv = x[15];

    real mozold = moz;                                             // :4647
    if (zlvl <= zpd) {                                             // :4649
        for (int k = 0; k < 10; ++k) y[k] = __int_as_float(0x7fc00000);
        return;
    }

    real tmpcm = r_log(DV(SU(zlvl, zpd), z0m));                    // :4654
    real tmpch = r_log(DV(SU(zlvl, zpd), z0h));                    // :4655
    real tmpcm2 = r_log(DV(AD(2.0f, z0m), z0m));                   // :4656
    real tmpch2 = r_log(DV(AD(2.0f, z0h), z0h));                   // :4657

    real moz2;
    if (iter == 1) {                                               // :4659
        fv = 0.0f;
        moz = 0.0f;
        moz2 = 0.0f;
    } else {
        real tvir = MU(AD(1.0f, MU(0.61f, qair)), sfctmp);         // :4665
        real tmp1 = DV(MU(MU(NMP_VKC, DV(NMP_GRAV, tvir)), h),
                       MU(rhoair, NMP_CPAIR));                     // :4666
        if (fabsf(tmp1) <= mpe) tmp1 = mpe;                        // :4667
        real mol = -DV(MU(1.0f, nmp_powi3(fv)), tmp1);             // :4668
        moz = fminf(DV(SU(zlvl, zpd), mol), 1.0f);                 // :4669
        moz2 = fminf(DV(AD(2.0f, z0h), mol), 1.0f);                // :4670
    }

    if (MU(mozold, moz) < 0.0f) mozsgn = mozsgn + 1;               // :4675
    if (mozsgn >= 2) {                                             // :4676
        moz = 0.0f; fm = 0.0f; fh = 0.0f;
        moz2 = 0.0f; fm2 = 0.0f; fh2 = 0.0f;
    }

    real fmnew, fhnew, fm2new, fh2new;
    if (moz < 0.0f) {                                              // :4686
        real t1 = r_pow(SU(1.0f, MU(16.0f, moz)), 0.25f);
        real t2 = r_log(DV(AD(1.0f, MU(t1, t1)), 2.0f));
        real t3 = r_log(DV(AD(1.0f, t1), 2.0f));
        fmnew = AD(SU(AD(MU(2.0f, t3), t2), MU(2.0f, r_atan(t1))),
                   1.5707963f);
        fhnew = MU(2.0f, t2);
        real t12 = r_pow(SU(1.0f, MU(16.0f, moz2)), 0.25f);        // :4694
        real t22 = r_log(DV(AD(1.0f, MU(t12, t12)), 2.0f));
        real t32 = r_log(DV(AD(1.0f, t12), 2.0f));
        fm2new = AD(SU(AD(MU(2.0f, t32), t22), MU(2.0f, r_atan(t12))),
                    1.5707963f);
        fh2new = MU(2.0f, t22);
    } else {                                                       // :4700
        fmnew = MU(-5.0f, moz);
        fhnew = fmnew;
        fm2new = MU(-5.0f, moz2);
        fh2new = fm2new;
    }

    if (iter == 1) {                                               // :4709
        fm = fmnew; fh = fhnew; fm2 = fm2new; fh2 = fh2new;
    } else {                                                       // :4714
        fm = MU(0.5f, AD(fm, fmnew));
        fh = MU(0.5f, AD(fh, fhnew));
        fm2 = MU(0.5f, AD(fm2, fm2new));
        fh2 = MU(0.5f, AD(fh2, fh2new));
    }

    fh = fminf(fh, MU(0.9f, tmpch));                               // :4722
    fm = fminf(fm, MU(0.9f, tmpcm));                               // :4723
    fh2 = fminf(fh2, MU(0.9f, tmpch2));                            // :4724
    fm2 = fminf(fm2, MU(0.9f, tmpcm2));                            // :4725

    real cmfm = SU(tmpcm, fm);                                     // :4727
    real chfh = SU(tmpch, fh);
    real cm2fm2 = SU(tmpcm2, fm2);
    real ch2fh2 = SU(tmpch2, fh2);
    if (fabsf(cmfm) <= mpe) cmfm = mpe;                            // :4731
    if (fabsf(chfh) <= mpe) chfh = mpe;
    if (fabsf(cm2fm2) <= mpe) cm2fm2 = mpe;
    if (fabsf(ch2fh2) <= mpe) ch2fh2 = mpe;
    real cm = DV(MU(NMP_VKC, NMP_VKC), MU(cmfm, cmfm));            // :4735
    real ch = DV(MU(NMP_VKC, NMP_VKC), MU(cmfm, chfh));            // :4736

    fv = MU(ur, __fsqrt_rn(cm));                                   // :4741
    real ch2 = DV(MU(NMP_VKC, fv), ch2fh2);                        // :4742

    y[0] = moz;
    y[1] = (real)mozsgn;
    y[2] = fm;  y[3] = fh;  y[4] = fm2; y[5] = fh2;
    y[6] = fv;  y[7] = cm;  y[8] = ch;  y[9] = ch2;
}

// --------------------------------------------------------------- STOMATA ----
// module_sf_noahmplsm.F:5005-5137, the live OPT_CRS = 1 canopy-conductance
// leaf.  VEGTYP, ILOC and JLOC are never referenced.  NITER is DATA-initialised
// to 3 at :5045 -- a compile-time constant, not a convergence test.

#define NMP_STOMATA_NITER 3

__device__ real nmp_f1(real ab, real tc)                           // :5074
{
    return r_pow(ab, DV(SU(tc, 25.0f), 10.0f));
}

__device__ real nmp_f2(real tc)                                    // :5075
{
    real s = AD(tc, 273.16f);
    return AD(1.0f, r_exp(DV(AD(-2.2e05f, MU(710.0f, s)), MU(8.314f, s))));
}

extern "C" __global__
void noahmp_fluxprep_stomata(const real* __restrict__ xs,
                             const int* __restrict__ ixs,
                             real* __restrict__ ys, int ncase)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncase) return;
    (void)ixs;
    const real* x = xs + (size_t)idx * 25;
    real* y = ys + (size_t)idx * 2;

    real mpe = x[0], apar = x[1], foln = x[2], tv = x[3], ei = x[4];
    real ea = x[5], sfctmp = x[6], sfcprs = x[7], fveg = x[8], o2 = x[9];
    real co2 = x[10], igs = x[11], btran = x[12], rb = x[13];
    real bp = x[14], folnmx = x[15], qe25 = x[16], kc25 = x[17];
    real akc = x[18], ko25 = x[19], ako = x[20], vcmx25 = x[21];
    real avcmx = x[22], c3psn = x[23], mp = x[24];

    real apar_scale = DV(apar, fmaxf(fveg, 1.0e-6f));              // :5083
    real cf = MU(DV(sfcprs, MU(8.314f, sfctmp)), 1.0e06f);         // :5084
    real rs = MU(DV(1.0f, bp), cf);                                // :5085
    real psn = 0.0f;                                               // :5086

    if (apar_scale <= 0.0f) {                                      // :5088
        y[0] = rs;
        y[1] = psn;
        return;
    }

    real fnf = fminf(DV(foln, fmaxf(mpe, folnmx)), 1.0f);          // :5090
    real tc = SU(tv, NMP_TFRZ);                                    // :5091
    real ppf = MU(4.6f, apar_scale);                               // :5092
    real j = MU(ppf, qe25);                                        // :5093
    real kc = MU(kc25, nmp_f1(akc, tc));                           // :5094
    real ko = MU(ko25, nmp_f1(ako, tc));                           // :5095
    real awc = MU(kc, AD(1.0f, DV(o2, ko)));                       // :5096
    real cp = MU(MU(DV(MU(0.5f, kc), ko), o2), 0.21f);             // :5097
    real vcmx = MU(MU(MU(DV(vcmx25, nmp_f2(tc)), fnf), btran),
                   nmp_f1(avcmx, tc));                             // :5098

    real ci = AD(MU(MU(0.7f, co2), c3psn),
                 MU(MU(0.4f, co2), SU(1.0f, c3psn)));              // :5102
    real rlb = DV(rb, cf);                                         // :5106
    real cea = fmaxf(AD(MU(MU(0.25f, ei), c3psn),
                        MU(MU(0.40f, ei), SU(1.0f, c3psn))),
                     fminf(ea, ei));                               // :5110

    for (int it = 0; it < NMP_STOMATA_NITER; ++it) {               // :5114
        real clipped = fmaxf(SU(ci, cp), 0.0f);
        real wj = AD(MU(DV(MU(clipped, j), AD(ci, MU(2.0f, cp))), c3psn),
                     MU(j, SU(1.0f, c3psn)));                      // :5115
        real wc = AD(MU(DV(MU(clipped, vcmx), AD(ci, awc)), c3psn),
                     MU(vcmx, SU(1.0f, c3psn)));                   // :5116
        real we = AD(MU(MU(0.5f, vcmx), c3psn),
                     MU(DV(MU(MU(4000.0f, vcmx), ci), sfcprs),
                        SU(1.0f, c3psn)));                         // :5117
        psn = MU(fminf(fminf(wj, wc), we), igs);                   // :5118

        real cs = fmaxf(SU(co2, MU(MU(MU(1.37f, rlb), sfcprs), psn)), mpe);
        real a = AD(DV(MU(MU(MU(mp, psn), sfcprs), cea), MU(cs, ei)), bp);
        real b = SU(MU(AD(DV(MU(MU(mp, psn), sfcprs), cs), bp), rlb), 1.0f);
        real c = -rlb;
        real disc = __fsqrt_rn(SU(MU(b, b), MU(MU(4.0f, a), c)));
        real q = (b >= 0.0f) ? MU(-0.5f, AD(b, disc))              // :5124
                             : MU(-0.5f, SU(b, disc));             // :5126
        real r1 = DV(q, a);                                        // :5129
        real r2 = DV(c, q);                                        // :5130
        rs = fmaxf(r1, r2);                                        // :5131
        ci = fmaxf(SU(cs, MU(MU(MU(psn, sfcprs), 1.65f), rs)), 0.0f);
    }

    y[0] = MU(rs, cf);                                             // :5136
    y[1] = psn;
}
