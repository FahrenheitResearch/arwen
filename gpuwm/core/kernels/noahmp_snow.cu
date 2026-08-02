// gpuwm/core/kernels/noahmp_snow.cu
//
// CUDA half of the Noah-MP snow-layer port.  Bitwise-equal to
// gpuwm/core/noahmp_snow.py and to the WRF v4.6.1 oracle
// (tree d66e442fccc04111067e29274c9f9eaccc3cef28,
//  sha256(phys/module_sf_noahmplsm.F) =
//  bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282).
//
// Covers module_sf_noahmplsm.F lines 6398-7230: SNOWWATER (6398), SNOWFALL
// (6539), COMBINE (6610), DIVIDE (6792), COMBO (6920), COMPACT (6974) and
// SNOWH2O (7085).
//
// Three rules make bitwise agreement possible and none is optional:
//
// 1. Every float32 operation uses an explicit rounding intrinsic
//    (__fadd_rn/__fsub_rn/__fmul_rn/__fdiv_rn) and every float64 operation
//    inside glibc_expf uses __dadd_rn/__dsub_rn/__dmul_rn/__fma_rn.  That
//    pins the hardware rounding mode AND makes nvcc's contraction pass a
//    no-op, so -fmad=true cannot fuse a site gfortran did not fuse.
// 2. Every constant lives in __constant__ memory as a bit pattern.  ptxas
//    12.8's constant folder does not honour round-to-nearest-even on literal
//    arrays, so a table of FP32 literals can have its differences mis-folded
//    at compile time; __fsub_rn pins the hardware, not the folder.  COMBINE's
//    DZMIN triple is exactly such a table, which is why C_DZMIN is in
//    __constant__ memory rather than a local literal array.
// 3. COMPACT's three EXP calls are glibc expf in the oracle, so glibc_expf
//    below transcribes glibc 2.39's own algorithm.  CUDA's expf, __expf and
//    exp2f are all different functions and none can hold a max_ulp-0 gate.
//
// Layer index convention is WRF's: -NSNOW+1 (top) .. 0 (bottom) for snow,
// 1 .. NSOIL for soil.  DZSNSO is a positive thickness at every leaf
// boundary.  COMBINE's PONDING1/PONDING2 are INTENT(OUT) but assigned on only
// some paths, and gfortran passes scalar dummies by reference, so they are
// taken and returned here rather than being zeroed on entry.
//
// Nothing in this span is option-gated: SNOWWATER's callees run under every
// Registry default, so there is no dead branch to assert off.  The only WRF
// code in range this kernel does not reach is the glacier column's separate
// SNOWWATER copy in module_sf_noahmp_glacier.F.

#define NSNOW 3
#define NSOIL 4
#define NFULL 7 /* NSNOW + NSOIL */

// Flat host<->device layout.  Slot meanings are per leaf; see the comment on
// each kernel.  Kept uniform so one host harness drives all seven.
#define ST_SNOWH   0
#define ST_SNEQV   1
#define ST_SNICE   2  /* 3 slots, j = -2..0 */
#define ST_SNLIQ   5  /* 3 slots            */
#define ST_STC     8  /* 7 slots, j = -2..4 */
#define ST_ZSNSO  15  /* 7 slots            */
#define ST_DZSNSO 22  /* 7 slots            */
#define ST_SH2O   29  /* 4 slots, j = 1..4  */
#define ST_SICE   33  /* 4 slots            */
#define NSTATE    37
#define IN_STRIDE 53  /* NSTATE + up to 16 scalars */
#define OUT_STRIDE 41 /* NSTATE + up to 4 extras   */

// --------------------------------------------------------------------------
// glibc __exp2f_data, as glibc stores it (double bit patterns)
// --------------------------------------------------------------------------
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
// [0] shift_scaled (unused here), [1] shift, [2] invln2_scaled
__constant__ unsigned long long C_EXP2F_MISC[3] = {
    0x42E8000000000000ULL, 0x4338000000000000ULL, 0x40471547652B82FEULL,
};
__constant__ unsigned long long C_EXP2F_POLY_SCALED[3] = {
    0x3EBC6AF84B912394ULL, 0x3F2EBFCE50FAC4F3ULL, 0x3F962E42FF0C52D6ULL,
};
// glibc expf's cutoffs: 0x1.62e42ep6 (overflow), -0x1.9fe368p6 (underflow)
__constant__ unsigned long long C_EXPF_LIMITS[2] = {
    0x40562E42E0000000ULL, 0xC059FE3680000000ULL,
};

// --------------------------------------------------------------------------
// Every float32 constant these leaves use, as bit patterns.
// K_C5 duplicates K_TWO and K_P2 duplicates K_P20 on purpose: the Fortran
// writes them as separate literals in separate expressions, and separate
// slots let the mutation study probe each site independently.
// --------------------------------------------------------------------------
__constant__ unsigned int C_F32[32] = {
    0x00000000u, /*  0 K_ZERO      0.0        */
    0x3F800000u, /*  1 K_ONE       1.0        */
    0x3F000000u, /*  2 K_HALF      0.5        */
    0x40000000u, /*  3 K_TWO       2.0        */
    0x4388947Bu, /*  4 K_TFRZ      273.16     */
    0x48A2E400u, /*  5 K_HFUS      0.3336e6   */
    0x4A7F9D80u, /*  6 K_CWAT      4.188e6    */
    0x49FF9D80u, /*  7 K_CICE      2.094e6    */
    0x447A0000u, /*  8 K_DENH2O    1000.0     */
    0x44654000u, /*  9 K_DENICE    917.0      */
    0x3CAC0831u, /* 10 K_C2        21.0e-3    */
    0x3627C5ACu, /* 11 K_C3        2.5e-6     */
    0x3D23D70Au, /* 12 K_C4        0.04       */
    0x40000000u, /* 13 K_C5        2.0        */
    0x42C80000u, /* 14 K_DM        100.0      */
    0x49A25A80u, /* 15 K_ETA0      1.33e6     */
    0xBD3C6A7Fu, /* 16 K_NEG46EM3  -46.0e-3   */
    0xBDA3D70Au, /* 17 K_NEG008    -0.08      */
    0x3A83126Fu, /* 18 K_P001      0.001      */
    0x3C23D70Au, /* 19 K_P01       0.01       */
    0x358637BDu, /* 20 K_1EM6      1.0e-6     */
    0xBF000000u, /* 21 K_NEGHALF   -0.5       */
    0x43FA0000u, /* 22 K_500       500.0      */
    0x42480000u, /* 23 K_50        50.0       */
    0x3CCCCCCDu, /* 24 K_P025      0.025      */
    0x3DCCCCCDu, /* 25 K_P1        0.1        */
    0x3D4CCCCDu, /* 26 K_P05       0.05       */
    0x3E4CCCCDu, /* 27 K_P20       0.20       */
    0x3E4CCCCDu, /* 28 K_P2        0.2        */
    0x3ECCCCCDu, /* 29 K_MAXLIQ    0.4        */
    0x322BCC77u, /* 30 K_1EM8      1.0e-8     */
    0x459C4000u, /* 31 K_5000      5000.0     */
};
#define K_ZERO      __uint_as_float(C_F32[0])
#define K_ONE       __uint_as_float(C_F32[1])
#define K_HALF      __uint_as_float(C_F32[2])
#define K_TWO       __uint_as_float(C_F32[3])
#define K_TFRZ      __uint_as_float(C_F32[4])
#define K_HFUS      __uint_as_float(C_F32[5])
#define K_CWAT      __uint_as_float(C_F32[6])
#define K_CICE      __uint_as_float(C_F32[7])
#define K_DENH2O    __uint_as_float(C_F32[8])
#define K_DENICE    __uint_as_float(C_F32[9])
#define K_C2        __uint_as_float(C_F32[10])
#define K_C3        __uint_as_float(C_F32[11])
#define K_C4        __uint_as_float(C_F32[12])
#define K_C5        __uint_as_float(C_F32[13])
#define K_DM        __uint_as_float(C_F32[14])
#define K_ETA0      __uint_as_float(C_F32[15])
#define K_NEG46EM3  __uint_as_float(C_F32[16])
#define K_NEG008    __uint_as_float(C_F32[17])
#define K_P001      __uint_as_float(C_F32[18])
#define K_P01       __uint_as_float(C_F32[19])
#define K_1EM6      __uint_as_float(C_F32[20])
#define K_NEGHALF   __uint_as_float(C_F32[21])
#define K_500       __uint_as_float(C_F32[22])
#define K_50        __uint_as_float(C_F32[23])
#define K_P025      __uint_as_float(C_F32[24])
#define K_P1        __uint_as_float(C_F32[25])
#define K_P05       __uint_as_float(C_F32[26])
#define K_P20       __uint_as_float(C_F32[27])
#define K_P2        __uint_as_float(C_F32[28])
#define K_MAXLIQ    __uint_as_float(C_F32[29])
#define K_1EM8      __uint_as_float(C_F32[30])
#define K_5000      __uint_as_float(C_F32[31])

// COMBINE's `DATA DZMIN /0.025, 0.025, 0.1/`.  A literal array of FP32
// constants is precisely the shape ptxas 12.8 mis-folds, so it lives here.
__constant__ unsigned int C_DZMIN[3] = { 0x3CCCCCCDu, 0x3CCCCCCDu, 0x3DCCCCCDu };
#define DZMIN(i) __uint_as_float(C_DZMIN[(i)])

#define QNAN __uint_as_float(0x7FC00000u)

// --------------------------------------------------------------------------
// rounding-pinned primitives
// --------------------------------------------------------------------------
#define FADD(a, b) __fadd_rn((a), (b))
#define FSUB(a, b) __fsub_rn((a), (b))
#define FMUL(a, b) __fmul_rn((a), (b))
#define FDIV(a, b) __fdiv_rn((a), (b))

#define DADD(a, b) __dadd_rn((a), (b))
#define DSUB(a, b) __dsub_rn((a), (b))
#define DMUL(a, b) __dmul_rn((a), (b))
#define DFMA(a, b, c) __fma_rn((a), (b), (c))

// Python's builtin max/min, which the CPU transcription uses, return the
// FIRST operand on a tie.  Fortran MAX/MIN may return either and the values
// are equal, so this is the tie-break to mirror.
__device__ __forceinline__ float f_max(float a, float b) { return (b > a) ? b : a; }
__device__ __forceinline__ float f_min(float a, float b) { return (b < a) ? b : a; }

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
// glibc 2.39 expf -- sysdeps/ieee754/flt-32/e_expf.c, FMA variant
// --------------------------------------------------------------------------
__device__ float glibc_expf(float x)
{
    unsigned int ix = __float_as_uint(x);
    unsigned int abstop = (ix >> 20) & 0x7FFu;

    if (abstop >= 0x42Bu) {                       // top12(88.0f) & 0x7ff
        if (ix == 0xFF800000u) return 0.0f;       // -inf
        if (abstop >= 0x7F8u) return FADD(x, x);  // +-inf or NaN
        if ((double) x > __longlong_as_double(C_EXPF_LIMITS[0]))
            return __uint_as_float(0x7F800000u);  // overflow
        if ((double) x < __longlong_as_double(C_EXPF_LIMITS[1]))
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
// column state, in WRF's index convention
// --------------------------------------------------------------------------
typedef struct {
    int   isnow;
    float snowh, sneqv;
    float snice[NSNOW];    /* j = -NSNOW+1 .. 0 */
    float snliq[NSNOW];
    float stc[NFULL];      /* j = -NSNOW+1 .. NSOIL */
    float zsnso[NFULL];
    float dzsnso[NFULL];
    float sh2o[NSOIL];     /* j = 1 .. NSOIL */
    float sice[NSOIL];
} Col;

#define SNICE(j)  c->snice[(j) + NSNOW - 1]
#define SNLIQ(j)  c->snliq[(j) + NSNOW - 1]
#define STC(j)    c->stc[(j) + NSNOW - 1]
#define ZSNSO(j)  c->zsnso[(j) + NSNOW - 1]
#define DZSNSO(j) c->dzsnso[(j) + NSNOW - 1]
#define SH2O(j)   c->sh2o[(j) - 1]
#define SICE(j)   c->sice[(j) - 1]

__device__ void col_load(Col *c, const float *in, int isnow)
{
    c->isnow = isnow;
    c->snowh = in[ST_SNOWH];
    c->sneqv = in[ST_SNEQV];
    for (int k = 0; k < NSNOW; ++k) { c->snice[k] = in[ST_SNICE + k];
                                      c->snliq[k] = in[ST_SNLIQ + k]; }
    for (int k = 0; k < NFULL; ++k) { c->stc[k]    = in[ST_STC + k];
                                      c->zsnso[k]  = in[ST_ZSNSO + k];
                                      c->dzsnso[k] = in[ST_DZSNSO + k]; }
    for (int k = 0; k < NSOIL; ++k) { c->sh2o[k] = in[ST_SH2O + k];
                                      c->sice[k] = in[ST_SICE + k]; }
}

__device__ void col_store(const Col *c, float *out, int *iout)
{
    *iout = c->isnow;
    out[ST_SNOWH] = c->snowh;
    out[ST_SNEQV] = c->sneqv;
    for (int k = 0; k < NSNOW; ++k) { out[ST_SNICE + k] = c->snice[k];
                                      out[ST_SNLIQ + k] = c->snliq[k]; }
    for (int k = 0; k < NFULL; ++k) { out[ST_STC + k]    = c->stc[k];
                                      out[ST_ZSNSO + k]  = c->zsnso[k];
                                      out[ST_DZSNSO + k] = c->dzsnso[k]; }
    for (int k = 0; k < NSOIL; ++k) { out[ST_SH2O + k] = c->sh2o[k];
                                      out[ST_SICE + k] = c->sice[k]; }
}

// ==========================================================================
// COMBO -- lines 6920-6970
// ==========================================================================
__device__ void combo(float *dz, float *wliq, float *wice, float *t,
                      float dz2, float wliq2, float wice2, float t2)
{
    float dzc = FADD(*dz, dz2);
    float wicec = FADD(*wice, wice2);
    float wliqc = FADD(*wliq, wliq2);
    float h = FADD(FMUL(FADD(FMUL(K_CICE, *wice), FMUL(K_CWAT, *wliq)),
                        FSUB(*t, K_TFRZ)), FMUL(K_HFUS, *wliq));
    float h2 = FADD(FMUL(FADD(FMUL(K_CICE, wice2), FMUL(K_CWAT, wliq2)),
                         FSUB(t2, K_TFRZ)), FMUL(K_HFUS, wliq2));
    float hc = FADD(h, h2);
    float tc;
    if (hc < K_ZERO) {
        tc = FADD(K_TFRZ, FDIV(hc, FADD(FMUL(K_CICE, wicec), FMUL(K_CWAT, wliqc))));
    } else if (hc <= FMUL(K_HFUS, wliqc)) {
        tc = K_TFRZ;
    } else {
        tc = FADD(K_TFRZ, FDIV(FSUB(hc, FMUL(K_HFUS, wliqc)),
                               FADD(FMUL(K_CICE, wicec), FMUL(K_CWAT, wliqc))));
    }
    *dz = dzc; *wice = wicec; *wliq = wliqc; *t = tc;
}

// ==========================================================================
// SNOWFALL -- lines 6539-6606
// ==========================================================================
__device__ void snowfall(Col *c, float dt, float qsnow, float snowhin, float sfctmp)
{
    int newnode = 0;

    if (c->isnow == 0 && qsnow > K_ZERO) {
        c->snowh = FADD(c->snowh, FMUL(snowhin, dt));
        c->sneqv = FADD(c->sneqv, FMUL(qsnow, dt));
    }

    // C.He removed the QSNOW>0 condition so ISNOW can still be adjusted from
    // SNOWH alone when nothing is falling.
    if (c->isnow == 0 && c->snowh >= K_P025) {
        c->isnow = -1;
        newnode = 1;
        DZSNSO(0) = c->snowh;
        c->snowh = K_ZERO;
        STC(0) = f_min(K_TFRZ, sfctmp);
        SNICE(0) = c->sneqv;
        SNLIQ(0) = K_ZERO;
    }

    if (c->isnow < 0 && newnode == 0 && qsnow > K_ZERO) {
        SNICE(c->isnow + 1) = FADD(SNICE(c->isnow + 1), FMUL(qsnow, dt));
        DZSNSO(c->isnow + 1) = FADD(DZSNSO(c->isnow + 1), FMUL(snowhin, dt));
    }
}

// ==========================================================================
// COMPACT -- lines 6974-7081
// ==========================================================================
__device__ void compact(Col *c, float dt, const int *imelt, const float *ficeold)
{
    float fice[NSNOW];
    float burden = K_ZERO;

    for (int j = c->isnow + 1; j <= 0; ++j) {
        int k = j + NSNOW - 1;
        float wx = FADD(SNICE(j), SNLIQ(j));
        fice[k] = FDIV(SNICE(j), wx);
        float voidf = FSUB(K_ONE,
            FDIV(FADD(FDIV(SNICE(j), K_DENICE), FDIV(SNLIQ(j), K_DENH2O)), DZSNSO(j)));

        if (voidf > K_P001 && SNICE(j) > K_P1) {
            float bi = FDIV(SNICE(j), DZSNSO(j));
            float td = f_max(K_ZERO, FSUB(K_TFRZ, STC(j)));
            float dexpf = glibc_expf(FMUL(-K_C4, td));

            float ddz1 = FMUL(-K_C3, dexpf);
            if (bi > K_DM)
                ddz1 = FMUL(ddz1, glibc_expf(FMUL(K_NEG46EM3, FSUB(bi, K_DM))));

            if (SNLIQ(j) > FMUL(K_P01, DZSNSO(j)))
                ddz1 = FMUL(ddz1, K_C5);

            float ddz2 = FDIV(FMUL(-FADD(burden, FMUL(K_HALF, wx)),
                                   glibc_expf(FSUB(FMUL(K_NEG008, td), FMUL(K_C2, bi)))),
                              K_ETA0);

            float ddz3;
            if (imelt[k] == 1) {
                ddz3 = f_max(K_ZERO, FDIV(FSUB(ficeold[k], fice[k]),
                                          f_max(K_1EM6, ficeold[k])));
                ddz3 = FDIV(-ddz3, dt);
            } else {
                ddz3 = K_ZERO;
            }

            float pdzdtc = FMUL(FADD(FADD(ddz1, ddz2), ddz3), dt);
            pdzdtc = f_max(K_NEGHALF, pdzdtc);

            DZSNSO(j) = FMUL(DZSNSO(j), FADD(K_ONE, pdzdtc));
            DZSNSO(j) = f_max(DZSNSO(j), FADD(FDIV(SNICE(j), K_DENICE),
                                              FDIV(SNLIQ(j), K_DENH2O)));
            // C.He: constrain snow density to 50~500 kg/m3
            DZSNSO(j) = f_min(f_max(DZSNSO(j), FDIV(FADD(SNICE(j), SNLIQ(j)), K_500)),
                              FDIV(FADD(SNICE(j), SNLIQ(j)), K_50));
        }

        burden = FADD(burden, wx);
    }
}

// ==========================================================================
// COMBINE -- lines 6610-6788
// ==========================================================================
__device__ void combine(Col *c, float *ponding1, float *ponding2)
{
    int isnow_old = c->isnow;

    for (int j = isnow_old + 1; j <= 0; ++j) {
        if (SNICE(j) <= K_P1) {
            if (j != 0) {
                SNLIQ(j + 1) = FADD(SNLIQ(j + 1), SNLIQ(j));
                SNICE(j + 1) = FADD(SNICE(j + 1), SNICE(j));
                DZSNSO(j + 1) = FADD(DZSNSO(j + 1), DZSNSO(j));
            } else {
                if (c->isnow < -1) {
                    SNLIQ(j - 1) = FADD(SNLIQ(j - 1), SNLIQ(j));
                    SNICE(j - 1) = FADD(SNICE(j - 1), SNICE(j));
                    DZSNSO(j - 1) = FADD(DZSNSO(j - 1), DZSNSO(j));
                } else {
                    if (SNICE(j) >= K_ZERO) {
                        *ponding1 = SNLIQ(j);
                        c->sneqv = SNICE(j);
                        c->snowh = DZSNSO(j);
                    } else { // SNICE over-sublimated earlier
                        *ponding1 = FADD(SNLIQ(j), SNICE(j));
                        if (*ponding1 < K_ZERO) {
                            SICE(1) = FADD(SICE(1),
                                FDIV(*ponding1, FMUL(DZSNSO(1), K_DENH2O)));
                            *ponding1 = K_ZERO;
                        }
                        c->sneqv = K_ZERO;
                        c->snowh = K_ZERO;
                    }
                    SNLIQ(j) = K_ZERO;
                    SNICE(j) = K_ZERO;
                    DZSNSO(j) = K_ZERO;
                }
            }

            if (j > c->isnow + 1 && c->isnow < -1) {
                for (int i = j; i >= c->isnow + 2; --i) {
                    STC(i) = STC(i - 1);
                    SNLIQ(i) = SNLIQ(i - 1);
                    SNICE(i) = SNICE(i - 1);
                    DZSNSO(i) = DZSNSO(i - 1);
                }
            }
            c->isnow = c->isnow + 1;
        }
    }

    if (SICE(1) < K_ZERO) {
        SH2O(1) = FADD(SH2O(1), SICE(1));
        SICE(1) = K_ZERO;
    }

    if (c->isnow == 0) return; // MB: get out if no longer multi-layer

    c->sneqv = K_ZERO;
    c->snowh = K_ZERO;
    float zwice = K_ZERO;
    float zwliq = K_ZERO;

    for (int j = c->isnow + 1; j <= 0; ++j) {
        // Fortran is `SNEQV = SNEQV + SNICE(J) + SNLIQ(J)`, which associates
        // left: (SNEQV + SNICE) + SNLIQ.  Summing the layer pair first is a
        // different rounding and costs a ULP.
        c->sneqv = FADD(FADD(c->sneqv, SNICE(j)), SNLIQ(j));
        c->snowh = FADD(c->snowh, DZSNSO(j));
        zwice = FADD(zwice, SNICE(j));
        zwliq = FADD(zwliq, SNLIQ(j));
    }

    if (c->snowh < K_P025 && c->isnow < 0) {
        c->isnow = 0;
        c->sneqv = zwice;
        *ponding2 = zwliq;
        if (c->sneqv <= K_ZERO) c->snowh = K_ZERO;
    }

    if (c->isnow < -1) {
        isnow_old = c->isnow;
        int mssi = 1;

        for (int i = isnow_old + 1; i <= 0; ++i) {
            if (DZSNSO(i) < DZMIN(mssi - 1)) {
                int neibor;
                if (i == c->isnow + 1) {
                    neibor = i + 1;
                } else if (i == 0) {
                    neibor = i - 1;
                } else {
                    neibor = i + 1;
                    if (FADD(DZSNSO(i - 1), DZSNSO(i)) < FADD(DZSNSO(i + 1), DZSNSO(i)))
                        neibor = i - 1;
                }

                int j, l;
                if (neibor > i) { j = neibor; l = i; } else { j = i; l = neibor; }

                combo(&DZSNSO(j), &SNLIQ(j), &SNICE(j), &STC(j),
                      DZSNSO(l), SNLIQ(l), SNICE(l), STC(l));

                if (j - 1 > c->isnow + 1) {
                    for (int k = j - 1; k >= c->isnow + 2; --k) {
                        STC(k) = STC(k - 1);
                        SNICE(k) = SNICE(k - 1);
                        SNLIQ(k) = SNLIQ(k - 1);
                        DZSNSO(k) = DZSNSO(k - 1);
                    }
                }

                c->isnow = c->isnow + 1;
                if (c->isnow >= -1) break;
            } else {
                mssi = mssi + 1;
            }
        }
    }
}

// ==========================================================================
// DIVIDE -- lines 6792-6916
// ==========================================================================
__device__ void divide(Col *c)
{
    // WRF leaves the slots above ABS(ISNOW) undefined.  Poisoned with a quiet
    // NaN here so an accidental read shows up rather than reading a zero; the
    // oracle build proves no emitted value depends on them by re-running the
    // whole fixture under -finit-real=snan.
    float dz[NSNOW], swice[NSNOW], swliq[NSNOW], tsno[NSNOW];
    for (int k = 0; k < NSNOW; ++k) {
        dz[k] = QNAN; swice[k] = QNAN; swliq[k] = QNAN; tsno[k] = QNAN;
    }
#define DZ(i)    dz[(i) - 1]
#define SWICE(i) swice[(i) - 1]
#define SWLIQ(i) swliq[(i) - 1]
#define TSNO(i)  tsno[(i) - 1]

    int isn = c->isnow;
    for (int j = 1; j <= NSNOW; ++j) {
        if (j <= abs(isn)) {
            DZ(j)    = DZSNSO(j + isn);
            SWICE(j) = SNICE(j + isn);
            SWLIQ(j) = SNLIQ(j + isn);
            TSNO(j)  = STC(j + isn);
        }
    }

    int msno = abs(isn);
    float drr, propor, zwice, zwliq, dtdz;

    if (msno == 1) {
        if (DZ(1) > K_P05) {
            msno = 2;
            DZ(1) = FDIV(DZ(1), K_TWO);
            SWICE(1) = FDIV(SWICE(1), K_TWO);
            SWLIQ(1) = FDIV(SWLIQ(1), K_TWO);
            DZ(2) = DZ(1);
            SWICE(2) = SWICE(1);
            SWLIQ(2) = SWLIQ(1);
            TSNO(2) = TSNO(1);
        }
    }

    if (msno > 1) {
        if (DZ(1) > K_P05) {
            drr = FSUB(DZ(1), K_P05);
            propor = FDIV(drr, DZ(1));
            zwice = FMUL(propor, SWICE(1));
            zwliq = FMUL(propor, SWLIQ(1));
            propor = FDIV(K_P05, DZ(1));
            SWICE(1) = FMUL(propor, SWICE(1));
            SWLIQ(1) = FMUL(propor, SWLIQ(1));
            DZ(1) = K_P05;

            combo(&DZ(2), &SWLIQ(2), &SWICE(2), &TSNO(2),
                  drr, zwliq, zwice, TSNO(1));

            // MB raised this limit from 0.10 to 0.20.
            if (msno <= 2 && DZ(2) > K_P20) {
                msno = 3;
                dtdz = FDIV(FSUB(TSNO(1), TSNO(2)), FDIV(FADD(DZ(1), DZ(2)), K_TWO));
                DZ(2) = FDIV(DZ(2), K_TWO);
                SWICE(2) = FDIV(SWICE(2), K_TWO);
                SWLIQ(2) = FDIV(SWLIQ(2), K_TWO);
                DZ(3) = DZ(2);
                SWICE(3) = SWICE(2);
                SWLIQ(3) = SWLIQ(2);
                TSNO(3) = FSUB(TSNO(2), FDIV(FMUL(dtdz, DZ(2)), K_TWO));
                if (TSNO(3) >= K_TFRZ) {
                    TSNO(3) = TSNO(2);
                } else {
                    TSNO(2) = FADD(TSNO(2), FDIV(FMUL(dtdz, DZ(2)), K_TWO));
                }
            }
        }
    }

    if (msno > 2) {
        if (DZ(2) > K_P2) {
            drr = FSUB(DZ(2), K_P2);
            propor = FDIV(drr, DZ(2));
            zwice = FMUL(propor, SWICE(2));
            zwliq = FMUL(propor, SWLIQ(2));
            propor = FDIV(K_P2, DZ(2));
            SWICE(2) = FMUL(propor, SWICE(2));
            SWLIQ(2) = FMUL(propor, SWLIQ(2));
            DZ(2) = K_P2;
            combo(&DZ(3), &SWLIQ(3), &SWICE(3), &TSNO(3),
                  drr, zwliq, zwice, TSNO(2));
        }
    }

    c->isnow = -msno;

    for (int j = c->isnow + 1; j <= 0; ++j) {
        DZSNSO(j) = DZ(j - c->isnow);
        SNICE(j)  = SWICE(j - c->isnow);
        SNLIQ(j)  = SWLIQ(j - c->isnow);
        STC(j)    = TSNO(j - c->isnow);
    }
#undef DZ
#undef SWICE
#undef SWLIQ
#undef TSNO
}

// ==========================================================================
// SNOWH2O -- lines 7085-7230
// ==========================================================================
__device__ float snowh2o(Col *c, float dt, float qsnfro, float qsnsub, float qrain,
                         float ssi, float snow_ret_fac, float *ponding1, float *ponding2)
{
    float vol_liq[NSNOW], vol_ice[NSNOW], epore[NSNOW];
    for (int k = 0; k < NSNOW; ++k) { vol_liq[k] = K_ZERO; vol_ice[k] = K_ZERO;
                                      epore[k] = K_ZERO; }

    // for the case when SNEQV becomes '0' after 'COMBINE'
    if (c->sneqv == K_ZERO) {
        // Barlage: SH2O -> SICE in v3.6
        SICE(1) = FADD(SICE(1), FDIV(FMUL(FSUB(qsnfro, qsnsub), dt),
                                     FMUL(DZSNSO(1), K_DENH2O)));
        if (SICE(1) < K_ZERO) {
            SH2O(1) = FADD(SH2O(1), SICE(1));
            SICE(1) = K_ZERO;
        }
    }

    // shallow snow without a layer: excess sublimation reduces soil water
    if (c->isnow == 0 && c->sneqv > K_ZERO) {
        float temp = c->sneqv;
        c->sneqv = FADD(FSUB(c->sneqv, FMUL(qsnsub, dt)), FMUL(qsnfro, dt));
        float propor = FDIV(c->sneqv, temp);
        c->snowh = f_max(K_ZERO, FMUL(propor, c->snowh));
        c->snowh = f_min(f_max(c->snowh, FDIV(c->sneqv, K_500)),
                         FDIV(c->sneqv, K_50));

        if (c->sneqv < K_ZERO) {
            SICE(1) = FADD(SICE(1), FDIV(c->sneqv, FMUL(DZSNSO(1), K_DENH2O)));
            c->sneqv = K_ZERO;
            c->snowh = K_ZERO;
        }
        if (SICE(1) < K_ZERO) {
            SH2O(1) = FADD(SH2O(1), SICE(1));
            SICE(1) = K_ZERO;
        }
    }

    if (c->snowh <= K_1EM8 || c->sneqv <= K_1EM6) {
        c->snowh = K_ZERO;
        c->sneqv = K_ZERO;
    }

    // for deep snow
    if (c->isnow < 0) {
        float wgdif = FADD(FSUB(SNICE(c->isnow + 1), FMUL(qsnsub, dt)),
                           FMUL(qsnfro, dt));
        SNICE(c->isnow + 1) = wgdif;
        if (wgdif < K_1EM6 && c->isnow < 0)
            combine(c, ponding1, ponding2);
        // KWM: COMBINE can change ISNOW back to 0
        if (c->isnow < 0) {
            SNLIQ(c->isnow + 1) = FADD(SNLIQ(c->isnow + 1), FMUL(qrain, dt));
            SNLIQ(c->isnow + 1) = f_max(K_ZERO, SNLIQ(c->isnow + 1));
        }
    }

    for (int j = c->isnow + 1; j <= 0; ++j) {
        int k = j + NSNOW - 1;
        vol_ice[k] = f_min(K_ONE, FDIV(SNICE(j), FMUL(DZSNSO(j), K_DENICE)));
        epore[k] = FSUB(K_ONE, vol_ice[k]);
    }

    float qin = K_ZERO;
    float qout = K_ZERO;

    for (int j = c->isnow + 1; j <= 0; ++j) {
        int k = j + NSNOW - 1;
        SNLIQ(j) = FADD(SNLIQ(j), qin);
        vol_liq[k] = FDIV(SNLIQ(j), FMUL(DZSNSO(j), K_DENH2O));
        qout = f_max(K_ZERO, FMUL(FSUB(vol_liq[k], FMUL(ssi, epore[k])), DZSNSO(j)));
        if (j == 0) {
            qout = f_max(FMUL(FSUB(vol_liq[k], epore[k]), DZSNSO(j)),
                         FMUL(FMUL(snow_ret_fac, dt), qout));
        }
        qout = FMUL(qout, K_DENH2O);
        SNLIQ(j) = FSUB(SNLIQ(j), qout);
        if (FDIV(SNLIQ(j), FADD(SNICE(j), SNLIQ(j))) > K_MAXLIQ) {
            qout = FADD(qout, FSUB(SNLIQ(j),
                        FMUL(FDIV(K_MAXLIQ, FSUB(K_ONE, K_MAXLIQ)), SNICE(j))));
            SNLIQ(j) = FMUL(FDIV(K_MAXLIQ, FSUB(K_ONE, K_MAXLIQ)), SNICE(j));
        }
        qin = qout;
    }

    for (int j = c->isnow + 1; j <= 0; ++j) {
        DZSNSO(j) = f_max(DZSNSO(j), FADD(FDIV(SNLIQ(j), K_DENH2O),
                                          FDIV(SNICE(j), K_DENICE)));
    }

    return FDIV(qout, dt);   // QSNBOT, mm/s
}

// ==========================================================================
// SNOWWATER -- lines 6398-6535
// ==========================================================================
__device__ void snowwater(Col *c, float dt, const float *zsoil,
                          const int *imelt, const float *ficeold,
                          float sfctmp, float snowhin, float qsnow,
                          float qsnfro, float qsnsub, float qrain,
                          float ssi, float snow_ret_fac,
                          float *qsnbot, float *snoflow,
                          float *ponding1, float *ponding2)
{
    *snoflow = K_ZERO;
    *ponding1 = K_ZERO;
    *ponding2 = K_ZERO;

    snowfall(c, dt, qsnow, snowhin, sfctmp);

    // MB: do each if block separately
    if (c->isnow < 0) compact(c, dt, imelt, ficeold);
    if (c->isnow < 0) combine(c, ponding1, ponding2);
    if (c->isnow < 0) divide(c);

    *qsnbot = snowh2o(c, dt, qsnfro, qsnsub, qrain, ssi, snow_ret_fac,
                      ponding1, ponding2);

    // set empty snow layers to zero
    for (int iz = -NSNOW + 1; iz <= c->isnow; ++iz) {
        SNICE(iz) = K_ZERO;
        SNLIQ(iz) = K_ZERO;
        STC(iz) = K_ZERO;
        DZSNSO(iz) = K_ZERO;
        ZSNSO(iz) = K_ZERO;
    }

    // equilibrium state of snow in the glacier region
    if (c->sneqv > K_5000) {
        float bdsnow = FDIV(SNICE(0), DZSNSO(0));
        *snoflow = FSUB(c->sneqv, K_5000);
        SNICE(0) = FSUB(SNICE(0), *snoflow);
        DZSNSO(0) = FSUB(DZSNSO(0), FDIV(*snoflow, bdsnow));
        *snoflow = FDIV(*snoflow, dt);
    }

    // sum up snow mass for layered snow
    if (c->isnow < 0) {
        c->sneqv = K_ZERO;
        for (int iz = c->isnow + 1; iz <= 0; ++iz)
            c->sneqv = FADD(FADD(c->sneqv, SNICE(iz)), SNLIQ(iz));
    }

    // Reset ZSNSO and layer thickness DZSNSO
    for (int iz = c->isnow + 1; iz <= 0; ++iz) DZSNSO(iz) = -DZSNSO(iz);

    DZSNSO(1) = zsoil[0];
    for (int iz = 2; iz <= NSOIL; ++iz) DZSNSO(iz) = FSUB(zsoil[iz - 1], zsoil[iz - 2]);

    ZSNSO(c->isnow + 1) = DZSNSO(c->isnow + 1);
    for (int iz = c->isnow + 2; iz <= NSOIL; ++iz)
        ZSNSO(iz) = FADD(ZSNSO(iz - 1), DZSNSO(iz));

    for (int iz = c->isnow + 1; iz <= NSOIL; ++iz) DZSNSO(iz) = -DZSNSO(iz);

    // C.He: update SNOWH for multi-layer snow
    if (c->isnow < 0) {
        c->snowh = K_ZERO;
        for (int iz = c->isnow + 1; iz <= 0; ++iz)
            c->snowh = FADD(c->snowh, DZSNSO(iz));
    }
}

// ==========================================================================
// Entry points.  One thread per fixture case.
//
// in  : [0..36] state, then per-leaf scalars from slot 37.
// iin : [0] isnow, [1..3] imelt(-2..0)
// out : [0..36] state, then per-leaf extras from slot 37.
// iout: [0] isnow
// ==========================================================================

#define TID_GUARD(n) int tid = blockIdx.x * blockDim.x + threadIdx.x; \
                     if (tid >= (n)) return;

extern "C" __global__ void noahmp_snow_combo(const float *in, float *out, int n)
{
    TID_GUARD(n)
    const float *p = in + (size_t) tid * IN_STRIDE;
    float *q = out + (size_t) tid * OUT_STRIDE;
    float dz = p[0], wliq = p[1], wice = p[2], t = p[3];
    combo(&dz, &wliq, &wice, &t, p[4], p[5], p[6], p[7]);
    q[0] = dz; q[1] = wliq; q[2] = wice; q[3] = t;
}

extern "C" __global__ void noahmp_snow_snowfall(const float *in, const int *iin,
                                                float *out, int *iout, int n)
{
    TID_GUARD(n)
    Col col; Col *c = &col;
    const float *p = in + (size_t) tid * IN_STRIDE;
    col_load(c, p, iin[(size_t) tid * 4]);
    snowfall(c, p[NSTATE + 0], p[NSTATE + 1], p[NSTATE + 2], p[NSTATE + 3]);
    col_store(c, out + (size_t) tid * OUT_STRIDE, iout + tid);
}

extern "C" __global__ void noahmp_snow_compact(const float *in, const int *iin,
                                               float *out, int *iout, int n)
{
    TID_GUARD(n)
    Col col; Col *c = &col;
    const float *p = in + (size_t) tid * IN_STRIDE;
    const int *ip = iin + (size_t) tid * 4;
    col_load(c, p, ip[0]);
    compact(c, p[NSTATE + 0], ip + 1, p + NSTATE + 1);
    col_store(c, out + (size_t) tid * OUT_STRIDE, iout + tid);
}

extern "C" __global__ void noahmp_snow_combine(const float *in, const int *iin,
                                               float *out, int *iout, int n)
{
    TID_GUARD(n)
    Col col; Col *c = &col;
    const float *p = in + (size_t) tid * IN_STRIDE;
    float *q = out + (size_t) tid * OUT_STRIDE;
    col_load(c, p, iin[(size_t) tid * 4]);
    float p1 = p[NSTATE + 0], p2 = p[NSTATE + 1];
    combine(c, &p1, &p2);
    col_store(c, q, iout + tid);
    q[NSTATE + 0] = p1; q[NSTATE + 1] = p2;
}

extern "C" __global__ void noahmp_snow_divide(const float *in, const int *iin,
                                              float *out, int *iout, int n)
{
    TID_GUARD(n)
    Col col; Col *c = &col;
    col_load(c, in + (size_t) tid * IN_STRIDE, iin[(size_t) tid * 4]);
    divide(c);
    col_store(c, out + (size_t) tid * OUT_STRIDE, iout + tid);
}

extern "C" __global__ void noahmp_snow_snowh2o(const float *in, const int *iin,
                                               float *out, int *iout, int n)
{
    TID_GUARD(n)
    Col col; Col *c = &col;
    const float *p = in + (size_t) tid * IN_STRIDE;
    float *q = out + (size_t) tid * OUT_STRIDE;
    col_load(c, p, iin[(size_t) tid * 4]);
    float p1 = p[NSTATE + 6], p2 = p[NSTATE + 7];
    float qsnbot = snowh2o(c, p[NSTATE + 0], p[NSTATE + 1], p[NSTATE + 2],
                           p[NSTATE + 3], p[NSTATE + 4], p[NSTATE + 5], &p1, &p2);
    col_store(c, q, iout + tid);
    q[NSTATE + 0] = qsnbot; q[NSTATE + 1] = p1; q[NSTATE + 2] = p2;
}

extern "C" __global__ void noahmp_snow_snowwater(const float *in, const int *iin,
                                                 float *out, int *iout, int n)
{
    TID_GUARD(n)
    Col col; Col *c = &col;
    const float *p = in + (size_t) tid * IN_STRIDE;
    const int *ip = iin + (size_t) tid * 4;
    float *q = out + (size_t) tid * OUT_STRIDE;
    col_load(c, p, ip[0]);
    float qsnbot, snoflow, p1, p2;
    // scalars: dt sfctmp snowhin qsnow qsnfro qsnsub qrain ssi srf
    //          ficeold[3] zsoil[4]
    snowwater(c, p[NSTATE + 0], p + NSTATE + 12, ip + 1, p + NSTATE + 9,
              p[NSTATE + 1], p[NSTATE + 2], p[NSTATE + 3], p[NSTATE + 4],
              p[NSTATE + 5], p[NSTATE + 6], p[NSTATE + 7], p[NSTATE + 8],
              &qsnbot, &snoflow, &p1, &p2);
    col_store(c, q, iout + tid);
    q[NSTATE + 0] = qsnbot; q[NSTATE + 1] = snoflow;
    q[NSTATE + 2] = p1; q[NSTATE + 3] = p2;
}

// Standalone probe for the glibc expf transcription, so a mis-folded table
// entry cannot hide inside a leaf that never reaches it.
extern "C" __global__ void noahmp_snow_expf_probe(const float *x, float *y, int n)
{
    TID_GUARD(n)
    y[tid] = glibc_expf(x[tid]);
}

// Negative control.  CUDA's own expf is a different function from glibc's and
// must be seen to disagree, otherwise the glibc transcription above could be
// passing by coincidence rather than by construction.
extern "C" __global__ void noahmp_snow_expf_native_probe(const float *x, float *y, int n)
{
    TID_GUARD(n)
    y[tid] = expf(x[tid]);
}

// Negative control.  The SNEQV accumulation associates left in Fortran:
// (SNEQV + SNICE) + SNLIQ.  Summing the layer pair first is a different
// rounding, and this kernel exists to show the fixture can see the difference
// -- it is the error the device gate actually caught during this port.
extern "C" __global__ void noahmp_snow_sneqv_misassociated(const float *in, float *out, int n)
{
    TID_GUARD(n)
    const float *p = in + (size_t) tid * IN_STRIDE;
    float *q = out + (size_t) tid * OUT_STRIDE;
    float left = K_ZERO, pair = K_ZERO;
    for (int k = 0; k < NSNOW; ++k) {
        left = FADD(FADD(left, p[ST_SNICE + k]), p[ST_SNLIQ + k]);
        pair = FADD(pair, FADD(p[ST_SNICE + k], p[ST_SNLIQ + k]));
    }
    q[0] = left; q[1] = pair;
}
