// gpuwm/core/kernels/thompson_fork.cu
//
// The operational HRRR fork's Thompson microphysics (RunConfig
// thompson_version = "wrf_39_noaa", mp_physics = 28), one CUDA thread per
// column, transcribed statement for statement from NOAA-EMC/HRRR tag
// v4.1.21 sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_mp_thompson.F (SHA-256
// 4d60011188443eb432294f7693beb64bdbc8f812541a15c7800013060c877283) and
// module_mp_radar.F (08329c87...).  Bare line numbers below are that file's.
//
// WHY ONE COLUMN KERNEL.  The fork's mp_thompson keeps every species in
// tendency form (Xten accumulated in REAL, applied once as X1d + Xten*DT at
// :3688-3756) and re-forms its working values X1d + DT*Xten at fixed points
// (:2967-3049, :3241-3248, :3324-3328).  The earlier multi-kernel fork path
// updated the species in place, which rounds differently at every stage; a
// column oracle against the fork's own Fortran measured that at thousands
// of cells.  This kernel holds the fork's tendencies and working arrays per
// column exactly as the Fortran does, so it can agree with it word for word.
//
// ARITHMETIC CONTRACT (graded by tools/thompson_fork_oracle/
// gpu_column_oracle.py against the unmodified fork under gfortran -O2):
//   * REAL is float, DOUBLE PRECISION is double, and every mixed expression
//     promotes exactly where Fortran does (C++'s usual conversions are
//     Fortran's here: a float operand meets a double one as a double).
//     Literals keep their Fortran kind: 0.1 in a REAL context is 0.1f, and
//     0.1f in a double expression is (double)0.1f, as gfortran evaluates
//     a REAL literal beside a DOUBLE operand.
//   * Operation order is the Fortran's left-to-right order.  The unit
//     compiles with --fmad=false (gpuwm/core/kernels/__init__.py
//     _NO_FMAD_MODULES), so no multiply-add is contracted: gfortran -O2 on
//     x86-64 emits none.
//   * REAL EXP/LOG/**(real)/LOG10 call WOOF's own float32 routines
//     (gfk_exp/gfk_log/gfk_pow from glibc_flt32.cuh, tf_log10f below);
//     DOUBLE EXP/LOG/** call glibc_exp/glibc_log/glibc_pow
//     (glibc_flt64.cuh).  CUDA's builtins are different functions.
//   * x**n with an INTEGER variable n is libgcc's repeated squaring
//     (__powisf2/__powidf2), tf_powif/tf_powid; with an INTEGER constant
//     it is gfortran's inline multiplication chain; **2.0 with a real
//     constant 2.0 is x*x (GCC folds pow(x, 2.0)), **1.0 is x.
//   * NINT is lroundf/lround (round half away from zero).
//
// Derived constants come from thompson_fork_constants.cuh, generated from
// the fork's own thompson_init/radar_init words (make_constants_header.py).
// Lookup tables are the fork's pinned set (gpuwm.core.thompson_contract.
// FORK_TABLE_ASSETS) as float64 Fortran-order device arrays.

// COLUMN ARENA.  The fork's column arrays live in a device workspace, not
// in the thread's local frame: a local frame is reserved by the driver for
// the card's whole resident-thread capacity (48 + 448 B per level before
// this arena, 6.5 GiB on a 188-SM card at 50 levels), while the arena is
// sized to the threads the launcher keeps in flight
// (gpuwm.core.thompson_fork.workspace_bytes).  Each array is one slot of
// nz + 1 levels; level k of slot n for thread t sits at
// (n * len + k) * stride + t, so a warp's accesses to one level coalesce.
// Storage moves; no expression changes, so no word changes.
#define TF_WS_FLOATS 79
#define TF_WS_DOUBLES 7
#define TF_WS_BOOLS 5

template <typename T>
struct TfCol {
    T* p;
    size_t s;
    __device__ __forceinline__ T& operator[](int k) const
    {
        return p[(size_t)k * s];
    }
};

struct TfArena {
    float* f;
    double* d;
    bool* b;
    size_t s;    // threads in flight
    size_t len;  // levels per slot (nz + 1)
    int nf, nd, nb;
    __device__ __forceinline__ TfCol<float> F()
    {
        TfCol<float> c = {f + (size_t)(nf++) * len * s, s};
        return c;
    }
    __device__ __forceinline__ TfCol<double> D()
    {
        TfCol<double> c = {d + (size_t)(nd++) * len * s, s};
        return c;
    }
    __device__ __forceinline__ TfCol<bool> B()
    {
        TfCol<bool> c = {b + (size_t)(nb++) * len * s, s};
        return c;
    }
};

// Fortran element accessors (1-based) of the constant arrays.
// A146: a literal-constant divisor is spelled FDIV, an IEEE quotient on every
// target (NVVM turns a plain one into a reciprocal multiply on compute_120).
#ifndef FDIV
#define FDIV(a, b) __fdiv_rn((a), (b))
#endif
#define CCE(a, n) TF_cce[((a) - 1) + 5 * ((n) - 1)]
#define CCG(a, n) TF_ccg[((a) - 1) + 5 * ((n) - 1)]
#define OCG1(n) TF_ocg1[(n) - 1]
#define OCG2(n) TF_ocg2[(n) - 1]
#define CIE(n) TF_cie[(n) - 1]
#define CIG(n) TF_cig[(n) - 1]
#define CRE(n) TF_cre[(n) - 1]
#define CRG(n) TF_crg[(n) - 1]
#define CSE(n) TF_cse[(n) - 1]
#define CSG(n) TF_csg[(n) - 1]
#define CGE(n) TF_cge[(n) - 1]
#define CGG(n) TF_cgg[(n) - 1]
#define SA(n) TF_sa[(n) - 1]
#define SB(n) TF_sb[(n) - 1]

// ---------------------------------------------------------------------------
// Intrinsics in the oracle's words.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float tf_expf(float x) { return gfk_exp(x); }
__device__ __forceinline__ float tf_logf(float x) { return gfk_log(x); }
__device__ __forceinline__ float tf_powf(float x, float y)
{
    return gfk_pow(x, y);
}
__device__ __forceinline__ double tf_exp(double x) { return glibc_exp(x); }
__device__ __forceinline__ double tf_log(double x) { return glibc_log(x); }
__device__ __forceinline__ double tf_pow(double x, double y)
{
    return glibc_pow(x, y);
}

// REAL LOG10: the algorithm of WOOF's own mynn_glibc_log10f (mynn_pbl.cu),
// FP32 arithmetic on top of gfk_log, pinned against contraction.
__device__ float tf_log10f(float x)
{
    const float ivln10 = __int_as_float(0x3ede5bd9);
    const float log10_2hi = __int_as_float(0x3e9a2080);
    const float log10_2lo = __int_as_float(0x355427db);
    const float two25 = __int_as_float(0x4c000000);
    unsigned int hx = __float_as_uint(x);
    int k = 0;
    if ((int)hx < 0x00800000) {
        if ((hx & 0x7fffffffu) == 0u) return __int_as_float(0xff800000);
        if ((int)hx < 0) return __int_as_float(0x7fc00000);
        k -= 25;
        x = FMUL(x, two25);
        hx = __float_as_uint(x);
    }
    if (hx >= 0x7f800000u) return FADD(x, x);
    k += (int)(hx >> 23) - 127;
    const int i = (k < 0) ? 1 : 0;
    hx = (hx & 0x007fffffu) | ((unsigned int)(0x7f - i) << 23);
    const float y = (float)(k + i);
    x = __uint_as_float(hx);
    const float z = FADD(FMUL(y, log10_2lo), FMUL(ivln10, gfk_log(x)));
    return FADD(z, FMUL(y, log10_2hi));
}

// DOUBLE LOG10 (DLOG10 at :2224/:2268, log10 at :5603).  CUDA's binary64
// log10; both callers round the result (NINT, or to a REAL dBZ), so a
// binary64 last-bit difference from the host's cannot reach a REAL output
// except at an exact rounding tie.
__device__ __forceinline__ double tf_log10(double x) { return log10(x); }

// REAL**INTEGER and DOUBLE**INTEGER with a variable exponent: libgcc's
// __powisf2/__powidf2 (repeated squaring, reciprocal last).
__device__ float tf_powif(float x, int m)
{
    unsigned int n = m < 0 ? 0u - (unsigned int)m : (unsigned int)m;
    float y = (n % 2u) ? x : 1.0f;
    while (n >>= 1) {
        x = x * x;
        if (n % 2u) y = y * x;
    }
    return m < 0 ? 1.0f / y : y;
}

__device__ __forceinline__ int tf_nint(float x) { return (int)lroundf(x); }
__device__ __forceinline__ int tf_nintd(double x) { return (int)lround(x); }

// Fortran MAX/MIN on two REAL / two DOUBLE operands (no NaN reaches them).
__device__ __forceinline__ float fmax_(float a, float b)
{
    return a > b ? a : b;
}
__device__ __forceinline__ float fmin_(float a, float b)
{
    return a < b ? a : b;
}
__device__ __forceinline__ double dmax_(double a, double b)
{
    return a > b ? a : b;
}
__device__ __forceinline__ double dmin_(double a, double b)
{
    return a < b ? a : b;
}

// ---------------------------------------------------------------------------
// Support functions (:5007-5170, :4579-4615, :4802-4877).
// ---------------------------------------------------------------------------

// RSLF (:5007-5037).
__device__ float tf_rslf(float P, float T)
{
    const float C0 = .611583699E03f, C1 = .444606896E02f,
                C2 = .143177157E01f, C3 = .264224321E-1f,
                C4 = .299291081E-3f, C5 = .203154182E-5f,
                C6 = .702620698E-8f, C7 = .379534310E-11f,
                C8 = -.321582393E-13f;
    const float X = fmax_(-80.0f, T - 273.16f);
    float ESL = C0 + X * (C1 + X * (C2 + X * (C3 + X * (C4 + X * (C5
                + X * (C6 + X * (C7 + X * C8)))))));
    ESL = fmin_(ESL, P * 0.15f);
    return .622f * ESL / (P - ESL);
}

// RSIF (:5042-5068).
__device__ float tf_rsif(float P, float T)
{
    const float C0 = .609868993E03f, C1 = .499320233E02f,
                C2 = .184672631E01f, C3 = .402737184E-1f,
                C4 = .565392987E-3f, C5 = .521693933E-5f,
                C6 = .307839583E-7f, C7 = .105785160E-9f,
                C8 = .161444444E-12f;
    const float X = fmax_(-80.0f, T - 273.16f);
    float ESI = C0 + X * (C1 + X * (C2 + X * (C3 + X * (C4 + X * (C5
                + X * (C6 + X * (C7 + X * C8)))))));
    ESI = fmin_(ESI, P * 0.15f);
    return .622f * ESI / (P - ESI);
}

// iceDeMott (:5071-5138): the live statements of the DeMott (2010) branch.
__device__ float tf_ice_demott(float tempc, float rho, float nifa)
{
    const float RHO_NOT0 = 101325.0f / (287.05f * 273.15f);
    const float nifa_cc = nifa * RHO_NOT0 * 1.E-6f / rho;
    float xni = (5.94e-5f * tf_powf(-tempc, 3.33f))
              * tf_powf(nifa_cc, (-0.0264f * (tempc)) + 0.0033f);
    xni = FDIV(xni * rho, RHO_NOT0) * 1000.0f;
    return fmax_(0.0f, xni);
}

// iceKoop (:5145-5170).
__device__ float tf_ice_koop(float temp, float qv, float qvs, float naero,
                             float DT)
{
    float xni = 0.0f;
    const float satw = qv / qvs;
    const float mu_diff = 210368.0f + (131.438f * temp)
                        - (3.32373E6f / temp)
                        - (41729.1f * tf_logf(temp));
    const float a_w_i = tf_expf(mu_diff / (TF_R_uni * temp));
    const float delta_aw = satw - a_w_i;
    float log_J_rate = -906.7f + (8502.0f * delta_aw)
                     - (26924.0f * delta_aw * delta_aw)
                     + (29180.0f * delta_aw * delta_aw * delta_aw);
    log_J_rate = fmin_(20.0f, log_J_rate);
    const float J_rate = tf_powf(10.0f, log_J_rate);
    const float prob_h = fmin_(1.0f - tf_expf(-J_rate * TF_ar_volume * DT),
                               1.0f);
    if (prob_h > 0.0f) xni = fmin_(prob_h * naero, 1000.E3f);
    return fmax_(0.0f, xni);
}

// Eff_aero (:4579-4615); species 0 = 'r', 1 = 's', 2 = 'g'.
__device__ float tf_eff_aero(float D, float Da, float visc, float rhoa,
                             float Temp, int species)
{
    const float boltzman = 1.3806503E-23f;
    const float meanPath = 0.0256E-6f;
    float vt = 1.0f;
    if (species == 0) {
        vt = -0.1021f + 4.932E3f * D - 0.9551E6f * D * D
           + 0.07934E9f * D * D * D - 0.002362E12f * D * D * D * D;
    } else if (species == 1) {
        vt = TF_av_s * tf_powf(D, TF_bv_s);
    } else if (species == 2) {
        vt = TF_av_g * tf_powf(D, TF_bv_g);
    }
    const float Cc = 1.0f + 2.0f * meanPath / Da
                   * (1.257f + 0.4f * tf_expf(FDIV(-0.55f * Da, meanPath)));
    const float diff = boltzman * Temp * Cc / (3.0f * TF_PI * visc * Da);
    const float Re = 0.5f * rhoa * D * vt / visc;
    const float Sc = visc / (rhoa * diff);
    const float St = Da * Da * vt * 1000.0f / (9.0f * visc * D);
    const float aval = 1.0f + tf_logf(1.0f + Re);
    const float St2 = (1.2f + 1.0f / 12.0f * aval) / (1.0f + aval);
    float Eff = 4.0f / (Re * Sc) * (1.0f + 0.4f * sqrtf(Re)
                * tf_powf(Sc, 0.3333f) + 0.16f * sqrtf(Re) * sqrtf(Sc))
              + 4.0f * Da / D * (0.02f + Da / D * (1.0f + 2.0f * sqrtf(Re)));
    if (St > St2) {
        Eff = Eff + tf_powf((St - St2) / (St - St2 + 0.666667f), 1.5f);
    }
    return fmax_(1.E-5f, fmin_(Eff, 1.0f));
}

// activ_ncloud (:4802-4877); tnccn_act is the float64 Fortran-order
// (7,9,7,5,4) copy of the REAL*4 CCN_ACTIVATE.BIN table.
__device__ float tf_activ_ncloud(float Tt, float Ww, float NCCN,
                                 const double* __restrict__ tnccn_act)
{
    float n_local = NCCN * 1.E-6f;
    float w_local = Ww;
    if (n_local >= TF_ta_Na[TF_ntb_arc - 1]) {
        n_local = TF_ta_Na[TF_ntb_arc - 1] - 1.0f;
    } else if (n_local <= TF_ta_Na[0]) {
        n_local = TF_ta_Na[0] + 1.0f;
    }
    int n;
    for (n = 2; n <= TF_ntb_arc; ++n) {
        if (n_local >= TF_ta_Na[n - 2] && n_local < TF_ta_Na[n - 1]) break;
    }
    const int i = n;
    const float x1 = tf_logf(TF_ta_Na[i - 2]);
    const float x2 = tf_logf(TF_ta_Na[i - 1]);
    if (w_local >= TF_ta_Ww[TF_ntb_arw - 1]) {
        w_local = TF_ta_Ww[TF_ntb_arw - 1] - 1.0f;
    } else if (w_local <= TF_ta_Ww[0]) {
        w_local = TF_ta_Ww[0] + 0.001f;
    }
    for (n = 2; n <= TF_ntb_arw; ++n) {
        if (w_local >= TF_ta_Ww[n - 2] && w_local < TF_ta_Ww[n - 1]) break;
    }
    const int j = n;
    const float y1 = tf_logf(TF_ta_Ww[j - 2]);
    const float y2 = tf_logf(TF_ta_Ww[j - 1]);
    const int k = max(1, min(tf_nint((Tt - TF_ta_Tk[0]) * 0.1f) + 1,
                             TF_ntb_art));
    const int l = 3;
    const int m = 2;
#define TCCN(a, b, c, d, e) ((float)tnccn_act[((a) - 1) + 7 * (((b) - 1) \
        + 9 * (((c) - 1) + 7 * (((d) - 1) + 5 * ((e) - 1))))])
    const float A = TCCN(i - 1, j - 1, k, l, m);
    const float B = TCCN(i, j - 1, k, l, m);
    const float C = TCCN(i, j, k, l, m);
    const float D = TCCN(i - 1, j, k, l, m);
#undef TCCN
    const float nx = tf_logf(n_local);
    const float wy = tf_logf(w_local);
    const float t = (nx - x1) / (x2 - x1);
    const float u = (wy - y1) / (y2 - y1);
    const float fraction = (1.0f - t) * (1.0f - u) * A + t * (1.0f - u) * B
                         + t * u * C + (1.0f - t) * u * D;
    return NCCN * fraction;
}

// The decade bin of a REAL content (e.g. :2162-2174): NINT(ALOG10(x)), the
// first of nn-1..nn+1 with 1 <= x/10**nn < 10 (or nn+1), then the bin.
__device__ int tf_decade_bin(float x, int first, int ntb)
{
    const int nic = tf_nint(tf_log10f(x));
    int n = nic - 1;
    for (int nn = nic - 1; nn <= nic + 1; ++nn) {
        n = nn;
        if ((x / tf_powif(10.0f, nn)) >= 1.0f
                && (x / tf_powif(10.0f, nn)) < 10.0f) break;
    }
    int idx = (int)(x / tf_powif(10.0f, n)) + 10 * (n - first)
            - (n - first);
    return max(1, min(idx, ntb));
}

// The same for a DOUBLE intercept (:2224-2232, :2268-2276).
__device__ int tf_decade_bin_d(double x, int first, int ntb)
{
    const int nir = tf_nintd(tf_log10(x));
    int n = nir - 1;
    for (int nn = nir - 1; nn <= nir + 1; ++nn) {
        n = nn;
        if ((x / (double)tf_powif(10.0f, nn)) >= 1.0
                && (x / (double)tf_powif(10.0f, nn)) < 10.0) break;
    }
    int idx = (int)(x / (double)tf_powif(10.0f, n)) + 10 * (n - first)
            - (n - first);
    return max(1, min(idx, ntb));
}

// Snow moment fits (:1954-2024 and their copies): the general form of
// loga_/b_ for moment index `m`, in Fortran's association.
__device__ __forceinline__ float tf_loga(float tc0, float m)
{
    return SA(1) + SA(2) * tc0 + SA(3) * m
         + SA(4) * tc0 * m + SA(5) * tc0 * tc0
         + SA(6) * m * m + SA(7) * tc0 * tc0 * m
         + SA(8) * tc0 * m * m + SA(9) * tc0 * tc0 * tc0
         + SA(10) * m * m * m;
}

__device__ __forceinline__ float tf_b(float tc0, float m)
{
    return SB(1) + SB(2) * tc0 + SB(3) * m + SB(4) * tc0 * m
         + SB(5) * tc0 * tc0 + SB(6) * m * m
         + SB(7) * tc0 * tc0 * m + SB(8) * tc0 * m * m
         + SB(9) * tc0 * tc0 * tc0 + SB(10) * m * m * m;
}

// :1969-1972, the zeroth moment's own (shorter) fit.
__device__ __forceinline__ float tf_loga0(float tc0)
{
    return SA(1) + SA(2) * tc0 + SA(5) * tc0 * tc0
         + SA(9) * tc0 * tc0 * tc0;
}
__device__ __forceinline__ float tf_b0(float tc0)
{
    return SB(1) + SB(2) * tc0 + SB(5) * tc0 * tc0
         + SB(9) * tc0 * tc0 * tc0;
}

// :1975-1985, the first moment's fit (cse replaced by 1 in the source).
__device__ __forceinline__ float tf_loga1(float tc0)
{
    return SA(1) + SA(2) * tc0 + SA(3)
         + SA(4) * tc0 + SA(5) * tc0 * tc0
         + SA(6) + SA(7) * tc0 * tc0
         + SA(8) * tc0 + SA(9) * tc0 * tc0 * tc0
         + SA(10);
}
__device__ __forceinline__ float tf_b1(float tc0)
{
    return SB(1) + SB(2) * tc0 + SB(3) + SB(4) * tc0
         + SB(5) * tc0 * tc0 + SB(6)
         + SB(7) * tc0 * tc0 + SB(8) * tc0
         + SB(9) * tc0 * tc0 * tc0 + SB(10);
}

// ---------------------------------------------------------------------------
// module_mp_radar.F's melting-particle backscatter (:265-620), COMPLEX*16.
// Products and quotients follow gfortran's inline expansion (Fortran rules:
// a*c - b*d, a*d + b*c; Smith's quotient with range reduction).  SQRT, LOG
// and ABS of a complex are binary64 library calls in the oracle (csqrt,
// clog, cabs); their last-bit differences reach ze_snow only through a
// SNGL of a 50-bin sum, so these generic forms are used.
// ---------------------------------------------------------------------------

struct tfc { double re, im; };

__device__ __forceinline__ tfc tfc_mk(double re, double im)
{
    tfc c; c.re = re; c.im = im; return c;
}
__device__ __forceinline__ tfc tfc_add(tfc a, tfc b)
{
    return tfc_mk(a.re + b.re, a.im + b.im);
}
__device__ __forceinline__ tfc tfc_sub(tfc a, tfc b)
{
    return tfc_mk(a.re - b.re, a.im - b.im);
}
__device__ __forceinline__ tfc tfc_addr(tfc a, double r)
{
    return tfc_mk(a.re + r, a.im);
}
__device__ __forceinline__ tfc tfc_subr(tfc a, double r)
{
    return tfc_mk(a.re - r, a.im);
}
__device__ __forceinline__ tfc tfc_rmul(double s, tfc a)
{
    return tfc_mk(s * a.re, s * a.im);
}
__device__ __forceinline__ tfc tfc_mul(tfc a, tfc b)
{
    return tfc_mk(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
__device__ tfc tfc_div(tfc a, tfc b)
{
    double tr, ti;
    if (fabs(b.re) < fabs(b.im)) {
        const double ratio = b.re / b.im;
        const double div = (b.re * ratio) + b.im;
        tr = (a.re * ratio) + a.im;
        ti = (a.im * ratio) - a.re;
        tr = tr / div;
        ti = ti / div;
    } else {
        const double ratio = b.im / b.re;
        const double div = (b.im * ratio) + b.re;
        tr = (a.im * ratio) + a.re;
        ti = a.im - (a.re * ratio);
        tr = tr / div;
        ti = ti / div;
    }
    return tfc_mk(tr, ti);
}
__device__ __forceinline__ double tfc_abs(tfc a) { return hypot(a.re, a.im); }
__device__ tfc tfc_sqrt(tfc a)
{
    if (a.re == 0.0 && a.im == 0.0) return tfc_mk(0.0, a.im);
    const double d = hypot(a.re, a.im);
    double r, s;
    if (a.re > 0.0) {
        r = sqrt(0.5 * (d + a.re));
        s = 0.5 * (a.im / r);
    } else {
        s = sqrt(0.5 * (d - a.re));
        r = fabs(0.5 * (a.im / s));
    }
    return tfc_mk(r, copysign(s, a.im));
}
__device__ __forceinline__ tfc tfc_log(tfc a)
{
    return tfc_mk(log(hypot(a.re, a.im)), atan2(a.im, a.re));
}

// m_complex_maxwellgarnett (:570-620), inclusion 'spheroidal' (radar_init
// sets it for snow and graupel).  *error is 1 on the volume-sum refusal.
__device__ tfc tf_maxwellgarnett(double vol1, double vol2, double vol3,
                                 tfc m1, tfc m2, tfc m3, int* error)
{
    *error = 0;
    if (fabs(vol1 + vol2 + vol3 - 1.0) > 1e-6) {
        *error = 1;
        return tfc_mk(-999.99, -999.99);
    }
    const tfc m1t = tfc_mul(m1, m1);
    const tfc m2t = tfc_mul(m2, m2);
    const tfc m3t = tfc_mul(m3, m3);
    const tfc beta2 = tfc_mul(
        tfc_div(tfc_rmul(2.0, m1t), tfc_sub(m2t, m1t)),
        tfc_subr(tfc_mul(tfc_div(m2t, tfc_sub(m2t, m1t)),
                         tfc_log(tfc_div(m2t, m1t))), 1.0));
    const tfc beta3 = tfc_mul(
        tfc_div(tfc_rmul(2.0, m1t), tfc_sub(m3t, m1t)),
        tfc_subr(tfc_mul(tfc_div(m3t, tfc_sub(m3t, m1t)),
                         tfc_log(tfc_div(m3t, m1t))), 1.0));
    const tfc num = tfc_add(tfc_add(tfc_rmul(1.0 - vol2 - vol3, m1t),
                                    tfc_mul(tfc_rmul(vol2, beta2), m2t)),
                            tfc_mul(tfc_rmul(vol3, beta3), m3t));
    const tfc den = tfc_add(tfc_addr(tfc_rmul(vol2, beta2),
                                     1.0 - vol2 - vol3),
                            tfc_rmul(vol3, beta3));
    return tfc_sqrt(tfc_div(num, den));
}

// get_m_mix_nested (:362-515) on radar_init's strings: host 'air', matrix
// 'water', hostmatrix 'icewater', mixing rule 'maxwellgarnett'.
__device__ tfc tf_m_mix_nested(tfc m_a, tfc m_i, tfc m_w, double volair,
                               double volice, double volwater, int* cumul)
{
    int error;
    *cumul = 0;
    const double vol1 = volice / dmax_(volice + volwater, 1e-10);
    const double vol2 = 1.0 - vol1;
    // get_m_mix(m_a, m_i, m_w, 0, vol1, vol2, matrix 'water')
    const tfc mtmp = tf_maxwellgarnett(vol2, 0.0, vol1, m_w, m_a, m_i,
                                       &error);
    *cumul += error;
    // get_m_mix(m_a, mtmp, 2.0*m_a, volair, 1-volair, 0, matrix 'ice')
    const tfc r = tf_maxwellgarnett(1.0 - volair, volair, 0.0, mtmp, m_a,
                                    tfc_rmul((double)2.0f, m_a), &error);
    *cumul += error;
    if (*cumul != 0) return tfc_mk(1.0, 0.0);
    return r;
}

// rayleigh_soak_wetgraupel (:265-358).
__device__ double tf_rayleigh_soak_wetgraupel(
    double x_g, double a_geo, double b_geo, double fmelt,
    double meltratio_outside, tfc m_w, tfc m_i)
{
    const double PIx = 3.1415926535897932384626434;
    const tfc m_air = tfc_mk(1.0, 0.0);
    const double fm = dmax_(dmin_(fmelt, 1.0), 0.0);
    double mra = dmax_(dmin_(meltratio_outside, 1.0), 0.0);
    mra = mra + (1.0 - mra) * fm;
    const double x_w = x_g * fm;
    const double D_g = a_geo * tf_pow(x_g, b_geo);
    if (!(D_g >= 1e-12)) return 0.0;
    double vg = PIx / (double)6.0f * (D_g * (D_g * D_g));
    const double rhog = dmax_(dmin_(x_g / vg, 900.0), 10.0);
    vg = x_g / rhog;
    const double grenz = 1.0 - rhog / (double)1000.0f;
    double volg;
    if (mra <= grenz) {
        volg = vg * (1.0 - mra * fm);
    } else {
        const double fmgrenz = ((double)900.0f - rhog)
            / (mra * (double)900.0f - rhog
               + (double)900.0f * rhog / (double)1000.0f);
        if (fm <= fmgrenz) {
            volg = ((double)1.0f - mra * fm) * vg;
        } else {
            volg = (x_g - x_w) / (double)900.0f + x_w / (double)1000.0f;
        }
    }
    const double D_large = tf_pow((double)6.0f / PIx * volg,
                                  (double)(1.0f / 3.0f));
    const double volice = (x_g - x_w) / (volg * (double)900.0f);
    const double volwater = x_w / ((double)1000.0f * volg);
    const double volair = (double)1.0f - volice - volwater;
    int error;
    const tfc m_core = tf_m_mix_nested(m_air, m_i, m_w, volair, volice,
                                       volwater, &error);
    if (error != 0) return 0.0;
    const tfc m2 = tfc_mul(m_core, m_core);
    const double a = tfc_abs(tfc_div(tfc_subr(m2, 1.0), tfc_addr(m2, 2.0)));
    const double d3 = D_large * (D_large * D_large);
    return a * a * TF_PI5 * (d3 * d3) / TF_lamda4;
}

// ---------------------------------------------------------------------------
// The lookup tables, float64 Fortran-order device arrays (thompson_runtime).
// ---------------------------------------------------------------------------
struct TfTables {
    const double* tcg_racg; const double* tmr_racg; const double* tcr_gacr;
    const double* tnr_racg; const double* tnr_gacr;
    const double* tcs_racs1; const double* tmr_racs1;
    const double* tcs_racs2; const double* tmr_racs2;
    const double* tcr_sacr1; const double* tms_sacr1;
    const double* tcr_sacr2; const double* tms_sacr2;
    const double* tnr_racs1; const double* tnr_racs2;
    const double* tnr_sacr1; const double* tnr_sacr2;
    const double* tpi_qrfz; const double* tni_qrfz;
    const double* tpg_qrfz; const double* tnr_qrfz;
    const double* tpi_qcfz; const double* tni_qcfz;
    const double* tps_iaus; const double* tni_iaus; const double* tpi_ide;
    const double* t_Efrw; const double* t_Efsw; const double* tnc_wev;
    const double* tnccn_act;
};

// (ntb_g1, ntb_g, ntb_r1, ntb_r) = (28, 28, 37, 37)
#define T_QG(t, a, b, c, d) (t)[((a) - 1) + 28 * (((b) - 1) + 28 * (((c) - 1) \
                                 + 37 * ((d) - 1)))]
// (ntb_s, ntb_t, ntb_r1, ntb_r) = (28, 9, 37, 37)
#define T_QS(t, a, b, c, d) (t)[((a) - 1) + 28 * (((b) - 1) + 9 * (((c) - 1) \
                                 + 37 * ((d) - 1)))]
// (ntb_r, ntb_r1, 45, ntb_IN) = (37, 37, 45, 55)
#define T_QRFZ(t, a, b, c, d) (t)[((a) - 1) + 37 * (((b) - 1) + 37 * (((c) - 1) \
                                   + 45 * ((d) - 1)))]
// (ntb_c, nbc, 45, ntb_IN) = (37, 100, 45, 55)
#define T_QCFZ(t, a, b, c, d) (t)[((a) - 1) + 37 * (((b) - 1) + 100 * (((c) - 1) \
                                   + 45 * ((d) - 1)))]
// (ntb_i, ntb_i1) = (64, 55)
#define T_I(t, a, b) (t)[((a) - 1) + 64 * ((b) - 1)]
// (nbr|nbs, nbc) = (100, 100)
#define T_EF(t, a, b) (t)[((a) - 1) + 100 * ((b) - 1)]
// (nbc, ntb_c, nbc) = (100, 37, 100)
#define T_WEV(t, a, b, c) (t)[((a) - 1) + 100 * (((b) - 1) + 37 * ((c) - 1))]

// ---------------------------------------------------------------------------
// mp_thompson (:1524-3763) on one column.  Arrays are 0-based (Fortran
// k = kts..kte is k = 0..nz-1); every other name is the Fortran's.
// Returns false at the no-microphysics exit (:1938), after the entry
// block's rewrite of the condensate below R1 (:1803-1881).
// ---------------------------------------------------------------------------
__device__ bool tf_mp_thompson(
    TfCol<float> qv1d, TfCol<float> qc1d, TfCol<float> qi1d,
    TfCol<float> qr1d, TfCol<float> qs1d, TfCol<float> qg1d,
    TfCol<float> ni1d, TfCol<float> nr1d, TfCol<float> nc1d,
    TfCol<float> nwfa1d, TfCol<float> nifa1d, TfCol<float> t1d,
    TfCol<float> p1d, TfCol<float> w1d, TfCol<float> dzq, float* pptrain,
    float* pptsnow, float* pptgraul, float* pptice, const int nz,
    const float dt, const int singular_fall, const TfTables& tb,
    TfArena ws)
{
    const int kts = 0, kte = nz - 1;
    const float rand1 = 0.0f, rand3 = 0.0f;

    TfCol<float> tten = ws.F(), qvten = ws.F(), qcten = ws.F(), qiten = ws.F();
    TfCol<float> qrten = ws.F(), qsten = ws.F(), qgten = ws.F(), niten = ws.F();
    TfCol<float> nrten = ws.F(), ncten = ws.F(), nwfaten = ws.F();
    TfCol<float> nifaten = ws.F();
    TfCol<double> prw_vcd = ws.D(), prr_gml_k = ws.D(), prr_sml_k = ws.D();
    TfCol<float> temp = ws.F(), pres = ws.F(), qv = ws.F();
    TfCol<float> rc = ws.F(), ri = ws.F(), rr = ws.F(), rs = ws.F(), rg = ws.F();
    TfCol<float> ni = ws.F(), nr = ws.F(), nc = ws.F(), nwfa = ws.F();
    TfCol<float> nifa = ws.F();
    TfCol<float> rho = ws.F(), rhof = ws.F(), rhof2 = ws.F();
    TfCol<float> qvs = ws.F(), qvsi = ws.F(), delQvs = ws.F();
    TfCol<float> ssatw = ws.F(), ssati = ws.F();
    TfCol<float> diffu = ws.F(), visco = ws.F(), vsc2 = ws.F(), tcond = ws.F();
    TfCol<float> lvap = ws.F(), ocp = ws.F(), lvt2 = ws.F();
    TfCol<double> ilamr = ws.D(), ilamg = ws.D(), N0_r = ws.D(), N0_g = ws.D();
    TfCol<float> mvd_r = ws.F(), mvd_c = ws.F();
    TfCol<float> smob = ws.F(), smo2 = ws.F(), smo1 = ws.F(), smo0 = ws.F();
    TfCol<float> smoc = ws.F(), smoe = ws.F(), smof = ws.F();
    TfCol<float> vts_boost = ws.F();
    TfCol<bool> L_qc = ws.B(), L_qi = ws.B(), L_qr = ws.B(), L_qs = ws.B();
    TfCol<bool> L_qg = ws.B();

    bool no_micro = true;
    const float dtsave = dt;
    const float odt = 1.0f / dt;
    const float odts = 1.0f / dtsave;
    int nu_c = 2;
    double lamc = 0.0, lami = 0.0, ilami = 0.0, lamr = 0.0, lamg = 0.0;
    double N0_exp, N0_min, lam_exp;

    for (int k = kts; k <= kte; ++k) {
        tten[k] = 0.0f; qvten[k] = 0.0f; qcten[k] = 0.0f; qiten[k] = 0.0f;
        qrten[k] = 0.0f; qsten[k] = 0.0f; qgten[k] = 0.0f; niten[k] = 0.0f;
        nrten[k] = 0.0f; ncten[k] = 0.0f; nwfaten[k] = 0.0f;
        nifaten[k] = 0.0f;
        prw_vcd[k] = 0.0; prr_gml_k[k] = 0.0; prr_sml_k[k] = 0.0;
        smo0[k] = 0.0f; smo1[k] = 0.0f; smo2[k] = 0.0f; smob[k] = 0.0f;
        smoc[k] = 0.0f; smoe[k] = 0.0f; smof[k] = 0.0f;
    }

    // :1773-1883, the column into local arrays.
    for (int k = kts; k <= kte; ++k) {
        temp[k] = t1d[k];
        qv[k] = fmax_(1.E-10f, qv1d[k]);
        pres[k] = p1d[k];
        rho[k] = 0.622f * pres[k] / (TF_R * temp[k] * (qv[k] + 0.622f));
        nwfa[k] = fmax_(11.1E6f, fmin_(9999.E6f, nwfa1d[k] * rho[k]));
        nifa[k] = fmax_(TF_naIN1 * 0.01f, fmin_(9999.E6f, nifa1d[k] * rho[k]));

        if (qc1d[k] > TF_R1) {
            no_micro = false;
            rc[k] = qc1d[k] * rho[k];
            nc[k] = fmax_(2.0f, nc1d[k] * rho[k]);
            L_qc[k] = true;
            nu_c = min(15, tf_nint(1000.E6f / nc[k]) + 2);
            lamc = (double)tf_powf(nc[k] * TF_am_r * CCG(2, nu_c)
                                   * OCG1(nu_c) / rc[k], TF_obmr);
            const float xDc = (float)((double)(TF_bm_r + (float)nu_c + 1.0f)
                                      / lamc);
            if (xDc < TF_D0c) {
                lamc = (double)FDIV(CCE(2, nu_c), TF_D0c);
            } else if (xDc > TF_D0r * 2.0f) {
                lamc = (double)FDIV(CCE(2, nu_c), TF_D0r * 2.0f);
            }
            nc[k] = (float)dmin_((double)TF_Nt_c_max,
                                 (double)FDIV(CCG(1, nu_c) * OCG2(nu_c) * rc[k],
                                              TF_am_r)
                                 * tf_pow(lamc, (double)TF_bm_r));
        } else {
            qc1d[k] = 0.0f;
            nc1d[k] = 0.0f;
            rc[k] = TF_R1;
            nc[k] = 2.0f;
            L_qc[k] = false;
        }

        if (qi1d[k] > TF_R1) {
            no_micro = false;
            ri[k] = qi1d[k] * rho[k];
            ni[k] = fmax_(TF_R2, ni1d[k] * rho[k]);
            if (ni[k] <= TF_R2) {
                lami = (double)FDIV(CIE(2), 25.E-6f);
                ni[k] = (float)dmin_(499.E3,
                    (double)FDIV(CIG(1) * TF_oig2 * ri[k], TF_am_i)
                    * tf_pow(lami, (double)TF_bm_i));
            }
            L_qi[k] = true;
            lami = (double)tf_powf(TF_am_i * CIG(2) * TF_oig1 * ni[k] / ri[k],
                                   TF_obmi);
            ilami = 1.0 / lami;
            const float xDi = (float)((double)(TF_bm_i + TF_mu_i + 1.0f)
                                      * ilami);
            if (xDi < 5.E-6f) {
                lami = (double)FDIV(CIE(2), 5.E-6f);
                ni[k] = (float)dmin_(499.E3,
                    (double)FDIV(CIG(1) * TF_oig2 * ri[k], TF_am_i)
                    * tf_pow(lami, (double)TF_bm_i));
            } else if (xDi > 300.E-6f) {
                lami = (double)FDIV(CIE(2), 300.E-6f);
                ni[k] = (float)((double)FDIV(CIG(1) * TF_oig2 * ri[k], TF_am_i)
                                * tf_pow(lami, (double)TF_bm_i));
            }
        } else {
            qi1d[k] = 0.0f;
            ni1d[k] = 0.0f;
            ri[k] = TF_R1;
            ni[k] = TF_R2;
            L_qi[k] = false;
        }

        if (qr1d[k] > TF_R1) {
            no_micro = false;
            rr[k] = qr1d[k] * rho[k];
            nr[k] = fmax_(TF_R2, nr1d[k] * rho[k]);
            if (nr[k] <= TF_R2) {
                mvd_r[k] = 1.0E-3f;
                lamr = (double)((3.0f + TF_mu_r + 0.672f) / mvd_r[k]);
                nr[k] = (float)((double)(CRG(2) * TF_org3 * rr[k])
                                * tf_pow(lamr, (double)TF_bm_r)
                                / (double)TF_am_r);
            }
            L_qr[k] = true;
            lamr = (double)tf_powf(TF_am_r * CRG(3) * TF_org2 * nr[k] / rr[k],
                                   TF_obmr);
            mvd_r[k] = (float)((double)(3.0f + TF_mu_r + 0.672f) / lamr);
            if (mvd_r[k] > 2.5E-3f) {
                mvd_r[k] = 2.5E-3f;
                lamr = (double)((3.0f + TF_mu_r + 0.672f) / mvd_r[k]);
                nr[k] = (float)((double)(CRG(2) * TF_org3 * rr[k])
                                * tf_pow(lamr, (double)TF_bm_r)
                                / (double)TF_am_r);
            } else if (mvd_r[k] < TF_D0r * 0.75f) {
                mvd_r[k] = TF_D0r * 0.75f;
                lamr = (double)((3.0f + TF_mu_r + 0.672f) / mvd_r[k]);
                nr[k] = (float)((double)(CRG(2) * TF_org3 * rr[k])
                                * tf_pow(lamr, (double)TF_bm_r)
                                / (double)TF_am_r);
            }
        } else {
            qr1d[k] = 0.0f;
            nr1d[k] = 0.0f;
            rr[k] = TF_R1;
            nr[k] = TF_R2;
            L_qr[k] = false;
        }
        if (qs1d[k] > TF_R1) {
            no_micro = false;
            rs[k] = qs1d[k] * rho[k];
            L_qs[k] = true;
        } else {
            qs1d[k] = 0.0f;
            rs[k] = TF_R1;
            L_qs[k] = false;
        }
        if (qg1d[k] > TF_R1) {
            no_micro = false;
            rg[k] = qg1d[k] * rho[k];
            L_qg[k] = true;
        } else {
            qg1d[k] = 0.0f;
            rg[k] = TF_R1;
            L_qg[k] = false;
        }
    }

    // :1903-1931.
    for (int k = kts; k <= kte; ++k) {
        const float tempc = temp[k] - 273.15f;
        rhof[k] = sqrtf(TF_rho_not / rho[k]);
        rhof2[k] = sqrtf(rhof[k]);
        qvs[k] = tf_rslf(pres[k], temp[k]);
        delQvs[k] = fmax_(0.0f, tf_rslf(pres[k], 273.15f) - qv[k]);
        if (tempc <= 0.0f) {
            qvsi[k] = tf_rsif(pres[k], temp[k]);
        } else {
            qvsi[k] = qvs[k];
        }
        const float satw = qv[k] / qvs[k];
        const float sati = qv[k] / qvsi[k];
        ssatw[k] = satw - 1.0f;
        ssati[k] = sati - 1.0f;
        if (fabsf(ssatw[k]) < TF_eps) ssatw[k] = 0.0f;
        if (fabsf(ssati[k]) < TF_eps) ssati[k] = 0.0f;
        if (no_micro && ssati[k] > 0.0f) no_micro = false;
        diffu[k] = 2.11E-5f * tf_powf(FDIV(temp[k], 273.15f), 1.94f)
                 * (101325.0f / pres[k]);
        if (tempc >= 0.0f) {
            visco[k] = (1.718f + 0.0049f * tempc) * 1.0E-5f;
        } else {
            visco[k] = (1.718f + 0.0049f * tempc - 1.2E-5f * tempc * tempc)
                     * 1.0E-5f;
        }
        ocp[k] = 1.0f / (TF_Cp * (1.0f + 0.887f * qv[k]));
        vsc2[k] = sqrtf(rho[k] / visco[k]);
        lvap[k] = TF_lvap0 + (2106.0f - 4218.0f) * tempc;
        tcond[k] = (5.69f + 0.0168f * tempc) * 1.0E-5f * 418.936f;
    }

    if (no_micro) return false;

    // :1943-2026, snow moments.
    for (int k = kts; k <= kte; ++k) {
        if (!L_qs[k]) continue;
        const float tc0 = fmin_(-0.1f, temp[k] - 273.15f);
        smob[k] = rs[k] * TF_oams;
        smo2[k] = smob[k];
        float loga_ = tf_loga0(tc0);
        float a_ = tf_powf(10.0f, loga_);
        float b_ = tf_b0(tc0);
        smo0[k] = a_ * tf_powf(smo2[k], b_);
        loga_ = tf_loga1(tc0);
        a_ = tf_powf(10.0f, loga_);
        b_ = tf_b1(tc0);
        smo1[k] = a_ * tf_powf(smo2[k], b_);
        loga_ = tf_loga(tc0, CSE(1));
        a_ = tf_powf(10.0f, loga_);
        b_ = tf_b(tc0, CSE(1));
        smoc[k] = a_ * tf_powf(smo2[k], b_);
        loga_ = tf_loga(tc0, CSE(13));
        a_ = tf_powf(10.0f, loga_);
        b_ = tf_b(tc0, CSE(13));
        smoe[k] = a_ * tf_powf(smo2[k], b_);
        loga_ = tf_loga(tc0, CSE(16));
        a_ = tf_powf(10.0f, loga_);
        b_ = tf_b(tc0, CSE(16));
        smof[k] = a_ * tf_powf(smo2[k], b_);
    }

    // :2031-2055, graupel intercept and slope.
    N0_min = (double)TF_gonv_max;
    int k_0 = kts;
    for (int k = kte; k >= kts; --k) {
        if (temp[k] >= 270.65f) k_0 = max(k_0, k);
    }
    for (int k = kte; k >= kts; --k) {
        float xslw1;
        if (k > k_0 && L_qr[k] && mvd_r[k] > 100.E-6f) {
            xslw1 = 4.01f + tf_log10f(mvd_r[k]);
        } else {
            xslw1 = 0.01f;
        }
        const float ygra1 = 4.31f + tf_log10f(fmax_(5.E-5f, rg[k]));
        const float zans1 = (3.1f + (100.0f / (300.0f * xslw1 * ygra1
            / (10.0f / xslw1 + 1.0f + 0.25f * ygra1) + 30.0f
            + 10.0f * ygra1))) + rand1;
        N0_exp = (double)tf_powf(10.0f, zans1);
        N0_exp = dmax_((double)TF_gonv_min, dmin_(N0_exp,
                                                  (double)TF_gonv_max));
        N0_min = dmin_(N0_exp, N0_min);
        N0_exp = N0_min;
        lam_exp = tf_pow(N0_exp * (double)TF_am_g * (double)CGG(1)
                         / (double)rg[k], (double)TF_oge1);
        lamg = lam_exp * (double)tf_powf(CGG(3) * TF_ogg2 * TF_ogg1,
                                         TF_obmg);
        ilamg[k] = 1.0 / lamg;
        N0_g[k] = N0_exp / ((double)CGG(2) * lam_exp)
                * tf_pow(lamg, (double)CGE(2));
    }

    // :2062-2067, rain intercept and slope.
    for (int k = kte; k >= kts; --k) {
        lamr = (double)tf_powf(TF_am_r * CRG(3) * TF_org2 * nr[k] / rr[k],
                               TF_obmr);
        ilamr[k] = 1.0 / lamr;
        mvd_r[k] = (float)((double)(3.0f + TF_mu_r + 0.672f) / lamr);
        N0_r[k] = (double)(nr[k] * TF_org2) * tf_pow(lamr, (double)CRE(2));
    }

    // :2073-2962.  The warm rates (:2073-2144), the frozen rates
    // (:2150-2685), the conservation limiters (:2693-2789) and the
    // tendencies (:2795-2962) are four loops over k in the Fortran, but no
    // statement in them reads another level, so they run fused per level
    // in the same per-level order.
    for (int k = kts; k <= kte; ++k) {
        double pnc_wau = 0.0, pnc_rcw = 0.0, pnc_scw = 0.0, pnc_gcw = 0.0;
        double pna_rca = 0.0, pna_sca = 0.0, pna_gca = 0.0;
        double pnd_rcd = 0.0, pnd_scd = 0.0, pnd_gcd = 0.0;
        double prr_wau = 0.0, prr_rcw = 0.0, prr_rcs = 0.0, prr_rcg = 0.0;
        double prr_sml = 0.0, prr_gml = 0.0, prr_rci = 0.0;
        double pnr_wau = 0.0, pnr_rcs = 0.0, pnr_rcg = 0.0, pnr_rci = 0.0;
        double pnr_sml = 0.0, pnr_gml = 0.0, pnr_rcr = 0.0, pnr_rfz = 0.0;
        double pri_inu = 0.0, pni_inu = 0.0, pri_ihm = 0.0, pni_ihm = 0.0;
        double pri_wfz = 0.0, pni_wfz = 0.0, pri_rfz = 0.0, pni_rfz = 0.0;
        double pri_ide = 0.0, pni_ide = 0.0, pri_rci = 0.0, pni_rci = 0.0;
        double pni_sci = 0.0, pni_iau = 0.0, pri_iha = 0.0, pni_iha = 0.0;
        double prs_iau = 0.0, prs_sci = 0.0, prs_rcs = 0.0, prs_scw = 0.0;
        double prs_sde = 0.0, prs_ihm = 0.0, prs_ide = 0.0;
        double prg_scw = 0.0, prg_rfz = 0.0, prg_gde = 0.0, prg_gcw = 0.0;
        double prg_rci = 0.0, prg_rcs = 0.0, prg_rcg = 0.0, prg_ihm = 0.0;
        float xDc = 0.0f;

        // ---- warm rain (:2077-2142) ----
        if (L_qr[k] && mvd_r[k] > TF_D0r) {
            const float Ef_rr = 1.0f - tf_expf(2300.0f * (mvd_r[k]
                                                          - 1950.0E-6f));
            pnr_rcr = (double)(Ef_rr * 2.0f * nr[k] * rr[k]);
        }

        mvd_c[k] = TF_D0c;
        if (L_qc[k]) {
            nu_c = min(15, tf_nint(1000.E6f / nc[k]) + 2);
            xDc = fmax_(TF_D0c * 1.E6f,
                        tf_powf(rc[k] / (TF_am_r * nc[k]), TF_obmr) * 1.E6f);
            lamc = (double)tf_powf(nc[k] * TF_am_r * CCG(2, nu_c)
                                   * OCG1(nu_c) / rc[k], TF_obmr);
            mvd_c[k] = (float)((double)(3.0f + (float)nu_c + 0.672f) / lamc);
        }

        if (rc[k] > 0.01e-3f) {
            const float Dc_g = (float)((double)tf_powf(CCG(3, nu_c)
                                       * OCG2(nu_c), TF_obmr) / lamc
                                       * (double)1.E6f);
            const float Dc_b = tf_powf(xDc * xDc * xDc * Dc_g * Dc_g * Dc_g
                                       - xDc * xDc * xDc * xDc * xDc * xDc,
                                       1.0f / 6.0f);
            const float zeta1 = 0.5f * ((6.25E-6f * xDc * Dc_b * Dc_b * Dc_b
                                         - 0.4f)
                                        + fabsf(6.25E-6f * xDc * Dc_b * Dc_b
                                                * Dc_b - 0.4f));
            const float zeta = 0.027f * rc[k] * zeta1;
            const float taud = 0.5f * ((0.5f * Dc_b - 7.5f)
                                       + fabsf(0.5f * Dc_b - 7.5f)) + TF_R1;
            const float tau = 3.72f / (rc[k] * taud);
            prr_wau = (double)(zeta / tau);
            prr_wau = dmin_((double)(rc[k] * odts), prr_wau);
            pnr_wau = prr_wau / (double)(TF_am_r * (float)nu_c * TF_D0r
                                         * TF_D0r * TF_D0r);
            pnc_wau = dmin_((double)(nc[k] * odts),
                            prr_wau / (double)(TF_am_r * mvd_c[k] * mvd_c[k]
                                               * mvd_c[k]));
        }

        if (L_qr[k] && mvd_r[k] > TF_D0r && mvd_c[k] > TF_D0c) {
            lamr = 1.0 / ilamr[k];
            int idx = 1 + (int)((double)TF_nbr
                * tf_log((double)mvd_r[k] / TF_Dr[0])
                / tf_log(TF_Dr[TF_nbr - 1] / TF_Dr[0]));
            idx = min(idx, TF_nbr);
            const float Ef_rw = (float)T_EF(tb.t_Efrw, idx,
                                           (int)(mvd_c[k] * 1.E6f));
            const double fall = tf_pow(lamr + (double)TF_fv_r,
                                       (double)(-CRE(9)));
            prr_rcw = (double)(rhof[k] * TF_t1_qr_qc * Ef_rw * rc[k])
                    * N0_r[k] * fall;
            prr_rcw = dmin_((double)(rc[k] * odts), prr_rcw);
            pnc_rcw = (double)(rhof[k] * TF_t1_qr_qc * Ef_rw * nc[k])
                    * N0_r[k] * fall;
            pnc_rcw = dmin_((double)(nc[k] * odts), pnc_rcw);
        }

        if (L_qr[k] && mvd_r[k] > TF_D0r) {
            float Ef_ra = tf_eff_aero(mvd_r[k], 0.04E-6f, visco[k], rho[k],
                                      temp[k], 0);
            lamr = 1.0 / ilamr[k];
            const double fall = tf_pow(lamr + (double)TF_fv_r,
                                       (double)(-CRE(9)));
            pna_rca = (double)(rhof[k] * TF_t1_qr_qc * Ef_ra * nwfa[k])
                    * N0_r[k] * fall;
            pna_rca = dmin_((double)(nwfa[k] * odts), pna_rca);
            Ef_ra = tf_eff_aero(mvd_r[k], 0.8E-6f, visco[k], rho[k],
                                temp[k], 0);
            pnd_rcd = (double)(rhof[k] * TF_t1_qr_qc * Ef_ra * nifa[k])
                    * N0_r[k] * fall;
            pnd_rcd = dmin_((double)(nifa[k] * odts), pnd_rcd);
        }

        // ---- frozen species (:2150-2685) ----
        vts_boost[k] = singular_fall ? 1.5f : 1.0f;
        const float tempc = temp[k] - 273.15f;
        const int idx_tc = max(1, min(tf_nint(-tempc), 45));
        int idx_t = (int)FDIV(tempc - 2.5f, 5.0f) - 1;
        idx_t = max(1, -idx_t);
        idx_t = min(idx_t, TF_ntb_t);

        const int idx_c = rc[k] > TF_r_c[0]
            ? tf_decade_bin(rc[k], TF_nic2, TF_ntb_c) : 1;
        int idx_n = tf_nintd((double)1.0f + (double)(float)TF_nbc
                             * tf_log((double)nc[k] / TF_t_Nc[0])
                             / (double)TF_nic1);
        idx_n = max(1, min(idx_n, TF_nbc));
        const int idx_i = ri[k] > TF_r_i[0]
            ? tf_decade_bin(ri[k], TF_nii2, TF_ntb_i) : 1;
        const int idx_i1 = ni[k] > TF_Nt_i[0]
            ? tf_decade_bin(ni[k], TF_nii3, TF_ntb_i1) : 1;
        int idx_r, idx_r1;
        if (rr[k] > TF_r_r[0]) {
            idx_r = tf_decade_bin(rr[k], TF_nir2, TF_ntb_r);
            lamr = 1.0 / ilamr[k];
            lam_exp = lamr * (double)tf_powf(CRG(3) * TF_org2 * TF_org1,
                                             TF_bm_r);
            N0_exp = (double)FDIV(TF_org1 * rr[k], TF_am_r)
                   * tf_pow(lam_exp, (double)CRE(1));
            idx_r1 = tf_decade_bin_d(N0_exp, TF_nir3, TF_ntb_r1);
        } else {
            idx_r = 1;
            idx_r1 = TF_ntb_r1;
        }
        const int idx_s = rs[k] > TF_r_s[0]
            ? tf_decade_bin(rs[k], TF_nis2, TF_ntb_s) : 1;
        int idx_g, idx_g1;
        if (rg[k] > TF_r_g[0]) {
            idx_g = tf_decade_bin(rg[k], TF_nig2, TF_ntb_g);
            lamg = 1.0 / ilamg[k];
            lam_exp = lamg * (double)tf_powf(CGG(3) * TF_ogg2 * TF_ogg1,
                                             TF_bm_g);
            N0_exp = (double)FDIV(TF_ogg1 * rg[k], TF_am_g)
                   * tf_pow(lam_exp, (double)CGE(1));
            idx_g1 = tf_decade_bin_d(N0_exp, TF_nig3, TF_ntb_g1);
        } else {
            idx_g = 1;
            idx_g1 = TF_ntb_g1;
        }

        // :2283-2299, deposition/sublimation prefactor.
        float otemp = 1.0f / temp[k];
        const float rvs = rho[k] * qvsi[k];
        const float rvs_p = rvs * otemp * (TF_lsub * otemp * TF_oRv - 1.0f);
        const float rvs_pp = rvs * (otemp * (TF_lsub * otemp * TF_oRv - 1.0f)
                                    * otemp * (TF_lsub * otemp * TF_oRv
                                               - 1.0f)
                                    + (-2.0f * TF_lsub * otemp * otemp * otemp
                                       * TF_oRv)
                                    + otemp * otemp);
        const float gamsc = TF_lsub * diffu[k] / tcond[k] * rvs_p;
        float alphsc = 0.5f * (gamsc / (1.0f + gamsc))
                     * (gamsc / (1.0f + gamsc)) * rvs_pp / rvs_p * rvs / rvs_p;
        alphsc = fmax_(1.E-9f, alphsc);
        float xsat = ssati[k];
        if (fabsf(xsat) < 1.E-9f) xsat = 0.0f;
        const float t1_subl = 4.0f * TF_PI * (1.0f - alphsc * xsat
            + 2.0f * alphsc * alphsc * xsat * xsat
            - 5.0f * alphsc * alphsc * alphsc * xsat * xsat * xsat)
            / (1.0f + gamsc);

        // :2302-2334, snow and graupel collecting cloud water.
        if (L_qc[k] && mvd_c[k] > TF_D0c) {
            float xDs = 0.0f;
            if (L_qs[k]) xDs = smoc[k] / smob[k];
            if (xDs > TF_D0s) {
                int idx = 1 + (int)((double)TF_nbs
                    * tf_log((double)xDs / TF_Ds[0])
                    / tf_log(TF_Ds[TF_nbs - 1] / TF_Ds[0]));
                idx = min(idx, TF_nbs);
                const float Ef_sw = (float)T_EF(tb.t_Efsw, idx,
                                               (int)(mvd_c[k] * 1.E6f));
                prs_scw = (double)(rhof[k] * TF_t1_qs_qc * Ef_sw * rc[k]
                                   * smoe[k]);
                pnc_scw = (double)(rhof[k] * TF_t1_qs_qc * Ef_sw * nc[k]
                                   * smoe[k]);
                pnc_scw = dmin_((double)(nc[k] * odts), pnc_scw);
            }
            if (rg[k] >= TF_r_g[0] && mvd_c[k] > TF_D0c) {
                const float xDg = (float)((double)(TF_bm_g + TF_mu_g + 1.0f)
                                          * ilamg[k]);
                const float vtg = (float)((double)(rhof[k] * TF_av_g * CGG(6)
                                                   * TF_ogg3)
                                          * tf_pow(ilamg[k], (double)TF_bv_g));
                const float stoke_g = mvd_c[k] * mvd_c[k] * vtg * TF_rho_w
                                    / (9.0f * visco[k] * xDg);
                if (xDg > TF_D0g) {
                    float Ef_gw = 0.0f;
                    if (stoke_g >= 0.4f && stoke_g <= 10.0f) {
                        Ef_gw = 0.55f * tf_log10f(2.51f * stoke_g);
                    } else if (stoke_g < 0.4f) {
                        Ef_gw = 0.0f;
                    } else if (stoke_g > 10.0f) {
                        Ef_gw = 0.77f;
                    }
                    const double gfall = tf_pow(ilamg[k], (double)CGE(9));
                    prg_gcw = (double)(rhof[k] * TF_t1_qg_qc * Ef_gw * rc[k])
                            * N0_g[k] * gfall;
                    pnc_gcw = (double)(rhof[k] * TF_t1_qg_qc * Ef_gw * nc[k])
                            * N0_g[k] * gfall;
                    pnc_gcw = dmin_((double)(nc[k] * odts), pnc_gcw);
                }
            }
        }

        // :2337-2358, snow and graupel collecting aerosols.
        if (rs[k] > TF_r_s[0]) {
            const float xDs = smoc[k] / smob[k];
            float Ef_sa = tf_eff_aero(xDs, 0.04E-6f, visco[k], rho[k],
                                      temp[k], 1);
            pna_sca = (double)(rhof[k] * TF_t1_qs_qc * Ef_sa * nwfa[k]
                               * smoe[k]);
            pna_sca = dmin_((double)(nwfa[k] * odts), pna_sca);
            Ef_sa = tf_eff_aero(xDs, 0.8E-6f, visco[k], rho[k], temp[k], 1);
            pnd_scd = (double)(rhof[k] * TF_t1_qs_qc * Ef_sa * nifa[k]
                               * smoe[k]);
            pnd_scd = dmin_((double)(nifa[k] * odts), pnd_scd);
        }
        if (rg[k] > TF_r_g[0]) {
            const float xDg = (float)((double)(TF_bm_g + TF_mu_g + 1.0f)
                                      * ilamg[k]);
            const double gfall = tf_pow(ilamg[k], (double)CGE(9));
            float Ef_ga = tf_eff_aero(xDg, 0.04E-6f, visco[k], rho[k],
                                      temp[k], 2);
            pna_gca = (double)(rhof[k] * TF_t1_qg_qc * Ef_ga * nwfa[k])
                    * N0_g[k] * gfall;
            pna_gca = dmin_((double)(nwfa[k] * odts), pna_gca);
            Ef_ga = tf_eff_aero(xDg, 0.8E-6f, visco[k], rho[k], temp[k], 2);
            pnd_gcd = (double)(rhof[k] * TF_t1_qg_qc * Ef_ga * nifa[k])
                    * N0_g[k] * gfall;
            pnd_gcd = dmin_((double)(nifa[k] * odts), pnd_gcd);
        }

        // :2363-2418, rain collecting snow and graupel.
        if (rr[k] >= TF_r_r[0]) {
            if (rs[k] >= TF_r_s[0]) {
                if (temp[k] < TF_T_0) {
                    prr_rcs = -(T_QS(tb.tmr_racs2, idx_s, idx_t, idx_r1, idx_r)
                              + T_QS(tb.tcr_sacr2, idx_s, idx_t, idx_r1, idx_r)
                              + T_QS(tb.tmr_racs1, idx_s, idx_t, idx_r1, idx_r)
                              + T_QS(tb.tcr_sacr1, idx_s, idx_t, idx_r1,
                                     idx_r));
                    prs_rcs = T_QS(tb.tmr_racs2, idx_s, idx_t, idx_r1, idx_r)
                            + T_QS(tb.tcr_sacr2, idx_s, idx_t, idx_r1, idx_r)
                            - T_QS(tb.tcs_racs1, idx_s, idx_t, idx_r1, idx_r)
                            - T_QS(tb.tms_sacr1, idx_s, idx_t, idx_r1, idx_r);
                    prg_rcs = T_QS(tb.tmr_racs1, idx_s, idx_t, idx_r1, idx_r)
                            + T_QS(tb.tcr_sacr1, idx_s, idx_t, idx_r1, idx_r)
                            + T_QS(tb.tcs_racs1, idx_s, idx_t, idx_r1, idx_r)
                            + T_QS(tb.tms_sacr1, idx_s, idx_t, idx_r1, idx_r);
                    prr_rcs = dmax_((double)(-rr[k] * odts), prr_rcs);
                    prs_rcs = dmax_((double)(-rs[k] * odts), prs_rcs);
                    prg_rcs = dmin_((double)((rr[k] + rs[k]) * odts),
                                    prg_rcs);
                    pnr_rcs = T_QS(tb.tnr_racs1, idx_s, idx_t, idx_r1, idx_r)
                            + T_QS(tb.tnr_racs2, idx_s, idx_t, idx_r1, idx_r)
                            + T_QS(tb.tnr_sacr1, idx_s, idx_t, idx_r1, idx_r)
                            + T_QS(tb.tnr_sacr2, idx_s, idx_t, idx_r1, idx_r);
                } else {
                    prs_rcs = -T_QS(tb.tcs_racs1, idx_s, idx_t, idx_r1, idx_r)
                            - T_QS(tb.tms_sacr1, idx_s, idx_t, idx_r1, idx_r)
                            + T_QS(tb.tmr_racs2, idx_s, idx_t, idx_r1, idx_r)
                            + T_QS(tb.tcr_sacr2, idx_s, idx_t, idx_r1, idx_r);
                    prs_rcs = dmax_((double)(-rs[k] * odts), prs_rcs);
                    prr_rcs = -prs_rcs;
                    pnr_rcs = T_QS(tb.tnr_racs2, idx_s, idx_t, idx_r1, idx_r)
                            + T_QS(tb.tnr_sacr2, idx_s, idx_t, idx_r1, idx_r);
                }
                pnr_rcs = dmin_((double)(nr[k] * odts), pnr_rcs);
            }
            if (rg[k] >= TF_r_g[0]) {
                if (temp[k] < TF_T_0) {
                    prg_rcg = T_QG(tb.tmr_racg, idx_g1, idx_g, idx_r1, idx_r)
                            + T_QG(tb.tcr_gacr, idx_g1, idx_g, idx_r1, idx_r);
                    prg_rcg = dmin_((double)(rr[k] * odts), prg_rcg);
                    prr_rcg = -prg_rcg;
                    pnr_rcg = T_QG(tb.tnr_racg, idx_g1, idx_g, idx_r1, idx_r)
                            + T_QG(tb.tnr_gacr, idx_g1, idx_g, idx_r1, idx_r);
                    pnr_rcg = dmin_((double)(nr[k] * odts), pnr_rcg);
                } else {
                    prr_rcg = T_QG(tb.tcg_racg, idx_g1, idx_g, idx_r1, idx_r);
                    prr_rcg = dmin_((double)(rg[k] * odts), prr_rcg);
                    prg_rcg = -prr_rcg;
                    pnr_rcg = (double)(-5.0f)
                            * T_QG(tb.tnr_gacr, idx_g1, idx_g, idx_r1, idx_r);
                }
            }
        }

        if (temp[k] < TF_T_0) {
            // :2424-2632, below 0 C.
            vts_boost[k] = 1.0f;
            const float rate_max = (qv[k] - qvsi[k]) * rho[k] * odts * 0.999f;
            float xni = tf_ice_demott(tempc, rho[k], nifa[k]);
            int idx_IN = 1;
            if (xni > TF_Nt_IN[0]) {
                idx_IN = tf_decade_bin(xni, TF_niIN2, TF_ntb_IN);
            }
            if (rr[k] > TF_r_r[0]) {
                prg_rfz = T_QRFZ(tb.tpg_qrfz, idx_r, idx_r1, idx_tc, idx_IN)
                        * (double)odts;
                pri_rfz = T_QRFZ(tb.tpi_qrfz, idx_r, idx_r1, idx_tc, idx_IN)
                        * (double)odts;
                pni_rfz = T_QRFZ(tb.tni_qrfz, idx_r, idx_r1, idx_tc, idx_IN)
                        * (double)odts;
                pnr_rfz = T_QRFZ(tb.tnr_qrfz, idx_r, idx_r1, idx_tc, idx_IN)
                        * (double)odts;
                pnr_rfz = dmin_((double)(nr[k] * odts), pnr_rfz);
            } else if (rr[k] > TF_R1 && temp[k] < TF_HGFR) {
                pri_rfz = (double)(rr[k] * odts);
                pnr_rfz = (double)(nr[k] * odts);
                pni_rfz = pnr_rfz;
            }
            if (rc[k] > TF_r_c[0]) {
                pri_wfz = T_QCFZ(tb.tpi_qcfz, idx_c, idx_n, idx_tc, idx_IN)
                        * (double)odts;
                pri_wfz = dmin_((double)(rc[k] * odts), pri_wfz);
                pni_wfz = T_QCFZ(tb.tni_qcfz, idx_c, idx_n, idx_tc, idx_IN)
                        * (double)odts;
                pni_wfz = dmin_(dmin_((double)(nc[k] * odts),
                                      pri_wfz / (double)(2.0f * TF_xm0i)),
                                pni_wfz);
            } else if (rc[k] > TF_R1 && temp[k] < TF_HGFR) {
                pri_wfz = (double)(rc[k] * odts);
                pni_wfz = (double)(nc[k] * odts);
            }

            if ((ssati[k] >= 0.25f) || (ssatw[k] > TF_eps
                                        && temp[k] < 253.15f)) {
                float xnc = tf_ice_demott(tempc, rho[k], nifa[k]);
                xnc = xnc * (1.0f + 3.0f * rand3);
                xni = (float)((double)ni[k] + (pni_rfz + pni_wfz)
                              * (double)dtsave);
                pni_inu = (double)(0.5f * (xnc - xni + fabsf(xnc - xni))
                                   * odts);
                pri_inu = dmin_((double)rate_max, (double)TF_xm0i * pni_inu);
                pni_inu = pri_inu / (double)TF_xm0i;
            }

            xni = (float)((double)(smo0[k] + ni[k])
                          + (pni_rfz + pni_wfz + pni_inu) * (double)dtsave);
            if ((xni <= 500.E3f) && (temp[k] < 238.0f)
                    && (ssati[k] >= 0.4f)) {
                const float xnc = tf_ice_koop(temp[k], qv[k], qvs[k],
                                              nwfa[k], dtsave);
                pni_iha = (double)(xnc * odts);
                pri_iha = dmin_((double)rate_max,
                                (double)(TF_xm0i * 0.1f) * pni_iha);
                pni_iha = pri_iha / (double)(TF_xm0i * 0.1f);
            }

            // :2517-2550, cloud ice deposition and autoconversion.
            if (L_qi[k]) {
                lami = (double)tf_powf(TF_am_i * CIG(2) * TF_oig1 * ni[k]
                                       / ri[k], TF_obmi);
                ilami = 1.0 / lami;
                const float xDi = (float)dmax_((double)TF_D0i,
                    (double)(TF_bm_i + TF_mu_i + 1.0f) * ilami);
                const float xmi = TF_am_i * tf_powf(xDi, TF_bm_i);
                const float oxmi = 1.0f / xmi;
                pri_ide = (double)(TF_C_cube * t1_subl * diffu[k] * ssati[k]
                                   * rvs * TF_oig1 * CIG(5) * ni[k]) * ilami;
                if (pri_ide < 0.0) {
                    pri_ide = dmax_(dmax_((double)(-ri[k] * odts), pri_ide),
                                    (double)rate_max);
                    pni_ide = pri_ide * (double)oxmi;
                    pni_ide = dmax_((double)(-ni[k] * odts), pni_ide);
                } else {
                    pri_ide = dmin_(pri_ide, (double)rate_max);
                    prs_ide = (1.0 - T_I(tb.tpi_ide, idx_i, idx_i1))
                            * pri_ide;
                    pri_ide = T_I(tb.tpi_ide, idx_i, idx_i1) * pri_ide;
                }
                if ((idx_i == TF_ntb_i) || (xDi > 5.0f * TF_D0s)) {
                    prs_iau = (double)(ri[k] * .99f * odts);
                    pni_iau = (double)(ni[k] * .95f * odts);
                } else if (xDi < 0.1f * TF_D0s) {
                    prs_iau = 0.0;
                    pni_iau = 0.0;
                } else {
                    prs_iau = T_I(tb.tps_iaus, idx_i, idx_i1) * (double)odts;
                    prs_iau = dmin_((double)(ri[k] * .99f * odts), prs_iau);
                    pni_iau = T_I(tb.tni_iaus, idx_i, idx_i1) * (double)odts;
                    pni_iau = dmin_((double)(ni[k] * .95f * odts), pni_iau);
                }
            }

            // :2554-2576, snow and graupel deposition/sublimation.
            if (L_qs[k]) {
                float C_snow = TF_C_sqrd + FDIV((tempc + 1.5f)
                             * (TF_C_cube - TF_C_sqrd), (-30.0f + 1.5f));
                C_snow = fmax_(TF_C_sqrd, fmin_(C_snow, TF_C_cube));
                prs_sde = (double)(C_snow * t1_subl * diffu[k] * ssati[k]
                                   * rvs * (TF_t1_qs_sd * smo1[k]
                                            + TF_t2_qs_sd * rhof2[k]
                                              * vsc2[k] * smof[k]));
                if (prs_sde < 0.0) {
                    prs_sde = dmax_(dmax_((double)(-rs[k] * odts), prs_sde),
                                    (double)rate_max);
                } else {
                    prs_sde = dmin_(prs_sde, (double)rate_max);
                }
            }
            if (L_qg[k] && ssati[k] < -TF_eps) {
                prg_gde = (double)(TF_C_cube * t1_subl * diffu[k] * ssati[k]
                                   * rvs) * N0_g[k]
                        * ((double)TF_t1_qg_sd
                           * tf_pow(ilamg[k], (double)CGE(10))
                           + (double)(TF_t2_qg_sd * vsc2[k] * rhof2[k])
                             * tf_pow(ilamg[k], (double)CGE(11)));
                if (prg_gde < 0.0) {
                    prg_gde = dmax_(dmax_((double)(-rg[k] * odts), prg_gde),
                                    (double)rate_max);
                } else {
                    prg_gde = dmin_(prg_gde, (double)rate_max);
                }
            }

            // :2579-2603, snow and rain collecting cloud ice.
            if (L_qi[k]) {
                lami = (double)tf_powf(TF_am_i * CIG(2) * TF_oig1 * ni[k]
                                       / ri[k], TF_obmi);
                ilami = 1.0 / lami;
                const float xDi = (float)dmax_((double)TF_D0i,
                    (double)(TF_bm_i + TF_mu_i + 1.0f) * ilami);
                const float xmi = TF_am_i * tf_powf(xDi, TF_bm_i);
                const float oxmi = 1.0f / xmi;
                if (rs[k] >= TF_r_s[0]) {
                    prs_sci = (double)(TF_t1_qs_qi * rhof[k] * TF_Ef_si * ri[k]
                                       * smoe[k]);
                    pni_sci = prs_sci * (double)oxmi;
                }
                if (rr[k] >= TF_r_r[0] && mvd_r[k] > 4.0f * xDi) {
                    lamr = 1.0 / ilamr[k];
                    const double f9 = tf_pow(lamr + (double)TF_fv_r,
                                             (double)(-CRE(9)));
                    pri_rci = (double)(rhof[k] * TF_t1_qr_qi * TF_Ef_ri
                                       * ri[k]) * N0_r[k] * f9;
                    pnr_rci = (double)(rhof[k] * TF_t1_qr_qi * TF_Ef_ri
                                       * ni[k]) * N0_r[k] * f9;
                    pni_rci = pri_rci * (double)oxmi;
                    prr_rci = (double)(rhof[k] * TF_t2_qr_qi * TF_Ef_ri
                                       * ni[k]) * N0_r[k]
                            * tf_pow(lamr + (double)TF_fv_r,
                                     (double)(-CRE(8)));
                    prr_rci = dmin_((double)(rr[k] * odts), prr_rci);
                    prg_rci = pri_rci + prr_rci;
                }
            }

            // :2606-2619, Hallett-Mossop.
            if (prg_gcw > (double)TF_eps && tempc > -8.0f) {
                float tf = 0.0f;
                if (tempc >= -5.0f && tempc < -3.0f) {
                    tf = 0.5f * (-3.0f - tempc);
                } else if (tempc > -8.0f && tempc < -5.0f) {
                    tf = 0.33333333f * (8.0f + tempc);
                }
                pni_ihm = (double)(3.5E8f * tf) * prg_gcw;
                pri_ihm = (double)TF_xm0i * pni_ihm;
                prs_ihm = prs_scw / (prs_scw + prg_gcw) * pri_ihm;
                prg_ihm = prg_gcw / (prs_scw + prg_gcw) * pri_ihm;
            }

            // :2625-2632, rimed snow to graupel.
            if (prs_scw > (double)2.0f * prs_sde && prs_sde > (double)TF_eps) {
                const float r_frac = (float)dmin_(30.0, prs_scw / prs_sde);
                const float g_frac = fmin_(0.95f, 0.15f + (r_frac - 2.0f)
                                                         * .028f);
                vts_boost[k] = fmin_(1.5f, 1.1f + (r_frac - 2.0f) * .016f);
                prg_scw = (double)g_frac * prs_scw;
                prs_scw = (double)(1.0f - g_frac) * prs_scw;
            }
        } else {
            // :2638-2683, at or above 0 C.
            if (L_qs[k]) {
                prr_sml = (double)((tempc * tcond[k] - TF_lvap0 * diffu[k]
                                    * delQvs[k])
                                   * (TF_t1_qs_me * smo1[k] + TF_t2_qs_me
                                      * rhof2[k] * vsc2[k] * smof[k]));
                prr_sml = prr_sml + (double)(4218.0f * TF_olfus * tempc)
                        * (prr_rcs + prs_scw);
                prr_sml = dmin_((double)(rs[k] * odts), dmax_(0.0, prr_sml));
                pnr_sml = (double)(smo0[k] / rs[k]) * prr_sml
                        * (double)tf_powf(10.0f, -0.25f * tempc);
                pnr_sml = dmin_((double)(smo0[k] * odts), pnr_sml);
                if (ssati[k] < 0.0f) {
                    prs_sde = (double)(TF_C_cube * t1_subl * diffu[k]
                                       * ssati[k] * rvs
                                       * (TF_t1_qs_sd * smo1[k] + TF_t2_qs_sd
                                          * rhof2[k] * vsc2[k] * smof[k]));
                    prs_sde = dmax_((double)(-rs[k] * odts), prs_sde);
                }
            }
            if (L_qg[k]) {
                prr_gml = (double)(tempc * tcond[k] - TF_lvap0 * diffu[k]
                                   * delQvs[k]) * N0_g[k]
                        * ((double)TF_t1_qg_me
                           * tf_pow(ilamg[k], (double)CGE(10))
                           + (double)(TF_t2_qg_me * rhof2[k] * vsc2[k])
                             * tf_pow(ilamg[k], (double)CGE(11)));
                prr_gml = dmin_((double)(rg[k] * odts), dmax_(0.0, prr_gml));
                pnr_gml = N0_g[k] * (double)CGG(2)
                        * tf_pow(ilamg[k], (double)CGE(2)) / (double)rg[k]
                        * prr_gml * (double)tf_powf(10.0f, -0.5f * tempc);
                if (ssati[k] < 0.0f) {
                    prg_gde = (double)(TF_C_cube * t1_subl * diffu[k]
                                       * ssati[k] * rvs) * N0_g[k]
                            * ((double)TF_t1_qg_sd
                               * tf_pow(ilamg[k], (double)CGE(10))
                               + (double)(TF_t2_qg_sd * vsc2[k] * rhof2[k])
                                 * tf_pow(ilamg[k], (double)CGE(11)));
                    prg_gde = dmax_((double)(-rg[k] * odts), prg_gde);
                }
            }
            if (dt > 120.0f) {
                prr_rcw = prr_rcw + prs_scw + prg_gcw;
                prs_scw = 0.0;
                prg_gcw = 0.0;
            }
        }

        // ---- conservation limiters (:2699-2787) ----
        {
            float sump = (float)(pri_inu + pri_ide + prs_ide + prs_sde
                                 + prg_gde + pri_iha);
            float rate_max = (qv[k] - qvsi[k]) * odts * 0.999f;
            if ((sump > TF_eps && sump > rate_max)
                    || (sump < -TF_eps && sump < rate_max)) {
                const float ratio = rate_max / sump;
                pri_inu = pri_inu * (double)ratio;
                pri_ide = pri_ide * (double)ratio;
                pni_ide = pni_ide * (double)ratio;
                prs_ide = prs_ide * (double)ratio;
                prs_sde = prs_sde * (double)ratio;
                prg_gde = prg_gde * (double)ratio;
                pri_iha = pri_iha * (double)ratio;
            }

            sump = (float)(-prr_wau - pri_wfz - prr_rcw - prs_scw - prg_scw
                           - prg_gcw);
            rate_max = -rc[k] * odts;
            if (sump < rate_max && L_qc[k]) {
                const float ratio = rate_max / sump;
                prr_wau = prr_wau * (double)ratio;
                pri_wfz = pri_wfz * (double)ratio;
                prr_rcw = prr_rcw * (double)ratio;
                prs_scw = prs_scw * (double)ratio;
                prg_scw = prg_scw * (double)ratio;
                prg_gcw = prg_gcw * (double)ratio;
            }

            sump = (float)(pri_ide - prs_iau - prs_sci - pri_rci);
            rate_max = -ri[k] * odts;
            if (sump < rate_max && L_qi[k]) {
                const float ratio = rate_max / sump;
                pri_ide = pri_ide * (double)ratio;
                prs_iau = prs_iau * (double)ratio;
                prs_sci = prs_sci * (double)ratio;
                pri_rci = pri_rci * (double)ratio;
            }

            sump = (float)(-prg_rfz - pri_rfz - prr_rci + prr_rcs + prr_rcg);
            rate_max = -rr[k] * odts;
            if (sump < rate_max && L_qr[k]) {
                const float ratio = rate_max / sump;
                prg_rfz = prg_rfz * (double)ratio;
                pri_rfz = pri_rfz * (double)ratio;
                prr_rci = prr_rci * (double)ratio;
                prr_rcs = prr_rcs * (double)ratio;
                prr_rcg = prr_rcg * (double)ratio;
            }

            sump = (float)(prs_sde - prs_ihm - prr_sml + prs_rcs);
            rate_max = -rs[k] * odts;
            if (sump < rate_max && L_qs[k]) {
                const float ratio = rate_max / sump;
                prs_sde = prs_sde * (double)ratio;
                prs_ihm = prs_ihm * (double)ratio;
                prr_sml = prr_sml * (double)ratio;
                prs_rcs = prs_rcs * (double)ratio;
            }

            sump = (float)(prg_gde - prg_ihm - prr_gml + prg_rcg);
            rate_max = -rg[k] * odts;
            if (sump < rate_max && L_qg[k]) {
                const float ratio = rate_max / sump;
                prg_gde = prg_gde * (double)ratio;
                prg_ihm = prg_ihm * (double)ratio;
                prr_gml = prr_gml * (double)ratio;
                prg_rcg = prg_rcg * (double)ratio;
            }

            pri_ihm = prs_ihm + prg_ihm;
            float ratio = (float)dmin_(fabs(prr_rcg), fabs(prg_rcg));
            prr_rcg = (double)(ratio * copysignf(1.0f, (float)prr_rcg));
            prg_rcg = -prr_rcg;
            if (temp[k] > TF_T_0) {
                ratio = (float)dmin_(fabs(prr_rcs), fabs(prs_rcs));
                prr_rcs = (double)(ratio * copysignf(1.0f, (float)prr_rcs));
                prs_rcs = -prr_rcs;
            }
        }

        // ---- tendencies (:2796-2960) ----
        {
            const float orho = 1.0f / rho[k];
            const float lfus2 = TF_lsub - lvap[k];

            nwfaten[k] = (float)((double)nwfaten[k] - (pna_rca + pna_sca
                                 + pna_gca + pni_iha) * (double)orho);
            nifaten[k] = (float)((double)nifaten[k] - (pnd_rcd + pnd_scd
                                 + pnd_gcd) * (double)orho);
            nifaten[k] = (float)((double)nifaten[k] - pni_inu
                                 * (double)orho);

            qvten[k] = (float)((double)qvten[k] + (-pri_inu - pri_iha
                               - pri_ide - prs_ide - prs_sde - prg_gde)
                               * (double)orho);

            qcten[k] = (float)((double)qcten[k] + (-prr_wau - pri_wfz
                               - prr_rcw - prs_scw - prg_scw - prg_gcw)
                               * (double)orho);

            ncten[k] = (float)((double)ncten[k] + (-pnc_wau - pnc_rcw
                               - pni_wfz - pnc_scw - pnc_gcw)
                               * (double)orho);

            float xrc = fmax_(TF_R1, (qc1d[k] + qcten[k] * dtsave) * rho[k]);
            float xnc = fmax_(2.0f, (nc1d[k] + ncten[k] * dtsave) * rho[k]);
            if (xrc > TF_R1) {
                nu_c = min(15, tf_nint(1000.E6f / xnc) + 2);
                lamc = (double)tf_powf(xnc * TF_am_r * CCG(2, nu_c)
                                       * OCG1(nu_c) / rc[k], TF_obmr);
                const float xDcb = (float)((double)(TF_bm_r + (float)nu_c
                                                    + 1.0f) / lamc);
                if (xDcb < TF_D0c) {
                    lamc = (double)FDIV(CCE(2, nu_c), TF_D0c);
                    xnc = (float)((double)FDIV(CCG(1, nu_c) * OCG2(nu_c) * xrc,
                                               TF_am_r)
                                  * tf_pow(lamc, (double)TF_bm_r));
                    ncten[k] = (xnc - nc1d[k] * rho[k]) * odts * orho;
                } else if (xDcb > TF_D0r * 2.0f) {
                    lamc = (double)FDIV(CCE(2, nu_c), TF_D0r * 2.0f);
                    xnc = (float)((double)FDIV(CCG(1, nu_c) * OCG2(nu_c) * xrc,
                                               TF_am_r)
                                  * tf_pow(lamc, (double)TF_bm_r));
                    ncten[k] = (xnc - nc1d[k] * rho[k]) * odts * orho;
                }
            } else {
                ncten[k] = -nc1d[k] * odts;
            }
            xnc = fmax_(0.0f, (nc1d[k] + ncten[k] * dtsave) * rho[k]);
            if (xnc > TF_Nt_c_max) {
                ncten[k] = (TF_Nt_c_max - nc1d[k] * rho[k]) * odts * orho;
            }

            qiten[k] = (float)((double)qiten[k] + (pri_inu + pri_iha
                               + pri_ihm + pri_wfz + pri_rfz + pri_ide
                               - prs_iau - prs_sci - pri_rci)
                               * (double)orho);

            niten[k] = (float)((double)niten[k] + (pni_inu + pni_iha
                               + pni_ihm + pni_wfz + pni_rfz + pni_ide
                               - pni_iau - pni_sci - pni_rci)
                               * (double)orho);

            const float xri = fmax_(TF_R1, (qi1d[k] + qiten[k] * dtsave)
                                           * rho[k]);
            float xni = fmax_(TF_R2, (ni1d[k] + niten[k] * dtsave) * rho[k]);
            if (xri > TF_R1) {
                lami = (double)tf_powf(TF_am_i * CIG(2) * TF_oig1 * xni / xri,
                                       TF_obmi);
                ilami = 1.0 / lami;
                const float xDi = (float)((double)(TF_bm_i + TF_mu_i + 1.0f)
                                          * ilami);
                if (xDi < 5.E-6f) {
                    lami = (double)FDIV(CIE(2), 5.E-6f);
                    xni = (float)dmin_(499.E3,
                        (double)FDIV(CIG(1) * TF_oig2 * xri, TF_am_i)
                        * tf_pow(lami, (double)TF_bm_i));
                    niten[k] = (xni - ni1d[k] * rho[k]) * odts * orho;
                } else if (xDi > 300.E-6f) {
                    lami = (double)FDIV(CIE(2), 300.E-6f);
                    xni = (float)((double)FDIV(CIG(1) * TF_oig2 * xri, TF_am_i)
                                  * tf_pow(lami, (double)TF_bm_i));
                    niten[k] = (xni - ni1d[k] * rho[k]) * odts * orho;
                }
            } else {
                niten[k] = -ni1d[k] * odts;
            }
            xni = fmax_(0.0f, (ni1d[k] + niten[k] * dtsave) * rho[k]);
            if (xni > 499.E3f) {
                niten[k] = (499.E3f - ni1d[k] * rho[k]) * odts * orho;
            }

            qrten[k] = (float)((double)qrten[k] + (prr_wau + prr_rcw
                               + prr_sml + prr_gml + prr_rcs + prr_rcg
                               - prg_rfz - pri_rfz - prr_rci)
                               * (double)orho);

            nrten[k] = (float)((double)nrten[k] + (pnr_wau + pnr_sml + pnr_gml
                               - (pnr_rfz + pnr_rcr + pnr_rcg + pnr_rcs
                                  + pnr_rci)) * (double)orho);

            const float xrr = fmax_(TF_R1, (qr1d[k] + qrten[k] * dtsave)
                                           * rho[k]);
            float xnr = fmax_(TF_R2, (nr1d[k] + nrten[k] * dtsave) * rho[k]);
            if (xrr > TF_R1) {
                lamr = (double)tf_powf(TF_am_r * CRG(3) * TF_org2 * xnr / xrr,
                                       TF_obmr);
                mvd_r[k] = (float)((double)(3.0f + TF_mu_r + 0.672f) / lamr);
                if (mvd_r[k] > 2.5E-3f) {
                    mvd_r[k] = 2.5E-3f;
                    lamr = (double)((3.0f + TF_mu_r + 0.672f) / mvd_r[k]);
                    xnr = (float)((double)(CRG(2) * TF_org3 * xrr)
                                  * tf_pow(lamr, (double)TF_bm_r)
                                  / (double)TF_am_r);
                    nrten[k] = (xnr - nr1d[k] * rho[k]) * odts * orho;
                } else if (mvd_r[k] < TF_D0r * 0.75f) {
                    mvd_r[k] = TF_D0r * 0.75f;
                    lamr = (double)((3.0f + TF_mu_r + 0.672f) / mvd_r[k]);
                    xnr = (float)((double)(CRG(2) * TF_org3 * xrr)
                                  * tf_pow(lamr, (double)TF_bm_r)
                                  / (double)TF_am_r);
                    nrten[k] = (xnr - nr1d[k] * rho[k]) * odts * orho;
                }
            } else {
                qrten[k] = -qr1d[k] * odts;
                nrten[k] = -nr1d[k] * odts;
            }

            qsten[k] = (float)((double)qsten[k] + (prs_iau + prs_sde
                               + prs_sci + prs_scw + prs_rcs + prs_ide
                               - prs_ihm - prr_sml) * (double)orho);

            qgten[k] = (float)((double)qgten[k] + (prg_scw + prg_rfz
                               + prg_gde + prg_rcg + prg_gcw + prg_rci
                               + prg_rcs - prg_ihm - prr_gml)
                               * (double)orho);

            if (temp[k] < TF_T_0) {
                tten[k] = (float)((double)tten[k]
                    + ((double)(TF_lsub * ocp[k]) * (pri_inu + pri_ide
                                                    + prs_ide + prs_sde
                                                    + prg_gde + pri_iha)
                       + (double)(lfus2 * ocp[k]) * (pri_wfz + pri_rfz
                                                     + prg_rfz + prs_scw
                                                     + prg_scw + prg_gcw
                                                     + prg_rcs + prs_rcs
                                                     + prr_rci + prg_rcg))
                      * (double)orho);
            } else {
                tten[k] = (float)((double)tten[k]
                    + ((double)(TF_lfus * ocp[k]) * (-prr_sml - prr_gml
                                                    - prr_rcg - prr_rcs)
                       + (double)(TF_lsub * ocp[k]) * (prs_sde + prg_gde))
                      * (double)orho);
            }
        }
        prr_gml_k[k] = prr_gml;
        prr_sml_k[k] = prr_sml;
    }

    // :2967-3049, TAU+1 working values.
    for (int k = kts; k <= kte; ++k) {
        temp[k] = t1d[k] + dt * tten[k];
        const float otemp = 1.0f / temp[k];
        const float tempc = temp[k] - 273.15f;
        qv[k] = fmax_(1.E-10f, qv1d[k] + dt * qvten[k]);
        rho[k] = 0.622f * pres[k] / (TF_R * temp[k] * (qv[k] + 0.622f));
        rhof[k] = sqrtf(TF_rho_not / rho[k]);
        rhof2[k] = sqrtf(rhof[k]);
        qvs[k] = tf_rslf(pres[k], temp[k]);
        ssatw[k] = qv[k] / qvs[k] - 1.0f;
        if (fabsf(ssatw[k]) < TF_eps) ssatw[k] = 0.0f;
        diffu[k] = 2.11E-5f * tf_powf(FDIV(temp[k], 273.15f), 1.94f)
                 * (101325.0f / pres[k]);
        if (tempc >= 0.0f) {
            visco[k] = (1.718f + 0.0049f * tempc) * 1.0E-5f;
        } else {
            visco[k] = (1.718f + 0.0049f * tempc - 1.2E-5f * tempc * tempc)
                     * 1.0E-5f;
        }
        vsc2[k] = sqrtf(rho[k] / visco[k]);
        lvap[k] = TF_lvap0 + (2106.0f - 4218.0f) * tempc;
        tcond[k] = (5.69f + 0.0168f * tempc) * 1.0E-5f * 418.936f;
        ocp[k] = 1.0f / (TF_Cp * (1.0f + 0.887f * qv[k]));
        lvt2[k] = lvap[k] * lvap[k] * ocp[k] * TF_oRv * otemp * otemp;

        nwfa[k] = fmax_(11.1E6f, (nwfa1d[k] + nwfaten[k] * dt) * rho[k]);

        if ((qc1d[k] + qcten[k] * dt) > TF_R1) {
            rc[k] = (qc1d[k] + qcten[k] * dt) * rho[k];
            nc[k] = fmax_(2.0f, (nc1d[k] + ncten[k] * dt) * rho[k]);
            L_qc[k] = true;
        } else {
            rc[k] = TF_R1;
            nc[k] = 2.0f;
            L_qc[k] = false;
        }
        if ((qi1d[k] + qiten[k] * dt) > TF_R1) {
            ri[k] = (qi1d[k] + qiten[k] * dt) * rho[k];
            ni[k] = fmax_(TF_R2, (ni1d[k] + niten[k] * dt) * rho[k]);
            L_qi[k] = true;
        } else {
            ri[k] = TF_R1;
            ni[k] = TF_R2;
            L_qi[k] = false;
        }
        if ((qr1d[k] + qrten[k] * dt) > TF_R1) {
            rr[k] = (qr1d[k] + qrten[k] * dt) * rho[k];
            nr[k] = fmax_(TF_R2, (nr1d[k] + nrten[k] * dt) * rho[k]);
            L_qr[k] = true;
            lamr = (double)tf_powf(TF_am_r * CRG(3) * TF_org2 * nr[k] / rr[k],
                                   TF_obmr);
            mvd_r[k] = (float)((double)(3.0f + TF_mu_r + 0.672f) / lamr);
            if (mvd_r[k] > 2.5E-3f) {
                mvd_r[k] = 2.5E-3f;
                lamr = (double)((3.0f + TF_mu_r + 0.672f) / mvd_r[k]);
                nr[k] = (float)((double)(CRG(2) * TF_org3 * rr[k])
                                * tf_pow(lamr, (double)TF_bm_r)
                                / (double)TF_am_r);
            } else if (mvd_r[k] < TF_D0r * 0.75f) {
                mvd_r[k] = TF_D0r * 0.75f;
                lamr = (double)((3.0f + TF_mu_r + 0.672f) / mvd_r[k]);
                nr[k] = (float)((double)(CRG(2) * TF_org3 * rr[k])
                                * tf_pow(lamr, (double)TF_bm_r)
                                / (double)TF_am_r);
            }
        } else {
            rr[k] = TF_R1;
            nr[k] = TF_R2;
            L_qr[k] = false;
        }
        if ((qs1d[k] + qsten[k] * dt) > TF_R1) {
            rs[k] = (qs1d[k] + qsten[k] * dt) * rho[k];
            L_qs[k] = true;
        } else {
            rs[k] = TF_R1;
            L_qs[k] = false;
        }
        if ((qg1d[k] + qgten[k] * dt) > TF_R1) {
            rg[k] = (qg1d[k] + qgten[k] * dt) * rho[k];
            L_qg[k] = true;
        } else {
            rg[k] = TF_R1;
            L_qg[k] = false;
        }
    }

    // :3056-3105, snow moments again (smob, smoc and the fall moment).
    for (int k = kts; k <= kte; ++k) {
        if (!L_qs[k]) continue;
        const float tc0 = fmin_(-0.1f, temp[k] - 273.15f);
        smob[k] = rs[k] * TF_oams;
        smo2[k] = smob[k];
        const float loga_ = tf_loga(tc0, CSE(1));
        const float a_ = tf_powf(10.0f, loga_);
        const float b_ = tf_b(tc0, CSE(1));
        smoc[k] = a_ * tf_powf(smo2[k], b_);
        // smod (:3094-3104) feeds nothing the fork reads afterwards.
    }

    // :3110-3134, graupel intercept and slope again.
    N0_min = (double)TF_gonv_max;
    k_0 = kts;
    for (int k = kte; k >= kts; --k) {
        if (temp[k] >= 270.65f) k_0 = max(k_0, k);
    }
    for (int k = kte; k >= kts; --k) {
        float xslw1;
        if (k > k_0 && L_qr[k] && mvd_r[k] > 100.E-6f) {
            xslw1 = 4.01f + tf_log10f(mvd_r[k]);
        } else {
            xslw1 = 0.01f;
        }
        const float ygra1 = 4.31f + tf_log10f(fmax_(5.E-5f, rg[k]));
        const float zans1 = (3.1f + (100.0f / (300.0f * xslw1 * ygra1
            / (10.0f / xslw1 + 1.0f + 0.25f * ygra1) + 30.0f
            + 10.0f * ygra1))) + rand1;
        N0_exp = (double)tf_powf(10.0f, zans1);
        N0_exp = dmax_((double)TF_gonv_min, dmin_(N0_exp,
                                                  (double)TF_gonv_max));
        N0_min = dmin_(N0_exp, N0_min);
        N0_exp = N0_min;
        lam_exp = tf_pow(N0_exp * (double)TF_am_g * (double)CGG(1)
                         / (double)rg[k], (double)TF_oge1);
        lamg = lam_exp * (double)tf_powf(CGG(3) * TF_ogg2 * TF_ogg1,
                                         TF_obmg);
        ilamg[k] = 1.0 / lamg;
        N0_g[k] = N0_exp / ((double)CGG(2) * lam_exp)
                * tf_pow(lamg, (double)CGE(2));
    }

    // :3141-3146, rain intercept and slope again.
    for (int k = kte; k >= kts; --k) {
        lamr = (double)tf_powf(TF_am_r * CRG(3) * TF_org2 * nr[k] / rr[k],
                               TF_obmr);
        ilamr[k] = 1.0 / lamr;
        mvd_r[k] = (float)((double)(3.0f + TF_mu_r + 0.672f) / lamr);
        N0_r[k] = (double)(nr[k] * TF_org2) * tf_pow(lamr, (double)CRE(2));
    }

    // :3156-3250, condensation/evaporation and droplet activation.
    for (int k = kts; k <= kte; ++k) {
        const float orho = 1.0f / rho[k];
        if ((ssatw[k] > TF_eps) || (ssatw[k] < -TF_eps && L_qc[k])) {
            double pnc_wcd = 0.0;
            float clap = (qv[k] - qvs[k]) / (1.0f + lvt2[k] * qvs[k]);
            for (int n = 1; n <= 3; ++n) {
                const float fcd = qvs[k] * tf_expf(lvt2[k] * clap) - qv[k]
                                + clap;
                const float dfcd = qvs[k] * lvt2[k] * tf_expf(lvt2[k] * clap)
                                 + 1.0f;
                clap = clap - fcd / dfcd;
            }
            const float xrc = rc[k] + clap * rho[k];
            float xnc = 0.0f;
            if (xrc > TF_R1) {
                prw_vcd[k] = (double)(clap * odt);
                if (clap > TF_eps) {
                    xnc = fmax_(2.0f, tf_activ_ncloud(temp[k], w1d[k] + rand3,
                                                      nwfa[k], tb.tnccn_act));
                    pnc_wcd = (double)(0.5f * (xnc - nc[k]
                                               + fabsf(xnc - nc[k]))
                                       * odts * orho);
                } else if (clap < -TF_eps && ssatw[k] < -1.E-6f) {
                    const float otemp = 1.0f / temp[k];
                    const float rvs = rho[k] * qvs[k];
                    const float rvs_p = rvs * otemp * (lvap[k] * otemp * TF_oRv
                                                       - 1.0f);
                    const float rvs_pp = rvs * (otemp * (lvap[k] * otemp
                                                         * TF_oRv - 1.0f)
                        * otemp * (lvap[k] * otemp * TF_oRv - 1.0f)
                        + (-2.0f * lvap[k] * otemp * otemp * otemp * TF_oRv)
                        + otemp * otemp);
                    const float gamsc = lvap[k] * diffu[k] / tcond[k] * rvs_p;
                    float alphsc = 0.5f * (gamsc / (1.0f + gamsc))
                                 * (gamsc / (1.0f + gamsc))
                                 * rvs_pp / rvs_p * rvs / rvs_p;
                    alphsc = fmax_(1.E-9f, alphsc);
                    float xsat = ssatw[k];
                    if (fabsf(xsat) < 1.E-9f) xsat = 0.0f;
                    const float t1_evap = 2.0f * TF_PI * (1.0f - alphsc * xsat
                        + 2.0f * alphsc * alphsc * xsat * xsat
                        - 5.0f * alphsc * alphsc * alphsc * xsat * xsat
                          * xsat) / (1.0f + gamsc);
                    const double Dc_star = sqrt(-2.0 * (double)dt
                        * (double)t1_evap / (double)(2.0f * TF_PI)
                        * (double)4.0f * (double)diffu[k] * (double)ssatw[k]
                        * (double)rvs / (double)TF_rho_w);
                    const int idx_d = max(1, min((int)((double)1.E6f
                                                       * Dc_star), TF_nbc));
                    int idx_n = tf_nintd((double)1.0f + (double)(float)TF_nbc
                                         * tf_log((double)nc[k] / TF_t_Nc[0])
                                         / (double)TF_nic1);
                    idx_n = max(1, min(idx_n, TF_nbc));
                    const int idx_c = rc[k] > TF_r_c[0]
                        ? tf_decade_bin(rc[k], TF_nic2, TF_ntb_c) : 1;
                    prw_vcd[k] = dmax_((double)(-rc[k] * 0.99f * orho * odt),
                                       prw_vcd[k]);
                    pnc_wcd = dmax_((double)(-nc[k] * 0.99f * orho * odt),
                                    -T_WEV(tb.tnc_wev, idx_d, idx_c, idx_n)
                                    * (double)orho * (double)odt);
                }
            } else {
                prw_vcd[k] = (double)(-rc[k] * orho * odt);
                pnc_wcd = (double)(-nc[k] * orho * odt);
            }

            qvten[k] = (float)((double)qvten[k] - prw_vcd[k]);
            qcten[k] = (float)((double)qcten[k] + prw_vcd[k]);
            ncten[k] = (float)((double)ncten[k] + pnc_wcd);
            nwfaten[k] = (float)((double)nwfaten[k] - pnc_wcd);
            tten[k] = (float)((double)tten[k] + (double)(lvap[k] * ocp[k])
                              * prw_vcd[k]);
            rc[k] = fmax_(TF_R1, (qc1d[k] + dt * qcten[k]) * rho[k]);
            nc[k] = fmax_(2.0f, (nc1d[k] + dt * ncten[k]) * rho[k]);
            qv[k] = fmax_(1.E-10f, qv1d[k] + dt * qvten[k]);
            temp[k] = t1d[k] + dt * tten[k];
            rho[k] = 0.622f * pres[k] / (TF_R * temp[k] * (qv[k] + 0.622f));
            qvs[k] = tf_rslf(pres[k], temp[k]);
            ssatw[k] = qv[k] / qvs[k] - 1.0f;
        }
    }

    // :3256-3330, rain evaporation.
    for (int k = kts; k <= kte; ++k) {
        if ((ssatw[k] < -TF_eps) && L_qr[k] && (!(prw_vcd[k] > 0.0))) {
            const float tempc = temp[k] - 273.15f;
            const float otemp = 1.0f / temp[k];
            const float orho = 1.0f / rho[k];
            rhof[k] = sqrtf(TF_rho_not * orho);
            rhof2[k] = sqrtf(rhof[k]);
            diffu[k] = 2.11E-5f * tf_powf(FDIV(temp[k], 273.15f), 1.94f)
                     * (101325.0f / pres[k]);
            if (tempc >= 0.0f) {
                visco[k] = (1.718f + 0.0049f * tempc) * 1.0E-5f;
            } else {
                visco[k] = (1.718f + 0.0049f * tempc
                            - 1.2E-5f * tempc * tempc) * 1.0E-5f;
            }
            vsc2[k] = sqrtf(rho[k] / visco[k]);
            lvap[k] = TF_lvap0 + (2106.0f - 4218.0f) * tempc;
            tcond[k] = (5.69f + 0.0168f * tempc) * 1.0E-5f * 418.936f;
            ocp[k] = 1.0f / (TF_Cp * (1.0f + 0.887f * qv[k]));

            const float rvs = rho[k] * qvs[k];
            const float rvs_p = rvs * otemp * (lvap[k] * otemp * TF_oRv
                                               - 1.0f);
            const float rvs_pp = rvs * (otemp * (lvap[k] * otemp * TF_oRv
                                                 - 1.0f)
                * otemp * (lvap[k] * otemp * TF_oRv - 1.0f)
                + (-2.0f * lvap[k] * otemp * otemp * otemp * TF_oRv)
                + otemp * otemp);
            const float gamsc = lvap[k] * diffu[k] / tcond[k] * rvs_p;
            float alphsc = 0.5f * (gamsc / (1.0f + gamsc))
                         * (gamsc / (1.0f + gamsc)) * rvs_pp / rvs_p * rvs
                         / rvs_p;
            alphsc = fmax_(1.E-9f, alphsc);
            const float xsat = fmin_(-1.E-9f, ssatw[k]);
            const float t1_evap = 2.0f * TF_PI * (1.0f - alphsc * xsat
                + 2.0f * alphsc * alphsc * xsat * xsat
                - 5.0f * alphsc * alphsc * alphsc * xsat * xsat * xsat)
                / (1.0f + gamsc);

            lamr = 1.0 / ilamr[k];
            double prv_rev;
            if (qv[k] / qvs[k] < 0.95f && rr[k] * orho <= 1.E-8f) {
                prv_rev = (double)(rr[k] * orho * odts);
            } else {
                prv_rev = (double)(t1_evap * diffu[k] * (-ssatw[k]))
                        * N0_r[k] * (double)rvs
                        * ((double)TF_t1_qr_ev
                           * tf_pow(ilamr[k], (double)CRE(10))
                           + (double)(TF_t2_qr_ev * vsc2[k] * rhof2[k])
                             * tf_pow(lamr + (double)(0.5f * TF_fv_r),
                                      (double)(-CRE(11))));
                const float rate_max = fmin_((rr[k] * orho * odts),
                                             (qvs[k] - qv[k]) * odts);
                prv_rev = dmin_((double)rate_max, prv_rev * (double)orho);
                if (prr_gml_k[k] > 0.0) {
                    const float eva_factor = fmin_(1.0f, 0.01f
                        + (0.99f - 0.01f) * FDIV(tempc, 20.0f));
                    prv_rev = prv_rev * (double)eva_factor;
                }
            }
            const double pnr_rev = dmin_((double)(nr[k] * 0.99f * orho
                                                  * odts),
                                         prv_rev * (double)nr[k]
                                         / (double)rr[k]);

            qrten[k] = (float)((double)qrten[k] - prv_rev);
            qvten[k] = (float)((double)qvten[k] + prv_rev);
            nrten[k] = (float)((double)nrten[k] - pnr_rev);
            nwfaten[k] = (float)((double)nwfaten[k] + pnr_rev);
            tten[k] = (float)((double)tten[k] - (double)(lvap[k] * ocp[k])
                              * prv_rev);

            rr[k] = fmax_(TF_R1, (qr1d[k] + dt * qrten[k]) * rho[k]);
            qv[k] = fmax_(1.E-10f, qv1d[k] + dt * qvten[k]);
            nr[k] = fmax_(TF_R2, (nr1d[k] + dt * nrten[k]) * rho[k]);
            temp[k] = t1d[k] + dt * tten[k];
            rho[k] = 0.622f * pres[k] / (TF_R * temp[k] * (qv[k] + 0.622f));
        }
    }

    // :3350-3518, fall speeds and sub-step counts.
    TfCol<float> vtrk = ws.F(), vtnrk = ws.F(), vtik = ws.F();
    TfCol<float> vtnik = ws.F(), vtsk = ws.F(), vtgk = ws.F();
    TfCol<float> vtck = ws.F(), vtnck = ws.F();
    float onstep[5];
    int ksed1[5];
    int nstep = 0;
    for (int i = 0; i < 5; ++i) { onstep[i] = 1.0f; ksed1[i] = kts; }
    for (int k = kte + 1; k >= kts; --k) {
        vtrk[k] = 0.0f; vtnrk[k] = 0.0f; vtik[k] = 0.0f; vtnik[k] = 0.0f;
        vtsk[k] = 0.0f; vtgk[k] = 0.0f; vtck[k] = 0.0f; vtnck[k] = 0.0f;
    }
    for (int k = kte; k >= kts; --k) {
        float vtr = 0.0f;
        rhof[k] = sqrtf(TF_rho_not / rho[k]);
        if (rr[k] > TF_R1) {
            lamr = (double)tf_powf(TF_am_r * CRG(3) * TF_org2 * nr[k] / rr[k],
                                   TF_obmr);
            vtr = (float)((double)(rhof[k] * TF_av_r * CRG(6) * TF_org3)
                          * tf_pow(lamr, (double)CRE(3))
                          * tf_pow(lamr + (double)TF_fv_r,
                                   (double)(-CRE(6))));
            vtrk[k] = vtr;
            vtr = (float)((double)(rhof[k] * TF_av_r * CRG(7) / CRG(12))
                          * tf_pow(lamr, (double)CRE(12))
                          * tf_pow(lamr + (double)TF_fv_r,
                                   (double)(-CRE(7))));
            vtnrk[k] = vtr;
        } else {
            vtrk[k] = vtrk[k + 1];
            vtnrk[k] = vtnrk[k + 1];
        }
        if (fmax_(vtrk[k], vtnrk[k]) > 1.E-3f) {
            ksed1[0] = max(ksed1[0], k);
            const float delta_tp = dzq[k] / (fmax_(vtrk[k], vtnrk[k]));
            nstep = max(nstep, (int)(dt / delta_tp + 1.0f));
        }
    }
    if (ksed1[0] == kte) ksed1[0] = kte - 1;
    if (nstep > 0) onstep[0] = 1.0f / (float)nstep;

    {
        float hgt_agl = 0.0f;
        for (int k = kts; k <= kte - 1; ++k) {
            if (rc[k] > TF_R2) ksed1[4] = k;
            hgt_agl = hgt_agl + dzq[k];
            if (hgt_agl > 500.0f) break;
        }
    }
    for (int k = ksed1[4]; k >= kts; --k) {
        if (rc[k] > TF_R1 && w1d[k] < 1.E-1f) {
            nu_c = min(15, tf_nint(1000.E6f / nc[k]) + 2);
            lamc = (double)tf_powf(nc[k] * TF_am_r * CCG(2, nu_c)
                                   * OCG1(nu_c) / rc[k], TF_obmr);
            const double ilamc = 1.0 / lamc;
            vtck[k] = (float)((double)(rhof[k] * TF_av_c * CCG(5, nu_c)
                                       * OCG2(nu_c)) * (ilamc * ilamc));
            vtnck[k] = (float)((double)(rhof[k] * TF_av_c * CCG(4, nu_c)
                                        * OCG1(nu_c)) * (ilamc * ilamc));
        }
    }

    nstep = 0;
    for (int k = kte; k >= kts; --k) {
        if (ri[k] > TF_R1) {
            lami = (double)tf_powf(TF_am_i * CIG(2) * TF_oig1 * ni[k] / ri[k],
                                   TF_obmi);
            ilami = 1.0 / lami;
            vtik[k] = (float)((double)(rhof[k] * TF_av_i * CIG(3) * TF_oig2)
                              * ilami);
            vtnik[k] = (float)((double)(rhof[k] * TF_av_i * CIG(6) / CIG(7))
                               * ilami);
        } else {
            vtik[k] = vtik[k + 1];
            vtnik[k] = vtnik[k + 1];
        }
        if (vtik[k] > 1.E-3f) {
            ksed1[1] = max(ksed1[1], k);
            const float delta_tp = dzq[k] / vtik[k];
            nstep = max(nstep, (int)(dt / delta_tp + 1.0f));
        }
    }
    if (ksed1[1] == kte) ksed1[1] = kte - 1;
    if (nstep > 0) onstep[1] = 1.0f / (float)nstep;

    nstep = 0;
    for (int k = kte; k >= kts; --k) {
        if (rs[k] > TF_R1) {
            const float xDs = smoc[k] / smob[k];
            const float Mrat = 1.0f / xDs;
            float ils1 = 1.0f / (Mrat * TF_Lam0 + TF_fv_s);
            float ils2 = 1.0f / (Mrat * TF_Lam1 + TF_fv_s);
            const float t1_vts = TF_Kap0 * CSG(4) * tf_powf(ils1, CSE(4));
            const float t2_vts = TF_Kap1 * tf_powf(Mrat, TF_mu_s) * CSG(10)
                               * tf_powf(ils2, CSE(10));
            ils1 = 1.0f / (Mrat * TF_Lam0);
            ils2 = 1.0f / (Mrat * TF_Lam1);
            const float t3_vts = TF_Kap0 * CSG(1) * tf_powf(ils1, CSE(1));
            const float t4_vts = TF_Kap1 * tf_powf(Mrat, TF_mu_s) * CSG(7)
                               * tf_powf(ils2, CSE(7));
            const float vts = rhof[k] * TF_av_s * (t1_vts + t2_vts)
                            / (t3_vts + t4_vts);
            if (singular_fall) {
                if (temp[k] > (TF_T_0 + 0.1f)) {
                    vtsk[k] = fmax_(vts * vts_boost[k],
                                    vts * ((vtrk[k] - vts * vts_boost[k])
                                           / (temp[k] - TF_T_0)));
                } else {
                    vtsk[k] = vts * vts_boost[k];
                }
            } else {
                // thompson_fork_snow_fall = "blend": WRF v4.6.1's melting
                // blend (v4.6.1 module_mp_thompson.F:3722-3726).
                if (prr_sml_k[k] > 0.0) {
                    const float SR = rs[k] / (rs[k] + rr[k]);
                    vtsk[k] = vts * SR + (1.0f - SR) * vtrk[k];
                } else {
                    vtsk[k] = vts * vts_boost[k];
                }
            }
        } else {
            vtsk[k] = vtsk[k + 1];
        }
        if (vtsk[k] > 1.E-3f) {
            ksed1[2] = max(ksed1[2], k);
            const float delta_tp = dzq[k] / vtsk[k];
            nstep = max(nstep, (int)(dt / delta_tp + 1.0f));
        }
    }
    if (ksed1[2] == kte) ksed1[2] = kte - 1;
    if (nstep > 0) onstep[2] = 1.0f / (float)nstep;

    nstep = 0;
    for (int k = kte; k >= kts; --k) {
        if (rg[k] > TF_R1) {
            const float vtg = (float)((double)(rhof[k] * TF_av_g * CGG(6)
                                               * TF_ogg3)
                                      * tf_pow(ilamg[k], (double)TF_bv_g));
            if (temp[k] > TF_T_0) {
                vtgk[k] = fmax_(vtg, vtrk[k]);
            } else {
                vtgk[k] = vtg;
            }
        } else {
            vtgk[k] = vtgk[k + 1];
        }
        if (vtgk[k] > 1.E-3f) {
            ksed1[3] = max(ksed1[3], k);
            const float delta_tp = dzq[k] / vtgk[k];
            nstep = max(nstep, (int)(dt / delta_tp + 1.0f));
        }
    }
    if (ksed1[3] == kte) ksed1[3] = kte - 1;
    if (nstep > 0) onstep[3] = 1.0f / (float)nstep;

    // :3530-3655, sedimentation in tendency form.
    TfCol<float> sed_a = ws.F(), sed_n = ws.F();
    nstep = tf_nint(1.0f / onstep[0]);
    for (int n = 1; n <= nstep; ++n) {
        for (int k = kte; k >= kts; --k) {
            sed_a[k] = vtrk[k] * rr[k];
            sed_n[k] = vtnrk[k] * nr[k];
        }
        int k = kte;
        float odzq = 1.0f / dzq[k];
        float orho = 1.0f / rho[k];
        qrten[k] = qrten[k] - sed_a[k] * odzq * onstep[0] * orho;
        nrten[k] = nrten[k] - sed_n[k] * odzq * onstep[0] * orho;
        rr[k] = fmax_(TF_R1, rr[k] - sed_a[k] * odzq * dt * onstep[0]);
        nr[k] = fmax_(TF_R2, nr[k] - sed_n[k] * odzq * dt * onstep[0]);
        for (k = ksed1[0]; k >= kts; --k) {
            odzq = 1.0f / dzq[k];
            orho = 1.0f / rho[k];
            qrten[k] = qrten[k] + (sed_a[k + 1] - sed_a[k])
                     * odzq * onstep[0] * orho;
            nrten[k] = nrten[k] + (sed_n[k + 1] - sed_n[k])
                     * odzq * onstep[0] * orho;
            rr[k] = fmax_(TF_R1, rr[k] + (sed_a[k + 1] - sed_a[k])
                                 * odzq * dt * onstep[0]);
            nr[k] = fmax_(TF_R2, nr[k] + (sed_n[k + 1] - sed_n[k])
                                 * odzq * dt * onstep[0]);
        }
        if (rr[kts] > TF_R1 * 10.0f) {
            *pptrain = *pptrain + sed_a[kts] * dt * onstep[0];
        }
    }

    for (int k = kte; k >= kts; --k) {
        sed_a[k] = vtck[k] * rc[k];
        sed_n[k] = vtnck[k] * nc[k];
    }
    for (int k = ksed1[4]; k >= kts; --k) {
        const float odzq = 1.0f / dzq[k];
        const float orho = 1.0f / rho[k];
        qcten[k] = qcten[k] + (sed_a[k + 1] - sed_a[k]) * odzq * orho;
        ncten[k] = ncten[k] + (sed_n[k + 1] - sed_n[k]) * odzq * orho;
        rc[k] = fmax_(TF_R1, rc[k] + (sed_a[k + 1] - sed_a[k]) * odzq * dt);
        nc[k] = fmax_(10.0f, nc[k] + (sed_n[k + 1] - sed_n[k]) * odzq * dt);
    }

    nstep = tf_nint(1.0f / onstep[1]);
    for (int n = 1; n <= nstep; ++n) {
        for (int k = kte; k >= kts; --k) {
            sed_a[k] = vtik[k] * ri[k];
            sed_n[k] = vtnik[k] * ni[k];
        }
        int k = kte;
        float odzq = 1.0f / dzq[k];
        float orho = 1.0f / rho[k];
        qiten[k] = qiten[k] - sed_a[k] * odzq * onstep[1] * orho;
        niten[k] = niten[k] - sed_n[k] * odzq * onstep[1] * orho;
        ri[k] = fmax_(TF_R1, ri[k] - sed_a[k] * odzq * dt * onstep[1]);
        ni[k] = fmax_(TF_R2, ni[k] - sed_n[k] * odzq * dt * onstep[1]);
        for (k = ksed1[1]; k >= kts; --k) {
            odzq = 1.0f / dzq[k];
            orho = 1.0f / rho[k];
            qiten[k] = qiten[k] + (sed_a[k + 1] - sed_a[k])
                     * odzq * onstep[1] * orho;
            niten[k] = niten[k] + (sed_n[k + 1] - sed_n[k])
                     * odzq * onstep[1] * orho;
            ri[k] = fmax_(TF_R1, ri[k] + (sed_a[k + 1] - sed_a[k])
                                 * odzq * dt * onstep[1]);
            ni[k] = fmax_(TF_R2, ni[k] + (sed_n[k + 1] - sed_n[k])
                                 * odzq * dt * onstep[1]);
        }
        if (ri[kts] > TF_R1 * 10.0f) {
            *pptice = *pptice + sed_a[kts] * dt * onstep[1];
        }
    }

    nstep = tf_nint(1.0f / onstep[2]);
    for (int n = 1; n <= nstep; ++n) {
        for (int k = kte; k >= kts; --k) {
            sed_a[k] = vtsk[k] * rs[k];
        }
        int k = kte;
        float odzq = 1.0f / dzq[k];
        float orho = 1.0f / rho[k];
        qsten[k] = qsten[k] - sed_a[k] * odzq * onstep[2] * orho;
        rs[k] = fmax_(TF_R1, rs[k] - sed_a[k] * odzq * dt * onstep[2]);
        for (k = ksed1[2]; k >= kts; --k) {
            odzq = 1.0f / dzq[k];
            orho = 1.0f / rho[k];
            qsten[k] = qsten[k] + (sed_a[k + 1] - sed_a[k])
                     * odzq * onstep[2] * orho;
            rs[k] = fmax_(TF_R1, rs[k] + (sed_a[k + 1] - sed_a[k])
                                 * odzq * dt * onstep[2]);
        }
        if (rs[kts] > TF_R1 * 10.0f) {
            *pptsnow = *pptsnow + sed_a[kts] * dt * onstep[2];
        }
    }

    nstep = tf_nint(1.0f / onstep[3]);
    for (int n = 1; n <= nstep; ++n) {
        for (int k = kte; k >= kts; --k) {
            sed_a[k] = vtgk[k] * rg[k];
        }
        int k = kte;
        float odzq = 1.0f / dzq[k];
        float orho = 1.0f / rho[k];
        qgten[k] = qgten[k] - sed_a[k] * odzq * onstep[3] * orho;
        rg[k] = fmax_(TF_R1, rg[k] - sed_a[k] * odzq * dt * onstep[3]);
        for (k = ksed1[3]; k >= kts; --k) {
            odzq = 1.0f / dzq[k];
            orho = 1.0f / rho[k];
            qgten[k] = qgten[k] + (sed_a[k + 1] - sed_a[k])
                     * odzq * onstep[3] * orho;
            rg[k] = fmax_(TF_R1, rg[k] + (sed_a[k + 1] - sed_a[k])
                                 * odzq * dt * onstep[3]);
        }
        if (rg[kts] > TF_R1 * 10.0f) {
            *pptgraul = *pptgraul + sed_a[kts] * dt * onstep[3];
        }
    }

    // :3661-3683, instant melt of ice above 0 C, instant freeze of cloud
    // below HGFR.
    for (int k = kts; k <= kte; ++k) {
        const float xri = fmax_(0.0f, qi1d[k] + qiten[k] * dt);
        if ((temp[k] > TF_T_0) && (xri > 0.0f)) {
            qcten[k] = qcten[k] + xri * odt;
            ncten[k] = ncten[k] + ni1d[k] * odt;
            qiten[k] = qiten[k] - xri * odt;
            niten[k] = -ni1d[k] * odt;
            tten[k] = tten[k] - TF_lfus * ocp[k] * xri * odt;
        }
        const float xrc = fmax_(0.0f, qc1d[k] + qcten[k] * dt);
        if ((temp[k] < TF_HGFR) && (xrc > 0.0f)) {
            const float lfus2 = TF_lsub - lvap[k];
            const float xnc = nc1d[k] + ncten[k] * dt;
            qiten[k] = qiten[k] + xrc * odt;
            niten[k] = niten[k] + xnc * odt;
            qcten[k] = qcten[k] - xrc * odt;
            ncten[k] = ncten[k] - xnc * odt;
            tten[k] = tten[k] + lfus2 * ocp[k] * xrc * odt;
        }
    }

    // :3688-3756, the one terminal apply.
    for (int k = kts; k <= kte; ++k) {
        t1d[k] = t1d[k] + tten[k] * dt;
        qv1d[k] = fmax_(1.E-10f, qv1d[k] + qvten[k] * dt);
        qc1d[k] = qc1d[k] + qcten[k] * dt;
        nc1d[k] = fmax_(2.0f / rho[k], nc1d[k] + ncten[k] * dt);
        nwfa1d[k] = fmax_(11.1E6f / rho[k], fmin_(9999.E6f / rho[k],
                          (nwfa1d[k] + nwfaten[k] * dt)));
        nifa1d[k] = fmax_(TF_naIN1 * 0.01f, fmin_(9999.E6f / rho[k],
                          (nifa1d[k] + nifaten[k] * dt)));

        if (qc1d[k] <= TF_R1) {
            qc1d[k] = 0.0f;
            nc1d[k] = 0.0f;
        } else {
            nu_c = min(15, tf_nint(1000.E6f / (nc1d[k] * rho[k])) + 2);
            lamc = (double)tf_powf(TF_am_r * CCG(2, nu_c) * OCG1(nu_c)
                                   * nc1d[k] / qc1d[k], TF_obmr);
            const float xDc = (float)((double)(TF_bm_r + (float)nu_c + 1.0f)
                                      / lamc);
            if (xDc < TF_D0c) {
                lamc = (double)FDIV(CCE(2, nu_c), TF_D0c);
            } else if (xDc > TF_D0r * 2.0f) {
                lamc = (double)FDIV(CCE(2, nu_c), TF_D0r * 2.0f);
            }
            nc1d[k] = (float)dmin_((double)FDIV(CCG(1, nu_c) * OCG2(nu_c)
                                                * qc1d[k], TF_am_r)
                                   * tf_pow(lamc, (double)TF_bm_r),
                                   (double)TF_Nt_c_max / (double)rho[k]);
        }

        qi1d[k] = qi1d[k] + qiten[k] * dt;
        ni1d[k] = fmax_(TF_R2 / rho[k], ni1d[k] + niten[k] * dt);
        if (qi1d[k] <= TF_R1) {
            qi1d[k] = 0.0f;
            ni1d[k] = 0.0f;
        } else {
            lami = (double)tf_powf(TF_am_i * CIG(2) * TF_oig1 * ni1d[k]
                                   / qi1d[k], TF_obmi);
            ilami = 1.0 / lami;
            const float xDi = (float)((double)(TF_bm_i + TF_mu_i + 1.0f)
                                      * ilami);
            if (xDi < 5.E-6f) {
                lami = (double)FDIV(CIE(2), 5.E-6f);
            } else if (xDi > 300.E-6f) {
                lami = (double)FDIV(CIE(2), 300.E-6f);
            }
            ni1d[k] = (float)dmin_((double)FDIV(CIG(1) * TF_oig2 * qi1d[k],
                                                TF_am_i)
                                   * tf_pow(lami, (double)TF_bm_i),
                                   499.E3 / (double)rho[k]);
        }
        qr1d[k] = qr1d[k] + qrten[k] * dt;
        nr1d[k] = fmax_(TF_R2 / rho[k], nr1d[k] + nrten[k] * dt);
        if (qr1d[k] <= TF_R1) {
            qr1d[k] = 0.0f;
            nr1d[k] = 0.0f;
        } else {
            lamr = (double)tf_powf(TF_am_r * CRG(3) * TF_org2 * nr1d[k]
                                   / qr1d[k], TF_obmr);
            mvd_r[k] = (float)((double)(3.0f + TF_mu_r + 0.672f) / lamr);
            if (mvd_r[k] > 2.5E-3f) {
                mvd_r[k] = 2.5E-3f;
            } else if (mvd_r[k] < TF_D0r * 0.75f) {
                mvd_r[k] = TF_D0r * 0.75f;
            }
            lamr = (double)((3.0f + TF_mu_r + 0.672f) / mvd_r[k]);
            nr1d[k] = (float)((double)(CRG(2) * TF_org3 * qr1d[k])
                              * tf_pow(lamr, (double)TF_bm_r)
                              / (double)TF_am_r);
        }
        qs1d[k] = qs1d[k] + qsten[k] * dt;
        if (qs1d[k] <= TF_R1) qs1d[k] = 0.0f;
        qg1d[k] = qg1d[k] + qgten[k] * dt;
        if (qg1d[k] <= TF_R1) qg1d[k] = 0.0f;
    }
    return true;
}

// ---------------------------------------------------------------------------
// calc_effectRad (:5218-5319) and the driver's clamps (:1470-1483).
// ---------------------------------------------------------------------------
__device__ void tf_effect_rad(TfCol<float> t1d, TfCol<float> p1d,
                              TfCol<float> qv1d, TfCol<float> qc1d,
                              TfCol<float> nc1d, TfCol<float> qi1d,
                              TfCol<float> ni1d, TfCol<float> qs1d,
                              TfCol<float> re_qc1d, TfCol<float> re_qi1d,
                              TfCol<float> re_qs1d, int nz)
{
    const float g_ratio[15] = {24.f, 60.f, 120.f, 210.f, 336.f, 504.f,
                               720.f, 990.f, 1320.f, 1716.f, 2184.f, 2730.f,
                               3360.f, 4080.f, 4896.f};
    for (int k = 0; k < nz; ++k) {
        const float rho = 0.622f * p1d[k] / (TF_R * t1d[k]
                                              * (qv1d[k] + 0.622f));
        const float rc = fmax_(TF_R1, qc1d[k] * rho);
        const float nc = fmax_(TF_R2, nc1d[k] * rho);
        if (!(rc <= TF_R1 || nc <= TF_R2)) {
            int inu_c;
            if (nc < 100.0f) {
                inu_c = 15;
            } else if (nc > 1.E10f) {
                inu_c = 2;
            } else {
                inu_c = min(15, tf_nint(1000.E6f / nc) + 2);
            }
            const double lamc = (double)tf_powf(nc * TF_am_r
                                                * g_ratio[inu_c - 1] / rc,
                                                TF_obmr);
            re_qc1d[k] = fmax_(2.51E-6f, fmin_((float)(0.5 * (double)(3.0f
                               + (float)inu_c) / lamc), 50.E-6f));
        }
        const float ri = fmax_(TF_R1, qi1d[k] * rho);
        const float ni = fmax_(TF_R2, ni1d[k] * rho);
        if (!(ri <= TF_R1 || ni <= TF_R2)) {
            const double lami = (double)tf_powf(TF_am_i * CIG(2) * TF_oig1
                                                * ni / ri, TF_obmi);
            re_qi1d[k] = fmax_(5.01E-6f, fmin_((float)(0.5 * (double)(3.0f
                               + TF_mu_i) / lami), 125.E-6f));
        }
        const float rs = fmax_(TF_R1, qs1d[k] * rho);
        if (!(rs <= TF_R1)) {
            const float tc0 = fmin_(-0.1f, t1d[k] - 273.15f);
            const float smob = rs * TF_oams;
            const float smo2 = smob;
            const float loga_ = tf_loga(tc0, CSE(1));
            const float a_ = tf_powf(10.0f, loga_);
            const float b_ = tf_b(tc0, CSE(1));
            const float smoc = a_ * tf_powf(smo2, b_);
            re_qs1d[k] = fmax_(10.E-6f, fmin_(0.5f * (smoc / smob),
                                              999.E-6f));
        }
    }
}

// ---------------------------------------------------------------------------
// calc_refl10cm (:5331-5651).
// ---------------------------------------------------------------------------
__device__ void tf_refl10cm(TfCol<float> qv1d, TfCol<float> qc1d,
                            TfCol<float> qr1d, TfCol<float> nr1d,
                            TfCol<float> qs1d, TfCol<float> qg1d,
                            TfCol<float> t1d, TfCol<float> p1d,
                            TfCol<float> dBZ, int nz, int first_time_step,
                            TfArena ws)
{
    const int kts = 0, kte = nz - 1;
    const float rand1 = 0.0f;
    const bool allow_wet_snow = !first_time_step;
    TfCol<float> temp = ws.F(), rr = ws.F(), rs = ws.F(), rg = ws.F();
    TfCol<float> mvd_r = ws.F(), smob = ws.F(), smoc = ws.F(), smoz = ws.F();
    TfCol<double> ilamr = ws.D(), ilamg = ws.D(), N0_r = ws.D(), N0_g = ws.D();
    TfCol<float> ze_rain = ws.F(), ze_snow = ws.F(), ze_graupel = ws.F();
    TfCol<bool> L_qr = ws.B(), L_qs = ws.B(), L_qg = ws.B();
    double lamr, lamg, N0_exp, N0_min, lam_exp;

    for (int k = kts; k <= kte; ++k) dBZ[k] = -35.0f;

    for (int k = kts; k <= kte; ++k) {
        temp[k] = t1d[k];
        const float qv = fmax_(1.E-10f, qv1d[k]);
        const float pres = p1d[k];
        const float rho = 0.622f * pres / (TF_R * temp[k] * (qv + 0.622f));
        if (qr1d[k] > TF_R1) {
            rr[k] = qr1d[k] * rho;
            const float nr = fmax_(TF_R2, nr1d[k] * rho);
            lamr = (double)tf_powf(TF_am_r * CRG(3) * TF_org2 * nr / rr[k],
                                   TF_obmr);
            ilamr[k] = 1.0 / lamr;
            N0_r[k] = (double)(nr * TF_org2) * tf_pow(lamr, (double)CRE(2));
            mvd_r[k] = (float)((double)(3.0f + TF_mu_r + 0.672f) * ilamr[k]);
            L_qr[k] = true;
        } else {
            rr[k] = TF_R1;
            mvd_r[k] = 50.E-6f;
            L_qr[k] = false;
        }
        if (qs1d[k] > TF_R2) {
            rs[k] = qs1d[k] * rho;
            L_qs[k] = true;
        } else {
            rs[k] = TF_R1;
            L_qs[k] = false;
        }
        if (qg1d[k] > TF_R2) {
            rg[k] = qg1d[k] * rho;
            L_qg[k] = true;
        } else {
            rg[k] = TF_R1;
            L_qg[k] = false;
        }
    }

    for (int k = kts; k <= kte; ++k) {
        const float tc0 = fmin_(-0.1f, temp[k] - 273.15f);
        smob[k] = rs[k] * TF_oams;
        const float smo2 = smob[k];
        float loga_ = tf_loga(tc0, CSE(1));
        float a_ = tf_powf(10.0f, loga_);
        float b_ = tf_b(tc0, CSE(1));
        smoc[k] = a_ * tf_powf(smo2, b_);
        loga_ = tf_loga(tc0, CSE(3));
        a_ = tf_powf(10.0f, loga_);
        b_ = tf_b(tc0, CSE(3));
        smoz[k] = a_ * tf_powf(smo2, b_);
    }

    N0_min = (double)TF_gonv_max;
    int k_0 = kts;
    for (int k = kte; k >= kts; --k) {
        if (temp[k] >= 270.65f) k_0 = max(k_0, k);
    }
    for (int k = kte; k >= kts; --k) {
        float xslw1;
        if (k > k_0 && L_qr[k] && mvd_r[k] > 100.E-6f) {
            xslw1 = 4.01f + tf_log10f(mvd_r[k]);
        } else {
            xslw1 = 0.01f;
        }
        const float ygra1 = 4.31f + tf_log10f(fmax_(5.E-5f, rg[k]));
        const float zans1 = (3.1f + (100.0f / (300.0f * xslw1 * ygra1
            / (10.0f / xslw1 + 1.0f + 0.25f * ygra1) + 30.0f
            + 10.0f * ygra1))) + rand1;
        N0_exp = (double)tf_powf(10.0f, zans1);
        N0_exp = dmax_((double)TF_gonv_min, dmin_(N0_exp,
                                                  (double)TF_gonv_max));
        N0_min = dmin_(N0_exp, N0_min);
        N0_exp = N0_min;
        lam_exp = tf_pow(N0_exp * (double)TF_am_g * (double)CGG(1)
                         / (double)rg[k], (double)TF_oge1);
        lamg = lam_exp * (double)tf_powf(CGG(3) * TF_ogg2 * TF_ogg1,
                                         TF_obmg);
        ilamg[k] = 1.0 / lamg;
        N0_g[k] = N0_exp / ((double)CGG(2) * lam_exp)
                * tf_pow(lamg, (double)CGE(2));
    }

    bool melti = false;
    k_0 = kts;
    for (int k = kte - 1; k >= kts; --k) {
        if ((temp[k] > 273.15f) && L_qr[k] && (L_qs[k + 1] || L_qg[k + 1])) {
            k_0 = max(k + 1, k_0);
            melti = true;
            break;
        }
    }

    const float ze_snow_c = FDIV(0.176f, 0.93f) * FDIV(6.0f, TF_PI) * FDIV(6.0f, TF_PI)
                          * FDIV(TF_am_s, 900.0f) * FDIV(TF_am_s, 900.0f);
    const float ze_graupel_c = FDIV(0.176f, 0.93f) * FDIV(6.0f, TF_PI)
                             * FDIV(6.0f, TF_PI) * FDIV(TF_am_g, 900.0f)
                             * FDIV(TF_am_g, 900.0f);
    for (int k = kts; k <= kte; ++k) {
        ze_rain[k] = 1.e-22f;
        ze_snow[k] = 1.e-22f;
        ze_graupel[k] = 1.e-22f;
        if (L_qr[k]) {
            ze_rain[k] = (float)(N0_r[k] * (double)CRG(4)
                                 * tf_pow(ilamr[k], (double)CRE(4)));
        }
        if (L_qs[k]) ze_snow[k] = ze_snow_c * smoz[k];
        if (L_qg[k]) {
            ze_graupel[k] = (float)((double)ze_graupel_c * N0_g[k]
                                    * (double)CGG(4)
                                    * tf_pow(ilamg[k], (double)CGE(4)));
        }
    }

    // :5553-5600: melting snow below the melting level (melting graupel is
    // switched off in the fork, allow_wet_graupel = .false.).
    if (melti && k_0 >= 1) {
        const tfc m_w_0 = tfc_mk(TF_m_w_0_RE, TF_m_w_0_IM);
        const tfc m_i_0 = tfc_mk(TF_m_i_0_RE, TF_m_i_0_IM);
        for (int k = k_0 - 1; k >= kts; --k) {
            if (allow_wet_snow && L_qs[k] && L_qs[k_0]) {
                const double fmelt_s = dmax_(0.05, dmin_(1.0
                    - (double)(rs[k] / rs[k_0]), 0.99));
                double eta = 0.0;
                const float oM3 = 1.0f / smoc[k];
                const float M0 = (smob[k] * oM3);
                const float Mrat = smob[k] * M0 * M0 * M0;
                const float slam1 = M0 * TF_Lam0;
                const float slam2 = M0 * TF_Lam1;
                for (int n = 1; n <= TF_nrbins; ++n) {
                    const double xd = TF_xxDs[n - 1];
                    const double x = (double)TF_am_s * (xd * xd);
                    const double CBACK = tf_rayleigh_soak_wetgraupel(
                        x, (double)TF_ocms, (double)TF_obms, fmelt_s,
                        TF_melt_outside_s, m_w_0, m_i_0);
                    const double f_d = (double)Mrat
                        * ((double)TF_Kap0 * tf_exp(-(double)slam1 * xd)
                           + (double)TF_Kap1
                             * tf_pow((double)M0 * xd, (double)TF_mu_s)
                             * tf_exp(-(double)slam2 * xd));
                    eta = eta + f_d * CBACK * TF_simpson[n - 1]
                        * TF_xdts[n - 1];
                }
                ze_snow[k] = (float)(TF_lamda4 / (TF_PI5 * TF_K_w) * eta);
            }
        }
    }

    for (int k = kte; k >= kts; --k) {
        dBZ[k] = (float)(10.0 * tf_log10((double)(ze_rain[k] + ze_snow[k]
                                                  + ze_graupel[k]) * 1.e18));
    }
}

// ---------------------------------------------------------------------------
// mp_gt_driver (:1078-1511) for one column per thread.  WOOF's arrays are
// (nz, ny, nx), so a column is ncol apart per level; ``w`` carries nz + 1
// faces and the Fortran's w1d(k) is the lower face of level k.  The grid is
// the launcher's tile of threads in flight, each walking the columns a
// whole grid apart with its own arena slots (ws_f, ws_d, ws_b: TF_WS_*
// slots of nz + 1 levels for every thread of the grid).
// ---------------------------------------------------------------------------
extern "C" __global__ void thompson_fork_column(
    const float* __restrict__ th, const float* __restrict__ pii,
    const float* __restrict__ p, const float* __restrict__ w,
    const float* __restrict__ dz,
    float* __restrict__ qv, float* __restrict__ qc, float* __restrict__ qr,
    float* __restrict__ qi, float* __restrict__ qs, float* __restrict__ qg,
    float* __restrict__ ni, float* __restrict__ nr, float* __restrict__ nc,
    float* __restrict__ nwfa, float* __restrict__ nifa,
    const float* __restrict__ nwfa2d, float* __restrict__ th_out,
    float* __restrict__ rainnc, float* __restrict__ rainncv,
    float* __restrict__ snownc, float* __restrict__ snowncv,
    float* __restrict__ graupelnc, float* __restrict__ graupelncv,
    float* __restrict__ sr, float* __restrict__ refl,
    float* __restrict__ effc, float* __restrict__ effi,
    float* __restrict__ effs,
    const double* tcg_racg, const double* tmr_racg, const double* tcr_gacr,
    const double* tnr_racg, const double* tnr_gacr,
    const double* tcs_racs1, const double* tmr_racs1,
    const double* tcs_racs2, const double* tmr_racs2,
    const double* tcr_sacr1, const double* tms_sacr1,
    const double* tcr_sacr2, const double* tms_sacr2,
    const double* tnr_racs1, const double* tnr_racs2,
    const double* tnr_sacr1, const double* tnr_sacr2,
    const double* tpi_qrfz, const double* tni_qrfz,
    const double* tpg_qrfz, const double* tnr_qrfz,
    const double* tpi_qcfz, const double* tni_qcfz,
    const double* tps_iaus, const double* tni_iaus, const double* tpi_ide,
    const double* t_Efrw, const double* t_Efsw, const double* tnc_wev,
    const double* tnccn_act,
    float* __restrict__ ws_f, double* __restrict__ ws_d,
    bool* __restrict__ ws_b,
    int nz, int ncol, float dt, int first_time_step, int do_refl,
    int singular_fall)
{
    const size_t thread = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const TfTables tb = {
        tcg_racg, tmr_racg, tcr_gacr, tnr_racg, tnr_gacr,
        tcs_racs1, tmr_racs1, tcs_racs2, tmr_racs2,
        tcr_sacr1, tms_sacr1, tcr_sacr2, tms_sacr2,
        tnr_racs1, tnr_racs2, tnr_sacr1, tnr_sacr2,
        tpi_qrfz, tni_qrfz, tpg_qrfz, tnr_qrfz, tpi_qcfz, tni_qcfz,
        tps_iaus, tni_iaus, tpi_ide, t_Efrw, t_Efsw, tnc_wev, tnccn_act};

#define AT(a, k) (a)[(size_t)(k) * (size_t)ncol + (size_t)col]
  for (size_t column = thread; column < (size_t)ncol; column += stride) {
    const int col = (int)column;
    TfArena ws = {ws_f + thread, ws_d + thread, ws_b + thread, stride,
                  (size_t)nz + 1, 0, 0, 0};
    TfCol<float> qv1d = ws.F(), qc1d = ws.F(), qi1d = ws.F(), qr1d = ws.F();
    TfCol<float> qs1d = ws.F(), qg1d = ws.F(), ni1d = ws.F(), nr1d = ws.F();
    TfCol<float> nc1d = ws.F(), nwfa1d = ws.F(), nifa1d = ws.F(), t1d = ws.F();
    TfCol<float> p1d = ws.F(), w1d = ws.F(), dz1d = ws.F(), dBZ = ws.F();
    TfCol<float> re_qc1d = ws.F(), re_qi1d = ws.F(), re_qs1d = ws.F();

    float pptrain = 0.0f, pptsnow = 0.0f, pptgraul = 0.0f, pptice = 0.0f;
    for (int k = 0; k < nz; ++k) {
        t1d[k] = AT(th, k) * AT(pii, k);
        p1d[k] = AT(p, k);
        w1d[k] = AT(w, k);
        dz1d[k] = AT(dz, k);
        qv1d[k] = AT(qv, k);
        qc1d[k] = AT(qc, k);
        qi1d[k] = AT(qi, k);
        qr1d[k] = AT(qr, k);
        qs1d[k] = AT(qs, k);
        qg1d[k] = AT(qg, k);
        ni1d[k] = AT(ni, k);
        nr1d[k] = AT(nr, k);
        nc1d[k] = AT(nc, k);
        nwfa1d[k] = AT(nwfa, k);
        nifa1d[k] = AT(nifa, k);
    }

    tf_mp_thompson(qv1d, qc1d, qi1d, qr1d, qs1d, qg1d, ni1d, nr1d, nc1d,
                   nwfa1d, nifa1d, t1d, p1d, w1d, dz1d, &pptrain, &pptsnow,
                   &pptgraul, &pptice, nz, dt, singular_fall, tb, ws);

    const float rainncv_v = pptrain + pptsnow + pptgraul + pptice;
    rainncv[col] = rainncv_v;
    rainnc[col] = rainnc[col] + pptrain + pptsnow + pptgraul + pptice;
    snowncv[col] = pptsnow + pptice;
    snownc[col] = snownc[col] + pptsnow + pptice;
    graupelncv[col] = pptgraul;
    graupelnc[col] = graupelnc[col] + pptgraul;
    sr[col] = (pptsnow + pptgraul + pptice) / (rainncv_v + 1.e-12f);

    // :1347, the fake surface source, after mp_thompson.
    nwfa1d[0] = nwfa1d[0] + nwfa2d[col] * dt;
    for (int k = 0; k < nz; ++k) {
        AT(nc, k) = nc1d[k];
        AT(nwfa, k) = nwfa1d[k];
        AT(nifa, k) = nifa1d[k];
    }
    for (int k = 0; k < nz; ++k) {
        AT(qv, k) = qv1d[k];
        AT(qc, k) = qc1d[k];
        AT(qi, k) = qi1d[k];
        AT(qr, k) = qr1d[k];
        AT(qs, k) = qs1d[k];
        AT(qg, k) = qg1d[k];
        AT(ni, k) = ni1d[k];
        AT(nr, k) = nr1d[k];
        AT(th_out, k) = t1d[k] / AT(pii, k);
        // :1440-1455: a negative vapour takes the mean of the level below
        // (already written back) and the level above (still the input).
        if (qv1d[k] < 0.0f) {
            if (k + 1 < nz - 2 && k + 1 > 2) {
                AT(qv, k) = fmax_(1.E-7f, 0.5f * (AT(qv, k - 1)
                                                  + AT(qv, k + 1)));
            } else {
                AT(qv, k) = 1.E-7f;
            }
        }
    }

    if (do_refl) {
        tf_refl10cm(qv1d, qc1d, qr1d, nr1d, qs1d, qg1d, t1d, p1d, dBZ, nz,
                    first_time_step, ws);
        for (int k = 0; k < nz; ++k) AT(refl, k) = fmax_(-35.0f, dBZ[k]);
    }

    for (int k = 0; k < nz; ++k) {
        re_qc1d[k] = 2.49E-6f;
        re_qi1d[k] = 4.99E-6f;
        re_qs1d[k] = 9.99E-6f;
    }
    tf_effect_rad(t1d, p1d, qv1d, qc1d, nc1d, qi1d, ni1d, qs1d, re_qc1d,
                  re_qi1d, re_qs1d, nz);
    // WOOF's radiation-facing micron convention on the driver's clamps.
    for (int k = 0; k < nz; ++k) {
        AT(effc, k) = fmax_(2.49E-6f, fmin_(re_qc1d[k], 50.E-6f)) * 1.0e6f;
        AT(effi, k) = fmax_(4.99E-6f, fmin_(re_qi1d[k], 125.E-6f)) * 1.0e6f;
        AT(effs, k) = fmax_(9.99E-6f, fmin_(re_qs1d[k], 999.E-6f)) * 1.0e6f;
    }
  }
#undef AT
}
