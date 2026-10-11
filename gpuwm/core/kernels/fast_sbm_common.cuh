// WRF mp_physics = 30 (fast spectral bin microphysics, FSBM-2): shared
// prelude of the WOOF device port.
//
// Transcribed from WRF phys/module_mp_fast_sbm.F, LF SHA-256
// fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e
// (identical in WRF 4.6.1, 4.7.1 and 4.8.0).  WRF is public domain; the
// notice travels in gpuwm/data/fast_sbm/LICENSE-WRF.txt.  Line numbers
// ":NNNN" in these files refer to that source.
//
// ONE SOURCE, TWO COMPILERS.  The fast_sbm_*.cuh headers compile under
// NVRTC (the forecast's device code, fast_sbm.cu) and under a host C++
// compiler (tools/fast_sbm_host/, the float64 verification build that is
// graded against WRF's own Fortran).  Every routine is a template on R,
// the kind of WRF's default REAL: R = float is WRF as it runs (the O32
// oracle build), R = double is WRF's source promoted with -freal-4-real-8
// (the O64 build).  Variables WRF declares REAL(KIND=8) or DOUBLE
// PRECISION are double in both.  Literals WRF writes as default REAL go
// through RL(), so a float build sees the float literal and promotes it
// exactly as gfortran does (0.622 in a double expression is
// (double)0.622f in the O32 build).
//
// Arithmetic order is Fortran's: left to right within a precedence level,
// every intermediate in the precision Fortran gives it.  The device unit is
// compiled with --fmad=false so no multiply-add is contracted.
//
// libm.  Device code calls WOOF's own float32 libm (gfk_exp, gfk_log,
// gfk_pow) and WOOF's float64 words (glibc_exp, glibc_log, glibc_pow); the
// host build calls the platform libm, which is what the Fortran oracle
// calls.  Integer powers follow gfortran -O0: x**2 is x*x, any other
// integer power is the square-and-multiply of libgcc's __powi (powi_gcc
// below), and a REAL exponent is a pow call even when it is 2.0.

#ifndef WOOF_FAST_SBM_COMMON_CUH
#define WOOF_FAST_SBM_COMMON_CUH

#if defined(__CUDACC_RTC__) || defined(__CUDACC__)
#define FSBM_DEVICE 1
#define FHD __device__
#define FHD_NOINLINE __device__ __noinline__
#else
#define FSBM_DEVICE 0
#include <cmath>
#include <cstdint>
#define FHD inline
#define FHD_NOINLINE
#endif

// ---------------------------------------------------------------------------
// Scheme parameters (:3880-3902, 33-bin line)
// ---------------------------------------------------------------------------
#define FSBM_NKR 33
#define FSBM_NKR_AERO 43
#define FSBM_ICEMAX 3
#define FSBM_JMAX 33
#define FSBM_JBREAK 18
#define FSBM_KRMIN_BREAKUP 31
#define FSBM_KRDROP 15
#define FSBM_KRICE 18
#define FSBM_KRFREEZ 21
#define FSBM_KR_ICEMPL 9
#define FSBM_KP_FLUX_MAX 44
#define FSBM_KR_SNOW_MIN 34
#define FSBM_KR_SNOW_MAX 35
#define FSBM_NCHEM 142              // drops 33, snow 33, dense ice 33, aerosol 43
#define FSBM_OFF_DROP 0
#define FSBM_OFF_SNOW 33
#define FSBM_OFF_DENSE 66
#define FSBM_OFF_AERO 99

// Error codes: the WRF source line of the wrf_error_fatal / STOP reached.
// A device thread that reaches one stores it and stops its point.
#define FSBM_OK 0

// ---------------------------------------------------------------------------
// Behaviour switches (the defect catalogue, gpuwm/core/fast_sbm/catalogue.py)
// ---------------------------------------------------------------------------
struct FsbmFlags {
  int hail;                    // dense ice: 1 hail (WRF default), 0 graupel
  int rainnc_own_column;       // D1
  int rainnc_bottom_outflow;   // D2
  int nucleation_water_sink;   // D3
  int dtcond_carry;            // D5 (host driver only; device resets)
  int breakup_retry_cap;       // D7
  int cloud_base_first_level;  // D9
  int fall_speed_own_bin;      // D10 (selects the ECOALMASSM table)
  int snow_clip_qi_plus_qs;    // D11
  int melt_exponential;        // D12
  int fusion_heat_334;         // D13
  int homfrz_any_saturation;   // D14
  int dsupice_w_own_point;     // D15
  int dsupice_wind_own_point;  // D17
  int supsat_rederived;        // S1, S2
  int sedimentation_cell_depth; // D19: physical cell volume on a stretched grid
  int collision_mass_conserved; // D20: conservative collision/breakup projection
};

// ---------------------------------------------------------------------------
// Literals, conversions, integer powers
// ---------------------------------------------------------------------------
template <class R> struct fsbm_is_double { static const bool value = false; };
template <> struct fsbm_is_double<double> { static const bool value = true; };

template <class R> FHD R fsbm_rl(double d, float f) {
  return fsbm_is_double<R>::value ? (R)d : (R)f;
}
// A default-REAL literal of the Fortran source, in the build's REAL kind.
#define RL(x) fsbm_rl<R>((double)(x), x##f)

// libgcc __powidf2 / __powisf2: what gfortran -O0 calls for x**n with an
// integer n other than 2 (x**2 it writes x*x).
template <class T> FHD T powi_gcc(T x, int m) {
  unsigned int n = m < 0 ? (unsigned int)(-(long long)m) : (unsigned int)m;
  T y = (n % 2) ? x : (T)1;
  while (n >>= 1) {
    x = x * x;
    if (n % 2) y = y * x;
  }
  return m < 0 ? (T)1 / y : y;
}

template <class T> FHD T fmax2(T a, T b) { return a > b ? a : b; }
template <class T> FHD T fmin2(T a, T b) { return a < b ? a : b; }
template <class T> FHD T fabs_(T a) { return a < (T)0 ? -a : a; }
template <class T> FHD bool fisnan(T a) { return a != a; }

// ---------------------------------------------------------------------------
// libm by precision
// ---------------------------------------------------------------------------
#if FSBM_DEVICE
FHD float fx_exp(float x) { return gfk_exp(x); }
FHD float fx_log(float x) { return gfk_log(x); }
FHD float fx_pow(float x, float y) { return gfk_pow(x, y); }
FHD float fx_sqrt(float x) { return sqrtf(x); }
FHD double fx_exp(double x) { return glibc_exp(x); }
FHD double fx_log(double x) { return glibc_log(x); }
FHD double fx_pow(double x, double y) { return glibc_pow(x, y); }
FHD double fx_sqrt(double x) { return sqrt(x); }
#else
inline float fx_exp(float x) { return ::expf(x); }
inline float fx_log(float x) { return ::logf(x); }
inline float fx_pow(float x, float y) { return ::powf(x, y); }
inline float fx_sqrt(float x) { return ::sqrtf(x); }
inline double fx_exp(double x) { return ::exp(x); }
inline double fx_log(double x) { return ::log(x); }
inline double fx_pow(double x, double y) { return ::pow(x, y); }
inline double fx_sqrt(double x) { return ::sqrt(x); }
#endif

// ---------------------------------------------------------------------------
// Tables.  Arrays in Fortran order, zero-based offsets: A(i,j) of an
// (n1, n2) array is a[(i-1) + (j-1)*n1].  REAL tables are R, REAL(KIND=8)
// module arrays double, as FAST_HUCMINIT declares them (:3936-3981).
// ---------------------------------------------------------------------------
template <class R> struct FsbmTab {
  // As read from SBM_input_33 (:5410-6039)
  const R *xl, *xi, *xs, *xg, *xh;          // xi (33,3)
  const R *vr1, *vr2, *vr3, *vr4, *vr5;     // vr2 (33,3)
  const R *rlec, *riec, *rsec, *rgec, *rhec;  // riec (33,3)
  const R *ro1bl, *ro2bl, *ro3bl, *ro4bl, *ro5bl;  // ro2bl (33,3)
  const R *ywll;   // (33,33,3): 1000, 750, 500 hPa
  const R *ywls;   // (33,33,3): 300, 500, 750 hPa
  const R *ywlg;   // (33,33,3): 300, 500, 750 hPa
  const R *ywlh;   // (33,33,3): 300, 500, 750 hPa
  const R *ywss;   // (33,33,3): 300, 500, 750 hPa
  const R *pkij;   // (18,18,18)
  const R *qkj;    // (18,18)
  // Derived by FAST_HUCMINIT (:6074-6224)
  const double *xl_mg, *xs_mg, *xg_mg, *xh_mg;  // (33)
  const int *ima;                               // (33,33)
  const double *chucm;                          // (33,33)
  const double *ecoalmassm;                     // (33,33), this build's D10 lookup
  const double *prob, *gain_var_new;            // (33), (33,33)
  const R *dropradii;                           // (33)
  const double *fccnr_con, *fccnr_mar;          // (43)
  const R *rccn;                                // (43)
  int ikr_spon_break;
};

#define T2(a, i, j, n1) ((a)[((i) - 1) + ((j) - 1) * (n1)])
#define TK(a, i, j, p) ((a)[((i) - 1) + ((j) - 1) * FSBM_NKR + (p) * FSBM_NKR * FSBM_NKR])

#endif  // WOOF_FAST_SBM_COMMON_CUH
