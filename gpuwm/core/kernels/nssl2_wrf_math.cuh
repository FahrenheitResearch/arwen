// NSSL option 18: WRF v4.6.1's own numerical words, shared by every NSSL
// production unit (tools/nssl2_wrf461_column_oracle grades them bitwise).
//
// 1. Folded REAL parameters.  gfortran folds a PARAMETER such as
//    0.523599*(6.e-3)**3 with one correctly rounded power, then one
//    multiply; a CUDA float expression (a*a)*a rounds twice.  The values
//    below are the bits WRF's module carries (read back from the compiled
//    module by tools/nssl2_wrf461_column_oracle/probes), so the kernels use
//    WRF's numbers instead of re-deriving them.
// 2. GAMMA_DP / GAMMA_SP (module_mp_nssl_2mom.F:3388, :3774): WRF does not
//    call a library gamma.  It evaluates a six-term Lanczos series in
//    double precision, whose error (1.3e-8 relative at x = 1) is larger
//    than a REAL rounding step, so a correctly rounded gamma is a
//    different number.  The series is reproduced term for term.
// 3. The gmoi lookup (module_mp_nssl_2mom.F:799, :1525, and its users):
//    gmoi(i) = GAMMA_DP(dgam*i) with dgam = 0.01 written as a REAL literal
//    into a DOUBLE PRECISION parameter, i.e. (double)0.01f, and the
//    interpolation weight stored in a REAL before the double blend.

#define NSSL2_WRF_XVCMN   __int_as_float(0x241A8A06)  // 3.351033731e-17
#define NSSL2_WRF_CWMASN  __int_as_float(0x2916EACA)  // 3.351033779e-14
#define NSSL2_WRF_CWMASX  __int_as_float(0x3078B42B)  // 9.047790850e-10
#define NSSL2_WRF_XVRMN   __int_as_float(0x2A96EAC9)  // 2.680826752e-13
#define NSSL2_WRF_XVRMX0  __int_as_float(0x33F2DFF3)  // 1.130973928e-07
#define NSSL2_WRF_XVSMN   __int_as_float(0x2616EAC9)  // 5.235989750e-16
#define NSSL2_WRF_XVSMX   __int_as_float(0x350C8D72)  // 5.235989420e-07
#define NSSL2_WRF_XVIMN   __int_as_float(0x2616EAC9)  // 5.235989750e-16
#define NSSL2_WRF_XVIMX   __int_as_float(0x318FED03)  // 4.188792868e-09
// Graupel's minimum volume is computed at init from the variable dhmn
// (0.523599*dhmn**3 with dhmn = 0.3e-3, an inline integer power); hail's is
// the folded parameter xvhlmn.  The two differ in the last bit.
#define NSSL2_WRF_XVHMN   __int_as_float(0x2D78B42D)  // 1.413717494e-11
#define NSSL2_WRF_XVHMX0  __int_as_float(0x368C8D72)  // 4.188791536e-06
#define NSSL2_WRF_XVHLMN  __int_as_float(0x2D78B42E)  // 1.413717581e-11
#define NSSL2_WRF_XVHLMX  __int_as_float(0x380C8D72)  // 3.351033229e-05
#define NSSL2_WRF_PI      3.141592653589793f
#define NSSL2_WRF_PIINV   (1.0f / NSSL2_WRF_PI)       // 0x3EA2F983

__device__ __forceinline__ double nssl2_gamma_dp(double xx)
{
    const double cof[6] = {76.18009172947146, -86.50532032941677,
                           24.01409824083091, -1.231739572450155,
                           0.1208650973866179e-2, -0.5395239384953e-5};
    const double stp = 2.5066282746310005;
    const double x = xx;
    double y = x;
    double tmp = x + 5.5;
    tmp = __dsub_rn(__dmul_rn(x + 0.5, log(tmp)), tmp);
    double ser = 1.000000000190015;
    for (int j = 0; j < 6; ++j) {
        y = y + 1.0;
        ser = ser + __ddiv_rn(cof[j], y);
    }
    return exp(tmp + log(__ddiv_rn(__dmul_rn(stp, ser), x)));
}

__device__ __forceinline__ float nssl2_gamma_sp(float xx)
{
    return (float)nssl2_gamma_dp((double)xx);
}

// gmoi(i), i >= 1.  gmoi(0) = 1.d32 is never reached by the production
// arguments (all >= 1).
__device__ __forceinline__ double nssl2_gmoi(int i)
{
    return nssl2_gamma_dp(__dmul_rn((double)0.01f, (double)i));
}

// The lookup as WRF writes it, with REAL tmp, del and result:
//   i = Int(dgami*tmp); del = tmp - dgam*i
//   x = gmoi(i) + (gmoi(i+1) - gmoi(i))*del*dgami
__device__ __forceinline__ float nssl2_gmoi_blend(
    float tmp, double lower, double upper)
{
    const int i = (int)__dmul_rn(100.0, (double)tmp);
    const float del = (float)__dsub_rn(
        (double)tmp, __dmul_rn((double)0.01f, (double)i));
    return (float)__dadd_rn(lower, __dmul_rn(
        __dmul_rn(__dsub_rn(upper, lower), (double)del), 100.0));
}

__device__ __forceinline__ int nssl2_gmoi_index(float tmp)
{
    return (int)__dmul_rn(100.0, (double)tmp);
}

__device__ __forceinline__ float nssl2_gmoi_lookup(float tmp)
{
    const int i = nssl2_gmoi_index(tmp);
    return nssl2_gmoi_blend(tmp, nssl2_gmoi(i), nssl2_gmoi(i + 1));
}
