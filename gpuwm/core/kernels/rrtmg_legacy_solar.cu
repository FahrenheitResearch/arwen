// WRF v4.6.1 solar geometry for the legacy RRTMG radiation call.
//
// Statement-for-statement transcription of phys/module_radiation_driver.F
// radconst (:3469-3511) and calc_coszen (:3514-3541), the pair the driver
// runs once per radiation call (:1188, then :1206-1208 with
// xtime + radt*0.5 for the interval-midpoint hour angle).
//
// Numerics.  Every add, subtract, multiply and divide is an explicit IEEE
// single-rounded intrinsic (FADD and friends from glibc_flt32.cuh, which
// the loader prepends for this module), so NVRTC neither contracts a
// multiply-add into an FMA nor rewrites a constant division.  SIN, COS
// and ASIN on REAL(4) are WOOF's own float32 sinf, cosf and asinf
// (glibc_trig_flt32.cuh), not CUDA's builtins and not NumPy's vector
// loops.  tests/test_rrtmg_legacy_column_oracle.py holds the result to
// WRF v4.6.1's Fortran word for word (tools/rrtmg_legacy_column_oracle).
//
// Shared radconst body: DECLIN and SOLCON (one thread).  out[0] = DECLIN,
// out[1] = SOLCON.  The strict caller (gpuwm/core/rrtmg_legacy.py
// solar_geometry_device under GPUWM_WRF_EXACT=1) also reads the ASIN
// argument, through rls_radconst_with_argument; the two-word rls_radconst
// entry point is unchanged.
__device__ __forceinline__ void rls_radconst_values(
    float julian, float degrad, float dpd, float* out, float* asin_argument)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float obecl = FMUL(23.5f, degrad);
    float sinob = glibc_sinf(obecl);
    float sxlong = 0.0f;
    if (julian >= 80.0f) sxlong = FMUL(dpd, FSUB(julian, 80.0f));
    if (julian < 80.0f) sxlong = FMUL(dpd, FADD(julian, 285.0f));
    sxlong = FMUL(sxlong, degrad);
    float arg = FMUL(sinob, glibc_sinf(sxlong));
    float declin = glibc_asinf(arg);
    // DJUL=JULIAN*360./365.; RJUL=DJUL*DEGRAD
    float djul = FDIV(FMUL(julian, 360.0f), 365.0f);
    float rjul = FMUL(djul, degrad);
    float two = FMUL(2.0f, rjul);
    // ECCFAC=1.000110+0.034221*COS(RJUL)+0.001280*SIN(RJUL)+0.000719*
    //        COS(2*RJUL)+0.000077*SIN(2*RJUL)   (left to right)
    float eccfac = FADD(1.000110f, FMUL(0.034221f, glibc_cosf(rjul)));
    eccfac = FADD(eccfac, FMUL(0.001280f, glibc_sinf(rjul)));
    eccfac = FADD(eccfac, FMUL(0.000719f, glibc_cosf(two)));
    eccfac = FADD(eccfac, FMUL(0.000077f, glibc_sinf(two)));
    out[0] = declin;
    out[1] = FMUL(1370.0f, eccfac);
    if (asin_argument) asin_argument[0] = arg;
}

extern "C" __global__ void rls_radconst(float julian, float degrad, float dpd,
                                        float* __restrict__ out)
{
    rls_radconst_values(julian, degrad, dpd, out, nullptr);
}

extern "C" __global__ void rls_radconst_with_argument(
    float julian, float degrad, float dpd, float* __restrict__ out,
    float* __restrict__ asin_argument)
{
    rls_radconst_values(julian, degrad, dpd, out, asin_argument);
}

// calc_coszen per point: COSZEN (clamped to [-1, 1] as 4.6.1 does) and
// HRANG.  declin is radconst's value, read from the device.
static __device__ __forceinline__ void rls_coszen_point(
    long long i, const float* __restrict__ xlat,
    const float* __restrict__ xlon, float julian, float xtime, float gmt,
    float degrad, int solar_480, const float* __restrict__ declin_solcon,
    float* __restrict__ coszen, float* __restrict__ hrang_out)
{
    float declin = declin_solcon[0];
    // da=6.2831853071795862*(julian-1)/365.
    float da = FDIV(FMUL(6.2831853071795862f, FSUB(julian, solar_480 ? 0.5f : 1.0f)), 365.0f);
    float two_da = FMUL(2.0f, da);
    float eot = FADD(0.000075f, FMUL(0.001868f, glibc_cosf(da)));
    eot = FSUB(eot, FMUL(0.032077f, glibc_sinf(da)));
    eot = FSUB(eot, FMUL(0.014615f, glibc_cosf(two_da)));
    eot = FSUB(eot, FMUL(solar_480 ? 0.040849f : 0.04089f, glibc_sinf(two_da)));
    eot = FMUL(eot, 229.18f);
    float xt24 = FADD(fmodf(xtime, 1440.0f), eot);
    float tloctm = FADD(FADD(gmt, FDIV(xt24, 60.0f)), FDIV(xlon[i], 15.0f));
    float hrang = FMUL(FMUL(15.0f, FSUB(tloctm, 12.0f)), degrad);
    float xxlat = FMUL(xlat[i], degrad);
    float c = FADD(FMUL(glibc_sinf(xxlat), glibc_sinf(declin)),
                   FMUL(FMUL(glibc_cosf(xxlat), glibc_cosf(declin)),
                        glibc_cosf(hrang)));
    // coszen = min(max(coszen, -1.0), 1.0)
    c = fminf(fmaxf(c, -1.0f), 1.0f);
    coszen[i] = c;
    hrang_out[i] = hrang;
}

extern "C" __global__ void rls_calc_coszen(
    long long n, const float* __restrict__ xlat,
    const float* __restrict__ xlon, float julian, float xtime, float gmt,
    float degrad, int solar_480, const float* __restrict__ declin_solcon,
    float* __restrict__ coszen, float* __restrict__ hrang_out)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    rls_coszen_point(i, xlat, xlon, julian, xtime, gmt, degrad, solar_480,
                     declin_solcon, coszen, hrang_out);
}

extern "C" __global__ void rls_calc_coszen_hrang(
    long long n, const float* __restrict__ xlat,
    const float* __restrict__ xlon, float julian, float xtime, float gmt,
    float degrad, int solar_480, const float* __restrict__ declin_solcon,
    float* __restrict__ coszen, float* __restrict__ hrang_out)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    rls_coszen_point(i, xlat, xlon, julian, xtime, gmt, degrad, solar_480,
                     declin_solcon, coszen, hrang_out);
}
