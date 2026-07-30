// GPUWM-specific Thompson MP8 -> NSSL-2 MP18 nest translation.
//
// This is not stock-WRF equivalence: WRF v4.6.1 normalizes every domain to
// the innermost microphysics selector.  The diagnosed values reproduce the
// admitted mass-only NSSL calcnfromq closure in nssl2.cu, but emit one field at
// a time so nest forcing reuses the existing full-parent scratch allocation.

struct NsslTransitionState {
    float qv, qc, qr, qi, qs, qg;
    float qh, qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh;
};

static __device__ __forceinline__
NsslTransitionState diagnose_mp8_mass_as_mp18(
    float rho, float qv, float qc, float qr, float qi, float qs, float qg)
{
    NsslTransitionState result;
    const double density_inverse = 1.0 / (double)rho;
    const float cxmin = 1.0e-8f;
    const float qxmin_init = 1.0e-8f;
    const float qxmin_cloud = 1.0e-13f;
    const float qxmin_rain = 1.0e-12f;

    float vapor = qv;
    float cloud = qc;
    float rain = qr;
    float ice = qi;
    float snow = qs;
    float graupel = qg;
    float cloud_number = 0.0f;
    float rain_number = 0.0f;
    float ice_number = 0.0f;
    float snow_number = 0.0f;
    float graupel_number = 0.0f;
    float ccn_number = (float)(
        (double)408163264.0f * (double)rho);
    float graupel_volume = 0.0f;

    if (cloud_number <= cxmin && cloud > qxmin_init) {
        const float qccn = 408163264.0f;
        const float cwmas_inverse = 327479132160.0f;
        cloud_number = fminf(qccn, cloud * cwmas_inverse) * rho;
        ccn_number -= cloud_number;
    } else if (cloud <= qxmin_cloud
               || (cloud_number <= cxmin && cloud <= qxmin_init)) {
        vapor += cloud;
        cloud_number = 0.0f;
        cloud = 0.0f;
    }

    if (ice_number <= cxmin && ice > qxmin_init) {
        const float xims = 4.7123910329460728e-10f;
        ice_number = rho * ice / xims;
    } else if (ice <= qxmin_cloud
               || (ice_number <= cxmin && ice <= qxmin_init)) {
        vapor += ice;
        ice_number = 0.0f;
        ice = 0.0f;
    }

    if (rain_number <= 0.1f * cxmin && rain > qxmin_init) {
        const float zrfac = 3.9788734806922577e-11f;
        const double lambda_inverse = pow(
            (double)rho * (double)rain * (double)zrfac, 0.25);
        const double n1 = lambda_inverse * (double)8000000.0f;
        rain_number = (float)(n1 * (double)20.0f / (double)20.0f);
    } else if (rain <= qxmin_rain
               || (rain_number <= cxmin && rain <= qxmin_init)) {
        vapor += rain;
        rain_number = 0.0f;
        rain = 0.0f;
    }

    if (snow_number <= 0.1f * cxmin && snow > qxmin_init) {
        const float zsfac = 1.0610329281846020e-9f;
        const double lambda_inverse = pow(
            (double)rho * (double)snow * (double)zsfac, 0.25);
        const double n1 = lambda_inverse * (double)3000000.0f;
        snow_number = (float)(
            n1 * (double)6.0000004768371582f / (double)20.0f);
    } else if (snow <= qxmin_cloud
               || (snow_number <= cxmin && snow <= qxmin_init)) {
        vapor += snow;
        snow_number = 0.0f;
        snow = 0.0f;
    }

    if (graupel_number <= 0.1f * cxmin && graupel > qxmin_init) {
        graupel_volume = graupel / 700.0f;
        const float zhfac = 2.2736419413860176e-9f;
        const float xgms = 9.8960235561662557e-9f;
        const double lambda_inverse = pow(
            (double)rho * (double)graupel * (double)zhfac, 0.25);
        const double intercept_number =
            lambda_inverse * (double)200000.0f;
        const double maximum_number =
            (double)rho * (double)graupel / (double)xgms;
        const double diagnosed = fmin(intercept_number, maximum_number);
        if (diagnosed > (double)cxmin) {
            graupel_number = (float)diagnosed;
        } else {
            graupel = 0.0f;
            graupel_number = 0.0f;
            graupel_volume = 0.0f;
        }
    } else if (graupel <= qxmin_rain
               || (graupel_number <= cxmin && graupel <= qxmin_init)) {
        vapor += graupel;
        graupel = 0.0f;
    }

    result.qv = vapor;
    result.qc = cloud;
    result.qr = rain;
    result.qi = ice;
    result.qs = snow;
    result.qg = graupel;
    result.qh = 0.0f;
    result.qndrop = (float)((double)cloud_number * density_inverse);
    result.qnr = (float)((double)rain_number * density_inverse);
    result.qni = (float)((double)ice_number * density_inverse);
    result.qns = (float)((double)snow_number * density_inverse);
    result.qng = (float)((double)graupel_number * density_inverse);
    result.qnh = 0.0f;
    result.qnn = (float)((double)ccn_number * density_inverse);
    result.qvolg = graupel_volume;
    result.qvolh = 0.0f;
    return result;
}

static __device__ __forceinline__
float transition_field(const NsslTransitionState& value, int field)
{
    if (field == 0) return value.qv;
    if (field == 1) return value.qc;
    if (field == 2) return value.qr;
    if (field == 3) return value.qi;
    if (field == 4) return value.qs;
    if (field == 5) return value.qg;
    if (field == 6) return value.qh;
    if (field == 7) return value.qndrop;
    if (field == 8) return value.qnr;
    if (field == 9) return value.qni;
    if (field == 10) return value.qns;
    if (field == 11) return value.qng;
    if (field == 12) return value.qnh;
    if (field == 13) return value.qnn;
    if (field == 14) return value.qvolg;
    return value.qvolh;
}

extern "C" __global__
void mp8_to_mp18_mass_diagnosed_field(
    const float* __restrict__ alt,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ qr,
    const float* __restrict__ qi,
    const float* __restrict__ qs,
    const float* __restrict__ qg,
    const float* __restrict__ mub2d,
    const float* __restrict__ mup,
    const float* __restrict__ c1h,
    const float* __restrict__ c2h,
    float* __restrict__ out,
    int field, int coupled, int nz, int ny, int nx)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nz * ny * nx) return;
    const int k = idx / (ny * nx);
    const int rem = idx - k * ny * nx;
    const int j = rem / nx;
    const int i = rem - j * nx;
    const NsslTransitionState diagnosed = diagnose_mp8_mass_as_mp18(
        1.0f / alt[idx], qv[idx], qc[idx], qr[idx], qi[idx], qs[idx], qg[idx]);
    float value = transition_field(diagnosed, field);
    if (coupled != 0) {
        const size_t q = (size_t)j * nx + i;
        const float base = __fadd_rn(__fmul_rn(c1h[k], mub2d[q]), c2h[k]);
        const float perturbation = __fmul_rn(c1h[k], mup[q]);
        const float hybrid_mass = __fadd_rn(base, perturbation);
        value = __fmul_rn(hybrid_mass, value);
    }
    out[idx] = value;
}
