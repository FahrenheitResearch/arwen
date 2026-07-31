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

// General ArWen MP edge matrix.  This entry point is never used by the
// bit-specified MP8->MP18 edge above.  Source moments are deliberately absent
// from the API: every destination moment is reconstructed from destination
// mass using the destination scheme's own entry closure.

static __device__ __forceinline__
float edge_source_mass(
    int code, int idx,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg, const float* qh)
{
    if (code == 0) return qv[idx];
    if (code == 1) return qc[idx];
    if (code == 2) return qr[idx];
    if (code == 3) return qi[idx];
    if (code == 4) return qs[idx];
    if (code == 5) return qg[idx];
    if (code == 6) return qh[idx];
    return 0.0f;
}

static __device__ __forceinline__
NsslTransitionState diagnose_edge_mass_as_mp18(
    float rho, float qv, float qc, float qr, float qi, float qs,
    float qg, float qh)
{
    NsslTransitionState result = diagnose_mp8_mass_as_mp18(
        rho, qv, qc, qr, qi, qs, qg);
    const double density_inverse = 1.0 / (double)rho;
    const float cxmin = 1.0e-8f;
    const float qxmin_init = 1.0e-8f;
    const float qxmin_rain = 1.0e-12f;
    float vapor = result.qv;
    float hail = qh;
    float hail_number = 0.0f;
    float hail_volume = 0.0f;

    // NSSL calcnfromq, module_mp_nssl_2mom.F:5428-5467.  The 900 kg/m3
    // volume default and alphahl=1 number diagnosis exactly match
    // nssl2_initial_state in nssl2.cu.
    if (hail_number <= 0.1f * cxmin && hail > qxmin_init) {
        hail_volume = hail / 900.0f;
        const float zhlfac = 8.8419414012719244e-9f;
        const double lambda_inverse = pow(
            (double)rho * (double)hail * (double)zhlfac, 0.25);
        const double n1 = lambda_inverse * (double)40000.0f;
        hail_number = (float)(
            n1 * (double)8.75f / (double)20.0f);
    } else if (hail <= qxmin_rain
               || (hail_number <= cxmin && hail <= qxmin_init)) {
        vapor += hail;
        hail = 0.0f;
    }
    result.qv = vapor;
    result.qh = hail;
    result.qnh = (float)((double)hail_number * density_inverse);
    result.qvolh = hail_volume;
    return result;
}

static __device__ __forceinline__
float thompson_edge_rain_number(float qr)
{
    if (qr <= 1.0e-12f) return 0.0f;
    const float am_r = 3.1415926536f * 1000.0f / 6.0f;
    const float lambda = 3.672f / 1.0e-3f;
    return (1.0f / 6.0f) * qr / am_r
        * lambda * lambda * lambda;
}

static __device__ __forceinline__
float thompson_edge_ice_number(float qi, float rho)
{
    if (qi <= 1.0e-12f) return 0.0f;
    const float am_i = 3.1415926536f * 890.0f / 6.0f;
    const float ice_mass = qi * rho;
    float lambda = 4.0f / 5.0e-6f;
    float number = fminf(
        999.0e3f,
        (1.0f / 6.0f) * ice_mass / am_i
            * lambda * lambda * lambda);
    lambda = cbrtf(am_i * 6.0f * number / ice_mass);
    const float diameter = 4.0f / lambda;
    if (diameter < 5.0e-6f) {
        lambda = 4.0f / 5.0e-6f;
        number = fminf(
            999.0e3f,
            (1.0f / 6.0f) * ice_mass / am_i
                * lambda * lambda * lambda);
    } else if (diameter > 300.0e-6f) {
        lambda = 4.0f / 300.0e-6f;
        number = (1.0f / 6.0f) * ice_mass / am_i
            * lambda * lambda * lambda;
    }
    return number / rho;
}

static __device__ __forceinline__
float morrison_edge_number(float q, int kind, float rimed_density)
{
    // A source-absent N=0 enters Morrison's PSD slope limiter at its lower
    // lambda bound (module_mp_morr_two_moment.F:1525-1638 and
    // morr_bound_one in morrison.cu).  Reconstruct the resulting prognostic
    // number mixing ratio directly.
    if (q < 1.0e-14f) return 0.0f;
    const float pi = 3.14159265358979323846f;
    float six_c;
    float lambda;
    if (kind == 0) {
        six_c = pi * 997.0f;
        lambda = 1.0f / 2800.0e-6f;
    } else if (kind == 1) {
        six_c = 500.0f * pi;
        lambda = 1.0f / 350.0e-6f;
    } else if (kind == 2) {
        six_c = 100.0f * pi;
        lambda = 1.0f / 2000.0e-6f;
    } else {
        six_c = rimed_density * pi;
        lambda = 1.0f / 2000.0e-6f;
    }
    return q * lambda * lambda * lambda / six_c;
}

static __device__ __forceinline__
float edge_plain_or_moment_field(
    int field, int target_mp, float rho, float morr_rhog,
    float qv, float qc, float qr, float qi, float qs, float qg, float qh)
{
    // Stable host field codes:
    // qv/qc/qr/qi/qs/qg=0..5, nr/ni/ns/ng=6..9, qh=10,
    // qndrop/qnr/qni/qns/qng/qnh/qnn/qvolg/qvolh=11..19.
    if (field == 0) return qv;
    if (field == 1) return qc;
    if (field == 2) return qr;
    if (field == 3) return qi;
    if (field == 4) return qs;
    if (field == 5) return qg;
    if (field == 10) return qh;
    if (target_mp == 8) {
        if (field == 6) return thompson_edge_rain_number(qr);
        return thompson_edge_ice_number(qi, rho);
    }
    if (target_mp == 10) {
        if (field == 6) return morrison_edge_number(qr, 0, morr_rhog);
        if (field == 7) return morrison_edge_number(qi, 1, morr_rhog);
        if (field == 8) return morrison_edge_number(qs, 2, morr_rhog);
        return morrison_edge_number(qg, 3, morr_rhog);
    }
    return 0.0f;
}

extern "C" __global__
void microphysics_edge_field(
    const float* __restrict__ alt,
    const float* __restrict__ source_qv,
    const float* __restrict__ source_qc,
    const float* __restrict__ source_qr,
    const float* __restrict__ source_qi,
    const float* __restrict__ source_qs,
    const float* __restrict__ source_qg,
    const float* __restrict__ source_qh,
    const float* __restrict__ mub2d,
    const float* __restrict__ mup,
    const float* __restrict__ c1h,
    const float* __restrict__ c2h,
    float* __restrict__ out,
    int qv_source, int qc_source, int qr_source, int qi_source,
    int qs_source, int qg_source, int qh_source,
    int field, int target_mp, float morr_rhog,
    int coupled, int nz, int ny, int nx)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= nz * ny * nx) return;
    const int k = idx / (ny * nx);
    const int rem = idx - k * ny * nx;
    const int j = rem / nx;
    const int i = rem - j * nx;
    const float rho = 1.0f / alt[idx];
    const float qv = edge_source_mass(
        qv_source, idx, source_qv, source_qc, source_qr, source_qi,
        source_qs, source_qg, source_qh);
    const float qc = edge_source_mass(
        qc_source, idx, source_qv, source_qc, source_qr, source_qi,
        source_qs, source_qg, source_qh);
    const float qr = edge_source_mass(
        qr_source, idx, source_qv, source_qc, source_qr, source_qi,
        source_qs, source_qg, source_qh);
    const float qi = edge_source_mass(
        qi_source, idx, source_qv, source_qc, source_qr, source_qi,
        source_qs, source_qg, source_qh);
    const float qs = edge_source_mass(
        qs_source, idx, source_qv, source_qc, source_qr, source_qi,
        source_qs, source_qg, source_qh);
    const float qg = edge_source_mass(
        qg_source, idx, source_qv, source_qc, source_qr, source_qi,
        source_qs, source_qg, source_qh);
    const float qh = edge_source_mass(
        qh_source, idx, source_qv, source_qc, source_qr, source_qi,
        source_qs, source_qg, source_qh);

    float value;
    if (target_mp == 18) {
        const NsslTransitionState diagnosed =
            diagnose_edge_mass_as_mp18(rho, qv, qc, qr, qi, qs, qg, qh);
        // MP18's local field order differs from the stable matrix codes.
        int nssl_field = field;
        if (field == 10) nssl_field = 6;
        else if (field >= 11) nssl_field = field - 4;
        value = transition_field(diagnosed, nssl_field);
    } else {
        value = edge_plain_or_moment_field(
            field, target_mp, rho, morr_rhog, qv, qc, qr, qi, qs, qg, qh);
    }
    if (coupled != 0) {
        const size_t q = (size_t)j * nx + i;
        const float base = __fadd_rn(__fmul_rn(c1h[k], mub2d[q]), c2h[k]);
        const float perturbation = __fmul_rn(c1h[k], mup[q]);
        const float hybrid_mass = __fadd_rn(base, perturbation);
        value = __fmul_rn(hybrid_mass, value);
    }
    out[idx] = value;
}
