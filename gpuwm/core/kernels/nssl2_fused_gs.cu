// Fused WRF v4.6.1 NSSL option-18 GS implementation.
//
// This source is deliberately independent of nssl2.cu.  The public kernels in
// that file are overlapping final-state oracle slices and must never be
// chained or interpreted as additive rates.  Every rate below is diagnosed
// from the one state loaded at kernel entry, WRF's shared limiters rescale the
// named rates, and the prognostics are advanced once at the end.

namespace {

constexpr int QV = 0;
constexpr int QC = 1;
constexpr int QR = 2;
constexpr int QI = 3;
constexpr int QS = 4;
constexpr int QG = 5;
constexpr int QH = 6;
constexpr int NC = 7;
constexpr int NR = 8;
constexpr int NI = 9;
constexpr int NS = 10;
constexpr int NG = 11;
constexpr int NH = 12;
constexpr int NN = 13;
constexpr int VG = 14;
constexpr int VH = 15;

constexpr float PI = 3.14159265358979323846f;
constexpr float CXMIN = 1.0e-8f;
constexpr float QC_MIN = 1.0e-13f;
constexpr float QR_MIN = 1.0e-12f;
constexpr float QI_MIN = 1.0e-13f;
constexpr float QS_MIN = 1.0e-13f;
constexpr float QD_MIN = 1.0e-12f;

constexpr float CLOUD_VMIN =
    0.523599f * (4.0e-6f * 4.0e-6f * 4.0e-6f);
constexpr float CLOUD_VMAX =
    0.523599f * (120.0e-6f * 120.0e-6f * 120.0e-6f);
constexpr float RAIN_VMIN =
    0.523599f * (80.0e-6f * 80.0e-6f * 80.0e-6f);
constexpr float RAIN_VMAX =
    (0.523599f * (6.0e-3f * 6.0e-3f * 6.0e-3f)) / (64.0f / 6.0f);
constexpr float ICE_MMIN = 6.88e-13f;
constexpr float ICE_MMAX = 1.0e-8f;
// The generic post-GS two-moment limiter uses setpar-derived ice volume
// bounds (10 um--2 mm diameter), which intentionally differ from the
// narrower initial setvtz crystal-mass bounds above.
constexpr float ICE_FINAL_MMIN =
    900.0f * 0.523599f * (10.0e-6f * 10.0e-6f * 10.0e-6f);
constexpr float ICE_FINAL_MMAX =
    900.0f * 0.523599f * (2.0e-3f * 2.0e-3f * 2.0e-3f);
constexpr float SNOW_VMIN =
    0.523599f * (0.01e-3f * 0.01e-3f * 0.01e-3f);
constexpr float SNOW_VMAX =
    0.523599f * (10.0e-3f * 10.0e-3f * 10.0e-3f);
constexpr float GRAUPEL_VMIN =
    0.523599f * (0.30e-3f * 0.30e-3f * 0.30e-3f);
constexpr float GRAUPEL_INITIAL_VMAX =
    0.523599f * (20.0e-3f * 20.0e-3f * 20.0e-3f);
constexpr float GRAUPEL_VMAX =
    (0.523599f * (20.0e-3f * 20.0e-3f * 20.0e-3f)) / (64.0f / 6.0f);
constexpr float HAIL_VMIN =
    0.523599f * (0.30e-3f * 0.30e-3f * 0.30e-3f);
constexpr float HAIL_INITIAL_VMAX =
    0.523599f * (40.0e-3f * 40.0e-3f * 40.0e-3f);
constexpr float HAIL_VMAX =
    (0.523599f * (40.0e-3f * 40.0e-3f * 40.0e-3f)) / (125.0f / 24.0f);

struct State {
    float qv, qc, qr, qi, qs, qg, qh;
    float nc, nr, ni, ns, ng, nh, nn;
    float vg, vh;
    float theta, temperature;
};

// Named WRF rates.  Keeping contributors separate is required: the cloud,
// rain, snow, and hail limiters rescale selected contributors and downstream
// vapor/heat/volume/number sums must then be rebuilt from the scaled values.
struct Rates {
    // Warm-rain mass and number.
    float qrcnw, crcnw, cautn;
    float qracw, cracw;
    float qrcev, crcev;
    float cracr;
    float qwcnr, qwshw, cwshw;

    // Cold-process members are populated by the source-ordered diagnosis
    // sections below this warm foundation.  Their explicit names prevent an
    // aggregate-only implementation from bypassing the WRF resums.
    float qiacw, ciacw;
    float qsacw, csacw;
    float qhacw, chacw, vhacw, qhacwmlr;
    float graupel_cloud_rime_density;
    float qhlacw, chlacw, vhlacw, qhlacwmlr;
    float qwfrz, cwfrz, qwfrzp, cwfrzp, qwfrzc, cwfrzc;
    float qwctfz, cwctfzp, qwctfzc, cwctfzc;
    float qiihr;
    float qiacr, ciacr, qiacrf, ciacrf, qiacrs, ciacrs, viacrf;
    float qrfrz, crfrz, qrfrzf, crfrzf, qrfrzs, crfrzs, vrfrzf;
    float qsacr, csacr;
    float qhacr, chacr, vhacr, qhacrmlr;
    float qhlacr, chlacr, vhlacr, qhlacrmlr;
    float qidpv, cidpv, qisbv, cisbv, qiint, ciint;
    float qsdpv, csdpv, qssbv, cssbv, qscev, cscev;
    float qhdpv, chdpv, qhsbv, chsbv, qhcev, chcev;
    float qhldpv, chldpv, qhlsbv, chlsbv, qhlcev, chlcev;
    float qscni, cscni, qscnvi, cscnvi;
    float qsaci, csaci, qraci, craci, qracif, csacs;
    float qhaci, chaci, qhaci0, chaci0;
    float qhlaci, chlaci, qhlaci0, chlaci0;
    float qhacs, chacs, qhacs0, chacs0;
    float qhlacs, chlacs, qhlacs0, chlacs0;
    float qhcni, chcni, chcnih, vhcni;
    float qhcns, chcns, chcnsh, vhcns;
    float qscnh, cscnh, vscnh;
    float qhlcnh, chlcnh, chlcnhhl, vhlcnh, vhlcnhl;
    float qhcnhl, chcnhl, vhcnhl;
    float qimlr, cimlr, qsmlr, csmlr, csmlrr;
    float qhmlr, chmlr, chmlrr, vhmlr;
    float qhlmlr, chlmlr, chlmlrr, vhlmlr;
    float qsshr, csshr, qhshr, chshr, chshrr;
    float qhlshr, chlshr, chlshrr, qrshr, crshr;
    float vhshdr, vhlshdr, vhsoak, vhlsoak;
    float wetgrowth_g, wetgrowth_h, wetsurface_g, wetsurface_h;
    float qsmul, csmul, qhmul1, chmul1, qhlmul1, chlmul1;
    float qsplinter, csplinter, qsplinter2, csplinter2;
};

struct Aggregates {
    float pqv_i, pqv_d;
    float pqc_i, pqc_d;
    float pqr_i, pqr_d;
    float pqi_i, pqi_d;
    float pqs_i, pqs_d;
    float pqg_i, pqg_d;
    float pqh_i, pqh_d;
    float pnc_i, pnc_d;
    float pnr_i, pnr_d;
    float pni_i, pni_d;
    float pns_i, pns_d;
    float png_i, png_d;
    float pnh_i, pnh_d;
    float pvg_i, pvg_d;
    float pvh_i, pvh_d;
};

struct ParticleProperties {
    float snow_density;
    float graupel_density;
    float hail_density;
};

__device__ __forceinline__ float latent_vapor(float temperature) {
    const float bounded = fminf(313.15f, fmaxf(233.15f, temperature));
    return 2500837.367f * powf(
        273.15f / bounded, 0.167f + 3.67e-4f * bounded);
}

__device__ __forceinline__ float latent_fusion(float temperature) {
    const float bounded = fminf(273.15f, fmaxf(223.15f, temperature));
    const float celsius = bounded - 273.15f;
    return 333690.6098f + 2030.61425f * celsius
        - 10.46708312f * celsius * celsius;
}

__device__ __forceinline__ float ice_saturation_mixing_ratio(
    float temperature, float pressure)
{
    int table_index = (int)((temperature - 163.15f) / 0.002f + 1.5f);
    table_index = min(1000001, max(1, table_index));
    // WRF fills TABQVS with separate single-precision multiply and add
    // operations.  NVRTC otherwise contracts this expression to an FMA,
    // moving some table temperatures by one ULP and amplifying the error in
    // the primary-ice target near saturation.
    const float table_temperature = __fadd_rn(
        163.15f, __fmul_rn((float)(table_index - 1), 0.002f));
    return (380.0f / pressure) * expf(
        21.87455f * (table_temperature - 273.15f)
        / (table_temperature - 7.66f));
}

__device__ __forceinline__ float liquid_saturation_mixing_ratio(
    float temperature, float pressure)
{
    int table_index = (int)((temperature - 163.15f) / 0.002f + 1.5f);
    table_index = min(1000001, max(1, table_index));
    const float table_temperature = __fadd_rn(
        163.15f, __fmul_rn((float)(table_index - 1), 0.002f));
    return (380.0f / pressure) * expf(
        17.2693882f * (table_temperature - 273.15f)
        / (table_temperature - 35.86f));
}

__device__ __forceinline__ float primary_ice_target(
    float vapor, float temperature, float rho, float pressure)
{
    // WRF 2834-2856, default icenucopt=1.  t7 is diagnosed by the outer
    // scheme from the unchanged column before GS and is not a Registry field.
    const float water_saturation =
        liquid_saturation_mixing_ratio(temperature, pressure);
    const float ice_saturation =
        ice_saturation_mixing_ratio(temperature, pressure);
    const float ice_relative = fminf(water_saturation, fmaxf(vapor, 0.0f))
        / ice_saturation;
    if (!(ice_relative > 1.0f) || !(temperature <= 268.15f)) return 0.0f;
    return (rho / 1.225f) * 1.0e3f * expf(fminf(
        57.0f, 12.96f * (ice_relative - 1.0f) - 0.639f));
}

__device__ __forceinline__ void dense_fall_coefficients(
    float density, float* coefficient, float* exponent)
{
    const float density_table[9] = {
        50.0f, 150.0f, 250.0f, 350.0f, 450.0f,
        550.0f, 650.0f, 750.0f, 850.0f};
    const float coefficient_table[9] = {
        62.923f, 94.122f, 114.74f, 131.21f, 145.26f,
        157.71f, 168.98f, 179.36f, 189.02f};
    const float exponent_table[9] = {
        0.67819f, 0.63789f, 0.62197f, 0.61240f, 0.60572f,
        0.60066f, 0.59663f, 0.59330f, 0.59048f};
    int table = (int)((density - 50.0f) / 100.0f) + 1;
    table = min(9, max(1, table)) - 1;
    const float fraction = fmaxf(
        0.0f, 0.01f * (density - density_table[table]));
    *coefficient = coefficient_table[table];
    *exponent = exponent_table[table];
    if (table < 8) {
        *coefficient += fraction
            * (coefficient_table[table + 1] - *coefficient);
        *exponent += fraction
            * (exponent_table[table + 1] - *exponent);
    }
}

// WRF initializes gmoi on a 0.01 grid with its double-precision Lanczos
// GAMMA_DP routine, then linearly interpolates the table into single-precision
// temporaries.  Dense-particle fall speeds use that interpolated value rather
// than the platform tgammaf result.  The distinction is small (about 5--32
// ppm for the admitted exponents) but is amplified by cloud-number collection.
__device__ __forceinline__ double wrf_gamma_dp(double x)
{
    double y = x;
    const double tmp0 = x + 5.5;
    const double tmp = (x + 0.5) * log(tmp0) - tmp0;
    double series = 1.000000000190015;
    y += 1.0;
    series += 76.18009172947146 / y;
    y += 1.0;
    series += -86.50532032941677 / y;
    y += 1.0;
    series += 24.01409824083091 / y;
    y += 1.0;
    series += -1.231739572450155 / y;
    y += 1.0;
    series += 0.1208650973866179e-2 / y;
    y += 1.0;
    series += -0.5395239384953e-5 / y;
    return exp(tmp + log(2.5066282746310005 * series / x));
}

__device__ __forceinline__ float wrf_gamma_lookup(float argument)
{
    const int index = (int)(100.0 * (double)argument);
    const float delta = (float)(
        (double)argument - 0.01 * (double)index);
    const double lower = wrf_gamma_dp(0.01 * (double)index);
    const double upper = wrf_gamma_dp(0.01 * (double)(index + 1));
    return (float)(lower + (upper - lower) * (double)delta * 100.0);
}

__device__ __forceinline__ float rain_tail_number_node(int bin) {
    return (float)exp(-0.25 * (double)bin);
}

__device__ __forceinline__ float rain_tail_mass_node(int bin) {
    const double x = 0.25 * (double)bin;
    const double x2 = x * x;
    return (float)(exp(-x) * (1.0 + x + 0.5 * x2 + x2 * x / 6.0));
}

__device__ __forceinline__ void rain_tail_fractions(
    float ratio, float* number_fraction, float* mass_fraction)
{
    ratio = fminf(100.0f, fmaxf(0.0f, ratio));
    const int bin = min(400, (int)(ratio * 4.0f));
    const int next_bin = min(400, bin + 1);
    const float weight = (ratio - (float)bin * 0.25f) * 4.0f;
    const float number_low = rain_tail_number_node(bin);
    const float mass_low = rain_tail_mass_node(bin);
    *number_fraction = number_low + weight
        * (rain_tail_number_node(next_bin) - number_low);
    *mass_fraction = mass_low + weight
        * (rain_tail_mass_node(next_bin) - mass_low);
}

__device__ __forceinline__ void bound_liquid(
    float q, float rho, float vmin, float vmax, float* number)
{
    if (q <= 0.0f) {
        *number = 0.0f;
    } else if (*number > CXMIN) {
        float volume = rho * q / (1000.0f * *number);
        if (volume < vmin || volume > vmax) {
            volume = fminf(vmax, fmaxf(vmin, volume));
            *number = rho * q / (1000.0f * volume);
        }
    }
}

__device__ __forceinline__ void bound_ice(State* s, float rho) {
    if (s->qi <= 0.0f) {
        s->ni = 0.0f;
    } else if (s->ni > CXMIN) {
        const float mass = rho * s->qi / s->ni;
        if (mass < ICE_FINAL_MMIN || mass > ICE_FINAL_MMAX) {
            s->ni = rho * s->qi /
                fminf(ICE_FINAL_MMAX, fmaxf(ICE_FINAL_MMIN, mass));
        }
    }
}

__device__ __forceinline__ void bound_snow(
    State* s, float rho, float density)
{
    if (s->qs <= 0.0f) {
        s->ns = 0.0f;
    } else if (s->ns > CXMIN) {
        float volume = rho * s->qs / (density * s->ns);
        const float maximum = SNOW_VMAX * fmaxf(
            1.0f, 100.0f / fminf(100.0f, density));
        if (volume < SNOW_VMIN || volume > maximum) {
            volume = fminf(maximum, fmaxf(SNOW_VMIN, volume));
            s->ns = rho * s->qs / (density * volume);
        }
    }
}

__device__ __forceinline__ void bound_dense_number(
    float q, float* number, float rho, float density,
    float vmin, float vmax)
{
    if (q <= 0.0f) {
        *number = 0.0f;
        return;
    }
    if (*number > CXMIN) {
        float mean_volume = rho * q / (density * *number);
        if (mean_volume < vmin || mean_volume > vmax) {
            mean_volume = fminf(vmax, fmaxf(vmin, mean_volume));
            *number = rho * q / (density * mean_volume);
        }
    }
}

__device__ __forceinline__ ParticleProperties normalize_state(
    State* s, float rho)
{
    // The admitted option-18 configuration overrides the module's raw
    // rho_qhl=800 declaration with nssl_params(9)=900 at initialization.
    ParticleProperties p{100.0f, 500.0f, 900.0f};

    // WRF 13766-13927 gather/cleanup precedes setvtz.  The exact-zero test is
    // intentionally distinct from the later CXMIN gates: a tiny active
    // rain/snow/graupel/hail mass with a missing number is made available to
    // diagnostic vapor, but that transfer is not part of final qv scatter.
    if (s->qr > QR_MIN) {
        if (s->nr == 0.0f && s->qr < 3.0f * QR_MIN) {
            s->qv += s->qr;
            s->qr = 0.0f;
        } else {
            s->nr = fmaxf(1.0e-9f, s->nr);
        }
    }
    if (s->qs > QS_MIN) {
        if (s->ns == 0.0f && s->qs < 3.0f * QS_MIN) {
            s->qv += s->qs;
            s->qs = 0.0f;
        } else {
            s->ns = fmaxf(1.0e-9f, s->ns);
        }
    }
    if (s->qg > QD_MIN) {
        if (s->ng == 0.0f && s->qg < 3.0f * QD_MIN) {
            s->qv += s->qg;
            s->qg = 0.0f;
        } else {
            s->ng = fmaxf(1.0e-9f, s->ng);
        }
    }
    if (s->qh > QD_MIN) {
        if (s->nh == 0.0f && s->qh < 3.0f * QD_MIN) {
            s->qv += s->qh;
            s->qh = 0.0f;
        } else {
            s->nh = fmaxf(1.0e-9f, s->nh);
        }
    }

    // Initial setvtz differs from the post-GS moment limiter.  Cloud retains
    // an existing number while clamping only its diagnostic mass/volume.
    if (s->qc > QC_MIN) {
        if (!(s->nc > CXMIN)) {
            s->nc = fmaxf(
                CXMIN, rho * s->qc / (1000.0f * CLOUD_VMAX));
        }
        // WRF setvtz (ipconc>=2) clamps the diagnostic cloud mass/volume
        // for an existing number moment but deliberately retains cx itself.
        // diagnose_warm repeats that diagnostic clamp locally; only the
        // post-GS limiter is allowed to rewrite an existing cloud number.
    } else {
        // GS gather clears cloud number whenever qc is not active.
        s->nc = 0.0f;
    }
    if (s->qr > QR_MIN) {
        float volume = rho * s->qr / (1000.0f * fmaxf(1.0e-11f, s->nr));
        if (volume < RAIN_VMIN || volume > RAIN_VMAX) {
            volume = fminf(RAIN_VMAX, fmaxf(RAIN_VMIN, volume));
            s->nr = rho * s->qr / (1000.0f * volume);
        }
        // Inactive rain retains the nonnegative number loaded by GS gather.
    }
    if (s->qi > QI_MIN) {
        s->ni = fmaxf(s->ni, rho * s->qi / ICE_MMAX);
        s->ni = fminf(s->ni, rho * s->qi / ICE_MMIN);
    } else {
        // GS gather clears crystal number whenever qi is not active.
        s->ni = 0.0f;
    }
    if (s->qs > QS_MIN) {
        float volume = rho * s->qs
            / (100.0f * fmaxf(1.0e-9f, s->ns));
        if (volume < SNOW_VMIN) {
            volume = SNOW_VMIN;
            s->ns = rho * s->qs / (100.0f * volume);
        }
        if (volume > SNOW_VMAX) {
            volume = fminf(SNOW_VMAX, fmaxf(SNOW_VMIN, volume));
            const float mass = 0.106214f * powf(volume, 2.0f / 3.0f);
            s->ns = rho * s->qs / mass;
            p.snow_density = 0.0346159f * sqrtf(s->ns / (s->qs * rho));
        }
    } else {
        // Snow is the setvtz exception: its number is cleared throughout the
        // non-active mixing-ratio branch rather than retained.
        s->ns = 0.0f;
    }
    if (s->qg > QD_MIN) {
        if (s->vg > 0.0f) {
            p.graupel_density = fminf(
                900.0f, fmaxf(170.0f, rho * s->qg / s->vg));
        }
        s->vg = rho * s->qg / p.graupel_density;
    }
    if (s->qh > QD_MIN) {
        if (s->vh > 0.0f) {
            p.hail_density = fminf(
                900.0f, fmaxf(500.0f, rho * s->qh / s->vh));
        }
        s->vh = rho * s->qh / p.hail_density;
    }
    if (s->qg > QD_MIN) {
        float mean_volume = rho * s->qg
            / (p.graupel_density * fmaxf(1.0e-9f, s->ng));
        if (mean_volume < GRAUPEL_VMIN
                || mean_volume > GRAUPEL_INITIAL_VMAX) {
            mean_volume = fminf(
                GRAUPEL_INITIAL_VMAX,
                fmaxf(GRAUPEL_VMIN, mean_volume));
            s->ng = rho * s->qg / (p.graupel_density * mean_volume);
        }
        // Inactive graupel retains its gathered number moment.
    }
    if (s->qh > QD_MIN) {
        float mean_volume = rho * s->qh
            / (p.hail_density * fmaxf(1.0e-9f, s->nh));
        if (mean_volume < HAIL_VMIN || mean_volume > HAIL_INITIAL_VMAX) {
            mean_volume = fminf(
                HAIL_INITIAL_VMAX, fmaxf(HAIL_VMIN, mean_volume));
            s->nh = rho * s->qh / (p.hail_density * mean_volume);
        }
    } else {
        // GS gather clears hail number whenever qh is not active.
        s->nh = 0.0f;
    }
    return p;
}

__device__ __forceinline__ void final_bounds(
    State* s, float rho, const ParticleProperties& p)
{
    // WRF can leave a negative FP32 donor-limit cancellation residue in the
    // mass work arrays.  GPUWM's production Registry and downstream physics
    // require exact nonnegativity, matching the final bounds already applied
    // to every number and dense-volume moment below.  Positive mass values
    // are preserved bit-for-bit.
    bound_liquid(s->qc, rho, CLOUD_VMIN, CLOUD_VMAX, &s->nc);
    bound_liquid(s->qr, rho, RAIN_VMIN, RAIN_VMAX, &s->nr);
    bound_ice(s, rho);
    bound_snow(s, rho, p.snow_density);
    bound_dense_number(
        s->qg, &s->ng, rho, p.graupel_density,
        GRAUPEL_VMIN, GRAUPEL_VMAX);
    bound_dense_number(
        s->qh, &s->nh, rho, p.hail_density,
        HAIL_VMIN, HAIL_VMAX);
    s->qv = fmaxf(s->qv, 0.0f);
    s->qc = fmaxf(s->qc, 0.0f);
    s->qr = fmaxf(s->qr, 0.0f);
    s->qi = fmaxf(s->qi, 0.0f);
    s->qs = fmaxf(s->qs, 0.0f);
    s->qg = fmaxf(s->qg, 0.0f);
    s->qh = fmaxf(s->qh, 0.0f);
    s->nc = fmaxf(s->nc, 0.0f);
    s->nr = fmaxf(s->nr, 0.0f);
    s->ni = fmaxf(s->ni, 0.0f);
    s->ns = fmaxf(s->ns, 0.0f);
    s->ng = fmaxf(s->ng, 0.0f);
    s->nh = fmaxf(s->nh, 0.0f);
    s->nn = fmaxf(s->nn, 0.0f);
    s->vg = fmaxf(s->vg, 0.0f);
    s->vh = fmaxf(s->vh, 0.0f);
}

__device__ __forceinline__ void diagnose_warm(
    const State& s, float rho, float pressure, float dt, Rates* r)
{
    const float dt_inverse = (float)(1.0 / (double)dt);
    float cloud_number = s.nc;
    float rain_number = s.nr;

    // setvtz cloud and rain reconstruction, option-18 cnu=alphar=0.
    float cloud_volume = CLOUD_VMIN;
    float cloud_diameter = 4.0e-6f;
    if (s.qc > QC_MIN) {
        const float mass_min = 1000.0f * CLOUD_VMIN;
        const float mass_max = 1000.0f * CLOUD_VMAX;
        float mass;
        if (cloud_number > CXMIN) {
            mass = fminf(fmaxf(s.qc * rho / cloud_number, mass_min), mass_max);
        } else {
            cloud_number = fmaxf(CXMIN, rho * s.qc / mass_max);
            mass = fminf(fmaxf(s.qc * rho / cloud_number, mass_min), mass_max);
        }
        cloud_volume = mass / 1000.0f;
        cloud_diameter = powf(mass * (6.0f / (PI * 1000.0f)), 1.0f / 3.0f);
    }

    float rain_volume = RAIN_VMIN;
    float rain_diameter = 80.0e-6f;
    float rain_characteristic = 1.0e-9f;
    if (s.qr > QR_MIN) {
        rain_volume = rho * s.qr /
            (1000.0f * fmaxf(1.0e-11f, rain_number));
        if (rain_volume < RAIN_VMIN || rain_volume > RAIN_VMAX) {
            rain_volume = fminf(RAIN_VMAX, fmaxf(RAIN_VMIN, rain_volume));
            rain_number = rho * s.qr / (1000.0f * rain_volume);
        }
        rain_diameter = powf(rain_volume * (6.0f / PI), 1.0f / 3.0f);
        rain_characteristic = powf(rain_volume / PI, 1.0f / 3.0f);
    }

    const double rb = (double)(0.5f * cloud_diameter);
    const float xl2p_prefactor =
        2.7e-2f * 1000.0f * cloud_number * cloud_volume;
    const double xl2p_shape =
        (double)5.0e19f * rb * rb * rb * (double)cloud_diameter
        - (double)0.4f;
    const double xl2p = fmax(0.0, (double)xl2p_prefactor * xl2p_shape);

    if (s.qc > QC_MIN && cloud_number > 1000.0f
            && s.temperature > 237.15f) {
        const float ccmxd = (float)(
            0.1 * (double)cloud_number * (double)dt_inverse);
        const float collision = 2.0f * 9.44e15f
            * (cloud_number * cloud_number)
            * (cloud_volume * cloud_volume);
        r->cautn = (float)fmax(0.0, (double)fminf(ccmxd, collision));
        if (rb > 7.51e-6) {
            const double t2s = (double)3.72f /
                ((double)1.0e6f * (rb - 7.500e-6)
                 * (double)rho * (double)s.qc);
            r->qrcnw = (float)fmax(0.0, xl2p / (t2s * (double)rho));
            r->crcnw = (float)fmax(
                0.0, fmin(3.5e9 * xl2p / t2s, 0.5 * (double)r->cautn));
            if ((double)(s.qr * rho) > 1.2 * xl2p
                    && rain_number > CXMIN && s.qr > 0.0f) {
                r->crcnw = rain_number / s.qr * r->qrcnw;
            }
            if (r->crcnw < 1.0e-30f) r->qrcnw = 0.0f;
        }
    }

    // Rain collection of cloud water and cloud number.
    if (s.qc > QC_MIN && s.qr > QR_MIN) {
        double initiation_radius = 41.0e-6;
        if (rb > 3.51e-6) {
            initiation_radius = fmax(
                41.0e-6, 6.3e-4 / (1.0e6 * (rb - 3.5e-6)));
        }
        const float rain_radius = 0.5f * rain_diameter;
        if ((double)rain_radius > initiation_radius) {
            if (rain_radius > 50.0e-6f) {
                r->qracw = 5.78e3f * rain_number * cloud_number
                    * (1000.0f * cloud_volume)
                    * (2.0f * cloud_volume + rain_volume) / rho;
                r->cracw = 5.78e3f * rain_number * cloud_number
                    * (cloud_volume + rain_volume);
            } else {
                r->qracw = 9.44e15f * rain_number * s.qc
                    * (6.0f * cloud_volume * cloud_volume
                       + 20.0f * rain_volume * rain_volume);
                r->cracw = 9.44e15f * rain_number * cloud_number
                    * (2.0f * cloud_volume * cloud_volume
                       + 20.0f * rain_volume * rain_volume);
            }
            r->qracw = fminf(
                r->qracw,
                (float)(0.1 * (double)s.qc * (double)dt_inverse));
            if (!(r->qracw > 0.0f)) r->cracw = 0.0f;
        }
    }

    // Rain self-collection/breakup number sink.
    if (s.qr > QR_MIN && rain_diameter - 0.1e-3f <= 1.9e-3f) {
        float efficiency = 1.0f;
        if (rain_diameter >= 6.1e-4f) {
            efficiency = expf(-50.0f * (50.0f * (rain_diameter - 6.0e-4f)));
        }
        if (0.5f * rain_diameter >= 50.0e-6f) {
            r->cracr = (float)((double)efficiency * (double)5.78e3f
                * (double)(rain_number * rain_number)
                * (double)rain_volume);
        } else {
            const float ratio = (6.0f * 5.0f * 4.0f) /
                (3.0f * 2.0f * 1.0f);
            const float nv = rain_number * rain_volume;
            r->cracr = (float)((double)efficiency * (double)9.44e15f
                * (double)(nv * nv) * (double)ratio);
        }
    }

    // Rain evaporation. Default rcond=0 makes this a non-positive rate.
    if (s.qr > QR_MIN) {
        int table_index = (int)((s.temperature - 163.15f) / 0.002f + 1.5f);
        table_index = min(1000001, max(1, table_index));
        const float table_temperature = __fadd_rn(
            163.15f, __fmul_rn((float)(table_index - 1), 0.002f));
        const float saturation = (380.0f / pressure) * expf(
            17.2693882f * (table_temperature - 273.15f)
            / (table_temperature - 35.86f));
        const float lv = latent_vapor(s.temperature);
        const float diffusivity = 2.11e-5f
            * powf(s.temperature / 273.15f, 1.94f)
            * (101325.0f / pressure);
        const float viscosity = 1.832e-5f
            * (416.16f / (s.temperature + 120.0f))
            * powf(s.temperature / 296.0f, 1.5f);
        const float kinematic = viscosity / rho;
        const float conductivity = 2.43e-2f * viscosity / 1.718e-5f;
        const float ventilation_factor = powf(
            kinematic / diffusivity, 1.0f / 3.0f) * powf(kinematic, -0.5f);
        const float fall_factor = sqrtf(1.225f / fmaxf(0.05f, rho));
        const float ventilation = 0.78f
            + 0.308f * 1.8273550271987915f * ventilation_factor
            * sqrtf(841.99666f * fall_factor)
            * powf(rain_characteristic, 0.9f);
        const float capacitance = 0.5f * rain_characteristic;
        const float resistance = lv * lv /
            (conductivity * 461.5f * s.temperature * s.temperature)
            + 1.0f / (rho * diffusivity * saturation);
        const float growth = (4.0f * PI / rho)
            * (s.qv / saturation - 1.0f) / resistance;
        r->qrcev = fminf(
            growth * rain_number * ventilation * capacitance, 0.0f);
        r->qrcev = fmaxf(
            r->qrcev,
            -(float)(0.1 * (double)s.qr * (double)dt_inverse));
        if (r->qrcev < 0.0f) {
            r->crcev = (rain_number / s.qr) * r->qrcev;
        }
    }
}

__device__ __forceinline__ void diagnose_cloud_riming(
    const State& s,
    const ParticleProperties& particles,
    float rho,
    float layer_depth,
    float dt,
    Rates* r)
{
    // WRF 15493-17238: all frozen collectors diagnose from the same cloud
    // state.  Their independent local caps are followed later by WRF's one
    // shared cloud mass/number donor limiter.
    // The process branches themselves gate on hydrometeor mass, not CXMIN;
    // initial setvtz has already made every active category number nonzero.
    // ICEZVD_GS initializes rimdn(:,:) to the fixed 500 kg m-3 `rimedens`.
    // Graupel/rain collection later (intentionally) reuses the cloud-riming
    // value through WRF's raindn/rimdn typo.
    r->graupel_cloud_rime_density = 500.0f;
    if (!(s.qc > QC_MIN)) return;
    const float dt_inverse = (float)(1.0 / (double)dt);
    const float cloud_mass = fminf(
        1000.0f * CLOUD_VMAX,
        fmaxf(1000.0f * CLOUD_VMIN, rho * s.qc / s.nc));
    const float cloud_volume = cloud_mass / 1000.0f;
    const float cloud_diameter =
        powf(cloud_volume * (6.0f / PI), 1.0f / 3.0f);
    const float cloud_radius = 0.5f * cloud_diameter;
    const float viscosity = 1.832e-5f
        * (416.16f / (s.temperature + 120.0f))
        * powf(s.temperature / 296.0f, 1.5f);
    const float cloud_velocity = 2.0f * 9.8f * 1000.0f
        * cloud_radius * cloud_radius / (9.0f * viscosity);
    const float density_factor = sqrtf(1.225f / fmaxf(0.05f, rho));

    // Cloud droplets collected by pristine cloud ice.
    if (s.temperature < 273.15f && s.qi > QI_MIN) {
        const float ice_mass = fmaxf(rho * s.qi / s.ni, ICE_MMIN);
        const float ice_diameter = 0.1871f * powf(ice_mass, 0.3429f);
        if (cloud_diameter > 15.0e-6f && ice_diameter > 30.0e-6f) {
            const float ice_volume = ice_mass / 900.0f;
            const float ice_velocity = 47.6273f * density_factor
                * powf(ice_volume, 0.18333f) * 1.091937899589539f;
            const float relative_velocity = sqrtf(
                (ice_velocity - cloud_velocity)
                    * (ice_velocity - cloud_velocity)
                + 0.04f * ice_velocity * cloud_velocity);
            const float geometry = tgammaf(1.6858f)
                    * ice_diameter * ice_diameter
                + 2.0f * tgammaf(1.3429f) * tgammaf(2.3333333333f)
                    * ice_diameter * cloud_diameter
                + tgammaf(2.6666666667f)
                    * cloud_diameter * cloud_diameter;
            r->qiacw = fminf(
                0.25f * PI * 0.5f * s.ni * s.qc
                    * relative_velocity * geometry,
                0.1f * s.qc * dt_inverse);
            r->ciacw = fminf(
                r->qiacw * rho / cloud_mass,
                0.1f * s.nc * dt_inverse);
        }
    }

    // Cloud droplets collected by snow.
    if (s.qs > QS_MIN) {
        const float snow_volume = fmaxf(
            SNOW_VMIN,
            rho * s.qs / (particles.snow_density * s.ns));
        const float collection_count = 0.104f * 5.78e3f
            * s.ns * s.nc * (2.0f * cloud_volume + snow_volume);
        r->qsacw = fminf(
            collection_count * cloud_mass / rho,
            0.1f * s.qc * dt_inverse);
        r->csacw = fminf(
            collection_count, 0.1f * s.nc * dt_inverse);
    }

    const float collection_efficiency = fminf(
        0.9f,
        fminf(
            -0.27544f + cloud_radius
                * (0.26249e6f + cloud_radius
                    * (-1.8896e10f + cloud_radius * 4.4626e14f)),
            1.0f));
    if (!(collection_efficiency > 0.0f) || cloud_diameter < 2.4e-6f) {
        return;
    }

    // Cloud droplets collected by graupel.
    if (s.qg > QD_MIN) {
        const float mean_volume = fmaxf(
            GRAUPEL_VMIN,
            rho * s.qg / (particles.graupel_density * s.ng));
        const float diameter = powf(mean_volume * (6.0f / PI), 1.0f / 3.0f);
        const float characteristic = powf(6.0f, -1.0f / 3.0f) * diameter;
        float fall_a, fall_b;
        dense_fall_coefficients(
            particles.graupel_density, &fall_a, &fall_b);
        const float velocity = density_factor * fall_a * powf(characteristic, fall_b)
            * wrf_gamma_lookup(4.0f + fall_b) / wrf_gamma_lookup(4.0f);
        // WRF's 70 and 150 m s-1 clamps belong only to sedimentation.  GS
        // microphysical collection uses the uncapped vtxbar fall speed.
        const float relative_velocity = fabsf(velocity - cloud_velocity);
        // WRF's delbk/delabk tables are initialized through GAMMA_SP and
        // rounded to real.  Preserve those admitted alpha_g=0/alpha_c=2
        // coefficients instead of recomputing them with CUDA tgammaf.
        const float geometry = 0.605706990f * diameter * diameter
            + 1.31048179f * cloud_diameter * diameter
            + 1.50459337f * cloud_diameter * cloud_diameter;
        r->qhacw = fminf(
            0.25f * PI * collection_efficiency * s.ng * s.qc
                * relative_velocity * geometry,
            0.5f * s.qc * dt_inverse);
        r->qhacwmlr = r->qhacw;
        r->chacw = fminf(
            r->qhacw * rho / cloud_mass,
            0.5f * s.nc * dt_inverse);
        float rime_density = 1000.0f;
        if (s.temperature < 273.15f) {
            const float rime_parameter = -(0.5f * 1.0e6f * cloud_diameter)
                * (0.60f * velocity) / (s.temperature - 273.15f);
            rime_density = fminf(
                900.0f,
                fmaxf(170.0f, 300.0f * powf(rime_parameter, 0.44f)));
        }
        r->graupel_cloud_rime_density = rime_density;
        r->vhacw = rho * r->qhacw / rime_density;
    }

    // Cloud droplets collected by hail.  WRF caps hail fall speed by dz/dt
    // specifically in this collection path.
    if (s.qh > QD_MIN) {
        const float mean_volume = fmaxf(
            HAIL_VMIN,
            rho * s.qh / (particles.hail_density * s.nh));
        const float diameter = powf(mean_volume * (6.0f / PI), 1.0f / 3.0f);
        const float characteristic = powf(24.0f, -1.0f / 3.0f) * diameter;
        float fall_a, fall_b;
        dense_fall_coefficients(particles.hail_density, &fall_a, &fall_b);
        float velocity = density_factor * fall_a * powf(characteristic, fall_b)
            * wrf_gamma_lookup(5.0f + fall_b) / wrf_gamma_lookup(5.0f);
        velocity = fminf(layer_depth * dt_inverse, velocity);
        const float relative_velocity = fabsf(velocity - cloud_velocity);
        // Static WRF Seifert-table coefficients for admitted alpha_hail=1
        // and alpha_cloud=2.  The mixed term is delabk(lhl,lc,k=1), not the
        // product of two independently recomputed diameter moments.
        const float geometry = 0.721125066f * diameter * diameter
            + 1.65110373f * cloud_diameter * diameter
            + 1.50459337f * cloud_diameter * cloud_diameter;
        r->qhlacw = fminf(
            0.25f * PI * collection_efficiency * s.nh * s.qc
                * relative_velocity * geometry,
            0.5f * s.qc * dt_inverse);
        r->qhlacwmlr = r->qhlacw;
        r->chlacw = fminf(
            r->qhlacw * rho / cloud_mass,
            0.5f * s.nc * dt_inverse);
        float rime_density = 1000.0f;
        if (s.temperature < 273.15f) {
            const float rime_parameter = -(0.5f * 1.0e6f * cloud_diameter)
                * (0.60f * velocity) / (s.temperature - 273.15f);
            rime_density = fminf(
                900.0f,
                fmaxf(500.0f, 300.0f * powf(rime_parameter, 0.44f)));
        }
        r->vhlacw = rho * r->qhlacw / rime_density;
    }
}

__device__ __forceinline__ void diagnose_rain_freezing(
    const State& s,
    const ParticleProperties& particles,
    float rho,
    float pressure,
    float dt,
    Rates* r)
{
    // WRF 16000-16045, 16742-16927, 17575-17918, 20318-20379.
    // Bigg freezing and rain/ice collision are diagnosed independently, then
    // share one rain heat budget before any number or mass donor limiter.
    const float dt_inverse = (float)(1.0 / (double)dt);
    const float temperature_c = s.temperature - 273.15f;
    float rain_number = s.nr;
    float rain_mean_volume = RAIN_VMIN;
    float rain_mean_diameter = 1.0e-9f;
    float rain_characteristic = 1.0e-9f;
    if (s.qr > QR_MIN) {
        rain_mean_volume = rho * s.qr
            / (1000.0f * fmaxf(1.0e-11f, rain_number));
        rain_mean_volume = fminf(
            RAIN_VMAX, fmaxf(RAIN_VMIN, rain_mean_volume));
        rain_number = rho * s.qr / (1000.0f * rain_mean_volume);
        rain_mean_diameter =
            powf(rain_mean_volume * (6.0f / PI), 1.0f / 3.0f);
        rain_characteristic = powf(rain_mean_volume / PI, 1.0f / 3.0f);
    }

    // Default Bigg option 2, including the active <0.30-mm snow split.
    if (s.qr > QR_MIN && temperature_c < -5.0f) {
        const float threshold_volume = expf(16.2f + temperature_c) * 1.0e-6f;
        const float bigg_diameter =
            powf((6.0f / PI) * threshold_volume, 1.0f / 3.0f);
        if (bigg_diameter < 8.0e-3f) {
            float number_fraction, mass_fraction;
            rain_tail_fractions(
                bigg_diameter / rain_characteristic,
                &number_fraction,
                &mass_fraction);
            r->crfrz = number_fraction * rain_number * dt_inverse;
            r->qrfrz = mass_fraction * s.qr * dt_inverse;
            r->crfrzf = r->crfrz;
            r->qrfrzf = r->qrfrz;
            if (r->qrfrz * dt < QD_MIN || r->crfrz * dt < CXMIN) {
                r->crfrz = 0.0f;
                r->qrfrz = 0.0f;
                r->crfrzf = 0.0f;
                r->qrfrzf = 0.0f;
            } else if (bigg_diameter < 0.30e-3f) {
                float dense_number_fraction, dense_mass_fraction;
                rain_tail_fractions(
                    0.30e-3f / rain_characteristic,
                    &dense_number_fraction,
                    &dense_mass_fraction);
                r->crfrzf = dense_number_fraction * rain_number * dt_inverse;
                r->qrfrzf = dense_mass_fraction * s.qr * dt_inverse;
                r->crfrzs = r->crfrz - r->crfrzf;
                r->qrfrzs = r->qrfrz - r->qrfrzf;
            }
            // Preserve the source's normally unreachable inverted correction.
            if (r->qrfrz * dt > s.qr) {
                const float factor = r->qrfrz * dt / s.qr;
                r->qrfrz *= factor;
                r->qrfrzs *= factor;
                r->qrfrzf *= factor;
                r->crfrz *= factor;
                r->crfrzs *= factor;
                r->crfrzf *= factor;
            }
            r->vrfrzf = rho * r->qrfrzf / 900.0f;
        }
    }

    const bool collision_active = s.qr > QR_MIN && s.qi > QI_MIN
        && s.ni > CXMIN;
    float ice_mass = ICE_MMIN;
    float ice_volume = ICE_MMIN / 900.0f;
    float ice_diameter = 1.0e-7f;
    if (collision_active) {
        ice_mass = fmaxf(rho * s.qi / s.ni, ICE_MMIN);
        ice_volume = ice_mass / 900.0f;
        ice_diameter = 0.1871f * powf(ice_mass, 0.3429f);
    }
    const float density_factor = sqrtf(1.225f / fmaxf(0.05f, rho));
    const float rain_velocity = s.qr > QR_MIN
        ? density_factor * 10.0f * (1.0f - powf(
            1.0f + 516.575f * rain_characteristic, -4.0f))
        : 0.0f;
    const float ice_velocity = collision_active
        ? 47.6273f * density_factor * powf(ice_volume, 0.18333f)
            * 1.091937899589539f
        : 0.0f;

    // Rain collection of cloud-ice mass/number (qraci/craci).
    if (collision_active && ice_diameter >= 10.0e-6f
            && rain_mean_diameter > 100.0e-6f) {
        const float collision_count = 0.1f * 5.78e3f
            * rain_number * s.ni * (2.0f * ice_volume + rain_mean_volume);
        r->qraci = fminf(
            0.1f * s.qi * dt_inverse,
            collision_count * ice_mass / rho);
        r->craci = fminf(
            0.1f * s.ni * dt_inverse, collision_count);
        if (s.temperature > 268.15f) r->qraci = 0.0f;
        r->qracif = r->qraci;
    }

    // Rain frozen by collision with cloud ice (qiacr/ciacr).
    const bool double_moment_collision_gate =
        collision_active && ice_diameter >= 10.0e-6f
        && s.temperature <= 270.15f;
    if (double_moment_collision_gate) {
        const float eligible_ice_number = s.ni
            * expf(-powf(40.0e-6f / ice_diameter, 3.0f));
        float rain_number_fraction, rain_mass_fraction;
        rain_tail_fractions(
            150.0e-6f / rain_characteristic,
            &rain_number_fraction,
            &rain_mass_fraction);
        const float eligible_rain_number = rain_number_fraction * rain_number;
        const float eligible_rain = rain_mass_fraction * s.qr;
        const float relative_velocity = sqrtf(
            (rain_velocity - ice_velocity) * (rain_velocity - ice_velocity)
            + 0.04f * rain_velocity * ice_velocity);
        float graupel_diameter = 1.0e-9f;
        if (s.qg > QD_MIN && s.ng > CXMIN) {
            const float graupel_volume = rho * s.qg
                / (particles.graupel_density * s.ng);
            graupel_diameter =
                powf(graupel_volume * (6.0f / PI), 1.0f / 3.0f);
        }
        const float gamma_ice = tgammaf(1.3429f);
        const float one_sixth = 1.0f / 6.0f;
        const float mass_geometry = tgammaf(1.6858f)
                * ice_diameter * ice_diameter
            + 48.0f * gamma_ice * powf(one_sixth, 4.0f / 3.0f)
                * graupel_diameter * ice_diameter
            + 120.0f * powf(one_sixth, 5.0f / 3.0f)
                * rain_mean_diameter * rain_mean_diameter;
        const float number_geometry = tgammaf(1.6858f)
                * ice_diameter * ice_diameter
            + 2.0f * gamma_ice * powf(one_sixth, 1.0f / 3.0f)
                * rain_mean_diameter * ice_diameter
            + 2.0f * powf(one_sixth, 2.0f / 3.0f)
                * rain_mean_diameter * rain_mean_diameter;
        r->qiacr = fminf(
            0.1f * s.qr * dt_inverse,
            0.25f * PI * 0.1f * eligible_ice_number * eligible_rain
                * relative_velocity * mass_geometry);
        r->ciacr = fminf(
            0.1f * rain_number * dt_inverse,
            0.25f * PI * 0.1f * eligible_ice_number * eligible_rain_number
                * relative_velocity * number_geometry);
        r->ciacrf = r->ciacr;
        float dense_fraction = 1.0f;
        if (r->ciacr > QD_MIN) {
            const float frozen_mean_volume =
                rho * r->qiacr / (r->ciacr * 900.0f);
            dense_fraction = 0.5f * (1.0f + tanhf(
                0.2e12f * (frozen_mean_volume
                    - 1.15f * GRAUPEL_VMIN)));
            r->qiacrs = (1.0f - dense_fraction) * r->qiacr;
            r->ciacrs = (1.0f - dense_fraction) * r->ciacrf;
        }
        r->qiacrf = dense_fraction * r->qiacr;
        r->ciacrf = dense_fraction * r->ciacrf;
        r->viacrf = rho * r->qiacrf / 900.0f;
    } else {
        // Source-exact dangling-ELSE behavior at WRF 16754-16886: this
        // nominal "single-moment rain" branch pairs with the outer
        // iacr/efficiency/temperature gate.  Therefore it executes for the
        // admitted two-moment configuration whenever that outer gate is
        // false (notably T > 270.15 K), while retaining eri as its multiplier.
        const float efficiency = collision_active
                && ice_diameter >= 10.0e-6f
            ? 0.1f
            : 0.0f;
        const float geometry =
            120.0f * rain_characteristic * rain_characteristic
            + 48.0f * rain_characteristic * ice_diameter
            + 12.0f * ice_diameter * ice_diameter;
        r->qiacr = fminf(
            0.1f * s.qr * dt_inverse,
            (0.25f / 6.0f) * PI * efficiency * s.ni * s.qr
                * fabsf(rain_velocity - ice_velocity) * geometry);
        r->qiacrf = r->qiacr;
        r->viacrf = rho * r->qiacrf / 900.0f;
    }

    const float freezing_total = r->qrfrz + r->qiacr + r->qsacr;
    float maximum_freezing = fmaxf(0.0f, freezing_total);
    if (!(temperature_c < -30.0f)) {
        const float diffusivity = 2.11e-5f
            * powf(s.temperature / 273.15f, 1.94f)
            * (101325.0f / pressure);
        const float viscosity = 1.832e-5f
            * (416.16f / (s.temperature + 120.0f))
            * powf(s.temperature / 296.0f, 1.5f);
        const float kinematic = viscosity / rho;
        const float ventilation_factor = powf(
            kinematic / diffusivity, 1.0f / 3.0f)
            * powf(kinematic, -0.5f);
        const float rain_ventilation = 0.78f
            + 0.308f * tgammaf(2.9f) * ventilation_factor
                * sqrtf(841.99666f * density_factor)
                * powf(rain_characteristic, 0.9f);
        const float bounded_celsius = fminf(
            273.15f, fmaxf(233.15f, s.temperature)) - 273.15f;
        const float liquid_offset = bounded_celsius - 35.0f;
        const float liquid_heat = 4203.1548f
            + 1.30572e-2f * liquid_offset * liquid_offset
            + 1.60056e-5f * liquid_offset * liquid_offset
                * liquid_offset * liquid_offset;
        const float conductivity = 2.43e-2f * viscosity / 1.718e-5f;
        const float wet_growth = (2.0f * PI)
            * (latent_vapor(s.temperature) * diffusivity * rho
                    * (380.0f / pressure - s.qv)
                - conductivity * temperature_c)
            / (rho * (latent_fusion(s.temperature)
                + liquid_heat * temperature_c));
        maximum_freezing = fmaxf(
            rain_characteristic * rain_ventilation * rain_number
                * wet_growth,
            0.0f);
        maximum_freezing = fminf(freezing_total, maximum_freezing);
        maximum_freezing = fminf(s.qr * dt_inverse, maximum_freezing);
    } else {
        maximum_freezing = s.qr * dt_inverse;
    }

    float factor = 1.0f;
    if (freezing_total > maximum_freezing && freezing_total > QR_MIN) {
        factor = maximum_freezing / freezing_total;
    }
    factor = fminf(1.0f, factor);
    if (s.temperature <= 273.15f && factor < 1.0f) {
        r->qrfrz *= factor;
        r->qrfrzs *= factor;
        r->qrfrzf *= factor;
        r->qiacr *= factor;
        r->qsacr *= factor;
        r->qiacrf *= factor;
        r->qiacrs *= factor;
        r->crfrz *= factor;
        r->crfrzf *= factor;
        r->crfrzs *= factor;
        r->ciacr *= factor;
        r->ciacrf *= factor;
        r->ciacrs *= factor;
        r->vrfrzf *= factor;
        r->viacrf *= factor;
    }
}

__device__ __forceinline__ void diagnose_cloud_freezing(
    const State& s,
    float rho,
    float pressure,
    float dt,
    Rates* r)
{
    // WRF 17927-18118, exact default ibfc=1 and icfn=2 paths.
    if (!(s.qc > QC_MIN)) return;
    const float dt_inverse = (float)(1.0 / (double)dt);
    const float temperature_c = s.temperature - 273.15f;
    const float cloud_volume = fminf(
        CLOUD_VMAX,
        fmaxf(
            CLOUD_VMIN,
            rho * s.qc / (1000.0f * fmaxf(s.nc, 1.0e-20f))));
    const float cloud_mass = 1000.0f * cloud_volume;
    const float cloud_diameter =
        powf(cloud_volume * (6.0f / PI), 1.0f / 3.0f);

    // Homogeneous freezing.  There is no local qcmxd/ccmxd cap for the
    // ipconc>=2 path; the later shared cloud limiters own the competition.
    if (s.temperature < 268.15f && s.nc > CXMIN
            && cloud_diameter > 0.0f) {
        const float threshold_volume =
            expf(16.2f + temperature_c) * 1.0e-6f;
        r->cwfrz = s.nc * expf(-threshold_volume / cloud_volume)
            * dt_inverse;
        r->qwfrz = r->cwfrz * 1000.0f / rho
            * (threshold_volume + cloud_volume);
        // Default crystal routing is entirely to columns/cloud ice.
        r->cwfrzc = r->cwfrz;
        r->qwfrzc = r->qwfrz;
    }

    // Cotton/Meyers contact freezing.  The official gate has no explicit
    // number threshold; a zero number simply makes fn1 and the rate zero.
    if (!(s.temperature < 271.15f)) return;
    const float ccia = expf(4.11f - 0.262f * temperature_c);
    constexpr float AEROSOL_RADIUS = 3.0e-7f;
    constexpr float AEROSOL_CONDUCTIVITY = 5.39e-3f;
    constexpr float BOLTZMANN = 1.3807e-23f;
    const float knudsen = 2.28e-5f * s.temperature
        / (pressure * AEROSOL_RADIUS);
    const float knudsen_adjustment =
        1.257f + 0.4f * expf(-1.1f / knudsen);
    const float viscosity = 1.832e-5f
        * (416.16f / (s.temperature + 120.0f))
        * powf(s.temperature / 296.0f, 1.5f);
    const float diffusivity = 2.11e-5f
        * powf(s.temperature / 273.15f, 1.94f)
        * (101325.0f / pressure);
    const float conductivity = 2.43e-2f * viscosity / 1.718e-5f;
    const float lv = latent_vapor(s.temperature);
    const float ls = lv + latent_fusion(s.temperature);
    const float ice_saturation =
        ice_saturation_mixing_ratio(s.temperature, pressure);
    const float water_saturation =
        liquid_saturation_mixing_ratio(s.temperature, pressure);
    const float fai = ls * ls
        / (conductivity * 461.5f * s.temperature * s.temperature);
    const float fbi = 1.0f / (rho * diffusivity * ice_saturation);
    const float growth_transport = 1.0f / (fai + fbi);
    const float aerosol_diffusivity = BOLTZMANN * s.temperature
        * (1.0f + knudsen_adjustment * knudsen)
        / (6.0f * PI * viscosity * AEROSOL_RADIUS);
    const float fn1 =
        2.0f * PI * cloud_diameter * s.nc * ccia;
    const float fn2 = -growth_transport
        * (s.qv / water_saturation - 1.0f) * lv / pressure;
    const float thermal_factor = 0.4f
        * (1.0f + 1.45f * knudsen
            + 0.4f * knudsen * expf(-1.0f / knudsen))
        * (conductivity + 2.5f * knudsen * AEROSOL_CONDUCTIVITY)
        / ((1.0f + 3.0f * knudsen)
            * (2.0f * conductivity
                + 5.0f * knudsen * AEROSOL_CONDUCTIVITY
                + AEROSOL_CONDUCTIVITY));
    const float brownian = fn1 * aerosol_diffusivity;
    const float thermophoretic = fn1 * fn2 * thermal_factor / rho;
    const float diffusiophoretic = fn1 * fn2 * 1000.0f * s.temperature
        / (lv * rho);
    const float raw_count =
        fmaxf(brownian + thermophoretic + diffusiophoretic, 0.0f);
    const float cwctfz = fminf(
        raw_count * dt_inverse,
        0.1f * s.nc * dt_inverse);
    r->qwctfz = cloud_mass * cwctfz / rho;
    r->cwctfzc = cwctfz;
    r->qwctfzc = r->qwctfz;
}

__device__ __forceinline__ void diagnose_frozen_collection(
    const State& s,
    const ParticleProperties& particles,
    float rho,
    float layer_depth,
    float dt,
    Rates* r)
{
    // WRF 15599-17323 default dry cross-collection.  The ipconc=5
    // snow-rain two-moment branch is intentionally empty, so qsacr remains 0.
    const float dt_inverse = (float)(1.0 / (double)dt);
    const float temperature_c = s.temperature - 273.15f;
    const float density_factor = sqrtf(1.225f / fmaxf(0.05f, rho));

    float rain_volume = RAIN_VMIN;
    float rain_diameter = 1.0e-9f;
    float rain_characteristic = 1.0e-9f;
    float rain_velocity = 0.0f;
    if (s.qr > QR_MIN && s.nr > 0.0f) {
        rain_volume = fminf(
            RAIN_VMAX,
            fmaxf(RAIN_VMIN, rho * s.qr / (1000.0f * s.nr)));
        rain_diameter = powf(rain_volume * (6.0f / PI), 1.0f / 3.0f);
        rain_characteristic = powf(rain_volume / PI, 1.0f / 3.0f);
        rain_velocity = density_factor * 10.0f * (1.0f - powf(
            1.0f + 516.575f * rain_characteristic, -4.0f));
    }

    float ice_mass = ICE_MMIN;
    float ice_volume = ICE_MMIN / 900.0f;
    float ice_diameter = 1.0e-9f;
    float ice_velocity = 0.0f;
    if (s.qi > QI_MIN && s.ni > 0.0f) {
        ice_mass = fmaxf(rho * s.qi / s.ni, ICE_MMIN);
        ice_volume = ice_mass / 900.0f;
        ice_diameter = 0.1871f * powf(ice_mass, 0.3429f);
        ice_velocity = 47.6273f * density_factor * powf(ice_volume, 0.18333f)
            * 1.091937899589539f;
    }

    float snow_volume = SNOW_VMIN;
    float snow_diameter = 1.0e-9f;
    float snow_velocity = 0.0f;
    if (s.qs > QS_MIN && s.ns > 0.0f) {
        snow_volume = fmaxf(
            SNOW_VMIN,
            rho * s.qs / (particles.snow_density * s.ns));
        snow_diameter = powf(snow_volume * (6.0f / PI), 1.0f / 3.0f);
        snow_velocity = 11.9495f * density_factor * powf(snow_volume, 0.14f);
    }

    float graupel_volume = GRAUPEL_VMIN;
    float graupel_diameter = 1.0e-9f;
    float graupel_velocity = 0.0f;
    if (s.qg > QD_MIN && s.ng > 0.0f) {
        graupel_volume = fmaxf(
            GRAUPEL_VMIN,
            rho * s.qg / (particles.graupel_density * s.ng));
        graupel_diameter =
            powf(graupel_volume * (6.0f / PI), 1.0f / 3.0f);
        const float characteristic =
            powf(6.0f, -1.0f / 3.0f) * graupel_diameter;
        float coefficient, exponent;
        dense_fall_coefficients(
            particles.graupel_density, &coefficient, &exponent);
        graupel_velocity = density_factor * coefficient
            * powf(characteristic, exponent)
            * tgammaf(4.0f + exponent) / tgammaf(4.0f);
    }

    float hail_volume = HAIL_VMIN;
    float hail_diameter = 1.0e-9f;
    float hail_velocity = 0.0f;
    if (s.qh > QD_MIN && s.nh > 0.0f) {
        hail_volume = fmaxf(
            HAIL_VMIN,
            rho * s.qh / (particles.hail_density * s.nh));
        hail_diameter = powf(hail_volume * (6.0f / PI), 1.0f / 3.0f);
        const float characteristic =
            powf(24.0f, -1.0f / 3.0f) * hail_diameter;
        float coefficient, exponent;
        dense_fall_coefficients(
            particles.hail_density, &coefficient, &exponent);
        hail_velocity = density_factor * coefficient
            * powf(characteristic, exponent)
            * tgammaf(5.0f + exponent) / tgammaf(5.0f);
        hail_velocity = fminf(layer_depth * dt_inverse, hail_velocity);
    }

    constexpr float DA0_R = 0.6057068643f;
    constexpr float DA1_R = 6.057068643f;
    constexpr float DA0_I = 0.9060338860f;
    constexpr float DA1_I = 1.527396263f;
    constexpr float DA0_S = 0.6987612361f;
    constexpr float DA1_S = 3.027901273f;
    constexpr float DA0_G = 0.6057068643f;
    constexpr float DA0_H = 0.7211247852f;
    constexpr float DAB0_GI = 0.9816705967f;
    constexpr float DAB1_GI = 1.318283033f;
    constexpr float DAB0_GS = 0.6824619383f;
    constexpr float DAB1_GS = 1.819762383f;
    constexpr float DAB0_GR = 0.6057068643f;
    constexpr float DAB1_GR = 2.422827457f;
    constexpr float DAB0_HI = 1.236827449f;
    constexpr float DAB1_HI = 1.660932543f;
    constexpr float DAB0_HS = 0.8598481618f;
    constexpr float DAB1_HS = 2.292756933f;
    constexpr float DAB0_HR = 0.7631428284f;
    constexpr float DAB1_HR = 3.052571313f;

    if (s.qs > QS_MIN && s.qi > QI_MIN && s.ns > 0.0f && s.ni > 0.0f) {
        const float efficiency = fminf(
            0.1f, 0.1f * expf(0.1f * fminf(temperature_c, 0.0f)));
        if (s.temperature <= 273.15f && efficiency > 0.0f) {
            const float collision = 0.104f * 5.78e3f * s.ns * s.ni
                * (2.0f * ice_volume + snow_volume);
            r->qsaci = fminf(
                0.1f * s.qi * dt_inverse,
                efficiency * collision * ice_mass / rho);
            r->csaci = fminf(
                0.1f * s.ni * dt_inverse,
                efficiency * collision);
        }
    }

    if (s.qg > QD_MIN && s.qi > QI_MIN && s.ng > 0.0f && s.ni > 0.0f) {
        const float efficiency = fminf(
            1.0f,
            fmaxf(0.0f, 0.1f * expf(0.1f * fminf(temperature_c, 0.0f))));
        const float relative_velocity = sqrtf(
            (graupel_velocity - ice_velocity)
                * (graupel_velocity - ice_velocity)
            + 0.04f * graupel_velocity * ice_velocity);
        const float mass_geometry = DA0_G * graupel_diameter * graupel_diameter
            + DAB1_GI * graupel_diameter * ice_diameter
            + DA1_I * ice_diameter * ice_diameter;
        const float number_geometry = DA0_G * graupel_diameter * graupel_diameter
            + DAB0_GI * graupel_diameter * ice_diameter
            + DA0_I * ice_diameter * ice_diameter;
        r->qhaci0 = 0.25f * PI * s.ng * s.qi
            * relative_velocity * mass_geometry;
        r->chaci0 = 0.25f * PI * s.ng * s.ni
            * relative_velocity * number_geometry;
        r->qhaci = fminf(
            efficiency * r->qhaci0, 0.1f * s.qi * dt_inverse);
        r->chaci = fminf(
            efficiency * r->chaci0, 0.1f * s.ni * dt_inverse);
    }

    if (s.qg > QD_MIN && s.qs > QS_MIN && s.ng > 0.0f && s.ns > 0.0f
            && s.qc >= QC_MIN) {
        float collision_efficiency = 0.5f;
        if (snow_diameter < 40.0e-6f) {
            collision_efficiency = 0.0f;
        } else if (snow_diameter < 150.0e-6f) {
            collision_efficiency = 0.5f
                * (snow_diameter - 40.0e-6f) / 110.0e-6f;
        }
        const float conversion_efficiency = fminf(
            0.5f,
            0.1f * expf(0.1f * fminf(temperature_c, 0.0f))
                * fminf(1.0f, fmaxf(
                    0.0f, particles.graupel_density - 300.0f) / 300.0f));
        if (collision_efficiency > 0.0f && conversion_efficiency > 0.0f) {
            const float relative_velocity = sqrtf(
                (graupel_velocity - snow_velocity)
                    * (graupel_velocity - snow_velocity)
                + 0.04f * graupel_velocity * snow_velocity);
            const float mass_geometry = DA0_G * graupel_diameter * graupel_diameter
                + DAB1_GS * graupel_diameter * snow_diameter
                + DA1_S * snow_diameter * snow_diameter;
            const float number_geometry = DA0_G * graupel_diameter * graupel_diameter
                + DAB0_GS * graupel_diameter * snow_diameter
                + DA0_S * snow_diameter * snow_diameter;
            r->qhacs0 = 0.25f * PI * collision_efficiency
                * s.ng * s.qs * relative_velocity * mass_geometry;
            r->chacs0 = 0.25f * PI * collision_efficiency
                * s.ng * s.ns * relative_velocity * number_geometry;
            r->qhacs = fminf(
                conversion_efficiency * r->qhacs0,
                0.1f * s.qs * dt_inverse);
            r->chacs = fminf(
                conversion_efficiency * r->chacs0,
                0.1f * s.ns * dt_inverse);
        }
    }

    if (s.qg > QD_MIN && s.qr > QR_MIN && s.ng > 0.0f && s.nr > 0.0f) {
        const float efficiency = fminf(
            1.0f,
            expf(-40.0e-6f / rain_diameter)
                * expf(-40.0e-6f / graupel_diameter));
        const float relative_velocity = sqrtf(
            (graupel_velocity - rain_velocity)
                * (graupel_velocity - rain_velocity)
            + 0.04f * graupel_velocity * rain_velocity);
        const float mass_geometry = DA0_G * graupel_diameter * graupel_diameter
            + DAB1_GR * graupel_diameter * rain_diameter
            + DA1_R * rain_diameter * rain_diameter;
        const float number_geometry = DA0_G * graupel_diameter * graupel_diameter
            + DAB0_GR * graupel_diameter * rain_diameter
            + DA0_R * rain_diameter * rain_diameter;
        r->qhacr = fminf(
            0.1f * s.qr * dt_inverse,
            0.25f * PI * efficiency * s.ng * s.qr
                * relative_velocity * mass_geometry);
        r->qhacrmlr = r->qhacr;
        r->chacr = fminf(
            0.1f * s.nr * dt_inverse,
            0.25f * PI * efficiency * s.ng * s.nr
                * relative_velocity * number_geometry);
        if (s.temperature > 273.15f) {
            r->qhacr = 0.0f;
            r->chacr = 0.0f;
        } else if (s.temperature == 273.15f) {
            // At exactly freezing WRF leaves collection active but follows
            // the non-cold rain-density branch, whose divisor is liquid water.
            r->vhacr = rho * r->qhacr / 1000.0f;
        } else {
            // WRF 16518-16523 computes a rain-rime density and then
            // overwrites it with clamp(rimdn_g), so the earlier cloud-rime
            // diagnosis (or its 500 default) is the observable divisor.
            r->vhacr = rho * r->qhacr /
                fminf(900.0f, fmaxf(
                    170.0f, r->graupel_cloud_rime_density));
        }
    }

    if (s.qh > QD_MIN && s.qi > QI_MIN && s.nh > 0.0f && s.ni > 0.0f
            && s.temperature <= 273.15f && s.qc >= QC_MIN) {
        const float relative_velocity = sqrtf(
            (hail_velocity - ice_velocity) * (hail_velocity - ice_velocity)
            + 0.04f * hail_velocity * ice_velocity);
        const float mass_geometry = DA0_H * hail_diameter * hail_diameter
            + DAB1_HI * hail_diameter * ice_diameter
            + DA1_I * ice_diameter * ice_diameter;
        const float number_geometry = DA0_H * hail_diameter * hail_diameter
            + DAB0_HI * hail_diameter * ice_diameter
            + DA0_I * ice_diameter * ice_diameter;
        r->qhlaci0 = 0.25f * PI * s.nh * s.qi
            * relative_velocity * mass_geometry;
        r->chlaci0 = 0.25f * PI * s.nh * s.ni
            * relative_velocity * number_geometry;
        r->qhlaci = fminf(
            0.2f * r->qhlaci0, 0.1f * s.qi * dt_inverse);
        r->chlaci = fminf(
            0.2f * r->chlaci0, 0.1f * s.ni * dt_inverse);
    }

    if (s.qh > QD_MIN && s.qs > QS_MIN && s.nh > 0.0f && s.ns > 0.0f) {
        const float efficiency = fminf(
            0.5f, 0.1f * expf(0.1f * fminf(temperature_c, 0.0f)));
        const float relative_velocity = sqrtf(
            (hail_velocity - snow_velocity) * (hail_velocity - snow_velocity)
            + 0.04f * hail_velocity * snow_velocity);
        const float mass_geometry = DA0_H * hail_diameter * hail_diameter
            + DAB1_HS * hail_diameter * snow_diameter
            + DA1_S * snow_diameter * snow_diameter;
        const float number_geometry = DA0_H * hail_diameter * hail_diameter
            + DAB0_HS * hail_diameter * snow_diameter
            + DA0_S * snow_diameter * snow_diameter;
        r->qhlacs0 = 0.25f * PI * s.nh * s.qs
            * relative_velocity * mass_geometry;
        r->chlacs0 = 0.25f * PI * s.nh * s.ns
            * relative_velocity * number_geometry;
        r->qhlacs = fminf(
            efficiency * r->qhlacs0, 0.1f * s.qs * dt_inverse);
        r->chlacs = fminf(
            efficiency * r->chlacs0, 0.1f * s.ns * dt_inverse);
    }

    if (s.qh > QD_MIN && s.qr > QR_MIN && s.nh > 0.0f && s.nr > 0.0f) {
        const float relative_velocity = sqrtf(
            (hail_velocity - rain_velocity) * (hail_velocity - rain_velocity)
            + 0.04f * hail_velocity * rain_velocity);
        const float mass_geometry = DA0_H * hail_diameter * hail_diameter
            + DAB1_HR * hail_diameter * rain_diameter
            + DA1_R * rain_diameter * rain_diameter;
        const float number_geometry = DA0_H * hail_diameter * hail_diameter
            + DAB0_HR * hail_diameter * rain_diameter
            + DA0_R * rain_diameter * rain_diameter;
        r->qhlacr = fminf(
            0.1f * s.qr * dt_inverse,
            0.25f * PI * s.nh * s.qr
                * relative_velocity * mass_geometry);
        r->qhlacrmlr = r->qhlacr;
        r->chlacr = fminf(
            0.1f * s.nr * dt_inverse,
            0.25f * PI * s.nh * s.nr
                * relative_velocity * number_geometry);
        if (s.temperature > 273.15f) {
            r->qhlacr = 0.0f;
            r->chlacr = 0.0f;
        } else {
            r->vhlacr = rho * r->qhlacr / 900.0f;
        }
    }
}

__device__ __forceinline__ void diagnose_snow_aggregation(
    const State& s,
    const ParticleProperties& particles,
    float rho,
    float dt,
    Rates* r)
{
    // WRF 16933-16955.  Snow self-collection removes only number; the shared
    // snow-number limiter later rescales csacs together with its other sinks.
    if (!(s.qs > QS_MIN) || !(s.ns > CXMIN)) return;
    const float temperature_c = s.temperature - 273.15f;
    float efficiency = 0.0f;
    if (temperature_c < 0.0f && temperature_c >= -15.0f) {
        if (temperature_c > -15.0f && temperature_c < -10.0f) {
            efficiency = 0.5f * expf(-0.5f)
                * (temperature_c + 15.0f) / 5.0f;
        } else if (temperature_c >= -10.0f) {
            efficiency = 0.5f
                * expf(0.05f * fminf(temperature_c, 0.0f));
        }
    }
    if (!(efficiency > 0.0f)) return;

    const float mean_volume = fmaxf(
        SNOW_VMIN,
        rho * s.qs / (particles.snow_density * s.ns));
    const float swept_volume_cap =
        4.0f * (1.0f / PI) / 3.0f * (0.02f * 0.02f * 0.02f);
    const float collected_volume = fminf(mean_volume, swept_volume_cap);
    const float raw = (float)(
        (double)0.104f * (double)5.78e3f * (double)efficiency
        * (double)(s.ns * s.ns) * (double)collected_volume);
    const float dt_inverse = (float)(1.0 / (double)dt);
    r->csacs = fminf(
        raw,
        (float)(0.1 * (double)s.ns * (double)dt_inverse));
}

__device__ __forceinline__ float liquid_heat_capacity(float temperature) {
    if (temperature < 273.15f) {
        const float celsius =
            fminf(273.15f, fmaxf(233.15f, temperature)) - 273.15f;
        const float offset = celsius - 35.0f;
        return 4203.1548f + 1.30572e-2f * offset * offset
            + 1.60056e-5f * offset * offset * offset * offset;
    }
    const float celsius =
        fminf(308.15f, fmaxf(273.15f, temperature)) - 273.15f;
    return 4243.1688f + 3.47104e-1f * celsius * celsius;
}

__device__ __forceinline__ void diagnose_melting(
    const State& s,
    const ParticleProperties& particles,
    float rho,
    float pressure,
    float layer_depth,
    float dt,
    Rates* r)
{
    // WRF 18534-18904.  Melt rates must exist before frozen-vapor diagnosis:
    // the latter suppresses category sublimation when that category melts.
    // WRF's operative melt/ventilation branches use mass thresholds only;
    // retain active low-positive number moments normalized by setvtz.
    if (!(s.temperature > 273.15f)) return;
    const float dt_inverse = (float)(1.0 / (double)dt);
    const float temperature_c = s.temperature - 273.15f;
    const float density_factor = sqrtf(1.225f / fmaxf(0.05f, rho));

    float snow_diameter = 1.0e-9f;
    float snow_ventilation = 0.0f;
    if (s.qs > QS_MIN) {
        const float snow_volume = fmaxf(
            SNOW_VMIN,
            rho * s.qs / (particles.snow_density * s.ns));
        snow_diameter = powf(snow_volume * (6.0f / PI), 1.0f / 3.0f);
        const float snow_velocity = 11.9495f * density_factor
            * powf(snow_volume, 0.14f);
        const float dynamic_viscosity = 1.832e-5f
            * (416.16f / (s.temperature + 120.0f))
            * powf(s.temperature / 296.0f, 1.5f);
        const float diffusivity = 2.11e-5f
            * powf(s.temperature / 273.15f, 1.94f)
            * (101325.0f / pressure);
        const float kinematic = dynamic_viscosity / rho;
        const float ventilation_factor = powf(
            kinematic / diffusivity, 1.0f / 3.0f)
            * powf(kinematic, -0.5f);
        snow_ventilation = 0.65f + 0.44f * ventilation_factor
            * sqrtf(snow_velocity * snow_diameter);
    }

    float graupel_volume = GRAUPEL_VMIN;
    float graupel_diameter = 1.0e-9f;
    float graupel_characteristic = 1.0e-9f;
    if (s.qg > QD_MIN) {
        graupel_volume = fmaxf(
            GRAUPEL_VMIN,
            rho * s.qg / (particles.graupel_density * s.ng));
        graupel_diameter =
            powf(graupel_volume * (6.0f / PI), 1.0f / 3.0f);
        graupel_characteristic =
            powf(6.0f, -1.0f / 3.0f) * graupel_diameter;
    }
    float hail_volume = HAIL_VMIN;
    float hail_diameter = 1.0e-9f;
    float hail_characteristic = 1.0e-9f;
    if (s.qh > QD_MIN) {
        hail_volume = fmaxf(
            HAIL_VMIN,
            rho * s.qh / (particles.hail_density * s.nh));
        hail_diameter = powf(hail_volume * (6.0f / PI), 1.0f / 3.0f);
        hail_characteristic =
            powf(24.0f, -1.0f / 3.0f) * hail_diameter;
    }

    const float dynamic_viscosity = 1.832e-5f
        * (416.16f / (s.temperature + 120.0f))
        * powf(s.temperature / 296.0f, 1.5f);
    const float kinematic = dynamic_viscosity / rho;
    const float diffusivity = 2.11e-5f
        * powf(s.temperature / 273.15f, 1.94f)
        * (101325.0f / pressure);
    const float ventilation_factor = powf(
        kinematic / diffusivity, 1.0f / 3.0f)
        * powf(kinematic, -0.5f);
    const float conductivity =
        2.43e-2f * dynamic_viscosity / 1.718e-5f;
    const float lv = latent_vapor(s.temperature);
    const float lf = latent_fusion(s.temperature);
    const float cw = liquid_heat_capacity(s.temperature);
    const float fmlt1 = 2.0f * PI
        * (lv * diffusivity * (380.0f / pressure - s.qv)
            - conductivity * temperature_c / rho)
        / lf;
    const float fmlt2 = -cw * temperature_c / lf;

    float graupel_ventilation = 0.0f;
    if (s.qg > QD_MIN) {
        const float drag = fminf(
            1.2f,
            fmaxf(0.45f, 0.45f + 0.55f
                * (800.0f - fminf(
                    800.0f, fmaxf(170.0f, particles.graupel_density)))
                / 630.0f));
        graupel_ventilation = 0.78f * tgammaf(2.0f)
            + 0.308f * tgammaf(2.75f)
                * powf(4.0f * 9.8f / (3.0f * drag), 0.25f)
                * ventilation_factor
                * powf(particles.graupel_density / rho, 0.25f)
                * powf(graupel_characteristic, 0.75f);
    }
    float hail_ventilation = 0.0f;
    if (s.qh > QD_MIN) {
        float coefficient, exponent;
        dense_fall_coefficients(
            particles.hail_density, &coefficient, &exponent);
        // setvtz already caps the corresponding hail fall moments by dz/dt;
        // the analytic ventilation is otherwise the Ferrier alpha=1 branch.
        (void)layer_depth;
        hail_ventilation = 1.56f
            + tgammaf(3.5f + 0.5f * exponent)
                * 0.308f * ventilation_factor
                * powf(
                    hail_characteristic,
                    0.5f + 0.5f * exponent)
                * sqrtf(coefficient * density_factor);
    }

    if (s.qs > QS_MIN) {
        const float c1sw = tgammaf(0.5333333333333333f)
            * powf(0.2f, -1.0f / 3.0f) / tgammaf(0.2f);
        r->qsmlr = fminf(
            c1sw * fmlt1 * s.ns * snow_ventilation * snow_diameter,
            0.0f);
        r->qsmlr = fmaxf(r->qsmlr, -0.7f * s.qs * dt_inverse);
        r->csmlr = (s.ns / s.qs) * r->qsmlr;
        // Default gamma-shape conversion from snow donor number to rain.
        r->csmlrr = r->csmlr / 0.30f;
    }
    if (s.qg > QD_MIN) {
        const float raw = fminf(
            fmlt1 * s.ng * graupel_ventilation
                * graupel_characteristic
                + fmlt2 * (r->qhacrmlr + r->qhacwmlr),
            0.0f);
        if (raw < 0.0f && particles.graupel_density < 900.0f) {
            const float available =
                (1.0f - particles.graupel_density / 900.0f)
                * (s.vg + rho * raw / particles.graupel_density)
                * dt_inverse;
            const float refrozen = -rho * raw / 900.0f;
            r->vhsoak = fminf(available, refrozen);
        }
        r->qhmlr = fmaxf(raw, -0.95f * s.qg * dt_inverse);
        r->chmlr = (s.ng / (s.qg + 1.0e-20f)) * r->qhmlr;
        const float maximum_rain_mass =
            1000.0f * 0.523599f * (6.0e-3f * 6.0e-3f * 6.0e-3f);
        const float a = -rho * r->qhmlr / fminf(
            maximum_rain_mass,
            particles.graupel_density * graupel_volume);
        const float three_mm_volume =
            0.523599f * (3.0e-3f * 3.0e-3f * 3.0e-3f);
        const float b = -rho * r->qhmlr / (1000.0f * three_mm_volume);
        const float interpolation = a
                * (20.0e-3f - graupel_diameter) / 12.0e-3f
            + b * (graupel_diameter - 8.0e-3f) / 12.0e-3f;
        r->chmlrr = -fmaxf(a, fminf(b, interpolation));
        r->vhmlr = r->qhmlr;
    }

    if (s.qh > QD_MIN) {
        const float raw = fminf(
            fmlt1 * s.nh * hail_ventilation * hail_characteristic
                + fmlt2 * (r->qhlacrmlr + r->qhlacwmlr),
            0.0f);
        if (raw < 0.0f && particles.hail_density < 900.0f) {
            const float available =
                (1.0f - particles.hail_density / 900.0f)
                * (s.vh + rho * raw / particles.hail_density)
                * dt_inverse;
            const float refrozen = -rho * raw / 900.0f;
            r->vhlsoak = fminf(available, refrozen);
        }
        r->qhlmlr = fmaxf(raw, -0.95f * s.qh * dt_inverse);
        r->chlmlr = (s.nh / (s.qh + 1.0e-20f)) * r->qhlmlr;
        const float maximum_rain_mass =
            1000.0f * 0.523599f * (6.0e-3f * 6.0e-3f * 6.0e-3f);
        const float a = -rho * r->qhlmlr / fminf(
            maximum_rain_mass,
            particles.hail_density * hail_volume);
        const float three_mm_volume =
            0.523599f * (3.0e-3f * 3.0e-3f * 3.0e-3f);
        const float b = -rho * r->qhlmlr / (1000.0f * three_mm_volume);
        const float interpolation = a
                * (20.0e-3f - hail_diameter) / 12.0e-3f
            + b * (hail_diameter - 8.0e-3f) / 12.0e-3f;
        r->chlmlrr = -fmaxf(a, fminf(b, interpolation));
        r->vhlmlr = r->qhlmlr;
    }
}

__device__ __forceinline__ void diagnose_wet_growth_shedding(
    const State& s,
    const ParticleProperties& particles,
    float rho,
    float pressure,
    float dt,
    Rates* r)
{
    // WRF 19436-19769.  qhwet/qhlwet and the shedding decision are formed
    // from the original dry collection rates.  Only afterward can wet growth
    // rewrite the ice/snow collection efficiencies and vapor contributors.
    // As in WRF, active dense categories are mass-gated here; setvtz already
    // guarantees their number moments are positive even below CXMIN.
    const float dt_inverse = (float)(1.0 / (double)dt);
    const float temperature_c = s.temperature - 273.15f;
    const float density_factor = sqrtf(1.225f / fmaxf(0.05f, rho));
    const float dynamic_viscosity = 1.832e-5f
        * (416.16f / (s.temperature + 120.0f))
        * powf(s.temperature / 296.0f, 1.5f);
    const float kinematic = dynamic_viscosity / rho;
    const float diffusivity = 2.11e-5f
        * powf(s.temperature / 273.15f, 1.94f)
        * (101325.0f / pressure);
    const float ventilation_factor = powf(
        kinematic / diffusivity, 1.0f / 3.0f)
        * powf(kinematic, -0.5f);
    const float conductivity =
        2.43e-2f * dynamic_viscosity / 1.718e-5f;
    const float lv = latent_vapor(s.temperature);
    const float lf = latent_fusion(s.temperature);
    const float cw = liquid_heat_capacity(s.temperature);
    const float bounded_ice_celsius =
        fminf(273.15f, fmaxf(233.15f, s.temperature)) - 273.15f;
    const float ci =
        (2.118636f + 0.007371f * bounded_ice_celsius) * 1.0e3f;
    const float denominator = lf + cw * temperature_c;
    const float fwet1 = 2.0f * PI
        * (lv * diffusivity * rho * (380.0f / pressure - s.qv)
            - conductivity * temperature_c)
        / (rho * denominator);
    const float fwet2 = 1.0f - ci * temperature_c / denominator;

    float graupel_diameter = 1.0e-9f;
    float graupel_characteristic = 1.0e-9f;
    float graupel_ventilation = 0.0f;
    if (s.qg > QD_MIN) {
        const float volume = fmaxf(
            GRAUPEL_VMIN,
            rho * s.qg / (particles.graupel_density * s.ng));
        graupel_diameter = powf(volume * (6.0f / PI), 1.0f / 3.0f);
        graupel_characteristic =
            powf(6.0f, -1.0f / 3.0f) * graupel_diameter;
        const float drag = fminf(
            1.2f,
            fmaxf(0.45f, 0.45f + 0.55f
                * (800.0f - fminf(
                    800.0f, fmaxf(170.0f, particles.graupel_density)))
                / 630.0f));
        graupel_ventilation = 0.78f * tgammaf(2.0f)
            + 0.308f * tgammaf(2.75f)
                * powf(4.0f * 9.8f / (3.0f * drag), 0.25f)
                * ventilation_factor
                * powf(particles.graupel_density / rho, 0.25f)
                * powf(graupel_characteristic, 0.75f);
    }
    float hail_diameter = 1.0e-9f;
    float hail_characteristic = 1.0e-9f;
    float hail_ventilation = 0.0f;
    if (s.qh > QD_MIN) {
        const float volume = fmaxf(
            HAIL_VMIN,
            rho * s.qh / (particles.hail_density * s.nh));
        hail_diameter = powf(volume * (6.0f / PI), 1.0f / 3.0f);
        hail_characteristic =
            powf(24.0f, -1.0f / 3.0f) * hail_diameter;
        float coefficient, exponent;
        dense_fall_coefficients(
            particles.hail_density, &coefficient, &exponent);
        hail_ventilation = 1.56f
            + tgammaf(3.5f + 0.5f * exponent)
                * 0.308f * ventilation_factor
                * powf(
                    hail_characteristic,
                    0.5f + 0.5f * exponent)
                * sqrtf(coefficient * density_factor);
    }

    const float graupel_dry =
        r->qhaci + r->qhacs + r->qhacr + r->qhacw;
    const float hail_dry =
        r->qhlaci + r->qhlacs + r->qhlacr + r->qhlacw;
    float graupel_wet = graupel_dry;
    float hail_wet = hail_dry;
    if (s.temperature > 243.15f && s.temperature < 273.15f) {
        graupel_wet = fmaxf(
            graupel_characteristic * graupel_ventilation * s.ng * fwet1
                + fwet2 * (r->qhaci + r->qhacs),
            0.0f);
        hail_wet = fmaxf(
            hail_characteristic * hail_ventilation * s.nh * fwet1
                + fwet2 * (r->qhlaci + r->qhlacs),
            0.0f);
    }
    r->qhshr = fminf(0.0f, graupel_wet - graupel_dry);
    r->qhlshr = fminf(0.0f, hail_wet - hail_dry);
    if (s.temperature < 243.15f) {
        r->qhshr = 0.0f;
        r->qhlshr = 0.0f;
    }
    if (s.temperature > 273.15f) {
        r->qsshr = -r->qsacr - r->qsacw;
        r->qhshr = -r->qhacw - r->qhacr;
        r->qhlshr = -r->qhlacw - r->qhlacr;
        r->vhshdr = -r->vhacw - r->vhacr;
        r->vhlshdr = -r->vhlacw - r->vhlacr;
        graupel_wet = 0.0f;
        hail_wet = 0.0f;
    }

    // WRF 19606-19615 final snow decision: shedding and vapor exchange are
    // mutually exclusive.  This occurs after frozen-vapor diagnosis, so the
    // already limited snow deposition/sublimation rates are explicitly
    // cleared whenever warm snow collection sheds to rain.
    if (r->qsshr < 0.0f) {
        r->qsdpv = 0.0f;
        r->qssbv = 0.0f;
    } else {
        r->qsshr = 0.0f;
    }

    r->wetgrowth_g =
        r->qhshr < 0.0f && s.temperature < 273.15f ? 1.0f : 0.0f;
    r->wetgrowth_h =
        r->qhlshr < 0.0f && s.temperature < 273.15f ? 1.0f : 0.0f;
    r->wetsurface_g = (r->wetgrowth_g > 0.0f
        || (r->qhmlr < -QD_MIN && s.temperature > 273.15f)) ? 1.0f : 0.0f;
    r->wetsurface_h = (r->wetgrowth_h > 0.0f
        || (r->qhlmlr < -QD_MIN && s.temperature > 273.15f)) ? 1.0f : 0.0f;

    constexpr float MASS_FACTOR_SHED = 4.5f;
    float graupel_shed_volume =
        0.523599f * (1.0e-3f * 1.0e-3f * 1.0e-3f);
    if (s.qg > QD_MIN) {
        const float weighted = 3.0f * graupel_characteristic;
        if (weighted > 20.0e-3f) {
            graupel_shed_volume =
                0.523599f * (1.5e-3f * 1.5e-3f * 1.5e-3f)
                / MASS_FACTOR_SHED;
        } else if (weighted > 8.0e-3f) {
            graupel_shed_volume =
                0.523599f * (3.0e-3f * 3.0e-3f * 3.0e-3f)
                / MASS_FACTOR_SHED;
        } else {
            graupel_shed_volume = fminf(
                0.523599f * (6.0e-3f * 6.0e-3f * 6.0e-3f),
                (6.0f / PI) * particles.graupel_density * 0.001f
                    * weighted * weighted * weighted)
                / MASS_FACTOR_SHED;
        }
    }
    float hail_shed_volume =
        0.523599f * (1.0e-3f * 1.0e-3f * 1.0e-3f);
    if (s.qh > QD_MIN) {
        const float weighted = 4.0f * hail_characteristic;
        if (weighted > 20.0e-3f) {
            hail_shed_volume =
                0.523599f * (1.5e-3f * 1.5e-3f * 1.5e-3f)
                / MASS_FACTOR_SHED;
        } else if (weighted > 8.0e-3f) {
            hail_shed_volume =
                0.523599f * (3.0e-3f * 3.0e-3f * 3.0e-3f)
                / MASS_FACTOR_SHED;
        } else {
            hail_shed_volume = fminf(
                0.523599f * (6.0e-3f * 6.0e-3f * 6.0e-3f),
                (6.0f / PI) * particles.hail_density * 0.001f
                    * weighted * weighted * weighted)
                / MASS_FACTOR_SHED;
        }
    }
    r->chshrr = rho * r->qhshr / (1000.0f * graupel_shed_volume);
    r->chlshrr = rho * r->qhlshr / (1000.0f * hail_shed_volume);
    r->qrshr = r->qsshr + r->qhshr + r->qhlshr;
    // ipconc=5 has rzxh=1 and rzxhl=0.4375 (alphar=alphah=0,
    // alphahl=1, imurain=1, and no predicted reflectivity moments).
    r->crshr = r->chshrr + r->chlshrr / 0.4375f;

    if (r->wetgrowth_g > 0.0f) {
        r->qhdpv = 0.0f;
        r->chdpv = 0.0f;
        r->qhaci = fminf(r->qhaci0, 0.1f * s.qi * dt_inverse);
        r->chaci = fminf(r->chaci0, 0.1f * s.ni * dt_inverse);
        r->qhacs = fminf(r->qhacs0, 0.1f * s.qs * dt_inverse);
        r->chacs = fminf(r->chacs0, 0.1f * s.ns * dt_inverse);
        r->vhacw = rho * r->qhacw / 900.0f;
        r->vhacr = rho * r->qhacr / 900.0f;
        const float available = particles.graupel_density < 900.0f
            ? (1.0f - particles.graupel_density / 900.0f)
                * s.vg * dt_inverse
            : 0.0f;
        const float refrozen = rho * graupel_wet / 900.0f;
        r->vhsoak = fminf(available, refrozen);
        r->vhshdr = fminf(
            0.0f, refrozen - r->vhacw - r->vhacr);
        r->wetsurface_g = 1.0f;
    }
    if (r->wetgrowth_h > 0.0f) {
        r->qhldpv = 0.0f;
        r->chldpv = 0.0f;
        r->qhlaci = fminf(r->qhlaci0, 0.1f * s.qi * dt_inverse);
        r->chlaci = fminf(r->chlaci0, 0.1f * s.ni * dt_inverse);
        r->qhlacs = fminf(r->qhlacs0, 0.1f * s.qs * dt_inverse);
        r->chlacs = fminf(r->chlacs0, 0.1f * s.ns * dt_inverse);
        r->vhlacw = rho * r->qhlacw / 900.0f;
        r->vhlacr = rho * r->qhlacr / 900.0f;
        const float available = particles.hail_density < 900.0f
            ? (1.0f - particles.hail_density / 900.0f)
                * s.vh * dt_inverse
            : 0.0f;
        const float refrozen = rho * hail_wet / 900.0f;
        r->vhlsoak = fminf(available, refrozen);
        r->vhlshdr = fminf(
            0.0f, refrozen - r->vhlacw - r->vhlacr);
        r->wetsurface_h = 1.0f;
    }
}

__device__ __forceinline__ void diagnose_crystal_to_snow(
    const State& s,
    const ParticleProperties& particles,
    float rho,
    Rates* r)
{
    // WRF 19335-19388, default iscni=4.  This conversion is diagnosed after
    // frozen vapor but before wet-growth can rewrite any collection rates.
    if (!(s.qi > QI_MIN) || !(s.ni > CXMIN) || !(r->qidpv > 0.0f)) {
        return;
    }
    const float ice_mass = fmaxf(rho * s.qi / s.ni, ICE_MMIN);
    const float ice_diameter = 0.1871f * powf(ice_mass, 0.3429f);
    if (!(ice_diameter >= 100.0e-6f)) return;
    const float fraction = fminf(0.5f, ice_diameter / 200.0e-6f);
    r->qscni = fraction * r->qidpv;
    // WRF's fscni multiplier is one here.  The diameter fraction already
    // entered qscni and must not be applied a second time to its number rate.
    r->cscni = rho * r->qscni / fmaxf(
        particles.snow_density * SNOW_VMIN, ice_mass);
}

__device__ __forceinline__ void diagnose_ice_snow_to_graupel(
    const State& s,
    const ParticleProperties& particles,
    float rho,
    Rates* r)
{
    // WRF 19774-19844 and 20184-20301.  Donor and receiver number rates are
    // intentionally distinct and remain stale if later cloud/snow limiters
    // rescale the collection rates that caused the conversion.
    const float temperature_c = s.temperature - 273.15f;
    const float density_factor = sqrtf(1.225f / fmaxf(0.05f, rho));
    float cloud_diameter = 1.0e-9f;
    if (s.qc > QC_MIN && s.nc > CXMIN) {
        const float cloud_volume = fminf(
            CLOUD_VMAX,
            fmaxf(CLOUD_VMIN, rho * s.qc / (1000.0f * s.nc)));
        cloud_diameter =
            powf(cloud_volume * (6.0f / PI), 1.0f / 3.0f);
    }
    if (s.temperature < 273.0f && s.qi > QI_MIN && s.ni > CXMIN
            && r->qiacw - r->qidpv > 0.0f) {
        const float ice_mass = fmaxf(rho * s.qi / s.ni, ICE_MMIN);
        const float ice_volume = ice_mass / 900.0f;
        const float ice_velocity = 47.6273f * density_factor
            * powf(ice_volume, 0.18333f) * 1.091937899589539f;
        float rime_density = 300.0f * powf(
            -(0.5f * 1.0e6f * cloud_diameter)
                * (0.60f * ice_velocity) / temperature_c,
            0.44f);
        rime_density = fminf(900.0f, fmaxf(170.0f, rime_density));
        if (rime_density >= 200.0f) {
            const float converted_density = fmaxf(
                170.0f, 0.5f * (900.0f + rime_density));
            r->qhcni = r->qiacw - r->qidpv;
            r->chcni = s.ni * r->qhcni / s.qi;
            r->chcnih = fminf(
                r->chcni,
                rho * r->qhcni / (converted_density * GRAUPEL_VMIN));
            r->vhcni = rho * r->qhcni / converted_density;
        }
    }

    if (s.temperature < 273.0f && s.qs > QS_MIN && s.ns > CXMIN
            && r->qsacw > 0.0f && r->qsacw - r->qsdpv > 0.0f) {
        const float snow_volume = fmaxf(
            SNOW_VMIN,
            rho * s.qs / (particles.snow_density * s.ns));
        const float snow_velocity = 11.9495f * density_factor
            * powf(snow_volume, 0.14f);
        float rime_density = 300.0f * powf(
            // WRF 20227-20231 intentionally uses the cloud diameter in the
            // snow-to-graupel rime-density diagnosis.
            -(0.5f * 1.0e6f * cloud_diameter)
                * (0.60f * snow_velocity) / temperature_c,
            0.44f);
        // Exact snow-conversion quirk: upper clamp only, no 170 lower clamp.
        rime_density = fminf(900.0f, rime_density);
        if (rime_density >= 200.0f) {
            const float converted_density = fmaxf(
                170.0f,
                0.5f * (particles.snow_density + rime_density));
            r->qhcns = r->qsacw - r->qsdpv;
            r->chcns = s.ns * r->qhcns / s.qs;
            r->chcnsh = fminf(
                r->chcns,
                rho * r->qhcns / (converted_density * GRAUPEL_VMIN));
            r->vhcns = rho * r->qhcns / converted_density;
        }
    }
}

__device__ __forceinline__ void diagnose_graupel_to_hail(
    const State& s,
    const ParticleProperties& particles,
    float rho,
    float pressure,
    float dt,
    Rates* r)
{
    // WRF 19847-20089, default ihlcnh=3.  The wet-growth diameter iteration
    // uses the unscaled Milbrandt-Morrison fall-speed law.  Conversion then
    // takes the upper gamma tails above dg0 from the immutable graupel state.
    if (!(s.qg > QD_MIN) || !(s.ng > CXMIN)) return;

    const float dt_inverse = (float)(1.0 / (double)dt);
    const float temperature_c = s.temperature - 273.15f;
    const bool riming_candidate =
        (r->qhacw + r->qhacr) * dt > QD_MIN
        && s.qg > 0.1e-3f && s.temperature <= 271.15f;
    float dg0 = 0.1501f;

    if ((riming_candidate && s.temperature > 242.0f)
            || (r->wetgrowth_g > 0.0f && s.qg > 0.1e-3f)) {
        float cloud_efficiency = 0.0f;
        float cloud_diameter = 1.0e-9f;
        float cloud_velocity = 0.0f;
        const float dynamic_viscosity = 1.832e-5f
            * (416.16f / (s.temperature + 120.0f))
            * powf(s.temperature / 296.0f, 1.5f);
        if (s.qc > QC_MIN && s.nc > CXMIN) {
            const float cloud_volume = fminf(
                CLOUD_VMAX,
                fmaxf(CLOUD_VMIN, rho * s.qc / (1000.0f * s.nc)));
            cloud_diameter = powf(
                cloud_volume * (6.0f / PI), 1.0f / 3.0f);
            const float cloud_radius = 0.5f * cloud_diameter;
            cloud_velocity = 2.0f * 9.8f * 1000.0f
                * cloud_radius * cloud_radius / (9.0f * dynamic_viscosity);
            cloud_efficiency = fminf(
                0.9f,
                fminf(
                    -0.27544f + cloud_radius
                        * (0.26249e6f + cloud_radius
                            * (-1.8896e10f
                                + cloud_radius * 4.4626e14f)),
                    1.0f));
            if (cloud_diameter < 2.4e-6f) cloud_efficiency = 0.0f;
        }

        float rain_efficiency = 0.0f;
        float rain_velocity = 0.0f;
        if (s.qr > QR_MIN && s.nr > CXMIN) {
            const float rain_volume = fminf(
                RAIN_VMAX,
                fmaxf(RAIN_VMIN, rho * s.qr / (1000.0f * s.nr)));
            const float rain_diameter = powf(
                rain_volume * (6.0f / PI), 1.0f / 3.0f);
            const float rain_characteristic =
                powf(6.0f, -1.0f / 3.0f) * rain_diameter;
            const float graupel_volume = fmaxf(
                GRAUPEL_VMIN,
                rho * s.qg / (particles.graupel_density * s.ng));
            const float graupel_diameter = powf(
                graupel_volume * (6.0f / PI), 1.0f / 3.0f);
            rain_efficiency = fminf(
                1.0f,
                expf(-40.0e-6f / rain_diameter)
                    * expf(-40.0e-6f / graupel_diameter));
            const float density_factor =
                sqrtf(1.225f / fmaxf(0.05f, rho));
            rain_velocity = density_factor * 10.0f
                * (1.0f - powf(
                    1.0f + 516.575f * rain_characteristic, -4.0f));
        }

        float ice_efficiency = 0.0f;
        float ice_velocity = 0.0f;
        if (s.qi > QI_MIN && s.ni > CXMIN) {
            const float ice_mass = fmaxf(rho * s.qi / s.ni, ICE_MMIN);
            const float ice_volume = ice_mass / 900.0f;
            ice_efficiency = fminf(
                1.0f,
                fmaxf(0.0f, 0.1f * expf(0.1f * fminf(temperature_c, 0.0f))));
            ice_velocity = 47.6273f
                * sqrtf(1.225f / fmaxf(0.05f, rho))
                * powf(ice_volume, 0.18333f)
                * tgammaf(2.18333f);
        }

        const float initial_denominator =
            1.1e4f * rho
                * (cloud_efficiency * s.qc + rain_efficiency * s.qr)
            - 1.3e3f * rho * s.qi + 1.0f;
        float dwr = 1.0e30f;
        if (initial_denominator > 1.0e-20f) {
            dwr = 0.01f * (
                expf(fminf(70.0f, -temperature_c / initial_denominator))
                - 1.0f);
        }

        float diameter = dwr;
        if (dwr < 0.2f && dwr > 0.0f
                && rho * (s.qc + s.qr) > 1.0e-4f) {
            const float kinematic_viscosity = dynamic_viscosity / rho;
            const float vapor_diffusivity = 2.11e-5f
                * powf(s.temperature / 273.15f, 1.94f)
                * (101325.0f / pressure);
            const float conductivity =
                2.43e-2f * dynamic_viscosity / 1.718e-5f;
            const float thermal_diffusivity =
                conductivity / (1004.0f * rho);
            const float prandtl =
                kinematic_viscosity / thermal_diffusivity;
            const float density_factor =
                sqrtf(1.225f / fmaxf(0.05f, rho));
            const float sqrt_density_factor = sqrtf(density_factor);
            const float heat_ventilation = sqrt_density_factor
                * powf(prandtl, 1.0f / 3.0f)
                * powf(kinematic_viscosity, -0.5f);
            const float h1 = -conductivity * temperature_c
                - latent_vapor(s.temperature) * vapor_diffusivity * rho
                    * (s.qv - 380.0f / pressure);
            const float bounded_ice_celsius =
                fminf(273.15f, fmaxf(233.15f, s.temperature)) - 273.15f;
            const float ice_heat_capacity =
                (2.118636f + 0.007371f * bounded_ice_celsius) * 1.0e3f;
            const float h2 = ice_efficiency * s.qi * rho
                * ice_heat_capacity * temperature_c;
            const float h3 = fmaxf(0.0f, cloud_efficiency) * s.qc;
            const float h4 = rain_efficiency * s.qr;
            const float heat_denominator =
                latent_fusion(s.temperature)
                + liquid_heat_capacity(s.temperature) * temperature_c;
            float fall_coefficient;
            float fall_exponent;
            dense_fall_coefficients(
                particles.graupel_density,
                &fall_coefficient,
                &fall_exponent);

            for (int iteration = 0; iteration < 10; ++iteration) {
                diameter = fmaxf(diameter, 1.0e-4f);
                const float previous = diameter;
                const float graupel_velocity = fall_coefficient
                    * powf(diameter, fall_exponent);
                const float ventilation_argument = heat_ventilation
                    * sqrt_density_factor
                    * sqrtf(diameter * graupel_velocity);
                const float heat_factor = ventilation_argument > 1.4f
                    ? 0.78f + 0.308f * ventilation_argument
                    : 1.0f + 0.108f * ventilation_argument
                        * ventilation_argument;
                const float denominator = (
                    fmaxf(0.001f, graupel_velocity - cloud_velocity) * h3
                    + fmaxf(0.001f, graupel_velocity - rain_velocity) * h4)
                        * rho * heat_denominator
                    + fmaxf(0.001f, graupel_velocity - ice_velocity) * h2;
                diameter = 8.0f * heat_factor * h1 / denominator;
                if (fabsf(previous - diameter) / previous < 0.05f
                        || (iteration >= 3 && diameter > 0.15f)) {
                    break;
                }
            }
        }
        dg0 = fminf(15.0e-3f, fmaxf(diameter, 5.0e-3f));
    } else if (s.qg > 0.1e-3f && s.temperature <= 271.15f) {
        dg0 = 15.0e-3f;
    }

    // ihlcnh=3 retains a secondary large-graupel route even when the wet
    // growth iteration itself did not return a usable diameter.
    if (riming_candidate) dg0 = fminf(dg0, 0.1499f);
    const bool wet_growth_test = dg0 > 0.0f && dg0 < 0.15f;
    if (!(wet_growth_test && r->qhacw * dt > QD_MIN
            && s.temperature < 271.15f && s.qg > 0.1e-3f)) {
        return;
    }

    const float mean_volume = fmaxf(
        GRAUPEL_VMIN,
        rho * s.qg / (particles.graupel_density * s.ng));
    const float mean_diameter = powf(
        mean_volume * (6.0f / PI), 1.0f / 3.0f);
    const float characteristic =
        powf(6.0f, -1.0f / 3.0f) * mean_diameter;
    float number_fraction;
    float mass_fraction;
    rain_tail_fractions(
        fminf(100.0f, dg0 / characteristic),
        &number_fraction,
        &mass_fraction);
    const float converted_mass = s.qg * mass_fraction;
    r->qhlcnh = converted_mass * dt_inverse;
    if (!(converted_mass > 10.0f * QD_MIN)) {
        r->qhlcnh = 0.0f;
        return;
    }
    r->chlcnh = s.ng * number_fraction * dt_inverse;
    r->chlcnhhl = r->chlcnh;
    r->vhlcnh = rho * r->qhlcnh / particles.graupel_density;
    r->vhlcnhl = rho * r->qhlcnh
        / fmaxf(500.0f, particles.graupel_density);
}

__device__ __forceinline__ void diagnose_hallett_mossop(
    const State& s,
    float rho,
    Rates* r)
{
    // WRF 20445-20639, default itype1=0/itype2=2.  H-M reads the exact
    // source-ordered wet-surface flags produced by the shedding decision.
    if (!(s.qc > QC_MIN) || !(s.nc > CXMIN)
            || s.temperature < 265.15f || s.temperature > 271.15f) {
        return;
    }
    const float cloud_volume = fminf(
        CLOUD_VMAX,
        fmaxf(CLOUD_VMIN, rho * s.qc / (1000.0f * s.nc)));
    const float tail = expf(-7.23e-15f / cloud_volume) / 250.0f;
    const float temperature_c = s.temperature - 273.15f;
    const float temperature_factor = fminf(
        1.0f,
        fmaxf(0.0f,
            -0.11f * temperature_c * temperature_c
                - 1.1f * temperature_c - 1.7f));
    constexpr float SPLINTER_MASS = 6.62e-11f;
    if (s.qg > QD_MIN && !(r->wetsurface_g > 0.0f)) {
        r->chmul1 = temperature_factor * tail * r->chacw;
        r->qhmul1 = SPLINTER_MASS * r->chmul1 / rho;
    }
    if (s.qh > QD_MIN && !(r->wetsurface_h > 0.0f)) {
        r->chlmul1 = temperature_factor * tail * r->chlacw;
        r->qhlmul1 = SPLINTER_MASS * r->chlmul1 / rho;
    }
}

__device__ __forceinline__ void diagnose_primary_ice(
    const State& s,
    float rho,
    float pressure,
    float velocity,
    float layer_depth,
    float target_minus,
    float target_plus,
    float vertical_span,
    float dt,
    Rates* r)
{
    // WRF 20708-20823, default icenucopt=1 and column-crystal routing.
    if (!(s.temperature < 268.15f) || !(s.ni < 1.0e6f)
            || !(velocity > 0.0f) || !(layer_depth > 0.0f)
            || !(vertical_span > 0.0f)) {
        return;
    }
    int table_index = (int)((s.temperature - 163.15f) / 0.002f + 1.5f);
    table_index = min(1000001, max(1, table_index));
    const float table_temperature = __fadd_rn(
        163.15f, __fmul_rn((float)(table_index - 1), 0.002f));
    const float ice_saturation = (380.0f / pressure) * expf(
        21.87455f * (table_temperature - 273.15f)
        / (table_temperature - 7.66f));
    if (!(s.qv / ice_saturation > 1.0f)) return;
    const float target_gradient = fmaxf(target_plus - target_minus, 0.0f);
    if (!(target_gradient > 0.0f)) return;

    const float lv = latent_vapor(s.temperature);
    const float feedback = lv * lv / (1004.0f * 461.5f);
    const float vapor_limit = 0.25f * fmaxf(
        (s.qv - ice_saturation)
            / (1.0f + feedback * ice_saturation
                / (s.temperature * s.temperature)),
        0.0f);
    constexpr float initial_mass = 6.88e-13f;
    r->qiint = (initial_mass / rho) * velocity * target_gradient
        / (layer_depth * vertical_span);
    r->qiint = fminf(r->qiint, vapor_limit);
    r->ciint = r->qiint * rho / initial_mass;
    r->ciint = fminf(
        r->ciint, fmaxf(0.0f, 1.0e6f - s.ni) / dt);
    r->qiint = r->ciint * initial_mass / rho;
}

__device__ __forceinline__ void diagnose_frozen_vapor(
    const State& s,
    const ParticleProperties& particles,
    float rho,
    float pressure,
    float exner,
    float dt,
    Rates* r)
{
    // WRF 18915-19333.  All four frozen categories share one two-pass
    // saturation-adjustment budget.  The resulting mass rates are scaled
    // together before their number companions are derived.
    const float dt_inverse = (float)(1.0 / (double)dt);
    // Default iqis0=2 caps the initial ice table lookup above 273.65 K.
    // The test-adjustment's pass-one trial lookup below intentionally does
    // not use this cap, matching WRF's qisstmp update.
    const float ice_saturation = ice_saturation_mixing_ratio(
        s.temperature <= 273.65f ? s.temperature : 273.15f,
        pressure);
    const float lv = latent_vapor(s.temperature);
    const float lf = latent_fusion(s.temperature);
    const float ls = lv + lf;
    const float diffusivity = 2.11e-5f
        * powf(s.temperature / 273.15f, 1.94f)
        * (101325.0f / pressure);
    const float viscosity = 1.832e-5f
        * (416.16f / (s.temperature + 120.0f))
        * powf(s.temperature / 296.0f, 1.5f);
    const float kinematic = viscosity / rho;
    const float conductivity = 2.43e-2f * viscosity / 1.718e-5f;
    const float schmidt = kinematic / diffusivity;
    const float ventilation_factor = powf(schmidt, 1.0f / 3.0f)
        * powf(kinematic, -0.5f);
    const float resistance = ls * ls
        / (conductivity * 461.5f * s.temperature * s.temperature)
        + 1.0f / (rho * diffusivity * ice_saturation);
    const float vapor_growth = (4.0f * PI / rho)
        * (s.qv / ice_saturation - 1.0f) / resistance;

    float raw_ice = 0.0f;
    if (s.qi > QI_MIN && s.ni > 0.0f) {
        const float mean_mass = fmaxf(rho * s.qi / s.ni, ICE_MMIN);
        const float diameter = 0.1871f * powf(mean_mass, 0.3429f);
        const float reynolds =
            (1.258e4f * powf(diameter, 2.331f)
             + 5.662e4f * powf(diameter, 2.373f))
            / (0.8241f * powf(diameter, -0.042f) + 1.70f);
        const float argument = powf(schmidt, 1.0f / 3.0f)
            * sqrtf(reynolds / kinematic);
        const float ventilation = argument < 1.0f
            ? 1.0f + 0.14f * argument * argument
            : 0.86f + 0.28f * argument;
        const float length = 0.4764f * powf(diameter, 0.958f);
        const float eccentricity = fminf(0.99f, sqrtf(fmaxf(
            0.0f, 1.0f - length * length / (diameter * diameter))));
        const float capacitance = diameter * eccentricity
            / logf(fabsf((1.0f + eccentricity) / (1.0f - eccentricity)));
        raw_ice = vapor_growth * s.ni * ventilation * capacitance;
    }

    float raw_snow = 0.0f;
    if (s.qs > QS_MIN && s.ns > 0.0f) {
        const float mean_volume = fmaxf(
            SNOW_VMIN,
            rho * s.qs / (particles.snow_density * s.ns));
        const float diameter = powf(mean_volume * (6.0f / PI), 1.0f / 3.0f);
        const float fall_speed = 11.9495f
            * sqrtf(1.225f / fmaxf(0.05f, rho))
            * powf(mean_volume, 0.14f);
        const float ventilation = 0.65f + 0.44f * ventilation_factor
            * sqrtf(fall_speed * diameter);
        raw_snow = vapor_growth * s.ns * ventilation * (0.5f * diameter);
    }

    float raw_graupel = 0.0f;
    if (s.qg > QD_MIN && s.ng > 0.0f) {
        const float mean_volume = fmaxf(
            GRAUPEL_VMIN,
            rho * s.qg / (particles.graupel_density * s.ng));
        const float mean_volume_diameter =
            powf(mean_volume * (6.0f / PI), 1.0f / 3.0f);
        const float diameter = powf(6.0f, -1.0f / 3.0f)
            * mean_volume_diameter;
        const float drag = fmaxf(0.45f, fminf(
            1.2f,
            0.45f + 0.55f
                * (800.0f - fmaxf(
                    170.0f, fminf(800.0f, particles.graupel_density)))
                / 630.0f));
        const float drag_factor = powf(4.0f * 9.8f / (3.0f * drag), 0.25f);
        const float ventilation = 0.78f
            + 0.308f * 1.6083594560623169f * drag_factor
                * ventilation_factor
                * powf(particles.graupel_density / rho, 0.25f)
                * powf(diameter, 0.75f);
        raw_graupel = vapor_growth * s.ng * ventilation * (0.5f * diameter);
    }

    float raw_hail = 0.0f;
    if (s.qh > QD_MIN && s.nh > 0.0f) {
        const float mean_volume = fmaxf(
            HAIL_VMIN,
            rho * s.qh / (particles.hail_density * s.nh));
        const float mean_volume_diameter =
            powf(mean_volume * (6.0f / PI), 1.0f / 3.0f);
        const float diameter = powf(24.0f, -1.0f / 3.0f)
            * mean_volume_diameter;
        const float density_table[9] = {
            50.0f, 150.0f, 250.0f, 350.0f, 450.0f,
            550.0f, 650.0f, 750.0f, 850.0f};
        const float fall_a_table[9] = {
            62.923f, 94.122f, 114.74f, 131.21f, 145.26f,
            157.71f, 168.98f, 179.36f, 189.02f};
        const float fall_b_table[9] = {
            0.67819f, 0.63789f, 0.62197f, 0.61240f, 0.60572f,
            0.60066f, 0.59663f, 0.59330f, 0.59048f};
        int table = (int)((particles.hail_density - 50.0f) / 100.0f) + 1;
        table = min(9, max(1, table)) - 1;
        const float fraction = fmaxf(
            0.0f, 0.01f * (particles.hail_density - density_table[table]));
        float fall_a = fall_a_table[table];
        float fall_b = fall_b_table[table];
        if (table < 8) {
            fall_a += fraction * (fall_a_table[table + 1] - fall_a);
            fall_b += fraction * (fall_b_table[table + 1] - fall_b);
        }
        const float fall_factor = sqrtf(1.225f / fmaxf(0.05f, rho));
        const float ventilation = 1.56f
            + tgammaf(3.5f + 0.5f * fall_b) * 0.308f
                * ventilation_factor
                * powf(diameter, 0.5f + 0.5f * fall_b)
                * sqrtf(fall_a * fall_factor);
        raw_hail = vapor_growth * s.nh * ventilation * (0.5f * diameter);
    }

    float maximum_deposition = 0.0f;
    float maximum_sublimation = 0.0f;
    const float total_frozen = s.qi + s.qs + s.qg + s.qh;
    if (total_frozen > QI_MIN) {
        float trial_frozen = total_frozen;
        float trial_vapor = s.qv;
        float trial_theta = s.theta;
        float trial_saturation = ice_saturation;
        float saturation_feedback = 5807.6953f * (ls / 1004.0f);
        float denominator_temperature =
            (s.temperature - 7.66f) * (s.temperature - 7.66f);
        if (s.temperature >= 273.15f) {
            saturation_feedback = 4098.0258f * (lv / 1004.0f);
            denominator_temperature =
                (s.temperature - 35.86f) * (s.temperature - 35.86f);
        }
        #pragma unroll
        for (int iteration = 0; iteration < 2; ++iteration) {
            const float difference = trial_vapor - trial_saturation;
            float adjustment;
            if (difference < 0.0f) {
                adjustment = fmaxf(difference, -trial_frozen);
            } else {
                adjustment = difference
                    / (1.0f + saturation_feedback * trial_saturation
                        / denominator_temperature);
            }
            trial_vapor -= adjustment;
            trial_frozen += adjustment;
            trial_theta += (ls / (1004.0f * exner)) * adjustment;
            if (iteration == 0) {
                trial_vapor = fmaxf(trial_vapor, 0.0f);
                trial_frozen = fmaxf(trial_frozen, 0.0f);
                trial_saturation = ice_saturation_mixing_ratio(
                    trial_theta * exner, pressure);
            }
        }
        trial_frozen = fmaxf(trial_frozen, 0.0f);
        const float net_adjustment = trial_frozen - total_frozen;
        maximum_deposition = fmaxf(net_adjustment * dt_inverse, 0.0f);
        maximum_sublimation = fmaxf(-net_adjustment * dt_inverse, 0.0f);
    }

    r->qidpv = fmaxf(raw_ice, 0.0f);
    r->qsdpv = fmaxf(raw_snow, 0.0f);
    r->qhdpv = fmaxf(raw_graupel, 0.0f);
    r->qhldpv = fmaxf(raw_hail, 0.0f);
    r->qisbv = fmaxf(
        fminf(raw_ice, 0.0f),
        fminf(-0.1f * s.qi * dt_inverse, -0.5f * s.qi * dt_inverse));
    r->qssbv = (s.temperature < 273.15f || !(r->qsmlr < 0.0f))
        ? fmaxf(
            fminf(raw_snow, 0.0f),
            fminf(-0.1f * s.qs * dt_inverse, -0.5f * s.qs * dt_inverse))
        : 0.0f;
    r->qhsbv = (s.temperature < 273.15f || !(r->qhmlr < 0.0f))
        ? fmaxf(fminf(raw_graupel, 0.0f), -0.1f * s.qg * dt_inverse)
        : 0.0f;
    r->qhlsbv = (s.temperature < 273.15f || !(r->qhlmlr < 0.0f))
        ? fmaxf(fminf(raw_hail, 0.0f), -0.1f * s.qh * dt_inverse)
        : 0.0f;

    const float deposition_total =
        r->qidpv + r->qsdpv + r->qhdpv + r->qhldpv;
    if (deposition_total > maximum_deposition) {
        const float fraction = maximum_deposition / deposition_total;
        r->qidpv *= fraction;
        r->qsdpv *= fraction;
        r->qhdpv *= fraction;
        r->qhldpv *= fraction;
    }
    const float sublimation_total =
        r->qisbv + r->qssbv + r->qhsbv + r->qhlsbv;
    if (sublimation_total < -maximum_sublimation) {
        const float fraction = -maximum_sublimation / sublimation_total;
        r->qisbv *= fraction;
        r->qssbv *= fraction;
        r->qhsbv *= fraction;
        r->qhlsbv *= fraction;
    }

    r->cidpv = 0.0f;
    r->csdpv = 0.0f;
    r->chdpv = 0.0f;
    r->chldpv = 0.0f;
    r->cisbv = (s.ni / (s.qi + 1.0e-20f)) * r->qisbv;
    r->cssbv = (s.ns / (s.qs + 1.0e-20f)) * r->qssbv;
    r->chsbv = (s.ng / (s.qg + 1.0e-20f)) * r->qhsbv;
    r->chlsbv = (s.nh / (s.qh + 1.0e-20f)) * r->qhlsbv;
}

__device__ __forceinline__ void diagnose_melting_vapor_exchange(
    const State& s,
    const ParticleProperties& particles,
    float rho,
    float pressure,
    float dt,
    Rates* r)
{
    // WRF 19214-19275, default evapfac=1/mixedphase=.false.  Once a
    // frozen category is melting, vapor exchange occurs against the liquid
    // coating rather than through that category's ice-saturation tendency.
    // The source leaves all three number companions identically zero.
    if (!(r->qsmlr < 0.0f || r->qhmlr < 0.0f || r->qhlmlr < 0.0f)) {
        return;
    }

    const float dt_inverse = (float)(1.0 / (double)dt);
    const float dynamic_viscosity = 1.832e-5f
        * (416.16f / (s.temperature + 120.0f))
        * powf(s.temperature / 296.0f, 1.5f);
    const float diffusivity = 2.11e-5f
        * powf(s.temperature / 273.15f, 1.94f)
        * (101325.0f / pressure);
    const float kinematic = dynamic_viscosity / rho;
    const float ventilation_factor = powf(
        kinematic / diffusivity, 1.0f / 3.0f)
        * powf(kinematic, -0.5f);
    const float conductivity =
        2.43e-2f * dynamic_viscosity / 1.718e-5f;
    const float lv = latent_vapor(s.temperature);
    const float liquid_saturation = liquid_saturation_mixing_ratio(
        s.temperature, pressure);
    // qss0 is WRF's 0 C liquid saturation value, not qvs(T).
    const float qss0 = 380.0f / pressure;
    const float fav = lv * lv
        / (conductivity * 461.5f * s.temperature * s.temperature);
    const float fbv = 1.0f / (rho * diffusivity * liquid_saturation);
    const float denominator = qss0 * (fav + fbv);
    const float vapor_difference = s.qv - qss0;
    const float density_factor = sqrtf(1.225f / fmaxf(0.05f, rho));

    if (r->qsmlr < 0.0f && s.qs > QS_MIN && s.ns > 0.0f) {
        const float mean_volume = fmaxf(
            SNOW_VMIN,
            rho * s.qs / (particles.snow_density * s.ns));
        const float diameter = powf(
            mean_volume * (6.0f / PI), 1.0f / 3.0f);
        const float fall_speed = 11.9495f * density_factor
            * powf(mean_volume, 0.14f);
        const float ventilation = 0.65f + 0.44f * ventilation_factor
            * sqrtf(fall_speed * diameter);
        const float raw = 4.0f * PI * vapor_difference * s.ns
            * (0.5f * diameter) * ventilation / denominator;
        r->qscev = fmaxf(
            fminf(0.0f, raw),
            fminf(
                -0.1f * s.qs * dt_inverse,
                -0.5f * s.qs * dt_inverse));
    }

    if (r->qhmlr < 0.0f && s.qg > QD_MIN && s.ng > 0.0f) {
        const float mean_volume = fmaxf(
            GRAUPEL_VMIN,
            rho * s.qg / (particles.graupel_density * s.ng));
        const float mean_volume_diameter = powf(
            mean_volume * (6.0f / PI), 1.0f / 3.0f);
        const float diameter = powf(6.0f, -1.0f / 3.0f)
            * mean_volume_diameter;
        const float drag = fmaxf(0.45f, fminf(
            1.2f,
            0.45f + 0.55f
                * (800.0f - fmaxf(
                    170.0f, fminf(800.0f, particles.graupel_density)))
                / 630.0f));
        const float ventilation = 0.78f
            + 0.308f * 1.6083594560623169f
                * powf(4.0f * 9.8f / (3.0f * drag), 0.25f)
                * ventilation_factor
                * powf(particles.graupel_density / rho, 0.25f)
                * powf(diameter, 0.75f);
        float raw = 2.0f * PI * vapor_difference * s.ng
            * diameter * ventilation / denominator;
        raw = fmaxf(raw, -0.1f * s.qg * dt_inverse);
        if (s.temperature > 273.15f) raw = fminf(0.0f, raw);
        r->qhcev = raw;
    }

    if (r->qhlmlr < 0.0f && s.qh > QD_MIN && s.nh > 0.0f) {
        const float mean_volume = fmaxf(
            HAIL_VMIN,
            rho * s.qh / (particles.hail_density * s.nh));
        const float mean_volume_diameter = powf(
            mean_volume * (6.0f / PI), 1.0f / 3.0f);
        const float diameter = powf(24.0f, -1.0f / 3.0f)
            * mean_volume_diameter;
        float coefficient, exponent;
        dense_fall_coefficients(
            particles.hail_density, &coefficient, &exponent);
        const float ventilation = 1.56f
            + tgammaf(3.5f + 0.5f * exponent)
                * 0.308f * ventilation_factor
                * powf(diameter, 0.5f + 0.5f * exponent)
                * sqrtf(coefficient * density_factor);
        float raw = 2.0f * PI * vapor_difference * s.nh
            * diameter * ventilation / denominator;
        raw = fmaxf(raw, -0.1f * s.qh * dt_inverse);
        if (s.temperature > 273.15f) raw = fminf(0.0f, raw);
        r->qhlcev = raw;
    }
}

__device__ __forceinline__ void assemble_and_limit(
    const State& s,
    const ParticleProperties& particles,
    float rho,
    float dt,
    Rates* r,
    Aggregates* a)
{
    const float dt_inverse = (float)(1.0 / (double)dt);
    const float cold = s.temperature < 273.15f ? 1.0f : 0.0f;
    const float warm = 1.0f - cold;

    // Number aggregates are formed and limited before mass aggregates.
    // WRF 20908-20970 forms cloud-ice number before cloud-number scaling.
    a->pni_i = r->ciint + cold * (r->cwfrzc + r->cwctfzc)
        + r->chmul1 + r->chlmul1 + r->csplinter + r->csplinter2
        + r->csmul;
    a->pni_d = cold * (-r->cscni - r->cscnvi - r->craci - r->csaci
        - r->chaci - r->chlaci - r->chcni + r->cisbv)
        - warm * r->cimlr;

    // WRF 20979-21078: shared cloud-number donor limiter.
    a->pnc_i = -r->cwshw;
    a->pnc_d = -r->cautn
        + cold * (-r->ciacw - r->cwfrz - r->cwctfzp - r->cwctfzc)
        - r->cracw - r->csacw - r->chacw - r->chlacw;
    if (-a->pnc_d * dt > s.nc) {
        const float fraction = -s.nc / (a->pnc_d * dt);
        a->pnc_d = -s.nc * dt_inverse;
        r->cautn *= fraction;
        r->ciacw *= fraction;
        r->cwfrz *= fraction;
        r->cwfrzp *= fraction;
        r->cwfrzc *= fraction;
        r->cwctfzp *= fraction;
        r->cwctfzc *= fraction;
        r->cracw *= fraction;
        r->csacw *= fraction;
        r->chacw *= fraction;
        r->chlacw *= fraction;
        // Exact source-order quirk: this correction reads the already scaled
        // receiving rates rather than rebuilding pni_i.
        a->pni_i -= (1.0f - fraction) * cold
            * (r->cwfrzc + r->cwctfzc);
    }

    // WRF 21089-21166: shared rain-number donor limiter.
    a->pnr_i = r->crcnw - r->crshr
        + warm * (-r->chmlrr - r->chlmlrr / 0.4375f
            - r->csmlrr - r->cimlr);
    a->pnr_d = cold * (-r->ciacr - r->crfrz) - r->chacr - r->chlacr
        + r->crcev - r->cracr;
    if (-a->pnr_d * dt > s.nr) {
        const float fraction = -s.nr / (a->pnr_d * dt);
        a->pnr_d = -s.nr * dt_inverse;
        r->ciacr *= fraction;
        r->ciacrf *= fraction;
        r->ciacrs *= fraction;
        r->crfrz *= fraction;
        r->crfrzf *= fraction;
        r->crfrzs *= fraction;
        r->chacr *= fraction;
        r->chlacr *= fraction;
        r->crcev *= fraction;
        r->cracr *= fraction;
    }

    // WRF 21173-21241: limit snow-number donors before crediting the routed
    // frozen-small-rain source (default ifrzs=1).
    a->pns_i = cold * (r->cscni + r->cscnvi) + r->cscnh;
    a->pns_d = -r->chacs - r->chlacs - r->chcns
        + warm * r->csmlr + r->csshr + r->cssbv - r->csacs;
    if (s.ns + dt * (a->pns_i + a->pns_d) < 0.0f) {
        const float fraction = (-s.ns + a->pns_i * dt)
            / (a->pns_d * dt);
        a->pns_d *= fraction;
        r->chacs *= fraction;
        r->chlacs *= fraction;
        r->chcns *= fraction;
        r->csmlr *= fraction;
        r->csshr *= fraction;
        r->cssbv *= fraction;
        r->csacs *= fraction;
    }
    a->pns_i += r->crfrzs + r->ciacrs;

    // WRF 21243-21260 has no shared graupel-number donor limiter.
    a->png_i = r->crfrzf + cold * r->ciacrf + r->chcnsh + r->chcnih
        + r->chcnhl;
    a->png_d = warm * r->chmlr + r->chsbv
        - cold * r->chlcnh - r->cscnh;

    // WRF 21264-21297: hail-number limiter.  png_i remains stale if this
    // scales chcnhl, exactly as in the source.
    a->pnh_i = 0.4375f * r->chlcnhhl;
    a->pnh_d = warm * r->chlmlr + r->chlsbv - r->chcnhl;
    if (s.nh + dt * (a->pnh_i + a->pnh_d) < 0.0f) {
        const float fraction = (-s.nh + a->pnh_i * dt)
            / (a->pnh_d * dt);
        a->pnh_d *= fraction;
        r->chlmlr *= fraction;
        r->chlsbv *= fraction;
        r->chcnhl *= fraction;
    }

    // Initial vapor sum (WRF 21401-21450).  The rain-mass limiter below owns
    // the peculiar conditional patch/re-sum semantics.
    a->pqv_i = -fminf(0.0f, r->qrcev) - fminf(0.0f, r->qhcev)
        - fminf(0.0f, r->qhlcev) - fminf(0.0f, r->qscev)
        - r->qhsbv - r->qhlsbv - r->qssbv - cold * r->qisbv;
    a->pqv_d = -fmaxf(0.0f, r->qrcev) - fmaxf(0.0f, r->qhcev)
        - fmaxf(0.0f, r->qhlcev) - fmaxf(0.0f, r->qscev)
        + cold * (-r->qiint - r->qhdpv - r->qsdpv - r->qhldpv)
        - cold * r->qidpv;

    // WRF 21454-21500: shared cloud-water mass donor limiter.
    a->pqc_i = r->qwcnr - r->qwshw;
    a->pqc_d = cold * (-r->qiacw - r->qwfrz - r->qwctfz - r->qiihr)
        - r->qracw - r->qsacw - r->qrcnw - r->qhacw - r->qhlacw;
    if (a->pqc_d < 0.0f && -a->pqc_d * dt > s.qc) {
        const float fraction = -fmaxf(0.0f, s.qc) / (a->pqc_d * dt);
        // Exact WRF quirk: pin the aggregate; do not recompute it from the
        // scaled named rates (qiihr is not scaled).
        a->pqc_d = -s.qc * dt_inverse;
        r->qiacw *= fraction;
        r->qwfrz *= fraction;
        r->qwfrzp *= fraction;
        r->qwfrzc *= fraction;
        r->qwctfz *= fraction;
        r->qwctfzc *= fraction;
        r->qracw *= fraction;
        r->qsacw *= fraction;
        r->qhacw *= fraction;
        r->vhacw *= fraction;
        r->qrcnw *= fraction;
        r->qhlacw *= fraction;
        r->vhlacw *= fraction;
    }

    // Cloud-ice mass is formed after the cloud donor scaling.
    a->pqi_i = cold * (r->qiint + r->qwfrzc + r->qwctfzc)
        + r->qhmul1 + r->qhlmul1
        + r->qsplinter + r->qsplinter2 + r->qsmul
        + cold * (r->qidpv + r->qiacw);
    a->pqi_d = cold * (-r->qscni - r->qscnvi - r->qraci - r->qsaci)
        - r->qhaci - r->qhlaci + cold * r->qisbv
        + warm * r->qimlr - r->qhcni;

    // WRF 21572-21678: shared rain-water donor limiter.  Vapor aggregates
    // are patched first, then conditionally fully rebuilt after scaling.
    a->pqr_i = r->qracw + r->qrcnw + fmaxf(0.0f, r->qrcev)
        + warm * (-r->qhmlr - r->qsmlr - r->qhlmlr - r->qimlr)
        - r->qrshr;
    a->pqr_d = cold * (-r->qiacr - r->qrfrz) - r->qsacr - r->qhacr
        - r->qhlacr - r->qwcnr + fminf(0.0f, r->qrcev);
    if (a->pqr_d < 0.0f
            && -(a->pqr_d + a->pqr_i) * dt > s.qr) {
        const float fraction =
            (-s.qr + a->pqr_i * dt) / (a->pqr_d * dt);
        const float old_qrcev = r->qrcev;
        a->pqv_i += fminf(0.0f, old_qrcev)
            - fraction * fminf(0.0f, old_qrcev);
        a->pqv_d += fmaxf(0.0f, old_qrcev)
            - fraction * fmaxf(0.0f, old_qrcev);
        r->qiacr *= fraction;
        r->qiacrf *= fraction;
        r->qiacrs *= fraction;
        r->viacrf *= fraction;
        r->qrfrz *= fraction;
        r->qrfrzs *= fraction;
        r->qrfrzf *= fraction;
        r->vrfrzf *= fraction;
        r->qsacr *= fraction;
        r->qhacr *= fraction;
        r->vhacr *= fraction;
        r->qhlacr *= fraction;
        r->vhlacr *= fraction;
        r->qrcev *= fraction;
        r->qhcev *= fraction;
        r->qhlcev *= fraction;

        // WRF's post-limit expression moves qsacr inside the cold gate.
        a->pqr_d = cold * (-r->qiacr - r->qrfrz - r->qsacr)
            - r->qhacr - r->qhlacr - r->qwcnr
            + fminf(0.0f, r->qrcev);
        if (r->qrcev != 0.0f) {
            a->pqv_i = -fminf(0.0f, r->qrcev)
                - fminf(0.0f, r->qhcev) - fminf(0.0f, r->qhlcev)
                - fminf(0.0f, r->qscev) - r->qhsbv - r->qhlsbv
                - r->qssbv - cold * r->qisbv;
            a->pqv_d = -fmaxf(0.0f, r->qrcev)
                - fmaxf(0.0f, r->qhcev) - fmaxf(0.0f, r->qhlcev)
                - fmaxf(0.0f, r->qscev)
                + cold * (-r->qiint - r->qhdpv - r->qsdpv - r->qhldpv)
                - cold * r->qidpv;
        }
    }

    // WRF 21687-21741 snow mass aggregate and donor limiter.  This limiter
    // does not rebuild vapor after scaling qssbv/negative qscev.
    a->pqs_i = cold * (r->qscni + r->qsaci + r->qsdpv + r->qscnvi
        + r->qiacrs + r->qrfrzs + r->qsacr) + fmaxf(0.0f, r->qscev)
        + r->qsacw + r->qscnh;
    a->pqs_d = -r->qhacs - r->qhlacs - r->qhcns + warm * r->qsmlr
        + r->qsshr + r->qssbv + fminf(0.0f, r->qscev) - r->qsmul;
    if (a->pqs_d < 0.0f
            && s.qs + dt * (a->pqs_i + a->pqs_d) < 0.0f) {
        const float fraction = (-s.qs + a->pqs_i * dt)
            / (a->pqs_d * dt);
        a->pqs_d *= fraction;
        r->qhacs *= fraction;
        r->qhlacs *= fraction;
        r->qhcns *= fraction;
        r->qsmlr *= fraction;
        r->qsshr *= fraction;
        r->qssbv *= fraction;
        r->qsmul *= fraction;
        if (r->qscev < 0.0f) r->qscev *= fraction;
    }

    // WRF 21743-21765 graupel has no shared mass donor limiter.
    a->pqg_i = cold * (r->qrfrzf + r->qiacrf + r->qracif + r->qhdpv)
        + fmaxf(0.0f, r->qhcev) + r->qhacr + r->qhacw + r->qhacs
        + r->qhaci + r->qhcns + r->qhcni + r->qhcnhl;
    a->pqg_d = r->qhshr + warm * r->qhmlr + r->qhsbv
        + fminf(0.0f, r->qhcev) - r->qhmul1 - r->qhlcnh
        - r->qscnh - r->qsplinter - r->qsplinter2;

    // WRF 21768-21809 hail mass limiter.  Earlier graupel/ice/rain/vapor
    // aggregates intentionally remain stale when this rescales members.
    a->pqh_i = cold * r->qhldpv + fmaxf(0.0f, r->qhlcev)
        + r->qhlacr + r->qhlacw + r->qhlacs + r->qhlaci + r->qhlcnh;
    a->pqh_d = r->qhlshr + warm * r->qhlmlr + r->qhlsbv
        + fminf(0.0f, r->qhlcev) - r->qhlmul1 - r->qhcnhl;
    if (s.qh + dt * (a->pqh_i + a->pqh_d) < 0.0f) {
        const float fraction = (-s.qh + a->pqh_i * dt)
            / (a->pqh_d * dt);
        a->pqh_d *= fraction;
        r->qhlmlr *= fraction;
        r->qhlsbv *= fraction;
        r->qhcnhl *= fraction;
        r->qhlmul1 *= fraction;
        if (r->qhlcev < 0.0f) r->qhlcev *= fraction;
    }

    // WRF 22493-22671.  Collected ice/snow volume routes are not temperature
    // gated; only dense deposition and frozen-rain routes carry il5.  Liquid
    // coating condensation is credited at liquid-water density.
    a->pvg_i = rho * cold * (
            r->qhdpv / 170.0f + r->qracif / 900.0f)
        + rho * (r->qhaci + r->qhacs) / 170.0f
        + rho * fmaxf(0.0f, r->qhcev) / 1000.0f
        + r->vhacw + r->vhacr + r->vhcni + r->vhcns
        + r->viacrf + r->vrfrzf;
    a->pvg_d = rho * (
            warm * r->vhmlr + r->qhsbv + fminf(0.0f, r->qhcev)
            - r->qhmul1) / particles.graupel_density
        - r->vhlcnh + r->vhshdr - r->vhsoak - r->vscnh;
    a->pvh_i = rho * cold * r->qhldpv / 500.0f
        + rho * (r->qhlaci + r->qhlacs) / 500.0f
        + rho * fmaxf(0.0f, r->qhlcev) / 1000.0f
        + r->vhlacw + r->vhlacr + r->vhlcnhl;
    a->pvh_d = rho * (
            warm * r->vhlmlr + r->qhlsbv + fminf(0.0f, r->qhlcev)
            - r->qhlmul1) / particles.hail_density
        + r->vhlshdr - r->vhlsoak;
}

__device__ __forceinline__ void aggregate_once(
    State* s, const Rates& r, const Aggregates& a,
    float rho, float exner, float dt)
{
    s->qv += dt * (a.pqv_i + a.pqv_d);
    s->qc += dt * (a.pqc_i + a.pqc_d);
    s->qr += dt * (a.pqr_i + a.pqr_d);
    s->qi += dt * (a.pqi_i + a.pqi_d);
    s->qs += dt * (a.pqs_i + a.pqs_d);
    s->qg += dt * (a.pqg_i + a.pqg_d);
    s->qh += dt * (a.pqh_i + a.pqh_d);
    s->nc += dt * (a.pnc_i + a.pnc_d);
    s->nr += dt * (a.pnr_i + a.pnr_d);
    s->ni += dt * (a.pni_i + a.pni_d);
    s->ns += dt * (a.pns_i + a.pns_d);
    s->ng += dt * (a.png_i + a.png_d);
    s->nh += dt * (a.pnh_i + a.pnh_d);
    s->vg += dt * (a.pvg_i + a.pvg_d);
    s->vh += dt * (a.pvh_i + a.pvh_d);

    // Default eqtset<=1 latent-heating form.
    const float cold = s->temperature < 273.15f ? 1.0f : 0.0f;
    const float warm = 1.0f - cold;
    const float freezing = warm * (
            r.qhmlr + r.qsmlr + r.qhlmlr)
        + cold * (
            r.qsacw + r.qhacw + r.qhlacw
            + r.qsacr + r.qhacr + r.qhlacr
            + r.qsshr + r.qhshr + r.qhlshr
            + r.qrfrz + r.qiacr
            + r.qwfrz + r.qwctfz + r.qiihr + r.qiacw);
    const float sublimation = cold * (
        r.qsdpv + r.qhdpv + r.qhldpv + r.qidpv + r.qisbv + r.qiint)
        + r.qssbv + r.qhsbv + r.qhlsbv;
    const float liquid_vapor = r.qrcev + r.qhcev + r.qscev + r.qhlcev;
    s->theta += dt / exner * (
        latent_fusion(s->temperature) / 1004.0f * freezing
        + (latent_vapor(s->temperature) + latent_fusion(s->temperature))
            / 1004.0f * sublimation
        + latent_vapor(s->temperature) / 1004.0f * liquid_vapor);

    // Dense volume and remaining number routes are populated alongside their
    // named cold rates; no intermediate state is committed here.
    (void)rho;
}

}  // namespace

extern "C" __global__ void nssl2_prepare_fused_gs(
    float* __restrict__ temperature_k,
    float* __restrict__ target_m3,
    const float* __restrict__ state,
    const float* __restrict__ full_theta,
    const float* __restrict__ air_density,
    const float* __restrict__ pressure_pa,
    const float* __restrict__ exner,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;
    // This separate prepass snapshots the complete immutable pre-GS t0/t7
    // fields.  The fused kernel may then read neighbouring targets without
    // racing a thread that has already advanced the prognostic workspace.
    const float temperature = full_theta[idx] * exner[idx];
    temperature_k[idx] = temperature;
    target_m3[idx] = primary_ice_target(
        state[QV * n + idx],
        temperature,
        air_density[idx],
        pressure_pa[idx]);
}

extern "C" __global__ void nssl2_fused_gs(
    float* __restrict__ state,
    float* __restrict__ full_theta,
    const float* __restrict__ air_density,
    const float* __restrict__ pressure_pa,
    const float* __restrict__ exner,
    float* __restrict__ temperature_k,
    const float* __restrict__ vertical_velocity,
    const float* __restrict__ primary_ice_target_m3,
    const float* __restrict__ dz,
    float dt,
    int nz,
    int ncol,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    // GS diagnoses from nonnegative gathered work arrays, but final scatter
    // uses the raw vapor base and restores each raw negative hydromass
    // residual.  Snapshot both views before normalization changes anything.
    const float raw_qv = state[QV * n + idx];
    const float raw_qc = state[QC * n + idx];
    const float raw_qr = state[QR * n + idx];
    const float raw_qi = state[QI * n + idx];
    const float raw_qs = state[QS * n + idx];
    const float raw_qg = state[QG * n + idx];
    const float raw_qh = state[QH * n + idx];

    State s{};
    s.qv = fmaxf(raw_qv, 0.0f);
    s.qc = fmaxf(raw_qc, 0.0f);
    s.qr = fmaxf(raw_qr, 0.0f);
    s.qi = fmaxf(raw_qi, 0.0f);
    s.qs = fmaxf(raw_qs, 0.0f);
    s.qg = fmaxf(raw_qg, 0.0f);
    s.qh = fmaxf(raw_qh, 0.0f);
    s.nc = fmaxf(state[NC * n + idx], 0.0f);
    s.nr = fmaxf(state[NR * n + idx], 0.0f);
    s.ni = fmaxf(state[NI * n + idx], 0.0f);
    s.ns = fmaxf(state[NS * n + idx], 0.0f);
    s.ng = fmaxf(state[NG * n + idx], 0.0f);
    s.nh = fmaxf(state[NH * n + idx], 0.0f);
    s.nn = fmaxf(state[NN * n + idx], 0.0f);
    s.vg = fmaxf(state[VG * n + idx], 0.0f);
    s.vh = fmaxf(state[VH * n + idx], 0.0f);
    s.theta = full_theta[idx];
    s.temperature = temperature_k[idx];

    const float rho = air_density[idx];
    const ParticleProperties particles = normalize_state(&s, rho);
    Rates rates{};
    Aggregates aggregates{};
    diagnose_warm(s, rho, pressure_pa[idx], dt, &rates);
    diagnose_cloud_riming(
        s, particles, rho, dz[idx], dt, &rates);
    diagnose_rain_freezing(
        s, particles, rho, pressure_pa[idx], dt, &rates);
    diagnose_cloud_freezing(s, rho, pressure_pa[idx], dt, &rates);
    diagnose_frozen_collection(
        s, particles, rho, dz[idx], dt, &rates);
    diagnose_snow_aggregation(s, particles, rho, dt, &rates);
    diagnose_melting(
        s, particles, rho, pressure_pa[idx], dz[idx], dt, &rates);

    const int k = idx / ncol;
    const int column_index = idx - k * ncol;
    // WRF's microphysics driver supplies a mass-level W field to NSSL, then
    // nssl_2mom_gs averages that value with the next mass level (kp1 is
    // clamped at the top).  GPUWM owns interface W, so reproduce both
    // averaging operations here rather than stopping after interface-to-mass
    // centering.  The explicit RN operations preserve the two FP32 stores in
    // the WRF driver/GS boundary.
    const int velocity_kp = min(k + 1, nz - 1);
    const float w_mass = __fmul_rn(0.5f, __fadd_rn(
        vertical_velocity[k * ncol + column_index],
        vertical_velocity[(k + 1) * ncol + column_index]));
    const float w_mass_kp = __fmul_rn(0.5f, __fadd_rn(
        vertical_velocity[velocity_kp * ncol + column_index],
        vertical_velocity[(velocity_kp + 1) * ncol + column_index]));
    const float w_center = __fmul_rn(
        0.5f, __fadd_rn(w_mass_kp, w_mass));
    const int target_km = max(k - 1, 0);
    // Exact WRF kgsp=MIN(k+1,nz-1) in one-based indexing.
    const int target_kp = min(k + 1, nz - 2);
    const float vertical_span = (float)(
        max(target_kp - k, 0) + max(k - target_km, 0));
    diagnose_primary_ice(
        s,
        rho,
        pressure_pa[idx],
        w_center,
        dz[idx],
        primary_ice_target_m3[target_km * ncol + column_index],
        primary_ice_target_m3[target_kp * ncol + column_index],
        vertical_span,
        dt,
        &rates);

    diagnose_frozen_vapor(
        s,
        particles,
        rho,
        pressure_pa[idx],
        exner[idx],
        dt,
        &rates);
    diagnose_melting_vapor_exchange(
        s,
        particles,
        rho,
        pressure_pa[idx],
        dt,
        &rates);
    diagnose_crystal_to_snow(s, particles, rho, &rates);
    diagnose_wet_growth_shedding(
        s, particles, rho, pressure_pa[idx], dt, &rates);
    diagnose_ice_snow_to_graupel(s, particles, rho, &rates);
    diagnose_graupel_to_hail(
        s, particles, rho, pressure_pa[idx], dt, &rates);
    diagnose_hallett_mossop(s, rho, &rates);

    // Source-ordered cold diagnosis is inserted here.  It reads only `s` and
    // the environmental arrays, never an already-updated prognostic.
    assemble_and_limit(
        s, particles, rho, dt, &rates, &aggregates);
    // qx(lv) above is the nonnegative diagnostic vapor (and may include a
    // tiny cleaned-up hydromass), whereas WRF advances vapor from raw qv0.
    s.qv = raw_qv;
    aggregate_once(&s, rates, aggregates, rho, exner[idx], dt);

    // WRF 23187-23224: t0 is diagnosed before warm cloud-ice melting and is
    // not recomputed afterward for option-18 ipconc=5/ibfc=1.
    s.temperature = s.theta * exner[idx];
    const float temperature_after_rates = s.temperature;
    if (s.temperature > 273.15f && s.qi > 0.0f) {
        const float melted = s.qi;
        s.qc += melted;
        s.nc += s.ni;
        s.qi = 0.0f;
        s.ni = 0.0f;
        s.theta -= latent_fusion(s.temperature) / (1004.0f * exner[idx])
            * melted;
    }

    // WRF 23658-23667 restores the original negative Registry residual to
    // qx before entering the post-GS two-moment size limiter.
    s.qc += fminf(raw_qc, 0.0f);
    s.qr += fminf(raw_qr, 0.0f);
    s.qi += fminf(raw_qi, 0.0f);
    s.qs += fminf(raw_qs, 0.0f);
    s.qg += fminf(raw_qg, 0.0f);
    s.qh += fminf(raw_qh, 0.0f);
    final_bounds(&s, rho, particles);
    state[QV * n + idx] = s.qv;
    state[QC * n + idx] = s.qc;
    state[QR * n + idx] = s.qr;
    state[QI * n + idx] = s.qi;
    state[QS * n + idx] = s.qs;
    state[QG * n + idx] = s.qg;
    state[QH * n + idx] = s.qh;
    state[NC * n + idx] = s.nc;
    state[NR * n + idx] = s.nr;
    state[NI * n + idx] = s.ni;
    state[NS * n + idx] = s.ns;
    state[NG * n + idx] = s.ng;
    state[NH * n + idx] = s.nh;
    state[NN * n + idx] = s.nn;
    state[VG * n + idx] = s.vg;
    state[VH * n + idx] = s.vh;
    full_theta[idx] = s.theta;
    temperature_k[idx] = temperature_after_rates;
}
