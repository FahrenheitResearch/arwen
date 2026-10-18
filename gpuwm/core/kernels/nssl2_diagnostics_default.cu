// WRF v4.6.1 NSSL option-18 default S-band radar diagnostic.
//
// Numerical authority: phys/module_mp_nssl_2mom.F:8588-9597, after the
// default nssl_2mom_init(ipconc=5, density_on, hail_on).  Registry number
// and volume moments are converted back to the source routine's internal
// per-volume convention with dry-air density.  This is the dry-category
// default (mixedphase=0), so temperature is an API/provenance input but does
// not change any active contribution.

extern "C" __global__ void nssl2_radardd02(
    const float* __restrict__ air_density,
    const float* __restrict__ temperature_k,
    const float* __restrict__ qr,
    const float* __restrict__ qi,
    const float* __restrict__ qs,
    const float* __restrict__ qg,
    const float* __restrict__ qh,
    const float* __restrict__ qnr,
    const float* __restrict__ qni,
    const float* __restrict__ qns,
    const float* __restrict__ qng,
    const float* __restrict__ qnh,
    const float* __restrict__ qvolg,
    const float* __restrict__ qvolh,
    float* __restrict__ refl_10cm,
    int concentration_space,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const float rho = air_density[idx];
    const float pi = 3.14159265358979323846f;
    const float water_density = 1000.0f;
    float total = 0.0f;
    (void)temperature_k;

    // Rain: imurain=1, alphar=0, sixth moment reconstructed from q and N.
    const float number_scale = concentration_space ? 1.0f : rho;
    const float volume_scale = concentration_space ? 1.0f : rho;
    const float rain_number = qnr[idx] * number_scale;
    if (qr[idx] >= 1.0e-12f && rain_number > 1.0e-3f) {
        const double g1 = 20.0;
        const double mass = (double)rho * (double)qr[idx];
        const double zx = g1 * mass * mass / (double)rain_number;
        const double scale = 6.0 / ((double)pi * 1000.0);
        const double ze = 1.0e18 * zx * scale * scale;
        total = (float)ze;
    }

    // Snow: the active dry Cox (1988) m=p*V^(2/3) relationship.
    const float snow_number = qns[idx] * number_scale;
    if (qs[idx] >= 1.0e-13f && snow_number > 1.0e-7f) {
        const float snu = -0.8f;
        const float gsnow1 = 4.590843677520752f;
        const float gsnow73 = 0.8877619504928589f;
        const float snow = 1.0e18f * 323.3226f
            * (0.106214f * 0.106214f)
            * (0.189f * qs[idx]) * qs[idx] * rho * rho * gsnow73
            / (snow_number * (917.0f * 917.0f) * gsnow1
               * powf(1.0f + snu, 4.0f / 3.0f));
        total += snow;
    }

    // Cloud ice: spherical 900-kg/m3 particles and cinu=0.
    const float ice_number = qni[idx] * number_scale;
    if (qi[idx] > 1.0e-13f && ice_number > 1.0f) {
        const float mean_volume =
            rho * qi[idx] / (900.0f * ice_number);
        const float ice = 0.224f * 3.6e18f * 2.0f
            * ice_number * mean_volume * mean_volume * (0.9f * 0.9f);
        total += ice;
    }

    // Variable-density graupel: alphah=0, 0.3--20-mm volume bounds.
    const float graupel_number = qng[idx] * number_scale;
    if (qg[idx] >= 1.0e-12f && graupel_number >= 1.0e-8f) {
        float density = 500.0f;
        const float volume_mixing_ratio = qvolg[idx] * volume_scale;
        if (volume_mixing_ratio > 0.0f) {
            density = rho * qg[idx] / volume_mixing_ratio;
            density = fminf(900.0f, fmaxf(100.0f, density));
        }
        float local_number = graupel_number;
        float mean_volume = rho * qg[idx]
            / (density * fmaxf(1.0e-3f, local_number));
        const float minimum_volume =
            0.523599f * (0.3e-3f * 0.3e-3f * 0.3e-3f);
        const float maximum_volume =
            0.523599f * (20.0e-3f * 20.0e-3f * 20.0e-3f);
        if (mean_volume < minimum_volume || mean_volume > maximum_volume) {
            mean_volume = fminf(
                maximum_volume, fmaxf(minimum_volume, mean_volume));
            local_number = rho * qg[idx] / (mean_volume * density);
        }
        if (local_number > 0.0f) {
            const double g1 = 20.0;
            const double zx = g1 * (double)rho * (double)rho
                * (0.224 * (double)qg[idx]) * (double)qg[idx]
                / (double)local_number;
            const double scale = 6.0 / ((double)pi * 1000.0);
            total += (float)(1.0e18 * zx * scale * scale);
        }
    }

    // Variable-density large hail: alphahl=1, 0.3--40-mm bounds.
    const float hail_number = qnh[idx] * number_scale;
    if (qh[idx] >= 1.0e-12f && hail_number > 0.0f) {
        float density = 900.0f;
        const float volume_mixing_ratio = qvolh[idx] * volume_scale;
        if (volume_mixing_ratio > 0.0f) {
            density = rho * qh[idx] / volume_mixing_ratio;
            density = fminf(900.0f, fmaxf(300.0f, density));
        }
        float local_number = hail_number;
        float mean_volume = rho * qh[idx]
            / (density * fmaxf(1.0e-9f, local_number));
        const float minimum_volume =
            0.523599f * (0.3e-3f * 0.3e-3f * 0.3e-3f);
        const float maximum_volume =
            0.523599f * (40.0e-3f * 40.0e-3f * 40.0e-3f);
        if (mean_volume < minimum_volume || mean_volume > maximum_volume) {
            mean_volume = fminf(
                maximum_volume, fmaxf(minimum_volume, mean_volume));
            local_number = rho * qh[idx] / (mean_volume * density);
        }
        if (local_number > 0.0f) {
            const double g1 = 8.75;
            const double zx = g1 * (double)rho * (double)rho
                * (0.224 * (double)qh[idx]) * (double)qh[idx]
                / (double)local_number;
            const double scale = 6.0 / ((double)pi * 1000.0);
            total += (float)(1.0e18 * zx * scale * scale);
        }
    }

    refl_10cm[idx] = total > 0.0f
        ? fmaxf(0.0f, 10.0f * log10f(total))
        : 0.0f;
}

// Concentration-space twin of the admitted Registry-facing effective-radius
// diagnostic.  This entry point is for direct views of the 16-field NSSL slab:
// number moments are already #/m3, so there is deliberately no multiply by
// dry-air density and no #/kg round-trip.  Outputs retain WRF's metre boundary.
extern "C" __global__ void nssl2_effective_radius_concentration(
    const float* __restrict__ air_density,
    const float* __restrict__ qc,
    const float* __restrict__ cloud_number,
    const float* __restrict__ qi,
    const float* __restrict__ ice_number,
    const float* __restrict__ qs,
    const float* __restrict__ snow_number,
    float* __restrict__ re_cloud,
    float* __restrict__ re_ice,
    float* __restrict__ re_snow,
    int n)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    float cloud_radius = 2.51e-6f;
    float ice_radius = 10.01e-6f;
    float snow_radius = 25.0e-6f;

    const float rho = air_density[idx];
    const float pi_over_six = 3.14159265358979323846f / 6.0f;
    const float one_third = 1.0f / 3.0f;
    const float cxmin = 1.0e-8f;
    const float qxmin = 1.0e-13f;

    const float cloud_mass = fmaxf(qc[idx], 0.0f);
    const float nc = fmaxf(cloud_number[idx], 0.0f);
    if (cloud_mass > qxmin && nc > cxmin) {
        const float lambda = powf(
            (nc * pi_over_six * 1000.0f) / (cloud_mass * rho),
            one_third);
        const float raw = 0.5f * 1.1077321767807007f / lambda;
        cloud_radius = fmaxf(2.51e-6f, fminf(raw, 50.0e-6f));
    }

    const float ice_mass = fmaxf(qi[idx], 0.0f);
    const float ni = fmaxf(ice_number[idx], 0.0f);
    if (ice_mass > qxmin && ni > cxmin) {
        const float lambda = powf(
            (ni * pi_over_six * 900.0f) / (ice_mass * rho),
            one_third);
        const float raw = 0.5f * 1.1077321767807007f / lambda;
        ice_radius = fmaxf(10.01e-6f, fminf(raw, 125.0e-6f));
    }

    const float snow_mass = fmaxf(qs[idx], 0.0f);
    const float ns = fmaxf(snow_number[idx], 0.0f);
    if (snow_mass > qxmin && ns > cxmin) {
        const float gamma_mass = 0.91816872358322144f;
        const float gamma_number = 4.5908436775207520f;
        const float radius_factor = 0.83693999052047729f;
        const float lambda = powf(
            (ns * pi_over_six * 100.0f * gamma_mass)
                / (snow_mass * rho * gamma_number),
            one_third);
        const float raw = 0.5f * radius_factor / lambda;
        snow_radius = fmaxf(25.0e-6f, fminf(raw, 999.0e-6f));
    }

    re_cloud[idx] = cloud_radius;
    re_ice[idx] = ice_radius;
    re_snow[idx] = snow_radius;
}
