"""Bit oracle for the shared velocity gamma table."""
from pathlib import Path

import numpy as np
import pytest


# Frozen d6929cb8d dense-frozen implementation, used as the bit oracle.
# A146: this frozen copy's divisions by a compile-time constant are spelled
# __fdiv_rn, as the production kernel has spelled them since that fix
# (NVRTC compiled them as reciprocal multiplies on Blackwell).  On sm_89
# both spellings are the same div.rn, so the copy is still the frozen base.
_BASE_DENSE = r"""
#define NSSL2_KMAX_SHALLOW 64
#define NSSL2_KMAX_GENERIC 256
__device__ __forceinline__ float nssl2_gamma_lookup(float argument)
{
    const double scaled = 100.0 * (double)argument;
    const int lower_index = (int)scaled;
    const double lower = 0.01 * (double)lower_index;
    const double fraction = (double)argument - lower;
    const double lower_gamma = tgamma(lower);
    const double upper_gamma = tgamma(lower + 0.01);
    return (float)(lower_gamma
        + (upper_gamma - lower_gamma) * fraction * 100.0);
}

__device__ __forceinline__ float nssl2_dense_frozen_z(
    float q, float number, float volume, float rho, bool hail)
{
    if (!(q > 1.0e-12f) || !(number > 1.0e-15f)) return 0.0f;

    const float minimum_volume =
        0.523599f * (0.3e-3f * 0.3e-3f * 0.3e-3f);
    const float maximum_diameter = hail ? 40.0e-3f : 20.0e-3f;
    const float maximum_volume = 0.523599f * maximum_diameter
        * maximum_diameter * maximum_diameter;
    float particle_density = hail ? 800.0f : 500.0f;
    if (volume > 0.0f) {
        particle_density = fminf(
            900.0f, fmaxf(170.0f, rho * q / volume));
    }
    float mean_volume = rho * q / (particle_density * number);
    float effective_number = number;
    if (mean_volume < minimum_volume || mean_volume > maximum_volume) {
        mean_volume = fminf(
            maximum_volume, fmaxf(minimum_volume, mean_volume));
        effective_number = rho * q / (particle_density * mean_volume);
    }
    const float z_factor =
        (6.0f / (3.14159265358979323846f * 1000.0f))
        * (6.0f / (3.14159265358979323846f * 1000.0f));
    const float moment_ratio = hail ? 8.75f : 20.0f;
    return moment_ratio * rho * rho * q * q
        / effective_number * z_factor;
}

__device__ __forceinline__ void nssl2_graupel_mm_coefficients(
    float particle_density, float* coefficient, float* exponent)
{
    const float table_density[9] = {
        50.0f, 150.0f, 250.0f, 350.0f, 450.0f,
        550.0f, 650.0f, 750.0f, 850.0f};
    const float table_coefficient[9] = {
        62.923f, 94.122f, 114.74f, 131.21f, 145.26f,
        157.71f, 168.98f, 179.36f, 189.02f};
    const float table_exponent[9] = {
        0.67819f, 0.63789f, 0.62197f, 0.61240f, 0.60572f,
        0.60066f, 0.59663f, 0.59330f, 0.59048f};

    int index = (int)(__fdiv_rn((particle_density - 50.0f), 100.0f));
    index = max(0, min(8, index));
    if (index < 8) {
        const float fraction = fmaxf(
            0.0f, 0.01f * (particle_density - table_density[index]));
        *coefficient = table_coefficient[index]
            + fraction * (table_coefficient[index + 1]
                          - table_coefficient[index]);
        *exponent = table_exponent[index]
            + fraction * (table_exponent[index + 1]
                          - table_exponent[index]);
    } else {
        *coefficient = table_coefficient[index];
        *exponent = table_exponent[index];
    }
}

template <int KMAX, bool HAIL>
__device__ __forceinline__ void nssl2_dense_frozen_sediment_impl(
    const float* __restrict__ air_density,
    float* __restrict__ qx,
    float* __restrict__ qnx,
    float* __restrict__ qvolx,
    const float* __restrict__ dz,
    float* __restrict__ frozennc,
    float* __restrict__ frozenncv,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho[KMAX];
    float graupel[KMAX];
    float number[KMAX];
    float volume[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float z_velocity[KMAX];
    float mass_flux[KMAX + 1];
    float number_flux[KMAX + 1];
    float volume_flux[KMAX + 1];
    float z_flux[KMAX + 1];
    float mass_number_flux[KMAX + 1];
    float z_initial[KMAX];
    float z_advected[KMAX];
    float number_mass_weighted[KMAX];

    const float pi = 3.14159265358979323846f;
    const float minimum_volume =
        0.523599f * (0.3e-3f * 0.3e-3f * 0.3e-3f);
    const float maximum_diameter = HAIL ? 40.0e-3f : 20.0e-3f;
    const float maximum_volume = 0.523599f * maximum_diameter
        * maximum_diameter * maximum_diameter;
    const float shape = HAIL ? 1.0f : 0.0f;
    const float characteristic_factor = powf(
        HAIL ? 24.0f : 6.0f, -1.0f / 3.0f);
    float maximum_courant_rate = 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        rho[k] = air_density[idx];
        graupel[k] = qx[idx];
        number[k] = qnx[idx];
        volume[k] = qvolx[idx];
        mass_velocity[k] = 0.0f;
        number_velocity[k] = 0.0f;
        z_velocity[k] = 0.0f;

        const float positive_graupel = fmaxf(graupel[k], 0.0f);
        if (positive_graupel > 1.0e-12f) {
            const float minimum_density = HAIL ? 500.0f : 170.0f;
            float particle_density = HAIL ? 800.0f : 500.0f;
            if (volume[k] > rho[k] * 1.0e-15f) {
                particle_density = fminf(
                    900.0f,
                    fmaxf(minimum_density,
                          rho[k] * positive_graupel / volume[k]));
            }
            float mean_volume = rho[k] * positive_graupel
                / (particle_density * fmaxf(1.0e-9f, number[k]));
            mean_volume = fminf(
                maximum_volume, fmaxf(minimum_volume, mean_volume));
            const float mass_diameter = powf(__fdiv_rn(6.0f * mean_volume, pi),
                                             1.0f / 3.0f);
            const float characteristic_diameter =
                characteristic_factor * mass_diameter;
            float coefficient;
            float exponent;
            nssl2_graupel_mm_coefficients(
                particle_density, &coefficient, &exponent);
            const float density_factor = sqrtf(
                1.225f * fminf(20.0f, 1.0f / rho[k]));
            const float base_speed = density_factor * coefficient
                * powf(characteristic_diameter, exponent);
            mass_velocity[k] = base_speed
                * nssl2_gamma_lookup(4.0f + shape + exponent)
                / nssl2_gamma_lookup(4.0f + shape);
            number_velocity[k] = base_speed
                * nssl2_gamma_lookup(1.0f + shape + exponent)
                / nssl2_gamma_lookup(1.0f + shape);
            z_velocity[k] = base_speed
                * nssl2_gamma_lookup(7.0f + shape + exponent)
                / nssl2_gamma_lookup(7.0f + shape);
            if (number_velocity[k] > mass_velocity[k]
                    || (mass_velocity[k] > z_velocity[k]
                        && z_velocity[k] > 0.0f)) {
                mass_velocity[k] = fmaxf(
                    mass_velocity[k], number_velocity[k]);
                z_velocity[k] = fmaxf(z_velocity[k], mass_velocity[k]);
            }
            mass_velocity[k] = fminf(70.0f, fminf(150.0f, mass_velocity[k]));
            number_velocity[k] = fminf(
                70.0f, fminf(150.0f, number_velocity[k]));
            z_velocity[k] = fminf(70.0f, fminf(150.0f, z_velocity[k]));
        }
        const float inverse_dz = 1.0f / dz[idx];
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, mass_velocity[k] * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, number_velocity[k] * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, z_velocity[k] * inverse_dz);
    }

    if (maximum_courant_rate == 0.0f) {
        frozenncv[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(
            2, (int)(__fdiv_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        for (int k = 0; k < nz; ++k) {
            z_initial[k] = nssl2_dense_frozen_z(
                graupel[k], number[k], volume[k], rho[k], HAIL);
            z_advected[k] = z_initial[k];
            number_mass_weighted[k] = number[k];
            mass_flux[k] = graupel[k] * mass_velocity[k] * rho[k];
            number_flux[k] = number[k] * number_velocity[k];
            volume_flux[k] = volume[k] * mass_velocity[k];
            z_flux[k] = z_initial[k] * z_velocity[k];
            mass_number_flux[k] = number[k] * mass_velocity[k];
        }
        mass_flux[nz] = 0.0f;
        number_flux[nz] = 0.0f;
        volume_flux[nz] = 0.0f;
        z_flux[nz] = 0.0f;
        mass_number_flux[nz] = 0.0f;

        surface_mean_flux +=
            graupel[0] * mass_velocity[0] * dt_fraction;

        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            const float inverse_dz = 1.0f / dz[idx];
            graupel[k] += dt_substep * inverse_dz / rho[k]
                * (mass_flux[k + 1] - mass_flux[k]);
            volume[k] += dt_substep * inverse_dz
                * (volume_flux[k + 1] - volume_flux[k]);
            number[k] += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
            z_advected[k] += dt_substep * inverse_dz
                * (z_flux[k + 1] - z_flux[k]);
            number_mass_weighted[k] += dt_substep * inverse_dz
                * (mass_number_flux[k + 1] - mass_number_flux[k]);
        }

        for (int k = 0; k < nz; ++k) {
            if (z_advected[k] > 0.0f) {
                const float diagnosed = nssl2_dense_frozen_z(
                    graupel[k], number[k], volume[k], rho[k], HAIL);
                if (diagnosed > z_advected[k]
                        && diagnosed > 0.0f
                        && z_advected[k] > z_initial[k]) {
                    const double z_factor =
                        (double)(6.0f / (pi * 1000.0f))
                        * (double)(6.0f / (pi * 1000.0f));
                    const double moment_ratio = HAIL ? 8.75 : 20.0;
                    const double reconstructed =
                        moment_ratio * (double)rho[k] * (double)rho[k]
                        * (double)graupel[k] * (double)graupel[k]
                        / ((double)z_advected[k] / z_factor);
                    const float reconstructed_real = (float)reconstructed;
                    number[k] = fmaxf(
                        fminf(reconstructed_real,
                              number_mass_weighted[k]),
                        number[k]);
                } else {
                    number[k] = fmaxf(
                        number_mass_weighted[k], number[k]);
                }
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        qx[idx] = graupel[k];
        qnx[idx] = number[k];
        qvolx[idx] = volume[k];
    }
    const float exported = dt * rho[0] * surface_mean_flux;
    frozenncv[column] = exported;
    frozennc[column] += exported;
}

#define NSSL2_GRAUPEL_SEDIMENT_PARAMETERS                              \
    const float* __restrict__ air_density, float* __restrict__ qg,      \
    float* __restrict__ qng, float* __restrict__ qvolg,                 \
    const float* __restrict__ dz, float* __restrict__ graupelnc,        \
    float* __restrict__ graupelncv, float dt, int nz, int ny, int nx

extern "C" __global__ void nssl2_graupel_sediment_64(
    NSSL2_GRAUPEL_SEDIMENT_PARAMETERS)
{
    nssl2_dense_frozen_sediment_impl<NSSL2_KMAX_SHALLOW, false>(
        air_density, qg, qng, qvolg, dz, graupelnc, graupelncv,
        dt, nz, ny, nx);
}

extern "C" __global__ void nssl2_graupel_sediment_256(
    NSSL2_GRAUPEL_SEDIMENT_PARAMETERS)
{
    nssl2_dense_frozen_sediment_impl<NSSL2_KMAX_GENERIC, false>(
        air_density, qg, qng, qvolg, dz, graupelnc, graupelncv,
        dt, nz, ny, nx);
}

#define NSSL2_HAIL_SEDIMENT_PARAMETERS                                  \
    const float* __restrict__ air_density, float* __restrict__ qh,      \
    float* __restrict__ qnh, float* __restrict__ qvolh,                 \
    const float* __restrict__ dz, float* __restrict__ hailnc,           \
    float* __restrict__ hailncv, float dt, int nz, int ny, int nx

extern "C" __global__ void nssl2_hail_sediment_64(
    NSSL2_HAIL_SEDIMENT_PARAMETERS)
{
    nssl2_dense_frozen_sediment_impl<NSSL2_KMAX_SHALLOW, true>(
        air_density, qh, qnh, qvolh, dz, hailnc, hailncv,
        dt, nz, ny, nx);
}

extern "C" __global__ void nssl2_hail_sediment_256(
    NSSL2_HAIL_SEDIMENT_PARAMETERS)
{
    nssl2_dense_frozen_sediment_impl<NSSL2_KMAX_GENERIC, true>(
        air_density, qh, qnh, qvolh, dz, hailnc, hailncv,
        dt, nz, ny, nx);
}


"""

@pytest.mark.parametrize("nz", [65, 128])
def test_dense_sediment_bits(nz):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device unavailable")
    from gpuwm.core.kernels import _preamble
    reference = cp.RawModule(code=_preamble() + _BASE_DENSE, options=("-std=c++17",))
    source_path = Path(__file__).parents[1] / "gpuwm/core/kernels/nssl2_driver_support.cu"
    actual = cp.RawModule(code=_preamble() + source_path.read_text(), options=("-std=c++17",))
    table = cp.empty((2, 120), dtype=cp.float64)
    actual.get_function("nssl2_fill_velocity_gamma")((2,), (64,), (table,))
    rng = np.random.default_rng(1802 + nz)
    shape = (nz, 1, 97)
    rho = cp.asarray(rng.uniform(0.3, 1.2, shape).astype(np.float32))
    dz = cp.asarray(rng.uniform(60, 500, shape).astype(np.float32))
    for category in ("graupel", "hail"):
        mass = rng.uniform(0, 0.006, shape).astype(np.float32)
        mass[rng.random(shape) < 0.3] = 0
        mass[:, :, 0] = 0
        number = rng.uniform(0.01, 2e4, shape).astype(np.float32)
        density = rng.uniform(170, 900, shape).astype(np.float32)
        volume = rho.get() * mass / density
        name = f"nssl2_{category}_sediment_{64 if nz <= 64 else 256}"
        for dt in (15, 45):
            outputs = []
            for module in (reference, actual):
                q, n, v = [cp.asarray(x.copy()) for x in (mass, number, volume)]
                accum = cp.full((1, 97), np.float32(0.125))
                exported = cp.empty_like(accum)
                kernel_name = name
                grid = 2
                extra = ()
                if module is actual:
                    route = "cached"
                    kernel_name = name.replace("_sediment_", f"_sediment_{route}_")
                    grid = 2
                    extra = (table[0 if category == "graupel" else 1],)
                module.get_function(kernel_name)((grid,), (64,),
                    (rho, q, n, v, dz, accum, exported,
                     np.float32(dt), np.int32(nz), np.int32(1), np.int32(97), *extra))
                outputs.append([x.get().view(np.uint32) for x in (q, n, v, accum, exported)])
            for expected, observed in zip(*outputs):
                np.testing.assert_array_equal(expected, observed)
