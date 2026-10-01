"""Frozen-base bit oracles for the smaller sedimentation categories."""
from pathlib import Path

import numpy as np
import pytest

# A146: this frozen copy's divisions by a compile-time constant are spelled
# __fdiv_rn, as the production kernel has spelled them since that fix
# (NVRTC compiled them as reciprocal multiplies on Blackwell).  On sm_89
# both spellings are the same div.rn, so the copy is still the frozen base.
_BASE_SMALL = r"""
#define NSSL2_KMAX_SHALLOW 64
#define NSSL2_KMAX_GENERIC 256
template <int KMAX>
__device__ __forceinline__ void nssl2_cloud_sediment_impl(
    const float* __restrict__ air_density,
    const float* __restrict__ temperature_k,
    float* __restrict__ qc,
    float* __restrict__ qndrop,
    const float* __restrict__ dz,
    float* __restrict__ cloud_surface_export,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho[KMAX];
    float cloud[KMAX];
    float number[KMAX];
    float velocity[KMAX];
    float mass_flux[KMAX + 1];
    float number_flux[KMAX + 1];

    const float pi = 3.14159265358979323846f;
    const float minimum_mass = 1000.0f * 0.523599f
        * (4.0e-6f * 4.0e-6f * 4.0e-6f);
    const float maximum_mass = 1000.0f * 0.523599f
        * (120.0e-6f * 120.0e-6f * 120.0e-6f);
    const float mass_to_diameter = 6.0f / (pi * 1000.0f);
    float maximum_courant_rate = 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        rho[k] = air_density[idx];
        cloud[k] = qc[idx];
        number[k] = qndrop[idx];
        velocity[k] = 0.0f;

        const float positive_cloud = fmaxf(cloud[k], 0.0f);
        if (positive_cloud > 1.0e-13f) {
            const float positive_number = fmaxf(number[k], 0.0f);
            const float effective_number = positive_number > 1.0e-8f
                ? positive_number
                : fmaxf(1.0e-8f,
                        __fdiv_rn(rho[k] * positive_cloud, maximum_mass));
            const float particle_mass = fminf(
                maximum_mass,
                fmaxf(minimum_mass,
                      positive_cloud * rho[k] / effective_number));
            const float diameter = powf(
                particle_mass * mass_to_diameter, 1.0f / 3.0f);
            const float radius = 0.5f * diameter;
            const float temperature = temperature_k[idx];
            const float viscosity = 1.832e-5f
                * (416.16f / (temperature + 120.0f))
                * powf(__fdiv_rn(temperature, 296.0f), 1.5f);
            if (viscosity > 0.0f) {
                velocity[k] = fminf(
                    70.0f,
                    2.0f * 9.8f * 1000.0f * radius * radius
                        / (9.0f * viscosity));
            }
        }
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, velocity[k] / dz[idx]);
    }

    if (maximum_courant_rate == 0.0f) {
        cloud_surface_export[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(2,
            (int)(__fdiv_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        for (int k = 0; k < nz; ++k) {
            mass_flux[k] = cloud[k] * velocity[k] * rho[k];
            number_flux[k] = number[k] * velocity[k];
        }
        mass_flux[nz] = 0.0f;
        number_flux[nz] = 0.0f;

        surface_mean_flux += cloud[0] * velocity[0] * dt_fraction;

        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            const float inverse_dz = 1.0f / dz[idx];
            cloud[k] += dt_substep * inverse_dz / rho[k]
                * (mass_flux[k + 1] - mass_flux[k]);
            number[k] += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        qc[idx] = cloud[k];
        qndrop[idx] = number[k];
    }
    cloud_surface_export[column] = dt * rho[0] * surface_mean_flux;
}

#define NSSL2_CLOUD_SEDIMENT_PARAMETERS                                 \
    const float* __restrict__ air_density,                              \
    const float* __restrict__ temperature_k, float* __restrict__ qc,    \
    float* __restrict__ qndrop, const float* __restrict__ dz,           \
    float* __restrict__ cloud_surface_export,                           \
    float dt, int nz, int ny, int nx

extern "C" __global__ void nssl2_cloud_sediment_64(
    NSSL2_CLOUD_SEDIMENT_PARAMETERS)
{
    nssl2_cloud_sediment_impl<NSSL2_KMAX_SHALLOW>(
        air_density, temperature_k, qc, qndrop, dz,
        cloud_surface_export, dt, nz, ny, nx);
}

extern "C" __global__ void nssl2_cloud_sediment_256(
    NSSL2_CLOUD_SEDIMENT_PARAMETERS)
{
    nssl2_cloud_sediment_impl<NSSL2_KMAX_GENERIC>(
        air_density, temperature_k, qc, qndrop, dz,
        cloud_surface_export, dt, nz, ny, nx);
}

template <int KMAX>
__device__ __forceinline__ void nssl2_snow_sediment_impl(
    const float* __restrict__ air_density,
    float* __restrict__ qs,
    float* __restrict__ qns,
    const float* __restrict__ dz,
    float* __restrict__ snownc,
    float* __restrict__ snowncv,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho[KMAX];
    float snow[KMAX];
    float number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float z_velocity[KMAX];
    float mass_flux[KMAX + 1];
    float number_flux[KMAX + 1];
    float mass_number_flux[KMAX + 1];
    float number_mass_weighted[KMAX];

    const float minimum_volume =
        0.523599f * (0.01e-3f * 0.01e-3f * 0.01e-3f);
    const float maximum_volume =
        0.523599f * (10.0e-3f * 10.0e-3f * 10.0e-3f);
    float maximum_courant_rate = 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        rho[k] = air_density[idx];
        snow[k] = qs[idx];
        number[k] = qns[idx];
        mass_velocity[k] = 0.0f;
        number_velocity[k] = 0.0f;
        z_velocity[k] = 0.0f;

        const float positive_snow = fmaxf(snow[k], 0.0f);
        if (positive_snow > 1.0e-13f) {
            const float local_number = fmaxf(number[k], 0.0f);
            float mean_volume = rho[k] * positive_snow
                / (100.0f * fmaxf(1.0e-9f, local_number));
            mean_volume = fminf(
                maximum_volume, fmaxf(minimum_volume, mean_volume));
            const float density_factor = sqrtf(
                1.225f * fminf(20.0f, 1.0f / rho[k]));
            const float size_factor = powf(mean_volume, 0.14f);
            mass_velocity[k] = fminf(
                70.0f, 11.9495f * density_factor * size_factor);
            number_velocity[k] = fminf(
                70.0f, 7.02909f * density_factor * size_factor);
            z_velocity[k] = fminf(
                70.0f, 13.3436f * density_factor * size_factor);
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
        snowncv[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(2,
            (int)(__fdiv_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        for (int k = 0; k < nz; ++k) {
            mass_flux[k] = snow[k] * mass_velocity[k] * rho[k];
            number_flux[k] = number[k] * number_velocity[k];
            mass_number_flux[k] = number[k] * mass_velocity[k];
            number_mass_weighted[k] = number[k];
        }
        mass_flux[nz] = 0.0f;
        number_flux[nz] = 0.0f;
        mass_number_flux[nz] = 0.0f;

        surface_mean_flux +=
            snow[0] * mass_velocity[0] * dt_fraction;

        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            const float inverse_dz = 1.0f / dz[idx];
            snow[k] += dt_substep * inverse_dz / rho[k]
                * (mass_flux[k + 1] - mass_flux[k]);
            number[k] += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
            number_mass_weighted[k] += dt_substep * inverse_dz
                * (mass_number_flux[k + 1] - mass_number_flux[k]);
        }

        for (int k = 0; k < nz; ++k) {
            number[k] = fmaxf(number[k], number_mass_weighted[k]);
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        qs[idx] = snow[k];
        qns[idx] = number[k];
    }
    const float exported = dt * rho[0] * surface_mean_flux;
    snowncv[column] = exported;
    snownc[column] += exported;
}

#define NSSL2_SNOW_SEDIMENT_PARAMETERS                                  \
    const float* __restrict__ air_density, float* __restrict__ qs,       \
    float* __restrict__ qns, const float* __restrict__ dz,               \
    float* __restrict__ snownc, float* __restrict__ snowncv,             \
    float dt, int nz, int ny, int nx

extern "C" __global__ void nssl2_snow_sediment_64(
    NSSL2_SNOW_SEDIMENT_PARAMETERS)
{
    nssl2_snow_sediment_impl<NSSL2_KMAX_SHALLOW>(
        air_density, qs, qns, dz, snownc, snowncv, dt, nz, ny, nx);
}

extern "C" __global__ void nssl2_snow_sediment_256(
    NSSL2_SNOW_SEDIMENT_PARAMETERS)
{
    nssl2_snow_sediment_impl<NSSL2_KMAX_GENERIC>(
        air_density, qs, qns, dz, snownc, snowncv, dt, nz, ny, nx);
}

template <int KMAX>
__device__ __forceinline__ void nssl2_ice_sediment_impl(
    const float* __restrict__ air_density,
    float* __restrict__ qi,
    float* __restrict__ qni,
    const float* __restrict__ dz,
    float* __restrict__ icenc,
    float* __restrict__ icencv,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho[KMAX];
    float ice[KMAX];
    float number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float mass_flux[KMAX + 1];
    float number_flux[KMAX + 1];
    float mass_number_flux[KMAX + 1];
    float number_mass_weighted[KMAX];

    const float minimum_mass = 6.88e-13f;
    const float maximum_mass = 1.0e-8f;
    const float gamma_1p18 = 0.922766923904419f;
    const float gamma_2p18 = 1.091937899589539f;
    float maximum_courant_rate = 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        rho[k] = air_density[idx];
        ice[k] = qi[idx];
        number[k] = qni[idx];
        mass_velocity[k] = 0.0f;
        number_velocity[k] = 0.0f;

        const float positive_ice = fmaxf(ice[k], 0.0f);
        if (positive_ice > 1.0e-13f) {
            float local_number = fmaxf(number[k], 0.0f);
            local_number = fmaxf(
                local_number, __fdiv_rn(rho[k] * positive_ice, maximum_mass));
            local_number = fminf(
                local_number, __fdiv_rn(rho[k] * positive_ice, minimum_mass));
            const float particle_mass = fmaxf(
                rho[k] * positive_ice / local_number, minimum_mass);
            const float mean_volume = __fdiv_rn(particle_mass, 900.0f);
            const float density_factor = sqrtf(
                1.225f * fminf(20.0f, 1.0f / rho[k]));
            const float tmp = 47.6273f * density_factor
                / powf(1.0f / mean_volume, 0.18333f);
            number_velocity[k] = fminf(70.0f, tmp * gamma_1p18);
            mass_velocity[k] = fminf(70.0f, tmp * gamma_2p18);
        }
        const float inverse_dz = 1.0f / dz[idx];
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, mass_velocity[k] * inverse_dz);
        maximum_courant_rate = fmaxf(
            maximum_courant_rate, number_velocity[k] * inverse_dz);
    }

    if (maximum_courant_rate == 0.0f) {
        icencv[column] = 0.0f;
        return;
    }

    int substeps;
    if (dt * maximum_courant_rate < 0.7f) {
        substeps = 1;
    } else if (dt > 20.0f) {
        substeps = max(2,
            (int)(__fdiv_rn(dt * maximum_courant_rate, 0.7f)) + 1);
    } else {
        substeps = 1 + (int)(dt * maximum_courant_rate + 0.301f);
    }
    const float dt_substep = dt / (float)substeps;
    const float dt_fraction = dt_substep / dt;
    float surface_mean_flux = 0.0f;

    for (int step = 0; step < substeps; ++step) {
        for (int k = 0; k < nz; ++k) {
            mass_flux[k] = ice[k] * mass_velocity[k] * rho[k];
            number_flux[k] = number[k] * number_velocity[k];
            mass_number_flux[k] = number[k] * mass_velocity[k];
            number_mass_weighted[k] = number[k];
        }
        mass_flux[nz] = 0.0f;
        number_flux[nz] = 0.0f;
        mass_number_flux[nz] = 0.0f;

        surface_mean_flux += ice[0] * mass_velocity[0] * dt_fraction;

        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            const float inverse_dz = 1.0f / dz[idx];
            ice[k] += dt_substep * inverse_dz / rho[k]
                * (mass_flux[k + 1] - mass_flux[k]);
            number[k] += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
            number_mass_weighted[k] += dt_substep * inverse_dz
                * (mass_number_flux[k + 1] - mass_number_flux[k]);
        }

        for (int k = 0; k < nz; ++k) {
            number[k] = fmaxf(number[k], number_mass_weighted[k]);
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        qi[idx] = ice[k];
        qni[idx] = number[k];
    }
    const float exported = dt * rho[0] * surface_mean_flux;
    icencv[column] = exported;
    icenc[column] += exported;
}

#define NSSL2_ICE_SEDIMENT_PARAMETERS                                  \
    const float* __restrict__ air_density, float* __restrict__ qi,       \
    float* __restrict__ qni, const float* __restrict__ dz,               \
    float* __restrict__ icenc, float* __restrict__ icencv,               \
    float dt, int nz, int ny, int nx

extern "C" __global__ void nssl2_ice_sediment_64(
    NSSL2_ICE_SEDIMENT_PARAMETERS)
{
    nssl2_ice_sediment_impl<NSSL2_KMAX_SHALLOW>(
        air_density, qi, qni, dz, icenc, icencv, dt, nz, ny, nx);
}

extern "C" __global__ void nssl2_ice_sediment_256(
    NSSL2_ICE_SEDIMENT_PARAMETERS)
{
    nssl2_ice_sediment_impl<NSSL2_KMAX_GENERIC>(
        air_density, qi, qni, dz, icenc, icencv, dt, nz, ny, nx);
}


"""

@pytest.mark.parametrize("nz", [65, 128])
@pytest.mark.parametrize("category", ["cloud", "snow", "ice"])
def test_small_sediment_bits(nz, category):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device unavailable")
    from gpuwm.core.kernels import _preamble
    source = (Path(__file__).parents[1] / "gpuwm/core/kernels/nssl2_driver_support.cu").read_text()
    reference = cp.RawModule(code=_preamble() + _BASE_SMALL, options=("-std=c++17",))
    actual = cp.RawModule(code=_preamble() + source, options=("-std=c++17",))
    rng = np.random.default_rng(1804 + nz)
    shape = (nz, 1, 97)
    rho = cp.asarray(rng.uniform(0.3, 1.2, shape).astype(np.float32))
    dz = cp.asarray(rng.uniform(60, 500, shape).astype(np.float32))
    temperature = cp.asarray(rng.uniform(230, 290, shape).astype(np.float32))
    mass = rng.uniform(0, 0.0002, shape).astype(np.float32)
    mass[rng.random(shape) < 0.3] = 0
    mass[:, :, 0] = 0
    number = rng.uniform(0.01, 2e4, shape).astype(np.float32)
    for dt in (15, 45):
        outputs = []
        for module in (reference, actual):
            q, n = [cp.asarray(x.copy()) for x in (mass, number)]
            accum = cp.full((1, 97), np.float32(0.125))
            exported = cp.empty_like(accum)
            suffix = "64" if nz <= 64 else "256"
            route = ""
            name = f"nssl2_{category}_sediment_{route}{suffix}"
            grid = 97 if route else 2
            args = ((rho, temperature, q, n, dz, exported) if category == "cloud"
                    else (rho, q, n, dz, accum, exported))
            module.get_function(name)((grid,), (64,),
                (*args, np.float32(dt), np.int32(nz), np.int32(1), np.int32(97)))
            outputs.append([x.get().view(np.uint32) for x in (q, n, accum, exported)])
        for expected, observed in zip(*outputs):
            np.testing.assert_array_equal(expected, observed)
