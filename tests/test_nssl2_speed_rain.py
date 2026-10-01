"""Frozen-base bit oracle for rain level mapping."""
from pathlib import Path

import numpy as np
import pytest

# A146: this frozen copy's divisions by a compile-time constant are spelled
# __fdiv_rn, as the production kernel has spelled them since that fix
# (NVRTC compiled them as reciprocal multiplies on Blackwell).  On sm_89
# both spellings are the same div.rn, so the copy is still the frozen base.
_BASE_RAIN = r"""
#define NSSL2_KMAX_SHALLOW 64
#define NSSL2_KMAX_GENERIC 256
__device__ __forceinline__ float nssl2_rain_z(
    float q, float number, float rho)
{
    if (!(q > 1.0e-12f) || !(number > 1.0e-15f)) return 0.0f;

    const float minimum_volume =
        0.523599f * (80.0e-6f * 80.0e-6f * 80.0e-6f);
    const float maximum_volume =
        0.523599f * (6.0e-3f * 6.0e-3f * 6.0e-3f);
    float mean_volume = rho * q / (1000.0f * number);
    float effective_number = number;
    if (mean_volume < minimum_volume || mean_volume > maximum_volume) {
        mean_volume = fminf(maximum_volume,
                            fmaxf(minimum_volume, mean_volume));
        effective_number = rho * q / (1000.0f * mean_volume);
    }
    const float z_factor =
        (6.0f / (3.14159265358979323846f * 1000.0f))
        * (6.0f / (3.14159265358979323846f * 1000.0f));
    return 120.0f * rho * rho * q * q / effective_number * z_factor;
}

template <int KMAX>
__device__ __forceinline__ void nssl2_rain_sediment_impl(
    const float* __restrict__ air_density,
    float* __restrict__ qr,
    float* __restrict__ qnr,
    const float* __restrict__ dz,
    float* __restrict__ rainnc,
    float* __restrict__ rainncv,
    float dt, int nz, int ny, int nx)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= ny * nx) return;
    const int j = column / nx;
    const int i = column - j * nx;

    float rho[KMAX];
    float rain[KMAX];
    float number[KMAX];
    float mass_velocity[KMAX];
    float number_velocity[KMAX];
    float z_velocity[KMAX];
    float mass_flux[KMAX + 1];
    float number_flux[KMAX + 1];
    float z_flux[KMAX + 1];
    float mass_number_flux[KMAX + 1];
    float z_initial[KMAX];
    float z_advected[KMAX];
    float number_mass_weighted[KMAX];

    const float pi = 3.14159265358979323846f;
    const float minimum_volume =
        0.523599f * (80.0e-6f * 80.0e-6f * 80.0e-6f);
    const float configured_maximum_volume =
        0.523599f * (6.0e-3f * 6.0e-3f * 6.0e-3f);
    const float maximum_speed_volume =
        configured_maximum_volume / (64.0f / 6.0f);
    float maximum_courant_rate = 0.0f;

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        rho[k] = air_density[idx];
        rain[k] = qr[idx];
        number[k] = qnr[idx];
        mass_velocity[k] = 0.0f;
        number_velocity[k] = 0.0f;
        z_velocity[k] = 0.0f;

        const float positive_rain = fmaxf(rain[k], 0.0f);
        if (positive_rain > 1.0e-12f) {
            const float local_number = fmaxf(number[k], 0.0f);
            float mean_volume = rho[k] * positive_rain
                / (1000.0f * fmaxf(1.0e-11f, local_number));
            if (mean_volume > maximum_speed_volume) {
                mean_volume = maximum_speed_volume;
            } else if (mean_volume < minimum_volume) {
                mean_volume = minimum_volume;
            }

            const float diameter = powf(
                __fdiv_rn((6.0f / pi) * mean_volume, (3.0f * 2.0f * 1.0f)),
                1.0f / 3.0f);
            const float density_factor = sqrtf(
                1.225f * fminf(20.0f, 1.0f / rho[k]));
            const float speed_base = 1.0f + 516.575f * diameter;
            float vm = density_factor * 10.0f
                * (1.0f - powf(speed_base, -4.0f));
            float vn = density_factor * 10.0f
                * (1.0f - powf(speed_base, -1.0f));
            float vz = density_factor * 10.0f
                * (1.0f - powf(speed_base, -7.0f));
            if (vn > vm || (vm > vz && vz > 0.0f)) {
                vm = fmaxf(vm, vn);
                vz = fmaxf(vz, vm);
            }
            mass_velocity[k] = fminf(70.0f, fminf(150.0f, vm));
            number_velocity[k] = fminf(70.0f, fminf(150.0f, vn));
            z_velocity[k] = fminf(70.0f, fminf(150.0f, vz));
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
        rainncv[column] = 0.0f;
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
        // Diagnose the pre-fallout reflectivity moment and preserve the
        // pre-fallout number for the parallel mass-weighted correction.
        for (int k = 0; k < nz; ++k) {
            z_initial[k] = nssl2_rain_z(rain[k], number[k], rho[k]);
            z_advected[k] = z_initial[k];
            number_mass_weighted[k] = number[k];
            mass_flux[k] = rain[k] * mass_velocity[k] * rho[k];
            number_flux[k] = number[k] * number_velocity[k];
            z_flux[k] = z_initial[k] * z_velocity[k];
            mass_number_flux[k] = number[k] * mass_velocity[k];
        }
        mass_flux[nz] = 0.0f;
        number_flux[nz] = 0.0f;
        z_flux[nz] = 0.0f;
        mass_number_flux[nz] = 0.0f;

        surface_mean_flux +=
            rain[0] * mass_velocity[0] * dt_fraction;

        // fallout1d computes every flux first, then updates every level.
        for (int k = 0; k < nz; ++k) {
            const size_t idx = IDX3(k, j, i);
            const float inverse_dz = 1.0f / dz[idx];
            rain[k] += dt_substep * inverse_dz / rho[k]
                * (mass_flux[k + 1] - mass_flux[k]);
            number[k] += dt_substep * inverse_dz
                * (number_flux[k + 1] - number_flux[k]);
            z_advected[k] += dt_substep * inverse_dz
                * (z_flux[k + 1] - z_flux[k]);
            number_mass_weighted[k] += dt_substep * inverse_dz
                * (mass_number_flux[k + 1] - mass_number_flux[k]);
        }

        // calcnfromz1d uses double temporaries for the inverse-Z number
        // reconstruction, but stores REAL(Nz) before its max/min correction.
        for (int k = 0; k < nz; ++k) {
            if (z_advected[k] > 0.0f) {
                const float diagnosed =
                    nssl2_rain_z(rain[k], number[k], rho[k]);
                if (diagnosed > z_advected[k]
                        && z_advected[k] > z_initial[k]) {
                    const double z_factor =
                        (double)(6.0f / (pi * 1000.0f))
                        * (double)(6.0f / (pi * 1000.0f));
                    const double reconstructed =
                        120.0 * (double)rho[k] * (double)rho[k]
                        * (double)rain[k] * (double)rain[k]
                        / ((double)z_advected[k] / z_factor);
                    const float reconstructed_real = (float)reconstructed;
                    number[k] = fmaxf(
                        fminf(reconstructed_real,
                              number_mass_weighted[k]),
                        number[k]);
                } else {
                    number[k] = fmaxf(number_mass_weighted[k], number[k]);
                }
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        const size_t idx = IDX3(k, j, i);
        qr[idx] = rain[k];
        qnr[idx] = number[k];
    }
    const float exported = dt * rho[0] * surface_mean_flux;
    rainncv[column] = exported;
    rainnc[column] += exported;
}

#define NSSL2_RAIN_SEDIMENT_PARAMETERS                                  \
    const float* __restrict__ air_density, float* __restrict__ qr,       \
    float* __restrict__ qnr, const float* __restrict__ dz,               \
    float* __restrict__ rainnc, float* __restrict__ rainncv,             \
    float dt, int nz, int ny, int nx

extern "C" __global__ void nssl2_rain_sediment_64(
    NSSL2_RAIN_SEDIMENT_PARAMETERS)
{
    nssl2_rain_sediment_impl<NSSL2_KMAX_SHALLOW>(
        air_density, qr, qnr, dz, rainnc, rainncv, dt, nz, ny, nx);
}

extern "C" __global__ void nssl2_rain_sediment_256(
    NSSL2_RAIN_SEDIMENT_PARAMETERS)
{
    nssl2_rain_sediment_impl<NSSL2_KMAX_GENERIC>(
        air_density, qr, qnr, dz, rainnc, rainncv, dt, nz, ny, nx);
}


"""

@pytest.mark.parametrize("nz", [65, 128])
def test_rain_sediment_bits(nz):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device unavailable")
    from gpuwm.core.kernels import _preamble
    source = (Path(__file__).parents[1] / "gpuwm/core/kernels/nssl2_driver_support.cu").read_text()
    reference = cp.RawModule(code=_preamble() + _BASE_RAIN, options=("-std=c++17",))
    actual = cp.RawModule(code=_preamble() + source, options=("-std=c++17",))
    rng = np.random.default_rng(1803 + nz)
    shape = (nz, 1, 97)
    rho = cp.asarray(rng.uniform(0.3, 1.2, shape).astype(np.float32))
    dz = cp.asarray(rng.uniform(60, 500, shape).astype(np.float32))
    mass = rng.uniform(0, 0.006, shape).astype(np.float32)
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
            name = f"nssl2_rain_sediment_{route}{suffix}"
            grid = 97 if route else 2
            module.get_function(name)((grid,), (64,),
                (rho, q, n, dz, accum, exported, np.float32(dt),
                 np.int32(nz), np.int32(1), np.int32(97)))
            outputs.append([x.get().view(np.uint32) for x in (q, n, accum, exported)])
        for expected, observed in zip(*outputs):
            np.testing.assert_array_equal(expected, observed)
