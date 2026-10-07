"""CPU verification arithmetic for UPP's source-level smoke inverse.

UPP 1296eebb295251d0fe1cc697f47058c62d887977 MDLFLD.f:2442;
params.F:57. Production arithmetic lives in the Rust bridge.
"""
import numpy as np


def effective_density(pressure, temperature):
    pressure, temperature = (np.asarray(a, dtype=np.float32)
                             for a in (pressure, temperature))
    return np.float32(np.float32(1) / np.float32(287.04)) * np.float32(pressure / temperature)


def mass_density_forward(mixing_ratio, pressure, temperature):
    rho = effective_density(pressure, temperature)
    return np.float32(np.float32(rho * np.asarray(mixing_ratio, np.float32)) / np.float32(1e9))


def mass_density_inverse(mass, pressure, temperature):
    mass = np.asarray(mass, dtype=np.float32)
    rho = effective_density(pressure, temperature)
    return np.float32(np.float32(mass / rho) * np.float32(1e9))
