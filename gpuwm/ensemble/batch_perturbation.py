"""Illustrative seeded uniform wind increments generated entirely on device.

This is an uncalibrated initial-condition wind ensemble. It supplies distinct,
reproducible inputs for a performance comparison, not a forecast-error model.
The same kernel and integer seed rule serve one member and a complete batch.
"""
from __future__ import annotations

import hashlib
import numpy as np

CONTRACT = "gpuwm-ensemble-uniform-wind-ic.v1"
_SOURCE = r'''
extern "C" __device__ __forceinline__ unsigned long long hash_seed(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
extern "C" __global__ void ensemble_uniform_wind_ic(
    float *values, const unsigned long long *seeds, unsigned long long cells,
    int members, unsigned long long axis_key, float amplitude) {
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cells * (unsigned long long)members) return;
    unsigned long long member = index / cells;
    unsigned int bits = (unsigned int)(hash_seed(seeds[member] ^ axis_key) >> 40);
    float unit = __fadd_rn(__fmul_rn(__uint2float_rn(bits), 0x1p-23f), -1.0f);
    values[index] = __fadd_rn(values[index], __fmul_rn(amplitude, unit));
}
'''


def initialize_member_winds(*, state, cfg, member_indices, seeds, phase,
                            amplitude=0.5, array_module=None):
    """Apply one const increment per member and wind axis to every C-grid face."""
    if phase != "after_physics_before_step":
        raise ValueError("uniform wind increments require initialized physics before stepping; cold-start fields must remain identical")
    if array_module is None:
        import cupy as array_module
    xp = array_module
    roster, seed_values = tuple(member_indices), tuple(seeds)
    if not roster or len(roster) != len(seed_values) or len(set(roster)) != len(roster):
        raise ValueError("wind initializer needs one unique member index and seed per array member")
    amplitude = np.float32(amplitude)
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("wind amplitude must be positive finite float32")
    members = len(roster)
    seeds_host = np.asarray(seed_values, dtype=np.uint64)
    seed_device = xp.asarray(seeds_host)
    from gpuwm.certify.kernel_manifest import record_module
    module = xp.RawModule(code=_SOURCE, options=("--std=c++17",))
    kernel = module.get_function("ensemble_uniform_wind_ic")
    record_module("gpuwm.ensemble.batch_perturbation:uniform-wind", source=_SOURCE,
                  options=("--std=c++17",), module=module)
    launches = []
    for axis, expected, key in (
        ("u", (cfg.nz, cfg.ny, cfg.nx + 1), 0x243F6A8885A308D3),
        ("v", (cfg.nz, cfg.ny + 1, cfg.nx), 0x13198A2E03707344),
    ):
        array = getattr(state, axis)
        shape = tuple(array.shape)
        if shape not in (expected, (members,) + expected) or (shape == expected and members != 1):
            raise ValueError(f"{axis} does not carry the requested complete member roster")
        if array.dtype != np.dtype("float32") or not array.flags.c_contiguous:
            raise ValueError(f"{axis} must be contiguous float32 C-grid storage")
        cells = int(np.prod(expected))
        kernel(((cells * members + 255) // 256,), (256,),
               (array, seed_device, np.uint64(cells), np.int32(members), np.uint64(key), amplitude))
        launches.append({"field": axis, "cells_per_member": cells, "axis_key": key})
    # The seed backing stays alive until both launches complete. Initialization
    # is a one-time stage, and synchronizing also bounds its measured wall time.
    xp.cuda.get_current_stream().synchronize()
    return {"contract": CONTRACT, "member_indices": list(roster),
            "seed_sha256": hashlib.sha256(seeds_host.tobytes()).hexdigest(),
            "amplitude_m_s": float(amplitude), "phase": phase,
            "kind": "illustrative uncalibrated uniform IC wind ensemble",
            "seed_allocation_bytes": int(seed_device.nbytes),
            "seed_pool_rounded_bytes": ((int(seed_device.nbytes) + 511) // 512) * 512,
            "kernel_source_sha256": hashlib.sha256(_SOURCE.encode()).hexdigest(),
            "launches": launches}
