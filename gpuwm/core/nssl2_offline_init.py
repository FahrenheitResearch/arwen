"""Bounded host adapter to the already qualified NSSL calcnfromq kernel.

Offline remapping keeps its fields on the host. The target's existing CUDA
initializer supplies the same missing moments, CCN consumption and negligible
mass return as its ordinary initialization, without a second numerical law.
"""
from __future__ import annotations

import numpy as np

from gpuwm.core.nssl2 import launch_initial_state


FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop", "qnr",
          "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh")
CHUNK_CELLS = 262144


def initialize_missing_moments(fields, air_density):
    """Apply native calcnfromq to complete state, preserving its water updates."""
    import cupy as cp

    rho = np.ascontiguousarray(air_density, dtype=np.float32)
    if not rho.size or not np.isfinite(rho).all() or np.any(rho <= 0):
        raise ValueError("NSSL initialization requires finite positive child air density")
    values = {}
    for name in FIELDS:
        value = np.asarray(fields.get(name, np.zeros(rho.shape, np.float32)),
                           dtype=np.float32)
        if value.shape != rho.shape or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f"NSSL initialization has invalid {name}")
        values[name] = np.ascontiguousarray(value).copy()
    flat = {name: value.reshape(-1) for name, value in values.items()}
    density = rho.reshape(-1)
    drift = 0.0
    for start in range(0, rho.size, CHUNK_CELLS):
        stop = min(start + CHUNK_CELLS, rho.size)
        before = sum(flat[name][start:stop].astype(np.float64)
                     for name in FIELDS[:7])
        device_rho = cp.asarray(density[start:stop])
        arrays = [cp.asarray(flat[name][start:stop]) for name in FIELDS]
        launch_initial_state(device_rho, *arrays)
        for name, device in zip(FIELDS, arrays):
            flat[name][start:stop] = cp.asnumpy(device)
        after = sum(flat[name][start:stop].astype(np.float64)
                    for name in FIELDS[:7])
        drift = max(drift, float(np.max(np.abs(after - before))))
        del arrays, device_rho
    return values, {
        "method": "gpuwm.core.nssl2.launch_initial_state",
        "authority": "WRF v4.6.1 module_mp_nssl_2mom.F:5168-5513 calcnfromq",
        "density": "child dry mass / actual geopotential layer thickness",
        "chunk_cells": CHUNK_CELLS,
        "mass_return_to_vapor_preserved": True,
        "max_total_water_change_kg_per_kg": drift,
    }
