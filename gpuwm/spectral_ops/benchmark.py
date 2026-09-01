"""CPU/GPU synthetic benchmark and analytic controls."""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np

from .backend import asnumpy, backend_name, namespace, synchronize
from .elliptic import solve_helmholtz
from .scalar import hyperdiffuse
from .transfer import Hyperdiffusion
from .vector import damp_divergence, spectral_divergence


def run_benchmark(*, nx: int = 512, ny: int = 384, levels: int = 32,
                  dx_m: float = 3000.0, dy_m: float = 3000.0,
                  backend: str = "numpy", repeats: int = 5) -> dict[str, object]:
    if min(nx, ny, levels, repeats) < 1:
        raise ValueError("benchmark dimensions and repeats must be positive")
    if backend == "numpy":
        xp = np
    elif backend == "cupy":
        import cupy as xp
    else:
        raise ValueError("benchmark backend must be numpy or cupy")
    y = xp.arange(ny, dtype=xp.float32)[:, None]
    x = xp.arange(nx, dtype=xp.float32)[None, :]
    base = (xp.sin(2.0 * math.pi * x / 16.0)
            + 0.35 * xp.cos(2.0 * math.pi * y / 41.0))
    scalar = xp.broadcast_to(base, (levels, ny, nx)).copy()
    u = xp.broadcast_to(xp.sin(2.0 * math.pi * x / 23.0),
                        (levels, ny, nx)).copy()
    v = xp.broadcast_to(xp.cos(2.0 * math.pi * y / 29.0),
                        (levels, ny, nx)).copy()
    spec = Hyperdiffusion(order=3, reference_wavelength_m=6.0 * dx_m,
                          e_fold_time_s=300.0,
                          protect_wavelength_m=24.0 * dx_m)
    timings = {"scalar_s": [], "vector_s": [], "helmholtz_s": []}
    scalar_result = vector_result = helmholtz_result = None
    for _ in range(repeats):
        synchronize(scalar)
        start = time.perf_counter()
        scalar_result = hyperdiffuse(
            scalar, dy_m=dy_m, dx_m=dx_m, dt_s=10.0, spec=spec,
            boundary="periodic", periodic_domain=True)
        synchronize(scalar)
        timings["scalar_s"].append(time.perf_counter() - start)
        start = time.perf_counter()
        vector_result = damp_divergence(
            u, v, dy_m=dy_m, dx_m=dx_m, dt_s=10.0,
            divergent=spec, boundary="periodic", periodic_domain=True)
        synchronize(u)
        timings["vector_s"].append(time.perf_counter() - start)
        start = time.perf_counter()
        helmholtz_result = solve_helmholtz(
            scalar, dy_m=dy_m, dx_m=dx_m, alpha=1.0, beta=1.0e8,
            boundary="periodic", periodic_domain=True)
        synchronize(scalar)
        timings["helmholtz_s"].append(time.perf_counter() - start)
    assert scalar_result and vector_result and helmholtz_result
    return {
        "schema": "gpuwm.spectral-operator-benchmark/v1",
        "backend": backend,
        "shape": [levels, ny, nx],
        "dx_m": dx_m,
        "dy_m": dy_m,
        "repeats": repeats,
        "median_seconds": {
            name: float(np.median(values)) for name, values in timings.items()},
        "controls": {
            "scalar_mean_drift": scalar_result.mean_after - scalar_result.mean_before,
            "scalar_rms_increment": scalar_result.rms_increment,
            "divergence_ratio": (
                vector_result.divergence_rms_after
                / max(vector_result.divergence_rms_before, 1.0e-30)),
            "helmholtz_residual_rms": helmholtz_result.residual_rms,
        },
    }


__all__ = ["run_benchmark"]
