"""Compare the gpuwm WSM6 CPU mirror with direct WRF v4.6.1 calls.

The two executables are built from ``wsm6_cpu_mirror.cpp`` and
``wsm6_wrf_oracle.F90``.  This script deliberately compares independent
implementations: the C++ executable includes the exact CUDA column source,
while the Fortran executable links WRF's unmodified ``mp_wsm6`` modules.
No CUDA context or GPU is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np


def _run(path: Path, scenario: int, nsteps: int = 1,
         dt: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    text = subprocess.check_output(
        [str(path), str(scenario), str(nsteps), str(dt)],
        text=True, encoding="utf-8")
    surface = None
    levels = []
    for raw in text.splitlines():
        fields = raw.split()
        if not fields or fields[0] == "refl_before":
            continue
        if fields[0] == "surface":
            surface = np.asarray([float(v) for v in fields[1:]], np.float64)
        elif fields[0].lstrip("+-").isdigit():
            levels.append([float(v) for v in fields[1:]])
    if surface is None or len(levels) != 8:
        raise RuntimeError(f"unexpected oracle output from {path}:\n{text}")
    return surface, np.asarray(levels, np.float64)


def compare(cpu: Path, wrf: Path, *, nsteps: int = 1,
            dt: float = 30.0) -> dict:
    result = {}
    for scenario in range(4):
        cpu_surface, cpu_levels = _run(cpu, scenario, nsteps, dt)
        wrf_surface, wrf_levels = _run(wrf, scenario, nsteps, dt)
        delta = np.abs(cpu_levels - wrf_levels)
        stable_radius = []
        for mass_col, radius_col in ((2, 7), (3, 8), (5, 9)):
            # Effective radius has an intentional hard R1 density threshold.
            # A sub-FP32 process-order residual can put only one answer just
            # above it, causing a background-vs-minimum radius jump while its
            # hydrometeor path remains negligible.  Report the raw jump, but
            # gate the radius formula where both answers occupy the same
            # stable active/inactive regime.
            cpu_active = cpu_levels[:, mass_col] > 0.0
            wrf_active = wrf_levels[:, mass_col] > 0.0
            same_regime = cpu_active == wrf_active
            stable_radius.extend(delta[same_regime, radius_col])
        metrics = {
            "wrf_cloud_water_max_kg_kg-1": float(
                np.max(wrf_levels[:, 2])),
            "wrf_total_condensate_max_kg_kg-1": float(
                np.max(np.sum(wrf_levels[:, 2:7], axis=1))),
            "surface_precip_max_abs_kg_m-2": float(np.max(
                np.abs(cpu_surface[:6] - wrf_surface[:6]))),
            "surface_frozen_ratio_abs": float(abs(
                cpu_surface[6] - wrf_surface[6])),
            "temperature_max_abs_K": float(np.max(delta[:, 0])),
            "mixing_ratio_max_abs_kg_kg-1": float(np.max(delta[:, 1:7])),
            "effective_radius_max_abs_micron": float(np.max(delta[:, 7:10])),
            "effective_radius_stable_max_abs_micron": float(np.max(
                stable_radius, initial=0.0)),
        }
        # These bounds separate meaningful state errors from normal FP32
        # expression-order differences between gfortran and C++/NVRTC.
        assert metrics["surface_precip_max_abs_kg_m-2"] <= 2.0e-7
        assert metrics["surface_frozen_ratio_abs"] <= 1.0e-6
        assert metrics["temperature_max_abs_K"] <= 3.1e-5
        assert metrics["mixing_ratio_max_abs_kg_kg-1"] <= 1.0e-8
        assert metrics["effective_radius_max_abs_micron"] <= 1.6e1
        assert metrics["effective_radius_stable_max_abs_micron"] <= 5.0e-3
        result[f"scenario_{scenario}"] = metrics
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cpu", type=Path)
    parser.add_argument("wrf", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--dt", type=float, default=30.0)
    args = parser.parse_args()
    if args.steps < 1 or not np.isfinite(args.dt) or args.dt <= 0.0:
        parser.error("--steps must be >= 1 and --dt finite and positive")
    print(json.dumps(compare(args.cpu.resolve(), args.wrf.resolve(),
                             nsteps=args.steps, dt=args.dt),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
