"""CPU-vs-GPU parity and per-call cost for the Level-2 spectral operators.

Ladder step 5 of the delivered Level-2 prompt.  The package pins the
arithmetic precision as "transform arithmetic follows the input/backend FFT
dtype; receipt reductions use host float64", so parity is held to the FFT
round-off bound that statement implies and NOT to bitwise equality: numpy's
pocketfft and cuFFT are different implementations of the same transform.

For an N-point FFT of a field with root-mean-square amplitude ``rms`` the
accumulated round-off of a well-conditioned implementation is bounded by
roughly ``eps * sqrt(log2(N)) * rms``; this harness uses the looser and
easier to defend ``eps * sqrt(N_2d) * rms`` where ``N_2d = ny * nx`` is the
transform length, and reports the realized ratio against it so a regression
that eats the margin is visible rather than merely under the bar.

Run:  python tools/spectral_gpu_parity.py --json OUT.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time

import numpy as np

from gpuwm.spectral_ops.elliptic import solve_helmholtz
from gpuwm.spectral_ops.scalar import hyperdiffuse
from gpuwm.spectral_ops.transfer import Hyperdiffusion
from gpuwm.spectral_ops.vector import damp_c_grid_divergence, damp_divergence


def _fields(levels: int, ny: int, nx: int, dtype) -> dict[str, np.ndarray]:
    """One deterministic synthetic state; identical bytes feed both backends."""
    rng = np.random.default_rng(20260817)
    z = np.arange(levels, dtype=np.float64)[:, None, None]
    y = np.arange(ny, dtype=np.float64)[None, :, None]
    x = np.arange(nx, dtype=np.float64)[None, None, :]
    smooth = (290.0
              + 8.0 * np.sin(2.0 * math.pi * x / 37.0)
              + 5.0 * np.cos(2.0 * math.pi * y / 23.0)
              - 0.9 * z)
    theta = smooth + 0.6 * rng.standard_normal((levels, ny, nx))
    qv = np.exp(-8.5 - 0.05 * z) * (1.0 + 0.3 * rng.random((levels, ny, nx)))
    u = (12.0 * np.sin(2.0 * math.pi * y / 19.0)
         + 2.0 * rng.standard_normal((levels, ny, nx + 1)))
    v = (-7.0 * np.cos(2.0 * math.pi * x / 29.0)
         + 2.0 * rng.standard_normal((levels, ny + 1, nx)))
    out = {"theta": theta, "qv": qv, "u": u, "v": v,
           "u_mass": u[:, :, :nx].copy(), "v_mass": v[:, :ny, :].copy()}
    return {k: np.ascontiguousarray(val, dtype=dtype) for k, val in out.items()}


def _bound(reference: np.ndarray, dtype) -> float:
    """The FFT round-off bound the pins imply for this dtype and length."""
    eps = float(np.finfo(dtype).eps)
    ny, nx = reference.shape[-2:]
    rms = float(np.sqrt(np.mean(np.asarray(reference, dtype=np.float64) ** 2)))
    return eps * math.sqrt(float(ny * nx)) * max(rms, 1.0e-30)


def _compare(name: str, cpu: np.ndarray, gpu, reference: np.ndarray,
             dtype) -> dict[str, object]:
    import cupy as cp
    host = cp.asnumpy(gpu)
    cpu = np.asarray(cpu)
    if host.shape != cpu.shape:
        raise AssertionError(f"{name}: shape {host.shape} != {cpu.shape}")
    if host.dtype != cpu.dtype:
        raise AssertionError(f"{name}: dtype {host.dtype} != {cpu.dtype}")
    diff = np.abs(host.astype(np.float64) - cpu.astype(np.float64))
    bound = _bound(reference, dtype)
    max_abs = float(diff.max())
    bitwise = bool(np.array_equal(host, cpu))
    return {
        "field": name,
        "dtype": str(np.dtype(dtype)),
        "shape": list(cpu.shape),
        "max_abs_difference": max_abs,
        "roundoff_bound": bound,
        "ratio_to_bound": max_abs / bound if bound else float("inf"),
        "bitwise_identical": bitwise,
        "within_bound": max_abs <= bound,
    }


def _scalar_pairs(cpu_receipt, gpu_receipt, keys) -> list[dict[str, object]]:
    rows = []
    for key in keys:
        a = float(getattr(cpu_receipt, key))
        b = float(getattr(gpu_receipt, key))
        scale = max(abs(a), abs(b), 1.0e-30)
        rows.append({"metric": key, "cpu": a, "gpu": b,
                     "relative_difference": abs(a - b) / scale})
    return rows


def parity(levels: int, ny: int, nx: int, dx: float, dt: float,
           dtype) -> dict[str, object]:
    import cupy as cp
    host = _fields(levels, ny, nx, dtype)
    dev = {k: cp.asarray(v) for k, v in host.items()}
    for k, v in host.items():
        assert np.array_equal(cp.asnumpy(dev[k]), v), f"upload changed {k}"
    spec = Hyperdiffusion(order=3, reference_wavelength_m=6.0 * dx,
                          e_fold_time_s=300.0,
                          protect_wavelength_m=24.0 * dx)
    wind_spec = Hyperdiffusion(order=2, reference_wavelength_m=4.0 * dx,
                               e_fold_time_s=300.0,
                               protect_wavelength_m=16.0 * dx,
                               maximum_damping_fraction=0.5)
    fields: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []

    for boundary, periodic in (("tapered", False), ("reflect", False),
                               ("periodic", True)):
        kw = dict(dy_m=dx, dx_m=dx, dt_s=dt, spec=spec, boundary=boundary,
                  edge_taper_cells=12, periodic_domain=periodic)
        a = hyperdiffuse(host["theta"], **kw)
        b = hyperdiffuse(dev["theta"], **kw)
        fields.append(_compare(f"hyperdiffuse[{boundary}].theta",
                               a.values, b.values, host["theta"], dtype))
        metrics.append({"operator": f"hyperdiffuse[{boundary}]",
                        "rows": _scalar_pairs(a, b, (
                            "mean_before", "mean_after", "rms_increment",
                            "max_abs_increment"))})

    kw = dict(dy_m=dx, dx_m=dx, dt_s=dt, spec=spec, boundary="tapered",
              edge_taper_cells=12, periodic_domain=False, space="log",
              floor=1.0e-12)
    a = hyperdiffuse(host["qv"], **kw)
    b = hyperdiffuse(dev["qv"], **kw)
    fields.append(_compare("hyperdiffuse[log].qv", a.values, b.values,
                           host["qv"], dtype))
    metrics.append({"operator": "hyperdiffuse[log]",
                    "rows": _scalar_pairs(a, b, (
                        "mean_before", "mean_after", "rms_increment",
                        "minimum_after"))})

    kw = dict(dy_m=dx, dx_m=dx, dt_s=dt, divergent=wind_spec,
              boundary="tapered", edge_taper_cells=12, periodic_domain=False)
    a = damp_divergence(host["u_mass"], host["v_mass"], **kw)
    b = damp_divergence(dev["u_mass"], dev["v_mass"], **kw)
    fields.append(_compare("damp_divergence.u", a.u, b.u, host["u_mass"], dtype))
    fields.append(_compare("damp_divergence.v", a.v, b.v, host["v_mass"], dtype))
    metrics.append({"operator": "damp_divergence",
                    "rows": _scalar_pairs(a, b, (
                        "divergence_rms_before", "divergence_rms_after",
                        "kinetic_energy_before", "kinetic_energy_after",
                        "rms_increment", "max_abs_increment"))})

    a = damp_c_grid_divergence(host["u"], host["v"], **kw)
    b = damp_c_grid_divergence(dev["u"], dev["v"], **kw)
    fields.append(_compare("damp_c_grid_divergence.u", a.u, b.u, host["u"],
                           dtype))
    fields.append(_compare("damp_c_grid_divergence.v", a.v, b.v, host["v"],
                           dtype))
    metrics.append({"operator": "damp_c_grid_divergence",
                    "rows": _scalar_pairs(a, b, (
                        "divergence_rms_before", "divergence_rms_after",
                        "rms_increment", "max_abs_increment"))})

    kw = dict(dy_m=dx, dx_m=dx, alpha=1.0, beta=1.0e8, boundary="periodic",
              periodic_domain=True)
    a = solve_helmholtz(host["theta"], **kw)
    b = solve_helmholtz(dev["theta"], **kw)
    fields.append(_compare("solve_helmholtz.values", a.values, b.values,
                           host["theta"], dtype))
    metrics.append({"operator": "solve_helmholtz",
                    "rows": _scalar_pairs(a, b, ("residual_rms",
                                                 "rhs_mean_removed"))})
    return {"fields": fields, "metrics": metrics}


def cost(levels: int, ny: int, nx: int, dx: float, dt: float, dtype,
         repeats: int) -> dict[str, object]:
    """Per-call wall cost of ONE full hook: two scalars plus C-grid wind."""
    import cupy as cp
    from gpuwm.spectral_ops import SpectralLargeStepHook
    from gpuwm.spectral_ops.config import (ScalarTarget,
                                           SpectralNumericsConfig, WindControl)
    spec = Hyperdiffusion(order=3, reference_wavelength_m=6.0 * dx,
                          e_fold_time_s=300.0,
                          protect_wavelength_m=24.0 * dx)
    wind_spec = Hyperdiffusion(order=2, reference_wavelength_m=4.0 * dx,
                               e_fold_time_s=300.0,
                               protect_wavelength_m=16.0 * dx,
                               maximum_damping_fraction=0.5)
    config = SpectralNumericsConfig(
        mode="shadow", boundary="tapered", edge_taper_cells=12,
        scalar_targets=(ScalarTarget(field="theta", diffusion=spec),
                        ScalarTarget(field="qv", diffusion=spec,
                                     space="log", floor=1.0e-12)),
        wind=WindControl(enabled=True, u_field="u", v_field="v",
                         staggering="cgrid", divergent=wind_spec))
    host = _fields(levels, ny, nx, dtype)
    out: dict[str, object] = {}
    for backend in ("numpy", "cupy"):
        if backend == "cupy":
            state = {k: cp.asarray(v) for k, v in host.items()}
        else:
            state = {k: v.copy() for k, v in host.items()}
        hook = SpectralLargeStepHook(config=config, dx_m=dx, dy_m=dx, dt_s=dt,
                                     domain="d01")
        hook(state, large_step=1)  # warm caches / cuFFT plans
        samples = []
        for step in range(2, 2 + repeats):
            if backend == "cupy":
                cp.cuda.get_current_stream().synchronize()
            start = time.perf_counter()
            hook(state, large_step=step)
            if backend == "cupy":
                cp.cuda.get_current_stream().synchronize()
            samples.append(time.perf_counter() - start)
        out[backend] = {
            "median_s": float(np.median(samples)),
            "min_s": float(min(samples)),
            "max_s": float(max(samples)),
            "samples": [float(s) for s in samples],
        }
    out["speedup_cpu_over_gpu"] = (out["numpy"]["median_s"]
                                   / out["cupy"]["median_s"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, default=49)
    parser.add_argument("--ny", type=int, default=400)
    parser.add_argument("--nx", type=int, default=480)
    parser.add_argument("--dx", type=float, default=3000.0)
    parser.add_argument("--dt", type=float, default=15.0)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--json")
    args = parser.parse_args()

    import cupy as cp
    device = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    report: dict[str, object] = {
        "schema": "gpuwm.spectral-gpu-parity/v1",
        "host": platform.node(),
        "device": props["name"].decode(),
        "compute_capability": f"{device.compute_capability}",
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "numpy": np.__version__,
        "shape": [args.levels, args.ny, args.nx],
        "dx_m": args.dx,
        "dt_s": args.dt,
        "tolerance_rule": (
            "pins: transform arithmetic follows the input/backend FFT dtype. "
            "Bound = eps(dtype) * sqrt(ny*nx) * rms(reference field); no "
            "bitwise equality is claimed between pocketfft and cuFFT."),
        "parity": {},
    }
    for dtype in (np.float32, np.float64):
        report["parity"][str(np.dtype(dtype))] = parity(
            args.levels, args.ny, args.nx, args.dx, args.dt, dtype)
    report["cost"] = cost(args.levels, args.ny, args.nx, args.dx, args.dt,
                          np.float32, args.repeats)

    failures = [row for block in report["parity"].values()
                for row in block["fields"] if not row["within_bound"]]
    report["parity_failures"] = failures
    report["verdict"] = "PASS" if not failures else "FAIL"

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text + "\n")
    print(text)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
