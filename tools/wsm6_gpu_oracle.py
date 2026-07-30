"""Run the CUDA WSM6 kernels against frozen direct-WRF v4.6.1 answers.

This is the first-device gate for the WSM6 port.  It intentionally avoids
the model driver: four one-column states isolate warm-cloud, cold-cloud,
graupel, hail, PLM sedimentation, effective radii, and native reflectivity.
The reference JSON was emitted by ``tools/wsm6_wrf_oracle.F90`` linked
against the unmodified local WRF v4.6.1 sources.

Run on a rented Linux GPU; this tool allocates CUDA memory and JIT-compiles
the production kernels.  It must not be run while another job owns a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORACLE = ROOT / "tests" / "data" / "wsm6_wrf461_oracle.json"

GATES = {
    "reflectivity_before_max_abs_dBZ": 2.0e-4,
    "surface_precip_max_abs_kg_m-2": 5.0e-6,
    "surface_frozen_ratio_abs": 5.0e-5,
    "temperature_max_abs_K": 2.0e-4,
    "mixing_ratio_max_abs_kg_kg-1": 2.0e-8,
    # WRF effectRad has a hard positive-mass branch.  A residual at the
    # FP32 cancellation floor can put only one implementation on that branch.
    "effective_radius_max_abs_micron": 16.1,
    "effective_radius_stable_max_abs_micron": 3.0e-2,
}


def _device_name(cp) -> str:
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props["name"]
    return name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name)


def _initial_column(scenario: int) -> dict[str, np.ndarray]:
    """Reproduce the IEEE-binary32 setup in wsm6_wrf_oracle.F90."""
    k = np.arange(8, dtype=np.float32)
    if scenario == 0:
        temperature = np.float32(286.0) - np.float32(0.7) * k
        qv = np.float32(0.011) - np.float32(0.0007) * k
        qc = np.float32(7.0e-4) + np.float32(2.0e-5) * k
    elif scenario == 1:
        temperature = np.float32(269.0) - np.float32(1.8) * k
        qv = np.float32(0.0038) - np.float32(0.00025) * k
        qc = np.full(8, np.float32(2.0e-5), dtype=np.float32)
    else:
        temperature = np.float32(280.0) - np.float32(2.0) * k
        qv = np.float32(0.0045) - np.float32(0.0002) * k
        qc = np.full(8, np.float32(1.0e-4), dtype=np.float32)

    pressure = np.float32(96000.0) - np.float32(7000.0) * k
    rho = pressure / (np.float32(287.0) * temperature)
    pii = np.power(
        pressure / np.float32(100000.0),
        np.float32(287.0) / np.float32(1004.5),
    ).astype(np.float32)
    theta = (temperature / pii).astype(np.float32)
    dz = np.float32(500.0) + np.float32(25.0) * k

    zeros = np.zeros(8, dtype=np.float32)
    if scenario >= 2:
        qi = np.float32(2.0e-5) + np.float32(1.0e-6) * k
        qr = np.float32(2.0e-4) + np.float32(1.0e-5) * k
        qs = np.float32(1.2e-4) + np.float32(8.0e-6) * k
        qg = np.float32(5.0e-5) + np.float32(5.0e-6) * k
    else:
        qi = qr = qs = qg = zeros

    return {
        "temperature": temperature,
        "theta": theta,
        "qv": qv,
        "qc": qc,
        "qi": qi.copy(),
        "qr": qr.copy(),
        "qs": qs.copy(),
        "qg": qg.copy(),
        "rho": rho.astype(np.float32),
        "pii": pii,
        "pressure": pressure,
        "dz": dz,
    }


def _scenario(cp, scenario: int, reference: dict) -> dict:
    from gpuwm.core.refl import launch_refl10cm_wsm6
    from gpuwm.core.wsm6 import launch_wsm6

    initial = _initial_column(scenario)
    dev = {
        name: cp.asarray(value.reshape(8, 1, 1), dtype=cp.float32)
        for name, value in initial.items()
        if name != "temperature"
    }
    hail_opt = int(reference["hail_opt"])
    surface = {
        name: cp.zeros((1, 1), dtype=cp.float32)
        for name in (
            "rainnc", "rainncv", "snownc", "snowncv",
            "graupelnc", "graupelncv", "sr",
        )
    }
    effc = cp.full((8, 1, 1), np.float32(2.49), dtype=cp.float32)
    effi = cp.full((8, 1, 1), np.float32(4.99), dtype=cp.float32)
    effs = cp.full((8, 1, 1), np.float32(9.99), dtype=cp.float32)

    refl = cp.empty((8, 1, 1), dtype=cp.float32)
    temperature_before = cp.asarray(
        initial["temperature"].reshape(8, 1, 1), dtype=cp.float32)
    launch_refl10cm_wsm6(
        dev["qv"], dev["qr"], dev["qs"], dev["qg"],
        temperature_before, dev["pressure"], refl, hail_opt=hail_opt,
    )

    launch_wsm6(
        dev["theta"], dev["qv"], dev["qc"], dev["qr"], dev["qi"],
        dev["qs"], dev["qg"], dev["rho"], dev["pii"], dev["pressure"],
        dev["dz"], surface["rainnc"], surface["rainncv"],
        surface["snownc"], surface["snowncv"], surface["graupelnc"],
        surface["graupelncv"], surface["sr"], 30.0,
        effc=effc, effi=effi, effs=effs, hail_opt=hail_opt,
    )
    cp.cuda.runtime.deviceSynchronize()

    levels = cp.stack(
        (
            dev["theta"] * dev["pii"], dev["qv"], dev["qc"], dev["qi"],
            dev["qr"], dev["qs"], dev["qg"], effc, effi, effs,
        ),
        axis=-1,
    )[:, 0, 0, :].get().astype(np.float64)
    refl_host = refl[:, 0, 0].get().astype(np.float64)
    surface_host = cp.stack(
        tuple(surface[name] for name in (
            "rainnc", "rainncv", "snownc", "snowncv",
            "graupelnc", "graupelncv", "sr",
        ))
    ).get().reshape(7).astype(np.float64)
    expected_levels = np.asarray(reference["levels"], dtype=np.float64)
    expected_surface = np.asarray(reference["surface"], dtype=np.float64)
    expected_refl = np.asarray(reference["reflectivity_before"], dtype=np.float64)
    delta = np.abs(levels - expected_levels)

    stable_radius_delta: list[float] = []
    for mass_col, radius_col in ((2, 7), (3, 8), (5, 9)):
        # Gate the radius formula only when both computations take the same
        # active/background branch; retain the raw jump as a separate gate.
        same_regime = ((levels[:, mass_col] > 0.0)
                       == (expected_levels[:, mass_col] > 0.0))
        stable_radius_delta.extend(delta[same_regime, radius_col].tolist())

    metrics = {
        "reflectivity_before_max_abs_dBZ": float(np.max(
            np.abs(refl_host - expected_refl))),
        "surface_precip_max_abs_kg_m-2": float(np.max(
            np.abs(surface_host[:6] - expected_surface[:6]))),
        "surface_frozen_ratio_abs": float(abs(
            surface_host[6] - expected_surface[6])),
        "temperature_max_abs_K": float(np.max(delta[:, 0])),
        "mixing_ratio_max_abs_kg_kg-1": float(np.max(delta[:, 1:7])),
        "effective_radius_max_abs_micron": float(np.max(delta[:, 7:10])),
        "effective_radius_stable_max_abs_micron": float(max(
            stable_radius_delta, default=0.0)),
        "minimum_hydrometeor_kg_kg-1": float(np.min(levels[:, 2:7])),
    }
    finite = bool(np.isfinite(levels).all()
                  and np.isfinite(surface_host).all()
                  and np.isfinite(refl_host).all())
    failures = [
        f"{name}={metrics[name]:.9g} exceeds {bound:.9g}"
        for name, bound in GATES.items()
        if not np.isfinite(metrics[name]) or metrics[name] > bound
    ]
    if not finite:
        failures.append("non-finite CUDA result")
    if metrics["minimum_hydrometeor_kg_kg-1"] < -1.0e-12:
        failures.append(
            "negative hydrometeor below -1e-12 kg kg-1: "
            f"{metrics['minimum_hydrometeor_kg_kg-1']:.9g}")
    return {"ok": not failures, "metrics": metrics, "failures": failures}


def run(oracle_path: Path, require_rtx_5090: bool) -> dict:
    import cupy as cp

    name = _device_name(cp)
    if require_rtx_5090 and "RTX 5090" not in name.upper():
        raise RuntimeError(
            f"--require-rtx-5090 requested, but CUDA device is {name!r}")
    with oracle_path.open("r", encoding="utf-8") as fh:
        oracle = json.load(fh)

    free0, total = cp.cuda.runtime.memGetInfo()
    started = time.perf_counter()
    scenarios = {
        key: _scenario(cp, int(key), reference)
        for key, reference in oracle["scenarios"].items()
    }
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - started
    free1, _ = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    failures = [
        f"scenario_{key}: {failure}"
        for key, result in scenarios.items()
        for failure in result["failures"]
    ]
    return {
        "ok": not failures,
        "device": name,
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "oracle": str(oracle_path.resolve()),
        "wrf_authority": oracle["provenance"]["authority"],
        "elapsed_seconds_including_jit": elapsed,
        "device_memory_total_bytes": int(total),
        "device_memory_delta_bytes": int(free0 - free1),
        "cupy_pool_used_bytes": int(pool.used_bytes()),
        "cupy_pool_held_bytes": int(pool.total_bytes()),
        "gates": GATES,
        "scenarios": scenarios,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument(
        "--require-rtx-5090", action="store_true",
        help="fail before field allocation unless the selected GPU is an RTX 5090",
    )
    args = parser.parse_args()
    result = run(args.oracle.resolve(), args.require_rtx_5090)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
