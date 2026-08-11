"""Byte-identity digest probe for the default NSSL mp18 production lane.

Runs one deterministic ``run_nssl2_production_step`` on seeded synthetic
fields (the same raw-array seam the DomainState runtime adapter delegates to)
and prints a SHA-256 digest per output array.  Capture the digests before and
after any NSSL change: identical digests prove the shipped mp18 default lane
is byte-identical on this box (same GPU + NVRTC platform; see
gpuwm/certify/compile_platform.py for the compile-identity caveat).

Usage:
    python tools/nssl2_mp18_digest_probe.py [--json OUT.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np


def _seeded_fields(nz: int, ny: int, nx: int):
    """Deterministic, physically plausible moist-column fields (FP32)."""
    rng = np.random.default_rng(1974)
    shape = (nz, ny, nx)
    column = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None, None]

    pressure = np.broadcast_to(
        100000.0 - 85000.0 * column, shape).astype(np.float32)
    pressure = pressure + rng.uniform(-50.0, 50.0, shape).astype(np.float32)
    exner = ((pressure / 100000.0) ** (287.04 / 1004.0)).astype(np.float32)
    theta = np.broadcast_to(300.0 + 45.0 * column, shape).astype(np.float32)
    theta = theta + rng.uniform(-0.5, 0.5, shape).astype(np.float32)
    temperature = theta * exner
    rho = (pressure / (287.04 * temperature)).astype(np.float32)

    qvs_proxy = (0.018 * np.exp(-4.0 * column)).astype(np.float32)
    fields = {
        "qv": (qvs_proxy * rng.uniform(0.6, 1.05, shape)).astype(np.float32),
        "qc": np.where(
            rng.uniform(size=shape) < 0.4,
            rng.uniform(0.0, 2.0e-3, shape), 0.0).astype(np.float32),
        "qr": np.where(
            rng.uniform(size=shape) < 0.3,
            rng.uniform(0.0, 4.0e-3, shape), 0.0).astype(np.float32),
        "qi": np.where(
            (column > 0.4) & (rng.uniform(size=shape) < 0.4),
            rng.uniform(0.0, 1.0e-3, shape), 0.0).astype(np.float32),
        "qs": np.where(
            (column > 0.35) & (rng.uniform(size=shape) < 0.4),
            rng.uniform(0.0, 2.5e-3, shape), 0.0).astype(np.float32),
        "qg": np.where(
            (column > 0.25) & (rng.uniform(size=shape) < 0.35),
            rng.uniform(0.0, 3.0e-3, shape), 0.0).astype(np.float32),
        "qh": np.where(
            (column > 0.25) & (rng.uniform(size=shape) < 0.2),
            rng.uniform(0.0, 2.0e-3, shape), 0.0).astype(np.float32),
    }
    # Numbers roughly consistent with the masses (Registry #/kg).
    fields["qndrop"] = (fields["qc"] * 1.0e11).astype(np.float32)
    fields["qnr"] = (fields["qr"] * 1.0e8).astype(np.float32)
    fields["qni"] = (fields["qi"] * 1.0e9).astype(np.float32)
    fields["qns"] = (fields["qs"] * 5.0e7).astype(np.float32)
    fields["qng"] = (fields["qg"] * 2.0e7).astype(np.float32)
    fields["qnh"] = (fields["qh"] * 5.0e6).astype(np.float32)
    fields["qnn"] = np.maximum(
        0.0, 408163264.0 - fields["qndrop"]).astype(np.float32)
    fields["qvolg"] = (fields["qg"] / 500.0).astype(np.float32)
    fields["qvolh"] = (fields["qh"] / 800.0).astype(np.float32)

    w = rng.uniform(-2.0, 8.0, (nz + 1, ny, nx)).astype(np.float32)
    dz = np.full(shape, 250.0, dtype=np.float32)
    environment = {
        "theta": theta,
        "rho": rho,
        "pressure": pressure,
        "exner": exner,
        "temperature": temperature.astype(np.float32),
        "w": w,
        "dz": dz,
    }
    return fields, environment


def run_probe(steps: int = 3, nz: int = 40, ny: int = 4, nx: int = 4):
    import cupy as cp

    import gpuwm
    from gpuwm.core.nssl2_driver_support import NSSL2DriverWorkspace
    from gpuwm.core.nssl2_fused_gs import launch_fused_gs
    from gpuwm.core.nssl2_nucond import launch_nucond
    from gpuwm.core.nssl2_production_coordinator import (
        NSSL2PrecipitationFields,
        NSSL2ProductionHooks,
        NSSL2RegistryFields,
        run_nssl2_production_step,
    )

    print(f"gpuwm.__file__ = {gpuwm.__file__}")

    fields, env = _seeded_fields(nz, ny, nx)
    device = {name: cp.asarray(value) for name, value in fields.items()}
    theta = cp.asarray(env["theta"])
    rho = cp.asarray(env["rho"])
    pressure = cp.asarray(env["pressure"])
    exner = cp.asarray(env["exner"])
    w = cp.asarray(env["w"])
    dz = cp.asarray(env["dz"])
    dt = 4.0

    shape = (nz, ny, nx)
    registry = NSSL2RegistryFields(**device)
    precipitation = NSSL2PrecipitationFields(**{
        name: cp.zeros((ny, nx), dtype=cp.float32)
        for name in ("rainnc", "rainncv", "snownc", "snowncv",
                     "graupelnc", "graupelncv", "hailnc", "hailncv", "sr")
    })

    temperature_k = cp.asarray(env["temperature"])
    ss_scratch = cp.empty(shape, dtype=cp.float32)
    fused_temperature = cp.empty(shape, dtype=cp.float32)
    primary_ice = cp.empty(shape, dtype=cp.float32)

    def fused_gs(workspace: NSSL2DriverWorkspace) -> None:
        launch_fused_gs(
            workspace, theta, rho, pressure, exner, w,
            fused_temperature, primary_ice, dz, dt)

    def nucond_qvexcess(workspace: NSSL2DriverWorkspace) -> None:
        launch_nucond(
            theta, rho, pressure, exner, w,
            workspace.field("qv"), workspace.field("qc"),
            workspace.field("qr"), workspace.field("qi"),
            workspace.field("qs"), workspace.field("qndrop"),
            workspace.field("qnr"), workspace.field("qni"),
            workspace.field("qns"), workspace.field("qnn"),
            dt, supersaturation_scratch=ss_scratch,
            concentration_space=True, validate_values=False)

    def finish(_workspace) -> None:
        pass

    hooks = NSSL2ProductionHooks(
        fused_gs=fused_gs,
        nucond_qvexcess=nucond_qvexcess,
        moist_physics_finish=finish,
    )

    refl = cp.zeros(shape, dtype=cp.float32)
    re_cloud = cp.zeros(shape, dtype=cp.float32)
    re_ice = cp.zeros(shape, dtype=cp.float32)
    re_snow = cp.zeros(shape, dtype=cp.float32)
    for step_index in range(steps):
        cp.multiply(theta, exner, out=temperature_k)
        run_nssl2_production_step(
            rho, dz, registry, precipitation, hooks, dt,
            first_step=(step_index == 0),
            output_due=True,
            temperature_k=temperature_k,
            refl_10cm=refl,
            radiation_due=True,
            re_cloud_m=re_cloud, re_ice_m=re_ice, re_snow_m=re_snow,
            validate_values=False,
        )

    digests = {}
    outputs = dict(device)
    outputs["theta"] = theta
    outputs["refl_10cm"] = refl
    outputs["re_cloud"] = re_cloud
    outputs["re_ice"] = re_ice
    outputs["re_snow"] = re_snow
    for name in ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
                 "graupelncv", "hailnc", "hailncv", "sr"):
        outputs[name] = getattr(precipitation, name)
    for name in sorted(outputs):
        host = cp.asnumpy(outputs[name])
        if not np.all(np.isfinite(host)):
            raise RuntimeError(f"non-finite values in {name}")
        digests[name] = hashlib.sha256(host.tobytes()).hexdigest()
    return digests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    digests = run_probe(steps=args.steps)
    for name, digest in digests.items():
        print(f"{digest}  {name}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(digests, handle, indent=1, sort_keys=True)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
