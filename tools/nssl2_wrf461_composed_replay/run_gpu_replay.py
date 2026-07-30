"""Run GPUWM's complete NSSL production coordinator on one real column."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cupy as cp
import numpy as np

from gpuwm.core.nssl2_driver_support import NSSL2DriverWorkspace
from gpuwm.core.nssl2_fused_gs import launch_fused_gs
from gpuwm.core.nssl2_nucond import launch_nucond
from gpuwm.core.nssl2_production_coordinator import (
    NSSL2PrecipitationFields,
    NSSL2ProductionHooks,
    NSSL2RegistryFields,
    run_nssl2_production_step,
)


REGISTRY_NAMES = (
    "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop", "qnr",
    "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh",
)
PRECIPITATION_NAMES = (
    "rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
    "graupelncv", "hailnc", "hailncv", "sr",
)
CSV_FIELDS = (
    "engine", "k", "theta", *REGISTRY_NAMES, "refl_10cm", "effc_m",
    "effi_m", "effs_m", *PRECIPITATION_NAMES,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell(value: np.ndarray, nz: int, name: str):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (nz,):
        raise ValueError(f"{name} must have shape {(nz,)}, got {array.shape}")
    return cp.asarray(np.ascontiguousarray(array.reshape(nz, 1, 1)))


def run(input_path: Path, output_csv: Path, receipt_path: Path) -> None:
    with np.load(input_path, allow_pickle=False) as source:
        nz = int(np.asarray(source["theta"]).size)
        shape = (nz, 1, 1)
        dt_s = float(np.asarray(source["dt_s"]))
        registry_values = {
            name: _cell(source[name], nz, name) for name in REGISTRY_NAMES
        }
        theta = _cell(source["theta"], nz, "theta")
        rho = _cell(source["rho"], nz, "rho")
        pressure = _cell(source["pressure"], nz, "pressure")
        exner = _cell(source["exner"], nz, "exner")
        dz = _cell(source["dz"], nz, "dz")
        w_host = np.asarray(source["w_interface"], dtype=np.float32)
        if w_host.shape != (nz + 1,):
            raise ValueError(
                f"w_interface must have shape {(nz + 1,)}, got {w_host.shape}")
        w = cp.asarray(np.ascontiguousarray(w_host.reshape(nz + 1, 1, 1)))

    surface_shape = (1, 1)
    precipitation_values = {
        name: cp.zeros(surface_shape, dtype=cp.float32)
        for name in PRECIPITATION_NAMES
    }
    workspace = NSSL2DriverWorkspace(
        cp.empty((16, *shape), dtype=cp.float32),
        cp.zeros((5, *surface_shape), dtype=cp.float32),
        shape,
        ignored_accumulator=cp.zeros(surface_shape, dtype=cp.float32),
    )
    temperature = cp.empty(shape, dtype=cp.float32)
    cp.multiply(theta, exner, out=temperature)
    primary_ice_target = cp.empty(shape, dtype=cp.float32)
    supersaturation = cp.empty(shape, dtype=cp.float32)
    reflectivity = cp.empty(shape, dtype=cp.float32)
    effc = cp.empty(shape, dtype=cp.float32)
    effi = cp.empty(shape, dtype=cp.float32)
    effs = cp.empty(shape, dtype=cp.float32)
    receipts: dict[str, np.ndarray] = {}

    def capture(name: str, active_workspace: NSSL2DriverWorkspace) -> None:
        cp.cuda.Stream.null.synchronize()
        receipts[name] = cp.asnumpy(active_workspace.state)
        receipts[f"{name}_theta"] = cp.asnumpy(theta)

    def fused(active_workspace: NSSL2DriverWorkspace) -> None:
        capture("post_sediment_concentration", active_workspace)
        launch_fused_gs(
            active_workspace,
            theta,
            rho,
            pressure,
            exner,
            w,
            temperature,
            primary_ice_target,
            dz,
            dt_s,
        )
        capture("post_fused_concentration", active_workspace)

    def nucond_qvexcess(active_workspace: NSSL2DriverWorkspace) -> None:
        launch_nucond(
            theta,
            rho,
            pressure,
            exner,
            w,
            active_workspace.field("qv"),
            active_workspace.field("qc"),
            active_workspace.field("qr"),
            active_workspace.field("qi"),
            active_workspace.field("qs"),
            active_workspace.field("qndrop"),
            active_workspace.field("qnr"),
            active_workspace.field("qni"),
            active_workspace.field("qns"),
            active_workspace.field("qnn"),
            dt_s,
            supersaturation_scratch=supersaturation,
            concentration_space=True,
            validate_values=True,
        )
        cp.multiply(theta, exner, out=temperature)
        capture("post_nucond_concentration", active_workspace)

    def finish(active_workspace: NSSL2DriverWorkspace) -> None:
        capture("pre_finish_concentration", active_workspace)

    run_nssl2_production_step(
        rho,
        dz,
        NSSL2RegistryFields(**registry_values),
        NSSL2PrecipitationFields(**precipitation_values),
        NSSL2ProductionHooks(
            fused_gs=fused,
            nucond_qvexcess=nucond_qvexcess,
            moist_physics_finish=finish,
        ),
        dt_s,
        first_step=False,
        cu_used=False,
        output_due=True,
        temperature_k=temperature,
        refl_10cm=reflectivity,
        radiation_due=True,
        re_cloud_m=effc,
        re_ice_m=effi,
        re_snow_m=effs,
        validate_values=True,
        workspace=workspace,
    )
    cp.cuda.Stream.null.synchronize()

    registry_host = {
        name: cp.asnumpy(value).reshape(nz)
        for name, value in registry_values.items()
    }
    theta_host = cp.asnumpy(theta).reshape(nz)
    reflectivity_host = cp.asnumpy(reflectivity).reshape(nz)
    radius_host = {
        "effc_m": cp.asnumpy(effc).reshape(nz),
        "effi_m": cp.asnumpy(effi).reshape(nz),
        "effs_m": cp.asnumpy(effs).reshape(nz),
    }
    precipitation_host = {
        name: float(cp.asnumpy(value)[0, 0])
        for name, value in precipitation_values.items()
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for k in range(nz):
            row = {
                "engine": "gpu",
                "k": k + 1,
                "theta": float(theta_host[k]),
                **{name: float(value[k]) for name, value in registry_host.items()},
                "refl_10cm": float(reflectivity_host[k]),
                **{name: float(value[k]) for name, value in radius_host.items()},
                **precipitation_host,
            }
            writer.writerow(row)

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        receipt_path,
        **receipts,
        registry_after=np.stack(
            [registry_host[name] for name in REGISTRY_NAMES], axis=0),
        theta_after=theta_host,
        temperature_after=cp.asnumpy(temperature).reshape(nz),
        reflectivity=reflectivity_host,
        effc_m=radius_host["effc_m"],
        effi_m=radius_host["effi_m"],
        effs_m=radius_host["effs_m"],
        precipitation=np.asarray(
            [precipitation_host[name] for name in PRECIPITATION_NAMES],
            dtype=np.float32,
        ),
    )
    evidence = {
        "schema": "gpuwm.nssl2.composed-gpu-replay/v1",
        "input": str(input_path.resolve()),
        "input_sha256": _sha256(input_path),
        "output_csv": str(output_csv.resolve()),
        "output_sha256": _sha256(output_csv),
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": _sha256(receipt_path),
        "nz": nz,
        "dt_s": dt_s,
        "device": int(cp.cuda.runtime.getDevice()),
    }
    print(json.dumps(evidence, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("receipt_npz", type=Path)
    args = parser.parse_args()
    run(args.input, args.output_csv, args.receipt_npz)


if __name__ == "__main__":
    main()
