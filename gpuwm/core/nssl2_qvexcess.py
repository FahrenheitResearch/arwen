"""Pure WRF v4.6.1 NSSL QVEXCESS and its default caller update.

QVEXCESS itself advances only private two-iteration trial variables and
returns a nonnegative vapor excess. The two launch functions preserve that
contract: all physical inputs are read-only and the caller owns the output
array. A separately named adapter reproduces the default maximum-
supersaturation caller block that applies the returned increment.
"""

from __future__ import annotations

import math

import numpy as np

from gpuwm.core.kernels import get_kernel
from gpuwm.core.nssl2_driver_support import NSSL2DriverWorkspace
from gpuwm.core.state import DTYPE


_TPB = 256


def _validate_fields(
        fields: dict[str, object],
        ) -> tuple[tuple[int, int, int], int]:
    first = next(iter(fields.values()))
    shape = first.shape
    if len(shape) != 3 or any(extent < 1 for extent in shape):
        raise ValueError(
            f"NSSL QVEXCESS fields must be nonempty 3-D arrays, got {shape}")
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    return shape, int(np.prod(shape, dtype=np.int64))


def _target32(value: float) -> np.float32:
    try:
        target = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "target_supersaturation_percent must be a finite scalar") from exc
    if not math.isfinite(target):
        raise ValueError(
            "target_supersaturation_percent must be a finite scalar")
    converted = np.float32(target)
    if not np.isfinite(converted):
        raise ValueError(
            "target_supersaturation_percent must be representable as float32")
    if converted <= np.float32(-100.0):
        raise ValueError(
            "target_supersaturation_percent must be greater than -100")
    return converted


def _reject_alias(output, fields: dict[str, object], label: str) -> None:
    import cupy as cp

    if any(cp.may_share_memory(output, value) for value in fields.values()):
        raise ValueError(f"{label} must not alias any physical input")


def _validate_finite(fields: dict[str, object]) -> None:
    import cupy as cp

    for name, value in fields.items():
        if bool(cp.any(~cp.isfinite(value))):
            raise ValueError(f"{name} must contain only finite values")


def launch_qvexcess_split(
        theta_base, theta_perturbation, pressure_pa, exner,
        qv_base, qv_perturbation, qc,
        condensation_factor, latent_over_cp,
        target_supersaturation_percent: float, qvex_output, *,
        validate_values: bool = True) -> None:
    """Evaluate the exact source-signature QVEXCESS state split.

    Every array is contiguous FP32 (nz, ny, nx). theta_base and
    theta_perturbation correspond to WRF theta0 and thetap0; qv_base and
    qv_perturbation correspond to qv0 and qwvp0. Pressure is Pa, Exner is
    dimensionless, water fields are kg/kg, and both coefficient arrays are
    the values frozen by the surrounding NUCOND call.

    The routine writes only qvex_output. It never scatters its private trial
    evaporation/condensation state back to any input.
    """
    fields = {
        "theta_base": theta_base,
        "theta_perturbation": theta_perturbation,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "qv_base": qv_base,
        "qv_perturbation": qv_perturbation,
        "qc": qc,
        "condensation_factor": condensation_factor,
        "latent_over_cp": latent_over_cp,
        "qvex_output": qvex_output,
    }
    _, size = _validate_fields(fields)
    physical = {name: value for name, value in fields.items()
                if name != "qvex_output"}
    _reject_alias(qvex_output, physical, "qvex_output")
    target = _target32(target_supersaturation_percent)

    if validate_values:
        import cupy as cp

        _validate_finite(physical)
        if bool(cp.any(pressure_pa <= DTYPE(0.0))):
            raise ValueError("pressure_pa must be strictly positive")
        if bool(cp.any(exner <= DTYPE(0.0))):
            raise ValueError("exner must be strictly positive")
        if bool(cp.any(condensation_factor < DTYPE(0.0))):
            raise ValueError("condensation_factor must be nonnegative")
        if bool(cp.any(latent_over_cp <= DTYPE(0.0))):
            raise ValueError("latent_over_cp must be strictly positive")
        if bool(cp.any(theta_base + theta_perturbation <= DTYPE(0.0))):
            raise ValueError("combined potential temperature must be positive")
        if bool(cp.any(qv_base + qv_perturbation < DTYPE(0.0))):
            raise ValueError("combined vapor must be nonnegative")
        if bool(cp.any(qc < DTYPE(0.0))):
            raise ValueError("qc must be nonnegative")

    blocks = (size + _TPB - 1) // _TPB
    get_kernel("nssl2_qvexcess", "nssl2_qvexcess_split")(
        (blocks,), (_TPB,),
        (theta_base, theta_perturbation, pressure_pa, exner,
         qv_base, qv_perturbation, qc,
         condensation_factor, latent_over_cp, target,
         qvex_output, np.int32(size)))


def launch_qvexcess_workspace(
        workspace: NSSL2DriverWorkspace,
        full_theta, pressure_pa, exner,
        condensation_factor, latent_over_cp,
        target_supersaturation_percent: float, qvex_output, *,
        validate_values: bool = True) -> None:
    """Evaluate pure-return QVEXCESS on the durable NSSL workspace.

    The workspace qv and qc views are passed directly to CUDA. Mass remains
    kg/kg and the workspace concentration moments are neither read nor
    converted. qvex_output is caller-owned FP32 scratch/output.
    """
    if not isinstance(workspace, NSSL2DriverWorkspace):
        raise TypeError("workspace must be NSSL2DriverWorkspace")
    qv = workspace.field("qv")
    qc = workspace.field("qc")
    fields = {
        "full_theta": full_theta,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "qv": qv,
        "qc": qc,
        "condensation_factor": condensation_factor,
        "latent_over_cp": latent_over_cp,
        "qvex_output": qvex_output,
    }
    shape, size = _validate_fields(fields)
    if shape != workspace.shape:
        raise ValueError(
            f"workspace has shape {workspace.shape}, inputs have shape {shape}")
    physical = {name: value for name, value in fields.items()
                if name != "qvex_output"}
    _reject_alias(qvex_output, physical, "qvex_output")
    target = _target32(target_supersaturation_percent)

    if validate_values:
        import cupy as cp

        _validate_finite(physical)
        for name, value in {
                "full_theta": full_theta,
                "pressure_pa": pressure_pa,
                "exner": exner,
                "latent_over_cp": latent_over_cp,
        }.items():
            if bool(cp.any(value <= DTYPE(0.0))):
                raise ValueError(f"{name} must be strictly positive")
        for name, value in {
                "qv": qv,
                "qc": qc,
                "condensation_factor": condensation_factor,
        }.items():
            if bool(cp.any(value < DTYPE(0.0))):
                raise ValueError(f"{name} must be nonnegative")

    blocks = (size + _TPB - 1) // _TPB
    get_kernel("nssl2_qvexcess", "nssl2_qvexcess_workspace")(
        (blocks,), (_TPB,),
        (full_theta, pressure_pa, exner, qv, qc,
         condensation_factor, latent_over_cp, target,
         qvex_output, np.int32(size)))


def apply_qvexcess_maxsup_to_workspace(
        workspace: NSSL2DriverWorkspace,
        full_theta, air_density, exner,
        background_ccn, cloud_mean_mass, latent_over_cp,
        qvex, new_cloud_number_output, *,
        couple_number: bool = True,
        validate_values: bool = True) -> None:
    """Apply WRF's default post-QVEXCESS maximum-SS caller block.

    This is deliberately separate from the pure-return routine. It applies
    lines 11544--11567 to the durable workspace: latent theta adjustment,
    vapor-to-cloud transfer, then optional active-default imaxsupopt=4
    cloud-number and predicted-CCN coupling.

    qndrop, qnn, and background_ccn are native concentrations (#/m3).
    cloud_mean_mass is the caller's current WRF xmas(lc) in kg/particle.
    No Registry-unit path exists in this adapter.
    """
    if not isinstance(workspace, NSSL2DriverWorkspace):
        raise TypeError("workspace must be NSSL2DriverWorkspace")
    if not isinstance(couple_number, bool):
        raise TypeError("couple_number must be bool")
    qv = workspace.field("qv")
    qc = workspace.field("qc")
    qndrop = workspace.field("qndrop")
    qnn = workspace.field("qnn")
    fields = {
        "full_theta": full_theta,
        "air_density": air_density,
        "exner": exner,
        "qv": qv,
        "qc": qc,
        "qndrop": qndrop,
        "qnn": qnn,
        "background_ccn": background_ccn,
        "cloud_mean_mass": cloud_mean_mass,
        "latent_over_cp": latent_over_cp,
        "qvex": qvex,
        "new_cloud_number_output": new_cloud_number_output,
    }
    shape, size = _validate_fields(fields)
    if shape != workspace.shape:
        raise ValueError(
            f"workspace has shape {workspace.shape}, inputs have shape {shape}")
    physical = {name: value for name, value in fields.items()
                if name != "new_cloud_number_output"}
    _reject_alias(
        new_cloud_number_output, physical, "new_cloud_number_output")

    if validate_values:
        import cupy as cp

        _validate_finite(physical)
        for name, value in {
                "full_theta": full_theta,
                "air_density": air_density,
                "exner": exner,
                "cloud_mean_mass": cloud_mean_mass,
                "latent_over_cp": latent_over_cp,
        }.items():
            if bool(cp.any(value <= DTYPE(0.0))):
                raise ValueError(f"{name} must be strictly positive")
        for name, value in {
                "qv": qv,
                "qc": qc,
                "qndrop": qndrop,
                "qnn": qnn,
                "background_ccn": background_ccn,
                "qvex": qvex,
        }.items():
            if bool(cp.any(value < DTYPE(0.0))):
                raise ValueError(f"{name} must be nonnegative")
        if bool(cp.any(qvex > qv)):
            raise ValueError("qvex must not exceed available vapor")

    blocks = (size + _TPB - 1) // _TPB
    get_kernel(
        "nssl2_qvexcess", "nssl2_qvexcess_apply_maxsup_default",
    )((blocks,), (_TPB,),
      (full_theta, air_density, exner, qv, qc, qndrop, qnn,
       background_ccn, cloud_mean_mass, latent_over_cp, qvex,
       new_cloud_number_output, np.int32(couple_number), np.int32(size)))


__all__ = [
    "apply_qvexcess_maxsup_to_workspace",
    "launch_qvexcess_split",
    "launch_qvexcess_workspace",
]
