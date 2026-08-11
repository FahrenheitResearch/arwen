"""Single-state fused GS launcher for WRF NSSL option 18.

The public process launchers in :mod:`gpuwm.core.nssl2` are isolated oracle
slices.  Some of those slices intentionally overlap, so they are not a valid
production dispatcher.  This module is the only production entry point for
the main ``nssl_2mom_gs`` process: it accepts the durable concentration-space
driver workspace, diagnoses all rates from that state, applies the shared WRF
limiters, and writes one aggregate update.

Production selection is owned by the outer microphysics boundary; this module
remains a transport-free numerical stage and never dispatches itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from gpuwm.core.kernels import get_kernel
from gpuwm.core.nssl2_driver_support import NSSL2DriverWorkspace
from gpuwm.core.state import DTYPE

_FIELD_COUNT = 16
_THREADS = 128
NSSL2_FUSED_TEMPERATURE_SCRATCH = "nssl2_fused_temperature"
NSSL2_PRIMARY_ICE_TARGET_SCRATCH = "nssl2_primary_ice_target"


def _step32(dt_s: float) -> np.float32:
    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    converted = np.float32(step)
    if not np.isfinite(converted) or converted <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")
    return converted


def _validate_workspace(workspace: NSSL2DriverWorkspace) -> int:
    if not isinstance(workspace, NSSL2DriverWorkspace):
        raise TypeError("workspace must be NSSL2DriverWorkspace")
    expected = (_FIELD_COUNT, *workspace.shape)
    if workspace.state.shape != expected:
        raise ValueError(
            f"workspace state must have shape {expected}, got "
            f"{workspace.state.shape}"
        )
    if workspace.state.dtype != DTYPE:
        raise TypeError(
            f"workspace state must be float32, got {workspace.state.dtype}"
        )
    if not workspace.state.flags.c_contiguous:
        raise ValueError("workspace state must be C-contiguous")
    return int(np.prod(workspace.shape, dtype=np.int64))


def _validate_environment(
    shape: tuple[int, int, int], fields: dict[str, object]
) -> None:
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")


def _validate_vertical_velocity(
    shape: tuple[int, int, int], vertical_velocity
) -> None:
    expected = (shape[0] + 1, shape[1], shape[2])
    if vertical_velocity.shape != expected:
        raise ValueError(
            "vertical_velocity must be the WRF interface field with shape "
            f"{expected}, got {vertical_velocity.shape}"
        )
    if vertical_velocity.dtype != DTYPE:
        raise TypeError(
            "vertical_velocity must be float32, got "
            f"{vertical_velocity.dtype}"
        )
    if not vertical_velocity.flags.c_contiguous:
        raise ValueError("vertical_velocity must be C-contiguous")


def launch_fused_gs(
    workspace: NSSL2DriverWorkspace,
    full_theta,
    air_density,
    pressure_pa,
    exner,
    vertical_velocity,
    temperature_k,
    primary_ice_target_m3,
    dz,
    dt_s: float,
    *,
    hail_on: bool = True,
) -> None:
    """Advance the complete default option-18 GS slab exactly once.

    ``hail_on`` selects WRF's hail category switch.  ``True`` is the resolved
    option-18 default (``nssl_hail_on=1``, ``lhl>1``).  ``False`` is the
    ``nssl_hail_on=0`` variant, where ``module_mp_nssl_2mom.F:1445-1447`` sets
    ``lhl=0`` and the entire graupel<->hail conversion block at :19860 is
    skipped with its rates left at the zeros written just above it
    (:19847-19857).

    ``workspace`` is the object returned by
    :func:`gpuwm.core.nssl2_driver_support.gather_initialize_and_sediment`.
    Its mass moments are kg/kg, number moments are #/m3, and graupel/hail
    volume moments are m3/m3 of air.  Every environmental field is a
    contiguous cell-centred ``(nz, ny, nx)`` FP32 array except
    ``vertical_velocity``, which is GPUWM's interface field with shape
    ``(nz + 1, ny, nx)``. The fused kernel first centres interface W onto mass
    levels as WRF's microphysics driver does, then reproduces NSSL GS's
    current/next-mass-level average with its top-level clamp.

    ``temperature_k`` and ``primary_ice_target_m3`` are required runtime-owned
    scratch arrays. A deterministic prepass overwrites them from the immutable
    pre-GS state: ``temperature_k=full_theta*exner`` and the default
    ``icenucopt=1`` outer-driver ``t7`` diagnosis. This prepass is required so
    neighbouring t7 reads in GS cannot race cells already updated by the fused
    kernel. ``full_theta``, both scratch arrays, and ``workspace.state`` are
    updated in place; the other environmental fields are read-only.

    The CUDA implementation owns source-order rate diagnosis, the exact WRF
    shared donor/category limiters and vapor re-sums, the upstream frozen-vapor
    test-adjustment ceiling, one aggregate mass/number/volume/theta update,
    and the final option-18 moment bounds. The two post-update iterative GS
    saturation-adjustment blocks are skipped by the exact default
    ``ipconc=5``/``ibfc=1`` guards. It never invokes the overlapping isolated
    process launchers.
    """
    if not isinstance(hail_on, bool):
        raise TypeError("hail_on must be bool")
    size = _validate_workspace(workspace)
    fields = {
        "full_theta": full_theta,
        "air_density": air_density,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "temperature_k": temperature_k,
        "primary_ice_target_m3": primary_ice_target_m3,
        "dz": dz,
    }
    _validate_environment(workspace.shape, fields)
    _validate_vertical_velocity(workspace.shape, vertical_velocity)
    nz, ny, nx = workspace.shape
    step = _step32(dt_s)

    blocks = (size + _THREADS - 1) // _THREADS
    launch = ((blocks,), (_THREADS,))
    get_kernel("nssl2_fused_gs", "nssl2_prepare_fused_gs")(
        *launch,
        (
            temperature_k,
            primary_ice_target_m3,
            workspace.state,
            full_theta,
            air_density,
            pressure_pa,
            exner,
            np.int32(size),
        ),
    )
    get_kernel("nssl2_fused_gs", "nssl2_fused_gs")(
        *launch,
        (
            workspace.state,
            full_theta,
            air_density,
            pressure_pa,
            exner,
            temperature_k,
            vertical_velocity,
            primary_ice_target_m3,
            dz,
            step,
            np.int32(nz),
            np.int32(ny * nx),
            np.int32(size),
            np.int32(1 if hail_on else 0),
        ),
    )


@dataclass(frozen=True)
class NSSL2FusedGS:
    """Runtime-stage adapter with restart-rebuilt t0/t7 scratch."""

    temperature_k: object
    primary_ice_target_m3: object
    dt_s: float
    hail_on: bool = True

    def __call__(self, workspace: NSSL2DriverWorkspace, fields, /) -> None:
        launch_fused_gs(
            workspace,
            fields.theta,
            fields.rho,
            fields.pressure,
            fields.pii,
            fields.w,
            self.temperature_k,
            self.primary_ice_target_m3,
            fields.dz,
            self.dt_s,
            hail_on=self.hail_on,
        )


__all__ = [
    "NSSL2FusedGS",
    "NSSL2_FUSED_TEMPERATURE_SCRATCH",
    "NSSL2_PRIMARY_ICE_TARGET_SCRATCH",
    "launch_fused_gs",
]
