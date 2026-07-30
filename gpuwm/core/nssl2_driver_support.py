"""Pre-GS WRF v4.6.1 NSSL option-18 driver support.

This module owns the low-level state transport path that surrounds NSSL's main
gather/scatter process routine.  It deliberately does not make ``mp_physics=18``
selectable and does not call GS, NUCOND, radar, effective-radius, finish, restart,
or preflight code.

The 16 prognostics are gathered once.  Number and volume moments remain in the
scheme's concentration convention throughout initialization, KF number-moment
diagnosis, and all five sedimentation categories, then are scattered once.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from gpuwm.core.kernels import get_kernel
from gpuwm.core.nssl2_contract import DEFAULT_RESTART_FIELDS
from gpuwm.core.state import DTYPE

_FIELD_COUNT = 16
_SHALLOW_KMAX = 64
_KMAX = 256
_ELEMENT_TPB = 256
_COLUMN_TPB = 64

_QV = 0
_QC = 1
_QR = 2
_QI = 3
_QS = 4
_QG = 5
_QH = 6
_NC = 7
_NR = 8
_NI = 9
_NS = 10
_NG = 11
_NH = 12
_NN = 13
_VG = 14
_VH = 15

NSSL2_DRIVER_FIELDS = DEFAULT_RESTART_FIELDS
NSSL2_SEDIMENT_EXPORTS = ("rain", "ice", "snow", "graupel", "hail")
NSSL2_DRIVER_STATE_SCRATCH = "nssl2_driver_state"
NSSL2_DRIVER_SURFACE_EXPORT_SCRATCH = "nssl2_driver_surface_export"
NSSL2_DRIVER_IGNORED_ACCUMULATOR_SCRATCH = \
    "nssl2_driver_ignored_accumulator"
_FIELD_INDEX = {name: index for index, name in enumerate(NSSL2_DRIVER_FIELDS)}


@dataclass(frozen=True)
class NSSL2DriverWorkspace:
    """Durable pre-GS NSSL state in native internal concentration units.

    ``state`` has shape ``(16, nz, ny, nx)``. Mass fields remain kg/kg,
    number fields are #/m3, predicted CCN is #/m3, and the two volume fields
    are m3 hydrometeor/m3 air. It is the production seam for later GS, NUCOND,
    QVEXCESS, and diagnostic phases; none of those phases should re-gather.

    ``category_surface_export`` has shape ``(5, ny, nx)`` in the fixed order
    rain/ice/snow/graupel/hail and units kg/m2 over the step. The standard WRF
    four-category precipitation reducer intentionally omits cloud-ice export.
    ``ignored_accumulator`` preserves the likewise-unreduced cloud-droplet
    surface export after sedimentation, allowing an exact water-budget receipt
    without adding it to WRF's precipitation accumulators.
    """

    state: object
    category_surface_export: object
    shape: tuple[int, int, int]
    ignored_accumulator: object | None = None

    def field(self, name: str):
        """Return one mutable internal-unit field view by Registry name."""
        try:
            index = _FIELD_INDEX[name]
        except KeyError as exc:
            raise KeyError(f"unknown NSSL driver field {name!r}") from exc
        return self.state[index]

    @property
    def ice_surface_export(self):
        """Cloud-ice surface export in kg/m2 over this step."""
        return self.category_surface_export[1]

    @property
    def cloud_surface_export(self):
        """Cloud-droplet surface export in kg/m2 over this step."""
        return self.ignored_accumulator


def validate_nssl2_driver_workspace(
        workspace: NSSL2DriverWorkspace,
        shape: tuple[int, int, int], /) -> None:
    """Validate every persistent driver buffer without copying or reducing."""
    if not isinstance(workspace, NSSL2DriverWorkspace):
        raise TypeError("workspace must be NSSL2DriverWorkspace")
    shape = tuple(shape)
    if tuple(workspace.shape) != shape:
        raise ValueError(
            f"workspace has shape {workspace.shape}, expected {shape}")
    _, ny, nx = shape
    buffers = {
        "workspace.state": (workspace.state, (_FIELD_COUNT, *shape)),
        "workspace.category_surface_export": (
            workspace.category_surface_export, (5, ny, nx)),
        "workspace.ignored_accumulator": (
            workspace.ignored_accumulator, (ny, nx)),
    }
    for name, (value, expected) in buffers.items():
        if value is None:
            raise ValueError(f"{name} is required for a reusable workspace")
        if tuple(value.shape) != expected:
            raise ValueError(
                f"{name} must have shape {expected}, got "
                f"{tuple(value.shape)}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")


def _validate_volume_fields(
        fields: dict[str, object],
        ) -> tuple[tuple[int, int, int], int]:
    first = next(iter(fields.values()))
    shape = first.shape
    if len(shape) != 3:
        raise ValueError(f"NSSL driver fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    if nz < 2 or nz > _KMAX:
        raise ValueError(
            f"NSSL driver support requires 2 <= nz <= {_KMAX}, got {nz}")
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    return (nz, ny, nx), int(np.prod(shape, dtype=np.int64))


def _validate_surface_fields(
        fields: dict[str, object], shape: tuple[int, int],
        ) -> None:
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")


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


def gather_initialize_and_sediment(
        air_density, dz,
        qv, qc, qr, qi, qs, qg, qh,
        qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
        dt_s: float, *, temperature_k,
        first_step: bool = False, cu_used: bool = False,
        qrcuten=None, qscuten=None, qicuten=None, qccuten=None,
        workspace: NSSL2DriverWorkspace | None = None,
        ) -> NSSL2DriverWorkspace:
    """Gather once, initialize moments, diagnose KF numbers, and sediment.

    Volume fields use contiguous ``(nz, ny, nx)`` FP32 arrays.  Mass fields are
    kg/kg; number fields are #/kg; graupel/hail volume fields are m3/kg of dry
    air; density is kg/m3; ``temperature_k`` is absolute temperature; and
    ``dz`` is metres. Inputs are never mutated. The returned workspace keeps
    all moments in internal concentration units so GS and subsequent phases can
    run before the single final scatter.

    ``first_step`` selects WRF's exact ``itimestep == 1`` ``calcnfromq`` path.
    When ``cu_used`` is true, each supplied KF ``q*cuten`` rate is converted to
    a step mass increment and passed through exact ``calcnfromcuten`` number
    diagnosis.  Missing KF arrays are zero rates.  As in WRF, these rates do not
    add mass here: dynamics has already applied their mass tendencies.
    """
    if not isinstance(first_step, bool):
        raise TypeError("first_step must be bool")
    if not isinstance(cu_used, bool):
        raise TypeError("cu_used must be bool")

    volume_fields = {
        "air_density": air_density,
        "temperature_k": temperature_k,
        "dz": dz,
        "qv": qv,
        "qc": qc,
        "qr": qr,
        "qi": qi,
        "qs": qs,
        "qg": qg,
        "qh": qh,
        "qndrop": qndrop,
        "qnr": qnr,
        "qni": qni,
        "qns": qns,
        "qng": qng,
        "qnh": qnh,
        "qnn": qnn,
        "qvolg": qvolg,
        "qvolh": qvolh,
    }
    rates = {
        "qrcuten": qrcuten,
        "qscuten": qscuten,
        "qicuten": qicuten,
        "qccuten": qccuten,
    }
    volume_fields.update({
        name: value for name, value in rates.items() if value is not None
    })
    (nz, ny, nx), size = _validate_volume_fields(volume_fields)
    step = _step32(dt_s)

    if workspace is not None:
        validate_nssl2_driver_workspace(workspace, (nz, ny, nx))
        if cu_used and any(value is None for value in rates.values()):
            raise ValueError(
                "a reusable NSSL workspace with cu_used=True requires all "
                "four KF rate arrays")

    # CuPy is imported lazily so CPU-only contract and lint tests remain usable.
    import cupy as cp

    state = (cp.empty((_FIELD_COUNT, nz, ny, nx), dtype=DTYPE)
             if workspace is None else workspace.state)
    zero_rate = None
    rate_args = []
    for value in rates.values():
        if value is None:
            if not cu_used:
                # The CUDA gather branch does not dereference KF pointers
                # when CU is disabled. Reuse a valid volume pointer instead
                # of allocating a full-volume zero field.
                rate_args.append(qv)
            else:
                if zero_rate is None:
                    zero_rate = cp.zeros_like(qv)
                rate_args.append(zero_rate)
        else:
            rate_args.append(value)

    element_blocks = (size + _ELEMENT_TPB - 1) // _ELEMENT_TPB
    get_kernel("nssl2_driver_support", "nssl2_driver_gather_initialize")(
        (element_blocks,), (_ELEMENT_TPB,),
        (air_density, qv, qc, qr, qi, qs, qg, qh,
         qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
         *rate_args, state, step, np.int32(first_step), np.int32(cu_used),
         np.int32(size)))

    ncol = ny * nx
    column_blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    if workspace is None:
        category_export = cp.empty((5, ny, nx), dtype=DTYPE)
        ignored_accumulator = cp.zeros((ny, nx), dtype=DTYPE)
    else:
        category_export = workspace.category_surface_export
        ignored_accumulator = workspace.ignored_accumulator
        # The five precipitating/ice kernels read-modify-write this temporary
        # WRF accumulator. Cloud fallout overwrites it with its own separate
        # export receipt after those launches.
        ignored_accumulator[...] = DTYPE(0.0)
    suffix = "64" if nz <= _SHALLOW_KMAX else "256"

    sediment_calls = (
        (f"nssl2_rain_sediment_{suffix}", _QR, _NR, None, 0),
        (f"nssl2_ice_sediment_{suffix}", _QI, _NI, None, 1),
        (f"nssl2_snow_sediment_{suffix}", _QS, _NS, None, 2),
        (f"nssl2_graupel_sediment_{suffix}", _QG, _NG, _VG, 3),
        (f"nssl2_hail_sediment_{suffix}", _QH, _NH, _VH, 4),
    )
    for kernel_name, mass_index, number_index, volume_index, export_index in (
            sediment_calls):
        arguments = [air_density, state[mass_index], state[number_index]]
        if volume_index is not None:
            arguments.append(state[volume_index])
        arguments.extend((
            dz, ignored_accumulator, category_export[export_index], step,
            np.int32(nz), np.int32(ny), np.int32(nx),
        ))
        get_kernel("nssl2_driver_support", kernel_name)(
            (column_blocks,), (_COLUMN_TPB,), tuple(arguments))

    # Cloud is disjoint from the other category states, so executing it after
    # the five standard exports is numerically identical to WRF's cloud-first
    # loop. Its bottom export is diagnosed but intentionally not reduced into
    # RAINNC, matching the official driver.
    get_kernel(
        "nssl2_driver_support", f"nssl2_cloud_sediment_{suffix}",
    )((column_blocks,), (_COLUMN_TPB,), (
        air_density, temperature_k, state[_QC], state[_NC], dz,
        ignored_accumulator, step,
        np.int32(nz), np.int32(ny), np.int32(nx),
    ))

    return NSSL2DriverWorkspace(
        state=state, category_surface_export=category_export,
        shape=(nz, ny, nx), ignored_accumulator=ignored_accumulator)


def reduce_nssl2_precipitation(
        workspace: NSSL2DriverWorkspace,
        rainnc, rainncv, snownc, snowncv,
        graupelnc, graupelncv, hailnc, hailncv, sr,
        ) -> None:
    """Apply WRF's four-category surface precipitation reducer and SR."""
    if not isinstance(workspace, NSSL2DriverWorkspace):
        raise TypeError("workspace must be NSSL2DriverWorkspace")
    _, ny, nx = workspace.shape
    _validate_surface_fields({
        "rainnc": rainnc,
        "rainncv": rainncv,
        "snownc": snownc,
        "snowncv": snowncv,
        "graupelnc": graupelnc,
        "graupelncv": graupelncv,
        "hailnc": hailnc,
        "hailncv": hailncv,
        "sr": sr,
    }, (ny, nx))
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    get_kernel(
        "nssl2_driver_support", "nssl2_driver_reduce_precipitation",
    )((blocks,), (_COLUMN_TPB,),
      (workspace.category_surface_export,
       rainnc, rainncv, snownc, snowncv,
       graupelnc, graupelncv, hailnc, hailncv, sr, np.int32(ncol)))


def scatter_nssl2_driver_workspace(
        workspace: NSSL2DriverWorkspace, air_density,
        qv, qc, qr, qi, qs, qg, qh,
        qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
        ) -> None:
    """Perform the one final concentration-to-Registry scatter in place."""
    if not isinstance(workspace, NSSL2DriverWorkspace):
        raise TypeError("workspace must be NSSL2DriverWorkspace")
    fields = {
        "air_density": air_density,
        "qv": qv,
        "qc": qc,
        "qr": qr,
        "qi": qi,
        "qs": qs,
        "qg": qg,
        "qh": qh,
        "qndrop": qndrop,
        "qnr": qnr,
        "qni": qni,
        "qns": qns,
        "qng": qng,
        "qnh": qnh,
        "qnn": qnn,
        "qvolg": qvolg,
        "qvolh": qvolh,
    }
    shape, size = _validate_volume_fields(fields)
    if shape != workspace.shape:
        raise ValueError(
            f"workspace has shape {workspace.shape}, output fields have "
            f"shape {shape}")
    blocks = (size + _ELEMENT_TPB - 1) // _ELEMENT_TPB
    get_kernel("nssl2_driver_support", "nssl2_driver_scatter")(
        (blocks,), (_ELEMENT_TPB,),
        (air_density, workspace.state, qv, qc, qr, qi, qs, qg, qh,
         qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
         np.int32(size)))


def launch_nssl2_driver_support(
        air_density, dz,
        qv, qc, qr, qi, qs, qg, qh,
        qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
        rainnc, rainncv, snownc, snowncv,
        graupelnc, graupelncv, hailnc, hailncv, sr,
        dt_s: float, *, temperature_k,
        first_step: bool = False, cu_used: bool = False,
        qrcuten=None, qscuten=None, qicuten=None, qccuten=None,
        ) -> NSSL2DriverWorkspace:
    """Round-trip convenience wrapper for isolated official-oracle gates.

    Production integration must use the three phase functions directly so GS,
    NUCOND/QVEXCESS, and diagnostics operate on the durable concentration-space
    workspace before :func:`scatter_nssl2_driver_workspace` is called once.
    """
    workspace = gather_initialize_and_sediment(
        air_density, dz,
        qv, qc, qr, qi, qs, qg, qh,
        qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
        dt_s, temperature_k=temperature_k,
        first_step=first_step, cu_used=cu_used,
        qrcuten=qrcuten, qscuten=qscuten,
        qicuten=qicuten, qccuten=qccuten)
    reduce_nssl2_precipitation(
        workspace, rainnc, rainncv, snownc, snowncv,
        graupelnc, graupelncv, hailnc, hailncv, sr)
    scatter_nssl2_driver_workspace(
        workspace, air_density,
        qv, qc, qr, qi, qs, qg, qh,
        qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh)
    return workspace


__all__ = [
    "NSSL2_DRIVER_IGNORED_ACCUMULATOR_SCRATCH",
    "NSSL2_DRIVER_STATE_SCRATCH",
    "NSSL2_DRIVER_SURFACE_EXPORT_SCRATCH",
    "NSSL2_DRIVER_FIELDS",
    "NSSL2_SEDIMENT_EXPORTS",
    "NSSL2DriverWorkspace",
    "gather_initialize_and_sediment",
    "launch_nssl2_driver_support",
    "reduce_nssl2_precipitation",
    "scatter_nssl2_driver_workspace",
    "validate_nssl2_driver_workspace",
]
