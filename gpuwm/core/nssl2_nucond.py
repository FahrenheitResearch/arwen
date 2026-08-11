"""Production WRF v4.6.1 NSSL ``NUCOND`` warm-phase stage.

This module is intentionally independent of the still fail-closed option-18
coordinator.  It exposes the complete default ``ipconc=5`` warm ``NUCOND``
stage at a narrow array boundary: joint cloud/rain condensation (``rcond=2``),
cloud evaporation, predicted-CCN/droplet coupling, default ``irenuc=2``
renucleation, the ordinary and 1.9-ratio maximum-supersaturation ``QVEXCESS``
paths, and native moment cleanup/bounds.
"""

from __future__ import annotations

import math

import numpy as np

from gpuwm.core.kernels import get_kernel
from gpuwm.core.state import DTYPE


_TPB = 256


def _validate_cell_fields(fields: dict[str, object]) -> tuple[tuple[int, ...], int]:
    first = next(iter(fields.values()))
    shape = first.shape
    if len(shape) != 3:
        raise ValueError(f"NSSL NUCOND fields must be 3-D, got {shape}")
    if shape[0] < 3:
        raise ValueError(f"NSSL NUCOND requires nz >= 3, got {shape[0]}")
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    return shape, int(np.prod(shape, dtype=np.int64))


def _validate_values(
        fields: dict[str, object], positive: tuple[str, ...],
        nonnegative: tuple[str, ...]) -> None:
    import cupy as cp

    for name, value in fields.items():
        if bool(cp.any(~cp.isfinite(value))):
            raise ValueError(f"{name} must contain only finite values")
        if name in positive:
            if bool(cp.any(value <= DTYPE(0.0))):
                raise ValueError(f"{name} must be strictly positive")
        elif name in nonnegative and bool(cp.any(value < DTYPE(0.0))):
            raise ValueError(f"{name} must be nonnegative")


def launch_nucond(
        full_theta, air_density, pressure_pa, exner, w_interface,
        qv, qc, qr, qi, qs, qndrop, qnr, qni, qns, qnn, dt_s: float, *,
        supersaturation_scratch=None,
        concentration_space: bool = False,
        predicted_ccn: bool = True,
        validate_values: bool = True) -> None:
    """Advance the exact default MP18 warm ``NUCOND`` stage in place.

    Cell fields use contiguous FP32 ``(nz, ny, nx)`` storage.  ``w_interface``
    is FP32 ``(nz + 1, ny, nx)`` so the kernel can reproduce WRF's
    ``0.5 * (w(k) + w(k+1))`` mass-level velocity.  Water fields are kg/kg
    dry air.  ``qi``, ``qs``, ``qni``, and ``qns`` are read-only but required
    because WRF gates clear-cloud CCN restoration on the primary-ice-plus-snow
    state remaining after its native mass/number cleanup.  By default
    number/CCN fields use Registry #/kg dry air.  Set
    ``concentration_space=True`` for production slab views already held in
    NSSL's internal #/m3 convention; that path performs no density round-trip.
    Potential temperature is K, pressure Pa, and density kg/m3.

    ``supersaturation_scratch`` may supply a reusable contiguous FP32 cell
    field.  The prepass reproduces WRF's immutable ``ssfilt`` field, avoiding
    an in-place neighbor-read race during existing-cloud renucleation.

    ``predicted_ccn`` is WRF's ``nssl_ccn_on``.  ``False`` selects the
    ``nssl_ccn_on=0`` variant, where the driver raises ``renucfrac`` to 1.0
    (``module_mp_nssl_2mom.F:2555-2557``) so the nucleation pool ``cnuc``
    becomes the actual diagnosed CCN instead of the background floor, and
    the low-temperature updraft limiter at :10120-10127 is armed.  The
    ``imaxsupopt=4`` saturation-adjustment activation is unaffected: it
    spells out ``Max(ccnc, cwnccn)`` at :11559 rather than using ``cnuc``.

    Value validation is synchronous and fail-closed by default.  Production
    callers that have already passed the global device health gate may set
    ``validate_values=False`` to avoid the redundant reductions; structural
    shape/dtype/contiguity and timestep checks are never optional.
    """
    fields = {
        "full_theta": full_theta,
        "air_density": air_density,
        "pressure_pa": pressure_pa,
        "exner": exner,
        "qv": qv,
        "qc": qc,
        "qr": qr,
        "qi": qi,
        "qs": qs,
        "qndrop": qndrop,
        "qnr": qnr,
        "qni": qni,
        "qns": qns,
        "qnn": qnn,
    }
    shape, size = _validate_cell_fields(fields)
    w_shape = (shape[0] + 1, shape[1], shape[2])
    if w_interface.shape != w_shape:
        raise ValueError(
            f"w_interface must have shape {w_shape}, got {w_interface.shape}")
    if w_interface.dtype != DTYPE:
        raise TypeError(f"w_interface must be float32, got {w_interface.dtype}")
    if not w_interface.flags.c_contiguous:
        raise ValueError("w_interface must be C-contiguous")
    if not isinstance(concentration_space, bool):
        raise TypeError("concentration_space must be bool")
    if not isinstance(predicted_ccn, bool):
        raise TypeError("predicted_ccn must be bool")

    if supersaturation_scratch is None:
        import cupy as cp

        supersaturation_scratch = cp.empty_like(qv)
    else:
        import cupy as cp

        scratch_shape, _ = _validate_cell_fields(
            {"supersaturation_scratch": supersaturation_scratch})
        if scratch_shape != shape:
            raise ValueError(
                "supersaturation_scratch must have shape "
                f"{shape}, got {scratch_shape}")
        if any(cp.may_share_memory(supersaturation_scratch, value)
               for value in fields.values()):
            raise ValueError(
                "supersaturation_scratch must not alias a NUCOND field")

    try:
        step = float(dt_s)
    except (TypeError, ValueError) as exc:
        raise TypeError("dt_s must be a positive finite scalar") from exc
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("dt_s must be a positive finite scalar")
    step32 = np.float32(step)
    if not np.isfinite(step32) or step32 <= 0.0:
        raise ValueError("dt_s must be representable as positive float32")

    if validate_values:
        _validate_values(
            {**fields, "w_interface": w_interface},
            ("full_theta", "air_density", "pressure_pa", "exner"),
            ("qv", "qc", "qr", "qi", "qs", "qndrop", "qnr", "qni",
             "qns", "qnn"))

    blocks = (size + _TPB - 1) // _TPB
    get_kernel("nssl2_nucond", "nssl2_nucond_supersaturation")(
        (blocks,), (_TPB,),
        (full_theta, pressure_pa, exner, qv, supersaturation_scratch,
         np.int32(size)))
    get_kernel("nssl2_nucond", "nssl2_nucond_default")(
        (blocks,), (_TPB,),
        (full_theta, air_density, pressure_pa, exner, w_interface,
         supersaturation_scratch,
         qv, qc, qr, qi, qs, qndrop, qnr, qni, qns, qnn, step32,
         np.int32(concentration_space),
         np.int32(size), np.int32(shape[0]),
         np.int32(shape[1] * shape[2]),
         np.int32(1 if predicted_ccn else 0)))


__all__ = ["launch_nucond"]
