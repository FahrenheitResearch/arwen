"""Radiation-facing adapter for the admitted NSSL radius diagnostic."""

from __future__ import annotations

import numpy as np

from gpuwm.core.kernels import get_kernel
from gpuwm.core.state import DTYPE


_TPB = 256


def _validate_structural(fields: dict[str, object]) -> tuple[int, ...]:
    first = next(iter(fields.values()))
    shape = first.shape
    if len(shape) != 3:
        raise ValueError(f"NSSL radius fields must be 3-D, got {shape}")
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    return shape


def _validate_inputs(fields: dict[str, object]) -> None:
    import cupy as cp

    for name, value in fields.items():
        if bool(cp.any(~cp.isfinite(value))):
            raise ValueError(f"{name} must contain only finite values")
        if name == "air_density":
            if bool(cp.any(value <= DTYPE(0.0))):
                raise ValueError("air_density must be strictly positive")
        elif bool(cp.any(value < DTYPE(0.0))):
            raise ValueError(f"{name} must be nonnegative")


def launch_effective_radius_concentration(
        air_density, qc, cloud_number, qi, ice_number, qs, snow_number,
        re_cloud_m, re_ice_m, re_snow_m, *,
        validate_values: bool = True) -> None:
    """Diagnose WRF NSSL radii directly from internal number concentration.

    Number inputs are #/m3 views of the production NSSL slab, mass fields are
    kg/kg dry air, and outputs are metres.  Unlike the frozen Registry-facing
    launcher, this entry point does not multiply #/kg by density and therefore
    introduces no scatter/gather unit round-trip before radiation.
    """
    inputs = {
        "air_density": air_density,
        "qc": qc,
        "cloud_number": cloud_number,
        "qi": qi,
        "ice_number": ice_number,
        "qs": qs,
        "snow_number": snow_number,
    }
    outputs = {
        "re_cloud_m": re_cloud_m,
        "re_ice_m": re_ice_m,
        "re_snow_m": re_snow_m,
    }
    shape = _validate_structural({**inputs, **outputs})
    if validate_values:
        _validate_inputs(inputs)

    import cupy as cp

    for output_name, output in outputs.items():
        if any(cp.may_share_memory(output, value)
               for value in inputs.values()):
            raise ValueError(
                f"{output_name} must not alias an NSSL radius input")
    if (cp.may_share_memory(re_cloud_m, re_ice_m)
            or cp.may_share_memory(re_cloud_m, re_snow_m)
            or cp.may_share_memory(re_ice_m, re_snow_m)):
        raise ValueError("NSSL effective-radius outputs must not alias")

    size = int(np.prod(shape, dtype=np.int64))
    blocks = (size + _TPB - 1) // _TPB
    get_kernel(
        "nssl2_diagnostics", "nssl2_effective_radius_concentration")(
            (blocks,), (_TPB,),
            (air_density, qc, cloud_number, qi, ice_number, qs, snow_number,
             re_cloud_m, re_ice_m, re_snow_m, np.int32(size)))


def update_effective_radii(
        state, *, air_density=None, radiation_due: bool = True,
        validate_values: bool = True) -> bool:
    """Populate ``state.effc/effi/effs`` in RRTMGP's micron convention.

    The already admitted low-level NSSL kernel intentionally preserves WRF's
    metre output boundary.  This adapter supplies the DomainState fields,
    writes those transient metre values directly into the final radius arrays,
    converts them in place to microns, and enforces the native WRF bounds.

    ``air_density`` is required on a due call so this independently admitted
    adapter neither allocates nor claims an integration-owned scratch slot.
    ``radiation_due=False`` returns before inspecting state or density and is
    bitwise no-op for all effective-radius fields.
    """
    if not isinstance(radiation_due, bool):
        raise TypeError("radiation_due must be bool")
    if not radiation_due:
        return False

    import cupy as cp

    required = (
        "qc", "qndrop", "qi", "qni", "qs", "qns",
        "effc", "effi", "effs")
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError(
            "NSSL effective radius lacks state fields: " + ", ".join(missing))

    if air_density is None:
        raise ValueError(
            "NSSL effective radius requires an admitted air_density field")

    inputs = {
        "air_density": air_density,
        "qc": state.qc,
        "qndrop": state.qndrop,
        "qi": state.qi,
        "qni": state.qni,
        "qs": state.qs,
        "qns": state.qns,
    }
    _validate_structural({
        **inputs,
        "effc": state.effc,
        "effi": state.effi,
        "effs": state.effs,
    })
    if validate_values:
        _validate_inputs(inputs)

    # Lazy import prevents this adapter from changing the frozen numerical
    # kernel module or option-18 selector behavior merely by being imported.
    from gpuwm.core.nssl2 import launch_effective_radius

    launch_effective_radius(
        air_density,
        state.qc, state.qndrop,
        state.qi, state.qni,
        state.qs, state.qns,
        state.effc, state.effi, state.effs)

    metre_to_micron = DTYPE(1.0e6)
    cp.multiply(state.effc, metre_to_micron, out=state.effc)
    cp.multiply(state.effi, metre_to_micron, out=state.effi)
    cp.multiply(state.effs, metre_to_micron, out=state.effs)

    if validate_values:
        bounds = (
            ("effc", state.effc, 2.51e-6, 50.0e-6),
            ("effi", state.effi, 10.01e-6, 125.0e-6),
            ("effs", state.effs, 25.0e-6, 999.0e-6),
        )
        for name, value, lower_m, upper_m in bounds:
            # Form bounds through the same FP32 metre-to-micron operation as
            # the outputs.  Decimal micron literals can round one ULP higher
            # than a valid native lower-bound result (notably 10.01 um).
            lower = DTYPE(lower_m) * metre_to_micron
            upper = DTYPE(upper_m) * metre_to_micron
            invalid = (~cp.isfinite(value)) | (value < lower) | (value > upper)
            if bool(cp.any(invalid)):
                raise RuntimeError(
                    f"NSSL {name} escaped native [{float(lower)}, "
                    f"{float(upper)}] micron "
                    "bounds")
    return True


__all__ = [
    "launch_effective_radius_concentration",
    "update_effective_radii",
]
