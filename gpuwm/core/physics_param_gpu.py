"""GPU scaling for the small table metadata of an active parameter set."""

from __future__ import annotations


def scale_physics_param_values(values, factors, crop_flags):
    """Scale table entries in binary64 and round once to binary32.

    The three sequences have matching lengths. ``crop_flags`` marks Z0
    entries for seasonal-crop classes, whose roughness must exceed the
    surface kernel's 0.125 m decrement. Every result must be finite and
    positive. Empty sequences return without opening a CUDA device.
    The forecast calls this during physics initialization for registered
    table cells only, never for a weather grid or a timestep field.
    """
    count = len(values)
    if len(factors) != count or len(crop_flags) != count:
        raise ValueError("physics parameter values, factors and crop flags differ in length")
    if not count:
        return ()

    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    inputs = cp.asarray(values, dtype=cp.float64)
    multipliers = cp.asarray(factors, dtype=cp.float64)
    crops = cp.asarray(crop_flags, dtype=cp.uint8)
    if any(array.shape != (count,) for array in (inputs, multipliers, crops)):
        raise ValueError("physics parameter values, factors and crop flags must be one-dimensional")
    outputs = cp.empty(count, dtype=cp.float32)
    status = cp.empty(count, dtype=cp.uint8)
    kernel = get_kernel("physics_params", "scale_physics_param_values")
    kernel(((count + 127) // 128,), (128,),
           (inputs, multipliers, crops, outputs, status, count))
    errors = status.get().tolist()
    for index, code in enumerate(errors):
        if code == 1:
            raise ValueError(f"physics parameter edit {index} produced a nonfinite value")
        if code == 2:
            raise ValueError(f"physics parameter edit {index} produced a nonpositive value")
        if code == 3:
            raise ValueError(
                f"physics parameter edit {index} produced seasonal-crop roughness "
                "at or below the 0.125 m surface-kernel decrement")
    return tuple(outputs.get().tolist())
