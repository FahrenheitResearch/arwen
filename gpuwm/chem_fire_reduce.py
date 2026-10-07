"""Array-only Rust bridge for previous-day fire inputs.

The reduction lives in the CPU preprocessing bridge
(``tools/grib1_bridge/src/chem_fire_reduce.rs``, exported from
``libgpuwm_preprocess_cpu``), found by the same resolution ladder as every
other preprocessing call.

No Python decode, remap or reduction. SourceFrames supplies remapped arrays.
The reducer follows ufs-srweather-app 2ad2bc5731819cdc5aef95372a8b27ffe417bf40,
ush/smoke_dust/core/cycle.py:252-307,330-376, with kg/cell/hour inputs.
"""
import ctypes
import numpy as np


def daily_inputs(mass, frp, ages, hwp, rain):
    """Return (mean kg/cell/hour, MW, fire age, prior HWP, prior rain)."""
    from gpuwm.ingest.cpu_backend import resolve_cpu_bridge
    path = resolve_cpu_bridge()
    arrays = [np.ascontiguousarray(a, dtype=np.float32) for a in (mass,frp,ages,hwp,rain)]
    mass,frp,ages,hwp,rain = arrays
    if mass.ndim != 3 or mass.shape[0] != 24 or frp.shape != mass.shape or ages.shape != (24,):
        raise ValueError("daily_mean_dcycle needs 24 matching source planes and ages to prevent a biased emitting-hour mean")
    if hwp.ndim != 3 or hwp.shape[0] == 0 or rain.shape != hwp.shape or hwp.shape[1:] != mass.shape[1:]:
        raise ValueError("daily_mean_dcycle needs matching prior HWP and rain planes to prevent using unrelated restart history")
    out = np.empty((5,*mass.shape[1:]),np.float32)
    lib = ctypes.CDLL(str(path))
    try:
        fn = lib.gpuwm_fire_daily
    except AttributeError as exc:
        raise RuntimeError(
            f"{path} predates the daily fire reduction "
            "(tools/grib1_bridge/src/chem_fire_reduce.rs); rebuild tools/grib1_bridge, "
            "because reducing the previous day in Python would put data-path "
            "arithmetic outside Rust") from exc
    pointer = ctypes.POINTER(ctypes.c_float)
    fn.argtypes = [ctypes.c_size_t]*3 + [pointer]*6
    fn.restype = ctypes.c_int
    rc = fn(mass.shape[1]*mass.shape[2],24,hwp.shape[0],
            *(a.ctypes.data_as(pointer) for a in (*arrays,out)))
    if rc:
        raise ValueError(f"daily_mean_dcycle Rust reduction refused code {rc}: invalid source arrays would corrupt emission or fire-age accounting")
    return tuple(out)
