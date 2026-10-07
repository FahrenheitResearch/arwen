"""Array checks shared by the pure GOCART column launchers."""
from __future__ import annotations

import numpy as np


def field(array, shape, label):
    import cupy as cp
    if not isinstance(array, cp.ndarray):
        raise TypeError(f"{label} must be a CuPy array")
    if array.dtype != np.float32 or array.shape != shape or not array.flags.c_contiguous:
        raise ValueError(f"{label} must be contiguous float32 with shape {shape}")
    return array


def columns(species):
    import cupy as cp
    if not species:
        raise ValueError("at least one species row is required")
    shape = species[0].shape
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("species shape must be (nz, ny, nx) with positive extents")
    spans = []
    for i, a in enumerate(species):
        field(a, shape, f"row {i}")
        span = (a.data.ptr, a.data.ptr + a.nbytes)
        if any(span[0] < end and start < span[1] for start, end in spans):
            raise ValueError("species rows must not overlap")
        spans.append(span)
    pointers = cp.asarray(np.array([a.data.ptr for a in species], dtype=np.uint64))
    return shape, pointers


def timestep(dt, *, allow_zero=False):
    dt = np.float32(dt)
    lower = 0 if allow_zero else 1
    if not np.isfinite(dt) or dt < lower or dt > 86400:
        raise ValueError(f"dt must be finite and in {lower}..86400 seconds")
    return dt


def parameter(array, size, dtype, label):
    import cupy as cp
    if not isinstance(array, cp.ndarray) or array.shape != (size,) or array.dtype != dtype:
        raise ValueError(f"{label} must be a device {dtype} array of length {size}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{label} must be contiguous")
    return array
