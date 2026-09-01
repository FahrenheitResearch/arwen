"""Array-backend helpers for regional spectral numerics.

The production forecast path may carry CuPy arrays while every policy,
registration and unit test must remain CPU-hermetic.  This module is the only
place that decides which array namespace owns an operand; importing it never
imports CuPy and never creates a CUDA context.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def is_cupy_array(value: Any) -> bool:
    module = type(value).__module__.split(".", 1)[0]
    return module == "cupy"


def namespace(value: Any):
    """Return :mod:`numpy` or the already-imported CuPy module for ``value``."""
    if is_cupy_array(value):
        import cupy as cp  # imported only for an actual CuPy operand
        return cp
    return np


def backend_name(value: Any) -> str:
    return "cupy" if is_cupy_array(value) else "numpy"


def asnumpy(value: Any) -> np.ndarray:
    if is_cupy_array(value):
        import cupy as cp
        return cp.asnumpy(value)
    return np.asarray(value)


def scalar(value: Any) -> float:
    """Move one scalar to the host without copying its parent array."""
    if is_cupy_array(value):
        return float(value.item())
    return float(value)


def synchronize(value: Any) -> None:
    """Synchronize only when an actual CuPy operand asks for a timed receipt."""
    if is_cupy_array(value):
        import cupy as cp
        cp.cuda.get_current_stream().synchronize()


__all__ = ["asnumpy", "backend_name", "is_cupy_array", "namespace", "scalar",
           "synchronize"]
