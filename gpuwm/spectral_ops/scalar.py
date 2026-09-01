"""Boundary-safe scalar spectral filtering and exact hyperdiffusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from .backend import namespace, scalar
from .grid import (angular_wavenumbers, crop_even_extension, edge_window,
                   even_extension)
from .transfer import Hyperdiffusion

BoundaryMode = Literal["periodic", "reflect", "tapered"]
TransformSpace = Literal["linear", "log"]


@dataclass(frozen=True)
class ScalarResult:
    values: Any
    mean_before: float
    mean_after: float
    rms_increment: float
    max_abs_increment: float
    minimum_before: float
    minimum_after: float


def _mean(array: Any):
    xp = namespace(array)
    return xp.mean(array, axis=(-2, -1), keepdims=True)


def _recenter(output: Any, original: Any):
    return output + (_mean(original) - _mean(output))


def _periodic_filter(array: Any, *, dy_m: float, dx_m: float, transfer_factory):
    xp = namespace(array)
    coeff = xp.fft.rfft2(array, axes=(-2, -1))
    _ky, _kx, _k2, magnitude = angular_wavenumbers(
        array, dy_m=dy_m, dx_m=dx_m)
    transfer = transfer_factory(magnitude)
    result = xp.fft.irfft2(coeff * transfer, s=array.shape[-2:], axes=(-2, -1))
    return result.astype(array.dtype, copy=False)


def apply_transfer(array: Any, *, dy_m: float, dx_m: float,
                   transfer_factory, boundary: BoundaryMode = "reflect",
                   edge_taper_cells: int = 12,
                   periodic_domain: bool = False, preserve_mean: bool = True):
    """Apply one real-valued transfer over the last two dimensions.

    Periodic wrapping is never inferred.  A caller must set
    ``periodic_domain=True`` or the periodic branch refuses before transforming
    anything.  ``reflect`` is the scalar default.  ``tapered`` leaves every
    outer-edge cell exactly unchanged and is useful when boundary values are
    externally prescribed.
    """
    xp = namespace(array)
    values = xp.asarray(array)
    if values.ndim < 2 or values.size == 0:
        raise ValueError("a scalar spectral operand needs two non-empty horizontal axes")
    if not bool(xp.all(xp.isfinite(values))):
        raise ValueError("scalar spectral operand is non-finite")
    if boundary == "periodic":
        if not periodic_domain:
            raise ValueError(
                "periodic spectral wrapping requires periodic_domain=true")
        result = _periodic_filter(values, dy_m=dy_m, dx_m=dx_m,
                                  transfer_factory=transfer_factory)
    elif boundary == "reflect":
        ny, nx = values.shape[-2:]
        extended = even_extension(values)
        result = crop_even_extension(
            _periodic_filter(extended, dy_m=dy_m, dx_m=dx_m,
                             transfer_factory=transfer_factory), ny, nx)
    elif boundary == "tapered":
        window = edge_window(values, edge_taper_cells)
        anomaly = values - _mean(values)
        working = anomaly * window
        filtered = _periodic_filter(
            working, dy_m=dy_m, dx_m=dx_m,
            transfer_factory=transfer_factory)
        result = values + window * (filtered - working)
    else:
        raise ValueError(f"unsupported scalar spectral boundary mode {boundary!r}")
    if preserve_mean:
        if boundary == "tapered":
            # Repair the correction's mean without changing the outer edge.
            window = edge_window(values, edge_taper_cells)
            delta = result - values
            correction = _mean(delta)
            weight_mean = _mean(window)
            result = result - window * correction / weight_mean
        else:
            result = _recenter(result, values)
    return result.astype(values.dtype, copy=False)


def _operate_in_space(array: Any, *, space: TransformSpace, floor: float,
                      preserve_mean: bool, operation):
    xp = namespace(array)
    if space == "linear":
        return operation(array)
    if space != "log":
        raise ValueError(f"unsupported transform space {space!r}")
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("log-space floor must be positive and finite")
    if bool(xp.any(array < 0.0)):
        raise ValueError("log-space spectral operation refuses negative input")
    before_mean = _mean(array)
    transformed = xp.log(xp.maximum(array, floor))
    output = xp.exp(operation(transformed))
    if preserve_mean:
        after_mean = _mean(output)
        output = output * xp.where(after_mean > 0.0, before_mean / after_mean, 1.0)
    return xp.maximum(output, floor).astype(array.dtype, copy=False)


def hyperdiffuse(array: Any, *, dy_m: float, dx_m: float, dt_s: float,
                 spec: Hyperdiffusion, boundary: BoundaryMode = "reflect",
                 edge_taper_cells: int = 12, periodic_domain: bool = False,
                 preserve_mean: bool = True, space: TransformSpace = "linear",
                 floor: float = 1.0e-20) -> ScalarResult:
    """Apply an exact exponential hyperdiffusion step."""
    xp = namespace(array)
    values = xp.asarray(array)

    def operation(operand):
        return apply_transfer(
            operand, dy_m=dy_m, dx_m=dx_m,
            transfer_factory=lambda magnitude: spec.transfer(magnitude, dt_s=dt_s),
            boundary=boundary, edge_taper_cells=edge_taper_cells,
            periodic_domain=periodic_domain, preserve_mean=preserve_mean)

    result = _operate_in_space(
        values, space=space, floor=floor, preserve_mean=preserve_mean,
        operation=operation)
    increment = result - values
    return ScalarResult(
        values=result,
        mean_before=scalar(xp.mean(values, dtype=xp.float64)),
        mean_after=scalar(xp.mean(result, dtype=xp.float64)),
        rms_increment=scalar(xp.sqrt(xp.mean(increment * increment,
                                               dtype=xp.float64))),
        max_abs_increment=scalar(xp.max(xp.abs(increment))),
        minimum_before=scalar(xp.min(values)),
        minimum_after=scalar(xp.min(result)),
    )


__all__ = ["BoundaryMode", "ScalarResult", "TransformSpace", "apply_transfer",
           "hyperdiffuse"]
