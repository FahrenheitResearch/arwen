"""Scale splitting, parent/child blending and weak large-scale nudging."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .backend import namespace, scalar
from .scalar import BoundaryMode, apply_transfer
from .transfer import RaisedCosineLowPass


@dataclass(frozen=True)
class ScaleSplit:
    low: Any
    high: Any
    reconstruction_max_abs: float


@dataclass(frozen=True)
class NudgingResult:
    values: Any
    increment_rms: float
    increment_max_abs: float
    relaxation_fraction: float
    boundary_max_abs: float


def split_scales(array: Any, *, dy_m: float, dx_m: float,
                 low_pass: RaisedCosineLowPass,
                 boundary: BoundaryMode = "reflect",
                 edge_taper_cells: int = 12,
                 periodic_domain: bool = False) -> ScaleSplit:
    """Split one field into exactly reconstructing low and residual parts."""
    xp = namespace(array)
    values = xp.asarray(array)
    low = apply_transfer(
        values, dy_m=dy_m, dx_m=dx_m,
        transfer_factory=low_pass.transfer, boundary=boundary,
        edge_taper_cells=edge_taper_cells, periodic_domain=periodic_domain,
        preserve_mean=True)
    high = values - low
    closure = low + high - values
    return ScaleSplit(low=low, high=high,
                      reconstruction_max_abs=scalar(xp.max(xp.abs(closure))))


def blend_parent_child(parent_on_child: Any, child: Any, *, dy_m: float,
                       dx_m: float, low_pass: RaisedCosineLowPass,
                       boundary: BoundaryMode = "tapered",
                       edge_taper_cells: int = 12,
                       periodic_domain: bool = False):
    """Use the parent's low modes and the child's residual on one common grid."""
    if parent_on_child.shape != child.shape:
        raise ValueError("parent/child scale blend requires one common grid shape")
    parent = split_scales(
        parent_on_child, dy_m=dy_m, dx_m=dx_m, low_pass=low_pass,
        boundary=boundary, edge_taper_cells=edge_taper_cells,
        periodic_domain=periodic_domain)
    offspring = split_scales(
        child, dy_m=dy_m, dx_m=dx_m, low_pass=low_pass,
        boundary=boundary, edge_taper_cells=edge_taper_cells,
        periodic_domain=periodic_domain)
    return parent.low + offspring.high


def nudge_large_scales(current: Any, target: Any, *, dy_m: float, dx_m: float,
                       dt_s: float, relaxation_time_s: float,
                       low_pass: RaisedCosineLowPass,
                       boundary: BoundaryMode = "tapered",
                       edge_taper_cells: int = 12,
                       periodic_domain: bool = False) -> NudgingResult:
    """Apply exact weak relaxation of only the selected large-scale mismatch."""
    xp = namespace(current)
    if namespace(target) is not xp or current.shape != target.shape:
        raise ValueError("spectral nudging operands must share backend and shape")
    if not all(math.isfinite(v) and v > 0.0
               for v in (dt_s, relaxation_time_s)):
        raise ValueError("nudging dt and relaxation time must be positive and finite")
    difference = xp.asarray(target) - xp.asarray(current)
    selected = apply_transfer(
        difference, dy_m=dy_m, dx_m=dx_m,
        transfer_factory=low_pass.transfer, boundary=boundary,
        edge_taper_cells=edge_taper_cells, periodic_domain=periodic_domain,
        preserve_mean=False)
    fraction = 1.0 - math.exp(-dt_s / relaxation_time_s)
    increment = fraction * selected
    values = xp.asarray(current) + increment
    edge = xp.concatenate((
        xp.ravel(increment[..., 0, :]), xp.ravel(increment[..., -1, :]),
        xp.ravel(increment[..., :, 0]), xp.ravel(increment[..., :, -1]),
    ))
    return NudgingResult(
        values=values,
        increment_rms=scalar(xp.sqrt(xp.mean(increment * increment,
                                               dtype=xp.float64))),
        increment_max_abs=scalar(xp.max(xp.abs(increment))),
        relaxation_fraction=float(fraction),
        boundary_max_abs=scalar(xp.max(xp.abs(edge))),
    )


__all__ = ["NudgingResult", "ScaleSplit", "blend_parent_child",
           "nudge_large_scales", "split_scales"]
