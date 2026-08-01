"""Field-comparison metrics for two WRF-shaped history streams.

Every function here takes its geometry and its thresholds as arguments.  The
module holds no campaign pins: no domain list, no grid spacing, no start
instant, no field list, no threshold.  A caller that wants a pinned metric set
registers the pins on its own side and passes them in, which is what makes the
same four metric classes usable for a campaign shadow gate, for a chaos
envelope over an arbitrary ensemble, and for a two-domain synthetic pair in a
unit test.

The four classes:

* **state** -- RMSE of the low-pass-filtered difference over the domain
  interior, aggregated over the scored times;
* **boundary** -- RMSE of the applied-increment difference over the outer
  ``spec_bdy_width`` frame of cells, aggregated over the scored times;
* **object** -- the valid time of the first qualifying composite-reflectivity
  object, from which callers form a timing difference;
* **neighborhood** -- fractions-skill-score distance ``1 - FSS`` at a physical
  radius, on ANY domain: the neighborhood half-width is derived from that
  domain's grid spacing, which is an argument.

Aggregation is pooled by element count, never an average of per-time RMSEs:
sums of squares and element counts accumulate across the scored times and the
square root is taken once.  Over equal-sized windows that is the quadratic
mean of the per-window RMSEs, which dominates their arithmetic mean, so one
loud window is not averaged away by a run of quiet ones.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import netCDF4
import numpy as np


def read_frame_field(path: str | Path, field: str) -> np.ndarray:
    """Read one leading-Time-dimension field as float64, refusing gaps."""
    path = Path(path)
    with netCDF4.Dataset(path) as dataset:
        if field not in dataset.variables:
            raise ValueError(f"history frame {path} is missing {field}")
        variable = dataset.variables[field]
        if variable.ndim < 3 or variable.shape[0] != 1:
            raise ValueError(f"field {field} in {path} has invalid shape")
        value = np.ma.asarray(variable[0])
        if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
            raise ValueError(f"field {field} in {path} carries masked data")
        result = np.asarray(value, dtype=np.float64)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"field {field} in {path} is empty/non-finite")
    return result


def boxcar(array: np.ndarray, width: int) -> np.ndarray:
    """Odd-width, edge-extended separable boxcar over the last two axes."""
    if width < 1 or width % 2 != 1 or array.ndim < 2:
        raise ValueError("boxcar width must be a positive odd integer")
    if width == 1:
        return np.asarray(array, dtype=np.float64)
    radius = width // 2
    result = np.asarray(array, dtype=np.float64)
    for axis in (-1, -2):
        pads = [(0, 0)] * result.ndim
        pads[axis] = (radius, radius)
        padded = np.pad(result, pads, mode="edge")
        prefix_shape = list(padded.shape)
        prefix_shape[axis] = 1
        prefix = np.concatenate(
            [np.zeros(prefix_shape, dtype=np.float64),
             np.cumsum(padded, axis=axis, dtype=np.float64)], axis=axis)
        high = [slice(None)] * result.ndim
        low = [slice(None)] * result.ndim
        high[axis] = slice(width, None)
        low[axis] = slice(None, -width)
        result = (prefix[tuple(high)] - prefix[tuple(low)]) / float(width)
    return result


def odd_width_cells(physical_width_m: float, dx_m: float) -> int:
    """Nearest odd cell count spanning ``physical_width_m`` at ``dx_m``."""
    cells = max(1, int(math.floor(physical_width_m / dx_m + 0.5)))
    return cells if cells % 2 else cells + 1


def half_width_cells(physical_radius_m: float, dx_m: float) -> int:
    """Neighborhood half-width in cells at this domain's grid spacing."""
    return max(0, int(math.floor(physical_radius_m / dx_m + 0.5)))


def minimum_object_cells(min_area_km2: float, dx_m: float) -> int:
    """Cell count covering ``min_area_km2`` at this domain's grid spacing."""
    return max(1, int(math.ceil(float(min_area_km2) * 1.0e6 / (dx_m * dx_m))))


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.size == 0:
        raise ValueError("RMSE operands must have one non-empty common shape")
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64)
    if not np.all(np.isfinite(difference)):
        raise ValueError("RMSE operand difference is non-finite")
    return float(np.sqrt(np.mean(difference * difference, dtype=np.float64)))


def aggregate_rmse(differences: Sequence[np.ndarray]) -> float:
    """Pooled RMSE over samples of possibly different sizes."""
    total = 0.0
    count = 0
    for difference in differences:
        value = np.asarray(difference, dtype=np.float64)
        if value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError("metric sample is empty/non-finite")
        total += float(np.sum(value * value, dtype=np.float64))
        count += value.size
    if count == 0:
        raise ValueError("metric has no samples")
    return float(math.sqrt(total / count))


def interior(array: np.ndarray, width: int) -> np.ndarray:
    ny, nx = array.shape[-2:]
    if ny <= 2 * width or nx <= 2 * width:
        raise ValueError("field is too small for the requested interior mask")
    return array[..., width:-width, width:-width]


def boundary_values(array: np.ndarray, width: int) -> np.ndarray:
    ny, nx = array.shape[-2:]
    if ny <= 2 * width or nx <= 2 * width:
        raise ValueError("field is too small for the requested boundary width")
    j, i = np.indices((ny, nx))
    mask = np.minimum.reduce((i, nx - 1 - i, j, ny - 1 - j)) < width
    return array[..., mask]


def neighborhood_fraction(events: np.ndarray, half_width: int) -> np.ndarray:
    if events.ndim != 2 or half_width < 0:
        raise ValueError("FSS events must be 2-D and radius non-negative")
    return boxcar(events.astype(np.float64), 2 * half_width + 1)


def fss_distance(left: np.ndarray, right: np.ndarray, *, threshold: float,
                 half_width: int) -> float:
    """``1 - FSS`` between two 2-D fields at one neighborhood radius."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("FSS operands must be common-shape 2-D fields")
    lhs = neighborhood_fraction(left >= threshold, half_width)
    rhs = neighborhood_fraction(right >= threshold, half_width)
    numerator = float(np.sum((lhs - rhs) ** 2, dtype=np.float64))
    denominator = float(np.sum(lhs * lhs + rhs * rhs, dtype=np.float64))
    fss = 1.0 if denominator == 0.0 else 1.0 - numerator / denominator
    distance = 1.0 - fss
    if not math.isfinite(distance):
        raise ValueError("FSS distance is non-finite")
    return float(min(1.0, max(0.0, distance)))


def composite(field: np.ndarray) -> np.ndarray:
    """Column maximum of a 3-D field; a 2-D field is already composite."""
    if field.ndim == 2:
        return field
    if field.ndim == 3:
        return np.max(field, axis=0)
    raise ValueError("composite input must be 2-D or 3-D after removing Time")


def has_qualifying_object(field: np.ndarray, *, threshold: float,
                          min_cells: int, connectivity: int = 8) -> bool:
    """NumPy-only connected-component test with deterministic scan order."""
    if field.ndim != 2 or min_cells < 1 or connectivity not in (4, 8):
        raise ValueError("invalid object labeling parameters")
    active = np.asarray(field >= threshold, dtype=bool)
    seen = np.zeros_like(active)
    ny, nx = active.shape
    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for j0 in range(ny):
        for i0 in range(nx):
            if not active[j0, i0] or seen[j0, i0]:
                continue
            seen[j0, i0] = True
            stack = [(j0, i0)]
            count = 0
            while stack:
                j, i = stack.pop()
                count += 1
                if count >= min_cells:
                    return True
                for dj, di in offsets:
                    jj, ii = j + dj, i + di
                    if (0 <= jj < ny and 0 <= ii < nx and active[jj, ii]
                            and not seen[jj, ii]):
                        seen[jj, ii] = True
                        stack.append((jj, ii))
    return False


def first_object_time(frames: Mapping[int, np.ndarray], *, threshold: float,
                      min_cells: int, quiet_time_seconds: float,
                      connectivity: int = 8) -> float:
    """Valid second of the first qualifying object, or the quiet sentinel."""
    for seconds in sorted(frames):
        if seconds == 0:
            continue
        if has_qualifying_object(
                composite(frames[seconds]), threshold=threshold,
                min_cells=min_cells, connectivity=connectivity):
            return float(seconds)
    return float(quiet_time_seconds)


def state_and_boundary_rmse(*, left_frames: Mapping[int, Path],
                            right_frames: Mapping[int, Path], field: str,
                            score_times: Iterable[int], cadence_seconds: int,
                            filter_width_cells: int,
                            interior_exclusion_cells: int,
                            boundary_width_cells: int,
                            label: str = "field") -> tuple[float, float]:
    """Pooled low-pass state RMSE and applied-boundary-increment RMSE.

    ``label`` only names the pair in error messages, so a caller can say which
    domain and field failed without this module knowing what a domain is.
    """
    low_pass_differences: list[np.ndarray] = []
    boundary_differences: list[np.ndarray] = []
    for seconds in score_times:
        lhs = read_frame_field(left_frames[seconds], field)
        rhs = read_frame_field(right_frames[seconds], field)
        if lhs.shape != rhs.shape:
            raise ValueError(f"{label} shapes differ")
        filtered = boxcar(lhs, filter_width_cells) - boxcar(
            rhs, filter_width_cells)
        low_pass_differences.append(
            interior(filtered, interior_exclusion_cells))
        lhs_prev = read_frame_field(left_frames[seconds - cadence_seconds], field)
        rhs_prev = read_frame_field(right_frames[seconds - cadence_seconds], field)
        if lhs_prev.shape != lhs.shape or rhs_prev.shape != rhs.shape:
            raise ValueError(f"{label} shape changes in time")
        increment_error = (lhs - lhs_prev) - (rhs - rhs_prev)
        boundary_differences.append(
            boundary_values(increment_error, boundary_width_cells))
    return (aggregate_rmse(low_pass_differences),
            aggregate_rmse(boundary_differences))


def mean_fss_distance(left_fields: Mapping[int, np.ndarray],
                      right_fields: Mapping[int, np.ndarray], *,
                      threshold: float, radius_m: float, dx_m: float,
                      times: Iterable[int]) -> float:
    """Mean ``1 - FSS`` over the scored times at one domain's spacing.

    The radius is physical and the spacing is this domain's, so the same call
    scores a 12 km parent and a 500 m child without either being privileged.
    """
    radius_cells = half_width_cells(radius_m, dx_m)
    distances = [
        fss_distance(composite(left_fields[seconds]),
                     composite(right_fields[seconds]),
                     threshold=threshold, half_width=radius_cells)
        for seconds in times]
    if not distances:
        raise ValueError("FSS series has no scored times")
    return float(np.mean(distances, dtype=np.float64))


__all__ = [
    "aggregate_rmse", "boundary_values", "boxcar", "composite",
    "first_object_time", "fss_distance", "half_width_cells",
    "has_qualifying_object", "interior", "mean_fss_distance",
    "minimum_object_cells", "neighborhood_fraction", "odd_width_cells",
    "read_frame_field", "rmse", "state_and_boundary_rmse",
]
