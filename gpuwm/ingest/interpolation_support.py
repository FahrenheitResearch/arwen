"""Payload support for a regular plan's existing FP32 coordinates.

This does not rebase geographic/projection axes or change donor arithmetic.
Masked/global searches and nearest-neighbor tie rules keep the full source.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegularSourceSupport:
    rows: slice
    columns: slice
    y: np.ndarray
    x: np.ndarray

    @property
    def shape(self):
        return (self.rows.stop - self.rows.start,
                self.columns.stop - self.columns.start)

    def crop(self, field, source_shape):
        from gpuwm.ingest.atmospheric_window import WindowedAtmosphericField
        if isinstance(field, WindowedAtmosphericField):
            return field.for_support(self, source_shape)
        if not hasattr(field, "shape"):
            field = np.asarray(field)
        if field.ndim < 2 or field.shape[-2:] != source_shape:
            raise ValueError("field trailing dimensions do not match source axes")
        return field[..., self.rows, self.columns]


def regular_source_support(source_shape, y, x):
    """Return proven parabolic/bilinear support, or None for the full path.

    Original FP32 indices own the fractions.  Subtracting the integer crop
    origin from those values is exact; recomputing local coordinates from
    original FP64 target axes is a different rounding path.  Prove the exact
    round trip anyway and retain the full source if it cannot be shown.
    """
    ny, nx = source_shape
    if min(ny, nx) < 2 or max(ny, nx) > 2**24:
        return None
    y = np.asarray(y, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if y.shape != x.shape or y.size == 0:
        return None
    if (not np.isfinite(y).all() or not np.isfinite(x).all()
            or np.any(y < 0) or np.any(y > ny - 1)
            or np.any(x < 0) or np.any(x > nx - 1)):
        return None
    y0 = max(0, int(np.floor(y.min())) - 1)
    y1 = min(ny, int(np.floor(y.max())) + 3)
    x0 = max(0, int(np.floor(x.min())) - 1)
    x1 = min(nx, int(np.floor(x.max())) + 3)
    if (y0, y1, x0, x1) == (0, ny, 0, nx):
        return None
    local = []
    for values, origin in ((y, y0), (x, x0)):
        shifted = (values.astype(np.float64) - origin).astype(np.float32)
        if not np.array_equal(
                shifted.astype(np.float64) + origin, values.astype(np.float64)):
            return None
        shifted.setflags(write=False)
        local.append(shifted)
    return RegularSourceSupport(slice(y0, y1), slice(x0, x1), *local)
