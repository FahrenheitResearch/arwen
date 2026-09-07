"""Typed, exact support for local atmospheric interpolation operands.

The source axes remain the original axes. Surface and soil arrays are never
windowed: their masked searches can select donors outside a local stencil.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from gpuwm.ingest.grib import Era5Snapshot


ATMOSPHERIC_FIELDS = frozenset({
    "T", "PRES", "SPFH", "U", "V", "GHT", "QC", "QR", "QI", "QS", "QG",
})
CANONICAL_ATMOSPHERIC_FIELDS = frozenset({
    "air_temperature", "air_pressure", "specific_humidity", "eastward_wind",
    "northward_wind", "geopotential_height",
})
WINDOW_SCHEMA = "gpuwm-mapped-atmospheric-window-v1"
WINDOWED_FRAMESET_SCHEMA = "gpuwm-mapped-windowed-frameset-v1"


@dataclass(frozen=True)
class AtmosphericWindow:
    source_shape: tuple[int, int]
    rows: tuple[int, int]
    columns: tuple[int, int]

    def __post_init__(self):
        for name in ("source_shape", "rows", "columns"):
            values = getattr(self, name)
            if len(values) != 2 or any(
                    isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                    for v in values):
                raise ValueError(f"atmospheric window {name} needs two integers")
            object.__setattr__(self, name, tuple(int(v) for v in values))
        for bounds, size in zip((self.rows, self.columns), self.source_shape):
            if not 0 <= bounds[0] < bounds[1] <= size:
                raise ValueError("atmospheric window must lie inside its source grid")

    @property
    def shape(self):
        return (self.rows[1] - self.rows[0], self.columns[1] - self.columns[0])

    def crop(self, values):
        if values.ndim != 3 or values.shape[-2:] != self.source_shape:
            raise ValueError("atmospheric field must retain all levels on its source grid")
        return values[:, slice(*self.rows), slice(*self.columns)]

    def contains(self, support):
        return support is not None and all(
            bounds[0] <= part.start and part.stop <= bounds[1]
            for bounds, part in zip((self.rows, self.columns),
                                    (support.rows, support.columns)))

    def contains_window(self, other):
        return (self.source_shape == other.source_shape and all(
            a[0] <= b[0] and b[1] <= a[1]
            for a, b in zip((self.rows, self.columns), (other.rows, other.columns))))

    def document(self, fields):
        return {"schema": WINDOW_SCHEMA, "source_shape": list(self.source_shape),
                "rows": list(self.rows), "columns": list(self.columns),
                "fields": sorted(fields),
                "operation": "regular-parabolic-bilinear-original-fp32-support"}


@dataclass(frozen=True)
class WindowedAtmosphericField:
    """An operand in original-source index space, not a smaller source grid."""
    values: object
    window: AtmosphericWindow

    def __post_init__(self):
        if self.values.ndim != 3 or self.values.shape[-2:] != self.window.shape:
            raise ValueError("windowed atmospheric operand has the wrong shape")

    def for_support(self, support, source_shape):
        if tuple(source_shape) != self.window.source_shape or not self.window.contains(support):
            raise ValueError("atmospheric window does not cover this interpolation support")
        return self.values[:,
            support.rows.start - self.window.rows[0]:support.rows.stop - self.window.rows[0],
            support.columns.start - self.window.columns[0]:support.columns.stop - self.window.columns[0]]


@dataclass(frozen=True)
class WindowedAtmosphericSnapshot(Era5Snapshot):
    window: AtmosphericWindow = field(kw_only=True)
    full_factory: Callable[[], Era5Snapshot] = field(kw_only=True, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.window, AtmosphericWindow) or not callable(self.full_factory):
            raise TypeError("windowed snapshot requires a typed window and full-source provider")
        if self.window.source_shape != (len(self.latitude), len(self.longitude)):
            raise ValueError("windowed snapshot must retain the original full source axes")
        super().__post_init__()
        if any(values.ndim != 3 for name, values in self.fields.items()
               if name in ATMOSPHERIC_FIELDS):
            raise ValueError("windowed atmospheric fields must retain a vertical dimension")

    def _field_horizontal_shape(self, name, horizontal):
        return self.window.shape if name in ATMOSPHERIC_FIELDS else horizontal

    def operand(self, name, values):
        return (WindowedAtmosphericField(values, self.window)
                if name in ATMOSPHERIC_FIELDS else values)

    def full_snapshot(self):
        """Reload original atmospheric fields, preserving any surface overlays."""
        full = self.full_factory()
        if (type(full) is not Era5Snapshot or full.valid_time != self.valid_time
                or full.projection != self.projection
                or any(not np.array_equal(getattr(full, name), getattr(self, name))
                       for name in ("latitude", "longitude", "levels_hpa"))):
            raise ValueError("full atmospheric provider changed its source geometry, clock, or ladder")
        return full.with_fields({name: values for name, values in self.fields.items()
                                 if name not in ATMOSPHERIC_FIELDS})

    def save_npz(self, path):
        # The old archive codec has no window descriptor. Its unchanged full
        # representation remains available rather than silently losing axes.
        self.full_snapshot().save_npz(path)


def atmospheric_window_for_grids(metadata, grids):
    """Union the existing FP32 mass/U/V stencils on the original geometry.

    A cyclic source keeps its full representation for now: reorientation may
    choose a different cut for another domain. Unknown/unproven support uses
    the same full fallback, without rejecting a source or a requested feature.
    """
    from gpuwm.ingest.horiz import (
        _regular_coordinates, global_longitude_period_columns,
        source_coordinate_transform,
    )
    from gpuwm.ingest.interpolation_support import regular_source_support
    from gpuwm.ingest.source_coverage import SourceCoverageRefusal

    shape = (len(metadata.latitude), len(metadata.longitude))
    if metadata.projection is None and global_longitude_period_columns(metadata.longitude) is not None:
        return None
    transform, _ = source_coordinate_transform(metadata)
    supports = []
    for grid in grids:
        for lat, lon in (grid.latlon_mass(), grid.latlon_u(), grid.latlon_v()):
            y, x = transform(lat, lon)
            try:
                y, x = _regular_coordinates(metadata.latitude, metadata.longitude, y, x)
            except SourceCoverageRefusal:
                # The existing consumer owns the original geometry refusal.
                return None
            support = regular_source_support(shape, y, x)
            if support is None:
                return None
            supports.append(support)
    if not supports:
        return None
    result = AtmosphericWindow(shape,
        (min(s.rows.start for s in supports), max(s.rows.stop for s in supports)),
        (min(s.columns.start for s in supports), max(s.columns.stop for s in supports)))
    return None if result.shape == shape else result


def window_request_response(event, grids):
    """Answer the writer only from its validated original geometry."""
    from gpuwm.ingest.source_metadata import SourceSnapshotMetadata
    from gpuwm.mapped_engine_bridge import _axis_values
    if event.get("contract") != WINDOW_SCHEMA:
        raise ValueError("mapped writer requested an unknown atmospheric window contract")
    latitude = _axis_values(event["latitude"], "latitude")
    longitude = _axis_values(event["longitude"], "longitude")
    geometry = event["geometry"]
    if (geometry["latitude_sha256"] != event["latitude"]["sha256"]
            or geometry["longitude_sha256"] != event["longitude"]["sha256"]):
        raise ValueError("mapped writer window geometry identity differs from its axes")
    grid = event["grid"]
    projection = (None if grid["projection"] == "regular_latitude_longitude"
                  else {"family": grid["projection"], "parameters": grid["parameters"]})
    metadata = SourceSnapshotMetadata(Era5Snapshot, latitude, longitude, projection)
    window = atmospheric_window_for_grids(metadata, grids)
    fields = CANONICAL_ATMOSPHERIC_FIELDS.intersection(event["fields"])
    response = {"schema": WINDOW_SCHEMA, "geometry": geometry, "mode": "full"}
    if window is not None and fields:
        response.update(window.document(fields), mode="window")
    return response
