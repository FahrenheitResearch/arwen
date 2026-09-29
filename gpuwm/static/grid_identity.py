"""Comparing grid geometry recorded on one machine with another's.

A grid is identified by its definition: the projection and its parameters
as given, the spacing, the dimensions, and for a nest the integer start
and ratio that place it (:meth:`ProjectedGrid.definition`).  Those values
are parsed from text or are integers, so every machine holds the same
bits for them, and they are compared exactly.

Positions that come out of projection arithmetic are different.  A nest's
reference point, a grid's centre and its latitude/longitude extremes are
computed through trigonometric functions whose last bit depends on the
math library and CPU path, so the same grid built on two machines can
disagree in the last digit (41.73852197541409 against 41.7385219754141
for one grid's northern edge).  Comparing those exactly refuses
a tree prepared on one machine and run on another although nothing about
the grid differs.  They are compared within
:data:`GRID_POSITION_TOLERANCE_CELLS` of a grid cell instead.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

from gpuwm.static.projection import EARTH_RADIUS_M

#: How far two computations of the same grid position may disagree, as a
#: fraction of the grid spacing, before they are two different grids.
#:
#: The smallest real difference a derived position can carry is a whole
#: cell: every value that places a grid (a root anchor, a start index, a
#: ratio, a placement offset) is compared exactly beside it, so a moved or
#: re-sized grid moves its derived positions by at least one cell of the
#: finer grid.  Math-library disagreement is a few units in the last place
#: of a degree value, about 1e-14 degrees or a micrometre on the ground.
#: One thousandth of a cell sits three orders below the first and, even
#: on a 10 m grid (1 cm), about six orders above the second.
GRID_POSITION_TOLERANCE_CELLS = 1.0e-3

#: Ground distance of one degree of latitude on the model sphere.
_METRES_PER_DEGREE = EARTH_RADIUS_M * math.pi / 180.0

_MISSING = object()


def position_tolerance_deg(dx_m: float, dy_m: float) -> float:
    """:data:`GRID_POSITION_TOLERANCE_CELLS` of the finer spacing, in degrees.

    Applied to longitudes as well: a degree of longitude is never longer
    on the ground than a degree of latitude, so the same number of degrees
    is the same distance or less there.
    """
    spacing = min(abs(float(dx_m)), abs(float(dy_m)))
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(
            f"grid spacing must be positive and finite, got dx={dx_m!r}, "
            f"dy={dy_m!r}")
    return GRID_POSITION_TOLERANCE_CELLS * spacing / _METRES_PER_DEGREE


def _offset_deg(recorded, expected, *, longitude: bool) -> float | None:
    """Degrees between two recorded coordinates, or None if not numbers."""
    for value in (recorded, expected):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
    offset = float(recorded) - float(expected)
    if longitude:
        offset = (offset + 180.0) % 360.0 - 180.0
    if not math.isfinite(offset):
        return None
    return abs(offset)


def _coordinate_list(value, length: int) -> list | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    return list(value)


def grid_record_drift(
        recorded: Mapping[str, object], expected: Mapping[str, object], *,
        dx_m: float, dy_m: float,
        points: Iterable[tuple[str, str]] = (),
        ranges: Iterable[tuple[str, bool]] = (),
        exact: Iterable[str] = (),
        optional: Iterable[str] = (),
) -> dict[str, dict[str, object]]:
    """Name every key on which two grid records describe different grids.

    ``points`` are ``(latitude key, longitude key)`` pairs and ``ranges``
    ``(key, is_longitude)`` ``[min, max]`` extremes; both are compared
    within :func:`position_tolerance_deg` unless a key is listed in
    ``exact``, which names derived-looking keys that are given values on
    this grid (a root's anchor).  Every other key is compared exactly.
    A key in ``optional`` is compared only when both records carry it;
    it is how a record written before that key existed stays readable,
    and it is only ever a key whose content the positions also pin.
    Returns ``{key: {"recorded": ..., "expected": ...}}``, empty when the
    records describe the same grid.
    """
    tolerance = position_tolerance_deg(dx_m, dy_m)
    exact = frozenset(exact)
    optional = frozenset(optional)
    positional: dict[str, bool] = {}
    for latitude, longitude in points:
        positional[latitude] = False
        positional[longitude] = True
    range_keys = dict(ranges)
    drift: dict[str, dict[str, object]] = {}
    for name in sorted(set(recorded) | set(expected)):
        left = recorded.get(name, _MISSING)
        right = expected.get(name, _MISSING)
        if name in optional and (left is _MISSING or right is _MISSING):
            continue
        entry = {"recorded": None if left is _MISSING else left,
                 "expected": None if right is _MISSING else right}
        if left is _MISSING or right is _MISSING:
            drift[name] = entry
            continue
        if name in exact or (name not in positional
                             and name not in range_keys):
            if left != right:
                drift[name] = entry
            continue
        if name in positional:
            offset = _offset_deg(left, right, longitude=positional[name])
            if offset is None or offset > tolerance:
                drift[name] = {**entry, "degrees": offset,
                               "tolerance_degrees": tolerance}
            continue
        pair_left = _coordinate_list(left, 2)
        pair_right = _coordinate_list(right, 2)
        offsets = (None if pair_left is None or pair_right is None else [
            _offset_deg(a, b, longitude=range_keys[name])
            for a, b in zip(pair_left, pair_right)])
        if offsets is None or any(item is None for item in offsets):
            drift[name] = entry
        elif max(offsets) > tolerance:
            drift[name] = {**entry, "degrees": max(offsets),
                           "tolerance_degrees": tolerance}
    return drift


__all__ = [
    "GRID_POSITION_TOLERANCE_CELLS",
    "grid_record_drift",
    "position_tolerance_deg",
]
