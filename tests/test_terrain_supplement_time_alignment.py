"""An invariant terrain supplement may be declared once per cycle.

Some producers write their static fields -- orography above all -- only
into the analysis frame of a cycle, not into every forecast step (ECMWF's
open data does exactly this).  The composition may DECLARE that shape:
``time_alignment: "cycle_invariant_broadcast"`` decodes the terrain wherever
the supplement supplies it, requires it invariant across every supplied
frame exactly as before, and carries the one invariant field to every
primary valid time.  The default ``valid_time_exact`` behaviour is
unchanged: a primary valid time with no terrain frame refuses.
"""

from __future__ import annotations

from datetime import datetime
from types import MappingProxyType

import numpy as np
import pytest

from gpuwm.mapped_composition import _compose_terrain
from gpuwm.mapped_source import _DecodedCollection, _DirectValue


_T0 = datetime(2026, 8, 16, 0)
_T3 = datetime(2026, 8, 16, 3)
_LAT = np.array([-1.0, 0.0, 1.0])
_LON = np.array([10.0, 11.0])


def _value(name: str, valid_time: datetime, values: np.ndarray) -> _DirectValue:
    return _DirectValue(
        name=name, valid_time=valid_time, member=None, source_cycle=_T0,
        axes=("y", "x"), values=values, missing_count=0,
        references=(f"@{name}",),
    )


def _primary() -> _DecodedCollection:
    field = np.zeros((3, 2))
    return _DecodedCollection(
        latitude=_LAT, longitude=_LON, vertical_values=np.array([100000.0]),
        direct=MappingProxyType({
            (_T0, None, "surface_pressure"): _value(
                "surface_pressure", _T0, field),
            (_T3, None, "surface_pressure"): _value(
                "surface_pressure", _T3, field),
        }),
        source_cycles=MappingProxyType({(_T0, None): _T0, (_T3, None): _T0}),
        grid_fingerprint="f" * 64,
    )


def _terrain(times: tuple[datetime, ...]) -> _DecodedCollection:
    values = np.arange(6, dtype=np.float64).reshape(3, 2)
    return _DecodedCollection(
        latitude=_LAT, longitude=_LON, vertical_values=np.array([]),
        direct=MappingProxyType({
            (when, None, "terrain_height"): _value(
                "terrain_height", when, values)
            for when in times
        }),
        source_cycles=MappingProxyType({(when, None): _T0 for when in times}),
        grid_fingerprint="f" * 64,
    )


def test_exact_alignment_still_refuses_a_missing_primary_time():
    with pytest.raises(ValueError, match="lacks exact primary valid time"):
        _compose_terrain(_primary(), _terrain((_T0,)))


def test_broadcast_alignment_carries_the_invariant_field_to_every_time():
    combined, receipt = _compose_terrain(
        _primary(), _terrain((_T0,)), time_alignment="cycle_invariant_broadcast")
    for when in (_T0, _T3):
        entry = combined.direct[(when, None, "terrain_height")]
        np.testing.assert_array_equal(
            entry.values, np.arange(6, dtype=np.float64).reshape(3, 2))
        assert entry.valid_time == when
    assert receipt["time_alignment"] == "cycle_invariant_broadcast"
    assert receipt["broadcast_primary_valid_times"] == [_T3.isoformat()]


def test_broadcast_alignment_with_full_coverage_broadcasts_nothing():
    combined, receipt = _compose_terrain(
        _primary(), _terrain((_T0, _T3)),
        time_alignment="cycle_invariant_broadcast")
    assert receipt["broadcast_primary_valid_times"] == []
    assert set(combined.direct) >= {
        (_T0, None, "terrain_height"), (_T3, None, "terrain_height")}


def test_broadcast_alignment_still_requires_invariance():
    terrain = _terrain((_T0, _T3))
    drifted = dict(terrain.direct)
    drifted[(_T3, None, "terrain_height")] = _value(
        "terrain_height", _T3, np.ones((3, 2)))
    terrain = _DecodedCollection(
        latitude=terrain.latitude, longitude=terrain.longitude,
        vertical_values=terrain.vertical_values,
        direct=MappingProxyType(drifted),
        source_cycles=terrain.source_cycles,
        grid_fingerprint=terrain.grid_fingerprint,
    )
    with pytest.raises(ValueError, match="changes across supplied valid times"):
        _compose_terrain(
            _primary(), terrain, time_alignment="cycle_invariant_broadcast")


def test_an_undeclared_alignment_spelling_refuses():
    with pytest.raises(ValueError, match="time_alignment"):
        _compose_terrain(
            _primary(), _terrain((_T0,)), time_alignment="nearest_neighbour")
