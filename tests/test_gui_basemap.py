"""The committed ``gpuwm gui`` basemap is what its builder writes.

``tools/gui_basemap.py`` simplifies the vendored shapefiles with its own
numpy Douglas-Peucker instead of shapely, which no shipped dependency
table declares.  The breakage this prevents: a builder whose arithmetic
drifts from the file the page serves, so the next rebuild silently moves
every coastline and county line on the map.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools import gui_basemap

pytest.importorskip("shapefile")


def test_douglas_peucker_keeps_endpoints_and_the_far_point():
    part = np.array([[0.0, 0.0], [1.0, 0.52], [2.0, 1.0], [3.0, 0.52], [4.0, 0.0]])
    out = gui_basemap._douglas_peucker(part, 0.1)
    assert out.tolist() == [[0.0, 0.0], [2.0, 1.0], [4.0, 0.0]]


def test_douglas_peucker_on_a_closed_ring_measures_from_the_shared_endpoint():
    ring = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    out = gui_basemap._douglas_peucker(ring, 0.5)
    assert out[0].tolist() == out[-1].tolist() == [0.0, 0.0]
    assert [1.0, 1.0] in out.tolist()


def test_ring_area_is_unsigned_and_closes_an_open_ring():
    square = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]])
    assert gui_basemap._ring_area(square) == 4.0
    assert gui_basemap._ring_area(square[::-1]) == 4.0


@pytest.mark.slow
def test_rebuild_reproduces_the_committed_basemap(tmp_path: Path):
    out = tmp_path / "basemap.bin"
    gui_basemap.build(out)
    assert out.read_bytes() == gui_basemap.OUT.read_bytes()
