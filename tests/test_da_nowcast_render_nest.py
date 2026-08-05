"""The render module's entry points for the fine nested field.

CPU only, and deliberately independent of the ``Gallery`` class: these
are the functions a gallery panel adopts, and they have to be usable
without a case directory, a basemap, or matplotlib.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.da_nowcast_render import (nest_composite_path, nest_latlon,
                                     nest_panel)

RATIO = 3
I_START = J_START = 5
PARENT_NY = PARENT_NX = 20
CHILD_NY = CHILD_NX = 9


def _parent_grid():
    """A parent whose lat/lon are exactly linear in grid index.

    Linearity is the point: bilinear sampling reproduces a linear field
    exactly, so the child coordinates can be checked against closed-form
    arithmetic rather than against a tolerance nobody can justify.
    """
    j = np.arange(PARENT_NY, dtype=float)[:, None]
    i = np.arange(PARENT_NX, dtype=float)[None, :]
    lat = 35.0 + 0.02 * np.broadcast_to(j, (PARENT_NY, PARENT_NX))
    lon = -97.0 + 0.03 * np.broadcast_to(i, (PARENT_NY, PARENT_NX))
    return lat, lon


def test_child_coordinates_follow_the_wrf_nest_down_pickup():
    lat, lon = _parent_grid()
    child_lat, child_lon = nest_latlon(
        lat, lon, i_parent_start=I_START, j_parent_start=J_START,
        ratio=RATIO, child_shape=(CHILD_NY, CHILD_NX))
    assert child_lat.shape == (CHILD_NY, CHILD_NX)

    for m in range(CHILD_NX):
        parent_index = (I_START - 1) - 0.5 + (m + 0.5) / RATIO
        assert child_lon[0, m] == pytest.approx(-97.0 + 0.03 * parent_index)
    for m in range(CHILD_NY):
        parent_index = (J_START - 1) - 0.5 + (m + 0.5) / RATIO
        assert child_lat[m, 0] == pytest.approx(35.0 + 0.02 * parent_index)


def test_the_child_centre_of_a_parent_cell_lands_on_the_parent_centre():
    """The middle sub-cell of an odd ratio sits on its donor exactly."""
    lat, lon = _parent_grid()
    child_lat, child_lon = nest_latlon(
        lat, lon, i_parent_start=I_START, j_parent_start=J_START,
        ratio=RATIO, child_shape=(CHILD_NY, CHILD_NX))
    # Child index 1 is the middle of the first covered parent cell.
    assert child_lon[0, 1] == pytest.approx(lon[0, I_START - 1])
    assert child_lat[1, 0] == pytest.approx(lat[J_START - 1, 0])


def test_the_child_spans_exactly_its_covered_parent_footprint():
    lat, lon = _parent_grid()
    child_lat, child_lon = nest_latlon(
        lat, lon, i_parent_start=I_START, j_parent_start=J_START,
        ratio=RATIO, child_shape=(CHILD_NY, CHILD_NX))
    covered = CHILD_NX // RATIO
    west_edge = (I_START - 1) - 0.5
    east_edge = west_edge + covered
    assert child_lon.min() > -97.0 + 0.03 * west_edge
    assert child_lon.max() < -97.0 + 0.03 * east_edge


def test_nest_latlon_refuses_a_malformed_parent_grid():
    with pytest.raises(ValueError, match="matching 2-D"):
        nest_latlon(np.zeros((4, 4)), np.zeros((4, 5)),
                    i_parent_start=2, j_parent_start=2, ratio=3,
                    child_shape=(3, 3))


def test_a_leg_without_a_nest_renders_as_none_not_an_error(tmp_path):
    """The assimilation legs have no nest; that is not a failure."""
    lat, lon = _parent_grid()
    assert nest_panel(tmp_path, 0, "control", lat, lon) is None


def test_nest_panel_reads_what_the_driver_writes(tmp_path):
    lat, lon = _parent_grid()
    composites = tmp_path / "composites"
    composites.mkdir()
    comp = np.arange(CHILD_NY * CHILD_NX, dtype=float).reshape(
        CHILD_NY, CHILD_NX)
    np.savez_compressed(
        nest_composite_path(tmp_path, 7, "control"),
        refl_colmax=comp.astype(np.float32),
        elapsed_seconds=np.float64(7200.0),
        dx_m=np.float64(1000.0),
        i_parent_start=np.int32(I_START),
        j_parent_start=np.int32(J_START),
        parent_grid_ratio=np.int32(RATIO))

    panel = nest_panel(tmp_path, 7, "control", lat, lon)
    assert panel is not None
    np.testing.assert_allclose(panel["refl_colmax"], comp)
    assert panel["dx_km"] == 1.0
    assert panel["dx_label"] == "Δx 1 km"
    assert panel["elapsed_seconds"] == 7200.0
    assert panel["lat"].shape == comp.shape
    assert panel["lon"].shape == comp.shape
    # The nest sits inside the parent's own footprint, which is what lets
    # a panel draw it on the same map frame.
    assert lat.min() <= panel["lat"].min() <= panel["lat"].max() <= lat.max()
    assert lon.min() <= panel["lon"].min() <= panel["lon"].max() <= lon.max()


def test_the_composite_name_carries_the_domain_so_both_can_be_drawn():
    path = nest_composite_path("cycle", 9, "3", grid_id=2)
    assert path.name == "leg09_3_d02.npz"
    assert path.parent.name == "composites"
