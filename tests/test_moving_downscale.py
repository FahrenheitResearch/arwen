"""gpuwm downscale --follow: the plan a moving step-down runs on.

The window cut and the warm start were proven on a real storm run (Milton,
2024-10-09 12Z); these
pin the arithmetic they rest on.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gpuwm.downscale import _centered_placement
from gpuwm.moving_downscale import (
    Footprint, MovingDownscaleRefusal, _start_offset, child_targets,
    plan_segments)


@pytest.mark.parametrize("span", range(10, 82))
def test_start_offset_is_the_doors_own_centring(span):
    # The point handed to each segment is the child's centre cell, so the
    # door's centring must put the child's first cell exactly at the plan's.
    ratio = 3
    parent = {"nx": 400, "ny": 400}
    for i0 in (span, span + 1, 150):
        placed = _centered_placement(parent, j0=i0, i0=i0, ratio=ratio,
                                     child_nx=span * ratio,
                                     child_ny=span * ratio)
        assert placed.i_parent_start - 1 == i0 - _start_offset(span)


def _moving(n=9, step=(4, 2), every=4, nx=200, ny=200):
    start = datetime(2024, 10, 9, 13)
    rows = []
    for k in range(n):
        moves = k // every
        rows.append(Footprint(path=Path(f"wrfout_d02_{k}"),
                              valid_time=start + timedelta(minutes=15 * k),
                              origin_i=moves * step[0], origin_j=moves * step[1],
                              nx=nx, ny=ny))
    return rows


class _Lattice:
    """A lattice whose mass index is the degree (1-based)."""

    def latlon_to_ij(self, lat, lon):
        return float(lon) + 1.0, float(lat) + 1.0


def test_a_child_following_its_parent_moves_with_it_and_stays_inside():
    frames = _moving()
    targets, basis = child_targets(frames, _Lattice(), mode="parent",
                                   point=(100.0, 100.0))
    assert basis == "its place in the moving parent"
    segments = plan_segments(frames, targets, span_i=60, span_j=60,
                             start=frames[0].valid_time, seconds=7200,
                             every=3600, edge=6)
    assert len(segments) == 2
    first, second = segments
    # One parent move (4, 2) inside the second hour, and the child follows
    # the mean place of the parent over its stretch.
    assert (second.child_i - first.child_i, second.child_j - first.child_j) in (
        (4, 2), (3, 2), (4, 1), (3, 1))
    for segment in segments:
        for frame in segment.frames:
            assert segment.window_i >= frame.origin_i + 6
            assert segment.window_i + segment.window_nx <= frame.origin_i + frame.nx - 6
            assert segment.window_j >= frame.origin_j + 6
            assert segment.window_j + segment.window_ny <= frame.origin_j + frame.ny - 6


def test_a_fixed_child_at_the_parent_edge_is_clamped_inside_every_frame():
    frames = _moving(step=(20, 0))
    targets, _ = child_targets(frames, _Lattice(), mode="fixed",
                               point=(100.0, 30.0))
    segments = plan_segments(frames, targets, span_i=60, span_j=60,
                             start=frames[0].valid_time, seconds=7200,
                             every=3600, edge=6)
    assert segments[1].clamped
    for frame in segments[1].frames:
        assert segments[1].window_i >= frame.origin_i + 6


def test_a_child_too_big_for_the_moving_parent_is_refused_by_name():
    frames = _moving(step=(40, 0))
    targets, _ = child_targets(frames, _Lattice(), mode="parent",
                               point=(100.0, 100.0))
    with pytest.raises(MovingDownscaleRefusal, match="never leaves room"):
        plan_segments(frames, targets, span_i=150, span_j=60,
                      start=frames[0].valid_time, seconds=7200, every=3600,
                      edge=6)


def test_segments_end_on_frames():
    frames = _moving()
    targets, _ = child_targets(frames, _Lattice(), mode="parent",
                               point=(100.0, 100.0))
    with pytest.raises(MovingDownscaleRefusal, match="frame"):
        plan_segments(frames, targets, span_i=60, span_j=60,
                      start=frames[0].valid_time, seconds=7200, every=1000,
                      edge=6)


def test_strip_rain_is_interpolated_from_the_parent_cells():
    import numpy as np

    from gpuwm.moving_downscale import _bilinear_to_child

    # A field linear in the parent index is reproduced at every child centre.
    span, ratio = 4, 3
    jj, ii = np.meshgrid(np.arange(-1, span + 1), np.arange(-1, span + 1),
                         indexing="ij")
    field = 2.0 * ii + 5.0 * jj
    child = _bilinear_to_child(field.astype(float), ratio=ratio, span_i=span,
                               span_j=span)
    x = (np.arange(span * ratio) + 0.5) / ratio - 0.5
    assert np.allclose(child, 2.0 * x[np.newaxis, :] + 5.0 * x[:, np.newaxis])


def test_the_earlier_childs_specified_ring_takes_the_parents_rain():
    from gpuwm.core.nest_relocation import Placement, plan_relocation
    from gpuwm.offline_child_warm import _strip_mask

    plan = plan_relocation(
        placement_from=Placement(grid_id=3, i_parent_start=10, j_parent_start=10),
        placement_to=Placement(grid_id=3, i_parent_start=13, j_parent_start=10),
        parent_grid_ratio=1, child_nx=12, child_ny=8)
    mask = _strip_mask(plan, (8, 12), ring=1)
    # The 3 new columns on the east, and the old west edge column... is left
    # behind; the old top and bottom rows still inside, and the old east
    # column, now 3 cells in, take the parent's rain.
    assert mask[:, 9:].all()
    assert mask[0, :].all() and mask[7, :].all()
    assert mask[:, 8].all()           # old east edge (11) -> 11 - 3 = 8
    assert not mask[1:7, 0:8].any()
