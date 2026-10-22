"""An MSLP-minimum follower parks when its storm is gone, and follows only the storm.

WHAT BREAKAGE THIS PREVENTS (gate law): WOOF 1.1 readiness g4 (GFS Isaias,
12 km parent, 3 km following nest).  The follower took the lowest cell of its
search box as the centre under a 1100 hPa ceiling every cell clears.  After the
storm left the parent the track writer ended (2026-10-12_08:45), and the nest
made 60+ more moves on no storm: along the parent's east edge, then west to
Iowa on search-box-edge minima, then back to Pennsylvania.  Early in the run
it also flipped between two centres 100 km apart every 15 minutes.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import gpuwm.core.storm_tracking as st
from gpuwm.core.storm_tracking import FollowConfig, NestFootprint, StormTracker

NY, NX = 160, 200
DX = 12000.0


def _footprint(i=80, j=60):
    # A 40-parent-cell nest at ratio 4 on a 12 km parent: the cyclone shape.
    return NestFootprint(grid_id=2, i_parent_start=i, j_parent_start=j,
                         child_nx=160, child_ny=160, parent_grid_ratio=4,
                         parent_dx_m=DX)


def _config():
    return FollowConfig(field="pressure", threshold=1100.0, level_hpa=0.0,
                        radius_km=60.0, search_margin_cells=20,
                        min_shift_cells=2, max_shift_cells=12,
                        cooldown_seconds=900.0, pressure_center="minimum",
                        fallback_level_hpa=850.0, fallback_deficit_m=25.0)


def _low(plane, cj, ci, depth, width=8.0):
    j, i = np.mgrid[0:NY, 0:NX]
    plane -= depth * np.exp(-((j - cj) ** 2 + (i - ci) ** 2) / (2 * width ** 2))
    return plane


def _patch(monkeypatch, plane, calls=None):
    def source(state, field, *, level_hpa=None, **kwargs):
        if calls is not None:
            calls.append(level_hpa)
        return plane
    monkeypatch.setattr(st, "_plane_from_state", source)


def _centre(fp):
    ci, cj = fp.center_parent_ij
    return int(round(ci)), int(round(cj))


def test_a_closed_low_is_followed(monkeypatch):
    fp = _footprint()
    ci, cj = _centre(fp)
    _patch(monkeypatch, _low(np.full((NY, NX), 1012.0), cj, ci + 6, 6.0))
    tracker = StormTracker(_config())
    assert tracker(SimpleNamespace(), fp, 900.0) == (6, 0)
    identity = tracker.last_fix.evidence["storm_identity"]
    assert identity["decision"] == "lowest"
    assert identity["lowest"]["deficit_hpa"] > st.IDENTITY_MIN_DEFICIT_HPA


def test_a_slope_on_the_search_box_edge_is_not_a_centre(monkeypatch):
    """The Iowa leg: a pressure gradient's lowest cell is the box edge."""
    fp = _footprint()
    j, i = np.mgrid[0:NY, 0:NX]
    calls = []
    _patch(monkeypatch, 1012.0 + 0.02 * i, calls)
    tracker = StormTracker(_config())
    assert tracker(SimpleNamespace(), fp, 900.0) is None
    identity = tracker.last_fix.evidence["storm_identity"]
    assert identity["decision"] == "no-closed-low"
    assert identity["lowest"]["on_search_box_edge"]
    # A slope is not a missing MSLP field: the 850 hPa fallback is not asked.
    assert 850.0 not in calls
    assert tracker.receipts[-1]["decision"] == "no-signal"


def test_a_minimum_that_reached_the_parent_edge_parks_the_nest(monkeypatch):
    """The storm leaving the parent ends the track and the nest stays put."""
    fp = _footprint(i=150)
    ci, cj = _centre(fp)
    # The storm, centred just past the parent's east edge.
    _patch(monkeypatch, _low(np.full((NY, NX), 1012.0), cj, NX + 2, 10.0))
    tracker = StormTracker(_config())
    assert tracker(SimpleNamespace(), fp, 900.0) is None
    assert tracker.receipts[-1]["decision"] == "parked:track-ended"
    # A perfectly good low later is NOT followed: the track has ended.
    ci, cj = _centre(fp)
    _patch(monkeypatch, _low(np.full((NY, NX), 1012.0), cj, ci - 8, 8.0))
    assert tracker(SimpleNamespace(), fp, 1800.0) is None
    assert tracker.receipts[-1]["decision"] == "parked"
    # And the park survives a restart.
    resumed = StormTracker(_config())
    resumed.restore_state(tracker.state_json())
    assert resumed(SimpleNamespace(), fp, 2700.0) is None
    assert resumed.receipts[-1]["decision"] == "parked"


def test_a_minimum_flipping_out_of_reach_keeps_the_storm_it_had(monkeypatch):
    """The 10-11 10:45-12:30 flip-flop: two centres 9 cells apart."""
    fp = _footprint()
    ci, cj = _centre(fp)
    tracker = StormTracker(_config())
    _patch(monkeypatch, _low(np.full((NY, NX), 1012.0), cj, ci + 3, 5.0))
    assert tracker(SimpleNamespace(), fp, 900.0) == (3, 0)
    tracker.notify_move_executed(900.0, (3, 0))
    moved = _footprint(i=83)
    # 15 min later a second minimum 9 cells north, a hair deeper.
    plane = _low(np.full((NY, NX), 1012.0), cj, ci + 3, 5.0)
    _low(plane, cj + 9, ci + 3, 3.0, width=3.0)
    _patch(monkeypatch, plane)
    tracker(SimpleNamespace(), moved, 1800.0)
    identity = tracker.last_fix.evidence["storm_identity"]
    assert identity["decision"] == "continuity"
    got_i, got_j = tracker.last_fix.center_parent_ij
    # The first low, nudged at most a cell by the second's flank.
    assert got_i == ci + 3 and abs(got_j - cj) <= 1


def test_a_clearly_deeper_centre_out_of_reach_takes_over(monkeypatch):
    """A coastal redevelopment: the new centre is deeper by the deficit."""
    fp = _footprint()
    ci, cj = _centre(fp)
    tracker = StormTracker(_config())
    _patch(monkeypatch, _low(np.full((NY, NX), 1012.0), cj, ci, 5.0))
    tracker(SimpleNamespace(), fp, 900.0)
    plane = _low(np.full((NY, NX), 1012.0), cj, ci, 4.0)
    _low(plane, cj - 2, ci + 25, 8.0)
    _patch(monkeypatch, plane)
    tracker(SimpleNamespace(), fp, 1800.0)
    assert tracker.last_fix.evidence["storm_identity"]["decision"] == "redeveloped"
    assert tracker.last_fix.center_parent_ij == (ci + 25, cj - 2)


def test_three_hours_without_a_storm_parks_and_ends_the_track(monkeypatch):
    fp = _footprint()
    ci, cj = _centre(fp)
    tracker = StormTracker(_config())
    _patch(monkeypatch, _low(np.full((NY, NX), 1012.0), cj, ci, 5.0))
    tracker(SimpleNamespace(), fp, 900.0)
    j, i = np.mgrid[0:NY, 0:NX]
    _patch(monkeypatch, 1012.0 + 0.02 * i)      # the storm has filled
    t = 1800.0
    while tracker.receipts[-1]["decision"] != "parked:storm-lost":
        assert tracker(SimpleNamespace(), fp, t) is None
        assert t - 1800.0 <= st.IDENTITY_LOST_AFTER_S
        t += 900.0
    assert "track_end_reason" in tracker.last_fix.evidence
    resumed = StormTracker(_config())
    resumed.restore_state(tracker.state_json())
    assert resumed._parked == tracker._parked
