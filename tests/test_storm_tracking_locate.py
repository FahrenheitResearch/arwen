"""``StormTracker.locate`` -- where the storm is, with no side effects.

The tracker's job splits cleanly in two: find the vortex, then decide
whether that is worth a move.  Only the first half is of any use to a
consumer that is not the mover -- the track writer wants a position, not
a proposal -- and asking for one through ``desired_shift`` would burn a
cooldown, leave a decision receipt, and change where the nest goes.

So ``locate`` is the first half, extracted, and this file pins the two
properties the extraction has to have:

* **One implementation.**  ``desired_shift`` is ``locate`` plus
  hysteresis, so a run's deck and its nest cannot disagree about where
  the vortex was.  A second centre-finder that drifted from the first is
  exactly the failure ``[relocation.containment]`` was shaped to avoid
  ("two trackers can disagree and fight").
* **No side effects at all.**  No receipt, no cooldown anchor, no
  hysteresis state.  Calling it a hundred times between cadences must
  leave the next ``desired_shift`` byte-identical to the one that would
  have happened had nothing called it -- that is what lets the track
  writer emit oftener than the nest moves.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import gpuwm.core.storm_tracking as st
from gpuwm.core.storm_tracking import FollowConfig, NestFootprint, StormTracker


def _footprint():
    return NestFootprint(grid_id=2, i_parent_start=102, j_parent_start=31,
                         child_nx=240, child_ny=240, parent_grid_ratio=3)


def _config(**over):
    base = dict(field="pressure", threshold=30.0, level_hpa=850.0,
                search_margin_cells=12, min_shift_cells=1,
                max_shift_cells=4, cooldown_seconds=600.0)
    base.update(over)
    return FollowConfig(**base)


def _bowl(ny, nx, cj, ci, depth=60.0, width=6.0):
    j, i = np.mgrid[0:ny, 0:nx]
    r2 = (j - cj) ** 2 + (i - ci) ** 2
    return (1500.0 - depth * np.exp(-r2 / (2.0 * width ** 2))).astype(
        np.float64)


def _plane_patch(monkeypatch, plane):
    monkeypatch.setattr(st, "_plane_from_state",
                        lambda *a, **k: plane)


def _tree():
    """Mover d02 at ratio 3 with a refine grid d03 at ratio 4."""
    d03 = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=3, i_parent_start=83, j_parent_start=83,
                            parent_grid_ratio=4),
        children=[], state=SimpleNamespace(), _started=True)
    d02 = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=2, i_parent_start=102, j_parent_start=31,
                            parent_grid_ratio=3),
        children=[d03], state=SimpleNamespace(), _started=True)
    return d02, d03


# ---------------------------------------------------------------------------
# The fix itself
# ---------------------------------------------------------------------------

def test_locate_returns_the_centre_and_the_extremum(monkeypatch):
    fp = _footprint()
    ci, cj = fp.center_parent_ij
    _plane_patch(monkeypatch, _bowl(200, 220, cj + 2.0, ci + 3.0))
    tracker = StormTracker(_config())
    fix = tracker.locate(SimpleNamespace(), fp, 600.0)
    assert fix.found is not None
    got_i, got_j = fix.center_parent_ij
    assert got_i == pytest.approx(ci + 3.0, abs=0.3)
    assert got_j == pytest.approx(cj + 2.0, abs=0.3)
    # A height minimum, in metres, restored to the field's own sign.
    assert fix.extremum == pytest.approx(1440.0, abs=1.0)
    assert fix.field_used == "pressure"
    assert fix.plane_shape == (200, 220)
    assert fix.raw_shift[0] == pytest.approx(3.0, abs=0.3)
    assert fix.raw_shift[1] == pytest.approx(2.0, abs=0.3)


def test_locate_on_a_no_signal_still_carries_complete_evidence(monkeypatch):
    """A hold that records nothing cannot be audited.

    The ABSOLUTE MSLP ceiling (``level_hpa = 0``) is the configuration
    that can genuinely find nothing: on an isobaric surface the threshold
    is measured from the search box's own minimum, so some cell always
    qualifies -- which is the whole point of the relative form and is
    asserted below.
    """
    fp = _footprint()
    _plane_patch(monkeypatch, np.full((200, 220), 1010.0))
    fix = StormTracker(
        _config(level_hpa=0, threshold=1004.0)).locate(
            SimpleNamespace(), fp, 600.0)
    assert fix.found is None
    assert fix.center_parent_ij is None and fix.extremum is None
    for key in ("t", "field_requested", "field_used", "threshold_used",
                "signal", "search_box", "footprint"):
        assert key in fix.evidence, key
    assert fix.evidence["footprint"]["grid_id"] == 2


def test_a_flat_box_is_no_signal_not_a_dead_band_hold(monkeypatch):
    """A relative threshold used to find something in ANY field, and that
    was the defect.

    ``locate_signal`` builds its ceiling as ``box minimum + threshold``,
    so on a box flatter than the threshold EVERY cell qualifies -- and
    the centroid of every cell in a box is the box's own centre, which
    is the nest's own centre, which rounds to a null shift.  A nest that
    had completely lost its storm therefore reported
    ``suppressed:dead-band``: indistinguishable from a nest sitting
    perfectly on its vortex.

    MEASURED with a storm parked 28 cells outside the search box: the
    box spanned 0.597 m against a 30 m threshold, the centroid landed on
    (164.0, 126.0) against a footprint centre of (163.929, 125.929), and
    the tracker held while calling it a dead-band.

    The criterion needs no tuning constant -- ``span < threshold`` IS
    "every cell qualifies", exactly.
    """
    fp = _footprint()
    _plane_patch(monkeypatch, np.full((200, 220), 1500.0))
    fix = StormTracker(_config()).locate(SimpleNamespace(), fp, 600.0)
    assert fix.found is None
    declined = fix.evidence["levels_declined"]
    assert len(declined) == 1
    assert declined[0]["signal_span"] == 0.0
    assert "not in the box" in declined[0]["reason"]


def test_the_box_span_is_recorded_even_when_it_is_fine(monkeypatch):
    """The one number that separates 'centred on the storm' from 'the
    storm is gone' on a held receipt, so it rides on every consultation
    rather than only on the bad ones."""
    fp = _footprint()
    ci, cj = fp.center_parent_ij
    _plane_patch(monkeypatch, _bowl(200, 220, cj, ci))
    fix = StormTracker(_config()).locate(SimpleNamespace(), fp, 600.0)
    assert fix.found is not None
    assert fix.evidence["search_box_signal_span"] > 30.0


def test_a_box_spanning_just_over_the_threshold_still_tracks(monkeypatch):
    """The boundary is not a cliff into refusal: a shallow but real
    vortex is still a vortex."""
    fp = _footprint()
    ci, cj = fp.center_parent_ij
    _plane_patch(monkeypatch, _bowl(200, 220, cj, ci, depth=31.0))
    fix = StormTracker(_config()).locate(SimpleNamespace(), fp, 600.0)
    assert fix.found is not None
    assert "levels_declined" not in fix.evidence


# ---------------------------------------------------------------------------
# No side effects -- the property the track writer's interval rests on
# ---------------------------------------------------------------------------

def test_locate_appends_no_receipt(monkeypatch):
    fp = _footprint()
    _plane_patch(monkeypatch, _bowl(200, 220, 60.0, 140.0))
    tracker = StormTracker(_config())
    tracker.drain_receipts()                      # drop the "configured" row
    for _ in range(25):
        tracker.locate(SimpleNamespace(), fp, 600.0)
    assert tracker.receipts == []


def test_locate_does_not_touch_the_cooldown_anchor(monkeypatch):
    fp = _footprint()
    ci, cj = fp.center_parent_ij
    _plane_patch(monkeypatch, _bowl(200, 220, cj + 6.0, ci + 6.0))
    tracker = StormTracker(_config(cooldown_seconds=1800.0))
    assert tracker.desired_shift(SimpleNamespace(), fp, 600.0) is not None
    anchor = tracker._last_proposal_t
    for t in range(700, 1400, 50):
        tracker.locate(SimpleNamespace(), fp, float(t))
    assert tracker._last_proposal_t == anchor
    assert tracker._last_move_t is None


def test_locating_between_cadences_changes_no_decision(monkeypatch):
    """The whole point: emit a track row oftener than the nest moves,
    and the nest goes to exactly the same places."""
    fp = _footprint()
    ci, cj = fp.center_parent_ij

    def run(with_extra_locates):
        planes = {}

        def plane_for(state, field, **kw):
            # A storm drifting steadily east-north-east.
            t = planes["t"]
            return _bowl(200, 220, cj + t / 400.0, ci + t / 300.0)

        monkeypatch.setattr(st, "_plane_from_state", plane_for)
        tracker = StormTracker(_config(cooldown_seconds=600.0))
        decisions = []
        for step in range(1, 25):
            t = 300.0 * step
            planes["t"] = t
            if with_extra_locates:
                for sub in range(1, 5):
                    planes["t"] = t - 300.0 + 60.0 * sub
                    tracker.locate(SimpleNamespace(), fp, planes["t"])
                planes["t"] = t
            decisions.append(tracker.desired_shift(SimpleNamespace(), fp, t))
        return decisions

    assert run(False) == run(True)


def test_receipts_are_identical_with_and_without_extra_locates(monkeypatch):
    fp = _footprint()
    ci, cj = fp.center_parent_ij
    _plane_patch(monkeypatch, _bowl(200, 220, cj + 6.0, ci + 6.0))

    def run(extra):
        tracker = StormTracker(_config(cooldown_seconds=0.0))
        for step in range(1, 6):
            t = 360.0 * step
            for _ in range(extra):
                tracker.locate(SimpleNamespace(), fp, t)
            tracker.desired_shift(SimpleNamespace(), fp, t)
        return tracker.receipts

    assert run(0) == run(7)


# ---------------------------------------------------------------------------
# Where the fine quantities come from
# ---------------------------------------------------------------------------

def test_refined_grid_and_cell_are_reported_when_stage_two_applies(
        monkeypatch):
    """A consumer wanting central pressure from the grid that resolves
    the vortex must be told WHICH grid and WHERE on it."""
    d02, _ = _tree()
    fp = _footprint()
    ci, cj = fp.center_parent_ij
    planes = iter((_bowl(200, 220, cj, ci - 3.0),      # parent, wrong by 3
                   _bowl(300, 300, 149.5, 149.5)))     # d03, centred
    monkeypatch.setattr(st, "_plane_from_state", lambda *a, **k: next(planes))
    refinement = st.refinement_from_node(d02, 3)
    fix = StormTracker(_config(refine_grid_id=3)).locate(
        SimpleNamespace(), fp, 600.0, refinement=refinement)
    assert fix.refined_on == 3
    assert fix.refined_cell_ij == pytest.approx((149.5, 149.5), abs=0.3)
    assert fix.evidence["refinement"]["applied"] is True


def test_no_refinement_reports_none_rather_than_a_guess(monkeypatch):
    fp = _footprint()
    _plane_patch(monkeypatch, _bowl(200, 220, 60.0, 140.0))
    fix = StormTracker(_config()).locate(SimpleNamespace(), fp, 600.0)
    assert fix.refined_on is None and fix.refined_cell_ij is None


def test_a_declined_refinement_reports_none(monkeypatch):
    """Declining keeps stage one's centre, so the fine grid must NOT be
    named -- a consumer would otherwise read a quantity off a grid whose
    own answer was rejected."""
    d02, _ = _tree()
    fp = _footprint()
    ci, cj = fp.center_parent_ij
    # The refined centre lands inside the refine grid's edge margin.
    planes = iter((_bowl(200, 220, cj, ci), _bowl(300, 300, 3.0, 3.0)))
    monkeypatch.setattr(st, "_plane_from_state", lambda *a, **k: next(planes))
    fix = StormTracker(_config(refine_grid_id=3)).locate(
        SimpleNamespace(), fp, 600.0,
        refinement=st.refinement_from_node(d02, 3))
    assert fix.evidence["refinement"]["applied"] is False
    assert fix.refined_on is None and fix.refined_cell_ij is None


def test_desired_shift_and_locate_agree_on_the_centre(monkeypatch):
    """One implementation, asserted rather than assumed."""
    fp = _footprint()
    ci, cj = fp.center_parent_ij
    _plane_patch(monkeypatch, _bowl(200, 220, cj + 5.0, ci + 5.0))
    tracker = StormTracker(_config(cooldown_seconds=0.0))
    fix = tracker.locate(SimpleNamespace(), fp, 600.0)
    tracker.desired_shift(SimpleNamespace(), fp, 600.0)
    row = next(r for r in tracker.receipts if r.get("decision") == "proposed")
    assert row["centroid_parent_ij"] == [
        round(fix.center_parent_ij[0], 3), round(fix.center_parent_ij[1], 3)]
