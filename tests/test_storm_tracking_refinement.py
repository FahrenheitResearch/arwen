"""Stage two of the tracker: refine the centre where the vortex is resolved.

Stage one finds the storm on the mover's PARENT, which is the only field
that can see outside the nest.  On a 13.5 km parent that is a coarse
answer for a nest whose whole purpose is 1.125 km, and the gap is not
academic.  MEASURED on Melissa (2025-10-22 01:12, d02 moving inside d01
with d03 at ratio 4):

    d01 (13.5 km) 850 hPa minimum   14.475 N, -73.422 E
    d02 ( 4.5 km) 850 hPa minimum   14.475 N, -73.509 E
    d03 (1.125km) 850 hPa minimum   14.375 N, -73.019 E

d01's answer sat 44.9 km from d03's, d02's sat 54.0 km from it, and the
tracker's own centroid 57.5 km -- so the nest was centred, to the
tracker's complete satisfaction, on a point 50-odd km from the vortex the
resolving grid actually had.  In a 338 km frame that is 16% off centre,
which is what it looks like in a plot.

Note the middle row: a finer PARENT is NOT the fix.  d02 at 4.5 km was
measured no better than d01 at 13.5 km, because at this stage the storm
is broad and each resolution finds a different local minimum.  Only the
grid that resolves the vortex knows where the vortex is.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import gpuwm.core.storm_tracking as st
from gpuwm.core.storm_tracking import FollowConfig, NestFootprint, StormTracker


def _tree():
    """The Melissa shape: mover d02 at ratio 3, refine grid d03 at ratio 4."""
    d03 = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=3, i_parent_start=83, j_parent_start=83,
                            parent_grid_ratio=4),
        children=[], state=SimpleNamespace(), _started=True)
    d02 = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=2, i_parent_start=102, j_parent_start=31,
                            parent_grid_ratio=3),
        children=[d03], state=SimpleNamespace(), _started=True)
    return d02, d03


def _footprint():
    return NestFootprint(grid_id=2, i_parent_start=102, j_parent_start=31,
                         child_nx=240, child_ny=240, parent_grid_ratio=3)


def _bowl(ny, nx, cj, ci, depth=60.0, width=6.0):
    """A 850 hPa height plane with one clean bowl centred at (cj, ci)."""
    j, i = np.mgrid[0:ny, 0:nx]
    r2 = (j - cj) ** 2 + (i - ci) ** 2
    return (1500.0 - depth * np.exp(-r2 / (2.0 * width ** 2))).astype(np.float64)


def _config(**over):
    base = dict(field="pressure", threshold=30.0, level_hpa=850.0,
                search_margin_cells=12, min_shift_cells=1,
                max_shift_cells=4, cooldown_seconds=0.0, refine_grid_id=3)
    base.update(over)
    return FollowConfig(**base)


# ---------------------------------------------------------------------------
# The map back to parent cells
# ---------------------------------------------------------------------------

def test_the_map_composes_the_placement_chain_exactly():
    """Two independent routes to one parent coordinate must agree.

    The affine map this builds, against NestFootprint's own convention
    applied twice up the chain.  If they disagree the refined centre is a
    correct answer applied at the wrong place, which is worse than no
    refinement at all.
    """
    d02, _ = _tree()
    r = st.refinement_from_node(d02, 3)
    assert r.scale_i == pytest.approx(1.0 / 12.0)
    assert r.scale_j == pytest.approx(1.0 / 12.0)
    centre_in_d02 = (83 - 1) + ((300 - 1) / 4) / 2.0
    ci, cj = r.to_parent(149.5, 149.5)
    assert ci == pytest.approx((102 - 1) + centre_in_d02 / 3.0)
    assert cj == pytest.approx((31 - 1) + centre_in_d02 / 3.0)
    back_i, back_j = r.from_parent(ci, cj)
    assert back_i == pytest.approx(149.5)
    assert back_j == pytest.approx(149.5)


def test_the_map_declines_when_the_grid_cannot_be_used():
    """Every decline is a reason to keep stage one, not to refuse.

    A coarse centre is still a centre; a nest that stops moving because
    its refine grid has not started yet is worse than one that moves
    imprecisely.
    """
    d02, d03 = _tree()
    assert st.refinement_from_node(d02, 9) is None       # not a descendant
    d03._started = False
    assert st.refinement_from_node(d02, 3) is None       # not started
    d03._started, d03.state = True, None
    assert st.refinement_from_node(d02, 3) is None       # no state yet


# ---------------------------------------------------------------------------
# The refinement itself
# ---------------------------------------------------------------------------

def test_stage_two_overrides_a_wrong_coarse_centre(monkeypatch):
    """The parent says move; the resolving grid says the nest is centred."""
    d02, _ = _tree()
    fp = _footprint()
    centre_i, centre_j = fp.center_parent_ij
    planes = iter((_bowl(200, 220, centre_j, centre_i - 3.0),   # parent
                   _bowl(300, 300, 149.5, 149.5)))              # d03
    monkeypatch.setattr(st, "_plane_from_state",
                        lambda state, field, **kw: next(planes))

    tracker = StormTracker(_config())
    shift = tracker.desired_shift(SimpleNamespace(), fp, 0.0,
                                  refinement=st.refinement_from_node(d02, 3))

    ref = tracker.receipts[-1]["refinement"]
    assert ref["applied"] and ref["grid_id"] == 3
    assert ref["refined_centroid_parent_ij"][0] == pytest.approx(centre_i, abs=0.05)
    # Stage one wanted three cells west; the refined centre says hold.
    assert ref["correction_parent_cells"][0] == pytest.approx(3.0, abs=0.1)
    assert shift is None, "a correctly centred nest must not move"


def test_stage_two_still_moves_when_the_storm_really_has_left(monkeypatch):
    """Refinement is not a brake: a real displacement still proposes."""
    d02, _ = _tree()
    fp = _footprint()
    centre_i, centre_j = fp.center_parent_ij
    # BOTH agree the vortex is well east; d03 puts it 36 of its own cells
    # (3 parent cells) east of its centre.
    planes = iter((_bowl(200, 220, centre_j, centre_i + 3.0),
                   _bowl(300, 300, 149.5, 149.5 + 36.0)))
    monkeypatch.setattr(st, "_plane_from_state",
                        lambda state, field, **kw: next(planes))
    tracker = StormTracker(_config())
    shift = tracker.desired_shift(SimpleNamespace(), fp, 0.0,
                                  refinement=st.refinement_from_node(d02, 3))
    assert shift == (3, 0)
    assert tracker.receipts[-1]["refinement"]["applied"]


def test_stage_two_declines_at_the_refine_grid_edge(monkeypatch):
    """Storm leaving d03: only the parent can still see where it goes.

    Refining onto a centre pinned against the boundary would follow the
    edge of the domain rather than the storm.
    """
    d02, _ = _tree()
    fp = _footprint()
    centre_i, centre_j = fp.center_parent_ij
    planes = iter((_bowl(200, 220, centre_j, centre_i - 3.0),
                   _bowl(300, 300, 149.5, 2.0)))     # hard against the edge
    monkeypatch.setattr(st, "_plane_from_state",
                        lambda state, field, **kw: next(planes))
    tracker = StormTracker(_config())
    shift = tracker.desired_shift(SimpleNamespace(), fp, 0.0,
                                  refinement=st.refinement_from_node(d02, 3))
    ref = tracker.receipts[-1]["refinement"]
    assert not ref["applied"] and "edge" in ref["declined"]
    assert shift == (-3, 0), "stage one's answer stands when stage two declines"


def test_stage_two_declines_when_the_signal_is_unavailable(monkeypatch):
    """A refine grid with no readable plane keeps stage one, loudly."""
    d02, _ = _tree()
    fp = _footprint()
    centre_i, centre_j = fp.center_parent_ij
    calls = {"n": 0}

    def plane(state, field, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _bowl(200, 220, centre_j, centre_i - 3.0)
        raise KeyError("refl_10cm")

    monkeypatch.setattr(st, "_plane_from_state", plane)
    tracker = StormTracker(_config())
    shift = tracker.desired_shift(SimpleNamespace(), fp, 0.0,
                                  refinement=st.refinement_from_node(d02, 3))
    ref = tracker.receipts[-1]["refinement"]
    assert not ref["applied"] and "signal unavailable" in ref["declined"]
    assert shift == (-3, 0)


def test_a_missing_refinement_source_is_recorded_not_silent(monkeypatch):
    """Configured but not supplied: the receipt says so."""
    fp = _footprint()
    centre_i, centre_j = fp.center_parent_ij
    monkeypatch.setattr(
        st, "_plane_from_state",
        lambda state, field, **kw: _bowl(200, 220, centre_j, centre_i - 3.0))
    tracker = StormTracker(_config())
    tracker.desired_shift(SimpleNamespace(), fp, 0.0)
    ref = tracker.receipts[-1]["refinement"]
    assert not ref["applied"] and "no refinement source" in ref["declined"]


def test_the_wrong_grid_refuses_rather_than_refining(monkeypatch):
    """Handed grid 4 when configured for 3, it refuses by name."""
    d02, _ = _tree()
    fp = _footprint()
    centre_i, centre_j = fp.center_parent_ij
    monkeypatch.setattr(
        st, "_plane_from_state",
        lambda state, field, **kw: _bowl(200, 220, centre_j, centre_i - 3.0))
    good = st.refinement_from_node(d02, 3)
    wrong = st.RefinementSource(
        grid_id=4, state=good.state, origin_i=good.origin_i,
        origin_j=good.origin_j, scale_i=good.scale_i, scale_j=good.scale_j)
    with pytest.raises(ValueError, match="refines on the grid its config names"):
        StormTracker(_config()).desired_shift(SimpleNamespace(), fp, 0.0,
                                              refinement=wrong)


def test_without_refine_grid_id_the_tracker_is_unchanged(monkeypatch):
    """The default path keeps its exact behaviour AND its exact receipt."""
    fp = _footprint()
    centre_i, centre_j = fp.center_parent_ij
    monkeypatch.setattr(
        st, "_plane_from_state",
        lambda state, field, **kw: _bowl(200, 220, centre_j, centre_i - 3.0))
    tracker = StormTracker(_config(refine_grid_id=None))
    assert tracker.desired_shift(SimpleNamespace(), fp, 0.0) == (-3, 0)
    assert "refinement" not in tracker.receipts[-1]


def test_the_map_degenerates_correctly_when_the_mover_refines_itself():
    """refine_grid_id == the mover's own id: the chain is [mover] alone.

    This is the two-mover shape ([relocation.containment]): d03 is the
    tracked mover, stage one searches its parent d02, and stage two
    refines on d03 itself.  The map must be the mover's own placement.
    """
    _d02, d03 = _tree()
    r = st.refinement_from_node(d03, 3)
    assert r.grid_id == 3
    assert r.scale_i == pytest.approx(1.0 / 4.0)
    assert r.origin_i == pytest.approx(83 - 1)
    ci, cj = r.to_parent(0.0, 0.0)
    assert ci == pytest.approx(82.0) and cj == pytest.approx(82.0)
    back = r.from_parent(*r.to_parent(149.5, 100.25))
    assert back[0] == pytest.approx(149.5)
    assert back[1] == pytest.approx(100.25)
