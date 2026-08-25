"""The radius bound: why the centroid is a vortex centre and not a
domain average.

THE DEFECT THIS CLOSES, measured on the 2025-10-24 12Z Melissa run.
``[relocation.follow]`` was ``field = "pressure"``, ``level_hpa = 850``,
``threshold = 30`` -- 30 metres of geopotential height above the search
box's own minimum, which the config docstring calls "a tropical-cyclone
inner core".  On a large domain that is true.  On d03 it was not: the
850 hPa height field across that 243 km nest spans only **41-49 m in
total**, so a 30 m threshold selected **44-65% of the entire grid**, and
on most frames that region was **clipped by the domain edge**.

Two things follow, and the second is the serious one:

1. the centroid was a weighted average of half the nest rather than of
   the vortex -- 10.68 km from the field's own minimum, worst frame
   21.78 km;
2. because the region hit the edge, the answer depended on where the
   grid stopped.  The nest's placement became an input to the centre
   that steers the nest: a feedback loop.

Bounding the same threshold to a disc around the extremum fixed both.
Measured through the shipped code path on 21 d03 frames:

    unbounded      offset 10.68 km (max 21.78)   jitter 7.83 km
    radius 50 km   offset  1.04 km (max  2.04)   jitter 5.74 km

10.2x closer, and it jitters LESS frame to frame than the bare field
minimum does (6.02 km), so the bound is not buying closeness with noise.
"""

from __future__ import annotations

import numpy as np
import pytest

import gpuwm.core.storm_tracking as st
from gpuwm.core.storm_tracking import (DEFAULT_CENTROID_RADIUS_KM,
                                       FollowConfig, NestFootprint,
                                       build_follow_config, locate_signal,
                                       radius_in_cells, weighted_centroid)


def _base(**over):
    base = dict(field="pressure", threshold=30.0, level_hpa=850.0,
                search_margin_cells=12, min_shift_cells=1,
                max_shift_cells=2, cooldown_seconds=360.0)
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# The conversion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("km,dx_m,cells", [
    (50.0, 642.8571428571429, 77.77777777777779),
    (50.0, 4500.0, 11.11111111111111),
    (50.0, 13500.0, 3.7037037037037037),
    (25.0, 1000.0, 25.0),
])
def test_a_radius_in_km_becomes_cells_on_each_grid(km, dx_m, cells):
    """The same physical radius is a different cell count per domain,
    which is the whole reason the key is in kilometres."""
    assert radius_in_cells(km, dx_m) == pytest.approx(cells)


@pytest.mark.parametrize("dx", [None, 0.0, -1.0, float("nan")])
def test_an_unusable_spacing_gives_no_bound_rather_than_a_wrong_one(dx):
    """A hand-built footprint or a test double has no dx; the centroid is
    then unbounded, which is the behaviour that predates the bound.
    Config cannot reach this -- every loaded domain has a resolved dx."""
    assert radius_in_cells(50.0, dx) is None


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------

def _bowl(ny, nx, cj, ci, depth=40.0, width=25.0, slope=0.10,
          floor=1500.0):
    """A vortex sitting in an environmental gradient.

    THE SLOPE IS THE POINT.  A perfectly symmetric bowl is the one case
    where the unbounded centroid is already exact -- the bias cancels --
    so a fixture without it cannot reproduce the defect and cannot test
    the fix.  A real 850 hPa height field slopes across the domain, and
    on Melissa's d03 that slope was comparable to the vortex's own depth
    (the field spanned 41-49 m against a 30 m threshold).  Here the span
    is ~56 m against a 40 m vortex, the same regime.
    """
    j, i = np.mgrid[0:ny, 0:nx]
    r2 = (j - cj) ** 2 + (i - ci) ** 2
    return (floor - depth * np.exp(-r2 / (2.0 * width ** 2))
            + slope * i + 0.6 * slope * j).astype(np.float64)


def test_the_regression_itself_a_threshold_wider_than_the_field():
    """The Melissa shape: a threshold comparable to the field's whole
    range, with the vortex in a gradient, so the qualifying region is
    large and asymmetric and drags the centroid down-gradient."""
    plane = _bowl(200, 200, 100.0, 100.0)
    box = (slice(0, 200), slice(0, 200))
    assert float(np.nanmax(plane) - np.nanmin(plane)) < 60.0

    unbounded = locate_signal(plane, "pressure", 30.0, box,
                              relative_to_minimum=True)
    bounded = locate_signal(plane, "pressure", 30.0, box,
                            relative_to_minimum=True, radius_cells=20.0)
    assert unbounded["cells"] > 0.20 * plane.size
    assert bounded["cells"] < unbounded["cells"]
    off_unbounded = np.hypot(unbounded["ci"] - 100.0, unbounded["cj"] - 100.0)
    off_bounded = np.hypot(bounded["ci"] - 100.0, bounded["cj"] - 100.0)
    # Measured on this fixture: 13.80 cells unbounded, 2.24 bounded.
    assert off_unbounded > 10.0
    assert off_bounded < 3.0
    assert off_bounded < off_unbounded / 4.0


def test_the_disc_is_anchored_on_the_extremum_not_on_the_centroid():
    """A centroid already dragged down-gradient would carry the disc with
    it, and the bound would be as biased as the thing it bounds.  Anchored
    on the extremum, the bounded answer sits near the true minimum even
    though the unbounded one does not."""
    plane = _bowl(200, 200, 100.0, 100.0)
    box = (slice(0, 200), slice(0, 200))
    drifted = locate_signal(plane, "pressure", 30.0, box,
                            relative_to_minimum=True)
    got = locate_signal(plane, "pressure", 30.0, box,
                        relative_to_minimum=True, radius_cells=20.0)
    assert np.hypot(drifted["ci"] - 100.0, drifted["cj"] - 100.0) > 10.0
    assert got["ci"] == pytest.approx(100.0, abs=3.0)
    assert got["cj"] == pytest.approx(100.0, abs=3.0)


def test_a_smaller_radius_sits_closer_to_the_extremum():
    """Monotone, which is what makes the knob predictable: tightening it
    always moves the answer toward the minimum, never past it."""
    plane = _bowl(200, 200, 100.0, 100.0)
    box = (slice(0, 200), slice(0, 200))
    offs = []
    for radius in (150.0, 80.0, 40.0, 20.0):
        got = locate_signal(plane, "pressure", 30.0, box,
                            relative_to_minimum=True, radius_cells=radius)
        offs.append(np.hypot(got["ci"] - 100.0, got["cj"] - 100.0))
    assert offs == sorted(offs, reverse=True), offs


def test_none_restores_the_unbounded_behaviour_exactly():
    """The pre-bound answer stays reachable and stays identical, so the
    measurement that justified the bound can always be re-run."""
    plane = _bowl(120, 120, 40.0, 70.0)
    box = (slice(0, 120), slice(0, 120))
    a = weighted_centroid(plane, 1480.0, box)
    b = weighted_centroid(plane, 1480.0, box, None)
    assert a == b


def test_a_radius_larger_than_the_grid_changes_nothing():
    plane = _bowl(80, 80, 40.0, 40.0)
    box = (slice(0, 80), slice(0, 80))
    wide = weighted_centroid(plane, 1480.0, box, 10_000.0)
    none = weighted_centroid(plane, 1480.0, box)
    # Same answer; the bounded call also reports its convergence, which
    # the unbounded single pass has nothing to say about.
    assert (wide["ci"], wide["cj"], wide["cells"]) ==         (none["ci"], none["cj"], none["cells"])
    assert wide["converged"] is True


def test_a_disc_clipped_by_the_grid_edge_still_answers_and_still_helps():
    """The storm at a corner: the disc runs off the domain, so only the
    cells that exist vote and the answer is still pulled inward -- a
    clipped disc cannot be symmetric.  It must still ANSWER, and still
    beat the unbounded region.  This is exactly the case the refinement
    stage declines on (``edge_margin_cells``), so the tracker never
    relies on it."""
    plane = _bowl(100, 100, 4.0, 4.0)
    box = (slice(0, 100), slice(0, 100))
    unbounded = locate_signal(plane, "pressure", 30.0, box,
                              relative_to_minimum=True)
    got = locate_signal(plane, "pressure", 30.0, box,
                        relative_to_minimum=True, radius_cells=20.0)
    assert got is not None
    off_un = np.hypot(unbounded["ci"] - 4.0, unbounded["cj"] - 4.0)
    off_got = np.hypot(got["ci"] - 4.0, got["cj"] - 4.0)
    assert off_got < off_un


def test_a_single_qualifying_cell_is_returned_unchanged():
    """cells == 1 skips the disc entirely: there is nothing to bound."""
    plane = np.zeros((20, 20))
    plane[7, 11] = 5.0
    got = weighted_centroid(plane, 1.0, (slice(0, 20), slice(0, 20)), 3.0)
    assert (got["ci"], got["cj"], got["cells"]) == (11.0, 7.0, 1)


def test_the_bound_applies_to_maximum_signals_too():
    """uh and reflectivity are maxima, and a 40 dBZ contour spanning a
    whole domain has the same failure mode as a 30 m one."""
    plane = np.full((100, 100), 45.0)
    plane[20:24, 20:24] = 60.0
    box = (slice(0, 100), slice(0, 100))
    wide = weighted_centroid(plane, 40.0, box)
    tight = weighted_centroid(plane, 40.0, box, 10.0)
    assert tight["cells"] < wide["cells"]
    assert np.hypot(tight["ci"] - 21.5, tight["cj"] - 21.5) < \
        np.hypot(wide["ci"] - 21.5, wide["cj"] - 21.5)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_the_default_is_a_real_radius_not_unbounded():
    """Absent must not mean 'no disc': no disc is the defect."""
    cfg = build_follow_config(_base(), "x")
    assert cfg.radius_km == DEFAULT_CENTROID_RADIUS_KM == 50.0
    assert cfg.to_json()["radius_km"] == 50.0


def test_an_explicit_radius_is_honoured_and_echoed():
    cfg = build_follow_config(_base(radius_km=25.0), "x")
    assert cfg.radius_km == 25.0
    assert cfg.to_json()["radius_km"] == 25.0


@pytest.mark.parametrize("bad", [0.0, 0.5, 501.0, -10.0])
def test_a_radius_outside_the_band_refuses_with_the_units(bad):
    with pytest.raises(ValueError, match="radius_km"):
        build_follow_config(_base(radius_km=bad), "x")
    with pytest.raises(ValueError, match="km"):
        build_follow_config(_base(radius_km=bad), "x")


def test_a_non_numeric_radius_refuses():
    with pytest.raises(ValueError, match="must be a number"):
        build_follow_config(_base(radius_km="50"), "x")


def test_radius_km_is_an_accepted_follow_key():
    assert "radius_km" in st.FOLLOW_KEYS
    # ...and is not in the required set, so every existing config loads.
    cfg = build_follow_config(_base(), "x")
    assert cfg.radius_km == DEFAULT_CENTROID_RADIUS_KM


def test_the_radius_applies_to_every_field():
    """uh and reflectivity get the bound too; the failure mode is the
    same shape whatever the units."""
    for extra in ({"field": "uh", "threshold": 25.0,
                   "fallback_threshold": 40.0, "level_hpa": None},
                  {"field": "reflectivity", "threshold": 40.0,
                   "level_hpa": None}):
        table = _base(**extra)
        table = {k: v for k, v in table.items() if v is not None}
        assert build_follow_config(table, "x").radius_km == 50.0


# ---------------------------------------------------------------------------
# The spacing reaches the estimator
# ---------------------------------------------------------------------------

def test_a_footprint_derives_the_parents_spacing_from_the_child():
    """child dx x ratio, so there is one source of truth and no caller
    supplies a second."""
    from types import SimpleNamespace
    cfg = SimpleNamespace(grid_id=3, i_parent_start=94, j_parent_start=94,
                          parent_grid_ratio=7,
                          run=SimpleNamespace(nx=378, ny=378,
                                              dx=642.8571428571429))
    fp = NestFootprint.coerce(cfg)
    assert fp.parent_dx_m == pytest.approx(4500.0)
    assert radius_in_cells(50.0, fp.parent_dx_m) == pytest.approx(11.111,
                                                                  abs=1e-3)


def test_a_hand_built_footprint_has_no_spacing_and_is_unbounded():
    fp = NestFootprint(grid_id=2, i_parent_start=10, j_parent_start=10,
                       child_nx=60, child_ny=60, parent_grid_ratio=3)
    assert fp.parent_dx_m is None
    assert radius_in_cells(50.0, fp.parent_dx_m) is None


def test_the_refinement_source_carries_the_refine_grids_own_spacing():
    from types import SimpleNamespace
    d03 = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=3, i_parent_start=94, j_parent_start=94,
                            parent_grid_ratio=7,
                            run=SimpleNamespace(dx=642.8571428571429)),
        children=[], state=SimpleNamespace(), _started=True)
    d02 = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=2, i_parent_start=74, j_parent_start=46,
                            parent_grid_ratio=3,
                            run=SimpleNamespace(dx=4500.0)),
        children=[d03], state=SimpleNamespace(), _started=True)
    src = st.refinement_from_node(d02, 3)
    assert src.dx_m == pytest.approx(642.857, abs=1e-3)
    # 50 km on a 643 m grid is ~78 cells -- the number the Melissa
    # measurement was made at.
    assert radius_in_cells(50.0, src.dx_m) == pytest.approx(77.78, abs=0.01)
