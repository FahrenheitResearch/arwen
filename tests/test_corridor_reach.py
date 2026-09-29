"""A moving nest's statics corridor covers the ground it can REACH.

Before 2.8 every corridor covered its whole frame at the child's
resolution.  A storm-following 500 m nest under a 2,700 x 3,000 km parent
therefore sealed a 25 GB corridor and its 6 h preparation peaked at
62 GB, although the nest could only get a few hundred kilometres from
where it started.  :mod:`gpuwm.core.nest_reach` bounds how far each mover
can travel (its follow settings, its itinerary, the run length and the
``reach_speed_m_s`` bound) and :func:`gpuwm.static.corridor.corridor_reach`
turns that into the window every child's corridor is built, priced,
surveyed and verified at.

The byte half of the contract -- a window build is the whole-frame
build's own cells -- is in tests/test_statics_corridor.py beside the crop
test it extends.  This file holds the arithmetic, the configuration, the
runner's clamp and the survey.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from gpuwm.core.nest_lifecycle import build_domain_follow_config
from gpuwm.core.nest_reach import (DEFAULT_REACH_SPEED_M_S,
                                   cadence_opportunities, most_moves,
                                   mover_reach, reach_clamp_for,
                                   speed_cells)
from gpuwm.static.corridor import (corridor_reach, planned_corridor,
                                   planned_corridor_cost)

#: The storm-following table a TC tier writes: 850 hPa vortex, moves of
#: 2 to 8 parent cells every 15 min with a 30 min cooldown.
_TC_FOLLOW = {
    "field": "pressure", "threshold": 25.0, "level_hpa": 850.0,
    "radius_km": 60.0, "refine_grid_id": 3, "search_margin_cells": 20,
    "min_shift_cells": 2, "max_shift_cells": 8, "cooldown_seconds": 1800.0,
    "cadence_seconds": 900.0, "max_move_parent_cells": 8,
    "min_overlap_fraction": 0.7}


def _dc(grid_id, parent_id, ip, jp, ratio, nx, ny, dx, *, follow=None,
        spawn=None):
    return SimpleNamespace(
        grid_id=grid_id, parent_id=parent_id, i_parent_start=ip,
        j_parent_start=jp, parent_grid_ratio=ratio, follow=follow,
        spawn=spawn,
        run=SimpleNamespace(nx=nx, ny=ny, dx=dx, dt=45.0,
                            use_adaptive_time_step=False))


_OFF = SimpleNamespace(enabled=False, grid_id=None, follow=None, moves=(),
                       containment=None)


def _exp(domains, run_seconds, relocation=_OFF):
    return SimpleNamespace(domains=tuple(domains), run_seconds=run_seconds,
                           relocation=relocation, root=domains[0])


def _tc_tree(run_seconds=21600.0, **follow_overrides):
    """A 9/3/1/0.5 km storm-following tree the shape of a TC hires tier:
    a 301x335 parent at 9 km, a following 3 km nest at (94, 30), a 1 km
    nest and a 500 m nest riding inside it."""
    follow = build_domain_follow_config(
        {**_TC_FOLLOW, **follow_overrides}, "test", grid_id=2)
    return _exp([
        _dc(1, 0, 1, 1, 1, 301, 335, 9000.0),
        _dc(2, 1, 94, 30, 3, 300, 300, 3000.0, follow=follow),
        _dc(3, 2, 101, 101, 3, 300, 300, 1000.0),
        _dc(4, 3, 76, 76, 2, 300, 300, 500.0),
    ], run_seconds)


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def test_the_opportunities_are_the_cadence_boundaries_inside_the_run():
    assert cadence_opportunities(21600.0, 900.0) == 23
    assert cadence_opportunities(21000.0, 900.0) == 23
    assert cadence_opportunities(900.0, 900.0) == 0
    # A cooldown of two cadences leaves a move every other boundary; one
    # that is not a whole number of cadences rounds up to the next.
    assert most_moves(23, 900.0, 1800.0) == 12
    assert most_moves(23, 900.0, 1000.0) == 12
    assert most_moves(23, 900.0, 900.0) == 23
    assert most_moves(23, 900.0, 0.0) == 23
    assert most_moves(0, 900.0, 0.0) == 0
    assert speed_cells(40.0, 21600.0, 9000.0) == 96
    assert speed_cells(40.0, 0.0, 9000.0) == 0


def test_a_follower_reach_is_its_settings_or_the_speed_bound():
    reach = mover_reach(_tc_tree(), 2)
    # 12 moves of 8 cells, plus one move of margin, is 104; 40 m/s for
    # 6 h is 96 cells plus one move, also 104.  The settings win a tie.
    assert reach.basis["moves"] == 12
    assert reach.basis["settings_parent_cells"] == 104
    assert reach.basis["speed_parent_cells"] == 104
    assert reach.basis["bounded_by"] == "follow settings"
    assert reach.basis["speed_source"] == "default"
    assert reach.basis["reach_speed_m_s"] == DEFAULT_REACH_SPEED_M_S
    # Clipped to where a placement can be: d02 starts at j = 30, so it
    # can go 29 cells south and no further.
    assert (reach.lo_i, reach.hi_i, reach.lo_j, reach.hi_j) == (
        -93, 104, -29, 104)
    # Allowed a move every cadence for five days, the settings reach
    # 3,840 cells and the speed bound is what holds.  (At the TC table's
    # own 8 cells per 30 min the settings ARE 40 m/s, so the two tie.)
    long = mover_reach(_tc_tree(run_seconds=432000.0,
                                cooldown_seconds=900.0), 2)
    assert long.basis["settings_parent_cells"] == 480 * 8
    assert long.basis["bounded_by"] == "reach_speed_m_s"
    assert long.basis["speed_parent_cells"] == 1920 + 8
    slow = mover_reach(_tc_tree(reach_speed_m_s=10.0), 2)
    assert slow.basis["speed_source"] == "configured"
    assert slow.basis["reach_parent_cells"] == 24 + 8


def test_an_itinerary_reaches_exactly_where_its_rows_go():
    moves = (SimpleNamespace(at_seconds=900.0, di_parent_cells=5,
                             dj_parent_cells=-2),
             SimpleNamespace(at_seconds=1800.0, di_parent_cells=-12,
                             dj_parent_cells=0),
             # Past the run's end: never fires, never counted.
             SimpleNamespace(at_seconds=9000.0, di_parent_cells=40,
                             dj_parent_cells=40))
    relocation = SimpleNamespace(
        enabled=True, grid_id=2, follow=None, moves=moves,
        containment=None, max_move_parent_cells=8)
    exp = _exp([_dc(1, 0, 1, 1, 1, 100, 100, 9000.0),
                _dc(2, 1, 40, 40, 3, 60, 60, 3000.0)], 3600.0, relocation)
    reach = mover_reach(exp, 2)
    # The -12 row clamps to 8 at the runner, so it can take the nest 8.
    assert (reach.lo_i, reach.hi_i, reach.lo_j, reach.hi_j) == (-8, 5, -2, 0)
    assert reach.basis["bounded_by"] == "itinerary"
    assert reach_clamp_for(exp, 2) is None


def test_a_dormant_nest_reaches_its_whole_frame_and_says_why():
    follow = build_domain_follow_config(dict(_TC_FOLLOW), "test", grid_id=2)
    exp = _exp([_dc(1, 0, 1, 1, 1, 100, 100, 9000.0),
                _dc(2, 1, 40, 40, 3, 60, 60, 3000.0, follow=follow,
                    spawn=object())], 3600.0)
    record = corridor_reach(exp, exp.domains[1])
    assert record["whole_frame"] is True
    assert "dormant" in record["unbounded"]
    assert reach_clamp_for(exp, 2) is None


# ---------------------------------------------------------------------------
# THE CORRIDOR IS THE REACH: sized, priced and verified at one window
# ---------------------------------------------------------------------------

def test_the_corridor_of_a_bounded_nest_is_its_reachable_region():
    """Before 2.8 each of these was the whole frame: 903x1005 cells for
    d02, 2709x3015 for d03 and 5418x6030 for d04."""
    exp = _tc_tree()
    d2, d3, d4 = exp.domains[1:]
    # d02 is framed on its parent; its window is its footprint widened
    # by (-93..104, -29..104) parent cells of 3 child cells each.
    assert corridor_reach(exp, d2)["window_child_cells"] == [0, 0, 891, 699]
    # d03 and d04 ride inside d02, so they are framed on the root and
    # widened by d02's range in their own cells (9 and 18 per d01 cell).
    assert corridor_reach(exp, d3)["window_child_cells"] == [
        300, 300, 2073, 1497]
    assert corridor_reach(exp, d4)["window_child_cells"] == [
        750, 750, 3846, 2694]
    whole = {2: 903 * 1005, 3: 2709 * 3015, 4: 5418 * 6030}
    for dc in (d2, d3, d4):
        cost = planned_corridor_cost(exp, dc)
        assert cost["whole_frame"] is False
        assert cost["cells"] < whole[int(dc.grid_id)]
        plan = planned_corridor(exp, dc)
        assert plan.geometry["corridor_nx"] == plan.window[2]
        assert plan.geometry["window_origin_child_cells"] == list(
            plan.window[:2])
    # The 500 m corridor is under a third of the frame it used to cover.
    assert planned_corridor_cost(exp, d4)["cells"] == 3846 * 2694
    assert 3846 * 2694 / whole[4] < 0.32


def test_the_declared_footprint_is_always_inside_the_window():
    exp = _tc_tree(run_seconds=900.0)
    for dc in exp.domains[1:]:
        plan = planned_corridor(exp, dc)
        ox, oy = plan.geometry["reference_origin_child_cells"]
        x0, y0, nx, ny = plan.window
        assert x0 <= ox and ox + dc.run.nx <= x0 + nx
        assert y0 <= oy and oy + dc.run.ny <= y0 + ny


def test_a_still_child_gets_its_own_footprint():
    """Bare --statics-corridor seals every child.  One that never moves
    and rides on nothing that moves used to get the whole frame too."""
    exp = _exp([_dc(1, 0, 1, 1, 1, 100, 100, 9000.0),
                _dc(2, 1, 40, 30, 3, 60, 45, 3000.0)], 3600.0)
    record = corridor_reach(exp, exp.domains[1])
    assert record["window_child_cells"] == [117, 87, 60, 45]
    assert record["movers"] == []


def test_a_long_run_reaches_the_whole_frame_and_the_receipt_says_so():
    exp = _tc_tree(run_seconds=432000.0, cooldown_seconds=900.0)
    record = corridor_reach(exp, exp.domains[1])
    assert record["whole_frame"] is True
    assert record["window_child_cells"] == [0, 0, 903, 1005]
    mover, = record["movers"]
    assert mover["bounded_by"] == "reach_speed_m_s"
    # The 500 m nest rides 41 parent cells in from d02's west edge and 41
    # up from its south edge, and d02 cannot leave its parent, so even
    # five days never take the 500 m nest to those strips.
    assert corridor_reach(exp, exp.domains[3])["window_child_cells"][:2] == [
        750, 750]
    # A slower declared bound is what narrows the rest again.
    slow = _tc_tree(run_seconds=432000.0, reach_speed_m_s=1.0)
    assert corridor_reach(slow, slow.domains[1])["whole_frame"] is False


# ---------------------------------------------------------------------------
# The runner holds the nest to the bound its corridor was sized for
# ---------------------------------------------------------------------------

def _node(ip, jp, *, parent=None):
    return SimpleNamespace(cfg=SimpleNamespace(
        grid_id=2, i_parent_start=ip, j_parent_start=jp), parent=parent)


def test_the_clamp_holds_the_nest_inside_its_allowance():
    exp = _tc_tree(reach_speed_m_s=5.0)
    clamp = reach_clamp_for(exp, 2)
    # 5 m/s for 1800 s is 1 cell of 9 km, plus one 8-cell move.
    assert clamp.allowance(1800.0) == 9
    # From the declared start, 8 east is inside, 12 is clipped to 9.
    assert clamp.bound(_node(94, 30), (8, 0), 1800.0) == (8, 0, False)
    assert clamp.bound(_node(94, 30), (12, -3), 1800.0) == (9, -3, True)
    # Already 7 east: only 2 more.
    assert clamp.bound(_node(101, 30), (8, 8), 1800.0) == (2, 8, True)
    # Never dragged back: past the allowance (a restart under a smaller
    # bound) the nest holds rather than moving toward its start.
    assert clamp.bound(_node(120, 30), (4, 0), 1800.0) == (0, 0, True)
    assert clamp.bound(_node(120, 30), (-4, 0), 1800.0) == (-4, 0, False)


@dataclass(frozen=True)
class _Placement:
    """The DomainConfig fields the runner's clamp walk reads."""

    grid_id: int
    i_parent_start: int
    j_parent_start: int
    parent_grid_ratio: int
    run: object


def test_the_runner_names_the_reach_when_it_clamps(monkeypatch):
    import gpuwm.core.nest_relocation as nest_relocation
    from gpuwm.core.relocation_runner import RelocationRunner

    # The parent-edge walk is not under test here: every placement the
    # clamps leave is admissible in this parent.
    monkeypatch.setattr(nest_relocation, "_prevalidate_placement",
                        lambda *args, **kwargs: None)
    runner = RelocationRunner.__new__(RelocationRunner)
    runner.config = SimpleNamespace(max_move_parent_cells=8,
                                    min_overlap_fraction=None)
    runner.reach_clamp = reach_clamp_for(_tc_tree(reach_speed_m_s=1.0), 2)
    run = SimpleNamespace(nx=300, ny=300)

    def node(ip):
        return SimpleNamespace(cfg=_Placement(2, ip, 30, 3, run), parent=None)

    # 1 m/s for 900 s is 1 cell, plus one move of margin: 9 from the
    # start.  From the start, max_move_parent_cells is what binds.
    assert runner._clamped_shift(node(94), (12, 0), 900.0) == (
        8, 0, ["max_move_parent_cells"])
    # Six cells out, only three more: the reach binds, and says so.
    assert runner._clamped_shift(node(100), (8, 0), 900.0) == (
        3, 0, ["reach_speed_m_s"])
    # Later in the run the allowance has grown with the time.
    assert runner._clamped_shift(node(100), (8, 0), 72000.0) == (8, 0, [])
    # No clamp object (a scripted itinerary): nothing but the old bounds.
    runner.reach_clamp = None
    assert runner._clamped_shift(node(100), (8, 0), 900.0) == (8, 0, [])


def test_the_containment_mover_counts_its_ground_not_its_compensation():
    """Under [relocation.containment] a slide moves the tracked mover's
    placement by -slide x ratio and leaves it where it was on the
    ground; the clamp must not count that as travel."""
    follow = build_domain_follow_config(dict(_TC_FOLLOW), "test", grid_id=3)
    tracker = follow.tracker
    relocation = SimpleNamespace(
        enabled=True, grid_id=3, follow=tracker, moves=(),
        cadence_seconds=900.0, max_move_parent_cells=8,
        reach_speed_m_s=None,
        containment=SimpleNamespace(grid_id=2, max_move_parent_cells=1,
                                    cadence_seconds=None, deadband_cells=8))
    exp = _exp([_dc(1, 0, 1, 1, 1, 200, 200, 9000.0),
                _dc(2, 1, 60, 60, 3, 150, 150, 3000.0),
                _dc(3, 2, 50, 50, 3, 150, 150, 1000.0)], 21600.0, relocation)
    clamp = reach_clamp_for(exp, 3)
    assert clamp.ancestor_grid_id == 2
    ancestor = SimpleNamespace(cfg=SimpleNamespace(
        grid_id=2, i_parent_start=62, j_parent_start=60))
    # d02 slid 2 east; d03 was compensated 6 west: no ground travel.
    moved = SimpleNamespace(cfg=SimpleNamespace(
        grid_id=3, i_parent_start=44, j_parent_start=50), parent=ancestor)
    assert clamp.displacement(moved) == (0, 0)
    # The slide is bounded by its own settings, not the speed clamp.
    assert reach_clamp_for(exp, 2) is None
    slide = mover_reach(exp, 2)
    assert slide.basis["bounded_by"] == "containment settings"
    assert slide.basis["reach_parent_cells"] == 24
    # d03's corridor ignores the slide (earth-fixed) and takes its own
    # tracker reach over the ground.
    record = corridor_reach(exp, exp.domains[2])
    assert [m["grid_id"] for m in record["movers"]] == [3]


# ---------------------------------------------------------------------------
# The configuration surface
# ---------------------------------------------------------------------------

def test_reach_speed_is_validated_where_it_is_declared():
    with pytest.raises(ValueError, match="finite, positive speed"):
        build_domain_follow_config({**_TC_FOLLOW, "reach_speed_m_s": 0.0},
                                   "test", grid_id=2)
    with pytest.raises(ValueError, match="metres per second"):
        build_domain_follow_config({**_TC_FOLLOW, "reach_speed_m_s": "fast"},
                                   "test", grid_id=2)
    follow = build_domain_follow_config(
        {**_TC_FOLLOW, "reach_speed_m_s": 12}, "test", grid_id=2)
    assert follow.reach_speed_m_s == 12.0
    assert follow.to_json()["reach_speed_m_s"] == 12.0
    assert "reach_speed_m_s" not in build_domain_follow_config(
        dict(_TC_FOLLOW), "test", grid_id=2).to_json()


def test_an_itinerary_refuses_a_speed_bound_it_would_ignore():
    from gpuwm.experiment import RelocationConfig, ScheduledRelocationMove

    with pytest.raises(ValueError, match="would bound nothing"):
        RelocationConfig(enabled=True, grid_id=2, reach_speed_m_s=20.0,
                         moves=(ScheduledRelocationMove(at_seconds=90.0,
                                                        di_parent_cells=1),))
    with pytest.raises(ValueError, match="runs none"):
        RelocationConfig(enabled=False, reach_speed_m_s=20.0)


def test_the_default_bound_leaves_every_fingerprint_where_it_was(tmp_path):
    """Absent, the key is absent from the restart identity (so every
    checkpoint written before it existed still resumes); declared, it
    binds, because it decides where the nest may go."""
    from gpuwm.core.model import restart_identity_payload
    from gpuwm.experiment import load_experiment

    base = _follow_config(tmp_path, "")
    declared = _follow_config(tmp_path, ", reach_speed_m_s = 15.0")
    plain = restart_identity_payload(load_experiment(base))
    bound = restart_identity_payload(load_experiment(declared))
    follow = next(d for d in plain["domains"] if d.get("follow"))["follow"]
    assert "reach_speed_m_s" not in follow
    follow = next(d for d in bound["domains"] if d.get("follow"))["follow"]
    assert follow["reach_speed_m_s"] == 15.0


def _follow_config(tmp_path, extra: str):
    from gpuwm.cli import main as cli_main
    from gpuwm.experiment import load_experiment

    base = tmp_path / "base.toml"
    if not base.exists():
        assert cli_main([
            "domain", "--point=35.3,-97.5", "--card", "24gb", "--ladder",
            "12", "--source", "gfs", "--cycle", "2026-07-29T18", "--hours",
            "6", "--physics-profile",
            "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1",
            "--out", str(base)]) == 0
    # On the root's own step lattice, so the cadence can fire.
    dt = float(load_experiment(base).root.run.dt)
    follow = (
        'field = "pressure", threshold = 25.0, level_hpa = 850.0, '
        'radius_km = 60.0, search_margin_cells = 6, min_shift_cells = 2, '
        f"max_shift_cells = 3, cooldown_seconds = {dt * 20:.1f}, "
        f"cadence_seconds = {dt * 10:.1f}")
    text = base.read_text(encoding="utf-8") + f"""
[[domain]]
grid_id = 2
parent_id = 1
i_parent_start = 30
j_parent_start = 30
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = 45
ny = 45
history_interval_s = 3600.0
follow = {{ {follow}{extra} }}
"""
    out = tmp_path / f"follow{len(extra)}.toml"
    out.write_text(text, encoding="utf-8")
    out.with_suffix(".namelist.wps").write_bytes(
        base.with_suffix(".namelist.wps").read_bytes())
    return out


# ---------------------------------------------------------------------------
# The terrain survey reads the corridor's own ground
# ---------------------------------------------------------------------------

def test_the_survey_reads_the_corridor_on_the_childs_lattice(monkeypatch):
    """The survey that derives the hybrid coordinate reads every terrain
    the run can touch, a following nest's corridor among them.  It used
    to translate the FRAME's grid by the corridor's child-cell offsets,
    so it read parent-resolution terrain over ratio times the frame's
    extent on each axis; it now reads the corridor's own window on the
    child's grid.
    """
    import numpy as np

    import gpuwm.static.build as build_module
    import gpuwm.vertical_adaptation as survey
    from gpuwm.static.lambert import LambertGrid

    exp = _tc_tree(run_seconds=3600.0)
    exp.domains[0].run.base_temp = 290.0
    for dc in exp.domains:
        dc.run.base_temp = 290.0
        dc.start_time = None
    parent = LambertGrid(ref_lat=20.0, ref_lon=-110.0, truelat1=18.0,
                         truelat2=34.0, stand_lon=-110.0, dx=9000.0,
                         dy=9000.0, e_we=302, e_sn=336)
    d2 = parent.nest(94, 30, 3, 301, 301)
    d3 = d2.nest(101, 101, 3, 301, 301)
    d4 = d3.nest(76, 76, 2, 301, 301)
    grids = [parent, d2, d3, d4]
    seen = []

    def fake_terrain(grid, root, selection=None):
        seen.append(grid)
        return np.zeros((int(grid.e_sn) - 1, int(grid.e_we) - 1))

    monkeypatch.setattr(build_module, "build_terrain", fake_terrain)
    monkeypatch.setattr(build_module, "geog_selection_from_catalog",
                        lambda catalog, gid: SimpleNamespace(root=None))
    fields = survey.run_terrain_fields(
        exp, grids, root_terrain=np.zeros((335, 301)),
        static_catalog=object())
    corridor = [f for f in fields if f.label == "d02 statics corridor"]
    assert len(corridor) == 1
    grid = seen[-1]
    plan = planned_corridor(exp, exp.domains[1])
    assert float(grid.dx) == 3000.0
    assert (int(grid.e_we) - 1, int(grid.e_sn) - 1) == plan.window[2:]
    # Its first cell is the window's first cell of d02's own lattice.
    x0, y0 = plan.window[:2]
    ox, oy = plan.geometry["reference_origin_child_cells"]
    lat, lon = grid.ij_to_latlon(1.0, 1.0)
    lat_ref, lon_ref = d2.ij_to_latlon(1.0 + x0 - ox, 1.0 + y0 - oy)
    assert (float(lat), float(lon)) == (float(lat_ref), float(lon_ref))
