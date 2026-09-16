"""The overlap floor is a BOUND on a move, not a verdict on the run.

``check_admissible`` has always judged two things: how far a move goes in
parent cells, and how much of the child it keeps.  The runner clamped a
proposal to the first and not to the second, so a follow source that
proposed one cell too much ended the forecast instead of making the move
it was allowed to make.

The two bounds are also not independent.  Overlap is separable -- a shift
of ``(m, n)`` parent cells keeps ``(1 - m*r/nx) * (1 - n*r/ny)`` of the
child -- so the binding case is the DIAGONAL move, and a floor ``f``
admits a per-axis magnitude of only ``1 - sqrt(f)`` of the nest's own
width in parent cells.  A configuration is free to write the two numbers
separately, and the shipped cyclone preset wrote 8 parent cells against a
floor of 0.7 on a nest 40 parent cells wide, whose floor admits 6.  Its
own maximum was therefore unreachable, and a storm that asked for it
ended the run.

What this file pins:

* the arithmetic, on both axes and both signs;
* the derivation, against the numbers every shipped configuration ships;
* the clamp: the largest move in the same direction that clears the
  floor, with the null move as the always-available floor of the walk;
* the receipt: the proposed shift, the applied shift, and the bound that
  cut it;
* the refusal that remains, for a caller that names a placement rather
  than following one.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm.core.nest_relocation import (Placement, RelocationRefusal,
                                        check_admissible,
                                        clamp_shift_to_overlap,
                                        max_parent_cells_for_overlap,
                                        overlap_fraction_for_shift,
                                        plan_relocation)
from gpuwm.core.relocation_runner import RelocationRunner
from gpuwm.experiment import RelocationConfig, ScheduledRelocationMove

from test_nest_relocation_staging import _cpu_tree, _initializer
from test_relocation_runner import (_advance, _containment_runner, _model,
                                    _tree3)

#: The quick-start's own geometry: a 160-cell nest at ratio 4 is 40 parent
#: cells wide, which is also what each slot in the shipped cyclone preset
#: is.  Every number below that mentions 6 comes from it.
CYCLONE = dict(parent_grid_ratio=4, child_nx=160, child_ny=160)


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def test_overlap_is_separable_and_the_diagonal_binds():
    """The number the failed run died on, and the two single-axis moves
    of the same size that would have been fine."""
    assert overlap_fraction_for_shift(8, 8, **CYCLONE) == pytest.approx(0.64)
    assert overlap_fraction_for_shift(8, 0, **CYCLONE) == pytest.approx(0.80)
    assert overlap_fraction_for_shift(0, 8, **CYCLONE) == pytest.approx(0.80)
    assert overlap_fraction_for_shift(6, 6, **CYCLONE) == pytest.approx(0.7225)
    assert overlap_fraction_for_shift(7, 7, **CYCLONE) == pytest.approx(
        0.680625)


def test_overlap_ignores_the_sign_of_the_shift():
    for di, dj in ((8, 8), (-8, 8), (8, -8), (-8, -8)):
        assert overlap_fraction_for_shift(di, dj, **CYCLONE) == pytest.approx(
            0.64)


def test_the_floor_implies_a_per_axis_bound():
    """``1 - sqrt(f)`` of the nest's width in parent cells, floored.

    40 * (1 - sqrt(0.7)) = 6.53, so 6.  The 120-cell nest at ratio 3 in
    the shipped preset is the same 40 parent cells wide and gets the same
    answer, which is the point of expressing it in the nest's own width.
    """
    assert max_parent_cells_for_overlap(0.7, **CYCLONE) == 6
    assert max_parent_cells_for_overlap(
        0.7, parent_grid_ratio=3, child_nx=120, child_ny=120) == 6
    assert max_parent_cells_for_overlap(
        0.5, parent_grid_ratio=3, child_nx=150, child_ny=150) == 14


def test_the_implied_bound_is_exactly_the_largest_admissible_diagonal():
    """Not a rule of thumb beside the check: the same answer, derived."""
    for floor in (0.2, 0.35, 0.5, 0.64, 0.7, 0.8, 0.9, 0.95):
        bound = max_parent_cells_for_overlap(floor, **CYCLONE)
        assert overlap_fraction_for_shift(bound, bound, **CYCLONE) >= floor
        assert overlap_fraction_for_shift(
            bound + 1, bound + 1, **CYCLONE) < floor


def test_the_degenerate_floors_are_stated_rather_than_guessed():
    assert max_parent_cells_for_overlap(None, **CYCLONE) is None
    # A floor that keeps the whole child admits the null move and nothing
    # else, which is a number and not a refusal.
    assert max_parent_cells_for_overlap(1.0, **CYCLONE) == 0


# ---------------------------------------------------------------------------
# The clamp
# ---------------------------------------------------------------------------

def test_a_diagonal_over_the_floor_clamps_to_the_largest_one_under_it():
    assert clamp_shift_to_overlap(
        -8, 8, min_overlap_fraction=0.7, **CYCLONE) == (-6, 6, True)
    assert clamp_shift_to_overlap(
        8, 8, min_overlap_fraction=0.7, **CYCLONE) == (6, 6, True)


def test_a_move_that_already_clears_the_floor_is_untouched():
    assert clamp_shift_to_overlap(
        6, 6, min_overlap_fraction=0.7, **CYCLONE) == (6, 6, False)
    # A single-axis move of the same size keeps 0.80 and is not diagonal,
    # so the diagonal bound does not apply to it and it must not be cut.
    assert clamp_shift_to_overlap(
        8, 0, min_overlap_fraction=0.7, **CYCLONE) == (8, 0, False)


def test_the_clamp_keeps_the_direction_it_was_given():
    """Scaling both axes by one factor, truncated toward zero: a diagonal
    stays diagonal, an axis-aligned move stays on its axis, and a move
    that is mostly north stays mostly north."""
    di, dj, clamped = clamp_shift_to_overlap(
        4, 16, min_overlap_fraction=0.7, **CYCLONE)
    assert clamped is True
    assert di > 0 and dj > 0 and dj > di
    assert overlap_fraction_for_shift(di, dj, **CYCLONE) >= 0.7
    south_i, south_j, south_clamped = clamp_shift_to_overlap(
        0, -16, min_overlap_fraction=0.7, **CYCLONE)
    assert south_clamped is True
    assert south_i == 0 and south_j < 0


def test_an_unbounded_floor_clamps_nothing():
    assert clamp_shift_to_overlap(
        40, 40, min_overlap_fraction=None, **CYCLONE) == (40, 40, False)


def test_the_walk_bottoms_out_on_the_null_move():
    """The null move keeps the whole child, so it always clears the
    floor; there is no proposal the clamp cannot answer.  That is why the
    runner no longer has a way to reach the overlap refusal."""
    di, dj, clamped = clamp_shift_to_overlap(
        30, 30, min_overlap_fraction=0.999, **CYCLONE)
    assert (di, dj, clamped) == (0, 0, True)


# ---------------------------------------------------------------------------
# The runner: the failed run's own geometry
# ---------------------------------------------------------------------------

def _floor_config(moves, floor=0.7, max_move=8):
    """The shape the cyclone quick-start shipped: a maximum its own floor
    refuses on a diagonal."""
    return RelocationConfig(
        enabled=True, grid_id=2, max_move_parent_cells=max_move,
        min_overlap_fraction=floor, moves=tuple(moves))


def _runner(parent_plane, config):
    return RelocationRunner(
        config=config,
        schedule=SimpleNamespace(period_ticks=60,
                                 clock=SimpleNamespace(tick_den=1)),
        staging="host", initializer=_initializer(parent_plane),
        static_provenance="footprint-parametric synthetic statics (test)",
        on_child_built=lambda *args: None)


def test_the_diagonal_maximum_was_refused_before_the_clamp_existed():
    """The plan the runner used to hand to relocate_child, unchanged.

    The CPU tree's child is 120 cells at ratio 3, which is the same 40
    parent cells wide as the quick-start's 160-cell nest at ratio 4, so
    the floor of 0.7 refuses a diagonal 8 here for the same arithmetic
    and with the same numbers.
    """
    _plane, _parent, child = _cpu_tree()
    plan = plan_relocation(
        placement_from=Placement(grid_id=2, i_parent_start=85,
                                 j_parent_start=45),
        placement_to=Placement(grid_id=2, i_parent_start=77,
                               j_parent_start=53, generation=1),
        parent_grid_ratio=int(child.cfg.parent_grid_ratio),
        child_nx=int(child.cfg.run.nx), child_ny=int(child.cfg.run.ny))
    assert plan.overlap_fraction == pytest.approx(0.64)
    with pytest.raises(RelocationRefusal, match="under the configured floor"):
        check_admissible(plan, _floor_config(()))


def test_the_diagonal_maximum_now_executes_clamped_and_the_receipt_says_so():
    parent_plane, parent, child = _cpu_tree()
    start_i = int(child.cfg.i_parent_start)
    start_j = int(child.cfg.j_parent_start)
    model = _model(parent, child)
    runner = _runner(parent_plane, _floor_config(
        (ScheduledRelocationMove(60.0, -8, 8),)))
    _advance(model, parent, child, 60)
    outcome = runner.on_period_begin(model, {1: parent.clock})
    assert outcome["event"] == "relocated"
    assert outcome["requested_shift_parent_cells"] == [-8, 8]
    assert outcome["executed_shift_parent_cells"] == [-6, 6]
    assert "min_overlap_fraction" in outcome["clamped_by"]
    assert outcome["overlap_fraction"] == pytest.approx(0.7225)
    assert outcome["placement_to"]["i_parent_start"] == start_i - 6
    assert outcome["placement_to"]["j_parent_start"] == start_j + 6


def test_the_nest_catches_up_on_the_following_cadence():
    """A clamped move is not a lost move: the follow source is consulted
    again at the next opportunity and the rest of the distance is made
    up there."""
    parent_plane, parent, child = _cpu_tree()
    start_i = int(child.cfg.i_parent_start)
    model = _model(parent, child)
    runner = _runner(parent_plane, _floor_config((
        ScheduledRelocationMove(60.0, -8, 8),
        ScheduledRelocationMove(120.0, -2, 2))))
    executed = []
    for ticks in (60, 120):
        _advance(model, parent, child, ticks)
        executed.append(runner.on_period_begin(model, {1: parent.clock}))
    assert [row["executed_shift_parent_cells"] for row in executed] == [
        [-6, 6], [-2, 2]]
    assert executed[1]["clamped_by"] == []
    assert int(child.cfg.i_parent_start) == start_i - 8


def test_an_in_bounds_move_still_records_no_clamp():
    parent_plane, parent, child = _cpu_tree()
    model = _model(parent, child)
    runner = _runner(parent_plane, _floor_config(
        (ScheduledRelocationMove(60.0, 3, 0),)))
    _advance(model, parent, child, 60)
    outcome = runner.on_period_begin(model, {1: parent.clock})
    assert outcome["executed_shift_parent_cells"] == [3, 0]
    assert outcome["clamped_by"] == []


def test_a_containment_slide_over_the_floor_is_clamped_and_receipted():
    """The sliding ancestor is clamped on the same footing as the mover.

    The floor is the runner's own ``min_overlap_fraction`` re-aimed at
    the ancestor, so a slide that would breach it makes the largest
    slide in the same direction that clears it.  Its row carries what
    the slide kept and which bound cut it, because the auditor's floor
    check reads exactly those fields and the refusal that used to keep
    an under-floor slide off a ledger no longer fires.

    d02 is 120 cells at ratio 3, the same 40 parent cells wide as the
    quick-start's nest, so a diagonal 7 keeps 0.6806 against a floor of
    0.7 and clamps to 6 at 0.7225 -- the mover's own arithmetic, on the
    ancestor.
    """
    # d03 21 cells off centre on both axes: round(21/3) = 7 per axis.
    parent_plane, parent, child, d03 = _tree3(d03_start=(77, 77))
    start_i = int(child.cfg.i_parent_start)
    start_j = int(child.cfg.j_parent_start)
    runner, model, clocks = _containment_runner(
        parent, child, d03, parent_plane, mover_cap=8, contain_cap=8,
        min_overlap_fraction=0.7)
    runner.on_period_begin(model, clocks)
    contained = runner.receipts[0]
    assert contained["event"] == "contained"
    assert contained["mover_deviation_cells"] == [21, 21]
    assert contained["requested_shift_parent_cells"] == [7, 7]
    assert contained["executed_shift_parent_cells"] == [6, 6]
    assert contained["clamped"] is True
    assert contained["clamped_by"] == ["min_overlap_fraction"]
    assert contained["overlap_fraction"] == pytest.approx(0.7225)
    assert overlap_fraction_for_shift(
        7, 7, parent_grid_ratio=3, child_nx=120, child_ny=120) < 0.7
    assert int(child.cfg.i_parent_start) == start_i + 6
    assert int(child.cfg.j_parent_start) == start_j + 6
    # The mover stayed earth-fixed under the slide, compensated exactly.
    assert int(d03.cfg.i_parent_start) == 77 - 6 * 3
    assert int(d03.cfg.j_parent_start) == 77 - 6 * 3


def test_a_containment_slide_inside_the_floor_names_no_clamp():
    """The same leg with a slide the floor admits: the row still carries
    the overlap, so the auditor judges every slide rather than only the
    clamped ones."""
    parent_plane, parent, child, d03 = _tree3(d03_start=(62, 62))
    runner, model, clocks = _containment_runner(
        parent, child, d03, parent_plane, mover_cap=8, contain_cap=8,
        deadband=4, min_overlap_fraction=0.7)
    runner.on_period_begin(model, clocks)
    contained = runner.receipts[0]
    assert contained["event"] == "contained"
    assert contained["executed_shift_parent_cells"] == [2, 2]
    assert contained["clamped"] is False
    assert contained["clamped_by"] == []
    assert contained["overlap_fraction"] == pytest.approx(0.9025)


def test_a_containment_slide_cut_by_its_own_cap_names_the_cap():
    """The cap and the floor are different bounds and the row says
    which one moved the number; the requested shift stays the slide the
    deviation asked for, as it does on the mover's leg."""
    parent_plane, parent, child, d03 = _tree3(d03_start=(77, 56))
    runner, model, clocks = _containment_runner(
        parent, child, d03, parent_plane, mover_cap=8, contain_cap=2,
        min_overlap_fraction=0.7)
    runner.on_period_begin(model, clocks)
    contained = runner.receipts[0]
    assert contained["requested_shift_parent_cells"] == [7, 0]
    assert contained["executed_shift_parent_cells"] == [2, 0]
    assert contained["clamped_by"] == ["containment.max_move_parent_cells"]
    assert contained["overlap_fraction"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# The refusal that remains
# ---------------------------------------------------------------------------

def test_a_named_placement_is_still_refused_and_is_told_the_way_out():
    """A follow source is FOLLOWING, so the runner may move it less.  A
    caller that names a placement asked for THAT position, and quietly
    putting the nest somewhere else would answer a question nobody asked,
    so the refusal stays -- with the bound it should have asked inside
    of, and the knob that would widen it.
    """
    plan = plan_relocation(
        placement_from=Placement(grid_id=2, i_parent_start=20,
                                 j_parent_start=20),
        placement_to=Placement(grid_id=2, i_parent_start=28,
                               j_parent_start=28, generation=1),
        parent_grid_ratio=4, child_nx=160, child_ny=160)
    with pytest.raises(RelocationRefusal) as raised:
        check_admissible(plan, RelocationConfig(
            enabled=True, grid_id=2, min_overlap_fraction=0.7))
    message = str(raised.value)
    assert "at most 6 parent cells per axis" in message
    assert "min_overlap_fraction" in message
    assert "clamp_shift_to_overlap" in message


# ---------------------------------------------------------------------------
# Every shipped configuration agrees with itself
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _shipped_bounds():
    """(label, floor, maximums, ratio, nx, ny) for every shipped follow.

    Both config shapes: a per-domain inline ``follow`` table carrying its
    own bounds, and a tree-level ``[relocation]`` naming a grid_id with
    ``[relocation.follow]`` beside it.
    """
    rows = []
    for path in sorted((_repo_root() / "configs").glob("*.toml")):
        table = tomllib.loads(path.read_text(encoding="utf-8"))
        domains = {int(row["grid_id"]): row
                   for row in table.get("domain", [])
                   if "grid_id" in row}
        for grid_id, row in sorted(domains.items()):
            follow = row.get("follow")
            if not isinstance(follow, dict):
                continue
            rows.append((f"{path.name}: domain {grid_id} follow",
                         follow.get("min_overlap_fraction"),
                         {key: follow[key] for key in
                          ("max_move_parent_cells", "max_shift_cells")
                          if key in follow},
                         int(row["parent_grid_ratio"]), int(row["nx"]),
                         int(row["ny"])))
        relocation = table.get("relocation")
        if isinstance(relocation, dict) and relocation.get("grid_id"):
            row = domains[int(relocation["grid_id"])]
            maximums = {key: relocation[key] for key in
                        ("max_move_parent_cells",) if key in relocation}
            nested = relocation.get("follow") or {}
            if "max_shift_cells" in nested:
                maximums["max_shift_cells"] = nested["max_shift_cells"]
            rows.append((f"{path.name}: [relocation] on domain "
                         f"{relocation['grid_id']}",
                         relocation.get("min_overlap_fraction"), maximums,
                         int(row["parent_grid_ratio"]), int(row["nx"]),
                         int(row["ny"])))
    return rows


def test_every_shipped_configuration_declares_a_reachable_maximum():
    """A maximum above the implied bound is a number a reader can never
    see used: the move passes the per-axis check and the floor then
    refuses it.  That is what ended the cyclone quick forecast."""
    rows = _shipped_bounds()
    assert rows, "no shipped follow configuration was found to check"
    for label, floor, maximums, ratio, nx, ny in rows:
        if floor is None:
            continue
        bound = max_parent_cells_for_overlap(
            floor, parent_grid_ratio=ratio, child_nx=nx, child_ny=ny)
        for key, value in maximums.items():
            assert int(value) <= bound, (
                f"{label}: {key} = {value} but min_overlap_fraction = "
                f"{floor} admits at most {bound} parent cells per axis on "
                f"a {nx}x{ny} nest at ratio {ratio}")


def test_the_quick_start_preset_agrees_with_the_nest_it_is_written_for():
    """The desktop's cyclone quick-start and the CLI's write ONE preset
    (gpuwm.companion_domains.VORTEX_PRESET, through
    gpuwm.cyclone_setup.configuration_text), so this is the check for
    both doors -- on the nest the preset was WRITTEN for.  Every other
    nest the door can emit is the test below."""
    from gpuwm.companion_domains import VORTEX_PRESET
    from gpuwm.cyclone_setup import CHILD_DIMS, RATIO

    bound = max_parent_cells_for_overlap(
        VORTEX_PRESET["min_overlap_fraction"], parent_grid_ratio=RATIO,
        child_nx=CHILD_DIMS[0], child_ny=CHILD_DIMS[1])
    assert bound == 6
    assert VORTEX_PRESET["max_move_parent_cells"] == bound
    assert VORTEX_PRESET["max_shift_cells"] == bound
    assert VORTEX_PRESET["min_shift_cells"] <= bound


def test_the_door_emits_a_reachable_maximum_on_every_rung_it_can_fit_to():
    """The preset is correct for its own nest and WRONG for every rung.

    ``gpuwm cyclone-setup`` on a card that cannot hold the requested
    layout proposes a smaller one, and the floor that implies the
    movement maximum is a fraction of the NEST's width, so the preset's 6
    is unreachable on every rung of that ladder -- worst at the smallest,
    where a 24-parent-cell nest admits 3.  Copying the preset onto a
    fitted child put back exactly the contradiction that ended the
    cyclone quick forecast, in the door the fix was measured through, so
    the check walks the whole ladder rather than the unfitted
    dimensions.

    Both emitted tables are checked, because a desktop reads the
    cyclone.json receipt and the run reads the toml beside it.
    """
    from gpuwm import cyclone_setup as tc

    intent = dict(cycle="2026090900", point=(18., -65.), hours=6,
                  name="Ladder bound check", source="cyclone.toml")
    _text, unfitted = tc.configuration_text(**intent)
    scales = tc._fit_scales(unfitted)
    assert scales, "the fit ladder proposed no rung to check"
    for scale in (1.0,) + tuple(scales):
        dims = ([list(tc.ROOT_DIMS), list(tc.CHILD_DIMS)] if scale == 1.0
                else tc._fit_dimensions(scale))
        text, experiment = tc.configuration_text(**intent, dimensions=dims)
        emitted = tomllib.loads(text)["domain"][1]["follow"]
        child = experiment.domains[1]
        bound = max_parent_cells_for_overlap(
            emitted["min_overlap_fraction"], parent_grid_ratio=tc.RATIO,
            child_nx=child.run.nx, child_ny=child.run.ny)
        for key in ("max_move_parent_cells", "max_shift_cells"):
            assert emitted[key] <= bound, (
                f"rung {scale}: {key} = {emitted[key]} on a "
                f"{child.run.nx}x{child.run.ny} nest, whose "
                f"min_overlap_fraction = {emitted['min_overlap_fraction']} "
                f"admits at most {bound} parent cells per axis")
            # And reachable: the bound itself must clear the floor, so
            # the maximum written is a move that can actually be made.
            assert overlap_fraction_for_shift(
                emitted[key], emitted[key], parent_grid_ratio=tc.RATIO,
                child_nx=child.run.nx, child_ny=child.run.ny
            ) >= emitted["min_overlap_fraction"]
        assert emitted["min_shift_cells"] <= emitted["max_shift_cells"]
        # The receipt's copy is the same table, derived from the same
        # dimensions rather than from the preset.
        assert tc.follow_table_for_nest(child.run.nx, child.run.ny) == {
            key: value for key, value in emitted.items() if key != "track"}
