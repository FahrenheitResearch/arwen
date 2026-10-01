"""The following nest grows to a memory budget, and its bounds grow with it.

Two properties are checked here and nothing else is asserted about them:
the nest this door chooses is the LARGEST admitted rung of its own ladder,
and the movement bounds written beside it are the ones that nest's own
overlap floor admits.  Both are measured through the same functions the
door uses, never against a copied number.
"""
import argparse
import dataclasses
import json
import math
import re
import tomllib

import pytest

from gpuwm import cyclone_setup as tc
from gpuwm import domain_wizard as dw
from gpuwm.companion_domains import VORTEX_PRESET
from gpuwm.configuration_recovery import MemoryAdmissionError

CYCLE = "2026090900"
POINT = (18., -65.)
CARD = dw.SizingBudget(16.0, int(9.0 * dw.GIB), None, "fixture", measured=False)


def priced(dims, sizing=CARD):
    """The peak envelope and the budget this tree is admitted against."""
    _text, exp = tc.configuration_text(cycle=CYCLE, point=POINT, hours=3,
                                       tiles="off", dimensions=dims)
    operands = dict(free_bytes=sizing.free_bytes, vram_gib=sizing.vram_gib,
                    profile=sizing.device_profile,
                    forcing_interval_seconds=10800.)
    phases = dw._sizing_phases(exp, source="gfs", machine=None, **operands)
    budget = dw.sizing_budget_bytes(exp, **operands)
    return phases.peak_envelope_bytes, budget


def test_the_nest_ladder_is_square_and_in_whole_even_parent_cells():
    _text, exp = tc.configuration_text(cycle=CYCLE, point=POINT, hours=3)
    ladder = tc._nest_ladder(exp)
    assert ladder[-1] == tc.PRESET_NEST_PARENT_CELLS
    assert list(ladder) == sorted(ladder, reverse=True)
    for cells in ladder:
        assert cells % 2 == 0
        parent, child = tc._nest_dimensions(cells)
        assert tuple(parent) == tc.ROOT_DIMS
        assert child[0] == child[1] == tc.RATIO * cells
        assert child[0] % (2 * tc.RATIO) == 0


def test_every_rung_keeps_its_own_tracker_window_inside_the_parent():
    from gpuwm.experiment import validate_spawn_placement
    _text, exp = tc.configuration_text(cycle=CYCLE, point=POINT, hours=3)
    for cells in tc._nest_ladder(exp):
        _rung_text, rung = tc.configuration_text(
            cycle=CYCLE, point=POINT, hours=3,
            dimensions=tc._nest_dimensions(cells))
        child = rung.domains[1]
        clearance = (child.follow.tracker.search_margin_cells
                     + max(child.follow.tracker.max_shift_cells,
                           child.follow.max_move_parent_cells))
        for di in (-clearance, clearance):
            for dj in (-clearance, clearance):
                validate_spawn_placement(rung, 2, child.i_parent_start + di,
                                         child.j_parent_start + dj)


@pytest.mark.parametrize("cells", [42, 44, 48, 52, 56, 60])
def test_the_derived_move_is_the_largest_the_nests_own_floor_admits(cells):
    """An overlap floor f admits a per-axis magnitude of 1 - sqrt(f) of the
    nest's width, and the binding case is the diagonal move, where both
    factors shrink at once."""
    follow = tc.follow_table_for_nest(tc.RATIO * cells, tc.RATIO * cells)
    floor = follow["min_overlap_fraction"]
    move = follow["max_move_parent_cells"]
    assert follow["max_shift_cells"] == move
    assert (1.0 - move / cells) ** 2 >= floor
    assert (1.0 - (move + 1) / cells) ** 2 < floor
    assert move == int(math.floor(cells * (1.0 - math.sqrt(floor)) + 1e-9))
    assert follow["min_shift_cells"] <= move
    assert follow["search_margin_cells"] == cells // 2


def test_the_preset_nest_keeps_the_preset_table_exactly():
    assert tc.follow_table_for_nest(*tc.CHILD_DIMS) == VORTEX_PRESET
    text, _exp = tc.configuration_text(cycle=CYCLE, point=POINT, hours=3)
    child = tomllib.loads(text)["domain"][1]
    assert {k: v for k, v in child["follow"].items() if k != "track"} == VORTEX_PRESET


def test_a_resized_nest_moves_only_the_bounds_the_size_derives():
    resized = tc.follow_table_for_nest(tc.RATIO * 52, tc.RATIO * 52)
    moved = {key for key in resized if resized[key] != VORTEX_PRESET[key]}
    assert moved <= {"max_move_parent_cells", "max_shift_cells",
                     "search_margin_cells", "min_shift_cells"}
    assert resized["min_overlap_fraction"] == VORTEX_PRESET["min_overlap_fraction"]
    assert resized["cadence_seconds"] == VORTEX_PRESET["cadence_seconds"]
    assert resized["field"] == VORTEX_PRESET["field"]


def test_a_budget_chooses_the_largest_rung_that_fits_with_the_admission_margin():
    plan = tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=CARD, hours=3,
                           tiles="off", nest_budget_gib=5.5)
    nest = plan["nest"]
    chosen = nest["parent_cells"]
    assert chosen > tc.PRESET_NEST_PARENT_CELLS
    assert nest["dimensions"] == [tc.RATIO * chosen, tc.RATIO * chosen]
    assert [d["nx"] for d in plan["domains"]] == [tc.ROOT_DIMS[0], tc.RATIO * chosen]
    assert plan["domains"][0]["ny"] == tc.ROOT_DIMS[1]
    assert nest["budget_bound_by"] == "request"
    assert plan["fitting"]["changed"] is False

    # Priced against the budget the plan was priced against, which is the
    # narrowed allowance and not the card's own.
    narrowed, bound = tc._budget_sizing(CARD, 5.5)
    assert bound == "request"
    envelope, budget = priced(tc._nest_dimensions(chosen), narrowed)
    assert envelope <= budget - dw.fit_headroom_bytes(budget)
    assert nest["peak_envelope_bytes"] == envelope
    assert nest["budget_bytes"] == budget
    # The ladder chose it, so the headroom the document names was held back.
    assert nest["headroom_bytes"] == dw.fit_headroom_bytes(budget)

    _text, exp = tc.configuration_text(cycle=CYCLE, point=POINT, hours=3)
    if chosen + 2 in tc._nest_ladder(exp):
        bigger, budget = priced(tc._nest_dimensions(chosen + 2), narrowed)
        assert bigger > budget - dw.fit_headroom_bytes(budget)


def test_the_budget_is_what_the_whole_tree_is_priced_against():
    """A budget narrows the allowance every later number is measured
    against, so a smaller budget can never choose a larger nest."""
    sizes = []
    for budget_gib in (5.1, 5.5, 6.5):
        plan = tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=CARD, hours=3,
                               tiles="off", nest_budget_gib=budget_gib)
        sizes.append(plan["nest"]["parent_cells"])
        assert plan["nest"]["budget_bytes"] <= int(budget_gib * dw.GIB)
    assert sizes == sorted(sizes)


def test_a_budget_over_the_card_is_bound_by_the_card_and_says_so():
    plan = tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=CARD, hours=3,
                           tiles="off", nest_budget_gib=64.0)
    assert plan["nest"]["budget_bound_by"] == "card"
    assert plan["nest"]["budget_bytes"] == dw.sizing_budget_bytes(
        None, free_bytes=CARD.free_bytes, vram_gib=CARD.vram_gib,
        forcing_interval_seconds=10800., profile=None)


def test_a_budget_under_the_preset_floor_refuses_naming_the_floor():
    with pytest.raises(Exception) as refusal:
        tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=CARD, hours=3,
                        tiles="off", nest_budget_gib=0.5)
    text = str(refusal.value)
    assert f"{tc.CHILD_DIMS[0]}x{tc.CHILD_DIMS[1]}" in text
    assert f"{tc.PRESET_NEST_PARENT_CELLS} parent cells" in text
    assert "sized UP from that floor" in text
    assert "raise --nest-budget-gib" in text


@pytest.mark.parametrize("budget", [0, -1.0, float("nan"), float("inf")])
def test_a_budget_that_is_not_a_size_is_refused(budget):
    with pytest.raises(ValueError, match="finite positive size in GiB"):
        tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=CARD, hours=3,
                        tiles="off", nest_budget_gib=budget)


def test_without_a_budget_the_preset_layout_is_untouched():
    plan = tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=CARD, hours=3,
                           tiles="off")
    assert plan["nest"]["dimensions"] == list(tc.CHILD_DIMS)
    assert plan["nest"]["sized_to_budget"] is False
    assert plan["nest"]["budget_gib"] is None
    assert plan["follow"] == VORTEX_PRESET
    assert plan["fitting"]["original_dimensions"] == [list(tc.ROOT_DIMS),
                                                     list(tc.CHILD_DIMS)]


def test_the_menu_carries_the_floor_a_budget_field_is_bounded_by(capsys):
    parser = argparse.ArgumentParser()
    tc.register_cli(parser.add_subparsers(dest="command", required=True))
    assert tc.main(parser.parse_args(["cyclone-setup", "--list-sources"])) == 0
    menu = json.loads(capsys.readouterr().out)
    assert menu["nest_floor"] == {"dimensions": list(tc.CHILD_DIMS),
                                  "parent_cells": tc.PRESET_NEST_PARENT_CELLS,
                                  "parent_dimensions": list(tc.ROOT_DIMS),
                                  "ratio": tc.RATIO}


def test_the_cli_prints_the_chosen_nest_and_the_priced_memory(monkeypatch, capsys):
    parser = argparse.ArgumentParser()
    tc.register_cli(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(["cyclone-setup", "--cycle", CYCLE, "--point=18,-65",
                              "--hours", "3", "--tiles", "off",
                              "--nest-budget-gib", "5.5", "--json"])
    monkeypatch.setattr(dw, "_domain_target_hardware",
                        lambda args: (CARD, None, False))
    assert tc.main(args) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    nest = result["nest"]
    assert nest["sized_to_budget"] is True
    line = [row for row in captured.err.splitlines() if row.startswith("plan:")]
    assert len(line) == 1
    assert f"{nest['dimensions'][0]}x{nest['dimensions'][1]}" in line[0]
    assert str(nest["peak_envelope_bytes"]) in line[0]
    assert str(result["memory"]["budget_bytes"]) in line[0]
    assert f"({nest['parent_cells']} parent cells)" in line[0]


@pytest.mark.parametrize("cells,move", [(24, 3), (30, 4), (36, 5), (38, 6),
                                        (40, 6)])
def test_a_nest_under_the_preset_derives_DOWN_from_the_same_floor(cells, move):
    """ONE derivation, both directions.

    The preset's own 6 is what a 40-parent-cell nest's floor admits, so a
    narrower nest admits fewer and the same function that grows the
    maximum for a budget-grown nest reduces it here.  A rung carrying the
    preset's number instead would declare a move its own floor refuses,
    which is the reduction ladder's half of this defect.
    """
    follow = tc.follow_table_for_nest(tc.RATIO * cells, tc.RATIO * cells)
    assert follow["max_move_parent_cells"] == follow["max_shift_cells"] == move
    assert (1.0 - move / cells) ** 2 >= follow["min_overlap_fraction"]
    assert (1.0 - (move + 1) / cells) ** 2 < follow["min_overlap_fraction"]
    # The search margin is the preset's own 20 at or under the preset's
    # size: a rung emits the window its configuration declares, and only
    # a nest grown past the preset grows it.
    assert follow["search_margin_cells"] == VORTEX_PRESET["search_margin_cells"]


def test_a_budget_sized_proposal_reduces_from_the_layout_it_chose():
    """A reduction is a reduction OF the requested layout, and once a budget
    can size the nest the requested layout is no longer the preset."""
    assert tc._reduction_dimensions() is tc._fit_dimensions
    assert tc._reduction_dimensions([list(tc.ROOT_DIMS), list(tc.CHILD_DIMS)]) \
        is tc._fit_dimensions
    sized = tc._reduction_dimensions([[200, 160], [224, 224]])
    assert sized(0.5) == [(100, 80), (112, 112)]
    assert tc._fit_dimensions(0.5)[1] == (80, 80)


# The card, not the budget, is what a small card refuses with ------------------
#
# `_budget_sizing` answers WHICH of the two numbers bound the run a hundred
# lines before the refusal is built, and the refusal used to ignore it: a
# 24 GiB budget on a card that holds 5 GiB was refused as "--nest-budget-gib
# is under the cyclone preset's own nest", reported `bound_by: nest-budget`
# to a form, and offered "raise --nest-budget-gib" as the way out.  24 GiB is
# not under a 4.8 GiB floor; the card is, raising the flag cannot move a card,
# and the way that does work -- drop the flag -- was not named.
#
# 5.15 GiB free, 5.25 until A163 measured the forecast margin at 1.13 of
# the subtotal: at 5.25 the flagless door now fits the whole preset
# (200x160 / 160x160) unchanged, so that card was no longer one the floor
# is too big for.  Read at 5.25, 5.15, 5.1, 5.05, 5.0 and 4.9 GiB: from
# 5.15 down the refusal's measured way out is the layout dropping the flag
# authors.
SMALL_CARD = dw.SizingBudget(6.0, int(5.15 * dw.GIB), None, "fixture",
                             measured=False)


def small_card_refusal(budget_gib):
    with pytest.raises(MemoryAdmissionError) as refusal:
        tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=SMALL_CARD, hours=3,
                        tiles="off", nest_budget_gib=budget_gib)
    return refusal.value


def test_a_card_too_small_for_the_floor_names_the_card_and_not_the_budget():
    refusal = small_card_refusal(24.0)
    text = str(refusal)
    assert refusal.memory["bound_by"] == "card"
    assert refusal.memory["reason"] == "nest-floor-card"
    # The sentence blames what refused, and never sends the reader after a
    # flag that cannot move: the budget asked for is larger than the card.
    assert "raise --nest-budget-gib" not in text
    assert "the card is the bound" in text
    assert str(int(24.0 * dw.GIB)) in text
    assert "cannot move it" in text


def test_the_card_bound_refusal_names_the_layout_dropping_the_flag_authors():
    """The way out is measured before it is offered, and it is the layout the
    same door authors when the flag is dropped -- not a mode to go and try."""
    refusal = small_card_refusal(24.0)
    found = refusal.memory["unbudgeted_alternative"]
    assert found is not None
    assert f"{found[0][0]}x{found[0][1]} / {found[1][0]}x{found[1][1]}" \
        in str(refusal)

    dropped = tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=SMALL_CARD,
                              hours=3, tiles="off")
    assert [[d["nx"], d["ny"]] for d in dropped["domains"]] == found
    assert dropped["fitting"]["changed"] is True


def test_a_budget_under_the_floor_on_a_card_that_holds_it_still_blames_itself():
    """The other half of the same question: on the 16 GiB fixture the card is
    not the bound, so the budget is, and raising it is the way out."""
    with pytest.raises(MemoryAdmissionError) as refusal:
        tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=CARD, hours=3,
                        tiles="off", nest_budget_gib=0.5)
    assert refusal.value.memory["bound_by"] == "nest-budget"
    assert refusal.value.memory["reason"] == "nest-budget-floor"
    assert refusal.value.memory["unbudgeted_alternative"] is None
    assert "raise --nest-budget-gib" in str(refusal.value)



def test_the_floor_price_is_named_on_the_default_tile_mode_too():
    """A REFUSED floor is still a priced floor.

    Under `--tiles auto` the estimator raises on the preset nest instead of
    returning phases, because the tile planner's tree road refuses before it
    answers, so the refusal named no price at all on the tile mode every user
    meets while `--tiles off` beside it named one.  The envelope the fitter
    measured travels on the error.
    """
    with pytest.raises(MemoryAdmissionError) as refusal:
        tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=SMALL_CARD, hours=3,
                        tiles="auto", nest_budget_gib=24.0)
    resident, _budget = priced(tc._nest_dimensions(tc.PRESET_NEST_PARENT_CELLS),
                               SMALL_CARD)
    assert f"which prices {resident} bytes here" in str(refusal.value)
    assert refusal.value.memory["bound_by"] == "card"

# The floor is judged the way the preset is judged without the flag -----------
#
# The floor IS the preset nest, so `--nest-budget-gib` holds it to the number
# this door holds the preset to when the flag is dropped: the budget, with no
# fit headroom held back (the headroom is what a nest GROWN past the floor
# leaves unspent).  It was held to the fit target, the budget less that
# headroom, and on the band between the two the flag refused the very layout
# the flagless door authors on that much free memory (A175): on SMALL_CARD,
# below, the card's own, since the flag asks for more than the card has; on
# the 16 GiB fixture, the budget the flag names.
FLOOR = tc._nest_dimensions(tc.PRESET_NEST_PARENT_CELLS)


def layout(plan):
    return [[d["nx"], d["ny"]] for d in plan["domains"]]


# Refused against the fit target until A175, inside the budget all along (read
# at 0.05 GiB steps from 4.6 to 5.05 at the 1.13 margin: 4.75 through 4.95).
@pytest.mark.parametrize("budget_gib", [4.75, 4.85, 4.95])
def test_a_floor_inside_its_budget_is_the_preset_and_not_a_refusal(budget_gib):
    narrowed, bound = tc._budget_sizing(CARD, budget_gib)
    assert bound == "request"
    cost, budget = priced(FLOOR, narrowed)
    # The band the floor was refused in: over the fit target, inside the
    # budget.
    assert budget - dw.fit_headroom_bytes(budget) < cost <= budget
    plan = tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=CARD, hours=3,
                           tiles="off", nest_budget_gib=budget_gib)
    assert plan["nest"]["dimensions"] == list(tc.CHILD_DIMS)
    assert plan["nest"]["budget_bytes"] == budget
    assert plan["nest"]["peak_envelope_bytes"] == cost
    assert plan["fitting"]["changed"] is False
    # Admitted as requested holds no headroom back, and the document says so
    # rather than naming bytes unspent that the envelope spent.
    assert plan["nest"]["headroom_bytes"] == 0
    # One answer: the flagless door on a card whose free memory is this
    # budget authors the same preset, unchanged, and reports it the same way.
    dropped = tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=narrowed,
                              hours=3, tiles="off")
    assert dropped["fitting"]["changed"] is False
    assert layout(dropped) == layout(plan)
    assert dropped["nest"]["headroom_bytes"] == 0
    assert dropped["memory"]["budget_bytes"] == budget


@pytest.mark.parametrize("budget_gib", [4.6, 4.7])
def test_the_floor_refusal_quotes_the_budget_it_was_compared_against(budget_gib):
    """The number printed is the number compared, so a refused floor's price
    is above it and raising the flag past that price is the way out."""
    with pytest.raises(MemoryAdmissionError) as refusal:
        tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=CARD, hours=3,
                        tiles="off", nest_budget_gib=budget_gib)
    text = str(refusal.value)
    budget = refusal.value.memory["budget_bytes"]
    narrowed = tc._budget_sizing(CARD, budget_gib)[0]
    cost, priced_budget = priced(FLOOR, narrowed)
    assert budget == priced_budget < cost
    assert f"prices {cost} bytes here, against a {budget} byte budget." in text
    assert "fit target" not in text and "headroom" not in text
    assert "raise --nest-budget-gib until it clears what the floor prices" \
        in text
    # And the flagless door on a card whose free memory is this budget
    # shrinks the preset: the refusal is where the two doors part.
    dropped = tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=narrowed,
                              hours=3, tiles="off")
    assert dropped["fitting"]["changed"] is True


def _both_doors(sizing, tiles, budget_gib):
    """``(flagless plan, flagged plan or the flag's refusal)`` on one card."""
    dropped = tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=sizing,
                              hours=3, tiles=tiles)
    try:
        return dropped, tc.plan_cyclone(cycle=CYCLE, point=POINT,
                                        sizing=sizing, hours=3, tiles=tiles,
                                        nest_budget_gib=budget_gib)
    except MemoryAdmissionError as refusal:
        return dropped, refusal


def _assert_one_answer(dropped, flagged, reason):
    """The flag refuses exactly where dropping it shrinks the preset, and
    admits, never below the floor, exactly where dropping it keeps it."""
    preset = [list(tc.ROOT_DIMS), list(tc.CHILD_DIMS)]
    if isinstance(flagged, MemoryAdmissionError):
        assert flagged.memory["reason"] == reason
        assert dropped["fitting"]["changed"] is True
        assert layout(dropped) != preset
        return "refused"
    assert dropped["fitting"]["changed"] is False
    assert layout(dropped) == preset
    assert flagged["nest"]["parent_cells"] >= tc.PRESET_NEST_PARENT_CELLS
    assert layout(flagged)[0] == list(tc.ROOT_DIMS)
    # Whatever chose it, the headroom the document names was held back.
    nest = flagged["nest"]
    assert nest["headroom_bytes"] in {0, dw.fit_headroom_bytes(nest["budget_bytes"])}
    assert (nest["peak_envelope_bytes"]
            <= nest["budget_bytes"] - nest["headroom_bytes"])
    return "admitted"


# Free sizes across the band, SMALL_CARD with a budget the card cannot honour.
# At 57066783c (margin 1.13), at 0.025 GiB steps from 5.1 to 5.6: `--tiles
# off` disagreed from 5.25 through 5.475 (the flag refused naming 180x144 /
# 144x144, then 190x152 / 152x152, while dropping it kept 200x160 / 160x160);
# `--tiles auto` agreed throughout, because the tree road's withholding binds
# below both numbers there.  The sweep starts two steps under the band and
# ends two over it, so both answers are reached on each road.
CARD_SWEEP_OFF = [round(5.2 + 0.025 * k, 3) for k in range(15)]
CARD_SWEEP_AUTO = [round(5.2 + 0.05 * k, 3) for k in range(8)]


@pytest.mark.parametrize("tiles,free_gib",
                         [("off", f) for f in CARD_SWEEP_OFF]
                         + [("auto", f) for f in CARD_SWEEP_AUTO])
def test_the_card_bound_flag_and_the_flagless_door_give_one_answer(tiles,
                                                                    free_gib):
    card = dataclasses.replace(SMALL_CARD, free_bytes=int(free_gib * dw.GIB))
    dropped, flagged = _both_doors(card, tiles, 24.0)
    answer = _assert_one_answer(dropped, flagged, "nest-floor-card")
    if answer == "refused":
        # And the way out it names is that shrunken layout, measured.
        assert flagged.memory["bound_by"] == "card"
        assert flagged.memory["unbudgeted_alternative"] == layout(dropped)
    else:
        assert flagged["nest"]["budget_bound_by"] == "card"


def test_the_card_sweep_reaches_both_answers_on_both_roads():
    """A sweep that never crossed the edge would prove nothing about it."""
    for tiles, sweep in (("off", CARD_SWEEP_OFF), ("auto", CARD_SWEEP_AUTO)):
        answers = set()
        for free_gib in (sweep[0], sweep[-1]):
            card = dataclasses.replace(SMALL_CARD,
                                       free_bytes=int(free_gib * dw.GIB))
            answers.add(_assert_one_answer(*_both_doors(card, tiles, 24.0),
                                           "nest-floor-card"))
        assert answers == {"refused", "admitted"}, tiles


# The same question with the budget the flag names as the bound, on the 16
# GiB fixture: the flagless door is asked on a card whose free memory IS that
# budget (`_budget_sizing`'s own narrowing).  Across the old band 4.75 to 4.95
# and one step either side.
REQUEST_SWEEP = [round(4.7 + 0.05 * k, 3) for k in range(7)]


@pytest.mark.parametrize("budget_gib", REQUEST_SWEEP)
def test_the_request_bound_flag_and_the_flagless_door_give_one_answer(
        budget_gib):
    narrowed, bound = tc._budget_sizing(CARD, budget_gib)
    assert bound == "request"
    dropped = tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=narrowed,
                              hours=3, tiles="off")
    try:
        flagged = tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=CARD,
                                  hours=3, tiles="off",
                                  nest_budget_gib=budget_gib)
    except MemoryAdmissionError as refusal:
        flagged = refusal
    answer = _assert_one_answer(dropped, flagged, "nest-budget-floor")
    if answer == "admitted":
        assert flagged["nest"]["budget_bound_by"] == "request"


# ... AND THE BUDGET IS ONLY THE FLAT ROAD'S BOUND ----------------------------
#
# On the DEFAULT `--tiles auto` the tree walk withholds the following nest's
# rebuild transient before it compares anything, so it refuses against a
# budget SMALLER than the flat road's, and smaller than the fit target too:
# 4.99 through 5.005 GiB refuse while the fit target sits ABOVE the price, and
# 5.01 admits.  The number the sentence quotes has to be the one that bound,
# which is at or under the price.  (Measured on the 50 hPa model top the GFS
# emission carries; the band sat 0.01 GiB higher on the old 100 hPa top, and
# at 5.035 through 5.05 while the RRTMGP solver's frame was priced at a stale
# 5,152 B, which charged this tree 8,355,840 B more.  It sat at 5.025
# through 5.045, admitting at 5.05, on the plan's 1.15 margin; A163 measured
# the forecast margin at 1.13 of the subtotal, read again at 0.005 GiB steps
# from 4.85 to 5.07.)
TILE_BAND = [4.99, 4.995, 5.0, 5.005]


@pytest.mark.parametrize("budget_gib", TILE_BAND)
def test_the_default_tile_modes_refusal_quotes_the_bound_that_bound_it(budget_gib):
    with pytest.raises(MemoryAdmissionError) as refusal:
        tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=CARD, hours=3,
                        tiles="auto", nest_budget_gib=budget_gib)
    text = str(refusal.value)
    cost = int(re.search(r"prices (\d+) bytes here", text).group(1))
    quoted = int(re.search(r"against the (\d+) byte admission budget",
                           text).group(1))
    budget = refusal.value.memory["budget_bytes"]
    target = budget - dw.fit_headroom_bytes(budget)
    # The band this sentence was wrong in, one road over: the fit target is
    # ABOVE the price and the run is refused anyway.
    assert target > cost
    # What is quoted is what refused, and the flat road's number is not
    # printed beside it for a reader to raise the flag against.
    assert quoted <= cost
    assert f"{target} byte fit target" not in text
    assert "withholds" in text and "rebuild" in text
    assert "raise --nest-budget-gib" in text


def test_the_tile_roads_band_ends_where_its_own_budget_clears_the_price():
    """And it ends there rather than where the flat road's budget or the fit
    target does, which is the whole reason neither can be the number quoted."""
    plan = tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=CARD, hours=3,
                           tiles="auto", nest_budget_gib=5.01)
    assert plan["nest"]["dimensions"] == list(tc.CHILD_DIMS)


def test_the_bound_is_read_off_the_walk_rather_than_recomputed():
    """The door quotes a number the walk carried out to it.

    ``TreeRoadPlan`` carries the refusal's own budget, its withholding and
    its way out as numbers and a string; parsing them back out of the
    sentence, or computing a second budget here, is the defect this carry
    exists to prevent.
    """
    narrowed, bound_by = tc._budget_sizing(CARD, TILE_BAND[-1])
    assert bound_by == "request"
    floor = tc._nest_dimensions(tc.PRESET_NEST_PARENT_CELLS)
    _text, exp = tc.configuration_text(cycle=CYCLE, point=POINT, hours=3,
                                       tiles="auto", dimensions=floor)
    with pytest.raises(dw.DomainFitError) as error:
        dw._sizing_phases(exp, source="gfs", machine=None,
                          free_bytes=narrowed.free_bytes,
                          vram_gib=narrowed.vram_gib,
                          profile=narrowed.device_profile,
                          forcing_interval_seconds=10800.)
    carried = tc._tile_road_bound(error.value)
    assert carried is not None
    assert f"{carried['budget_bytes']} byte admission budget" in str(error.value)
    assert carried["withheld_bytes"] > 0
    assert carried["withheld_for"] == "d02"
    assert carried["remedy"] and carried["remedy"] in str(error.value)
    # And an error that carries no walk carries no bound, which is what
    # keeps the flat road's sentence quoting its budget.
    assert tc._tile_road_bound(
        dw.DomainFitError("no phases", resource="vram")) is None


# The derived maximum, measured rung by rung -----------------------------------
#
# This table stepped DOWN at 42 while the preset declared 8 at 40 parent
# cells: a diagonal 8 at 40 keeps 0.64 of the child against a 0.7 floor, so
# the derivation could not reproduce the preset and the first rungs above it
# derived less.  The preset's maximum is now the 6 its own floor admits, so
# the table is monotone from 40 up and the step is gone.  It is asserted
# whole so it cannot move unnoticed, and so the prose describing it cannot go
# stale again.
DERIVED_MOVE = {40: 6, 42: 6, 44: 7, 46: 7, 48: 7,
                50: 8, 52: 8, 54: 8, 56: 9, 58: 9, 60: 9}


@pytest.mark.parametrize("cells,move", sorted(DERIVED_MOVE.items()))
def test_the_derived_maximum_is_this_measured_table(cells, move):
    follow = tc.follow_table_for_nest(tc.RATIO * cells, tc.RATIO * cells)
    assert follow["max_move_parent_cells"] == move
    assert follow["max_shift_cells"] == move


def test_the_presets_own_maximum_is_exactly_what_its_own_floor_admits():
    """Why the table is monotone now, stated as a measurement rather than
    left for a reader to rediscover from the numbers."""
    cells = tc.PRESET_NEST_PARENT_CELLS
    floor = VORTEX_PRESET["min_overlap_fraction"]
    preset_move = VORTEX_PRESET["max_move_parent_cells"]
    assert (1.0 - preset_move / cells) ** 2 >= floor
    assert (1.0 - (preset_move + 1) / cells) ** 2 < floor
    assert preset_move == int(math.floor(cells * (1.0 - math.sqrt(floor)) + 1e-9))
    # So the derivation reproduces the preset at the preset's own size, and
    # a default run emits the shipped table byte for byte.
    assert tc.follow_table_for_nest(*tc.CHILD_DIMS) == VORTEX_PRESET
