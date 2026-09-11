"""``--tiles auto`` on a TREE means resident when the whole tree fits.

The single-domain road has asked the resident question first since the
configured admission context existed; the tree road did not.  It charged
the tile planner's shared process/radiation floor -- a STREAMED process's
per-rung tables plus a radiation reservation the configured envelope
already carries inside itself -- before one domain was priced, so on a
card with a desktop on it no candidate could pass while the same card
held the tree resident with room to spare.

MEASURED, and these are the numbers below: the 12/3 km moving-nest
cyclone tree (200x160 at 12 km, 160x160 at 3 km, 49 levels, the shipped
GFS suite) against a 6,855,065,600 byte admission budget -- an RTX 3080
with a desktop on it.  The floor is 6,978,986,310 bytes and refused;
the configured tree wants about 5.1e9 bytes and runs.

Three further things this file pins, each of which was a way for the
admitted road and the road actually taken to come apart:

* the refusal a domain's own ``mode = "on"`` provokes on a tree that
  FITS names that table and not the card;
* the plan review and the run door price the admission from ONE shared
  estimate, so a tree the review admits is not refused at run start;
* an automatic tree that MOVES a nest withholds that nest's rebuild from
  the admission budget and records the pinned host copy the move stages
  through, neither of which the steady-state envelope models.
"""
import re

import pytest

from gpuwm import cyclone_setup as tc, domain_wizard as dw
from gpuwm import prepared_domain_tree_forecast as pdtf
from gpuwm.core import preflight as pf, streaming as st
from gpuwm.core.streamed_relocation import mark_reconstruction_nodes
from tilestream import autoplan as ap

GIB = 1024 ** 3
#: The desktop's own budget on the card that produced the refusal.
BUDGET = 6_855_065_600
FREE = BUDGET + pf.EXTERNAL_MARGIN_BYTES
#: The centre the refusal was raised on.
POINT = (16.38581807563628, -123.76623740203063)
CYCLE = "2026091112"


def _cyclone(tiles="auto"):
    _text, exp = tc.configuration_text(cycle=CYCLE, point=POINT, hours=6,
                                       tiles=tiles)
    return exp


def _machine(free_bytes=FREE):
    return ap.Machine(int(free_bytes), 256 * GIB)


def _sentences(text):
    return [part for part in re.split(r"(?<=\.)\s+", text.strip()) if part]


def _itemized(estimate):
    return {int(d.grid_id): int(d.resident_bytes) for d in estimate.domains}


def test_a_fitting_cyclone_tree_is_admitted_resident_without_the_tile_planner(
        monkeypatch):
    exp = _cyclone()
    nodes = st._config_tree_nodes(exp.domains)
    estimate = pf.estimate_experiment(exp)
    # The two numbers the defect sat between: the floor exceeds the budget,
    # the configured tree does not.  Both are read off the shipped models,
    # so this test fails if either model moves rather than passing vacuously.
    floor = (st._tree_process_overhead_bytes(nodes)
             + st._tree_radiation_transient_bytes(nodes))
    assert floor > BUDGET
    assert estimate.peak_envelope_bytes <= BUDGET

    monkeypatch.setattr(ap, "plan", lambda *a, **kw: pytest.fail(
        "the tile planner was consulted for a tree that fits resident"))
    rows = {}
    result = st.decide_tree(nodes, exp.tiles, machine=_machine(),
                            decisions=rows, resident_estimate=estimate)
    assert sorted(rows) == [1, 2]
    assert not any(row.stream for row in rows.values())
    assert all(row.reason == "the configured resident envelope fits this budget"
               for row in rows.values())
    assert all(row.budget_bytes == BUDGET for row in rows.values())
    assert result.priced and result.host_spent_bytes == 0
    assert result.vram_spent_bytes == estimate.peak_envelope_bytes
    assert result.total_budget_bytes == BUDGET


def test_each_resident_row_carries_its_own_domains_bytes_and_the_tree_shares_the_rest():
    # The whole-tree envelope written onto every row made two domains sum
    # to twice the tree while each row claimed 0.00 GiB beside it.
    exp = _cyclone()
    estimate = pf.estimate_experiment(exp)
    itemized = _itemized(estimate)
    assert sorted(itemized) == [1, 2] and len(set(itemized.values())) == 2

    rows = {}
    st.decide_tree(st._config_tree_nodes(exp.domains), exp.tiles,
                   machine=_machine(), decisions=rows,
                   resident_estimate=estimate)
    assert {gid: row.resident_bytes for gid, row in rows.items()} == itemized
    assert {gid: row.detail["claim_bytes"] for gid, row in rows.items()} == itemized
    # Rows plus the tree's stated remainder ARE the envelope: nothing is
    # counted twice and nothing is dropped.
    shared = {row.detail["resident_admission"]["tree_shared_bytes"]
              for row in rows.values()}
    assert len(shared) == 1
    remainder = shared.pop()
    assert sum(itemized.values()) + remainder == estimate.peak_envelope_bytes
    # And the report that prints those rows adds them up in words too.
    road = st.tree_road_plan(exp, machine=_machine(),
                             resident_estimate=estimate)
    lines = road.row_lines()
    assert [line.split(":")[0] for line in lines[:2]] == ["d01 resident",
                                                          "d02 resident"]
    assert lines[-1].startswith("the tree's shared residency")


def test_the_run_door_and_the_plan_review_take_that_same_decision():
    # THE BAND.  The door used to price the admission with the prepared
    # cache's retained forcing intervals; the review priced it without
    # them.  For a budget between the two envelopes the review admitted
    # the tree and the door refused it -- after authority, fetch, manifest
    # and prepare.  Both now ask preflight.admission_estimate, so the band
    # has one answer in it.
    exp = _cyclone()
    machine = _machine()
    lean = pf.admission_estimate(exp, machine=machine).peak_envelope_bytes
    rich = pf.estimate_experiment(
        exp, forcing_intervals=24).peak_envelope_bytes
    assert rich > lean, "the retained-interval term must still move the envelope"

    nodes = st._config_tree_nodes(exp.domains)
    mark_reconstruction_nodes(nodes, exp)
    withheld = st._relocation_rebuild_bytes(nodes, pf.admission_estimate(
        exp, machine=machine))
    # A budget strictly inside the band, expressed as the card that yields it.
    budget = (lean + rich) // 2
    inside = _machine(budget + pf.EXTERNAL_MARGIN_BYTES + withheld)

    review = st.tree_road_plan(exp, machine=inside,
                               resident_estimate=pf.admission_estimate(
                                   exp, machine=inside))
    door_rows = {}
    door = pdtf.cold_tree_streaming_decision(
        exp, st._config_tree_nodes(exp.domains), machine=inside,
        decisions=door_rows)

    assert review.refusal is None and review.report_error is None
    assert [row["road"] for row in review.rows] == ["resident", "resident"]
    assert door is not None and not any(row.stream for row in door_rows.values())
    # And the two judged from the SAME number, not merely to the same verdict.
    assert {row.detail["resident_admission"]["envelope_bytes"]
            for row in door_rows.values()} == {lean}


def test_the_plan_reviews_tree_road_is_priced_from_the_shared_admission_estimate():
    # The review's own surface, not a hand-built call: estimate_phases is
    # what `gpuwm check` and `gpuwm go` price with.
    exp = _cyclone()
    machine = _machine()
    phases = pf.estimate_phases(exp, source=None, machine=machine)
    rows = phases.tree_road.rows
    assert [row["road"] for row in rows] == ["resident", "resident"]

    door_rows = {}
    pdtf.cold_tree_streaming_decision(
        exp, st._config_tree_nodes(exp.domains), machine=machine,
        decisions=door_rows)
    assert ([row["claim_bytes"] for row in rows]
            == [door_rows[int(row["grid_id"])].detail["claim_bytes"]
                for row in rows])


def test_the_cyclone_search_admits_the_requested_layout_unshrunk_as_resident():
    sizing = dw.SizingBudget(FREE / GIB, FREE, None, "fixture", measured=True)
    result = tc.plan_cyclone(cycle=CYCLE, point=POINT, hours=6, tiles="auto",
                             sizing=sizing, target_machine=_machine())
    assert result["kind"] == "configuration"
    assert not result["fitting"]["changed"]
    assert result["fitting"]["proposed_dimensions"] == [[200, 160], [160, 160]]
    assert [[d["nx"], d["ny"]] for d in result["domains"]] == [[200, 160],
                                                               [160, 160]]
    assert result["fitting"]["keeps_coverage"] is None
    # The document says HOW it runs, not only which mode was asked for.
    streaming = result["streaming"]
    assert streaming["mode"] == "auto" and streaming["road"] == "resident"
    assert streaming["budget_bytes"] == BUDGET
    assert [row["grid_id"] for row in streaming["domains"]] == [1, 2]
    assert all(row["road"] == "resident" and row["tile"] is None
               for row in streaming["domains"])
    assert all(row["why"] == "the configured resident envelope fits this budget"
               for row in streaming["domains"])
    assert result["memory"]["peak_envelope_bytes"] <= BUDGET


def test_an_unpriced_tree_road_is_reported_as_unpriced_and_never_as_resident():
    # "resident" with no row behind it is a claim about how the run goes,
    # made where nothing walked.  Say which, and say what said so.
    from types import SimpleNamespace

    refused = SimpleNamespace(rows=(), refusal="no tile fits in 0.50 GiB of VRAM",
                              report_error=None)
    phases = SimpleNamespace(tree_road=refused, peak_envelope_bytes=123)
    entry = tc._streaming_entry(phases, "auto", BUDGET)
    assert entry["road"] is None and entry["domains"] == []
    assert entry["reason"] == "no tile fits in 0.50 GiB of VRAM"

    unpriced = SimpleNamespace(tree_road=None, peak_envelope_bytes=123)
    assert tc._streaming_entry(unpriced, "auto", BUDGET)["road"] is None
    # `off` is an ANSWER, not an absent one, and stays resident.
    off = tc._streaming_entry(unpriced, "off", BUDGET)
    assert off["road"] == "resident" and "mode = 'off'" in off["reason"]


def test_a_tree_the_card_cannot_hold_resident_still_reaches_the_tile_planner():
    # The other direction, and the reason the planner is still there: a tree
    # whose configured envelope exceeds the budget is planned, not refused.
    from dataclasses import replace
    from gpuwm.experiment import DomainConfig
    from tilestream.test_ledger_gate import _exp

    exp = _exp(704, "auto")
    root = exp.root
    run = replace(root.run, grid_id=2, nx=256, ny=256, dx=root.run.dx / 3,
                  dy=root.run.dy / 3, dt=root.run.dt / 3, nested=True,
                  specified=False)
    child = DomainConfig(grid_id=2, parent_id=1, i_parent_start=20,
                         j_parent_start=20, parent_grid_ratio=3,
                         parent_time_step_ratio=3,
                         history_interval_s=root.history_interval_s, run=run)
    exp = replace(exp, domains=(root, child))
    estimate = pf.estimate_experiment(exp)
    machine = _machine(int(13.5 * GIB))
    budget = machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES
    assert estimate.peak_envelope_bytes > budget

    rows = {}
    st.decide_tree(st._config_tree_nodes(exp.domains), exp.tiles,
                   machine=machine, decisions=rows, resident_estimate=estimate)
    assert any(row.stream for row in rows.values())


def test_a_tree_that_fits_neither_road_refuses_in_two_sentences_with_both_numbers():
    exp = _cyclone()
    nodes = st._config_tree_nodes(exp.domains)
    estimate = pf.estimate_experiment(exp)
    budget = 1 * GIB
    machine = _machine(budget + pf.EXTERNAL_MARGIN_BYTES)
    floor = (st._tree_process_overhead_bytes(nodes)
             + st._tree_radiation_transient_bytes(nodes))
    assert estimate.peak_envelope_bytes > budget and floor > budget

    with pytest.raises(st.StreamingRefused) as caught:
        st.decide_tree(nodes, exp.tiles, machine=machine,
                       resident_estimate=estimate)
    text = str(caught.value)
    assert len(_sentences(text)) <= 2, text
    # Both envelopes, the budget, and something to do about it.
    assert str(estimate.peak_envelope_bytes) in text
    assert str(floor) in text and str(budget) in text
    # "both above" is a claim about two numbers, and here both really are.
    assert f"both above the {budget} byte admission budget" in text
    assert "Free VRAM on this card" in text and "reduce the tree" in text
    # Auto is what selects the mode, so it never sends the user back to a flag.
    assert "--tiles off" not in text


def test_a_domain_whose_own_table_says_on_still_reaches_the_tile_planner(
        monkeypatch):
    # ``on`` asks for the tiled road on a domain that would have fitted:
    # a benchmark, a bit-exactness proof.  The tree-wide mode is still
    # auto and the tree still fits, so the resident short-circuit is the
    # thing that must not swallow the nest's own declared preference.
    from dataclasses import replace
    exp = _cyclone()
    nest = replace(exp.domains[1], tiles=st.StreamingOptions(mode="on"))
    exp = replace(exp, domains=(exp.domains[0], nest))
    nodes = st._config_tree_nodes(exp.domains)
    assert st.options_for_domain(exp.domains[1], exp.tiles).mode == "on"
    consulted = []
    original = ap.plan
    monkeypatch.setattr(ap, "plan", lambda *a, **kw: consulted.append(a)
                        or original(*a, **kw))
    with pytest.raises((st.StreamingRefused, ap.CannotPlan)):
        st.decide_tree(nodes, exp.tiles, machine=_machine())
    assert consulted, "mode 'on' must still consult the tile planner"


def test_a_fitting_tree_compelled_by_its_own_table_is_refused_naming_that_table():
    # THE SENTENCE THAT WAS FALSE.  With the nest carrying mode = 'on' the
    # tiled road's floor exceeds the budget while the TREE fits, and the
    # refusal said both numbers were above the budget and offered a bigger
    # card.  What the reader needs is the table that compelled the road.
    from dataclasses import replace
    exp = _cyclone()
    nest = replace(exp.domains[1], tiles=st.StreamingOptions(mode="on"))
    exp = replace(exp, domains=(exp.domains[0], nest))
    nodes = st._config_tree_nodes(exp.domains)
    estimate = pf.estimate_experiment(exp)
    floor = (st._tree_process_overhead_bytes(nodes)
             + st._tree_radiation_transient_bytes(nodes))
    assert estimate.peak_envelope_bytes <= BUDGET < floor

    with pytest.raises(st.StreamingRefused) as caught:
        st.decide_tree(nodes, exp.tiles, machine=_machine(),
                       resident_estimate=estimate)
    text = str(caught.value)
    assert len(_sentences(text)) <= 2, text
    assert "d02 sets [tiles] mode = 'on'" in text
    assert f"{floor} bytes, above the {BUDGET} byte admission budget" in text
    assert "both above" not in text
    assert (f"The configured tree fits resident at "
            f"{estimate.peak_envelope_bytes} bytes against that budget") in text
    assert "delete the [tiles] table on d02 or set its mode to 'auto'" in text
    assert "Free VRAM on this card" not in text


def _refuse_walk(monkeypatch, error):
    def raiser(nodes, options=None, *, machine=None, decisions=None,
               forced_stream=frozenset()):
        raise error

    monkeypatch.setattr(st, "_decide_tree", raiser)


def test_the_last_refusal_keeps_the_planners_sentence_and_names_the_card_for_memory(
        monkeypatch):
    exp = _cyclone()
    nodes = st._config_tree_nodes(exp.domains)
    estimate = pf.estimate_experiment(exp)
    # Past the fixed-floor refusal, so the walk's own exhaustion is what
    # raises, which is the refusal under test.
    monkeypatch.setattr(st, "_tree_process_overhead_bytes", lambda nodes: 1)
    monkeypatch.setattr(st, "_tree_radiation_transient_bytes", lambda nodes: 1)
    _refuse_walk(monkeypatch, st.StreamingRefused(
        "no tile of d02 fits in 0.25 GiB of VRAM", resource="vram"))
    budget = 1 * GIB
    with pytest.raises(st.StreamingRefused) as caught:
        st.decide_tree(nodes, exp.tiles,
                       machine=_machine(budget + pf.EXTERNAL_MARGIN_BYTES),
                       resident_estimate=estimate)
    text = str(caught.value)
    assert caught.value.resource == "memory"
    assert "no tile of d02 fits in 0.25 GiB of VRAM" in text
    assert "Free VRAM on this card" in text


def test_a_geometry_refusal_names_the_tiling_and_the_domain_and_never_vram(
        monkeypatch):
    # A geometry or redundancy refusal and a VRAM refusal produced the same
    # words -- "Free VRAM" -- with resource None, which sent the reader to
    # the card for something no card changes.
    exp = _cyclone()
    nodes = st._config_tree_nodes(exp.domains)
    estimate = pf.estimate_experiment(exp)
    monkeypatch.setattr(st, "_tree_process_overhead_bytes", lambda nodes: 1)
    monkeypatch.setattr(st, "_tree_radiation_transient_bytes", lambda nodes: 1)
    _refuse_walk(monkeypatch, ap.CannotPlan(
        "no tiling divides 160x160 with halo 5", "geometry", {}))
    budget = 1 * GIB
    with pytest.raises(st.StreamingRefused) as caught:
        st.decide_tree(nodes, exp.tiles,
                       machine=_machine(budget + pf.EXTERNAL_MARGIN_BYTES),
                       resident_estimate=estimate)
    text = str(caught.value)
    assert caught.value.resource is None
    assert "no tiling divides 160x160 with halo 5" in text
    assert "refused on its tiling rather than on memory" in text
    assert "d01's tiling geometry, not the card" in text
    assert "Free VRAM" not in text


def test_a_moving_nest_withholds_its_rebuild_and_records_its_pinned_snapshot():
    # The exact-fit admission carried no allowance for the relocation
    # transient.  MEASURED on this very tree: the device pool went
    # 1,611,680,256 -> 1,849,708,032 bytes across the rebuild and
    # transplant, and 141,067,520 bytes of pinned host snapshot appeared in
    # no ledger at all.
    exp = _cyclone()
    estimate = pf.estimate_experiment(exp)
    still = st._config_tree_nodes(exp.domains)
    moving = st._config_tree_nodes(exp.domains)
    mark_reconstruction_nodes(moving, exp)
    assert st._relocating_grid_ids(still) == ()
    assert st._relocating_grid_ids(moving) == (2,)

    rebuild = st._relocation_rebuild_bytes(moving, estimate)
    snapshot = st._relocation_host_snapshot_bytes(moving, estimate)
    # Priced from the nest's own state arrays, and covering what was measured.
    assert rebuild >= 238_027_776
    assert snapshot == 141_067_520

    rows = {}
    result = st.decide_tree(moving, exp.tiles, machine=_machine(),
                            decisions=rows, resident_estimate=estimate)
    assert result.total_budget_bytes == BUDGET - rebuild
    assert result.host_spent_bytes == snapshot
    admission = rows[2].detail["resident_admission"]
    assert admission["withheld_bytes"] == rebuild
    assert admission["relocation_host_snapshot_bytes"] == snapshot
    assert "d02 moves" in admission["withheld_basis"]
    # THE RUN'S OWN RECEIPT carries both, because that is where a move's
    # cost was recorded nowhere at all.
    receipt = st.streaming_receipt(exp.tiles, rows)
    assert receipt["relocation"]["rebuild_withheld_from_budget_bytes"] == rebuild
    assert receipt["relocation"]["pinned_host_snapshot_bytes"] == snapshot
    assert "d02 moves" in receipt["relocation"]["basis"]
    # A still tree keeps the budget it always had, and its receipt keeps
    # the shape it had before this term existed.
    still_rows = {}
    assert st.decide_tree(still, exp.tiles, machine=_machine(),
                          decisions=still_rows,
                          resident_estimate=estimate).total_budget_bytes == BUDGET
    assert "relocation" not in st.streaming_receipt(exp.tiles, still_rows)


def test_the_resident_admission_records_the_acoustic_envelope_of_an_adaptive_clock():
    # decide() attaches it on the single-domain road; the tree's resident
    # short-circuit dropped it, so an adaptive-dt tree kept no record of
    # the sound-step ceiling its halo was sized for.
    from dataclasses import replace
    exp = _cyclone()
    domains = tuple(replace(dc, run=replace(dc.run, use_adaptive_time_step=True))
                    for dc in exp.domains)
    exp = replace(exp, domains=domains)
    rows = {}
    st.decide_tree(st._config_tree_nodes(exp.domains), exp.tiles,
                   machine=_machine(), decisions=rows,
                   resident_estimate=pf.estimate_experiment(exp))
    for row in rows.values():
        envelope = row.detail["acoustic_envelope"]
        assert envelope["maximum_sound_steps"] >= 1
        assert envelope["halo_cells"] >= 1
        assert envelope["geometry_status"]


def test_the_tiles_on_door_names_the_mode_that_compels_the_tiled_road():
    # The neighbouring door told a `--tiles on` reader to re-run with
    # `--tiles off` and blamed the computer for a tree the computer holds.
    admitted = {"tiles": "off", "dimensions": [[200, 160], [160, 160]],
                "peak_envelope_bytes": 3_950_025_504, "budget_bytes": BUDGET}
    on = tc._keeps_coverage_sentence(admitted, "on")
    assert "--tiles on is what compels the tiled road here, not the computer" in on
    assert "re-run with --tiles auto" in on
    assert "--tiles off" not in on
    auto = tc._keeps_coverage_sentence(admitted, "auto")
    assert "re-run with --tiles off" in auto
