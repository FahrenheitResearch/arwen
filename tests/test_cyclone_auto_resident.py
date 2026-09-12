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
    # THE PAYLOAD CARRIES THE MODE IT WAS PRICED IN, and the sentence names
    # that mode: the on branch is priced through `auto`, which is what it
    # recommends, and the auto branch through `off`.
    dims = [[200, 160], [160, 160]]
    on = tc._keeps_coverage_sentence(
        {"tiles": "auto", "dimensions": dims,
         "peak_envelope_bytes": 5_141_378_237, "budget_bytes": BUDGET}, "on")
    assert "--tiles on is what compels the tiled road here, not the computer" in on
    assert "re-run with --tiles auto" in on
    assert "--tiles off" not in on
    auto = tc._keeps_coverage_sentence(
        {"tiles": "off", "dimensions": dims,
         "peak_envelope_bytes": 5_149_977_376, "budget_bytes": BUDGET}, "auto")
    assert "re-run with --tiles off" in auto
    assert tc._recommended_mode("on") == "auto"
    assert tc._recommended_mode("auto") == "off"


# ---------------------------------------------------------------------------
# ONE ADMISSION PER RUN, ONE ENVELOPE PER CARD, AND REFUSALS THAT DESCRIBE
# THE REFUSAL THEY CAME FROM.  Six defects found on re-review, each of them
# a second answer to a question the surrounding code had already answered
# once: a second admission at build time, a second way to pass the device
# profile, a floor quoted as a cause where the floor was under the budget,
# a budget printed net of a withholding nobody named, a door pricing one
# tile mode and recommending another, and a refusal classified by a flag an
# earlier attempt set rather than by what actually refused.
# ---------------------------------------------------------------------------


class _Model:
    """The live tree as :func:`steppers_for_tree` reads it: nodes and states.

    Deliberately NOT the nodes the door decided on -- the door decides on
    planning nodes before a GPU state exists -- so that a build pass which
    carried decisions by identity rather than by grid id fails here.
    """

    def __init__(self, exp):
        self._nodes = st._config_tree_nodes(exp.domains)
        for node in self._nodes:
            node.state = object()
        self._declared_experiment = exp

    def walk_parent_first(self):
        return list(self._nodes)


def _band_machine(exp):
    """A card in the band the two admissions used to disagree across."""
    machine = _machine()
    lean = pf.admission_estimate(exp, machine=machine).peak_envelope_bytes
    rich = pf.estimate_experiment(exp, forcing_intervals=24).peak_envelope_bytes
    assert rich > lean, "the retained-interval term must still move the envelope"
    nodes = st._config_tree_nodes(exp.domains)
    mark_reconstruction_nodes(nodes, exp)
    withheld = st._relocation_rebuild_bytes(
        nodes, pf.admission_estimate(exp, machine=machine))
    assert withheld > 0, "this tree must move a nest for the band to exist"
    budget = (lean + rich) // 2
    return _machine(budget + pf.EXTERNAL_MARGIN_BYTES + withheld), lean, withheld


def test_the_build_pass_consumes_the_doors_admission_and_takes_no_second_one(
        monkeypatch):
    # BL-1.  The door admitted from preflight.admission_estimate against the
    # cold planning machine; the build pass then ran decide_tree AGAIN with
    # the run's own richer ledger estimate, after authority, fetch, manifest
    # and prepare.  Two admissions, two envelopes, two budgets -- and the
    # second one's relocation marking came off the model rather than off the
    # declared experiment, so its budget could carry no withholding where
    # the first withheld one.
    exp = _cyclone()
    machine, lean, withheld = _band_machine(exp)

    door_rows = {}
    door = pdtf.cold_tree_streaming_decision(
        exp, st._config_tree_nodes(exp.domains), machine=machine,
        decisions=door_rows)
    assert door is not None
    assert door.total_budget_bytes == (machine.vram_bytes
                                       - pf.EXTERNAL_MARGIN_BYTES - withheld)
    assert not any(row.stream for row in door_rows.values())

    def _second_admission(*args, **kwargs):
        raise AssertionError("a second admission was taken at build time")

    monkeypatch.setattr(st, "decide_tree", _second_admission)
    monkeypatch.setattr(pf, "admission_estimate", _second_admission)
    built = []
    monkeypatch.setattr(st, "make_stepper",
                        lambda state, cfg, options, **kw:
                        built.append((cfg, kw["decision"])) or object())

    seen = {}
    out = st.steppers_for_tree(_Model(exp), exp.tiles, decisions=seen,
                               machine=machine, tree_decision=door)
    # Nothing streams on a tree this card holds, so no stepper is returned...
    assert out == {}
    # ...but every domain was BUILT from the door's own decision object.
    assert ([decision for _cfg, decision in built]
            == [entry[4] for entry in door.decided])
    assert all(built[i][1] is door.decided[i][4] for i in range(len(built)))
    # And the receipt the run publishes is the door's verdict, unchanged.
    assert ({gid: row.reason for gid, row in seen.items()}
            == {gid: row.reason for gid, row in door_rows.items()})
    assert {row.detail["resident_admission"]["envelope_bytes"]
            for row in seen.values()} == {lean}
    assert {row.detail["resident_admission"]["withheld_bytes"]
            for row in seen.values()} == {withheld}


def test_a_grid_the_admission_never_saw_is_refused_rather_than_decided_late():
    # The other half of consuming a decision: a build pass handed a decision
    # that does not cover the tree must not quietly plan the remainder.
    exp = _cyclone()
    machine = _machine()
    door = pdtf.cold_tree_streaming_decision(
        exp, st._config_tree_nodes(exp.domains), machine=machine)
    partial = st.TreeDecision(
        door.decided[:1], door.priced, door.process_overhead_bytes,
        door.radiation_transient_bytes, door.total_budget_bytes,
        door.vram_spent_bytes, door.host_spent_bytes, door.host_budget_bytes)
    with pytest.raises(st.StreamingRefused) as caught:
        st.steppers_for_tree(_Model(exp), exp.tiles, machine=machine,
                             tree_decision=partial)
    text = str(caught.value)
    assert "d02 is in this run's domain tree but not in the admission" in text
    assert "Decide the whole tree at the door" in text


def test_the_review_machine_and_the_run_doors_machine_price_one_envelope(
        monkeypatch):
    # BL-2.  admission_estimate took a `profile` that defaulted to the
    # machine's, so the review passed its own and the door passed none and
    # took machine.device_profile -- one function, two device terms, and a
    # band of budgets where the review admits what the door refuses.
    import inspect

    assert "profile" not in inspect.signature(pf.admission_estimate).parameters

    exp = _cyclone()
    card = pf.DeviceLocalMemoryProfile("fixture card", 68, 1536, 1024)
    monkeypatch.setattr(st, "_host_total_bytes", lambda: 256 * GIB)
    monkeypatch.setattr(ap.Machine, "detect", classmethod(
        lambda cls, **kw: cls(FREE, 256 * GIB, name="fixture card",
                              device_profile=card)))

    # The RUN DOOR's machine, built the way the door builds it.
    door = st.cold_planning_machine(exp)
    # The REVIEW's machine, built the way `gpuwm check` and `gpuwm go`
    # build it: from a figure already read, carrying the profile already
    # read beside it.
    review = st.planner_machine(vram_bytes=FREE, name="gpuwm check budget",
                                device_profile=card)
    assert door.device_profile is card and review.device_profile is card
    assert (pf.admission_estimate(exp, machine=review).peak_envelope_bytes
            == pf.admission_estimate(exp, machine=door).peak_envelope_bytes)

    # NOT A VACUOUS EQUALITY: the field the two sides used to differ on is
    # the field that moves the envelope, so an arm that dropped it could
    # not have come back equal.
    bare = st.planner_machine(vram_bytes=FREE, name="gpuwm check budget")
    assert bare.device_profile is None
    assert (pf.admission_estimate(exp, machine=bare).peak_envelope_bytes
            != pf.admission_estimate(exp, machine=review).peak_envelope_bytes)


def test_the_go_gate_builds_its_planner_machine_from_both_halves_of_its_probe(
        monkeypatch):
    # The review caller that actually holds a probe: its device half must
    # reach the machine, because that is where the admission reads it.
    from gpuwm import go_cli

    monkeypatch.setattr(st, "_host_total_bytes", lambda: 256 * GIB)
    probe = {"free_bytes": FREE,
             "profile": {"name": "fixture card", "multiprocessor_count": 68,
                         "max_threads_per_multiprocessor": 1536,
                         "default_stack_limit_bytes": 1024}}
    profile = pf.profile_from_device_probe(probe)
    assert profile is not None
    machine = go_cli._planner_machine(probe, profile)
    assert machine.vram_bytes == FREE and machine.device_profile is profile


def _compelled(tiles_on_nest=True):
    """A tree driven onto the tiled road, by the nest's table or the tree's.

    The tree-wide arm keeps ONE auto domain, because a tree with no auto
    domain at all never reaches the resident question: the compelled road
    is the only road there is, and the refusal under test is the one a
    reader gets when a fitting tree is compelled past a resident answer.
    """
    from dataclasses import replace
    exp = _cyclone()
    if tiles_on_nest:
        nest = replace(exp.domains[1], tiles=st.StreamingOptions(mode="on"))
        return replace(exp, domains=(exp.domains[0], nest))
    nest = replace(exp.domains[1], tiles=replace(exp.tiles, mode="auto"))
    return replace(exp, tiles=replace(exp.tiles, mode="on"),
                   domains=(exp.domains[0], nest))


def test_a_compelled_fitting_tree_refused_on_vram_never_quotes_a_floor_it_cleared(
        monkeypatch):
    # BL-3.  With the nest on "on", a card whose fixed floor is UNDER the
    # budget, and the walk refusing on vram, the refusal read "no streamed
    # road fits either, the tile planner's floor alone being F bytes,
    # against a B byte admission budget" and then, one sentence later,
    # "The configured tree fits resident at N bytes against that budget".
    # It quoted a floor it had already cleared as the reason, and refuted
    # its own first half in its second.
    exp = _compelled()
    nodes = st._config_tree_nodes(exp.domains)
    estimate = pf.estimate_experiment(exp)
    assert estimate.peak_envelope_bytes <= BUDGET
    monkeypatch.setattr(st, "_tree_process_overhead_bytes", lambda nodes: 1)
    monkeypatch.setattr(st, "_tree_radiation_transient_bytes", lambda nodes: 1)
    _refuse_walk(monkeypatch, st.StreamingRefused(
        "no tile of d02 fits in 0.25 GiB of VRAM", resource="vram"))

    with pytest.raises(st.StreamingRefused) as caught:
        st.decide_tree(nodes, exp.tiles, machine=_machine(),
                       resident_estimate=estimate)
    text = str(caught.value)
    assert caught.value.resource == "memory"
    assert len(_sentences(text)) <= 2, text
    # The cause is the road the table compelled...
    assert "d02 sets [tiles] mode = 'on', which compels the tiled road" in text
    assert "no tiling of that road fits" in text
    assert "no tile of d02 fits in 0.25 GiB of VRAM" in text
    # ...and the floor, which this card clears, is not quoted as anything.
    assert "floor" not in text
    assert "no streamed road fits either" not in text
    # The way out stays the table, and the tree's own figure stands.
    assert (f"The configured tree fits resident at "
            f"{estimate.peak_envelope_bytes} bytes against that budget") in text
    assert "delete the [tiles] table on d02" in text
    assert "Free VRAM on this card" not in text


def test_a_tree_wide_tiles_table_is_named_where_it_lives_not_on_a_domain():
    # ADVISORY.  `mode = "on"` written once on the tree reaches every
    # domain, and the refusal told the reader to "delete the [tiles] table
    # on d01" -- a table that does not exist on d01 at all.
    exp = _compelled(tiles_on_nest=False)
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
    assert "the tree-wide [tiles] table sets mode = 'on' for d01" in text
    assert ("delete the tree-wide [tiles] table that d01 takes that mode "
            "from or set its mode to 'auto'") in text
    assert "[tiles] table on d01" not in text


def _withholding_machine(exp):
    """A card whose UNWITHHELD budget holds the tree and whose net one does not."""
    machine = _machine()
    estimate = pf.admission_estimate(exp, machine=machine)
    nodes = st._config_tree_nodes(exp.domains)
    mark_reconstruction_nodes(nodes, exp)
    withheld = st._relocation_rebuild_bytes(nodes, estimate)
    envelope = int(estimate.peak_envelope_bytes)
    # budget = free - margin - withheld, so free - margin == envelope puts
    # the tree exactly one byte the wrong side of the net budget.
    return (_machine(envelope + pf.EXTERNAL_MARGIN_BYTES), nodes, envelope,
            withheld)


def test_a_refusal_whose_budget_withholds_a_move_names_the_move_and_the_way_out():
    # BL-4.  The budget is already net of a moving nest's rebuild, and the
    # refusal printed the net figure as "the N byte admission budget" with
    # nothing saying why free VRAM minus the external margin did not equal
    # N -- then sent the reader to free VRAM or shrink the tree, when the
    # tree fits and the MOVE is what does not.
    exp = _cyclone()
    machine, nodes, envelope, withheld = _withholding_machine(exp)
    budget = machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES - withheld
    assert budget < envelope <= budget + withheld

    with pytest.raises(st.StreamingRefused) as caught:
        st.decide_tree(nodes, exp.tiles, machine=machine,
                       resident_estimate=pf.admission_estimate(
                           exp, machine=machine))
    text = str(caught.value)
    assert len(_sentences(text)) <= 2, text
    assert (f"the {budget} byte admission budget, which withholds "
            f"{withheld} bytes for d02's rebuild") in text
    assert (f"The configured tree fits resident at {envelope} bytes before "
            "the withholding") in text
    assert "what this card cannot hold is d02's move, not the tree" in text
    assert "hold it still" in text
    assert "Free VRAM on this card" not in text


def test_a_refusal_on_a_budget_that_withholds_nothing_keeps_its_plain_wording():
    # The control: a still tree's refusal must not grow a withholding
    # clause, so the clause above is evidence of a move and not decoration.
    exp = _cyclone()
    nodes = st._config_tree_nodes(exp.domains)
    budget = 1 * GIB
    with pytest.raises(st.StreamingRefused) as caught:
        st.decide_tree(nodes, exp.tiles,
                       machine=_machine(budget + pf.EXTERNAL_MARGIN_BYTES),
                       resident_estimate=pf.estimate_experiment(exp))
    text = str(caught.value)
    assert f"both above the {budget} byte admission budget." in text
    assert "withholds" not in text


def test_the_resident_road_weighs_its_pinned_host_copy_against_the_allowance():
    # ADVISORY.  Only the tiled road weighed host bytes.  A moving nest
    # stages its outgoing state through a PINNED host copy on either road,
    # and a page-locked allocation past the allowance does not degrade --
    # it fails, at the first move, in a run this walk had admitted.
    exp = _cyclone()
    nodes = st._config_tree_nodes(exp.domains)
    mark_reconstruction_nodes(nodes, exp)
    estimate = pf.estimate_experiment(exp)
    snapshot = st._relocation_host_snapshot_bytes(nodes, estimate)
    assert snapshot > 0
    # A host whose page-lockable share is one byte under the snapshot.
    from tilestream.autoplan import PINNED_FRACTION
    machine = ap.Machine(FREE, int((snapshot - 1) / PINNED_FRACTION))
    assert machine.host_budget_bytes < snapshot

    with pytest.raises(st.StreamingRefused) as caught:
        st.decide_tree(nodes, exp.tiles, machine=machine,
                       resident_estimate=estimate)
    text = str(caught.value)
    assert caught.value.resource == "host"
    assert (f"the pinned host copy its outgoing state is staged through "
            f"needs {snapshot} bytes") in text
    assert f"{machine.host_budget_bytes} byte page-lockable host allowance" in text
    assert "hold the nest still" in text
    # And a host with room admits, so the refusal is the allowance's doing.
    assert st.decide_tree(nodes, exp.tiles, machine=_machine(),
                          resident_estimate=estimate).host_spent_bytes == snapshot


def test_the_final_refusal_is_classified_by_what_refused_last_not_by_a_flag(
        monkeypatch):
    # BL-6.  `non_memory_refusal` was set on any non-memory refusal and
    # never cleared, so a geometry refusal on attempt 1 made a later VRAM
    # refusal read as a tiling one with resource None -- sending the reader
    # after cell counts for a card that was out of memory.
    exp = _cyclone()
    nodes = st._config_tree_nodes(exp.domains)
    estimate = pf.estimate_experiment(exp)
    monkeypatch.setattr(st, "_tree_process_overhead_bytes", lambda nodes: 1)
    monkeypatch.setattr(st, "_tree_radiation_transient_bytes", lambda nodes: 1)
    raised = []

    def raiser(nodes, options=None, *, machine=None, decisions=None,
               forced_stream=frozenset()):
        raised.append(len(raised))
        if len(raised) == 1:
            raise ap.CannotPlan("no tiling divides 160x160 with halo 5",
                                "geometry", {})
        raise st.StreamingRefused("no tile of d02 fits in 0.25 GiB of VRAM",
                                  resource="vram")

    monkeypatch.setattr(st, "_decide_tree", raiser)
    budget = 1 * GIB
    with pytest.raises(st.StreamingRefused) as caught:
        st.decide_tree(nodes, exp.tiles,
                       machine=_machine(budget + pf.EXTERNAL_MARGIN_BYTES),
                       resident_estimate=estimate)
    assert len(raised) > 1, "the later attempts must have run"
    text = str(caught.value)
    assert caught.value.resource == "memory"
    assert "no tile of d02 fits in 0.25 GiB of VRAM" in text
    assert "refused on its tiling rather than on memory" not in text
    assert "tiling geometry, not the card" not in text
    assert "Free VRAM on this card" in text


def test_a_search_that_ends_on_its_own_arithmetic_is_a_memory_refusal_after_a_geometry_attempt(
        monkeypatch):
    # The other half of the classification: after a geometry refusal on
    # attempt 1, later attempts that PLAN but come back over budget are
    # rejected by this function's own arithmetic, not by the planner, and
    # the final refusal must say memory, not tiling.
    exp = _cyclone()
    nodes = st._config_tree_nodes(exp.domains)
    estimate = pf.estimate_experiment(exp)
    monkeypatch.setattr(st, "_tree_process_overhead_bytes", lambda nodes: 1)
    monkeypatch.setattr(st, "_tree_radiation_transient_bytes", lambda nodes: 1)
    calls = []

    def walk(nodes, options=None, *, machine=None, decisions=None,
             forced_stream=frozenset()):
        calls.append(len(calls))
        if len(calls) == 1:
            raise ap.CannotPlan("no tiling divides 160x160 with halo 5",
                                "geometry", {})
        decided = []
        for node in nodes:
            row = st.StreamingDecision(
                True, "stub", 32, 32, 2, 5, "host", "ring",
                detail={"claim_bytes": 0, "corridor_claim_bytes": 0})
            decisions[int(node.cfg.grid_id)] = row
            decided.append((node, node.cfg, None, machine, row))
        return st.TreeDecision(decided, True, 1, 1, int(machine.vram_bytes),
                               10 ** 12, 0, None)

    monkeypatch.setattr(st, "_decide_tree", walk)
    budget = 1 * GIB
    with pytest.raises(st.StreamingRefused) as caught:
        st.decide_tree(nodes, exp.tiles,
                       machine=_machine(budget + pf.EXTERNAL_MARGIN_BYTES),
                       resident_estimate=estimate)
    assert len(calls) > 1, "the later attempts must have planned"
    text = str(caught.value)
    assert caught.value.resource == "memory"
    assert "refused on its tiling rather than on memory" not in text
    assert "tiling geometry, not the card" not in text
    assert "no tiling divides 160x160" not in text


def test_a_compelled_table_beside_a_withholding_gets_both_ways_out_and_no_resident_promise(
        monkeypatch):
    # A remedy that said "hold the nest still, and the tree runs resident"
    # while d02's own table said mode = 'on' promised a road that table
    # forbids.  Both facts have to be named, and nothing promised.
    exp = _compelled()
    machine, nodes, envelope, withheld = _withholding_machine(exp)
    assert withheld > 0
    monkeypatch.setattr(st, "_tree_process_overhead_bytes", lambda nodes: 10 ** 12)
    monkeypatch.setattr(st, "_tree_radiation_transient_bytes", lambda nodes: 1)
    _refuse_walk(monkeypatch, st.StreamingRefused(
        "no tile of d02 fits in 0.25 GiB of VRAM", resource="vram"))
    with pytest.raises(st.StreamingRefused) as caught:
        st.decide_tree(nodes, exp.tiles, machine=machine,
                       resident_estimate=pf.admission_estimate(exp, machine=machine))
    text = str(caught.value)
    assert "and the tree runs resident" not in text
    assert "compels the tiled road" in text
    assert "give that nest a smaller grid or hold it still" in text
    assert "set its mode to 'auto'" in text
    assert len(_sentences(text)) <= 2, text


def test_the_tile_search_plans_on_the_budget_the_withholding_already_reduced(
        monkeypatch):
    # ADVISORY.  _resident_admission said the withholding came off the
    # budget so that "every comparison downstream, the tile search
    # included" spent the reduced allowance.  The first search attempt
    # planned on the machine's full VRAM instead.
    exp = _cyclone()
    machine, nodes, envelope, withheld = _withholding_machine(exp)
    budget = machine.vram_bytes - pf.EXTERNAL_MARGIN_BYTES - withheld
    monkeypatch.setattr(st, "_tree_process_overhead_bytes", lambda nodes: 1)
    monkeypatch.setattr(st, "_tree_radiation_transient_bytes", lambda nodes: 1)
    seen = []

    def raiser(nodes, options=None, *, machine=None, decisions=None,
               forced_stream=frozenset()):
        seen.append(int(machine.vram_bytes))
        raise st.StreamingRefused("no tile fits", resource="vram")

    monkeypatch.setattr(st, "_decide_tree", raiser)
    with pytest.raises(st.StreamingRefused):
        st.decide_tree(nodes, exp.tiles, machine=machine,
                       resident_estimate=pf.admission_estimate(
                           exp, machine=machine))
    assert seen, "the tile search must have run"
    assert set(seen) == {budget}
    assert machine.vram_bytes not in seen


def test_the_tiles_on_door_prices_the_mode_it_recommends():
    # BL-5.  The `--tiles on` door recommends `--tiles auto`, and priced
    # that recommendation through the `--tiles off` route, which never
    # withholds a moving nest's rebuild.  In the band between the withheld
    # and unwithheld budgets the door recommended auto and auto refused.
    intent = dict(cycle=CYCLE, point=POINT, hours=6,
                  name="GFS cyclone 12 km to 3 km", tiles="on",
                  source="cyclone-setup.toml")

    def admission(free_bytes):
        sizing = dw.SizingBudget(free_bytes / GIB, free_bytes, None,
                                 "fixture", measured=True)
        operands = dict(free_bytes=sizing.free_bytes, vram_gib=sizing.vram_gib,
                        profile=sizing.device_profile,
                        forcing_interval_seconds=10800.)
        machine = _machine(free_bytes)
        return tc._unreduced_resident_admission(
            intent,
            lambda exp: dw.sizing_budget_bytes(exp, **operands),
            lambda exp: dw._sizing_phases(exp, source="gfs", machine=machine,
                                          **operands))

    # IN THE BAND: `--tiles off` admits this tree and `--tiles auto` does
    # not, so there is no recommendation to make and the door makes none.
    inside = int(5.4 * GIB)
    _text, off_exp = tc.configuration_text(**{**intent, "tiles": "off"})
    off_operands = dict(free_bytes=inside, vram_gib=inside / GIB, profile=None,
                        forcing_interval_seconds=10800.)
    off = dw._sizing_phases(off_exp, source="gfs", machine=_machine(inside),
                            **off_operands)
    assert off.peak_envelope_bytes <= dw.sizing_budget_bytes(
        off_exp, **off_operands), "the band needs --tiles off to admit here"
    assert admission(inside) is None

    # ABOVE THE BAND: auto admits the whole tree resident, and the door
    # says so in the mode it priced.
    above = int(6.5 * GIB)
    admitted = admission(above)
    assert admitted is not None and admitted["tiles"] == "auto"
    sentence = tc._keeps_coverage_sentence(admitted, "on")
    assert "--tiles auto admits" in sentence
    assert "re-run with --tiles auto" in sentence
    assert "--tiles off" not in sentence


def test_the_tiles_auto_door_still_prices_and_names_tiles_off():
    # The other branch is unchanged: under `auto` there is no mode to
    # withdraw, `off` is the way to keep the ground, and `off` is what is
    # priced.
    intent = dict(cycle=CYCLE, point=POINT, hours=6,
                  name="GFS cyclone 12 km to 3 km", tiles="auto",
                  source="cyclone-setup.toml")
    free = int(6.5 * GIB)
    sizing = dw.SizingBudget(free / GIB, free, None, "fixture", measured=True)
    operands = dict(free_bytes=sizing.free_bytes, vram_gib=sizing.vram_gib,
                    profile=sizing.device_profile,
                    forcing_interval_seconds=10800.)
    admitted = tc._unreduced_resident_admission(
        intent,
        lambda exp: dw.sizing_budget_bytes(exp, **operands),
        lambda exp: dw._sizing_phases(exp, source="gfs",
                                      machine=_machine(free), **operands))
    assert admitted is not None and admitted["tiles"] == "off"
    assert "re-run with --tiles off" in tc._keeps_coverage_sentence(
        admitted, "auto")
