"""One admission per run, on every route, and the advisories around it.

Each test here pins a place where two surfaces answered one question
twice.  The headline is ``gpuwm run``'s tree route: it was the last door
that priced its ``[tiles]`` admission from the run's own memory ledger,
at build time, after the whole case had been fetched -- while
``gpuwm check`` priced the same tree from
:func:`gpuwm.core.preflight.admission_estimate` before anything was
spent.  MEASURED on the 12/3 km moving-nest cyclone tree, the two
estimates differ by hundreds of megabytes, so there is a band of budgets
in which the review admitted the tree and the run then refused it.

The rest are the advisories the 2.7.3 reviews recorded against the same
seam: a catalog spawned once per plan review, a machine built twice, a
moving subtree the build pass stopped knowing about, a budget the cyclone
door quoted unwithheld, and a dead render's working stores.
"""
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import cyclone_setup as tc
from gpuwm import downscale_pricing, go_cli, render, runplan, runtime
from gpuwm.core import preflight as pf, streaming as st
from gpuwm.core.streamed_relocation import mark_reconstruction_nodes
from tilestream import autoplan as ap

GIB = 1024 ** 3
#: The desktop budget the 12/3 km cyclone refusal was raised on.
BUDGET = 6_855_065_600
FREE = BUDGET + pf.EXTERNAL_MARGIN_BYTES
POINT = (16.38581807563628, -123.76623740203063)
CYCLE = "2026091112"


def _cyclone(tiles="auto"):
    _text, exp = tc.configuration_text(cycle=CYCLE, point=POINT, hours=6,
                                       tiles=tiles)
    return exp


def _machine(free_bytes=FREE):
    return ap.Machine(int(free_bytes), 256 * GIB)


def _in_band_machine(exp):
    """A card whose budget sits between the two estimates that disagreed.

    At this budget the review's lean admission estimate admits the tree;
    the run route's old ledger estimate, priced with the cache's retained
    forcing intervals, does not.
    """
    machine = _machine()
    lean = pf.admission_estimate(exp, machine=machine).peak_envelope_bytes
    rich = pf.estimate_experiment(exp, forcing_intervals=24).peak_envelope_bytes
    assert rich > lean, "the retained-interval term must still move the envelope"
    nodes = st._config_tree_nodes(exp.domains)
    mark_reconstruction_nodes(nodes, exp)
    withheld = st._relocation_rebuild_bytes(
        nodes, pf.admission_estimate(exp, machine=machine))
    budget = (lean + rich) // 2
    return _machine(budget + pf.EXTERNAL_MARGIN_BYTES + withheld), lean


class _StopBeforeFetch(Exception):
    """Raised in place of the build pass, which is where the fetch begins."""


def test_the_run_route_takes_its_admission_from_the_reviews_own_function(
        monkeypatch, tmp_path):
    """``gpuwm run``'s tree route asks the shared question, before the fetch.

    Red on the base: the route reached ``build_experiment`` -- the call
    that fetches, decodes and ingests the case -- with no admission taken
    at all, and decided the tree's roads afterwards from
    ``model.memory_ledger.estimate``.
    """
    exp = _cyclone()
    machine, lean = _in_band_machine(exp)

    seen: dict = {}

    # The SHARED pricing function, which both trees have: what is red on
    # the base is that the run route never reaches it before the fetch.
    real_estimate = pf.admission_estimate

    def priced(exp_arg, *, machine=None):
        estimate = real_estimate(exp_arg, machine=machine)
        seen.setdefault("machine", machine)
        seen.setdefault("estimate", estimate)
        return estimate

    monkeypatch.setattr(pf, "admission_estimate", priced)

    real = getattr(st, "cold_tree_streaming_decision", None)

    def spy(exp_arg, nodes, *, machine=None, decisions=None):
        outcome = real(exp_arg, nodes, machine=machine, decisions=decisions)
        seen["decision"] = outcome
        seen["rows"] = dict(decisions or {})
        return outcome

    if real is not None:
        monkeypatch.setattr(st, "cold_tree_streaming_decision", spy)
    monkeypatch.setattr(st, "cold_planning_machine", lambda _exp: machine)

    import gpuwm.core.model as core_model

    def refuse_to_build(*args, **kwargs):
        raise _StopBeforeFetch("the fetch must not happen before the admission")

    monkeypatch.setattr(core_model, "build_experiment", refuse_to_build)

    with pytest.raises(_StopBeforeFetch):
        runtime.run_experiment(exp, SimpleNamespace(), tmp_path / "out")

    assert seen, "the run route took no admission before the fetch"
    assert seen["machine"] is machine
    assert seen["decision"] is not None

    # THE EQUALITY.  Same tree, same machine, same band: the review's road
    # and the run route's road are the same road, priced from the same
    # envelope against the same budget.
    review = st.tree_road_plan(
        exp, machine=machine,
        resident_estimate=pf.admission_estimate(exp, machine=machine))
    assert review.refusal is None and review.report_error is None
    rows = seen["rows"]
    assert ([row["road"] for row in review.rows]
            == [("streamed" if rows[int(row["grid_id"])].stream else "resident")
                for row in review.rows])
    assert ([row["claim_bytes"] for row in review.rows]
            == [rows[int(row["grid_id"])].detail["claim_bytes"]
                for row in review.rows])
    assert {row.detail["resident_admission"]["envelope_bytes"]
            for row in rows.values()} == {lean}
    assert review.total_budget_bytes == seen["decision"].total_budget_bytes


def test_the_run_route_carries_that_decision_into_the_build_pass(monkeypatch,
                                                                tmp_path):
    """The road is DECIDED once and CONSUMED once, never asked twice.

    Red on the base: the build pass took a SECOND admission of its own,
    from the run's memory ledger, and this route handed it no decision at
    all -- so the road the user was shown and the road the run walked
    were two answers to one question.
    """
    exp = _cyclone()
    machine, _lean = _in_band_machine(exp)
    monkeypatch.setattr(st, "cold_planning_machine", lambda _exp: machine)

    taken = []
    rows: dict = {}
    real = getattr(st, "cold_tree_streaming_decision", None)

    def spy(exp_arg, nodes, *, machine=None, decisions=None):
        outcome = real(exp_arg, nodes, machine=machine, decisions=decisions)
        taken.append(outcome)
        rows.update(decisions or {})
        return outcome

    if real is not None:
        monkeypatch.setattr(st, "cold_tree_streaming_decision", spy)

    handed = {}

    def capture(model, options=None, **kwargs):
        handed.update(kwargs)
        raise _StopBeforeFetch("stop at the build pass")

    monkeypatch.setattr(st, "steppers_for_tree", capture)
    monkeypatch.setattr(st, "builders_for_tree", lambda *a, **k: {})

    ledger_estimate = pf.admission_estimate(exp, machine=machine)
    model = _built_model_stub(monkeypatch, exp, ledger_estimate)
    import gpuwm.core.model as core_model
    monkeypatch.setattr(core_model, "build_experiment", lambda *a, **k: model)
    monkeypatch.setattr(runtime, "resolved_tree_config_report",
                        lambda *a, **k: "")

    with pytest.raises(_StopBeforeFetch):
        runtime.run_experiment(exp, SimpleNamespace(output_title="fixture"),
                               tmp_path / "out")

    # THE HAND-OFF.  The decision the door took is the decision the build
    # pass consumes, and the run's own ledger estimate is not a second
    # admission behind it.
    assert taken, "the door took no admission"
    assert handed.get("tree_decision") is taken[0]
    assert handed.get("resident_estimate") is None
    assert handed.get("machine") is machine
    # The receipt source is the door's own rows, not a fresh walk.
    assert rows, "the door recorded no per-grid road"
    assert handed.get("decisions") == rows


def _built_model_stub(monkeypatch, exp, ledger_estimate):
    """A stand-in for the built model, complete enough to be RUN.

    ``_run_built_experiment`` reads a live tree, its prepared cases and
    its writers before it ever reaches the ``[tiles]`` pass, so a bare
    namespace stops the route early and pins nothing.  Everything stubbed
    here is a stage this test is not about -- relocation runners, the
    lifecycle publisher, the perturbation receipt, the writer set -- and
    what is left real is the pass under test.
    """
    import contextlib

    from gpuwm.io import wrfout as io_wrfout

    live = st._config_tree_nodes(exp.domains)
    model = SimpleNamespace(
        walk_parent_first=lambda: live,
        nodes_by_grid_id={int(node.cfg.grid_id): node for node in live},
        root=live[0],
        _prepared_by_grid_id={int(node.cfg.grid_id):
                              SimpleNamespace(streamed_store=None)
                              for node in live},
        _initial_perturbation_receipts=(),
        _input_catalog=None,
        _declared_experiment=exp,
        memory_ledger=SimpleNamespace(estimate=ledger_estimate),
    )

    monkeypatch.setattr(runtime, "build_real_relocation_runners",
                        lambda *a, **k: None)
    monkeypatch.setattr(runtime, "build_real_spawn_runner",
                        lambda *a, **k: None)
    monkeypatch.setattr(runtime, "publish_lifecycle_runners",
                        lambda *a, **k: None)
    monkeypatch.setattr(runtime, "admit_restart_with_lifecycle",
                        lambda *a, **k: False)
    monkeypatch.setattr(runtime, "_write_initial_perturbation_receipt",
                        lambda *a, **k: None)

    @contextlib.contextmanager
    def writers(*args, **kwargs):
        yield SimpleNamespace(attach_writers=lambda *a, **k: None)

    monkeypatch.setattr(io_wrfout, "PerDomainWrfoutWriters", writers)
    return model


def test_the_build_pass_marks_the_moving_subtree_on_the_live_nodes(monkeypatch):
    """A road that came in from the door still knows which grids MOVE.

    Red on the base: ``steppers_for_tree`` marked the live nodes only on
    the branch that decided for itself, so a run whose admission arrived
    from its door walked a live tree that read as stationary.
    """
    exp = _cyclone()
    machine = _machine()
    door_rows: dict = {}
    # Built through decide_tree rather than through the door, so that what
    # is red on the base is the BUILD PASS's behaviour and not the door's
    # name: both trees have decide_tree, and both take a tree_decision.
    priced = st._config_tree_nodes(exp.domains)
    mark_reconstruction_nodes(priced, exp)
    decision = st.decide_tree(
        priced, exp.tiles, machine=machine, decisions=door_rows,
        resident_estimate=pf.admission_estimate(exp, machine=machine))
    assert decision is not None

    live = st._config_tree_nodes(exp.domains)
    for node in live:
        node.state = object()
    model = SimpleNamespace(walk_parent_first=lambda: live,
                            _declared_experiment=exp)
    monkeypatch.setattr(st, "make_stepper", lambda *a, **k: object())

    st.steppers_for_tree(model, exp.tiles, builders={}, decisions={},
                         machine=machine, tree_decision=decision)

    marked = {int(node.cfg.grid_id) for node in live
              if getattr(node, "_streamed_reconstruction_required", False)}
    reference = st._config_tree_nodes(exp.domains)
    mark_reconstruction_nodes(reference, exp)
    assert marked == {int(node.cfg.grid_id) for node in reference
                      if getattr(node, "_streamed_reconstruction_required",
                                 False)}
    assert marked, "this fixture tree moves a nest; something must be marked"
    assert all(getattr(node, "_reconstruction_p_top", None)
               == pytest.approx(float(exp.vertical.p_top))
               for node in live if int(node.cfg.grid_id) in marked)


def _single_domain_experiment():
    """The cyclone's 12 km root alone: an experiment of ONE domain.

    The moving nest and the ``[relocation]`` table it is configured by go
    with it; a follow source with no nest to move is refused at the front
    door, and the question here is the root's own admission.
    """
    from dataclasses import replace

    from gpuwm.experiment import RelocationConfig

    exp = _cyclone()
    return replace(exp, domains=exp.domains[:1],
                   relocation=RelocationConfig())


def _single_domain_band_machine(exp):
    """A card between the review's estimate and the run's old ledger one.

    MEASURED on this root on an 8 GiB card: 4,749,512,312 bytes on the
    review's side against 4,757,701,968 / 4,806,839,904 / 4,937,874,400 at
    2 / 8 / 24 retained forcing intervals.  Inside that band the review
    admitted the domain resident and the run route refused it.
    """
    probe = _machine()
    options = st.options_for_domain(exp.domains[0], exp.tiles)
    lean = pf.admission_estimate(exp, machine=probe).peak_envelope_bytes
    rich = pf.estimate_experiment(exp, forcing_intervals=24).peak_envelope_bytes
    assert rich > lean, "the retained-interval term must still move the envelope"
    admission = st._resident_admission(
        options, probe, pf.admission_estimate(exp, machine=probe))
    assert admission is not None
    gap = int(probe.vram_bytes) - int(admission["budget_bytes"])
    machine = _machine((lean + rich) // 2 + gap)
    checked = st._resident_admission(
        options, machine, pf.admission_estimate(exp, machine=machine))
    assert lean <= int(checked["budget_bytes"]) < rich, (
        "this fixture must sit inside the band the two estimates disagree in")
    return machine, lean


def test_the_single_domain_run_route_and_the_review_take_one_admission(
        monkeypatch, tmp_path):
    """One domain, one estimate, one budget, and taken before the fetch.

    Red on the base twice over: the run route priced this arm from
    ``estimate_experiment`` with the schedule's retained interval count
    folded in, and it priced it only AFTER ``build_input_catalog`` and
    every forcing snapshot had been decoded -- so inside the band above,
    ``gpuwm check`` admitted the domain resident and the run refused it
    with the download already spent.
    """
    exp = _single_domain_experiment()
    machine, lean = _single_domain_band_machine(exp)
    monkeypatch.setattr(st, "cold_planning_machine", lambda _exp: machine)

    asked: list = []
    real_decide = st.decide

    def spy(cfg, options=None, **kwargs):
        outcome = real_decide(cfg, options, **kwargs)
        asked.append({"estimate": kwargs.get("resident_estimate"),
                      "decision": outcome})
        return outcome

    monkeypatch.setattr(st, "decide", spy)

    # THE REVIEW, through the surface `gpuwm check` prices with.
    from gpuwm import domain_wizard as dw
    phases = dw._sizing_phases(exp, free_bytes=int(machine.vram_bytes),
                               machine=machine, source="gfs")
    assert phases is not None
    review = asked[-1]
    assert review["estimate"] is not None

    # THE RUN ROUTE, stopped where the fetch begins.
    asked.clear()
    import gpuwm.ingest.preflight as ingest_preflight

    def refuse_to_fetch(*args, **kwargs):
        raise _StopBeforeFetch("the admission must be taken before the fetch")

    monkeypatch.setattr(ingest_preflight, "build_input_catalog",
                        refuse_to_fetch)

    with pytest.raises(_StopBeforeFetch):
        runtime.run_experiment(exp, SimpleNamespace(), tmp_path / "out")

    assert asked, "the run route took no admission before the fetch"
    door = asked[-1]

    # THE EQUALITY.
    assert (door["estimate"].peak_envelope_bytes
            == review["estimate"].peak_envelope_bytes == lean)
    assert door["decision"].stream is review["decision"].stream is False
    assert (door["decision"].resident_bytes
            == review["decision"].resident_bytes)
    assert (door["decision"].budget_bytes
            == review["decision"].budget_bytes)


def _pinned_options(exp):
    from dataclasses import replace

    options = st.options_for_domain(exp.domains[0], exp.tiles)
    return replace(options, mode="on", tile_nx=64, tile_ny=64, nbuffers=2)


def test_a_pinned_single_domain_tiling_consults_no_admission_estimate(
        monkeypatch):
    """One guard over the question, and it lives with the question.

    ``decide`` returns on a pinned tiling before it looks at any
    estimate -- the configuration IS the decision -- so pricing one is
    work nobody reads.  Red on the base: the shared admission function
    priced it anyway, while the run door spelled the same condition for
    itself one module away and skipped it, which is two guards over one
    question and the shape this lane exists to remove.
    """
    exp = _single_domain_experiment()
    pinned = _pinned_options(exp)
    machine = _machine()

    asked: list = []

    def priced(exp_arg, *, machine=None):
        asked.append(machine)
        raise AssertionError("a pinned tiling priced an admission estimate")

    monkeypatch.setattr(pf, "admission_estimate", priced)

    decision = st.cold_single_domain_decision(exp, machine=machine,
                                              options=pinned)
    assert decision.stream is True
    assert decision.tile_nx == 64 and decision.tile_ny == 64
    assert asked == [], "a pinned tiling consulted the admission estimate"
    assert st.cold_single_domain_admission(exp, machine=machine,
                                           options=pinned) is None
    off = st.options_for_domain(exp.domains[0], st.OFF)
    assert st.cold_single_domain_admission(exp, machine=machine,
                                           options=off) is None


def test_the_single_domain_run_door_prices_its_admission_once(monkeypatch,
                                                              tmp_path):
    """The estimate it keeps for the streamed walk IS the one it decided on.

    Red on the base: the route took the admission for the refinement pass
    and the decision took a second, identical one of its own, so the
    unpinned single-domain route priced the same envelope twice per run.
    """
    exp = _single_domain_experiment()
    machine, _lean = _single_domain_band_machine(exp)
    monkeypatch.setattr(st, "cold_planning_machine", lambda _exp: machine)

    real_estimate = pf.admission_estimate
    calls: list = []

    def priced(exp_arg, *, machine=None):
        calls.append(machine)
        return real_estimate(exp_arg, machine=machine)

    monkeypatch.setattr(pf, "admission_estimate", priced)

    import gpuwm.ingest.preflight as ingest_preflight

    def refuse_to_fetch(*args, **kwargs):
        raise _StopBeforeFetch("the admission must be taken before the fetch")

    monkeypatch.setattr(ingest_preflight, "build_input_catalog",
                        refuse_to_fetch)

    with pytest.raises(_StopBeforeFetch):
        runtime.run_experiment(exp, SimpleNamespace(), tmp_path / "out")

    assert len(calls) == 1, (
        f"the single-domain run door priced its admission {len(calls)} times")


def test_the_prepared_single_domain_door_resolves_the_domains_own_table():
    """The prepared forecast judges the domain on the table that governs it.

    Red on the base: this door read ``exp.tiles`` raw, so a single domain
    carrying its own ``tiles = {...}`` table was judged on the tree-wide
    table here and on its own table at the review, at ``gpuwm run`` and
    at the run plan -- one configuration, two answers, and the second one
    only after the prepared cache had been restored.  The seam is checked
    on the route's own source because the enclosing function is a whole
    forecast; the resolution it names is
    :func:`gpuwm.core.streaming.options_for_domain`, which every other
    surface reaches through :func:`cold_single_domain_decision`.
    """
    import inspect

    from gpuwm import prepared_single_domain_forecast as psdf

    source = inspect.getsource(psdf.run_prepared_forecast)
    assert "tiles_options = streaming.options_for_domain(" in source, (
        "the prepared forecast door does not resolve the governing table")
    assert "tiles_options = getattr(exp, \"tiles\", None)" not in source
    # And the admission it hands the decision is the one it keeps for the
    # streamed walk, taken through the shared guard.
    assert "options=tiles_options)" in source
    assert "streaming.cold_single_domain_admission(" in source


def _root_with_its_own_table():
    """One domain carrying ``tiles = {...}`` under a tree-wide ``mode = "off"``.

    The configuration class the per-domain resolution exists for: the
    tree-wide table says "nothing streams" and the domain row says
    "stream this one".  Every run door reads the row, through
    :func:`gpuwm.core.streaming.options_for_domain`; three REVIEW
    surfaces read the tree-wide table raw and therefore answered that
    this configuration needed no decision at all.
    """
    from dataclasses import replace

    exp = _single_domain_experiment()
    own = st.StreamingOptions(mode="on", tile_nx=64, tile_ny=64, nbuffers=2)
    return replace(exp, tiles=st.OFF,
                   domains=[replace(exp.domains[0], tiles=own)])


def _own_table_case():
    """The fixture above, its governing table, and the run door's answer.

    The door's answer is taken through :func:`gpuwm.core.streaming.decide`
    on the governing table, which is the primitive BOTH trees have and the
    one ``cold_single_domain_decision`` resolves to for a pinned tiling
    (its admission is ``None`` when nothing has to be planned).  So what
    is red on the base in each test below is the REVIEW surface's answer,
    not a missing name.
    """
    exp = _root_with_its_own_table()
    machine = _machine(8 * GIB)
    assert (getattr(exp, "tiles", None) or st.OFF).mode == "off", (
        "the tree-wide table must be off for this fixture to mean anything")
    governing = st.options_for_domain(exp.domains[0], exp.tiles)
    assert governing.enabled and governing.mode == "on"
    door = st.decide(exp.domains[0].run, governing, machine=machine,
                     resident_estimate=None)
    assert door.stream is True
    assert (door.tile_nx, door.tile_ny, door.nbuffers) == (64, 64, 2)
    return exp, machine, door


def test_a_domains_own_tiles_table_is_priced_by_the_streamed_envelope():
    """The admission gate prices the root on the table that governs it.

    Red on the base: :func:`gpuwm.core.preflight.streamed_forecast_envelope`
    read ``exp.tiles`` raw, so with the tree-wide table off it returned
    ``None`` -- the gate priced this root RESIDENT while every run door
    streamed it.
    """
    exp, machine, door = _own_table_case()
    envelope = pf.streamed_forecast_envelope(exp, machine=machine)
    assert envelope is not None, (
        "the root's own table was not priced by the streamed envelope")
    assert (envelope.tile_nx, envelope.tile_ny) == (door.tile_nx, door.tile_ny)
    assert envelope.nbuffers == door.nbuffers


def test_a_domains_own_tiles_table_takes_the_reviews_tiled_road():
    """``gpuwm check``'s sizing enters the tiled road for that domain.

    Red on the base: :func:`gpuwm.domain_wizard._sizing_phases` returned
    on the tree-wide ``mode == "off"`` before the tiled road was ever
    entered, so ``phases.streamed`` was ``None`` and the whole review was
    conducted on a resident forecast term.
    """
    exp, machine, door = _own_table_case()
    from gpuwm import domain_wizard as dw

    phases = dw._sizing_phases(exp, free_bytes=int(machine.vram_bytes),
                               machine=machine, source="gfs")
    assert phases.streamed is not None, (
        "the review did not take the tiled road for a domain whose own "
        "table is enabled")
    assert ((phases.streamed.tile_nx, phases.streamed.tile_ny)
            == (door.tile_nx, door.tile_ny))
    envelope = pf.streamed_forecast_envelope(exp, machine=machine)
    assert (int(phases.streamed.peak_vram_bytes)
            == int(envelope.peak_vram_bytes))
    assert int(phases.streamed.host_bytes) == int(envelope.host_bytes)


def test_a_domains_own_tiles_table_is_reported_streamed_by_the_run_plan():
    """The run plan's execution block says streamed, not resident.

    Red on the base: :func:`gpuwm.runplan._execution_estimate` short-cut
    on the tree-wide ``mode == "off"`` and reported the configuration
    resolved-without-deciding, with ``streamed_forecast`` false and the
    resident term selected, for a domain the run door streams.
    """
    exp, machine, _door = _own_table_case()
    from gpuwm import domain_wizard as dw

    phases = dw._sizing_phases(exp, free_bytes=int(machine.vram_bytes),
                               machine=machine, source="gfs")
    block = runplan._execution_estimate(phases, exp, machine)
    assert block["resolved"] is True
    assert block["planner_refusal"] is None
    assert block["streamed_forecast"] is True, (
        "the run plan reported a streamed domain as resident")
    assert (int(block["selected_forecast_envelope_bytes"])
            == int(phases.forecast_envelope_bytes))
    assert block["host_bytes"] == int(phases.streamed.host_bytes)


def test_a_domains_own_table_gives_the_review_and_the_doors_one_answer():
    """The three surfaces above and the shared admission are one answer."""
    exp, machine, door = _own_table_case()
    from gpuwm import domain_wizard as dw

    shared = st.cold_single_domain_decision(exp, machine=machine)
    assert shared.stream is door.stream
    assert ((shared.tile_nx, shared.tile_ny, shared.nbuffers)
            == (door.tile_nx, door.tile_ny, door.nbuffers))
    envelope = pf.streamed_forecast_envelope(exp, machine=machine)
    phases = dw._sizing_phases(exp, free_bytes=int(machine.vram_bytes),
                               machine=machine, source="gfs")
    block = runplan._execution_estimate(phases, exp, machine)
    assert (envelope.tile_nx, envelope.tile_ny) == (shared.tile_nx,
                                                    shared.tile_ny)
    assert (int(phases.streamed.peak_vram_bytes)
            == int(envelope.peak_vram_bytes))
    assert block["host_bytes"] == int(envelope.host_bytes)


def test_a_domains_own_tiles_table_reaches_the_reports_streaming_sentence():
    """The report says out loud which allocation it priced, for that domain.

    Red on the base: :func:`gpuwm.core.preflight.streaming_advisory` read
    ``exp.tiles`` raw and returned ``None`` on a tree-wide ``mode =
    "off"``, so a configuration whose only enabled table is the domain
    row's own got NO sentence -- the report quoted streamed figures with
    nothing saying they were streamed, which is the one thing this
    advisory exists to say.
    """
    exp, machine, door = _own_table_case()
    sentence = pf.streaming_advisory(exp, machine=machine)
    assert sentence is not None, (
        "the report said nothing about a domain its own table streams")
    assert "[tiles] mode = 'on'" in sentence
    assert "STREAMED allocation" in sentence


def test_the_nested_streaming_sentence_walks_the_shared_admission(monkeypatch):
    """The sentence's road is priced from the admission, not a third basis.

    Red on the base: the walk was asked with no ``resident_estimate``, so
    it fell back to an ``estimate_experiment`` of its own and described a
    road priced differently from the verdict printed beside it.
    """
    exp = _cyclone()
    machine = _machine()
    seen: list = []
    real = st.tree_road_plan

    def spy(exp_arg, *, machine=None, resident_estimate=None):
        seen.append(resident_estimate)
        return real(exp_arg, machine=machine,
                    resident_estimate=resident_estimate)

    monkeypatch.setattr(st, "tree_road_plan", spy)
    pf.streaming_advisory(exp, machine=machine)
    assert seen, "the nested sentence never priced a road"
    assert seen[0] is not None, (
        "the nested sentence priced its road from the walk's own fallback")
    assert (seen[0].peak_envelope_bytes
            == pf.admission_estimate(exp, machine=machine).peak_envelope_bytes)


# ---------------------------------------------------------------------------
# The advisories recorded beside that seam.
# ---------------------------------------------------------------------------

def _staged_renderer(monkeypatch, tmp_path, stdout, calls):
    binary = tmp_path / "rw_wrfbatch"
    binary.write_text("binary", encoding="utf-8")
    from gpuwm import rustwx

    monkeypatch.setattr(render, "_resolve_engine", lambda _mode: ("rust", ""))
    monkeypatch.setattr(render, "matplotlib_workaround_notice",
                        lambda _engine: None)
    monkeypatch.setattr(rustwx, "find_renderer", lambda: binary)
    monkeypatch.setattr(rustwx, "renderer_env", lambda: {})

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return binary


def test_the_render_catalog_asks_the_renderer_once_per_process(monkeypatch,
                                                               tmp_path):
    """Plan review stopped spawning ``rw_wrfbatch`` once per question."""
    # Through getattr so that what is red on the base is the SPAWNING,
    # not the cache's name.
    getattr(runplan, "_RENDER_CATALOG_CACHE", {}).clear()
    calls: list = []
    stdout = ("products:\n  reflectivity\n  mslp\n"
              "group keywords: severe, basic\nselectable_slugs=2\n")
    binary = _staged_renderer(monkeypatch, tmp_path, stdout, calls)

    first = runplan.render_catalog()
    second = runplan.render_catalog()
    assert [entry["name"] for entry in first["products"]] == ["reflectivity",
                                                              "mslp"]
    assert second == first
    assert len(calls) == 1, "the renderer was spawned more than once"

    # The caller's copy is the caller's own.
    first["products"].append({"name": "invented"})
    assert ([entry["name"] for entry
             in runplan.render_catalog()["products"]]
            == ["reflectivity", "mslp"])
    assert len(calls) == 1

    # A RESTAGED renderer is a different renderer, and is asked again.
    binary.write_text("a different binary entirely", encoding="utf-8")
    stat = binary.stat()
    os.utime(binary, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
    runplan.render_catalog()
    assert len(calls) == 2
    getattr(runplan, "_RENDER_CATALOG_CACHE", {}).clear()


def test_the_declared_planning_machine_is_built_once(monkeypatch):
    """The review's machine carries its profile from the constructor.

    Red on the base: the profile was replaced onto a bare machine after
    it was built, so the object the review priced with was a second
    machine and the shared builder was never told what card it was for.
    """
    seen: dict = {}
    built = ap.Machine(4 * GIB, 64 * GIB, name="fixture card",
                       host_source="probe", device_profile="the profile")

    def spy(*, vram_bytes, name, device_profile=None):
        seen.update(vram_bytes=vram_bytes, name=name,
                    device_profile=device_profile)
        return built

    monkeypatch.setattr(st, "planner_machine", spy)
    machine = downscale_pricing.declared_machine(
        free_bytes=4 * GIB, name="fixture card", device_profile="the profile")
    assert machine is built
    assert seen["device_profile"] == "the profile"


def test_the_cyclone_door_quotes_the_budget_auto_actually_used():
    """Auto is judged NET of the withholding, so that is what is printed.

    Red on the base: the sentence quoted the whole-process allowance
    while the tree walk had already withheld a moving nest's rebuild from
    it, so the door published two budgets for one decision.
    """
    intent = {"cycle": CYCLE, "point": POINT, "hours": 6, "tiles": "on"}
    unwithheld = 8_000_000_000
    walked = 7_500_000_000
    phases = SimpleNamespace(
        peak_envelope_bytes=5_000_000_000,
        tree_road=SimpleNamespace(
            rows=({"road": "resident"}, {"road": "resident"}),
            refusal=None, report_error=None,
            total_budget_bytes=walked))

    admitted = tc._unreduced_resident_admission(
        intent, lambda _exp: unwithheld, lambda _exp: phases)
    assert admitted is not None
    assert admitted["tiles"] == "auto"
    assert admitted["budget_bytes"] == walked
    sentence = tc._keeps_coverage_sentence(admitted, "on")
    assert str(walked) in sentence and str(unwithheld) not in sentence


def test_the_cyclone_door_quotes_the_plain_budget_where_nothing_is_withheld():
    """``off`` withholds nothing, so the allowance is the allowance."""
    intent = {"cycle": CYCLE, "point": POINT, "hours": 6, "tiles": "auto"}
    unwithheld = 8_000_000_000
    phases = SimpleNamespace(peak_envelope_bytes=5_000_000_000,
                             tree_road=None)
    admitted = tc._unreduced_resident_admission(
        intent, lambda _exp: unwithheld, lambda _exp: phases)
    assert admitted["tiles"] == "off"
    assert admitted["budget_bytes"] == unwithheld


def _abandoned_stores(root: Path, count: int, first: int = 0, *,
                      prefix: str = "rwstore-") -> list:
    stores = []
    for index in range(first, first + count):
        store = root / f"{prefix}{index}"
        (store / "d01").mkdir(parents=True)
        (store / "d01" / "hour.rws").write_bytes(b"x" * 64)
        stores.append(store)
    return stores


def test_abandoned_render_stores_are_swept(tmp_path):
    """A store whose process never reached its own cleanup is removable."""
    delivery = tmp_path / "case"
    delivery.mkdir()
    root = render.scratch_root_for(delivery)
    root.mkdir(parents=True)
    mine = render.stage_scratch_prefix()
    stores = _abandoned_stores(root, 3, prefix=mine)
    keeper = root / "something-else"
    keeper.mkdir()

    removed = render.sweep_abandoned_scratch(delivery, prefix=mine)
    assert sorted(removed) == sorted(stores)
    assert not any(store.exists() for store in stores)
    assert keeper.exists(), "only the render's own stores are swept"
    assert render.sweep_abandoned_scratch(delivery, prefix=mine) == []


def test_the_sweep_refuses_the_prefix_that_matches_every_store(tmp_path):
    """The plain prefix is not a default, because it is the blanket sweep.

    ``scratch_store`` promises a concurrent render that nobody removes
    its working store, and the plain prefix matches every store there is
    -- including a live one, which on POSIX ``rmtree`` deletes out from
    under the process still writing it.  So the ownership token is
    required and the plain prefix is refused, by a sentence that names
    the breakage and how to sweep properly.
    """
    delivery = tmp_path / "case"
    delivery.mkdir()
    root = render.scratch_root_for(delivery)
    root.mkdir(parents=True)
    live = _abandoned_stores(root, 1)

    with pytest.raises(TypeError):
        render.sweep_abandoned_scratch(delivery)
    with pytest.raises(ValueError) as caught:
        render.sweep_abandoned_scratch(
            delivery, prefix=render.DEFAULT_SCRATCH_PREFIX)
    text = str(caught.value)
    assert "stage_scratch_prefix()" in text
    assert "concurrent render" in text
    assert live[0].exists(), "the refused sweep removed a store anyway"


def test_a_minted_token_is_short_enough_for_a_long_delivery_path():
    """The token rides on every working-store path; it stays short."""
    minted = render.stage_scratch_prefix()
    assert minted.startswith(render.DEFAULT_SCRATCH_PREFIX)
    body = minted[len(render.DEFAULT_SCRATCH_PREFIX):-1]
    assert len(body) == 8 and int(body, 16) >= 0
    assert len(set(render.stage_scratch_prefix() for _ in range(64))) == 64


def test_a_store_the_sweeping_door_never_started_survives(tmp_path):
    """Ownership is the minted token, not when the store appeared.

    ``scratch_store`` promises concurrent renders into one delivery that
    none of them removes another's working store, and on POSIX a blanket
    ``rmtree`` deletes a directory whose files a live process still holds
    open.  A store carrying another token, or the plain prefix, is not
    this door's whatever order the two renders started in.
    """
    delivery = tmp_path / "case"
    delivery.mkdir()
    root = render.scratch_root_for(delivery)
    root.mkdir(parents=True)
    mine, theirs = "rwstore-aaaa-", "rwstore-bbbb-"

    ours = _abandoned_stores(root, 2, prefix=mine)
    other_door = _abandoned_stores(root, 1, prefix=theirs)
    plain = _abandoned_stores(root, 1, first=7)

    removed = render.sweep_abandoned_scratch(delivery, prefix=mine)
    assert sorted(removed) == sorted(ours)
    assert other_door[0].exists(), "another door's live store was removed"
    assert plain[0].exists(), "a store no door owns was removed"


def test_a_render_store_takes_the_prefix_its_door_handed_down(monkeypatch,
                                                              tmp_path):
    """The token reaches the store, so the sweep can match on it."""
    delivery = tmp_path / "case"
    delivery.mkdir()
    minted = render.stage_scratch_prefix()
    monkeypatch.setenv(render.SCRATCH_PREFIX_ENV, minted)
    with render.scratch_store(delivery) as store:
        assert store.name.startswith(minted)
    # No door, no token: the plain prefix, and no door will sweep it.
    monkeypatch.delenv(render.SCRATCH_PREFIX_ENV)
    with render.scratch_store(delivery) as store:
        assert store.name.startswith(render.DEFAULT_SCRATCH_PREFIX)
        assert not store.name.startswith(minted)


def test_a_prefix_that_is_not_a_token_renders_anyway(monkeypatch, tmp_path,
                                                     capsys):
    """An unusable token warns once and renders; it never refuses.

    The only consequence is that the door which handed it down will not
    recognise the store as its own, so the store is kept rather than a
    render stopped.
    """
    delivery = tmp_path / "case"
    delivery.mkdir()
    monkeypatch.setattr(render, "_WARNED_SCRATCH_PREFIX", False)
    monkeypatch.setenv(render.SCRATCH_PREFIX_ENV, "../escape")
    with render.scratch_store(delivery) as store:
        assert store.name.startswith(render.DEFAULT_SCRATCH_PREFIX)
        assert store.parent == render.scratch_root_for(delivery)
    warning = capsys.readouterr().err
    assert render.SCRATCH_PREFIX_ENV in warning


def test_a_failed_render_stage_sweeps_its_scratch_and_says_so(monkeypatch,
                                                             tmp_path, capsys):
    """The door that saw the render die clears the working stores it left.

    Red on the base: the stage raised and the sibling scratch tree stayed
    beside the delivery, holding whole tiled hour files nothing would ever
    read again, with no line anywhere saying it existed.

    Red on the repair pass's own first answer too: the concurrent render
    here opens its store WHILE the failing stage runs, which is the
    ordinary interleaving of two ``gpuwm go`` calls into one case, and a
    sweep that spared only what was standing before the stage started
    deleted it out from under the live renderer.
    """
    delivery = tmp_path / "case"
    delivery.mkdir()
    root = render.scratch_root_for(delivery)
    root.mkdir(parents=True)

    plan = {"render_products": "reflectivity", "render": delivery,
            "run": tmp_path / "run"}
    frame = tmp_path / "wrfout_d01"
    frame.write_text("frame", encoding="utf-8")

    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(go_cli, "wrfout_frames", lambda _plan: [frame])
    import gpuwm.first_products as first_products
    monkeypatch.setattr(first_products, "published_frames",
                        lambda frames, _plan: (frames, [], None))
    monkeypatch.setattr(go_cli, "render_command", lambda *a, **k: ["renderer"])

    stores: list = []
    live: list = []

    def fail(*args, **kwargs):
        # The token the door minted reaches the stage, and the stores
        # this stage opens carry it.  Read without assuming the door has
        # one to give, so that what is red where it has none is the
        # DATA LOSS below and not a missing name.
        handed = (kwargs.get("env") or {}).get(
            "GPUWM_RENDER_SCRATCH_PREFIX", "rwstore-")
        stores.extend(_abandoned_stores(root, 2, prefix=handed))
        # A concurrent render into the same case opens its own store
        # while this stage is running, and is still working in it.
        live.extend(_abandoned_stores(root, 1, prefix="rwstore-", first=9))
        # The stage then dies without reaching its own cleanup, which is
        # the whole case.
        raise go_cli.GoStageFailed(1)

    monkeypatch.setattr(go_cli, "_run_stage", fail)

    with pytest.raises(go_cli.GoStageFailed):
        go_cli._render_stage(plan, explain=False)

    assert not any(store.exists() for store in stores)
    assert live[0].exists(), "the concurrent render's LIVE store was removed"
    assert root.exists(), "the live render's scratch root was removed"
    warning = capsys.readouterr().err
    assert "render: warning:" in warning
    assert str(render.scratch_root_for(delivery)) in warning


def test_a_failed_forecast_stage_sweeps_the_early_renders_scratch(monkeypatch,
                                                                 tmp_path,
                                                                 capsys):
    """The other stage that draws owns its working stores on the same terms.

    Red on the base and on this lane's own first answer: the early render
    runs INSIDE the forecast subprocess and writes into the delivery, but
    only the render stage minted a token, so the forecast stage's stores
    took the plain prefix and no door would ever sweep them -- and once
    the sweep refused the plain prefix, nothing could.  A forecast that
    dies mid-frame therefore left whole tiled hour files beside a
    delivery that published nothing, permanently.
    """
    delivery = tmp_path / "case"
    delivery.mkdir()
    root = render.scratch_root_for(delivery)
    root.mkdir(parents=True)

    plan = {"render_products": "reflectivity", "render": delivery,
            "run": tmp_path / "run", "runner": "gpuwm.runner",
            "source": "gfs", "prepared": tmp_path / "prepared",
            "authority": tmp_path / "authority", "domains": 1}
    (tmp_path / "authority").mkdir()
    monkeypatch.setattr(go_cli, "render_extra_missing", lambda: None)
    monkeypatch.setattr(go_cli, "_profile_flags", lambda _plan: [])

    stores: list = []
    live: list = []

    def fail(*args, **kwargs):
        # Read the token without assuming the door has one to give, so
        # what is red where it has none is the DATA LOSS and not a name.
        handed = (kwargs.get("env") or {}).get(
            "GPUWM_RENDER_SCRATCH_PREFIX", "rwstore-")
        stores.extend(_abandoned_stores(root, 2, prefix=handed))
        # A render into the same case, working while this stage runs.
        live.extend(_abandoned_stores(root, 1, prefix="rwstore-", first=9))
        raise go_cli.GoStageFailed(1)

    monkeypatch.setattr(go_cli, "_run_stage", fail)

    digests = {"proof": "a", "source_manifest": "b", "prepared_content": "c"}
    observer = SimpleNamespace(hosts_forecast=False)
    with pytest.raises(go_cli.GoStageFailed):
        go_cli._run_forecast(plan, digests, explain=False, observer=observer)

    assert not any(store.exists() for store in stores), (
        "the failed forecast stage left its early render's working stores")
    assert live[0].exists(), "the concurrent render's LIVE store was removed"
    warning = capsys.readouterr().err
    assert "forecast: warning:" in warning
    assert str(render.scratch_root_for(delivery)) in warning


def test_a_forecast_stage_that_cannot_draw_mints_no_token(monkeypatch,
                                                          tmp_path):
    """A stage carrying no early render owns no store and is handed none.

    An install that cannot draw at all takes no early render, so the
    forecast subprocess opens no working store and there is nothing for
    a token to name.  Pinned so the sweep stays tied to the stage that
    actually draws rather than becoming a thing every stage carries.
    """
    plan = {"render": tmp_path / "case", "run": tmp_path / "run",
            "runner": "gpuwm.runner", "source": "gfs",
            "prepared": tmp_path / "prepared",
            "authority": tmp_path / "authority", "domains": 1}
    (tmp_path / "authority").mkdir()
    monkeypatch.setattr(go_cli, "render_extra_missing",
                        lambda: "no renderer is staged")
    monkeypatch.setattr(go_cli, "_profile_flags", lambda _plan: [])
    seen: list = []

    monkeypatch.setattr(go_cli, "_run_stage",
                        lambda *a, **k: seen.append(k.get("env")))
    digests = {"proof": "a", "source_manifest": "b", "prepared_content": "c"}
    go_cli._run_forecast(plan, digests, explain=False,
                         observer=SimpleNamespace(hosts_forecast=False))
    assert seen == [None]


# ---------------------------------------------------------------------------
# The acoustic reach a cold door prices, against the one the run walks
# ---------------------------------------------------------------------------

#: A budget the widened fixture below streams its root against.  Its own
#: resident envelope is 27,836,807,643 bytes, far past it.
ADAPTIVE_BUDGET = 20 * GIB


def _adaptive_tree(point=(40.0, -95.0), root_n=700, child_n=400):
    """The same two-domain tree with ADAPTIVE clocks, widened to stream.

    The cyclone door authors fixed clocks, so its own configuration cannot
    see this seam at all; adaptive is an ordinary authorable choice.  The
    grids are widened because the shipped fixture's resident road fits
    every budget its streamed road does, so nothing in it is ever handed
    a tile halo -- and the halo is the question.  The point is
    mid-latitude, which is where an ordinary Mercator root carries a
    conformal scale above unity.
    """
    from dataclasses import replace

    _text, exp = tc.configuration_text(cycle=CYCLE, point=point, hours=6,
                                       tiles="auto")
    return replace(exp, domains=tuple(
        replace(dc, run=replace(
            dc.run, use_adaptive_time_step=True,
            nx=(root_n if dc.parent_id == 0 else child_n),
            ny=(root_n if dc.parent_id == 0 else child_n)))
        for dc in exp.domains))


def _live_nodes_with_real_map_factors(exp):
    """Live nodes carrying the float32 map factors the domains will load."""
    import numpy as np

    from gpuwm.static.projection import grids_from_projection_config

    grids = {int(dc.grid_id): grid for dc, grid
             in zip(exp.domains, grids_from_projection_config(exp))}
    nodes = st._config_tree_nodes(exp.domains)
    for node in nodes:
        grid = grids[int(node.cfg.grid_id)]
        node.state = SimpleNamespace(
            msfu=np.asarray(grid.mapfac_u(), dtype=np.float32),
            msfv=np.asarray(grid.mapfac_v(), dtype=np.float32))
    return nodes


def _stepper_arguments(monkeypatch, exp, nodes, machine, estimate, *,
                       tree_decision=None):
    """What ``steppers_for_tree`` hands ``make_stepper``, per grid."""
    handed = []

    def capture(state, cfg, options, *, decision, machine=None, build=None):
        handed.append((int(cfg.grid_id),
                       None if decision.halo is None else int(decision.halo),
                       options.acoustic_map_factor,
                       int(decision.detail.get("claim_bytes") or 0)))
        return object()

    monkeypatch.setattr(st, "make_stepper", capture)
    model = SimpleNamespace(walk_parent_first=lambda: nodes,
                            _declared_experiment=exp)
    st.steppers_for_tree(
        model, exp.tiles, builders={}, decisions={}, machine=machine,
        tree_decision=tree_decision,
        resident_estimate=(None if tree_decision is not None else estimate))
    monkeypatch.undo()
    return handed


def test_the_cold_admission_walks_the_reach_the_live_run_walks(monkeypatch):
    """A door deciding before the fetch prices the halo the run needs.

    Red on the branch this repairs: the cold admission nodes carried no
    map factors at all, so an adaptive domain's acoustic reach was priced
    from a UNIT factor and the halo the decision carried into the build
    pass was narrower than the one the same tree gets when the build pass
    decides for itself on the live nodes.  A halo below the dependency
    radius does not fail: :func:`gpuwm.core.streaming.decide` says of it
    that tile interiors are silently wrong and the run is FASTER, which
    is how it hides.

    The map factor is a function of the projection and latitude alone, so
    a cold surface can resolve it exactly rather than assume unity --
    which is what makes these two roads comparable at all.
    """
    from gpuwm.core.adaptive_clock import maximum_map_factor

    exp = _adaptive_tree()
    live = _live_nodes_with_real_map_factors(exp)
    factors = {int(node.cfg.grid_id): maximum_map_factor(node.state)
               for node in live}
    assert max(factors.values()) > 1.2, (
        "this fixture must carry a conformal scale far enough above unity "
        f"to move the acoustic substep count; it carries {factors}")

    machine = _machine(ADAPTIVE_BUDGET)
    estimate = pf.admission_estimate(exp, machine=machine)

    door = st.cold_tree_streaming_decision(
        exp, st.cold_tree_admission_nodes(exp), machine=machine)
    assert door is not None
    from_the_door = _stepper_arguments(monkeypatch, exp, live, machine,
                                       estimate, tree_decision=door)
    decided_here = _stepper_arguments(
        monkeypatch, exp, _live_nodes_with_real_map_factors(exp), machine,
        estimate)

    assert from_the_door == decided_here, (
        "the road the door decided and the road the build pass decides "
        "hand the executor different tiles for the same tree")
    assert any(entry[1] for entry in from_the_door), (
        "this fixture must stream something, or the halo is not exercised")
    assert {entry[0]: entry[2] for entry in from_the_door} == factors, (
        "the cold walk resolved a different conformal scale than the live "
        "domains carry")
