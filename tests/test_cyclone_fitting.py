"""Cyclone fitting uses real authoring and the shared bounded fit contract.

The policy tests inject admission outcomes, not replacement geometry or physics.
The final integration tests run the unmodified memory planner with declared
hardware; they need netCDF4 (a normal runtime dependency), but no GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from gpuwm import cyclone_setup as tc
from gpuwm import domain_wizard as dw
from gpuwm.configuration_recovery import MemoryAdmissionError
from gpuwm.core import streaming
from gpuwm.experiment import validate_spawn_placement
from gpuwm.starter_template import changes

GIB = dw.GIB
INTENT = dict(cycle="2026090900", point=(18., -65.), hours=27,
              name="Reviewed cyclone", source="cyclone.toml")


def phases(peak=9 * GIB, *, resident_floor=GIB, streaming_floor=None):
    return SimpleNamespace(
        peak_envelope_bytes=int(peak), binding_phase="forecast",
        verdict=lambda budget: f"fixture peak {int(peak)} exceeds budget {budget}",
        forecast=SimpleNamespace(
            fixed_envelope_bytes=resident_floor),
        tree_road=SimpleNamespace(streaming_fixed_floor_bytes=streaming_floor))


@pytest.fixture
def hardware(monkeypatch):
    sizing = dw.SizingBudget(12., 11 * GIB, None, "declared fixture", measured=False)
    machine = object()
    monkeypatch.setattr(dw, "sizing_budget_bytes", lambda *a, **kw: 10 * GIB)
    return dict(sizing=sizing, target_machine=machine)


#: The one extra price a tiled request now pays: the UNREDUCED 200x160 /
#: 160x160 domain, priced in the mode this door would name as the way to
#: keep that ground, so a reduced proposal or a refusal can name it.  It
#: never becomes the proposal, so every sequence assertion about the
#: SEARCH separates it out by the only two things that identify it: a
#: mode that is not the requested one, and the requested dimensions.
def is_resident_probe(exp, requested="auto") -> bool:
    """A price of the RESIDENT route asked on behalf of a tiled request.

    Two of them exist: the unreduced-coverage probe that lets a proposal
    name the mode that keeps the requested ground, and -- on a refusal
    only -- the bounded resident ladder that measures what `--tiles off`
    actually authors, so the refusal names a layout instead of a
    suggestion.  The coverage probe is priced in the mode the sentence
    NAMES -- `auto` under `--tiles on`, because auto weighs a tree
    resident before it consults the planner, and `off` under `auto` --
    since pricing one mode and recommending another is how this door came
    to recommend `--tiles auto` on the strength of what `--tiles off`
    costs.  Neither probe can become the proposal, so every assertion
    about the SEARCH separates them out.  A request that already asked
    for `off` has nothing to be told, and its own prices look exactly
    like these.
    """

    if requested == "off":
        return False
    mode = exp.tiles.mode if exp.tiles else "off"
    return mode in {"off", tc._recommended_mode(requested)}


def is_resident_coverage_probe(exp, requested="auto") -> bool:
    """The unreduced-coverage probe specifically: resident, at the
    dimensions that were actually requested."""

    return (is_resident_probe(exp, requested)
            and [[d.run.nx, d.run.ny] for d in exp.domains]
            == [list(tc.ROOT_DIMS), list(tc.CHILD_DIMS)])


def shrinking_price(monkeypatch, *, threshold=170, resource="vram"):
    calls = []

    def price(exp, **kwargs):
        calls.append((exp, kwargs))
        nx = exp.domains[0].run.nx
        if nx > threshold:
            if resource == "vram":
                # 180x144 fits the raw budget, but NOT the fit headroom.
                return phases(9.8 * GIB if nx == 180 else 11 * GIB)
            raise dw.DomainFitError("fixture host-store admission", resource=resource,
                                    phases=phases(11 * GIB))
        return phases()

    monkeypatch.setattr(dw, "_sizing_phases", price)
    return calls


@pytest.mark.parametrize("tiles", ["off", "auto", "on"])
def test_fitting_defaults_are_byte_identical_even_inside_new_fit_headroom(monkeypatch, hardware, tiles):
    expected, _ = tc.configuration_text(**INTENT, tiles=tiles)
    calls = []
    monkeypatch.setattr(dw, "_sizing_phases", lambda exp, **kw:
                        calls.append(kw) or phases(10 * GIB))
    monkeypatch.setattr(dw, "fit_ladder", lambda **kw: pytest.fail("fitting defaults must not be resized"))
    result = tc.plan_cyclone(**INTENT, **hardware, tiles=tiles)
    assert result["config_text"] == expected
    assert result["kind"] == "configuration" and not result["fitting"]["changed"]
    assert result["fitting"]["changes"] == []
    assert result["fitting"]["proposed_dimensions"] == [[200, 160], [160, 160]]
    assert len(calls) == 1 and calls[0]["machine"] is hardware["target_machine"]


@pytest.mark.parametrize("tiles", ["off", "auto", "on"])
@pytest.mark.parametrize("point", [(18., -65.), (35., -75.), (-18., 65.)])
def test_reduced_proposal_preserves_intent_and_uses_same_admission(monkeypatch, hardware, tiles, point):
    calls = shrinking_price(monkeypatch)
    intent = {**INTENT, "point": point, "tiles": tiles}
    original, _ = tc.configuration_text(**intent)
    result = tc.plan_cyclone(**intent, **hardware)
    fitted = tomllib.loads(result["config_text"])
    expected = tomllib.loads(original)
    allowed = {"domain[0].nx", "domain[0].ny", "domain[1].nx", "domain[1].ny",
               "domain[1].i_parent_start", "domain[1].j_parent_start", "fetch.area"}
    diff = changes(expected, fitted)
    assert diff and all(field in allowed for field, _, _ in diff)
    assert result["fitting"]["changes"] == [
        {"field": f, "before": a, "after": b} for f, a, b in diff]
    assert result["kind"] == "proposal" and result["fitting"]["review_required"]
    assert result["fitting"]["proposed_dimensions"] == [[170, 136], [136, 136]]
    assert result["point"] == list(point) and result["hours"] == INTENT["hours"]
    assert result["cycle"] == INTENT["cycle"] and result["source"] == "gfs"
    assert result["tiles"] == tiles
    assert result["forecast_started"] is False and result["created"] is False
    # Five fit prices -- default, 95%, 90%, 85%, final re-admission -- plus,
    # for a TILED request, exactly one probe of the UNREDUCED domain as a
    # resident allocation.  That probe is what lets a reduced proposal name
    # `--tiles off` as the way to keep the requested coverage; it is asked
    # once, on the same hardware snapshot and the same operands as every
    # other price here, and it never becomes the proposal.
    resident_probe = 0 if tiles == "off" else 1
    assert len(calls) == 5 + resident_probe
    probes = [(exp, kw) for exp, kw in calls
              if (exp.tiles.mode if exp.tiles else "off") != tiles]
    assert len(probes) == resident_probe
    for exp, _kw in probes:
        assert (exp.tiles.mode if exp.tiles else "off")             == tc._recommended_mode(tiles)
        assert [[d.run.nx, d.run.ny] for d in exp.domains] == [[200, 160], [160, 160]]
    for exp, kw in calls:
        assert kw["machine"] is hardware["target_machine"]
        assert kw["free_bytes"] == hardware["sizing"].free_bytes
        assert kw["vram_gib"] == hardware["sizing"].vram_gib
        assert kw["profile"] is hardware["sizing"].device_profile
        assert kw["forcing_interval_seconds"] == 10800. and kw["source"] == "gfs"
        assert [d.run.dx for d in exp.domains] == [12000., 3000.]
    for exp, _kw in calls:
        if (exp, _kw) not in probes:
            assert (exp.tiles.mode if exp.tiles else "off") == tiles
    assert result["memory"]["peak_envelope_bytes"] <= (
        result["memory"]["budget_bytes"] - result["memory"]["fit_headroom_bytes"])


def test_identical_inputs_have_stable_proposals_hashes_and_search_order(monkeypatch, hardware):
    calls = shrinking_price(monkeypatch)
    first = tc.plan_cyclone(**INTENT, **hardware)
    first_order = [e.domains[0].run.nx for e, _ in calls]
    calls.clear()
    second = tc.plan_cyclone(**INTENT, **hardware)
    assert first == second
    assert first_order == [e.domains[0].run.nx for e, _ in calls]
    assert first["fitting"]["fit_id"] == hashlib.sha256(first["config_text"].encode()).hexdigest()


def test_every_rung_has_aligned_containment_and_search_plus_move_clearance():
    _, original = tc.configuration_text(**INTENT)
    scales = tc._fit_scales(original)
    assert scales == (.95, .9, .85, .8, .75, .7, .65)
    assert tc._fit_dimensions(scales[-1]) == [(130, 104), (104, 104)]
    for scale in scales:
        text, exp = tc.configuration_text(**INTENT, dimensions=tc._fit_dimensions(scale))
        parent, child = exp.domains
        assert child.run.nx % (2 * tc.RATIO) == child.run.ny % (2 * tc.RATIO) == 0
        assert child.i_parent_start == 1 + (parent.run.nx - child.run.nx // 4) // 2
        assert child.j_parent_start == 1 + (parent.run.ny - child.run.ny // 4) // 2
        follow = child.follow
        assert follow == original.domains[1].follow
        clearance = follow.tracker.search_margin_cells + max(
            follow.tracker.max_shift_cells, follow.max_move_parent_cells)
        for di in (-clearance, clearance):
            for dj in (-clearance, clearance):
                validate_spawn_placement(exp, 2, child.i_parent_start + di,
                                         child.j_parent_start + dj)
        # No manual tile dimensions or halo overrides are introduced.
        assert tomllib.loads(text)["tiles"] == tomllib.loads(
            tc.configuration_text(**INTENT)[0])["tiles"]


@pytest.mark.parametrize("tiles", ["off", "auto", "on"])
def test_proven_fixed_floors_terminate_without_search(monkeypatch, hardware, tiles):
    blocked = phases(20 * GIB, resident_floor=11 * GIB, streaming_floor=12 * GIB)
    seen = []

    def price(exp, **kw):
        seen.append(exp)
        if tiles == "off":
            return blocked
        raise dw.DomainFitError("shared process/radiation floor", resource="vram", phases=blocked)

    monkeypatch.setattr(dw, "_sizing_phases", price)
    monkeypatch.setattr(dw, "fit_ladder", lambda **kw: pytest.fail("proven floors must stop before search"))
    with pytest.raises(MemoryAdmissionError, match="resizing cannot help") as caught:
        tc.plan_cyclone(**INTENT, **hardware, tiles=tiles)
    assert caught.value.memory["reason"] == "fixed-floor"
    assert caught.value.memory["bound_by"] == "computer"
    assert caught.value.memory["resident_fixed_floor_bytes"] == 11 * GIB
    assert caught.value.memory["streaming_fixed_floor_bytes"] == 12 * GIB
    probes = [exp for exp in seen if is_resident_probe(exp, tiles)]
    assert len(seen) - len(probes) == 1
    assert len(probes) == (0 if tiles == "off" else 1)
    # The refusal law's harder half: nothing was priced here, so nothing
    # is offered -- but what would MOVE this is known, and is named, the
    # way the softer refusal beside it names "more free VRAM".
    text = str(caught.value)
    assert "What moves this is the budget or the floor itself" in text
    assert f"more free VRAM raises the {10 * GIB} byte budget" in text
    # Neither mode's floor is unexhausted in this fixture, so no mode is
    # named as one whose floor is not what is exhausted.
    assert "own fixed floor is" not in text


def test_a_fixed_floor_refusal_names_the_other_mode_only_as_an_unexhausted_floor(
        monkeypatch, hardware):
    """A `--tiles on` request whose STREAMING floor exhausts the budget
    while the resident floor does not.

    The floor comparison is a measurement and is named as one; the
    resident LAYOUT is not, because on this path no rung is priced.  So
    the sentence says the floor is not what is exhausted there, and says
    plainly that nothing was offered -- the distinction between a
    measured way through and a suggestion."""

    blocked = phases(20 * GIB, resident_floor=9 * GIB, streaming_floor=12 * GIB)
    monkeypatch.setattr(dw, "_sizing_phases", lambda exp, **kw: blocked)
    monkeypatch.setattr(dw, "fit_ladder",
                        lambda **kw: pytest.fail("proven floors must stop before search"))
    with pytest.raises(MemoryAdmissionError) as caught:
        tc.plan_cyclone(**INTENT, **hardware, tiles="on")
    text = str(caught.value)
    assert caught.value.memory["reason"] == "fixed-floor"
    assert f"--tiles off's own fixed floor is {9 * GIB} bytes" in text
    assert "no layout was priced on that road, and none is offered" in text
    # Never a layout, and never an admission claim about that road.
    assert caught.value.memory["resident_alternative"] is None
    assert "authors" not in text


def test_streaming_floor_does_not_preclude_smaller_resident_auto(monkeypatch, hardware):
    calls = []

    probes = []

    def price(exp, **kw):
        if is_resident_coverage_probe(exp):
            probes.append(exp.domains[0].run.nx)
            # 11 GiB against the fixture's 10 GiB budget: resident does not
            # keep the coverage either, so nothing is named.
            return phases(11 * GIB)
        calls.append(exp.domains[0].run.nx)
        assert exp.tiles.mode == "auto"
        if exp.domains[0].run.nx > 170:
            raise dw.DomainFitError("shared process/radiation floor", resource="vram",
                phases=phases(11 * GIB, resident_floor=GIB, streaming_floor=12 * GIB))
        return phases()

    monkeypatch.setattr(dw, "_sizing_phases", price)
    result = tc.plan_cyclone(**INTENT, **hardware)
    assert result["fitting"]["proposed_dimensions"][0] == [170, 136]
    assert calls == [200, 190, 180, 170, 170]
    assert probes == [200] and result["fitting"]["keeps_coverage"] is None


def test_host_limit_can_be_recovered_without_altering_host_or_tile_policy(monkeypatch, hardware):
    calls = shrinking_price(monkeypatch, threshold=160, resource="host")
    result = tc.plan_cyclone(**INTENT, **hardware, tiles="on")
    assert result["fitting"]["proposed_dimensions"] == [[160, 128], [128, 128]]
    assert all(kw["machine"] is hardware["target_machine"] for exp, kw in calls)
    assert all(exp.tiles.mode == "on" for exp, _kw in calls
               if not is_resident_probe(exp, "on"))
    assert [exp for exp, _kw in calls if is_resident_probe(exp, "on")]


@pytest.mark.parametrize("resource", ["host", "vram", "memory"])
def test_exhaustion_is_bounded_and_not_misreported_as_a_fixed_floor(monkeypatch, hardware, resource):
    calls = []

    probes = []

    def refuse(exp, **kw):
        (probes if is_resident_probe(exp) else calls).append(
            exp.domains[0].run.nx)
        raise dw.DomainFitError("fixture admission", resource=resource,
                                phases=phases(11 * GIB, streaming_floor=2 * GIB))

    monkeypatch.setattr(dw, "_sizing_phases", refuse)
    with pytest.raises(MemoryAdmissionError) as caught:
        tc.plan_cyclone(**INTENT, **hardware)
    assert caught.value.memory["reason"] == "bounded-search"
    assert caught.value.memory["resource"] == resource
    assert "resizing cannot help" not in str(caught.value)
    assert calls == [200, 190, 180, 170, 160, 150, 140, 130]
    # The unreduced-coverage probe, then the resident ladder the refusal
    # measures -- every rung of which is refused by this fixture too, so
    # the refusal names no alternative.
    assert probes[0] == 200 and probes[1:] == [190, 180, 170, 160, 150, 140, 130]
    assert caught.value.memory["resident_alternative"] is None


@pytest.mark.parametrize("resource", [None, "geometry", "unknown-host", "unknown-gpu"])
def test_unsupported_or_unknown_admission_is_not_treated_as_resizable(monkeypatch, hardware, resource):
    seen = []

    def refuse(exp, **kw):
        seen.append(exp)
        raise dw.DomainFitError("budget words in a non-memory failure", resource=resource)

    monkeypatch.setattr(dw, "_sizing_phases", refuse)
    with pytest.raises(dw.DomainFitError) as caught:
        tc.plan_cyclone(**INTENT, **hardware)
    assert caught.value.resource == resource and len(seen) == 1


def test_final_proposal_is_readmitted_before_return(monkeypatch, hardware):
    seen = []

    def price(exp, **kw):
        nx = exp.domains[0].run.nx
        if is_resident_probe(exp):
            return phases(11 * GIB)
        seen.append(nx)
        if nx == 170 and seen.count(nx) == 2:
            raise dw.DomainFitError("final host refusal", resource="host", phases=phases())
        return phases(11 * GIB if nx > 170 else 9 * GIB)

    monkeypatch.setattr(dw, "_sizing_phases", price)
    with pytest.raises(MemoryAdmissionError, match="final host refusal"):
        tc.plan_cyclone(**INTENT, **hardware)
    assert seen[-2:] == [170, 170]


@pytest.mark.parametrize("when", [0, 1, 2])
def test_cancellation_stops_before_or_between_planner_calls(monkeypatch, hardware, when):
    seen = []

    def price(exp, **kw):
        seen.append(exp)
        return phases(11 * GIB)

    monkeypatch.setattr(dw, "_sizing_phases", price)
    with pytest.raises(dw.DomainFitCancelled, match="cancelled"):
        tc.plan_cyclone(**INTENT, **hardware, cancelled=lambda: len(seen) >= when)
    assert len(seen) == when


def cli_args(out, *extra):
    parser = argparse.ArgumentParser()
    tc.register_cli(parser.add_subparsers(dest="command", required=True))
    return parser.parse_args(["cyclone-setup", "--cycle", INTENT["cycle"],
        "--point=18,-65", "--hours", "27", "--name", INTENT["name"],
        "--vram-gib", "12", "--out", str(out), *extra])


@pytest.fixture
def cli_hardware(monkeypatch, hardware):
    from gpuwm import companion_query
    monkeypatch.setattr(dw, "_domain_target_hardware", lambda args:
        (hardware["sizing"], hardware["target_machine"], False))
    monkeypatch.setattr(companion_query, "inspect_configuration", lambda path:
        {"config_path": str(path)})
    return hardware


def test_cli_preview_then_exact_acceptance_writes_matching_toml_wps_receipt(
        tmp_path, monkeypatch, capsys, cli_hardware):
    from gpuwm.namelist_import import parse_namelist_text
    shrinking_price(monkeypatch)
    out = tmp_path / "not-created-by-preview" / "cyclone.toml"
    assert tc.main(cli_args(out)) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["kind"] == "proposal" and preview["created"] is False
    assert "config_path" not in preview and not out.parent.exists()
    assert tc.main(cli_args(out, "--accept-fit", preview["fitting"]["fit_id"])) == 0
    saved = json.loads(capsys.readouterr().out)
    assert saved["created"] and not saved["forecast_started"]
    assert saved["kind"] == "configuration" and not saved["fitting"]["review_required"]
    assert out.read_text() == preview["config_text"]
    receipt = json.loads(out.with_suffix(".cyclone.json").read_text())
    assert receipt["fitting"]["fit_id"] == receipt["output_sha256"]
    assert receipt["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    wps = out.with_suffix(".namelist.wps")
    assert receipt["wps_sha256"] == hashlib.sha256(wps.read_bytes()).hexdigest()
    parsed = parse_namelist_text(wps.read_text())
    assert parsed["geogrid"]["e_we"] == [171, 137]
    assert parsed["geogrid"]["e_sn"] == [137, 137]
    assert parsed["geogrid"]["parent_grid_ratio"] == [1, 4]
    assert parsed["share"]["interval_seconds"] == [10800]


def test_new_hardware_result_cannot_silently_replace_reviewed_proposal(
        tmp_path, monkeypatch, capsys, cli_hardware):
    out = tmp_path / "new" / "cyclone.toml"
    shrinking_price(monkeypatch, threshold=170)
    assert tc.main(cli_args(out)) == 0
    preview = json.loads(capsys.readouterr().out)
    shrinking_price(monkeypatch, threshold=160, resource="host")
    assert tc.main(cli_args(out, "--accept-fit", preview["fitting"]["fit_id"])) == 1
    assert "does not match" in json.loads(capsys.readouterr().out)["error"]
    assert not out.parent.exists()


@pytest.mark.parametrize("existing", [".toml", ".namelist.wps", ".cyclone.json"])
def test_approved_proposal_never_overwrites_any_existing_companion(
        tmp_path, monkeypatch, capsys, cli_hardware, existing):
    shrinking_price(monkeypatch)
    out = tmp_path / "cyclone.toml"
    proposal = tc.plan_cyclone(**INTENT, **cli_hardware)
    out.with_suffix(existing).write_bytes(b"previously approved bytes\x00\xff")
    (tmp_path / "unrelated.txt").write_bytes(b"also preserve me")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert tc.main(cli_args(out, "--accept-fit", proposal["fitting"]["fit_id"])) == 1
    assert "never overwrites" in json.loads(capsys.readouterr().out)["error"]
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize("failure", ["memory", "cancel", "unsupported"])
def test_failed_fit_changes_no_existing_files_and_creates_no_output_directory(
        tmp_path, monkeypatch, capsys, cli_hardware, failure):
    approved = tmp_path / "approved.toml"
    approved.write_bytes(b"user-approved\n")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    out = tmp_path / "not-created" / "new.toml"

    def refuse(exp, **kw):
        if failure == "cancel":
            raise KeyboardInterrupt()
        raise dw.DomainFitError("fixture refusal",
                                resource="host" if failure == "memory" else "geometry",
                                phases=phases(11 * GIB))

    monkeypatch.setattr(dw, "_sizing_phases", refuse)
    assert tc.main(cli_args(out)) == (130 if failure == "cancel" else 1)
    result = json.loads(capsys.readouterr().out)
    assert not result["created"] and not result["forecast_started"]
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_publication_failure_rolls_back_new_files_without_touching_existing(
        tmp_path, monkeypatch, capsys, cli_hardware, failure):
    shrinking_price(monkeypatch)
    out = tmp_path / "cyclone.toml"
    proposal = tc.plan_cyclone(**INTENT, **cli_hardware)
    (tmp_path / "approved.toml").write_bytes(b"previous approved configuration")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    original_open = Path.open

    def fail_receipt(path, mode="r", *args, **kwargs):
        if path == out.with_suffix(".cyclone.json") and mode == "x":
            raise failure("fixture publication interruption")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_receipt)
    assert tc.main(cli_args(out, "--accept-fit", proposal["fitting"]["fit_id"])) == (
        130 if failure is KeyboardInterrupt else 1)
    assert not json.loads(capsys.readouterr().out)["created"]
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize("resource", ["vram", "host", "memory", "geometry", None])
def test_shared_sizing_preserves_typed_refusals_and_report_errors(monkeypatch, resource):
    _, exp = tc.configuration_text(**INTENT)
    result = phases()
    result.tree_road = SimpleNamespace(priced=False, refusal="same words", report_error=None,
                                       refusal_resource=resource)
    monkeypatch.setattr(dw, "estimate_phases", lambda *a, **kw: result)
    with pytest.raises(dw.DomainFitError) as caught:
        dw._sizing_phases(exp, free_bytes=11 * GIB, machine=object(), source="gfs")
    assert caught.value.resource == resource and caught.value.phases is result
    result.tree_road.refusal = None
    result.tree_road.report_error = "TypeError: broken report"
    result.tree_road.refusal_resource = None
    with pytest.raises(dw.DomainFitError, match="TypeError: broken report") as caught:
        dw._sizing_phases(exp, free_bytes=11 * GIB, machine=object(), source="gfs")
    assert caught.value.resource is None


@pytest.mark.parametrize("resource", ["vram", "host", "memory", "geometry", None])
def test_tree_floor_metadata_comes_from_shared_terms_and_only_memory_refusals(monkeypatch, resource):
    _, exp = tc.configuration_text(**INTENT)

    def refuse(*a, **kw):
        raise streaming.StreamingRefused("fixture refusal", resource=resource)

    monkeypatch.setattr(streaming, "decide_tree", refuse)
    monkeypatch.setattr(streaming, "_tree_process_overhead_bytes", lambda nodes: 123)
    monkeypatch.setattr(streaming, "_tree_radiation_transient_bytes", lambda nodes: 456)
    result = streaming.tree_road_plan(exp)
    assert result.refusal_resource == resource
    assert result.streaming_fixed_floor_bytes == (579 if resource in {"vram", "host", "memory"} else None)


@pytest.mark.parametrize("scales", [(), [.9, .8], (True,), (".9",), (1., 1.), (.9, 1.), (float("nan"),),
                                     (0.,), (-1.,), tuple(1 / n for n in range(1, 66))])
def test_bounded_fitter_rejects_invalid_or_unbounded_candidate_lists(monkeypatch, scales):
    monkeypatch.setattr(dw, "_sizing_phases", lambda *a, **kw: pytest.fail("invalid search must not price"))
    with pytest.raises(ValueError, match="candidate_scales"):
        dw.fit_ladder(ratios=(4,), free_bytes=11 * GIB, hours=6,
                      start_time=tc._cycle(INTENT["cycle"]), projection={}, source="gfs",
                      name="bounded", candidate_scales=scales)


def test_real_planner_declared_hardware_no_gpu_probe(monkeypatch):
    """Unmocked CPU planner/author round trip, including a reduced config."""
    pytest.importorskip("netCDF4", reason="normal CPU planner runtime dependency")
    from tilestream.autoplan import Machine
    monkeypatch.setattr(Machine, "detect", lambda *a, **kw: pytest.fail("no GPU probe permitted"))
    found_reduction = False
    # 5 GiB is the band where the layout GENUINELY reduces: the requested
    # tree does not fit the card resident there.  MEASURED across these
    # rungs, the tile planner's tree road charging its streamed fixed floor
    # against a card that holds the tree resident cost the whole middle of
    # this sweep: 5 and 6 GiB were REFUSED outright, 7.25 and 7.5 GiB were
    # REDUCED (to 130x104 and 160x128), and only 8 and 12 GiB admitted the
    # requested ground.  5 GiB now proposes a real smaller layout and every
    # rung above it admits the request, which `unreduced` below pins.
    unreduced = set()
    for free_gib in (3., 5., 6., 7.25, 7.5, 8., 12.):
        free = int(free_gib * GIB)
        sizing = dw.SizingBudget(free_gib + .75, free, None, "CPU declaration", measured=False)
        machine = Machine(vram_bytes=free, host_bytes=128 * GIB, name="CPU declaration")
        try:
            result = tc.plan_cyclone(**INTENT, sizing=sizing, target_machine=machine)
        except MemoryAdmissionError:
            continue
        exp = dw.experiment_from_text(result["config_text"], source=INTENT["source"])
        priced = dw._sizing_phases(exp, free_bytes=free, vram_gib=sizing.vram_gib,
                                  machine=machine, source="gfs", profile=sizing.device_profile,
                                  forcing_interval_seconds=10800.)
        budget = dw.sizing_budget_bytes(exp, free_bytes=free, vram_gib=sizing.vram_gib,
                                       profile=sizing.device_profile, forcing_interval_seconds=10800.)
        assert priced.peak_envelope_bytes <= budget - result["memory"]["fit_headroom_bytes"]
        assert tc.plan_cyclone(**INTENT, sizing=sizing, target_machine=machine) == result
        found_reduction |= result["fitting"]["changed"]
        if not result["fitting"]["changed"]:
            unreduced.add(free_gib)
            assert result["streaming"]["road"] == "resident"
            assert result["fitting"]["proposed_dimensions"] == [
                list(tc.ROOT_DIMS), list(tc.CHILD_DIMS)]
    assert found_reduction, "declared hardware sweep must exercise an actual reduced plan"
    assert {6., 7.25, 7.5, 8., 12.} <= unreduced


def test_real_fixed_floor_excludes_grid_sized_column_workspaces():
    pytest.importorskip("netCDF4", reason="normal CPU planner runtime dependency")
    from gpuwm.core import preflight
    _, large = tc.configuration_text(**INTENT)
    _, small = tc.configuration_text(**INTENT, dimensions=[(130, 104), (104, 104)])
    a = preflight.estimate_experiment(large, vram_gib=8., forcing_interval_seconds=10800.)
    b = preflight.estimate_experiment(small, vram_gib=8., forcing_interval_seconds=10800.)
    assert a.column_workspace_bytes > b.column_workspace_bytes
    assert a.envelope_intercept_bytes > b.envelope_intercept_bytes
    assert a.fixed_envelope_bytes == b.fixed_envelope_bytes > 0
    assert a.fixed_envelope_bytes < b.peak_envelope_bytes < a.peak_envelope_bytes


def test_local_host_is_snapshotted_once_and_keeps_the_selected_device(monkeypatch):
    from tilestream.autoplan import Machine
    sizing = dw.SizingBudget(12., 11 * GIB, None, "snapshot", measured=False)
    machine = Machine(vram_bytes=sizing.free_bytes, host_bytes=64 * GIB, name="host snapshot")
    snapshots = []

    def snapshot(**kwargs):
        snapshots.append(kwargs)
        return machine

    monkeypatch.setattr(streaming, "planner_machine", snapshot)
    monkeypatch.setattr(dw, "sizing_budget_bytes", lambda *a, **kw: 10 * GIB)
    calls = shrinking_price(monkeypatch)
    tc.plan_cyclone(**INTENT, sizing=sizing)
    # The profile travels WITH the machine now, rather than being patched
    # onto it a line later: the tree admission takes its device from the
    # machine and from nowhere else.
    assert snapshots == [{"vram_bytes": sizing.free_bytes,
                          "name": "gpuwm cyclone budget",
                          "device_profile": sizing.device_profile}]
    used = calls[0][1]["machine"]
    assert used.host_bytes == machine.host_bytes and used.vram_bytes == sizing.free_bytes
    assert used.device_profile is sizing.device_profile
    assert all(kwargs["machine"] is used for _, kwargs in calls)


@pytest.mark.parametrize("tiles,free_gib,host_gib,reason", [
    ("off", 2., 128., "fixed-floor"),
    ("on", 6., 128., "fixed-floor"),
    ("on", 10., .1, "bounded-search"),
])
def test_real_planner_distinguishes_fixed_floors_from_host_exhaustion(
        monkeypatch, tiles, free_gib, host_gib, reason):
    pytest.importorskip("netCDF4", reason="normal CPU planner runtime dependency")
    from tilestream.autoplan import Machine
    monkeypatch.setattr(Machine, "detect", lambda *a, **kw: pytest.fail("no GPU probe permitted"))
    free = int(free_gib * GIB)
    sizing = dw.SizingBudget(free_gib + .75, free, None, "CPU declaration", measured=False)
    machine = Machine(vram_bytes=free, host_bytes=int(host_gib * GIB), name="CPU declaration")
    with pytest.raises(MemoryAdmissionError) as caught:
        tc.plan_cyclone(**INTENT, tiles=tiles, sizing=sizing, target_machine=machine)
    assert caught.value.memory["reason"] == reason
    assert ("resizing cannot help" in str(caught.value)) == (reason == "fixed-floor")
    if reason == "bounded-search":
        assert caught.value.memory["resource"] == "host"
    else:
        key = "resident_fixed_floor_bytes" if tiles == "off" else "streaming_fixed_floor_bytes"
        assert caught.value.memory[key] >= caught.value.memory["budget_bytes"]


#: What a reduced cyclone proposal costs in SCIENCE, and the remedy that
#: does not cost it.  The 12 km parent is there to carry the steering
#: environment; 2,400x1,920 km carries it and 1,560x1,248 km does not.  On
#: the 6-8 GiB band the tile planner's tree road refused layouts the same
#: card holds resident, so the default door proposed the degraded
#: configuration -- or refused with a hardware claim its own payload
#: contradicted -- while `--tiles off` ran the requested domain, and the
#: door never said so.  Reported, never applied.
def _resident_keeps_coverage(monkeypatch, *, refusals="all"):
    """Auto/on refuses; the unreduced domain priced resident is admitted."""

    calls = []

    def price(exp, **kw):
        calls.append(exp)
        if is_resident_coverage_probe(exp):
            return phases(4 * GIB)          # admitted under the 10 GiB budget
        if refusals == "all" or exp.domains[0].run.nx > 170:
            raise dw.DomainFitError("fixture tree-road admission",
                                    resource="vram",
                                    phases=phases(11 * GIB, resident_floor=GIB,
                                                  streaming_floor=12 * GIB))
        return phases()

    monkeypatch.setattr(dw, "_sizing_phases", price)
    return calls


def test_a_reduced_proposal_names_tiles_off_when_that_keeps_the_ground(
        monkeypatch, hardware):
    calls = _resident_keeps_coverage(monkeypatch, refusals="above-170")
    result = tc.plan_cyclone(**INTENT, **hardware)

    assert result["kind"] == "proposal"
    keeps = result["fitting"]["keeps_coverage"]
    assert keeps["tiles"] == "off"
    assert keeps["dimensions"] == [list(tc.ROOT_DIMS), list(tc.CHILD_DIMS)]
    assert keeps["peak_envelope_bytes"] == 4 * GIB
    assert keeps["budget_bytes"] == 10 * GIB
    notice = result["fitting"]["notice"]
    assert "--tiles off" in notice and "keep the requested coverage" in notice
    # The numbers that make it a claim rather than a suggestion.
    assert str(4 * GIB) in notice and str(10 * GIB) in notice
    # REPORTED, never applied: the proposal is still the reduced auto tree.
    assert result["tiles"] == "auto"
    assert result["fitting"]["proposed_dimensions"] == [[170, 136], [136, 136]]
    assert tomllib.loads(result["config_text"])["tiles"]["mode"] == "auto"
    assert any(is_resident_coverage_probe(exp) for exp in calls)


def test_a_bounded_refusal_names_tiles_off_when_that_keeps_the_ground(
        monkeypatch, hardware):
    _resident_keeps_coverage(monkeypatch)
    with pytest.raises(MemoryAdmissionError) as caught:
        tc.plan_cyclone(**INTENT, **hardware)
    assert "--tiles off" in str(caught.value)
    assert "keep the requested coverage" in str(caught.value)
    assert caught.value.memory["keeps_coverage"]["peak_envelope_bytes"] == 4 * GIB


def test_a_tree_road_refusal_is_not_reported_as_the_computer(
        monkeypatch, hardware):
    """The false hardware claim, pinned.

    At 6 GiB the door refused with "The selected computer cannot admit"
    while its own payload carried a resident fixed floor a gigabyte and a
    half UNDER the budget and `--tiles off` authored a real layout.  A
    refusal that misnames what refused sends the reader after a bigger
    card and hides the remedy that works.
    """

    _resident_keeps_coverage(monkeypatch)
    with pytest.raises(MemoryAdmissionError) as caught:
        tc.plan_cyclone(**INTENT, **hardware)
    text = str(caught.value)
    assert "The selected computer cannot admit" not in text
    assert "No --tiles auto cyclone proposal was admitted" in text
    assert "not the computer" in text
    assert caught.value.memory["bound_by"] == "tiles-auto-tree-road"
    assert caught.value.memory["reason"] == "bounded-search"
    # And the numbers that make the attribution checkable.
    assert (caught.value.memory["resident_fixed_floor_bytes"]
            < caught.value.memory["budget_bytes"])


def test_the_hardware_claim_survives_a_real_fixed_floor(monkeypatch, hardware):
    """The other direction: when the floor really does exhaust the budget,
    the sentence about the computer is true and stays."""

    blocked = phases(20 * GIB, resident_floor=11 * GIB, streaming_floor=12 * GIB)

    def refuse(exp, **kw):
        raise dw.DomainFitError("floor", resource="vram", phases=blocked)

    monkeypatch.setattr(dw, "_sizing_phases", refuse)
    with pytest.raises(MemoryAdmissionError) as caught:
        tc.plan_cyclone(**INTENT, **hardware)
    assert "The selected computer" in str(caught.value)
    assert caught.value.memory["reason"] == "fixed-floor"


def test_nothing_is_named_when_resident_does_not_keep_the_ground(
        monkeypatch, hardware):
    """No claim without the measurement behind it.

    The resident probe is priced by the same estimator against the same
    budget as every other candidate; when it does not fit, the notice says
    nothing about `--tiles off` rather than offering a mode that would
    also be refused.
    """

    calls = shrinking_price(monkeypatch)     # the probe prices 11 GiB > 10 GiB
    result = tc.plan_cyclone(**INTENT, **hardware)
    assert result["fitting"]["keeps_coverage"] is None
    assert "--tiles off" not in result["fitting"]["notice"]
    assert any(is_resident_coverage_probe(exp) for exp, _kw in calls)


def test_an_empty_rung_ladder_refuses_by_name(monkeypatch, hardware):
    """A ladder with no rung left names its breakage and a way out.

    Left alone it surfaced fit_ladder's internal contract message
    ("candidate_scales must be a tuple of 1..64 decreasing positive
    finite scales") to a reader who never chose a scale ladder: no
    breakage named, no remedy named.
    """

    # Through the real _fit_scales, not around it: a parent no wider than
    # its child's footprint leaves no rung with clearance at all.
    monkeypatch.setattr(tc, "_fit_dimensions", lambda scale: [(40, 40), (160, 160)])
    shrinking_price(monkeypatch, threshold=0)
    assert tc._fit_dimensions(1.0) == [(40, 40), (160, 160)]
    with pytest.raises(ValueError) as caught:
        tc.plan_cyclone(**INTENT, **hardware)
    text = str(caught.value)
    assert "candidate_scales must be" not in text
    assert "tracker" in text and "boundary and blend zone" in text
    assert "--tiles off" in text


def test_missing_host_memory_names_the_breakage_and_the_way_out(
        monkeypatch, hardware):
    """The shared planner's own wording, said once and typed.

    It carried no resource, so it escaped with no memory payload, and it
    never said WHAT missing host RAM blocks -- auto tiling cannot be
    admitted without a host store to pin into.
    """

    monkeypatch.setattr(streaming, "planner_machine", lambda **kw: None)
    with pytest.raises(dw.DomainFitError) as caught:
        tc.plan_cyclone(**INTENT, sizing=hardware["sizing"])
    text = str(caught.value)
    assert "needs host RAM available to the shared planner" in text
    assert "run the wizard on the forecast host or use --tiles off" in text
    assert caught.value.resource == "host"


def test_a_cancelled_fit_exits_130_from_the_predicate_seam(monkeypatch, capsys):
    """One cancellation answer for both seams.

    DomainFitCancelled IS a RuntimeError, so the generic handler turned a
    cancelled fit into exit 1 with no `cancelled` key -- indistinguishable
    from a fit that failed by the only two things a caller reads.
    """

    def refuse(*a, **kw):
        raise dw.DomainFitCancelled("Domain fitting cancelled")

    monkeypatch.setattr(tc, "plan_cyclone", refuse)
    monkeypatch.setattr(dw, "_domain_target_hardware",
                        lambda args: (dw.SizingBudget(12., 11 * GIB, None, "x",
                                                      measured=False), None, None))
    args = argparse.Namespace(
        latest_map=False, cycle="2026090900", point="18,-65", hours=27,
        name="Reviewed cyclone", tiles="auto", out=None, accept_fit=None,
        json=True, hardware_json=None, target_host_memory_json=None,
        vram_gib=12., card=None)
    assert tc.main(args) == 130
    document = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert document["cancelled"] is True and document["created"] is False


def test_the_document_schema_is_v2_because_a_result_can_now_be_a_proposal():
    """`kind` gained "proposal", the result gained `fitting`/`created`, and
    `--out` on a proposal exits 0 having written nothing.  A v1 reader is
    correct to treat exit 0 plus `--out` as "the file is there" and wrong
    under this document; the version string is the only part such a reader
    is guaranteed to look at."""

    assert tc.SCHEMA == "arwen.cyclone-setup.v2"


def test_a_refusal_names_the_resident_layout_it_measured(monkeypatch, hardware):
    """A refusal that names no way through is the defect the refusal law is
    about, and "try --tiles off" with no layout behind it is a suggestion.

    So when the requested coverage cannot be kept resident either, the
    refusal runs the same bounded ladder against the resident route and
    reports the dimensions it FOUND -- measured on the same estimator and
    the same budget as every other candidate, and labelled as the
    reduction it is.
    """

    def price(exp, **kw):
        if is_resident_probe(exp):
            # The requested 200x160 does not fit resident; 170x136 does.
            return phases(11 * GIB if exp.domains[0].run.nx > 170 else 4 * GIB)
        raise dw.DomainFitError("fixture tree-road admission", resource="vram",
                                phases=phases(11 * GIB, resident_floor=GIB,
                                              streaming_floor=12 * GIB))

    monkeypatch.setattr(dw, "_sizing_phases", price)
    with pytest.raises(MemoryAdmissionError) as caught:
        tc.plan_cyclone(**INTENT, **hardware)
    found = caught.value.memory["resident_alternative"]
    assert found == {"tiles": "off", "dimensions": [[170, 136], [136, 136]]}
    text = str(caught.value)
    assert "--tiles off is not refused here" in text
    assert "170x136 / 136x136" in text
    assert "review it as a reduction" in text
    assert caught.value.memory["keeps_coverage"] is None


#: A POLAR request, where the card is not the only thing that decides how
#: big a domain grown from a point may be.  The requested 200x160 was
#: authored without ever facing the bounds the fit is bounded by, so
#: above about 81 N the door emitted a 12 km parent whose footprint
#: contains the projection pole -- not pole-capable in lat-lon source
#: interpolation or static-tile windowing -- while the same door, one
#: gigabyte of budget lower, refused a SMALLER pole-reaching rung for
#: exactly that reason.  Reproduced on the real door at --point=82,-20:
#: 9 GiB emitted 200x160 (unpreparable), 8.45 GiB proposed 160x128.
POLAR = dict(INTENT, point=(82., -20.))


def test_a_polar_request_is_sized_off_the_pole_not_emitted_into_it(
        monkeypatch, hardware):
    monkeypatch.setattr(dw, "_sizing_phases", lambda exp, **kw: phases(GIB))
    result = tc.plan_cyclone(**POLAR, **hardware)

    assert result["kind"] == "proposal"
    assert result["fitting"]["proposed_dimensions"] == [[160, 128], [128, 128]]
    assert "reaches the north pole" in result["fitting"]["reason"]
    # The bound the SEARCH stopped on, carried out of the fitter rather
    # than re-derived from the emitted root.
    assert result["fitting"]["stopped_by"]["scope"] == dw.POINT_FIT_PROJECTION_SCOPE
    assert "170 x 136" in result["fitting"]["stopped_by"]["reason"]
    assert "The next larger layout was rejected on the PROJECTION" in         result["fitting"]["notice"]
    # The layout it proposes is one the pole guard passes -- which is the
    # whole point of shrinking rather than emitting.
    projection = tomllib.loads(result["config_text"])["projection"]
    assert not dw._footprint_contains_pole(projection, 160, 128, tc.ROOT_DX_M)


def test_a_request_bound_reduction_is_never_offered_a_tile_mode(
        monkeypatch, hardware):
    """`--tiles off` moves where the bytes land, not where the domain
    sits.  Naming it against a bound no memory mode can fix would send
    the reader after a remedy that cannot help -- and the resident probe
    would have admitted the requested domain here, because memory is not
    what refused it."""

    priced = []

    def price(exp, **kw):
        priced.append(exp)
        return phases(GIB)

    monkeypatch.setattr(dw, "_sizing_phases", price)
    result = tc.plan_cyclone(**POLAR, **hardware)
    assert result["fitting"]["keeps_coverage"] is None
    assert "--tiles off" not in result["fitting"]["notice"]
    # And the probe was never even priced: no claim, and no cost.
    assert not [exp for exp in priced if is_resident_coverage_probe(exp)]


#: The same rule on the case the cell above cannot reach: a request
#: bound BOTH ways at once.
#:
#: The cell above holds `--tiles off` back from a request-bound
#: reduction, but its fixture FITS -- memory never binds there, so the
#: door reaches the request bound through the one branch that tests it.
#: When memory binds too, the memory refusal is raised FIRST and leaves
#: `refusal.resource` reading `vram`, so a gate that asked only "is this
#: a memory refusal" priced the unreduced 200x160 resident, found it
#: admitted, and said `--tiles off` keeps the requested coverage -- in
#: the same notice that then reported the PROJECTION had rejected a
#: SMALLER root for reaching the pole.  One notice, two contradictory
#: claims, and the named remedy did not do what it said.
#:
#: Reproduced on the door at --point=82,-20 --vram-gib 8.45: the notice
#: named `--tiles off` at 200x160 / 160x160 (5,210,171,347 bytes against
#: a 7,730,941,132 byte budget), while `--tiles off` at that point and
#: that budget authors 160x128 / 128x128 -- the layout auto had already
#: proposed.  `fitting.keeps_coverage` carried the same false payload to
#: a machine reader.
def test_a_polar_request_that_binds_on_memory_too_is_offered_no_tile_mode(
        monkeypatch, hardware):
    def price(exp, **kw):
        # The resident route ADMITS the unreduced domain here.  That is
        # the point: nothing about memory suppresses the claim, so what
        # the assertions below see is the request bound doing it.
        if is_resident_probe(exp):
            return phases(4 * GIB)
        return phases(11 * GIB if exp.domains[0].run.nx > 170 else GIB)

    monkeypatch.setattr(dw, "_sizing_phases", price)
    polar = tc.plan_cyclone(**POLAR, **hardware)
    assert polar["fitting"]["keeps_coverage"] is None
    assert "--tiles off" not in polar["fitting"]["notice"]
    # And the notice still says what DID stop the fit, so withholding the
    # remedy does not cost the reader the reason.
    assert "reaches the north pole" in polar["fitting"]["notice"]
    assert polar["fitting"]["proposed_dimensions"] == [[160, 128], [128, 128]]

    # CONTROL: the same fixture, the same budget, the same refusal
    # resource -- one thing different, a point the request bound does not
    # stop.  There the sentence is true and is still named, which is what
    # makes the None above the GATE rather than the pricing.
    ordinary = tc.plan_cyclone(**INTENT, **hardware)
    assert ordinary["fitting"]["keeps_coverage"] == {
        "tiles": "off", "dimensions": [list(tc.ROOT_DIMS), list(tc.CHILD_DIMS)],
        "peak_envelope_bytes": 4 * GIB, "budget_bytes": 10 * GIB}
    assert "--tiles off admits the requested" in ordinary["fitting"]["notice"]


def test_an_ordinary_point_faces_the_same_question_and_is_untouched(
        monkeypatch, hardware):
    """The bound is asked of every request; it binds on almost none."""

    monkeypatch.setattr(dw, "_sizing_phases", lambda exp, **kw: phases(GIB))
    result = tc.plan_cyclone(**INTENT, **hardware)
    assert result["kind"] == "configuration"
    assert result["fitting"]["proposed_dimensions"] == [
        list(tc.ROOT_DIMS), list(tc.CHILD_DIMS)]
    assert result["fitting"]["stopped_by"] is None


#: The SAME misattribution, pointed the other way, on a band the door
#: reaches without --vram-gib: "measured-available" sizing puts a 16 GiB
#: card with other work on it into a budget between the resident fixed
#: floor and what the smallest rung actually costs.  There the floor sits
#: UNDER the budget -- so the floor inference said "a resident tree could
#: start here, the tree road is what refused you" -- while `--tiles off`
#: on the same card was refused too, and the refusal named no way out at
#: all.  Reproduced on the real door at 4, 4.5 and 5 GiB before the fix.
#:
#: The evidence was already on the payload: the refusal path WALKS the
#: resident ladder, and its coming back empty is a measurement that no
#: resident layout fits.  So the tile planner is named only when a
#: resident layout was found.
def test_the_computer_is_named_when_the_resident_ladder_admits_nothing(
        monkeypatch, hardware):
    priced = []

    def refuse(exp, **kw):
        priced.append(exp)
        # A resident fixed floor a long way UNDER the budget -- the
        # inference the old attribution rested on -- and every rung, in
        # both tile modes, refused by the card.
        raise dw.DomainFitError("fixture admission", resource="vram",
                                phases=phases(11 * GIB, resident_floor=GIB,
                                              streaming_floor=2 * GIB))

    monkeypatch.setattr(dw, "_sizing_phases", refuse)
    with pytest.raises(MemoryAdmissionError) as caught:
        tc.plan_cyclone(**INTENT, **hardware)
    text = str(caught.value)
    assert caught.value.memory["bound_by"] == "computer"
    assert "The selected computer cannot admit" in text
    assert "not the computer" not in text and "tree road" not in text
    # The floor inference that used to decide this is still on the
    # payload, still under the budget, and is no longer what decides.
    assert (caught.value.memory["resident_fixed_floor_bytes"]
            < caught.value.memory["budget_bytes"])
    assert caught.value.memory["resident_alternative"] is None
    assert caught.value.memory["keeps_coverage"] is None
    # And the refusal that used to end at "no smaller candidate passed"
    # now names what would move it, with the measurement behind it.
    assert "--tiles off is not a way through here either" in text
    assert "more free VRAM" in text
    assert str(10 * GIB) in text
    # The measurement itself: the resident ladder really was walked.
    assert [exp for exp in priced if is_resident_probe(exp)]


def test_the_tree_road_is_named_only_when_a_resident_layout_was_found(
        monkeypatch, hardware):
    """The other direction of the same rule, on the band the patch serves.

    A resident rung IS admitted here, so what refused the tiled request
    is the tile planner, the refusal says so, and it names the layout."""

    def price(exp, **kw):
        if is_resident_probe(exp):
            return phases(11 * GIB if exp.domains[0].run.nx > 170 else 4 * GIB)
        raise dw.DomainFitError("fixture tree-road admission", resource="vram",
                                phases=phases(11 * GIB, resident_floor=GIB,
                                              streaming_floor=12 * GIB))

    monkeypatch.setattr(dw, "_sizing_phases", price)
    with pytest.raises(MemoryAdmissionError) as caught:
        tc.plan_cyclone(**INTENT, **hardware)
    assert caught.value.memory["bound_by"] == "tiles-auto-tree-road"
    assert "not the computer" in str(caught.value)
    assert caught.value.memory["resident_alternative"] == {
        "tiles": "off", "dimensions": [[170, 136], [136, 136]]}


def test_an_unpriceable_resident_route_licenses_no_claim_either_way(
        monkeypatch, hardware):
    """A resident route that could not be PRICED is not a measurement.

    The hardware sentence stands (nothing was found, so the tile planner
    is not blamed), but the sentence that says `--tiles off` admits
    nothing is withheld: no rung was ever priced to know it.
    """

    def refuse(exp, **kw):
        if is_resident_probe(exp):
            raise ValueError("fixture: this candidate cannot be authored")
        raise dw.DomainFitError("fixture admission", resource="vram",
                                phases=phases(11 * GIB, resident_floor=GIB,
                                              streaming_floor=2 * GIB))

    monkeypatch.setattr(dw, "_sizing_phases", refuse)
    with pytest.raises(MemoryAdmissionError) as caught:
        tc.plan_cyclone(**INTENT, **hardware)
    assert caught.value.memory["bound_by"] == "computer"
    assert "--tiles off is not a way through" not in str(caught.value)


def test_a_resident_request_is_never_told_a_tree_road_refused_it(
        monkeypatch, hardware):
    """`--tiles off` has no tree road to blame.

    The attribution fix must not travel: a resident request refused by the
    card is refused by the card, and naming a tile planner there would be
    the same false claim pointed the other way.
    """

    def price(exp, **kw):
        return phases(11 * GIB, resident_floor=GIB, streaming_floor=None)

    monkeypatch.setattr(dw, "_sizing_phases", price)
    with pytest.raises(MemoryAdmissionError) as caught:
        tc.plan_cyclone(**INTENT, **hardware, tiles="off")
    assert "The selected computer cannot admit" in str(caught.value)
    assert "tree road" not in str(caught.value)
    assert caught.value.memory["bound_by"] == "computer"
    assert caught.value.memory["resident_alternative"] is None


#: The one soundness condition `fixed_envelope_bytes` rests on, held as a
#: test rather than as a sentence in a docstring.
#:
#: `fixed_envelope_bytes` is a LOWER bound only while
#: `column_workspace_bytes` is the ONLY grid-scaling term inside
#: `non_pool_device_bytes`: it subtracts exactly that one term and keeps
#: the rest as grid-independent.  That is true today -- the sum is
#: `cuda_context + kernel_local_memory + column_workspace` -- but nothing
#: made it fail if a fourth, grid-scaling term joined the sum later.  The
#: bound would then silently OVER-estimate, and an over-estimated fixed
#: floor is the refusal `_fixed_floors` turns into "resizing cannot help":
#: a resizable configuration refused as impossible, on a number nobody
#: could see was wrong.
#:
#: Two checks, because either alone is weak.  The identity pins the
#: composition of the sum; the invariance pins the property the identity
#: is there to protect -- that everything left after the subtraction does
#: not move when only the grid does.
def _cyclone_experiment(dimensions=None):
    _text, exp = tc.configuration_text(**INTENT, dimensions=dimensions)
    return exp


def test_the_non_pool_sum_is_the_three_terms_the_fixed_bound_assumes():
    from gpuwm.core import preflight as pf

    exp = _cyclone_experiment()
    profile = pf.MEASURED_LOCAL_MEMORY_PROFILE
    assert pf.non_pool_device_bytes(exp, profile=profile) == (
        profile.cuda_context_bytes
        + pf.kernel_local_memory_bytes(exp, profile=profile)
        + pf.column_workspace_bytes(exp, profile=profile))


def test_everything_the_fixed_bound_keeps_is_grid_independent():
    """Two grids, one ladder, one physics suite: the terms
    `fixed_envelope_bytes` keeps must be equal, and the term it subtracts
    is the one allowed to move."""

    from gpuwm.core import preflight as pf

    profile = pf.MEASURED_LOCAL_MEMORY_PROFILE
    big = _cyclone_experiment()
    small = _cyclone_experiment(tc._fit_dimensions(.65))
    assert [d.run.nx for d in big.domains] != [d.run.nx for d in small.domains]

    def kept(exp):
        return (pf.non_pool_device_bytes(exp, profile=profile)
                - pf.column_workspace_bytes(exp, profile=profile))

    assert kept(big) == kept(small)
    # And the subtracted term is not vacuously zero, which would make the
    # equality above true of a bound that subtracts nothing.
    assert pf.column_workspace_bytes(big, profile=profile) > 0


#: The OTHER half of `_fixed_floors`, held to the same condition as the
#: resident half above.
#:
#: `streaming_fixed_floor_bytes` is `_tree_process_overhead_bytes +
#: _tree_radiation_transient_bytes`, and both of those read
#: `autoplan.footprint_for(node.cfg.run)` -- which sees nx and ny.  The
#: resident half is pinned by the two cells above; this half was not.  If
#: either term scaled with the grid, `_fixed_floors` would price the
#: REQUESTED (largest) grid's floor and turn it into "resizing cannot
#: help" for a `--tiles on` request that a smaller rung would have held:
#: the same silent over-estimate, on the streaming road.
#:
#: It holds because neither term reads a cell count.
#: `Footprint.process_overhead_bytes` is the CUDA context plus the rung's
#: process-fixed bytes, and `radiation_transient_bytes` is a per-rung
#: constant.  What `footprint_for` varies with is the RUNG and nz, never
#: nx/ny -- so this cell fails the moment a grid-scaling term joins
#: either.
def test_the_streaming_fixed_floor_is_grid_independent_too():
    from types import SimpleNamespace as node

    from tilestream import autoplan

    def tree(dimensions=None):
        return [node(cfg=domain)
                for domain in _cyclone_experiment(dimensions).domains]

    def floor(nodes):
        return (streaming._tree_process_overhead_bytes(nodes)
                + streaming._tree_radiation_transient_bytes(nodes))

    big, small = tree(), tree(tc._fit_dimensions(.65))
    assert [n.cfg.run.nx for n in big] != [n.cfg.run.nx for n in small]
    assert floor(big) == floor(small) > 0
    # Not vacuous: the same footprints, asked what the DOMAINS cost, do
    # move with the grid.  So the equality above is a property of these
    # two terms and not of the fixture.
    def resident(nodes):
        return sum(autoplan.footprint_for(n.cfg.run).resident_bytes(
            n.cfg.run.nx * n.cfg.run.ny * n.cfg.run.nz) for n in nodes)

    assert resident(big) > resident(small)
