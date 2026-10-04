from datetime import datetime, timedelta, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.automatic_sources import (
    POLICY_SCHEMA, DomainSourceContext, EnsembleSourceContext, FileAuthority,
    NativeSourceTemplate, SourceSelectionPolicy, resolve_ensemble_sources)
from gpuwm.ensemble.recipes import SourceTrajectory, build_recipe

START = datetime(2026, 10, 1, 18, tzinfo=timezone.utc)


def context(source="gfs", *, start=START, hours=12, **kwargs):
    return EnsembleSourceContext(start, start + timedelta(hours=hours), object(),
        None if source is None else SourceTrajectory(source, start),
        "supplied-artifact" if source is None else "requested-trajectory", **kwargs)


def selection_evidence(tmp_path, *, kind="recentered", amplitude=0.5, members=20):
    from gpuwm.ensemble.calibrated_policy import reference
    def save(name, document):
        path = tmp_path / name
        path.write_text(json.dumps(document))
        return path, reference(path)
    split = {"training_cases": ["train-a", "train-b"], "held_out_cases": ["hold-a"]}
    _, plan = save("plan.json", {"schema": "ensemble-calibration.amplitude-selection.v1", **split,
        "scored_lead_hours": [3, 6, 12], "recenter_amplitudes": [0.5, 1.0, 1.5], "stochastic_amplitudes": [0.5, 1.0]})
    _, campaign = save("campaign.json", {"schema": "gpuwm-ensemble-calibration.campaign.v1", **split,
        "scored_leads": [3, 6, 12], "members": members, "time_lag_members": 12, "time_lag_max_age_hours": 36,
        "donor_source": "gefs", "stochastic_on": ["sppt", "skebs", "spp_pbl", "spp_lsm"]})
    selected = {"schema": "gpuwm-ensemble-amplitude-selection-result.v1", **split,
        "status": "training-selected-not-held-out-validated", "selection_plan": plan, "campaign_plan": campaign,
        "products": ["temperature_2m"], "selected_recenter_amplitude": amplitude}
    _, prior = save("recenter-selection.json", {**selected, "stage": "recenter"})
    winner = {"recipe": kind, "stochastic_amplitude": 0, "arm": "selected-arm"}
    selection, _ = save("training-selection.json", {**selected, "stage": "stochastic-and-default",
        "previous_selection": prior, "selected_stochastic_amplitude": 0.5, "selected_default": winner,
        "default_ranking": [{**winner, "objective": 0.8}],
        "declared_arms": [{"score_recipe": "selected-arm", "kind": kind, "stochastic_amplitude": 0,
                           "recenter_amplitude": amplitude}]})
    count = 12 if kind == "time-lagged" else members
    held_out, _ = save("held-out.json", {"schema": "gpuwm-ensemble-campaign-scores.v1", "status": "complete",
        "lead_hours": [3, 6, 12], "products": [{"case_id": "hold-a", "held_out": True, "recipe": "selected-arm",
            "quantity": "temperature_2m", "members": count, "common_observation_mask": {"sha256": "a"*64},
            "scores": {"samples": 17, "missing_members": 0, "crps": 1.0, "members": count}}]})
    return selection, held_out


def policy(tmp_path, *, source="hrrr", kind="recentered", amplitude=0.5):
    selection, held_out = selection_evidence(tmp_path, kind=kind, amplitude=amplitude)
    result = SourceSelectionPolicy.from_training_selection(selection, held_out, policy_id="test-only",
        source_defaults={source: {"max_donor_age_hours": 6}})
    path = tmp_path / "policy.json"
    path.write_text(result.document_json + "\n")
    return SourceSelectionPolicy.load(path), path, held_out


def test_singleton_preserves_ordinary_inputs_without_policy_or_source_io(tmp_path):
    value = context("hrrr", authorities=(FileAuthority("removed", tmp_path / "missing", "bad"),))
    selection = resolve_ensemble_sources(1, value)
    assert selection.mode == "ordinary" and selection.recipe is None
    assert selection.stochastic is None and selection.policy_sha256 == ""


@pytest.mark.parametrize("source,expected", [
    ("gfs", [("gefs", "c00"), ("gefs", "p01"), ("gefs", "p02")]),
    ("ifs", [("ecmwf-open-data", None), ("ecmwf-ens", "p01"), ("ecmwf-ens", "p02")]),
    ("aigfs", [("aigefs", "mem000"), ("aigefs", "mem001"), ("aigefs", "mem002")]),
])
def test_operational_selection_uses_real_registered_member_population(source, expected):
    # IFS's selected cycles must be supported by its own adapter as well.
    value = context(source, start=START.replace(hour=12))
    selection = resolve_ensemble_sources({"members": 3, "base_seed": 99}, value)
    assert selection.mode == "source-recipe"
    assert [(member.trajectory.source, member.trajectory.member) for member in selection.recipe.members] == expected
    assert selection.recipe.kind == "input-ensemble"
    assert selection.calibration_evidence_json is None


def test_invalid_operational_population_keeps_actionable_diagnostic():
    with pytest.raises(ValueError, match="supplies 31 distinct trajectories"):
        resolve_ensemble_sources(32, context())
    with pytest.raises(ValueError, match="ends at"):
        resolve_ensemble_sources(3, context(hours=1000))


def test_unknown_artifact_refuses_uncalibrated_reference_fallback():
    value = context(None)
    with pytest.raises(ValueError, match="calibrated against observations"):
        resolve_ensemble_sources(3, value)
    assert value.describe()["trajectory"] is None


def test_cam_requires_measured_or_explicit_controls():
    with pytest.raises(ValueError, match="no fitted automatic source policy"):
        resolve_ensemble_sources(3, context("hrrr"))
    with pytest.raises(ValueError, match="calibrated against observations"):
        resolve_ensemble_sources({"members": 3, "stochastic": {"sppt": True}}, context("hrrr"))


def test_analysis_without_operational_ensemble_refuses_uncalibrated_fallback():
    with pytest.raises(ValueError, match="calibrated against observations"):
        resolve_ensemble_sources(3, context("era5"))


def test_measured_policy_retains_donor_population_amplitude_and_native_brackets(tmp_path):
    measured, _, _ = policy(tmp_path)
    value = context("hrrr", start=START + timedelta(hours=1))
    selection = resolve_ensemble_sources({"members": 20, "base_seed": 77}, value, policy=measured)
    assert selection.amplitude == 0.5
    assert selection.recipe.calibration == "policy:" + measured.sha256
    assert selection.recipe.start == value.start
    assert len(selection.recipe.donor_population) == 30
    assert all(item.cycle == START for item in selection.recipe.donor_population)
    assert selection.recipe.acquisition_window(selection.recipe.members[0].trajectory) == (0, 15)
    assert selection.calibration_evidence_json is not None
    replay = resolve_ensemble_sources({"members": 20, "base_seed": 77}, value, policy=measured)
    assert selection.sha256 == replay.sha256


def test_policy_and_scoring_evidence_are_byte_pinned(tmp_path):
    measured, path, evidence = policy(tmp_path)
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="calibration-evidence authority changed"):
        resolve_ensemble_sources(3, context("hrrr"), policy=measured)
    measured, path, _ = policy(tmp_path)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="source-policy authority changed"):
        resolve_ensemble_sources(3, context("hrrr"), policy=measured)


def test_arbitrary_hashed_file_cannot_be_labelled_calibration(tmp_path):
    from gpuwm.ensemble.calibrated_policy import reference
    fake = tmp_path / "fake.json"
    fake.write_text('{"status":"PASS"}')
    with pytest.raises(ValueError, match="final frozen training selection"):
        SourceSelectionPolicy.from_training_selection(fake, fake, source_defaults={"hrrr": {}}, policy_id="invalid")


def test_measured_policy_exports_only_portable_evidence_and_preserves_originals(tmp_path):
    import shutil
    value, _, _ = policy(tmp_path)
    originals = {path: path.read_bytes() for path in tmp_path.glob("*.json")}
    saved = value.write(tmp_path / "portable" / "source_defaults.json")
    assert all(path.read_bytes() == data for path, data in originals.items())
    exported = list((tmp_path / "portable").rglob("*.json"))
    assert exported and all(str(tmp_path) not in path.read_text() for path in exported)
    assert all("original_receipt_sha256" in json.loads(path.read_text()) for path in exported if path.name != "source_defaults.json")
    moved = tmp_path / "relocated"
    shutil.copytree(tmp_path / "portable", moved)
    reopened = SourceSelectionPolicy.load(moved / "source_defaults.json")
    assert reopened.sha256 == saved.sha256
    chosen = resolve_ensemble_sources(3, context("hrrr"), policy=reopened)
    assert chosen.amplitude == 0.5


def test_selected_stochastic_controls_cannot_be_retuned_behind_the_same_amplitude(tmp_path):
    selection, held_out = selection_evidence(tmp_path)
    document = json.loads(selection.read_bytes())
    document["selected_default"]["stochastic_amplitude"] = 0.5
    document["default_ranking"][0]["stochastic_amplitude"] = 0.5
    document["declared_arms"][0]["stochastic_amplitude"] = 0.5
    selection.write_text(json.dumps(document))
    value = SourceSelectionPolicy.from_training_selection(selection, held_out,
        source_defaults={"hrrr": {}}, policy_id="test")
    row = value.document["source_defaults"]["hrrr"]
    assert row["stochastic"]["spp"] == {"conv": 0, "pbl": 1, "lsm": 1}
    changed = value.document
    changed["source_defaults"]["hrrr"]["stochastic"]["sppt"]["stddev"] *= 2
    with pytest.raises(ValueError, match="selected campaign stochastic parameters"):
        SourceSelectionPolicy(json.dumps(changed)).verify()


@pytest.mark.parametrize("change", ["training-leak", "missing-product", "member-count", "retune", "lag-span"])
def test_policy_requires_unchanged_winner_and_complete_held_out_evidence(tmp_path, change):
    from gpuwm.ensemble.calibrated_policy import reference
    kind = "time-lagged" if change == "lag-span" else "recentered"
    selection, held_out = selection_evidence(tmp_path, kind=kind)
    if change in {"training-leak", "missing-product", "member-count"}:
        document = json.loads(held_out.read_bytes())
        if change == "training-leak": document["products"][0]["case_id"] = "train-a"
        if change == "missing-product": document["products"] = []
        if change == "member-count": document["products"][0]["members"] = 2
        held_out.write_text(json.dumps(document))
        with pytest.raises(ValueError, match="held-out|Held-out"):
            SourceSelectionPolicy.from_training_selection(selection, held_out, source_defaults={"hrrr": {}}, policy_id="invalid")
    else:
        value = SourceSelectionPolicy.from_training_selection(selection, held_out,
            source_defaults={"hrrr": {}}, policy_id="test")
        document = value.document
        document["source_defaults"]["hrrr"]["max_lag_hours" if change == "lag-span" else "amplitude"] = 24 if change == "lag-span" else 1.5
        with pytest.raises(ValueError, match="calibrated time-lag|measured donor or recenter"):
            SourceSelectionPolicy(json.dumps(document)).verify()


def test_measured_multimodel_rule_keeps_original_ids_and_native_cycle_alignment(tmp_path):
    from gpuwm.ensemble.calibrated_policy import reference, trajectories_for_rule
    selection, held_out = selection_evidence(tmp_path, kind="multi-model", members=3)
    rule = [{"source": source, "member": None, "cycle_anchor": "latest-native-cycle", "cycle_offset_hours": 0}
            for source in ("hrrr", "rap", "gfs")]
    cases = {}
    for case, cycle in (("train-a", START), ("train-b", START-timedelta(days=1)), ("hold-a", START-timedelta(days=2))):
        cases[case] = {"multi-model": {"base": {"cycle": cycle.isoformat()},
            "members": [{"trajectory": {"source": item.source, "cycle": item.cycle.isoformat(), "member": item.member}}
                        for item in trajectories_for_rule(rule, cycle)]}}
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({"schema": "ensemble-calibration.rosters.v1", "cases": cases}))
    value = SourceSelectionPolicy.from_training_selection(selection, held_out, policy_id="test-only",
        source_defaults={"hrrr": {"trajectory_rule": rule, "selection_order": [0, 2, 1], "roster_evidence": reference(roster)}})
    chosen = resolve_ensemble_sources({"members": 2, "base_seed": 17}, context("hrrr", start=START+timedelta(hours=1)), policy=value)
    assert chosen.recipe.kind == "multi-model"
    assert [member.index for member in chosen.recipe.members] == [0, 2]
    assert [member.trajectory.source for member in chosen.recipe.members] == ["hrrr", "gfs"]
    assert chosen.recipe.members[1].trajectory.cycle == START
    with pytest.raises(ValueError, match="supplies 3 distinct"):
        resolve_ensemble_sources(4, context("hrrr"), policy=value)
    altered = value.document
    altered["source_defaults"]["hrrr"]["trajectory_rule"][2]["cycle_offset_hours"] = -6
    with pytest.raises(ValueError, match="changed the calibrated source"):
        SourceSelectionPolicy(json.dumps(altered)).verify()


def test_explicit_recipe_preserves_sparse_original_ids_and_requires_amplitude():
    value = context("hrrr")
    full = build_recipe(source="hrrr", cycle=START, start=START, end=value.end,
        count=20, base_seed=77, kind="recentered", donor=SourceTrajectory("gefs", START))
    recipe = full.select_members((19, 7))
    with pytest.raises(ValueError, match="requires its finite"):
        resolve_ensemble_sources(2, value, recipe=recipe)
    selection = resolve_ensemble_sources(2, value, recipe=recipe, amplitude=1.5)
    assert [member.index for member in selection.recipe.members] == [19, 7]
    assert selection.recipe.members == (full.members[19], full.members[7])
    assert len(selection.recipe.donor_population) == 30


def test_measured_time_lag_count_is_not_a_generic_policy_error(tmp_path):
    measured, _, _ = policy(tmp_path, kind="time-lagged")
    with pytest.raises(ValueError, match="supplies .* distinct trajectories"):
        resolve_ensemble_sources(100, context("hrrr"), policy=measured)


def test_original_domain_ids_and_activation_times_are_retained(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("content")
    authority = FileAuthority.capture("experiment", config)
    domains = (DomainSourceContext(1, None, START, (authority,)),
               DomainSourceContext(4, 1, START + timedelta(hours=2), (authority,)))
    value = context(None, domains=domains)
    with pytest.raises(ValueError, match="calibrated against observations"):
        resolve_ensemble_sources(3, value)
    assert [item["grid_id"] for item in value.describe()["domains"]] == [1, 4]
    with pytest.raises(ValueError, match="parent-first"):
        context(None, domains=tuple(reversed(domains)))
    config.write_text("drift")
    with pytest.raises(ValueError, match="experiment authority changed"):
        resolve_ensemble_sources(3, value)


def test_artifact_cycle_and_unverified_binding_cannot_be_invented():
    with pytest.raises(ValueError, match="cannot acquire an inferred"):
        EnsembleSourceContext(START, START + timedelta(hours=12), object(), SourceTrajectory("gfs", START))
    with pytest.raises(ValueError, match="requires its original acquisition"):
        EnsembleSourceContext(START, START + timedelta(hours=12), object(), SourceTrajectory("gfs", START),
                              "verified-acquisition")


def test_native_template_pins_config_and_refuses_source_ownership_changes(tmp_path):
    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\nname='test'\n")
    template = NativeSourceTemplate.capture("gfs_pgrb2_0p25_v1", ("--experiment-config", str(config)))
    value = context(templates=(template,))
    assert value.template_for(value.trajectory) == template
    with pytest.raises(ValueError, match="capture config"):
        value.template_for(SourceTrajectory("gefs", START))
    with pytest.raises(ValueError, match="cannot replace"):
        NativeSourceTemplate.capture("gfs_pgrb2_0p25_v1", ("--source", "gfs"))
    config.write_text("changed")
    with pytest.raises(ValueError, match="changed configuration"):
        template.verify()
