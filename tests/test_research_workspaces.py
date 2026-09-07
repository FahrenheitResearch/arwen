"""Research creation preserves science, sampled budgets and existing files."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib

import pytest

from gpuwm import domain_wizard as wizard
from gpuwm import research_workspaces as research


def _args(tmp_path, recipe="scenario-convection.gentle", *extra):
    parser = argparse.ArgumentParser()
    research.register_cli(parser.add_subparsers(required=True))
    return parser.parse_args(["research", "create", recipe, "--point=35.3,-97.5",
                              "--cycle=2026-09-05T18", "--hardware-class=8", "--vram-gib=8",
                              f"--out={tmp_path / 'new.toml'}", *extra])


@pytest.fixture
def known_products(monkeypatch):
    return research._native_products()


def test_all_packaged_recipes_validate_against_native_product_inventory(known_products):
    recipes = research.catalog_document()["configurations"]
    assert len(recipes) >= 114
    for recipe in recipes:
        research.validate_recipe(recipe, known_products=known_products)


@pytest.mark.parametrize("hardware_class", research.HARDWARE_CLASSES)
def test_profiles_keep_duration_cadence_minimum_and_controlled_comparison(hardware_class):
    recipes = research.catalog_document()["configurations"]
    for recipe in recipes:
        profile = research.effective_profile(recipe, hardware_class)
        for key in ("forecast_hours", "history_interval_s", "minimum_root_span_km"):
            assert profile["geometry"][key] == recipe["geometry"][key]
        if recipe["method"] == "controlled_scenario":
            assert profile["geometry"] == recipe["geometry"]
            assert profile["nz"] == 49


def test_class_does_not_declare_capacity_and_auto_uses_one_measured_budget(monkeypatch, tmp_path, known_products):
    budget = wizard.SizingBudget(32, int(8.5 * 2**30), None, "test measured free", measured=True)
    calls = []
    def resolve(card, capacity):
        calls.append((card, capacity))
        return budget
    monkeypatch.setattr(wizard, "resolve_sizing_budget", resolve)
    original = wizard.domain_main
    admitted = []
    def create(native, *, sizing_budget):
        admitted.append(sizing_budget)
        return original(native, sizing_budget=sizing_budget)
    monkeypatch.setattr(wizard, "domain_main", create)
    args = _args(tmp_path, "scenario-convection.gentle", "--hardware-class=32")
    args.vram_gib = None
    receipt = research.create_workspace(args)
    assert calls == [(None, None)]
    assert admitted == [budget]
    assert receipt["hardware"]["capacity_gib"] == 32
    assert receipt["hardware"]["free_bytes"] == budget.free_bytes
    assert receipt["hardware"]["basis"] == "measured"
    assert receipt["effective_profile"]["nz"] == 49
    assert receipt["admission"]["envelope_budget_bytes"] < budget.free_bytes


def test_automatic_class_uses_free_memory_and_keeps_explicit_class_independent():
    busy = wizard.SizingBudget(32, int(8.5 * 2**30), None, "busy 32 GiB", measured=True)
    assert research._hardware_class(busy, "auto") == "8"
    assert research._hardware_class(busy, "32") == "32"
    assert "8.50 GiB free" in research._budget_document(busy, "auto")["selection_reason"]
    declared = wizard.resolve_sizing_budget(None, 10)
    assert research._hardware_class(declared, "auto") == "8"
    for capacity, free, expected in [
        (15.5084, 15.28, "16"),  # measured physical RTX 5070 Ti
        (31.3, 30.5, "32"),
        (32, 8.5, "8"),
        (15.5084, 15.039, "12"),  # below class 16's 15.04 GiB allowance
        (32, 29.99, "24"),
    ]:
        sampled = wizard.SizingBudget(capacity, int(free * 2**30), None, "sampled", measured=True)
        assert research._hardware_class(sampled, "auto") == expected
        result = research._budget_document(sampled, "auto")
        assert result["capacity_gib"] == capacity
        assert result["free_bytes"] == sampled.free_bytes


def test_measured_5070ti_profile_survives_native_postcreation_check(monkeypatch, tmp_path, known_products):
    from gpuwm.core import preflight
    profile = preflight.DeviceLocalMemoryProfile(
        name="NVIDIA GeForce RTX 5070 Ti", multiprocessor_count=70,
        max_threads_per_multiprocessor=1536, default_stack_limit_bytes=1024,
        bare_context_bytes=241172480)
    budget = wizard.SizingBudget(15.5084228515625, 16409559040, profile,
                                 "observed physical 5070 Ti", measured=True)
    calls = []
    def resolve(card, capacity):
        calls.append((card, capacity))
        return budget
    def repeated_probe(*args, **kwargs):
        raise AssertionError("An immutable measured sizing sample must not be probed again")
    monkeypatch.setattr(wizard, "resolve_sizing_budget", resolve)
    monkeypatch.setattr(preflight, "declares_the_local_card", repeated_probe)
    monkeypatch.setattr(preflight, "live_device_local_memory_profile", repeated_probe)
    monkeypatch.setattr(preflight, "device_memory_probe_subprocess", repeated_probe)
    args = _args(tmp_path, "regional-evolution.reference", "--hardware-class=auto")
    args.vram_gib = None
    receipt = research.create_workspace(args)
    assert calls == [(None, None)]
    assert receipt["hardware"]["selected_class"] == "16"
    assert receipt["hardware"]["device_profile"] == vars(profile)
    assert receipt["hardware"]["free_bytes"] == budget.free_bytes
    assert receipt["admission"]["envelope_budget_bytes"] == budget.free_bytes - preflight.EXTERNAL_MARGIN_BYTES


@pytest.mark.parametrize("key,value", [
    ("alloc", True), ("budget_gib", 8), ("rail_mib", 8192),
    ("vram_gib", 32), ("free_gib", 16),
])
def test_internal_measured_check_refuses_changed_budget_before_reading_config(key, value):
    from gpuwm.cli import build_parser
    from gpuwm.core.preflight import check_main
    budget = wizard.SizingBudget(16, 15 * 2**30, None, "sampled", measured=True)
    args = build_parser().parse_args(["check", "must-not-be-read.toml", "--free-gib=15", "--vram-gib=16"])
    args._shared_sizing_budget = budget
    setattr(args, key, value)
    with pytest.raises(ValueError, match="internal sizing sample"):
        check_main(args)


@pytest.mark.parametrize("recipe_id,amplitude,rh", [
    ("scenario-convection.gentle", 1, False),
    ("scenario-convection.stronger", 2, False),
    ("scenario-convection.rh", 1, True),
])
def test_native_creation_honors_reviewed_bubble_and_exact_diagnostics(tmp_path, known_products, recipe_id, amplitude, rh):
    # Catalog ids can change editorially; select the declared scientific case.
    recipe = next(row for row in research.catalog_document()["configurations"]
                  if row["scenario"] and row["scenario"]["amplitude_k"] == amplitude
                  and row["scenario"]["rh_preserve"] == rh)
    receipt = research.create_workspace(_args(tmp_path, recipe["id"]))
    config = tmp_path / "new.toml"
    raw = tomllib.loads(config.read_text())
    bubble = raw["perturbation"]["bubbles"][0]
    assert bubble == {"center_lat": 35.3, "center_lon": -97.5, "center_height_m": 1500,
                      "radius_km": 10, "depth_m": 1500, "amplitude_k": amplitude, "rh_preserve": rh}
    assert len(raw["domain"]) == 2
    sidecar = json.loads(Path(str(config) + ".arwen-plots.json").read_text())
    assert sidecar["products"] == ",".join(recipe["diagnostics"])
    assert sidecar["schema"] == "gpuwm-tui-plots-v1"
    saved = json.loads(Path(str(config) + ".arwen-research.json").read_text())
    assert saved["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert saved["admission"]["science_validation"] == "unvalidated"
    assert saved["admission"]["forecast_started"] is False
    assert receipt["domains"][1]["dx_km"] == 4
    assert ".arwen-research-" not in json.dumps(saved)
    assert ".arwen-research-" not in config.read_text()
    assert Path(raw["fetch"]["out"]).is_absolute()
    assert (tmp_path / "new.namelist.wps").is_file()


def test_moving_recipe_uses_effective_inner_domain_and_native_units(tmp_path, known_products):
    recipe = next(row for row in research.catalog_document()["configurations"]
                  if row["method"] == "moving_nest" and row["tracker"]["field"] == "uh")
    receipt = research.create_workspace(_args(tmp_path, recipe["id"], "--hardware-class=16", "--vram-gib=16"))
    raw = tomllib.loads((tmp_path / "new.toml").read_text())
    assert len(raw["domain"]) == 3
    assert raw["relocation"]["grid_id"] == 3
    assert raw["relocation"]["follow"]["field"] == "uh"
    assert raw["relocation"]["follow"]["threshold"] == recipe["tracker"]["threshold"]
    assert raw["relocation"]["follow"]["fallback_threshold"] == recipe["tracker"]["fallback_threshold"]
    assert raw["relocation"]["track"] == {"path": "storm-track.csv"}
    assert receipt["effective_profile"]["geometry"]["nest_ratios"] == [4, 3]
    assert any("statics-corridor" in step for step in receipt["required_next_steps"])


@pytest.mark.parametrize("recipe", ["rotation.follow", "tropical-track.follow"])
def test_follow_disclosure_matches_native_placement_edges_and_plan_track(tmp_path, known_products, recipe):
    from dataclasses import replace
    from types import SimpleNamespace
    from gpuwm.core.nest_relocation import _prevalidate_placement
    from gpuwm.runplan import PLAN_SCHEMA, load_plan, resolve_plan

    receipt = research.create_workspace(_args(tmp_path, recipe, "--hardware-class=16", "--vram-gib=16"))
    config = tmp_path / "new.toml"
    original = config.read_bytes()
    exp = wizard.experiment_from_text(config.read_text(), source=str(config))
    moving = receipt["geometry_review"]["moving_nest"]
    child = next(domain for domain in exp.domains if domain.grid_id == moving["grid_id"])
    parent = next(domain for domain in exp.domains if domain.grid_id == moving["parent_id"])
    assert child.parent_id == parent.grid_id
    assert moving["parent_span_x_km"] == parent.run.nx * parent.run.dx / 1000
    assert moving["parent_span_y_km"] == parent.run.ny * parent.run.dy / 1000
    assert moving["placement_status"] == "native_stencil_checked"
    assert moving["centroid_radius_km"] == 50.0
    assert "unvalidated" in moving["centroid_notice"]
    starts = moving["admissible_starts_parent_cells"]
    parent_node = SimpleNamespace(cfg=parent)
    for i in starts["i_parent_start"]:
        for j in starts["j_parent_start"]:
            _prevalidate_placement(replace(child, i_parent_start=i, j_parent_start=j), parent_node)
    for key, (low, high) in starts.items():
        for position in (low - 1, high + 1):
            with pytest.raises(ValueError, match="outside the parent"):
                _prevalidate_placement(replace(child, **{key: position}), parent_node)
    assert "do not guarantee" in receipt["geometry_review"]["coverage_notice"]
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps({"schema": PLAN_SCHEMA, "name": "follow-review",
        "route": "prepared", "config": {"path": str(config)},
        "output_root": str(tmp_path / "no-forecast")}), encoding="utf-8")
    _, resolved, _ = resolve_plan(load_plan(plan_file), require_inputs=False)
    assert resolved.relocation.track.path == "storm-track.csv"
    assert resolved.relocation.track.interval_seconds is None
    assert config.read_bytes() == original
    assert not (tmp_path / "no-forecast").exists()
    assert not (tmp_path / "storm-track.csv").exists()


def test_fixed_research_creation_prints_actual_spans_and_finite_coverage(tmp_path, known_products, capsys):
    assert research.research_main(_args(tmp_path)) == 0
    output = capsys.readouterr().out
    assert "Coverage: d01" in output and "d02" in output
    assert "Fixed domains have finite coverage" in output
    saved = json.loads((tmp_path / "new.toml.arwen-research.json").read_text())
    assert "moving_nest" not in saved["geometry_review"]
    assert len(saved["geometry_review"]["domain_spans"]) == 2
    assert "[relocation.track]" not in (tmp_path / "new.toml").read_text()


def test_existing_configuration_is_preserved_before_any_probe(tmp_path, monkeypatch, known_products):
    config = tmp_path / "new.toml"
    original = b"# user settings\n"
    config.write_bytes(original)
    monkeypatch.setattr(wizard, "resolve_sizing_budget", lambda *args: pytest.fail("should not probe"))
    with pytest.raises(FileExistsError, match="preserves existing"):
        research.create_workspace(_args(tmp_path))
    assert config.read_bytes() == original


def test_existing_companion_is_preserved_and_no_partial_bundle(tmp_path, known_products):
    companion = tmp_path / "new.namelist.wps"
    companion.write_bytes(b"user's WPS\n")
    with pytest.raises(FileExistsError, match="preserves existing"):
        research.create_workspace(_args(tmp_path))
    assert list(tmp_path.iterdir()) == [companion]
    assert companion.read_bytes() == b"user's WPS\n"


def test_create_only_publication_rolls_back_on_racing_companion(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "new.namelist.wps").write_bytes(b"ours")
    (stage / "new.toml").write_bytes(b"ours")
    original_link = research.os.link
    def racing_link(source, destination):
        if destination.name == "new.toml":
            destination.write_bytes(b"another creator")
        return original_link(source, destination)
    monkeypatch.setattr(research.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        research._publish_bundle(stage, tmp_path / "new.toml")
    assert not (tmp_path / "new.namelist.wps").exists()
    assert (tmp_path / "new.toml").read_bytes() == b"another creator"


def test_unknown_scenario_field_cannot_be_silently_dropped(known_products):
    recipe = deepcopy(research._recipe("scenario-convection.gentle"))
    recipe["scenario"]["hurricane_insertion"] = True
    with pytest.raises(ValueError, match="Unknown scenario"):
        research.validate_recipe(recipe)


@pytest.mark.parametrize("method", ["archived_downscale", "regional"])
def test_required_existing_state_refuses_without_probing(method, tmp_path, monkeypatch, known_products):
    recipe = next(row for row in research.catalog_document()["configurations"]
                  if row["method"] == method and row["requires_existing_state"])
    monkeypatch.setattr(wizard, "resolve_sizing_budget", lambda *args: pytest.fail("should not probe"))
    with pytest.raises(ValueError, match="requires"):
        research.create_workspace(_args(tmp_path, recipe["id"]))
    assert not list(tmp_path.iterdir())


def test_insufficient_scientific_extent_refuses_without_publication(tmp_path, monkeypatch, known_products):
    original = research._recipe
    def require_impossible_extent(recipe_id, **kwargs):
        recipe = deepcopy(original(recipe_id, **kwargs))
        recipe["geometry"]["minimum_root_span_km"] = 100000
        return recipe
    monkeypatch.setattr(research, "_recipe", require_impossible_extent)
    with pytest.raises(ValueError, match="minimum span"):
        research.create_workspace(_args(tmp_path))
    assert not list(tmp_path.iterdir())


def test_eight_gib_refusal_points_to_a_fitting_profile_for_the_same_question(tmp_path, known_products):
    for hardware_class in ("24", "32"):
        args = _args(tmp_path, "tropical-track.steering", f"--hardware-class={hardware_class}")
        args.point = "25,-80"
        with pytest.raises(ValueError, match="1200 km minimum span") as failed:
            research.create_workspace(args)
        assert "Retry this same question (tropical-track.steering)" in str(failed.value)
        assert "--hardware-class 8" in str(failed.value)
        assert not list(tmp_path.iterdir())
    args.hardware_class = "8"
    receipt = research.create_workspace(args)
    assert receipt["recipe"]["id"] == "tropical-track.steering"
    assert receipt["hardware"]["capacity_gib"] == 8
    assert receipt["hardware"]["selected_class"] == "8"
    assert min(receipt["domains"][0]["span_x_km"], receipt["domains"][0]["span_y_km"]) >= 1200
    assert receipt["admission"]["forecast_started"] is False


def test_registry_exposes_exact_gui_flags_and_explicit_overrides(tmp_path):
    from gpuwm.cli import build_parser
    parsed = build_parser().parse_args([
        "research", "create", "recipe.with.dots", "--polygon=polygon.json", "--source=gfs",
        "--cycle=2026-09-05T18", "--hours=3", "--hardware-class=32", "--out=new.toml",
        "--name=Study", "--physics-profile=morrison", "--nz=64", "--tiles=auto"])
    assert parsed.configuration_id == "recipe.with.dots"
    assert parsed.vram_gib is None
    assert parsed.nz == 64
    assert parsed.func is research.research_main


@pytest.mark.parametrize("source", ["gfs", "gdas", "hrrr", "era5", "20crv3"])
def test_native_sources_preserve_companions_and_labels_are_not_data_paths(tmp_path, known_products, source):
    cycle = "2015-09-05T18" if source == "20crv3" else "2020-09-05T18"
    # This checks source companions, using enough capacity for HRRR's default suite.
    args = _args(tmp_path, "regional-evolution.reference", f"--source={source}", "--hours=3",
                 f"--cycle={cycle}", "--name=Research: local / coast", "--vram-gib=16")
    receipt = research.create_workspace(args)
    config = tmp_path / "new.toml"
    raw = tomllib.loads(config.read_text())
    assert raw["experiment"]["name"] == "Research: local / coast"
    assert receipt["source"] == source
    if "fetch" in raw:
        assert Path(raw["fetch"]["out"]) == tmp_path / "data" / "new"
    assert ".arwen-research-" not in config.read_text()
    if source == "era5":
        case = raw["case_data"]
        assert case["wps_namelist"] == "new.namelist.wps"
        assert case["vtable"] == "new.Vtable.ERA5_CDO"
        assert (tmp_path / case["vtable"]).read_bytes() == wizard._PACKAGED_VTABLE.read_bytes()
        assert [Path(value) for value in case["forcing"]] == [
            tmp_path / "data" / "new" / "era5-combined.grib"]
    if source == "hrrr":
        assert {"new.namelist.input", "new.stock.namelist.input", "new.d01-target.json"} <= {
            path.name for path in tmp_path.iterdir()}


@pytest.mark.parametrize("source", ["era5", "gdas", "hrrr"])
def test_unadmitted_controlled_source_refuses_before_probe(tmp_path, known_products, monkeypatch, source):
    monkeypatch.setattr(wizard, "resolve_sizing_budget", lambda *args: pytest.fail("should not probe"))
    with pytest.raises(ValueError, match="Controlled-scenario creation"):
        research.create_workspace(_args(tmp_path, "scenario-convection.gentle", f"--source={source}"))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("hardware_class", ["8", "12"])
def test_broad_storm_context_keeps_one_km_and_root_study_extent(hardware_class, known_products):
    broad = research.effective_profile(research._recipe("rotation.structure"), hardware_class)
    local = research.effective_profile(research._recipe("downburst.pulse"), hardware_class)
    assert broad["geometry"]["root_dx_km"] == 12
    assert broad["geometry"]["nest_ratios"] == [4, 3]
    assert broad["effective_finest_dx_km"] == 1
    assert "600 km root span" in broad["tradeoff"]
    assert local["geometry"]["root_dx_km"] == 4
    assert local["geometry"]["nest_ratios"] == [4]


def test_native_attribute_request_preserves_registry_fields_and_omits_display_units(known_products):
    recipe = deepcopy(research._recipe("rotation.follow"))
    recipe["tracker"] = {"kind": "feature_follow", "field": "attribute", "attribute": "theta",
                         "extremum": "min", "reduction": "model_level", "model_level": 2,
                         "units": "K", "threshold": 298, "cadence_seconds": 300,
                         "level_hpa": None, "fallback_threshold": None, "notes": "Explicit cold pool study"}
    research.validate_recipe(recipe)
    text = research._method_text(recipe, lat=35.3, lon=-97.5, grid_id=3)
    follow = tomllib.loads(text)["relocation"]["follow"]
    assert follow["field"] == "attribute"
    assert follow["attribute"] == "theta"
    assert follow["extremum"] == "min"
    assert follow["reduction"] == "model_level"
    assert follow["model_level"] == 2
    assert "units" not in follow
    assert "level_hpa" not in follow
    assert "fallback_threshold" not in follow
    recipe["tracker"]["units"] = "degC"
    with pytest.raises(ValueError, match="requires units K"):
        research.validate_recipe(recipe)


def test_help_explanation_flag_works_at_nested_research_boundary(tmp_path):
    from gpuwm.cli import build_parser
    args = build_parser().parse_args(["research", "hardware", "--vram-gib=8", "--explain"])
    assert args.explain is True


def test_attribute_discovery_exposes_native_registry_without_probing_or_creating(monkeypatch, capsys):
    from gpuwm.cli import build_parser
    from gpuwm.core.attribute_tracking import ATTRIBUTE_UNITS, ATTRIBUTE_EXTREMA, ATTRIBUTE_REDUCTIONS
    monkeypatch.setattr(wizard, "resolve_sizing_budget", lambda *args: pytest.fail("discovery must not probe GPU"))
    monkeypatch.setattr(research, "create_workspace", lambda *args: pytest.fail("discovery must not create files"))
    args = build_parser().parse_args(["research", "attributes", "--json"])
    assert args.func(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema"] == "arwen.research.attributes.v1"
    assert {row["id"]: row["units"] for row in document["attributes"]} == dict(ATTRIBUTE_UNITS)
    assert document["extrema"] == list(ATTRIBUTE_EXTREMA)
    assert document["reductions"] == list(ATTRIBUTE_REDUCTIONS)
    assert document["model_level"]["minimum"] == 0
    assert document["model_level"]["allowed_when"] == "reduction = model_level"
    assert "unweighted" in document["reduction_semantics"]["column_mean"].lower()
    assert all(row["requires_moist"] == (row["id"] in {"qv", "qc", "qr"}) for row in document["attributes"])
