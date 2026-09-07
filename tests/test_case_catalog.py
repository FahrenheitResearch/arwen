"""Case data cannot replace scientific settings or overwrite an existing study."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib
import zipfile

import pytest

from gpuwm import case_catalog as catalog


EXAMPLES = Path(catalog.__file__).parent / "data" / "case-catalog"
NOW = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)


@pytest.fixture
def document():
    return json.loads((EXAMPLES / "example.json").read_text(encoding="utf-8"))


def write_catalog(tmp_path, document, name="cases.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_json_and_toml_examples_are_the_same_data_and_keep_original_bytes():
    left = catalog.load_catalog(EXAMPLES / "example.json")
    right = catalog.load_catalog(EXAMPLES / "example.toml")
    assert left.document == right.document
    assert left.original == (EXAMPLES / "example.json").read_bytes()
    assert right.original == (EXAMPLES / "example.toml").read_bytes()
    assert left.sha256 == hashlib.sha256(left.original).hexdigest()
    assert left.sha256 != right.sha256


def test_distributed_schema_validates_both_authoring_examples():
    import jsonschema
    schema = json.loads(catalog.SCHEMA_PATH.read_text(encoding="utf-8"))
    for name in ("example.json", "example.toml"):
        jsonschema.Draft202012Validator(schema).validate(catalog.load_catalog(EXAMPLES / name).document)


def test_catalog_prose_never_runs_a_command(tmp_path, document, monkeypatch):
    document["cases"][0]["metadata"] = {"command": "touch DO-NOT-CREATE", "environment": {"SECRET": "not a setting"}}
    document["cases"][0]["recommendations"].append({"topic": "unknown-science", "text": "$(run something) is only prose"})
    path = write_catalog(tmp_path, document)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("catalog load executed a process"))
    loaded = catalog.load_catalog(path)
    assert loaded.document["cases"][0]["metadata"]["command"] == "touch DO-NOT-CREATE"
    assert not (tmp_path / "DO-NOT-CREATE").exists()


def test_source_aliases_and_offset_times_normalize_without_changing_original(tmp_path, document):
    option = document["cases"][0]["source_options"][0]
    option.update(source="GFS", cycle_utc="2026-09-01T03:00:00+03:00")
    path = write_catalog(tmp_path, document)
    loaded = catalog.load_catalog(path)
    normalized = loaded.document["cases"][0]["source_options"][0]
    assert normalized["source"] == "gfs"
    assert normalized["cycle_utc"] == "2026-09-01T00:00:00Z"
    assert b"+03:00" in loaded.original


@pytest.mark.parametrize("identity", ["../escape", "C:drive", "has/slash", "UPPER", "con", "a" * 129])
def test_case_ids_cannot_be_paths_or_reserved_filenames(document, identity):
    document["cases"][0]["id"] = identity
    with pytest.raises(catalog.CatalogError, match="safe lowercase ID"):
        catalog.validate_catalog(document)


def test_duplicate_case_ids_and_duplicate_json_choices_are_refused(tmp_path, document):
    document["cases"][1]["id"] = document["cases"][0]["id"]
    with pytest.raises(catalog.CatalogError, match="Duplicate case ID"):
        catalog.validate_catalog(document)
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    with pytest.raises(catalog.CatalogError, match="Duplicate JSON field"):
        catalog.load_catalog(path)


@pytest.mark.parametrize("mutation,match", [
    (lambda row: row["tiers"]["lower"]["bounds_degrees"].update(south=40), "nonzero area"),
    (lambda row: row["tiers"]["lower"].update(root_dx_km=float("nan")), "finite"),
    (lambda row: row["tiers"]["lower"].update(run_hours=True), "finite integer"),
    (lambda row: row["tiers"]["lower"].update(root_dx_m=12000), "Unknown"),
    (lambda row: row["tiers"]["lower"].update(run_hours=24), "must not decrease"),
    (lambda row: row["tiers"]["lower"].update(root_dx_km=0.1), "coarser to finer"),
    (lambda row: row["tiers"]["lower"]["bounds_degrees"].update(west=-110), "footprint"),
    (lambda row: row["source_options"][0].update(cycle_utc="2026-09-01T00:30:00Z"), "exact UTC hour"),
])
def test_invalid_units_ranges_and_tier_order_never_reach_creation(document, mutation, match):
    mutation(document["cases"][0])
    with pytest.raises(catalog.CatalogError, match=match):
        catalog.validate_catalog(document)


def test_antimeridian_tiers_use_eastward_containment(document):
    row = document["cases"][0]
    for tier, west, east in (("lower", 178, -178), ("recommended", 170, -170), ("upper", 160, -160)):
        row["tiers"][tier]["bounds_degrees"].update(west=west, east=east)
    assert catalog.validate_catalog(document)["cases"][0]["tiers"]["lower"]["bounds_degrees"]["west"] == 178


@pytest.mark.parametrize("key", ["command", "environment", "output_dir", "api_token", "wif_climatology_path", "km_opt_zero_acknowledgement"])
def test_only_scientific_native_override_fields_are_admitted(key):
    with pytest.raises(catalog.CatalogError, match="not a recognized native scientific setting"):
        catalog.validate_native_overrides({"shared": {key: "do not execute"}})


def test_native_parameter_types_and_values_use_the_actual_registry():
    catalog.validate_native_overrides({"shared": {"num_soil_layers": 4}, "domains": [{"grid_id": 1, "settings": {"diff_6th_factor": 0.1}}]})
    for value in (True, "0.1", -1.0, 2.0):
        with pytest.raises(catalog.CatalogError):
            catalog.validate_native_overrides({"domains": [{"grid_id": 1, "settings": {"diff_6th_factor": value}}]})
    with pytest.raises(catalog.CatalogError, match="scope"):
        catalog.validate_native_overrides({"domains": [{"grid_id": 1, "settings": {"num_soil_layers": 4}}]})


def test_search_is_paged_and_filters_by_case_kind_and_source():
    result = catalog.list_cases(EXAMPLES / "example.json", query="synthetic", source="GFS", event_kind="synthetic", limit=1)
    assert result["total"] == 2
    assert len(result["cases"]) == 1
    assert catalog.list_cases(EXAMPLES / "example.json", query="typed")["total"] == 1
    assert catalog.list_cases(EXAMPLES / "example.json", source="era5")["total"] == 0


def test_unknown_future_source_remains_data_but_cannot_be_selected(tmp_path, document):
    document["cases"][0]["source_options"][0]["source"] = "future-model"
    path = write_catalog(tmp_path, document)
    assert catalog.list_cases(path, source="future-model")["total"] == 1
    with pytest.raises(catalog.CatalogError, match="Source option"):
        catalog.preview_case(path, document["cases"][0]["id"], now=NOW)


def test_preview_exposes_selected_settings_and_does_not_create_files(tmp_path, document):
    path = write_catalog(tmp_path, document)
    before = list(tmp_path.iterdir())
    result = catalog.preview_case(path, "synthetic-overrides-example", tier="lower", now=NOW,
                                  native_overrides={"domains": [{"grid_id": 1, "settings": {"epssm": 0.3}}]})
    assert result["cycle"] == "2026-09-01T00"
    assert result["native_overrides"]["domains"][0]["settings"]["epssm"] == 0.3
    assert result["forecast_started"] is False
    assert list(tmp_path.iterdir()) == before


def test_a_domain_override_cannot_be_silently_dropped_by_a_smaller_tier(tmp_path, document):
    path = write_catalog(tmp_path, document)
    with pytest.raises(catalog.CatalogError, match="creates only 1 domain"):
        catalog.preview_case(path, document["cases"][0]["id"], tier="lower", now=NOW,
                             native_overrides={"domains": [{"grid_id": 2, "settings": {"epssm": 0.3}}]})


def test_export_preserves_original_toml_and_refuses_overwrite(tmp_path):
    source = EXAMPLES / "example.toml"
    target = tmp_path / "original.toml"
    result = catalog.export_catalog(source, target, original=True)
    assert target.read_bytes() == source.read_bytes()
    assert result["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        catalog.export_catalog(source, target)
    assert target.read_bytes() == source.read_bytes()


def test_catalog_changed_since_preview_cannot_create_different_science(tmp_path, document):
    path = write_catalog(tmp_path, document)
    preview = catalog.preview_case(path, document["cases"][0]["id"], tier="lower", now=NOW)
    document["cases"][0]["tiers"]["lower"]["run_hours"] = 2
    write_catalog(tmp_path, document)
    with pytest.raises(catalog.CatalogError, match="changed after the preview"):
        catalog.create_case(path, document["cases"][0]["id"], out=tmp_path / "changed.toml", tier="lower", now=NOW,
                            expected_catalog_sha256=preview["provenance"]["original_sha256"])
    assert not (tmp_path / "changed.toml").exists()


def test_native_creation_applies_science_and_preserves_original_and_receipt(tmp_path):
    source = EXAMPLES / "example.json"
    out = tmp_path / "study.toml"
    result = catalog.create_case(source, "synthetic-overrides-example", out=out,
                                 tier="lower", vram_gib=32, now=NOW)
    raw = tomllib.loads(out.read_text(encoding="utf-8"))
    assert raw["shared"]["num_soil_layers"] == 4
    assert raw["domain"][0]["epssm"] == 0.4
    assert raw["domain"][0]["diff_6th_factor"] == 0.1
    assert Path(result["original_catalog"]).read_bytes() == source.read_bytes()
    receipt = json.loads(Path(str(out) + ".arwen-case.json").read_text())
    assert receipt["original_catalog_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["config_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert receipt["admission"]["forecast_started"] is False
    assert receipt["files"] and result["domains"]
    preserved = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        catalog.create_case(source, "synthetic-profile-example", out=out,
                            tier="lower", vram_gib=32, now=NOW)
    assert preserved == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


def test_invalid_native_combination_leaves_no_published_configuration(tmp_path):
    out = tmp_path / "invalid.toml"
    with pytest.raises(ValueError, match="sf_surface_physics"):
        catalog.create_case(EXAMPLES / "example.json", "synthetic-profile-example", out=out,
                            tier="lower", vram_gib=32, now=NOW,
                            native_overrides={"shared": {"sf_surface_physics": 999}})
    assert not out.exists()
    assert not list(tmp_path.glob("invalid*"))


def test_cli_dispatch_lists_cases_as_compact_json(capsys):
    from gpuwm.cli import main
    assert main(["case-catalog", "list", "--catalog", str(EXAMPLES / "example.json"), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema"] == "arwen.case-list.v1"
    assert output["total"] == 2


def proposal_document():
    """Synthetic data in the uploaded proposal's shape; no historical claims."""
    selectors = {"moist": True, "mp_physics": 8, "bl_pbl_physics": 1,
                 "sf_sfclay_physics": 1, "sf_surface_physics": 2,
                 "ra_lw_physics": 4, "ra_sw_physics": 4,
                 "ra_rrtmg_variant": "rrtmg_legacy", "cu_physics": 0,
                 "diff_opt": 2, "km_opt": 4}
    presets, schedules = {}, {}
    for tier, cycle, hours in (("minimum", "2013-05-31T14:00:00Z", 11),
                               ("preferred", "2013-05-31T12:00:00Z", 13),
                               ("large", "2013-05-31T00:00:00Z", 25)):
        domains = []
        for index, dx, width in ((1, 12, 600), (2, 3, 216), (3, 1, 72)):
            choice = dict(selectors, cu_physics=1 if index == 1 else 0)
            domains.append({"id": f"d{index:02d}", "parent_id": f"d{index-1:02d}" if index > 1 else None,
                            "center_lat": 35.5, "center_lon": -97.95, "dx_km": dx,
                            "width_km": width, "height_km": width, "selectors": choice})
        presets[tier] = {"domains": domains, "vertical_grid": {"levels_intent": 49}}
        schedules[tier] = {"start_utc": cycle, "total_integration_hours": hours,
                           "source_files_verified": False}
    return {"schema": "arwen.case-catalog/v1", "catalog_version": "1.0.0-research-proposal",
            "title": "Synthetic proposal import", "metadata": {"readiness": "planning only"},
            "initialization_sources": {"era5": {"engine_source_id": "era5", "time_cadence_hours": 1}},
            "physics_profiles": {"synthetic-control": {"soil_layers_required": 4}},
            "cases": [{"id": "synthetic-import", "name": "Synthetic import", "category": "synthetic",
                       "initialization": {"default_source_id": "era5", "alternatives": [
                           {"source_id": "era5", "schedules": schedules}]},
                       "physics": {"recommended_profile_id": "synthetic-control"}, "presets": presets}]}


def proposal_zip(tmp_path, *, extra_member=None, document=None):
    path = tmp_path / "proposal.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("proposal/catalog.json", json.dumps(document or proposal_document()))
        archive.writestr("proposal/authoring.py", "raise RuntimeError('never execute catalog scripts')")
        if extra_member:
            # ZipInfo normally normalizes Windows separators at construction;
            # retain the deliberately malformed on-wire name for this control.
            info = zipfile.ZipInfo("fixture")
            info.filename = extra_member
            info.orig_filename = extra_member
            archive.writestr(info, "untrusted")
    return path


def test_proposal_zip_preserves_original_and_maps_source_specific_tiers(tmp_path):
    path = proposal_zip(tmp_path)
    loaded = catalog.load_catalog(path)
    assert loaded.original == path.read_bytes()
    assert loaded.format == "zip"
    mapping = loaded.document["catalog"]["metadata"]["import_mapping"]
    assert mapping["archive"]["scripts_executed"] is False
    assert mapping["archive"]["files_extracted"] == 0
    assert not mapping["issues"]
    lower = catalog.preview_case(loaded, "synthetic-import", tier="lower", now=NOW)
    preferred = catalog.preview_case(loaded, "synthetic-import", tier="recommended", now=NOW)
    assert (lower["cycle"], lower["geometry"]["run_hours"]) == ("2013-05-31T14", 11)
    assert (preferred["cycle"], preferred["geometry"]["run_hours"]) == ("2013-05-31T12", 13)
    assert lower["source_option"]["cadence_hours"] == 1
    assert lower["native_overrides"]["shared"]["mp_physics"] == 8
    assert not list(tmp_path.glob("*.py"))


@pytest.mark.parametrize("member", ["../escape.json", "/absolute.json", "C:/outside.json", "folder\\escape.json"])
def test_zip_paths_cannot_escape_even_when_the_member_is_not_selected(tmp_path, member):
    with pytest.raises(catalog.CatalogError, match="Unsafe catalog ZIP member"):
        catalog.load_catalog(proposal_zip(tmp_path, extra_member=member))


def test_unsupported_imported_selector_is_visible_and_blocks_creation(tmp_path):
    document = proposal_document()
    document["cases"][0]["presets"]["minimum"]["domains"][0]["selectors"]["command"] = "do not run"
    path = proposal_zip(tmp_path, document=document)
    loaded = catalog.load_catalog(path)
    assert catalog.list_cases(loaded)["total"] == 1
    preview = catalog.preview_case(loaded, "synthetic-import", tier="lower", now=NOW)
    assert any("command" in issue for issue in preview["import_issues"])
    with pytest.raises(catalog.CatalogError, match="unsupported settings"):
        catalog.create_case(loaded, "synthetic-import", out=tmp_path / "blocked.toml", tier="lower", now=NOW)
    assert not (tmp_path / "blocked.toml").exists()


def test_imported_native_creation_honors_hourly_cadence_and_unique_vtables(tmp_path):
    path = proposal_zip(tmp_path)
    for name in ("one", "two"):
        out = tmp_path / (name + ".toml")
        receipt = catalog.create_case(path, "synthetic-import", out=out, tier="lower", vram_gib=32, now=NOW)
        raw = tomllib.loads(out.read_text(encoding="utf-8"))
        assert raw["fetch"]["cycle"] == "2013-05-31T14"
        assert raw["fetch"]["cadence"] == 1 and raw["fetch"]["hours"] == 11
        assert raw["case_data"]["forcing_interval_s"] == 3600
        assert raw["case_data"]["vtable"].startswith(name + ".")
        assert (tmp_path / raw["case_data"]["vtable"]).is_file()
        assert "interval_seconds = 3600" in out.with_suffix(".namelist.wps").read_text()
        assert Path(receipt["original_catalog"]).read_bytes() == path.read_bytes()
        assert receipt["admission"]["forcing_interval_seconds"] == 3600
        assert [d["dx_km"] for d in receipt["domains"]] == [12, 3, 1]


def worldwide_proposal_document():
    from copy import deepcopy
    document = proposal_document()
    document.update(schema="arwen.case-catalog/v2", catalog_version="2.0.0-worldwide-research-proposal")
    document["initialization_sources"]["era5"] = {"boundary_interval_hours": 6}
    document["initialization_sources"]["hrrr"] = {"boundary_interval_hours": 1}
    case = document["cases"][0]
    primary = case["initialization"].pop("alternatives")[0]
    primary["domain_recipe"] = "standard_12_3_1"
    regional = deepcopy(primary)
    regional.update(source_id="hrrr", domain_recipe="regional_3_1", status_by_tier={})
    for tier in ("minimum", "preferred", "large"):
        primary["schedules"][tier]["boundary_interval_hours"] = 6
        regional["schedules"][tier] = {"start_utc": "2026-09-01T18:00:00Z",
            "total_integration_hours": 6, "boundary_interval_hours": 1}
        regional["status_by_tier"][tier] = {"blocking_reasons": [], "actual_files_verified": False}
        domains = deepcopy(case["presets"][tier]["domains"][1:])
        for index, domain in enumerate(domains):
            domain.update(id=f"d{index+1:02d}", parent_id=None if index == 0 else f"d{index:02d}")
        case["presets"][tier]["source_domain_recipes"] = {"regional_3_1": {"domains": domains}}
    case["initialization"]["source_options"] = {"era5": primary, "hrrr": regional}
    return document


@pytest.mark.parametrize("source,spacing,cadence", [("era5", [12,3,1], 6), ("hrrr", [3,1], 1)])
def test_worldwide_source_geometry_and_cadence_reach_native_creation(tmp_path, source, spacing, cadence):
    path = proposal_zip(tmp_path, document=worldwide_proposal_document())
    loaded = catalog.load_catalog(path)
    preview = catalog.preview_case(loaded, "synthetic-import", tier="lower", source_option=source, now=NOW)
    assert [row["dx_km"] for row in preview["geometry"]["domain_intents"]] == spacing
    assert preview["source_option"]["cadence_hours"] == cadence
    assert preview["provenance"]["catalog"]["provenance"][0]["source_schema"] == "arwen.case-catalog/v2"
    assert preview["source_availability"]["hours"] == (12 if source == "era5" else 6)
    output = tmp_path / f"{source}.toml"
    receipt = catalog.create_case(loaded, "synthetic-import", out=output, tier="lower",
                                  source_option=source, vram_gib=32, now=NOW)
    raw = tomllib.loads(output.read_text(encoding="utf-8"))
    from gpuwm.experiment import load_experiment
    assert [row.run.dx / 1000 for row in load_experiment(output).domains] == spacing
    assert raw["fetch"].get("cadence", 1) == cadence
    assert raw["experiment"]["run_seconds"] == preview["geometry"]["run_hours"] * 3600
    assert receipt["admission"]["forecast_started"] is False
    assert Path(receipt["original_catalog"]).read_bytes() == path.read_bytes()


def test_worldwide_source_blocking_reasons_remain_visible_and_prevent_creation(tmp_path):
    document = worldwide_proposal_document()
    document["cases"][0]["initialization"]["source_options"]["hrrr"]["status_by_tier"]["minimum"]["blocking_reasons"] = ["outside_declared_source_coverage"]
    loaded = catalog.load_catalog(proposal_zip(tmp_path, document=document))
    preview = catalog.preview_case(loaded, "synthetic-import", tier="lower", source_option="hrrr", now=NOW)
    assert "outside_declared_source_coverage" in preview["import_issues"]
    with pytest.raises(catalog.CatalogError, match="outside_declared_source_coverage"):
        catalog.create_case(loaded, "synthetic-import", out=tmp_path / "blocked.toml", tier="lower", source_option="hrrr", vram_gib=32, now=NOW)
    assert not (tmp_path / "blocked.toml").exists()


def test_archive_budget_applies_to_selected_bytes_not_unread_duplicates(tmp_path, monkeypatch):
    from gpuwm.case_catalog_import import read_archive
    path = proposal_zip(tmp_path)
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("proposal/duplicate-browser.html", "x" * 100_000)
    original_open = zipfile.ZipFile.open
    opened = []
    def selected_only(self, member, *args, **kwargs):
        name = getattr(member, "filename", member)
        opened.append(name)
        assert name == "proposal/catalog.json"
        return original_open(self, member, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "open", selected_only)
    document, report = read_archive(path.read_bytes(), limit=20_000, object_pairs_hook=catalog._json_object)
    assert document["schema"] == "arwen.case-catalog/v1"
    assert report["expanded_bytes"] > 20_000 and report["selected_bytes_read"] < 20_000
    assert opened == ["proposal/catalog.json"]
    with pytest.raises(catalog.CatalogError, match="expanded catalog.json"):
        read_archive(path.read_bytes(), limit=100, object_pairs_hook=catalog._json_object)
