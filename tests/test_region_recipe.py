"""Complete recipe geometry and aerosol selection are metadata work."""
import json
from pathlib import Path
import tomllib

import pytest

from gpuwm.experiment import load_experiment


def recipe_table():
    from gpuwm.region_recipe import RECIPE_TABLE
    return RECIPE_TABLE


def emit(tmp_path, **overrides):
    from gpuwm.region_recipe import emit_recipe
    table = tmp_path / "regions.json"
    table.write_text(json.dumps({"regions": [{"id": "invented-row", "polygon": {
        "type": "Polygon", "coordinates": [[[121, 10], [122, 10], [122, 11], [121, 11], [121, 10]]]
    }}]}))
    options = dict(recipe="hrrr_v4_gsd41", region="invented-row", spacing_km=3,
                   source="gfs", cycle="2025-11-03T00", hours=6, region_table=table)
    options.update(overrides)
    return emit_recipe(**options)


def test_new_region_row_emits_complete_valid_recipe(tmp_path):
    text, receipt = emit(tmp_path)
    emitted = tomllib.loads(text)
    original = tomllib.loads((recipe_table().parent / "hrrr_v4_gsd41.toml").read_text())
    changed = {"map_proj", "mp28_aerosol_source", "use_rap_aero_icbc", "aer_init_opt", "wif_input_opt"}
    for key, value in original["shared"].items():
        if key not in changed:
            assert emitted["shared"][key] == value, key
    for key, value in original["domain"][0].items():
        if key not in {"nx", "ny", "dx"}:
            assert emitted["domain"][0][key] == value, key
    path = tmp_path / "config.toml"
    path.write_text(text)
    config = load_experiment(path)
    assert config.domains[0].run.dx == 3000
    assert receipt["region"] == "invented-row"
    assert emitted["shared"]["mp28_aerosol_source"] == "climatology"
    assert emitted["fetch"]["cadence"] == original["fetch"]["cadence"]
    assert receipt["aerosol_candidate"]["id"] == "monthly-climatology"
    # The recipe's native-grid statics belong to its own grid, not this one.
    assert "static" not in emitted
    assert receipt["native_static_set_aside"] == original["static"]


def test_packaged_recipe_matches_authored_source():
    root = Path(__file__).resolve().parents[1]
    assert (recipe_table().parent / "hrrr_v4_gsd41.toml").read_bytes() == (root / "configs/recipes/hrrr_v4_gsd41.toml").read_bytes()


def test_new_model_row_needs_no_code(tmp_path):
    registry = json.loads(recipe_table().read_text())
    row = registry["recipes"][0]
    row["id"] = "invented-model"
    row["config"] = str(recipe_table().parent / row["config"])
    table = tmp_path / "recipes.json"
    table.write_text(json.dumps(registry))
    text, receipt = emit(tmp_path, recipe="invented-model", recipe_table=table)
    config = tmp_path / "new-model.toml"
    config.write_text(text)
    assert load_experiment(config).domains[0].run.dx == 3000
    assert receipt["recipe"] == "invented-model"


def test_aerosol_preference_is_ordered_data():
    from gpuwm.region_recipe import select_aerosol
    policy = json.loads(recipe_table().read_text())["recipes"][0]["aerosol_sources"]
    assert select_aerosol(policy, "rap-native")[0]["settings"]["use_rap_aero_icbc"]
    assert select_aerosol(policy, "gfs")[0]["settings"]["mp28_aerosol_source"] == "climatology"
    assert select_aerosol(list(reversed(policy)), "rap-native")[0]["id"] == "monthly-climatology"
    with pytest.raises(ValueError, match="aerosol_source_unavailable.*QNWFA.*QNIFA"):
        select_aerosol(policy[:1], "gfs")


def test_climatology_receipt_declares_difference_and_analysis_unchanged():
    from gpuwm.aerosol_source_receipt import aerosol_source_report_entry, AEROSOL_SOURCE_KEY
    monthly = aerosol_source_report_entry({"aerosol_source": "wif-climatology"}, mp_physics=28, when_unrecorded="missing")
    assert "replaces analysed" in monthly[AEROSOL_SOURCE_KEY]["declared_forcing_difference"]
    analysed = aerosol_source_report_entry({"aerosol_source": "metgrid-analyzed"}, mp_physics=28, when_unrecorded="missing")
    assert "declared_forcing_difference" not in analysed[AEROSOL_SOURCE_KEY]


def test_ordinary_cli_recipe_with_new_row(tmp_path):
    from gpuwm.cli import main
    emit(tmp_path)
    out = tmp_path / "door.toml"
    assert main(["domain", "--recipe", "hrrr_v4_gsd41", "--region", "invented-row",
        "--region-table", str(tmp_path / "regions.json"), "--spacing-km", "3",
        "--source", "gfs", "--cycle", "2025-11-03T00", "--out", str(out)]) == 0
    assert load_experiment(out).domains[0].run.dx == 3000
    assert out.with_suffix(".recipe.json").is_file()


def test_recipe_cli_implicit_source_comes_from_metadata(tmp_path):
    from gpuwm.cli import main
    emit(tmp_path)
    out = tmp_path / "implicit-source.toml"
    assert main(["domain", "--recipe", "hrrr_v4_gsd41", "--region", "invented-row",
        "--region-table", str(tmp_path / "regions.json"), "--spacing-km", "3",
        "--cycle", "2025-11-03T00", "--out", str(out)]) == 0
    assert tomllib.loads(out.read_text())["fetch"]["source"] == "gfs"


def test_missing_declared_aerosol_refuses_before_emission(tmp_path):
    registry = json.loads(recipe_table().read_text())
    row = registry["recipes"][0]
    row["config"] = str(recipe_table().parent / row["config"])
    row["aerosol_sources"] = []
    table = tmp_path / "recipes.json"
    table.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="aerosol_source_unavailable.*zero aerosol"):
        emit(tmp_path, recipe_table=table)


@pytest.mark.parametrize("spacing", [float("nan"), float("inf"), -1, 0])
def test_invalid_spacing_names_stencil_breakage(tmp_path, spacing):
    with pytest.raises(ValueError, match="domain stencil spacing limits"):
        emit(tmp_path, spacing_km=spacing)


def test_complete_recipe_geography_has_pinned_container(tmp_path):
    from gpuwm.geog_assets import geography_for_run
    text, _ = emit(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text(text)
    assert "orogwd3_10m" in geography_for_run(path)["datasets"]


def test_archive_children_are_generic_metadata(monkeypatch):
    from gpuwm import geog_assets as assets
    monkeypatch.setattr(assets, "GEOG_ARCHIVES", (
        assets.GeogArchive("invented-container", "unused", 1, "0" * 64, 1,
                           index_subdirs=("invented-child",)),))
    assert assets.archive_names_for_paths({"invented-container/invented-child"}) == {"invented-container"}


def test_go_geography_checks_declared_container_children(tmp_path, monkeypatch):
    from gpuwm import geog_assets as assets
    from gpuwm.go_cli import geography_refusal
    row = assets.GeogArchive("invented-container", "unused", 1, "0" * 64, 1,
                              index_subdirs=("invented-child",))
    monkeypatch.setattr(assets, "GEOG_ARCHIVES", (row,))
    monkeypatch.setattr(assets, "geography_for_run", lambda path: {"datasets": [row.dataset]})
    root = tmp_path / "geog"
    assert "invented-container/invented-child/index" in geography_refusal(root, config="unused.toml")
    child = root / row.dataset / "invented-child"
    child.mkdir(parents=True)
    (child / "index").write_text("dataset fixture")
    assert geography_refusal(root, config="unused.toml") is None


def test_static_catalog_preserves_selected_root_for_supplemental_fields(tmp_path):
    from types import SimpleNamespace
    from gpuwm.static.build import geog_selection_from_catalog
    wps = tmp_path / "namelist.wps"
    wps.write_text("&geogrid\n geog_data_res='default',\n/\n")
    donor = tmp_path / "shared-baseline" / "topo" / "index"
    donor.parent.mkdir(parents=True)
    donor.write_text("index fixture")
    selected = tmp_path / "selected-root"
    selected.mkdir()
    catalog = SimpleNamespace(geog_root=selected, files=(
        SimpleNamespace(role="wps_namelist", path=wps),
        SimpleNamespace(role="geog_index", path=donor)))
    assert geog_selection_from_catalog(catalog, 1).root == selected


def test_whole_object_route_emits_preparation_crop_only(tmp_path):
    from gpuwm.region_recipe import emit_recipe
    region = tmp_path / "polygon.json"
    region.write_text(json.dumps({"type": "Polygon", "coordinates": [[
        [-100.5, 38.5], [-99.5, 38.5], [-99.5, 39.5], [-100.5, 39.5], [-100.5, 38.5]]]}))
    text, receipt = emit_recipe(recipe="hrrr_v4_gsd41", region=str(region),
        spacing_km=3, source="hrrr-native", cycle="2022-06-13T18", hours=1)
    assert "area" not in tomllib.loads(text)["fetch"]
    assert receipt["aerosol_candidate"]["settings"]["use_rap_aero_icbc"]


def test_analysed_aerosol_recipe_survives_the_namelist_route(tmp_path):
    # The nested HRRR source prepares from namelists written from the TOML.
    # Its analysed-aerosol row once stated mp28_aerosol_source = "analysis"
    # beside use_rap_aero_icbc, which the namelist pair reads back as 'auto',
    # so `gpuwm go --dry-run` refused the configuration `gpuwm domain
    # --recipe` had just written (fresh-install QA, 2026-10-12).  The row now
    # carries the shipped recipe's own spelling: use_rap_aero_icbc decides
    # the analysed source (gpuwm/ingest/real.py _resolved_mp28_aerosol_source).
    from gpuwm.hrrr_route_inputs import run_route_inputs
    from gpuwm.ingest.real import _resolved_mp28_aerosol_source
    from gpuwm.region_recipe import emit_recipe
    region = tmp_path / "polygon.json"
    region.write_text(json.dumps({"type": "Polygon", "coordinates": [[
        [-100.5, 38.5], [-99.5, 38.5], [-99.5, 39.5], [-100.5, 39.5], [-100.5, 38.5]]]}))
    text, receipt = emit_recipe(recipe="hrrr_v4_gsd41", region=str(region),
        spacing_km=3, source="hrrr", cycle="2026-10-11T12", hours=1)
    assert receipt["aerosol_candidate"]["id"] == "analysed"
    path = tmp_path / "recipe.toml"
    path.write_text(text)
    exp = load_experiment(path)
    assert _resolved_mp28_aerosol_source(exp.domains[0].run) == "analysis"
    assert run_route_inputs(path, exp, raw=tomllib.loads(text)) is None


def test_suite_region_reference_is_data_only(tmp_path):
    from gpuwm.region_recipe import emit_recipe
    table = tmp_path / "regions.v1.json"
    table.write_text(json.dumps({"regions": {"new-country": {
        "polygon": "new-envelope", "sources": {"global_ic": ["gfs"]}}}}))
    (tmp_path / "references.v1.json").write_text(json.dumps({"registries": {
        "polygons": {"new-envelope": {"bounds": [10, 5, 11, 6]}}}}))
    text, receipt = emit_recipe(recipe="hrrr_v4_gsd41", region="new-country",
        region_table=table, spacing_km=3, source=None, cycle="2025-11-03T00", hours=6)
    path = tmp_path / "recipe.toml"
    path.write_text(text)
    assert load_experiment(path).domains[0].run.dx == 3000
    assert receipt["forcing_source"] == "gfs"


def test_readiness_uses_the_same_bound_clock_defaults_as_forecast(tmp_path, monkeypatch):
    from gpuwm import stage_cli, prepared_single_domain_forecast as runner
    config = tmp_path / "clock.toml"
    config.write_text("[experiment]\nrun_seconds=21600\n[[domain]]\ngrid_id=1\nhistory_interval_s=3600\n")
    command = ["python", "-m", "gpuwm.prepared_single_domain_forecast",
        "--source", "gfs", "--prepared-root", str(tmp_path),
        "--proof-sha256", "a" * 64, "--source-manifest-sha256", "b" * 64,
        "--prepared-content-sha256", "c" * 64,
        "--experiment-config", str(config), "--wps-namelist", str(tmp_path / "wps"),
        "--io-mode", "history", "--outdir", str(tmp_path / "out")]
    monkeypatch.setattr(stage_cli, "sim_command", lambda *a, **kw: command)
    def preflight(**kwargs):
        assert kwargs["run_seconds"] == 21600
        assert kwargs["history_interval_seconds"] == 3600
        return "validated"
    monkeypatch.setattr(runner, "preflight_prepared_forecast", preflight)
    assert stage_cli.preflight_bundle({"layout":"single", "document":tmp_path / "proof.json"},
        experiment_config=config) == "validated"


def test_catalog_builder_requests_configured_drag_statics(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from gpuwm.static import build
    from gpuwm.static.lambert import LambertGrid
    from gpuwm.static.orographic import required_static_fields
    grid = LambertGrid(ref_lat=35., ref_lon=-90., truelat1=30., truelat2=60.,
        stand_lon=-90., dx=3000., dy=3000., e_we=5, e_sn=5)
    selection = build.GeogSelection.fallback(tmp_path)
    monkeypatch.setattr(build, 'geog_selection_from_catalog', lambda *a: selection)
    monkeypatch.setattr(build, '_CATALOG_STATIC_CACHE', {})
    required = required_static_fields(0, 3)
    def sample(grid, root, *, selection, timing_report):
        assert set(selection.orographic) == set(required)
        return {name: np.zeros((4, 4), dtype=np.float32) for name in required}
    monkeypatch.setattr(build, 'build_static', sample)
    fields = build.build_static_for_domain(grid, object(), 1,
        cfg=SimpleNamespace(topo_wind=0, gwd_opt=3, sf_lake_physics=0))
    assert set(fields) == set(required)


def test_source_adapter_passes_drag_config_without_lake(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from gpuwm import era5_direct as adapter
    cfg = SimpleNamespace(topo_wind=0, gwd_opt=3, sf_lake_physics=0, nx=4, ny=4)
    monkeypatch.setattr(adapter, 'verified_static_catalog', lambda *a, **kw: (object(), {}))
    def build(*args, **kwargs):
        assert kwargs.get('cfg') is cfg
        return {}
    monkeypatch.setattr(adapter, 'build_static_for_domain', build)
    monkeypatch.setattr(adapter, 'geog_selection_from_catalog', lambda *a:
        SimpleNamespace(landuse_global_attrs=lambda: {}))
    monkeypatch.setattr(adapter, '_validated_static', lambda fields, *a, **kw: fields)
    adapter._static_from_geog(tmp_path / 'wps', tmp_path, object(), cfg)


def test_incomplete_prepared_drag_is_named_before_gpu():
    from types import SimpleNamespace
    from gpuwm.static.orographic import require_prepared_terrain_drag, required_static_fields
    cfg = SimpleNamespace(topo_wind=0, gwd_opt=3)
    with pytest.raises(ValueError, match='prepared_terrain_drag_missing:.*VARLS'):
        require_prepared_terrain_drag({}, cfg)
    require_prepared_terrain_drag(dict.fromkeys(required_static_fields(0, 3)), cfg)
    require_prepared_terrain_drag({}, SimpleNamespace(topo_wind=0, gwd_opt=0))
