"""Declared static settings must reach preparation and its sealed footprints."""
from dataclasses import replace
from datetime import date
import hashlib
import json

import numpy as np
import pytest

from gpuwm.static import highres_production as owner


def _config(tmp_path, *, enabled=True):
    source = tmp_path / "input" / "case.toml"
    source.parent.mkdir(exist_ok=True)
    source.write_text(
        "[static.highres]\n" + f"enabled = {str(enabled).lower()}\n" +
        'cache_root = "not-created/cache"\nfields = "terrain"\n', encoding="utf-8")
    return source


def test_static_owner_uses_captured_bytes_and_original_path_base(tmp_path, monkeypatch):
    from gpuwm.config_authority import authority_environment
    source = _config(tmp_path)
    payload = source.read_bytes()
    capture = tmp_path / "capture.toml"
    capture.write_bytes(payload)
    for key, value in authority_environment(source=source, payload_path=capture,
            sha256=hashlib.sha256(payload).hexdigest()).items():
        monkeypatch.setenv(key, value)
    source.write_text("changed original", encoding="utf-8")
    config = owner.load_static_highres(source)
    assert config.enabled is True
    assert config.cache_root == source.parent / "not-created/cache"
    capture.write_bytes(payload + b"# altered")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        owner.load_static_highres(source)


def test_absent_or_disabled_prepared_overlay_preserves_objects(tmp_path, monkeypatch):
    baseline, receipt = object(), object()
    monkeypatch.setattr(owner, "apply_highres_statics", lambda *a, **k: pytest.fail("inert overlay ran"))
    for config in (None, owner.load_static_highres(_config(tmp_path, enabled=False))):
        fields, evidence = owner.apply_prepared_highres(baseline, None,
            config=config, domain_id=1, case_date=None,
            landuse_attrs=None, baseline_receipt=receipt)
        assert fields is baseline and evidence is receipt


def _grid():
    from test_static_highres_warp_routing import _grid_from_spec, _meta
    return _grid_from_spec(_meta()["terrain_warp"]["grid_spec"])


@pytest.mark.parametrize("change", ["settings", "date", "placement", "receipt"])
def test_prepared_overlay_binding_rejects_semantic_drift(tmp_path, change):
    config = owner.load_static_highres(_config(tmp_path))
    grid, day = _grid(), date(2026, 9, 5)
    receipt = {"highres": {"status": "APPLIED", "config": config.echo(),
        "case_date": day.isoformat(), "grid": owner._grid_identity(grid, 1)}}
    owner.require_prepared_highres(receipt, grid, config=config, domain_id=1, case_date=day)
    if change == "settings":
        config = replace(config, terrain_source="copernicus-dem-glo30")
    elif change == "date":
        day = date(2026, 9, 6)
    elif change == "placement":
        grid.known_x += 1
    else:
        receipt = {}
    with pytest.raises(ValueError, match="do not bind"):
        owner.require_prepared_highres(receipt, grid, config=config, domain_id=1, case_date=day)


def test_real_rust_overlay_reaches_prepared_fields_and_reuses_verified_result(tmp_path, monkeypatch):
    from test_static_highres_warp_routing import (
        HIGHRES_FIXTURES, _bound, _bridge_or_fail, no_python_geography_stack)
    from test_static_highres_international import _baseline
    from gpuwm.static.highres import build_terrain_override, merge_terrain_override
    _bridge_or_fail()
    config = owner.load_static_highres(_config(tmp_path))
    grid, day = _grid(), date(2026, 9, 5)
    source = _bound(HIGHRES_FIXTURES / "terrain_clip.tif", role="terrain")
    calls = []
    def retained_terrain(*args, **kwargs):
        calls.append(kwargs["grid"])
        return source, {"terrain_bytes_fetched": 0, "retained_sha256": source.sha256}
    monkeypatch.setattr(owner, "_fetch_terrain", retained_terrain)
    baseline = _baseline(grid.e_sn - 1, grid.e_we - 1)
    before = {key: value.tobytes() for key, value in baseline.items()}
    with no_python_geography_stack():
        fields, receipt = owner.apply_prepared_highres(baseline, grid,
            config=config, domain_id=1, case_date=day, landuse_attrs=None)
        # Independent lower-level numerical path, with the same bound raster.
        overrides, _ = build_terrain_override(grid, terrain=source, halo=owner.HALO)
        expected, _ = merge_terrain_override(baseline, overrides)
    assert set(fields) == set(expected)
    assert all(fields[key].tobytes() == expected[key].tobytes() for key in fields)
    assert fields["HGT_M"].tobytes() != before["HGT_M"]
    assert fields["TMN"].shape == fields["HGT_M"].shape
    assert all(baseline[key].tobytes() == value for key, value in before.items())
    assert receipt["highres"]["status"] == "APPLIED"
    assert receipt["highres"]["static_compute"] == "rust static-fields bridge"
    owner.require_prepared_highres(receipt, grid, config=config, domain_id=1, case_date=day)
    reused, evidence = owner.apply_prepared_highres(fields, grid,
        config=config, domain_id=1, case_date=day, landuse_attrs=None,
        baseline_receipt=receipt)
    assert reused is fields and evidence is receipt
    assert len(calls) == 1


@pytest.mark.parametrize("domain_count", [1, 2])
def test_mapped_public_preparation_consumes_highres_before_initialization(tmp_path, monkeypatch, domain_count):
    from test_mapped_direct import _install_prepare_fakes
    from gpuwm import mapped_direct
    args, calls, expected = _install_prepare_fakes(
        monkeypatch, tmp_path, domain_count=domain_count, backend="cpu")
    args["experiment_config"].write_text(
        '[static.highres]\nenabled = true\ncache_root = "hr-cache"\n', encoding="utf-8")
    seen = []
    def apply(fields, grid, **kwargs):
        assert calls["interpolate"] == 0
        assert kwargs["config"].cache_root == tmp_path / "hr-cache"
        assert fields is expected.static and grid is expected.grids[0]
        assert kwargs["case_date"] == expected.exp.start_time.date()
        seen.append(kwargs["config"])
        return fields, {"highres": {"status": "APPLIED", "test_operand": True}}
    monkeypatch.setattr(owner, "apply_prepared_highres", apply)
    identities = []
    previous_identity = mapped_direct.prepared_cache_identity
    def capture_identity(**kwargs):
        identities.append(kwargs["source_identity"])
        return previous_identity(**kwargs)
    monkeypatch.setattr(mapped_direct, "prepared_cache_identity", capture_identity)
    proof = mapped_direct.prepare_mapped_wrf(**args)
    assert len(seen) == 1
    if domain_count == 2:
        assert calls["hierarchy"][0]["static_highres"] is seen[0]
    assert proof["execution_inputs"]["root_static_receipt"]["highres"]["test_operand"] is True
    if domain_count == 1:
        assert len(identities) == 1
        assert identities[0]["static_highres"] == owner.static_highres_identity(seen[0])


def test_shared_hierarchy_retains_static_config_on_child_catalog(tmp_path, monkeypatch):
    from test_source_hierarchy import _call
    config = owner.load_static_highres(_config(tmp_path))
    _, observed = _call(tmp_path, monkeypatch, static_highres=config)
    assert observed["initialize"]["catalog"].static_highres is config


def test_case_path_publication_resolves_only_owned_keys_without_expanding_globs(tmp_path):
    from gpuwm.case_data import resolved_case_data_paths
    raw = {"forcing": ["forcing/*.grib"], "vtable": "Vtable", "geog_root": "future-geog",
           "water_temperature_overlay": "water.nc", "source_orography": {"d02": "terrain.nc"},
           "source_orography_variable": "height", "output_title": "unchanged title"}
    # A coincidentally existing file must not make a variable name into a path.
    (tmp_path / "height").write_text("unrelated")
    resolved = resolved_case_data_paths(raw, source="case.toml", base_dir=tmp_path)
    assert resolved["forcing"] == [str(tmp_path / "forcing/*.grib")]
    assert resolved["geog_root"] == str(tmp_path / "future-geog")
    assert resolved["source_orography"]["d02"] == str(tmp_path / "terrain.nc")
    assert resolved["source_orography_variable"] == "height"
    assert resolved["output_title"] == "unchanged title"
    assert raw["forcing"] == ["forcing/*.grib"]


def test_native_static_tool_applies_declared_overlay_and_reseals_prebuilt_bytes(tmp_path, monkeypatch):
    import sys
    from test_hrrr_native_static import _fixture
    from tools import hrrr_build_native_static as producer
    from gpuwm.hrrr_native_static import verify_hrrr_native_static
    target, cache, receipt_path = _fixture(tmp_path)
    original_cache, original_receipt = cache.read_bytes(), receipt_path.read_bytes()
    receipt = json.loads(original_receipt)
    receipt["geog_root"] = str(tmp_path)
    receipt["geog_selection"] = {key: value["dataset"]
                                 for key, value in receipt["geog_source_coverage"].items()}
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    original_receipt = receipt_path.read_bytes()
    domain_path = tmp_path / "domain.json"
    domain_path.write_text(json.dumps(target.to_payload()), encoding="utf-8")
    config_path = _config(tmp_path)
    config = owner.load_static_highres(config_path)
    out, evidence = tmp_path / "out.npz", tmp_path / "out.json"
    monkeypatch.setattr(producer.GeogSelection, "landuse_global_attrs",
                        lambda self: {"ISWATER": 17, "ISLAKE": 21})
    calls = []
    def overlay(fields, grid, **kwargs):
        calls.append(kwargs)
        assert kwargs["config"] == config
        assert kwargs["case_date"] == date(2026, 9, 5)
        changed = dict(fields)
        changed["HGT_M"] = fields["HGT_M"] + 19.
        detail = {"status": "APPLIED", "config": config.echo(),
                  "case_date": "2026-09-05", "grid": owner._grid_identity(grid, 1)}
        return changed, {"highres": detail, "baseline": kwargs["baseline_receipt"]}
    monkeypatch.setattr(owner, "apply_prepared_highres", overlay)
    monkeypatch.setattr(sys, "argv", ["native-static", "--static-cache", str(cache),
        "--static-receipt", str(receipt_path), "--domain-spec", str(domain_path),
        "--experiment-config", str(config_path), "--case-date", "2026-09-05",
        "--output", str(out), "--receipt", str(evidence)])
    producer.main()
    fields, sealed = verify_hrrr_native_static(out, evidence, target)
    assert np.all(fields["HGT_M"] == 19.)
    assert sealed["highres"]["config"] == config.echo()
    assert sealed["cache"]["sha256"] != receipt["cache"]["sha256"]
    assert len(calls) == 1
    assert cache.read_bytes() == original_cache
    assert receipt_path.read_bytes() == original_receipt


def test_real_highres_corridor_crops_match_initial_and_moved_footprints(tmp_path, monkeypatch):
    from datetime import datetime
    from types import SimpleNamespace
    from test_statics_corridor import _synthetic_wps_geog, _catalog
    from test_static_highres_warp_routing import HIGHRES_FIXTURES, _bound, _bridge_or_fail
    from gpuwm.static.lambert import LambertGrid
    from gpuwm.static.build import build_static, geog_selection_from_catalog
    from gpuwm.static.corridor import (build_child_statics_corridor,
        write_statics_corridor_set, load_child_statics_corridor, corridor_footprint_statics_builder)
    _bridge_or_fail()
    import test_statics_corridor as corridor_fixture
    original_regional = corridor_fixture._fine_regional
    def swiss_region(kv_extra=None):
        kv, nx, ny = original_regional(kv_extra)
        kv.update(known_lat=45., known_lon=6.)
        return kv, nx, ny
    monkeypatch.setattr(corridor_fixture, "_fine_regional", swiss_region)
    geog = _synthetic_wps_geog(tmp_path / "geog")
    catalog = _catalog(geog, tmp_path)
    parent = LambertGrid(ref_lat=46.5, ref_lon=7.5, truelat1=46., truelat2=47.,
                         stand_lon=7.5, dx=750., dy=750., e_we=7, e_sn=7)
    child = SimpleNamespace(grid_id=2, parent_id=1, parent_grid_ratio=3,
        i_parent_start=3, j_parent_start=3, start_time=datetime(2026, 9, 5),
        run=SimpleNamespace(nx=3, ny=3))
    parent_run = SimpleNamespace(nx=6, ny=6)
    reference = parent.nest(3, 3, 3, 4, 4)
    config = owner.load_static_highres(_config(tmp_path))
    source = _bound(HIGHRES_FIXTURES / "terrain_clip.tif", role="terrain")
    monkeypatch.setattr(owner, "_fetch_terrain", lambda *a, **k:
        (source, {"terrain_bytes_fetched": 0}))
    built = build_child_statics_corridor(child_dc=child, parent_run=parent_run,
        reference_grid=reference, static_catalog=catalog, static_highres=config)
    directory = tmp_path / "corridor"
    receipt = write_statics_corridor_set(directory, [built])
    corridor = load_child_statics_corridor(directory, expected_set_receipt=receipt,
        grid_id=2, child_dc=child, parent_run=parent_run, reference_grid=reference)
    assert corridor_footprint_statics_builder(corridor).highres_applied is True
    selection = geog_selection_from_catalog(catalog, 2)
    for i, j in [(3, 3), (2, 4)]:
        grid = parent.nest(i, j, 3, 4, 4)
        baseline = build_static(grid, selection.root, selection=selection)
        exact, _ = owner.apply_highres_statics(baseline, grid, config=config,
            domain_id=2, case_date=child.start_time.date(),
            landuse_attrs=selection.landuse_global_attrs())
        cropped = corridor.crop(i, j)
        assert set(cropped) == set(exact)
        for key in exact:
            assert cropped[key].tobytes() == exact[key].tobytes(), (i, j, key)


def test_native_publication_retains_captured_relative_paths_when_relocated(tmp_path, monkeypatch):
    from test_hrrr_configured_physics import _case
    from gpuwm.hrrr_configuration import resolve_root_experiment
    from gpuwm.experiment_document import publish_experiment_document
    from gpuwm.config_authority import authority_environment
    exp, target, source, namelist, wps = _case(tmp_path)
    source.write_text(source.read_text() +
        '\n[static.highres]\nenabled = true\ncache_root = "new-cache"\n', encoding="utf-8")
    payload = source.read_bytes()
    capture = tmp_path / "captured.toml"
    capture.write_bytes(payload)
    for key, value in authority_environment(source=source, payload_path=capture,
            sha256=hashlib.sha256(payload).hexdigest()).items():
        monkeypatch.setenv(key, value)
    source.write_text("changed source", encoding="utf-8")
    actual, raw = resolve_root_experiment(target=target, vertical=exp.vertical,
        namelist_input=namelist, start_time=exp.start_time, run_seconds=exp.run_seconds,
        experiment_config=source, wps_namelist=wps)
    published = publish_experiment_document(tmp_path / "relocated" / "case.toml", raw, actual)
    import tomllib
    result = tomllib.loads(published.read_text(encoding="utf-8"))
    assert result["static"]["highres"]["cache_root"] == str(source.parent / "new-cache")
    assert capture.read_bytes() == payload
    assert source.read_text(encoding="utf-8") == "changed source"


@pytest.mark.parametrize("baseline", ["geog", "prebuilt"])
def test_native_wrapper_forwards_static_authority_before_decoding(tmp_path, monkeypatch, baseline):
    from test_prepare_hrrr_wrf import _wrapper_case, prepare
    argv, commands, output = _wrapper_case(tmp_path, monkeypatch)
    from pathlib import Path
    config = Path(argv[argv.index("--experiment-config") + 1])
    config.write_text(config.read_text() +
        '\n[static.highres]\nenabled = true\ncache_root = "hr-cache"\n', encoding="utf-8")
    if baseline == "geog":
        for flag in ("--static-cache", "--static-receipt"):
            index = argv.index(flag)
            del argv[index:index+2]
        root = tmp_path / "geog"
        root.mkdir()
        from gpuwm.ingest.hrrr_target import load_hrrr_target_domain
        domain_path = tmp_path / "domain.json"
        domain_path.write_text(json.dumps(load_hrrr_target_domain(None).to_payload()), encoding="utf-8")
        argv += ["--geog-root", str(root), "--domain-spec", str(domain_path)]
    class ReachedStatic(Exception):
        pass
    def observe(command, _env, **kwargs):
        assert any(str(value).endswith("hrrr_build_native_static.py") for value in command)
        assert command[command.index("--experiment-config") + 1] == str(config.resolve())
        assert command[command.index("--case-date") + 1] == "2026-07-18"
        assert ("--static-cache" in command) == (baseline == "prebuilt")
        assert ("--geog-root" in command) == (baseline == "geog")
        raise ReachedStatic
    monkeypatch.setattr(prepare, "_run", observe)
    with pytest.raises(ReachedStatic):
        prepare.main(argv)


def test_native_highres_declaration_is_bound_to_the_preparation_proof(tmp_path):
    from test_hrrr_prepared_background import _identity_pair, runner
    identity, proof = _identity_pair()
    value = owner.static_highres_identity(owner.load_static_highres(_config(tmp_path)))
    identity["static_highres"] = value
    proof["static_highres"] = dict(value)
    assert runner._validate_hrrr_source_identity(identity, proof) is identity
    proof["static_highres"]["fields"] = "all"
    with pytest.raises(ValueError, match="high-resolution settings differ"):
        runner._validate_hrrr_source_identity(identity, proof)
    identity["static_highres"] = proof["static_highres"] = {**value, "enabled": "true"}
    with pytest.raises(ValueError, match="must be a boolean"):
        runner._validate_hrrr_source_identity(identity, proof)


@pytest.mark.parametrize("change", ["unchanged", "settings", "disable", "newly-enabled"])
def test_sealed_extension_keeps_the_static_request_bound_before_suffix_work(tmp_path, monkeypatch, change):
    from datetime import datetime
    from types import SimpleNamespace
    from test_prepared_cache import _fixture, _extension_identity
    from gpuwm.ingest.prepared_cache import write_prepared_cache
    from gpuwm.physics_compat import WSM6_PROFILE_ID
    from tools import prepare_hrrr_wrf as prepare

    cycle = datetime(2026, 7, 18, 5)
    config_path = _config(tmp_path)
    config = owner.load_static_highres(config_path)
    prior = tmp_path / "prior"
    native = prior / "native"
    native.mkdir(parents=True)
    for relative in ("native/preparation-report/report.json", "native/source-manifest.snapshot",
                     "native-static.npz", "native-static-receipt.json", "native-geometry-receipt.json"):
        path = prior / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sealed fixture", encoding="utf-8")
    (native / "native-bridge").mkdir()
    (prior / "public-wrapper-result.json").write_text(json.dumps({
        "status": "PASS", "prepared_cache_contract": {"mode": "sealed-prefix-v1"},
        "source_cycle": cycle.isoformat(), "source_forecast_hours": [0, 1],
        "physics": {"profile": WSM6_PROFILE_ID}, "history_interval_seconds": 3600.,
    }), encoding="utf-8")
    identity = _extension_identity(source_hours=[0, 1], model_start=cycle,
        domain_start=cycle, bridge="a" * 64, source_manifest="b" * 64)
    if change != "newly-enabled":
        identity["source_identity"]["static_highres"] = owner.static_highres_identity(config)
    initial, met, boundaries = _fixture()
    # Real writer/reader and payload hashes; the unrelated process record is
    # supplied so this witness reaches the static request comparison.
    initial.hydrometeor_initialization = {
        "schema": "gpuwm-real-hydrometeor-correspondence-v2",
        "vertical_disposition": {
            "schema": "gpuwm-wrf-real-hydrometeor-vertical-disposition-v1"}}
    write_prepared_cache(native / "prepared-cache", identity=identity,
        initial_result=initial, met=met, boundaries=boundaries,
        sealed_forcing_extension=True)
    if change == "settings":
        config_path.write_text(config_path.read_text().replace('fields = "terrain"', 'fields = "all"'))
    elif change == "disable":
        config_path.write_text(config_path.read_text().replace("enabled = true", "enabled = false"))
    source_manifest = tmp_path / "source-manifest"
    source_manifest.write_text("extended source fixture", encoding="utf-8")
    monkeypatch.setattr(prepare, "_source_manifest_extension", lambda **kwargs: {})
    monkeypatch.setattr(prepare, "_run", lambda *a, **k: pytest.fail("suffix work started"))
    args = SimpleNamespace(extend_root_preparation=prior, forecast_start_hour=0,
        run_seconds=7200., physics_profile=WSM6_PROFILE_ID, history_interval_seconds=3600.,
        source_manifest=source_manifest, source_manifest_sha256=prepare._sha256(source_manifest),
        source_root=tmp_path, experiment_config=config_path)
    # The unchanged control must pass static binding and reach the separate
    # intentionally different namelist check; changed statics must fail first.
    expected = "native namelist changes immutable" if change == "unchanged" else "changes high-resolution statics"
    with pytest.raises(ValueError, match=expected):
        prepare._sealed_extension(args, valid_time=cycle, source_forecast_hours=(0, 1, 2),
            output=tmp_path / "output", env={}, decoder=tmp_path / "decoder", started=0.,
            namelist_invariant={"sha256": "different"})
    assert not (tmp_path / "output").exists()
