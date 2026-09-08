"""Small-input staging and reviewed dispatch; no SSH, GPU, fetch, or forecast."""
from argparse import ArgumentParser
import base64
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from gpuwm import remote_cli as rc, remote_plan as rp, remote_worker as rw


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def saved(tmp_path):
    from gpuwm import domain_wizard as dw
    config = tmp_path / "saved map.toml"
    text = dw.render_config(name="remote-map", start_time=datetime(2026, 9, 7, 18),
        hours=24, projection=dw._projection_entries(18, -162.1, "auto"),
        dims=dw._dims_for_scale(1, ()), ratios=(),
        fetch_hints=dict(source="gfs", cycle="2026-09-07T12", forecast_start_hour=6,
                         hours=24, out=str(tmp_path / "forcing"), cadence=3), case_data=None)
    config.write_text(text, encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "gpuwm.run-plan.v1", "name": "reviewed-map",
        "route": "prepared", "config": {"path": str(config)}, "output_root": str(tmp_path / "local-output"),
        "run_options": {"render_products": "none"}}), encoding="utf-8")
    return config, plan


def bundle(saved):
    config, plan = saved
    return rp.build_bundle(plan, workspace="/node/work", outdir="/node/work/new-output",
        geog_root="/node/geography", expected_plan_sha256=sha(plan.read_bytes()),
        expected_config_sha256=sha(config.read_bytes()))


def contents(document, name):
    return base64.b64decode(next(f["data"] for f in document["files"] if f["name"] == name))


def rehash(document):
    document["sha256"] = rp._sha(rp._encoded({k: v for k, v in document.items() if k != "sha256"}))


def local_worker_bundle(saved, tmp_path):
    document = bundle(saved)
    document["workspace"] = str(tmp_path.resolve())
    rehash(document)
    return document


def test_stages_only_selected_companions_and_keeps_plan_options(saved):
    config, plan = saved
    unrelated = config.parent / "unrelated-private.txt"
    unrelated.write_text("must not travel")
    config.with_suffix(".d01-target.json").write_text('{"nx":32}')
    original = tomllib.loads(config.read_text())
    document = bundle(saved)
    assert {f["name"] for f in document["files"]} == {"case.toml", "plan.json", "case.d01-target.json"}
    actual = tomllib.loads(contents(document, "case.toml").decode())
    assert actual["domain"] == original["domain"]
    assert actual["experiment"] == original["experiment"]
    assert actual["fetch"]["forecast_start_hour"] == 6
    remote = json.loads(contents(document, "plan.json"))
    assert remote["config"] == {"path": "case.toml"}
    assert remote["run_options"]["render_products"] == "none"
    assert remote["run_options"]["geog_root"] == "/node/geography"
    assert remote["output_root"] == "/node/work/new-output"
    assert document["source"]["plan_sha256"] == sha(plan.read_bytes())
    assert str(unrelated) not in document["source_inputs"]


def test_repeat_plans_share_the_native_canonical_acquisition_key_but_not_outputs(saved):
    from gpuwm.go_cli import managed_download_key
    from gpuwm.toml_document import emit_experiment_toml
    config, _plan = saved
    raw = tomllib.loads(config.read_text())
    first, second = bundle(saved), bundle(saved)
    assert first["id"] != second["id"]
    assert first["data_cache_key"] == second["data_cache_key"] == managed_download_key(raw["fetch"])
    first_config = tomllib.loads(contents(first, "case.toml").decode())
    second_config = tomllib.loads(contents(second, "case.toml").decode())
    assert first_config["fetch"]["out"] == second_config["fetch"]["out"]
    assert json.loads(contents(first, "plan.json"))["run_options"]["data_dir"] == first_config["fetch"]["out"]
    raw["fetch"]["hours"] = 27
    config.write_text(emit_experiment_toml(raw))
    assert bundle(saved)["data_cache_key"] != first["data_cache_key"]


@pytest.mark.parametrize("source", ["gfs", "hrrr", "rap"])
def test_ordinary_generic_prepared_sources_keep_native_acquisition_owner(saved, source):
    from gpuwm.go_cli import managed_download_key
    from gpuwm.toml_document import emit_experiment_toml
    config, _plan = saved
    raw = tomllib.loads(config.read_text())
    raw["experiment"]["run_seconds"] = 6 * 3600
    raw["fetch"].update(source=source, cycle="2026-09-07T18", forecast_start_hour=0, hours=6)
    if source != "gfs":
        raw["fetch"].pop("cadence", None)
    config.write_text(emit_experiment_toml(raw))
    document = bundle(saved)
    staged = tomllib.loads(contents(document, "case.toml").decode())
    plan = json.loads(contents(document, "plan.json"))
    assert staged["fetch"]["source"] == source and plan["route"] == "prepared"
    assert staged["fetch"]["hours"] == 6 and staged["fetch"]["cycle"] == "2026-09-07T18"
    assert plan["run_options"]["data_dir"] == staged["fetch"]["out"]
    assert document["data_cache_key"] == managed_download_key(raw["fetch"])
    assert "case_data" not in staged and document["blobs"] == []


@pytest.mark.parametrize("extended_path", [False, True])
def test_real_generated_case_data_shape_preserves_remote_acquisition(saved, extended_path):
    from gpuwm.toml_document import emit_experiment_toml
    config, _plan = saved
    raw = tomllib.loads(config.read_text())
    original_forcing = config.parent / "forcing" / "era5-combined.grib"
    if extended_path:
        if __import__("os").name != "nt":
            pytest.skip("Windows extended filesystem paths")
        original_forcing = Path("\\\\?\\" + str(original_forcing))
    wps = config.with_suffix(".namelist.wps")
    wps.write_text("&share\n max_dom=1,\n interval_seconds=10800,\n/\n&geogrid\n dx=12000.,\n dy=12000.,\n/\n")
    vtable = config.parent / "selected.Vtable"
    vtable.write_text("selected scientific Vtable")
    raw["fetch"].update(source="era5", era5_provider="cds", era5_product="ensemble_members",
                        member=7, retrieve=True)
    raw["fetch"].pop("forecast_start_hour")
    raw["fetch"]["cycle"] = "2026-09-07T18"
    raw["case_data"] = {"forcing": [str(original_forcing)], "vtable": str(vtable),
        "wps_namelist": str(wps), "geog_root": str(config.parent / "local-geog"),
        "forcing_interval_s": 10800, "sfcp_to_sfcp": True, "output_domain": 1,
        "output_title": "saved map"}
    config.write_text(emit_experiment_toml(raw))
    document = bundle(saved)
    actual = tomllib.loads(contents(document, "case.toml").decode())
    assert actual["case_data"]["forcing"] == [actual["fetch"]["out"] + "/era5-combined.grib"]
    assert actual["case_data"]["forcing_interval_s"] == 10800
    assert actual["case_data"]["vtable"] == "inputs/Vtable"
    assert actual["case_data"]["wps_namelist"] == "case.namelist.wps"
    assert actual["case_data"]["geog_root"] == "/node/geography"
    assert actual["fetch"]["member"] == 7 and actual["fetch"]["era5_product"] == "ensemble_members"
    assert actual["experiment"] == raw["experiment"] and actual["domain"] == raw["domain"]
    assert document["expected_downloads"][0]["path"] == actual["case_data"]["forcing"][0]
    assert "data_dir" not in json.loads(contents(document, "plan.json"))["run_options"]
    assert not original_forcing.exists()


@pytest.mark.parametrize("value,expected", [
    (r"\\?\C:\cache\era5-combined.grib", False),
    (r"\\?\UNC\server\share\era5-combined.grib", False),
    (r"\\?\C:\cache\*.grib", True),
    (r"\\?\C:\cache\era5-?.grib", True),
    (r"C:\cache\[ab].grib", True), ("/cache/era5-combined.grib", False)])
def test_native_forcing_globs_exclude_windows_namespace_prefix(value, expected):
    from gpuwm.case_data import forcing_has_glob
    assert forcing_has_glob(value) is expected


def test_native_future_extended_forcing_remains_a_literal_acquisition_output(tmp_path):
    from gpuwm.case_data import _resolve_forcing
    if __import__("os").name != "nt":
        pytest.skip("Windows extended filesystem paths")
    missing = "\\\\?\\" + str(tmp_path / "future-cache" / "era5-combined.grib")
    assert _resolve_forcing(tmp_path, [missing], "saved map.toml") == (Path(missing),)
    from gpuwm.runplan import declared_forcing_fetch
    acquisition = declared_forcing_fetch({"fetch": {"source":"era5", "cycle":"2026-09-07T18",
        "hours":6, "cadence":1, "era5_provider":"cds", "out":str(tmp_path / "future-cache")}},
        SimpleNamespace(forcing=(Path(missing),)))
    assert "--retrieve" in acquisition
    assert acquisition[acquisition.index("--out")+1] == str((tmp_path / "future-cache").resolve())


def test_changed_local_plan_is_refused_before_staging(saved):
    config, plan = saved
    with pytest.raises(ValueError, match="run plan changed"):
        rp.build_bundle(plan, workspace="/node/work", outdir="/node/work/new",
            expected_plan_sha256="0" * 64, expected_config_sha256=sha(config.read_bytes()))


@pytest.mark.parametrize("action", ["status", "logs", "stop"])
def test_staged_review_dispatch_does_not_shadow_existing_job_directory_lookup(tmp_path, monkeypatch, action):
    monkeypatch.setattr(rw.sys, "platform", "linux")
    monkeypatch.setattr(rw, "_workspace", lambda _: tmp_path)
    seen = []
    monkeypatch.setattr(rw, "_directory", lambda workspace, identifier: seen.append(identifier) or tmp_path)
    monkeypatch.setattr(rw, "_status", lambda directory: {"id": "job-fixture", "state": "running"})
    monkeypatch.setattr(rw, "_logs", lambda directory, request: {"text": "existing job log"})
    monkeypatch.setattr(rw, "_stop", lambda directory: {"job": {"id": "job-fixture", "state": "stopped"}})
    reply = rw.dispatch({"schema": "gpuwm.remote.request.v1", "action": action, "workspace": str(tmp_path), "job": "job-fixture"})
    assert seen == ["job-fixture"] and reply


def test_job_list_does_not_multiply_full_selected_render_summaries(tmp_path, monkeypatch):
    identifier = "20260907T180000-" + "a" * 16
    directory = tmp_path / identifier
    directory.mkdir()
    monkeypatch.setattr(rw.sys, "platform", "linux")
    monkeypatch.setattr(rw, "_workspace", lambda _:tmp_path)
    monkeypatch.setattr(rw, "_store", lambda _:tmp_path)
    monkeypatch.setattr(rw, "_directory", lambda *_:directory)
    summary = {"schema":"gpuwm.render-summary.v1", "rendered_png_count":25}
    monkeypatch.setattr(rw, "_status", lambda _:{"id":identifier,"state":"completed","stage":"completed","render_summary":summary})
    listed = rw.dispatch({"schema":"gpuwm.remote.request.v1","action":"list","workspace":str(tmp_path)})
    assert listed["jobs"] == [{"id":identifier,"state":"completed","stage":"completed"}]
    selected = rw.dispatch({"schema":"gpuwm.remote.request.v1","action":"status","workspace":str(tmp_path),"job":identifier})
    assert selected["job"]["render_summary"] == summary


def test_persisted_remote_job_retains_original_reviewed_source_identity_not_current_editor(tmp_path, monkeypatch):
    from gpuwm import remote_artifacts
    record={"id":"job-fixture","created_at":"2026-09-07T18:00:00Z","config":"/node/rewritten-case.toml",
        "config_sha256":"b"*64,"outdir":"/node/run","runtime":{},"action":"start-plan",
        "source":{"config_path":r"\\?\C:\saved\original-case.toml","config_sha256":"a"*64}}
    monkeypatch.setattr(rw,"_record",lambda _:copy.deepcopy(record))
    monkeypatch.setattr(remote_artifacts,"native_progress",lambda *_:{})
    value=rw._status(tmp_path)
    assert value["source_config_sha256"]=="a"*64
    assert value["source_config_path"]==record["source"]["config_path"]
    record["source"]["config_path"]="relative-case.toml"
    assert "source_config_sha256" not in rw._status(tmp_path)
    record.pop("source")
    assert "source_config_sha256" not in rw._status(tmp_path)


def test_large_selected_input_names_the_required_file(saved):
    config, _ = saved
    config.with_suffix(".d01-target.json").write_bytes(b"x" * (rp.MAX_SINGLE_BYTES + 1))
    with pytest.raises(ValueError, match="saved map.d01-target.json.*automatic companion staging"):
        bundle(saved)


def test_remote_wps_relocation_preserves_stable_ids_and_compact_slots(tmp_path):
    from gpuwm.namelist_import import parse_namelist_text
    from gpuwm.wps_domain_ids import domain_ids_from_wps_text
    original = "! GPUWM_DOMAIN_IDS_V1 = 1,3,4\n&share\n max_dom = 3,\n/\n&geogrid\n parent_id = 1, 1, 2,\n geog_data_res = '5m', 'default', 'modis_lai',\n/\n"
    actual = rp._wps_bytes(original.encode(), "/owned/geography", lambda *_: pytest.fail("no file lookup expected"), [], source=tmp_path / "case.wps").decode()
    assert domain_ids_from_wps_text(actual, 3) == (1, 3, 4)
    tables = parse_namelist_text(actual)
    assert tables["geogrid"]["parent_id"] == [1, 1, 2]
    assert tables["geogrid"]["geog_data_res"] == ["5m", "default", "modis_lai"]
    assert tables["geogrid"]["geog_data_path"] == ["/owned/geography"]


@pytest.mark.parametrize("comment", ["! GPUWM_DOMAIN_IDS_V1 = 1,3,3", "! GPUWM_DOMAIN_IDS_V2 = 1,3,4",
    "! GPUWM_DOMAIN_IDS_V1 = 1,3", "! GPUWM_DOMAIN_IDS_V1 = 1,3,4\n! GPUWM_DOMAIN_IDS_V1 = 1,3,4"])
def test_remote_wps_malformed_or_ambiguous_identity_is_not_silently_dropped(tmp_path, comment):
    with pytest.raises(ValueError, match="WPS domain identity"):
        rp._wps_bytes((comment + "\n&share\n max_dom=3,\n/\n").encode(), None, lambda *_: None, [], source=tmp_path / "case.wps")


def test_remote_wps_contiguous_canonical_bytes_and_settings_remain_unchanged(tmp_path):
    from gpuwm.namelist_import import parse_namelist_text
    original = "&share\n max_dom = 3,\n/\n&geogrid\n parent_id = 1, 1, 2,\n geog_data_res = '5m', 'default', 'modis_lai',\n/\n"
    actual = rp._wps_bytes(original.encode(), None, lambda *_: None, [], source=tmp_path / "case.wps")
    assert actual == original.encode()
    assert parse_namelist_text(actual.decode()) == parse_namelist_text(original)


def _selected_raw_case(saved, *, glob=False):
    from gpuwm.toml_document import emit_experiment_toml
    config, _plan = saved
    raw = tomllib.loads(config.read_text())
    selected = []
    for index in range(3):
        path = config.parent / f"selected-{index}.grib"
        path.write_bytes((f"raw protocol input {index}; not weather\n".encode()) * 3000)
        selected.append(path)
    wps = config.with_suffix(".namelist.wps")
    wps.write_text("&share\n max_dom = 1,\n/\n&geogrid\n dx = 12000.0,\n dy = 12000.0,\n/\n")
    vtable = config.parent / "Vtable"
    vtable.write_text("explicit selected table")
    raw["case_data"] = {"forcing": [str(config.parent / "selected-*.grib")] if glob else list(map(str, selected)),
        "vtable": str(vtable), "wps_namelist": str(wps), "geog_root": str(config.parent / "geog"),
        "sfcp_to_sfcp": True, "output_title": "selected raw inputs", "forcing_interval_s": 10800}
    config.write_text(emit_experiment_toml(raw))
    return selected


@pytest.mark.parametrize("glob", [False, True])
def test_large_selected_tuple_or_glob_uses_raw_hashes_without_json_payloads(saved, glob):
    selected = _selected_raw_case(saved, glob=glob)
    document = bundle(saved)
    assert document["schema"] == rp.BLOB_BUNDLE_SCHEMA
    assert len(document["blobs"]) == 3
    assert all("data" not in item for item in document["blobs"])
    assert len(rp._encoded(document)) < 120 * 1024
    raw = tomllib.loads(contents(document, "case.toml").decode())
    for index, (path, blob) in enumerate(zip(selected, document["blobs"])):
        assert blob["source_path"] == str(path.resolve())
        assert blob["sha256"] == sha(path.read_bytes())
        assert raw["case_data"]["forcing"][index].endswith(f"forcing/{index:02d}-{path.name}")
        assert str(path.resolve()) not in document["source_inputs"]
    assert not document["expected_downloads"]


def test_large_bundle_needs_verified_objects_and_captures_them_without_bulk_snapshot_bytes(saved, tmp_path, monkeypatch):
    from gpuwm import remote_input_transfer as transfer
    selected = _selected_raw_case(saved)
    document = local_worker_bundle(saved, tmp_path)
    with pytest.raises(ValueError, match="not completed"):
        rp.stage(document, tmp_path)
    assert not (tmp_path / ".arwen-plan-inputs" / document["id"]).exists()
    for item in document["blobs"]:
        transfer.receive({"schema": "gpuwm.remote.request.v1", "action": "put-input", "workspace": str(tmp_path),
                          "size": item["size"], "sha256": item["sha256"]}, tmp_path, __import__("io").BytesIO(Path(item["source_path"]).read_bytes()))
    rp.stage(document, tmp_path)
    assert rp.stage(document, tmp_path)["bundle_sha256"] == document["sha256"]
    saved_bundle, directory = rp.read_bundle(tmp_path, document["id"], document["sha256"])
    for item, source in zip(document["blobs"], selected):
        assert (directory / item["name"]).read_bytes() == source.read_bytes()
    value = {"memory": {"measured": True, "refuse": False}, "source_blobs": transfer.source_blobs(document),
             "plan_sha256": "a", "config_sha256": "b", "input_sha256": "c"}
    monkeypatch.setattr(rp, "review", lambda *_: (copy.deepcopy(value), saved_bundle, directory))
    observed = []
    monkeypatch.setattr(rw, "_launch_review", lambda *args: observed.append(args) or {"job": {"id": "fixture"}})
    request = {"expected_plan_sha256": "a", "expected_config_sha256": "b", "expected_input_sha256": "c"}
    with pytest.raises(ValueError, match="fresh verification"):
        rp.launch(request, tmp_path)
    request["expected_source_blobs_sha256"] = sha(rp._encoded(value["source_blobs"]))
    assert rp.launch(request, tmp_path)["job"]["id"] == "fixture"
    review, sources, snapshots = observed[0][2:]
    assert len(review["external_inputs"]) == 3
    assert sum(map(len, snapshots.values())) <= rp.MAX_INPUT_BYTES
    assert all(Path(path).stat().st_size > rp.MAX_SINGLE_BYTES for path in review["external_inputs"])


def test_cached_eda_keeps_exact_receipt_member_and_combined_file_together(saved, monkeypatch):
    from gpuwm import era5_member
    from gpuwm.toml_document import emit_experiment_toml
    _selected_raw_case(saved)
    config, _plan = saved
    raw = tomllib.loads(config.read_text())
    raw["fetch"].update(source="era5", cycle="2026-09-07T18", era5_provider="cds", era5_product="ensemble_members", member=7)
    raw["fetch"].pop("forecast_start_hour")
    cache = Path(raw["fetch"]["out"])
    cache.mkdir()
    forcing = cache / "era5-combined.grib"
    forcing.write_bytes(b"selected EDA byte-transport protocol fixture; no weather" * 2000)
    receipt = cache / "era5-acquisition.json"
    receipt.write_text(json.dumps({"schema": "arwen.era5-acquisition.v1", "status": "validated", "request": {"member": 7},
        "artifact": {"name": forcing.name, "bytes": forcing.stat().st_size, "sha256": sha(forcing.read_bytes())},
        "member_selection": {"member": 7, "byte_preserving": True, "native_identity": "retained unchanged"}}))
    raw["case_data"]["forcing"] = [str(forcing)]
    config.write_text(emit_experiment_toml(raw))
    checks = []
    monkeypatch.setattr(era5_member, "check_member", lambda path, member: checks.append((Path(path), member)))
    document = bundle(saved)
    relocated = tomllib.loads(contents(document, "case.toml").decode())
    assert checks == [(forcing, 7)]
    assert relocated["case_data"]["forcing"] == [relocated["fetch"]["out"] + "/era5-combined.grib"]
    assert not document["expected_downloads"]
    assert {blob["name"] for blob in document["blobs"]} == {forcing.name, receipt.name}
    assert all(blob["placement"] == "data" for blob in document["blobs"])
    saved_receipt = next(blob for blob in document["blobs"] if blob["name"] == receipt.name)
    assert saved_receipt["sha256"] == sha(receipt.read_bytes())
    wrong = json.loads(receipt.read_text())
    wrong["request"]["member"] = 8
    receipt.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="different native ensemble member"):
        bundle(saved)


def test_remote_cds_acquisition_refuses_missing_credentials_before_memory_or_launch(saved, tmp_path, monkeypatch):
    from gpuwm import cds_credentials
    config, plan = saved
    document = bundle(saved)
    document["geog_root"] = str(tmp_path)
    document["outdir"] = str(tmp_path / "new-output")
    document["expected_downloads"] = [{"source": "era5", "recipe": {"era5_provider": "cds"}}]
    stage = tmp_path / "staged"
    stage.mkdir()
    (stage / "case.toml").write_bytes(config.read_bytes())
    (stage / "plan.json").write_bytes(plan.read_bytes())
    monkeypatch.setattr(rp, "read_bundle", lambda *_: (document, stage))
    monkeypatch.setattr(cds_credentials, "acquisition_readiness", lambda: {"ready": False, "message": "configure the selected node's CDS credentials"}, raising=False)
    monkeypatch.setattr(rp, "memory_review", lambda *_args, **_kwargs: pytest.fail("credential refusal must precede memory/launch"))
    with pytest.raises(ValueError, match="Selected node cannot acquire ERA5 from CDS.*selected node"):
        rp.review({}, tmp_path)


def test_required_missing_forcing_without_matching_fetch_is_not_dropped(saved):
    from gpuwm.toml_document import emit_experiment_toml
    config, _ = saved
    raw = tomllib.loads(config.read_text())
    raw["case_data"] = {"forcing": [str(config.parent / "specifically-selected.grib")],
        "vtable": "Vtable", "wps_namelist": "namelist.wps", "geog_root": "geo",
        "sfcp_to_sfcp": True, "output_title": "selected forcing"}
    config.write_text(emit_experiment_toml(raw))
    with pytest.raises(ValueError, match="specifically-selected.grib.*not the exact output"):
        bundle(saved)


def test_worker_staging_is_idempotent_and_detects_changed_input(saved, tmp_path):
    document = local_worker_bundle(saved, tmp_path)
    first = rp.stage(document, tmp_path)
    assert rp.stage(document, tmp_path) == first
    _, directory = rp.read_bundle(tmp_path, document["id"], document["sha256"])
    (directory / "case.toml").write_text("changed")
    with pytest.raises(ValueError, match="case.toml.*changed"):
        rp.read_bundle(tmp_path, document["id"], document["sha256"])
    assert not (tmp_path / ".arwen-jobs").exists()


@pytest.mark.parametrize("name", ["../outside", "/outside", "a/../../outside", "a\\outside"])
def test_stage_rejects_traversal_before_publication(saved, tmp_path, name):
    document = local_worker_bundle(saved, tmp_path)
    document["files"][0]["name"] = name
    rehash(document)
    with pytest.raises(ValueError, match="inside its bundle"):
        rp.stage(document, tmp_path)
    assert not (tmp_path / ".arwen-plan-inputs").exists()


def test_stage_rejects_foreign_workspace_and_tampered_bytes(saved, tmp_path):
    document = bundle(saved)
    with pytest.raises(ValueError, match="different remote workspace"):
        rp.stage(document, tmp_path)
    document = local_worker_bundle(saved, tmp_path)
    document["files"][0]["data"] = base64.b64encode(b"tampered").decode()
    rehash(document)
    with pytest.raises(ValueError, match="size or SHA-256"):
        rp.stage(document, tmp_path)


@pytest.mark.parametrize("streamed", [False, True])
def test_memory_review_uses_exact_native_gate_and_resolved_experiment(monkeypatch, saved, streamed):
    from dataclasses import replace
    from gpuwm import go_cli
    from gpuwm.core import preflight
    calls = []
    exp = preflight._load_experiment_any(saved[0])
    phases = preflight.estimate_phases(exp, source="gfs")
    probe = {"total_bytes": 1000, "free_bytes": 200, "private_unrelated": "not for transport",
             "profile": {"name": "selected device", "multiprocessor_count": 70,
                "max_threads_per_multiprocessor": 1536, "default_stack_limit_bytes": 1024,
                "bare_context_bytes": None}}
    if streamed:
        phases = replace(phases, streamed=SimpleNamespace(), forecast_envelope_bytes=1234)
    def gate(plan, **kwargs):
        calls.append((plan, kwargs))
        return {"free_bytes": 200, "budget_bytes": 180, "refuse": False, "warn": False,
                "verdict": "native verdict", "phases": phases, "device_probe": probe}
    monkeypatch.setattr(go_cli, "memory_gate", gate)
    value = rp.memory_review(Path("saved.toml"), experiment=exp, cadence=3)
    assert calls == [({"config": "saved.toml", "cadence": 3}, {"experiment": exp})]
    assert value["measured"] and value["peak_envelope_bytes"] == phases.peak_envelope_bytes and value["ingest_priced"]
    detail = value["breakdown"]
    assert detail["peak_envelope_bytes"] == value["peak_envelope_bytes"]
    assert detail["binding_phase"] == phases.binding_phase
    assert detail["forecast_envelope_bytes"] == phases.forecast_envelope_bytes
    assert detail["forecast"]["resident_bytes"] == phases.forecast.resident_bytes
    assert detail["forecast"]["workspace_bytes"] == phases.forecast.workspace_bytes
    assert detail["forecast"]["non_pool_device_bytes"] == phases.forecast.non_pool_device_bytes
    assert detail["forecast"]["allocation_scope"] == ("resident_reference" if streamed else "resident_execution")
    assert value["execution"]["configured_mode"] == exp.tiles.mode
    assert value["execution"]["streamed_forecast"] == streamed
    assert value["execution"]["selected_forecast_envelope_bytes"] == phases.forecast_envelope_bytes
    assert value["execution"]["resident_reference_bytes"] == phases.resident_forecast_envelope_bytes
    assert detail["forecast"]["terms"] == phases.forecast.peak_envelope_terms()
    assert [domain["grid_id"] for domain in detail["domains"]] == [domain.grid_id for domain in exp.domains]
    assert value["sizing"]["total_bytes"] == probe["total_bytes"]
    assert value["sizing"]["free_bytes"] == probe["free_bytes"]
    assert value["sizing"]["profile"] == probe["profile"]
    assert "private_unrelated" not in value["sizing"]
    json.dumps(value, allow_nan=False)


def test_connection_probes_sizing_once_and_review_reuses_the_existing_measurement(monkeypatch):
    from gpuwm.core import preflight
    from gpuwm import runplan
    calls = []
    probe = {"total_bytes": 32 * 1024 ** 3, "free_bytes": 29 * 1024 ** 3,
             "profile": {"name": "selected GPU", "multiprocessor_count": 170,
                "max_threads_per_multiprocessor": 1536, "default_stack_limit_bytes": 1024,
                "bare_context_bytes": 512 * 1024 ** 2}}
    def measure():
        calls.append(True)
        return probe
    monkeypatch.setattr(preflight, "device_memory_probe_subprocess", measure)
    monkeypatch.setattr(runplan, "probe_environment", lambda **_: {"devices": []})
    connected = rp.hardware_probe()
    assert calls == [True]
    reviewed = rp.hardware_probe(sizing=connected["sizing"], measure_sizing=False)
    assert calls == [True] and reviewed["sizing"] == connected["sizing"]
    assert rp.hardware_probe(measure_sizing=False)["sizing"] is None
    assert calls == [True]


@pytest.mark.parametrize("measured,refused", [(False, False), (True, True)])
def test_launch_needs_measured_admitted_memory(monkeypatch, tmp_path, measured, refused):
    review = {"memory": {"measured": measured, "refuse": refused, "verdict": "native refusal"},
              "plan_sha256": "a" * 64, "config_sha256": "b" * 64, "input_sha256": "c" * 64}
    monkeypatch.setattr(rp, "review", lambda *_: (review, {}, tmp_path))
    request = {"expected_plan_sha256": "a" * 64, "expected_config_sha256": "b" * 64,
               "expected_input_sha256": "c" * 64}
    with pytest.raises(ValueError, match="not launch-ready: native refusal"):
        rp.launch(request, tmp_path)
    assert not (tmp_path / ".arwen-jobs").exists()


def test_review_cli_stages_then_reviews_without_starting(saved, monkeypatch, capsys):
    config, plan = saved
    parser = ArgumentParser()
    rc.register_cli(parser.add_subparsers())
    args = parser.parse_args(["remote", "review-plan", "--host", "node-1", "--python", "/node/python",
        "--workspace", "/node/work", "--plan", str(plan), "--outdir", "/node/work/new",
        "--expected-plan-sha256", sha(plan.read_bytes()), "--expected-config-sha256", sha(config.read_bytes()), "--json"])
    monkeypatch.setattr(rc, "ssh_command", lambda _: ["never-executed"])
    calls = []
    def transport(_command, request, **_kwargs):
        calls.append(request)
        if request["action"] == "stage-plan":
            return rc.result("stage-plan", bundle_id=request["bundle"]["id"], bundle_sha256=request["bundle"]["sha256"])
        return rc.result("review-plan", dry_run=True, review={"source": calls[0]["bundle"]["source"]})
    monkeypatch.setattr(rc, "_transport", transport)
    assert rc.remote_main(args) == 0
    assert [c["action"] for c in calls] == ["stage-plan", "review-plan"]
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
