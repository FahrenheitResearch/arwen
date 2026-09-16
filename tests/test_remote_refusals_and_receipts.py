"""Refusals name the breakage and the way out; receipts survive to the status door.

Every fixture here is protocol metadata and CPU-only bytes: no weather is
decoded, no forecast is run and no card is touched.
"""
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import (remote_artifacts as ra, remote_cli as rc, remote_input_transfer as transfer,
                   remote_plan as rp, remote_worker as rw)


@pytest.fixture(autouse=True)
def cpu_only_memory_review(monkeypatch):
    monkeypatch.setattr("gpuwm.remote_plan.memory_review", lambda *_a, **_k:
        {"measured": False, "free_bytes": None, "refuse": False, "warn": True,
         "verdict": "CPU-only refusal contract test"})


def config(tmp_path, *, restart_interval_s=3600.0, name="refusal-contract"):
    from gpuwm import domain_wizard as dw
    path = tmp_path / "science's 日本語.toml"
    text = dw.render_config(name=name, start_time=datetime(2026, 9, 5), hours=3,
        projection=dw._projection_entries(40, -100, "auto"), dims=dw._dims_for_scale(1, ()), ratios=(),
        fetch_hints=dict(source="gfs", cycle="2026-09-05T00", hours=3, out="data/cache", cadence=3),
        case_data=None)
    text = text.replace("restart_interval_s = 3600.0", f"restart_interval_s = {restart_interval_s}")
    path.write_text(text, encoding="utf-8")
    assert f"restart_interval_s = {restart_interval_s}" in text
    return path


# ---------------------------------------------------------------- restart cadence

@pytest.mark.parametrize("interval", [0.0, 3600.0])
def test_checkpointless_resume_names_the_cadence_the_source_job_declared(tmp_path, interval):
    source = config(tmp_path, restart_interval_s=interval)
    output = tmp_path / "old output"
    output.mkdir()
    with pytest.raises(ValueError) as failure:
        rw._checkpoint({"outdir": str(output), "snapshot_config": str(source)}, "latest")
    message = str(failure.value)
    assert "No manifest-valid checkpoint exists" in message
    assert f"restart_interval_s = {interval}" in message
    if interval == 0.0:
        assert "wrote no checkpoints" in message
    assert list(output.iterdir()) == []


# ------------------------------------------------------- transfer integrity wording

def _receiver_program():
    return ("import json,sys;from pathlib import Path;from gpuwm.remote_input_transfer import receive;"
            "v=json.loads(sys.stdin.buffer.readline());r=receive(v,Path(v['workspace']),sys.stdin.buffer);"
            "print(json.dumps(dict(schema='gpuwm.remote.result.v1',ok=True,action='put-input',**r)))")


def _upload(tmp_path, source, reviewed, *, mutate):
    import sys
    node = tmp_path / "node"
    node.mkdir()
    value = {"schema": "gpuwm.remote.request.v1", "action": "put-input", "workspace": str(node),
             "size": len(reviewed), "sha256": hashlib.sha256(reviewed).hexdigest()}
    source.write_bytes(mutate)
    with pytest.raises(ValueError) as failure:
        transfer.upload([sys.executable, "-c", _receiver_program()], value, source, timeout=20)
    return str(failure.value), value


def test_changed_input_refusal_names_the_file_its_digest_and_the_review(tmp_path):
    reviewed = b"reviewed raw protocol fixture" * 500
    source = tmp_path / "selected forcing.grib2"
    source.write_bytes(reviewed)
    changed = bytes(len(reviewed))
    message, value = _upload(tmp_path, source, reviewed, mutate=changed)
    assert "selected forcing.grib2" in message
    assert hashlib.sha256(changed).hexdigest() in message
    assert value["sha256"] in message
    assert "Review this plan again" in message


def test_grown_input_refusal_names_the_file_the_reviewed_size_and_the_review(tmp_path):
    reviewed = b"reviewed raw protocol fixture" * 500
    source = tmp_path / "selected forcing.grib2"
    source.write_bytes(reviewed)
    message, value = _upload(tmp_path, source, reviewed, mutate=reviewed + b"X" * (2 * 1024 * 1024))
    assert "selected forcing.grib2" in message
    assert str(value["size"]) in message
    assert "Review this plan again" in message


# ------------------------------------------------------------------- geography

def test_one_geography_absence_sentence_from_both_staging_doors(tmp_path):
    declared = "/declared/geography"
    first = rp._geography_absence(declared)
    assert declared in first and "64 KiB" in first
    assert "Remote geography folder" in first
    # The fact is one sentence set, so the two doors cannot drift apart.
    assert first == rp._geography_absence(declared)
    assert "declared at" not in rp._geography_absence(None)
    assert "64 KiB" in rp._geography_absence(None)


def test_the_plan_relocation_door_uses_the_same_geography_sentence(tmp_path):
    """The second door that decides the same fact says the same thing."""
    from datetime import datetime
    from gpuwm import domain_wizard as dw
    saved_config = tmp_path / "saved map.toml"
    saved_config.write_text(dw.render_config(name="remote-map", start_time=datetime(2026, 9, 7, 18),
        hours=24, projection=dw._projection_entries(18, -162.1, "auto"),
        dims=dw._dims_for_scale(1, ()), ratios=(),
        fetch_hints=dict(source="gfs", cycle="2026-09-07T12", forecast_start_hour=6, hours=24,
                         out=str(tmp_path / "forcing"), cadence=3), case_data=None), encoding="utf-8")
    declared = str(tmp_path / "authored-geography")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "gpuwm.run-plan.v1", "name": "reviewed-map",
        "route": "prepared", "config": {"path": str(saved_config)},
        "output_root": str(tmp_path / "local-output"),
        "run_options": {"render_products": "none", "geog_root": declared}}), encoding="utf-8")
    with pytest.raises(ValueError) as failure:
        rp.build_bundle(plan, workspace="/node/work", outdir="/node/work/new-output", geog_root=None,
                        expected_plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
                        expected_config_sha256=hashlib.sha256(saved_config.read_bytes()).hexdigest())
    assert str(failure.value) == rp._geography_absence(declared)


# ---------------------------------------------------------------- route statement

def test_the_node_configuration_review_states_the_route_it_planned(tmp_path):
    source = config(tmp_path)
    request = {"schema": "gpuwm.remote.request.v1", "action": "start", "workspace": str(tmp_path),
               "config": str(source), "outdir": str(tmp_path / "new output"), "products": "none"}
    review, _sources, _snapshots = rw._review(request, tmp_path)
    assert review["route"] == "node_config"
    assert str(source) in review["route_reason"]


def test_an_unregistered_route_is_refused_by_name_with_the_registered_set():
    with pytest.raises(ValueError) as failure:
        rw.route_statement("no-such-route", "detail")
    message = str(failure.value)
    assert "no-such-route" in message
    assert "node_config" in message and "staged_plan" in message


# ------------------------------------------------------------ status diagnostics

@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """A durable job record on disk, with no worker process behind it."""
    store = tmp_path / ".arwen-jobs"
    store.mkdir(mode=0o700)
    directory = store / "20260907T180000-0123456789abcdef"
    directory.mkdir(mode=0o700)
    outdir = tmp_path / "run"
    outdir.mkdir()
    inputs = directory / "inputs"
    inputs.mkdir()
    saved = inputs / "case.toml"
    saved.write_text("[experiment]\n")
    record = {"schema": "gpuwm.remote.job.v1", "id": directory.name, "token": "a" * 64,
              "created_at": "2026-09-07T17:59:59+00:00", "action": "start", "config": str(saved),
              "outdir": str(outdir), "runtime": {"version": "2.7.4"}, "snapshot_plan": None,
              "snapshot_config": str(saved), "snapshot_sha256": rw._file_sha(saved),
              "config_sha256": rw._file_sha(saved)}
    rw._write(directory / "job.json", record)
    return SimpleNamespace(tmp_path=tmp_path, directory=directory, record=record, outdir=outdir)


def test_status_names_the_unreadable_native_receipt_instead_of_looking_like_preparing(recorded, monkeypatch):
    monkeypatch.setattr(ra, "native_progress", lambda *_:
        (_ for _ in ()).throw(ValueError("Remote run manifest does not match this job's saved identity")))
    job = rw._status(recorded.directory)
    assert job["state"] == "starting"
    assert "does not match this job's saved identity" in job["native_progress_error"]["message"]
    assert job["native_progress_error"]["receipt"] == str(recorded.outdir / "run-manifest.json")
    assert job["native_progress_error"]["class"] == "ValueError"


def test_a_clean_native_progress_read_adds_no_diagnosis_key(recorded, monkeypatch):
    monkeypatch.setattr(ra, "native_progress", lambda *_: {})
    assert "native_progress_error" not in rw._status(recorded.directory)


def test_status_reports_a_watcher_that_could_not_start(recorded):
    rw._write(recorded.directory / "preparation-start-error.json", {
        "token": recorded.record["token"], "at": "2026-09-07T18:00:05+00:00",
        "background_maps": {"error": "map watcher import failed", "at": "2026-09-07T18:00:05+00:00"},
        "native_plots": {"error": "gallery watcher import failed", "at": "2026-09-07T18:00:05+00:00"}})
    job = rw._status(recorded.directory)
    for key, text in (("background_maps", "map watcher"), ("native_plots", "gallery watcher")):
        assert job[key]["state"] == "start_failed"
        assert text in job[key]["error"]
        assert job[key]["at"] == "2026-09-07T18:00:05+00:00"


def test_a_watcher_start_receipt_from_another_job_is_refused(recorded):
    """The refusal names the breakage, the receipt it read and the way out."""
    receipt = recorded.directory / "preparation-start-error.json"
    rw._write(receipt, {"token": "b" * 64, "background_maps": {"error": "planted", "at": "now"}})
    with pytest.raises(ValueError) as failure:
        rw._status(recorded.directory)
    message = str(failure.value)
    assert "another job's failed watcher" in message
    assert str(receipt) in message and "read this job's status again" in message


def test_a_watcher_start_receipt_written_before_the_token_was_recorded_is_reported_not_refused(recorded):
    """A pre-2.7.5 worker wrote no token; the status door reads it on the basis of its folder."""
    receipt = recorded.directory / "preparation-start-error.json"
    rw._write(receipt, {"background_maps": {"error": "planted", "at": "now"}})
    status = rw._status(recorded.directory)
    assert status["background_maps"]["state"] == "start_failed"
    assert status["background_maps"]["error"] == "planted"
    assert status["background_maps"]["basis"] == rw.LEGACY_RECEIPT_BASIS
    # A token that is present and different is still another job's receipt.
    rw._write(receipt, {"token": "b" * 64, "background_maps": {"error": "planted", "at": "now"}})
    with pytest.raises(ValueError, match="another job's failed watcher"):
        rw._status(recorded.directory)


def test_a_cleanup_receipt_written_before_the_token_was_recorded_is_reported_not_refused(recorded):
    receipt = recorded.directory / "cleanup-error.json"
    rw._write(receipt, {"error": "planted", "surviving": []})
    status = rw._status(recorded.directory)
    assert status["state"] == "cleanup_failed"
    assert status["basis"] == rw.LEGACY_RECEIPT_BASIS
    assert "planted" in status["error"]


def test_a_cleanup_receipt_from_another_job_is_refused(recorded):
    """Same register at the cleanup door: breakage, receipt, way out."""
    receipt = recorded.directory / "cleanup-error.json"
    rw._write(receipt, {"token": "b" * 64, "error": "planted", "surviving": []})
    with pytest.raises(ValueError) as failure:
        rw._status(recorded.directory)
    message = str(failure.value)
    assert "cannot be signalled as this job's" in message
    assert str(receipt) in message and "stop this job again" in message


def test_one_watcher_failing_to_start_does_not_cancel_the_other(recorded, monkeypatch):
    started = []
    monkeypatch.setattr(rw, "_watcher_starters", lambda: {
        "background_maps": lambda *_: (_ for _ in ()).throw(ImportError("no map watcher here")),
        "native_plots": lambda *args: started.append(args)})
    failures = rw._start_watchers(recorded.tmp_path, recorded.record["id"],
                                 recorded.directory, recorded.record["token"])
    assert set(failures) == {"background_maps"}
    assert len(started) == 1
    assert rw._status(recorded.directory)["background_maps"]["state"] == "start_failed"
    assert "native_plots" not in rw._status(recorded.directory)


def test_two_pollers_retrying_one_watcher_do_not_fail_on_each_other(recorded, monkeypatch):
    """A status retry is a read for the caller: the loser of the race is silent."""
    receipt = recorded.directory / "preparation-start-error.json"
    rw._write(receipt, {"token": recorded.record["token"], "at": "2026-09-07T18:00:05+00:00",
                        "background_maps": {"error": "map watcher import failed",
                                            "at": "2026-09-07T18:00:05+00:00"}})
    def racing(_workspace, _job):
        receipt.unlink(missing_ok=True)  # The other poller's retry already cleared it.
    monkeypatch.setattr(rw, "_watcher_starters",
                        lambda: {"background_maps": racing, "native_plots": lambda *_: None})
    assert rw._start_watchers(recorded.tmp_path, recorded.record["id"], recorded.directory,
                              recorded.record["token"], only={"background_maps"}) == {}
    assert not receipt.exists()
    assert "background_maps" not in rw._status(recorded.directory)


def test_a_successful_retry_clears_the_start_failure(recorded, monkeypatch):
    rw._write(recorded.directory / "preparation-start-error.json", {
        "token": recorded.record["token"], "at": "2026-09-07T18:00:05+00:00",
        "background_maps": {"error": "map watcher import failed", "at": "2026-09-07T18:00:05+00:00"}})
    assert rw._status(recorded.directory)["background_maps"]["state"] == "start_failed"
    monkeypatch.setattr(rw, "_watcher_starters", lambda: {
        "background_maps": lambda *_: None, "native_plots": lambda *_: None})
    reply = rw.dispatch({"schema": "gpuwm.remote.request.v1", "action": "status",
                         "workspace": str(recorded.tmp_path), "job": recorded.record["id"]})
    assert "background_maps" not in reply["job"]
    assert not (recorded.directory / "preparation-start-error.json").exists()


# ----------------------------------------------------------------- cleanup failure

def test_status_reports_cleanup_failure_with_the_surviving_owned_processes(recorded):
    rw._write(recorded.directory / "cleanup-error.json", {
        "token": recorded.record["token"], "at": "2026-09-07T18:09:00+00:00",
        "error": "owned descendants still exist after termination attempts",
        "surviving": [{"pid": 4321, "uid": 1000, "boot_id": "fixture", "start_ticks": 7}]})
    job = rw._status(recorded.directory)
    assert job["state"] == "cleanup_failed"
    assert "owned descendants still exist" in job["error"]
    assert "Stop this job again" in job["error"]
    assert job["surviving"][0]["pid"] == 4321


def test_cleanup_failed_is_not_terminal_and_stop_can_still_reap_it(recorded, monkeypatch):
    rw._write(recorded.directory / "cleanup-error.json", {
        "token": recorded.record["token"], "at": "2026-09-07T18:09:00+00:00",
        "error": "owned descendants still exist after termination attempts",
        "surviving": [{"pid": 4321, "uid": 1000, "boot_id": "fixture", "start_ticks": 7}]})
    assert "cleanup_failed" not in rw.TERMINAL
    monkeypatch.setattr(rw, "_reap_owned", lambda *_a, **_k: [])
    job = rw._stop(recorded.directory)["job"]
    assert job["state"] == "stopped"
    assert not (recorded.directory / "cleanup-error.json").exists()


def test_a_stop_that_cannot_reap_names_the_surviving_processes(recorded, monkeypatch):
    rw._write(recorded.directory / "cleanup-error.json", {
        "token": recorded.record["token"], "at": "2026-09-07T18:09:00+00:00",
        "error": "owned descendants still exist after termination attempts", "surviving": []})
    monkeypatch.setattr(rw, "_reap_owned", lambda *_a, **_k: [{"pid": 4321}])
    with pytest.raises(ValueError, match="4321"):
        rw._stop(recorded.directory)


# --------------------------------------------------- acquisition binding wording

def _case_data(tmp_path, forcing, geog_root):
    """A complete declared [case_data] table; every key the schema requires."""
    wps = tmp_path / "saved map.namelist.wps"
    wps.write_text("&share\n max_dom=1,\n interval_seconds=10800,\n/\n&geogrid\n dx=12000.,\n dy=12000.,\n/\n")
    vtable = tmp_path / "selected.Vtable"
    vtable.write_text("selected scientific Vtable")
    return {"forcing": [str(forcing)], "vtable": str(vtable), "wps_namelist": str(wps),
            "geog_root": geog_root, "forcing_interval_s": 10800, "sfcp_to_sfcp": True,
            "output_domain": 1, "output_title": "saved map"}


def test_the_data_dir_refusal_names_the_breakage_and_both_ways_out(tmp_path):
    """A plan that names both [case_data].forcing and run_options.data_dir."""
    import tomllib
    from datetime import datetime
    from gpuwm import domain_wizard as dw
    from gpuwm.toml_document import emit_experiment_toml
    saved_config = tmp_path / "saved map.toml"
    saved_config.write_text(dw.render_config(name="remote-map", start_time=datetime(2026, 9, 7, 18),
        hours=24, projection=dw._projection_entries(18, -162.1, "auto"),
        dims=dw._dims_for_scale(1, ()), ratios=(),
        fetch_hints=dict(source="gfs", cycle="2026-09-07T12", forecast_start_hour=6, hours=24,
                         out=str(tmp_path / "forcing"), cadence=3), case_data=None), encoding="utf-8")
    raw = tomllib.loads(saved_config.read_text())
    forcing = tmp_path / "forcing" / "declared.grib2"
    forcing.parent.mkdir(exist_ok=True)
    forcing.write_bytes(b"protocol fixture forcing bytes")
    raw["case_data"] = _case_data(tmp_path, forcing, str(tmp_path / "local-geog"))
    saved_config.write_text(emit_experiment_toml(raw))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "gpuwm.run-plan.v1", "name": "reviewed-map",
        "route": "prepared", "config": {"path": str(saved_config)},
        "output_root": str(tmp_path / "local-output"),
        "run_options": {"render_products": "none", "data_dir": str(tmp_path / "cache")}}), encoding="utf-8")
    with pytest.raises(ValueError) as failure:
        rp.build_bundle(plan, workspace="/node/work", outdir="/node/work/new-output",
                        geog_root="/node/geography",
                        expected_plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
                        expected_config_sha256=hashlib.sha256(saved_config.read_bytes()).hexdigest())
    message = str(failure.value)
    assert "none of the running forcing comes from" in message
    assert "Omit that run option" in message
    assert "remove [case_data] and let the saved [fetch] recipe" in message
    assert "is unused" not in message
    assert not (tmp_path / ".arwen-plan-inputs").exists()


def test_a_geography_less_case_data_configuration_gets_the_shared_sentence(tmp_path):
    import tomllib
    from datetime import datetime
    from gpuwm import domain_wizard as dw
    from gpuwm.toml_document import emit_experiment_toml
    saved_config = tmp_path / "saved map.toml"
    saved_config.write_text(dw.render_config(name="remote-map", start_time=datetime(2026, 9, 7, 18),
        hours=24, projection=dw._projection_entries(18, -162.1, "auto"),
        dims=dw._dims_for_scale(1, ()), ratios=(),
        fetch_hints=dict(source="gfs", cycle="2026-09-07T12", forecast_start_hour=6, hours=24,
                         out=str(tmp_path / "forcing"), cadence=3), case_data=None), encoding="utf-8")
    raw = tomllib.loads(saved_config.read_text())
    forcing = tmp_path / "forcing" / "declared.grib2"
    forcing.parent.mkdir(exist_ok=True)
    forcing.write_bytes(b"protocol fixture forcing bytes")
    declared_geography = str(tmp_path / "local-geog")
    raw["case_data"] = _case_data(tmp_path, forcing, declared_geography)
    saved_config.write_text(emit_experiment_toml(raw))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": "gpuwm.run-plan.v1", "name": "reviewed-map",
        "route": "prepared", "config": {"path": str(saved_config)},
        "output_root": str(tmp_path / "local-output"),
        "run_options": {"render_products": "none"}}), encoding="utf-8")
    with pytest.raises(ValueError) as failure:
        rp.build_bundle(plan, workspace="/node/work", outdir="/node/work/new-output", geog_root=None,
                        expected_plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
                        expected_config_sha256=hashlib.sha256(saved_config.read_bytes()).hexdigest())
    message = str(failure.value)
    assert message == rp._geography_absence(declared_geography)
    assert declared_geography in message and "64 KiB" in message
    assert "Remote geography folder" in message
