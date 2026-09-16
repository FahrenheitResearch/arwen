"""A job's manifest binding names its run, not the action word it was started by.

Every fixture here is protocol metadata: no weather is decoded and no forecast
is run. The configuration route records ``snapshot_config`` and no plan, and its
run publishes ``plan_source = "gpuwm go <saved config>"``; the staged route
records the plan document. Both are bindings the doors can prove.
"""
import json
from types import SimpleNamespace

import pytest

from gpuwm import (remote_artifacts as ra, remote_native_plots as plots,
                   remote_preparation_v2 as preparation, remote_processed as legacy)
from test_remote_artifacts import case, encoded


@pytest.fixture
def go_case(case):
    """The same durable tree, recorded the way a configuration job records it."""
    inputs = case.job_dir / "inputs"
    inputs.mkdir()
    config = inputs / "case.toml"
    config.write_text("[experiment]\nstart_time = 2026-09-07T18:00:00Z\nrun_seconds = 3600.0\nrestart_interval_s = 3600.0\n"
                      "[[domain]]\ngrid_id = 1\nhistory_interval_s = 900.0\n")
    digest = ra._file_sha(config)
    case.record.clear()
    case.record.update({"id": "job-fixture", "action": "start", "outdir": str(case.output),
                        "snapshot_plan": None, "snapshot_config": str(config),
                        "config_sha256": digest, "snapshot_sha256": digest,
                        "created_at": "2026-09-07T17:59:59+00:00", "token": "a" * 64})
    case.manifest["plan_source"] = "gpuwm go " + str(config)
    case.manifest["plan_sha256"] = "c" * 64
    case.manifest_path.write_bytes(encoded(case.manifest))
    return SimpleNamespace(config=config, **vars(case))


def test_a_configuration_route_job_serves_its_committed_frames(go_case):
    value = ra.catalog(go_case.request, go_case.tmp_path)
    assert not value["waiting"]
    assert value["run_id"] == "run-fixture"
    assert value["frames"][0]["sha256"] == ra._file_sha(go_case.frame)


def test_a_resumed_configuration_job_binds_its_own_saved_config(go_case):
    go_case.record.update({"action": "resume", "parent_job": "job-parent"})
    value = ra.catalog(go_case.request, go_case.tmp_path)
    assert value["run_id"] == "run-fixture"
    # Nothing of the parent is inherited: the binding is this record's own.
    assert go_case.record["snapshot_plan"] is None
    assert str(go_case.config.parent) == str(go_case.job_dir / "inputs")


@pytest.mark.parametrize("field,value", [("plan_source", "gpuwm go /elsewhere/case.toml"),
                                         ("plan_source", "planted"), ("plan_sha256", "not-hex")])
def test_a_manifest_naming_another_identity_is_still_refused(go_case, field, value):
    go_case.manifest[field] = value
    go_case.manifest_path.write_bytes(encoded(go_case.manifest))
    with pytest.raises(ValueError, match="does not match this job's saved plan"):
        ra.catalog(go_case.request, go_case.tmp_path)


def test_a_record_with_no_manifest_binding_at_all_still_declines(go_case):
    go_case.record["snapshot_config"] = None
    assert ra.bound_manifest(go_case.record, go_case.status) is None
    assert ra.catalog(go_case.request, go_case.tmp_path)["waiting"]


def test_native_stores_bind_a_configuration_route_job(go_case):
    record, state, bound, commits = legacy._job(go_case.tmp_path, record_id(go_case))
    assert bound is not None and bound[2]["run_id"] == "run-fixture"
    assert len(commits) == 1 and commits[0][0]["sequence"] == 1


def test_native_store_queue_registers_a_configuration_route_job(go_case):
    legacy.ensure(go_case.tmp_path, record_id(go_case))
    marker = legacy._root(go_case.tmp_path) / "requests" / (record_id(go_case) + ".json")
    assert json.loads(marker.read_text())["schema"] == legacy.QUEUE_SCHEMA


def test_both_detached_watchers_launch_for_a_configuration_route_job(go_case, monkeypatch):
    launched = []

    def capture(argv, **kwargs):
        launched.append(argv)
        return SimpleNamespace(pid=1, poll=lambda: None)

    monkeypatch.setattr(preparation.subprocess, "Popen", capture)
    monkeypatch.setattr(plots.subprocess, "Popen", capture)
    preparation.ensure(go_case.tmp_path, record_id(go_case))
    plots.ensure(go_case.tmp_path, record_id(go_case))
    assert len(launched) == 2
    for argv in launched:
        assert argv[argv.index("--job") + 1] == record_id(go_case)
        assert argv[argv.index("--workspace") + 1] == str(go_case.tmp_path)


def test_preparation_publishes_its_waiting_receipt_before_the_manifest_exists(go_case):
    go_case.manifest_path.unlink()
    state = preparation.prepare(go_case.tmp_path, record_id(go_case), start=False)
    assert state["state"] == "waiting_for_output"


def test_native_progress_reads_the_heartbeat_a_configuration_job_writes(go_case):
    from gpuwm.supervisor import HEARTBEAT_SCHEMA
    progress_path = go_case.output / "run-progress.json"
    go_case.manifest["progress_path"] = str(progress_path)
    go_case.manifest_path.write_bytes(encoded(go_case.manifest))
    # The run loads the snapshot, so the snapshot digest is what it stamps; the
    # source file's own digest is a different number on this route.
    go_case.record["config_sha256"] = "d" * 64
    progress_path.write_bytes(encoded({
        "schema": HEARTBEAT_SCHEMA, "run_id": go_case.manifest["run_id"], "pid": 123,
        "config_digest": go_case.record["snapshot_sha256"],
        "started_at_utc": go_case.manifest["started_at_utc"],
        "updated_at_utc": "2026-09-07T18:00:20Z", "status": "integrating",
        "model_elapsed_seconds": 120.0}))
    summary = ra.native_progress(go_case.record, go_case.status)
    assert summary["phase"] == "integrating" and summary["model_elapsed_seconds"] == 120.0
    assert summary["valid_time"] == go_case.event["valid_time"]


def test_the_viewer_reads_the_initialization_from_the_snapshot_the_run_loads(go_case):
    """Staging re-emits the configuration, so the two digests are different.

    The saved snapshot is the document the run loads and the one this read
    opens, so its own recorded digest binds it. Comparing it against the source
    file's digest refused every configuration job after its run had started.
    """
    from gpuwm import remote_processed_v2 as viewer
    go_case.record["config_sha256"] = "d" * 64
    assert viewer._initialization(go_case.record) == ra._timestamp("2026-09-07T18:00:00Z") // 1000


def test_a_re_emitted_snapshot_keeps_its_forecast_progress_source_identity(go_case):
    from gpuwm.supervisor import HEARTBEAT_SCHEMA
    progress_path = go_case.output / "run-progress.json"
    go_case.manifest["progress_path"] = str(progress_path)
    go_case.manifest_path.write_bytes(encoded(go_case.manifest))
    go_case.record["config_sha256"] = "d" * 64
    progress_path.write_bytes(encoded({
        "schema": HEARTBEAT_SCHEMA, "run_id": go_case.manifest["run_id"], "pid": 123,
        "config_digest": go_case.record["snapshot_sha256"],
        "started_at_utc": go_case.manifest["started_at_utc"],
        "updated_at_utc": "2026-09-07T18:00:20Z", "status": "integrating",
        "model_elapsed_seconds": 120.0}))
    progress = ra.native_progress(go_case.record, go_case.status)["progress"]
    # The reported field names the snapshot, so it carries the snapshot digest.
    assert progress["source"]["snapshot_config_sha256"] == go_case.record["snapshot_sha256"]


def record_id(fixture):
    return fixture.record["id"]


def test_a_jobs_frames_are_read_from_the_run_folder_it_recorded(go_case):
    """C-224: the run writes into its own folder under the directory it was given."""
    run = go_case.output / "run-20260907-180000Z"
    run.mkdir()
    frame, events = run / go_case.frame.name, run / go_case.events.name
    frame.write_bytes(go_case.raw)
    events.write_bytes(encoded({**go_case.event, "path": str(frame)}))
    (run / "run-manifest.json").write_bytes(encoded({
        **go_case.manifest, "run_dir": str(run), "outputs_dir": str(run), "events_path": str(events)}))
    go_case.record["run_root"] = str(run)
    value = ra.catalog(go_case.request, go_case.tmp_path)
    assert not value["waiting"] and value["run_root"] == str(run)
    # The directory the job was given is still what it was given: the run folder
    # is the new fact beside it, not a redefinition of the old one.
    assert value["remote_output_root"] == str(go_case.output)
    assert value["frames"][0]["sha256"] == ra._file_sha(frame)
    # A job recorded before the run folder was recorded still reads its own tree.
    go_case.record.pop("run_root")
    assert ra.run_root(go_case.record) == go_case.output
    assert not ra.catalog(go_case.request, go_case.tmp_path)["waiting"]
