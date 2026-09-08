"""Protocol-only metadata/raw-byte fixtures; no synthetic weather is decoded."""
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from gpuwm import remote_artifacts as ra, remote_worker as rw


def encoded(value):
    return (json.dumps(value) + "\n").encode()


@pytest.fixture
def case(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    plan = job_dir / "plan.json"
    plan.write_text("{}")
    record = {"id": "job-fixture", "action": "start-plan", "outdir": str(output), "snapshot_plan": str(plan),
              "plan_sha256": "b" * 64, "created_at": "2026-09-07T17:59:59+00:00", "token": "a" * 64}
    status = {"id": "job-fixture", "state": "running"}
    monkeypatch.setattr(rw, "_directory", lambda *_: job_dir)
    monkeypatch.setattr(rw, "_record", lambda *_: record)
    monkeypatch.setattr(rw, "_status", lambda *_: status)
    monkeypatch.setattr(rw, "_has_token", lambda *_: True)
    events = output / "run-events.jsonl"
    manifest = {"schema": "gpuwm.run-manifest.v1", "run_id": "run-fixture", "pid": 123,
                "started_at_utc": "2026-09-07T18:00:00Z", "run_dir": str(output), "outputs_dir": str(output),
                "plan_source": str(plan), "plan_sha256": "b" * 64, "events_path": str(events),
                "provenance": {"source": "unmodified test identity"}}
    manifest_path = output / "run-manifest.json"
    manifest_path.write_bytes(encoded(manifest))
    raw = b"protocol fixture bytes only; not meteorological data"
    frame = output / "wrfout_d01_fixture"
    frame.write_bytes(raw)
    event = {"schema_version": "gpuwm.run-plan.event.v1", "sequence": 1, "event": "output_committed",
             "emitted_unix_ms": ra._timestamp("2026-09-07T18:00:01Z"), "domain": 1,
             "path": str(frame), "valid_time": "2026-09-07T18:00:00Z", "size_bytes": len(raw),
             "geometry": {"native_identity": "must remain byte-exact"}}
    events.write_bytes(encoded(event))
    request = {"schema": "gpuwm.remote.request.v1", "action": "artifacts", "workspace": str(tmp_path),
               "job": "job-fixture", "domain": 1}
    return SimpleNamespace(**locals())


def catalog(case, **kwargs):
    return ra.catalog(case.request, case.tmp_path, **kwargs)


def stream_request(case, value):
    frame = value["frames"][0]
    return {**case.request, "action": "stream-artifact", "sequence": frame["commit"]["sequence"],
            "expected_frame_sha256": frame["sha256"], "expected_commit_sha256": frame["commit"]["sha256"],
            "expected_manifest_sha256": value["run_manifest"]["sha256"]}


def test_catalog_preserves_exact_manifest_commit_geometry_and_raw_bytes(case):
    value = catalog(case)
    assert not value["waiting"]
    assert value["run_manifest"]["utf8"].encode() == case.manifest_path.read_bytes()
    frame = value["frames"][0]
    assert frame["commit"]["utf8"].encode() == case.events.read_bytes()
    assert json.loads(frame["commit"]["utf8"])["geometry"] == case.event["geometry"]
    assert frame["sha256"] == hashlib.sha256(case.raw).hexdigest()
    output = io.BytesIO()
    ra.stream(stream_request(case, value), case.tmp_path, output)
    assert output.getvalue() == case.raw


def test_only_committed_selected_domain_and_latest_sequence_are_admitted(case):
    (case.output / "wrfout_uncommitted").write_bytes(b"uncommitted")
    second = {**case.event, "sequence": 2, "domain": 2}
    latest = {**case.event, "sequence": 3, "valid_time": "2026-09-07T18:05:00Z"}
    case.events.write_bytes(encoded(case.event) + encoded(second) + encoded(latest))
    value = catalog(case)
    assert value["available_domains"] == [1, 2]
    assert len(value["frames"]) == 1 and value["frames"][0]["commit"]["sequence"] == 3
    assert catalog(case, sequence=1)["frames"][0]["commit"]["sequence"] == 1
    case.request["domain"] = 3
    assert catalog(case)["waiting"]


def test_timeline_paginates_all_committed_times_without_opening_or_hashing_wrf(case, monkeypatch):
    from datetime import timedelta
    start = ra.datetime(2026, 9, 7, 18, tzinfo=ra.timezone.utc)
    events = [{**case.event, "sequence": index + 1,
               "valid_time": (start + timedelta(minutes=15 * index)).isoformat()}
              for index in range(501)]
    case.events.write_bytes(b"".join(encoded(event) for event in events))
    case.frame.unlink()  # Discovery is only the native committed metadata.
    monkeypatch.setattr(ra, "_file_sha", lambda *_: pytest.fail("Timeline must not hash any raw frame"))
    request = {**case.request, "action": "artifact-index", "after_sequence": 0}
    first = ra.catalog(request, case.tmp_path, metadata_only=True)
    assert first["schema"] == ra.INDEX_SCHEMA and first["domain"] == 1 and not first["waiting"]
    assert len(first["entries"]) == 256 and first["next_after_sequence"] == 256
    assert first["latest_sequence"] == 501 and "frames" not in first
    request["after_sequence"] = first["next_after_sequence"]
    second = ra.catalog(request, case.tmp_path, metadata_only=True)
    assert len(second["entries"]) == 245 and second["next_after_sequence"] is None
    assert [e["sequence"] for e in first["entries"] + second["entries"]] == list(range(1, 502))
    assert second["entries"][-1]["valid_time"] == events[-1]["valid_time"]
    request["after_sequence"] = 501
    last = ra.catalog(request, case.tmp_path, metadata_only=True)
    assert last["entries"] == [] and not last["waiting"] and last["latest_sequence"] == 501


def test_explicit_sequence_transfers_that_exact_native_time_instead_of_latest(case):
    newest = {**case.event, "sequence": 2, "valid_time": "2026-09-07T19:00:00Z"}
    case.events.write_bytes(encoded(case.event) + encoded(newest))
    case.request["sequence"] = 1
    selected = catalog(case)
    assert selected["frames"][0]["commit"]["sequence"] == 1
    assert selected["frames"][0]["valid_time"] == case.event["valid_time"]
    output = io.BytesIO()
    ra.stream(stream_request(case, selected), case.tmp_path, output)
    assert output.getvalue() == case.raw
    case.request["sequence"] = 3
    assert catalog(case)["waiting"]


@pytest.mark.parametrize("value", [None, 0, -1, True, 1.5, "2", 1 << 63])
def test_explicit_frame_sequence_is_a_bounded_positive_integer(case, value):
    case.request["sequence"] = value
    with pytest.raises(ValueError, match="sequence"):
        catalog(case)


def test_native_hosted_timeline_uses_the_same_exact_producer_proof(hosted):
    h = hosted
    value = ra.catalog({**h.case.request, "action": "artifact-index"}, h.case.tmp_path, metadata_only=True)
    assert value["run_id"] == h.manifest["run_id"] and value["entries"] == [
        {"sequence": h.commit["sequence"], "domain": 1, "valid_time": h.commit["valid_time"]}]
    assert value["producer_binding"]["producer_resolved"]["utf8"].encode() == encoded(h.producer_resolved)


def test_live_partial_tail_waits_but_terminal_partial_and_bad_complete_record_fail(case):
    case.events.write_bytes(encoded(case.event) + b'{"incomplete"')
    assert len(catalog(case)["frames"]) == 1
    case.status["state"] = "completed"
    with pytest.raises(ValueError, match="incomplete final"):
        catalog(case)
    case.status["state"] = "running"
    case.events.write_bytes(encoded(case.event) + b'{"incomplete"\n')
    with pytest.raises(ValueError):
        catalog(case)
    case.events.write_bytes(encoded(case.event) * 2)
    with pytest.raises(ValueError, match="sequence"):
        catalog(case)


@pytest.mark.parametrize("key,value", [("plan_sha256", "c" * 64), ("plan_source", "/other-plan.json"),
                                      ("run_dir", "/other-run"), ("pid", 0),
                                      ("started_at_utc", "2026-09-07T17:00:00Z")])
def test_foreign_manifest_plan_path_process_or_time_is_refused(case, key, value):
    case.manifest[key] = value
    case.manifest_path.write_bytes(encoded(case.manifest))
    with pytest.raises(ValueError, match="does not match"):
        catalog(case)


def test_active_process_ownership_and_unresolved_jobs_are_refused(case, monkeypatch):
    monkeypatch.setattr(rw, "_has_token", lambda *_: False)
    with pytest.raises(ValueError, match="belongs"):
        catalog(case)
    case.status["state"] = "lost"
    with pytest.raises(ValueError, match="ownership"):
        catalog(case)


def test_previous_run_commits_and_missing_manifest_do_not_expose_files(case):
    case.event["emitted_unix_ms"] -= 10_000
    case.events.write_bytes(encoded(case.event))
    assert catalog(case)["waiting"]
    case.manifest_path.unlink()
    assert catalog(case)["waiting"]


def test_native_status_reads_bound_progress_and_plain_failure_without_raw_frames(case, monkeypatch):
    from gpuwm.supervisor import HEARTBEAT_SCHEMA
    progress_path = case.output / "run-progress.json"
    case.record["config_sha256"] = "d" * 64
    case.manifest["progress_path"] = str(progress_path)
    case.manifest_path.write_bytes(encoded(case.manifest))
    progress = {"schema": HEARTBEAT_SCHEMA, "run_id": case.manifest["run_id"], "pid": 123,
        "config_digest": "d" * 64, "started_at_utc": case.manifest["started_at_utc"],
        "updated_at_utc": "2026-09-07T18:00:20Z", "status": "integrating", "model_elapsed_seconds": 120.0}
    progress_path.write_bytes(encoded(progress))
    failure = {"schema_version": "gpuwm.run-plan.event.v1", "sequence": 2, "event": "failed",
        "emitted_unix_ms": ra._timestamp("2026-09-07T18:00:21Z"), "stage": "fetch", "message": "CDS credentials are unavailable.\nConfigure this node."}
    case.events.write_bytes(encoded(case.event) + encoded(failure))
    monkeypatch.setattr(ra, "_file_sha", lambda *_: pytest.fail("status must not hash meteorological payloads"))
    summary = ra.native_progress(case.record, case.status)
    assert summary == {"stage": "fetch", "model_elapsed_seconds": 120.0, "valid_time": case.event["valid_time"],
                       "phase": "integrating", "phase_updated_unix_ms": ra._timestamp("2026-09-07T18:00:20Z"),
                       "error": "CDS credentials are unavailable. Configure this node."}
    progress["pid"] = 999
    progress_path.write_bytes(encoded(progress))
    assert "model_elapsed_seconds" not in ra.native_progress(case.record, case.status)
    case.manifest["plan_sha256"] = "wrong"
    case.manifest_path.write_bytes(encoded(case.manifest))
    with pytest.raises(ValueError, match="does not match"):
        ra.native_progress(case.record, case.status)


def test_current_native_heartbeat_phase_is_not_replaced_by_an_older_stage_label(case, monkeypatch):
    from gpuwm.supervisor import HEARTBEAT_SCHEMA
    progress_path=case.output/"run-progress.json"
    case.record["config_sha256"]="d"*64
    case.manifest["progress_path"]=str(progress_path)
    case.manifest_path.write_bytes(encoded(case.manifest))
    heartbeat={"schema":HEARTBEAT_SCHEMA,"run_id":case.manifest["run_id"],"pid":123,
        "config_digest":"d"*64,"started_at_utc":case.manifest["started_at_utc"],
        "updated_at_utc":"2026-09-07T18:00:20Z","status":"preparing:prepare-case","model_elapsed_seconds":0.}
    progress_path.write_bytes(encoded(heartbeat))
    stage={"schema_version":"gpuwm.run-plan.event.v1","sequence":2,"event":"stage_started",
        "emitted_unix_ms":ra._timestamp("2026-09-07T18:00:10Z"),"stage":"finalize","phase":"render"}
    case.events.write_bytes(encoded(case.event)+encoded(stage))
    monkeypatch.setattr(ra,"_file_sha",lambda *_:pytest.fail("phase reporting must not inspect weather arrays"))
    result=ra.native_progress(case.record,case.status)
    assert result["phase"]=="preparing:prepare-case"
    assert result["phase_updated_unix_ms"]==ra._timestamp(heartbeat["updated_at_utc"])
    stage["emitted_unix_ms"]=ra._timestamp("2026-09-07T18:00:21Z")
    case.events.write_bytes(encoded(case.event)+encoded(stage))
    result=ra.native_progress(case.record,case.status)
    assert result["phase"]=="render" and result["phase_updated_unix_ms"]==stage["emitted_unix_ms"]


@pytest.mark.parametrize("nested", [False, True])
def test_native_status_forwards_exact_bounded_render_families_and_skip_reasons(case, monkeypatch, nested):
    summary = {"schema":"gpuwm.render-summary.v1", "requested_specs":["qpf_1h", "10m_wind_speed_and_direction"],
        "rendered_png_count":1, "rendered_family_count":1, "rendered_families":[{"name":"10m_wind_speed_and_direction", "count":1}],
        "skipped_count":1, "skipped_family_count":1,
        "skipped_families":[{"name":"qpf_1h", "count":1, "reasons":["Previous committed frame is unavailable — no hourly difference was made."], "additional_reasons":0}],
        "failure_count":0, "failures":[], "additional_failures":0, "invocation_count":1}
    event = {"schema_version":"gpuwm.run-plan.event.v1", "sequence":2, "event":"completed" if nested else "stage_finished",
             "stage":"finalize", "emitted_unix_ms":ra._timestamp("2026-09-07T18:00:02Z")}
    event.update({"summary":{"render_summary":summary}} if nested else {"render_summary":summary})
    case.events.write_bytes(encoded(case.event) + encoded(event))
    monkeypatch.setattr(ra, "_file_sha", lambda *_:pytest.fail("Render summary must not hash raw WRF files"))
    result = ra.native_progress(case.record, case.status)
    assert result["render_summary"] == summary
    assert result["stage"] == ("completed" if nested else "finalize")
    event["summary"] = {"render_summary":{"schema":"gpuwm.render-summary.v1", "oversized":"x" * (65 * 1024)}}
    event.pop("render_summary", None)
    case.events.write_bytes(encoded(case.event) + encoded(event))
    result = ra.native_progress(case.record, case.status)
    assert "render_summary" not in result and result["valid_time"] == case.event["valid_time"]


@pytest.fixture
def hosted(case):
    """Actual hosted-run receipt shape, carrying protocol-only frame bytes."""
    config = case.job_dir / "case.toml"
    config.write_bytes(b'[experiment]\nstart_time="2026-09-07T18:00:00"\nrun_seconds=21600\nrestart_interval_s=3600\n[[domain]]\ngrid_id=1\nhistory_interval_s=900\n')
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    case.record.update(snapshot_config=str(config), snapshot_sha256=digest, config_sha256=digest)
    case.manifest["route"] = "prepared"
    case.manifest_path.write_bytes(encoded(case.manifest))
    chain = case.output / "chain"
    chain.mkdir()
    producer = chain / "run-20260907-180001Z_i201305311200Z"
    producer.mkdir()
    pointer = chain / "latest-run.txt"
    pointer.write_bytes((producer.name + "\n").encode())
    parent_resolved = {"schema_version": "gpuwm.run-plan.event.v1", "sequence": 1,
        "event": "resolved_plan", "emitted_unix_ms": ra._timestamp("2026-09-07T18:00:00Z"),
        "config_source": str(config), "config_sha256": digest}
    case.events.write_bytes(encoded(parent_resolved))
    producer_events = producer / "events.jsonl"
    producer_resolved = {**parent_resolved, "emitted_unix_ms": ra._timestamp("2026-09-07T18:00:01Z")}
    frame = producer / case.frame.name
    frame.write_bytes(case.raw)
    commit = {**case.event, "sequence": 2, "path": str(frame)}
    producer_events.write_bytes(encoded(producer_resolved) + encoded(commit))
    manifest = {**case.manifest, "run_id": "inner-run", "route": "experiment",
        "run_dir": str(producer), "outputs_dir": str(producer), "events_path": str(producer_events),
        "plan_source": "gpuwm go " + str(config), "plan_sha256": "e" * 64,
        "started_at_utc": "2026-09-07T18:00:01Z"}
    manifest_path = producer / "run-manifest.json"
    manifest_path.write_bytes(encoded(manifest))
    return SimpleNamespace(**locals())


def test_native_pointer_attaches_exact_inner_manifest_and_commit_with_config_proof(hosted):
    h = hosted
    value = catalog(h.case)
    assert not value["waiting"] and value["remote_output_root"] == str(h.case.output)
    assert value["run_id"] == "inner-run"
    assert value["run_manifest"]["utf8"].encode() == h.manifest_path.read_bytes()
    binding = value["producer_binding"]
    assert binding["schema"] == "gpuwm.remote-producer-binding.v1"
    assert binding["parent_manifest"]["utf8"].encode() == h.case.manifest_path.read_bytes()
    assert binding["chain_pointer"]["utf8"].encode() == h.pointer.read_bytes()
    assert binding["parent_resolved"]["utf8"].encode() == h.case.events.read_bytes()
    assert binding["producer_resolved"]["utf8"].encode() == encoded(h.producer_resolved)
    assert value["frames"][0]["commit"]["utf8"].encode() == encoded(h.commit)
    output = io.BytesIO()
    ra.stream(stream_request(h.case, value), h.case.tmp_path, output)
    assert output.getvalue() == h.case.raw
    assert ra.native_progress(h.case.record, h.case.status)["valid_time"] == h.commit["valid_time"]


def test_hosted_pointer_is_required_and_never_falls_back_to_a_directory_glob(hosted):
    hosted.pointer.unlink()
    assert catalog(hosted.case)["waiting"]
    assert "producer_binding" not in catalog(hosted.case)


@pytest.mark.parametrize("value", ["../outside", "/absolute", "run-not-a-native-stamp", "x" * 257])
def test_hosted_pointer_must_name_one_native_owned_run(hosted, value):
    hosted.pointer.write_text(value)
    with pytest.raises(ValueError, match="pointer"):
        catalog(hosted.case)


@pytest.mark.parametrize("key,value", [("pid", 999), ("plan_source", "gpuwm go /another/config"),
    ("started_at_utc", "2026-09-07T17:00:00Z"), ("outputs_dir", "/another/run"),
    ("plan_sha256", "bad"), ("run_id", "run-fixture")])
def test_hosted_producer_process_config_time_and_output_linkage_is_required(hosted, key, value):
    hosted.manifest[key] = value
    hosted.manifest_path.write_bytes(encoded(hosted.manifest))
    with pytest.raises(ValueError, match="does not match"):
        catalog(hosted.case)


@pytest.mark.parametrize("producer,key,value", [(False, "config_sha256", "f" * 64),
    (True, "config_source", "/other/config"), (True, "config_sha256", "a" * 64)])
def test_both_native_resolved_config_receipts_must_match_saved_bytes(hosted, producer, key, value):
    h = hosted
    resolved = h.producer_resolved if producer else h.parent_resolved
    resolved[key] = value
    path = h.producer_events if producer else h.case.events
    path.write_bytes(encoded(resolved) + (encoded(h.commit) if producer else b""))
    with pytest.raises(ValueError, match="configuration receipts disagree"):
        catalog(h.case)


def test_changed_saved_config_and_after_end_producer_are_refused(hosted):
    h = hosted
    h.case.status.update(state="completed", ended_at="2026-09-07T18:00:00Z")
    with pytest.raises(ValueError, match="does not match"):
        catalog(h.case)
    h.case.status["ended_at"] = "2026-09-07T18:00:10Z"
    h.config.write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        catalog(h.case)


def test_paths_outside_run_or_changed_commit_size_are_refused(case):
    outside = case.tmp_path / "outside.wrf"
    outside.write_bytes(case.raw)
    case.event["path"] = str(outside)
    case.events.write_bytes(encoded(case.event))
    with pytest.raises(ValueError, match="outside"):
        catalog(case)
    case.event["path"] = str(case.frame)
    case.events.write_bytes(encoded(case.event))
    case.frame.write_bytes(case.raw + b"changed")
    with pytest.raises(ValueError, match="length changed"):
        catalog(case)


def test_hash_change_after_discovery_cannot_stream_under_old_binding(case):
    value = catalog(case)
    case.frame.write_bytes(b"X" * len(case.raw))
    output = io.BytesIO()
    with pytest.raises(ValueError, match="authority changed"):
        ra.stream(stream_request(case, value), case.tmp_path, output)
    assert output.getvalue() == b""


@pytest.mark.parametrize("key,value", [("path", "/arbitrary"), ("sequence", 0), ("domain", True),
                                      ("expected_frame_sha256", "bad")])
def test_stream_rejects_untyped_or_arbitrary_selectors(case, key, value):
    request = stream_request(case, catalog(case))
    request[key] = value
    with pytest.raises(ValueError):
        ra.stream(request, case.tmp_path, io.BytesIO())


def test_cache_reuses_exact_object_and_new_commit_keeps_source_identity(case, monkeypatch):
    from gpuwm import remote_cli
    value = catalog(case)
    monkeypatch.setattr(remote_cli, "_transport", lambda *_args, **_kwargs: {"ok": True, "artifacts": copy.deepcopy(value)})
    transfers = []
    def download(_command, request, path, frame):
        transfers.append(request)
        path.write_bytes(case.raw)
    monkeypatch.setattr(ra, "_download", download)
    args = SimpleNamespace(domain=1, workspace=str(case.tmp_path), job="job-fixture", cache_root=str(case.tmp_path / "cache"))
    first = ra.sync(args, [], [])
    again = ra.sync(args, [], [])
    assert first["transferred_bytes"] == len(case.raw) and again["transferred_bytes"] == 0
    assert len(transfers) == 1
    assert first["artifacts"]["frames"][0]["remote_path"] == str(case.frame)
    assert Path(first["artifacts"]["frames"][0]["path"]).read_bytes() == case.raw
    value["frames"][0]["commit"]["sequence"] = 2
    assert ra.sync(args, [], [])["artifacts"]["frames"][0]["commit"]["sequence"] == 2
    assert len(transfers) == 1


def test_retained_corrupt_frame_returns_typed_recovery_without_an_admittable_receipt(case, monkeypatch):
    from gpuwm import remote_cli, remote_artifact_cache
    value = catalog(case)
    monkeypatch.setattr(remote_cli, "_transport", lambda *_args, **_kwargs: {"ok": True, "artifacts": copy.deepcopy(value)})
    def retained(*_args, **_kwargs):
        raise remote_artifact_cache.CacheRecovery(value["frames"][0]["sha256"])
    monkeypatch.setattr(remote_artifact_cache.Cache, "obtain", retained)
    args = SimpleNamespace(domain=1, workspace=str(case.tmp_path), job="job-fixture", reader_leases=True,
                           cache_root=str(case.tmp_path / "cache"))
    result = ra.sync(args, [], [])
    assert result["artifacts"]["waiting"] and result["artifacts"]["frames"] == [] and result["transferred_bytes"] == 0
    assert result["cache_recovery"] == {"schema": "arwen.artifact-cache-recovery.v1",
        "sha256": value["frames"][0]["sha256"], "reason": "corrupt_retained_object"}


def test_index_cli_and_worker_dispatch_keep_the_read_only_typed_selector(case, monkeypatch):
    from argparse import ArgumentParser
    from gpuwm import remote_cli
    parser = ArgumentParser()
    remote_cli.register_cli(parser.add_subparsers())
    args = parser.parse_args(["remote", "artifact-index", "--host", "fixture", "--python", "/bin/python",
        "--workspace", str(case.tmp_path), "--job", "job-fixture", "--domain", "1", "--after-sequence", "0"])
    monkeypatch.setattr(rw.sys, "platform", "linux")
    monkeypatch.setattr(rw, "_workspace", lambda *_: case.tmp_path)
    seen = []
    def transport(_command, request, **_kwargs):
        seen.append(request)
        return {"ok": True, **rw.dispatch(request)}
    monkeypatch.setattr(remote_cli, "_transport", transport)
    result = ra.index(args, [])
    assert result["artifact_index"]["entries"][0]["sequence"] == 1
    assert seen == [{"schema":"gpuwm.remote.request.v1", "action":"artifact-index", "workspace":str(case.tmp_path),
        "job":"job-fixture", "domain":1, "after_sequence":0}]
    selected = parser.parse_args(["remote", "sync-artifacts", "--host", "fixture", "--python", "/bin/python",
        "--workspace", str(case.tmp_path), "--job", "job-fixture", "--cache-root", str(case.tmp_path / "cache"),
        "--sequence", "1", "--reader-leases"])
    assert selected.sequence == 1 and selected.reader_leases


def test_bounded_binary_transfer_is_atomic_and_bad_stream_never_publishes(tmp_path):
    raw = b"exact protocol bytes" * 100
    frame = {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    path = tmp_path / "object.wrf"
    script = "import sys;sys.stdin.buffer.read();sys.stdout.buffer.write(bytes.fromhex(sys.argv[1]))"
    ra._download([sys.executable, "-c", script, raw.hex()], {}, path, frame, timeout=10)
    assert path.read_bytes() == raw
    bad = tmp_path / "bad.wrf"
    with pytest.raises(ValueError, match="SHA-256"):
        ra._download([sys.executable, "-c", script, b"wrong".hex()], {}, bad, frame, timeout=10)
    assert not bad.exists() and not list(tmp_path.glob("*.part"))


def test_stream_overflow_and_timeout_leave_no_published_object(tmp_path):
    frame = {"size_bytes": 1, "sha256": hashlib.sha256(b"X").hexdigest()}
    path = tmp_path / "object.wrf"
    with pytest.raises(ValueError, match="exceeds"):
        ra._download([sys.executable, "-c", "import sys;sys.stdout.buffer.write(b'X'*2)"], {}, path, frame, timeout=10)
    with pytest.raises(ValueError, match="timed out"):
        ra._download([sys.executable, "-c", "import time;time.sleep(10)"], {}, path, frame, timeout=.1)
    assert not path.exists() and not list(tmp_path.glob("*.part"))


@pytest.fixture
def timed_case(case):
    config = case.job_dir / "case.toml"
    config.write_text('[experiment]\nstart_time=2013-05-20T00:00:00\nrun_seconds=21600\nrestart_interval_s=3600\n'
                      '[[domain]]\ngrid_id=1\nhistory_interval_s=3600\n'
                      '[[domain]]\ngrid_id=2\nhistory_interval_s=900\n'
                      '[[domain]]\ngrid_id=3\nhistory_interval_s=900\n', encoding="utf-8")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    case.record.update(snapshot_config=str(config), config_sha256=digest, snapshot_sha256=digest)
    checkpoint = case.output / "gpuwmrst_d01_2013-05-20_04_00_00__owned-set.npz"
    checkpoint.write_bytes(b"protocol-only checkpoint placeholder; must never be opened")
    model = {"schema_version":"gpuwm.run-plan.event.v1", "sequence":4, "event":"model_progress",
             "emitted_unix_ms":ra._timestamp("2026-09-07T18:06:00Z"), "domain":1,
             "outer_step":256, "model_seconds":15360., "wall_seconds":352.633586,
             "speed_x":43.558, "step_ms":949.312, "phase":"post-d01-sync",
             "domains":[{"domain":domain,"model_seconds":15360.} for domain in (1,2,3)],
             "last_checkpoint":str(checkpoint)}
    commits = [{**case.event, "sequence":domain, "domain":domain,
                "valid_time":"2013-05-20T04:00:00" if domain==1 else "2013-05-20T04:15:00"}
               for domain in (1,2,3)]
    case.events.write_bytes(b"".join(encoded(event) for event in [*commits, model]))
    return SimpleNamespace(**locals())


def test_forecast_progress_preserves_native_speed_clock_and_distinct_save_schedules(timed_case, monkeypatch):
    t = timed_case
    real_open = Path.open
    def no_weather_reads(path, *args, **kwargs):
        assert path not in (t.case.frame, t.checkpoint), "Progress must not read WRF or checkpoint arrays"
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_weather_reads)
    progress = ra.native_progress(t.case.record, t.case.status)["progress"]
    assert progress["schema"] == "arwen.forecast-progress.v1"
    for key in ("outer_step", "model_seconds", "wall_seconds", "speed_x", "step_ms"):
        assert progress[key] == t.model[key]
    assert progress["run_seconds"] == 21600
    assert progress["valid_time"] == "2013-05-20T04:16:00Z"
    assert progress["updated_unix_ms"] == t.model["emitted_unix_ms"]
    assert progress["source"]["event_sequence"] == 4
    assert progress["source"]["snapshot_config_sha256"] == t.digest
    assert [row["last_save_model_seconds"] for row in progress["domains"]] == [14400, 15300, 15300]
    assert [row["next_save_in_seconds"] for row in progress["domains"]] == [2640, 840, 840]
    assert progress["checkpoint"] == {"interval_seconds":3600, "last_saved_model_seconds":14400,
                                       "next_model_seconds":18000, "in_seconds":2640}


def test_missing_child_clocks_and_unknown_native_rates_are_not_invented(timed_case):
    t=timed_case
    t.model.pop("domains")
    t.model.update(wall_seconds=float("nan"), speed_x=None, step_ms=0)
    t.case.events.write_bytes(encoded(t.model))
    progress=ra.native_progress(t.case.record,t.case.status)["progress"]
    assert progress["wall_seconds"] is None and progress["speed_x"] is None and progress["step_ms"] is None
    assert progress["domains"][0]["model_seconds"] == 15360
    assert progress["domains"][1]["model_seconds"] is None
    assert progress["domains"][1]["next_save_model_seconds"] is None
    assert progress["domains"][0]["last_save_model_seconds"] is None


def test_disabled_checkpoint_and_completed_forecast_do_not_promise_future_writes(timed_case):
    t=timed_case
    t.config.write_text(t.config.read_text().replace("restart_interval_s=3600", "restart_interval_s=0"))
    digest=hashlib.sha256(t.config.read_bytes()).hexdigest()
    t.case.record.update(config_sha256=digest,snapshot_sha256=digest)
    progress=ra.native_progress(t.case.record,t.case.status)["progress"]
    assert progress["checkpoint"]["interval_seconds"] == 0
    assert progress["checkpoint"]["next_model_seconds"] is None
    t.model["model_seconds"]=21600
    for row in t.model["domains"]: row["model_seconds"]=21600
    t.case.events.write_bytes(encoded(t.model))
    progress=ra.native_progress(t.case.record,t.case.status)["progress"]
    assert all(row["next_save_model_seconds"] is None for row in progress["domains"])
    assert progress["checkpoint"]["in_seconds"] is None


def test_progress_rejects_changed_configuration_and_backward_native_clocks(timed_case):
    t=timed_case
    older={**t.model,"sequence":5,"model_seconds":15000}
    t.case.events.write_bytes(encoded(t.model)+encoded(older))
    with pytest.raises(ValueError,match="clock moved backward"):
        ra.native_progress(t.case.record,t.case.status)
    t.case.events.write_bytes(encoded(t.model))
    t.config.write_text(t.config.read_text()+"# changed after launch\n")
    with pytest.raises(ValueError,match="saved configuration"):
        ra.native_progress(t.case.record,t.case.status)


def test_pipeline_progress_reports_native_transfer_counts_and_preparation_without_model_steps(timed_case, monkeypatch):
    t=timed_case;start=ra._timestamp("2026-09-07T18:00:00Z")
    def event(sequence,tag,**fields):
        return {"schema_version":"gpuwm.run-plan.event.v1","sequence":sequence,"event":tag,"emitted_unix_ms":start+sequence*1000,**fields}
    values=[event(1,"stage_started",stage="fetch"),
            event(2,"fetch_progress",acquisition={"schema":"arwen.acquisition-progress.v1","source":"era5","provider":"cds","phase":"cds_running","requests_total":2,"requests_completed":1,"files_total":2,"forcing_hours":3,"forcing_times_total":4}),
            event(3,"fetch_completed",file="part-0.grib",bytes=4*1024**2,failed=False),
            event(4,"fetch_progress",file="part-1.grib",bytes=2*1024**2)]
    monkeypatch.setattr(ra.time,"time",lambda:(start+60000)/1000)
    t.case.events.write_bytes(b"".join(encoded(v) for v in values))
    result=ra.native_progress(t.case.record,t.case.status)
    assert result["progress"]["outer_step"] is None
    pipe=result["pipeline_progress"]
    assert pipe["stage"]=="fetch" and pipe["phase"]=="cds_running" and pipe["wall_seconds"]==59
    assert pipe["acquisition"]["requests_completed"]==1
    assert pipe["acquisition"]["files_completed"]==1
    assert pipe["acquisition"]["transferred_bytes"]==6*1024**2
    assert pipe["acquisition"]["expected_bytes"] is None
    prep={"schema":"gpuwm.prep-stage.v1","stage":"static_fields","label":"Preparing domain geography","event":"started","index":2,"count":3,"backend":"native"}
    values.extend([event(5,"stage_started",stage="prepare"),event(6,"warning",code="preparation_progress",phase="prepare",preparation=prep)])
    t.case.events.write_bytes(b"".join(encoded(v) for v in values))
    result=ra.native_progress(t.case.record,t.case.status)
    assert result["pipeline_progress"]["preparation"]==prep
    assert result["pipeline_progress"]["phase"]=="Preparing domain geography"
