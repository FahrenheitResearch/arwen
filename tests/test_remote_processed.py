"""Native processor orchestration/transport only; fixtures contain no weather."""
import io
import json
from pathlib import Path
import shutil
import tarfile
from types import SimpleNamespace

import pytest

from gpuwm import remote_processed as rp, remote_artifacts as ra, remote_cli, render
from test_remote_artifacts import case, encoded


@pytest.fixture
def native(case, monkeypatch):
    calls = []
    monkeypatch.setattr(rp, "ensure", lambda *args, **kwargs: None)
    monkeypatch.setattr(rp, "_processor_identity", lambda: {"path": "fixture-native-processor", "stamp": [1, 2, 3, 4, 5]})
    monkeypatch.setattr(render, "require_renderer", lambda: Path("fixture-native-processor"))
    def run(command, **kwargs):
        request = json.loads(Path(command[command.index("--process-request") + 1]).read_text())
        calls.append(request)
        root = Path(request["store_root"])
        store = root / "identity/rw-store"
        directory = store / "wrf/native-run"
        directory.mkdir(parents=True, exist_ok=True)
        hour, grid, manifest, receipt = (directory / name for name in ("f000.rws", "grid.rwg", "run.json", "companion-frame.json"))
        hour.write_bytes(b"protocol-only immutable native store fixture")
        grid.write_bytes(b"protocol-only native grid fixture")
        frame = {"schema": "arwen.processed-frame.v1", "id": "native-fixture",
                 "identity": {"case_id": request["case_id"], "source": "arwen", "model": "wrf-" + request["domain"],
                    "member": None, "valid_unix": ra._timestamp(request["valid_utc"]) // 1000,
                    "source_sha256": request["source_sha256"], "lead_seconds": 0},
                 "store_root": str(store), "hour_path": str(hour), "model_slug": "wrf", "run_slug": "native-run",
                 "storage_slot": 0, "rws_sha256": ra._file_sha(hour), "rws_bytes": hour.stat().st_size}
        manifest.write_bytes(encoded({"field_manifest": "native-only", "grid_path": str(grid)}))
        receipt.write_bytes(encoded(frame))
        result = {"schema": "arwen.wrf-process-result.v1", "frame": frame, "domain": request["domain"],
                  "grid_path": str(grid), "run_json_path": str(manifest), "receipt_path": str(receipt), "cache_hit": False,
                  "files": [{"path": str(path), "sha256": ra._file_sha(path), "bytes": path.stat().st_size}
                            for path in (hour, grid, manifest, receipt)]}
        Path(command[command.index("--process-result") + 1]).write_bytes(encoded(result))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(rp.subprocess, "run", run)
    return SimpleNamespace(case=case, calls=calls)


def request(case, **fields):
    return {**case.request, "action": "processed-frame", **fields}


def test_queue_processes_every_committed_domain_time_once_and_catches_later_outputs(native):
    c = native.case
    second = {**c.event, "sequence": 2, "domain": 2}
    c.events.write_bytes(encoded(c.event) + encoded(second))
    assert rp._work_job(c.tmp_path, c.record["id"]) is False
    assert [value["domain"] for value in native.calls] == ["d01", "d02"]
    assert all(value["case_id"] == c.manifest["run_id"] for value in native.calls)
    assert all(value["heavy_ecape"] is False for value in native.calls)
    assert rp._work_job(c.tmp_path, c.record["id"]) is False
    assert len(native.calls) == 2
    third = {**c.event, "sequence": 3, "valid_time": "2026-09-07T19:00:00Z"}
    c.events.write_bytes(encoded(c.event) + encoded(second) + encoded(third))
    c.status["state"] = "completed"
    assert rp._work_job(c.tmp_path, c.record["id"]) is True
    assert len(native.calls) == 3
    status = rp.index_metadata(c.tmp_path, c.record["id"])
    assert status["ready"] == 3 and status["done"] is True and status["failed"] == 0
    assert c.frame.read_bytes() == c.raw


def test_ready_catalog_retains_exact_manifest_commit_and_never_rereads_raw_data(native, monkeypatch):
    c = native.case
    rp._work_job(c.tmp_path, c.record["id"])
    monkeypatch.setattr(ra, "_file_sha", lambda *args: pytest.fail("metadata request reopened an already processed WRF"))
    value = rp.catalog(request(c), c.tmp_path)
    assert value["waiting"] is False
    assert value["run_manifest"]["utf8"].encode() == c.manifest_path.read_bytes()
    assert value["commit"]["utf8"].encode() == c.events.read_bytes()
    assert value["run_id"] == c.manifest["run_id"]
    assert value["frame"]["identity"]["case_id"] == c.manifest["run_id"]
    c.frame.write_bytes(c.raw + b"changed")
    with pytest.raises(ValueError, match="changed after"):
        rp.catalog(request(c), c.tmp_path)


def test_store_archive_roundtrip_rebases_metadata_and_reuses_immutable_local_cache(native, monkeypatch):
    c = native.case
    rp._work_job(c.tmp_path, c.record["id"])
    expected = rp.catalog(request(c), c.tmp_path)
    def transport(command, req, **kwargs):
        return {"ok": True, "processed_frame": rp.catalog(req, c.tmp_path)}
    transfers = []
    def transfer(command, req, destination, frame, **kwargs):
        payload = io.BytesIO()
        rp.stream(req, c.tmp_path, payload)
        assert len(payload.getvalue()) == frame["size_bytes"]
        destination.write_bytes(payload.getvalue())
        transfers.append(frame["sha256"])
    monkeypatch.setattr(remote_cli, "_transport", transport)
    monkeypatch.setattr(ra, "_download", transfer)
    options = SimpleNamespace(workspace=str(c.tmp_path), job=c.record["id"], domain=1, sequence=1,
                              cache_root=str(c.tmp_path / "local-cache"))
    reply = rp.sync(options, ["unused-rpc"], ["unused-stream"])
    value = reply["processed_frame"]
    result_path = Path(value["local_result_path"])
    result = json.loads(result_path.read_text())
    assert result_path.is_relative_to(rp._local_cache_path(options.cache_root))
    assert result["remote_source"]["commit"] == expected["commit"]
    assert result["remote_source"]["source_sha256"] == expected["source_sha256"]
    assert result["frame"]["identity"] == expected["frame"]["identity"]
    assert Path(result["frame"]["hour_path"]).is_file()
    assert Path(result["frame"]["store_root"]).is_relative_to(result_path.parent)
    assert json.loads(Path(result["receipt_path"]).read_text()) == result["frame"]
    for row in result["files"]:
        path = Path(row["path"])
        assert path.stat().st_size == row["bytes"] and ra._file_sha(path) == row["sha256"]
    assert rp.sync(options, ["unused-rpc"], ["unused-stream"])["transferred_bytes"] == 0
    assert len(transfers) == 1


def test_native_failure_is_visible_and_does_not_repeat_or_change_original(native, monkeypatch):
    c = native.case
    monkeypatch.setattr(rp.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=9))
    rp._work_job(c.tmp_path, c.record["id"])
    value = rp.catalog(request(c), c.tmp_path)
    assert value["waiting"] and value["processing"]["state"] == "failed"
    assert "exited 9" in value["processing"]["error"]
    monkeypatch.setattr(rp.subprocess, "run", lambda *args, **kwargs: pytest.fail("failed native operation retried without a new identity"))
    rp._work_job(c.tmp_path, c.record["id"])
    assert c.frame.read_bytes() == c.raw


def test_unsafe_archive_member_is_rejected_before_any_extraction(tmp_path):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as output:
        member = tarfile.TarInfo("../outside"); member.size = 1
        output.addfile(member, io.BytesIO(b"x"))
    destination = tmp_path / "cache"
    destination.mkdir()
    with pytest.raises(ValueError, match="unsafe member"):
        rp._unpack(archive, destination, {})
    assert not (tmp_path / "outside").exists()


def test_native_process_identity_rejects_wrong_run_domain_time_or_source(native):
    c = native.case
    rp._work_job(c.tmp_path, c.record["id"])
    value = rp.catalog(request(c), c.tmp_path)
    result = {"frame": value["frame"], "domain": "d01"}
    for field, wrong in (("case_id", "other-run"), ("model", "wrf-d02"), ("valid_unix", 0), ("source_sha256", "a" * 64)):
        bad = json.loads(json.dumps(result)); bad["frame"]["identity"][field] = wrong
        with pytest.raises(ValueError, match="does not match"):
            rp._check_identity(bad, value["run_id"], c.event, value["source_sha256"])


def test_newly_selected_time_is_next_in_single_worker_queue(case, monkeypatch):
    c = case
    c.events.write_bytes(b"".join(encoded({**c.event, "sequence": seq}) for seq in (1, 2, 3)))
    root = rp._root(c.tmp_path)
    marker = rp._owned_directory(root / "requests") / (c.record["id"] + ".json")
    visited = []
    def convert(root, record, bound, event, authority):
        visited.append(event["sequence"])
        if len(visited) == 1:
            rp._write(marker, {"schema": rp.QUEUE_SCHEMA, "job_id": record["id"],
                               "priority": {"domain": 1, "sequence": 3}})
        return {"state": "ready", "sequence": event["sequence"], "commit_sha256": authority["sha256"]}
    monkeypatch.setattr(rp, "_convert", convert)
    monkeypatch.setattr(rp, "_processor_identity", lambda: {"path": "fixture-native-processor", "stamp": [1, 2, 3, 4, 5]})
    rp._work_job(c.tmp_path, c.record["id"])
    assert visited == [1, 3, 2]


def test_failed_conversion_can_retry_after_the_native_processor_changes(native, monkeypatch):
    c = native.case
    original = rp.subprocess.run
    monkeypatch.setattr(rp.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=9))
    rp._work_job(c.tmp_path, c.record["id"])
    monkeypatch.setattr(rp.subprocess, "run", original)
    monkeypatch.setattr(rp, "_processor_identity", lambda: {"path": "new-native-processor", "stamp": [1, 2, 3, 4, 6]})
    rp._work_job(c.tmp_path, c.record["id"])
    assert not rp.catalog(request(c), c.tmp_path)["waiting"]
