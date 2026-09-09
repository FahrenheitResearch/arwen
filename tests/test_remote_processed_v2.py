"""Compact viewer transport/ownership tests; fixture bytes are not weather."""
import copy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import remote_artifacts as ra, remote_cli, remote_processed as legacy, render
from gpuwm import remote_processed_v2 as viewer, remote_processed_cache_v2 as cache
from test_remote_artifacts import case, encoded


@pytest.fixture
def native(case, monkeypatch):
    calls = []
    monkeypatch.setattr(viewer, "_launch_worker", lambda *_: None)
    monkeypatch.setattr(legacy, "_processor_identity", lambda: {"path": "fixture-native", "stamp": [1, 2, 3, 4, 5]})
    monkeypatch.setattr(render, "require_renderer", lambda: Path("fixture-native"))
    initial = ra._timestamp(case.event["valid_time"]) // 1000
    def run(command, **kwargs):
        request = json.loads(Path(command[command.index("--process-request") + 1]).read_text())
        calls.append(request)
        store = Path(request["store_root"]) / "frame-identity" / "rw-store"
        directory = store / "wrf" / "run-fixture"
        directory.mkdir(parents=True)
        paths = [directory / name for name in ("f000.rws", "grid.rwg", "run.json", "companion-frame.json")]
        paths[0].write_bytes(b"native RWS protocol fixture, no meteorology")
        paths[1].write_bytes(b"native RWG protocol fixture, no meteorology")
        valid = ra._timestamp(request["valid_utc"]) // 1000
        frame = {"schema": "arwen.companion-store-frame.v1", "id": "fixture-frame",
                 "identity": {"case_id": request["case_id"], "source": "arwen", "model": "wrf-" + request["domain"],
                    "member": None, "valid_unix": valid, "lead_seconds": valid - initial, "source_sha256": request["source_sha256"]},
                 "store_root": str(store), "model_slug": "wrf", "run_slug": "run-fixture", "storage_slot": 0,
                 "hour_path": str(paths[0]), "grid_sha256": ra._file_sha(paths[1]), "shape": [2, 2],
                 "variables": ["temperature_2m"], "levels_hpa": [], "rws_sha256": ra._file_sha(paths[0]),
                 "rws_bytes": paths[0].stat().st_size, "processing_ms": 1.0}
        paths[2].write_bytes(encoded({"schema": "rw-store-run-v2", "grid_path": str(paths[1])}))
        paths[3].write_bytes(encoded(frame))
        rows = [{"key": key, "relative_path": path.relative_to(store).as_posix(), "path": str(path), "kind": kind,
                 "bytes": path.stat().st_size, "sha256": ra._file_sha(path), "grid_sha256": frame["grid_sha256"]}
                for key, kind, path in zip(("fields", "grid", "run", "frame"), ("rws_2d", "rwg", "metadata", "metadata"), paths)]
        result = {"schema": "arwen.wrf-process-result.v2", "profile": viewer.PROFILE, "domain": request["domain"],
                  "frame": frame, "grid_path": str(paths[1]), "run_json_path": str(paths[2]), "receipt_path": str(paths[3]),
                  "initialization_unix": initial, "cache_hit": False, "notes": [], "files": rows, "members": rows,
                  "products": [{"slug": slug, "available": True, "source_fields": ["temperature_2m"], "missing_reasons": []}
                               for slug in request.get("products", viewer.DEFAULT_PRODUCTS)]}
        if request["schema"] == "arwen.wrf-process-request.v1":
            result["schema"] = "arwen.wrf-process-result.v1"
            for key in ("profile", "initialization_unix", "products", "members"):
                result.pop(key)
            result["files"] = [{key: row[key] for key in ("path", "bytes", "sha256")} for row in rows]
        Path(command[command.index("--process-result") + 1]).write_bytes(encoded(result))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(viewer.subprocess, "run", run)
    return SimpleNamespace(case=case, calls=calls)


def query(c, **fields):
    return {**c.request, "action": "processed-frame-v2", "profile": viewer.PROFILE, "products": ["2m_temperature"],
            "expected_run_id": c.manifest["run_id"], **fields}


def test_only_selected_and_explicit_loop_frames_are_queued_in_priority_order(native):
    c = native.case
    c.events.write_bytes(b"".join(encoded({**c.event, "sequence": index, "valid_time": f"2026-09-07T18:{index:02d}:00Z"}) for index in range(1, 12)))
    waiting = viewer.catalog(query(c, sequence=4, prefetch_sequences=[5, 6]), c.tmp_path)
    assert waiting["waiting"] and native.calls == []
    queued = viewer._queue(viewer._root(c.tmp_path), c.record["id"])
    assert [row["sequence"] for row in queued] == [4, 5, 6]
    viewer._work_job(c.tmp_path, c.record["id"])
    assert len(native.calls) == 1 and native.calls[0]["valid_utc"] == "2026-09-07T18:04:00Z"
    viewer.catalog(query(c, sequence=9), c.tmp_path)
    viewer._work_job(c.tmp_path, c.record["id"])
    assert native.calls[-1]["valid_utc"] == "2026-09-07T18:09:00Z"
    viewer._work_job(c.tmp_path, c.record["id"]); viewer._work_job(c.tmp_path, c.record["id"])
    assert len(native.calls) == 4 and c.frame.read_bytes() == c.raw
    assert not list(viewer._root(c.tmp_path).rglob("*.tar"))
    ready = viewer.catalog(query(c, sequence=4), c.tmp_path, start=False)
    assert not ready["waiting"] and ready["lead_seconds"] == 240
    assert ready["initialization_unix"] == ra._timestamp(c.event["valid_time"]) // 1000


def test_compact_member_roundtrip_preserves_authority_and_rebases_lease(native, monkeypatch):
    c = native.case
    viewer.catalog(query(c, sequence=1), c.tmp_path); viewer._work_job(c.tmp_path, c.record["id"])
    monkeypatch.setattr(remote_cli, "_transport", lambda _command, request, **_kwargs: {"ok": True, "processed_frame": viewer.catalog(request, c.tmp_path)})
    transfers = []
    def download(_command, request, path, member, **_kwargs):
        stream = io.BytesIO(); viewer.stream(request, c.tmp_path, stream)
        path.write_bytes(stream.getvalue()); transfers.append(request["member_key"])
        assert path.stat().st_size == member["size_bytes"]
    monkeypatch.setattr(ra, "_download", download)
    destination = Path(os.environ.get("ARWEN_COMPACT_FIXTURE_DIR", str(c.tmp_path / "local-cache")))
    options = SimpleNamespace(workspace=str(c.tmp_path), job=c.record["id"], domain=1, sequence=1,
        cache_root=str(destination), profile=viewer.PROFILE, products="2m_temperature", reader_leases=True,
        prefetch_sequences=[], expected_run_id=c.manifest["run_id"])
    reply = cache.sync(options, [], [])
    value = reply["processed_frame"]; result = json.loads(Path(value["local_result_path"]).read_text())
    assert transfers == ["fields", "grid", "run", "frame"]
    assert value["frame"]["identity"] == result["frame"]["identity"]
    assert result["frame"]["cache_lease_path"] == result["cache_lease_path"] == value["cache_lease_path"]
    assert json.loads(Path(result["receipt_path"]).read_text()) == result["frame"]
    assert result["remote_source"]["commit"]["utf8"].encode() == c.events.read_bytes()
    assert all(Path(row["path"]).is_relative_to(Path(value["object_root"])) for row in value["local_members"])
    assert not list(destination.rglob("*.tar"))
    assert cache.sync(options, [], [])["transferred_bytes"] == 0 and len(transfers) == 4
    if os.environ.get("ARWEN_COMPACT_FIXTURE_DIR"):
        (destination / "ready-response.json").write_bytes(encoded(value))
        (destination / "request.json").write_bytes(encoded({"action": "sync_processed_frame_v2", "job_id": options.job,
            "domain": 1, "sequence": 1, "expected_run_id": options.expected_run_id, "products": ["2m_temperature"], "reader_leases": True, "prefetch_sequences": []}))


def test_compact_source_run_time_and_grid_changes_are_rejected(native):
    c = native.case
    with pytest.raises(ValueError, match="run changed"):
        viewer.catalog(query(c, expected_run_id="other-run"), c.tmp_path)
    viewer.catalog(query(c, sequence=1), c.tmp_path); viewer._work_job(c.tmp_path, c.record["id"])
    value = viewer.catalog(query(c, sequence=1), c.tmp_path, start=False)
    bad = copy.deepcopy(value["native_result"]); bad["members"][0]["grid_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="another geographic grid"):
        viewer._native_members(bad, Path(value["native_store_root"]))
    bad = copy.deepcopy(value["native_result"]); bad["initialization_unix"] += 1
    with pytest.raises(ValueError, match="initialization"):
        viewer._check_time(bad, c.record, c.event)
    c.frame.write_bytes(c.raw + b"changed")
    with pytest.raises(ValueError, match="changed after"):
        viewer.catalog(query(c, sequence=1), c.tmp_path, start=False)


def test_missing_prefetch_and_oversized_queue_fail_without_native_work(native):
    c = native.case
    with pytest.raises(ValueError, match="at most eight"):
        viewer.catalog(query(c, prefetch_sequences=list(range(1, 10))), c.tmp_path)
    with pytest.raises(ValueError, match="outside the selected"):
        viewer.catalog(query(c, prefetch_sequences=[2]), c.tmp_path)
    assert native.calls == []


def test_failed_native_processing_is_visible_and_original_history_survives(native, monkeypatch):
    c = native.case
    viewer.catalog(query(c, sequence=1), c.tmp_path)
    monkeypatch.setattr(viewer.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=9))
    viewer._work_job(c.tmp_path, c.record["id"])
    value = viewer.catalog(query(c, sequence=1), c.tmp_path)
    assert value["waiting"] and value["state"] == "failed" and "exited 9" in value["error"]
    monkeypatch.setattr(viewer.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Failed native identity retried"))
    viewer._work_job(c.tmp_path, c.record["id"])
    assert c.frame.read_bytes() == c.raw and not list(viewer._root(c.tmp_path).rglob("*.rws"))


def test_cache_evicts_only_derived_members_and_can_rederive_on_demand(native):
    c = native.case
    viewer.catalog(query(c, sequence=1), c.tmp_path); viewer._work_job(c.tmp_path, c.record["id"])
    root = viewer._root(c.tmp_path); first = viewer.catalog(query(c, sequence=1), c.tmp_path, start=False)
    viewer._prune(root, c.record["id"], first["bytes"], incoming=first["bytes"])
    assert viewer.catalog(query(c, sequence=1), c.tmp_path)["state"] == "evicted"
    viewer._work_job(c.tmp_path, c.record["id"])
    assert not viewer.catalog(query(c, sequence=1), c.tmp_path, start=False)["waiting"]
    assert len(native.calls) == 2 and c.frame.read_bytes() == c.raw


def test_explicit_full_science_processes_only_the_selected_frame(native):
    c = native.case
    c.events.write_bytes(b"".join(encoded({**c.event, "sequence": index, "valid_time": f"2026-09-07T18:{index:02d}:00Z"}) for index in range(1, 6)))
    request = query(c, sequence=3, profile=viewer.SCIENCE_PROFILE)
    viewer.catalog(request, c.tmp_path); viewer._work_job(c.tmp_path, c.record["id"])
    value = viewer.catalog(request, c.tmp_path, start=False)
    assert len(native.calls) == 1 and native.calls[0]["schema"] == "arwen.wrf-process-request.v1"
    assert "profile" not in native.calls[0] and "products" not in native.calls[0]
    assert value["profile"] == viewer.SCIENCE_PROFILE and value["native_result"]["schema"] == "arwen.wrf-process-result.v1"
    assert not value["waiting"] and value["lead_seconds"] == 180 and len(value["members"]) == 4
    assert not viewer._queue(viewer._root(c.tmp_path), c.record["id"]) and c.frame.read_bytes() == c.raw


def test_local_cache_rotation_respects_active_reader_lease(tmp_path):
    root = cache._owned_directory(tmp_path / "cache")
    key = "a" * 64
    objects = cache._owned_directory(root / "objects")
    directory = cache._owned_directory(objects / key)
    (directory / "field.rws").write_bytes(b"retained")
    leases = cache._owned_directory(root / "leases")
    with cache.Lease(leases / (key + ".lock")) as reader:
        assert reader.file is not None
        with pytest.raises(viewer.Backpressure, match="retained by active frames"):
            cache._prune(root, 8, 8, "b" * 64, reader_leases=True)
        assert directory.exists()
    assert cache._prune(root, 8, 8, "b" * 64, reader_leases=True) == 0
    assert not directory.exists()
