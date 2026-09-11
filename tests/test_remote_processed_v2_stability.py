"""Worker scheduling and publication disappearance against the real module."""
from pathlib import Path

import pytest

from gpuwm import remote_artifacts as ra, remote_processed_v2 as viewer


def test_worker_services_queued_job_after_large_historical_workspace(tmp_path, monkeypatch):
    root = viewer._root(tmp_path)
    for index in range(1024):
        (root / f"history-{index:04}").mkdir()
    active = root / "z-active"
    active.mkdir()
    (active / "queue.json").write_text("[]")
    queued = [True]
    processed = []

    def work(_workspace, job, *, completion):
        # Faithful to the callee: the worker hands every job its completion
        # wait, so a runner-exit window cannot empty this queue.
        assert isinstance(completion, ra.CompletionWait)
        processed.append(job)
        queued.clear()
        return False

    monkeypatch.setattr(viewer, "_work_job", work)
    monkeypatch.setattr(viewer, "_queue", lambda _root, _job: queued[:])
    monkeypatch.setattr(viewer.time, "sleep", lambda _seconds: pytest.fail("Queued job was never serviced"))
    monkeypatch.setattr(viewer.os, "nice", lambda _priority: None, raising=False)
    assert viewer.worker(tmp_path) == 0
    assert processed == ["z-active"]
    assert not queued


def ready_entry(tmp_path):
    member = tmp_path / "frame.rws"
    member.write_bytes(b"fixture")
    return {"state": "ready", "object_root": str(tmp_path),
            "members": [{"path": str(member), "bytes": 7}]}


def test_availability_treats_eviction_between_checks_as_cache_miss(tmp_path, monkeypatch):
    entry = ready_entry(tmp_path)

    class EvictingPath(type(Path())):
        def is_file(self):
            exists = super().is_file()
            if exists and self.name == "frame.rws":
                self.unlink()
            return exists

    monkeypatch.setattr(viewer, "Path", EvictingPath)
    assert viewer._available(entry) is False


def test_availability_preserves_permission_and_metadata_failures(tmp_path, monkeypatch):
    entry = ready_entry(tmp_path)

    class DeniedPath(type(Path())):
        def is_dir(self):
            raise PermissionError("fixture permission denied")

    monkeypatch.setattr(viewer, "Path", DeniedPath)
    with pytest.raises(PermissionError):
        viewer._available(entry)
    monkeypatch.setattr(viewer, "Path", Path)
    del entry["members"][0]["bytes"]
    with pytest.raises(KeyError):
        viewer._available(entry)


def test_availability_keeps_ready_size_and_state_checks(tmp_path):
    entry = ready_entry(tmp_path)
    assert viewer._available(entry) is True
    entry["members"][0]["bytes"] += 1
    assert viewer._available(entry) is False
    for value in [None, {"state": "queued"}, {"state": "failed"}]:
        assert viewer._available(value) is False
