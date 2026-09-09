"""Committed-frame scheduling contracts; fixture bytes contain no weather."""
import pytest

from gpuwm import remote_preparation_v2 as preparation, remote_processed_v2 as viewer
from test_remote_artifacts import case, encoded
from test_remote_processed_v2 import native, query


def test_background_covers_all_domains_with_bounded_interactive_priority(native):
    c = native.case
    c.events.write_bytes(b"".join(encoded({**c.event, "sequence": index,
        "domain": 1 + index % 3, "valid_time": f"2026-09-07T18:{index:02d}:00Z"}) for index in range(1, 45)))
    request = query(c, domain=2, sequence=1, products=viewer.DEFAULT_PRODUCTS)
    viewer.catalog(request, c.tmp_path)
    state = preparation.prepare(c.tmp_path, c.record["id"], start=False)
    root = viewer._root(c.tmp_path)
    rows = viewer._queue(root, c.record["id"])
    assert state["committed"] == 44 and state["queued"] == 32 and state["pending"] == 12
    assert rows[0]["sequence"] == 1 and rows[1]["sequence"] == 44
    assert {row["domain"] for row in rows} == {1, 2, 3}
    assert preparation.prepare(c.tmp_path, c.record["id"], start=False)["added"] == 0
    viewer._work_job(c.tmp_path, c.record["id"])
    state = preparation.prepare(c.tmp_path, c.record["id"], start=False)
    assert state["ready"] == 1 and state["added"] == 1
    assert len(viewer._queue(root, c.record["id"])) == viewer.MAX_QUEUED
    assert c.frame.read_bytes() == c.raw


def test_background_finishes_terminal_job_and_does_not_rebuild_evicted_frames(native):
    c = native.case
    preparation.prepare(c.tmp_path, c.record["id"], start=False)
    viewer._work_job(c.tmp_path, c.record["id"])
    c.status["state"] = "completed"
    state = preparation.prepare(c.tmp_path, c.record["id"], start=False)
    assert state["done"] and state["state"] == "complete" and state["ready"] == 1
    root = viewer._root(c.tmp_path)
    viewer._prune(root, c.record["id"], 0)
    state = preparation.prepare(c.tmp_path, c.record["id"], start=False)
    assert state["done"] and state["evicted"] == 1 and state["added"] == 0
    assert len(native.calls) == 1 and c.frame.read_bytes() == c.raw


def test_background_rejects_changed_authority_without_queue_mutation(native):
    c = native.case
    preparation.prepare(c.tmp_path, c.record["id"], start=False)
    viewer._work_job(c.tmp_path, c.record["id"])
    c.events.write_bytes(encoded({**c.event, "geometry": {"changed": True}}))
    with pytest.raises(ValueError, match="disagrees"):
        preparation.prepare(c.tmp_path, c.record["id"], start=False)
    assert not viewer._queue(viewer._root(c.tmp_path), c.record["id"])
    assert len(native.calls) == 1


def test_background_waits_without_manifest_and_stops_failed_empty_job(native):
    c = native.case
    c.manifest_path.unlink()
    assert preparation.prepare(c.tmp_path, c.record["id"], start=False)["state"] == "waiting_for_output"
    c.status["state"] = "failed"
    state = preparation.prepare(c.tmp_path, c.record["id"], start=False)
    assert state["done"] and state["committed"] == 0


def test_background_launch_is_cpu_bounded_and_detached(native, monkeypatch):
    c = native.case
    calls = []
    monkeypatch.setattr(preparation.subprocess, "Popen", lambda argv, **kwargs: calls.append((argv, kwargs)))
    preparation.ensure(c.tmp_path, c.record["id"])
    argv, options = calls[0]
    assert options["start_new_session"] and options["close_fds"]
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert all(options["env"][name] == "2" for name in (
        "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
    assert "--job" in argv and argv[-1] == c.record["id"]
