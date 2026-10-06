"""Checkpoint barriers preserve asynchronous product and retirement identities."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from threading import Event, RLock, Thread

import numpy as np
import pytest

from gpuwm.ensemble.batch_product_output import HeadlineDiagnosticCollector, NativeDiagnosticSpool
from gpuwm.ensemble.batch_products import FieldProducts
from gpuwm.ensemble.product_consumer import DiagnosticProductConsumer
from gpuwm.ensemble.restart_roster import CheckpointProgress
from gpuwm.output_identity import completed_file_record


def test_idle_barrier_is_bounded_and_does_not_close_later_hour_admission():
    consumer = DiagnosticProductConsumer()
    entered, release = Event(), Event()
    consumer.submit(("d01", "hour1"), device=0,
        replay=lambda: (entered.set(), release.wait(5)))
    assert entered.wait(5)
    try:
        with pytest.raises(TimeoutError, match="bounded hold"):
            consumer.wait_idle(timeout=.01)
        assert not consumer.receipt()["closed"]
        release.set()
        consumer.wait_idle(timeout=5)
        completed = []
        consumer.submit(("d01", "hour2"), device=0, replay=lambda: completed.append(2))
        consumer.wait_idle(timeout=5)
        assert completed == [2]
    finally:
        release.set()
        consumer.close()


def test_timeout_keeps_old_roster_without_binding_new_checkpoint_or_step_retry(tmp_path, monkeypatch):
    from gpuwm.ensemble import restart_roster
    roster = tmp_path / restart_roster.ROSTER
    roster.write_bytes(b"previous coherent member checkpoint")
    calls, bound, sealed, events = [], [], [], []
    class Collector:
        def save_resume(self):
            calls.append(1)
            raise TimeoutError("busy product owner")
    monkeypatch.setattr(restart_roster, "bind_checkpoint_member", lambda *args: bound.append(args))
    monkeypatch.setattr(restart_roster, "seal", lambda *args: sealed.append(args))
    progress = CheckpointProgress(lambda **event: events.append(event), collector=Collector(),
        root=tmp_path, manifest_lock=RLock())
    progress(last_checkpoint=tmp_path / "new.npz", outer_step=10)
    progress(last_checkpoint=tmp_path / "new.npz", outer_step=11)
    assert calls == [1] and not bound and not sealed
    assert len(events) == 2 and "ensemble_checkpoint_deferred" in events[0]
    assert roster.read_bytes() == b"previous coherent member checkpoint"
    assert not (tmp_path / "new.member.json").exists()


def test_checkpoint_barrier_propagates_consumer_failure():
    consumer = DiagnosticProductConsumer()
    consumer.submit(("d01", "hour1"), device=0,
        replay=lambda: (_ for _ in ()).throw(ValueError("wrong diagnostic identity")))
    with pytest.raises(RuntimeError, match="wrong diagnostic identity"):
        consumer.wait_idle(timeout=5)
    consumer.close(cancel=True)


def collector(root, **options):
    return HeadlineDiagnosticCollector(root, members=2, renderer="rw_wrfbatch",
        start_time=datetime(2024, 1, 1), requests=(FieldProducts("wind10", "m s-1", (25.,)),),
        array_module=np, available_bytes=0, keep_member_files=True, **options)


def spool(owner):
    coords = np.zeros((2, 2), np.float32)
    geometry = hashlib.sha256(coords.tobytes() * 2).hexdigest()
    result = NativeDiagnosticSpool(owner.root, members=owner.members, requests=owner.requests,
        latitude=coords, longitude=coords, renderer="rw_wrfbatch", keep_member_files=True)
    owner.spools[("d01", geometry)] = result
    return result


def pack_record(path, member=0):
    return {"path": path, "member_ids": (member,), "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def frame(valid, packs, members, *, status="pending", products=()):
    return {"valid_time": valid, "domain": "d01", "status": status,
        "members_received": list(members), "members_expected": 2, "packs": packs,
        "products": list(products), "maps": [], "available_fields": ["wind10"], "unavailable_fields": []}


def test_snapshot_waits_for_inflight_retirement_and_retains_finished_product(tmp_path):
    original = collector(tmp_path)
    output = spool(original)
    diagnostic = tmp_path / "diagnostic.nc"
    diagnostic.write_bytes(b"immutable diagnostic")
    valid = "2024-01-01_00:00:00"
    output.frames[valid] = frame(valid, [pack_record(diagnostic)], (0, 1))
    output._manifest()
    entered, release, saved = Event(), Event(), []
    product = tmp_path / "product.nc"
    def replay():
        entered.set()
        assert release.wait(5)
        with output._lock:
            product.write_bytes(b"verified aggregate product")
            output.frames[valid].update(status="complete", products=["product.nc"])
            diagnostic.unlink()
            output._manifest()
    original.product_consumer.submit(("d01", valid), device=0, replay=replay)
    assert entered.wait(5)
    writer = Thread(target=lambda: saved.append(original.save_resume(timeout=5)))
    writer.start()
    assert writer.is_alive()
    release.set()
    writer.join(5)
    assert not writer.is_alive() and len(saved) == 1
    assert "product.nc" in saved[0]["files"] and "diagnostic.nc" not in saved[0]["files"]
    assert not original.product_consumer.receipt()["closed"]
    original.product_consumer.close()


def test_fresh_process_snapshot_keeps_retired_history_identity_and_pending_pack(tmp_path):
    source, fresh = tmp_path / "source", tmp_path / "fresh"
    original = collector(source)
    output = spool(original)
    raw = source / "members/member-0000/wrfout_d01_2024-01-01_00-00-00"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"completed history")
    identity = original.history_ledger.register(completed_file_record(raw), member_id=0,
        grid_id=1, episode=0, valid_time=datetime(2024, 1, 1))
    original.member_files = [{"member_id": 0, "domain": "d01", "path": identity["path"], "bytes": identity["bytes"]}]
    output.member_files = list(original.member_files)
    receipt_dir = original.history_ledger.retirements
    receipt_dir.mkdir()
    receipt = receipt_dir / (hashlib.sha256(identity["path"].encode()).hexdigest() + ".json")
    receipt.write_text(json.dumps({"schema": original.history_ledger.RETIREMENT,
        **{name: identity[name] for name in ("path", "bytes", "sha256")},
        "retired_at": "2024-01-01T00:00:30Z", "product_manifest_sha256": "b" * 64,
        "products": [{"path": "viewer/hour-0.pack", "bytes": 15, "sha256": "c" * 64}]}))
    raw.unlink()
    diagnostic = source / "diagnostic.nc"
    diagnostic.write_bytes(b"pending member one words")
    product = source / "product.nc"
    product.write_bytes(b"published complete hour")
    coordinate = source / "coordinate.nc"
    coordinate.write_bytes(b"sealed coordinate words")
    output.coordinate_file = {"path": "coordinate.nc", "bytes": coordinate.stat().st_size,
        "sha256": hashlib.sha256(coordinate.read_bytes()).hexdigest()}
    output.frames["2024-01-01_00:00:00"] = frame("2024-01-01_00:00:00", [], (0, 1),
        status="complete", products=("product.nc",))
    output.frames["2024-01-01_01:00:00"] = frame("2024-01-01_01:00:00", [pack_record(diagnostic, 1)], (1,))
    output._manifest()
    document = original.save_resume()
    original.product_consumer.close()
    assert identity["path"] not in document["files"]
    for name in [*document["files"], ".ensemble-resume/collector.json"]:
        destination = fresh / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
    resumed = collector(fresh, resume=True)
    resumed.restore_resume()
    restored = next(iter(resumed.spools.values()))
    assert resumed.history_ledger.inventory()[0]["retirement_state"] == "retired"
    assert resumed.member_files[0]["path"] == identity["path"]
    assert not (fresh / identity["path"]).exists()
    assert restored.coordinate_file == output.coordinate_file and not restored._publish_coordinates
    assert restored.frames["2024-01-01_01:00:00"]["packs"][0]["path"] == fresh / "diagnostic.nc"
    with pytest.raises(ValueError, match="already scheduled"):
        resumed.product_consumer.submit(("d01", "2024-01-01_00:00:00"), device=0, replay=lambda: None)
    resumed.product_consumer.close()
    (fresh / "diagnostic.nc").write_bytes(b"changed pending words")
    refused = collector(fresh, resume=True)
    with pytest.raises(ValueError, match="committed hash"):
        refused.restore_resume()


def test_complete_pending_hour_is_readmitted_once_on_cpu_after_restore(tmp_path, monkeypatch):
    original = collector(tmp_path)
    output = spool(original)
    paths = [tmp_path / f"member-{member}.nc" for member in (0, 1)]
    for member, path in enumerate(paths):
        path.write_bytes(f"sealed member {member} diagnostic".encode())
    valid = "2024-01-01_01:00:00"
    output.frames[valid] = frame(valid, [pack_record(path, member) for member, path in enumerate(paths)], (0, 1))
    output._manifest()
    original.save_resume()
    original.product_consumer.close(cancel=True)
    calls = []
    def finish(restored, clock, *, available_bytes, consumer, render_products):
        calls.append((clock, available_bytes, consumer.gpu_replay))
        restored.frames[clock].update(status="complete")
        restored._manifest()
    monkeypatch.setattr(NativeDiagnosticSpool, "finish_in_subprocess", finish)
    resumed = collector(tmp_path, resume=True)
    resumed.restore_resume()
    resumed.product_consumer.wait_idle(timeout=5)
    assert calls == [(valid, 0, False)]
    with pytest.raises(ValueError, match="already scheduled"):
        resumed.product_consumer.submit(("d01", valid), device=0, replay=lambda: None)
    resumed.product_consumer.close()
