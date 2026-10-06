"""Raw histories can leave while their original writer identity remains."""
from datetime import datetime
import hashlib
import json

import pytest

from gpuwm.ensemble.history_ledger import EnsembleHistoryLedger
from gpuwm.output_identity import completed_file_record


def history(tmp_path):
    path = tmp_path / "members" / "member-0001" / "wrfout_d01_2026-10-04_01-00-00"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original completed history")
    proof = completed_file_record(path)
    ledger = EnsembleHistoryLedger(tmp_path, member_order=(0, 1))
    record = ledger.register(proof, member_id=1, grid_id=1, episode=0,
        valid_time=datetime(2026, 10, 4, 1))
    return path, proof, ledger, record


def retirement(ledger, record):
    ledger.retirements.mkdir()
    path = ledger.retirements / (hashlib.sha256(record["path"].encode()).hexdigest() + ".json")
    receipt = {"schema": ledger.RETIREMENT,
        **{key: record[key] for key in ("path", "bytes", "sha256")},
        "retired_at": "2026-10-04T01:00:30Z", "product_manifest_sha256": "b" * 64,
        "products": [{"path": "viewer/hour-1.pack", "sha256": "c" * 64, "bytes": 15}]}
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_verified_hourly_retirement_keeps_original_path_bytes_hash_and_order(tmp_path):
    raw, proof, ledger, record = history(tmp_path)
    assert ledger.register(proof, member_id=1, grid_id=1, episode=0,
        valid_time=datetime(2026, 10, 4, 1)) == record
    retirement(ledger, record)
    raw.unlink()
    final = ledger.validate_retirements()
    assert final[0]["retirement_state"] == "retired"
    assert all(final[0][key] == record[key] for key in ("path", "bytes", "sha256"))
    assert json.loads(ledger.path.read_text())["records"] == final


def test_missing_unregistered_deletion_refuses_a_success_authority(tmp_path):
    raw, _, ledger, _ = history(tmp_path)
    raw.unlink()
    with pytest.raises(RuntimeError, match="missing without its verified retirement receipt"):
        ledger.validate_retirements()


def test_receipt_replacement_cannot_retire_different_history_words(tmp_path):
    _, _, ledger, record = history(tmp_path)
    path = retirement(ledger, record)
    receipt = json.loads(path.read_text())
    receipt["sha256"] = "d" * 64
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not verify"):
        ledger.validate_retirements()


def test_eager_runtime_capture_registers_exact_completed_proof(tmp_path):
    # The capture hands the exact proof to its owner, which decides whether
    # the history is its own to ledger (the capture flag only says the
    # writer must write it; simulated radar needs histories it deletes).
    from gpuwm.ensemble.runtime_context import MemberOutputCapture
    raw, proof, _, _ = history(tmp_path)
    calls = []
    class Collector:
        def submit(self, **kwargs):
            pass
        def history_committed(self, proof, **kwargs):
            calls.append((proof, kwargs))
    capture = MemberOutputCapture(Collector().submit, 1, keep_member_files=True)
    capture.history_committer(grid_id=1, episode=2)(proof, datetime(2026, 10, 4, 1))
    assert calls[0][0] is proof and proof.path == str(raw.resolve())
    assert calls[0][1]["member_id"] == 1 and calls[0][1]["episode"] == 2


def _collector(root, *, keep):
    import numpy as np
    from gpuwm.ensemble.batch_product_output import HeadlineDiagnosticCollector
    return HeadlineDiagnosticCollector(root, members=2, renderer="rw_wrfbatch",
        start_time=datetime(2026, 10, 4), keep_member_files=keep, array_module=np)


def test_retained_history_inside_the_tree_enters_the_ledger(tmp_path):
    from gpuwm.ensemble.runtime_context import MemberOutputCapture
    raw, proof, _, _ = history(tmp_path)
    collector = _collector(tmp_path, keep=True)
    capture = MemberOutputCapture(collector.submit, 1, keep_member_files=True)
    capture.history_committer(grid_id=1)(proof, datetime(2026, 10, 4, 1))
    (row,) = collector.history_ledger.inventory()
    assert row["path"] == raw.relative_to(tmp_path).as_posix() and row["sha256"] == proof.sha256
    assert collector.member_files == [{"member_id": 1, "domain": "d01",
        "path": row["path"], "bytes": raw.stat().st_size}]


def test_radar_history_without_retention_is_not_ledgered(tmp_path):
    # member_history_required() makes the capture write histories for a
    # member's simulated radar although the request does not retain them.
    # Registering them used to fail the writer ("member histories require
    # keep_member_files"); ledgering them would demand them back later.
    from gpuwm.ensemble.runtime_context import MemberOutputCapture
    _, proof, _, _ = history(tmp_path)
    collector = _collector(tmp_path / "ensemble", keep=False)
    capture = MemberOutputCapture(collector.submit, 1, keep_member_files=True)
    assert capture.history_committer(grid_id=1)(proof, datetime(2026, 10, 4, 1)) is None
    assert not collector.history_ledger.inventory() and not collector.member_files


def test_history_written_outside_the_ensemble_tree_is_not_ledgered(tmp_path):
    # A caller may run an ordinary member into its own directory under a
    # capture scope. That file is not the ensemble's to retire; it used to
    # fail the writer with "is not in the subpath".
    from gpuwm.ensemble.runtime_context import MemberOutputCapture
    _, proof, _, _ = history(tmp_path)
    collector = _collector(tmp_path / "ordinary-products", keep=True)
    capture = MemberOutputCapture(collector.submit, 1, keep_member_files=True)
    assert capture.history_committer(grid_id=1)(proof, datetime(2026, 10, 4, 1)) is None
    assert not collector.history_ledger.inventory() and not collector.member_files
