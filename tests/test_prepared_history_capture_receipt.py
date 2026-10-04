"""Native output receipts retain cadence without inventing suppressed files."""
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from gpuwm.prepared_single_domain_forecast import _completed_history_inventory


def schedule():
    start = datetime(2024, 1, 1)
    return tuple((offset, start + timedelta(seconds=offset),
                  (start + timedelta(seconds=offset)).strftime("wrfout_d01_%Y-%m-%d_%H_%M_%S"))
                 for offset in (0, 3600))


def test_suppressed_histories_record_capture_without_native_file_claims():
    requested = schedule()
    inventory, fields = _completed_history_inventory(
        [], requested, captured_paths=[Path(row[2]) for row in requested])
    assert inventory == []
    assert fields["expected_frame_count"] == 0
    assert fields["scheduled_frame_count"] == fields["capture"]["frame_count"] == 2
    assert fields["capture"]["exact_schedule_verified"] is True
    assert fields["capture"]["history_files_retained"] is False
    assert fields["capture"]["frames"] == [
        {"name": name, "model_elapsed_seconds": offset, "valid_time": valid.isoformat()}
        for offset, valid, name in requested]
    for name in ("initial_frame_verified", "last_scheduled_frame_verified",
                 "initial_and_final_frames_verified", "all_frames_readback_verified"):
        assert fields[name] is False


def test_retained_history_receipt_is_unchanged_and_missing_frames_still_fail():
    requested = schedule()
    records = [{"path": name, "bytes": 1024, "sha256": str(index) * 64}
               for index, (_, _, name) in enumerate(requested)]
    inventory, fields = _completed_history_inventory(records, requested)
    assert fields == {}
    assert inventory == [{**record, "model_elapsed_seconds": offset,
                          "valid_time": valid.isoformat(), "atomic_writer_readback_verified": True}
                         for record, (offset, valid, _) in zip(records, requested)]
    with pytest.raises(ValueError):
        _completed_history_inventory(records[:1], requested)


@pytest.mark.parametrize("indices", [(0,), (1, 0), (0, 0), ()])
def test_missing_reordered_or_duplicate_capture_cannot_claim_completed_cadence(indices):
    requested = schedule()
    with pytest.raises(RuntimeError, match="hash-bound history handoff"):
        _completed_history_inventory([], requested,
            captured_paths=[Path(requested[index][2]) for index in indices])


def test_suppressed_history_cannot_misreport_retained_native_records():
    requested = schedule()
    with pytest.raises(RuntimeError, match="unexpectedly retained native file records"):
        _completed_history_inventory([{"path": requested[0][2]}], requested,
            captured_paths=[Path(row[2]) for row in requested])
