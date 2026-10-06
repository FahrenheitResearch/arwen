"""Completed forecasts keep state and frame evidence after history deletion."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import output_identity, runtime, supervisor
from gpuwm.prepared_single_domain_forecast import _completed_history_inventory
from gpuwm.state_digest import canonical_state_digest
from gpuwm.verify.cases.real74_d02 import canonical_state_digest as reference_digest
from test_wrfout import _manual_async_writer, _queue_cpu_ticket


def _write_trajectory(folder, *, delete_on_landing):
    """Use the native writer and advance resident state between landings."""
    state = SimpleNamespace(
        thp=np.full((1, 1, 1), 271.5, np.float32),
        elapsed_seconds=0.0, physics=None, _scratch={})
    clock = SimpleNamespace(dtbc_fp32=np.float32(12.0))
    writer = _manual_async_writer(Event())
    landed = []
    start = datetime(2026, 10, 4, tzinfo=timezone.utc)
    schedule = []

    def landing(*, domain, valid_time, path):
        payload = path.read_bytes()
        # Completion precedes this callback, so the harness can discard
        # every file here without ever racing its hash.
        proof = writer.completed_records[-1]
        landed.append((path, len(payload), hashlib.sha256(payload).hexdigest(),
                       proof.record()))
        if delete_on_landing:
            path.unlink()

    writer.landing_observer = landing
    try:
        for index in range(3):
            path = folder / f"wrfout_d01_{index:02d}"
            offset = index * 12.0
            schedule.append((offset, start + timedelta(seconds=offset), path.name))
            _queue_cpu_ticket(writer, path, fields={"T": state.thp.copy()})
            writer.drain()
            # A deleted frame is followed by another model step and write,
            # rather than only disappearing after forecast completion.
            assert path.exists() is not delete_on_landing
            state.thp += np.float32(0.25)
            state.elapsed_seconds += 12.0
    finally:
        writer.close()

    assert len(landed) == len(writer.paths) == len(writer.completed_records) == 3
    for path, size, sha256, record in landed:
        assert record == {"path": str(path.resolve()), "bytes": size,
                          "sha256": sha256}
    return SimpleNamespace(
        paths=writer.paths, completed=writer.completed_records,
        landed=landed, schedule=schedule, state=state, clock=clock)


def test_frames_deleted_mid_run_keep_hashes_and_final_digest(
        tmp_path, monkeypatch, caplog):
    run = _write_trajectory(tmp_path / "discarded", delete_on_landing=True)
    assert all(not path.exists() for path in run.paths)
    expected_digest = canonical_state_digest(run.state, run.clock)
    forbidden = {str(path.resolve()) for path in run.paths}
    opening = Path.open

    def no_history_read(path, mode="r", *args, **kwargs):
        if "r" in mode and str(path.resolve()) in forbidden:
            raise AssertionError(f"finalization re-read discarded history {path}")
        return opening(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_history_read)
    with caplog.at_level(logging.WARNING):
        records = runtime._frame_records(run.paths, completed_records=run.completed)
    assert records == [
        {**proof.record(), "available": False,
         "identity_source": "writer-completion"}
        for proof in run.completed]
    assert all(str(path) in caplog.text for path in run.paths)
    assert canonical_state_digest(run.state, run.clock) == expected_digest

    inventory, capture = _completed_history_inventory(records, run.schedule)
    assert capture == {}
    assert all(record["atomic_writer_readback_verified"] for record in inventory)
    assert [row["sha256"] for row in inventory] == [row[2] for row in run.landed]
    summary = runtime.ExperimentRunSummary(
        wrfout_paths=tuple(run.paths), completed_seconds=run.state.elapsed_seconds,
        nan_free=True, frame_records=tuple(records),
        trajectory_digest={"d01": expected_digest})
    receipt = supervisor._success_output(summary)
    persisted = json.loads(json.dumps(receipt))
    assert persisted["frames"] == records
    assert persisted["trajectory_digest"]["d01"] == expected_digest


@pytest.mark.parametrize("scope", ["trajectory", "full"])
def test_final_digest_matches_undeleted_run_and_existing_digest(tmp_path, scope):
    retained = _write_trajectory(tmp_path / "retained", delete_on_landing=False)
    discarded = _write_trajectory(tmp_path / "discarded", delete_on_landing=True)
    before = canonical_state_digest(retained.state, retained.clock, scope=scope)
    assert before == reference_digest(retained.state, retained.clock, scope=scope)
    assert canonical_state_digest(discarded.state, discarded.clock, scope=scope) == before
    for path in retained.paths:
        path.unlink()
    assert canonical_state_digest(retained.state, retained.clock, scope=scope) == before


def test_retained_frame_receipts_keep_the_existing_format(tmp_path):
    run = _write_trajectory(tmp_path, delete_on_landing=False)
    expected = [
        {"path": str(path.resolve()), "bytes": size, "sha256": sha256}
        for path, size, sha256, _proof in run.landed]
    records = runtime._frame_records(run.paths, completed_records=run.completed)
    assert records == expected
    assert records == runtime._frame_records(run.paths)
    inventory, capture = _completed_history_inventory(records, run.schedule)
    assert capture == {}
    assert inventory == [
        {**record, "model_elapsed_seconds": offset,
         "valid_time": valid_time.isoformat(),
         "atomic_writer_readback_verified": True}
        for record, (offset, valid_time, _name) in zip(expected, run.schedule, strict=True)]


def test_missing_legacy_frame_is_named_and_preserves_receipt_shape(tmp_path, caplog):
    missing = tmp_path / "wrfout_d01_missing"
    with caplog.at_level(logging.WARNING):
        records = runtime._frame_records([missing])
    assert records == [{
        "path": str(missing.resolve()), "bytes": None, "sha256": None,
        "available": False, "identity_source": "unavailable"}]
    assert str(missing) in caplog.text
    schedule = [(0.0, datetime(2026, 10, 4, tzinfo=timezone.utc), missing.name)]
    inventory, capture = _completed_history_inventory(records, schedule)
    assert capture == {}
    assert inventory[0]["atomic_writer_readback_verified"] is False
    # Authority checks outside forecast finalization still require a file.
    with pytest.raises(FileNotFoundError):
        output_identity.file_records([missing])


@pytest.mark.parametrize("replacement", [False, True])
def test_retained_frame_mutation_still_refuses_stale_writer_evidence(
        tmp_path, replacement):
    run = _write_trajectory(tmp_path, delete_on_landing=False)
    path = run.paths[-1]
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    if replacement:
        changed = tmp_path / "replacement"
        changed.write_bytes(payload)
        changed.replace(path)
    else:
        path.write_bytes(payload)
    with pytest.raises(output_identity.OutputChangedError, match="changed"):
        runtime._frame_records(run.paths, completed_records=run.completed)


@pytest.mark.parametrize("discard_after_rename", [False, True])
def test_synchronous_case_output_collects_native_frame_hashes(
        tmp_path, monkeypatch, discard_after_rename):
    import gpuwm.io.wrfout as wrfout

    fields = {"T": np.full((1, 1, 1), 271.5, np.float32)}
    state = SimpleNamespace(_streamed_domain=SimpleNamespace(
        history_fields=lambda: {name: value.copy() for name, value in fields.items()}))
    prepared = SimpleNamespace(
        initial_result=SimpleNamespace(state=state, coord=None),
        cfg=SimpleNamespace(nx=1, ny=1, nz=1, dx=1.0, dy=1.0,
                            mp_physics=0, sf_surface_physics=2, num_soil_layers=4),
        grid=None, static_fields={})
    # Only geographic orchestration is synthetic. The field tape,
    # completion validation, hash and publication all use the native writer.
    monkeypatch.setattr(runtime, "_metadata_frame", lambda *_args: {})
    monkeypatch.setattr(runtime, "_global_wrf_attrs", lambda *_args, **_kwargs: {})
    publish = wrfout.replace_file_with_retry
    expected = []

    def observe_publication(source, destination, *args, **kwargs):
        payload = Path(source).read_bytes()
        expected.append({"path": str(Path(destination).resolve()),
                         "bytes": len(payload),
                         "sha256": hashlib.sha256(payload).hexdigest()})
        result = publish(source, destination, *args, **kwargs)
        if discard_after_rename:
            Path(destination).unlink()
        return result

    monkeypatch.setattr(wrfout, "replace_file_with_retry", observe_publication)
    completed = []
    paths = []
    start = datetime(2026, 10, 4)
    for index in range(2):
        paths.append(runtime.write_case_output(
            prepared, tmp_path, start + timedelta(seconds=12 * index),
            start_time=start, title="forecast", expect_refl_10cm=False,
            completed_records=completed))
        fields["T"] += np.float32(0.25)
    assert [proof.record() for proof in completed] == expected
    assert len(completed) == len(paths) == 2
    assert all(path.exists() == (not discard_after_rename) for path in paths)
    records = runtime._frame_records(paths, completed_records=completed)
    assert records == ([{**row, "available": False,
                         "identity_source": "writer-completion"}
                        for row in expected] if discard_after_rename else expected)


@pytest.mark.skipif(os.name == "nt", reason="unlinking an open address needs Linux sharing")
def test_deletion_between_publication_address_and_revision_checks_keeps_identity(
        tmp_path, monkeypatch):
    path = tmp_path / "frame"
    address_matches = output_identity._address_matches
    discarded = []

    def discard_after_address_check(candidate, revision):
        matched = address_matches(candidate, revision)
        if Path(candidate) == path and matched:
            payload = path.read_bytes()
            discarded.append({"path": str(path.resolve()), "bytes": len(payload),
                              "sha256": hashlib.sha256(payload).hexdigest()})
            path.unlink()
        return matched

    monkeypatch.setattr(output_identity, "_address_matches", discard_after_address_check)
    writer = _manual_async_writer(Event())
    try:
        _queue_cpu_ticket(writer, path)
    finally:
        writer.close()
    assert not path.exists()
    assert writer.paths == [path]
    assert len(discarded) == len(writer.completed_records) == 1
    assert writer.completed_records[0].record() == discarded[0]
    assert runtime._frame_records(writer.paths, completed_records=writer.completed_records) == [
        {**discarded[0], "available": False, "identity_source": "writer-completion"}]


@pytest.mark.skipif(os.name == "nt", reason="Windows verifies retained payloads rather than reusing a revision")
def test_deletion_between_reusable_address_and_revision_checks_keeps_identity(
        tmp_path, monkeypatch, caplog):
    run = _write_trajectory(tmp_path, delete_on_landing=False)
    path = run.paths[-1]
    proof = run.completed[-1]
    if not proof.revision.reusable:
        pytest.skip("the filesystem does not provide a reusable file revision")
    address_matches = output_identity._address_matches
    discarded = []

    def discard_after_address_check(candidate, revision):
        matched = address_matches(candidate, revision)
        if Path(candidate) == path and matched:
            assert revision == proof.revision
            discarded.append(Path(candidate))
            path.unlink()
        return matched

    monkeypatch.setattr(output_identity, "_address_matches", discard_after_address_check)
    with caplog.at_level(logging.WARNING):
        records = runtime._frame_records(run.paths, completed_records=run.completed)
    assert discarded == [path]
    assert not path.exists()
    assert records == [
        *[completed.record() for completed in run.completed[:-1]],
        {**proof.record(), "available": False, "identity_source": "writer-completion"}]
    assert str(path) in caplog.text
    assert records[-1]["sha256"] == run.landed[-1][2]
