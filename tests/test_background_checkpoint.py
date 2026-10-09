"""The prepared door's background checkpoint writer and its two orderings.

Breakage this guards (1 km CONUS, 8 x RTX 5090, 2026-10-04): every hourly
checkpoint stopped the model for 31 to 37 s -- about 12 s waiting for the
hour's history frame to land, then 18 to 21 s writing a 25 GB archive.  The
checkpoint is now snapshotted between two steps and written on a thread of
its own.  What must not change with it:

* a checkpoint is published only after every history frame up to its valid
  time is durable (``PerDomainWrfoutWriters.durability_barrier``);
* the published archive is the synchronous writer's, byte for byte;
* the snapshot owns what it writes, so later steps cannot change it;
* a failed write is raised on the stepping thread, never lost;
* a frame's landing latency is timed from the frame, not from the log's
  creation or from whatever step the model has since reached.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pytest

from tests.test_restart import (_LIFECYCLE_FREE_CHILD_DIGEST,
                                _LIFECYCLE_FREE_ROOT_DIGEST,
                                _PINNED_UNDER_MEASURED_CLOCK,
                                _canonical_member_digest,
                                _sealed_tree_fixture)
from tests.test_wrfout import _manual_async_writer, _queue_cpu_ticket


# -- the publisher ------------------------------------------------------------

def test_a_failed_background_write_is_raised_on_the_stepping_thread():
    from gpuwm.prepared_single_domain_forecast import _CheckpointPublisher

    publisher = _CheckpointPublisher()

    def fails():
        raise OSError("disk full")

    publisher.submit(fails)
    with pytest.raises(RuntimeError, match="checkpoint writer failed: OSError: disk full"):
        publisher.wait()
    publisher.wait()  # reported once, then clear


def test_at_most_one_checkpoint_is_in_flight():
    from gpuwm.prepared_single_domain_forecast import _CheckpointPublisher

    publisher = _CheckpointPublisher()
    release = threading.Event()
    order = []

    def first():
        release.wait()
        order.append("first")

    publisher.submit(first)
    threading.Timer(0.2, release.set).start()
    publisher.submit(lambda: order.append("second"))
    publisher.wait()
    assert order == ["first", "second"]


# -- the snapshot -------------------------------------------------------------

def test_a_deferred_checkpoint_is_the_synchronous_one_byte_for_byte(
        monkeypatch, tmp_path):
    from gpuwm.io import restart

    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31,
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)
    pending = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600),
        defer_publish=True)
    assert isinstance(pending, restart.PendingTreeCheckpoint)
    assert pending.deferrable
    assert not list(tmp_path.glob("gpuwmrst_*")), "nothing lands before publish"
    assert source._last_checkpoint is None
    root_path = pending.publish()
    child_path = next(p for p in tmp_path.glob("gpuwmrst_d02_*.npz"))
    assert source._last_checkpoint == root_path
    assert _canonical_member_digest(root_path) == _LIFECYCLE_FREE_ROOT_DIGEST
    assert _canonical_member_digest(child_path) == _LIFECYCLE_FREE_CHILD_DIGEST


def test_a_deferred_checkpoint_owns_what_it_writes(monkeypatch, tmp_path):
    """Steps taken between the snapshot and the write change nothing in it."""
    from gpuwm.io import restart

    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31,
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)
    pending = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600),
        defer_publish=True)
    touched = 0
    for node in source.walk_parent_first():
        for value in restart.state_manifest(node.state).values():
            if isinstance(value, np.ndarray) and value.size \
                    and value.dtype.kind == "f":
                value[...] += 1.0
                touched += 1
    assert touched
    root_path = pending.publish()
    assert _canonical_member_digest(root_path) == _LIFECYCLE_FREE_ROOT_DIGEST


def test_frames_in_flight_refuse_a_synchronous_checkpoint_but_not_a_deferred_one(
        monkeypatch, tmp_path):
    """The D2H count stops a checkpoint landing ahead of its frames.  A
    deferred one is published only after them, so it is not refused; found
    by the first real run, whose CPU fixtures carry no io manager."""
    from types import SimpleNamespace
    from gpuwm.io import restart

    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31,
        run_overrides=_PINNED_UNDER_MEASURED_CLOCK)
    source._io_manager = SimpleNamespace(pending=1)
    with pytest.raises(restart.RestartMismatchError, match="'D2H': 1"):
        restart.write_tree_restart(
            tmp_path, source, start + timedelta(seconds=3600))
    pending = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600),
        defer_publish=True)
    root_path = pending.publish()
    assert _canonical_member_digest(root_path) == _LIFECYCLE_FREE_ROOT_DIGEST


def test_a_failed_publish_removes_the_generation_and_releases_its_store(
        monkeypatch, tmp_path):
    from gpuwm.io import restart

    source, start = _sealed_tree_fixture(
        monkeypatch, forcing_count=2, run_seconds=3600.0, payload_seed=31)

    class Lender:
        guards = []

        def add_store_guard(self, guard):
            self.guards.append(guard)

    pending = restart.write_tree_restart(
        tmp_path, source, start + timedelta(seconds=3600),
        defer_publish=True)
    lender = Lender()
    # Rebuild with a lender and a root member that fails after the child.
    members = list(pending._members)
    child_gid, child_member = members[0]

    def root_fails():
        raise OSError("injected root write failure")

    failing = restart.PendingTreeCheckpoint(
        directory=tmp_path, model=source, root_id=1,
        members=[(child_gid, child_member), (1, root_fails)],
        lenders=[lender])
    assert len(lender.guards) == 1
    with pytest.raises(OSError, match="injected root write failure"):
        failing.publish()
    assert not list(tmp_path.glob("gpuwmrst_*")), \
        "a half-written generation must not stay on disk"
    lender.guards[0]()  # released: returns at once instead of blocking
    assert source._last_checkpoint is None


# -- the history ordering -----------------------------------------------------

class _BlockingWriter:
    started = None
    finish = None

    def __init__(self, _path, **_kwargs):
        pass

    def __enter__(self):
        return self

    def write_frame(self, _time_str, _fields):
        type(self).started.set()
        type(self).finish.wait()

    def __exit__(self, *_args):
        return False


def test_the_barrier_waits_for_frames_queued_before_it_and_no_later(
        monkeypatch, tmp_path):
    import gpuwm.io.wrfout as wrfout

    _BlockingWriter.started = threading.Event()
    _BlockingWriter.finish = threading.Event()
    monkeypatch.setattr(wrfout, "WrfoutWriter", _BlockingWriter)
    writer = _manual_async_writer(threading.Event())
    try:
        _queue_cpu_ticket(writer, tmp_path / "f00")
        assert _BlockingWriter.started.wait(timeout=30)
        target = writer.durability_target()
        assert target == 1
        landed = threading.Event()
        waiter = threading.Thread(
            target=lambda: (writer.wait_landed(target), landed.set()))
        waiter.start()
        assert not landed.wait(timeout=0.3), "returned before the frame landed"
        _BlockingWriter.finish.set()
        assert landed.wait(timeout=30)
        waiter.join()
        # A frame queued after the barrier is not waited for.
        _BlockingWriter.started.clear()
        _BlockingWriter.finish.clear()
        _queue_cpu_ticket(writer, tmp_path / "f01")
        assert _BlockingWriter.started.wait(timeout=30)
        writer.wait_landed(target)  # returns at once
        _BlockingWriter.finish.set()
        writer.drain()
        assert writer.durability_target() == 2
    finally:
        _BlockingWriter.finish.set()
        writer.close()


def test_a_frame_that_never_lands_fails_the_barrier_by_name(
        monkeypatch, tmp_path):
    import gpuwm.io.wrfout as wrfout

    class FailingWriter(_BlockingWriter):
        def write_frame(self, _time_str, _fields):
            raise OSError("injected history failure")

    monkeypatch.setattr(wrfout, "WrfoutWriter", FailingWriter)
    writer = _manual_async_writer(threading.Event())
    try:
        _queue_cpu_ticket(writer, tmp_path / "f00")
        target = writer.durability_target()
        with pytest.raises(RuntimeError, match="injected history failure"):
            writer.wait_landed(target)
    finally:
        with pytest.raises(RuntimeError):
            writer.close()


# -- the landing latency ------------------------------------------------------

def test_a_frame_is_timed_from_its_own_hand_off(tmp_path):
    from gpuwm.progress_log import StepLog

    log = StepLog(start_time=datetime(2026, 10, 3, 21), run_seconds=7200.0,
                  text_stream=StringIO(), jsonl_path=tmp_path / "p.jsonl")
    valid0 = datetime(2026, 10, 3, 21)
    valid1 = valid0 + timedelta(hours=1)
    frame0 = tmp_path / "wrfout_d01_0"
    frame1 = tmp_path / "wrfout_d01_1"
    frame0.write_bytes(b"x")
    frame1.write_bytes(b"x")
    time.sleep(0.5)                       # preflight, cache restore, ...
    log.frame_submitted(domain=1, valid_time=valid0)
    log.output_committed(domain=1, valid_time=valid0, path=frame0)
    log.domain_step(grid_id=1, step_count=1, model_seconds=3600.0,
                    step_wall_seconds=0.1)
    log.frame_submitted(domain=1, valid_time=valid1)
    time.sleep(0.3)                       # the write
    log.domain_step(grid_id=1, step_count=2, model_seconds=3606.0,
                    step_wall_seconds=0.1)  # the model moved on meanwhile
    log.output_committed(domain=1, valid_time=valid1, path=frame1)
    log.close()
    import json
    rows = [json.loads(line) for line in
            (tmp_path / "p.jsonl").read_text().splitlines()]
    latencies = [r["durable_after_seconds"] for r in rows
                 if r["event"] == "output_written"]
    assert latencies[0] < 0.4, "frame 0 was timed from the log's creation"
    assert latencies[1] >= 0.3, "frame 1 was timed from a later step"
