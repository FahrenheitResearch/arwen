"""A resident history download must not fence the next model mutation."""
from datetime import datetime
import threading
import time
import weakref

import netCDF4
import numpy as np
import pytest

from conftest import requires_gpu


@requires_gpu
@pytest.mark.gpu
def test_resident_capture_releases_producer_before_download(monkeypatch, tmp_path):
    import cupy as cp
    import gpuwm.io.wrfout as wrfout

    # Strided fields, aliases, a device scalar and a host extra all reach
    # the real writer, then the live device arrays change immediately.
    original = np.arange(48, dtype=np.float32).reshape(2, 4, 6)
    original[0, 0, :2] = np.array([0x80000000, 1], np.uint32).view(np.float32)
    producer = cp.cuda.Stream(non_blocking=True)
    with producer:
        base = cp.asarray(original)
        temperature = base[:, :, ::2]
        mu = cp.arange(12, dtype=cp.float32).reshape(4, 3)
        scalar = cp.asarray(np.array(1, np.uint32).view(np.float32))
        # Compile the later mutations before holding a CUDA stream.
        base.fill(900)
        mu.fill(901)
        scalar.fill(902)
        base.set(original)
        mu.set(np.arange(12, dtype=np.float32).reshape(4, 3))
        scalar.set(np.array(1, np.uint32).view(np.float32))
        fields = {"T": temperature, "THM": temperature, "MU": mu,
                  "FMOIST_LASTTIME": scalar}
        expected = {name: cp.asnumpy(value) for name, value in fields.items()}
        monkeypatch.setattr(wrfout, "_device_state_frame", lambda *a, **k: fields)
        writer = wrfout.AsyncDomainWrfoutWriter(
            nx=3, ny=4, nz=2, dx=1, dy=1, title="capture", global_attrs={})
        release = threading.Event()
        entered = threading.Event()
        allocate = cp.cuda.alloc_pinned_memory
        # cudaHostAlloc can synchronize streams. Reserve real pinned storage
        # before the hold so the test isolates the stream dependency.
        staging_memory = [allocate(512) for _ in range(len(fields))]
        held = []

        def hold_download(_):
            entered.set()
            release.wait(10)

        def allocate_after_capture(nbytes):
            assert nbytes <= 512
            if not held:
                # This is after the device snapshot and before the first
                # D2H. Hold only the download stream, without CUDA calls in
                # the callback, so independence is an ordering assertion.
                held.append(True)
                cp.cuda.runtime.launchHostFunc(writer.stream.ptr, hold_download, 0)
            return staging_memory.pop()

        monkeypatch.setattr(cp.cuda, "alloc_pinned_memory", allocate_after_capture)
        path = tmp_path / "capture"
        try:
            writer.submit(path, datetime(2026, 10, 18), None,
                          extra_fields={"P_TOP": np.array(5000, np.float32)})
            base.fill(900)
            mu.fill(901)
            scalar.fill(902)
            advanced = cp.cuda.Event()
            advanced.record(producer)
            assert entered.wait(5), "download did not reach its hold"
            deadline = time.monotonic() + 5
            while not advanced.done and time.monotonic() < deadline:
                time.sleep(0.005)
            assert advanced.done, "model stream still waits for the history download"
        finally:
            release.set()
            writer.close()

    with netCDF4.Dataset(path) as dataset:
        dataset.set_auto_mask(False)
        for name, value in expected.items():
            actual = np.asarray(dataset.variables[name][0])
            assert actual.shape == value.shape
            assert actual.tobytes() == value.tobytes(), name
        assert np.asarray(dataset.variables["P_TOP"][0]).shape == ()


@requires_gpu
@pytest.mark.gpu
def test_next_capture_releases_downloaded_queued_snapshot(monkeypatch, tmp_path):
    import cupy as cp
    import gpuwm.io.wrfout as wrfout

    producer = cp.cuda.Stream(non_blocking=True)
    expected = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    with producer:
        field = cp.asarray(expected)
        writer = wrfout.AsyncDomainWrfoutWriter(
            nx=3, ny=4, nz=2, dx=1, dy=1, title="backpressure", global_attrs={})
    monkeypatch.setattr(wrfout, "_device_state_frame", lambda *a, **k: {"T": field})
    entered = threading.Event()
    release = threading.Event()
    third_staged = threading.Event()
    tickets = []
    failures = []
    write_frame = wrfout.WrfoutWriter.write_frame
    admit = writer._admit

    def held_write(self, *args, **kwargs):
        if not entered.is_set():
            entered.set()
            if not release.wait(10):
                raise RuntimeError("history write hold timed out")
        return write_frame(self, *args, **kwargs)

    def record_ticket(ticket):
        tickets.append(ticket)
        if len(tickets) == 3:
            third_staged.set()
        return admit(ticket)

    monkeypatch.setattr(wrfout.WrfoutWriter, "write_frame", held_write)
    monkeypatch.setattr(writer, "_admit", record_ticket)

    def submit(index):
        with producer:
            writer.submit(tmp_path / f"capture-{index}", datetime(2026, 10, 18), None)

    def submit_third():
        try:
            submit(2)
        except BaseException as exc:
            failures.append(exc)

    thread = None
    try:
        submit(0)
        assert entered.wait(5)
        submit(1)
        writer.stream.synchronize()
        queued_refs = [weakref.ref(value) for value in tickets[1].device_refs]
        assert queued_refs and all(ref() is not None for ref in queued_refs)
        thread = threading.Thread(target=submit_third)
        thread.start()
        assert third_staged.wait(5)
        assert thread.is_alive(), "bounded queue did not apply backpressure"
        assert all(ref() is None for ref in queued_refs)
        assert tickets[1].pinned_refs, "host staging must survive until writing"
    finally:
        release.set()
        if thread is not None:
            thread.join(10)
        writer.close()
    assert thread is not None and not thread.is_alive()
    assert not failures
    for index in range(3):
        with netCDF4.Dataset(tmp_path / f"capture-{index}") as dataset:
            dataset.set_auto_mask(False)
            assert np.asarray(dataset.variables["T"][0]).tobytes() == expected.tobytes()


@requires_gpu
@pytest.mark.gpu
def test_frame_past_the_snapshot_budget_is_copied_straight_to_host(monkeypatch, tmp_path):
    """A field the device has no room to snapshot goes straight to host.

    Breakage this prevents: full-CONUS HRRR resident on one 96 GB card ran
    every step and died at its first history frame in the snapshot loop
    ("Out of memory allocating 68,585,472 bytes (allocated so far:
    99,013,528,576 bytes)").  With no budget every field takes the direct
    copy, the frame is byte-identical, and the live arrays can change the
    moment submit returns.
    """
    import cupy as cp
    import gpuwm.io.wrfout as wrfout

    original = np.arange(48, dtype=np.float32).reshape(2, 4, 6)
    producer = cp.cuda.Stream(non_blocking=True)
    with producer:
        base = cp.asarray(original)
        strided = base[:, :, ::2]
        mu = cp.arange(12, dtype=cp.float32).reshape(4, 3)
        fields = {"T": strided, "THM": strided, "MU": mu}
        expected = {name: cp.asnumpy(value) for name, value in fields.items()}
        monkeypatch.setattr(wrfout, "_device_state_frame", lambda *a, **k: fields)
        monkeypatch.setattr(wrfout, "_history_snapshot_budget", lambda: 0)
        writer = wrfout.AsyncDomainWrfoutWriter(
            nx=3, ny=4, nz=2, dx=1, dy=1, title="direct", global_attrs={})
        path = tmp_path / "direct"
        snapshots = []
        array = cp.array

        def counted(*args, **kwargs):
            snapshots.append(args)
            return array(*args, **kwargs)

        monkeypatch.setattr(cp, "array", counted)
        try:
            writer.submit(path, datetime(2026, 10, 18), None)
            base.fill(900)
            mu.fill(901)
        finally:
            writer.close()
    assert not snapshots, "a field was snapshotted on the device past the budget"
    with netCDF4.Dataset(path) as dataset:
        dataset.set_auto_mask(False)
        for name, value in expected.items():
            assert np.asarray(dataset.variables[name][0]).tobytes() == value.tobytes(), name
