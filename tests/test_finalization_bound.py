"""Finalization is timed by its own bound, sized from the work it declares.

After the last model step a worker drains its history writers, checks and
digests each domain's final state and hashes every emitted frame.  None of
that is a model step, so the step bound, max(3*p99, 120 s), says nothing
about how long it may take: a large tree's drain or a multi-GiB frame's
hash runs past it on a healthy machine, and the supervisor killed workers
that were finishing.  Each finalizing heartbeat declares the bytes the
worker will move before its next one; the phase's bound is the step bound
plus that work at a floor rate, so a worker that stops beating past it is
still stopped.
"""

from __future__ import annotations

import json
import sys
import textwrap
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

import gpuwm.runtime as runtime
import gpuwm.supervisor as supervisor
import test_supervisor as harness
from gpuwm.supervisor import (
    HEARTBEAT_SCHEMA, Heartbeat, RuntimeHeartbeat, SupervisorError,
    read_heartbeat, supervise_experiment,
)

GIB = 1024 ** 3
MIB = 1024 ** 2


class _DeclaringProcess(harness._ScriptedProcess):
    """A scripted worker whose finalizing records can declare their work.

    The declaration is written into the record as the worker writes it, a
    plain key of the published JSON, so the monitor under test reads it
    exactly as it reads a real worker's.
    """

    def _publish(self, event):
        super()._publish(event)
        if "work_bytes" in event:
            payload = json.loads(self.heartbeat_path.read_text(encoding="utf-8"))
            payload["work_bytes"] = event["work_bytes"]
            supervisor.atomic_write_json(self.heartbeat_path, payload)


def _scripted(monkeypatch, tmp_path, scripts, **kwargs):
    monkeypatch.setattr(harness, "_ScriptedProcess", _DeclaringProcess)
    return harness._install_scripted_supervisor(
        monkeypatch, tmp_path, scripts, **kwargs)


# --- the bound --------------------------------------------------------------


def test_the_finalization_bound_is_the_step_bound_plus_the_declared_work():
    floor = supervisor.FINALIZATION_FLOOR_BYTES_PER_SECOND
    bound = supervisor.finalization_stale_threshold_seconds
    assert bound(120.0, None) == 120.0
    assert bound(120.0, 0) == 120.0
    assert bound(120.0, floor) == pytest.approx(121.0)
    assert bound(300.0, 30 * floor) == pytest.approx(330.0)
    # One 4 GiB frame on the slowest storage the floor allows for.
    assert bound(120.0, 4 * GIB) == pytest.approx(120.0 + 4 * GIB / floor)


# --- the monitor ------------------------------------------------------------


@pytest.mark.parametrize("phase", [
    "drain-history-writers-d01", "hash-output-frames-3-of-10",
    "trajectory-digest-d02"])
def test_declared_finalization_work_may_outlast_the_step_bound(
        monkeypatch, tmp_path, phase):
    """4 GiB to move, 183 s of silence: past the step bound, inside its own."""
    config, checkpoint, processes = _scripted(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 2},
            {"status": f"finalizing:{phase}", "step": 2,
             "work_bytes": 4 * GIB},
            {}, {}, {},
            {"status": "complete", "step": 2, "exit": 0},
        ]], clock_step=61.0)

    result = supervise_experiment(
        config, tmp_path / "out", restart=checkpoint,
        max_restarts=0, poll_seconds=0.05)

    assert result.attempts == 1
    assert result.heartbeat.status == "complete"
    assert processes[0].terminated is False


def test_a_worker_silent_past_its_declared_work_is_still_stopped(
        monkeypatch, tmp_path):
    """CONTROL: 64 MiB buys 8 s past the step bound, not forever."""
    config, checkpoint, processes = _scripted(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 2},
            {"status": "finalizing:hash-output-frames-1-of-1", "step": 2,
             "work_bytes": 64 * MIB},
            {}, {}, {}, {}, {}, {},
        ]], clock_step=61.0)

    with pytest.raises(SupervisorError) as caught:
        supervise_experiment(
            config, tmp_path / "out", restart=checkpoint,
            max_restarts=0, poll_seconds=0.05)

    message = str(caught.value)
    assert "finalization heartbeat (finalizing:hash-output-frames-1-of-1)" \
        in message
    assert "became stale after 183.0 s" in message
    assert "64.0 MiB of declared work" in message
    assert processes[0].terminated


def test_a_write_between_two_steps_may_outlast_the_step_bound(
        monkeypatch, tmp_path):
    """A 4 GiB history frame written between steps 2 and 3, 183 s silent.

    Timed as a model step this was a stall: a finished 1132x906x55
    streamed forecast was stopped while it wrote its last frame.  The
    write declares its bytes and is bounded by them, and integration's
    bound comes back once the write is over.
    """
    config, checkpoint, processes = _scripted(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 2},
            {"status": "writing:history-d01", "step": 2,
             "work_bytes": 4 * GIB},
            {}, {}, {},
            {"status": "integrating", "step": 2},
            {"status": "integrating", "step": 3},
            {"status": "complete", "step": 3, "exit": 0},
        ]], clock_step=61.0)

    result = supervise_experiment(
        config, tmp_path / "out", restart=checkpoint,
        max_restarts=0, poll_seconds=0.05)

    assert result.attempts == 1
    assert result.heartbeat.status == "complete"
    assert processes[0].terminated is False


def test_a_write_silent_past_its_declared_bytes_is_still_stopped(
        monkeypatch, tmp_path):
    """CONTROL: 64 MiB buys 8 s past the step bound, not forever."""
    config, checkpoint, processes = _scripted(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 2},
            {"status": "writing:checkpoint", "step": 2,
             "work_bytes": 64 * MIB},
            {}, {}, {}, {}, {}, {},
        ]], clock_step=61.0)

    with pytest.raises(SupervisorError) as caught:
        supervise_experiment(
            config, tmp_path / "out", restart=checkpoint,
            max_restarts=0, poll_seconds=0.05)

    message = str(caught.value)
    assert "write heartbeat (writing:checkpoint)" in message
    assert "became stale after 183.0 s" in message
    assert "64.0 MiB of declared work" in message
    assert processes[0].terminated


def test_a_finalizing_phase_that_declares_nothing_keeps_the_step_bound(
        monkeypatch, tmp_path):
    """CONTROL: a device synchronize that never returns is a hang."""
    config, checkpoint, processes = _scripted(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 2},
            {"status": "finalizing:synchronize-device", "step": 2},
            {}, {}, {},
        ]], clock_step=61.0)

    with pytest.raises(SupervisorError, match="became stale after 122.0 s"):
        supervise_experiment(
            config, tmp_path / "out", restart=checkpoint,
            max_restarts=0, poll_seconds=0.05)
    assert processes[0].terminated


def test_the_preparation_timeout_does_not_time_finalization(
        monkeypatch, tmp_path):
    """A run resumed at its stop tick finalizes without integrating.

    ``--prep-timeout`` is a bound on preparation.  Applied to that run's
    finalization it killed the worker 30 s into writing its capsule and
    then refused to relaunch it as a deterministic preparation failure.
    """
    config, checkpoint, processes = _scripted(
        monkeypatch, tmp_path, [[
            {"status": "preparing:restore-tree-checkpoint", "step": 2},
            {"status": "finalizing:run-capsule", "step": 2},
            {}, {},
            {"status": "complete", "step": 2, "exit": 0},
        ]], clock_step=31.0)

    result = supervise_experiment(
        config, tmp_path / "out", restart=checkpoint, max_restarts=0,
        prep_timeout_seconds=30.0, poll_seconds=0.05)

    assert result.heartbeat.status == "complete"
    assert processes[0].terminated is False


def test_the_preparation_timeout_retires_at_a_published_complete(
        monkeypatch, tmp_path):
    config, checkpoint, processes = _scripted(
        monkeypatch, tmp_path, [[
            {"status": "preparing:restore-tree-checkpoint", "step": 2},
            {"status": "complete", "step": 2},
            {}, {},
            {"exit": 0},
        ]], clock_step=31.0)

    result = supervise_experiment(
        config, tmp_path / "out", restart=checkpoint, max_restarts=0,
        prep_timeout_seconds=30.0, poll_seconds=0.05)

    assert result.heartbeat.status == "complete"
    assert processes[0].terminated is False


def test_a_wedged_preparation_is_still_timed_out(monkeypatch, tmp_path):
    """CONTROL: the preparation timeout still applies to preparation."""
    config, checkpoint, processes = _scripted(
        monkeypatch, tmp_path, [[
            {"status": "preparing:prepare-case", "step": 0},
        ]], clock_step=31.0)

    with pytest.raises(SupervisorError, match="preparation timed out"):
        supervise_experiment(
            config, tmp_path / "out", restart=checkpoint, max_restarts=0,
            prep_timeout_seconds=30.0, poll_seconds=0.05)
    assert processes[0].terminated


@pytest.mark.parametrize("exit_code", [0, 3])
def test_finalizing_without_complete_never_counts_as_success(
        monkeypatch, tmp_path, exit_code):
    """CONTROL: the terminal record and the exit status still decide."""
    config, checkpoint, processes = _scripted(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 2},
            {"status": "finalizing:hash-output-frames-1-of-1", "step": 2,
             "work_bytes": MIB},
            {"exit": exit_code},
        ]])

    with pytest.raises(SupervisorError,
                       match=f"worker exited with status {exit_code}"):
        supervise_experiment(
            config, tmp_path / "out", restart=checkpoint,
            max_restarts=0, poll_seconds=0.05)
    assert processes[0].terminated is False


# --- the record -------------------------------------------------------------


def _record(**overrides):
    fields = dict(
        schema=HEARTBEAT_SCHEMA, run_id="run-1", config_digest="c" * 64,
        pid=7, started_at_utc="2026-07-16T00:00:00Z",
        updated_at_utc="2026-07-16T00:00:01Z", status="integrating",
        model_elapsed_seconds=60.0, outer_step=1, last_durable_wrfout=None,
        last_checkpoint=None)
    fields.update(overrides)
    return Heartbeat(**fields)


def test_a_record_without_declared_work_keeps_the_published_field_set():
    for status in ("integrating", "finalizing:run-capsule", "complete"):
        assert set(_record(status=status).as_dict()) == \
            supervisor._HEARTBEAT_FIELDS


def test_declared_work_round_trips_through_the_published_record(tmp_path):
    path = tmp_path / "run-progress.json"
    supervisor.write_heartbeat(path, _record(
        status="finalizing:drain-history-writers-d01", work_bytes=5 * GIB))
    assert read_heartbeat(path).work_bytes == 5 * GIB


@pytest.mark.parametrize("status,work_bytes", [
    ("integrating", 10), ("complete", 10), ("preparing:prepare-case", 10),
    ("finalizing:run-capsule", -1), ("finalizing:run-capsule", True),
    ("finalizing:run-capsule", 1.5)])
def test_declared_work_is_a_nonnegative_count_on_a_finalizing_record(
        status, work_bytes):
    with pytest.raises(ValueError, match="work_bytes"):
        _record(status=status, work_bytes=work_bytes)


def test_the_runtime_heartbeat_publishes_and_then_clears_declared_work(
        tmp_path):
    path = tmp_path / "run-progress.json"
    progress = RuntimeHeartbeat(
        path, run_id="run-1", config_sha256="b" * 64,
        started_at_utc="2026-07-16T00:00:00Z")
    progress.finalizing("hash-output-frames-1-of-2", work_bytes=4096)
    assert read_heartbeat(path).work_bytes == 4096
    progress.finalizing("run-capsule")
    assert read_heartbeat(path).work_bytes is None
    progress.complete(60.0)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "complete" and "work_bytes" not in record


def test_the_step_record_after_a_checkpoint_declares_the_read_back_first(
        tmp_path, monkeypatch):
    """The heartbeat reads a new checkpoint back before its step record.

    After a 1132x906x55 streamed forecast's stop-tick checkpoint that read
    was 10.6 GB and 104 s, silent under the step's status.  It is a
    ``writing:`` record sized from the file now, and the step record ends it.
    """
    path = tmp_path / "run-progress.json"
    checkpoint = tmp_path / "gpuwmrst_d01.npz"
    checkpoint.write_bytes(bytes(4096))
    progress = RuntimeHeartbeat(
        path, run_id="run-1", config_sha256="b" * 64,
        started_at_utc="2026-07-16T00:00:00Z")
    progress(model_elapsed_seconds=60.0, outer_step=1,
             last_durable_wrfout=None, last_checkpoint=None)
    seen = []

    def validate(target):
        seen.append(read_heartbeat(path))
        return Path(target).resolve()

    monkeypatch.setattr(supervisor, "validate_manifest_checkpoint", validate)
    progress(model_elapsed_seconds=120.0, outer_step=2,
             last_durable_wrfout=None, last_checkpoint=checkpoint)
    assert [(r.status, r.work_bytes) for r in seen] == [
        ("writing:verify-checkpoint", 4096)]
    record = read_heartbeat(path)
    assert (record.status, record.outer_step, record.work_bytes) == (
        "integrating", 2, None)
    # The same checkpoint is not read again, and says nothing more.
    progress(model_elapsed_seconds=180.0, outer_step=3,
             last_durable_wrfout=None, last_checkpoint=checkpoint)
    assert len(seen) == 1


# --- what the worker declares ----------------------------------------------


class _Recorder:
    def __init__(self):
        self.beats: list[tuple[str, int | None]] = []

    def finalizing(self, phase, *, work_bytes=None):
        self.beats.append((phase, work_bytes))


class _Domain:
    """One domain's writer: ``work`` bytes left until it is drained."""

    def __init__(self, work, drained):
        self.pending_work_bytes = work
        self.paths = []
        self._drained = drained

    def drain(self):
        self._drained.append(self)
        self.pending_work_bytes = 0


def test_the_drain_declares_every_domains_remaining_bytes_before_each_wait():
    from gpuwm.io.wrfout import PerDomainWrfoutWriters

    drained = []
    writers = object.__new__(PerDomainWrfoutWriters)
    writers._writers = {2: _Domain(3 * GIB, drained),
                        1: _Domain(5 * GIB, drained)}
    recorder = _Recorder()

    writers.drain(before_domain=runtime._drain_progress(recorder))

    assert recorder.beats == [
        ("drain-history-writers-d01", 8 * GIB),
        ("drain-history-writers-d02", 3 * GIB),
    ]
    assert [domain.pending_work_bytes for domain in drained] == [0, 0]


def test_a_domain_state_digest_declares_the_bytes_it_hashes():
    from gpuwm import state_digest

    seen = []
    manifest = {"a": np.zeros((4, 5), np.float32),
                "b": np.zeros(7, np.float64)}
    state_digest._canonical_digest_document(
        manifest, {"elapsed_seconds": 0.0, "dtbc_fp32_bits": 0,
                   "driver": None},
        "trajectory", before_hash=seen.append)
    assert seen == [4 * 5 * 4 + 7 * 8]


def test_resident_final_health_beat_declares_zero_host_scan_bytes(
        monkeypatch, tmp_path):
    from types import SimpleNamespace
    import gpuwm.core.health as health

    state = SimpleNamespace(
        scratch=lambda shape, name: np.zeros(shape, dtype=np.float32))
    node = SimpleNamespace(state=state, cfg=SimpleNamespace(grid_id=1))
    model = SimpleNamespace(walk_parent_first=lambda: iter([node]))
    path = tmp_path / "run-progress.json"
    progress = RuntimeHeartbeat(
        path, run_id="run-1", config_sha256="b" * 64,
        started_at_utc="2026-07-16T00:00:00Z")

    def require_healthy(self, *, phase):
        # Replace only the CUDA scan; check the published beat before it runs.
        assert self.state is state
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["status"] == "finalizing:final-health-d01"
        assert record.get("work_bytes") == 0
        assert read_heartbeat(path).work_bytes == 0
        return health.ValidationReport(True, 0, phase=phase)

    monkeypatch.setattr(health.StateHealthValidator, "require_healthy",
                        require_healthy)

    reports = runtime._final_health_reports(model, progress)

    assert len(reports) == 1
    assert reports[0].ok
    assert reports[0].phase == "final-state.d01"


def test_the_final_health_check_declares_a_host_stores_bytes(monkeypatch):
    """A host store is scanned on the CPU, whole; a device scan moves none."""
    import gpuwm.core.health as health
    from types import SimpleNamespace

    host = object.__new__(health.StoreHealthValidator)
    host.device = False
    host.bundle = SimpleNamespace(store={
        "t": np.zeros((3, 4, 5), np.float32), "mu": np.zeros((4, 5))})
    device = object.__new__(health.StoreHealthValidator)
    device.device = True
    device.bundle = host.bundle
    assert host.host_scan_bytes == 3 * 4 * 5 * 4 + 4 * 5 * 8
    assert device.host_scan_bytes == 0

    class Resident(health.StateHealthValidator):
        def __init__(self):
            pass

        def require_healthy(self, *, phase):
            return SimpleNamespace(ok=True, phase=phase)

    class Stored(Resident):
        host_scan_bytes = 6 * GIB

    validators = {1: Resident(), 2: Stored()}
    nodes = [SimpleNamespace(cfg=SimpleNamespace(grid_id=gid))
             for gid in (1, 2)]
    model = SimpleNamespace(walk_parent_first=lambda: iter(nodes))
    monkeypatch.setattr(health, "health_validator_for_domain",
                        lambda model, node: validators[node.cfg.grid_id])
    recorder = _Recorder()

    reports = runtime._final_health_reports(model, recorder)

    assert recorder.beats == [("final-health-d01", 0),
                              ("final-health-d02", 6 * GIB)]
    assert [report.phase for report in reports] == [
        "final-state.d01", "final-state.d02"]


class _Done:
    @staticmethod
    def synchronize():
        return None


def _cpu_writer():
    """A CPU-only AsyncDomainWrfoutWriter with its real worker thread."""
    from gpuwm.io.wrfout import AsyncDomainWrfoutWriter

    writer = object.__new__(AsyncDomainWrfoutWriter)
    writer.nx = writer.ny = writer.nz = 1
    writer.dx = writer.dy = 1.0
    writer.soil_layers = 4
    writer.title = "test"
    writer.global_attrs = {}
    writer._queue = AsyncDomainWrfoutWriter._new_ticket_queue()
    writer.stream = nullcontext()
    writer._condition = threading.Condition()
    writer._pending = 0
    writer._failure = None
    writer._failure_traceback = None
    writer._closed = False
    writer._abort_event = threading.Event()
    writer.paths = []
    writer._thread = threading.Thread(target=writer._worker, daemon=True)
    writer._thread.start()
    return writer


def test_a_domain_writer_counts_the_bytes_it_still_has_to_move(
        monkeypatch, tmp_path):
    """Write, then read back for the identity: twice, then once, then none."""
    import gpuwm.io.wrfout as wrfout

    writing, finish_write = threading.Event(), threading.Event()
    hashing, finish_hash = threading.Event(), threading.Event()

    class HeldWriter:
        publication_revision = 1

        def __init__(self, _path, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write_frame(self, _time_str, _fields):
            writing.set()
            assert finish_write.wait(timeout=60)

        def complete_output_identity(self, *, cancel_event):
            hashing.set()
            assert finish_hash.wait(timeout=60)
            return "proof"

    monkeypatch.setattr(wrfout, "WrfoutWriter", HeldWriter)
    writer = _cpu_writer()
    fields = {"T": np.zeros((2, 3, 4), np.float32),
              "MU": np.zeros((3, 4), np.float32)}
    frame_bytes = 2 * 3 * 4 * 4 + 3 * 4 * 4
    writer._admit(wrfout._AsyncFrame(
        path=tmp_path / "wrfout_d01", time_str="2026-07-16_00:00:00",
        fields=fields, event=_Done(), device_refs=(), pinned_refs=()))
    try:
        assert writing.wait(timeout=60)
        assert writer.pending_work_bytes == 2 * frame_bytes
        finish_write.set()
        assert hashing.wait(timeout=60)
        assert writer.pending_work_bytes == frame_bytes
    finally:
        finish_write.set()
        finish_hash.set()
    writer.drain()
    assert writer.pending_work_bytes == 0
    assert writer.paths == [tmp_path / "wrfout_d01"]
    writer._queue.put(None)
    writer._thread.join(timeout=60)


# --- a real worker process --------------------------------------------------

_WORKER = textwrap.dedent('''
    import os, sys, time
    sys.path.insert(0, {repo!r})
    from pathlib import Path
    from gpuwm import output_identity, runtime
    from gpuwm.supervisor import HEARTBEAT_NAME, RuntimeHeartbeat

    out = Path(sys.argv[1])
    mode = sys.argv[2]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    beat = RuntimeHeartbeat(
        out / HEARTBEAT_NAME, run_id=os.environ["GPUWM_RUN_ID"],
        config_sha256=os.environ["GPUWM_CONFIG_DIGEST"],
        started_at_utc=os.environ["GPUWM_STARTED_AT_UTC"])
    frame = out / "wrfout_d01_2026-07-16_00-00-00"
    frame.write_bytes(bytes(range(256)) * 4096)
    beat(model_elapsed_seconds=60.0, outer_step=1,
         last_durable_wrfout=None, last_checkpoint=None)
    deadline = time.monotonic() + 60
    while not (out / "integrating.observed").exists():
        assert time.monotonic() < deadline, "parent never saw integrating"
        time.sleep(0.01)
    if mode == "slow-storage":
        # The frame sits on storage slower than a model step.
        read = output_identity._hash_open_file
        def slow(*args, **kwargs):
            time.sleep({slow_seconds!r})
            return read(*args, **kwargs)
        output_identity._hash_open_file = slow
        runtime._frame_records([frame], progress_callback=beat)
        (out / "finalized.txt").write_text("hashed", encoding="utf-8")
        beat.complete(60.0)
    else:
        # Hung after declaring one frame's read: no further beat, ever.
        beat.finalizing("hash-output-frames-1-of-1",
                        work_bytes=frame.stat().st_size)
        time.sleep(3600)
''')

_FRAME_BYTES = 256 * 4096


def _real_worker(monkeypatch, tmp_path, mode, *, step_bound, frame_seconds,
                 slow_seconds=0.0):
    """Supervise one real CPU subprocess through the production monitor.

    Device selection and input hashing are replaced: there is no GPU and
    no case.  The heartbeat file, the monitor, the lock and the process
    termination are the real ones.  The step bound and the floor rate are
    scaled down together so the test takes seconds, not minutes.
    """
    config = tmp_path / "experiment.toml"
    config.write_text('[experiment]\nname="finalization-fixture"\n',
                      encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_text(_WORKER.format(
        repo=str(Path(supervisor.__file__).resolve().parents[1]),
        slow_seconds=float(slow_seconds)), encoding="utf-8")
    monkeypatch.setattr(supervisor, "select_gpu", lambda uuid: supervisor
                        .GPUIdentity("", "fixture", "CPU fixture", 0))
    monkeypatch.setattr(supervisor, "preflight_exclusive_gpu",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(supervisor, "priced_reservation_bytes", lambda path: 0)
    monkeypatch.setattr(supervisor, "resolved_input_hashes",
                        lambda *args, **kwargs: {})
    monkeypatch.setattr(supervisor, "git_commit", lambda: None)
    monkeypatch.setattr(supervisor, "installed_identity",
                        lambda: {"kind": "cpu-fixture"})
    monkeypatch.setattr(
        supervisor, "_worker_command",
        lambda config, payload, outdir, **kwargs: [
            sys.executable, "-B", str(worker), str(outdir), mode])
    monkeypatch.setattr(supervisor.RollingStepWall, "stale_threshold_seconds",
                        property(lambda self: step_bound))
    monkeypatch.setattr(supervisor, "FINALIZATION_FLOOR_BYTES_PER_SECOND",
                        _FRAME_BYTES / frame_seconds, raising=False)
    monkeypatch.delenv(supervisor.SHARED_INPUT_AUTHORITY_ROOT_ENV,
                       raising=False)
    out = tmp_path / "out"

    def progress(heartbeat):
        if heartbeat.status == "integrating":
            (out / "integrating.observed").write_text("seen", encoding="utf-8")

    return supervise_experiment(
        config, out, max_restarts=0, poll_seconds=0.05,
        lock_path=tmp_path / "fixture.lock", on_progress=progress), out


def test_a_real_worker_hashing_a_frame_on_slow_storage_finishes(
        monkeypatch, tmp_path):
    """The hash takes 4 s against a 2 s step bound; the frame is priced
    at 20 s, so the worker finishes and publishes complete."""
    result, out = _real_worker(
        monkeypatch, tmp_path, "slow-storage", step_bound=2.0,
        frame_seconds=20.0, slow_seconds=4.0)

    assert result.attempts == 1
    assert result.heartbeat.status == "complete"
    assert (out / "finalized.txt").read_text(encoding="utf-8") == "hashed"


def test_a_real_worker_hung_in_finalization_is_stopped_after_its_bound(
        monkeypatch, tmp_path):
    """CONTROL: the same declared frame, and then nothing.  Stopped once
    its bound (0.5 s step + 2 s of declared work) has passed."""
    started = time.monotonic()
    with pytest.raises(SupervisorError) as caught:
        _real_worker(monkeypatch, tmp_path, "hang", step_bound=0.5,
                     frame_seconds=2.0)
    wall = time.monotonic() - started

    message = str(caught.value)
    assert ("finalization heartbeat (finalizing:hash-output-frames-1-of-1)"
            in message)
    assert "its bound was 2.5 s for 1.0 MiB of declared work" in message
    # Stopped once the bound had passed, not an hour later.
    assert 2.5 <= wall < 60.0
