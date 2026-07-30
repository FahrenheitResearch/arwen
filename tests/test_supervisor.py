"""CPU/Windows gates for the Phase-5 fresh-process supervisor."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import gpuwm.case_data as case_data
import gpuwm.supervisor as supervisor
from gpuwm.supervisor import (
    HEARTBEAT_SCHEMA, GPUAlreadyLockedError, GPUFileLock, GPUIdentity,
    GPUPreflightError, GPUProcess, Heartbeat, RollingStepWall,
    RuntimeHeartbeat, SupervisorError, atomic_publish_file, atomic_write_json,
    _current_transition_receipt, _parse_pmon_output, is_cuda_fatal,
    parse_compute_apps_output,
    preflight_exclusive_gpu,
    quarantine_file, read_heartbeat, stale_threshold_seconds,
    supervise_experiment, validate_manifest_checkpoint, write_heartbeat,
)


def _heartbeat(**updates):
    values = dict(
        schema=HEARTBEAT_SCHEMA, run_id="run-1", config_digest="a" * 64,
        pid=max(os.getpid(), 1), started_at_utc="2026-07-16T00:00:00Z",
        updated_at_utc="2026-07-16T00:00:01Z", status="integrating",
        model_elapsed_seconds=60.0, outer_step=1,
        last_durable_wrfout=None, last_checkpoint=None)
    values.update(updates)
    return Heartbeat(**values)


def test_heartbeat_exact_schema_round_trip(tmp_path):
    path = tmp_path / "run-progress.json"
    write_heartbeat(path, _heartbeat(last_durable_wrfout="wrfout.nc"))
    assert read_heartbeat(path) == _heartbeat(
        last_durable_wrfout="wrfout.nc")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema", "run_id", "config_digest", "pid", "started_at_utc",
        "updated_at_utc", "status", "model_elapsed_seconds", "outer_step",
        "last_durable_wrfout", "last_checkpoint",
    }


def test_supervisor_ignores_stale_transition_receipt(tmp_path):
    path = tmp_path / "microphysics-transitions.json"
    payload = {
        "schema": "gpuwm.microphysics-transitions/v1",
        "status": "PASS",
        "run_id": "old-run",
        "config_digest": "a" * 64,
        "transitions": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _current_transition_receipt(
        tmp_path, "new-run", "a" * 64) == (None, None)
    assert _current_transition_receipt(
        tmp_path, "old-run", "b" * 64) == (None, None)
    current, digest = _current_transition_receipt(
        tmp_path, "old-run", "a" * 64)
    assert current == path.resolve()
    assert digest is not None and len(digest) == 64


def test_heartbeat_remains_valid_json_when_writer_is_killed_mid_publish(
        tmp_path):
    """Terminate a separate Windows process after fsync but before replace."""
    final = tmp_path / "run-progress.json"
    ready = tmp_path / "ready"
    atomic_write_json(final, {"generation": 1})
    code = "\n".join((
        "import time",
        "from pathlib import Path",
        "from gpuwm.supervisor import atomic_write_json",
        f"final = Path({str(final)!r})",
        f"ready = Path({str(ready)!r})",
        "def pause(_):",
        "    ready.write_text('ready', encoding='utf-8')",
        "    while True:",
        "        time.sleep(0.01)",
        "atomic_write_json(final, {'generation': 2}, _before_replace=pause)",
    ))
    process = subprocess.Popen(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, close_fds=True)
    deadline = time.monotonic() + 10.0
    while not ready.exists() and process.poll() is None:
        if time.monotonic() > deadline:
            process.kill()
            pytest.fail(process.stderr.read().decode("utf-8", errors="replace"))
        time.sleep(0.01)
    process.terminate()
    process.wait(timeout=10)
    assert json.loads(final.read_text(encoding="utf-8")) == {"generation": 1}
    orphans = list(tmp_path.glob("run-progress.json.tmp.*"))
    assert len(orphans) == 1
    assert quarantine_file(orphans[0], reason="kill-test").exists()


def test_atomic_publication_never_exposes_incomplete_final(tmp_path):
    final = tmp_path / "wrfout.nc"
    final.write_bytes(b"old-complete")

    def partial(path):
        path.write_bytes(b"readable-but-incomplete")

    def reject(path):
        assert path.read_bytes() == b"readable-but-incomplete"
        raise ValueError("missing completion attribute")

    with pytest.raises(ValueError, match="completion"):
        atomic_publish_file(final, partial, reject)
    assert final.read_bytes() == b"old-complete"
    assert not list(tmp_path.glob(".wrfout.nc.tmp.*"))
    quarantined = list((tmp_path / ".quarantine").iterdir())
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"readable-but-incomplete"


def test_atomic_publication_handles_producer_exception(tmp_path):
    final = tmp_path / "wrfout.nc"

    def explode(path):
        path.write_bytes(b"prefix")
        raise OSError("injected disk failure")

    with pytest.raises(OSError, match="injected"):
        atomic_publish_file(final, explode, lambda _: None)
    assert not final.exists()
    assert list((tmp_path / ".quarantine").iterdir())


def test_heartbeat_replace_survives_a_concurrent_reader(tmp_path):
    """A Windows reader without FILE_SHARE_DELETE must not kill the writer."""
    path = tmp_path / "run-progress.json"
    write_heartbeat(path, _heartbeat(outer_step=1))
    opened = threading.Event()

    def hold_reader():
        with path.open("rb"):
            opened.set()
            time.sleep(0.08)

    reader = threading.Thread(target=hold_reader)
    reader.start()
    assert opened.wait(timeout=2.0)
    write_heartbeat(path, _heartbeat(
        updated_at_utc="2026-07-16T00:00:02Z", outer_step=2))
    reader.join(timeout=2.0)
    assert not reader.is_alive()
    assert read_heartbeat(path).outer_step == 2


def _try_lock_in_fresh_process(path: Path) -> subprocess.CompletedProcess:
    code = "\n".join((
        "import sys",
        "from gpuwm.supervisor import GPUAlreadyLockedError, GPUFileLock",
        "try:",
        f"    lock = GPUFileLock('GPU-test', path={str(path)!r}).acquire()",
        "except GPUAlreadyLockedError:",
        "    sys.exit(23)",
        "else:",
        "    lock.release()",
        "    sys.exit(0)",
    ))
    return subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=15, close_fds=True)


def test_gpu_uuid_lock_excludes_a_second_process_on_windows(tmp_path):
    path = tmp_path / "gpu.lock"
    with GPUFileLock("GPU-test", path=path, run_id="owner"):
        blocked = _try_lock_in_fresh_process(path)
        assert blocked.returncode == 23, blocked.stderr
    acquired = _try_lock_in_fresh_process(path)
    assert acquired.returncode == 0, acquired.stderr


def test_lock_contention_raises_specific_error_in_process(tmp_path):
    # POSIX flock is process-scoped on some platforms; this assertion is only
    # about the Windows byte-range implementation used by the target box.
    if os.name != "nt":
        pytest.skip("Windows byte-range lock behavior")
    path = tmp_path / "gpu.lock"
    first = GPUFileLock("GPU-test", path=path).acquire()
    try:
        with pytest.raises(GPUAlreadyLockedError):
            GPUFileLock("GPU-test", path=path).acquire()
    finally:
        first.release()


def test_lock_error_classification_does_not_conflate_disk_errors():
    import errno
    import gpuwm.supervisor as supervisor

    contention = errno.EACCES
    assert supervisor._is_lock_contention(OSError(contention, "held"))
    assert not supervisor._is_lock_contention(OSError(errno.EIO, "disk"))


def test_captured_wddm_compute_apps_fixture_parses_desktop_rows():
    fixture = (Path(__file__).with_name("data") /
               "nvidia_smi_wddm_compute_apps_61074.csv")
    processes = parse_compute_apps_output(
        fixture.read_text(encoding="utf-8"))
    assert len(processes) == 29
    assert all(process.uuid.startswith("GPU-") for process in processes)
    assert all(process.used_gpu_memory_mib is None for process in processes)
    assert any(process.process_name == "[Insufficient Permissions]"
               for process in processes)
    assert any(process.process_name.endswith("explorer.exe")
               for process in processes)


def test_captured_wddm_pmon_fixture_parses_exact_twelve_columns():
    fixture = (Path(__file__).with_name("data") /
               "nvidia_smi_wddm_pmon_61074.txt")
    modes = _parse_pmon_output(fixture.read_text(encoding="utf-8"))
    assert len(modes) == 27
    assert modes[632] == ("C+G", 0)
    assert modes[12888] == ("C+G", 0)
    assert modes[39444] == ("C", 0)


def test_wddm_preflight_rejects_pure_cuda_with_dash_or_zero_fb(monkeypatch):
    import gpuwm.supervisor as supervisor

    gpu = GPUIdentity("GPU-test", "610.74", "RTX 5090", 0)
    modes = _parse_pmon_output(
        "0 41001 C - - - - - - - 0 python.exe\n"
        "0 41005 C 25 0 - - - - 0 0 python.exe\n")
    assert modes == {41001: ("C", None), 41005: ("C", 0)}
    processes = tuple(
        GPUProcess(gpu.uuid, pid, "python.exe", memory, mode)
        for pid, (mode, memory) in modes.items())
    monkeypatch.setattr(supervisor, "select_gpu", lambda uuid: gpu)
    monkeypatch.setattr(
        supervisor, "query_compute_processes", lambda uuid: processes)
    with pytest.raises(GPUPreflightError) as caught:
        preflight_exclusive_gpu(gpu.uuid)
    assert "pid=41001" in str(caught.value)
    assert "pid=41005" in str(caught.value)
    assert "memory=unmeasured" in str(caught.value)


def test_wddm_preflight_permits_c_plus_g_pmon_rows(monkeypatch):
    import gpuwm.supervisor as supervisor

    gpu = GPUIdentity("GPU-test", "610.74", "RTX 5090", 0)
    modes = _parse_pmon_output(
        "0 41002 C+G - - - - - - - 0 explorer.exe\n"
        "0 41003 C+G - - - - - - 0 0 chrome.exe\n")
    processes = tuple(
        GPUProcess(gpu.uuid, pid, f"desktop-{pid}.exe", memory, mode)
        for pid, (mode, memory) in modes.items())
    assert modes == {41002: ("C+G", None), 41003: ("C+G", 0)}
    monkeypatch.setattr(supervisor, "select_gpu", lambda uuid: gpu)
    monkeypatch.setattr(
        supervisor, "query_compute_processes", lambda uuid: processes)
    preflight_exclusive_gpu(gpu.uuid)


def test_parse_pmon_output_rejects_malformed_column_count():
    with pytest.raises(GPUPreflightError, match="malformed pmon row"):
        _parse_pmon_output("0 41004 C - - - - - - 0 bad.exe\n")


def test_linux_pmon_idle_row_is_an_explicit_empty_sample():
    idle = (
        "# gpu pid type sm mem enc dec jpg ofa fb ccpm command\n"
        "0 - - - - - - - - - - -\n"
    )
    assert _parse_pmon_output(idle) == {}
    with pytest.raises(GPUPreflightError, match="malformed pmon PID"):
        _parse_pmon_output("0 - C - - - - - - - - python\n")


def test_wddm_preflight_permits_graphics_and_rejects_large_pure_cuda(
        monkeypatch):
    import gpuwm.supervisor as supervisor

    gpu = GPUIdentity("GPU-test", "610.74", "RTX 5090", 0)
    processes = (
        GPUProcess(gpu.uuid, 101, "explorer.exe", None, "C+G"),
        GPUProcess(gpu.uuid, 102, "python.exe", 16, "C"),
        GPUProcess(gpu.uuid, 103, "python.exe", 512, "C"),
    )
    monkeypatch.setattr(supervisor, "select_gpu", lambda uuid: gpu)
    monkeypatch.setattr(
        supervisor, "query_compute_processes", lambda uuid: processes)
    with pytest.raises(GPUPreflightError, match="pid=103"):
        preflight_exclusive_gpu(gpu.uuid)
    preflight_exclusive_gpu(gpu.uuid, approved_pids={103})
    preflight_exclusive_gpu(gpu.uuid, allow_shared_gpu=True)


def test_stale_threshold_is_three_p99_with_120_second_floor():
    assert stale_threshold_seconds([1.0, 2.0, 40.0]) == 120.0
    assert stale_threshold_seconds([1.0, 2.0, 50.0]) == 150.0
    history = RollingStepWall(maxlen=3)
    for value in (1.0, 2.0, 3.0, 60.0):
        history.add(value)
    assert history.p99 == 60.0
    assert history.stale_threshold_seconds == 180.0


def test_runtime_heartbeat_durability_work_runs_once_per_new_artifact(
        monkeypatch, tmp_path):
    import gpuwm.supervisor as supervisor

    wrfout = tmp_path / "wrfout"
    checkpoint = tmp_path / "checkpoint.npz"
    wrfout.write_bytes(b"complete")
    checkpoint.write_bytes(b"manifest-valid")
    calls = {"fsync": 0, "checkpoint": 0}

    def fsync(path):
        calls["fsync"] += 1
        return Path(path)

    def validate(path):
        calls["checkpoint"] += 1
        return Path(path).resolve()

    monkeypatch.setattr(supervisor, "fsync_file", fsync)
    monkeypatch.setattr(supervisor, "validate_manifest_checkpoint", validate)
    progress = RuntimeHeartbeat(
        tmp_path / "run-progress.json", run_id="run",
        config_sha256="a" * 64, started_at_utc="2026-07-16T00:00:00Z")
    for step in (1, 2):
        progress(
            model_elapsed_seconds=60.0 * step, outer_step=step,
            last_durable_wrfout=wrfout, last_checkpoint=checkpoint)
    assert calls == {"fsync": 1, "checkpoint": 1}


def test_cuda_fatal_classification_policy():
    assert is_cuda_fatal("CUDA_ERROR_ILLEGAL_ADDRESS")
    assert is_cuda_fatal(RuntimeError("unspecified launch failure"))
    assert is_cuda_fatal("device-lost while synchronizing")
    assert not is_cuda_fatal(ValueError("ordinary bad configuration"))


def test_manifest_checkpoint_validation_reads_all_declared_members(tmp_path):
    arrays = {"state/u": np.arange(6, dtype=np.float32).reshape(2, 3)}
    header = {
        "array_manifest": {
            "state/u": {"shape": [2, 3], "dtype": "float32"}},
    }
    payload = {
        "__gpuwm_restart_header__": np.frombuffer(
            json.dumps(header).encode("utf-8"), dtype=np.uint8),
        **arrays,
    }
    path = tmp_path / "checkpoint.npz"
    with path.open("wb") as stream:
        np.savez(stream, **payload)
    assert validate_manifest_checkpoint(path) == path.resolve()

    with np.load(path, allow_pickle=False) as archive:
        tampered = {key: archive[key] for key in archive.files
                    if key != "state/u"}
    broken = tmp_path / "broken.npz"
    with broken.open("wb") as stream:
        np.savez(stream, **tampered)
    with pytest.raises(Exception, match="manifest"):
        validate_manifest_checkpoint(broken)


class _FakeClock:
    def __init__(self, step):
        self.now = 0.0
        self.step = float(step)

    def monotonic(self):
        return self.now

    def sleep(self, _seconds):
        self.now += self.step


class _ScriptedProcess:
    """Popen-shaped worker whose poll calls publish scripted heartbeats."""

    def __init__(self, command, env, script, pid):
        self.command = tuple(command)
        self.env = env
        self.script = list(script)
        self.pid = pid
        self.returncode = None
        self.terminated = False
        outdir = Path(command[command.index("--outdir") + 1])
        self.heartbeat_path = outdir / "run-progress.json"
        self.initial_checkpoint = (
            command[command.index("--restart") + 1]
            if "--restart" in command else None)
        self._generation = 0

    def _publish(self, event):
        self._generation += 1
        status = event["status"]
        checkpoint = event.get("checkpoint", self.initial_checkpoint)
        heartbeat_pid = int(event.get("heartbeat_pid", self.pid))
        write_heartbeat(self.heartbeat_path, Heartbeat(
            HEARTBEAT_SCHEMA, self.env["GPUWM_RUN_ID"],
            self.env["GPUWM_CONFIG_DIGEST"], heartbeat_pid,
            self.env["GPUWM_STARTED_AT_UTC"],
            f"2026-07-16T00:00:{self._generation:02d}Z", status,
            float(event.get("model_seconds", event.get("step", 0) * 60.0)),
            int(event.get("step", 0)), None,
            None if checkpoint is None else str(Path(checkpoint).resolve())))

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self.script:
            event = self.script.pop(0)
            if "status" in event:
                self._publish(event)
            if "exit" in event:
                self.returncode = int(event["exit"])
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.terminated = True
        self.returncode = -9


def _install_scripted_supervisor(monkeypatch, tmp_path, scripts, *,
                                 clock_step=1.0,
                                 publish_before_popen_returns=False):
    import gpuwm.supervisor as supervisor

    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\nname='fake'\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"fixture")
    gpu = GPUIdentity("GPU-test", "610.74", "RTX 5090", 0)
    processes = []
    scripts = [list(script) for script in scripts]
    clock = _FakeClock(clock_step)

    class _Lock:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def popen(command, *, env, **_kwargs):
        process = _ScriptedProcess(
            command, env, scripts[len(processes)], 41_000 + len(processes))
        if publish_before_popen_returns:
            process.poll()
        processes.append(process)
        return process

    monkeypatch.setattr(supervisor, "config_digest", lambda _path: "a" * 64)
    monkeypatch.setattr(supervisor, "resolved_input_hashes",
                        lambda _path, **_kwargs: {})
    monkeypatch.setattr(supervisor, "select_gpu", lambda _uuid: gpu)
    monkeypatch.setattr(supervisor, "preflight_exclusive_gpu",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(supervisor, "GPUFileLock", _Lock)
    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    monkeypatch.setattr(supervisor, "validate_manifest_checkpoint",
                        lambda path: Path(path).resolve())
    monkeypatch.setattr(supervisor, "_has_worker_failure_capsule",
                        lambda *args, **kwargs: False)
    monkeypatch.setattr(
        supervisor, "write_failure_capsule",
        lambda path, **kwargs: Path(path))
    monkeypatch.setattr(supervisor.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(supervisor.time, "sleep", clock.sleep)
    return config, checkpoint, processes


def test_supervise_monitor_accepts_normal_complete_exit(monkeypatch, tmp_path):
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 1},
            {"status": "complete", "step": 2, "exit": 0},
        ]])
    result = supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert result.attempts == 1
    assert result.heartbeat.status == "complete"
    assert len(processes) == 1


def test_supervise_monitor_accepts_windows_redirector_child_pid(
        monkeypatch, tmp_path):
    child_pid = 52_001
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "preparing:worker-start", "step": 0,
             "heartbeat_pid": child_pid},
            {"status": "integrating", "step": 1,
             "heartbeat_pid": child_pid},
            {"status": "complete", "step": 2, "exit": 0,
             "heartbeat_pid": child_pid},
        ]])
    result = supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert result.attempts == 1
    assert result.heartbeat.status == "complete"
    assert result.heartbeat.pid == child_pid
    assert result.heartbeat.pid != processes[0].pid


def test_supervise_monitor_accepts_child_completion_before_first_poll(
        monkeypatch, tmp_path):
    child_pid = 52_001
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "complete", "step": 2, "exit": 0,
             "heartbeat_pid": child_pid},
        ]], publish_before_popen_returns=True)
    result = supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert result.attempts == 1
    assert result.heartbeat.status == "complete"
    assert result.heartbeat.pid == child_pid
    assert result.heartbeat.pid != processes[0].pid


def test_supervise_monitor_rejects_worker_pid_change(monkeypatch, tmp_path):
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 1,
             "heartbeat_pid": 52_001},
            {"status": "complete", "step": 2, "exit": 0,
             "heartbeat_pid": 52_002},
        ]])
    with pytest.raises(SupervisorError, match="heartbeat identity violation"):
        supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert len(processes) == 1


def test_mismatched_final_heartbeat_cannot_supply_recovery_checkpoint(
        monkeypatch, tmp_path):
    mismatched_checkpoint = tmp_path / "checkpoint.npz"
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 1,
             "heartbeat_pid": 52_001, "checkpoint": None},
            {"status": "complete", "step": 2, "exit": 0,
             "heartbeat_pid": 52_002,
             "checkpoint": mismatched_checkpoint},
        ]])
    validated = []

    def validate(path):
        validated.append(Path(path).resolve())
        return Path(path).resolve()

    import gpuwm.supervisor as supervisor
    monkeypatch.setattr(supervisor, "validate_manifest_checkpoint", validate)
    with pytest.raises(SupervisorError, match="heartbeat identity violation"):
        supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert validated == []
    assert len(processes) == 1


def test_existing_child_failure_capsule_is_matched_by_effective_pid(
        monkeypatch, tmp_path):
    child_pid = 52_001
    config, _, _ = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 1,
             "heartbeat_pid": child_pid},
            {"exit": 7},
        ]])
    checked = []

    def has_capsule(path, *, run_id, worker_pid):
        checked.append((Path(path), run_id, worker_pid))
        return True

    def unexpected_write(*args, **kwargs):
        raise AssertionError("supervisor overwrote the worker failure capsule")

    import gpuwm.supervisor as supervisor
    monkeypatch.setattr(supervisor, "_has_worker_failure_capsule", has_capsule)
    monkeypatch.setattr(supervisor, "write_failure_capsule", unexpected_write)
    with pytest.raises(SupervisorError, match="no durable manifest-valid"):
        supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert len(checked) == 1
    assert checked[0][2] == child_pid


def test_supervise_monitor_validates_final_heartbeat_regression(
        monkeypatch, tmp_path):
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 5},
            {"status": "complete", "step": 1, "exit": 0},
        ]])
    with pytest.raises(SupervisorError, match="outer_step moved backward"):
        supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert len(processes) == 1


def test_supervise_monitor_rejects_change_after_failed_heartbeat(
        monkeypatch, tmp_path):
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "failed", "step": 1},
            {"status": "complete", "step": 2, "exit": 0},
        ]])
    with pytest.raises(SupervisorError, match="terminal status failed changed"):
        supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert len(processes) == 1


def test_rejected_final_mutation_cannot_reuse_earlier_complete_heartbeat(
        monkeypatch, tmp_path):
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "complete", "step": 1,
             "heartbeat_pid": 52_001},
            {"status": "complete", "step": 2, "exit": 0,
             "heartbeat_pid": 52_001},
        ]])
    with pytest.raises(SupervisorError, match="terminal status complete changed"):
        supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert len(processes) == 1


def test_supervise_monitor_crash_resumes_in_fresh_process(monkeypatch,
                                                          tmp_path):
    config, checkpoint, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [
            [{"status": "integrating", "step": 2}, {"exit": 7}],
            [{"status": "integrating", "step": 3},
             {"status": "complete", "step": 4, "exit": 0}],
        ])
    # Passing an initial manifest-valid checkpoint proves every relaunch is a
    # resume and avoids any restart-from-scratch policy ambiguity.
    result = supervise_experiment(
        config, tmp_path / "out", restart=checkpoint,
        max_restarts=1, poll_seconds=0.05)
    assert result.attempts == 2
    assert all("--restart" in process.command for process in processes)


def test_supervise_monitor_crash_loop_hits_restart_cap(monkeypatch, tmp_path):
    config, checkpoint, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [
            [{"status": "integrating", "step": 1}, {"exit": 11}],
            [{"status": "integrating", "step": 1}, {"exit": 12}],
        ])
    with pytest.raises(SupervisorError, match="exhausted 1"):
        supervise_experiment(
            config, tmp_path / "out", restart=checkpoint,
            max_restarts=1, poll_seconds=0.05)
    assert len(processes) == 2


def test_supervise_monitor_kills_stale_integrating_worker(monkeypatch,
                                                          tmp_path):
    config, checkpoint, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 2},
        ]], clock_step=61.0)
    with pytest.raises(SupervisorError, match="became stale"):
        supervise_experiment(
            config, tmp_path / "out", restart=checkpoint,
            max_restarts=0, poll_seconds=0.05)
    assert processes[0].terminated


def test_supervise_monitor_slow_preparation_has_no_default_timeout(
        monkeypatch, tmp_path):
    config, _, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "preparing:load-config", "step": 0},
            {"status": "preparing:prepare-case", "step": 0},
            {"status": "complete", "step": 0, "exit": 0},
        ]], clock_step=500.0)
    result = supervise_experiment(config, tmp_path / "out", poll_seconds=0.05)
    assert result.attempts == 1
    assert not processes[0].terminated


def test_supervise_monitor_wedged_preparation_times_out_once(monkeypatch,
                                                             tmp_path):
    config, checkpoint, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "preparing:prepare-case", "step": 0},
        ]], clock_step=31.0)
    with pytest.raises(SupervisorError, match="refusing a deterministic"):
        supervise_experiment(
            config, tmp_path / "out", restart=checkpoint,
            max_restarts=3, prep_timeout_seconds=30.0, poll_seconds=0.05)
    assert len(processes) == 1
    assert processes[0].terminated


def test_supervise_monitor_detects_heartbeat_regression(monkeypatch, tmp_path):
    config, checkpoint, processes = _install_scripted_supervisor(
        monkeypatch, tmp_path, [[
            {"status": "integrating", "step": 2},
            {"status": "integrating", "step": 1},
        ]])
    with pytest.raises(SupervisorError, match="outer_step moved backward"):
        supervise_experiment(
            config, tmp_path / "out", restart=checkpoint,
            max_restarts=3, poll_seconds=0.05)
    assert len(processes) == 1
    assert processes[0].terminated


# --- directory-input identity ------------------------------------------
#
# The dual-run byte comparison is only as good as the binding between a
# run and its inputs.  Files are content-hashed; a *directory* input --
# in practice the static geography tree -- is bound by inventory
# (path/size/mtime) by default, which has one false-positive and one
# false-negative mode.  Both are gated here, at both mode values, because
# docs/public/DETERMINISM.md tells scientists exactly which one they get.


def _geography_tree(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "nested").mkdir()
    (root / "index").write_bytes(b"projection=regular_ll\n")
    (root / "nested" / "00001-01200.00001-01200").write_bytes(
        bytes(range(256)) * 4)
    return root


def _stamp(root: Path, mtime_ns: int) -> None:
    for child in sorted(root.rglob("*")):
        if child.is_file():
            os.utime(child, ns=(mtime_ns, mtime_ns))


@pytest.mark.parametrize(
    "mode, copies_agree",
    [("inventory", False), ("content", True)])
def test_directory_hash_and_a_byte_identical_copy(tmp_path, mode,
                                                  copies_agree):
    """A staged copy with fresh mtimes: inventory false-positives."""
    left = _geography_tree(tmp_path / "left")
    right = _geography_tree(tmp_path / "right")
    _stamp(left, 1_500_000_000_000_000_000)
    _stamp(right, 1_600_000_000_000_000_000)

    left_digest = supervisor._hash_directory_manifest(left, mode=mode)
    right_digest = supervisor._hash_directory_manifest(right, mode=mode)
    assert (left_digest == right_digest) is copies_agree


@pytest.mark.parametrize(
    "mode, edit_is_seen",
    [("inventory", False), ("content", True)])
def test_directory_hash_and_a_size_and_mtime_preserving_edit(tmp_path, mode,
                                                             edit_is_seen):
    """A same-size edit with the mtime restored: inventory misses it."""
    tree = _geography_tree(tmp_path / "geog")
    _stamp(tree, 1_500_000_000_000_000_000)
    before = supervisor._hash_directory_manifest(tree, mode=mode)

    target = tree / "nested" / "00001-01200.00001-01200"
    payload = bytearray(target.read_bytes())
    payload[17] ^= 0x01                      # one bit, same length
    target.write_bytes(bytes(payload))
    _stamp(tree, 1_500_000_000_000_000_000)  # and the mtime is put back

    after = supervisor._hash_directory_manifest(tree, mode=mode)
    assert (before != after) is edit_is_seen


def test_directory_hash_inventory_digest_is_unchanged_by_the_mode_argument(
        tmp_path):
    """Old inventory digests stay comparable: no domain prefix was added."""
    tree = _geography_tree(tmp_path / "geog")
    _stamp(tree, 1_500_000_000_000_000_000)
    expected = hashlib.sha256()
    for relative in ("index", "nested/00001-01200.00001-01200"):
        child = tree / relative
        expected.update(
            f"{relative}\0{child.stat().st_size}\0"
            f"{child.stat().st_mtime_ns}\n".encode("utf-8"))
    assert supervisor._hash_directory_manifest(tree, mode="inventory") == \
        expected.hexdigest()


@pytest.mark.parametrize("mode", ["inventory", "content"])
def test_resolved_input_hashes_records_the_algorithm_it_used(
        monkeypatch, tmp_path, mode):
    tree = _geography_tree(tmp_path / "geog")
    forcing = tmp_path / "forcing.grib2"
    forcing.write_bytes(b"GRIB")

    class _Case:
        def resolved_inputs(self):
            return (case_data.ResolvedInput(role="forcing", path=forcing),
                    case_data.ResolvedInput(role="geog_root", path=tree))

    monkeypatch.setattr(case_data, "load_experiment_case",
                        lambda _path: (None, _Case()))
    hashes = supervisor.resolved_input_hashes(
        tmp_path / "case.toml", directory_hash=mode)
    assert hashes[f"geog_root:{tree}"]["algorithm"] == \
        f"sha256-directory-{mode}"
    assert hashes[f"forcing:{forcing}"]["algorithm"] == "sha256"


@pytest.mark.parametrize(
    "requested, env, expected",
    [(None, None, "inventory"),
     (None, "content", "content"),
     ("content", None, "content"),
     ("inventory", "content", "inventory")])
def test_directory_hash_mode_resolution(monkeypatch, requested, env,
                                        expected):
    monkeypatch.delenv(supervisor.DIRECTORY_HASH_ENV, raising=False)
    if env is not None:
        monkeypatch.setenv(supervisor.DIRECTORY_HASH_ENV, env)
    assert supervisor.directory_hash_mode(requested) == expected


@pytest.mark.parametrize("bad", ["mtime", "sha256", ""])
def test_directory_hash_mode_refuses_an_unknown_mode(monkeypatch, bad):
    monkeypatch.delenv(supervisor.DIRECTORY_HASH_ENV, raising=False)
    with pytest.raises(ValueError, match="is not one of"):
        supervisor.directory_hash_mode(bad)
    monkeypatch.setenv(supervisor.DIRECTORY_HASH_ENV, bad)
    with pytest.raises(ValueError, match="environment variable"):
        supervisor.directory_hash_mode()
    with pytest.raises(ValueError, match="is not one of"):
        supervisor._hash_directory_manifest(Path("."), mode=bad)
