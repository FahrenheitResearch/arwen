"""Replay never holds a member callback or loses a committed clock."""
from threading import Event, Thread
import time

import numpy as np
import pytest

from gpuwm.ensemble.product_consumer import DiagnosticProductConsumer


def test_busy_replay_does_not_block_next_hour_and_retains_out_of_order_clocks():
    consumer = DiagnosticProductConsumer(np)
    entered, release = Event(), Event()
    completed = []
    def slow():
        entered.set()
        assert release.wait(5)
        completed.append(2)
    consumer.submit(("d01", "hour2"), device=0, replay=slow)
    assert entered.wait(5)
    started = time.perf_counter()
    consumer.submit(("d01", "hour1"), device=0, replay=lambda: completed.append(1))
    consumer.submit(("d02", "hour1"), device=0, replay=lambda: completed.append(3))
    assert time.perf_counter() - started < 1
    assert consumer.receipt()["queued_device_payload_bytes"] == 0
    assert consumer.receipt()["active_replay_limit"] == 1
    release.set()
    consumer.close()
    assert completed == [2, 1, 3]
    assert consumer.receipt()["closed"] and consumer.receipt()["pending_frames"] == 0


def test_failed_replay_propagates_to_forecast_and_does_not_publish_later_frames():
    consumer = DiagnosticProductConsumer(np)
    completed = []
    entered, release = Event(), Event()
    def fail():
        entered.set()
        assert release.wait(5)
        raise ValueError("diagnostic identity mismatch")
    consumer.submit(("d01", "hour0"), device=0, replay=fail)
    assert entered.wait(5)
    consumer.submit(("d01", "hour1"), device=0, replay=lambda: completed.append(1))
    release.set()
    with pytest.raises(RuntimeError, match="diagnostic identity mismatch"):
        consumer.close()
    with pytest.raises(RuntimeError, match="diagnostic identity mismatch"):
        consumer.check()
    assert completed == []


def test_cancel_finishes_owned_active_stream_and_leaves_pending_spills_unconsumed():
    consumer = DiagnosticProductConsumer(np)
    entered, release = Event(), Event()
    completed = []
    def replay():
        entered.set()
        assert release.wait(5)
        completed.append(0)
    consumer.submit(("d01", "hour0"), device=0, replay=replay)
    assert entered.wait(5)
    with pytest.raises(ValueError, match="already scheduled"):
        consumer.submit(("d01", "hour0"), device=0, replay=replay)
    consumer.submit(("d01", "hour1"), device=0, replay=lambda: completed.append(1))
    closer = Thread(target=lambda: consumer.close(cancel=True))
    closer.start()
    deadline = time.monotonic() + 5
    while not consumer.receipt()["cancelled"] and time.monotonic() < deadline:
        time.sleep(.001)
    assert consumer.receipt()["cancelled"] and closer.is_alive()
    release.set()
    closer.join(5)
    assert not closer.is_alive() and completed == [0]
    assert consumer.receipt()["cancelled"] and consumer.receipt()["pending_frames"] == 0


def test_product_failure_is_checked_at_original_member_step_boundary():
    from gpuwm.ensemble.progress import EnsembleProgressAdapter
    def check():
        raise RuntimeError("aggregate renderer failed")
    adapter = EnsembleProgressAdapter(None, member_ids=(0, 1), run_seconds=60,
        check_products=check)
    with pytest.raises(RuntimeError, match="aggregate renderer failed"):
        adapter.callback_for_member(1)(model_elapsed_seconds=12, outer_step=1)


def test_last_member_timestamp_names_forecast_step_before_products_close():
    from gpuwm.ensemble.progress import EnsembleProgressAdapter
    adapter = EnsembleProgressAdapter(None, member_ids=(0, 1), run_seconds=60)
    first = adapter.callback_for_member(0)
    second = adapter.callback_for_member(1)
    first(model_elapsed_seconds=60, outer_step=5, phase="post-d01-sync")
    assert adapter.receipt()["last_member_forecast_step_at"] is None
    second(model_elapsed_seconds=60, outer_step=5, backend="native_member_batched")
    recorded = adapter.receipt()["last_member_forecast_step_at"]
    first.complete(60)
    second.complete(60)
    assert adapter.receipt()["last_member_forecast_step_at"] == recorded
    second.restarting("original member checkpoint")
    assert adapter.receipt()["last_member_forecast_step_at"] is None


def test_default_native_consumer_never_opens_a_cuda_owner():
    class ForecastArrayModule:
        @property
        def cuda(self):
            pytest.fail("CPU replay touched the forecast CUDA runtime")
    consumer = DiagnosticProductConsumer(ForecastArrayModule())
    finished = []
    consumer.submit(("d01", "hour1"), device=3, replay=lambda: finished.append(1))
    consumer.close()
    assert finished == [1]
    assert not consumer.receipt()["gpu_replay"]
    assert consumer.receipt()["native_stream_priorities"] == {}


def test_cancel_reaps_owned_cpu_process_without_running_next_frame(tmp_path):
    import os
    import sys
    consumer = DiagnosticProductConsumer()
    launched = Event()
    def replay():
        launched.set()
        consumer.run_owned([sys.executable, "-c", "import time; time.sleep(60)"],
            environment=dict(os.environ), log_path=tmp_path / "child.log")
    consumer.submit(("d01", "hour0"), device=0, replay=replay)
    assert launched.wait(5)
    deadline = time.monotonic() + 5
    while consumer._process is None and time.monotonic() < deadline:
        time.sleep(.001)
    process = consumer._process
    assert process is not None
    consumer.close(cancel=True)
    assert process.poll() is not None and consumer._process is None


def test_owned_cpu_failure_propagates_exact_child_exit_and_is_reaped(tmp_path):
    import os
    import sys
    consumer = DiagnosticProductConsumer()
    consumer.submit(("d01", "hour0"), device=0,
        replay=lambda: consumer.run_owned([sys.executable, "-c", "raise SystemExit(7)"],
            environment=dict(os.environ), log_path=tmp_path / "child.log"))
    with pytest.raises(RuntimeError, match="exited 7"):
        consumer.close()
    assert consumer._process is None


def test_owned_cpu_normal_wait_is_bounded_and_reaps_the_child(tmp_path):
    import os
    import sys
    consumer = DiagnosticProductConsumer()
    consumer.PROCESS_TIMEOUT_SECONDS = .1
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="product process exceeded"):
        consumer.run_owned([sys.executable, "-c", "import time; time.sleep(60)"],
            environment=dict(os.environ), log_path=tmp_path / "timeout-child.log")
    assert consumer._process is None
    assert time.monotonic() - started < 6


@pytest.mark.skipif(__import__("sys").platform != "linux", reason="Linux owned process-group cleanup")
def test_successful_leader_exit_reaps_a_live_grandchild_that_ignores_term(tmp_path):
    import json
    import os
    from pathlib import Path
    import signal
    import sys
    ready = tmp_path / "grandchild-ready.json"
    child_script = "\n".join((
        "import json, os, pathlib, signal, sys, time",
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({'child': os.getpid(), 'leader': os.getppid()}))",
        "time.sleep(60)",
    ))
    leader_script = "\n".join((
        "import pathlib, subprocess, sys, time",
        "child = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])",
        "deadline = time.monotonic() + 5",
        "while not pathlib.Path(sys.argv[2]).exists() and time.monotonic() < deadline: time.sleep(.01)",
        "if not pathlib.Path(sys.argv[2]).exists(): raise SystemExit(3)",
    ))
    consumer = DiagnosticProductConsumer()
    try:
        consumer.run_owned([sys.executable, "-c", leader_script, child_script, str(ready)],
            environment=dict(os.environ), log_path=tmp_path / "orphan-child.log")
        assert consumer._process is None
        child_pid = json.loads(ready.read_text())["child"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = Path(f"/proc/{child_pid}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            time.sleep(.02)
        else:
            pytest.fail("live grandchild survived the owned replay leader's successful exit")
    finally:
        if ready.is_file():
            try:
                os.killpg(json.loads(ready.read_text())["leader"], signal.SIGKILL)
            except ProcessLookupError:
                pass
