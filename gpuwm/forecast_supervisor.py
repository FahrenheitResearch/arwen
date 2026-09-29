"""Progress and bounded subprocess supervision for prepared forecasts."""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time
import uuid

from gpuwm import supervisor


# Two render joins (HALT_WAIT_SECONDS = 3 each) and their two-second
# interrupt grace periods can finish before the worker is forcibly ended.
STOP_GRACE_SECONDS = 10.0


class ForecastWatchdog:
    def __init__(self, command):
        # Preparation has no deadline, as gpuwm run has none by default: a
        # first run's kernel compilation and a large restore have no
        # measured price to bound them with.
        self.path = Path(command[command.index("--outdir") + 1]).resolve() / supervisor.HEARTBEAT_NAME
        self.run_id = str(uuid.uuid4())
        self.started_at = supervisor.utc_now()
        config = (Path(command[command.index("--experiment-config") + 1])
                  if "--experiment-config" in command else None)
        # The runner owns the named refusal for a missing input. Do not fail
        # between stage_begin and process launch while collecting telemetry.
        self.digest = (supervisor.config_digest(config)
                       if config is not None and config.is_file() else "")
        self.env = {"GPUWM_FORECAST_RUN_ID": self.run_id,
                    "GPUWM_FORECAST_STARTED_AT": self.started_at,
                    "GPUWM_FORECAST_CONFIG_DIGEST": self.digest}
        self.command = [command[0], "-m", __name__, *command[2:]]
        self.last = None
        self.last_signal = time.monotonic()
        self.history = supervisor.RollingStepWall()
        self.worker_pid = None

    def check(self, pid):
        now = time.monotonic()
        try:
            current = supervisor.read_heartbeat(self.path)
        except (OSError, ValueError):
            current = None
        if current is not None:
            self.worker_pid, error = supervisor._bind_attempt_heartbeat(
                current, run_id=self.run_id, config_digest=self.digest,
                started_at_utc=self.started_at, launch_pid=pid,
                effective_worker_pid=self.worker_pid)
            if error:
                return f"forecast heartbeat identity violation: {error}"
            if self.last is not None:
                error = supervisor._heartbeat_regression(self.last, current)
                if error:
                    return f"forecast heartbeat regression: {error}"
            if current != self.last:
                if (self.last is not None and self.last.status == "integrating"
                        and current.status == "integrating"
                        and current.outer_step > self.last.outer_step):
                    from datetime import datetime
                    self.history.add(
                        (datetime.fromisoformat(current.updated_at_utc)
                         - datetime.fromisoformat(self.last.updated_at_utc)).total_seconds())
                self.last = current
                self.last_signal = now
        status = "preparing:launch" if self.last is None else self.last.status
        if status == "complete":
            return None
        if status.startswith("preparing:"):
            return None
        # A write between two steps (``writing:``) is timed by the bytes it
        # declares, as finalization is: the step bound alone stopped a
        # finished 1132x906x55 streamed run while it wrote its last frame.
        if status.startswith(supervisor.WORK_SIZED_PREFIXES):
            bound = supervisor.finalization_stale_threshold_seconds(
                self.history.stale_threshold_seconds, self.last.work_bytes)
            if status.startswith("finalizing:finish-first-products"):
                from gpuwm.live_products import landing_render_wait_seconds

                bound += landing_render_wait_seconds()
        else:
            bound = self.history.stale_threshold_seconds
        if now - self.last_signal > bound:
            return (f"forecast stalled in {status}: no progress for "
                    f"{now - self.last_signal:.1f} s (bound {bound:.1f} s); "
                    "stopped the worker to release its GPU and prevent an indefinite wait")
        return None

    def failed(self):
        if self.last is not None:
            supervisor.write_heartbeat(self.path, dataclasses.replace(
                self.last, status="failed", work_bytes=None,
                updated_at_utc=supervisor.utc_now()))

    def terminate(self, process):
        if process.poll() is not None:
            return
        if os.name == "posix":
            import signal

            # Keep the terminal's process group. The worker catches SIGINT,
            # halts its renders and writes its own failure receipt first.
            try:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=STOP_GRACE_SECONDS)
                return
            except ProcessLookupError:
                return
            except subprocess.TimeoutExpired:
                pass
        if os.name == "nt":
            # A venv launcher owns a child interpreter on Windows. Stop the
            # owned process tree so its pipes and CUDA context also close.
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=15)
        supervisor._terminate_fresh_worker(process, timeout=2.0)


class ForecastHeartbeat(supervisor.RuntimeHeartbeat):
    #: The outer step of this process's first step beat, once published.
    entry_step = None

    def _write(self, status, *, work_bytes=None):
        # A process's first step beat announces entry, not completed work:
        # step 0 on a cold start, the checkpoint's step on a restart. The
        # kernel compilation inside the step after it is still preparation,
        # so steps are timed only once the step count moves past entry.
        if status == "integrating":
            if self.entry_step is None:
                self.entry_step = self.last_step
            if self.last_step <= self.entry_step:
                status = "preparing:first-step"
        super()._write(status, work_bytes=work_bytes)


def main(argv=None):
    if os.name == "posix":
        import signal

        # Detached launches can inherit SIGINT ignored. The parent's direct
        # stop request must still reach the runner's KeyboardInterrupt cleanup.
        signal.signal(signal.SIGINT, signal.default_int_handler)
    args = list(sys.argv[1:] if argv is None else argv)
    module, argv = args[0], args[1:]
    outdir = Path(argv[argv.index("--outdir") + 1])
    heartbeat = ForecastHeartbeat(
        outdir / supervisor.HEARTBEAT_NAME,
        run_id=os.environ["GPUWM_FORECAST_RUN_ID"],
        config_sha256=os.environ["GPUWM_FORECAST_CONFIG_DIGEST"],
        started_at_utc=os.environ["GPUWM_FORECAST_STARTED_AT"])

    def owns_record():
        try:
            return supervisor.read_heartbeat(heartbeat.path).run_id == heartbeat.run_id
        except (OSError, ValueError):
            return False
    # The runner claims a new output directory. Do not create it before that
    # claim or write a terminal record over an existing run after a refusal.
    try:
        code = importlib.import_module(module).main(argv, observer=heartbeat)
    except BaseException:
        if owns_record():
            heartbeat.failed()
        raise
    if owns_record():
        if code:
            heartbeat.failed()
        else:
            elapsed = heartbeat.model_elapsed_seconds
            # A checkpoint at the stop tick finalizes without a step callback.
            for path in (outdir / "progress.json", outdir / "evidence/progress.json"):
                if path.is_file():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    elapsed = payload.get("model_elapsed_seconds", elapsed)
                    break
            heartbeat.complete(elapsed)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
