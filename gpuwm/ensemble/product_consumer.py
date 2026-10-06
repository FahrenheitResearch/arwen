"""One owned replay consumer, independent of member history callbacks."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime, timezone
import threading
import time
import os
import signal
import subprocess


class DiagnosticProductConsumer:
    """Drain durable diagnostic spills on one spare CPU owner.

    Pending entries contain only a key, device and replay callback. They never
    retain model arrays. Only one replay task owns its bounded host buffers;
    the rest of the backlog remains in the producer's diagnostic spill files.
    A condition signal coalesces wakeups without dropping any committed hour.
    """

    # One hour's product replay must finish or fail within a bounded drain.
    # A hung native subprocess cannot hold finalization indefinitely.
    PROCESS_TIMEOUT_SECONDS = 900

    def __init__(self, array_module=None, *, gpu_replay=False):
        self.xp = array_module
        self.gpu_replay = bool(gpu_replay)
        self._condition = threading.Condition()
        self._pending = OrderedDict()
        self._active = None
        self._completed = set()
        self._failure = None
        self._closing = False
        self._cancelled = False
        self._thread = None
        self._streams = {}
        self._stream_priorities = {}
        self._rows = []
        self._cpu_priority = None
        self._cpu_affinity = None
        self._process = None

    def run_owned(self, command, *, environment, log_path):
        """Wait for one isolated CPU replay and reap its complete process group."""
        with self._condition:
            if self._cancelled:
                raise RuntimeError("ensemble product consumer was cancelled")
            if self._process is not None:
                raise RuntimeError("ensemble product consumer already owns a child")
            with open(log_path, "wb") as log:
                process = subprocess.Popen(command, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
            self._process = process
        try:
            try:
                code = process.wait(timeout=self.PROCESS_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"isolated ensemble product process exceeded {self.PROCESS_TIMEOUT_SECONDS}s; log: {log_path}") from error
            if code:
                raise RuntimeError(f"isolated ensemble product process exited {code}; log: {log_path}")
            with self._condition:
                if self._cancelled:
                    raise RuntimeError("ensemble product consumer was cancelled")
        finally:
            self._stop_process(process)
            with self._condition:
                self._process = None

    @staticmethod
    def _stop_process(process):
        if os.name == "posix":
            # The leader can exit after spawning a renderer. Its owned group
            # still needs termination even when Popen has already reaped it.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
        if os.name == "posix":
            # A descendant can ignore TERM independently of the exited leader.
            # No process outside the session created by run_owned is signaled.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def check(self):
        with self._condition:
            if self._failure is not None:
                raise RuntimeError(f"ensemble product consumer failed: {self._failure}") from self._failure

    def wait_idle(self, *, timeout=60):
        """Reach a bounded checkpoint barrier without closing later admission.

        ``timeout=None`` waits for every scheduled hour. Each replay is still
        bounded by PROCESS_TIMEOUT_SECONDS, so the wait cannot hang on a stuck
        native child; it is what a roster-completeness check needs.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("ensemble checkpoint drain timeout must be nonnegative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._pending or self._active is not None:
                self.check()
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("ensemble checkpoint product drain exceeded its bounded hold")
                self._condition.wait(remaining)
            self.check()

    def restore_completed(self, keys):
        """Retain published hour identities in a fresh process schedule."""
        with self._condition:
            if self._thread is not None or self._pending or self._active is not None:
                raise ValueError("ensemble product schedule must be restored before admission")
            self._completed.update(tuple(key) for key in keys)

    def submit(self, key, *, device, replay):
        """Commit a reference to a durable spill without waiting for replay."""
        with self._condition:
            self.check()
            if self._closing:
                raise RuntimeError("ensemble product consumer is closed")
            if key in self._pending or key == self._active or key in self._completed:
                raise ValueError("ensemble product frame was already scheduled")
            self._pending[key] = (int(device), replay, time.perf_counter(),
                datetime.now(timezone.utc).isoformat())
            if self._thread is None:
                self._thread = threading.Thread(target=self._run,
                    name="gpuwm-ensemble-products", daemon=True)
                self._thread.start()
            self._condition.notify_all()

    def _device(self, device):
        if not self.gpu_replay:
            return nullcontext()
        cuda = getattr(self.xp, "cuda", None)
        return nullcontext() if cuda is None else cuda.Device(device)

    def _stream(self, device):
        if not self.gpu_replay:
            return nullcontext()
        cuda = getattr(self.xp, "cuda", None)
        if cuda is None:
            return nullcontext()
        if device not in self._streams:
            # CUDA clamps an out-of-range positive priority to the lowest
            # supported priority. Smaller numbers mean higher priority.
            stream = cuda.Stream(non_blocking=True, priority=(1 << 31) - 1)
            self._streams[device] = stream
            self._stream_priorities[device] = int(stream.priority)
        return self._streams[device]

    def _record_failure(self, error):
        visited = set()
        current = error
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            current.__traceback__ = None
            current = current.__cause__ or current.__context__
        with self._condition:
            if self._failure is None:
                self._failure = error
            self._cancelled = self._closing = True
            self._pending.clear()
            self._active = None
            self._condition.notify_all()

    def _run(self):
        try:
            if hasattr(os, "setpriority") and hasattr(os, "PRIO_PROCESS"):
                task = threading.get_native_id()
                self._cpu_priority = max(10, os.getpriority(os.PRIO_PROCESS, task))
                os.setpriority(os.PRIO_PROCESS, task, self._cpu_priority)
            if hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity"):
                task = threading.get_native_id()
                self._cpu_affinity = sorted(os.sched_getaffinity(task))[-2:]
                os.sched_setaffinity(task, self._cpu_affinity)
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._pending or self._closing)
                    if self._cancelled or not self._pending:
                        return
                    key, (device, replay, enqueued, enqueued_at) = self._pending.popitem(last=False)
                    self._active = key
                started = time.perf_counter()
                started_at = datetime.now(timezone.utc).isoformat()
                with self._device(device), self._stream(device) as stream:
                    replay()
                    synchronize = getattr(stream, "synchronize", None)
                    if synchronize is not None:
                        synchronize()
                with self._condition:
                    self._completed.add(key)
                    self._rows.append({"frame": list(key), "device": device,
                        "enqueued_at": enqueued_at, "started_at": started_at,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "queue_wait_seconds": started - enqueued,
                        "wall_seconds": time.perf_counter() - started})
                    self._active = None
                    self._condition.notify_all()
        except BaseException as error:
            self._record_failure(error)
        finally:
            # Synchronize owned streams even when a replay raises, then drop
            # their ownership. No model producer stream is synchronized here.
            for device, stream in tuple(self._streams.items()):
                try:
                    with self._device(device):
                        stream.synchronize()
                except BaseException as error:
                    self._record_failure(error)
            self._streams.clear()

    def close(self, *, cancel=False):
        with self._condition:
            self._closing = True
            if cancel:
                self._cancelled = True
                self._pending.clear()
            self._condition.notify_all()
            thread = self._thread
            process = self._process if cancel else None
        if process is not None:
            self._stop_process(process)
        if thread is not None:
            thread.join()
        if not cancel:
            self.check()

    def receipt(self):
        with self._condition:
            return {"schema": "gpuwm-ensemble-product-consumer.v1",
                "execution": ("owned nonblocking CUDA reference stream, lowest priority"
                    if self.gpu_replay else "owned isolated Rust CPU replay, no CUDA owner"),
                "gpu_replay": self.gpu_replay,
                "active_replay_limit": 1, "queued_device_payload_bytes": 0,
                "native_thread_niceness": self._cpu_priority,
                "native_thread_cpu_affinity": self._cpu_affinity,
                "native_stream_priorities": dict(self._stream_priorities),
                "pending_frames": len(self._pending), "active_frame": self._active,
                "closed": self._closing, "cancelled": self._cancelled,
                "frames": list(self._rows)}
