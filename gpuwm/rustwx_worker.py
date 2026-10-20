"""Bounded orchestration for the native resident renderer (held candidate).

Start outside the model step loop. ``try_submit`` admits one path-only request;
it never waits for rendering and has no pending frame queue. A controller thread
drains both native pipes and enforces the request deadline while the caller runs.
The caller owns GPU admission, required-output validation and atomic publication.
No native result, by itself, declares a forecast's products complete.
"""
from __future__ import annotations

from collections import deque
import json
import math
import os
import selectors
import signal
import socket
import subprocess
import threading
import time
from typing import Mapping, Sequence


class ResidentRenderer:
    """Linux native worker, at most four CPUs and one unconsumed result.

    Both pipes use bounded framing. Diagnostics retain the last 64 lines per
    pipe; rendered records retain at most 1 MiB per request. Deadline/failure
    retires this process; restart belongs outside the forecast's step path.
    This does not acquire or release an external GPU lease.
    """

    schema = "gpuwm-rw-wrfbatch-worker-v1"
    line_bytes = 65536
    records_bytes = 1048576
    max_records = 1024

    def __init__(self, renderer: str, fixed: Sequence[str], *,
                 env: Mapping[str, str], cpus: Sequence[int],
                 host_mib: int = 512, startup_seconds: float = 10):
        chosen = tuple(sorted(set(cpus)))
        if not chosen or len(chosen) > 4 or not set(chosen) <= os.sched_getaffinity(0):
            raise ValueError("renderer needs 1..4 available CPUs")
        if host_mib <= 0 or not math.isfinite(startup_seconds) or startup_seconds <= 0:
            raise ValueError("host budget and startup deadline must be positive")
        self.cpus = chosen
        self._cv = threading.Condition()
        self._ready = None
        self._active = None
        self._result = None
        self._error = None
        self._closed = False
        self._pipes_closed = False
        self._host_bytes = host_mib * 1048576
        self._started = False
        self._deadline = time.monotonic() + startup_seconds
        self._tx = b""
        self._records = []
        self._record_bytes = 0
        self.diagnostics = {name: deque(maxlen=64) for name in ("stdout", "stderr")}
        child_env = dict(env)
        for name in ("RAYON_NUM_THREADS", "RUSTWX_RENDER_THREADS",
                     "RUSTWX_DERIVED_RENDER_THREADS", "OMP_NUM_THREADS"):
            child_env[name] = str(len(chosen))
        # taskset execs the renderer: no threaded preexec_fn and no helper child.
        self.argv = ["taskset", "--cpu-list", ",".join(map(str, chosen)), renderer,
                     "--worker", "--worker-host-mib", str(host_mib), *fixed]
        self.process = subprocess.Popen(self.argv, env=child_env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, bufsize=0)
        self._wake_r, self._wake_w = socket.socketpair()
        self._wake_r.setblocking(False)
        self._wake_w.setblocking(False)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            os.set_blocking(stream.fileno(), False)
        self._thread = threading.Thread(target=self._drain, name="resident-renderer", daemon=True)
        self._thread.start()
        try:
            with self._cv:
                while self._ready is None:
                    self._check()
                    self._cv.wait(timeout=0.05)
        except BaseException:
            self.close()
            raise

    @property
    def ready(self):
        return dict(self._ready)

    def _check(self):
        if self._error is not None:
            raise RuntimeError(self._error)
        if self._closed:
            raise RuntimeError("renderer is closed")

    def _wake(self):
        try:
            self._wake_w.send(b"x")
        except BlockingIOError:
            pass

    def try_submit(self, request: Mapping, *, timeout_seconds: float) -> bool:
        """False means backpressure: retain/retry metadata at the owning scheduler.

        No request is copied/encoded when busy. There is no internal waiting
        frame queue, and completed results occupy the admission slot until taken.
        """
        if not self._cv.acquire(blocking=False):
            return False
        try:
            self._check()
            if self._active is not None or self._result is not None:
                return False
            if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
                raise ValueError("request deadline must be positive")
            if request.get("op") != self._ready["request_mode"]:
                raise ValueError("request does not match fixed renderer mode")
            mode = request["op"]
            allowed = ({"op", "id", "input", "out_dir"} if mode == "ensemble_products"
                       else {"op", "id", "store_root", "out_dir", "inputs", "from_import", "save_import"})
            if len(request) > len(allowed) or any(key not in allowed for key in request):
                raise ValueError("only native path/id metadata belongs in a request")
            size = 0
            for key, value in request.items():
                if key == "inputs":
                    if not isinstance(value, (list, tuple)) or len(value) > 64:
                        raise ValueError("native worker accepts at most 64 input paths")
                    values = value
                else:
                    values = (value,)
                for item in values:
                    if item is None and key in ("from_import", "save_import"):
                        continue
                    if not isinstance(item, str):
                        raise ValueError("worker metadata must contain string paths/ids")
                    size += len(item)
                    if size > self.line_bytes:
                        raise ValueError("request exceeds native 65536-byte framing")
            ident = request.get("id")
            if not isinstance(ident, str) or not 1 <= len(ident.encode()) <= 256:
                raise ValueError("request needs a 1..256-byte id")
            raw = (json.dumps(dict(request), separators=(",", ":")) + "\n").encode()
            if len(raw) > self.line_bytes:
                raise ValueError("request exceeds native 65536-byte framing")
            self._active = ident
            self._started = False
            self._records = []
            self._record_bytes = 0
            self._tx = raw
            self._deadline = time.monotonic() + timeout_seconds
            self._wake()
            return True
        finally:
            self._cv.release()

    def take_result(self):
        """Return an available native outcome/records, without waiting."""
        with self._cv:
            self._check()
            result, self._result = self._result, None
            return result

    def wait_result(self, *, timeout_seconds: float):
        """Finish/drain helper; use outside the model step loop."""
        until = time.monotonic() + timeout_seconds
        with self._cv:
            while True:
                self._check()
                if self._result is not None:
                    result, self._result = self._result, None
                    return result
                left = until - time.monotonic()
                if left <= 0:
                    raise TimeoutError("caller stopped waiting; native request deadline remains active")
                self._cv.wait(timeout=left)

    def _line(self, kind, raw):
        line = raw.decode("utf-8", errors="replace")
        if kind == "stdout" and line.startswith("WORKER\t"):
            value = json.loads(line.split("\t", 1)[1])
            if value.get("schema") != self.schema:
                raise RuntimeError("unexpected native worker schema")
            event = value.get("event")
            if event == "ready" and self._ready is None:
                if value.get("pid") != self.process.pid or value.get("max_inflight") != 1:
                    raise RuntimeError("native worker identity/admission mismatch")
                if (value.get("host_budget_bytes") != self._host_bytes
                        or value.get("render_threads") != str(len(self.cpus))
                        or value.get("request_mode") not in ("render", "ensemble_products")):
                    raise RuntimeError("native worker resource/mode mismatch")
                self._ready = value
                self._deadline = None
            elif event == "started" and self._active is not None and value.get("id") == self._active and not self._started:
                self._started = True
            elif event == "result" and self._started and value.get("id") == self._active:
                self._result = {"native": value, "records": self._records}
                self._active = None
                self._records = []
                self._deadline = None
                if value.get("exit_code") != 0:
                    raise RuntimeError("native render failed: " + str(value.get("error")))
            else:
                raise RuntimeError("unexpected native worker event: " + line[:2048])
            self._cv.notify_all()
        elif kind == "stdout" and line.startswith("RENDERED"):
            if self._active is None:
                raise RuntimeError("rendered record outside an active request")
            self._record_bytes += len(raw)
            if self._record_bytes > self.records_bytes or len(self._records) >= self.max_records:
                raise RuntimeError("rendered records exceed per-request 1 MiB/1024-record bound")
            self._records.append(line)
        else:
            self.diagnostics[kind].append(line[-2048:])

    def _drain(self):
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self._wake_r, selectors.EVENT_READ, "wake")
                for name in buffers:
                    selector.register(getattr(self.process, name), selectors.EVENT_READ, name)
                writing = False
                while True:
                    with self._cv:
                        if self._closed:
                            break
                        if self._deadline is not None and time.monotonic() >= self._deadline:
                            raise TimeoutError("native renderer startup/request deadline exceeded")
                        if self._tx and not writing:
                            selector.register(self.process.stdin, selectors.EVENT_WRITE, "stdin")
                            writing = True
                        delay = 1 if self._deadline is None else min(1, max(0, self._deadline-time.monotonic()))
                    for key, _ in selector.select(delay):
                        with self._cv:
                            if key.data == "wake":
                                self._wake_r.recv(4096)
                            elif key.data == "stdin":
                                count = os.write(key.fd, self._tx)
                                self._tx = self._tx[count:]
                                if not self._tx:
                                    selector.unregister(self.process.stdin)
                                    writing = False
                            else:
                                chunk = os.read(key.fd, 65536)
                                if not chunk:
                                    selector.unregister(key.fileobj)
                                    if buffers[key.data]:
                                        raise RuntimeError("unterminated native response")
                                    if key.data == "stdout":
                                        raise RuntimeError("native renderer exited")
                                    continue
                                buf = buffers[key.data]
                                buf.extend(chunk)
                                while b"\n" in buf:
                                    line, _, rest = buf.partition(b"\n")
                                    if len(line) + 1 > self.line_bytes:
                                        raise RuntimeError("native response exceeds bounded framing")
                                    self._line(key.data, line)
                                    buf[:] = rest
                                if len(buf) >= self.line_bytes:
                                    raise RuntimeError("native response exceeds bounded framing")
        except BaseException as error:
            with self._cv:
                self._error = str(error)
                self._cv.notify_all()
        finally:
            # The native renderer is a single process with Rust threads. Stop its
            # session group before waiting for/reaping the actual renderer PID.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait()
            with self._cv:
                self._cv.notify_all()

    def close(self):
        """Cancel unfinished work and reap the renderer before returning."""
        with self._cv:
            if self._pipes_closed:
                return
            self._closed = True
            self._wake()
            self._cv.notify_all()
        self._thread.join(timeout=2)
        if self._thread.is_alive():
            raise RuntimeError("renderer cleanup did not finish; retain external admission")
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr,
                       self._wake_r, self._wake_w):
            stream.close()
        self._pipes_closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
