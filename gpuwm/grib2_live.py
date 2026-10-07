"""Queue committed history files for native GRIB2 export without blocking a run.

Python owns file names, subprocess lifetime, and progress only. All history
reading, field calculations, temporal windows, packing, and archives are Rust.
"""
from __future__ import annotations

import json
from pathlib import Path
import queue
import subprocess
import threading
from typing import Callable

from gpuwm import grib2_export


def validate_member_contract(enabled: bool, request) -> None:
    """Keep different members from sharing one domain/time export identity."""
    if enabled and request is not None and request.members > 1:
        raise ValueError(
            "--grib2 / run_options.grib2 needs separate member identities and "
            "archive paths for an ensemble. The current domain/time export "
            "identity would collide across members, and default ensembles "
            "do not retain individual history files. Next: omit live GRIB2 "
            "export, or retain member histories and export each member "
            "separately with export-grib2.")


def find_exporter() -> Path:
    executable = grib2_export.BINARY.find()
    if executable is None:
        raise RuntimeError(grib2_export.BINARY.remedy())
    valid, reason = grib2_export.BINARY.probe(executable)
    if not valid:
        raise RuntimeError(reason)
    return executable


def request_for(inputs, out: Path, *, mode: str = "run", archive: bool = False) -> dict:
    """The same defaults as the standalone export front door."""
    return {
        "schema": grib2_export.REQUEST_SCHEMA, "mode": mode,
        "inputs": [str(Path(path).resolve()) for path in inputs],
        "out": str(out.resolve()), "fields": "standard", "packing": "complex",
        "definitions": "woof", "winds": "grid",
        "bits": 20, "levels": list(range(100, 1001, 25)),
        "domains": [], "times": [], "start": None, "end": None,
        "zip": archive, "threads": None,
        # Spec D30: on the forecast's own card when it has room, else CPU.
        "post_device": "auto", "gust": "auto",
        "extrema_interval_seconds": None, "composite_level_type": 200,
    }


class _Progress:
    def __init__(self, report: Callable[[dict], None]):
        self.report = report

    def write(self, text: str) -> int:
        for line in text.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                self.report(event)
        return len(text)

    def flush(self) -> None:
        pass


class LiveGrib2Export:
    """One serial exporter queue, independent of map rendering.

    Per-domain writer callbacks preserve frame order. The first append from
    each history folder includes its existing committed files once; later
    requests contain only the new path. Rust retains temporal context and
    recorded file identities durably.
    """

    def __init__(self, out: Path, *, report=None, warn=None, executable=None,
                 invoke=None, initial_inputs=()):
        self.out = Path(out).resolve()
        self._report = report or (lambda event: None)
        self._warn = warn or (lambda code, message, **fields: None)
        self._exe = executable or find_exporter()
        self._invoke = invoke or grib2_export.invoke
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self._closed = False
        self._process: subprocess.Popen | None = None
        self._seen: set[str] = set()
        self._initial_inputs = [str(Path(path).resolve()) for path in initial_inputs]
        self._scanned_parents: set[Path] = set()
        self._errors: list[str] = []
        self._completed = 0
        self._finalized = False
        self._worker = threading.Thread(target=self._work, name="grib2-export", daemon=True)
        self._worker.start()

    def frame_committed(self, *, domain: int, valid_time, path) -> bool:
        path = str(Path(path).resolve())
        valid = valid_time.isoformat() if hasattr(valid_time, "isoformat") else str(valid_time)
        with self._lock:
            if self._closed or path in self._seen:
                return False
            self._seen.add(path)
            self._queue.put((valid, int(domain), path))
        return True

    def _set_process(self, process) -> None:
        with self._lock:
            self._process = process
            halted = self._halt.is_set()
        if process is not None and halted and process.poll() is None:
            process.terminate()  # only the explicitly owned Popen PID

    def _run(self, request: dict) -> bool:
        if self._halt.is_set():
            return False
        code = self._invoke(self._exe, request, as_json=True,
                            stream=_Progress(self._report),
                            on_process=self._set_process, cancel_event=self._halt)
        if self._halt.is_set():
            return False
        if code:
            raise RuntimeError(f"native GRIB2 exporter exited {code}")
        return True

    def _work(self) -> None:
        while True:
            entry = self._queue.get()
            try:
                if entry is None:
                    break
                if self._halt.is_set():
                    continue
                _, _, path = entry
                parent = Path(path).parent
                context = []
                if parent not in self._scanned_parents:
                    self._scanned_parents.add(parent)
                    # Direct siblings only. Archived forecast attempts live
                    # below other folders and must never enter this attempt.
                    context = [str(p.resolve()) for p in sorted(parent.glob("wrfout_d*"))
                               if p.is_file() and all(marker not in p.name
                                   for marker in (".tmp", ".part", ".gz"))]
                inputs = list(dict.fromkeys([*self._initial_inputs, *context, path]))
                self._initial_inputs = []
                try:
                    if self._run(request_for(inputs, self.out, mode="append")):
                        self._completed += 1
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    self._errors.append(message)
                    self._warn("grib2_export_failed", message, frame=path)
            finally:
                self._queue.task_done()
        if not self._halt.is_set() and not self._errors and self._completed:
            try:
                self._finalized = self._run(request_for([], self.out, mode="finalize", archive=True))
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                self._errors.append(message)
                self._warn("grib2_export_failed", message)

    def _close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(None)

    def stop(self) -> dict:
        """Finish successful queued frames and build the archive in Rust."""
        self._close()
        self._worker.join()
        if self._errors:
            raise RuntimeError("GRIB2 export incomplete: " + self._errors[0])
        return self.summary()

    def halt(self, timeout: float = 10.0) -> dict:
        """Cancel queued work and stop only the exporter process we started."""
        self._halt.set()
        self._close()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            except ProcessLookupError:
                pass
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            self._warn("grib2_export_failed", "GRIB2 export did not stop before its bounded wait")
        return self.summary()

    def summary(self) -> dict:
        return {"out": str(self.out), "queued_frames": len(self._seen),
                "completed_frames": self._completed, "finalized": self._finalized,
                "halted": self._halt.is_set(), "errors": list(self._errors),
                "ended": not self._worker.is_alive()}
