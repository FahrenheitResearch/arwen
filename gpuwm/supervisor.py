"""Fresh-process forecast supervision and durable run progress.

The supervisor process never imports CuPy.  It owns the physical-GPU lock,
preflights active compute processes through ``nvidia-smi``, launches a fresh
Python worker, and watches an atomically published heartbeat.  A CUDA-fatal
condition is terminal for that worker process.  Recovery, when possible, is a
new process restored only from the most recent durable manifest-valid restart;
there is no in-process retry path.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import errno
import hashlib
import io
import itertools
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


HEARTBEAT_SCHEMA = "gpuwm.run-progress/v1"
FAILURE_CAPSULE_SCHEMA = "gpuwm.failure-capsule/v1"
HEARTBEAT_NAME = "run-progress.json"
FAILURE_CAPSULE_NAME = "failure-capsule.json"
COMPUTE_MEMORY_THRESHOLD_MIB = 64
MICROPHYSICS_TRANSITION_RECEIPT_NAME = "microphysics-transitions.json"

# Windows sharing violations are normally transient (the supervisor, an
# editor, or an indexer has the old publication open without FILE_SHARE_DELETE).
# Retry for at most 0.50 s total, then let durable artifacts fail loudly while
# heartbeat callers quarantine their unique temporary and keep the worker up.
_REPLACE_BACKOFF_SECONDS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.19)
_TEMP_COUNTER = itertools.count()

_HEARTBEAT_FIELDS = frozenset({
    "schema", "run_id", "config_digest", "pid", "started_at_utc",
    "updated_at_utc", "status", "model_elapsed_seconds", "outer_step",
    "last_durable_wrfout", "last_checkpoint",
})
_CUDA_FATAL_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"device(?: |-)lost", r"cudaErrorDeviceLost", r"illegal(?: memory)? address",
    r"cudaErrorIllegalAddress", r"cuda.error.illegal.address",
    r"unspecified launch failure",
    r"cudaErrorLaunchFailure", r"launch failure", r"context is destroyed",
))


class SupervisorError(RuntimeError):
    """The supervised forecast could not safely continue."""


class GPUPreflightError(SupervisorError):
    """GPU identity/process state could not be proven exclusive."""


class GPUAlreadyLockedError(GPUPreflightError):
    """Another gpuwm supervisor owns the UUID-keyed lock."""


class CheckpointValidationError(SupervisorError):
    """A proposed recovery file is not manifest-valid."""


def utc_now() -> str:
    """UTC ISO-8601 timestamp with an explicit ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability on POSIX; Windows has no dir fsync."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_file(path: str | Path) -> Path:
    """Flush an already closed file through the OS durability boundary."""
    path = Path(path)
    # Windows' CRT ``_commit`` (used by os.fsync) rejects a read-only file
    # descriptor with EBADF.  Reopen read/write without changing contents.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())
    return path


def unique_temp_path(path: str | Path, *, hidden: bool = False) -> Path:
    """Return a per-writer temporary name (PID + process-local counter)."""
    path = Path(path)
    prefix = "." if hidden else ""
    return path.with_name(
        f"{prefix}{path.name}.tmp.{os.getpid()}.{next(_TEMP_COUNTER)}")


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Replace after bounded Windows sharing-violation backoff (0.50 s)."""
    for delay in (*_REPLACE_BACKOFF_SECONDS, None):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def replace_file_with_retry(source: str | Path,
                            destination: str | Path) -> Path:
    """Fail-loud bounded-retry atomic replace for durable publications."""
    source = Path(source)
    destination = Path(destination)
    _replace_with_retry(source, destination)
    return destination


def atomic_write_json(path: str | Path, payload: dict[str, Any], *,
                      _before_replace: Callable[[Path], None] | None = None,
                      _quarantine_on_permission_error: bool = False
                      ) -> Path:
    """Publish JSON through tmp + flush + fsync + ``os.replace``.

    ``_before_replace`` is a deterministic kill/fault-injection seam for the
    CPU atomicity test.  Production callers never pass it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = unique_temp_path(path)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode("utf-8")
    with temp.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if _before_replace is not None:
        _before_replace(temp)
    try:
        replace_file_with_retry(temp, path)
    except PermissionError:
        if not _quarantine_on_permission_error:
            raise
        # A stale heartbeat is safer than terminating a healthy CUDA worker
        # because a reader held run-progress.json open for >0.50 s.
        quarantine_file(temp, reason="heartbeat-sharing-violation")
        return path
    _fsync_directory(path.parent)
    return path


def quarantine_file(path: str | Path, *, reason: str = "incomplete",
                    quarantine_dir: str | Path | None = None) -> Path | None:
    """Atomically move one orphan/incomplete artifact out of publication."""
    path = Path(path)
    if not path.exists():
        return None
    directory = (Path(quarantine_dir) if quarantine_dir is not None
                 else path.parent / ".quarantine")
    directory.mkdir(parents=True, exist_ok=True)
    safe_reason = re.sub(r"[^A-Za-z0-9_.-]+", "-", reason).strip("-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    target = directory / f"{path.name}.{safe_reason}.{stamp}.{os.getpid()}"
    replace_file_with_retry(path, target)
    _fsync_directory(directory)
    return target


def atomic_publish_file(
        final_path: str | Path, producer: Callable[[Path], None],
        validator: Callable[[Path], None], *,
        quarantine_dir: str | Path | None = None) -> Path:
    """Produce, durably validate, and atomically publish one artifact.

    A producer or validator exception can leave only a quarantined temporary
    file; the old final remains unchanged and no incomplete final name is
    exposed.  The wrfout handoff patch uses this exact helper.
    """
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp = unique_temp_path(final_path, hidden=True)
    try:
        producer(temp)
        fsync_file(temp)
        validator(temp)
        replace_file_with_retry(temp, final_path)
        _fsync_directory(final_path.parent)
    except BaseException:
        if temp.exists():
            quarantine_file(temp, reason="failed-publication",
                            quarantine_dir=quarantine_dir)
        raise
    return final_path


@dataclass(frozen=True)
class Heartbeat:
    schema: str
    run_id: str
    config_digest: str
    pid: int
    started_at_utc: str
    updated_at_utc: str
    status: str
    model_elapsed_seconds: float
    outer_step: int
    last_durable_wrfout: str | None
    last_checkpoint: str | None

    def __post_init__(self) -> None:
        if self.schema != HEARTBEAT_SCHEMA:
            raise ValueError(f"unsupported heartbeat schema {self.schema!r}")
        if (self.status not in {"integrating", "complete", "failed"}
                and not self.status.startswith("preparing:")):
            raise ValueError(f"invalid heartbeat status {self.status!r}")
        if self.pid <= 0 or self.outer_step < 0:
            raise ValueError("heartbeat pid must be positive and step nonnegative")
        if (not math.isfinite(self.model_elapsed_seconds)
                or self.model_elapsed_seconds < 0.0):
            raise ValueError("heartbeat model time must be finite and nonnegative")

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "Heartbeat":
        extra = set(payload) - _HEARTBEAT_FIELDS
        missing = _HEARTBEAT_FIELDS - set(payload)
        if extra or missing:
            raise ValueError(
                f"heartbeat fields mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}")
        return cls(**payload)


def read_heartbeat(path: str | Path) -> Heartbeat:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("heartbeat JSON root must be an object")
    return Heartbeat.from_mapping(payload)


def write_heartbeat(path: str | Path, heartbeat: Heartbeat) -> Path:
    return atomic_write_json(
        path, heartbeat.as_dict(), _quarantine_on_permission_error=True)


class RollingStepWall:
    """Bounded rolling wall-time history with a nearest-rank p99."""

    def __init__(self, maxlen: int = 256):
        if maxlen < 1:
            raise ValueError("maxlen must be positive")
        self._values: deque[float] = deque(maxlen=maxlen)

    def add(self, seconds: float) -> None:
        value = float(seconds)
        if math.isfinite(value) and value > 0.0:
            self._values.append(value)

    @property
    def p99(self) -> float:
        if not self._values:
            return 0.0
        values = sorted(self._values)
        index = max(0, math.ceil(0.99 * len(values)) - 1)
        return values[index]

    @property
    def stale_threshold_seconds(self) -> float:
        return max(3.0 * self.p99, 120.0)


def stale_threshold_seconds(step_wall_seconds: list[float] | tuple[float, ...]
                            ) -> float:
    history = RollingStepWall(maxlen=max(1, len(step_wall_seconds)))
    for value in step_wall_seconds:
        history.add(value)
    return history.stale_threshold_seconds


def config_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_directory_manifest(path: Path) -> str:
    """Hash a deterministic directory inventory without rereading GEOG GBs."""
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        stat = child.stat()
        record = (f"{child.relative_to(path).as_posix()}\0{stat.st_size}\0"
                  f"{stat.st_mtime_ns}\n").encode("utf-8")
        digest.update(record)
    return digest.hexdigest()


def resolved_input_hashes(config_path: str | Path) -> dict[str, Any]:
    """Hash declared case inputs before any device import/allocation."""
    from gpuwm.case_data import load_experiment_case

    _, data = load_experiment_case(config_path)
    result: dict[str, Any] = {}
    for record in data.resolved_inputs():
        path = Path(record.path)
        key = f"{record.role}:{path}"
        if path.is_file():
            result[key] = {"algorithm": "sha256", "digest": _hash_file(path),
                           "detail": record.detail}
        elif path.is_dir():
            result[key] = {
                "algorithm": "sha256-directory-inventory",
                "digest": _hash_directory_manifest(path),
                "detail": record.detail,
            }
        else:
            raise FileNotFoundError(f"declared input disappeared: {path}")
    return result


def git_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=False,
        capture_output=True, text=True, timeout=10)
    return (result.stdout.strip() if result.returncode == 0
            else f"unavailable: {result.stderr.strip()}")


@dataclass(frozen=True)
class GPUIdentity:
    uuid: str
    driver_version: str
    name: str
    index: int | None = None


@dataclass(frozen=True)
class GPUProcess:
    uuid: str
    pid: int
    process_name: str
    used_gpu_memory_mib: int | None = None
    process_type: str | None = None


def _run_nvidia_smi(arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", *arguments], check=False, capture_output=True,
            text=True, timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        raise GPUPreflightError(
            f"GPU preflight failed closed: nvidia-smi unavailable: {exc}") from exc
    if result.returncode != 0:
        raise GPUPreflightError(
            "GPU preflight failed closed: nvidia-smi returned "
            f"{result.returncode}: {result.stderr.strip()}")
    return result.stdout


def query_gpus() -> tuple[GPUIdentity, ...]:
    output = _run_nvidia_smi([
        "--query-gpu=index,uuid,driver_version,name",
        "--format=csv,noheader,nounits"])
    identities = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 3)]
        if (len(parts) != 4 or not parts[1].startswith("GPU-")
                or not parts[0].isdigit()):
            raise GPUPreflightError(
                f"GPU preflight failed closed: malformed GPU row {line!r}")
        identities.append(GPUIdentity(
            parts[1], parts[2], parts[3], int(parts[0])))
    if not identities:
        raise GPUPreflightError("GPU preflight failed closed: no GPU reported")
    return tuple(identities)


def select_gpu(requested_uuid: str | None = None) -> GPUIdentity:
    identities = query_gpus()
    if requested_uuid is None:
        if len(identities) != 1:
            raise GPUPreflightError(
                "multiple GPUs are present; --gpu-uuid is required to avoid "
                "ambiguous locking")
        return identities[0]
    matches = [gpu for gpu in identities if gpu.uuid == requested_uuid]
    if len(matches) != 1:
        raise GPUPreflightError(
            f"requested GPU UUID {requested_uuid!r} was not reported by "
            "nvidia-smi")
    return matches[0]


def _memory_mib(value: str) -> int | None:
    normalized = value.strip().strip("[]").strip()
    if normalized.lower() in {"n/a", "-", "not supported"}:
        return None
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise GPUPreflightError(
            f"GPU preflight failed closed: malformed memory value {value!r}") from exc
    if parsed < 0:
        raise GPUPreflightError(
            f"GPU preflight failed closed: negative memory value {value!r}")
    return parsed


def parse_compute_apps_output(output: str) -> tuple[GPUProcess, ...]:
    """Parse the WDDM four-column query-compute-apps CSV shape."""
    processes = []
    for row in csv.reader(io.StringIO(output)):
        if not row or not any(part.strip() for part in row):
            continue
        if "no running processes" in ",".join(row).lower():
            continue
        parts = [part.strip() for part in row]
        if len(parts) != 4 or not parts[0].startswith("GPU-"):
            raise GPUPreflightError(
                f"GPU preflight failed closed: malformed process row {row!r}")
        try:
            pid = int(parts[1])
        except ValueError as exc:
            raise GPUPreflightError(
                f"GPU preflight failed closed: malformed process PID {row!r}") from exc
        if pid <= 0:
            raise GPUPreflightError(
                f"GPU preflight failed closed: nonpositive process PID {row!r}")
        processes.append(GPUProcess(
            parts[0], pid, parts[2], _memory_mib(parts[3])))
    return tuple(processes)


def _parse_pmon_output(output: str) -> dict[int, tuple[str, int | None]]:
    """Return PID -> (C/C+G type, framebuffer MiB) from one pmon sample."""
    result: dict[int, tuple[str, int | None]] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # ``pmon -s um`` emits gpu,pid,type,sm,mem,enc,dec,jpg,ofa,fb,ccpm,name.
        parts = stripped.split(maxsplit=11)
        if len(parts) != 12:
            raise GPUPreflightError(
                f"GPU preflight failed closed: malformed pmon row {line!r}")
        if parts[1] == "-" and all(part == "-" for part in parts[1:]):
            # Linux/TCC drivers emit one canonical all-dash row for an idle
            # physical GPU.  It is an explicit absence record, not a process.
            continue
        try:
            pid = int(parts[1])
        except ValueError as exc:
            raise GPUPreflightError(
                f"GPU preflight failed closed: malformed pmon PID {line!r}") from exc
        process_type = parts[2]
        if process_type not in {"C", "G", "C+G"}:
            raise GPUPreflightError(
                f"GPU preflight failed closed: unknown pmon type {process_type!r}")
        result[pid] = process_type, _memory_mib(parts[9])
    return result


def query_compute_processes(gpu_uuid: str) -> tuple[GPUProcess, ...]:
    """Return WDDM rows enriched with pmon's compute-vs-graphics type."""
    apps = parse_compute_apps_output(_run_nvidia_smi([
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"]))
    modes = _parse_pmon_output(_run_nvidia_smi([
        "pmon", "-i", gpu_uuid, "-c", "1", "-s", "um"]))
    enriched = []
    for process in apps:
        if process.uuid != gpu_uuid:
            continue
        process_type, pmon_memory = modes.get(process.pid, (None, None))
        memories = [value for value in (
            process.used_gpu_memory_mib, pmon_memory) if value is not None]
        enriched.append(dataclasses.replace(
            process, process_type=process_type,
            used_gpu_memory_mib=(max(memories) if memories else None)))
    return tuple(enriched)


def preflight_exclusive_gpu(gpu_uuid: str, *,
                            approved_pids: set[int] | None = None,
                            memory_threshold_mib: int =
                            COMPUTE_MEMORY_THRESHOLD_MIB,
                            allow_shared_gpu: bool = False) -> None:
    """Verify identity and reject substantial pure-CUDA contenders.

    The UUID file lock is authoritative for excluding other gpuwm runs.
    WDDM reports desktop graphics contexts in ``query-compute-apps``; pmon
    labels those ``C+G`` and they are explicitly permitted.  A pure ``C``
    process is a contender when its memory is unmeasured/zero (the normal
    WDDM failure mode) or its measured memory exceeds the small context-noise
    threshold.  Tool/parse failures still fail closed.  ``allow_shared_gpu``
    bypasses only a proven contender and is an unsupported operator escape.
    """
    if memory_threshold_mib < 0:
        raise ValueError("memory_threshold_mib must be nonnegative")
    # Re-query on every launch so a stale UUID selection cannot silently
    # migrate the worker to a different physical device.
    select_gpu(gpu_uuid)
    approved = set() if approved_pids is None else set(approved_pids)
    # WDDM pmon can report fb=0 for an active pure-C row, so zero is not
    # evidence that the context is harmless; treat it as unmeasured.
    conflicts = [
        process for process in query_compute_processes(gpu_uuid)
        if (process.pid not in approved
            and process.process_type == "C"
            and (process.used_gpu_memory_mib in (None, 0)
                 or process.used_gpu_memory_mib > memory_threshold_mib))
    ]
    if conflicts and not allow_shared_gpu:
        detail = ", ".join(
            f"pid={process.pid} name={process.process_name!r} "
            f"memory={'unmeasured' if process.used_gpu_memory_mib in (None, 0) else f'{process.used_gpu_memory_mib}MiB'}"
            for process in conflicts)
        raise GPUPreflightError(
            f"GPU {gpu_uuid} has CUDA compute contender(s) with unmeasured "
            f"memory or above {memory_threshold_mib} MiB: {detail}; stop "
            "them or use "
            "--allow-shared-gpu (unsupported)")


def default_lock_path(gpu_uuid: str) -> Path:
    digest = hashlib.sha256(gpu_uuid.encode("utf-8")).hexdigest()[:24]
    if os.name == "nt":
        root = Path(os.environ.get("PROGRAMDATA", tempfile.gettempdir()))
    else:
        root = Path(tempfile.gettempdir())
    return root / "gpuwm" / "locks" / f"gpu-{digest}.lock"


class GPUFileLock:
    """Cross-process UUID-keyed exclusive lock (Windows byte-range lock)."""

    def __init__(self, gpu_uuid: str, *, path: str | Path | None = None,
                 run_id: str | None = None):
        self.gpu_uuid = gpu_uuid
        self.path = default_lock_path(gpu_uuid) if path is None else Path(path)
        self.run_id = run_id
        self._stream = None

    def acquire(self) -> "GPUFileLock":
        if self._stream is not None:
            raise RuntimeError("GPU lock is already held by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
                os.fsync(stream.fileno())
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            if _is_lock_contention(exc):
                raise GPUAlreadyLockedError(
                    f"GPU {self.gpu_uuid} lock is held: {self.path}") from exc
            raise
        self._stream = stream
        owner = json.dumps({"gpu_uuid": self.gpu_uuid, "pid": os.getpid(),
                            "run_id": self.run_id, "acquired_at_utc": utc_now()},
                           sort_keys=True).encode("utf-8")
        stream.seek(1)
        stream.truncate()
        stream.write(owner)
        stream.flush()
        os.fsync(stream.fileno())
        return self

    def release(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self) -> "GPUFileLock":
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


def _is_lock_contention(exc: OSError) -> bool:
    """Distinguish a held byte-range/flock from disk, ACL, or FD errors."""
    if os.name == "nt":
        return exc.errno in {errno.EACCES, errno.EDEADLK}
    return exc.errno in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}


def validate_manifest_checkpoint(path: str | Path) -> Path:
    """Read every NPZ member and prove agreement with the restart manifest."""
    from gpuwm.io.restart import read_restart_header

    path = Path(path)
    try:
        header = read_restart_header(path)
        manifest = header.get("array_manifest")
        if not isinstance(manifest, dict):
            raise CheckpointValidationError(
                f"checkpoint {path} has no array_manifest object")
        with np.load(path, allow_pickle=False) as archive:
            members = set(archive.files)
            payload_members = {name for name in members
                               if name != "__gpuwm_restart_header__"}
            if payload_members != set(manifest):
                raise CheckpointValidationError(
                    f"checkpoint {path} member set disagrees with manifest")
            for name, expected in manifest.items():
                array = archive[name]
                if (list(array.shape) != expected.get("shape")
                        or str(array.dtype) != expected.get("dtype")):
                    raise CheckpointValidationError(
                        f"checkpoint {path} member {name!r} disagrees with "
                        "its shape/dtype manifest")
                # Accessing the last byte forces lazy zip decompression/read.
                if array.size:
                    array.reshape(-1)[-1]
    except CheckpointValidationError:
        raise
    except Exception as exc:
        raise CheckpointValidationError(
            f"checkpoint {path} is not manifest-valid: {exc}") from exc
    fsync_file(path)
    return path.resolve()


def is_cuda_fatal(value: BaseException | str) -> bool:
    text = f"{type(value).__name__}: {value}" if isinstance(
        value, BaseException) else str(value)
    return any(pattern.search(text) for pattern in _CUDA_FATAL_PATTERNS)


def write_failure_capsule(
        path: str | Path, *, run_id: str, config_path: str | Path,
        config_sha256: str, input_hashes: dict[str, Any], gpu: GPUIdentity,
        last_phase: str, last_step: int, exception_type: str,
        exception_message: str, exception_traceback: str,
        last_durable_wrfout: str | None, last_checkpoint: str | None,
        worker_pid: int | None = None) -> Path:
    payload = {
        "schema": FAILURE_CAPSULE_SCHEMA,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config_sha256,
        "input_hashes": input_hashes,
        "git_commit": git_commit(),
        "gpu": dataclasses.asdict(gpu),
        "worker_pid": worker_pid,
        "last_phase": last_phase,
        "last_step": int(last_step),
        "last_durable_wrfout": last_durable_wrfout,
        "last_checkpoint": last_checkpoint,
        "exception": {
            "type": exception_type,
            "message": exception_message,
            "traceback": exception_traceback,
            "cuda_fatal": is_cuda_fatal(
                f"{exception_type}: {exception_message}\n{exception_traceback}"),
        },
    }
    return atomic_write_json(path, payload)


class RuntimeHeartbeat:
    """Runtime callback installed by the deferred runtime handoff patch."""

    def __init__(self, path: str | Path, *, run_id: str,
                 config_sha256: str, started_at_utc: str,
                 initial_checkpoint: str | None = None):
        self.path = Path(path)
        self.run_id = run_id
        self.config_sha256 = config_sha256
        self.started_at_utc = started_at_utc
        self.last_wrfout: str | None = None
        self.last_checkpoint: str | None = (
            None if initial_checkpoint is None
            else str(Path(initial_checkpoint).resolve()))
        self.last_phase = "preparing:worker-start"
        self.last_step = 0
        self.model_elapsed_seconds = 0.0

    def _write(self, status: str) -> None:
        write_heartbeat(self.path, Heartbeat(
            HEARTBEAT_SCHEMA, self.run_id, self.config_sha256, os.getpid(),
            self.started_at_utc, utc_now(), status,
            self.model_elapsed_seconds, self.last_step, self.last_wrfout,
            self.last_checkpoint))

    def __call__(self, *, model_elapsed_seconds: float, outer_step: int,
                 last_durable_wrfout: str | Path | None,
                 last_checkpoint: str | Path | None,
                 phase: str = "synchronized-step", **_: Any) -> None:
        if last_durable_wrfout is not None:
            wrfout = Path(last_durable_wrfout)
            resolved_wrfout = str(wrfout.resolve())
            if resolved_wrfout != self.last_wrfout:
                fsync_file(wrfout)
                self.last_wrfout = resolved_wrfout
        if last_checkpoint is not None:
            resolved_checkpoint = str(Path(last_checkpoint).resolve())
            if resolved_checkpoint != self.last_checkpoint:
                self.last_checkpoint = str(validate_manifest_checkpoint(
                    last_checkpoint))
        self.model_elapsed_seconds = float(model_elapsed_seconds)
        self.last_step = int(outer_step)
        self.last_phase = phase
        self._write("integrating")

    def preparing(self, phase: str) -> None:
        """Publish an immediate/named preparation-stage heartbeat."""
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", phase).strip("-")
        if not normalized:
            raise ValueError("preparation phase must not be empty")
        self.last_phase = f"preparing:{normalized}"
        self._write(self.last_phase)

    def starting(self) -> None:
        """Backward-compatible spelling for the immediate worker heartbeat."""
        self.preparing("worker-start")

    def complete(self, model_elapsed_seconds: float) -> None:
        self.model_elapsed_seconds = float(model_elapsed_seconds)
        self.last_phase = "complete"
        self._write("complete")

    def failed(self) -> None:
        self._write("failed")


@dataclass(frozen=True)
class SupervisorResult:
    run_id: str
    attempts: int
    heartbeat: Heartbeat
    stdout_logs: tuple[Path, ...]
    stderr_logs: tuple[Path, ...]


def _terminate_fresh_worker(process: subprocess.Popen, *, timeout: float = 10.0
                            ) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def _tail(path: Path, limit: int = 32_768) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode("utf-8", errors="replace")


def _has_worker_failure_capsule(path: Path, *, run_id: str,
                                worker_pid: int) -> bool:
    """Whether the worker already published a richer capsule for this exit."""
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        return (payload.get("schema") == FAILURE_CAPSULE_SCHEMA
                and payload.get("run_id") == run_id
                and payload.get("worker_pid") == worker_pid)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _worker_command(config_path: Path, outdir: Path, *, restart: Path | None,
                    health_debug: bool) -> list[str]:
    command = [sys.executable, "-m", "gpuwm.supervisor", "worker",
               "--config", str(config_path), "--outdir", str(outdir)]
    if restart is not None:
        command.extend(("--restart", str(restart)))
    if health_debug:
        command.append("--health-debug")
    return command


def _heartbeat_regression(previous: Heartbeat, current: Heartbeat) -> str | None:
    if (previous.status in {"complete", "failed"}
            and current != previous):
        return f"terminal status {previous.status} changed after publication"
    previous_time = datetime.fromisoformat(
        previous.updated_at_utc.replace("Z", "+00:00"))
    current_time = datetime.fromisoformat(
        current.updated_at_utc.replace("Z", "+00:00"))
    if current_time < previous_time:
        return "updated_at_utc moved backward"
    if current.outer_step < previous.outer_step:
        return (f"outer_step moved backward from {previous.outer_step} to "
                f"{current.outer_step}")
    if current.model_elapsed_seconds < previous.model_elapsed_seconds:
        return ("model_elapsed_seconds moved backward from "
                f"{previous.model_elapsed_seconds} to "
                f"{current.model_elapsed_seconds}")
    if (previous.status == "integrating"
            and current.status.startswith("preparing:")):
        return f"status moved backward from integrating to {current.status}"
    return None


def _bind_attempt_heartbeat(
        heartbeat: Heartbeat, *, run_id: str, config_digest: str,
        started_at_utc: str, launch_pid: int,
        effective_worker_pid: int | None) -> tuple[int | None, str | None]:
    """Validate one heartbeat and pin the real interpreter PID.

    A Windows venv executable can be a redirector: ``Popen.pid`` belongs to
    the redirector while the Python interpreter that writes heartbeats has a
    descendant PID.  ``preparing:launch`` is the supervisor's provisional
    record.  The first worker-authored status pins the effective PID; later
    PID changes fail closed.  This is correlation and binding within a trusted
    output directory, not authentication against a hostile local writer.
    """
    if heartbeat.run_id != run_id:
        return effective_worker_pid, "run_id does not match this attempt"
    if heartbeat.config_digest != config_digest:
        return effective_worker_pid, "config_digest does not match this attempt"
    if heartbeat.started_at_utc != started_at_utc:
        return effective_worker_pid, "started_at_utc does not match this attempt"
    if heartbeat.status == "preparing:launch":
        if heartbeat.pid != launch_pid:
            return effective_worker_pid, (
                "provisional launch heartbeat PID does not match Popen PID")
        if effective_worker_pid is not None:
            return effective_worker_pid, (
                "heartbeat reverted to provisional launch after worker PID pin")
        return effective_worker_pid, None
    if effective_worker_pid is None:
        return heartbeat.pid, None
    if heartbeat.pid != effective_worker_pid:
        return effective_worker_pid, (
            f"worker heartbeat PID changed from {effective_worker_pid} to "
            f"{heartbeat.pid}")
    return effective_worker_pid, None


def supervise_experiment(
        config_path: str | Path, outdir: str | Path, *,
        restart: str | Path | None = None, gpu_uuid: str | None = None,
        max_restarts: int = 3, poll_seconds: float = 1.0,
        prep_timeout_seconds: float | None = None,
        health_debug: bool = False, allow_shared_gpu: bool = False,
        lock_path: str | Path | None = None) -> SupervisorResult:
    """Run an experiment under exclusive-GPU fresh-process supervision."""
    if max_restarts < 0:
        raise ValueError("max_restarts must be nonnegative")
    if not 0.05 <= poll_seconds <= 60.0:
        raise ValueError("poll_seconds must be in [0.05, 60]")
    if (prep_timeout_seconds is not None
            and (not math.isfinite(prep_timeout_seconds)
                 or prep_timeout_seconds <= 0.0)):
        raise ValueError("prep_timeout_seconds must be finite and positive")
    config_path = Path(config_path).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    digest = config_digest(config_path)
    inputs = resolved_input_hashes(config_path)
    gpu = select_gpu(gpu_uuid)
    checkpoint = (None if restart is None
                  else validate_manifest_checkpoint(restart))
    heartbeat_path = outdir / HEARTBEAT_NAME
    capsule_path = outdir / FAILURE_CAPSULE_NAME
    stdout_logs: list[Path] = []
    stderr_logs: list[Path] = []
    attempts = 0

    with GPUFileLock(gpu.uuid, path=lock_path, run_id=run_id):
        while True:
            # Repeat before every fresh worker.  The UUID lock excludes other
            # gpuwm supervisors; this NVML/nvidia-smi view catches unrelated
            # compute processes that appeared between recovery attempts.
            preflight_exclusive_gpu(
                gpu.uuid, approved_pids={os.getpid()},
                allow_shared_gpu=allow_shared_gpu)
            attempts += 1
            # Every fresh process gets fresh preparation and step clocks.  A
            # recovery launch can never inherit the dead worker's stale age or
            # p99 history.
            history = RollingStepWall()
            started_at = utc_now()
            stdout_path = outdir / f"worker-{attempts:02d}.stdout.log"
            stderr_path = outdir / f"worker-{attempts:02d}.stderr.log"
            stdout_logs.append(stdout_path)
            stderr_logs.append(stderr_path)
            env = os.environ.copy()
            env.update({
                "GPUWM_RUN_ID": run_id,
                "GPUWM_CONFIG_DIGEST": digest,
                "GPUWM_STARTED_AT_UTC": started_at,
                "GPUWM_GPU_UUID": gpu.uuid,
                "GPUWM_GPU_DRIVER": gpu.driver_version,
                "GPUWM_GPU_NAME": gpu.name,
                "GPUWM_INPUT_HASHES_JSON": json.dumps(
                    inputs, sort_keys=True, separators=(",", ":")),
            })
            command = _worker_command(
                config_path, outdir, restart=checkpoint,
                health_debug=health_debug)
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    command, cwd=Path(__file__).resolve().parents[1], env=env,
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    close_fds=True)
            starting = Heartbeat(
                HEARTBEAT_SCHEMA, run_id, digest, process.pid, started_at,
                utc_now(), "preparing:launch", 0.0, 0, None,
                None if checkpoint is None else str(checkpoint))
            # A tiny fixture worker can publish before Popen returns.  Never
            # overwrite a newer record from this attempt with parent state.
            # On Windows a venv ``python.exe`` may be a redirector whose
            # Popen PID differs from the interpreter PID that publishes the
            # heartbeat.  The run id, config digest, and per-attempt start
            # time correlate the record; the first worker-authored record
            # then pins the effective interpreter PID for the whole attempt.
            try:
                existing = read_heartbeat(heartbeat_path)
            except (OSError, ValueError, json.JSONDecodeError):
                existing = None
            if (existing is None or existing.run_id != run_id
                    or existing.config_digest != digest
                    or existing.started_at_utc != started_at):
                write_heartbeat(heartbeat_path, starting)

            last_heartbeat: Heartbeat | None = None
            last_signal_monotonic = time.monotonic()
            integrating_seen = False
            effective_worker_pid: int | None = None
            monitor_failure: str | None = None
            monitor_failure_kind: str | None = None
            while process.poll() is None:
                time.sleep(poll_seconds)
                try:
                    current = read_heartbeat(heartbeat_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    current = None
                if current is not None:
                    effective_worker_pid, attempt_error = (
                        _bind_attempt_heartbeat(
                            current, run_id=run_id, config_digest=digest,
                            started_at_utc=started_at,
                            launch_pid=process.pid,
                            effective_worker_pid=effective_worker_pid))
                    if attempt_error is not None:
                        monitor_failure_kind = "heartbeat-identity"
                        monitor_failure = (
                            f"worker heartbeat identity violation: {attempt_error}")
                        _terminate_fresh_worker(process)
                        break
                    if last_heartbeat is not None:
                        regression = _heartbeat_regression(
                            last_heartbeat, current)
                        if regression is not None:
                            monitor_failure_kind = "heartbeat-regression"
                            monitor_failure = f"worker heartbeat regression: {regression}"
                            _terminate_fresh_worker(process)
                            break
                    if current != last_heartbeat:
                        if (last_heartbeat is not None
                                and last_heartbeat.status == "integrating"
                                and current.status == "integrating"
                                and current.outer_step
                                > last_heartbeat.outer_step):
                            previous = datetime.fromisoformat(
                                last_heartbeat.updated_at_utc.replace(
                                    "Z", "+00:00")).timestamp()
                            updated = datetime.fromisoformat(
                                current.updated_at_utc.replace(
                                    "Z", "+00:00")).timestamp()
                            history.add(updated - previous)
                        last_heartbeat = current
                        last_signal_monotonic = time.monotonic()
                    integrating_seen |= current.status == "integrating"
                silent_seconds = time.monotonic() - last_signal_monotonic
                if (not integrating_seen and prep_timeout_seconds is not None
                        and silent_seconds > prep_timeout_seconds):
                    monitor_failure_kind = "prep-timeout"
                    phase = ("preparing:launch" if last_heartbeat is None
                             else last_heartbeat.status)
                    monitor_failure = (
                        f"worker preparation timed out in {phase} after "
                        f"{silent_seconds:.1f} s without a heartbeat")
                    _terminate_fresh_worker(process)
                    break
                if (integrating_seen
                        and silent_seconds > history.stale_threshold_seconds):
                    monitor_failure_kind = "stale-integration"
                    monitor_failure = (
                        "worker integrating heartbeat became stale after "
                        f"{silent_seconds:.1f} s")
                    _terminate_fresh_worker(process)
                    break
            return_code = process.wait()
            if monitor_failure is None:
                try:
                    current = read_heartbeat(heartbeat_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    current = None
                if current is not None:
                    effective_worker_pid, attempt_error = (
                        _bind_attempt_heartbeat(
                            current, run_id=run_id, config_digest=digest,
                            started_at_utc=started_at,
                            launch_pid=process.pid,
                            effective_worker_pid=effective_worker_pid))
                    if attempt_error is not None:
                        monitor_failure_kind = "heartbeat-identity"
                        monitor_failure = (
                            f"worker heartbeat identity violation: {attempt_error}")
                    else:
                        regression = (
                            None if last_heartbeat is None
                            else _heartbeat_regression(last_heartbeat, current))
                        if regression is not None:
                            monitor_failure_kind = "heartbeat-regression"
                            monitor_failure = (
                                f"worker heartbeat regression: {regression}")
                        else:
                            last_heartbeat = current
            if (monitor_failure is None
                    and return_code == 0 and last_heartbeat is not None
                    and last_heartbeat.run_id == run_id
                    and last_heartbeat.config_digest == digest
                    and last_heartbeat.started_at_utc == started_at
                    and last_heartbeat.status == "complete"):
                return SupervisorResult(
                    run_id, attempts, last_heartbeat, tuple(stdout_logs),
                    tuple(stderr_logs))

            message = (monitor_failure if monitor_failure is not None else
                       f"worker exited with status {return_code}")
            stderr_tail = _tail(stderr_path)
            hb = last_heartbeat
            worker_pid = effective_worker_pid or process.pid
            if not _has_worker_failure_capsule(
                    capsule_path, run_id=run_id, worker_pid=worker_pid):
                write_failure_capsule(
                    capsule_path, run_id=run_id, config_path=config_path,
                    config_sha256=digest, input_hashes=inputs, gpu=gpu,
                    last_phase=(monitor_failure_kind or "worker-exit"),
                    last_step=0 if hb is None else hb.outer_step,
                    exception_type=("WorkerMonitorFailure"
                                    if monitor_failure is not None
                                    else "WorkerExit"),
                    exception_message=message,
                    exception_traceback=stderr_tail,
                    last_durable_wrfout=(None if hb is None else
                                         hb.last_durable_wrfout),
                    last_checkpoint=(None if hb is None
                                     else hb.last_checkpoint),
                    worker_pid=worker_pid)
            if monitor_failure_kind in {
                    "prep-timeout", "heartbeat-identity",
                    "heartbeat-regression"}:
                raise SupervisorError(
                    f"{message}; refusing a deterministic relaunch loop "
                    f"(failure capsule: {capsule_path})")
            proposed = None if hb is None else hb.last_checkpoint
            if proposed is None:
                raise SupervisorError(
                    f"{message}; no durable manifest-valid checkpoint is "
                    f"available (failure capsule: {capsule_path})")
            checkpoint = validate_manifest_checkpoint(proposed)
            if attempts > max_restarts:
                raise SupervisorError(
                    f"{message}; exhausted {max_restarts} fresh-process "
                    f"restart(s) (failure capsule: {capsule_path})")


def _worker_main(args: argparse.Namespace) -> int:
    """Fresh CUDA worker entry.  Never called inside the supervisor process."""
    config_path = Path(args.config).resolve()
    outdir = Path(args.outdir).resolve()
    run_id = os.environ["GPUWM_RUN_ID"]
    digest = os.environ["GPUWM_CONFIG_DIGEST"]
    started_at = os.environ["GPUWM_STARTED_AT_UTC"]
    gpu = GPUIdentity(os.environ["GPUWM_GPU_UUID"],
                      os.environ["GPUWM_GPU_DRIVER"],
                      os.environ["GPUWM_GPU_NAME"])
    progress = RuntimeHeartbeat(
        outdir / HEARTBEAT_NAME, run_id=run_id, config_sha256=digest,
        started_at_utc=started_at,
        initial_checkpoint=(None if args.restart is None
                            else str(Path(args.restart).resolve())))
    # Publish before importing the runtime/CuPy-facing module graph.  This is
    # the first executable worker action after environment/path setup.
    progress.preparing("worker-start")
    input_hashes: dict[str, Any] = {}
    try:
        encoded_inputs = os.environ.get("GPUWM_INPUT_HASHES_JSON", "{}")
        try:
            decoded_inputs = json.loads(encoded_inputs)
        except json.JSONDecodeError as exc:
            raise SupervisorError(
                "worker received malformed parent input-hash inventory") from exc
        if not isinstance(decoded_inputs, dict):
            raise SupervisorError(
                "worker input-hash inventory must be an object")
        input_hashes = decoded_inputs
        progress.preparing("import-runtime")
        from gpuwm.case_data import load_experiment_case
        from gpuwm import runtime

        progress.preparing("load-config")
        exp, data = load_experiment_case(config_path)
        progress.preparing("prepare-case")
        summary = runtime.run_experiment(
            exp, data, outdir, restart=args.restart,
            progress_callback=progress, health_debug=args.health_debug)
        progress.complete(summary.completed_seconds)
        return 0
    except BaseException as exc:
        try:
            progress.failed()
            write_failure_capsule(
                outdir / FAILURE_CAPSULE_NAME, run_id=run_id,
                config_path=config_path, config_sha256=digest,
                input_hashes=input_hashes, gpu=gpu,
                last_phase=progress.last_phase, last_step=progress.last_step,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                exception_traceback=traceback.format_exc(),
                last_durable_wrfout=progress.last_wrfout,
                last_checkpoint=progress.last_checkpoint,
                worker_pid=os.getpid())
        finally:
            # Re-raising terminates this CUDA process.  The supervisor alone
            # decides whether a manifest-valid checkpoint permits a NEW one.
            raise


def register_cli(subparsers: argparse._SubParsersAction,
                 command: str = "run") -> None:
    """Attach Task-15 run flags without owning ``gpuwm/cli.py``.

    ``command`` names the already-registered subparser to decorate: the
    ``run`` parser by default, and ``resume`` for the sugar command that
    continues a supervised run and therefore takes the identical
    supervision surface.
    """
    run = subparsers.choices.get(command)
    if run is None:
        raise ValueError(
            f"register_cli requires the existing {command!r} parser")
    run.add_argument(
        "--no-supervise", action="store_true",
        help="run the experiment in this process (escape hatch; disables "
             "fresh-process recovery and exclusive-GPU supervision)")
    run.add_argument("--gpu-uuid", default=None, metavar="GPU-UUID",
                     help="physical GPU UUID to lock (required on multi-GPU hosts)")
    run.add_argument("--supervisor-max-restarts", type=int, default=3,
                     metavar="N", help="fresh-process recovery attempts (default 3)")
    run.add_argument(
        "--prep-timeout", type=float, default=None, metavar="SECONDS",
        help="optional preparation heartbeat timeout; default is no timeout "
             "until integration begins")
    run.add_argument(
        "--allow-shared-gpu", action="store_true",
        help="UNSUPPORTED: permit another substantial CUDA compute context; "
             "device verification and the GPUWM UUID lock remain enforced")
    run.add_argument("--health-debug", action="store_true",
                     help="enable debug phase health attribution hooks")


def supervise_from_cli(args: argparse.Namespace) -> int:
    result = supervise_experiment(
        args.config, args.outdir, restart=args.restart,
        gpu_uuid=args.gpu_uuid,
        max_restarts=args.supervisor_max_restarts,
        prep_timeout_seconds=args.prep_timeout,
        allow_shared_gpu=args.allow_shared_gpu,
        health_debug=args.health_debug)
    transition_receipt, transition_sha = _current_transition_receipt(
        args.outdir, result.run_id, config_digest(args.config))
    print({"run_id": result.run_id, "status": result.heartbeat.status,
           "attempts": result.attempts,
           "completed_seconds": result.heartbeat.model_elapsed_seconds,
           "last_durable_wrfout": result.heartbeat.last_durable_wrfout,
           "last_checkpoint": result.heartbeat.last_checkpoint,
           "microphysics_transition_receipt": (
               str(transition_receipt) if transition_receipt is not None
               else None),
           "microphysics_transition_receipt_sha256": transition_sha})
    return 0


def _current_transition_receipt(
        outdir: str | Path, run_id: str, digest: str
        ) -> tuple[Path | None, str | None]:
    """Return only a receipt bound to this supervised run/config."""

    path = (Path(outdir) / MICROPHYSICS_TRANSITION_RECEIPT_NAME).resolve()
    if not path.is_file():
        return None, None
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, json.JSONDecodeError):
        return None, None
    if (not isinstance(payload, dict)
            or payload.get("run_id") != run_id
            or payload.get("config_digest") != digest):
        return None, None
    if (payload.get("schema") != "gpuwm.microphysics-transitions/v1"
            or payload.get("status") != "PASS"
            or not isinstance(payload.get("transitions"), list)):
        raise SupervisorError(
            "current-run microphysics transition receipt is malformed")
    return path, hashlib.sha256(encoded).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gpuwm.supervisor")
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--outdir", type=Path, required=True)
    worker.add_argument("--restart", type=Path, default=None)
    worker.add_argument("--health-debug", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "worker":
        return _worker_main(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "COMPUTE_MEMORY_THRESHOLD_MIB", "FAILURE_CAPSULE_NAME",
    "GPUAlreadyLockedError", "GPUFileLock", "GPUIdentity",
    "GPUPreflightError", "GPUProcess", "HEARTBEAT_NAME",
    "HEARTBEAT_SCHEMA", "Heartbeat", "RollingStepWall", "RuntimeHeartbeat",
    "SupervisorError", "SupervisorResult", "atomic_publish_file",
    "atomic_write_json", "config_digest", "fsync_file", "is_cuda_fatal",
    "parse_compute_apps_output", "preflight_exclusive_gpu",
    "quarantine_file", "read_heartbeat", "register_cli",
    "replace_file_with_retry", "resolved_input_hashes", "select_gpu",
    "stale_threshold_seconds", "supervise_experiment",
    "supervise_from_cli", "utc_now", "validate_manifest_checkpoint",
    "write_failure_capsule", "write_heartbeat", "unique_temp_path",
]
