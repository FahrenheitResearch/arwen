"""Chained preparation: the forecast starts while later boundaries are built.

Boundary interval k needs only forcing times k and k+1.  A preparation used
to build every forcing time, pack every interval and publish the whole
prepared tree in one rename before the model could take its first step.
This module splits the prepared tree into three parts so the model can
start once the start state and the next forcing time exist:

* the HEAD: everything the start time makes (static fields, receipts, the
  non-boundary arrays of ``prepared-cache/``) plus ``boundary-stream/
  head.json``, which carries the full interval schedule.  It is published
  with the route's own single rename, early;
* one SEGMENT per interval: its arrays written straight into
  ``prepared-cache/`` under the file numbers the one-shot writer would have
  given them, then ``boundary-stream/segments/{k:05d}.json`` written last.
  The marker is the only ready signal, so the rule works across processes
  and across machines (a copy loop copies arrays first, marker last);
* the SEAL: the same ``prepared-cache/header.json`` the one-shot writer
  writes (so ``content_sha256`` is unchanged), then the companion files,
  then ``proof.json`` last, which stays every existing consumer's
  completion marker.

A tree whose segments all exist before the run starts is the same source,
already complete: there is one writer (:class:`PreparedTreeWriter`), one
reader (:class:`StreamedIntervals`) and one wait, used by every route that
builds boundaries from forcing times.

The early publication is on by default (see :data:`CHAINED_DEFAULT`);
``GPUWM_CHAINED_PREP=0`` turns it off for diagnosis, and then the same
writer publishes the head at the seal, so the run starts after preparation
exactly as before.  An installation with no forecast (the RW-WPS
preparation package, see :func:`forecast_installed`) always publishes at
the seal.
"""

from __future__ import annotations

from collections.abc import Sequence
import contextvars
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import threading
import time
import traceback
from typing import Callable, Mapping

import numpy as np


STREAM_DIRNAME = "boundary-stream"
HEAD_NAME = "head.json"
SEGMENTS_DIRNAME = "segments"
PRODUCER_NAME = "producer.json"
FAILED_NAME = "failed.json"
STOP_NAME = "stop.json"
PROOF_NAME = "proof.json"
HEAD_SCHEMA = "gpuwm-boundary-stream-head-v1"
SEGMENT_SCHEMA = "gpuwm-boundary-stream-segment-v1"
CHAINED_ENV = "GPUWM_CHAINED_PREP"

#: How often the producer refreshes ``producer.json``.
HEARTBEAT_SECONDS = 15.0
#: How often a waiting consumer republishes its wait reason.  The ``gpuwm
#: run`` supervisor reads a moving heartbeat as alive, so a wait that
#: publishes nothing for longer than its hang threshold would be killed as
#: a hang while the producer is still working.
WAIT_REPORT_SECONDS = 5.0
#: The shortest heartbeat age that counts as a silent producer.  The limit
#: grows to three times the slowest forcing-time build seen so far, so a
#: slow machine is not mistaken for a dead one.
SILENT_FLOOR_SECONDS = 120.0

#: Proof keys only the seal can write: they bind the complete cache, the
#: companion WRF export and the wall times of the whole preparation.  The
#: head carries the proof WITHOUT them (``proof_head``); the sealed proof
#: must equal the head's proof plus exactly these keys, which is what lets a
#: runner validate everything else at the head and the rest at the seal.
SEAL_ONLY_PROOF_KEYS = frozenset({
    "prepared_cache", "export", "initialization_artifacts",
    "timing_seconds", "proof_content_sha256", "boundary_stream",
})


class BoundaryStreamError(RuntimeError):
    """A streamed boundary source cannot deliver the interval asked for."""


class BoundaryProducerFailed(BoundaryStreamError):
    """The producer wrote ``failed.json``: the run ends with its reason."""


class BoundaryProducerSilent(BoundaryStreamError):
    """The producer stopped refreshing its heartbeat without failing."""


class BoundaryStreamStopped(BoundaryStreamError):
    """The consumer wrote ``stop.json``; the producer exits unsealed."""


#: Whether preparations chain when ``GPUWM_CHAINED_PREP`` is unset.  On:
#: the GPU proof runs of 2026-09-28 wrote byte-identical history files
#: chained, unchained and on the line before this change, for a single
#: mapped domain, a nested tree, a tiled root, a tiled child, a GFS domain,
#: a restart before and after the seal and a downscaled child.  The
#: variable stays as the off switch for diagnosis.
CHAINED_DEFAULT = True


def chained_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether a preparation publishes its head before its seal.

    ``GPUWM_CHAINED_PREP=1`` (or on/true/yes) turns it on and ``0`` (or
    off/false/no) off; unset, :data:`CHAINED_DEFAULT` decides.  Every
    stage subprocess inherits the variable.
    """

    value = (os.environ if environ is None else environ).get(CHAINED_ENV)
    if value is None or not str(value).strip():
        return CHAINED_DEFAULT
    return str(value).strip().lower() not in {"0", "off", "false", "no"}


#: What a chained head needs on the machine that writes it: the forecast's
#: memory admission (:func:`chained_admission` reads it from
#: ``gpuwm.core.preflight``) and a forecast executor to bind the head.  The
#: RW-WPS preparation package ships neither.
CHAINED_FORECAST_MODULES = ("gpuwm.core.preflight", "gpuwm.core.model")

#: The decision a writer records when :func:`forecast_installed` is false.
PREPARATION_ONLY_REASON = (
    "this installation prepares inputs and carries no forecast to start on "
    "the head, so the tree is published at its seal")


def forecast_installed() -> bool:
    """Whether this installation can run a forecast on a chained head.

    False in the RW-WPS preparation package, which stages this module for
    the era5, gfs and mapped routes but none of
    :data:`CHAINED_FORECAST_MODULES`.  There a CUDA preparation's
    admission died on ``No module named 'gpuwm.core.preflight'`` right
    after its start time was built, and a CPU one declined chaining only
    because its RAM reader lives in that same missing module.
    """

    for name in CHAINED_FORECAST_MODULES:
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def _say(line: str) -> None:
    """One line on stderr: a hosting door (run-plan) owns stdout."""

    print(line, file=sys.stderr, flush=True)


def _sealed_line(reason: str) -> str:
    return f"prepare: {reason}; the forecast starts after preparation"


#: Why a preparation is published at its seal rather than its head, one
#: row per kind of preparation that builds every forcing time before its
#: forecast can start.  Each is said once, where that preparation builds
#: its forcing (:func:`say_prepared_sealed`), so a run that did not start
#: early says why.
SEALED_REASONS = {
    "domain_tree": (
        "chained preparation not used: a domain tree's children are "
        "prepared from every root boundary interval, so the tree is "
        "published sealed"),
    "water_overlay": (
        "chained preparation not used: the water-temperature overlay binds "
        "its receipt over every forcing time into the cache identity, so "
        "the head cannot exist before the last forcing time"),
    "met_em": (
        "chained preparation not used: the met_em route writes each "
        "domain's prepared cache whole"),
    "native_hrrr": (
        "chained preparation not used: the native HRRR route builds every "
        "boundary hour into one prepared cache"),
    "experiment_run": (
        "chained preparation not used: this route builds its forcing "
        "times in the forecast's own process, before the model starts"),
}


def say_prepared_sealed(kind: str) -> None:
    """Say why this preparation is published at its seal, not its head.

    ``kind`` is a row of :data:`SEALED_REASONS`.  The line is the one a
    chained route prints when it declines, on stderr.  Silent when
    ``GPUWM_CHAINED_PREP=0`` turned chaining off, because then nothing
    was expected to start early.
    """

    reason = SEALED_REASONS[kind]
    if chained_enabled():
        _say(_sealed_line(reason))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value) -> str:
    from gpuwm.ingest.prepared_cache import _canonical as canonical

    return canonical(value)


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None


def stream_dir(root) -> Path:
    return Path(root) / STREAM_DIRNAME


def segment_marker_path(root, index: int) -> Path:
    return stream_dir(root) / SEGMENTS_DIRNAME / f"{int(index):05d}.json"


def head_sha256(head: Mapping[str, object]) -> str:
    """The digest a runner is pinned to: sha256 of the head's basis."""

    return hashlib.sha256(
        _canonical(head["basis"]).encode("utf-8")).hexdigest()


def read_head(root, *, expected_sha256: str | None = None) -> dict:
    """Load ``head.json`` and check its digest (and the caller's pin)."""

    path = stream_dir(root) / HEAD_NAME
    head = _read_json(path)
    if (not isinstance(head, dict) or head.get("schema") != HEAD_SCHEMA
            or not isinstance(head.get("basis"), dict)):
        raise BoundaryStreamError(f"{path} is not a readable {HEAD_SCHEMA}")
    digest = head_sha256(head)
    if head.get("head_sha256") != digest:
        raise BoundaryStreamError(f"{path} fails its own head digest")
    if expected_sha256 is not None and digest != str(expected_sha256).lower():
        raise BoundaryStreamError(
            f"{path} is head {digest}, not the pinned head {expected_sha256}")
    return head


def bind_head(root, head_sha256: str, *,
              require_manifest: bool = False) -> dict:
    """The head a forecast is started on, checked the same way at every door.

    Refused (``BoundaryStreamError``, which each door wraps in its own
    refusal) when the head fails its digest or the pin, when the
    preparation under it will never finish (failed, stopped or silent,
    :func:`unfinished_tree_reason`), which is the breakage this prevents:
    a forecast started on boundaries that never arrive; and, with
    ``require_manifest``, when the head names no source manifest for the
    forecast to bind.
    """

    root = Path(root)
    head = read_head(root, expected_sha256=head_sha256)
    reason = unfinished_tree_reason(root)
    if reason is not None:
        raise BoundaryStreamError(
            f"the prepared head in {root} will never be sealed: {reason}")
    if require_manifest and not isinstance(
            head["basis"].get("input_manifest_sha256"), str):
        raise BoundaryStreamError(
            f"the prepared head in {root} names no source manifest, so the "
            "forecast has nothing to bind it to")
    return head


def live_chained_head(root) -> dict | None:
    """The chained head of a preparation still being produced, or ``None``.

    ``None`` for a sealed tree, a tree published at its seal, and a
    preparation that will never finish.
    """

    root = Path(root)
    if prepared_tree_complete(root) or unfinished_tree_reason(root):
        return None
    stream = stream_dir(root)
    if (stream / FAILED_NAME).exists() or (stream / STOP_NAME).exists():
        return None
    try:
        head = read_head(root)
    except BoundaryStreamError:
        return None
    if not (head.get("decision") or {}).get("chained", False):
        return None
    return head


def prepared_tree_complete(root) -> bool:
    """A prepared tree is complete when ``proof.json`` exists.

    The one test for "prepared", so no code reads "the output root exists"
    as "the preparation finished" once a head can be published early.
    """

    return (Path(root) / PROOF_NAME).is_file()


def unfinished_tree_reason(root, *, now: float | None = None) -> str | None:
    """Why ``root`` is a head without a seal that nothing will finish.

    ``None`` means the tree is complete, absent, or still being produced by
    a live producer.  Otherwise the reason names the marker that decided it.
    """

    root = Path(root)
    if not root.exists() or prepared_tree_complete(root):
        return None
    stream = stream_dir(root)
    if not (stream / HEAD_NAME).is_file():
        return None
    failed = _read_json(stream / FAILED_NAME)
    if isinstance(failed, dict):
        return f"its producer failed: {failed.get('reason')}"
    stop = _read_json(stream / STOP_NAME)
    if isinstance(stop, dict):
        return f"its forecast stopped it: {stop.get('reason')}"
    beat = _read_json(stream / PRODUCER_NAME)
    age = _heartbeat_age(beat, now=now)
    if age is None or age > _silence_limit(beat):
        return ("its producer has been silent for "
                f"{'an unknown time' if age is None else f'{age:.0f} s'}")
    return None


def remove_unfinished_tree(root, *, log=None) -> bool:
    """Remove an unfinished tree so the next preparation can rebuild it.

    It is this tool's own staging product: a head published early whose
    producer failed, was stopped or went silent.  Refused for anything
    else, which is the breakage this prevents: deleting a complete
    preparation or one a live producer is still writing.
    """

    reason = unfinished_tree_reason(root)
    if reason is None:
        return False
    (log or _say)(f"prepare: removing the unfinished preparation {root} "
                  f"({reason}) and building it again")
    shutil.rmtree(root)
    return True


#: Share of the card a chained forecast and its producer leave unclaimed.
CHAINED_HEADROOM = 0.10
_GIB = float(1 << 30)


def producer_device_bytes(backend: str) -> int | None:
    """What a CUDA producer holds right after building one forcing time.

    The CuPy pool keeps the blocks a build used, so its total after the
    start time is built is that build's device footprint.  ``None`` for a
    host producer.
    """

    if str(backend) != "cuda":
        return None
    try:
        import cupy

        return int(cupy.get_default_memory_pool().total_bytes())
    except Exception:  # noqa: BLE001 - no device, nothing to measure
        return None


def _measure_card() -> tuple[int, int] | None:
    try:
        import cupy

        free, _total = cupy.cuda.runtime.memGetInfo()
        return int(free), int(cupy.get_default_memory_pool().total_bytes())
    except Exception:  # noqa: BLE001 - no device to measure
        return None


def chained_admission(*, experiment, backend: str,
                      device_bytes: int | None = None,
                      card: tuple[int, int] | None = None) -> dict:
    """Whether a forecast may run beside this producer; see ``admit``."""

    if str(backend) != "cuda":
        # The card is not shared; host RAM is, and write_head prices it
        # for every backend (host_admission).
        return {"admitted": True, "device": "host",
                "reason": "the producer prepares on the host, so the "
                          "forecast's card is not shared"}
    from gpuwm.core.preflight import admission_estimate

    forecast_bytes = int(admission_estimate(experiment).alloc_estimate_bytes)
    card = _measure_card() if card is None else card
    if card is None or device_bytes is None:
        return {"admitted": False, "device": "cuda",
                "forecast_bytes": forecast_bytes,
                "reason": ("chained preparation not admitted: the producer "
                           "prepares on the GPU and its memory could not be "
                           "measured, so a forecast beside it could run "
                           "both out of memory")}
    free, pool = card
    budget = int((free + pool) * (1.0 - CHAINED_HEADROOM))
    need = forecast_bytes + int(device_bytes)
    admitted = need <= budget
    return {
        "admitted": admitted, "device": "cuda",
        "forecast_bytes": forecast_bytes, "producer_bytes": int(device_bytes),
        "budget_bytes": budget,
        "reason": (
            f"forecast {forecast_bytes / _GIB:.2f} GiB + preparation "
            f"{device_bytes / _GIB:.2f} GiB fit {budget / _GIB:.2f} GiB "
            "on the GPU" if admitted else
            f"chained preparation not admitted: forecast "
            f"{forecast_bytes / _GIB:.2f} GiB + preparation "
            f"{device_bytes / _GIB:.2f} GiB > {budget / _GIB:.2f} GiB on "
            "the GPU"),
    }


def process_memory_bytes() -> tuple[int, int] | None:
    """This process's ``(resident, peak resident)`` bytes, or ``None``.

    Linux reads ``VmRSS`` and ``VmHWM`` from ``/proc/self/status``;
    Windows reads the working set and its peak.  Anything else is
    unmeasured.
    """

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _Counters()
            counters.cb = ctypes.sizeof(counters)
            psapi = ctypes.WinDLL("psapi")
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            if not psapi.GetProcessMemoryInfo(
                    kernel32.GetCurrentProcess(), ctypes.byref(counters),
                    counters.cb):
                return None
            return (int(counters.WorkingSetSize),
                    int(counters.PeakWorkingSetSize))
        except Exception:  # noqa: BLE001 - unmeasured, said so by None
            return None
    values = {}
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                name, _, rest = line.partition(":")
                if name in {"VmRSS", "VmHWM"}:
                    values[name] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    if set(values) != {"VmRSS", "VmHWM"}:
        return None
    return values["VmRSS"], values["VmHWM"]


def _host_available() -> int | None:
    try:
        from gpuwm.core.preflight import host_available_bytes

        return host_available_bytes()
    except Exception:  # noqa: BLE001 - unmeasured, said so by None
        return None


#: Host RAM a forecast process holds whatever its domain: the interpreter,
#: NumPy and CuPy, the CUDA context and runtime, the physics tables and the
#: host copies a restore makes on the way to the card.  Measured as
#: ``memory.cpu_peak_rss_bytes`` in the ``report.json`` of whole forecasts
#: on domains whose own arrays are a few hundred MiB at most: 1.34 GiB at
#: 80 x 64 x 32, 1.58 GiB at 73 x 73 x 49, 1.86 GiB at 200 x 200 x 49 and
#: 1.87 GiB at 162 x 162 x 49.  The floor sits above all four.  With the
#: head and the whole boundary series added
#: (:func:`forecast_host_bytes`), the 1792 x 1024 x 55 CONUS forecast of
#: 2026-09-28 prices at 13.6 GiB against its measured 11.2 GiB peak.
FORECAST_HOST_FLOOR_BYTES = 2 * (1 << 30)


def forecast_host_bytes(*, head_payload_bytes: int,
                        interval_host_bytes: int | None,
                        intervals: int) -> dict:
    """What a chained forecast holds in host RAM, in its three parts.

    The head's arrays, read into host RAM by the restore; the whole
    boundary series, because :class:`StreamedIntervals` keeps every
    interval it loads for the rest of the run; and
    :data:`FORECAST_HOST_FLOOR_BYTES`, the process itself.
    ``interval_host_bytes`` is one loaded interval (``None`` when it could
    not be priced, and then so is the total).
    """

    series = (None if interval_host_bytes is None
              else int(interval_host_bytes) * int(intervals))
    return {
        "head_payload_bytes": int(head_payload_bytes),
        "boundary_series_bytes": series,
        "process_floor_bytes": FORECAST_HOST_FLOOR_BYTES,
        "total_bytes": (None if series is None else
                        int(head_payload_bytes) + series
                        + FORECAST_HOST_FLOOR_BYTES),
    }


def host_admission(*, forecast_bytes: int | None,
                   producer_bytes: int | None = None,
                   available_bytes: int | None = None) -> dict:
    """Whether the machine's RAM holds a forecast beside its producer.

    Before chaining, host RAM held the preparation and then the forecast;
    chained, it holds both at once, on every producer backend.  The
    breakage this prevents is a machine that runs out of RAM mid-run with
    both of them half done (on Linux the kernel kills a process from
    outside, with no gpuwm message).  ``available_bytes`` is the RAM
    available right after the start time was built, so everything the
    producer keeps is already out of it; ``producer_bytes`` is what the
    producer's builds take on top of that (its measured peak resident
    size less its resident size now); ``forecast_bytes`` is what the
    forecast process holds in host RAM (:func:`forecast_host_bytes`), or
    ``None`` when it could not be priced.  Admitted with 10% of the
    available RAM left over, otherwise the tree is published at its seal
    as before.
    """

    available = (_host_available() if available_bytes is None
                 else int(available_bytes))
    if producer_bytes is None:
        memory = process_memory_bytes()
        producer_bytes = (None if memory is None
                          else max(0, memory[1] - memory[0]))
    if available is None or producer_bytes is None or forecast_bytes is None:
        return {"admitted": False, "memory": "host",
                "forecast_host_bytes": (None if forecast_bytes is None
                                        else int(forecast_bytes)),
                "reason": ("chained preparation not admitted: this "
                           "machine's available RAM, the producer's own "
                           "or the forecast's could not be measured, so a "
                           "forecast beside it could run the machine out "
                           "of RAM")}
    budget = int(available * (1.0 - CHAINED_HEADROOM))
    need = int(forecast_bytes) + int(producer_bytes)
    admitted = need <= budget
    return {
        "admitted": admitted, "memory": "host",
        "forecast_host_bytes": int(forecast_bytes),
        "producer_host_bytes": int(producer_bytes),
        "host_budget_bytes": budget,
        "reason": (
            f"forecast {forecast_bytes / _GIB:.2f} GiB + preparation "
            f"{producer_bytes / _GIB:.2f} GiB fit {budget / _GIB:.2f} GiB "
            "of host RAM" if admitted else
            f"chained preparation not admitted: forecast "
            f"{forecast_bytes / _GIB:.2f} GiB + preparation "
            f"{producer_bytes / _GIB:.2f} GiB > {budget / _GIB:.2f} GiB of "
            "available host RAM"),
    }


def request_stop(root, reason: str) -> bool:
    """Ask a producer to exit unsealed: its forecast ended first.

    Writes ``stop.json`` only beside a head without a seal; the producer
    reads it between forcing times and on each heartbeat.
    """

    root = Path(root)
    if prepared_tree_complete(root) or not (
            stream_dir(root) / HEAD_NAME).is_file():
        return False
    try:
        _write_json_atomic(stream_dir(root) / STOP_NAME, {
            "reason": str(reason), "stopped_utc": _utc_now()})
    except OSError:
        return False
    return True


#: The preparation threads :func:`run_chained` is running, by output root,
#: so a forecast in the same process can tell a producer that ended from
#: one that is still building (see :meth:`StreamedIntervals._producer_verdict`).
_LOCAL_PRODUCERS: dict[str, threading.Thread] = {}
_LOCAL_PRODUCERS_LOCK = threading.Lock()


def _root_key(root) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(root)))


def _local_producer(root) -> threading.Thread | None:
    with _LOCAL_PRODUCERS_LOCK:
        return _LOCAL_PRODUCERS.get(_root_key(root))


def _head_created_epoch(head: Mapping[str, object]) -> float | None:
    try:
        created = datetime.fromisoformat(str(head["created_utc"]))
    except (KeyError, TypeError, ValueError):
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created.timestamp()


def run_chained(*, prepared_root, prepare: Callable[[], object],
                forecast: Callable[[str | None], object],
                on_head: Callable[[str], None] | None = None,
                poll_seconds: float = 0.5):
    """Run ``prepare`` beside ``forecast``; the forecast starts at the head.

    The one orchestration every chain uses.  ``prepare`` runs the
    preparation to its seal on a worker thread (the preparation itself is
    whatever the chain already runs: a stage subprocess, or the preparer
    in a spawned process); this thread waits until
    ``<prepared_root>/boundary-stream/head.json`` is written by THIS
    preparation or the preparation ends, then calls
    ``forecast(head_sha256)`` for a chained head, or ``forecast(None)``
    for a tree that was published sealed or reused whole (so the caller
    binds its proof as before).

    Only a head created after this call started is bound, which is the
    breakage this prevents: a retry that reuses its output root finds the
    previous attempt's head there, and a forecast bound to it dies on that
    attempt's failure marker or runs on a preparation the retry is about
    to replace.

    A producer that fails reaches the waiting forecast through
    ``failed.json`` and, in this process, through the worker thread
    having ended.  A forecast that fails leaves the producer running to
    its seal and waits for it, so a retry reuses the complete preparation
    instead of building every forcing time again.  A stop asks the
    producer to stop (``stop.json``), waits for it to exit and re-raises:
    ``KeyboardInterrupt``, ``SystemExit``, and every exception
    :func:`gpuwm.runplan._is_interrupt` reads as the user's stop, which
    is how a Ctrl-C reaches this thread from ``gpuwm go``
    (``GoInterrupted``) and ``gpuwm run-plan``.  The breakage this
    prevents: a stopped ``gpuwm go`` reported a failed forecast, left the
    producer running and did not return until it sealed.

    Returns ``(prepare_result, forecast_result)``.
    """

    root = Path(prepared_root)
    box: dict[str, object] = {}
    started_epoch = time.time()

    def produce():
        try:
            box["result"] = prepare()
        except BaseException as error:  # noqa: BLE001 - re-raised below
            box["error"] = error
            _mark_failed_if_unsealed(root, error, since=started_epoch)

    # In the caller's context, as the preparation would be on the caller's
    # own thread: a host's output redirect (gpuwm go's launch log) is a
    # context variable, and a thread starts without it.  The breakage this
    # prevents: under `gpuwm go` a chained preparation missed the redirect,
    # so its preparer ran behind a second host with its own log and each
    # step was said twice under --explain.
    context = contextvars.copy_context()
    worker = threading.Thread(target=context.run, args=(produce,),
                              name="chained-preparation", daemon=True)
    key = _root_key(root)
    with _LOCAL_PRODUCERS_LOCK:
        _LOCAL_PRODUCERS[key] = worker
    try:
        worker.start()
        head_path = stream_dir(root) / HEAD_NAME
        head = None
        while worker.is_alive():
            if head_path.is_file():
                head = _fresh_head(root, since=started_epoch)
                if head is not None:
                    break
            worker.join(poll_seconds)
        if head is not None and not (head.get("decision") or {}).get(
                "chained", False):
            # Published at its seal: nothing to start early.
            head = None
        if head is None:
            worker.join()
            if "error" in box:
                raise box["error"]
            return box.get("result"), forecast(None)
        if on_head is not None:
            on_head(str(head["head_sha256"]))
        try:
            result = forecast(str(head["head_sha256"]))
        except BaseException as error:
            if isinstance(error, Exception) and not _is_stop(error):
                if worker.is_alive():
                    _say("prepare: the forecast failed; the preparation "
                         "continues to its seal so a retry reuses it")
                worker.join()
                raise
            request_stop(root, f"the forecast ended: "
                               f"{type(error).__name__}: {error}")
            worker.join()
            raise
        worker.join()
        if "error" in box:
            raise box["error"]
        return box.get("result"), result
    finally:
        with _LOCAL_PRODUCERS_LOCK:
            if _LOCAL_PRODUCERS.get(key) is worker:
                del _LOCAL_PRODUCERS[key]


def _is_stop(error: BaseException) -> bool:
    """Whether a forecast exception is the user's stop, not a failure."""

    from gpuwm.runplan import _is_interrupt

    return _is_interrupt(error)


def _fresh_head(root, *, since: float) -> dict | None:
    """The readable head in ``root`` if it was created at or after ``since``."""

    try:
        head = read_head(root)
    except BoundaryStreamError:
        return None
    created = _head_created_epoch(head)
    if created is None or created < since:
        return None
    return head


def _mark_failed_if_unsealed(root, error: BaseException, *,
                             since: float) -> None:
    """Write ``failed.json`` for a head this run published and never sealed.

    The preparation normally writes it itself (:meth:`PreparedTreeWriter
    .fail`); this covers the producer that could not, such as a stage
    subprocess killed from outside, so a forecast in another process ends
    by name instead of waiting for the heartbeat to go stale.
    """

    root = Path(root)
    stream = stream_dir(root)
    if (prepared_tree_complete(root) or (stream / FAILED_NAME).exists()
            or _fresh_head(root, since=since) is None):
        return
    try:
        _write_json_atomic(stream / FAILED_NAME, {
            "reason": str(error) or type(error).__name__,
            "stage": "producing",
            "error_type": type(error).__name__,
            "failed_utc": _utc_now(),
        })
    except OSError:
        pass


def _heartbeat_age(beat, *, now: float | None = None) -> float | None:
    if not isinstance(beat, dict):
        return None
    try:
        updated = float(beat["updated_epoch"])
    except (KeyError, TypeError, ValueError):
        return None
    return max(0.0, (time.time() if now is None else now) - updated)


def _same_process(beat) -> bool:
    return (isinstance(beat, dict) and beat.get("pid") == os.getpid()
            and beat.get("host") == socket.gethostname())


def _silence_limit(beat) -> float:
    slowest = 0.0
    if isinstance(beat, dict):
        try:
            slowest = float(beat.get("slowest_build_seconds") or 0.0)
        except (TypeError, ValueError):
            slowest = 0.0
    return max(SILENT_FLOOR_SECONDS, 3.0 * slowest)


# ---------------------------------------------------------------------------
# The producer
# ---------------------------------------------------------------------------


class _ConsoleAfterReader:
    """A console stream that outlives the process reading it.

    Writes go to ``stream`` until it reports the reader gone (a broken
    pipe); from then on they are dropped, since nothing is left to read
    them.  Everything else is the wrapped stream's.
    """

    def __init__(self, stream):
        self._stream = stream
        self.reader_gone = False

    def _lost(self) -> None:
        self.reader_gone = True
        self._stream = open(os.devnull, "w", encoding="utf-8")

    def write(self, text):
        if not self.reader_gone:
            try:
                return self._stream.write(text)
            except (BrokenPipeError, ConnectionResetError):
                self._lost()
        return len(text)

    def flush(self):
        if not self.reader_gone:
            try:
                self._stream.flush()
            except (BrokenPipeError, ConnectionResetError):
                self._lost()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def keep_console_after_reader_exit() -> None:
    """Let a chained producer survive the death of its console's reader.

    A preparation stage prints its progress into a pipe its parent reads,
    and in a ``gpuwm go`` chain that parent also hosts the forecast.  When
    that process dies (killed, out of memory, a driver crash), the
    producer's next line raised a broken pipe and the preparation failed,
    which is the breakage this prevents: a failed forecast leaves its
    producer running to the seal so a retry reuses it, and a checkpoint
    written before the seal resumes only on that same preparation.
    Idempotent, and a no-op for a console nobody closes.
    """

    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is not None and not isinstance(stream, _ConsoleAfterReader):
            setattr(sys, name, _ConsoleAfterReader(stream))


class PreparedTreeWriter:
    """Write a prepared tree as a head, one segment per interval, a seal.

    ``staging`` is the route's own staging directory and ``output_root``
    the name its single rename publishes.  With ``chained`` on,
    :meth:`write_head` performs that rename right after the head is
    written; otherwise :meth:`publish` performs it after ``proof.json``, so
    the unchained run is today's run through the same code.  Either way
    :attr:`root` is where the tree currently lives, and every path a route
    writes after the head is taken from it.
    """

    def __init__(self, *, staging, output_root, identity,
                 chained: bool | None = None,
                 sealed_forcing_extension: bool = False,
                 cache_name: str = "prepared-cache",
                 publish: Callable[[Path, Path], None] | None = None):
        from gpuwm.ingest.prepared_cache import PreparedCacheStream

        self.staging = Path(staging)
        self.output_root = Path(output_root)
        self.chained = chained_enabled() if chained is None else bool(chained)
        self.cache_name = str(cache_name)
        self._publish_tree = publish or (lambda src, dst: os.replace(src, dst))
        self.published = False
        self.head: dict | None = None
        self.head_sha256: str | None = None
        self._cache = PreparedCacheStream(
            self.staging / self.cache_name, identity=identity,
            sealed_forcing_extension=sealed_forcing_extension)
        self._heartbeat: _Heartbeat | None = None
        self._segments_written = 0
        self._build_seconds: list[float] = []
        self._decision: dict[str, object] = {"chained": self.chained}
        self.forecast_installed = forecast_installed()
        if not self.forecast_installed:
            # Nothing here can bind the head, and admission cannot be
            # priced without the forecast (see forecast_installed).
            self.chained = False
            self._decision = {"chained": False,
                              "reason": PREPARATION_ONLY_REASON}
        self.head_seconds: float | None = None
        self._started = time.perf_counter()

    @property
    def root(self) -> Path:
        return self.output_root if self.published else self.staging

    @property
    def cache_path(self) -> Path:
        return self.root / self.cache_name

    @property
    def stream_path(self) -> Path:
        return stream_dir(self.root)

    def decline_chaining(self, reason: str) -> None:
        """Publish at the seal instead of the head, with the named reason."""

        if self.head is not None:
            raise RuntimeError("chaining is decided before the head")
        if self.chained:
            _say(_sealed_line(reason))
        self.chained = False
        self._decision = {"chained": False, "reason": str(reason)}

    def admit(self, *, experiment, backend: str,
              device_bytes: int | None = None,
              card: tuple[int, int] | None = None) -> dict:
        """Admit the forecast and this producer on one machine, or decline.

        The breakage this prevents is both processes out of memory mid-run:
        a chained forecast starts on the card while the producer is still
        building forcing times.  A host (CPU) producer never shares the
        forecast's device and is admitted.  A CUDA producer is admitted
        when the forecast's admission estimate plus the producer's own
        measured holding for one forcing time (``device_bytes``, the
        memory pool after the start time was built) fits what the card
        can give both, with 10% headroom; otherwise the head is published
        at the seal and the run starts after preparation, as before.
        ``card`` is ``(free_bytes, pool_bytes)`` when the caller measured
        it; otherwise it is measured here.  An installation with no
        forecast has nothing to admit and keeps the decision it recorded
        at construction.
        """

        if not self.forecast_installed:
            return dict(self._decision)
        decision = chained_admission(
            experiment=experiment, backend=backend,
            device_bytes=device_bytes, card=card)
        if self.chained and not decision["admitted"]:
            self.decline_chaining(decision["reason"])
        self._decision = {"chained": self.chained, **decision}
        return dict(self._decision)

    def write_head(self, *, initial_result, met, surface=None, metadata=None,
                   lbc, proof_head: Mapping[str, object],
                   input_manifest_sha256: str | None = None,
                   reservation: Mapping[str, object] | None = None,
                   forcing=None) -> str:
        """Write everything the start time makes; publish it when chained.

        ``lbc`` is ``{"spec_bdy_width", "spec_zone", "relax_zone",
        "schedule": [[start, end], ...], "fields": [...]}``.  ``proof_head``
        is the route's proof document without the seal-only keys.
        ``forcing`` is the route's
        :class:`gpuwm.ingest.lateral_bc.StateBoundaryFrames` holding the
        start time, which prices one boundary interval for the host RAM
        admission; a chained head without it is not admitted, because the
        boundary series the forecast will hold would be unpriced.
        """

        stray = sorted(set(proof_head) & SEAL_ONLY_PROOF_KEYS)
        if stray:
            raise ValueError(f"the head proof carries seal-only keys {stray}")
        cache_head = self._cache.write_head(
            initial_result=initial_result, met=met, surface=surface,
            metadata=metadata, lbc=lbc)
        cache_head["directory"] = self.cache_name
        if self.chained:
            # Host RAM, on every backend: the forecast process holds the
            # head's arrays, every boundary interval it loads and its own
            # working set while the producer keeps building beside it.
            forecast = forecast_host_bytes(
                head_payload_bytes=int(cache_head["payload_bytes"]),
                interval_host_bytes=(None if forcing is None
                                     else forcing.interval_host_bytes),
                intervals=len(lbc["schedule"]))
            host = {**host_admission(forecast_bytes=forecast["total_bytes"]),
                    "forecast_host_parts": forecast}
            if not host["admitted"]:
                self.decline_chaining(host["reason"])
            self._decision = {**self._decision, "chained": self.chained,
                              "host": host}
        basis = {
            "schema": HEAD_SCHEMA,
            "cache": cache_head,
            "proof_head": json.loads(_canonical(proof_head)),
            "input_manifest_sha256": input_manifest_sha256,
        }
        head = {
            "schema": HEAD_SCHEMA,
            "basis": basis,
            "created_utc": _utc_now(),
            "decision": dict(self._decision),
            "reservation": dict(reservation or {}),
        }
        head["head_sha256"] = head_sha256(head)
        _write_json_atomic(self.stream_path / HEAD_NAME, head)
        (self.stream_path / SEGMENTS_DIRNAME).mkdir(parents=True, exist_ok=True)
        self.head = head
        self.head_sha256 = head["head_sha256"]
        self.head_seconds = time.perf_counter() - self._started
        if self.chained:
            # From here on a forecast may bind this head, so the producer
            # must outlive the process reading its console.
            keep_console_after_reader_exit()
            # The first heartbeat is written before the rename, so no
            # reader ever sees a published head without a producer.
            heartbeat = _Heartbeat(self.stream_path, self)
            self._publish_tree(self.staging, self.output_root)
            self.published = True
            self._cache.move(self.cache_path)
            heartbeat.path = self.stream_path
            self._heartbeat = heartbeat
            self._heartbeat.start()
            _say(f"prepare: head published at {self.output_root}; the "
                 "forecast may start while the remaining boundary "
                 "intervals are built")
        return self.head_sha256

    def note_build_seconds(self, seconds: float) -> None:
        """Record one forcing time's build wall (sizes the silence limit)."""

        self._build_seconds.append(float(seconds))

    @property
    def times_built(self) -> int:
        return len(self._build_seconds)

    @property
    def slowest_build_seconds(self) -> float:
        return max(self._build_seconds, default=0.0)

    def check_stop(self) -> None:
        """Raise when the consumer asked this producer to stop."""

        stop = _read_json(self.stream_path / STOP_NAME)
        if isinstance(stop, dict):
            raise BoundaryStreamStopped(
                "the forecast stopped this preparation: "
                f"{stop.get('reason', 'no reason given')}")

    def write_segment(self, index: int, interval) -> dict:
        """Write interval ``index``'s arrays, then its ready marker."""

        if self.head is None:
            raise RuntimeError("a segment needs its head first")
        self.check_stop()
        segment = self._cache.write_segment(int(index), interval)
        marker = {
            "schema": SEGMENT_SCHEMA,
            "head_sha256": self.head_sha256,
            **segment,
        }
        _write_json_atomic(segment_marker_path(self.root, index), marker)
        self._segments_written += 1
        return marker

    def stream_forcing_times(self, *, count: int,
                             build_forcing_time: Callable[[int], tuple],
                             forcing, times: Sequence,
                             release: Callable[[], None] | None = None
                             ) -> None:
        """Build forcing times 1..count-1 and write each interval as it closes.

        The one loop every route that builds boundaries from forcing times
        runs after its head.  ``build_forcing_time(index)`` returns the
        route's ``(met, initialized)`` for that time, built exactly as the
        route builds it; ``forcing`` is the route's
        :class:`gpuwm.ingest.lateral_bc.StateBoundaryFrames`, which already
        holds the start time.  Interval k is written as soon as time k+1
        exists, then time k is released, so at most two forcing times are
        held and the files are numbered in the order the one-shot writer
        numbered them.  ``release`` hands the build's device or pool memory
        back after each time.
        """

        from gpuwm.progress import prep_progress

        for index in range(1, int(count)):
            built = time.perf_counter()
            met, initialized = build_forcing_time(index)
            forcing.add_state(initialized.state, index=index)
            del met, initialized
            if release is not None:
                release()
            self.note_build_seconds(time.perf_counter() - built)
            self.write_segment(index - 1, forcing.interval(index - 1, times))
            forcing.release(index - 1)
            # Said per interval: this loop is most of a preparation's wall on
            # a long run, and a chained forecast is already stepping beside
            # it, so "boundary times 3 of 16 ready" is what a watcher needs.
            prep_progress("root_boundaries", label="Boundary times",
                          done=index, count=int(count) - 1)

    def write_intervals(self, intervals) -> None:
        """Write intervals a route already holds: a source already complete."""

        for index, interval in enumerate(intervals):
            self.write_segment(index, interval)

    def seal_cache(self) -> dict:
        """Write ``header.json``; return the one-shot writer's receipt."""

        return self._cache.seal()

    def publish(self, proof: dict) -> dict:
        """Write ``proof.json`` last (and the tree's rename when unchained)."""

        if self.head is None:
            raise RuntimeError("the seal needs its head first")
        stray = {key: value for key, value in proof.items()
                 if key not in SEAL_ONLY_PROOF_KEYS}
        if json.loads(_canonical(stray)) != self.head["basis"]["proof_head"]:
            differing = sorted(
                key for key in set(stray) | set(self.head["basis"]["proof_head"])
                if json.loads(_canonical(stray.get(key)))
                != self.head["basis"]["proof_head"].get(key))
            raise RuntimeError(
                "the sealed proof differs from the head it was published "
                f"under in {differing}")
        _write_json_atomic(self.root / PROOF_NAME, proof)
        if not self.published:
            self._publish_tree(self.staging, self.output_root)
            self.published = True
        self._stop_heartbeat(final="sealed")
        return proof

    def boundary_stream_proof(self) -> dict:
        """The one proof field the stream adds: the head it sealed."""

        return {"head_sha256": self.head_sha256}

    def fail(self, error: BaseException) -> None:
        """Record the producer's failure so a waiting forecast can end."""

        try:
            self._record_failure(error)
        finally:
            self._stop_heartbeat(
                final=("stopped" if isinstance(error, BoundaryStreamStopped)
                       else "failed"))

    def _record_failure(self, error: BaseException) -> None:
        if self.published:
            reason = (str(error) or type(error).__name__)
            stage = ("stopped" if isinstance(error, BoundaryStreamStopped)
                     else "producing")
            try:
                _write_json_atomic(self.stream_path / FAILED_NAME, {
                    "reason": reason,
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "traceback_tail": "".join(traceback.format_exception(
                        type(error), error, error.__traceback__))[-4000:],
                    "failed_utc": _utc_now(),
                })
            except (OSError, TypeError, ValueError):
                pass

    def _stop_heartbeat(self, *, final: str) -> None:
        if self._heartbeat is not None:
            self._heartbeat.stop(final=final)
            self._heartbeat = None


class _Heartbeat(threading.Thread):
    def __init__(self, path: Path, writer: PreparedTreeWriter):
        super().__init__(name="boundary-stream-heartbeat", daemon=True)
        self.path = Path(path)
        self.writer = writer
        self._done = threading.Event()
        self.beat("producing")

    def beat(self, state: str) -> None:
        try:
            _write_json_atomic(self.path / PRODUCER_NAME, {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "state": state,
                "times_built": self.writer.times_built,
                "segments_written": self.writer._segments_written,
                "slowest_build_seconds": self.writer.slowest_build_seconds,
                "updated_utc": _utc_now(),
                "updated_epoch": time.time(),
            })
        except OSError:
            pass

    def run(self) -> None:
        while not self._done.wait(HEARTBEAT_SECONDS):
            self.beat("producing")

    def stop(self, *, final: str) -> None:
        self._done.set()
        self.beat(final)


# ---------------------------------------------------------------------------
# The consumer
# ---------------------------------------------------------------------------


class StreamedIntervals(Sequence):
    """The lazy interval sequence over a prepared tree's segments.

    ``len`` is the declared count and :attr:`bounds` the schedule, so an
    interval search never loads an interval it does not return.
    ``self[k]`` waits for segment k's marker, loads and hash-checks its
    arrays against the marker's manifest rows, validates it against
    interval 0's layout, and returns the same object on every later call
    (the device slot reloads by object identity).  Iteration walks every
    index, so a consumer that needs the whole set simply waits for it.
    """

    def __init__(self, root, *, head: Mapping[str, object],
                 on_wait: Callable[[dict | None], None] | None = None,
                 poll_seconds: float = 0.2,
                 validate: Callable[[object], None] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.root = Path(root)
        self.head = head
        self.head_sha256 = str(head["head_sha256"])
        cache = head["basis"]["cache"]
        lbc = cache.get("lbc")
        if not isinstance(lbc, dict) or not lbc.get("schedule"):
            raise BoundaryStreamError(
                f"{self.root} declares no external boundary schedule")
        self.lbc = lbc
        self.bounds = tuple(
            (float(start), float(end)) for start, end in lbc["schedule"])
        self.fields = tuple(lbc["fields"])
        self.cache_path = self.root / str(cache["directory"])
        self.on_wait = on_wait
        self.poll_seconds = float(poll_seconds)
        self.validate = validate
        self._clock = clock
        self._loaded: dict[int, object] = {}
        self._markers: dict[int, dict] = {}
        self._lock = threading.RLock()
        #: One entry per wait the run actually took: (index, seconds).
        self.waits: list[tuple[int, float]] = []

    # Sequence protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self.bounds)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[k] for k in range(*index.indices(len(self))))
        k = int(index)
        if k < 0:
            k += len(self)
        if not 0 <= k < len(self):
            raise IndexError(index)
        with self._lock:
            cached = self._loaded.get(k)
            if cached is not None:
                return cached
            marker = self.require(k)
            interval = self._load(k, marker)
            self._loaded[k] = interval
            return interval

    def __iter__(self):
        for k in range(len(self)):
            yield self[k]

    # Readiness ---------------------------------------------------------

    def is_ready(self, index: int) -> bool:
        return segment_marker_path(self.root, index).is_file()

    def ready_prefix(self) -> int:
        """How many intervals, from the first, have their markers."""

        count = 0
        while count < len(self) and self.is_ready(count):
            count += 1
        return count

    def sealed(self) -> bool:
        return prepared_tree_complete(self.root)

    def _wait_reason(self, index: int) -> str:
        start, end = self.bounds[index]
        return (f"boundary interval {index} ({start:g} s to {end:g} s) is "
                "not prepared yet")

    def _producer_verdict(self, index: int,
                          ready: Callable[[], bool] | None = None) -> None:
        stream = stream_dir(self.root)
        failed = _read_json(stream / FAILED_NAME)
        if isinstance(failed, dict):
            raise BoundaryProducerFailed(
                "the boundary producer failed: "
                f"{failed.get('reason', 'no reason recorded')}")
        worker = _local_producer(self.root)
        if (worker is not None and not worker.is_alive()
                and not (ready is not None and ready())):
            # The preparation thread of this process ended and what this
            # wait needs never arrived: a producer that could not even
            # write failed.json (a full disk) must not leave the run
            # waiting forever.
            raise BoundaryProducerFailed(
                "the boundary producer ended without preparing interval "
                f"{index}")
        beat = _read_json(stream / PRODUCER_NAME)
        if isinstance(beat, dict) and beat.get("state") in {"failed",
                                                            "stopped"}:
            raise BoundaryProducerFailed(
                f"the boundary producer {beat.get('state')} after building "
                f"time {beat.get('times_built')}; interval {index} will "
                "never arrive")
        age = _heartbeat_age(beat)
        if _same_process(beat):
            # A producer on a thread of this process is judged by the
            # thread itself (above), and a long call holding the
            # interpreter would stall its heartbeat and this check alike:
            # its age proves nothing here.
            age = None
        if age is not None and age > _silence_limit(beat):
            built = beat.get("times_built") if isinstance(beat, dict) else None
            raise BoundaryProducerSilent(
                f"boundary producer silent for {age:.0f} s after building "
                f"time {built}; interval {index} will never arrive")
        if beat is None and not (stream / HEAD_NAME).is_file():
            raise BoundaryProducerSilent(
                f"{self.root} has no boundary producer and no marker for "
                f"interval {index}")

    def require(self, index: int) -> dict:
        """Wait for segment ``index``'s marker and return it."""

        k = int(index)
        marker = self._markers.get(k)
        if marker is not None:
            return marker
        path = segment_marker_path(self.root, k)
        started = None
        last_report = None
        while not path.is_file():
            now = self._clock()
            if started is None:
                started = now
            if last_report is None or now - last_report >= WAIT_REPORT_SECONDS:
                last_report = now
                if self.on_wait is not None:
                    self.on_wait({"reason": self._wait_reason(k),
                                  "interval": k,
                                  "waited_seconds": now - started})
            self._producer_verdict(k, ready=path.is_file)
            time.sleep(self.poll_seconds)
        if started is not None:
            waited = self._clock() - started
            self.waits.append((k, waited))
            if self.on_wait is not None:
                self.on_wait(None)
        marker = _read_json(path)
        if (not isinstance(marker, dict)
                or marker.get("schema") != SEGMENT_SCHEMA
                or marker.get("head_sha256") != self.head_sha256
                or int(marker.get("index", -1)) != k):
            raise BoundaryStreamError(
                f"segment marker {path} does not belong to head "
                f"{self.head_sha256}")
        if [float(marker["start_seconds"]), float(marker["end_seconds"])] \
                != list(self.bounds[k]):
            raise BoundaryStreamError(
                f"segment {k} bounds differ from the head's schedule")
        self._markers[k] = marker
        return marker

    def wait_sealed(self, *, timeout: float | None = None) -> None:
        """Wait for ``proof.json``; the producer's verdict ends a dead wait."""

        started = self._clock()
        last_report = None
        while not self.sealed():
            now = self._clock()
            if last_report is None or now - last_report >= WAIT_REPORT_SECONDS:
                last_report = now
                if self.on_wait is not None:
                    self.on_wait({
                        "reason": "the preparation is not sealed yet",
                        "interval": None,
                        "waited_seconds": now - started})
            self._producer_verdict(len(self) - 1, ready=self.sealed)
            if timeout is not None and now - started > timeout:
                raise BoundaryStreamError(
                    f"{self.root} was not sealed within {timeout:g} s")
            time.sleep(self.poll_seconds)
        if self.on_wait is not None and last_report is not None:
            self.on_wait(None)

    def stop(self, reason: str) -> None:
        """Ask the producer to exit unsealed (the run ended first)."""

        request_stop(self.root, reason)

    # Loading -----------------------------------------------------------

    def _load(self, k: int, marker: Mapping[str, object]):
        from gpuwm.ingest.lateral_bc import (
            BoundaryInterval, FieldBoundary, RationalTimeLaw, SideBoundary,
        )
        from gpuwm.ingest.prepared_cache import (
            PreparedCacheCorruptError, read_manifest_array,
        )

        arrays = marker["arrays"]
        if sorted(marker["fields"]) != sorted(self.fields):
            raise BoundaryStreamError(
                f"segment {k} carries fields {sorted(marker['fields'])}, "
                f"not the head's {sorted(self.fields)}")
        fields = {}
        for name in marker["fields"]:
            sides = {}
            for side_name in ("west", "east", "south", "north"):
                prefix = f"lbc/{k}/{name}/{side_name}"
                laws = [f"{prefix}/rational_time_v1/{coefficient}"
                        for coefficient in ("quadratic", "denominator_rate")]
                present = [key in arrays for key in laws]
                if any(present) and not all(present):
                    raise PreparedCacheCorruptError(
                        f"segment {k} {prefix} has an incomplete rational "
                        "time law")
                law = (RationalTimeLaw(*(read_manifest_array(
                    self.cache_path, key, arrays[key]) for key in laws))
                    if all(present) else None)
                sides[side_name] = SideBoundary(
                    read_manifest_array(self.cache_path, f"{prefix}/value",
                                        arrays[f"{prefix}/value"]),
                    read_manifest_array(self.cache_path, f"{prefix}/tendency",
                                        arrays[f"{prefix}/tendency"]),
                    law)
            fields[name] = FieldBoundary(**sides)
        interval = BoundaryInterval(
            float(marker["start_seconds"]), float(marker["end_seconds"]),
            fields)
        if k > 0:
            _require_same_layout(self[0], interval, k)
        if self.validate is not None:
            self.validate(interval)
        return interval

    def consumed_markers(self) -> dict[int, dict]:
        return dict(self._markers)


def _require_same_layout(first, interval, index: int) -> None:
    from gpuwm.ingest.lateral_bc import _boundary_field_shape

    def layout(value):
        return tuple(
            (name,) + _boundary_field_shape(value.fields[name]) + tuple(
                getattr(value.fields[name], side).time_law is not None
                for side in ("west", "east", "south", "north"))
            for name in sorted(value.fields))

    if layout(interval) != layout(first):
        raise BoundaryStreamError(
            f"boundary interval {index} has a different inventory or side "
            "layout than interval 0")


def streamed_boundaries(root, *, head, on_wait=None, validate=None):
    """A :class:`LateralBoundaries` whose intervals stream from ``root``."""

    from gpuwm.ingest.lateral_bc import LateralBoundaries

    intervals = StreamedIntervals(root, head=head, on_wait=on_wait,
                                  validate=validate)
    lbc = intervals.lbc
    return LateralBoundaries(
        intervals, int(lbc["spec_bdy_width"]), int(lbc["spec_zone"]),
        int(lbc["relax_zone"]))


def verify_seal(root, *, head: Mapping[str, object],
                consumed: Mapping[int, Mapping[str, object]] | None = None
                ) -> dict:
    """Check a sealed tree against the head it was published under.

    The header's ``content_sha256`` must equal the recomputation from the
    head's arrays plus every segment marker, ``proof.json`` must name this
    head, and every marker a forecast consumed must be the marker the seal
    counted.  Returns ``{"proof_sha256", "content_sha256", "head_sha256"}``.
    """

    from gpuwm.ingest.prepared_cache import PREPARED_CACHE_SCHEMA

    root = Path(root)
    head_digest = str(head["head_sha256"])
    proof_path = root / PROOF_NAME
    proof = _read_json(proof_path)
    if not isinstance(proof, dict):
        raise BoundaryStreamError(f"{proof_path} is not readable")
    named = (proof.get("boundary_stream") or {}).get("head_sha256")
    if named != head_digest:
        raise BoundaryStreamError(
            f"{proof_path} seals head {named}, not the pinned head "
            f"{head_digest}")
    stray = {key: value for key, value in proof.items()
             if key not in SEAL_ONLY_PROOF_KEYS}
    if json.loads(_canonical(stray)) != head["basis"]["proof_head"]:
        raise BoundaryStreamError(
            f"{proof_path} differs from the head proof outside the seal keys")
    cache = head["basis"]["cache"]
    cache_path = root / str(cache["directory"])
    header = _read_json(cache_path / "header.json")
    if not isinstance(header, dict) or header.get("schema") \
            != PREPARED_CACHE_SCHEMA:
        raise BoundaryStreamError(f"{cache_path} has no sealed header")
    arrays = dict(cache["arrays"])
    payload = int(cache["payload_bytes"])
    for k in range(len(cache["lbc"]["schedule"])):
        marker = _read_json(segment_marker_path(root, k))
        if not isinstance(marker, dict) \
                or marker.get("head_sha256") != head_digest:
            raise BoundaryStreamError(f"segment {k} marker is missing at seal")
        if consumed and k in consumed and consumed[k] != marker:
            raise BoundaryStreamError(
                f"segment {k} changed after the forecast consumed it")
        arrays.update(marker["arrays"])
        payload += int(marker["payload_bytes"])
    if header.get("arrays") != arrays or int(header.get("payload_bytes", -1)) \
            != payload:
        raise BoundaryStreamError(
            "the sealed header's array table is not the head plus its "
            "segments")
    basis = {key: header[key] for key in (
        "schema", "identity", "metadata", "arrays", "payload_bytes")}
    content = hashlib.sha256(_canonical(basis).encode("utf-8")).hexdigest()
    if content != header.get("content_sha256"):
        raise BoundaryStreamError("the sealed header fails its content digest")
    if header.get("identity") != cache["identity"]:
        raise BoundaryStreamError("the sealed header names another identity")
    for key, value in cache["metadata"].items():
        if header["metadata"].get(key) != value:
            raise BoundaryStreamError(
                f"the sealed header metadata {key!r} differs from the head")
    digest = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    return {"proof_sha256": digest, "content_sha256": content,
            "head_sha256": head_digest}


__all__ = [
    "BoundaryProducerFailed", "BoundaryProducerSilent", "BoundaryStreamError",
    "BoundaryStreamStopped", "CHAINED_DEFAULT", "CHAINED_ENV", "FAILED_NAME",
    "FORECAST_HOST_FLOOR_BYTES", "HEAD_NAME", "HEAD_SCHEMA", "PRODUCER_NAME",
    "PreparedTreeWriter", "SEAL_ONLY_PROOF_KEYS", "SEGMENT_SCHEMA",
    "STOP_NAME", "STREAM_DIRNAME", "StreamedIntervals", "bind_head",
    "chained_enabled", "forecast_host_bytes", "forecast_installed",
    "head_sha256", "host_admission", "live_chained_head",
    "prepared_tree_complete",
    "process_memory_bytes", "read_head", "remove_unfinished_tree",
    "SEALED_REASONS", "request_stop", "run_chained", "say_prepared_sealed",
    "segment_marker_path", "stream_dir", "streamed_boundaries",
    "unfinished_tree_reason", "verify_seal",
]
