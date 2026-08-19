"""Bounded-concurrency transfer pool for multi-file acquisitions.

The scaling study (ARWEN-PRESIM-SCALING-2026-08-16) measured the cold
fetch stage at 0.47 s + 3.14 s **per file**, serial: file count dominates
and bytes barely matter, because each request spends most of its wall
clock waiting on the service, not on the pipe.  Sources that publish one
object per field per lead (ICON-EU: 251 objects per state; GEM: 351)
turn that per-file constant into the whole pre-sim budget.  The remedy
is to keep several requests in flight, bounded, without weakening a
single integrity property of the serial transport.

This module is the one shared implementation of that remedy.  It moves
no bytes itself: a :class:`TransferJob` carries an ``action`` callable
that downloads AND verifies one file exactly as the serial loop did
(envelope walk, record bar, sha256 -- whatever that route's bars are),
returning the manifest entry for it.  What the pool owns is scheduling
and bookkeeping:

* **Bounded workers.**  ``workers`` file transfers at most are in
  flight; :data:`DEFAULT_FILE_WORKERS` is the engine default and 1 is
  the serial transport, byte-for-byte the old loop (no threads are
  created at all).
* **Per-host politeness.**  A same-host cap bounds how many of those
  workers may target one host.  NOMADS is capped at
  :data:`NOMADS_FILE_WORKER_CAP` regardless of the pool size: the
  service is fragile, server-limited, and every request to it is
  *additionally* paced by the node-wide 2.5 s governor
  (:mod:`gpuwm.nomads_governor`), which this pool never bypasses --
  concurrency there only overlaps in-flight service time, it never
  raises the request rate.
* **In-order admission.**  ``on_admitted`` fires on the caller's
  thread, in submission order, as the verified prefix grows -- so a
  route's per-file manifest publication keeps its exact serial
  semantics (an interrupted fetch still records a contiguous verified
  prefix, and receipts are never written from two threads).
* **Fail-closed per file.**  The first failed job (in submission
  order) fails the whole request: jobs not yet started are cancelled,
  jobs in flight run to completion (their files stay on disk,
  unclaimed by any receipt, and the next run re-verifies them under
  the ordinary bars), and the original refusal -- whose message names
  the file -- is re-raised unchanged.
* **An honest receipt.**  Files, bytes, workers requested and
  effective, the per-host caps that actually bound this run, wall
  seconds, the serial model (the sum of per-file seconds), and the
  effective speedup against it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
from typing import Callable
from urllib.parse import urlsplit

#: The engine default: how many file transfers are in flight at once.
#: Chosen against the measured fetch shape -- per-file service latency
#: dominates, so a handful of overlapped requests recovers most of the
#: idle pipe without presenting as a burst to any provider.
DEFAULT_FILE_WORKERS = 6

#: Same-host politeness cap for NOMADS, applied on top of the node-wide
#: request-spacing governor.  Two is deliberate: the governor spaces
#: request *starts* 2.5 s apart node-wide, and the grib-filter service
#: answers in ~3.1 s, so a second in-flight request is enough to keep
#: the pipeline full -- more would only queue on the governor while
#: holding connections open against a fragile public service.
NOMADS_FILE_WORKER_CAP = 2

#: Hosts with a politeness cap tighter than the pool size.
HOST_FILE_WORKER_CAPS = {
    "nomads.ncep.noaa.gov": NOMADS_FILE_WORKER_CAP,
}

#: Schema of the receipt :func:`run_transfers` returns.
POOL_RECEIPT_SCHEMA = "gpuwm-fetch-pool-receipt-v1"


@dataclass(frozen=True)
class TransferJob:
    """One file's transfer-and-verify unit of work.

    ``name`` is what a refusal names.  ``url`` exists for host
    politeness only (None means no host, so no cap); the ``action``
    already knows where its bytes come from.  ``action`` returns the
    route's manifest entry for the file and raises the route's own
    refusal on any failure -- the pool adds no bars and removes none.
    """

    name: str
    url: str | None
    action: Callable[[], dict]


def host_key(url: str | None) -> str:
    """The politeness key for ``url``: its lower-cased netloc."""

    if not url:
        return ""
    return urlsplit(url).netloc.lower()


def host_worker_cap(host: str, workers: int) -> int:
    """How many of ``workers`` may target ``host`` at once."""

    cap = HOST_FILE_WORKER_CAPS.get(host)
    if cap is None:
        return workers
    return min(cap, workers)


def resolve_file_workers(requested: int | None) -> int:
    """The file-worker count a route runs with.

    ``None`` is the engine default.  1 is the serial transport -- a
    first-class knob, not a workaround.  Zero or a negative count names
    no schedulable pool and refuses.
    """

    if requested is None:
        return DEFAULT_FILE_WORKERS
    workers = int(requested)
    if workers < 1:
        raise ValueError(
            f"--fetch-workers {requested} is not a schedulable pool; pass "
            "a positive count (1 is the serial transport)")
    return workers


def _receipt(*, entries: list[dict], workers_requested: int,
             workers_effective: int, host_caps: dict[str, int],
             wall_seconds: float, serial_seconds: float) -> dict:
    total_bytes = sum(
        entry["bytes"] for entry in entries
        if isinstance(entry.get("bytes"), int)
        and not isinstance(entry.get("bytes"), bool))
    speedup = (round(serial_seconds / wall_seconds, 2)
               if wall_seconds > 0.0 else None)
    return {
        "schema": POOL_RECEIPT_SCHEMA,
        "files": len(entries),
        "bytes": total_bytes,
        "workers_requested": workers_requested,
        "workers_effective": workers_effective,
        "host_caps": dict(host_caps),
        "wall_seconds": round(wall_seconds, 6),
        "modeled_serial_seconds": round(serial_seconds, 6),
        "effective_speedup": speedup,
    }


def run_transfers(jobs, *, workers: int,
                  on_admitted: Callable[[int, dict], None] | None = None
                  ) -> tuple[list[dict], dict]:
    """Run every job; return ``(entries, receipt)`` in submission order.

    ``on_admitted(index, entry)`` fires on the caller's thread, in
    submission order, as each verified file joins the contiguous
    admitted prefix -- publish receipts there exactly as the serial
    loop did after each file.

    Failure semantics: the earliest submitted job that failed re-raises
    its original exception after the verified prefix before it has been
    admitted.  Later jobs are cancelled if not yet started; ones in
    flight finish (their files stay on disk, unclaimed).  A
    ``KeyboardInterrupt`` -- whether it lands here or inside an action
    -- propagates the same way, so route-level interrupt handling keeps
    working unchanged.
    """

    jobs = list(jobs)
    workers = resolve_file_workers(workers)
    workers_effective = min(workers, len(jobs)) if jobs else 0
    seen_hosts = {host_key(job.url) for job in jobs}
    host_caps = {
        host: host_worker_cap(host, workers)
        for host in sorted(seen_hosts)
        if host and host_worker_cap(host, workers) < workers_effective}
    entries: list[dict] = []
    serial_seconds = 0.0
    started = time.perf_counter()

    def admit(index: int, entry: dict) -> None:
        entries.append(entry)
        if on_admitted is not None:
            on_admitted(index, entry)

    if workers_effective <= 1:
        # The serial transport: the caller's thread, in order, no
        # thread machinery at all -- byte-for-byte the old loop.
        for index, job in enumerate(jobs):
            job_started = time.perf_counter()
            entry = job.action()
            serial_seconds += time.perf_counter() - job_started
            admit(index, entry)
        return entries, _receipt(
            entries=entries, workers_requested=workers,
            workers_effective=workers_effective, host_caps=host_caps,
            wall_seconds=time.perf_counter() - started,
            serial_seconds=serial_seconds)

    semaphores = {
        host: threading.BoundedSemaphore(host_worker_cap(host, workers))
        for host in seen_hosts if host}
    timing_lock = threading.Lock()

    def timed(job: TransferJob) -> dict:
        nonlocal serial_seconds
        gate = semaphores.get(host_key(job.url))
        if gate is not None:
            gate.acquire()
        try:
            job_started = time.perf_counter()
            entry = job.action()
            elapsed = time.perf_counter() - job_started
        finally:
            if gate is not None:
                gate.release()
        with timing_lock:
            serial_seconds += elapsed
        return entry

    executor = ThreadPoolExecutor(
        max_workers=workers_effective,
        thread_name_prefix="gpuwm-fetch")
    futures = []
    try:
        futures = [executor.submit(timed, job) for job in jobs]
        for index, future in enumerate(futures):
            # In submission order: the admitted prefix is contiguous by
            # construction, and the earliest submitted failure is the
            # one that propagates -- deterministic under any completion
            # order.
            entry = future.result()
            admit(index, entry)
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        # In-flight actions finish before control returns: files a
        # worker is still writing must never race the caller's own
        # disk scan (the interrupt path reads the directory).
        executor.shutdown(wait=True)
    return entries, _receipt(
        entries=entries, workers_requested=workers,
        workers_effective=workers_effective, host_caps=host_caps,
        wall_seconds=time.perf_counter() - started,
        serial_seconds=serial_seconds)


__all__ = [
    "DEFAULT_FILE_WORKERS", "HOST_FILE_WORKER_CAPS",
    "NOMADS_FILE_WORKER_CAP", "POOL_RECEIPT_SCHEMA", "TransferJob",
    "host_key", "host_worker_cap", "resolve_file_workers", "run_transfers",
]
