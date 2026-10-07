"""Host-parallel LETKF: the chunk loop of :func:`gpuwm.da.letkf.analyze` on
many cores, producing the same bytes as one core.

WHY.  The host solve (``solve_device="host"``, and the host half of the
host-staged device solve) walked its chunks one after another in one
process.  On a 241 x 241 x 49 grid with 32 members that is one core busy
for tens of minutes while the rest of a 64-core box idles.  The local
transform at one gridpoint never reads another gridpoint's answer, so the
chunks are independent work.

HOW, and why it is bit-identical.  The parent plans the chunk sequence
exactly as the single-process loop does (same chunk size, same span
splits, same reach rejection), then forks worker processes that each run
the unchanged per-chunk code on whole chunks and write their increments
straight into shared memory at the chunk's own gridpoints.  Nothing is
re-associated across gridpoints, so the only way a byte could move is a
gridpoint's own arithmetic changing, and two bounded splits are the only
places that could happen:

* phase 1 (weights) walks a chunk in row blocks.  Every weight is an
  elementwise function of one (gridpoint, slot) pair and the positive
  roster is read off row-major, so concatenating the blocks' rosters in row
  order is the whole chunk's roster.
* phase 2 (transform) walks each packed solve in pieces of whole rows at
  the solve's own width, so each row's packed arrays are exactly the ones
  the unsplit solve builds; numpy's batched matmul and eigh are per-matrix
  loops, and every other step is elementwise or reduces over the member or
  slot axis of one row.  A piece is never one row unless the solve is.

Both claims are measured, not assumed: tests/test_letkf_host_parallel.py
compares every worker count against the single-process loop byte for byte,
and the real-case receipt does the same at 8, 32 and 64 workers.

Memory.  The configured ``memory_budget_mib`` is the solve's scratch
promise.  Workers share it: each gets ``budget / workers`` for its row
blocks and pieces, so the sum stays the promise the single-process loop
made.  The increments live once, in shared memory, and are returned as
ordinary numpy arrays over it.

Fork is required (the workers inherit the prior, the observations and the
closures without a copy).  Where fork is unavailable the loop runs in one
process and the receipt says why.  Workers never touch a device: this path
is taken only when the solve namespace is numpy.
"""
from __future__ import annotations

import mmap
import multiprocessing
import os

import numpy as np

#: Automatic worker count never exceeds this.  Past 64 the chunk count of a
#: CONUS-shaped analysis no longer divides evenly enough to pay, and a shared
#: box has other tenants.  An explicit request or GPUWM_DA_HOST_WORKERS is
#: not capped.
HOST_WORKERS_AUTO_CAP = 64

#: Environment override for the worker count (an integer >= 1).
HOST_WORKERS_ENV = "GPUWM_DA_HOST_WORKERS"

#: Automatic mode stays in one process below this many (gridpoint, stencil
#: slot) pairs in the planned chunks: forking the workers costs tens of
#: milliseconds each, which a small analysis (every unit test, a 33 x 33
#: smoke) does not repay.  A 241 x 241 x 49 analysis is three orders of
#: magnitude above it.  An explicit worker count is always honoured.
HOST_PARALLEL_MIN_PAIRS = 200_000_000

_TASK = None


def fork_available() -> bool:
    try:
        return "fork" in multiprocessing.get_all_start_methods()
    except Exception:                                     # pragma: no cover
        return False


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):                      # pragma: no cover
        return int(os.cpu_count() or 1)


def resolve_host_workers(requested, *, tasks: int,
                         pairs: int = 0) -> tuple[int, str]:
    """``(workers, reason)`` for a host solve with ``tasks`` chunks to run.

    ``pairs`` is the (gridpoint, stencil slot) count of those chunks, the
    size the automatic mode judges the analysis by.
    """
    env = os.environ.get(HOST_WORKERS_ENV, "").strip()
    if env:
        try:
            value = int(env)
        except ValueError:
            value = 0
        if value < 1:
            raise ValueError(
                f"{HOST_WORKERS_ENV} must be an integer >= 1, got {env!r}")
        want, source = value, f"{HOST_WORKERS_ENV}={value}"
    elif requested is not None:
        want, source = int(requested), f"host_workers={int(requested)}"
    else:
        cpus = available_cpus()
        want = min(cpus, HOST_WORKERS_AUTO_CAP)
        source = (f"automatic: {cpus} CPUs in this process's affinity, "
                  f"capped at {HOST_WORKERS_AUTO_CAP}")
        if int(pairs) < HOST_PARALLEL_MIN_PAIRS:
            return 1, (f"automatic: {int(pairs)} gridpoint-slot pairs is "
                       f"below {HOST_PARALLEL_MIN_PAIRS}, so one process")
    if want <= 1:
        return 1, source
    if tasks <= 1:
        return 1, f"{source}; {tasks} chunk(s) to solve, so one process"
    if not fork_available():
        return 1, f"{source}; fork is unavailable here, so one process"
    workers = min(want, int(tasks))
    if workers < want:
        return workers, f"{source}; limited to the {tasks} chunks to solve"
    return workers, source


def shared_copy(array: np.ndarray) -> np.ndarray:
    """A C-contiguous copy of ``array`` in anonymous shared memory.

    Forked workers write into it and the parent sees the writes.  The
    returned array keeps its mapping alive for as long as it is referenced.
    """
    array = np.asarray(array)
    nbytes = max(1, int(array.nbytes))
    buf = mmap.mmap(-1, nbytes)
    out = np.frombuffer(buf, dtype=array.dtype,
                        count=int(array.size)).reshape(array.shape)
    if array.size and np.any(array):
        np.copyto(out, array)
    return out


def _initializer() -> None:
    # One BLAS thread per worker: the workers ARE the parallelism, and a
    # BLAS pool per worker would oversubscribe the cores many times over.
    try:
        from threadpoolctl import threadpool_limits     # noqa: PLC0415
        threadpool_limits(1)
    except Exception:
        pass


def _call(index):
    return _TASK(index)


class HostWorkerLost(RuntimeError):
    """A forked worker died (killed or crashed) before returning its chunk.

    The breakage this names: a ``multiprocessing.Pool`` waits forever for a
    task whose worker was SIGKILLed (the kernel's OOM killer) or segfaulted,
    so one lost worker hung the whole cycle -- holding every card lock --
    until the box's dead-man timer fired (reproduced on Python 3.12.3).  The
    executor below reports the loss instead, and the analysis falls back to
    its single-process loop.
    """


def run(task, order, workers: int):
    """Yield ``task(i)`` for each ``i`` in ``order`` from ``workers`` forks.

    ``task`` is inherited through fork, never pickled, so it may be a
    closure over the whole analysis.  Results arrive in completion order.
    An exception in a worker is re-raised here and the pool is torn down;
    a worker that dies without returning raises :class:`HostWorkerLost`.
    """
    from concurrent.futures import (FIRST_COMPLETED,  # noqa: PLC0415
                                    ProcessPoolExecutor, wait)
    from concurrent.futures.process import BrokenProcessPool  # noqa: PLC0415

    global _TASK
    if _TASK is not None:
        raise RuntimeError("a host-parallel analysis is already running in "
                           "this process; analyses do not nest")
    _TASK = task
    pool = None
    try:
        context = multiprocessing.get_context("fork")
        pool = ProcessPoolExecutor(max_workers=int(workers),
                                   mp_context=context,
                                   initializer=_initializer)
        pending = {pool.submit(_call, index) for index in list(order)}
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
        except BrokenProcessPool as exc:
            raise HostWorkerLost(
                "a host-parallel LETKF worker died before returning its "
                f"chunk ({exc})") from exc
        finally:
            for future in pending:
                future.cancel()
    finally:
        _TASK = None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


#: Threads for the whole-field passes (validation, perturbations, spread
#: diagnostics).  Each pass streams a (members, nz, ny, nx) field through
#: memory, so past a handful of fields at once the memory bus, not the core
#: count, is the limit; every concurrent field also holds its own
#: whole-field temporaries.
FIELD_THREADS_CAP = 16


def field_threads(fields: int) -> int:
    """Threads for ``fields`` whole-field host passes.

    The worker override applies here too, so ``GPUWM_DA_HOST_WORKERS=1``
    puts the whole analysis back in one thread.
    """
    env = os.environ.get(HOST_WORKERS_ENV, "").strip()
    cap = FIELD_THREADS_CAP
    if env:
        try:
            cap = max(1, min(cap, int(env)))
        except ValueError:
            cap = 1
    return max(1, min(int(fields), available_cpus(), cap))


def map_ordered(fn, items, threads: int):
    """``[fn(x) for x in items]`` on up to ``threads`` threads, in order.

    An exception surfaces for the first failing item in order, as the
    plain loop would raise it.
    """
    items = list(items)
    if threads <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor   # noqa: PLC0415

    with ThreadPoolExecutor(max_workers=int(threads)) as pool:
        return list(pool.map(fn, items))


#: Transform piece and phase-1 row block sizes, in bytes of the solve's own
#: conservative price (``_packed_bytes_per_point`` per row for a piece, the
#: host geometry price per row for a block).  Any size gives the same bytes;
#: these decide speed.  A piece or block that fits near the worker's cache
#: stops 64 workers from fighting over the memory bus: on the 241 x 241 x 49,
#: 32-member synthetic case at 64 workers, the solve took 118 s with each
#: worker's whole budget share (224 MiB) as its piece and block, 36 s with
#: 16 MiB pieces and 4-8 MiB blocks, and 48 s with 4 MiB pieces and 2 MiB
#: blocks, where per-call overhead takes over.  Never above the worker's
#: share of the budget.
PIECE_BYTES_DEFAULT = 16 << 20
BLOCK_BYTES_DEFAULT = 8 << 20

#: Tuning overrides for the two sizes, in MiB.
PIECE_MIB_ENV = "GPUWM_DA_HOST_PIECE_MIB"
BLOCK_MIB_ENV = "GPUWM_DA_HOST_BLOCK_MIB"


def _sized(env_name: str, default: int, per_worker: int) -> int:
    env = os.environ.get(env_name, "").strip()
    want = int(float(env) * (1 << 20)) if env else int(default)
    return max(1, min(int(per_worker), want))


def piece_bytes(per_worker: int) -> int:
    return _sized(PIECE_MIB_ENV, PIECE_BYTES_DEFAULT, per_worker)


def block_bytes(per_worker: int) -> int:
    return _sized(BLOCK_MIB_ENV, BLOCK_BYTES_DEFAULT, per_worker)
