"""Whole-ensemble statistics on many host cores, with numpy's own bytes.

The cycle driver summarises every analysis on the host: which fields have
ensemble spread (the field selection), the increments' size and support
(the leg record), and so on.  Each of those was one numpy call over a
whole-ensemble float64 stack on one core -- on the 9 km CONUS case
(598 x 351 x 49, 32 members, fourteen fields) about 80 s of every
analysis while the box's other cores and every card sat idle.

Every function here returns exactly what the single numpy call returns,
byte for byte, while spreading the work over threads (numpy releases the
GIL inside its loops):

* reductions over the member axis are per point and sequential in member
  order in numpy, so cutting the points into spans does not change one;
* maxima, minima, counts and logical reductions do not depend on order;
* a whole-array float sum is numpy's pairwise sum, which splits at fixed
  points (half the length, rounded down to a multiple of eight, above 128
  elements), so the top of its tree is walked here and the leaves, each a
  numpy sum of its own, run on the threads.  :func:`exact_sum` checks that
  identity against numpy once per process and uses numpy alone if it ever
  fails.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import os

import numpy as np

__all__ = ["exact_sum", "field_spread", "increment_stats", "support",
           "threads"]

#: numpy's pairwise-sum block: a run this long or shorter is summed with
#: eight accumulators, a longer one is split in two.
_PW_BLOCK = 128
#: Leaves at least this long are worth a thread.
_LEAF = 1 << 20


def threads() -> int:
    """Threads for one statistic: the usable cores, at most 64."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    return max(1, min(64, int(cores)))


def _spans(size: int, parts: int):
    parts = max(1, min(int(parts), int(size)))
    edges = np.linspace(0, size, parts + 1).astype(np.int64)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _pairwise_split(n: int) -> int:
    n2 = n // 2
    return n2 - n2 % 8


def _tree_sum(a, pool, leaf):
    """numpy's pairwise sum of contiguous 1-D ``a``, leaves on ``pool``."""
    leaves = []

    def plan(lo, hi):
        n = hi - lo
        if n <= max(leaf, _PW_BLOCK):
            leaves.append((lo, hi))
            return ("leaf", len(leaves) - 1)
        mid = lo + _pairwise_split(n)
        return ("add", plan(lo, mid), plan(mid, hi))

    tree = plan(0, int(a.size))
    sums = list(pool.map(lambda span: np.add.reduce(a[span[0]:span[1]]),
                         leaves))

    def walk(node):
        if node[0] == "leaf":
            return sums[node[1]]
        return walk(node[1]) + walk(node[2])

    return walk(tree)


_IDENTITY_CHECKED = []


def _identity_holds() -> bool:
    if not _IDENTITY_CHECKED:
        rng = np.random.default_rng(20261005)
        ok = True
        with ThreadPoolExecutor(max_workers=4) as pool:
            for n in (1000, 4097, 100003, 3 * _LEAF + 77):
                a = rng.standard_normal(n) * np.exp(rng.standard_normal(n) * 4)
                ok &= bool(_tree_sum(a, pool, 997) == np.add.reduce(a))
        _IDENTITY_CHECKED.append(ok)
    return _IDENTITY_CHECKED[0]


def exact_sum(a, *, workers: int | None = None):
    """``np.add.reduce(a)`` of a 1-D float64 array, the same bytes."""
    a = np.ascontiguousarray(a)
    if a.ndim != 1 or a.dtype != np.float64 or a.size <= 2 * _LEAF \
            or not _identity_holds():
        return np.add.reduce(a)
    workers = threads() if workers is None else int(workers)
    leaf = max(_LEAF, int(a.size) // (4 * workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return _tree_sum(a, pool, leaf)


def _flat(array):
    return np.asarray(array).reshape(-1)


def _nanmax(values):
    """``max`` as numpy's: NaN if any part is NaN."""
    values = [float(v) for v in values]
    if any(math.isnan(v) for v in values):
        return float("nan")
    return max(values)


def field_spread(members, *, workers: int | None = None):
    """``(max |x|, max std(ddof=1))`` over the member stack, as the driver's

    ``stack = np.stack(members).astype(np.float64)``,
    ``float(np.abs(stack).max())``, ``float(stack.std(axis=0, ddof=1).max())``
    computes them, without the stack: each thread stacks its own span of
    points.
    """
    flats = [_flat(m) for m in members]
    size = int(flats[0].size)
    workers = threads() if workers is None else int(workers)

    def one(span):
        lo, hi = span
        stack = np.stack([f[lo:hi] for f in flats]).astype(np.float64)
        return float(np.abs(stack).max()), float(
            stack.std(axis=0, ddof=1).max())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(one, _spans(size, 4 * workers)))
    return (_nanmax(p[0] for p in parts), _nanmax(p[1] for p in parts))


def increment_stats(members, *, workers: int | None = None) -> dict:
    """The driver's per-field increment record over ``np.stack(members)``:
    finite, max_abs, rms_where_nonzero and nonzero_fraction, the same
    bytes as the whole-stack numpy expressions."""
    flats = [_flat(m) for m in members]
    size = int(flats[0].size)
    total = size * len(flats)
    workers = threads() if workers is None else int(workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def scan(job):
            m, lo, hi = job
            x = flats[m][lo:hi]
            nz = x != 0.0
            return (bool(np.isfinite(x).all()), float(np.abs(x).max())
                    if x.size else 0.0, int(np.count_nonzero(nz)))

        jobs = [(m, lo, hi) for m in range(len(flats))
                for lo, hi in _spans(size, max(1, (4 * workers)
                                               // max(1, len(flats))))]
        scans = list(pool.map(scan, jobs))
        finite = all(s[0] for s in scans)
        max_abs = _nanmax(s[1] for s in scans)
        count = sum(s[2] for s in scans)
        rms = 0.0
        if count:
            # stack[stack != 0] ** 2 in the stack's order (member, then
            # point), filled span by span at its known offsets.
            squares = np.empty(count, dtype=np.result_type(
                *[f.dtype for f in flats]))
            offsets = np.concatenate(([0], np.cumsum([s[2] for s in scans])))

            def fill(index):
                m, lo, hi = jobs[index]
                x = flats[m][lo:hi]
                squares[offsets[index]:offsets[index + 1]] = x[x != 0.0] ** 2

            list(pool.map(fill, range(len(jobs))))
            if squares.dtype == np.float64:
                total_sq = exact_sum(squares, workers=workers)
            else:
                total_sq = np.add.reduce(squares)
            rms = float(np.sqrt(total_sq / count))
    return {"finite": finite, "max_abs": max_abs,
            "rms_where_nonzero": rms,
            "nonzero_fraction": float(count / total)}


def support(members, *, workers: int | None = None):
    """``np.any(np.stack(members) != 0.0, axis=0)`` without the stack."""
    flats = [_flat(m) for m in members]
    shape = np.shape(members[0])
    size = int(flats[0].size)
    out = np.zeros(size, dtype=bool)
    workers = threads() if workers is None else int(workers)

    def one(span):
        lo, hi = span
        acc = out[lo:hi]
        for f in flats:
            acc |= f[lo:hi] != 0.0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, _spans(size, 4 * workers)))
    return out.reshape(shape)
