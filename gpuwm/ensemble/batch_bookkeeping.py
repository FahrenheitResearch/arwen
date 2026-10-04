"""Single-launch RK word operations over explicitly owned member slabs."""
from __future__ import annotations

from operator import index

import numpy as np

from gpuwm.core.kernels import get_kernel


def prepare_bookkeeping(pairs, *, members, zero=False):
    """Bind arrays shaped (member, ...) without mixing or aliasing members.

    N=1 uses the current scalar RK kernel and its unchanged argument table.
    The closure retains its arrays, because descriptor pointers must stay live
    until all queued launches that consume them have finished.
    """
    import cupy as cp

    if isinstance(members, (bool, np.bool_)):
        raise TypeError("members must be an integer")
    members = index(members)
    if members < 1 or members > 65535:
        raise ValueError("bookkeeping needs 1..65535 members for the CUDA z grid")
    pairs = tuple(tuple(pair) for pair in pairs)
    if not pairs or len(pairs) > 64:
        raise ValueError("bookkeeping requires 1..64 independent array rows")
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("bookkeeping rows are (source, destination) pairs")
        for array in pair:
            if (not isinstance(array, cp.ndarray) or array.dtype != np.dtype("float32")
                    or not array.flags.c_contiguous or array.ndim < 2
                    or array.shape[0] != members or array.size == 0):
                raise ValueError("RK batch arrays must be nonempty contiguous float32 member slabs")
        if pair[0].shape != pair[1].shape:
            raise ValueError("RK batch source and destination shapes must match")
    for row, (src, dst) in enumerate(pairs):
        lo, hi = dst.data.ptr, dst.data.ptr + dst.nbytes
        for other_row, pair in enumerate(pairs):
            for col, other in enumerate(pair):
                if other_row == row and (col == 1 or (zero and col == 0)):
                    continue
                if lo < other.data.ptr + other.nbytes and other.data.ptr < hi:
                    raise ValueError("RK batch rows overlap; unordered writes would change member bytes")
    words = [dst.size // members for _, dst in pairs]
    width = 3 if members == 1 else 5
    rows = np.zeros((64, width), np.uint64)
    for row, ((src, dst), count) in enumerate(zip(pairs, words)):
        rows[row, :3] = (src.data.ptr, dst.data.ptr, count)
        if members != 1:
            rows[row, 3:] = (count, count)
    table = rows.reshape(-1).view(np.dtype(("V", rows.nbytes)))[0]
    if members == 1:
        kernel = get_kernel("rk_bookkeeping", "rk_zero_words" if zero else "rk_copy_words")
    else:
        kernel = get_kernel("ensemble_bookkeeping", "ensemble_zero_words" if zero else "ensemble_copy_words")
    grid = ((max(words) + 511) // 512, len(pairs), members)

    def launch():
        # Retain the owning arrays along with the pointer descriptor.
        _ = pairs
        kernel(grid, (128,), (table,))

    return launch
