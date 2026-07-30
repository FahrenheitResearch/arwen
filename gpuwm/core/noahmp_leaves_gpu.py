"""CUDA wrappers for pinned WRF v4.6.1 Noah-MP leaf routines.

One thread per column/case.  The device layout is the flat slot layout the
oracle harness packs, so ``gpuwm/data/noahmp/oracle/noahmp-leaves.csv`` is
replayed slot for slot with no repacking on either side.

These wrappers are validation surfaces for the leaf ports, not a runtime path:
Noah-MP is not dispatchable and ``sf_surface_physics=4`` stays blocked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gpuwm.core.kernels import get_kernel


@dataclass(frozen=True)
class LeafLayout:
    """Flat slot counts for one leaf, matching run_leaves.F90."""

    n_int: int
    n_in: int
    n_out: int


LEAF_LAYOUTS: dict[str, LeafLayout] = {
    "atm": LeafLayout(n_int=0, n_in=11, n_out=17),
    "esat": LeafLayout(n_int=0, n_in=1, n_out=4),
    "rosr12": LeafLayout(n_int=1, n_in=28, n_out=21),
    "csnow": LeafLayout(n_int=1, n_in=9, n_out=15),
    "tdfcnd": LeafLayout(n_int=0, n_in=4, n_out=1),
    "thermoprop": LeafLayout(n_int=4, n_in=44, n_out=30),
    "wdfcnd1": LeafLayout(n_int=0, n_in=6, n_out=2),
    "wdfcnd2": LeafLayout(n_int=0, n_in=6, n_out=2),
}

_THREADS = 64


def evaluate_leaf(leaf: str, x, ix=None):
    """Run one Noah-MP leaf kernel over a batch of columns.

    ``x`` is ``(ncase, n_in)`` FP32 and ``ix`` is ``(ncase, n_int)`` int32.
    Returns a device array of shape ``(ncase, n_out)``, FP32.
    """
    import cupy as cp

    try:
        layout = LEAF_LAYOUTS[leaf]
    except KeyError:
        raise KeyError(
            f"unknown Noah-MP leaf {leaf!r}; ported leaves are "
            f"{sorted(LEAF_LAYOUTS)}") from None

    host_x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if host_x.ndim != 2 or host_x.shape[1] != layout.n_in:
        raise ValueError(
            f"{leaf}: expected x with shape (ncase, {layout.n_in}), "
            f"got {host_x.shape}")
    ncase = host_x.shape[0]

    if layout.n_int == 0:
        host_ix = np.zeros((ncase, 1), dtype=np.int32)
    else:
        if ix is None:
            raise ValueError(f"{leaf}: needs an integer topology vector")
        host_ix = np.ascontiguousarray(np.asarray(ix, dtype=np.int32))
        if host_ix.shape != (ncase, layout.n_int):
            raise ValueError(
                f"{leaf}: expected ix with shape ({ncase}, {layout.n_int}), "
                f"got {host_ix.shape}")

    device_x = cp.asarray(host_x)
    device_ix = cp.asarray(host_ix)
    device_y = cp.zeros((ncase, layout.n_out), dtype=cp.float32)

    kernel = get_kernel("noahmp_leaves", f"noahmp_leaf_{leaf}")
    blocks = (ncase + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,),
           (device_x, device_ix, device_y, np.int32(ncase)))
    return device_y
