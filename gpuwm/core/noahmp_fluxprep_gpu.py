"""CUDA wrappers for the pinned WRF v4.6.1 Noah-MP flux-preparation leaves.

One thread per column/case.  The device layout is the flat slot layout the
oracle harness packs, so ``gpuwm/data/noahmp/oracle/noahmp-fluxprep.csv`` is
replayed slot for slot with no repacking on either side.

These wrappers are validation surfaces for the leaf ports, not a runtime path:
Noah-MP is not dispatchable and ``sf_surface_physics=4`` stays blocked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gpuwm.core.kernels import get_kernel


@dataclass(frozen=True)
class FluxprepLayout:
    """Flat slot counts for one leaf, matching run_fluxprep.F90."""

    n_int: int
    n_in: int
    n_out: int


FLUXPREP_LAYOUTS: dict[str, FluxprepLayout] = {
    "ragrb": FluxprepLayout(n_int=4, n_in=17, n_out=6),
    "sfcdif1": FluxprepLayout(n_int=4, n_in=16, n_out=10),
    "stomata": FluxprepLayout(n_int=3, n_in=25, n_out=2),
}

_THREADS = 64


def evaluate_fluxprep_leaf(leaf: str, x, ix=None):
    """Run one flux-preparation leaf kernel over a batch of columns.

    ``x`` is ``(ncase, n_in)`` FP32 and ``ix`` is ``(ncase, n_int)`` int32.
    Returns a device array of shape ``(ncase, n_out)``, FP32.
    """
    import cupy as cp

    try:
        layout = FLUXPREP_LAYOUTS[leaf]
    except KeyError:
        raise KeyError(
            f"unknown Noah-MP flux-preparation leaf {leaf!r}; ported leaves "
            f"are {sorted(FLUXPREP_LAYOUTS)}") from None

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

    kernel = get_kernel("noahmp_fluxprep", f"noahmp_fluxprep_{leaf}")
    blocks = (ncase + _THREADS - 1) // _THREADS
    kernel((blocks,), (_THREADS,),
           (device_x, device_ix, device_y, np.int32(ncase)))
    return device_y
