"""YSU planetary-boundary-layer column kernel.

This module is the device launcher for the WRF v4.6.1 YSU transcription in
``kernels/ysu.cu``.  The scheme consumes mass-point wind, potential
temperature, moisture, pressure/Exner, layer depths, and the surface-layer
coupling fields (UST, HFX/QFX, PSIM/PSIH).  It returns tendencies rather than
updating the state; Phase 3 Task 12's physics driver owns their application.

Array layout is gpuwm's standard ``(nz, ny, nx)`` with x fastest.  Every CUDA
thread owns one complete column and performs the implicit vertical solves
in-thread, matching the acoustic Thomas-solve pattern.  The float64 authority
is :func:`gpuwm.verify.npref.np_ysu_column`.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import cupy as cp

from gpuwm.core.kernels import get_kernel
from gpuwm.core.state import DTYPE
from gpuwm.physics_vertical_contract import (
    outside_vertical_bounds, refuse_vertical_levels)

_TPB = 32
_VALIDATE_TPB = 256
_KMAX = 128

# ---------------------------------------------------------------------------
# The per-thread column workspace
# ---------------------------------------------------------------------------
# ysu.cu used to keep the scheme's column arrays in the per-thread local
# frame, and CUDA prices a local frame at the card's RESIDENT-THREAD
# CAPACITY -- one per-context backing store of
# ``(frame - 1024) * SMs * maxThreadsPerSM``, taken at the first LAUNCH of
# the kernel and never returned.  MEASURED on node-1 (weather-node-1, RTX
# 5070 Ti, 70 SMs x 1,536, sm_120, CuPy 14.0.1) through this launcher at
# nz=49: the 9,232 B frame took 844.0 MiB while the kernel only ever had a
# fraction of that many threads in flight.  ``bl_pbl_physics = 1`` is the
# shipped default, so every default run paid it.
#
# The arrays now live in a global workspace this launcher allocates, sized
# to the threads ACTUALLY in flight, and the columns are launched in tiles
# of that size.
#
# These must match ysu.cu's YSUWS_SLOTS / YSUWS_LANES;
# tests/test_ysu_workspace.py re-derives them from the .cu source and fails
# if either side moves alone.
YSUWS_SLOTS = 18

#: Launch block, and the tile's granularity.  ysu.cu indexes the workspace
#: by the thread's lane within its block, so a tile is always a whole
#: number of blocks.  This is the same number as :data:`_TPB`; both are
#: pinned against the kernel's ``YSUWS_LANES``.
YSU_BLOCK = _TPB

#: Blocks per SM the tile is sized for.  MEASURED, not assumed -- see
#: docs/kernel_local_memory_bounds.md for the sweep this came from.  The
#: query in :func:`ysu_tile_columns` only ever lowers it.
YSU_TILE_BLOCKS_PER_SM = 16


def ysu_workspace_floats(nz: int, columns: int) -> int:
    """Workspace floats for ``columns`` columns in flight at this ``nz``.

    Rounded up to whole blocks: ysu.cu interleaves the workspace by LANE
    within a block, the way CUDA lays local memory out across a warp, so
    the unit of allocation is one block's region, not one column's.

    The per-slot extent is ``nz + 1`` rather than ``_KMAX``: ``zq`` is the
    one array indexed at ``nz``, and unlike the compile-time frame this is
    allocated when ``nz`` is known.  A 49-level run therefore holds 50
    levels of arrays where the frame had to hold 128.
    """
    blocks = (int(columns) + YSU_BLOCK - 1) // YSU_BLOCK
    return blocks * YSUWS_SLOTS * (int(nz) + 1) * YSU_BLOCK


def ysu_tile_columns(fn, ncol: int) -> int:
    """Columns to keep in flight: enough to fill the card, and no more.

    The whole point of the workspace is that it is charged per thread IN
    FLIGHT where the local-memory backing store was charged per thread the
    card could ever hold.  Over-sizing the tile hands that ratio back.
    """
    import cupy as cp

    dev = cp.cuda.Device()
    sms = dev.attributes["MultiProcessorCount"]
    per_sm = YSU_TILE_BLOCKS_PER_SM
    try:
        from cupy_backends.cuda.api import driver

        resident = driver.occupancyMaxActiveBlocksPerMultiprocessor(
            fn.kernel.ptr, YSU_BLOCK, 0)
        # Never launch more blocks than the card can hold resident: those
        # columns would wait while their workspace slots stayed allocated.
        per_sm = min(per_sm, int(resident))
    except Exception:                              # noqa: BLE001
        pass                                       # keep the measured value
    per_sm = max(1, int(per_sm))
    # Floored at one block.  The tile is the STEP of the launcher's loop, so
    # a zero here would be a zero-step ``range`` -- a ValueError instead of
    # the empty launch a zero-column domain should produce.
    return int(max(YSU_BLOCK, min(int(ncol), sms * per_sm * YSU_BLOCK)))

#: The launcher's first-call vertical bound, restated for front doors by
#: ``gpuwm.physics_vertical_contract.YSU_VERTICAL_LEVEL_BOUNDS`` and bound to
#: it by ``tests/test_pbl_vertical_bounds.py``.
VERTICAL_LEVEL_BOUNDS = (4, _KMAX)

_YSU_3D_FLOAT_OUTPUTS = (
    "du", "dv", "dtheta", "dqv", "dqc", "dqi", "exch_h", "exch_m",
)
_YSU_2D_OUTPUTS = (
    "hpbl", "kpbl", "wstar", "delta", "topdown_radsum", "wstar3_2",
    "cloudflg",
)
_YSU_OUTPUTS = _YSU_3D_FLOAT_OUTPUTS + _YSU_2D_OUTPUTS
_YSU_2D_FLOAT_OUTPUTS = tuple(
    name for name in _YSU_2D_OUTPUTS if name not in ("kpbl", "cloudflg"))


def launch_ysu(u, v, theta, qv, qc, qi, p, p_interface, exner, dz,
               rthraten=None, *,
               psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, xland,
               u10, v10, dt: float, ysu_topdown_pblmix: int = 1):
    """Launch YSU for a batch of device columns and return device outputs.

    Three-dimensional inputs have shape ``(nz, ny, nx)`` except
    ``p_interface`` which is ``(nz+1, ny, nx)``.  Surface inputs have shape
    ``(ny, nx)``.  All are C-contiguous float32 CuPy arrays.  The returned
    dict contains 3-D ``du/dv/dtheta/dqv/dqc/dqi/exch_h/exch_m``, 2-D
    ``hpbl/wstar/delta`` float32 arrays, and 2-D one-based ``kpbl`` int32.
    """
    columns = {"u": u, "v": v, "theta": theta, "qv": qv, "qc": qc,
               "qi": qi, "p": p, "exner": exner, "dz": dz}
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"theta must have shape (nz, ny, nx), got {shape}")
    nz, ny, nx = shape
    if outside_vertical_bounds(nz, VERTICAL_LEVEL_BOUNDS):
        raise refuse_vertical_levels(
            "YSU PBL", VERTICAL_LEVEL_BOUNDS, nz,
            breakage=(
                "one CUDA thread owns a whole column and holds it in "
                f"per-thread local memory at YSU_KMAX={_KMAX} "
                "(kernels/ysu.cu), and the counter-gradient and entrainment "
                "layers the scheme indexes need four levels to exist."))
    for name, arr in columns.items():
        if not isinstance(arr, cp.ndarray) or arr.shape != shape:
            raise ValueError(f"{name} must be a CuPy array with shape {shape}")
        if arr.dtype != DTYPE or not arr.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous float32")
    if rthraten is None:
        rthraten = cp.zeros_like(theta)
    elif (not isinstance(rthraten, cp.ndarray) or rthraten.shape != shape
          or rthraten.dtype != DTYPE or not rthraten.flags.c_contiguous):
        raise ValueError("rthraten must be C-contiguous float32 with shape "
                         f"{shape}")
    if (not isinstance(p_interface, cp.ndarray)
            or p_interface.shape != (nz + 1, ny, nx)
            or p_interface.dtype != DTYPE
            or not p_interface.flags.c_contiguous):
        raise ValueError("p_interface must be C-contiguous float32 with shape "
                         f"{(nz + 1, ny, nx)}")
    surfaces = {"psfc": psfc, "znt": znt, "ust": ust, "hfx": hfx,
                "qfx": qfx, "wspd": wspd, "br": br, "psim": psim,
                "psih": psih, "xland": xland, "u10": u10, "v10": v10}
    for name, arr in surfaces.items():
        if (not isinstance(arr, cp.ndarray) or arr.shape != (ny, nx)
                or arr.dtype != DTYPE or not arr.flags.c_contiguous):
            raise ValueError(f"{name} must be a C-contiguous float32 CuPy "
                             f"array with shape {(ny, nx)}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"YSU requires a positive finite dt, got {dt}")
    if ysu_topdown_pblmix not in (0, 1, False, True):
        raise ValueError("ysu_topdown_pblmix must be 0 or 1")

    out = {name: cp.empty_like(theta) for name in _YSU_3D_FLOAT_OUTPUTS}
    out.update(hpbl=cp.empty((ny, nx), dtype=DTYPE),
               kpbl=cp.empty((ny, nx), dtype=cp.int32),
               wstar=cp.empty((ny, nx), dtype=DTYPE),
               delta=cp.empty((ny, nx), dtype=DTYPE),
               topdown_radsum=cp.empty((ny, nx), dtype=DTYPE),
               wstar3_2=cp.empty((ny, nx), dtype=DTYPE),
               cloudflg=cp.empty((ny, nx), dtype=cp.int32))
    ncol = ny * nx
    kernel = get_kernel("ysu", "ysu_column")
    # The column arrays live in a global workspace sized to the threads in
    # flight, not in the per-thread local frame (which CUDA prices at the
    # card's whole resident-thread capacity).  Columns therefore go in
    # tiles of `tile`, each launched with the SAME one-thread-one-column
    # mapping the kernel has always had -- the tile is passed as a column
    # OFFSET rather than as a slice, because the (nz, ny, nx) fields make a
    # column tile non-contiguous, so the kernel indexes them exactly as
    # before.
    tile = ysu_tile_columns(kernel, ncol)
    wskp = nz + 1
    ws = cp.empty(ysu_workspace_floats(nz, tile), dtype=DTYPE)
    try:
        for lo in range(0, ncol, tile):
            count = min(tile, ncol - lo)
            blocks = (count + YSU_BLOCK - 1) // YSU_BLOCK
            kernel((blocks,), (YSU_BLOCK,),
                   (u, v, theta, qv, qc, qi, p, p_interface, exner, dz,
                    rthraten, psfc, znt, ust, hfx, qfx, wspd, br, psim,
                    psih, xland, u10, v10,
                    out["du"], out["dv"], out["dtheta"], out["dqv"],
                    out["dqc"], out["dqi"], out["hpbl"], out["kpbl"],
                    out["exch_h"], out["exch_m"], out["wstar"],
                    out["delta"], DTYPE(dt),
                    out["topdown_radsum"], out["wstar3_2"],
                    out["cloudflg"],
                    np.int32(int(ysu_topdown_pblmix)),
                    np.int32(nz), np.int32(ny), np.int32(nx),
                    ws, np.int32(wskp), np.int32(lo)))
    finally:
        del ws
    return out


def validate_ysu_outputs(
        out: Mapping[str, cp.ndarray], status: cp.ndarray,
        *, refuse=None) -> str | None:
    """Return the first non-finite native YSU output name, or ``None``.

    ``launch_ysu`` owns this exact output layout.  One validation kernel
    records a bit per floating-point output and one scalar readback preserves
    the driver's historical first-invalid error ordering.  The two integer
    outputs are necessarily finite and need no device work.

    ``refuse`` opts this site into :mod:`gpuwm.core.health_ledger`.  It is
    the caller's own refusal -- the closure that builds the forensic message
    -- and with a ledger active it is called at the drain with the first
    flagged name, preserving the historical first-invalid ordering because
    the bit order is the launcher order.  Without it the word is read here.
    """
    if tuple(out) != _YSU_OUTPUTS:
        raise ValueError(
            "native YSU validation requires outputs in launcher order "
            f"{_YSU_OUTPUTS}, got {tuple(out)}")
    if (not isinstance(status, cp.ndarray) or status.shape != (1,)
            or status.dtype != cp.uint32 or not status.flags.c_contiguous):
        raise ValueError(
            "YSU validation status must be a C-contiguous uint32 device "
            "array with shape (1,)")

    shape_3d = out["du"].shape
    shape_2d = out["hpbl"].shape
    if len(shape_3d) != 3 or len(shape_2d) != 2:
        raise ValueError("native YSU outputs must retain 3-D/2-D shapes")
    for name in _YSU_3D_FLOAT_OUTPUTS:
        value = out[name]
        if (not isinstance(value, cp.ndarray) or value.shape != shape_3d
                or value.dtype != DTYPE or not value.flags.c_contiguous):
            raise ValueError(
                f"native YSU output {name} must be C-contiguous float32 "
                f"with shape {shape_3d}")
    for name in _YSU_2D_FLOAT_OUTPUTS:
        value = out[name]
        if (not isinstance(value, cp.ndarray) or value.shape != shape_2d
                or value.dtype != DTYPE or not value.flags.c_contiguous):
            raise ValueError(
                f"native YSU output {name} must be C-contiguous float32 "
                f"with shape {shape_2d}")

    status.fill(cp.uint32(0))
    count_3d = out["du"].size
    count_2d = out["hpbl"].size
    blocks = (max(count_3d, count_2d) + _VALIDATE_TPB - 1) // _VALIDATE_TPB
    kernel = get_kernel("ysu_validation", "ysu_validate_outputs")
    kernel(
        (blocks,), (_VALIDATE_TPB,),
        tuple(out[name] for name in _YSU_3D_FLOAT_OUTPUTS)
        + tuple(out[name] for name in _YSU_2D_FLOAT_OUTPUTS)
        + (status, np.int64(count_3d), np.int64(count_2d)))
    from gpuwm.core import health_ledger

    def _first_name(flags: int) -> str | None:
        for bit, name in enumerate(_YSU_OUTPUTS):
            if flags & (1 << bit):
                return name
        return None

    invalid = health_ledger.read_status(
        status, site="ysu",
        describe=None if refuse is None
        else lambda flags: refuse(_first_name(flags)))
    return _first_name(invalid)
