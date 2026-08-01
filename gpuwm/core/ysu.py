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

_TPB = 32
_VALIDATE_TPB = 256
_KMAX = 128

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
    if nz > _KMAX:
        raise ValueError(f"nz={nz} exceeds YSU_KMAX={_KMAX}")
    if nz < 4:
        raise ValueError(f"YSU requires nz >= 4, got {nz}")
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
    blocks = (ncol + _TPB - 1) // _TPB
    kernel = get_kernel("ysu", "ysu_column")
    kernel((blocks,), (_TPB,),
           (u, v, theta, qv, qc, qi, p, p_interface, exner, dz, rthraten,
            psfc, znt, ust, hfx, qfx, wspd, br, psim, psih, xland, u10, v10,
            out["du"], out["dv"], out["dtheta"], out["dqv"], out["dqc"],
            out["dqi"], out["hpbl"], out["kpbl"], out["exch_h"],
            out["exch_m"], out["wstar"], out["delta"], DTYPE(dt),
            out["topdown_radsum"], out["wstar3_2"], out["cloudflg"],
            np.int32(int(ysu_topdown_pblmix)),
            np.int32(nz), np.int32(ny), np.int32(nx)))
    return out


def validate_ysu_outputs(
        out: Mapping[str, cp.ndarray], status: cp.ndarray) -> str | None:
    """Return the first non-finite native YSU output name, or ``None``.

    ``launch_ysu`` owns this exact output layout.  One validation kernel
    records a bit per floating-point output and one scalar readback preserves
    the driver's historical first-invalid error ordering.  The two integer
    outputs are necessarily finite and need no device work.
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
    invalid = int(status[0].item())
    for bit, name in enumerate(_YSU_OUTPUTS):
        if invalid & (1 << bit):
            return name
    return None
