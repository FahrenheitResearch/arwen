"""Device launchers for the MYNN mixscalars arms (W4 full admission, GPU).

Launches the two kernels of the NEW translation unit
``gpuwm/core/kernels/mynn_scalar_mix.cu`` — the frozen ``mynn_pbl.cu`` is
never touched.  CPU reference of record: :mod:`gpuwm.core.mynn_scalar_mix`
(bit-exact vs the anchored ``w4-oracle-fixtures`` family); the
probe harness ``tools/mynn_pbl_wrf461_oracle/probe_mynn_scalar_mix_gpu.py``
gates these launchers against it column-for-column over every fixture case.

One launch handles ONE species over a column batch, so the same two entry
points serve qni/qnc/qnwfa/qnifa/qnbca, and a later wave can add further
banks (the no-floor mirror identity).
"""

from __future__ import annotations

import numpy as np

from gpuwm.core.kernels import get_kernel

DTYPE = np.float32
_TPB = 128

#: Must match SMX_SOLVE_SCRATCH_FLOATS in kernels/mynn_scalar_mix.cu.
def _solve_scratch_floats(nz: int) -> int:
    return 7 * nz + 1


def _device_2d(cp, value, shape, name):
    array = cp.ascontiguousarray(cp.asarray(value, dtype=DTYPE))
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def mynn_mix_scalar_columns_cuda(
    qn, dz, rho, dfh, s_aw, s_awqn, delt, *, scratch=None,
):
    """One stock qn tridiagonal solve + tendency for a column batch.

    Device twin of :func:`gpuwm.core.mynn_scalar_mix.mix_scalar_column`;
    the ``dtz``/``rhoinv``/``khdz``/``hdz``/``dzinv`` arrays the CPU
    reference consumes are rebuilt in-kernel op-for-op from the primitive
    driver arrays (``module_bl_mynn.F:4137-4171`` construction including
    the ``:4163-4169`` stability floors).  Returns ``(qn2, dqn)`` on
    device, both ``(ncol, nz)``.
    """

    import cupy as cp

    qn = cp.ascontiguousarray(cp.asarray(qn, dtype=DTYPE))
    if qn.ndim != 2:
        raise ValueError("qn must have shape (ncol, nz)")
    ncol, nz = qn.shape
    if nz < 4:
        raise ValueError("MYNN mixscalars solve requires nz >= 4")
    dz = _device_2d(cp, dz, (ncol, nz), "dz")
    rho = _device_2d(cp, rho, (ncol, nz), "rho")
    dfh = _device_2d(cp, dfh, (ncol, nz), "dfh")
    s_aw = _device_2d(cp, s_aw, (ncol, nz + 1), "s_aw")
    s_awqn = _device_2d(cp, s_awqn, (ncol, nz + 1), "s_awqn")
    delt = cp.ascontiguousarray(cp.asarray(delt, dtype=DTYPE))
    if delt.shape != (ncol,):
        raise ValueError(f"delt must have shape ({ncol},)")
    qn2 = cp.empty((ncol, nz), dtype=DTYPE)
    dqn = cp.empty((ncol, nz), dtype=DTYPE)
    if scratch is None:
        scratch = cp.empty((ncol, _solve_scratch_floats(nz)), dtype=DTYPE)
    blocks = (ncol + _TPB - 1) // _TPB
    kernel = get_kernel("mynn_scalar_mix", "mynn_mix_scalar_columns")
    kernel(
        (blocks,), (_TPB,),
        (qn, dz, rho, dfh, s_aw, s_awqn, delt, qn2, dqn, scratch,
         np.int32(nz), np.int32(ncol)),
    )
    return qn2, dqn


def mynn_dmp_qn_flux_columns_cuda(
    qn, dz, zw, up_w, up_a, ent, rhoz, psig_w, plume_active,
    limiter_adjustment, *, scratch=None,
):
    """One species' ``s_awqn`` for a column batch.

    Device twin of :func:`gpuwm.core.mynn_scalar_mix.dmp_qn_flux_column`.
    ``up_w``/``up_a`` are ``(ncol, nz+1, nup)`` (``up_a`` PRE-limiter),
    ``ent`` is ``(ncol, nz, nup)``, ``rhoz`` is ``(ncol, nz)``;
    ``psig_w``/``limiter_adjustment`` are ``(ncol,)`` float32 and
    ``plume_active`` is ``(ncol,)`` int32 (WRF's ``NUP2 > 0`` gate).
    Returns ``s_awqn`` on device, ``(ncol, nz+1)``.
    """

    import cupy as cp

    qn = cp.ascontiguousarray(cp.asarray(qn, dtype=DTYPE))
    if qn.ndim != 2:
        raise ValueError("qn must have shape (ncol, nz)")
    ncol, nz = qn.shape
    up_w = cp.ascontiguousarray(cp.asarray(up_w, dtype=DTYPE))
    if up_w.ndim != 3 or up_w.shape[:2] != (ncol, nz + 1):
        raise ValueError("up_w must have shape (ncol, nz+1, nup)")
    nup = up_w.shape[2]
    dz = _device_2d(cp, dz, (ncol, nz), "dz")
    zw = _device_2d(cp, zw, (ncol, nz + 1), "zw")
    up_a = cp.ascontiguousarray(cp.asarray(up_a, dtype=DTYPE))
    if up_a.shape != (ncol, nz + 1, nup):
        raise ValueError(f"up_a must have shape ({ncol},{nz + 1},{nup})")
    ent = cp.ascontiguousarray(cp.asarray(ent, dtype=DTYPE))
    if ent.shape != (ncol, nz, nup):
        raise ValueError(f"ent must have shape ({ncol},{nz},{nup})")
    rhoz = _device_2d(cp, rhoz, (ncol, nz), "rhoz")
    psig_w = cp.ascontiguousarray(cp.asarray(psig_w, dtype=DTYPE))
    limiter_adjustment = cp.ascontiguousarray(
        cp.asarray(limiter_adjustment, dtype=DTYPE))
    plume_active = cp.ascontiguousarray(
        cp.asarray(plume_active, dtype=np.int32))
    for name, array in (("psig_w", psig_w), ("plume_active", plume_active),
                        ("limiter_adjustment", limiter_adjustment)):
        if array.shape != (ncol,):
            raise ValueError(f"{name} must have shape ({ncol},)")
    s_awqn = cp.empty((ncol, nz + 1), dtype=DTYPE)
    if scratch is None:
        scratch = cp.empty((ncol, (nz + 1) * nup), dtype=DTYPE)
    blocks = (ncol + _TPB - 1) // _TPB
    kernel = get_kernel("mynn_scalar_mix", "mynn_dmp_qn_flux_columns")
    kernel(
        (blocks,), (_TPB,),
        (qn, dz, zw, up_w, up_a, ent, rhoz, psig_w, plume_active,
         limiter_adjustment, s_awqn, scratch,
         np.int32(nz), np.int32(nup), np.int32(ncol)),
    )
    return s_awqn


__all__ = ["mynn_mix_scalar_columns_cuda", "mynn_dmp_qn_flux_columns_cuda"]
