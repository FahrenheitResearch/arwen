"""Native WRF cold-start vertical velocity initialization."""
from __future__ import annotations

import numpy as np


def initialize_wrfinput_vertical_velocity(
        state, restored, cfg, *, use_input_w=False,
        periodic_x=False, periodic_y=False):
    """Apply WRF v4.7.1 start_em.F:1475-1532 to a cold imported state.

    WRF diagnoses a near-zero input surface W regardless of use_input_w.
    Otherwise use_input_w preserves the file when true and diagnoses when
    false. The caller supplies the producing namelist's periodic controls.
    Both current and previous time levels receive the initialized column.
    """
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    for name, value in (("use_input_w", use_input_w),
                        ("periodic_x", periodic_x), ("periodic_y", periodic_y)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be the producing WRF logical value")
    nz, ny, nx = state.u.shape[0], state.w.shape[-2], state.w.shape[-1]
    if nz < 3 or state.w.shape != (nz + 1, ny, nx):
        raise ValueError("WRF cold-start W requires three wind levels and one extra W interface")
    if state.u.shape != (nz, ny, nx + 1) or state.v.shape != (nz, ny + 1, nx):
        raise ValueError("WRF cold-start W requires native staggered U and V dimensions")
    for name in ("u", "v", "w", "ht", "znw"):
        value = getattr(state, name)
        if value.dtype != np.dtype(np.float32) or not value.flags.c_contiguous:
            raise ValueError(f"WRF cold-start W needs contiguous stored FP32 {name} for the kernel layout")
    surface = state.w[0]
    threshold = float(np.float32(1.e-6))
    zero_surface = (abs(float(cp.max(surface).item())) < threshold
                    and abs(float(cp.min(surface).item())) < threshold)
    if use_input_w and not zero_surface:
        return False

    raw = restored.raw
    # Directional factors retain their producing WRF values; they are not
    # combined into the runtime's single isotropic map-factor carrier.
    mx_source = raw["MAPFAC_MX"] if "MAPFAC_MX" in raw else raw["MAPFAC_M"]
    my_source = raw["MAPFAC_MY"] if "MAPFAC_MY" in raw else raw["MAPFAC_M"]
    mx = cp.ascontiguousarray(cp.asarray(mx_source, dtype=cp.float32))
    my = cp.ascontiguousarray(cp.asarray(my_source, dtype=cp.float32))
    if mx.shape != (ny, nx) or my.shape != (ny, nx):
        raise ValueError("WRF cold-start W map factors must match the native mass grid")
    rdx = np.float32(np.float32(1.0) / np.float32(cfg.dx))
    rdy = np.float32(np.float32(1.0) / np.float32(cfg.dy))
    kernel = get_kernel("wrf_cold_start_w", "wrf_cold_start_w")
    kernel(((ny * nx + 127) // 128,), (128,),
           (state.u, state.v, state.ht, mx, my, state.znw, state.w,
            np.float32(state.cf1), np.float32(state.cf2), np.float32(state.cf3),
            rdx, rdy, np.int32(periodic_x), np.int32(periodic_y),
            np.int32(nz), np.int32(ny), np.int32(nx)))
    if getattr(state, "w0", None) is not None:
        state.w0[...] = state.w
    return True
