"""WSM6 single-moment six-class microphysics (WRF ``mp_physics=6``).

The CUDA kernel owns one complete column and updates qv/qc/qr/qi/qs/qg,
potential temperature, surface rain/snow/graupel, frozen fraction, and the
scheme-native radiation effective radii.  WRF v4.6.1
``phys/physics_mmm/mp_wsm6.F90`` and ``mp_wsm6_effectRad.F90`` are the
primary numerical authorities.

The local process equations and WRF's forward semi-Lagrangian monotone-PLM
sedimentation are ported and checked against direct calls into the v4.6.1
Fortran.  This remains an executable integration port, not a whole-model or
bitwise WRF-parity claim; device and coupled-trajectory evidence are separate
release gates.
"""

from __future__ import annotations

import numpy as np
import cupy as cp

from gpuwm.config import RunConfig
from gpuwm.core import constants as c
from gpuwm.core.kernels import get_kernel, get_kernel_int_defines
from gpuwm.core.state import DTYPE, DomainState

_COLUMN_TPB = 32
_SHALLOW_KMAX = 64
_KMAX = 80
VERTICAL_LEVEL_BOUNDS = (2, _KMAX)


def _kernel_capacity(nz: int) -> int:
    """Return the smallest compiled WSM6 local-array tier for ``nz``."""
    minimum, maximum = VERTICAL_LEVEL_BOUNDS
    if not minimum <= nz <= maximum:
        raise ValueError(
            f"WSM6 requires {minimum} <= nz <= {maximum}, got {nz}")
    return _SHALLOW_KMAX if nz <= _SHALLOW_KMAX else _KMAX


def launch_wsm6(theta, qv, qc, qr, qi, qs, qg, rho, pii, pressure, dz,
                rainnc, rainncv, snownc, snowncv,
                graupelnc, graupelncv, sr, dt: float, *,
                effc, effi, effs, hail_opt: int = 0) -> None:
    """Launch one WSM6 call over contiguous FP32 ``(nz,ny,nx)`` fields."""
    shape = theta.shape
    if len(shape) != 3:
        raise ValueError(f"WSM6 fields must be 3-D, got {shape}")
    nz, ny, nx = shape
    capacity = _kernel_capacity(nz)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    if int(hail_opt) not in (0, 1):
        raise ValueError(f"WSM6 hail_opt must be 0 or 1, got {hail_opt}")
    volume = {
        "theta": theta, "qv": qv, "qc": qc, "qr": qr,
        "qi": qi, "qs": qs, "qg": qg, "rho": rho, "pii": pii,
        "pressure": pressure, "dz": dz,
        "effc": effc, "effi": effi, "effs": effs,
    }
    for name, value in volume.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    surface_shape = (ny, nx)
    for name, value in {
            "rainnc": rainnc, "rainncv": rainncv,
            "snownc": snownc, "snowncv": snowncv,
            "graupelnc": graupelnc, "graupelncv": graupelncv,
            "sr": sr}.items():
        if value.shape != surface_shape or value.dtype != DTYPE:
            raise ValueError(f"{name} must be float32 {surface_shape}, got "
                             f"{value.dtype} {value.shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    ncol = ny * nx
    blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    kernel = (get_kernel("wsm6", "wsm6_column")
              if capacity == _SHALLOW_KMAX else
              get_kernel_int_defines(
                  "wsm6", "wsm6_column", (("WSM6_KMAX", capacity),)))
    kernel(
        (blocks,), (_COLUMN_TPB,),
        (theta, qv, qc, qi, qr, qs, qg, rho, pressure, pii, dz,
         rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr,
         effc, effi, effs, DTYPE(dt), np.int32(hail_opt),
         np.int32(nz), np.int32(ny), np.int32(nx)))


def apply(state: DomainState, cfg: RunConfig, dt: float, *,
          refl_10cm_due: bool = False):
    """Prepare WRF fields, run WSM6, and finish retained MP heating."""
    nz, ny, nx = state.p.shape
    required = ("qi", "qs", "qg", "effc", "effi", "effs")
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=6 state lacks WSM6 fields: "
                         + ", ".join(missing))
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    phb = state.phb if state.phb.ndim == 3 else state.phb[:, None, None]
    theta = state.scratch((nz, ny, nx), "wsm6_theta")
    rho = state.scratch((nz, ny, nx), "wsm6_rho")
    pii = state.scratch((nz, ny, nx), "wsm6_pii")
    dz = state.scratch((nz, ny, nx), "wsm6_dz")
    z8w = state.scratch((nz + 1, ny, nx), "wsm6_z8w")
    theta[...] = thb + state.thp
    rho[...] = 1.0 / state.alt
    pii[...] = cp.power(state.p / DTYPE(c.P0), DTYPE(c.RCP))
    z8w[...] = (phb + state.php) / DTYPE(c.G)
    dz[...] = z8w[1:] - z8w[:-1]

    surface = (ny, nx)
    rainnc = state.scratch(surface, "mp_rainnc")
    rainncv = state.scratch(surface, "mp_rainncv")
    snownc = state.scratch(surface, "mp_snownc")
    snowncv = state.scratch(surface, "mp_snowncv")
    graupelnc = state.scratch(surface, "mp_graupelnc")
    graupelncv = state.scratch(surface, "mp_graupelncv")
    sr = state.scratch(surface, "mp_sr")

    from gpuwm.core.microphysics import (MicrophysicsDiagnostics,
                                         moist_physics_finish,
                                         save_pre_mp_theta)
    save_pre_mp_theta(state)
    launch_wsm6(
        theta, state.qv, state.qc, state.qr, state.qi, state.qs, state.qg,
        rho, pii, state.p, dz, rainnc, rainncv, snownc, snowncv,
        graupelnc, graupelncv, sr, dt, effc=state.effc,
        effi=state.effi, effs=state.effs, hail_opt=cfg.wsm6_hail_opt)
    if refl_10cm_due:
        from gpuwm.core.refl import compute_and_stash_refl_10cm
        refl_t = state.scratch((nz, ny, nx), "refl_t")
        refl_t[...] = theta * pii
        compute_and_stash_refl_10cm(state, cfg, refl_t, state.p)
    moist_physics_finish(state, cfg, theta, dt)
    return MicrophysicsDiagnostics(
        rainnc=rainnc, rainncv=rainncv, sr=sr,
        snownc=snownc, snowncv=snowncv,
        graupelnc=graupelnc, graupelncv=graupelncv)


__all__ = ["apply", "launch_wsm6"]
