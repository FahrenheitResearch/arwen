"""Purdue Lin six-class single-moment microphysics, WRF mp_physics=2.

The column kernel retains the WRF 4.6.1 default F_QNDROP=false process order,
adaptive sedimentation and Newton saturation adjustment. Strict arithmetic
uses the common WOOF compiler route and shared float32 math routines.
"""
from __future__ import annotations
import numpy as np
from gpuwm.core.kernels import get_kernel, get_kernel_int_defines


VERTICAL_LEVEL_BOUNDS = (2, 512)


def launch_lin(theta, qv, qc, qr, qi, qs, qg, rho, pii, pressure, z, dz, ht,
               rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr,
               dt: float, *, kernel=None):
    """Update contiguous FP32 columns and the seven surface diagnostics."""
    if len(theta.shape) != 3:
        raise ValueError(f"Lin fields must be 3-D, got {theta.shape}")
    nz, ny, nx = theta.shape
    if nz < 2 or nz > 512:
        raise ValueError(f"Lin needs 2..512 levels to retain complete sedimentation, got {nz}")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be finite and positive, got {dt}")
    volumes = (theta,qv,qc,qr,qi,qs,qg,rho,pii,pressure,z,dz)
    surfaces = (ht,rainnc,rainncv,snownc,snowncv,graupelnc,graupelncv,sr)
    for a in volumes:
        if a.shape != theta.shape or a.dtype != np.float32 or not a.flags.c_contiguous:
            raise ValueError("Lin volumes must have one contiguous float32 (nz,ny,nx) shape")
    for a in surfaces:
        if a.shape != (ny,nx) or a.dtype != np.float32 or not a.flags.c_contiguous:
            raise ValueError("Lin surfaces must be contiguous float32 (ny,nx)")
    capacity = next(tier for tier in (128,256,512) if tier >= nz)
    if kernel is None:
        kernel = (get_kernel("lin", "lin_column") if capacity == 128 else
                  get_kernel_int_defines("lin", "lin_column", (("LIN_KMAX", capacity),)))
    ncol = ny*nx
    kernel(((ncol+31)//32,), (32,),
           (*volumes, ht, rainnc, rainncv, snownc, snowncv, graupelnc,
            graupelncv, sr, np.float32(dt), np.int32(nz), np.int32(ncol)))


import cupy as cp
from gpuwm.config import RunConfig
from gpuwm.core.state import DTYPE, DomainState
from gpuwm.core.wsm_family import (finish_fields, precipitation_surface)

def apply(state: DomainState, cfg: RunConfig, dt: float, *,
          refl_10cm_due: bool = False):
    """Prepare WRF fields, run Lin, and finish retained MP heating."""
    nz, ny, nx = state.p.shape
    required = ("qi", "qs", "qg")
    missing = [name for name in required if getattr(state, name, None) is None]
    if missing:
        raise ValueError("mp_physics=2 state lacks Lin fields: "
                         + ", ".join(missing))
    from gpuwm.core.wsm6 import _wsm6_prepare
    theta, rho, pii, dz = _wsm6_prepare(state, "lin")

    z8w = state.existing_scratch("lin_z8w")
    z = state.scratch((nz, ny, nx), "lin_z")
    z[...] = DTYPE(0.5) * (z8w[1:] + z8w[:-1])
    surface = precipitation_surface(state)

    from gpuwm.core.microphysics import save_pre_mp_theta
    save_pre_mp_theta(state)
    launch_lin(
        theta, state.qv, state.qc, state.qr, state.qi, state.qs, state.qg,
        rho, pii, state.p, z, dz, state.ht, dt=dt, **surface)
    return finish_fields(state, cfg, theta, pii, dt, surface,
                         refl_10cm_due=refl_10cm_due)




def launch_refl10cm_lin(qv, qr, qs, qg, t, p, refl):
    """WRF fixed intercepts, including the native melting-particle diagnostic."""
    from gpuwm.core.refl import _device_tables, radar_init, _check_fields
    nz, ny, nx = refl.shape
    _check_fields(dict(qv=qv,qr=qr,qs=qs,qg=qg,t=t,p=p,refl=refl),refl.shape)
    if nz > 512:
        raise ValueError("Lin radar columns above 512 levels cannot retain the melting-level scan")
    rc=radar_init(0)
    kernel=get_kernel_int_defines("lin_refl", "refl10cm_lin_column", (("REFL_KMAX",nz),))
    kernel(((ny*nx+31)//32,),(32,),
           (qv,qr,qs,qg,t,p,_device_tables(),np.float64(rc.k_w),
            np.float64(rc.m_w_0.real),np.float64(rc.m_w_0.imag),
            np.float64(rc.m_i_0.real),np.float64(rc.m_i_0.imag),
            np.float64(209.4393310546875),np.float64(4.0e6),refl,
            np.int32(nz),np.int32(ny),np.int32(nx)))
