"""Replay WRF vertical/TKE fixtures through the production kernel loader."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
from gpuwm.verify.diffusion_oracle import device_state


def _config(metadata, **changes):
    from gpuwm.config import RunConfig
    nz, ny, nx = (metadata[k] for k in ("nz", "ny", "nx"))
    values = dict(nx=nx,ny=ny,nz=nz,dx=metadata["dx"],dy=metadata["dy"],
                  dt=12.,ztop=20000.,run_seconds=0.,km_opt=2,bl_pbl_physics=0,
                  open_x=bool(metadata["bx"]),open_y=bool(metadata["by"]),
                  isfflx=0,tke_drag_coefficient=0.,tke_heat_flux=0.,
                  sf_sfclay_physics=0,tke_budget=1)
    values.update(changes)
    return RunConfig(**values)


def vertical_gpu(arrays,metadata,*,kmv,khv,var,doing_tke=False,full_theta=False):
    """Actual vertical launcher and its production open-strip staging."""
    import cupy as cp
    from gpuwm.core.dycore import launch_wrf_smag2d_vertical, _zero_open_strips
    cfg = _config(metadata)
    state = device_state(arrays,cf=tuple(metadata[k] for k in ("cf1","cf2","cf3")))
    tu,tv,tw = (cp.zeros_like(getattr(state,k)) for k in ("u","v","w"))
    ts,rth = cp.zeros_like(state.alt),cp.zeros_like(state.alt)
    k = cp.asarray(kmv,dtype=cp.float32)
    scalar = cp.asarray(var,dtype=cp.float32)
    launch_wrf_smag2d_vertical(state,cfg,k,ru=tu,rv=tv,rw=tw,rth=rth,
                             rqv=None,time_t=False,kmv=k,
                             khv=cp.asarray(khv,dtype=cp.float32),
                             scalar_rows=[(scalar,ts,full_theta)])
    if doing_tke:
        ts *= np.float32(2.)
    # _compute_wrf_smag_tendencies applies exactly this engine staging after
    # the vertical package. Retain and measure every excluded boundary word.
    for a in (tu,tv,tw,ts):
        _zero_open_strips(a,cfg,1)
    return {name:cp.asnumpy(a) for name,a in
            zip(("vertical_u","vertical_v","vertical_w","vertical_s"),(tu,tv,tw,ts))}


def tke_gpu(arrays,metadata,deformation,coefficients,bn2,*,isfflx=0,isotropic=0,
            c_k=.15,dt=12.,cd0=.0013,heat=.24):
    """The production RHS kernel with identical explicit coefficient inputs.

    The kernel computes D13/D23 and vertical geometry using the same inline
    functions the forecast uses. The Fortran tensors remain independently
    available in the fixture, so any upstream discrepancy is observable.
    """
    import cupy as cp
    from gpuwm.core.dycore import _wrf_smag_grid_args
    from gpuwm.core.kernels import get_kernel
    from gpuwm.core.state import DTYPE
    cfg = _config(metadata,isfflx=isfflx,mix_isotropic=isotropic,c_k=c_k,dt=dt,
                  tke_drag_coefficient=cd0,tke_heat_flux=heat)
    state = device_state(arrays,cf=tuple(metadata[k] for k in ("cf1","cf2","cf3")))
    nz,ny,nx = state.alt.shape
    common = _wrf_smag_grid_args(state,cfg,time_t=False)
    fp = lambda x:cp.asarray(x,dtype=cp.float32)
    inputs = [state.thp,state.thb,np.int32(state.thb.ndim==3),state.tke,fp(bn2)]
    inputs += [fp(deformation[k]) for k in ("d11","d22","d12")]
    inputs += [fp(coefficients[k]) for k in ("kmh","kmv","khv")]
    inputs += [state.mut,state.c1h,state.c2h,state.ust,state.hfx,
               np.int32(1),np.int32(1),DTYPE(c_k),DTYPE(dt),DTYPE(cd0),
               DTYPE(heat),np.int32(isfflx)]
    rhs = cp.zeros_like(state.alt)
    terms = [cp.zeros_like(rhs) for _ in range(4)]
    dims = [np.int32(nz),np.int32(ny),np.int32(nx),np.int32(state.phb.ndim==3),
            np.int32(metadata["bx"]),np.int32(metadata["by"])]
    get_kernel("smag2d","wrf_tke_rhs")(((nx+127)//128,ny,nz),(128,1,1),
        tuple(common+inputs+[rhs]+terms+[np.int32(1)]+dims))
    shear,buoy,diss,lim = terms
    # Export cumulative states matching the three actual WRF calls. Budget
    # subtraction/addition is recorded separately from the final RHS words.
    return {"tke_shear":cp.asnumpy(shear),
            "tke_buoyancy":cp.asnumpy(shear+buoy),
            "tke_dissip":cp.asnumpy(shear+buoy+diss),
            "tke_rhs":cp.asnumpy(rhs),
            "budget_shear":cp.asnumpy(shear),"budget_buoyancy":cp.asnumpy(buoy),
            "budget_dissipation":cp.asnumpy(diss),"budget_limiter":cp.asnumpy(lim)}


def vertical_driver_gpu(arrays,metadata,coefficients,*,km_opt=2,isfflx=0,cd0=.0013,heat=.24,initial_tke=0.):
    """Replay the actual vertical launcher, with its active surface fields."""
    import cupy as cp
    from gpuwm.core.dycore import launch_wrf_smag2d_vertical,_wrf_smag_grid_args,_zero_open_strips
    from gpuwm.core.kernels import get_kernel
    cfg = _config(metadata,km_opt=km_opt,isfflx=isfflx,sf_sfclay_physics=1,
                  tke_drag_coefficient=cd0,tke_heat_flux=heat)
    state = device_state(arrays,cf=tuple(metadata[k] for k in ("cf1","cf2","cf3")))
    state.physics = SimpleNamespace(fields={"ustm":state.ust,"hfx":state.hfx,"qfx":state.qfx})
    outputs = {k:cp.zeros_like(getattr(state,k)) for k in ("u","v","w","thp","tke","qv","qc","qi")}
    outputs["tke"].fill(initial_tke)
    coef = {k:cp.asarray(v,dtype=cp.float32) for k,v in coefficients.items()}
    launch_wrf_smag2d_vertical(state,cfg,coef["kmh"],ru=outputs["u"],rv=outputs["v"],rw=outputs["w"],
        rth=outputs["thp"],rqv=outputs["qv"],time_t=False,kmv=coef["kmv"],khv=coef["khv"],
        scalar_rows=[(state.thp,outputs["thp"],True),(state.qv,outputs["qv"],False),
                     (state.qc,outputs["qc"],False),(state.qi,outputs["qi"],False)])
    if km_opt==2:
        nz,ny,nx = state.alt.shape
        args = _wrf_smag_grid_args(state,cfg,time_t=False)
        args += [state.tke,state.thb,np.int32(0),np.int32(state.thb.ndim==3),coef["kmv"],outputs["tke"]]
        args += [np.int32(nz),np.int32(ny),np.int32(nx),np.int32(state.phb.ndim==3),
                 np.int32(metadata["bx"]),np.int32(metadata["by"])]
        get_kernel("smag2d","wrf_smag_vd_s")(((nx+127)//128,ny,nz),(128,1,1),tuple(args))
        _zero_open_strips(outputs["tke"],cfg,1)
        outputs["tke"] *= np.float32(2.)
    result = {"theta" if k=="thp" else k:cp.asnumpy(v) for k,v in outputs.items()}
    result.update({k+"_after":cp.asnumpy(getattr(state,k)) for k in ("thp","qv","qc","qi","hfx","qfx")})
    return result
