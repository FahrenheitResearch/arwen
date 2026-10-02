"""Synthetic production states used by the coordinate diffusion gates."""
from dataclasses import fields, replace
import numpy as np
from gpuwm.config import RunConfig


def configuration(km=4,diff=1,mix=False,boundary=False,moist=False):
    knobs=dict(nx=12,ny=9,nz=6,dx=800.,dy=1400.,ztop=3000.,dt=.5,
               run_seconds=30.,km_opt=km,bl_pbl_physics=0,isfflx=0,
               moist=moist,open_x=boundary,open_y=boundary,
               diff_6th_opt=2,diff_6th_factor=.12)
    declared={f.name for f in fields(RunConfig)}
    if "diff_opt" in declared:knobs.update(diff_opt=diff,mix_full_fields=mix)
    return RunConfig(**knobs)


def model_state(cfg):
    import cupy as cp
    from gpuwm.core.grid import make_base_state,make_vertical_coord
    from gpuwm.core.state import init_theta_perturbation
    from gpuwm.core.diagnostics import update_diagnostics
    vc=make_vertical_coord(cfg.nz)
    base=make_base_state(vc,lambda z:300.+.003*np.asarray(z,float),cfg.p_surf,cfg.ztop)
    state=init_theta_perturbation(cfg,vc,base,
        lambda x,z:np.broadcast_to(.15*np.sin(2*np.pi*x/(cfg.nx*cfg.dx))[None,None,:],
                                   (cfg.nz,cfg.ny,cfg.nx)))
    rng=np.random.default_rng(6171)
    u=(rng.normal(size=state.u.shape)*.7).astype("f4")
    v=(rng.normal(size=state.v.shape)*.7).astype("f4")
    u[:,:,-1]=u[:,:,0];v[:,-1,:]=v[:,0,:]
    state.u[:]=cp.asarray(u);state.v[:]=cp.asarray(v)
    if state.tke is not None:state.tke[:]=cp.asarray((.2+rng.random(state.tke.shape)*.3).astype("f4"))
    if state.qv is not None:
        state.qv[:]=.002;state.qc[:]=0.;state.qr[:]=0.
    update_diagnostics(state,cfg.hypsometric_opt)
    return state


def km2_case(case,values):
    import cupy as cp
    from gpuwm.core.grid import make_base_state,make_vertical_coord
    from gpuwm.core.state import init_at_rest
    get=lambda key:values[case["name"]+"_"+key]
    cfg=replace(configuration(km=2),dx=case["dx"],dy=case["dy"],dt=1.,
                open_x=bool(case["bx"]),open_y=bool(case["by"]),
                mix_isotropic=case.get("isotropic",0),isfflx=case.get("isfflx",0),
                tke_drag_coefficient=.0013,tke_heat_flux=.24)
    # Under diff_opt=1 WRF seeds even when these diff_opt=2 flux constants
    # are nonzero. This is the concrete branch distinction in tke_km.
    vc=make_vertical_coord(cfg.nz)
    base=make_base_state(vc,lambda z:np.full_like(np.asarray(z,float),300.),cfg.p_surf,cfg.ztop)
    state=init_at_rest(cfg,vc,base)
    state.phb=cp.asarray(get("phb"));state.php[:]=0.
    state.thb[:]=300.;state.thp[:]=0.;state.p[:]=100000.;state.alt[:]=1.
    if case["family"]=="deform":
        for name in ("u","v","w","dn","dnw","fnm","fnp"):
            target=getattr(state,name)
            value=get(name)[:target.size] if target.ndim==1 else get(name)
            target[:]=cp.asarray(value)
        state.cf1=1.875;state.cf2=-1.25;state.cf3=.375
    else:
        state.tke[:]=cp.asarray(get("tke"))
    state.msft[:]=cp.asarray(get("mt"));state.msfu[:]=cp.asarray(get("mfu"));state.msfv[:]=cp.asarray(get("mfv"))
    return cfg,state


def word_arrays(state):
    import cupy as cp
    names=("u","v","w","thp","php","mup","p","alt","qv","qc","qr","tke")
    arrays={name:cp.asnumpy(getattr(state,name)) for name in names if getattr(state,name,None) is not None}
    arrays.update({"scratch_"+name:cp.asnumpy(value) for name,value in state._scratch.items()
                   if name.startswith(("smag_","diff1_"))})
    return arrays
