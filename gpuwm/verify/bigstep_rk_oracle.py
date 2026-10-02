"""Replay compiled WRF RK fixtures through the production engine launches.

All output memory words, including unchanged halos and top rows, are measured.
No tolerance is chosen here. The parity tests pin the observed measurements.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.verify.wrf471_fixtures import require_fixture_dir

# Test data in a source checkout, not package data (gpuwm.verify.wrf471_fixtures).
RK_ORACLE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "wrf471_bigstep"


def load_rk_oracle():
    path=require_fixture_dir(RK_ORACLE_DIR, "big-step RK") / "rk-cases.npz"
    with np.load(path) as src:
        arrays={key:src[key] for key in src.files}
    meta=json.loads(path.with_suffix(".json").read_text())
    return arrays,meta


def _crop3(a,nz,ny,nx):
    return np.ascontiguousarray(a[4:4+ny,:nz,4:4+nx].transpose(1,0,2))


def _put3(a,value,nz,ny,nx):
    a[4:4+ny,:nz,4:4+nx]=value.transpose(1,0,2)


def replay_rk_case(arrays,case):
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.core.state import DomainState
    from gpuwm.core.dycore import (_couple_dry_mixing_map_factor,
        _smag2d_specs, add_fixed_dry_tendencies, add_h_diabatic_tendency,
        _prepare_bookkeeping)
    from gpuwm.core.moist import (_update_scalar_in_place,
                                  _exclude_specified_ring_advection)

    nx,ny,nz=(int(arrays[k]) for k in ("nx","ny","nz"))
    ident=case["id"]
    c1=cp.asarray(arrays["c1"][:nz])
    c2=cp.asarray(arrays["c2"][:nz])
    if case["kind"]=="dry":
        prefix=f"dry{ident}_"
        a,b,mu=(arrays[prefix+k].copy() for k in ("a","b","mu"))
        maps=arrays[prefix+"maps"]
        # A real DomainState instance selects the production held-heating launch.
        state=DomainState.__new__(DomainState)
        state.c1h,state.c2h=c1,c2
        state.c1f=cp.zeros(nz+1,cp.float32)
        state.c2f=cp.zeros(nz+1,cp.float32)
        state.qv=None
        state.p=cp.empty((nz,ny,nx),cp.float32)
        state.has_msf=True
        state.msft=cp.asarray(maps[1,4:4+ny,4:4+nx])
        state.msfu=cp.asarray(maps[3,4:4+ny,4:5+nx])
        state.msfv=cp.asarray(maps[4,4:5+ny,4:4+nx])
        state.mub2d=cp.asarray(arrays[prefix+"mut"][4:4+ny,4:4+nx])
        state.mup=cp.zeros((ny,nx),cp.float32)
        state.h_diabatic=cp.asarray(_crop3(arrays[prefix+"saves"][5],nz,ny,nx))
        state._scratch={}
        shape=((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx),(nz+1,ny,nx),(nz,ny,nx))
        for index,((nk,nj,ni),name,slot) in enumerate(zip(shape,
                ("ru_t","rv_t","rw_t","rph_t","rth_t"),
                ("smag_ru","smag_rv","smag_rw",None,"smag_rth"))):
            value=cp.asarray(_crop3(a[index],nk,nj,ni))
            setattr(state,name,value)
            if slot is not None:
                state._scratch[slot]=cp.asarray(_crop3(b[index],nk,nj,ni))
        state.u0=state.ru_t; state.v0=state.rv_t
        state.w0=state.rw_t; state.thp0=state.rth_t
        specs=_smag2d_specs(state,None,None,time_t=True)
        _couple_dry_mixing_map_factor(state,specs)
        add_fixed_dry_tendencies(state,SimpleNamespace(km_opt=4,diff_6th_opt=0))
        add_h_diabatic_tendency(state)
        for index,((nk,nj,ni),name) in enumerate(zip(shape,
                ("ru_t","rv_t","rw_t","rph_t","rth_t"))):
            _put3(a[index],cp.asnumpy(getattr(state,name)),nk,nj,ni)
        updated=cp.asarray(mu[0,4:4+ny,4:4+nx])
        updated+=cp.asarray(mu[1,4:4+ny,4:4+nx])
        mu[0,4:4+ny,4:4+nx]=cp.asnumpy(updated)
        return {"a":a,"b":b,"mu":mu}

    prefix=f"scalar{ident}_"
    s1,s2,decomp=(arrays[prefix+k].copy() for k in ("s1","s2","decomp"))
    mu=arrays[prefix+"mu"]
    mu0=cp.asarray(np.ascontiguousarray(mu[0,4:4+ny,4:4+nx]))
    mub=cp.asarray(np.ascontiguousarray(mu[2,4:4+ny,4:4+nx]))
    mu0+=mub
    munew=cp.asarray(np.ascontiguousarray(mu[1,4:4+ny,4:4+nx]))
    munew+=mub
    adv=cp.asarray(_crop3(arrays[prefix+"adv"],nz,ny,nx))
    if case["specified"] or case["nested"]:
        _exclude_specified_ring_advection(adv,case["spec_zone"])
    msft=cp.asarray(np.ascontiguousarray(arrays[prefix+"maps"][1,4:4+ny,4:4+nx]))
    for species in range(s1.shape[0]):
        q0=cp.asarray(_crop3(s1[species],nz,ny,nx))
        q=cp.asarray(_crop3(s2[species],nz,ny,nx))
        if case["step"]==1:
            # The dycore's same cached word-copy kernel saves time-t scalars.
            launch=_prepare_bookkeeping(SimpleNamespace(),((q,q0),))
            launch()
        source=cp.asarray(_crop3(arrays[prefix+"st"][species],nz,ny,nx))
        _update_scalar_in_place(q,q0,adv,c1,c2,mu0.reshape(-1),munew.reshape(-1),
                                case["dt"],msft=msft.reshape(-1),
                                physics=source,clamp=case["step"]==3)
        _put3(s1[species],cp.asnumpy(q0),nz,ny,nx)
        _put3(s2[species],cp.asnumpy(q),nz,ny,nx)
    return {"s1":s1,"s2":s2,"decomp":decomp}


def measure_rk_case(arrays,case):
    got=replay_rk_case(arrays,case)
    prefix=f"{case['kind']}{case['id']}_"
    result={}
    for name,value in got.items():
        want=arrays[prefix+name+"_wrf"]
        distances=fp32_ulp_distance(value,want)
        bits=value.view(np.uint32)!=want.view(np.uint32)
        worst=int(distances.max(initial=0))
        index=tuple(int(k) for k in np.unravel_index(np.argmax(distances),distances.shape))
        result[name]={"max_ulp":worst,"different_words":int(bits.sum()),
                      "words":int(value.size),"worst_index":list(index),
                      "gpu_word":int(value.view(np.uint32)[index]),
                      "wrf_word":int(want.view(np.uint32)[index]),
                      "actual_sha256":hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                      "reference_sha256":hashlib.sha256(np.ascontiguousarray(want).tobytes()).hexdigest()}
    return result


def measure_rk_parity():
    arrays,meta=load_rk_oracle()
    return {case["kind"]+":"+case["name"]:measure_rk_case(arrays,case)
            for case in meta["cases"]}


def load_rk_tendency_oracle(filename="rk-tendency.npz"):
    path=require_fixture_dir(RK_ORACLE_DIR, "big-step RK") / filename
    with np.load(path) as src:
        arrays={key:src[key] for key in src.files}
    return arrays,json.loads(path.with_suffix(".json").read_text())


def replay_rk_tendency_case(arrays,case):
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.state import DomainState
    from gpuwm.core.dycore import (_add_slow_tendencies,_prepare_tendency_zero,
        enable_wrf_cfl_recording,record_wrf_vertical_cfl,take_wrf_cfl,
        reset_wrf_cfl_recording)
    from gpuwm.core.acoustic import prepare_moist_cq
    nx,ny,nz=(int(arrays[k]) for k in ("nx","ny","nz"))
    prefix=f"tendency{case['id']}_"
    x,m,z,q,coeff=(arrays[prefix+k] for k in ("x","m","z","q","coeff"))
    native_prep=arrays[prefix+"prep_before_wrf"]
    state=DomainState.__new__(DomainState)
    state._scratch={}
    state.has_msf=True;state.rotational=True
    state.u=cp.asarray(_crop3(x[0],nz,ny,nx+1))
    state.v=cp.asarray(_crop3(x[1],nz,ny+1,nx))
    state.w=cp.asarray(_crop3(x[2],nz+1,ny,nx))
    # The Fortran input is the same full theta the production advection reads.
    state.thb=cp.zeros(nz,cp.float32)
    state.thp=cp.asarray(_crop3(x[3],nz,ny,nx))
    state.php=cp.asarray(_crop3(x[4],nz+1,ny,nx))
    state.phb=cp.asarray(_crop3(x[5],nz+1,ny,nx))
    state.pb=cp.asarray(_crop3(x[6],nz,ny,nx))
    # WRF stores perturbation P; the engine stores total pressure P+PB.
    state.p=cp.asarray(_crop3(x[7]+x[6],nz,ny,nx))
    state.al=cp.asarray(_crop3(x[8],nz,ny,nx))
    state.alt=cp.asarray(_crop3(native_prep[5],nz,ny,nx))
    for n,name in enumerate(("mup","mub2d","_msftx","msft","_msfux","msfu", "msfv","_msfv_inv","_msfvy",
                             "_clat","f","e","sina","cosa","_ht")):
        nj=ny+1 if n in (6,7,8) else ny
        ni=nx+1 if n in (4,5) else nx
        setattr(state,name,cp.asarray(np.ascontiguousarray(m[n,4:4+nj,4:4+ni])))
    for n,name in enumerate(("c1h","c2h","c1f","c2f","fnm","fnp","dnw","rdn","rdnw")):
        count=nz+1 if name in ("c1f","c2f","fnm","fnp") else nz
        setattr(state,name,cp.asarray(z[n,:count]))
    state.cf1,state.cf2,state.cf3,state.cfn,state.cfn1=(np.float32(v) for v in coeff)
    for n,name in enumerate(("qv","qc","qr","qi","qs","qg"),1):
        setattr(state,name,cp.asarray(_crop3(q[n],nz,ny,nx)))
    shapes=((nz,ny,nx+1),(nz,ny+1,nx),(nz+1,ny,nx),(nz+1,ny,nx),(nz,ny,nx))
    for name,shape in zip(("ru_t","rv_t","rw_t","rph_t","rth_t"),shapes):
        setattr(state,name,cp.empty(shape,cp.float32))
    state.rmu_t=cp.empty((ny,nx),cp.float32)
    _prepare_tendency_zero(state)()
    ru=cp.asarray(_crop3(native_prep[0],nz,ny,nx+1))
    rv=cp.asarray(_crop3(native_prep[1],nz,ny+1,nx))
    ww=cp.asarray(_crop3(native_prep[3],nz+1,ny,nx))
    cqu=cp.asarray(_crop3(native_prep[6],nz,ny,nx+1))
    cqv=cp.asarray(_crop3(native_prep[7],nz,ny+1,nx))
    cqw=cp.asarray(_crop3(native_prep[8],nz+1,ny,nx))
    cfg=RunConfig(nx=nx,ny=ny,nz=nz,dt=20.,dx=3000.,dy=3000.,
                  ztop=float(np.max(x[4]+x[5])/9.81),run_seconds=20.,specified=True,
                  h_sca_adv_order=5,top_lid=True,w_damping=0,mp_physics=6,
                  moist=True,moist_cq=True)
    # Its documented representation is pg_buoy_w's consumed reciprocal,
    # rather than calc_cq's pre-buoyancy mixing-ratio temporary.
    _generated_cqu,_generated_cqv,cqw,_use_cq=prepare_moist_cq(state,cfg)
    _add_slow_tendencies(state,cfg,ru,rv,ww,cq=(cqu,cqv,cqw,True))
    tend=np.zeros((5,ny+8,nz+1,nx+8),np.float32)
    for n,(name,(nk,nj,ni)) in enumerate(zip(("ru_t","rv_t","rw_t","rph_t","rth_t"),shapes)):
        _put3(tend[n],cp.asnumpy(getattr(state,name)),nk,nj,ni)
    held=np.zeros_like(tend);saved=np.zeros_like(tend)
    mtend=np.zeros((2,ny+8,nx+8),np.float32)
    mtend[0,4:4+ny,4:4+nx]=cp.asnumpy(state.rmu_t)
    misc=np.zeros((5,ny+8,nz+1,nx+8),np.float32)
    _put3(misc[1],cp.asnumpy(ww),nz+1,ny,nx)
    enable_wrf_cfl_recording()
    try:
        record_wrf_vertical_cfl(state,cfg,ww)
        cfl=np.array(take_wrf_cfl(cfg.grid_id),np.float32)
    finally:
        reset_wrf_cfl_recording()
    # Include the pre-consumed engine cqw representation in the word report.
    cqw_memory=native_prep[8].copy()
    _put3(cqw_memory,cp.asnumpy(cqw),nz+1,ny,nx)
    return {"tend":tend,"held":held,"saved":saved,"mtend":mtend,
            "misc":misc,"cqw":cqw_memory,"cfl":cfl}


def measure_rk_tendency_case(arrays,case):
    got=replay_rk_tendency_case(arrays,case)
    prefix=f"tendency{case['id']}_"
    result={}
    nx,ny,nz=(int(arrays[k]) for k in ("nx","ny","nz"))
    for name,value in got.items():
        want=(arrays[prefix+"prep_wrf"][8]
              if name=="cqw" else arrays[prefix+name+"_wrf"])
        distances=fp32_ulp_distance(value,want)
        bits=value.view(np.uint32)!=want.view(np.uint32)
        index=tuple(int(k) for k in np.unravel_index(np.argmax(distances),distances.shape))
        result[name]={"max_ulp":int(distances.max(initial=0)),"different_words":int(bits.sum()),
                      "words":int(value.size),"worst_index":list(index),
                      "gpu_word":int(value.view(np.uint32)[index]),"wrf_word":int(want.view(np.uint32)[index]),
                      "actual_sha256":hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                      "reference_sha256":hashlib.sha256(np.ascontiguousarray(want).tobytes()).hexdigest()}
        if name=="tend":
            result[name]["fields"]={}
            for field,actual,reference in zip(("ru","rv","rw","ph","theta"),value,want):
                d=fp32_ulp_distance(actual,reference)
                at=tuple(int(k) for k in np.unravel_index(np.argmax(d),d.shape))
                result[name]["fields"][field]={
                    "max_ulp":int(d.max(initial=0)),
                    "different_words":int(np.count_nonzero(actual.view(np.uint32)!=reference.view(np.uint32))),
                    "max_abs":float(np.max(np.abs(actual.astype(np.float64)-reference))),
                    "worst_index":list(at),"actual":float(actual[at]),"reference":float(reference[at]),
                    "actual_sha256":hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest(),
                    "reference_sha256":hashlib.sha256(np.ascontiguousarray(reference).tobytes()).hexdigest()}
    return result


def measure_rk_tendency_parity(filename="rk-tendency.npz"):
    arrays,meta=load_rk_tendency_oracle(filename)
    return {case["name"]:measure_rk_tendency_case(arrays,case) for case in meta["cases"]}
