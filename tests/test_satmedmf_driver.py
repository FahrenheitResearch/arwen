"""Real engine integration and restart for the new CCPP column."""
import hashlib,json
import numpy as np
import pytest
from gpuwm.config import RunConfig

def cfg(**overrides):
    values=dict(nx=16,ny=16,nz=24,dx=3000.,dy=3000.,dt=1.,ztop=15000.,
        run_seconds=4.,moist=True,mp_physics=6,sf_sfclay_physics=91,
        bl_pbl_physics=901,sf_surface_physics=2,num_soil_layers=4,
        bldt=0.,radt=0.,cu_physics=0,km_opt=4,c_s=0.25)
    values.update(overrides)
    return RunConfig(**values)

def make_state(c):
    import cupy as cp
    from gpuwm.core.grid import make_base_state,make_vertical_coord
    from gpuwm.core.state import init_at_rest
    from gpuwm.core.physics import initialize_physics,DECLARED_CONSTANT_GLW_WM2
    coord=make_vertical_coord(c.nz)
    base=make_base_state(coord,lambda z:300.+0.003*np.asarray(z,np.float64),p_surf=c.p_surf,ztop=c.ztop)
    state=init_at_rest(c,coord,base)
    state.u.fill(cp.float32(6.0));state.qv.fill(cp.float32(0.008))
    initialize_physics(state,c,landmask=1.,tsk=310.,glw=DECLARED_CONSTANT_GLW_WM2,swdown=600.)
    return state

def hashes(state):
    from gpuwm.state_serialization_contract import serialized_state_attrs
    out={}
    for name in serialized_state_attrs(state):
        value=getattr(state,name,None)
        if value is None:continue
        if hasattr(value,"get"):value=value.get()
        if isinstance(value,np.ndarray):
            out[name]=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    driver=getattr(state,"physics",None)
    if driver is not None:
        for name,value in driver.fields.items():
            if hasattr(value,"get"):value=value.get()
            if isinstance(value,np.ndarray):
                out["physics/"+name]=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return out

@pytest.mark.gpu
def test_real_driver_advects_tke_and_restart_is_byte_identical(tmp_path):
    import cupy as cp
    from gpuwm.core.dycore import run_steps
    from gpuwm.io.restart import write_restart,restore_restart
    c=cfg();live=make_state(c)
    initial=live.tke_edmf.copy()
    run_steps(live,c,2)
    assert not bool(cp.array_equal(initial,live.tke_edmf))
    assert bool(cp.isfinite(live.tke_edmf).all())
    assert live.physics.scheme_dispatch["bl_pbl_physics"]=="_run_satmedmf"
    from gpuwm.core.preflight import physics_array_shapes
    priced = physics_array_shapes(c)
    actual_rain = live.physics.pbl_tendencies.rqr
    assert actual_rain is not None
    assert priced["pbl_tendencies/rqr"] == actual_rain.shape
    assert int(np.prod(priced["pbl_tendencies/rqr"])) * actual_rain.dtype.itemsize == actual_rain.nbytes
    path=write_restart(tmp_path/"restart.npz",live,c)
    resumed=make_state(c)
    restore_restart(path,resumed,c)
    assert hashes(live)==hashes(resumed)
    run_steps(live,c,2);run_steps(resumed,c,2)
    assert hashes(live)==hashes(resumed)
    receipt={"verdict":"PASS","steps_before_restart":2,"steps_after_restart":2,
             "field_hashes":hashes(live)}
    (tmp_path/"restart-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")

@pytest.mark.gpu
def test_existing_scalar_path_transports_energy_without_counting_water():
    import cupy as cp
    from gpuwm.core.dycore import _save_time_t,stage_fluxes
    from gpuwm.core.moist import extra_moist_species,WRF_MOIST_ARRAY_SPECIES
    state=make_state(cfg(mp_physics=1))
    assert extra_moist_species(state)==("tke_edmf",)
    assert "tke_edmf" not in WRF_MOIST_ARRAY_SPECIES
    state.tke_edmf[:,4:8,4:8]=cp.float32(0.8)
    _save_time_t(state)
    assert bool(cp.array_equal(state.tke_edmf,state.tke_edmf0))
    from gpuwm.core.moist import advance_scalars_stage
    c=cfg(mp_physics=1)
    before=state.tke_edmf.copy()
    ru,rv,ww=stage_fluxes(state,c)
    advance_scalars_stage(state,c,ru,rv,ww,dt_eff=60.,final=True)
    assert not bool(cp.array_equal(before,state.tke_edmf))
    assert bool(cp.isfinite(state.tke_edmf).all())
    chm=state.c1h[:,None,None]*state.total_mu()[None]+state.c2h[:,None,None]
    old_mass=float(cp.sum(chm.astype(cp.float64)*before.astype(cp.float64)).get())
    new_mass=float(cp.sum(chm.astype(cp.float64)*state.tke_edmf.astype(cp.float64)).get())
    assert abs(new_mass-old_mass)/old_mass<1.e-6
    x=cp.arange(c.nx,dtype=cp.float64)[None,None,:]
    old_center=float(cp.sum((before-0.2)*x).get())
    new_center=float(cp.sum((state.tke_edmf-0.2)*x).get())
    assert new_center>old_center


@pytest.mark.gpu
def test_real_satmedmf_driver_supplies_grell_freitas_coupling():
    import cupy as cp
    from gpuwm.core.dycore import run_steps
    c = cfg(cu_physics=3)
    state = make_state(c)
    run_steps(state, c, 2)
    assert bool(cp.isfinite(state.tke_edmf).all())
    assert state.physics.scheme_dispatch["bl_pbl_physics"] == "_run_satmedmf"
    assert hashes(state)
    for value in state.physics.fields.values():
        if isinstance(value, cp.ndarray) and value.dtype.kind == "f":
            assert bool(cp.isfinite(value).all())
