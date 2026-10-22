"""Actual surface driver, cadence, checkpoint and OML kernel gates."""
import hashlib
import json
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.config import RunConfig

def make_state(cp, *, ocean=1, hml0=50.):
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics
    cfg=RunConfig(nx=8,ny=6,nz=16,dx=10000.,dy=10000.,ztop=12000.,dt=5.,run_seconds=80.,
                  time_step_sound=4,moist=True,mp_physics=0,sf_sfclay_physics=1,
                  sf_surface_physics=0,bl_pbl_physics=0,sf_ocean_physics=ocean,oml_hml0=hml0,bldt=0.5,
                  ocean_mld_climatology='configured_uniform_mld')
    coord=make_vertical_coord(cfg.nz)
    base=make_base_state(coord,lambda z:298.+0.004*np.asarray(z),p_surf=cfg.p_surf,ztop=cfg.ztop)
    state=init_moist_balanced(cfg,coord,base,lambda z:0.015*np.exp(-np.asarray(z)/2400.))
    state.u[...] = cp.float32(35.); state.v[...] = cp.float32(-10.)
    land=np.zeros((cfg.ny,cfg.nx),np.float32); land[:,-1]=1.
    initial = None if hml0 > 0 else {"h0ml":cp.full((cfg.ny,cfg.nx),20.,dtype=cp.float32),
        "huml":cp.full((cfg.ny,cfg.nx),0.2,dtype=cp.float32),
        "hvml":cp.full((cfg.ny,cfg.nx),-0.1,dtype=cp.float32),
        "tmoml":cp.full((cfg.ny,cfg.nx),297.,dtype=cp.float32)}
    driver=initialize_physics(state,cfg,landmask=land,tsk=302.,glw=310.,swdown=0.,ocean_initial=initial)
    return state,cfg,driver

@requires_gpu
@pytest.mark.gpu
def test_surface_cadence_and_land_mask():
    import cupy as cp
    from gpuwm.core.dycore import run_steps
    state,cfg,driver=make_state(cp)
    run_steps(state,cfg,1)
    first=cp.asnumpy(driver.fields["tml"])
    assert np.any(first[:,:-1] != np.float32(302.))
    assert np.all(first[:,-1] == np.float32(302.))
    run_steps(state,cfg,4)
    assert cp.asnumpy(driver.fields["tml"]).tobytes()==first.tobytes(), "OML ran between BLDT calls"
    run_steps(state,cfg,1)
    assert cp.asnumpy(driver.fields["tml"]).tobytes()!=first.tobytes(), "OML missed the next surface call"

@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("hml0", [50.,0.,-1.])
def test_actual_restart_roundtrip_and_continuation(tmp_path,hml0):
    import cupy as cp
    from gpuwm.core.dycore import run_steps
    from gpuwm.io import restart
    a,cfg,da=make_state(cp,hml0=hml0); run_steps(a,cfg,16)
    reference=restart.write_restart(tmp_path/"reference.npz",a,cfg)
    b,cfg,db=make_state(cp,hml0=hml0); run_steps(b,cfg,8)
    mid=restart.write_restart(tmp_path/"mid.npz",b,cfg)
    c,cfg,dc=make_state(cp,hml0=hml0); restart.restore_restart(mid,c,cfg)
    for name in ("tml","t0ml","hml","h0ml","huml","hvml","tmoml","tsk"):
        assert cp.asnumpy(dc.fields[name]).tobytes()==cp.asnumpy(db.fields[name]).tobytes(),name
    run_steps(c,cfg,8)
    resumed=restart.write_restart(tmp_path/"resumed.npz",c,cfg)
    hashes={}
    with np.load(reference,allow_pickle=False) as expected,np.load(resumed,allow_pickle=False) as got:
        assert set(expected.files)==set(got.files)
        for name in expected.files:
            if name == "__gpuwm_restart_header__":
                left=json.loads(expected[name].tobytes())
                right=json.loads(got[name].tobytes())
                # The archive records wall time and producer bookkeeping.
                # Compare every header member that determines continuation.
                for key in ("elapsed_seconds", "setup_fingerprint", "physics_setup_fingerprint",
                            "physics_setup", "driver", "array_manifest", "config"):
                    assert left[key] == right[key], key
                continue
            assert expected[name].tobytes()==got[name].tobytes(),name
            hashes[name]=hashlib.sha256(got[name].tobytes()).hexdigest()
    (tmp_path/"restart-hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")


@requires_gpu
@pytest.mark.gpu
def test_restart_or_gather_buffer_does_not_require_cold_start_inputs():
    import cupy as cp
    from gpuwm.core.surface_ocean import initialize_ocean, STATE_FIELDS
    cfg=RunConfig(nx=8,ny=6,nz=16,dx=10000.,dy=10000.,ztop=12000.,dt=5.,run_seconds=0.,
                  sf_ocean_physics=1,oml_hml0=-1.)
    fields={"tsk":cp.full((6,8),302.,dtype=cp.float32)}
    initialize_ocean(fields,cfg,start=False)
    assert all(bool(cp.all(fields[name] == 0.)) for name in STATE_FIELDS)
