"""Actual CUDA SASE held-step, restart and reused-tile control."""
import numpy as np
import pytest


def _gpu_state(cadence=0.1):
    import cupy as cp
    from gpuwm.config import RunConfig, SASE_PBL_SCHEME
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import RadiationResult, initialize_physics
    cfg=RunConfig(nx=12,ny=10,nz=16,dx=2000.,dy=2000.,ztop=8000.,
        dt=1.,run_seconds=8.,time_step_sound=4,moist=True,mp_physics=10,
        ra_physics=4,sf_sfclay_physics=1,sf_surface_physics=2,
        km_opt=0,bl_pbl_physics=SASE_PBL_SCHEME,bldt=cadence,
        sase_flux_diag=True,hmix_k_diag=True)
    coord=make_vertical_coord(cfg.nz)
    base=make_base_state(coord,lambda z:300.+0.004*np.asarray(z,np.float64),
        p_surf=cfg.p_surf,ztop=cfg.ztop)
    state=init_moist_balanced(cfg,coord,base,
        lambda z:0.010*np.exp(-np.asarray(z,np.float64)/2400.))
    shear=(5.+8.*state.height_half()/cfg.ztop).astype(np.float32)
    state.u[...] = cp.asarray(np.broadcast_to(shear[:,None,None],state.u.shape))
    state.v[...] = cp.float32(1.)
    state.w[...] = cp.asarray((0.05*np.sin(np.pi*np.arange(cfg.nz+1)/cfg.nz)[:,None,None]
        *np.ones((1,cfg.ny,1))*np.sin(2*np.pi*np.arange(cfg.nx)/cfg.nx)[None,None,:]).astype(np.float32))
    def radiation(**kwargs):
        z3=cp.zeros(state.p.shape,cp.float32)
        z2=cp.zeros(state.mup.shape,cp.float32)
        return RadiationResult(z3,cp.zeros_like(z3),z2,cp.zeros_like(z2))
    radiation.restart_identity = {"algorithm": "zero-flux-cadence-control-v1", "above_atmosphere_policy": "no-external-atmosphere"}
    initialize_physics(state,cfg,landmask=1.,tsk=302.,swdown=400.,glw=320.,radiation=radiation)
    return cfg,state


@pytest.mark.gpu
def test_actual_sase_held_step_restart_and_reused_tile_carriers(tmp_path):
    import cupy as cp
    from gpuwm.core import dycore
    from gpuwm.io import restart
    from tilestream import gather as G
    from tilestream.spec import plan_tiles
    from tilestream.physics_inventory import carrier_manifest
    cfg,straight=_gpu_state()
    dycore.step(straight,cfg)
    assert straight.physics.call_counts['sase']==1
    held=cp.asnumpy(straight.physics.pbl_tendencies.rw)
    assert np.isfinite(held).all() and np.any(held!=0.)
    checkpoint=restart.write_restart(tmp_path/'after-due.npz',straight,cfg)
    _,resumed=_gpu_state()
    restart.restore_restart(checkpoint,resumed,cfg)
    for _ in range(3):
        dycore.step(straight,cfg)
        dycore.step(resumed,cfg)
    assert straight.physics.call_counts['sase']==resumed.physics.call_counts['sase']==1
    for key,expected in carrier_manifest(straight).items():
        actual=carrier_manifest(resumed)[key]
        np.testing.assert_array_equal(cp.asnumpy(actual),cp.asnumpy(expected),err_msg=key)
        assert bool(cp.isfinite(actual).all()),key
    # Transfer the actual producer's values through four tile windows and
    # two reused device buffers; the z-face extent and seams must survive.
    fields={k:v for k,v in carrier_manifest(straight).items()
        if k=='driver/pbl_tendencies/rw' or k=='pbl/dw' or k.startswith('pbl/diagnostics/')}
    assert len(fields)==8
    store={key:G.pinned_copy(value) for key,value in fields.items()}
    sink={key:G.pinned_empty_like(value) for key,value in fields.items()}
    for value in sink.values(): value.fill(np.nan)
    specs=plan_tiles(cfg.nx,cfg.ny,6,5,halo=2,periodic_x=True,periodic_y=True)
    buffers=[{k:cp.empty(v.shape[:-2]+(9,10),dtype=v.dtype) for k,v in fields.items()} for _ in range(2)]
    stream=cp.cuda.Stream(non_blocking=True)
    for index,spec in enumerate(specs):
        tile=buffers[index%2]
        for value in tile.values(): value.fill(cp.nan)
        cp.cuda.get_current_stream().synchronize()
        G.gather_tile(store,tile,spec,stream,nz=cfg.nz,names=tuple(fields))
        G.scatter_tile(tile,sink,spec,stream,nz=cfg.nz,names=tuple(fields))
        stream.synchronize()
    for key in fields:
        np.testing.assert_array_equal(sink[key],store[key],err_msg=key)


