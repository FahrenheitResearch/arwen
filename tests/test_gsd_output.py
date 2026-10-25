"""Pinned operational GSD output, device formulas and state lifecycle."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.config import RunConfig
from gpuwm.core.gsd_diagnostics import HISTORY, reset_interval, update, update_helicity

ROOT=Path(__file__).resolve().parents[1]


def test_history_reset_accepts_a_reduced_state_without_a_scratch_pool():
    from gpuwm.core.uh_diag import reset_up_heli_max
    reset_up_heli_max(SimpleNamespace())

def test_gsd_native_route_round_trip(tmp_path):
    from test_route_round_trip_fields import _emit,_replay
    _,paths=_emit(tmp_path,"gsd-windows",gsd_diagnostics=1,reset_interval1=30)
    cfg=_replay(paths).root.run
    assert (cfg.gsd_diagnostics,cfg.reset_interval1)==(1,30)


def test_output_kernels_are_priced_and_compiled_before_first_step():
    from dataclasses import replace
    from gpuwm.experiment import load_experiment
    from gpuwm.core import preflight
    from gpuwm import kernel_prewarm
    exp=load_experiment(ROOT/"gpuwm/data/recipes/hrrr_v4_gsd41.toml")
    modules={"gsd_diagnostics","gsd_helicity","output_diagnostics"}
    assert modules <= preflight.physics_kernel_modules(exp)
    assert modules <= set(kernel_prewarm.static_modules(exp))
    assert modules.isdisjoint(preflight.assumed_bound_modules(modules))
    assert all(preflight.KERNEL_MAX_LOCAL_SIZE_BYTES[name]==0 for name in modules)
    dc=exp.root
    inactive=replace(dc,run=replace(dc.run,gsd_diagnostics=0,prec_acc_dt=0.,prec_acc_dt1=0.))
    assert modules.isdisjoint(preflight.domain_kernel_modules(inactive,prices_refl=False))


def test_gsd_restart_history_and_memory_inventory(monkeypatch,tmp_path):
    from test_restart import _cfg,_shim_state,_fill_setup
    from gpuwm.io import restart
    from gpuwm.io.history_layout import produced_history_shapes,live_state_history_fields
    from gpuwm.core import preflight
    from gpuwm.core.output_diagnostics import scratch_shapes
    from gpuwm.core.uh_diag import reset_up_heli_max
    cfg=_cfg(gsd_diagnostics=1)
    source=_shim_state(cfg,monkeypatch);_fill_setup(source)
    for slot in HISTORY:
        source.existing_scratch(slot)[...] = 7.25
    reset_up_heli_max(source)
    for slot,(symbol,_,_) in HISTORY.items():
        np.testing.assert_array_equal(source.existing_scratch(slot),7.25)
        assert symbol in live_state_history_fields(source)
        assert produced_history_shapes(cfg)[symbol] == source.existing_scratch(slot).shape
    path=restart.write_restart(tmp_path/"gsd.npz",source,cfg)
    target=_shim_state(cfg,monkeypatch);_fill_setup(target)
    restart.restore_restart(path,target,cfg)
    for slot in HISTORY:
        np.testing.assert_array_equal(target.existing_scratch(slot),7.25)
    for slot,shape in scratch_shapes(cfg).items():
        assert preflight.scratch_slot_registry(cfg)[slot] == shape
        assert preflight.scratch_slot_lifetime(slot).kind == "carrying"
        assert restart.classify_scratch_slot(slot) in ("serialize","rebuild")


@requires_gpu
def test_fork_thompson_hail_against_extracted_fortran_words():
    import cupy as cp
    from gpuwm.core.kernels import get_kernel
    rows=[l.split() for l in (ROOT/"tests/data/oracles/gsd_output/vectors.txt").read_text().splitlines() if l.startswith("HAIL")]
    words=np.array([[int(x,16) for x in row[1:]] for row in rows],dtype=np.uint32)
    mass=cp.asarray(words[:,0].copy().view(np.float32));out=cp.zeros_like(mass)
    get_kernel("gsd_diagnostics","gsd_hail_probe")((1,),(32,),(mass,out,np.int32(len(rows))))
    np.testing.assert_array_equal(cp.asnumpy(out).view(np.uint32),words[:,1])


def _gpu_state():
    import cupy as cp
    from gpuwm.core.state import DomainState
    from gpuwm.core.grid import make_base_state,make_vertical_coord
    cfg=RunConfig(nx=6,ny=6,nz=12,dx=3000.,dy=3000.,dt=20.,ztop=12000.,
        run_seconds=3600.,specified=True,gsd_diagnostics=1,moist=True,mp_physics=28,hailcast_opt=0)
    state=DomainState(cfg);vc=make_vertical_coord(cfg.nz)
    base=make_base_state(vc,lambda z:300.+.003*np.asarray(z,float),p_surf=cfg.p_surf,ztop=cfg.ztop)
    state.load_base(vc,base)
    # Manufactured flat metric column, rho=1 and exact geopotential heights.
    state.p[...] = 100000.;state.alt[...] = 1.;state.qv[...] = 0.
    state.qc[...] = .001;state.qi[...] = .001;state.qs[...] = 0.;state.qr[...] = .001
    state.qg[...] = .001
    state.php[...] = 0.;state.phb[...] = cp.arange(cfg.nz+1,dtype=cp.float32)*9810.
    state.thp[...] = 0.;state.thb[...] = 250.
    state.w[...] = 5.;state.w[0]=0.;state.w[-1]=0.
    state.u[...] = 3.;state.v[...] = 4.
    state.physics=SimpleNamespace(fields={"u10":cp.full((6,6),3.,dtype=cp.float32),
        "v10":cp.full((6,6),4.,dtype=cp.float32)})
    return state,cfg


@requires_gpu
def test_gsd_gpu_windows_columns_and_uh_bands():
    import cupy as cp
    state,cfg=_gpu_state()
    # Solid-body rotation, positive vorticity at every model level.
    state.u[...] = -cp.arange(cfg.ny,dtype=cp.float32)[None,:,None]*3.
    state.v[...] = cp.arange(cfg.nx,dtype=cp.float32)[None,None,:]*3.
    state.elapsed_seconds=20.
    update_helicity(state,cfg,history_due=True)
    heights=cp.asnumpy(state.phb)/np.float32(9.81)
    from gpuwm.core.gsd_diagnostics import BANDS
    for suffix,lower,upper in BANDS:
        expected=sum(.002 * (3.75 if k==0 else 5.) * float(heights[k+1]-heights[k])
                     for k in range(cfg.nz) if heights[k]>=lower and heights[k+1]<=upper)
        got=cp.asnumpy(state.existing_scratch("up_heli_max"+suffix))[2:4,2:4]
        np.testing.assert_allclose(got,expected,atol=.0001,rtol=1.e-6)
    update(state,cfg,history_due=True)
    for slot,want in (("wspd10max",5.),("w_up_max",5.),("w_dn_max",0.),
                      ("w_mean",5.),("grpl_max",12.)):
        np.testing.assert_allclose(cp.asnumpy(state.existing_scratch(slot)),want,atol=2.e-5)
    assert cp.asnumpy(state.existing_scratch("hail_max2d")).min()>0.
    before=state.existing_scratch("wspd10max").copy()
    state.physics.fields["u10"][...] = 0.;state.physics.fields["v10"][...] = 1.
    state.elapsed_seconds=900.;reset_interval(state,cfg);update(state,cfg,history_due=True)
    np.testing.assert_array_equal(cp.asnumpy(state.existing_scratch("wspd10max")),cp.asnumpy(before))
    state.elapsed_seconds=3600.;reset_interval(state,cfg)
    np.testing.assert_array_equal(cp.asnumpy(state.existing_scratch("up_heli_max")),0.)
    update(state,cfg,history_due=False)
    np.testing.assert_array_equal(cp.asnumpy(state.existing_scratch("wspd10max")),1.)
    np.testing.assert_array_equal(cp.asnumpy(state.existing_scratch("wspd10")),-99.)


@requires_gpu
def test_vorticity_previous_value_uses_native_step_modulo_with_changing_dt():
    from dataclasses import replace
    import cupy as cp
    state,cfg=_gpu_state()
    state.u[...] = -cp.arange(cfg.ny,dtype=cp.float32)[None,:,None]*3.
    state.v[...] = cp.arange(cfg.nx,dtype=cp.float32)[None,None,:]*3.
    maximum=state.existing_scratch("rel_vort_max")
    total=state.existing_scratch("gsd_total_steps")
    # At timestep 18 and dt=20, INT(3600/20)=180: retain the previous max.
    total[...] = 17.;maximum[...] = 100.
    update_helicity(state,cfg)
    np.testing.assert_array_equal(cp.asnumpy(maximum)[2:4,2:4],100.)
    # Changing dt changes native idump, independently of clock-hour reset.
    total[...] = 72.;maximum[...] = 100.;state.elapsed_seconds=1700.
    update_helicity(state,replace(cfg,dt=50.))
    np.testing.assert_allclose(cp.asnumpy(maximum)[2:4,2:4],.002,rtol=2.e-6)
    np.testing.assert_array_equal(cp.asnumpy(total),73.)
    state.elapsed_seconds=3600.
    reset_interval(state,cfg)
    np.testing.assert_array_equal(cp.asnumpy(total),73.)


@requires_gpu
def test_gsd_domain_lightning_reduction_is_independent_of_tile_partition():
    import cupy as cp
    from gpuwm.core.gsd_diagnostics import finalize_history_fields
    a=np.array([[2.,40.,1.,1.]],dtype=np.float32)
    b=np.array([[80.,2.,1.,20.]],dtype=np.float32)
    full={"LTG1_MAX":cp.asarray(a),"LTG2_MAX":cp.asarray(b),"LTG3_MAX":cp.zeros_like(cp.asarray(a))}
    finalize_history_fields(full)
    # A streamed frame has assembled the carriers; only its final global
    # ratio can undo the incorrect tile-local maximum ratios.
    host={"LTG1_MAX":a,"LTG2_MAX":b,"LTG3_MAX":np.full_like(a,-123.)}
    finalize_history_fields(host)
    np.testing.assert_array_equal(host["LTG3_MAX"],cp.asnumpy(full["LTG3_MAX"]))
    np.testing.assert_allclose(host["LTG3_MAX"],.95*a+.05*(40./80.)*b,rtol=1.e-7)


def test_gsd_does_not_change_completed_step_tracker_samples(monkeypatch):
    from dataclasses import replace
    from test_uh_lifecycle import _loaded_numpy_state, _step_uh
    from gpuwm.core.uh_diag import update_up_heli_max
    plain,cfg=_loaded_numpy_state(monkeypatch)
    gsd,_=_loaded_numpy_state(monkeypatch)
    gsd_cfg=replace(cfg,gsd_diagnostics=1)
    gsd.existing_scratch('up_heli_max')[...]=1234.
    for scale in (.4,1.,.2):
        _step_uh(plain,cfg,scale)
        gsd.u[...]=plain.u;gsd.v[...]=plain.v;gsd.w[...]=plain.w
        update_up_heli_max(gsd,gsd_cfg)
    for slot in ('uh_follow_window','uh_spawn_window'):
        np.testing.assert_array_equal(gsd.existing_scratch(slot),plain.existing_scratch(slot))
    np.testing.assert_array_equal(gsd.existing_scratch('up_heli_max'),1234.)


@requires_gpu
def test_gsd_holds_start_step_eos_and_writes_native_reflectivity_slices():
    import cupy as cp
    state,cfg=_gpu_state()
    state.p[...] = cp.linspace(95000.,15000.,cfg.nz,dtype=cp.float32)[:,None,None]
    state.alt[...] = 2.;state.qv[...] = .02
    state.elapsed_seconds=20.
    update_helicity(state,cfg,history_due=True)
    pw=cp.asnumpy(state.existing_scratch('gsd_pressure_w')).copy()
    rho=cp.asnumpy(state.existing_scratch('gsd_density')).copy()
    np.testing.assert_array_equal(rho,np.float32(.51))
    p=cp.asnumpy(state.p)[:,0,0]
    np.testing.assert_allclose(pw[0],1.5*p[0]-.5*p[1],rtol=2.e-7)
    np.testing.assert_allclose(pw[-1],np.exp(1.5*np.log(p[-1])-.5*np.log(p[-2])),rtol=2.e-6)
    # The post-RK EOS differs; LWP must still use the pre-RK p8w.
    state.p[...] = 80000.;state.alt[...] = 3.
    refl=state.scratch(state.p.shape,'refl_10cm')
    refl[...] = cp.arange(cfg.nz,dtype=cp.float32)[:,None,None]
    update(state,cfg,history_due=True)
    expected=np.maximum((pw[:-1]-pw[1:])/np.float32(9.81)*np.float32(.001),0).sum(axis=0)*1000.
    np.testing.assert_allclose(cp.asnumpy(state.existing_scratch('lwp')),expected,rtol=3.e-7)
    for name,want in (('composite_refl_10cm',11.),('refl_10cm_1km',2.),('refl_10cm_4km',5.)):
        np.testing.assert_allclose(cp.asnumpy(state.existing_scratch(name)),want,atol=2.e-5)


@requires_gpu
def test_gsd_output_is_trajectory_inert_through_real_steps(tmp_path):
    import cupy as cp
    from gpuwm.core.dycore import step
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.refl import consume_refl_10cm
    from test_mp_spec_zone_ring import _square_moist_state
    off,cfg0=_square_moist_state(mp=28,open_x=True,open_y=True,dt=2.,hailcast_opt=0)
    on,cfg1=_square_moist_state(mp=28,open_x=True,open_y=True,dt=2.,hailcast_opt=0,
                              gsd_diagnostics=1,prec_acc_dt=60.,prec_acc_dt1=15.)
    initialize_physics(off,cfg0)
    initialize_physics(on,cfg1)
    for _ in range(3):
        step(off,cfg0,refl_10cm_due=True)
        step(on,cfg1,refl_10cm_due=True)
        consume_refl_10cm(off);consume_refl_10cm(on)
    for name in ('u','v','w','thp','php','mup','p','al','alt','qv','qc','qr','qi','qs','qg','qn','qni','h_diabatic'):
        a,b=getattr(off,name,None),getattr(on,name,None)
        if a is not None:
            np.testing.assert_array_equal(cp.asnumpy(a),cp.asnumpy(b),err_msg=name)
    assert cp.asnumpy(on.existing_scratch('w_up_max')).max()>0.
    # Exercise the production asynchronous writer and Rust NetCDF path,
    # including all new 2-D fields and the 3-D vorticity field. Mutating
    # the live accumulators after submit must not change the snapshot.
    from gpuwm.io.wrfout import AsyncDomainWrfoutWriter
    from gpuwm.core.output_diagnostics import PRECIP_HISTORY
    from datetime import datetime
    from netCDF4 import Dataset
    inventory=HISTORY | PRECIP_HISTORY
    expected={symbol:cp.asnumpy(on.existing_scratch(slot))
              for slot,(symbol,_,_) in inventory.items()}
    path=tmp_path/'diagnostic-snapshot.nc'
    writer=AsyncDomainWrfoutWriter(nx=cfg1.nx,ny=cfg1.ny,nz=cfg1.nz,
        dx=cfg1.dx,dy=cfg1.dy,title='output-control-snapshot',global_attrs={})
    try:
        writer.submit(path,datetime(2026,10,21,0,0,6),on)
        for slot in inventory:
            on.existing_scratch(slot)[...] = -999.
    finally:
        writer.close()
    with Dataset(path) as nc:
        for symbol,values in expected.items():
            np.testing.assert_array_equal(nc.variables[symbol][0],values,err_msg=symbol)
