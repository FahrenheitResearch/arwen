"""Native HRRR t=0 helper: fallback, first-call scattering and history path."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.core.gsd_initial_reflectivity import launch

VECTORS=Path(__file__).parent/'data/oracles/gsd_initial_reflectivity/vectors.npz'

@requires_gpu
@pytest.mark.parametrize("mode", [0, 1])
def test_initial_helper_matches_extracted_fortran_words_without_changing_inputs(mode):
    import cupy as cp
    with np.load(VECTORS) as v:
        for case, values in enumerate(v['inputs'][mode]):
            fields=[cp.asarray(a) for a in values]
            refl=cp.empty_like(fields[0])
            launch(*fields,refl,state_inputs=bool(mode))
            got=cp.asnumpy(refl).view(np.uint32)
            for mp in (0,1):
                np.testing.assert_array_equal(got,v['expected'][mode,case,mp],err_msg=f'mode {mode}, case {case}, mp {mp}')
            for name, field, original in zip(v['names'], fields, values):
                np.testing.assert_array_equal(cp.asnumpy(field).view(np.uint32),original.view(np.uint32),err_msg=name)


@pytest.mark.parametrize("gsd,mp,radar,want", [(1,8,True,True), (1,28,True,True),
    (0,28,True,False), (1,6,True,False), (1,28,False,False)])
def test_native_initial_helper_selection(gsd, mp, radar, want):
    from gpuwm.core.gsd_initial_reflectivity import enabled
    assert enabled(SimpleNamespace(gsd_diagnostics=gsd,mp_physics=mp,do_radar_ref=radar)) == want
    assert not enabled(SimpleNamespace())


def _state(values, cfg, *, host=False, profile=False):
    import cupy as cp
    names=('thb','thp','p','alt','qv','qc','nr','qr','qs','qg')
    arrays={name:(value.copy() if host else cp.asarray(value))
            for name,value in zip(names,values)}
    if profile:
        arrays['thb']=arrays['thb'][:,0,0].copy()
    buffers={}
    def scratch(shape, name):
        if name not in buffers:
            buffers[name]=cp.zeros(shape,cp.float32)
        return buffers[name]
    return SimpleNamespace(**arrays,scratch=scratch,existing_scratch=buffers.get,_scratch=buffers,
                           physics=SimpleNamespace(mp_physics=cfg.mp_physics,refl_config=cfg))


@requires_gpu
@pytest.mark.parametrize("mp", [8,28])
@pytest.mark.parametrize("placement", ['resident','host','streamed'])
def test_first_history_uses_native_helper_and_preserves_prognostics(tmp_path, monkeypatch, mp, placement):
    import cupy as cp
    from gpuwm import wrf_exact
    from gpuwm.prepared_single_domain_forecast import _consume_due_native_refl_10cm
    from gpuwm.io.wrfout import WrfoutWriter
    from netCDF4 import Dataset
    # Even exact mode must use the HRRR helper rather than the stock t=0 zeros.
    monkeypatch.setattr(wrf_exact,'ENABLED',True)
    cfg=SimpleNamespace(gsd_diagnostics=1,mp_physics=mp,do_radar_ref=True)
    with np.load(VECTORS) as v:
        # 75 rows force two GPU slabs and catch reads from a stale slab template.
        values=np.tile(v['inputs'][1,2],(1,1,25,1))
        expected=np.tile(v['expected'][1,2,(0 if mp==8 else 1)],(1,25,1))
    state=_state(values,cfg,host=placement!='resident',profile=True)
    originals={n:cp.asnumpy(getattr(state,n)).copy() for n in
               ('thb','thp','p','alt','qv','qc','nr','qr','qs','qg')}
    if placement=='streamed':
        store={'state/'+n:getattr(state,n) for n in originals if n!='thb'}
        state._streamed_domain=SimpleNamespace(store=store,_geography={'setup/thb':state.thb})
        for name in originals:
            if name!='thb':
                setattr(state,name,np.full((9,1,8),-999,np.float32))
    def no_stash(_):
        pytest.fail('initial history tried to consume a forecast-time stash')
    frame=_consume_due_native_refl_10cm(state,0,no_stash,shape=expected.shape)
    np.testing.assert_array_equal(cp.asnumpy(frame).view(np.uint32),expected)
    # Delayed domain activation is also first_time_step in the fork.
    activated=_consume_due_native_refl_10cm(
        state,120,no_stash,domain_start_ticks=120,shape=expected.shape)
    np.testing.assert_array_equal(cp.asnumpy(activated).view(np.uint32),expected)
    if placement=='resident':
        assert not cp.any(state._scratch['refl_10cm'])
    for name,original in originals.items():
        actual=(state._streamed_domain.store['state/'+name]
                if placement=='streamed' and name!='thb' else getattr(state,name))
        np.testing.assert_array_equal(cp.asnumpy(actual).view(np.uint32),original.view(np.uint32),err_msg=name)
    # Later frames retain the forecast microphysics handoff.
    marker=object()
    assert _consume_due_native_refl_10cm(state,1,lambda _:marker) is marker
    path=tmp_path/'first.nc'
    frame=cp.asnumpy(frame)
    with WrfoutWriter(path,nx=8,ny=75,nz=9,dx=3000,dy=3000,
                     field_schema={'REFL_10CM':frame},global_attrs={}) as writer:
        writer.write_frame('2026-10-21_00:00:00',{'REFL_10CM':frame})
    with Dataset(path) as nc:
        np.testing.assert_array_equal(np.asarray(nc['REFL_10CM'][0]).view(np.uint32),expected)


def test_initial_helper_preflight_prices_the_actual_bounded_arena():
    from dataclasses import replace
    from datetime import datetime
    from gpuwm.config import RunConfig
    from gpuwm.experiment import experiment_from_run_config
    from gpuwm.core import preflight as pf
    from gpuwm.core.gsd_initial_reflectivity import tile_threads, workspace_bytes
    from gpuwm.kernel_prewarm import static_modules
    cfg=RunConfig(nx=100,ny=80,nz=50,dx=3000,dy=3000,ztop=20000,
                  dt=20,run_seconds=60,mp_physics=28,gsd_diagnostics=1,do_radar_ref=True)
    exp=experiment_from_run_config(cfg,datetime(2026,10,21))
    profile=pf.MEASURED_LOCAL_MEMORY_PROFILE
    want=workspace_bytes(50,tile_threads(profile.multiprocessor_count,8000))
    assert pf.gsd_initial_reflectivity_workspace_bytes(exp)==want
    assert 'gsd_initial_reflectivity' in pf.physics_kernel_modules(exp)
    assert 'gsd_initial_reflectivity' in static_modules(exp)
    off=experiment_from_run_config(replace(cfg,gsd_diagnostics=0),datetime(2026,10,21))
    assert pf.gsd_initial_reflectivity_workspace_bytes(off)==0
    assert 'gsd_initial_reflectivity' not in pf.physics_kernel_modules(off)
    assert pf.column_workspace_bytes(exp)-pf.column_workspace_bytes(off)==want


@requires_gpu
def test_stock_exact_initial_frame_stays_zero_without_gsd(monkeypatch):
    from gpuwm import wrf_exact
    from gpuwm.core.refl import analysis_refl_10cm
    monkeypatch.setattr(wrf_exact,'ENABLED',True)
    state=SimpleNamespace(qv=np.ones((3,2,4),np.float32))
    assert not np.any(analysis_refl_10cm(state))


@requires_gpu
def test_tree_history_calls_initial_helper_at_domain_activation():
    import cupy as cp
    from gpuwm.runtime import _submit_tree_history_frame
    cfg=SimpleNamespace(gsd_diagnostics=1,mp_physics=28,do_radar_ref=True,nx=8,ny=3,nz=9)
    with np.load(VECTORS) as v:
        state=_state(v['inputs'][1,0],cfg)
        expected=v['expected'][1,0,1].copy()
    node=SimpleNamespace(state=state,cfg=SimpleNamespace(run=cfg),
                         clock=SimpleNamespace(spec=SimpleNamespace(start_ticks=120)))
    frames=[]
    writers=SimpleNamespace(submit=lambda n,ticks,*,refl_field: frames.append(refl_field))
    _submit_tree_history_frame(writers,node,120)
    np.testing.assert_array_equal(cp.asnumpy(frames[0]).view(np.uint32),expected)
