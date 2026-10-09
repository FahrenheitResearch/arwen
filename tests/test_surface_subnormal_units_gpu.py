"""Portable default-on regressions pinned to fresh scalar Fortran hashes."""
import hashlib
from pathlib import Path
import numpy as np
import pytest
from conftest import requires_gpu

FIXTURE=Path(__file__).parent/'data/oracles/surface-subnormal'
pytestmark=pytest.mark.gpu
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def load(unit):
    with np.load(FIXTURE/(unit+'.npz')) as f:return {k:f[k] for k in f.files}
def inputs(f,prefix):return {k[len(prefix):]:v for k,v in f.items() if k.startswith(prefix)}

@requires_gpu
def test_mynn_first_step_preserves_positive_subnormal_qsfc():
    import cupy as cp
    from gpuwm.core.mynn_sfclay import seed_mynn_surface_first_step
    q=cp.asarray(np.array([1,2,0xfff,0x7fffff],np.uint32).view(np.float32).reshape(1,-1))
    arrays=[cp.zeros_like(q) for _ in range(4)]
    seed_mynn_surface_first_step(cp.ones_like(q),cp.zeros_like(q),q,ust=arrays[0],mol=arrays[1],qsfc=arrays[2],qstar=arrays[3])
    assert np.array_equal(cp.asnumpy(arrays[2]).view(np.uint32),cp.asnumpy(q).view(np.uint32))

@requires_gpu
def test_mynn_surface_subnormal_column_words():
    import cupy as cp
    from gpuwm.core import mynn_sfclay as m
    from tools.mynn_sfclay_wrf461_column_oracle.compare import STATIC_FIELDS
    f=load('mynn_surface');a=inputs(f,'in/');state=inputs(f,'state/');n=a['u1'].size
    v={k:cp.asarray(a[k].reshape(1,n)) for k in STATIC_FIELDS}
    for k in ('hfx','qfx','znt','qsfc','ust'):v[k]=cp.asarray(state[k].reshape(1,n))
    got=m.mynn_surface_layer(v,dx=3000.,itimestep=1,isfflx=1,isftcflx=0,mol=cp.asarray(state['mol'].reshape(1,n)),ustm=cp.asarray(state['ustm'].reshape(1,n)),variant='wrf_461')
    for k in inputs(f,'hash/'):
        assert sha(cp.asnumpy(getattr(got,k)).reshape(n))==str(f['hash/'+k]),k

@requires_gpu
def test_classic_subnormal_regime_and_humidity_words():
    from tools.sfclay_classic_wrf461_oracle import validate_sfclay_classic_oracle as c
    f=load('classic');a=inputs(f,'in/');state={k:a[k] for k in c.INOUT_FIELDS}
    got=c.woof_step(a,state,isfflx=1,isftcflx=0,iz0tlnd=1,dx=3000.)
    assert sha(got)==str(f['hash'])

@requires_gpu
def test_revised_subnormal_words():
    import _sfclayrev_oracle as r
    from gpuwm.core.physics_inventory import SFCLAY_OUTPUTS
    f=load('revised');a=inputs(f,'in/');n=a['u'].size
    oracle=r.Fixture(np.arange(n),tuple(str(i) for i in range(n)),a,np.zeros((6,3,n,len(SFCLAY_OUTPUTS)),np.float32))
    got=r.port_outputs(oracle)
    for j,k in enumerate(SFCLAY_OUTPUTS):
        v=got[...,j]
        if k=='lh':v=v[:5]
        assert sha(v)==str(f['hash/'+k]),k

@requires_gpu
def test_eta_subnormal_words():
    import cupy as cp
    from gpuwm.core import myjsfc as m
    from gpuwm.verify import myjsfc_oracle as e
    f=load('eta');a=inputs(f,'subnormal_probe/in/');n,nz,steps,it0=map(int,f['subnormal_probe/meta'])
    columns,surface=e._device_inputs(cp,a,m)
    state={k:cp.asarray(a[e._STATE_KEYS[k]].reshape(1,n)) for k in m.MYJ_SFCLAY_INOUT}
    hashes={k:hashlib.sha256() for k in inputs(f,'hash/')}
    for step in range(steps):
        outputs={k:cp.zeros((1,n),cp.float32) for k in m.MYJ_SFCLAY_OUTPUTS}
        m.launch_myj_sfclay(columns,surface,state,outputs,itimestep=it0+step)
        for k,v in {**state,**outputs}.items():hashes[k].update(cp.asnumpy(v).reshape(n).tobytes())
    for k,h in hashes.items():assert h.hexdigest()==str(f['hash/'+k]),k

@requires_gpu
def test_gsd41_dry_convection_preserves_qsq_production():
    import cupy as cp
    import _mynn_families_gsd41 as g
    from gpuwm.core.mynn_pbl_gpu import mynn_bl_driver_cuda
    f=load('mynn_pbl');state=inputs(f,'in/');sm=np.zeros_like(state['qv'])
    for step,init in enumerate(f['initflags']):
        v=g.driver_values(state,sm)
        got=mynn_bl_driver_cuda({k:cp.asarray(np.ascontiguousarray(a)) for k,a in v.items()},initflag=int(init),delt=f['delt'][()],flag_qs=False,bl_mynn_mixlength=2,bl_mynn_version='gsd_41',bl_mynn_cloud_tendency_form='gsd_41')
        got={k:cp.asnumpy(v) for k,v in got.items()};sm=got['sm'].copy()
        assert sha(got['qsq'])==str(f['qsq_hashes'][step]),step
        state=g.advance(state,got,f['delt'][()])
