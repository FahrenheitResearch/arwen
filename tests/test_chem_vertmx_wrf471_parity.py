"""WRF default vertmx parity, including its frozen top-level writeback."""
from types import SimpleNamespace
import numpy as np
import pytest
from gpuwm.core.chem_vertmx import launch_vertmx,step,init,ALLOCATES
from gpuwm.verify.chem_vertmx_ref import vertmx_reference, _solve
from gpuwm.verify.chem_oracle import ORACLE_ROOT,load,ulp_table

CASES=load(ORACLE_ROOT/'core'/'vertmx')


def _inputs(a,r):
    p=a['indices'];anth=a['anth']
    def v(key):return np.ascontiguousarray(a[key][:,None,None])
    args=(np.ascontiguousarray(a['input'][:,r,None,None]),v('alt'),v('z_at_w'),
          v('z'),v('dz8w'),v('exch_h'),np.full((1,1),a['vd'][r],dtype=np.float32),a['dt'])
    def plane(x):return np.full((1,1),x,dtype=np.float32)
    kw=dict(phase=('gas','aerosol')[r],
            anth_co_kts=plane(anth[p[0]-1]) if p[0]>=2 else None,
            fire_co_k1=plane(a['fire']) if p[3]>=2 else None,
            anth_pm25_pair=(plane(anth[2]),plane(anth[3])) if p[1]>2 else None,
            anth_pm25=plane(anth[p[2]-1]) if p[2]>2 else None,
            sf_urban_physics=int(a['urban']))
    return args,kw


def _tile(args,kw):
    args=tuple(np.tile(x,(1,2,3)) if isinstance(x,np.ndarray) and x.ndim==3 else
               np.tile(x,(2,3)) if isinstance(x,np.ndarray) else x for x in args)
    kw={k:(tuple(np.tile(x,(2,3)) for x in v) if isinstance(v,tuple) else
           np.tile(v,(2,3)) if isinstance(v,np.ndarray) else v) for k,v in kw.items()}
    return args,kw


def _assert_words(actual,expected):
    expected=np.broadcast_to(expected,actual.shape).copy()
    measured=ulp_table(actual,expected)
    assert measured=={'max_ulp':0,'n_nonzero':0,'n':expected.size},measured
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))


def _identity(out,a,r):
    _assert_words(out['output'],a['output'][:,r,None,None])
    _assert_words(out['ekmfull'],a['ekmfull'][:,None,None])
    _assert_words(out['ddmassn'],a['ddmassn'][r:r+1].reshape(1,1))
    if r==1:_assert_words(out['mixed'],a['mixed'][:,None,None])
    _assert_words(np.float32(a['accum_in'][r]+out['ddmassn']),a['accum'][r:r+1].reshape(1,1))


@pytest.mark.parametrize('case',CASES)
@pytest.mark.parametrize('tiled',(False,True))
def test_cpu_oracle(case,tiled):
    a=CASES[case]
    for r in range(2):
        args,kw=_inputs(a,r)
        if tiled:args,kw=_tile(args,kw)
        _identity(vertmx_reference(*args,**kw),a,r)


@pytest.mark.gpu
@pytest.mark.parametrize('case',CASES)
@pytest.mark.parametrize('tiled',(False,True))
def test_gpu_oracle(case,tiled):
    import cupy as cp
    a=CASES[case]
    for r in range(2):
        args,kw=_inputs(a,r)
        if tiled:args,kw=_tile(args,kw)
        device=[cp.asarray(v) for v in args[:-1]]+[args[-1]]
        kw={k:(tuple(cp.asarray(x) for x in v) if isinstance(v,tuple) else
               cp.asarray(v) if isinstance(v,np.ndarray) else v) for k,v in kw.items()}
        mixed=cp.empty_like(device[0]);ek=cp.empty_like(device[2]);dd=cp.empty(args[0].shape[1:],dtype=cp.float32)
        accum=cp.full(args[0].shape[1:],a['accum_in'][r],dtype=cp.float32)
        launch_vertmx(*device,accum,**kw,mixed=mixed,ekmfull=ek,ddmassn=dd)
        _identity({'output':cp.asnumpy(device[0]),'mixed':cp.asnumpy(mixed),
                   'ekmfull':cp.asnumpy(ek),'ddmassn':cp.asnumpy(dd)},a,r)
        _assert_words(cp.asnumpy(accum),a['accum'][r:r+1].reshape(1,1))


def _mass(q,rho,dz):
    # Sum in FP64 only to measure FP32 solver error, not to compute the port.
    return np.sum(q.astype(np.float64)*rho.astype(np.float64)*dz.astype(np.float64))


def test_cpu_conservation_and_wrf_top_writeback():
    a=CASES['family4_nz49_step1'];args,kw=_inputs(a,1)
    args=list(args);args[5]=np.full_like(args[5],100);args[6]=np.zeros((1,1),dtype=np.float32)
    q,alt,zw,z,dz,exch,vd,dt=args
    rho=np.float32(1)/alt
    solved=_solve(q[:,0,0],np.r_[np.float32(0),exch[1:,0,0],np.float32(0)],
                  rho[:,0,0],zw[:,0,0]-zw[0,0,0],z[:,0,0]-zw[0,0,0],dt,np.float32(0))
    before=_mass(q,rho,dz)
    assert abs(_mass(solved[:,None,None],rho,dz)-before)/before < 8*np.finfo(np.float32).eps
    result=vertmx_reference(*args,**kw)
    defect=abs(_mass(result['output'],rho,dz)-before)/before
    assert defect > 100*np.finfo(np.float32).eps
    assert result['output'][-1,0,0]==q[-1,0,0]
    # Conservation of the WRF writeback requires no exchange into frozen top.
    args[5][-1]=0
    result=vertmx_reference(*args,**kw)
    assert abs(_mass(result['output'],rho,dz)-before)/before < 8*np.finfo(np.float32).eps


def test_process_rows_gates_and_accumulation():
    a=CASES['family3_nz49_step1'];args,kw=_inputs(a,1)
    row=SimpleNamespace(state_attr='chem_arbitrary_row',phase='aerosol',floor=1e-16)
    class Table:
        def rows_for(self,key):return (row,)
    class Context:
        cfg=SimpleNamespace(vertmix_onoff=1,mynn_chem_vertmx=False,sf_urban_physics=0)
        table=Table()
        physics_fields={'exch_h':args[5]}
        diag={'vertmx_deposited':np.full((1,1,1),2,dtype=np.float32)}
        ddvel=args[6][None,...]
        state=SimpleNamespace(chem_arbitrary_row=args[0].copy())
        def field(self,row):return getattr(self.state,row.state_attr)
        def row_index(self,row):return 0
        def met(self,name):return dict(alt=args[1],z_at_w=args[2],z=args[3],dz8w=args[4],exch_h=args[5])[name]
    ctx=Context();initial=ctx.field(row).copy()
    assert ALLOCATES[0].shape=='rows_2d' and ALLOCATES[0].restart=='serialize'
    init(ctx);assert ctx.diag['vertmx_deposited'][0,0,0]==2
    for gate in ('ktau','vertmix_onoff','mynn_chem_vertmx','num_vert_mix'):
        ctx.cfg.vertmix_onoff=0 if gate=='vertmix_onoff' else 1
        ctx.cfg.mynn_chem_vertmx=gate=='mynn_chem_vertmx'
        ctx.physics_fields['num_vert_mix']=int(gate=='num_vert_mix')
        step(ctx,args[-1],2 if gate=='ktau' else 3)
        np.testing.assert_array_equal(ctx.field(row),initial)
    ctx.cfg.mynn_chem_vertmx=False;ctx.physics_fields['num_vert_mix']=0
    with pytest.raises(TypeError, match="host pointers"):
        step(ctx,args[-1],3)
    np.testing.assert_array_equal(ctx.field(row),initial)
    assert ctx.diag['vertmx_deposited'][0,0,0]==2


@pytest.mark.gpu
def test_gpu_process_step_accumulates_native_deposition():
    import cupy as cp

    a=CASES['family3_nz49_step1'];args,_kw=_inputs(a,1)
    row=SimpleNamespace(state_attr='chem_arbitrary_row',phase='aerosol',floor=1e-16)
    table=SimpleNamespace(rows_for=lambda key: (row,))
    met=dict(zip(('alt','z_at_w','z','dz8w','exch_h'),args[1:6]))
    state=SimpleNamespace(chem_arbitrary_row=cp.asarray(args[0]))
    ctx=SimpleNamespace(
        cfg=SimpleNamespace(vertmix_onoff=1,mynn_chem_vertmx=False,sf_urban_physics=0),
        table=table,physics_fields={},
        diag={'vertmx_deposited':cp.full((1,1,1),2,dtype=cp.float32)},
        ddvel=cp.asarray(args[6][None,...]),
        field=lambda row: getattr(state,row.state_attr),
        row_index=lambda row: 0,met=lambda name: cp.asarray(met[name]))
    accum=np.float32(2)
    for ktau in range(3,8):
        expected=vertmx_reference(cp.asnumpy(ctx.field(row)),*args[1:],phase=row.phase)
        accum=np.float32(accum+expected['ddmassn'][0,0])
        step(ctx,args[-1],ktau)
        _assert_words(cp.asnumpy(ctx.field(row)),expected['output'])
        assert float(ctx.diag['vertmx_deposited'][0,0,0])==accum
