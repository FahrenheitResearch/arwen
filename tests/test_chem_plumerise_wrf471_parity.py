"""WRF-Chem full-solver parity, with an explicit CPU representative subset.

All 108 cases are graded by audit_plume_cpu.py on the CPU node. The ordinary
NumPy subset covers a mixed-group column and a windy single-pass column on
both vertical grids; no optional JIT dependency is required by these tests.
"""
import hashlib
from pathlib import Path
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.verify.chem_oracle import load,ulp_table,ORACLE_ROOT
from gpuwm.verify.chem_plumerise_ref import reference_column

ROOT=ORACLE_ROOT/'smoke/plumerise_wrfchem'
CPU_CASES=('nz49_s1_f05','nz59_s4_f04')


def assert_words(actual,expected):
    for name,value in actual.items():
        value=np.asarray(value); ref=np.asarray(expected[name])
        assert ulp_table(value,ref)==dict(max_ulp=0,n_nonzero=0,n=value.size),name
        assert np.array_equal(value.view(np.uint32),ref.view(np.uint32)),name


def test_cpu_solver_parity():
    cases=load(ROOT)
    assert len(cases)==108
    for name in CPU_CASES: assert_words(reference_column(cases[name],'landuse'),cases[name])


def test_zero_driver_gates():
    cases=load(ROOT)
    for name in ('nz49_s1_f07','nz49_s1_f08','nz49_s1_f09'):
        assert_words(reference_column(cases[name],'landuse'),cases[name])


def test_fixture_pins():
    assert hashlib.sha256((ROOT/'oracle-sha256sums.txt').read_bytes()).hexdigest()=='84c100995cb689853bfa9a5ed156b72d771e4013893ada0d65023d9fe8c85db6'
    for line in (ROOT/'oracle-sha256sums.txt').read_text().splitlines():
        digest,name=line.split(None,1)
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest


def _device_case(case,cp):
    met={k:cp.asarray(case[k][:,None,None]) for k in
         ('t_phy','p_phy','rho','u_phy','v_phy','z','z_at_w')}
    met['qv']=cp.asarray(case['qv_in'][:,None,None])
    emitted=cp.asarray(case['ebu_in'][:,None,None])
    return cp,met,emitted


@pytest.mark.gpu
@requires_gpu
def test_kernel_all_cases():
    cp=pytest.importorskip('cupy')
    from gpuwm.core.chem_plumerise import launch
    for name,case in load(ROOT).items():
        cp,met,emitted=_device_case(case,cp)
        result=launch(met,emitted,arm='landuse',mean_fct=cp.asarray(case['mean_fct'][:,None,None]),firesize=cp.asarray(case['firesize'][:,None,None]),audit=True)
        if not result['columns'].size:
            assert name.endswith(('f07','f08','f09')); continue
        actual=dict(ebu=cp.asnumpy(result['ebu'][0]).T,
                    ztopmax=cp.asnumpy(result['diagnostics'][0,1:9]).reshape(4,2).T,
                    steps=cp.asnumpy(result['steps'][0]).reshape(4,2).T)
        for f,key in enumerate(('w','t','qv','qc','qh','qi','radius')):
            actual[key]=cp.asnumpy(result['profiles'][0,f])
        assert_words(actual,case)


@pytest.mark.gpu
@requires_gpu
def test_nonfire_gather_scatter_and_cached_landuse():
    cp=pytest.importorskip('cupy')
    from types import SimpleNamespace
    from gpuwm.core.chem_plumerise import launch,scatter,apply_cached
    case=load(ROOT)['nz49_s1_f05']; cp,met,emitted=_device_case(case,cp)
    met={k:cp.ascontiguousarray(cp.repeat(v,3,axis=2)) for k,v in met.items()}
    emitted=cp.ascontiguousarray(cp.repeat(emitted,3,axis=2))
    mean=cp.zeros((4,1,3),dtype=cp.float32); area=mean.copy()
    mean[:,:,1]=cp.asarray(case['mean_fct'][:,None]); area[:,:,1]=cp.asarray(case['firesize'][:,None])
    result=launch(met,emitted,arm='landuse',mean_fct=mean,firesize=area,audit=True)
    assert np.array_equal(cp.asnumpy(result['columns']),[1])
    bounds=[cp.full((1,3),-7,dtype=cp.int32) for _ in range(2)]
    frac=cp.full((1,3),-7,dtype=cp.float32); top=frac.copy()
    profile=cp.full((49,1,3),-7,dtype=cp.float32)
    scatter(result,*bounds,frac,top,profile)
    for a in [*bounds,frac,top,profile]:
        assert np.all(cp.asnumpy(a)[...,0]==-7); assert np.all(cp.asnumpy(a)[...,2]==-7)
    cache=cp.zeros((49,1,3),dtype=cp.float32); cache[:16,:,1]=result['groups'].T
    ctx=SimpleNamespace(cfg=SimpleNamespace(plume_fire_properties='landuse'),diag={'plume_group_cache':cache},met=lambda name:met[name])
    out=cp.full((3,49,1,3),-7,dtype=cp.float32)
    apply_cached(ctx,emitted,out,result['columns'])
    assert_words({'ebu':cp.asnumpy(out[:,:,0,1]).T},case)
    assert np.all(cp.asnumpy(out)[...,0]==-7); assert np.all(cp.asnumpy(out)[...,2]==-7)
