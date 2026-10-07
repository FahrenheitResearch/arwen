"""Dense WRF saturation fixture, including table ULPs and pressure singularities."""
from pathlib import Path
import os
import numpy as np
import pytest
from gpuwm.verify.uwpbl_oracle import load
from gpuwm.verify.uwpbl_ref.wvsat import estblf,aqsat,fqsatd
from test_uwpbl_vdiff_wrf471_parity import words,gpu_module,ROOT

def fixture():
    stem = ROOT/'tests/data/oracles/uwpbl/run_stage_wvsat'
    if not stem.with_suffix('.bin').exists():
        directory = os.environ.get('UWPBL_ORACLE_DIR')
        if not directory:pytest.skip('saturation fixture absent and UWPBL_ORACLE_DIR unset')
        stem=Path(directory)/'run_stage_wvsat'
    if not stem.with_suffix('.bin').exists():pytest.skip(f'saturation fixture absent: {stem}')
    return load(stem)

def test_cpu_dense():
    fx=fixture();t=[None]+fx['t'].tolist();p=[None]+fx['p'].tolist();n=len(t)-1
    es=[None]+[0.0]*n;qs=es.copy()
    aqsat(t,p,es,qs,1,n)
    words(es[1:],fx['aqsat_es'],'CPU aqsat es');words(qs[1:],fx['aqsat_qs'],'CPU aqsat qs')
    values=[fqsatd(t[k],p[k]) for k in range(1,n+1)]
    assert np.array_equal([v[0] for v in values],fx['status'])
    for j,name in enumerate(['fqsatd_es','fqsatd_qs','fqsatd_gam'],1):words([v[j] for v in values],fx[name],'CPU '+name)
    words([estblf(x) for x in t[1:]],fx['estblf'],'CPU estblf')
    print(f'CPU saturation records={n}, outputs={n*7}, mismatches=0')

@pytest.mark.gpu
def test_gpu_dense():
    fx=fixture();n=len(fx['t'])
    kernel='''extern "C" __global__ void grade(double* t,double* p,double* out,int* status,int n){
     int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
     V tv{t+c,n},pv{p+c,n},es{out+n+c,n},qs{out+2*n+c,n};
     out[c]=uw_estblf(tv(1)).v;uw_aqsat(tv,pv,es,qs,1,1);
     R8 e,q,g;status[c]=uw_fqsatd(tv(1),pv(1),e,q,g);
     out[3*n+c]=e.v;out[4*n+c]=q.v;out[5*n+c]=g.v;
    }'''
    cp,mod=gpu_module(kernel)
    t=cp.array(fx['t']);p=cp.array(fx['p']);out=cp.empty((6,n),dtype=cp.float64);status=cp.empty(n,dtype=cp.int32)
    mod.get_function('grade')(((n+127)//128,),(128,),(t,p,out,status,np.int32(n)))
    values=out.get();assert np.array_equal(status.get(),fx['status'])
    for j,name in enumerate(['estblf','aqsat_es','aqsat_qs','fqsatd_es','fqsatd_qs','fqsatd_gam']):words(values[j],fx[name],'GPU '+name)
    print(f'GPU saturation records={n}, outputs={n*7}, mismatches=0')
