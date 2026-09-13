"""Actual WDM6 count arithmetic and launcher rejection, without a long loop."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


@pytest.fixture(scope='module')
def counts(tmp_path_factory):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('the exact CPU count helper requires a C++ compiler')
    source = Path(__file__).resolve().parents[1]/'gpuwm/core/kernels/wdm6.cu'
    directory = tmp_path_factory.mktemp('wdm6-counts')
    wrapper = directory/'counts.cpp'
    wrapper.write_text('#define WDM6_CPU_MIRROR\n#include "'+source.as_posix()+'"\n'+r'''
extern "C" int count(float value) { return wdm6_checked_steps(value); }
extern "C" int rain(float rho,float dz,float dt) { return wdm6_rain_steps(rho,dz,dt); }
extern "C" unsigned flags(const float*rho,const float*dz,float dt,int nz,int stride,int col) {
    int loops=wdm6_checked_steps(dt/120.0f+0.5f);
    if(loops<0) return 2u;
    float minor=dt<=120.0f?dt:dt/(float)loops;
    unsigned result=0u;
    for(int k=0;k<nz;++k) {
        int i=k*stride+col;
        if(wdm6_rain_steps(rho[i],dz[i],minor)<0) result|=2u;
    }
    return result;
}
extern "C" int prior(float rho,float dz,float dt) {
    float df=sqrtf(1.28f/rho);
    int n=(int)floorf(2.5f*2.99849272e3f*powf(1.0e-3f,0.8f)*df/dz*dt+1.0f);
    return n<1?1:n;
}
''')
    library = directory/'counts.so'
    subprocess.run([compiler, '-std=c++17', '-O2', '-shared', '-fPIC',
                    '-ffp-contract=off', str(wrapper), '-o', str(library)],
                   check=True, capture_output=True, text=True)
    native = ctypes.CDLL(str(library))
    native.count.argtypes = [ctypes.c_float]
    native.count.restype = ctypes.c_int
    for name in ('rain', 'prior'):
        getattr(native,name).argtypes = [ctypes.c_float]*3
        getattr(native,name).restype = ctypes.c_int
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    native.flags.argtypes = [array,array,ctypes.c_float,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    native.flags.restype = ctypes.c_uint
    return native


def fields(xp, shape=(2,1,2)):
    values = {name:xp.full(shape, value, dtype=xp.float32) for name,value in dict(
        theta=285.,qv=.006,qc=.0001,qr=.001,qi=0.,qs=0.,qg=0.,nn=5e8,nc=1e8,nr=1000.,
        rho=1.,pii=1.,pressure=90000.,dz=100.,effc=2.49,effi=4.99,effs=9.99).items()}
    values.update({name:xp.full(shape[1:],value,dtype=xp.float32) for name,value in dict(
        xland=1.,rainnc=3.,rainncv=4.,snownc=5.,snowncv=6.,graupelnc=7.,graupelncv=8.,sr=.2).items()})
    return values


def launch(values, dt=60., **kwargs):
    from gpuwm.core.wdm6 import launch_wdm6
    positional = ('theta','qv','qc','qr','qi','qs','qg','nn','nc','nr','rho','pii','pressure',
                  'dz','xland','rainnc','rainncv','snownc','snowncv','graupelnc','graupelncv','sr')
    return launch_wdm6(*(values[name] for name in positional),dt,
                       **{name:values[name] for name in ('effc','effi','effs')}, **kwargs)


@pytest.mark.parametrize('value,expected', [(0.,1),(1.,1),(3.9,3),(2147483520.,2147483520),
    (2147483648.,-1),(float('inf'),-1),(float('nan'),-1),(-1.,-1)])
def test_checked_conversion_at_actual_signed_limit(counts, value, expected):
    assert counts.count(value) == expected


def test_normal_counts_equal_the_prior_expression(counts):
    for rho in (.1,.5,1.,1.5):
        for dz in (1.,10.,100.,1000.):
            for dt in (.001,1.,60.,120.):
                assert counts.rain(rho,dz,dt) == counts.prior(rho,dz,dt)


def test_exact_returned_witness_is_unrepresentable_without_running_its_loop(counts):
    assert counts.rain(1.,9e-7,60.) == -1
    assert counts.rain(1.,1e-6,60.) > 0
    rho=np.ones((2,1,1),np.float32)
    dz=np.array([9e-7,100.],np.float32).reshape(2,1,1)
    assert counts.flags(rho,dz,60.,2,1,0) == 2


def test_launcher_reports_actual_count_failure(counts, monkeypatch):
    from gpuwm.core import wdm6
    monkeypatch.setattr(wdm6,'cp',np)
    values=fields(np)
    values['dz'][0,0,1]=np.float32(9e-7)
    before={name:value.copy() for name,value in values.items()}
    def kernel(name,symbol):
        def run(grid,block,args):
            assert symbol == 'wdm6_column'
            rho,dz,dt,nz,ny,nx,status=args[10],args[13],args[25],args[27],args[28],args[29],args[30]
            stride=ny*nx
            for col in range(stride):
                status[0] |= counts.flags(rho,dz,dt,nz,stride,col)
        return run
    monkeypatch.setattr(wdm6,'get_kernel',kernel)
    with pytest.raises(FloatingPointError,match='signed 32-bit counter'):
        launch(values)
    for name,value in values.items():
        np.testing.assert_array_equal(value,before[name])


@pytest.mark.parametrize('dt', [1e100,1e-100])
def test_unrepresentable_device_interval_stops_before_kernel(dt, monkeypatch):
    from gpuwm.core import wdm6
    monkeypatch.setattr(wdm6,'get_kernel',lambda *a:pytest.fail('invalid interval reached a kernel'))
    with pytest.raises(ValueError,match='float32'):
        launch(fields(np),dt=dt)


def test_count_status_is_priced_and_owned_by_the_scratch_registry():
    from gpuwm.config import RunConfig
    from gpuwm.core import preflight
    cfg=RunConfig(nx=8,ny=6,nz=4,dx=1000.,dy=1000.,ztop=10000.,
                  dt=1.,run_seconds=10.,moist=True,moist_cq=True,mp_physics=16)
    assert preflight.scratch_slot_registry(cfg)['wdm6_count_status'] == (1,)
    assert preflight.scratch_slot_lifetime('wdm6_count_status').kind == 'write_before_read'
