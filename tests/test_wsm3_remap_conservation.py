"""WSM3's default remap keeps upper-cell mass and surface fallout.

Uses the delivered WSM6/WDM6 lane's pure transport control on WSM3's own
production subroutine. Strict verification intentionally carries stock WRF.
"""
import re
import numpy as np
import pytest
from conftest import requires_gpu

PROBE = r'''
extern "C" __global__ void remap_probe(int nz, float dt, float *out) {
    float q[80], w[80], dz[80], rho[80], t[80], denfac[80];
    for (int k=0; k<nz; ++k) {
        q[k]=0.001f; w[k]=1.0f; dz[k]=100.0f;
        rho[k]=1.0f; t[k]=260.0f; denfac[k]=1.0f;
    }
    float precip;
    wsm3_nislfv(nz,nullptr,rho,dz,t,denfac,w,q,&precip,dt,0,1);
    out[0]=precip;
    for (int k=0; k<nz; ++k) out[k+1]=q[k];
}
'''

def probe_module(reverted=False):
    import cupy as cp
    from gpuwm.core.kernels import module_source
    from gpuwm.wrf_exact import ENABLED
    if ENABLED:
        pytest.skip('strict mode retains stock WRF remap for the unmodified oracle')
    source = module_source('wsm3')
    if reverted:
        pattern = r'const int kt_start = kt;.*?kt = \(kt - 1 > kb\) \? kt - 1 : kb;'
        old = ('for (int kk = kt; kk < km; ++kk)\n'
               'if (zi[k + 1] <= za[kk]) { kt = kk; break; }\nkt = kt - 1;')
        source, count = re.subn(pattern, old, source, flags=re.S)
        assert count == 1
    return cp.RawModule(code='#define WSM3_KMAX 80\n'+source+PROBE,
                        options=('-std=c++17','--fmad=false'))

def inventory_error(module, nz, dt):
    import cupy as cp
    out = cp.zeros(nz+1, cp.float32)
    module.get_function('remap_probe')((1,), (1,),
        (np.int32(nz), np.float32(dt), out))
    out = cp.asnumpy(out)
    initial = float(np.float32(.001))*100*nz
    final = float(out[1:].sum(dtype=np.float64))*100+float(out[0])
    return abs(final-initial)/initial

@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize('nz',(2,50,80))
@pytest.mark.parametrize('dt',(.001,1.,20.))
def test_default_remap_closes_column_plus_surface(nz,dt):
    assert inventory_error(probe_module(),nz,dt) < 1.e-6

@pytest.mark.gpu
@requires_gpu
def test_planted_stock_search_is_caught():
    module=probe_module(reverted=True)
    for nz in (2,50,80):
        for dt in (.001,1.,20.):
            assert inventory_error(module,nz,dt) > .01
