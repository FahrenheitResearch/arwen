"""Every recorded state and diagnostic word on the actual CUDA entry."""
from pathlib import Path
import json,subprocess,sys
import pytest
from conftest import requires_gpu


@pytest.mark.gpu
@requires_gpu
def test_shared_diffusion_reference_restores_before_its_first_allocation(tmp_path):
    import cupy as cp
    import numpy as np
    from test_nssl2_selector import _cfg, _initialized_gpu_mp18_state
    from gpuwm.io import restart
    cfg=_cfg(diff_opt=1)
    written,_=_initialized_gpu_mp18_state(cfg)
    reference=cp.arange(written.thp.size,dtype=cp.float32).reshape(written.thp.shape)
    written._scratch['diff1_theta_initial']=reference
    path=restart.write_restart(tmp_path/'shared-reference.npz',written,cfg)
    fresh,_=_initialized_gpu_mp18_state(cfg)
    assert 'diff1_theta_initial' not in fresh._scratch
    restart.restore_restart(path,fresh,cfg)
    np.testing.assert_array_equal(cp.asnumpy(fresh._scratch['diff1_theta_initial']),cp.asnumpy(reference))


@pytest.mark.gpu
@requires_gpu
def test_moisture_range_midpoint_rounds_to_the_even_word():
    import cupy as cp
    import numpy as np
    from gpuwm.core.kernels import module_source, module_options
    from gpuwm.cuda_subnormals import install
    install()
    source = module_source('urban_ucm480') + '''
extern "C" __global__ void urban480_sub_probe(unsigned int* out,const float* a,const float* b,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i==0) out[0]=__float_as_uint(u480_sub(__uint_as_float(0x3ee0c49cu),__uint_as_float(0x3d872b02u)));
    if(i<n) out[i+1]=__float_as_uint(u480_sub(a[i],b[i]));
}
'''
    module=cp.RawModule(code=source,options=module_options('urban_ucm480'))
    rng=np.random.default_rng(314159)
    bits=rng.integers(0,2**32,size=(2,16384),dtype=np.uint32)
    exceptional=(bits & np.uint32(0x7f800000))==np.uint32(0x7f800000)
    bits[exceptional] &= np.uint32(0xff7fffff)
    bits[:, :4]=np.asarray([[0,0x80000000,1,0x7f7fffff],
                           [0x80000000,0,0x80000001,0xff7fffff]],np.uint32)
    a,b=bits.view(np.float32)
    with np.errstate(over='ignore'):
        expected=(a-b).view(np.uint32)
    out=cp.zeros(len(a)+1,cp.uint32)
    module.get_function('urban480_sub_probe')(((len(a)+127)//128,),(128,),
        (out,cp.asarray(a),cp.asarray(b),np.int32(len(a))))
    result=out.get()
    assert int(result[0])==int(np.float32(.37300002574920654).view(np.uint32))
    np.testing.assert_array_equal(result[1:],expected)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize('table',('nlcd','lcz'))
@pytest.mark.parametrize('arm',('tt0','tt1','tree0'))
def test_full_release_column_words(tmp_path,table,arm):
    root=Path(__file__).resolve().parents[1]
    receipt=tmp_path/'grade.json'
    args=[sys.executable,str(root/'tools/urban_wrf480_oracle/grade_full.py'),'--gpu',
        '--listing',str(root/'gpuwm/core/kernels/urban_ucm480.cuh'),
        '--csv',str(root/f'tests/data/oracles/urban/ucm480/{table}-{arm}.csv.gz'),
        '--tables',str(root/'gpuwm/data/urban/wrf480'),'--out',str(receipt),
        '--lcz',str(int(table=='lcz')),'--tree',str(int(arm!='tree0')),
        '--tt',str(int(arm=='tt1')),'--fvg','0' if arm=='tree0' else '.2','--switch','groption=0']
    run=subprocess.run(args,text=True,capture_output=True)
    assert run.returncode==0,run.stdout+run.stderr
    result=json.loads(receipt.read_text())
    assert result['words']==result['rows']*77
    assert result['differing_words']==0
