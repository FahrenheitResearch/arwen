"""Public Fortran ULP oracle and threshold fault control on the card."""
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
cp = pytest.importorskip('cupy')
from gpuwm.core import rrtmgp

@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    root = os.environ.get('WRF_SOURCE_ROOT')
    if not root:
        pytest.skip('WRF_SOURCE_ROOT is required for the public Fortran oracle')
    out = tmp_path_factory.mktemp('cldfra2-oracle')
    here = Path(__file__).resolve().parents[1]
    subprocess.run(['bash', str(here/'tools/cldfra2_wrf461_oracle/build.sh'), root, str(out)], check=True)
    with (out/'columns.bin').open('rb') as f:
        nc,nk = np.fromfile(f, '<i4', count=2)
        records = []
        for _ in range(8):
            fqc,fqi,driver = np.fromfile(f, '<i4', count=3)
            qc,qi,cf = [np.fromfile(f, '<f4', count=nc*nk).reshape((nc,nk), order='F') for _ in range(3)]
            records.append((bool(fqc), bool(fqi), driver, qc, qi, cf))
    return records

@pytest.mark.parametrize('exact', ['0','1'])
def test_public_column_oracle(oracle, monkeypatch, exact):
    monkeypatch.setenv('GPUWM_WRF_EXACT', exact)
    for fqc,fqi,driver,qc,qi,want in oracle:
        got = rrtmgp.cal_cldfra2(cp.asarray(qc),cp.asarray(qi),f_qc=fqc,f_qi=fqi)
        np.testing.assert_array_equal(cp.asnumpy(got).view('u4'),want.view('u4'))
    seed = cp.full((64,4), np.float32(.375))
    got = rrtmgp.cal_cldfra2(seed, seed, f_qc=False,f_qi=False,out=seed)
    assert got is seed
    np.testing.assert_array_equal(cp.asnumpy(got),np.full((64,4),np.float32(.375)))

def test_threshold_bit_mutation_is_caught(oracle, monkeypatch):
    from gpuwm.core import rrtmg_legacy_device as dev
    from gpuwm import nvrtc_ptx_cache as compiler
    source = (Path(dev.__file__).parent/'kernels/rrtmg_legacy_prep.cu').read_text()
    assert 'condensate > 1.0e-6f' in source
    original = np.float32(1.e-6).view(np.uint32)
    changed = np.asarray(original ^ np.uint32(1), dtype=np.uint32).view(np.float32)
    assert int(original ^ changed.view(np.uint32)) == 1
    source = source.replace('condensate > 1.0e-6f', f'condensate > {float(changed):.18e}f')
    ptx,_ = compiler.compile_using_nvrtc(source, ('-std=c++17','--ftz=false'), None, 'threshold_fault.cu')
    module=cp.cuda.function.Module(); module.load(ptx.encode() if isinstance(ptx,str) else ptx)
    monkeypatch.setattr(dev,'_gpu_module',lambda *args: module)
    misses=0
    for fqc,fqi,driver,qc,qi,want in oracle:
        got=cp.asnumpy(rrtmgp.cal_cldfra2(cp.asarray(qc),cp.asarray(qi),f_qc=fqc,f_qi=fqi))
        misses += np.count_nonzero(got.view('u4') != want.view('u4'))
    assert misses > 0, 'one-bit threshold mutation escaped the public oracle'
