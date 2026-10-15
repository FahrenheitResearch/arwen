"""The public cal_cldfra2 to radiation solves must retain zero ULP."""
import json,os,subprocess
from pathlib import Path
import pytest
cp=pytest.importorskip('cupy')

@pytest.mark.parametrize('exact',['0','1'])
def test_public_coupled_radiation(exact,tmp_path,monkeypatch):
    root=os.environ.get('GPUWM_CLDFRA2_COUPLED_ROOT')
    if not root:
        pytest.skip('set GPUWM_CLDFRA2_COUPLED_ROOT to coupled_build.sh output')
    repo=Path(__file__).resolve().parents[1]
    receipt=tmp_path/'coupled.json'
    env=dict(os.environ,GPUWM_WRF_EXACT=exact)
    subprocess.run(['python',str(repo/'tools/cldfra2_wrf461_oracle/validate_coupled.py'),root,str(receipt)],check=True,env=env)
    measured=json.loads(receipt.read_text())
    assert measured['columns_lw']>=64 and measured['columns_sw']>=64
    assert set(measured['fields'])=={'CLDFRA_LW','CLDFRA_SW','GLW','OLR','SWDOWN','RTHRATEN_LW','RTHRATEN_SW'}
    assert all(f['max_ulp']==0 and f['mismatched_words']==0 for f in measured['fields'].values()), measured
