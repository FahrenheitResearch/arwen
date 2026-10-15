"""WRF module_convtrans_prep, including complete averaging windows."""
from pathlib import Path
import numpy as np
import pytest
cp=pytest.importorskip('cupy')
from gpuwm import wrf_exact
from gpuwm.core.g3 import run_g3_diag

@pytest.mark.parametrize('label', ['fixed','adaptive'])
def test_time_average_matches_wrf(label):
    with np.load(Path(__file__).parent/'data/oracles/g3_diag'/f'diag-{label}.npz') as z:
        f={k:z[k] for k in z.files}
    diag={}
    for n in range(len(f['dt'])):
        def volume(k): return cp.asarray(np.ascontiguousarray(f['in_'+k][n].transpose(1,0,2)))
        run_g3_diag(diag, cloud=volume('cloud'), ice=volume('ice'),
            rain=cp.asarray(f['in_rain'][n]), temperature=volume('t'), pressure=volume('p'),
            qv=volume('qv'), qc=volume('qc'), qi=volume('qi'), fqi=int(f['fqi']),
            dt=float(f['dt'][n]), curr_secs=float(f['curr'][n]),
            avg_minutes=float(f['avg']), adaptive=bool(f['adapt']))
        for key in sorted(k for k in f if k.startswith('out_')):
            field=key[4:]; got=diag[field].get(); want=f[key][n]
            if got.ndim==3: got=got.transpose(1,0,2)
            assert np.array_equal(got.view('u4'),want.view('u4')), (label,n,field)
    assert int(diag['clock'].get()[0]) == len(f['dt'])


def test_diagnostic_restart_preserves_the_next_full_window():
    with np.load(Path(__file__).parent/'data/oracles/g3_diag/diag-fixed.npz') as z:
        f={k:z[k] for k in z.files}
    continuous, resumed = {}, {}
    for n in range(len(f['dt'])):
        def volume(k): return cp.asarray(np.ascontiguousarray(f['in_'+k][n].transpose(1,0,2)))
        kwargs = dict(cloud=volume('cloud'),ice=volume('ice'),rain=cp.asarray(f['in_rain'][n]),
            temperature=volume('t'),pressure=volume('p'),qv=volume('qv'),qc=volume('qc'),qi=volume('qi'),
            fqi=int(f['fqi']),dt=float(f['dt'][n]),curr_secs=float(f['curr'][n]),avg_minutes=float(f['avg']))
        run_g3_diag(continuous,**kwargs)
        run_g3_diag(resumed,**kwargs)
        if n==4:
            # Round-trip every field through host float/int words at a mid-window restart.
            resumed={key:cp.asarray(value.get()) for key,value in resumed.items()}
        for key in continuous:
            assert np.array_equal(continuous[key].get().view('u4'),resumed[key].get().view('u4')), (n,key)
