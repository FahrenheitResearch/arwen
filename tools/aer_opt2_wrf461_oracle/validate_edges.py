"""Zero AOD behavior and the concrete subnormal-numerator refusal."""
from pathlib import Path
import numpy as np
from tools.aer_opt2_wrf461_oracle.validate import read_columns,max_ulp


def validate():
    import cupy as cp
    from gpuwm.core.aer_opt2 import optics_device,INPUT_FIELDS
    root=Path(__file__).resolve().parent/'fixtures'
    divergences=0
    for case in read_columns(root/'zero-columns.bin'):
        p,t,q,dz=[cp.asarray(v[None,:]) for v in case['inputs']]
        actual=optics_device(p,t,q,dz,cp.asarray(case['ht']),aer_type=case['aer_type'],
            options=case['options'],values=case['values'],
            inputs={key:cp.asarray(case['surface'][i:i+1]) for i,key in enumerate(INPUT_FIELDS)},
            scheme=case['scheme'])
        for a,b in zip(actual[:3],case['outputs']):assert max_ulp(cp.asnumpy(a)[0],b)==0
        if case["scheme"]=="rrtmg":
            assert max_ulp(cp.asnumpy(actual[3])[0],case["outputs"][0,9,:])==0
        diag=cp.asnumpy(actual[4])[0]
        if case['options'][1]==3:
            assert np.isnan(case['diag'][1]) and diag[1]==0
            divergences+=1
        else:assert max_ulp(diag,case['diag'])==0
    # The valid WRF value below produces a subnormal numerator. The SW
    # hardware division has no qualified gradual-underflow arm for it.
    p=cp.full((1,2),95000,cp.float32);t=cp.full((1,2),295,cp.float32)
    q=cp.zeros_like(p);dz=cp.full_like(p,300)
    try:
        optics_device(p,t,q,dz,cp.zeros(1,cp.float32),values=(1e-38,1.3,0.85,0.9))
    except ValueError as e:
        assert 'subnormal aerosol asymmetry numerator' in str(e)
    else:raise AssertionError('subnormal numerator control was not refused')
    return dict(zero_aod_columns=192,zero_optics_max_ulp=0,
                declared_zero_aod_angstrom_diagnostics=divergences,
                wrf_angstrom='NaN from log(0/0)',engine_angstrom=0,
                subnormal_numerator_control='refused before SW')


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    result=validate();Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
