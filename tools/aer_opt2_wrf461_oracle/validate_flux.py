"""GPU optics plus SW composition against WRF 4.6.1, 96 columns."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from tools.aer_opt2_wrf461_oracle.validate import read_columns,max_ulp


def validate():
    import cupy as cp
    from gpuwm.core import rrtmg_sw as sw
    from gpuwm.core.aer_opt2 import optics_device,INPUT_FIELDS
    tree=Path(__file__).resolve().parents[2]
    root=tree/'tools/aer_opt2_wrf461_oracle/fixtures'
    fixture=dict(np.load(root/'sw_flux.npz'))
    dump=dict(np.load(tree/'tools/rrtmg_wrf461_oracle/sw_fixtures/sw_tables.npz'))
    cuda=sw.CudaSW(sw.tables_from_dump(dump))
    fields={}; hashes=[]
    for case in read_columns(root/'columns.bin'):
        if case['scheme']!='rrtmg':continue
        c=f"a{case['column']:03d}"
        e=lambda k:fixture[c+'/entry/'+k]
        p,t,q,dz=[cp.asarray(v[None,:]) for v in case['inputs']]
        actual=optics_device(p,t,q,dz,cp.asarray(case['ht']),aer_type=case['aer_type'],
            options=case['options'],values=case['values'],
            inputs={key:cp.asarray(case['surface'][i:i+1]) for i,key in enumerate(INPUT_FIELDS)},nlayers=9)
        cols=('play','plev','tlay','tlev')
        gases=('h2ovmr','o3vmr','co2vmr','ch4vmr','n2ovmr','o2vmr')
        scalars=('asdir','asdif','aldir','aldif','coszen','adjes')
        mc=('cldfmcl','taucmcl','ssacmcl','asmcmcl','fsfcmcl','ciwpmcl','clwpmcl','cswpmcl')
        args=(1,int(e('nlay')),int(e('icld')),
              *[e(k)[None,:] for k in cols],np.asarray(e('tsfc')).reshape(1),
              *[e(k)[None,:] for k in gases],*[np.asarray(e(k)).reshape(1) for k in scalars],
              int(e('dyofyr')),np.asarray(e('scon')).reshape(1),
              int(e('inflgsw')),int(e('iceflgsw')),int(e('liqflgsw')),
              *[e(k)[:,None,:] for k in mc],
              *[e(k)[None,:] for k in ('reicmcl','relqmcl','resnmcl')])
        result=sw.sw_batched_to_host(cuda.rrtmg_sw_batched_device(*args,aer_opt=2,aerosol=(actual[0],actual[2],actual[1])))
        for key in sorted(k.split('/')[-1] for k in fixture if k.startswith(c+'/out/')):
            a=result[key][0];b=fixture[c+'/out/'+key]
            row=fields.setdefault(key,dict(max_ulp=0,max_abs=0.,words=0,worst_column=0))
            ulp=max_ulp(a,b)
            if ulp>row['max_ulp']:row.update(max_ulp=ulp,worst_column=case['column'])
            row['max_abs']=max(row['max_abs'],float(np.max(np.abs(a-b))))
            row['words']+=a.size
        # Guard retirement must preserve aer_opt=0 exactly, using the
        # caller-supplied neutral optics path as a separate negative control.
        if case['column'] in (0,31,63,95):
            zero=sw.sw_batched_to_host(cuda.rrtmg_sw_batched_device(*args,aer_opt=0))
            neutral=(cp.zeros_like(actual[0]),cp.zeros_like(actual[2]),cp.ones_like(actual[1]))
            explicit=sw.sw_batched_to_host(cuda.rrtmg_sw_batched_device(*args,aer_opt=2,aerosol=neutral))
            h=hashlib.sha256()
            for k in zero:
                assert np.array_equal(zero[k].view('u4'),explicit[k].view('u4')),(c,k)
                h.update(np.asarray(zero[k],'<f4').tobytes())
            hashes.append(dict(column=case['column'],sha256=h.hexdigest()))
    return dict(columns=96,fields=fields,neutral_controls=hashes)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    result=validate();Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    assert all(r['max_ulp']==0 for r in result['fields'].values()),'WRF SW composition mismatch'
