"""Compare the real GPU producer with byte-unmodified WRF oracle columns."""
from pathlib import Path
import json
import argparse
import numpy as np


def read_columns(path):
    cases=[]
    with open(path,'rb') as f:
        def read(dtype,n):
            a=np.frombuffer(f.read(np.dtype(dtype).itemsize*n),dtype=dtype).copy()
            assert len(a)==n
            return a
        nc,nz=read('<i4',2)
        for scheme,nb in [('rrtmg',14),('goddard',11)]:
            for c in range(nc):
                typ,*opts=read('<i4',5)
                scalars=read('<f4',9)
                inputs=read('<f4',4*(nz+1)).reshape(4,nz+1)[:,:nz].copy()
                outputs=read('<f4',3*nb*(nz+1)).reshape(3,nb,nz+1)[:,:,:nz].copy()
                rh=read('<f4',nz+1)[:nz]
                diag=read('<f4',4)
                cases.append(dict(scheme=scheme,column=c,aer_type=int(typ),
                                  options=tuple(int(x) for x in opts),values=scalars[:4],
                                  ht=scalars[4:5],surface=scalars[5:],inputs=inputs,
                                  outputs=outputs,rh=rh,diag=diag))
        assert not f.read(1),'trailing oracle bytes'
    return cases


def max_ulp(a,b):
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
    ua=a.view(np.uint32).astype(np.int64);ub=b.view(np.uint32).astype(np.int64)
    ua=np.where(ua&0x80000000,0x80000000-(ua&0x7fffffff),ua+0x80000000)
    ub=np.where(ub&0x80000000,0x80000000-(ub&0x7fffffff),ub+0x80000000)
    nan=np.isnan(a)&np.isnan(b)
    return int(np.max(np.where(nan,0,np.abs(ua-ub))))


def validate(path):
    import cupy as cp
    from gpuwm.core.aer_opt2 import optics_device,INPUT_FIELDS
    result={}
    for case in read_columns(path):
        p,t,q,dz=[cp.asarray(v[None,:]) for v in case['inputs']]
        inputs={key:cp.asarray(case['surface'][i:i+1]) for i,key in enumerate(INPUT_FIELDS)}
        actual=optics_device(p,t,q,dz,cp.asarray(case['ht']),
                             aer_type=case['aer_type'],options=case['options'],
                             values=case['values'],inputs=inputs,nlayers=p.shape[1]+1,
                             scheme=case['scheme'])
        row=result.setdefault(case['scheme'],dict(columns=0,fields={}))
        row['columns']+=1
        expected=list(case['outputs'])+[case['outputs'][0,9,:],case['diag']]
        got=[cp.asnumpy(v)[0,:,:-1] for v in actual[:3]]+[cp.asnumpy(actual[3])[0],cp.asnumpy(actual[4])[0]]
        for name,a,b in zip(('tauaer','ssaaer','asyaer','band10_aod','surface_diagnostics'),got,expected):
            if case['scheme']=='goddard' and name=='band10_aod': continue
            ulp=max_ulp(a,b)
            maximum=float(np.max(np.abs(np.asarray(a)-b)))
            f=row['fields'].setdefault(name,dict(max_ulp=0,max_abs=0.,worst_column=0))
            if ulp>f['max_ulp']:
                f.update(max_ulp=ulp,worst_column=case['column'])
            f['max_abs']=max(f['max_abs'],maximum)
        for a,neutral in zip(actual[:3],(0.,1.,0.)):
            assert np.all(cp.asnumpy(a)[:,:,-1]==neutral)
    from gpuwm.certify.kernel_manifest import kernel_manifest
    result['kernel_manifest']=kernel_manifest()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('columns');p.add_argument('--out',required=True)
    args=p.parse_args()
    result=validate(args.columns)
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert all(f['max_ulp']==0 for k,r in result.items() if k!='kernel_manifest' for f in r['fields'].values()),'WRF aerosol column mismatch'
