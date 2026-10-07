"""Package native stream words without numerical field calculation."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('native',type=Path);p.add_argument('fixtures',type=Path);a=p.parse_args()
    a.fixtures.mkdir(parents=True,exist_ok=True)
    receipt={'schema':'sfire-initializer-atmos-native-v1','cases':{},
             'extraction':json.loads((a.native/'extraction-receipt.json').read_text()),
             'build':json.loads((a.native/'build-receipt.json').read_text()),
             'raw_sounding':{'psurf':1000.,'theta_surface':289.,'qv_surface':21.,
                 'height':[0.,100.,700.,1800.,4000.,7000.],'theta':[289.,289.,293.,300.,315.,340.],
                 'qv':[18.,12.,7.,3.,1.,0.],'u':[1.,2.,4.,7.,9.,12.],'v':[-4.,-2.,0.,1.,3.,4.]}}
    extra=a.native/'extra-build-receipt.json'
    if extra.is_file():receipt['extra_build']=json.loads(extra.read_text())
    for root in sorted((a.native/'output').iterdir()):
        if not root.is_dir():continue
        fields={};raw_sha={}
        for line in (root/'MANIFEST.txt').read_text().splitlines():
            name,kind,rank,*sizes=line.split();shape=tuple(map(int,sizes));rank=int(rank)
            path=root/(name+'.bin');raw=path.read_bytes();raw_sha[name]=hashlib.sha256(raw).hexdigest()
            data=np.frombuffer(raw,dtype='<'+kind).reshape(shape,order='F')
            if rank==2:data=data.T
            elif rank==3:data=data.transpose(1,2,0)
            elif rank>3:raise ValueError('unexpected native rank')
            fields[name]=np.ascontiguousarray(data).reshape(data.shape)
        target=a.fixtures/(root.name+'.npz');np.savez_compressed(target,**fields)
        receipt['cases'][root.name]={'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
             'native_stream_sha256':raw_sha,'field_shapes':{n:list(v.shape) for n,v in fields.items()}}
    (a.fixtures/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'cases':len(receipt['cases']),'bytes':sum(p.stat().st_size for p in a.fixtures.iterdir())}))


if __name__=='__main__':main()
