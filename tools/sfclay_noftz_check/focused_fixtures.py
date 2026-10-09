"""Reduce fresh Fortran streams to small portable regression fixtures.

Retain inputs and expected SHA-256 values only. No raw output stream enters
the portable fixture. Full-field comparisons remain in the box receipts.
"""
import hashlib
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
L=Path(os.environ.get('SFCLAY_CHECK_ROOT','/work/pool-sfclay-noftz-mynn-mm5-eta'))
out=L/'portable';out.mkdir(exist_ok=True)
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
from tools.mynn_sfclay_wrf461_column_oracle import compare as m
m.CONFIGS = m.CONFIGS + tuple((f'new_{i}_{spp}',i,1,dx,first,seeded,12,spp)
    for i,dx,first,seeded in [(0,1.0,1,True),(1,1e6,2000000,False),(2,250.0,1,True),(3,1e7,100,False)] for spp in (0,1))
arrays=m.read_columns(L/'subnormal/columns.bin')
_,records=m.read_oracle(L/'subnormal/oracle.bin')
cfg,step,it,state,want=records[0]
fixture={'in/'+k:v for k,v in arrays.items()}
fixture.update({'state/'+k:v for k,v in state.items()})
fixture.update({'hash/'+k:np.array(sha(v)) for k,v in want.items()})
np.savez_compressed(out/'mynn_surface.npz',**fixture)

from tools.sfclay_classic_wrf461_oracle import validate_sfclay_classic_oracle as c
groups,_=c.load_columns(L/'oracles/classic/fringe-columns.txt')
_,refs=c.load_wrf(L/'oracles/classic/fringe-corrected.txt')
fixture={'in/'+k:v for k,v in groups[0][-1].items()}
fixture['hash']=np.array(sha(refs[(1,1)]))
np.savez_compressed(out/'classic.npz',**fixture)

import _sfclayrev_oracle as r
f=r.load_fixture(L/'oracles/revised-subnormal')
fixture={'in/'+k:v for k,v in f.inputs.items()}
from gpuwm.core.physics_inventory import SFCLAY_OUTPUTS
for j,k in enumerate(SFCLAY_OUTPUTS):
    a=f.outputs[...,j]
    if k=='lh':a=a[:5]
    fixture['hash/'+k]=np.array(sha(a))
np.savez_compressed(out/'revised.npz',**fixture)

from gpuwm.verify import myjsfc_oracle as e
f=e.load(L/'extras/extras.npz')
fixture={k:v for k,v in f.items() if k=='fields' or k.startswith('tables/') or k.startswith('subnormal_probe/')}
expected=fixture.pop('subnormal_probe/out')
for j,k in enumerate(fixture['fields']): fixture['hash/'+str(k)]=np.array(sha(expected[:,j]))
np.savez_compressed(out/'eta.npz',**fixture)

import _mynn_columns_gsd41 as p
oracle=p.load(path=L/'extra/dt-20/sq.bin.gz')
initial=oracle['steps'][0][1]
fixture={'in/'+k:v for k,v in initial.items()}
fixture['delt']=np.array(oracle['delt'])
fixture['initflags']=np.array([i for i,_,_ in oracle['steps']],np.int32)
fixture['qsq_hashes']=np.array([sha(v['qsq']) for _,_,v in oracle['steps']])
np.savez_compressed(out/'mynn_pbl.npz',**fixture)
