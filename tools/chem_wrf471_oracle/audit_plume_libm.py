"""Grade portable float32 transcendentals at every unique oracle argument."""
import argparse
import json
from pathlib import Path
import numpy as np
from gpuwm.verify.chem_plumerise_ref import fexp as expf,fpow as powf
from gpuwm.verify.chem_oracle import ulp_table

p=argparse.ArgumentParser(); p.add_argument('trace'); p.add_argument('output'); a=p.parse_args()
rows=np.fromfile(a.trace,dtype='<f4').reshape(-1,4)
rows=np.unique(rows.view(np.uint32),axis=0).view(np.float32)
totals={}; failures=[]
for operation,fn in [(1,expf),(2,powf)]:
    r=rows[rows[:,0]==operation]; actual=[]
    for op,x,y,expected in r:
        value=fn(x) if operation==1 else fn(x,y)
        actual.append(value)
        if np.asarray(value,dtype=np.float32).view(np.uint32)!=expected.view(np.uint32):
            failures.append(dict(operation=operation,x=int(x.view(np.uint32)),y=int(y.view(np.uint32)),expected=int(expected.view(np.uint32)),actual=int(np.asarray(value,dtype=np.float32).view(np.uint32))))
    totals['expf' if operation==1 else 'powf']=ulp_table(np.asarray(actual,dtype=np.float32),r[:,3])
report=dict(ulp=totals,failures=failures)
Path(a.output).write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)
if failures: raise SystemExit(1)
