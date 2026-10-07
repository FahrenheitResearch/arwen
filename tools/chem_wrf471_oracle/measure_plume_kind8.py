"""Report KIND=8 differences in plume bounds, tops and diagnostic profiles."""
import argparse,json
from pathlib import Path
import numpy as np
from gpuwm.verify.chem_oracle import load,ulp_table
p=argparse.ArgumentParser(); p.add_argument('kind4'); p.add_argument('kind8'); p.add_argument('output'); a=p.parse_args()
x=load(a.kind4); y=load(a.kind8); stats={}; differences=[]
for case in x:
    for key in ('k_min','k_max','ztopmax','ebu','w','t','qv','qc','qh','qi','radius'):
        d=ulp_table(y[case][key],x[case][key]); r=stats.setdefault(key,dict(max_ulp=0,n_nonzero=0,n=0,max_absolute=0.))
        r['max_ulp']=max(r['max_ulp'],d['max_ulp']);r['n_nonzero']+=d['n_nonzero'];r['n']+=d['n']
        r['max_absolute']=max(r['max_absolute'],float(np.max(np.abs(np.asarray(y[case][key],dtype=np.float64)-x[case][key]))))
        if key in ('k_min','k_max','ztopmax') and d['n_nonzero']:
            differences.append(dict(case=case,output=key,kind4=np.asarray(x[case][key]).tolist(),kind8=np.asarray(y[case][key]).tolist()))
result=dict(stats=stats,differences=differences)
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(stats),flush=True)
