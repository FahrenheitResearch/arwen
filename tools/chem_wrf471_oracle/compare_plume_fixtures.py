"""Compare two fixture trees word for word, including integer diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
from gpuwm.verify.chem_oracle import load,ulp_table
p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');p.add_argument('output');a=p.parse_args()
x=load(a.reference);y=load(a.candidate);failures=[]
if set(x)!=set(y):raise RuntimeError('fixture case sets differ: the comparison would omit solver cases')
for case in x:
    if set(x[case])!=set(y[case]):raise RuntimeError('fixture fields differ: diagnostics would not be compared')
    for key in x[case]:
        left=np.asarray(x[case][key]);right=np.asarray(y[case][key])
        if not np.array_equal(left.view(np.uint32),right.view(np.uint32)):
            failures.append(dict(case=case,field=key,ulp=ulp_table(right,left)))
Path(a.output).write_text(json.dumps(dict(cases=len(x),failures=failures),indent=2));print(len(x),'cases',len(failures),'different fields')
if failures:raise SystemExit(1)
