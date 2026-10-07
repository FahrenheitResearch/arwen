"""Run and compare field-by-field poisoned and retained-column states."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from gpuwm.verify.chem_oracle import load,ulp_table

p=argparse.ArgumentParser(); p.add_argument('scratch'); p.add_argument('source_root')
a=p.parse_args(); root=Path(a.scratch).resolve(); report={}
for arm,build,name,exe in [('wrf','wrf4','plumerise_wrfchem','run_plumerise_wrfchem_trace'),('gsl','gsl4','plumerise_frp_gsl','gsl_run_plumerise_frp_trace')]:
    b=root/build; baseline=load(root/'fixtures'/name)
    fields=json.loads((root/(arm+'-poison-fields.json')).read_text())
    subprocess.run([sys.executable,str(root/'harness/build_plume_observed.py'),arm,a.source_root,str(b),str(root/(arm+'-harness')),'--experiment'],check=True)
    results={}; started=time.monotonic()
    for field in ['all','carry',*fields]:
        out=b/'state_experiment'/field
        try:
            run=subprocess.run([str(b/exe),str(out),field],cwd=b,capture_output=True,timeout=45)
        except subprocess.TimeoutExpired:
            results[field]={'timeout':True}; continue
        if run.returncode:
            results[field]={'returncode':run.returncode,'stderr':run.stderr.decode()[-400:]}; continue
        cases=load(out); different={}
        for case,data in cases.items():
            for key in ('ebu','ztopmax','steps','solver_k_min','solver_k_max','w','t','qv','qc','qh','qi','radius'):
                d=ulp_table(data[key],baseline[case][key])
                if d['n_nonzero']:
                    v=different.setdefault(key,dict(cases=0,max_ulp=0,words=0))
                    v['cases']+=1; v['max_ulp']=max(v['max_ulp'],d['max_ulp']); v['words']+=d['n_nonzero']
        results[field]=different
    report[arm]=dict(seconds=time.monotonic()-started,experiments=results)
    (root/'state-audit.json').write_text(json.dumps(report,indent=2))
    meaningful={k:v for k,v in results.items() if any(x in v for x in ('ebu','ztopmax','steps','solver_k_min','solver_k_max','timeout','returncode'))}
    print(arm,json.dumps(meaningful),flush=True)
