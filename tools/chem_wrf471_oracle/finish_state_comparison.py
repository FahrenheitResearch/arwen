"""Record forward/reverse and index-isolated CVI experiments."""
import argparse,json
from pathlib import Path
from gpuwm.verify.chem_oracle import load,ulp_table
p=argparse.ArgumentParser();p.add_argument('scratch');a=p.parse_args();root=Path(a.scratch)
report=json.loads((root/'state-audit.json').read_text())
for arm,build,name in [('wrf','wrf4','plumerise_wrfchem'),('gsl','gsl4','plumerise_frp_gsl')]:
    baseline=load(root/'fixtures'/name)
    carried=load(root/build/'state_experiment/carry')
    for field in (['cvi_1','cvi_tail','carry_reverse'] if arm=='wrf' else ['carry_reverse']):
        data=load(root/build/'state_experiment'/field); totals={}
        for case,c in data.items():
            for k in ('ebu','ztopmax','steps','solver_k_min','solver_k_max','w','t','qv','qc','qh','qi','radius'):
                ref=baseline[case][k] if field!='carry_reverse' else carried[case][k]
                d=ulp_table(c[k],ref)
                if d['n_nonzero']:
                    t=totals.setdefault(k,dict(cases=0,max_ulp=0,words=0));t['cases']+=1;t['max_ulp']=max(t['max_ulp'],d['max_ulp']);t['words']+=d['n_nonzero']
        report[arm]['experiments'][field]=totals
(root/'state-audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({arm:{k:v['experiments'][k] for k in ('carry_reverse',*(['cvi_1','cvi_tail'] if arm=='wrf' else []))} for arm,v in report.items()}))
