from pathlib import Path
import json, numpy as np
from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.core.physics_inventory import SFCLAY_OUTPUTS
import os
L=Path(os.environ['SFCLAYREV_CHECK_ROOT'])
reports={}
for mode in ('strict','default'):
    file=L/f'extra-{mode}-words.npz'
    if not file.exists(): continue
    data=np.load(file); w,p,mask=[data[k] for k in ('wrf','port','mask')]
    finite=mask&np.isfinite(w)&np.isfinite(p)
    bits=(w.view(np.uint32)!=p.view(np.uint32))&mask
    bad_columns=np.any((~np.isfinite(w)|~np.isfinite(p))&mask,axis=(0,1,3))
    fields={}
    for f,name in enumerate(SFCLAY_OUTPUTS):
        sel=finite[...,f]
        ww=w[...,f][sel]; pp=p[...,f][sel]
        fields[name]=dict(finite_words=int(sel.sum()),max_finite_ulp=int(fp32_ulp_distance(pp,ww).max()),max_finite_abs=float(np.max(np.abs(ww.astype(float)-pp.astype(float)))),finite_word_differences=int((ww.view(np.uint32)!=pp.view(np.uint32)).sum()),nan_words=int((np.isnan(w[...,f])&mask[...,f]).sum()),inf_words=int((np.isinf(w[...,f])&mask[...,f]).sum()),raw_word_differences=int(bits[...,f].sum()))
    samples=[]
    for a,s,c,f in list(zip(*np.nonzero(bits)))[:12]: samples.append(dict(arm=int(a)+1,step=int(s)+1,case=int(c)+1,field=SFCLAY_OUTPUTS[f],wrf_word=f'{w.view(np.uint32)[a,s,c,f]:08X}',port_word=f'{p.view(np.uint32)[a,s,c,f]:08X}'))
    r=dict(finite_words=int(finite.sum()),finite_word_differences=int((bits&finite).sum()),max_finite_ulp=max(x['max_finite_ulp'] for x in fields.values()),max_finite_abs=max(x['max_finite_abs'] for x in fields.values()),nonfinite_columns=(np.flatnonzero(bad_columns)+1).tolist(),fully_finite_columns=int((~bad_columns).sum()),raw_differences=int(bits.sum()),nan_classification_differences=int(((np.isnan(w)!=np.isnan(p))&mask).sum()),inf_classification_differences=int(((np.isinf(w)!=np.isinf(p))&mask).sum()),samples=samples,fields=fields)
    (L/f'results/extra-finite-{mode}.json').write_text(json.dumps(r,indent=1)); reports[mode]={k:v for k,v in r.items() if k!='fields'}
(L/'results/extra-finite-summary.json').write_text(json.dumps(reports,indent=1))
print(json.dumps(reports,indent=1))
