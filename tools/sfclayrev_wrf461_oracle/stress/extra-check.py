from pathlib import Path
import json, sys, numpy as np
import cupy as cp
import _sfclayrev_oracle as h
from gpuwm.core.physics_inventory import SFCLAY_OUTPUTS
from gpuwm.core.sfclay import SFClayResult, launch_sfclay
import os
L=Path(os.environ['SFCLAYREV_CHECK_ROOT'])
mode=sys.argv[1]; kind=sys.argv[2]
if kind=='baseline-audit':
    fixture=h.load_fixture(outputs_name=str(L/'oracle/sfclayrev-outputs.hex'))
    port=h.port_outputs(fixture)
elif kind=='extra':
    design=json.loads((L/'results/extra-design.json').read_text())
    h.ARMS=tuple(tuple(x) for x in design['arms']); h.STEPS=design['steps']
    h.WRF_UNDEFINED=frozenset((i,'lh') for i,a in enumerate(h.ARMS) if a[0]==0)
    fixture=h.load_fixture(L/'extra'); port=h.port_outputs(fixture)
else:
    fixture=h.load_fixture(outputs_name=str(L/'results/seeded-outputs.hex'))
    port=np.zeros(fixture.outputs.shape,np.float32)
    carry=('znt','ust','ustm','mol','hfx','qfx','qsfc','zol')
    seed=dict(chs=.125,chs2=.25,cqs2=.375,flhc=.5,flqc=.001)
    for a,switches in enumerate(h.ARMS):
        for dx in np.unique(fixture.inputs['dx']):
            sel=np.flatnonzero(fixture.inputs['dx']==dx)
            g={k:cp.asarray(v[sel][None,:]) for k,v in fixture.inputs.items()}
            arrays={k:g[k].copy() if k in carry else cp.full((1,len(sel)),seed.get(k,0.),cp.float32) for k in SFCLAY_OUTPUTS}
            result=SFClayResult(**arrays)
            for step in range(h.STEPS):
                launch_sfclay(*[g[k] for k in ('u','v','t','qv','p','dz8w','psfc','tsk','pblh','mavail','xland','lakemask')],result,option=1,dx=float(dx),isfflx=bool(switches[0]),isftcflx=switches[1],iz0tlnd=switches[2])
                cp.cuda.Device().synchronize()
                for f,name in enumerate(SFCLAY_OUTPUTS): port[a,step,sel,f]=cp.asnumpy(getattr(result,name))[0]
report=h.measure(fixture,port)
# Independent raw-word census also counts signed zero and nonfinite words.
mask=np.ones(port.shape,bool)
for a,name in h.WRF_UNDEFINED: mask[a,...,SFCLAY_OUTPUTS.index(name)]=False
rawdiff=(port.view(np.uint32)!=fixture.outputs.view(np.uint32))&mask
report.update(columns=len(fixture.cases),arms=h.ARMS,steps=h.STEPS,compared_words=int(mask.sum()),raw_word_differences=int(rawdiff.sum()),nan_reference=int((np.isnan(fixture.outputs)&mask).sum()),nan_port=int((np.isnan(port)&mask).sum()),inf_reference=int((np.isinf(fixture.outputs)&mask).sum()),inf_port=int((np.isinf(port)&mask).sum()),mismatch_words=sum(v['lanes_differ'] for v in report['fields'].values()))
report['mismatch_samples']=sorted(report.pop('lanes'),key=lambda x:-x[5])[:30]
(L/f'results/{kind}-{mode}.json').write_text(json.dumps(report,indent=1,default=str))
np.savez(L/f'{kind}-{mode}-words.npz',port=port,wrf=fixture.outputs,mask=mask)
print(json.dumps({k:v for k,v in report.items() if k!='fields'},indent=1,default=str))
for k,v in report['fields'].items(): print(k,v)
sys.exit(int(report['max_ulp']!=0 or report['raw_word_differences']!=0))
