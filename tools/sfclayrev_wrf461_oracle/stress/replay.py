from pathlib import Path
import json, sys, subprocess, numpy as np
import _sfclayrev_oracle as h
from gpuwm.core.physics_inventory import SFCLAY_OUTPUTS
from gpuwm.core.fp32_ulp import fp32_ulp_distance
import os
L=Path(os.environ['SFCLAYREV_CHECK_ROOT'])
if sys.argv[1]=='build':
    driver=(L/'src/tools/sfclayrev_wrf461_oracle/run_sfclayrev.F90').read_text()
    start=driver.index('  ! arm:'); end=driver.index('  character(len=512)',start)
    driver=driver[:start]+'  integer, parameter :: arms(3, narms) = reshape((/1,0,0/),(/3,narms/))\n\n'+driver[end:]
    driver=driver.replace('nsteps = 3, narms = 6','nsteps = 3, narms = 1')
    driver=driver.replace('br, arms(1, arm), dx','br, merge(1,0,step == 1), dx')
    driver=driver.replace('arm, arms(:, arm), &','arm, merge(1,0,step == 1), arms(2:3,arm), &')
    (L/'replay-driver.F90').write_text(driver)
    subprocess.run(['gfortran','-c','-O0','-ffree-form','-ffree-line-length-none','-I','.','../replay-driver.F90','-o','replay-driver.o'],cwd=L/'oracle',check=True)
    subprocess.run(['gfortran','-o','run_replay','ccpp_kind_types.o','sf_sfclayrev_O0.o','module_sf_sfclayrev_O0.o','replay-driver.o'],cwd=L/'oracle',check=True)
    subprocess.run(['./run_replay','../src/tests/data/oracles/sfclayrev/sfclayrev-inputs.hex','../replay-outputs.hex'],cwd=L/'oracle',check=True)
    sys.exit(0)
import cupy as cp
from gpuwm.core.sfclay import SFClayResult, launch_sfclay
mode=sys.argv[1]
fixture=h.load_fixture(); inp=fixture.inputs
out=np.zeros((1,3,len(fixture.cases),34),np.float32); wrf=out.copy()
for line in (L/'replay-outputs.hex').read_text().splitlines():
    if not line or line.startswith('#'): continue
    tok=line.split(); arm,ff,ft,iz,step,case=map(int,tok[:6])
    assert (ff,ft,iz)==(int(step==1),0,0)
    wrf[0,step-1,case-1]=h._words(tok[6:])
for dx in np.unique(inp['dx']):
    sel=np.flatnonzero(inp['dx']==dx); g={k:cp.asarray(v[sel][None,:]) for k,v in inp.items()}
    carry=('znt','ust','ustm','mol','hfx','qfx','qsfc','zol')
    arrays={k:g[k].copy() if k in carry else cp.zeros((1,len(sel)),cp.float32) for k in SFCLAY_OUTPUTS}
    result=SFClayResult(**arrays)
    for step in range(3):
        launch_sfclay(*[g[k] for k in ('u','v','t','qv','p','dz8w','psfc','tsk','pblh','mavail','xland','lakemask')],result,option=1,dx=float(dx),isfflx=step==0)
        cp.cuda.Device().synchronize()
        for f,name in enumerate(SFCLAY_OUTPUTS): out[0,step,sel,f]=cp.asnumpy(getattr(result,name))[0]
mask=np.ones(out.shape,bool);mask[:,1:,:,SFCLAY_OUTPUTS.index('lh')]=False
fields={}; samples=[]
for f,name in enumerate(SFCLAY_OUTPUTS):
    ww=wrf[...,f];pp=out[...,f];m=mask[...,f]
    ulp=np.where(m,fp32_ulp_distance(ww,pp),0)
    fields[name]=dict(max_ulp=int(ulp.max()),max_abs=float(np.max(np.where(m,np.abs(ww.astype(float)-pp.astype(float)),0))),lanes_differ=int((ulp!=0).sum()),lanes=int(m.sum()))
    for a,s,c in list(zip(*np.nonzero(ulp)))[:3]: samples.append(dict(field=name,step=int(s)+1,case=int(c)+1,wrf=float(ww[a,s,c]),port=float(pp[a,s,c]),ulp=int(ulp[a,s,c])))
report=dict(mode=mode,columns=110,steps=3,flux_sequence=[1,0,0],compared_words=int(mask.sum()),max_ulp=max(x['max_ulp'] for x in fields.values()),max_abs=max(x['max_abs'] for x in fields.values()),mismatch_words=sum(x['lanes_differ'] for x in fields.values()),fields=fields,samples=samples)
(L/f'results/replay-{mode}.json').write_text(json.dumps(report,indent=1))
print(json.dumps({k:v for k,v in report.items() if k!='fields'},indent=1))
sys.exit(int(report['mismatch_words']!=0))
