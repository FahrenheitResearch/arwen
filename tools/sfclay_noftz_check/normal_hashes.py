"""Record strict actual words for the original five column suites."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import cupy as cp
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
L = Path(os.environ.get('SFCLAY_CHECK_ROOT','/work/pool-sfclay-noftz-mynn-mm5-eta'))
result = {}
digest = hashlib.sha256()
def add(x):
    digest.update(np.ascontiguousarray(x).tobytes())
from tools.mynn_sfclay_wrf461_column_oracle import compare as m
old = m._grade
def grade(got, want):
    for f in want:
        add(got[f])
    return old(got,want)
m._grade = grade
fixture=ROOT/'tests/data/oracles/mynn/sfclay-columns-wrf461'
r=m.summarize(m.run(fixture/'oracle-o2.bin.gz',fixture))
result['mynn_surface']=dict(actual_sha256=digest.hexdigest(),words=r['compared_values'],mismatches=r['mismatched_values'])

from tools.sfclay_classic_wrf461_oracle import validate_sfclay_classic_oracle as c
digest=hashlib.sha256()
old_step=c.woof_step
def step(*a,**kw):
    got=old_step(*a,**kw); add(got); return got
c.woof_step=step
r=c.grade(L/'oracles/classic/columns.txt',L/'oracles/classic/wrf-ck-corrected.txt')
result['classic']=dict(actual_sha256=digest.hexdigest(),words=sum(r[m]['words_compared'] for m in ('free','replay')),mismatches=sum(sum(r[m]['differing_words'].values()) for m in ('free','replay')))

import _sfclayrev_oracle as rev
from gpuwm.core.physics_inventory import SFCLAY_OUTPUTS
digest=hashlib.sha256(); f=rev.load_fixture(); got=rev.port_outputs(f)
bad=0; count=0
for j,name in enumerate(SFCLAY_OUTPUTS):
    a,b=got[...,j],f.outputs[...,j]
    if name=='lh': a,b=a[:5],b[:5]
    add(a); bad+=int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)));count+=a.size
result['revised']=dict(actual_sha256=digest.hexdigest(),words=count,mismatches=bad)

from gpuwm.verify import myjsfc_oracle as eta
digest=hashlib.sha256(); old_stats=eta._stats
def stats(a,b):
    add(a); row=old_stats(a,b);row['mismatch']=int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)));return row
eta._stats=stats
r=eta.measure(eta.load())
result['eta']=dict(actual_sha256=digest.hexdigest(),words=sum(v['count'] for m in ('replay','free_run') for v in r[m].values()),mismatches=sum(v['mismatch'] for m in ('replay','free_run') for v in r[m].values()))

import _mynn_columns_gsd41 as p
from gpuwm.core.mynn_pbl_gpu import mynn_bl_driver_cuda
digest=hashlib.sha256();bad=count=0
def driver(v,**kw):
    return mynn_bl_driver_cuda({k:cp.asarray(np.ascontiguousarray(a)) for k,a in v.items()},**kw)
for asis in (False,True):
    for replay in (False,True):
        for snow in (False,True):
            rows=p.integrate(driver,asis=asis,replay=replay,snow=snow,to_host=cp.asnumpy,bl_mynn_cloud_tendency_form='gsd_41')
            for _,a,b in rows:
                for name in (*p.PROFILE_OUTPUTS,*p.COLUMN_OUTPUTS,*p.INDEX_OUTPUTS): add(a[name])
            table=p.compare(rows);bad+=sum(x['bits'] for x in table.values());count+=sum(x['values'] for x in table.values())
result['mynn_pbl']=dict(actual_sha256=digest.hexdigest(),words=count,mismatches=bad)
mode='strict' if os.environ.get('GPUWM_WRF_EXACT')=='1' else 'default'
(L/'receipts'/f'normal-{mode}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
raise SystemExit(any(r['mismatches'] for r in result.values()))
