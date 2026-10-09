"""Require before/after strict hashes, positive real coverage, and no misses."""
import hashlib
import json
import os
from pathlib import Path
L=Path(os.environ.get('SFCLAY_CHECK_ROOT','/work/pool-sfclay-noftz-mynn-mm5-eta'))
result={'units':{}}
for unit in ('mynn_surface','mynn_pbl','classic','revised','eta'):
    before=json.loads((L/'before-strict'/f'{unit}-strict.json').read_text())
    after=json.loads((L/'receipts'/f'{unit}-strict.json').read_text())
    default=json.loads((L/'after-default'/f'{unit}-default.json').read_text())
    assert after['mismatches']==default['mismatches']==0,(unit,after['mismatches'],default['mismatches'])
    assert before['actual_sha256']==after['actual_sha256'],unit
    result['units'][unit]={'strict_sha256':after['actual_sha256'],'strict_unchanged':True,'default_mismatches':0}
before=json.loads((L/'before-strict/normal-strict.json').read_text())
after=json.loads((L/'receipts/normal-strict.json').read_text())
assert before==after,(before,after)
result['normal_columns']=after
paths=[]
for name in ('real-before2','real-after'):
    matches=list((L/name/'woof').glob('*/*/hashes.json'))
    assert len(matches)==1,(name,matches)
    d=json.loads(matches[0].read_text())
    assert d['woof']['status']=='OK' and d['frames']>1,(name,d['woof'],d['frames'])
    assert all(v for f in d['fields'].values() for v in f['sha256']),name
    paths.append(d)
a,b=paths
different={k for k in a['fields'].keys()|b['fields'].keys() if a['fields'].get(k)!=b['fields'].get(k)}
result['real_run']={'frames':a['frames'],'fields':len(a['fields']),'field_frame_hashes':sum(len(f['sha256']) for f in a['fields'].values()),'different_fields':sorted(different),'before_status':a['woof']['status'],'after_status':b['woof']['status'],'aggregate_sha256':hashlib.sha256(json.dumps(a['fields'],sort_keys=True,separators=(',',':')).encode()).hexdigest()}
(L/'receipts/final-verdict.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert not different,different
