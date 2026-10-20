"""Compare independent whole-period endpoint arrays and host metadata."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--prefix',default='whole-trajectory',choices=('whole-trajectory','whole-trajectory-wall','whole-trajectory-profile','whole-continuation'));p.add_argument('--numerics',choices=('strict','fast'),required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];paths=[root/f'receipts/{a.prefix}-{a.numerics}-{arm}.json' for arm in ('python','native')]
left,right=[json.loads(p.read_text()) for p in paths]
assert left['status']==right['status'] and left['status'] in ('EIGHT_INTERVALS_UNCOMPARED','TWELVE_INTERVALS_UNCOMPARED','TEN_INTERVALS_WITH_ORDINARY_TAIL_UNCOMPARED')
assert left['source']==right['source'],'source changed between arms'
l,r=left['final'],right['final'];lm={v['name']:v for v in l['arrays']};rm={v['name']:v for v in r['arrays']}
names=sorted(set(lm)|set(rm))
def same(k):
 x,y=lm.get(k),rm.get(k)
 if x is not None and y is not None and 'pointer_targets' in x and 'pointer_targets' in y:return {a:b for a,b in x.items() if a!='sha256'}=={a:b for a,b in y.items() if a!='sha256'}
 return x==y
changed=[k for k in names if not same(k)]
import math
wall_names=sorted(k for k in set(l['host'])|set(r['host']) if k=='root.model._step_log._started_wall' or k.startswith(('root.model._step_log._domain_wall.','root.model._step_log._first_steps.')))
for k in wall_names:
 for arm in (l,r):
  assert k in arm['host'] and isinstance(arm['host'][k],(int,float)) and math.isfinite(arm['host'][k]) and arm['host'][k]>=0,k
# Cache announcements describe process startup, not evolving model state.
# Keep only this exact diagnostic schema separate; unknown fields still fail.
compile_root='root.model._step_log._compile_announcement'
compile_names={compile_root,*(compile_root+'.'+k for k in
    ('reason','compute_capability','cached_entries','cached_entries_for_this_card'))}
compile_values={}
for name,arm in (('python',l),('native',r)):
 values={k:v for k,v in arm['host'].items() if k in compile_names}
 if compile_root in values:
  assert values[compile_root] is None and len(values)==1
 else:
  assert isinstance(values[compile_root+'.reason'],str)
  assert values[compile_root+'.compute_capability'] is None or isinstance(values[compile_root+'.compute_capability'],str)
  for field in ('cached_entries','cached_entries_for_this_card'):
   value=values[compile_root+'.'+field]
   assert isinstance(value,int) and not isinstance(value,bool) and value>=0
 compile_values[name]=values
host_changes={k:{'python':l['host'].get(k),'native':r['host'].get(k)} for k in sorted(set(l['host'])|set(r['host'])) if k not in wall_names and k not in compile_names and (k not in l['host'] or k not in r['host'] or l['host'][k]!=r['host'][k])}
summary={'numerics':a.numerics,'source':right['source'],'receipts':{str(p.relative_to(root)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in paths},'array_views':len(names),'typed_words':sum(v['words'] for v in lm.values()),'different_arrays':changed,'semantic_pointer_tables':sum('pointer_targets' in v for v in lm.values()),'host_differences':host_changes,'process_compile_announcements':compile_values,'process_wall_values_checked_separately':wall_names,'native_submissions':sum(c.get('submissions',0) for c in right['cases']),'cases':right['cases'],'status':'PASS' if not changed and not host_changes else 'FAIL'}
if a.prefix=='whole-trajectory-wall':
 import statistics
 summary['timing']={}
 for name,arm in (('python',left),('native',right)):
  rows=[r['ms'] for r in arm['parent_intervals_ms'] if r['parent_step']>=5]
  summary['timing'][name]={'mean_ms':statistics.mean(rows),'median_ms':statistics.median(rows),'samples_ms':rows,'intervals':'5..12'}
(root/f'receipts/{a.prefix}-{a.numerics}-comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2));assert summary['status']=='PASS'
