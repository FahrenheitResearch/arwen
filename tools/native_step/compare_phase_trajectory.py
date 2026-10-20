"""Compare independent whole-period endpoint arrays and host metadata."""
import argparse,hashlib,json
from pathlib import Path
from receipt_io import receipt_bytes
p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--numerics',choices=('strict','fast'),required=True);p.add_argument('--python-attempt');a=p.parse_args()
root=Path(__file__).resolve().parents[2];paths=[root/f'receipts/phase-trajectory-{a.numerics}-{arm}-{attempt}.json' for arm,attempt in (('python',a.python_attempt or a.attempt),('native',a.attempt))]
left,right=[json.loads(receipt_bytes(p)) for p in paths]
assert left['status']==right['status']=='WHOLE_PHASE_TRAJECTORY_UNCOMPARED'
assert left['source']==right['source'],'source changed between arms'
comparisons={}
bootstrap=[case for case in right['cases'] if case['path']=='bootstrap']
assert len(bootstrap)==1 and bootstrap[0]['period']==67 and bootstrap[0]['phase_count']==7 and bootstrap[0]['capture_submissions']==1
assert [case['period'] for case in right['cases'] if case['path']=='native']==list(range(68,103))
assert set(left['endpoints'])=={'67','102','104'}
assert left['endpoints'].keys()==right['endpoints'].keys()
for period in left['endpoints']:
 l,r=left['endpoints'][period],right['endpoints'][period];lm={v['name']:v for v in l['arrays']};rm={v['name']:v for v in r['arrays']}
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
 summary={'numerics':a.numerics,'source':right['source'],'receipts':{str(p.relative_to(root)):{'sha256':hashlib.sha256(receipt_bytes(p)).hexdigest(),'bytes':len(receipt_bytes(p))} for p in paths},'array_views':len(names),'typed_words':sum(v['words'] for v in lm.values()),'different_arrays':changed,'semantic_pointer_tables':sum('pointer_targets' in v for v in lm.values()),'host_differences':host_changes,'process_compile_announcements':compile_values,'process_wall_values_checked_separately':wall_names,'native_submissions':sum(c.get('submissions',0) for c in right['cases']),'cases':right['cases'],'status':'PASS' if not changed and not host_changes else 'FAIL'}
 comparisons[period]=summary
assert right.get('closed_before_ordinary_tail') is True
summary={'status':'PASS' if all(row['status']=='PASS' for row in comparisons.values()) else 'FAIL','endpoints':comparisons,'native_submissions':sum(c.get('submissions',0) for c in right['cases']),'native_radiation_calls':sum(step['radiation_due'] for case in right['cases'] if case['path']=='native' for step in case['domain_steps'])}
assert summary['native_submissions']==35 and summary['native_radiation_calls']==2
output=root/f'receipts/phase-trajectory-{a.numerics}-comparison-{a.attempt}.json';assert not output.exists(),'preserve prior comparison'
output.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='endpoints'}));assert summary['status']=='PASS'
