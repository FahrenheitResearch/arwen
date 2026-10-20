"""Compare separate pinned trajectory arms without tolerances."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--prefix',default='trajectory',choices=('trajectory','trajectory-wall','trajectory-profile','radiation-trajectory'));p.add_argument('--numerics',choices=('strict','fast'),required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];paths=[root/f'receipts/{a.prefix}-{a.numerics}-{arm}.json' for arm in ('python','native')]
left,right=[json.loads(p.read_text()) for p in paths]
assert left['status']==right['status'] and left['status'] in ('FOUR_INTERVALS_COMPLETE_UNCOMPARED','TWELVE_INTERVALS_COMPLETE_UNCOMPARED','FORTY_INTERVALS_WITH_RADIATION_FALLBACK_UNCOMPARED')
assert left['source']==right['source'], 'source changed between arms'
summary={'numerics':a.numerics,'source':right['source'],'receipts':{str(p.relative_to(root)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in paths},'domains':[],'native_submissions':sum(c.get('submissions',0) for c in right['cases']),'cases':[{k:v for k,v in c.items() if k not in ('operations','bindings')} for c in right['cases']]}
for l,r in zip(left['final'],right['final'],strict=True):
 assert l['shape']==r['shape']
 lm={v['name']:v for v in l['arrays']};rm={v['name']:v for v in r['arrays']}
 names=sorted(set(lm)|set(rm))
 def same(k):
  x,y=lm.get(k),rm.get(k)
  if x is not None and y is not None and 'pointer_targets' in x and 'pointer_targets' in y:
   return {a:b for a,b in x.items() if a!='sha256'}=={a:b for a,b in y.items() if a!='sha256'}
  return x==y
 changed=[k for k in names if not same(k)]
 host_changes={k:{'python':l['host'].get(k),'native':r['host'].get(k)} for k in sorted(set(l['host'])|set(r['host'])) if l['host'].get(k)!=r['host'].get(k)}
 summary['domains'].append({'shape':l['shape'],'array_views':len(names),'typed_words':sum(v['words'] for v in lm.values()),'different_arrays':changed,'semantic_pointer_tables':sum('pointer_targets' in v for v in lm.values()),'bitwise_words':sum(v['words'] for v in lm.values() if 'pointer_targets' not in v),'semantic_pointer_words':sum(v['words'] for v in lm.values() if 'pointer_targets' in v),'host_differences':host_changes,'model_time':r['host']['state.elapsed_seconds']})
if a.prefix=='trajectory-wall':
 import statistics
 summary['timing']={}
 for name,arm in (('python',left),('native',right)):
  rows=[r['ms'] for r in arm['parent_intervals_ms'] if r['parent_step']>=3]
  summary['timing'][name]={'mean_ms':statistics.mean(rows),'median_ms':statistics.median(rows),'samples_ms':rows,'intervals':'3..12'}
summary['status']='PASS' if all(not d['different_arrays'] and not d['host_differences'] for d in summary['domains']) else 'FAIL'
(root/f'receipts/{a.prefix}-{a.numerics}-comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ('status','domains','native_submissions')},indent=2));assert summary['status']=='PASS'
