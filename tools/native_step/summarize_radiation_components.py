"""Compact exact component results and remaining audited transport sites."""
from collections import Counter
import datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(name):
 p=ROOT/'receipts'/name
 if not p.exists():return None
 value=json.loads(p.read_text())
 return value,{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def plan_summary(plan):
 events=plan.get('audit_events',[]);sites=Counter()
 for event in events:
  frames=[f for f in event.get('stack',[]) if '/gpuwm/' in f['file']]
  frame=frames[-1] if frames else {}
  sites[(event['reason'],frame.get('file',''),frame.get('line',0))]+=1
 return {'operations':len(plan['operations']),'kernels':plan['kernel_launches'],'audit_event_count':len(events),'remaining_sites':[dict(reason=k[0],file=k[1],line=k[2],count=n) for k,n in sorted(sites.items())],'ready':plan['ready'],'error':plan['error']}
r={'created':datetime.datetime.now(datetime.timezone.utc).isoformat(),'full_step_qualified':False,'full_radiation_replay':False,'timing_claim':False,'sources':[],'components':{}}
for numerics in ('strict','fast'):
 for kind in ('solar-transport','solar-transport-daytime','native-ozone','deferred-radiation-audit'):
  name=f'{kind}-{numerics}.json';loaded=read(name);key=f'{kind}-{numerics}'
  if loaded is None:r['components'][key]={'status':'UNRUN'};continue
  value,pin=loaded;r['sources'].append(pin)
  row={'status':value['status'],'error':value.get('error'),'numerics':numerics,'cases':[]}
  for case in value.get('cases',[]):
   item={k:v for k,v in case.items() if k not in ('plan','audit','annotations','deferred_annotations')}
   if 'plan' in case:item['plan']=plan_summary(case['plan'])
   if 'audit' in case:item['audit']={k:plan_summary(v['plan']) if 'plan' in v else v for k,v in case['audit'].items()}
   if 'deferred_annotations' in case:item['deferred_annotations']=case['deferred_annotations']
   row['cases'].append(item)
  if 'plan' in value:row['plan']=plan_summary(value['plan'])
  if 'daylight_plan' in value:row['daylight_plan']=plan_summary(value['daylight_plan'])
  for k in ('workers','upload_bytes','native_submissions','inputs_unchanged','ordinary_grid_unchanged','daylight_readback_bytes','daylight_original_unchanged','daylight_capture_host_array_ignored'):
   if k in value:row[k]=value[k]
  r['components'][key]=row
p=ROOT/'receipts/radiation-components-summary.json';p.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v['status'] for k,v in r['components'].items()}))
