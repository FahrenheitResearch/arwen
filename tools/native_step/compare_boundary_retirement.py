"""Strict model state and scheduled health comparison, with named cache differences.

The full raw inventory comparator remains a separate, stricter receipt.
"""
import argparse,hashlib,json,re
from pathlib import Path
from receipt_io import receipt_bytes
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',required=True);p.add_argument('--numerics',choices=('strict','fast'),required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2]
paths=[root/f'receipts/boundary-trajectory-{a.numerics}-{arm}-{a.attempt}.json' for arm in ('python','native')]
left,right=[json.loads(receipt_bytes(p)) for p in paths]
full=json.loads((root/f'receipts/boundary-trajectory-{a.numerics}-comparison-{a.attempt}.json').read_text())
health=json.loads((root/f'receipts/boundary-health-{a.numerics}-comparison-{a.attempt}.json').read_text())
assert health['status']=='PASS' and left['source']==right['source']==health['source']
assert health['receipts']=={str(p.relative_to(root)):hashlib.sha256(receipt_bytes(p)).hexdigest() for p in paths}
assert full['native_submissions']==337 and full['native_radiation_calls']==20
prefix={1:'root.model.root.state',2:'root.model.root.children[0].state'}
counter_prefix=prefix[1]+'.physics.radiation_callable._cuda_sw._scratch_by_stream.0.'
counters={counter_prefix+k for k in ('allocations','filled_bytes','zeroed_bytes')}
summary={}
last_native=max(c['period'] for c in right['cases'] if c['path']=='native')
assert last_native==404 and right['closed_before_ordinary_tail'] is True
retired={x['name']:x for x in right['endpoints'][str(last_native)]['arrays']}

for period,l in left['endpoints'].items():
 r=right['endpoints'][period];lm={x['name']:x for x in l['arrays']};rm={x['name']:x for x in r['arrays']}
 cached={};latest=max(int(k) for k in left['scheduled_health_witnesses'] if int(k)<=int(period))
 for name in sorted(lm.keys()-rm.keys()):
  match=re.fullmatch(r'root\.validators\.([12])\.fields\[(\d+)\]\.values',name)
  assert match and (int(period)+1)%4!=0,('unexpected missing native array',name,period)
  gid,index=map(int,match.groups());field=left['scheduled_health_witnesses'][str(latest)][str(gid)]['fields'][index]
  assert field['name']==l['host'][f'root.validators.{gid}.fields[{index}].name']==r['host'][f'root.validators.{gid}.fields[{index}].name']
  assert {k:v for k,v in lm[name].items() if k!='name'}==field['values'],('retained array is not last checked input',name)
  fname=field['name']
  if fname.startswith('held.pbl.'):
   target=prefix[gid]+'.physics.pbl_tendencies.'+fname.removeprefix('held.pbl.')
  elif fname.startswith('held.radiation.'):
   target=prefix[gid]+'.physics.radiation_tendencies.'+fname.removeprefix('held.radiation.')
  else:
   assert fname in ('held.rthratenlw','held.rthratensw'),fname
   target=prefix[gid]+'.physics.'+fname.removeprefix('held.')
  assert target in lm and target in rm and lm[target]==rm[target],('current held field differs',target)
  cached[name]=dict(last_checked_period=latest,field=fname,current_target=target,typed_words=lm[name]['words'],sha256=lm[name]['sha256'])
 assert not rm.keys()-lm.keys(),('unexpected native-only arrays',sorted(rm.keys()-lm.keys()))
 pointer_changes={};exact=[];retired_caches={}
 for name in sorted(lm.keys()&rm.keys()):
  x,y=lm[name],rm[name]
  if x!=y and 'pointer_targets' not in x:
   match=re.fullmatch(r'root\.validators\.([12])\.fields\[(\d+)\]\.values',name)
   assert match and int(period)>last_native and int(period)<407,('non-retirement state differs',period,name)
   gid,index=map(int,match.groups());field=left['scheduled_health_witnesses'][str(latest)][str(gid)]['fields'][index]
   assert latest==403 and field['name'].startswith('held.pbl.')
   assert field['name']==l['host'][f'root.validators.{gid}.fields[{index}].name']==r['host'][f'root.validators.{gid}.fields[{index}].name']
   target=prefix[gid]+'.physics.pbl_tendencies.'+field['name'].removeprefix('held.pbl.')
   assert {k:v for k,v in x.items() if k!='name'}==field['values'],('ordinary retired cache is not last checked input',name)
   assert {k:v for k,v in y.items() if k!='name'}=={k:v for k,v in retired[target].items() if k!='name'},('native retired cache is not last native output',name)
   assert lm[target]==rm[target],('current model field changed after retirement',target)
   retired_caches[name]=dict(field=field['name'],ordinary_last_check=latest,native_last_write=last_native,current_target=target,typed_words=y['words'],ordinary_sha256=x['sha256'],native_sha256=y['sha256'])
   continue
  if 'pointer_targets' not in x:
   assert x==y,('current state array differs',period,name);exact.append(name);continue
  assert {k:v for k,v in x.items() if k not in ('sha256','pointer_targets')}=={k:v for k,v in y.items() if k not in ('sha256','pointer_targets')}
  assert len(x['pointer_targets'])==len(y['pointer_targets'])
  changes=[]
  for i,(old,new) in enumerate(zip(x['pointer_targets'],y['pointer_targets'])):
   if old==new:continue
   assert name.endswith('.integration_health_field_ptr') and isinstance(old,list) and len(old)==1 and old[0] in cached
   assert new==[cached[old[0]]['current_target']],('unknown pointer representation',name,i,old,new)
   changes.append(dict(index=i,retained_input=old[0],current_target=new[0]))
  if changes:pointer_changes[name]=changes
 host=full['endpoints'][period]['host_differences']
 assert set(host)<=counters,('non-accounting host difference',host)
 for name,values in host.items():
  assert all(isinstance(v,int) and not isinstance(v,bool) and v>=0 for v in values.values())
 summary[period]=dict(exact_array_views=len(exact),exact_typed_words=sum(lm[n]['words'] for n in exact),semantic_pointer_tables=sum('pointer_targets' in x for x in lm.values()),retained_validator_inputs=cached,pointer_representation_changes=pointer_changes,allocation_accounting=host,retired_validator_caches=retired_caches)
out=dict(status='PASS',scope='all current model arrays exact; retired validator caches tied to last ordinary check or last native write; all pointer changes explained by retained last-check inputs versus current fields; all 85 scheduled health checks exact; three SW allocation counters reported separately',endpoints=summary,scheduled_health_checks=len(health['checks']),scheduled_health_fields=health['total_fields'],scheduled_health_typed_words=health['total_typed_words'],native_submissions=337,native_radiation_calls=20,post_retirement_scheduled_check_proved=False,raw_inventory_status=full['status'],source=right['source'],receipts=health['receipts'])
f=root/f'receipts/boundary-retirement-{a.numerics}-comparison-{a.attempt}.json';assert not f.exists();f.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('endpoints','source','receipts')}))
