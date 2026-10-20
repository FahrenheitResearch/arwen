"""Reduce separate CUDA radiation transport profile and diagnostic transfers."""
from pathlib import Path
import collections,gzip,hashlib,json,sqlite3
root=Path(__file__).resolve().parents[2]
result={'scope':'second natural AC radiation call per domain; component only','comparisons':[]}
for numerics in ('strict','fast'):
 path=root/f'receipts/radiation-transport-{numerics}.json'
 receipt=json.loads(path.read_text());assert receipt['status']=='EXACT_RADIATION_TRANSPORT_PAIR'
 rows=[]
 for case in receipt['cases']:
  row={k:case[k] for k in ('shape','model_time','timing_mean_ms','inputs_unchanged')}
  row['comparison']={k:case['comparison'][k] for k in ('words','different_words')}
  row['operations']={}
  for arm,audit in case['audit'].items():
   plan=audit['plan'];row['operations'][arm]={'description_keys':list(plan),'error':plan['error']}
   # Complete recorded operation tables remain in the source receipt.
   ops=plan.get('operations',plan.get('ops',[]))
   if isinstance(ops,list):
    row['operations'][arm]['kinds']=dict(collections.Counter(o.get('api',o.get('kind','unknown')) for o in ops))
  if 'child_snapshot_not_alias' in case:row['child_snapshot_not_alias']=case['child_snapshot_not_alias']
  rows.append(row)
 result['comparisons'].append({'numerics':numerics,'receipt_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'cases':rows})
path=root/'receipts/radiation-transport.sqlite'
if path.exists():
 c=sqlite3.connect(path);c.row_factory=sqlite3.Row
 kernels=[dict(r) for r in c.execute('select k.*,s.value as name from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start')]
 starts=[k for k in kernels if k['name']=='awstep_radiation_start'];ends=[k for k in kernels if k['name']=='awstep_radiation_end'];assert len(starts)==len(ends)==5,(len(starts),len(ends))
 warnings=[dict(r) for r in c.execute('select * from DIAGNOSTIC_EVENT where severity>=2')]
 apis={(r['globalTid']&~((1<<24)-1),r['correlationId']):dict(r) for r in c.execute('select a.*,s.value as name from CUPTI_ACTIVITY_KIND_RUNTIME a join StringIds s on s.id=a.nameId')}
 raw=[];summaries=[]
 for label,start,end in zip(('parent_base','parent_candidate','child_base','child_candidate'),starts[1:],ends[1:]):
  lo,hi,pid=start['end'],end['start'],start['globalPid'];assert lo<hi and pid==end['globalPid']
  events=[]
  for table,kind in (('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_MEMCPY','copy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')):
   cross=c.execute(f'select count(*) from {table} where globalPid=? and ((start<? and end>?) or (start<? and end>?))',(pid,lo,lo,hi,hi)).fetchone()[0];assert cross==0,(label,kind,cross)
   if kind=='kernel':events.extend(dict(k,kind=kind) for k in kernels if k['globalPid']==pid and lo<=k['start'] and k['end']<=hi)
   else:events.extend(dict(r,kind=kind) for r in c.execute(f'select * from {table} where globalPid=? and start>=? and end<=?',(pid,lo,hi)))
  events.sort(key=lambda e:e['start']);busy=0;endpoint=lo
  for e in events:
   e['api']=apis.get((pid,e['correlationId']));assert e['api'] is not None,(label,e)
   busy+=max(0,e['end']-max(e['start'],endpoint));endpoint=max(endpoint,e['end'])
  kinds=collections.Counter(e['kind'] for e in events);copies=collections.defaultdict(lambda:{'calls':0,'bytes':0})
  for e in events:
   if e['kind']=='copy':copies[str(e['copyKind'])]['calls']+=1;copies[str(e['copyKind'])]['bytes']+=e['bytes']
  summaries.append({'range':label,'event_counts':dict(kinds),'copies_by_cupti_kind':dict(copies),'device_ms':(hi-lo)/1e6,'busy_ms':busy/1e6,'gap_ms':(hi-lo-busy)/1e6,'boundary_crossers':0,'api_correlated_events':len(events)})
  raw.append({'range':label,'start_marker':start,'end_marker':end,'events':events})
 output=root/'receipts/radiation-transport-events.json.gz'
 with gzip.open(output,'wt',encoding='utf-8',compresslevel=9) as f:json.dump(raw,f,separators=(',',':'))
 result['profile']={'copy_kind_enum':[dict(r) for r in c.execute('select * from ENUM_CUDA_MEMCPY_OPER')],'sqlite_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'warnings':warnings,'ranges':summaries,'events_sha256':hashlib.sha256(output.read_bytes()).hexdigest()};c.close()
(root/'receipts/radiation-transport-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
