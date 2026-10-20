"""Reduce CUDA-device marker intervals without NVTX collection dependence."""
from pathlib import Path
import sqlite3,json,hashlib,statistics,collections,gzip
root=Path(__file__).resolve().parents[2]
result={'scope':'completed AC parent and three child steps, intervals 3..12, non-radiation phase only','arms':[]}
for arm in ('python','native'):
 path=root/f'receipts/trajectory-{arm}.sqlite';c=sqlite3.connect(path);c.row_factory=sqlite3.Row
 kernels=[dict(k) for k in c.execute('select k.start,k.end,k.correlationId,k.globalPid,k.streamId,s.value as name from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start')]
 starts=[k for k in kernels if k['name']=='awstep_parent_interval_start'];ends=[k for k in kernels if k['name']=='awstep_parent_interval_end']
 assert len(starts)==len(ends)==12,(len(starts),len(ends))
 warnings=[dict(r) for r in c.execute('select * from DIAGNOSTIC_EVENT where severity>=2')]
 assert len({k['globalPid'] for k in starts+ends})==1
 pid=starts[0]['globalPid'];rows=[];raw=[]
 api={(r['globalTid']&~((1<<24)-1),r['correlationId']):dict(r) for r in c.execute('select a.start,a.end,a.globalTid,a.correlationId,s.value as name from CUPTI_ACTIVITY_KIND_RUNTIME a join StringIds s on s.id=a.nameId')}
 for step,(start,end) in enumerate(zip(starts,ends),1):
  assert start['end']<end['start']
  lo,hi=start['end'],end['start']
  for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
   crossers=c.execute(f'select count(*) from {table} where globalPid=? and ((start<? and end>?) or (start<? and end>?))',(pid,lo,lo,hi,hi)).fetchone()[0]
   assert crossers==0,(arm,step,table,crossers)
  events=[dict(k,kind='kernel') for k in kernels if k['globalPid']==pid and lo<=k['start'] and k['end']<=hi]
  for table,kind in (('CUPTI_ACTIVITY_KIND_MEMCPY','copy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')):
   events.extend(dict(r,kind=kind) for r in c.execute(f'select start,end,correlationId,globalPid,streamId from {table} where globalPid=? and start>=? and end<=?',(pid,lo,hi)))
  for e in events:
   matched=api.get((pid,e['correlationId']))
   assert matched is not None,(arm,step,e)
   if e['kind']=='kernel':assert matched['name'] in ('cuLaunchKernel','cudaLaunchKernel_v7000'),matched
   e['api']=matched
  events.sort(key=lambda r:r['start']);busy=0;endpoint=lo
  for e in events:
   busy+=max(0,e['end']-max(e['start'],endpoint));endpoint=max(endpoint,e['end'])
  counts=collections.Counter(e['kind'] for e in events)
  symbols=collections.defaultdict(lambda:{'calls':0,'sum_ns':0})
  for e in events:
   if e['kind']=='kernel':symbols[e['name']]['calls']+=1;symbols[e['name']]['sum_ns']+=e['end']-e['start']
  row={'parent_interval':step,'device_range_ns':hi-lo,'device_busy_ns':busy,'device_gap_ns':hi-lo-busy,'event_counts':dict(counts),'boundary_crossing_activities':0,'matched_api_activities':len(events),'symbols':dict(symbols)}
  rows.append(row)
  if step>=3:raw.append({'parent_interval':step,'start_marker':start,'end_marker':end,'events':events})
 selected=rows[2:]
 compact=root/f'receipts/trajectory-{arm}-events.json.gz'
 with gzip.open(compact,'wt',encoding='utf-8',compresslevel=9) as f:json.dump(raw,f,separators=(',',':'))
 entry={'arm':arm,'sqlite_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'warnings':warnings,'ranges':rows,'retained_events':{'path':str(compact.relative_to(root)),'bytes':compact.stat().st_size,'sha256':hashlib.sha256(compact.read_bytes()).hexdigest()},'summary_ms':{key:statistics.mean(r[key] for r in selected)/1e6 for key in ('device_range_ns','device_busy_ns','device_gap_ns')},'mean_kernel_count':statistics.mean(r['event_counts']['kernel'] for r in selected)}
 result['arms'].append(entry);c.close()
result['status']='COLLECTOR_WARNING' if any(a['warnings'] for a in result['arms']) else 'CUDA_DEVICE_INTERVAL_PROFILE'
(root/'receipts/trajectory-profile-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'arms':[{k:a[k] for k in ('arm','warnings','summary_ms','mean_kernel_count','retained_events')} for a in result['arms']]},indent=2))
