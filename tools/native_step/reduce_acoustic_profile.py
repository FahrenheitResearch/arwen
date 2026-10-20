"""Reduce explicit acoustic NVTX ranges; retain collector limitations."""
from pathlib import Path
import sqlite3,json,hashlib,statistics
root=Path(__file__).resolve().parents[2]
path=root/'receipts/acoustic-native.sqlite'
c=sqlite3.connect(path);c.row_factory=sqlite3.Row
ranges=list(c.execute("select start,end,globalTid,text from NVTX_EVENTS where text like 'acoustic/%' order by start"))
assert len(ranges)==24,len(ranges)
rows=[]
for r in ranges:
 api=list(c.execute('select a.*,s.value as name from CUPTI_ACTIVITY_KIND_RUNTIME a join StringIds s on s.id=a.nameId where a.globalTid=? and a.start>=? and a.end<=?',(r['globalTid'],r['start'],r['end'])))
 launches=[a for a in api if a['name']=='cuLaunchKernel']
 pid=r['globalTid']&~((1<<24)-1)
 kernels=[]
 for a in launches:
  kernels.extend(dict(k) for k in c.execute('select k.start,k.end,k.correlationId,k.globalPid,s.value as name from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.correlationId=? and k.globalPid=?',(a['correlationId'],pid)))
 assert len(launches)==4 and len(kernels)==4,(r['text'],len(launches),len(kernels))
 kernels.sort(key=lambda k:k['start'])
 assert all(r['start']<=k['start']<k['end']<=r['end'] for k in kernels)
 union=0;end=0
 for k in kernels:
  union+=max(0,k['end']-max(k['start'],end));end=max(end,k['end'])
 rows.append({'range':r['text'],'range_ns':r['end']-r['start'],'observed_launches':len(launches),
  'observed_kernels':len(kernels),'kernel_sum_ns':sum(k['end']-k['start'] for k in kernels),
  'device_busy_union_ns':union,'range_minus_observed_busy_ns':r['end']-r['start']-union,
  'device_internal_gap_ns':kernels[-1]['end']-kernels[0]['start']-union,
  'kernels':kernels,'api':[{k:a[k] for k in ('start','end','correlationId','name')} for a in api]})
warnings=[dict(r) for r in c.execute('select * from DIAGNOSTIC_EVENT where severity>=2')]
summary=[]
for shape in ('120x70','267x117'):
 for mode in ('python','native'):
  group=[r for r in rows if f'/{shape}/' in r['range'] and r['range'].endswith('/'+mode)]
  summary.append({'shape':shape,'mode':mode,'cases':len(group),**{k:statistics.mean(r[k] for r in group)/1e6 for k in ('range_ns','kernel_sum_ns','device_busy_union_ns','range_minus_observed_busy_ns','device_internal_gap_ns')}})
result={'status':'DIAGNOSTIC_COLLECTOR_WARNING' if warnings else 'COMPONENT_PROFILE',
 'scope':'four-kernel acoustic call only, not domain or completed parent step',
 'source_sqlite_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
 'warnings':warnings,'summary_ms':summary,'ranges':rows}
(root/'receipts/acoustic-profile-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('status','summary_ms','warnings')},indent=2))
