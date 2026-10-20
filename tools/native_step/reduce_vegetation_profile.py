"""Reduce only correlated assembled-column profile ranges, retaining warnings."""
from pathlib import Path
import sqlite3,json,hashlib,collections
root=Path(__file__).resolve().parents[2];path=root/'receipts/vegetation-native.sqlite'
c=sqlite3.connect(path);c.row_factory=sqlite3.Row
ranges=list(c.execute("select start,end,globalTid,text from NVTX_EVENTS where text like 'vegetation/%' order by start"));assert len(ranges)==4,len(ranges)
rows=[]
for r in ranges:
 api=list(c.execute('select a.*,s.value as name from CUPTI_ACTIVITY_KIND_RUNTIME a join StringIds s on s.id=a.nameId where a.globalTid=? and a.start>=? and a.end<=?',(r['globalTid'],r['start'],r['end'])))
 launches=[a for a in api if a['name']=='cuLaunchKernel'];pid=r['globalTid']&~((1<<24)-1)
 kernels=[]
 for a in launches:
  kernels.extend(dict(k) for k in c.execute('select k.start,k.end,k.correlationId,k.globalPid,s.value as name from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.correlationId=? and k.globalPid=?',(a['correlationId'],pid)))
 assert len(launches)==len(kernels) and len(kernels)>0
 kernels.sort(key=lambda k:k['start'])
 assert all(r['start']<=k['start']<k['end']<=r['end'] for k in kernels)
 union=0;end=0
 for k in kernels:union+=max(0,k['end']-max(k['start'],end));end=max(end,k['end'])
 rows.append({'range':r['text'],'range_ms':(r['end']-r['start'])/1e6,'observed_launches':len(launches),'observed_kernels':len(kernels),'kernel_sum_ms':sum(k['end']-k['start'] for k in kernels)/1e6,'device_busy_union_ms':union/1e6,'range_minus_observed_kernel_busy_ms':(r['end']-r['start']-union)/1e6,'api_counts':dict(collections.Counter(a['name'] for a in api)),'kernels':kernels})
warnings=[dict(r) for r in c.execute('select * from DIAGNOSTIC_EVENT where severity>=2')]
result={'status':'DIAGNOSTIC_COLLECTOR_WARNING' if warnings else 'COMPONENT_PROFILE','scope':'assembled land-column call only; busy includes kernels only','source_sqlite_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'warnings':warnings,'ranges':rows}
(root/'receipts/vegetation-profile-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'warnings':warnings,'ranges':[{k:v for k,v in r.items() if k!='kernels'} for r in rows]},indent=2))
