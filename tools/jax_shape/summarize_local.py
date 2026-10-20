"""Retain bounded component evidence without raw fields or environment tables."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,sqlite3,statistics
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'evidence/jax-shape/local'
fast=json.loads((OUT/'timing-fast.json').read_text()); strict=json.loads((OUT/'timing-strict.json').read_text())
counts={(x['module'],x['function'],tuple(x['shape'])):x['count'] for x in fast['counts_second_period']}
rows=[];families=defaultdict(lambda:{'count':0,'before_estimate_ms':0.,'after_estimate_ms':0.})
for r in fast['rows']:
 key=(r['module'],r['function'],tuple(r['shape']));n=counts.get(key,0);a,b=(r['median_ms'][v] for v in ('before','after'))
 row=dict(module=r['module'],function=r['function'],shape=r['shape'],second_period_calls=n,before_median_ms=a,after_median_ms=b,speedup=a/b,sample_count=len(r['samples_ms']['before']),before_range_ms=[min(r['samples_ms']['before']),max(r['samples_ms']['before'])],after_range_ms=[min(r['samples_ms']['after']),max(r['samples_ms']['after'])]);rows.append(row)
 f=families[r['function']];f['count']+=n;f['before_estimate_ms']+=n*a;f['after_estimate_ms']+=n*b
before=sum(f['before_estimate_ms'] for f in families.values());after=sum(f['after_estimate_ms'] for f in families.values())
summary=dict(status='component_measured_full_step_unmeasured',word_checks={m:dict(rows=len(d['rows']),words=sum(x['words'] for x in d['rows']),different_bytes=sum(x['different_bytes'] for x in d['rows']),instrumented_seconds=d['seconds']) for m,d in [('fast',fast),('strict',strict)]},weighted_estimate=dict(scope='First-call medians weighted by counts in second completed parent interval; not measured step time',calls=sum(f['count'] for f in families.values()),before_ms=before,after_ms=after,reduction_percent=100*(1-after/before),speedup=before/after),families=dict(families),rows=rows)
c=sqlite3.connect(ROOT/'.shape-local/component-profile.sqlite');c.row_factory=sqlite3.Row
sql="""SELECT n.text AS marker,s.value AS function,k.end-k.start AS ns,k.registersPerThread AS registers,k.localMemoryPerThread AS local_bytes,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.graphId,k.graphNodeId FROM NVTX_EVENTS n JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.globalTid=n.globalTid AND r.start>=n.start AND r.end<=n.end JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId JOIN StringIds s ON s.id=k.shortName WHERE n.text LIKE 'shape.%' ORDER BY n.start"""
profile=[dict(r) for r in c.execute(sql)]; assert len(profile)==54 and len({r['marker'] for r in profile})==54
assert all(r['graphId'] is None and r['graphNodeId'] is None for r in profile)
diagnostics=[dict(r) for r in c.execute('select * from DIAGNOSTIC_EVENT')]
summary['profile']=dict(scope='Diagnostic due to collector CUDA/NVTX loss warnings; 54 expected marked component launches all matched once',observed_launches={'before':27,'after':27},before_ms=sum(r['ns'] for r in profile if r['marker'].startswith('shape.before:'))/1e6,after_ms=sum(r['ns'] for r in profile if r['marker'].startswith('shape.after:'))/1e6,rows=profile,diagnostics=diagnostics,query=sql)
# Only the tables/rows needed to independently reproduce the marked-launch join.
# Never copy profiler environment or process metadata tables.
target=OUT/'marked-launches.sqlite';o=sqlite3.connect(target)
for name,where in [('NVTX_EVENTS',"text LIKE 'shape.%'"),('CUPTI_ACTIVITY_KIND_RUNTIME',"EXISTS(SELECT 1 FROM NVTX_EVENTS n WHERE n.text LIKE 'shape.%' AND n.globalTid=CUPTI_ACTIVITY_KIND_RUNTIME.globalTid AND CUPTI_ACTIVITY_KIND_RUNTIME.start>=n.start AND CUPTI_ACTIVITY_KIND_RUNTIME.end<=n.end)"),('CUPTI_ACTIVITY_KIND_KERNEL',"correlationId IN (SELECT r.correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN NVTX_EVENTS n ON n.globalTid=r.globalTid AND r.start>=n.start AND r.end<=n.end WHERE n.text LIKE 'shape.%')")]:
 o.execute('DROP TABLE IF EXISTS '+name)
 o.execute(c.execute('select sql from sqlite_master where name=?',(name,)).fetchone()[0])
 rr=c.execute('SELECT * FROM '+name+' WHERE '+where).fetchall()
 if rr:o.executemany('INSERT INTO '+name+' VALUES('+','.join('?' for _ in rr[0])+')',rr)
o.execute('DROP TABLE IF EXISTS StringIds');o.execute('CREATE TABLE StringIds (id INTEGER PRIMARY KEY,value TEXT)')
ids={r[0] for r in o.execute('SELECT shortName FROM CUPTI_ACTIVITY_KIND_KERNEL')}
o.executemany('INSERT INTO StringIds VALUES (?,?)',[tuple(c.execute('SELECT id,value FROM StringIds WHERE id=?',(i,)).fetchone()) for i in ids]);o.commit()
assert len(o.execute(sql).fetchall())==54;o.close();c.close()
(OUT/'measurement-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','families','profile')},indent=2));print('diagnostic_profile_first_calls_ms',summary['profile']['before_ms'],summary['profile']['after_ms'])
