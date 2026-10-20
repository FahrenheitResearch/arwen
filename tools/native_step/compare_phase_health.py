"""Compare scheduled health inputs without waiving full inventory differences."""
import argparse,hashlib,json
from pathlib import Path
from receipt_io import receipt_bytes
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',required=True);p.add_argument('--numerics',required=True,choices=('strict','fast'));a=p.parse_args()
root=Path(__file__).resolve().parents[2]
paths=[root/f'receipts/phase-trajectory-{a.numerics}-{arm}-{a.attempt}.json' for arm in ('python','native')]
left,right=[json.loads(receipt_bytes(p)) for p in paths]
assert left['status']==right['status']=='WHOLE_PHASE_TRAJECTORY_UNCOMPARED'
assert left['source']==right['source']
checks=left['scheduled_health_witnesses'];other=right['scheduled_health_witnesses']
assert sorted(map(int,checks))==list(range(67,104,4)) and checks.keys()==other.keys()
rows={}
for period,l in checks.items():
 r=other[period];changed=[gid for gid in set(l)|set(r) if l.get(gid)!=r.get(gid)]
 rows[period]=dict(equal=not changed,different_domains=changed,fields=sum(len(v['fields']) for v in l.values()),typed_words=sum(f['values']['words']+(f['auxiliary']['words'] if f['auxiliary'] else 0) for v in l.values() for f in v['fields']),domains=sorted(l))
# Keep the unscoped comparison untouched. This second receipt narrows its
# assertion explicitly to the arrays submitted to each actual health check.
summary=dict(status='PASS' if all(r['equal'] for r in rows.values()) else 'FAIL',scope='scheduled health fields, auxiliaries, rules, launch geometry and result; device descriptors checked against actual fields in each arm',checks=rows,total_fields=sum(r['fields'] for r in rows.values()),total_typed_words=sum(r['typed_words'] for r in rows.values()),whole_inventory_pass=False,receipts={str(p.relative_to(root)):hashlib.sha256(receipt_bytes(p)).hexdigest() for p in paths},source=right['source'])
output=root/f'receipts/phase-health-{a.numerics}-comparison-{a.attempt}.json';assert not output.exists();output.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ('checks','source','receipts')}));assert summary['status']=='PASS'
