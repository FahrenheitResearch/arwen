"""CPU original cadence check for changing parent and child radiation phases."""
import ast,hashlib,json,math
from pathlib import Path
import numpy as np
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[2]
from radiation_step import PHYSICS_SHA256
raw=(ROOT/'gpuwm/core/physics.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==PHYSICS_SHA256
methods=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in ('_physics_interval_steps','_radiation_step_due')]
assert len(methods)==2
ns={'np':np,'math':math,'Fraction':Fraction};exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),'<original cadence>','exec'),ns)
interval=ns['_physics_interval_steps'];due=ns['_radiation_step_due'];assert interval(29.7,54.)==33 and interval(30.,18.)==100
rows=[]
for period in range(63,105):
 phases=[]
 for gid,dt,minutes,indices in [(1,54.,29.7,[period]),(2,18.,30.,range(period*3,period*3+3))]:
  for index in indices:
   now=index*dt;n=int(np.floor(now/dt+.5))+1
   phases.append(dict(gid=gid,step=n,time=now,due=bool(due(n,interval(minutes,dt),minutes))))
 rows.append(dict(period=period,phases=phases))
by={r['period']:r for r in rows}
assert [p['due'] for p in by[66]['phases']]==[True,False,False,True]
assert [p['due'] for p in by[99]['phases']]==[True,False,False,False]
assert [p['due'] for p in by[100]['phases']]==[False,True,False,False]
assert sum(p['due'] for r in rows if 67<r['period']<=102 for p in r['phases'])==2
assert (67+1)%4==0 and not any(p['due'] for p in by[67]['phases'])
receipt=dict(status='PASS',scope=__doc__,physics_sha256=PHYSICS_SHA256,source_lines={n.name:n.lineno for n in methods},capture_period=66,validator_capture_period=67,native_periods=[68,102],native_parent_periods=35,native_radiation_calls=2,rows=rows,gpu_calls=0)
(ROOT/'receipts/phase-cadence-cpu.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='rows'}))
