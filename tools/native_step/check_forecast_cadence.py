"""CPU check of original clock recurrence and radiation cadence at the all three interior seams."""
import ast,copy,hashlib,json,math
from pathlib import Path
from types import SimpleNamespace
from fractions import Fraction
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
from radiation_step import PHYSICS_SHA256
raw=(ROOT/'gpuwm/core/clock.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='c569d49e47be055b6fff0fc0983f8d43198df2f048fb991c32b9d6b79ebd11e2'
node=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='DomainClock')
ns={'np':np,'_FP32_ALL_INTEGERS_EXACT_BOUND':1<<24}
body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node]
exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),'<original clock>','exec'),ns)
Clock=ns['DomainClock'];clock=Clock(SimpleNamespace(step_ticks=54,dt_fp32=np.float32(54),lbc_interval_ticks=21600),1,86400)
rows=[]
for period in range(1600):
 planned=copy.copy(clock)
 if planned.lbc_reset_due():planned.mark_force()
 planned.prepare_step()
 if clock.lbc_reset_due():clock.mark_force()
 clock.prepare_step()
 assert planned.ticks==clock.ticks and planned.dtbc_launch_fp32.tobytes()==clock.dtbc_launch_fp32.tobytes()
 rows.append(dict(period=period,elapsed=clock.elapsed_seconds,dtbc=float(clock.dtbc_launch_fp32)))
 clock.advance()
assert rows[399]=={'period':399,'elapsed':21546.0,'dtbc':21600.0}
assert rows[400]=={'period':400,'elapsed':21600.0,'dtbc':54.0}
for i in (400,800,1200):assert rows[i]['dtbc']==54.0 and rows[i-1]['dtbc']==21600.0
wrong=Clock(SimpleNamespace(step_ticks=54,dt_fp32=np.float32(54),lbc_interval_ticks=21600),1,86400);wrong.ticks=21600;wrong.dtbc_fp32=np.float32(21600);wrong.prepare_step();assert float(wrong.dtbc_fp32)==21654.0 and wrong.dtbc_fp32!=rows[400]['dtbc']
raw=(ROOT/'gpuwm/core/physics.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==PHYSICS_SHA256
methods=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in ('_physics_interval_steps','_radiation_step_due')]
ns={'np':np,'math':math,'Fraction':Fraction};exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),'<original radiation cadence>','exec'),ns)
interval,due=ns['_physics_interval_steps'],ns['_radiation_step_due'];calls=[]
for period in range(68,1598):
 for gid,dt,minutes,indices in [(1,54.,29.7,[period]),(2,18.,30.,range(period*3,period*3+3))]:
  for index in indices:
   if due(index+1,interval(minutes,dt),minutes):calls.append(dict(period=period,gid=gid,step=index+1))
assert len(calls)==91 and len(range(67,1600,4))==384
receipt=dict(status='PASS',clock_periods_checked=1600,native_periods=1530,health_checks=384,native_radiation_calls=len(calls),seam_rows=[rows[i] for i in (399,400,799,800,1199,1200)],missing_reset_control=float(wrong.dtbc_fp32),calls=calls,gpu_calls=0,source_lines={'DomainClock':node.lineno,**{n.name:n.lineno for n in methods}},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(ROOT/'receipts/forecast-cadence-cpu.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
