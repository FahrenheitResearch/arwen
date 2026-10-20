"""CPU-only check of production history ordering on the original clock executor."""
import ast,copy,hashlib,json,math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
raw=(ROOT/'gpuwm/core/clock.py').read_bytes()
names={'DomainClock','execute_schedule','fixed_periods'}
nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
class Report:
 def __init__(self,**kw):
  self.steps=self.forces=self.feedback_calls=self.restarts=self.lbc_resets=0
  self.__dict__.update(kw)
ns={'np':np,'_FP32_ALL_INTEGERS_EXACT_BOUND':1<<24,'STEP':'STEP','FORCE':'FORCE','ExecutionReport':Report}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])),'<original-clock>','exec'),ns)
Clock=ns['DomainClock'];specs=[NS(grid_id=gid,step_ticks=dt,dt_fp32=np.float32(dt),lbc_interval_ticks=21600 if gid==1 else None,start_ticks=0,restart_ticks=None,history_offset_due=lambda t:t%21600==0) for gid,dt in ((1,54),(2,18))]
clocks={s.grid_id:Clock(s,1,86400) for s in specs}
ops=[NS(kind='STEP',grid_id=1),NS(kind='FORCE',grid_id=2,parent_id=1),*[NS(kind='STEP',grid_id=2) for _ in range(3)],NS(kind='FEEDBACK',grid_id=2,parent_id=1)]
schedule=NS(clock=NS(root=specs[0],domains=specs,run_ticks=86400),periods=1600,period_ticks=54,period_ops=lambda _:ops)
model=NS(node=lambda gid:NS(clock=clocks[gid]))
driver=ast.parse((ROOT/'tools/native_step/run_canary_step.py').read_text())
func=next(n for n in ast.walk(driver) if isinstance(n,ast.FunctionDef) and n.name=='history_production_due')
ns2={'model':model,'copy':copy};exec(compile(ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[])),'<production-predicate>','exec'),ns2)
physical={1:0,2:0};live={};fallbacks=[];segmented=[];histories=[];seams=[];native=[]
def begin(period,cs):
 live.update(period=period,native=67<=period<=1597)
 if live['native'] and ns2['history_production_due']():fallbacks.append(period);live['native']=False
 split=False
 if live['native']:
  for gid,count in ((1,1),(2,3)):
   dom=copy.copy(cs[gid])
   for _ in range(count):split=split or dom.history_due();dom.advance()
  if split:segmented.append(period)
  else:physical.update({1:(period+1)*54,2:(period+1)*54})
  native.append(period)
 live['split']=split
 def not_used():pass
def step(gid,dom):
 if not live['native'] or live['split']:physical[gid]=dom.ticks+dom.step_ticks
 if gid==1 and dom.ticks in (21600,43200,64800):
  assert float(dom.dtbc_launch_fp32)==54.0;seams.append(dom.ticks)
def history(gid,ticks):
 assert physical[gid]==ticks,(gid,ticks,physical[gid],live)
 histories.append((gid,ticks))
report=ns['execute_schedule'](schedule,clocks=clocks,on_period_begin=begin,on_step=step,on_history=history)
assert report.steps==6400 and report.forces==1600 and report.feedback_calls==1600
assert report.histories=={1:5,2:5} and report.lbc_resets==4
assert fallbacks==[399,799,1199] and segmented==[400,800,1200]
assert len(native)==1528 and len(histories)==10 and len(seams)==3
physics=(ROOT/'gpuwm/core/physics.py').read_bytes()
methods=[n for n in ast.parse(physics).body if isinstance(n,ast.FunctionDef) and n.name in ('_physics_interval_steps','_radiation_step_due')]
ns3={'np':np,'math':math,'Fraction':Fraction};exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),'<original-radiation-cadence>','exec'),ns3)
for period in fallbacks:
 for gid,dt,minutes,indices in ((1,54.,29.7,[period]),(2,18.,30.,range(period*3,period*3+3))):
  for index in indices:assert not ns3['_radiation_step_due'](index+1,ns3['_physics_interval_steps'](minutes,dt),minutes)
receipt=dict(status='PASS',scope='CPU clock/order only, no physical field or GPU qualification',gpu_calls=0,periods=1600,domain_steps=report.steps,histories=histories,history_production_fallbacks=fallbacks,native_history_segments=segmented,external_seams=seams,clock_sha256=hashlib.sha256(raw).hexdigest(),driver_sha256=hashlib.sha256((ROOT/'tools/native_step/run_canary_step.py').read_bytes()).hexdigest())
(ROOT/'receipts/canary-history-order-cpu.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
