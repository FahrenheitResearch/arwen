"""CPU-only checks for new live MYNN scalar bindings and exact original flags."""
import ast,ctypes as C,hashlib,json,sys
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace as NS
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
from radiation_capture import RecordedRadiation,RadiationFrame,SOURCE_SHA256
raw=(ROOT/'gpuwm/core/rrtmg_legacy.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA256
tree=ast.parse(raw);flags={}
for node in ast.walk(tree):
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Call) and isinstance(node.func.func,ast.Name) and node.func.func.id=='_adapter_kernel' and node.func.args[0].value in ('rla_mynn','rla_mynn_gsd41'):
  flags[node.func.args[0].value]=compile(ast.Expression(node.args[1].elts[7]),'<original MYNN flag>','eval')
assert len(flags)==2
checks=[]
for kernel in flags:
 for step in (1,2,34,101,3701):
  binds=[];obj=RecordedRadiation.__new__(RecordedRadiation);obj.active_bl=True;obj.mynn_kernel=kernel;obj.plan=NS(bind=lambda *args:binds.append(args))
  obj.bind_mynn([dict(kernel=kernel,index=3,argument_bytes=[8]*7+[4]),dict(kernel=kernel,index=9,argument_bytes=[8]*7+[4])],RadiationFrame,'')
  obj.adapter=NS(start_time=datetime(2026,2,28));obj.cfg=NS(dt=18.,radt=30.);obj.year=2026;obj.module=NS(wrf_itimestep=lambda elapsed,dt:step);obj.bound=True;obj.frame_type=RadiationFrame;obj.prefix='';obj.check_bindings=lambda:None;obj.ozone=None
  frame=RadiationFrame();obj.fill(frame,1800.)
  expected=int(eval(flags[kernel],{'np':np,'itimestep':step}));field='mynn_first' if kernel.endswith('gsd41') else 'mynn_later'
  assert getattr(frame,field)==expected and [b[0] for b in binds]==[3,9] and all(b[1]==7 and b[3]==field for b in binds)
  checks.append(dict(kernel=kernel,step=step,value=expected))
for label,ops in [('missing',[]),('variant',[dict(kernel='rla_mynn_gsd41',index=0,argument_bytes=[8]*7+[4])]),('width',[dict(kernel='rla_mynn',index=0,argument_bytes=[8]*8)])]:
 obj.mynn_kernel='rla_mynn'
 try:obj.bind_mynn(ops,RadiationFrame,'')
 except RuntimeError as error:checks.append(dict(refusal=label,error=str(error)))
 else:raise AssertionError(label)
for value in (0,-1,1.0):
 obj.module.wrf_itimestep=lambda elapsed,dt:value
 try:obj.fill(frame,1800.)
 except ValueError as error:checks.append(dict(refusal=str(value),error=str(error)))
 else:raise AssertionError(value)
receipt=dict(status='PASS',scope=__doc__,checks=checks,gpu_calls=0,original_sha256=SOURCE_SHA256,port_sha256=hashlib.sha256((ROOT/'tools/native_step/radiation_capture.py').read_bytes()).hexdigest())
(ROOT/'receipts/radiation-mynn-cpu.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
