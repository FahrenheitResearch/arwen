"""Compare port calendar frame words with the exact original AST statements, CPU only."""
import ast,ctypes as C,hashlib,json,sys
from pathlib import Path
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace as NS
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from radiation_capture import RecordedRadiation,RadiationFrame,SOURCE_SHA256
source=ROOT/'gpuwm/core/rrtmg_legacy.py';raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA256
cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='RRTMGLegacyRadiation')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
lines={1944,1946,1947,1948,1951,1964,1967,1968,1974}
statements=[n for n in method.body if n.lineno in lines]
assert {n.lineno for n in statements}==lines
code=compile(ast.fix_missing_locations(ast.Module(body=statements,type_ignores=[])),str(source),'exec')
checks=[]
for stamp in [datetime(2026,2,28,tzinfo=timezone.utc),datetime(2024,2,28,23,59,59,123456,tzinfo=timezone.utc)]:
 for seconds in [1782.,1800.,66582.,66600.,86399.999999,172800.25]:
  cfg=NS(radt=30.,radt_minutes=20.);state=NS(elapsed_seconds=seconds,fnm=np.ones(4,np.float32),fnp=np.ones(4,np.float32));adapter=NS(start_time=stamp,_ozone_provider=None)
  namespace={'self':adapter,'state':state,'cfg':cfg,'timedelta':timedelta,'F':np.float32}
  exec(code,namespace)
  obj=RecordedRadiation.__new__(RecordedRadiation);obj.adapter=adapter;obj.cfg=cfg;obj.state=state;obj.vertical=(state.fnm,state.fnp);obj.provider=None;obj.parent_ozone=None;obj.ozone=None;obj.bound=True;obj.frame_type=RadiationFrame;obj.prefix='';obj.year=stamp.year;obj.active_bl=False
  frame=RadiationFrame();obj.fill(frame,seconds)
  expected={key:int(np.float32(namespace[key]).view(np.uint32)) for key in ('julian','xt_mid','gmt')}
  actual={key:int(np.float32(getattr(frame,key)).view(np.uint32)) for key in expected}
  assert actual==expected,(stamp,seconds,expected,actual)
  checks.append({'start':stamp.isoformat(),'elapsed':seconds,'words':actual})
# New port refuses crossing a trace-gas year before submitting a frame.
obj.adapter.start_time=datetime(2026,12,31,23,59,tzinfo=timezone.utc);obj.year=2026
try:obj.fill(RadiationFrame(),120.)
except RuntimeError as e:assert 'trace-gas year changed' in str(e);refusal=str(e)
else:raise AssertionError('new year accepted')
receipt={'status':'PASS','scope':__doc__,'command':'CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1 python -B tools/native_step/check_radiation_calendar.py','original_source_sha256':SOURCE_SHA256,'port_source_sha256':hashlib.sha256((ROOT/'tools/native_step/radiation_capture.py').read_bytes()).hexdigest(),'original_ast_lines':sorted(lines),'cases':checks,'exact_float32_words':len(checks)*3,'year_refusal':refusal,'gpu_calls':0}
(ROOT/'receipts/radiation-calendar-cpu.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
