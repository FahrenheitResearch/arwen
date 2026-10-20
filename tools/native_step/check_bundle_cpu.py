"""Compiled CPU checks for ordered native phases and exact cadence AST preservation."""
import ast,ctypes as C,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
from native_bundle import Bundle
from radiation_step import transformed_compute,PHYSICS_SHA256
ROOT=Path(__file__).resolve().parents[2]
class Trace(C.Structure):_fields_=[('size',C.c_uint64),('words',C.c_uint64*64)]
class Fixture(C.Structure):_fields_=[('trace',C.POINTER(Trace)),('id',C.c_uint64)]
class Frame(C.Structure):_fields_=[('value',C.c_uint64)]
lib=C.CDLL(str(ROOT/'receipts/libbundle-fixture.so'))
trace=Trace();owners=[Fixture(C.pointer(trace),i) for i in (1,2,3)]
def check(rc):
 if rc:raise ValueError('original compiled refusal '+str(rc))
plans=[NS(library=lib,handle=C.addressof(x),_ready=True,_same_thread=lambda:None,_check=check) for x in owners]
bundle=Bundle(ROOT/'receipts/libnative-bundle.so',[plans[0],plans[1],plans[1],plans[2]])
frames=[Frame(x) for x in (10,20,30,40)]
bundle.submit(frames,[1,2,3,4]);assert list(trace.words[:trace.size])==[1011,2022,2033,3044] and bundle.completed==4
for x in frames:x.value+=100
bundle.submit(frames,[8,0,8,0]);assert list(trace.words[4:trace.size])==[1118,2120,2138,3140]
checks=['ordered phases','same child plan repeated with separate live frame','changing predicate flags']
for frames_bad,flags in [(frames[:3],[0]*4),(frames,[0]*3),(frames,[0,0,-1,0])]:
 before=trace.size
 try:bundle.submit(frames_bad,flags)
 except ValueError:pass
 else:raise AssertionError('bad input accepted')
 assert trace.size==before
 checks.append('invalid input refused before any phase')
seen=[]
try:bundle.submit(frames,[0,1<<63,0,0],failure_stage=seen.append)
except ValueError as error:assert str(error)=='original compiled refusal -77'
else:raise AssertionError('missing failure')
assert bundle.completed==1 and seen==[1] and trace.size==9 and not plans[1]._ready
checks.append('first failing phase stops later phases and preserves error')
try:bundle.submit(frames,[0]*4)
except RuntimeError as error:assert 'terminal' in str(error)
else:raise AssertionError('partially submitted bundle reused')
checks.append('partial failure terminal')
raw=(ROOT/'gpuwm/core/physics.py').read_bytes();new=transformed_compute(raw);compile(new,'<cadence>','exec')
cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='PhysicsDriver');old=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='compute')
assert len(old.body)==len(new.body[0].body)
for a,b in zip(old.body,new.body[0].body):
 if a.lineno==5686:
  assert isinstance(a,ast.If) and isinstance(b,ast.With)
  assert ast.dump(ast.Module(body=a.body,type_ignores=[]))==ast.dump(ast.Module(body=b.body,type_ignores=[]))
 else:assert ast.dump(a)==ast.dump(b)
checks.append('entire original compute AST unchanged except due branch scope')
try:transformed_compute(raw+b'\n')
except RuntimeError:checks.append('source change refuses')
else:raise AssertionError('changed source admitted')
receipt=dict(status='PASS',scope=__doc__,checks=checks,gpu_calls=0,physics_sha256=PHYSICS_SHA256,source={n:hashlib.sha256((ROOT/'tools/native_step'/n).read_bytes()).hexdigest() for n in ('native_bundle.c','native_bundle.py','radiation_step.py','check_bundle_fixture.c','check_bundle_cpu.py')})
(ROOT/'receipts/native-bundle-cpu.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
