"""Actual Plan API wired to compiled recorder with CPU CUDA-call doubles."""
from pathlib import Path
import ctypes as C
import importlib.util
import json
import os
ROOT=Path(__file__).resolve().parents[2]
LIB=ROOT/'receipts/libnative-host-cpu-fixture.so'
os.environ['LD_PRELOAD']=str(LIB)
spec=importlib.util.spec_from_file_location('native_step_cpu',ROOT/'gpuwm/native_step/__init__.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class Args(C.Structure):
    _fields_=[('input',C.POINTER(C.c_int64)),('output',C.POINTER(C.c_int64)),('delta',C.c_int64)]
class Frame(C.Structure):
    _fields_=[('args',Args)]
plan=module.Plan(LIB);lib=plan.library
for name,types in [('fixture_begin',[C.c_void_p]),('fixture_end',[C.c_void_p]),('fixture_read',[C.POINTER(C.c_int64)]*2)]:
    getattr(lib,name).argtypes=types;getattr(lib,name).restype=C.c_int
lib.fixture_kernels.restype=C.c_uint
assert lib.fixture_begin(plan.handle)==0
plan._recording=True
source,ordinary,out=C.c_int64(12),C.c_int64(),C.c_int64()
checks=[]
with plan.readback() as operations:
    assert lib.fixture_read(C.byref(ordinary),C.byref(source))==0
address,size=plan.host_buffer(operations[0]);assert size==8
args=Args(C.cast(address,C.POINTER(C.c_int64)),C.pointer(out),9)
errors=[]
def error():
    errors.append(True)
    return ValueError('original fixture rejection')
operation=plan.native_host(LIB,'fixture_consumer',args,out,error=error)
assert out.value==21 and len(plan.resources)==3;checks.append('capture and retained provider/struct/owner')
args.delta=88
assert lib.fixture_end(plan.handle)==0
plan._recording=False;plan._ready=True
source.value=15;ordinary.value=999
plan.submit()
assert out.value==24 and ordinary.value==999 and not errors;checks.append('snapshot and live owned readback')
plan.bind(operation,0,Frame,'args')
frame=Frame(Args(C.cast(address,C.POINTER(C.c_int64)),C.pointer(out),13))
source.value=21;plan.submit(frame)
assert out.value==34;checks.append('changing scalar frame')
source.value=-1;before=lib.fixture_kernels()
try:
    plan.submit(frame)
except ValueError as exc:
    assert str(exc)=='original fixture rejection'
else:
    raise AssertionError('missing original error')
assert errors==[True] and lib.fixture_kernels()==before and not plan._ready
checks.append('exact failure factory and no downstream kernel')
try:
    plan.submit(frame)
except module.RecordingError as exc:
    assert str(exc)=='recording did not complete'
else:
    raise AssertionError('invalid plan accepted')
checks.append('partial plan refuses reuse')
plan.close();assert not plan.resources and plan.handle is None
checks.append('owners released on close')
assert 'cupy' not in __import__('sys').modules
print(json.dumps({'status':'PASS','python_c_integration_cases':len(checks),'cases':checks,'cuda_calls':0}))
