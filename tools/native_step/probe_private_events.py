"""CUDA event lifetime regression and explicit private-upload observer refusals."""
import ctypes as C
import hashlib,importlib.util,json,os
from pathlib import Path
import cupy as cp
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('native_step',ROOT/'gpuwm/native_step/__init__.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
driver=C.CDLL('libcuda.so.1')
for name,types in {'cuEventCreate':[C.POINTER(C.c_void_p),C.c_uint], 'cuEventDestroy_v2':[C.c_void_p], 'cuEventRecord':[C.c_void_p,C.c_void_p], 'cuEventQuery':[C.c_void_p], 'cuStreamWaitEvent':[C.c_void_p,C.c_void_p,C.c_uint], 'cuMemcpyHtoDAsync_v2':[C.c_uint64,C.c_void_p,C.c_size_t,C.c_void_p]}.items():
 fn=getattr(driver,name);fn.argtypes=types;fn.restype=C.c_int

def call(name,*args):
 rc=getattr(driver,name)(*args);assert rc==0,(name,rc)
def event():
 e=C.c_void_p();call('cuEventCreate',C.byref(e),2);return e

def close(plan):
 cp.cuda.Device().synchronize();plan.close()

out=cp.zeros(8,cp.int32)
add=cp.RawKernel('extern "C" __global__ void add(int*x){x[threadIdx.x]+=1;}','add')
add((1,),(8,),(out,));cp.cuda.Device().synchronize()
pinned=cp.cuda.alloc_pinned_memory(32);source=np.frombuffer(pinned,dtype=np.int32,count=8);source[:]=np.arange(8,dtype=np.int32)
receipt={'purpose':__doc__,'pid':os.getpid(),'checks':[],'source':{}}
for path in ('gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/probe_private_events.py','receipts/libawstep-private-events.so'):
 receipt['source'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

def upload(e):
 call('cuMemcpyHtoDAsync_v2',out.data.ptr,source.ctypes.data,source.nbytes,cp.cuda.get_current_stream().ptr)
 call('cuEventRecord',e,cp.cuda.get_current_stream().ptr)

# Destroy the original event, mutate the original host bytes, then replay.
e=event();plan=native.Plan()
try:
 with plan.record(add,out):
  with plan.constant_uploads(private_events=True):upload(e)
  add((1,),(8,),(out,))
 cp.cuda.Device().synchronize();call('cuEventDestroy_v2',e);e=None
 description=plan.describe();assert sum(op.get('private_event',False) for op in description['operations'])==1
 source[:]=-9
 for _ in range(5):
  out.fill(-1);plan.submit();cp.cuda.Device().synchronize()
  assert np.array_equal(cp.asnumpy(out),np.arange(8,dtype=np.int32)+1)
 receipt['checks'].append({'kind':'original_event_destroyed','replays':5,'different_words':0,'operations':description['operations']})
finally:
 close(plan)
 if e:call('cuEventDestroy_v2',e)

# Public event still belongs to the caller and can be queried after replay.
e=event();plan=native.Plan()
try:
 with plan.record(add,out,e):
  add((1,),(8,),(out,));call('cuEventRecord',e,cp.cuda.get_current_stream().ptr)
 assert not any(op.get('private_event') for op in plan.describe()['operations'])
 plan.submit();cp.cuda.Device().synchronize();call('cuEventQuery',e)
 receipt['checks'].append({'kind':'public_event_query','query_rc':0})
finally:close(plan);call('cuEventDestroy_v2',e)

for mode,message in [('external_wait','private upload event has an external waiter'),('inside_wait','private upload scope cannot wait on events'),('external_record','private upload event escaped its scope')]:
 e=event();plan=native.Plan()
 try:
  try:
   with plan.record(out):
    with plan.constant_uploads(private_events=True):
     upload(e)
     if mode=='inside_wait':call('cuStreamWaitEvent',cp.cuda.get_current_stream().ptr,e,0)
    if mode=='external_wait':call('cuStreamWaitEvent',cp.cuda.get_current_stream().ptr,e,0)
    if mode=='external_record':call('cuEventRecord',e,cp.cuda.get_current_stream().ptr)
  except native.RecordingError as error:
   assert message in str(error),(mode,str(error))
   receipt['checks'].append({'kind':mode,'message':str(error),'ready':plan.describe()['ready']})
  else:raise AssertionError(f'{mode} accepted')
 finally:close(plan);call('cuEventDestroy_v2',e)
receipt['status']='PASS'
(ROOT/'receipts/private-events-contract.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(receipt['checks'])}))
