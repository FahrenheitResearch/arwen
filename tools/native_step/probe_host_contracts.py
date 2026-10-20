"""Real CUDA retained-host and native model-validation guard contracts."""
import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import cupy as cp
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('native_step', ROOT/'gpuwm/native_step/__init__.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
receipt={'purpose':__doc__,'pid':os.getpid(),'checks':[], 'source':{}}
for name in ('gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py',
             'tools/native_step/probe_host_contracts.py', 'gpuwm/core/noahmp_energy_slab.py',
             'gpuwm/core/noahmp_sflx_post_slab.py'):
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
class Frame(C.Structure):
 _fields_=[('bad',C.c_int32),('code',C.c_int32)]
add=cp.RawKernel('extern "C" __global__ void add(int*x){x[threadIdx.x]+=1;}','add')
make=cp.RawKernel('extern "C" __global__ void make(int*x,int bad,int code){x[threadIdx.x]=threadIdx.x==bad?code:0;}','make')
x=cp.zeros(8,cp.int32); out=cp.zeros(8,cp.int32)
add((1,),(8,),(out,));make((1,),(8,),(x,np.int32(-1),np.int32(0)))
cp.cuda.Device().synchronize()
started=time.perf_counter()

def close(plan):
 cp.cuda.Device().synchronize();plan.close()

def bind_make(plan):
 op=next(op['index'] for op in plan.describe()['operations'] if op.get('kernel')=='make')
 plan.bind(op,1,Frame,'bad');plan.bind(op,2,Frame,'code')

def exception_from(function, *args):
 try:function(*args)
 except Exception as error:return error
 raise AssertionError('model validation did not reject bad input')

# The original host buffer can go away after the explicit invariant upload.
source=np.arange(8,dtype=np.int32)
plan=native.Plan()
try:
 with plan.record(add):
  with plan.constant_uploads():
   uploaded=cp.asarray(source)
  add((1,),(8,),(uploaded,))
 plan.resources.append(uploaded)
 cp.cuda.Device().synchronize()
 source[:]=-91
 uploaded.fill(-17)
 plan.submit();cp.cuda.Device().synchronize()
 result=cp.asnumpy(uploaded)
 assert np.array_equal(result,np.arange(8,dtype=np.int32)+1),result
 receipt['checks'].append({'kind':'retained_constant_upload','words':8,'differences':0,
                           'original_host_mutated_after_capture':True,'operations':plan.describe()['operations']})
finally:close(plan)

# Guard runs inside the native sequence and prevents a later output write.
plan=native.Plan()
try:
 out.fill(0)
 with plan.record(make,add,x,out):
  make((1,),(8,),(x,np.int32(-1),np.int32(0)))
  with plan.zero_guard(lambda index,value: ValueError(f'status[{index}]={value}'),word_bytes=4):
   status=cp.asnumpy(x)
  add((1,),(8,),(out,))
 bind_make(plan)
 plan.submit(Frame(-1,0));cp.cuda.Device().synchronize()
 before=cp.asnumpy(out)
 try:plan.submit(Frame(3,7))
 except ValueError as error:message=str(error)
 else:raise AssertionError('failed native guard accepted')
 cp.cuda.Device().synchronize()
 assert message=='status[3]=7',message
 assert np.array_equal(before,cp.asnumpy(out))
 try:plan.submit(Frame(-1,0))
 except native.RecordingError:pass
 else:raise AssertionError('partially submitted failed plan reused')
 receipt['checks'].append({'kind':'native_status_stop','first_word':3,'value':7,
                           'downstream_different_words':0,'failed_plan_rejected':True,
                           'operations':plan.describe()['operations']})
finally:close(plan)

# Capture itself refuses an already failing guard, with the same exception.
plan=native.Plan()
try:
 try:
  with plan.record(make,x):
   make((1,),(8,),(x,np.int32(2),np.int32(5)))
   with plan.zero_guard(lambda index,value: RuntimeError(f'bad {index}/{value}'),word_bytes=4):
    status=cp.asnumpy(x)
 except RuntimeError as error:assert str(error)=='bad 2/5',str(error)
 else:raise AssertionError('invalid capture accepted')
 assert not plan.describe()['ready']
 receipt['checks'].append({'kind':'failing_capture','message':'bad 2/5','replayable':False})
finally:close(plan)

# Exact original Noah-MP admission exception, including column and wording.
from gpuwm.core.noahmp_energy_slab import _refuse_unadmitted
from gpuwm.core.noahmp_sflx_post_slab import _raise_for_status
original_asnumpy=cp.asnumpy

def model_case(label,call,guard_width,code):
 plan=native.Plan()
 def original_error(index,value):
  previous=cp.asnumpy;cp.asnumpy=original_asnumpy
  try:return exception_from(call)
  finally:cp.asnumpy=previous
 def read_with_guard(array,*args,**kwargs):
  with plan.zero_guard(original_error,word_bytes=guard_width):
   return original_asnumpy(array,*args,**kwargs)
 try:
  out.fill(0)
  with plan.record(make,add,x,out,call):
   make((1,),(8,),(x,np.int32(-1),np.int32(0)))
   cp.asnumpy=read_with_guard
   try:call()
   finally:cp.asnumpy=original_asnumpy
   add((1,),(8,),(out,))
  bind_make(plan)
  plan.submit(Frame(-1,0));cp.cuda.Device().synchronize()
  before=cp.asnumpy(out)
  make((1,),(8,),(x,np.int32(6),np.int32(code)))
  expected=exception_from(call)
  make((1,),(8,),(x,np.int32(-1),np.int32(0)))
  cp.cuda.Device().synchronize()
  try:plan.submit(Frame(6,code))
  except Exception as actual:
   assert type(actual) is type(expected),(type(actual),type(expected))
   assert str(actual)==str(expected),(str(actual),str(expected))
  else:raise AssertionError('model guard passed')
  cp.cuda.Device().synchronize()
  assert np.array_equal(before,cp.asnumpy(out))
  receipt['checks'].append({'kind':label,'exception_type':type(expected).__name__,
    'exception_text':str(expected),'downstream_different_words':0,
    'native_guard_count':sum(op['kind']==11 for op in plan.describe()['operations'])})
 finally:
  cp.asnumpy=original_asnumpy;close(plan)

ist=cp.ones(8,cp.int32)
model_case('noahmp_ice_admission',lambda:_refuse_unadmitted(x,ist),1,1)
fields=cp.asarray(np.tile(np.array([1.25,-2.5,3.75],np.float32),(8,1)))
slots={'errsw':0,'erreng':1,'errwat':2}
for code in (1,2,3):
 model_case(f'noahmp_balance_code{code}',lambda:_raise_for_status(cp,x,fields,slots),4,code)

# More than one result in a guard scope cannot accidentally validate the last.
plan=native.Plan()
try:
 try:
  with plan.record(make,x):
   make((1,),(8,),(x,np.int32(-1),np.int32(0)))
   with plan.zero_guard(lambda i,v:ValueError('bad'),word_bytes=4):
    first=cp.asnumpy(x);second=cp.asnumpy(x)
 except native.RecordingError as error:
  assert 'host scope transfer count mismatch' in str(error),str(error)
 else:raise AssertionError('multi-read scope accepted')
 receipt['checks'].append({'kind':'ambiguous_multi_read','replayable':False})
finally:close(plan)

# A conditional scope must keep read, synchronization and guard together.
plan=native.Plan()
try:
 try:
  with plan.record(make,x):
   make((1,),(8,),(x,np.int32(-1),np.int32(0)))
   with plan.zero_guard(lambda i,v:ValueError('bad'),word_bytes=4):
    with plan.when(require=1):
     cp.asnumpy(x)
 except native.RecordingError:pass
 else:raise AssertionError('predicate mutation inside host scope accepted')
 receipt['checks'].append({'kind':'host_scope_predicate_mutation','replayable':False})
finally:close(plan)

# A high 64-bit status word is preserved without int32 truncation.
wide=cp.zeros(8,cp.uint64)
plan=native.Plan()
try:
 with plan.record(add,wide,out):
  with plan.zero_guard(lambda i,v:ValueError(f'wide {i}/{v}'),word_bytes=8):
   cp.asnumpy(wide)
  add((1,),(8,),(out,))
 wide[5]=np.uint64(1<<40)
 before=cp.asnumpy(out)
 try:plan.submit()
 except ValueError as error:assert str(error)==f'wide 5/{1<<40}',str(error)
 else:raise AssertionError('wide guard accepted')
 assert np.array_equal(before,cp.asnumpy(out))
 receipt['checks'].append({'kind':'wide_status','index':5,'value':1<<40,'downstream_different_words':0})
finally:close(plan)

receipt.update(status='PASS',wall_s=time.perf_counter()-started)
(ROOT/'receipts/host-contracts.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(receipt['checks']),'wall_s':receipt['wall_s']}))
