"""CUDA native health consumers: IEEE words, selected slots, clear-before-raise."""
import ctypes as C
import hashlib,importlib.util,json,os,sys
from pathlib import Path
import cupy as cp
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('native_step',ROOT/'gpuwm/native_step/__init__.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
from whole_health_annotations import annotate_health
from gpuwm.core.health_ledger import HealthLedger
receipt={'purpose':__doc__,'pid':os.getpid(),'checks':[],'source':{}}
for path in ('gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/whole_health_annotations.py','tools/native_step/probe_health_consumers.py','gpuwm/core/health_ledger.py','receipts/libawstep-health.so'):
 receipt['source'][path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
add=cp.RawKernel('extern "C" __global__ void add(int*x){x[0]+=1;}','add')
out=cp.zeros(1,cp.int32);add((1,),(1,),(out,));cp.cuda.Device().synchronize()
def close(plan):cp.cuda.Device().synchronize();plan.close()
for dtype,udtype in ((np.float32,np.uint32),(np.float64,np.uint64)):
 width=np.dtype(dtype).itemsize;mask=0x27
 values=np.array([0.,-0.,np.nextafter(dtype(0),dtype(1)),np.nan,np.inf,np.finfo(dtype).max,np.nan,-np.inf],dtype=dtype)
 x=cp.asarray(values)
 plan=native.Plan()
 try:
  with plan.record(x,out,add):
   with plan.finite_guard(lambda i,b:ValueError(f'{i}/{b}'),word_bytes=width,mask=mask):cp.asnumpy(x)
   add((1,),(1,),(out,))
  plan.submit();cp.cuda.Device().synchronize()
  receipt['checks'].append({'kind':'finite_selected_words','dtype':str(np.dtype(dtype)),'signed_zero_subnormal_max_accepted':True,'unselected_nonfinite_ignored':True})
 finally:close(plan)
 for index,value in ((0,np.nan),(1,np.inf),(2,-np.inf),(5,np.nan)):
  x=cp.asarray(values);plan=native.Plan()
  try:
   with plan.record(x,out,add):
    with plan.finite_guard(lambda i,b:ValueError(f'{i}/{b}'),word_bytes=width,mask=mask):cp.asnumpy(x)
    add((1,),(1,),(out,))
   x[index]=dtype(value);cp.cuda.Device().synchronize();before=cp.asnumpy(out)
   expected_bits=int(np.asarray(value,dtype=dtype).view(udtype))
   try:plan.submit()
   except ValueError as error:assert str(error)==f'{index}/{expected_bits}',(str(error),expected_bits)
   else:raise AssertionError('nonfinite value accepted')
   assert np.array_equal(cp.asnumpy(out),before)
   receipt['checks'].append({'kind':'finite_failure','dtype':str(np.dtype(dtype)),'index':index,'bits':expected_bits,'downstream_differences':0})
  finally:close(plan)

# Actual ledger method: its defined clear happens before a refusal.
def describe(value):raise FloatingPointError(f'fixture health flags {value:#x}')
for bad in (False,True):
 ledger=HealthLedger();ledger._index={'fixture':0};ledger._describe={'fixture':describe}
 plan=native.Plan();annotations=[]
 try:
  with plan.record(ledger,out,add):
   with annotate_health(plan,annotations):ledger.drain()
   add((1,),(1,),(out,))
  ledger._slots[63]=np.uint32(0xdeadbeef)
  if bad:ledger._slots[0]=np.uint32(5)
  # Obtain the exact ordinary refusal from the original method, then restore.
  before_slots=ledger._slots.copy()
  expected=None
  try:ledger.drain()
  except Exception as error:expected=error
  ledger._slots[:]=before_slots;before=cp.asnumpy(out)
  try:plan.submit()
  except Exception as actual:
   assert expected is not None and type(actual) is type(expected) and str(actual)==str(expected)
  else:assert expected is None
  cp.cuda.Device().synchronize();assert not np.any(cp.asnumpy(ledger._slots))
  delta=int(cp.asnumpy(out)[0]-before[0]);assert delta==(0 if bad else 1)
  receipt['checks'].append({'kind':'ledger_clear_and_refusal','bad':bad,'unused_nonzero_ignored':True,'all_slots_cleared':True,'error':str(expected) if expected else None,'downstream_increment':delta,'annotations':annotations})
 finally:close(plan)

x=cp.zeros(8,cp.float32);plan=native.Plan()
try:
 try:
  with plan.record(x):
   with plan.finite_guard(lambda i,b:ValueError('bad'),word_bytes=4,mask=1<<8):cp.asnumpy(x)
 except native.RecordingError as error:assert 'guard mask exceeds result words' in str(error)
 else:raise AssertionError('out of range mask accepted')
 receipt['checks'].append({'kind':'mask_out_of_range','replayable':False})
finally:close(plan)
receipt['status']='PASS';(ROOT/'receipts/health-consumers-contract.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(receipt['checks'])}))
