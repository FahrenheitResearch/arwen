"""Exact original validator errors, selected sentinel and downstream stop."""
import dataclasses,hashlib,json,os
from pathlib import Path
import cupy as cp
import numpy as np
from gpuwm.native_step import Plan
from gpuwm.core.health import StateHealthValidator,HealthField,FieldRule
from whole_health_annotations import annotate_health
from validator_annotations import annotate_descriptors,error_factory,original_decoder
ROOT=Path(__file__).resolve().parents[2]
class State:
 def __init__(self):self.buffers={}
 def scratch(self,shape,name):
  if name not in self.buffers:self.buffers[name]=cp.empty(shape,cp.float32)
  return self.buffers[name]
add=cp.RawKernel('extern "C" __global__ void add(int*x){x[0]+=1;}','add');out=cp.zeros(1,cp.int32);add((1,),(1,),(out,));cp.cuda.Device().synchronize()
receipt={'purpose':__doc__,'pid':os.getpid(),'checks':[],'source':{}}
for name in ('gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/whole_health_annotations.py','tools/native_step/validator_annotations.py','tools/native_step/probe_validator.py','gpuwm/core/health.py','receipts/libawstep-value.so'):
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
for mode in ('plain','direct','level'):
 x=cp.ones((2,2,2),cp.float32);aux=None if mode=='plain' else cp.ones((2,) if mode=='level' else x.shape,cp.float32)
 field=HealthField('fixture',x,FieldRule('generic',lower=0,upper=4),aux,None if aux is None else mode,4 if mode=='level' else 0)
 validator=StateHealthValidator(State(),field_provider=lambda:(field,));validator.require_healthy(phase='fixture')
 plan=Plan();annotations=[]
 try:
  with plan.record(validator,x,aux,out,add):
   with annotate_descriptors(plan,annotations),annotate_health(plan,annotations):validator.require_healthy(phase='fixture')
   add((1,),(1,),(out,))
  if mode=='plain':x[1,0,1]=np.float32(-2)
  elif mode=='direct':aux[1,0,1]=np.float32(7)
  else:aux[1]=np.float32(7)
  try:validator.require_healthy(phase='fixture')
  except Exception as error:expected=error
  else:raise AssertionError('ordinary validator did not reject corruption')
  before=cp.asnumpy(out)
  try:plan.submit()
  except Exception as error:actual=error
  else:raise AssertionError('native validator did not reject corruption')
  cp.cuda.Device().synchronize()
  assert type(actual) is type(expected) and str(actual)==str(expected),(repr(actual),repr(expected))
  assert dataclasses.asdict(actual.report)==dataclasses.asdict(expected.report)
  assert np.array_equal(before,cp.asnumpy(out))
  receipt['checks'].append({'case':mode,'exception':str(actual),'report':dataclasses.asdict(actual.report),'downstream_differences':0,'annotations':annotations})
 finally:cp.cuda.Device().synchronize();plan.close()
# Pin the unusual allowed status value and exercise the invalid field branch.
validator._result.set(np.asarray([9,(1<<64)-1],np.uint64).view(np.float32))
plan=Plan()
try:
 with plan.record(validator,out,add):
  with plan.value_guard(error_factory(validator,'sentinel'),word_bytes=8,mask=2,expected=(1<<64)-1):cp.asnumpy(validator._result)
  add((1,),(1,),(out,))
 before=int(cp.asnumpy(out)[0]);plan.submit();cp.cuda.Device().synchronize();assert int(cp.asnumpy(out)[0])==before+1
 report=original_decoder()(validator,np.array([9,(1<<64)-1],np.uint64),'sentinel');assert report.ok and report.status_bits==9
 receipt['checks'].append({'case':'nonzero_status_with_no_failure','accepted':True,'status_bits':9})
 validator._result.set(np.asarray([9,5<<48],np.uint64).view(np.float32));before=cp.asnumpy(out)
 try:plan.submit()
 except RuntimeError as error:assert str(error)=='health kernel returned invalid field id 5'
 else:raise AssertionError('invalid field accepted')
 assert np.array_equal(before,cp.asnumpy(out));receipt['checks'].append({'case':'invalid_field','error':'health kernel returned invalid field id 5','downstream_differences':0})
finally:cp.cuda.Device().synchronize();plan.close()
receipt['status']='PASS';(ROOT/'receipts/validator-contract.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':'PASS','checks':len(receipt['checks'])}))
