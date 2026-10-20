"""Original model stability refusal versus fixed-time native guard and stale-clock control."""
import ast,hashlib,importlib.util,json,math,os,types
from pathlib import Path
import cupy as cp
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
from gpuwm.native_step import Plan
from gpuwm.core import dycore
from whole_health_annotations import annotate_health
spec=importlib.util.spec_from_file_location('before_step_number',ROOT/'receipts/whole-health-annotations-pre-step-number.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
# Execute the exact validation block from the original nested on_step. This
# retains its original step_health call, branch order and exception wording.
tree=ast.parse((ROOT/'gpuwm/core/model.py').read_text())
function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='on_step')
gate=next(n for n in function.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='validate_state') if any(isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='validate_state' for n in function.body) else next(n for n in ast.walk(function) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='validate_state')
wrapper=ast.parse('def on_step(node,clock,grid_id):\n steppers={}\n').body[0]
wrapper.body.extend(gate.body);module=ast.Module(body=[wrapper],type_ignores=[]);ast.fix_missing_locations(module)
namespace={'math':math};exec(compile(module,str(ROOT/'gpuwm/core/model.py'),'exec'),namespace);on_step=namespace['on_step']
class State:
 def __init__(self):
  self.u=cp.ones((2,4,5),cp.float32);self.w=cp.ones((3,4,4),cp.float32);self.thp=cp.ones((2,4,4),cp.float32)
  self.php=cp.zeros((3,4,4),cp.float32);self.phb=cp.asarray(np.arange(3,dtype=np.float32)*np.float32(100)*np.float32(dycore.c.G));self.buffers={}
 def scratch(self,shape,name):
  if name not in self.buffers:self.buffers[name]=cp.empty(shape,cp.float32)
  return self.buffers[name]
node=types.SimpleNamespace(state=State(),cfg=types.SimpleNamespace(run=types.SimpleNamespace(dt=18.,dx=3000.,spec_bdy_width=1)))
clock=types.SimpleNamespace(step_count=7)
on_step(node,clock,2)
add=cp.RawKernel('extern "C" __global__ void add(int*x){x[0]+=1;}','add');out=cp.zeros(1,cp.int32);add((1,),(1,),(out,));cp.cuda.Device().synchronize()
receipt={'purpose':__doc__,'pid':os.getpid(),'checks':[],'source':{}}
for name in ('gpuwm/core/model.py','gpuwm/core/dycore.py','gpuwm/core/streaming.py','gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/whole_health_annotations.py','tools/native_step/probe_stability_consumer.py','receipts/whole-health-annotations-pre-step-number.py','receipts/libawstep-health.so'):
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
for problem in ('nonfinite_state','collapsed_layer'):
 row={'problem':problem};expected=None
 for label,adapter in [('before',old.annotate_health),('candidate',annotate_health)]:
  node.state.u.fill(1);node.state.php.fill(0);clock.step_count=7;plan=Plan();annotations=[]
  try:
   with plan.record(node,out,add):
    with adapter(plan,annotations):on_step(node,clock,2)
    add((1,),(1,),(out,))
   if problem=='nonfinite_state':node.state.u[0,1,1]=np.float32(np.nan)
   else:node.state.php[1,:,:]=-node.state.phb[1]
   try:on_step(node,clock,2)
   except RuntimeError as error:expected=error
   else:raise AssertionError('ordinary stability gate accepted the negative control')
   clock.step_count=100;before=cp.asnumpy(out)
   try:plan.submit()
   except RuntimeError as error:actual=error
   else:raise AssertionError('native stability consumer accepted the negative control')
   cp.cuda.Device().synchronize();assert np.array_equal(before,cp.asnumpy(out))
   same=type(actual) is type(expected) and str(actual)==str(expected)
   if label=='candidate':assert same,(str(actual),str(expected))
   else:assert not same and 'step 101' in str(actual),(str(actual),str(expected))
   row[label]={'same_original_exception':same,'message':str(actual),'downstream_different_words':0,'annotations':annotations}
  finally:cp.cuda.Device().synchronize();plan.close()
 row['original_message']=str(expected);receipt['checks'].append(row)
receipt['status']='PASS';(ROOT/'receipts/stability-consumer-contract.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':'PASS','checks':len(receipt['checks'])}))
