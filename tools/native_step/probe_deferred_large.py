"""Large concatenation descriptor upload with native ordered radiation guard."""
import ast,copy,hashlib,json,os
from pathlib import Path
import cupy as cp
import numpy as np
from gpuwm.native_step import Plan,RecordingError
from gpuwm.core import deferred_device_checks as checks
from deferred_annotations import annotate_deferred,PINS,SITES,validate_sources
ROOT=Path(__file__).resolve().parents[2];validate_sources(ROOT)
def verdict(site):
 filename,name,line=site;path=ROOT/'gpuwm/core'/filename
 tree=ast.parse(path.read_text());matches=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name and n.lineno==line];assert len(matches)==1
 module=ast.Module(body=[copy.deepcopy(matches[0])],type_ignores=[]);ns={'np':np,'name':'re_cloud'}
 exec(compile(module,str(path),'exec'),ns);return ns[name]
sites=[next(iter(SITES))]*130;functions={site:verdict(site) for site in sites}
add=cp.RawKernel('extern "C" __global__ void downstream(int*x){x[0]+=1;}','downstream');out=cp.zeros(1,cp.int32);add((1,),(1,),(out,));cp.cuda.Device().synchronize()
r={'purpose':__doc__,'pid':os.getpid(),'checks':[],'source':{}}
for name in ('gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/deferred_annotations.py','tools/native_step/probe_deferred_large.py','receipts/libawstep-numeric.so'):
 r['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
r['verdict_source']=PINS

def run_case(label,changes):
 # Include all six original functions, plus a 130-word LW flag vector.
 values=[]
 for site in sites:
  if site[0]=='rrtmg_lw.py':v=cp.zeros(130,cp.int32)
  elif site[1]=='_cldprmc_verdict':v=cp.zeros(1,cp.int32)
  elif SITES[site]==2:v=cp.asarray(np.float32(-np.inf))
  else:v=cp.asarray(np.bool_(False))
  values.append(v)
 def ordinary():
  with checks.deferred():
   for site,v in zip(sites,values):checks.check(v,functions[site])
 plan=Plan();annotations=[]
 try:
  with plan.record(values,out,add):
   with annotate_deferred(plan,annotations):ordinary()
   add((1,),(1,),(out,))
  for i,index,value in changes:values[i].reshape(-1)[index]=value
  expected=None
  try:ordinary()
  except Exception as e:expected=e
  before=cp.asnumpy(out);actual=None
  try:plan.submit()
  except Exception as e:actual=e
  cp.cuda.Device().synchronize();after=cp.asnumpy(out)
  if expected is None:
   assert actual is None,repr(actual);assert int(after[0])==int(before[0])+1
  else:
   assert type(actual) is type(expected) and str(actual)==str(expected),(repr(actual),repr(expected))
   assert np.array_equal(before,after),'downstream executed after failure'
  r['checks'].append({'case':label,'ordinary_exception':None if expected is None else {'type':type(expected).__name__,'text':str(expected)},'native_exception':None if actual is None else {'type':type(actual).__name__,'text':str(actual)},'downstream_delta':int(after[0]-before[0]),'annotations':annotations,'plan':plan.describe()})
 finally:cp.cuda.Device().synchronize();plan.close()

try:
 run_case('healthy_130_groups',[])
 run_case('last_group_failure',[(129,129,7)])
 r['status']='PASS'
except BaseException as error:r.update(status='FAILED',error=repr(error));raise
finally:
 (ROOT/'receipts/deferred-contract-large.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({'status':r['status'],'checks':len(r['checks'])}))
