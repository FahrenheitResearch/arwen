"""Native fixed-time AC domain-step replay, not a changing-time adapter."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--numerics',choices=('strict','fast'),required=True)
a=p.parse_args();binding=json.loads((ROOT/'receipts/local-binding.json').read_text())
mask=os.environ['CUDA_VISIBLE_DEVICES'];os.environ.update(binding['runtime_env'])
os.environ.update(CUDA_VISIBLE_DEVICES=mask,GPUWM_NUMERICS=a.numerics)
for key,relative in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','CUPY_CACHE_DIR':'cache/cupy','CUDA_CACHE_PATH':'cache/cuda','GPUWM_GENERATED_TABLE_ROOT':'cache/generated'}.items():
 path=ROOT/relative;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
sys.path[:0]=binding['runtime_env']['PYTHONPATH'].split(':')
import cupy as cp
import numpy as np
from gpuwm.core import dycore
from gpuwm.prepared_domain_tree_forecast import main as run
from gpuwm.native_step import Plan
from host_annotations import annotate
class Complete(BaseException):pass
receipt={'purpose':__doc__,'pid':os.getpid(),'numerics':a.numerics,'binding':binding,'cases':[], 'source':{},'completed_forecast':False}
for name in ('gpuwm/core/dycore.py','gpuwm/core/physics.py','gpuwm/core/noahmp_column_slab.py','gpuwm/core/noahmp_flux_slab.py','gpuwm/core/kernels/noahmp_vegeflux.cu','gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/host_annotations.py','tools/native_step/probe_domain_step.py'):
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()

def inventory(state):
 seen=set();arrays={}
 def visit(value,label):
  if id(value) in seen:return
  seen.add(id(value))
  if isinstance(value,cp.ndarray):
   key=(value.data.ptr,value.shape,value.strides,value.dtype.str)
   arrays.setdefault(key,{'label':label,'array':value});return
  if isinstance(value,dict):
   for name,item in value.items():visit(item,label+'.'+str(name))
  elif isinstance(value,(list,tuple)):
   for i,item in enumerate(value):visit(item,label+f'[{i}]')
  elif type(value).__module__.startswith('gpuwm.') and hasattr(value,'__dict__'):
   for name,item in vars(value).items():visit(item,label+'.'+name)
 visit(state,'state')
 return arrays

def snapshot(arrays):
 return {key:entry['array'].copy() for key,entry in arrays.items()}
def restore(arrays,saved):
 for key,entry in arrays.items():cp.copyto(entry['array'],saved[key])
 cp.cuda.get_current_stream().synchronize()
def compare(arrays,reference):
 total=changed=0;details=[]
 for key,entry in arrays.items():
  array=entry['array'];got=cp.asnumpy(array);want=cp.asnumpy(reference[key]);width=got.dtype.itemsize
  g=np.frombuffer(got.tobytes(),np.uint8).reshape(-1,width);w=np.frombuffer(want.tobytes(),np.uint8).reshape(-1,width)
  n=int(np.count_nonzero(np.any(g!=w,axis=1)))
  total+=got.size;changed+=n
  details.append({'name':entry['label'],'shape':list(got.shape),'dtype':got.dtype.str,'words':got.size,'different_words':n,'expected_sha256':hashlib.sha256(want.tobytes()).hexdigest(),'actual_sha256':hashlib.sha256(got.tobytes()).hexdigest()})
 return {'words':total,'different_words':changed,'arrays':details}
original=dycore.step;counts={}
def step(state,cfg,**kwargs):
 key=(cfg.nx,cfg.ny,cfg.nz);counts[key]=counts.get(key,0)+1
 if not ((key==(120,70,44) and counts[key]==2) or (key==(267,117,44) and counts[key]==4)):
  return original(state,cfg,**kwargs)
 entry={'shape':list(key),'time_before':state.elapsed_seconds,'host_annotations':[],'kwargs':{k:v for k,v in kwargs.items() if isinstance(v,(str,int,float,bool,type(None)))}}
 receipt['cases'].append(entry)
 before=inventory(state);nbytes=sum(v['array'].nbytes for v in before.values())
 entry.update(input_arrays=len(before),input_bytes=nbytes)
 if nbytes*3>12*1024**3:raise RuntimeError(f'snapshot budget exceeded: {nbytes}')
 saved=snapshot(before);cp.cuda.get_current_stream().synchronize()
 plan=Plan();plan.trace_failures()
 try:
  with plan.record(state,cfg):
   with annotate(plan,entry['host_annotations'],layout_guard=True):original(state,cfg,**kwargs)
  cp.cuda.get_current_stream().synchronize()
  after=inventory(state);expected=snapshot(after)
  entry.update(time_after=state.elapsed_seconds,output_arrays=len(after),new_array_views=[v['label'] for k,v in after.items() if k not in before],plan=plan.describe())
  restore(before,saved)
  # Preserve Python's one-step metadata from capture. This probe verifies
  # the captured CUDA computation only; it does not advance time natively.
  plan.submit();cp.cuda.get_current_stream().synchronize()
  entry['compare']=compare(after,expected)
  entry['submissions']=1
  # Restore the known ordinary result before continuing to the next domain.
  restore(after,expected)
  assert entry['compare']['different_words']==0,[(r['name'],r['different_words']) for r in entry['compare']['arrays'] if r['different_words']]
  print(json.dumps({k:v for k,v in entry.items() if k not in ('plan','compare')}|{'words':entry['compare']['words'],'differences':entry['compare']['different_words']}),flush=True)
 finally:
  cp.cuda.get_current_stream().synchronize();plan.close()
 if len(receipt['cases'])==2:raise Complete()

dycore.step=step
command=['--prepared-root',binding['prepared_root'],binding['proof_option'],binding['prepared_head_sha256'],'--experiment-config',binding['config'],'--experiment-config-sha256',binding['config_sha256'],'--progress-format','jsonl','--render-products','none','--io-mode','none','--outdir',str(ROOT/('domain-step-check-'+str(time.time_ns())))]
receipt['argv']=command;started=time.perf_counter()
try:
 run(command);raise RuntimeError('did not reach parent and child')
except Complete:receipt['status']='PASS_FIXED_TIME_DOMAIN_REPLAY_ONLY'
except BaseException as error:
 receipt.update(status='FAILED',error=repr(error));raise
finally:
 receipt['wall_s']=time.perf_counter()-started
 (ROOT/f'receipts/domain-step-{a.numerics}.json').write_text(json.dumps(receipt,indent=2)+'\n')
