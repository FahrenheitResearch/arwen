"""Compare shape variants on the first real AC root and nested launches.

This is a word/resource check with explicit copies and synchronizations. It is
never a model-throughput timing run. Baseline outputs continue the trajectory.
"""
import json,os,sys,time,hashlib,statistics,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'evidence/jax-shape/local'
INPUT=json.loads((RUN/'input-admission.json').read_text())['bindings']
sys.path.insert(0,str(ROOT))
import cupy as cp
free,total=cp.cuda.runtime.memGetInfo()
if free < 14*1024**3: raise RuntimeError(f'local component needs 14GiB headroom; free={free}')
cp.get_default_memory_pool().set_limit(size=12*1024**3)
from gpuwm.core import kernels
from gpuwm.core.kernel_shape import FUNCTIONS
from gpuwm.numerics import receipt as numerics_receipt,effective_options
from gpuwm.certify.kernel_manifest import kernel_manifest
from cupy.cuda import compiler
compile_calls=[]
real_compile=compiler._NVRTCProgram.compile
def noted_compile(self, options=(), log_stream=None):
 compile_calls.append({'effective_options':list(effective_options(options))})
 return real_compile(self,options,log_stream)
compiler._NVRTCProgram.compile=noted_compile
hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,memory.used','--format=csv,noheader'],text=True).strip()
processes_before=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory,process_name','--format=csv,noheader'],text=True).strip()
PROFILE=os.environ.get('SHAPE_PROFILE')=='1'
profile_started=False
period=0;counts={}
original=kernels.get_kernel
seen=set();rows=[]
mode=os.environ.get('GPUWM_NUMERICS','fast')
receipt=RUN/(('profile-' if PROFILE else 'timing-')+mode+'.json')
started=time.perf_counter()
def save(status):
 receipt.write_text(json.dumps(dict(status=status,numerics=numerics_receipt(),wrf_exact=os.environ.get('GPUWM_WRF_EXACT'),seconds=time.perf_counter()-started,hardware=hardware,processes_before=processes_before,pool_limit_bytes=12*1024**3,pool_total_bytes=cp.get_default_memory_pool().total_bytes(),inputs=INPUT,compile_calls=compile_calls,kernel_manifest=kernel_manifest(),counts_second_period=[dict(module=k[0],function=k[1],shape=k[2],count=v) for k,v in counts.items()],rows=rows),indent=2)+'\n')
class Checked:
 def __init__(self,base,candidate,key): self.base,self.candidate,self.key=base,candidate,key
 def __getattr__(self,name): return getattr(self.candidate,name)
 def __call__(self,grid,block,args,**kwargs):
  if period==1: counts[self.key]=counts.get(self.key,0)+1
  if self.key in seen: return self.base(grid,block,args,**kwargs)
  seen.add(self.key)
  unique={}
  for a in args:
   if isinstance(a,cp.ndarray): unique[(a.data.ptr,a.shape,a.strides)]=a
  arrays=list(unique.values())
  before=[a.copy() for a in arrays]
  global profile_started
  if PROFILE and not profile_started:
   cp.cuda.runtime.deviceSynchronize()
   cp.cuda.profiler.start();profile_started=True
  label=self.key[1]+':'+','.join(str(x) for x in self.key[2])
  if PROFILE: cp.cuda.nvtx.RangePush('shape.before:'+label)
  self.base(grid,block,args,**kwargs)
  if PROFILE: cp.cuda.nvtx.RangePop()
  after=[a.copy() for a in arrays]
  for a,b in zip(arrays,before): cp.copyto(a,b)
  if PROFILE: cp.cuda.nvtx.RangePush('shape.after:'+label)
  self.candidate(grid,block,args,**kwargs)
  if PROFILE: cp.cuda.nvtx.RangePop()
  array_counts=[];differences=0;words=0;max_abs=0.0
  for a,b in zip(arrays,after):
   av=cp.ascontiguousarray(a).view(cp.uint8);bv=cp.ascontiguousarray(b).view(cp.uint8)
   diff=int(cp.count_nonzero(av!=bv).item());differences+=diff;words+=a.nbytes//4
   if diff and a.dtype==cp.float32:
    finite=cp.isfinite(a)&cp.isfinite(b)
    delta=float(cp.max(cp.where(finite,cp.abs(a-b),0)).item())
    max_abs=max(max_abs,delta)
   array_counts.append(dict(shape=a.shape,dtype=str(a.dtype),different_bytes=diff))
  samples={'before':[],'after':[]}
  if not differences and mode=='fast' and not PROFILE:
   stream=kwargs.get('stream') or cp.cuda.get_current_stream()
   for repeat in range(10):
    order=(('before',self.base),('after',self.candidate)) if repeat%2==0 else (('after',self.candidate),('before',self.base))
    for label,kernel in order:
     for a,b in zip(arrays,before): cp.copyto(a,b)
     first,last=cp.cuda.Event(),cp.cuda.Event()
     first.record(stream);kernel(grid,block,args,**kwargs);last.record(stream);last.synchronize()
     if repeat: samples[label].append(float(cp.cuda.get_elapsed_time(first,last)))
  for a,b in zip(arrays,after): cp.copyto(a,b)
  row=dict(module=self.key[0],function=self.key[1],shape=self.key[2],grid=grid,block=block,words=words,different_bytes=differences,max_abs=max_abs,before=self.base.attributes,after=self.candidate.attributes,arrays=array_counts,samples_ms=samples,median_ms={label:statistics.median(values) if values else None for label,values in samples.items()})
  rows.append(row);save('running')
  print(json.dumps({k:row[k] for k in ('function','shape','words','different_bytes','max_abs')}),flush=True)
  if differences: raise RuntimeError('shape words differ: '+str(self.key))
  return None
handles={}
def checked(name,func,shape=()):
 if not shape or func not in FUNCTIONS.get(name,()): return original(name,func,shape)
 key=(name,func,shape)
 if key not in handles: handles[key]=Checked(original(name,func),original(name,func,shape),key)
 return handles[key]
kernels.get_kernel=checked
from gpuwm.core import model
execute=model.execute_experiment
class Finished(Exception): pass
def bounded(*args,**kwargs):
 callback=kwargs.get('progress_callback')
 def progress(**event):
  global period
  if callback: callback(**event)
  if event.get('phase')=='post-d01-sync':
   period=event.get('outer_step',0)
   if period>=2: raise Finished()
 kwargs['progress_callback']=progress
 return execute(*args,**kwargs)
model.execute_experiment=bounded
from gpuwm.prepared_domain_tree_forecast import main
command=['--prepared-root',INPUT['prepared_root'],'--preparation-receipt-sha256',INPUT['preparation_receipt_sha256'],'--experiment-config',INPUT['experiment_config'],'--experiment-config-sha256',INPUT['experiment_config_sha256'],'--devices','1','--progress-format','jsonl','--io-mode','history','--outdir',str(ROOT/'.shape-local'/(('profile-' if PROFILE else 'replay-')+mode))]
try:
 result=main(command)
 if result: raise RuntimeError('native main returned '+str(result))
except Finished:
 if profile_started:
  cp.cuda.runtime.deviceSynchronize();cp.cuda.profiler.stop()
 save('bounded_complete')
except BaseException:
 save('failed');raise
else: save('completed_full')
