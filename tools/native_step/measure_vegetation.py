"""Actual AC land-column before/after and live-mask native replay proof."""
import argparse,hashlib,importlib.util,json,os,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--numerics',choices=('strict','fast'),required=True);p.add_argument('--profile-only',action='store_true')
a=p.parse_args();binding=json.loads((ROOT/'receipts/local-binding.json').read_text())
mask=os.environ['CUDA_VISIBLE_DEVICES'];os.environ.update(binding['runtime_env'])
os.environ.update(CUDA_VISIBLE_DEVICES=mask,GPUWM_NUMERICS=a.numerics)
for key,relative in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','CUPY_CACHE_DIR':'cache/cupy','CUDA_CACHE_PATH':'cache/cuda','GPUWM_GENERATED_TABLE_ROOT':'cache/generated'}.items():
 path=ROOT/relative;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
sys.path[:0]=binding['runtime_env']['PYTHONPATH'].split(':')
import cupy as cp
import numpy as np
from gpuwm.core import dycore,noahmp_column_slab as column,noahmp_flux_slab as flux
from gpuwm.core.noahmp_slab_libm import take_slab_rows
from gpuwm.prepared_domain_tree_forecast import main as run
from gpuwm.native_step import Plan
spec=importlib.util.spec_from_file_location('gpuwm.core._vegetation_base',ROOT/'receipts/vegetation-base/noahmp_column_slab.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
receipt={'purpose':__doc__,'pid':os.getpid(),'numerics':a.numerics,'binding':binding,'cases':[], 'source':{},'completed_forecast':False}
for name in ('gpuwm/core/noahmp_column_slab.py','gpuwm/core/noahmp_flux_slab.py','gpuwm/core/kernels/noahmp_vegeflux.cu','gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/measure_vegetation.py','receipts/vegetation-base/noahmp_column_slab.py'):
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
class Complete(BaseException):pass
original_step=dycore.step;original_column=column.evaluate_sflx_slab
original_masked=column.evaluate_vege_flux_masked_slab
counts={};active=None;measured=set();component=None

def bits(mapping):return {name:cp.asnumpy(v) for name,v in mapping.items()}
def compare(got,want):
 total=changed=0;detail=[]
 assert set(got)==set(want),(set(got),set(want))
 for name,value in got.items():
  reference=want[name];host=cp.asnumpy(value)
  assert host.shape==reference.shape and host.dtype==reference.dtype,(name,host.shape,reference.shape)
  width=host.dtype.itemsize
  diff=int(np.count_nonzero(np.any(np.frombuffer(host.tobytes(),np.uint8).reshape(-1,width)!=np.frombuffer(reference.tobytes(),np.uint8).reshape(-1,width),axis=1)))
  total+=host.size;changed+=diff
  if diff:detail.append({'name':name,'words':diff})
 return {'words':total,'different_words':changed,'differences':detail}

def old_leaf(fields,n,live):
 index=cp.nonzero(live)[0]
 partial=flux.evaluate_vege_flux_slab(take_slab_rows(index,fields,n),int(index.size))
 result={}
 for name,value in partial.items():
  full=cp.zeros(n,cp.float32);full[index]=value;result[name]=full
 return result

def masked(fields,n,live,**kwargs):
 if component is None:return original_masked(fields,n,live,**kwargs)
 # Every fixture comes directly from this actual domain step's column call.
 live=live.copy();natural=live.copy()
 component['land_columns']=n;component['natural_active']=int(cp.count_nonzero(live))
 original_masked(fields,n,live,**kwargs);old_leaf(fields,n,live)
 cp.cuda.get_current_stream().synchronize()
 plan=Plan()
 try:
  with plan.record(fields,live):recorded=original_masked(fields,n,live,**kwargs)
  plan.resources.append(recorded)
  patterns=[('natural',natural),('empty',cp.zeros_like(natural)),('alternating',natural&(cp.arange(n)%2==0)),('natural_again',natural)]
  checks=[]
  for label,pattern in patterns:
   cp.copyto(live,pattern)
   want=bits(old_leaf(fields,n,live))
   plan.submit();cp.cuda.get_current_stream().synchronize()
   result=compare(recorded,want);assert result['different_words']==0,(label,result)
   checks.append({'mask':label,'active':int(cp.count_nonzero(live)),**result})
  component['native_live_mask_checks']=checks
  component['native_kernels']=plan.describe()['kernel_launches']
  component['native_operations']=len(plan.describe()['operations'])
  component['native_pool_reserved_bytes']=plan.describe()['pool_reserved_bytes']
 finally:
  cp.cuda.get_current_stream().synchronize();plan.close()
 return original_masked(fields,n,natural,**kwargs)

def assembled(fields,n,**kwargs):
 global component
 if active is None or active in measured:return original_column(fields,n,**kwargs)
 entry={'domain':list(active),'n':n,'samples_ms':{'base':[],'candidate':[]}}
 # These routines return independent arrays. Verify both preserve inputs.
 before=bits(fields)
 want=bits(base.evaluate_sflx_slab(fields,n,**kwargs))
 component=entry
 try:got=original_column(fields,n,**kwargs)
 finally:component=None
 entry['whole_column_words']=compare(got,want)
 entry['input_words']=compare(fields,before)
 assert entry['whole_column_words']['different_words']==0,entry
 assert entry['input_words']['different_words']==0,entry
 if a.profile_only:
  for arm in ('base','candidate'):
   cp.cuda.get_current_stream().synchronize()
   cp.cuda.nvtx.RangePush(f'vegetation/{active[0]}x{active[1]}/{arm}')
   result=(base.evaluate_sflx_slab if arm=='base' else original_column)(fields,n,**kwargs)
   cp.cuda.get_current_stream().synchronize();cp.cuda.nvtx.RangePop()
 else:
  for repeat in range(9):
   for arm in (('base','candidate') if repeat%2==0 else ('candidate','base')):
    cp.cuda.get_current_stream().synchronize();start=time.perf_counter_ns()
    result=(base.evaluate_sflx_slab if arm=='base' else original_column)(fields,n,**kwargs)
    cp.cuda.get_current_stream().synchronize()
    entry['samples_ms'][arm].append((time.perf_counter_ns()-start)/1e6)
  entry['mean_ms']={k:statistics.mean(v) for k,v in entry['samples_ms'].items()}
  entry['median_ms']={k:statistics.median(v) for k,v in entry['samples_ms'].items()}
 receipt['cases'].append(entry);measured.add(active)
 print(json.dumps(entry),flush=True)
 return got

def step(state,cfg,**kwargs):
 global active
 key=(cfg.nx,cfg.ny,cfg.nz);counts[key]=counts.get(key,0)+1
 active=key if ((key==(120,70,44) and counts[key]==2) or (key==(267,117,44) and counts[key]==4)) else None
 result=original_step(state,cfg,**kwargs)
 if len(measured)==2:raise Complete()
 return result

column.evaluate_sflx_slab=assembled;column.evaluate_vege_flux_masked_slab=masked;dycore.step=step
command=['--prepared-root',binding['prepared_root'],binding['proof_option'],binding['prepared_head_sha256'],'--experiment-config',binding['config'],'--experiment-config-sha256',binding['config_sha256'],'--progress-format','jsonl','--render-products','none','--io-mode','none','--outdir',str(ROOT/('vegetation-check-'+str(time.time_ns())))]
receipt['argv']=command;started=time.perf_counter()
try:
 run(command);raise RuntimeError('did not reach parent and child')
except Complete:receipt['status']='PASS_COMPONENT_ONLY'
except BaseException as error:
 receipt.update(status='FAILED',error=repr(error));raise
finally:
 receipt['wall_s']=time.perf_counter()-started
 (ROOT/f"receipts/vegetation-{a.numerics}{'-profile' if a.profile_only else ''}.json").write_text(json.dumps(receipt,indent=2)+'\n')
