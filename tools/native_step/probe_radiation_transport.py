"""Compare original and device transport on second natural AC radiation calls."""
import argparse,ast,copy,dataclasses,hashlib,json,os,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--numerics',choices=('strict','fast'),required=True);p.add_argument('--profile-only',action='store_true')
a=p.parse_args();binding=json.loads((ROOT/'receipts/local-binding.json').read_text())
mask=os.environ['CUDA_VISIBLE_DEVICES'];os.environ.update(binding['runtime_env']);os.environ.update(CUDA_VISIBLE_DEVICES=mask,GPUWM_NUMERICS=a.numerics)
for key,relative in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','CUPY_CACHE_DIR':'cache/cupy','CUDA_CACHE_PATH':'cache/cuda','GPUWM_GENERATED_TABLE_ROOT':'cache/generated'}.items():
 path=ROOT/relative;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
sys.path[:0]=binding['runtime_env']['PYTHONPATH'].split(':')
import cupy as cp
import numpy as np
from gpuwm.core import rrtmg_legacy as module
from gpuwm.prepared_domain_tree_forecast import main as run
from gpuwm.native_step import Plan,RecordingError
from gpuwm.core.physics_inventory import RADIATION_BOUNDARY_FLUXES
klass=module.RRTMGLegacyRadiation;candidate=klass.__call__
base_path=ROOT/'receipts/rrtmg-transport-base.py'
tree=ast.parse(base_path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==klass.__name__)
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
namespace=dict(vars(module));exec(compile(ast.Module(body=[method],type_ignores=[]),str(base_path),'exec'),namespace);baseline=namespace['__call__']
receipt={'purpose':__doc__,'pid':os.getpid(),'numerics':a.numerics,'profile_only':a.profile_only,'binding':binding,'source':{},'cases':[],'completed_forecast':False}
for name in ('gpuwm/core/rrtmg_legacy.py','receipts/rrtmg-transport-base.py','tools/native_step/probe_radiation_transport.py','gpuwm/core/physics.py','gpuwm/native_step/recorder.c'):
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
class Complete(BaseException):pass
counts={};done=set()
def sync():cp.cuda.get_current_stream().synchronize()
from radiation_inputs import arrays, bits, compare, clone, input_hash
marker=None
if a.profile_only:
 code='extern "C" __global__ void awstep_radiation_start() {}\nextern "C" __global__ void awstep_radiation_end() {}'
 marker=cp.RawModule(code=code);start=marker.get_function('awstep_radiation_start');end=marker.get_function('awstep_radiation_end');start((1,),(1,),());end((1,),(1,),());sync()

def check(adapter,**kwargs):
 counts[id(adapter)]=counts.get(id(adapter),0)+1
 if counts[id(adapter)]!=2:return candidate(adapter,**kwargs)
 state=kwargs['state'];cfg=kwargs['cfg'];key=(cfg.nx,cfg.ny,cfg.nz)
 assert key not in done
 row={'shape':list(key),'model_time':state.elapsed_seconds,'update_count_before':adapter.update_count,'timings_ms':{'base':[],'candidate':[]}}
 before=input_hash(kwargs);old=clone(adapter);new=clone(adapter)
 output_initial={k:v.copy() for k,v in kwargs['fields'].items() if k in RADIATION_BOUNDARY_FLUXES}
 row['declared_output_buffers']=sorted(output_initial)
 def restore_outputs():
  for k,v in output_initial.items():cp.copyto(kwargs['fields'][k],v)
 want=bits(baseline(old,**kwargs));want['retained_ozone']=cp.asnumpy(old.o33d_device())
 after_old=input_hash(kwargs)
 row['base_input_changes']=[k for k in before if before[k]!=after_old.get(k)]
 restore_outputs()
 got=bits(candidate(new,**kwargs));got['retained_ozone']=cp.asnumpy(new.o33d_device())
 row['comparison']=compare(got,want);assert old.update_count==new.update_count==adapter.update_count+1
 owner=getattr(state.physics,'cam_ozone',None)
 if owner is not None and owner.mode=='parent-interpolated':
  assert new.__dict__['_o33d_host'] is None
  saved=state.physics.o3rad.ravel()[:1].copy();ptr=new.o33d_device().data.ptr
  try:
   state.physics.o3rad.ravel()[0]=cp.float32(-123.0)
   retained=cp.asnumpy(new.o33d_device())
   row['retention_after_provider_change']=compare({'ozone':retained},{'ozone':got['retained_ozone']})
   row['lazy_host']=compare({'ozone':new._o33d_grid},{'ozone':got['retained_ozone']})
   assert ptr!=state.physics.o3rad.data.ptr
  finally:state.physics.o3rad.ravel()[:1]=saved
  row['child_snapshot_not_alias']=True
 after_new=input_hash(kwargs)
 row['candidate_input_changes']=[k for k in after_old if after_old[k]!=after_new.get(k)]
 if after_new!=before:
  receipt['cases'].append(row)
  raise AssertionError(('radiation changed input array words',row['base_input_changes'],row['candidate_input_changes']))
 if not a.profile_only:
  for pair in range(9):
   for label,fn,obj in ([('base',baseline,old),('candidate',candidate,new)] if pair%2==0 else [('candidate',candidate,new),('base',baseline,old)]):
    restore_outputs();sync();t=time.perf_counter_ns();result=fn(obj,**kwargs);sync();row['timings_ms'][label].append((time.perf_counter_ns()-t)/1e6)
  row['timing_mean_ms']={k:statistics.mean(v) for k,v in row['timings_ms'].items()}
  row['audit']={}
  for label,fn,obj in [('base',baseline,old),('candidate',candidate,new)]:
   restore_outputs();sync();plan=Plan(diagnostic=True)
   try:
    try:
     with plan.record(kwargs,obj):result=fn(obj,**kwargs)
    except RecordingError as error:row['audit'][label]={'error':str(error)}
    sync();row['audit'][label]['plan']=plan.describe()
   finally:plan.close()
 else:
  for label,fn,obj in [('base',baseline,old),('candidate',candidate,new)]:
   restore_outputs();sync();start((1,),(1,),());result=fn(obj,**kwargs);end((1,),(1,),());sync()
   row.setdefault('profile_ranges',[]).append(label)
 row['inputs_unchanged']=input_hash(kwargs)==before;assert row['inputs_unchanged']
 receipt['cases'].append(row);done.add(key)
 print(json.dumps({k:v for k,v in row.items() if k not in ('audit','comparison','retention_after_provider_change','lazy_host')}),flush=True)
 if len(done)==2:raise Complete()
 restore_outputs()
 return candidate(adapter,**kwargs)
klass.__call__=check
command=['--prepared-root',binding['prepared_root'],binding['proof_option'],binding['prepared_head_sha256'],'--experiment-config',binding['config'],'--experiment-config-sha256',binding['config_sha256'],'--progress-format','jsonl','--render-products','none','--io-mode','none','--outdir',str(ROOT/('radiation-transport-'+str(time.time_ns())))]
receipt['argv']=command
try:
 # Float32 exceptional words and strided weights, plus fallback conversion.
 words=np.array([0,0x80000000,0x7f800000,0xff800000,0x7fc12345,0xffc54321,1,0x807fffff],dtype=np.uint32)
 checks=[]
 for value in (cp.asarray(words.view(np.float32)),cp.asarray(words.view(np.float32))[::2],np.array([0.,1.,-1.],np.float64),cp.asarray([0.,1.,-1.],dtype=cp.float64)):
  expected=cp.asarray(np.ascontiguousarray(np.asarray(klass._host(value),np.float32)))
  checks.append(compare({'weights':cp.asnumpy(klass._vertical_weight_device(value))},{'weights':cp.asnumpy(expected)}))
 receipt['weight_fixtures']=checks
 run(command);raise RuntimeError('second radiation calls not both reached')
except Complete:receipt['status']='EXACT_RADIATION_TRANSPORT_PAIR'
except BaseException as error:receipt.update(status='FAILED',error=repr(error));raise
finally:
 suffix='profile' if a.profile_only else a.numerics
 (ROOT/f'receipts/radiation-transport-{suffix}.json').write_text(json.dumps(receipt,indent=2)+'\n')
