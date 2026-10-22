"""Composed native AC radiation capture and changing-calendar replay. No timing claim."""
import argparse,ast,copy,ctypes as C,dataclasses,hashlib,json,os,statistics,sys,time,traceback
from pathlib import Path
from datetime import timedelta
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--numerics',choices=('strict','fast'),required=True)
p.add_argument('--attempt',required=True)
a=p.parse_args();binding=json.loads((ROOT/'receipts/local-binding.json').read_text())
mask=os.environ['CUDA_VISIBLE_DEVICES'];os.environ.update(binding['runtime_env']);os.environ.update(CUDA_VISIBLE_DEVICES=mask,GPUWM_NUMERICS=a.numerics)
for key,relative in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','CUPY_CACHE_DIR':'cache/cupy','CUDA_CACHE_PATH':'cache/cuda','GPUWM_GENERATED_TABLE_ROOT':'cache/generated'}.items():
 path=ROOT/relative;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
os.environ['PYTHONPATH']=str(ROOT);sys.path.insert(0,str(ROOT))
import cupy as cp
import numpy as np
from gpuwm.core import rrtmg_legacy as module
from gpuwm.prepared_domain_tree_forecast import main as run
from gpuwm.native_step import Plan,RecordingError
from gpuwm.core.physics_inventory import RADIATION_BOUNDARY_FLUXES
class Complete(BaseException):pass
counts={};done=set()
def sync():cp.cuda.get_current_stream().synchronize()
from radiation_inputs import arrays, bits, compare, clone, input_hash, isolated_kwargs, sample


from gpuwm.core import deferred_device_checks as checks
from radiation_capture import RecordedRadiation,RadiationFrame
from radiation_deferred import DeferredNativeChecks
klass=module.RRTMGLegacyRadiation;baseline=klass.__call__
helper=ROOT/'receipts/libradiation-composed-host.so'
receipt={'purpose':__doc__,'pid':os.getpid(),'numerics':a.numerics,'attempt':a.attempt,'binding':binding,'source':{},'cases':[],'completed_forecast':False,'production_dispatch':False,'timing_claim':False}
assert a.attempt.replace('-','').isalnum()
destination=ROOT/f'receipts/radiation-replay-{a.numerics}-{a.attempt}.json'
assert not destination.exists(),'preserve prior receipt'
for name in ['gpuwm/core/rrtmg_legacy.py','gpuwm/core/physics.py','gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/probe_radiation_replay.py','tools/native_step/radiation_capture.py','tools/native_step/radiation_daylight.py','tools/native_step/radiation_ozone.py','tools/native_step/radiation_deferred.py','tools/native_step/radiation_verdict_pins.json','tools/native_step/radiation_composed_host.c','receipts/libradiation-composed-host.so']:
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
receipt['recorder_library_sha256']=hashlib.sha256(Path(os.environ['GPUWM_NATIVE_STEP_LIBRARY']).read_bytes()).hexdigest()

def protected_model_hash(kwargs):
 result=input_hash(kwargs)
 for key,value in kwargs['fields'].items():
  if key in RADIATION_BOUNDARY_FLUXES:
   result['original_boundary.'+key]=hashlib.sha256(cp.asnumpy(value).tobytes()).hexdigest()
 return result



def check(adapter,**kwargs):
 counts[id(adapter)]=counts.get(id(adapter),0)+1
 if counts[id(adapter)]!=2:return baseline(adapter,**kwargs)
 state=kwargs['state'];cfg=kwargs['cfg'];key=(cfg.nx,cfg.ny,cfg.nz);assert key not in done
 row={'shape':list(key),'model_time':float(state.elapsed_seconds),'trials':[],'stage':'setup'};receipt['cases'].append(row)
 before=protected_model_hash(kwargs);candidate_kwargs=isolated_kwargs(kwargs);new=clone(adapter)
 child=adapter._ozone_provider is not None
 if child:
  candidate_kwargs['state'].physics=copy.copy(state.physics)
  candidate_kwargs['state'].physics.o3rad=state.physics.o3rad.copy()
  carrier=candidate_kwargs['state'].physics.o3rad;carrier_initial=carrier.copy()
 else:carrier=None
 def reference(elapsed):
  reference_kwargs=isolated_kwargs(candidate_kwargs);reference_kwargs['state'].elapsed_seconds=elapsed
  old=clone(adapter);result=baseline(old,**reference_kwargs);sync()
  return sample(result,old),bits(old._night_outputs),old.update_count
 elapsed=float(state.elapsed_seconds)
 row['stage']='baseline_daytime_warmup'
 warm,warm_night,warm_updates=reference(elapsed+18*3600)
 row['warmup_day_columns']=int(np.count_nonzero(warm['.coszen']>np.float32(0)));assert row['warmup_day_columns']>0
 del warm,warm_night
 row['stage']='baseline_current';want,want_night,want_updates=reference(elapsed)
 plan=Plan();deferred=DeferredNativeChecks(plan,helper,checks);recorded=RecordedRadiation(plan,helper,module,deferred)
 try:
  row['stage']='capture'
  with plan.record(candidate_kwargs,new,recorded):
   with recorded.install():result=klass.__call__(new,**candidate_kwargs)
  sync();row['plan']=plan.describe();row['deferred']=deferred.groups
  row['capture']=compare(sample(result,new),want);row['capture_night']=compare(bits(new._night_outputs),want_night)
  assert new.update_count==want_updates==adapter.update_count+1
  frame=RadiationFrame();recorded.bind(frame)
  trials=[('first_timestep',0.),('current',elapsed),('daytime',elapsed+18*3600),('night_again',elapsed)]
  if child:trials.extend([('changed_child_ozone',elapsed+18*3600),('restored_child_ozone',elapsed+18*3600)])
  for label,clock in trials:
   row['stage']=label
   if label=='changed_child_ozone':cp.multiply(carrier,np.float32(1.01),out=carrier)
   elif label=='restored_child_ozone':cp.copyto(carrier,carrier_initial)
   expected,expected_night,expected_updates=reference(clock)
   changed_coszen=int(np.count_nonzero(expected['.coszen'].view(np.uint32)!=want['.coszen'].view(np.uint32)))
   changed_ozone=int(np.count_nonzero(expected['retained_ozone'].view(np.uint32)!=want['retained_ozone'].view(np.uint32)))
   if label=='daytime':assert changed_coszen>0
   if label=='changed_child_ozone':assert changed_ozone>0
   recorded.fill(frame,clock)
   # Poison output allocations, but never aliased model inputs or carrier arrays.
   # The exact input hashes below detect any unintended in-place input mutation.
   for name,value in arrays(result):
    if name!='.cldfra':value.fill(cp.nan)
   sync();plan.submit(frame);sync();recorded.publish_metadata()
   got=sample(result,new)
   trial={'label':label,'elapsed_seconds':clock,'comparison':compare(got,expected),'night_metadata':compare(bits(new._night_outputs),expected_night),'day_columns':int(recorded.day.counts[0]),'night_columns':int(recorded.day.counts[1]),'python_submissions':1,'reference_coszen_words_changed_from_capture':changed_coszen,'reference_ozone_words_changed_from_capture':changed_ozone}
   expected_day=int(np.count_nonzero(expected['.coszen']>np.float32(0)));expected_night_count=int(np.count_nonzero(expected['.coszen']<=np.float32(0)))
   assert (trial['day_columns'],trial['night_columns'])==(expected_day,expected_night_count)
   if label=='daytime':assert trial['day_columns']>0
   if label in ('current','night_again'):assert trial['day_columns']==0
   trial['original_model_inputs_unchanged']=protected_model_hash(kwargs)==before;assert trial['original_model_inputs_unchanged']
   row['trials'].append(trial);print(json.dumps({'shape':key,'label':label,'words':trial['comparison']['words'],'different_words':trial['comparison']['different_words'],'day_columns':trial['day_columns']}),flush=True)
  row['stage']='PASS';row['plan_after']=plan.describe();row['native_update_count_not_published']=new.update_count
  assert not plan.audit_events
 finally:
  if row['stage']!='PASS':
   row['failure_plan']=plan.describe()
  sync();plan.close()
 done.add(key)
 assert protected_model_hash(kwargs)==before
 if len(done)==2:raise Complete()
 return baseline(adapter,**kwargs)
klass.__call__=check
command=['--prepared-root',binding['prepared_root'],binding['proof_option'],binding['prepared_head_sha256'],'--experiment-config',binding['config'],'--experiment-config-sha256',binding['config_sha256'],'--progress-format','jsonl','--render-products','none','--io-mode','none','--outdir',str(ROOT/('radiation-replay-'+a.attempt+'-'+str(time.time_ns())))]
receipt['argv']=command
try:
 run(command);raise RuntimeError('second radiation calls not both reached')
except Complete:receipt['status']='EXACT_COMPOSED_RADIATION_REPLAY'
except BaseException as error:receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());raise
finally:destination.write_text(json.dumps(receipt,indent=2)+'\n')
