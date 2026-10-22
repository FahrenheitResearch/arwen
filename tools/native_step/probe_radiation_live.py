"""Natural next-call radiation replay with live atmosphere staging and ordinary caller publication."""
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
counts={}
def sync():cp.cuda.get_current_stream().synchronize()
from radiation_inputs import arrays, bits, compare, clone, isolated_kwargs, sample


from gpuwm.core import deferred_device_checks as checks
from radiation_capture import RecordedRadiation,RadiationFrame
from radiation_deferred import DeferredNativeChecks
klass=module.RRTMGLegacyRadiation;baseline=klass.__call__
helper=ROOT/'receipts/libradiation-composed-host.so'
receipt={'purpose':__doc__,'pid':os.getpid(),'numerics':a.numerics,'attempt':a.attempt,'binding':binding,'source':{},'cases':[],'completed_forecast':False,'production_dispatch':False,'timing_claim':False}
assert a.attempt.replace('-','').isalnum()
destination=ROOT/f'receipts/radiation-live-{a.numerics}-{a.attempt}.json'
assert not destination.exists(),'preserve prior receipt'
for name in ['gpuwm/core/rrtmg_legacy.py','gpuwm/core/physics.py','gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/probe_radiation_live.py','tools/native_step/radiation_inputs.py','tools/native_step/radiation_capture.py','tools/native_step/radiation_daylight.py','tools/native_step/radiation_ozone.py','tools/native_step/radiation_deferred.py','tools/native_step/radiation_verdict_pins.json','tools/native_step/radiation_composed_host.c','receipts/libradiation-composed-host.so']:
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
receipt['recorder_library_sha256']=hashlib.sha256(Path(os.environ['GPUWM_NATIVE_STEP_LIBRARY']).read_bytes()).hexdigest()



from radiation_inputs import AtmosphericInputs
plans={};finished=set()
def stable_bindings(kwargs):
 values={k:v for k,v in kwargs['fields'].items() if isinstance(v,cp.ndarray)}
 values.update({'state.'+k:getattr(kwargs['state'],k,None) for k in ('alt','ht','fnm','fnp','aerodm')})
 return {k:(v.data.ptr,v.shape,v.strides,v.dtype.str) for k,v in values.items() if isinstance(v,cp.ndarray)}
def check(adapter,**kwargs):
 counts[id(adapter)]=counts.get(id(adapter),0)+1;count=counts[id(adapter)]
 state=kwargs['state'];cfg=kwargs['cfg'];key=(cfg.nx,cfg.ny,cfg.nz)
 if count not in (2,3,4):return baseline(adapter,**kwargs)
 if count==4:
  assert adapter.update_count==3
  result=baseline(adapter,**kwargs);sync();finished.add(key)
  receipt['cases'].append(dict(shape=list(key),stage='ORDINARY_TAIL',model_time=float(state.elapsed_seconds),update_count=adapter.update_count))
  return result
 reference_kwargs=isolated_kwargs(kwargs);old=clone(adapter)
 if count==2:
  warm_kwargs=isolated_kwargs(kwargs);warm_kwargs['state'].elapsed_seconds+=18*3600
  warm_adapter=clone(adapter);warm=baseline(warm_adapter,**warm_kwargs);sync();del warm,warm_adapter,warm_kwargs
 expected_result=baseline(old,**reference_kwargs);sync();expected=sample(expected_result,old);expected_night=bits(old._night_outputs)
 row=dict(shape=list(key),model_time=float(state.elapsed_seconds),call=count,stage='START');receipt['cases'].append(row)
 if count==2:
  plan=Plan();inputs=AtmosphericInputs(plan,kwargs['atmosphere'],cp)
  recorded=RecordedRadiation(plan,helper,module,DeferredNativeChecks(plan,helper,checks))
  staged=dict(kwargs,atmosphere=inputs.values)
  plans[key]=dict(plan=plan,inputs=inputs,recorded=recorded,kwargs=staged)
  with plan.record(adapter,staged,inputs,recorded):
   inputs.record()
   with recorded.install():result=klass.__call__(adapter,**staged)
  sync();frame=inputs.frame_type();recorded.bind(frame);inputs.bind()
  plans[key].update(frame=frame,result=result,ozone=adapter.o33d_device(),bindings=stable_bindings(kwargs))
  row['plan']=plan.describe();row['stage']='CAPTURE'
 else:
  item=plans[key];plan=item['plan'];inputs=item['inputs'];recorded=item['recorded'];frame=item['frame'];result=item['result']
  assert stable_bindings(kwargs)==item['bindings'],'live radiation non-atmosphere pointers changed'
  row['changed_atmosphere_pointers']=inputs.fill(frame,kwargs['atmosphere']);recorded.fill(frame,state.elapsed_seconds)
  before=adapter.update_count
  plan.submit(frame);sync();recorded.publish_metadata()
  # The ordinary root caller aliases retained ozone to its carrier after each call.
  # Restore this template's output before that caller publishes the new snapshot.
  adapter._o33d_device=item['ozone'];adapter._o33d_host=None;adapter.update_count=before+1
  row.update(stage='NATIVE',python_submissions=1)
 row['comparison']=compare(sample(result,adapter),expected);row['night_metadata']=compare(bits(adapter._night_outputs),expected_night)
 assert adapter.update_count==old.update_count
 print(json.dumps({k:row[k] for k in ('shape','model_time','stage')}),flush=True)
 return result
klass.__call__=check
from gpuwm.core import clock as schedule_clock
from whole_inventory import endpoint
original_schedule=schedule_clock.execute_schedule
def schedule(*args,**kwargs):
 callbacks={k:v for k,v in kwargs.items() if k.startswith('on_') and v is not None};closure={}
 for callback in callbacks.values():
  if callback.__closure__:closure.update(zip(callback.__code__.co_freevars,(c.cell_contents for c in callback.__closure__)))
 original_end=kwargs['on_period_end']
 def end(*args):
  original_end(*args)
  if len(finished)==2:
   receipt['publication']={}
   for gid,node in closure['model'].nodes_by_grid_id.items():
    p=node.state.physics;receipt['publication'][str(gid)]=dict(call_counts=dict(p.call_counts),model_time=node.state.elapsed_seconds,carriers=repr(vars(p.carriers)))
   raise Complete()
 kwargs['on_period_end']=end
 return original_schedule(*args,**kwargs)
schedule_clock.execute_schedule=schedule
command=['--prepared-root',binding['prepared_root'],binding['proof_option'],binding['prepared_head_sha256'],'--experiment-config',binding['config'],'--experiment-config-sha256',binding['config_sha256'],'--progress-format','jsonl','--render-products','none','--io-mode','none','--outdir',str(ROOT/('radiation-live-'+a.attempt+'-'+str(time.time_ns())))]
receipt['argv']=command
try:
 run(command);raise RuntimeError('natural fourth radiation calls not both reached')
except Complete:receipt['status']='EXACT_NATURAL_RADIATION_REPLAY_WITH_ORDINARY_TAIL'
except BaseException as error:receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());raise
finally:
 for item in plans.values():
  sync();item['plan'].close()
 destination.write_text(json.dumps(receipt,indent=2)+'\n')
