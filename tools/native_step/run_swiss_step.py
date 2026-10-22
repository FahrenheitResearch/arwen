"""Whole Swiss forecast with native steps and original history consumers."""
import argparse,copy,ctypes as C,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
from contextlib import ExitStack,contextmanager,nullcontext
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--arm',choices=('python','native'),required=True);p.add_argument('--numerics',choices=('strict','fast'),required=True);p.add_argument('--attempt',required=True);a=p.parse_args()
assert a.attempt.replace('-','').isalnum()
binding=json.loads(Path(os.environ.get('GPUWM_STEP_BINDING',ROOT/'receipts/local-binding.json')).read_text());mask=os.environ['CUDA_VISIBLE_DEVICES'];os.environ.update(binding['runtime_env']);os.environ.update(CUDA_VISIBLE_DEVICES=mask,GPUWM_NUMERICS=a.numerics)
for key,relative in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','CUPY_CACHE_DIR':'cache/cupy','CUDA_CACHE_PATH':'cache/cuda','GPUWM_GENERATED_TABLE_ROOT':'cache/generated'}.items():
 path=ROOT/relative;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
sys.path.insert(0,str(ROOT))
import cupy as cp
import numpy as np
from gpuwm.core import clock,physics,rrtmg_legacy,deferred_device_checks
from gpuwm.core.physics import _physics_interval_steps,_radiation_step_due,_surface_pbl_step_due,_prepare_atmosphere
from gpuwm.core.radiation_composition import legacy_radiation_adapter
from gpuwm.core.physics_inventory import RADIATION_BOUNDARY_FLUXES
from gpuwm.core.state import refresh_model_time
from gpuwm.wrfinput_forecast import run_wrf_forecast as run
from gpuwm.native_step import Plan
from host_annotations import annotate,contexts
from whole_health_annotations import annotate_health
from validator_annotations import annotate_descriptors
from whole_metadata import Metadata
from whole_inventory import inventory
from time_slots import Frame,frame_for,capture_slots,bind_slots
from radiation_capture import RadiationFrame,RecordedRadiation
from radiation_deferred import DeferredNativeChecks
from radiation_step import conditional_step,RADIATION_FLAG,CadenceScope
from native_bundle import Bundle
from rational_boundary_slots import RationalBoundarySlots
from glacier_capture import RecordedGlacier
from gpuwm.ingest import lateral_bc
class DomainFrame(C.Structure):
 _fields_=[('time_'+name,kind) for name,kind in Frame._fields_]+[('rad_'+name,kind) for name,kind in RadiationFrame._fields_]
receipt=dict(purpose=__doc__,arm=a.arm,numerics=a.numerics,attempt=a.attempt,binding=binding,source={},cases=[],completed_forecast=False,timing_claim=True,production_dispatch='scoped-Swiss-history',history_events=[],period_wall_seconds=[])
destination=ROOT/f'receipts/swiss-step-{a.numerics}-{a.arm}-{a.attempt}.json';assert not destination.exists()
for name in ['gpuwm/prepared_domain_tree_forecast.py','tools/native_step/whole_inventory.py','tools/native_step/run_canary_step.sh','gpuwm/core/physics.py','gpuwm/core/dycore.py','gpuwm/core/rrtmg_legacy.py','gpuwm/core/clock.py','gpuwm/core/model.py','gpuwm/core/nest.py','gpuwm/native_step/__init__.py','gpuwm/native_step/recorder.c','tools/native_step/run_swiss_step.py','tools/native_step/boundary_slots.py','tools/native_step/rational_boundary_slots.py','tools/native_step/glacier_capture.py','gpuwm/ingest/lateral_bc.py','tools/native_step/phase_health_witness.py','tools/native_step/native_bundle.py','tools/native_step/native_bundle.c','tools/native_step/radiation_step.py','tools/native_step/radiation_capture.py','tools/native_step/radiation_daylight.py','tools/native_step/radiation_deferred.py','tools/native_step/radiation_ozone.py','tools/native_step/host_annotations.py','tools/native_step/whole_health_annotations.py','tools/native_step/whole_metadata.py','tools/native_step/time_slots.py','tools/native_step/validator_annotations.py','receipts/libnative-bundle.so','receipts/libradiation-composed-host.so']:
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
receipt['recorder_library_sha256']=hashlib.sha256(Path(os.environ['GPUWM_NATIVE_STEP_LIBRARY']).read_bytes()).hexdigest()
plans={};original_schedule=clock.execute_schedule
def sync():cp.cuda.get_current_stream().synchronize()
from whole_metadata import due
def warm_day(node):
 adapter=legacy_radiation_adapter(node.state.physics.radiation_callable);new=copy.copy(adapter)
 if new.__dict__.get('_ozmixt_buf') is not None:new._ozmixt_buf=new._ozmixt_buf.copy()
 state=copy.copy(node.state);state.elapsed_seconds+=12*3600;fields=dict(state.physics.fields)
 for key,value in fields.items():
  if key in RADIATION_BOUNDARY_FLUXES:fields[key]=value.copy()
 rrtmg_legacy.RRTMGLegacyRadiation.__call__(new,atmosphere=_prepare_atmosphere(node.state),fields=fields,state=state,cfg=node.cfg.run);sync()

class RadiationMetadata(Metadata):
 def __init__(self,model,closure,recordings):super().__init__(model,closure);self.recordings=recordings;self.radiation_records={}
 def step(self,gid,clock):
  node=self.model.node(gid);p=node.state.physics
  state=copy.copy(node.state);refresh_model_time(state,clock,kernel_launch=True);is_due=due(state,node.cfg.run);now=float(state.elapsed_seconds)
  saved=self.step_deltas[gid];counts,micro,records,drains=saved;counts=dict(counts)
  if not is_due:
   assert counts.pop('radiation')==1
   records-=self.radiation_records[gid];assert records>=0
   if node.parent is None and p.cam_ozone is not None:assert counts.pop('cam_ozone')==1
  self.step_deltas[gid]=(counts,micro,records,drains)
  try:super().step(gid,clock)
  finally:self.step_deltas[gid]=saved
  if is_due:
   recorded=self.recordings[gid];adapter=recorded.adapter;adapter.update_count+=1;adapter._o33d_host=None;recorded.publish_metadata()
   for name in ('glw','swdown','gsw','coszen'):
    if name in p.fields:p.carriers.declare(name,source=physics.CARRIER_SOURCE_RADIATION_SCHEME,model_time=now)
   if node.parent is None and p.cam_ozone is not None:p.carriers.declare('o3rad',source='cam_ozone',model_time=now)


def schedule(*args,**kwargs):
 callbacks={k:v for k,v in kwargs.items() if k.startswith('on_') and v is not None};closure={}
 for callback in callbacks.values():
  if callback.__closure__:closure.update(zip(callback.__code__.co_freevars,(c.cell_contents for c in callback.__closure__)))
 poison=closure['poison'];closure.update(zip(poison.__code__.co_freevars,(c.cell_contents for c in poison.__closure__)))
 model=closure['model'];assert set(model.nodes_by_grid_id)=={1}
 node=model.root;assert node.clock.run_ticks/node.clock.tick_den==86400
 assert node.clock.step_ticks/node.clock.tick_den==18
 for name in ('adaptive_driver','spectral_seam','relocation_runner'):assert closure.get(name) is None
 assert not closure['health_debug'] and not closure['arena_nan_poison'] and not closure['steppers']
 assert not node.cfg.run.cu_physics and node.state.physics.fire is None
 assert _surface_pbl_step_due(2,node.state.physics.stepbl,node.cfg.run.bldt,None)
 recordings={};metadata=RadiationMetadata(model,closure,recordings)
 objects={'model':model,'status':closure['status'],'ledger':closure['step_ledger'],'feedback_scratch':closure['feedback_scratch'],'validators':closure['validators']}
 live={'period':-1,'plan':None,'capture':False,'step_numbers':{}}
 def retire(reason):
  if live['plan'] is not None:
   sync();before=cp.get_default_memory_pool().total_bytes()
   live['plan'].close();plans.clear();live['plan']=None
   recordings.clear()
   for key in ('bundle','boundary','frame_type'):live.pop(key,None)
   # Source-pinned adapters have function namespaces pointing back to their
   # owner. Collect those closed capture cycles at this output boundary.
   gc.collect();cp.get_default_memory_pool().free_all_blocks()
   receipt.setdefault('retirements',[]).append(dict(period=live['period'],reason=reason,pool_before=before,pool_after=cp.get_default_memory_pool().total_bytes()))
 def begin(period,clocks):
  live['period_started']=time.perf_counter();live['period']=period
  callbacks['on_period_begin'](period,clocks)
  state=copy.copy(node.state);refresh_model_time(state,node.clock,kernel_launch=True)
  history=node.clock.history_rings_within_step()
  future=copy.copy(node.clock);future.advance()
  assert not closure['restart_request'].pending()
  assert not closure['restart_enabled'] or not future.restart_due()
  # An output step produces REFL_10CM inside microphysics. Keep that full
  # original path, retiring all recorded pointers before it can allocate.
  if history and a.arm=='native':
   retire('original history production')
   receipt.setdefault('history_production_fallbacks',[]).append(period)
   print(json.dumps(dict(event='ordinary-history-production-fallback',period=period)),flush=True)
  live['capture']=a.arm=='native' and live['plan'] is None and period>0 and due(state,node.cfg.run) and not history
  if live['capture'] and not live.get('warmed'):
   warm_day(node);live['warmed']=True
 def step(gid,dom):
  live['step_numbers'][gid]=dom.step_count+1
  if a.arm=='python':return callbacks['on_step'](gid,dom)
  if live['capture']:
   plan=Plan();plan.trace_failures();plans['root']=plan;live['plan']=plan;slots=[]
   boundary=RationalBoundarySlots(plan,node.state,lateral_bc)
   glacier=RecordedGlacier(plan,node.state.physics,receipt.setdefault('glacier_transport',[]))
   frame_type=type('SwissDomainFrame',(C.Structure,),{'_fields_':DomainFrame._fields_+boundary.fields})
   recorded=RecordedRadiation(plan,ROOT/'receipts/libradiation-composed-host.so',rrtmg_legacy,DeferredNativeChecks(plan,ROOT/'receipts/libradiation-composed-host.so',deferred_device_checks));recordings[gid]=recorded
   cls=rrtmg_legacy.RRTMGLegacyRadiation;original=cls.__call__
   def capture_radiation(adapter,**inputs):
    with recorded.install():return cls.__call__(adapter,**inputs)
   cls.__call__=capture_radiation
   before=metadata.capture_before(gid)
   try:
    with contexts(plan,objects,receipt,live),capture_slots(plan,slots),conditional_step(CadenceScope(plan,metadata,gid),physics),boundary.capture(),glacier.install():callbacks['on_step'](gid,dom)
   finally:cls.__call__=original
   sync();metadata.step_deltas.pop(gid,None);metadata.capture_after(gid,before)
   frame=frame_type();recorded.bind(frame,prefix='rad_');recorded.frame_template=bytes(frame);boundary.bind(frame_type)
   bindings=bind_slots(plan,slots,frame_type=frame_type,prefix='time_')
   live.update(frame_type=frame_type,boundary=boundary,bundle=Bundle(ROOT/'receipts/libnative-bundle.so',[plan]))
   receipt.setdefault('captures',[]).append(dict(period=live['period'],plan=plan.describe(),bindings=bindings))
  elif live['plan'] is None:
   callbacks['on_step'](gid,dom)
  else:
   state=copy.copy(node.state);refresh_model_time(state,dom,kernel_launch=True)
   base=frame_for(state,node.cfg.run);recorded=recordings[gid];frame=live['frame_type'].from_buffer_copy(recorded.frame_template)
   for name,_ in Frame._fields_:setattr(frame,'time_'+name,getattr(base,name))
   external=live['boundary'].fill(frame,state);recorded.fill(frame,state.elapsed_seconds)
   flag=RADIATION_FLAG if due(state,node.cfg.run) else 0
   live['bundle'].submit([frame],[flag]);metadata.step(gid,dom)
   receipt['native_steps']=receipt.get('native_steps',0)+1
   receipt.setdefault('boundary_intervals',{})[str(external)]=live['period']
 def history(gid,ticks):
  sync();callbacks['on_history'](gid,ticks)
  receipt['history_events'].append(dict(gid=gid,ticks=int(ticks),elapsed=float(node.state.elapsed_seconds),period=live['period']))
 def commit(period,clocks):
  now=time.perf_counter();receipt['period_wall_seconds'].append(now-live.get('previous_commit',live['period_started']));live['previous_commit']=now
  callbacks['on_period_commit'](period,clocks)
 started=time.perf_counter()
 kwargs.update(on_period_begin=begin,on_step=step,on_history=history,on_period_commit=commit)
 try:report=original_schedule(*args,**kwargs)
 finally:retire('end of schedule')
 sync();receipt['schedule_wall_seconds']=time.perf_counter()-started
 receipt['schedule_report']={name:getattr(report,name) for name in ('steps','forces','feedback_calls','restarts','lbc_resets','histories')}
 receipt['final_clocks']={str(gid):dict(ticks=int(dom.ticks),step_count=int(dom.step_count),run_ticks=int(dom.run_ticks),dtbc=float(dom.dtbc_fp32)) for gid,dom in report.clocks.items()}
 return report
clock.execute_schedule=schedule
outdir=ROOT/('swiss-step-'+a.arm+'-'+a.attempt+'-'+str(time.time_ns()))
receipt['command']=dict(directory='/workspace/shared-prepared/jax-arwen-swiss-original',outdir=str(outdir),run_seconds=86400,benchmark=True,rrtmg_variant='rrtmg_legacy',thompson_version='wrf_461',io_mode='history',exclusive_gpu=False)
try:
 started=time.perf_counter();result=run(**receipt['command']);receipt['cli_wall_seconds']=time.perf_counter()-started
 assert result==0 and receipt['final_clocks']['1']['step_count']==4800
 receipt.update(status='FULL_HISTORY_FORECAST_UNCOMPARED',completed_forecast=True,returncode=result)
except BaseException as error:receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());raise
finally:
 for name,plan in plans.items():
  receipt.setdefault('plan_audit_events',{})[name]=plan.audit_events
  sync();plan.close()
 walls=receipt['period_wall_seconds']
 if len(walls)>1:receipt['mean_complete_period_2_to_N_ms']=1000*sum(walls[1:])/len(walls[1:])
 destination.write_text(json.dumps(receipt,indent=2)+'\n')
