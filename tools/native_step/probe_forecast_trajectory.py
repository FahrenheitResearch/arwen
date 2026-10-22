"""Independent complete AC forecast with native interior periods and ordinary finalization."""
import argparse,ast,copy,ctypes as C,hashlib,json,os,sys,time,traceback
from pathlib import Path
from contextlib import ExitStack,contextmanager,nullcontext
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--arm',choices=('python','native'),required=True);p.add_argument('--numerics',choices=('strict','fast'),required=True);p.add_argument('--attempt',required=True);a=p.parse_args()
assert a.attempt.replace('-','').isalnum()
binding=json.loads((ROOT/'receipts/local-binding.json').read_text());mask=os.environ['CUDA_VISIBLE_DEVICES'];os.environ.update(binding['runtime_env']);os.environ.update(CUDA_VISIBLE_DEVICES=mask,GPUWM_NUMERICS=a.numerics)
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
from gpuwm.prepared_domain_tree_forecast import main as run
from gpuwm.native_step import Plan
from host_annotations import annotate,contexts
from whole_health_annotations import annotate_health
from validator_annotations import annotate_descriptors
from whole_inventory import inventory,endpoint
from phase_health_witness import health_witness
from whole_metadata import Metadata
from time_slots import Frame,frame_for,capture_slots,bind_slots
from radiation_capture import RadiationFrame,RecordedRadiation
from radiation_deferred import DeferredNativeChecks
from radiation_step import conditional_step,RADIATION_FLAG,CadenceScope
from native_bundle import Bundle
from boundary_slots import ExternalBoundarySlots
from gpuwm.ingest import lateral_bc
class DomainFrame(C.Structure):
 _fields_=[('time_'+name,kind) for name,kind in Frame._fields_]+[('rad_'+name,kind) for name,kind in RadiationFrame._fields_]
receipt=dict(purpose=__doc__,arm=a.arm,numerics=a.numerics,attempt=a.attempt,binding=binding,source={},cases=[],completed_forecast=False,timing_claim=False,production_dispatch=False)
destination=ROOT/f'receipts/forecast-trajectory-{a.numerics}-{a.arm}-{a.attempt}.json';assert not destination.exists()
for name in ['gpuwm/prepared_domain_tree_forecast.py','tools/native_step/whole_inventory.py','tools/native_step/run_forecast_trajectory.sh','gpuwm/core/physics.py','gpuwm/core/dycore.py','gpuwm/core/rrtmg_legacy.py','gpuwm/core/clock.py','gpuwm/core/model.py','gpuwm/core/nest.py','gpuwm/native_step/__init__.py','gpuwm/native_step/recorder.c','tools/native_step/probe_forecast_trajectory.py','tools/native_step/boundary_slots.py','gpuwm/ingest/lateral_bc.py','tools/native_step/phase_health_witness.py','tools/native_step/native_bundle.py','tools/native_step/native_bundle.c','tools/native_step/radiation_step.py','tools/native_step/radiation_capture.py','tools/native_step/radiation_daylight.py','tools/native_step/radiation_deferred.py','tools/native_step/radiation_ozone.py','tools/native_step/host_annotations.py','tools/native_step/whole_health_annotations.py','tools/native_step/whole_metadata.py','tools/native_step/time_slots.py','tools/native_step/validator_annotations.py','receipts/libnative-bundle.so','receipts/libradiation-composed-host.so']:
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
receipt['recorder_library_sha256']=hashlib.sha256(Path(os.environ['GPUWM_NATIVE_STEP_LIBRARY']).read_bytes()).hexdigest()
CAPTURE_PERIOD=66;VALIDATOR_PERIOD=67;LAST_NATIVE=1597;LAST_PERIOD=1599
ENDPOINTS=(67,399,400,799,800,1199,1200,1597,1599)
plans={};original_schedule=clock.execute_schedule
def sync():cp.cuda.get_current_stream().synchronize()
from whole_metadata import due, warm_day, RadiationMetadata


def schedule(*args,**kwargs):
 callbacks={k:v for k,v in kwargs.items() if k.startswith('on_') and v is not None};closure={}
 for callback in callbacks.values():
  if callback.__closure__:closure.update(zip(callback.__code__.co_freevars,(c.cell_contents for c in callback.__closure__)))
 poison=closure['poison'];closure.update(zip(poison.__code__.co_freevars,(c.cell_contents for c in poison.__closure__)))
 model=closure['model'];recordings={};boundary_slots={};frame_types={};metadata=RadiationMetadata(model,closure,recordings)
 for name in ('adaptive_driver','spectral_seam','relocation_runner','history_handler'):assert closure.get(name) is None,(name,type(closure.get(name)))
 assert not closure['health_debug'] and not closure['arena_nan_poison'] and not closure['steppers']
 objects={'model':model,'status':closure['status'],'ledger':closure['step_ledger'],'feedback_scratch':closure['feedback_scratch'],'validators':closure['validators']}
 live=dict(period=-1,replaying=False,child_phase=0,bundle=None,step_numbers={},stage_rows=[])
 def admission():
  future=copy.copy(model.root.clock);future.advance()
  assert not closure['restart_enabled'] or not future.restart_due()
  assert not closure['restart_request'].pending()
  for node in model.nodes_by_grid_id.values():
   state=node.state;cfg=node.cfg.run;p=state.physics
   assert getattr(state,'_streamed_domain',None) is None and getattr(state,'chem',None) is None
   assert not cfg.cu_physics and p.fire is None and p.swint is None and p.surface_moisture_ledger is None
   # io-mode none has no history consumer; original scheduler retains alarms.
   assert all(getattr(p,name,None) is None for name in ('radiation_due_override','surface_pbl_due_override','cumulus_due_override'))
   assert _surface_pbl_step_due(2,p.stepbl,cfg.bldt,None)
 def make_plan(name):
  assert name not in plans;plan=Plan();plan.trace_failures();plans[name]=plan;return plan
 def record_plain(name,callback,*callargs):
  plan=make_plan(name)
  with contexts(plan,objects,receipt,live):callback(*callargs)
  sync();receipt.setdefault('phase_plans',{})[name]=plan.describe()
 def frames_and_flags():
  domain_frames=[];domain_flags=[];stage_rows=[]
  for gid,count in ((1,1),(2,3)):
   node=model.node(gid);dom=copy.copy(node.clock);state=copy.copy(node.state);resident=copy.copy(state._lateral_boundary_device);resident.clock=dom;state._lateral_boundary_device=resident
   if node.parent is not None:dom.mark_force()
   for _ in range(count):
    if dom.lbc_reset_due():dom.mark_force()
    dom.prepare_step();refresh_model_time(state,dom,kernel_launch=True);base=frame_for(state,node.cfg.run);frame=frame_types[gid].from_buffer_copy(recordings[gid].frame_template)
    external_index=boundary_slots[gid].fill(frame,state) if gid in boundary_slots else None
    for name,_ in Frame._fields_:setattr(frame,'time_'+name,getattr(base,name))
    recordings[gid].fill(frame,state.elapsed_seconds);flag=RADIATION_FLAG if due(state,node.cfg.run) else 0
    domain_frames.append(frame);domain_flags.append(flag);stage_rows.append(dict(gid=gid,step=dom.step_count+1,elapsed=float(state.elapsed_seconds),radiation_due=bool(flag),external_interval=external_index,boundary_dtbc=float(base.boundary_dtbc)));dom.advance()
  validator=1 if (model.root.clock.step_count+1)%4==0 else 0
  return [domain_frames[0],None,*domain_frames[1:],None,None,None],[domain_flags[0],0,*domain_flags[1:],0,0,validator],stage_rows
 def begin(period,clocks):
  callbacks['on_period_begin'](period,clocks);live.update(period=period,replaying=a.arm=='native' and CAPTURE_PERIOD<period<=LAST_NATIVE,child_phase=0)
  if a.arm=='python':return
  if period==CAPTURE_PERIOD:
   admission()
   for node in model.nodes_by_grid_id.values():warm_day(node)
  if not live['replaying']:
   if period==LAST_NATIVE+1:
    for plan in plans.values():sync();plan.close()
    plans.clear();receipt['closed_before_ordinary_tail']=True
   return
  admission()
  current={row['label']:key for key,row in inventory(objects).items()};assert current==live['identity'],'reachable pointer/layout drift'
  names=('root','force','child','child','child','commit','finalize','validators')
  bootstrap=period==VALIDATOR_PERIOD
  if bootstrap:bundle=Bundle(ROOT/'receipts/libnative-bundle.so',[plans[n] for n in names[:-1]])
  else:
   if live['bundle'] is None:live['bundle']=Bundle(ROOT/'receipts/libnative-bundle.so',[plans[n] for n in names])
   bundle=live['bundle']
  frames,flags,rows=frames_and_flags();live['stage_rows']=rows
  def failure(index):
   mapping={0:0,2:1,3:2,4:3}
   if index in mapping:
    row=rows[mapping[index]];live['step_numbers'][row['gid']]=row['step']
  bundle.submit(frames[:-1] if bootstrap else frames,flags[:-1] if bootstrap else flags,failure_stage=failure);sync()
  case=dict(period=period,path='bootstrap' if bootstrap else 'native',phase_count=7 if bootstrap else 8,domain_steps=rows,validator_due=bool(flags[-1]))
  case['capture_submissions' if bootstrap else 'submissions']=1
  receipt['cases'].append(case)
 def step(gid,dom):
  if gid==2:live['child_phase']+=1
  if live['replaying']:return metadata.step(gid,dom)
  capture=a.arm=='native' and live['period']==CAPTURE_PERIOD and (gid==1 or live['child_phase']==3)
  if not capture:return callbacks['on_step'](gid,dom)
  node=model.node(gid);state=copy.copy(node.state);refresh_model_time(state,dom,kernel_launch=True);assert due(state,node.cfg.run)
  name='root' if gid==1 else 'child';plan=make_plan(name);slots=[];before=metadata.capture_before(gid)
  boundary=ExternalBoundarySlots(plan,node.state,lateral_bc) if gid==1 else None
  if boundary is not None:boundary_slots[gid]=boundary
  frame_type=type('BoundaryDomainFrame'+str(gid),(C.Structure,),{'_fields_':DomainFrame._fields_+(boundary.fields if boundary is not None else [])});frame_types[gid]=frame_type
  recorded=RecordedRadiation(plan,ROOT/'receipts/libradiation-composed-host.so',rrtmg_legacy,DeferredNativeChecks(plan,ROOT/'receipts/libradiation-composed-host.so',deferred_device_checks),parent_snapshot_buffer=cp.empty_like(node.state.physics.o3rad) if gid==2 else None);recordings[gid]=recorded
  cls=rrtmg_legacy.RRTMGLegacyRadiation;original=cls.__call__
  def capture_radiation(adapter,**inputs):
   with recorded.install():return cls.__call__(adapter,**inputs)
  cls.__call__=capture_radiation
  try:
   with contexts(plan,objects,receipt,live),capture_slots(plan,slots),conditional_step(CadenceScope(plan,metadata,gid),physics),boundary.capture() if boundary is not None else nullcontext():callbacks['on_step'](gid,dom)
  finally:cls.__call__=original
  sync();metadata.capture_after(gid,before);frame=frame_type();recorded.bind(frame,prefix='rad_');recorded.frame_template=bytes(frame)
  if boundary is not None:
   boundary.bind(frame_type);receipt['boundary_bindings']=boundary.bindings
  bound=bind_slots(plan,slots,frame_type=frame_type,prefix='time_');receipt.setdefault('bindings',{})[name]=bound
  receipt.setdefault('phase_plans',{})[name]=plan.describe();receipt.setdefault('metadata_deltas',{})[str(gid)]=metadata.step_deltas[gid];receipt.setdefault('radiation_ledger_records',{})[str(gid)]=metadata.radiation_records[gid]
 def force(*callargs):
  if live['replaying']:return metadata.force(*callargs)
  if a.arm=='native' and live['period']==CAPTURE_PERIOD:return record_plain('force',callbacks['on_force'],*callargs)
  return callbacks['on_force'](*callargs)
 def commit(*callargs):
  if live['replaying']:return metadata.feedback_commit(*callargs)
  if a.arm=='native' and live['period']==CAPTURE_PERIOD:return record_plain('commit',callbacks['on_feedback_commit'],*callargs)
  return callbacks['on_feedback_commit'](*callargs)
 def finalize(*callargs):
  if live['replaying']:return metadata.feedback_finalize(*callargs)
  if a.arm=='native' and live['period']==CAPTURE_PERIOD:return record_plain('finalize',callbacks['on_feedback_finalize'],*callargs)
  return callbacks['on_feedback_finalize'](*callargs)
 def end(period,clocks):
  if a.arm=='native' and period==VALIDATOR_PERIOD:
   assert clocks[1].step_count%4==0
   plan=make_plan('validators')
   with contexts(plan,objects,receipt,live),plan.when(require=1):callbacks['on_period_end'](period,clocks)
   sync();receipt.setdefault('phase_plans',{})['validators']=plan.describe()
  elif live['replaying']:metadata.end()
  else:callbacks['on_period_end'](period,clocks)
  if a.arm=='native' and period in (CAPTURE_PERIOD,VALIDATOR_PERIOD):
   live['identity']={row['label']:key for key,row in inventory(objects).items()}
   expected={'root','child','force','commit','finalize'}
   if period==VALIDATOR_PERIOD:expected.add('validators')
   assert set(plans)==expected
  if period>=VALIDATOR_PERIOD and clocks[1].step_count%4==0:
   receipt.setdefault('scheduled_health_witnesses',{})[str(period)]=health_witness(closure['validators'])
  if period==CAPTURE_PERIOD:
   receipt['capture_endpoint']=endpoint(objects)
  if period in ENDPOINTS:
   receipt.setdefault('endpoints',{})[str(period)]=endpoint(objects)
   print(json.dumps(dict(period=period,arm=a.arm,endpoint=True)),flush=True)
  root_state=model.root.state;external=lateral_bc._served_external_interval(root_state,root_state._lateral_boundary_device)[0]
  receipt.setdefault('boundary_clock_witnesses',{})[str(period)]=dict(root_ticks=int(model.root.clock.ticks),root_dtbc=float(model.root.clock.dtbc_fp32),root_elapsed=float(root_state.elapsed_seconds),served_interval_start=float(external.start_seconds),served_interval_end=float(external.end_seconds))
  if not live['replaying']:receipt['cases'].append(dict(period=period,path='capture' if a.arm=='native' and period==CAPTURE_PERIOD else 'python'))
  if period==LAST_PERIOD:receipt['last_period_reached']=period
 kwargs.update(on_period_begin=begin,on_step=step,on_force=force,on_feedback_commit=commit,on_feedback_finalize=finalize,on_period_end=end)
 report=original_schedule(*args,**kwargs)
 receipt['schedule_report']={name:getattr(report,name) for name in ('steps','forces','feedback_calls','restarts','lbc_resets','histories')}
 receipt['final_clocks']={str(gid):dict(ticks=int(dom.ticks),step_count=int(dom.step_count),run_ticks=int(dom.run_ticks),dtbc=float(dom.dtbc_fp32)) for gid,dom in report.clocks.items()}
 return report
clock.execute_schedule=schedule
command=['--prepared-root',binding['prepared_root'],binding['proof_option'],binding['prepared_head_sha256'],'--experiment-config',binding['config'],'--experiment-config-sha256',binding['config_sha256'],'--progress-format','jsonl','--render-products','none','--io-mode','none','--outdir',str(ROOT/('forecast-trajectory-'+a.arm+'-'+a.attempt+'-'+str(time.time_ns())))]
receipt['argv']=command
try:
 result=run(command);assert result==0,('ordinary forecast finalization failed',result)
 assert receipt.get('last_period_reached')==1599
 receipt.update(status='FULL_FORECAST_UNCOMPARED',completed_forecast=True,returncode=result)
except BaseException as error:receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());raise
finally:
 for name,plan in plans.items():
  receipt.setdefault('final_plans',{})[name]=plan.describe();sync();plan.close()
 destination.write_text(json.dumps(receipt,indent=2)+'\n')
