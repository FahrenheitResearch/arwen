"""Eight advancing AC intervals with one native submit per replayed interval."""
import argparse,hashlib,json,os,sys,time,types
from collections.abc import Mapping
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--arm',choices=('python','native'),required=True);p.add_argument('--numerics',choices=('strict','fast'),required=True)
p.add_argument('--ordinary-tail',type=int,choices=(0,2),default=0);a=p.parse_args();binding=json.loads((ROOT/'receipts/local-binding.json').read_text())
mask=os.environ['CUDA_VISIBLE_DEVICES'];os.environ.update(binding['runtime_env'])
os.environ.update(CUDA_VISIBLE_DEVICES=mask,GPUWM_NUMERICS=a.numerics)
for key,relative in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','CUPY_CACHE_DIR':'cache/cupy','CUDA_CACHE_PATH':'cache/cuda','GPUWM_GENERATED_TABLE_ROOT':'cache/generated'}.items():
 path=ROOT/relative;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
sys.path[:0]=binding['runtime_env']['PYTHONPATH'].split(':')
import cupy as cp
import numpy as np
from gpuwm.core import clock
from gpuwm.prepared_domain_tree_forecast import main as run
from gpuwm.native_step import Plan
from host_annotations import annotate
from whole_health_annotations import annotate_health
from validator_annotations import annotate_descriptors
class Complete(BaseException):pass
receipt={'purpose':__doc__,'pid':os.getpid(),'numerics':a.numerics,'binding':binding,'cases':[],'source':{},'completed_forecast':False,'advancing_native_trajectory':True,'host_metadata':'explicit publication','arm':a.arm}
for name in ('gpuwm/core/dycore.py','gpuwm/core/physics.py','gpuwm/core/noahmp_column_slab.py','gpuwm/core/noahmp_flux_slab.py','gpuwm/core/kernels/noahmp_vegeflux.cu','gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/host_annotations.py','tools/native_step/whole_health_annotations.py','tools/native_step/probe_whole_trajectory.py','tools/native_step/whole_metadata.py','tools/native_step/whole_inventory.py','tools/native_step/whole_period_frame.py','tools/native_step/time_slots.py','tools/native_step/validator_annotations.py','gpuwm/core/health.py','gpuwm/core/model.py','gpuwm/core/clock.py','gpuwm/core/health_ledger.py','gpuwm/core/nest.py','gpuwm/ingest/lateral_bc.py','receipts/libawstep-value.so'):
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()

from contextlib import ExitStack
from whole_inventory import inventory,endpoint
from whole_metadata import Metadata
from whole_period_frame import PeriodFrame,frame_for_period
from time_slots import capture_slots,bind_slots
from gpuwm.core.physics import _physics_interval_steps,_radiation_step_due,_surface_pbl_step_due
original_schedule=clock.execute_schedule

def schedule(*args,**kwargs):
 callbacks={k:v for k,v in kwargs.items() if k.startswith('on_') and v is not None}
 closure={}
 for callback in callbacks.values():
  if callback.__closure__:closure.update(zip(callback.__code__.co_freevars,(c.cell_contents for c in callback.__closure__)))
 poison=closure['poison']
 closure.update(zip(poison.__code__.co_freevars,(c.cell_contents for c in poison.__closure__)))
 model=closure['model'];metadata=Metadata(model,closure)
 for name in ('adaptive_driver','spectral_seam','relocation_runner','history_handler'):
  assert closure.get(name) is None,(name,type(closure.get(name)))
 assert not closure['health_debug'] and not closure['arena_nan_poison']
 assert not closure['steppers']
 objects={'model':model,'status':closure['status'],'ledger':closure['step_ledger'],'feedback_scratch':closure['feedback_scratch'],'validators':closure['validators']}
 live={'period':-1,'plan':None,'stack':None,'replaying':False,'capture_counts':{},'live_counts':{},'ranges':[],'slots':[]}
 def admission():
  import copy
  future=copy.copy(model.root.clock);future.advance()
  assert not closure["restart_enabled"] or not future.restart_due()
  assert not closure["restart_request"].pending()
  for node in model.nodes_by_grid_id.values():
   state=node.state;cfg=node.cfg.run;physics=state.physics
   assert getattr(state,'_streamed_domain',None) is None and getattr(state,'chem',None) is None
   assert not cfg.cu_physics and physics.fire is None
   assert not node.clock.history_due() and not node.clock.lbc_reset_due()
   epoch=float(getattr(state,'domain_start_offset',0.0) or 0.0)
   steps=1 if node.parent is None else 3
   for offset in range(steps):
    itimestep=int(np.floor((state.elapsed_seconds+offset*cfg.dt-epoch)/cfg.dt+0.5))+1
    assert not _radiation_step_due(itimestep,_physics_interval_steps(physics.radt_minutes,cfg.dt),physics.radt_minutes)
    assert _surface_pbl_step_due(itimestep,physics.stepbl,cfg.bldt,None)
   assert all(getattr(physics,name,None) is None for name in ('radiation_due_override','surface_pbl_due_override','cumulus_due_override'))
 def begin(period,clocks):
  callbacks['on_period_begin'](period,clocks);live['period']=period
  live['replaying']=a.arm=='native' and 3<period<8
  if period>=8:
   if live['plan'] is not None:
    cp.cuda.get_current_stream().synchronize();live['plan'].close();live['plan']=None
    receipt['plan_closed_before_ordinary_tail']=True
   return
  if a.arm=='python' or period<3:return
  admission()
  if period==3:
   plan=Plan();plan.trace_failures();stack=ExitStack();live.update(plan=plan,stack=stack)
   live['capture_counts']={gid:c.step_count for gid,c in clocks.items()}
   live['live_counts']=dict(live['capture_counts'])
   stack.enter_context(plan.record(objects))
   stack.enter_context(annotate(plan,receipt.setdefault('annotations',[]),layout_guard=True))
   stack.enter_context(annotate_health(plan,receipt.setdefault('health_annotations',[]),step_label=lambda gid,n:live['live_counts'][gid]+n-live['capture_counts'][gid]))
   stack.enter_context(annotate_descriptors(plan,receipt['health_annotations']))
  else:
   plan=live['plan']
   current={row['label']:key for key,row in inventory(objects).items()}
   assert current==live['identity'],'reachable pointer/layout drift'
   live['live_counts']={gid:c.step_count for gid,c in clocks.items()}
   frame=frame_for_period(model);flags=1 if (clocks[1].step_count+1)%4==0 else 0
   plan.submit(frame,flags=flags);cp.cuda.get_current_stream().synchronize()
   receipt['cases'].append({'period':period,'path':'native','submissions':1,'validator_due':bool(flags),'frame':{name:getattr(frame,name) for name,_ in PeriodFrame._fields_}})
 def step(gid,dom):
  if live['replaying']:return metadata.step(gid,dom)
  if a.arm=='native' and live['period']==3:
   plan=live['plan'];start=plan.library.awstep_count(plan.handle);slots=[];before=metadata.capture_before(gid)
   with capture_slots(plan,slots):callbacks['on_step'](gid,dom)
   metadata.capture_after(gid,before)
   live['ranges'].append((start,plan.library.awstep_count(plan.handle),slots,gid))
  else:callbacks['on_step'](gid,dom)
 def end(period,clocks):
  if live['replaying']:metadata.end()
  elif a.arm=='native' and period==3:
   with live['plan'].when(require=1):callbacks['on_period_end'](period,clocks)
   live['stack'].close();live['stack']=None
   bound=[]
   for index,(start,end,slots,gid) in enumerate(live['ranges']):bound.extend(bind_slots(live['plan'],slots,frame_type=PeriodFrame,prefix=f'd{index}_',start=start,end=end))
   assert len(live['ranges'])==4
   live['identity']={row['label']:key for key,row in inventory(objects).items()}
   receipt['plan']=live['plan'].describe();receipt['bindings']=bound
   receipt['metadata_step_deltas']=metadata.step_deltas
  else:callbacks['on_period_end'](period,clocks)
  if not live['replaying']:receipt['cases'].append({'period':period,'path':'capture' if a.arm=='native' and period==3 else 'python'})
  if period==7+a.ordinary_tail:
   receipt['final']=endpoint(objects);raise Complete()
 def force(*args):return metadata.force(*args) if live['replaying'] else callbacks['on_force'](*args)
 def commit(*args):return metadata.feedback_commit(*args) if live['replaying'] else callbacks['on_feedback_commit'](*args)
 def finalize(*args):return metadata.feedback_finalize(*args) if live['replaying'] else callbacks['on_feedback_finalize'](*args)
 kwargs.update(on_period_begin=begin,on_step=step,on_force=force,on_feedback_commit=commit,on_feedback_finalize=finalize,on_period_end=end)
 try:return original_schedule(*args,**kwargs)
 finally:
  if live['stack'] is not None:live['stack'].__exit__(*sys.exc_info())
  if live['plan'] is not None:cp.cuda.get_current_stream().synchronize();live['plan'].close()
clock.execute_schedule=schedule
command=['--prepared-root',binding['prepared_root'],binding['proof_option'],binding['prepared_head_sha256'],'--experiment-config',binding['config'],'--experiment-config-sha256',binding['config_sha256'],'--progress-format','jsonl','--render-products','none','--io-mode','none','--outdir',str(ROOT/('whole-trajectory-'+str(time.time_ns())))]
receipt['argv']=command;started=time.perf_counter()
try:run(command);raise RuntimeError('did not reach eight parent intervals')
except Complete:receipt['status']='TEN_INTERVALS_WITH_ORDINARY_TAIL_UNCOMPARED' if a.ordinary_tail else 'EIGHT_INTERVALS_UNCOMPARED'
except BaseException as error:receipt.update(status='FAILED',error=repr(error));raise
finally:
 receipt['wall_s']=time.perf_counter()-started
 (ROOT/f'receipts/{"whole-continuation" if a.ordinary_tail else "whole-trajectory"}-{a.numerics}-{a.arm}.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['status'],flush=True)
