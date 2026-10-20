"""Audit original radiation with pinned native deferred verdict consumer; never replay."""
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
from gpuwm.native_step import Plan,RecordingError
from host_annotations import annotate
from deferred_annotations import annotate_deferred
from time_slots import Frame,frame_for,capture_slots,bind_slots
class Complete(BaseException):pass
receipt={'purpose':__doc__,'pid':os.getpid(),'numerics':a.numerics,'binding':binding,'cases':[], 'source':{},'completed_forecast':False}
for name in ('gpuwm/core/dycore.py','gpuwm/core/physics.py','gpuwm/core/noahmp_column_slab.py','gpuwm/core/noahmp_flux_slab.py','gpuwm/core/kernels/noahmp_vegeflux.cu','gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/host_annotations.py','tools/native_step/audit_deferred_radiation.py','tools/native_step/deferred_annotations.py','gpuwm/core/deferred_device_checks.py','receipts/libawstep-host-argument.so','tools/native_step/time_slots.py','gpuwm/ingest/lateral_bc.py'):
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()

from gpuwm.core.physics import _physics_interval_steps,_radiation_step_due
original=dycore.step;counts={};done=set()
def step(state,cfg,**kwargs):
 key=(cfg.nx,cfg.ny,cfg.nz)
 epoch=float(getattr(state,'domain_start_offset',0.0) or 0.0)
 itimestep=int(np.floor((state.elapsed_seconds-epoch)/cfg.dt+0.5))+1
 physics=state.physics
 due=_radiation_step_due(itimestep,_physics_interval_steps(physics.radt_minutes,cfg.dt),physics.radt_minutes)
 if due:counts[key]=counts.get(key,0)+1
 if not due or counts[key]!=2 or key in done:return original(state,cfg,**kwargs)
 plan=Plan(diagnostic=True);entry={'shape':list(key),'model_time':state.elapsed_seconds,'itimestep':itimestep,'annotations':[]}
 try:
  try:
   with plan.record(state,cfg):
    with annotate(plan,entry['annotations'],layout_guard=True),annotate_deferred(plan,entry.setdefault('deferred_annotations',[])):original(state,cfg,**kwargs)
  except RecordingError as error:entry['error']=str(error)
  cp.cuda.get_current_stream().synchronize()
  entry['plan']=plan.describe();receipt['cases'].append(entry)
  assert entry['plan']['error']=='diagnostic capture cannot replay',entry['plan']['error']
 finally:plan.close()
 done.add(key)
 if len(done)==2:raise Complete()
dycore.step=step
command=['--prepared-root',binding['prepared_root'],binding['proof_option'],binding['prepared_head_sha256'],'--experiment-config',binding['config'],'--experiment-config-sha256',binding['config_sha256'],'--progress-format','jsonl','--render-products','none','--io-mode','none','--outdir',str(ROOT/('deferred-radiation-audit-'+str(time.time_ns())))]
receipt['argv']=command;started=time.perf_counter()
try:
 run(command);raise RuntimeError('did not reach both second radiation calls')
except Complete:receipt['status']='DIAGNOSTIC_DEFERRED_RADIATION_AUDIT_COMPLETE'
except BaseException as error:
 receipt.update(status='FAILED',error=repr(error));raise
finally:
 receipt['wall_s']=time.perf_counter()-started
 (ROOT/f'receipts/deferred-radiation-audit-{a.numerics}.json').write_text(json.dumps(receipt,indent=2)+'\n')
