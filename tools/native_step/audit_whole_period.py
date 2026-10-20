"""Audit one complete non-radiation parent/nest interval including FORCE and feedback."""
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
from time_slots import Frame,frame_for,capture_slots,bind_slots
class Complete(BaseException):pass
receipt={'purpose':__doc__,'pid':os.getpid(),'numerics':a.numerics,'binding':binding,'cases':[], 'source':{},'completed_forecast':False}
for name in ('gpuwm/core/dycore.py','gpuwm/core/physics.py','gpuwm/core/noahmp_column_slab.py','gpuwm/core/noahmp_flux_slab.py','gpuwm/core/kernels/noahmp_vegeflux.cu','gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/host_annotations.py','tools/native_step/audit_whole_period.py','gpuwm/core/model.py','gpuwm/core/clock.py','tools/native_step/time_slots.py','gpuwm/ingest/lateral_bc.py'):
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()

from gpuwm.core import clock
original_schedule=clock.execute_schedule

def schedule(*args,**kwargs):
 original_begin=kwargs.get('on_period_begin');original_end=kwargs.get('on_period_end')
 callback=kwargs['on_step']
 closure=dict(zip(callback.__code__.co_freevars,(c.cell_contents for c in callback.__closure__)))
 model=closure['model'];active={}
 def begin(period,clocks):
  if original_begin is not None:original_begin(period,clocks)
  if period!=1:return
  plan=Plan(diagnostic=True);entry={'period':period,'annotations':[],'start_seconds':model.root.state.elapsed_seconds}
  context=plan.record(model);context.__enter__()
  annotations=annotate(plan,entry['annotations'],layout_guard=True);annotations.__enter__()
  active.update(plan=plan,context=context,annotations=annotations,entry=entry)
 def end(period,clocks):
  if period==1:
   plan=active['plan'];entry=active['entry']
   active['annotations'].__exit__(None,None,None)
   try:active['context'].__exit__(None,None,None)
   except RecordingError as error:entry['error']=str(error)
   cp.cuda.get_current_stream().synchronize()
   entry['plan']=plan.describe();entry['end_seconds']=model.root.state.elapsed_seconds
   receipt['cases'].append(entry);plan.close();active.clear()
   raise Complete()
  if original_end is not None:original_end(period,clocks)
 kwargs.update(on_period_begin=begin,on_period_end=end)
 try:return original_schedule(*args,**kwargs)
 finally:
  if active:
   active['annotations'].__exit__(*sys.exc_info())
   try:active['context'].__exit__(*sys.exc_info())
   except RecordingError:pass
   cp.cuda.get_current_stream().synchronize();active['plan'].close()
clock.execute_schedule=schedule
command=['--prepared-root',binding['prepared_root'],binding['proof_option'],binding['prepared_head_sha256'],'--experiment-config',binding['config'],'--experiment-config-sha256',binding['config_sha256'],'--progress-format','jsonl','--render-products','none','--io-mode','none','--outdir',str(ROOT/('whole-period-audit-'+str(time.time_ns())))]
receipt['argv']=command;started=time.perf_counter()
try:
 run(command);raise RuntimeError('did not reach second complete parent interval')
except Complete:receipt['status']='DIAGNOSTIC_WHOLE_PERIOD_AUDIT_COMPLETE'
except BaseException as error:
 receipt.update(status='FAILED',error=repr(error));raise
finally:
 receipt['wall_s']=time.perf_counter()-started
 (ROOT/f'receipts/whole-period-audit-{a.numerics}.json').write_text(json.dumps(receipt,indent=2)+'\n')
