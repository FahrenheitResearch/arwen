"""Fixed-time one-submit AC parent+three-child+FORCE+feedback CUDA sequence."""
import argparse,hashlib,json,os,sys,time,types
from collections.abc import Mapping
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
from gpuwm.core import clock
from gpuwm.prepared_domain_tree_forecast import main as run
from gpuwm.native_step import Plan
from host_annotations import annotate
from whole_health_annotations import annotate_health
class Complete(BaseException):pass
receipt={'purpose':__doc__,'pid':os.getpid(),'numerics':a.numerics,'binding':binding,'cases':[],'source':{},'completed_forecast':False,'advancing_native_trajectory':False,'host_metadata':'held from original capture'}
for name in ('gpuwm/core/dycore.py','gpuwm/core/physics.py','gpuwm/core/noahmp_column_slab.py','gpuwm/core/noahmp_flux_slab.py','gpuwm/core/kernels/noahmp_vegeflux.cu','gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/host_annotations.py','tools/native_step/whole_health_annotations.py','tools/native_step/probe_whole_period.py','gpuwm/core/model.py','gpuwm/core/clock.py','gpuwm/core/health_ledger.py','gpuwm/core/nest.py','gpuwm/ingest/lateral_bc.py','receipts/libawstep-health.so'):
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()

def inventory(value):
 seen=set();arrays={}
 def visit(value,label):
  if id(value) in seen:return
  seen.add(id(value))
  if isinstance(value,cp.ndarray):
   key=(value.data.ptr,value.shape,value.strides,value.dtype.str)
   arrays.setdefault(key,{'label':label,'array':value});return
  if isinstance(value,Mapping):
   for name,item in value.items():visit(item,label+'.'+str(name))
  elif isinstance(value,(list,tuple)):
   for i,item in enumerate(value):visit(item,label+f'[{i}]')
  elif isinstance(value,types.FunctionType) and value.__closure__:
   for name,cell in zip(value.__code__.co_freevars,value.__closure__):visit(cell.cell_contents,label+'.closure.'+name)
  elif type(value).__module__.startswith('gpuwm.') and hasattr(value,'__dict__'):
   for name,item in vars(value).items():visit(item,label+'.'+name)
 visit(value,'callbacks');return arrays

original_schedule=clock.execute_schedule

def schedule(*args,**kwargs):
 original_begin=kwargs.get('on_period_begin');original_end=kwargs.get('on_period_end')
 callbacks={k:v for k,v in kwargs.items() if k.startswith('on_') and v is not None}
 active={}
 def begin(period,clocks):
  if original_begin is not None:original_begin(period,clocks)
  if period!=1:return
  inputs=inventory(callbacks)
  print('SNAPSHOT_ADMISSION',json.dumps({'views':len(inputs),'bytes':sum(row['array'].nbytes for row in inputs.values()),'pool_total_bytes':cp.get_default_memory_pool().total_bytes()}),flush=True)
  copies={key:entry['array'].copy() for key,entry in inputs.items()}
  cp.cuda.get_current_stream().synchronize()
  plan=Plan();plan.trace_failures();entry={'period':period,'annotations':[],'health_annotations':[],'start_ticks':{gid:c.ticks for gid,c in clocks.items()},'input_array_views':len(inputs),'input_snapshot_bytes':sum(x.nbytes for x in copies.values())}
  context=plan.record(callbacks);context.__enter__()
  annotations=annotate(plan,entry['annotations'],layout_guard=True);annotations.__enter__()
  health=annotate_health(plan,entry['health_annotations']);health.__enter__()
  active.update(plan=plan,context=context,annotations=annotations,health=health,entry=entry,inputs=inputs,copies=copies)
 def end(period,clocks):
  if period!=1:
   if original_end is not None:original_end(period,clocks)
   return
  plan=active['plan'];entry=active['entry']
  active['health'].__exit__(None,None,None);active['annotations'].__exit__(None,None,None);active['context'].__exit__(None,None,None)
  active['sealed']=True
  cp.cuda.get_current_stream().synchronize()
  entry['plan']=plan.describe();outputs=inventory(callbacks)
  expected={key:hashlib.sha256(cp.asnumpy(row['array']).tobytes()).hexdigest() for key,row in outputs.items()}
  for key,row in active['inputs'].items():row['array'][...]=active['copies'][key]
  cp.cuda.get_current_stream().synchronize()
  plan.submit();cp.cuda.get_current_stream().synchronize()
  rows=[]
  for key,row in outputs.items():
   array=row['array'];digest=hashlib.sha256(cp.asnumpy(array).tobytes()).hexdigest()
   rows.append({'name':row['label'],'shape':list(array.shape),'dtype':array.dtype.str,'words':int(array.size),'reference_sha256':expected[key],'native_sha256':digest,'equal':digest==expected[key]})
  entry.update(end_ticks={gid:c.ticks for gid,c in clocks.items()},native_submissions=1,arrays=rows,different_arrays=[r['name'] for r in rows if not r['equal']],typed_words=sum(r['words'] for r in rows))
  receipt['cases'].append(entry)
  assert not entry['different_arrays'],entry['different_arrays']
  plan.close();active.clear();raise Complete()
 kwargs.update(on_period_begin=begin,on_period_end=end)
 try:return original_schedule(*args,**kwargs)
 finally:
  if active:
   if not active.get('sealed'):
    active['health'].__exit__(*sys.exc_info());active['annotations'].__exit__(*sys.exc_info())
    try:active['context'].__exit__(*sys.exc_info())
    except Exception:pass
   cp.cuda.get_current_stream().synchronize()
   if active['entry'] not in receipt['cases']:
    active['entry']['failed_plan']=active['plan'].describe();receipt['cases'].append(active['entry'])
   active['plan'].close()
clock.execute_schedule=schedule
command=['--prepared-root',binding['prepared_root'],binding['proof_option'],binding['prepared_head_sha256'],'--experiment-config',binding['config'],'--experiment-config-sha256',binding['config_sha256'],'--progress-format','jsonl','--render-products','none','--io-mode','none','--outdir',str(ROOT/('whole-period-check-'+str(time.time_ns())))]
receipt['argv']=command;started=time.perf_counter()
try:
 run(command);raise RuntimeError('did not reach second complete parent interval')
except Complete:receipt['status']='FIXED_TIME_WHOLE_PERIOD_PASS'
except BaseException as error:receipt.update(status='FAILED',error=repr(error));raise
finally:
 receipt['wall_s']=time.perf_counter()-started
 (ROOT/f'receipts/whole-period-{a.numerics}.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(receipt['status'],flush=True)
