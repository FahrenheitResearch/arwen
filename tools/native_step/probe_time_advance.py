"""Native AC domain-step replay at the next natural time; metadata still supplied by the ordinary reference."""
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
from gpuwm.native_step import Plan
from host_annotations import annotate,inventory,host_inventory
from time_slots import Frame,frame_for,capture_slots,bind_slots
class Complete(BaseException):pass
receipt={'purpose':__doc__,'pid':os.getpid(),'numerics':a.numerics,'binding':binding,'cases':[], 'source':{},'completed_forecast':False}
for name in ('gpuwm/core/dycore.py','gpuwm/core/physics.py','gpuwm/core/noahmp_column_slab.py','gpuwm/core/noahmp_flux_slab.py','gpuwm/core/kernels/noahmp_vegeflux.cu','gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/host_annotations.py','tools/native_step/probe_time_advance.py','tools/native_step/time_slots.py','gpuwm/ingest/lateral_bc.py'):
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()

def snapshot(arrays):
 return {key:entry['array'].copy() for key,entry in arrays.items()}
def restore(arrays,saved):
 for key,entry in arrays.items():cp.copyto(entry['array'],saved[key])
 cp.cuda.get_current_stream().synchronize()
def compare(arrays,reference):
 total=changed=0;details=[]
 for key,entry in arrays.items():
  array=entry['array'];got=cp.asnumpy(array);want=cp.asnumpy(reference[key]);width=got.dtype.itemsize
  g=np.frombuffer(got.tobytes(),np.uint8).reshape(-1,width);w=np.frombuffer(want.tobytes(),np.uint8).reshape(-1,width)
  n=int(np.count_nonzero(np.any(g!=w,axis=1)))
  total+=got.size;changed+=n
  details.append({'name':entry['label'],'shape':list(got.shape),'dtype':got.dtype.str,'words':got.size,'different_words':n,'expected_sha256':hashlib.sha256(want.tobytes()).hexdigest(),'actual_sha256':hashlib.sha256(got.tobytes()).hexdigest()})
 return {'words':total,'different_words':changed,'arrays':details}
original=dycore.step;counts={};plans={};finished=set()

def step(state,cfg,**kwargs):
 key=(cfg.nx,cfg.ny,cfg.nz);counts[key]=counts.get(key,0)+1
 capture_at={(120,70,44):2,(267,117,44):4}.get(key)
 if capture_at is None or counts[key] not in (capture_at,capture_at+1):
  return original(state,cfg,**kwargs)
 if counts[key]==capture_at:
  before=inventory(state);host_before=host_inventory(state)
  plan=Plan();plan.trace_failures();slots=[];annotations=[];frame=frame_for(state,cfg)
  try:
   with plan.record(state,cfg):
    with capture_slots(plan,slots),annotate(plan,annotations,layout_guard=True):
     original(state,cfg,**kwargs)
   cp.cuda.get_current_stream().synchronize()
   bound=bind_slots(plan,slots);after=inventory(state);host_after=host_inventory(state)
   entry={'shape':list(key),'captured_time':state.elapsed_seconds-cfg.dt,
    'slot_bindings':bound,'host_annotations':annotations,'plan':plan.describe(),
    'capture_host_changes':{k:{'before':host_before.get(k),'after':v} for k,v in host_after.items() if host_before.get(k)!=v}}
   receipt['cases'].append(entry)
   plans[key]=(plan,before,after,frame,entry)
  except BaseException:
   cp.cuda.get_current_stream().synchronize();plan.close();raise
  return
 plan,captured_before,captured_after,frozen_frame,entry=plans[key]
 entry['replay_time']=state.elapsed_seconds
 before=inventory(state);saved=snapshot(before);frame=frame_for(state,cfg)
 entry['frame_before']={name:getattr(frozen_frame,name) for name,_ in Frame._fields_}
 entry['frame_now']={name:getattr(frame,name) for name,_ in Frame._fields_}
 host_before=host_inventory(state)
 original(state,cfg,**kwargs)
 cp.cuda.get_current_stream().synchronize()
 after=inventory(state);expected=snapshot(after);host_after=host_inventory(state)
 entry['reference_host_changes']={k:{'before':host_before.get(k),'after':v} for k,v in host_after.items() if host_before.get(k)!=v}
 input_by_name={v['label']:saved[k] for k,v in before.items()}
 expected_by_name={v['label']:expected[k] for k,v in after.items()}
 missing=[v['label'] for v in captured_before.values() if v['label'] not in input_by_name]
 assert not missing,missing
 reference={k:expected_by_name[v['label']] for k,v in captured_after.items()}
 def restore_inputs():
  restore(before,saved)
  for v in captured_before.values():cp.copyto(v['array'],input_by_name[v['label']])
  cp.cuda.get_current_stream().synchronize()
 try:
  restore_inputs();plan.submit(frozen_frame);cp.cuda.get_current_stream().synchronize()
  primary={k:v for k,v in captured_after.items() if v['label'] in ('state.u','state.v','state.w','state.thp','state.php','state.mup','state.qv','state.qc','state.qr')}
  entry['frozen_frame_negative_control']=compare(primary,reference)
  restore_inputs();plan.submit(frame);cp.cuda.get_current_stream().synchronize()
  entry['compare']=compare(captured_after,reference)
  entry['submissions']=1
  restore(after,expected)
  assert entry['compare']['different_words']==0,[(r['name'],r['different_words']) for r in entry['compare']['arrays'] if r['different_words']]
  print(json.dumps({'shape':list(key),'time':entry['replay_time'],'words':entry['compare']['words'],'differences':entry['compare']['different_words'],'frozen_differences':entry['frozen_frame_negative_control']['different_words']}),flush=True)
 finally:
  cp.cuda.get_current_stream().synchronize();plan.close();del plans[key]
 finished.add(key)
 if len(finished)==2:raise Complete()

dycore.step=step
command=['--prepared-root',binding['prepared_root'],binding['proof_option'],binding['prepared_head_sha256'],'--experiment-config',binding['config'],'--experiment-config-sha256',binding['config_sha256'],'--progress-format','jsonl','--render-products','none','--io-mode','none','--outdir',str(ROOT/('time-advance-check-'+str(time.time_ns())))]
receipt['argv']=command;started=time.perf_counter()
try:
 run(command);raise RuntimeError('did not reach parent and child')
except Complete:receipt['status']='PASS_NEXT_TIME_DOMAIN_REPLAY_METADATA_FROM_REFERENCE'
except BaseException as error:
 receipt.update(status='FAILED',error=repr(error));raise
finally:
 for plan,*rest in plans.values():
  cp.cuda.get_current_stream().synchronize();plan.close()
 receipt['wall_s']=time.perf_counter()-started
 (ROOT/f'receipts/time-advance-{a.numerics}.json').write_text(json.dumps(receipt,indent=2)+'\n')
