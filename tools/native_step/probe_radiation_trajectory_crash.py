"""Forty AC parent intervals with explicit ordinary radiation fallback and recapture."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--arm',choices=('python','native'),required=True);p.add_argument('--numerics',choices=('strict','fast'),required=True)
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
from host_annotations import annotate
from time_slots import Frame,frame_for,capture_slots,bind_slots
class Complete(BaseException):pass
receipt={'purpose':__doc__,'pid':os.getpid(),'numerics':a.numerics,'binding':binding,'cases':[], 'arm':a.arm, 'source':{},'completed_forecast':False}
for name in ('gpuwm/core/dycore.py','gpuwm/core/physics.py','gpuwm/core/noahmp_column_slab.py','gpuwm/core/noahmp_flux_slab.py','gpuwm/core/kernels/noahmp_vegeflux.cu','gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/host_annotations.py','tools/native_step/probe_radiation_trajectory_crash.py','receipts/recorder-crash-trace.c','gpuwm/core/rrtmg_legacy.py','tools/native_step/time_slots.py','gpuwm/ingest/lateral_bc.py'):
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()

def inventory(state):
 seen=set();arrays={}
 def visit(value,label):
  if id(value) in seen:return
  seen.add(id(value))
  if isinstance(value,cp.ndarray):
   key=(value.data.ptr,value.shape,value.strides,value.dtype.str)
   arrays.setdefault(key,{'label':label,'array':value});return
  if isinstance(value,dict):
   for name,item in value.items():visit(item,label+'.'+str(name))
  elif isinstance(value,(list,tuple)):
   for i,item in enumerate(value):visit(item,label+f'[{i}]')
  elif type(value).__module__.startswith('gpuwm.') and hasattr(value,'__dict__'):
   for name,item in vars(value).items():visit(item,label+'.'+name)
 visit(state,'state')
 return arrays

def host_inventory(state):
 seen=set();values={}
 def visit(value,label):
  if isinstance(value,(bool,int,float,str,type(None))):
   values[label]=value;return
  if id(value) in seen:return
  seen.add(id(value))
  if isinstance(value,(cp.ndarray,np.ndarray)):return
  if isinstance(value,dict):
   for k,v in value.items():visit(v,label+'.'+str(k))
  elif isinstance(value,(list,tuple)):
   for i,v in enumerate(value):visit(v,label+f'[{i}]')
  elif type(value).__module__.startswith('gpuwm.') and hasattr(value,'__dict__'):
   for k,v in vars(value).items():visit(v,label+'.'+k)
 visit(state,'state');return values

from gpuwm.core.physics import _physics_interval_steps, _radiation_step_due, _surface_pbl_step_due
original=dycore.step;counts={};plans={};states={}
def phase(state,cfg,kwargs):
    physics=state.physics
    epoch=float(getattr(state,'domain_start_offset',0.0) or 0.0)
    itimestep=int(np.floor((state.elapsed_seconds-epoch)/cfg.dt+0.5))+1
    assert all(getattr(physics,name,None) is None for name in ('radiation_due_override','surface_pbl_due_override','cumulus_due_override'))
    assert not cfg.cu_physics and physics.fire is None and getattr(state,'chem',None) is None
    return (_radiation_step_due(itimestep,_physics_interval_steps(physics.radt_minutes,cfg.dt),physics.radt_minutes),
            _surface_pbl_step_due(itimestep,physics.stepbl,cfg.bldt,None),tuple(sorted(kwargs.items())))
def finish():
    cp.cuda.get_current_stream().synchronize()
    receipt['final']=[]
    for key,state in states.items():
        rows=[]
        arrays=inventory(state)
        addresses={}
        for item in arrays.values():addresses.setdefault(item['array'].data.ptr,[]).append(item['label'])
        def pointer(value):
            value=int(value)
            if value==0:return None
            assert value in addresses,('unresolved device address',value)
            return sorted(addresses[value])
        for entry in arrays.values():
            x=cp.asnumpy(entry['array'])
            row={'name':entry['label'],'shape':list(x.shape),'dtype':x.dtype.str,'words':int(x.size),'sha256':hashlib.sha256(x.tobytes()).hexdigest()}
            if entry['label'] in ('state._scratch.integration_health_field_ptr','state._scratch.integration_health_aux_ptr'):
                row['pointer_targets']=[pointer(v) for v in x.view(np.uint64)]
            rows.append(row)
        host=host_inventory(state)
        for name,value in list(host.items()):
            if any(name.startswith('state.'+prefix+'[0]') for prefix in ('_rk_copy_launch','_rk_zero_launch','_glue_add_launch')) and name.endswith('[0]'):
                host[name]={'pointer_target':pointer(value)}
        receipt['final'].append({'shape':list(key),'arrays':rows,'host':host})
    raise Complete()

def step(state,cfg,**kwargs):
    key=(cfg.nx,cfg.ny,cfg.nz);states[key]=state;counts[key]=counts.get(key,0)+1
    if key==(120,70,44) and counts[key]==41:finish()
    capture_at={(120,70,44):2,(267,117,44):4}.get(key)
    assert capture_at is not None,key
    entry={'shape':list(key),'call':counts[key],'start_seconds':state.elapsed_seconds}
    marker_now=phase(state,cfg,kwargs)
    if a.arm=='native' and key in plans and marker_now[:2]==(False,True):
        old,_,captured=plans[key]
        current=inventory(state)
        old_by_name={x['label']:k for k,x in captured.items()}
        new_by_name={x['label']:k for k,x in current.items()}
        if old_by_name!=new_by_name:
            entry['invalidated_array_paths']=[name for name in sorted(set(old_by_name)|set(new_by_name)) if old_by_name.get(name)!=new_by_name.get(name)]
            print(f'EXPLICIT DIAGNOSTIC RECAPTURE shape={key} time={state.elapsed_seconds}: reachable arrays changed',flush=True)
            cp.cuda.get_current_stream().synchronize();old.close();plans.pop(key)
    if a.arm=='python' or counts[key]<capture_at:
        original(state,cfg,**kwargs);entry['path']='python'
    elif marker_now[:2]!=(False,True):
        if key in plans:
            old,*_=plans.pop(key);cp.cuda.get_current_stream().synchronize();old.close()
        print(f'EXPLICIT DIAGNOSTIC FALLBACK shape={key} time={state.elapsed_seconds} phase={marker_now[:2]}: ordinary Python radiation; next non-radiation call recaptures',flush=True)
        original(state,cfg,**kwargs);entry['path']='ordinary_radiation_fallback'
    elif key not in plans:
        marker=marker_now
        plan=Plan();plan.trace_failures();slots=[];annotations=[]
        with plan.record(state,cfg):
            with capture_slots(plan,slots),annotate(plan,annotations,layout_guard=True):
                original(state,cfg,**kwargs)
        cp.cuda.get_current_stream().synchronize()
        bound=bind_slots(plan,slots)
        plans[key]=(plan,marker,inventory(state))
        entry.update(path='capture',bindings=bound,operations=plan.describe()['operations'])
    else:
        plan,marker,captured=plans[key]
        assert phase(state,cfg,kwargs)==marker,'cadence or call arguments changed'
        current=inventory(state)
        old_by_name={x['label']:k for k,x in captured.items()}
        new_by_name={x['label']:k for k,x in current.items()}
        assert old_by_name==new_by_name,'reachable array identity/layout changed'
        if cfg.nested and state.elapsed_seconds==1782.0:os.environ['AWSTEP_CRASH_TRACE']='1'
        plan.submit(frame_for(state,cfg))
        os.environ.pop('AWSTEP_CRASH_TRACE',None)
        # Diagnostic publishes only after completion. General asynchronous
        # model admission/error semantics have not been implemented.
        cp.cuda.get_current_stream().synchronize()
        for name in ('sfclay','noah','ysu'):state.physics.call_counts[name]+=1
        state.physics.microphysics_updates+=1
        state.elapsed_seconds+=cfg.dt
        entry['path']='native';entry['submissions']=1
    entry['end_seconds']=state.elapsed_seconds
    receipt['cases'].append(entry)

dycore.step=step
command=['--prepared-root',binding['prepared_root'],binding['proof_option'],binding['prepared_head_sha256'],'--experiment-config',binding['config'],'--experiment-config-sha256',binding['config_sha256'],'--progress-format','jsonl','--render-products','none','--io-mode','none','--outdir',str(ROOT/('radiation-trajectory-'+str(time.time_ns())))]
receipt['argv']=command;started=time.perf_counter()
try:
    run(command);raise RuntimeError('did not reach forty complete parent intervals')
except Complete:receipt['status']='FORTY_INTERVALS_WITH_RADIATION_FALLBACK_UNCOMPARED'
except BaseException as error:
    receipt.update(status='FAILED',error=repr(error));raise
finally:
    for plan,*rest in plans.values():
        cp.cuda.get_current_stream().synchronize();plan.close()
    receipt['wall_s']=time.perf_counter()-started
    (ROOT/f'receipts/radiation-trajectory-crash-{a.numerics}-{a.arm}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt['status'],flush=True)
