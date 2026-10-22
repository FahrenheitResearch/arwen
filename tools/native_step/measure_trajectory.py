"""Twelve parent intervals, unprofiled wall or separate CUDA marker profile; pinned AC diagnostic only."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--profile-only',action='store_true');p.add_argument('--arm',choices=('python','native'),required=True);p.add_argument('--numerics',choices=('strict','fast'),required=True)
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
from host_annotations import annotate,inventory,host_inventory,phase,finish_trajectory
from time_slots import Frame,frame_for,capture_slots,bind_slots
class Complete(BaseException):pass
receipt={'purpose':__doc__,'pid':os.getpid(),'numerics':a.numerics,'binding':binding,'cases':[], 'arm':a.arm, 'source':{},'completed_forecast':False}
for name in ('gpuwm/core/dycore.py','gpuwm/core/physics.py','gpuwm/core/noahmp_column_slab.py','gpuwm/core/noahmp_flux_slab.py','gpuwm/core/kernels/noahmp_vegeflux.cu','gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/host_annotations.py','tools/native_step/measure_trajectory.py','tools/native_step/time_slots.py','gpuwm/ingest/lateral_bc.py'):
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()

original=dycore.step;counts={};plans={};states={};interval_started=None;interval_index=None
receipt['parent_intervals_ms']=[];receipt['profile_only']=a.profile_only
markers=None
if a.profile_only:
    module=cp.RawModule(code='extern \"C\" __global__ void awstep_parent_interval_start() {}\nextern \"C\" __global__ void awstep_parent_interval_end() {}')
    markers=(module.get_function('awstep_parent_interval_start'),module.get_function('awstep_parent_interval_end'))
def finish():
    finish_trajectory(states,receipt,Complete)

def step(state,cfg,**kwargs):
    global interval_started,interval_index
    key=(cfg.nx,cfg.ny,cfg.nz);states[key]=state;counts[key]=counts.get(key,0)+1
    if key==(120,70,44):
        if interval_started is not None:
            if markers:markers[1]((1,),(1,),())
            cp.cuda.get_current_stream().synchronize()
            receipt['parent_intervals_ms'].append({'parent_step':interval_index,'ms':(time.perf_counter()-interval_started)*1000})
        if counts[key]==13:finish()
        cp.cuda.get_current_stream().synchronize()
        interval_started=time.perf_counter();interval_index=counts[key]
        if markers:markers[0]((1,),(1,),())
    capture_at={(120,70,44):2,(267,117,44):4}.get(key)
    assert capture_at is not None,key
    entry={'shape':list(key),'call':counts[key],'start_seconds':state.elapsed_seconds}
    if a.arm=='python' or counts[key]<capture_at:
        original(state,cfg,**kwargs);entry['path']='python'
    elif counts[key]==capture_at:
        marker=phase(state,cfg,kwargs);assert marker[:2]==(False,True),marker
        plan=Plan();plan.trace_failures();slots=[];annotations=[]
        with plan.record(state,cfg):
            with capture_slots(plan,slots),annotate(plan,annotations,layout_guard=True):
                original(state,cfg,**kwargs)
        cp.cuda.get_current_stream().synchronize()
        bound=bind_slots(plan,slots)
        plans[key]=(plan,marker,{n:getattr(state,n).data.ptr for n in ('u','v','w','thp','php','mup')})
        entry.update(path='capture',bindings=bound,operations=plan.describe()['operations'])
    else:
        plan,marker,captured=plans[key]
        assert phase(state,cfg,kwargs)==marker,'cadence or call arguments changed'
        # Source/config pinned diagnostic. Full buffer identity is checked
        # in probe_trajectory, excluded from this wall arm.
        assert tuple((getattr(state,n).data.ptr) for n in ('u','v','w','thp','php','mup'))==tuple(captured[n] for n in ('u','v','w','thp','php','mup'))
        plan.submit(frame_for(state,cfg))
        # Host metadata follows enqueue, as in the ordinary Python path.
        # The parent interval endpoint supplies the timing completion fence.
        for name in ('sfclay','noah','ysu'):state.physics.call_counts[name]+=1
        state.physics.microphysics_updates+=1
        state.elapsed_seconds+=cfg.dt
        entry['path']='native';entry['submissions']=1
    entry['end_seconds']=state.elapsed_seconds
    receipt['cases'].append(entry)

dycore.step=step
command=['--prepared-root',binding['prepared_root'],binding['proof_option'],binding['prepared_head_sha256'],'--experiment-config',binding['config'],'--experiment-config-sha256',binding['config_sha256'],'--progress-format','jsonl','--render-products','none','--io-mode','none','--outdir',str(ROOT/('trajectory-check-'+str(time.time_ns())))]
receipt['argv']=command;started=time.perf_counter()
try:
    run(command);raise RuntimeError('did not reach twelve complete parent intervals')
except Complete:receipt['status']='TWELVE_INTERVALS_COMPLETE_UNCOMPARED'
except BaseException as error:
    receipt.update(status='FAILED',error=repr(error));raise
finally:
    for plan,*rest in plans.values():
        cp.cuda.get_current_stream().synchronize();plan.close()
    receipt['wall_s']=time.perf_counter()-started
    (ROOT/f"receipts/trajectory-{'profile' if a.profile_only else 'wall'}-{a.numerics}-{a.arm}.json").write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt['status'],flush=True)
