"""Compare one device solar-constant transfer on second natural AC radiation calls. No timing claim."""
import argparse,ast,copy,dataclasses,hashlib,json,os,statistics,sys,time
from pathlib import Path
from datetime import timedelta
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--numerics',choices=('strict','fast'),required=True)
p.add_argument('--daytime-control',action='store_true')
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
klass=module.RRTMGLegacyRadiation;baseline=klass.__call__
base_path=ROOT/'receipts/solar-transport-candidate.py'
original_source=(ROOT/'gpuwm/core/rrtmg_legacy.py').read_text()
assert hashlib.sha256(original_source.encode()).hexdigest()=='6270ca552b88fb6bdfd7e31018b1de2cad311db30245f3075160be979f90a240'
needle='            declin, solcon = map(F, cp.asnumpy(ds))'
assert original_source.count(needle)==1
assert base_path.read_text()==original_source.replace(needle,'            solcon = ds[1]')

tree=ast.parse(base_path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==klass.__name__)
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
namespace=dict(vars(module));exec(compile(ast.Module(body=[method],type_ignores=[]),str(base_path),'exec'),namespace);candidate=namespace['__call__']
receipt={'purpose':__doc__,'pid':os.getpid(),'numerics':a.numerics,'binding':binding,'source':{},'cases':[],'completed_forecast':False}
for name in ('gpuwm/core/rrtmg_legacy.py','receipts/solar-transport-candidate.py','tools/native_step/probe_solar_transport.py','gpuwm/core/physics.py','gpuwm/native_step/recorder.c'):
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
class Complete(BaseException):pass
counts={};done=set()
def sync():cp.cuda.get_current_stream().synchronize()
def arrays(value,prefix=''):
 if isinstance(value,(cp.ndarray,np.ndarray)):yield prefix,value
 elif dataclasses.is_dataclass(value):
  for field in dataclasses.fields(value):yield from arrays(getattr(value,field.name),prefix+'.'+field.name)
 elif isinstance(value,dict):
  for key,v in sorted(value.items()):yield from arrays(v,prefix+'.'+str(key))
def bits(value):return {name:cp.asnumpy(v) if isinstance(v,cp.ndarray) else v.copy() for name,v in arrays(value)}
def compare(got,want):
 assert got.keys()==want.keys(),(got.keys(),want.keys())
 rows=[]
 for name,x in got.items():
  y=want[name];assert x.dtype==y.dtype and x.shape==y.shape,(name,x.dtype,y.dtype,x.shape,y.shape)
  diff=int(np.count_nonzero(np.any(x.view(np.uint8).reshape(-1,x.dtype.itemsize)!=y.view(np.uint8).reshape(-1,y.dtype.itemsize),axis=1)))
  rows.append({'name':name,'shape':list(x.shape),'dtype':str(x.dtype),'words':x.size,'different_words':diff})
 result={'arrays':rows,'words':sum(r['words'] for r in rows),'different_words':sum(r['different_words'] for r in rows)}
 assert result['different_words']==0,result
 return result
def clone(adapter):
 obj=copy.copy(adapter)
 if obj.__dict__.get('_ozmixt_buf') is not None:obj._ozmixt_buf=obj._ozmixt_buf.copy()
 return obj
def input_hash(kwargs):
 values=dict(kwargs['atmosphere']);values.update({'fields.'+k:v for k,v in kwargs['fields'].items() if k not in RADIATION_BOUNDARY_FLUXES})
 values.update({'state.'+k:v for k,v in vars(kwargs['state']).items() if isinstance(v,(cp.ndarray,np.ndarray))})
 values['ozone']=kwargs['state'].physics.o3rad
 return {k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in bits(values).items()}

def check(adapter,**kwargs):
 counts[id(adapter)]=counts.get(id(adapter),0)+1
 if counts[id(adapter)]!=2:return baseline(adapter,**kwargs)
 state=kwargs['state'];cfg=kwargs['cfg'];key=(cfg.nx,cfg.ny,cfg.nz)
 assert key not in done
 row={'shape':list(key),'model_time':state.elapsed_seconds,'update_count_before':adapter.update_count}
 before=input_hash(kwargs);old=clone(adapter);new=clone(adapter)
 if a.daytime_control:
  old.start_time+=timedelta(hours=18);new.start_time+=timedelta(hours=18)
  row['calendar_offset_hours']=18
 output_initial={k:v.copy() for k,v in kwargs['fields'].items() if k in RADIATION_BOUNDARY_FLUXES}
 row['declared_output_buffers']=sorted(output_initial)
 def restore_outputs():
  for k,v in output_initial.items():cp.copyto(kwargs['fields'][k],v)
 want=bits(baseline(old,**kwargs));want['retained_ozone']=cp.asnumpy(old.o33d_device())
 after_old=input_hash(kwargs)
 row['day_columns']=int(np.count_nonzero(want['.coszen']>np.float32(0)))
 row['night_columns']=int(np.count_nonzero(want['.coszen']<=np.float32(0)))
 if a.daytime_control:assert row['day_columns']>0,'daytime calendar control did not produce daylight'
 row['base_input_changes']=[k for k in before if before[k]!=after_old.get(k)]
 restore_outputs()
 got=bits(candidate(new,**kwargs));got['retained_ozone']=cp.asnumpy(new.o33d_device())
 row['comparison']=compare(got,want);assert old.update_count==new.update_count==adapter.update_count+1
 after_new=input_hash(kwargs)
 row['candidate_input_changes']=[k for k in after_old if after_old[k]!=after_new.get(k)]
 if after_new!=before:
  receipt['cases'].append(row)
  raise AssertionError(('radiation changed input array words',row['base_input_changes'],row['candidate_input_changes']))
 row['audit']={}
 for label,fn,obj in ([] if a.daytime_control else [('base',baseline,old),('candidate',candidate,new)]):
  restore_outputs();sync();plan=Plan(diagnostic=True)
  try:
   try:
    with plan.record(kwargs,obj):result=fn(obj,**kwargs)
   except RecordingError as error:row['audit'][label]={'error':str(error)}
   sync();row['audit'][label]['plan']=plan.describe()
  finally:plan.close()
 row['inputs_unchanged']=input_hash(kwargs)==before;assert row['inputs_unchanged']
 receipt['cases'].append(row);done.add(key)
 print(json.dumps({k:v for k,v in row.items() if k not in ('audit','comparison','retention_after_provider_change','lazy_host')}),flush=True)
 if len(done)==2:raise Complete()
 restore_outputs()
 return baseline(adapter,**kwargs)
klass.__call__=check
command=['--prepared-root',binding['prepared_root'],binding['proof_option'],binding['prepared_head_sha256'],'--experiment-config',binding['config'],'--experiment-config-sha256',binding['config_sha256'],'--progress-format','jsonl','--render-products','none','--io-mode','none','--outdir',str(ROOT/('solar-transport-'+str(time.time_ns())))]
receipt['argv']=command;receipt['calendar_offset_hours']=18 if a.daytime_control else 0;receipt['audit_skipped_for_daytime_control']=a.daytime_control
try:
 run(command);raise RuntimeError('second radiation calls not both reached')
except Complete:receipt['status']='EXACT_SOLAR_TRANSPORT_PAIR'
except BaseException as error:receipt.update(status='FAILED',error=repr(error));raise
finally:
 suffix=('daytime-' if a.daytime_control else '')+a.numerics
 (ROOT/f'receipts/solar-transport-{suffix}.json').write_text(json.dumps(receipt,indent=2)+'\n')
