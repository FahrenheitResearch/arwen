"""Actual AC root ozone native producer/upload/kernel component; no timing claim."""
import argparse,ctypes as C,hashlib,json,os,sys,time
from datetime import timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--numerics',choices=('strict','fast'),required=True)
a=p.parse_args();binding=json.loads((ROOT/'receipts/local-binding.json').read_text())
mask=os.environ['CUDA_VISIBLE_DEVICES'];os.environ.update(binding['runtime_env']);os.environ.update(CUDA_VISIBLE_DEVICES=mask,GPUWM_NUMERICS=a.numerics)
for key,relative in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','CUPY_CACHE_DIR':'cache/cupy','CUDA_CACHE_PATH':'cache/cuda','GPUWM_GENERATED_TABLE_ROOT':'cache/generated'}.items():
 path=ROOT/relative;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
os.environ['PYTHONPATH']=str(ROOT);sys.path.insert(0,str(ROOT))
import cupy as cp
import numpy as np
from gpuwm.core import rrtmg_legacy as module,portable_math,solar
from gpuwm.ingest import wrf_ozone as ozone
from gpuwm.prepared_domain_tree_forecast import main as run
from gpuwm.native_step import Plan
class Args(C.Structure):
 _fields_=[('function',C.c_void_p),('values',C.c_void_p),('nvalues',C.c_size_t),('latitude',C.c_void_p),('nlatitude',C.c_size_t),('ozone',C.c_void_p),('levels',C.c_size_t),('months',C.c_size_t),('nm',C.c_size_t),('np',C.c_size_t),('fact1',C.c_float),('fact2',C.c_float),('workers',C.c_size_t),('out',C.c_void_p)]
class Frame(C.Structure):_fields_=[('args',Args)]
class Daylight(C.Structure):
 _fields_=[('coszen',C.c_void_p),('ncols',C.c_size_t),('capacity',C.c_size_t),('workers',C.c_size_t),('day',C.c_void_p),('night',C.c_void_p),('counts',C.c_void_p),('work',C.c_void_p)]
class Complete(BaseException):pass
receipt={'purpose':__doc__,'pid':os.getpid(),'numerics':a.numerics,'binding':binding,'source':{},'cases':[],'completed_forecast':False,'model_state_publication':False}
for name in ('gpuwm/core/rrtmg_legacy.py','gpuwm/ingest/wrf_ozone.py','gpuwm/native_step/recorder.c','gpuwm/native_step/__init__.py','tools/native_step/radiation_host.c','tools/native_step/probe_native_ozone.py'):
 receipt['source'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
receipt['preprocess_library_sha256']=hashlib.sha256(Path(os.environ['GPUWM_CPU_PREPROCESS_BRIDGE']).read_bytes()).hexdigest()
assert receipt['source']['gpuwm/core/rrtmg_legacy.py']=='6270ca552b88fb6bdfd7e31018b1de2cad311db30245f3075160be979f90a240'
klass=module.RRTMGLegacyRadiation;baseline=klass.__call__;original_provider=ozone.ozn_latitude_time_int;original_kernel=module._adapter_kernel
active={};calls={};captured={};original_solar=solar.solar_geometry_device
def solar_capture(*args,**kwargs):
 result=original_solar(*args,**kwargs)
 if active.get('selected'):captured['coszen_device']=result[1]
 return result
def sync():cp.cuda.get_current_stream().synchronize()
def host(value):return cp.asnumpy(value) if isinstance(value,cp.ndarray) else np.asarray(value)
def digest(value):return hashlib.sha256(np.ascontiguousarray(host(value)).tobytes()).hexdigest()
def compare(got,want):
 x,y=host(got),host(want);assert x.shape==y.shape and x.dtype==y.dtype
 count=int(np.count_nonzero(np.any(x.view(np.uint8).reshape(-1,x.dtype.itemsize)!=y.view(np.uint8).reshape(-1,y.dtype.itemsize),axis=1)))
 result={'shape':list(x.shape),'words':int(x.size),'different_words':count};assert count==0,result;return result
def provider(julday,julian,xlat,climo=None,**kwargs):
 result=original_provider(julday,julian,xlat,climo,**kwargs)
 if active.get('selected'):captured.update(julday=julday,julian=julian,xlat=xlat,climo=climo,reference=result.copy())
 return result

def component(kernel,grid,block,args,launch_kwargs):
 adapter=active['adapter'];kwargs=active['kwargs'];elapsed=float(kwargs['state'].elapsed_seconds)
 assert len(args)==11 and captured and int(args[0])==120*70
 reference=captured['reference'];entries=ozone._native_ozone_entries();assert entries is not None,'existing native ozone ABI unavailable'
 values=ozone._native_xlat(entries,np.asarray(captured['xlat'],np.float32))
 inputs=ozone._native_climatology(captured['climo']);assert inputs is not None,'native climatology admission failed'
 latitude,climo=inputs;workers=portable_math._workers(None);assert workers>1
 immutable={'values':values,'latitude':latitude,'climatology':climo,'pressure':args[3],'levels':args[6]}
 before={k:digest(v) for k,v in immutable.items()}
 original_grid=host(args[8]).copy();output=cp.empty_like(args[8]);ordinary=np.empty_like(reference)
 def arguments(julian,out):
  nm,np_,f1,f2=ozone._ozone_time_weights(julian)
  return Args(C.cast(entries[2],C.c_void_p),values.ctypes.data,values.size,latitude.ctypes.data,latitude.size,climo.ctypes.data,ozone.LEVSIZ,12,nm,np_,f1,f2,workers,out)
 plan=Plan()
 try:
  with plan.record(args,kernel,output,ordinary,values,latitude,climo,entries):
   producer=plan.native_host(ROOT/'receipts/libradiation-host.so','host_ozone',arguments(captured['julian'],ordinary.ctypes.data),ordinary,values,latitude,climo,entries)
   with plan.native_upload() as uploads:device=cp.asarray(ordinary)
   replaced=list(args);replaced[7]=device;replaced[8]=output
   kernel(grid,block,tuple(replaced),**launch_kwargs)
  sync();plan.resources.append(device)
  receipt['cases'].append({'case':'capture at second natural root radiation','elapsed_seconds':elapsed,'host_ozone':compare(ordinary,reference),'device_ozone':compare(device,reference),'pressure_grid':compare(output,original_grid)})
  address,size=plan.host_buffer(uploads[0]);assert size==reference.nbytes
  plan.bind(producer,0,Frame,'args')
  ordinary.fill(np.nan)
  frame=Frame(arguments(captured['julian'],address));plan.submit(frame);sync()
  receipt['cases'].append({'case':'replay current calendar from owned host buffer','host_ozone':compare(np.ctypeslib.as_array((C.c_float*reference.size).from_address(address)).reshape(reference.shape),reference),'device_ozone':compare(device,reference),'pressure_grid':compare(output,original_grid),'original_capture_buffer_ignored':bool(np.isnan(ordinary).all())})
  assert np.isnan(ordinary).all()
  # Only the time scalars change. Pressure/latitude remain the actual sampled
  # second-radiation inputs, so this is not a subsequent forecast-state proof.
  cadence=calls[id(adapter)][1]-calls[id(adapter)][0];assert cadence>0
  valid=adapter.start_time+timedelta(seconds=elapsed+cadence)
  day=int(valid.timetuple().tm_yday);hour=valid.hour+valid.minute/60.+valid.second/3600.+valid.microsecond/3.6e9
  julian=np.float32((day-1)+hour/24.)
  next_reference=original_provider(day,julian,captured['xlat'],captured['climo'])
  changed=int(np.count_nonzero(next_reference.view(np.uint32)!=reference.view(np.uint32)));assert changed>0,'calendar negative control did not change ozone'
  reference_device=cp.asarray(next_reference);reference_grid=cp.empty_like(output)
  refargs=list(args);refargs[7]=reference_device;refargs[8]=reference_grid
  kernel(grid,block,tuple(refargs),**launch_kwargs);sync()
  frame=Frame(arguments(julian,address));plan.submit(frame);sync()
  receipt['cases'].append({'case':'replay next cadence calendar with fixed sampled pressure','elapsed_seconds_for_weights':elapsed+cadence,'julian':float(julian),'calendar_changed_words':changed,'device_ozone':compare(device,next_reference),'pressure_grid':compare(output,reference_grid)})
  receipt['plan']=plan.describe();receipt.update(workers=workers,upload_bytes=size,native_submissions=2,original_kernel_name='rla_ozn_p_int',input_hashes=before,inputs_unchanged={k:digest(v) for k,v in immutable.items()}==before,ordinary_grid_unchanged=digest(args[8])==digest(original_grid))
  assert receipt['inputs_unchanged'] and receipt['ordinary_grid_unchanged']
 finally:plan.close()
 daylight_component(captured['coszen_device'],workers)
 raise Complete()

def daylight_component(original,workers):
 # Original comparisons at rrtmg_legacy.py:2303-2304. Shared compiled stable
 # partition, with actual AC solar field and a sign-inverted live-input control.
 before=digest(original);base=host(original).copy();n=base.size
 device=original.copy();day=np.empty(n,np.int64);night=np.empty(n,np.int64)
 counts=np.zeros(2,np.uint64);work=np.empty(2*workers,np.uintp)
 branches=cp.zeros(2,cp.int64)
 kernel=cp.RawKernel('extern "C" __global__ void native_day_branch(long long* b, int i){b[i]+=1;}','native_day_branch')
 kernel((1,),(1,),(branches,np.int32(0)));sync();branches.fill(0);sync()
 plan=Plan()
 def result(expected):
  wantday=np.nonzero(expected>np.float32(0))[0];wantnight=np.nonzero(expected<=np.float32(0))[0]
  assert tuple(counts)==(wantday.size,wantnight.size)
  return {'day_count':int(counts[0]),'night_count':int(counts[1]),'day_indices':compare(day[:counts[0]],wantday),'night_indices':compare(night[:counts[1]],wantnight)}
 try:
  with plan.record(device,day,night,counts,work,branches,kernel):
   with plan.readback() as reads:observed=cp.asnumpy(device)
   address,size=plan.host_buffer(reads[0]);assert size==base.nbytes
   args=Daylight(address,n,n,workers,day.ctypes.data,night.ctypes.data,counts.ctypes.data,work.ctypes.data)
   plan.native_host(ROOT/'receipts/libradiation-host.so','host_daylight',args,day,night,counts,work)
   plan.count_predicate(C.c_uint64.from_buffer(counts,0),greater_than=0,bit=0)
   plan.count_predicate(C.c_uint64.from_buffer(counts,8),greater_than=0,bit=1)
   with plan.when(require=1):kernel((1,),(1,),(branches,np.int32(0)))
   with plan.when(require=2):kernel((1,),(1,),(branches,np.int32(1)))
  sync();receipt['cases'].append(dict(case='daylight capture actual solar field',**result(base)))
  observed.fill(np.nan)
  for label,expected in [('actual',base),('sign inverted control',np.negative(base))]:
   if label!='actual':cp.negative(original,out=device)
   branches.fill(0);plan.submit();sync()
   wanted=np.asarray([int(np.any(expected>0)),int(np.any(expected<=0))],np.int64)
   receipt['cases'].append(dict(case='daylight replay '+label,**result(expected),selected_groups=compare(branches,wanted)))
  receipt['daylight_plan']=plan.describe();receipt['daylight_readback_bytes']=size
  receipt['daylight_original_unchanged']=digest(original)==before;assert receipt['daylight_original_unchanged']
  receipt['daylight_capture_host_array_ignored']=bool(np.isnan(observed).all());assert receipt['daylight_capture_host_array_ignored']
 finally:plan.close()

def get_kernel(name):
 kernel=original_kernel(name)
 if name!='rla_ozn_p_int':return kernel
 def wrapped(grid,block,args,**kwargs):
  result=kernel(grid,block,args,**kwargs)
  if active.get('selected'):sync();component(kernel,grid,block,args,kwargs)
  return result
 return wrapped

def radiation(adapter,**kwargs):
 times=calls.setdefault(id(adapter),[]);times.append(float(kwargs['state'].elapsed_seconds))
 selected=len(times)==2 and adapter._ozone_provider is None and adapter.o3input!=0
 active.update(adapter=adapter,kwargs=kwargs,selected=selected)
 try:return baseline(adapter,**kwargs)
 finally:active.clear()
ozone.ozn_latitude_time_int=provider;module._adapter_kernel=get_kernel;klass.__call__=radiation;solar.solar_geometry_device=solar_capture
command=['--prepared-root',binding['prepared_root'],binding['proof_option'],binding['prepared_head_sha256'],'--experiment-config',binding['config'],'--experiment-config-sha256',binding['config_sha256'],'--progress-format','jsonl','--render-products','none','--io-mode','none','--outdir',str(ROOT/('native-ozone-'+str(time.time_ns())))]
receipt['argv']=command
try:
 run(command);raise RuntimeError('second natural root ozone component not reached')
except Complete:receipt['status']='EXACT_NATIVE_OZONE_COMPONENT'
except BaseException as error:receipt.update(status='FAILED',error=repr(error));raise
finally:
 ozone.ozn_latitude_time_int=original_provider;module._adapter_kernel=original_kernel;klass.__call__=baseline;solar.solar_geometry_device=original_solar
 (ROOT/f'receipts/native-ozone-{a.numerics}.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({'status':receipt.get('status'),'cases':len(receipt['cases']),'error':receipt.get('error')}),flush=True)
