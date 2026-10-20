"""CPU semantic-boundary ABI proof; exact original MU caller with fake device arrays."""
import ast,copy,ctypes as C,hashlib,json,sys,types
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from boundary_slots import ExternalBoundarySlots,SOURCE_SHA256,SITES,SIDES,PARTS
ROOT=Path(__file__).resolve().parents[2];path=ROOT/'gpuwm/ingest/lateral_bc.py';raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA256
tree=ast.parse(raw);methods={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
checks=[]
for kernel,(function,start,key) in SITES.items():
 method=methods[function]
 calls=[n for n in ast.walk(method) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='kernel']
 assert len(calls)==1
 arguments=calls[0].args[2]
 if isinstance(arguments,ast.BinOp):arguments=arguments.left
 assert isinstance(arguments,ast.Tuple)
 stars=[i for i,x in enumerate(arguments.elts) if isinstance(x,ast.Starred)]
 assert stars==[start] and ast.unparse(arguments.elts[start].value)=='sides'
 loops=[n for n in method.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='side']
 assert len(loops)==1 and ast.unparse(loops[0].iter)=='(boundary.west, boundary.east, boundary.south, boundary.north)'
 assert ast.unparse(loops[0].body[0])=='sides.extend((side.value, side.tendency))'
 checks.append('original '+kernel+' eight pointer positions')

def array(ptr,shape=(1,4,2)):
 return NS(shape=shape,dtype='float32',strides=(32,8,4),data=NS(ptr=ptr),flags=NS(c_contiguous=True))
intervals=[]
for index in range(2):
 fields={}
 for number,name in enumerate(('mu','theta')):
  fields[name]=NS(**{side:NS(value=array(10000+index*1000+number*100+i*8),tendency=array(10004+index*1000+number*100+i*8),time_law=None) for i,side in enumerate(SIDES)})
 intervals.append(NS(fields=fields))
hosts=[NS(start_seconds=0.),NS(start_seconds=100.)];resident=NS(intervals=tuple(intervals),rolling=False,streaming_external=False,host_interval_indices={id(h):i for i,h in enumerate(hosts)},clock=None)
state=NS(_lateral_boundary_device=resident,elapsed_seconds=0.,mup=array(999,(4,6)),scratch=lambda shape,key:array(888,shape),lateral_boundaries=NS(interval_at=lambda elapsed:hosts[0 if elapsed<100 else 1]))
ops=[];bindings=[]
plan=NS(resources=[],handle=1,library=NS(awstep_count=lambda handle:len(ops)),describe=lambda:{'operations':ops},bind=lambda *args:bindings.append(args))
module=types.ModuleType('original_boundary_cpu');module.__file__=str(path);module._perimeter_count=lambda *args:20;module._THREADS=256

def get_kernel(group,name):
 def launch(grid,block,args):
  ops.append({'kernel':name,'argument_bytes':[8 if hasattr(x,'data') else 4 for x in args]})
 return launch
module.get_kernel=get_kernel
selected=[methods['_served_external_interval'],methods['_launch_mu_boundary_values']]
exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(path),'exec'),vars(module))
module.np=np
slot=ExternalBoundarySlots(plan,state,module)
old_cupy=sys.modules.get('cupy');sys.modules['cupy']=NS(float32=np.float32)
try:
 with slot.capture():module._launch_mu_boundary_values(state,intervals[0].fields['mu'],np.float32(1),1)
finally:
 if old_cupy is None:sys.modules.pop('cupy')
 else:sys.modules['cupy']=old_cupy
assert len(ops)==1 and len(slot.bindings)==8 and module.get_kernel is get_kernel
Frame=type('Frame',(C.Structure,),{'_fields_':slot.fields});slot.bind(Frame);assert len(bindings)==8
frame=Frame();assert slot.fill(frame,state)==0;first=[getattr(frame,n) for n,_ in slot.fields]
state.elapsed_seconds=100.;assert slot.fill(frame,state)==1;second=[getattr(frame,n) for n,_ in slot.fields]
assert all(b-a==1000 for a,b in zip(first,second))
checks.extend(['exact original MU caller captured eight semantic bindings','all sixteen frame pointers select next served interval','capture get_kernel restored'])
for name in ('rolling','streaming_external'):
 setattr(resident,name,True)
 try:slot.fill(frame,state)
 except RuntimeError:checks.append(name+' ownership refusal')
 else:raise AssertionError('changed ownership accepted')
 setattr(resident,name,False)
for mutation,error in [('dtype','contiguous float32'),('shape','layouts changed'),('time_law','nonlinear boundary')]:
 bad=copy.deepcopy(state);side=bad._lateral_boundary_device.intervals[1].fields['mu'].west
 if mutation=='dtype':side.value.dtype='float64'
 elif mutation=='shape':side.value.shape=(2,4,2)
 else:side.time_law=object()
 try:ExternalBoundarySlots(plan,bad,module)
 except RuntimeError as actual:assert error in str(actual);checks.append(mutation+' admission refusal')
 else:raise AssertionError('bad interval accepted')
receipt=dict(status='PASS',scope=__doc__,checks=checks,gpu_calls=0,original_source_sha256=SOURCE_SHA256,source={name:hashlib.sha256((ROOT/'tools/native_step'/name).read_bytes()).hexdigest() for name in ('boundary_slots.py','check_boundary_slots_cpu.py')},actual_gpu_binding='UNRUN',whole_forecast='UNRUN')
(ROOT/'receipts/boundary-slots-cpu.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
