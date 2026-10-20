# CPU contract for nonlinear coefficient selection and original launch slots.
import ctypes as C,json,sys
from pathlib import Path
from types import SimpleNamespace as S
sys.path.insert(0,str(Path(__file__).parent))
from rational_boundary_slots import RationalBoundarySlots
ROOT=Path(__file__).resolve().parents[2]
class Array:
 def __init__(self,p):self.data=S(ptr=p);self.shape=(2,);self.dtype='float32';self.strides=(4,);self.flags=S(c_contiguous=True)
class Plan:
 def __init__(self):self.resources=[];self.handle=1;self.ops=[];self.bound=[];self.library=S(awstep_count=lambda _:len(self.ops))
 def bind(self,op,arg,frame,field):self.bound.append((op,arg,field))
 def describe(self):return {'operations':self.ops}
p=Plan();iv=[]
for j in range(2):
 sides={}
 for k,name in enumerate(('west','east','south','north')):
  start=1000+100*j+10*k
  sides[name]=S(value=Array(start),tendency=Array(start+1),time_law=(S(quadratic=Array(start+2),denominator_rate=Array(start+3)) if k%2 else None))
 iv.append(S(fields={'th':S(**sides)}))
resident=S(rolling=False,streaming_external=False,intervals=iv,evaluated_packed=Array(999),host_interval_indices={id(x):i for i,x in enumerate(iv)})
state=S(_lateral_boundary_device=resident,index=0)
module=S(__file__=str(ROOT/'gpuwm/ingest/lateral_bc.py'),_served_external_interval=lambda state,resident:(iv[state.index],0))
def get_kernel(group,name):
 def launch(grid,block,arguments):p.ops.append({'kernel':name,'argument_bytes':[C.sizeof(C.c_void_p)]*len(arguments)})
 return launch
module.get_kernel=get_kernel
slots=RationalBoundarySlots(p,state,module)
namespace={'module':module,'sides':('west','east','south','north')}
exec(compile('''def _evaluate_device_interval(state,interval):
 for name,boundary in interval.fields.items():
  for side_name in sides:
   side=getattr(boundary,side_name)
   args=(side.value,side.tendency)
   kernel_name='evaluate_linear_boundary'
   if side.time_law is not None:
    args+=(side.time_law.quadratic,side.time_law.denominator_rate)
    kernel_name='evaluate_rational_boundary'
   module.get_kernel('lbc_time',kernel_name)((1,),(256,),args)
''',module.__file__,'exec'),namespace)
with slots.capture():namespace['_evaluate_device_interval'](state,iv[0])
Frame=type('Frame',(C.Structure,),{'_fields_':slots.fields});slots.bind(Frame)
rows=[]
for state.index in (0,1,0):
 frame=Frame();index=slots.fill(frame,state)
 expected=[a.data.ptr for a in slots.owners[index]]
 assert [getattr(frame,f) for f,_ in slots.fields]==expected
 rows.append({'interval':index,'pointers':expected})
assert len(p.bound)==12 and len(p.ops)==4
resident.evaluated_packed=Array(888)
try:slots.fill(Frame(),state);raise RuntimeError('changed packed allocation accepted')
except AssertionError:pass
print(json.dumps({'status':'PASS','recorded_original_evaluation_kernels':4,'coefficient_slots':12,'interval_selections':rows,'changed_packed_storage_rejected':True}))
