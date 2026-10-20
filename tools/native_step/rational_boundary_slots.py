"""Live coefficient pointers for original nonlinear WRF boundary evaluation."""
import ctypes as C
import hashlib,sys
from pathlib import Path
from contextlib import contextmanager
from boundary_slots import ExternalBoundarySlots,SOURCE_SHA256,SIDES,layout,pointer
class RationalBoundarySlots(ExternalBoundarySlots):
 def __init__(self,plan,state,module,*,prefix='boundary_'):
  if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=SOURCE_SHA256:raise RuntimeError('changed original lateral boundary source')
  self.plan,self.state,self.module,self.prefix=plan,state,module,prefix
  resident=state._lateral_boundary_device
  if resident.rolling or resident.streaming_external:raise RuntimeError('nonlinear slots require eager non-rolling storage')
  self.intervals=resident.intervals;self.names=tuple(self.intervals[0].fields)
  self.keys=tuple((name,side,part) for name in self.names for side in SIDES for part,_ in self.array_items(getattr(self.intervals[0].fields[name],side)))
  self.positions={key:i for i,key in enumerate(self.keys)};self.owners=[];self.schema=None
  for interval in self.intervals:
   assert tuple(interval.fields)==self.names
   keys=tuple((name,side,part) for name in self.names for side in SIDES for part,_ in self.array_items(getattr(interval.fields[name],side)))
   assert keys==self.keys,'nonlinear boundary coefficient inventory changed'
   arrays=tuple(array for name in self.names for side in SIDES for _,array in self.array_items(getattr(interval.fields[name],side)))
   assert all(str(x.dtype)=='float32' and x.flags.c_contiguous for x in arrays)
   schema=tuple(layout(x) for x in arrays)
   assert self.schema is None or schema==self.schema
   self.schema=schema;self.owners.append(arrays)
  self.packed=resident.evaluated_packed
  assert self.packed is not None,'nonlinear evaluated storage is missing'
  self.packed_binding=(pointer(self.packed),layout(self.packed))
  self.fields=[(prefix+str(i),C.c_void_p) for i in range(len(self.keys))]
  self.bindings=[];self.bound=False;plan.resources.extend([*self.owners,self.packed])
 @staticmethod
 def array_items(side):
  yield 'value',side.value;yield 'tendency',side.tendency
  if side.time_law is not None:
   yield 'quadratic',side.time_law.quadratic
   yield 'denominator_rate',side.time_law.denominator_rate
 @contextmanager
 def capture(self):
  old=self.module.get_kernel
  def get_kernel(module,name,*args,**kwargs):
   kernel=old(module,name,*args,**kwargs)
   if name not in ('evaluate_linear_boundary','evaluate_rational_boundary'):return kernel
   def launch(grid,block,arguments,*more,**options):
    caller=sys._getframe(1)
    assert caller.f_code.co_filename==self.module.__file__ and caller.f_code.co_name=='_evaluate_device_interval'
    assert caller.f_locals['state'] is self.state
    field,side=caller.f_locals['name'],caller.f_locals['side_name']
    _,interval=self.selected(self.state);row=getattr(interval.fields[field],side)
    assert caller.f_locals['side'] is row
    source=[]
    for argument,(part,array) in enumerate(self.array_items(row)):
     assert pointer(arguments[argument])==pointer(array)
     source.append((argument,self.positions[(field,side,part)]))
    before=self.plan.library.awstep_count(self.plan.handle)
    result=kernel(grid,block,arguments,*more,**options)
    assert self.plan.library.awstep_count(self.plan.handle)==before+1
    self.bindings.extend((before,arg,slot,name) for arg,slot in source)
    return result
   return launch
  self.module.get_kernel=get_kernel
  try:yield
  finally:self.module.get_kernel=old
 def fill(self,frame,state):
  packed=state._lateral_boundary_device.evaluated_packed
  assert (pointer(packed),layout(packed))==self.packed_binding
  return super().fill(frame,state)
