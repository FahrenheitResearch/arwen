"""Semantic live pointer slots for eager linear external boundary intervals.

Unarmed prototype. Streaming and nonlinear intervals need separate contracts.
"""
import ctypes as C
import hashlib
import sys
from contextlib import contextmanager
from pathlib import Path

# Re-pinned to lane/speed2-integrate e18bd09a9ca: lateral_bc.py differs from the audited 4ac8acea.. copy only by get_kernel grid-shape constants (5d86ac9f37b); slot layout unchanged.
SOURCE_SHA256='dbe7c363317d22ce456525956226014f3fadb8d90ac3b16da96cd4cdc4ea52c8'
SIDES=('west','east','south','north')
PARTS=('value','tendency')
SITES={
 'state_specified_relaxation':('_launch_state_relaxation',18,'field_name'),
 'install_mu_boundary':('_launch_mu_boundary_values',2,None),
 'finalize_state_field':('_launch_finalize_field',13,'name'),
}

def layout(array):
 return (tuple(array.shape),str(array.dtype),tuple(array.strides))

def pointer(array):
 return int(array.data.ptr)

class ExternalBoundarySlots:
 def __init__(self,plan,state,module,*,prefix='boundary_'):
  if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!=SOURCE_SHA256:
   raise RuntimeError('changed original lateral boundary source')
  self.plan,self.state,self.module,self.prefix=plan,state,module,prefix
  resident=state._lateral_boundary_device
  if resident.rolling or resident.streaming_external:
   raise RuntimeError('live external slots require eager non-rolling boundary storage')
  self.intervals=resident.intervals
  if not self.intervals:raise RuntimeError('no external intervals')
  self.names=tuple(self.intervals[0].fields)
  self.keys=tuple((name,side,part) for name in self.names for side in SIDES for part in PARTS)
  self.positions={key:i for i,key in enumerate(self.keys)}
  self.schema=None;self.owners=[]
  for interval in self.intervals:
   if tuple(interval.fields)!=self.names:raise RuntimeError('external field inventory changed')
   arrays=[]
   for name,side,part in self.keys:
    row=getattr(interval.fields[name],side)
    if row.time_law is not None:raise RuntimeError('nonlinear boundary needs live evaluation slots')
    array=getattr(row,part)
    if str(array.dtype)!='float32' or not array.flags.c_contiguous:raise RuntimeError('external arrays require contiguous float32')
    arrays.append(array)
   schema=tuple(layout(x) for x in arrays)
   if self.schema is not None and schema!=self.schema:raise RuntimeError('external interval array layouts changed')
   self.schema=schema;self.owners.append(tuple(arrays))
  self.fields=[(prefix+str(i),C.c_void_p) for i in range(len(self.keys))]
  self.bindings=[];self.bound=False
  plan.resources.extend(self.owners)
 def selected(self,state):
  resident=state._lateral_boundary_device
  if resident.intervals is not self.intervals or resident.rolling or resident.streaming_external:
   raise RuntimeError('external boundary ownership changed')
  host,_=self.module._served_external_interval(state,resident)
  index=resident.host_interval_indices[id(host)]
  return index,self.intervals[index]
 @contextmanager
 def capture(self):
  old=self.module.get_kernel
  def get_kernel(module,name,*args,**kwargs):
   kernel=old(module,name,*args,**kwargs)
   if name not in SITES:return kernel
   def launch(grid,block,arguments,*more,**options):
    caller=sys._getframe(1);function,start,key=SITES[name]
    if caller.f_code.co_filename!=self.module.__file__ or caller.f_code.co_name!=function:
     raise RuntimeError('external boundary kernel callsite changed: '+name)
    if caller.f_locals['state'] is not self.state:raise RuntimeError('external boundary state changed during capture')
    field='mu' if key is None else caller.f_locals[key]
    _,interval=self.selected(self.state)
    if caller.f_locals['boundary'] is not interval.fields[field]:raise RuntimeError('captured boundary is not the served external interval')
    source=[]
    for side in SIDES:
     for part in PARTS:
      slot=self.positions[(field,side,part)];array=getattr(getattr(interval.fields[field],side),part)
      argument=start+len(source)
      if pointer(arguments[argument])!=pointer(array):raise RuntimeError('original boundary argument identity differs')
      source.append((argument,slot))
    before=self.plan.library.awstep_count(self.plan.handle)
    result=kernel(grid,block,arguments,*more,**options)
    after=self.plan.library.awstep_count(self.plan.handle)
    if after!=before+1:raise RuntimeError('boundary launch operation inventory differs')
    self.bindings.extend((before,arg,slot,name) for arg,slot in source)
    return result
   return launch
  self.module.get_kernel=get_kernel
  try:yield
  finally:self.module.get_kernel=old
 def bind(self,frame_type):
  if not self.bindings or self.bound:raise RuntimeError('boundary slots require one completed capture')
  fields=dict(frame_type._fields_);ops=self.plan.describe()['operations']
  for operation,arg,slot,name in self.bindings:
   field=self.prefix+str(slot)
   if fields.get(field) is not C.c_void_p or ops[operation]['kernel']!=name or ops[operation]['argument_bytes'][arg]!=C.sizeof(C.c_void_p):
    raise RuntimeError('boundary slot ABI differs')
   self.plan.bind(operation,arg,frame_type,field)
  self.frame_type=frame_type;self.bound=True
 def fill(self,frame,state):
  if not self.bound or type(frame) is not self.frame_type:raise RuntimeError('boundary frame is not bound')
  index,_=self.selected(state)
  for i,array in enumerate(self.owners[index]):
   if layout(array)!=self.schema[i]:raise RuntimeError('held external boundary layout changed')
   setattr(frame,self.prefix+str(i),pointer(array))
  return index
