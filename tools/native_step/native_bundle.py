"""Single-FFI native composition of fixed recorded phases; terminal after failure."""
import ctypes as C
from pathlib import Path
class Stage(C.Structure):
 _fields_=[('plan',C.c_void_p),('frame',C.c_void_p),('bytes',C.c_size_t),('flags',C.c_uint64)]
class Bundle:
 def __init__(self,library,plans):
  if not plans or len(plans)>32:raise ValueError('bundle needs 1..32 phases')
  self.plans=tuple(plans);self.library=C.CDLL(str(Path(library).resolve()));self.failed=False;self.completed=0
  self.execute=C.cast(plans[0].library.awstep_execute,C.c_void_p)
  if any(C.cast(p.library.awstep_execute,C.c_void_p).value!=self.execute.value for p in plans):raise ValueError('bundle recorder libraries differ')
  self.call=self.library.awstep_bundle;self.call.argtypes=[C.c_void_p,C.POINTER(Stage),C.c_size_t,C.POINTER(C.c_size_t)];self.call.restype=C.c_int
 def submit(self,frames,flags,*,failure_stage=None):
  if self.failed:raise RuntimeError('bundle is terminal after a partial submission failure')
  if len(frames)!=len(self.plans) or len(flags)!=len(self.plans):raise ValueError('bundle frame count differs')
  stages=(Stage*len(self.plans))()
  for i,(plan,frame,flag) in enumerate(zip(self.plans,frames,flags)):
   plan._same_thread()
   if not plan._ready:raise RuntimeError('bundle phase recording did not complete')
   if not isinstance(flag,int) or not 0<=flag<2**64:raise ValueError('invalid phase predicate flags')
   if frame is not None and not isinstance(frame,C.Structure):raise TypeError('bundle frames require ctypes Structures')
   stages[i]=Stage(plan.handle,C.addressof(frame) if frame is not None else None,C.sizeof(frame) if frame is not None else 0,flag)
  completed=C.c_size_t()
  rc=self.call(self.execute,stages,len(stages),C.byref(completed))
  self.completed=completed.value
  if rc:
   self.failed=True
   if self.completed>=len(self.plans):raise RuntimeError('invalid failing bundle phase')
   plan=self.plans[self.completed];plan._ready=False
   if failure_stage is not None:failure_stage(self.completed)
   plan._check(rc)
  assert self.completed==len(self.plans)
