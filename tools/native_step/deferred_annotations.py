"""Pinned radiation deferred verdicts, evaluated by one native readback consumer.

Diagnostic only. No solar/day compaction or production dispatcher admission.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib,sys

PINS = {'deferred_device_checks.py': '8271be8825ce6045a55a2e1946bcf8b2aa1ee3d912311c687c05937526049f57', 'rrtmg_lw.py': 'e899ce92bb2e09ed8d70a5368795fed21eb1bbd0c4faf8d3cd0195d3182bc6cd', 'rrtmg_sw.py': '4521cad52318b43c2fac6813c0a685bc092918084660ae5a348aa75530862997', 'rrtmg_mcica.py': '3947f739003363e0902d1c058e80ca0111545c4e0b2acc0548891041fa0be7de', 'rrtmg_legacy_device.py': 'b46e1d54451e45e3d880f34344f6e048b239473e149f053320eec79c25b5e869', 'rrtmg_legacy.py': '6270ca552b88fb6bdfd7e31018b1de2cad311db30245f3075160be979f90a240'}
# filename, nested function name, first source line -> native numeric rule.
SITES = {
 ('rrtmg_lw.py','_cldprmc_verdict',4938):1,
 ('rrtmg_sw.py','_cldprmc_verdict',3873):1,
 ('rrtmg_sw.py','_day_verdict',3575):1,
 ('rrtmg_mcica.py','_seed_verdict',640):1,
 ('rrtmg_legacy_device.py','_night_verdict',184):1,
 ('rrtmg_legacy.py','verdict',1739):2,
}

def validate_sources(root):
 for name,digest in PINS.items():
  path=root/'gpuwm/core'/name
  assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,('deferred verdict source changed',name)

@contextmanager
def annotate_deferred(plan,receipt):
 import cupy as cp
 import numpy as np
 from gpuwm.core import deferred_device_checks as checks
 validate_sources(Path(checks.__file__).resolve().parents[2])
 original=cp.asnumpy;original_concat=cp.concatenate
 def asnumpy(array,*args,**kwargs):
  caller=sys._getframe(1)
  if (Path(caller.f_code.co_filename).name,caller.f_lineno)!=('deferred_device_checks.py',61):
   return original(array,*args,**kwargs)
  sink=tuple(caller.f_locals['sink']);rules=[];groups=[]
  assert array.dtype==cp.float64 and array.ndim==1
  for value,verdict in sink:
   code=verdict.__code__;site=(Path(code.co_filename).name,code.co_name,code.co_firstlineno)
   assert site in SITES,('unadmitted deferred verdict',site)
   rule=SITES[site]
   if rule==1:assert value.dtype in (cp.dtype('bool'),cp.dtype('int32'))
   else:assert value.dtype==cp.float32 and value.size==1
   groups.append({'site':site,'dtype':value.dtype.str,'shape':list(value.shape),'offset':len(rules),'words':int(value.size),'rule':rule})
   rules.extend([rule]*int(value.size))
  assert len(rules)==array.size
  def error(index,bits):
   # On failure alone, use the original ordered verdicts and conversions.
   # No result is published and no downstream plan operation has run.
   host=original(array)
   assert int(host.view(np.uint64)[index])==bits,'guard readback changed before error decoding'
   offset=0
   for value,verdict in sink:
    size=int(value.size);part=host[offset:offset+size].astype(value.dtype).reshape(value.shape);offset+=size
    try:verdict(np.asarray(part))
    except Exception as actual:return actual
   return RuntimeError('native deferred guard failed but original verdicts passed')
  with plan.numeric_guard(error,rules=rules,threshold=1.0e-3):
   result=original(array,*args,**kwargs)
  receipt.append({'groups':groups,'words':len(rules),'threshold':1.0e-3})
  return result
 def concatenate(arrays,*args,**kwargs):
  caller=sys._getframe(1)
  assert (Path(caller.f_code.co_filename).name,caller.f_lineno)==('deferred_device_checks.py',59)
  assert all(isinstance(value,cp.ndarray) and value.dtype==cp.float64 and value.ndim==1 for value in arrays)
  # CuPy uploads its descriptor table here. It contains fixed captured
  # pointers and extents, while every value is still produced on device.
  with plan.constant_uploads(private_events=True,allow_empty=True):
   result=original_concat(arrays,*args,**kwargs)
  receipt.append({'kind':'retained_concatenate_descriptors','arrays':len(arrays),'words':sum(int(value.size) for value in arrays)})
  return result
 original_deferred=checks.deferred
 @contextmanager
 def deferred():
  scope=original_deferred();scope.__enter__()
  try:yield
  except BaseException:
   if not scope.__exit__(*sys.exc_info()):raise
  else:
   # Intercept only the original scope's closing read. Other model wrappers
   # keep their real callsites during the body and all normal checks run.
   before,before_concat=cp.asnumpy,cp.concatenate;cp.asnumpy,cp.concatenate=asnumpy,concatenate
   try:scope.__exit__(None,None,None)
   finally:cp.asnumpy,cp.concatenate=before,before_concat
 checks.deferred=deferred
 try:yield
 finally:checks.deferred=original_deferred
