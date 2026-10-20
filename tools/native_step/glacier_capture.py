"""Recorded glacier transport around the unchanged NOAHMP_GLACIER kernel.

Only host/device packing changes. Original balance gates remain native guards.
The source-pinned runtime write-back and its physical solver are unchanged.
"""
import ast,copy,hashlib
from pathlib import Path
from contextlib import contextmanager
import numpy as np
SOURCE_SHA256='56f4f21f38583b9242fca8bad309260dda34bae98aabd698451eae663c3cd679'

def transformed(source):
 if hashlib.sha256(source).hexdigest()!=SOURCE_SHA256:raise RuntimeError('changed glacier runtime source')
 node=copy.deepcopy(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_glacier_lsm_step'))
 seen=set()
 def expr(s):return ast.parse(s,mode='eval').body
 class Rewrite(ast.NodeTransformer):
  def visit_Assign(self,n):
   replacements={297:'_recorded.mask',300:'_recorded.indices',309:'True',310:'_recorded.indices',343:'_recorded.divide(qv, one + qv)',347:'_recorded.divide(rainbl, np.float32(dt32))',383:'_recorded.evaluate(inputs, count, float(dt32))'}
   if n.lineno in replacements:
    seen.add(n.lineno);n.value=expr(replacements[n.lineno]);return n
   return self.generic_visit(n)
  def visit_Call(self,n):
   name=ast.unparse(n.func)
   if name=='cp.asnumpy':
    seen.add(n.lineno);assert len(n.args)==1 and not n.keywords;return self.visit(n.args[0])
   if name in ('np.minimum','np.arange','np.zeros','np.concatenate','np.ascontiguousarray'):
    seen.add(n.lineno);n.func=expr('cp.'+n.func.attr)
   if name=='np.divide':
    seen.add(n.lineno);n.func=expr('_recorded.divide')
   return self.generic_visit(n)
 node=Rewrite().visit(node)
 expected={297,300,309,310,314,315,319,320,329,331,343,347,348,357,359,361,378,383,392}
 assert seen==expected,('glacier transport sites changed',seen^expected)
 return ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))

class RecordedGlacier:
 def __init__(self,plan,physics,receipt):
  import cupy as cp
  from gpuwm.core import noahmp_runtime as runtime,noahmp_glacier_gpu as gpu
  self.plan,self.runtime,self.gpu=plan,runtime,gpu
  cached=physics.noahmp_params.__dict__.get('_surface_layout_cache')
  assert cached is not None,'glacier needs the original checked surface layout'
  self.mask=cached[2]['glacier'];self.indices=tuple(cp.ascontiguousarray(x) for x in cp.nonzero(self.mask))
  self.count=int(self.indices[0].size)
  assert self.count==cached[2]['census']['glacier']
  plan.resources.extend((self.mask,*self.indices))
  self.quotient=cp.ElementwiseKernel('float32 x,float32 y','float32 z','z=__fdiv_rn(x,y);','native_glacier_host_equivalent_divide',options=('--fmad=false',))
  namespace=dict(vars(runtime),_recorded=self)
  exec(compile(transformed(Path(runtime.__file__).read_bytes()),runtime.__file__,'exec'),namespace)
  self.method=namespace['_glacier_lsm_step']
  receipt.append(dict(kind='device_glacier_transport',columns=self.count,source_sha256=SOURCE_SHA256,solver='original noahmp_glacier_column',balance_guards=True))
 def divide(self,x,y,*,out=None,where=None):
  import cupy as cp
  value=self.quotient(x,y)
  if out is None:return value
  cp.copyto(out,value,where=where);return out
 def evaluate(self,inputs,count,dt):
  import cupy as cp
  g=self.gpu;assert count==self.count
  packed=cp.empty((count,g.N_INPUT),dtype=cp.float32);base=0
  for name in g.GLACIER_INPUT_SCALARS:
   value=inputs[name];assert value.shape==(count,);packed[:,base]=value;base+=1
  for name,width in g.GLACIER_INPUT_ARRAYS:
   value=inputs[name];assert value.shape==(count,width);packed[:,base:base+width]=value;base+=width
  assert base==g.N_INPUT
  isnow=cp.ascontiguousarray(inputs['isnow'],dtype=cp.int32)
  with self.plan.constant_uploads(private_events=True):zsoil=cp.asarray(np.ascontiguousarray(inputs['zsoil'],dtype=np.float32))
  device_out=cp.empty((count,g.N_OUTPUT),dtype=cp.float32);device_io=cp.empty((count,2),dtype=cp.int32)
  kernel=g.glacier_module(noahmp_snow_age=inputs.get('noahmp_snow_age','wrf_470')).get_function('noahmp_glacier_column')
  kernel(((count+g.THREADS-1)//g.THREADS,),(g.THREADS,),(packed,isnow,zsoil,np.float32(dt),device_out,device_io,np.int32(count)))
  error_values=cp.ascontiguousarray(device_io[:,1])
  def error(index,value):return g.GlacierBalanceError(f'NOAHMP_GLACIER native balance gate: column {index}, {g._ERROR_NAMES.get(value,value)}')
  with self.plan.zero_guard(error,word_bytes=4):errors=cp.asnumpy(error_values)
  g._raise_on_errors(errors)
  result={};base=0
  for name in g.GLACIER_OUTPUT_SCALARS:result[name]=device_out[:,base].copy();base+=1
  for name,width in g.GLACIER_OUTPUT_ARRAYS:result[name]=device_out[:,base:base+width].copy();base+=width
  assert base==g.N_OUTPUT;result['isnow']=device_io[:,0].copy()
  return result
 @contextmanager
 def install(self):
  original=self.runtime._glacier_lsm_step
  self.runtime._glacier_lsm_step=self.method
  try:yield
  finally:self.runtime._glacier_lsm_step=original
