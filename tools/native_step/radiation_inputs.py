"""Explicit live atmospheric input staging for a recorded radiation call."""
import ctypes as C
import numpy as np
from radiation_capture import RadiationFrame
class AtmosphericInputs:
 def __init__(self,plan,atmosphere,cp):
  self.plan,self.cp=plan,cp;self.values=dict(atmosphere);self.original=dict(atmosphere);self.rows=[]
  for key,value in sorted(atmosphere.items()):
   if isinstance(value,cp.ndarray):
    if not value.flags.c_contiguous:raise ValueError('radiation input must be C-contiguous: '+key)
    held=cp.empty_like(value);self.values[key]=held
    self.rows.append((key,held,value.shape,value.strides,value.dtype.str))
  self.frame_type=type('LiveRadiationFrame',(C.Structure,),{'_fields_':RadiationFrame._fields_+[(f'input_{i}',C.c_void_p) for i in range(len(self.rows))]})
  self.kernel=cp.RawKernel('extern "C" __global__ void awrad_stage(const unsigned char* src,unsigned char* dst,unsigned long long n){unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dst[i]=src[i];}','awrad_stage')
  self.kernel.compile();self.operations=[]
 def record(self):
  for key,held,*_ in self.rows:
   before=self.plan.library.awstep_count(self.plan.handle)
   self.kernel(((held.nbytes+255)//256,),(256,),(self.original[key],held,np.uint64(held.nbytes)))
   after=self.plan.library.awstep_count(self.plan.handle)
   assert after==before+1
   self.operations.append(before)
 def bind(self):
  operations=self.plan.describe()['operations']
  for i,index in enumerate(self.operations):
   assert operations[index]['kernel']=='awrad_stage' and operations[index]['argument_bytes']==[8,8,8]
   self.plan.bind(index,0,self.frame_type,f'input_{i}')
 def fill(self,frame,atmosphere):
  if atmosphere.keys()!=self.original.keys():raise ValueError('radiation atmosphere keys changed')
  names={r[0] for r in self.rows}
  for key in atmosphere.keys()-names:
   if atmosphere[key]!=self.original[key]:raise ValueError('unstaged atmosphere scalar changed: '+key)
  changed=0
  for i,(key,held,shape,strides,dtype) in enumerate(self.rows):
   value=atmosphere[key]
   if not isinstance(value,self.cp.ndarray) or (value.shape,value.strides,value.dtype.str)!=(shape,strides,dtype):raise ValueError('radiation atmosphere layout changed: '+key)
   setattr(frame,f'input_{i}',value.data.ptr)
   changed+=int(value.data.ptr!=self.original[key].data.ptr)
  return changed

import copy,dataclasses,hashlib
import cupy as cp
from gpuwm.core.physics_inventory import RADIATION_BOUNDARY_FLUXES

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

def isolated_kwargs(kwargs):
 result=dict(kwargs);result['state']=copy.copy(kwargs['state']);result['fields']=dict(kwargs['fields'])
 for key,value in kwargs['fields'].items():
  if key in RADIATION_BOUNDARY_FLUXES:result['fields'][key]=value.copy()
 return result

def sample(result,obj):
 values=bits(result);values['retained_ozone']=cp.asnumpy(obj.o33d_device())
 return values
