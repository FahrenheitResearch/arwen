"""Endpoint arrays and scalar metadata for independent whole-period probes."""
import hashlib,types
from collections.abc import Mapping
import cupy as cp
import numpy as np

def inventory(value):
 seen=set();arrays={}
 def visit(value,label):
  if id(value) in seen:return
  seen.add(id(value))
  if isinstance(value,cp.ndarray):
   key=(value.data.ptr,value.shape,value.strides,value.dtype.str)
   arrays.setdefault(key,{'label':label,'array':value});return
  if isinstance(value,Mapping):
   for name,item in value.items():visit(item,label+'.'+str(name))
  elif isinstance(value,(list,tuple)):
   for i,item in enumerate(value):visit(item,label+f'[{i}]')
  elif type(value).__module__.startswith('gpuwm.') and hasattr(value,'__dict__'):
   for name,item in vars(value).items():visit(item,label+'.'+name)
 visit(value,'root');return arrays

def host_inventory(value):
 seen=set();values={}
 def visit(value,label):
  if isinstance(value,np.generic):value=value.item()
  if isinstance(value,(bool,int,float,str,type(None))):values[label]=value;return
  if id(value) in seen:return
  seen.add(id(value))
  if isinstance(value,cp.ndarray):return
  if isinstance(value,np.ndarray):
   values[label]={'shape':list(value.shape),'dtype':value.dtype.str,'sha256':hashlib.sha256(value.tobytes()).hexdigest()};return
  if isinstance(value,Mapping):
   for name,item in value.items():visit(item,label+'.'+str(name))
  elif isinstance(value,(list,tuple)):
   for i,item in enumerate(value):visit(item,label+f'[{i}]')
  elif type(value).__module__.startswith('gpuwm.') and hasattr(value,'__dict__'):
   for name,item in vars(value).items():visit(item,label+'.'+name)
 visit(value,'root');return values

def endpoint(objects):
 cp.cuda.get_current_stream().synchronize();arrays=inventory(objects);addresses={}
 for row in arrays.values():addresses.setdefault(row['array'].data.ptr,[]).append(row['label'])
 def pointer(value):
  value=int(value)
  if value==0:return None
  assert value in addresses,('unresolved address',value)
  return sorted(addresses[value])
 rows=[]
 for row in arrays.values():
  x=cp.asnumpy(row['array']);name=row['label']
  out={'name':name,'shape':list(x.shape),'dtype':x.dtype.str,'words':int(x.size),'sha256':hashlib.sha256(x.tobytes()).hexdigest()}
  if name.endswith(('.integration_health_field_ptr','.integration_health_aux_ptr')):out['pointer_targets']=[pointer(v) for v in x.view(np.uint64)]
  rows.append(out)
 host=host_inventory(objects)
 interval_ids={}
 for gid,node in objects['model'].nodes_by_grid_id.items():
  resident=node.state._lateral_boundary_device
  if resident.rolling:
   assert not resident.host_interval_indices
   continue
  expected={id(interval):i for i,interval in enumerate(node.state.lateral_boundaries.intervals)}
  assert dict(resident.host_interval_indices)==expected
  interval_ids.update({address:f'd{gid}.interval[{index}]' for address,index in expected.items()})
 for name,value in list(host.items()):
  if '.host_interval_indices.' in name:
   prefix,address=name.rsplit('.',1);assert int(address) in interval_ids
   host[prefix+'.'+interval_ids[int(address)]]=host.pop(name)
 for name,value in list(host.items()):
  if any('.'+prefix+'[0]' in name for prefix in ('_rk_copy_launch','_rk_zero_launch','_glue_add_launch')) and name.endswith('[0]'):
   host[name]={'pointer_target':pointer(value)}
 return {'arrays':rows,'host':host}
