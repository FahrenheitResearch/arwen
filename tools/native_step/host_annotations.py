"""Explicit callsite annotations for the pinned AC diagnostic, never dispatch.

This inventory is deliberately narrow. It does not infer contracts from host
values or approve changing layouts. Optional layout_guard supports pinned
diagnostic replay only, with a generic diagnostic exception on failure.
"""
from contextlib import contextmanager
from pathlib import Path
import sys

# Merged 2.9.3 caches latitude/soil-depth uploads in device_static_fields.
# The glacier drainage addition moves the unchanged surface_layout read by 2.
UPLOADS = {('noahmp_runtime.py', 1174): 'static_geometry',
           ('noahmp_runtime.py', 1998): 'urban_categories'}
ZERO_READS = {('noahmp_sflx_pre_slab.py', 219): 'crop_irrigation',
              ('noahmp_energy_slab.py', 100): 'ice_admission',
              ('noahmp_energy_slab.py', 107): 'land_admission',
              ('noahmp_energy_slab.py', 273): 'longwave',
              ('noahmp_sflx_post_slab.py', 483): 'surface_balance'}

@contextmanager
def annotate(plan, receipt, *, layout_guard=False):
    import cupy as cp
    from gpuwm.core import mynn_pbl_gpu, mynn_pbl_runtime
    old_runtime_mask = mynn_pbl_runtime._flag_mask
    old_array, old_numpy, old_mask = cp.asarray, cp.asnumpy, mynn_pbl_gpu._flag_mask
    def error(site):
        # Generic diagnostic exception, not the production model adapter.
        return lambda index,value: RuntimeError(f'audit status {site}: word {index}={value}')
    def asarray(value, *args, **kwargs):
        caller=sys._getframe(1)
        key=(Path(caller.f_code.co_filename).name, caller.f_lineno)
        if key in UPLOADS:
            with plan.constant_uploads(private_events=True):
                result=old_array(value,*args,**kwargs)
            receipt.append({'kind':'retained_upload','private_events':True,'site':UPLOADS[key],
                            'file':key[0],'line':key[1]})
            return result
        return old_array(value,*args,**kwargs)
    zero_reads = dict(ZERO_READS)
    if layout_guard:
        zero_reads[("noahmp_runtime.py",981)] = "surface_layout_unchanged"
    def asnumpy(value,*args,**kwargs):
        caller=sys._getframe(1)
        # The explicit whole-period consumer is a transparent wrapper here;
        # contracts still belong to the original, pinned model callsite.
        while (Path(caller.f_code.co_filename).name == "whole_health_annotations.py"
               and caller.f_code.co_name == "asnumpy"):
            caller=caller.f_back
        key=(Path(caller.f_code.co_filename).name, caller.f_lineno)
        if key in zero_reads:
            with plan.zero_guard(error(zero_reads[key]),word_bytes=value.dtype.itemsize):
                result=old_numpy(value,*args,**kwargs)
            receipt.append({'kind':'native_zero_guard','site':zero_reads[key],
                            'file':key[0],'line':key[1]})
            return result
        return old_numpy(value,*args,**kwargs)
    def flags(kernel, arrays, buffer):
        with plan.zero_guard(error('mynn_flags'),word_bytes=buffer.dtype.itemsize):
            result=old_mask(kernel,arrays,buffer)
        receipt.append({'kind':'native_zero_guard','site':'mynn_flags',
                        'file':'mynn_pbl_gpu.py','line':429})
        return result
    cp.asarray,cp.asnumpy,mynn_pbl_gpu._flag_mask=asarray,asnumpy,flags
    mynn_pbl_runtime._flag_mask=flags
    try:
        yield
    finally:
        cp.asarray,cp.asnumpy,mynn_pbl_gpu._flag_mask=old_array,old_numpy,old_mask
        mynn_pbl_runtime._flag_mask=old_runtime_mask


def inventory(state):
 import cupy as cp
 seen=set();arrays={}
 def visit(value,label):
  if id(value) in seen:return
  seen.add(id(value))
  if isinstance(value,cp.ndarray):
   key=(value.data.ptr,value.shape,value.strides,value.dtype.str)
   arrays.setdefault(key,{'label':label,'array':value});return
  if isinstance(value,dict):
   for name,item in value.items():visit(item,label+'.'+str(name))
  elif isinstance(value,(list,tuple)):
   for i,item in enumerate(value):visit(item,label+f'[{i}]')
  elif type(value).__module__.startswith('gpuwm.') and hasattr(value,'__dict__'):
   for name,item in vars(value).items():visit(item,label+'.'+name)
 visit(state,'state')
 return arrays

def host_inventory(state):
 import cupy as cp
 import numpy as np
 seen=set();values={}
 def visit(value,label):
  if isinstance(value,(bool,int,float,str,type(None))):
   values[label]=value;return
  if id(value) in seen:return
  seen.add(id(value))
  if isinstance(value,(cp.ndarray,np.ndarray)):return
  if isinstance(value,dict):
   for k,v in value.items():visit(v,label+'.'+str(k))
  elif isinstance(value,(list,tuple)):
   for i,v in enumerate(value):visit(v,label+f'[{i}]')
  elif type(value).__module__.startswith('gpuwm.') and hasattr(value,'__dict__'):
   for k,v in vars(value).items():visit(v,label+'.'+k)
 visit(state,'state');return values

def phase(state,cfg,kwargs):
    import numpy as np
    from gpuwm.core.physics import _physics_interval_steps, _radiation_step_due, _surface_pbl_step_due
    physics=state.physics
    epoch=float(getattr(state,'domain_start_offset',0.0) or 0.0)
    itimestep=int(np.floor((state.elapsed_seconds-epoch)/cfg.dt+0.5))+1
    assert all(getattr(physics,name,None) is None for name in ('radiation_due_override','surface_pbl_due_override','cumulus_due_override'))
    assert not cfg.cu_physics and physics.fire is None and getattr(state,'chem',None) is None
    return (_radiation_step_due(itimestep,_physics_interval_steps(physics.radt_minutes,cfg.dt),physics.radt_minutes),
            _surface_pbl_step_due(itimestep,physics.stepbl,cfg.bldt,None),tuple(sorted(kwargs.items())))

def finish_trajectory(states,receipt,complete):
    import cupy as cp
    import numpy as np
    import hashlib
    cp.cuda.get_current_stream().synchronize()
    receipt['final']=[]
    for key,state in states.items():
        rows=[]
        arrays=inventory(state)
        addresses={}
        for item in arrays.values():addresses.setdefault(item['array'].data.ptr,[]).append(item['label'])
        def pointer(value):
            value=int(value)
            if value==0:return None
            assert value in addresses,('unresolved device address',value)
            return sorted(addresses[value])
        for entry in arrays.values():
            x=cp.asnumpy(entry['array'])
            row={'name':entry['label'],'shape':list(x.shape),'dtype':x.dtype.str,'words':int(x.size),'sha256':hashlib.sha256(x.tobytes()).hexdigest()}
            if entry['label'] in ('state._scratch.integration_health_field_ptr','state._scratch.integration_health_aux_ptr'):
                row['pointer_targets']=[pointer(v) for v in x.view(np.uint64)]
            rows.append(row)
        host=host_inventory(state)
        for name,value in list(host.items()):
            if any(name.startswith('state.'+prefix+'[0]') for prefix in ('_rk_copy_launch','_rk_zero_launch','_glue_add_launch')) and name.endswith('[0]'):
                host[name]={'pointer_target':pointer(value)}
        receipt['final'].append({'shape':list(key),'arrays':rows,'host':host})
    raise complete()

from contextlib import ExitStack
from whole_health_annotations import annotate_health
from validator_annotations import annotate_descriptors

@contextmanager
def contexts(plan,objects,receipt,live):
 with ExitStack() as stack:
  stack.enter_context(plan.record(objects))
  stack.enter_context(annotate(plan,receipt.setdefault('annotations',[]),layout_guard=True))
  stack.enter_context(annotate_health(plan,receipt.setdefault('health_annotations',[]),step_label=lambda gid,n:live['step_numbers'].get(gid,n)))
  stack.enter_context(annotate_descriptors(plan,receipt['health_annotations']))
  yield
