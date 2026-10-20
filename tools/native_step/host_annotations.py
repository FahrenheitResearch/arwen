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
