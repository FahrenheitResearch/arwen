"""Explicit scalar slots for an AC non-radiation domain-step probe.

This is pinned source instrumentation, not a production admission adapter.
No scalar values are matched heuristically across unrelated kernel arguments.
"""
from contextlib import contextmanager
import ctypes as C
from pathlib import Path
import sys
import numpy as np

class Frame(C.Structure):
    _fields_=[('julian',C.c_double),('yearlen',C.c_int64),
              ('boundary_dtbc',C.c_float),('forcing_dtbc',C.c_float)]

def frame_for(state,cfg):
    from gpuwm.core.noahmp_runtime import year_length
    from gpuwm.ingest.lateral_bc import _served_external_interval
    geometry=state.physics.noahmp_geometry
    resident=state._lateral_boundary_device
    if cfg.nested:
        forcing=resident.clock.dtbc_launch_fp32
        boundary=forcing
    else:
        interval,forcing=_served_external_interval(state,resident)
        nonlinear=any(side.time_law is not None for field in interval.fields.values()
                      for side in (field.west,field.east,field.south,field.north))
        boundary=0.0 if nonlinear else forcing
    return Frame(np.float32(geometry.julian(state.elapsed_seconds)),
                 year_length(geometry.year(state.elapsed_seconds)),
                 np.float32(boundary),np.float32(forcing))

@contextmanager
def capture_slots(plan,slots):
    import cupy as cp
    original=cp.full
    def full(shape,value,*args,**kwargs):
        caller=sys._getframe(1)
        site=(Path(caller.f_code.co_filename).name,caller.f_lineno)
        # Merged hydro drainage, table seals and cached static geometry move
        # these unchanged cp.full calls by 23 lines from speed2-integrate.
        field={('noahmp_runtime.py',2262):'julian',
               ('noahmp_runtime.py',2267):'yearlen'}.get(site)
        if field is None:return original(shape,value,*args,**kwargs)
        start=plan.library.awstep_count(plan.handle)
        result=original(shape,value,*args,**kwargs)
        end=plan.library.awstep_count(plan.handle)
        dtype=np.int64 if field=='yearlen' else np.float64
        slots.append({'start':start,'end':end,'field':field,
                      'expected_bytes':np.asarray(value,dtype=dtype).tobytes().hex(),
                      'source_file':site[0],'source_line':site[1]})
        return result
    cp.full=full
    try:yield
    finally:cp.full=original

def bind_slots(plan,slots, *, frame_type=Frame, prefix="", start=0, end=None):
    operations=plan.describe()['operations'];bound=[]
    for slot in slots:
        group=operations[slot['start']:slot['end']]
        kernels=[o for o in group if o['kind']==1]
        assert len(group)==1 and len(kernels)==1,slot
        op=kernels[0]
        assert op['kernel'].startswith('cupy_copy__') and op['argument_bytes'][0]==8,op
        raw=C.create_string_buffer(8)
        count=plan.library.awstep_arg(plan.handle,op['index'],0,raw,8)
        assert count==8 and raw.raw.hex()==slot['expected_bytes'],(op,slot,raw.raw.hex())
        plan.bind(op['index'],0,frame_type,prefix+slot['field'])
        bound.append({'operation':op['index'],'argument':0,'field':slot['field'],'source':slot})
    assert {x['field'] for x in slots}=={'julian','yearlen'},slots
    # ABI positions are from the exact tuple construction in lateral_bc.py:
    # state relaxation:1197, mu install:2474, finalizer:2500; lbc_time:885.
    specs={'state_specified_relaxation':(28,'boundary_dtbc'),
           'specified_relaxation':(12,'boundary_dtbc'),
           'install_mu_boundary':(10,'boundary_dtbc'),
           'finalize_state_field':(21,'boundary_dtbc'),
           'evaluate_linear_boundary':(4,'forcing_dtbc'),
           'evaluate_rational_boundary':(6,'forcing_dtbc')}
    for op in operations[start:end]:
        if op.get('kernel') in specs:
            index,field=specs[op['kernel']]
            assert op['argument_bytes'][index]==4,op
            plan.bind(op['index'],index,frame_type,prefix+field)
            bound.append({'operation':op['index'],'argument':index,'field':field,'kernel':op['kernel']})
    return bound
