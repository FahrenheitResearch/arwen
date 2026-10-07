"""WRF v4.7.1 SFIRE fuel-moisture time-lag model on the device."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math

import numpy as np

from gpuwm.core.sfire_phys import FuelTable, MODULE_OPTIONS, _array

MODULE_KEY = "gpuwm.core.sfire_moisture:sfire_moisture"
STATE_FIELDS = ("rain_old", "t2_old", "q2_old", "psfc_old", "fmc_gc", "fmep",
                "fmc_equi", "fmc_lag", "rh_fire")


@dataclass
class MoistureState:
    """Persistent moisture fields; all arrays participate in restart state."""
    rain_old: object
    t2_old: object
    q2_old: object
    psfc_old: object
    fmc_gc: object
    fmep: object
    fmc_equi: object
    fmc_lag: object
    rh_fire: object

    @classmethod
    def zeros(cls, shape, table=None, *, class_count=None, require_active=True):
        import cupy as cp
        table = table or FuelTable()
        active = int(table.moisture["moisture_classes"])
        nc = active if class_count is None else int(class_count)
        if nc < 1 or (require_active and active > nc):
            raise ValueError("nfmc must be positive and contain every active moisture class")
        zeros = lambda extra=(): cp.zeros((*extra,*shape), dtype=cp.float32)
        return cls(zeros(),zeros(),zeros(),zeros(),zeros((nc,)),zeros((2,)),
                   zeros((nc,)),zeros((nc,)),zeros())

    def arrays(self):
        return {name: getattr(self,name) for name in STATE_FIELDS}


def module_source(kernel_dir=None):
    from gpuwm.core.kernels import _preamble
    root = Path(__file__).parent / "kernels" if kernel_dir is None else Path(kernel_dir)
    return _preamble(root)+"".join((root/name).read_text(encoding="utf-8") for name in
                              ("glibc_flt32.cuh","sfire_libm.cuh","sfire_moisture.cu"))


@lru_cache(maxsize=None)
def _module(device):
    import cupy as cp
    from cupy.cuda import compiler
    from gpuwm.certify.kernel_manifest import record_module
    from gpuwm.kernel_compile_notice import observe_module_compile
    source = module_source()
    with cp.cuda.Device(device), observe_module_compile(MODULE_KEY):
        ptx,_ = compiler.compile_using_nvrtc(source,MODULE_OPTIONS,None,"sfire_moisture.cu")
        result = cp.cuda.function.Module()
        result.load(ptx.encode() if isinstance(ptx,str) else ptx)
    record_module(MODULE_KEY,source=source,options=MODULE_OPTIONS,module=None)
    return result


def advance_moisture(state, moisture_dt, rainc, rainnc, t2, q2, psfc, *,
                     initialize=False, table=None, fmep_decay_tlag=999999.0):
    """Update state and diagnostics from accumulated rain and surface fields.

    ``fmc_gc`` and its diagnostics are ``(class,ny,nx)``, ``fmep`` is
    ``(2,ny,nx)`` and surface fields are ``(ny,nx)``. ``initialize=True``
    follows each class's namelist initialization option. Subsequent calls
    average the previous and current surface conditions, as WRF does.
    """
    import cupy as cp
    table = table or FuelTable()
    table.validate()
    if not math.isfinite(moisture_dt) or moisture_dt < 0:
        raise ValueError("moisture_dt must be nonnegative to avoid reversing moisture evolution")
    if not math.isfinite(fmep_decay_tlag) or fmep_decay_tlag <= 0:
        raise ValueError("fmep_decay_tlag must be positive to define assimilated-offset decay")
    nc = int(table.moisture["moisture_classes"])
    shape = state.rain_old.shape
    if state.fmc_gc.ndim != 3 or state.fmc_gc.shape[0] < nc or state.fmc_gc.shape[1:] != shape or state.fmep.shape != (2,*shape):
        raise ValueError("moisture state class and offset dimensions must match the fuel table")
    for name,value in state.arrays().items():
        if not isinstance(value,cp.ndarray) or value.dtype != cp.float32 or not value.flags.c_contiguous:
            raise ValueError(f"moisture state {name} must be a contiguous float32 device array")
        expected=state.fmc_gc.shape if name in ("fmc_gc","fmc_equi","fmc_lag") else ((2,*shape) if name == "fmep" else shape)
        if value.shape != expected:
            raise ValueError(f"moisture state {name} shape must be {expected}")
    inputs = [_array(value,shape) for value in (rainc,rainnc,t2,q2,psfc)]
    m=table.moisture
    initial=[m[key] for key in ("fmc_1h","fmc_10h","fmc_100h","fmc_1000h","fmc_live")]
    params=cp.asarray(np.asarray([m["drying_lag"],m["wetting_lag"],m["saturation_moisture"],
                      m["saturation_rain"],m["rain_threshold"],initial],dtype=np.float32).T.copy())
    initialization=cp.asarray(m["fmc_gc_initialization"],dtype=cp.int32)
    status=cp.zeros((),dtype=cp.int32)
    n=state.rain_old.size
    kernel=_module(cp.cuda.Device().id).get_function("sfire_advance_moisture")
    kernel(((n+127)//128,),(128,),
        (*inputs,*[getattr(state,name) for name in STATE_FIELDS],params,initialization,status,
         np.int32(n),np.int32(nc),np.int32(bool(initialize)),np.float32(moisture_dt),
         np.float32(fmep_decay_tlag),np.float32(table.scalars["fuelmc_g"])))
    error=int(status.item())
    if error == 1:
        raise ValueError("SFIRE moisture requires positive T2 and PSFC and nonnegative Q2")
    if error == 2:
        raise ValueError("SFIRE equilibrium moisture produced NaN from surface conditions")
    return state
