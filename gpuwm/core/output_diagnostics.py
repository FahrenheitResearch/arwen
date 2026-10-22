"""GPU output accumulators from the pinned operational WRF diagnostics.

Bucket reset occurs at step start, independently of history writes.
Source: NOAA-EMC/HRRR 40ee6058, phys/module_diag_gsd.F:724-813.
"""
from __future__ import annotations
import math
import numpy as np

PRECIP_HISTORY = {
    prefix + suffix: (prefix.upper() + suffix, "mm", description)
    for suffix in ("", "1")
    for prefix, description in (("prec_acc_c", "Interval convective precipitation"),
        ("prec_acc_nc", "Interval grid-scale precipitation"),
        ("snow_acc_nc", "Interval snow including frozen convective rain"),
        ("graup_acc_nc", "Interval grid-scale graupel"))}
from gpuwm.core.gsd_diagnostics import HISTORY as GSD_HISTORY, PERSISTENT as GSD_PERSISTENT
HISTORY = PRECIP_HISTORY | GSD_HISTORY
PERSISTENT = (*PRECIP_HISTORY, *GSD_PERSISTENT)

def scratch_shapes(cfg):
    shapes = {slot: (cfg.ny, cfg.nx) for slot in PRECIP_HISTORY
              if getattr(cfg, "prec_acc_dt1" if slot.endswith("1") else "prec_acc_dt") > 0}
    if shapes:
        shapes["output_zero"] = (cfg.ny, cfg.nx)
    from gpuwm.core.gsd_diagnostics import scratch_shapes as gsd_shapes
    shapes.update(gsd_shapes(cfg))
    return shapes

def initialize(state, cfg):
    for slot, shape in scratch_shapes(cfg).items():
        state.scratch(shape, slot)
    if cfg.gsd_diagnostics:
        for slot in ("wspd10", "wspd80"):
            state.existing_scratch(slot)[...] = -99.

def boundary_due(seconds, interval_seconds):
    # The fork uses MOD(curr_secs2, interval)==0, at the beginning of a step.
    return seconds == 0 or math.isclose(seconds % interval_seconds, 0., abs_tol=1.e-6)

def update_precipitation(state, cfg):
    if cfg.prec_acc_dt <= 0 and cfg.prec_acc_dt1 <= 0:
        return
    from gpuwm.core.kernels import get_kernel
    physics = state.physics
    micro = getattr(physics, "microphysics", None)
    zero = state.scratch((cfg.ny, cfg.nx), "output_zero")
    rainc = getattr(physics, "cu_raincv", None)
    def amount(name):
        value = getattr(micro, name, None)
        return zero if value is None or cfg.mp_physics == 0 else value
    temperature = getattr(physics, "fields", {}).get("t2", zero)
    count = cfg.nx * cfg.ny
    for suffix, minutes in (("", cfg.prec_acc_dt), ("1", cfg.prec_acc_dt1)):
        if minutes <= 0:
            continue
        get_kernel("output_diagnostics", "precipitation_buckets")(((count+255)//256,), (256,),
            (zero if rainc is None else rainc, amount("rainncv"),
             amount("snowncv"), amount("graupelncv"), temperature,
             *(state.existing_scratch(name+suffix) for name in
               ("prec_acc_c", "prec_acc_nc", "snow_acc_nc", "graup_acc_nc")),
             np.int32(count), np.int32(boundary_due(state.elapsed_seconds, minutes*60.))))
