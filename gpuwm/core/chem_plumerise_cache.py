"""Reapply a GSL FRP injection interval to new emissions on selected columns.

This is a standalone cached-profile launcher, not the Freitas process surface.
ccpp-physics 3e6660c6df54e95a0871e990c2294dd397ae3860:
https://github.com/ufs-community/ccpp-physics/blob/3e6660c6df54e95a0871e990c2294dd397ae3860/physics/smoke_dust/module_plumerise.F90#L158-L166
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np


#: The plume translation unit's compile options.  ``--ftz=false`` because
#: CuPy's RawModule appends ``-ftz=true`` to every module it builds, and the
#: Freitas time integration carries cloud and hail water down through the
#: subnormal range: flushed, those words and the step count after them left
#: the gfortran oracle (measured on the RTX 4090 and 5090: qc/qh profile
#: words of 1e-33 to 1e-37 and 564 against 565 steps on three WRF-Chem
#: columns), unflushed they are its words.  The unit compiles through NVRTC
#: directly, as the legacy RRTMG adapter does (gpuwm/core/rrtmg_legacy.py).
PLUME_COMPILE_OPTIONS = ("-std=c++17", "--ftz=false")


@lru_cache(maxsize=None)
def _plume_module(device: int):
    import cupy as cp
    from cupy.cuda import compiler
    from gpuwm.core.kernels import MODULE_KEY_ROOT, module_source

    source = module_source("chem_plumerise")
    with cp.cuda.Device(device):
        ptx, _mapping = compiler.compile_using_nvrtc(
            source, ("-std=c++17", "--ftz=false"), None, "chem_plumerise.cu")
        module = cp.cuda.function.Module()
        module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    from gpuwm.certify.kernel_manifest import record_module
    record_module(f"{MODULE_KEY_ROOT}:chem_plumerise",
                  source=source, options=PLUME_COMPILE_OPTIONS)
    return module


def plume_function(name: str):
    """One entry of kernels/chem_plumerise.cu, compiled unflushed for this device."""
    import cupy as cp

    return _plume_module(cp.cuda.runtime.getDevice()).get_function(name)


@dataclass(frozen=True)
class PlumeParameters:
    """GSL wrapper:245-250 and rrfs_smoke_config.F90:42,45 defaults.

    Source: ccpp-physics commit 3e6660c6, physics/smoke_dust/.
    FRP inputs to the solver are already diurnally scaled by the fire process.
    """

    frp_min: float = 1.e7
    frp_max: float = 2.e10
    frp_wthreshold: float = 1.e9
    zpbl_threshold: float = 2.e3
    uspd_threshold: float = 5.
    alpha: float = .05
    wind_eff_opt: int = 1
    # rrfs_smoke_config.F90:31; Registry/registry.chem:3854.
    frp_cadence_minutes: int = 60
    landuse_cadence_minutes: int = 180
    # WRF module_plumerise1.F:225-231.
    landuse_gate: float = 1.e-6

    def __post_init__(self):
        values=(self.frp_min,self.frp_max,self.frp_wthreshold,self.zpbl_threshold,
                self.uspd_threshold,self.alpha,self.landuse_gate)
        if (not all(math.isfinite(v) for v in values) or self.frp_min<=0
                or self.frp_max<self.frp_min or self.frp_wthreshold<self.frp_min
                or self.alpha<=0 or self.uspd_threshold<0 or self.zpbl_threshold<0
                or self.wind_eff_opt not in (0,1) or self.landuse_gate!=1.e-6):
            raise ValueError('plume parameters need finite ordered FRP thresholds, positive entrainment, valid wind option and the pinned landuse gate; invalid parameters suppress real fires or divide by zero in plume heating')


def plume_call_due(arm, curr_secs, ktau, frequency_minutes, *, enabled=True,
                   adaptive=False, dt=0., stepfirepl=None):
    """Cadence only, before biomass-option admission by the integrating lane.

    GSL rrfs_smoke_wrapper.F90:326-327; WRF emissions_driver.F:686-698.
    WRF's fixed-step branch receives the driver's stepfirepl unchanged.
    """
    if arm not in ('frp', 'landuse'):
        raise ValueError("plume arm must be frp or landuse; another value "
                         "would apply the wrong call cadence")
    if not enabled:
        return False
    if arm == 'frp':
        if int(frequency_minutes)!=frequency_minutes:
            raise ValueError('FRP plume cadence requires integer minutes; fractional intervals schedule calls that the GSL integer namelist cannot represent')
        return frequency_minutes > 0 and (
            int(np.float32(curr_secs)) % max(1, 60*frequency_minutes) == 0 or ktau == 2)
    if ktau == 1:
        return True
    if adaptive:
        if frequency_minutes <= 0:
            return True
        # WRF uses REAL*8 clock arithmetic and the unsuffixed REAL(4) .01.
        period = float(np.float32(np.float32(frequency_minutes)*np.float32(60.)))
        target = int(curr_secs/period+1)*period
        return curr_secs+float(dt)+float(np.float32(.01)) >= target
    if stepfirepl is None or stepfirepl <= 0:
        raise ValueError("fixed-step landuse cadence needs positive stepfirepl; "
                         "a missing interval causes wrong emission call times "
                         "or integer division by zero")
    return ktau % stepfirepl == 0 or stepfirepl == 1


def distribute_cached(columns, k_min, k_max, flam_frac, ebu_in, z_at_w, ebu):
    """Write selected columns in-place; leave every other column untouched.

    Emissions have shape (nrows, ny, nx), output (nrows, nz, ny, nx).
    Bounds retain one-based GSL indices and exclusive k_max. Scratch storage
    is O(nfire) for validation, plus a gathered w-level array O(nfire*nz).
    """
    if os.environ.get("GPUWM_NO_LOCAL_GPU", "") not in ("", "0"):
        raise RuntimeError("GPUWM_NO_LOCAL_GPU forbids cached injection launch; "
                           "launching would use the protected local device")
    import cupy as cp

    arrays = (columns, k_min, k_max, flam_frac, ebu_in, z_at_w, ebu)
    if any(not isinstance(a, cp.ndarray) for a in arrays):
        raise TypeError("cached injection requires device arrays; host pointers "
                        "would be invalid CUDA kernel addresses")
    if any(not a.flags.c_contiguous for a in arrays):
        raise ValueError("cached injection requires contiguous arrays; strided "
                         "storage would inject emissions into the wrong cells")
    if columns.ndim != 1 or columns.dtype != cp.int32:
        raise ValueError("columns must be an int32 vector; invalid offsets "
                         "would address outside the grid")
    if ebu.ndim != 4 or ebu_in.ndim != 3 or z_at_w.ndim != 3:
        raise ValueError("injection needs rows_3d output, rows_2d emissions and "
                         "3d_w heights; other ranks corrupt level addressing")
    nr, nz, ny, nx = ebu.shape
    if (ebu_in.shape != (nr, ny, nx) or z_at_w.shape != (nz+1, ny, nx)
            or any(a.shape != (ny, nx) for a in (k_min, k_max, flam_frac))):
        raise ValueError("injection arrays must share the grid and row count; "
                         "mismatched shapes write outside the output")
    if (k_min.dtype != cp.int32 or k_max.dtype != cp.int32
            or any(a.dtype != cp.float32 for a in (flam_frac, ebu_in, z_at_w, ebu))):
        raise ValueError("injection uses int32 bounds and float32 fields; "
                         "other words are misread by the CUDA entry")
    if any(cp.shares_memory(ebu, a) for a in arrays[:-1]):
        raise ValueError("injection output cannot alias its inputs; writes "
                         "would overwrite later emissions or heights")
    if not columns.size:
        return
    nxy = ny * nx
    if bool(cp.any((columns < 0) | (columns >= nxy))):
        raise ValueError("fire column offsets must be in the grid; invalid "
                         "offsets read and write outside allocations")
    if cp.unique(columns).size != columns.size:
        raise ValueError("fire column offsets must be unique; duplicate "
                         "threads race on the same output column")
    lo = k_min.ravel()[columns]
    hi = k_max.ravel()[columns]
    if bool(cp.any((lo < 1) | (hi <= lo) | (hi > nz+1))):
        raise ValueError("injection bounds must address distinct w levels; "
                         "invalid bounds read outside the column or divide by zero")
    heights = z_at_w.reshape(nz+1, nxy)[:, columns]
    if bool(cp.any(~cp.isfinite(heights))) or bool(cp.any(cp.diff(heights, axis=0) <= 0)):
        raise ValueError("w levels must be finite and strictly increasing; "
                         "invalid thickness gives negative or undefined emissions")
    fraction = flam_frac.ravel()[columns]
    incoming = ebu_in.reshape(nr, nxy)[:, columns]
    if (bool(cp.any(~cp.isfinite(fraction)))
            or bool(cp.any((fraction < 0) | (fraction > 1)))
            or bool(cp.any(~cp.isfinite(incoming))) or bool(cp.any(incoming < 0))):
        raise ValueError("injection fractions must be finite in [0,1] and "
                         "emissions finite and nonnegative; invalid inputs "
                         "produce negative or undefined emitted mass")
    kernel = plume_function("ebu_distribute")
    kernel(((columns.size+63)//64,), (64,),
           (columns, k_min, k_max, flam_frac, ebu_in, z_at_w, ebu,
            np.int32(columns.size), np.int32(nz), np.int32(nxy), np.int32(nr)))
