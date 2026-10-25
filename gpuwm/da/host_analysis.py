"""Buffer and metadata adapter for the mandatory Rust host analysis kernels."""
from __future__ import annotations
import ctypes
import numpy as np


def _entry(name):
    from gpuwm.core.portable_math import _load
    entry = getattr(_load(), name, None)
    if entry is None:
        raise RuntimeError(f"the CPU bridge lacks {name}; rebuild tools/grib1_bridge")
    entry.restype = ctypes.c_int32
    return entry


def binary(first, second, *, subtract=False, clip=False):
    """Float64 addition/subtraction, clipping and statistics in Rust."""
    first, second = np.broadcast_arrays(np.asarray(first, dtype=np.float64),
                                        np.asarray(second, dtype=np.float64))
    shape = first.shape
    first, second = (np.ascontiguousarray(v) for v in (first, second))
    out = np.empty_like(first)
    stats = np.empty(5, dtype=np.float64)
    entry = _entry("gpuwm_analysis_binary_f64")
    entry.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    result = entry(first.ctypes.data, second.ctypes.data, out.ctypes.data, stats.ctypes.data,
                   out.size, int(subtract), int(clip))
    if result == 3:
        raise ValueError("analysis arithmetic is not finite")
    if result:
        raise RuntimeError(f"Rust analysis arithmetic failed with code {result}")
    return out.reshape(shape), dict(values=int(stats[0]), min=float(stats[1]), nonzero=int(stats[2]),
                     rms=float(stats[3]), max_abs=float(stats[4]))


def hydrostatic(state, before, after, *, hypsometric_opt, mup_after=None):
    """Run the host before/after column integration, retaining CUDA's own route."""
    from gpuwm.core import constants as c
    from gpuwm.da.hydrostatic import HydrostaticError, loading_masses
    shape = tuple(state.p.shape)
    if len(shape) != 3 or not all(shape):
        raise HydrostaticError(f"hydrostatic column needs a nonempty 3D pressure, got {shape}")
    nz, ny, nx = shape
    masses = loading_masses(state)
    if "qv" not in masses:
        raise HydrostaticError("a hydrostatic rebalance needs a moist state (no qv)")
    def array(value, expected):
        try:
            return np.ascontiguousarray(np.broadcast_to(np.asarray(value, dtype=np.float64), expected))
        except ValueError as exc:
            raise HydrostaticError(f"column shape does not match {expected}: {np.shape(value)}") from exc
    def profile(name, size):
        value = getattr(state, name, None)
        if value is None:
            raise HydrostaticError(f"column lacks coordinate {name}")
        if np.shape(value) != (size,):
            raise HydrostaticError(f"coordinate {name} needs {size} levels")
        return array(value, (size,))
    def loading(values):
        total = np.zeros(shape, dtype=np.float64)
        for name in masses:
            total, _ = binary(total, array(values[name], shape))
        return total
    thb = np.asarray(state.thb, dtype=np.float64)
    if thb.ndim == 1:
        thb = thb[:, None, None]
    if mup_after is not None and np.shape(mup_after) != np.shape(state.mup):
        raise HydrostaticError(f"the analysed dry mass is {np.shape(mup_after)}, the column's {np.shape(state.mup)}")
    option = int(hypsometric_opt)
    if option not in (1, 2):
        raise HydrostaticError(f"hypsometric_opt must be 1 or 2, got {option}")
    if option == 2 and state.p_top is None:
        raise HydrostaticError("hypsometric_opt=2 needs state.p_top (load_base)")
    arrays = [array(before["p"], shape), array(before["thp"], shape), array(after["thp"], shape),
              array(before["qv"], shape), array(after["qv"], shape), loading(before), loading(after),
              array(thb, shape), array(state.mub2d, (ny,nx)), array(state.mup, (ny,nx)),
              array(state.mup if mup_after is None else mup_after, (ny,nx))]
    for name in ("c1f","c2f","c1h","c2h","c3f","c4f","c3h","c4h","rdnw","rdn","dnw"):
        # The other layer operator's profiles are never read by the kernel.
        unused = (option == 1 and name in ("c3f","c4f","c3h","c4h")) or (option == 2 and name in ("c1h","c2h","dnw"))
        arrays.append(np.zeros(nz+1 if name in ("c1f","c2f","c3f","c4f") else nz)
                      if unused else profile(name, nz+1 if name in ("c1f","c2f","c3f","c4f") else nz))
    constants = np.array([c.RD,c.RVOVRD,c.P0,c.RCP,float(state.p_top or 0.)], dtype=np.float64)
    phi, pressure = np.empty((nz+1,ny,nx),dtype=np.float64), np.empty(shape,dtype=np.float64)
    pointers = (ctypes.c_void_p * len(arrays))(*(v.ctypes.data for v in arrays))
    entry = _entry("gpuwm_analysis_hydro_f64")
    entry.argtypes = [ctypes.POINTER(ctypes.c_void_p)] + [ctypes.c_void_p]*3 + [ctypes.c_size_t]*2 + [ctypes.c_uint32]*2
    result = entry(pointers,constants.ctypes.data,phi.ctypes.data,pressure.ctypes.data,nz,ny*nx,option,int(mup_after is not None))
    if result == 3:
        raise HydrostaticError("the analysed hydrostatic pressure is not finite and positive or the geopotential increment is not finite")
    if result:
        raise HydrostaticError(f"Rust hydrostatic integration failed with code {result}")
    return phi, pressure
