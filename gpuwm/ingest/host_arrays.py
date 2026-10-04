"""Large host setup arrays copied directly by bounded native workers."""
from __future__ import annotations
import ctypes
import numpy as np


def _library():
    from gpuwm.core import portable_math as pm
    try:
        return pm, pm._load()
    except FileNotFoundError:
        return pm, None


def copy_float32(target, source, *, workers=None):
    """Return whether a supported native copy filled this host target."""
    if (not isinstance(target, np.ndarray) or not isinstance(source, np.ndarray) or source.shape != target.shape
            or source.dtype not in (np.float64, np.float32)
            or target.dtype != np.float32 or not target.flags.writeable
            or not source.flags.c_contiguous or not target.flags.c_contiguous
            or not source.flags.aligned or not target.flags.aligned
            or np.shares_memory(source, target)):
        return False
    pm, library = _library()
    entry = getattr(library, "gpuwm_host_copy_f64_f32" if source.dtype == np.float64
                     else "gpuwm_parallel_copy_f32", None)
    if entry is None:
        return False
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    entry.argtypes = [pointer, pointer, size, size]
    entry.restype = ctypes.c_int32
    result = entry(source.ctypes.data, target.ctypes.data, source.size, pm._workers(workers))
    if result:
        raise RuntimeError(f"native host setup copy failed with code {result}")
    return True


def stagger_pressure(source, axis, *, workers=None):
    """Return an exact native staggered array, or None for the reference path."""
    if (not isinstance(source, np.ndarray) or source.dtype != np.float64
            or source.ndim != 3 or not source.flags.c_contiguous or not source.flags.aligned
            or source.size == 0 or axis not in (1, 2)):
        return None
    pm, library = _library()
    entry = getattr(library, "gpuwm_host_stagger_f64", None)
    if entry is None:
        return None
    shape = list(source.shape)
    shape[axis] += 1
    target = np.empty(shape, dtype=np.float64)
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    entry.argtypes = [pointer, pointer, size, size, size, ctypes.c_uint32, size]
    entry.restype = ctypes.c_int32
    result = entry(source.ctypes.data, target.ctypes.data, *source.shape,
                   axis, pm._workers(workers))
    if result:
        raise RuntimeError(f"native host pressure staggering failed with code {result}")
    return target


def sum_fields(sources, *, workers=None):
    """Widen and sum independent fields in the original left-to-right order."""
    sources = tuple(sources)
    if (not sources or any(not isinstance(value, np.ndarray)
                          or value.dtype not in (np.float32, np.float64)
                          or not value.flags.c_contiguous or not value.flags.aligned for value in sources)
            or any(value.shape != sources[0].shape for value in sources)):
        return None
    pm, library = _library()
    entry = getattr(library, "gpuwm_host_sum_f64", None)
    if entry is None:
        return None
    target = np.empty(sources[0].shape, dtype=np.float64)
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    source_pointers = (pointer * len(sources))(*(value.ctypes.data for value in sources))
    kinds = (ctypes.c_uint32 * len(sources))(*(value.itemsize for value in sources))
    entry.argtypes = [ctypes.POINTER(pointer), ctypes.POINTER(ctypes.c_uint32),
                      size, pointer, size, size]
    entry.restype = ctypes.c_int32
    result = entry(source_pointers, kinds, len(sources), target.ctypes.data,
                   target.size, pm._workers(workers))
    if result:
        raise RuntimeError(f"native host field sum failed with code {result}")
    return target


def difference_float32(target, first, second, *, workers=None):
    """Fill target with the exact widened subtraction followed by f32 cast."""
    if (not isinstance(target, np.ndarray) or target.dtype != np.float32
            or not target.flags.c_contiguous or not target.flags.aligned or not target.flags.writeable
            or any(not isinstance(value, np.ndarray) or value.dtype != np.float64
                   or not value.flags.c_contiguous or not value.flags.aligned or value.shape != target.shape
                   or np.shares_memory(value, target) for value in (first, second))):
        return False
    pm, library = _library()
    entry = getattr(library, "gpuwm_host_difference_f64_f32", None)
    if entry is None:
        return False
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    entry.argtypes = [pointer, pointer, pointer, size, size]
    entry.restype = ctypes.c_int32
    result = entry(first.ctypes.data, second.ctypes.data, target.ctypes.data,
                   target.size, pm._workers(workers))
    if result:
        raise RuntimeError(f"native host state difference failed with code {result}")
    return True


def deepest_level(pressure, field, *, workers=None):
    """Each column's ``field`` value at its level of greatest ``pressure``.

    The surface pseudo-level WPS takes for a number field that has no
    two-metre product, whichever way the source levels run.  The level is
    the first one holding the column's maximum and a NaN pressure counts
    as the maximum, the order :func:`numpy.argmax` reads a column in, so
    the result is byte for byte ``take_along_axis(field, argmax(pressure,
    axis=0)[None], axis=0)[0]``.  Returns ``None`` for the reference path
    (a device array, an unsupported layout, or a library that predates
    the entry).
    """
    if (not isinstance(pressure, np.ndarray) or not isinstance(field, np.ndarray)
            or pressure.dtype not in (np.float32, np.float64) or field.dtype != np.float32
            or pressure.ndim != 3 or pressure.shape != field.shape or pressure.size == 0
            or not pressure.flags.c_contiguous or not field.flags.c_contiguous
            or not pressure.flags.aligned or not field.flags.aligned):
        return None
    pm, library = _library()
    entry = getattr(library, "gpuwm_host_deepest_level_f32", None)
    if entry is None:
        return None
    levels = pressure.shape[0]
    target = np.empty(pressure.shape[1:], dtype=np.float32)
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    entry.argtypes = [pointer, ctypes.c_uint32, pointer, pointer, size, size, size]
    entry.restype = ctypes.c_int32
    result = entry(pressure.ctypes.data, pressure.itemsize, field.ctypes.data,
                   target.ctypes.data, levels, target.size, pm._workers(workers))
    if result:
        raise RuntimeError(f"native host deepest-level selection failed with code {result}")
    return target


def geopotential_cache(state, source, gravity):
    """Return whether the exact native store/cache operation was available."""
    if (source.ndim < 1 or source.dtype != np.float64 or not source.flags.c_contiguous or not source.flags.aligned
            or source.size == 0 or source.shape != state.phb.shape
            or state.phb.dtype != np.float32 or state.dphb_resid.dtype != np.float32
            or state.dphb_resid.shape != source[1:].shape
            or not state.phb.flags.c_contiguous or not state.dphb_resid.flags.c_contiguous
            or not state.phb.flags.aligned or not state.dphb_resid.flags.aligned
            or not state.phb.flags.writeable or not state.dphb_resid.flags.writeable
            or np.shares_memory(source, state.phb)
            or np.shares_memory(source, state.dphb_resid)
            or np.shares_memory(state.phb, state.dphb_resid)):
        return False
    pm, library = _library()
    entry = getattr(library, "gpuwm_host_geopotential", None)
    if entry is None:
        return False
    snapshot = np.empty_like(source)
    minimum = ctypes.c_double()
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    entry.argtypes = [pointer, pointer, pointer, pointer, size, size,
                      ctypes.c_double, ctypes.POINTER(ctypes.c_double), size]
    entry.restype = ctypes.c_int32
    result = entry(source.ctypes.data, state.phb.ctypes.data, snapshot.ctypes.data,
                   state.dphb_resid.ctypes.data, source.shape[0],
                   source.size // source.shape[0], gravity, ctypes.byref(minimum), pm._workers(None))
    if result:
        raise RuntimeError(f"native host geopotential cache failed with code {result}")
    state._phb_host = snapshot
    if source.shape[0] <= 2:
        state._dz_min = None
    elif np.isnan(minimum.value):
        # Preserve the original reduction's NaN/signed-zero behavior for
        # unusual profiles, without changing the normal data path.
        height = 0.5 * (source[:-1] + source[1:]) / gravity
        state._dz_min = float(np.diff(height, axis=0).min())
    else:
        state._dz_min = minimum.value
    return True
