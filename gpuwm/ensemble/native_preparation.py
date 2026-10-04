"""Buffer orchestration for native ensemble input transforms."""
from __future__ import annotations

import ctypes
import numpy as np


class NativeEnsemblePreparation:
    def __init__(self, bridge=None):
        from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
        self.backend = CpuPreprocessBackend(bridge)
        self.path = self.backend.path
        try:
            version = self.backend._library.gpuwm_ensemble_preparation_abi_version
            version.argtypes = []
            version.restype = ctypes.c_uint32
            if int(version()) != 3:
                raise RuntimeError("ensemble preparation bridge ABI differs; rebuild tools/grib1_bridge")
            self._recenter = self.backend._library.gpuwm_ensemble_recenter_f32
            self._blend = self.backend._library.gpuwm_ensemble_time_blend_f32
            self._stagger = self.backend._library.gpuwm_ensemble_pressure_stagger_f32
            self._levels = self.backend._library.gpuwm_ensemble_pressure_levels_f32
            self._humidity = self.backend._library.gpuwm_ensemble_humidity_f32
        except AttributeError as error:
            raise RuntimeError(
                "the CPU preparation bridge lacks ensemble recentering and "
                "physical-time interpolation; rebuild tools/grib1_bridge "
                "before preparing member anomalies") from error
        pointer, size, real = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double
        self._recenter.argtypes = [pointer]*4 + [size]*3 + [real]*6 + [size]
        self._recenter.restype = ctypes.c_int32
        self._blend.argtypes = [pointer]*3 + [size, real]
        self._blend.restype = ctypes.c_int32
        self._stagger.argtypes = [pointer]*2 + [size]*3 + [ctypes.c_int32]
        self._stagger.restype = ctypes.c_int32
        self._levels.argtypes = [pointer]*2 + [size]*2
        self._levels.restype = ctypes.c_int32
        self._humidity.argtypes = [pointer]*4 + [size, ctypes.c_int32] + [real]*5
        self._humidity.restype = ctypes.c_int32

    def recenter(self, base, donors, indices, out, bounds, *, workers):
        if type(workers) is not int or workers < 1:
            raise ValueError("ensemble preprocessing workers must be a positive integer")
        code = self._recenter(base.ctypes.data, donors.ctypes.data, indices.ctypes.data,
                              out.ctypes.data, base.size, donors.shape[0], indices.size,
                              bounds.amplitude, bounds.max_increment, bounds.lower,
                              bounds.upper, bounds.lower if bounds.input_lower is None else bounds.input_lower,
                              bounds.upper if bounds.input_upper is None else bounds.input_upper,
                              workers)
        self.backend._raise(code, "ensemble recentering")

    def time_blend(self, left, right, weight):
        for value in (left, right):
            if (not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32")
                    or not value.flags.c_contiguous):
                raise ValueError("physical time interpolation requires contiguous float32 arrays")
        if left.shape != right.shape:
            raise ValueError("physical time interpolation grids differ")
        out = np.empty_like(left)
        code = self._blend(left.ctypes.data, right.ctypes.data, out.ctypes.data, left.size, weight)
        self.backend._raise(code, "physical-time interpolation")
        return out

    def pressure_stagger(self, pressure, axis):
        if (pressure.ndim != 3 or pressure.dtype != np.dtype("float32")
                or not pressure.flags.c_contiguous or axis not in (1, 2)):
            raise ValueError("pressure staggering requires contiguous float32[z,y,x] and axis 1/2")
        shape = list(pressure.shape)
        shape[axis] += 1
        out = np.empty(shape, dtype=np.float32)
        code = self._stagger(pressure.ctypes.data, out.ctypes.data, *pressure.shape, axis)
        self.backend._raise(code, "ensemble pressure staggering")
        return out

    def pressure_levels(self, levels_hpa, shape_yx):
        levels = np.ascontiguousarray(levels_hpa, dtype=np.float64)
        if levels.ndim != 1 or len(shape_yx) != 2 or any(int(n) < 1 for n in shape_yx):
            raise ValueError("native pressure levels need one axis and a positive horizontal shape")
        out = np.empty((len(levels), *shape_yx), dtype=np.float32)
        code = self._levels(levels.ctypes.data, out.ctypes.data, len(levels), out.shape[1]*out.shape[2])
        self.backend._raise(code, "ensemble pressure levels")
        return out

    def humidity(self, temperature, pressure, values, *, to_relative):
        from gpuwm.ingest import real
        arrays = [np.ascontiguousarray(value, dtype=np.float32) for value in (temperature, pressure, values)]
        if not all(value.shape == arrays[0].shape for value in arrays) or not arrays[0].size:
            raise ValueError("humidity conversion requires one common physical grid")
        out = np.empty_like(arrays[0])
        lower, _ = real._specific_humidity_undershoot_bound(True, None)
        code = self._humidity(*(value.ctypes.data for value in arrays), out.ctypes.data,
                              out.size, 1 if to_relative else 2, lower,
                              real.c.SVP1, real.c.SVP2, real.c.SVPT0, real.c.SVP3)
        self.backend._raise(code, "ensemble native humidity representation")
        return out
