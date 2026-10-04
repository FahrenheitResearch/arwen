"""Host-retained CUDA preparation, using the ordinary kernels in bounded batches.

Only transfers, slicing and kernel launches live here. Column recurrences retain
their whole vertical column, and staggered row batches retain their donor halo.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gpuwm.ingest.preprocess_backend import CudaPreprocessBackend


def _host(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


def _release():
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()


def _batches(length, count):
    for start in range(0, length, count):
        yield start, min(length, start + count)


def _elementwise(function, values, cells, **options):
    arrays = np.broadcast_arrays(*(_host(value) for value in values))
    shape = arrays[0].shape
    outputs = None
    multiple = False
    for start, stop in _batches(arrays[0].size, cells):
        # flat slicing copies only this batch. Reshaping a broadcast 2-D
        # rotation plane into a 3-D flat operand copied all levels on host.
        result = function(*(value.flat[start:stop] for value in arrays), **options)
        multiple = isinstance(result, tuple)
        parts = result if multiple else (result,)
        if outputs is None:
            outputs = [np.empty(arrays[0].size, dtype=value.dtype) for value in parts]
        for output, value in zip(outputs, parts):
            output[start:stop] = _host(value)
        del result, parts, value
        _release()
    if outputs is None:
        return np.empty(shape, dtype=np.float64)
    outputs = tuple(value.reshape(shape) for value in outputs)
    return outputs if multiple else outputs[0]


class _RegularPlan:
    def __init__(self, backend, latitude, longitude, target_lat, target_lon):
        self.backend = backend
        self.latitude, self.longitude = latitude, longitude
        self.target_lat, self.target_lon = target_lat, target_lon
        self.source_shape = (len(latitude), len(longitude))
        self.target_shape = target_lat.shape

    def apply(self, field, method="parabolic", *, source_support=False):
        from gpuwm.ingest.horiz import _RegularGpuPlan
        from gpuwm.ingest.atmospheric_window import WindowedAtmosphericField
        values = _host(getattr(field, "values", field))
        leading = values.shape[:-2]
        planes = values.reshape((-1, *values.shape[-2:]))
        output = np.empty((planes.shape[0], *self.target_shape), dtype=np.float32)
        rows = max(1, self.backend.chunk_cells // max(1, self.target_shape[-1]))
        for start, stop in _batches(self.target_shape[0], rows):
            plan = _RegularGpuPlan(self.latitude, self.longitude,
                                   self.target_lat[start:stop], self.target_lon[start:stop])
            for level in range(planes.shape[0]):
                operand = planes[level]
                if isinstance(field, WindowedAtmosphericField):
                    operand = WindowedAtmosphericField(planes[level:level + 1], field.window)
                result = plan.apply(operand, method=method, source_support=source_support)
                output[level, start:stop] = _host(result).reshape((stop - start, *self.target_shape[1:]))
                del result
                _release()
            del plan
            _release()
        return output.reshape((*leading, *self.target_shape))


class _VerticalPlan:
    def __init__(self, backend, source, surface, target):
        self.backend = backend
        self.source, self.surface, self.target = map(_host, (source, surface, target))

    def apply(self, field, surface_value, **options):
        field, surface_value = map(_host, (field, surface_value))
        output = np.empty(self.target.shape, dtype=np.float32)
        rows = self.backend.column_rows((max(self.source.shape[0], self.target.shape[0]),
                                         *self.source.shape[1:]))
        for start, stop in _batches(self.source.shape[1], rows):
            plan = CudaPreprocessBackend.prepare_wrf_vertical(
                self.backend, self.source[:, start:stop], self.surface[start:stop],
                self.target[:, start:stop])
            result = plan.apply(field[:, start:stop], surface_value[start:stop], **options)
            output[:, start:stop] = _host(result)
            del plan, result
            _release()
        return output


class BoundedCudaPreprocessBackend(CudaPreprocessBackend):
    """The same CUDA arithmetic, with host ownership between kernel batches."""
    bounded_cuda = True

    def __init__(self, *, device_budget_bytes, host_workers=None, source_staging_bytes=0):
        super().__init__(host_workers=host_workers)
        self.device_budget_bytes = int(device_budget_bytes)
        self.source_staging_bytes = int(source_staging_bytes)
        # Sixty-four FP64 arrays per cell cover the largest REAL operation;
        # half the pool budget stays free for geometry and allocator fragments.
        self.chunk_cells = max(1, (self.device_budget_bytes - self.source_staging_bytes) // 1024)
        self.real_ops = BoundedRealOperations(self)

    @property
    def array_module(self):
        return np

    def column_rows(self, shape):
        return max(1, self.chunk_cells // max(1, int(shape[0]) * int(shape[-1])))

    def float32(self, value):
        return _host(value).astype(np.float32, copy=False)

    def bool_array(self, value):
        return _host(value).astype(np.bool_, copy=False)

    def regular_plan(self, latitude, longitude, target_lat, target_lon):
        return _RegularPlan(self, latitude, longitude, target_lat, target_lon)

    def masked_nearest(self, field, latitude, longitude, target_lat, target_lon,
                       source_landmask, target_landmask, **kwargs):
        from gpuwm.ingest.horiz import masked_nearest_gpu
        output = np.empty(target_lat.shape, dtype=np.float32)
        rows = max(1, self.chunk_cells // max(1, target_lat.shape[-1]))
        for start, stop in _batches(target_lat.shape[0], rows):
            result = masked_nearest_gpu(field, latitude, longitude,
                target_lat[start:stop], target_lon[start:stop], source_landmask,
                target_landmask[start:stop], **kwargs)
            output[start:stop] = _host(result)
            del result
            _release()
        return output

    def rotate_earth_to_grid(self, *args):
        from gpuwm.ingest.horiz import rotate_earth_to_grid_gpu
        return _elementwise(rotate_earth_to_grid_gpu, args, self.chunk_cells)

    def era5_rh_to_water(self, *args):
        from gpuwm.ingest.horiz import _era5_rh_to_water_gpu
        return _elementwise(_era5_rh_to_water_gpu, args, self.chunk_cells)

    def divide_float32(self, value, divisor):
        from gpuwm.ingest.horiz import _divide_float32_gpu
        return _elementwise(lambda value: _divide_float32_gpu(value, divisor),
                            (value,), self.chunk_cells)

    def prepare_wrf_vertical(self, source_pressure, surface_pressure, target_pressure):
        return _VerticalPlan(self, source_pressure, surface_pressure, target_pressure)


class BoundedRealOperations:
    """CUDA REAL helpers with full columns and bounded horizontal residency."""
    def __init__(self, backend):
        self.backend = backend

    @staticmethod
    def _cp():
        return np

    @staticmethod
    def widen(value):
        return _host(value).astype(np.float64, copy=False)

    @staticmethod
    def float32(value):
        return _host(value).astype(np.float32, copy=False)

    @staticmethod
    def upload_base(base):
        return base

    @staticmethod
    def load_base(state, coord, host_base, base):
        state.load_base(coord, host_base)

    @staticmethod
    def export_result(result, *, base=None):
        return result

    def __getattr__(self, name):
        from gpuwm.ingest import real, real_device
        if name in ("_ordered_levels", "_wrf_flag_sh_surface_specific_humidity",
                    "_refuse_non_finite_prognostic_qv", "_floor_flag_sh_surface_mixing_ratio",
                    "_floor_sh_vertical_undershoot"):
            return getattr(real, name)
        function = getattr(real_device, name)
        def apply(*args, **options):
            try:
                return _elementwise(function, args, self.backend.chunk_cells, **options)
            except ValueError:
                # Preserve global refusal counts and the first failing CPU slab.
                getattr(real, name)(*args, **options)
                raise
        return apply

    def _column_call(self, function, arrays, *, base=None, **options):
        shape = next(value.shape for value in arrays if getattr(value, "ndim", 0) == 3)
        rows = self.backend.column_rows(shape)
        outputs = None
        multiple = False
        for start, stop in _batches(shape[1], rows):
            def slab(value):
                return value[..., start:stop, :] if getattr(value, "ndim", 0) >= 2 else value
            kw = dict(options)
            if base is not None:
                kw["base"] = SimpleNamespace(**{name: slab(value) for name, value in vars(base).items()})
            result = function(*(slab(value) for value in arrays), **kw)
            multiple = isinstance(result, tuple)
            parts = result if multiple else (result,)
            if outputs is None:
                outputs = [np.empty((*value.shape[:-2], shape[1], shape[2]), dtype=value.dtype)
                           if value.ndim >= 2 else _host(value) for value in parts]
            for output, value in zip(outputs, parts):
                if value.ndim >= 2:
                    output[..., start:stop, :] = _host(value)
            del result, parts, value
            _release()
        return tuple(outputs) if multiple else outputs[0]

    def _integrate_moisture(self, *args, **options):
        from gpuwm.ingest import real, real_device
        order = np.argsort(-args[1][:, 0, 0])
        try:
            return self._column_call(real_device._integrate_moisture, args,
                                     _order=order, **options)
        except ValueError:
            # Report the column in the whole grid, not in the failing batch.
            real._integrate_moisture(*args, **options)
            raise

    def dry_pressure_ladder(self, dry_mass, c3, c4, p_top):
        from gpuwm.ingest import real_device
        shape = (len(c3), *dry_mass.shape)
        output = np.empty(shape, dtype=np.float64)
        for start, stop in _batches(shape[1], self.backend.column_rows(shape)):
            result = real_device.dry_pressure_ladder(dry_mass[start:stop], c3, c4, p_top)
            output[:, start:stop] = result.get()
            del result
            _release()
        return output

    def _rebalance_moist_pressure(self, pressure_guess, qtot, dry_mass, base, coord, **options):
        from gpuwm.ingest import real_device
        return self._column_call(
            lambda p, q, m, base: real_device._rebalance_moist_pressure(p, q, m, base, coord, **options),
            (pressure_guess, qtot, dry_mass), base=base)

    def _fp32_geopotential_split(self, base, coord, dry_mass, alpha, **options):
        from gpuwm.ingest import real_device
        return self._column_call(
            lambda m, a, base: real_device._fp32_geopotential_split(base, coord, m, a, **options),
            (dry_mass, alpha), base=base)

    def _pressure_at_u(self, pressure):
        return self._stagger(pressure, 2)

    def _pressure_at_v(self, pressure):
        return self._stagger(pressure, 1)

    def _stagger(self, pressure, axis):
        from gpuwm.ingest import real_device
        shape = list(pressure.shape)
        shape[axis] += 1
        output = np.empty(shape, dtype=np.float64)
        rows = self.backend.column_rows(pressure.shape)
        for start, stop in _batches(shape[1], rows):
            first = max(0, start - 1) if axis == 1 else start
            last = min(pressure.shape[1], stop + 1) if axis == 1 else stop
            result = real_device._pressure_at(pressure[:, first:last], axis)
            output[:, start:stop] = result[:, start - first:stop - first].get()
            del result
            _release()
        return output
