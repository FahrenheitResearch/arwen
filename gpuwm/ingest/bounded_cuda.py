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


def _plane_rotation(function, values, cells):
    """A wind rotation in whole horizontal planes, or ``None`` when it is not one.

    The rotation kernel reads its geometry modulo the plane
    (``horizontal_rotate``), so a batch of whole planes takes the 2-D sin/cos
    once, as the resident road does.  Named breakage: :func:`_elementwise`
    broadcast the 2-D float64 geometry to the 3-D wind and flat-copied and
    cast it on the host for every batch of every wind pair, about 9 s of the
    start's main thread on one core (a 5090 host, full CONUS, 2026-10-21).  The
    words written are the same: each output cell is the same kernel on the
    same four float32 operands.
    """
    if len(values) != 4:
        return None
    u, v, sina, cosa = (_host(value) for value in values)
    if (u.shape != v.shape or sina.shape != cosa.shape or sina.ndim < 1
            or u.ndim <= sina.ndim or u.shape[-sina.ndim:] != sina.shape):
        return None
    plane = int(sina.size)
    u_planes = np.ascontiguousarray(u).reshape(-1, plane)
    v_planes = np.ascontiguousarray(v).reshape(-1, plane)
    out_u = np.empty(u_planes.shape, dtype=np.float32)
    out_v = np.empty(v_planes.shape, dtype=np.float32)
    per_batch = max(1, int(cells) // max(1, plane))
    geometry = (sina, cosa)
    for start, stop in _batches(u_planes.shape[0], per_batch):
        ou, ov = function(u_planes[start:stop].reshape((stop - start, *sina.shape)),
                          v_planes[start:stop].reshape((stop - start, *sina.shape)),
                          *geometry)
        out_u[start:stop] = _host(ou).reshape(stop - start, plane)
        out_v[start:stop] = _host(ov).reshape(stop - start, plane)
        del ou, ov
        _release()
    return out_u.reshape(u.shape), out_v.reshape(u.shape)


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
    #: A boolean field (the hydrometeor receipt's one-hot level masks) is
    #: uploaded as bytes and widened to the same 0/1 float32 words on the card.
    accepts_bool_mask = True

    #: Share of the device budget the prepared batch geometries may keep
    #: between applies.  A geometry is the batch's three float32 pressure
    #: blocks (about 0.8 GB for a whole 3 km CONUS column set).
    GEOMETRY_CACHE_SHARE = 0.25

    def __init__(self, backend, source, surface, target):
        self.backend = backend
        self.source, self.surface, self.target = map(_host, (source, surface, target))
        # Prepared geometries by batch, kept while they fit the share above.
        # Named breakage: the hydrometeor receipt applies this plan once per
        # source level (50+ times) and every apply re-uploaded and re-checked
        # each batch's pressures (8 s of the receipt thread, plus the pool
        # flushes after them; a 5090 host, full CONUS, 2026-10-21).  The geometry
        # depends only on the pressures, so a kept one is the same plan.
        self._geometries: dict = {}
        self._geometry_bytes = 0

    def _geometry(self, start, stop):
        plan = self._geometries.get((start, stop))
        if plan is not None:
            return plan
        plan = CudaPreprocessBackend.prepare_wrf_vertical(
            self.backend, self.source[:, start:stop], self.surface[start:stop],
            self.target[:, start:stop])
        size = 4 * (self.source[:, start:stop].size + self.surface[start:stop].size
                    + self.target[:, start:stop].size)
        budget = int(self.backend.device_budget_bytes * self.GEOMETRY_CACHE_SHARE)
        if self._geometry_bytes + size <= budget:
            self._geometries[(start, stop)] = plan
            self._geometry_bytes += size
        return plan

    def apply(self, field, surface_value, **options):
        field, surface_value = map(_host, (field, surface_value))
        output = np.empty(self.target.shape, dtype=np.float32)
        rows = self.backend.column_rows((max(self.source.shape[0], self.target.shape[0]),
                                         *self.source.shape[1:]))
        device_mask = None
        if field.dtype == np.bool_:
            # One contiguous upload of the mask bytes (a quarter of the float
            # words); each batch is cut and widened on the card.
            import cupy as cp
            device_mask = cp.asarray(np.ascontiguousarray(field))
        for start, stop in _batches(self.source.shape[1], rows):
            plan = self._geometry(start, stop)
            batch = (field[:, start:stop] if device_mask is None
                     else device_mask[:, start:stop].astype(cp.float32))
            result = plan.apply(batch, surface_value[start:stop], **options)
            output[:, start:stop] = _host(result)
            del plan, result, batch
            _release()
        del device_mask
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
        planes = _plane_rotation(rotate_earth_to_grid_gpu, args, self.chunk_cells)
        if planes is not None:
            return planes
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
