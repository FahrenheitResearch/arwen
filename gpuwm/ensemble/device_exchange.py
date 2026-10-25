# SPDX-License-Identifier: Apache-2.0
"""Same-process borrowing of retained member arrays across CUDA streams.

The CUDA factory allocates one retained increment bank. Borrowing performs no
packing, host field transfer or numerical conversion. Backgrounds are read-only.
The caller owns scientific writeback (including consistency repair and derived
fields). A failed in-place consumer poisons the exchange: it cannot resume a
forecast with a partially changed state. This is not a cross-process protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from operator import index
from threading import get_ident
from types import MappingProxyType


@dataclass(frozen=True)
class StateIdentity:
    grid: str
    member_ids: tuple[int, ...]
    ticks: int
    tick_den: int

    def __post_init__(self):
        if any(isinstance(i, bool) for i in (*self.member_ids, self.ticks, self.tick_den)):
            raise TypeError('member IDs and clock ticks must be integers, not booleans')
        ids = tuple(index(i) for i in self.member_ids)
        if not isinstance(self.grid, str) or not self.grid or not ids or len(set(ids)) != len(ids) or min(ids) < 0:
            raise ValueError('grid identity and unique nonnegative global member IDs required')
        if index(self.ticks) < 0 or index(self.tick_den) <= 0:
            raise ValueError('valid time needs nonnegative ticks and a positive denominator')
        object.__setattr__(self, 'member_ids', ids)
        object.__setattr__(self, 'ticks', index(self.ticks))
        object.__setattr__(self, 'tick_den', index(self.tick_den))


class DeviceField:
    """Read-only metadata and strong allocation ownership, with native strides."""

    def __init__(self, array, *, writable, shared, members, device):
        desc = dict(array.__cuda_array_interface__)
        shape = tuple(desc['shape'])
        if (desc.get('version', 0) < 3 or not shape or min(shape) < 1
                or desc['data'][0] == 0 or desc.get('mask') is not None):
            raise ValueError('field requires a nonempty unmasked CUDA array interface v3')
        if int(array.device.id) != device:
            raise ValueError('all borrowed arrays must belong to the same CUDA device')
        if not shared and (len(shape) < 2 or shape[0] != members):
            raise ValueError('mutable member slabs must retain the leading member dimension')
        if writable and (shared or desc['data'][1]):
            raise ValueError('shared or read-only fields cannot be borrowed for writing')
        allowed = ('<f4', '<f8', '|f4', '|f8')
        if not writable:
            allowed += ('<i4', '|i4')
        if desc['typestr'] not in allowed:
            raise ValueError('field needs float32/float64, or read-only int32')
        strides = desc.get('strides')
        if strides is None:
            step = int(array.dtype.itemsize)
            result = []
            for extent in reversed(shape):
                result.append(step)
                step *= extent
            strides = tuple(reversed(result))
        if len(strides) != len(shape) or any(s <= 0 for s in strides):
            raise ValueError('negative or broadcast strides have no writable member ownership')
        span = int(array.dtype.itemsize)
        for stride, extent in sorted(zip(strides, shape)):
            if extent > 1 and stride < span:
                raise ValueError('overlapping field strides do not have exclusive element ownership')
            span += stride * (extent - 1)
        self._array = array
        self._pointer = int(desc['data'][0])
        self.byte_span = span
        self.shape, self.strides = shape, tuple(strides)
        self.dtype, self.device = desc['typestr'], device
        self.writable, self.shared = writable, shared

    @property
    def pointer(self):
        return self._pointer

    def descriptor(self):
        return dict(pointer=self.pointer, shape=self.shape, strides=self.strides,
                    dtype=self.dtype, device=self.device, writable=self.writable,
                    shared=self.shared)

    def unchanged(self, array):
        if array is not self._array:
            return False
        current = DeviceField(array, writable=self.writable, shared=self.shared,
                              members=self.shape[0], device=self.device)
        return current.descriptor() == self.descriptor()


class DeviceLease:
    def __init__(self, exchange, identity, fields, increments, stream):
        self._exchange = exchange
        self.identity = identity
        self.fields = MappingProxyType(fields)
        self.increments = MappingProxyType(increments)
        self.generation = exchange.generation
        self.stream = stream
        self.closed = False

    def complete(self, *, identity, refresh, increment_digests=None):
        """Finish increment writes, then run mandatory engine consistency work.

        refresh runs on the forecast stream after the consumer event. It must
        retain active physics, repair policies and boundary semantics. Passing
        a no-op is appropriate only for a verified unchanged-state consumer.
        """
        return self._exchange._complete(self, identity=identity, refresh=refresh,
                                        increment_digests=increment_digests)

    def seal(self):
        """Compute consumer-side increment digests on the consumer stream."""
        self._exchange._check(self)
        if self._exchange.digest is None:
            raise RuntimeError('this exchange has no device digest implementation')
        with self.stream:
            return {name: self._exchange.digest(array)
                    for name, array in self._exchange.increment_arrays.items()}

    def abort(self):
        self._exchange._abort(self)


class DeviceExchange:
    """One exclusive lease at a time, owned by the forecast's control thread."""

    def __init__(self, batch, *, producer_stream, event_factory, device,
                 writable_fields, increment_arrays, digest=None):
        self.batch = batch
        self.producer_stream = producer_stream
        self.event_factory = event_factory
        self.device = index(device)
        self.writable = frozenset(writable_fields)
        if not self.writable or self.writable - batch.storage.arrays.keys():
            raise ValueError('every writable field must exist in retained batch storage')
        if set(increment_arrays) != self.writable:
            raise ValueError('retained increment arrays must cover exactly the writable fields')
        self.increment_arrays = dict(increment_arrays)
        self.digest = digest
        for name, array in self.increment_arrays.items():
            target = batch.storage.arrays[name]
            field = DeviceField(array, writable=True, shared=False,
                                members=batch.members, device=self.device)
            if (field.shape != tuple(target.shape) or array is target
                    or field.pointer == int(target.__cuda_array_interface__['data'][0])):
                raise ValueError(f'{name}: increment must be separate storage with the background shape')
            if batch.storage.specs[name].ownership == 'shared':
                raise ValueError(f'{name}: shared fields cannot have increments')
        self.thread = get_ident()
        self.active = None
        self.poisoned = False
        self.generation = 0
        self._events = None
        self._validate_storage()

    def _validate_storage(self):
        backgrounds = [DeviceField(array, writable=False,
            shared=self.batch.storage.specs[name].ownership == 'shared',
            members=self.batch.members, device=self.device)
            for name, array in self.batch.storage.arrays.items()
            if not name.startswith('scratch:')]
        occupied = [(field.pointer, field.pointer + field.byte_span)
                    for field in backgrounds]
        for name, array in self.increment_arrays.items():
            field = DeviceField(array, writable=True, shared=False,
                                members=self.batch.members, device=self.device)
            if field.shape != tuple(self.batch.storage.arrays[name].shape):
                raise ValueError(f'{name}: increment shape differs from background')
            lo, hi = field.pointer, field.pointer + field.byte_span
            if any(lo < end and start < hi for start, end in occupied):
                raise ValueError(f'{name}: increment byte range overlaps retained storage')
            occupied.append((lo, hi))

    def _validate_stream(self, stream):
        # CUDA streams carry a device ordinal. Lightweight test streams may not.
        if getattr(stream, 'device_id', self.device) not in (-1, self.device):
            raise ValueError('stream belongs to a different CUDA device')

    def require_forecast_ready(self):
        if get_ident() != self.thread:
            raise RuntimeError('exchange must be controlled by the forecast thread')
        if self.poisoned:
            raise RuntimeError('consumer failed in-place; restore a known state before forecasting')
        if self.active is not None:
            raise RuntimeError('forecast cannot run while a device lease is outstanding')

    def borrow(self, *, identity, consumer_stream):
        self.require_forecast_ready()
        self._validate_stream(self.producer_stream)
        self._validate_stream(consumer_stream)
        self._validate_storage()
        if not isinstance(identity, StateIdentity) or len(identity.member_ids) != self.batch.members:
            raise ValueError('identity must name every member in storage order')
        fields = {}
        for name, array in self.batch.storage.arrays.items():
            if name.startswith('scratch:'):
                continue
            spec = self.batch.storage.specs[name]
            fields[name] = DeviceField(array, writable=False,
                shared=spec.ownership == 'shared', members=self.batch.members, device=self.device)
        increments = {name: DeviceField(array, writable=True, shared=False,
            members=self.batch.members, device=self.device)
            for name, array in self.increment_arrays.items()}
        try:
            with self.producer_stream:
                for array in self.increment_arrays.values():
                    array.fill(0)
                self.background_digests = (None if self.digest is None else
                    {name: self.digest(field._array) for name, field in fields.items()})
            ready = self.event_factory()
            ready.record(self.producer_stream)
            consumer_stream.wait_event(ready)
        except BaseException:
            self.poisoned = True
            self.producer_stream.synchronize()
            raise
        lease = DeviceLease(self, identity, fields, increments, consumer_stream)
        self.active = lease
        # Retain events until the reverse dependency has completed or a new
        # cycle has established its ordering. CUDA destroys events asynchronously.
        self._events = (ready,)
        return lease

    def _check(self, lease):
        if get_ident() != self.thread or lease is not self.active or lease.closed:
            raise RuntimeError('lease is stale or belongs to another control thread')

    def _complete(self, lease, *, identity, refresh, increment_digests=None):
        self._check(lease)
        try:
            if identity != lease.identity:
                raise ValueError('consumer changed grid, member ordering or valid time')
            if not callable(refresh):
                raise TypeError('writeback requires an engine consistency callback')
            for name, field in lease.fields.items():
                if not field.unchanged(self.batch.storage.arrays[name]):
                    raise RuntimeError(f'{name}: retained allocation was replaced during borrowing')
            for name, field in lease.increments.items():
                if not field.unchanged(self.increment_arrays[name]):
                    raise RuntimeError(f'{name}: increment allocation was replaced during borrowing')
            done = self.event_factory()
            done.record(lease.stream)
            self.producer_stream.wait_event(done)
            with self.producer_stream:
                if self.digest is not None:
                    current = {name: self.digest(field._array) for name, field in lease.fields.items()}
                    if current != self.background_digests:
                        raise ValueError('consumer changed a read-only background field')
                    actual = {name: self.digest(a) for name, a in self.increment_arrays.items()}
                    if actual != increment_digests:
                        raise ValueError('increment digests differ from the consumer completion record')
                receipt = refresh(self.batch, MappingProxyType(self.increment_arrays))
            self._events = self._events + (done,)
        except BaseException:
            self._abort(lease)
            raise
        lease.closed = True
        self.active = None
        self.generation += 1
        return receipt

    def _abort(self, lease):
        self._check(lease)
        # Even if synchronization fails, the allocation owners remain retained
        # by this poisoned exchange. No next forecast can race unfinished work.
        self.poisoned = True
        try:
            lease.stream.synchronize()
            self.producer_stream.synchronize()
        finally:
            lease.closed = True
            self.active = None


def cuda_exchange(batch, *, writable_fields, increment_dtype=None):
    """Bind the current CUDA device/stream without changing array ownership."""
    import cupy as cp
    from gpuwm.ensemble.device_digest import tree_digest
    names = frozenset(writable_fields)
    dtype = None if increment_dtype is None else cp.dtype(increment_dtype)
    if dtype is not None and dtype not in (cp.dtype('float32'), cp.dtype('float64')):
        raise ValueError('increment dtype must be float32 or float64')
    if not names or names - batch.storage.arrays.keys():
        raise ValueError('increment fields must be present in the retained state')
    for name in names:
        if batch.storage.specs[name].ownership == 'shared':
            raise ValueError(f'{name}: shared fields cannot have increments')
        DeviceField(batch.storage.arrays[name], writable=True, shared=False,
                    members=batch.members, device=cp.cuda.runtime.getDevice())
    required = sum(batch.storage.arrays[name].size *
                   (dtype or batch.storage.arrays[name].dtype).itemsize for name in names)
    if required + (512 << 20) > cp.cuda.runtime.memGetInfo()[0]:
        raise MemoryError(f'increment bank requires {required} bytes plus 512 MiB reserve; use a smaller member group')
    increments = {name: cp.zeros_like(batch.storage.arrays[name], dtype=dtype) for name in names}
    return DeviceExchange(batch, producer_stream=cp.cuda.get_current_stream(),
        event_factory=lambda: cp.cuda.Event(disable_timing=True),
        device=cp.cuda.runtime.getDevice(), writable_fields=names,
        increment_arrays=increments, digest=tree_digest)
