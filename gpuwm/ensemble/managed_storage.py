# SPDX-License-Identifier: Apache-2.0
"""Keep array ownership intact while moving managed pages between CPU and GPU.

This allocator must surround construction and later allocations of one owner.
It does not rewrite arrays, callbacks, views, clocks or physical state. Migration
is an explicit, synchronous control-thread operation at an idle boundary. A
caller must keep all users idle until migration has finished and must resume the
owner before using its arrays. Ordinary external device allocations are outside
this pool and must be accounted for separately.
"""
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
from threading import get_ident
from time import perf_counter
from weakref import WeakValueDictionary
import sys


@dataclass(eq=False)
class _Allocation:
    memory: object

    @property
    def pointer(self):
        return int(self.memory.ptr)

    @property
    def size(self):
        return int(self.memory.mem.size)


class ManagedStorage:
    """A per-owner managed pool with bounded bookkeeping and explicit migration.

    References held by arrays and the pool own the backing allocation. The
    registry is weak, so it cannot retain dead temporaries after pool trimming.
    CUDA device synchronization includes kernels on streams outside the current
    stream. It cannot prevent a different CPU thread from submitting more work;
    exclusive owner control is a required caller invariant.
    """

    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.device = int(cp.cuda.Device().id)
        properties = cp.cuda.runtime.getDeviceProperties(self.device)
        if not properties.get('managedMemory') or not properties.get('concurrentManagedAccess'):
            raise RuntimeError('managed storage requires concurrent managed-memory support')
        self.thread = get_ident()
        self.phase = 'device'
        self.active = False
        self._live = WeakValueDictionary()
        self.pool = cp.cuda.MemoryPool(self._allocate_backing)
        self.migration_count = 0
        self.requested_prefetch_bytes = 0
        self.last_migration = None
        # The CUDA 13 branch of CuPy 14.2's runtime.memPrefetchAsync marks
        # every destination as a device, including CPU (-1). Bind the stable
        # driver entry point with its explicit CUdevice argument instead.
        # CUDA retains this symbol for ABI compatibility (Driver API 12.8,
        # Unified Addressing, cuMemPrefetchAsync). Do not bind the v2 symbol
        # with this signature: v2 takes a location struct and an extra flag.
        self._driver = (ctypes.WinDLL('nvcuda.dll') if sys.platform == 'win32'
                        else ctypes.CDLL('libcuda.so.1'))
        self._prefetch = self._driver.cuMemPrefetchAsync
        self._prefetch.argtypes = [ctypes.c_uint64, ctypes.c_size_t,
                                   ctypes.c_int, ctypes.c_void_p]
        self._prefetch.restype = ctypes.c_int

    def _check(self):
        if get_ident() != self.thread:
            raise RuntimeError('managed storage belongs to its creating control thread')
        if int(self.cp.cuda.Device().id) != self.device:
            raise RuntimeError('managed storage device changed')
        if self.phase == 'failed':
            raise RuntimeError('managed storage is unusable after failed migration')

    def _allocate_backing(self, size):
        self._check()
        if not self.active or self.phase != 'device':
            raise RuntimeError('managed allocation requires an active owner allocation context')
        memory = self.cp.cuda.malloc_managed(size)
        if not size:
            return memory
        owner = _Allocation(memory)
        self._live[owner.pointer] = owner
        # A Python owner supports weak references across CuPy versions, whereas
        # CuPy's Cython ManagedMemory need not support them. UnownedMemory owns
        # that Python object, which in turn owns the actual managed allocation.
        wrapped = self.cp.cuda.UnownedMemory(owner.pointer, owner.size, owner,
                                             device_id=self.device)
        return self.cp.cuda.MemoryPointer(wrapped, 0)

    @contextmanager
    def allocate(self):
        """Route this thread's new arrays to this pool while the owner is active."""
        self._check()
        if self.phase != 'device' or self.active:
            raise RuntimeError('owner must be resumed and allocation contexts cannot nest')
        self.active = True
        try:
            with self.cp.cuda.using_allocator(self.pool.malloc):
                yield self
        finally:
            self.active = False

    def inventory(self):
        self._check()
        return dict(phase=self.phase, device=self.device,
                    allocations=len(self._live),
                    backing_bytes=sum(item.size for item in self._live.values()),
                    pool_used_bytes=int(self.pool.used_bytes()),
                    pool_total_bytes=int(self.pool.total_bytes()),
                    migration_count=self.migration_count,
                    requested_prefetch_bytes=self.requested_prefetch_bytes)

    def check_arrays(self, arrays):
        """Require each supplied array span to belong to this owner's pool.

        This checks the supplied inventory only. It is not a recursive model
        audit and makes no claim about external library workspaces.
        """
        self._check()
        ranges = [(a.pointer, a.pointer + a.size) for a in self._live.values()]
        count = 0
        for array in arrays:
            if not isinstance(array, self.cp.ndarray) or int(array.device.id) != self.device:
                raise ValueError('owner inventory must contain arrays on its device')
            if not array.size:
                continue
            lo = hi = int(array.data.ptr)
            for extent, stride in zip(array.shape, array.strides):
                delta = (extent - 1) * stride
                lo += min(delta, 0)
                hi += max(delta, 0)
            hi += array.dtype.itemsize
            if not any(start <= lo and hi <= end for start, end in ranges):
                raise ValueError('array allocation is outside this managed owner')
            count += 1
        return count

    def _migrate(self, destination):
        self._check()
        if self.active:
            raise RuntimeError('migration requires an idle allocation context')
        started = perf_counter()
        try:
            self.cp.cuda.runtime.deviceSynchronize()
            self.pool.free_all_blocks()
            before = int(self.cp.cuda.runtime.memGetInfo()[0])
            stream = self.cp.cuda.get_current_stream()
            allocations = list(self._live.values())
            for allocation in allocations:
                status = self._prefetch(
                    allocation.pointer, allocation.size, destination, stream.ptr)
                if status:
                    raise RuntimeError(f'cuMemPrefetchAsync failed with CUDA driver status {status}')
            stream.synchronize()
            self.phase = 'host' if destination == -1 else 'device'
            row = dict(destination=self.phase, bytes=sum(a.size for a in allocations),
                       api='CUDA driver cuMemPrefetchAsync (CUdevice ABI)',
                       allocations=len(allocations), seconds=perf_counter() - started,
                       device_free_before=before,
                       device_free_after=int(self.cp.cuda.runtime.memGetInfo()[0]))
            self.migration_count += 1
            self.requested_prefetch_bytes += row['bytes']
            self.last_migration = row
            return row
        except BaseException:
            self.phase = 'failed'
            raise

    def park(self):
        """Synchronize, release unused pool blocks, and prefetch live pages to CPU."""
        return self._migrate(-1)

    def resume(self):
        """Synchronize and prefetch this owner's live allocations to its GPU."""
        return self._migrate(self.device)
