"""CPU checks of storage ownership failures; actual page movement is GPU-qualified."""
from contextlib import contextmanager
from types import SimpleNamespace
import sys

import pytest

from gpuwm.ensemble import managed_storage as storage


@pytest.fixture
def owner(monkeypatch):
    events = []
    device = SimpleNamespace(id=0)
    stream = SimpleNamespace(ptr=3, synchronize=lambda: events.append('stream-sync'))

    class Pool:
        def __init__(self, allocator):
            self.malloc = allocator
        def free_all_blocks(self):
            events.append('trim')
        def used_bytes(self):
            return 0
        def total_bytes(self):
            return 0

    @contextmanager
    def using_allocator(allocator):
        events.append('allocator-enter')
        try:
            yield
        finally:
            events.append('allocator-exit')

    class Prefetch:
        status = 0
        def __call__(self, *args):
            events.append(('prefetch', args))
            return self.status

    prefetch = Prefetch()
    fake = SimpleNamespace(cuda=SimpleNamespace(
        Device=lambda: device, MemoryPool=Pool, using_allocator=using_allocator,
        get_current_stream=lambda: stream,
        runtime=SimpleNamespace(getDeviceProperties=lambda _: dict(
            managedMemory=True, concurrentManagedAccess=True),
            deviceSynchronize=lambda: events.append('device-sync'),
            memGetInfo=lambda: (1000, 2000))))
    monkeypatch.setitem(sys.modules, 'cupy', fake)
    loader = 'WinDLL' if sys.platform == 'win32' else 'CDLL'
    monkeypatch.setattr(storage.ctypes, loader,
                        lambda _: SimpleNamespace(cuMemPrefetchAsync=prefetch))
    result = storage.ManagedStorage()
    result.test_events = events
    result.test_device = device
    result.test_prefetch = prefetch
    return result


def test_scope_restores_allocator_after_exception(owner):
    with pytest.raises(ValueError):
        with owner.allocate():
            raise ValueError('caller failed')
    assert not owner.active
    assert owner.test_events == ['allocator-enter', 'allocator-exit']
    with owner.allocate():
        with pytest.raises(RuntimeError, match='cannot nest'):
            with owner.allocate():
                pass


def test_migration_requires_idle_and_park_blocks_new_allocations(owner):
    with owner.allocate():
        with pytest.raises(RuntimeError, match='idle'):
            owner.park()
    assert owner.phase == 'device'
    owner.park()
    with pytest.raises(RuntimeError, match='resumed'):
        with owner.allocate():
            pass
    owner.resume()
    with owner.allocate():
        pass


def test_migration_orders_sync_trim_prefetch_and_completion(owner):
    allocation = storage._Allocation(SimpleNamespace(ptr=4096, mem=SimpleNamespace(size=128)))
    owner._live[4096] = allocation
    row = owner.park()
    assert owner.test_events == ['device-sync', 'trim',
                                 ('prefetch', (4096, 128, -1, 3)), 'stream-sync']
    assert row['bytes'] == 128 and row['allocations'] == 1


def test_failed_migration_poison_prevents_later_use(owner):
    allocation = storage._Allocation(SimpleNamespace(ptr=4096, mem=SimpleNamespace(size=128)))
    owner._live[4096] = allocation
    owner.test_prefetch.status = 101
    with pytest.raises(RuntimeError, match='status 101'):
        owner.park()
    with pytest.raises(RuntimeError, match='unusable'):
        owner.resume()
    with pytest.raises(RuntimeError, match='unusable'):
        owner.inventory()


def test_control_thread_and_device_are_fixed(owner, monkeypatch):
    monkeypatch.setattr(storage, 'get_ident', lambda: owner.thread + 1)
    with pytest.raises(RuntimeError, match='control thread'):
        owner.park()
    monkeypatch.setattr(storage, 'get_ident', lambda: owner.thread)
    owner.test_device.id = 1
    with pytest.raises(RuntimeError, match='device changed'):
        owner.resume()


def test_array_span_checks_negative_views_and_foreign_allocations(owner):
    class Array:
        def __init__(self, pointer, strides):
            self.data = SimpleNamespace(ptr=pointer)
            self.shape, self.strides, self.size = (4,), strides, 4
            self.dtype = SimpleNamespace(itemsize=4)
            self.device = SimpleNamespace(id=0)
    owner.cp.ndarray = Array
    allocation = storage._Allocation(SimpleNamespace(ptr=4096, mem=SimpleNamespace(size=16)))
    owner._live[4096] = allocation
    assert owner.check_arrays([Array(4096, (4,)), Array(4108, (-4,))]) == 2
    with pytest.raises(ValueError, match='outside'):
        owner.check_arrays([Array(4096, (-4,))])
