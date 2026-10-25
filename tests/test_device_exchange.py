"""Prevent use-after-release, cross-member writes and stream races (CPU ABI tests)."""
from types import SimpleNamespace
import numpy as np
import pytest

from gpuwm.ensemble.device_exchange import DeviceExchange, StateIdentity


class Array:
    def __init__(self, pointer, shape, device=0):
        self.__cuda_array_interface__ = dict(version=3, shape=shape, typestr='<f4',
            data=(pointer, False), strides=None)
        self.shape = shape
        self.dtype = np.dtype('float32')
        self.device = SimpleNamespace(id=device)


    def fill(self, value):
        self.last_fill = value


class Stream:
    def __init__(self, log, name):
        self.log, self.name = log, name
    def wait_event(self, event):
        self.log.append(('wait', self.name, event.number))
    def synchronize(self):
        self.log.append(('sync', self.name))
    def __enter__(self):
        self.log.append(('enter', self.name))
    def __exit__(self, *args):
        self.log.append(('exit', self.name))


def fixture_exchange():
    log = []
    producer, consumer = Stream(log, 'forecast'), Stream(log, 'analysis')
    events = []
    class Event:
        def __init__(self):
            self.number = len(events)
            events.append(self)
        def record(self, stream):
            log.append(('record', stream.name, self.number))
    arrays = {'u': Array(4096, (2, 3, 4, 6)), 'ht': Array(8192, (4, 5))}
    specs = {'u': SimpleNamespace(ownership='member'), 'ht': SimpleNamespace(ownership='shared')}
    batch = SimpleNamespace(members=2, storage=SimpleNamespace(arrays=arrays, specs=specs))
    ex = DeviceExchange(batch, producer_stream=producer, event_factory=Event,
                        device=0, writable_fields=('u',),
                        increment_arrays={'u': Array(16384, (2, 3, 4, 6))})
    identity = StateIdentity('grid', (7, 12), 90, 1)
    return ex, consumer, identity, log


def test_borrow_keeps_native_shape_pointer_and_event_order():
    ex, consumer, identity, log = fixture_exchange()
    lease = ex.borrow(identity=identity, consumer_stream=consumer)
    assert lease.fields['u'].pointer == 4096
    assert lease.fields['u'].shape == (2, 3, 4, 6)
    assert lease.fields['u'].strides == (288, 96, 24, 4)
    assert not lease.fields['ht'].writable
    assert not lease.fields['u'].writable
    assert lease.increments['u'].writable
    assert lease.increments['u'].pointer == 16384
    assert log == [('enter', 'forecast'), ('exit', 'forecast'),
                   ('record', 'forecast', 0), ('wait', 'analysis', 0)]
    result = lease.complete(identity=identity, refresh=lambda batch, increments: log.append(('refresh',)) or 'ok')
    assert result == 'ok'
    assert log[4:] == [('record', 'analysis', 1), ('wait', 'forecast', 1),
                       ('enter', 'forecast'), ('refresh',), ('exit', 'forecast')]
    ex.require_forecast_ready()
    assert ex.generation == 1
    with pytest.raises(RuntimeError, match='stale'):
        lease.complete(identity=identity, refresh=lambda b, increments: None)


def test_fp64_increment_descriptor_preserves_fp32_background():
    ex, stream, identity, _ = fixture_exchange()
    increment = ex.increment_arrays['u']
    increment.dtype = np.dtype('float64')
    increment.__cuda_array_interface__['typestr'] = '<f8'
    lease = ex.borrow(identity=identity, consumer_stream=stream)
    assert lease.fields['u'].dtype == '<f4'
    assert lease.increments['u'].dtype == '<f8'
    assert lease.increments['u'].strides == (576,192,48,8)
    assert lease.increments['u'].byte_span == 1152
    lease.complete(identity=identity, refresh=lambda *args: None)


def test_fp64_bank_admission_counts_output_precision_before_allocating(monkeypatch):
    import sys
    from gpuwm.ensemble.device_exchange import cuda_exchange
    ex, _, _, _ = fixture_exchange()
    ex.batch.storage.arrays['u'].size = 144
    calls = []
    cp = SimpleNamespace(dtype=np.dtype,
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: 0,
            memGetInfo=lambda: ((512 << 20)+576, 1 << 30))),
        zeros_like=lambda *args, **kwargs: calls.append('allocated'))
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    with pytest.raises(MemoryError, match='requires 1152 bytes'):
        cuda_exchange(ex.batch, writable_fields=('u',), increment_dtype='float64')
    assert not calls
    with pytest.raises(ValueError, match='float32 or float64'):
        cuda_exchange(ex.batch, writable_fields=('u',), increment_dtype='int32')
    assert not calls


def test_no_forecast_or_second_borrow_while_consumer_owns_state():
    ex, stream, identity, _ = fixture_exchange()
    ex.borrow(identity=identity, consumer_stream=stream)
    with pytest.raises(RuntimeError, match='outstanding'):
        ex.require_forecast_ready()
    with pytest.raises(RuntimeError, match='outstanding'):
        ex.borrow(identity=identity, consumer_stream=stream)


@pytest.mark.parametrize('failure', ['identity', 'replacement', 'refresh', 'abort'])
def test_partial_failure_poison_prevents_next_forecast(failure):
    ex, stream, identity, log = fixture_exchange()
    lease = ex.borrow(identity=identity, consumer_stream=stream)
    def refresh(batch, increments):
        if failure == 'refresh':
            raise RuntimeError('incomplete scientific refresh')
    if failure == 'replacement':
        ex.batch.storage.arrays['u'] = Array(12288, (2, 3, 4, 6))
    if failure == 'abort':
        lease.abort()
    else:
        with pytest.raises((RuntimeError, ValueError)):
            lease.complete(identity=StateIdentity('wrong', (7, 12), 90, 1)
                           if failure == 'identity' else identity, refresh=refresh)
    assert lease.closed and ex.poisoned
    assert ('sync', 'analysis') in log
    with pytest.raises(RuntimeError, match='restore a known state'):
        ex.require_forecast_ready()


def test_device_mismatch_refused_before_event_submission():
    ex, stream, identity, log = fixture_exchange()
    ex.batch.storage.arrays['u'].device.id = 1
    with pytest.raises(ValueError, match='same CUDA device'):
        ex.borrow(identity=identity, consumer_stream=stream)
    assert not log


def test_shared_mutation_refused():
    ex, stream, identity, log = fixture_exchange()
    from gpuwm.ensemble.device_exchange import DeviceField
    with pytest.raises(ValueError, match='shared or read-only'):
        DeviceField(ex.batch.storage.arrays['ht'], writable=True, shared=True, members=2, device=0)
    assert not log


@pytest.mark.parametrize('ids', [(1, 1), (), (-1, 4)])
def test_member_identity_rejects_ambiguous_storage_order(ids):
    with pytest.raises(ValueError):
        StateIdentity('grid', ids, 90, 1)


@pytest.mark.parametrize('mutation', ['strides', 'readonly', 'device'])
def test_descriptor_change_cannot_resume_forecast(mutation):
    ex, stream, identity, _ = fixture_exchange()
    lease = ex.borrow(identity=identity, consumer_stream=stream)
    arr = ex.batch.storage.arrays['u']
    if mutation == 'strides':
        arr.__cuda_array_interface__['strides'] = (576, 192, 48, 8)
    elif mutation == 'readonly':
        arr = ex.increment_arrays['u']
        arr.__cuda_array_interface__['data'] = (16384, True)
    else:
        arr.device.id = 1
    with pytest.raises((RuntimeError, ValueError)):
        lease.complete(identity=identity, refresh=lambda b, increments: None)
    assert ex.poisoned


def test_overlapping_member_slabs_refused():
    ex, stream, identity, _ = fixture_exchange()
    ex.batch.storage.arrays['u'].__cuda_array_interface__['strides'] = (4, 96, 24, 4)
    with pytest.raises(ValueError, match='overlapping'):
        ex.borrow(identity=identity, consumer_stream=stream)


def test_increment_subview_of_background_refused_before_zeroing():
    ex, stream, identity, log = fixture_exchange()
    ex.increment_arrays['u'] = Array(4100, (2, 3, 4, 6))
    with pytest.raises(ValueError, match='byte range overlaps'):
        ex.borrow(identity=identity, consumer_stream=stream)
    assert not log and not hasattr(ex.increment_arrays['u'], 'last_fill')


def test_wrong_stream_device_refused_before_work():
    ex, stream, identity, log = fixture_exchange()
    stream.device_id = 1
    with pytest.raises(ValueError, match='different CUDA device'):
        ex.borrow(identity=identity, consumer_stream=stream)
    assert not log


def test_publish_failure_poisoned_and_synchronised():
    ex, stream, identity, log = fixture_exchange()
    def failed_digest(array):
        raise RuntimeError('injected digest launch failure')
    ex.digest = failed_digest
    with pytest.raises(RuntimeError, match='injected'):
        ex.borrow(identity=identity, consumer_stream=stream)
    assert ex.poisoned and ('sync', 'forecast') in log
    with pytest.raises(RuntimeError, match='restore a known state'):
        ex.require_forecast_ready()
