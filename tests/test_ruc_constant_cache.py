"""Read-only lookup caching must preserve input edits and stream ordering."""
import numpy as np
import cupy as cp
import pytest

from conftest import requires_gpu
from gpuwm.core.ruc_gpu import _constant_array, _RUC_CONSTANT_CACHE


@pytest.fixture(autouse=True)
def clear_cache():
    _RUC_CONSTANT_CACHE.clear()
    yield
    _RUC_CONSTANT_CACHE.clear()


@requires_gpu
def test_lookup_upload_reuses_only_equal_bytes():
    bits = np.array([0, 0x80000000, 1, 0x7fc12345], np.uint32)
    host = bits.view(np.float32)
    first = _constant_array(host, dtype=np.float32)
    assert first is _constant_array(host.copy(), dtype=np.float32)
    np.testing.assert_array_equal(cp.asnumpy(first).view(np.uint32), bits)
    bits[0] = np.uint32(0x3f800000)
    second = _constant_array(host, dtype=np.float32)
    assert second is not first
    np.testing.assert_array_equal(cp.asnumpy(second).view(np.uint32), bits)
    integers = _constant_array([1, 2, 3], dtype=np.int32)
    np.testing.assert_array_equal(cp.asnumpy(integers), np.array([1, 2, 3], np.int32))


@requires_gpu
def test_table_cache_is_bounded_and_orders_other_streams():
    producer = cp.cuda.Stream(non_blocking=True)
    consumer = cp.cuda.Stream(non_blocking=True)
    host = np.arange(64, dtype=np.float32)
    with producer:
        first = _constant_array(host, dtype=np.float32)
    with consumer:
        second = _constant_array(host, dtype=np.float32)
        copied = second.copy()
    consumer.synchronize()
    assert first is second
    np.testing.assert_array_equal(cp.asnumpy(copied).view(np.uint32), host.view(np.uint32))
    for value in range(40):
        _constant_array([value], dtype=np.int32)
    assert len(_RUC_CONSTANT_CACHE) == 32


@requires_gpu
def test_device_table_keeps_the_original_array_contract():
    table = cp.arange(8, dtype=cp.float32)
    assert _constant_array(table, dtype=np.float32) is table
    table[0] = cp.float32(7)
    assert int(_constant_array(table, dtype=np.float32)[0]) == 7
