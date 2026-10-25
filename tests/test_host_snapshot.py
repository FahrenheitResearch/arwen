"""Checkpoint snapshots own deferred data and finish even exceptional copies."""
import numpy as np
import pytest

from gpuwm.io.host_snapshot import snapshot_arrays


def test_deferred_host_snapshot_preserves_raw_words_after_source_changes():
    words = np.array([0, 0x80000000, 0x7fc12345, 0x3f800000], dtype=np.uint32)
    source = words.view(np.float32)
    snapshot = snapshot_arrays({'field': source}, owned=True)
    source.fill(2)
    assert snapshot['field'].view(np.uint32).tolist() == [
        0, 0x80000000, 0x7fc12345, 0x3f800000]


@pytest.mark.gpu
def test_device_snapshot_orders_stream_and_owns_strided_and_empty_fields():
    import cupy as cp

    with cp.cuda.Stream(non_blocking=True):
        source = cp.arange(48, dtype=cp.float64).reshape(6, 8)
        source *= 3
        fields = {'full': source, 'strided': source[:, ::2],
                  'empty': cp.empty((0,), dtype=cp.float32)}
        expected = {key: value.get().tobytes() for key, value in fields.items()}
        captured = snapshot_arrays(fields, owned=True)
        source.fill(-1)
        assert {key: value.tobytes() for key, value in captured.items()} == expected


@pytest.mark.gpu
def test_pinned_allocation_failure_warns_and_preserves_every_field(monkeypatch):
    import cupy as cp

    source = cp.arange(32, dtype=cp.int64)
    def exhausted(size):
        raise MemoryError('injected pinned exhaustion')
    monkeypatch.setattr(cp.cuda, 'alloc_pinned_memory', exhausted)
    with pytest.warns(RuntimeWarning, match='complete pageable copies'):
        captured = snapshot_arrays({'first': source, 'second': source + 1})
    assert np.array_equal(captured['first'], np.arange(32, dtype=np.int64))
    assert np.array_equal(captured['second'], np.arange(32, dtype=np.int64) + 1)


@pytest.mark.gpu
def test_later_field_failure_drains_transfers_before_snapshot_unwinds():
    import cupy as cp

    class FailedField:
        def get(self):
            raise RuntimeError('later field failed')
    with cp.cuda.Stream(non_blocking=True) as stream:
        source = cp.arange(100, dtype=cp.float64)
        with pytest.raises(RuntimeError, match='later field failed'):
            snapshot_arrays({'a': source, 'z': FailedField()})
        assert stream.done
