"""CPU allocator routing checks, not CUDA or hydrologic numerical proofs."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ensemble.nextgen_points import ReachAccumulator, SCHEMA


@pytest.fixture
def checkpoint(monkeypatch, tmp_path):
    """A decoded checkpoint transport and allocation-only CUDA stand-in."""
    from gpuwm import netcdf_bridge

    class Array(np.ndarray):
        @property
        def device(self):
            return SimpleNamespace(id=3)

    class Allocations:
        active = None
        fail = False

        def __init__(self):
            self.uploads = []

        @contextmanager
        def using_allocator(self, allocator):
            previous, self.active = self.active, allocator
            try:
                yield
            finally:
                self.active = previous

        def asarray(self, value, dtype):
            self.uploads.append(self.active)
            if self.fail and self.active is pool.malloc:
                raise MemoryError("private product pool exhausted")
            return np.asarray(value, dtype=dtype).view(Array)

    allocations = Allocations()
    # A stable callable identity, as supplied by a CuPy pool's allocator.
    pool = SimpleNamespace(malloc=object())
    event = SimpleNamespace(record=lambda: None)
    cp = SimpleNamespace(asarray=allocations.asarray, uint32=np.uint32, int64=np.int64,
        array_equal=np.array_equal,
        cuda=SimpleNamespace(using_allocator=allocations.using_allocator,
            runtime=SimpleNamespace(getDevice=lambda: 3), Event=lambda **kw: event))
    points = ReachAccumulator((4, 7), array_module=cp)
    points.devices[3] = {"pool": pool}
    monkeypatch.setattr(points, "_device", lambda device: points.devices[device])
    shard_calls = []

    def shard(ids):
        shard_calls.append(allocations.active)
        return {"ids": ids.copy(), "count": np.zeros((3, 2), dtype=np.uint32),
                "finite": np.zeros(2, dtype=np.uint32), "done": None}

    monkeypatch.setattr(points, "_shard", shard)
    variables = {}
    for prefix in ("f0_d0", "f0_d1", "f1_d0"):
        variables[prefix + "_ids"] = np.array([2**60 + 3, 9], dtype=np.int64).view(np.uint32)
        variables[prefix + "_count"] = np.ones((3, 2), dtype=np.uint32)
        variables[prefix + "_finite"] = np.ones(2, dtype=np.uint32)

    class Dataset:
        def __init__(self, path):
            self.variables = variables

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_auto_maskandscale(self, value):
            assert value is False

    monkeypatch.setattr(netcdf_bridge, "Dataset", Dataset)
    (tmp_path / "checkpoint.nc").touch()
    document = dict(schema=SCHEMA, members=[4, 7], options=points.options,
        arrays="checkpoint.nc", frames=[
            dict(valid_time="T0", members=[4, 7], shards=["f0_d0", "f0_d1"]),
            dict(valid_time="T1", members=[4], shards=["f1_d0"])])
    return points, document, tmp_path, allocations, pool, shard_calls


def test_restore_uploads_and_shards_use_reserved_pool(checkpoint):
    points, document, root, allocations, pool, shards = checkpoint
    points.restore(root, document)
    assert allocations.uploads == [pool.malloc] * 9
    assert shards == [pool.malloc] * 2
    assert allocations.active is None
    assert points.restored == {("T0", 4), ("T0", 7), ("T1", 4)}
    assert set(points.frames["T0"]["cards"]) == {3}
    target = points.frames["T0"]["cards"][3]
    np.testing.assert_array_equal(target["ids"], [2**60 + 3, 9])
    np.testing.assert_array_equal(target["count"], np.full((3, 2), 2))


def test_restore_does_not_escape_exhausted_pool(checkpoint):
    points, document, root, allocations, pool, shards = checkpoint
    allocations.fail = True
    with pytest.raises(MemoryError, match="private product pool exhausted"):
        points.restore(root, document)
    assert allocations.uploads == [pool.malloc]
    assert not shards and not points.frames
    assert allocations.active is None
