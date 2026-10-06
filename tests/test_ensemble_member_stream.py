"""Concurrent ordinary member scopes retain their own CUDA owners."""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, local
from types import ModuleType, SimpleNamespace

import pytest

from gpuwm.ensemble.member_stream import member_cuda_scope


def _fake_cuda():
    current = local()
    made = {"streams": [], "pools": []}

    class Device:
        def __init__(self, device_id):
            self.device_id = device_id

        def __enter__(self):
            self.before = getattr(current, "device", None)
            current.device = self.device_id
            return self

        def __exit__(self, *unused):
            current.device = self.before

    class Stream:
        def __init__(self, *, non_blocking, priority):
            assert non_blocking
            assert priority == -(1 << 31)
            self.priority = -2
            self.ptr = id(self)
            self.synchronized = False
            made["streams"].append(self)

        def __enter__(self):
            self.before = getattr(current, "stream", None)
            current.stream = self
            return self

        def __exit__(self, *unused):
            current.stream = self.before

        def synchronize(self):
            assert current.device is not None
            self.synchronized = True

    class MemoryPool:
        def __init__(self):
            self.reserved = 512
            self.freed = False
            made["pools"].append(self)

        def malloc(self, *unused):
            pass

        def used_bytes(self):
            return 32

        def total_bytes(self):
            return self.reserved

        def free_all_blocks(self):
            assert current.stream.synchronized
            self.freed = True
            self.reserved = 32

    @contextmanager
    def using_allocator(allocator):
        before = getattr(current, "allocator", None)
        current.allocator = allocator
        try:
            yield
        finally:
            current.allocator = before

    return SimpleNamespace(cuda=SimpleNamespace(Device=Device, Stream=Stream,
        MemoryPool=MemoryPool, using_allocator=using_allocator)), current, made


def test_simultaneous_workers_use_distinct_streams_and_restore_their_scopes(monkeypatch):
    # No GPU module is imported by the ordinary scope when RUC is unused.
    monkeypatch.delitem(__import__("sys").modules, "gpuwm.core.ruc_gpu", raising=False)
    module, current, made = _fake_cuda()
    both_running = Barrier(2)

    def worker(member):
        with member_cuda_scope(device_id=0, member_id=member, array_module=module) as scope:
            both_running.wait(timeout=5)
            assert current.device == 0
            assert current.stream is scope.stream
            assert current.allocator.__self__ is scope.pool
        assert current.device is None
        assert current.stream is None
        assert current.allocator is None
        return scope.receipt()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(worker, (2, 5)))
    assert results[0]["stream_ptr"] != results[1]["stream_ptr"]
    assert [result["member_id"] for result in results] == [2, 5]
    assert all(result["stream_synchronized"] for result in results)
    assert all(result["stream_priority"] == -2 for result in results)
    assert all(pool.freed for pool in made["pools"])
    assert all(result["pool_reserved_after_release_bytes"] == 32 for result in results)


def test_failure_finishes_its_queue_and_retires_only_its_scratch(monkeypatch):
    module, current, made = _fake_cuda()
    released = []

    def release(*, device_id, stream):
        assert stream.synchronized
        assert current.stream is stream
        released.append((device_id, stream.ptr))
        return {"scratch": 3}

    monkeypatch.setitem(__import__("sys").modules, "gpuwm.core.ruc_gpu",
                        SimpleNamespace(release_ruc_stream_scratch=release))
    with pytest.raises(RuntimeError, match="member failure"):
        with member_cuda_scope(device_id=1, member_id=4, array_module=module) as scope:
            raise RuntimeError("member failure")
    assert released == [(1, scope.stream.ptr)]
    assert made["pools"][0].freed
    assert scope.receipt()["retired_ruc_cache_entries"] == {"scratch": 3}


def test_simulated_scope_does_not_cleanup_an_already_imported_real_runtime(monkeypatch):
    module, _, made = _fake_cuda()
    imported = ModuleType("gpuwm.core.mynn_pbl_runtime")
    imported.release_mynn_stream_scratch = lambda **unused: pytest.fail("real runtime touched")
    monkeypatch.setitem(__import__("sys").modules, "gpuwm.core.mynn_pbl_runtime", imported)
    with member_cuda_scope(device_id=0, member_id=0, array_module=module) as scope:
        pass
    assert made["pools"][0].freed
    assert "gpuwm.core.mynn_pbl_runtime" not in scope.receipt()["retired_physics_cache_entries"]


def test_cleanup_failure_retains_the_original_member_error(monkeypatch):
    module, _, _ = _fake_cuda()
    def failed_cleanup(**unused):
        raise ValueError("cleanup failure")
    monkeypatch.setitem(__import__("sys").modules, "gpuwm.core.ruc_gpu",
                        SimpleNamespace(release_ruc_stream_scratch=failed_cleanup))
    with pytest.raises(RuntimeError, match="forecast failure") as failure:
        with member_cuda_scope(device_id=0, member_id=0, array_module=module):
            raise RuntimeError("forecast failure")
    assert any("cleanup failure" in note for note in failure.value.__notes__)


def _release_function():
    # This host-only ownership check executes the actual cleanup function
    # without importing a CUDA runtime on CPU test hosts.
    path = Path(__file__).resolve().parents[1] / "gpuwm/core/ruc_gpu.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "release_ruc_stream_scratch")
    namespace = {
        "cp": SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: 0))),
        "_SFCTMP_SCRATCH": {(0, 11, 20, 6): object(), (0, 22, 20, 6): object(), (1, 11, 20, 6): object()},
        "_SFCTMP_TABLE_CACHE": {(0, 11, "table"): object(), (0, 22, "table"): object()},
        "_SFCTMP_UPLOADS": {(0, 11): object(), (0, 22): object()},
        "_SFCTMP_FLAG_CONTEXT": {(0, 101): object(), (0, 102): object(), (1, 101): object()},
        "_SFCTMP_CONTEXT_STREAMS": {(0, 101): 11, (0, 102): 22, (1, 101): 11},
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_ruc_retirement_preserves_other_active_stream_and_card():
    namespace = _release_function()
    sync = []
    stream = SimpleNamespace(ptr=11, synchronize=lambda: sync.append(True))
    result = namespace["release_ruc_stream_scratch"](device_id=0, stream=stream)
    assert sync == [True]
    assert result == {"scratch": 1, "tables": 1, "uploads": 1, "flag_contexts": 1}
    assert set(namespace["_SFCTMP_SCRATCH"]) == {(0, 22, 20, 6), (1, 11, 20, 6)}
    assert set(namespace["_SFCTMP_FLAG_CONTEXT"]) == {(0, 102), (1, 101)}
    assert namespace["release_ruc_stream_scratch"](device_id=0, stream=stream) == {
        "scratch": 0, "tables": 0, "uploads": 0, "flag_contexts": 0}


def test_ruc_retirement_requires_the_selected_card():
    namespace = _release_function()
    stream = SimpleNamespace(ptr=11, synchronize=lambda: pytest.fail("wrong device synchronized"))
    with pytest.raises(ValueError, match="owning CUDA device"):
        namespace["release_ruc_stream_scratch"](device_id=1, stream=stream)
    assert len(namespace["_SFCTMP_SCRATCH"]) == 3
