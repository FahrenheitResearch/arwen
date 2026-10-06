"""Owned CUDA scopes for concurrent ordinary ensemble members.

Each scope has one stream and allocator. The ordinary model retains every
physics, nest and clock operation; this scope changes only their CUDA queue
and allocation owner. Stream-local physics caches are retired when the queue
has completed, so a finished member cannot retain another wave's scratch.
"""
from __future__ import annotations

from contextlib import contextmanager
from operator import index
import sys
import gc
from types import ModuleType

from gpuwm.core.cfl_member import member_cfl_scope


_STREAM_LOCAL_RELEASERS = (
    ("gpuwm.core.ruc_fused", "release_ruc_driver_stream_scratch"),
    ("gpuwm.core.ruc_gpu", "release_ruc_stream_scratch"),
    ("gpuwm.core.rrtmgp", "release_rrtmgp_stream_scratch"),
    ("gpuwm.core.mynn_pbl_runtime", "release_mynn_stream_scratch"),
)


def _nonnegative(value, label):
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    value = index(value)
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


class MemberCudaScope:
    """One ordinary member's private queue and measured allocation owner."""

    def __init__(self, *, device_id, member_id, stream, pool):
        self.device_id = device_id
        self.member_id = member_id
        self.stream = stream
        self.pool = pool
        self._release = None

    def receipt(self):
        result = {
            "backend": "ordinary_member_cuda_stream",
            "member_id": self.member_id,
            "device_id": self.device_id,
            "stream_ptr": int(self.stream.ptr),
            "stream_priority": int(self.stream.priority),
            "allocator": "member_owned_memory_pool",
            "model_operation_policy": "unchanged_ordinary_member_operations",
        }
        if self._release is not None:
            result.update(self._release)
        return result


@contextmanager
def member_cuda_scope(*, device_id, member_id, array_module=None):
    """Run a member on an independent stream without global allocator edits.

    CUDA device and stream selection and ``using_allocator`` are scoped to
    the worker thread. The member must finish using its model before leaving
    this scope. Cleanup synchronizes this queue before releasing its cached
    RUC scratch and the unused blocks of this pool. Immutable shared tables
    remain owned by their existing event-ordered caches.
    """
    device_id = _nonnegative(device_id, "device_id")
    member_id = _nonnegative(member_id, "member_id")
    if array_module is None:
        import cupy as array_module
    cuda = array_module.cuda
    with cuda.Device(device_id):
        # CUDA clamps an out-of-range negative priority to the highest
        # supported priority. Product replay owns the lowest priority.
        stream = cuda.Stream(non_blocking=True, priority=-(1 << 31))
        pool = cuda.MemoryPool()
        owned = MemberCudaScope(device_id=device_id, member_id=member_id,
                                stream=stream, pool=pool)
        with member_cfl_scope() as cfl_owner, stream, cuda.using_allocator(pool.malloc):
            failure = None
            try:
                yield owned
            except BaseException as error:
                failure = error
                raise
            finally:
                try:
                    stream.synchronize()
                    before_live, before_reserved = int(pool.used_bytes()), int(pool.total_bytes())
                    # Do not import the GPU physics stack just for an unused
                    # scheme, or let a simulated CUDA backend retire a real
                    # runtime's buffers in CPU ownership checks.
                    retired = {}
                    for module_name, entry in _STREAM_LOCAL_RELEASERS:
                        module = sys.modules.get(module_name)
                        if (isinstance(module, ModuleType)
                                and sys.modules.get("cupy") is not array_module):
                            continue
                        if module is not None:
                            retired[module_name] = getattr(module, entry)(
                                device_id=device_id, stream=stream)
                    for bank in cfl_owner.banks.values():
                        bank.clear()
                    gc.collect()
                    pool.free_all_blocks()
                    owned._release = {
                        "stream_synchronized": True,
                        "pool_live_before_release_bytes": before_live,
                        "pool_reserved_before_release_bytes": before_reserved,
                        "pool_live_after_release_bytes": int(pool.used_bytes()),
                        "pool_reserved_after_release_bytes": int(pool.total_bytes()),
                        "retired_ruc_cache_entries": retired.get("gpuwm.core.ruc_gpu", {}),
                        "retired_physics_cache_entries": retired,
                        "adaptive_cfl_ownership": "independent_member_context",
                    }
                except BaseException as cleanup_error:
                    if failure is None:
                        raise
                    failure.add_note(f"member CUDA cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")


__all__ = ["MemberCudaScope", "member_cuda_scope"]
