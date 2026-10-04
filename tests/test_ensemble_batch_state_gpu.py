"""Real host-to-GPU packing must preserve words and admitted allocation sizes."""
import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("members", [1, 4, 10, 20, 40])
def test_prepared_host_words_pack_into_exact_gpu_backings(members):
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.ensemble.batch_state import (
        BatchedDomainState, PreparedHostMember, SHARED_STATE_CANDIDATES,
        state_array_specs)
    from gpuwm.ensemble.batch_storage import BatchArraySpec, BatchMemoryPlan
    from gpuwm.core.preflight import scratch_slot_registry

    cfg = RunConfig(nx=11, ny=7, nz=5, dx=1000.0, dy=1000.0,
                    ztop=10000.0, dt=3.0, run_seconds=30.0, terrain_opt=1)
    shapes = state_array_shapes(cfg)
    shared = tuple(sorted(SHARED_STATE_CANDIDATES & shapes.keys()))
    rng = np.random.default_rng(3125)
    shared_words = {name: rng.integers(0, 2**32, shape, dtype=np.uint32)
                    for name, shape in shapes.items() if name in shared}
    scratch_shape = scratch_slot_registry(cfg)["rk_ru"]
    prepared = []
    for member in range(members):
        arrays = {}
        for name, shape in shapes.items():
            words = (shared_words[name].copy() if name in shared_words else
                     rng.integers(0, 2**32, shape, dtype=np.uint32))
            words.reshape(-1)[:4] = (0x7fc00001, 0xffc00002, 0x80000000, 0)
            arrays[name] = words.view(np.float32)
        scalars = {"mub": np.float32(90000), "p_top": np.float32(5000),
                   "cf1": np.float32(1), "cf2": np.float32(0), "cf3": np.float32(0),
                   "cfn": np.float32(1), "cfn1": np.float32(0),
                   "has_msf": False, "rotational": False, "elapsed_seconds": 0.0}
        clock = {"ticks": 0, "step_ticks": 3, "tick_den": 1, "run_ticks": 30,
                 "step_count": 0, "dt_fp32": np.float32(3), "dtbc_fp32": np.float32(0)}
        scratch = {"rk_ru": rng.integers(0, 2**32, scratch_shape, dtype=np.uint32).view(np.float32)}
        prepared.append(PreparedHostMember(cfg, arrays, scalars, clock, scratch=scratch))
    specs = state_array_specs(cfg, shared_fields=shared) + (
        BatchArraySpec("scratch:rk_ru", scratch_shape, "member"),)
    plan = BatchMemoryPlan(specs, reserved_bytes=0)
    # Warm the allocator/runtime before observing the admitted live array delta.
    warm = cp.zeros(1, cp.float32)
    del warm
    cp.cuda.get_current_stream().synchronize()
    before = cp.get_default_memory_pool().used_bytes()
    batch = BatchedDomainState.from_prepared(
        prepared, array_module=cp, available_bytes=plan.required_bytes(members),
        shared_fields=shared)
    after = cp.get_default_memory_pool().used_bytes()
    assert after - before == plan.required_bytes(members)
    assert batch.storage.payload_bytes == sum(r["payload_bytes"] for r in plan.inventory(members))
    for name in shapes:
        for member in range(members):
            observed = cp.asnumpy(batch.member_view(name, member)).view(np.uint32).tobytes()
            expected = prepared[member].arrays[name].view(np.uint32).tobytes()
            assert observed == expected, (name, member)
    for member in range(members):
        assert cp.asnumpy(batch.scratch_member_view("rk_ru", member)).view(np.uint32).tobytes() == (
            prepared[member].scratch["rk_ru"].view(np.uint32).tobytes())
