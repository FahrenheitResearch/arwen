"""Periodic face closure must copy exact words with no CUDA temporaries."""
from dataclasses import replace
import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("count", (1, 4, 10, 20, 40))
@pytest.mark.parametrize("boundary", ("periodic", "open_x", "open_y", "specified", "nested"))
def test_alias_words_and_allocation_requests(count, boundary):
    import cupy as cp
    from gpuwm.core.dycore import close_periodic_alias
    from gpuwm.ensemble.batch_glue import prepare_periodic_alias
    from gpuwm.ensemble.batch_state import BatchedDomainState
    from test_ensemble_batch_dycore_gpu import _prepared
    from test_ensemble_batch_bigstep_gpu import _scalar_state
    inputs, specs, slots = _prepared(count, terrain=False, mapped=False, order=5, emdiv=0)
    words = np.array([0, 0x80000000, 1, 0x80000001, 0x7F800000,
                      0xFF800000, 0x7FC01234, 0x7FA05678], np.uint32)
    prepared = []
    for index, member in enumerate(inputs):
        cfg = replace(member.cfg, open_x=boundary == "open_x", open_y=boundary == "open_y",
                      specified=boundary == "specified", nested=boundary == "nested")
        arrays = dict(member.arrays)
        for name in ("u", "v"):
            arrays[name] = member.arrays[name].copy()
            arrays[name].view(np.uint32)[...] = np.resize(np.roll(words, index), arrays[name].size).reshape(arrays[name].shape)
        prepared.append(replace(member, cfg=cfg, arrays=arrays))
    batch = BatchedDomainState.from_prepared(prepared, array_module=cp, available_bytes=2**30,
                                             extra_specs=specs, scratch_slots=slots)
    launch = prepare_periodic_alias(batch)
    launch()
    allocated = []
    allocator = cp.cuda.get_allocator()
    def track(size):
        allocated.append(int(size))
        return allocator(size)
    with cp.cuda.using_allocator(track):
        launch()
    cp.cuda.get_current_stream().synchronize()
    assert not allocated, allocated
    for index, member in enumerate(prepared):
        scalar = _scalar_state(member)
        close_periodic_alias(scalar, member.cfg)
        for name in ("u", "v"):
            assert cp.asnumpy(batch.member_view(name, index)).view(np.uint32).tobytes() == cp.asnumpy(getattr(scalar, name)).view(np.uint32).tobytes()
