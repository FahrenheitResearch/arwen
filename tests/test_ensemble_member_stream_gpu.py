"""The real concurrent member scopes own separate fused RUC working banks."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from gpuwm.core import ruc_gpu
from gpuwm.ensemble.member_stream import member_cuda_scope

pytestmark = pytest.mark.gpu


def test_member_streams_keep_ruc_scratch_separate_and_release_it():
    barrier = Barrier(2)
    def worker(member):
        with member_cuda_scope(device_id=0, member_id=member, array_module=cp) as scope:
            scratch = ruc_gpu._sfctmp_scratch(16, 6)[0]
            scratch.fill(np.float32(member + 1))
            pointer = int(scratch.data.ptr)
            barrier.wait(timeout=30)
            assert bool(cp.all(scratch == np.float32(member + 1)))
            stream_key = (0, int(scope.stream.ptr), 16, 6)
            assert stream_key in ruc_gpu._SFCTMP_SCRATCH
            del scratch
        assert stream_key not in ruc_gpu._SFCTMP_SCRATCH
        return pointer, scope.receipt()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(worker, (0, 1)))
    assert results[0][0] != results[1][0]
    assert all(receipt["retired_ruc_cache_entries"]["scratch"] == 1 for _, receipt in results)
    assert all(receipt["pool_live_after_release_bytes"] == 0 for _, receipt in results)


def test_concurrent_member_cfl_words_equal_standalone_same_grid_reductions():
    from types import SimpleNamespace
    from gpuwm.config import RunConfig
    from gpuwm.core import dycore
    from gpuwm.core.cfl_member import current_cfl_member
    cfg = RunConfig(nx=8, ny=6, nz=4, dx=1000., dy=1000., ztop=10000.,
                    dt=1., run_seconds=10.)
    def reduce(member):
        zero = lambda shape: cp.zeros(shape, dtype=cp.float32)
        state = SimpleNamespace(mup=zero((6, 8)), mub2d=cp.ones((6, 8), dtype=cp.float32),
            c1f=cp.ones(5, dtype=cp.float32), c2f=zero(5), rdnw=cp.ones(4, dtype=cp.float32),
            u=cp.full((4, 6, 9), member + 1, dtype=cp.float32), v=zero((4, 7, 8)),
            msfu=cp.ones((6, 9), dtype=cp.float32), msfv=cp.ones((7, 8), dtype=cp.float32))
        ww = cp.full((5, 6, 8), (member + 1) * .125, dtype=cp.float32)
        for _ in range(3):
            dycore.record_wrf_vertical_cfl(state, cfg, ww)
        return dycore.take_wrf_cfl(1), dycore._wrf_cfl_bank("_WRF_CFL_STAT")[1][0].get().tobytes()

    reference = []
    for member in range(2):
        dycore.reset_wrf_cfl_recording()
        dycore.enable_wrf_cfl_recording()
        reference.append(reduce(member))
        dycore.reset_wrf_cfl_recording()
    assert reference[0] != reference[1]
    barrier = Barrier(2)
    def worker(member):
        with member_cuda_scope(device_id=0, member_id=member, array_module=cp):
            dycore.enable_wrf_cfl_recording()
            barrier.wait(timeout=30)
            result = reduce(member)
            barrier.wait(timeout=30)
            assert current_cfl_member().enabled
            dycore.reset_wrf_cfl_recording()
            return result
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = list(executor.map(worker, (0, 1)))
    assert actual == reference
