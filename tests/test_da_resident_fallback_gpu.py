"""Actual CUDA residency and allocator-failure recovery on small columns."""
from dataclasses import replace
import weakref

import cupy as cp
import numpy as np
import pytest

from gpuwm.da import radar_assimilation as owner
from gpuwm.da.static_covariance import static_analysis
from test_letkf_bounded_staging import problem

pytestmark = pytest.mark.gpu


def test_resident_default_matches_existing_bounded_cuda_transform():
    prior, obs, grid, cfg = problem(fields=14, batches=20, members=9)
    cfg = replace(cfg, eigensolver='auto')
    reference = static_analysis(prior, obs, grid, cfg, solve_namespace=cp)
    progress = []
    result = owner._execute_analysis(static_analysis, prior, obs, grid, cfg,
        namespace=cp, device='cuda', progress=progress.append)
    actual, diag, _, _, storage, attempts = result
    assert storage == 'cuda-resident' and not diag.host_staging
    assert len(attempts) == 1
    assert progress[-1]['phase'] == 'complete'
    assert progress[-1]['storage'] == 'cuda-resident'
    for name in prior:
        np.testing.assert_allclose(actual[name], reference[name], atol=2e-12, rtol=2e-11)


def test_actual_partial_device_allocation_is_released_before_host_retry():
    prior, obs, grid, cfg = problem(fields=14, batches=20, members=9)
    cfg = replace(cfg, eigensolver='auto')
    original = {name: value.copy() for name, value in prior.items()}
    original_obs = [{key: np.array(getattr(batch, key), copy=True)
                     for key in ('values', 'errors', 'simulated', 'mask')} for batch in obs]
    reference = static_analysis(prior, obs, grid, cfg, solve_namespace=cp)
    pool = cp.cuda.MemoryPool()
    block = ((next(iter(prior.values())).nbytes + 511)//512)*512
    pool.set_limit(size=2*block)
    class Namespace:
        recovered = False
        references = []
        def __getattr__(self, name):
            if name in ('used_bytes', 'total_bytes', 'free_bytes'):
                return getattr(pool, name)
            return getattr(cp, name)
        def asarray(self, values, *args, **kwargs):
            result = cp.asarray(values, *args, **kwargs)
            if not self.recovered:
                self.references.append(weakref.ref(result))
            return result
        def get_default_memory_pool(self):
            return self
        def free_all_blocks(self):
            if not self.recovered:
                assert len(self.references) == 2
                assert all(reference() is None for reference in self.references)
                assert pool.used_bytes() == 0
                pool.set_limit(size=0)
                self.recovered = True
            pool.free_all_blocks()
    namespace = Namespace()
    with cp.cuda.using_allocator(pool.malloc):
        result = owner._execute_analysis(static_analysis, prior, obs, grid, cfg,
            namespace=namespace, device='cuda')
    actual, diag, _, _, storage, attempts = result
    assert namespace.recovered
    assert storage == 'host-staged-cuda' and diag.host_staging
    assert attempts[0]['error_type'] == 'OutOfMemoryError'
    assert [row['status'] for row in attempts] == ['memory-failed', 'computed']
    for name in prior:
        np.testing.assert_array_equal(prior[name], original[name])
        np.testing.assert_allclose(actual[name], reference[name], atol=2e-12, rtol=2e-11)
    for batch, snapshot in zip(obs, original_obs):
        for key, value in snapshot.items():
            np.testing.assert_array_equal(getattr(batch, key), value)
