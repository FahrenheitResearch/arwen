"""Actual device transform with host storage and bounded transfers."""
from dataclasses import replace

import cupy as cp
import numpy as np
import pytest

from gpuwm.da.letkf import LetkfDiagnostics, analyze
from gpuwm.da.static_covariance import static_analysis
from test_letkf_bounded_staging import problem

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize('static', [False, True])
@pytest.mark.parametrize('relaxation', ['rtps', 'rtpp'])
def test_actual_device_full_transform_matches_host_staged_transform(static, relaxation):
    prior, obs, grid, cfg = problem(fields=14, batches=20)
    cfg = replace(cfg, eigensolver='auto', relaxation=relaxation)
    solver = static_analysis if static else analyze
    device_prior = {k: cp.asarray(v) for k, v in prior.items()}
    device_obs = [replace(o, values=cp.asarray(o.values), errors=cp.asarray(o.errors),
                          mask=cp.asarray(o.mask), simulated=cp.asarray(o.simulated))
                  for o in obs]
    old_diag, new_diag = LetkfDiagnostics(), LetkfDiagnostics()
    old = solver(device_prior, device_obs, grid, cfg, old_diag)
    new = solver(prior, obs, grid, cfg, new_diag, solve_namespace=cp)
    for name in prior:
        np.testing.assert_allclose(new[name], cp.asnumpy(old[name]),
                                   atol=2e-12, rtol=2e-11)
    assert old_diag.active_points == new_diag.active_points
    assert old_diag.max_local_obs == new_diag.max_local_obs
    assert new_diag.host_staging
    assert new_diag.staging_peak_bytes < cfg.chunk_points * new_diag.solve_bytes_per_point
    assert all(isinstance(v, np.ndarray) for v in new.values())


def test_actual_pool_reuses_split_free_block_at_zero_reported_driver_free(monkeypatch):
    from gpuwm.da.letkf import _device_capacity, _device_free_bytes
    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        whole = cp.empty(2 << 20, dtype=cp.uint8)
        del whole
        live = cp.empty(1 << 20, dtype=cp.uint8)
        # One half is live, preventing the other half returning to the driver.
        pool.free_all_blocks()
        assert pool.free_bytes() >= 1 << 20
        monkeypatch.setattr(cp, 'get_default_memory_pool', lambda: pool)
        monkeypatch.setattr(cp.cuda.runtime, 'memGetInfo', lambda: (0, 10 << 30))
        assert _device_capacity(cp) == (0, pool.free_bytes())
        assert _device_free_bytes(cp) == pool.free_bytes()
        reused = cp.empty(1 << 20, dtype=cp.uint8)
        assert pool.free_bytes() == 0
        assert live.nbytes == reused.nbytes
