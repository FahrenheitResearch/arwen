"""Storage placement preserves the complete local transform and its inputs."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.da import letkf
from gpuwm.da.letkf import (GriddedObs, GridGeometry, LetkfConfig,
                           LetkfDiagnostics, Localization, analyze)
from gpuwm.da.static_covariance import static_analysis


def problem(*, fields=3, batches=4, ny=6, nx=7, members=8):
    rng = np.random.default_rng(892)
    shape = (3, ny, nx)
    prior = {f"field_{j}": rng.normal(size=(members, *shape)) + j
             for j in range(fields)}
    z = np.array([100., 800., 1700.])[:, None, None]
    terrain = np.arange(ny * nx).reshape(ny, nx) * 3.
    grid = GridGeometry(dx_m=1000., dy_m=1200., heights_m=z+terrain)
    obs = []
    for j in range(batches):
        sim = prior['field_0'] * (.7 + j*.1) + .02*prior['field_1']**2
        mask = rng.random(shape) < .1
        mask[:, 0, 0] = True
        values = sim.mean(axis=0) + rng.normal(size=shape)*.2
        errors = np.full(shape, .8+j*.2)
        sim = np.where(mask[None], sim, np.nan)
        values = np.where(mask, values, np.nan)
        errors = np.where(mask, errors, np.nan)
        window = (0, ny-1, 1, nx-1) if j % 2 else None
        if window:
            values, errors, sim, mask = (a[..., 1:] for a in (values, errors, sim, mask))
        obs.append(GriddedObs(f"observation_{j}", values, errors, sim, mask,
                             window=window))
    cfg = LetkfConfig(Localization(2600., 1500.), tuple(prior), .4,
                      chunk_points=13, eigensolver='library')
    return prior, obs, grid, cfg


@pytest.mark.parametrize('relaxation', ['rtps', 'rtpp'])
@pytest.mark.parametrize('static', [False, True])
def test_compact_transform_preserves_every_positive_neighbor(relaxation, static):
    prior, obs, grid, cfg = problem()
    cfg = replace(cfg, relaxation=relaxation)
    snapshots = {k: v.copy() for k, v in prior.items()}
    solver = static_analysis if static else analyze
    old_diag, new_diag = LetkfDiagnostics(), LetkfDiagnostics()
    old = solver(prior, obs, grid, cfg, old_diag)
    new = solver(prior, obs, grid, cfg, new_diag, solve_namespace=np)
    for name in prior:
        # Only zero terms leave the sum; float64 BLAS reduction order can
        # differ. This bound is below meaningful input perturbations.
        np.testing.assert_allclose(new[name], old[name], atol=2e-13, rtol=2e-12)
        np.testing.assert_array_equal(prior[name], snapshots[name])
    assert old_diag.active_points == new_diag.active_points
    assert old_diag.max_local_obs == new_diag.max_local_obs
    assert old_diag.stencil_slots == new_diag.stencil_slots
    # The default host-staged route finds neighbours with the once-built
    # index (gpuwm.da.neighbor_index), not the per-chunk geometry walk.
    assert new_diag.neighbor_search == "index"
    assert new_diag.neighbor_index["neighbours"] > 0
    forward_diag = LetkfDiagnostics()
    solver(prior, obs, grid, replace(cfg, neighbor_search="forward"),
           forward_diag, solve_namespace=np)
    assert forward_diag.geometry_reuses > forward_diag.geometry_evaluations
    assert new_diag.host_staging
    assert new_diag.staging_peak_bytes > 0


def test_progress_counts_completed_work_and_zero_observation_exit():
    prior, obs, grid, cfg = problem()
    progress = []
    diag = LetkfDiagnostics()
    analyze(prior, obs, grid, cfg, diag, solve_namespace=np, progress=progress.append)
    assert progress[0]['gridpoints_done'] == 0
    final = progress[-1]
    assert final['schema'] == 'gpuwm-da.analysis-progress.v1'
    assert final['phase'] == 'complete'
    assert final['gridpoints_done'] == final['gridpoints_total'] == 126
    assert final['active_points'] == diag.active_points
    assert all(a['gridpoints_done'] <= b['gridpoints_done']
               for a, b in zip(progress, progress[1:]))
    empty = []
    answer = analyze(prior, [], grid, cfg, solve_namespace=np, progress=empty.append)
    assert empty[-1]['gridpoints_done'] == 126
    for a in answer.values():
        np.testing.assert_array_equal(a, 0)


def test_capacity_counts_only_driver_free_and_remaining_idle_pool_bytes():
    class Pool:
        def __init__(self):
            self.released = False
        def free_all_blocks(self):
            self.released = True
        def free_bytes(self):
            assert self.released
            return 64 << 20
        def used_bytes(self):
            return 4 << 30
    pool = Pool()
    device = SimpleNamespace(
        get_default_memory_pool=lambda: pool,
        cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (0, 10 << 30))))
    assert letkf._device_free_bytes(device) == 64 << 20
    assert letkf._device_capacity(device) == (0, 64 << 20)
    pool.free_bytes = lambda: 0
    assert letkf._device_free_bytes(device) == 0


def test_unmasked_invalid_input_still_refuses_before_staging():
    prior, obs, grid, cfg = problem()
    first = obs[0]
    values = first.values.copy()
    values[first.mask] = np.nan
    obs[0] = replace(first, values=values)
    with pytest.raises(letkf.LetkfError, match='non-finite|finite'):
        analyze(prior, obs, grid, cfg, solve_namespace=np)


class RecordingNamespace:
    """An allocation witness for the real bounded gather and solver."""
    def __init__(self):
        self.transfers = []
    def __getattr__(self, name):
        return getattr(np, name)
    def asarray(self, a, *args, **kwargs):
        self.transfers.append((a.shape, a.nbytes))
        return np.asarray(a, *args, **kwargs)
    def asnumpy(self, a):
        return np.asarray(a)


def test_staging_contains_chunks_never_whole_prior_or_observation_cubes():
    prior, obs, grid, cfg = problem(fields=14, batches=20)
    recorder = RecordingNamespace()
    diag = LetkfDiagnostics()
    analyze(prior, obs, grid, cfg, diag, solve_namespace=recorder)
    assert recorder.transfers
    assert all(len(shape) <= 3 for shape, _ in recorder.transfers)
    assert sum(size for _, size in recorder.transfers) == diag.staging_bytes
    assert diag.staging_peak_bytes < cfg.chunk_points * diag.solve_bytes_per_point
    # Repeat with a larger domain but the same per-chunk work limit.
    larger = problem(fields=14, batches=20, ny=12, nx=14)
    next_diag = LetkfDiagnostics()
    analyze(*larger[:2], larger[2], larger[3], next_diag, solve_namespace=RecordingNamespace())
    assert next_diag.staging_peak_bytes < 2 * diag.staging_peak_bytes


def test_geometry_tile_and_actual_positive_device_chunks_have_separate_bounds():
    prior, obs, grid, cfg = problem(fields=14, batches=20, ny=12, nx=14)
    cfg = replace(cfg, chunk_points=None, memory_budget_mib=4)
    diag = LetkfDiagnostics()
    answer = analyze(prior, obs, grid, cfg, diag, solve_namespace=np)
    legacy = letkf.chunk_points_for_budget(
        diag.reachable_stencil_slots, 8, 8, 4 << 20, 504)
    assert diag.chunk_points > legacy
    assert diag.chunk_points * diag.host_geometry_bytes_per_point <= 4 << 20
    assert diag.staging_peak_bytes < 4 << 20
    assert diag.device_chunks < (504 + legacy - 1)//legacy
    reference = analyze(prior, obs, grid, replace(cfg, chunk_points=13))
    for name in prior:
        np.testing.assert_allclose(answer[name], reference[name], atol=2e-13, rtol=2e-12)


@pytest.mark.parametrize('card_limited', [False, True])
def test_one_positive_point_must_respect_the_binding_scratch_ceiling(monkeypatch, card_limited):
    prior, obs, grid, cfg = problem()
    cfg = replace(cfg, chunk_points=None,
                  memory_budget_mib=4 if card_limited else .0001)
    if card_limited:
        monkeypatch.setattr(letkf, '_device_capacity', lambda xp: (0, 0))
    with pytest.raises(letkf.LetkfError) as error:
        analyze(prior, obs, grid, cfg, solve_namespace=np)
    message = str(error.value)
    assert 'positive neighbours' in message
    if card_limited:
        assert '0 driver-free bytes + 0 reusable pool bytes' in message
        assert 'increasing the configured budget cannot create card capacity' in message
    else:
        assert 'Increase memory_budget_mib' in message


def test_host_staged_allocation_retry_preserves_completed_rows(monkeypatch):
    prior, obs, grid, cfg = problem()
    reference = analyze(prior, obs, grid, cfg, solve_namespace=np)
    real = letkf._eigendecompose
    def limited(xp, matrices, which):
        if matrices.shape[0] > 2:
            raise MemoryError('allocation failed')
        return real(xp, matrices, which)
    monkeypatch.setattr(letkf, '_eigendecompose', limited)
    diag = LetkfDiagnostics()
    result = analyze(prior, obs, grid, cfg, diag, solve_namespace=np)
    assert diag.chunk_oom_shrinks > 0
    for name in prior:
        np.testing.assert_allclose(result[name], reference[name], atol=2e-13, rtol=2e-12)
