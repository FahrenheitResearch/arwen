"""Positive-neighbour staging on full-depth, bounded horizontal columns."""
from dataclasses import replace

import numpy as np

from gpuwm.da import letkf
from gpuwm.da.letkf import (GridGeometry, GriddedObs, LetkfConfig,
                           LetkfDiagnostics, Localization)
from test_letkf_bounded_staging import problem


def full_depth_problem(*, fields=14, batches=20, members=9, chunk=96):
    rng = np.random.default_rng(9201)
    shape = (49, 6, 7)
    terrain = np.arange(42).reshape(6, 7) * .25
    grid = GridGeometry(dx_m=3000., dy_m=3000.,
        heights_m=np.arange(49)[:, None, None]*140. + terrain + 100.)
    prior = {f'field_{n}': rng.normal(size=(members, *shape))+n
             for n in range(fields)}
    obs = []
    for n in range(batches):
        mask = rng.random(shape) < .01
        # Exercise top, bottom, and horizontal boundary observations too.
        mask[0, 0, 0] = mask[-1, -1, -1] = True
        sim = .8*prior['field_0'] + .01*prior['field_1']**2
        values = np.mean(sim, axis=0) + rng.normal(size=shape)*.2
        errors = np.full(shape, .8+n*.02)
        sim = np.where(mask[None], sim, np.nan)
        values = np.where(mask, values, np.nan)
        errors = np.where(mask, errors, np.nan)
        window = (0, 5, 1, 6) if n % 2 else None
        if window:
            values, errors, sim, mask = (a[..., 1:] for a in (values, errors, sim, mask))
        obs.append(GriddedObs(f'observation_{n}', values, errors, sim, mask, window=window))
    cfg = LetkfConfig(Localization(15000., 3000.), tuple(prior), .4,
                      chunk_points=chunk, memory_budget_mib=256, eigensolver='library')
    assert len(letkf._vertical_stencil(cfg.localization, grid)) == 43
    return prior, obs, grid, cfg


def test_host_path_never_concatenates_dense_all_batch_stencils(monkeypatch):
    prior, obs, grid, cfg = problem(fields=14, batches=20)
    reference = letkf.analyze(prior, obs, grid, cfg, solve_namespace=np)
    concatenate = np.concatenate
    def guarded(arrays, *args, **kwargs):
        axis = kwargs.get('axis', args[0] if args else 0)
        if axis == 1 and len(arrays) > 1 and arrays[0].ndim == 2:
            raise AssertionError('Dense all-batch stencil concatenation')
        return concatenate(arrays, *args, **kwargs)
    monkeypatch.setattr(np, 'concatenate', guarded)
    diag = LetkfDiagnostics()
    result = letkf.analyze(prior, obs, grid, cfg, diag, solve_namespace=np)
    for name in prior:
        np.testing.assert_array_equal(result[name], reference[name])
    assert diag.sparse_neighbor_peak_bytes > 0
    dense_pair = cfg.chunk_points * diag.stencil_slots * 16
    assert diag.sparse_neighbor_peak_bytes < dense_pair


def test_full_depth_sparse_staging_matches_dense_transform_and_all_neighbors():
    prior, obs, grid, cfg = full_depth_problem(fields=3, batches=4, chunk=13)
    reference_diag, diag = LetkfDiagnostics(), LetkfDiagnostics()
    reference = letkf.analyze(prior, obs, grid, cfg, reference_diag)
    result = letkf.analyze(prior, obs, grid, cfg, diag, solve_namespace=np)
    for name in prior:
        np.testing.assert_allclose(result[name], reference[name], atol=2e-13, rtol=2e-12)
    assert diag.active_points == reference_diag.active_points
    assert diag.max_local_obs == reference_diag.max_local_obs
    assert diag.stencil_slots == reference_diag.stencil_slots


def test_solve_precision_conversion_precedes_positive_neighbor_count(monkeypatch):
    prior, obs, grid, cfg = problem()
    cfg = replace(cfg, solve_dtype='float32')
    weight = letkf.gaspari_cohn
    def tiny_off_axis(distance, cutoff):
        # Two positive off-axis factors produce a positive float64 product
        # that underflows in float32. It must not enter the active roster.
        return weight(distance, cutoff) * np.where(np.asarray(distance) > 0, 1e-30, 1.)
    monkeypatch.setattr(letkf, 'gaspari_cohn', tiny_off_axis)
    reference_diag, diag = LetkfDiagnostics(), LetkfDiagnostics()
    reference = letkf.analyze(prior, obs, grid, cfg, reference_diag)
    result = letkf.analyze(prior, obs, grid, cfg, diag, solve_namespace=np)
    assert diag.active_points == reference_diag.active_points
    assert diag.max_local_obs == reference_diag.max_local_obs
    for name in prior:
        np.testing.assert_allclose(result[name], reference[name], atol=2e-6, rtol=2e-5)
