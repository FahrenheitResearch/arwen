"""The observation-sparse device LETKF against the host oracle.

``gpuwm.da.letkf.analyze`` on numpy is the reference.  The device route must
agree with it to rounding, give the same bytes run to run and for every
chunk size, keep the bitwise-zero increment beyond the localisation lens at
``prior_inflation = 1``, and refuse what the host refuses.
"""
from dataclasses import replace

import cupy as cp
import numpy as np
import pytest

from gpuwm.da.letkf import (GriddedObs, GridGeometry, LetkfConfig,
                            LetkfDiagnostics, LetkfError, Localization,
                            analyze)
from gpuwm.da.letkf_device import analyze_device
from test_letkf_bounded_staging import problem

pytestmark = pytest.mark.gpu


def geodesic_problem(*, members=32, fields=4, batches=3, seed=5):
    """A small terrain-following, geolocated grid with windowed batches."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = 6, 11, 12
    shape = (nz, ny, nx)
    prior = {f"f{j}": rng.normal(size=(members, *shape)) * (1 + j) + 10 * j
             for j in range(fields)}
    lat = 31.0 + np.arange(ny)[:, None] * 0.027 + np.zeros((1, nx))
    lon = -89.0 + np.arange(nx)[None, :] * 0.031 + 0.001 * np.arange(ny)[:, None]
    terrain = rng.uniform(0, 300, size=(ny, nx))
    z = (np.array([20., 150., 500., 1100., 2000., 3500.])[:, None, None]
         + terrain[None])
    grid = GridGeometry(dx_m=3000., dy_m=3000., heights_m=z,
                        lat_deg=lat, lon_deg=lon)
    obs = []
    for b in range(batches):
        sim = (prior["f0"] * (0.6 + 0.2 * b) + 0.05 * prior["f1"] ** 2
               + rng.normal(size=(members, *shape)) * 0.1)
        mask = rng.random(shape) < 0.25
        values = sim.mean(axis=0) + rng.normal(size=shape) * 0.5
        errors = np.full(shape, 0.7 + 0.3 * b)
        window = (2, ny - 2, 1, nx - 3) if b == 1 else None
        loc = Localization(9000., 1800.) if b == 2 else None
        if window:
            j0, j1, i0, i1 = window
            cut = (slice(None), slice(j0, j1 + 1), slice(i0, i1 + 1))
            values, errors, mask = values[cut], errors[cut], mask[cut]
            sim = sim[(slice(None),) + cut]
        obs.append(GriddedObs(f"b{b}", values, errors, sim, mask,
                              localization=loc, window=window))
    cfg = LetkfConfig(Localization(12000., 2500.), tuple(prior), 0.8,
                      chunk_points=97, eigensolver="library")
    return prior, obs, grid, cfg


def _device(prior, obs, grid, cfg, **kw):
    diag = LetkfDiagnostics()
    return analyze_device(prior, obs, grid, cfg, diag, **kw), diag


def _assert_close_to_host(new, old, *, atol, rtol):
    for name in old:
        np.testing.assert_allclose(new[name], old[name], atol=atol, rtol=rtol)


@pytest.mark.parametrize("relaxation", ["rtps", "rtpp"])
@pytest.mark.parametrize("rho", [1.0, 1.15])
def test_matches_host_oracle_on_the_staging_problem(relaxation, rho):
    prior, obs, grid, cfg = problem(fields=5, batches=6)
    cfg = replace(cfg, relaxation=relaxation, prior_inflation=rho)
    host_diag = LetkfDiagnostics()
    host = analyze(prior, obs, grid, cfg, host_diag)
    new, diag = _device(prior, obs, grid, cfg)
    _assert_close_to_host(new, host, atol=1e-11, rtol=1e-10)
    assert diag.active_points == host_diag.active_points
    assert diag.max_local_obs == host_diag.max_local_obs
    for f in prior:
        assert diag.posterior_spread[f] == pytest.approx(
            host_diag.posterior_spread[f], rel=1e-10)
        assert diag.prior_spread[f] == pytest.approx(
            host_diag.prior_spread[f], rel=1e-12)
        assert diag.mean_increment_rms[f] == pytest.approx(
            host_diag.mean_increment_rms[f], rel=1e-8)


@pytest.mark.parametrize("relaxation", ["rtps", "rtpp"])
def test_matches_host_oracle_geodesic_32_members(relaxation):
    prior, obs, grid, cfg = geodesic_problem()
    cfg = replace(cfg, relaxation=relaxation)
    host_diag = LetkfDiagnostics()
    host = analyze(prior, obs, grid, cfg, host_diag)
    new, diag = _device(prior, obs, grid, cfg)
    _assert_close_to_host(new, host, atol=1e-10, rtol=1e-9)
    assert diag.active_points == host_diag.active_points
    assert diag.max_local_obs == host_diag.max_local_obs
    # The transform split is measured, and is inside the transform phase.
    split = diag.eigen_seconds + diag.products_seconds + diag.apply_seconds
    assert diag.eigen_seconds > 0 and diag.apply_seconds > 0
    assert split <= diag.transform_seconds


def test_matches_existing_device_resident_path():
    prior, obs, grid, cfg = geodesic_problem(seed=11)
    cfg = replace(cfg, eigensolver="auto", chunk_points=None,
                  memory_budget_mib=512.)
    dprior = {k: cp.asarray(v) for k, v in prior.items()}
    dobs = [replace(o, values=cp.asarray(o.values),
                    errors=cp.asarray(o.errors), mask=cp.asarray(o.mask),
                    simulated=cp.asarray(o.simulated)) for o in obs]
    old = {k: cp.asnumpy(v) for k, v in analyze(dprior, dobs, grid, cfg).items()}
    new, _diag = _device(prior, obs, grid, cfg)
    _assert_close_to_host(new, old, atol=1e-11, rtol=1e-10)


def test_run_to_run_and_chunk_size_give_the_same_bytes():
    prior, obs, grid, cfg = geodesic_problem(seed=3)
    first, _ = _device(prior, obs, grid, cfg, chunk_points=100000)
    second, _ = _device(prior, obs, grid, cfg, chunk_points=100000)
    small, diag = _device(prior, obs, grid, cfg, chunk_points=37)
    assert diag.device_chunks > 1
    for name in prior:
        assert first[name].tobytes() == second[name].tobytes()
        assert first[name].tobytes() == small[name].tobytes()


def test_device_inputs_give_the_host_input_bytes():
    prior, obs, grid, cfg = geodesic_problem(seed=4, members=12)
    host_in, _ = _device(prior, obs, grid, cfg)
    dprior = {k: cp.asarray(v) for k, v in prior.items()}
    dobs = [replace(o, values=cp.asarray(o.values),
                    errors=cp.asarray(o.errors), mask=cp.asarray(o.mask),
                    simulated=cp.asarray(o.simulated)) for o in obs]
    dev_in, _ = _device(dprior, dobs, grid, cfg)
    for name in prior:
        assert host_in[name].tobytes() == dev_in[name].tobytes()


def test_zero_beyond_the_lens_and_no_observation_cycle():
    prior, obs, grid, cfg = geodesic_problem(seed=8)
    new, diag = _device(prior, obs, grid, cfg)
    host = analyze(prior, obs, grid, cfg)
    for name in prior:
        untouched = np.all(host[name] == 0.0, axis=0)
        assert np.all(new[name][:, untouched] == 0.0)
    empty = [replace(o, mask=np.zeros_like(o.mask)) for o in obs]
    zero, diag = _device(prior, empty, grid, cfg)
    assert diag.active_points == 0
    for name in prior:
        assert not np.any(zero[name])
    none, _ = _device(prior, [], grid, cfg)
    for name in prior:
        assert not np.any(none[name])


def test_inflated_inactive_points_take_the_closed_form():
    prior, obs, grid, cfg = geodesic_problem(seed=9)
    cfg = replace(cfg, prior_inflation=1.3)
    empty = [replace(o, mask=np.zeros_like(o.mask)) for o in obs]
    host = analyze(prior, empty, grid, cfg)
    new, _ = _device(prior, empty, grid, cfg)
    _assert_close_to_host(new, host, atol=1e-13, rtol=1e-13)


def test_refusals_match_the_host():
    prior, obs, grid, cfg = geodesic_problem(seed=2)
    flat = dict(prior)
    flat["f1"] = np.broadcast_to(prior["f1"][:1], prior["f1"].shape).copy()
    with pytest.raises(LetkfError, match="no usable ensemble spread"):
        _device(flat, obs, grid, cfg)
    bad = dict(prior)
    bad["f2"] = prior["f2"].copy()
    bad["f2"][3, 2, 2, 2] = np.nan
    with pytest.raises(LetkfError, match="non-finite"):
        _device(bad, obs, grid, cfg)
    neg = [replace(obs[0], errors=-np.abs(np.asarray(obs[0].errors)))] + obs[1:]
    with pytest.raises(LetkfError, match="finite and"):
        _device(prior, neg, grid, cfg)
    tiny = replace(cfg, localization=Localization(10., 2500.))
    with pytest.raises(LetkfError, match="does not reach"):
        _device(prior, [obs[0]], grid, tiny)
