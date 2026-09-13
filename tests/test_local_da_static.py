"""Analytic checks for deterministic covariance analysis, not weather skill."""
from dataclasses import replace
import numpy as np
import pytest
from gpuwm.da.letkf import GriddedObs, GridGeometry, LetkfConfig, Localization
from gpuwm.da.static_covariance import static_analysis, perturbation_options


def problem(values=(2., 2., 4.), observed=5., background_h=2.):
    prior = {'u': np.broadcast_to(np.asarray(values)[:, None, None, None], (len(values), 1, 3, 3)).copy()}
    sim = np.broadcast_to(np.asarray([background_h, 2., 4.])[:, None, None, None], (3, 1, 3, 3)).copy()
    mask = np.zeros((1, 3, 3), bool)
    mask[0, 1, 1] = True
    batch = GriddedObs('test', np.full((1, 3, 3), observed), 1., sim, mask)
    cfg = LetkfConfig(localization=Localization(1500., 1000.),
                       analysis_fields=('u',), rtps_alpha=.8)
    grid = GridGeometry(dx_m=1000., dy_m=1000., heights_m=np.array([0.]))
    return prior, [batch], grid, cfg


def test_linear_scalar_matches_kalman_mean_not_a_one_member_filter():
    prior, batches, grid, cfg = problem()
    original = prior['u'].copy()
    inc = static_analysis(prior, batches, grid, cfg)
    # The two static states have sample covariance 2. The innovation is
    # y - H(background)=3, NOT y - mean(H(samples))=2.
    np.testing.assert_allclose(inc['u'][0, 0, 1, 1], 2. * 3. / (2. + 1.))
    np.testing.assert_array_equal(inc['u'][1:], 0.)
    np.testing.assert_array_equal(prior['u'], original)


def test_nonlinear_operator_uses_background_innovation():
    prior, batches, grid, cfg = problem(observed=8., background_h=4.)
    sim = np.broadcast_to(np.array([4., 4., 16.])[:, None, None, None], (3, 1, 3, 3)).copy()
    batches = [replace(batches[0], simulated=sim)]
    result = static_analysis(prior, batches, grid, cfg)
    # Cov(x,x^2)=12, Var(x^2)=72. Static mean H is 10, but H(xb)=4.
    np.testing.assert_allclose(result['u'][0, 0, 1, 1], 12. / 73. * 4.)


def test_zero_innovation_is_exact_identity_even_if_samples_are_biased():
    prior, batches, grid, cfg = problem(observed=2.)
    np.testing.assert_array_equal(static_analysis(prior, batches, grid, cfg)['u'], 0.)


def test_masked_observation_is_exact_identity():
    prior, batches, grid, cfg = problem()
    batches = [replace(batches[0], mask=np.zeros((1, 3, 3), bool))]
    np.testing.assert_array_equal(static_analysis(prior, batches, grid, cfg)['u'], 0.)


def test_not_enough_static_samples_refused_by_name():
    prior, batches, grid, cfg = problem(values=(2., 4.))
    batches = [replace(batches[0], simulated=batches[0].simulated[:2])]
    with pytest.raises(ValueError, match='two covariance samples'):
        static_analysis(prior, batches, grid, cfg)


def test_covariance_options_use_the_existing_perturbation_contract():
    from gpuwm.da.perturb import PerturbationConfig
    config = PerturbationConfig.from_mapping(dict(dx_km=3., dy_km=3.,
                                                 **perturbation_options()))
    assert {'u', 'v', 'theta', 'qv'}.issubset(set(config.field_names))
    assert config.qv_floor >= 0.


def test_static_samples_refresh_eos_and_preserve_background():
    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_vertical_coord, make_base_state
    from gpuwm.core.state import DomainState
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.ensemble.state_sha import serialized_state_attrs
    from gpuwm.da.static_covariance import covariance_states
    cfg = RunConfig(nx=16, ny=16, nz=24, dx=3000., dy=3000., ztop=5000.,
                    dt=10., run_seconds=900., moist=True, mp_physics=6, hypsometric_opt=1)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 300. + .003 * np.asarray(z), cfg.p_surf, cfg.ztop)
    state = DomainState(cfg, array_module=np)
    state.load_base(coord, base)
    state.qv[...] = .003
    update_diagnostics(state, cfg.hypsometric_opt)
    background = {name: np.array(getattr(state, name), copy=True) for name in serialized_state_attrs()
                  if getattr(state, name, None) is not None}
    original = {name: values.copy() for name, values in background.items()}
    names = ('thb', 'phb', 'dphb_resid', 'alb', 'rdnw', 'c1h', 'c2h', 'c3h', 'c4h',
             'c3f', 'c4f', 'dc3f', 'dc4f', 'mub2d', 'p_top', 'dnw')
    setup = {name: getattr(state, name) for name in names}
    samples, report = covariance_states(background, setup, cfg, samples=3, seed=10,
        options=perturbation_options(length_scale_km=6., rim_width=1, mp_physics=6))
    assert len(samples) == 4 and report['forecast_trajectories'] == 1
    for name, value in original.items():
        np.testing.assert_array_equal(background[name], value)
    assert np.any(samples[1]['thp'] != background['thp'])
    assert np.any(samples[1]['p'] != background['p'])
    assert all(np.all(np.isfinite(s['p'])) for s in samples)
    assert not np.shares_memory(samples[1]['thp'], samples[2]['thp'])
