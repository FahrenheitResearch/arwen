"""Independent CPU references for the spread-repair numerical contracts.

These tests integrate the mathematical density over a wide domain using
composite Simpson quadrature. They do not call the production quadrature or
inverse-gamma fitting helpers. Run them on a rented box CPU, not the desktop.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from gpuwm.da.spread_repair import (
    AdaptiveInflationState,
    SpreadRepairConfig,
    TimeShiftSnapshot,
    time_shifted_covariance,
    update_adaptive_inflation,
)


def _reference_posterior_moments(
    shape, scale, members, observed, error_std, gamma=1.0
):
    """Integrate the IG prior times the enhanced finite-N likelihood.

    The density is written first with respect to lambda, then changed to
    log(lambda) for integration. Invalid innovation variance has zero
    likelihood. The broad domain intentionally differs from the production
    integration support, so missing tails are detectable.
    """
    members = np.asarray(members, dtype=np.float64).reshape(-1)
    count = len(members)
    centre = sum(float(value) for value in members) / count
    prior_variance = sum((float(value) - centre) ** 2 for value in members) / (
        count - 1
    )
    innovation = float(observed) - centre
    error_variance = float(error_std) ** 2
    log_lambda = np.linspace(math.log(1e-7), math.log(1e7), 262145)
    lam = np.exp(log_lambda)
    obs_variance = error_variance + (
        (1.0 + float(gamma) * (np.sqrt(lam) - 1.0)) ** 2 - 1.0 / count
    ) * prior_variance
    log_density = np.full(lam.shape, -np.inf)
    valid = obs_variance > 0
    # Constants independent of lambda cancel from normalized expectations.
    log_density[valid] = (
        -(float(shape) + 1.0) * log_lambda[valid]
        - float(scale) / lam[valid]
        - 0.5 * np.log(obs_variance[valid])
        - 0.5 * innovation ** 2 / obs_variance[valid]
    )
    log_integrand = log_density + log_lambda
    density = np.exp(log_integrand - np.max(log_integrand))
    coefficients = np.ones(lam.shape)
    coefficients[1:-1:2] = 4.0
    coefficients[2:-1:2] = 2.0
    weights = coefficients * density
    # The common Simpson step / 3 cancels in each ratio.
    total = weights.sum()
    posterior_mean = (weights * lam).sum() / total
    posterior_variance = (
        weights * (lam - posterior_mean) ** 2
    ).sum() / total
    return float(posterior_mean), float(posterior_variance)


def _state_moments(state):
    shape = float(np.asarray(state.shape).item())
    scale = float(np.asarray(state.scale).item())
    return (
        scale / (shape - 1.0),
        scale ** 2 / ((shape - 1.0) ** 2 * (shape - 2.0)),
    )


def _single_cell_update(
    *, members=(-1.0, 0.0, 1.0), observed=4.0, error_std=1.0,
    gamma=1.0, shape=8.0, scale=9.45, config=None,
):
    config = config or SpreadRepairConfig()
    state = AdaptiveInflationState(
        np.full((1, 1, 1), shape), np.full((1, 1, 1), scale)
    )
    updated = update_adaptive_inflation(
        state,
        np.full((1, 1, 1), observed),
        np.asarray(members, dtype=np.float64).reshape(-1, 1, 1, 1),
        np.full((1, 1, 1), error_std),
        np.ones((1, 1, 1), dtype=bool),
        config,
        localization_weight=gamma,
    )
    return state, updated, config


@pytest.mark.parametrize("mode,sd", [(1.01, 0.1), (1.5, 0.8), (2.0, 2.0)])
def test_initial_inverse_gamma_matches_mode_and_variance(mode, sd):
    config = SpreadRepairConfig(
        inflation_initial_mode=mode, inflation_initial_sd=sd
    )
    state = AdaptiveInflationState.initial((1, 1, 1), config)
    shape = float(state.shape.item())
    scale = float(state.scale.item())
    assert shape > 2.0
    assert scale / (shape + 1.0) == pytest.approx(mode, rel=1e-9)
    assert _state_moments(state)[1] == pytest.approx(sd ** 2, rel=1e-8)


@pytest.mark.parametrize(
    "members,observed,error_std,gamma",
    [
        ((-1.0, 0.0, 1.0), 4.0, 1.0, 1.0),
        ((-1.0, 0.0, 1.0), 0.0, 2.0, 0.35),
        ((-1.5, -0.5, 0.5, 1.5), 3.0, 2.0, 0.8),
    ],
)
def test_adaptive_posterior_matches_independent_density_integral(
    members, observed, error_std, gamma
):
    _, updated, _ = _single_cell_update(
        members=members, observed=observed, error_std=error_std, gamma=gamma
    )
    expected_mean, expected_variance = _reference_posterior_moments(
        8.0, 9.45, members, observed, error_std, gamma
    )
    actual_mean, actual_variance = _state_moments(updated)
    assert actual_mean == pytest.approx(expected_mean, rel=0.004)
    assert actual_variance == pytest.approx(expected_variance, rel=0.012)


def test_observation_error_changes_inferred_inflation():
    _, precise, config = _single_cell_update(error_std=1.0)
    _, noisy, _ = _single_cell_update(error_std=4.0)
    assert float(precise.mode(config).item()) > float(noisy.mode(config).item())
    assert _state_moments(precise)[0] > _state_moments(noisy)[0]


def test_narrow_restart_prior_is_resolved_under_weak_information():
    cfg = SpreadRepairConfig(inflation_initial_sd=1e-4)
    prior = AdaptiveInflationState.initial((1,1,1),cfg)
    updated = update_adaptive_inflation(prior,np.ones((1,1,1)),
        np.array([-1.,0.,1.])[:,None,None,None],np.full((1,1,1),100.),
        np.ones((1,1,1),bool),cfg)
    assert float(updated.mode(cfg).item()) == pytest.approx(1.05,abs=2e-6)
    assert _state_moments(updated)[1] == pytest.approx(1e-8,rel=.02)


@pytest.mark.parametrize(
    "members,gamma", [((-1.0, 0.0, 1.0), 0.0), ((2.0, 2.0, 2.0), 1.0)]
)
def test_no_information_preserves_the_full_prior_distribution(members, gamma):
    before, after, config = _single_cell_update(
        members=members, observed=50.0, gamma=gamma
    )
    np.testing.assert_array_equal(after.shape, before.shape)
    np.testing.assert_array_equal(after.scale, before.scale)
    np.testing.assert_array_equal(after.mode(config), before.mode(config))
    assert after.cycles == before.cycles + 1


def test_finite_n_likelihood_excludes_invalid_nodes_without_rejecting_observation():
    # P=100, N=3, R=0.25: low lambda is impossible, while the posterior
    # above lambda=1/3-R/P is well defined and has finite moments.
    members = (-10.0, 0.0, 10.0)
    _, updated, _ = _single_cell_update(
        members=members, observed=12.0, error_std=0.5,
        config=SpreadRepairConfig(quadrature_points=513),
    )
    expected_mean, expected_variance = _reference_posterior_moments(
        8.0, 9.45, members, 12.0, 0.5
    )
    actual_mean, actual_variance = _state_moments(updated)
    assert actual_mean == pytest.approx(expected_mean, rel=0.01)
    assert actual_variance == pytest.approx(expected_variance, rel=0.025)


def test_applied_bounds_do_not_truncate_a_diffuse_inflation_posterior():
    # alpha=3 is a proper, finite-variance IG prior. Its high-lambda tail
    # materially affects posterior moments even though the applied factor
    # is capped at three. Clipping the output must not remove that tail.
    config = SpreadRepairConfig(quadrature_points=513)
    _, updated, _ = _single_cell_update(
        shape=3.0, scale=4.2, observed=0.0, error_std=2.0, config=config
    )
    expected_mean, expected_variance = _reference_posterior_moments(
        3.0, 4.2, (-1.0, 0.0, 1.0), 0.0, 2.0
    )
    actual_mean, actual_variance = _state_moments(updated)
    assert actual_mean == pytest.approx(expected_mean, rel=0.01)
    assert actual_variance == pytest.approx(expected_variance, rel=0.025)
    factor = float(updated.mode(config).item())
    assert config.inflation_min <= factor <= config.inflation_max


def test_extreme_innovation_remains_finite_and_bounded():
    _, updated, config = _single_cell_update(observed=1e8)
    assert np.isfinite(updated.shape).all()
    assert np.isfinite(updated.scale).all()
    assert (updated.shape > 2.0).all()
    factor = float(updated.mode(config).item())
    assert math.isfinite(factor)
    assert config.inflation_min <= factor <= config.inflation_max


def test_restart_roundtrip_reproduces_following_cycles_exactly():
    config = SpreadRepairConfig()
    grid_shape = (1, 1, 3)
    members = np.array([-1.0, 0.0, 1.0])[:, None, None, None]
    hx = members * np.array([1.0, 2.0, 0.0])[None, None, None, :]
    errors = np.full(grid_shape, 2.0)
    gate = np.array([True, True, False]).reshape(grid_shape)
    initial = AdaptiveInflationState.initial(grid_shape, config)
    state = update_adaptive_inflation(
        initial, np.array([4.0, 3.0, 90.0]).reshape(grid_shape),
        hx, errors, gate, config,
    )
    record = state.checkpoint()
    serialized = json.loads(json.dumps({
        "schema": record["schema"], "shape": record["shape"].tolist(),
        "scale": record["scale"].tolist(), "cycles": record["cycles"],
    }))
    restored = AdaptiveInflationState.restore(serialized, grid_shape)
    for values in ([2.0, 5.0, 0.0], [6.0, 1.0, 100.0]):
        observed = np.asarray(values).reshape(grid_shape)
        state = update_adaptive_inflation(state, observed, hx, errors, gate, config)
        restored = update_adaptive_inflation(
            restored, observed, hx, errors, gate, config
        )
        np.testing.assert_array_equal(restored.shape, state.shape)
        np.testing.assert_array_equal(restored.scale, state.scale)
        assert restored.cycles == state.cycles
    # Restoring must also own its arrays instead of sharing a mutable record.
    saved = state.checkpoint()
    independent = AdaptiveInflationState.restore(saved, grid_shape)
    saved["shape"][...] = 99.0
    np.testing.assert_array_equal(independent.shape, state.shape)


def _phase_snapshots():
    count = 4
    centre = np.array([10.0, 100.0])
    phase = np.array([2.0, -1.0])
    return [
        TimeShiftSnapshot(
            offset_seconds=offset,
            states={"thp": np.tile(centre + step * phase, (count, 1))},
            forecast_origin_seconds=8000,
            observation_cutoff_seconds=8000,
        )
        for offset, step in ((900, 1), (-900, -1), (0, 0))
    ]


def test_vtsm_retains_phase_covariance_that_vtsp_removes():
    snapshots = _phase_snapshots()
    vtsm, member_receipt = time_shifted_covariance(
        snapshots, analysis_seconds=10000, mode="vtsm"
    )
    vtsp, perturbation_receipt = time_shifted_covariance(
        snapshots, analysis_seconds=10000, mode="vtsp"
    )
    # Each time slab has exactly zero covariance. The only spread comes
    # from temporal phase changes. Derive its covariance analytically.
    expected = (8.0 / 11.0) * np.array([[4.0, -2.0], [-2.0, 1.0]])
    np.testing.assert_allclose(np.cov(vtsm["thp"], rowvar=False), expected)
    np.testing.assert_allclose(np.cov(vtsp["thp"], rowvar=False), np.zeros((2, 2)))
    central = next(slot.states["thp"] for slot in snapshots if slot.offset_seconds == 0)
    np.testing.assert_array_equal(vtsm["thp"][4:8], central)
    for receipt in (member_receipt, perturbation_receipt):
        assert receipt["trajectory_members"] == 4
        assert receipt["covariance_samples"] == 12
        assert receipt["central_indices"] == [4, 5, 6, 7]


@pytest.mark.parametrize("offset,cutoff", [(-900, 10000), (900, 10001)])
def test_vts_rejects_observations_after_each_permitted_cutoff(offset, cutoff):
    snapshots = [
        TimeShiftSnapshot(
            slot.offset_seconds, slot.states, slot.forecast_origin_seconds,
            cutoff if slot.offset_seconds == offset else slot.observation_cutoff_seconds,
        )
        for slot in _phase_snapshots()
    ]
    with pytest.raises(ValueError, match="permitted cutoff"):
        time_shifted_covariance(snapshots, analysis_seconds=10000)
