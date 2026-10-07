"""The mean-preserving positivity policy, the default of every DA door.

The clip it replaced raised every negative member to zero and lowered none,
so each analysis added mass, always wetward (gpuwm/da/positivity.py).  The
mean-preserving rule ends the undershooting members at zero and scales the
members' non-negative analyses so the ensemble keeps the filter's mean.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.da import positivity


def _ensemble(seed=3, members=6, shape=(4, 5, 5)):
    rng = np.random.default_rng(seed)
    prior = np.abs(rng.normal(0.0, 1.0e-4, (members,) + shape)).astype(
        np.float32)
    increment = rng.normal(0.0, 1.6e-4, (members,) + shape).astype(
        np.float32)
    return {"qr": prior}, {"qr": increment}


def test_the_default_policy_is_mean_preserving():
    assert positivity.DEFAULT_POLICY == "mean-preserving"
    assert positivity.POLICIES[0] == "mean-preserving"
    assert "mean-preserving" in positivity.BOUNDING_POLICIES


def test_no_member_ends_negative_and_the_ensemble_mean_is_kept():
    prior, increment = _ensemble()
    bounded, receipt = positivity.apply_positivity(
        prior, increment, policy="mean-preserving")
    positivity.verify_non_negative(prior, bounded)
    before = prior["qr"].astype(np.float64) + increment["qr"]
    after = prior["qr"].astype(np.float64) + bounded["qr"]
    assert receipt["negative_points"] > 0, "the case must go negative"
    mean_before = before.mean(axis=0)
    mean_after = after.mean(axis=0)
    kept = mean_before > 0.0
    # The ensemble mean is the filter's wherever that mean is non-negative
    # (to float32 rounding of the stored increment).
    np.testing.assert_allclose(mean_after[kept], mean_before[kept],
                               rtol=1e-5, atol=1e-11)
    # Where the filter's mean is itself negative every member ends at zero.
    assert np.all(after[:, ~kept] == 0.0)


def test_it_adds_far_less_mass_than_the_clip_and_says_how_much():
    prior, increment = _ensemble()
    _, clipped = positivity.apply_positivity(prior, increment, policy="clip")
    bounded, receipt = positivity.apply_positivity(
        prior, increment, policy="mean-preserving")
    before = prior["qr"].astype(np.float64) + increment["qr"]
    after = prior["qr"].astype(np.float64) + bounded["qr"]
    added = after.sum() - before.sum()
    assert receipt["mass_added"] == pytest.approx(added, rel=1e-4)
    assert receipt["mass_added"] < 0.5 * clipped["mass_added_by_clip"]
    assert receipt["mass_added"] + receipt["mass_redistributed"] == \
        pytest.approx(clipped["mass_added_by_clip"], rel=1e-9)
    assert receipt["mass_added_by_clip"] == 0.0


def test_cells_no_member_undershoots_come_back_bit_for_bit():
    prior, increment = _ensemble()
    bounded, _ = positivity.apply_positivity(
        prior, increment, policy="mean-preserving")
    before = prior["qr"].astype(np.float64) + increment["qr"]
    untouched = ~(before < 0.0).any(axis=0)
    assert untouched.any()
    assert np.array_equal(bounded["qr"][:, untouched],
                          increment["qr"][:, untouched])
    assert bounded["qr"].dtype == increment["qr"].dtype


def test_one_member_has_no_mean_to_keep_so_the_rule_is_the_clip():
    prior, increment = _ensemble(members=1)
    one_prior = {"qr": prior["qr"][0]}
    one_increment = {"qr": increment["qr"][0]}
    clipped, _ = positivity.apply_positivity(one_prior, one_increment,
                                             policy="clip")
    bounded, receipt = positivity.apply_positivity(
        one_prior, one_increment, policy="mean-preserving")
    assert np.array_equal(clipped["qr"], bounded["qr"])
    assert receipt["per_field"][0]["member_axis"] is False


def test_a_worked_cell():
    """Three members at -1, 1, 3 (mean 1): the bound gives 0, 0.75, 2.25."""
    prior = {"qr": np.array([[[[1.0]]], [[[1.0]]], [[[1.0]]]], np.float64)}
    increment = {"qr": np.array([[[[-2.0]]], [[[0.0]]], [[[2.0]]]],
                                np.float64)}
    bounded, receipt = positivity.apply_positivity(
        prior, increment, policy="mean-preserving")
    after = (prior["qr"] + bounded["qr"]).ravel()
    np.testing.assert_allclose(after, [0.0, 0.75, 2.25])
    assert receipt["mass_added"] == pytest.approx(0.0, abs=1e-15)
    assert receipt["mass_redistributed"] == pytest.approx(1.0)


def test_the_cycle_driver_bounds_the_whole_ensemble_at_once(tmp_path):
    """gpuwm.ensemble.cycle keeps the ENSEMBLE mean, not each member's."""
    from gpuwm.ensemble.cycle import _enforce_positivity_ensemble

    shape = (2, 3, 3)
    backgrounds, increments = {}, {}
    for index, (base, step) in enumerate(((1e-3, -2e-3), (1e-3, 0.0),
                                          (1e-3, 2e-3))):
        path = tmp_path / f"m{index}.npz"
        np.savez(path, **{"state/qr": np.full(shape, base, np.float32)})
        backgrounds[index] = path
        increments[index] = {"qr": np.full(shape, step, np.float32),
                             "u": np.ones(shape, np.float32)}
    out = _enforce_positivity_ensemble(backgrounds, increments)
    analyses = [1e-3 + out[i][0]["qr"].astype(np.float64) for i in range(3)]
    np.testing.assert_allclose([a.mean() for a in analyses],
                               [0.0, 0.75e-3, 2.25e-3], atol=1e-9)
    # Unconstrained fields pass through untouched.
    assert all(np.array_equal(out[i][0]["u"], increments[i]["u"])
               for i in range(3))
    receipts = [out[i][1] for i in range(3)]
    assert all(r["scope"] == "ensemble" for r in receipts)
    assert sum(r["negative_points"] for r in receipts) == int(np.prod(shape))
