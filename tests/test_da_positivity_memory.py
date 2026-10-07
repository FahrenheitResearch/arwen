"""apply_positivity evaluates one field at a time and moves no bit.

The policy is pointwise, so its result cannot depend on whether every
constrained field's float64 analysis is held at once.  These pin the
increments and the receipt against a direct elementwise evaluation, for
whole-ensemble arrays the size of a member stack, under every policy.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.da.positivity import apply_positivity, constrained_fields


def _case(seed=11):
    rng = np.random.default_rng(seed)
    shape = (6, 4, 7, 9)
    prior = {"qr": np.abs(rng.normal(size=shape)) * 1e-4,
             "qs": np.abs(rng.normal(size=shape)).astype(np.float32) * 1e-4,
             "nr": np.abs(rng.normal(size=shape)) * 1e3,
             "thp": rng.normal(size=shape)}
    increments = {name: rng.normal(size=shape) * (2e-4 if name != "nr" else 2e3)
                  for name in prior}
    return prior, increments


@pytest.mark.parametrize("policy", ["clip", "reject", "none"])
def test_policy_result_is_the_elementwise_reference(policy):
    prior, increments = _case()
    adjusted, receipt = apply_positivity(prior, increments, policy=policy)
    names = constrained_fields(tuple(increments))
    assert "thp" not in names and adjusted["thp"] is increments["thp"]
    analyses = {n: np.asarray(prior[n], np.float64) + increments[n]
                for n in names}
    shared = np.zeros(prior["qr"].shape, bool)
    for n in names:
        shared |= analyses[n] < 0.0
    for row, n in zip(receipt["per_field"], names, strict=True):
        negative = analyses[n] < 0.0
        assert row["field"] == n
        assert row["negative_points"] == int(negative.sum())
        mass = float(-analyses[n][negative].sum()) if negative.any() else 0.0
        if policy == "clip":
            expected = np.where(negative, -np.asarray(prior[n], np.float64),
                                increments[n])
            assert row["mass_added_by_clip"] == mass
        elif policy == "reject":
            expected = np.where(shared, 0.0, increments[n])
            assert row["mass_that_would_have_been_added"] == mass
            assert row["points_reverted"] == int(shared.sum())
        else:
            expected = increments[n]
            assert row["mass_left_negative"] == mass
        assert adjusted[n].dtype == increments[n].dtype
        assert adjusted[n].tobytes() == expected.astype(
            increments[n].dtype).tobytes()


def test_shape_disagreement_is_still_refused_by_field():
    from gpuwm.da.positivity import PositivityError

    prior, increments = _case()
    increments["nr"] = increments["nr"][:, :2]
    with pytest.raises(PositivityError, match="nr: prior"):
        apply_positivity(prior, increments, policy="clip")
