"""The policy that bounds an analysis and the gate that refuses one agree.

Two contracts decide whether a cycled analysis survives the process boundary,
and they are written in different files.  ``gpuwm.core.health.rule_for_field``
says which fields the pre-leg gate refuses below zero; the tuple
``gpuwm.da.positivity.NON_NEGATIVE_FIELDS`` says which fields the run's
positivity policy has an opinion about.  A field in the first and not in the
second is analysed with nothing bounding it and then refused by the gate one
leg later, in another process, with no clue pointing back here.

That is not hypothetical.  The aerosol-aware tracers ``nwfa``/``nifa`` joined
the health contract before they joined the policy's list, and a real radar
cycle saved a generation whose water-friendly aerosol number reached
-2.96e9 kg^-1 in 4,178 cells.
The policy's own receipt named them "unconstrained" and the driver believed
it.  Both cycles that hit it burned a card for four minutes to find out.

So the driver asks the two contracts whether they agree, before it saves
anything, and refuses by name when they do not.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.health import rule_for_field
from gpuwm.da import positivity as pos

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "da_cycle_prepared", _ROOT / "tools" / "da_cycle_prepared.py")
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


def _field(value):
    return np.full((2, 3, 3), value, dtype=np.float32)


def _without(*names):
    """The policy's list with ``names`` taken out of it."""
    return tuple(n for n in pos.NON_NEGATIVE_FIELDS if n not in names)


def test_a_field_the_gate_floors_at_zero_and_the_policy_ignores_is_refused(
        monkeypatch):
    monkeypatch.setattr(pos, "NON_NEGATIVE_FIELDS", _without("nwfa"))
    prior = {"nwfa": _field(2.8e9), "qv": _field(0.01)}
    with pytest.raises(pos.PositivityError) as excinfo:
        driver.merge_hotstart_increments(
            {"nwfa": _field(-4.4e9), "qv": _field(-0.001)}, {},
            prior=prior, positivity_policy="clip")
    message = str(excinfo.value)
    assert "nwfa" in message, (
        "the refusal has to name the field, or the operator is back to "
        f"reading a health gate in another process: {message}")


def test_the_refusal_names_both_contracts(monkeypatch):
    monkeypatch.setattr(pos, "NON_NEGATIVE_FIELDS", _without("nwfa", "nifa"))
    prior = {"nwfa": _field(1.0), "nifa": _field(1.0)}
    with pytest.raises(pos.PositivityError) as excinfo:
        driver.merge_hotstart_increments(
            {"nwfa": _field(-2.0), "nifa": _field(-2.0)}, {},
            prior=prior, positivity_policy="clip")
    message = str(excinfo.value)
    assert "nifa" in message and "nwfa" in message, (
        f"every disagreeing field is named, not the first: {message}")
    assert "health" in message.lower(), (
        f"the refusal says which two contracts disagree: {message}")


def test_a_field_with_a_floor_above_zero_is_not_the_policys_business():
    # thp's floor is 100 K, not 0: a theta perturbation is not a
    # positivity question and inventing one here would be a bug that
    # looks like caution.
    assert rule_for_field("thp").lower == 100.0
    prior = {"thp": _field(300.0), "u": _field(5.0)}
    merged, _overlap, receipt = driver.merge_hotstart_increments(
        {"thp": _field(-1.0), "u": _field(-1.0)}, {},
        prior=prior, positivity_policy="clip")
    assert receipt is not None
    assert np.asarray(merged["thp"]).min() == pytest.approx(-1.0)


def test_a_run_that_stated_no_policy_is_not_given_this_refusal(monkeypatch):
    monkeypatch.setattr(pos, "NON_NEGATIVE_FIELDS", _without("nwfa"))
    prior = {"nwfa": _field(1.0)}
    merged, _overlap, receipt = driver.merge_hotstart_increments(
        {"nwfa": _field(-2.0)}, {}, prior=prior, positivity_policy=None)
    assert receipt is None
    assert np.asarray(merged["nwfa"]).min() == pytest.approx(-2.0)


def test_the_two_contracts_agree_on_the_tree_as_it_stands():
    """The guard is green here, which is the point: it fires on a STALE
    installed package, not on this tree."""
    floored_at_zero = tuple(
        name for name in pos.NON_NEGATIVE_FIELDS
        if rule_for_field(name).lower == 0.0)
    assert floored_at_zero, "the health gate floors some of these at zero"
    for name in ("nwfa", "nifa", "qv", "qg", "nc"):
        assert name in pos.NON_NEGATIVE_FIELDS
        assert rule_for_field(name).lower == 0.0


def test_the_fields_this_cycle_analyses_all_pass_the_guard():
    analysed = ("thp", "qv", "u", "v", "qs", "qg", "qc", "nc", "qr", "nr",
                "qi", "ni", "nwfa", "nifa")
    prior = {name: _field(1.0) for name in analysed}
    merged, _overlap, receipt = driver.merge_hotstart_increments(
        {name: _field(-0.5) for name in analysed}, {},
        prior=prior, positivity_policy="clip")
    assert receipt is not None
    assert set(merged) == set(analysed)
