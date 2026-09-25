"""The correction a child takes is bounded against the CHILD's background.

The parent's analysis is bounded so that the parent's background plus its
increment is non-negative.  The child's background is a different field --
its own 1 km state, evolved since the nest was born -- and the same
correction added to that can land a positive-definite species below zero
without any arithmetic noise being involved.  Nothing bounded it: the only
thing between the correction and the child's pre-leg health gate was a clamp
deliberately held to rounding scale, which is right for rounding and silent
about this.

Measured on the card, on an 80-level cycle: water vapour at
-1.014e-5 kg kg-1 on the child's boundary row at leg 1, refused by the
child's own gate.  That is four orders of magnitude deeper than the
rounding clamp's floor, so the clamp correctly declined to hide it and the
leg stopped.

The bound is the run's OWN positivity policy, the one the parent's analysis
already went through, applied against the child's background and counted in
the child's receipt.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "da_cycle_prepared", _ROOT / "tools" / "da_cycle_prepared.py")
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


class _Child:
    """A child state with the fields this correction names."""

    def __init__(self, **fields):
        for name, values in fields.items():
            setattr(self, name, values)


def _field(value, shape=(2, 3, 3)):
    return np.full(shape, value, dtype=np.float32)


def test_a_correction_that_drives_the_child_negative_is_clipped():
    child = _Child(qv=_field(4.0e-6), thp=_field(300.0))
    correction = {"qv": _field(-1.0144e-5), "thp": _field(-0.5)}
    bounded, receipt = driver.bound_child_correction(
        child, correction, policy="clip", array_module=np)
    analysis = (child.qv.astype(np.float64)
                + np.asarray(bounded["qv"], dtype=np.float64))
    assert analysis.min() >= 0.0, (
        f"the child is handed {analysis.min():.3e} kg/kg of water vapour: "
        "its own pre-leg gate refuses that")
    assert receipt is not None and receipt["negative_points"] > 0


def test_a_field_with_no_positivity_constraint_is_left_alone():
    child = _Child(qv=_field(1.0e-2), thp=_field(300.0))
    correction = {"qv": _field(-1.0e-3), "thp": _field(-0.5)}
    bounded, _receipt = driver.bound_child_correction(
        child, correction, policy="clip", array_module=np)
    assert np.asarray(bounded["thp"]).min() == pytest.approx(-0.5), (
        "a theta correction has no positivity constraint and inventing "
        "one would be a bug that looks like caution")


def test_a_correction_the_child_can_take_is_returned_unchanged():
    child = _Child(qv=_field(1.0e-2))
    correction = {"qv": _field(-1.0e-3)}
    bounded, receipt = driver.bound_child_correction(
        child, correction, policy="clip", array_module=np)
    assert np.asarray(bounded["qv"]).min() == pytest.approx(-1.0e-3)
    assert receipt["negative_points"] == 0


def test_a_run_that_stated_no_policy_gets_no_policy_here_either():
    child = _Child(qv=_field(4.0e-6))
    correction = {"qv": _field(-1.0144e-5)}
    bounded, receipt = driver.bound_child_correction(
        child, correction, policy=None, array_module=np)
    assert receipt is None
    assert np.asarray(bounded["qv"]).min() == pytest.approx(-1.0144e-5)


def test_a_field_the_child_does_not_carry_is_not_asked_about():
    child = _Child(qv=_field(1.0e-2))
    correction = {"qv": _field(-1.0e-3), "qg": _field(-1.0)}
    bounded, receipt = driver.bound_child_correction(
        child, correction, policy="clip", array_module=np)
    assert "qg" in bounded
    assert "qg" not in receipt["constrained_fields"]


def test_the_receipt_counts_what_the_bound_added():
    child = _Child(qv=_field(4.0e-6))
    correction = {"qv": _field(-1.0144e-5)}
    _bounded, receipt = driver.bound_child_correction(
        child, correction, policy="clip", array_module=np)
    cells = 2 * 3 * 3
    assert receipt["negative_points"] == cells
    assert receipt["mass_added_by_clip"] == pytest.approx(
        cells * (1.0144e-5 - 4.0e-6), rel=1e-3)
