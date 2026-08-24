"""The positive-definite fix-up a parent-SINT birth needs, and its limit.

``parent_only_init`` fills a newborn nest by interpolating its parent.
Interpolation is a weighted sum of float32 donors, so a field that is
non-negative everywhere in the parent can land a few units in the last
place BELOW zero in the child.  The health gate holds number
concentrations to ``>= 0`` and refuses the leg, which is how a measured
``nr`` of ``-2**-31`` -- one part in ten billion of a single particle
per kilogram -- stopped a run at the instant its nest was born.

The fix-up is bounded on purpose.  A clamp with no ceiling would absorb
a genuinely broken interpolation, which is the one thing the health gate
exists to catch, so the tests below pin BOTH directions: a
rounding-scale undershoot is set to zero and counted, and anything
larger is left exactly where it is for the gate to refuse.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core.health import rule_for_field
from gpuwm.ingest.nest_init import (POSITIVE_DEFINITE_MOMENTS,
                                    clamp_parent_sint_undershoot,
                                    positive_definite_clamp_tolerance)

#: The value measured on the card when a spawned nest's first leg was
#: refused: float32's smallest normal-scale rounding artefact.
OBSERVED = -(2.0 ** -31)


class _Child:
    """The attribute surface the clamp reads: named arrays, or None."""

    def __init__(self, **fields):
        for name in POSITIVE_DEFINITE_MOMENTS:
            setattr(self, name, None)
        for name, values in fields.items():
            setattr(self, name, values)


def _field(*values, peak=1.0e3):
    """One moment field carrying ``values`` beside a realistic peak."""
    return np.array([peak, 0.0, *values], dtype=np.float32)


# ---------------------------------------------------------------------------
# The membership question: what the clamp covers, and why
# ---------------------------------------------------------------------------

def test_every_clamped_field_is_bounded_at_zero_by_the_health_gate():
    """The clamp and the gate must not disagree about the rule.

    If a field in this list were not bounded at zero below, clamping it
    would be changing physics rather than removing a rounding artefact.
    """
    for name in POSITIVE_DEFINITE_MOMENTS:
        rule = rule_for_field(name)
        assert rule.status_class == "moment"
        assert rule.lower == 0.0


# ---------------------------------------------------------------------------
# Direction one: a rounding-scale undershoot is absorbed and counted
# ---------------------------------------------------------------------------

def test_the_measured_undershoot_clamps_to_zero():
    child = _Child(nr=_field(OBSERVED))
    report = clamp_parent_sint_undershoot(child)
    assert float(child.nr.min()) == 0.0
    assert report["nr"]["cells"] == 1
    assert report["nr"]["most_negative"] == pytest.approx(OBSERVED)


def test_a_field_with_no_undershoot_is_not_reported_at_all():
    """Silence is the identity path: no entry, nothing to explain."""
    child = _Child(nr=_field(1.0, 2.0))
    assert clamp_parent_sint_undershoot(child) == {}


def test_absent_fields_are_skipped():
    """A scheme that predicts no number concentrations carries none."""
    assert clamp_parent_sint_undershoot(_Child()) == {}


def test_every_declared_field_is_clamped_not_just_the_first():
    child = _Child(nr=_field(OBSERVED), ni=_field(OBSERVED))
    report = clamp_parent_sint_undershoot(child)
    assert sorted(report) == ["ni", "nr"]


# ---------------------------------------------------------------------------
# Direction two: a broken interpolation still reaches the health gate
# ---------------------------------------------------------------------------

def test_a_broken_interpolation_is_left_for_the_health_gate():
    """Half a particle per kilogram is not a rounding artefact.

    Against a peak of 1e3 the tolerance is under a thousandth, so -0.5
    survives the clamp untouched and the gate refuses the leg -- which
    is the outcome a genuinely wrong interpolation must produce.
    """
    child = _Child(nr=_field(-0.5))
    report = clamp_parent_sint_undershoot(child)
    assert float(child.nr.min()) == pytest.approx(-0.5)
    assert report == {}


def test_a_mixed_field_clamps_only_the_rounding_scale_values():
    child = _Child(nr=_field(OBSERVED, -0.5))
    report = clamp_parent_sint_undershoot(child)
    assert report["nr"]["cells"] == 1
    assert float(child.nr.min()) == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# The tolerance is scaled to the field, because the rounding error is
# ---------------------------------------------------------------------------

def test_the_tolerance_scales_with_the_fields_own_magnitude():
    """A SINT's rounding error tracks the LARGEST donor it summed, so a
    fixed absolute budget would be too tight for a field of 1e9 and far
    too loose for one of 1e-3."""
    assert (positive_definite_clamp_tolerance(1.0e9)
            > positive_definite_clamp_tolerance(1.0e3))


def test_an_all_zero_field_still_has_an_absolute_floor():
    """The measured case: the field was zero everywhere and the peak
    carried no scale at all, so the floor is what caught it."""
    tol = positive_definite_clamp_tolerance(0.0)
    assert tol > 0.0
    assert tol > abs(OBSERVED)
    child = _Child(nr=np.array([0.0, OBSERVED], dtype=np.float32))
    assert clamp_parent_sint_undershoot(child)["nr"]["cells"] == 1


def test_the_floor_stays_below_one_particle_per_kilogram():
    """A clamp that could erase a countable particle would be physics."""
    assert positive_definite_clamp_tolerance(0.0) < 1.0
