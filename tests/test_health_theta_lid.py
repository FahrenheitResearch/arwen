"""The health gate's theta ceiling is a function of the configured lid.

The defect: the ceiling was a flat 600 K, and 600 K is not a property of the
atmosphere -- it is a property of the atmosphere UNDER A 100 hPa LID.  Theta
grows as the dry adiabat carries a level's temperature to 1000 hPa, so the
same physically ordinary stratospheric air is ~600 K under a 100 hPa lid and
~950 K under a 20 hPa one.  A real deep-top initial state was therefore
refused by a gate that exists to catch runaway thermodynamics, and the run
never started.

What must NOT change: the admitted lid temperature.  600 K at 100 hPa is
310.6 K carried down the adiabat, and that is the number this gate always
admitted; every lid now admits exactly the same one.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.health import (
    THETA_CEILING_REFERENCE_LID_PA,
    rule_for_field,
    theta_ceiling_for_lid,
    validate_state_cpu,
)


DEFAULT_LID_PA = 10000.0
DEEP_LID_PA = 2000.0


def test_the_reference_lid_reproduces_the_ceiling_it_replaces():
    """No silent change at the lid the flat number was calibrated at."""
    assert THETA_CEILING_REFERENCE_LID_PA == DEFAULT_LID_PA
    assert theta_ceiling_for_lid(DEFAULT_LID_PA) == pytest.approx(600.0)
    assert rule_for_field("thp").upper == pytest.approx(600.0)
    assert rule_for_field("thp", p_top=None).upper == pytest.approx(600.0)


def test_a_shallower_lid_is_never_tightened_below_the_old_ceiling():
    """Opening a gate must not close it somewhere else."""
    for p_top in (DEFAULT_LID_PA, 20000.0, 50000.0):
        assert theta_ceiling_for_lid(p_top) >= 600.0


def test_the_ceiling_is_the_same_lid_temperature_at_every_lid():
    """The gate admits ONE temperature; the lid decides what theta that is."""
    kappa = 2.0 / 7.0
    for p_top in (DEEP_LID_PA, 5000.0, 1000.0, DEFAULT_LID_PA):
        admitted_temperature = (
            theta_ceiling_for_lid(p_top) * (p_top / 1.0e5) ** kappa)
        assert admitted_temperature == pytest.approx(
            600.0 * (DEFAULT_LID_PA / 1.0e5) ** kappa, rel=1e-9)


def test_a_twenty_hpa_lid_admits_ordinary_stratospheric_theta():
    """220 K at 20 hPa is 673 K of theta -- ordinary air, refused at 600."""
    ordinary = 220.0 * (1.0e5 / DEEP_LID_PA) ** (2.0 / 7.0)
    assert ordinary > 600.0
    assert theta_ceiling_for_lid(DEEP_LID_PA) > ordinary


def _deep_top_state(top_theta: float) -> SimpleNamespace:
    """A 4-level column whose top level carries ``top_theta`` under a 20 hPa lid."""
    thb = np.array([300.0, 400.0, 500.0, 600.0], dtype=np.float32)
    thp = np.zeros((4, 1, 1), dtype=np.float32)
    thp[3, 0, 0] = np.float32(top_theta - 600.0)
    return SimpleNamespace(thp=thp, thb=thb, p_top=np.float32(DEEP_LID_PA),
                           physics=None, lateral_boundaries=None, _scratch={})


def test_a_legitimate_deep_top_state_passes():
    ordinary = 220.0 * (1.0e5 / DEEP_LID_PA) ** (2.0 / 7.0)
    report = validate_state_cpu(_deep_top_state(ordinary), phase="initial")
    assert report.ok, report.reason


def test_a_corrupt_deep_top_state_still_refuses_and_names_the_lid():
    report = validate_state_cpu(_deep_top_state(5000.0), phase="initial")
    assert not report.ok
    assert report.first_bad_field == "thp"
    assert "theta" in report.failing_classes
    reason = report.reason
    assert "exceeds upper bound" in reason
    # The refusal must say WHICH ceiling it applied and why it is that one,
    # or a user reading it has no way to tell a lid-aware pass from a
    # silently loosened gate.
    assert "2000" in reason
    assert "lid" in reason


def test_a_road_whose_state_is_a_template_can_name_the_domains_lid():
    """The store-direct road gates a SLAB-HEIGHT template, not the domain.

    Its template need never have loaded a base, so the lid comes from the
    domain's base object instead; without that seam a deep-top domain would
    be gated against the 100 hPa reference ceiling on that road alone.
    """
    from gpuwm.core.health import collect_state_fields

    thb = np.array([300.0], dtype=np.float32)
    template = SimpleNamespace(thp=np.zeros((1, 1, 1), dtype=np.float32),
                               thb=thb, p_top=None, physics=None,
                               lateral_boundaries=None, _scratch={})
    fields = collect_state_fields(template, p_top=DEEP_LID_PA)
    thp = next(f for f in fields if f.name == "thp")
    assert thp.rule.upper == pytest.approx(theta_ceiling_for_lid(DEEP_LID_PA))
    assert collect_state_fields(template)[0].rule.upper == pytest.approx(600.0)


def test_a_state_with_no_lid_keeps_the_calibrated_ceiling():
    """p_top is unset until the base state loads; fall back, never guess."""
    thb = np.array([300.0], dtype=np.float32)
    thp = np.array([[[301.0]]], dtype=np.float32)
    state = SimpleNamespace(thp=thp, thb=thb, p_top=None, physics=None,
                            lateral_boundaries=None, _scratch={})
    report = validate_state_cpu(state)
    assert not report.ok
    assert "600.0" in report.reason
