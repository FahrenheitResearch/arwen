"""The sweep planner's filter probe checks the knobs the arms turn on.

CPU only.  The planner runs no model: it pushes every step's argv through
the real parser of the tool that will run it and asks the real
``RadarAssimilationConfig`` whether the configuration it describes would
be accepted.  These cells pin what that probe can and cannot see, because
the audit (R-051) found it constructing a configuration with the
reflectivity arm hardcoded OFF -- so a plan whose run would assimilate
reflectivity was checked as if it would not.
"""
from __future__ import annotations

from types import SimpleNamespace

from tools.da_cycle_plan import _filter_config_problems


def _step(**updates):
    values = dict(
        horizontal_loc_m=12000.0, vertical_loc_m=3000.0, rtps_alpha=0.9,
        relaxation="rtps", thin_cells=1, err_inflation=1.0,
        z_thin_cells=1, z_err_inflation=1.0,
        cwp_thin_cells=1, cwp_err_inflation=1.0,
        positivity_policy="clip", solve_device="host",
        memory_budget_mib=512.0,
        hydrometeors=True, reflectivity_analysis=False, goes_cwp=[])
    values.update(updates)
    return SimpleNamespace(**values)


def test_a_deflated_observation_error_is_reported_before_the_arm_runs():
    """The refusal the probe was written for, unchanged.

    The filter refuses an inflation below 1 whatever the arms are, so a
    scaled per-volume setting is caught in the plan rather than
    forty-nine seconds into the arm.
    """
    problems = _filter_config_problems(_step(err_inflation=0.9129), "arm/step")
    assert len(problems) == 1, problems
    assert "would refuse this configuration" in problems[0]
    assert "deflating stated observation errors" in problems[0]


def test_a_reflectivity_step_is_checked_with_its_arm_on():
    """The arms come from argv now, and the field set cannot false-refuse.

    A reflectivity or CWP analysis against a wind-only state vector is a
    refusal the filter states, and it turns on an ARM -- so hardcoding the
    arm off meant the probe never asked it.  The field set the probe pairs
    with the arm is ``gpuwm.da.moments.DEFAULT_BASE_FIELDS``, the
    scheme-independent floor of what the driver derives per leg, so a
    plan the driver would run is never reported here.
    """
    from gpuwm.da import moments

    assert _filter_config_problems(
        _step(reflectivity_analysis=True), "arm/step") == []
    assert _filter_config_problems(
        _step(goes_cwp=["a.nc"]), "arm/step") == []
    assert set(moments.DEFAULT_BASE_FIELDS) & {"thp", "qv"}

    # And with the hydrometeor half off, the same arm IS reported.  The
    # cycling driver's own parser already refuses that pairing, so this is
    # the second net rather than the first: what it pins is that the arm
    # reaches the filter at all.
    problems = _filter_config_problems(
        _step(reflectivity_analysis=True, hydrometeors=False), "arm/step")
    assert len(problems) == 1, problems
    assert "wind-only state vector" in problems[0]


def test_the_probe_says_nothing_about_a_configuration_the_filter_accepts():
    assert _filter_config_problems(_step(), "arm/step") == []
