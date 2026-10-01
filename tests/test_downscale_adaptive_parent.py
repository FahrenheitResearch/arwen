"""A child of a parent that ran the adaptive clock takes a fixed step of its own.

The parent's checkpoint echoes the step its adaptive clock was on when the set
was written; divided by the ratio that made the child's step a property of the
checkpoint instant (a 3 km West Coast parent's 30-minute set carried 11.25 s,
so the 1 km child took 3.75 s) and, on a calm hour, a step no 900 s history
interval is a whole number of, which the child's clock contract refuses with no
way out on the --point route.  These pin the replacement: the engine's clock
convention at the child's spacing, down to the largest step every clock the
child keeps is a whole number of, and a child config that says it runs fixed.
"""

from fractions import Fraction

import pytest

from gpuwm.config import RunConfig
from gpuwm.downscale import _derive_child_run_config, adaptive_parent_child_step
from gpuwm.offline_child_run import child_cadence

_ADAPTIVE_PARENT = {
    "nx": 360, "ny": 640, "nz": 4, "dx": 3000.0, "dy": 3000.0,
    "ztop": 9000.0, "dt": 11.25, "run_seconds": 172800.0,
    "output_interval_s": 900.0, "restart_interval_s": 86400.0,
    "hybrid_opt": 2, "etac": 0.2, "hypsometric_opt": 2, "moist": True,
    "mp_physics": 8, "specified": True, "nested": False, "terrain_opt": 1,
    "map_proj": 1, "grid_id": 1, "time_step_sound": 4, "spec_bdy_width": 5,
    "spec_zone": 1, "relax_zone": 4, "use_adaptive_time_step": True,
}
_GEOMETRY = {"nx": 360, "ny": 640, "dx": 3000.0, "dy": 3000.0}


def _child(parent_config, *, ratio=3, centre_lat=34.2, run_seconds=172800.0,
           output_interval_s=900.0, health=60.0):
    return _derive_child_run_config(
        parent_config, parent=dict(_GEOMETRY), ratio=ratio, child_nx=180,
        child_ny=150, run_seconds=run_seconds,
        output_interval_s=output_interval_s, centre_lat=centre_lat,
        clock_seconds=(health,))


def test_the_child_step_does_not_depend_on_the_parents_live_step():
    steps = {_child(dict(_ADAPTIVE_PARENT, dt=live))["dt"]
             for live in (11.25, 13.5, 27.0, 37.21)}
    assert steps == {5.0}


def test_a_live_step_the_clock_contract_refused_now_runs():
    # 37.21 / 3 = 12.403 s: 900 s is not a whole number of it.  The derived
    # child must pass the same cadence check the runner integrates on.
    merged = _child(dict(_ADAPTIVE_PARENT, dt=37.21))
    clock = child_cadence(RunConfig(**merged), health_interval_seconds=60.0)
    assert clock.output_steps == 180
    assert clock.restart_steps == 17280
    assert merged["use_adaptive_time_step"] is False


def test_a_fixed_parent_keeps_its_step_over_the_ratio():
    merged = _child(dict(_ADAPTIVE_PARENT, dt=15.0, use_adaptive_time_step=False))
    assert merged["dt"] == 5.0
    merged = _child(dict(_ADAPTIVE_PARENT, dt=18.0, use_adaptive_time_step=False))
    assert merged["dt"] == 6.0


def test_the_tropics_take_half_the_step():
    assert _child(dict(_ADAPTIVE_PARENT), centre_lat=10.0)["dt"] == 2.5


@pytest.mark.parametrize("output", [900.0, 600.0, 280.0, 77.0])
def test_every_clock_the_child_keeps_is_a_whole_number_of_steps(output):
    merged = _child(dict(_ADAPTIVE_PARENT), output_interval_s=output,
                    run_seconds=output * 8)
    dt = Fraction(merged["dt"]).limit_denominator(10 ** 6)
    assert dt <= 5
    for clock in (output, output * 8, 86400.0, 60.0):
        assert (Fraction(clock) / dt).denominator == 1
    child_cadence(RunConfig(**merged), health_interval_seconds=60.0)


def test_a_third_of_a_kilometre_lands_on_five_thirds_of_a_second():
    step = adaptive_parent_child_step(child_dx=1000.0 / 3, child_dy=1000.0 / 3,
                                      centre_lat=34.0,
                                      clocks=(3600.0, 900.0, 0.0, 60.0))
    assert Fraction(step).limit_denominator(1000) == Fraction(5, 3)
    cfg = RunConfig(**dict(_child(dict(_ADAPTIVE_PARENT)),
                           dx=1000.0 / 3, dy=1000.0 / 3, dt=step,
                           run_seconds=3600.0))
    child_cadence(cfg, health_interval_seconds=60.0)


def test_without_a_centre_the_parents_own_latitude_decides():
    import numpy as np

    geometry = dict(_GEOMETRY, xlat=np.full((4, 4), 5.0))
    merged = _derive_child_run_config(
        dict(_ADAPTIVE_PARENT), parent=geometry, ratio=3, child_nx=180,
        child_ny=150, run_seconds=172800.0, output_interval_s=900.0)
    assert merged["dt"] == 2.5
