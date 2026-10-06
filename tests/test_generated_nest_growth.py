"""Generated nests declare child growth without changing targets or bounds."""
from datetime import datetime
from fractions import Fraction
from types import SimpleNamespace
import tomllib

import pytest

from gpuwm import domain_wizard as wizard
from gpuwm.core.adaptive_clock import AdaptiveClockDriver
from gpuwm.core.clock import resolve_clock


def emitted(ratios=(), clock="adaptive"):
    return wizard.render_config(name="generated-growth", start_time=datetime(2026, 10, 4),
        hours=1, projection=wizard._projection_entries(35.5, -97.5, "auto"),
        dims=[(64, 64)]+[(48, 48)]*len(ratios), ratios=ratios,
        fetch_hints=wizard._candidate_fetch_hints("gfs"), case_data=None,
        root_dx_m=12000., clock=clock)


@pytest.mark.parametrize("clock", ["auto", "adaptive"])
@pytest.mark.parametrize("ratios", [(3,), (3, 3)])
def test_nested_growth_is_explicit_and_other_controller_controls_are_retained(clock, ratios):
    text = emitted(ratios, clock)
    raw = tomllib.loads(text)
    assert raw["shared"]["max_step_increase_pct"] == 5
    assert not {"target_cfl", "target_hcfl", "max_time_step"}.intersection(raw["shared"])
    experiment = wizard.experiment_from_text(text, source="generated-growth")
    for domain in experiment.domains:
        assert domain.run.max_step_increase_pct == 5
        assert (domain.run.target_cfl, domain.run.target_hcfl) == (1.2, .84)
        assert domain.run.max_time_step == -1
        assert "max_time_step" not in raw["domain"][domain.grid_id-1]


@pytest.mark.parametrize("ratios,clock", [((), "adaptive"), ((), "auto"), ((3,), "fixed")])
def test_growth_emission_does_not_extend_to_single_domain_or_fixed(ratios, clock):
    assert "max_step_increase_pct" not in emitted(ratios, clock)


def test_real_nested_controllers_use_five_percent_and_keep_prior_targets_and_caps():
    experiment = wizard.experiment_from_text(emitted((3,)), source="generated-growth")
    calendar = resolve_clock(experiment, lbc_interval_s=3600.)
    clocks = calendar.clocks()
    nodes = {domain.grid_id: SimpleNamespace(cfg=domain, clock=clocks[domain.grid_id],
                                            state=None, children=[]) for domain in experiment.domains}
    nodes[1].children.append(nodes[2])
    model = SimpleNamespace(root=nodes[1], node=lambda grid_id: nodes[grid_id])
    driver = AdaptiveClockDriver(model, cfl_source=lambda grid_id: (0., 0.),
        tick_den=calendar.tick_den, map_factor_source=lambda grid_id: 1.)
    expected = {1: (60, 63, 96), 2: (20, 21, 32)}
    for grid_id, controller in driver.controllers.items():
        initial, next_step, cap = expected[grid_id]
        assert controller.first_step() == Fraction(initial)
        assert controller.next_dt(max_vert_cfl=0., max_horiz_cfl=0.) == Fraction(next_step)
        assert controller.max_dt == Fraction(cap)
        assert controller.max_increase_factor == pytest.approx(1.05)
        assert (controller.target_cfl, controller.target_hcfl) == (1.2, .84)
