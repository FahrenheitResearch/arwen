"""The plan-time advisory for an adaptive nest whose step may grow fast.

The breakage it names: a 500 m child at max_step_increase_pct = 51 took a
freshly measured CFL after an output alarm, grew to about three times its
spacing rule and went non-finite hours into a nested forecast; the same
forecast at 5 percent completed at the same pace.  The advisory must fire
for exactly that configuration and stay silent for every other one.
"""

from types import SimpleNamespace

from gpuwm.core.preflight import (GENERATED_NEST_STEP_GROWTH_PCT,
                                  nest_step_growth_advisories)


def _domain(grid_id, parent_id, *, adaptive=True, growth=5):
    return SimpleNamespace(
        grid_id=grid_id, parent_id=parent_id,
        run=SimpleNamespace(use_adaptive_time_step=adaptive,
                            max_step_increase_pct=growth))


def _exp(*domains):
    return SimpleNamespace(domains=tuple(domains))


def test_a_nest_at_the_wrf_nest_growth_is_named_with_the_remedy():
    lines = nest_step_growth_advisories(
        _exp(_domain(1, 0, growth=5), _domain(2, 1, growth=51)))
    assert len(lines) == 1
    (line,) = lines
    assert line.startswith("d02 ")
    assert "max_step_increase_pct = 51" in line
    assert "non-finite" in line
    assert f"max_step_increase_pct = {GENERATED_NEST_STEP_GROWTH_PCT}" in line


def test_every_fast_nest_gets_its_own_line_and_the_root_never_does():
    lines = nest_step_growth_advisories(_exp(
        _domain(1, 0, growth=51), _domain(2, 1, growth=51),
        _domain(3, 2, growth=20)))
    assert [line[:3] for line in lines] == ["d02", "d03"]


def test_the_generated_default_and_slower_growth_are_silent():
    assert GENERATED_NEST_STEP_GROWTH_PCT == 5
    assert nest_step_growth_advisories(_exp(
        _domain(1, 0), _domain(2, 1, growth=5), _domain(3, 2, growth=3))) == []


def test_a_fixed_step_nest_is_silent_whatever_its_growth_field_says():
    assert nest_step_growth_advisories(_exp(
        _domain(1, 0, adaptive=False),
        _domain(2, 1, adaptive=False, growth=51))) == []


def test_the_advisory_carries_no_place_or_case_name():
    (line,) = nest_step_growth_advisories(
        _exp(_domain(1, 0), _domain(2, 1, growth=51)))
    lowered = line.lower()
    for token in ("toronto", "boston", "hrrr", "ohio"):
        assert token not in lowered
