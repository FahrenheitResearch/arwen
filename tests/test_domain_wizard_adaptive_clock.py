"""`gpuwm domain --clock`: the adaptive clock through the existing config keys.

The wizard writes ``use_adaptive_time_step`` and nothing else, so each
domain keeps WRF's own per-spacing bounds and its configured first step,
and the terrain clock's launch-time ceiling still lands on the domains
that need it.  ``auto`` picks adaptive only where both tables cover the
grid; ``adaptive`` on a grid its bounds cannot hold is refused by name.
"""

import argparse
from datetime import datetime
from fractions import Fraction

import pytest

from gpuwm import domain_wizard as wizard
from gpuwm.domain_wizard import experiment_from_text


def render(*, lat=35.3, dx=3000., ratios=(), clock=None):
    kwargs = {} if clock is None else {"clock": clock}
    return wizard.render_config(
        name="clock", start_time=datetime(2026, 9, 5), hours=6,
        projection=wizard._projection_entries(lat, -97.5, "auto"),
        dims=[(64, 64)] + [(48, 48)] * len(ratios), ratios=ratios,
        fetch_hints=wizard._candidate_fetch_hints("gfs"), case_data=None,
        root_dx_m=dx, **kwargs)


def first_step(domain):
    return Fraction(domain.time_step) + Fraction(
        domain.time_step_fract_num, domain.time_step_fract_den)


def test_adaptive_writes_the_one_key_and_keeps_every_bound_per_domain():
    fixed = experiment_from_text(render(ratios=(3,)), source="<fixed>")
    text = render(ratios=(3,), clock="adaptive")
    exp = experiment_from_text(text, source="<adaptive>")
    assert "use_adaptive_time_step = true" in text
    assert "# CLOCK: adaptive." in text
    assert all(dc.run.use_adaptive_time_step for dc in exp.domains)
    assert exp.root.run.step_to_output_time is True
    # No clamp is written, in [shared] or per domain: each keeps WRF's
    # -1 fill-in for its own spacing and its configured first step.
    for dc in exp.domains:
        for key in ("starting_time_step", "max_time_step", "min_time_step"):
            assert getattr(dc.run, key) == -1, (dc.grid_id, key)
    assert first_step(exp.root) == first_step(fixed.root) == 15
    assert [dc.run.dt for dc in exp.domains] == [dc.run.dt for dc in fixed.domains]


def test_the_library_default_and_fixed_keep_the_fixed_bytes():
    assert render() == render(clock="fixed")
    assert "use_adaptive_time_step" not in render()
    assert "CLOCK:" not in render(clock="fixed")


@pytest.mark.parametrize("lat,dx,ratios,adaptive", [
    (35.3, 3000., (), True),
    (35.3, 12000., (4,), True),
    (35.3, 3000., (3, 2), True),
    (35.3, 1000., (2,), True),
    # Past the terrain clock's measured spacings (500 m to 12 km).
    (35.3, 1000., (4,), False),
    (35.3, 48000., (), False),
    # The tropical clock's 2.5 s per km starts under the 3 s per km floor.
    (14., 3000., (), False),
])
def test_auto_is_adaptive_only_where_both_tables_cover_the_grid(lat, dx, ratios, adaptive):
    text = render(lat=lat, dx=dx, ratios=ratios, clock="auto")
    exp = experiment_from_text(text, source="<auto>")
    assert exp.root.run.use_adaptive_time_step is adaptive
    chosen, why = wizard.clock_decision(
        "auto", time_step=first_step(exp.root), root_dx_m=dx, ratios=ratios)
    assert chosen is adaptive
    assert why.startswith("auto chose adaptive" if adaptive else "auto kept the fixed step")


def test_adaptive_is_refused_on_the_tropical_clock_naming_the_grid():
    with pytest.raises(ValueError) as refused:
        render(lat=14., dx=3000., clock="adaptive")
    message = str(refused.value)
    assert "d01 at 3000 m starts at 7.5 s" in message
    assert "9..24 s" in message and "--clock fixed" in message


def test_adaptive_is_refused_where_the_floor_rounds_to_zero():
    with pytest.raises(ValueError, match=r"d01 at 100 m .* outside the 0\.\.1 s"):
        render(dx=100., clock="adaptive")


def test_the_terrain_clock_still_caps_a_wizard_adaptive_domain():
    from gpuwm.terrain_clock import CrestWind, derive_clock, retime_experiment

    exp = experiment_from_text(render(clock="adaptive"), source="<adaptive>")
    crest = CrestWind(label="d01", crest_height_m=4500., wind_m_s=50.,
                      when="start", source="d01", height_m=4600.)
    clock = derive_clock(1, exp.root.run, exp.dt_exact(1), 0.41, crest)
    # WRF's fill-in ceiling at 3 km is 24 s; steep ground under a strong
    # crest-level wind holds less, and the rule writes that as the cap.
    assert clock.ceiling is not None and clock.ceiling < 24
    retimed, _ = retime_experiment(exp, {1: clock.division},
                                   {1: clock.time_step_sound},
                                   {1: clock.ceiling})
    assert Fraction(retimed.root.run.max_time_step,
                    retimed.root.run.max_time_step_den or 1) == clock.ceiling
    assert retimed.root.run.use_adaptive_time_step


def _door(tmp_path, *extra, explain=False):
    parser = argparse.ArgumentParser()
    wizard.register_cli(parser.add_subparsers())
    out = tmp_path / "forecast.toml"
    argv = ["domain", "--source", "gfs", "--cycle", "2026-09-05T00",
            "--hours", "6", "--root-dx", "3", "--point", "35.3,-97.5",
            "--point-extent-km", "300", "--vram-gib", "16", "--tiles", "off",
            "--out", str(out), *extra]
    args = parser.parse_args(argv)
    # --explain belongs to the top-level parser; the door reads it off args.
    args.explain = explain
    assert wizard.domain_main(args) == 0
    return experiment_from_text(out.read_text(), source=str(out))


@pytest.mark.parametrize("flag,adaptive", [((), True), (("--clock", "adaptive"), True),
                                           (("--clock", "fixed"), False)])
def test_the_door_writes_the_clock_it_was_asked_for(tmp_path, capsys, flag, adaptive):
    exp = _door(tmp_path, *flag)
    assert exp.root.run.use_adaptive_time_step is adaptive
    headline = next(line for line in capsys.readouterr().out.splitlines()
                    if line.startswith("gpuwm domain: '"))
    assert headline.endswith(", adaptive time step") is adaptive


def test_explain_says_why_the_clock_was_chosen(tmp_path, capsys):
    _door(tmp_path, "--clock", "auto", explain=True)
    said = capsys.readouterr().out
    assert ("domain: adaptive time step (auto chose adaptive: every grid "
            "starts inside the adaptive clock's bounds") in said


def test_a_run_plan_intent_carries_the_clock_to_the_door(tmp_path):
    from gpuwm.runplan import _INTENT_DELIVERY, intent_arguments

    assert _INTENT_DELIVERY["clock"] == "config"
    argv = intent_arguments({"clock": "fixed", "source": "gfs"}, out=tmp_path / "x.toml")
    assert argv[argv.index("--clock") + 1] == "fixed"


def _drafts():
    from gpuwm.gui.api import CreateMixin

    class Drafts(CreateMixin):
        def _offered(self):
            return {"sources": [{"id": "gfs", "route": "prepared"}]}

        def availability_of(self, *args, **kwargs):
            return {"starts": "yes"}

    return Drafts()


def _payload(**extra):
    return {"name": "clock-run", "source": "gfs", "cycle": "2026-09-24T00",
            "card": "8gb", "lat": 35, "lon": -100, **extra}


def test_new_forecast_passes_its_clock_choice_into_the_plan(tmp_path):
    from gpuwm.gui.api import CreateMixin

    draft = _drafts().draft(_payload(clock="adaptive"), need_name=True)
    intent = CreateMixin.plan_document(draft, tmp_path)["config"]["intent"]
    assert intent["clock"] == "adaptive"
    left = _drafts().draft(_payload(), need_name=True)
    assert "clock" not in CreateMixin.plan_document(left, tmp_path)["config"]["intent"]


def test_new_forecast_refuses_a_clock_the_door_does_not_take():
    from gpuwm.gui.api import ApiError

    with pytest.raises(ApiError, match="adaptive or fixed"):
        _drafts().draft(_payload(clock="sometimes"), need_name=True)


def test_the_storm_following_route_refuses_a_clock_choice_by_name():
    # The cyclone setup writes its own configuration, clock included, so
    # a choice beside it would be written into the plan and never run.
    from gpuwm.gui.api import ApiError

    with pytest.raises(ApiError) as refused:
        _drafts().draft(_payload(clock="adaptive", following=True), need_name=True)
    assert "a time step choice would not be used" in str(refused.value)
