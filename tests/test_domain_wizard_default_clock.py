"""An author must not derive a clock that contradicts requested event times."""
from datetime import datetime
from fractions import Fraction
import argparse
import json
import tomllib
import pytest
from gpuwm import domain_wizard as wizard
from gpuwm.domain_wizard import experiment_from_text


def render(*, lat=35.3, dx=45125., ratios=(), history=720., child_history=720., hours=6):
    return wizard.render_config(
        name="default clock", start_time=datetime(2026, 9, 5), hours=hours,
        projection=wizard._projection_entries(lat, -97.5, "auto"),
        dims=[(64,64)] + [(48,48)] * len(ratios), ratios=ratios,
        fetch_hints=wizard._candidate_fetch_hints("gfs"), case_data=None,
        root_dx_m=dx, history_interval_s=history,
        nest_history_interval_s=child_history)


def clock(domain):
    return Fraction(domain.time_step) + Fraction(domain.time_step_fract_num,
                                                 domain.time_step_fract_den)


@pytest.mark.parametrize("hours,checkpoint", [(6,3600.), (1,3600.), (.5,1800.)])
@pytest.mark.parametrize("ratios", [(), (3,)])
def test_new_checkpoint_schedule_uses_the_actual_event_clock_and_preserves_off(hours,checkpoint,ratios):
    text=render(hours=hours,ratios=ratios)
    exp=experiment_from_text(text,source="<new checkpoint defaults>")
    assert exp.restart_interval_s==checkpoint
    assert (Fraction(checkpoint)/clock(exp.root)).denominator==1
    assert (Fraction(str(exp.run_seconds))/clock(exp.root)).denominator==1
    assert exp.root.history_interval_s==720.
    assert exp.root.run.dx==45125.
    off=text.replace(f"restart_interval_s = {checkpoint}","restart_interval_s = 0.0")
    assert experiment_from_text(off,source="<explicit checkpoints off>").restart_interval_s==0.


def test_fractional_spacing_keeps_requested_geometry_and_event_times():
    text=render()
    exp=experiment_from_text(text, source="<actual wizard>")
    assert exp.domains[0].run.dx == 45125.
    assert exp.domains[0].history_interval_s == 720.
    assert exp.run_seconds == 21600.
    assert clock(exp.domains[0]) == 60
    assert "adjusted 225.625 -> 60 s" in text
    assert wizard.root_time_step_s(35.3,45125.) == Fraction(1805,8)


@pytest.mark.parametrize("lat,dx,history", [(35.3,12000.,3600.),(35.3,9000.,3600.),
                                         (14.,3000.,900.),(35.3,48000.,720.)])
def test_already_compatible_defaults_keep_the_exact_clock(lat,dx,history):
    text=render(lat=lat,dx=dx,history=history)
    exp=experiment_from_text(text,source="<unchanged default>")
    assert clock(exp.domains[0]) == wizard.root_time_step_s(lat,dx)
    assert "derived root time step adjusted" not in text


def test_nested_event_clock_uses_the_complete_ratio_chain_and_restart():
    selected=wizard.derived_time_step_s(35.3,45125.,run_seconds=3600.,
        ratios=(3,4),history_interval_s=720.,nest_history_interval_s=7.,
        restart_interval_s=60.)
    assert selected == 3
    for interval in (Fraction(3600),Fraction(720),Fraction(60),
                     Fraction(7)*3,Fraction(7)*12):
        assert (interval/selected).denominator==1
    exp=experiment_from_text(render(ratios=(3,),child_history=7.),source="<child events>")
    assert clock(exp.domains[0]) == 3
    assert exp.domains[1].history_interval_s == 7.
    assert exp.restart_interval_s == 3600.


def test_explicit_authored_incompatible_clock_still_refuses_without_rewriting():
    text=render().replace("time_step = 60", "time_step = 225\ntime_step_fract_num = 5\ntime_step_fract_den = 8")
    with pytest.raises(ValueError,match="history_interval_s"):
        experiment_from_text(text,source="<explicit user clock>")
    assert tomllib.loads(text)["domain"][0]["time_step"] == 225
    compatible=render().replace("time_step = 60", "time_step = 30")
    assert clock(experiment_from_text(compatible,source="<explicit valid clock>").domains[0])==30


@pytest.mark.parametrize("door", ["point","polygon"])
def test_public_wizard_prices_and_emits_the_same_requested_event_clock(tmp_path,door,capsys):
    parser=argparse.ArgumentParser()
    wizard.register_cli(parser.add_subparsers())
    out=tmp_path/"forecast.toml"
    argv=["domain","--source","gfs","--cycle","2026-09-05T00", "--hours","6",
          "--root-dx","45.125","--history-interval","720",
          "--nest-history-interval","720","--vram-gib",("5" if door=="point" else "16"), "--tiles","off",
          "--out",str(out)]
    if door=="point":
        argv += ["--point","35.3,-97.5"]
    else:
        polygon=tmp_path/"region.geojson"
        polygon.write_text(json.dumps({"type":"Polygon","coordinates":[[
            [-118.,28.],[-77.,28.],[-77.,42.],[-118.,42.],[-118.,28.]]]}))
        argv += ["--polygon",str(polygon)]
    assert wizard.domain_main(parser.parse_args(argv)) == 0
    text=out.read_text()
    exp=experiment_from_text(text,source=str(out))
    assert exp.domains[0].run.dx==45125.
    assert exp.domains[0].history_interval_s==720.
    assert clock(exp.domains[0])==60
    assert "adjusted 225.625 -> 60 s" in capsys.readouterr().out


def test_unconstrained_non_decimal_clock_would_break_original_physics_periods():
    # The closest event-only clock 720/11 changes KF5min ->12min in the old
    # cadence author. Bind the original period instead of changing physics.
    old, _ = wizard.snap_cadences_to_clock(Fraction(720,11),
        {"radt":12.,"cudt_minutes":5.,"cu_physics":1})
    assert old["cudt_minutes"]==12.
    exp=experiment_from_text(render(dx=13125.),source="<original physics periods>")
    assert clock(exp.domains[0])==60
    assert exp.domains[0].run.cudt_minutes==5.
    assert exp.domains[0].run.radt==12.


def test_non_decimal_clock_round_trips_when_original_periods_are_bound():
    exp=experiment_from_text(render(dx=3311.),source="<rational eleventh>")
    assert clock(exp.domains[0])==Fraction(180,11)
    assert exp.domains[0].run.radt==12.
    assert exp.domains[0].run.cu_physics==0
    assert wizard.derived_time_step_s(35.3,12000.,run_seconds=1000.,ratios=(),
        history_interval_s=720.,restart_interval_s=60.)==20


def test_named_suite_preserves_physics_where_omitted_suite_reconciled_defaults():
    kwargs=dict(name="named",start_time=datetime(2026,9,5),hours=6,
        projection=wizard._projection_entries(35.3,-97.5,"auto"), dims=[(64,64)],
        ratios=(),fetch_hints=wizard._candidate_fetch_hints("gfs"),case_data=None,
        root_dx_m=9000.,profile=wizard.DEFAULT_PHYSICS_PROFILE)
    implicit=wizard.render_config(**kwargs)
    explicit=wizard.render_config(**kwargs,cumulus_requested=True)
    a=experiment_from_text(implicit,source="<omitted profile>")
    b=experiment_from_text(explicit,source="<named profile>")
    assert clock(a.root)==45 and a.root.run.cudt_minutes==5.25
    assert "cudt_minutes adjusted 5 -> 5.25" in implicit
    assert clock(b.root)==30 and b.root.run.cudt_minutes==5.
    assert b.root.run.radt==12.
    assert "cudt_minutes adjusted" not in explicit


def test_selected_clock_is_the_largest_exact_choice_preserving_active_periods():
    exp=experiment_from_text(render(),source="<closest exact clock>")
    dt=clock(exp.root)
    limit=wizard.root_time_step_s(35.3,45125.)
    periods=(Fraction(720),Fraction(300),Fraction(21600))
    assert dt==60 and all((period/dt).denominator==1 for period in periods)
    # Every larger output-compatible clock below the recommendation fails
    # one of the original physics periods; no guessed performance threshold.
    for count in range(1,int(Fraction(720)/dt)):
        candidate=Fraction(720,count)
        if candidate <= limit:
            assert any((period/candidate).denominator!=1 for period in periods)


def test_active_pbl_period_constrains_the_omitted_clock_without_changing_it(monkeypatch):
    original=wizard.shared_physics
    def selected(profile):
        result=original(profile)
        result['bldt']=1.75
        return result
    monkeypatch.setattr(wizard,'shared_physics',selected)
    text=render()
    exp=experiment_from_text(text,source="<PBL period>")
    assert clock(exp.root)==15
    assert exp.root.run.bldt==1.75
    assert exp.root.run.cudt_minutes==5.
    assert "active physics periods: 105, 300, 720 s" in text


def test_active_radiation_uses_the_same_legacy_alias_precedence_as_the_loader():
    physics={'radt':0.,'cu_physics':0}
    shared={'ra_lw_physics':4,'ra_sw_physics':4,'radt_minutes':1.75}
    assert wizard._root_physics_periods(physics,shared)==(Fraction(105),)
    assert wizard._root_physics_periods(dict(physics,radt=2.),shared)==(Fraction(120),)
    assert wizard._root_physics_periods(physics,dict(shared,ra_lw_physics=0,ra_sw_physics=0))==()
