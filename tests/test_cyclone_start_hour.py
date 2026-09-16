"""The cyclone door starts at a forecast lead, not only at the analysis.

The lead grammar is the FETCH ROUTE's, so these tests ask every registered
source that publishes leads the same questions and name no model in a
default.  A source that publishes none must say so rather than silently
initialise from its analysis.
"""
from datetime import datetime, timedelta
import argparse
import json
import tomllib

import pytest

from gpuwm import cyclone_setup as tc
from gpuwm import cyclone_sources as cs
from gpuwm import domain_wizard as dw
from gpuwm.source_adapters import get_source_adapter

CYCLE = "2026090900"
MOMENT = datetime(2026, 9, 9, 0)
POINT = (18., -65.)


def lead_sources():
    """Registered cyclone sources that publish forecast leads at all."""
    return [source for source in cs.source_ids()
            if get_source_adapter(source).max_forecast_hour > 0]


def analysis_only_sources():
    return [source for source in cs.source_ids()
            if get_source_adapter(source).max_forecast_hour == 0]


def test_the_default_is_the_analysis_and_is_byte_identical_to_the_old_door():
    """A door that grew a flag must not move the file nobody set it on."""
    plain, _ = tc.configuration_text(cycle=CYCLE, point=POINT)
    zero, exp = tc.configuration_text(cycle=CYCLE, point=POINT, start_hour=0)
    assert plain == zero
    assert exp.start_time == MOMENT
    assert "forecast_start_hour" not in tomllib.loads(plain)["fetch"]
    assert "f000 analysis" in plain.splitlines()[1]


@pytest.mark.parametrize("source", lead_sources())
def test_every_lead_publishing_source_begins_at_the_lead_it_is_given(source):
    """One grammar for all of them: cycle picks the run, the lead picks
    where in it the forecast begins."""
    adapter = get_source_adapter(source)
    step = int(adapter.forcing_interval_seconds // 3600)
    lead = min(step * 2, adapter.max_forecast_hour - step)
    if lead <= 0:
        pytest.skip(f"{source} publishes no lead beyond one forcing interval")
    text, exp = tc.configuration_text(cycle=CYCLE, point=tc_point(source),
                                      hours=step, forcing_source=source,
                                      start_hour=lead)
    table = tomllib.loads(text)
    assert exp.start_time == MOMENT + timedelta(hours=lead)
    assert table["fetch"]["forecast_start_hour"] == lead
    assert table["fetch"]["cycle"] == "2026-09-09T00"
    from gpuwm.fetch import validate_fetch_hints
    validate_fetch_hints(table["fetch"], source="fixture")


def tc_point(source):
    from gpuwm.source_coverage import window_centre
    window = get_source_adapter(source).coverage_window
    return window_centre(window) if window is not None else POINT


def test_the_lead_reaches_the_clock_the_header_and_the_document():
    text, exp = tc.configuration_text(cycle=CYCLE, point=POINT, hours=3,
                                      start_hour=186)
    assert exp.start_time == datetime(2026, 9, 16, 18)
    assert "f186 forecast (valid 2026-09-16 18 UTC)" in text.splitlines()[1]
    sizing = dw.SizingBudget(32., 30 * dw.GIB, None, "fixture", measured=False)
    result = tc.plan_cyclone(cycle=CYCLE, point=POINT, sizing=sizing, hours=3,
                             tiles="off", start_hour=186)
    assert result["forecast_start_hour"] == 186
    assert result["start_time"] == "2026-09-16 18:00:00"
    assert result["cycle"] == "2026090900"


def test_the_selection_map_previews_the_same_lead_the_run_begins_at(monkeypatch):
    monkeypatch.setattr(dw, "_resolve_cycle", lambda raw, **kw: MOMENT)
    result = tc.latest_map(start_hour=186)
    assert result["map_request"]["forecast_hour"] == 186
    assert result["forecast_start_hour"] == 186
    assert result["valid_time"] == "2026-09-16 18:00:00"
    assert "f186" in result["selection"]
    assert tc.latest_map()["map_request"]["forecast_hour"] == 0


def test_a_lead_past_the_cycles_horizon_names_the_horizon_and_the_lead():
    with pytest.raises(ValueError) as refusal:
        tc.configuration_text(cycle=CYCLE, point=POINT, hours=6, start_hour=384)
    text = str(refusal.value)
    assert "f384" in text and "f390" in text
    assert "shorten --hours" in text and "earlier lead" in text


def test_the_horizon_is_the_cycles_own_not_the_registry_ceiling():
    """A producer whose horizon depends on the cycle hour is judged on the
    cycle it was given, which is the only number its files obey."""
    cycle_bound = [source for source in lead_sources()
                   if (cs.forecast_horizon(source, MOMENT) or 0)
                   < get_source_adapter(source).max_forecast_hour]
    if not cycle_bound:
        pytest.skip("no registered source declares a per-cycle horizon here")
    source = cycle_bound[0]
    horizon = cs.forecast_horizon(source, MOMENT)
    with pytest.raises(ValueError, match=f"f{horizon:03d}"):
        tc.configuration_text(cycle=CYCLE, point=tc_point(source), hours=3,
                              forcing_source=source, start_hour=horizon + 3)


@pytest.mark.parametrize("source", analysis_only_sources())
def test_an_analysis_only_source_refuses_a_lead_and_names_why(source):
    with pytest.raises(ValueError, match="max_forecast_hour = 0"):
        tc.configuration_text(cycle=CYCLE, point=tc_point(source), hours=6,
                              forcing_source=source, start_hour=6)
    # And still authors its analysis configuration, unchanged.
    text, exp = tc.configuration_text(cycle=CYCLE, point=tc_point(source),
                                      hours=6, forcing_source=source)
    assert exp.start_time == MOMENT
    assert "forecast_start_hour" not in tomllib.loads(text)["fetch"]


def test_a_negative_lead_is_refused_before_anything_is_resolved():
    with pytest.raises(ValueError, match="nonnegative whole forecast lead"):
        tc.configuration_text(cycle=CYCLE, point=POINT, start_hour=-1)


def test_a_lead_the_producers_ladder_skips_is_refused_by_that_ladder():
    """The refusal is the fetch route's own, not a second opinion here: a
    window that cannot be downloaded must not become a configuration."""
    with pytest.raises(ValueError) as refusal:
        tc.configuration_text(cycle=CYCLE, point=POINT, hours=3,
                              forcing_source="gfs", start_hour=187)
    assert "f187" in str(refusal.value)


def test_the_source_menu_carries_the_lead_ceiling_a_form_bounds_its_field_by(capsys):
    parser = argparse.ArgumentParser()
    tc.register_cli(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(["cyclone-setup", "--list-sources"])
    assert tc.main(args) == 0
    menu = json.loads(capsys.readouterr().out)
    rows = {row["source"]: row for row in menu["sources"]}
    assert set(rows) == set(cs.source_ids())
    for source, row in rows.items():
        assert row["max_forecast_hour"] == get_source_adapter(source).max_forecast_hour
    assert all(rows[source]["max_forecast_hour"] == 0
               for source in analysis_only_sources())


def test_the_cli_carries_the_lead_into_the_document(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    parser = argparse.ArgumentParser()
    tc.register_cli(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(["cyclone-setup", "--cycle", CYCLE, "--point=18,-65",
                              "--start-hour", "186", "--hours", "3",
                              "--vram-gib", "32", "--tiles", "off", "--json"])
    budget = dw.SizingBudget(32., 30 * dw.GIB, None, "fixture", measured=False)
    monkeypatch.setattr(dw, "_domain_target_hardware", lambda args: (budget, None, False))
    monkeypatch.setattr(dw, "_sizing_phases", lambda *a, **k:
        SimpleNamespace(peak_envelope_bytes=100, binding_phase="forecast",
                        tree_road=None))
    monkeypatch.setattr(dw, "sizing_budget_bytes", lambda *a, **k: 200)
    assert tc.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["forecast_start_hour"] == 186
    assert result["start_time"] == "2026-09-16 18:00:00"
    assert tomllib.loads(result["config_text"])["fetch"]["forecast_start_hour"] == 186
