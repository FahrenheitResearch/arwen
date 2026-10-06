"""Cyclone doors consume registry and route facts rather than source names."""
from datetime import datetime
from types import SimpleNamespace
import argparse
import json
import tomllib

import pytest

from gpuwm import cyclone_setup as tc
from gpuwm import cyclone_sources as cs
from gpuwm import domain_wizard as dw
from gpuwm import fetch_routes as routes
from gpuwm.fetch import validate_fetch_hints
from gpuwm.companion_domains import VORTEX_PRESET
from gpuwm.source_adapters import get_source_adapter

from cyclone_preset_fit import center, cycle_for, holds_the_preset_root


def refused_for_its_window(source):
    """A source too small for the preset refuses and names covering sources."""
    if holds_the_preset_root(source):
        return False
    with pytest.raises(ValueError, match="covering sources"):
        tc.configuration_text(cycle=cycle_for(source), point=center(source),
                              forcing_source=source)
    return True


# The walk is the sources the door OFFERS (cs.source_ids: fetchable and
# wizard-planable).  A fetchable row with no initialization route is
# refused by name and is not on the menu, so it is not walked here.
@pytest.mark.parametrize("source", cs.source_ids())
def test_all_fetchable_sources_author_valid_configuration(source):
    if refused_for_its_window(source):
        return
    text, exp = tc.configuration_text(cycle=cycle_for(source), point=center(source),
                                      forcing_source=source)
    table = tomllib.loads(text)
    assert table["fetch"]["source"] == source
    validate_fetch_hints(table["fetch"], source="fixture")
    assert table["fetch"].get("member") == cs.selected_member(source)
    assert exp.run_seconds == 21600
    assert [d.run.dx for d in exp.domains] == [12000., 3000.]
    assert {k: v for k, v in table["domain"][1]["follow"].items() if k != "track"} == VORTEX_PRESET


@pytest.mark.parametrize("source", cs.source_ids())
def test_the_plan_route_is_the_one_the_authored_configuration_belongs_to(source):
    """A152: a storm-following plan names the route its configuration runs on.

    The config-driven route decodes a configuration only through its
    [case_data] table and refuses one that carries [fetch] alone as
    belonging to the prepared route, which is what `gpuwm run-plan` said
    of every GFS storm-following plan the desktop wrote with the
    config-driven route.  The route is the source row's, so it is the one
    the setup's own configuration needs.
    """

    if refused_for_its_window(source):
        return
    text, _exp = tc.configuration_text(cycle=cycle_for(source), point=center(source),
                                       forcing_source=source)
    table = tomllib.loads(text)
    assert cs.plan_route(source) == (
        "experiment" if "case_data" in table else "prepared")


def test_gfs_storms_run_on_the_prepared_route():
    assert cs.plan_route("gfs") == "prepared"


@pytest.mark.parametrize("source,member", [("gefs", "p03"), ("aigefs", "mem017")])
def test_member_survives_config_and_acquisition_identity(source, member, tmp_path):
    text, _ = tc.configuration_text(cycle="2026090900", point=(18., -65.),
                                    forcing_source=source, member=member)
    table = tomllib.loads(text)["fetch"]
    assert table["member"] == member and table["out"].endswith(member)
    plan = routes.resolve_request(source, cycle=datetime(2026, 9, 9), hours=table["hours"],
                                  cadence=table["cadence"], member=table["member"],
                                  out=tmp_path)
    assert plan.member == member
    assert plan.member_set == routes.route_for(source).prep.get("member_prep")
    assert routes.resolve_member(routes.route_for(source), member)[1] in plan.objects[0].key


@pytest.mark.parametrize("source,member", [("gefs", "p99"), ("aigefs", "mem099"),
                                          ("rap", "p01"), ("gfs", "p01")])
def test_invalid_members_refuse_at_config_review(source, member):
    with pytest.raises(ValueError):
        validate_fetch_hints({"source": source, "member": member}, source="fixture")


def test_hourly_and_three_hourly_cycles_use_source_schedule():
    assert tc._cycle("2026090901", forcing_source="rap").hour == 1
    assert tc._cycle("2026090903", forcing_source="icon-eu").hour == 3
    with pytest.raises(ValueError, match="00/12"):
        tc._cycle("2026090906", forcing_source="gem-gdps")


def test_uncovered_center_names_position_and_covering_sources():
    with pytest.raises(ValueError, match="position.*outside.*grid.*gfs"):
        tc.configuration_text(cycle="2026090900", point=(18., -65.), forcing_source="icon-eu")


def test_parent_coverage_is_checked_separately_from_center():
    adapter = get_source_adapter("icon-eu")
    s, w, n, e = adapter.coverage_window.envelope()
    point = (s + 1, (w + e) / 2)
    cs.validate_center("icon-eu", point)
    with pytest.raises(ValueError, match="root.*outside|outside.*root"):
        tc.configuration_text(cycle="2026090900", point=point, forcing_source="icon-eu")


def test_short_cycle_horizon_refuses_before_sizing(monkeypatch):
    with pytest.raises(ValueError, match="ends at.*shorten"):
        tc.configuration_text(cycle="2026090901", point=center("rap"), hours=48,
                              forcing_source="rap")


def test_plan_prices_selected_source_and_interval(monkeypatch):
    priced, budgeted = [], []
    monkeypatch.setattr(dw, "_sizing_phases", lambda *a, **kw:
                        (priced.append(kw) or SimpleNamespace(peak_envelope_bytes=100,
                                                              binding_phase="forecast")))
    monkeypatch.setattr(dw, "sizing_budget_bytes", lambda *a, **kw:
                        (budgeted.append(kw) or 200))
    result = tc.plan_cyclone(cycle="2026090900", point=(18., -65.), forcing_source="aigefs",
                             member="mem007", sizing=dw.SizingBudget(32., 30 * dw.GIB,
                                                                    None, "fixture"))
    assert priced and all(row["source"] == "aigefs" for row in priced)
    assert all(row["forcing_interval_seconds"] == 21600
               for row in priced + budgeted)
    assert result["source"] == "aigefs" and result["member"] == "mem007"
    assert result["forcing_interval_seconds"] == 21600
    assert result["schema"] == "arwen.cyclone-setup.v2"


def test_source_menu_uses_routes_and_member_grammar():
    options = {row["source"]: row for row in cs.source_options()}
    # The menu is exactly the sources the door can author: offering a
    # fetchable row that then refuses for want of an initialization route
    # would be a dead door.
    assert set(options) == set(cs.source_ids())
    assert set(options) <= set(routes.all_fetchable_sources())
    for source in set(routes.all_fetchable_sources()) - set(options):
        with pytest.raises(ValueError, match="--list-sources"):
            cs.source_adapter(source)
    for source, option in options.items():
        if source in routes.route_ids():
            assert option["members"] == list(routes.member_tokens(routes.route_for(source)))


def test_cli_source_and_member_selection_reaches_map(capsys):
    parser = argparse.ArgumentParser()
    tc.register_cli(parser.add_subparsers())
    args = parser.parse_args(["cyclone-setup", "--latest-map", "--cycle", "2026090900",
                             "--source", "gefs", "--member", "p04"])
    assert tc.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["map_request"]["source"] == "gefs"
    assert result["map_request"]["member"] == 4
    assert result["member"] == "p04"


@pytest.mark.parametrize("source", cs.source_ids())
def test_wps_interval_uses_same_registry_value_as_fetch_and_memory(source):
    from gpuwm.hrrr_prepared_bundle import render_wps_namelist
    if refused_for_its_window(source):
        return
    _, experiment = tc.configuration_text(cycle=cycle_for(source), point=center(source),
                                          forcing_source=source)
    interval = get_source_adapter(source).forcing_interval_seconds
    text = render_wps_namelist(experiment, interval_seconds=interval)
    assert f" interval_seconds = {int(interval)}," in text


@pytest.mark.parametrize("interval", [0, -1, 0.5, float("nan"), float("inf")])
def test_wps_rejects_invalid_intervals(interval):
    from gpuwm.hrrr_prepared_bundle import render_wps_namelist, HrrrBundleError
    with pytest.raises(HrrrBundleError, match="positive whole"):
        render_wps_namelist(None, interval_seconds=interval)


def test_combined_grib_source_declares_real_forcing_and_wps(tmp_path):
    from pathlib import Path
    source = "era5"
    out = tmp_path / "subfolder" / "cyclone.toml"
    text, _ = tc.configuration_text(cycle="2020010100", point=(18.,-65.),
                                    forcing_source=source, source=str(out))
    raw = tomllib.loads(text)
    data = raw["case_data"]
    from gpuwm.fetch import era5_combined_name
    # The declared file is the one the declared provider publishes.
    expected = (Path(raw["fetch"]["out"]).resolve()
                / era5_combined_name(raw["fetch"].get("era5_provider")))
    assert (out.parent / data["forcing"][0]).resolve() == expected
    assert data["forcing_interval_s"] == raw["fetch"]["cadence"]*3600
    assert data["wps_namelist"] == "cyclone.namelist.wps"
    companions = cs.companion_input_files(source,out)
    assert companions[0][0] == out.with_suffix(".Vtable")
    assert len(companions[0][1]) > 100


def test_track_output_is_default_and_round_trips(tmp_path):
    from gpuwm.runtime import build_track_writer
    from dataclasses import replace
    from gpuwm.experiment import RelocationConfig
    _, exp = tc.configuration_text(cycle="2026090900",point=(18.,-65.))
    follow = exp.domains[1].follow
    assert follow.track.path == "storm-track.d02.csv"
    view = replace(exp,relocation=RelocationConfig(enabled=True,grid_id=2,
                   follow=follow.tracker,track=follow.track,cadence_seconds=follow.cadence_seconds))
    writer = build_track_writer(view,tmp_path)
    assert writer is not None
    writer.close()
    assert (tmp_path/follow.track.path).read_text().startswith("valid_time,")


def test_two_track_writers_cannot_clobber_same_output():
    from dataclasses import replace
    from gpuwm.core.nest_lifecycle import validate_follow_tracks
    from gpuwm.experiment import RelocationConfig
    _, exp = tc.configuration_text(cycle="2026090900",point=(18.,-65.))
    follow = exp.domains[1].follow
    legacy = RelocationConfig(enabled=True,grid_id=2,follow=follow.tracker,track=follow.track)
    with pytest.raises(ValueError, match="same track path"):
        validate_follow_tracks(exp.domains,legacy,"fixture.toml")


def test_a_reanalysis_cyclone_reads_the_keyless_store_and_declares_the_file_it_publishes():
    # Breakage: the fetch's default ERA5 provider is the keyed CDS service,
    # so a cyclone configuration from a computer without a CDS key failed at
    # acquisition after it had been authored and accepted.
    from gpuwm.fetch import era5_combined_name

    text, _ = tc.configuration_text(cycle="2005082706", point=(24.6, -84.9), forcing_source="era5")
    table = tomllib.loads(text)
    assert table["fetch"]["era5_provider"] == "arco"
    assert table["case_data"]["forcing"][0].endswith(era5_combined_name("arco"))
