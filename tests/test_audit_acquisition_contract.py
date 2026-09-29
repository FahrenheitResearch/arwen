"""Offline acquisition regressions for capability audit C-001/004/005/012/038/039.

No provider requests or model forecasts run in this suite. CLI parser tests use
fetch's real argument registration without importing unrelated model modules.
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gpuwm import fetch, fetch_routes
from gpuwm.era5_member import validate_selection
from gpuwm.source_cycles import CycleGrid


def args(*words):
    parser = argparse.ArgumentParser()
    fetch.register_cli(parser.add_subparsers(dest="command", required=True))
    return parser.parse_args(["fetch", *words])


@pytest.mark.parametrize("source", fetch_routes.route_ids())
def test_every_route_rejects_a_partial_final_cadence(source):
    route = fetch_routes.route_for(source)
    cadence = next((step for step in route.cadences if step > 1), None)
    if cadence is None:
        return  # Every integer endpoint is on this route's sole hourly cadence.
    cycle = datetime(2026, 8, 17, route.cycle_hours[0])
    with pytest.raises(ValueError, match="multiple"):
        fetch_routes.resolve_leads(route, cycle, cadence + 1, cadence=cadence)


@pytest.mark.parametrize("field,value", [
    ("hours", True), ("hours", 6.5), ("hours", "6"),
    ("cadence", True), ("cadence", 1.5), ("cadence", "1"),
    ("start_hour", True), ("start_hour", -1), ("start_hour", 0.5),
])
def test_route_window_requires_integer_nonnegative_inputs(field, value):
    kwargs = dict(hours=6, cadence=1, start_hour=0)
    kwargs[field] = value
    with pytest.raises(ValueError):
        fetch_routes.resolve_request("rap", cycle=datetime(2026, 8, 17, 9), **kwargs)


@pytest.mark.parametrize("cadence", [2, 4, 5, 12, 24])
def test_hres_accepts_every_positive_whole_hour_cadence(cadence):
    validate_selection(cadence=cadence)
    cycle = datetime(2026, 7, 20)
    assert fetch._era5_times(cycle, 2 * cadence, cadence) == (
        cycle, cycle + timedelta(hours=cadence), cycle + timedelta(hours=2 * cadence))


@pytest.mark.parametrize("cadence", [3, 6, 9, 12, 24])
def test_eda_accepts_subsequences_of_its_native_ladder(cadence):
    assert validate_selection(product_type="ensemble_members", member=7,
        cadence=cadence, provider="cds", cycle=datetime(2026, 7, 20, 3)) == 7


@pytest.mark.parametrize("cadence", [0, -1, True, 2.5, "3"])
def test_era5_time_builder_rejects_bad_cadence(cadence):
    with pytest.raises(ValueError):
        fetch._era5_times(datetime(2026, 7, 20), 6, cadence)


@pytest.mark.parametrize("hours", [-1, True, 2.5, "3", 5])
def test_era5_time_builder_rejects_bad_window(hours):
    with pytest.raises(ValueError):
        fetch._era5_times(datetime(2026, 7, 20), hours, 3)


@pytest.mark.parametrize("provider,cadence,member", [("arco", 3, 1), ("cds", 4, 1),
                                                   ("cds", 3, None), ("cds", 3, 10)])
def test_eda_provider_member_and_ladder_guards_remain(provider, cadence, member):
    with pytest.raises(ValueError):
        validate_selection(product_type="ensemble_members", member=member,
                           cadence=cadence, provider=provider)




_BAD_WINDOW_SPELLINGS = [
    (key, value)
    for key in ("hours", "forecast_start_hour", "cadence")
    for value in (True, False, "3", 3.0, 1.5, -1, float("nan"), float("inf"))
] + [("cadence", 0)]


def test_hint_source_catalog_is_not_empty():
    assert fetch._fetch_hint_sources(), "no source was available for the window sweep"


@pytest.mark.parametrize("source", fetch._fetch_hint_sources())
@pytest.mark.parametrize("key,value", _BAD_WINDOW_SPELLINGS)
def test_all_hint_sources_reject_invalid_window_spellings(source, key, value):
    table = {"source": source, key: value}
    with pytest.raises(ValueError) as refusal:
        fetch.validate_fetch_hints(table, source="window.toml")
    assert key in str(refusal.value), refusal.value


@pytest.mark.parametrize("key,value", _BAD_WINDOW_SPELLINGS)
def test_future_hint_row_is_type_checked_before_source_dispatch(monkeypatch, key, value):
    source = "fixture-source-not-in-a-model-list"
    monkeypatch.setattr(fetch, "_fetch_hint_sources", lambda: (source,))
    def unexpected(*args, **kwargs):
        pytest.fail("source dispatch ran before common scalar validation")
    monkeypatch.setattr(fetch, "parse_cycle", unexpected)
    monkeypatch.setattr(fetch_routes, "route_for", unexpected)
    with pytest.raises(ValueError) as refusal:
        fetch.validate_fetch_hints({"source": source, "cycle": "2000-01-01T00", key: value},
                                  source="window.toml")
    assert key in str(refusal.value), refusal.value


def test_latest_analysis_window_ends_at_or_before_declared_publication(monkeypatch):
    grid = CycleGrid(hours=(0, 6, 12, 18), delay_hours=120)
    monkeypatch.setattr(fetch, "require_cycle_grid", lambda source: grid)
    now = datetime(2026, 7, 28, 5, 30)
    latest = fetch.resolve_latest_cycle("era5", 18, now=now,
        probe=lambda url: pytest.fail("A CDS job API is not a file server"))
    assert latest == datetime(2026, 7, 22, 6)
    assert latest + timedelta(hours=18) <= grid.newest(now)


def test_latest_closed_analysis_archive_accounts_for_the_requested_end(monkeypatch):
    grid = CycleGrid(hours=(0, 6, 12, 18), record_end=datetime(2020, 1, 1, 18))
    monkeypatch.setattr(fetch, "require_cycle_grid", lambda source: grid)
    latest = fetch.resolve_latest_cycle("era5", 12, now=datetime(2026, 7, 28))
    assert latest == datetime(2020, 1, 1, 6)


def test_cli_parser_does_not_carry_a_second_era5_cadence_menu(tmp_path):
    parsed = args("--source", "era5", "--cycle", "2026-07-20T00", "--hours", "24",
                  "--cadence", "12", "--area", "30,-100,40,-90", "--out", str(tmp_path))
    assert parsed.cadence == 12


@pytest.mark.parametrize("source,member", [("gefs", "p01"), ("aigefs", "mem001")])
def test_ensemble_member_fetch_hints_use_route_vocabulary(source, member):
    fetch.validate_fetch_hints(dict(source=source, member=member, hours=6,
                                   cycle="2026-08-17T00"), source="case.toml")


@pytest.mark.parametrize("table", [
    dict(source="rap", cycle="2026-08-17T07", hours=24),
    dict(source="icon-eu", cycle="2026-08-17T01", hours=6),
    dict(source="rap", cycle="not-a-cycle", hours=6),
    dict(source="rap", cycle="latest", hours=999),
    dict(source="rap", hours=-1),
    dict(source="rap", hours=6, cadence=1.5),
    dict(source="gfs", forecast_start_hour=120, hours=6, cadence=2),
    dict(source="era5", hours=5, cadence=3),
    dict(source="hrrr", hours=6, cadence=3),
    dict(source="gefs", member="not-a-member"),
    dict(source="rap", member="p01"),
    dict(source="gfs", point="35,-97"),
    dict(source="gfs", radius_km=100),
    dict(source="gfs", area="30,-100,40,-90", point="35,-97", radius_km=100),
])
def test_bad_fetch_hints_are_refused_before_any_io(table):
    with pytest.raises(ValueError, match=r"\[fetch\].*case.toml"):
        fetch.validate_fetch_hints(table, source="case.toml")

@pytest.mark.parametrize("source", fetch_routes.route_ids())
def test_table_cli_resolves_latest_and_keeps_the_resolved_request(source, tmp_path, monkeypatch):
    route = fetch_routes.route_for(source)
    selected = datetime(2026, 8, 17, route.cycle_hours[0])
    calls = []
    def latest(name, last_hour, **kwargs):
        calls.append(("latest", name, last_hour, kwargs))
        return selected
    monkeypatch.setattr(fetch, "resolve_latest_cycle", latest)
    monkeypatch.setattr(fetch, "require_published_cycle",
                        lambda name, cycle, last_hour, **kw: calls.append(("published", name, cycle, last_hour)))
    monkeypatch.setattr(fetch_routes, "run_plan", lambda plan, **kw: calls.append(("download", plan)))
    monkeypatch.setattr(fetch_routes, "write_handoff", lambda *a, **kw: None)
    monkeypatch.setattr(fetch, "_fetch_route_donors", lambda *a: {})
    monkeypatch.setattr(fetch_routes, "handoff_lines", lambda *a: ())
    parsed = args("--source", source, "--cycle", "latest", "--hours", str(route.default_cadence),
                  "--out", str(tmp_path / source))
    assert fetch.fetch_main(parsed) == 0
    assert calls[0][:3] == ("latest", source, route.default_cadence)
    # Resolving latest already asked about the primary; only a declared
    # donor is asked again, and before the download.
    donors = [("published", row.source, selected, max(row.leads)) for row in route.donors]
    assert calls[1:-1] == donors
    assert calls[-1][0] == "download" and calls[-1][1].cycle == selected


def test_latest_route_checks_its_donor_publication_before_starting_download(tmp_path, monkeypatch):
    route = fetch_routes.route_for("aigefs")
    selected = datetime(2026, 8, 17, route.cycle_hours[0])
    calls = []
    monkeypatch.setattr(fetch, "resolve_latest_cycle", lambda name, last_hour, **kw: selected)
    def published(name, cycle, last_hour, **kw):
        calls.append((name, cycle, last_hour))
        raise RuntimeError(f"{name.upper()} cycle {cycle:%Y-%m-%dT%H}Z is not published through f000 yet")
    monkeypatch.setattr(fetch, "require_published_cycle", published)
    monkeypatch.setattr(fetch_routes, "run_plan", lambda *a, **kw: pytest.fail("download before donor publication check"))
    parsed = args("--source", "aigefs", "--cycle", "latest", "--hours", str(route.default_cadence),
                  "--out", str(tmp_path))
    with pytest.raises(RuntimeError, match=re.escape(
            "--source aigefs takes part of its start from the GDAS analysis of its own "
            f"cycle, and GDAS cycle {selected:%Y-%m-%dT%H}Z is not published")):
        fetch.fetch_main(parsed)
    assert calls == [("gdas", selected, 0)]


def test_selected_ensemble_member_is_used_in_the_actual_publication_probe():
    from gpuwm import fetch_endpoints
    selected = datetime(2026, 8, 17)
    seen = []
    latest = fetch.resolve_latest_cycle("gefs", 6, now=selected + timedelta(hours=fetch.require_cycle_grid("gefs").delay_hours, minutes=30),
        member="p02", cadence=3, transport="aws", probe=lambda url: seen.append(url) or True)
    assert latest == selected
    assert seen and all("gep02" in url for url in seen)
    assert all("nomads" not in url for url in seen)


def test_named_route_checks_publication_before_starting_download(tmp_path, monkeypatch):
    calls = []
    def refused(*a, **kw):
        calls.append((a, kw))
        raise RuntimeError("fixture: selected member is not published")
    monkeypatch.setattr(fetch, "require_published_cycle", refused)
    monkeypatch.setattr(fetch_routes, "run_plan", lambda *a, **kw: pytest.fail("download before publication check"))
    parsed = args("--source", "gefs", "--cycle", "2026-08-17T00", "--hours", "6",
                  "--member", "p02", "--forecast-start-hour", "3", "--out", str(tmp_path))
    with pytest.raises(RuntimeError, match="selected member"):
        fetch.fetch_main(parsed)
    assert calls[0][0] == ("gefs", datetime(2026, 8, 17), 9)
    assert calls[0][1]["member"] == "p02"
    assert calls[0][1]["start_hour"] == 3


def test_named_route_checks_its_donor_publication_before_starting_download(tmp_path, monkeypatch):
    calls = []
    def published(source, cycle, last_hour, **kw):
        calls.append((source, cycle, last_hour))
        if source == "gdas":
            raise RuntimeError("fixture: donor analysis is not published")
    monkeypatch.setattr(fetch, "require_published_cycle", published)
    monkeypatch.setattr(fetch_routes, "run_plan", lambda *a, **kw: pytest.fail("download before donor publication check"))
    route = fetch_routes.route_for("aigfs")
    cycle = datetime(2026, 8, 17, route.cycle_hours[0])
    parsed = args("--source", "aigfs", "--cycle", f"{cycle:%Y-%m-%dT%H}",
                  "--hours", str(route.default_cadence), "--out", str(tmp_path))
    with pytest.raises(RuntimeError, match="donor analysis"):
        fetch.fetch_main(parsed)
    assert calls == [("aigfs", cycle, route.default_cadence), ("gdas", cycle, 0)]


def test_a_route_refuses_a_flag_it_does_not_take_before_reading_its_window(tmp_path):
    parsed = args("--source", "gefs", "--cycle", "2026-08-17T00", "--hours", "7",
                  "--cadence", "3", "--engine", "rust", "--out", str(tmp_path / "absent"))
    with pytest.raises(ValueError, match="--engine: --source gefs is a table-driven route"):
        fetch.fetch_main(parsed)
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("extra", [("--hours", "7", "--cadence", "3"),
                                  ("--hours", "6", "--member", "bogus"),
                                  ("--hours", "999"),
                                  ("--hours", "6", "--transport", "dwd")])
def test_latest_invalid_request_never_probes_or_creates_output(extra, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "resolve_latest_cycle", lambda *a, **kw: pytest.fail("invalid request performed a probe"))
    out = tmp_path / "absent"
    with pytest.raises(ValueError):
        fetch.fetch_main(args("--source", "gefs", "--cycle", "latest", "--out", str(out), *extra))
    assert not out.exists()


def test_latest_era5_front_door_writes_template_for_the_resolved_start(tmp_path, monkeypatch, capsys):
    selected = datetime(2026, 8, 10, 6)
    monkeypatch.setattr(fetch, "resolve_latest_cycle", lambda name, end: selected)
    calls = []
    monkeypatch.setattr(fetch, "write_era5_request", lambda **kw: calls.append(kw))
    assert fetch.fetch_main(args("--source", "era5", "--cycle", "latest", "--hours", "24",
        "--cadence", "12", "--area", "30,-100,40,-90", "--out", str(tmp_path))) == 0
    assert calls[0]["cycle"] == selected
    assert calls[0]["hours"] == 24 and calls[0]["cadence"] == 12
    assert "declared" in capsys.readouterr().out.lower()


@pytest.mark.parametrize('cycle,hours,cadence', [
    (datetime(2026, 7, 20, 18), 12, 6),
    (datetime(2026, 7, 20, 1), 35, 5),
    (datetime(2026, 7, 20, 18), 54, 3),
    (datetime(2026, 7, 20), 72, 24),
])
def test_cds_template_has_exact_times_not_a_date_time_cross_product(cycle, hours, cadence):
    template = fetch.era5_request_template(cycle=cycle, hours=hours,
        cadence=cadence, area=fetch.parse_area('30,-100,40,-90'))
    expected = fetch._era5_times(cycle, hours, cadence)
    for dataset in ('reanalysis-era5-pressure-levels', 'reanalysis-era5-single-levels'):
        actual = [datetime.strptime(day + 'T' + clock, '%Y-%m-%dT%H:%M')
                  for row in template['requests'] if row['dataset'] == dataset
                  for day in row['request']['date'] for clock in row['request']['time']]
        assert tuple(actual) == expected
    targets = [row['target'] for row in template['requests']]
    assert len(targets) == len(set(targets))


def test_invalid_era5_template_does_not_create_an_output_directory(tmp_path):
    out = tmp_path / 'not-created'
    with pytest.raises(ValueError):
        fetch.write_era5_request(cycle=datetime(2026, 7, 20), hours=5, cadence=3,
            area=fetch.parse_area('30,-100,40,-90'), out=out)
    assert not out.exists()


@pytest.mark.parametrize("source", ["gfs", "gdas", "hrrr", "era5", *fetch_routes.route_ids()])
def test_each_acquisition_ladder_accepts_one_analysis(source):
    cycle = datetime(2026, 8, 17, 0)
    if source == "gfs":
        values = fetch.gfs_forecast_hours(0, 1)
    elif source == "gdas":
        values = fetch.gdas_forecast_hours(0, 3)
    elif source == "hrrr":
        values = fetch.hrrr_forecast_hours(0, cycle)
    elif source == "era5":
        assert fetch._era5_times(cycle, 0, 6) == (cycle,)
        return
    else:
        route = fetch_routes.route_for(source)
        values = fetch_routes.resolve_leads(route, cycle, 0)
    assert values == (0,)


def test_one_hrrr_frame_remains_insufficient_for_a_forecast():
    from gpuwm.hrrr_forecast import validate_hrrr_source_forecast_hours
    with pytest.raises(ValueError, match="at least two"):
        validate_hrrr_source_forecast_hours([0], cycle=datetime(2026, 8, 17))


def test_analysis_cannot_be_misrepresented_as_a_gfs_forecast_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "_load_fetch_manifest", lambda out: {
        "source": "gfs", "cycle": "2026-08-17T00:00:00Z", "forecast_hours": [0]})
    with pytest.raises(ValueError, match="at least two"):
        fetch.author_gfs_front_door_manifest(out=tmp_path, bridge=tmp_path / "bridge",
            wps_namelist=tmp_path / "namelist.wps", experiment_config=tmp_path / "experiment.toml")
    assert not list(tmp_path.iterdir())


def test_analysis_era5_template_has_one_instant(tmp_path):
    template = fetch.era5_request_template(cycle=datetime(2026, 7, 20), hours=0,
        area=fetch.parse_area("30,-100,40,-90"), out=tmp_path)
    assert len(template["requests"]) == 2
    assert all(item["request"]["time"] == ["00:00"] for item in template["requests"])


#: Every refusal kept in this lane's review of the acquisition delivery has to
#: say WHAT BREAKS, not only what the rule is: the three two-frame refusals all
#: guard the same physical fact, that a lateral boundary is interpolated
#: BETWEEN forcing frames, so one frame leaves every interval empty.  A reader
#: who is told "needs at least two" and nothing else cannot tell a policy from
#: a physical limit, and the way out ("fetch one more forcing time", "keep the
#: analysis") is what separates the two.
_BREAKAGE = "boundary"
_WAY_OUT = ("analysis", "cadence step", "widen the window", "one more forcing")


def _names_breakage_and_way_out(message: str) -> None:
    assert _BREAKAGE in message, message
    assert any(phrase in message for phrase in _WAY_OUT), message


def test_the_manifest_two_frame_refusal_names_the_empty_boundary_interval(
        tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "_load_fetch_manifest", lambda out: {
        "source": "gfs", "cycle": "2026-08-17T00:00:00Z", "forecast_hours": [0]})
    with pytest.raises(ValueError) as refusal:
        fetch.author_gfs_front_door_manifest(
            out=tmp_path, bridge=tmp_path / "bridge",
            wps_namelist=tmp_path / "namelist.wps",
            experiment_config=tmp_path / "experiment.toml")
    _names_breakage_and_way_out(str(refusal.value))


def test_the_hours_zero_manifest_refusal_names_the_empty_boundary_interval(
        tmp_path):
    # The manifest's own file arguments are supplied, so the refusal under
    # test is the window one and not the earlier "this flag needs those
    # flags" one.
    parsed = args("--source", "gfs", "--cycle", "2026-08-17T00", "--hours", "0",
                  "--out", str(tmp_path), "--author-front-door-manifest",
                  "--wps-namelist", str(tmp_path / "namelist.wps"),
                  "--experiment-config", str(tmp_path / "experiment.toml"))
    with pytest.raises(ValueError) as refusal:
        fetch.fetch_main(parsed)
    _names_breakage_and_way_out(str(refusal.value))


def test_the_hrrr_two_frame_refusal_names_the_empty_interval_it_prevents():
    from gpuwm.hrrr_forecast import validate_hrrr_source_forecast_hours
    with pytest.raises(ValueError) as refusal:
        validate_hrrr_source_forecast_hours([0], cycle=datetime(2026, 8, 17))
    assert "interval" in str(refusal.value), refusal.value
    assert "Widen the window" in str(refusal.value), refusal.value


def test_the_hrrr_two_frame_refusal_names_no_flag_its_caller_cannot_accept():
    """A way out names a knob the door that refused can actually take.

    The validator serves six callers and only one of them parses flags: a
    preparation proof, a gate file and three tools reach it with no command
    line at all, and the one command line that does spells its knob
    ``--forecast-hours``.  So the default refusal names no flag, and a door
    that has one passes its own spelling in.
    """

    from gpuwm.hrrr_forecast import validate_hrrr_source_forecast_hours
    with pytest.raises(ValueError) as refusal:
        validate_hrrr_source_forecast_hours([0], cycle=datetime(2026, 8, 17))
    message = str(refusal.value)
    assert re.search(r"--[A-Za-z][A-Za-z0-9-]*", message) is None, message
    assert "gpuwm fetch" in message, message

    with pytest.raises(ValueError) as flagged:
        validate_hrrr_source_forecast_hours(
            [0], cycle=datetime(2026, 8, 17), window_flag="--forecast-hours")
    assert "`--forecast-hours`" in str(flagged.value), flagged.value


def test_the_subset_door_names_its_own_window_flag_not_another_door_s():
    """The tool whose flag is ``--forecast-hours`` says ``--forecast-hours``."""

    import tools.download_hrrr_native_subset as subset

    with pytest.raises(ValueError) as refusal:
        subset._hours("0", cycle=datetime(2026, 8, 17))
    message = str(refusal.value)
    assert "`--forecast-hours`" in message, message
    assert "--hours " not in message, message
