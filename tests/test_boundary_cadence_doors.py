"""A boundary cadence a door accepts is one the preparation takes.

``gpuwm domain --cadence N``, the ``[fetch]`` table's config-load check and
``gpuwm fetch`` each accept a cadence and write it into the run: the fetch
downloads that ladder and the companion namelist carries it as
``interval_seconds``.  The decode then holds the fetched series to the
packaged mapping's ``target.boundary_interval_seconds``.  When the two
disagreed the doors said yes, the whole window was downloaded, and the
decode refused it.

These tests hold the doors and the decode to one answer, read from the same
packaged mapping, for every source whose preparation is a packaged mapped
profile.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuwm import fetch, fetch_routes
from gpuwm.cli import main as cli_main
from gpuwm.source_adapters import get_source_adapter
from gpuwm.source_authorities import (
    BOUNDARY_MULTIPLES_KEY,
    boundary_interval_refusal,
    packaged_authorities,
    packaged_mapping_target,
)


def _mapped_front_door_sources() -> list[str]:
    return sorted(
        source for source in fetch.fetch_front_door_sources()
        if get_source_adapter(source).runner == "mapped_composition_v1"
        and get_source_adapter(source).packaged_profile)


def _offered_cadences(source: str) -> tuple[int, ...]:
    """Every cadence the fetch grammar accepts for SOURCE, in hours."""

    if source in fetch_routes.route_ids():
        # A173: a table route lists no spacings; every whole hour that
        # plans a two-step window on some cycle is one the fetch offers.
        route = fetch_routes.route_for(source)
        accepted = []
        for cadence in range(1, 49):
            for cycle in fetch_routes.planning_cycles(route):
                try:
                    fetch_routes.resolve_leads(
                        route, cycle, 2 * cadence, cadence=cadence)
                except ValueError:
                    continue
                accepted.append(cadence)
                break
        return tuple(accepted)
    if not fetch.fetch_accepts_cadence(source):
        return ()
    # A container ladder: every whole-hour spacing with a window on the
    # published hours is a cadence its planner accepts.
    horizon = get_source_adapter(source).max_forecast_hour
    accepted = []
    for cadence in range(1, horizon + 1):
        try:
            fetch.container_forecast_hours(source, cadence, cadence)
        except ValueError:
            continue
        accepted.append(cadence)
    return tuple(accepted)


@pytest.mark.parametrize("source", _mapped_front_door_sources())
def test_every_cadence_a_fetch_offers_is_one_its_preparation_takes(source):
    target = packaged_mapping_target(get_source_adapter(source).packaged_profile)
    offered = _offered_cadences(source)
    refused = {cadence: boundary_interval_refusal(target, cadence * 3600)
               for cadence in offered}
    assert not {c: why for c, why in refused.items() if why is not None}, (
        f"{source}: the fetch offers cadences its preparation refuses")
    for cadence in offered:
        assert fetch.preparation_cadence_refusal(source, cadence) is None


def test_the_gdas_decode_contract_takes_the_three_hour_ladder():
    target = packaged_mapping_target(
        get_source_adapter("gdas").packaged_profile)
    assert target["boundary_interval_seconds"] == 3600
    assert target[BOUNDARY_MULTIPLES_KEY] is True
    for cadence in (1, 2, 3, 9):
        assert boundary_interval_refusal(target, cadence * 3600) is None
    assert boundary_interval_refusal(target, 5400) is not None


def test_the_door_writes_a_cadence_the_gdas_preparation_takes(tmp_path):
    """Through the wizard, then through the prep contract check."""

    from gpuwm import mapped_direct
    from gpuwm.experiment import load_experiment

    out = tmp_path / "coarse.toml"
    rc = cli_main([
        "domain", "--point=35.5,-97.5", "--root-dx", "3", "--card", "5gb",
        "--hours", "3", "--cadence", "3", "--source", "gdas",
        "--cycle", "2023-03-31T18", "--out", str(out)])
    assert rc == 0
    namelist = out.with_name("coarse.namelist.wps").read_text(encoding="utf-8")
    assert "interval_seconds = 10800," in namelist
    mapping = json.loads(packaged_authorities(
        get_source_adapter("gdas").packaged_profile)["mapping"]
        .read_text(encoding="utf-8"))
    receipt = mapped_direct._validate_target_contract(
        mapping, load_experiment(out), 10800, hierarchy=False,
        experiment_config=out)
    assert receipt["boundary_interval_seconds"] == 10800


@pytest.mark.parametrize("source", ["hrrr-prs", "rap", "rrfs", "icon-d2",
                                    "icon-eu"])
def test_an_hourly_regional_route_offers_and_prepares_every_whole_hour(
        source):
    """A159, then A173: each of these publishes every hour, so a series at
    any whole number of hours is a series of published valid times; the
    route plans every one its hourly ladder reaches and the packaged
    mapping takes it, and a spacing that is not a whole hour is refused."""

    target = packaged_mapping_target(get_source_adapter(source).packaged_profile)
    assert target["boundary_interval_seconds"] == 3600
    assert target[BOUNDARY_MULTIPLES_KEY] is True
    route = fetch_routes.route_for(source)
    assert route.default_cadence == 1
    cycle = max(fetch_routes.planning_cycles(route),
                key=lambda each: fetch_routes.ladder_for(route, each)[-1])
    assert fetch_routes.window_spacings(route, 0, 12, cycle=cycle) == (
        1, 2, 3, 4, 6, 12)
    for cadence in (1, 2, 3, 4, 5, 6, 12):
        assert fetch_routes.resolve_leads(
            route, cycle, 2 * cadence, cadence=cadence) == (
                0, cadence, 2 * cadence)
        assert boundary_interval_refusal(target, cadence * 3600) is None
        assert fetch.preparation_cadence_refusal(source, cadence) is None
    assert "not a whole multiple" in boundary_interval_refusal(target, 5400)
    assert boundary_interval_refusal(target, 1800) is not None
    fetch.validate_fetch_hints(
        {"source": source, "hours": 6, "cadence": 3},
        source=fetch.COMMAND_LINE_HINTS)


def test_the_hrrr_prs_fetch_plans_every_third_lead():
    from datetime import datetime

    plan = fetch_routes.resolve_request(
        "hrrr-prs", cycle=datetime(2026, 9, 30, 12), hours=18, cadence=3)
    assert plan.leads == (0, 3, 6, 9, 12, 15, 18)
    assert [p.name for p in plan.primary_files] == [
        f"hrrr.t12z.wrfprsf{lead:02d}.grib2" for lead in plan.leads]
    bare = fetch_routes.resolve_request(
        "hrrr-prs", cycle=datetime(2026, 9, 30, 12), hours=3)
    assert bare.leads == (0, 1, 2, 3)


def test_the_door_writes_a_three_hour_hrrr_prs_cadence_the_preparation_takes(
        tmp_path):
    """Through the wizard, then through the prep contract check, which
    still refuses a spacing that is not a whole number of hours."""

    from gpuwm import mapped_direct
    from gpuwm.experiment import load_experiment

    out = tmp_path / "coarse.toml"
    rc = cli_main([
        "domain", "--point=35.5,-97.5", "--root-dx", "3", "--card", "5gb",
        "--hours", "6", "--cadence", "3", "--source", "hrrr-prs",
        "--cycle", "2026-09-30T12", "--out", str(out)])
    assert rc == 0
    namelist = out.with_name("coarse.namelist.wps").read_text(encoding="utf-8")
    assert "interval_seconds = 10800," in namelist
    mapping = json.loads(packaged_authorities(
        get_source_adapter("hrrr-prs").packaged_profile)["mapping"]
        .read_text(encoding="utf-8"))
    exp = load_experiment(out)
    receipt = mapped_direct._validate_target_contract(
        mapping, exp, 10800, hierarchy=False, experiment_config=out)
    assert receipt["boundary_interval_seconds"] == 10800
    with pytest.raises(ValueError, match="not a whole multiple"):
        mapped_direct._validate_target_contract(
            mapping, exp, 5400, hierarchy=False, experiment_config=out)


def test_aigefs_takes_whole_multiples_of_its_six_hour_spacing():
    """A164: the single 6 h spacing named no breakage.  Its member hybrid
    is the aigfs composition with member files, and aigfs already takes
    whole multiples (A159); the sealed member demo binding the mapping
    digest was a cost, paid by preparing it again."""

    target = packaged_mapping_target(get_source_adapter("aigefs").packaged_profile)
    assert target["boundary_interval_seconds"] == 21600
    assert target[BOUNDARY_MULTIPLES_KEY] is True
    for hours in (6, 12, 18, 24):
        assert boundary_interval_refusal(target, hours * 3600) is None
        assert fetch.preparation_cadence_refusal("aigefs", hours) is None
    for hours in (3, 9):
        assert "not a whole multiple" in boundary_interval_refusal(
            target, hours * 3600)
        assert "any whole multiple of 6 h" in fetch.preparation_cadence_refusal(
            "aigefs", hours)
    fetch.validate_fetch_hints(
        {"source": "aigefs", "hours": 12, "cadence": 6},
        source=fetch.COMMAND_LINE_HINTS)


@pytest.mark.parametrize("source", sorted(fetch_routes.route_ids()))
def test_every_table_route_takes_a_spacing_unless_a_breakage_refuses_it(
        source):
    """A173: a spacing is refused only for what breaks it.

    For every whole hour from 1 to 24 and every planning cycle, the
    two-step window 0, c, 2c either plans, or is refused because the
    cycle's ladder ends before it or does not publish one of its leads
    (named, with the ladder), or because the packaged preparation refuses
    the spacing (named, with the mapping).  The answer is checked against
    the rows themselves, and "the route's own row" (A164's sentence for a
    spacing only the retired cadences list left out) is never the
    reason."""

    route = fetch_routes.route_for(source)
    adapter = get_source_adapter(route.prep.get("source", source))
    target = packaged_mapping_target(adapter.packaged_profile)
    for cycle in fetch_routes.planning_cycles(route):
        published = set(fetch_routes.ladder_for(route, cycle))
        for cadence in range(1, 25):
            wanted = (0, cadence, 2 * cadence)
            on_ladder = all(lead in published for lead in wanted)
            taken = boundary_interval_refusal(target, cadence * 3600) is None
            if on_ladder and taken:
                assert fetch_routes.resolve_leads(
                    route, cycle, 2 * cadence, cadence=cadence) == wanted
                continue
            with pytest.raises(ValueError) as refused:
                fetch_routes.resolve_leads(
                    route, cycle, 2 * cadence, cadence=cadence)
            message = str(refused.value)
            assert "the route's own row" not in message
            assert "publishes at" not in message
            if 2 * cadence > max(published):
                assert f"forecasts through f{max(published):03d}" in message
            elif not on_ladder:
                missing = min(lead for lead in wanted if lead not in published)
                assert (f"does not publish f{missing:03d} at --cadence "
                        f"{cadence}; its ladder runs") in message
            else:
                assert "refused by the decode" in message
                assert adapter.packaged_profile in message


def test_an_hourly_route_takes_two_hours_at_every_door(tmp_path, capsys):
    """The A164 case, through the doors that take a cadence: hrrr-prs posts
    every hour and its preparation takes any whole hour, so nothing breaks
    a 2 h series, and since A173 nothing refuses it either."""

    from datetime import datetime

    plan = fetch_routes.resolve_request(
        "hrrr-prs", cycle=datetime(2026, 9, 30, 12), hours=6, cadence=2)
    assert plan.leads == (0, 2, 4, 6)
    assert [p.name for p in plan.primary_files] == [
        f"hrrr.t12z.wrfprsf{lead:02d}.grib2" for lead in plan.leads]
    fetch.validate_fetch_hints(
        {"source": "hrrr-prs", "cycle": "2026-09-30T12", "hours": 6,
         "cadence": 2}, source="hrrr-prs.toml")
    rc = cli_main(["fetch", "--source", "hrrr-prs", "--cycle", "2026-09-30T12",
                   "--hours", "6", "--cadence", "2", "--readiness",
                   "--no-probe", "--out", str(tmp_path / "out")])
    out = capsys.readouterr()
    assert rc in (0, 75), out.err
    readiness = json.loads(out.out)
    assert [row["lead"] for row in readiness["leads"]] == [0, 2, 4, 6]


def test_a_cadence_below_one_hour_is_refused_without_reading_the_ladder():
    route = fetch_routes.route_for("hrrr-prs")
    for cadence in (0, -3, 1.5, True):
        with pytest.raises(ValueError, match="a cadence is a whole number of hours"):
            fetch_routes.resolve_leads(
                route, fetch_routes.planning_cycles(route)[0], 6, cadence=cadence)


def test_a_cadence_the_mapping_refuses_is_refused_at_every_door(
        tmp_path, monkeypatch, capsys):
    """A mapping that takes one spacing only is refused before a download."""

    import gpuwm.source_authorities as authorities

    exact = {"boundary_interval_seconds": 3600,
             "require_lateral_boundaries": True}
    monkeypatch.setattr(authorities, "packaged_mapping_target",
                        lambda profile_id: exact)
    with pytest.raises(ValueError) as config_refusal:
        fetch.validate_fetch_hints(
            {"source": "gdas", "hours": 3, "cadence": 3}, source="gdas.toml")
    message = str(config_refusal.value)
    assert "cadence 3" in message and "takes 1 h and no other spacing" in message

    rc = cli_main(["fetch", "--source", "gdas", "--cycle", "2023-03-31T18",
                   "--hours", "3", "--cadence", "3",
                   "--area", "30,-100,40,-90", "--out", str(tmp_path / "out")])
    assert rc == 2
    assert "no other spacing" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()

    out = tmp_path / "refused.toml"
    rc = cli_main([
        "domain", "--point=35.5,-97.5", "--root-dx", "3", "--card", "5gb",
        "--hours", "3", "--cadence", "3", "--source", "gdas",
        "--cycle", "2023-03-31T18", "--out", str(out)])
    assert rc == 2
    assert "no other spacing" in capsys.readouterr().err
    assert not out.exists()


def test_a_bare_gdas_fetch_takes_the_row_spacing():
    """An omitted cadence is the registry row's, so any whole window plans."""

    assert fetch.container_default_cadence("gdas") == 1
    assert fetch.container_default_cadence("gfs") == 3
    assert fetch.container_forecast_hours("gdas", 1) == (0, 1)
    assert fetch.container_forecast_hours("gdas", 3) == (0, 1, 2, 3)
    assert fetch.gdas_forecast_hours(2) == (0, 1, 2)
    fetch.validate_fetch_hints(
        {"source": "gdas", "cycle": "2023-03-31T18", "hours": 1},
        source=fetch.COMMAND_LINE_HINTS)
    # The GFS default is the same number it always was.
    assert fetch.container_forecast_hours("gfs", 6) == (0, 3, 6)


# --------------------------------------------------------------------------
# A173: the spacings the retired cadences lists refused naming no breakage
# --------------------------------------------------------------------------

#: The per-route cadences lists 2.8.0 shipped, kept only to show that a
#: window which names no cadence resolves exactly as it did then.
RETIRED_CADENCES = {
    "icon-global": (3,), "hrrr-prs": (1, 3, 6), "rap": (1, 3, 6),
    "rrfs": (1, 3, 6), "gefs": (3, 6), "aigfs": (6,), "aigefs": (6,),
    "ecmwf-open-data": (3, 6), "aifs": (6,), "icon-eu": (1, 3, 6),
    "icon-d2": (1, 3, 6), "gem-gdps": (3, 6),
}


def _retired_window_cadence(route, start_hour, hours, *, cycle, floor,
                            round_up):
    """2.8.0's window_cadence, over its listed spacings."""

    published = frozenset(fetch_routes.ladder_for(route, cycle))
    for cadence in sorted(value for value in RETIRED_CADENCES[route.source_id]
                          if value >= floor):
        if round_up:
            span = max(cadence, -(-hours // cadence) * cadence)
        elif hours % cadence:
            continue
        else:
            span = hours
        wanted = range(start_hour, start_hour + span + 1, cadence)
        if all(lead in published for lead in wanted):
            return cadence
    return None


@pytest.mark.parametrize("source", sorted(fetch_routes.route_ids()))
def test_a_window_that_names_no_cadence_resolves_as_it_did(source):
    """Defaults stay: every window 2.8.0 resolved without a named cadence
    resolves to the same spacing, on every cycle hour, start lead and
    length swept, and through the door's rounding too.  Only windows it
    could not resolve may now find a spacing."""

    assert set(RETIRED_CADENCES) == set(fetch_routes.route_ids())
    route = fetch_routes.route_for(source)
    floor = route.default_cadence
    for cycle in fetch_routes.planning_cycles(route):
        for start in (0, 1, 2, 3, 6, 12, 30, 78):
            for hours in (*range(0, 25), 36, 48, 60, 90, 120, 150, 240, 252,
                          300, 360):
                for round_up in (False, True):
                    before = _retired_window_cadence(
                        route, start, hours, cycle=cycle, floor=floor,
                        round_up=round_up)
                    if before is None:
                        continue
                    assert fetch_routes.window_cadence(
                        route, start, hours, cycle=cycle,
                        round_up=round_up) == before, (
                            source, cycle.hour, start, hours, round_up)


@pytest.mark.parametrize("source,cadence,hours", [
    ("hrrr-prs", 2, 6), ("rap", 2, 6), ("rrfs", 2, 6), ("icon-eu", 2, 6),
    ("icon-d2", 2, 6),
    ("aigefs", 12, 24), ("aigfs", 12, 24), ("aifs", 12, 24),
    ("gefs", 12, 24), ("ecmwf-open-data", 12, 24), ("gem-gdps", 12, 24),
    ("icon-global", 12, 24), ("icon-global", 6, 12), ("icon-global", 1, 6),
])
def test_each_spacing_a173_names_plans_at_the_route_and_the_config_door(
        source, cadence, hours):
    """The row's list of refusals, each planned: a spacing the publisher
    posts and the preparation takes is a fetch plan, at the route and at
    the [fetch] table check every door shares."""

    route = fetch_routes.route_for(source)
    cycle = fetch_routes.planning_cycles(route)[0]
    leads = fetch_routes.resolve_leads(route, cycle, hours, cadence=cadence)
    assert leads == tuple(range(0, hours + 1, cadence))
    assert fetch.preparation_cadence_refusal(source, cadence) is None
    fetch.validate_fetch_hints(
        {"source": source, "cycle": cycle.strftime("%Y-%m-%dT%H"),
         "hours": hours, "cadence": cadence}, source=f"{source}.toml")


def test_aigefs_fetch_takes_the_twelve_hours_its_preparation_takes(
        tmp_path, capsys):
    """A164 let gpuwm prep take a 12 h aigefs series while gpuwm fetch
    still refused --cadence 12; the fetch door now plans it."""

    rc = cli_main(["fetch", "--source", "aigefs", "--cycle", "2026-08-17T00",
                   "--hours", "24", "--cadence", "12", "--readiness",
                   "--no-probe", "--out", str(tmp_path / "out")])
    out = capsys.readouterr()
    assert rc in (0, 75), out.err
    assert [row["lead"] for row in json.loads(out.out)["leads"]] == [0, 12, 24]
    assert fetch.preparation_cadence_refusal("aigefs", 12) is None


def test_icon_global_takes_hourly_leads_through_f078_and_names_the_ladder_past_it():
    """DWD posts every ICON global field hourly to f078 and three-hourly
    after (listed 2026-10-01 00Z).  The 3 h mapping refused the hourly
    series naming no breakage; the mapping and the normalization now
    declare the publisher's 1 h spacing.  Past f078 the ladder is the
    breakage, and the refusal names the lead and the ladder.  The default
    stays 3 h."""

    from datetime import datetime

    target = packaged_mapping_target(
        get_source_adapter("icon-global").packaged_profile)
    assert target["boundary_interval_seconds"] == 3600
    assert target[BOUNDARY_MULTIPLES_KEY] is True
    assert get_source_adapter("icon-global").forcing_interval_seconds == 10800
    route = fetch_routes.route_for("icon-global")
    assert route.default_cadence == 3
    cycle = datetime(2026, 10, 1, 0)
    assert fetch_routes.resolve_leads(route, cycle, 78, cadence=1) == tuple(
        range(79))
    assert fetch_routes.resolve_leads(route, cycle, 6) == (0, 3, 6)
    with pytest.raises(ValueError) as refused:
        fetch_routes.resolve_leads(route, cycle, 80, cadence=1)
    message = str(refused.value)
    assert "does not publish f079 at --cadence 1" in message
    assert "f000..f078 every 1 h, then f081..f180 every 3 h" in message
    assert "remedy: cadence 2" not in message
    with pytest.raises(ValueError, match="exact multiple of the 3 h"):
        fetch_routes.resolve_leads(route, cycle, 76)


def test_a_spacing_the_decode_refuses_is_refused_naming_the_decode(monkeypatch):
    """The breakage that remains on a route: a mapping that takes one
    spacing only refuses every other before a download, naming itself."""

    import gpuwm.source_authorities as authorities

    exact = {"boundary_interval_seconds": 3600,
             "require_lateral_boundaries": True}
    monkeypatch.setattr(authorities, "packaged_mapping_target",
                        lambda profile_id: exact)
    route = fetch_routes.route_for("hrrr-prs")
    cycle = fetch_routes.planning_cycles(route)[0]
    assert fetch_routes.resolve_leads(route, cycle, 2) == (0, 1, 2)
    with pytest.raises(ValueError) as refused:
        fetch_routes.resolve_leads(route, cycle, 6, cadence=3)
    message = str(refused.value)
    assert "takes 1 h and no other spacing" in message
    assert "refused by the decode" in message
    assert "hrrr-prs-grib2-v1" in message
    assert fetch_routes.window_spacings(route, 0, 6, cycle=cycle) == (1,)
