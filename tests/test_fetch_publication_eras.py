"""Historical publications resolve from rows before network activity."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

import pytest

from gpuwm import fetch, fetch_endpoints, fetch_routes
from gpuwm.source_authorities import packaged_authorities


@pytest.mark.parametrize("cycle", [None, "latest"])
def test_fetch_hints_accept_current_publication(cycle):
    request = {"source": "ecmwf-open-data", "hours": 24}
    if cycle is not None:
        request["cycle"] = cycle
    fetch.validate_fetch_hints(request, source="config")


@pytest.mark.parametrize("source", ["ecmwf-open-data", "ifs"])
@pytest.mark.parametrize("cycle,reason", [
    (datetime(2024, 1, 15), "pressure"),
    (datetime(2024, 2, 15), "pressure"),
    (datetime(2024, 3, 1), "pressure"),
    (datetime(2024, 3, 5), "pressure"),
    (datetime(2024, 3, 5, 6), "soil"),
    (datetime(2024, 3, 10), "soil"),
    (datetime(2024, 3, 18, 6), "soil"),
])
def test_legacy_publication_refuses_preparation_before_probe(source, cycle, reason):
    def no_probe(url):
        pytest.fail(f"unsupported preparation must not probe {url}")

    with pytest.raises(ValueError, match=reason) as error:
        fetch_routes.resolve_request(source, cycle=cycle, hours=0)
    with pytest.raises(ValueError, match=reason):
        fetch.cycle_publication_check(
            fetch_routes.canonical_source(source), cycle, 0, probe=no_probe)
    message = str(error.value)
    assert "2024-03-18T12" in message
    assert "not published" not in message
    assert "0.4 degree" not in message
    if reason == "pressure":
        assert "100, 150, 400, 600 hPa" in message
        assert "vertical coverage" in message
    else:
        assert "one soil temperature layer and two soil moisture layers" in message
        assert "four" in message


def test_pressure_refusal_still_matches_preparation():
    # Retire the refusal if preparation gains this publication's ladder.
    mapping = json.loads(packaged_authorities(
        "ecmwf-open-data-oper-grib2-v1")["mapping"].read_text(encoding="utf-8"))
    vertical = mapping["coordinates"]["vertical"]
    published = {5000, 20000, 25000, 30000, 50000, 70000, 85000, 92500, 100000}
    assert all(not set(levels) <= published
               for levels in [vertical["levels"], *vertical["era_ladders"]])


def test_soil_refusal_still_matches_preparation():
    mapping = json.loads(packaged_authorities(
        "ecmwf-open-data-oper-grib2-v1")["mapping"].read_text(encoding="utf-8"))
    for name, count in (("soil_temperature", 1), ("volumetric_soil_moisture", 2)):
        assert len(mapping["fields"][name]["selectors"]) > count


@pytest.mark.parametrize("hour,stream", [(0, "oper"), (6, "scda"),
                                         (12, "oper"), (18, "scda")])
def test_legacy_publication_paths_and_resolution(hour, stream):
    route = fetch_routes.route_for("ifs")
    era = fetch_routes.publication_era(route, datetime(2024, 1, 15, hour))
    assert era.resolution_degrees == 0.4
    assert era.valid_from == datetime(2022, 1, 25)
    assert era.valid_until == datetime(2024, 1, 31, 6)
    assert era.files[0].path == (
        f"{{YYYYMMDD}}/{{HH}}z/0p4-beta/{stream}/"
        f"{{YYYYMMDDHHMMSS}}-{{F}}h-{stream}-fc.grib2")


@pytest.mark.parametrize("cycle", [datetime(2024, 3, 18, 12), datetime(2024, 3, 20), datetime(2026, 8, 16)])
def test_current_publication_keeps_existing_path(cycle):
    plan = fetch_routes.resolve_request("ifs", cycle=cycle, hours=0)
    assert "/ifs/0p25/oper/" in plan.objects[0].url


def test_era_start_and_utc_boundary():
    route = fetch_routes.route_for("ifs")
    first = fetch_routes.publication_era(route, datetime(2022, 1, 25))
    assert first.resolution_degrees == 0.4
    with pytest.raises(ValueError, match="begins 2022-01-25T00 UTC; this cycle predates it"):
        fetch_routes.resolve_request("ifs", cycle=datetime(2022, 1, 24), hours=0)
    cycle = datetime(2024, 1, 31, 19, tzinfo=timezone(timedelta(hours=-5)))
    assert fetch_routes.publication_era(route, cycle).resolution_degrees == 0.25


def test_an_arbitrary_source_uses_era_rows(monkeypatch):
    original = fetch_routes.route_for("ifs")
    era = fetch_routes.publication_era(original, datetime(2024, 1, 15))
    row = replace(era, label="Earlier publication", prep_refusal="",
                  files=(replace(era.files[0], role="different", magic="BZh",
                                 path="old/{YYYYMMDD}/{FFF}.grib2"),))
    route = replace(original, source_id="synthetic", publication_eras=(row,))
    monkeypatch.setattr(fetch_routes, "_ROUTES", {"synthetic": route})
    monkeypatch.setattr(fetch_endpoints, "ladder", lambda source: route.hosts)
    plan = fetch_routes.resolve_request("synthetic", cycle=datetime(2024, 1, 15), hours=3)
    assert [obj.key for obj in plan.objects] == [
        "old/20240115/000.grib2", "old/20240115/003.grib2"]
    assert len(plan.primary_files) == 2
    assert plan.files == row.files
    assert fetch_routes._magic_for(plan, "different") == "BZh"


@pytest.mark.parametrize("cycle,path", [
    (datetime(2024, 1, 31, 0), "0p4-beta/oper"),
    (datetime(2024, 1, 31, 6), "0p25/scda"),
    (datetime(2024, 2, 15), "0p25/oper"),
    (datetime(2024, 2, 28, 0), "0p25/oper"),
    (datetime(2024, 2, 28, 6), "ifs/0p25/scda"),
])
def test_measured_layout_boundaries(cycle, path):
    era = fetch_routes.publication_era(fetch_routes.route_for("ifs"), cycle)
    assert era.files[0].path.startswith("{YYYYMMDD}/{HH}z/" + path + "/")


@pytest.mark.parametrize("cycle,stream", [
    (datetime(2025, 6, 1, 6), "scda"),
    (datetime(2026, 5, 11, 18), "scda"),
    (datetime(2026, 5, 12, 6), "oper"),
    (datetime(2026, 6, 1, 6), "oper"),
])
def test_short_cycle_stream(cycle, stream):
    plan = fetch_routes.resolve_request("ifs", cycle=cycle, hours=0)
    assert plan.objects[0].key == (
        f"{cycle:%Y%m%d}/{cycle:%H}z/ifs/0p25/{stream}/"
        f"{cycle:%Y%m%d%H}0000-0h-{stream}-fc.grib2")


@pytest.mark.parametrize("cycle,last", [
    (datetime(2024, 1, 30, 6), 90),
    (datetime(2024, 1, 31, 6), 90),
    (datetime(2024, 2, 28, 6), 90),
    (datetime(2024, 3, 5, 6), 90),
    (datetime(2024, 3, 18, 18), 90),
    (datetime(2024, 11, 11, 18), 90),
    (datetime(2024, 11, 12, 6), 144),
    (datetime(2025, 6, 1, 6), 144),
    (datetime(2026, 5, 12, 6), 144),
    (datetime(2024, 11, 12, 0), 240),
    (datetime(2024, 11, 12, 12), 360),
])
def test_era_lead_limits_match_mirror(cycle, last):
    route = fetch_routes.route_for("ifs")
    assert fetch_routes.ladder_for(route, cycle)[-1] == last
    with pytest.raises(ValueError, match=f"through f{last:03d}"):
        fetch_routes.resolve_leads(route, cycle, last + 6, cadence=6)
