"""A calendar must not offer a source period its own metadata rules out."""
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from gpuwm import source_adapters
from gpuwm.source_availability import (
    ArchiveWindow, availability, parse_cycle, resolve_latest, validate_cycle,
)
from gpuwm.source_cycles import CycleGrid


NOW = datetime(2026, 9, 6, 15, 25)


def test_current_transport_layout_bound_is_not_the_scientific_record_start():
    for source in ("gfs", "gfs-0p25", "gdas"):
        document = availability(source, 6, now=NOW)
        assert document["earliest"] == "2021-03-22T12"
        assert document["cycle_hours"] == [0, 6, 12, 18]
        assert document["latest_candidate"] == "2026-09-06T12"
        archive = next(row for row in document["transports"] if row["transport"] == "s3")
        assert "/atmos/" in archive["note"]
        assert archive["documentation"]
    # A reader who deliberately selected the rolling endpoint is not handed
    # its archive's range or an automatic probe against a different host.
    rolling = availability("gfs", 6, now=NOW, transport="nomads")
    assert rolling["earliest"] is None
    assert rolling["transports"][0]["retention_hours"] == 240
    assert rolling["latest_supported"] is False


def test_utc_hour_choices_follow_the_requested_forecast_horizon():
    assert availability("hrrr", 18, now=NOW)["cycle_hours"] == list(range(24))
    extended = availability("hrrr", 24, now=NOW)
    assert extended["cycle_hours"] == [0, 6, 12, 18]
    assert extended["latest_candidate"] == "2026-09-06T12"
    assert extended["earliest"] == "2014-07-30T18"
    pressure = availability("hrrr-prs", 6, now=NOW)
    assert pressure["earliest"] == "2014-07-30T18"
    assert any(row["transport"] == "aws" and row["record_start"] for row in pressure["transports"])
    assert not availability("hrrr", 49, now=NOW)["latest_supported"]
    assert not availability("gfs", 385, now=NOW)["latest_supported"]


@pytest.mark.parametrize("source,cycles", [("era5", list(range(24))), ("era5-ml", list(range(24)))])
def test_analysis_latest_keeps_the_whole_period_behind_publication_lag(source, cycles):
    document = availability(source, 24, now=NOW)
    assert document["earliest"] == "1940-01-01T00"
    assert document["cycle_hours"] == cycles
    assert document["latest_label"] == "Latest expected"
    end = parse_cycle(document["latest_candidate"]) + timedelta(hours=24)
    assert end <= NOW - timedelta(hours=120)
    assert any("whole 24-hour" in note for note in document["notes"])
    assert "preliminary ERA5T" in document["transports"][0]["note"]


def test_explicit_hourly_analysis_start_matches_the_actual_native_request():
    from gpuwm import fetch
    cycle = fetch.parse_cycle("2013-05-31T14", "era5")
    document = availability("era5", 11, now=NOW)
    assert validate_cycle(document, "2013-05-31T14:00:00Z")[0] == "2013-05-31T14"
    request = fetch.era5_request_template(
        cycle=cycle, hours=11, cadence=1, area=fetch.parse_area("30,-100,40,-90"))
    times = fetch._era5_times(cycle, 11, 1)
    assert times[0] == cycle and times[-1] == datetime(2013, 6, 1, 1)
    assert len(times) == 12
    assert request["requests"][0]["request"]["time"] == [
        "00:00", "01:00", "14:00", "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"]


def test_undeclared_retention_never_becomes_an_unlimited_archive():
    document = availability("aifs", 6, now=NOW)
    assert document["earliest"] is None
    assert all(row["record_start"] is None for row in document["transports"])
    assert any("not fully declared" in note for note in document["notes"])


def test_metadata_lookup_does_not_contact_provider(monkeypatch):
    from gpuwm import fetch
    monkeypatch.setattr(fetch, "_head_ok", lambda _url: pytest.fail("metadata must stay offline"))
    assert availability("gfs", 6, now=NOW)["probeable"]


def test_latest_uses_the_real_resolver_and_checks_the_required_forecast_hour():
    checked = []

    def provider(url):
        checked.append(url)
        # A deterministic transport seam: the new 12Z cycle is incomplete.
        # The actual resolver must walk its registered candidates and URLs.
        return "gfs.t06z" in url and ".f006" in url

    result = resolve_latest("gfs", 6, now=NOW, probe=provider)
    assert result["selected_cycle"] == "2026-09-06T06"
    assert result["resolution"]["basis"] == "provider_object_probe"
    assert len(checked) >= 2
    assert all("f006" in url for url in checked)
    assert result["resolution"]["objects"][-1]["available"]


def test_analysis_estimate_does_not_masquerade_as_a_provider_probe():
    result = resolve_latest("era5", 24, now=NOW,
                            probe=lambda _url: pytest.fail("CDS exposes no object HEAD probe"))
    assert result["selected_cycle"] == "2026-08-31T12"
    assert result["resolution"]["basis"] == "estimated_publication_schedule"
    assert result["resolution"]["objects"] == []


def test_new_source_and_closed_archive_are_table_work(monkeypatch):
    from gpuwm import fetch_endpoints
    source = replace(source_adapters.get_source_adapter("era5"),
        source_id="new-analysis", aliases=(),
        cadence_mapping="uniform-declared-analysis-series-v1",
        cycle_grid=CycleGrid(hours=(3, 15), delay_hours=2,
                             record_end=datetime(2020, 2, 29, 15)),
        archive_windows=(ArchiveWindow("archive", "1900-01-01T03",
                         "Declared test record", ("https://example.invalid/provider",)),))
    monkeypatch.setattr(source_adapters, "get_source_adapter", lambda _source: source)
    monkeypatch.setattr(fetch_endpoints, "has_ladder", lambda _source: False)
    document = availability("new-analysis", 24, now=NOW)
    assert document["source_id"] == "new-analysis"
    assert document["cycle_hours"] == [3, 15]
    assert document["earliest"] == "1900-01-01T03"
    assert document["latest_candidate"] == "2020-02-28T15"
    result = resolve_latest("new-analysis", 24, now=NOW)
    assert result["selected_cycle"] == "2020-02-28T15"


@pytest.mark.parametrize("value", ["2026-09-05", "2026-09-05T06:30", "2026-09-05T06+02:00", "2026-02-29T06"])
def test_cycle_parser_does_not_silently_rewrite_the_intended_time(value):
    with pytest.raises(ValueError):
        parse_cycle(value)


def test_manual_date_has_the_same_known_bounds_and_horizon_rules():
    document = availability("hrrr", 24, now=NOW)
    for value in ("2014-07-29T00", "2026-09-05T07", "2026-09-07T00"):
        with pytest.raises(ValueError):
            validate_cycle(document, value)
    assert validate_cycle(document, "2026-09-05T06:00:00Z")[0] == "2026-09-05T06"


def test_saved_fetch_span_and_lead_are_not_shortened_to_the_experiment_window(tmp_path):
    from gpuwm.source_availability import _context
    config = tmp_path / "forecast.toml"
    config.write_text('[experiment]\nrun_seconds=21600\n[fetch]\nsource="hrrr"\nhours=24\nforecast_start_hour=6\n', encoding="utf-8")
    source, hours, _ = _context(None, None, config, None)
    assert source == "hrrr" and hours == 30
    assert availability(source, hours, now=NOW)["cycle_hours"] == [0, 6, 12, 18]
    # Editing run duration does not silently reduce an existing fetch request.
    assert _context(None, 3, config, None)[1] == 30
