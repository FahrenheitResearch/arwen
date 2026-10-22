"""WOA18 MLD staging: cache first, retried multi-source fetch, a named refusal.

Breakage these guard (2.9.4 WOOF readiness, 2026-10-22): the default
mixed-layer ocean read its WOA18 month from NCEI THREDDS once, at forecast
start, with no retry, no second route and no way to stage it ahead of time;
THREDDS answered 503 for an hour and a Mexico City run died at its first
step after 60 minutes of preparation.
"""
import hashlib
import io
from urllib.error import HTTPError, URLError

import pytest

from gpuwm.ocean.mixed_layer import climatology


PAYLOAD = b"CDF\x01 fake WOA18 MLD month for the staging contract"


class _FakeDataset:
    variables = {"M_an": None, "lat": None, "lon": None}
    title = "World Ocean Atlas 2018 : ocean_mixed_layer_thickness test"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def pinned(monkeypatch):
    """Month 7 pinned to PAYLOAD; NetCDF decode stubbed; no real sleeping."""
    import gpuwm.netcdf_bridge as bridge
    monkeypatch.setattr(climatology, "MONTH_PINS",
                        {7: (len(PAYLOAD), hashlib.sha256(PAYLOAD).hexdigest())})
    monkeypatch.setattr(bridge, "open_dataset", lambda path: _FakeDataset())
    monkeypatch.delenv(climatology.URL_BASE_ENV, raising=False)
    monkeypatch.delenv(climatology.CACHE_ENV, raising=False)


def _opener(script, calls):
    """paced_urlopen stand-in: ``script`` maps a URL to a list of outcomes."""
    def fake(url, timeout=None):
        calls.append(url)
        outcome = script[url].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return io.BytesIO(outcome)
    return fake


def _http(url, code):
    return HTTPError(url, code, "test", {}, None)


def test_cache_hit_never_touches_the_network(pinned, monkeypatch, tmp_path):
    import gpuwm.nomads_governor as governor
    (tmp_path / climatology.filename(7)).write_bytes(PAYLOAD)

    def refuse(*args, **kwargs):
        raise AssertionError("a staged month must not reach the network")

    monkeypatch.setattr(governor, "paced_urlopen", refuse)
    monkeypatch.setenv(climatology.URL_BASE_ENV, "http://127.0.0.1:9/dead")
    assert climatology.acquire(7, cache=tmp_path) == tmp_path / climatology.filename(7)


def test_cache_env_is_the_root_fetch_tables_stages_into(pinned, monkeypatch, tmp_path):
    monkeypatch.setenv(climatology.CACHE_ENV, str(tmp_path))
    assert climatology.cache_root() == tmp_path
    (tmp_path / climatology.filename(7)).write_bytes(PAYLOAD)
    assert climatology.acquire(7) == tmp_path / climatology.filename(7)


def test_a_503_is_retried_and_the_second_ncei_route_is_used(pinned, monkeypatch, tmp_path):
    import gpuwm.nomads_governor as governor
    https, thredds = climatology.source_urls(7)
    assert "thredds-ocean" in thredds and "/data/oceans/woa/WOA18/" in https
    calls, sleeps = [], []
    script = {https: [_http(https, 503), _http(https, 503)],
              thredds: [URLError("timed out"), PAYLOAD]}
    monkeypatch.setattr(governor, "paced_urlopen", _opener(script, calls))
    path = climatology.acquire(7, cache=tmp_path, sleeper=sleeps.append)
    assert path.read_bytes() == PAYLOAD
    assert calls == [https, thredds, https, thredds]
    assert sleeps == [climatology.RETRY_BACKOFF_S[1]]
    receipt = path.with_suffix(".json").read_text()
    assert thredds in receipt and '"pinned": true' in receipt


def test_wrong_bytes_are_never_installed(pinned, monkeypatch, tmp_path):
    import gpuwm.nomads_governor as governor
    https, thredds = climatology.source_urls(7)
    calls = []
    script = {https: [PAYLOAD + b"truncated?"], thredds: [PAYLOAD]}
    monkeypatch.setattr(governor, "paced_urlopen", _opener(script, calls))
    path = climatology.acquire(7, cache=tmp_path, sleeper=lambda s: None)
    assert path.read_bytes() == PAYLOAD
    assert calls == [https, thredds]


def test_damaged_cache_is_moved_aside_and_refetched(pinned, monkeypatch, tmp_path):
    import gpuwm.nomads_governor as governor
    (tmp_path / climatology.filename(7)).write_bytes(b"half a file")
    url = climatology.source_urls(7)[0]
    monkeypatch.setattr(governor, "paced_urlopen", _opener({url: [PAYLOAD]}, []))
    assert climatology.acquire(7, cache=tmp_path).read_bytes() == PAYLOAD
    assert len(list(tmp_path.glob("*.invalid-*.nc"))) == 1


def test_final_failure_names_fetch_tables_ocean(pinned, monkeypatch, tmp_path):
    import gpuwm.nomads_governor as governor
    thredds, https = climatology.source_urls(7)
    rounds = len(climatology.RETRY_BACKOFF_S)
    calls, sleeps = [], []
    script = {thredds: [_http(thredds, 503)] * rounds,
              https: [URLError("timed out")] * rounds}
    monkeypatch.setattr(governor, "paced_urlopen", _opener(script, calls))
    with pytest.raises(climatology.OceanClimatologyUnavailable,
                       match="gpuwm fetch-tables --ocean"):
        climatology.acquire(7, cache=tmp_path, sleeper=sleeps.append)
    assert len(calls) == 2 * rounds
    assert sleeps == [s for s in climatology.RETRY_BACKOFF_S if s]
    assert not list(tmp_path.glob("*.nc*"))


def test_a_404_is_not_retried(pinned, monkeypatch, tmp_path):
    import gpuwm.nomads_governor as governor
    monkeypatch.setenv(climatology.URL_BASE_ENV, "https://mirror.invalid/woa")
    [url] = climatology.source_urls(7)
    assert url == "https://mirror.invalid/woa/" + climatology.filename(7)
    calls = []
    monkeypatch.setattr(governor, "paced_urlopen", _opener({url: [_http(url, 404)]}, calls))
    with pytest.raises(climatology.OceanClimatologyUnavailable, match="fetch-tables --ocean"):
        climatology.acquire(7, cache=tmp_path, sleeper=lambda s: None)
    assert calls == [url]


def test_fetch_tables_ocean_stages_every_month_from_a_directory(pinned, monkeypatch, tmp_path):
    import argparse
    from gpuwm import table_assets
    source, cache = tmp_path / "src", tmp_path / "cache"
    (source / "ocean").mkdir(parents=True)
    pins = {}
    for month in range(1, 13):
        body = PAYLOAD + bytes([month])
        pins[month] = (len(body), hashlib.sha256(body).hexdigest())
        (source / "ocean" / climatology.filename(month)).write_bytes(body)
    monkeypatch.setattr(climatology, "MONTH_PINS", pins)
    args = argparse.Namespace(ocean=True, ocean_only=True, ocean_root=str(cache),
                              from_dir=str(source))
    assert table_assets.fetch_tables_main(args) == 0
    root, valid, damaged = climatology.staged_state(cache)
    assert root == cache and valid == list(range(1, 13)) and damaged == []
    # Idempotent: a second run verifies and installs nothing new.
    assert table_assets.fetch_tables_main(args) == 0


def test_fetch_tables_ocean_refuses_with_a_nonzero_exit_when_a_month_is_missing(
        pinned, monkeypatch, tmp_path, capsys):
    import argparse
    from gpuwm import table_assets
    args = argparse.Namespace(ocean=True, ocean_only=True, ocean_root=str(tmp_path / "c"),
                              from_dir=str(tmp_path))
    assert table_assets.fetch_tables_main(args) == 2
    assert "REFUSED" in capsys.readouterr().out


def test_fetch_tables_parser_accepts_the_ocean_flags():
    import argparse
    from gpuwm import table_assets
    parser = argparse.ArgumentParser()
    table_assets.register_cli(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["fetch-tables", "--ocean", "--ocean-only", "--ocean-root", "X"])
    assert args.ocean and args.ocean_only and args.ocean_root == "X"


def test_setup_stages_the_ocean_with_the_tables(monkeypatch):
    from gpuwm import setup_cli
    seen = {}

    def fake_run_step(module_name, overrides):
        seen[module_name] = dict(overrides)
        return 0, ""

    monkeypatch.setattr(setup_cli, "_run_step", fake_run_step)
    import gpuwm.doctor as doctor
    monkeypatch.setattr(doctor, "collect_checks", lambda: [])
    import argparse
    setup_cli.setup_main(argparse.Namespace(with_geog=False, with_earth2=False,
                                            from_dir=None, explain=False))
    assert seen["gpuwm.table_assets"].get("ocean") is True


def test_doctor_row_names_the_staging_command(pinned, monkeypatch, tmp_path):
    from gpuwm import doctor
    monkeypatch.setenv(climatology.CACHE_ENV, str(tmp_path))
    check = doctor._ocean_climatology_check()
    assert check.status == "missing" and check.action == "gpuwm fetch-tables --ocean"
    assert check.blocking is False and check.severity == "degraded"
    pins = {}
    for month in range(1, 13):
        body = PAYLOAD + bytes([month])
        pins[month] = (len(body), hashlib.sha256(body).hexdigest())
        (tmp_path / climatology.filename(month)).write_bytes(body)
    monkeypatch.setattr(climatology, "MONTH_PINS", pins)
    assert doctor._ocean_climatology_check().status == "verified"


def test_fetch_tables_ocean_stops_hammering_a_dead_source_after_one_month(
        pinned, monkeypatch, tmp_path):
    import gpuwm.nomads_governor as governor
    calls = []

    def dead(url, timeout=None):
        calls.append(url)
        raise HTTPError(url, 503, "Service Unavailable", {}, None)

    monkeypatch.setattr(governor, "paced_urlopen", dead)
    monkeypatch.setattr(climatology, "RETRY_BACKOFF_S", (0.0,))
    monkeypatch.setattr(climatology.acquire, "__kwdefaults__",
                        {**climatology.acquire.__kwdefaults__, "backoff": (0.0,)})
    (tmp_path / climatology.filename(7)).write_bytes(PAYLOAD)
    lines = []
    staged, failed = climatology.stage_all(cache=tmp_path, progress=lines.append)
    assert staged == [7] and failed == [m for m in range(1, 13) if m != 7]
    assert len(calls) == 2          # month 01 only, both routes, then stop
    assert sum("not attempted" in line for line in lines) == 10
