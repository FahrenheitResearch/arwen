"""Finite initialization follows actual publication without moving valid time."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import re
import tomllib

import pytest

from gpuwm import fetch, local_da
from tests.test_local_da_plan import availability, price


def request():
    return local_da.Request(epoch='2026-09-13T02:00:00Z',
        region=(-102.72238935377757, 35.094089288936146, -95.72540303303529, 38.426167555765915),
        scale=1, card=local_da.Card(vram_gib=10., free_gib=6.494140625, host_gib=32.),
        budget_seconds=3600., forecast_seconds=1800.)


def test_initial_lead_is_not_added_to_the_requested_duration():
    req = request()
    text = local_da.configuration(req, local_da.derive_rung(req, 1))[0]
    raw = tomllib.loads(text)
    cycle = datetime.fromisoformat(raw['fetch']['cycle']).replace(tzinfo=timezone.utc)
    initial = datetime.fromisoformat(req.epoch)
    assert cycle + timedelta(hours=raw['fetch'].get('forecast_start_hour', 0)) == initial
    end = cycle + timedelta(hours=raw['fetch'].get('forecast_start_hour', 0) + raw['fetch']['hours'])
    assert end == initial + timedelta(hours=1), raw['fetch']
    assert raw['experiment']['run_seconds'] == 2700.


def test_actual_cli_selects_older_complete_cycle_with_recomputed_leads(tmp_path, monkeypatch, capsys):
    seen = []
    def probe(url):
        seen.append(url)
        return 'gfs.20260912/18/' in url and bool(re.search(r'f00[89](?:$|[.?])', url))
    monkeypatch.setattr(fetch, '_head_answer', probe)
    monkeypatch.setattr(local_da, 'observation_routes', availability)
    monkeypatch.setattr(local_da, 'price_rung', lambda req, rung, exp, streams, **kw: price(req, rung, exp, streams))
    from dataclasses import asdict
    document = dict(schema='arwen.local-da-request.v1', **asdict(request()))
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(document))
    parser = argparse.ArgumentParser()
    local_da.register_cli(parser.add_subparsers())
    args = parser.parse_args(['local-da', '--request-json', str(path), '--out', str(tmp_path / 'case')])
    assert local_da.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    raw = tomllib.loads(result['configuration']['experiment'])
    assert raw['fetch']['cycle'] == '2026-09-12T18', raw['fetch']
    assert raw['fetch']['forecast_start_hour'] == 8 and raw['fetch']['hours'] == 1
    assert raw['experiment']['start_time'] == datetime(2026, 9, 13, 2)
    assert result['selected']['members'] == 1 and result['selected']['forecast_seconds'] == 1800.
    assert seen and any('f009' in url for url in seen) and any('f008' in url for url in seen)
    from gpuwm.local_da_runtime import read_plan
    monkeypatch.setattr(fetch, '_head_answer', lambda url: pytest.fail('saved review must not select another cycle'))
    loaded = read_plan(tmp_path / 'case/local-da.json')
    assert loaded['background']['selection']['cycle'] == '2026-09-12T18:00:00+00:00'


@pytest.mark.parametrize('missing', [2, 3])
def test_missing_initial_or_final_object_selects_an_older_covering_window(missing):
    from gpuwm.background_contract import plan
    def probe(url):
        return not ('gfs.20260913/00/' in url and f'f{missing:03d}' in url)
    selected = plan('gfs', init=datetime(2026, 9, 13, 2, tzinfo=timezone.utc),
        now=datetime(2026, 9, 13, 6, tzinfo=timezone.utc), run_seconds=2700., probe=probe)
    assert selected.cycle == '2026-09-12T18:00:00+00:00'
    assert selected.forecast_leads == (8, 9)


def test_explicit_young_but_available_cycle_ignores_the_delay_estimate():
    from gpuwm.background_contract import plan
    when = datetime(2026, 9, 13, tzinfo=timezone.utc)
    selected = plan('gfs', init=when, cycle=when, now=when, run_seconds=900., probe=lambda url: True)
    assert selected.cycle == when.isoformat() and selected.forecast_leads == (0, 1)


def test_explicit_unavailable_cycle_is_not_silently_replaced():
    from gpuwm.background_contract import BackgroundWindowError, plan
    when = datetime(2026, 9, 13, tzinfo=timezone.utc)
    with pytest.raises(BackgroundWindowError, match='not all available'):
        plan('gfs', init=when, cycle=when, now=when, run_seconds=900., probe=lambda url: False)


def test_table_route_checks_selected_member_and_all_frames():
    from gpuwm.background_contract import plan
    seen = []
    def probe(url):
        seen.append(url)
        return True
    selected = plan('gefs', init=datetime(2026, 9, 12, 6, tzinfo=timezone.utc),
        now=datetime(2026, 9, 12, 6, tzinfo=timezone.utc), run_seconds=3600., member='p01', probe=probe)
    assert selected.member == 'p01'
    assert seen and all('gep01' in url for url in seen)
    assert any('f000' in url for url in seen) and any('f003' in url for url in seen)
