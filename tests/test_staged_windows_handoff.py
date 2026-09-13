"""Actual acquired handoffs survive a deep Windows cache into preparation."""

from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import fetch_routes, member_prep, runplan, stage_cli
from gpuwm.experiment import RelocationConfig
from gpuwm.mapped_source import read_input_list
from gpuwm.source_adapters import get_source_adapter
from test_fetch_routes import _fake_downloader


@pytest.mark.parametrize("source", ["ecmwf-open-data", "gefs"])
def test_staged_chain_reads_the_published_handoff_from_a_deep_cache(tmp_path, monkeypatch, source):
    cache = tmp_path / ("cache-" + "x" * max(1, 270 - len(str(tmp_path)) - 7))
    fetched = fetch_routes.resolve_request(source, cycle=datetime(2026, 9, 12), hours=0, out=cache)
    config = tmp_path / "config.toml"
    hints = {'source': source, 'cycle': '2026-09-12T00', 'hours': 0}
    if fetched.member is not None:
        hints['member'] = fetched.member
    config.write_text('[fetch]\n' + ''.join(f'{key}={json.dumps(value)}\n' for key, value in hints.items()), encoding='utf-8')
    config.with_suffix('.namelist.wps').write_text('&share /\n', encoding='utf-8')
    calls = []

    def acquire(*args, **kwargs):
        result = fetch_routes.run_plan(fetched, out=cache, downloader=_fake_downloader([]),
            probe=lambda _: False, progress=lambda _: None)
        fetch_routes.write_handoff(fetched, cache)
        return result

    def prepare(arguments):
        Path(arguments[arguments.index('--output-root') + 1]).mkdir(parents=True, exist_ok=True)
        listing = Path(arguments[arguments.index('--input-list') + 1])
        paths = read_input_list(listing)
        assert all(path.is_file() for path in paths)
        assert [path.name for path in paths] == [path.name for path in fetched.primary_files]
        calls.append((listing, paths))
        return {'prepared': True}

    # Weather decoding is outside this transport test. The real member owner
    # still checks source/cycle/member, hashes every file and writes its bound
    # input-list and receipt; only its native GRIB inventory is a fixture.
    monkeypatch.setattr(member_prep, 'verify_member_file',
        lambda *args: SimpleNamespace(to_dict=lambda: {'fixture_inventory': True}))
    monkeypatch.setattr(runplan, '_run_fetch', acquire)
    monkeypatch.setattr(runplan, '_run_prep', prepare)
    monkeypatch.setattr(runplan, '_prepare_stage', lambda root, **kw: kw['run']())
    monkeypatch.setattr(stage_cli, 'resolve_bundle', lambda root: {'layout': 'single'})
    plan = SimpleNamespace(run_options={'data_dir': str(cache), 'geog_root': str(tmp_path)}, config_intent=None)
    observer = SimpleNamespace(enter_stage=lambda *a, **kw: None, finish_stage=lambda *a, **kw: None)
    exp = SimpleNamespace(relocation=RelocationConfig(), domains=())
    result = runplan._staged_chain(plan, config_path=config, exp=exp, observer=observer,
        run_dir=tmp_path / 'run', prepare_only=True)
    assert result['forecast_started'] is False
    assert len(calls) == 1
    if get_source_adapter(source).member_set is not None:
        assert calls[0][0].name.startswith('member-inputs-')
