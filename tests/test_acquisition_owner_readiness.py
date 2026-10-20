"""Generated ensemble readiness asks the declared acquisition owner, without GPU work."""
from dataclasses import replace
from datetime import datetime
import json
import sys
from types import SimpleNamespace

import pytest

from gpuwm import earth2_run, source_adapters, source_readiness

ARGV = [
    'ensemble', '--recipe', 'wofs', '--source', 'earth2-fcn3:18,gefs:18',
    '--members', '36', '--init', '2022-06-13T18', '--center', '44.0,-94.5',
    '--hours', '6', '--extent-km', '900', '--readiness',
]


@pytest.mark.parametrize('code,state', [(0, 'ready'), (75, 'waiting'), (2, 'refused')])
def test_generated_readiness_preserves_donor_verdict(monkeypatch, code, state):
    original = source_readiness.readiness
    calls = []

    def donor(source, cycle, hours, **options):
        calls.append((source, cycle, hours, options))
        return {'source': source, 'cycle': '2022-06-13T18', 'state': state,
                'ready': code == 0, 'refusal': 'donor unavailable' if code == 2 else None,
                'retry_after_seconds': 30 if code == 75 else None}, code

    monkeypatch.setattr(source_readiness, 'readiness', donor)
    document, result = original('earth2-fcn3', '2022-06-13T18', 6, member='17',
                                no_probe=True, now=datetime(2026, 10, 19))
    assert result == code and document['state'] == state
    assert document['source'] == 'earth2-fcn3' and document['member'] == '17'
    assert calls[0][:3] == ('era5', datetime(2022, 6, 13, 18), 0)
    assert calls[0][3]['provider'] == 'arco'
    assert calls[0][3]['no_probe'] is True
    assert 'member' not in calls[0][3]
    assert document['generation']['description'] == 'generated on the card at run time'
    assert document['generation']['readiness']['source'] == 'era5'


def test_new_adapter_owner_controls_both_readiness_and_inventory(monkeypatch):
    """An arbitrary row needs neither a fetch route nor a generic model branch."""
    from gpuwm import fetch_routes, runplan
    adapter = replace(source_adapters.get_source_adapter('earth2-fcn3'),
                      source_id='synthetic-generator', acquisition_owner='synthetic_owner')
    original = source_adapters.get_source_adapter
    monkeypatch.setattr(source_adapters, 'get_source_adapter',
                        lambda source: adapter if source == adapter.source_id else original(source))
    expected = ({'source': adapter.source_id, 'state': 'waiting'}, 75)
    monkeypatch.setitem(sys.modules, 'synthetic_owner', SimpleNamespace(
        readiness=lambda *args, **kwargs: expected,
        acquisition_facts=lambda source: {'generation': 'synthetic owner description'}))
    assert source_readiness.readiness(adapter.source_id, '2022-06-13T18', 6) == expected
    facts = runplan._source_fetch_facts(fetch_routes, adapter.source_id)
    assert facts['refusal'] is None and facts['kind'] == 'acquisition_owner'
    assert facts['generation'] == 'synthetic owner description'


def test_exact_mixed_command_and_sources_without_gpu(monkeypatch, tmp_path, capsys):
    from gpuwm import cli, domain_wizard
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GPUWM_NO_LOCAL_GPU', '1')
    # Geometry sizing is orthogonal to source readiness. Supply an explicit
    # capacity fixture at its existing seam; fail if anything invokes CUDA.
    budget = domain_wizard.resolve_sizing_budget(card='32gb', vram_gib=None)
    monkeypatch.setattr(domain_wizard, 'resolve_sizing_budget', lambda *args, **kwargs: budget)
    monkeypatch.setattr(domain_wizard, 'device_memory_probe_subprocess',
                        lambda *a, **k: pytest.fail('readiness must not probe CUDA'))
    monkeypatch.setattr(earth2_run, 'earth2_python',
                        lambda: pytest.fail('readiness must not load a model runtime'))
    monkeypatch.setattr(source_readiness, 'need_answer',
                        lambda *a, **k: {'answer': 'posted', 'endpoint': 'test'})
    assert cli.main(ARGV) in (0, 75)
    output = capsys.readouterr().out
    document = json.loads(output[output.index('\n{') + 1:])
    members = document['recipe']['member_windows']
    assert len(members) == 36
    generated = [row['readiness'] for row in members
                 if row['readiness']['source'] == 'earth2-fcn3']
    assert len(generated) == 18
    assert all(row['generation']['donor']['provider'] == 'arco' for row in generated)
    assert all(row['acquisition_owner'] == 'gpuwm.earth2_run' for row in generated)
    assert cli.main(['sources', 'earth2-fcn3']) == 0
    text = capsys.readouterr().out
    assert 'generated on the card at run time' in text
    assert 'gpuwm.earth2_run' in text and 'era5-arco' in text
    assert 'bring the files yourself' not in text
