"""Original step metadata supplies complete per-member C1 intervals."""
import json
import time

import pytest

from gpuwm.ensemble.timeline import member_timeline


def test_member_relay_drains_original_steps_with_card_and_distinct_span(tmp_path, monkeypatch):
    ledger = tmp_path / 'timeline.jsonl'
    monkeypatch.setenv('GPUWM_TIME_LEDGER', str(ledger))
    monkeypatch.setenv('GPU_MUTEX_CARDS', '4,7')
    member = tmp_path / 'member'
    member.mkdir()
    with member_timeline(member, member=3, card=1):
        stamp = time.time() * 1000
        rows = [{'event': 'step', 'domain': 1, 'emitted_unix_ms': stamp,
                 'step_wall_seconds': .25},
                {'event': 'step', 'domain': 1, 'emitted_unix_ms': stamp + 10,
                 'step_wall_seconds': .01}]
        (member / 'progress.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    steps = [r for r in events if r['stage'] == 'forecast.integrate']
    assert [r['phase'] for r in steps] == ['begin', 'end']
    assert all(r['card'] == 7 and r['attrs']['member_id'] == 3 for r in events)
    assert all(r['attrs']['cuda_device_id'] == 1 for r in events)
    assert all(r['attrs']['span_id'] == 'ensemble-member-3' for r in events)
    assert steps[-1]['t_mono'] - steps[0]['t_mono'] == pytest.approx(.26, abs=.002)


def test_broken_ledger_keeps_original_forecast_exception(tmp_path, monkeypatch, capsys):
    impossible = tmp_path / 'regular-file'
    impossible.write_bytes(b'cannot contain a ledger')
    monkeypatch.setenv('GPUWM_TIME_LEDGER', str(impossible / 'timeline.jsonl'))
    with pytest.raises(RuntimeError, match='original forecast failed'):
        with member_timeline(tmp_path, member=0, card=0):
            raise RuntimeError('original forecast failed')
    assert str(impossible / 'timeline.jsonl') in capsys.readouterr().err
