"""Actual stage failure transport and frozen publication input binding."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

from gpuwm import cli, go_cli, local_da_runtime
from test_local_da_runtime import Backend, saved


def test_stage_failure_returns_protocol_json_and_preserves_execution_failure(tmp_path, monkeypatch, capsys):
    path, _ = saved(tmp_path)
    class FailingPreparation(Backend):
        def prepare(self):
            go_cli._run_stage('prepare', [sys.executable, '-c', 'raise SystemExit(7)'], explain=False)
    launch = local_da_runtime.launch
    monkeypatch.setattr(local_da_runtime, 'launch', lambda path: launch(path, backend=FailingPreparation()))
    assert cli.main(['local-da', '--launch', str(path)]) == 1
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert document['code'] == 'LOCAL_DA_ERROR'
    assert document['details']['stage_exit_code'] == 7
    assert document['forecast_started'] is False
    assert 'Traceback' not in captured.err
    execution = json.loads((path.parent / 'execution.json').read_text())
    assert execution['status'] == 'FAILED' and 'stage exited 7' in execution['error']


def test_unexpected_programming_failure_is_not_swallowed(monkeypatch):
    def fail(path):
        raise LookupError('unexpected programming failure')
    monkeypatch.setattr(local_da_runtime, 'launch', fail)
    with pytest.raises(LookupError, match='unexpected programming failure'):
        cli.main(['local-da', '--launch', 'unused.json'])


def test_production_context_binds_the_frozen_window_and_never_refetches_a_lost_one(tmp_path):
    from gpuwm.local_da_fetch import WINDOW_SCHEMA
    from gpuwm.output_identity import file_record
    from gpuwm.local_da import PlanError
    prepared = tmp_path / 'prepared'; prepared.mkdir()
    member = tmp_path / 'member_000'; member.mkdir()
    asset = tmp_path / 'input.bin'; asset.write_bytes(b'input')
    for path in [tmp_path / 'experiment.toml', tmp_path / 'ensemble.toml',
                 prepared / 'proof.json', member / 'surface-end.npz']:
        path.write_bytes(b'identity fixture')
    when = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
    window_path = tmp_path / 'observations/cycle_000/window.json'
    calls = []
    def window(index, instant, states):
        calls.append((index, instant))
        window_path.parent.mkdir(parents=True, exist_ok=True)
        window_path.write_text(json.dumps(dict(schema=WINDOW_SCHEMA,
            assets=[dict(path=str(asset), sha256=file_record(asset)['sha256'])])))
    backend = SimpleNamespace(root=tmp_path, go={'prepared': prepared},
        inputs=SimpleNamespace(proof_path=prepared / 'proof.json'),
        plan={'review_sha256': 'a' * 64, 'analysis_times': ['2026-09-10T12:00:00Z'],
              'cadence_settings': {'applied': {'rtps_alpha': .9}},
              'selected': {'covariance_members': 8}, 'request': {'base_seed': 7}},
        analysis_times=[when], grid=SimpleNamespace(identity_sha256=lambda: 'b' * 64),
        _observation_window=window)
    states = {0: {'member_dir': str(member)}}
    context = local_da_runtime.PreparedBackend.analysis_context(backend, 0, states, recovering=False)
    assert context['analysis_time'] == backend.plan['analysis_times'][0]
    assert context['method_settings'] == {'rtps_alpha': .9}
    assert context['base_seed'] == 7 and context['covariance_members'] == 8
    assert {row['path'] for row in context['assets']} == {str(path) for path in
        [window_path, asset, prepared / 'proof.json', tmp_path / 'experiment.toml',
         tmp_path / 'ensemble.toml', member / 'surface-end.npz']}
    assert calls == [(0, when)]
    window_path.unlink()
    with pytest.raises(PlanError, match='original observation window is missing'):
        local_da_runtime.PreparedBackend.analysis_context(backend, 0, states, recovering=True)
    assert calls == [(0, when)]
