"""Cross-owner command registration and diagnostic licence boundaries."""
from pathlib import Path

import pytest


def test_full_cli_has_one_diagnostic_command_with_both_public_forms():
    from gpuwm.cli import build_parser
    parser = build_parser()
    direct = parser.parse_args(['earth2-diagnose', '--model', 'derived-ws10m',
                               '--input', 'forecast.nc', '--out', 'diagnostic'])
    consumer = parser.parse_args(['earth2-diagnose', 'forecast.nc', '--for',
                                 'derived-ws10m', '--out', 'diagnostic'])
    assert direct.func is consumer.func
    assert direct.model == 'derived-ws10m' and direct.input == Path('forecast.nc')
    assert consumer.consumers == 'derived-ws10m' and consumer.store == Path('forecast.nc')
    assert parser.parse_args(['earth2', '--model', 'ucast', '--cycle', '2024-09-24T12',
                             '--hours', '12', '--area', '24,-90,30,-84']).model == 'ucast'
    saved = parser.parse_args(['earth2', '--spec', 'spec.json', '--earth2-input', 'completed'])
    assert saved.earth2_input == Path('completed')


def test_restricted_diagnostic_is_refused_through_both_forms_before_io(tmp_path):
    from gpuwm.earth2_diagnose import resolve
    from gpuwm.earth2_diagnostics import run_diagnostic
    with pytest.raises(ValueError, match='NVIDIA Community Model License'):
        resolve('windgust-afno')
    destination = tmp_path / 'diagnostic'
    with pytest.raises(ValueError, match='NVIDIA Community Model License'):
        run_diagnostic('windgust', tmp_path/'absent.nc', destination)
    assert not destination.exists()


def test_exact_diagnostic_licence_reaches_worker_and_keeps_other_refusals(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from gpuwm import earth2_diagnostics as diagnostic
    from gpuwm.earth2_diagnose import resolve
    from gpuwm.earth2_job import _preflight
    from gpuwm.earth2_door import build_spec
    from gpuwm.earth2_licensing import licence_scope
    licence = 'NVIDIA Community Model License'
    spec = build_spec(model='fcn3', cycle='2024-09-24T12', area='24,-90,30,-84', diagnose='windgust')
    with pytest.raises(ValueError, match='--accept-licence'):
        _preflight(spec)
    with licence_scope(['NVIDIA Open Model License']):
        with pytest.raises(ValueError, match=licence):
            resolve('windgust-afno')
    monkeypatch.setattr(diagnostic, 'earth2_python', lambda: 'unused-python')
    monkeypatch.setattr(diagnostic.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    destination = tmp_path/'accepted'
    with licence_scope([licence]):
        _preflight(spec)
        assert resolve('windgust-afno') == ['windgust-afno']
        assert diagnostic.run_diagnostic('windgust', tmp_path/'forecast.nc', destination) == 0
        request = json.loads((destination/'request.json').read_text())
        assert request['accept_licence'] == [licence]
        assert request['row']['licence_class'] == 'restricted'
        row = dict(request['row'], runtime_refusal='test missing required operator')
        monkeypatch.setattr(diagnostic, 'diagnostic_rows', lambda: {'arbitrary-diagnostic': row})
        with pytest.raises(ValueError, match='missing required operator'):
            diagnostic.run_diagnostic('arbitrary-diagnostic', tmp_path/'forecast.nc', tmp_path/'refused')
    with pytest.raises(ValueError, match='--accept-licence'):
        resolve('windgust-afno')


def test_diagnostic_worker_checks_same_policy_before_optional_imports(tmp_path):
    import json
    from gpuwm.earth2_door import diagnostic_rows
    from gpuwm.earth2_diagnostic_runner import main
    request = tmp_path/'request.json'
    request.write_text(json.dumps({'model': 'windgust', 'row': diagnostic_rows()['windgust']}))
    with pytest.raises(ValueError, match='--accept-licence'):
        main(request)


def test_job_stage_forwards_runtime_acceptance_to_diagnostic_cli(tmp_path, monkeypatch):
    import sys
    from gpuwm import earth2_job
    from gpuwm.earth2_licensing import licence_scope
    launched = []
    class Completed:
        pid = 0
        returncode = 0
        def __init__(self, command, **kwargs):
            launched.append(command)
        def wait(self, **kwargs):
            return 0
    monkeypatch.setattr(earth2_job.subprocess, 'Popen', Completed)
    receipt = {'stages': []}
    with licence_scope(['NVIDIA Community Model License']):
        earth2_job._command('fetch-model', [sys.executable, '-m', 'gpuwm.earth2_job', 'fetch',
            'job-spec.json', str(tmp_path/'source')], tmp_path, receipt)
        earth2_job._command('diagnose', [sys.executable, '-m', 'gpuwm', 'earth2-diagnose',
            '--model', 'windgust', '--input', 'forecast.nc', '--out', str(tmp_path/'out')], tmp_path, receipt)
    assert launched[0][-2:] == launched[1][-2:] == ['--accept-licence', 'NVIDIA Community Model License']
    assert receipt['stages'][0]['unwrapped_command'] == launched[0]
