"""Operational worker failures keep their cause, evidence and failure status."""
from dataclasses import asdict
import os
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('code, expected, message', [
    (0, 0, ''), (1, 1, 'exit 1'), (-15, 143, 'SIGTERM'), (-9, 137, 'signal 9' if os.name == 'nt' else 'SIGKILL')])
def test_wrf_worker_exit_is_a_shell_status_with_a_named_signal(code, expected, message, capsys):
    from gpuwm.wrfinput_forecast import worker_exit_status
    assert worker_exit_status(code) == expected
    assert message in capsys.readouterr().err


def test_actual_radiation_substitution_is_shown_without_full_report(capsys):
    from gpuwm.namelist_import import Substitution, SubstitutionReport
    from gpuwm.wrfinput_forecast import announce_wrf_substitutions
    report = SubstitutionReport((Substitution('ra_lw_physics', 4, 'WRF RRTMG',
        'ra_lw_physics', 4, 'RTE+RRTMGP'),), ())
    announce_wrf_substitutions(SimpleNamespace(substitution_report=report), 'input/wrf-import.json')
    output = capsys.readouterr().out
    assert 'WRF RRTMG' in output and 'RTE+RRTMGP' in output
    assert 'input/wrf-import.json' in output
    assert 'Not implemented' not in output
    assert asdict(report)['substitutions'][0]['wrf_name'] == 'WRF RRTMG'


@pytest.mark.parametrize('explain', [False, True])
def test_supervisor_worker_value_error_keeps_failure_status_and_capsule(monkeypatch, capsys, explain):
    import gpuwm.cli as cli
    from gpuwm import capabilities, provenance_gate
    from gpuwm.supervisor import SupervisorError
    monkeypatch.setattr(capabilities, 'require_for_command', lambda *a: None)
    monkeypatch.setattr(provenance_gate, 'announce', lambda *a: None)
    def fail(args):
        raise SupervisorError('worker exited with status 1: ValueError: tile plus halo exceeds domain '
                              '(failure capsule: out/failure.json)')
    monkeypatch.setattr(cli, '_dispatch', fail)
    assert cli.main(['run', 'case.toml', *(['--explain'] if explain else [])]) == 1
    output = capsys.readouterr().err
    assert 'tile plus halo exceeds domain' in output
    assert 'out/failure.json' in output
    assert ('Traceback' in output) == explain


def test_unrelated_runtime_error_still_propagates(monkeypatch):
    import gpuwm.cli as cli
    from gpuwm import capabilities, provenance_gate
    monkeypatch.setattr(capabilities, 'require_for_command', lambda *a: None)
    monkeypatch.setattr(provenance_gate, 'announce', lambda *a: None)
    def fail(args):
        raise RuntimeError('unexpected internal defect')
    monkeypatch.setattr(cli, '_dispatch', fail)
    with pytest.raises(RuntimeError, match='unexpected internal defect'):
        cli.main(['run', 'case.toml'])
