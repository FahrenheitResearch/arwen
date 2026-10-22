"""Optional package provenance must never prevent a job receipt being written."""
import json
from types import SimpleNamespace

import pytest

from gpuwm import build_provenance


def owner(distribution_version='2.9.3', **stamp):
    return SimpleNamespace(version=distribution_version, read_text=lambda name: json.dumps(stamp))


def test_matching_build_stamp_uses_owning_distribution_without_subprocess(monkeypatch, capsys):
    monkeypatch.setattr(build_provenance, 'owning_distribution', lambda: owner(
        schema=build_provenance.BUILD_SCHEMA, version='2.9.3', commit='a' * 40))
    identity = build_provenance.package_build_identity()
    assert identity['version'] == '2.9.3'
    assert identity['commit'] == 'a' * 40
    assert capsys.readouterr().err == ''


@pytest.mark.parametrize('stamp', [{}, {'commit': 'bad'},
    {'schema': build_provenance.BUILD_SCHEMA, 'version': 'other', 'commit': 'a' * 40}])
def test_missing_or_mismatched_stamp_has_unknown_and_receipt_line(monkeypatch, capsys, stamp):
    monkeypatch.setattr(build_provenance, 'owning_distribution', lambda: owner(**stamp))
    identity = build_provenance.package_build_identity()
    assert identity['commit'] == 'unknown'
    assert identity['receipt_line'] in capsys.readouterr().err


def test_unreadable_metadata_is_optional(monkeypatch, capsys):
    def fail():
        raise OSError('unreadable package metadata')
    monkeypatch.setattr(build_provenance, 'owning_distribution', fail)
    identity = build_provenance.package_build_identity()
    assert identity['version'] == identity['commit'] == 'unknown'
    assert 'continuing' in capsys.readouterr().err


@pytest.mark.parametrize('installed', [False, True])
def test_job_without_git_reaches_validation_and_writes_receipt(tmp_path, monkeypatch, installed):
    from gpuwm import earth2_job
    from gpuwm.earth2_door import build_spec
    monkeypatch.setattr(build_provenance, 'owning_distribution', lambda: owner(
        schema=build_provenance.BUILD_SCHEMA, version='2.9.3', commit='b' * 40) if installed else None)
    def no_tools(*args, **kwargs):
        raise AssertionError('receipt/validation must not launch git or another tool')
    monkeypatch.setattr(earth2_job.subprocess, 'run', no_tools)
    # No area is deliberately invalid: this reaches the real preflight without
    # downloading data or opening a GPU, after receipt identity has been read.
    with pytest.raises(RuntimeError, match='regional forecast needs --area'):
        earth2_job.run_job(build_spec(model='fcn3', cycle='2024-09-24T12'), tmp_path)
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['code_revision'] == ('b' * 40 if installed else 'unknown')
    assert receipt['package_provenance']['commit'] == receipt['code_revision']
