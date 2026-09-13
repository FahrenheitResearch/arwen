"""Interrupted publication preserves the original analysis decision."""
import hashlib
import json

import numpy as np
import pytest

from gpuwm.ensemble import cycle
from gpuwm.ensemble.config import load_ensemble_config
from gpuwm.ensemble.manifest import member_directory_name
from tests.test_ensemble_engine import _contract_names, _cycling_runner, _write_overlay


@pytest.mark.parametrize("interruption,n_members", [("outer_done", 1), ("outer_done", 3), ("member_rename", 3), ("none", 3)])
def test_published_analysis_is_not_recomputed(tmp_path, monkeypatch, record_property, interruption, n_members):
    cfg = load_ensemble_config(_write_overlay(tmp_path, n_members=n_members, perturbation="none"))
    root = tmp_path / "cycle-run"
    calls = []
    field = _contract_names()[0]

    def assimilate(index, states):
        calls.append(index)
        increment = 0.25 * len(calls)
        return ({member: {field: np.full((2, 3), increment, np.float32)} for member in states}, {"method": "counted-increment", "call": len(calls)})

    original_manifest = cycle.write_manifest_atomically
    original_publish = cycle.publish_staged_analysis
    fault = {"raised": False, "renames": 0}

    def publish_manifest(path, document):
        if interruption == "outer_done" and not fault["raised"] and any(entry.get("status") == "DONE" for entry in document.get("cycles", ())):
            fault["raised"] = True
            raise RuntimeError("interrupted before outer completion publication")
        return original_manifest(path, document)

    def publish_member(staged, analysis):
        fault["renames"] += 1
        if interruption == "member_rename" and fault["renames"] == 2:
            fault["raised"] = True
            raise RuntimeError("interrupted during member publication")
        return original_publish(staged, analysis)

    monkeypatch.setattr(cycle, "write_manifest_atomically", publish_manifest)
    monkeypatch.setattr(cycle, "publish_staged_analysis", publish_member)
    arguments = dict(n_cycles=1, cycle_seconds=60.0, assimilate=assimilate, runner=_cycling_runner)
    if interruption == "none":
        cycle.run_cycles(cfg, root, **arguments)
    else:
        with pytest.raises(RuntimeError, match="interrupted"):
            cycle.run_cycles(cfg, root, **arguments)
        assert fault["raised"]
    first_member = cycle.cycle_root(root, 0) / member_directory_name(0) / cycle.ANALYSIS_NAME
    before = hashlib.sha256(first_member.read_bytes()).hexdigest()
    with np.load(first_member, allow_pickle=False) as data:
        before_value = float(data["state/" + field].flat[0])

    monkeypatch.setattr(cycle, "write_manifest_atomically", original_manifest)
    monkeypatch.setattr(cycle, "publish_staged_analysis", original_publish)
    result = cycle.run_cycles(cfg, root, **arguments)
    after = hashlib.sha256(first_member.read_bytes()).hexdigest()
    with np.load(first_member, allow_pickle=False) as data:
        after_value = float(data["state/" + field].flat[0])
    document = json.loads(result.manifest_path.read_text())
    facts = {"source": cycle.__file__, "interruption": interruption, "members": n_members, "method_calls": len(calls), "before_value": before_value, "after_value": after_value, "published_bytes_preserved": before == after, "completion_status": document["status"], "method_receipt": document["cycles"][0]["assimilation"]["method"]["provenance"]}
    record_property("publication_observation", json.dumps(facts, sort_keys=True))
    assert len(calls) == 1, facts
    assert before == after, facts
    assert document["cycles"][0]["assimilation"]["method"]["provenance"]["call"] == 1


def _case(tmp_path, members=3):
    cfg = load_ensemble_config(_write_overlay(tmp_path, n_members=members, perturbation="none"))
    root = tmp_path / 'cycles'
    calls = []
    field = _contract_names()[0]
    def method(index, states):
        calls.append(index)
        return {i: {field: np.full((2, 3), .25, np.float32)} for i in states}, {'method': 'fixed-increment'}
    return cfg, root, dict(n_cycles=1, cycle_seconds=60., assimilate=method, runner=_cycling_runner), calls


@pytest.mark.parametrize('change', ['analysis', 'background', 'policy', 'method', 'declared'])
def test_completed_decision_rejects_changed_context_without_recomputing(tmp_path, change):
    cfg, root, args, calls = _case(tmp_path)
    cycle.run_cycles(cfg, root, **args)
    member = cycle.cycle_root(root, 0) / member_directory_name(0)
    analysis = member / cycle.ANALYSIS_NAME
    before = analysis.read_bytes()
    if change in ('analysis', 'background'):
        target = analysis if change == 'analysis' else next(member.glob('gpuwmrst_*.npz'))
        with np.load(target, allow_pickle=False) as file:
            payload = {name: file[name] for name in file.files}
        payload['state/' + _contract_names()[0]] += np.float32(.5)
        with target.open('wb') as file:
            np.savez(file, **payload)
    elif change == 'policy':
        args['moment_repair'] = False
    elif change == 'method':
        def different(index, states):
            raise AssertionError('a different method must not run over a committed decision')
        args['assimilate'] = different
    else:
        args['assimilation_method'] = {'observations': 'different-set'}
    with pytest.raises(ValueError):
        cycle.run_cycles(cfg, root, **args)
    assert calls == [0]
    if change != 'analysis':
        assert analysis.read_bytes() == before


def _partial(tmp_path, monkeypatch):
    cfg, root, args, calls = _case(tmp_path)
    original = cycle.publish_staged_analysis
    count = 0
    def interrupt(staged, analysis):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError('controlled member interruption')
        return original(staged, analysis)
    monkeypatch.setattr(cycle, 'publish_staged_analysis', interrupt)
    with pytest.raises(OSError, match='controlled member interruption'):
        cycle.run_cycles(cfg, root, **args)
    monkeypatch.setattr(cycle, 'publish_staged_analysis', original)
    return cfg, root, args, calls


def test_corrupt_last_member_prevents_all_remaining_renames(tmp_path, monkeypatch):
    cfg, root, args, calls = _partial(tmp_path, monkeypatch)
    leg = cycle.cycle_root(root, 0)
    first = leg / member_directory_name(0) / cycle.ANALYSIS_NAME
    original = first.read_bytes()
    last = leg / member_directory_name(2) / (cycle.ANALYSIS_NAME + cycle.STAGED_SUFFIX)
    data = bytearray(last.read_bytes()); data[len(data) // 2] ^= 1; last.write_bytes(data)
    with pytest.raises(ValueError, match='digest'):
        cycle.run_cycles(cfg, root, **args)
    assert not (leg / member_directory_name(1) / cycle.ANALYSIS_NAME).exists()
    assert first.read_bytes() == original and calls == [0]


@pytest.mark.parametrize('after_write', [False, True])
def test_interrupted_commit_record_reuses_the_prospective_receipt(tmp_path, monkeypatch, after_write):
    from pathlib import Path
    from gpuwm.ensemble import analysis_commit
    cfg, root, args, calls = _case(tmp_path)
    write = analysis_commit.write_record
    interrupted = False
    def fault(path, document):
        nonlocal interrupted
        if Path(path).name == analysis_commit.COMMIT_NAME and not interrupted:
            interrupted = True
            if after_write:
                write(path, document)
            raise OSError('controlled commit interruption')
        return write(path, document)
    monkeypatch.setattr(analysis_commit, 'write_record', fault)
    with pytest.raises(OSError, match='controlled commit interruption'):
        cycle.run_cycles(cfg, root, **args)
    leg = cycle.cycle_root(root, 0)
    before = {i: (leg / member_directory_name(i) / cycle.ANALYSIS_NAME).read_bytes() for i in range(3)}
    intent = analysis_commit.read_record(cycle.publication_marker_path(leg))
    monkeypatch.setattr(analysis_commit, 'write_record', write)
    result = cycle.run_cycles(cfg, root, **args)
    receipt = json.loads(result.manifest_path.read_text())['cycles'][0]['assimilation']
    assert calls == [0] and receipt == intent['receipt']
    assert before == {i: (leg / member_directory_name(i) / cycle.ANALYSIS_NAME).read_bytes() for i in range(3)}


def test_self_sealed_stale_policy_is_not_recovery_authority(tmp_path, monkeypatch):
    from gpuwm.ensemble.analysis_commit import digest
    cfg, root, args, calls = _partial(tmp_path, monkeypatch)
    leg = cycle.cycle_root(root, 0)
    path = cycle.publication_marker_path(leg)
    document = json.loads(path.read_text())
    document['context']['moment_repair'] = False
    document['self_sha256'] = digest({k: v for k, v in document.items() if k != 'self_sha256'})
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='receipt moment_repair.*contradicts'):
        cycle.run_cycles(cfg, root, **args)
    assert calls == [0]
    assert not (leg / member_directory_name(1) / cycle.ANALYSIS_NAME).exists()


def test_reader_rejects_a_self_sealed_incomplete_publication_roster(tmp_path, monkeypatch):
    from gpuwm.ensemble.analysis_commit import digest
    cfg, root, args, calls = _partial(tmp_path, monkeypatch)
    leg = cycle.cycle_root(root, 0)
    path = cycle.publication_marker_path(leg)
    document = json.loads(path.read_text())
    document['members'].pop()
    document['member_count'] -= 1
    document['context']['members'].pop()
    document['receipt']['member_count'] -= 1
    document['self_sha256'] = digest({k: v for k, v in document.items() if k != 'self_sha256'})
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='member directories'):
        cycle.read_analysis_roster(leg, n_members=3)
    assert not (leg / member_directory_name(1) / cycle.ANALYSIS_NAME).exists()


def test_frozen_owner_inputs_are_rechecked_without_repeating_analysis(tmp_path):
    import os
    from gpuwm.output_identity import file_record
    cfg, root, args, calls = _case(tmp_path)
    asset = tmp_path / 'observations.bin'; asset.write_bytes(b'first')
    context_calls = []
    def context(index, states, *, recovering):
        context_calls.append(recovering)
        return {'selection': 'fixed-window', 'assets': [file_record(asset)]}
    args['analysis_context'] = context
    cycle.run_cycles(cfg, root, **args)
    cycle.run_cycles(cfg, root, **args)
    assert calls == [0] and context_calls == [False, True]
    stamp = asset.stat(); asset.write_bytes(b'other'); os.utime(asset, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match='different inputs or policies'):
        cycle.run_cycles(cfg, root, **args)
    assert calls == [0]


def test_completed_legacy_receipt_can_be_checked_without_replacement(tmp_path):
    from gpuwm.ensemble.analysis_commit import COMMIT_NAME
    cfg, root, args, calls = _case(tmp_path)
    cycle.run_cycles(cfg, root, **args)
    leg = cycle.cycle_root(root, 0)
    cycle.publication_marker_path(leg).unlink()
    (leg / COMMIT_NAME).unlink()
    paths = cycle.read_analysis_roster(leg, n_members=3)
    before = {i: path.read_bytes() for i, path in paths.items()}
    cycle.run_cycles(cfg, root, **args)
    assert calls == [0]
    assert before == {i: path.read_bytes() for i, path in paths.items()}
    assert not cycle.publication_marker_path(leg).exists()


def test_legacy_partial_marker_without_method_receipt_preserves_every_file(tmp_path):
    leg = tmp_path / 'cycle_000'; pairs = []
    for index in range(2):
        directory = leg / member_directory_name(index); directory.mkdir(parents=True)
        analysis = directory / cycle.ANALYSIS_NAME
        staged = directory / (cycle.ANALYSIS_NAME + cycle.STAGED_SUFFIX)
        (analysis if index == 0 else staged).write_bytes(b'original-analysis')
        pairs.append(dict(member=index, analysis=str(analysis), staged=str(staged)))
    path = cycle.publication_marker_path(leg)
    path.write_text(json.dumps(dict(schema=cycle.PUBLICATION_MARKER_SCHEMA, cycle=0, members=pairs, member_count=2)))
    before = {str(path): path.read_bytes() for path in leg.rglob('*') if path.is_file()}
    with pytest.raises(ValueError, match='legacy publication'):
        cycle.recover_analysis_publication(leg)
    assert before == {str(path): path.read_bytes() for path in leg.rglob('*') if path.is_file()}
