"""Current decision, receipt and reader agreement on tiny real publications."""
import json
from pathlib import Path

import pytest

from gpuwm.ensemble import analysis_commit, cycle
from gpuwm.ensemble.manifest import member_directory_name
from tests.test_analysis_publication_recovery import _case, _partial


def rewrite_intent(path, document):
    document['self_sha256'] = analysis_commit.digest({
        key: value for key, value in document.items() if key != 'self_sha256'})
    path.write_text(json.dumps(document))


def test_success_after_member_event_failure_reuses_the_decision(tmp_path, monkeypatch):
    cfg, root, args, calls = _case(tmp_path)
    def fail_event(event):
        if event.get('event') == 'member-assimilated':
            raise OSError('event delivery interrupted after actual rename')
    with pytest.raises(OSError, match='event delivery interrupted'):
        cycle.run_cycles(cfg, root, on_event=fail_event, **args)
    leg = cycle.cycle_root(root, 0)
    first = leg / member_directory_name(0) / cycle.ANALYSIS_NAME
    initial = first.read_bytes()
    intended = analysis_commit.read_record(cycle.publication_marker_path(leg))['receipt']
    result = cycle.run_cycles(cfg, root, **args)
    assert calls == [0] and first.read_bytes() == initial
    assert json.loads(result.manifest_path.read_text())['cycles'][0]['assimilation'] == intended


def test_pending_run_rejects_changed_caller_context_before_any_remaining_rename(tmp_path, monkeypatch):
    cfg, root, args, calls = _partial(tmp_path, monkeypatch)
    args['analysis_context'] = lambda index, states, recovering: {'method_settings': {'inflation': 2}}
    leg = cycle.cycle_root(root, 0)
    initial = {str(p): p.read_bytes() for p in leg.rglob('analysis.npz*')}
    with pytest.raises(ValueError, match='different inputs or policies'):
        cycle.run_cycles(cfg, root, **args)
    assert calls == [0]
    assert initial == {str(p): p.read_bytes() for p in leg.rglob('analysis.npz*')}


def test_current_run_rejects_receipt_that_contradicts_its_bound_method(tmp_path, monkeypatch):
    cfg, root, args, calls = _partial(tmp_path, monkeypatch)
    leg = cycle.cycle_root(root, 0)
    marker = cycle.publication_marker_path(leg)
    decision = json.loads(marker.read_text())
    decision['receipt']['method']['callable'] = 'different.analysis.method'
    rewrite_intent(marker, decision)
    with pytest.raises(ValueError):
        cycle.run_cycles(cfg, root, **args)
    assert calls == [0]
    assert not (leg / member_directory_name(1) / cycle.ANALYSIS_NAME).exists()


def test_reader_rejects_policy_inconsistent_with_the_current_run_binding(tmp_path, monkeypatch):
    _, root, _, calls = _partial(tmp_path, monkeypatch)
    leg = cycle.cycle_root(root, 0)
    marker = cycle.publication_marker_path(leg)
    decision = json.loads(marker.read_text())
    decision['context']['moment_repair'] = False
    rewrite_intent(marker, decision)
    with pytest.raises(ValueError):
        cycle.read_analysis_roster(leg, n_members=3)
    assert calls == [0]
    assert not (leg / member_directory_name(1) / cycle.ANALYSIS_NAME).exists()


def test_reader_cannot_redefine_a_three_member_run_as_two_surviving_members(tmp_path, monkeypatch):
    _, root, _, calls = _partial(tmp_path, monkeypatch)
    leg = cycle.cycle_root(root, 0)
    marker = cycle.publication_marker_path(leg)
    decision = json.loads(marker.read_text())
    last = member_directory_name(2)
    (leg / last).rename(tmp_path / 'retained-last-member')
    decision['members'].pop()
    decision['member_count'] = 2
    decision['context']['members'].pop()
    decision['context']['backgrounds'].pop()
    decision['context']['assets'] = [row for row in decision['context']['assets']
        if last not in Path(row['path']).parts]
    decision['receipt']['member_count'] = 2
    decision['receipt']['receipts'].pop()
    rewrite_intent(marker, decision)
    # The retained run binding still explicitly requires three members.
    assert decision['context']['run_binding']['n_members'] == 3
    with pytest.raises(ValueError):
        cycle.read_analysis_roster(leg, n_members=2)
    assert calls == [0]
    assert not (leg / member_directory_name(1) / cycle.ANALYSIS_NAME).exists()


@pytest.mark.parametrize('key,value', [
    ('moment_repair', False), ('moment_policy', 'mass-only'),
    ('positivity', 'none'), ('mp_physics', 8),
    ('method', {'callable': 'different.method', 'implementation_sha256': None}),
    ('declared_method', {'method': 'different-method'}),
])
def test_reader_checks_a_matching_receipt_and_context_against_the_run(
        tmp_path, monkeypatch, key, value):
    _, root, _, calls = _partial(tmp_path, monkeypatch)
    leg = cycle.cycle_root(root, 0)
    marker = cycle.publication_marker_path(leg)
    decision = json.loads(marker.read_text())
    decision['context'][key] = value
    if key in ('moment_repair', 'moment_policy'):
        decision['receipt'][key] = value
    elif key == 'positivity':
        decision['receipt']['positivity_policy'] = value
    elif key == 'method':
        decision['receipt']['method']['callable'] = value['callable']
    elif key == 'declared_method':
        decision['receipt']['method'].update(declared_by='caller', provenance=value)
    rewrite_intent(marker, decision)
    before = {str(p): p.read_bytes() for p in leg.rglob('analysis.npz*')}
    with pytest.raises(ValueError, match='run .*contradicts'):
        cycle.read_analysis_roster(leg, n_members=3)
    assert calls == [0]
    assert before == {str(p): p.read_bytes() for p in leg.rglob('analysis.npz*')}


@pytest.mark.parametrize('key,value', [
    ('positivity_policy', 'none'), ('moment_policy', 'mass-only'),
    ('moment_repair', False), ('member-receipts', None),
    ('undeclared-provenance', None), ('declared-provenance', {'method': 'different'}),
])
def test_receipt_cannot_claim_other_policies_or_a_partial_member_set(
        tmp_path, monkeypatch, key, value):
    cfg, root, args, calls = _partial(tmp_path, monkeypatch)
    leg = cycle.cycle_root(root, 0)
    marker = cycle.publication_marker_path(leg)
    decision = json.loads(marker.read_text())
    if key == 'member-receipts':
        decision['receipt']['receipts'].pop()
    elif key == 'undeclared-provenance':
        decision['receipt']['method']['declared_by'] = None
    elif key == 'declared-provenance':
        decision['receipt']['method'].update(declared_by='caller', provenance=value)
    else:
        decision['receipt'][key] = value
    rewrite_intent(marker, decision)
    before = {str(p): p.read_bytes() for p in leg.rglob('analysis.npz*')}
    with pytest.raises(ValueError, match='receipt .*contradicts'):
        cycle.run_cycles(cfg, root, **args)
    assert calls == [0]
    assert before == {str(p): p.read_bytes() for p in leg.rglob('analysis.npz*')}


def test_declared_method_receipt_survives_normal_recovery(tmp_path):
    cfg, root, args, calls = _case(tmp_path)
    original = args['assimilate']
    def method(index, states):
        return original(index, states)[0]
    args.update(assimilate=method, assimilation_method={'method': 'fixed-increment', 'version': 1})
    cycle.run_cycles(cfg, root, **args)
    result = cycle.run_cycles(cfg, root, **args)
    receipt = json.loads(result.manifest_path.read_text())['cycles'][0]['assimilation']
    assert receipt['method']['declared_by'] == 'caller'
    assert receipt['method']['provenance'] == args['assimilation_method']
    assert calls == [0]


def test_custom_callback_without_inspectable_identity_keeps_explicit_context(tmp_path):
    cfg, root, args, calls = _case(tmp_path)
    original = args['assimilate']
    class Callback:
        def __call__(self, index, states):
            return original(index, states)
    args.update(assimilate=Callback(), analysis_context=lambda *args, **kwargs: {
        'custom_contract': 'fixed-increment-v1', 'assets': []})
    cycle.run_cycles(cfg, root, **args)
    cycle.run_cycles(cfg, root, **args)
    assert calls == [0]
