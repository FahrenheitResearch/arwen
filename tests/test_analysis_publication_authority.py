import json
from pathlib import Path
import pytest
from gpuwm.ensemble import analysis_commit, cycle
from gpuwm.ensemble.manifest import member_directory_name, CYCLE_MANIFEST_NAME
from tests.test_analysis_publication_recovery import _partial


@pytest.mark.parametrize('omission', ['null', 'empty', 'absent'])
def test_reader_cannot_drop_an_existing_run_binding_to_publish_a_reduced_roster(tmp_path, monkeypatch, omission):
    _, root, _, calls = _partial(tmp_path, monkeypatch)
    leg = cycle.cycle_root(root, 0)
    marker = cycle.publication_marker_path(leg)
    intent = json.loads(marker.read_text())
    run_before = (root / CYCLE_MANIFEST_NAME).read_bytes()
    assert json.loads(run_before)['cycle_binding']['n_members'] == 3
    last = member_directory_name(2)
    (leg/last).rename(tmp_path/'preserved-member-2')
    intent['members'].pop()
    intent['member_count'] = 2
    intent['context']['members'].pop()
    intent['context']['backgrounds'].pop()
    intent['context']['assets'] = [row for row in intent['context']['assets'] if last not in Path(row['path']).parts]
    intent['receipt']['member_count'] = 2
    intent['receipt']['receipts'].pop()
    if omission == 'absent':
        intent['context'].pop('run_binding')
    else:
        intent['context']['run_binding'] = None if omission == 'null' else {}
    intent['self_sha256'] = analysis_commit.digest({k:v for k,v in intent.items() if k != 'self_sha256'})
    marker.write_text(json.dumps(intent))
    before = {str(p):p.read_bytes() for p in leg.rglob('analysis.npz*')}
    with pytest.raises(ValueError):
        cycle.read_analysis_roster(leg, n_members=2)
    assert before == {str(p):p.read_bytes() for p in leg.rglob('analysis.npz*')}
    assert (root/CYCLE_MANIFEST_NAME).read_bytes() == run_before
    assert calls == [0]


def test_standalone_partial_publication_keeps_its_supported_recovery(tmp_path, monkeypatch):
    import numpy as np
    from gpuwm.ensemble.state_sha import checkpoint_state_sha256
    from tests.test_ensemble_engine import _contract_names
    field = _contract_names()[0]
    leg = tmp_path / 'standalone'
    states = {}
    for index in range(2):
        member = leg / member_directory_name(index)
        member.mkdir(parents=True)
        checkpoint = member / 'gpuwmrst_000060.npz'
        np.savez(checkpoint, **{'state/' + field: np.zeros((2, 3), np.float32)})
        states[index] = dict(member_dir=str(member), seed=index,
            state_sha256=checkpoint_state_sha256(checkpoint))
    calls = []
    def method(index, members):
        calls.append(index)
        return {member: {field: np.full((2, 3), .25, np.float32)} for member in members}
    publish = cycle.publish_staged_analysis
    published = []
    def interrupt(staged, analysis):
        if published:
            raise OSError('standalone publication interrupted')
        publish(staged, analysis)
        published.append(analysis)
    monkeypatch.setattr(cycle, 'publish_staged_analysis', interrupt)
    with pytest.raises(OSError, match='standalone publication interrupted'):
        cycle._assimilate_cycle(method, 0, states, leg_root=leg)
    original = published[0].read_bytes()
    assert not (tmp_path / CYCLE_MANIFEST_NAME).exists()
    monkeypatch.setattr(cycle, 'publish_staged_analysis', publish)
    roster = cycle.read_analysis_roster(leg, n_members=2)
    assert set(roster) == {0, 1} and calls == [0]
    assert published[0].read_bytes() == original
