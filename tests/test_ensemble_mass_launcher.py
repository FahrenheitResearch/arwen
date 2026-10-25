"""Checkpoint rosters preserve the public source label independently of placement."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gpuwm.ensemble.mass_launcher import branch_members, load_parents, select_cards
from gpuwm.ensemble.wofs_recipe import provenance_label, surface_draw
from gpuwm.supervisor import GPUIdentity


def parents():
    return [dict(parent_id=index, checkpoint=f'/parent-{index}.npz', provenance={
        'source': 'gefs', 'cycle': '2022-06-13T18', 'source_member': member,
        'physics_arm': 'thompson-ysu', 'lag_hours': 0}) for index, member in enumerate(('c00', 'p01'))]


def test_roster_balances_parents_and_uses_the_one_shared_label():
    source = parents()
    before = deepcopy(source)
    roster = branch_members(source, 1000, seed=73, surface=True)
    assert source == before
    assert [sum(m['parent_index'] == index for m in roster) for index in (0, 1)] == [500, 500]
    assert len({m['seed'] for m in roster}) == 1000
    assert roster == branch_members(source, 1000, seed=73, surface=True)
    for row in roster:
        label = row['provenance']
        assert label['label'] == provenance_label(label)
        assert json.loads(label['label'])['parent'] == source[row['parent_index']]['provenance']
        assert label['surface'] == surface_draw(row['seed'])
        assert label['branch'] == {'kind': 'saved-checkpoint', 'checkpoint': row['checkpoint']}
        assert 'after parent checkpoint restore' in label['perturbation']['applied']


def test_saved_checkpoint_label_replaces_any_init_time_zero_branch(monkeypatch):
    from gpuwm.ensemble.wofs_recipe import LadderMember
    original = LadderMember.provenance
    def time_zero(self):
        row = original(self)
        row['branch'] = {'kind': 'initialized-checkpoint', 'model_elapsed_seconds': 0}
        return row
    monkeypatch.setattr(LadderMember, 'provenance', time_zero)
    row = branch_members(parents(), 1)[0]
    assert row['provenance']['branch'] == {'kind': 'saved-checkpoint', 'checkpoint': row['checkpoint']}
    assert row['provenance']['label'] == provenance_label(row['provenance'])


def test_off_switch_does_not_fabricate_distinct_forecasts():
    assert len(branch_members(parents(), 2, amplitude=0)) == 2
    with pytest.raises(ValueError, match='duplicate trajectories'):
        branch_members(parents(), 3, amplitude=0)
    assert len(branch_members(parents(), 3, amplitude=0, surface=True)) == 3


@pytest.mark.parametrize('amplitude', [-1, float('nan'), float('inf')])
def test_bad_amplitudes_refuse_before_cuda(amplitude):
    with pytest.raises(ValueError, match='nonnegative finite'):
        branch_members(parents(), 2, amplitude=amplitude)


def test_card_ordinals_are_within_inherited_visibility():
    identities = tuple(GPUIdentity(f'GPU-{i}abc', 'driver', f'card-{i}', i) for i in range(4))
    assert select_cards('all', identities, visibility='3,1') == (identities[3], identities[1])
    assert select_cards('1', identities, visibility='3,1') == (identities[1],)
    assert select_cards('GPU-3abc', identities, visibility='3,1') == (identities[3],)
    for selection in ('0,0', '2', '', 'GPU-0abc'):
        with pytest.raises(ValueError):
            select_cards(selection, identities, visibility='3,1')


def test_parent_paths_resolve_beside_roster_and_source_is_required(tmp_path):
    (tmp_path/'checkpoint.npz').touch()
    (tmp_path/'run').mkdir()
    path = tmp_path/'parents.json'
    row = dict(checkpoint='checkpoint.npz', wrfinput='run', provenance=parents()[0]['provenance'])
    path.write_text(json.dumps({'parents': [row]}))
    result = load_parents(path)
    assert result[0]['checkpoint'] == str(tmp_path/'checkpoint.npz')
    assert result[0]['parent_id'] == 0
    del row['provenance']
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match='source identity'):
        load_parents(path)


def test_checkpoint_surface_branch_reaches_surface_validation_without_changing_restart_default():
    from gpuwm.ensemble.surface_recipe import apply_surface_recipe
    state = SimpleNamespace(elapsed_seconds=7200)
    options = dict(value={'kind': 'surface-state', 'sst_offset_k': 0.1}, member_id=1, seed=9)
    assert not apply_surface_recipe(state, **options)['applied']
    with pytest.raises(ValueError, match='initialized physics surface fields'):
        apply_surface_recipe(state, **options, at_checkpoint_branch=True)


def test_public_command_needs_no_config_and_uses_shared_wind_controls(tmp_path, capsys):
    import argparse
    from gpuwm.go_cli import register_cli, go_main
    (tmp_path/'checkpoint.npz').touch()
    (tmp_path/'run').mkdir()
    path = tmp_path/'parents.json'
    path.write_text(json.dumps([dict(checkpoint='checkpoint.npz', wrfinput='run',
                                    provenance=parents()[0]['provenance'])]))
    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers(dest='command'))
    args = parser.parse_args(['ensemble', '--members', '3', '--branch-from', str(path),
        '--cards', 'all', '--wind-amplitude', '0.25', '--ensemble-seed', '73', '--dry-run'])
    assert args.config is None
    assert go_main(args) == 0
    roster = json.loads(capsys.readouterr().out)['members']
    assert len(roster) == 3
    assert all(m['provenance']['perturbation']['amplitude_m_s'] == 0.25 for m in roster)


def test_100_children_preserve_all_36_equal_ancestor_masses():
    source = [dict(parents()[0], parent_id=i, checkpoint=f'/parent-{i}.npz', weight=1/36)
              for i in range(36)]
    children = branch_members(source, 100)
    counts = [sum(c['parent_id'] == i for c in children) for i in range(36)]
    assert counts.count(3) == 28 and counts.count(2) == 8
    for i in range(36):
        assert sum(c['weight'] for c in children if c['parent_id'] == i) == pytest.approx(1/36)
        assert all(c['checkpoint'] == source[i]['checkpoint'] for c in children if c['parent_id'] == i)
    proof = branch_members(source, 8)
    assert all(c['provenance']['ancestry']['represented_parent_mass'] == pytest.approx(8/36) for c in proof)
    assert sum(c['weight'] for c in proof) == pytest.approx(1)


def test_unequal_ancestor_weights_and_invalid_weights():
    source = parents()
    source[0]['weight'], source[1]['weight'] = .9, .1
    children = branch_members(source, 11)
    assert sum(c['weight'] for c in children if c['parent_id'] == 0) == pytest.approx(.9)
    for bad in (0, -1, float('nan'), float('inf')):
        source[0]['weight'] = bad
        with pytest.raises(ValueError, match='positive finite'):
            branch_members(source, 11)


def test_resume_recovers_worker_commit_after_coordinator_crash(tmp_path):
    from gpuwm.ensemble.mass_launcher import atomic_json, resume_completed
    source = parents()
    members = branch_members(source, 2)
    receipt = dict(parents=source, members=members, branch_hours=3, completed=[])
    frame = tmp_path/'frame.json'
    frame.touch()
    atomic_json(tmp_path/'members/member-00000/receipt.json',
                {**members[0], 'status': 'completed', 'frames': [{'path': str(frame)}]})
    atomic_json(tmp_path/'members/member-00001/receipt.json',
                {**members[1], 'status': 'failed'})
    assert [r['member_id'] for r in resume_completed(tmp_path, receipt, source, members, 3)] == [0]
    frame.unlink()
    assert resume_completed(tmp_path, receipt, source, members, 3) == []
    with pytest.raises(ValueError, match='same parent'):
        resume_completed(tmp_path, receipt, source, members, 18)


def test_members_per_card_auto_sizes_by_memory_and_caps():
    from gpuwm.ensemble.mass_launcher import (AUTO_MEMBERS_PER_CARD_CAP, members_per_card,
                                              parse_members_per_card)
    gib = 1 << 30
    # A 6.4 GiB member on a free 95 GiB card: memory allows 14, the cap binds.
    slots, receipt = members_per_card('auto', member_bytes=int(6.4*gib), free_bytes=95*gib,
                                      members=20, cards=1, cpus=16)
    assert (slots, receipt['binding']) == (AUTO_MEMBERS_PER_CARD_CAP, 'cap')
    # A 30 GiB member on a 32 GiB card: one.
    assert members_per_card('auto', member_bytes=30*gib, free_bytes=32*gib, members=20, cards=1, cpus=16)[0] == 1
    # Memory binds below the cap.
    slots, receipt = members_per_card('auto', member_bytes=10*gib, free_bytes=32*gib, members=20, cards=1, cpus=16)
    assert (slots, receipt['binding']) == (3, 'card memory')
    # Never more processes than the card's members, or than half the host cores.
    assert members_per_card('auto', member_bytes=gib, free_bytes=95*gib, members=3, cards=1, cpus=16)[0] == 3
    assert members_per_card('auto', member_bytes=gib, free_bytes=95*gib, members=20, cards=2, cpus=4)[0] == 1
    # Unpriced member or unreadable card: one, and said so.
    slots, receipt = members_per_card('auto', member_bytes=None, free_bytes=95*gib, members=20, cards=1, cpus=16)
    assert slots == 1 and 'unpriced' in receipt['binding']
    assert members_per_card(6, member_bytes=None, free_bytes=None, members=20, cards=1)[0] == 6
    assert parse_members_per_card('auto') == 'auto' and parse_members_per_card('3') == 3
    for bad in ('0', '-1', 'two'):
        with pytest.raises(ValueError, match='members-per-card'):
            parse_members_per_card(bad)


def test_member_sharing_contexts_and_missing_controller(tmp_path, monkeypatch):
    import shutil
    from gpuwm.ensemble.mass_launcher import member_sharing
    with member_sharing('contexts', tmp_path, ()) as sharing:
        assert sharing['mode'] == 'contexts'
    monkeypatch.delenv('CUDA_MPS_PIPE_DIRECTORY', raising=False)
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    with member_sharing('auto', tmp_path, ()) as sharing:
        assert sharing['mode'] == 'contexts' and 'no MPS controller' in sharing['reason']
    with pytest.raises(RuntimeError, match='nvidia-cuda-mps-control'):
        with member_sharing('mps', tmp_path, ()):
            pass
