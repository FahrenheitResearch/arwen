"""Durable windows use the real ensemble, checkpoint and analysis owners."""
import hashlib
import json

import numpy as np
import pytest

from gpuwm.ensemble import cycle
from gpuwm.ensemble.config import load_ensemble_config
from test_ensemble_engine import _contract_names, _cycling_runner, _write_overlay


def clocked_runner(**kwargs):
    """Only the forecast is substituted; preserve its supplied checkpoint."""
    from dataclasses import replace
    from gpuwm.ensemble.state_sha import checkpoint_state_sha256
    outcome = _cycling_runner(**kwargs)
    path = kwargs['member_dir'] / 'gpuwmrst_d01_000.npz'
    source = kwargs.get('restart') or path
    with np.load(source, allow_pickle=False) as file:
        payload = {name: np.array(file[name], copy=True) for name in file.files}
    payload['meta/elapsed_seconds'] = np.asarray(kwargs['run_seconds'])
    with path.open('wb') as file:
        np.savez(file, **payload)
    return replace(outcome, final_state_sha256=checkpoint_state_sha256(path))


def test_separate_windows_keep_cumulative_clocks_and_committed_decisions(tmp_path, monkeypatch):
    cfg = load_ensemble_config(_write_overlay(tmp_path, n_members=3, perturbation='none'))
    field = _contract_names()[0]
    calls = []
    def method(index, states):
        calls.append(index)
        return {i: {field: np.full((2, 3), .25, np.float32)} for i in states}, {'window': index}

    prior = None
    prior_bytes = {}
    real_publish = cycle.write_manifest_atomically
    interrupted = False
    def publish(path, document):
        nonlocal interrupted
        if not interrupted and any(row['cycle'] == 1 and row['status'] == 'DONE'
                                   for row in document.get('cycles', ())):
            interrupted = True
            raise OSError('lost window completion')
        return real_publish(path, document)
    monkeypatch.setattr(cycle, 'write_manifest_atomically', publish)
    for index in range(4):
        root = tmp_path / f'window-{index}'
        args = dict(n_cycles=1, first_cycle=index, initial_restarts=prior,
                    input_binding={'boundary_generation': index // 2},
                    cycle_seconds=60., runner=clocked_runner, assimilate=method)
        if index == 1:
            with pytest.raises(OSError, match='lost window completion'):
                cycle.run_cycles(cfg, root, **args)
        result = cycle.run_cycles(cfg, root, **args)
        assert result.status == 'COMPLETE'
        doc = json.loads(result.manifest_path.read_text())
        assert [row['cycle'] for row in doc['cycles']] == [index]
        assert doc['cycles'][0]['run_seconds_total'] == (index + 1) * 60
        for path, expected in prior_bytes.items():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        prior = cycle.read_analysis_roster(cycle.cycle_root(root, index), n_members=3)
        for path in prior.values():
            assert cycle._restart_elapsed_seconds(path) == (index + 1) * 60
            prior_bytes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert calls == [0, 1, 2, 3]


def test_later_window_cannot_cold_start(tmp_path):
    cfg = load_ensemble_config(_write_overlay(tmp_path, n_members=1, perturbation='none'))
    with pytest.raises(ValueError, match='complete prior analysis roster'):
        cycle.run_cycles(cfg, tmp_path / 'later', n_cycles=1, first_cycle=1,
                         cycle_seconds=60., runner=_cycling_runner)
    assert not (tmp_path / 'later').exists()


def test_changed_renewal_receipt_cannot_replace_committed_window(tmp_path):
    cfg = load_ensemble_config(_write_overlay(tmp_path, n_members=1, perturbation='none'))
    root = tmp_path / 'window'
    args = dict(n_cycles=1, cycle_seconds=60., runner=_cycling_runner,
                input_binding={'boundary_generation': 'original'})
    cycle.run_cycles(cfg, root, **args)
    args['input_binding'] = {'boundary_generation': 'different'}
    with pytest.raises(ValueError, match='different|changed|incompatible'):
        cycle.run_cycles(cfg, root, **args)
