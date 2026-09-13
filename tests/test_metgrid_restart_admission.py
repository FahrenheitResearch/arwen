"""A preparation's resource observation cannot change its science identity."""
from datetime import timedelta
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest


def _identity(tmp_path, exp, *, free_bytes, source_digest='input', cache_digest='cache',
              tamper_after_binding=False):
    from gpuwm import metem_forecast, prepared_domain_tree_forecast as tree
    directory = tmp_path/f'preparation-{free_bytes}-{source_digest}-{cache_digest}'
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory/'metgrid-import.json'
    metem_forecast._json(receipt, {'schema': 'gpuwm-metgrid-import-v1',
        'source_files': {'original': source_digest}, 'domains': {'d01': {'soil_contract': 'same'}},
        'vertical_coordinate': 'same', 'memory_admission': {'available_device_bytes': free_bytes}})
    config = tmp_path/'case.toml'
    config.write_text('[experiment]\n')
    inputs = SimpleNamespace(experiment=exp, boundaries=None, experiment_config=config,
        artifact_paths={'preparation_receipt': receipt},
        authority_sha256={'preparation_receipt': hashlib.sha256(receipt.read_bytes()).hexdigest(),
                          'experiment_config': hashlib.sha256(config.read_bytes()).hexdigest()},
        domains=tuple(SimpleNamespace(grid_id=d.grid_id,
            cache_reader=SimpleNamespace(content_sha256=cache_digest),
            authority_sha256={'cache_content': cache_digest}) for d in exp.domains),
        execution_plan=tree.resolve_execution_plan(exp))
    if tamper_after_binding:
        receipt.write_bytes(receipt.read_bytes() + b' ')
    initializer = metem_forecast.MetemInitialization(inputs)
    parts = tree.tree_restart_identity_components(inputs, {'runtime': 'fixed'}, initializer)
    return hashlib.sha256(tree._canonical(parts).encode()).hexdigest(), parts


@pytest.mark.parametrize('mutation', ['memory', 'source', 'cache'])
def test_actual_tree_checkpoint_distinguishes_science_from_free_memory(tmp_path, monkeypatch, mutation):
    from test_checkpoint_route_contract import _wizard_config, _raw
    from test_restart import _sealed_tree_fixture
    from gpuwm.experiment import build_experiment
    from gpuwm.io import restart
    exp = build_experiment(_raw(_wizard_config(tmp_path/'config', ladder='12-3')),
                           source='checkpoint fixture')
    initial_id, initial_parts = _identity(tmp_path/'first', exp, free_bytes=16*2**30)
    changed = dict(free_bytes=16*2**30)
    if mutation == 'memory': changed['free_bytes'] += 2**20
    if mutation == 'source': changed['source_digest'] = 'changed input'
    if mutation == 'cache': changed['cache_digest'] = 'changed cache'
    next_id, next_parts = _identity(tmp_path/'next', exp, **changed)
    source, start = _sealed_tree_fixture(monkeypatch, forcing_count=2,
        run_seconds=7200., payload_seed=31)
    source.experiment_fingerprint = initial_id
    source._experiment_fingerprint_components = initial_parts
    checkpoint = restart.write_tree_restart(tmp_path/'checkpoint', source,
        start+timedelta(seconds=3600))
    resumed, _ = _sealed_tree_fixture(monkeypatch, forcing_count=2,
        run_seconds=7200., payload_seed=91)
    resumed.experiment_fingerprint = next_id
    resumed._experiment_fingerprint_components = next_parts
    if mutation == 'memory':
        assert not np.array_equal(resumed.root.state.u, source.root.state.u)
        info = restart.restore_tree_restart(checkpoint, resumed)
        assert info.elapsed_ticks == 3600
        assert initial_id == next_id
        np.testing.assert_array_equal(resumed.root.state.u, source.root.state.u)
    else:
        component = ('preparation_receipt_sha256' if mutation == 'source'
                     else 'domain_cache_content_sha256')
        with pytest.raises(restart.RestartMismatchError, match=component):
            restart.restore_tree_restart(checkpoint, resumed)


def test_raw_receipt_integrity_is_checked_before_excluding_observations(tmp_path):
    from test_checkpoint_route_contract import _wizard_config, _raw
    from gpuwm.experiment import build_experiment
    exp = build_experiment(_raw(_wizard_config(tmp_path/'config', ladder='12-3')),
                           source='receipt fixture')
    with pytest.raises(RuntimeError, match='receipt changed'):
        _identity(tmp_path/'receipt', exp, free_bytes=16*2**30, tamper_after_binding=True)
