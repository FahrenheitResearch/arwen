"""Tree checkpoints stage canonical stores before mutating any sibling."""
from dataclasses import replace
import numpy as np
import pytest

from gpuwm.io import restart
from tilestream import restart_stream, physics_inventory
from test_restart import _cfg, _fill_setup, _rewrite_restart_archive
from test_held_pbl_restart import _shim_driver_state


def _member(tmp_path, monkeypatch, gid):
    cfg = replace(_cfg(moist=True, cu_physics=3, bldt=2.), grid_id=gid)
    state, _ = _shim_driver_state(cfg, monkeypatch)
    _fill_setup(state)
    state.thp.fill(gid * 7)
    state.elapsed_seconds = 20.
    path = restart.write_restart(tmp_path/f'd{gid}.npz', state, cfg)
    store = {k: np.zeros_like(v) for k, v in
             physics_inventory.carrier_manifest(state).items()}
    return cfg, state, path, store


def test_staged_siblings_do_not_mutate_until_complete_validation(tmp_path, monkeypatch):
    members = [_member(tmp_path, monkeypatch, gid) for gid in (1, 2)]
    staged = []
    scalars = []
    for cfg, state, path, store in members:
        clock = {'elapsed_seconds': 0.}
        staged.append(restart_stream.validate_streamed_restart(
            path, store, cfg, setup=restart_stream.capture_domain_setup(state),
            template_state=state, scalars=clock))
        scalars.append(clock)
        assert not store['state/thp'].any()
        assert clock['elapsed_seconds'] == 0.
    cfg, state, path, store = members[1]
    def remove(payload, header):
        payload.pop('state/thp')
    bad = _rewrite_restart_archive(path, tmp_path/'bad.npz', remove)
    with pytest.raises(restart_stream.RestartRefused, match='missing'):
        restart_stream.validate_streamed_restart(
            bad, store, cfg, setup=restart_stream.capture_domain_setup(state),
            template_state=state, scalars=scalars[1])
    assert all(not member[3]['state/thp'].any() for member in members)
    # Application uses the already validated payload, never a second file read.
    for _, _, path, _ in members:
        path.write_bytes(b'the file changed after staging')
    for validation, (cfg, state, _, store), clock in zip(staged, members, scalars):
        result = validation.apply()
        assert result.device_copies == 0
        assert clock['elapsed_seconds'] == 20.
        np.testing.assert_array_equal(store['state/thp'], state.thp)


def test_store_writer_preserves_tree_header_and_protects_base_keys(tmp_path, monkeypatch):
    cfg, state, _, _ = _member(tmp_path, monkeypatch, 1)
    store = {k: v.copy() for k, v in physics_inventory.carrier_manifest(state).items()}
    args = dict(setup=restart_stream.capture_domain_setup(state),
                template_state=state, scalars=physics_inventory.carrier_scalars(state),
                check_pinned=False)
    header = {'checkpoint_set_id': 'one-generation', 'elapsed_ticks': 20000,
              'tick_den': 1000, 'adaptive_clock': {'step_count': 3}}
    result = restart_stream.write_streamed_restart(
        tmp_path/'tree.npz', store, cfg, tree_header=header, **args)
    written, arrays = restart._load_restart(result.path, with_arrays=True)
    assert all(written[key] == value for key, value in header.items())
    np.testing.assert_array_equal(arrays['state/thp'], state.thp)
    with pytest.raises(ValueError, match='may not replace base keys'):
        restart_stream.write_streamed_restart(
            tmp_path/'collision.npz', store, cfg,
            tree_header={'elapsed_seconds': 999.}, **args)
    assert not (tmp_path/'collision.npz').exists()


def test_actual_tree_door_uses_both_stores_and_reseeds_after_validation(tmp_path, monkeypatch):
    from datetime import timedelta
    from types import SimpleNamespace
    from gpuwm.core.streaming import StreamedDomain
    from test_restart import _lifecycle_free_model

    class Endpoint:
        write_restart = StreamedDomain.write_restart
        validate_restart = StreamedDomain.validate_restart
        apply_restart = StreamedDomain.apply_restart
        impose_clock = StreamedDomain.impose_clock

    def attach(node, *, clear):
        endpoint = Endpoint()
        endpoint.store = {k: v.copy() for k, v in
                          physics_inventory.carrier_manifest(node.state).items()}
        if clear:
            for array in endpoint.store.values():
                array.fill(0)
        endpoint.template_state = node.state
        setup = restart_stream.capture_domain_setup(node.state)
        endpoint.restart_setup = lambda: setup
        endpoint.scalars = physics_inventory.carrier_scalars(node.state)
        endpoint.host_store = False  # CPU storage witness; no pinning or GPU.
        endpoint.reseeds = []
        endpoint._run = SimpleNamespace(reseed_clock=lambda value:
                                        endpoint.reseeds.append(dict(value)))
        node.state._streamed_domain = endpoint
        return endpoint

    source, start = _lifecycle_free_model(monkeypatch)
    endpoints = {node.cfg.grid_id: attach(node, clear=False)
                 for node in source.walk_parent_first()}
    for node in source.walk_parent_first():
        endpoints[node.cfg.grid_id].store['state/thp'].fill(100 + node.cfg.grid_id)
        node.state.thp.fill(-99)  # stale resident copy must never be published.
    def resident_forbidden(*args, **kwargs):
        raise AssertionError('tree checkpoint reached the resident payload route')
    monkeypatch.setattr(restart, 'write_restart', resident_forbidden)
    path = restart.write_tree_restart(tmp_path/'source', source,
                                      start+timedelta(seconds=3600))
    resumed, _ = _lifecycle_free_model(monkeypatch)
    targets = {node.cfg.grid_id: attach(node, clear=True)
               for node in resumed.walk_parent_first()}
    monkeypatch.setattr(restart, '_validate_restart', resident_forbidden)
    result = restart.restore_tree_restart(path, resumed)
    for gid, endpoint in targets.items():
        np.testing.assert_array_equal(endpoint.store['state/thp'],
                                      endpoints[gid].store['state/thp'])
        assert endpoint.reseeds[-1]['elapsed_seconds'] == 3600.
    assert result.already_complete
