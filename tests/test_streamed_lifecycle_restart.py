"""Canonical lifecycle checkpoints preserve live windows and every sibling."""
from datetime import timedelta
from types import SimpleNamespace
import numpy as np
import pytest
from gpuwm.io import restart
from gpuwm.core.streaming import StreamedDomain
from tilestream import physics_inventory, restart_stream
from test_restart import _lifecycle_tree_fixture, _rewrite_restart_archive

class Endpoint:
    write_restart = StreamedDomain.write_restart
    validate_restart = StreamedDomain.validate_restart
    apply_restart = StreamedDomain.apply_restart
    impose_clock = StreamedDomain.impose_clock

    def publish(self, names):
        # The existing header publication only touches tracker planes.
        for name in names:
            self.template_state._scratch[name.split('/', 1)[1]][...] = self.store[name]
        return sum(self.store[name].nbytes for name in names)

def attach(node, *, clear=False):
    endpoint = Endpoint()
    endpoint.store = {k: v.copy() for k, v in physics_inventory.carrier_manifest(node.state).items()}
    if clear:
        for value in endpoint.store.values(): value.fill(-3)
    endpoint.template_state = node.state
    setup = restart_stream.capture_domain_setup(node.state)
    endpoint.restart_setup = lambda: setup
    endpoint.scalars = physics_inventory.carrier_scalars(node.state)
    endpoint.host_store = False
    endpoint.reseeds = []
    endpoint._run = SimpleNamespace(reseed_clock=lambda x: endpoint.reseeds.append(dict(x)))
    node.state._streamed_domain = endpoint
    return endpoint

def fixture(monkeypatch, *, streamed, clear=False):
    model, start = _lifecycle_tree_fixture(monkeypatch, follow=False, spawn=False)
    endpoints = {}
    for node in model.walk_parent_first():
        # The old format fixture intentionally fills impossible MP boundary
        # rings; normalize it to a state the present forecast can produce.
        from gpuwm.core.microphysics import normalize_spec_zone_ring_after_restore
        normalize_spec_zone_ring_after_restore(node.state, node.cfg.run)
        if streamed:
            endpoints[node.cfg.grid_id] = attach(node, clear=clear)
    assert restart.declares_nest_lifecycle(model)
    return model, start, endpoints

@pytest.mark.parametrize('restore_streamed', [False, True])
def test_lifecycle_tree_writes_live_store_and_restores_fixed_windows(tmp_path, monkeypatch, restore_streamed):
    source, start, endpoints = fixture(monkeypatch, streamed=True)
    for node in source.walk_parent_first():
        store = endpoints[node.cfg.grid_id].store
        store['state/thp'].fill(100 + node.cfg.grid_id)
        node.state.thp.fill(-99)
        for key, value in store.items():
            if key.startswith('scratch/uh_') and key.endswith('_window'):
                value.fill(37 + node.cfg.grid_id)
                node.state._scratch[key.split('/', 1)[1]].fill(-99)
    def forbid(*args, **kwargs):
        raise AssertionError('canonical lifecycle store reached stale resident writer')
    with monkeypatch.context() as patch:
        patch.setattr(restart, 'write_restart', forbid)
        path = restart.write_tree_restart(tmp_path/'source', source, start+timedelta(seconds=3600))
    resumed, _, targets = fixture(monkeypatch, streamed=restore_streamed, clear=True)
    if restore_streamed:
        monkeypatch.setattr(restart, '_validate_restart', forbid)
    result = restart.restore_tree_restart(path, resumed)
    assert result.already_complete
    for node in resumed.walk_parent_first():
        gid = node.cfg.grid_id
        got = targets[gid].store if restore_streamed else physics_inventory.carrier_manifest(node.state)
        for key, expected in endpoints[gid].store.items():
            if key in got: np.testing.assert_array_equal(got[key], expected, err_msg=key)
        if restore_streamed:
            assert targets[gid].reseeds[-1]['elapsed_seconds'] == 3600.
    assert restart.read_tree_lifecycle_header(path, resumed).window_slots

def test_invalid_sibling_lifecycle_window_cannot_partially_restore_store(tmp_path, monkeypatch):
    source, start, _ = fixture(monkeypatch, streamed=True)
    path = restart.write_tree_restart(tmp_path/'source', source, start+timedelta(seconds=3600))
    child = next(path.parent.glob('gpuwmrst_d02_*.npz'))
    def remove(payload, header): payload.pop('scratch/uh_follow_window')
    corrupt = _rewrite_restart_archive(child, tmp_path/'corrupt.npz', remove)
    child.write_bytes(corrupt.read_bytes())
    resumed, _, targets = fixture(monkeypatch, streamed=True, clear=True)
    before = {gid: {k: v.copy() for k, v in endpoint.store.items()} for gid, endpoint in targets.items()}
    clocks = {gid: dict(endpoint.scalars) for gid, endpoint in targets.items()}
    with pytest.raises((restart.RestartMismatchError, restart_stream.RestartRefused), match='missing'):
        restart.restore_tree_restart(path, resumed)
    for gid, endpoint in targets.items():
        for key, expected in before[gid].items(): np.testing.assert_array_equal(endpoint.store[key], expected)
        assert endpoint.scalars == clocks[gid]
        assert not endpoint.reseeds


def test_ordinary_stream_checkpoint_still_omits_and_resets_carried_windows(tmp_path, monkeypatch):
    model, _, endpoints = fixture(monkeypatch, streamed=True)
    endpoint = endpoints[1]
    for key, value in endpoint.store.items():
        if key.startswith('scratch/uh_') and key.endswith('_window'): value.fill(81.)
    result = endpoint.write_restart(tmp_path/'ordinary.npz', model.root.cfg.run)
    header = restart.read_restart_header(result.path)
    assert not any(key.endswith('_window') for key in header['array_manifest'])
    endpoint.apply_restart(endpoint.validate_restart(result.path, model.root.cfg.run))
    for key, value in endpoint.store.items():
        if key.startswith('scratch/uh_') and key.endswith('_window'): assert not value.any()


from conftest import requires_gpu

@requires_gpu
@pytest.mark.gpu
def test_actual_lifecycle_tree_checkpoint_preserves_tiled_trajectory(monkeypatch, tmp_path):
    from test_both_streamed_nesting import trajectory
    def run(mode, **kwargs):
        return trajectory(monkeypatch, mode=mode, feedback=1, steps=5,
                          lifecycle=True, **kwargs)
    resident = run("resident", checkpoint_dir=tmp_path/'resident')
    streamed = run("parent", checkpoint_dir=tmp_path/'streamed')
    assert streamed['domains'] == resident['domains']
    assert streamed['cadence'] == resident['cadence']
    for initial, mode in ((streamed, 'parent'), (streamed, 'resident'), (resident, 'parent')):
        resumed = run(mode, restart_from=initial['checkpoints'][0])
        assert resumed['domains'] == initial['domains']
        assert resumed['cadence'] == initial['cadence']
    with np.load(streamed['checkpoints'][0], allow_pickle=False) as checkpoint:
        for key in ('scratch/uh_follow_window', 'scratch/uh_spawn_window'):
            assert np.all(checkpoint[key] >= 123.25)
