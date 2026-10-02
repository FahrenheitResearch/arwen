"""Tile checkpoints retain prepared forcing without waiting for its seal."""
import numpy as np
import pytest

from gpuwm.io import restart
from tilestream import physics_inventory, restart_stream
from test_restart_preserved_forcing import (
    _clear_out_state, _specified, _streamed,
)


def _store(state):
    return {key: value.copy() for key, value in
            physics_inventory.carrier_manifest(state).items()}


def _write(path, state, cfg):
    return restart_stream.write_streamed_restart(
        path, _store(state), cfg,
        scalars=physics_inventory.carrier_scalars(state),
        setup=restart_stream.capture_domain_setup(state),
        template_state=state, check_pinned=False)


def _validate(path, state, cfg, store, scalars):
    return restart_stream.validate_streamed_restart(
        path, store, cfg, setup=restart_stream.capture_domain_setup(state),
        template_state=state, scalars=scalars)


def test_tile_checkpoint_before_seal_reads_only_the_ready_prefix(tmp_path, monkeypatch):
    cfg = _specified()
    original = _streamed(_clear_out_state(cfg, monkeypatch), ready=1)
    original.elapsed_seconds = 60.
    info = _write(tmp_path / 'prefix.npz', original, cfg)
    header = info.header
    assert header[restart.BOUNDARY_STREAM_HEADER_KEY] == {'head_sha256': 'a' * 64}
    assert header['forcing_extension_mode'] == restart.PRESERVED_FORCING_PREFIX_MODE
    assert len(header['lateral_boundary_prefix']['intervals']) == 1
    assert header['setup_core_fingerprint'] == restart.setup_core_fingerprint(original)


@pytest.mark.parametrize('live_mode', ['streaming', 'sealed', 'eager'])
def test_tile_prefix_checkpoint_restores_every_carrier_on_its_twin(
        tmp_path, monkeypatch, live_mode):
    cfg = _specified()
    original = _streamed(_clear_out_state(cfg, monkeypatch), ready=2)
    original.elapsed_seconds = 3600.
    original.thp.fill(31.)
    info = _write(tmp_path / 'prefix.npz', original, cfg)
    live = _clear_out_state(cfg, monkeypatch)
    if live_mode != 'eager':
        _streamed(live, ready=2 if live_mode == 'streaming' else 3,
                  sealed=live_mode == 'sealed')
    live.thp.fill(-7.)
    store = _store(live)
    for value in store.values():
        value.fill(0.)
    scalars = {'elapsed_seconds': 0.}
    staged = _validate(info.path, live, cfg, store, scalars)
    assert scalars['elapsed_seconds'] == 0.
    assert all(not value.any() for value in store.values())
    restored = staged.apply()
    assert restored.elapsed_seconds == scalars['elapsed_seconds'] == 3600.
    assert restored.device_copies == 0
    for key, expected in _store(original).items():
        np.testing.assert_array_equal(store[key], expected)
    # The resident reader recognizes the same head-bound prefix contract.
    restart.restore_restart(info.path, live, cfg)
    for key, expected in _store(original).items():
        np.testing.assert_array_equal(_store(live)[key], expected)


@pytest.mark.parametrize('mutation,reason', [
    ('head', 'prepared head'),
    ('base', 'immutable base state'),
    ('prefix', 'changed a previously declared interval'),
    ('splice', 'discontinuous preserved forcing frame'),
    ('shorter', 'changed a previously declared interval'),
])
def test_tile_prefix_restore_refuses_changed_forcing_before_mutation(
        tmp_path, monkeypatch, mutation, reason):
    cfg = _specified()
    original = _streamed(_clear_out_state(cfg, monkeypatch), ready=2)
    original.elapsed_seconds = 3600.
    info = _write(tmp_path / 'prefix.npz', original, cfg)
    kwargs = ({'seed': 4} if mutation == 'prefix'
              else {'jump': 1e-4} if mutation == 'splice'
              else {'count': 2} if mutation == 'shorter' else {})
    live = _clear_out_state(cfg, monkeypatch, **kwargs)
    _streamed(live, ready=len(live.lateral_boundaries.intervals), sealed=True,
              head='b' * 64 if mutation == 'head' else 'a' * 64)
    if mutation == 'base':
        live.thb += 1.
    store = _store(live)
    before = {key: value.copy() for key, value in store.items()}
    scalars = {'elapsed_seconds': 0.}
    with pytest.raises(restart_stream.RestartRefused, match=reason):
        _validate(info.path, live, cfg, store, scalars)
    assert scalars['elapsed_seconds'] == 0.
    for key in store:
        np.testing.assert_array_equal(store[key], before[key])


@pytest.mark.parametrize('mutation,reason', [
    ('clock', 'ends before the checkpoint clock'),
    ('splice', 'discontinuous preserved forcing frame'),
])
def test_tile_prefix_writer_retains_forcing_guards(tmp_path, monkeypatch, mutation, reason):
    cfg = _specified()
    original = _streamed(_clear_out_state(
        cfg, monkeypatch, jump=1e-4 if mutation == 'splice' else 0.), ready=2)
    original.elapsed_seconds = 8000. if mutation == 'clock' else 3600.
    path = tmp_path / 'refused.npz'
    with pytest.raises(restart_stream.RestartRefused, match=reason):
        _write(path, original, cfg)
    assert not path.exists()


def test_tile_checkpoint_after_seal_uses_ordinary_full_setup(tmp_path, monkeypatch):
    cfg = _specified()
    eager = _clear_out_state(cfg, monkeypatch)
    eager.elapsed_seconds = 3600.
    sealed = _streamed(_clear_out_state(cfg, monkeypatch), ready=3, sealed=True)
    sealed.elapsed_seconds = 3600.
    one = _write(tmp_path / 'eager.npz', eager, cfg)
    two = _write(tmp_path / 'sealed.npz', sealed, cfg)
    for key in ('setup_fingerprint', 'config', 'array_manifest'):
        assert one.header[key] == two.header[key]
    assert restart.BOUNDARY_STREAM_HEADER_KEY not in two.header
    assert 'forcing_extension_mode' not in two.header
    _, expected = restart._load_restart(one.path, with_arrays=True)
    _, actual = restart._load_restart(two.path, with_arrays=True)
    assert set(expected) == set(actual)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
