"""A checkpoint inside a forcing interval can retain the full old prefix."""
import numpy as np
import pytest

from gpuwm.io import restart
from test_restart import _cfg, _extension_state, _external_mirror


def _live(cfg, monkeypatch, **kwargs):
    state = _extension_state(cfg, monkeypatch, **kwargs)
    state._lateral_boundary_device = _external_mirror(bound=True)
    return state


@pytest.mark.parametrize('count', [2, 3])
def test_preserved_prefix_restores_full_state_before_or_after_append(tmp_path, monkeypatch, count):
    cfg = _cfg(moist=True, mp_physics=1, specified=True, spec_bdy_width=3, spec_zone=1, relax_zone=2)
    original = _live(cfg, monkeypatch, count=2)
    original.elapsed_seconds = 60.
    original.thp.fill(17.)
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg, preserved_forcing_prefix=True)
    live = _live(cfg, monkeypatch, count=count)
    live.thp.fill(-2.)
    info = restart.restore_restart(path, live, cfg, preserved_forcing_prefix=True)
    assert info.elapsed_seconds == 60. and live.elapsed_seconds == 60.
    np.testing.assert_array_equal(live.thp, original.thp)


@pytest.mark.parametrize('mutation', ['future-in-old-prefix', 'base', 'shorter', 'clock', 'ordinary'])
def test_preserved_prefix_rejects_reinterpretation_before_state_mutation(tmp_path, monkeypatch, mutation):
    cfg = _cfg(moist=True, mp_physics=1, specified=True, spec_bdy_width=3, spec_zone=1, relax_zone=2)
    original = _live(cfg, monkeypatch, count=3)
    original.elapsed_seconds = 60.
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg,
                                 preserved_forcing_prefix=mutation != 'ordinary')
    live = _live(cfg, monkeypatch, count=2 if mutation == 'shorter' else 4,
                 seed=12 if mutation == 'future-in-old-prefix' else 11)
    if mutation == 'base':
        live.thb += 1
    if mutation == 'clock':
        # A real writer must reject a clock beyond the available forcing.
        original.elapsed_seconds = 10000.
        with pytest.raises(restart.RestartMismatchError, match='ends before'):
            restart.write_restart(tmp_path / 'bad-clock.npz', original, cfg, preserved_forcing_prefix=True)
        return
    before = live.thp.copy()
    with pytest.raises(restart.RestartMismatchError):
        restart.restore_restart(path, live, cfg, preserved_forcing_prefix=True)
    np.testing.assert_array_equal(live.thp, before)


# ---------------------------------------------------------------------------
# Chained preparation: a checkpoint written before the preparation seals
# ---------------------------------------------------------------------------

from collections.abc import Sequence
from dataclasses import replace


class _Streaming(Sequence):
    """A streamed boundary series with ``ready`` intervals prepared.

    Asking for an interval that is not prepared yet fails loudly here,
    which is how these tests prove a pre-seal checkpoint never waits.
    """

    def __init__(self, intervals, *, ready, head="a" * 64, sealed=False):
        self._intervals = tuple(intervals)
        self.ready = ready
        self.head_sha256 = head
        self._sealed = sealed
        self.bounds = tuple((i.start_seconds, i.end_seconds)
                            for i in self._intervals)

    def ready_prefix(self):
        return len(self._intervals) if self._sealed else self.ready

    def sealed(self):
        return self._sealed

    def __len__(self):
        return len(self._intervals)

    def __getitem__(self, index):
        if index >= len(self._intervals):
            raise IndexError(index)
        if index >= self.ready_prefix():
            raise AssertionError(f"waited on unprepared interval {index}")
        return self._intervals[index]


def _streamed(state, **kwargs):
    boundaries = state.lateral_boundaries
    state.lateral_boundaries = replace(
        boundaries, intervals=_Streaming(boundaries.intervals, **kwargs))
    return state


def _specified():
    return _cfg(moist=True, mp_physics=1, specified=True, spec_bdy_width=3,
                spec_zone=1, relax_zone=2)


def test_a_checkpoint_before_the_seal_waits_for_nothing(tmp_path,
                                                        monkeypatch):
    cfg = _specified()
    original = _streamed(_live(cfg, monkeypatch, count=4), ready=1)
    original.elapsed_seconds = 60.
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg)
    header, _ = restart._load_restart(path, with_arrays=False)
    assert header['forcing_extension_mode'] == \
        restart.PRESERVED_FORCING_PREFIX_MODE
    assert header['boundary_stream'] == {'head_sha256': 'a' * 64}
    assert len(header['lateral_boundary_prefix']['intervals']) == 1


def test_a_pre_seal_checkpoint_resumes_on_the_sealed_series(tmp_path,
                                                             monkeypatch):
    cfg = _specified()
    original = _streamed(_live(cfg, monkeypatch, count=4), ready=2)
    original.elapsed_seconds = 60.
    original.thp.fill(23.)
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg)
    live = _streamed(_live(cfg, monkeypatch, count=4), ready=4, sealed=True)
    live.thp.fill(-1.)
    info = restart.restore_restart(path, live, cfg)
    assert info.elapsed_seconds == 60.
    np.testing.assert_array_equal(live.thp, original.thp)


@pytest.mark.parametrize('ready', [2, 3])
def test_a_pre_seal_checkpoint_resumes_on_the_series_still_streaming(
        tmp_path, monkeypatch, ready):
    """A forecast killed before the seal resumes beside the same
    preparation: only the recorded intervals are compared, so the resume
    never waits for an interval the preparation has not built yet."""

    cfg = _specified()
    original = _streamed(_live(cfg, monkeypatch, count=4), ready=2)
    original.elapsed_seconds = 60.
    original.thp.fill(29.)
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg)
    live = _streamed(_live(cfg, monkeypatch, count=4), ready=ready)
    live.thp.fill(-1.)
    info = restart.restore_restart(path, live, cfg)
    assert info.elapsed_seconds == 60.
    np.testing.assert_array_equal(live.thp, original.thp)


def test_a_pre_seal_checkpoint_refuses_changed_bytes_while_streaming(
        tmp_path, monkeypatch):
    cfg = _specified()
    original = _streamed(_live(cfg, monkeypatch, count=4), ready=2)
    original.elapsed_seconds = 60.
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg)
    live = _streamed(_live(cfg, monkeypatch, count=4, seed=12), ready=2)
    before = live.thp.copy()
    with pytest.raises(restart.RestartMismatchError):
        restart.restore_restart(path, live, cfg)
    np.testing.assert_array_equal(live.thp, before)


def test_a_pre_seal_checkpoint_refuses_another_head(tmp_path, monkeypatch):
    cfg = _specified()
    original = _streamed(_live(cfg, monkeypatch, count=3), ready=1)
    original.elapsed_seconds = 60.
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg)
    live = _streamed(_live(cfg, monkeypatch, count=3), ready=3, sealed=True,
                     head='b' * 64)
    before = live.thp.copy()
    with pytest.raises(restart.RestartMismatchError, match='prepared head'):
        restart.restore_restart(path, live, cfg)
    np.testing.assert_array_equal(live.thp, before)


@pytest.mark.parametrize('seed', [11, 12])
def test_a_pre_seal_checkpoint_resumes_under_the_sealed_binding_only_on_the_same_bytes(
        tmp_path, monkeypatch, seed):
    """Resumed on the sealed tree bound by its proof (an eager series), the
    recorded intervals must still be byte-identical: a different
    preparation (another seed) is refused before any state is touched."""

    cfg = _specified()
    original = _streamed(_live(cfg, monkeypatch, count=3), ready=2)
    original.elapsed_seconds = 60.
    original.thp.fill(5.)
    path = restart.write_restart(tmp_path / 'state.npz', original, cfg)
    live = _live(cfg, monkeypatch, count=3, seed=seed)
    live.thp.fill(-3.)
    if seed == 11:
        restart.restore_restart(path, live, cfg)
        np.testing.assert_array_equal(live.thp, original.thp)
    else:
        before = live.thp.copy()
        with pytest.raises(restart.RestartMismatchError):
            restart.restore_restart(path, live, cfg)
        np.testing.assert_array_equal(live.thp, before)


def test_a_checkpoint_after_the_seal_is_the_ordinary_checkpoint(
        tmp_path, monkeypatch):
    cfg = _specified()
    eager = _live(cfg, monkeypatch, count=3)
    eager.elapsed_seconds = 60.
    streamed = _streamed(_live(cfg, monkeypatch, count=3), ready=3,
                         sealed=True)
    streamed.elapsed_seconds = 60.
    one = restart.write_restart(tmp_path / 'eager.npz', eager, cfg)
    two = restart.write_restart(tmp_path / 'streamed.npz', streamed, cfg)
    first, a = restart._load_restart(one, with_arrays=True)
    second, b = restart._load_restart(two, with_arrays=True)
    for key in ('setup_fingerprint', 'config', 'array_manifest'):
        assert first[key] == second[key]
    assert 'boundary_stream' not in second
    assert 'forcing_extension_mode' not in second
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
