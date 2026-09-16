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
