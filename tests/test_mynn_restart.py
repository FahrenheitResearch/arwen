"""Rounded mixing length has a distinct continuation identity."""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.io import restart
from test_restart import _cfg, _fill_setup, _shim_state


def test_previous_mixing_length_is_rejected_before_state_restore(
        tmp_path, monkeypatch):
    cfg = _cfg(moist=True, bl_pbl_physics=5)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    with monkeypatch.context() as old:
        old.setitem(restart.PBL_ALGORITHM_IDENTITIES, 5,
                    "mynn-edmf-pbl-wrf-v4.6.1-v1")
        path = restart.write_restart(tmp_path / "previous.npz", source, cfg)
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    fresh.qv.fill(np.float32(.0123))
    before = fresh.qv.tobytes()
    with pytest.raises(restart.RestartMismatchError, match="physics setup"):
        restart.restore_restart(path, fresh, cfg)
    assert fresh.qv.tobytes() == before


def test_rounded_mixing_length_checkpoint_round_trips(tmp_path, monkeypatch):
    cfg = _cfg(moist=True, bl_pbl_physics=5)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    source.qv.fill(np.float32(.0042))
    path = restart.write_restart(tmp_path / "current.npz", source, cfg)
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)
    np.testing.assert_array_equal(fresh.qv, source.qv)
