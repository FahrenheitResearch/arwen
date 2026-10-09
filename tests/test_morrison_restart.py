"""Finite-transfer corrections are a distinct continuation algorithm."""
from __future__ import annotations

import re

import numpy as np
import pytest

from gpuwm.io import restart
from test_restart import _cfg, _fill_setup, _shim_state


@pytest.mark.parametrize("mode", (0, 1))
def test_prior_morrison_algorithm_is_rejected_before_state_restore(
        mode, tmp_path, monkeypatch):
    cfg = _cfg(moist=True, mp_physics=10, morr_rimed_ice=mode)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    previous = "morrison-two-moment-v2-kf-number-seeding"
    with monkeypatch.context() as old:
        old.setitem(restart.MICROPHYSICS_ALGORITHM_IDENTITIES, 10, previous)
        path = restart.write_restart(tmp_path / "previous.npz", source, cfg)
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    fresh.qv.fill(np.float32(.0123))
    before = fresh.qv.tobytes()
    moved = re.escape(
        f"mp_physics=10 (microphysics): {previous} -> "
        f"{restart.MICROPHYSICS_ALGORITHM_IDENTITIES[10]}")
    with pytest.raises(restart.RestartMismatchError, match=moved):
        restart.restore_restart(path, fresh, cfg)
    assert fresh.qv.tobytes() == before


@pytest.mark.parametrize("mode", (0, 1))
def test_current_morrison_algorithm_round_trips(mode, tmp_path, monkeypatch):
    cfg = _cfg(moist=True, mp_physics=10, morr_rimed_ice=mode)
    source = _shim_state(cfg, monkeypatch)
    _fill_setup(source)
    source.qv.fill(np.float32(.0042))
    path = restart.write_restart(tmp_path / "current.npz", source, cfg)
    fresh = _shim_state(cfg, monkeypatch)
    _fill_setup(fresh)
    restart.restore_restart(path, fresh, cfg)
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni", "ns", "ng"):
        np.testing.assert_array_equal(getattr(fresh, name), getattr(source, name))
