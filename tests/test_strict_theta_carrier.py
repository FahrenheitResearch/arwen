"""Strict mode's theta carrier on the prepared door (lane 291 strict-nonfinite).

The breakage this pins: under GPUWM_WRF_EXACT=1 every strict kernel reads
``thp`` as WRF's ``t = theta - 300`` (kernels/diagnostics.cu), but the
prepared door left the preparation's 290-464 K base profile in ``thb``, so a
default ``gpuwm go`` config read theta up to 164 K low and went non-finite in
its first steps.  CPU only: the re-expression and the setup fingerprint.
"""
from types import SimpleNamespace

import numpy as np

import gpuwm.wrf_exact as wrf_exact
from gpuwm.core.state import adopt_wrf_theta_carrier
from gpuwm.state_serialization_contract import (
    STATE_SETUP_ARRAYS, STATE_SETUP_SCALARS, setup_core_fingerprint)


def _prepared_like_state():
    rng = np.random.default_rng(7)
    state = SimpleNamespace()
    for name in STATE_SETUP_ARRAYS:
        setattr(state, name, rng.random(4, dtype=np.float32))
    for name in STATE_SETUP_SCALARS:
        setattr(state, name, 1.5)
    state.thb = np.linspace(290.0, 464.0, 3 * 2 * 2,
                            dtype=np.float32).reshape(3, 2, 2)
    state.thp = rng.uniform(-12.0, 26.0, (3, 2, 2)).astype(np.float32)
    return state


def test_strict_prepared_state_holds_wrf_theta_and_keeps_its_setup(monkeypatch):
    monkeypatch.setattr(wrf_exact, "ENABLED", True)
    state = _prepared_like_state()
    history_t = (state.thb + state.thp) - np.float32(300.0)
    fingerprint = setup_core_fingerprint(state)
    adopt_wrf_theta_carrier(state)
    assert (state.thb == np.float32(300.0)).all()
    # The carrier word is the default run's history T word, bit for bit.
    assert state.thp.tobytes() == history_t.astype(np.float32).tobytes()
    # The setup the preparation recorded still identifies this state.
    assert setup_core_fingerprint(state) == fingerprint
    # A second call (thb already 300) changes nothing.
    before = state.thp.copy()
    adopt_wrf_theta_carrier(state)
    assert state.thp.tobytes() == before.tobytes()


def test_default_mode_leaves_the_prepared_state_alone(monkeypatch):
    monkeypatch.setattr(wrf_exact, "ENABLED", False)
    state = _prepared_like_state()
    thb, thp = state.thb.copy(), state.thp.copy()
    fingerprint = setup_core_fingerprint(state)
    adopt_wrf_theta_carrier(state)
    assert state.thb.tobytes() == thb.tobytes()
    assert state.thp.tobytes() == thp.tobytes()
    assert not hasattr(state, "_setup_thb")
    assert setup_core_fingerprint(state) == fingerprint
