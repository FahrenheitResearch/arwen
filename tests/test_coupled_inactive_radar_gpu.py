"""Strict history retains the stock inactive radar slot, with default intact."""
from types import SimpleNamespace
import numpy as np
import pytest
cp=pytest.importorskip("cupy")


def test_kessler_stock_history_and_default_rain_diagnostic(monkeypatch):
    from gpuwm import wrf_exact
    from gpuwm.core.refl import compute_and_stash_refl_10cm, compute_refl_10cm
    shape=(3,2,2)
    buffer=cp.empty(shape,cp.float32)
    state=SimpleNamespace(p=cp.full(shape,90000.,cp.float32),
                          qv=cp.full(shape,.01,cp.float32),
                          qr=cp.full(shape,.002,cp.float32),
                          physics=SimpleNamespace(refl_10cm=None),
                          scratch=lambda *_:buffer)
    cfg=SimpleNamespace(mp_physics=1)
    temperature=cp.full(shape,290.,cp.float32)
    monkeypatch.setattr(wrf_exact,"ENABLED",True)
    compute_and_stash_refl_10cm(state,cfg,temperature,state.p)
    np.testing.assert_array_equal(cp.asnumpy(state.physics.refl_10cm).view('u4'),0)
    # The physical diagnostic remains callable in the same strict process.
    direct=compute_refl_10cm(state,cfg,temperature=temperature,pressure=state.p)
    assert bool(cp.all(direct>0))
    expected=cp.asnumpy(direct).copy()
    state.physics.refl_10cm=None
    monkeypatch.setattr(wrf_exact,"ENABLED",False)
    compute_and_stash_refl_10cm(state,cfg,temperature,state.p)
    np.testing.assert_array_equal(cp.asnumpy(state.physics.refl_10cm).view('u4'),expected.view('u4'))
