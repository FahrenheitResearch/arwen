"""Grade the heating fold against its original eager operation chain."""
import numpy as np
import pytest


def test_custom_mass_provider_keeps_eager_contract():
    from types import SimpleNamespace
    from gpuwm.core.dycore import add_h_diabatic_tendency
    state = SimpleNamespace(
        rth_t=np.ones((2, 2, 2), np.float32),
        h_diabatic=np.ones((2, 2, 2), np.float32),
        c1h=np.asarray([1, 2], np.float32),
        c2h=np.asarray([3, 4], np.float32), has_msf=False,
        total_mu=lambda: np.full((2, 2), 20, np.float32))
    add_h_diabatic_tendency(state)
    np.testing.assert_array_equal(state.rth_t,
                                  np.asarray([np.full((2, 2), 24),
                                              np.full((2, 2), 45)], np.float32))


@pytest.mark.gpu
@pytest.mark.parametrize('mapped', [False, True])
def test_heating_words_match_eager(mapped):
    import cupy as cp
    from gpuwm.core.state import DomainState
    from gpuwm.core.dycore import (add_h_diabatic_tendency,
                                   _add_h_diabatic_tendency_eager)
    rng = np.random.default_rng(4913)
    state = DomainState.__new__(DomainState)
    nz, ny, nx = 19, 13, 17
    for name in ['rth_t', 'h_diabatic']:
        setattr(state, name, cp.asarray(rng.normal(size=(nz, ny, nx)), dtype=cp.float32))
    state.mub2d = cp.asarray(rng.uniform(50000, 90000, (ny, nx)), dtype=cp.float32)
    state.mup = cp.asarray(rng.normal(size=(ny, nx)), dtype=cp.float32)
    state.msft = cp.asarray(rng.uniform(.7, 1.4, (ny, nx)), dtype=cp.float32)
    state.c1h = cp.asarray(rng.uniform(0, 1, nz), dtype=cp.float32)
    state.c2h = cp.asarray(rng.uniform(0, 10000, nz), dtype=cp.float32)
    state.has_msf = mapped
    expected = DomainState.__new__(DomainState)
    expected.__dict__.update(state.__dict__)
    expected.rth_t = state.rth_t.copy()
    _add_h_diabatic_tendency_eager(expected)
    add_h_diabatic_tendency(state)
    np.testing.assert_array_equal(state.rth_t.view(cp.uint32).get(),
                                  expected.rth_t.view(cp.uint32).get())
