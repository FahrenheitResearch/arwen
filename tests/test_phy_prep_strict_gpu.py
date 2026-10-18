"""Strict phy_prep must preserve the scalar reference's individual roundings."""
import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu


def _state(cp):
    shape = (4, 2, 3)
    # Saved-style float32 inputs covering different pressure and moisture words.
    rng = np.random.default_rng(78201)
    pressure = cp.asarray(rng.uniform(3000, 105000, shape).astype(np.float32))
    theta = cp.asarray(rng.uniform(270, 345, shape).astype(np.float32))
    alt = cp.asarray(rng.uniform(0.6, 4, shape).astype(np.float32))
    qv = cp.asarray(rng.uniform(0, 0.025, shape).astype(np.float32))
    # First density mismatch in the instrumented real step-1 field.
    alt[0, 0, 0] = np.array([0x3f692402], dtype=np.uint32).view(np.float32)[0]
    qv[0, 0, 0] = np.array([0x3c75aa09], dtype=np.uint32).view(np.float32)[0]
    return SimpleNamespace(
        p=pressure, alt=alt, total_theta=lambda: theta,
        u=cp.zeros((4, 2, 4), dtype=cp.float32),
        v=cp.zeros((4, 3, 3), dtype=cp.float32),
        phb=cp.arange(5, dtype=cp.float32)*np.float32(981),
        php=cp.zeros((5, 2, 3), dtype=cp.float32),
        c1h=cp.ones(4, dtype=cp.float32), c2h=cp.zeros(4, dtype=cp.float32),
        total_mu=lambda: cp.full((2, 3), 90000, dtype=cp.float32),
        dnw=cp.full(4, -0.25, dtype=cp.float32), p_top=np.float32(2000),
        qv=qv, qc=cp.zeros(shape, dtype=cp.float32), qi=None, qs=None,
        scratch=lambda shape, name: cp.zeros(shape, dtype=cp.float32))


@requires_gpu
def test_strict_phy_prep_exner_uses_the_scalar_reference_power():
    import cupy as cp
    from gpuwm.core import constants as c
    from gpuwm.core.physics import _prepare_atmosphere
    from gpuwm.wrf_exact import ENABLED
    if not ENABLED:
        pytest.skip("strict process required")
    state = _state(cp)
    actual = cp.asnumpy(_prepare_atmosphere(state)["exner"])
    libm = ctypes.CDLL("libm.so.6")
    powf = libm.powf
    powf.argtypes, powf.restype = [ctypes.c_float, ctypes.c_float], ctypes.c_float
    ratio = cp.asnumpy(state.p) / np.float32(c.P0)
    expected = np.asarray([powf(float(x), float(np.float32(c.RCP)))
                           for x in ratio.flat], dtype=np.float32).reshape(ratio.shape)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@requires_gpu
def test_strict_phy_prep_density_rounds_reciprocal_before_multiplication():
    import cupy as cp
    from gpuwm.core.physics import _prepare_atmosphere
    from gpuwm.wrf_exact import ENABLED
    if not ENABLED:
        pytest.skip("strict process required")
    state = _state(cp)
    alt, qv = cp.asnumpy(state.alt), cp.asnumpy(state.qv)
    expected = (np.float32(1)/alt) * (np.float32(1)+qv)
    old = (np.float32(1)+qv)/alt
    assert np.any(expected.view(np.uint32) != old.view(np.uint32))
    actual = cp.asnumpy(_prepare_atmosphere(state)["rho"])
    assert actual.view(np.uint32)[0, 0, 0] == 0x3f8ea865
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
