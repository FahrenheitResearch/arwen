"""Coordinate mixing must retain evolved canonical perturbation bits."""
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.mark.parametrize("base_3d", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_evolved_theta_keeps_the_selected_basis_words(monkeypatch, strict, base_3d):
    from gpuwm.core import dycore
    monkeypatch.setattr(dycore, "WRF_EXACT", strict)
    theta = np.array([0x3f800001, 0xbf800001], np.uint32).view(np.float32).reshape(2, 1, 1)
    base = np.full(theta.shape if base_3d else (2,), 300, np.float32)
    buffers = {}
    def scratch(shape, name):
        buffers.setdefault(name, np.zeros(shape, np.float32))
        return buffers[name]
    state = SimpleNamespace(thb=base, scratch=scratch)
    result = dycore._coordinate_theta_field(state, theta)
    expected = np.array([0x3f800001, 0xbf800001] if strict else
                        [0x3f800000, 0xbf800000], np.uint32).reshape(theta.shape)
    np.testing.assert_array_equal(result.view(np.uint32), expected)
    assert (result is theta) == strict
    np.testing.assert_array_equal(theta.view(np.uint32),
                                  [np.array([[0x3f800001]], np.uint32),
                                   np.array([[0xbf800001]], np.uint32)])
