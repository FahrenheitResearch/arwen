"""Stock driver publication retains the rounded total and its SW residual."""
import numpy as np
import pytest


@pytest.mark.parametrize("strict", [False, True])
def test_driver_publication_keeps_the_total_at_a_lost_shortwave_bit(monkeypatch, strict):
    from gpuwm.core import physics
    monkeypatch.setattr(physics,"WRF_EXACT",strict)
    lw=np.array([1.,-1.,0.],np.float32)
    sw=np.array([2.**-26,-2.**-26,2.],np.float32)
    native=sw.copy()
    total=physics._radiation_theta_total(lw,sw)
    # Independent REAL words for the stock driver's rounded total and
    # subsequent subtraction at these three representable inputs.
    expected_total=np.array([0x3f800000,0xbf800000,0x40000000],np.uint32)
    expected_sw=np.array([0,0,0x40000000],np.uint32)
    np.testing.assert_array_equal(total.view(np.uint32),expected_total)
    np.testing.assert_array_equal(sw.view(np.uint32),
                                  expected_sw if strict else native.view(np.uint32))
    np.testing.assert_array_equal((lw+sw).view(np.uint32),expected_total)
