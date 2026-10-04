"""Independent humidity reference for the native ensemble donor conversion."""
import numpy as np
import pytest


def test_ordinary_initializer_humidity_fp64_goldens_pin_operation_order():
    from gpuwm.core import portable_math as pm
    from gpuwm.ingest import real

    if pm.implementation() != pm.IMPLEMENTATION:
        pytest.skip("requires the existing portable native math implementation")
    # Measured through the ordinary initializer, independently of the new
    # ensemble entry point. The private Rust humidity_value test holds the
    # same FP64 words before its public ABI rounds to FP32. Reassociating
    # RH*0.01 with the precomputed saturation changes these words by 1 ULP
    # while leaving their final FP32 values equal, so the older FP32-only
    # sample would not detect that source-order drift.
    temperature = np.array([0x43390000, 0x433e0000], dtype="u4").view("f4")
    pressure = np.full(2, 10000., dtype="f4")
    rh = np.array([0x42ab6db7, 0x41dedb6e], dtype="u4").view("f4")
    mixing = real._saturation_mixing_ratio(temperature, pressure, rh)
    specific = mixing / (1. + mixing)
    np.testing.assert_array_equal(specific.view("u8"),
        np.array([0x3eb82c02ed050cea, 0x3eb2a1d20be17f2f], dtype="u8"))
