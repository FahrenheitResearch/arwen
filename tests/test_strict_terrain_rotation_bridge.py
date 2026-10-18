"""Real Rust boundary checks for strict static rotation initialization."""
import ctypes
import os

import numpy as np
import pytest

from gpuwm.static import rust_bridge


def test_scalar_rotation_keeps_signed_zero_and_endpoint_words():
    values = np.array([[0.0, -0.0], [1.0, -1.0]], dtype=np.float32)
    try:
        result = rust_bridge.scalar_rotation_asin_f32(values)
    except (FileNotFoundError, rust_bridge.StaticBridgeError) as error:
        if os.environ.get("GPUWM_REQUIRE_TERRAIN_SCALAR") == "1":
            raise
        pytest.skip(str(error))
    assert result.shape == values.shape
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result.view(np.uint32),
                                 [[0, 0x80000000], [0x3fc90fdb, 0xbfc90fdb]])


def test_prior_real_bridge_names_the_terrain_breakage(monkeypatch):
    previous = os.environ.get("GPUWM_TERRAIN_PRIOR_BRIDGE")
    if not previous:
        pytest.skip("the previous real bridge is required")
    library = ctypes.CDLL(previous)
    assert not hasattr(library, "gpuwm_static_scalar_asin_f32")
    monkeypatch.setattr(rust_bridge, "load", lambda: library)
    with pytest.raises(rust_bridge.StaticBridgeError, match="SLP_AZI and SWNORM.*Noah"):
        rust_bridge.scalar_rotation_asin_f32(np.zeros((2, 2), np.float32))
