"""Read an actual native ice-tendency record and the emitted forecast field."""
import os
from pathlib import Path
import numpy as np
import pytest

@pytest.mark.parametrize("step", [0, 2])
def test_published_ice_tendency_matches_native_words(step):
    history = os.environ.get("GPUWM_EXACT_HISTORY_DIR")
    native = os.environ.get("GPUWM_EXACT_NATIVE_HISTORY")
    if not (history and native):
        pytest.skip("set actual forecast history directory and native control history")
    from gpuwm.verify_exact.replay import read_frame
    from gpuwm import netcdf_bridge
    actual = read_frame(sorted(Path(history).glob("wrfout_d01_*"))[step])
    assert "RQIBLTEN" in actual, "MYNN's produced ice tendency is absent from history"
    with netcdf_bridge.open_dataset(native) as stock:
        stock.set_auto_maskandscale(False)
        variable = stock.variables["RQIBLTEN"]
        assert variable.stored_dtype == "F32"
        # The Rust bridge exposes numeric reads as float64. Recover the
        # verified stored FP32 type, exactly as the forecast reader does.
        expected = np.asarray(variable[step], dtype=np.float32)
    assert actual["RQIBLTEN"].shape == expected.shape
    assert actual["RQIBLTEN"].dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual["RQIBLTEN"].view("u4"), expected.view("u4"))

@pytest.mark.parametrize("step", [0, 2])
def test_normal_history_publishes_the_active_ice_tendency(step):
    history = os.environ.get("GPUWM_DEFAULT_HISTORY_DIR")
    if not history:
        pytest.skip("set actual default forecast history directory")
    from gpuwm.verify_exact.replay import read_frame
    actual = read_frame(sorted(Path(history).glob("wrfout_d01_*"))[step])
    assert "RQIBLTEN" in actual, "the normal forecast omits its active ice tendency"
    rate = actual["RQIBLTEN"]
    assert rate.shape == actual["QICE"].shape
    assert rate.dtype == np.float32
    assert np.isfinite(rate).all()
    if step == 2:
        assert np.any(rate != 0), "the evolving producer was replaced by an inactive zero"
