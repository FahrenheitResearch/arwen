"""Grade emitted radiation fields against a separately identified native history."""
import os
from pathlib import Path
import numpy as np
import pytest

@pytest.mark.parametrize("field", ["GLW", "SWDOWN", "RTHRATEN"])
def test_emitted_radiation_matches_native_sizing_policy(field):
    history = os.environ.get("GPUWM_EXACT_HISTORY")
    native = os.environ.get("GPUWM_EXACT_NATIVE_HISTORY")
    if not (history and native):
        pytest.skip("set real emitted history and native history")
    from gpuwm import netcdf_bridge
    from gpuwm.verify_exact.replay import read_frame, _STORED_DTYPE
    actual = read_frame(Path(history))
    step = int(os.environ.get("GPUWM_EXACT_STEP", "5"))
    with netcdf_bridge.open_dataset(native) as stock:
        stock.set_auto_maskandscale(False)
        variable = stock.variables[field]
        expected = np.asarray(variable[step], dtype=_STORED_DTYPE[variable.stored_dtype])
    assert actual[field].shape == expected.shape
    assert actual[field].dtype == expected.dtype
    np.testing.assert_array_equal(actual[field].view("u4"), expected.view("u4"))
