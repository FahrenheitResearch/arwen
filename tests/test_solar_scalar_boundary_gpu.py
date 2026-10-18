"""Exercise the real solar caller and its existing device output boundary."""
import os
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def test_original_radconst_entry_keeps_its_two_word_output():
    from gpuwm.core.kernels import load_module
    from gpuwm.core.rrtmg_legacy import _DEGRAD, _DPD
    output = cp.asarray(np.array([0, 0, 0x7f123456], np.uint32)).view(cp.float32)
    load_module("rrtmg_legacy_solar").get_function("rls_radconst")(
        (1,), (1,), (np.float32(120.25), _DEGRAD, _DPD, output))
    assert int(cp.asnumpy(output).view(np.uint32)[2]) == 0x7f123456


def test_actual_native_solar_words_and_default_preservation():
    reference = os.environ.get("GPUWM_SOLAR_REFERENCE")
    before = os.environ.get("GPUWM_SOLAR_BEFORE_HISTORY")
    clock = os.environ.get("GPUWM_SOLAR_CLOCK")
    step = os.environ.get("GPUWM_EXACT_STEP")
    if not (reference and before and clock and step):
        pytest.skip("set an actual native recording, before history and call clock")
    from gpuwm.verify_exact.recording import load_run_record
    from gpuwm.verify_exact.replay import read_frame
    from gpuwm.core.rrtmg_legacy import solar_geometry_device
    from gpuwm.wrf_exact import ENABLED

    values = np.load(clock, allow_pickle=False)
    stock = load_run_record(Path(reference)).raw(int(step))
    cosine, _, _, _ = solar_geometry_device(
        *values["calendar"], cp.asarray(stock["XLAT"]), cp.asarray(stock["XLONG"]))
    expected = stock["COSZEN"] if ENABLED else read_frame(Path(before))["COSZEN"]
    actual = cp.asnumpy(cosine).reshape(expected.shape)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
