"""LSMINIT branches on the stored FP32 guard word."""
import numpy as np

from gpuwm.core.noah import load_tables, pack_params, sh2o_init


def test_fp32_freezing_guard_word_copies_moisture_exactly():
    params = pack_params(load_tables(mminlu="MODIFIED_IGBP_MODIS_NOAH"))
    moisture = np.full((4, 1, 3), np.float32(0.3), dtype=np.float32)
    temperature = np.full(moisture.shape, np.float32(273.149), dtype=np.float32)
    soil = np.array([[1, 6, 14]], dtype=np.int32)
    liquid = np.asarray(sh2o_init(moisture, temperature, soil, params), np.float32)
    np.testing.assert_array_equal(liquid.view(np.uint32), moisture.view(np.uint32))
