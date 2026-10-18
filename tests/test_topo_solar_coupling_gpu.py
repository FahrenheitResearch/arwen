"""The terrain consumer must receive the stock WRF solar word."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np

import pytest

from conftest import requires_gpu
from gpuwm import wrf_exact


@requires_gpu
@pytest.mark.skipif(not wrf_exact.ENABLED, reason=(
    "the stock WRF word is the strict (GPUWM_WRF_EXACT=1) wrf_461 route "
    "(gpuwm.core.wrf_radiation_clock); a default process keeps the solar "
    "generation's geometry"))
def test_terrain_radiation_call_preserves_measured_coszen_word():
    import cupy as cp
    from gpuwm.core.topo_radiation import TopoShortwave

    # Stock scalar WRF step-1 dump at p20_noah_in, one observed column.
    shape = (4, 4)
    terrain = TopoShortwave.__new__(TopoShortwave)
    terrain.shape = shape
    terrain.start_time = datetime(2024, 5, 21, 21)
    terrain.shading = False
    terrain.scheme_splits_direct_beam = True
    terrain.xlat = cp.full(shape, np.float32(41.04186248779297), cp.float32)
    terrain.xlong = cp.full(shape, np.float32(-94.68722534179688), cp.float32)
    terrain._xlat_host = cp.asnumpy(terrain.xlat).ravel()
    terrain._xlong_host = cp.asnumpy(terrain.xlong).ravel()
    terrain.fields = {n: cp.zeros(shape, cp.float32) for n in (
        "topo_coszen", "hrang", "diffuse_frac", "swdown")}
    terrain.fields["swdown"].fill(np.float32(774.4592895507812))
    terrain.fields["topo_declin"] = cp.zeros(1, cp.float32)
    state = SimpleNamespace(elapsed_seconds=0.0, ht=cp.zeros(shape, cp.float32))
    cfg = SimpleNamespace(radt=1.0, radt_minutes=1.0,
                          solar_generation="wrf_461")
    result = SimpleNamespace(swddif=cp.full(shape, np.float32(53.823), cp.float32))
    terrain.after_radiation(result, state, cfg)
    words = cp.asnumpy(terrain.fields["topo_coszen"]).view(np.uint32)
    np.testing.assert_array_equal(words, np.full(shape, 0x3f427504, np.uint32))
    assert cp.asnumpy(terrain.fields["hrang"]).view(np.uint32)[0, 0] == 0x3f38cc4f
