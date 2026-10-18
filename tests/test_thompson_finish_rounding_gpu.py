"""The real finish launcher keeps WRF's float32 tight-limiter word."""
from types import SimpleNamespace
import numpy as np
import pytest
from conftest import requires_gpu


@pytest.mark.gpu
@requires_gpu
def test_tight_limiter_heating_uses_the_recorded_single_precision_limit():
    import cupy as cp
    from gpuwm.core.thompson_classic import launch_classic_finish

    state = SimpleNamespace(thp=cp.zeros(2, cp.float32),
                            h_diabatic=cp.zeros(2, cp.float32))
    cfg = SimpleNamespace(mp_tend_lim=0.001, no_mp_heating=0)
    launch_classic_finish(cp.asarray([1.0, -1.0], cp.float32), state,
                          cp.zeros(2, cp.float32), cfg, 20.0, size=2)
    # WRF's float32 limit argument from the independent finish witness.
    # The old double product supplied 0x3ca3d70a instead.
    limit = np.asarray([0x3ca3d70b], np.uint32).view(np.float32)[0]
    expected = np.asarray([limit, -limit], np.float32)
    np.testing.assert_array_equal(cp.asnumpy(state.thp).view(np.uint32),
                                  expected.view(np.uint32))
    np.testing.assert_array_equal(cp.asnumpy(state.h_diabatic).view(np.uint32),
                                  (expected / np.float32(20.0)).view(np.uint32))
