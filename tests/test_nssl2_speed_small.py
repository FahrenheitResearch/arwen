"""The small sedimentation categories give the same bits at either column cap.

These tests used to compare the production kernels against a frozen copy of
the pre-speedup kernels.  The 2026-10-07 column oracle replaced the
sedimentation with a transliteration of WRF v4.6.1 (bitwise at stage 1 of
tools/nssl2_wrf461_column_oracle), so the frozen copy pinned bits WRF does
not produce and was retired.  What a speed route must still not do is change
the answer: the shallow (KMAX 64) and generic (KMAX 256) instantiations of
the same production kernel must agree bit for bit on every column both can
hold, and the oracle grades the shallow path against WRF.
"""
import numpy as np
import pytest

from nssl2_sediment_paths import device_or_skip, kernel

# The device comes through nssl2_sediment_paths, which the conftest AST marker
# cannot see; the GPU shard leg selects by this marker.
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("nz", [49, 64])
@pytest.mark.parametrize("category", ["cloud", "snow", "ice"])
def test_small_sediment_bits(nz, category):
    cp = device_or_skip()
    rng = np.random.default_rng(1804 + nz)
    shape = (nz, 1, 97)
    rho = cp.asarray(rng.uniform(0.3, 1.2, shape).astype(np.float32))
    dz = cp.asarray(rng.uniform(60, 500, shape).astype(np.float32))
    temperature = cp.asarray(rng.uniform(230, 290, shape).astype(np.float32))
    mass = rng.uniform(0, 0.0002, shape).astype(np.float32)
    mass[rng.random(shape) < 0.3] = 0
    mass[:, :, 0] = 0
    number = rng.uniform(0.01, 2e4, shape).astype(np.float32)
    for dt in (15, 45):
        outputs = []
        for suffix in ("64", "256"):
            q, n = [cp.asarray(x.copy()) for x in (mass, number)]
            accum = cp.full((1, 97), np.float32(0.125))
            exported = cp.empty_like(accum)
            args = ((rho, temperature, q, n, dz, exported) if category == "cloud"
                    else (rho, q, n, dz, accum, exported))
            kernel(f"nssl2_{category}_sediment_{suffix}")((2,), (64,),
                (*args, np.float32(dt), np.int32(nz), np.int32(1), np.int32(97)))
            outputs.append([x.get().view(np.uint32) for x in (q, n, accum, exported)])
        assert np.any(outputs[0][0] != mass.view(np.uint32))
        for expected, observed in zip(*outputs):
            np.testing.assert_array_equal(expected, observed)
