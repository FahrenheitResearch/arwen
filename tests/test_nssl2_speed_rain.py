"""Rain sedimentation gives the same bits at either column cap.

The frozen pre-speedup rain kernel these tests used to compare against was
retired when the sedimentation became a transliteration of WRF v4.6.1
(bitwise at stage 1 of tools/nssl2_wrf461_column_oracle): it pinned bits WRF
does not produce.  The shallow (KMAX 64) and generic (KMAX 256)
instantiations of the production kernel must agree bit for bit, including
the surface xfall export the precipitation reducer sums.
"""
import numpy as np
import pytest

from nssl2_sediment_paths import device_or_skip, kernel

# The device comes through nssl2_sediment_paths, which the conftest AST marker
# cannot see; the GPU shard leg selects by this marker.
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("nz", [49, 64])
def test_rain_sediment_bits(nz):
    cp = device_or_skip()
    rng = np.random.default_rng(1803 + nz)
    shape = (nz, 1, 97)
    rho = cp.asarray(rng.uniform(0.3, 1.2, shape).astype(np.float32))
    dz = cp.asarray(rng.uniform(60, 500, shape).astype(np.float32))
    mass = rng.uniform(0, 0.006, shape).astype(np.float32)
    mass[rng.random(shape) < 0.3] = 0
    mass[:, :, 0] = 0
    number = rng.uniform(0.01, 2e4, shape).astype(np.float32)
    for dt in (15, 45):
        outputs = []
        for suffix in ("64", "256"):
            q, n = [cp.asarray(x.copy()) for x in (mass, number)]
            accum = cp.full((1, 97), np.float32(0.125))
            exported = cp.empty_like(accum)
            kernel(f"nssl2_rain_sediment_{suffix}")((2,), (64,),
                (rho, q, n, dz, accum, exported, np.float32(dt),
                 np.int32(nz), np.int32(1), np.int32(97)))
            outputs.append([x.get().view(np.uint32) for x in (q, n, accum, exported)])
        assert np.any(outputs[0][0] != mass.view(np.uint32))
        for expected, observed in zip(*outputs):
            np.testing.assert_array_equal(expected, observed)
