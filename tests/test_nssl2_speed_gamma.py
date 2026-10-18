"""The cached-gamma dense routes give the same bits as the per-call routes.

Graupel and hail fall speeds need WRF's own gamma values.  The speed route
reads them from a per-device table (nssl2_fill_velocity_gamma); the base
route evaluates them per call.  The frozen pre-speedup copy these tests used
to compare against was retired when the sedimentation became a
transliteration of WRF v4.6.1 (bitwise at stage 1 of
tools/nssl2_wrf461_column_oracle).  The routes must still agree bit for bit
on the deep columns (nz > 64) that only the sequential kernels serve.
"""
import numpy as np
import pytest

from nssl2_sediment_paths import device_or_skip, kernel

# The device comes through nssl2_sediment_paths, which the conftest AST marker
# cannot see; the GPU shard leg selects by this marker.
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("nz", [65, 128])
def test_dense_sediment_bits(nz):
    cp = device_or_skip()
    table = cp.empty((2, 120), dtype=cp.float64)
    kernel("nssl2_fill_velocity_gamma")((2,), (64,), (table,))
    rng = np.random.default_rng(1802 + nz)
    shape = (nz, 1, 97)
    rho = cp.asarray(rng.uniform(0.3, 1.2, shape).astype(np.float32))
    dz = cp.asarray(rng.uniform(60, 500, shape).astype(np.float32))
    for category in ("graupel", "hail"):
        mass = rng.uniform(0, 0.006, shape).astype(np.float32)
        mass[rng.random(shape) < 0.3] = 0
        mass[:, :, 0] = 0
        number = rng.uniform(0.01, 2e4, shape).astype(np.float32)
        density = rng.uniform(170, 900, shape).astype(np.float32)
        volume = rho.get() * mass / density
        for dt in (15, 45):
            outputs = []
            for cached in (False, True):
                q, n, v = [cp.asarray(x.copy()) for x in (mass, number, volume)]
                accum = cp.full((1, 97), np.float32(0.125))
                exported = cp.empty_like(accum)
                name = f"nssl2_{category}_sediment_{'cached_' if cached else ''}256"
                extra = (table[0 if category == "graupel" else 1],) if cached else ()
                kernel(name)((2,), (64,),
                    (rho, q, n, v, dz, accum, exported,
                     np.float32(dt), np.int32(nz), np.int32(1), np.int32(97), *extra))
                outputs.append([x.get().view(np.uint32) for x in (q, n, v, accum, exported)])
            assert np.any(outputs[0][0] != mass.view(np.uint32))
            for expected, observed in zip(*outputs):
                np.testing.assert_array_equal(expected, observed)
