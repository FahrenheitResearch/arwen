"""The one-launch sedimentation route gives the same bits as the per-category
kernels.

nssl2_sediment_all_parallel_64 is the route production takes for nz <= 64,
and it is the route tools/nssl2_wrf461_column_oracle grades against WRF
v4.6.1 (bitwise).  The per-category sequential kernels serve deeper columns
and the no-gamma-table fallback, so they must give the same bits on every
column both can hold.  (These tests used to compare against a frozen copy of
the pre-speedup kernels, retired when the sedimentation became a WRF
transliteration: it pinned bits WRF does not produce.)
"""
import numpy as np
import pytest

from nssl2_sediment_paths import device_or_skip, kernel

# The device comes through nssl2_sediment_paths, which the conftest AST marker
# cannot see; the GPU shard leg selects by this marker.
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("nz", [2, 49, 64])
def test_fused_sediment_bits(nz):
    cp = device_or_skip()
    rng = np.random.default_rng(1805 + nz)
    shape = (nz, 1, 97)
    rho_host = rng.uniform(0.3, 1.2, shape).astype(np.float32)
    rho = cp.asarray(rho_host)
    dz = cp.asarray(rng.uniform(60, 500, shape).astype(np.float32))
    temperature = cp.asarray(rng.uniform(230, 290, shape).astype(np.float32))
    state = rng.uniform(0.01, 2e4, (16, *shape)).astype(np.float32)
    for index in range(7):
        state[index] = rng.uniform(0, 0.006, shape).astype(np.float32)
        state[index][rng.random(shape) < 0.3] = 0
        state[index, :, :, 0] = 0
    for mass, volume in ((5, 14), (6, 15)):
        state[volume] = rho_host * state[mass] / rng.uniform(170, 900, shape).astype(np.float32)
    table = cp.empty((2, 120), dtype=cp.float64)
    kernel("nssl2_fill_velocity_gamma")((2,), (64,), (table,))
    for dt in (15, 45):
        outputs = []
        for fused in (False, True):
            data = cp.asarray(state.copy())
            accum = cp.full((1, 97), np.float32(0.125))
            exported = cp.empty((5, 1, 97), dtype=cp.float32)
            scalars = (np.float32(dt), np.int32(nz), np.int32(1), np.int32(97))
            if fused:
                kernel("nssl2_sediment_all_parallel_64")((97, 6), (64,),
                    (rho, temperature, data, dz, accum, exported, table, *scalars))
            else:
                for category, mass, number, volume, slot in (
                    ("rain", 2, 8, None, 0), ("ice", 3, 9, None, 1),
                    ("snow", 4, 10, None, 2), ("graupel", 5, 11, 14, 3),
                    ("hail", 6, 12, 15, 4)):
                    args = [rho, data[mass], data[number]]
                    if volume is not None:
                        args.append(data[volume])
                    kernel(f"nssl2_{category}_sediment_64")((2,), (64,),
                        (*args, dz, accum, exported[slot], *scalars))
                kernel("nssl2_cloud_sediment_64")((2,), (64,),
                    (rho, temperature, data[1], data[7], dz, accum, *scalars))
            outputs.append([x.get().view(np.uint32) for x in (data, accum, exported)])
        assert np.any(outputs[0][0] != state.view(np.uint32))
        for expected, observed in zip(*outputs):
            np.testing.assert_array_equal(expected, observed)
