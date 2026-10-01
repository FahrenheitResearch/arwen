"""Compare the one-launch sedimentation path against frozen base kernels."""
from pathlib import Path
import importlib.util

import numpy as np
import pytest

def _reference(name, attribute):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attribute)


_BASE_DENSE = _reference("test_nssl2_speed_gamma", "_BASE_DENSE")
_BASE_RAIN = _reference("test_nssl2_speed_rain", "_BASE_RAIN")
_BASE_SMALL = _reference("test_nssl2_speed_small", "_BASE_SMALL")


@pytest.mark.parametrize("nz", [2, 49, 64])
def test_fused_sediment_bits(nz):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device unavailable")
    from gpuwm.core.kernels import _preamble
    source = (Path(__file__).parents[1] / "gpuwm/core/kernels/nssl2_driver_support.cu").read_text()
    reference = cp.RawModule(code=_preamble() + _BASE_RAIN + _BASE_SMALL + _BASE_DENSE,
                             options=("-std=c++17",))
    actual = cp.RawModule(code=_preamble() + source, options=("-std=c++17",))
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
    actual.get_function("nssl2_fill_velocity_gamma")((2,), (64,), (table,))
    for dt in (15, 45):
        outputs = []
        for module in (reference, actual):
            data = cp.asarray(state.copy())
            accum = cp.full((1, 97), np.float32(0.125))
            exported = cp.empty((5, 1, 97), dtype=cp.float32)
            scalars = (np.float32(dt), np.int32(nz), np.int32(1), np.int32(97))
            if module is actual:
                module.get_function("nssl2_sediment_all_parallel_64")((97, 6), (64,),
                    (rho, temperature, data, dz, accum, exported, table, *scalars))
            else:
                for category, mass, number, volume, slot in (
                    ("rain", 2, 8, None, 0), ("ice", 3, 9, None, 1),
                    ("snow", 4, 10, None, 2), ("graupel", 5, 11, 14, 3),
                    ("hail", 6, 12, 15, 4)):
                    args = [rho, data[mass], data[number]]
                    if volume is not None:
                        args.append(data[volume])
                    module.get_function(f"nssl2_{category}_sediment_64")((2,), (64,),
                        (*args, dz, accum, exported[slot], *scalars))
                module.get_function("nssl2_cloud_sediment_64")((2,), (64,),
                    (rho, temperature, data[1], data[7], dz, accum, *scalars))
            outputs.append([x.get().view(np.uint32) for x in (data, accum, exported)])
        for expected, observed in zip(*outputs):
            np.testing.assert_array_equal(expected, observed)
