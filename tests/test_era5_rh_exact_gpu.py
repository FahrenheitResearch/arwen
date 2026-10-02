"""Portable binary64 RH conversion equals the host expression bits."""
import numpy as np
import pytest


@pytest.fixture
def cp():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device unavailable")
    return cp


def arguments(kind):
    from gpuwm.core import portable_math as pm
    rng = np.random.default_rng(572)
    if kind == "midpoint":
        t = rng.uniform(180, 273.15, 10001)
        eis = .01 * pm.exp(9.550426 - 5723.265 / t + 3.53068 * pm.log(t) - .00728332 * t)
        ews = 6.112 * pm.exp(17.67 * (t - 273.15) / ((t - 273.15) + 243.5))
        frac = (273.15 - t) / 20
        ratio = np.where(t > 253.15, frac * eis + (1 - frac) * ews, eis) / ews
        midpoint = (np.float64(np.float32(50)) + np.float64(np.nextafter(np.float32(50), np.float32(100)))) * .5
        return midpoint / ratio, t
    if kind == "dense":
        # Broad atmospheric envelope, including supersaturation.
        t = rng.uniform(100, 400, 262144)
        rh = rng.uniform(0, 200, t.size)
        return rh, t
    if kind == "edges":
        centers = np.array([150., 253.15, 273.15, 350.])
        t = np.concatenate([np.nextafter(centers, -np.inf), centers,
                            np.nextafter(centers, np.inf)])
        rh = np.array([0., -0., 1e-44, 1e-38, 1., 50., 100., 200.])
        tt, rr = np.meshgrid(t, rh)
        return rr, tt
    rh, t = arguments("dense")
    return rh.astype(np.float32), t.astype(np.float32)


@pytest.mark.parametrize("kind", ["midpoint", "dense", "edges", "float32"])
def test_rh_equals_host_bits(cp, kind):
    from gpuwm.ingest.horiz import _era5_rh_to_water_gpu
    from gpuwm.ingest.preprocess_backend import _era5_rh_to_water_cpu
    rh, t = arguments(kind)
    expected = _era5_rh_to_water_cpu(rh, t)
    actual = _era5_rh_to_water_gpu(cp.asarray(rh), cp.asarray(t)).get()
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_rh_shapes(cp):
    from gpuwm.ingest.horiz import _era5_rh_to_water_gpu
    assert _era5_rh_to_water_gpu(np.array(50.), np.array(253.15)).shape == ()
    assert _era5_rh_to_water_gpu(np.empty((0, 2)), np.empty((0, 2))).shape == (0, 2)
    with pytest.raises(ValueError, match="relative_humidity and temperature shapes differ"):
        _era5_rh_to_water_gpu(np.ones(2), np.ones(3))
