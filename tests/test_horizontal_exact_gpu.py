"""Horizontal device arithmetic must equal the unchanged Rust bridge bits."""
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


@pytest.mark.parametrize("method", ["nearest", "bilinear", "parabolic"])
@pytest.mark.parametrize("support", [False, True])
def test_regular_equals_bridge_bits(cp, monkeypatch, method, support):
    from gpuwm.ingest import horiz
    from gpuwm.ingest.cpu_backend import shared_cpu_backend
    rng = np.random.default_rng(17329)
    y = rng.uniform(3, 8, (17, 19))
    x = rng.uniform(4, 10, y.shape)
    # Include ties, integer boundaries and coordinates that round up in FP32.
    y.flat[:8] = [3, 3.5, 4.5, 5.5, 6, 7.99999999, 4.00000001, 3.99999999]
    monkeypatch.setattr(horiz, "_regular_coordinates", lambda *a, **k: (y, x))
    gpu = horiz._RegularGpuPlan(np.arange(13), np.arange(15), y, x)
    cpu = shared_cpu_backend().indexed_plan((13, 15), y, x)
    normal = rng.normal(size=(5, 13, 15)).astype(np.float32)
    normal[0, ::2, ::2] = 0
    fields = [normal, normal * np.float32(1e-20), normal * np.float32(1e-38)]
    bits = rng.integers(0, 0x00800001, normal.shape, dtype=np.uint32)
    bits |= rng.integers(0, 2, normal.shape, dtype=np.uint32) << 31
    fields.append(bits.view(np.float32))
    fields.append(np.zeros_like(normal))
    exponents = rng.integers(-149, 65, normal.shape)
    fields.append(np.ldexp(normal.astype(np.float64), exponents).astype(np.float32))
    for field in fields:
        expected = cpu.apply(field, method, source_support=support)
        actual = gpu.apply(cp.asarray(field), method, source_support=support).get()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("method", ["nearest", "bilinear", "parabolic"])
def test_regular_edges_equal_bridge_bits(cp, monkeypatch, method):
    from gpuwm.ingest import horiz
    from gpuwm.ingest.cpu_backend import shared_cpu_backend
    y, x = np.meshgrid([0, 1e-44, 1e-38, .5, 1.5, 3.99999999, 4], [0, 1e-44, 1e-38, .5, 2.5, 5], indexing="ij")
    monkeypatch.setattr(horiz, "_regular_coordinates", lambda *a, **k: (y, x))
    gpu = horiz._RegularGpuPlan(np.arange(5), np.arange(6), y, x)
    cpu = shared_cpu_backend().indexed_plan((5, 6), y, x)
    field = np.random.default_rng(37).normal(size=(5, 6)).astype(np.float32)
    field[::2, ::2] = 0
    expected = cpu.apply(field, method)
    actual = gpu.apply(field, method).get()
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_rotation_and_cast_equal_host_bits(cp):
    from gpuwm.ingest.horiz import rotate_earth_to_grid_gpu
    from gpuwm.ingest.preprocess_backend import _rotate_earth_to_grid_cpu, CudaPreprocessBackend
    rng = np.random.default_rng(138)
    u = rng.normal(size=(3, 8, 9)).astype(np.float32)
    v = rng.normal(size=u.shape).astype(np.float32)
    sina = rng.uniform(-1, 1, (8, 9)).astype(np.float32)
    cosa = rng.uniform(-1, 1, sina.shape).astype(np.float32)
    for scale in [np.float32(1), np.float32(1e-38), np.float32(1e-44)]:
        a, b = _rotate_earth_to_grid_cpu(u * scale, v * scale, sina, cosa)
        c, d = rotate_earth_to_grid_gpu(u * scale, v * scale, sina, cosa)
        np.testing.assert_array_equal(c.get().view(np.uint32), a.view(np.uint32))
        np.testing.assert_array_equal(d.get().view(np.uint32), b.view(np.uint32))
    values = rng.normal(size=4096) * 1e-38
    expected = values.astype(np.float32)
    actual = CudaPreprocessBackend().float32(cp.asarray(values)).get()
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    from gpuwm.ingest.horiz import _divide_float32_gpu
    for values in [rng.normal(size=4096).astype(np.float32), expected]:
        expected_division = values / np.float32(9.81)
        actual_division = _divide_float32_gpu(cp.asarray(values), 9.81).get()
        np.testing.assert_array_equal(actual_division.view(np.uint32), expected_division.view(np.uint32))


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("field_shape,geometry_shape", [
    ((2, 3), (2, 1)), ((2, 3), (1, 3)), ((2, 3), ()),
    ((4, 2, 3), (1, 2, 1)), ((2, 1, 3), (4, 3)),
])
def test_rotation_broadcast_geometry_matches_host_bits(cp, inverse, field_shape, geometry_shape):
    from gpuwm.ingest import horiz, preprocess_backend as backend
    rng = np.random.default_rng(583)
    u = rng.normal(size=field_shape).astype(np.float32)
    v = rng.normal(size=field_shape).astype(np.float32)
    sine = np.asarray(rng.uniform(-1, 1, geometry_shape), dtype=np.float32)
    cosine = np.asarray(rng.uniform(-1, 1, geometry_shape), dtype=np.float32)
    device = horiz.rotate_grid_to_earth_gpu if inverse else horiz.rotate_earth_to_grid_gpu
    expected_pair = ((u * cosine - v * sine, v * cosine + u * sine) if inverse
                     else backend._rotate_earth_to_grid_cpu(u, v, sine, cosine))
    for expected, actual in zip(expected_pair, device(u, v, sine, cosine)):
        np.testing.assert_array_equal(actual.get().view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("surface", ["match", "land", "water"])
def test_masked_nearest_equals_bridge_bits(cp, surface):
    from gpuwm.ingest.horiz import masked_nearest_gpu
    from gpuwm.ingest.preprocess_backend import _masked_nearest_cpu
    rng = np.random.default_rng(661)
    lat, lon = np.arange(9, dtype=float), np.arange(11, dtype=float)
    y, x = np.meshgrid(np.linspace(0, 8, 13), np.linspace(0, 10, 17), indexing="ij")
    field = rng.normal(size=(9, 11)).astype(np.float32)
    field.flat[::7] = np.nan
    field.flat[1::7] = np.float32(1e-44)
    land = rng.random(field.shape) > .5
    target_land = rng.random(y.shape) > .5
    for radius in [0, 1, 3]:
        args = (field, lat, lon, y, x, land, target_land)
        kw = dict(surface=surface, search_radius=radius, fill_value=1e-44, strict=False)
        a = _masked_nearest_cpu(*args, **kw)
        b = masked_nearest_gpu(*args, **kw).get()
        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
    args = (np.full(field.shape, np.nan, dtype=np.float32), lat, lon, y, x, land, target_land)
    with pytest.raises(ValueError, match="no matching source surface within search_radius"):
        masked_nearest_gpu(*args, surface=surface)


def test_envelope_preserves_subnormal_extrema(cp):
    from gpuwm.ingest.horiz import parabolic_undershoot_floor
    bits = np.random.default_rng(199).integers(0, 0x00800000, 4096, dtype=np.uint32)
    for field in [bits.view(np.float32), (bits | 0x80000000).view(np.float32),
                  np.array([np.nan, 1], dtype=np.float32),
                  np.array([np.inf, 1], dtype=np.float32)]:
        assert parabolic_undershoot_floor(cp.asarray(field), _device=True) == parabolic_undershoot_floor(field)


def test_reference_envelope_does_not_select_new_kernel(cp, monkeypatch):
    from gpuwm.ingest.horiz import parabolic_undershoot_floor
    field = cp.asarray([0, 1], dtype=cp.float32)
    def refuse(*args):
        pytest.fail("reference helper selected the new CUDA reduction")
    monkeypatch.setattr("gpuwm.core.kernels.get_kernel", refuse)
    assert parabolic_undershoot_floor(field) == parabolic_undershoot_floor(field.get())
