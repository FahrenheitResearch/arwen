"""CUDA vertical output words equal the Rust bridge, including tiny fields."""
import numpy as np
import pytest

from conftest import requires_gpu


def seeded_columns(depth=37, seed=281):
    rng = np.random.default_rng(seed)
    shape = (depth, 3, 11)
    source = np.broadcast_to(
        np.geomspace(100000.0, 5000.0, depth)[:, None, None], shape).copy()
    source *= 1.0 + rng.uniform(-1.e-4, 1.e-4, shape)
    source = source.astype(np.float32)
    surface = rng.uniform(97000.0, 102000.0, shape[1:]).astype(np.float32)
    target = (5000.0 + np.linspace(1.03, 0.001, 53)[:, None, None]
              * (surface[None] - 5000.0)).astype(np.float32)
    target = np.maximum(target, source[-1][None])
    field = rng.uniform(220.0, 310.0, shape).astype(np.float32)
    sfc = rng.uniform(280.0, 300.0, shape[1:]).astype(np.float32)
    return field, sfc, source, surface, target


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize('depth', [7, 37, 63, 159, 255])
@pytest.mark.parametrize('tiny', [False, True])
@pytest.mark.parametrize('order', ['descending', 'mixed'])
def test_vertical_cuda_equals_rust_words(depth, tiny, order):
    cp = pytest.importorskip('cupy')
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
    from gpuwm.ingest.vert import wrf_vert_interp_gpu
    try:
        bridge = CpuPreprocessBackend()
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f'CPU bridge unavailable: {exc}')
    args = list(seeded_columns(depth))
    if order == 'mixed':
        args[4] = args[4][np.random.default_rng(284).permutation(len(args[4]))]
    if tiny:
        rng = np.random.default_rng(282)
        for i in (0, 1):
            words = rng.integers(0, 0x02000000, args[i].shape, dtype=np.uint32)
            words |= rng.integers(0, 2, args[i].shape, dtype=np.uint32) << 31
            words.ravel()[:4] = [0, 0x80000000, 1, 0x80000001]
            args[i] = words.view(np.float32)
    for logp in (False, True):
        options = dict(interp_in_logp=logp,
                       extrap='constant' if tiny else 'temperature',
                       force_sfc_in_vinterp=1, zap_close_levels=500.0, vboundb=4)
        expected = bridge.wrf_vertical_interpolate(*args, workers=1, **options)
        actual = wrf_vert_interp_gpu(*args, **options).get()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@requires_gpu
@pytest.mark.gpu
def test_vertical_frame_matches_this_compile_platform():
    pytest.importorskip('cupy')
    from gpuwm.certify.compile_platform import compile_platform_fingerprint
    from gpuwm.core.preflight import kernel_frame_recording_for
    from gpuwm.ingest.vert import _wrf_vert_kernel
    recording = kernel_frame_recording_for(compile_platform_fingerprint())
    if recording is None or 'vert_interp' not in recording.frames:
        pytest.skip('no vertical frame recording on this compile platform')
    assert _wrf_vert_kernel(64).attributes['local_size_bytes'] == recording.frames['vert_interp']


@requires_gpu
@pytest.mark.gpu
def test_device_float64_value_cast_equals_bridge_words():
    cp = pytest.importorskip('cupy')
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
    from gpuwm.ingest.vert import wrf_vert_interp_gpu
    try:
        bridge = CpuPreprocessBackend()
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f'CPU bridge unavailable: {exc}')
    args = list(seeded_columns(7))
    rng = np.random.default_rng(287)
    for i in (0, 1):
        words = rng.integers(1, 0x00800000, args[i].shape, dtype=np.uint32)
        words |= rng.integers(0, 2, args[i].shape, dtype=np.uint32) << 31
        args[i] = words.view(np.float32).astype(np.float64)
        args[i] += rng.integers(-1, 2, args[i].shape) * (2.0 ** -150)
    for reverse in (False, True):
        device = [cp.asarray(a) for a in args]
        host = [a.copy() for a in args]
        if reverse:
            for i in (0, 1):
                device[i] = device[i][..., ::-1]
                host[i] = host[i][..., ::-1]
        expected = bridge.wrf_vertical_interpolate(*host, workers=1)
        actual = wrf_vert_interp_gpu(*device).get()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    with pytest.raises(ValueError, match='source_pressure shape does not match field'):
        wrf_vert_interp_gpu(cp.empty((0, 3, 11), dtype=cp.float64), *args[1:])


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize('vbound', [-2147483648, -2, -1, 2147483647])
def test_admitted_vbound_integer_edges_equal_bridge(vbound):
    pytest.importorskip('cupy')
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
    from gpuwm.ingest.vert import wrf_vert_interp_gpu
    try:
        bridge = CpuPreprocessBackend()
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f'CPU bridge unavailable: {exc}')
    args = seeded_columns(7, seed=288)
    for logp in (False, True):
        expected = bridge.wrf_vertical_interpolate(
            *args, workers=1, vboundb=vbound, interp_in_logp=logp)
        actual = wrf_vert_interp_gpu(*args, vboundb=vbound, interp_in_logp=logp).get()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
