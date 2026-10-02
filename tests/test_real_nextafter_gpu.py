"""Exact FP32 nextafter words and zero-centre REAL split regressions."""
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.kernels import module_source
from gpuwm.ingest import real as host
from gpuwm.ingest import real_device as dev

pytestmark = [requires_gpu, pytest.mark.gpu]


_PROBE = r'''
extern "C" __global__ void real_nextafter_probe(
    const unsigned int* a_words, const unsigned int* b_words,
    unsigned int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = __uint_as_float(a_words[i]);
    float b = __uint_as_float(b_words[i]);
    out[i] = __float_as_uint(real_nextafter32(a, b));
    float candidates[3] = {
        real_nextafter32(a, __uint_as_float(0xff800000u)), a,
        real_nextafter32(a, __uint_as_float(0x7f800000u))
    };
    for (int j = 0; j < 3; ++j)
        out[(j + 1) * n + i] = __float_as_uint(candidates[j]);
}
'''


_EDGE_WORDS = np.array([
    0x00000000, 0x80000000,
    0x00000001, 0x00000002, 0x00000003,
    0x80000001, 0x80000002, 0x80000003,
    0x003fffff, 0x00400000, 0x00400001,
    0x803fffff, 0x80400000, 0x80400001,
    0x007ffffe, 0x007fffff, 0x00800000, 0x00800001,
    0x807ffffe, 0x807fffff, 0x80800000, 0x80800001,
    0x3f7fffff, 0x3f800000, 0x3f800001,
    0xbf7fffff, 0xbf800000, 0xbf800001,
    0x7f7ffffe, 0x7f7fffff, 0x7f800000,
    0xff7ffffe, 0xff7fffff, 0xff800000,
    0x7f800001, 0x7fa12345, 0x7fbfffff,
    0xff800001, 0xffa12345, 0xffbfffff,
    0x7fc00000, 0x7fc01234, 0x7fffffff,
    0xffc00000, 0xffc05678, 0xffffffff,
], dtype=np.uint32)


@pytest.fixture(scope="module", params=["--ftz=false", "--ftz=true"])
def nextafter_probe(request):
    import cupy as cp

    # Compile the production source, including its exact common header.
    # CuPy can append its own FTZ option, so these are requested flags;
    # effective compiler options must be checked in the GPU run receipt.
    module = cp.RawModule(code=module_source("real_init") + _PROBE,
                          options=("-std=c++17", request.param))
    kernel = module.get_function("real_nextafter_probe")

    def run(a_words, b_words):
        a_words = np.ascontiguousarray(a_words, dtype=np.uint32).reshape(-1)
        b_words = np.ascontiguousarray(b_words, dtype=np.uint32).reshape(-1)
        assert a_words.shape == b_words.shape
        out = cp.empty((4, a_words.size), dtype=cp.uint32)
        kernel(((a_words.size + 127) // 128,), (128,), (
            cp.asarray(a_words), cp.asarray(b_words), out,
            np.int32(a_words.size)))
        return out.get()

    return run


def _numpy_nextafter_words(a_words, b_words):
    a = np.asarray(a_words, dtype=np.uint32).view(np.float32)
    b = np.asarray(b_words, dtype=np.uint32).view(np.float32)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        return np.nextafter(a, b).view(np.uint32)


def test_nextafter_boundary_matrix_words(nextafter_probe):
    # The Cartesian product includes every equal operand and signed-zero
    # pairing, plus both directions across zero, normal and infinity edges.
    a_words = np.repeat(_EDGE_WORDS, _EDGE_WORDS.size)
    b_words = np.tile(_EDGE_WORDS, _EDGE_WORDS.size)
    observed = nextafter_probe(a_words, b_words)[0]
    expected = _numpy_nextafter_words(a_words, b_words)
    np.testing.assert_array_equal(observed, expected)


def test_nextafter_random_words(nextafter_probe):
    rng = np.random.default_rng(28171)
    a_words = rng.integers(0, 1 << 32, 8192, dtype=np.uint32)
    b_words = rng.integers(0, 1 << 32, 8192, dtype=np.uint32)
    observed = nextafter_probe(a_words, b_words)[0]
    expected = _numpy_nextafter_words(a_words, b_words)
    np.testing.assert_array_equal(observed, expected)


def test_nextafter_ordered_candidate_words(nextafter_probe):
    rng = np.random.default_rng(28172)
    centres = np.concatenate((
        _EDGE_WORDS, rng.integers(0, 1 << 32, 4096, dtype=np.uint32)))
    observed = nextafter_probe(centres, centres)[1:]
    expected = np.stack((
        _numpy_nextafter_words(centres, np.full_like(centres, 0xff800000)),
        centres,
        _numpy_nextafter_words(centres, np.full_like(centres, 0x7f800000)),
    ))
    np.testing.assert_array_equal(observed, expected)
    # Both signed-zero centres retain their sign in the middle candidate.
    np.testing.assert_array_equal(observed[:, :2], np.array([
        [0x80000001, 0x80000001],
        [0x00000000, 0x80000000],
        [0x00000001, 0x00000001],
    ], dtype=np.uint32))


def test_nextafter_nan_payload_precedence(nextafter_probe):
    nan_words = np.array([
        0x7f800001, 0xff800002, 0x7fa12345, 0xffa54321,
        0x7fc01234, 0xffc05678, 0x7fffffff, 0xffffffff,
    ], dtype=np.uint32)
    values = np.concatenate((nan_words, np.array([
        0x00000000, 0x80000000, 0x00000001, 0x80000001,
        0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
    ], dtype=np.uint32)))
    a_words = np.repeat(values, values.size)
    b_words = np.tile(values, values.size)
    a_nan = (a_words & 0x7fffffff) > 0x7f800000
    b_nan = (b_words & 0x7fffffff) > 0x7f800000
    mask = a_nan | b_nan
    a_words, b_words = a_words[mask], b_words[mask]
    # NumPy nextafter quiets NaNs and retains a NaN target's sign and payload.
    expected = np.where(b_nan[mask], b_words, a_words) | np.uint32(0x00400000)
    np.testing.assert_array_equal(
        _numpy_nextafter_words(a_words, b_words), expected)
    np.testing.assert_array_equal(nextafter_probe(a_words, b_words)[0], expected)


@pytest.mark.parametrize("hypsometric_opt", [1, 2])
def test_zero_centre_split_keeps_first_equal_error_candidate(hypsometric_opt):
    coord = SimpleNamespace(
        dnw=np.array([-1.0]), rdnw=np.array([-1.0]),
        c1h=np.array([1.0]), c2h=np.array([0.0]),
        c3f=np.array([1.0, 0.0]), c4f=np.zeros(2),
        c3h=np.array([0.0]), c4h=np.array([0.0]),
    )
    mu = np.ones((2, 3), dtype=np.float64)
    target = np.array([[0.5, 1.0, 2.0], [4.0, 8.0, 16.0]], dtype=np.float32)
    if hypsometric_opt == 1:
        thickness = target.copy()
    else:
        # pfu = phm = dpf = 1, so desired = target * log1pf(1).
        ratio = host.log1pf_array(np.ones_like(target))
        thickness = np.asarray(target * np.float32(1.0) * ratio, dtype=np.float32)
    base = SimpleNamespace(
        p_top=1.0,
        phb=np.stack((np.zeros_like(mu), thickness.astype(np.float64))),
    )
    alpha = target.astype(np.float64)[None]

    # desired == dphb gives centre +0. All three candidates round to the
    # same nonzero thickness, so their diagnostic errors are exactly tied.
    candidates = np.array([0x80000001, 0, 1], dtype=np.uint32).view(np.float32)
    for candidate in candidates:
        np.testing.assert_array_equal(
            np.asarray(thickness + candidate, dtype=np.float32), thickness)
    with np.errstate(under="ignore"):
        expected = host._fp32_geopotential_split(
            base, coord, mu, alpha, hypsometric_opt)
    expected_words = np.zeros((2, *mu.shape), dtype=np.uint32)
    expected_words[1] = 0x80000001
    np.testing.assert_array_equal(expected.view(np.uint32), expected_words)
    observed = dev._fp32_geopotential_split(
        base, coord, mu, alpha, hypsometric_opt).get()
    np.testing.assert_array_equal(observed.view(np.uint32), expected_words)
