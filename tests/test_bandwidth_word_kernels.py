"""Vector copies preserve words; vector SAXPY keeps scalar NVRTC arithmetic."""
from functools import lru_cache

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = pytest.mark.gpu

_SIZES = (0, 1, 3, 4, 5, 127, 128, 129, 511, 512, 513, 4099)
_SENTINEL = np.uint32(0xA5A51234)
_SPECIAL = np.asarray([
    0, 0x80000000, 1, 0x007FFFFF, 0x00800000, 0x3F800001,
    0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC12345, 0xFFC54321,
    0x7F812345, 0xFF854321,
], dtype=np.uint32)


def _words(n, seed):
    rng = np.random.default_rng(seed)
    words = rng.integers(0, 1 << 32, size=n, dtype=np.uint32)
    words[:min(n, len(_SPECIAL))] = _SPECIAL[:min(n, len(_SPECIAL))]
    return words


def _guarded(words, offset):
    import cupy as cp
    leading = 4 + offset
    storage = cp.full(leading + len(words) + 8, _SENTINEL, dtype=cp.uint32)
    view = storage[leading:leading + len(words)]
    view[...] = cp.asarray(words)
    return storage, view, leading


def _table(pairs):
    rows = np.zeros((64, 3), dtype=np.uint64)
    rows[:len(pairs)] = [(src.data.ptr, dst.data.ptr, dst.size)
                        for src, dst in pairs]
    return rows.reshape(-1).view(np.dtype(('V', rows.nbytes)))[0]


@requires_gpu
@pytest.mark.parametrize('zero', [False, True])
@pytest.mark.parametrize('offsets', [(0, 0), (1, 0), (0, 1), (2, 3), (3, 2)])
@pytest.mark.parametrize('legacy_grid', [False, True])
def test_rk_words_mixed_lengths_alignment_and_guards(zero, offsets, legacy_grid):
    """The table spans exact vector/block boundaries and every slice alignment."""
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    pairs, receipts = [], []
    for index, n in enumerate(_SIZES):
        words = _words(n, index)
        src_store, src, src_leading = _guarded(words, offsets[0])
        dst_store, dst, dst_leading = _guarded(
            np.full(n, _SENTINEL, np.uint32), offsets[1])
        pairs.append((src, dst))
        receipts.append((src_store, src_leading, dst_store, dst_leading, words))
    kernel = get_kernel('rk_bookkeeping', 'rk_zero_words' if zero else 'rk_copy_words')
    tpb = 128
    words_per_block = tpb if legacy_grid else 4 * tpb
    kernel(((max(_SIZES) + words_per_block - 1) // words_per_block, len(pairs)),
           (tpb,), (_table(pairs),))
    for src_store, src_leading, dst_store, dst_leading, words in receipts:
        src_expected = np.full(src_store.size, _SENTINEL, np.uint32)
        src_expected[src_leading:src_leading + len(words)] = words
        np.testing.assert_array_equal(src_store.get(), src_expected)
        expected = np.full(dst_store.size, _SENTINEL, np.uint32)
        expected[dst_leading:dst_leading + len(words)] = 0 if zero else words
        np.testing.assert_array_equal(dst_store.get(), expected)


@requires_gpu
def test_rk_words_full_table_large_nonmultiple_tail():
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    words = _words((1 << 20) + 3, 730)
    src = cp.asarray(words)
    # Exercise the last table row without allocating 64 full-size copies.
    small = [cp.full(i + 1, _SENTINEL, dtype=cp.uint32) for i in range(63)]
    dst = cp.empty_like(src)
    pairs = [(src[:i + 1], out) for i, out in enumerate(small)] + [(src, dst)]
    get_kernel('rk_bookkeeping', 'rk_copy_words')(
        ((src.size + 511) // 512, 64), (128,), (_table(pairs),))
    np.testing.assert_array_equal(dst.get(), words)
    for i, out in enumerate(small):
        np.testing.assert_array_equal(out.get(), words[:i + 1])
    get_kernel('rk_bookkeeping', 'rk_zero_words')(
        ((src.size + 511) // 512, 64), (128,), (_table(pairs),))
    np.testing.assert_array_equal(dst.get(), np.zeros_like(words))
    for out in small:
        np.testing.assert_array_equal(out.get(), np.zeros(out.size, np.uint32))


@lru_cache(maxsize=None)
def _scalar_saxpy(dtype):
    import cupy as cp
    # The expression and default FMA policy are the scalar kernel's original.
    source = f'''typedef {dtype} real;
    extern "C" __global__
    void scalar_saxpy(real a, const real* x, const real* y, real* out, int n) {{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) out[i] = a * x[i] + y[i];
    }}'''
    return cp.RawModule(code=source, options=('-std=c++17',)).get_function('scalar_saxpy')


@requires_gpu
@pytest.mark.parametrize('offsets', [(0, 0, 0), (1, 0, 0), (0, 1, 0),
                                    (0, 0, 1), (2, 3, 1)])
@pytest.mark.parametrize('legacy_grid', [False, True])
def test_saxpy_words_match_scalar_with_special_values_and_guards(offsets, legacy_grid):
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    vector = get_kernel('saxpy', 'saxpy')
    scalar = _scalar_saxpy('float')
    alpha_words = np.asarray([0, 0x80000000, 0x3F800001, 0x3DCCCCCD,
                              1, 0x7F800000, 0xFF800000, 0x7FC12456], np.uint32)
    for n in _SIZES:
        _, x, _ = _guarded(_words(n, n + 11), offsets[0])
        _, y, _ = _guarded(_words(n, n + 29)[::-1].copy(), offsets[1])
        storage, out, leading = _guarded(np.full(n, _SENTINEL, np.uint32), offsets[2])
        for alpha in alpha_words.view(np.float32):
            expected = cp.full(storage.size, _SENTINEL, dtype=cp.uint32)
            reference = expected[leading:leading + n]
            scalar((max(1, (n + 255) // 256),), (256,),
                   (alpha, x, y, reference, np.int32(n)))
            words_per_block = 256 if legacy_grid else 1024
            vector((max(1, (n + words_per_block - 1) // words_per_block),), (256,),
                   (alpha, x, y, out, np.int32(n)))
            np.testing.assert_array_equal(storage.get(), expected.get())


@requires_gpu
@pytest.mark.parametrize('destination', ['x', 'y'])
def test_saxpy_exact_in_place_large_tail(destination):
    import cupy as cp
    from gpuwm.core.kernels import get_kernel

    n = (1 << 20) + 3
    x = cp.asarray(_words(n, 101)).view(cp.float32)
    y = cp.asarray(_words(n, 102)).view(cp.float32)
    expected_x, expected_y = x.copy(), y.copy()
    actual = x if destination == 'x' else y
    expected = expected_x if destination == 'x' else expected_y
    alpha = np.float32(0.10000001)
    _scalar_saxpy('float')(((n + 255) // 256,), (256,),
                           (alpha, expected_x, expected_y, expected, np.int32(n)))
    get_kernel('saxpy', 'saxpy')(((n + 1023) // 1024,), (256,),
                               (alpha, x, y, actual, np.int32(n)))
    np.testing.assert_array_equal(actual.view(cp.uint32).get(), expected.view(cp.uint32).get())


@requires_gpu
def test_saxpy_preserves_real_double_specialization():
    import cupy as cp
    from gpuwm.core.kernels import module_source

    source = module_source('saxpy').replace('typedef float real;', 'typedef double real;')
    vector = cp.RawModule(code=source, options=('-std=c++17',)).get_function('saxpy')
    n = 1027
    rng = np.random.default_rng(54)
    x = cp.asarray(rng.normal(size=n))
    y = cp.asarray(rng.normal(size=n))
    actual, expected = cp.empty_like(x), cp.empty_like(x)
    args = (np.float64(0.10000000000001), x, y)
    _scalar_saxpy('double')(((n + 255) // 256,), (256,), (*args, expected, np.int32(n)))
    vector(((n + 1023) // 1024,), (256,), (*args, actual, np.int32(n)))
    np.testing.assert_array_equal(actual.view(cp.uint64).get(), expected.view(cp.uint64).get())
