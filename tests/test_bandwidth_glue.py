"""Optimized glue retains the exact uint32 results of eager CuPy math."""
from types import SimpleNamespace

import numpy as np
import pytest


def _state():
    import cupy as cp
    buffers = {}

    def scratch(shape, slot):
        if slot not in buffers:
            buffers[slot] = cp.empty(shape, dtype=cp.float32)
        return buffers[slot]
    return SimpleNamespace(scratch=scratch)


def _words(rng, n):
    words = rng.integers(0, 2**32, n, dtype=np.uint32)
    specials = np.asarray([0, 0x80000000, 1, 0x80000001, 0x007fffff,
                           0x00800000, 0x7f7fffff, 0x7f800000,
                           0xff800000, 0x7fc12345, 0xffc54321,
                           0x7f812345, 0xff854321], dtype=np.uint32)
    words[:min(n, specials.size)] = specials[:min(n, specials.size)]
    return words


def _array(words, *, offset=0, shape=None):
    import cupy as cp
    backing = cp.empty(words.size + offset, dtype=cp.uint32)
    backing[offset:] = cp.asarray(words)
    array = backing[offset:].view(cp.float32)
    return array if shape is None else array.reshape(shape)


def _assert_words(actual, expected):
    import cupy as cp
    np.testing.assert_array_equal(actual.view(cp.uint32).get(),
                                  expected.view(cp.uint32).get())


@pytest.mark.gpu
@pytest.mark.parametrize('offset', [0, 1])
def test_grouped_add_words_mixed_lengths_cache_and_refresh(offset):
    from gpuwm.core.bandwidth_glue import prepare_add_arrays
    rng = np.random.default_rng(74320)
    pairs = tuple((_array(_words(rng, n), offset=offset),
                   _array(_words(rng, n), offset=offset))
                  for n in (0, 1, 3, 127, 129, 511, 513, 2051))
    expected = tuple(dst + src for src, dst in pairs)
    state = _state()
    launch = prepare_add_arrays(state, pairs)
    assert launch is not None
    assert prepare_add_arrays(state, pairs) is launch
    launch()
    for (_, actual), reference in zip(pairs, expected):
        _assert_words(actual, reference)
    replacement = pairs[-1][1].copy()
    new_pairs = pairs[:-1] + ((pairs[-1][0], replacement),)
    refreshed = prepare_add_arrays(state, new_pairs)
    assert refreshed is not launch
    new_expected = tuple(dst + src for src, dst in new_pairs)
    refreshed()
    for (_, actual), reference in zip(new_pairs, new_expected):
        _assert_words(actual, reference)


@pytest.mark.gpu
def test_grouped_add_rejects_overlap_and_strides_and_accepts_exact_alias():
    import cupy as cp
    from gpuwm.core.bandwidth_glue import prepare_add_arrays
    state = _state()
    a = cp.arange(260, dtype=cp.float32)
    assert prepare_add_arrays(state, ((a[:130], a[1:131]),)) is None
    assert prepare_add_arrays(state, ((a[:130], a[130:]),
                                      (a[130:], a[:130]))) is None
    assert prepare_add_arrays(state, ((a[::2], cp.empty(130, cp.float32)),)) is None
    assert prepare_add_arrays(state, ((a.astype(cp.float64), a),)) is None
    reference = a + a
    prepare_add_arrays(state, ((a, a),))()
    _assert_words(a, reference)


@pytest.mark.gpu
@pytest.mark.parametrize('offset', [0, 1])
def test_direct_add_exact_words_tails_and_partial_alias_fallback(offset):
    import cupy as cp
    from gpuwm.core.bandwidth_glue import add_array
    rng = np.random.default_rng(105041)
    for n in (0, 1, 3, 127, 129, 511, 513, 2051):
        src = _array(_words(rng, n), offset=offset)
        dst = _array(_words(rng, n), offset=offset)
        reference = dst + src
        assert add_array(src, dst)
        _assert_words(dst, reference)
    a = cp.arange(260, dtype=cp.float32)
    assert not add_array(a[:130], a[1:131])
    assert not add_array(a[::2], cp.empty(130, cp.float32))
    reference = a + a
    assert add_array(a, a)
    _assert_words(a, reference)


@pytest.mark.gpu
@pytest.mark.parametrize('full', [False, True])
@pytest.mark.parametrize('offset', [0, 1])
def test_total_theta_exact_words_and_profile_boundary(full, offset):
    from gpuwm.core.bandwidth_glue import total_theta
    rng = np.random.default_rng(34505)
    state = _state()
    # Odd ncol makes a float4 cross a level boundary and leaves a tail.
    shape = (19, 13, 17)
    n = np.prod(shape)
    state.thp = _array(_words(rng, n), offset=offset, shape=shape)
    bshape = shape if full else (shape[0],)
    state.thb = _array(_words(rng, np.prod(bshape)), offset=offset, shape=bshape)
    reference = ((state.thb if full else state.thb[:, None, None])
                 + state.thp)
    _assert_words(total_theta(state), reference)


@pytest.mark.gpu
def test_total_theta_preserves_strided_eager_fallback():
    import cupy as cp
    from gpuwm.core.bandwidth_glue import total_theta
    state = _state()
    state.thp = cp.arange(3 * 5 * 14, dtype=cp.float32).reshape(3, 5, 14)[..., ::2]
    state.thb = cp.arange(3, dtype=cp.float32)
    reference = state.thb[:, None, None] + state.thp
    state.total_theta = lambda: reference
    assert total_theta(state) is reference


@pytest.mark.gpu
@pytest.mark.parametrize('mapped', [False, True])
@pytest.mark.parametrize('offset', [0, 1])
def test_capture_theta_forcing_exact_words_with_special_division(mapped, offset):
    import cupy as cp
    from gpuwm.core.bandwidth_glue import capture_theta_forcing
    rng = np.random.default_rng(74416)
    shape = (19, 13, 17)
    nz, ny, nx = shape
    state = _state()
    state.rth_t = _array(_words(rng, np.prod(shape)), offset=offset, shape=shape)
    state.rthften = cp.empty_like(state.rth_t)
    for name in ('mub2d', 'mup', 'msft'):
        setattr(state, name, _array(_words(rng, ny * nx), offset=offset,
                                    shape=(ny, nx)))
    for name in ('c1h', 'c2h'):
        setattr(state, name, _array(_words(rng, nz), offset=offset))
    state.has_msf = mapped
    reference = state.rth_t / (state.c1h[:, None, None]
                               * (state.mub2d + state.mup)[None]
                               + state.c2h[:, None, None])
    if mapped:
        reference *= state.msft[None]
    assert capture_theta_forcing(state)
    _assert_words(state.rthften, reference)


@pytest.mark.gpu
def test_capture_rejects_input_overlap_and_noncontiguous_output():
    import cupy as cp
    from gpuwm.core.bandwidth_glue import capture_theta_forcing
    shape = (3, 5, 7)
    state = _state()
    state.rth_t = cp.ones(shape, cp.float32)
    state.rthften = state.rth_t
    state.mub2d = cp.ones(shape[1:], cp.float32)
    state.mup = cp.ones(shape[1:], cp.float32)
    state.msft = cp.ones(shape[1:], cp.float32)
    state.c1h = cp.ones(shape[:1], cp.float32)
    state.c2h = cp.ones(shape[:1], cp.float32)
    state.has_msf = False
    assert not capture_theta_forcing(state)
    state.rthften = cp.empty((3, 5, 14), cp.float32)[..., ::2]
    assert not capture_theta_forcing(state)


@pytest.mark.gpu
@pytest.mark.parametrize('mapped', [False, True])
def test_capture_theta_forcing_exact_physical_range(mapped):
    import cupy as cp
    from gpuwm.core.bandwidth_glue import capture_theta_forcing
    rng = np.random.default_rng(81091)
    shape = (19, 13, 17)
    nz, ny, nx = shape
    state = _state()
    state.rth_t = cp.asarray(rng.normal(0, 5000, shape), cp.float32)
    state.rthften = cp.empty_like(state.rth_t)
    state.mub2d = cp.asarray(rng.uniform(50000, 90000, (ny, nx)), cp.float32)
    state.mup = cp.asarray(rng.normal(0, 100, (ny, nx)), cp.float32)
    state.msft = cp.asarray(rng.uniform(0.7, 1.4, (ny, nx)), cp.float32)
    state.c1h = cp.asarray(rng.uniform(0, 1, nz), cp.float32)
    state.c2h = cp.asarray(rng.uniform(0, 10000, nz), cp.float32)
    state.has_msf = mapped
    reference = state.rth_t / (state.c1h[:, None, None]
                               * (state.mub2d + state.mup)[None]
                               + state.c2h[:, None, None])
    if mapped:
        reference *= state.msft[None]
    assert capture_theta_forcing(state)
    _assert_words(state.rthften, reference)
