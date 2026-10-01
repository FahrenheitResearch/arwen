"""RK bookkeeping preserves every word and refreshes replaced buffers."""
from types import SimpleNamespace

import numpy as np
import pytest


def _state():
    import cupy as cp
    slots = {}

    def scratch(shape, slot, dtype):
        if slot not in slots:
            slots[slot] = cp.zeros(shape, dtype=dtype)
        return slots[slot]
    return SimpleNamespace(scratch=scratch)


@pytest.mark.gpu
def test_copy_words_and_refresh():
    import cupy as cp
    from gpuwm.core.dycore import _prepare_bookkeeping
    state = _state()
    words = np.asarray([0, 0x80000000, 1, 0x7f800000, 0x7fc12345,
                        0xffc54321, 0x3f800000], dtype=np.uint32)
    src = cp.asarray(words).view(cp.float32)
    dst = cp.empty_like(src)
    other = cp.arange(259, dtype=cp.float32)
    other_dst = cp.empty_like(other)
    pairs = ((src, dst), (other, other_dst))
    launch = _prepare_bookkeeping(state, pairs)
    launch()
    np.testing.assert_array_equal(dst.view(cp.uint32).get(), words)
    np.testing.assert_array_equal(other_dst.get(), other.get())
    assert _prepare_bookkeeping(state, pairs) is launch
    replaced = cp.empty_like(dst)
    refreshed = _prepare_bookkeeping(state, ((src, replaced), pairs[1]))
    assert refreshed is not launch
    refreshed()
    np.testing.assert_array_equal(replaced.view(cp.uint32).get(), words)


@pytest.mark.gpu
def test_zero_words_mixed_sizes():
    import cupy as cp
    from gpuwm.core.dycore import _prepare_bookkeeping
    arrays = tuple(cp.full(n, np.nan, dtype=cp.float32)
                   for n in (1, 129, 513))
    launch = _prepare_bookkeeping(_state(), tuple((a, a) for a in arrays),
                                  zero=True)
    launch()
    for a in arrays:
        np.testing.assert_array_equal(a.view(cp.uint32).get(),
                                      np.zeros(a.size, np.uint32))


@pytest.mark.gpu
def test_unsupported_layout_and_overlap_use_original_path():
    import cupy as cp
    from gpuwm.core.dycore import _prepare_bookkeeping
    state = _state()
    a = cp.arange(260, dtype=cp.float32)
    assert _prepare_bookkeeping(state, ((a[::2], cp.empty(130, cp.float32)),)) is None
    assert _prepare_bookkeeping(state, ((a[:130], a[1:131]),)) is None
    assert _prepare_bookkeeping(state, ((a[:130], a[130:]),
                                       (a[130:], a[:130]))) is None
    wide = cp.empty(2, dtype=cp.float64)
    assert _prepare_bookkeeping(state, ((wide, wide.copy()),)) is None
