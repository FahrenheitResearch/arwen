"""Exact registered child windows, including non-ratio-aligned slab edges."""
import numpy as np
import pytest

from gpuwm.core.nest_interp import blend_terrain, register_nest, sint, window_registration


@pytest.mark.parametrize("ratio", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("stagger", ["", "x", "y"])
@pytest.mark.parametrize("ndim", [2, 3])
def test_window_sint_matches_full_registered_field(ratio, stagger, ndim):
    reg = register_nest(nri=ratio, nrj=ratio, i_parent_start=7,
                        j_parent_start=8, child_nx=17, child_ny=19,
                        parent_nx=40, parent_ny=42, stagger=stagger)
    shape = (reg.nyp, reg.nxp) if ndim == 2 else (4, reg.nyp, reg.nxp)
    source = np.random.default_rng(17).normal(size=shape).astype(np.float32)
    full = sint(source, reg)
    assembled = np.empty_like(full)
    for y0, y1 in ((0, 2), (2, 9), (9, reg.nyc)):
        for x0, x1 in ((0, 3), (3, 11), (11, reg.nxc)):
            window = (slice(y0, y1), slice(x0, x1))
            result = sint(source, reg, window=window)
            assembled[(...,) + window] = result
            cropped, donor = window_registration(reg, window)
            assert cropped.ci.min() == cropped.cj.min() == 2
            assert cropped.ci.max() == cropped.nxp - 3
            assert cropped.cj.max() == cropped.nyp - 3
            np.testing.assert_array_equal(cropped.ip, reg.ip[window[1]])
            np.testing.assert_array_equal(cropped.jp, reg.jp[window[0]])
            assert cropped._device is not reg._device and not cropped._device
    np.testing.assert_array_equal(assembled.view(np.uint32), full.view(np.uint32))


@pytest.mark.parametrize("window", [
    (slice(None), slice(0, 3)), (slice(-1, 2), slice(0, 3)),
    (slice(2, 2), slice(0, 3)), (slice(0, 100), slice(0, 3)),
    (slice(0, 3, 2), slice(0, 3)), (slice(0, 3),),
])
def test_window_refuses_ambiguous_or_invalid_slices(window):
    reg = register_nest(nri=3, nrj=3, i_parent_start=6,
                        j_parent_start=6, child_nx=17, child_ny=19,
                        parent_nx=40, parent_ny=42)
    with pytest.raises(ValueError, match="SINT window"):
        window_registration(reg, window)


@pytest.mark.parametrize("ndim", [2, 3])
def test_window_terrain_blends_at_domain_edges_only(ndim):
    shape = (29, 31) if ndim == 2 else (5, 29, 31)
    rng = np.random.default_rng(91)
    coarse = rng.normal(size=shape).astype(np.float32)
    fine = rng.normal(size=shape).astype(np.float32)
    expected = fine.copy()
    blend_terrain(coarse, expected)
    actual = fine.copy()
    for j0, j1 in ((0, 7), (7, 21), (21, 29)):
        for i0, i1 in ((0, 3), (3, 19), (19, 31)):
            sl = (..., slice(j0, j1), slice(i0, i1))
            chunk = fine[sl].copy()
            blend_terrain(coarse[sl].copy(), chunk,
                          domain_shape=(29, 31), origin=(j0, i0))
            actual[sl] = chunk
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

