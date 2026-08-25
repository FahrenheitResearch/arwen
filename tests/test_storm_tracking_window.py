"""The MSLP reduction's optional search window must be BITWISE exact.

``mslp_hpa_from_state(state, window=...)`` exists so a consumer can ask
for sea-level pressure more than once a relocation cadence: the
full-domain reduction pulls four 3-D fields to the host and runs
DCOMPUTESEAPRS over all of them in host float64 (MEASURED 240 ms on a
378x378x49 grid, RTX 5070 Ti, of which 211 ms is host arithmetic), and
cropping to the box the caller actually searches costs 26 ms.

That is only admissible if the crop changes no number.  It does not,
and the reason is structural rather than lucky:

* ``gpuwm.verify.metrics._dcomputeseaprs`` is COLUMN-LOCAL -- every
  operation is elementwise over ``(ny, nx)`` or a ``take_along_axis``
  along k -- so a cropped reduction equals the full one on the crop;
* the nine-point smoother is not, but three passes reach exactly three
  cells, so a three-cell halo reproduces the interior;
* where the halo runs off the domain it is clipped, and there the crop's
  edge IS the domain's edge, so the same row is replicated either way.

Every one of those is a case below, at the interior, at all four edges,
at both corners, at the whole domain, and at a window one cell wide --
plus the degenerate ``phb`` column, which must not be cropped at all.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core import storm_tracking as st

NZ, NY, NX = 12, 40, 52


def _state(seed=3, phb_1d=False):
    """A hydrostatic column with structure the smoother can bite on.

    Deliberately noisy: a smooth field would agree between the cropped
    and the full reduction even if the halo were wrong.
    """
    rng = np.random.default_rng(seed)
    z_half = np.linspace(20.0, 18000.0, NZ)
    jj, ii = np.mgrid[0:NY, 0:NX]
    psfc = (101000.0
            - 4000.0 * np.exp(-(((ii - 30) / 6.0) ** 2
                                + ((jj - 20) / 6.0) ** 2))
            + rng.normal(0.0, 60.0, (NY, NX)))
    pressure = psfc[None] * np.exp(-z_half[:, None, None] / 8000.0)
    z_full = np.empty(NZ + 1)
    z_full[1:-1] = 0.5 * (z_half[:-1] + z_half[1:])
    z_full[0] = z_half[0] - (z_full[1] - z_half[0])
    z_full[-1] = z_half[-1] + (z_half[-1] - z_full[-2])
    phi = z_full[:, None, None] * st.GRAVITY_M_S2 * np.ones((NZ + 1, NY, NX))
    php = phi + rng.normal(0.0, 30.0, phi.shape)
    phb = (np.zeros(NZ + 1) if phb_1d else np.zeros_like(phi))
    state = SimpleNamespace(p=pressure, php=php, phb=phb,
                            qv=rng.uniform(0.0, 0.015, pressure.shape))
    state.total_theta = lambda: np.full_like(pressure, 300.0)
    return state


WINDOWS = {
    "interior": (slice(10, 30), slice(12, 40)),
    "touching j=0": (slice(0, 18), slice(12, 40)),
    "touching i=0": (slice(10, 30), slice(0, 18)),
    "touching j=ny": (slice(NY - 18, NY), slice(12, 40)),
    "touching i=nx": (slice(10, 30), slice(NX - 18, NX)),
    "corner 0,0": (slice(0, 7), slice(0, 7)),
    "far corner": (slice(NY - 7, NY), slice(NX - 7, NX)),
    "whole domain": (slice(0, NY), slice(0, NX)),
    "one cell": (slice(21, 22), slice(31, 32)),
    "halo-width band": (slice(3, 6), slice(3, 6)),
    "narrower than the halo": (slice(1, 3), slice(1, 3)),
}


@pytest.mark.parametrize("label", sorted(WINDOWS))
@pytest.mark.parametrize("phb_1d", [False, True], ids=["phb3d", "phb1d"])
def test_window_is_bitwise_against_the_full_domain_reduction(label, phb_1d):
    box = WINDOWS[label]
    state = _state(phb_1d=phb_1d)
    full = st.mslp_hpa_from_state(state)
    windowed = st.mslp_hpa_from_state(state, window=box)
    assert windowed.shape == full.shape
    inner_full = full[box[0], box[1]]
    inner_win = windowed[box[0], box[1]]
    # BITWISE, not close.  A tolerance here would hide exactly the halo
    # error this test exists to catch.
    assert np.array_equal(inner_full, inner_win), (
        f"{label}: max|delta| = {np.abs(inner_full - inner_win).max():.3e}")


@pytest.mark.parametrize("label", sorted(WINDOWS))
def test_outside_the_window_is_NaN_and_inside_is_finite(label):
    box = WINDOWS[label]
    windowed = st.mslp_hpa_from_state(_state(), window=box)
    assert np.isfinite(windowed[box[0], box[1]]).all()
    outside = np.ones(windowed.shape, dtype=bool)
    outside[box[0], box[1]] = False
    if outside.any():
        # NaN rather than zero or a fill: weighted_centroid and
        # locate_signal both mask on finiteness, so a cell that was
        # never reduced cannot vote -- and a caller reading outside the
        # window gets an obviously-absent value, not a quietly wrong one.
        assert np.isnan(windowed[outside]).all()


def test_no_window_is_byte_for_byte_the_shipped_path():
    """Every caller that predates the window keeps its exact answer."""
    state = _state()
    assert np.array_equal(st.mslp_hpa_from_state(state),
                          st.mslp_hpa_from_state(state, window=None))


def test_a_windowed_minimum_is_the_full_domain_minimum_on_that_window():
    """What the track writer actually asks for: the central pressure."""
    state = _state()
    box = (slice(8, 34), slice(18, 44))
    full = st.mslp_hpa_from_state(state)[box[0], box[1]]
    win = st.mslp_hpa_from_state(state, window=box)[box[0], box[1]]
    assert float(full.min()) == float(win.min())
    assert np.unravel_index(int(full.argmin()), full.shape) == \
        np.unravel_index(int(win.argmin()), win.shape)


@pytest.mark.parametrize("box,why", [
    ((slice(-4, 10), slice(0, 10)), "negative row start"),
    ((slice(0, 10), slice(-4, 10)), "negative column start"),
    ((slice(0, NY + 4), slice(0, 10)), "runs past the last row"),
    ((slice(0, 10), slice(0, NX + 4)), "runs past the last column"),
    ((slice(10, 10), slice(0, 10)), "empty in j"),
    ((slice(0, 10), slice(10, 10)), "empty in i"),
    ((slice(30, 10), slice(0, 10)), "inverted in j"),
])
def test_a_window_off_the_field_refuses_by_name(box, why):
    """Fail closed: a window that does not lie on the field it crops
    would quietly reduce a different piece of the domain than the caller
    searched, which is a wrong answer that looks right."""
    with pytest.raises(st.TrackerRefusal, match="reduction window"):
        st.mslp_hpa_from_state(_state(), window=box)


def test_plane_from_state_forwards_the_window_only_for_mslp():
    """level_hpa takes the DEVICE reduction, which a window cannot help;
    passing one must not change its answer."""
    state = _state()
    box = (slice(10, 30), slice(12, 40))
    plain = st._plane_from_state(state, "pressure", level_hpa=850.0)
    with_box = st._plane_from_state(state, "pressure", level_hpa=850.0,
                                    window=box)
    assert np.array_equal(plain, with_box)
    mslp_full = st._plane_from_state(state, "pressure")
    mslp_win = st._plane_from_state(state, "pressure", window=box)
    assert np.array_equal(mslp_full[box[0], box[1]],
                          mslp_win[box[0], box[1]])
    assert np.isnan(mslp_win[0, 0])
