"""The cropped iteration must equal the whole-window one, BITWISE.

``weighted_centroid`` iterates on the disc's bounding box rather than on
the whole search window -- MEASURED 3.8x faster, 44.5 ms -> 11.6 ms per
consultation on a 378x378 plane.  That is only admissible if it changes
no number, and the first attempt at it did not: cropping the ``total``
reduction moved the answer by 1 ULP on 5 of 21 real frames, because
numpy's pairwise summation groups by element count and a cropped array
has a different count.

So the shipped version keeps ONE full-window reduction (the normaliser,
which is 2% of the cost) and crops everything else.  The crop is exact
for a reason worth stating:

* ``np.nonzero`` over a contiguous crop containing every mask cell
  yields those cells in the SAME row-major order as over the whole
  window, so adding the crop origin back -- exact integer arithmetic --
  reproduces the same index array;
* therefore the two centroid reductions see the same values, in the same
  order, at the same 1-D length, and round identically.

This file is the gate.  The reference below is the obvious whole-window
implementation, written out longhand, and every field must match it
exactly on fields shaped like the ones that motivated the estimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core.storm_tracking import (CENTROID_MAX_ITERATIONS,
                                       CENTROID_TOLERANCE_CELLS,
                                       weighted_centroid)


def _reference(plane, threshold, box, radius_cells):
    """The whole-window fixed point, longhand.  No crop anywhere."""
    import math

    j_slice, i_slice = box
    window = plane[j_slice, i_slice]
    with np.errstate(invalid="ignore"):
        qualifies = np.isfinite(window) & (window >= threshold)
    if not qualifies.any():
        return None
    jj, ii = np.ogrid[0:window.shape[0], 0:window.shape[1]]
    radius2 = float(radius_cells) ** 2

    def centroid_of(mask):
        weights = np.where(mask, window - threshold, 0.0)
        total = float(weights.sum())
        if total <= 0.0:
            weights = mask.astype(np.float64)
            total = float(mask.sum())
        wj, wi = np.nonzero(mask)
        w = weights[wj, wi]
        return (float((wj * w).sum() / total),
                float((wi * w).sum() / total))

    def settle(seed):
        cj, ci = seed
        mask = qualifies
        converged, iterations = False, 0
        for iterations in range(1, CENTROID_MAX_ITERATIONS + 1):
            step = qualifies & (((jj - cj) ** 2 + (ii - ci) ** 2) <= radius2)
            if not step.any():
                break
            mask = step
            new_cj, new_ci = centroid_of(mask)
            moved = math.hypot(new_cj - cj, new_ci - ci)
            cj, ci = new_cj, new_ci
            if moved < CENTROID_TOLERANCE_CELLS:
                converged = True
                break
        weight = float(np.where(mask, window - threshold, 0.0).sum())
        return (cj, ci), mask, converged, iterations, weight

    extremum = np.unravel_index(
        int(np.nanargmax(np.where(qualifies, window, -np.inf))),
        window.shape)
    seeds = [(float(extremum[0]), float(extremum[1])),
             centroid_of(qualifies)]
    best = None
    for seed in seeds:
        cand = settle(seed)
        if best is None or cand[4] > best[4]:
            best = cand
    (cj, ci), mask, converged, iterations, _ = best
    return {"ci": ci + i_slice.start, "cj": cj + j_slice.start,
            "cells": int(mask.sum()),
            "max_value": float(window[mask].max()),
            "iterations": int(iterations), "converged": bool(converged)}


def _field(ny, nx, seed, vortices, slope, noise):
    """A vortex field with the ingredients that break naive estimators:
    an environmental gradient, and grid-scale noise."""
    rng = np.random.default_rng(seed)
    j, i = np.mgrid[0:ny, 0:nx]
    plane = np.full((ny, nx), 1500.0) + slope * i + 0.6 * slope * j
    for cj, ci, depth, width in vortices:
        plane = plane - depth * np.exp(
            -((j - cj) ** 2 + (i - ci) ** 2) / (2.0 * width ** 2))
    return plane + rng.normal(0.0, noise, (ny, nx))


CASES = [
    ("one vortex, gradient", 200, 200, [(100.0, 100.0, 40.0, 25.0)],
     0.10, 0.0),
    ("off-centre", 200, 200, [(60.0, 140.0, 40.0, 25.0)], 0.10, 0.0),
    ("noisy", 200, 200, [(100.0, 100.0, 40.0, 25.0)], 0.10, 0.8),
    ("two centres", 200, 200,
     [(100.0, 70.0, 40.0, 18.0), (100.0, 130.0, 38.0, 18.0)], 0.05, 0.3),
    ("near a corner", 160, 160, [(25.0, 25.0, 40.0, 20.0)], 0.08, 0.2),
    ("shallow, wide", 180, 220, [(90.0, 110.0, 12.0, 45.0)], 0.04, 0.4),
    ("non-square", 140, 260, [(70.0, 180.0, 30.0, 22.0)], 0.07, 0.2),
]


@pytest.mark.parametrize("label,ny,nx,vort,slope,noise", CASES,
                         ids=[c[0] for c in CASES])
@pytest.mark.parametrize("radius", [12.0, 30.0, 77.777, 150.0])
def test_the_cropped_iteration_is_bitwise_the_whole_window_one(
        label, ny, nx, vort, slope, noise, radius):
    plane = _field(ny, nx, hash(label) & 0xFFFF, vort, slope, noise)
    box = (slice(0, ny), slice(0, nx))
    ceiling = float(np.nanmin(plane)) + 30.0
    got = weighted_centroid(-plane, -ceiling, box, radius)
    ref = _reference(-plane, -ceiling, box, radius)
    assert got is not None and ref is not None
    for key in ("ci", "cj", "cells", "max_value", "iterations", "converged"):
        assert got[key] == ref[key], (
            f"{key}: {got[key]!r} != {ref[key]!r} "
            f"(delta {got[key] - ref[key]:.3e})"
            if isinstance(got[key], float) else f"{key} differs")


@pytest.mark.parametrize("offset", [(0, 0), (17, 23), (40, 5)])
def test_a_sub_box_of_the_plane_is_bitwise_too(offset):
    """The search box is rarely the whole plane -- the tracker searches
    the footprint plus a margin -- so the crop must compose with it."""
    plane = _field(240, 240, 7, [(120.0, 120.0, 40.0, 25.0)], 0.09, 0.3)
    dj, di = offset
    box = (slice(dj, dj + 150), slice(di, di + 150))
    ceiling = float(np.nanmin(plane)) + 30.0
    got = weighted_centroid(-plane, -ceiling, box, 40.0)
    ref = _reference(-plane, -ceiling, box, 40.0)
    for key in ("ci", "cj", "cells", "max_value", "iterations", "converged"):
        assert got[key] == ref[key], key


def test_the_degenerate_uniform_field_takes_the_same_fallback():
    """total <= 0 swaps the weights for the mask itself; the cropped form
    has to take that branch identically."""
    plane = np.full((80, 80), 1500.0)
    box = (slice(0, 80), slice(0, 80))
    got = weighted_centroid(-plane, -1500.0, box, 20.0)
    ref = _reference(-plane, -1500.0, box, 20.0)
    for key in ("ci", "cj", "cells", "iterations", "converged"):
        assert got[key] == ref[key], key
