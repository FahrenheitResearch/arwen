"""The NumPy builder of the native HRRR route's soil stencil: a test oracle.

This is the host code ``gpuwm.ingest.hrrr`` ran to map HRRR soil
temperature and soil moisture onto a target grid (the land-only convex
bilinear stencil with its nearest-land fallback, and its apply) until
both moved into the Rust preprocessing library
(``tools/grib1_bridge/src/masked_stencil.rs``, entries
``gpuwm_masked_bilinear_stencil_f64`` and ``gpuwm_masked_stencil_apply_f32``).
It is kept verbatim, and ONLY as the reference the native stencil is held
to byte for byte, indices, weights and report alike
(``tests/test_masked_stencil_native.py``).

Nothing outside ``gpuwm/verify`` and ``tests`` may import it: a runtime
that reached it would be a silent fallback to one core and a SciPy tree.
"""
from __future__ import annotations

import numpy as np

from gpuwm.ingest.hrrr import (
    _DISTANT_DONORS_LISTED,
    DISTANT_DONOR_CELLS,
    WINDOW_EDGES,
    SurfaceDonorSearchError,
)


def _window_reach(points_x, points_y, shape, closed_edges):
    """How far each point sees before a source cell could lie outside.

    Source cells beyond the west edge sit at ``x <= -1``, beyond the east
    edge at ``x >= nx``, and likewise in ``y``, so no cell outside the
    window is nearer to a point than this.  An edge in ``closed_edges``
    is also the edge of the whole source grid: nothing lies beyond it,
    and it does not limit the reach.
    """
    ny, nx = shape
    gaps = {
        "west": points_x + 1.0,
        "east": float(nx) - points_x,
        "south": points_y + 1.0,
        "north": float(ny) - points_y,
    }
    open_gaps = [gap for edge, gap in gaps.items() if edge not in closed_edges]
    if not open_gaps:
        return np.full(np.shape(points_x), np.inf)
    return np.min(np.stack(open_gaps), axis=0)


def _lower_corner(points, cells):
    """The lower bilinear corner of each coordinate on an axis of CELLS.

    A coordinate exactly on the last cell (``cells - 1``, two cells or
    more) is spelled as the cell before with a unit fraction, so its whole
    weight lands on its own cell and its zero-weight partner stays in the
    window; every other coordinate floors as it always has.  The Rust
    builder's ``lower_corner`` is the same rule: a target that IS the
    source grid puts its last column and row there, and the stencil of
    that grid was refused as leaving the window.
    """
    corner = np.floor(points).astype(np.int64)
    if cells >= 2:
        corner = np.where(points == float(cells - 1), corner - 1, corner)
    return corner


def _nearest_valid_cells(cells_x, cells_y, points_x, points_y):
    """The nearest valid cell to each point and its squared distance.

    ``cells_x``/``cells_y`` are the valid cells in row-major order, as
    ``np.nonzero`` returns them.  The k-d tree only proposes candidates:
    the choice is made on the same float64 squared distance the radius
    scan computes, and among equal distances the first cell in row-major
    order wins -- the lowest row, then the lowest column, which is the
    order the radius scan meets them in.  A tree's own tie order is its
    traversal order, and a donor must not depend on that.
    """
    from scipy.spatial import cKDTree

    cells = np.column_stack((cells_x, cells_y)).astype(np.float64)
    points = np.column_stack((points_x, points_y))
    tree = cKDTree(cells)
    distance, _ = tree.query(points, k=1)
    # Every cell within a hair of the tree's distance, so an exact tie in
    # the float64 distance below is never decided by the tree.
    candidates = tree.query_ball_point(
        points, distance * (1.0 + 1.0e-9) + 1.0e-9)
    chosen = np.empty(points_x.size, dtype=np.int64)
    best = np.empty(points_x.size, dtype=np.float64)
    for index, found in enumerate(candidates):
        found = np.sort(np.asarray(found, dtype=np.int64))
        distance2 = ((cells_x[found] - points_x[index]) ** 2
                     + (cells_y[found] - points_y[index]) ** 2)
        first = int(np.argmin(distance2))
        chosen[index] = found[first]
        best[index] = distance2[first]
    return chosen, best


def _build_masked_bilinear_stencil(
        x, y, source_valid, target_apply, *, fallback_radius=8,
        closed_edges=()):
    """Build a convex, surface-type-aware bilinear stencil on the CPU.

    Invalid source corners receive zero weight and the remaining weights are
    renormalized.  A target with no valid bilinear corner receives the
    NEAREST valid source cell, ties going to the lowest row and then the
    lowest column.  The search scans the ``fallback_radius`` disk first,
    exactly as it always has; a target with nothing there -- a land cell
    the source's land mask has as sea, such as a small island -- is
    searched further, and takes the nearest valid cell in the window only
    when that cell is nearer than any cell outside the window could be
    (``closed_edges`` names the window edges that are also edges of the
    whole source grid, beyond which nothing lies).  So a donor is always
    the nearest valid cell of the whole source grid, whatever the radius,
    and a target whose nearest cell this window cannot vouch for is
    refused with the radius that would decide it.

    Nothing distant is silent: every donor farther than
    :data:`DISTANT_DONOR_CELLS` is listed in the report with its target
    cell, its source cell and the distance.  Before the search went past
    the radius, a two-cell island 33 km from HRRR's nearest land refused
    a whole 750 m nest at radius 8.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    source_valid = np.asarray(source_valid, dtype=bool)
    target_apply = np.asarray(target_apply, dtype=bool)
    if x.shape != y.shape or x.shape != target_apply.shape:
        raise ValueError("masked-bilinear target arrays must have equal shapes")
    if source_valid.ndim != 2:
        raise ValueError("masked-bilinear source_valid must be 2-D")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("masked-bilinear coordinates must be finite")
    fallback_radius = int(fallback_radius)
    if fallback_radius < 0:
        raise ValueError("fallback_radius must be non-negative")
    closed_edges = tuple(sorted(set(closed_edges)))
    unknown = set(closed_edges) - set(WINDOW_EDGES)
    if unknown:
        raise ValueError(
            f"closed_edges names {sorted(unknown)}; the window edges are "
            f"{', '.join(WINDOW_EDGES)}")

    ny, nx = source_valid.shape
    x0 = _lower_corner(x, nx)
    y0 = _lower_corner(y, ny)
    x1 = x0 + 1
    y1 = y0 + 1
    if (np.min(x0) < 0 or np.max(x1) >= nx
            or np.min(y0) < 0 or np.max(y1) >= ny):
        raise ValueError("masked-bilinear coordinates leave the source window")
    fx = x - x0
    fy = y - y0
    indices_x = np.stack((x0, x1, x0, x1))
    indices_y = np.stack((y0, y0, y1, y1))
    weights = np.stack((
        (1.0 - fx) * (1.0 - fy),
        fx * (1.0 - fy),
        (1.0 - fx) * fy,
        fx * fy,
    ))
    corner_valid = source_valid[indices_y, indices_x]
    weights *= corner_valid
    raw_support = np.sum(weights, axis=0)
    direct = target_apply & (raw_support > 0.0)
    weights[:, direct] /= raw_support[direct]

    needs_fallback = target_apply & ~direct
    fallback_count = int(np.count_nonzero(needs_fallback))
    fallback_max_distance = 0.0
    fallback_distance_histogram: dict[str, int] = {}
    if fallback_count:
        flat_missing = np.flatnonzero(needs_fallback)
        missing_x = x.ravel()[flat_missing]
        missing_y = y.ravel()[flat_missing]
        center_x = np.floor(missing_x + 0.5).astype(np.int64)
        center_y = np.floor(missing_y + 0.5).astype(np.int64)
        best_distance2 = np.full(flat_missing.size, np.inf)
        donor_x = np.full(flat_missing.size, -1, dtype=np.int64)
        donor_y = np.full(flat_missing.size, -1, dtype=np.int64)
        for offset_y in range(-fallback_radius, fallback_radius + 1):
            candidate_y = center_y + offset_y
            y_inside = (candidate_y >= 0) & (candidate_y < ny)
            for offset_x in range(-fallback_radius, fallback_radius + 1):
                candidate_x = center_x + offset_x
                inside = y_inside & (candidate_x >= 0) & (candidate_x < nx)
                if not np.any(inside):
                    continue
                candidate_valid = np.zeros(flat_missing.size, dtype=bool)
                positions = np.flatnonzero(inside)
                candidate_valid[positions] = source_valid[
                    candidate_y[positions], candidate_x[positions]]
                distance2 = ((candidate_x - missing_x) ** 2
                             + (candidate_y - missing_y) ** 2)
                candidate_valid &= distance2 <= float(fallback_radius ** 2)
                improve = candidate_valid & (distance2 < best_distance2)
                donor_x[improve] = candidate_x[improve]
                donor_y[improve] = candidate_y[improve]
                best_distance2[improve] = distance2[improve]
        reach = _window_reach(missing_x, missing_y, source_valid.shape,
                              closed_edges)
        beyond = np.flatnonzero(donor_x < 0)
        required_radius = None
        worst_distance = None
        if beyond.size:
            valid_cells_y, valid_cells_x = np.nonzero(source_valid)
            unresolved_mask = np.zeros(flat_missing.size, dtype=bool)
            if valid_cells_y.size:
                # Past the radius: the nearest valid cell in the window,
                # taken only where it is nearer than anything outside.
                chosen, nearest2 = _nearest_valid_cells(
                    valid_cells_x, valid_cells_y,
                    missing_x[beyond], missing_y[beyond])
                vouched = nearest2 < reach[beyond] ** 2
                taken = beyond[vouched]
                donor_x[taken] = valid_cells_x[chosen[vouched]]
                donor_y[taken] = valid_cells_y[chosen[vouched]]
                best_distance2[taken] = nearest2[vouched]
                unresolved_mask[beyond[~vouched]] = True
                if np.any(~vouched):
                    # The smallest integer radius whose donor disk reaches
                    # a valid cell for EVERY failing point -- measured, so
                    # the refusal's remediation can be validated instead
                    # of guessed.  Raising the radius to it widens the
                    # window by as much on every side, which puts each of
                    # those cells' nearest donor inside the new reach.
                    worst_distance = float(np.sqrt(np.max(
                        nearest2[~vouched])))
                    required_radius = int(np.ceil(worst_distance))
            else:
                unresolved_mask[beyond] = True
            if np.any(unresolved_mask):
                unresolved = int(np.count_nonzero(unresolved_mask))
                unresolved_targets = tuple(zip(*np.unravel_index(
                    flat_missing[unresolved_mask], x.shape)))
                reason = (
                    "" if worst_distance is None else
                    f"; the nearest one in the decoded source window is "
                    f"up to {worst_distance:.1f} cells away, farther than "
                    "the window reaches from those points, so a nearer one "
                    "outside it cannot be ruled out")
                raise SurfaceDonorSearchError(
                    f"no valid surface-matched HRRR donor within "
                    f"{fallback_radius} cells for {unresolved} target "
                    f"point(s){reason}",
                    fallback_radius_cells=fallback_radius,
                    required_radius_cells=required_radius,
                    unresolved_targets=unresolved_targets,
                    search_inputs=(x, y, source_valid, target_apply,
                                   fallback_radius, closed_edges))
        flat_x = indices_x.reshape(4, -1)
        flat_y = indices_y.reshape(4, -1)
        flat_weights = weights.reshape(4, -1)
        flat_x[:, flat_missing] = donor_x[None, :]
        flat_y[:, flat_missing] = donor_y[None, :]
        flat_weights[:, flat_missing] = 0.0
        flat_weights[0, flat_missing] = 1.0
        fallback_distances = np.sqrt(best_distance2)
        fallback_max_distance = float(np.max(fallback_distances))
        distance_bins = np.ceil(fallback_distances).astype(np.int64)
        fallback_distance_histogram = {
            str(int(cell_radius)): int(np.count_nonzero(
                distance_bins == cell_radius))
            for cell_radius in np.unique(distance_bins)
        }
        distant = np.flatnonzero(
            fallback_distances > float(DISTANT_DONOR_CELLS))
        distant_count = int(distant.size)
        distant_rows, distant_cols = np.unravel_index(
            flat_missing[distant[:_DISTANT_DONORS_LISTED]], x.shape)
        # A window whose four edges are all HRRR's own edges has nothing
        # beyond it, so its reach is unlimited (infinity in the search
        # above).  The receipt says that as null, with closed_window_edges
        # naming why: infinity is not JSON, and the prepared cache refused
        # to write a receipt carrying it.
        distant_donors = [
            {"target_index": [int(row), int(col)],
             "source_index": [int(donor_y[k]), int(donor_x[k])],
             "distance_cells": float(fallback_distances[k]),
             "window_reach_cells": (float(reach[k])
                                    if np.isfinite(reach[k]) else None)}
            for row, col, k in zip(distant_rows, distant_cols,
                                   distant[:_DISTANT_DONORS_LISTED])]
    else:
        distant_count = 0
        distant_donors = []

    # Non-applicable targets are overwritten by the complementary stencil or
    # an explicit physical fill.  Give them a harmless unit-sum donor so no
    # NaN can be created transiently on the GPU.
    unused = ~target_apply
    weights[:, unused] = 0.0
    weights[0, unused] = 1.0
    sums = np.sum(weights, axis=0)
    selected_source_is_valid = source_valid[indices_y, indices_x]
    cross_surface = (
        (weights > 0.0) & target_apply[None, :, :]
        & ~selected_source_is_valid)
    cross_surface_count = int(np.count_nonzero(cross_surface))
    if cross_surface_count:
        raise AssertionError(
            "masked-bilinear stencil selected an incompatible surface donor")
    if (not np.isfinite(weights).all() or np.any(weights < 0.0)
            or not np.allclose(sums, 1.0, rtol=0.0, atol=2.0e-15)):
        raise AssertionError("masked-bilinear stencil is not finite and convex")
    if sum(fallback_distance_histogram.values()) != fallback_count:
        raise AssertionError("fallback distance histogram is incomplete")
    report = {
        "operator": "masked_convex_bilinear_with_nearest_valid_fallback",
        "source_valid_count": int(np.count_nonzero(source_valid)),
        "target_apply_count": int(np.count_nonzero(target_apply)),
        "direct_target_count": int(np.count_nonzero(direct)),
        "renormalized_target_count": int(np.count_nonzero(
            direct & (raw_support < 1.0 - 1.0e-12))),
        "fallback_target_count": fallback_count,
        "fallback_radius_cells": fallback_radius,
        "fallback_max_distance_cells": fallback_max_distance,
        "fallback_distance_ceiling_histogram_cells": (
            fallback_distance_histogram),
        "donor_rule": (
            "nearest surface-matched source cell, ties to the lowest row "
            "then the lowest column; past fallback_radius_cells only when "
            "nearer than any cell outside the window"),
        "closed_window_edges": list(closed_edges),
        "distant_donor_threshold_cells": DISTANT_DONOR_CELLS,
        "distant_donor_count": distant_count,
        "distant_donors": distant_donors,
        "distant_donors_not_listed": distant_count - len(distant_donors),
        "unresolved_target_count": 0,
        "cross_surface_donor_count": cross_surface_count,
        "donor_surface_class": "land",
        "minimum_nonzero_raw_support": (
            float(np.min(raw_support[direct])) if np.any(direct) else None),
        "weight_sum_minimum": float(np.min(sums)),
        "weight_sum_maximum": float(np.max(sums)),
        "negative_weight_count": int(np.count_nonzero(weights < 0.0)),
    }
    return (indices_y.astype(np.int32), indices_x.astype(np.int32),
            weights.astype(np.float32), report)


def apply_masked_bilinear_stencil(stencil, field):
    """``_CpuMaskedBilinearStencil.apply`` as it ran: four float32 terms."""
    field = np.asarray(field, dtype=np.float32)
    if field.ndim < 2 or field.shape[-2:] != stencil.source_shape:
        raise ValueError("HRRR field trailing dimensions do not match window")
    lead = (slice(None),) * (field.ndim - 2)
    expand = (None,) * (field.ndim - 2)
    result = None
    for corner in range(4):
        value = field[lead + (
            stencil.indices_y[corner], stencil.indices_x[corner])]
        term = value * stencil.weights[corner][expand]
        result = term if result is None else result + term
    return np.asarray(result, dtype=np.float32)
