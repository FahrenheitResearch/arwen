"""Score a cycled run's free-forecast legs with the gallery's own FSS.

**This is the scorer of record.**  It reproduces the published gallery
numbers bit for bit, and the HRRR comparison, the verification ladder
and the sweep arms are all keyed to it.  Any other scorer in this tree
is a *caller*: it imports these constants and this neighborhood
derivation rather than restating them, and adds only what it reports.
``tools/ens_sweep/score_free_forecast.py`` is the one such caller --
it exists for the per-member distribution that an ensemble-size axis
needs and that this module does not emit.

That rule is not bookkeeping.  The two scorers once carried their own
copies of the same four constants, independently written against the
same published anchor.  They agreed at 3 km and only at 3 km: a
hard-coded half-width of 4 cells is the 27 km box at that spacing and a
13.5 km box at 1.5 km, so the copy silently meant a different metric
under the same name.  :func:`half_width_cells` exists so the derivation
is written down once.

The rolling verifier in ``tools/da_nowcast.py`` grades a *front-door case
directory*.  A sweep arm that drives ``tools/da_cycle_prepared.py``
straight at an already-prepared case has no such directory -- it has a
composites folder and a pile of observation files -- and yet its numbers
have to land on the same axis as the ones already published, or the sweep
answers nothing.

So this module does not re-implement the metric.  It imports the very
constants and the very function the renderer calls
(:func:`gpuwm.verify.field_metrics.fss_distance`, ``FSS_BOX_KM``,
``FSS_THRESHOLD_DBZ``, ``MISSING_OBS_FILL_DBZ``,
``COLUMN_THRESHOLD_DBZ``) and reproduces
``NowcastRender.verify_numbers`` step for step.  If the renderer's
constants ever move, this follows them; if the renderer cannot be
imported at all, the fallback literals are used AND the fact is recorded
in ``constants_source``, so a reader can always tell which happened.

The neighborhood convention is the renderer's and is stated here because
it is the single number a comparison against published skill turns on:
``FSS_BOX_KM`` is the *side length of a square box*, and the half-width
handed to the boxcar is ``round(FSS_BOX_KM / 2 / dx_km)`` cells, so the
scored neighborhood is ``2 * half_width + 1`` cells ACROSS.  At 3 km that
is a 9-cell, 27 km-wide box.  It is not a radius.

**Two things a single flattering number hides, both reported here.**

*One scale is not a result.*  FSS rises monotonically with neighborhood
size and reaches 1 when the box covers the domain, so a single box is a
point on a curve chosen in advance.  ``--neighborhood-km`` repeats and
produces ``neighborhood_curve``; the published box is always scored
whatever else is asked for, and its numbers keep the flat key names the
existing receipts use, so nothing that reads this file today changes.

*The scored field is the ensemble MEAN.*  Averaging ten members'
column-max reflectivity smooths the field before the metric's own boxcar
smooths it again, which flatters the score relative to any member the
model could actually produce.  ``per_member`` scores each member's own
composite at the published box and reports the spread, so the mean's
number is never read without the distribution it came from.  The two
answer different questions and neither replaces the other.

Structure, beside skill
-----------------------

FSS answers "is the area in about the right place".  It does not answer
"does the field look like weather", and a neighborhood score can be *won*
by a field that has diffused its storms into one smooth patch of
above-threshold area.  Three diagnostics are therefore computed on every
scored frame and written beside FSS.  None of them replaces it and none of
them is subtracted from it:

* **object statistics** at :data:`OBJECT_THRESHOLD_DBZ` -- connected-
  component count, the area distribution, and the mean nearest-neighbour
  centroid separation, so "N objects against M observed" is a number
  rather than an impression.  This is the MODE family's first stage
  (Davis, Brown and Bullock 2006, *Mon. Wea. Rev.* **134**, 1772-1784 and
  1785-1795; Bullock, Brown and Fowler 2016, NCAR MET technical note), cut
  down to threshold-and-label: no convolution radius, no merging, no
  fuzzy-logic matching.  The separation statistic is the classical
  nearest-neighbour distance of Clark and Evans (1954, *Ecology* **35**,
  445-453).
* **radial power spectrum**, from the repo's own pinned implementation
  (:mod:`gpuwm.verify.spectral`): detrend, Hann window, FFT periodogram,
  radial average.  Detrending before an FFT on a limited-area field is
  Errico (1985, *Mon. Wea. Rev.* **113**, 1554-1562); the aperiodicity it
  treats, and the DCT alternative, are Denis, Cote and Laprise (2002,
  *Mon. Wea. Rev.* **130**, 1812-1829).  The band ratios and the
  effective-resolution wavelength read as in Skamarock (2004, *Mon. Wea.
  Rev.* **132**, 3019-3032): the scale at which a forecast's variance
  falls away from the reference's is that forecast's real resolution, and
  a field that has diffused its small scales says so here immediately.
* **intensity distribution** -- P50/P90/P99 of in-echo dBZ and the
  histogram-overlap skill score of Perkins et al. (2007, *J. Climate*
  **20**, 4356-4376), which catches "right coverage, wrong texture".

ONE SCORING RULE COMES WITH THEM.  An ensemble mean is smooth *because it
is an average*, not because its members are (Surcel, Zawadzki and Yau,
2014, "On the filtering properties of ensemble averaging for storm-scale
precipitation forecasts", *Mon. Wea. Rev.*), so structure metrics computed
on the mean measure the averaging operator and not the model.  Every
structure block therefore reports the individual members -- with the
min/median/max across them -- and the mean separately, each labelled.
Reading the mean's object count as the model's object count is the error
this labelling exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from gpuwm.verify import spectral
from gpuwm.verify.field_metrics import fss_distance

#: Fallback values, used only when the renderer will not import.  They are
#: duplicated deliberately rather than defaulted silently: a sweep that
#: scored itself against different constants than the gallery would be
#: worse than one that refused to score at all, so the choice is recorded.
_FALLBACK = {"FSS_BOX_KM": 27.0,
             "FSS_THRESHOLD_DBZ": 30.0,
             "MISSING_OBS_FILL_DBZ": -35.0,
             "COLUMN_THRESHOLD_DBZ": 35.0}

LEG_NAME = re.compile(r"^leg(\d+)_(.+)\.npz$")
# A nest composite lands in the SAME composites/ directory as the parent
# members, named leg{NN}_{name}_d{GG}.npz by
# tools/da_nowcast_render.py:nest_composite_path.  It is a fine-grid view
# OF one member, not an extra member.  Without this filter "leg00_3_d02"
# parses out as a member named "3_d02" and is averaged into the ensemble
# mean this file scores -- every FSS number moves, and nothing raises.
# The committed live-fire-3 composites predate the nest, so the scorer's
# own regression fixtures would not have caught it either.
NEST_SUFFIX = re.compile(r"_d\d+$")


def metric_constants() -> tuple[dict, str]:
    """The renderer's constants, or the recorded fallback."""

    try:
        from tools import da_nowcast_render as render
        return ({k: getattr(render, k) for k in _FALLBACK},
                "tools.da_nowcast_render")
    except Exception as error:            # pragma: no cover - import guard
        return dict(_FALLBACK), f"fallback literals ({error.__class__.__name__}: {error})"


def _dirs(composites) -> list[Path]:
    """Accept one directory or several, uniformly.

    A cycled run that carries its ensemble across process boundaries --
    which is how a cadence that follows the radar is driven -- writes one
    composites directory per process.  The legs are globally numbered by
    ``--leg-number-offset``, so the union of those directories is exactly
    the one directory a single-process run would have written.
    """

    return [Path(c) for c in (composites if isinstance(composites, (list,
                                                                    tuple))
                              else [composites])]


def load_composite(composites, leg: int, name: str) -> np.ndarray:
    wanted = f"leg{leg:02d}_{name}.npz"
    for directory in _dirs(composites):
        path = directory / wanted
        if path.is_file():
            with np.load(path) as handle:
                return np.asarray(handle["refl_colmax"], float)
    raise SystemExit(
        f"no {wanted} in " + ", ".join(str(d) for d in _dirs(composites)))


def member_names(composites, leg: int) -> list[str]:
    """Every member composite present for ``leg``, control excluded.

    A leg appearing in two directories is a duplicate, not two members:
    the name is what identifies a trajectory, so the set is taken by name.
    """

    names: set[str] = set()
    for directory in _dirs(composites):
        for path in directory.glob(f"leg{leg:02d}_*.npz"):
            match = LEG_NAME.match(path.name)
            if (match and match.group(2) != "control"
                    and not NEST_SUFFIX.search(match.group(2))):
                names.add(match.group(2))
    names = list(names)
    # Numeric member ids sort numerically; anything else sorts as text, so
    # a mixed set still has a deterministic order.
    return sorted(names, key=lambda n: (not n.isdigit(), int(n) if n.isdigit() else n))


def _half_width_for_box(box_km: float, dx_km: float) -> int:
    """The renderer's own conversion from a box SIDE to a boxcar half-width.

    Takes the box explicitly so the neighborhood curve can score boxes
    other than the published one.  :func:`half_width_cells` is the
    published-box spelling and the one other scorers import.
    """

    return max(1, round(float(box_km) / 2.0 / float(dx_km)))


def half_width_cells(dx_km: float, const: dict) -> int:
    """Neighborhood half-width in cells for a grid of spacing ``dx_km``.

    The box is a fixed distance -- ``FSS_BOX_KM`` across -- so the cell
    count has to be derived from the spacing, never written down.  A
    scorer that hard-codes the 3 km answer (4) silently means a 13.5 km
    box at 1.5 km and a 54 km box at 6 km, which is a different metric
    reported under the same name.  Any second scorer calls this rather
    than restating the constant.
    """

    return _half_width_for_box(const["FSS_BOX_KM"], dx_km)


def _fss(field, truth, *, threshold, half_width) -> float:
    return round(1.0 - fss_distance(field, truth, threshold=threshold,
                                    half_width=half_width), 4)


# --------------------------------------------------------------------------
# structure diagnostics
# --------------------------------------------------------------------------

#: The object threshold.  It is the same 35 dBZ the column counts beside FSS
#: already use, named once here so the two can never drift apart.
OBJECT_THRESHOLD_DBZ = 35.0

#: Objects smaller than this are dropped before anything is counted.  Without
#: a floor the count is dominated by single-cell specks, and a noisier field
#: would score as a better-structured one.  Four cells is 36 km2 at 3 km.
OBJECT_MIN_AREA_CELLS = 4

#: Moore neighbourhood.  Diagonal touching joins one object rather than two,
#: which is the convention that keeps a thin diagonal squall line from being
#: counted as a string of separate cells.
OBJECT_CONNECTIVITY = 8

#: Below this the field is treated as no-echo.  It is the renderer's own plot
#: mask, and it exists here for a second reason: sources disagree about what
#: value means "nothing" (our composites floor at -35 dBZ, a regridded
#: external field may floor at -10), and an un-clipped comparison would read
#: that bookkeeping difference as a difference in variance.
ECHO_FLOOR_DBZ = 5.0

#: Spectral bands, by wavelength in grid intervals.  ``2-4dx`` is the band a
#: diffused field empties first; it is also the band no model actually
#: resolves, which is the point -- the question is whether OUR loss there is
#: worse than the reference's.
SPECTRAL_BANDS: tuple[tuple[str, float, float], ...] = (
    ("2_4dx", 2.0, 4.0),
    ("4_10dx", 4.0, 10.0),
    ("ge_10dx", 10.0, math.inf),
)

#: A forecast is said to still resolve a scale while it holds at least this
#: fraction of the reference's variance there.
SPECTRAL_EFFECTIVE_RATIO = 0.5

#: Histogram bins for the distribution overlap, in dBZ.
HISTOGRAM_EDGES_DBZ = tuple(float(v) for v in range(5, 81, 5))

#: Reported in-echo quantiles.
DISTRIBUTION_QUANTILES = (50.0, 90.0, 99.0)

#: Which member statistics get a min/median/max across the ensemble.
MEMBER_SPREAD_KEYS = (
    ("objects", "count"),
    ("objects", "median_object_area_cells"),
    ("objects", "max_object_area_cells"),
    ("objects", "largest_object_area_fraction"),
    ("objects", "mean_nearest_neighbor_km"),
    ("spectrum", "variance_fraction_2_4dx"),
    ("spectrum", "power_ratio_2_4dx"),
    ("spectrum", "effective_resolution_dx"),
    ("distribution", "p99_dbz"),
    ("distribution", "histogram_overlap"),
)

STRUCTURE_CITATIONS = {
    "objects": "MODE family, threshold-and-label stage only: Davis, Brown "
               "and Bullock (2006), Mon. Wea. Rev. 134, 1772-1784 (Part I) "
               "and 1785-1795 (Part II); Bullock, Brown and Fowler (2016), "
               "NCAR MET technical note.  No convolution radius, no merging "
               "and no fuzzy-logic matching are applied here",
    "nearest_neighbor": "Clark and Evans (1954), Ecology 35, 445-453",
    "spectrum": "detrend-then-FFT on a limited-area field: Errico (1985), "
                "Mon. Wea. Rev. 113, 1554-1562; aperiodicity and the DCT "
                "alternative: Denis, Cote and Laprise (2002), Mon. Wea. "
                "Rev. 130, 1812-1829; effective resolution read from the "
                "departure of a spectrum from a reference: Skamarock "
                "(2004), Mon. Wea. Rev. 132, 3019-3032.  Implementation is "
                "gpuwm.verify.spectral, whose pins are hashed",
    "distribution": "histogram overlap skill score: Perkins, Pitman, "
                    "Holbrook and McAneney (2007), J. Climate 20, "
                    "4356-4376",
    "ensemble_mean_caveat":
        "ensemble averaging is a spatial filter: Surcel, Zawadzki and Yau "
        "(2014), 'On the filtering properties of ensemble averaging for "
        "storm-scale precipitation forecasts', Mon. Wea. Rev.",
}

#: Said on every figure and in every receipt that carries a mean's numbers.
ENSEMBLE_MEAN_STRUCTURE_WARNING = (
    "structure metrics on an ensemble mean measure the AVERAGING, not the "
    "model: the mean of N members is smooth even when every member is not.  "
    "Read the member block for the model's structure and this block only for "
    "what the published mean field looks like")


def label_objects(mask: np.ndarray, *,
                  connectivity: int = OBJECT_CONNECTIVITY) -> np.ndarray:
    """Connected-component labels of a boolean mask, 1-based, 0 for off.

    A two-pass union-find over the set cells only, so the cost follows the
    object area rather than the domain area.  Written out rather than taken
    from ``scipy.ndimage.label`` because scipy is not a declared dependency
    of this package and a verification number must not depend on whether an
    optional import happened to be present.
    """

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("objects are labelled on a 2-D mask")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, not {connectivity!r}")

    ny, nx = values.shape
    labels = np.zeros((ny, nx), dtype=np.int64)
    parent = [0]

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for j, i in np.argwhere(values):
        j, i = int(j), int(i)
        seen = []
        if i > 0 and labels[j, i - 1]:
            seen.append(int(labels[j, i - 1]))
        if j > 0:
            if labels[j - 1, i]:
                seen.append(int(labels[j - 1, i]))
            if connectivity == 8:
                if i > 0 and labels[j - 1, i - 1]:
                    seen.append(int(labels[j - 1, i - 1]))
                if i + 1 < nx and labels[j - 1, i + 1]:
                    seen.append(int(labels[j - 1, i + 1]))
        if not seen:
            parent.append(len(parent))
            labels[j, i] = len(parent) - 1
            continue
        lowest = min(seen)
        labels[j, i] = lowest
        for other in seen:
            union(lowest, other)

    if len(parent) == 1:
        return labels
    roots = np.array([find(node) for node in range(len(parent))],
                     dtype=np.int64)
    unique, compact = np.unique(roots[1:], return_inverse=True)
    remap = np.zeros(len(parent), dtype=np.int64)
    remap[1:] = compact + 1
    return remap[labels]


def object_statistics(field: np.ndarray, *, dx_km: float,
                      threshold: float = OBJECT_THRESHOLD_DBZ,
                      min_area_cells: int = OBJECT_MIN_AREA_CELLS) -> dict:
    """Count, area distribution and separation of the above-threshold objects.

    ``largest_object_area_fraction`` is the single number that separates "one
    blob" from "several cells": it is 1.0 when all the above-threshold area
    belongs to one component.
    """

    values = np.asarray(field, dtype=float)
    labels = label_objects(values >= threshold)
    count = int(labels.max())
    areas = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    keep = np.nonzero(areas >= min_area_cells)[0] + 1
    cell_km2 = dx_km * dx_km

    out = {
        "threshold_dbz": threshold,
        "min_area_cells": min_area_cells,
        "connectivity": OBJECT_CONNECTIVITY,
        "count": int(keep.size),
        "count_before_area_filter": count,
        "total_area_cells": int(areas[keep - 1].sum()) if keep.size else 0,
        "total_area_km2": round(float(areas[keep - 1].sum() * cell_km2), 2)
                          if keep.size else 0.0,
        "median_object_area_cells": None,
        "max_object_area_cells": None,
        "median_object_area_km2": None,
        "max_object_area_km2": None,
        "largest_object_area_fraction": None,
        "mean_nearest_neighbor_km": None,
        "median_nearest_neighbor_km": None,
    }
    if not keep.size:
        return out

    kept_areas = areas[keep - 1].astype(float)
    out["median_object_area_cells"] = round(float(np.median(kept_areas)), 2)
    out["max_object_area_cells"] = int(kept_areas.max())
    out["max_object_area_km2"] = round(float(kept_areas.max() * cell_km2), 2)
    out["median_object_area_km2"] = round(
        float(np.median(kept_areas) * cell_km2), 2)
    out["largest_object_area_fraction"] = round(
        float(kept_areas.max() / kept_areas.sum()), 4)

    # Centroids on the projection plane; index distance times dx is the
    # plane distance, which is what the grid is defined in.
    rows, cols = np.nonzero(labels)
    which = labels[rows, cols]
    centroids = []
    for label in keep:
        pick = which == label
        centroids.append((rows[pick].mean() * dx_km, cols[pick].mean() * dx_km))
    if len(centroids) >= 2:
        points = np.asarray(centroids, dtype=float)
        separation = np.hypot(points[:, None, 0] - points[None, :, 0],
                              points[:, None, 1] - points[None, :, 1])
        np.fill_diagonal(separation, np.inf)
        nearest = separation.min(axis=1)
        out["mean_nearest_neighbor_km"] = round(float(nearest.mean()), 3)
        out["median_nearest_neighbor_km"] = round(float(np.median(nearest)), 3)
    return out


def spectral_profile(field: np.ndarray, *, dx_km: float,
                     floor_dbz: float = ECHO_FLOOR_DBZ) -> dict:
    """Radial PSD of one field, on the pinned implementation.

    The field is clipped from below at ``floor_dbz`` first so that two
    sources' different no-echo encodings cannot enter the variance.
    """

    plane = np.maximum(np.asarray(field, dtype=float), floor_dbz)
    psd = spectral.radial_psd(plane, dx_km * 1000.0)
    wavenumber = np.asarray(psd["wavenumber_cycles_per_m"], dtype=float)
    counts = np.asarray(psd["mode_count"], dtype=float)
    power = np.asarray(psd["power"], dtype=float)
    # Variance carried by a radial bin is its mean power times the number of
    # Fourier modes it holds, so bands are compared by variance and not by a
    # mean that ignores how many modes a band contains.
    return {
        "wavelength_dx": 1.0 / (wavenumber * dx_km * 1000.0),
        "power": power,
        "variance": power * counts,
        "dx_km": dx_km,
    }


def spectral_report(profile: dict, reference: dict | None = None) -> dict:
    """JSON-safe band statistics, and the ratios against a reference."""

    wavelength = profile["wavelength_dx"]
    variance = profile["variance"]
    total = float(variance.sum())
    out: dict = {"floor_dbz": ECHO_FLOOR_DBZ,
                 "retained_bins": int(variance.size),
                 "band_wavelengths_dx": {
                     name: ([lo, None] if math.isinf(hi) else [lo, hi])
                     for name, lo, hi in SPECTRAL_BANDS}}
    for name, low, high in SPECTRAL_BANDS:
        band = (wavelength >= low) & (wavelength < high)
        out[f"variance_fraction_{name}"] = (
            round(float(variance[band].sum() / total), 5) if total > 0 else None)
    out["effective_resolution_dx"] = None
    out["effective_resolution_km"] = None
    if reference is None:
        return out

    reference_variance = reference["variance"]
    if reference_variance.shape != variance.shape:
        raise ValueError("spectral profiles compared on different grids")
    for name, low, high in SPECTRAL_BANDS:
        band = (wavelength >= low) & (wavelength < high)
        denominator = float(reference_variance[band].sum())
        out[f"power_ratio_{name}"] = (
            round(float(variance[band].sum() / denominator), 4)
            if denominator > 0 else None)

    # Effective resolution: walk from the longest wavelength down and stop at
    # the first bin holding less than SPECTRAL_EFFECTIVE_RATIO of the
    # reference's variance.  The last bin before that is the shortest scale
    # the field still carries.  Bins arrive in ascending wavenumber, which is
    # descending wavelength, so the walk is simply in array order.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(reference["variance"] > 0.0,
                         profile["variance"] / reference["variance"], np.nan)
    resolved = np.nonzero(~(ratio >= SPECTRAL_EFFECTIVE_RATIO))[0]
    if resolved.size == 0:
        index = ratio.size - 1
    elif resolved[0] == 0:
        index = None
    else:
        index = int(resolved[0]) - 1
    if index is not None:
        out["effective_resolution_dx"] = round(float(wavelength[index]), 3)
        out["effective_resolution_km"] = round(
            float(wavelength[index] * profile["dx_km"]), 3)
    out["effective_resolution_ratio"] = SPECTRAL_EFFECTIVE_RATIO
    return out


def intensity_distribution(field: np.ndarray,
                           reference: np.ndarray | None = None,
                           *, floor_dbz: float = ECHO_FLOOR_DBZ) -> dict:
    """In-echo quantiles, and the overlap with a reference's histogram.

    "In-echo" is each field's OWN echo: a forecast that has put its echo
    somewhere else still has an intensity texture, and reporting its
    quantiles only inside the observed echo would confound texture with
    placement, which is what FSS is already for.
    """

    values = np.asarray(field, dtype=float)
    inside = values[values >= floor_dbz]
    edges = np.asarray(HISTOGRAM_EDGES_DBZ, dtype=float)
    out: dict = {
        "floor_dbz": floor_dbz,
        "echo_area_cells": int(inside.size),
        "max_dbz": round(float(values.max()), 3),
    }
    for quantile in DISTRIBUTION_QUANTILES:
        key = f"p{quantile:g}_dbz"
        out[key] = (round(float(np.percentile(inside, quantile)), 3)
                    if inside.size else None)
    if reference is None:
        return out

    other = np.asarray(reference, dtype=float)
    outside = other[other >= floor_dbz]
    for quantile in DISTRIBUTION_QUANTILES:
        key = f"p{quantile:g}_dbz"
        out[f"{key}_bias"] = (
            round(float(np.percentile(inside, quantile)
                        - np.percentile(outside, quantile)), 3)
            if inside.size and outside.size else None)
    if inside.size and outside.size:
        left = np.histogram(inside, bins=edges)[0].astype(float)
        right = np.histogram(outside, bins=edges)[0].astype(float)
        left /= max(left.sum(), 1.0)
        right /= max(right.sum(), 1.0)
        out["histogram_overlap"] = round(float(np.minimum(left, right).sum()), 4)
    else:
        out["histogram_overlap"] = None
    out["histogram_edges_dbz"] = list(HISTOGRAM_EDGES_DBZ)
    out["echo_area_ratio"] = (round(float(inside.size / outside.size), 4)
                              if outside.size else None)
    return out


def field_structure(field: np.ndarray, observed: np.ndarray | None, *,
                    dx_km: float,
                    observed_profile: dict | None = None) -> dict:
    """The three diagnostics for one field, against one reference field."""

    profile = spectral_profile(field, dx_km=dx_km)
    if observed is not None and observed_profile is None:
        observed_profile = spectral_profile(observed, dx_km=dx_km)
    return {
        "objects": object_statistics(field, dx_km=dx_km),
        "spectrum": spectral_report(profile, observed_profile),
        "distribution": intensity_distribution(field, observed),
    }


def _spread(values: list) -> dict | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return {"min": round(float(np.min(present)), 4),
            "median": round(float(np.median(present)), 4),
            "max": round(float(np.max(present)), 4),
            "n": len(present)}


def member_spread(members: dict) -> dict:
    """min/median/max of the member structure statistics, key by key."""

    out = {}
    for group, key in MEMBER_SPREAD_KEYS:
        values = [block[group].get(key) for block in members.values()]
        spread = _spread(values)
        if spread is not None:
            out[f"{group}.{key}"] = spread
    return out


def structure_block(*, observed: np.ndarray, dx_km: float,
                    ensemble_mean: np.ndarray | None = None,
                    members: dict | None = None,
                    extra: dict | None = None) -> dict:
    """Every structure diagnostic for one frame, with the mean kept apart.

    ``members`` and ``ensemble_mean`` are reported as two separate things on
    purpose.  See :data:`ENSEMBLE_MEAN_STRUCTURE_WARNING`.
    """

    observed_profile = spectral_profile(observed, dx_km=dx_km)
    block: dict = {
        "settings": {
            "object_threshold_dbz": OBJECT_THRESHOLD_DBZ,
            "object_min_area_cells": OBJECT_MIN_AREA_CELLS,
            "object_connectivity": OBJECT_CONNECTIVITY,
            "echo_floor_dbz": ECHO_FLOOR_DBZ,
            "spectral_effective_ratio": SPECTRAL_EFFECTIVE_RATIO,
            "spectral_pins_sha256": spectral.PINS_SHA256,
            "dx_km": dx_km,
        },
        "citations": dict(STRUCTURE_CITATIONS),
        "observed": field_structure(observed, None, dx_km=dx_km),
    }
    if ensemble_mean is not None:
        entry = field_structure(ensemble_mean, observed, dx_km=dx_km,
                                observed_profile=observed_profile)
        entry["is_ensemble_average"] = True
        entry["warning"] = ENSEMBLE_MEAN_STRUCTURE_WARNING
        block["ensemble_mean"] = entry
    if members:
        block["members"] = {
            name: field_structure(field, observed, dx_km=dx_km,
                                  observed_profile=observed_profile)
            for name, field in members.items()}
        block["member_spread"] = member_spread(block["members"])
    for label, field in (extra or {}).items():
        block[label] = field_structure(field, observed, dx_km=dx_km,
                                       observed_profile=observed_profile)
    return block


def score_leg(*, composites: Path, obs_path: Path, leg: int, dx_km: float,
              const: dict, neighborhoods_km=None,
              structure: bool = True) -> dict:
    """``verify_numbers`` for one leg, step for step, plus the two views
    a single flattered number leaves out.

    ``neighborhoods_km`` is the curve to trace.  The renderer's own
    ``FSS_BOX_KM`` is always scored and always fills the flat keys, so a
    reader written against the published receipts is unaffected by what
    else was asked for.

    ``structure`` adds the object/spectrum/distribution block; it is on by
    default because a receipt without it cannot answer the only question a
    high neighborhood score does not already answer.
    """

    import netCDF4

    with netCDF4.Dataset(str(obs_path)) as ds:
        z = np.asarray(ds["z_obs"][:], float)
        zmask = np.asarray(ds["z_mask"][:]).astype(bool)
        obs_valid = ds.getncattr("valid_time")

    echo2d = zmask.any(axis=0)
    obs_comp = np.where(zmask, z, -np.inf).max(axis=0)
    obs_comp = np.where(np.isfinite(obs_comp), obs_comp,
                        const["MISSING_OBS_FILL_DBZ"])

    names = member_names(composites, leg)
    if not names:
        raise SystemExit(f"no member composites for leg {leg} in {composites}")
    member_fields = {name: load_composite(composites, leg, name)
                     for name in names}
    members = list(member_fields.values())
    fcst = np.mean(members, axis=0)
    ctrl = load_composite(composites, leg, "control")

    if fcst.shape != obs_comp.shape:
        raise SystemExit(
            f"leg {leg}: composite {fcst.shape} and observation "
            f"{obs_comp.shape} are different grids; these are not the "
            "same case and scoring them together would be meaningless")

    published = float(const["FSS_BOX_KM"])
    half_width = half_width_cells(dx_km, const)
    threshold = const["FSS_THRESHOLD_DBZ"]
    column = const["COLUMN_THRESHOLD_DBZ"]

    # -- the curve.  The published box is in it whatever else was asked --
    wanted = list(neighborhoods_km or ())
    if not any(abs(float(k) - published) < 1e-9 for k in wanted):
        wanted.append(published)
    # A half-width is an integer cell count, so two requested boxes can
    # round onto the same one -- at dx = 3 km, 9 km and 15 km are both 2
    # cells.  They are scored once and the collapse is recorded, because
    # two identical rows under different labels read as two measurements.
    by_half_width: dict[int, list[float]] = {}
    for box_km in sorted(float(k) for k in wanted):
        by_half_width.setdefault(_half_width_for_box(box_km, dx_km),
                                 []).append(round(box_km, 3))
    curve = []
    for hw in sorted(by_half_width):
        requested = by_half_width[hw]
        curve.append({
            "box_km_requested": requested,
            "half_width_cells": hw,
            "box_cells_across": 2 * hw + 1,
            # What was actually scored: the honest label, which is not
            # always what was asked for.
            "box_km_across": round((2 * hw + 1) * dx_km, 3),
            "fss30_fcst": _fss(fcst, obs_comp, threshold=threshold,
                               half_width=hw),
            "fss30_control": _fss(ctrl, obs_comp, threshold=threshold,
                                  half_width=hw),
        })

    # -- the ensemble mean is a field no member produced; score them too --
    per_member = [_fss(member, obs_comp, threshold=threshold,
                       half_width=half_width) for member in members]

    row = {
        "leg": leg,
        "members_scored": len(names),
        "obs_valid_time": obs_valid,
        "obs_cols_gt35": int(((z * zmask).max(axis=0) >= column).sum()),
        "fcst_cols_gt35_in_echo": int((fcst >= column)[echo2d].sum()),
        "control_cols_gt35_in_echo": int((ctrl >= column)[echo2d].sum()),
        "fss30_fcst": _fss(fcst, obs_comp, threshold=threshold,
                           half_width=half_width),
        "fss30_control": _fss(ctrl, obs_comp, threshold=threshold,
                              half_width=half_width),
        "fss_half_width_cells": half_width,
        "fss_box_cells_across": 2 * half_width + 1,
        "fss_box_km_across": round((2 * half_width + 1) * dx_km, 3),
        "per_member": {
            "scored_field": ("each member's own column-max reflectivity, "
                             "at the published box"),
            "member_names": list(names),
            "fss30": per_member,
            "mean": round(float(np.mean(per_member)), 4),
            "min": round(float(np.min(per_member)), 4),
            "max": round(float(np.max(per_member)), 4),
            "stdev": round(float(np.std(per_member, ddof=1)), 4)
                     if len(per_member) > 1 else 0.0,
            "note": ("fss30_fcst above scores the MEAN of these members' "
                     "fields, which is smoother than any of them and "
                     "therefore scores higher; the gap between "
                     "per_member.mean and fss30_fcst is that smoothing"),
        },
        "neighborhood_curve": curve,
    }
    if structure:
        row["structure"] = structure_block(
            observed=obs_comp, dx_km=dx_km, ensemble_mean=fcst,
            members=member_fields, extra={"control": ctrl})
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.da_sweep_score",
        description=__doc__.splitlines()[0])
    parser.add_argument("--composites", type=Path, required=True,
                        action="append", default=[],
                        help="directory of legNN_<name>.npz column maxima. "
                             "Repeatable: a run whose ensemble crossed "
                             "process boundaries writes one per process, "
                             "and the legs are globally numbered, so the "
                             "union is what a single process would have "
                             "written")
    parser.add_argument("--obs-dir", type=Path, required=True,
                        help="directory of verification radar-grid files")
    parser.add_argument("--obs-glob", default="*verify*.nc",
                        help="pattern selecting the verification files, "
                             "in valid-time order (default *verify*.nc)")
    parser.add_argument("--first-free-leg", type=int, required=True,
                        help="leg index of the first free-forecast leg "
                             "(6 for the six-cycle demo shape)")
    parser.add_argument("--dx-km", type=float, required=True)
    parser.add_argument("--label", required=True,
                        help="arm name, carried into the receipt")
    parser.add_argument("--neighborhood-km", type=float, action="append",
                        default=[], metavar="KM",
                        help="repeatable: also score at this square box "
                             "SIDE length, so the receipt carries an "
                             "FSS-versus-scale curve instead of one point "
                             "on it. The renderer's own box is always "
                             "scored and always fills the flat keys")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-structure", action="store_true",
                        help="skip the object/spectrum/distribution block.  "
                             "It is on by default: FSS says whether the area "
                             "is in the right place and nothing else does")
    args = parser.parse_args(argv)
    for box_km in args.neighborhood_km:
        if not np.isfinite(box_km) or box_km <= 0.0:
            raise SystemExit(
                f"--neighborhood-km {box_km} is not a box side length; it "
                "must be finite and positive")

    const, source = metric_constants()
    obs_files = sorted(args.obs_dir.glob(args.obs_glob))
    if not obs_files:
        raise SystemExit(f"no observation files matched {args.obs_glob} "
                         f"in {args.obs_dir}")

    frames = []
    for offset, obs_path in enumerate(obs_files):
        frames.append(score_leg(
            composites=args.composites, obs_path=obs_path,
            leg=args.first_free_leg + offset, dx_km=args.dx_km,
            const=const, neighborhoods_km=args.neighborhood_km,
            structure=not args.no_structure))

    # The curve, averaged over frames: one row per scale, so "FSS rises
    # with the box" is visible as a shape rather than asserted.
    boxes = [row["box_km_across"] for row in frames[0]["neighborhood_curve"]]
    curve_mean = []
    for index, box_km in enumerate(boxes):
        curve_mean.append({
            "box_km_across": box_km,
            "box_cells_across":
                frames[0]["neighborhood_curve"][index]["box_cells_across"],
            "fss30_fcst_mean": round(float(np.mean(
                [f["neighborhood_curve"][index]["fss30_fcst"]
                 for f in frames])), 4),
            "fss30_control_mean": round(float(np.mean(
                [f["neighborhood_curve"][index]["fss30_control"]
                 for f in frames])), 4),
        })

    per_member_means = [f["per_member"]["mean"] for f in frames]

    payload = {
        "schema": "gpuwm-da.sweep-score.v2",
        "label": args.label,
        "dx_km": args.dx_km,
        "constants_source": source,
        "constants": const,
        "neighborhood_convention":
            "FSS_BOX_KM is a square SIDE LENGTH, not a radius; the scored "
            "box is (2*half_width+1) cells across",
        "scored_field":
            "fss30_fcst scores the arithmetic mean over members of each "
            "member's column-max reflectivity -- one deterministic map, "
            "not an ensemble FSS; per_member carries each member's own "
            "score at the same box",
        "truth_smoothing":
            "gpuwm.verify.field_metrics.fss_distance applies the same "
            "boxcar to the observation as to the forecast (Roberts & Lean "
            "smoothed-truth form), which scores higher than a "
            "binary-truth FSS at the same box",
        "composites": [str(c) for c in args.composites],
        "obs_dir": str(args.obs_dir),
        "frames": frames,
        "fss30_fcst_mean": round(
            float(np.mean([f["fss30_fcst"] for f in frames])), 4),
        "fss30_control_mean": round(
            float(np.mean([f["fss30_control"] for f in frames])), 4),
        "fss30_per_member_mean": round(float(np.mean(per_member_means)), 4),
        "fss30_per_member_spread": {
            "min": round(float(np.min([f["per_member"]["min"]
                                       for f in frames])), 4),
            "max": round(float(np.max([f["per_member"]["max"]
                                       for f in frames])), 4),
        },
        "neighborhood_curve_mean": curve_mean,
    }
    if not args.no_structure:
        payload["structure_means"] = structure_means(frames)
        payload["structure_reading_rule"] = ENSEMBLE_MEAN_STRUCTURE_WARNING
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for frame in frames:
        member = frame["per_member"]
        line = (f"leg {frame['leg']:2d}  obs {frame['obs_valid_time']}  "
                f"FSS30 mean-field {frame['fss30_fcst']:.4f}  "
                f"per-member {member['mean']:.4f} "
                f"[{member['min']:.4f}-{member['max']:.4f}]  "
                f"ctrl {frame['fss30_control']:.4f}")
        if "structure" in frame:
            block = frame["structure"]
            line += (f"  objects obs {block['observed']['objects']['count']}"
                     f" / mean {block['ensemble_mean']['objects']['count']}"
                     f" / members "
                     f"{block['member_spread']['objects.count']['min']:.0f}-"
                     f"{block['member_spread']['objects.count']['max']:.0f}")
        print(line)
    print(f"mean FSS30 mean-field {payload['fss30_fcst_mean']:.4f}  "
          f"per-member {payload['fss30_per_member_mean']:.4f}  "
          f"ctrl {payload['fss30_control_mean']:.4f}  "
          f"[constants from {source}]")
    if len(curve_mean) > 1:
        print("FSS vs neighborhood (square side, km):")
        for row in curve_mean:
            print(f"  {row['box_km_across']:8.1f} km  "
                  f"({row['box_cells_across']:3d} cells)  "
                  f"fcst {row['fss30_fcst_mean']:.4f}  "
                  f"ctrl {row['fss30_control_mean']:.4f}")
    return 0


#: The headline structure numbers, as (group, key) pairs.  Anything a
#: scoreboard or a figure caption quotes comes from here so the two cannot
#: quote different statistics under the same name.
HEADLINE_STRUCTURE_KEYS = (
    ("objects", "count"),
    ("objects", "mean_nearest_neighbor_km"),
    ("objects", "largest_object_area_fraction"),
    ("objects", "median_object_area_cells"),
    ("spectrum", "power_ratio_2_4dx"),
    ("spectrum", "effective_resolution_dx"),
    ("distribution", "p99_dbz"),
    ("distribution", "histogram_overlap"),
)


def structure_means(frames: list[dict], extra_labels: tuple = ()) -> dict:
    """Frame-averaged structure numbers, mean and members kept apart.

    ``extra_labels`` names any further field a caller put in the block -- an
    external model, a control -- so a sibling module does not re-derive the
    averaging.
    """

    rows = [f["structure"] for f in frames if "structure" in f]
    if not rows:
        return {}

    def average(pick) -> float | None:
        values = []
        for row in rows:
            try:
                value = pick(row)
            except KeyError:
                value = None
            if value is not None:
                values.append(value)
        return round(float(np.mean(values)), 4) if values else None

    out: dict = {"warning": ENSEMBLE_MEAN_STRUCTURE_WARNING}
    for label in ("observed", "ensemble_mean") + tuple(extra_labels):
        if label not in rows[0]:
            continue
        out[label] = {
            f"{group}.{key}": average(
                lambda r, l=label, g=group, k=key: r[l][g].get(k))
            for group, key in HEADLINE_STRUCTURE_KEYS}
    if "member_spread" in rows[0]:
        out["members"] = {
            key: {statistic: average(
                      lambda r, k=key, s=statistic: r["member_spread"][k][s])
                  for statistic in ("min", "median", "max")}
            for key in rows[0]["member_spread"]}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
