"""Neighborhood fractions skill score against observations, with masks.

The arithmetic is Roberts & Lean (2008, *MWR* 136, 78-97) and the engine is
:mod:`gpuwm.verify.field_metrics` -- the same boxcar and the same
``1 - FSS`` definition the model-vs-model lanes already use.  This module
adds the one thing scoring against observations needs and scoring against
another model does not: **a validity mask**.

Observed grids have holes.  MRMS flags cells the radar network cannot see,
the interior mask excludes the specified and relaxation rows plus a rim, and
neither hole may be scored as if it were a confident zero.  Zero is not
missing -- treating a masked cell as "no reflectivity" inflates agreement
wherever both fields are quiet for the same uninformative reason.

The masked formulation, registered:

* the scored validity is the **intersection** of the model's and the
  observation's validity, so both fields answer for the same cells;
* the neighborhood fraction at a cell is
  ``boxcar(event AND valid) / boxcar(valid)``, i.e. the event fraction
  *among the valid cells in the box*, not among all cells in the box;
* a cell whose box holds no valid cell at all is dropped from the sums;
* the FSS sums run over the scored region only, and the denominator
  convention matches the existing engine exactly: ``FSS = 1`` when both
  fraction fields are identically zero over the scored region.

With an all-true mask, an all-true scored region and the ``edge`` boundary
below, this reduces term for term to ``1 - field_metrics.fss_distance``;
``tests/test_obs_fss.py`` pins that identity rather than asserting it here.

One boundary property worth stating because it is load-bearing and easy to
get quietly wrong.  :func:`gpuwm.verify.field_metrics.boxcar` pads by edge
extension, which replicates the outermost row outward -- i.e. it assumes the
field continues past the domain.  For a model-vs-model comparison that is
harmless; against observations it is a claim that cells outside the domain
were observed, and at the widest registered neighborhood the scored region
does reach within one cell of the array edge.  So this module registers the
boundary treatment as a parameter:

``zero`` (the battery's registered choice)
    outside the array is not observed.  The arrays are explicitly zero-padded
    by the neighborhood half-width before the boxcar runs, so the padded ring
    contributes nothing to either the numerator or the denominator -- and
    because the denominator counts only valid cells, those padded cells drop
    out of the fraction entirely rather than diluting it.  A box straddling
    the edge reports the event fraction among the observations it actually
    contains.

``edge``
    the existing engine's own behavior, kept so the reduction to
    ``1 - field_metrics.fss_distance`` is exact and testable rather than
    approximate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from gpuwm.verify import field_metrics

#: Boundary treatments for the neighborhood boxcar.  ``ZERO_BOUNDARY`` says
#: outside the array was not observed; ``EDGE_BOUNDARY`` is the existing
#: engine's edge extension, kept so the reduction to that engine is exact.
ZERO_BOUNDARY = "zero"
EDGE_BOUNDARY = "edge"
BOUNDARIES = (ZERO_BOUNDARY, EDGE_BOUNDARY)


def box_length_m(half_width: int, dx_m: float) -> float:
    """Physical edge length of the neighborhood box.

    Receipts publish this rather than the half-width in cells, because a
    half-width is not comparable across grid spacings and a length is.
    """
    if half_width < 0:
        raise ValueError("neighborhood half-width must be non-negative")
    return float((2 * int(half_width) + 1) * float(dx_m))


@dataclass(frozen=True)
class FssResult:
    """One FSS evaluation: the score and everything needed to read it."""

    fss: float
    threshold_model: float
    threshold_obs: float
    half_width: int
    observed_base_rate: float
    model_base_rate: float
    fss_useful: float
    scored_cells: int
    frequency_matched: bool

    def record(self) -> dict[str, object]:
        return {
            "fss": self.fss,
            "threshold_model": self.threshold_model,
            "threshold_obs": self.threshold_obs,
            "half_width_cells": int(self.half_width),
            "observed_base_rate": self.observed_base_rate,
            "model_base_rate": self.model_base_rate,
            "fss_useful": self.fss_useful,
            "scored_cells": int(self.scored_cells),
            "frequency_matched": bool(self.frequency_matched),
        }


def shared_validity(*masks: np.ndarray) -> np.ndarray:
    """The cells every mask calls valid."""
    if not masks:
        raise ValueError("shared validity needs at least one mask")
    shape = np.asarray(masks[0]).shape
    result = np.ones(shape, dtype=bool)
    for mask in masks:
        array = np.asarray(mask)
        if array.shape != shape:
            raise ValueError("validity masks must share one shape")
        if array.dtype != np.bool_:
            raise ValueError("a validity mask must be boolean")
        result &= array
    return result


def _boxcar(field: np.ndarray, half_width: int, boundary: str) -> np.ndarray:
    """Boxcar mean under the registered boundary treatment.

    Under ``zero`` the array is explicitly zero-padded by the half-width
    first, so the engine's edge extension only ever replicates the pad.  The
    padded box for every original cell lies wholly inside the padded array,
    which makes the result exactly the zero-padded local sum divided by the
    full box area.
    """
    width = 2 * int(half_width) + 1
    if boundary == EDGE_BOUNDARY:
        return field_metrics.boxcar(field, width)
    if boundary != ZERO_BOUNDARY:
        raise ValueError(f"unknown boundary {boundary!r}; expected {BOUNDARIES}")
    if half_width == 0:
        return np.asarray(field, dtype=np.float64)
    padded = np.pad(np.asarray(field, dtype=np.float64),
                    ((half_width, half_width), (half_width, half_width)),
                    mode="constant", constant_values=0.0)
    smoothed = field_metrics.boxcar(padded, width)
    return smoothed[half_width:-half_width, half_width:-half_width]


def masked_neighborhood_fraction(events: np.ndarray, valid: np.ndarray,
                                 half_width: int, *,
                                 boundary: str = ZERO_BOUNDARY
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Event fraction among valid cells in each box, and the box valid count.

    Returns ``(fraction, valid_count)``.  ``fraction`` is zero wherever the
    box holds no valid cell; ``valid_count`` is what callers use to drop
    those cells rather than score a fabricated zero.
    """
    events = np.asarray(events, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if events.shape != valid.shape or events.ndim != 2:
        raise ValueError("events and validity must be common-shape 2-D fields")
    if half_width < 0:
        raise ValueError("neighborhood half-width must be non-negative")
    counted = _boxcar((events & valid).astype(np.float64), int(half_width),
                      boundary)
    denominator = _boxcar(valid.astype(np.float64), int(half_width), boundary)
    fraction = np.zeros_like(counted)
    populated = denominator > 0.0
    np.divide(counted, denominator, out=fraction, where=populated)
    return fraction, denominator


def frequency_matched_threshold(field: np.ndarray, valid: np.ndarray,
                                target_fraction: float) -> float:
    """The threshold at which ``field`` covers ``target_fraction`` of the mask.

    Amplitude bias aliases into a fixed-threshold FSS: a model that is
    uniformly a few dBZ cold scores badly at 30 dBZ for a reason that has
    nothing to do with where it put the storms.  Matching the frequency first
    removes that term (Roberts & Lean 2008 sec. 4; Mittermaier & Roberts 2010,
    *WAF* 25, 343-354).

    The quantile is linear-interpolated over the valid cells; the degenerate
    ends are defined rather than left to the quantile routine: a target of
    zero coverage returns a threshold above the field maximum (nothing
    exceeds it), and full coverage returns one at or below the minimum.
    """
    values = np.asarray(field, dtype=np.float64)[np.asarray(valid, dtype=bool)]
    if values.size == 0:
        raise ValueError("frequency matching has no valid cells to rank")
    target = float(target_fraction)
    if not 0.0 <= target <= 1.0:
        raise ValueError("target exceedance fraction must lie in [0, 1]")
    if target <= 0.0:
        return float(np.nextafter(values.max(), np.inf))
    if target >= 1.0:
        return float(values.min())
    return float(np.quantile(values, 1.0 - target, method="linear"))


def masked_fss(model: np.ndarray, obs: np.ndarray, *, valid: np.ndarray,
               threshold: float, half_width: int,
               score_mask: np.ndarray | None = None,
               frequency_matched: bool = False,
               boundary: str = ZERO_BOUNDARY) -> FssResult:
    """FSS between a forecast and an observation over the valid, scored cells.

    ``threshold`` is the observed threshold always.  Under
    ``frequency_matched`` the model side instead uses the threshold that
    reproduces the observed exceedance fraction; the pair of thresholds is
    reported, so a reader can never mistake a frequency-matched row for a
    fixed-threshold one.

    ``boundary`` is the registered neighborhood boundary treatment; see the
    module docstring for why it is a parameter and not a convention.
    """
    model = np.asarray(model, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if model.shape != obs.shape or model.ndim != 2:
        raise ValueError("FSS operands must be common-shape 2-D fields")
    if valid.shape != model.shape:
        raise ValueError("the validity mask must match the scored fields")
    if score_mask is None:
        scored = np.ones_like(valid)
    else:
        scored = np.asarray(score_mask, dtype=bool)
        if scored.shape != model.shape:
            raise ValueError("the scored-region mask must match the fields")
    if boundary not in BOUNDARIES:
        raise ValueError(f"unknown boundary {boundary!r}; expected {BOUNDARIES}")
    if not valid.any():
        raise ValueError("FSS has no valid cells")

    obs_events = (obs >= float(threshold)) & valid
    denominator_cells = int(np.count_nonzero(valid & scored))
    if denominator_cells == 0:
        raise ValueError("FSS has no cells inside both the mask and the region")
    observed_rate = float(
        np.count_nonzero(obs_events & scored) / denominator_cells)

    model_threshold = float(threshold)
    if frequency_matched:
        model_threshold = frequency_matched_threshold(
            model, valid & scored, observed_rate)
    model_events = (model >= model_threshold) & valid
    model_rate = float(
        np.count_nonzero(model_events & scored) / denominator_cells)

    model_fraction, counts = masked_neighborhood_fraction(
        model_events, valid, half_width, boundary=boundary)
    obs_fraction, _ = masked_neighborhood_fraction(
        obs_events, valid, half_width, boundary=boundary)
    region = scored & (counts > 0.0)
    scored_cells = int(np.count_nonzero(region))
    if scored_cells == 0:
        raise ValueError("no scored cell has a populated neighborhood")
    difference = model_fraction[region] - obs_fraction[region]
    numerator = float(np.sum(difference * difference, dtype=np.float64))
    reference = float(np.sum(model_fraction[region] ** 2
                             + obs_fraction[region] ** 2, dtype=np.float64))
    fss = 1.0 if reference == 0.0 else 1.0 - numerator / reference
    if not math.isfinite(fss):
        raise ValueError("FSS is non-finite")
    return FssResult(
        fss=float(min(1.0, max(0.0, fss))),
        threshold_model=model_threshold,
        threshold_obs=float(threshold),
        half_width=int(half_width),
        observed_base_rate=observed_rate,
        model_base_rate=model_rate,
        fss_useful=0.5 + observed_rate / 2.0,
        scored_cells=scored_cells,
        frequency_matched=bool(frequency_matched),
    )


def fss_matrix(model: np.ndarray, obs: np.ndarray, *, valid: np.ndarray,
               thresholds: Sequence[float], half_widths: Sequence[int],
               score_mask: np.ndarray | None = None,
               frequency_matched: bool = False,
               boundary: str = ZERO_BOUNDARY,
               ) -> dict[tuple[float, int], FssResult]:
    """The threshold x neighborhood matrix at one valid time."""
    if not thresholds or not half_widths:
        raise ValueError("an FSS matrix needs thresholds and neighborhoods")
    results: dict[tuple[float, int], FssResult] = {}
    for threshold in thresholds:
        for half_width in half_widths:
            key = (float(threshold), int(half_width))
            if key in results:
                raise ValueError(f"duplicate FSS matrix cell {key}")
            results[key] = masked_fss(
                model, obs, valid=valid, threshold=float(threshold),
                half_width=int(half_width), score_mask=score_mask,
                frequency_matched=frequency_matched, boundary=boundary)
    return results


def mean_fss(results: Sequence[FssResult]) -> float:
    """Unweighted mean over valid times -- the primary scalar's reducer.

    Unweighted on purpose: every scored hour is one observation of the
    forecast's skill, and weighting by coverage would let a single
    heavily-covered hour speak for the case.
    """
    values = [float(result.fss) for result in results]
    if not values:
        raise ValueError("an FSS mean needs at least one scored time")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("an FSS series carries a non-finite score")
    return float(np.mean(values, dtype=np.float64))


def matrix_records(matrix: Mapping[tuple[float, int], FssResult], *,
                   dx_m: float) -> list[dict[str, object]]:
    """The published rows for one matrix, ordered and physically labelled."""
    rows: list[dict[str, object]] = []
    for (threshold, half_width) in sorted(matrix):
        record = matrix[(threshold, half_width)].record()
        record["box_length_m"] = box_length_m(half_width, dx_m)
        rows.append(record)
    return rows


__all__ = [
    "BOUNDARIES", "EDGE_BOUNDARY", "FssResult", "ZERO_BOUNDARY",
    "box_length_m", "frequency_matched_threshold",
    "fss_matrix", "masked_fss", "masked_neighborhood_fraction",
    "matrix_records", "mean_fss", "shared_validity",
]
