"""Contingency-table scores for a forecast against an observation.

The 2x2 table and the scores derived from it are textbook (Schaefer 1990,
*WAF* 5, 570-575, for ETS; Wilks 2011, *Statistical Methods in the
Atmospheric Sciences*, ch. 8, for the family).  The formulas already existed
in this tree inside a case-named lane comparator, where they could only ever
compare one campaign's two model arms.  They are re-stated here in a
mechanism-named module that takes its fields, its threshold and its validity
mask as arguments and knows nothing else -- so the same call scores a
forecast against radar, against a gauge analysis, or against another model.

Two deliberate properties:

* **An undefined score is ``None``, never zero.**  POD with no observed
  events is not "zero detection", it is a question the day did not ask.
  Publishing zero there would drag a case average toward a number nobody
  measured.
* **The observation is the first argument.**  Hits, misses and false alarms
  are asymmetric, and the argument order fixes which field is the referee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContingencyTable:
    """Counts of the four outcomes over the scored cells."""

    hits: int
    misses: int
    false_alarms: int
    correct_negatives: int

    @property
    def total(self) -> int:
        return (int(self.hits) + int(self.misses) + int(self.false_alarms)
                + int(self.correct_negatives))

    def record(self) -> dict[str, int]:
        return {
            "hits": int(self.hits), "misses": int(self.misses),
            "false_alarms": int(self.false_alarms),
            "correct_negatives": int(self.correct_negatives),
            "total": self.total,
        }


def contingency_table(observed: np.ndarray, forecast: np.ndarray, *,
                      threshold: float,
                      valid: np.ndarray | None = None) -> ContingencyTable:
    """The 2x2 table for ``field >= threshold`` over the valid cells."""
    observed = np.asarray(observed, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    if observed.shape != forecast.shape or observed.ndim != 2:
        raise ValueError("contingency operands must share one 2-D grid")
    if valid is None:
        mask = np.ones(observed.shape, dtype=bool)
    else:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != observed.shape:
            raise ValueError("the validity mask must match the scored grid")
    if not mask.any():
        raise ValueError("a contingency table needs at least one valid cell")
    observed_event = (observed >= float(threshold)) & mask
    forecast_event = (forecast >= float(threshold)) & mask
    return ContingencyTable(
        hits=int(np.count_nonzero(observed_event & forecast_event)),
        misses=int(np.count_nonzero(observed_event & ~forecast_event & mask)),
        false_alarms=int(
            np.count_nonzero(~observed_event & forecast_event & mask)),
        correct_negatives=int(
            np.count_nonzero(~observed_event & ~forecast_event & mask)),
    )


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def contingency_scores(table: ContingencyTable) -> dict[str, float | int | None]:
    """POD, FAR, CSI, frequency bias, ETS and HSS from one table."""
    hits = int(table.hits)
    misses = int(table.misses)
    false_alarms = int(table.false_alarms)
    correct_negatives = int(table.correct_negatives)
    total = table.total
    if total == 0:
        raise ValueError("a contingency table with no cells has no scores")
    random_hits = ((hits + misses) * (hits + false_alarms) / total
                   if total else 0.0)
    hss_numerator = 2.0 * (hits * correct_negatives - misses * false_alarms)
    hss_denominator = ((hits + misses) * (misses + correct_negatives)
                       + (hits + false_alarms)
                       * (false_alarms + correct_negatives))
    scores: dict[str, float | int | None] = dict(table.record())
    scores.update({
        "observed_event_fraction": _ratio(hits + misses, total),
        "forecast_event_fraction": _ratio(hits + false_alarms, total),
        "probability_of_detection": _ratio(hits, hits + misses),
        "false_alarm_ratio": _ratio(false_alarms, hits + false_alarms),
        "critical_success_index": _ratio(hits, hits + misses + false_alarms),
        "frequency_bias": _ratio(hits + false_alarms, hits + misses),
        "equitable_threat_score": _ratio(
            hits - random_hits,
            hits + misses + false_alarms - random_hits),
        "heidke_skill_score": _ratio(hss_numerator, hss_denominator),
    })
    return scores


def score_field(observed: np.ndarray, forecast: np.ndarray, *,
                threshold: float, valid: np.ndarray | None = None
                ) -> dict[str, float | int | None]:
    """Table and scores in one call, the form a receipt row carries."""
    scores = contingency_scores(
        contingency_table(observed, forecast, threshold=threshold,
                          valid=valid))
    scores["threshold"] = float(threshold)
    return scores


__all__ = [
    "ContingencyTable", "contingency_scores", "contingency_table",
    "score_field",
]
