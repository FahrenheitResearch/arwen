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

from gpuwm import obs_score_bridge


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
    return ContingencyTable(*obs_score_bridge.contingency_table(
        observed, forecast, mask, float(threshold)))


def contingency_scores(table: ContingencyTable) -> dict[str, float | int | None]:
    """POD, FAR, CSI, frequency bias, ETS and HSS from one table."""
    scores: dict[str, float | int | None] = dict(table.record())
    names = ("observed_event_fraction", "forecast_event_fraction",
             "probability_of_detection", "false_alarm_ratio",
             "critical_success_index", "frequency_bias",
             "equitable_threat_score", "heidke_skill_score")
    scores.update(zip(names, obs_score_bridge.contingency_scores(
        (int(table.hits), int(table.misses), int(table.false_alarms),
         int(table.correct_negatives)))))
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
