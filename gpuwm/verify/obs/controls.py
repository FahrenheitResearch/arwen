"""Controls the instrument must pass before any of its verdicts count.

A verification system that has never been shown to fail is not evidence; it
is a number generator with good manners.  Each control here is a deliberate
attempt to make the battery produce a *wrong* answer, and each one has a
registered criterion written before the data existed:

``persistence floor``
    every model arm must beat persistence beyond the early hours.  Failing
    this indicts the case or the scoring, not the model, and the record says
    so -- an arm that cannot beat "the radar an hour ago, forever" usually
    means the observation is misaligned, not that the forecast is worthless.

``wrong-day negative control``
    scoring a forecast against a different day's observations must collapse
    the score.  If it does not, the instrument is measuring climatology, the
    domain mask, or nothing at all.

``station-shuffle mutation``
    permuting station-to-gridpoint assignment must blow up surface RMSE.  If
    it does not, the matching code never actually read a location.

``regrid sensitivity``
    the same case scored under both registered remap operators.  The delta is
    published; the registered choice stands unless the delta exceeds the twin
    band, in which case it is re-registered once, before any campaign run.

``reflectivity-operator cross-check``
    the science core's own column-max reflectivity beside the stored
    ``REFL_10CM`` column max.  **This is a receipt, not a gate.**  The two
    operators are allowed to differ -- they are different operators -- and
    what matters is that the difference is measured and published rather than
    assumed away.

``twin non-degeneracy``
    the perturbed-twin pair must actually differ on the primary score.  A
    twin band of exactly zero measures nothing and would silently pass every
    chaos-floor clause, so a zero band escalates the perturbation instead of
    being published as a very tight bound.  This consumes a lesson already
    paid for once, where a one-ULP twin ensemble came out output-identical.

``determinism``
    every scored integration's dual-run byte pair is identical, and every
    observation re-hashes to the digest recorded at fetch.

Every criterion that needed a number takes it from the battery's own twin
band or its own useful-skill line rather than from a constant somebody
picked, so there is no invented threshold here to argue with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from gpuwm.certify.dualrun import compare_capsules
from gpuwm.verify.obs import fss as fss_mod
from gpuwm.verify.obs import regrid as regrid_mod
from gpuwm.verify.obs.contracts import ObsGridField
from gpuwm.verify.obs.stations import shuffle_positions

CONTROL_SCHEMA = "gpuwm.obs-battery-control/v1"

PASS = "pass"
FAIL = "fail"
INVESTIGATE = "investigate"
RECEIPT = "receipt-only"


def _record(control: str, status: str, **fields: object) -> dict[str, object]:
    return {"schema": CONTROL_SCHEMA, "control": control, "status": status,
            **fields}


class PersistenceForecast:
    """The zero-skill reference: the observation at init, held forever.

    It satisfies the same reading protocol as a model run, so the scoring
    pass treats it exactly like an arm -- which is the only way its score is
    comparable to one.  Cells the observation could not see at init stay
    unseeable at every lead: they are returned non-finite so the scorer's
    shared-validity mask drops them, rather than being handed out as a
    confident zero.
    """

    def __init__(self, field: ObsGridField, *, grid, method: str,
                 max_distance_m: float) -> None:
        plan = regrid_mod.build_plan(
            source_latitude=field.latitude, source_longitude=field.longitude,
            destination_latitude=grid.latitude,
            destination_longitude=grid.longitude, method=method,
            max_distance_m=max_distance_m)
        values, valid = regrid_mod.apply_plan(plan, field.values, field.valid)
        self._values = np.where(valid, values, np.nan)
        self._grid = grid
        self._plan = plan
        self._source_time = field.valid_time
        self._provenance = field.provenance

    def grid(self):
        return self._grid

    def composite_reflectivity(self, valid_time: str) -> np.ndarray:
        return self._values.copy()

    def record(self) -> dict[str, object]:
        return {
            "reference": "persistence",
            "source_valid_time": self._source_time,
            "source_sha256": self._provenance.sha256,
            "source_is_stub": bool(self._provenance.is_stub),
            "regrid": self._plan.record(),
        }


def persistence_floor(*, arm_primary_by_lead: Mapping[str, Mapping[int, float]],
                      persistence_by_lead: Mapping[int, float],
                      first_enforced_lead_hour: int) -> dict[str, object]:
    """Every arm must beat persistence at and beyond the enforced lead."""
    if not arm_primary_by_lead:
        raise ValueError("the persistence floor needs at least one arm")
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for arm in sorted(arm_primary_by_lead):
        series = arm_primary_by_lead[arm]
        enforced = [lead for lead in sorted(series)
                    if int(lead) >= int(first_enforced_lead_hour)]
        if not enforced:
            raise ValueError(
                f"arm {arm} has no lead at or beyond "
                f"{first_enforced_lead_hour} h to enforce the floor on")
        for lead in enforced:
            if lead not in persistence_by_lead:
                raise ValueError(
                    f"persistence carries no score at lead {lead} h")
            model_score = float(series[lead])
            floor = float(persistence_by_lead[lead])
            beats = model_score > floor
            rows.append({"arm": arm, "lead_hours": int(lead),
                         "arm_primary": model_score,
                         "persistence_primary": floor, "beats_floor": beats})
            if not beats:
                failures.append(f"{arm}@{lead}h")
    return _record(
        "persistence-floor", PASS if not failures else INVESTIGATE,
        first_enforced_lead_hour=int(first_enforced_lead_hour),
        rows=rows, failures=failures,
        note=("a failure indicts the case or the scoring before it indicts "
              "the model: investigate the observation alignment and the masks"))


def wrong_day_negative(*, same_day_primary: float, wrong_day_primary: float,
                       twin_band: float, fss_useful: float,
                       wrong_day_source: str) -> dict[str, object]:
    """Scoring against another registered day's observations must collapse.

    "Decisively" is spelled with the battery's own quantities rather than an
    invented constant: the score must drop, the drop must exceed the twin
    band, and the wrong-day score must fall below the useful-skill line.
    """
    for name, value in (("same_day_primary", same_day_primary),
                        ("wrong_day_primary", wrong_day_primary),
                        ("twin_band", twin_band), ("fss_useful", fss_useful)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} is not finite")
    drop = float(same_day_primary) - float(wrong_day_primary)
    clauses = {
        "score_dropped": bool(drop > 0.0),
        "drop_exceeds_twin_band": bool(drop > float(twin_band)),
        "wrong_day_below_useful_skill": bool(
            float(wrong_day_primary) < float(fss_useful)),
    }
    return _record(
        "wrong-day-negative", PASS if all(clauses.values()) else FAIL,
        same_day_primary=float(same_day_primary),
        wrong_day_primary=float(wrong_day_primary), drop=drop,
        twin_band=float(twin_band), fss_useful=float(fss_useful),
        wrong_day_source=str(wrong_day_source), clauses=clauses,
        criterion=("the score drops, the drop exceeds the twin band, and the "
                   "wrong-day score falls below 0.5 + f_obs/2"))


def station_shuffle_mutation(*, baseline_guardrails: Mapping[str, float],
                             shuffled_guardrails: Mapping[str, float],
                             twin_bands: Mapping[str, float],
                             seed: int) -> dict[str, object]:
    """A deranged station-to-gridpoint map must blow up every surface RMSE."""
    if set(baseline_guardrails) != set(shuffled_guardrails):
        raise ValueError(
            "the shuffled run scored a different guardrail set than the "
            "baseline; the mutation must change the assignment and nothing else")
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for name in sorted(baseline_guardrails):
        if name not in twin_bands:
            raise ValueError(f"guardrail {name!r} has no twin band")
        base = float(baseline_guardrails[name])
        mutated = float(shuffled_guardrails[name])
        increase = mutated - base
        blew_up = bool(increase > float(twin_bands[name]))
        rows.append({"guardrail": name, "baseline_rmse": base,
                     "shuffled_rmse": mutated, "increase": increase,
                     "twin_band": float(twin_bands[name]),
                     "blew_up": blew_up})
        if not blew_up:
            failures.append(name)
    return _record(
        "station-shuffle-mutation", PASS if not failures else FAIL,
        seed=int(seed), rows=rows, failures=failures,
        criterion=("every scored surface RMSE rises by more than that "
                   "metric's own twin band under a derangement of the "
                   "station-to-gridpoint assignment"))


def build_shuffled_stations(frozen, *, seed: int):
    """The mutated station set the control scores; the set is not otherwise touched."""
    from dataclasses import replace

    return replace(frozen, positions=shuffle_positions(frozen.positions,
                                                       seed=int(seed)))


def regrid_sensitivity(*, registered_method: str, registered_primary: float,
                       alternate_method: str, alternate_primary: float,
                       twin_band: float) -> dict[str, object]:
    """Publish the remap delta and say whether the registered choice stands."""
    if registered_method == alternate_method:
        raise ValueError("the sensitivity needs two different remap operators")
    delta = abs(float(registered_primary) - float(alternate_primary))
    within = bool(delta <= float(twin_band))
    return _record(
        "regrid-sensitivity", PASS if within else INVESTIGATE,
        registered_method=str(registered_method),
        registered_primary=float(registered_primary),
        alternate_method=str(alternate_method),
        alternate_primary=float(alternate_primary),
        delta=delta, twin_band=float(twin_band),
        registered_choice_stands=within,
        note=("a delta beyond the twin band means the remap operator is a "
              "scoring choice with consequences: re-register once, before any "
              "campaign run, never after"))


@dataclass(frozen=True)
class OperatorDelta:
    """Summary statistics for two reflectivity operators on one field."""

    mean_difference: float
    median_absolute_difference: float
    rmse: float
    maximum_absolute_difference: float
    correlation: float
    sample_count: int

    def record(self) -> dict[str, object]:
        return {
            "mean_difference": self.mean_difference,
            "median_absolute_difference": self.median_absolute_difference,
            "rmse": self.rmse,
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "correlation": self.correlation,
            "sample_count": int(self.sample_count),
        }


def reflectivity_operator_crosscheck(
        *, stored_column_max: np.ndarray, core_maxdbz: np.ndarray,
        valid: np.ndarray, thresholds_dbz: Sequence[float],
        operator_pins: Mapping[str, object], label: str) -> dict[str, object]:
    """Measure the two reflectivity operators against each other. Not a gate.

    Also reports the exceedance coverage each operator produces at each
    scored threshold, because a coverage difference is what would actually
    move an FSS -- a difference in the mean says much less.
    """
    stored = np.asarray(stored_column_max, dtype=np.float64)
    core = np.asarray(core_maxdbz, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if stored.shape != core.shape or stored.shape != mask.shape:
        raise ValueError("the cross-check operands must share one grid")
    if not mask.any():
        raise ValueError("the cross-check has no valid cells")
    left = stored[mask]
    right = core[mask]
    difference = right - left
    correlation = 0.0
    if left.size > 1 and float(left.std()) > 0.0 and float(right.std()) > 0.0:
        correlation = float(np.corrcoef(left, right)[0, 1])
    delta = OperatorDelta(
        mean_difference=float(np.mean(difference, dtype=np.float64)),
        median_absolute_difference=float(np.median(np.abs(difference))),
        rmse=float(np.sqrt(np.mean(difference * difference, dtype=np.float64))),
        maximum_absolute_difference=float(np.max(np.abs(difference))),
        correlation=correlation,
        sample_count=int(left.size))
    coverage = []
    for threshold in thresholds_dbz:
        coverage.append({
            "threshold_dbz": float(threshold),
            "stored_coverage": float(np.mean(left >= float(threshold),
                                             dtype=np.float64)),
            "core_coverage": float(np.mean(right >= float(threshold),
                                           dtype=np.float64)),
        })
    return _record(
        "reflectivity-operator-crosscheck", RECEIPT, label=str(label),
        primary_operator="column maximum over k of stored REFL_10CM",
        cross_check_operator="science-core maxdbz",
        operator_pins=dict(operator_pins), delta=delta.record(),
        coverage=coverage,
        note=("operator provenance, published as a receipt; the two operators "
              "may legitimately differ and neither result gates anything"))


def twin_non_degeneracy(*, control_primary: Mapping[str, float],
                        twin_primary: Mapping[str, float],
                        current_rung: str,
                        next_rung: str) -> dict[str, object]:
    """A twin band of exactly zero measures nothing; escalate rather than publish it.

    Registered before data because it has already happened once: a one-ULP
    perturbed ensemble came out output-identical and every envelope metric
    read zero.  A zero band would pass every chaos-floor clause by
    construction, which is the opposite of what the clause is for.
    """
    cases = sorted(set(control_primary) & set(twin_primary))
    if not cases:
        raise ValueError("the twin check needs at least one paired case")
    if set(control_primary) != set(twin_primary):
        raise ValueError(
            "twin and control cover different cases: "
            f"{sorted(set(control_primary) ^ set(twin_primary))}")
    deltas = {case: abs(float(twin_primary[case]) - float(control_primary[case]))
              for case in cases}
    band = float(np.median(np.asarray(sorted(deltas.values()),
                                      dtype=np.float64)))
    degenerate = band == 0.0
    return _record(
        "twin-non-degeneracy", FAIL if degenerate else PASS,
        per_case_absolute_difference=deltas, twin_band=band,
        pair_count=len(cases), degenerate=degenerate,
        current_rung=str(current_rung),
        recommended_rung=str(next_rung) if degenerate else str(current_rung),
        note=("a degenerate band escalates the perturbation ladder as one "
              "registered motion, before any campaign case is scored, never "
              "per case"))


def determinism(*, dual_run_pairs: Sequence[Mapping[str, object]],
                rehash_records: Sequence[Mapping[str, object]]
                ) -> dict[str, object]:
    """Dual-run byte screens and observation re-hashes, both total."""
    if not dual_run_pairs:
        raise ValueError("the determinism control needs at least one run pair")
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for pair in dual_run_pairs:
        run_id = str(pair["run_id"])
        comparison = compare_capsules(pair["capsule_a"], pair["capsule_b"])
        rows.append({"run_id": run_id, "identical": comparison.identical,
                     "first_divergent_field": comparison.first_divergent_field,
                     "divergence_count": len(comparison.divergences)})
        if not comparison.identical:
            failures.append(run_id)
    rehash_rows = []
    for record in rehash_records:
        matches = bool(record.get("matches", False))
        performed = bool(record.get("rehash_performed", False))
        rehash_rows.append({"uri": record.get("uri", "<unknown>"),
                            "matches": matches,
                            "rehash_performed": performed})
        if not (performed and matches):
            failures.append(str(record.get("uri", "<unknown>")))
    if not rehash_records:
        failures.append("no observation re-hash records")
    return _record(
        "determinism", PASS if not failures else FAIL,
        dual_run=rows, observation_rehash=rehash_rows, failures=failures,
        note=("the card carries no ECC: a dual-run byte pair is the "
              "corruption detector, and an unperformed re-hash counts as a "
              "failed one"))


def qualification_summary(controls: Sequence[Mapping[str, object]]
                          ) -> dict[str, object]:
    """Roll the controls into one verdict, naming what is missing.

    The required set is enumerated so an *absent* control fails the summary.
    A qualification that only reports the controls somebody remembered to run
    is not a qualification.
    """
    required = {
        "persistence-floor", "wrong-day-negative",
        "station-shuffle-mutation", "regrid-sensitivity",
        "reflectivity-operator-crosscheck", "twin-non-degeneracy",
        "determinism",
    }
    present = {str(control["control"]) for control in controls}
    missing = sorted(required - present)
    gating = [control for control in controls
              if str(control["status"]) != RECEIPT]
    failed = sorted(str(control["control"]) for control in gating
                    if str(control["status"]) != PASS)
    return {
        "schema": CONTROL_SCHEMA,
        "control": "qualification-summary",
        "status": PASS if not missing and not failed else FAIL,
        "required": sorted(required),
        "present": sorted(present),
        "missing": missing,
        "failed": failed,
        "receipt_only": sorted(str(control["control"]) for control in controls
                               if str(control["status"]) == RECEIPT),
    }


def useful_skill_line(results: Sequence[fss_mod.FssResult]) -> float:
    """Mean ``0.5 + f_obs/2`` over a scored series, for the wrong-day criterion."""
    values = [float(result.fss_useful) for result in results]
    if not values:
        raise ValueError("the useful-skill line needs at least one scored time")
    return float(np.mean(values, dtype=np.float64))


__all__ = [
    "CONTROL_SCHEMA", "FAIL", "INVESTIGATE", "PASS", "RECEIPT",
    "OperatorDelta", "PersistenceForecast", "build_shuffled_stations",
    "determinism", "persistence_floor", "qualification_summary",
    "reflectivity_operator_crosscheck", "regrid_sensitivity",
    "station_shuffle_mutation", "twin_non_degeneracy", "useful_skill_line",
    "wrong_day_negative",
]
