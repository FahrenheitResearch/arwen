"""The promotion evaluator: four clauses, and none of them self-ratified.

A patch is promoted to the model's default only when all four registered
clauses hold on the frozen case set:

``R1`` direction and significance
    a one-sided exact Wilcoxon signed-rank test on the per-case differences,
    against the direction registered *before* any data existed.  Exact, by
    enumerating the null distribution of the signed-rank sum over every sign
    assignment of the observed absolute ranks -- not a normal approximation,
    which at seven cases would be a fiction.  The honest consequence is
    printed with every verdict: at n = 7 the smallest attainable p is 1/128,
    so this battery can certify consistent effects and nothing else.

``R2`` chaos floor
    the cross-case median improvement must exceed the twin band -- what the
    same score moves when two runs of the *same* model differ only by a
    perturbation too small to mean anything.  An improvement inside that band
    is recorded as indistinguishable from chaos and the patch stays a
    candidate.

``R3`` no-harm guardrails
    reflectivity skill may not be bought with a broken surface climate or
    broken precipitation.  Each guardrail has a registered direction, and a
    patch may not degrade the cross-case median of any of them by more than
    that metric's own twin band.

``R4`` integrity
    every scored integration passed its dual-run byte screen, every
    observation re-hashed to the digest recorded at fetch, every score file
    carries the registration hash and evaluator commit it claims -- and no
    stand-in input appears anywhere in the evidence.

And one thing this module will not do: call a patch promoted under a rule the
owner has not ratified.  A registration built without a ratification
reference produces verdicts labelled ``proposed-unratified``, with
``promoted`` false and the reason stated.  The arithmetic is still run and
still published; what is withheld is the authority, which was never the
evaluator's to grant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from gpuwm.verify.obs.registration import (
    HIGHER_IS_BETTER, LOWER_IS_BETTER, RATIFIED_UNDER_DELEGATION,
    RATIFYING_STATUSES, validate_registration,
)

PROMOTION_SCHEMA = "gpuwm.obs-battery-promotion/v1"

GREATER = "greater"
LESS = "less"


def average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks with ties averaged, ascending, one-based."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("ranking an empty sample is undefined")
    order = np.argsort(array, kind="stable")
    ranks = np.empty(array.size, dtype=np.float64)
    index = 0
    while index < array.size:
        stop = index
        while (stop + 1 < array.size
               and array[order[stop + 1]] == array[order[index]]):
            stop += 1
        mean_rank = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[order[position]] = mean_rank
        index = stop + 1
    return [float(value) for value in ranks]


@dataclass(frozen=True)
class SignedRankResult:
    """The exact one-sided signed-rank test on a set of paired differences."""

    p_value: float
    signed_rank_sum: float
    sample_size: int
    dropped_zero_count: int
    alternative: str
    minimum_attainable_p: float

    def record(self) -> dict[str, object]:
        return {
            "p_value": self.p_value,
            "signed_rank_sum": self.signed_rank_sum,
            "sample_size": int(self.sample_size),
            "dropped_zero_count": int(self.dropped_zero_count),
            "alternative": self.alternative,
            "minimum_attainable_p": self.minimum_attainable_p,
            "test": "exact Wilcoxon signed-rank, enumerated null distribution",
        }


def exact_signed_rank_test(differences: Sequence[float], *,
                           alternative: str = GREATER) -> SignedRankResult:
    """Exact one-sided Wilcoxon signed-rank test (Wilcoxon 1945).

    Exact-zero differences are dropped and the sample size reduced, which is
    Wilcoxon's own procedure and what the registration pins.  Tied magnitudes
    take average ranks; the null distribution is then enumerated over sign
    assignments of the *observed* absolute ranks, so the test stays exact
    under ties rather than falling back on a tie correction.

    Enumeration is a dynamic program over doubled ranks (average ranks are
    half-integers), with exact integer counts -- so the p-value is a rational
    number evaluated in integers and rounded once, not accumulated in floats.
    """
    if alternative not in (GREATER, LESS):
        raise ValueError(f"unknown alternative {alternative!r}")
    values = [float(value) for value in differences]
    if not values:
        raise ValueError("the signed-rank test needs at least one difference")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("a per-case difference is non-finite")
    if alternative == LESS:
        values = [-value for value in values]
    nonzero = [value for value in values if value != 0.0]
    dropped = len(values) - len(nonzero)
    if not nonzero:
        return SignedRankResult(
            p_value=1.0, signed_rank_sum=0.0, sample_size=0,
            dropped_zero_count=dropped, alternative=alternative,
            minimum_attainable_p=1.0)
    ranks = average_ranks([abs(value) for value in nonzero])
    doubled = [int(round(rank * 2.0)) for rank in ranks]
    observed = sum(rank for rank, value in zip(doubled, nonzero) if value > 0.0)
    total = sum(doubled)
    counts = [0] * (total + 1)
    counts[0] = 1
    for rank in doubled:
        updated = [0] * (total + 1)
        for value, count in enumerate(counts):
            if not count:
                continue
            updated[value] += count
            updated[value + rank] += count
        counts = updated
    tail = sum(counts[observed:])
    space = 1 << len(nonzero)
    return SignedRankResult(
        p_value=float(tail / space),
        signed_rank_sum=observed / 2.0,
        sample_size=len(nonzero),
        dropped_zero_count=dropped,
        alternative=alternative,
        minimum_attainable_p=float(1.0 / space),
    )


def twin_band(control: Mapping[str, float], twin: Mapping[str, float]
              ) -> float:
    """Cross-case median of ``|S(twin) - S(control)|`` for one score.

    One perturbed twin per case and the median across cases: with a case set
    this small a per-case twin *ensemble* is unaffordable, so this is the
    honest estimator and the pair count travels with it in the receipt.
    """
    cases = sorted(set(control) & set(twin))
    if not cases:
        raise ValueError("the twin band needs at least one paired case")
    if set(control) != set(twin):
        raise ValueError(
            "twin and control cover different cases: "
            f"{sorted(set(control) ^ set(twin))}")
    deltas = [abs(float(twin[case]) - float(control[case])) for case in cases]
    return float(np.median(np.asarray(deltas, dtype=np.float64)))


def _paired_differences(patch: Mapping[str, float],
                        baseline: Mapping[str, float], *,
                        case_ids: Sequence[str]) -> list[float]:
    missing = sorted(set(case_ids) - (set(patch) & set(baseline)))
    if missing:
        raise ValueError(
            f"the frozen case set is not covered by both arms: {missing}; a "
            f"case is excluded for every arm symmetrically or for none")
    return [float(patch[case]) - float(baseline[case]) for case in case_ids]


@dataclass(frozen=True)
class GuardrailOutcome:
    """One no-harm guardrail evaluated against its own twin band."""

    name: str
    direction: str
    median_change: float
    degradation: float
    band: float
    passed: bool

    def record(self) -> dict[str, object]:
        return {
            "name": self.name, "direction": self.direction,
            "median_change": self.median_change,
            "degradation": self.degradation, "twin_band": self.band,
            "passed": bool(self.passed),
        }


def evaluate_guardrail(*, name: str, direction: str,
                       patch: Mapping[str, float],
                       baseline: Mapping[str, float],
                       band: float, case_ids: Sequence[str]
                       ) -> GuardrailOutcome:
    """Whether a guardrail's cross-case median moved harmfully past its band."""
    if direction not in (LOWER_IS_BETTER, HIGHER_IS_BETTER):
        raise ValueError(f"guardrail {name}: unregistered direction {direction!r}")
    if not math.isfinite(float(band)) or float(band) < 0.0:
        raise ValueError(f"guardrail {name}: the twin band must be finite and >= 0")
    changes = _paired_differences(patch, baseline, case_ids=case_ids)
    median_change = float(np.median(np.asarray(changes, dtype=np.float64)))
    degradation = (median_change if direction == LOWER_IS_BETTER
                   else -median_change)
    return GuardrailOutcome(
        name=str(name), direction=str(direction), median_change=median_change,
        degradation=float(degradation), band=float(band),
        passed=bool(degradation <= float(band)))


def evaluate_integrity(evidence: Mapping[str, object]) -> dict[str, object]:
    """R4: the evidence is what it says it is.

    Every clause is a refusal, not a warning.  A score file built from a
    stand-in observation is not evidence of anything about the sky, and it is
    the one failure this clause is guaranteed to catch even when everything
    else looks clean.
    """
    failures: list[str] = []
    runs = list(evidence.get("runs", ()))
    if not runs:
        failures.append("no scored integrations are recorded")
    for run in runs:
        name = str(run.get("run_id", "<unnamed>"))
        if run.get("engine_requires_dual_run", True):
            if not run.get("dual_run_identical", False):
                failures.append(f"{name}: dual-run byte screen did not pass")
        if not str(run.get("registration_sha256", "")).strip():
            failures.append(f"{name}: score file carries no registration hash")
        if not str(run.get("evaluator_commit", "")).strip():
            failures.append(f"{name}: score file carries no evaluator commit")
        if run.get("uses_stub_inputs", False):
            failures.append(f"{name}: scored against stand-in observations")
    rehash = list(evidence.get("observation_rehash", ()))
    if not rehash:
        failures.append("no observation re-hash records are present")
    for record in rehash:
        if not record.get("matches", False):
            failures.append(
                f"observation {record.get('uri', '<unknown>')} re-hashed "
                f"differently at scoring")
    return {"passed": not failures, "failures": failures,
            "run_count": len(runs), "observation_count": len(rehash)}


def evaluate_patch(*, registration: Mapping[str, object], patch_id: str,
                   primary_patch: Mapping[str, float],
                   primary_baseline: Mapping[str, float],
                   primary_twin_band: float,
                   guardrail_values: Mapping[str, Mapping[str, Mapping[str, float]]],
                   guardrail_bands: Mapping[str, float],
                   integrity_evidence: Mapping[str, object],
                   ) -> dict[str, object]:
    """Run R1-R4 on one patch arm and return the verdict record.

    ``guardrail_values`` maps a guardrail name to ``{"patch": {case: value},
    "baseline": {case: value}}``.  Every guardrail the registration names
    must be present: a missing guardrail is a refusal, because "we did not
    measure it" and "it did not degrade" are different sentences.
    """
    reg = validate_registration(registration)
    parameters = reg["parameters"]
    promotion = parameters["promotion"]
    case_ids = [str(case["case_id"]) for case in parameters["cases"]]
    alpha = float(promotion["alpha"])

    differences = _paired_differences(
        primary_patch, primary_baseline, case_ids=case_ids)
    test = exact_signed_rank_test(differences, alternative=GREATER)
    r1_passed = bool(test.p_value <= alpha)

    median_difference = float(
        np.median(np.asarray(differences, dtype=np.float64)))
    band = float(primary_twin_band)
    if not math.isfinite(band) or band < 0.0:
        raise ValueError("the primary twin band must be finite and >= 0")
    r2_passed = bool(median_difference > band)

    outcomes: list[GuardrailOutcome] = []
    for entry in promotion["guardrails"]:
        name = str(entry["name"])
        if name not in guardrail_values:
            raise ValueError(
                f"the registration names guardrail {name!r} and the evidence "
                f"does not carry it")
        if name not in guardrail_bands:
            raise ValueError(f"guardrail {name!r} has no twin band")
        values = guardrail_values[name]
        outcomes.append(evaluate_guardrail(
            name=name, direction=str(entry["direction"]),
            patch=values["patch"], baseline=values["baseline"],
            band=float(guardrail_bands[name]), case_ids=case_ids))
    r3_passed = all(outcome.passed for outcome in outcomes)

    integrity = evaluate_integrity(integrity_evidence)
    r4_passed = bool(integrity["passed"])

    clauses_passed = r1_passed and r2_passed and r3_passed and r4_passed
    # Which authority a verdict rests on is part of the verdict.  A rule
    # ratified under a standing delegation promotes exactly as a rule the
    # owner ruled on number by number does -- and says which it was, because
    # the two are not the same provenance and a reader may weigh them
    # differently.
    ratified = reg["rule_status"] in RATIFYING_STATUSES
    delegated = reg["rule_status"] == RATIFIED_UNDER_DELEGATION
    blocked = "" if ratified else (
        "the promotion rule's numbers are proposals until the owner ratifies "
        "them; this record states the arithmetic and withholds the verdict")
    promotion_pins = reg["parameters"]["promotion"]
    authority = {
        "rule_status": reg["rule_status"],
        "ratification_reference": str(
            promotion_pins.get("ratification_reference", "")),
        "delegation_reference": str(
            promotion_pins.get("delegation_reference", "")),
        "overrule_window": str(promotion_pins.get("overrule_window", "")),
        "note": ("promoted under a standing delegation, not under a "
                 "number-by-number owner ruling"
                 if delegated else
                 "promoted under the owner's own ruling on these numbers"
                 if ratified else
                 "no promotion authority: these numbers are proposals"),
    }
    if not r2_passed and r1_passed:
        summary = "indistinguishable-from-chaos"
    elif clauses_passed:
        summary = "meets-rule" if not ratified else "promote"
    else:
        summary = "does-not-meet-rule"
    return {
        "schema": PROMOTION_SCHEMA,
        "patch_id": str(patch_id),
        "registration_sha256": reg["registration_sha256"],
        "evaluator_commit": reg["evaluator_commit"],
        "rule_status": reg["rule_status"],
        "rule_authority": authority,
        "case_ids": case_ids,
        "case_count": len(case_ids),
        "per_case_difference": {
            case: float(value) for case, value in zip(case_ids, differences)},
        "r1_direction_and_significance": {
            "passed": r1_passed, "alpha": alpha, **test.record()},
        "r2_chaos_floor": {
            "passed": r2_passed, "median_difference": median_difference,
            "twin_band": band},
        "r3_guardrails": {
            "passed": r3_passed,
            "outcomes": [outcome.record() for outcome in outcomes]},
        "r4_integrity": {"passed": r4_passed, **integrity},
        "all_clauses_passed": clauses_passed,
        "promoted": bool(clauses_passed and ratified),
        "promotion_withheld_reason": blocked if clauses_passed and not ratified
                                     else "",
        "verdict": summary,
    }


def evaluate_scoreboard(*, registration: Mapping[str, object],
                        label: str,
                        challenger: Mapping[str, float],
                        incumbent: Mapping[str, float],
                        twin_band_value: float) -> dict[str, object]:
    """The scoreboard comparison: R1 and R2 between two engines' arms.

    The same two clauses, run both ways round by the caller.  If the sentence
    does not pass, the tables are published and no sentence is claimed; if it
    passes in the other direction, that is published in the same type size.
    This function states which arm it was handed and nothing else.
    """
    reg = validate_registration(registration)
    parameters = reg["parameters"]
    case_ids = [str(case["case_id"]) for case in parameters["cases"]]
    alpha = float(parameters["promotion"]["alpha"])
    differences = _paired_differences(
        challenger, incumbent, case_ids=case_ids)
    test = exact_signed_rank_test(differences, alternative=GREATER)
    median_difference = float(
        np.median(np.asarray(differences, dtype=np.float64)))
    r1 = bool(test.p_value <= alpha)
    r2 = bool(median_difference > float(twin_band_value))
    return {
        "schema": PROMOTION_SCHEMA,
        "comparison": str(label),
        "registration_sha256": reg["registration_sha256"],
        "evaluator_commit": reg["evaluator_commit"],
        "rule_status": reg["rule_status"],
        "case_ids": case_ids,
        "per_case_difference": {
            case: float(value) for case, value in zip(case_ids, differences)},
        "r1_direction_and_significance": {"passed": r1, "alpha": alpha,
                                          **test.record()},
        "r2_chaos_floor": {"passed": r2,
                           "median_difference": median_difference,
                           "twin_band": float(twin_band_value)},
        "claim_supported": bool(r1 and r2),
        "claim_status": ("supported" if r1 and r2 else "not-supported"),
        "note": ("a claim is supported by this record only when the rule "
                 "carrying it has been ratified; rule_status says whether it "
                 "has"),
    }


__all__ = [
    "GREATER", "LESS", "PROMOTION_SCHEMA", "GuardrailOutcome",
    "SignedRankResult", "average_ranks", "evaluate_guardrail",
    "evaluate_integrity", "evaluate_patch", "evaluate_scoreboard",
    "exact_signed_rank_test", "twin_band",
]
