"""The promotion rule's four clauses, and its refusal to ratify itself.

The exact test's small-sample arithmetic is pinned against the numbers the
specification states in prose, because those numbers are the honest limit of
what a seven-case battery can certify and a silently-approximated p-value
would erase exactly that limit: at n = 7 the minimum attainable p is 1/128, a
6-1 split passes only when the loss is the smallest in magnitude (2/128), and
a loss ranked fourth already fails (7/128 > 0.05).

The rest of the file tries to get something promoted that should not be:
an improvement inside the chaos band, an improvement bought with a wrecked
surface climate, evidence built on stand-in observations, and -- the one that
matters most -- a clean sweep under a rule the owner has never ratified.
"""

from __future__ import annotations

import pytest

from gpuwm.verify.obs import registration as reg_mod
from gpuwm.verify.obs.promotion import (
    GREATER, LESS, average_ranks, evaluate_guardrail, evaluate_integrity,
    evaluate_patch, evaluate_scoreboard, exact_signed_rank_test, twin_band,
)

COMMIT = "0" * 40
CASES = tuple(f"case-{index:02d}" for index in range(1, 8))
GUARDRAILS = tuple(str(entry["name"]) for entry in reg_mod.DEFAULT_GUARDRAILS)


def _registration(*, ratification_reference: str = "",
                  delegation_reference: str = "",
                  overrule_window: str = "") -> dict:
    return reg_mod.make_registration(
        evaluator_commit=COMMIT,
        reflectivity=reg_mod.reflectivity_parameters(),
        surface=reg_mod.surface_parameters(),
        precipitation=reg_mod.precipitation_parameters(),
        promotion=reg_mod.promotion_parameters(
            ratification_reference=ratification_reference,
            delegation_reference=delegation_reference,
            overrule_window=overrule_window),
        cases=[{"case_id": case, "init_time": "2026-08-03T12:00:00"}
               for case in CASES],
        arms=[{"arm_id": "faithful"}, {"arm_id": "patch"}],
        twin={"rung": 1, "perturbation": "one documented FP ULP"})


def _series(values):
    return {case: value for case, value in zip(CASES, values)}


# --------------------------------------------------------------------------
# the exact test
# --------------------------------------------------------------------------


def test_seven_wins_is_the_smallest_attainable_p_at_n_equals_seven():
    result = exact_signed_rank_test([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    assert result.p_value == pytest.approx(1.0 / 128.0)
    assert result.minimum_attainable_p == pytest.approx(1.0 / 128.0)
    assert result.sample_size == 7
    assert result.signed_rank_sum == 28.0


def test_a_six_one_split_passes_only_when_the_loss_is_the_smallest():
    smallest_loss = exact_signed_rank_test(
        [-0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    assert smallest_loss.p_value == pytest.approx(2.0 / 128.0)
    assert smallest_loss.p_value <= reg_mod.DEFAULT_ALPHA


def test_a_loss_ranked_fourth_already_fails_at_alpha_five_percent():
    fourth_loss = exact_signed_rank_test(
        [0.01, 0.02, 0.03, -0.04, 0.05, 0.06, 0.07])
    assert fourth_loss.p_value == pytest.approx(7.0 / 128.0)
    assert fourth_loss.p_value > reg_mod.DEFAULT_ALPHA


def test_five_cases_can_only_certify_a_clean_sweep():
    assert exact_signed_rank_test(
        [0.1, 0.2, 0.3, 0.4, 0.5]).p_value == pytest.approx(1.0 / 32.0)
    assert exact_signed_rank_test(
        [-0.1, 0.2, 0.3, 0.4, 0.5]).p_value == pytest.approx(2.0 / 32.0)
    assert exact_signed_rank_test([-0.1, 0.2, 0.3, 0.4, 0.5]).p_value > 0.05


def test_zero_differences_are_dropped_and_the_sample_shrinks():
    result = exact_signed_rank_test([0.0, 0.0, 0.1, 0.2, 0.3])
    assert result.dropped_zero_count == 2
    assert result.sample_size == 3
    assert result.p_value == pytest.approx(1.0 / 8.0)


def test_an_all_zero_sample_is_no_evidence_rather_than_an_error():
    result = exact_signed_rank_test([0.0, 0.0, 0.0])
    assert result.p_value == 1.0
    assert result.sample_size == 0


def test_ties_take_average_ranks_and_the_test_stays_exact():
    assert average_ranks([5.0, 1.0, 1.0, 3.0]) == [4.0, 1.5, 1.5, 3.0]
    # Three equal-magnitude wins: every sign assignment is enumerated over
    # the observed absolute ranks, so the sweep is still 1/8.
    assert exact_signed_rank_test(
        [0.2, 0.2, 0.2]).p_value == pytest.approx(1.0 / 8.0)


def test_the_direction_is_registered_and_the_mirror_image_agrees():
    positive = exact_signed_rank_test([0.1, 0.2, 0.3], alternative=GREATER)
    negative = exact_signed_rank_test([-0.1, -0.2, -0.3], alternative=LESS)
    assert positive.p_value == negative.p_value
    assert exact_signed_rank_test(
        [-0.1, -0.2, -0.3], alternative=GREATER).p_value == 1.0


def test_a_non_finite_difference_is_refused():
    with pytest.raises(ValueError, match="non-finite"):
        exact_signed_rank_test([0.1, float("inf")])


# --------------------------------------------------------------------------
# the twin band and the guardrails
# --------------------------------------------------------------------------


def test_the_twin_band_is_the_cross_case_median_absolute_difference():
    control = _series([0.50, 0.55, 0.60, 0.45, 0.52, 0.58, 0.61])
    twin = _series([0.52, 0.53, 0.66, 0.45, 0.55, 0.57, 0.60])
    assert twin_band(control, twin) == pytest.approx(0.02)


def test_the_twin_band_refuses_an_unpaired_case_set():
    with pytest.raises(ValueError, match="different cases"):
        twin_band({"a": 0.1, "b": 0.2}, {"a": 0.1})


def test_a_guardrail_reads_its_direction():
    worse_rmse = evaluate_guardrail(
        name="median_station_rmse:temperature_2m",
        direction=reg_mod.LOWER_IS_BETTER,
        patch=_series([1.5] * 7), baseline=_series([1.0] * 7),
        band=0.1, case_ids=CASES)
    assert worse_rmse.degradation == pytest.approx(0.5)
    assert not worse_rmse.passed

    better_rmse = evaluate_guardrail(
        name="median_station_rmse:temperature_2m",
        direction=reg_mod.LOWER_IS_BETTER,
        patch=_series([0.9] * 7), baseline=_series([1.0] * 7),
        band=0.1, case_ids=CASES)
    assert better_rmse.passed

    worse_fss = evaluate_guardrail(
        name="mean_fss:precipitation", direction=reg_mod.HIGHER_IS_BETTER,
        patch=_series([0.30] * 7), baseline=_series([0.50] * 7),
        band=0.05, case_ids=CASES)
    assert worse_fss.degradation == pytest.approx(0.20)
    assert not worse_fss.passed


def test_a_guardrail_inside_its_own_band_still_passes():
    outcome = evaluate_guardrail(
        name="median_station_rmse:wind_speed_10m",
        direction=reg_mod.LOWER_IS_BETTER,
        patch=_series([1.05] * 7), baseline=_series([1.0] * 7),
        band=0.1, case_ids=CASES)
    assert outcome.passed
    assert outcome.degradation == pytest.approx(0.05)


# --------------------------------------------------------------------------
# integrity
# --------------------------------------------------------------------------


def _clean_evidence():
    return {
        "runs": [{"run_id": "arm-a", "dual_run_identical": True,
                  "registration_sha256": "x", "evaluator_commit": COMMIT,
                  "uses_stub_inputs": False}],
        "observation_rehash": [{"uri": "s3://obs/1", "matches": True}],
    }


def test_clean_evidence_passes_integrity():
    assert evaluate_integrity(_clean_evidence())["passed"] is True


def test_a_failed_dual_run_screen_fails_integrity():
    evidence = _clean_evidence()
    evidence["runs"][0]["dual_run_identical"] = False
    outcome = evaluate_integrity(evidence)
    assert not outcome["passed"]
    assert "dual-run byte screen" in outcome["failures"][0]


def test_stand_in_observations_can_never_become_evidence():
    evidence = _clean_evidence()
    evidence["runs"][0]["uses_stub_inputs"] = True
    outcome = evaluate_integrity(evidence)
    assert not outcome["passed"]
    assert any("stand-in" in failure for failure in outcome["failures"])


def test_an_observation_that_rehashed_differently_fails_integrity():
    evidence = _clean_evidence()
    evidence["observation_rehash"][0]["matches"] = False
    assert not evaluate_integrity(evidence)["passed"]


def test_missing_rehash_records_are_a_failure_not_an_omission():
    evidence = _clean_evidence()
    evidence["observation_rehash"] = []
    assert not evaluate_integrity(evidence)["passed"]


# --------------------------------------------------------------------------
# the whole rule
# --------------------------------------------------------------------------


def _patch_call(*, primary_patch, primary_baseline, twin=0.005,
                guardrail_patch=None, ratification="", delegation="",
                overrule_window=""):
    baseline_guardrails = {name: _series([1.0] * 7) for name in GUARDRAILS}
    patch_guardrails = guardrail_patch or {
        name: _series([1.0] * 7) for name in GUARDRAILS}
    return evaluate_patch(
        registration=_registration(
            ratification_reference=ratification,
            delegation_reference=delegation,
            overrule_window=overrule_window),
        patch_id="L4",
        primary_patch=primary_patch, primary_baseline=primary_baseline,
        primary_twin_band=twin,
        guardrail_values={
            name: {"patch": patch_guardrails[name],
                   "baseline": baseline_guardrails[name]}
            for name in GUARDRAILS},
        guardrail_bands={name: 0.1 for name in GUARDRAILS},
        integrity_evidence=_clean_evidence())


def test_a_clean_sweep_meets_the_rule_but_is_not_promoted_unratified():
    verdict = _patch_call(
        primary_patch=_series([0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61]),
        primary_baseline=_series([0.50] * 7))
    assert verdict["r1_direction_and_significance"]["passed"]
    assert verdict["r2_chaos_floor"]["passed"]
    assert verdict["r3_guardrails"]["passed"]
    assert verdict["r4_integrity"]["passed"]
    assert verdict["all_clauses_passed"] is True
    assert verdict["promoted"] is False
    assert verdict["rule_status"] == reg_mod.UNRATIFIED
    assert "ratifies them" in verdict["promotion_withheld_reason"]
    assert verdict["verdict"] == "meets-rule"


def test_the_same_sweep_is_promoted_once_the_owner_has_ratified():
    verdict = _patch_call(
        primary_patch=_series([0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61]),
        primary_baseline=_series([0.50] * 7),
        ratification="owner ruling 2026-08-03")
    assert verdict["rule_status"] == reg_mod.RATIFIED
    assert verdict["promoted"] is True
    assert verdict["verdict"] == "promote"
    assert verdict["promotion_withheld_reason"] == ""


def test_a_delegated_ratification_promotes_and_says_that_is_what_it_was():
    # A standing delegation carries the same promotion authority and NOT the
    # same provenance, so the verdict has to be able to say which one it
    # rested on.  A reader who weighs the two differently is entitled to.
    verdict = _patch_call(
        primary_patch=_series([0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61]),
        primary_baseline=_series([0.50] * 7),
        delegation="standing delegation, 2026-08-04: 'just make right calls "
                   "and keep going'",
        overrule_window="until the campaign launches")
    assert verdict["rule_status"] == reg_mod.RATIFIED_UNDER_DELEGATION
    assert verdict["promoted"] is True
    assert verdict["verdict"] == "promote"
    assert verdict["promotion_withheld_reason"] == ""
    authority = verdict["rule_authority"]
    assert authority["ratification_reference"] == ""
    assert "standing delegation" in authority["delegation_reference"]
    assert authority["overrule_window"] == "until the campaign launches"
    assert "standing delegation" in authority["note"]


def test_a_delegated_rule_must_carry_its_provenance_and_its_overrule_window():
    # The whole difference between this status and plain ratification is that
    # a reader can see where the authority came from and when it stops being
    # overrulable.  Neither may be omitted, and neither may be forged by
    # claiming both authorities at once.
    with pytest.raises(ValueError, match="overrule window"):
        reg_mod.promotion_parameters(delegation_reference="a delegation")
    with pytest.raises(ValueError, match="never both"):
        reg_mod.promotion_parameters(
            ratification_reference="owner ruling",
            delegation_reference="a delegation",
            overrule_window="until launch")
    document = _registration(delegation_reference="a delegation",
                             overrule_window="until launch")
    document["parameters"]["promotion"]["overrule_window"] = ""
    document["registration_sha256"] = reg_mod.canonical_hash(
        document["parameters"])
    with pytest.raises(ValueError, match="overrule_window"):
        reg_mod.validate_registration(document)


def test_an_improvement_inside_the_chaos_band_is_not_a_result():
    verdict = _patch_call(
        primary_patch=_series([0.501, 0.502, 0.503, 0.504, 0.505, 0.506,
                               0.507]),
        primary_baseline=_series([0.50] * 7),
        twin=0.02, ratification="owner ruling 2026-08-03")
    assert verdict["r1_direction_and_significance"]["passed"] is True
    assert verdict["r2_chaos_floor"]["passed"] is False
    assert verdict["promoted"] is False
    assert verdict["verdict"] == "indistinguishable-from-chaos"


def test_reflectivity_skill_cannot_be_bought_with_a_broken_surface():
    broken = {name: _series([1.0] * 7) for name in GUARDRAILS}
    broken["median_station_rmse:temperature_2m"] = _series([2.0] * 7)
    verdict = _patch_call(
        primary_patch=_series([0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61]),
        primary_baseline=_series([0.50] * 7),
        guardrail_patch=broken, ratification="owner ruling 2026-08-03")
    assert verdict["r1_direction_and_significance"]["passed"]
    assert verdict["r3_guardrails"]["passed"] is False
    assert verdict["promoted"] is False


def test_a_missing_guardrail_is_a_refusal_not_a_pass():
    baseline = {name: _series([1.0] * 7) for name in GUARDRAILS}
    incomplete = {name: {"patch": baseline[name], "baseline": baseline[name]}
                  for name in GUARDRAILS[:-1]}
    with pytest.raises(ValueError, match="does not carry it"):
        evaluate_patch(
            registration=_registration(), patch_id="L3",
            primary_patch=_series([0.6] * 7),
            primary_baseline=_series([0.5] * 7), primary_twin_band=0.01,
            guardrail_values=incomplete,
            guardrail_bands={name: 0.1 for name in GUARDRAILS},
            integrity_evidence=_clean_evidence())


def test_a_case_missing_from_one_arm_is_refused_symmetrically():
    short = _series([0.6] * 7)
    short.pop(CASES[-1])
    with pytest.raises(ValueError, match="excluded for every arm"):
        _patch_call(primary_patch=short,
                    primary_baseline=_series([0.5] * 7))


def test_the_scoreboard_comparison_states_the_arm_it_was_handed():
    registration = _registration(ratification_reference="owner ruling")
    record = evaluate_scoreboard(
        registration=registration, label="challenger vs incumbent",
        challenger=_series([0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61]),
        incumbent=_series([0.50] * 7), twin_band_value=0.005)
    assert record["claim_supported"] is True
    assert record["comparison"] == "challenger vs incumbent"

    reversed_record = evaluate_scoreboard(
        registration=registration, label="incumbent vs challenger",
        challenger=_series([0.50] * 7),
        incumbent=_series([0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61]),
        twin_band_value=0.005)
    assert reversed_record["claim_supported"] is False
    assert reversed_record["claim_status"] == "not-supported"
