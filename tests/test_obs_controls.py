"""The controls that qualify the instrument, each shown to fire.

A control that only ever passes is decoration.  Every control here is
exercised on both branches -- the case it is meant to accept and the case it
exists to catch -- and the wrong-day control gets the strongest form of that:
it is run end to end through the real scoring pass, scoring a forecast
against a different day's field, and the score has to collapse.

The summary is pinned to fail on an *absent* control as well as a failing
one, because a qualification that reports only the controls somebody
remembered to run is not a qualification.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from gpuwm.verify.obs import battery, controls, regrid, stubs
from gpuwm.verify.obs import registration as reg_mod
from gpuwm.verify.obs.stations import StationPosition, shuffle_positions

from test_obs_battery import (
    CENTER_LAT, CENTER_LON, INIT, LEADS, MemoryArm, _reflectivity_obs,
    _registration, _station_fixture,
)


@pytest.fixture(autouse=True)
def _quiet_stub_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stubs.ObsStubWarning)
        yield


# --------------------------------------------------------------------------
# persistence floor
# --------------------------------------------------------------------------


def test_the_persistence_floor_passes_when_every_arm_clears_it():
    outcome = controls.persistence_floor(
        arm_primary_by_lead={"faithful": {3: 0.40, 4: 0.38, 5: 0.35},
                             "patch": {3: 0.42, 4: 0.39, 5: 0.36}},
        persistence_by_lead={3: 0.20, 4: 0.12, 5: 0.05},
        first_enforced_lead_hour=4)
    assert outcome["status"] == controls.PASS
    assert outcome["failures"] == []
    # Lead 3 is below the enforced lead and is not judged.
    assert {row["lead_hours"] for row in outcome["rows"]} == {4, 5}


def test_an_arm_below_persistence_asks_for_an_investigation_not_a_verdict():
    outcome = controls.persistence_floor(
        arm_primary_by_lead={"faithful": {4: 0.10, 5: 0.35}},
        persistence_by_lead={4: 0.30, 5: 0.05},
        first_enforced_lead_hour=4)
    assert outcome["status"] == controls.INVESTIGATE
    assert outcome["failures"] == ["faithful@4h"]
    assert "indicts the case or the scoring" in outcome["note"]


def test_persistence_is_scored_exactly_like_an_arm():
    obs = _reflectivity_obs()
    arm = MemoryArm(obs, displacement=2)
    grid = arm.grid()
    reference = controls.PersistenceForecast(
        obs.field(INIT), grid=grid, method=regrid.NEAREST,
        max_distance_m=6000.0)

    payload = battery.score_case_arm(
        registration=_registration(), case_id="case-01",
        arm_id="persistence", init_time=INIT, model=reference,
        reflectivity_obs=obs, grid=grid, boundary_width_cells=2)
    assert 0.0 <= payload["primary_scalar"] <= 1.0
    assert reference.record()["reference"] == "persistence"
    assert reference.record()["source_is_stub"] is True


def test_persistence_hands_out_no_observation_where_it_saw_none():
    obs = _reflectivity_obs()
    arm = MemoryArm(obs)
    field = obs.field(INIT)
    reference = controls.PersistenceForecast(
        field, grid=arm.grid(), method=regrid.NEAREST, max_distance_m=6000.0)
    values = reference.composite_reflectivity(INIT)
    assert np.isnan(values).any()
    assert not np.isnan(values).all()


# --------------------------------------------------------------------------
# wrong-day negative control
# --------------------------------------------------------------------------


def test_scoring_against_another_days_observations_collapses_the_score():
    """End to end, through the real scoring pass, on three wrong days.

    Three independent wrong days rather than one, and the aggregate rather
    than a chosen realization: on manufactured fields two random convective
    days overlap enough at a 27 km neighborhood that a single realization can
    land just above the useful-skill line, so picking the realization that
    passes would be picking the answer.  What is asserted is what does not
    depend on the draw -- the score drops, and the drop beats the twin band,
    on every one -- plus the aggregate falling below useful skill.
    """
    obs = _reflectivity_obs()
    arm = MemoryArm(obs, displacement=0)

    same_day = battery.score_case_arm(
        registration=_registration(), case_id="case-01", arm_id="faithful",
        init_time=INIT, model=arm, reflectivity_obs=obs,
        boundary_width_cells=2)
    assert same_day["primary_scalar"] == pytest.approx(1.0)

    primaries = []
    usefuls = []
    for seed in (99999991, 424242, 7777777):
        wrong_day_obs = stubs.StubGriddedObsSource(
            acknowledgement=stubs.STUB_ACKNOWLEDGEMENT,
            quantity="composite_reflectivity", center_latitude=CENTER_LAT,
            center_longitude=CENTER_LON, shape=(280, 280), spacing_deg=0.01,
            seed=seed)
        wrong_day = battery.score_case_arm(
            registration=_registration(), case_id="case-01",
            arm_id="wrong-day", init_time=INIT, model=arm,
            reflectivity_obs=wrong_day_obs, boundary_width_cells=2)
        primaries.append(wrong_day["primary_scalar"])
        usefuls.append(wrong_day["reflectivity"]["primary_fss_useful"])

        outcome = controls.wrong_day_negative(
            same_day_primary=same_day["primary_scalar"],
            wrong_day_primary=wrong_day["primary_scalar"],
            twin_band=0.02,
            fss_useful=wrong_day["reflectivity"]["primary_fss_useful"],
            wrong_day_source=f"a different registered day (draw {seed})")
        assert outcome["clauses"]["score_dropped"] is True
        assert outcome["clauses"]["drop_exceeds_twin_band"] is True

    aggregate = controls.wrong_day_negative(
        same_day_primary=same_day["primary_scalar"],
        wrong_day_primary=float(np.mean(primaries)), twin_band=0.02,
        fss_useful=float(np.mean(usefuls)),
        wrong_day_source="three different registered days")
    assert aggregate["status"] == controls.PASS
    assert aggregate["clauses"] == {"score_dropped": True,
                                    "drop_exceeds_twin_band": True,
                                    "wrong_day_below_useful_skill": True}


def test_a_wrong_day_score_that_does_not_collapse_fails_the_control():
    outcome = controls.wrong_day_negative(
        same_day_primary=0.61, wrong_day_primary=0.60, twin_band=0.02,
        fss_useful=0.55, wrong_day_source="a different registered day")
    assert outcome["status"] == controls.FAIL
    assert outcome["clauses"]["score_dropped"] is True
    assert outcome["clauses"]["drop_exceeds_twin_band"] is False
    assert outcome["clauses"]["wrong_day_below_useful_skill"] is False


# --------------------------------------------------------------------------
# station shuffle
# --------------------------------------------------------------------------


def test_a_deranged_station_map_must_blow_up_every_surface_rmse():
    outcome = controls.station_shuffle_mutation(
        baseline_guardrails={"median_station_rmse:temperature_2m": 1.2,
                             "median_station_rmse:wind_speed_10m": 1.8},
        shuffled_guardrails={"median_station_rmse:temperature_2m": 4.5,
                             "median_station_rmse:wind_speed_10m": 5.1},
        twin_bands={"median_station_rmse:temperature_2m": 0.1,
                    "median_station_rmse:wind_speed_10m": 0.1},
        seed=17)
    assert outcome["status"] == controls.PASS
    assert all(row["blew_up"] for row in outcome["rows"])


def test_a_shuffle_that_changes_nothing_indicts_the_matching_code():
    outcome = controls.station_shuffle_mutation(
        baseline_guardrails={"median_station_rmse:temperature_2m": 1.2},
        shuffled_guardrails={"median_station_rmse:temperature_2m": 1.2},
        twin_bands={"median_station_rmse:temperature_2m": 0.1}, seed=17)
    assert outcome["status"] == controls.FAIL
    assert outcome["failures"] == ["median_station_rmse:temperature_2m"]


def test_the_mutation_replaces_only_the_positions():
    from gpuwm.verify.obs.stations import FrozenStationSet

    positions = {f"S{index}": StationPosition(f"S{index}", float(index), 1.0)
                 for index in range(5)}
    frozen = FrozenStationSet(
        station_ids=tuple(sorted(positions)), positions=positions,
        drops=(), parameters={"note": "test"})
    mutated = controls.build_shuffled_stations(frozen, seed=3)
    assert mutated.station_ids == frozen.station_ids
    assert mutated.parameters == frozen.parameters
    assert mutated.positions != frozen.positions
    assert all(mutated.positions[key].x != frozen.positions[key].x
               for key in positions)
    assert {p.x for p in mutated.positions.values()} == {
        p.x for p in frozen.positions.values()}


def test_a_shuffled_station_set_really_does_wreck_the_surface_scores():
    obs = _reflectivity_obs()
    arm = MemoryArm(obs)
    hours = list(battery.valid_times(INIT, LEADS))
    observations, frozen = _station_fixture(arm, hours)
    assert len(frozen.station_ids) >= 4

    # A model field with real spatial structure, so position matters at all.
    ny, nx = arm.grid().shape
    j, _i = np.indices((ny, nx))
    structured = 285.0 + 0.6 * j

    class Structured(MemoryArm):
        def surface_field(self, valid_time, variable):
            if variable == "temperature_2m":
                return structured
            return super().surface_field(valid_time, variable)

    structured_arm = Structured(obs)
    baseline = battery.score_case_arm(
        registration=_registration(), case_id="case-01", arm_id="baseline",
        init_time=INIT, model=structured_arm, reflectivity_obs=obs,
        boundary_width_cells=2, station_obs=observations,
        frozen_stations=frozen)
    mutated = battery.score_case_arm(
        registration=_registration(), case_id="case-01", arm_id="shuffled",
        init_time=INIT, model=structured_arm, reflectivity_obs=obs,
        boundary_width_cells=2, station_obs=observations,
        frozen_stations=controls.build_shuffled_stations(frozen, seed=5))
    key = "median_station_rmse:temperature_2m"
    assert mutated["guardrails"][key] != baseline["guardrails"][key]


# --------------------------------------------------------------------------
# regrid sensitivity
# --------------------------------------------------------------------------


def test_the_registered_remap_stands_when_the_delta_is_inside_the_band():
    outcome = controls.regrid_sensitivity(
        registered_method=regrid.NEAREST, registered_primary=0.412,
        alternate_method=regrid.CELL_AVERAGE, alternate_primary=0.418,
        twin_band=0.02)
    assert outcome["status"] == controls.PASS
    assert outcome["registered_choice_stands"] is True
    assert outcome["delta"] == pytest.approx(0.006)


def test_a_remap_delta_beyond_the_band_forces_a_re_registration():
    outcome = controls.regrid_sensitivity(
        registered_method=regrid.NEAREST, registered_primary=0.412,
        alternate_method=regrid.CELL_AVERAGE, alternate_primary=0.520,
        twin_band=0.02)
    assert outcome["status"] == controls.INVESTIGATE
    assert outcome["registered_choice_stands"] is False


def test_both_remap_operators_can_actually_be_run_on_one_case():
    obs = _reflectivity_obs()
    arm = MemoryArm(obs, displacement=3)
    primaries = {}
    for method in (regrid.NEAREST, regrid.CELL_AVERAGE):
        registration = reg_mod.make_registration(
            evaluator_commit="1" * 40,
            reflectivity=reg_mod.reflectivity_parameters(
                half_widths=(1, 2, 4), primary_half_width=4,
                regrid_method=method, regrid_max_distance_m=6000.0),
            surface=reg_mod.surface_parameters(),
            precipitation=reg_mod.precipitation_parameters(),
            promotion=reg_mod.promotion_parameters(),
            cases=[{"case_id": "case-01", "init_time": INIT}],
            arms=[{"arm_id": "faithful"}], twin={"rung": 1},
            scored_lead_hours_=LEADS)
        primaries[method] = battery.score_case_arm(
            registration=registration, case_id="case-01", arm_id="faithful",
            init_time=INIT, model=arm, reflectivity_obs=obs,
            boundary_width_cells=2)["primary_scalar"]
    outcome = controls.regrid_sensitivity(
        registered_method=regrid.NEAREST,
        registered_primary=primaries[regrid.NEAREST],
        alternate_method=regrid.CELL_AVERAGE,
        alternate_primary=primaries[regrid.CELL_AVERAGE], twin_band=0.05)
    assert outcome["delta"] >= 0.0


# --------------------------------------------------------------------------
# reflectivity operator cross-check
# --------------------------------------------------------------------------


def test_the_operator_cross_check_is_a_receipt_and_never_a_gate():
    generator = np.random.default_rng(7)
    stored = generator.uniform(-10.0, 60.0, size=(30, 30))
    core = stored + generator.normal(0.0, 1.5, size=(30, 30))
    outcome = controls.reflectivity_operator_crosscheck(
        stored_column_max=stored, core_maxdbz=core,
        valid=np.ones((30, 30), dtype=bool),
        thresholds_dbz=[20.0, 30.0, 40.0],
        operator_pins={"use_varint": False, "use_liqskin": False},
        label="one arm, one lead")
    assert outcome["status"] == controls.RECEIPT
    assert outcome["delta"]["correlation"] > 0.9
    assert outcome["delta"]["sample_count"] == 900
    assert [row["threshold_dbz"] for row in outcome["coverage"]] == [
        20.0, 30.0, 40.0]
    assert "neither result gates anything" in outcome["note"]


def test_the_cross_check_needs_a_common_grid():
    with pytest.raises(ValueError, match="share one grid"):
        controls.reflectivity_operator_crosscheck(
            stored_column_max=np.zeros((4, 4)), core_maxdbz=np.zeros((5, 5)),
            valid=np.ones((4, 4), dtype=bool), thresholds_dbz=[30.0],
            operator_pins={}, label="mismatch")


# --------------------------------------------------------------------------
# twin non-degeneracy
# --------------------------------------------------------------------------


def test_a_twin_that_moves_the_score_keeps_its_rung():
    outcome = controls.twin_non_degeneracy(
        control_primary={"case-01": 0.41, "case-02": 0.38},
        twin_primary={"case-01": 0.43, "case-02": 0.37},
        current_rung="rung-1-one-ulp", next_rung="rung-2-seeded-noise")
    assert outcome["status"] == controls.PASS
    assert outcome["recommended_rung"] == "rung-1-one-ulp"
    assert outcome["twin_band"] == pytest.approx(0.015)


def test_an_output_identical_twin_escalates_instead_of_publishing_zero():
    outcome = controls.twin_non_degeneracy(
        control_primary={"case-01": 0.41, "case-02": 0.38},
        twin_primary={"case-01": 0.41, "case-02": 0.38},
        current_rung="rung-1-one-ulp", next_rung="rung-2-seeded-noise")
    assert outcome["status"] == controls.FAIL
    assert outcome["degenerate"] is True
    assert outcome["twin_band"] == 0.0
    assert outcome["recommended_rung"] == "rung-2-seeded-noise"


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def _capsule(**overrides):
    document = {"run": {"frames": ["a", "b"], "digest": "abc"},
                "pins": {"dt": 15}}
    document.update(overrides)
    return document


def test_matching_dual_run_capsules_and_clean_rehashes_pass():
    outcome = controls.determinism(
        dual_run_pairs=[{"run_id": "faithful-a",
                         "capsule_a": _capsule(), "capsule_b": _capsule()}],
        rehash_records=[{"uri": "s3://obs/1", "matches": True,
                         "rehash_performed": True}])
    assert outcome["status"] == controls.PASS
    assert outcome["dual_run"][0]["identical"] is True


def test_a_diverged_dual_run_pair_names_the_first_divergent_field():
    outcome = controls.determinism(
        dual_run_pairs=[{"run_id": "faithful-a", "capsule_a": _capsule(),
                         "capsule_b": _capsule(
                             pins={"dt": 15}, run={"frames": ["a", "b"],
                                                   "digest": "xyz"})}],
        rehash_records=[{"uri": "s3://obs/1", "matches": True,
                         "rehash_performed": True}])
    assert outcome["status"] == controls.FAIL
    assert outcome["dual_run"][0]["first_divergent_field"]
    assert outcome["failures"] == ["faithful-a"]


def test_an_unperformed_rehash_counts_as_a_failed_one():
    outcome = controls.determinism(
        dual_run_pairs=[{"run_id": "faithful-a", "capsule_a": _capsule(),
                         "capsule_b": _capsule()}],
        rehash_records=[{"uri": "s3://obs/1", "matches": True,
                         "rehash_performed": False}])
    assert outcome["status"] == controls.FAIL


def test_determinism_refuses_an_empty_run_set():
    with pytest.raises(ValueError, match="at least one run pair"):
        controls.determinism(dual_run_pairs=[], rehash_records=[])


def test_each_dual_run_row_says_how_much_it_compared():
    """`identical` without a size is a claim with its scale cut off.

    The dual-run CLI prints the count for this reason: "identical field
    for field" is the same sentence over four quantities and over
    seventy, and this control's row is what a reader of the obs battery
    sees instead of that line.  Both rows below are `identical`; only
    the count distinguishes a screen worth trusting from a thin one.
    """

    from gpuwm.certify.dualrun import capsule_field_paths

    small = _capsule()
    big = _capsule(run={"frames": ["a", "b", "c", "d"], "digest": "abc"},
                   pins={"dt": 15, "diff_opt": 2, "km_opt": 4})
    outcome = controls.determinism(
        dual_run_pairs=[
            {"run_id": "small", "capsule_a": small, "capsule_b": small},
            {"run_id": "big", "capsule_a": big, "capsule_b": big}],
        rehash_records=[{"uri": "s3://obs/1", "matches": True,
                         "rehash_performed": True}])
    assert outcome["status"] == controls.PASS
    rows = {row["run_id"]: row for row in outcome["dual_run"]}
    assert rows["small"]["identical"] is rows["big"]["identical"] is True
    # Against the comparator's own leaf enumeration, not a transcribed number.
    assert rows["small"]["compared_count"] == len(capsule_field_paths(small))
    assert rows["big"]["compared_count"] == len(capsule_field_paths(big))
    assert rows["big"]["compared_count"] > rows["small"]["compared_count"]


def test_a_dual_run_pair_with_nothing_to_compare_refuses(tmp_path):
    """The control cannot be green on nothing either.

    ``compare_capsules`` raises on a pair that offers no leaf, and the
    control lets it out: a ValueError, consistent with its own empty-pairs
    refusal, rather than a PASS row reading `identical: True`.
    """

    with pytest.raises(ValueError):
        controls.determinism(
            dual_run_pairs=[{"run_id": "hollow", "capsule_a": {},
                             "capsule_b": {}}],
            rehash_records=[{"uri": "s3://obs/1", "matches": True,
                             "rehash_performed": True}])


# --------------------------------------------------------------------------
# the summary
# --------------------------------------------------------------------------


def _all_controls():
    return [
        controls.persistence_floor(
            arm_primary_by_lead={"faithful": {4: 0.4}},
            persistence_by_lead={4: 0.1}, first_enforced_lead_hour=4),
        controls.wrong_day_negative(
            same_day_primary=0.5, wrong_day_primary=0.05, twin_band=0.02,
            fss_useful=0.55, wrong_day_source="another day"),
        controls.station_shuffle_mutation(
            baseline_guardrails={"median_station_rmse:temperature_2m": 1.0},
            shuffled_guardrails={"median_station_rmse:temperature_2m": 4.0},
            twin_bands={"median_station_rmse:temperature_2m": 0.1}, seed=1),
        controls.regrid_sensitivity(
            registered_method=regrid.NEAREST, registered_primary=0.4,
            alternate_method=regrid.CELL_AVERAGE, alternate_primary=0.41,
            twin_band=0.05),
        controls.reflectivity_operator_crosscheck(
            stored_column_max=np.zeros((4, 4)), core_maxdbz=np.zeros((4, 4)),
            valid=np.ones((4, 4), dtype=bool), thresholds_dbz=[30.0],
            operator_pins={}, label="flat"),
        controls.twin_non_degeneracy(
            control_primary={"case-01": 0.4}, twin_primary={"case-01": 0.43},
            current_rung="rung-1", next_rung="rung-2"),
        controls.determinism(
            dual_run_pairs=[{"run_id": "a", "capsule_a": _capsule(),
                             "capsule_b": _capsule()}],
            rehash_records=[{"uri": "s3://obs/1", "matches": True,
                             "rehash_performed": True}]),
    ]


def test_a_complete_qualification_passes():
    summary = controls.qualification_summary(_all_controls())
    assert summary["status"] == controls.PASS
    assert summary["missing"] == []
    assert summary["receipt_only"] == ["reflectivity-operator-crosscheck"]


def test_an_absent_control_fails_the_summary_as_loudly_as_a_failing_one():
    summary = controls.qualification_summary(_all_controls()[:-1])
    assert summary["status"] == controls.FAIL
    assert summary["missing"] == ["determinism"]


def test_a_failing_control_fails_the_summary():
    running = _all_controls()
    running[5] = controls.twin_non_degeneracy(
        control_primary={"case-01": 0.4}, twin_primary={"case-01": 0.4},
        current_rung="rung-1", next_rung="rung-2")
    summary = controls.qualification_summary(running)
    assert summary["status"] == controls.FAIL
    assert summary["failed"] == ["twin-non-degeneracy"]


def test_the_useful_skill_line_averages_the_scored_series():
    from gpuwm.verify.obs.fss import FssResult

    series = [FssResult(fss=0.4, threshold_model=30.0, threshold_obs=30.0,
                        half_width=4, observed_base_rate=rate,
                        model_base_rate=rate, fss_useful=0.5 + rate / 2.0,
                        scored_cells=100, frequency_matched=False)
              for rate in (0.02, 0.04)]
    assert controls.useful_skill_line(series) == pytest.approx(0.515)


def test_the_shuffle_helper_and_the_control_agree_on_determinism():
    positions = {f"S{index}": StationPosition(f"S{index}", float(index), 0.0)
                 for index in range(6)}
    assert ({key: value.x for key, value in
             shuffle_positions(positions, seed=11).items()}
            == {key: value.x for key, value in
                shuffle_positions(positions, seed=11).items()})
