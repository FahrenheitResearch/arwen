"""The registration, and the scoring pass end to end.

The forecast here is a deliberately simple in-memory arm rather than a
history file: this file is testing the harness, and a harness that can only
be exercised through 17 GB of wrfout is a harness nobody exercises.  The
observations are the loud stand-ins, and the last test in this file is the
one that matters -- a score file built on them says so, and that flag is what
stops it becoming a verdict.

The scored/reported split gets its own pin.  Spin-up leads are reported and
never scored, so a lead that appears in the reported block must be incapable
of reaching the primary scalar; the test asks for it explicitly.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime

import numpy as np
import pytest

from gpuwm.verify.obs import battery, registration as reg_mod, stubs
from gpuwm.verify.obs.contracts import ModelGrid, ObservedFractionBelowFloor
from gpuwm.verify.obs.stations import StationPosition, freeze_station_set

COMMIT = "1" * 40
INIT = "2026-08-03T12:00:00"
CENTER_LAT = 37.0
CENTER_LON = -97.0
LEADS = (2, 3, 4)


@pytest.fixture(autouse=True)
def _quiet_stub_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stubs.ObsStubWarning)
        yield


def _registration(*, leads=LEADS, ratification="", windows=(1,)):
    return reg_mod.make_registration(
        evaluator_commit=COMMIT,
        reflectivity=reg_mod.reflectivity_parameters(
            half_widths=(1, 2, 4), primary_half_width=4,
            regrid_max_distance_m=6000.0),
        surface=reg_mod.surface_parameters(),
        precipitation=reg_mod.precipitation_parameters(
            half_widths=(1, 2, 4), guardrail_half_width=4,
            window_hours=tuple(windows),
            guardrail_window_hours=max(windows),
            regrid_max_distance_m=8000.0),
        promotion=reg_mod.promotion_parameters(
            ratification_reference=ratification),
        cases=[{"case_id": "case-01", "init_time": INIT}],
        arms=[{"arm_id": "faithful"}],
        twin={"rung": 1},
        scored_lead_hours_=leads)


class MemoryArm:
    """One forecast arm held in memory, satisfying the reading protocol.

    ``displacement`` shifts the forecast against the observation, so a test
    can make an arm better or worse on demand without touching the scorer.
    """

    def __init__(self, obs_source, *, shape=(80, 80), spacing_deg=0.03,
                 displacement=0, surface_offset=0.0, rain_rate=2.0):
        latitude, longitude = stubs.regular_grid(
            center_latitude=CENTER_LAT, center_longitude=CENTER_LON,
            shape=shape, spacing_deg=spacing_deg)
        self._grid = ModelGrid(latitude=latitude, longitude=longitude,
                               dx_m=3000.0,
                               terrain_m=np.full(shape, 300.0))
        self._obs = obs_source
        self._displacement = int(displacement)
        self._surface_offset = float(surface_offset)
        self._rain_rate = float(rain_rate)
        self._shape = shape

    def grid(self):
        return self._grid

    def _observed_on_model_grid(self, valid_time):
        from gpuwm.verify.obs import regrid

        field = self._obs.field(valid_time)
        plan = regrid.build_plan(
            source_latitude=field.latitude, source_longitude=field.longitude,
            destination_latitude=self._grid.latitude,
            destination_longitude=self._grid.longitude,
            method=regrid.NEAREST, max_distance_m=6000.0)
        values, _valid = regrid.apply_plan(plan, field.values, field.valid)
        return values

    def composite_reflectivity(self, valid_time):
        values = self._observed_on_model_grid(valid_time)
        return np.roll(values, self._displacement, axis=1)

    def surface_field(self, valid_time, variable):
        base = {"temperature_2m": 290.0, "dewpoint_2m": 285.0,
                "wind_speed_10m": 5.0}[variable]
        return np.full(self._shape, base + self._surface_offset)

    def precipitation_accumulation(self, valid_time):
        from gpuwm.verify.obs.contracts import parse_valid_time
        from datetime import datetime

        hours = (parse_valid_time(valid_time)
                 - datetime.fromisoformat(INIT)).total_seconds() / 3600.0
        return np.full(self._shape, self._rain_rate * max(hours, 0.0))

    def record(self):
        return {"reader": "in-memory test arm"}


def _reflectivity_obs():
    return stubs.StubGriddedObsSource(
        acknowledgement=stubs.STUB_ACKNOWLEDGEMENT,
        quantity="composite_reflectivity", center_latitude=CENTER_LAT,
        center_longitude=CENTER_LON, shape=(280, 280), spacing_deg=0.01)


def _precipitation_obs():
    return stubs.StubGriddedObsSource(
        acknowledgement=stubs.STUB_ACKNOWLEDGEMENT,
        quantity="precipitation_accumulation", center_latitude=CENTER_LAT,
        center_longitude=CENTER_LON, shape=(280, 280), spacing_deg=0.01,
        peak_value=30.0, background_value=0.0)


def _station_fixture(arm, hours):
    source = stubs.StubStationObsSource(
        acknowledgement=stubs.STUB_ACKNOWLEDGEMENT,
        center_latitude=CENTER_LAT, center_longitude=CENTER_LON,
        station_count=16, span_deg=0.6, elevation_m=300.0)
    observations = source.observations(hours)
    grid = arm.grid()
    ny, nx = grid.shape
    positions = {}
    for station in observations.stations:
        # A plain index lookup on the regular test mesh stands in for the
        # science core's projection: this file is testing the harness, not
        # the projection.
        x = (station.longitude - float(grid.longitude[0, 0])) / 0.03
        y = (station.latitude - float(grid.latitude[0, 0])) / 0.03
        positions[station.station_id] = StationPosition(
            station.station_id, float(np.clip(x, 0.0, nx - 1)),
            float(np.clip(y, 0.0, ny - 1)))
    frozen = freeze_station_set(
        observations.stations, positions, observations=observations,
        valid_times=hours, interior_mask=np.ones(grid.shape, dtype=bool),
        land_mask=np.ones(grid.shape, dtype=bool),
        terrain_m=grid.terrain_m, elevation_tolerance_m=100.0,
        minimum_reporting_fraction=0.8, match_tolerance_seconds=600,
        maximum_screen_fraction=0.05)
    return observations, frozen


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def test_a_registration_hashes_its_own_pins_and_validates():
    registration = _registration()
    assert reg_mod.validate_registration(registration) == registration
    assert registration["rule_status"] == reg_mod.UNRATIFIED


def test_a_tampered_pin_invalidates_the_registration_hash():
    registration = _registration()
    registration["parameters"]["reflectivity"]["primary_threshold_dbz"] = 20.0
    with pytest.raises(ValueError, match="hash does not match"):
        reg_mod.validate_registration(registration)


def test_the_primary_must_be_one_of_the_scored_cells():
    with pytest.raises(ValueError, match="primary threshold"):
        reg_mod.reflectivity_parameters(primary_threshold_dbz=35.0)
    with pytest.raises(ValueError, match="primary neighborhood"):
        reg_mod.reflectivity_parameters(primary_half_width=7)


def test_the_ratification_status_travels_with_the_registration():
    ratified = _registration(ratification="owner ruling 2026-08-03")
    assert ratified["rule_status"] == reg_mod.RATIFIED
    assert reg_mod.validate_registration(ratified)["rule_status"] == \
        reg_mod.RATIFIED


def test_a_rule_status_that_disagrees_with_its_pins_is_refused():
    registration = _registration()
    registration["rule_status"] = reg_mod.RATIFIED
    with pytest.raises(ValueError, match="disagrees with its promotion pins"):
        reg_mod.validate_registration(registration)


def test_the_scored_window_reports_spinup_separately():
    leads = reg_mod.scored_lead_hours()
    assert leads[0] == reg_mod.DEFAULT_SPINUP_END_HOUR
    assert leads[-1] == reg_mod.DEFAULT_SCORE_END_HOUR
    assert len(leads) == 17


# --------------------------------------------------------------------------
# the interior mask
# --------------------------------------------------------------------------


def test_the_interior_mask_excludes_the_boundary_and_the_rim():
    mask = battery.interior_mask((60, 80), boundary_width_cells=5,
                                 rim_m=45000.0, dx_m=3000.0)
    assert mask.shape == (60, 80)
    # 5 boundary rows + 15 rim cells = 20 excluded on every side.
    assert not mask[:20, :].any() and not mask[-20:, :].any()
    assert mask[20:40, 20:60].all()
    assert int(mask.sum()) == 20 * 40


def test_a_domain_with_no_interior_is_a_refusal():
    with pytest.raises(ValueError, match="no interior"):
        battery.interior_mask((20, 20), boundary_width_cells=5,
                              rim_m=45000.0, dx_m=3000.0)


def test_valid_times_are_seam_timestamps_at_the_requested_leads():
    assert battery.valid_times(INIT, [0, 2, 18]) == (
        "2026-08-03T12:00:00", "2026-08-03T14:00:00", "2026-08-04T06:00:00")


# --------------------------------------------------------------------------
# the scoring pass
# --------------------------------------------------------------------------


def test_scoring_one_arm_produces_a_complete_score_file(tmp_path):
    obs = _reflectivity_obs()
    arm = MemoryArm(obs, displacement=2)
    hours = list(battery.valid_times(INIT, LEADS))
    observations, frozen = _station_fixture(arm, hours)

    payload = battery.score_case_arm(
        registration=_registration(), case_id="case-01", arm_id="faithful",
        init_time=INIT, model=arm, reflectivity_obs=obs,
        boundary_width_cells=2, station_obs=observations,
        frozen_stations=frozen,
        precipitation_obs={1: _precipitation_obs()},
        rehash=lambda provenance: True)

    assert payload["schema"] == battery.SCORE_SCHEMA
    assert payload["case_id"] == "case-01"
    assert payload["registration_sha256"] == \
        _registration()["registration_sha256"]
    assert payload["scored_lead_hours"] == list(LEADS)
    assert 0.0 <= payload["primary_scalar"] <= 1.0
    assert set(payload["reflectivity"]["primary_by_lead"]) == {"2", "3", "4"}
    assert len(payload["reflectivity"]["leads"]) == len(LEADS)
    assert payload["surface"]["station_set"]["station_count"] >= 1
    assert payload["precipitation"]["windows"]
    assert set(payload["guardrails"]) == {
        "median_station_rmse:temperature_2m",
        "median_station_rmse:dewpoint_2m",
        "median_station_rmse:wind_speed_10m",
        "mean_fss:precipitation"}
    assert payload["model_reader"] == {"reader": "in-memory test arm"}
    # The entry-receipt number: how much of the scored interior the
    # observation could actually see, per lead and reduced.
    coverage = payload["reflectivity"]
    assert 0.0 < coverage["minimum_interior_valid_fraction"] <= 1.0
    assert coverage["minimum_interior_valid_fraction"] <=         coverage["mean_interior_valid_fraction"]
    assert all(0.0 < row["interior_valid_fraction"] <= 1.0
               for row in coverage["leads"])

    written = battery.write_score_file(tmp_path / "scores.json", payload)
    assert json.loads(written.read_text(encoding="utf-8"))["arm_id"] == \
        "faithful"


def test_the_score_file_records_that_it_was_built_on_stand_ins():
    obs = _reflectivity_obs()
    arm = MemoryArm(obs)
    payload = battery.score_case_arm(
        registration=_registration(), case_id="case-01", arm_id="faithful",
        init_time=INIT, model=arm, reflectivity_obs=obs,
        boundary_width_cells=2)
    assert payload["uses_stub_inputs"] is True
    assert all(record["is_stub"] for record in
               payload["observation_provenance"])
    assert all(row["observation_is_stub"] for row in
               payload["reflectivity"]["leads"])


def test_an_unperformed_rehash_counts_as_a_failed_one():
    obs = _reflectivity_obs()
    arm = MemoryArm(obs)
    payload = battery.score_case_arm(
        registration=_registration(), case_id="case-01", arm_id="faithful",
        init_time=INIT, model=arm, reflectivity_obs=obs,
        boundary_width_cells=2)
    assert payload["observation_rehash"]
    assert all(not record["matches"] and not record["rehash_performed"]
               for record in payload["observation_rehash"])


def test_a_better_arm_scores_higher_on_the_primary_scalar():
    obs = _reflectivity_obs()
    scores = []
    for displacement in (0, 3, 12):
        arm = MemoryArm(obs, displacement=displacement)
        payload = battery.score_case_arm(
            registration=_registration(), case_id="case-01",
            arm_id=f"shift{displacement}", init_time=INIT, model=arm,
            reflectivity_obs=obs, boundary_width_cells=2)
        scores.append(payload["primary_scalar"])
    assert scores[0] == pytest.approx(1.0)
    assert scores == sorted(scores, reverse=True)


def test_reported_leads_are_reported_and_cannot_reach_the_primary_scalar():
    obs = _reflectivity_obs()
    arm = MemoryArm(obs, displacement=2)
    with_spinup = battery.score_case_arm(
        registration=_registration(), case_id="case-01", arm_id="faithful",
        init_time=INIT, model=arm, reflectivity_obs=obs,
        boundary_width_cells=2, reported_lead_hours=(0, 1))
    without = battery.score_case_arm(
        registration=_registration(), case_id="case-01", arm_id="faithful",
        init_time=INIT, model=arm, reflectivity_obs=obs,
        boundary_width_cells=2)
    assert with_spinup["primary_scalar"] == without["primary_scalar"]
    assert set(with_spinup["reported_not_scored"]["primary_by_lead"]) == {
        "0", "1"}
    assert without["reported_not_scored"] is None


def test_a_lead_cannot_be_both_scored_and_reported():
    obs = _reflectivity_obs()
    arm = MemoryArm(obs)
    with pytest.raises(ValueError, match="both scored and reported"):
        battery.score_case_arm(
            registration=_registration(), case_id="case-01",
            arm_id="faithful", init_time=INIT, model=arm,
            reflectivity_obs=obs, boundary_width_cells=2,
            reported_lead_hours=(2,))


class _OutageArchive:
    """The stand-in archive, with named leads suffering an ingest outage.

    It answers exactly as the real source does when every frame inside the
    registered tolerance is mostly mask: the seam's own
    ``ObservedFractionBelowFloor``, carrying the candidates that failed.
    """

    def __init__(self, inner, *, outages, floor=0.9):
        self._inner = inner
        self._outages = {str(when) for when in outages}
        self._floor = float(floor)

    def quantity(self):
        return self._inner.quantity()

    def field(self, valid_time):
        if str(valid_time) in self._outages:
            raise ObservedFractionBelowFloor(
                f"every composite_reflectivity frame within 240 s of "
                f"{valid_time} observes less than {self._floor:.4g} of the "
                f"packed subdomain",
                valid_time=str(valid_time),
                minimum_observed_fraction=self._floor,
                candidates=[{"valid_time": str(valid_time),
                             "offset_seconds": 35.0,
                             "observed_fraction": 0.1586}])
        return self._inner.field(valid_time)


def test_a_lead_whose_whole_window_is_mask_is_excluded_and_named():
    obs = _reflectivity_obs()
    arm = MemoryArm(obs, displacement=2)
    outage = battery.valid_times(INIT, [3])[0]

    payload = battery.score_case_arm(
        registration=_registration(), case_id="case-01", arm_id="faithful",
        init_time=INIT, model=arm,
        reflectivity_obs=_OutageArchive(obs, outages=[outage]),
        boundary_width_cells=2)
    reflectivity = payload["reflectivity"]

    # Named, with what it saw -- never a silent drop.
    assert [row["lead_hours"] for row in reflectivity["excluded_leads"]] == [3]
    excluded = reflectivity["excluded_leads"][0]
    assert excluded["reason"] == "observed_fraction_below_floor"
    assert excluded["valid_time"] == outage
    assert excluded["minimum_observed_fraction"] == 0.9
    assert excluded["candidate_frames"][0]["observed_fraction"] == 0.1586
    # And the lead mean is over the leads that carried an observation.
    assert reflectivity["lead_hours_requested"] == [2, 3, 4]
    assert reflectivity["lead_hours_scored"] == [2, 4]
    assert set(reflectivity["primary_by_lead"]) == {"2", "4"}
    assert len(reflectivity["leads"]) == 2

    without = battery.score_case_arm(
        registration=_registration(leads=(2, 4)), case_id="case-01",
        arm_id="faithful", init_time=INIT, model=arm, reflectivity_obs=obs,
        boundary_width_cells=2)
    assert payload["primary_scalar"] == without["primary_scalar"]


def test_a_pass_with_every_lead_excluded_fails_rather_than_scoring_zero():
    obs = _reflectivity_obs()
    arm = MemoryArm(obs)
    outages = list(battery.valid_times(INIT, LEADS))
    with pytest.raises(ValueError, match="unscoreable case"):
        battery.score_case_arm(
            registration=_registration(), case_id="case-01",
            arm_id="faithful", init_time=INIT, model=arm,
            reflectivity_obs=_OutageArchive(obs, outages=outages),
            boundary_width_cells=2)


def test_a_case_with_no_outage_records_no_exclusion():
    obs = _reflectivity_obs()
    arm = MemoryArm(obs, displacement=2)
    payload = battery.score_case_arm(
        registration=_registration(), case_id="case-01", arm_id="faithful",
        init_time=INIT, model=arm, reflectivity_obs=obs,
        boundary_width_cells=2)
    assert payload["reflectivity"]["excluded_leads"] == []
    assert payload["reflectivity"]["lead_hours_scored"] == list(LEADS)


class _AskedFor:
    """One observation archive, plus a note of every frame asked of it.

    The first full scoring pass asked a 6-hourly product for hourly frames
    and the reader refused ten of them; what a test can hold onto is the
    question, so this records the valid times the scorer asks for and hands
    the call through unchanged.  ``hole`` refuses one valid time the way the
    real reader refuses a frame it does not hold.
    """

    def __init__(self, inner, *, hole=None):
        self._inner = inner
        self._hole = hole
        self.asked: list[str] = []

    def quantity(self):
        return self._inner.quantity()

    def field(self, valid_time):
        self.asked.append(str(valid_time))
        if self._hole is not None and str(valid_time) == str(self._hole):
            raise LookupError(
                f"no precipitation_accumulation frame within 1800 s of "
                f"{valid_time}; refusing rather than reaching further")
        return self._inner.field(valid_time)

    def leads(self, init_time=INIT):
        start = datetime.fromisoformat(init_time)
        return [int((datetime.fromisoformat(when) - start).total_seconds()
                    // 3600) for when in self.asked]


def _score_precipitation(sources, *, leads, windows):
    arm = MemoryArm(_precipitation_obs())
    grid = arm.grid()
    return battery.score_precipitation(
        registration=_registration(leads=leads, windows=windows),
        model=arm, obs_sources=sources, init_time=INIT,
        lead_hours=list(leads), grid=grid,
        model_scored_region=np.ones(grid.shape, dtype=bool),
        collected_provenance=[])


def test_an_accumulation_window_is_asked_for_only_where_it_closes():
    leads = tuple(range(2, 19))
    hourly = _AskedFor(_precipitation_obs())
    six_hourly = _AskedFor(_precipitation_obs())

    result = _score_precipitation({1: hourly, 6: six_hourly},
                                  leads=leads, windows=(1, 6))

    # A 6 h product holds a frame every 6 h. Asking it at lead 7 asks for a
    # frame that cannot exist, and the reader refused ten such leads on the
    # first real scoring pass.
    assert six_hourly.leads() == [6, 12, 18]
    # The 1 h window is unchanged: the alignment test is a no-op there.
    assert hourly.leads() == list(leads)
    assert {row["window_hours"] for row in result["windows"]} == {1, 6}
    assert [row["lead_hours"] for row in result["windows"]
            if row["window_hours"] == 6] == [6, 12, 18]
    assert 0.0 <= result["guardrails"]["mean_fss:precipitation"] <= 1.0


def test_a_window_wider_than_the_forecast_is_asked_for_nothing():
    six_hourly = _AskedFor(_precipitation_obs())
    with pytest.raises(ValueError, match="produced no accumulations"):
        _score_precipitation({6: six_hourly}, leads=(2, 3, 4), windows=(6,))
    assert six_hourly.asked == []


def test_a_missing_frame_at_an_aligned_lead_still_refuses_loudly():
    """Alignment is declarative; absence is not silently tolerated."""
    leads = tuple(range(2, 19))
    missing = battery.valid_times(INIT, [12])[0]
    six_hourly = _AskedFor(_precipitation_obs(), hole=missing)

    with pytest.raises(LookupError, match="refusing rather than reaching"):
        _score_precipitation({1: _AskedFor(_precipitation_obs()),
                              6: six_hourly}, leads=leads, windows=(1, 6))

    # It asked for the aligned lead and let the refusal out, rather than
    # walking past a hole in the archive.
    assert six_hourly.leads() == [6, 12]


def test_a_non_positive_accumulation_window_is_refused():
    registration = _registration(windows=(1,))
    registration["parameters"]["precipitation"]["window_hours"] = [0]
    arm = MemoryArm(_precipitation_obs())
    grid = arm.grid()
    with pytest.raises(ValueError, match="positive number of hours"):
        battery.score_precipitation(
            registration=registration, model=arm,
            obs_sources={0: _precipitation_obs()}, init_time=INIT,
            lead_hours=[2, 3], grid=grid,
            model_scored_region=np.ones(grid.shape, dtype=bool),
            collected_provenance=[])


def test_surface_scoring_refuses_to_freeze_its_own_station_set():
    obs = _reflectivity_obs()
    arm = MemoryArm(obs)
    hours = list(battery.valid_times(INIT, LEADS))
    observations, _frozen = _station_fixture(arm, hours)
    with pytest.raises(ValueError, match="freezing.*per arm"):
        battery.score_case_arm(
            registration=_registration(), case_id="case-01",
            arm_id="faithful", init_time=INIT, model=arm,
            reflectivity_obs=obs, boundary_width_cells=2,
            station_obs=observations)


def test_a_growing_surface_bias_shows_up_in_the_guardrail():
    obs = _reflectivity_obs()
    hours = list(battery.valid_times(INIT, LEADS))
    clean = MemoryArm(obs)
    observations, frozen = _station_fixture(clean, hours)

    # Offsets chosen past the point where the arm already runs warm, so each
    # step is unambiguously further from the reports rather than across them.
    guardrails = []
    for offset in (5.0, 10.0, 15.0):
        arm = MemoryArm(obs, surface_offset=offset)
        payload = battery.score_case_arm(
            registration=_registration(), case_id="case-01",
            arm_id=f"offset{offset:g}", init_time=INIT, model=arm,
            reflectivity_obs=obs, boundary_width_cells=2,
            station_obs=observations, frozen_stations=frozen)
        guardrails.append(
            payload["guardrails"]["median_station_rmse:temperature_2m"])
    assert guardrails == sorted(guardrails)
    assert guardrails[-1] > guardrails[0] + 5.0


def test_the_scorer_refuses_an_observation_source_of_the_wrong_quantity():
    arm = MemoryArm(_reflectivity_obs())
    with pytest.raises(ValueError, match="the registration scores"):
        battery.score_case_arm(
            registration=_registration(), case_id="case-01",
            arm_id="faithful", init_time=INIT, model=arm,
            reflectivity_obs=_precipitation_obs(), boundary_width_cells=2)


# --------------------------------------------------------------------------
# collecting arms across cases
# --------------------------------------------------------------------------


def test_collecting_an_arm_across_cases_refuses_a_duplicate_case():
    files = [
        {"arm_id": "faithful", "case_id": "case-01", "primary_scalar": 0.5,
         "guardrails": {"g": 1.0}},
        {"arm_id": "faithful", "case_id": "case-02", "primary_scalar": 0.6,
         "guardrails": {"g": 1.1}},
        {"arm_id": "patch", "case_id": "case-01", "primary_scalar": 0.7,
         "guardrails": {"g": 1.2}},
    ]
    assert battery.collect_primary(files, arm_id="faithful") == {
        "case-01": 0.5, "case-02": 0.6}
    assert battery.collect_guardrail(files, arm_id="faithful", name="g") == {
        "case-01": 1.0, "case-02": 1.1}
    with pytest.raises(ValueError, match="two score files"):
        battery.collect_primary(files + [files[0]], arm_id="faithful")
    with pytest.raises(ValueError, match="no score file carries"):
        battery.collect_primary(files, arm_id="absent")
