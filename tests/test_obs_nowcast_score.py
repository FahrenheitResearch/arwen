"""The nowcast registration, its receipt, and the persistence baseline.

The instrument is tested before it is wired into a run: a synthetic storm
placed exactly where the radar saw it must score 1, the same storm moved must
score less and must recover as the neighbourhood grows, and a lead with no
scan must come out pending rather than zero.  A zero here would be the one
failure nobody catches by reading the receipt, because zero is a number a
forecast can legitimately earn.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from gpuwm.verify.obs import fss, nowcast
from gpuwm.verify.obs.contracts import (
    ModelGrid, ObsGridField, ObsProvenance, ObservedFractionBelowFloor,
    format_valid_time,
)

ANALYSIS = "2026-09-10T12:15:00"
DX_M = 3000.0
SHAPE = (40, 40)


def _geometry():
    """A 40 x 40 lat/lon lattice near 35.2N 97.4W, spaced about 3 km."""
    degrees_lat = DX_M / 111000.0
    degrees_lon = DX_M / (111000.0 * np.cos(np.deg2rad(35.2)))
    rows = 35.2 + (np.arange(SHAPE[0]) - SHAPE[0] / 2.0) * degrees_lat
    cols = -97.4 + (np.arange(SHAPE[1]) - SHAPE[1] / 2.0) * degrees_lon
    longitude, latitude = np.meshgrid(cols, rows)
    return latitude, longitude


def _grid():
    latitude, longitude = _geometry()
    return ModelGrid(latitude=latitude, longitude=longitude, dx_m=DX_M)


def _storm(center_row: int, center_col: int, *, peak: float = 48.0,
           radius: float = 4.0) -> np.ndarray:
    """A smooth blob of reflectivity with a quiet -30 dBZ background."""
    rows, cols = np.indices(SHAPE)
    distance = np.hypot(rows - center_row, cols - center_col)
    field = peak - 6.0 * distance
    return np.maximum(field, -30.0).astype(np.float64)


def _provenance(tag: str) -> ObsProvenance:
    return ObsProvenance(source="mrms",
                         product="MergedReflectivityQCComposite_00.50",
                         uri=f"s3://noaa-mrms-pds/{tag}.grib2.gz",
                         sha256=f"{abs(hash(tag)) % (16 ** 16):016x}" * 4,
                         fetched_at="2026-09-10T13:30:00")


def _field(values: np.ndarray, valid_time: str, *,
           valid: np.ndarray | None = None) -> ObsGridField:
    latitude, longitude = _geometry()
    return ObsGridField(
        quantity="composite_reflectivity", valid_time=valid_time,
        values=np.asarray(values, dtype=np.float64),
        valid=(np.ones(SHAPE, dtype=bool) if valid is None
               else np.asarray(valid, dtype=bool)),
        latitude=latitude, longitude=longitude, units="dBZ",
        provenance=_provenance(valid_time))


class FakeModel:
    """A forecast that holds one composite per lead."""

    def __init__(self, composites: dict[int, np.ndarray]):
        self._composites = dict(composites)

    def lead_minutes(self):
        return tuple(sorted(self._composites))

    def composite(self, minutes: int) -> np.ndarray:
        return self._composites[int(minutes)]

    def record(self):
        return {"route": "fake", "leads": list(self.lead_minutes())}


class FakeObs:
    """An archive holding frames at exact valid times, and nothing else."""

    def __init__(self, frames: dict[str, ObsGridField], *,
                 below_floor: tuple[str, ...] = (),
                 unreachable: tuple[str, ...] = ()):
        self._frames = dict(frames)
        self._below_floor = set(below_floor)
        self._unreachable = set(unreachable)

    def field(self, valid_time: str) -> ObsGridField:
        if valid_time in self._unreachable:
            raise nowcast.ObservationsUnreachable(
                "the MRMS archive could not be reached: no network")
        if valid_time in self._below_floor:
            raise ObservedFractionBelowFloor(
                "every frame observes less than the floor",
                valid_time=valid_time, minimum_observed_fraction=0.9,
                candidates=[{"valid_time": valid_time,
                             "offset_seconds": 0.0,
                             "observed_fraction": 0.2}])
        if valid_time not in self._frames:
            raise LookupError(f"no composite_reflectivity frame at {valid_time}")
        return self._frames[valid_time]

    def record(self):
        return {"route": "fake", "frames": sorted(self._frames)}


def _registration(**overrides):
    parameters = nowcast.nowcast_parameters(
        **{"interior_rim_m": 0.0, "boundary_width_cells": 2, **overrides})
    return nowcast.make_nowcast_registration(
        evaluating_tree={"package": "gpuwm", "version": "2.7.5-test"},
        parameters=parameters)


def _lead_time(minutes: int) -> str:
    return nowcast.lead_valid_times(ANALYSIS, [minutes])[0]


def _score(model, observations, *, registration=None, now=None):
    return nowcast.score_nowcast(
        registration=registration or _registration(), analysis_time=ANALYSIS,
        model=model, observations=observations, grid=_grid(),
        now=now or datetime(2026, 9, 10, 14, 0, tzinfo=timezone.utc))


def _row(document, minutes):
    return next(row for row in document["leads"]
                if row["lead_minutes"] == minutes)


# -- registration -------------------------------------------------------

def test_registration_hashes_its_own_pins_and_refuses_a_changed_one():
    registration = _registration()
    assert registration["schema"] == nowcast.NOWCAST_REGISTRATION_SCHEMA
    assert nowcast.validate_nowcast_registration(registration) == registration
    tampered = dict(registration)
    tampered["parameters"] = dict(registration["parameters"],
                                  primary_threshold_dbz=20.0)
    with pytest.raises(ValueError, match="does not match its pins"):
        nowcast.validate_nowcast_registration(tampered)


def test_registered_defaults_are_the_leads_thresholds_and_boxes_asked_for():
    parameters = nowcast.nowcast_parameters()
    assert parameters["lead_minutes"] == [15, 30, 45, 60]
    assert parameters["thresholds_dbz"] == [20.0, 30.0, 40.0]
    assert parameters["neighborhood_half_widths_cells"] == [1, 4]
    assert parameters["primary_threshold_dbz"] == 30.0
    assert parameters["primary_half_width_cells"] == 1
    assert parameters["neighborhood_boundary"] == fss.ZERO_BOUNDARY
    assert fss.box_length_m(parameters["primary_half_width_cells"],
                            DX_M) == 9000.0
    assert fss.box_length_m(parameters["neighborhood_half_widths_cells"][1],
                            DX_M) == 27000.0


def test_a_primary_outside_the_scored_set_is_refused():
    with pytest.raises(ValueError, match="primary threshold"):
        nowcast.nowcast_parameters(primary_threshold_dbz=45.0)
    with pytest.raises(ValueError, match="primary neighbourhood"):
        nowcast.nowcast_parameters(primary_half_width=8)


def test_an_evaluating_tree_without_a_version_is_refused():
    with pytest.raises(ValueError, match="evaluating tree record needs"):
        nowcast.make_nowcast_registration(
            evaluating_tree={"package": "gpuwm"})


# -- the receipt --------------------------------------------------------

def test_the_receipt_carries_every_field_the_contract_promises():
    truth = _storm(20, 20)
    observations = FakeObs({ANALYSIS: _field(truth, ANALYSIS),
                            _lead_time(15): _field(truth, _lead_time(15))})
    document = _score(FakeModel({15: truth}), observations)
    assert document["schema"] == nowcast.NOWCAST_SCORE_SCHEMA
    assert document["analysis_time"] == ANALYSIS
    assert document["registration_sha256"] == _registration()["registration_sha256"]
    assert document["evaluating_tree"]["version"] == "2.7.5-test"
    assert document["primary"]["threshold_dbz"] == 30.0
    assert document["primary"]["box_length_m"] == 9000.0
    assert document["grid"]["dx_m"] == DX_M
    row = _row(document, 15)
    assert row["valid_time"] == _lead_time(15)
    assert row["scan"]["sha256"] == observations.field(
        _lead_time(15)).provenance.sha256
    assert row["scan"]["offset_seconds"] == 0.0
    assert 0.0 <= row["observed_coverage_fraction"] <= 1.0
    # Every registered threshold x box cell is published, not only the primary.
    assert len(row["fss"]) == 6
    assert {entry["box_length_m"] for entry in row["fss"]} == {9000.0, 27000.0}
    assert row["persistence"]["primary_fss"] is not None
    assert document["primary_by_lead"] == {"15": row["primary_fss"]}
    assert document["difference_by_lead"] == {"15": row["difference_primary"]}


def test_a_lead_the_forecast_never_reached_is_absent_rather_than_unscored():
    truth = _storm(20, 20)
    frames = {ANALYSIS: _field(truth, ANALYSIS)}
    for minutes in (15, 30, 45, 60):
        frames[_lead_time(minutes)] = _field(truth, _lead_time(minutes))
    document = _score(FakeModel({15: truth, 30: truth}), FakeObs(frames))
    assert [row["lead_minutes"] for row in document["leads"]] == [15, 30]


# -- the score itself ---------------------------------------------------

def test_an_exact_match_scores_one():
    truth = _storm(20, 20)
    observations = FakeObs({ANALYSIS: _field(truth, ANALYSIS),
                            _lead_time(15): _field(truth, _lead_time(15))})
    document = _score(FakeModel({15: truth}), observations)
    row = _row(document, 15)
    assert row["status"] == nowcast.SCORED
    assert row["primary_fss"] == pytest.approx(1.0)
    assert all(entry["fss"] == pytest.approx(1.0) for entry in row["fss"])


def test_fss_falls_with_displacement_and_rises_with_the_box():
    truth = _storm(20, 20)
    observations = FakeObs({ANALYSIS: _field(truth, ANALYSIS),
                            _lead_time(15): _field(truth, _lead_time(15))})
    scores = {}
    for shift in (0, 3, 8):
        document = _score(FakeModel({15: _storm(20, 20 + shift)}), observations)
        row = _row(document, 15)
        scores[shift] = {(entry["threshold_model"], entry["half_width_cells"]):
                         entry["fss"] for entry in row["fss"]}
    small = (30.0, 1)
    large = (30.0, 4)
    assert scores[0][small] > scores[3][small] > scores[8][small]
    for shift in (3, 8):
        assert scores[shift][large] > scores[shift][small]


def test_a_forecast_of_nothing_against_a_storm_scores_zero_and_says_scored():
    truth = _storm(20, 20)
    observations = FakeObs({ANALYSIS: _field(truth, ANALYSIS),
                            _lead_time(15): _field(truth, _lead_time(15))})
    quiet = np.full(SHAPE, -30.0)
    row = _row(_score(FakeModel({15: quiet}), observations), 15)
    assert row["status"] == nowcast.SCORED
    assert row["primary_fss"] == pytest.approx(0.0)


# -- the persistence baseline -------------------------------------------

def test_a_stationary_storm_gives_persistence_one_and_a_moved_one_less():
    truth = _storm(20, 20)
    still = FakeObs({ANALYSIS: _field(truth, ANALYSIS),
                     _lead_time(15): _field(truth, _lead_time(15))})
    stationary = _row(_score(FakeModel({15: truth}), still), 15)
    assert stationary["persistence"]["primary_fss"] == pytest.approx(1.0)
    assert stationary["persistence"]["scan"]["valid_time"] == ANALYSIS

    moved = _storm(20, 26)
    drifting = FakeObs({ANALYSIS: _field(truth, ANALYSIS),
                        _lead_time(15): _field(moved, _lead_time(15))})
    row = _row(_score(FakeModel({15: moved}), drifting), 15)
    assert row["persistence"]["primary_fss"] < 1.0
    assert row["persistence"]["primary_fss"] < stationary[
        "persistence"]["primary_fss"]


def test_the_difference_is_the_model_minus_persistence_at_the_same_lead():
    analysis_truth = _storm(20, 20)
    lead_truth = _storm(20, 26)
    observations = FakeObs({ANALYSIS: _field(analysis_truth, ANALYSIS),
                            _lead_time(15): _field(lead_truth, _lead_time(15))})
    # A forecast that advected the storm correctly must beat the last scan.
    row = _row(_score(FakeModel({15: lead_truth}), observations), 15)
    assert row["primary_fss"] == pytest.approx(1.0)
    assert row["difference_primary"] == pytest.approx(
        row["primary_fss"] - row["persistence"]["primary_fss"])
    assert row["difference_primary"] > 0.0

    # A forecast that left the storm where it was can only equal the scan.
    stale = _row(_score(FakeModel({15: analysis_truth}), observations), 15)
    assert stale["difference_primary"] == pytest.approx(0.0)


def test_model_and_persistence_score_the_identical_cells():
    truth = _storm(20, 20)
    partial = np.ones(SHAPE, dtype=bool)
    partial[:, :12] = False
    observations = FakeObs({
        ANALYSIS: _field(truth, ANALYSIS, valid=partial),
        _lead_time(15): _field(truth, _lead_time(15))})
    row = _row(_score(FakeModel({15: truth}), observations), 15)
    persistence_cells = {entry["scored_cells"]
                         for entry in row["persistence"]["fss"]}
    model_cells = {entry["scored_cells"] for entry in row["fss"]}
    assert persistence_cells == model_cells


# -- leads that cannot be scored ----------------------------------------

def test_a_future_lead_is_pending_and_not_a_zero():
    truth = _storm(20, 20)
    observations = FakeObs({ANALYSIS: _field(truth, ANALYSIS)})
    document = _score(
        FakeModel({15: truth, 60: truth}), observations,
        now=datetime(2026, 9, 10, 12, 40, tzinfo=timezone.utc))
    row = _row(document, 60)
    assert row["status"] == nowcast.PENDING
    assert "has not arrived" in row["reason"]
    assert "primary_fss" not in row
    # 15 is pending too, for the other reason: its scan is not in the archive
    # yet.  Both are pending and neither is a zero, which is the whole point.
    assert document["leads_pending"] == [15, 60]
    assert document["primary_by_lead"] == {}


def test_an_unpublished_scan_inside_the_grace_is_pending():
    truth = _storm(20, 20)
    observations = FakeObs({ANALYSIS: _field(truth, ANALYSIS)})
    document = _score(FakeModel({15: truth}), observations,
                      now=datetime(2026, 9, 10, 12, 35, tzinfo=timezone.utc))
    row = _row(document, 15)
    assert row["status"] == nowcast.PENDING
    assert "publication grace" in row["reason"]


def test_a_scan_absent_long_after_its_time_is_missing_obs():
    truth = _storm(20, 20)
    observations = FakeObs({ANALYSIS: _field(truth, ANALYSIS)})
    document = _score(FakeModel({15: truth}), observations,
                      now=datetime(2026, 9, 11, tzinfo=timezone.utc))
    row = _row(document, 15)
    assert row["status"] == nowcast.MISSING_OBS
    assert document["leads_missing_obs"] == [15]
    assert document["primary_by_lead"] == {}


def test_a_lead_under_the_coverage_floor_is_missing_obs_with_its_candidates():
    truth = _storm(20, 20)
    observations = FakeObs({ANALYSIS: _field(truth, ANALYSIS)},
                           below_floor=(_lead_time(15),))
    row = _row(_score(FakeModel({15: truth}), observations), 15)
    assert row["status"] == nowcast.MISSING_OBS
    assert row["minimum_observed_fraction"] == 0.9
    assert row["candidate_frames"][0]["observed_fraction"] == 0.2


def test_an_unreachable_archive_is_unavailable_with_the_reason():
    truth = _storm(20, 20)
    observations = FakeObs({}, unreachable=(ANALYSIS, _lead_time(15)))
    document = _score(FakeModel({15: truth}), observations)
    row = _row(document, 15)
    assert row["status"] == nowcast.UNAVAILABLE
    assert "no network" in row["reason"]
    assert document["leads_unavailable"] == [15]
    assert document["analysis_scan"]["status"] == nowcast.UNAVAILABLE


def test_a_missing_analysis_scan_still_scores_the_model_and_says_why():
    truth = _storm(20, 20)
    observations = FakeObs({_lead_time(15): _field(truth, _lead_time(15))})
    row = _row(_score(FakeModel({15: truth}), observations), 15)
    assert row["status"] == nowcast.SCORED
    assert row["primary_fss"] == pytest.approx(1.0)
    assert row["persistence"]["status"] != nowcast.SCORED
    assert row["difference_primary"] is None


# -- the compact form a status document carries -------------------------

def test_the_lead_summary_carries_all_three_numbers_per_lead():
    analysis_truth = _storm(20, 20)
    lead_truth = _storm(20, 26)
    observations = FakeObs({ANALYSIS: _field(analysis_truth, ANALYSIS),
                            _lead_time(15): _field(lead_truth, _lead_time(15))})
    document = _score(FakeModel({15: lead_truth, 30: lead_truth}), observations)
    rows = nowcast.lead_summary(document)
    assert [row["lead_minutes"] for row in rows] == [15, 30]
    scored = rows[0]
    assert scored["status"] == nowcast.SCORED
    assert scored["primary_fss"] is not None
    assert scored["persistence_primary_fss"] is not None
    assert scored["difference_primary"] == pytest.approx(
        scored["primary_fss"] - scored["persistence_primary_fss"])
    assert "reason" in rows[1]


def test_the_scored_region_excludes_the_boundary_and_the_registered_rim():
    registration = nowcast.make_nowcast_registration(
        evaluating_tree={"package": "gpuwm", "version": "2.7.5-test"},
        parameters=nowcast.nowcast_parameters(
            interior_rim_m=9000.0, boundary_width_cells=2))
    region = nowcast.scored_region(SHAPE, registration, DX_M)
    assert region.shape == SHAPE
    # two boundary rows plus three rim rows on every side
    assert int(np.count_nonzero(region)) == (40 - 10) ** 2
    assert not region[4].any() and region[5, 5]


def test_the_seam_timestamps_of_the_registered_leads():
    start = datetime.fromisoformat(ANALYSIS)
    assert nowcast.lead_valid_times(ANALYSIS, [15, 60]) == (
        format_valid_time(start + timedelta(minutes=15)),
        format_valid_time(start + timedelta(minutes=60)))


def test_a_clear_box_scores_one_and_the_compact_row_says_the_box_was_empty():
    """The first fresh run on this build produced exactly this.

    No cell of the scored interior reached 30 dBZ in any scan, so the
    fractions the score compares were zero everywhere, the convention gave
    FSS 1 to the model and 1 to persistence, and the difference was 0.0.
    The three numbers a status document publishes were then
    indistinguishable from a forecast that put the storms in the right
    place.  The base rates beside them are what tells a reader the box was
    empty, which is why they are in the compact row and not only in the
    receipt.
    """
    quiet = np.full(SHAPE, -20.0)
    lead = _lead_time(15)
    document = _score(FakeModel({15: quiet}),
                      FakeObs({ANALYSIS: _field(quiet, ANALYSIS),
                               lead: _field(quiet, lead)}))
    row = document["leads"][0]
    assert row["status"] == nowcast.SCORED
    assert row["primary_fss"] == pytest.approx(1.0)
    assert row["persistence"]["primary_fss"] == pytest.approx(1.0)
    assert row["difference_primary"] == pytest.approx(0.0)
    assert row["primary_observed_base_rate"] == pytest.approx(0.0)

    compact = nowcast.lead_summary(document)[0]
    assert compact["primary_fss"] == pytest.approx(1.0)
    assert compact["primary_observed_base_rate"] == pytest.approx(0.0)
    assert compact["primary_model_base_rate"] == pytest.approx(0.0)


def test_the_compact_row_carries_the_base_rates_of_a_real_echo():
    truth = _storm(20, 20)
    lead = _lead_time(15)
    document = _score(FakeModel({15: truth}),
                      FakeObs({ANALYSIS: _field(truth, ANALYSIS),
                               lead: _field(truth, lead)}))
    compact = nowcast.lead_summary(document)[0]
    assert compact["primary_observed_base_rate"] > 0.0
    assert compact["primary_model_base_rate"] > 0.0
    assert compact["primary_observed_base_rate"] == pytest.approx(
        document["leads"][0]["primary_observed_base_rate"])
    # An unscored lead has no base rate to carry and must say so with None
    # rather than a zero, which is the value a real empty box reports.
    pending = nowcast.lead_summary(
        _score(FakeModel({15: truth, 60: truth}),
               FakeObs({ANALYSIS: _field(truth, ANALYSIS),
                        lead: _field(truth, lead)}),
               now=datetime(2026, 9, 10, 12, 35, tzinfo=timezone.utc)))[1]
    assert pending["status"] == nowcast.PENDING
    assert pending["primary_observed_base_rate"] is None
    assert pending["primary_model_base_rate"] is None
