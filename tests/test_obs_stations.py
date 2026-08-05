"""Surface verification against point observations.

The failure mode this file exists for is a matcher that never actually reads
a location and still produces plausible RMSEs.  So the interpolation is
pinned against a field whose value at every point is known in closed form,
the admission rules are each shown to fire on the case they are for, and the
station-shuffle mutation is shown to be a derangement rather than "a shuffle
that might leave everyone in place".
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.obs.contracts import Station, StationObsSet, StationReport
from gpuwm.verify.obs.stations import (
    BILINEAR, DROP_NOT_LAND, DROP_OUTSIDE_DOMAIN, DROP_OUTSIDE_INTERIOR,
    DROP_REPORTING, DROP_SCREEN, DROP_TERRAIN_MISMATCH, NEAREST,
    StationPosition, freeze_station_set, match_reports, sample_field,
    screen_report, shuffle_positions, surface_scores,
)

DIGEST = "b" * 64


def _provenance():
    from gpuwm.verify.obs.contracts import ObsProvenance

    return ObsProvenance(source="TEST", product="surface", uri="test://obs",
                         sha256=DIGEST, fetched_at="2026-08-03T00:00:00")


def _hours(count=4, start_hour=12):
    return [f"2026-08-03T{start_hour + index:02d}:00:00"
            for index in range(count)]


# --------------------------------------------------------------------------
# interpolation
# --------------------------------------------------------------------------


def test_bilinear_reproduces_a_bilinear_field_exactly():
    ny, nx = 9, 11
    j, i = np.indices((ny, nx)).astype(np.float64)
    field = 3.0 + 2.0 * i - 0.5 * j + 0.25 * i * j
    for x, y in ((0.0, 0.0), (4.25, 3.5), (10.0, 8.0), (7.75, 0.5)):
        expected = 3.0 + 2.0 * x - 0.5 * y + 0.25 * x * y
        assert sample_field(field, StationPosition("S", x, y),
                            method=BILINEAR) == pytest.approx(expected)


def test_nearest_and_bilinear_differ_off_a_gridpoint():
    field = np.arange(25, dtype=np.float64).reshape(5, 5)
    position = StationPosition("S", 1.4, 2.6)
    assert (sample_field(field, position, method=NEAREST)
            != sample_field(field, position, method=BILINEAR))
    assert sample_field(field, StationPosition("S", 2.0, 3.0),
                        method=NEAREST) == field[3, 2]


def test_sampling_outside_the_grid_is_a_refusal_not_a_clamp():
    field = np.zeros((5, 5))
    with pytest.raises(ValueError, match="outside the"):
        sample_field(field, StationPosition("S", 5.5, 2.0))


# --------------------------------------------------------------------------
# quality screen
# --------------------------------------------------------------------------


def test_the_gross_range_screen_fires_on_each_registered_bound():
    def report(**values):
        return StationReport(station_id="S", valid_time="2026-08-03T12:00:00",
                             values=values)

    assert screen_report(report(temperature_2m=290.0,
                                dewpoint_2m=285.0)) == ()
    assert screen_report(report(temperature_2m=400.0)) == ("temperature_2m",)
    assert screen_report(report(wind_speed_10m=120.0)) == ("wind_speed_10m",)
    # Supersaturation is charged to the dewpoint, which is where it lives.
    assert screen_report(report(temperature_2m=290.0,
                                dewpoint_2m=295.0)) == ("dewpoint_2m",)


# --------------------------------------------------------------------------
# time matching
# --------------------------------------------------------------------------


def test_matching_takes_the_nearest_report_inside_the_tolerance():
    station = Station("S", 37.0, -97.0, 300.0)
    reports = tuple(
        StationReport(station_id="S", valid_time=stamp,
                      values={"temperature_2m": value})
        for stamp, value in (("2026-08-03T11:52:00", 288.0),
                             ("2026-08-03T11:58:00", 289.0),
                             ("2026-08-03T12:20:00", 291.0)))
    observations = StationObsSet(stations=(station,), reports=reports,
                                 provenance=_provenance())
    matched = match_reports(observations, ["2026-08-03T12:00:00"],
                            tolerance_seconds=600)
    assert matched[("S", "2026-08-03T12:00:00")].values["temperature_2m"] == 289.0


def test_a_hour_with_no_report_inside_the_tolerance_is_simply_missing():
    station = Station("S", 37.0, -97.0, 300.0)
    observations = StationObsSet(
        stations=(station,),
        reports=(StationReport(station_id="S",
                               valid_time="2026-08-03T12:30:00",
                               values={"temperature_2m": 290.0}),),
        provenance=_provenance())
    assert match_reports(observations, ["2026-08-03T12:00:00"],
                         tolerance_seconds=600) == {}


def test_an_equidistant_tie_resolves_to_the_earlier_report():
    station = Station("S", 37.0, -97.0, 300.0)
    reports = tuple(
        StationReport(station_id="S", valid_time=stamp,
                      values={"temperature_2m": value})
        for stamp, value in (("2026-08-03T11:55:00", 288.0),
                             ("2026-08-03T12:05:00", 292.0)))
    observations = StationObsSet(stations=(station,), reports=reports,
                                 provenance=_provenance())
    matched = match_reports(observations, ["2026-08-03T12:00:00"],
                            tolerance_seconds=600)
    assert matched[("S", "2026-08-03T12:00:00")].valid_time == "2026-08-03T11:55:00"


# --------------------------------------------------------------------------
# admission
# --------------------------------------------------------------------------


def _admission_case(*, hours, terrain_value=300.0, screen_values=None,
                    report_hours=None):
    """One station per drop reason, plus one that should survive."""
    stations = (
        Station("KEEP", 37.0, -97.0, terrain_value),
        Station("FAR", 37.5, -96.5, terrain_value),
        Station("RIM", 37.1, -97.1, terrain_value),
        Station("SEA", 37.2, -97.2, terrain_value),
        Station("HIGH", 37.3, -97.3, terrain_value + 500.0),
        Station("BAD", 37.4, -97.4, terrain_value),
        Station("QUIET", 36.9, -96.9, terrain_value),
    )
    positions = {
        "KEEP": StationPosition("KEEP", 10.0, 10.0),
        "FAR": StationPosition("FAR", 40.0, 40.0),
        "RIM": StationPosition("RIM", 1.0, 1.0),
        "SEA": StationPosition("SEA", 11.0, 11.0),
        "HIGH": StationPosition("HIGH", 12.0, 12.0),
        "BAD": StationPosition("BAD", 13.0, 13.0),
        "QUIET": StationPosition("QUIET", 14.0, 14.0),
    }
    interior = np.zeros((20, 20), dtype=bool)
    interior[3:17, 3:17] = True
    land = np.ones((20, 20), dtype=bool)
    land[11, 11] = False
    terrain = np.full((20, 20), terrain_value, dtype=np.float64)

    reports = []
    for station in stations:
        stamps = report_hours if (report_hours and station.station_id == "QUIET"
                                  ) else hours
        for stamp in stamps:
            values = {"temperature_2m": 290.0, "dewpoint_2m": 285.0,
                      "wind_speed_10m": 5.0}
            if station.station_id == "BAD":
                values["temperature_2m"] = 400.0
            reports.append(StationReport(station_id=station.station_id,
                                         valid_time=stamp, values=values))
    observations = StationObsSet(stations=stations, reports=tuple(reports),
                                 provenance=_provenance())
    return stations, positions, interior, land, terrain, observations


def test_every_admission_rule_fires_on_the_station_it_is_for():
    hours = _hours(5)
    (stations, positions, interior, land, terrain,
     observations) = _admission_case(hours=hours, report_hours=hours[:1])

    frozen = freeze_station_set(
        stations, positions, observations=observations, valid_times=hours,
        interior_mask=interior, land_mask=land, terrain_m=terrain,
        elevation_tolerance_m=100.0, minimum_reporting_fraction=0.8,
        match_tolerance_seconds=600, maximum_screen_fraction=0.05)

    assert frozen.station_ids == ("KEEP",)
    reasons = {str(drop["station_id"]): str(drop["reason"])
               for drop in frozen.drops}
    assert reasons == {
        "FAR": DROP_OUTSIDE_DOMAIN,
        "RIM": DROP_OUTSIDE_INTERIOR,
        "SEA": DROP_NOT_LAND,
        "HIGH": DROP_TERRAIN_MISMATCH,
        "BAD": DROP_SCREEN,
        "QUIET": DROP_REPORTING,
    }
    record = frozen.record()
    assert record["station_count"] == 1
    assert record["dropped_count"] == 6
    assert record["parameters"]["elevation_tolerance_m"] == 100.0


def test_a_station_is_dropped_for_terrain_never_corrected():
    hours = _hours(3)
    (stations, positions, interior, land, terrain,
     observations) = _admission_case(hours=hours)
    frozen = freeze_station_set(
        stations, positions, observations=observations, valid_times=hours,
        interior_mask=interior, land_mask=land, terrain_m=terrain,
        elevation_tolerance_m=100.0, minimum_reporting_fraction=0.8,
        match_tolerance_seconds=600, maximum_screen_fraction=0.05)
    assert "HIGH" not in frozen.station_ids
    assert "HIGH" not in frozen.positions
    detail = [drop["detail"] for drop in frozen.drops
              if drop["station_id"] == "HIGH"][0]
    assert detail.startswith("-500") or detail.startswith("+500")


# --------------------------------------------------------------------------
# scores
# --------------------------------------------------------------------------


def _scoring_fixture(hours):
    stations = tuple(
        Station(f"S{index}", 37.0 + index * 0.1, -97.0, 300.0)
        for index in range(4))
    positions = {station.station_id:
                 StationPosition(station.station_id, 5.0 + index, 5.0)
                 for index, station in enumerate(stations)}
    reports = tuple(
        StationReport(station_id=station.station_id, valid_time=stamp,
                      values={"temperature_2m": 290.0})
        for station in stations for stamp in hours)
    observations = StationObsSet(stations=stations, reports=reports,
                                 provenance=_provenance())
    interior = np.ones((20, 20), dtype=bool)
    land = np.ones((20, 20), dtype=bool)
    terrain = np.full((20, 20), 300.0)
    frozen = freeze_station_set(
        stations, positions, observations=observations, valid_times=hours,
        interior_mask=interior, land_mask=land, terrain_m=terrain,
        elevation_tolerance_m=100.0, minimum_reporting_fraction=0.8,
        match_tolerance_seconds=600, maximum_screen_fraction=0.05)
    matched = match_reports(observations, hours, tolerance_seconds=600)
    return frozen, matched


def test_bias_and_rmse_are_computed_over_station_by_hour():
    hours = _hours(3)
    frozen, matched = _scoring_fixture(hours)
    assert len(frozen.station_ids) == 4

    def model_value(station_id, valid_time, variable):
        # Constant +2 K warm bias everywhere.
        return 292.0

    scores = surface_scores(frozen, matched=matched, model_value=model_value,
                            valid_times=hours, variables=["temperature_2m"])
    score = scores["temperature_2m"]
    assert score.bias == pytest.approx(2.0)
    assert score.rmse == pytest.approx(2.0)
    assert score.median_station_rmse == pytest.approx(2.0)
    assert score.sample_count == 12
    assert score.station_count == 4
    assert sorted(score.hourly_bias) == [12, 13, 14]


def test_the_guardrail_is_the_median_over_stations_not_the_pooled_rmse():
    hours = _hours(3)
    frozen, matched = _scoring_fixture(hours)
    # One station is wildly wrong; the median must not follow it, the pooled
    # RMSE must.  That difference is the whole reason the guardrail is a
    # median.
    def model_value(station_id, valid_time, variable):
        return 320.0 if station_id == "S0" else 291.0

    score = surface_scores(frozen, matched=matched, model_value=model_value,
                           valid_times=hours,
                           variables=["temperature_2m"])["temperature_2m"]
    assert score.median_station_rmse == pytest.approx(1.0)
    assert score.rmse > 10.0


def test_a_missing_model_frame_is_skipped_not_scored_as_zero():
    hours = _hours(3)
    frozen, matched = _scoring_fixture(hours)

    def model_value(station_id, valid_time, variable):
        return None if valid_time.endswith("13:00:00") else 291.0

    score = surface_scores(frozen, matched=matched, model_value=model_value,
                           valid_times=hours,
                           variables=["temperature_2m"])["temperature_2m"]
    assert score.sample_count == 8
    assert score.bias == pytest.approx(1.0)


def test_scoring_refuses_a_variable_with_no_matched_pairs():
    hours = _hours(2)
    frozen, matched = _scoring_fixture(hours)
    with pytest.raises(ValueError, match="no matched"):
        surface_scores(frozen, matched=matched,
                       model_value=lambda *_args: 5.0, valid_times=hours,
                       variables=["wind_speed_10m"])


# --------------------------------------------------------------------------
# the mutation control's mutation
# --------------------------------------------------------------------------


def test_the_station_shuffle_is_a_derangement_and_is_deterministic():
    positions = {f"S{index}": StationPosition(f"S{index}", float(index), 0.0)
                 for index in range(8)}
    shuffled = shuffle_positions(positions, seed=17)
    again = shuffle_positions(positions, seed=17)

    assert set(shuffled) == set(positions)
    assert all(shuffled[key].x != positions[key].x for key in positions)
    assert sorted(item.x for item in shuffled.values()) == sorted(
        item.x for item in positions.values())
    assert {key: value.x for key, value in shuffled.items()} == {
        key: value.x for key, value in again.items()}


def test_a_shuffle_needs_somebody_to_swap_with():
    with pytest.raises(ValueError, match="at least two stations"):
        shuffle_positions({"S": StationPosition("S", 1.0, 1.0)}, seed=1)
