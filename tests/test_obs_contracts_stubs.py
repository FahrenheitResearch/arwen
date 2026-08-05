"""The observation seam refuses what it cannot honestly score.

Two halves.  The first pins the seam's refusals: an observed field that
carries a NaN where a mask belongs, a longitude nobody wrapped, a value that
cannot be the quantity it claims, a station report with an unknown variable.
Each of those has exactly one honest reading, and the seam takes it before a
score is computed on top.

The second half pins the stand-ins.  The ingest lane is being built beside
this one, so the scorer's fixtures are manufactured -- and the entire design
of that is to make a manufactured field impossible to mistake for an
observation: it cannot be built without saying so out loud, it warns when it
is, and the flag it sets is what the promotion evaluator refuses.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.verify.obs import stubs
from gpuwm.verify.obs.contracts import (
    ObsGridField, ObsProvenance, Station, StationObsSet, StationReport,
    normalize_longitude, parse_valid_time,
)

DIGEST = "a" * 64


def _provenance(**overrides):
    fields = {"source": "TEST", "product": "test-product",
              "uri": "test://object", "sha256": DIGEST,
              "fetched_at": "2026-08-03T00:00:00"}
    fields.update(overrides)
    return ObsProvenance(**fields)


def _field(**overrides):
    lat, lon = stubs.regular_grid(center_latitude=37.0,
                                  center_longitude=-97.0, shape=(6, 6),
                                  spacing_deg=0.01)
    fields = {
        "quantity": "composite_reflectivity",
        "valid_time": "2026-08-03T12:00:00",
        "values": np.zeros((6, 6), dtype=np.float64),
        "valid": np.ones((6, 6), dtype=bool),
        "latitude": lat, "longitude": lon, "units": "dBZ",
        "provenance": _provenance(),
    }
    fields.update(overrides)
    return ObsGridField(**fields)


# --------------------------------------------------------------------------
# the seam
# --------------------------------------------------------------------------


def test_a_well_formed_field_is_accepted():
    field = _field()
    assert field.shape == (6, 6)
    assert field.provenance.is_stub is False


def test_a_missing_value_belongs_under_the_mask_not_in_the_data():
    values = np.zeros((6, 6))
    values[2, 2] = np.nan
    with pytest.raises(ValueError, match="belongs under the valid mask"):
        _field(values=values)


def test_units_are_pinned_not_negotiated():
    with pytest.raises(ValueError, match="seam pins"):
        _field(units="dbZ")


def test_an_unknown_quantity_is_refused():
    with pytest.raises(ValueError, match="unknown gridded quantity"):
        _field(quantity="brightness_temperature")


def test_a_value_that_cannot_be_the_quantity_is_refused():
    values = np.zeros((6, 6))
    values[0, 0] = 5000.0
    with pytest.raises(ValueError, match="outside the seam bound"):
        _field(values=values)


def test_an_unwrapped_longitude_is_refused():
    _lat, lon = stubs.regular_grid(center_latitude=37.0,
                                   center_longitude=-97.0, shape=(6, 6),
                                   spacing_deg=0.01)
    with pytest.raises(ValueError, match="not wrapped"):
        _field(longitude=lon + 360.0)


def test_values_must_arrive_as_float64():
    with pytest.raises(ValueError, match="float64 at the seam"):
        _field(values=np.zeros((6, 6), dtype=np.float32))


def test_out_of_range_values_under_the_mask_are_not_read():
    values = np.zeros((6, 6))
    values[0, 0] = -9999.0
    mask = np.ones((6, 6), dtype=bool)
    mask[0, 0] = False
    assert _field(values=values, valid=mask).values[0, 0] == -9999.0


def test_a_stub_provenance_must_say_why_and_a_real_one_may_not():
    with pytest.raises(ValueError, match="must say why"):
        _provenance(is_stub=True)
    with pytest.raises(ValueError, match="may not carry a stub reason"):
        _provenance(stub_reason="because")


def test_a_provenance_digest_must_be_a_real_digest():
    with pytest.raises(ValueError, match="64 hex"):
        _provenance(sha256="not-a-digest")


def test_a_zoned_timestamp_is_refused_rather_than_interpreted():
    with pytest.raises(ValueError, match="UTC ISO-8601"):
        parse_valid_time("2026-08-03T12:00:00Z")


def test_a_station_report_refuses_an_unknown_variable_and_a_nan():
    with pytest.raises(ValueError, match="unknown surface variable"):
        StationReport(station_id="A", valid_time="2026-08-03T12:00:00",
                      values={"ceiling": 1000.0})
    with pytest.raises(ValueError, match="absent key, not a NaN"):
        StationReport(station_id="A", valid_time="2026-08-03T12:00:00",
                      values={"temperature_2m": float("nan")})


def test_an_observation_set_refuses_a_report_from_an_unknown_station():
    station = Station(station_id="A", latitude=37.0, longitude=-97.0,
                      elevation_m=300.0)
    with pytest.raises(ValueError, match="absent from the set"):
        StationObsSet(
            stations=(station,),
            reports=(StationReport(station_id="B",
                                   valid_time="2026-08-03T12:00:00",
                                   values={"temperature_2m": 290.0}),),
            provenance=_provenance())


def test_longitudes_wrap_into_the_seam_range():
    assert normalize_longitude(np.array([180.0, 200.0, -190.0])).tolist() == [
        -180.0, -160.0, 170.0]


# --------------------------------------------------------------------------
# the stand-ins
# --------------------------------------------------------------------------


def test_a_stand_in_cannot_be_built_without_saying_so():
    with pytest.raises(ValueError, match="refusing to manufacture"):
        stubs.StubGriddedObsSource(
            acknowledgement="", quantity="composite_reflectivity",
            center_latitude=37.0, center_longitude=-97.0, shape=(20, 20))


def test_building_a_stand_in_warns():
    with pytest.warns(stubs.ObsStubWarning, match="not measurements"):
        stubs.StubGriddedObsSource(
            acknowledgement=stubs.STUB_ACKNOWLEDGEMENT,
            quantity="composite_reflectivity", center_latitude=37.0,
            center_longitude=-97.0, shape=(20, 20))


def test_a_stand_in_field_is_flagged_at_every_level(recwarn):
    source = stubs.StubGriddedObsSource(
        acknowledgement=stubs.STUB_ACKNOWLEDGEMENT,
        quantity="composite_reflectivity", center_latitude=37.0,
        center_longitude=-97.0, shape=(40, 40), spacing_deg=0.02)
    field = source.field("2026-08-03T14:00:00")
    assert field.provenance.is_stub is True
    assert field.provenance.stub_reason
    assert field.provenance.source == stubs.STUB_SOURCE
    assert field.provenance.uri.startswith("stand-in://")
    assert stubs.uses_stub(field.provenance)
    assert field.provenance.record()["is_stub"] is True


def test_stand_in_fields_differ_between_hours_and_carry_a_coverage_hole():
    source = stubs.StubGriddedObsSource(
        acknowledgement=stubs.STUB_ACKNOWLEDGEMENT,
        quantity="composite_reflectivity", center_latitude=37.0,
        center_longitude=-97.0, shape=(60, 60), spacing_deg=0.02)
    early = source.field("2026-08-03T12:00:00")
    late = source.field("2026-08-03T18:00:00")
    assert not np.array_equal(early.values, late.values)
    assert not early.valid.all()
    assert (early.values >= 30.0).any()


def test_the_stand_in_station_archive_reports_and_flags_itself():
    source = stubs.StubStationObsSource(
        acknowledgement=stubs.STUB_ACKNOWLEDGEMENT, center_latitude=37.0,
        center_longitude=-97.0, station_count=9, span_deg=1.0)
    observations = source.observations(["2026-08-03T12:00:00",
                                        "2026-08-03T13:00:00"])
    assert observations.provenance.is_stub is True
    assert len(observations.stations) == 9
    grouped = observations.by_station()
    assert set(grouped) == {station.station_id
                            for station in observations.stations}
    assert all(reports for reports in grouped.values())
    # The deliberately impossible station is there for the terrain rule.
    elevations = sorted(station.elevation_m
                        for station in observations.stations)
    assert elevations[-1] > elevations[0] + 1000.0
