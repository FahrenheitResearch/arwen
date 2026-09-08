"""The acquisition must deliver the lake state it requests, on the same grid."""
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import pytest

from gpuwm import era5_acquisition, fetch
from gpuwm.ingest.grib import _NATIVE_LAKE_SPECS


def validation_records():
    time = datetime(2013, 5, 31, 12)
    grid = fetch.Grib1Grid(ni=2, nj=2, lat_first=1.0, lon_first=0.0,
                          lat_last=0.0, lon_last=1.0, di_deg=1.0, dj_deg=1.0)
    rows = [fetch.Grib1Record(130, 100, level, time, grid, 128, 98, "same-gds")
            for level in fetch.ERA5_PRESSURE_LEVELS_HPA]
    rows.extend(fetch.Grib1Record(parameter, level_type, level, time, grid,
                                 table, center, "same-gds")
                for center, table, parameter, level_type, level in _NATIVE_LAKE_SPECS)
    return time, rows


def validate(monkeypatch, rows, time):
    report = SimpleNamespace(ok=True)
    monkeypatch.setattr(fetch, "validate_era5_files", lambda *args, **kwargs: report)
    monkeypatch.setattr(fetch, "read_grib1_records", lambda path: rows)
    return era5_acquisition._validate("test.grib", times=(time,), area=None)


def test_new_request_includes_all_three_lake_fields_and_validates_their_identity(monkeypatch):
    time, rows = validation_records()
    template = fetch.era5_request_template(cycle=time, hours=6,
        area=fetch.Area(0.0, 0.0, 1.0, 1.0))
    variables = template["requests"][1]["request"]["variable"]
    assert {"lake_mix_layer_temperature", "lake_ice_temperature", "lake_ice_depth"} <= set(variables)
    assert validate(monkeypatch, rows, time).ok


@pytest.mark.parametrize("change, match", [
    ("missing", "exactly one LAKE_ICE_DEPTH"),
    ("duplicate", "exactly one LAKE_ICE_DEPTH"),
    ("wrong_table", "exactly one LAKE_ICE_DEPTH"),
    ("wrong_center", "exactly one LAKE_ICE_DEPTH"),
    ("wrong_time", "valid times differ"),
    ("different_origin", "atmospheric source grid"),
    ("different_scan", "atmospheric source grid"),
])
def test_atmospheric_completeness_cannot_hide_a_wrong_lake_provider(monkeypatch, change, match):
    time, rows = validation_records()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[-1])
    elif change == "wrong_table":
        rows[-1] = replace(rows[-1], table_version=128)
    elif change == "wrong_center":
        rows[-1] = replace(rows[-1], center=7)
    elif change == "wrong_time":
        rows[-1] = replace(rows[-1], valid_time=datetime(2013, 5, 31, 18))
    elif change == "different_origin":
        rows[-1] = replace(rows[-1], grid=replace(rows[-1].grid, lon_first=0.25))
    else:
        rows[-1] = replace(rows[-1], grid_definition_sha256="changed-scan")
    with pytest.raises(ValueError, match=match):
        validate(monkeypatch, rows, time)
