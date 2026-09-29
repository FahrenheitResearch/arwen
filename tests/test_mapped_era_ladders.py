"""A product publication carrying fewer levels decodes on its era ladder.

ECMWF's open-data IFS files carry 14 pressure levels, 10 to 1000 hPa, and
earlier publications of the same product carry 13, without 10 hPa.
``vertical.era_ladders`` is the table that says which whole ladders other
publications of a product carry; the decoders build the column from the
largest declared ladder every stacked field carries in full, and a file
missing a level no declared ladder omits still refuses at that level.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.mapped_source import _assemble_grib, _GribRecord, load_mapping
from gpuwm.source_authorities import packaged_authorities
from gpuwm.source_frame import VerticalDescriptor

IFS_PROFILE = "ecmwf-open-data-oper-grib2-v1"
CYCLE = datetime(2024, 5, 17, 0)
NY, NX = 3, 4

#: The pressure levels (Pa) an earlier open-data file carries, read from
#: its ECMWF index: 13 levels, no 10 hPa.
EARLIER_OPEN_DATA_LEVELS = (
    5000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0, 40000.0,
    50000.0, 60000.0, 70000.0, 85000.0, 92500.0, 100000.0,
)
#: The same product's current files add 10 hPa on top.
CURRENT_OPEN_DATA_LEVELS = (1000.0, *EARLIER_OPEN_DATA_LEVELS)


def _record(index: int, selector: dict, level: float) -> _GribRecord:
    return _GribRecord(
        source=Path("/staged/open-data.grib2"),
        index=index,
        reference_time=CYCLE,
        valid_time=CYCLE,
        member=None,
        parameter=int(selector["parameter"]),
        level_type=int(selector["level_type"]),
        level_value=float(level),
        table_version=None,
        center=None,
        subcenter=None,
        master_table_version=None,
        local_table_version=None,
        discipline=int(selector["discipline"]),
        category=int(selector["category"]),
        second_level_type=None,
        second_level_value=None,
        process_identity=None,
        time_semantics=(0,),
        values=np.full((NY, NX), float(level)),
        latitude=np.linspace(40.0, 38.0, NY),
        longitude=np.linspace(-100.0, -97.0, NX),
        grid_fingerprint="grid",
    )


def _stacked_fields(mapping: dict) -> list[str]:
    return [
        name for name, field in mapping["fields"].items()
        if field.get("derivation") is None
        and "vertical" in field["source_axes"]
    ]


def _records(mapping: dict, levels_by_field: dict[str, tuple[float, ...]]):
    records = []
    for name, levels in levels_by_field.items():
        selector = mapping["fields"][name]["selectors"][0]
        for level in levels:
            records.append(_record(len(records), selector, level))
    return records


def _ifs_mapping() -> dict:
    return load_mapping(packaged_authorities(IFS_PROFILE)["mapping"])


def _field(collection, name: str):
    return next(value for value in collection.direct.values()
                if value.name == name)


def test_the_packaged_ifs_mapping_decodes_a_publication_without_10_hpa():
    mapping = _ifs_mapping()
    stacked = _stacked_fields(mapping)
    assert set(stacked) == {
        "geopotential_height", "air_temperature", "specific_humidity",
        "eastward_wind", "northward_wind",
    }
    collection = _assemble_grib(mapping, _records(
        mapping, {name: EARLIER_OPEN_DATA_LEVELS for name in stacked}))
    assert tuple(collection.vertical_values) == EARLIER_OPEN_DATA_LEVELS
    temperature = _field(collection, "air_temperature")
    assert temperature.values.shape == (13, NY, NX)
    assert temperature.values[0, 0, 0] == 5000.0
    assert temperature.values[-1, 0, 0] == 100000.0


def test_the_packaged_ifs_mapping_keeps_all_14_levels_of_a_current_file():
    mapping = _ifs_mapping()
    collection = _assemble_grib(mapping, _records(
        mapping,
        {name: CURRENT_OPEN_DATA_LEVELS for name in _stacked_fields(mapping)}))
    assert tuple(collection.vertical_values) == CURRENT_OPEN_DATA_LEVELS
    assert _field(collection, "air_temperature").values.shape == (14, NY, NX)


def test_a_level_every_ifs_publication_carries_still_refuses_by_name():
    mapping = _ifs_mapping()
    gappy = tuple(level for level in EARLIER_OPEN_DATA_LEVELS if level != 50000.0)
    with pytest.raises(ValueError) as refusal:
        _assemble_grib(mapping, _records(
            mapping, {name: gappy for name in _stacked_fields(mapping)}))
    assert "vertical coverage mismatch; missing=[1000.0, 50000.0], extra=[]" \
        in str(refusal.value)


def test_a_level_only_some_fields_carry_is_left_out_of_every_field():
    mapping = _ifs_mapping()
    levels = {name: EARLIER_OPEN_DATA_LEVELS for name in _stacked_fields(mapping)}
    levels["air_temperature"] = CURRENT_OPEN_DATA_LEVELS
    records = _records(mapping, levels)
    collection = _assemble_grib(mapping, records)
    assert tuple(collection.vertical_values) == EARLIER_OPEN_DATA_LEVELS
    temperature = _field(collection, "air_temperature")
    assert temperature.values.shape == (13, NY, NX)
    stacked = {f"{record.source}:{record.index}" for record in records
               if record.category == 0 and record.parameter == 0
               and record.level_value != 1000.0}
    assert set(temperature.references) == stacked


def test_without_era_ladders_the_absent_level_refuses_as_it_always_did(tmp_path):
    raw = json.loads(
        packaged_authorities(IFS_PROFILE)["mapping"].read_text(encoding="utf-8"))
    del raw["coordinates"]["vertical"]["era_ladders"]
    path = tmp_path / "no-era.mapping.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    mapping = load_mapping(path)
    with pytest.raises(ValueError, match=r"vertical coverage mismatch; "
                                         r"missing=\[1000\.0\], extra=\[\]"):
        _assemble_grib(mapping, _records(
            mapping,
            {name: EARLIER_OPEN_DATA_LEVELS for name in _stacked_fields(mapping)}))


@pytest.mark.parametrize(("change", "message"), [
    (lambda vertical: vertical.update(era_ladders=[[5000, 7]]),
     r"names levels \[7\.0\] that vertical\.levels does not declare"),
    (lambda vertical: vertical.update(era_ladders=[[]]),
     r"era_ladders\[0\] must be a non-empty numeric list"),
    (lambda vertical: vertical.update(era_ladders=[[5000, 5000]]),
     r"era_ladders\[0\] must be a unique numeric list"),
    (lambda vertical: vertical.update(era_ladders=[]),
     r"era_ladders must be a non-empty list"),
    (lambda vertical: vertical.pop("levels"),
     r"era_ladders needs vertical\.levels"),
])
def test_era_ladders_are_held_to_the_declared_levels(tmp_path, change, message):
    raw = json.loads(
        packaged_authorities(IFS_PROFILE)["mapping"].read_text(encoding="utf-8"))
    change(raw["coordinates"]["vertical"])
    path = tmp_path / "era.mapping.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_mapping(path)


def _bundle(levels: tuple[float, ...]):
    header = SimpleNamespace(vertical_coordinates={
        "atmosphere": VerticalDescriptor(
            coordinate="pressure", level_count=len(levels),
            level_values=levels, positive="down", units="Pa"),
    })
    return SimpleNamespace(frames=(SimpleNamespace(header=header),))


def test_the_receipt_names_the_levels_the_files_did_not_carry():
    from gpuwm.mapped_composition import decoded_vertical_ladder

    mapping = _ifs_mapping()
    entry = decoded_vertical_ladder(_bundle(EARLIER_OPEN_DATA_LEVELS), mapping)
    assert entry == {
        "units": "Pa",
        "declared_levels": list(CURRENT_OPEN_DATA_LEVELS),
        "decoded_levels": list(EARLIER_OPEN_DATA_LEVELS),
        "absent_levels": [1000.0],
        "absent_level_count": 1,
    }
    assert decoded_vertical_ladder(
        _bundle(CURRENT_OPEN_DATA_LEVELS), mapping) is None


def test_one_warning_says_which_levels_the_column_lacks(capsys):
    from gpuwm.mapped_composition import decoded_vertical_ladder
    from gpuwm.mapped_direct import _announce_vertical_ladder

    mapping = _ifs_mapping()
    _announce_vertical_ladder(
        decoded_vertical_ladder(_bundle(EARLIER_OPEN_DATA_LEVELS), mapping))
    lines = [line for line in capsys.readouterr().err.splitlines()
             if line.startswith("warning:")]
    assert len(lines) == 1
    assert "13 of the 14 vertical levels" in lines[0]
    assert "absent: 10 hPa" in lines[0]


def test_era_ladders_on_a_netcdf_mapping_are_refused_as_unread():
    from gpuwm.mapped_source import _validate_era_ladders

    with pytest.raises(ValueError, match="read by the GRIB decoders only"):
        _validate_era_ladders(
            [list(EARLIER_OPEN_DATA_LEVELS)], CURRENT_OPEN_DATA_LEVELS, "netcdf")
    _validate_era_ladders(
        [list(EARLIER_OPEN_DATA_LEVELS)], CURRENT_OPEN_DATA_LEVELS, "grib1")
