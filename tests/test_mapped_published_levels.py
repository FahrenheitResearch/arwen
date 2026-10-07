"""A field published on part of the ladder decodes on the whole ladder.

NCEP's GFS/GDAS pgrb2.0p25 carries its isobaric state on 33 levels, 1 to
1000 hPa, and its five hydrometeor masses (CLMR, ICMR, RWMR, SNMR, GRLE)
on 22 of them: 50 to 1000 hPa without 70 hPa (measured on
gfs.t18z.pgrb2.0p25.f003 of 2026-10-01).  One ladder is chosen for every
stacked field at once, so declaring the masses plainly would refuse every
file or drop temperature, humidity and wind above 50 hPa.
``fields.<name>.published_levels`` is the table row that says which
levels a field is published on; on the others the field is zero, and a
published level a file leaves out still refuses by name.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gpuwm.mapped_source import (PUBLISHED_LEVELS_ZERO_FIELDS, _assemble_grib,
                                 _GribRecord, load_mapping)
from gpuwm.source_authorities import packaged_authorities

GDAS_PROFILE = "gdas-pgrb2-0p25-grib2-v1"
CYCLE = datetime(2026, 10, 1, 18)
NY, NX = 3, 4

#: The isobaric levels (Pa) GFS pgrb2.0p25 writes CLMR/ICMR/RWMR/SNMR/GRLE
#: on, read from the 2026-10-01 18Z f003 inventory.
GFS_HYDROMETEOR_LEVELS = (
    5000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0, 35000.0, 40000.0,
    45000.0, 50000.0, 55000.0, 60000.0, 65000.0, 70000.0, 75000.0, 80000.0,
    85000.0, 90000.0, 92500.0, 95000.0, 97500.0, 100000.0,
)
HYDROMETEORS = {
    "cloud_water_mixing_ratio": 22, "cloud_ice_mixing_ratio": 23,
    "rain_water_mixing_ratio": 24, "snow_mixing_ratio": 25,
    "graupel_or_hail_mixing_ratio": 32,
}


def _record(index: int, selector: dict, level: float) -> _GribRecord:
    return _GribRecord(
        source=Path("/staged/gfs.pgrb2.0p25"),
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
        values=np.full((NY, NX), float(level) * 1e-9 + 1e-6),
        latitude=np.linspace(40.0, 38.0, NY),
        longitude=np.linspace(-100.0, -97.0, NX),
        grid_fingerprint="grid",
    )


def _gdas_mapping() -> dict:
    return load_mapping(packaged_authorities(GDAS_PROFILE)["mapping"])


def _stacked_fields(mapping: dict) -> list[str]:
    return [name for name, field in mapping["fields"].items()
            if field.get("derivation") is None
            and "vertical" in field["source_axes"]]


def _records(mapping: dict, levels_by_field: dict[str, tuple[float, ...]]):
    records = []
    for name, levels in levels_by_field.items():
        selector = mapping["fields"][name]["selectors"][0]
        for level in levels:
            records.append(_record(len(records), selector, level))
    return records


def _field(collection, name: str):
    return next(value for value in collection.direct.values()
                if value.name == name)


def _publication(mapping: dict, *, hydrometeor_levels=GFS_HYDROMETEOR_LEVELS):
    ladder = tuple(float(level) for level in
                   mapping["coordinates"]["vertical"]["levels"])
    return {name: (hydrometeor_levels if name in HYDROMETEORS else ladder)
            for name in _stacked_fields(mapping)}


def test_the_packaged_gdas_mapping_declares_the_five_masses_gfs_writes():
    mapping = _gdas_mapping()
    for name, parameter in HYDROMETEORS.items():
        field = mapping["fields"][name]
        assert field["selectors"] == [{
            "format": "grib2", "discipline": 0, "category": 1,
            "parameter": parameter, "level_type": 100}]
        assert field["units"] == {"source": "kg kg-1", "target": "kg kg-1"}
        assert field["published_levels"] == {
            "levels": [int(level) for level in GFS_HYDROMETEOR_LEVELS],
            "absent": "zero"}
        assert name in {row["name"] for row in mapping["target"]["required_fields"]}
        assert name not in mapping["target"]["initialization_policies"]
    # The state ladder is unchanged: 33 levels, 1 to 1000 hPa.
    levels = mapping["coordinates"]["vertical"]["levels"]
    assert len(levels) == 33 and levels[0] == 100 and levels[-1] == 100000
    assert "era_ladders" not in mapping["coordinates"]["vertical"]


def test_a_gfs_publication_decodes_every_field_on_the_whole_ladder():
    mapping = _gdas_mapping()
    collection = _assemble_grib(mapping, _records(mapping, _publication(mapping)))
    ladder = tuple(float(level) for level in
                   mapping["coordinates"]["vertical"]["levels"])
    assert tuple(collection.vertical_values) == ladder
    temperature = _field(collection, "air_temperature")
    assert temperature.values.shape == (33, NY, NX)
    assert np.isfinite(temperature.values).all()
    unpublished = [index for index, level in enumerate(ladder)
                   if level not in GFS_HYDROMETEOR_LEVELS]
    assert [ladder[index] for index in unpublished] == [
        100.0, 200.0, 300.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0, 3000.0,
        4000.0, 7000.0]
    for name in HYDROMETEORS:
        value = _field(collection, name)
        assert value.values.shape == (33, NY, NX)
        assert value.missing_count == 0
        assert (value.values[unpublished] == 0.0).all()
        published = [index for index in range(33) if index not in unpublished]
        assert (value.values[published] > 0.0).all()
        assert len(value.references) == 22


def test_a_published_level_the_file_leaves_out_still_refuses_by_name():
    mapping = _gdas_mapping()
    gappy = tuple(level for level in GFS_HYDROMETEOR_LEVELS if level != 50000.0)
    with pytest.raises(ValueError) as refusal:
        _assemble_grib(mapping, _records(
            mapping, _publication(mapping, hydrometeor_levels=gappy)))
    assert "vertical coverage mismatch; missing=[50000.0], extra=[]" \
        in str(refusal.value)


def test_a_record_on_an_unpublished_level_is_read():
    mapping = _gdas_mapping()
    with_70 = tuple(sorted({*GFS_HYDROMETEOR_LEVELS, 7000.0}))
    collection = _assemble_grib(mapping, _records(
        mapping, _publication(mapping, hydrometeor_levels=with_70)))
    ladder = list(collection.vertical_values)
    cloud = _field(collection, "cloud_water_mixing_ratio")
    assert cloud.values[ladder.index(7000.0), 0, 0] > 0.0
    assert cloud.values[ladder.index(4000.0), 0, 0] == 0.0


def test_the_other_fields_decode_byte_identically_with_and_without_the_masses():
    """Declaring the masses moves no other field: the T/q/wind/height arrays
    of a frame decoded by the packaged mapping equal the ones the same
    mapping decodes with the five fields taken back out."""
    mapping = _gdas_mapping()
    without = copy.deepcopy(mapping)
    for name in HYDROMETEORS:
        del without["fields"][name]
    records = _records(mapping, _publication(mapping))
    full = _assemble_grib(mapping, records)
    bare = _assemble_grib(without, records)
    for name in ("air_temperature", "specific_humidity", "eastward_wind",
                 "northward_wind", "geopotential_height"):
        assert _field(full, name).values.tobytes() == _field(bare, name).values.tobytes()
    assert tuple(full.vertical_values) == tuple(bare.vertical_values)


# ---------------------------------------------------------------- validation

def _document(**changes) -> dict:
    document = json.loads(
        packaged_authorities(GDAS_PROFILE)["mapping"].read_text(encoding="utf-8"))
    for path, value in changes.items():
        node = document
        keys = path.split("/")
        for key in keys[:-1]:
            node = node[key]
        if value is None:
            del node[keys[-1]]
        else:
            node[keys[-1]] = value
    return document


def _load(tmp_path, document):
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return load_mapping(path)


def test_the_declaration_is_restricted_to_the_hydrometeor_masses(tmp_path):
    assert set(PUBLISHED_LEVELS_ZERO_FIELDS) == set(HYDROMETEORS)
    document = _document()
    document["fields"]["air_temperature"]["published_levels"] = {
        "levels": [5000, 10000], "absent": "zero"}
    with pytest.raises(ValueError, match="restricted to the hydrometeor mass"):
        _load(tmp_path, document)


@pytest.mark.parametrize("declaration, message", [
    ({"levels": [5000, 7], "absent": "zero"}, "does not declare"),
    ({"levels": [], "absent": "zero"}, "non-empty"),
    ({"levels": [5000, 5000], "absent": "zero"}, "unique"),
    ({"levels": [5000], "absent": "nan"}, "must be 'zero'"),
    ({"levels": [5000]}, "absent"),
])
def test_a_malformed_declaration_refuses_by_name(tmp_path, declaration, message):
    document = _document()
    document["fields"]["snow_mixing_ratio"]["published_levels"] = declaration
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, document)


def test_a_declaration_naming_every_level_changes_nothing_and_refuses(tmp_path):
    document = _document()
    document["fields"]["snow_mixing_ratio"]["published_levels"]["levels"] = (
        document["coordinates"]["vertical"]["levels"])
    with pytest.raises(ValueError, match="would change nothing"):
        _load(tmp_path, document)


def test_the_published_levels_keep_the_reject_policy(tmp_path):
    document = _document()
    document["fields"]["snow_mixing_ratio"]["missing"] = {
        "kind": "value", "value": 0.0}
    with pytest.raises(ValueError, match="reject missing policy"):
        _load(tmp_path, document)
