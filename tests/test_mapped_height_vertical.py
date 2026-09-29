"""Height interfaces are a generic coordinate, independent of source identity."""
from dataclasses import replace
from copy import deepcopy
from datetime import timedelta
import json

import numpy as np
import pytest

from gpuwm import mapped_source as ms
import test_mapped_hybrid_vertical as fixture


def height_fixture():
    mapping = fixture._hybrid_mapping()
    mapping["name"] = "arbitrary-height-coordinate"
    mapping["coordinates"]["vertical"] = {
        "kind": "model_level", "units": "1", "positive": "down",
        "levels": [1, 2, 3], "interface_levels": [7, 9, 12, 18],
    }
    mapping["target"]["pressure_requirement"] = "air_pressure"
    fields = mapping["fields"]
    fields["air_pressure"].pop("derivation")
    fields["air_pressure"]["selectors"] = [fixture._grib2_selector(0, 3, 0)]
    fields["geopotential_height"]["derivation"] = "midpoint-height"
    fields["coordinate_height"] = fixture._hybrid_field(
        "coordinate_height", "m", ["half_level", "y", "x"], "mass",
        fixture._grib2_selector(0, 3, 6))
    fields["coordinate_height"]["time_binding"] = "cycle_invariant"
    mapping["derivations"] = [{"name": "midpoint-height",
        "operation": "height_from_interfaces", "source": "coordinate_height"}]
    records = fixture._records(pv=None)
    records = [replace(r, values=np.full_like(r.values, 0.004))
               if r.category == 1 and r.parameter == 0 else r for r in records]
    for parameter, name in [(22, "cloud_water_mixing_ratio"),
                            (24, "rain_water_mixing_ratio")]:
        fields[name] = fixture._hybrid_field(name, "kg kg-1",
            ["vertical", "y", "x"], "mass", fixture._grib2_selector(0, 1, parameter))
        mapping["target"]["initialization_policies"].pop(name)
        for k in range(1, 4):
            records.append(replace(fixture._record("air_temperature", parameter*10+k, k,
                pv=None, base=k*0.0001), category=1, parameter=parameter))
    for k, pressure in enumerate([4000.0, 40000.0, 95000.0], 1):
        records.append(replace(fixture._record("air_temperature", 100+k, k,
            pv=None, base=pressure), category=3, parameter=0))
    # Each column has its own terrain displacement. No column is flattened.
    terrain = np.arange(fixture.NY * fixture.NX).reshape(fixture.NY, fixture.NX)
    for k, height in zip([7, 9, 12, 18], [23000.0, 19000.0, 3000.0, 0.0]):
        records.append(replace(fixture._record("air_temperature", 200+k, k,
            pv=None, base=height), category=3, parameter=6, values=height+terrain))
    records += [replace(r, index=r.index+1000,
        valid_time=r.valid_time+timedelta(hours=1)) for r in records if r.parameter != 6]
    moisture = ["specific_humidity", "cloud_water_mixing_ratio", "rain_water_mixing_ratio"]
    raw_names = ["vapor_fraction", "cloud_fraction", "rain_fraction"]
    for name, raw_name in zip(moisture, raw_names):
        fields[raw_name] = deepcopy(fields[name])
        fields[name].pop("selectors")
        fields[name]["derivation"] = "rebase-"+name
        mapping["derivations"].append({"name": "rebase-"+name,
            "operation": "mass_fraction_rebase", "source": raw_name,
            "exclude": raw_names[1:] if name == "specific_humidity" else raw_names})
    # A mass layer is bounded by two fixed surfaces, independently of the
    # interface coordinate's own record labels.
    for field in fields.values():
        if field.get("selectors") and field["source_axes"] == ["vertical", "y", "x"]:
            selector = field["selectors"][0]
            field["selectors"] = [dict(selector, level_type=150, level_value=k,
                second_level_type=150, second_level_value=k+1) for k in (1, 2, 3)]
    records = [replace(r, level_type=150, second_level_type=150,
                       second_level_value=r.level_value+1)
               if r.level_type == 105 and not (r.category == 3 and r.parameter == 6)
               else r for r in records]
    return mapping, records


def materialize(tmp_path, mapping, records):
    path = tmp_path / "height.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    mapping = ms.load_mapping(path)
    collection = ms._assemble_grib(mapping, records)
    return ms._materialize_frames(mapping, collection,
        mapping_sha256="a"*64, input_sha256={"synthetic.grib2": "b"*64})


def test_arbitrary_height_interfaces_preserve_columns_and_forecast_times(tmp_path):
    mapping, records = height_fixture()
    frames = materialize(tmp_path, mapping, records)
    assert len(frames) == 2
    for frame in frames:
        fields = frame.fields
        terrain = np.arange(fixture.NY * fixture.NX).reshape(fixture.NY, fixture.NX)
        expected = np.array([21000.0, 11000.0, 1500.0])[:, None, None] + terrain
        np.testing.assert_array_equal(fields["geopotential_height"].values, expected)
        assert fields["air_pressure"].values[0, 0, 0] == 4000.0
    snapshots = ms.mapped_frames_to_regular_snapshots(frames,
        initialize_absent_hydrometeors=True)
    assert len(snapshots) == 2
    for snapshot in snapshots:
        assert np.all(snapshot.fields["QC"] > 0)
        assert np.all(snapshot.fields["QR"] > 0)
    fields = frames[0].fields
    vapor = fields["specific_humidity"].values
    for k in range(3):
        dry = 1.0-0.004-2*(k+1)*0.0001
        np.testing.assert_allclose(vapor[k]/(1-vapor[k]), 0.004/dry)
        np.testing.assert_allclose(fields["cloud_water_mixing_ratio"].values[k], (k+1)*0.0001/dry)


def test_missing_interface_refuses_instead_of_shortening_column(tmp_path):
    mapping, records = height_fixture()
    records = [r for r in records if r.index != 218]
    with pytest.raises(ValueError, match="coverage mismatch"):
        materialize(tmp_path, mapping, records)


def test_crossing_interfaces_refuse_instead_of_inverting_layers(tmp_path):
    mapping, records = height_fixture()
    records = [replace(r, values=np.full_like(r.values, 25000.0))
        if r.index == 212 else r for r in records]
    with pytest.raises(ValueError, match="crossing layers"):
        materialize(tmp_path, mapping, records)


def test_interface_count_must_bound_mass_levels(tmp_path):
    mapping, records = height_fixture()
    mapping["coordinates"]["vertical"]["interface_levels"].pop()
    with pytest.raises(ValueError, match="N\\+1"):
        materialize(tmp_path, mapping, records)


def test_excluded_mass_cannot_consume_the_reference_mass(tmp_path):
    mapping, records = height_fixture()
    records = [replace(r, values=np.ones_like(r.values))
        if r.parameter == 22 else r for r in records]
    with pytest.raises(ValueError, match="no positive reference mass"):
        materialize(tmp_path, mapping, records)


def test_era_ladder_cannot_shorten_mass_levels_under_fixed_interfaces(tmp_path):
    mapping, _ = height_fixture()
    mapping["coordinates"]["vertical"]["era_ladders"] = [[1, 2]]
    path = tmp_path / "mixed-coordinate.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ValueError, match="era_ladders.*interface_levels.*shorten"):
        ms.load_mapping(path)


RAW_FRACTIONS = ("vapor_fraction", "cloud_fraction", "rain_fraction")


def dependency_only_fixture():
    """The height fixture with its raw fractions declared inputs only."""
    mapping, records = height_fixture()
    for name in RAW_FRACTIONS:
        mapping["fields"][name]["dependency_only"] = True
    return mapping, records


def test_dependency_only_inputs_feed_their_derivations_and_stay_off_the_frame(tmp_path):
    """A raw input written beside what is derived from it is stream nobody reads.

    Named breakage: ICON-D2's six raw mass fractions were 390 of the 1,217
    layers each valid time wrote, about 16 GB of a 48 h 1 km run.
    """
    mapping, records = height_fixture()
    published = materialize(tmp_path, mapping, records)
    marked, records = dependency_only_fixture()
    (tmp_path / "marked").mkdir()
    frames = materialize(tmp_path / "marked", marked, records)
    assert len(frames) == len(published) == 2
    for frame, reference in zip(frames, published):
        assert not set(RAW_FRACTIONS) & set(frame.fields)
        assert set(RAW_FRACTIONS) <= set(reference.fields)
        assert tuple(frame.fields) == tuple(
            name for name in reference.fields if name not in RAW_FRACTIONS)
        for name, field in frame.fields.items():
            np.testing.assert_array_equal(field.values, reference.fields[name].values)
        described = [row.canonical_name for row in frame.header.fields]
        assert described == list(frame.fields)


def test_a_required_field_cannot_be_held_off_the_frame(tmp_path):
    mapping, _ = dependency_only_fixture()
    mapping["fields"]["air_temperature"]["dependency_only"] = True
    path = tmp_path / "withheld.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\['air_temperature'\] are required by "
                       "the target and marked dependency_only"):
        ms.load_mapping(path)


def test_dependency_only_is_a_boolean(tmp_path):
    mapping, _ = dependency_only_fixture()
    mapping["fields"]["vapor_fraction"]["dependency_only"] = "yes"
    path = tmp_path / "spelled.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ValueError, match="vapor_fraction.dependency_only must be true or false"):
        ms.load_mapping(path)
