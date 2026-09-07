"""Field-wise packing retains constructor checks without a decoded atmosphere."""
from dataclasses import replace
import gc
import json
import weakref

import numpy as np
import pytest

from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.mapped_source import _array_sha256, mapped_frames_to_regular_snapshots
import test_mapped_frameset_streaming as fixture
from test_mapped_snapshot_packing import _assert_same_snapshot


@pytest.fixture
def frame(monkeypatch):
    monkeypatch.setattr(fixture, "_NY", 4)
    monkeypatch.setattr(fixture, "_NX", 8)
    return fixture._one_frame()


def _stream(tmp_path, frame):
    directory = fixture.engine_bridge.write_frameset(tmp_path / "frames", (frame,))
    return fixture.engine_bridge.open_frameset(directory)


def _pack(frames, **kwargs):
    return mapped_frames_to_regular_snapshots(
        (frames.fieldwise_frame(0),), initialize_absent_hydrometeors=True,
        **kwargs)[0]


def _unused(frame):
    fields = dict(frame.fields)
    fields["unused_diagnostic"] = replace(
        fields["air_temperature"], name="unused_diagnostic")
    return replace(frame, fields=fields)


def test_fieldwise_pack_has_exact_owned_bytes_and_bounded_live_inputs(frame, tmp_path, monkeypatch):
    frame = _unused(frame)
    expected = mapped_frames_to_regular_snapshots(
        (frame,), initialize_absent_hydrometeors=True)[0]
    directory = _stream(tmp_path, frame).directory
    authority = tmp_path / "authority"
    authority.write_text("field-wise consumer witness", encoding="utf-8")
    bundle = fixture._bundle(directory, authority)
    frames = bundle.frames
    reads, references, live_bytes = [], [], []
    read = frames._read_field
    def measured(index, document):
        field = read(index, document)
        references.append(weakref.ref(field.values))
        live_bytes.append(sum(array.nbytes for ref in references
                              if (array := ref()) is not None))
        reads.append(field.name)
        return field
    monkeypatch.setattr(frames, "_read_field", measured)
    monkeypatch.setattr(frames, "_materialize", lambda *_: pytest.fail("whole frame read"))
    # The ordinary preparation consumer must select the field-wise route.
    actual = bundle.regular_snapshots()[0]
    _assert_same_snapshot(actual, expected)
    assert set(reads) == set(frame.fields)
    gc.collect()
    assert all(ref() is None for ref in references)
    # Pressure temporarily overlaps the first field; the soil pair is the
    # other intentional group. No source-field bank survives the copy.
    largest = max(field.values.nbytes for field in frame.fields.values())
    assert max(live_bytes) <= 2 * largest
    assert frames._cached_frame is None


@pytest.mark.parametrize("name", ["air_pressure", "unused_diagnostic"])
def test_early_and_late_corrupt_fields_refuse_before_snapshot_return(frame, tmp_path, name):
    frames = _stream(tmp_path, _unused(frame))
    document = json.loads((frames.directory / "frames.json").read_text())
    row = next(row for row in document["frames"][0]["fields"] if row["name"] == name)
    with (frames.directory / "frames.f64").open("r+b") as stream:
        stream.seek(row["offset"])
        old = stream.read(1)
        stream.seek(row["offset"])
        stream.write(bytes([old[0] ^ 1]))
    with pytest.raises(ValueError, match="hashes to"):
        _pack(frames)
    assert frames._cached_frame is None


@pytest.mark.parametrize("bad", ["missing", "rank", "horizontal", "vertical", "infinity"])
def test_unused_field_keeps_canonical_value_and_frame_grid_checks(frame, tmp_path, bad):
    frames = _stream(tmp_path, _unused(frame))
    path = frames.directory / "frames.json"
    document = json.loads(path.read_text())
    row = next(row for row in document["frames"][0]["fields"]
               if row["name"] == "unused_diagnostic")
    if bad == "missing":
        row["missing_count"] = 1
        message = "missing count"
    elif bad == "rank":
        row["axes"] = ["y", "x"]
        message = "rank"
    elif bad == "horizontal":
        row["axes"] = ["y", "vertical", "x"]
        message = "horizontal grid"
    elif bad == "vertical":
        row["shape"][0] //= 2
        row["length"] //= 2
        with (frames.directory / "frames.f64").open("rb") as stream:
            stream.seek(row["offset"])
            values = np.frombuffer(stream.read(row["length"]), dtype="<f8").reshape(row["shape"])
        row["sha256"] = _array_sha256(values)
        message = "vertical coordinate"
    else:
        with (frames.directory / "frames.f64").open("r+b") as stream:
            stream.seek(row["offset"])
            values = np.frombuffer(stream.read(row["length"]), dtype="<f8").reshape(row["shape"]).copy()
            values.flat[0] = np.inf
            stream.seek(row["offset"])
            stream.write(values.tobytes())
        row["sha256"] = _array_sha256(values)
        message = "infinity"
    path.write_text(json.dumps(document))
    for fieldwise in (False, True):
        reader = fixture.engine_bridge.open_frameset(frames.directory)
        with pytest.raises(ValueError, match=message):
            if fieldwise:
                _pack(reader)
            else:
                reader[0]


@pytest.mark.parametrize("case", ["time", "axis", "header"])
def test_fieldwise_metadata_uses_same_frame_invariants(frame, tmp_path, case):
    frames = _stream(tmp_path, frame)
    path = frames.directory / "frames.json"
    document = json.loads(path.read_text())
    entry = document["frames"][0]
    if case == "time":
        entry["valid_time"] = "2020-01-01T00:00:00"
        message = "precedes source cycle"
    elif case == "axis":
        axis = np.asarray(entry["latitude"]["values"], dtype=np.float64)
        axis[1] = axis[0]
        entry["latitude"] = fixture.engine_bridge._axis_document(axis)
        message = "strictly monotonic"
    else:
        entry["header"]["initialization_policies"].pop("cloud_water_mixing_ratio")
        message = "explicit.*policy"
    path.write_text(json.dumps(document))
    for fieldwise in (False, True):
        reader = fixture.engine_bridge.open_frameset(frames.directory)
        with pytest.raises(ValueError, match=message):
            if fieldwise:
                _pack(reader)
            else:
                reader[0]


def test_global_soil_donor_wrap_and_water_missing_values_are_unchanged(frame, tmp_path):
    fields = dict(frame.fields)
    land = np.ones_like(fields["land_fraction"].values)
    land[0, 3] = 0.
    fields["land_fraction"] = replace(fields["land_fraction"], values=land)
    for name in ("soil_temperature", "volumetric_soil_moisture"):
        values = fields[name].values.copy()
        values[:, 0, 0:2] = np.nan
        values[:, 0, 3] = np.nan
        values[:, 1, 0] = np.nan
        fields[name] = replace(fields[name], values=values,
                               missing_count=int(np.isnan(values).sum()))
    frame = replace(frame, fields=fields,
                    longitude=np.arange(fixture._NX, dtype=np.float64) * (360. / fixture._NX))
    policy = {"kind": "nearest_soil_column_within_cells", "maximum_cells": 2}
    expected = mapped_frames_to_regular_snapshots(
        (frame,), initialize_absent_hydrometeors=True, soil_land_repair=policy)[0]
    actual = _pack(_stream(tmp_path, frame), soil_land_repair=policy)
    _assert_same_snapshot(actual, expected)
    # Across the seam is the only radius-one donor for (0,0).
    np.testing.assert_array_equal(actual.fields["RW_SOIL_TEMPERATURE"][:, 0, 0],
                                  fields["soil_temperature"].values[:, 0, -1])
    assert np.isnan(actual.fields["RW_SOIL_TEMPERATURE"][:, 0, 3]).all()


def test_projected_metadata_and_external_frame_remain_owned(frame, tmp_path):
    grid = replace(frame.header.grid, projection="lambert_conformal_conic",
                   parameters={"standard_parallel_1": 30., "standard_parallel_2": 60.,
                               "central_longitude": -97., "latitude_of_origin": 40.,
                               "axis_unit_m": 1000.})
    frame = replace(frame, header=replace(frame.header, grid=grid))
    frames = _stream(tmp_path, frame)
    held = frames[0]
    wanted = held.fields["air_temperature"].values.tobytes()
    actual = _pack(frames[0:1])
    assert actual.projection["family"] == grid.projection
    assert actual.projection["parameters"] == grid.parameters
    assert held.fields["air_temperature"].values.tobytes() == wanted
    assert not np.shares_memory(actual.fields["T"], held.fields["air_temperature"].values)
    assert (frames.directory / "frames.f64").is_file()
    assert frames[0].fields["air_temperature"].values.tobytes() == wanted


def test_field_items_constructor_copies_before_next_item_and_rejects_duplicates(frame):
    raw = np.ones((fixture._NY, fixture._NX), dtype=np.float64)
    def items():
        yield "A", raw
        raw[:] = 2.
        yield "B", raw
    kwargs = dict(valid_time=frame.valid_time, levels_hpa=frame.vertical_values,
                  latitude=frame.latitude, longitude=frame.longitude)
    actual = Era5Snapshot.from_field_items(**kwargs, field_items=items())
    raw[:] = 3.
    np.testing.assert_array_equal(actual.fields["A"], 1.)
    np.testing.assert_array_equal(actual.fields["B"], 2.)
    with pytest.raises(ValueError, match="repeated snapshot field"):
        Era5Snapshot.from_field_items(**kwargs, field_items=(("A", raw), ("A", raw)))


@pytest.mark.parametrize("pressure", [-1., np.nan])
def test_fieldwise_pressure_admission_is_unchanged(frame, tmp_path, pressure):
    fields = dict(frame.fields)
    values = fields["air_pressure"].values.copy()
    values[0, 0, 0] = pressure
    fields["air_pressure"] = replace(fields["air_pressure"], values=values,
                                      missing_count=int(np.isnan(values).sum()))
    frame = replace(frame, fields=fields)
    with pytest.raises(ValueError, match="finite and positive"):
        _pack(_stream(tmp_path, frame))


def test_fieldwise_does_not_hide_soil_gaps_or_default_zero_policy(frame, tmp_path):
    fields = dict(frame.fields)
    values = fields["soil_temperature"].values.copy()
    values[0, 0, 0] = np.nan
    fields["soil_temperature"] = replace(fields["soil_temperature"], values=values,
                                          missing_count=1)
    with pytest.raises(ValueError, match="missing source-land"):
        _pack(_stream(tmp_path, replace(frame, fields=fields)))
    policies = dict(frame.header.initialization_policies)
    policies["cloud_water_mixing_ratio"] = "declared_but_not_implemented"
    frame = replace(frame, header=replace(frame.header, initialization_policies=policies))
    frames = fixture.engine_bridge.open_frameset(
        fixture.engine_bridge.write_frameset(tmp_path / "policy", (frame,)))
    with pytest.raises(ValueError, match="explicit-zero policy"):
        _pack(frames)
    ordinary = mapped_frames_to_regular_snapshots((frames.fieldwise_frame(0),))[0]
    assert "QC" not in ordinary.fields


def test_real_rust_frameset_packs_identically_without_materializing(tmp_path, monkeypatch):
    import test_mapped_source as nc
    mapping, source = tmp_path / "mapping.json", tmp_path / "input.nc"
    nc._write_mapping(mapping, nc._mapping())
    nc._write_source(source)
    output = tmp_path / "native"
    fixture.engine_bridge.run_engine("decode", mapping=mapping,
                                      files=[source], output=output)
    frames = fixture.engine_bridge.open_frameset(output)
    expected = mapped_frames_to_regular_snapshots(
        tuple(frames), initialize_absent_hydrometeors=True)
    monkeypatch.setattr(frames, "_materialize", lambda *_: pytest.fail("whole frame read"))
    for index, wanted in enumerate(expected):
        actual = mapped_frames_to_regular_snapshots(
            (frames.fieldwise_frame(index),), initialize_absent_hydrometeors=True)[0]
        _assert_same_snapshot(actual, wanted)
