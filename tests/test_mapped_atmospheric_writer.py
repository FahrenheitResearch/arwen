"""The native publisher retains atmospheric support after complete validation."""
from dataclasses import replace
import json
import subprocess

import netCDF4
import numpy as np
import pytest

from gpuwm import mapped_engine_bridge as bridge
from gpuwm.ingest.atmospheric_window import (
    AtmosphericWindow, CANONICAL_ATMOSPHERIC_FIELDS, WINDOW_SCHEMA,
    WINDOWED_FRAMESET_SCHEMA, WindowedAtmosphericSnapshot,
)
from gpuwm.ingest.horiz import interpolate_era5_to_lambert
from gpuwm.static.lambert import LambertGrid
import test_mapped_source as source_fixture
import test_mapped_frameset_streaming as bundle_fixture
from test_atmospheric_window import assert_horizontal_exact


def target(lat=32., lon=-99.5):
    return LambertGrid(lat, lon, 30., 60., -97., 1000., 1000., 3, 3)


@pytest.fixture
def source(tmp_path):
    mapping = tmp_path / "mapping.json"
    data = tmp_path / "source.nc"
    source_fixture._write_mapping(mapping, source_fixture._mapping())
    source_fixture._write_source(data)
    with netCDF4.Dataset(data, "r+") as dataset:
        dataset["terrain_height"][1] = dataset["terrain_height"][0]
        for name in source_fixture.THREE_D:
            array = dataset[name][:]
            array += np.arange(array.size).reshape(array.shape) * 0.00001
            dataset[name][:] = array
    return mapping, data


@pytest.fixture
def engine():
    binary = bridge.require_engine()
    result = subprocess.run([str(binary), "capabilities"], capture_output=True,
                            text=True, check=True)
    if json.loads(result.stdout).get("features", {}).get("atmospheric_window") != WINDOW_SCHEMA:
        pytest.skip("selected native engine predates atmospheric window publication")
    return binary


def decode(source, engine, path, grids=()):
    mapping, data = source
    return bridge.run_engine("decode", mapping=mapping, files=(data,),
                             output=path, engine=engine, atmospheric_grids=grids)


@pytest.fixture
def pair(source, engine, tmp_path):
    decode(source, engine, tmp_path / "full")
    result = decode(source, engine, tmp_path / "small", (target(),))
    full = bridge.open_frameset(tmp_path / "full")
    small = bridge.open_frameset(tmp_path / "small", full_fallback=lambda: full)
    return full, small, result


def test_writer_retains_only_atmosphere_and_original_identity(pair, source):
    full, small, result = pair
    assert "--atmospheric-window" in result["command"]
    document = json.loads((small.directory / "frames.json").read_text())
    assert document["schema"] == WINDOWED_FRAMESET_SCHEMA
    window = small._published_window(0)
    assert window is not None and window.shape != window.source_shape
    fields = small.fieldwise_frame(0).with_atmospheric_window(window).fields
    saved = 0
    for name in full.field_names(0):
        original = full.field(0, name).values
        actual = fields[name].values
        if name in CANONICAL_ATMOSPHERIC_FIELDS:
            assert actual.tobytes() == window.crop(original).tobytes(), name
            saved += original.nbytes - actual.nbytes
        else:
            assert actual.tobytes() == original.tobytes(), name
        assert small.field_digest(0, name) == full.field_digest(0, name)
    assert full._stream.stat().st_size - small._stream.stat().st_size == saved * len(full)
    assert small.header(0) == full.header(0)
    assert small.pressure_levels_hpa(0).tobytes() == full.pressure_levels_hpa(0).tobytes()
    assert small._full_frames is None
    # A regular consumer asks for support using the unchanged source axes.
    authority = source[0]
    full_bundle = replace(bundle_fixture._bundle(full.directory, authority), frames=full)
    small_bundle = replace(bundle_fixture._bundle(small.directory, authority), frames=small)
    original = full_bundle.regular_snapshots()[0]
    actual = small_bundle.regular_snapshots().for_grids((target(),))[0]
    assert isinstance(actual, WindowedAtmosphericSnapshot)
    assert_horizontal_exact(interpolate_era5_to_lambert(original, target(), backend="cpu"),
                            interpolate_era5_to_lambert(actual, target(), backend="cpu"))
    assert small._full_frames is None


def test_full_and_outside_reads_use_original_provider(pair):
    full, small, _ = pair
    outside = AtmosphericWindow((5, 6), (0, 2), (0, 2))
    fields = small.fieldwise_frame(0).with_atmospheric_window(outside).fields
    assert fields["air_temperature"].values.tobytes() == outside.crop(
        full.field(0, "air_temperature").values).tobytes()
    assert small._full_frames is full
    for name in full.field_names(0):
        assert small.field(0, name).values.tobytes() == full.field(0, name).values.tobytes()


def test_unproven_coverage_preserves_full_writer(source, engine, tmp_path):
    decode(source, engine, tmp_path / "full")
    decode(source, engine, tmp_path / "outside", (target(50., -110.),))
    for name in ("frames.json", "frames.f64"):
        assert (tmp_path / "full" / name).read_bytes() == (tmp_path / "outside" / name).read_bytes()


def test_source_corruption_outside_retained_support_still_refuses(source, engine, tmp_path):
    with netCDF4.Dataset(source[1], "r+") as data:
        data["air_temperature"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="missing|finite|NaN"):
        decode(source, engine, tmp_path / "bad", (target(),))
    assert not (tmp_path / "bad" / "frames.json").exists()


def test_retained_payload_corruption_still_refuses(pair):
    _, small, _ = pair
    row = next(row for row in small._entries[0]["fields"] if row["name"] == "air_temperature")
    with small._stream.open("r+b") as stream:
        stream.seek(row["offset"])
        value = stream.read(1)
        stream.seek(row["offset"])
        stream.write(bytes([value[0] ^ 1]))
    fields = small.fieldwise_frame(0).with_atmospheric_window(small._published_window(0)).fields
    with pytest.raises(ValueError, match="hashes to"):
        fields["air_temperature"]


@pytest.mark.parametrize("change", ["old_schema", "orphan_payload", "original_shape", "inventory", "pressure"])
def test_published_descriptor_cannot_replace_original_contract(pair, change):
    _, small, _ = pair
    path = small.directory / "frames.json"
    document = json.loads(path.read_text())
    frame = document["frames"][0]
    if change == "old_schema":
        document["schema"] = bridge.FRAMESET_SCHEMA
    elif change == "orphan_payload":
        del frame["atmospheric_window"]
    elif change == "original_shape":
        field = next(row for row in frame["fields"] if "original" in row)
        field["original"]["shape"][1] += 1
    elif change == "inventory":
        frame["atmospheric_window"]["fields"].append("surface_pressure")
    else:
        frame["original_pressure_hpa"]["values"][0] += 1.
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        bridge.open_frameset(small.directory)


def test_original_provider_identity_is_checked_before_full_fallback(pair):
    full, small, _ = pair
    full._entries[0]["source_cycle"] = "1999-01-01T00:00:00"
    with pytest.raises(ValueError, match="geometry"):
        small.field(0, "air_temperature")


def test_native_request_refuses_present_nonconsumed_vertical_field(source, engine, tmp_path, monkeypatch):
    """A valid canonical 3-D diagnostic still lies outside this consumer ABI."""
    import gpuwm.ingest.atmospheric_window as module
    mapping = json.loads(source[0].read_text())
    mapping["fields"]["cloud_water_mixing_ratio"] = source_fixture._field(
        "cloud_water_mixing_ratio", "kg kg-1", ["vertical", "y", "x"], "mass")
    source_fixture._write_mapping(source[0], mapping)
    with netCDF4.Dataset(source[1], "r+") as dataset:
        field = dataset.createVariable("cloud_water_mixing_ratio", "f8", ("time", "level", "y", "x"))
        field.units = "kg kg-1"
        field[:] = 0.001
    # Establish that it is an actual validated source field with the right axes.
    decode(source, engine, tmp_path / "full-with-cloud")
    full = bridge.open_frameset(tmp_path / "full-with-cloud")
    assert full.field(0, "cloud_water_mixing_ratio").values.shape == (3, 5, 6)
    original = module.window_request_response
    def invalid(event, grids):
        response = original(event, grids)
        response["fields"].append("cloud_water_mixing_ratio")
        return response
    monkeypatch.setattr(module, "window_request_response", invalid)
    with pytest.raises(ValueError, match="atmospheric.*inventory"):
        decode(source, engine, tmp_path / "bad-cloud-window", (target(),))


@pytest.mark.parametrize("change", ["geometry", "bounds", "surface", "duplicate", "shape"])
def test_native_request_rejects_invalid_support(source, engine, tmp_path, monkeypatch, change):
    import gpuwm.ingest.atmospheric_window as module
    original = module.window_request_response
    def invalid(event, grids):
        response = original(event, grids)
        assert response["mode"] == "window"
        if change == "geometry":
            response["geometry"] = dict(response["geometry"], frame_index=999)
        elif change == "bounds":
            response["rows"] = [0, 999]
        elif change == "surface":
            response["fields"].append("surface_pressure")
        elif change == "duplicate":
            response["fields"].append(response["fields"][0])
        else:
            response["source_shape"] = [50, 60]
        return response
    monkeypatch.setattr(module, "window_request_response", invalid)
    with pytest.raises(ValueError, match="window"):
        decode(source, engine, tmp_path / change, (target(),))
    assert not (tmp_path / change / "frames.json").exists()
