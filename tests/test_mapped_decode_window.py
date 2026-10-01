"""A GRIB2 source decoded over its atmospheric window reads as the whole decode cropped.

The mapped engine asks for the window from Section 3 before a record is
unpacked and decodes the granted vertical fields over it alone.  These
cells run real GRIB2 bytes (regular latitude/longitude, GDT 0, scan 0x40)
through the engine with and without target grids, on a regional and a
global grid, and hold every windowed read to the whole decode's crop.

Named breakages they hold:

* a pressure field the source publishes directly (ICON-D2 does) was
  decoded over the window, published without the whole plane's per-level
  pressures, and the reader could not open the frameset;
* a frame that leaves a geopotential height level out completes it
  hydrostatically from whole columns; the completion was refused for every
  pressure-level source given a window, because its temperature and
  humidity were decoded over the window beside full-grid surface pressure.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from gpuwm import mapped_engine_bridge as bridge
from gpuwm.ingest.atmospheric_window import (
    CANONICAL_ATMOSPHERIC_FIELDS, WINDOW_DECODED_VALIDATION, WINDOW_SCHEMA,
)
from gpuwm.static.lambert import LambertGrid

ROOT = Path(__file__).resolve().parents[1]
GDPS = ROOT / "gpuwm" / "authorities" / "rw-wps-gem-gdps-grib2.mapping.json"
LEVELS = (50000.0, 85000.0, 100000.0)
HEIGHTS = {50000.0: 5600.0, 85000.0: 1500.0, 100000.0: 110.0}
CYCLE = datetime(2026, 9, 29, 12)
COMPLETE = "complete-canonical-field-before-window-v1"
#: (latitudes, longitudes in degrees east) of the two source grids: a
#: regional crop and a whole ring whose stored cut (180 E) is far from the
#: target, so the window is granted on both.
GRIDS = {
    "regional": (np.arange(20.0, 50.5, 1.0), np.arange(240.0, 290.5, 1.0)),
    "global": (np.arange(-90.0, 90.5, 3.0), np.arange(0.0, 359.0, 3.0)),
}


def target():
    return LambertGrid(35.0, -97.5, 30.0, 60.0, -97.5, 12000.0, 12000.0, 20, 20)


def _octets(value):
    """GRIB2 sign-and-magnitude, four octets."""
    magnitude = abs(int(round(value)))
    return ((0x80000000 | magnitude) if value < 0 else magnitude).to_bytes(4, "big")


def _message(key, level, values, latitude, longitude, valid_time):
    """One GDT 0 / IEEE float64 record, every octet stated."""
    discipline, category, parameter, level_type = key[:4]
    second = key[4] if len(key) > 4 else None

    def section(number, size):
        data = bytearray(size)
        data[:4] = size.to_bytes(4, "big")
        data[4] = number
        return data

    s1 = section(1, 21)
    s1[5:7] = (54).to_bytes(2, "big")
    s1[9], s1[11] = 5, 1
    s1[12:14] = CYCLE.year.to_bytes(2, "big")
    s1[14:19] = bytes([CYCLE.month, CYCLE.day, CYCLE.hour, 0, 0])
    s3 = section(3, 72)
    s3[6:10] = values.size.to_bytes(4, "big")
    s3[14] = 6
    s3[30:34] = len(longitude).to_bytes(4, "big")
    s3[34:38] = len(latitude).to_bytes(4, "big")
    s3[46:50] = _octets(latitude[0] * 1e6)
    s3[50:54] = _octets(longitude[0] * 1e6)
    s3[54] = 48
    s3[55:59] = _octets(latitude[-1] * 1e6)
    s3[59:63] = _octets(longitude[-1] * 1e6)
    s3[63:67] = _octets((longitude[1] - longitude[0]) * 1e6)
    s3[67:71] = _octets((latitude[1] - latitude[0]) * 1e6)
    s3[71] = 0x40
    s4 = section(4, 34)
    s4[9:14] = bytes([category, parameter, 0, 0, 153])
    s4[17] = 13
    s4[18:22] = int((valid_time - CYCLE).total_seconds()).to_bytes(4, "big")
    s4[22], s4[23] = level_type, 3
    s4[24:28] = _octets(level * 1000)
    s4[28] = 255 if second is None else second[0]
    if second is not None:
        s4[29] = 3
        s4[30:34] = _octets(second[1] * 1000)
    s5 = section(5, 12)
    s5[5:9] = values.size.to_bytes(4, "big")
    s5[9:11] = (4).to_bytes(2, "big")
    s5[11] = 2
    s6 = section(6, 6)
    s6[5] = 255
    s7 = section(7, 5 + values.size * 8)
    s7[5:] = np.ascontiguousarray(values, dtype=">f8").tobytes()
    body = b"".join([s1, s3, s4, s5, s6, s7]) + b"7777"
    return b"GRIB\0\0" + bytes([discipline, 2]) + (16 + len(body)).to_bytes(8, "big") + body


#: (discipline, category, parameter, level_type[, (second type, value)]),
#: the GDPS mapping's selectors, with each record's level and value.
VERTICAL = {
    "air_temperature": ((0, 0, 0, 100), lambda p: 300.0 - (100000.0 - p) * 0.0008),
    "specific_humidity": ((0, 1, 0, 100), lambda p: 0.008 * p / 100000.0),
    "eastward_wind": ((0, 2, 2, 100), lambda p: 5.0 + (100000.0 - p) * 0.0002),
    "northward_wind": ((0, 2, 3, 100), lambda p: 1.0),
    "geopotential_height": ((0, 3, 5, 100), lambda p: HEIGHTS[p]),
}
SURFACE = {
    "surface_pressure": ((0, 3, 0, 1), 0.0, 98000.0),
    "terrain_height": ((0, 3, 5, 1), 0.0, 250.0),
    "skin_temperature": ((0, 0, 17, 1), 0.0, 290.0),
    "air_temperature_2m": ((0, 0, 0, 103), 2.0, 289.0),
    "specific_humidity_2m": ((0, 1, 0, 103), 2.0, 0.007),
    "eastward_wind_10m": ((0, 2, 2, 103), 10.0, 3.0),
    "northward_wind_10m": ((0, 2, 3, 103), 10.0, 1.0),
    "snow_depth": ((0, 1, 11, 1), 0.0, 0.0),
    "soil_temperature": ((2, 0, 2, 106, (106, 0.1)), 0.0, 285.0),
    "volumetric_soil_moisture": ((2, 0, 25, 106, (106, 0.1)), 0.0, 0.25),
}
#: Cycle-invariant: published at the analysis only.
INVARIANT = {
    "land_fraction": ((2, 0, 0, 1), 0.0, 0.75),
    "sea_ice_fraction": ((10, 2, 0, 1), 0.0, 0.0),
}
PRESSURE = ((0, 3, 0, 100), lambda p: p)


def _write_source(directory, grid, *, direct_pressure=False, leave_out=(), times=2,
                  invariant_every_time=False):
    """Three-hourly valid times, one object each; ``leave_out`` names (time, name, level).

    ``invariant_every_time`` publishes the land and sea-ice fractions in
    every object instead of the analysis alone, so no object is read by
    more than one valid time and the engine may decode several at once.
    """
    latitude, longitude = GRIDS[grid]
    rows, columns = np.meshgrid(np.arange(len(latitude)), np.arange(len(longitude)), indexing="ij")
    ramp = 0.01 * rows + 0.002 * columns
    paths = []
    for time_index in range(times):
        valid_time = CYCLE + timedelta(hours=3 * time_index)
        drift = 0.25 * time_index
        messages = []
        vertical = dict(VERTICAL)
        if direct_pressure:
            vertical["air_pressure"] = PRESSURE
        for name, (key, profile) in vertical.items():
            for level in LEVELS:
                if (time_index, name, level) in leave_out:
                    continue
                scale = 10.0 if name == "geopotential_height" else 1.0
                # A pressure level carries its own pressure at every cell.
                values = profile(level) + (0.0 * ramp if name == "air_pressure"
                                           else scale * (ramp + drift))
                messages.append(_message(key, level, values, latitude, longitude, valid_time))
        surface = dict(SURFACE)
        if time_index == 0 or invariant_every_time:
            surface.update(INVARIANT)
        for name, (key, level, base) in surface.items():
            varies = name not in INVARIANT and base > 0.0 and name != "volumetric_soil_moisture"
            values = base + (ramp + drift if varies else 0.0 * ramp)
            messages.append(_message(key, level, values, latitude, longitude, valid_time))
        path = directory / f"f{3 * time_index:03d}.grib2"
        path.write_bytes(b"".join(messages))
        paths.append(path)
    return tuple(paths)


def _write_mapping(path, *, direct_pressure=False, invariant_every_time=False):
    mapping = json.loads(GDPS.read_text(encoding="utf-8"))
    mapping["coordinates"]["vertical"]["levels"] = list(LEVELS)
    if invariant_every_time:
        for name in INVARIANT:
            del mapping["fields"][name]["time_binding"]
    if direct_pressure:
        field = mapping["fields"]["air_pressure"]
        del field["derivation"]
        field["selectors"] = [{"format": "grib2", "discipline": 0, "category": 3,
                               "parameter": 0, "level_type": 100}]
    path.write_text(json.dumps(mapping, indent=1), encoding="utf-8")
    return path


@pytest.fixture
def engine():
    binary = bridge.require_engine()
    result = subprocess.run([str(binary), "capabilities"], capture_output=True,
                            text=True, check=True)
    if json.loads(result.stdout).get("features", {}).get("atmospheric_window") != WINDOW_SCHEMA:
        pytest.skip("selected native engine predates atmospheric window publication")
    return binary


def _pair(tmp_path, engine, grid, **options):
    """The whole decode and the windowed one of the same bytes."""
    mapping = _write_mapping(tmp_path / "mapping.json",
                             direct_pressure=options.get("direct_pressure", False))
    files = _write_source(tmp_path, grid, **options)
    for name, grids in (("full", ()), ("small", (target(),))):
        bridge.run_engine("decode", mapping=mapping, files=files, output=tmp_path / name,
                          engine=engine, atmospheric_grids=grids)
    full = bridge.open_frameset(tmp_path / "full")
    small = bridge.open_frameset(tmp_path / "small", full_fallback=lambda: full)
    return full, small


def _validations(frames, index):
    return {str(row["name"]): row["original"]["validation"]
            for row in frames._entries[index]["fields"] if "original" in row}


def _assert_the_window_is_the_whole_decode_cropped(full, small):
    assert len(small) == len(full) == 2
    for index in range(len(full)):
        window = small._published_window(index)
        assert window is not None and window.shape != window.source_shape
        fields = small.fieldwise_frame(index).with_atmospheric_window(window).fields
        assert sorted(fields) == sorted(full.field_names(index))
        for name in full.field_names(index):
            original = full.field(index, name).values
            expected = window.crop(original) if name in CANONICAL_ATMOSPHERIC_FIELDS else original
            assert fields[name].values.tobytes() == expected.tobytes(), (index, name)
        assert (small.pressure_levels_hpa(index).tobytes()
                == full.pressure_levels_hpa(index).tobytes())
    assert small._full_frames is None
    # The fallback's own identity check holds every window-decoded payload
    # to the whole decode's crop, byte for byte.
    assert small._original_frames() is full


@pytest.mark.parametrize("grid", sorted(GRIDS))
def test_a_direct_pressure_field_is_decoded_whole_and_the_frameset_opens(tmp_path, engine, grid):
    full, small = _pair(tmp_path, engine, grid, direct_pressure=True)
    _assert_the_window_is_the_whole_decode_cropped(full, small)
    for index in range(2):
        validation = _validations(small, index)
        assert validation["air_pressure"] == COMPLETE
        assert "original_pressure_hpa" in small._entries[index]
        for name in VERTICAL:
            assert validation[name] == WINDOW_DECODED_VALIDATION, (index, name)


@pytest.mark.parametrize("grid", sorted(GRIDS))
def test_a_frame_that_leaves_a_height_level_out_is_decoded_whole(tmp_path, engine, grid):
    full, small = _pair(tmp_path, engine, grid,
                        leave_out={(1, "geopotential_height", 85000.0)})
    # The whole decode completed the level the source left out.
    completed = full.field(1, "geopotential_height")
    assert np.isfinite(completed.values).all()
    assert any(reference.startswith("@completed.hypsometric")
               for reference in completed.source_references)
    _assert_the_window_is_the_whole_decode_cropped(full, small)
    # Only the frame that completes a level is decoded whole; the other
    # keeps the window-decoded columns.
    first, second = _validations(small, 0), _validations(small, 1)
    for name in VERTICAL:
        assert first[name] == WINDOW_DECODED_VALIDATION, name
        assert second[name] == COMPLETE, name


@pytest.mark.parametrize("leave_out", [frozenset(), frozenset({(1, "geopotential_height", 85000.0)})],
                         ids=["windowed", "one-frame-whole"])
def test_many_valid_times_in_flight_write_the_one_lane_window_decoded_frameset(
        tmp_path, engine, monkeypatch, leave_out):
    """The decode on every core keeps the window decode's frames, byte for byte.

    Breakage prevented: the many-lane writer decodes each valid time on its
    own (``DecodeStream::slice_detached``) and publishes each frame's window
    from its ordered thread; a lane that decoded whole fields, asked for a
    window the decode was not granted, or skipped the whole re-decode of a
    frame that completes a height level would publish other bytes than the
    one-lane writer, or refuse.
    """
    mapping = _write_mapping(tmp_path / "mapping.json", invariant_every_time=True)
    files = _write_source(tmp_path, "global", leave_out=leave_out, times=3,
                          invariant_every_time=True)
    outputs = {}
    for arm, threads, lanes in (("one", 1, None), ("many", 4, "3")):
        if lanes is None:
            monkeypatch.delenv("GPUWM_MAPPED_ENGINE_LANES", raising=False)
        else:
            monkeypatch.setenv("GPUWM_MAPPED_ENGINE_LANES", lanes)
        outputs[arm] = tmp_path / arm
        bridge.run_engine("decode", mapping=mapping, files=files, output=outputs[arm],
                          engine=engine, atmospheric_grids=(target(),), threads=threads)
    for name in ("frames.json", "frames.f64"):
        assert (outputs["one"] / name).read_bytes() == (outputs["many"] / name).read_bytes(), name
    frames = bridge.open_frameset(outputs["many"])
    assert len(frames) == 3
    for index in range(3):
        whole = any(time == index for time, _name, _level in leave_out)
        validation = _validations(frames, index)
        for name in VERTICAL:
            assert validation[name] == (COMPLETE if whole else WINDOW_DECODED_VALIDATION), (index, name)
