"""Read a complete height-coordinate GRIB stream through the native executable."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from gpuwm import mapped_engine_bridge as bridge
from test_mapped_height_vertical import (RAW_FRACTIONS, dependency_only_fixture,
                                         height_fixture, materialize)


def grib_message(record):
    """Tiny regular GDT 0 / IEEE 64 fixture, with all metadata explicit."""
    def section(number, size):
        data = bytearray(size)
        data[:4] = size.to_bytes(4, "big")
        data[4] = number
        return data

    def word(data, offset, value, size=4):
        data[offset:offset+size] = int(value).to_bytes(size, "big")

    s1 = section(1, 21)
    word(s1, 5, record.center, 2)
    s1[9] = 5
    s1[11] = 1
    stamp = record.reference_time
    word(s1, 12, stamp.year, 2)
    s1[14:19] = bytes([stamp.month, stamp.day, stamp.hour, stamp.minute, stamp.second])
    s3 = section(3, 72)
    word(s3, 6, record.values.size)
    s3[14] = 6
    word(s3, 30, len(record.longitude))
    word(s3, 34, len(record.latitude))
    for offset, value in [(46, record.latitude[0]), (50, record.longitude[0]),
                          (55, record.latitude[-1]), (59, record.longitude[-1]),
                          (63, np.diff(record.longitude)[0]),
                          (67, np.diff(record.latitude)[0])]:
        word(s3, offset, round(value*1e6))
    s3[54], s3[71] = 48, 64
    s4 = section(4, 34)
    s4[9:14] = bytes([record.category, record.parameter, 0, 0, 153])
    s4[17] = 13
    word(s4, 18, (record.valid_time-stamp).total_seconds())
    s4[22], s4[23] = record.level_type, 3
    word(s4, 24, round(record.level_value*1000))
    s4[28] = record.second_level_type if record.second_level_type is not None else 255
    if s4[28] != 255:
        s4[29] = 3
        word(s4, 30, round(record.second_level_value*1000))
    s5 = section(5, 12)
    word(s5, 5, record.values.size)
    word(s5, 9, 4, 2)
    s5[11] = 2
    s6 = section(6, 6)
    s6[5] = 255
    s7 = section(7, 5+record.values.size*8)
    s7[5:] = np.asarray(record.values, dtype=">f8").tobytes()
    body = b"".join([s1, s3, s4, s5, s6, s7])+b"7777"
    return b"GRIB\0\0"+bytes([record.discipline, 2])+(16+len(body)).to_bytes(8, "big")+body


def test_native_height_decode_matches_reference_and_stream_reader(tmp_path):
    executable = os.environ.get("GPUWM_MAPPED_ENGINE_BIN")
    if not executable:
        pytest.skip("set GPUWM_MAPPED_ENGINE_BIN to the freshly built native engine")
    mapping, records = height_fixture()
    expected = materialize(tmp_path, mapping, records)
    data = tmp_path / "columns.grib2"
    data.write_bytes(b"".join(grib_message(record) for record in records))
    authority = tmp_path / "native-mapping.json"
    authority.write_text(json.dumps(mapping), encoding="utf-8")
    output = tmp_path / "frames"
    bridge.run_engine("decode", mapping=authority, files=(data,),
                      output=output, engine=Path(executable))
    actual = bridge.read_frameset(output)
    assert len(actual) == len(expected) == 2
    for native, reference in zip(actual, expected):
        assert native.fields.keys() == reference.fields.keys()
        for name in native.fields:
            np.testing.assert_array_equal(native.fields[name].values,
                                          reference.fields[name].values)
    streamed = bridge.open_frameset(output)
    for k, reference in enumerate(expected):
        frame = streamed.fieldwise_frame(k)
        np.testing.assert_array_equal(frame.fields["geopotential_height"].values,
                                      reference.fields["geopotential_height"].values)


def test_native_stream_carries_no_dependency_only_input(tmp_path):
    """The engine a bare run uses holds a raw input off frames.json and frames.f64.

    Named breakage: ICON-D2's six raw mass fractions were written beside the
    mixing ratios rebased from them, 390 of 1,217 layers per valid time.
    """
    executable = os.environ.get("GPUWM_MAPPED_ENGINE_BIN")
    if not executable:
        pytest.skip("set GPUWM_MAPPED_ENGINE_BIN to the freshly built native engine")
    mapping, records = dependency_only_fixture()
    expected = materialize(tmp_path, mapping, records)
    data = tmp_path / "columns.grib2"
    data.write_bytes(b"".join(grib_message(record) for record in records))
    authority = tmp_path / "native-mapping.json"
    authority.write_text(json.dumps(mapping), encoding="utf-8")
    output = tmp_path / "frames"
    bridge.run_engine("decode", mapping=authority, files=(data,),
                      output=output, engine=Path(executable))
    document = json.loads((output / "frames.json").read_text(encoding="utf-8"))
    written = 0
    for entry, reference in zip(document["frames"], expected, strict=True):
        names = [field["name"] for field in entry["fields"]]
        assert names == list(reference.fields)
        assert not set(RAW_FRACTIONS) & set(names)
        assert [field["canonical_name"] for field in entry["header"]["fields"]] == names
        written += sum(field["length"] for field in entry["fields"])
    assert document["stream"]["bytes"] == written
    assert (output / "frames.f64").stat().st_size == written
    for native, reference in zip(bridge.read_frameset(output), expected):
        assert native.fields.keys() == reference.fields.keys()
        for name in native.fields:
            np.testing.assert_array_equal(native.fields[name].values,
                                          reference.fields[name].values)
