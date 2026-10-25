"""The history writer's frame sink: the sink is handed the fields the file is written from, before the write; with
``write_files = False`` the frame lands in the sink only and still counts as landed for the durability waits.
CPU shells of the writer (tests/test_wrfout.py's), no CUDA."""
from __future__ import annotations

import threading
from pathlib import Path

import netCDF4
import numpy as np

from gpuwm.io.frame_sink import FrameSink, frame_from_wrfout
from test_wrfout import _manual_async_writer, _queue_cpu_ticket


def _fields():
    return {"T": np.arange(1, dtype=np.float32).reshape(1, 1, 1) + 7.5, "MU": np.full((1, 1), 3.0, np.float32),
            "LANDMASK": np.ones((1, 1), np.float64), "LU_INDEX": np.full((1, 1), 17., np.float64)}


def test_the_sink_holds_the_words_the_file_holds(tmp_path):
    writer = _manual_async_writer(threading.Event())
    sink = FrameSink(4, fields=None)
    writer.frame_sink = sink
    path = tmp_path / "wrfout_d01_1974-04-03_12_00_00"
    _queue_cpu_ticket(writer, path, fields=_fields())
    writer.drain()
    frame = sink.frame("1974-04-03_12:00:00", timeout=30)
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        for name in _fields():
            on_disk = np.asarray(ds[name][0])
            assert on_disk.dtype == frame.arrays[name].dtype, name
            assert np.array_equal(on_disk.view(np.uint8), frame.arrays[name].view(np.uint8)), name
    assert writer.paths == [path]
    writer.close()


def test_a_sink_only_frame_lands_without_a_file(tmp_path):
    writer = _manual_async_writer(threading.Event())
    sink = FrameSink(4, fields=None)
    writer.frame_sink = sink
    writer.write_files = False
    path = tmp_path / "wrfout_d01_1974-04-03_12_00_00"
    target = writer.durability_target() + 1
    _queue_cpu_ticket(writer, path, fields=_fields())
    writer.wait_landed(target)
    writer.drain()
    assert not Path(path).exists() and writer.paths == []
    assert sink.frame("1974-04-03_12:00:00", timeout=30).arrays["MU"][0, 0] == 3.0
    assert writer.pending == 0
    writer.close()
    assert writer._frame_timings[str(path)]["sink_only"] is True
    assert writer._frame_timings[str(path)]["durable_after_s"] >= 0


def test_rust_frame_read_preserves_payload_bits_and_stored_types(tmp_path):
    path = tmp_path / "words.nc"
    floats = np.array([0, 0x80000000, 0x7fc12345, 0x7f812345], dtype="u4").view("f4")
    integers = np.array([2**53 + 1, 2**63 + 9, 0, 2**64 - 1], dtype="u8")
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("x", 4)
        ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[0] = np.frombuffer(
            b"1974-04-03_12:00:00", dtype="S1")
        ds.createVariable("T", "f4", ("Time", "x"))[0] = floats
        ds.createVariable("member_bits", "u8", ("x",))[:] = integers
    frame = frame_from_wrfout(path, 2, fields="all")
    assert frame.valid_time == "1974-04-03_12:00:00"
    assert frame.arrays["T"].dtype == floats.dtype
    assert frame.arrays["T"].tobytes() == floats.tobytes()
    assert frame.arrays["member_bits"].dtype == integers.dtype
    assert frame.arrays["member_bits"].tobytes() == integers.tobytes()


def test_missing_field_cannot_receive_a_bitwise_frame_verdict():
    from gpuwm.io.frame_sink import MemberFrame, compare_frames
    a = MemberFrame(1, 1, "1974-04-03_12:00:00", {"T": np.zeros(1, "f4")})
    b = MemberFrame(1, 1, "1974-04-03_12:00:00", {"T": np.zeros(1, "f4"), "MU": np.zeros(1, "f4")})
    comparison = compare_frames(a, b)
    assert comparison["differing_words"] == 0
    assert comparison["reference_only"] == ["MU"]
    assert comparison["verdict"] == "DIFFER"
