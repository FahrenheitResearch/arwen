"""The member session's CPU-checkable parts: specs, checkpoint cadence, frame sink words.

The forecast itself (members built through the door, advanced and analysed in one
process, word for word the door's per-member legs) is the C15 proof on a card."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ensemble import member_session as ms
from gpuwm.io import frame_sink as fs
from gpuwm.io.wrfout import wrf_axis_array


def test_a_spec_names_a_global_member_and_a_storage():
    spec = ms.MemberSpec(3, "run", "ck.npz", "out", 600.0)
    assert spec.member_id == 3 and spec.storage == "device"
    with pytest.raises(ms.MemberSessionError, match="1-based"):
        ms.MemberSpec(0, "run", "ck.npz", "out", 600.0)
    with pytest.raises(ms.MemberSessionError, match="device or managed"):
        ms.MemberSpec(1, "run", "ck.npz", "out", 600.0, storage="disk")


def test_the_cadence_decides_which_restart_instants_write():
    with pytest.raises(ms.MemberSessionError, match="cadence"):
        ms.MemberSession(cadence="hourly", synchronize=lambda: None)
    member = SimpleNamespace(spec=SimpleNamespace(member_id=4))
    every = ms.MemberSession(cadence="every-instant", synchronize=lambda: None)
    assert every._checkpoint_wanted(member, 1200)
    none = ms.MemberSession(cadence="none", synchronize=lambda: None)
    assert not none._checkpoint_wanted(member, 1200)
    asked = ms.MemberSession(cadence="requested", synchronize=lambda: None)
    asked._requested[4] = {2400}
    assert asked._checkpoint_wanted(member, 2400) and not asked._checkpoint_wanted(member, 1200)


def test_a_poisoned_session_refuses_every_member():
    session = ms.MemberSession(synchronize=lambda: None)
    session.members[1] = object()
    session.poisoned = True
    with pytest.raises(ms.MemberSessionError, match="poisoned"):
        session._member(1)


def test_the_plan_folder_is_its_valid_time():
    assert ms._stamp_of_plan({"valid_time": "2022-06-13_23:05:00"}) == "20220613230500"


def _fields(nz=3, ny=4, nx=5):
    rng = np.random.default_rng(7)
    return {"T": rng.standard_normal((nz, ny, nx)).astype(np.float32),
            "P": rng.standard_normal((nz, ny, nx)).astype(np.float32),
            "PH": rng.standard_normal((nz + 1, ny, nx)).astype(np.float32),
            "XLAND": np.ones((ny, nx), np.float32), "IVGTYP": np.arange(ny * nx, dtype=np.int32).reshape(ny, nx),
            "UNUSED": np.zeros((2,), np.float32)}


def test_the_sink_keeps_the_words_the_writer_writes():
    fields = _fields()
    sink = fs.FrameSink(2, fields=("T", "PH", "IVGTYP", "QHAIL"))
    sink(grid_id=1, valid_time=None, time_str="2022-06-13_23:05:00", fields=fields, nx=5, ny=4, nz=3,
         global_attrs={"TITLE": "x"})
    frame = sink.frame("2022-06-13_23:05:00", timeout=0)
    assert sorted(frame.arrays) == ["IVGTYP", "PH", "T"]       # a field the run does not write is absent
    for name in ("T", "PH", "IVGTYP"):
        expected = wrf_axis_array(name, fields[name], nx=5, ny=4, nz=3)
        assert frame.arrays[name].dtype == expected.dtype
        assert np.array_equal(frame.arrays[name].view(np.uint8), expected.view(np.uint8))
    fields["T"][...] = 0.0                                      # the writer's staging is reused: a copy is kept
    assert np.any(frame.arrays["T"] != 0.0)
    assert frame.attrs["TITLE"] == "x" and frame.member_id == 2
    assert frame.receipt()["schema"] == fs.SCHEMA


def test_the_sink_keeps_its_newest_frames_and_waits_for_one():
    sink = fs.FrameSink(1, fields=None, keep=2)
    for minute in (0, 5, 10):
        sink(grid_id=1, valid_time=None, time_str=f"2022-06-13_23:{minute:02d}:00", fields=_fields(),
             nx=5, ny=4, nz=3)
    assert not sink.has("2022-06-13_23:00:00") and sink.has("2022-06-13_23:10:00")
    with pytest.raises(TimeoutError, match="23:15:00"):
        sink.frame("2022-06-13_23:15:00", timeout=0.01)
    assert sink.frames_seen == 3


def test_frames_compare_word_for_word_with_signed_zero_counted():
    a = fs.MemberFrame(1, 1, "2022-06-13_23:05:00", {"T": np.zeros((2, 2), np.float32)})
    b = fs.MemberFrame(1, 1, "2022-06-13_23:05:00", {"T": -np.zeros((2, 2), np.float32),
                                                     "Times": np.zeros(19, "S1")})
    row = fs.compare_frames(a, b)
    assert row["differing_words"] == 4 and row["verdict"] == "DIFFER"
    assert row["reference_only"] == ["Times"]
    assert fs.compare_frames(a, a)["verdict"] == "bitwise"


def test_a_frame_names_a_real_instant_and_member():
    with pytest.raises(ValueError):
        fs.MemberFrame(0, 1, "2022-06-13_23:05:00", {})
    with pytest.raises(ValueError):
        fs.MemberFrame(1, 1, "23:05", {})


def test_a_frame_names_each_fields_dimensions_as_the_file_does():
    sink = fs.FrameSink(1, fields=None)
    sink(grid_id=1, valid_time=None, time_str="2022-06-13_23:05:00", fields=_fields(), nx=5, ny=4, nz=3)
    frame = sink.frame("2022-06-13_23:05:00", timeout=0)
    assert frame.dims["T"] == ("bottom_top", "south_north", "west_east")
    assert frame.dims["PH"] == ("bottom_top_stag", "south_north", "west_east")
    assert frame.dims["XLAND"] == ("south_north", "west_east")
    assert "UNUSED" in frame.arrays and "UNUSED" not in frame.dims
    assert fs.frame_dims("U", (3, 4, 6), nx=5, ny=4, nz=3) == ("bottom_top", "south_north", "west_east_stag")
    with pytest.raises(ValueError, match="no history dimension"):
        fs.frame_dims("X", (7, 7), nx=5, ny=4, nz=3)
