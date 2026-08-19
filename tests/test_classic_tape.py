"""The one netCDF4-shaped facade over the Rust classic writer.

Two products drive it -- the wrfout history tape and the
wrfinput/wrfbdy export -- so what it refuses matters as much as what it
writes.  These pin the shared surface itself; each product's adoption is
pinned in its own suite (``test_wrfout_engine.py``,
``test_wrf_direct_writer.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from gpuwm.io import nc_writer_bridge
from gpuwm.io.classic_tape import ClassicTape, classic_attr_value

_RUST_UNAVAILABLE = nc_writer_bridge.unavailable_reason()


def _require_rust_writer():
    if _RUST_UNAVAILABLE:
        pytest.skip(f"the Rust NetCDF writer library is not built: "
                    f"{_RUST_UNAVAILABLE}")


def _tape(path):
    _require_rust_writer()
    tape = ClassicTape(path)
    tape.createDimension("Time", None)
    tape.createDimension("south_north", 2)
    tape.createDimension("west_east", 3)
    return tape


def test_a_fully_sliced_subscript_writes_the_same_record_as_a_bare_index(
        tmp_path):
    """``var[t, :, :] = arr`` is the wrf_direct spelling; ``var[t] = arr``
    is the wrfout one.  One facade, both spellings, same bytes."""
    field = np.arange(6, dtype="f4").reshape(2, 3)
    written = {}
    for name, key in (("sliced.nc", (0, slice(None), slice(None))),
                      ("bare.nc", 0)):
        path = tmp_path / name
        tape = _tape(path)
        variable = tape.createVariable(
            "T2", "f4", ("Time", "south_north", "west_east"))
        variable[key] = field
        tape.close()
        written[name] = path.read_bytes()
    assert written["sliced.nc"] == written["bare.nc"]


def test_a_partial_slab_is_refused_by_name(tmp_path):
    """The concrete breakage: the classic format has no way to say a
    region was never filled, so the untouched half of the slab would read
    back as whatever the filesystem left there."""
    tape = _tape(tmp_path / "partial.nc")
    variable = tape.createVariable(
        "T2", "f4", ("Time", "south_north", "west_east"))
    with pytest.raises(IndexError, match="whole record"):
        variable[0, 0, :] = np.zeros(3, dtype="f4")
    tape.abort()


def test_a_short_subscript_is_refused_rather_than_broadcast(tmp_path):
    tape = _tape(tmp_path / "short.nc")
    variable = tape.createVariable(
        "T2", "f4", ("Time", "south_north", "west_east"))
    with pytest.raises(IndexError, match="dimension"):
        variable[0, slice(None)] = np.zeros((2, 3), dtype="f4")
    tape.abort()


def test_hdf5_storage_keywords_are_refused_not_ignored(tmp_path):
    """A caller who believes a file is compressed when it is not has been
    told something untrue about it."""
    tape = _tape(tmp_path / "compressed.nc")
    with pytest.raises(TypeError, match="zlib"):
        tape.createVariable("T2", "f4", ("Time", "south_north", "west_east"),
                            zlib=True, complevel=2)
    tape.abort()


def test_the_header_freezes_at_the_first_data_byte(tmp_path):
    tape = _tape(tmp_path / "frozen.nc")
    variable = tape.createVariable(
        "T2", "f4", ("Time", "south_north", "west_east"))
    variable[0] = np.zeros((2, 3), dtype="f4")
    with pytest.raises(RuntimeError, match="HGT"):
        tape.createVariable("HGT", "f4", ("south_north", "west_east"))
    with pytest.raises(RuntimeError, match="TITLE"):
        tape.setncattr("TITLE", "too late")
    tape.abort()


def test_no_completion_attribute_unless_the_caller_asks_for_one(tmp_path):
    """The history tape stamps GPUWM_WRITE_COMPLETE; a wrfinput must not
    carry an attribute stock WRF's own real.exe never writes."""
    path = tmp_path / "plain.nc"
    tape = _tape(path)
    tape.setncattr("TITLE", " OUTPUT FROM GPUWM")
    variable = tape.createVariable(
        "T2", "f4", ("Time", "south_north", "west_east"))
    variable[0] = np.zeros((2, 3), dtype="f4")
    tape.close()
    with netCDF4.Dataset(path) as ds:
        assert ds.ncattrs() == ["TITLE"]

    stamped = tmp_path / "stamped.nc"
    tape = ClassicTape(stamped, completion_attr="GPUWM_WRITE_COMPLETE")
    tape.createDimension("Time", None)
    tape.createDimension("south_north", 2)
    variable = tape.createVariable("HGT", "f4", ("Time", "south_north"))
    variable[0] = np.zeros(2, dtype="f4")
    tape.close()
    with netCDF4.Dataset(stamped) as ds:
        assert int(getattr(ds, "GPUWM_WRITE_COMPLETE")) == 1


def test_a_bare_python_number_takes_the_width_classic_can_hold():
    # numpy 2 spells a bare int int64, which CDF-1/CDF-2 cannot hold at
    # all; netCDF4-python stores it as NC_INT.
    assert classic_attr_value(7).dtype == np.int32
    assert classic_attr_value(True).dtype == np.int32
    assert classic_attr_value(2.5).dtype == np.float64
    assert classic_attr_value(np.float32(2.5)).dtype == np.float32
    assert classic_attr_value("text") == "text"


def test_an_aborted_tape_leaves_numrecs_at_zero(tmp_path):
    path = tmp_path / "aborted.nc"
    tape = _tape(path)
    variable = tape.createVariable(
        "T2", "f4", ("Time", "south_north", "west_east"))
    variable[0] = np.zeros((2, 3), dtype="f4")
    tape.abort()
    with netCDF4.Dataset(path) as ds:
        assert len(ds.dimensions["Time"]) == 0
