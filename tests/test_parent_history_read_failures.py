"""A parent history refusal blames the file only when the file is at fault.

Opening a parent history file that is truncated or corrupt refuses in one
sentence naming the file and saying to restore or regenerate it.  That
remedy is the file's.  A missing, stale or incompatible ``rw_netcdf`` (or a
``GPUWM_RW_NETCDF`` naming nothing) is the reader's failure: translated as
well, it called a healthy parent damaged, told its reader to run the parent
again (hours of GPU), and cut off the reader's own build instructions.
These tests hold both halves.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess

import netCDF4
import pytest

from conftest import requires_netcdf_bridge
from gpuwm import netcdf_bridge
from gpuwm.offline_child import (
    OfflineChildContractError,
    _ParentHistory,
    derive_child_surface_from_parent,
    interpolate_parent_boundary_snapshot,
    interpolate_parent_initial_state,
    read_parent_microphysics,
)
from test_offline_child_vertical import (
    _deep_history,
    _physics_binding,
    _placement,
)

#: Words that belong to the FILE's remedy and never to the reader's.
_FILE_REMEDY = ("restore it", "run the parent again", "damaged")


def _routes(tmp_path):
    """Every door that decodes a parent history file through the Rust reader."""

    binding = _physics_binding(tmp_path)
    return {
        "initial state": lambda path: interpolate_parent_initial_state(
            path, _placement(), physics_binding=binding, backend="cpu"),
        "boundary snapshot": lambda path: interpolate_parent_boundary_snapshot(
            path, _placement(), physics_binding=binding, backend="cpu"),
        "child surface": lambda path: derive_child_surface_from_parent(
            path, placement=_placement(), num_soil_layers=4),
        "microphysics": lambda path: read_parent_microphysics(
            path, source_mp_physics=8),
    }


_ROUTE_NAMES = ("initial state", "boundary snapshot", "child surface",
                "microphysics")


def _good_frame(tmp_path) -> Path:
    path = tmp_path / "wrfout_d01_1974-04-03_12_00_00"
    _deep_history(path, datetime(1974, 4, 3, 12), nz=8)
    return path


def _not_the_files_fault(message: str, path: Path) -> None:
    for words in _FILE_REMEDY:
        assert words not in message, (words, message)
    assert f"{path} cannot be read" not in message


# ---------------------------------------------------------------------------
# The reader's failures keep the reader's words
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", _ROUTE_NAMES)
def test_a_stale_reader_is_named_and_the_healthy_parent_is_not(
        tmp_path, monkeypatch, route):
    """A reader built before this release is refused as a reader, remedy intact."""

    frame = _good_frame(tmp_path)
    stale = tmp_path / "old-reader" / "rw_netcdf"
    stale.parent.mkdir()
    stale.write_bytes(b"#!/bin/sh\n# a reader built from an older checkout\n")
    stale.chmod(0o755)
    monkeypatch.setenv(netcdf_bridge.NETCDF_ENV, str(stale))

    with pytest.raises(netcdf_bridge.NetcdfDecodeError) as caught:
        _routes(tmp_path)[route](frame)

    assert not isinstance(caught.value, OfflineChildContractError)
    assert not isinstance(caught.value, netcdf_bridge.NetcdfFileError)
    message = str(caught.value)
    assert "Replace it with this release's reader" in message
    # The build instructions travel whole, not cut at the first line.
    assert netcdf_bridge.netcdf_remedy() in message
    _not_the_files_fault(message, frame)


@pytest.mark.parametrize("route", _ROUTE_NAMES)
def test_a_reader_override_naming_nothing_is_named_and_the_parent_is_not(
        tmp_path, monkeypatch, route):
    frame = _good_frame(tmp_path)
    missing = tmp_path / "no-such-reader" / "rw_netcdf"
    monkeypatch.setenv(netcdf_bridge.NETCDF_ENV, str(missing))

    with pytest.raises(FileNotFoundError) as caught:
        _routes(tmp_path)[route](frame)

    message = str(caught.value)
    assert netcdf_bridge.NETCDF_ENV in message and str(missing) in message
    _not_the_files_fault(message, frame)


def test_no_reader_at_all_is_the_install_refusal(tmp_path, monkeypatch):
    frame = _good_frame(tmp_path)
    monkeypatch.setattr(netcdf_bridge, "find_netcdf_bin", lambda: None)

    with pytest.raises(netcdf_bridge.NetcdfBridgeMissing) as caught:
        _routes(tmp_path)["child surface"](frame)

    assert netcdf_bridge.netcdf_remedy() in str(caught.value)
    _not_the_files_fault(str(caught.value), frame)


# ---------------------------------------------------------------------------
# The file's failures are the file's sentence
# ---------------------------------------------------------------------------


@requires_netcdf_bridge
@pytest.mark.parametrize("route", ("child surface", "microphysics"))
@pytest.mark.parametrize("damage", ("truncated HDF5", "corrupt classic"))
def test_a_damaged_parent_read_by_the_rust_reader_is_named_in_words(
        tmp_path, route, damage):
    """The reader ran and refused this file: the refusal names it and the way back."""

    frame = tmp_path / "wrfout_d01_1974-04-03_12_00_00"
    frame.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 40
                      if damage == "truncated HDF5"
                      else b"CDF\x02" + b"\xff" * 7)

    with pytest.raises(OfflineChildContractError) as caught:
        _routes(tmp_path)[route](frame)

    message = str(caught.value)
    assert message.startswith(f"{frame} cannot be read as a parent history file (")
    assert "restore it from the parent run" in message
    assert "run the parent again" in message


def test_a_parent_file_that_is_gone_is_said_to_be_gone(tmp_path, monkeypatch):
    monkeypatch.setattr(netcdf_bridge, "resolve_netcdf_bin",
                        lambda: tmp_path / "rw_netcdf")
    gone = tmp_path / "wrfout_d01_1974-04-03_12_00_00"

    with pytest.raises(OfflineChildContractError) as caught:
        _routes(tmp_path)["microphysics"](gone)

    message = str(caught.value)
    assert message.startswith(f"{gone} is not there any more")
    assert "Restore it from the parent run" in message
    assert "damaged" not in message


def test_only_a_refusal_of_this_file_is_translated(tmp_path, monkeypatch):
    """Inside the block: this file's refusal is the file's; nothing else is."""

    path = tmp_path / "wrfout_d01_1974-04-03_12_00_00"
    other = tmp_path / "some-other-file.nc"
    opened = []

    class _Dataset:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def _open(target, *, executable=None):
        opened.append((Path(target), executable))
        return _Dataset()

    reader = tmp_path / "rw_netcdf"
    monkeypatch.setattr(netcdf_bridge, "resolve_netcdf_bin", lambda: reader)
    monkeypatch.setattr(netcdf_bridge, "open_dataset", _open)

    # The resolved reader is the one the file is opened with.
    with _ParentHistory(path):
        pass
    assert opened == [(path, reader)]

    with pytest.raises(OfflineChildContractError, match="restore it"):
        with _ParentHistory(path):
            raise netcdf_bridge.NetcdfFileError(
                f"NetCDF decode failed for T in {path}: HDF error", path=path)

    stranger = netcdf_bridge.NetcdfFileError(
        f"NetCDF decode failed for X in {other}: HDF error", path=other)
    with pytest.raises(netcdf_bridge.NetcdfFileError) as caught:
        with _ParentHistory(path):
            raise stranger
    assert caught.value is stranger

    contract = netcdf_bridge.NetcdfDecodeError(
        "rw_netcdf did not acknowledge layer water conversion; rebuild the reader")
    with pytest.raises(netcdf_bridge.NetcdfDecodeError) as caught:
        with _ParentHistory(path):
            raise contract
    assert caught.value is contract


def test_a_library_opener_still_owns_its_open_failure(tmp_path):
    """netCDF4 has no separate reader: its OSError is the file's."""

    frame = tmp_path / "wrfout_d01_1974-04-03_12_00_00"
    frame.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 40)

    with pytest.raises(OfflineChildContractError) as caught:
        with _ParentHistory(frame, netCDF4.Dataset):
            pass
    assert str(caught.value).startswith(f"{frame} cannot be read")
    assert "restore it" in str(caught.value)


# ---------------------------------------------------------------------------
# A classic file cut off partway through its data
# ---------------------------------------------------------------------------


def _classic_frame(tmp_path) -> Path:
    """A CDF-2 history frame with an unlimited Time, as gpuwm and WRF write it."""

    from test_downscale_cli import _as_classic_history

    frame = _good_frame(tmp_path)
    _as_classic_history(frame)
    return frame


@requires_netcdf_bridge
def test_a_classic_frame_cut_short_is_refused_before_netcdf4_reads_zeros(
        tmp_path):
    """netCDF4 opens a cut CDF-2 file and reads its missing data as zeros.

    The contract reader proves the file whole first, so the refusal is the
    file's sentence rather than whatever a field of zeros trips later; a
    file put back whole afterwards is read again, not remembered as bad.
    """

    from gpuwm.offline_child import inspect_parent_history_frame

    frame = _classic_frame(tmp_path)
    whole = frame.read_bytes()
    inspect_parent_history_frame(frame)
    frame.write_bytes(whole[:-5])
    with netCDF4.Dataset(frame) as dataset:
        # What made this silent: the header survives, so the file opens.
        assert "Times" in dataset.variables
    with pytest.raises(OfflineChildContractError) as caught:
        inspect_parent_history_frame(frame)
    message = str(caught.value)
    assert message.startswith(f"{frame} cannot be read as a parent history file (")
    assert "restore it from the parent run" in message
    frame.write_bytes(whole)
    assert inspect_parent_history_frame(frame).path == frame


@requires_netcdf_bridge
def test_a_reader_older_than_the_extent_check_is_asked_for_the_last_bytes(
        tmp_path, monkeypatch):
    """An installed reader that predates the check still refuses the cut file.

    Its inventory reads only the header, which a cut file keeps, so it
    answers as it did for the whole file and says nothing of the extent;
    the variable stored last is then decoded, and that decode checks the
    same final bytes.
    """

    from gpuwm import offline_child

    frame = _classic_frame(tmp_path)
    reader = netcdf_bridge.resolve_netcdf_bin()
    header_only = netcdf_bridge._run(
        [str(reader), "inventory", str(frame)], what="inventory")
    document = json.loads(header_only.stdout)
    assert document.pop("extent_checked") is True
    with netcdf_bridge.open_dataset(frame) as dataset:
        last = offline_child._last_stored_variable(dataset)
        # The last record variable, since the records follow the fixed ones.
        records = [name for name, variable in dataset.variables.items()
                   if variable.dimensions[:1] == ("Time",)]
        assert last == records[-1]
    real_run = netcdf_bridge._run

    def older_reader(arguments, **kwargs):
        if arguments[1] == "inventory":
            return subprocess.CompletedProcess(
                arguments, 0, stdout=json.dumps(document), stderr="")
        return real_run(arguments, **kwargs)

    monkeypatch.setattr(netcdf_bridge, "_run", older_reader)
    whole = frame.read_bytes()
    frame.write_bytes(whole[:-5])
    with pytest.raises(OfflineChildContractError) as caught:
        offline_child.inspect_parent_history_frame(frame)
    message = str(caught.value)
    assert message.startswith(f"{frame} cannot be read as a parent history file (")
    assert "beyond file" in message
    assert "restore it from the parent run" in message


@pytest.mark.parametrize("status, expected", [
    (2, "file"), (1, "reader"), (101, "reader"), (-9, "reader")])
def test_only_the_readers_own_refusal_of_a_named_file_blames_that_file(
        tmp_path, status, expected):
    """Exit status 2 is ``rw_netcdf`` refusing what it was given; any other
    status (a panic, an error exit, a kill) is the reader failing, and says
    nothing about the file it was reading.  The history-file sweep
    quarantines on the first and never on the second, so the parent read
    must draw the same line."""

    import sys

    if status < 0 and sys.platform == "win32":
        pytest.skip("a process killed by a signal is a POSIX status")
    frame = tmp_path / "parent.nc"
    frame.write_bytes(b"CDF\x02")
    code = (f"import os, sys; sys.stderr.write('rw_netcdf: refused'); "
            f"sys.stderr.flush(); "
            + (f"os.kill(os.getpid(), {-status})" if status < 0
               else f"sys.exit({status})"))
    with pytest.raises(netcdf_bridge.NetcdfDecodeError) as caught:
        netcdf_bridge._run([sys.executable, "-c", code], what="inventory",
                           file=frame)
    blamed = isinstance(caught.value, netcdf_bridge.NetcdfFileError)
    assert blamed is (expected == "file")
    assert isinstance(caught.value, netcdf_bridge.NetcdfInputError) is blamed
    if blamed:
        assert caught.value.path == frame
        assert caught.value.reason == "refused"


def test_the_readers_refusal_without_a_named_file_is_an_input_refusal():
    import sys

    with pytest.raises(netcdf_bridge.NetcdfInputError) as caught:
        netcdf_bridge._run(
            [sys.executable, "-c", "import sys; sys.exit(2)"], what="usage")
    assert not isinstance(caught.value, netcdf_bridge.NetcdfFileError)
