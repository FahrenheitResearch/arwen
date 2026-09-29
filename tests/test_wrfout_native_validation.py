"""The native reader's validation of a history file, the route a Windows Unicode path takes, on any machine.

On Windows a history file under a folder with non-ASCII letters is checked
through the native reader, because the NetCDF-C open path there cannot open
it: a valid Unicode frame failed its reopen and was quarantined as damaged.
The Windows cases are in test_wrfout_unicode_recovery.py; these send an
ASCII file through the same native route so the checks and the quarantine
rules are held wherever the reader is built:

- a complete frame passes every check (inventory, shapes, Times) and is kept;
- a frame the reader refuses as input, or one without its completion mark,
  is quarantined;
- a reader that crashes, times out or answers nonsense says nothing about
  the frame: the error is raised and the frame is left where it is.
"""

from __future__ import annotations

from functools import partial
import subprocess

import netCDF4
import numpy as np
import pytest

from gpuwm import netcdf_bridge
from gpuwm.io import wrfout

TIME = "2026-09-26_12:00:00"
FIELDS = {"T": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}


@pytest.fixture(autouse=True)
def native_route(monkeypatch):
    if netcdf_bridge.find_netcdf_bin() is None:
        pytest.skip("the Rust NetCDF reader is not built")
    monkeypatch.setattr(wrfout, "_validation_reader",
                        lambda path: (partial(netcdf_bridge.open_dataset,
                                              executable=netcdf_bridge.resolve_netcdf_bin()), True))


def _frame(folder, *, complete=1):
    path = folder / "wrfout_d01_2026-09-26_12_00_00"
    with wrfout.WrfoutWriter(path, nx=5, ny=4, nz=3, dx=1000, dy=1000, engine="python") as writer:
        writer.write_frame(TIME, FIELDS)
    if complete != 1:
        with netCDF4.Dataset(path, "a") as data:
            data.GPUWM_WRITE_COMPLETE = np.int32(complete)
    return path


def test_a_complete_frame_passes_every_check_and_is_kept(tmp_path):
    path = _frame(tmp_path)
    with netCDF4.Dataset(path) as data:
        inventory = tuple(data.variables)
        shapes = {name: var.shape for name, var in data.variables.items()}
    wrfout.validate_wrfout_file(path, inventory=inventory, shapes=shapes, times=[TIME])
    with pytest.raises(ValueError, match="inventory mismatch"):
        wrfout.validate_wrfout_file(path, inventory=["Times"], shapes=shapes, times=[TIME])
    with pytest.raises(ValueError, match="Times mismatch"):
        wrfout.validate_wrfout_file(path, inventory=inventory, shapes=shapes, times=["2026-09-26_18:00:00"])
    before = path.read_bytes()
    assert wrfout.quarantine_orphan_wrfouts(tmp_path) == ()
    assert path.read_bytes() == before


@pytest.mark.parametrize("damage", ["incomplete", "not a NetCDF file"])
def test_a_frame_the_reader_refuses_or_that_never_completed_is_quarantined(tmp_path, damage):
    path = _frame(tmp_path, complete=0 if damage == "incomplete" else 1)
    if damage != "incomplete":
        path.write_bytes(b"this is not a NetCDF file")
    moved = wrfout.quarantine_orphan_wrfouts(tmp_path)
    assert len(moved) == 1 and moved[0].is_file() and not path.exists()


@pytest.mark.parametrize("failure", ["crash", "timeout", "nonsense"])
def test_a_reader_that_fails_leaves_a_healthy_frame_where_it_is(tmp_path, monkeypatch, failure):
    path = _frame(tmp_path)
    before = path.read_bytes()
    if failure == "nonsense":
        answer = subprocess.CompletedProcess([], 0, "not json", "")
        monkeypatch.setattr(netcdf_bridge, "_run", lambda *a, **k: answer)
    else:
        original = netcdf_bridge.subprocess.run

        def fail(argv, *args, **kwargs):
            if "inventory" not in argv:
                return original(argv, *args, **kwargs)
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, 1)
            return subprocess.CompletedProcess(argv, 9, "", "reader crashed")

        monkeypatch.setattr(netcdf_bridge.subprocess, "run", fail)
    with pytest.raises((netcdf_bridge.NetcdfDecodeError, subprocess.TimeoutExpired, ValueError)):
        wrfout.quarantine_orphan_wrfouts(tmp_path)
    assert path.read_bytes() == before
    assert not (tmp_path / ".quarantine").exists()
