"""Unicode Windows publication must validate real Rust tapes without copying them."""
import os
from pathlib import Path
import shutil
import subprocess

import netCDF4
import numpy as np
import pytest

from gpuwm import netcdf_bridge
from gpuwm.io import nc_writer_bridge, wrfout

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows NetCDF-C filename decoding")
TIME = "2026-09-26_12:00:00"
FIELDS = {"T": np.arange(60, dtype=np.float32).reshape(3, 4, 5)}


@pytest.fixture(autouse=True)
def native_reader_and_writer(monkeypatch):
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.delenv(wrfout.WRFOUT_WRITER_ENV, raising=False)
    reason = nc_writer_bridge.unavailable_reason()
    if reason:
        pytest.skip(f"Rust writer is not built: {reason}")
    if netcdf_bridge.find_netcdf_bin() is None:
        pytest.skip("Rust NetCDF reader is not built")


def write_frame(path):
    with wrfout.WrfoutWriter(path, nx=5, ny=4, nz=3, dx=1000, dy=1000) as writer:
        writer.write_frame(TIME, FIELDS)


def unicode_copy(tmp_path, *, complete=1):
    original = tmp_path / "ascii" / "wrfout_d01_reference.nc"
    write_frame(original)
    if complete != 1:
        with netCDF4.Dataset(original, "a") as data:
            data.GPUWM_WRITE_COMPLETE = np.int32(complete)
    target = tmp_path / "René Ω" / original.name
    target.parent.mkdir()
    shutil.copyfile(original, target)
    return target


@pytest.mark.parametrize("folder", ["Omega-Ω", "München René", "long-Ω/" + "/".join(["directory" * 5] * 5)])
def test_default_rust_writer_publishes_unicode_paths(tmp_path, folder):
    target = tmp_path / folder / "wrfout_d01_unicode.nc"
    if folder.startswith("long-"):
        # The runtime's established Windows I/O spelling for deep output trees.
        from gpuwm.filesystem_paths import io_path
        target = io_path(target)
    write_frame(target)
    assert target.is_file()
    with netcdf_bridge.open_dataset(target) as data:
        assert data.GPUWM_WRITE_COMPLETE == 1
        np.testing.assert_array_equal(data.variables["T"][:][0], FIELDS["T"])
        assert data.variables["Times"][:][0].tobytes().decode("ascii") == TIME
    assert not list(target.parent.glob(".wrfout*"))


def test_quarantine_preserves_a_valid_unicode_frame(tmp_path):
    target = unicode_copy(tmp_path)
    before = target.read_bytes()
    assert wrfout.quarantine_orphan_wrfouts(target.parent) == ()
    assert target.read_bytes() == before


def test_unicode_validation_and_quarantine_accept_existing_python_hdf5(tmp_path):
    original = tmp_path / "legacy" / "wrfout_d01_hdf5.nc"
    with wrfout.WrfoutWriter(original, nx=5, ny=4, nz=3, dx=1000, dy=1000,
                             engine="python") as writer:
        writer.write_frame(TIME, FIELDS)
    with netCDF4.Dataset(original) as data:
        inventory = tuple(data.variables)
        shapes = {name: var.shape for name, var in data.variables.items()}
    target = tmp_path / "legacy-Ω" / original.name
    target.parent.mkdir()
    shutil.copyfile(original, target)
    before = target.read_bytes()
    wrfout.validate_wrfout_file(target, inventory=inventory, shapes=shapes, times=[TIME])
    assert wrfout.quarantine_orphan_wrfouts(target.parent) == ()
    assert target.read_bytes() == before


@pytest.mark.parametrize("damage", ["incomplete", "corrupt"])
def test_quarantine_still_removes_actual_invalid_frames(tmp_path, damage):
    target = unicode_copy(tmp_path, complete=0 if damage == "incomplete" else 1)
    if damage == "corrupt":
        target.write_bytes(b"this is not a NetCDF file")
    moved = wrfout.quarantine_orphan_wrfouts(target.parent)
    assert len(moved) == 1 and moved[0].is_file()
    assert not target.exists()


def test_unicode_validation_decodes_only_times_and_keeps_every_check(tmp_path, monkeypatch):
    target = unicode_copy(tmp_path)
    with netcdf_bridge.open_dataset(target) as data:
        inventory = tuple(data.variables)
        shapes = {name: var.shape for name, var in data.variables.items()}
    decoded = []
    original = netcdf_bridge.Dataset._decode
    def read(self, name, **kwargs):
        decoded.append(name)
        assert name == "Times", "Validation must not decode meteorological arrays"
        return original(self, name, **kwargs)
    monkeypatch.setattr(netcdf_bridge.Dataset, "_decode", read)
    wrfout.validate_wrfout_file(target, inventory=inventory, shapes=shapes, times=[TIME])
    assert decoded == ["Times"]
    with pytest.raises(ValueError, match="inventory mismatch"):
        wrfout.validate_wrfout_file(target, inventory=["Times"], shapes=shapes, times=[TIME])
    with pytest.raises(ValueError, match="shape"):
        wrfout.validate_wrfout_file(target, inventory=inventory, shapes={**shapes, "T": (1, 1, 1, 1)}, times=[TIME])
    with pytest.raises(ValueError, match="Times mismatch"):
        wrfout.validate_wrfout_file(target, inventory=inventory, shapes=shapes, times=["2026-09-26_18:00:00"])


@pytest.mark.parametrize("failure", ["missing", "crash", "timeout", "json", "schema"])
def test_decoder_failure_does_not_quarantine_a_healthy_frame(tmp_path, monkeypatch, failure):
    target = unicode_copy(tmp_path)
    before = target.read_bytes()
    if failure == "missing":
        monkeypatch.setenv(netcdf_bridge.NETCDF_ENV, str(tmp_path / "missing-reader.exe"))
    else:
        reader = netcdf_bridge.resolve_netcdf_bin()
        monkeypatch.setattr(netcdf_bridge, "resolve_netcdf_bin", lambda: reader)
        if failure in ("json", "schema"):
            result = subprocess.CompletedProcess([], 0, "not json" if failure == "json" else '{"schema":"wrong"}', "")
            monkeypatch.setattr(netcdf_bridge, "_run", lambda *a, **k: result)
        else:
            original = netcdf_bridge.subprocess.run
            def fail(argv, *args, **kwargs):
                if "inventory" not in argv:
                    return original(argv, *args, **kwargs)
                if failure == "timeout":
                    raise subprocess.TimeoutExpired(argv, 1)
                return subprocess.CompletedProcess(argv, 9, "", "reader crashed")
            monkeypatch.setattr(netcdf_bridge.subprocess, "run", fail)
    with pytest.raises((FileNotFoundError, netcdf_bridge.NetcdfBridgeMissing, netcdf_bridge.NetcdfDecodeError, subprocess.TimeoutExpired)):
        wrfout.quarantine_orphan_wrfouts(target.parent)
    assert target.read_bytes() == before
    assert not (target.parent / ".quarantine").exists()
