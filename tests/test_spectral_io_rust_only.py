"""The spectral score door reads its gridded planes in Rust, not netCDF4.

``gpuwm spectral score`` / ``run`` opens a forecast history file and pulls a
whole gridded plane out of it -- a decode, on the bare default of a shipped
door.  Until the 2026-08-18 boundary audit (finding F2) it did that with
``netCDF4.Dataset``, which is the C library, while every other field read in
the estate had already moved to ``rw_netcdf``.

These tests pin the flip, in :mod:`gpuwm.downscale`'s shape: FIELD values come
through :mod:`gpuwm.netcdf_bridge`, and the door never reaches a Python NetCDF
library to get them.  The concrete breakage they prevent: two decoders reading
the same wrfout, so a spectral receipt could disagree with the renderer and the
verifier about what the model actually wrote, with nothing in the receipt
saying which library produced the numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from gpuwm import netcdf_bridge
from gpuwm.verify import spectral_io

netCDF4 = pytest.importorskip("netCDF4")


def _require_bridge() -> None:
    if netcdf_bridge.find_netcdf_bin() is None:
        pytest.skip("rw_netcdf is not built here")


def _write_history(path: Path, *, values: np.ndarray,
                   fill: float | None = None) -> None:
    """A wrfout-shaped history file: suffixless, staggered, time-first."""

    nz, ny, nx = values.shape
    with netCDF4.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("bottom_top", nz)
        dataset.createDimension("south_north", ny)
        dataset.createDimension("west_east_stag", nx)
        variable = dataset.createVariable(
            "U", "f4",
            ("Time", "bottom_top", "south_north", "west_east_stag"),
            **({} if fill is None else {"fill_value": np.float32(fill)}))
        variable.units = "m s-1"
        variable.stagger = "X"
        variable[0] = values.astype(np.float32)


def _plane(n: int = 12) -> np.ndarray:
    x = np.arange(n, dtype=np.float64)
    base = np.cos(2.0 * np.pi * 3 * x / n)
    return np.stack([base * (level + 1) for level in range(3)]).reshape(
        3, 1, n) * np.ones((3, n, n))


class _ForbiddenNetcdf4:
    """Any attribute touch is the failure, and it says which one."""

    def __getattr__(self, name: str):
        raise AssertionError(
            "the bare `gpuwm spectral score` field read reached the netCDF4 "
            f"module (asked for {name!r})")


def test_the_bare_default_never_opens_netcdf4(tmp_path: Path, monkeypatch):
    """The positive claim, not merely the absence of an import line.

    A module can drop ``import netCDF4`` from its header and still reach the
    C library through a function-local import, which is exactly the spelling
    this reader used.  Poisoning ``sys.modules`` catches both.
    """

    _require_bridge()
    source = tmp_path / "wrfout_d02_2020-01-01_00_00_00"
    values = _plane()
    _write_history(source, values=values)

    monkeypatch.setitem(sys.modules, "netCDF4", _ForbiddenNetcdf4())
    array, metadata = spectral_io.load_array(source, variable="U",
                                             time_index=0)
    assert array.shape == values.shape
    assert metadata["format"] == "netcdf"
    assert metadata["netcdf_identified_by"] == "signature"
    assert metadata["stagger_attribute"] == "X"
    assert metadata["units"] == "m s-1"
    assert metadata["dimensions"] == [
        "bottom_top", "south_north", "west_east_stag"]


def test_the_rust_read_is_bit_identical_to_the_c_library(tmp_path: Path):
    """Same file, same numbers: the flip may not move a single receipt bit.

    The spectral receipts are self-hashed and campaign-registered, so a read
    that differed in the last bit would silently invalidate every prior
    receipt while every gate still said PASS.
    """

    _require_bridge()
    source = tmp_path / "wrfout_d02_2020-01-01_00_00_00"
    values = _plane()
    _write_history(source, values=values)

    with netCDF4.Dataset(source) as dataset:
        expected = np.asarray(
            dataset.variables["U"][0], dtype=np.float64)

    array, _ = spectral_io.load_array(source, variable="U", time_index=0)
    assert array.dtype == np.float64
    assert array.tobytes() == expected.tobytes()


def test_a_fill_valued_plane_is_still_refused(tmp_path: Path):
    """Missing data stays a refusal, not a silently filled sample.

    The reader's contract is that every scored value is a real one.  The C
    library expressed that as a masked array; the bridge expresses it as
    NaN.  Either way the plane must not be scored.
    """

    _require_bridge()
    source = tmp_path / "wrfout_d02_2020-01-01_00_00_00"
    values = _plane()
    values[1, 2, 3] = -9999.0
    _write_history(source, values=values, fill=-9999.0)

    with pytest.raises(ValueError, match="non-finite|masked"):
        spectral_io.load_array(source, variable="U", time_index=0)


def test_a_missing_bridge_is_a_named_refusal(tmp_path: Path, monkeypatch):
    """No fallback to a second decoder, and the refusal says what to build.

    The breakage a fallback would cause: the spectral receipt's numbers
    would depend on which library happened to be installed on the box that
    scored it, and the receipt records neither.
    """

    source = tmp_path / "wrfout_d02_2020-01-01_00_00_00"
    _write_history(source, values=_plane())
    monkeypatch.setattr(netcdf_bridge, "find_netcdf_bin", lambda: None)
    with pytest.raises(netcdf_bridge.NetcdfBridgeMissing) as caught:
        spectral_io.load_array(source, variable="U", time_index=0)
    assert "rw_netcdf" in str(caught.value)
