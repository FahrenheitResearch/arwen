"""The check door reads source orography VALUES in Rust, not netCDF4.

``gpuwm check`` validates the shape, stored type and finiteness of the source
orography field every domain will be initialised from.  Until the 2026-08-18
boundary audit (HIT-5) it pulled those values out with ``netCDF4.Dataset`` on
its bare default -- a gridded meteorological field, decoded by the C library,
on a shipped door.

The concrete breakage these tests prevent: the check door and the run door
reading the same orography file with different decoders, so a file this door
called finite could still reach the model as something else -- and the
disagreement would surface as a blown-up integration, not as a check failure.

``preflight`` keeps ``netCDF4`` for the RRTMGP asset-table sweep, so the poison
here is aimed where it belongs: at ``netCDF4.Dataset``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import netcdf_bridge
from gpuwm.ingest import preflight

netCDF4 = pytest.importorskip("netCDF4")


def _require_bridge() -> None:
    if netcdf_bridge.find_netcdf_bin() is None:
        pytest.skip("rw_netcdf is not built here")


NY, NX = 20, 24


def _case(path: Path, variable: str = "SOILHGT"):
    domain = SimpleNamespace(grid_id=1, run=SimpleNamespace(ny=NY, nx=NX))
    exp = SimpleNamespace(domains=(domain,), domain=lambda _id: domain)
    case_data = SimpleNamespace(
        source_orography=SimpleNamespace(path=str(path), variable=variable),
        output_domain=1)
    catalog = SimpleNamespace(inventory=())
    return exp, case_data, catalog


def _write(path: Path, values: np.ndarray, *, dtype: str = "f4",
           dims=("Time", "south_north", "west_east")) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        for name, size in zip(dims, values.shape):
            dataset.createDimension(name, size)
        variable = dataset.createVariable("SOILHGT", dtype, dims)
        variable[:] = values


class _ForbiddenDataset:
    def __call__(self, *_args, **_kwargs):
        raise AssertionError(
            "the bare `gpuwm check` orography read opened netCDF4.Dataset")


def test_a_clean_orography_passes_without_opening_netcdf4(tmp_path: Path,
                                                          monkeypatch):
    """The positive claim: the default path validates, and on Rust."""

    _require_bridge()
    source = tmp_path / "invariant.nc"
    _write(source, np.full((1, NY, NX), 137.5, dtype=np.float32))

    monkeypatch.setattr(netCDF4, "Dataset", _ForbiddenDataset())
    issues, checks = preflight._check_orography(*_case(source))
    assert issues == []
    assert any("d01" in item for item in checks)


def test_a_non_finite_orography_is_still_caught(tmp_path: Path, monkeypatch):
    """The check keeps its teeth: NaN is named, with file, variable, index."""

    _require_bridge()
    source = tmp_path / "invariant.nc"
    values = np.full((1, NY, NX), 137.5, dtype=np.float32)
    values[0, 3, 5] = np.nan
    _write(source, values)

    monkeypatch.setattr(netCDF4, "Dataset", _ForbiddenDataset())
    issues, _ = preflight._check_orography(*_case(source))
    assert [item.code for item in issues] == ["orography-nonfinite"]
    assert issues[0].index == (3, 5)
    assert issues[0].variable == "SOILHGT"


def test_a_wrong_shape_is_still_caught_without_decoding(tmp_path: Path,
                                                        monkeypatch):
    """Shape is settled from the inventory, so no plane is decoded at all."""

    _require_bridge()
    source = tmp_path / "invariant.nc"
    _write(source, np.full((1, NY - 1, NX), 1.0, dtype=np.float32))

    monkeypatch.setattr(netCDF4, "Dataset", _ForbiddenDataset())
    issues, _ = preflight._check_orography(*_case(source))
    assert [item.code for item in issues] == ["orography-shape"]


def test_an_absent_variable_names_what_the_file_does_carry(tmp_path: Path,
                                                           monkeypatch):
    _require_bridge()
    source = tmp_path / "invariant.nc"
    _write(source, np.full((1, NY, NX), 1.0, dtype=np.float32))

    monkeypatch.setattr(netCDF4, "Dataset", _ForbiddenDataset())
    issues, _ = preflight._check_orography(
        *_case(source, variable="HGT_M"))
    assert [item.code for item in issues] == ["orography-variable"]
    assert "SOILHGT" in issues[0].message


def test_the_rust_values_are_bit_identical_to_the_c_library(tmp_path: Path):
    """A read that moved a bit would move what 'finite' means at the edge."""

    _require_bridge()
    source = tmp_path / "invariant.nc"
    values = (np.arange(NY * NX, dtype=np.float32).reshape(1, NY, NX)
              * np.float32(0.1))
    _write(source, values)

    with netCDF4.Dataset(source) as dataset:
        expected = np.asarray(dataset.variables["SOILHGT"][0],
                              dtype=np.float64)
    with netcdf_bridge.open_dataset(source) as dataset:
        actual = np.asarray(dataset.variables["SOILHGT"][0])
    assert actual.tobytes() == expected.tobytes()
