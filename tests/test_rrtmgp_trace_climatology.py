"""RRTMGP's trace-gas climatology: a derived table instead of the RFMIP file.

Until 2.8.0 the column driver opened the RFMIP clear-sky input NetCDF at
model construction and took 136 numbers from it.  That file is no longer
carried (its licence attribute is ambiguous, gpuwm/core/rfmip_upstream.py),
so the numbers ship as ``rrtmgp-trace-gas-climatology.json``.  These tests
hold the three things that swap must not change: the numbers are the ones
the driver used to compute, bit for bit; the driver refuses a table nobody
derived; and a checkpoint written while the driver read the NetCDF still
names the same asset.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from gpuwm import data_assets
from gpuwm.core import rfmip_upstream
from gpuwm.core.trace_gases import RFMIP_GAS_NAMES

TABLE = data_assets.rrtmgp_data_dir() / "rrtmgp-trace-gas-climatology.json"


def test_the_table_matches_its_pin_and_names_its_source():
    from gpuwm.core.rrtmgp import load_trace_climatology

    assert hashlib.sha256(TABLE.read_bytes()).hexdigest() \
        == rfmip_upstream.TRACE_CLIMATOLOGY_SHA256
    climatology = load_trace_climatology()
    assert set(climatology.trace_vmr) == set(RFMIP_GAS_NAMES)
    assert all(0.0 < value < 1.0 for value in climatology.trace_vmr.values())
    assert climatology.pressure_layer_pa.shape == (60,)
    assert climatology.ozone_vmr.shape == (60,)
    assert climatology.pressure_layer_pa.dtype == np.float64
    assert np.all(np.isfinite(climatology.ozone_vmr))
    _url, size, sha256, _licence = \
        rfmip_upstream.RFMIP_FILES["rfmip-clear-sky-inputs.nc"]
    assert rfmip_upstream.TRACE_CLIMATOLOGY_SOURCE == {
        "path": "data/rrtmgp/rfmip-clear-sky-inputs.nc",
        "bytes": size, "sha256": sha256}


def test_the_table_is_what_the_driver_computed_from_the_netcdf():
    """Re-derived from the pinned upstream file: byte-identical text.

    JSON keeps each float64 exactly, so byte equality of the rendered
    table is equality of every number the driver reads.  Also checked the
    long way: the old in-driver arithmetic, value by value.
    """

    from netCDF4 import Dataset

    try:
        inputs = rfmip_upstream.fetch_rfmip("rfmip-clear-sky-inputs.nc")
    except rfmip_upstream.RfmipUnavailable as error:
        pytest.skip(f"RFMIP upstream file unavailable: {error}")
    import importlib.util
    tool_path = (Path(__file__).resolve().parents[1] / "tools"
                 / "derive_rrtmgp_trace_climatology.py")
    spec = importlib.util.spec_from_file_location("derive_tool", tool_path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    rendered = tool.render_trace_climatology(
        tool.derive_trace_climatology(inputs))
    assert rendered.encode("utf-8") == TABLE.read_bytes()

    from gpuwm.core.rrtmgp import load_trace_climatology
    climatology = load_trace_climatology()
    with Dataset(inputs, "r") as ncfile:
        ncfile.set_auto_mask(False)
        for gas, rfmip_name in RFMIP_GAS_NAMES.items():
            variable = ncfile[rfmip_name + "_GM"]
            scale = float(getattr(variable, "units", "1").replace(" ", ""))
            assert climatology.trace_vmr[gas] == float(variable[0]) * scale
        pressure = np.median(
            np.asarray(ncfile["pres_layer"][:], np.float64), axis=0)
        ozone = np.median(np.asarray(ncfile["ozone"][0], np.float64), axis=0)
    assert climatology.pressure_layer_pa.tobytes() == pressure.tobytes()
    assert climatology.ozone_vmr.tobytes() == ozone.tobytes()


def test_a_table_nobody_derived_is_refused(monkeypatch, tmp_path):
    from gpuwm.core import rrtmgp

    edited = tmp_path / "rrtmgp-trace-gas-climatology.json"
    edited.write_bytes(TABLE.read_bytes().replace(b'"cf4": ', b'"cf4":  '))
    monkeypatch.setattr(rrtmgp, "_table", lambda name: edited)
    rrtmgp.load_trace_climatology.cache_clear()
    try:
        with pytest.raises(ValueError, match="nobody derived"):
            rrtmgp.load_trace_climatology()
    finally:
        rrtmgp.load_trace_climatology.cache_clear()


def test_a_pre_2_8_checkpoint_still_names_the_same_asset():
    """The pinned table records the source file; other bytes record themselves."""

    from gpuwm.io import restart

    relative = restart.PHYSICS_ASSET_PATHS["rrtmgp_rfmip"]
    assert relative.name == "rrtmgp-trace-gas-climatology.json"
    recorded = restart._recorded_asset_identity(
        "rrtmgp_rfmip", relative, TABLE.stat().st_size,
        rfmip_upstream.TRACE_CLIMATOLOGY_SHA256)
    # Exactly what every RRTMGP manifest written before 2.8.0 carries.
    assert recorded == {
        "path": "data/rrtmgp/rfmip-clear-sky-inputs.nc",
        "bytes": 1859666,
        "sha256": "b8dc05d7cd2e0e6354b4a6198771ddf3bc09f18d72b49f20a41e2024e2fd51f4",
    }
    other = restart._recorded_asset_identity(
        "rrtmgp_rfmip", relative, 10, "0" * 64)
    assert other == {"path": relative.as_posix(), "bytes": 10,
                     "sha256": "0" * 64}
    gas = restart._recorded_asset_identity(
        "rrtmgp_gas_lw", Path("data/rrtmgp/rrtmgp-gas-lw-g256.nc"), 5,
        rfmip_upstream.TRACE_CLIMATOLOGY_SHA256)
    assert gas["path"] == "data/rrtmgp/rrtmgp-gas-lw-g256.nc"


def test_the_fetch_refuses_bytes_nobody_pinned(tmp_path):
    wrong = tmp_path / "rfmip-clear-sky-inputs.nc"
    wrong.write_bytes(b"not the file")
    with pytest.raises(rfmip_upstream.RfmipUnavailable, match="refused"):
        rfmip_upstream.fetch_rfmip("rfmip-clear-sky-inputs.nc", path=wrong)
    with pytest.raises(rfmip_upstream.RfmipUnavailable, match="fetched from"):
        rfmip_upstream.fetch_rfmip(
            "rfmip-clear-sky-inputs.nc", path=tmp_path / "absent.nc")
