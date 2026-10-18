"""The hex-to-ArWen parent route: a gpuwm-hex export frame decodes through the
packaged hex-global profile with no per-source code.

The breakage this holds closed: before it, a hex frame had no 1-D coordinates,
the NetCDF mapped route could not read half-level (interface) heights, and the
closed derivation catalog had no theta-to-T or mixing-ratio-to-q entry, so no
ArWen domain could take a hex run as its initial and boundary source.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from gpuwm import mapped_source as ms
from gpuwm.source_adapters import get_source_adapter
from gpuwm.source_authorities import packaged_authorities

NZ, NY, NX, NSOIL = 55, 3, 4, 4
EPOCH_SECONDS = 1680285600  # 2023-03-31 18:00:00 UTC


def _frame(path: Path, epoch: int = EPOCH_SECONDS) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    k = np.arange(NZ, dtype=np.float64)[:, None, None]
    shape3 = (NZ, NY, NX)
    pressure = 100000.0 * np.exp(-(k + 0.5) / 9.0) * np.ones(shape3)
    theta = 295.0 + 2.0 * k + rng.uniform(0.0, 1.0, shape3)
    qv = 0.012 * np.exp(-k / 6.0) * np.ones(shape3)
    zgrid = 260.0 * np.arange(NZ + 1, dtype=np.float64)[:, None, None] + 40.0 * np.ones((NZ + 1, NY, NX))
    tsk = 290.0 + rng.uniform(0.0, 5.0, (NY, NX))
    q2 = np.full((NY, NX), 0.011)
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_DATA") as ds:
        for name, size in (("Time", None), ("west_east", NX), ("south_north", NY), ("bottom_top", NZ),
                           ("bottom_top_stag", NZ + 1), ("soil_layers_stag", NSOIL)):
            ds.createDimension(name, size)

        def coord(name, dim, values, units, **attrs):
            v = ds.createVariable(name, "f8", (dim,))
            v.units = units
            for key, value in attrs.items():
                setattr(v, key, value)
            v[:] = values

        coord("lat", "south_north", [35.0, 36.0, 37.0], "degrees_north", standard_name="latitude")
        coord("lon", "west_east", [-93.0, -92.0, -91.0, -90.0], "degrees_east", standard_name="longitude")
        coord("Time", "Time", [epoch], "seconds since 1970-01-01 00:00:00",
              calendar="standard", standard_name="time")
        coord("bottom_top", "bottom_top", np.arange(1, NZ + 1), "1")
        coord("bottom_top_stag", "bottom_top_stag", np.arange(1, NZ + 2), "1")
        coord("soil_layers_stag", "soil_layers_stag", [0.0, 10.0, 40.0, 100.0], "cm")

        def var(name, dims, values, units):
            v = ds.createVariable(name, "f4", ("Time", *dims))
            v.units = units
            v[:] = np.asarray(values, dtype=np.float32)[None]

        m3 = ("bottom_top", "south_north", "west_east")
        s2 = ("south_north", "west_east")
        var("T", m3, theta - 300.0, "K")
        var("PB", m3, pressure, "Pa")
        var("QVAPOR", m3, qv, "kg kg-1")
        for name in ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"):
            var(name, m3, np.zeros(shape3), "kg kg-1")
        var("RADAR_U_EARTH", m3, np.full(shape3, 10.0), "m s-1")
        var("RADAR_V_EARTH", m3, np.full(shape3, -3.0), "m s-1")
        var("PHB", ("bottom_top_stag", "south_north", "west_east"), 9.80665 * zgrid, "m2 s-2")
        var("PSFC", s2, np.full((NY, NX), 99000.0), "Pa")
        var("HGT", s2, np.full((NY, NX), 40.0), "m")
        var("TSK", s2, tsk, "K")
        var("T2", s2, tsk - 1.0, "K")
        var("Q2", s2, q2, "kg kg-1")
        var("U10", s2, np.full((NY, NX), 4.0), "m s-1")
        var("V10", s2, np.full((NY, NX), 1.0), "m s-1")
        var("LANDMASK", s2, np.array([[1, 1, 0, 0]] * NY), "1")
        var("SEAICE", s2, np.zeros((NY, NX)), "1")
        var("SNOW", s2, np.zeros((NY, NX)), "kg m-2")
        var("SNOWH", s2, np.zeros((NY, NX)), "m")
        soil = ("soil_layers_stag", "south_north", "west_east")
        var("TSLB", soil, np.full((NSOIL, NY, NX), 285.0), "K")
        var("SMOIS", soil, np.full((NSOIL, NY, NX), 0.3), "m3 m-3")
    return dict(pressure=pressure, theta=theta, qv=qv, zgrid=zgrid, tsk=tsk, q2=q2)


def test_the_adapter_row_names_the_packaged_hex_profile():
    adapter = get_source_adapter("hex")
    assert adapter.source_id == "hex-global"
    assert adapter.runner == "mapped_composition_v1"
    assert adapter.packaged_profile == "hex-history-netcdf-v1"
    assert set(packaged_authorities("hex-history-netcdf-v1")) >= {"mapping", "composition", "provenance"}


def test_a_hex_export_frame_decodes_to_the_canonical_state(tmp_path, monkeypatch):
    from gpuwm.netcdf_bridge import find_netcdf_bin
    # The frame is read through the rw_netcdf bridge, as the door reads it; the Windows CPU partition of
    # public CI builds no natives (2.9.2 R7a route C: NetcdfBridgeMissing).
    if find_netcdf_bin() is None:
        pytest.skip("rw_netcdf is not built in this tree")
    monkeypatch.setenv("GPUWM_MAPPED_ENGINE", "python")
    frame = tmp_path / "wrfout_d01_2023-03-31_18_00_00"
    truth = _frame(frame)
    later = tmp_path / "wrfout_d01_2023-03-31_19_00_00"
    _frame(later, EPOCH_SECONDS + 3600)
    mapping = packaged_authorities("hex-history-netcdf-v1")["mapping"]
    decoded, following = ms.decode_mapped_source(mapping, [frame, later])
    assert following.valid_time.isoformat() == "2023-03-31T19:00:00"
    fields = decoded.fields
    # f32 storage of theta - 300 K and of the pressure is the only rounding.
    expected_t = truth["theta"] * (truth["pressure"] / 1.0e5) ** (2.0 / 7.0)
    np.testing.assert_allclose(fields["air_temperature"].values, expected_t, rtol=2e-6)
    np.testing.assert_allclose(fields["specific_humidity"].values,
                               truth["qv"] / (1.0 + truth["qv"]), rtol=1e-6)
    np.testing.assert_allclose(fields["specific_humidity_2m"].values,
                               truth["q2"] / (1.0 + truth["q2"]), rtol=1e-6)
    np.testing.assert_allclose(fields["geopotential_height"].values,
                               0.5 * (truth["zgrid"][:-1] + truth["zgrid"][1:]), rtol=1e-6)
    np.testing.assert_allclose(fields["sea_surface_temperature"].values, truth["tsk"], rtol=1e-7)
    assert fields["soil_temperature"].values.shape == (NSOIL, NY, NX)
    assert decoded.valid_time.isoformat() == "2023-03-31T18:00:00"
    for hidden in ("potential_temperature", "water_vapor_mixing_ratio", "interface_height"):
        assert hidden not in fields
