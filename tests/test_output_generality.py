"""Phase-5 Task-5 G6/G7: config-only static and generic WRF output."""

from __future__ import annotations

import os

from datetime import datetime, timedelta
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from gpuwm import runtime
from gpuwm.case_data import load_experiment_case
from gpuwm.io.wrfout import (WrfoutWriter, wrf_global_attrs,
                             wrfout_filename)
from gpuwm.static.build import GeogSelection
from gpuwm.static.lambert import LambertGrid

BUNDLE = Path(os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                    "gpuwm-fixture-unset/wrf74-bundle"))
GEOG_ROOT = BUNDLE / "static" / "WPS_GEOG"
requires_bundle = pytest.mark.skipif(
    not GEOG_ROOT.is_dir(), reason="reference WPS_GEOG tree not present")


def _oklahoma_case(tmp_path: Path) -> Path:
    """A small loadable real-case declaration; WPS carries GEOG tokens only."""
    for name in ("forcing.grb", "Vtable", "source_orography.nc"):
        (tmp_path / name).write_bytes(b"fixture")
    (tmp_path / "namelist.wps").write_text(
        "&share\n max_dom = 1,\n/\n"
        "&geogrid\n geog_data_res = 'default',\n/\n",
        encoding="utf-8")
    geog = GEOG_ROOT.as_posix()
    path = tmp_path / "oklahoma.toml"
    path.write_text(f"""
[experiment]
name = "oklahoma_config_fixture"
start_time = 1999-05-03T12:00:00
run_seconds = 3600.0
restart_interval_s = 0.0

[projection]
map_proj = "lambert"
ref_lat = 35.4676
ref_lon = -97.5164
truelat1 = 30.0
truelat2 = 60.0
stand_lon = -97.0

[shared]
nz = 4
ztop = 12000.0
p_top = 10000.0
eta_levels = [1.0, 0.8, 0.55, 0.25, 0.0]
map_proj = 1

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 14
ny = 12
time_step = 60
dx = 12000.0
history_interval_s = 1800.0

[case_data]
forcing = "forcing.grb"
vtable = "Vtable"
wps_namelist = "namelist.wps"
geog_root = "{geog}"
source_orography = "source_orography.nc"
source_orography_variable = "HGT"
sfcp_to_sfcp = true
co2_vmr = 370.0e-6
forcing_interval_s = 21600.0
output_domain = 1
output_title = "Oklahoma config fixture"
""", encoding="utf-8")
    return path


def _synthetic_wrfout(path: Path, grid: LambertGrid,
                      start_time: datetime) -> Path:
    nz = 4
    ny, nx = grid.e_sn - 1, grid.e_we - 1
    lat, lon = grid.latlon_mass()
    pressure = np.array([90000.0, 75000.0, 60000.0, 45000.0],
                        dtype=np.float32)
    heights = np.array([0.0, 1000.0, 2500.0, 4500.0, 7000.0],
                       dtype=np.float32)
    pb = np.broadcast_to(
        pressure[:, None, None], (nz, ny, nx)).copy()
    phb = np.broadcast_to(
        (np.float32(9.81) * heights)[:, None, None],
        (nz + 1, ny, nx)).copy()
    fields = {
        "T": np.zeros((nz, ny, nx), dtype=np.float32),
        "P": np.zeros((nz, ny, nx), dtype=np.float32),
        "PB": pb,
        "QVAPOR": np.full((nz, ny, nx), 0.005, dtype=np.float32),
        "PH": np.zeros((nz + 1, ny, nx), dtype=np.float32),
        "PHB": phb,
        "HGT": np.zeros((ny, nx), dtype=np.float32),
        "T2": np.full((ny, nx), 290.0, dtype=np.float32),
        "XLAT": lat,
        "XLONG": lon,
    }
    with WrfoutWriter(
            path, nx=nx, ny=ny, nz=nz, dx=grid.dx, dy=grid.dy,
            title="Synthetic generic tooling smoke",
            global_attrs=wrf_global_attrs(grid, start_time)) as writer:
        writer.write_frame(start_time.strftime("%Y-%m-%d_%H:%M:%S"),
                           fields)
    return path


def test_non_reference_date_filename_is_colon_free_and_second_complete():
    valid = datetime(2001, 6, 7, 8, 9, 10)
    name = wrfout_filename(valid, domain_id=3)
    assert name == "wrfout_d03_2001-06-07_08_09_10"
    assert ":" not in name
    with pytest.raises(ValueError, match="positive integer"):
        wrfout_filename(valid, domain_id=0)


@requires_bundle
def test_oklahoma_toml_builds_static_and_wrf_metadata(tmp_path):
    """G6: recentered 12-km grid, static fields, and identity from TOML."""
    exp, data = load_experiment_case(_oklahoma_case(tmp_path))
    grid = runtime.experiment_grid(exp, data)
    assert (grid.ref_lat, grid.ref_lon, grid.dx) == (
        35.4676, -97.5164, 12000.0)
    center = grid.ij_to_latlon(grid.e_we / 2.0, grid.e_sn / 2.0)
    np.testing.assert_allclose(center, (35.4676, -97.5164),
                               rtol=0, atol=1e-12)

    x = np.array([1.0, grid.e_we - 1.0, 1.0, grid.e_we - 1.0])
    y = np.array([1.0, 1.0, grid.e_sn - 1.0, grid.e_sn - 1.0])
    lat, lon = grid.ij_to_latlon(x, y)
    xr, yr = grid.latlon_to_ij(lat, lon)
    np.testing.assert_allclose(xr, x, rtol=0, atol=1e-9)
    np.testing.assert_allclose(yr, y, rtol=0, atol=1e-9)
    mapfac = grid.mapfac_m()
    assert np.isfinite(mapfac).all() and np.all(mapfac > 0.0)

    static_path = runtime.write_static(
        exp, data, tmp_path / "oklahoma_static.npz")
    with np.load(static_path) as static:
        for name in ("HGT_M", "LANDMASK", "LU_INDEX"):
            assert static[name].shape[-2:] == (12, 14)
            assert np.isfinite(static[name]).all()
    assert mapfac.shape == (12, 14)

    selection = GeogSelection.from_case_data(data, domain_id=1)
    valid = exp.start_time + timedelta(minutes=30)
    output = tmp_path / wrfout_filename(valid, data.output_domain)
    attrs = wrf_global_attrs(
        grid, exp.start_time,
        landuse_attrs=selection.landuse_global_attrs())
    with WrfoutWriter(
            output, nx=14, ny=12, nz=4, dx=12000.0, dy=12000.0,
            title=data.output_title, global_attrs=attrs) as writer:
        writer.write_frame(valid.strftime("%Y-%m-%d_%H:%M:%S"), {
            "XLAT": grid.latlon_mass()[0],
            "XLONG": grid.latlon_mass()[1],
            "T2": np.full((12, 14), 290.0, dtype=np.float32),
        })
    assert output.name == "wrfout_d01_1999-05-03_12_30_00"
    with netCDF4.Dataset(output) as ds:
        assert ds.TITLE == "Oklahoma config fixture"
        assert int(ds.MAP_PROJ) == 1
        assert ds.MAP_PROJ_CHAR == "Lambert Conformal"
        assert (float(ds.POLE_LAT), float(ds.POLE_LON)) == (90.0, 0.0)
        assert ds.MMINLU == "MODIFIED_IGBP_MODIS_NOAH"
        assert (int(ds.ISWATER), int(ds.ISLAKE), int(ds.ISICE),
                int(ds.ISURBAN)) == (17, 21, 15, 13)
        assert ds.START_DATE == "1999-05-03_12:00:00"
        assert ds.SIMULATION_START_DATE == ds.START_DATE


def test_wrf_getvar_slp_t2_smoke_is_finite_and_georeferenced(tmp_path):
    """G7: standard WRF tooling reads synthetic diagnostics and XLAT/LONG."""
    wrf = pytest.importorskip("wrf")
    start = datetime(2001, 6, 7, 8, 9, 10)
    grid = LambertGrid(
        ref_lat=35.4676, ref_lon=-97.5164, truelat1=30.0,
        truelat2=60.0, stand_lon=-97.0, dx=12000.0, dy=12000.0,
        e_we=9, e_sn=7)
    path = _synthetic_wrfout(
        tmp_path / wrfout_filename(start, domain_id=1), grid, start)

    pathname_backend = hasattr(wrf, "WrfFile")
    source = (wrf.WrfFile(str(path)) if pathname_backend
              else netCDF4.Dataset(path))
    try:
        slp = wrf.getvar(source, "slp", meta=True)
        t2 = wrf.getvar(source, "T2", meta=True)
        for name, value in (("slp", slp), ("T2", t2)):
            array = np.asarray(value)
            assert array.shape == (6, 8), name
            assert np.isfinite(array).all(), name

        if hasattr(slp, "coords"):
            lat, lon = wrf.latlon_coords(slp)
        else:
            lat = wrf.getvar(source, "XLAT", meta=False)
            lon = wrf.getvar(source, "XLONG", meta=False)
        expected_lat, expected_lon = grid.latlon_mass()
        np.testing.assert_allclose(np.asarray(lat), expected_lat,
                                   rtol=0, atol=5e-6)
        np.testing.assert_allclose(np.asarray(lon), expected_lon,
                                   rtol=0, atol=5e-6)
    finally:
        if not pathname_backend:
            source.close()
