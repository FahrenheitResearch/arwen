"""A lat-lon grid (WPS map_proj 'lat-lon', WRF MAP_PROJ 6) is refused BY
NAME on every import door of this release.

This release has no lat-lon grid.  Before this refusal the doors met one
three ways, none of them naming it: the importer asked for a missing
``truelat1`` or called the degree dx/dy a "mismatch" with the metre
spacing in namelist.input; the support report added a second refusal that
read 0.108 degrees as 0.108 m; and ``projection_class`` raised a bare
NotImplementedError past every door that prints ValueError as a sentence
(``target_from_wps`` among them).  Each now gives the one sentence
:func:`gpuwm.static.projection.latlon_refusal` publishes.

CPU only.  The two file doors are exercised on real NetCDF headers
(MAP_PROJ = 6) through their own entry points.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gpuwm.namelist_compat import analyze_namelists
from gpuwm.namelist_import import import_namelists
from gpuwm.static import projection as projection_module
from gpuwm.static.projection import (LatLonProjectionRefusal,
                                     grids_from_wps_namelist, latlon_refusal)

from conftest import requires_netcdf_bridge
from test_namelist_compat import _write_pair

NAMED = "map_proj lat-lon (WRF 6) is not supported in this release"
BREAKAGE = "conformal map factors"


def _latlon_pair(tmp_path, *, max_dom=1, global_grid=False):
    """A namelist pair as WPS writes a regional rotated lat-lon grid:
    dx/dy in degrees, no true latitudes, the rotated pole."""
    wps, inp = _write_pair(tmp_path, max_dom=max_dom, mp=6)
    text = (wps.read_text(encoding="utf-8")
            .replace("map_proj = 'lambert'", "map_proj = 'lat-lon'")
            .replace(" truelat1 = 30.0,\n", "")
            .replace(" truelat2 = 60.0,\n", "")
            .replace(" stand_lon = -97.0,\n",
                     " stand_lon = -10.0,\n pole_lat = 40.0,\n"
                     " pole_lon = 180.0,\n"))
    if global_grid:
        text = text.replace(" dx = 12000,\n", "").replace(" dy = 12000,\n", "")
    else:
        text = (text.replace("dx = 12000,", "dx = 0.108,")
                .replace("dy = 12000,", "dy = 0.108,"))
    wps.write_text(text, encoding="utf-8")
    return wps, inp


def test_the_sentence_names_what_and_why():
    sentence = latlon_refusal()
    assert NAMED in sentence and BREAKAGE in sentence


@pytest.mark.parametrize("max_dom,global_grid",
                         [(1, False), (2, False), (1, True)])
def test_support_report_refuses_latlon_by_name_and_only_by_name(
        tmp_path, max_dom, global_grid):
    report = analyze_namelists(*_latlon_pair(
        tmp_path, max_dom=max_dom, global_grid=global_grid))
    assert report["verdict"] == "FAIL"
    blocking = [issue for issue in report["issues"]
                if issue["severity"] == "blocking"]
    assert [issue["code"] for issue in blocking] == ["UNSUPPORTED_PROJECTION"]
    assert latlon_refusal() in blocking[0]["message"]
    # The degree spacing is no longer read as metres.
    assert not any(" m from its parent ratio" in issue["message"]
                   for issue in report["issues"])


@pytest.mark.parametrize("max_dom,global_grid",
                         [(1, False), (2, False), (1, True)])
def test_import_refuses_latlon_by_name_first(tmp_path, max_dom, global_grid):
    wps, inp = _latlon_pair(tmp_path, max_dom=max_dom,
                            global_grid=global_grid)
    with pytest.raises(ValueError) as refused:
        import_namelists(wps, inp)
    message = str(refused.value)
    assert latlon_refusal() in message
    assert "missing required key" not in message
    assert "mismatch" not in message


def test_import_namelist_cli_prints_the_refusal(tmp_path, capsys):
    import gpuwm.cli as cli
    wps, inp = _latlon_pair(tmp_path)
    assert cli.main(["import-namelist", str(wps), str(inp)]) != 0
    err = capsys.readouterr().err
    assert latlon_refusal() in err and "Traceback" not in err


def test_projection_class_refusal_is_a_sentence_on_value_error_doors(
        tmp_path):
    wps, _ = _latlon_pair(tmp_path)
    with pytest.raises(ValueError) as refused:
        grids_from_wps_namelist(wps)
    assert isinstance(refused.value, NotImplementedError)
    assert isinstance(refused.value, LatLonProjectionRefusal)
    assert latlon_refusal() in str(refused.value)
    assert projection_module.latlon_blocker() in str(refused.value)

    from gpuwm.source_normalization import target_from_wps
    with pytest.raises(ValueError, match="WRF 6"):
        target_from_wps(None, wps)


def _header(path: Path, *, kind: str) -> None:
    """A real NetCDF header on a MAP_PROJ = 6 grid: dimensions and the
    global attributes real.exe / metgrid.exe stamp, no fields."""
    netCDF4 = pytest.importorskip("netCDF4")
    # The extent _write_pair's namelist.input declares (e_we 121, e_sn
    # 101, 50 levels), so the header agrees with its namelist on
    # everything but the projection.
    nx, ny, nz = 120, 100, 50
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("west_east", nx)
        dataset.createDimension("west_east_stag", nx + 1)
        dataset.createDimension("south_north", ny)
        dataset.createDimension("south_north_stag", ny + 1)
        if kind == "wrfinput":
            dataset.createDimension("bottom_top", nz)
            dataset.createDimension("bottom_top_stag", nz + 1)
            dataset.createDimension("soil_layers_stag", 4)
        else:
            dataset.createDimension("num_metgrid_levels", 34)
        attributes = {
            "TITLE": " OUTPUT FROM REAL_EM V4.6.0 PREPROCESSOR",
            "START_DATE": "2020-05-01_00:00:00",
            "MAP_PROJ": 6, "MAP_PROJ_CHAR": "Cylindrical Equidistant",
            "DX": 12000.0, "DY": 12000.0,
            "CEN_LAT": 50.0, "CEN_LON": 10.0, "TRUELAT1": 0.0,
            "TRUELAT2": 0.0, "STAND_LON": -10.0, "MOAD_CEN_LAT": 50.0,
            "POLE_LAT": 40.0, "POLE_LON": 180.0,
            "GRID_ID": 1, "PARENT_ID": 0, "I_PARENT_START": 1,
            "J_PARENT_START": 1, "PARENT_GRID_RATIO": 1,
            "MP_PHYSICS": 8, "SF_SURFACE_PHYSICS": 2,
            "USE_THETA_M": 0, "HYBRID_OPT": 2, "ETAC": 0.2,
            "MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "NUM_LAND_CAT": 21,
        }
        for name, value in attributes.items():
            dataset.setncattr(name, value)
        dataset.createDimension("DateStrLen", 19)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = list(b"2020-05-01_00:00:00")


@requires_netcdf_bridge
def test_wrfinput_door_refuses_a_map_proj_6_header_by_name(tmp_path, capsys):
    from gpuwm.wrfinput_door import resolve_wrfinput_run
    from gpuwm.wrfinput_forecast import main as wrfinput_main

    _, inp = _write_pair(tmp_path, max_dom=1, mp=8)
    _header(tmp_path / "wrfinput_d01", kind="wrfinput")
    (tmp_path / "wrfbdy_d01").write_bytes(b"")
    with pytest.raises(ValueError) as refused:
        resolve_wrfinput_run(tmp_path)
    assert "MAP_PROJ=6" in str(refused.value)
    assert latlon_refusal() in str(refused.value)

    assert wrfinput_main(["--wrfinput", str(tmp_path),
                          "--outdir", str(tmp_path / "out")]) == 2
    assert latlon_refusal() in capsys.readouterr().err


@requires_netcdf_bridge
def test_met_em_door_refuses_a_map_proj_6_header_by_name(tmp_path, capsys):
    from gpuwm.metem_door import resolve_metem_run
    from gpuwm.metem_forecast import main as metem_main

    _write_pair(tmp_path, max_dom=1, mp=8)
    for hour in ("00", "12"):
        # The Windows-safe spelling the reader accepts (no colons).
        _header(tmp_path / f"met_em.d01.2020-05-01_{hour}_00_00.nc",
                kind="met_em")
    with pytest.raises(ValueError) as refused:
        resolve_metem_run(tmp_path)
    assert "MAP_PROJ=6" in str(refused.value)
    assert latlon_refusal() in str(refused.value)

    assert metem_main(["--met-em", str(tmp_path),
                       "--outdir", str(tmp_path / "out")]) == 2
    assert latlon_refusal() in capsys.readouterr().err


def test_met_em_reader_names_map_proj_6():
    from gpuwm.ingest.metem import MetgridRefusal, _check_projection
    with pytest.raises(MetgridRefusal, match="MAP_PROJ=6: map_proj lat-lon"):
        _check_projection({"map_proj": 6}, {})
