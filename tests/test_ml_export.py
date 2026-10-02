"""``gpuwm ml-export``: the front door, its tables and its contract.

The arithmetic is the Rust binary's and is tested there
(``tools/rustwx/crates/rw-mlexport``: analytic columns, the ECMWF
below-ground rules, the Blosc frames, the STORED ZIP, the append-against-run
byte identity, the refusals).  This file holds the Python half to what it
is: the tables are complete and consistent, the option grammar turns into
the request the binary speaks, the contract markers on both sides agree,
the binary ships in the bundle and is reported by ``gpuwm doctor``, and,
where a built binary is on this machine, a real history file goes through
the door and comes out as a dataset xarray opens.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pytest

from gpuwm import bridge_assets, bridges, cli, ml_export

REPO = Path(__file__).resolve().parents[1]
RUST_LIB = REPO / "tools" / "rustwx" / "crates" / "rw-mlexport" / "src" / "lib.rs"
RUST_OPS = REPO / "tools" / "rustwx" / "crates" / "rw-mlexport" / "src" / "ops.rs"


@pytest.fixture(scope="module")
def tables():
    return ml_export.load_tables()


def _args(**overrides):
    base = dict(inputs=["wrfout_d01_2026-09-29_00_00_00"], out=Path("out-x"),
                levels="wb13", variables=None, grid="native", regrid="bilinear",
                names="wb2", layout="analysis", domains=None, every=None,
                start=None, end=None, config=None, zip=False, overwrite=False,
                skip_unavailable=False, threads=None, append=False,
                finalize=False, list=False, json=False)
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------

def test_every_level_set_is_well_formed(tables):
    ids = tables.level_ids
    assert tables.levels["default"] in ids
    assert len(set(ids)) == len(ids)
    for row in tables.levels["sets"]:
        if row["kind"] == "pressure":
            hpa = row["hpa"]
            assert hpa and len(set(hpa)) == len(hpa)
            assert all(isinstance(p, int) and 1 <= p <= 1100 for p in hpa)
        else:
            assert row["kind"] == "model"
    by_id = {row["id"]: row for row in tables.levels["sets"]}
    assert by_id["wb13"]["hpa"] == [50, 100, 150, 200, 250, 300, 400, 500,
                                    600, 700, 850, 925, 1000]
    assert len(by_id["era5-37"]["hpa"]) == 37


def test_every_variable_row_is_complete_under_every_scheme(tables):
    schemes = tables.scheme_ids
    assert tables.names["default"] in schemes
    for scheme in schemes:
        names = [row["names"][scheme] for row in tables.variables]
        assert len(set(names)) == len(names), (
            f"two rows share a name under {scheme}; one would overwrite the other")
    for row in tables.variables:
        assert row["kind"] in ("level", "surface", "static"), row["id"]
        assert set(row["names"]) == set(schemes), row["id"]
        assert row["units"] and row["long_name"], row["id"]
        assert row["fields"], f"{row['id']} names no history field"
        if row["kind"] == "level":
            assert row["level_kinds"], row["id"]
            assert set(row["level_kinds"]) <= {"pressure", "model"}
            if "pressure" in row["level_kinds"]:
                assert row["below_ground"] in (
                    "ecmwf-temperature", "ecmwf-geopotential", "lowest-level"), row["id"]


def test_every_row_names_an_operator_the_binary_parses(tables):
    """The table is data; ops.rs is the only code.  Every operator a row
    names must be one of the catalogue's spellings, so a typo is caught
    here rather than as a refusal in someone's export."""

    source = RUST_OPS.read_text(encoding="utf-8")
    bare = set(re.findall(r'^\s+"([a-z0-9-]+)" => Op::', source, re.M))
    prefixed = set(re.findall(r'Some\(\("([a-z0-9-]+)", ', source))
    for row in tables.variables:
        op = row["op"]
        head, _, tail = op.partition(":")
        if tail:
            assert head in prefixed, f"{row['id']}: {op}"
        else:
            assert op in bare, f"{row['id']}: {op}"


def test_the_default_dataset_is_the_one_the_design_names(tables):
    pressure = [r["id"] for r in ml_export.select_variables(None, tables, "pressure")]
    assert pressure == [
        "geopotential", "temperature", "u_component_of_wind", "v_component_of_wind",
        "specific_humidity", "2m_temperature", "10m_u_component_of_wind",
        "10m_v_component_of_wind", "mean_sea_level_pressure", "surface_pressure",
        "total_precipitation", "total_precipitation_6hr", "total_column_water_vapour",
        "geopotential_at_surface", "land_sea_mask"]
    model = [r["id"] for r in ml_export.select_variables(None, tables, "model")]
    assert "pressure" in model


def test_the_options_document_is_the_tables(tables):
    doc = ml_export.options_document(tables)
    assert doc["schema"] == "ml-export.options/v1"
    assert [row["id"] for row in doc["levels"]] == tables.level_ids
    assert [row["id"] for row in doc["variables"]] == [r["id"] for r in tables.variables]
    assert doc["spacings_deg"] == tables.spacings
    json.dumps(doc)


# ---------------------------------------------------------------------------
# The option grammar
# ---------------------------------------------------------------------------

def test_level_grammar(tables):
    assert ml_export.parse_levels("wb13", tables)["hpa"][0] == 50
    spec = ml_export.parse_levels("model:1-3,7", tables)
    assert spec["kind"] == "model" and spec["model_levels"] == [1, 2, 3, 7]
    assert ml_export.parse_levels("model", tables)["model_levels"] == []
    assert ml_export.parse_levels("850,500", tables) == {
        "set": "custom", "kind": "pressure", "hpa": [850, 500], "model_levels": []}
    for bad in ("500,500", "0,500", "2000", "fifty", "wb13:3", "model:0", "model:5-2"):
        with pytest.raises(ml_export.MlExportRefusal):
            ml_export.parse_levels(bad, tables)


def test_variable_grammar(tables):
    plus = [r["id"] for r in ml_export.select_variables("+w,relative_humidity", tables, "pressure")]
    assert plus[-2:] == ["vertical_velocity", "relative_humidity"]
    exact = [r["id"] for r in ml_export.select_variables("t,z", tables, "pressure")]
    assert exact == ["temperature", "geopotential"]
    everything = ml_export.select_variables("all", tables, "pressure")
    assert "pressure" not in [r["id"] for r in everything]
    with pytest.raises(ml_export.MlExportRefusal, match="not a row"):
        ml_export.select_variables("vorticity", tables, "pressure")
    with pytest.raises(ml_export.MlExportRefusal, match="model"):
        ml_export.select_variables("pressure", tables, "pressure")


def test_grid_grammar():
    assert ml_export.parse_grid("native", "bilinear")["kind"] == "native"
    assert ml_export.parse_grid("latlon:0.25", "area-mean") == {
        "kind": "latlon", "deg": 0.25, "method": "area-mean"}
    assert ml_export.parse_grid("latlon", "bilinear")["deg"] is None
    for bad in ("latlon:x", "latlon:0", "latlon:20", "native:1", "gaussian"):
        with pytest.raises(ml_export.MlExportRefusal):
            ml_export.parse_grid(bad, "bilinear")


def test_the_request_carries_no_machine_path_in_its_provenance(tables, tmp_path):
    config = tmp_path / "experiment.toml"
    config.write_text("[experiment]\nname = 'x'\n", encoding="utf-8")
    request = ml_export.build_request(_args(config=str(config), zip=True), tables)
    assert request["schema"] == ml_export.REQUEST_SCHEMA == "ml-export.request/v1"
    assert request["mode"] == "run" and request["zip"] is True
    provenance = request["provenance"]
    assert len(provenance["config_sha256"]) == 64
    assert provenance["engine"] == ml_export.engine_name()
    assert provenance["history_attributes"] == tables.names["history_attributes"]
    assert provenance["history_engines"] == tables.names["history_engines"]
    assert provenance["history_engine_titles"] == tables.names["history_engine_titles"]
    assert "WOOF_VERSION" in provenance["history_attributes"]["version"]
    assert str(tmp_path) not in json.dumps(provenance)
    assert "experiment.toml" not in provenance["options"]
    assert request["levels"]["set"] == "wb13"
    assert request["spacings_deg"] == tables.spacings


def test_append_and_finalize_are_separate_calls(tables):
    assert ml_export.build_request(_args(append=True), tables)["mode"] == "append"
    assert ml_export.build_request(_args(inputs=[], finalize=True), tables)["mode"] == "finalize"
    with pytest.raises(ml_export.MlExportRefusal):
        ml_export.build_request(_args(finalize=True), tables)
    with pytest.raises(ml_export.MlExportRefusal):
        ml_export.build_request(_args(inputs=[]), tables)
    with pytest.raises(ml_export.MlExportRefusal, match="domain id"):
        ml_export.build_request(_args(domains="d1"), tables)


# ---------------------------------------------------------------------------
# The contract, the bundle and the doctor
# ---------------------------------------------------------------------------

def test_the_three_contract_markers_agree():
    rust = RUST_LIB.read_text(encoding="utf-8")
    match = re.search(r'pub const ABI: &str =\s*"([^"]+)";', rust)
    assert match, "rw_mlexport::ABI not found"
    assert match.group(1) == ml_export.ABI_MARKER
    assert bridges.BRIDGE_ABI_MARKERS["rw_mlexport"].decode() == ml_export.ABI_MARKER


def test_the_binary_ships_in_the_bundle_under_its_own_variable():
    row = next(a for a in bridge_assets.BUNDLED_ARTIFACTS if a.name == "rw_mlexport")
    assert row.crate == bridges.RUSTWX_CRATE_RELATIVE
    assert row.env_var == ml_export.BINARY_ENV == "GPUWM_RW_MLEXPORT"


def test_the_command_is_registered():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert "ml-export" in sub.choices
    args = parser.parse_args(["ml-export", "a", "b", "--out", "x", "--grid", "latlon:0.25"])
    assert args.inputs == ["a", "b"] and args.grid == "latlon:0.25"


def test_list_json_is_the_options_document(capsys):
    assert cli.main(["ml-export", "--list", "--json"]) == 0
    out = capsys.readouterr().out
    doc = json.loads(out)
    assert doc["schema"] == "ml-export.options/v1"


def test_a_missing_binary_names_what_supplies_it(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ml_export.Binary, "find", lambda self: None)
    code = ml_export.main(_args(out=tmp_path / "x"))
    assert code == ml_export.EXIT_NOT_STAGED
    said = capsys.readouterr().err
    assert "rw_mlexport" in said and "GPUWM_RW_MLEXPORT" in said


def test_progress_events_read_as_one_line_each():
    line = ml_export.describe_event({"event": "frame", "frame": 2, "of": 9, "domain": "d02",
                                     "valid": "2026-09-29T01:00:00", "seconds": 1.25,
                                     "read_seconds": 0.5, "bytes": 2_000_000})
    assert line.startswith("frame 2/9  d02") and "2.0 MB" in line
    assert ml_export.describe_event({"event": "unknown"}) is None


# ---------------------------------------------------------------------------
# Through the door, where a built binary is on this machine
# ---------------------------------------------------------------------------

def _tiny_wrfout(path: Path, hour: int) -> None:
    netCDF4 = pytest.importorskip("netCDF4")
    import numpy as np

    nx, ny, nz = 6, 5, 4
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        for name, size in (("Time", None), ("DateStrLen", 19), ("west_east", nx),
                           ("south_north", ny), ("bottom_top", nz), ("west_east_stag", nx + 1),
                           ("south_north_stag", ny + 1), ("bottom_top_stag", nz + 1)):
            ds.createDimension(name, size)
        for key, value in (("MAP_PROJ", 6), ("GRID_ID", 1), ("PARENT_ID", 0), ("DX", 1e5),
                           ("DY", 1e5), ("POLE_LAT", 90.0), ("POLE_LON", 0.0),
                           ("SIMULATION_START_DATE", "2026-09-29_00:00:00")):
            ds.setncattr(key, value)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = np.array(list(f"2026-09-29_{hour:02d}:00:00"), dtype="S1")
        lat = np.repeat(np.linspace(30, 34, ny)[:, None], nx, axis=1)
        lon = np.repeat(np.linspace(-100, -95, nx)[None, :], ny, axis=0)
        ps = np.full((ny, nx), 100_000.0)
        face = np.array([100_000.0 - k * 20_000.0 for k in range(nz + 1)])
        mass = np.sqrt(face[:-1] * face[1:])
        plane = ("Time", "south_north", "west_east")
        volume = ("Time", "bottom_top", "south_north", "west_east")
        for name, dims, value in (
                ("XLAT", plane, lat), ("XLONG", plane, lon), ("PSFC", plane, ps),
                ("T2", plane, np.full((ny, nx), 290.0)), ("HGT", plane, np.zeros((ny, nx))),
                ("LANDMASK", plane, np.ones((ny, nx))), ("SINALPHA", plane, np.zeros((ny, nx))),
                ("COSALPHA", plane, np.ones((ny, nx))), ("RAINNC", plane, np.full((ny, nx), 1.0 * hour)),
                ("P", volume, np.zeros((nz, ny, nx))),
                ("PB", volume, np.broadcast_to(mass[:, None, None], (nz, ny, nx))),
                ("T", volume, np.zeros((nz, ny, nx))),
                ("QVAPOR", volume, np.full((nz, ny, nx), 0.001))):
            var = ds.createVariable(name, "f4", dims)
            var[0] = np.ascontiguousarray(value)
        u = ds.createVariable("U", "f4", ("Time", "bottom_top", "south_north", "west_east_stag"))
        u[0] = np.full((nz, ny, nx + 1), 5.0)
        v = ds.createVariable("V", "f4", ("Time", "bottom_top", "south_north_stag", "west_east"))
        v[0] = np.full((nz, ny + 1, nx), -3.0)
        for name in ("PH", "PHB"):
            var = ds.createVariable(name, "f4", ("Time", "bottom_top_stag", "south_north", "west_east"))
            var[0] = np.zeros((nz + 1, ny, nx)) if name == "PH" else np.ascontiguousarray(
                np.broadcast_to((287.0 * 250.0 * np.log(1e5 / face))[:, None, None], (nz + 1, ny, nx)))


def test_a_history_file_goes_through_the_door_and_opens_in_xarray(tmp_path, monkeypatch):
    found = ml_export.BINARY.find()
    if found is None or not ml_export.BINARY.probe(found)[0]:
        pytest.skip("rw_mlexport is not built on this machine")
    for hour in (0, 1):
        _tiny_wrfout(tmp_path / f"wrfout_d01_2026-09-29_{hour:02d}_00_00", hour)
    out = tmp_path / "export"
    code = cli.main(["ml-export", str(tmp_path), "--out", str(out), "--levels", "850,500",
                     "--variables", "temperature,u_component_of_wind,2m_temperature,total_precipitation",
                     "--zip"])
    assert code == 0
    assert (out / "d01.zarr" / ".zmetadata").is_file()
    assert (tmp_path / "export-ml.zip").is_file()
    receipt = json.loads((out / "ml-export-receipt.json").read_text(encoding="utf-8"))
    assert [f["input"] for f in receipt["domains"][0]["frames"]] == [
        "wrfout_d01_2026-09-29_00_00_00", "wrfout_d01_2026-09-29_01_00_00"]
    assert str(tmp_path) not in (out / "d01.zarr" / ".zattrs").read_text(encoding="utf-8")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    ds = xr.open_zarr(out / "d01.zarr")
    assert list(ds.level.values) == [850, 500]
    assert float(ds.total_precipitation.isel(time=1).mean()) == pytest.approx(0.001)
    assert float(ds.u_component_of_wind.isel(time=0).sel(level=500).mean()) == pytest.approx(5.0)
    assert ds.temperature.attrs["units"] == "K"


def test_levels_match_geocat_above_and_below_ground(tmp_path):
    """An independent implementation of the same column arithmetic.

    GeoCAT's ``interp_hybrid_to_pressure(method="log", extrapolate=True)``
    puts each column on the same levels from the same inputs (its hybrid
    coefficients are the column's own pressures); 1000 hPa sits under the
    lowest model level here, so the ECMWF below-ground rules are compared as
    well as the interpolation.  Agreement is float32 round-off.
    """

    found = ml_export.BINARY.find()
    if found is None or not ml_export.BINARY.probe(found)[0]:
        pytest.skip("rw_mlexport is not built on this machine")
    np = pytest.importorskip("numpy")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    geocat = pytest.importorskip("geocat.comp")
    netCDF4 = pytest.importorskip("netCDF4")
    path = tmp_path / "wrfout_d01_2026-09-29_00_00_00"
    _tiny_wrfout(path, 0)
    out = tmp_path / "export"
    assert cli.main(["ml-export", str(path), "--out", str(out), "--levels", "1000,850,500",
                     "--variables", "temperature,geopotential,u_component_of_wind"]) == 0
    ds = xr.open_zarr(out / "d01.zarr").isel(time=0)
    with netCDF4.Dataset(path) as d:
        p = (d["P"][0] + d["PB"][0]).astype(np.float64)
        t = (d["T"][0].astype(np.float64) + 300.0) * (p / 1e5) ** 0.2857142857
        faces = (d["PH"][0] + d["PHB"][0]).astype(np.float64)
        u = d["U"][0].astype(np.float64)
        psfc = d["PSFC"][0].astype(np.float64)
    phi = 0.5 * (faces[:-1] + faces[1:])
    u = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    levels = np.array([100000.0, 85000.0, 50000.0])
    g = 9.80616
    for j, i in ((0, 0), (2, 3), (4, 5)):
        coords = {"lev": p[:, j, i]}
        hyam = xr.DataArray(p[:, j, i] / 1e5, dims=["lev"], coords=coords)
        hybm = xr.DataArray(np.zeros(p.shape[0]), dims=["lev"], coords=coords)
        common = dict(p0=1e5, new_levels=levels, lev_dim="lev", method="log", extrapolate=True,
                      t_bot=xr.DataArray(t[0, j, i]), phi_sfc=xr.DataArray(faces[0, j, i]))
        for name, field, kind, scale in (("temperature", t, "temperature", 1.0),
                                         ("geopotential", phi, "geopotential", g),
                                         ("u_component_of_wind", u, "other", 1.0)):
            data = xr.DataArray(field[:, j, i] / scale, dims=["lev"], coords=coords)
            expected = geocat.interp_hybrid_to_pressure(
                data, xr.DataArray(psfc[j, i]), hyam, hybm, variable=kind, **common).values * scale
            got = ds[name].values[:, j, i].astype(np.float64)
            assert np.allclose(got, expected, rtol=2e-6, atol=1e-3), (name, got, expected)
