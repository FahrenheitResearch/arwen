"""The dual-write verification that justifies the writer flip, on fixtures.

One identical stream of frames is written through BOTH engines --
``engine="python"`` (netCDF4/HDF5, the 2.4 writer) and ``engine="rust"``
(the classic writer behind ``gpuwm.io.nc_writer_bridge``, the 2.5.0
default) -- and the two files must be semantically identical: same
structure, same attribute inventory with bitwise-equal values, same
variables in the same order, bitwise-equal payloads.  Where the product
renderer is available, ``rw_wrfbatch`` renders of the two files must be
byte-identical, product for product.

Three fixture cases span the production surface the flip touches:

* ``gfs_small`` -- a small-GFS-go-shaped d01: the full real-case frame
  (winds, mass, moisture, diagnostics, soil, surface, reflectivity),
  WRF projection/topology globals, physics selectors, XTIME/ITIMESTEP
  from START_DATE+DT, initial-condition provenance and carrier
  provenance globals;
* ``twentycrv3_demo`` -- the 20CRv3 demo-case shape: 30 km d01,
  provenance naming the reanalysis source, surface/radiation/snow
  fields;
* ``nested_d02`` -- a nested domain: parent topology globals, Noah-MP
  snow-stack axes (``snow_layers_stag``/``snso_layers_stag``), an
  integer scheme field, and a Z-staggered MYNN carrier the writer lifts.

The FULL-SIZE version of this verification -- real wrfouts from real
runs, replayed through both engines and rendered -- is
``tools/wrfout_dual_write.py``; its receipts ride the flip's evidence.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import netCDF4
import pytest

from gpuwm.io import nc_writer_bridge
from gpuwm.io.wrfout import (
    WrfoutWriter, initial_condition_global_attrs, wrf_global_attrs,
)
from gpuwm.io.wrf_output_schema import SCHEME_OUTPUT_FIELDS
from tools.wrfout_dual_write import compare_renders, semantic_diff

_RUST_UNAVAILABLE = nc_writer_bridge.unavailable_reason()

pytestmark = pytest.mark.skipif(
    _RUST_UNAVAILABLE is not None,
    reason=f"the Rust NetCDF writer library is not built: {_RUST_UNAVAILABLE}")

_NZ, _NY, _NX = 6, 24, 32
_START = datetime(1974, 4, 3, 12, 0, 0)


def _grid():
    return SimpleNamespace(
        wrf_map_proj=1, map_proj_char="Lambert Conformal",
        truelat1=30.0, truelat2=60.0, stand_lon=-96.5,
        ref_lat=39.0, ref_lon=-96.5, moad_cen_lat=39.0,
        cen_lat=39.0, cen_lon=-96.5)


def _latlon():
    lat = np.tile(np.linspace(38.0, 40.0, _NY)[:, None], (1, _NX))
    lon = np.tile(np.linspace(-98.0, -95.0, _NX)[None, :], (_NY, 1))
    return lat.astype(np.float32), lon.astype(np.float32)


def _base_frame(seed):
    """A render-viable real-case frame: 3-D state + surface diagnostics."""
    rng = np.random.default_rng(seed)
    lat, lon = _latlon()
    frame = {
        "T": rng.uniform(-5.0, 5.0, (_NZ, _NY, _NX)).astype(np.float32),
        "U": rng.uniform(-20.0, 20.0, (_NZ, _NY, _NX + 1)).astype(np.float32),
        "V": rng.uniform(-20.0, 20.0, (_NZ, _NY + 1, _NX)).astype(np.float32),
        "W": rng.uniform(-2.0, 2.0, (_NZ + 1, _NY, _NX)).astype(np.float32),
        "PH": rng.uniform(-50.0, 50.0, (_NZ + 1, _NY, _NX)).astype(np.float32),
        "PHB": np.cumsum(rng.uniform(200.0, 300.0, (_NZ + 1, _NY, _NX)),
                         axis=0).astype(np.float32),
        "MU": rng.uniform(-100.0, 100.0, (_NY, _NX)).astype(np.float32),
        "MUB": np.full((_NY, _NX), 97000.0, np.float32),
        "HGT": rng.uniform(200.0, 400.0, (_NY, _NX)).astype(np.float32),
        "QVAPOR": rng.uniform(0.001, 0.014,
                              (_NZ, _NY, _NX)).astype(np.float32),
        "T2": rng.uniform(280.0, 300.0, (_NY, _NX)).astype(np.float32),
        "Q2": rng.uniform(0.004, 0.012, (_NY, _NX)).astype(np.float32),
        "PSFC": rng.uniform(96000.0, 98000.0, (_NY, _NX)).astype(np.float32),
        "U10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "V10": rng.uniform(-10.0, 10.0, (_NY, _NX)).astype(np.float32),
        "RAINC": rng.uniform(0.0, 5.0, (_NY, _NX)).astype(np.float32),
        "RAINNC": rng.uniform(0.0, 30.0, (_NY, _NX)).astype(np.float32),
        "XLAT": lat, "XLONG": lon,
        "SINALPHA": np.zeros((_NY, _NX), np.float32),
        "COSALPHA": np.ones((_NY, _NX), np.float32),
        "P_TOP": np.asarray(10000.0, dtype=np.float32),
        "ZNU": np.linspace(0.9, 0.1, _NZ).astype(np.float32),
        "ZNW": np.linspace(1.0, 0.0, _NZ + 1).astype(np.float32),
    }
    # The values a tolerance would forgive and a bitwise compare must not.
    frame["T"][0, 0, 0] = np.nan
    frame["T2"][1, 1] = np.float32(-0.0)
    return frame


def _provenance(lead: int):
    return {
        "initial_forecast_lead_hours": lead,
        "initial_condition_kind": "analysis" if lead == 0 else "forecast",
        "forecast_generating_process_id": 81 if lead == 0 else 96,
        "cycle": "1974-04-03T12:00:00Z",
        "model_start_time": (
            "1974-04-03T12:00:00Z" if lead == 0 else "1974-04-03T18:00:00Z"),
        "statement": "fixture provenance for the dual-write verification",
    }


def _case_gfs_small():
    """The small-GFS-go shape: full frame, full attr surface, 2 frames."""
    attrs = wrf_global_attrs(
        _grid(), _START, grid_id=1, parent_id=0, i_parent_start=1,
        j_parent_start=1, parent_grid_ratio=1, dt=30.0,
        hybrid_opt=2, etac=0.2)
    attrs.update(initial_condition_global_attrs(
        _provenance(0), source="gfs"))
    # Carrier provenance, as the runtime stamps it per frame.
    attrs.update({
        "GPUWM_SURFACE_RADIATION_POLICY": "prescribed",
        "GPUWM_CARRIER_GLW_SOURCE": "declared-constant",
        "GPUWM_CARRIER_GLW_LAST_UPDATE": np.float64(-1.0),
    })
    frames = []
    for index, seed in enumerate((7, 11)):
        frame = _base_frame(seed)
        rng = np.random.default_rng(100 + seed)
        frame["QCLOUD"] = rng.uniform(
            0.0, 1e-4, (_NZ, _NY, _NX)).astype(np.float32)
        frame["QRAIN"] = rng.uniform(
            0.0, 1e-4, (_NZ, _NY, _NX)).astype(np.float32)
        frame["REFL_10CM"] = rng.uniform(
            -20.0, 65.0, (_NZ, _NY, _NX)).astype(np.float32)
        frame["TSLB"] = rng.uniform(
            270.0, 300.0, (4, _NY, _NX)).astype(np.float32)
        frame["SMOIS"] = rng.uniform(
            0.1, 0.4, (4, _NY, _NX)).astype(np.float32)
        frame["SH2O"] = rng.uniform(
            0.1, 0.4, (4, _NY, _NX)).astype(np.float32)
        frame["ISLTYP"] = np.full((_NY, _NX), 6, np.int32)
        frame["IVGTYP"] = np.full((_NY, _NX), 10, np.int32)
        frames.append(frame)
    stamps = ("1974-04-03_12:00:00", "1974-04-03_13:00:00")
    return dict(grid_id=1, dx=12000.0, attrs=attrs, stamps=stamps,
                frames=frames, soil_layers=4)


def _case_twentycrv3_demo():
    """The 20CRv3 demo shape: 30 km d01 with reanalysis provenance."""
    attrs = wrf_global_attrs(
        _grid(), _START, grid_id=1, parent_id=0, i_parent_start=1,
        j_parent_start=1, parent_grid_ratio=1, dt=90.0)
    attrs.update(initial_condition_global_attrs(
        _provenance(0), source="20crv3-cf"))
    frames = []
    for seed in (19, 23):
        frame = _base_frame(seed)
        rng = np.random.default_rng(300 + seed)
        frame["TSK"] = rng.uniform(
            278.0, 302.0, (_NY, _NX)).astype(np.float32)
        frame["SWDOWN"] = rng.uniform(
            0.0, 900.0, (_NY, _NX)).astype(np.float32)
        frame["GLW"] = rng.uniform(
            250.0, 420.0, (_NY, _NX)).astype(np.float32)
        frame["OLR"] = rng.uniform(
            180.0, 300.0, (_NY, _NX)).astype(np.float32)
        frame["SNOW"] = rng.uniform(0.0, 40.0, (_NY, _NX)).astype(np.float32)
        frame["SNOWH"] = rng.uniform(0.0, 0.4, (_NY, _NX)).astype(np.float32)
        frame["SNOWC"] = (frame["SNOW"] > 20.0).astype(np.float32)
        frames.append(frame)
    stamps = ("1974-04-03_12:00:00", "1974-04-03_15:00:00")
    return dict(grid_id=1, dx=30000.0, attrs=attrs, stamps=stamps,
                frames=frames, soil_layers=4)


def _case_nested_d02():
    """A nested domain: topology globals plus the scheme axes.

    Noah-MP's snow stack brings the two axes only a nest with a live
    land-surface scheme exercises (``snow_layers_stag``,
    ``snso_layers_stag``), ``ISNOW`` is integer on disk, and ``EL_PBL``
    arrives on the mass column and must be LIFTED onto WRF's staggered
    axis with a zero top interface -- all of which must survive both
    engines identically.
    """
    snow_layers = 3
    attrs = wrf_global_attrs(
        _grid(), _START, grid_id=2, parent_id=1, i_parent_start=12,
        j_parent_start=9, parent_grid_ratio=3, dt=10.0,
        hybrid_opt=2, etac=0.2)
    tsno = SCHEME_OUTPUT_FIELDS["tsnoxy"].netcdf_name
    snice = SCHEME_OUTPUT_FIELDS["snicexy"].netcdf_name
    snliq = SCHEME_OUTPUT_FIELDS["snliqxy"].netcdf_name
    zsnso = SCHEME_OUTPUT_FIELDS["zsnsoxy"].netcdf_name
    isnow = SCHEME_OUTPUT_FIELDS["isnowxy"].netcdf_name
    el_pbl = SCHEME_OUTPUT_FIELDS["el_pbl"].netcdf_name
    qke = SCHEME_OUTPUT_FIELDS["qke"].netcdf_name
    frames = []
    for seed in (31, 37):
        frame = _base_frame(seed)
        rng = np.random.default_rng(500 + seed)
        frame[tsno] = rng.uniform(
            250.0, 273.0, (snow_layers, _NY, _NX)).astype(np.float32)
        frame[snice] = rng.uniform(
            0.0, 20.0, (snow_layers, _NY, _NX)).astype(np.float32)
        frame[snliq] = rng.uniform(
            0.0, 2.0, (snow_layers, _NY, _NX)).astype(np.float32)
        frame[zsnso] = -np.cumsum(
            rng.uniform(0.02, 0.3, (snow_layers + 4, _NY, _NX)),
            axis=0).astype(np.float32)
        frame[isnow] = rng.integers(
            -snow_layers, 1, (_NY, _NX)).astype(np.int32)
        frame[el_pbl] = rng.uniform(
            0.0, 500.0, (_NZ, _NY, _NX)).astype(np.float32)
        frame[qke] = rng.uniform(
            0.0, 3.0, (_NZ, _NY, _NX)).astype(np.float32)
        frames.append(frame)
    stamps = ("1974-04-03_12:00:00", "1974-04-03_12:15:00")
    return dict(grid_id=2, dx=4000.0, attrs=attrs, stamps=stamps,
                frames=frames, soil_layers=4)


_CASES = {
    "gfs_small": _case_gfs_small,
    "twentycrv3_demo": _case_twentycrv3_demo,
    "nested_d02": _case_nested_d02,
}


def _write_case(directory, case, engine):
    """One case's history through one engine; same basename either way."""
    spec = _CASES[case]()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"wrfout_d{spec['grid_id']:02d}_1974-04-03_12_00_00"
    writer = WrfoutWriter(
        path, nx=_NX, ny=_NY, nz=_NZ, dx=spec["dx"], dy=spec["dx"],
        global_attrs=spec["attrs"], soil_layers=spec["soil_layers"],
        field_schema=spec["frames"][0], engine=engine)
    try:
        for stamp, frame in zip(spec["stamps"], spec["frames"]):
            writer.write_frame(stamp, frame)
    except BaseException:
        writer.abort()
        raise
    writer.close()
    return path


@pytest.fixture(scope="module", params=sorted(_CASES))
def dual_pair(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"dual-{request.param}")
    python_path = _write_case(root / "python", request.param, "python")
    rust_path = _write_case(root / "rust", request.param, "rust")
    return request.param, python_path, rust_path


def test_both_engines_write_the_same_tape(dual_pair):
    case, python_path, rust_path = dual_pair
    problems = semantic_diff(python_path, rust_path)
    assert problems == [], f"{case}: {problems}"


def test_the_two_containers_are_the_expected_two(dual_pair):
    """The one difference the flip makes is the envelope, and only that."""
    _case, python_path, rust_path = dual_pair
    with netCDF4.Dataset(python_path) as ds:
        assert ds.file_format == "NETCDF4_CLASSIC"
    with netCDF4.Dataset(rust_path) as ds:
        assert ds.file_format == "NETCDF3_64BIT_OFFSET"


def test_rw_wrfbatch_renders_are_byte_identical(dual_pair, tmp_path):
    """The artifact-level half: the product renderer cannot tell them apart."""
    from gpuwm import rustwx

    renderer = rustwx.find_renderer()
    if renderer is None:
        pytest.skip("rw_wrfbatch is not resolvable here")
    usable, evidence = rustwx.probe_renderer(renderer)
    if not usable:
        pytest.skip(f"rw_wrfbatch is not usable here: {evidence}")

    from tools.wrfout_dual_write import render_products

    case, python_path, rust_path = dual_pair
    products = {}
    for label, wrfout in (("python", python_path), ("rust", rust_path)):
        products[label] = render_products(
            wrfout, tmp_path / f"render-{case}-{label}",
            width=800, height=600)
    assert products["python"], f"{case}: the renderer produced no products"
    matched, problems = compare_renders(products["python"], products["rust"])
    assert problems == [], f"{case}: {problems}"
    assert matched == len(products["python"])
