"""Real-CUDA structural gate for a deliberately unlike vertical count."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
import hashlib
import json

import netCDF4
import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig, validate_run_config
from gpuwm.core.grid import make_vertical_coord
from gpuwm.gfs_direct import _geometry_contract
from gpuwm.ingest.horiz import HorizontalSnapshot
from gpuwm.ingest.lateral_bc import (
    attach_lateral_boundaries,
    build_state_lateral_boundaries,
)
from gpuwm.ingest.prepared_cache import write_prepared_cache
from gpuwm.ingest.real import initialize_real
from gpuwm.static.lambert import LambertGrid
from gpuwm.wrf_direct import export_prepared_wrf


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(cp, ny: int, nx: int) -> HorizontalSnapshot:
    levels = np.array(
        [100.0, 150.0, 200.0, 300.0, 500.0, 700.0, 850.0, 1000.0])
    pressure = levels[:, None, None] * 100.0
    shape = (levels.size, ny, nx)
    temperature = np.broadcast_to(
        215.0 + 72.0 * (pressure / 100000.0) ** 0.20, shape).copy()
    height = np.broadcast_to(
        -7800.0 * np.log(pressure / 100000.0), shape).copy()
    rh = np.broadcast_to(
        35.0 + 50.0 * (pressure / 100000.0), shape).copy()
    u = np.broadcast_to(
        12.0 + 0.8 * np.log(100000.0 / pressure),
        (levels.size, ny, nx + 1)).copy()
    v = np.broadcast_to(
        -3.0 + 0.4 * np.log(100000.0 / pressure),
        (levels.size, ny + 1, nx)).copy()
    fields = {
        "TT": temperature,
        "GHT": height,
        "RH": rh,
        "UU": u,
        "VV": v,
        "PSFC": np.full((ny, nx), 96_000.0),
        "T2": np.full((ny, nx), 286.0),
        "D2": np.full((ny, nx), 279.0),
        "U10": np.full((ny, nx + 1), 11.0),
        "V10": np.full((ny + 1, nx), -2.0),
    }
    return HorizontalSnapshot(
        valid_time=datetime(2026, 7, 20),
        levels_hpa=levels,
        fields={name: cp.asarray(value, cp.float32)
                for name, value in fields.items()},
    )


def _static(grid, ny: int, nx: int) -> dict[str, np.ndarray]:
    plane = np.ones((ny, nx), dtype=np.float64)
    mapfac_m = grid.mapfac_m()
    mapfac_u = grid.mapfac_u()
    mapfac_v = grid.mapfac_v()
    coriolis, curvature = grid.coriolis_m()
    sina, cosa = grid.rotation_m()
    return {
        "HGT_M": np.zeros((ny, nx), dtype=np.float64),
        "LU_INDEX": np.ones((ny, nx), dtype=np.int32),
        "SCT_DOM": np.full((ny, nx), 6, dtype=np.int32),
        "LANDMASK": plane,
        "GREENFRAC": np.full((12, ny, nx), 0.5),
        "ALBEDO12M": np.full((12, ny, nx), 20.0),
        "LAI12M": np.full((12, ny, nx), 2.0),
        "SNOALB": np.full((ny, nx), 60.0),
        "LANDUSEF": np.broadcast_to(
            plane, (21, ny, nx)).copy() / 21.0,
        "SOILCTOP": np.broadcast_to(
            plane, (16, ny, nx)).copy() / 16.0,
        "SOILCBOT": np.broadcast_to(
            plane, (16, ny, nx)).copy() / 16.0,
        "MAPFAC_M": mapfac_m,
        "MAPFAC_U": mapfac_u,
        "MAPFAC_V": mapfac_v,
        "F": coriolis,
        "E": curvature,
        "SINALPHA": sina,
        "COSALPHA": cosa,
    }


@requires_gpu
@pytest.mark.gpu
def test_native_80_level_initialize_cache_and_stock_wrf_export_structure(
        tmp_path):
    import cupy as cp

    nz, ny, nx = 80, 13, 13
    p_top = 10_000.0
    eta = np.linspace(1.0, 0.0, nz + 1)
    cfg = validate_run_config(RunConfig(
        nx=nx, ny=ny, nz=nz,
        dx=3000.0, dy=3000.0, ztop=20_000.0,
        dt=15.0, run_seconds=3600.0,
        hybrid_opt=2, etac=0.37, hypsometric_opt=2,
        moist=True, mp_physics=6,
        terrain_opt=1, map_proj=1,
        specified=True, nested=False,
        spec_bdy_width=5, spec_zone=1, relax_zone=4,
        sf_sfclay_physics=91, sf_surface_physics=2,
        bl_pbl_physics=1,
    ))
    grid = LambertGrid(
        39.0, -84.0, 30.0, 60.0, -84.0,
        cfg.dx, cfg.dy, nx + 1, ny + 1)
    static = _static(grid, ny, nx)
    snapshot = _snapshot(cp, ny, nx)
    coord = make_vertical_coord(
        nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac, eta_levels=eta)
    result = initialize_real(
        snapshot, cfg, coord, static["HGT_M"],
        source_orography=static["HGT_M"],
        p_top=p_top, sfcp_to_sfcp=True)
    result.state.set_map_coriolis(
        static["MAPFAC_M"], static["MAPFAC_U"], static["MAPFAC_V"],
        static["F"], static["E"],
        sina=static["SINALPHA"], cosa=static["COSALPHA"])
    times = (snapshot.valid_time, snapshot.valid_time + timedelta(hours=1))
    boundaries = build_state_lateral_boundaries(
        [result.state, result.state], times,
        spec_bdy_width=5, spec_zone=1, relax_zone=4)
    attach_lateral_boundaries(result.state, boundaries)

    static_path = tmp_path / "native-static.npz"
    np.savez(static_path, **static)
    geometry = _geometry_contract(grid, cfg)
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(json.dumps({
        "schema": "gpuwm-native-static-direct-v1",
        "status": "PASS",
        "cache": {
            "path": static_path.name,
            "bytes": static_path.stat().st_size,
            "sha256": _sha256(static_path),
        },
        "geometry": geometry,
    }), encoding="utf-8")

    host_plane = np.ones((ny, nx), dtype=np.float32)
    met = type("Met", (), {"fields": {
        "LANDSEA": host_plane,
        "SKINTEMP": 286.0 * host_plane,
        "T2": 286.0 * host_plane,
        "U10": np.full((ny, nx + 1), 11.0, dtype=np.float32),
        "V10": np.full((ny + 1, nx), -2.0, dtype=np.float32),
    }})()
    surface = {
        "TSK": 286.0 * host_plane,
        "TSLB": np.full((4, ny, nx), 285.0, dtype=np.float32),
        "SMOIS": np.full((4, ny, nx), 0.25, dtype=np.float32),
        "SH2O": np.full((4, ny, nx), 0.25, dtype=np.float32),
        "TMN": 284.0 * host_plane,
        "SEAICE": np.zeros((ny, nx), dtype=np.float32),
        "XLAND": host_plane,
        "LANDMASK": host_plane,
        "SNOW": np.zeros((ny, nx), dtype=np.float32),
        "SNOWH": np.zeros((ny, nx), dtype=np.float32),
    }
    identity = {
        "domain_config": {"run": dataclasses.asdict(cfg)},
        "forcing_hours": [0, 1],
        "static_cache_sha256": _sha256(static_path),
    }
    cache_path = tmp_path / "prepared-cache"
    write_prepared_cache(
        cache_path,
        identity=identity,
        initial_result=result,
        met=met,
        surface=surface,
        boundaries=boundaries,
        metadata={"initial_valid_time": snapshot.valid_time.isoformat()},
    )
    output = tmp_path / "wrf-native-input"
    export_prepared_wrf(
        cache_path, static_path, geometry_path, output,
        valid_time=snapshot.valid_time,
        boundary_interval_seconds=3600)

    with netCDF4.Dataset(output / "wrfinput_d01") as dataset:
        assert len(dataset.dimensions["bottom_top"]) == nz
        assert len(dataset.dimensions["bottom_top_stag"]) == nz + 1
        np.testing.assert_array_equal(
            dataset.variables["ZNW"][0], eta.astype(np.float32))
        assert float(dataset.variables["P_TOP"][0]) == p_top
        assert dataset.variables["T"].shape == (1, nz, ny, nx)
    with netCDF4.Dataset(output / "wrfbdy_d01") as dataset:
        assert len(dataset.dimensions["bottom_top"]) == nz
        assert len(dataset.dimensions["bottom_top_stag"]) == nz + 1
