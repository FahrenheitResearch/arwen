"""CPU integrity gates for atomic per-domain native WRF artifacts."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
import json
from types import SimpleNamespace

import numpy as np
import pytest

import gpuwm.native_domain_artifacts as artifacts_module
from gpuwm.config import RunConfig
from gpuwm.core.grid import BaseState, make_vertical_coord
from gpuwm.experiment import DomainConfig
from gpuwm.ingest.lateral_bc import (
    BoundaryInterval,
    FieldBoundary,
    LateralBoundaries,
    SideBoundary,
)
from gpuwm.native_domain_artifacts import (
    _atomic_staging_sibling,
    write_native_domain_artifacts,
    write_native_hierarchy_artifacts,
)
from gpuwm.io.restart import STATE_SETUP_ARRAYS, STATE_SETUP_SCALARS
from gpuwm.static.lambert import LambertGrid
from gpuwm.wrf_direct import load_domain_artifacts_manifest


def _domain(grid_id: int, parent_id: int) -> DomainConfig:
    root = parent_id == 0
    cfg = RunConfig(
        nx=2, ny=2, nz=2, dx=12_000.0, dy=12_000.0,
        ztop=16_000.0, dt=60.0 if root else 20.0,
        run_seconds=3_600.0, moist=True, terrain_opt=1, map_proj=1,
        specified=root, nested=not root, grid_id=grid_id,
        mp_physics=6, bl_pbl_physics=1, sf_sfclay_physics=91,
        sf_surface_physics=2, hybrid_opt=2, hypsometric_opt=2,
    )
    return DomainConfig(
        grid_id=grid_id, parent_id=parent_id,
        i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1 if root else 3,
        parent_time_step_ratio=1 if root else 3,
        history_interval_s=3_600.0, run=cfg,
        time_step=60 if root else None,
    )


def _inputs():
    coord = make_vertical_coord(
        2, hybrid_opt=2, etac=0.2, eta_levels=(1.0, 0.5, 0.0))
    base = BaseState(
        mub=np.full((2, 2), 90_000.0), p_top=10_000.0,
        pb=np.full((2, 2, 2), 50_000.0),
        alb=np.full((2, 2, 2), 0.8),
        thb=np.full((2, 2, 2), 290.0),
        phb=np.zeros((3, 2, 2)), terrain_z=np.zeros((2, 2)))
    state = SimpleNamespace(u=np.arange(12, dtype=np.float32).reshape(3, 2, 2))
    for index, name in enumerate(STATE_SETUP_ARRAYS):
        setattr(state, name, np.array([index], dtype=np.float32))
    scalar_values = {
        "mub": None, "p_top": 10_000.0,
        "cf1": 1.0, "cf2": 2.0, "cf3": 3.0,
        "cfn": 4.0, "cfn1": 5.0,
        "has_msf": True, "rotational": True,
    }
    assert set(scalar_values) == set(STATE_SETUP_SCALARS)
    for name, value in scalar_values.items():
        setattr(state, name, value)
    state.lateral_boundaries = None
    initial = SimpleNamespace(
        state=state, coord=coord, base=base,
        surface_pressure=np.full((2, 2), 99_000.0),
        surface_qv=np.full((2, 2), 0.01))
    plane = np.ones((2, 2), dtype=np.float32)
    met = SimpleNamespace(fields={
        "LANDSEA": plane,
        "SKINTEMP": 280.0 * plane,
        "T2": 279.0 * plane,
        "U10": np.ones((2, 3), dtype=np.float32),
        "V10": np.ones((3, 2), dtype=np.float32),
    })
    soil = SimpleNamespace(
        tsk=280.0 * plane,
        soil_temperature=np.full((4, 2, 2), 279.0, dtype=np.float32),
        soil_moisture=np.full((4, 2, 2), 0.2, dtype=np.float32),
        liquid_moisture=np.full((4, 2, 2), 0.2, dtype=np.float32),
        deep_soil_temperature=278.0 * plane,
        xice=np.zeros((2, 2), dtype=np.float32),
        xland=plane,
        landmask=plane,
        snow_water=np.zeros((2, 2), dtype=np.float32),
        snow_depth=np.zeros((2, 2), dtype=np.float32),
    )
    static = {
        "HGT_M": np.zeros((2, 2)),
        "LANDMASK": plane,
        "LU_INDEX": plane,
        "SCT_DOM": plane,
        "LANDUSEF": np.ones((2, 2, 2)),
        "SOILCTOP": np.ones((2, 2, 2)),
        "SOILCBOT": np.ones((2, 2, 2)),
        "GREENFRAC": np.ones((12, 2, 2)),
        "LAI12M": np.ones((12, 2, 2)),
        "ALBEDO12M": np.ones((12, 2, 2)),
        "SNOALB": plane,
        "TMN": 278.0 * plane,
    }
    side = SideBoundary(
        np.zeros((1, 2, 2), dtype=np.float64),
        np.zeros((1, 2, 2), dtype=np.float64))
    boundaries = LateralBoundaries((BoundaryInterval(
        0.0, 3_600.0,
        {name: FieldBoundary(side, side, side, side)
         for name in ("u", "v", "theta", "phi", "mu")}),), 5, 1, 4)
    grid = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.0, dx=12_000.0, dy=12_000.0, e_we=3, e_sn=3)
    return initial, met, soil, static, boundaries, grid


def test_atomic_staging_sibling_is_compact_and_target_independent(tmp_path):
    short = _atomic_staging_sibling(
        tmp_path / "prepared", nonce="0123456789")
    long = _atomic_staging_sibling(
        tmp_path / ("user-selected-output-" + "x" * 120),
        nonce="0123456789")

    assert short.name == long.name == ".d-0123456789"
    assert short.parent == long.parent == tmp_path
    assert len(short.name) == 13


@pytest.mark.parametrize("nonce", (
    "", "0" * 9, "0" * 11, "ABCDEF0123", "not-hex000",
))
def test_atomic_staging_sibling_rejects_unsafe_explicit_nonce(
        tmp_path, nonce):
    with pytest.raises(ValueError, match="10 lowercase hex"):
        _atomic_staging_sibling(tmp_path / "prepared", nonce=nonce)


def test_atomic_staging_sibling_fixes_exact_gfs_windows_path_budget(tmp_path):
    # Reproduce the failed proof's 160-character case root.  Win32 directory
    # creation reserves room for an 8.3 name, so the legacy 248-character
    # innermost directory raised WinError 206 despite being below 260.
    if len(str(tmp_path)) >= 160:
        pytest.skip("temporary root already exceeds the regression budget")
    case = tmp_path / ("x" * (160 - len(str(tmp_path)) - 1))
    assert len(str(case)) == 160
    legacy_outer = case / (
        "prepared.tmp-43108-3215f7017179439092dac2df80ebaeac")
    legacy_domain = (
        legacy_outer / ".tmp-432931ed" / "domains" / ".tmp-c97b40ee")
    assert len(str(legacy_domain)) == 248

    outer = _atomic_staging_sibling(
        case / "prepared", nonce="0123456789")
    hierarchy = _atomic_staging_sibling(
        outer / "hierarchy-artifacts", nonce="123456789a")
    domain = _atomic_staging_sibling(
        hierarchy / "domains" / "d01", nonce="23456789ab")
    prepared = domain / ".p-3456789abc" / "header.json"

    assert len(str(domain)) == 210
    assert len(str(prepared)) == 236
    prepared.parent.mkdir(parents=True)
    prepared.write_bytes(b"path-budget-pass")
    assert prepared.read_bytes() == b"path-budget-pass"


def test_domain_artifact_staging_collision_preserves_foreign_tree(
        tmp_path, monkeypatch):
    initial, met, soil, static, boundaries, grid = _inputs()
    initial.state.lateral_boundaries = boundaries
    output = tmp_path / "d01"
    staging = tmp_path / ".d-0123456789"
    staging.mkdir()
    sentinel = staging / "foreign.txt"
    sentinel.write_text("foreign", encoding="utf-8")
    monkeypatch.setattr(
        artifacts_module, "_atomic_staging_sibling", lambda _output: staging)

    with pytest.raises(FileExistsError):
        write_native_domain_artifacts(
            output, domain=_domain(1, 0), grid=grid,
            initial_result=initial, met=met, soil=soil,
            static_fields=static, boundaries=boundaries,
            bridge_manifest_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            namelist_sha256="c" * 64,
            forcing_hours=(0, 1), source_identity={"source": "fixture"},
            valid_time=datetime(2026, 7, 20),
        )

    assert sentinel.read_text(encoding="utf-8") == "foreign"
    assert not output.exists()


def test_domain_artifact_mid_write_failure_cleans_only_owned_staging(
        tmp_path, monkeypatch):
    initial, met, soil, static, boundaries, grid = _inputs()
    initial.state.lateral_boundaries = boundaries
    output = tmp_path / "d01"
    staging = tmp_path / ".d-0123456789"
    monkeypatch.setattr(
        artifacts_module, "_atomic_staging_sibling", lambda _output: staging)
    monkeypatch.setattr(
        artifacts_module, "write_native_static_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        write_native_domain_artifacts(
            output, domain=_domain(1, 0), grid=grid,
            initial_result=initial, met=met, soil=soil,
            static_fields=static, boundaries=boundaries,
            bridge_manifest_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            namelist_sha256="c" * 64,
            forcing_hours=(0, 1), source_identity={"source": "fixture"},
            valid_time=datetime(2026, 7, 20),
        )

    assert not staging.exists()
    assert not output.exists()


@pytest.mark.parametrize("grid_id,parent_id,boundary_mode", (
    (1, 0, "external-specified"),
    (2, 1, "nested-parent-forced"),
))
def test_domain_artifact_writer_publishes_root_or_child_atomically(
        tmp_path, grid_id, parent_id, boundary_mode):
    initial, met, soil, static, boundaries, grid = _inputs()
    initial.state.lateral_boundaries = boundaries if parent_id == 0 else None
    output = tmp_path / f"d{grid_id:02d}"
    build = write_native_domain_artifacts(
        output, domain=_domain(grid_id, parent_id), grid=grid,
        initial_result=initial, met=met, soil=soil,
        static_fields=static,
        boundaries=boundaries if parent_id == 0 else None,
        bridge_manifest_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        namelist_sha256="c" * 64,
        forcing_hours=(0, 1), source_identity={"source": "fixture"},
        valid_time=datetime(2026, 7, 20),
    )

    assert build.receipt["status"] == "READY"
    assert build.receipt["boundary_mode"] == boundary_mode
    assert build.artifacts.grid_id == grid_id
    assert build.artifacts.prepared_cache.is_dir()
    assert build.artifacts.static_cache.is_file()
    assert build.artifacts.geometry_receipt.is_file()
    assert (output / "receipt.json").is_file()
    assert not tuple(tmp_path.glob(".d-*"))


def test_domain_artifact_writer_rejects_wrong_boundary_mode_without_publish(
        tmp_path):
    initial, met, soil, static, boundaries, grid = _inputs()
    output = tmp_path / "d02"
    with pytest.raises(ValueError, match="must omit external LBCs"):
        write_native_domain_artifacts(
            output, domain=_domain(2, 1), grid=grid,
            initial_result=initial, met=met, soil=soil,
            static_fields=static, boundaries=boundaries,
            bridge_manifest_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            namelist_sha256="c" * 64,
            forcing_hours=(0, 1), source_identity={"source": "fixture"},
            valid_time=datetime(2026, 7, 20),
        )
    assert not output.exists()


def test_hierarchy_writer_joins_root_lbc_and_child_without_lbc_atomically(
        tmp_path):
    root_initial, root_met, root_soil, root_static, boundaries, grid = _inputs()
    child_initial, child_met, child_soil, child_static, _, child_grid = _inputs()
    root_initial.state.lateral_boundaries = boundaries
    child_initial.state.lateral_boundaries = None
    root_domain = _domain(1, 0)
    root_time = datetime(2026, 7, 20)
    child_domain = replace(
        _domain(2, 1), start_time=root_time + timedelta(minutes=5))
    exp = SimpleNamespace(domains=(root_domain, child_domain))
    child_result = SimpleNamespace(
        domain=child_domain,
        real=child_initial, grid=child_grid, horizontal=child_met,
        soil=child_soil, static_fields=child_static,
        preprocess_receipt={"backend": "cpu", "workers": 4},
        input_preparation_seconds=0.25)
    output = tmp_path / "hierarchy"

    build = write_native_hierarchy_artifacts(
        output, exp=exp, root_grid=grid,
        root_initial_result=root_initial, root_met=root_met,
        root_soil=root_soil, root_static_fields=root_static,
        root_boundaries=boundaries, child_results=(child_result,),
        bridge_manifest_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        namelist_sha256="c" * 64, forcing_hours=(0, 1),
        source_identity={"source": "fixture"},
        valid_time=root_time)

    assert build.receipt["status"] == "READY"
    assert build.receipt["boundary_inventory"] == {
        "external": [1], "nested_parent_forced": [2]}
    assert [artifact.grid_id for artifact in build.artifacts] == [1, 2]
    loaded = load_domain_artifacts_manifest(build.manifest)
    assert [artifact.grid_id for artifact in loaded] == [1, 2]
    assert all(artifact.prepared_cache.is_dir() for artifact in loaded)
    assert ".tmp-" not in json.dumps(dict(build.receipt))
    assert [domain["verification"]["path"]
            for domain in build.receipt["domains"]] == [
                "prepared-cache", "prepared-cache"]
    assert build.receipt["domains"][1]["valid_time"] == \
        "2026-07-20T00:05:00"
    assert not tuple(tmp_path.glob(".d-*"))


def test_hierarchy_writer_rejects_incomplete_child_without_partial_tree(
        tmp_path):
    initial, met, soil, static, boundaries, grid = _inputs()
    initial.state.lateral_boundaries = boundaries
    exp = SimpleNamespace(domains=(_domain(1, 0), _domain(2, 1)))
    output = tmp_path / "hierarchy"
    incomplete = SimpleNamespace(
        domain=exp.domains[1],
        real=None, static_fields=None, horizontal=None, soil=None)

    with pytest.raises(ValueError, match="not a complete real-data child"):
        write_native_hierarchy_artifacts(
            output, exp=exp, root_grid=grid,
            root_initial_result=initial, root_met=met, root_soil=soil,
            root_static_fields=static, root_boundaries=boundaries,
            child_results=(incomplete,),
            bridge_manifest_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            namelist_sha256="c" * 64, forcing_hours=(0, 1),
            source_identity={"source": "fixture"},
            valid_time=datetime(2026, 7, 20))
    assert not output.exists()


def test_hierarchy_writer_rejects_reordered_child_result_identity(tmp_path):
    initial, met, soil, static, boundaries, grid = _inputs()
    initial.state.lateral_boundaries = boundaries
    exp = SimpleNamespace(domains=(_domain(1, 0), _domain(2, 1)))
    output = tmp_path / "hierarchy"
    wrong_domain = SimpleNamespace(grid_id=3)
    misbound = SimpleNamespace(domain=wrong_domain)

    with pytest.raises(ValueError, match=r"identity.*d02"):
        write_native_hierarchy_artifacts(
            output, exp=exp, root_grid=grid,
            root_initial_result=initial, root_met=met, root_soil=soil,
            root_static_fields=static, root_boundaries=boundaries,
            child_results=(misbound,),
            bridge_manifest_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            namelist_sha256="c" * 64, forcing_hours=(0, 1),
            source_identity={"source": "fixture"},
            valid_time=datetime(2026, 7, 20))
    assert not output.exists()
