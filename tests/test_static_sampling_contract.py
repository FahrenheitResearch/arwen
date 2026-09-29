"""Reject incompatible prepared statics before starting a moving tree."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_bridge_contract_marker_binds_loader_and_resolution_ladder():
    from gpuwm import bridges
    from gpuwm.static import rust_bridge
    assert rust_bridge.ABI_MARKER == b"gpuwm_static_sampling_portable_v1"
    assert bridges.BRIDGE_ABI_MARKERS["static_fields"] == rust_bridge.ABI_MARKER


def test_old_bridge_is_refused_even_when_abi_number_matches(monkeypatch):
    from gpuwm.static import rust_bridge
    class Function:
        def __call__(self):
            return rust_bridge.STATIC_ABI
    library = SimpleNamespace(gpuwm_static_abi_version=Function())
    monkeypatch.setattr(rust_bridge, "_LIBRARY", None)
    monkeypatch.setattr(rust_bridge, "resolve_static_bridge", lambda: Path("old-library"))
    monkeypatch.setattr(rust_bridge.ctypes, "CDLL", lambda path: library)
    monkeypatch.setattr(rust_bridge, "_bind_entry_points", lambda *a, **kw: None)
    with pytest.raises(rust_bridge.StaticBridgeError, match="predates this release"):
        rust_bridge.load()


@pytest.mark.parametrize("recorded", [None, "platform-libm-v0", "python-platform-sampling"])
def test_incompatible_receipt_refuses_at_native_preflight(tmp_path, monkeypatch, recorded):
    from gpuwm.native_wrf_contract import (verify_native_static_receipt,
        write_native_geometry_receipt, write_native_static_cache)
    from gpuwm.static import sampling_contract as contract
    from test_static_projection_portability import witness_grid
    monkeypatch.setattr(contract, "current_sampling_contract", lambda: contract.PORTABLE_SAMPLING_CONTRACT)
    grid = witness_grid(2000)
    cfg = SimpleNamespace(nx=30, ny=30, nz=24, dx=2000., dy=2000.)
    cache = tmp_path / "static.npz"
    receipt = tmp_path / "geometry.json"
    write_native_static_cache(cache, {"HGT_M": np.zeros((30, 30))})
    doc = write_native_geometry_receipt(receipt, grid, cfg, cache)
    assert doc["static_sampling_contract"] == contract.PORTABLE_SAMPLING_CONTRACT
    if recorded is None:
        doc.pop("static_sampling_contract")
    else:
        doc["static_sampling_contract"] = recorded
    receipt.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="first move.*overlap-statics"):
        verify_native_static_receipt(receipt, cache, grid, cfg, relocating=True)
    # A fixed domain never rebuilds a moved footprint and remains readable.
    verify_native_static_receipt(receipt, cache, grid, cfg)


def test_python_fallback_cannot_move_rust_prepared_statics(monkeypatch):
    from gpuwm.static import sampling_contract as contract
    monkeypatch.setattr(contract, "current_sampling_contract", lambda: "python-platform-sampling")
    with pytest.raises(ValueError, match="gpuwm go CONFIG"):
        contract.require_relocation_sampling_contract(contract.PORTABLE_SAMPLING_CONTRACT)


@pytest.mark.parametrize("recorded", [None, "platform-libm-v0", "wps-sampling-portable-v1"])
def test_same_process_contract_rejects_missing_or_different_preparation(monkeypatch, recorded):
    from gpuwm.static import sampling_contract as contract
    monkeypatch.setattr(contract, "current_sampling_contract", lambda: "python-platform-sampling")
    with pytest.raises(ValueError, match="first move.*overlap-statics"):
        contract.require_relocation_sampling_contract(recorded, same_process=True)


def test_matching_python_contract_from_disk_still_requires_portability(monkeypatch):
    from gpuwm.static import sampling_contract as contract
    monkeypatch.setattr(contract, "current_sampling_contract", lambda: "python-platform-sampling")
    with pytest.raises(ValueError, match="gpuwm go CONFIG"):
        contract.require_relocation_sampling_contract("python-platform-sampling")


@pytest.mark.parametrize("backend", ["explicit-fallback", "missing-bridge"])
def test_python_fallback_in_memory_run_builds_its_runner(tmp_path, monkeypatch, backend):
    from gpuwm.core.relocation_runner import RelocationRunner
    from gpuwm.experiment import RelocationConfig, ScheduledRelocationMove
    from gpuwm.runtime import RealRelocationChildPreparer, build_real_relocation_runner
    from gpuwm.static import rust_bridge, sampling_contract as contract
    from gpuwm.static.lambert import LambertGrid
    from test_nest_relocation_staging import _scaffold

    if backend == "explicit-fallback":
        monkeypatch.setenv("GPUWM_STATIC_PYTHON", "1")
    else:
        monkeypatch.delenv("GPUWM_STATIC_PYTHON", raising=False)
        monkeypatch.setattr(rust_bridge, "unavailable_reason", lambda: "bridge is absent")
    recorded = contract.current_sampling_contract()
    assert recorded == "python-platform-sampling"
    scaffold = _scaffold()
    child = scaffold.domains[1]
    exp = SimpleNamespace(
        relocation=RelocationConfig(enabled=True, grid_id=2,
            moves=(ScheduledRelocationMove(60., di_parent_cells=1),)),
        domains=scaffold.domains, vertical=object(), start_time=None)
    grid = LambertGrid(
        ref_lat=35., ref_lon=-97., truelat1=30., truelat2=60.,
        stand_lon=-97., dx=1000., dy=1000., e_we=13, e_sn=13)
    node = SimpleNamespace(cfg=child, grid=grid)
    model = SimpleNamespace(
        node=lambda grid_id: node, _input_catalog=object(),
        _prepared_by_grid_id={2: SimpleNamespace(static_sampling_contract=recorded)},
        schedule=SimpleNamespace(period_ticks=60, clock=SimpleNamespace(tick_den=1)))
    runner = build_real_relocation_runner(exp, None, model, tmp_path)
    assert isinstance(runner, RelocationRunner)
    assert isinstance(runner.on_child_built, RealRelocationChildPreparer)
    assert callable(runner.initializer)


def test_preflight_checks_the_sampling_contract_of_a_movers_descendant(tmp_path, monkeypatch):
    from gpuwm.experiment import load_experiment
    from gpuwm.native_wrf_contract import (
        verify_native_static_receipt, write_native_geometry_receipt)
    from gpuwm.static import corridor, sampling_contract as contract
    import test_prepared_domain_tree_forecast as tree

    write_config = tree._write_two_domain_config

    def three_domain_config(directory):
        path = write_config(directory)
        path.write_text(path.read_text(encoding="utf-8") + """
[[domain]]
grid_id = 3
parent_id = 2
i_parent_start = 21
j_parent_start = 21
parent_grid_ratio = 3
parent_time_step_ratio = 3
nx = 60
ny = 60
history_interval_s = 600.0
""", encoding="utf-8")
        return path

    monkeypatch.setattr(tree, "_write_two_domain_config", three_domain_config)
    prepared, receipt, config = tree._synthetic_prepared_tree(tmp_path, monkeypatch)
    tree._with_relocation(config, tree._RELOCATION_FOLLOW_TOML)
    exp = load_experiment(config)
    from gpuwm.native_wrf_contract import grids_from_projection_config
    grids = grids_from_projection_config(exp)
    monkeypatch.setattr(tree.runner, "grids_from_projection_config", lambda _exp: grids)
    monkeypatch.setattr(contract, "current_sampling_contract", lambda: contract.PORTABLE_SAMPLING_CONTRACT)
    preparation = json.loads(receipt.read_text(encoding="utf-8"))
    preparation["domain_count"] = 3
    artifact_receipt = preparation["artifact_receipt"]
    artifact_receipt["domain_count"] = 3
    artifact_receipt["boundary_inventory"]["nested_parent_forced"] = [2, 3]
    hierarchy = prepared / "hierarchy-artifacts"
    for domain, grid, domain_receipt in zip(exp.domains, grids, artifact_receipt["domains"]):
        bundle = hierarchy / "domains" / f"d{domain.grid_id:02d}"
        geometry_path = bundle / "geometry-receipt.json"
        geometry = write_native_geometry_receipt(
            bundle / "contract-geometry.json", grid, domain.run,
            bundle / "native-static.npz")
        if domain.grid_id == 3:
            geometry.pop("static_sampling_contract")
        geometry_path.write_text(json.dumps(geometry), encoding="utf-8")
        domain_receipt["artifacts"]["geometry_receipt"].update(
            sha256=tree._sha(geometry_path), geometry=geometry["geometry"])
        (bundle / "receipt.json").write_text(json.dumps(domain_receipt), encoding="utf-8")
    (hierarchy / "receipt.json").write_text(json.dumps(artifact_receipt), encoding="utf-8")
    receipt.write_text(json.dumps(preparation), encoding="utf-8")

    # Isolate receipt verification from the earlier source-availability gate.
    # Geometry, cache hashes, subtree traversal and contract checks stay real.
    monkeypatch.setattr(corridor, "config_declares_follow_source", lambda _exp: False)
    checked = []

    def verify(geometry_path, static_path, grid, cfg, *, relocating=False):
        checked.append((cfg.nx, relocating))
        return verify_native_static_receipt(
            geometry_path, static_path, grid, cfg, relocating=relocating)

    monkeypatch.setattr(tree.runner, "verify_native_static_receipt", verify)
    with pytest.raises(ValueError, match="first move.*overlap-statics"):
        tree._preflight(prepared, receipt, config)
    assert checked == [(100, False), (90, True), (60, True)]


def test_old_tree_is_refused_before_the_relocation_runner_is_built(monkeypatch, tmp_path):
    from gpuwm.runtime import build_real_relocation_runner
    from gpuwm.experiment import RelocationConfig, ScheduledRelocationMove
    from gpuwm.static import sampling_contract as contract
    monkeypatch.setattr(contract, "current_sampling_contract", lambda: contract.PORTABLE_SAMPLING_CONTRACT)
    exp = SimpleNamespace(relocation=RelocationConfig(enabled=True, grid_id=2,
        moves=(ScheduledRelocationMove(6., di_parent_cells=1),)))
    model = SimpleNamespace(_prepared_by_grid_id={2: SimpleNamespace()},
        node=lambda _: pytest.fail("incompatible preparation reached node initialization"))
    with pytest.raises(ValueError, match="first move.*overlap-statics"):
        build_real_relocation_runner(exp, None, model, tmp_path)


def test_terrain_survey_matches_the_default_portable_build(tmp_path):
    from gpuwm.static.build import build_terrain, build_static
    from test_static_projection_portability import geog_30arcsecond, witness_grid
    root = geog_30arcsecond(tmp_path)
    grid = witness_grid(1000)
    np.testing.assert_array_equal(build_terrain(grid, root), build_static(grid, root)["HGT_M"])


def test_terrain_survey_does_not_require_landuse_or_climatology(tmp_path):
    import shutil
    from gpuwm.static.build import build_terrain
    from test_static_projection_portability import geog_30arcsecond, witness_grid
    complete = geog_30arcsecond(tmp_path / "complete")
    terrain_only = tmp_path / "terrain-only"
    shutil.copytree(complete / "topo_gmted2010_30s", terrain_only / "topo_gmted2010_30s")
    grid = witness_grid(1000)
    np.testing.assert_array_equal(build_terrain(grid, terrain_only), build_terrain(grid, complete))
