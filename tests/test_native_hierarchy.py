"""Composition gates for parallel native hierarchy to stock-WRF export."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

import gpuwm.native_hierarchy as hierarchy
from gpuwm.native_domain_artifacts import NativeHierarchyArtifactBuild


def _inputs():
    boundaries = object()
    state = SimpleNamespace(lateral_boundaries=boundaries)
    root = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1), state=state, grid=object())
    initial = SimpleNamespace(state=state)
    exp = SimpleNamespace(
        domains=(SimpleNamespace(grid_id=1), SimpleNamespace(grid_id=2)),
        start_time=datetime(2026, 7, 20))
    return exp, root, initial, boundaries


def test_parallel_initializer_artifact_join_and_wrf_export_are_composed(
        tmp_path, monkeypatch):
    exp, root, initial, boundaries = _inputs()
    child_results = (object(),)
    events = []

    def initialize(*args, **kwargs):
        assert args == (exp, root, "catalog", "orography")
        assert kwargs == {
            "workers": 8, "preprocess_backend": "cpu",
            "cpu_bridge": "bridge", "scratch_arena": "arena",
            "dycore_state_workspace": "workspace",
            "state_backend": "preprocess",
            "sfcp_to_sfcp": False,
            "soil_layer_contract": "soil-contract"}
        events.append("initialize")
        return child_results

    artifacts = (SimpleNamespace(grid_id=1), SimpleNamespace(grid_id=2))
    artifact_build = NativeHierarchyArtifactBuild(
        artifacts=artifacts, manifest=tmp_path / "artifacts.json",
        receipt={"manifest": {"sha256": "d" * 64}})

    def write(output, **kwargs):
        assert output == tmp_path / "artifacts"
        assert kwargs["child_results"] is child_results
        assert kwargs["root_grid"] is root.grid
        assert kwargs["root_boundaries"] is boundaries
        assert kwargs["valid_time"] == exp.start_time
        events.append("artifacts")
        return artifact_build

    wrf_manifest = {"status": "READY", "files": {}}

    def export(actual_exp, actual_artifacts, output, **kwargs):
        assert actual_exp is exp
        assert actual_artifacts == artifacts
        assert output == tmp_path / "wrf"
        assert kwargs["boundary_interval_seconds"] == 3600
        assert kwargs["input_provenance"][
            "native_artifact_manifest_sha256"] == "d" * 64
        assert kwargs["input_provenance"]["source_manifest_sha256"] == \
            "e" * 64
        events.append("export")
        return wrf_manifest

    monkeypatch.setattr(hierarchy, "initialize_child_chain_parallel", initialize)
    monkeypatch.setattr(hierarchy, "write_native_hierarchy_artifacts", write)
    monkeypatch.setattr(hierarchy, "export_prepared_wrf_hierarchy", export)

    result = hierarchy.initialize_and_export_native_hierarchy(
        exp=exp, root_node=root, catalog="catalog",
        artifact_output=tmp_path / "artifacts", wrf_output=tmp_path / "wrf",
        root_initial_result=initial, root_met="met", root_soil="soil",
        root_static_fields={"HGT_M": 0}, root_boundaries=boundaries,
        bridge_manifest_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        namelist_sha256="c" * 64, forcing_hours=(0, 1),
        source_identity={"source": "fixture"}, source_orography="orography",
        workers=8, preprocess_backend="cpu", cpu_bridge="bridge",
        scratch_arena="arena", dycore_state_workspace="workspace",
        sfcp_to_sfcp=False, soil_layer_contract="soil-contract",
        input_provenance={"source_manifest_sha256": "e" * 64})

    assert result.artifacts is artifact_build
    assert result.wrf_manifest == wrf_manifest
    assert events == ["initialize", "artifacts", "export"]
    assert set(result.timings_seconds) == {
        "parallel_child_initialization", "verified_hierarchy_artifacts",
        "direct_stock_wrf_export", "total"}


def test_hierarchy_preflight_rejects_root_drift_before_expensive_work(
        tmp_path, monkeypatch):
    exp, root, initial, boundaries = _inputs()
    monkeypatch.setattr(
        hierarchy, "initialize_child_chain_parallel",
        lambda *_args, **_kwargs: pytest.fail("initialization must not launch"))
    wrong = SimpleNamespace(state=object())
    with pytest.raises(ValueError, match="same state"):
        hierarchy.initialize_and_export_native_hierarchy(
            exp=exp, root_node=root, catalog=object(),
            artifact_output=tmp_path / "artifacts",
            wrf_output=tmp_path / "wrf", root_initial_result=wrong,
            root_met=object(), root_soil=object(), root_static_fields={},
            root_boundaries=boundaries, bridge_manifest_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            namelist_sha256="c" * 64, forcing_hours=(0, 1),
            source_identity={})


def test_hierarchy_rejects_reserved_provenance_before_expensive_work(
        tmp_path, monkeypatch):
    exp, root, initial, boundaries = _inputs()
    monkeypatch.setattr(
        hierarchy, "initialize_child_chain_parallel",
        lambda *_args, **_kwargs: pytest.fail("initialization must not launch"))
    with pytest.raises(ValueError, match="reserved keys"):
        hierarchy.initialize_and_export_native_hierarchy(
            exp=exp, root_node=root, catalog=object(),
            artifact_output=tmp_path / "artifacts",
            wrf_output=tmp_path / "wrf", root_initial_result=initial,
            root_met=object(), root_soil=object(), root_static_fields={},
            root_boundaries=boundaries, bridge_manifest_sha256="a" * 64,
            source_manifest_sha256="b" * 64,
            namelist_sha256="c" * 64, forcing_hours=(0, 1),
            source_identity={}, input_provenance={
                "native_artifact_manifest": "editable"})
