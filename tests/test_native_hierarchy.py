"""Composition gates for parallel native hierarchy to stock-WRF export."""

from __future__ import annotations

from datetime import datetime
import json
from types import SimpleNamespace

import pytest

import gpuwm.native_hierarchy as hierarchy
from gpuwm.native_domain_artifacts import NativeHierarchyArtifactBuild
from gpuwm.wrf_direct import (
    HIERARCHY_EXPORT_SCHEMA,
    StockWrfExportUnsupported,
)


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


# ---------------------------------------------------------------------------
# F2: the boundary between preparing a forecast and exporting stock WRF
# ---------------------------------------------------------------------------
#
# This join used to call the exporter unconditionally, so
# `export_prepared_wrf`'s profile-free compatibility branch decided which
# domain trees could be PREPARED at all.  On the GFS route the only preparable
# root was YSU + classic-MM5 + Noah -- registry reachability, an accepted
# `expert-tuple-v1` acknowledgement and an explicitly named shipped profile
# were all overruled by a downstream file-format contract that predates them.
#
# The gate's content is unchanged (see tests/test_wrf_direct.py).  What is
# explicit now is the request, at both of its values, against both a stock and
# a non-stock root: four cells, four outcomes.


def _hierarchy_with(monkeypatch, tmp_path, *, exporter):
    exp, root, initial, boundaries = _inputs()
    artifacts = (SimpleNamespace(grid_id=1), SimpleNamespace(grid_id=2))
    artifact_build = NativeHierarchyArtifactBuild(
        artifacts=artifacts, manifest=tmp_path / "artifacts.json",
        receipt={"manifest": {"sha256": "d" * 64}})
    monkeypatch.setattr(
        hierarchy, "initialize_child_chain_parallel",
        lambda *_a, **_k: (object(),))
    monkeypatch.setattr(
        hierarchy, "write_native_hierarchy_artifacts",
        lambda *_a, **_k: artifact_build)
    monkeypatch.setattr(hierarchy, "export_prepared_wrf_hierarchy", exporter)
    return exp, root, initial, boundaries, artifact_build


def _run(hierarchy_inputs, tmp_path, **kwargs):
    exp, root, initial, boundaries, _build = hierarchy_inputs
    return hierarchy.initialize_and_export_native_hierarchy(
        exp=exp, root_node=root, catalog="catalog",
        artifact_output=tmp_path / "artifacts", wrf_output=tmp_path / "wrf",
        root_initial_result=initial, root_met="met", root_soil="soil",
        root_static_fields={}, root_boundaries=boundaries,
        bridge_manifest_sha256="a" * 64, source_manifest_sha256="b" * 64,
        namelist_sha256="c" * 64, forcing_hours=(0, 1),
        source_identity={"source": "fixture"}, **kwargs)


#: What the real exporter raises for the MYNN tree the probes drove.
_MYNN_REFUSAL = StockWrfExportUnsupported(
    "unsupported direct-export configuration: "
    "{'bl_pbl_physics': (5, 1), 'sf_sfclay_physics': (5, 91)}",
    unsupported={"bl_pbl_physics": (5, 1), "sf_sfclay_physics": (5, 91)})


def _stock_exporter(calls):
    def export(*_args, **_kwargs):
        calls.append("export")
        return {"schema": HIERARCHY_EXPORT_SCHEMA, "status": "READY",
                "files": {"wrfinput_d01": {}}}
    return export


def _non_stock_exporter(calls):
    def export(*_args, **_kwargs):
        calls.append("export")
        raise _MYNN_REFUSAL
    return export


def test_requested_export_with_stock_physics_publishes_the_wrf_file_set(
        tmp_path, monkeypatch):
    """Cell 1: unchanged from every release that could reach it."""

    calls: list[str] = []
    inputs = _hierarchy_with(
        monkeypatch, tmp_path, exporter=_stock_exporter(calls))

    result = _run(inputs, tmp_path, stock_wrf_export="optional")

    assert calls == ["export"]
    assert result.wrf_manifest["status"] == "READY"
    assert "files" in result.wrf_manifest


def test_requested_export_with_non_stock_physics_refuses_only_the_export(
        tmp_path, monkeypatch):
    """Cell 2: the headline.  The forecast is prepared; the export is not.

    The refusal keeps the export gate's own message, names the selector
    deltas as data, and does not reach the preparation: the verified
    hierarchy artifacts the GPU runner consumes are returned intact.
    """

    calls: list[str] = []
    inputs = _hierarchy_with(
        monkeypatch, tmp_path, exporter=_non_stock_exporter(calls))
    _exp, _root, _initial, _boundaries, artifact_build = inputs

    result = _run(inputs, tmp_path, stock_wrf_export="optional")

    assert calls == ["export"]
    assert result.artifacts is artifact_build
    assert result.wrf_manifest == {
        "schema": HIERARCHY_EXPORT_SCHEMA,
        "status": "REFUSED",
        "reason": (
            "unsupported direct-export configuration: "
            "{'bl_pbl_physics': (5, 1), 'sf_sfclay_physics': (5, 91)}"),
        "unsupported": {
            "bl_pbl_physics": [5, 1], "sf_sfclay_physics": [5, 91]},
    }
    # The refusal document survives a JSON round trip byte for byte: it
    # goes into a proof that a runner reads back.
    published = dict(result.wrf_manifest)
    assert json.loads(json.dumps(published)) == published


def test_unrequested_export_with_stock_physics_is_never_attempted(
        tmp_path, monkeypatch):
    """Cell 3: representable, and still not exported, because nobody asked."""

    calls: list[str] = []
    inputs = _hierarchy_with(
        monkeypatch, tmp_path, exporter=_stock_exporter(calls))

    result = _run(inputs, tmp_path, stock_wrf_export="off")

    assert calls == []
    assert result.wrf_manifest["status"] == "NOT_REQUESTED"
    assert not (tmp_path / "wrf").exists()


def test_unrequested_export_with_non_stock_physics_is_not_a_refusal(
        tmp_path, monkeypatch):
    """Cell 4: nothing was asked for, so nothing is refused."""

    calls: list[str] = []
    inputs = _hierarchy_with(
        monkeypatch, tmp_path, exporter=_non_stock_exporter(calls))

    result = _run(inputs, tmp_path, stock_wrf_export="off")

    assert calls == []
    assert result.wrf_manifest["status"] == "NOT_REQUESTED"


def test_a_caller_whose_product_is_the_export_still_fails_on_refusal(
        tmp_path, monkeypatch):
    """The default is unchanged: `required` propagates, as it always did."""

    calls: list[str] = []
    inputs = _hierarchy_with(
        monkeypatch, tmp_path, exporter=_non_stock_exporter(calls))

    with pytest.raises(StockWrfExportUnsupported, match="bl_pbl_physics"):
        _run(inputs, tmp_path)

    with pytest.raises(StockWrfExportUnsupported, match="bl_pbl_physics"):
        _run(inputs, tmp_path, stock_wrf_export="required")


def test_the_export_request_is_fail_closed_on_an_unknown_value(
        tmp_path, monkeypatch):
    calls: list[str] = []
    inputs = _hierarchy_with(
        monkeypatch, tmp_path, exporter=_stock_exporter(calls))

    with pytest.raises(ValueError, match="stock_wrf_export must be one of"):
        _run(inputs, tmp_path, stock_wrf_export="maybe")
    assert calls == []
