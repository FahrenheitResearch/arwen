"""Whole-tree reuse, checked through canonical artifact producers/readers."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from gpuwm import stage_reuse
from gpuwm.experiment import load_experiment
from gpuwm.ingest.prepared_cache import write_prepared_cache
from gpuwm.runplan import _prepare_stage
from gpuwm.wrf_direct import PreparedDomainArtifacts, write_domain_artifacts_manifest
from test_prepared_cache import _fixture
from test_prepared_domain_tree_forecast import _write_two_domain_config


ENGINE = {"identity_source": "git", "git_commit": "1" * 40,
          "git_tree": "2" * 40, "git_status_short": []}
SOURCE = "a" * 64


def _json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _domain(root, grid_id, *, engine=None):
    root.mkdir(parents=True)
    static = root / "native-static.npz"
    np.savez(static, terrain=np.full((2, 2), grid_id, dtype=np.float32))
    static_sha = hashlib.sha256(static.read_bytes()).hexdigest()
    receipt = root / "geometry-receipt.json"
    _json(receipt, {"schema": "gpuwm-native-static-direct-v1", "status": "PASS",
                   "cache": {"path": static.name, "bytes": static.stat().st_size,
                             "sha256": static_sha},
                   "geometry": {"grid_id": grid_id, "nx": 2, "ny": 2}})
    identity = {"source_manifest_sha256": SOURCE,
                "namelist_sha256": str(grid_id) * 64,
                "static_cache_sha256": static_sha,
                "domain_config": {"grid_id": grid_id}, "forcing_hours": [0, 1],
                "source_identity": {"adapter": "synthetic", **(engine or {})}}
    initial, met, boundaries = _fixture()
    write_prepared_cache(root / "prepared-cache", identity=identity,
                         initial_result=initial, met=met, boundaries=boundaries)
    return PreparedDomainArtifacts(grid_id, root / "prepared-cache", static, receipt)


def _tree(root, *, child_engine=None):
    records = [_domain(root / "domains" / f"d{grid_id:02}", grid_id,
                       engine=child_engine if grid_id == 2 else None)
               for grid_id in (1, 2)]
    write_domain_artifacts_manifest(root / "domain-artifacts.json", records)
    (root / "experiment.toml").write_text("[domains]\ncount = 2\n", encoding="utf-8")
    (root / "domains" / "d02" / "wrfinput_d02").write_bytes(b"export bytes")
    (root / "domains" / "d02" / "progress.json").write_text("child receipt")
    (root / "statics-corridor-d02.npz").write_bytes(b"corridor bytes")
    return root


@pytest.fixture
def bound_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_reuse, "engine_source_identity", lambda: dict(ENGINE))
    root = _tree(tmp_path / "hierarchy")
    stage_reuse.write_binding(root, arguments=["--cycle", "2026-08-25T18:00:00"],
                              stated={"source_manifest_sha256": SOURCE})
    return root


def _decide(root, arguments=None):
    return stage_reuse.decide(root, stated={"source_manifest_sha256": SOURCE},
                              arguments=arguments or ["--cycle", "2026-08-25T18:00:00"])


def test_complete_tree_reuses_every_verified_domain(bound_tree):
    result = _decide(bound_tree)
    assert result["decision"] == stage_reuse.REUSE
    assert result["domains"] == ["d01", "d02"]
    headers = [json.loads(p.read_text()) for p in bound_tree.rglob("prepared-cache/header.json")]
    assert result["verified_prepared_bytes"] == sum(h["payload_bytes"] for h in headers)
    assert result["verified_prepared_arrays"] == sum(len(h["arrays"]) for h in headers)
    assert result["verified_artifact_bytes"] > 0
    assert result["verification_seconds"] > 0


@pytest.mark.parametrize("name", ["experiment.toml", "domains/d02/wrfinput_d02",
                                 "domains/d02/progress.json", "statics-corridor-d02.npz"])
@pytest.mark.parametrize("mutation", ["change", "remove"])
def test_child_configuration_exports_and_optional_artifacts_remain_bound(
        bound_tree, name, mutation):
    path = bound_tree / name
    if mutation == "change":
        path.write_bytes(path.read_bytes() + b"changed")
    else:
        path.unlink()
    result = _decide(bound_tree)
    assert result["decision"] == stage_reuse.REBUILD
    assert any(name in d["field"] for d in result["differences"])


@pytest.mark.parametrize("name", ["domains/d02/native-static.npz",
                                 "domains/d02/geometry-receipt.json",
                                 "domains/d02/prepared-cache/header.json"])
@pytest.mark.parametrize("mutation", ["change", "remove"])
def test_canonical_child_artifact_corruption_refuses(bound_tree, name, mutation):
    path = bound_tree / name
    if mutation == "change":
        path.write_bytes(b"not the declared artifact")
    else:
        path.unlink()
    assert _decide(bound_tree)["decision"] == stage_reuse.REBUILD


def test_child_array_corruption_is_detected_without_header_change(bound_tree):
    path = next((bound_tree / "domains/d02/prepared-cache").glob("*.npy"))
    array = np.load(path, allow_pickle=False)
    array.flat[0] += 1
    np.save(path, array)
    result = _decide(bound_tree)
    assert result["decision"] == stage_reuse.REBUILD
    assert result["differences"][0]["field"] == "prepared_payload"


def test_unknown_legacy_tree_and_unlisted_cache_are_refused(bound_tree):
    manifest = bound_tree / "domain-artifacts.json"
    original = manifest.read_bytes()
    manifest.unlink()
    assert _decide(bound_tree)["decision"] == stage_reuse.REBUILD
    manifest.write_bytes(original)
    _domain(bound_tree / "unlisted", 3)
    assert _decide(bound_tree)["decision"] == stage_reuse.REBUILD


def test_changed_child_identity_and_engine_are_not_hidden_by_root(bound_tree):
    child = bound_tree / "domains/d02"
    shutil.rmtree(child)
    _domain(child, 2, engine={"git_commit": "9" * 40})
    # Even writing a new stage seal cannot turn a child built by other code into
    # agreement with the current engine. The domain's own code identity wins.
    stage_reuse.write_binding(bound_tree, arguments=["--cycle", "2026-08-25T18:00:00"],
                              stated={"source_manifest_sha256": SOURCE})
    result = _decide(bound_tree)
    assert result["decision"] == stage_reuse.REBUILD
    assert any(d["field"] == "d02.source_identity.git_commit" for d in result["differences"])


def test_root_progress_and_complete_relocation_preserve_reuse(bound_tree, tmp_path):
    (bound_tree / "progress.json").write_text("current progress", encoding="utf-8")
    assert _decide(bound_tree)["decision"] == stage_reuse.REUSE
    moved = tmp_path / "relocated" / bound_tree.name
    moved.parent.mkdir()
    shutil.move(bound_tree, moved)
    assert _decide(moved)["decision"] == stage_reuse.REUSE


@pytest.mark.parametrize("value", [None, [], 42, {"domains": 42}, {"files": [1]}])
def test_missing_or_malformed_whole_tree_seal_refuses(bound_tree, value):
    path = bound_tree / stage_reuse.BINDING_NAME
    binding = json.loads(path.read_text())
    binding["publication"] = value
    _json(path, binding)
    assert _decide(bound_tree)["decision"] == stage_reuse.REBUILD


@pytest.mark.parametrize("spelling", ["split", "equals"])
def test_root_preparation_content_is_bound_and_payload_verified(bound_tree, tmp_path, spelling):
    source = tmp_path / "root-preparation"
    artifact = _domain(source / "native", 1)
    arguments = (["--root-preparation", str(source)] if spelling == "split"
                 else [f"--root-preparation={source}"])
    stage_reuse.write_binding(bound_tree, arguments=arguments,
                              stated={"source_manifest_sha256": SOURCE})
    first = _decide(bound_tree, arguments)
    assert first["decision"] == stage_reuse.REUSE
    source_header = json.loads((artifact.prepared_cache / "header.json").read_text())
    assert first["verified_prepared_arrays"] == 3 * len(source_header["arrays"])
    array_path = next(artifact.prepared_cache.glob("*.npy"))
    array = np.load(array_path, allow_pickle=False)
    array.flat[0] += 1
    np.save(array_path, array)
    assert _decide(bound_tree, arguments)["decision"] == stage_reuse.REBUILD
    # A valid replacement at the same basename must also invalidate the seal.
    shutil.rmtree(source)
    _domain(source / "native", 1, engine={"adapter_revision": "changed"})
    result = _decide(bound_tree, arguments)
    assert result["decision"] == stage_reuse.REBUILD
    assert any(d["field"] == "prepared_inputs" for d in result["differences"])


def test_runplan_skips_an_unchanged_canonical_hierarchy(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_reuse, "engine_source_identity", lambda: dict(ENGINE))
    root = tmp_path / "hierarchy"
    calls = []

    def prepare():
        calls.append("prepare")
        _tree(root)

    first = _prepare_stage(root, arguments=[], stated={}, run=prepare)
    second = _prepare_stage(root, arguments=[], stated={}, run=prepare)
    assert [first["decision"], second["decision"]] == [stage_reuse.BUILD, stage_reuse.REUSE]
    assert calls == ["prepare"]


def test_valid_child_replacement_changes_publication(bound_tree):
    child = bound_tree / "domains/d02"
    shutil.rmtree(child)
    _domain(child, 2, engine={"adapter_revision": "new"})
    result = _decide(bound_tree)
    assert result["decision"] == stage_reuse.REBUILD
    assert any(d["field"] == "publication.domains.d02" for d in result["differences"])


def test_extra_cache_member_is_not_ignored(bound_tree):
    (bound_tree / "domains/d02/prepared-cache/extra.npy").write_bytes(b"undeclared")
    assert _decide(bound_tree)["decision"] == stage_reuse.REBUILD


def test_manifest_cannot_hide_or_alias_a_child(bound_tree):
    path = bound_tree / "domain-artifacts.json"
    payload = json.loads(path.read_text())
    payload["domains"][1]["prepared_cache"] = payload["domains"][0]["prepared_cache"]
    _json(path, payload)
    assert _decide(bound_tree)["decision"] == stage_reuse.REBUILD


def test_only_root_bookkeeping_is_exempt_from_artifact_identity(bound_tree):
    (bound_tree / "archive.superseded-example").write_bytes(b"must be accounted for")
    assert _decide(bound_tree)["decision"] == stage_reuse.REBUILD


def test_engine_advance_does_not_rehash_payloads(bound_tree, monkeypatch):
    from gpuwm.ingest.prepared_cache import PreparedCacheReader
    monkeypatch.setattr(stage_reuse, "engine_source_identity",
                        lambda: {**ENGINE, "git_commit": "3" * 40})

    def unexpected_verify(_self):
        pytest.fail("known engine drift should rebuild without reading all payloads")

    monkeypatch.setattr(PreparedCacheReader, "verify_all", unexpected_verify)
    assert _decide(bound_tree)["decision"] == stage_reuse.REBUILD


@pytest.mark.parametrize("spelling", ["split", "equals"])
def test_runplan_reuses_preparation_across_forecast_controls(
        tmp_path, monkeypatch, spelling):
    monkeypatch.setattr(stage_reuse, "engine_source_identity", lambda: dict(ENGINE))
    config = _write_two_domain_config(tmp_path)
    config.write_text(config.read_text().replace(
        "mp_physics = 6", "mp_physics = 6\ninflow_perturbation_seed = 17"))
    other = tmp_path / "different-seed.toml"
    other.write_text(config.read_text().replace(
        "inflow_perturbation_seed = 17", "inflow_perturbation_seed = 29")
        .replace("[shared]", "[shared]\nstep_to_output_time = true")
        + "\ntarget_cfl = 1.1\n")
    root = tmp_path / "hierarchy"
    calls = []

    def prepare():
        calls.append("prepare")
        _tree(root)

    def arguments(path):
        return (["--experiment-config", str(path)] if spelling == "split"
                else [f"--experiment-config={path}"])

    first = _prepare_stage(root, arguments=arguments(config), stated={}, run=prepare)
    second = _prepare_stage(root, arguments=arguments(other), stated={}, run=prepare)
    assert [first["decision"], second["decision"]] == [stage_reuse.BUILD, stage_reuse.REUSE]
    assert calls == ["prepare"]
    assert second["domains"] == ["d01", "d02"]
    assert second["verified_prepared_arrays"] > 0


@pytest.mark.parametrize("mutation", ["vertical", "physics", "geometry", "initial-bubble"])
def test_scientific_preparation_changes_remain_bound(bound_tree, tmp_path, mutation):
    config = _write_two_domain_config(tmp_path)
    arguments = ["--experiment-config", str(config)]
    stage_reuse.write_binding(bound_tree, arguments=arguments,
                              stated={"source_manifest_sha256": SOURCE})
    original = config.read_text()
    changes = {
        "vertical": original.replace("p_top = 10000.0", "p_top = 12000.0"),
        "physics": original.replace("mp_physics = 6", "mp_physics = 8"),
        "geometry": original.replace("nx = 90", "nx = 87"),
        "initial-bubble": original + """
[[perturbation.bubbles]]
center_lat = 40.0
center_lon = -83.0
center_height_m = 500.0
radius_km = 10.0
depth_m = 1000.0
amplitude_k = 1.0
""",
    }
    config.write_text(changes[mutation])
    load_experiment(config)
    result = _decide(bound_tree, arguments)
    assert result["decision"] == stage_reuse.REBUILD
    assert any(d["field"] == "arguments --experiment-config"
               for d in result["differences"])


def test_control_projection_keeps_relative_input_directory_bound(tmp_path):
    config = _write_two_domain_config(tmp_path)
    other = tmp_path / "elsewhere" / config.name
    other.parent.mkdir()
    other.write_bytes(config.read_bytes())
    assert stage_reuse.argument_binding(["--experiment-config", str(config)]) != (
        stage_reuse.argument_binding(["--experiment-config", str(other)]))


def test_invalid_control_config_keeps_conservative_byte_identity(tmp_path):
    config = _write_two_domain_config(tmp_path)
    config.write_text(config.read_text() + "\ninflow_perturbation_seed = -1\n")
    binding = stage_reuse.argument_binding(["--experiment-config", str(config)])
    assert binding["--experiment-config"] == {
        "name": config.name, "sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
