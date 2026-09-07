"""Caller-authored authorities enter the shared sealed-cache forecast path."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gpuwm import prepared_single_domain_forecast as runner, stage_cli
import test_prepared_single_domain_forecast as fixtures


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    fixtures._write_json(path, value)


def _seal(value, key):
    value.pop(key, None)
    value[key] = hashlib.sha256(runner._canonical(value).encode()).hexdigest()


def _generic_fixture(tmp_path, *, export="ready", decoder_roles=("gpuwm_mapped_engine",)):
    fixture = fixtures._prepared_fixture(
        tmp_path, "20crv3", twentycr_decoder_roles=decoder_roles)
    evidence = fixture.prepared / "source-evidence"
    mapping = _json(evidence / "mapping.json")
    mapping["name"] = "caller-authored-pressure-fields"
    _write(evidence / "mapping.json", mapping)
    composition = _json(evidence / "composition.json")
    terrain = composition["supplements"]["terrain_height"]
    proof = _json(fixture.proof)
    old = _json(fixture.source_manifest)
    provenance_path = evidence / "provenance-test.json"
    manifest = {
        "schema": runner._SOURCE_SCHEMA["mapped"],
        "mapping_sha256": fixtures._sha256(evidence / "mapping.json"),
        "composition_sha256": fixtures._sha256(evidence / "composition.json"),
        "primary_files": [{k: row[k] for k in ("path", "bytes", "sha256")}
                          for row in old["files"] if row["role"] == "pl"],
        "supplements": {terrain["data_role"]: [
            {k: row[k] for k in ("path", "bytes", "sha256")}
            for row in old["files"] if row["role"] == "sfc"]},
        "provenance": {terrain["provenance_role"]: {
            "path": str(provenance_path), "bytes": provenance_path.stat().st_size,
            "sha256": fixtures._sha256(provenance_path)}},
        "decoders": proof["execution_inputs"]["decoders"],
    }
    _write(fixture.source_manifest, manifest)
    receipt = proof["source_composition"]
    receipt["mapping"]["sha256"] = manifest["mapping_sha256"]
    receipt["input_manifest"] = {"path": str(fixture.source_manifest),
                                 "sha256": fixtures._sha256(fixture.source_manifest)}
    receipt["alignment"].pop("member")
    receipt["alignment"].pop("member_identity")
    _seal(receipt, "receipt_content_sha256")
    header_path = fixture.domain_bundle / "prepared-cache" / "header.json"
    header = _json(header_path)
    identity = header["identity"]
    identity.update(bridge_manifest_sha256=receipt["input_manifest"]["sha256"],
                    source_manifest_sha256=receipt["input_manifest"]["sha256"])
    identity["source_identity"].update(
        adapter=runner._SOURCE_ADAPTER["mapped"],
        mapping_sha256=manifest["mapping_sha256"],
        input_manifest_sha256=receipt["input_manifest"]["sha256"],
        composition_receipt_sha256=receipt["receipt_content_sha256"])
    header["metadata"]["user"]["composition_receipt_sha256"] = receipt["receipt_content_sha256"]
    basis = {k: header[k] for k in ("schema", "identity", "metadata", "arrays", "payload_bytes")}
    header["content_sha256"] = hashlib.sha256(runner._canonical(basis).encode()).hexdigest()
    _write(header_path, header)
    proof["prepared_cache"]["content_sha256"] = header["content_sha256"]
    proof["export"]["source"].update(
        prepared_header_sha256=fixtures._sha256(header_path),
        prepared_content_sha256=header["content_sha256"])
    if export != "ready":
        from gpuwm.wrf_direct import stock_wrf_export_not_requested, stock_wrf_export_refused, StockWrfExportUnsupported
        proof["stock_wrf_export"] = export
        if export == "off":
            proof["export"] = stock_wrf_export_not_requested(schema="gpuwm-native-direct-wrf-export-v3")
        else:
            proof["export"] = stock_wrf_export_refused(
                StockWrfExportUnsupported("synthetic unrepresentable companion", unsupported={"mp_physics": (6, 99)}),
                schema="gpuwm-native-direct-wrf-export-v3")
    _seal(proof, "proof_content_sha256")
    _write(fixture.proof, proof)
    fixture.source = "mapped"
    fixture.content_sha256 = header["content_sha256"]
    return fixture


@pytest.mark.parametrize("export", ["ready", "off", "optional"])
@pytest.mark.parametrize("roles", [(), ("caller_decoder",), ("gpuwm_mapped_engine",)])
def test_custom_mapping_enters_shared_preflight_and_reports_true_identity(tmp_path, monkeypatch, export, roles):
    fixture = _generic_fixture(tmp_path, export=export, decoder_roles=roles)
    fixtures._bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    inputs = fixtures._preflight_fixture(fixture)
    runner._verify_inputs_unchanged(inputs)
    bundle = stage_cli.resolve_bundle(fixture.prepared)
    command = stage_cli.sim_command(bundle, experiment_config=fixture.experiment,
                                    wps_namelist=fixture.wps, outdir=tmp_path / "run")
    assert command[command.index("--source") + 1] == "mapped"
    assert inputs.source == "mapped"
    assert inputs.source_member is None
    assert inputs.cache_identity["source_identity"]["adapter"] == runner._SOURCE_ADAPTER["mapped"]
    assert inputs.physics_receipt["source"] == "mapped"


@pytest.mark.parametrize("role", ["mapping", "composition", "provenance"])
def test_changed_copied_authority_fails_without_relabeling_model(tmp_path, monkeypatch, role):
    fixture = _generic_fixture(tmp_path)
    fixtures._bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    evidence = fixture.prepared / "source-evidence"
    path = evidence / ("provenance-test.json" if role == "provenance" else role + ".json")
    path.write_bytes(path.read_bytes() + b" ")
    assert stage_cli.resolve_bundle(fixture.prepared)["source"] == "mapped"
    with pytest.raises(ValueError, match="authorit"):
        fixtures._preflight_fixture(fixture)


@pytest.mark.parametrize("section", ["mapping", "primary", "decoder", "composition"])
def test_resealed_manifest_receipt_cannot_substitute_the_prepared_cache_inputs(tmp_path, monkeypatch, section):
    fixture = _generic_fixture(tmp_path)
    fixtures._bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    manifest = _json(fixture.source_manifest)
    proof = _json(fixture.proof)
    receipt = proof["source_composition"]
    if section in {"mapping", "composition"}:
        path = fixture.prepared / "source-evidence" / (section + ".json")
        path.write_bytes(path.read_bytes() + b" ")
        manifest[section + "_sha256"] = fixtures._sha256(path)
        receipt[section]["sha256"] = manifest[section + "_sha256"]
    elif section == "primary":
        manifest["primary_files"][0]["sha256"] = "1" * 64
    else:
        role = next(iter(manifest["decoders"]))
        manifest["decoders"][role]["sha256"] = "1" * 64
        receipt["decoders"][role]["sha256"] = "1" * 64
        proof["execution_inputs"]["decoders"][role]["sha256"] = "1" * 64
    _write(fixture.source_manifest, manifest)
    receipt["input_manifest"]["sha256"] = fixtures._sha256(fixture.source_manifest)
    _seal(receipt, "receipt_content_sha256")
    _seal(proof, "proof_content_sha256")
    _write(fixture.proof, proof)
    with pytest.raises(ValueError, match="source identity differs"):
        fixtures._preflight_fixture(fixture)


def test_authority_mutation_after_successful_preflight_is_caught(tmp_path, monkeypatch):
    fixture = _generic_fixture(tmp_path)
    fixtures._bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    inputs = fixtures._preflight_fixture(fixture)
    path = fixture.prepared / "source-evidence" / "mapping.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="inputs changed"):
        runner._verify_inputs_unchanged(inputs)


def test_generic_mapping_does_not_acquire_an_explicit_packaged_identity(tmp_path, monkeypatch):
    fixture = _generic_fixture(tmp_path)
    fixture.source = "20crv3-cf"
    fixtures._bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    with pytest.raises(ValueError, match="packaged.*authorities"):
        fixtures._preflight_fixture(fixture)


@pytest.mark.parametrize("field,value", [("stock_wrf_export", "required"), ("stock_wrf_export", []), ("export_schema", "invented")])
def test_missing_or_inconsistent_wrf_export_is_not_silently_accepted(tmp_path, monkeypatch, field, value):
    fixture = _generic_fixture(tmp_path, export="off")
    fixtures._bind_synthetic_preflight_geometry(monkeypatch, hierarchy=False)
    proof = _json(fixture.proof)
    if field == "export_schema":
        proof["export"]["schema"] = value
    else:
        proof[field] = value
    _seal(proof, "proof_content_sha256")
    _write(fixture.proof, proof)
    with pytest.raises(ValueError, match="export"):
        fixtures._preflight_fixture(fixture)


@pytest.mark.parametrize("profile", [None, "hrrr-prs-grib2-v1"])
def test_tree_source_identity_uses_authorities_not_a_fixed_model_name(tmp_path, profile):
    from gpuwm import prepared_domain_tree_forecast as tree
    from gpuwm.source_authorities import packaged_authorities
    root = tmp_path / "prepared"
    evidence = root / "source-evidence"
    evidence.mkdir(parents=True)
    if profile:
        authorities = packaged_authorities(profile)
        for name in ("mapping", "composition"):
            (evidence / (name + ".json")).write_bytes(authorities[name].read_bytes())
    else:
        for name in ("mapping", "composition"):
            _write(evidence / (name + ".json"), {"name": "caller"})
    _write(evidence / "input-manifest.json", {"schema": runner._SOURCE_SCHEMA["mapped"]})
    proof = {"schema": runner._HIERARCHY_PROOF_SCHEMA["mapped"],
             "status": "READY_NOT_YET_STOCK_WRF_GATED", "domain_count": 2}
    _write(root / "proof.json", proof)
    _, _, source = tree._load_hierarchy_document(root, fixtures._sha256(root / "proof.json"))
    assert source == ("hrrr-prs" if profile else "mapped")
    assert stage_cli.resolve_bundle(root)["source"] == source


@pytest.mark.parametrize("grid_id", [1, 2, 3])
def test_every_mapped_domain_binds_the_same_composition_and_its_own_grid(tmp_path, grid_id):
    fixture = _generic_fixture(tmp_path)
    proof, manifest = _json(fixture.proof), _json(fixture.source_manifest)
    paths, authority, _ = runner._validate_packaged_mapped_evidence(
        prepared_root=fixture.prepared, proof=proof, manifest=manifest,
        manifest_sha256=fixtures._sha256(fixture.source_manifest),
        experiment_config=None, wps_namelist=None, source="mapped")
    identity = _json(fixture.domain_bundle / "prepared-cache" / "header.json")["identity"]["source_identity"]
    identity.update(grid_id=grid_id, target_contract=authority["target_contract"],
                    nested_source_orography={}, hierarchy_implementation_sha256={})
    kwargs = dict(source="mapped", manifest_sha256=fixtures._sha256(fixture.source_manifest),
                  manifest_files={}, proof=proof, layout="mapped-hierarchy-d01-v1",
                  mapped_authority=authority, grid_id=grid_id)
    runner._validate_source_identity(identity=identity, **kwargs)
    identity["composition_receipt_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="source identity differs"):
        runner._validate_source_identity(identity=identity, **kwargs)


@pytest.mark.parametrize("mutation", [None, "policy", "overlay", "incomplete", "extra", "config-pin"])
def test_mapped_preparation_case_identity_binds_current_pinned_authority(tmp_path, mutation):
    from gpuwm.case_data import preparation_case_policy
    config = tmp_path / "case.toml"
    config.write_text('[experiment]\nname="mapped-case-identity"\n')
    authority = dict(mapping_sha256="a"*64, composition_sha256="b"*64,
                     receipt_content_sha256="c"*64, preprocessing={})
    identity = dict(adapter=runner._SOURCE_ADAPTER["mapped"],
                    mapping_sha256=authority["mapping_sha256"],
                    composition_sha256=authority["composition_sha256"],
                    composition_receipt_sha256=authority["receipt_content_sha256"],
                    input_manifest_sha256="d"*64, preprocessing={},
                    preparation_case_policy=preparation_case_policy(None),
                    water_temperature_overlay=None)
    pin = fixtures._sha256(config)
    if mutation == "policy":
        identity["preparation_case_policy"]["sfcp_to_sfcp"] = False
    elif mutation == "overlay":
        identity["water_temperature_overlay"] = {"sha256": "e"*64}
    elif mutation == "incomplete":
        identity.pop("water_temperature_overlay")
    elif mutation == "extra":
        identity["unrecognized"] = True
    elif mutation == "config-pin":
        pin = "f"*64
    kwargs = dict(source="mapped", identity=identity, manifest_sha256="d"*64,
                  manifest_files={}, proof={}, layout="mapped-direct-v1",
                  mapped_authority=authority, experiment_config=config,
                  experiment_config_sha256=pin)
    if mutation is None:
        assert runner._validate_source_identity(**kwargs) is identity
    else:
        with pytest.raises(ValueError):
            runner._validate_source_identity(**kwargs)


@pytest.mark.parametrize("source", ["20crv3", "gfs"])
def test_d01_cache_forecast_does_not_require_an_unrequested_wrf_companion(tmp_path, monkeypatch, source):
    fixture = fixtures._prepared_fixture(tmp_path, source, hierarchy=True)
    fixtures._bind_synthetic_preflight_geometry(monkeypatch, hierarchy=True)
    proof = _json(fixture.proof)
    from gpuwm.wrf_direct import stock_wrf_export_not_requested
    proof["stock_wrf_export"] = "off"
    proof["wrf_manifest"] = stock_wrf_export_not_requested()
    if source == "20crv3":
        _seal(proof, "proof_content_sha256")
    _write(fixture.proof, proof)
    (fixture.prepared / "wrf-native-input" / "manifest.json").unlink()
    inputs = fixtures._preflight_fixture(fixture)
    assert inputs.export_source_receipt["status"] == "NOT_REQUESTED"
    assert "wrf_manifest" not in inputs.authority_paths
    command = stage_cli.sim_command(
        stage_cli.resolve_bundle(fixture.prepared),
        experiment_config=fixture.experiment, wps_namelist=fixture.wps,
        outdir=tmp_path / "run", runner="single")
    assert command[2] == stage_cli.SINGLE_DOMAIN_RUNNER
    assert command[command.index("--prepared-content-sha256") + 1] == inputs.cache_reader.content_sha256
    runner._verify_inputs_unchanged(inputs)


def test_capabilities_describe_generic_source_without_a_profile_permission_list():
    capabilities = runner.runner_capabilities()
    assert "mapped" in capabilities["supported_sources"]
    assert capabilities["source_profiles"]["mapped"]["single_d01_gpu_execution"]
    assert capabilities["source_profiles"]["mapped"]["physics_profile_ids"] == []
    assert capabilities["physics_admission"]["verification_status"] == "reported-never-gating"


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_parent", "missing_cache"])
def test_d01_stage_does_not_invent_an_identity_for_a_damaged_root_receipt(tmp_path, mutation):
    fixture = fixtures._prepared_fixture(tmp_path, "gfs", hierarchy=True)
    proof = _json(fixture.proof)
    domains = proof["artifact_receipt"]["domains"]
    if mutation == "duplicate":
        domains.append(domains[0])
    elif mutation == "wrong_parent":
        domains[0]["parent_id"] = 1
    else:
        domains[0]["artifacts"].pop("prepared_cache")
    _write(fixture.proof, proof)
    with pytest.raises(stage_cli.StageRefusal, match="artifact receipt|prepared-cache identity"):
        stage_cli.single_domain_digests(stage_cli.resolve_bundle(fixture.prepared))
