"""A surface donor retains the primary terrain alignment and every donor pin."""
from copy import deepcopy
import hashlib

import pytest

from gpuwm import prepared_single_domain_forecast as runner
from test_generic_mapped_forecast import _generic_fixture, _json, _seal, _write
from test_prepared_single_domain_forecast import _sha256


def _vegetation_fixture(tmp_path):
    fixture = _generic_fixture(tmp_path, export="off")
    evidence = fixture.prepared / "source-evidence"
    composition = _json(evidence / "composition.json")
    manifest = _json(fixture.source_manifest)
    proof = _json(fixture.proof)
    provenance = evidence / "provenance-vegetation.json"
    _write(provenance, {"schema": "test-surface-provenance", "field": "vegetation_fraction"})
    mapping_digest = hashlib.sha256(b"surface-vegetation-mapping").hexdigest()
    donor = {
        "source_id": "surface-fields", "mapping_role": "vegetation_mapping",
        "mapping_sha256": mapping_digest, "data_role": "vegetation",
        "provenance_role": "vegetation_provenance", "fields": ["vegetation_fraction"],
        "grid_alignment": "exact_coordinate_subset", "time_alignment": "valid_time_exact",
    }
    composition["field_sources"] = {"vegetation_surface": donor}
    _write(evidence / "composition.json", composition)
    manifest["composition_sha256"] = _sha256(evidence / "composition.json")
    manifest["supplements"]["vegetation"] = deepcopy(manifest["primary_files"])
    manifest["provenance"]["vegetation_provenance"] = {
        "path": str(provenance), "bytes": provenance.stat().st_size, "sha256": _sha256(provenance)}
    _write(fixture.source_manifest, manifest)
    receipt = proof["source_composition"]
    receipt["composition"]["sha256"] = manifest["composition_sha256"]
    receipt["input_manifest"]["sha256"] = _sha256(fixture.source_manifest)
    receipt["contributing_sources"] = [{
        "binding": "vegetation_surface", "source_id": donor["source_id"],
        "mapping": {"path": "vegetation.mapping.json", "sha256": mapping_digest},
        "alignment": {"status": "PASS"},
        "data": deepcopy(manifest["supplements"]["vegetation"]),
        "provenance": deepcopy(manifest["provenance"]["vegetation_provenance"]),
    }]
    _seal(receipt, "receipt_content_sha256")
    _seal(proof, "proof_content_sha256")
    _write(fixture.proof, proof)
    return fixture, proof, manifest


def _validate(fixture, proof, manifest):
    return runner._validate_packaged_mapped_evidence(
        prepared_root=fixture.prepared, proof=proof, manifest=manifest,
        manifest_sha256=_sha256(fixture.source_manifest),
        experiment_config=None, wps_namelist=None, source="mapped")


def test_nonterrain_donor_keeps_original_terrain_alignment_and_its_own_provenance(tmp_path):
    fixture, proof, manifest = _vegetation_fixture(tmp_path)
    paths, authority, member = _validate(fixture, proof, manifest)
    assert proof["source_composition"]["alignment"]["schema"] == "gpuwm-mapped-exact-subset-binding-v1"
    assert "mapped_provenance:vegetation_provenance" in paths
    assert authority["composition_sha256"] == manifest["composition_sha256"]
    assert member is None


@pytest.mark.parametrize("mutation, reason", [
    ("mapping", "mapping hash"), ("data", "supplement inventory"),
    ("provenance", "provenance differs"), ("terrain", "alignment receipt differs"),
    ("terrain_subset", "terrain subset sha256"),
])
def test_nonterrain_donor_does_not_bypass_terrain_or_contributing_authorities(tmp_path, mutation, reason):
    fixture, proof, manifest = _vegetation_fixture(tmp_path)
    receipt = proof["source_composition"]
    donor = receipt["contributing_sources"][0]
    if mutation == "mapping":
        donor["mapping"]["sha256"] = "1" * 64
    elif mutation == "data":
        donor["data"][0]["sha256"] = "1" * 64
    elif mutation == "provenance":
        donor["provenance"]["sha256"] = "1" * 64
    elif mutation == "terrain":
        receipt["alignment"]["matched_primary_valid_times"] = []
    else:
        receipt["alignment"]["terrain_subset_sha256"] = "invalid"
    _seal(receipt, "receipt_content_sha256")
    _seal(proof, "proof_content_sha256")
    _write(fixture.proof, proof)
    with pytest.raises(ValueError, match=reason):
        _validate(fixture, proof, manifest)
